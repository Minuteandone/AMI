"""Count-aware AMI Name -> Elements model.

This does not take user-specified elements or a count. It predicts the basic-element
count (0-5), special-element count (0-2), and individual labels independently,
then selects the top-scoring labels from each category.

Only the 263 official names and their existing labeled elements are used for
training. Associated and in-game elements are not the same concept; see README.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import json, pathlib, argparse, random, statistics
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from train_elements import (ROOT, RECORDS, LABELS, CORE, CHAR, C2I, L, X, Y,
                            IDX, NAMES, FOLDS, split, posweight)

BASIC=np.asarray(IDX)
SPECIAL=np.asarray([i for i in range(len(LABELS)) if i not in IDX])
YC=Y[:, BASIC].sum(1).long()
YS=Y[:, SPECIAL].sum(1).long()
COUNT_CORE=int(YC.max())+1
COUNT_SPECIAL=int(YS.max())+1

class CountAwareCNN(nn.Module):
    def __init__(self,dropout=.28):
        super().__init__()
        self.emb=nn.Embedding(len(CHAR),24,padding_idx=0)
        self.convs=nn.ModuleList([nn.Conv1d(24,40,k,padding='same') for k in (2,3,4)])
        self.project=nn.Sequential(nn.Linear(240,96),nn.GELU(),nn.Dropout(dropout),nn.Linear(96,64),nn.GELU())
        self.label_head=nn.Linear(64,len(LABELS))
        self.core_count=nn.Linear(64,COUNT_CORE)
        self.special_count=nn.Linear(64,COUNT_SPECIAL)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        e=self.emb(x).transpose(1,2)
        mask=(x!=0)
        z=[]
        for conv in self.convs:
            a=F.gelu(conv(e))
            z.extend([a.masked_fill(~mask[:,None],-1e4).amax(-1),
                      (a*mask[:,None]).sum(-1)/mask.sum(-1)[:,None].clamp_min(1)])
        h=self.project(torch.cat(z,dim=-1))
        return self.label_head(self.dropout(h)), self.core_count(h), self.special_count(h)

def weights_for_count(y,train_ids,num):
    h=torch.bincount(y[train_ids],minlength=num).float()
    return (h.sum()/h.clamp_min(1)).sqrt().clamp(1,3)

def train(train_ids, steps=260, seed=1, dev_ids=None):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    g=torch.Generator().manual_seed(seed+321)
    model=CountAwareCNN()
    opt=torch.optim.AdamW(model.parameters(),lr=.0016,weight_decay=.03)
    pos=posweight(train_ids)
    wc=weights_for_count(YC,train_ids,COUNT_CORE)
    ws=weights_for_count(YS,train_ids,COUNT_SPECIAL)
    ids=torch.tensor(train_ids)
    best=(-float('inf'),None,None)
    history=[]
    for step in range(1,steps+1):
        model.train()
        b=ids[torch.randint(len(ids),(min(64,len(ids)),),generator=g)]
        p,c,s=model(X[b])
        loss=F.binary_cross_entropy_with_logits(p,Y[b],pos_weight=pos)+ \
             .45*F.cross_entropy(c,YC[b],weight=wc)+ \
             .35*F.cross_entropy(s,YS[b],weight=ws)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.0);opt.step()
        if dev_ids is not None and step%20==0:
            pred=forward_names(model,X[dev_ids]); m=evaluate(dev_ids,pred)
            # Prefer selecting correct count of elements plus some label ranking ability.
            metric=.30*m['core_count_accuracy']+.20*m['special_count_accuracy'] + .25*m['exact_two_counts_accuracy']+.25*m['core_micro_AP']
            history.append({'step':step,'selection_metric':round(metric,5),**m})
            if metric>best[0]+1e-6:
                best=(metric,step,{k:v.detach().clone() for k,v in model.state_dict().items()})
    if dev_ids is not None:
        model.load_state_dict(best[2]);steps=best[1]
    model.eval()
    return model,steps,history

@torch.no_grad()
def forward_names(model, x):
    model.eval()
    z,c,s=model(x)
    return {'label_probs':torch.sigmoid(z).numpy(),
            'core_count_probs':F.softmax(c,dim=-1).numpy(),
            'special_count_probs':F.softmax(s,dim=-1).numpy()}

def choose(pred):
    """Return binary predictions with inferred (not supplied) element counts."""
    probs=pred['label_probs']
    core=np.argmax(pred['core_count_probs'],axis=1)
    special=np.argmax(pred['special_count_probs'],axis=1)
    selected=np.zeros_like(probs,dtype=np.int8)
    for i in range(len(probs)):
        if core[i]: selected[i,BASIC[np.argsort(-probs[i,BASIC])[:int(core[i])]]]=1
        if special[i]: selected[i,SPECIAL[np.argsort(-probs[i,SPECIAL])[:int(special[i])]]]=1
    return selected,core,special

def evaluate(indices,pred):
    sel,c,s=choose(pred)
    ys=Y[indices].numpy().astype(int)
    true_c=YC[indices].numpy()
    true_s=YS[indices].numpy()
    threshold=(pred['label_probs']>=.25).astype(int)
    return {'n':len(indices),
      'core_count_accuracy':float(np.mean(c==true_c)),
      'special_count_accuracy':float(np.mean(s==true_s)),
      'exact_two_counts_accuracy':float(np.mean((c==true_c)&(s==true_s))),
      'total_count_accuracy':float(np.mean(c+s==true_c+true_s)),
      'core_micro_AP':float(average_precision_score(ys[:,BASIC].ravel(),pred['label_probs'][:,BASIC].ravel())),
      'all_micro_F1_count_aware':float(f1_score(ys.ravel(),sel.ravel(),zero_division=0)),
      'all_micro_F1_threshold_025':float(f1_score(ys.ravel(),threshold.ravel(),zero_division=0)),
      'exact_all_element_set_accuracy':float(np.mean(np.all(sel==ys,axis=1))),
      'mean_predicted_total_count':float(np.mean(c+s)),
      'mean_actual_total_count':float(np.mean(true_c+true_s)),
    }

def encode(name):
    ids=[C2I.get(ch,1) for ch in name]
    # The convolutional model can process names longer than the training maximum;
    # retain all characters instead of silently discarding the ending.
    return torch.tensor([ids+[0]*max(0,L-len(ids))],dtype=torch.long)

@torch.no_grad()
def inference(name,model):
    pred=forward_names(model,encode(name))
    selected,core,special=choose(pred)
    order=np.argsort(-pred['label_probs'][0])
    return {
       'name':name,
       'predicted_element_count':int(core[0]+special[0]),
       'predicted_basic_count':int(core[0]),
       'predicted_special_count':int(special[0]),
       'elements':[LABELS[i] for i in order if selected[0,i]],
       'basic_elements':[LABELS[i] for i in order if selected[0,i] and i in BASIC],
       'special_elements':[LABELS[i] for i in order if selected[0,i] and i in SPECIAL],
       'count_distribution':{'basic':np.round(pred['core_count_probs'][0],4).tolist(),
                             'special':np.round(pred['special_count_probs'][0],4).tolist()},
       'top10_elements':[{'element':LABELS[i],'score':round(float(pred['label_probs'][0,i]),4)} for i in order[:10]],
       'notes':'Learned counts are uncertain; training tags include associated and class-like special elements, not just in-game elemental icons.'
    }

def run_evaluation():
    (ROOT/'count_models').mkdir(exist_ok=True)
    per_fold=[];oof=[];steps=[]
    for fold in range(3):
        tr,dev,pool,test=split(fold)
        m,n,h=train(tr,steps=320,seed=401+fold,dev_ids=dev)
        model,_,_=train(pool,steps=n,seed=701+fold)
        p=forward_names(model,X[test]);metrics=evaluate(test,p)
        fold_record={'fold':fold,'best_step':n,'metrics':metrics,'dev_history':h}
        steps.append(n);per_fold.append(fold_record)
        selected,c,s=choose(p)
        for j,idx in enumerate(test):
            oof.append({'name':NAMES[idx],'actual_elements':[LABELS[k] for k in np.where(Y[idx].numpy()>0)[0]],
                        'predicted_elements':[LABELS[k] for k in np.where(selected[j])[0]],
                        'actual_basic_count':int(YC[idx]),'predicted_basic_count':int(c[j]),
                        'actual_special_count':int(YS[idx]),'predicted_special_count':int(s[j]),
                        'predicted_scores':{LABELS[k]:round(float(p['label_probs'][j,k]),4) for k in range(len(LABELS))}})
        print(f'fold {fold}: steps={n}, core-count={metrics["core_count_accuracy"]:.3f}, total-count={metrics["total_count_accuracy"]:.3f}, label F1={metrics["all_micro_F1_count_aware"]:.3f}',flush=True)
    # Aggregate the true out-of-fold predictions, not an unweighted average of fold metrics.
    records_by_name={r['name']:r for r in oof}
    sorted_records=[records_by_name[n] for n in NAMES]
    true=np.array([[int(l in r['actual_elements']) for l in LABELS] for r in sorted_records])
    result=np.array([[int(l in r['predicted_elements']) for l in LABELS] for r in sorted_records])
    old_thresh=np.array([[int(r['predicted_scores'][l]>=.25) for l in LABELS] for r in sorted_records])
    ac=np.array([r['actual_basic_count'] for r in sorted_records]); pc=np.array([r['predicted_basic_count'] for r in sorted_records])
    ass=np.array([r['actual_special_count'] for r in sorted_records]); ps=np.array([r['predicted_special_count'] for r in sorted_records])
    total=ac+ass
    macro={'n':len(NAMES),
      'core_count_accuracy':float(np.mean(ac==pc)),
      'special_count_accuracy':float(np.mean(ass==ps)),
      'total_count_accuracy':float(np.mean(total==pc+ps)),
      'joint_two_counts_accuracy':float(np.mean((ac==pc)&(ass==ps))),
      'majority_core_count':int(np.bincount(ac).argmax()),
      'majority_core_count_accuracy':float(np.max(np.bincount(ac))/len(ac)),
      'majority_special_count':int(np.bincount(ass).argmax()),
      'majority_special_count_accuracy':float(np.max(np.bincount(ass))/len(ass)),
      'majority_total_count':int(np.bincount(total).argmax()),
      'majority_total_count_accuracy':float(np.max(np.bincount(total))/len(total)),
      'all_micro_F1_count_aware':float(f1_score(true.ravel(),result.ravel(),zero_division=0)),
      'all_micro_F1_threshold_025':float(f1_score(true.ravel(),old_thresh.ravel(),zero_division=0)),
      'all_exact_sets_accuracy':float(np.mean(np.all(true==result,axis=1))),
      'actual_total_count_histogram':{str(i):int(np.sum(total==i)) for i in range(6)},
      'predicted_total_count_histogram':{str(i):int(np.sum((pc+ps)==i)) for i in range(8)},
      'count_confusion':[[int(np.sum((total==i)&((pc+ps)==j))) for j in range(8)] for i in range(6)],
      'folds':per_fold,
    }
    json.dump(macro,open(ROOT/'count_cv_metrics.json','w'),ensure_ascii=False,indent=2)
    json.dump(sorted_records,open(ROOT/'count_heldout_predictions.json','w'),ensure_ascii=False,indent=2)
    med=int(statistics.median(steps))
    final,_,_=train(list(range(len(RECORDS))),steps=med,seed=20268)
    torch.save({'state':final.state_dict(),'labels':LABELS,'chars':CHAR,'maxlen':L,'architecture':'count_aware_char_cnn_v2',
                'training_steps':med,'train_examples':len(RECORDS)},ROOT/'count_models'/'elements_with_count.pt')
    print('aggregate',json.dumps({k:v for k,v in macro.items() if k not in ('folds','count_confusion')},indent=2),flush=True)
    return macro

def load():
    q=torch.load(ROOT/'count_models'/'elements_with_count.pt',map_location='cpu',weights_only=True)
    if q['labels']!=LABELS or q['chars']!=CHAR:raise ValueError('Dataset/vocabulary mismatch')
    m=CountAwareCNN();m.load_state_dict(q['state']);m.eval();return m

if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--train',action='store_true');p.add_argument('--name',default='Bumbleler');a=p.parse_args()
    if a.train: run_evaluation()
    else:print(json.dumps(inference(a.name,load()),indent=2,ensure_ascii=False))
