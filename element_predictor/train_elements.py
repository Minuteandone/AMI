"""AMI Name -> Elements. Only official AMI name/condition pairs, no language pretraining.
Run: python train_elements.py --train  (3 stratified held-out folds + final model).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse,json,math,random,collections,pathlib,statistics
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score,f1_score,precision_score,recall_score

ROOT=pathlib.Path(__file__).resolve().parent
SOURCE=ROOT/'dataset.jsonl'
RECORDS=[json.loads(s) for s in SOURCE.read_text(encoding='utf-8').splitlines() if s.strip()]
LABELS=sorted({t[len('element:'):] for r in RECORDS for t in r['conditions'] if t.startswith('element:')})
CORE=['Air','Cold','Earth','Water','Plant','Fire','Light','Psychic','Faerie','Bone','Plasma','Shadow','Mech','Crystal','Poison','Control','Hoax','Ruin','Depths','Electricity']
CHAR=['<PAD>','<UNK>']+sorted({c for r in RECORDS for c in r['name']})
C2I={x:i for i,x in enumerate(CHAR)}
L=max(len(r['name']) for r in RECORDS)
NAMES=[r['name'] for r in RECORDS]
X=torch.tensor([[C2I.get(c,1) for c in s]+[0]*(L-len(s)) for s in NAMES],dtype=torch.long)
Y=torch.tensor([[float('element:'+l in r['conditions']) for l in LABELS] for r in RECORDS],dtype=torch.float32)
IDX=[LABELS.index(c) for c in CORE]
YCORE=Y[:,IDX]
CLASS=[next(t for t in r['conditions'] if t.startswith('class:')) for r in RECORDS]
K=len(LABELS)
class CNN(nn.Module):
    def __init__(self,dropout=.3):
        super().__init__()
        self.emb=nn.Embedding(len(CHAR),24,padding_idx=0)
        self.convs=nn.ModuleList([nn.Conv1d(24,40,k,padding='same') for k in [2,3,4]])
        self.project=nn.Sequential(nn.Linear(240,96),nn.GELU(),nn.Dropout(dropout),nn.Linear(96,64),nn.GELU())
        self.out=nn.Linear(64,K)
        self.count=nn.Linear(64,6)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        e=self.emb(x).transpose(1,2);mask=(x!=0)
        a=[]
        for conv in self.convs:
            z=F.gelu(conv(e));poolmax=z.masked_fill(~mask[:,None],-1e4).amax(-1)
            poolavg=(z*mask[:,None]).sum(-1)/mask.sum(-1)[:,None].clamp_min(1)
            a.extend([poolmax,poolavg])
        z=self.project(torch.cat(a,dim=-1))
        return self.out(self.dropout(z)),self.count(z)

def folds():
    # 3 folds, group by class to avoid concentrating scarce classes in one fold.
    rng=random.Random(44591);g=collections.defaultdict(list);out=[[],[],[]]
    for i,c in enumerate(CLASS):g[c].append(i)
    for a in g.values():
        rng.shuffle(a)
        start=min(range(3),key=lambda i:len(out[i]))
        for j,index in enumerate(a):out[(start+j)%3].append(index)
    return [sorted(q) for q in out]
FOLDS=folds()

def split(fold):
    test=FOLDS[fold];remain=sorted(set(range(len(RECORDS)))-set(test));rng=random.Random(10000+fold)
    groups=collections.defaultdict(list)
    for i in remain:groups[CLASS[i]].append(i)
    dev=[]
    for v in groups.values():
        rng.shuffle(v)
        if len(v)>=4:dev.extend(v[:max(1,round(.18*len(v)))])
    train=sorted(set(remain)-set(dev))
    return train,sorted(dev),remain,test

def posweight(ids):
    p=Y[ids].sum(0)
    # A 1-example element is not learnable with ordinary held-out validation.
    return ((len(ids)-p).clamp_min(1)/p.clamp_min(1)).sqrt().clamp(1.,5.)

def train(ids,steps,seed,dev=None):
    torch.manual_seed(seed);torch.set_num_threads(1)
    g=torch.Generator().manual_seed(seed+11);m=CNN();opt=torch.optim.AdamW(m.parameters(),lr=.0018,weight_decay=.035)
    pos=posweight(ids);train_idx=torch.tensor(ids);history=[];beststep=steps;bestmetric=-1.;beststate=None
    yc=YCORE.sum(-1).long().clamp(0,5)
    for step in range(1,steps+1):
        m.train();b=train_idx[torch.randint(len(train_idx),(min(64,len(train_idx)),),generator=g)];z,c=m(X[b]);loss=F.binary_cross_entropy_with_logits(z,Y[b],pos_weight=pos)+.2*F.cross_entropy(c,yc[b]);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
        if dev is not None and step%20==0:
            p=predict(m,dev);metric=average_precision_score(YCORE[dev].numpy().ravel(),p[:,IDX].ravel())
            trloss=float(F.binary_cross_entropy_with_logits(m(X[train_idx])[0],Y[train_idx],pos_weight=pos))
            history.append({'step':step,'dev_core_micro_AP':metric,'train_loss':trloss})
            if metric>bestmetric+1e-6:
                bestmetric=metric;beststep=step;beststate={k:v.detach().clone() for k,v in m.state_dict().items()}
    if dev is not None:m.load_state_dict(beststate)
    m.eval();return m,beststep,history

@torch.no_grad()
def predict(model,ids):
    model.eval();return torch.sigmoid(model(X[ids])[0]).cpu().numpy()

def nearest(train_ids,test_ids):
    vector=TfidfVectorizer(analyzer='char',ngram_range=(2,4),lowercase=False,sublinear_tf=True)
    train_names=[NAMES[i] for i in train_ids]
    a=vector.fit_transform(train_names);b=vector.transform([NAMES[i] for i in test_ids]);sim=(b@a.T).toarray()
    ys=Y[train_ids].numpy();idx=np.argsort(-sim,axis=1)[:,:7];near=np.take_along_axis(sim,idx,axis=1)
    weights=np.maximum(near,1e-8)**2
    scores=(ys[idx]*weights[:,:,None]).sum(1)/(weights.sum(1)[:,None]+1e-8)
    # Blend 20% global prior, to avoid treating unseen letters as negative certainty
    return .8*scores+.2*ys.mean(0)[None,:]

def metrics(test,pred,threshold=.25):
    actual=Y[test].numpy();a=actual[:,IDX];b=pred[:,IDX];P=(pred>=threshold).astype(int);C=(b>=threshold).astype(int)
    return {'n':len(test),'core_micro_AP':float(average_precision_score(a.ravel(),b.ravel())),
        'all_micro_AP':float(average_precision_score(actual.ravel(),pred.ravel())),
        'core_micro_F1':float(f1_score(a.ravel(),C.ravel(),zero_division=0)),
        'all_micro_F1':float(f1_score(actual.ravel(),P.ravel(),zero_division=0)),
        'core_precision':float(precision_score(a.ravel(),C.ravel(),zero_division=0)),
        'core_recall':float(recall_score(a.ravel(),C.ravel(),zero_division=0)),
        'core_top1_hit':float(np.mean([a[i,np.argmax(b[i])]>0 for i in range(len(a))])),
        'exact_core_match':float(np.mean(np.all(C==a,axis=1))),
        'core_mean_predicted_count':float(C.sum(-1).mean()),'core_mean_actual_count':float(a.sum(-1).mean())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');q=ap.parse_args()
    if not q.train:ap.print_help();return
    (ROOT/'models').mkdir(exist_ok=True)
    folds_results=[];alltest=[];allpred=[];allbase=[];beststeps=[]
    for f in range(3):
        tr,dev,pool,test=split(f)
        m,step,h=train(tr,280,seed=610+f,dev=dev)
        # threshold calibrated on the development group only
        pd=predict(m,dev)
        scores=[(t,f1_score(YCORE[dev].numpy().ravel(),(pd[:,IDX]>=t).ravel(),zero_division=0)) for t in np.arange(.12,.51,.02)]
        threshold=float(max(scores,key=lambda q:q[1])[0])
        full,_,_=train(pool,step,seed=710+f)
        p=predict(full,test);pb=nearest(pool,test)
        foldrow={'fold':f,'steps':step,'threshold_from_dev':threshold,'test_names':[NAMES[i] for i in test],
            'neural':metrics(test,p,threshold),'knn':metrics(test,pb,threshold), 'dev_history':h}
        folds_results.append(foldrow);alltest+=test;allpred.extend(p.tolist());allbase.extend(pb.tolist());beststeps.append(step)
        print('fold',f,'steps',step,'threshold',round(threshold,3),'core AP',round(foldrow['neural']['core_micro_AP'],3),'kNN',round(foldrow['knn']['core_micro_AP'],3),flush=True)
    pred=np.array(allpred);base=np.array(allbase);ids=np.array(alltest)
    # global out-of-fold scores with fair per-fold model separation
    aggregate={'n':len(ids),'neural':metrics(ids,pred,.25),'knn':metrics(ids,base,.25),'core_prevalence':float(YCORE.numpy().mean()),'all_prevalence':float(Y.numpy().mean()),'labels':LABELS}
    outof=[]
    for i,p,b in zip(ids,pred,base):
        order=np.argsort(-p)
        outof.append({'name':NAMES[i],'actual':list(l for l in LABELS if Y[i,LABELS.index(l)]==1),
                      'predicted_top8':[{ 'element':LABELS[j],'score':round(float(p[j]),4)} for j in order[:8]]})
    json.dump({'folds':folds_results,'aggregate':aggregate},open(ROOT/'cv_metrics.json','w'),indent=2,ensure_ascii=False)
    json.dump(outof,open(ROOT/'heldout_predictions.json','w'),indent=2,ensure_ascii=False)
    finalsteps=round(statistics.median(beststeps));m,_,_=train(list(range(len(RECORDS))),finalsteps,seed=3801)
    torch.save({'state':m.state_dict(),'chars':CHAR,'labels':LABELS,'maxlen':L,'steps':finalsteps,'architecture':'char_cnn_multilabel_v1','source':'263 official MSM names and their saved element associations'},ROOT/'models'/'elements_cnn.pt')
    print('AGGREGATE',json.dumps(aggregate,indent=2)[:3000],flush=True)
if __name__=='__main__':main()
