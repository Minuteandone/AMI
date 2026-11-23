"""AMI Islands main-game: a name-only, multi-label island classifier.
Training inputs are exclusively Unicode monster name characters; no class/element/metadata.
Uses authentic island-placement labels transcribed from https://www.jetpunk.com/user-quizzes/1629255/all-monsters-in-my-singing-monsters-by-island
"""
import argparse,json,random,copy,math,pathlib,collections,time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
P=pathlib.Path(__file__).resolve().parent
ISLANDS=json.loads((P/'islands.json').read_text())
RECS=[x for x in json.loads((P/'placements.json').read_text()) if x.get('training_eligible',bool(x['islands']))]
NAMES=[x['name'] for x in RECS]; Y=np.asarray([x['island_vector'] for x in RECS],dtype=np.float32)
CHARS=['<pad>','<unk>']+sorted({ch for name in NAMES for ch in name}); C={ch:i for i,ch in enumerate(CHARS)}
MAXLEN=max(len(n) for n in NAMES)+4
NONMIRROR=[j for j,s in enumerate(ISLANDS) if not s.startswith('Mirror ')]
torch.set_num_threads(1)

def encode(names):
    return torch.tensor([[C.get(ch,1) for ch in name[:MAXLEN]]+[0]*max(0,MAXLEN-len(name)) for name in names],dtype=torch.long)

class IslandsCNN(nn.Module):
    def __init__(self,vocab=len(CHARS),num_islands=len(ISLANDS),emb=24,width=32,hidden=80,drop=.35):
        super().__init__();self.embed=nn.Embedding(vocab,emb,padding_idx=0)
        self.convs=nn.ModuleList([nn.Conv1d(emb,width,k,padding='same') for k in (2,3,4,5)])
        self.norm=nn.LayerNorm(width*8)
        self.fc=nn.Sequential(nn.Linear(width*8,hidden),nn.GELU(),nn.Dropout(drop),nn.Linear(hidden,num_islands))
    def forward(self,x):
        e=self.embed(x).transpose(1,2)
        feat=[]
        # Padding can influence the pooled output; consistent full-length padded training fixes this for v1.
        for conv in self.convs:
            h=F.gelu(conv(e));feat.extend([h.max(2).values,h.mean(2)])
        return self.fc(self.norm(torch.cat(feat,1)))

def metrics(y,p,thr,select=NONMIRROR):
    yy=y[:,select];pp=p[:,select];b=(pp>=thr).astype(int)
    ans={'micro_f1':round(float(f1_score(yy,b,average='micro',zero_division=0)),4),
         'macro_f1':round(float(f1_score(yy,b,average='macro',zero_division=0)),4),
         'micro_precision':round(float(precision_score(yy,b,average='micro',zero_division=0)),4),
         'micro_recall':round(float(recall_score(yy,b,average='micro',zero_division=0)),4),
         'subset_accuracy':round(float(np.mean((yy==b).all(axis=1))),4),
         'mean_predicted_islands':round(float(b.sum(axis=1).mean()),2),
         'mean_actual_islands':round(float(yy.sum(axis=1).mean()),2)}
    aps=[]
    for j in range(yy.shape[1]):
        if 0<yy[:,j].sum()<len(yy): aps.append(float(average_precision_score(yy[:,j],pp[:,j])))
    ans['mean_average_precision']=round(float(np.mean(aps)) if aps else 0,4)
    return ans

def best_thr(y,p):
    # Fast vectorized micro-F1 across non-Mirror islands. Don't recalculate AP 20x per epoch.
    yy=y[:,NONMIRROR].astype(bool);pp=p[:,NONMIRROR]
    choices=np.arange(.10,.601,.025)
    b=pp[:,:,None]>=choices[None,None,:]
    tp=(b&yy[:,:,None]).sum(axis=(0,1)); pred=b.sum(axis=(0,1));actual=yy.sum()
    scores=2*tp/np.maximum(1,pred+actual)
    return float(choices[np.argmax(scores-1e-7*np.abs(choices-.35))])

def split(seed):
    a,b=train_test_split(np.arange(len(NAMES)),test_size=.2,random_state=seed,shuffle=True)
    tr,v=train_test_split(a,test_size=.2,random_state=seed+111,shuffle=True)
    return list(tr),list(v),list(b)

def train_cnn(train_ids,val_ids,seed,max_epochs=160):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    m=IslandsCNN();opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.04)
    pos=Y[train_ids].sum(0); neg=len(train_ids)-pos
    # Rare island positives get some extra weight, but not enough to swamp all others.
    weight=np.clip(np.sqrt((neg+1)/(pos+1)),1,4)
    wt=torch.from_numpy(weight.astype('float32'))
    x=encode([NAMES[i] for i in train_ids]); yy=torch.from_numpy(Y[train_ids]);v_x=encode([NAMES[i] for i in val_ids]); v_y=Y[val_ids]
    best=None; score=-1e9;hist=[];best_epoch=1;bad=0
    for epoch in range(1,max_epochs+1):
        m.train();order=torch.randperm(len(train_ids));losses=[]
        for batch in order.split(32):
            logits=m(x[batch]);loss=F.binary_cross_entropy_with_logits(logits, yy[batch],pos_weight=wt)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();losses.append(float(loss))
        m.eval()
        with torch.no_grad():pv=torch.sigmoid(m(v_x)).numpy()
        vl=float(F.binary_cross_entropy(torch.from_numpy(np.clip(pv,1e-6,1-1e-6)),torch.from_numpy(v_y)).item())
        t=best_thr(v_y,pv); yybool=v_y[:,NONMIRROR].astype(bool); pred=(pv[:,NONMIRROR]>=t); mf1=float((2*(pred&yybool).sum())/max(1,pred.sum()+yybool.sum()));met={'micro_f1':round(mf1,4)}
        # Prefer f1, with slight tie break to lower BCE. Validation selection only.
        q=met['micro_f1']-.02*vl
        if q>score+0.0001:
            score=q;best=copy.deepcopy(m.state_dict());best_epoch=epoch;best_t=t;best_val=met;bad=0
        else:bad+=1
        if epoch in (1,5,10,20,40,80,120,160) or epoch==best_epoch:
            hist.append({'epoch':epoch,'train_bce':float(np.mean(losses)), 'val_bce':vl,'val_f1':met['micro_f1'],'threshold':t})
        if bad>=35 and epoch>=45:break
    m.load_state_dict(best);m.eval()
    return m,{'best_epoch':best_epoch,'best_threshold':best_t,'dev':best_val,'history':hist}

@torch.no_grad()
def nn_probs(m,names):
    m.eval();return torch.sigmoid(m(encode(names))).numpy()

def train_linear(tr,val):
    v=TfidfVectorizer(analyzer='char',ngram_range=(2,5),lowercase=False,binary=False,sublinear_tf=True)
    xt=v.fit_transform([NAMES[i] for i in tr]); xv=v.transform([NAMES[i] for i in val])
    classifier=OneVsRestClassifier(LogisticRegression(C=2,solver='liblinear',class_weight='balanced',max_iter=400))
    # If some rare classes all 0 in train, OneVsRest handles with constant predictor.
    classifier.fit(xt,Y[tr]);return v,classifier,classifier.predict_proba(xv)

def save_json(name,obj):(P/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False))

def main():
    results=[]; test_predictions=[]
    for seed in (1231,2342,3453):
        tr,val,test=split(seed)
        m,info=train_cnn(tr,val,seed)
        prob=nn_probs(m,[NAMES[i] for i in test]);t=info['best_threshold']; me=metrics(Y[test],prob,t)
        v,lin,lp=train_linear(tr,val);lt=best_thr(Y[val],lp)
        lprob=lin.predict_proba(v.transform([NAMES[i] for i in test])); lm=metrics(Y[test],lprob,lt)
        prev=Y[tr].mean(0); pb=np.tile(prev,(len(test),1)); bt=best_thr(Y[val],np.tile(prev,(len(val),1)));bm=metrics(Y[test],pb,bt)
        out={'seed':seed,'counts':{'train':len(tr),'validation':len(val),'test':len(test)}, 'cnn':{'threshold':t,**me,'best_epoch':info['best_epoch'],'validation':info['dev']},'linear':{'threshold':lt,**lm},'prevalence':{'threshold':bt,**bm}}
        results.append(out)
        for row,idx in enumerate(test):
            test_predictions.append({'split_seed':seed,'name':NAMES[idx],'true_islands':[ISLANDS[j] for j in np.where(Y[idx])[0]],'predicted_islands':[ISLANDS[j] for j in np.where(prob[row]>=t)[0]], 'top_5':[[ISLANDS[j],round(float(prob[row,j]),3)] for j in np.argsort(-prob[row])[:5]]})
        print('SPLIT',seed,'CNN',me['micro_f1'],'LINEAR',lm['micro_f1'],'PREV',bm['micro_f1'],'epoch',info['best_epoch'],'threshold',round(t,3),flush=True)
        if seed==1231:
            torch.save({'state':m.state_dict(),'chars':CHARS,'islands':ISLANDS,'maxlen':MAXLEN,'threshold':t, 'seed':seed,'train_names':[NAMES[i] for i in tr],'holdout_names':[NAMES[i] for i in test]},P/'validation_model.pt')
            save_json('train_history.json',info['history'])
            save_json('holdout_split.json',{'train':[NAMES[i] for i in tr], 'validation':[NAMES[i] for i in val], 'test':[NAMES[i] for i in test]})
    means={key:{m:round(float(np.mean([r[key][m] for r in results])),4) for m in ['micro_f1','macro_f1','micro_precision','micro_recall','mean_average_precision']} for key in ['cnn','linear','prevalence']}
    save_json('evaluation.json',{'splits':results,'mean':means,'note':'Repeated random splits are exploratory; same monsters can appear in test across seeds. No elements or classes used as predictor input.'})
    save_json('test_predictions.json',test_predictions)
    # Refit final model on all source-verified names (not unlabeled names).
    training_epoch=max(20,int(np.median([r['cnn']['best_epoch'] for r in results])))
    train,dev,_=split(8812)
    # Choose approximate epoch, but train on all verified rows, no test data involved in selecting architecture.
    random.seed(8812);np.random.seed(8812);torch.manual_seed(8812)
    m=IslandsCNN();opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.04)
    pos=Y.sum(0);wt=torch.from_numpy(np.clip(np.sqrt((len(Y)-pos+1)/(pos+1)),1,4).astype('float32'))
    x=encode(NAMES);yy=torch.from_numpy(Y)
    for epoch in range(training_epoch):
        m.train()
        for b in torch.randperm(len(x)).split(32):
            l=F.binary_cross_entropy_with_logits(m(x[b]),yy[b],pos_weight=wt);opt.zero_grad();l.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
    final_t=float(np.median([r['cnn']['threshold'] for r in results]))
    m.eval();torch.save({'state':m.state_dict(),'chars':CHARS,'islands':ISLANDS,'maxlen':MAXLEN,'threshold':final_t, 'seed':8812,'epochs':training_epoch,'train_names':NAMES},P/'island_model.pt')
    v=TfidfVectorizer(analyzer='char',ngram_range=(2,5),lowercase=False,sublinear_tf=True)
    classifier=OneVsRestClassifier(LogisticRegression(C=2,solver='liblinear',class_weight='balanced',max_iter=400));classifier.fit(v.fit_transform(NAMES),Y)
    joblib.dump((v,classifier,final_t),P/'linear_baseline.joblib',compress=3)
    print('FINAL',training_epoch,'threshold',final_t,'metrics',means,flush=True)
    # Sanity predictions are on seen names; they cannot demonstrate true heldout accuracy.
    samples=[]
    for name in ['Fwog','Ghazt','Shugabush','Shugarock','Anglow','Scaratar','Rootitoot','Superno-Fa','STRingmaster']:
        if name not in NAMES:continue
        p=nn_probs(m,[name])[0];samples.append({'name':name,'actual':RECS[NAMES.index(name)]['islands'], 'predicted':[[ISLANDS[j],round(float(p[j]),3)] for j in np.argsort(-p)[:7]]})
    save_json('sanity_examples.json',samples)

if __name__=='__main__':main()
