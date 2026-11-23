"""Train three small experimental island models: name RNN, gimmick Ridge,
teleport roster LogisticRegression. All labels come from island_corpus.json.
No synthetic island names/rosters are added to supervised training.
"""
import json,random,collections,pathlib,math,time,sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold,LeaveOneOut
from sklearn.metrics import f1_score, average_precision_score, log_loss
import joblib
P=pathlib.Path(__file__).resolve().parent
D=json.loads((P/'island_corpus.json').read_text()); G=json.loads((P/'gimmicks.json').read_text())
M=json.loads((P/'source/monsters.json').read_text());K=[r for r in D if r['eligible_for_training']]
torch.set_num_threads(1)
class NameGRU(nn.Module):
 def __init__(self,nchar,ncond,dim=16,hid=36,drop=.25):
  super().__init__();self.emb=nn.Embedding(nchar,dim);self.context=nn.Linear(ncond,16);self.gru=nn.GRU(dim+16,hid,batch_first=True);self.out=nn.Linear(hid,nchar);self.drop=nn.Dropout(drop)
 def forward(self,x,c):
  v=self.context(c).tanh().unsqueeze(1).expand(-1,x.shape[1],-1)
  z,_=self.gru(torch.cat([self.emb(x),v],-1));return self.out(self.drop(z))

def name_setup():
 chars=['<pad>','<bos>','<eos>']+sorted(set(''.join(r['name'] for r in K)))
 # allow basic elements not observed in actual breeding-island name annotations
 el_vocab=sorted(set(json.loads((P/'basic_elements.json').read_text()))|{s for r in K for s in r['elements']})
 typ=['breeding','teleporting','zapping'];vocab={x:i for i,x in enumerate(chars)}
 def enc(r):
  t=[1]+[vocab[ch] for ch in r['name']]+[2]
  vec=np.zeros(len(typ)+len(el_vocab),dtype='float32');vec[typ.index(r['type'])]=1
  for e in r['elements']:
   if e in el_vocab:vec[len(typ)+el_vocab.index(e)]=1
  return t,vec
 return chars,el_vocab,typ,enc

def train_name():
 chars,el_vocab,types,enc=name_setup(); recs=K; toks,conds=zip(*(enc(r) for r in recs));n=len(recs)
 ids=sorted(set(r['group'] for r in recs));random.Random(12).shuffle(ids)
 groups={g:i%4 for i,g in enumerate(ids)}
 summaries=[]
 def dofit(trainids,steps,seed,validids=None,save=False):
  torch.manual_seed(seed); rng=np.random.default_rng(seed+1)
  net=NameGRU(len(chars),3+len(el_vocab));opt=torch.optim.AdamW(net.parameters(),lr=.004,weight_decay=.04)
  def loss_subset(ii):
   xx=[toks[j] for j in ii];leng=max(map(len,xx));x=torch.zeros((len(ii),leng-1),dtype=torch.long);y=torch.zeros_like(x)
   for row,t in enumerate(xx):x[row,:len(t)-1]=torch.tensor(t[:-1]);y[row,:len(t)-1]=torch.tensor(t[1:])
   cond=torch.tensor(np.stack([conds[j] for j in ii]));return x,y,cond
  track=[];valid_cache=loss_subset(validids) if validids else None
  for step in range(steps):
   net.train();chosen=rng.choice(trainids,min(16,len(trainids)),replace=True).tolist();x,y,c=loss_subset(chosen)
   pred=net(x,c);loss=F.cross_entropy(pred.transpose(1,2),y,ignore_index=0)
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
   if valid_cache and (step+1)%10==0:
    net.eval()
    with torch.no_grad():
     a,b,c=valid_cache;vl=float(F.cross_entropy(net(a,c).transpose(1,2),b,ignore_index=0))
    track.append([step+1,round(vl,4)])
  if save:torch.save({'state':net.state_dict(),'chars':chars,'elements':el_vocab,'types':types,'steps':steps,'parameters':sum(p.numel() for p in net.parameters())},P/'models/island_name_gru.pt')
  return net,track
 for fold in range(4):
  tr=[i for i,r in enumerate(recs) if groups[r['group']]!=fold];va=[i for i,r in enumerate(recs) if groups[r['group']]==fold]
  _,curve=dofit(tr,120,1100+fold,va);summaries.append({'fold':fold,'train_names':len(tr),'held_out_names':len(va),'curve':curve})
 pooled=[]
 for step in range(10,121,10):
  vals=[next(v for s,v in q['curve'] if s==step) for q in summaries];pooled.append({'step':step,'mean_heldout_loss':float(np.mean(vals)),'std':float(np.std(vals))})
 # Early stopping is constrained to at most 100 steps to avoid name memorization.
 best=min([p for p in pooled if p['step']<=100],key=lambda x:x['mean_heldout_loss'])
 dofit(list(range(n)),best['step'],3301,save=True)
 results={'total_training_names':n,'independent_groups':len(ids),'train_by_type':dict(collections.Counter(r['type'] for r in recs)),'parameters':sum(p.numel() for p in NameGRU(len(chars),3+len(el_vocab)).parameters()),'selected_steps':best['step'],'cv':pooled,'folds':summaries}
 (P/'name_cv.json').write_text(json.dumps(results,indent=2));return results

# Name-only gimmick classifier: linear kernel of character n-grams. Count and presence
# of actual monster classes come from an existing roster, never generated from a name.
# Its output is multi-label but may be unreliable for unseen fantasy island names.
def train_gimmicks():
 R=[r for r in K if r['type']=='breeding' and not r['mirror']]; names=[r['name'] for r in R]
 labels=np.array([[float(t in r['gimmicks']) for t in G] for r in R]);vec=TfidfVectorizer(analyzer='char',ngram_range=(2,4),lowercase=True)
 X=vec.fit_transform(names).toarray();lo=LeaveOneOut();pred=np.zeros_like(labels)
 for train,test in lo.split(X):
  model=Ridge(alpha=1.0).fit(X[train],labels[train]);pred[test]=np.clip(model.predict(X[test]),0,1)
 # With very few islands, a single global threshold is interpretable and not tuned
 # directly against the final one-island examples.
 baseline=np.repeat(labels.mean(0)[None,:],len(R),axis=0)
 def metrics(pp):
  b=pp>=.45
  return {'micro_f1':round(float(f1_score(labels.astype(int).ravel(),b.astype(int).ravel(),zero_division=0)),4),
          'macro_f1':round(float(f1_score(labels.astype(int),b.astype(int),average='macro',zero_division=0)),4),
          'mean_absolute_error':round(float(abs(pp-labels).mean()),4)}
 final=Ridge(alpha=1.0).fit(X,labels)
 joblib.dump({'vectorizer':vec,'regression':final,'tags':G,'threshold':.45},P/'models/gimmicks.joblib')
 out={'training_breeding_islands':len(R),'gimmicks':G,'heldout_leave_one_island_out':metrics(pred),'constant_prevalence_baseline':metrics(baseline),'positive_counts':dict(zip(G,labels.sum(0).astype(int).tolist())),
      'caveat':'Only released non-mirror breeding islands used in training; uncommon gimmicks have extremely few examples.'}
 (P/'gimmick_cv.json').write_text(json.dumps(out,indent=2));return out

# Pairwise classifier. Only the two names are included in the input feature string,
# plus generic NAME-to-NAME spelling similarities. No real monster elements/classes.
def pair_str(a,b):
 return f'ISLAND {a} MONSTER {b}'
def train_teleport():
 R=[r for r in K if r['type']=='teleporting' and not r['mirror']]
 names=[m['name'] for m in M]
 data=[(r['name'],monster,int(monster in set(r['roster']))) for r in R for monster in names]
 y=np.array([z for _,_,z in data]);groups=np.array([r for r,_,_ in data]);s=[pair_str(a,b) for a,b,_ in data]
 vec=TfidfVectorizer(analyzer='char',ngram_range=(2,4),min_df=2,max_features=6000,sublinear_tf=True)
 X=vec.fit_transform(s);gkf=GroupKFold(n_splits=min(4,len(R)));scores=[];outprobs=np.zeros(len(data))
 for fold,(tr,va) in enumerate(gkf.split(X,y,groups)):
  model=LogisticRegression(C=.5,max_iter=500,class_weight='balanced',random_state=15)
  model.fit(X[tr],y[tr]);pp=model.predict_proba(X[va])[:,1];outprobs[va]=pp
  scores.append({'fold':fold,'islands_held_out':sorted(set(groups[va])),'average_precision':round(float(average_precision_score(y[va],pp)),4),'prevalence':round(float(y[va].mean()),4)})
 # Pick threshold from OOF only. This is an island-heldout test of generalization.
 best=max((f1_score(y,outprobs>=t),t) for t in [.15,.2,.25,.3,.35,.4,.45,.5,.55,.6]);threshold=float(best[1])
 final=LogisticRegression(C=.5,max_iter=500,class_weight='balanced',random_state=15).fit(X,y)
 joblib.dump({'vectorizer':vec,'classifier':final,'threshold':threshold},P/'models/teleport_pair.joblib')
 prevalence=float(y.mean());out={'training_islands':len(R),'pairs':len(data),'positives':int(y.sum()),'prevalence':round(prevalence,4),'heldout_island_cv':scores,'out_of_fold_AP':round(float(average_precision_score(y,outprobs)),4),'frequency_baseline_AP':round(prevalence,4),'chosen_threshold':threshold,'out_of_fold_micro_f1':round(float(best[0]),4),
 'caveat':'For completely invented islands, generalization is highly uncertain. Does not learn actual teleport mechanics.'}
 (P/'teleport_cv.json').write_text(json.dumps(out,indent=2));return out

if __name__=='__main__':
 (P/'models').mkdir(exist_ok=True)
 print('name...',flush=True);a=train_name();print('name best step',a['selected_steps'],'params',a['parameters'],flush=True)
 print('gimmick...',flush=True);b=train_gimmicks();print('gimmick cv',b['heldout_leave_one_island_out'],flush=True)
 print('teleport...',flush=True);c=train_teleport();print('teleport cv',c['out_of_fold_AP'],'vs',c['frequency_baseline_AP'],flush=True)
