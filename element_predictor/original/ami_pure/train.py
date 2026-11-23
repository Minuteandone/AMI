import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import json,random,collections,pathlib,concurrent.futures,statistics,torch
from torch import nn
import torch.nn.functional as F
P=pathlib.Path(__file__).resolve().parent
R=[json.loads(l) for l in (P/'dataset.jsonl').read_text().splitlines()]; CH=['<pad>','<bos>','<eos>']+sorted(set(''.join(r['name'] for r in R))); T=sorted({s for r in R for s in r['conditions']});L=max(len(r['name']) for r in R)+2
CORE=[T.index('element:'+e) for e in 'Air Cold Earth Water Plant Fire Light Psychic Faerie Bone Plasma Shadow Mech Crystal Poison Control Hoax Ruin Depths Electricity'.split()]; K=len(CORE); META=[i for i in range(len(T)) if i not in CORE]
X=torch.tensor([[1]+[CH.index(c) for c in r['name']]+[2]+[0]*(L-len(r['name'])-2) for r in R]); C=torch.tensor([[float(t in r['conditions']) for t in T] for r in R]); CL=[next(t for t in r['conditions'] if t.startswith('class:')) for r in R]; S=[tuple(j for j in range(K) if C[i,CORE[j]]) for i in range(len(R))]
OPT=collections.defaultdict(set)
for i,s in enumerate(S):OPT[CL[i],len(s)].add(s)
def dump(n,o):(P/n).write_text(json.dumps(o,indent=2,ensure_ascii=False))
def folds():
 g=collections.defaultdict(list);rng=random.Random(44591)
 for i,k in enumerate(CL):g[k].append(i)
 out=[[],[],[]]
 for a in g.values():
  rng.shuffle(a);start=min(range(3),key=lambda i:len(out[i]))
  for j,i in enumerate(a):out[(start+j)%3].append(i)
 return out
FOLDS=folds()
def split(f):
 test=FOLDS[f];pool=sorted(set(range(len(R)))-set(test));rng=random.Random(883+f);g=collections.defaultdict(list)
 for i in pool:g[CL[i]].append(i)
 dev=[]
 for a in g.values():
  rng.shuffle(a)
  if len(a)>=4:dev+=a[:max(1,round(.2*len(a)))]
 return sorted(set(pool)-set(dev)),dev,pool,test
CONFIG={'baseline':(0,0,0,False,False),'contrast':(1.,0,0,False,False),'reverse':(0,.6,0,False,False),'fragments':(0,0,.3,False,False),'combined':(1.,.6,.3,False,False),'balanced':(1.,.6,.3,True,True),'strong':(2.,.8,.3,True,True)}
class Net(nn.Module):
 def __init__(self,flags,motifs,d=32):
  super().__init__();self.flags=flags;self.motifs=motifs;self.d=d;self.emb=nn.Embedding(len(CH),d,padding_idx=0);self.local=nn.Linear(d*4,d);self.context=nn.Linear(d*2,d);self.norm=nn.LayerNorm(d);self.style=nn.Linear(len(META),d,bias=False);self.element=nn.Linear(K,d,bias=False);self.fuse=nn.Linear(d*2,d);self.out=nn.Linear(d,len(CH));self.e_gate=nn.Linear(d,d);self.e_out=nn.Linear(d,len(CH),bias=False);self.drop=nn.Dropout(.18);self.rev=nn.Linear(d*2,K);self.frag=nn.Linear(d,max(1,len(motifs)))
 def features(self,x):
  e=self.emb(x);p=F.pad(e,(0,0,3,0));a=F.gelu(self.local(torch.cat([p[:,j:j+x.shape[1]] for j in range(4)],-1)));h=a.cumsum(1)/torch.arange(1,x.shape[1]+1)[None,:,None];return self.norm(F.gelu(self.context(torch.cat([a,h],-1)))+a)
 def forward(self,x,c,aux=False):
  h=self.features(x);s=c[:,META];e=c[:,CORE];s=self.style(s/s.sum(1,keepdim=True).clamp_min(1).sqrt());e=self.element(e/e.sum(1,keepdim=True).clamp_min(1).sqrt());z=F.gelu(self.fuse(torch.cat([h,s[:,None].expand_as(h)],-1)))+h;out=self.out(self.drop(z))+self.e_out(torch.tanh(self.e_gate(z))*e[:,None]);return (out,self.frag(z)) if aux else out
 def reverse(self,x):
  h=self.features(x);mask=(x>2).unsqueeze(-1);a=(h*mask).sum(1)/mask.sum(1).clamp_min(1);b=h.masked_fill(~mask,-1e4).max(1).values;return self.rev(torch.cat([a,b],-1))
def losses(z,y):return F.cross_entropy(z.transpose(1,2),y,ignore_index=0,reduction='none').sum(1)/(y!=0).sum(1)
@torch.no_grad()
def nll(m,x,c):
 m.eval();return torch.cat([losses(m(a[:,:-1],b),a[:,1:]) for a,b in zip(x.split(128),c.split(128))])
@torch.no_grad()
def evaluate(m,ids):
 m.eval();xx=[];cc=[];cases=[];rev=m.reverse(X[ids]);base=nll(m,X[ids],C[ids]);
 for n,i in enumerate(ids):
  s=S[i];opts=sorted(OPT[CL[i],len(s)])
  if not s or len(opts)<2:continue
  a=len(xx)
  for q in opts:
   v=C[i].clone();v[CORE]=0;v[[CORE[j] for j in q]]=1;xx.append(X[i]);cc.append(v)
  cases.append((i,a,len(opts),opts.index(s),n,opts))
 vals=nll(m,torch.stack(xx),torch.stack(cc));rows=[]
 for i,a,k,p,n,opts in cases:
  v=vals[a:a+k];r=v[p];rank=1+int((v<r-1e-7).sum());ties=int((abs(v-r)<=1e-7).sum());rv=torch.tensor([sum(float(rev[n,j]) for j in q) for q in opts]);rr=rv[p];rrank=1+int((rv>rr+1e-7).sum());rties=int((abs(rv-rr)<=1e-7).sum());rows.append(dict(name=R[i]['name'],class_=CL[i],rank=rank,accuracy=1/ties if rank==1 else 0,chance=1/k,margin=float((v.sum()-r)/(k-1)-r),reverse_accuracy=1/rties if rrank==1 else 0,reverse_rank=rrank))
 return dict(nll=float(base.mean()),n=len(rows),**{k:statistics.mean(a[k] for a in rows) for k in ['accuracy','chance','margin','reverse_accuracy']},detail=rows)
def motifs_for(ids):
 count=collections.Counter(tuple(row[j:j+2]) for row in X[ids].tolist() for j in range(1,L-1) if min(row[j:j+2])>2);return sorted(q for q,n in count.items() if n>=2)
def train(name,ids,steps,seed,dev=None):
 torch.set_num_threads(1);torch.manual_seed(seed);g=torch.Generator().manual_seed(seed+1);flags=CONFIG[name];cont,rev,frag,balance,drop=flags;motifs=motifs_for(ids) if frag else [];m=Net(flags,motifs);opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.035);ii=torch.tensor(ids);e=C[ii][:,CORE];freq=e.sum(0).clamp_min(1);prob=(e@freq.rsqrt()/e.sum(1).clamp_min(1)).clamp_min(.12);prob=prob/prob.sum();pw=((len(ids)-freq)/freq).sqrt().clamp(1,5);lookup=torch.full((len(CH),len(CH)),-100,dtype=torch.long)
 for j,(a,b) in enumerate(motifs):lookup[a,b]=j
 options=collections.defaultdict(set)
 for i in ids:options[CL[i],len(S[i])].add(S[i])
 negs={i:[q for q in sorted(options[CL[i],len(S[i])]) if q!=S[i]] for i in ids};rng=random.Random(seed+4);logs=[];best=None;bestscore=-1e9
 for step in range(1,steps+1):
  m.train();b=ii[torch.multinomial(prob,32,replacement=True,generator=g)] if balance else ii[torch.randint(len(ii),(32,),generator=g)];x=X[b];x=x[:,:int((x!=0).sum(1).max())];y=x[:,1:];c=C[b].clone();c[:,META]*=(torch.rand((len(b),len(META)),generator=g)>(.22 if drop else .08));orig=c.clone()
  if drop:c[:,CORE]*=(torch.rand((len(b),K),generator=g)>.10)
  z,a=m(x[:,:-1],c,True);l=losses(z,y).mean()
  if cont:
   good=[];wrong=[]
   for j,i in enumerate(b.tolist()):
    if negs[i]:
     q=rng.choice(negs[i]);v=orig[j].clone();v[CORE]=0;v[[CORE[k] for k in q]]=1;good.append(j);wrong.append(v)
   if good:
    zx=x[good];pos,neg=losses(m(zx[:,:-1].repeat(2,1),torch.cat([orig[good],torch.stack(wrong)])),zx[:,1:].repeat(2,1)).chunk(2);l=l+cont*F.softplus(2*(pos-neg)).mean()
  if rev:l=l+rev*F.binary_cross_entropy_with_logits(m.reverse(x),C[b][:,CORE],pos_weight=pw)
  if frag:
   t=lookup[y[:,:-1],y[:,1:]];mask=t!=-100
   if mask.any():l=l+frag*F.cross_entropy(a[:,:-1][mask],t[mask])
  opt.zero_grad();l.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  if dev is not None and step%40==0:
   q=evaluate(m,dev);q.pop('detail');q['step']=step;q['train_nll']=float(nll(m,X[ids],C[ids]).mean());q['score']=-q['nll']+1.1*q['margin']+.5*q['accuracy'];logs.append(q)
   if q['score']>bestscore:bestscore=q['score'];best=step
 return m,best,logs
def job(arg):
 name,f=arg;tr,d,pool,test=split(f);m,steps,logs=train(name,tr,240,1801+f,d);m,_,_=train(name,pool,steps,2801+f);q=evaluate(m,test);q['train_nll']=float(nll(m,X[pool],C[pool]).mean());rec=dict(name=name,fold=f,steps=steps,dev=logs,test=q,parameters=sum(p.numel() for p in m.parameters()));dump(f'models/{name}_fold{f}.json',rec);print(name,f,steps,round(q['nll'],3),round(q['accuracy'],3),round(q['margin'],3),flush=True);return rec
def full_job(arg):
 name,steps=arg;m,_,_=train(name,list(range(len(R))),steps,3801);torch.save(dict(state=m.state_dict(),flags=CONFIG[name],motifs=m.motifs,chars=CH,tags=T,steps=steps),P/'models'/f'{name}.pt');return name
if __name__=='__main__':
 torch.set_num_threads(1);dump('splits.json',[{k:[R[i]['name'] for i in v] for k,v in zip(['dev_train','dev','outer_train','test'],split(f))} for f in range(3)])
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:rs=list(ex.map(job,[(n,f) for n in CONFIG for f in range(3)]))
 dump('results.json',rs);summary={}
 for n in CONFIG:
  a=[r for r in rs if r['name']==n];rows=sum([r['test']['detail'] for r in a],[]);summary[n]={'n':len(rows),'steps':int(statistics.median(r['steps'] for r in a)),'parameters':a[0]['parameters'],'nll':statistics.mean(r['test']['nll'] for r in a),'train_nll':statistics.mean(r['test']['train_nll'] for r in a),**{k:statistics.mean(q[k] for q in rows) for k in ['accuracy','chance','margin','reverse_accuracy']}}
 dump('summary.json',summary);print(summary,flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:list(ex.map(full_job,[(n,q['steps']) for n,q in summary.items()]))
