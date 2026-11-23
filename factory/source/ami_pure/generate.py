import argparse,json,torch
from train import P,Net,CH,T,CORE,R,L,CONFIG,dump

def load(name):
 q=torch.load(P/'models'/f'{name}.pt',map_location='cpu',weights_only=True);m=Net(q['flags'],q['motifs']);m.load_state_dict(q['state']);m.eval();return m
@torch.no_grad()
def generate(m,tags,n=48,seed=123,temp=.85):
 g=torch.Generator().manual_seed(seed);c=torch.tensor([[float(t in tags) for t in T]]).expand(n,-1);x=torch.ones(n,1,dtype=torch.long);done=torch.zeros(n,dtype=torch.bool);out=['']*n
 for _ in range(L+3):
  z=m(x,c)[:,-1]/temp;z[:,:2]=-1e9
  if x.shape[1]==1:z[:,2]=-1e9
  p=z.softmax(-1);a=torch.multinomial(p,1,generator=g).flatten()
  for i,j in enumerate(a.tolist()):
   if not done[i] and j>2:out[i]+=CH[j]
  done|=a==2;x=torch.cat([x,a[:,None]],1)
  if done.all():break
 return out
if __name__=='__main__':
 torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('conditions',nargs='+');p.add_argument('--model',choices=list(CONFIG)+['baseline_creative','contrast_creative','combined_creative','late_ensemble','count'],default='contrast_creative');p.add_argument('--n',type=int,default=20);p.add_argument('--seed',type=int,default=123);p.add_argument('--temperature',type=float,default=.85);a=p.parse_args();tags=[]
 for s in a.conditions:
  t=s if s in T else 'element:'+s
  if t not in T: p.error('unknown condition '+s)
  tags.append(t)
 if a.model=='count':
  from count_model import Counts
  settings=json.loads((P/'count_settings.json').read_text());out=Counts(list(range(len(R))),**settings).generate(tags,a.n,a.seed,a.temperature)
 elif a.model=='late_ensemble':
  from ensemble import Ensemble
  m=Ensemble([load(n+'_creative') for n in ['baseline','contrast','combined']]);out=generate(m,tags,a.n,a.seed,a.temperature)
 else:out=generate(load(a.model),tags,a.n,a.seed,a.temperature)
 print('\n'.join(out))
