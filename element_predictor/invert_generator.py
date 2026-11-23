"""No new weights: invert trained conditional name generator by scoring P(name|known conditioning templates).
This is a constrained similarity probe, not a calibrated posterior and not proof of generalization.
Example: python invert_generator.py Bumbleler
"""
import os,sys,json,argparse,pathlib,collections
import numpy as np, torch
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'original'/'ami_pure'))
import train as original

def load():
    q=torch.load(ROOT/'models'/'original_solstice_corrected_generator.pt',map_location='cpu',weights_only=True)
    m=original.Net(original.CONFIG['contrast'],[])
    m.load_state_dict(q['state']);m.eval();return m

@torch.no_grad()
def inverse(name,m=None,temperature=1.):
    m=m if m is not None else load()
    T=original.T;CH=original.CH
    if any(c not in CH for c in name):
        # Original model has no unknown-char token; it can't score inputs with new chars.
        return {'name':name,'error':'The original generator cannot score one or more unseen characters.'}
    if not name:return {'name':name,'error':'Name must not be empty.'}
    namesigs=collections.Counter(tuple(sorted(r['conditions'])) for r in original.R)
    sigs=list(namesigs)
    v=torch.tensor([[float(t in sig) for t in T] for sig in sigs],dtype=torch.float32)
    L=len(name)+1
    inp=torch.tensor([[CH.index('<bos>')]+[CH.index(c) for c in name]],dtype=torch.long)
    tgt=torch.tensor([[CH.index(c) for c in name]+[CH.index('<eos>')]],dtype=torch.long)
    batch=128;allnll=[]
    for start in range(0,len(sigs),batch):
        c=v[start:start+batch];x=inp.expand(len(c),-1);y=tgt.expand(len(c),-1)
        score=original.losses(m(x,c),y).cpu().numpy();allnll.extend(score.tolist())
    nll=np.asarray(allnll)
    # Each recorded monster is equally weighted a priori (thus repeat templates count more).
    logp=-nll*(len(name)+1)/temperature+np.log(np.array([namesigs[s] for s in sigs],dtype=float))
    logp-=logp.max();weights=np.exp(logp);weights/=weights.sum()
    elements=sorted({t[8:] for r in original.R for t in r['conditions'] if t.startswith('element:')})
    rankings=[]
    for el in elements:
        prob=sum(weights[i] for i,s in enumerate(sigs) if 'element:'+el in s)
        rankings.append((el,float(prob)))
    rankings.sort(key=lambda q:-q[1]);ix=np.argsort(-weights)[:5]
    return {'name':name,'element_marginals':[{'element':e,'relative_weight':round(v,4)} for e,v in rankings[:10]],
      'top_5_training_condition_templates':[{'conditions':list(sigs[i]),'relative_weight':round(float(weights[i]),4)} for i in ix],
      'notes':'Bayes-like reweighting across known *training* condition combinations only; no new weights. Scores depend strongly on the template prior and are not calibrated.'}

if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--temperature',type=float,default=1);a=p.parse_args();print(json.dumps(inverse(a.name,temperature=a.temperature),indent=2))
