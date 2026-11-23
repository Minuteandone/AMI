"""Small statistical island-name baseline with NO neural weights.
Evaluation uses the same Mirror-family grouped folds as the GRU.
The baseline conditions on island type only (elements are too sparse).
"""
import json,collections,random,math,pathlib,numpy as np
P=pathlib.Path(__file__).resolve().parent
R=[r for r in json.loads((P/'island_corpus.json').read_text()) if r['eligible_for_training']]
abc=sorted(set(''.join(r['name'] for r in R)+'\x03'))
ids=sorted({r['group'] for r in R});random.Random(12).shuffle(ids);groups={g:i%4 for i,g in enumerate(ids)}

def train(rows,n=3):
 m=collections.defaultdict(collections.Counter);un=collections.defaultdict(collections.Counter)
 for r in rows:
  seq='^'*(n-1)+r['name']+'\x03'
  for i in range(n-1,len(seq)):
   m[(r['type'],seq[i-n+1:i])][seq[i]]+=1
   un[r['type']][seq[i]]+=1
 return m,un

def prob(m,un,kind,ctx,ch,alpha=1.):
 z=m.get((kind,ctx),collections.Counter());u=un[kind]
 # type-conditioned unigram acts as a smoothing prior; never divide by 0
 prior=(u[ch]+.5)/(sum(u.values())+.5*len(abc))
 return (z[ch]+alpha*prior)/(sum(z.values())+alpha)

def eval_rows(m,un,rows,n,alpha):
 loss=0;cnt=0
 for r in rows:
  seq='^'*(n-1)+r['name']+'\x03'
  for i in range(n-1,len(seq)):
   loss-=math.log(max(prob(m,un,r['type'],seq[i-n+1:i],seq[i],alpha),1e-12));cnt+=1
 return loss/cnt
res=[]
for n in [2,3,4]:
 for alpha in [.5,2.,8.]:
  losses=[]
  for fold in range(4):
   tr=[r for r in R if groups[r['group']]!=fold];va=[r for r in R if groups[r['group']]==fold]
   model,prior=train(tr,n);losses.append(eval_rows(model,prior,va,n,alpha))
  res.append({'n':n,'smoothing':alpha,'cv_heldout_nats_per_character':round(float(np.mean(losses)),4),'folds':[round(x,4) for x in losses]})
res.sort(key=lambda x:x['cv_heldout_nats_per_character'])
(P/'ngram_cv.json').write_text(json.dumps({'baseline_type_only':True,'results':res,'best':res[0]},indent=2))
print('NGRAM best',res[0])
