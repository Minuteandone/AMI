"""Exploratory out-of-fold cardinality baselines; hyperparameters not independently tuned."""
import collections,json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from train_elements import NAMES,FOLDS
from count_aware import YC,YS

N=len(NAMES);tc=YC.numpy();ts=YS.numpy()

def cv_knn(k,alpha,pow):
    pc=np.zeros(N,dtype=int);ps=np.zeros(N,dtype=int)
    for holdout in FOLDS:
        tr=sorted(set(range(N))-set(holdout));v=TfidfVectorizer(analyzer='char',ngram_range=(2,4),lowercase=False,sublinear_tf=True)
        a=v.fit_transform([NAMES[i] for i in tr]);b=v.transform([NAMES[i] for i in holdout]);similar=cosine_similarity(b,a)
        for j,i in enumerate(holdout):
            top=np.argsort(-similar[j])[:k]
            w=np.maximum(similar[j,top],1e-4)**pow;w/=w.sum()
            bc=np.bincount(tc[np.array(tr)[top]],weights=w,minlength=6)
            bs=np.bincount(ts[np.array(tr)[top]],weights=w,minlength=3)
            bc=(1-alpha)*bc+alpha*np.bincount(tc[tr],minlength=6)/len(tr)
            bs=(1-alpha)*bs+alpha*np.bincount(ts[tr],minlength=3)/len(tr)
            pc[i]=bc.argmax();ps[i]=bs.argmax()
    return pc,ps

def cv_logistic(C=1.0,features=(2,5)):
    pc=np.zeros(N,dtype=int);ps=np.zeros(N,dtype=int)
    for holdout in FOLDS:
        tr=sorted(set(range(N))-set(holdout));v=TfidfVectorizer(analyzer='char',ngram_range=features,lowercase=False,sublinear_tf=True)
        a=v.fit_transform([NAMES[i] for i in tr]);b=v.transform([NAMES[i] for i in holdout]);
        m=LogisticRegression(C=C,max_iter=300,solver='lbfgs');m.fit(a,tc[tr]);pc[holdout]=m.predict(b)
        m=LogisticRegression(C=C,max_iter=300,solver='lbfgs');m.fit(a,ts[tr]);ps[holdout]=m.predict(b)
    return pc,ps

def report(pc,ps):
 return { 'basic_count_accuracy':round(float((pc==tc).mean()),4),
          'special_count_accuracy':round(float((ps==ts).mean()),4),
          'total_count_accuracy':round(float((pc+ps==tc+ts).mean()),4),
          'joint_count_accuracy':round(float(((pc==tc)&(ps==ts)).mean()),4)}

out=[]
for c in [.01,.1,1,10,100,1000]:
 p,q=cv_logistic(c);out.append({'method':'logreg','C':c,**report(p,q)})
for k in [1,3,5,9,15]:
 for alpha in [0,.2,.5]:
  for power in [1,2,4]:
   p,q=cv_knn(k,alpha,power);out.append({'method':'knn','k':k,'alpha':alpha,'power':power,**report(p,q)})
out.sort(key=lambda r:(r['total_count_accuracy'],r['joint_count_accuracy']),reverse=True)
print('top methods:',*out[:20],sep='\n')
json.dump(out,open('count_method_exploration.json','w'),indent=2)
