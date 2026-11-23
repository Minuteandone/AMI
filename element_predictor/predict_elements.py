"""Predict 46 element labels from a monster name alone.
Example: python predict_elements.py Bumbleler --threshold .35
"""
import argparse,json,pathlib,sys
import numpy as np,torch
from train_elements import CNN,ROOT,C2I,CHAR,LABELS,CORE

def load_model():
    q=torch.load(ROOT/'models'/'elements_cnn.pt',map_location='cpu',weights_only=True)
    # Uses matching architecture; save metadata guards against accidental mismatch
    assert q['labels']==LABELS and q['chars']==CHAR
    model=CNN();model.load_state_dict(q['state']);model.eval();return model

@torch.no_grad()
def predict(name,model=None,threshold=.35):
    model=model or load_model()
    from train_elements import L
    letters=[C2I.get(c,1) for c in name]
    x=torch.tensor([letters+[0]*max(0,L-len(letters))],dtype=torch.long)
    z,_=model(x)
    p=torch.sigmoid(z)[0].cpu().numpy()
    ranked=sorted(zip(LABELS,p),key=lambda v:-v[1]);on=[e for e,v in ranked if v>=threshold]
    return {'name':name,'threshold':threshold,'on':on,'top10':[{'element':n,'score':round(float(v),4)} for n,v in ranked[:10]],
            'all_elements':{n:round(float(v),6) for n,v in ranked},
            'notes':'Scores are uncalibrated; element labels include associated elements and monster-class special elements.'}

if __name__=='__main__':
    torch.set_num_threads(1)
    a=argparse.ArgumentParser();a.add_argument('name');a.add_argument('--threshold',type=float,default=.35);args=a.parse_args()
    print(json.dumps(predict(args.name,threshold=args.threshold),indent=2))
