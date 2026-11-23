"""Predict MSM island on/off states using ONLY a monster name. No elements, class or subclass inputs."""
import json, pathlib, argparse
import torch
from torch import nn
import torch.nn.functional as F
P=pathlib.Path(__file__).resolve().parent

class IslandsCNN(nn.Module):
    def __init__(self,vocab,num_islands,emb=24,width=32,hidden=80,drop=.35):
        super().__init__()
        self.embed=nn.Embedding(vocab,emb,padding_idx=0)
        self.convs=nn.ModuleList([nn.Conv1d(emb,width,k,padding='same') for k in (2,3,4,5)])
        self.norm=nn.LayerNorm(width*8)
        self.fc=nn.Sequential(nn.Linear(width*8,hidden),nn.GELU(),nn.Dropout(drop),nn.Linear(hidden,num_islands))
    def forward(self,x):
        e=self.embed(x).transpose(1,2)
        feat=[]
        for conv in self.convs:
            h=F.gelu(conv(e));feat.extend([h.max(2).values,h.mean(2)])
        return self.fc(self.norm(torch.cat(feat,1)))

class IslandPredictor:
    def __init__(self,path=P/'island_model.pt'):
        torch.set_num_threads(1)
        ck=torch.load(path,map_location='cpu',weights_only=True)
        self.names=ck['islands'];self.chars=ck['chars'];self.lookup={c:i for i,c in enumerate(ck['chars'])};self.maxlen=ck['maxlen'];self.threshold=ck['threshold']
        self.model=IslandsCNN(len(self.chars),len(self.names));self.model.load_state_dict(ck['state']);self.model.eval()
    @torch.inference_mode()
    def predict(self,monster_names,threshold=None):
        if isinstance(monster_names,str):monster_names=[monster_names]
        thr=self.threshold if threshold is None else threshold
        x=torch.tensor([[self.lookup.get(c,1) for c in name[:self.maxlen]]+[0]*max(0,self.maxlen-len(name)) for name in monster_names],dtype=torch.long)
        p=torch.sigmoid(self.model(x)).numpy()
        return [dict(name=name,threshold=thr,islands_on=[i for i,v in zip(self.names,prob) if float(v)>=thr],
           on_off={i:bool(float(v)>=thr) for i,v in zip(self.names,prob)},
           probabilities={i:round(float(v),4) for i,v in zip(self.names,prob)},
           top_ranked=[{'island':self.names[j],'probability':round(float(prob[j]),4)} for j in prob.argsort()[::-1][:8]])
          for name,prob in zip(monster_names,p)]

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('names',nargs='+');parser.add_argument('--threshold',type=float,default=None);a=parser.parse_args()
    print(json.dumps(IslandPredictor().predict(a.names,threshold=a.threshold),indent=2,ensure_ascii=False))
