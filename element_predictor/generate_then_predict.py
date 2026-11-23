"""Two-stage demo. Stage 2 gets name ONLY; stage 1's conditions are printed only for comparison.
Usage: python generate_then_predict.py Fire Poison --class Mythical --n 5 --seed 123
"""
import sys,argparse,json,torch
from predict_elements import predict,load_model
from train_elements import ROOT
sys.path.insert(0,str(ROOT/'original'/'ami_pure'))
from generate import generate
from invert_generator import load as load_original

def main():
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('elements',nargs='+');p.add_argument('--class',dest='class_',default='Mythical');p.add_argument('--subclass',default='');p.add_argument('--n',type=int,default=5);p.add_argument('--temperature',type=float,default=.8);p.add_argument('--seed',type=int,default=123);p.add_argument('--threshold',type=float,default=.35);a=p.parse_args()
    import train as original
    tags=['class:'+a.class_]+(['subclass:'+a.subclass] if a.subclass else [])+['element:'+x for x in a.elements]
    bad=[t for t in tags if t not in original.T]
    if bad:p.error('Unknown conditions: '+','.join(bad))
    gen=load_original();classifier=load_model()
    names=generate(gen,tags,n=a.n,seed=a.seed,temp=a.temperature)
    out=[]
    for name in names:
        res=predict(name,model=classifier,threshold=a.threshold)
        out.append({'original_generation_input':tags,'name':name,'predicted_elements_on':res['on'],'top5':res['top10'][:5]})
    print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
