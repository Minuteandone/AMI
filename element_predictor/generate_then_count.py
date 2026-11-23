"""Generate an AMI Pure monster name, then infer HOW MANY elements and WHICH elements.

The element model receives only the generated name.  Example:
python generate_then_count.py Fire Poison --class Mythical --seed 5 --n 6
"""
import sys,argparse,json
import torch
from count_aware import inference,load as load_count
from train_elements import ROOT
sys.path.insert(0,str(ROOT/'original'/'ami_pure'))
from generate import generate
from invert_generator import load as load_name_generator

def main():
    torch.set_num_threads(1)
    p=argparse.ArgumentParser()
    p.add_argument('elements',nargs='+')
    p.add_argument('--class',dest='class_',default='Mythical')
    p.add_argument('--subclass',default='')
    p.add_argument('--n',type=int,default=6)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--temperature',type=float,default=.8)
    a=p.parse_args()
    import train as old
    cond=['class:'+a.class_]+(['subclass:'+a.subclass] if a.subclass else [])+['element:'+x for x in a.elements]
    unknown=[c for c in cond if c not in old.T]
    if unknown:p.error('Unrecognized conditions '+repr(unknown))
    if a.n<1 or a.n>500:p.error('n must be between 1 and 500')
    m=load_name_generator()
    preds=load_count()
    names=generate(m,cond,n=a.n,seed=a.seed,temp=a.temperature)
    result=[]
    for name in names:
        x=inference(name,preds)
        result.append({'name':name,'original_generator_conditions':cond,
                       'predicted_total_element_count':x['predicted_element_count'],
                       'predicted_basic_count':x['predicted_basic_count'],
                       'predicted_special_count':x['predicted_special_count'],
                       'predicted_elements':x['elements'],
                       'top10_scores':x['top10_elements']})
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
