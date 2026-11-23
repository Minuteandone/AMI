"""AMI: elemental conditions -> NAME GENERATOR -> NAME-ONLY island classifier.
The island classifier does not receive or load elements, class or subclass.
"""
import sys,json,pathlib,argparse,torch
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P/'ami_pure'))
from train import Net, CONFIG, CH, T
from generate import generate
from predict_islands import IslandPredictor

def make_name_model():
    raw=torch.load(P/'solstice_corrected.pt',map_location='cpu',weights_only=True)
    m=Net(CONFIG['contrast'],[])
    m.load_state_dict(raw['state']);m.eval()
    return m

def names_from_elements(elements,monster_class='Mythical',subclass=None,n=6,seed=1,temp=.9):
    # Only model 1 sees the conditions. The return value is names, NOT elements.
    tags=['class:'+monster_class]
    for element in elements:tags.append('element:'+element)
    intrinsic={'Mythical':['Mythical'],'Celestial':['Celestial'],'Legendary':['Legendary'],'Dipster':['Dipster'],'Supernatural':['Electricity']}
    tags.extend('element:'+s for s in intrinsic.get(monster_class,[]))
    if subclass:
        tags.append('subclass:'+subclass)
        if subclass=='Dreamythical':tags.append('element:Dream')
    invalid=sorted(set(tags)-set(T))
    if invalid:raise ValueError(f'Unsupported name-generator conditions: {invalid}')
    return generate(make_name_model(),tags,n=n,seed=seed,temp=temp)

def chain(elements,monster_class='Mythical',subclass=None,n=6,seed=1,temp=.9,threshold=None):
    names=names_from_elements(elements,monster_class,subclass,n,seed,temp)
    # Important: the next function receives only `names`.
    return IslandPredictor().predict(names,threshold=threshold)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('elements',nargs='+');p.add_argument('--class',dest='monster_class',default='Mythical');p.add_argument('--subclass',default=None);p.add_argument('--n',type=int,default=6);p.add_argument('--seed',type=int,default=1);p.add_argument('--temperature',type=float,default=.9);p.add_argument('--threshold',type=float,default=None);a=p.parse_args()
    print(json.dumps(chain(a.elements,a.monster_class,a.subclass,a.n,a.seed,a.temperature,a.threshold),indent=2,ensure_ascii=False))
