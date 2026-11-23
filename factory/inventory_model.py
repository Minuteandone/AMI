"""Small data-trained probabilistic inventory generator for AMI Island Factory.

NOT a language model. Fits empirical egg selection frequencies and inventory-size /
quantity distributions to real Wublin, Celestial and Amber examples, then samples
new egg inventories from those statistics. No hand-authored required-egg recipes.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import random

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data'
STYLES = {'wublin': 'W', 'celestial': 'C', 'amber': 'A'}


def read_references():
    names=json.loads((DATA/'egg_names.json').read_text(encoding='utf-8'))
    recipes=[]
    for lineno,line in enumerate((DATA/'reference_inventories.txt').read_text(encoding='utf-8').splitlines(),1):
        if not line.strip() or line.startswith('#'):continue
        try:
            code,target,raw=line.strip().split('|')
            eggs={key:int(v) for key,v in (pair.split('x') for pair in raw.split(','))}
        except Exception as exc:
            raise ValueError(f'Bad source row {lineno}: {line}') from exc
        if not eggs or any(q<1 for q in eggs.values()) or any(k not in names for k in eggs):
            raise ValueError(f'Invalid egg id/count in row {lineno}: {line}')
        recipes.append({'style':code,'target':target,'eggs':eggs})
    return recipes,names


def fit_model():
    recipes,names=read_references()
    learned={}
    for style,code in STYLES.items():
        examples=[r for r in recipes if r['style']==code]
        occurrences=Counter()
        counts=defaultdict(list)
        profiles=[]
        for example in examples:
            for egg,qty in example['eggs'].items():
                occurrences[egg]+=1
                counts[egg].append(qty)
            profiles.append({'source_target':example['target'],
                             'distinct_eggs':len(example['eggs']),
                             'total_eggs':sum(example['eggs'].values()),
                             'quantity_pattern':list(example['eggs'].values())})
        learned[style]={'training_examples':len(examples),'egg_frequency':dict(occurrences),
                        'egg_quantity_examples':dict(counts),'profiles':profiles}
    return {'model_type':'empirical_conditional_inventory_sampler_v1','source_records':len(recipes),
            'styles':learned,'egg_names':names,
            'source':'getinski/msm-breeding-planner game_data.json, MIT, commit 53aa0df3979e466932095506980560e4b019f6f3',
            'method':'bootstrap recipe shape + weighted egg sampling + empirical per-egg and profile counts'}


def weighted_sample_without_replacement(rng,choices,weights,n):
    # Exponential race weighted sampling; no repeated egg types in the final inventory.
    ranked=[]
    for value,weight in zip(choices,weights):
        u=max(rng.random(),1e-12)
        key=-math.log(u)/max(weight,1e-6)
        ranked.append((key,value))
    ranked.sort()
    return [value for _,value in ranked[:n]]


def generate_inventory(name,island,style='celestial',seed=20261122,model=None,intensity=1.0):
    """Generated fantasy inventory. Uses official EGGS, but not a real game's recipe."""
    if model is None:
        path=DATA/'inventory_model.json'
        model=json.loads(path.read_text()) if path.exists() else fit_model()
    if style not in STYLES:raise ValueError(f'Unknown style {style}')
    if not (0.25<=intensity<=2.0):raise ValueError('intensity needs to be 0.25–2.0')
    h=hashlib.blake2b(f'{seed}|{island}|{name}|{style}'.encode('utf-8'),digest_size=8).digest()
    rng=random.Random(int.from_bytes(h,'big'))
    params=model['styles'][style]
    profile=rng.choice(params['profiles'])
    n=profile['distinct_eggs']
    # Small random recipe-shape changes, while respecting the training distribution.
    n=max(2,min(len(params['egg_frequency']),n+rng.choice([-2,-1,0,0,0,1,2])))
    frequencies=params['egg_frequency']
    egg_ids=list(frequencies)
    # Smooth observed appearance frequencies: existing rare egg types remain possible.
    weights=[float(frequencies[e])+.2 for e in egg_ids]
    picks=weighted_sample_without_replacement(rng,egg_ids,weights,n)
    source_quantities=list(profile['quantity_pattern'])
    rng.shuffle(source_quantities)
    values=[]
    for index,egg in enumerate(picks):
        local=params['egg_quantity_examples'].get(egg,[])
        # One part egg-specific learned count, two parts reference-shape count.
        if local and rng.random()<.35:quantity=rng.choice(local)
        else:quantity=source_quantities[index % len(source_quantities)]
        # All eggs have positive quantities. A variation of 0.75x–1.25x avoids copying a source exactly.
        quantity=max(1,int(round(quantity*rng.uniform(.75,1.25)*intensity)))
        values.append((egg,quantity))
    values.sort(key=lambda x:(-x[1],model['egg_names'][x[0]]))
    entries=[{'egg':model['egg_names'][egg], 'egg_id':int(egg),'required':qty} for egg,qty in values]
    return {'monster':name,'style':style,'egg_types':len(entries),
            'total_eggs':sum(x['required'] for x in entries),'requirements':entries,
            'training_reference_style':style,'generated':True,
            'provenance':'statistically sampled from real-game-reference inventory patterns; NOT an official recipe'}


def for_island(island,seed=None,style=None,intensity=1.0,model=None):
    if island['primary_type']!='zapping':return {}
    if style is None:
        style='celestial' if island.get('zap_generation_style')=='Celestial' else 'wublin'
    if seed is None:seed=island.get('generation_seed',20261122)
    return {row['name']:generate_inventory(row['name'],island['island_name'],style,seed,model,intensity)
            for row in island['monster_list']}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--train',action='store_true');p.add_argument('--name',default='Fop')
    p.add_argument('--style',choices=STYLES,default='celestial');p.add_argument('--seed',type=int,default=20261122)
    a=p.parse_args()
    model=fit_model()
    if a.train:
        (DATA/'inventory_model.json').write_text(json.dumps(model,indent=2,ensure_ascii=False))
        print('Saved learned model to',DATA/'inventory_model.json')
    else:print(json.dumps(generate_inventory(a.name,'Hoax Cosmos',a.style,a.seed,model),indent=2))
