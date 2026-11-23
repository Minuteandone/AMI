#!/usr/bin/env python3
"""Three-model chain: elements -> name -> islands and independently generated bios.
Bio models receive ONLY the generated monster name, not island or element labels.
"""
import argparse,json,sys
from pathlib import Path
from generate_bio import generate_bio
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'previous_models'))
from pipeline import chain

def run(elements,monster_class='Mythical',subclass=None,n=1,seed=17,temperature=.8,bio_seed=9):
    rows=chain(elements,monster_class,subclass,n,seed,temperature)
    for i,row in enumerate(rows):
        name=row['name']
        row['bios']={variant:generate_bio(name,variant=variant,seed=bio_seed+i,temperature=.8,top_k=25)
                      for variant in ('free','template')}
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('elements',nargs='+')
    ap.add_argument('--class',dest='monster_class',default='Mythical')
    ap.add_argument('--subclass',default=None)
    ap.add_argument('--n',type=int,default=1);ap.add_argument('--seed',type=int,default=17)
    ap.add_argument('--bio-seed',type=int,default=9)
    args=ap.parse_args()
    print(json.dumps(run(args.elements,args.monster_class,args.subclass,args.n,args.seed,bio_seed=args.bio_seed),ensure_ascii=False,indent=2))
