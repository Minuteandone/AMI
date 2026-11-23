from pathlib import Path
import sys,json
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'factory'))
from inventory_model import fit_model,for_island,generate_inventory,read_references

ref,names=read_references();model=fit_model()
assert len(ref)==62
assert model['source_records']==62
assert sum(sum(x['eggs'].values()) for x in ref)==3360
assert [model['styles'][x]['training_examples'] for x in ['wublin','celestial','amber']]==[19,12,31]
island=json.loads((BASE/'hoax_cosmos.json').read_text())
assert island['island_name']=='Hoax Cosmos'
assert island['original_generated_name']=='Fon Islao'
assert island['generation_seed']==20261122
assert len(island['monster_list'])==12
assert island['primary_type']=='zapping'
assert set(island['egg_inventories'])=={x['name'] for x in island['monster_list']}
assert island['egg_inventories']==for_island(island,model=model)
for name,r in island['egg_inventories'].items():
 assert r['style']=='celestial'
 assert r['monster']==name
 assert len(r['requirements'])==len({x['egg_id'] for x in r['requirements']})==r['egg_types']
 assert sum(x['required'] for x in r['requirements'])==r['total_eggs']
 assert all(x['required']>=1 and str(x['egg_id']) in names for x in r['requirements'])
 # Unofficial target names are not used as egg sources.
 assert not any(x['egg'] in {m['name'] for m in island['monster_list']} for x in r['requirements'])
 assert generate_inventory(name,island['island_name'],'celestial',20261122,model)==r
# Renaming the island or monster changes the seeded outputs (not a fixed recipe lookup).
assert generate_inventory('Fop','Another Cosmos',model=model)!=island['egg_inventories']['Fop']
assert generate_inventory('Fop','Hoax Cosmos',seed=1234,model=model)!=island['egg_inventories']['Fop']
print('All 62 reference inventories and 12 Hoax Cosmos inventories validate.')
