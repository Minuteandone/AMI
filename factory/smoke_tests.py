"""Fast invariant checks. Assumes train_models.py has been executed once."""
import json,math
from generate_island import Factory,LOOK,DATA
f=Factory(seed=43,temperature=.83)
assert len(json.load(open('island_corpus.json')))==41
assert len(json.load(open('basic_elements.json')))==20
assert len(json.load(open('gimmicks.json')))>=10
# 4 basic elements produce 2^4 - 1 slots, regardless of how many combinations exist.
b=f.create(kind='breeding',elements=['Plant','Cold','Fire','Bone'])
assert b['core_breeding_slots']==15 and b['primary_type']=='breeding'
assert all(m['source'] in ('existing_official_element_combo','generated_missing_combo','gimmick_existing_guest') for m in b['monster_list'])
assert any(x['name']=='Potbelly' and x['source']=='existing_official_element_combo' for x in b['monster_list'])
assert any(x['source']=='generated_missing_combo' for x in b['monster_list'])
assert len({m['name'] for m in b['monster_list']})==len(b['monster_list'])
assert f.gimmicks_for_name('Light Island')[0] is not None
for name in ['Magical Carnival','Shugabush Island']:
 gimmicks,scores=f.gimmicks_for_name(name)
 assert isinstance(scores,dict)
t=f.create(kind='teleporting');assert t['seed_elements']==[] and t['monster_count']>0
assert any(m['source']=='existing_via_pair_model' for m in t['monster_list'])
assert any(m['source']=='generated_via_pair_model' for m in t['monster_list'])
z=f.create(kind='zapping');assert len(z['seed_elements'])==1 and 6<=z['monster_count']<=14
assert all(m['source']=='generated_only' for m in z['monster_list'])
assert all(m['name'] not in {r['name'] for r in DATA} for m in z['monster_list'])
assert f.gimmicks_for_name('test')[0] is not None
print('All invariant tests passed')
