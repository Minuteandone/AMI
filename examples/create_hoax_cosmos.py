"""Upgrade the existing first island, preserving its original model outputs."""
from pathlib import Path
import json,csv,sys
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'factory'))
from inventory_model import for_island,fit_model

original=json.loads((BASE/'data/original_fon_islao.json').read_text())
assert original['island_name']=='Fon Islao' and original['primary_type']=='zapping' and len(original['monster_list'])==12
hoax=original.copy()
hoax['original_generated_name']=hoax['island_name']
hoax['island_name']='Hoax Cosmos'
hoax['name_origin']='Named by the user; original model output retained in original_generated_name'
hoax['egg_inventories']=for_island(hoax,seed=20261122,model=fit_model())
hoax['egg_inventory_generator']='empirical_conditional_inventory_sampler_v1'
hoax['inventory_status']='generated_fictional_not_official_requirements'
(BASE/'hoax_cosmos.json').write_text(json.dumps(hoax,indent=2,ensure_ascii=False)+'\n')
rows=[]
for monster,recipe in hoax['egg_inventories'].items():
    for item in recipe['requirements']:
        rows.append({'island':hoax['island_name'],'monster':monster,'style':recipe['style'],
                     'egg':item['egg'],'egg_id':item['egg_id'],'quantity':item['required']})
with (BASE/'hoax_cosmos_inventories.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['island','monster','style','egg','egg_id','quantity'])
    writer.writeheader();writer.writerows(rows)
print(f"{hoax['island_name']}: {len(hoax['monster_list'])} monsters, "+
      f"{sum(x['total_eggs'] for x in hoax['egg_inventories'].values())} total requested eggs")
for name,inventory in hoax['egg_inventories'].items():
    top=', '.join(f"{r['required']}× {r['egg']}" for r in inventory['requirements'][:3])
    print(f"{name:16s}: {inventory['total_eggs']:3d} eggs, {inventory['egg_types']:2d} types | {top}")
