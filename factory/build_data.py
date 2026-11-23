"""Create a reproducible experimental island corpus from the user's AMI artifacts.
Only released original-game islands are used as positive training rows; nine Mirror Islands
share labels with originals, so split by family during evaluation. Metadata is curated and
its assumptions are recorded; it is NOT an authoritative game-mechanics export.
"""
import json, pathlib, random, itertools, collections
P=pathlib.Path(__file__).resolve().parent
S=P/'source'
monsters=json.loads((S/'monsters.json').read_text())
placements=json.loads((S/'placements.json').read_text())
islands=json.loads((S/'islands.json').read_text())
look={m['name']:m for m in monsters}
base_els=['Plant','Cold','Air','Water','Earth','Fire','Light','Psychic','Faerie','Bone','Plasma','Shadow','Mech','Crystal','Poison','Control','Hoax','Ruin','Depths','Electricity']
# Canonically familiar element pools for the original breeding islands; manual map, not
# a dynamically extracted game database. Special-case unusual islands separately.
pools={
'Plant Island':['Plant','Cold','Water','Earth'],
'Cold Island':['Cold','Air','Water','Plant'],
'Air Island':['Air','Water','Earth','Cold'],
'Water Island':['Water','Plant','Earth','Air'],
'Earth Island':['Earth','Cold','Air','Plant'],
'Fire Haven':['Plant','Earth','Cold','Fire'],
'Fire Oasis':['Air','Water','Cold','Fire'],
'Light Island':['Plant','Cold','Fire','Light'],
'Psychic Island':['Water','Earth','Fire','Psychic'],
'Faerie Island':['Air','Cold','Fire','Faerie'],
'Bone Island':['Plant','Water','Fire','Bone'],
'Ethereal Island':['Plasma','Shadow','Mech','Crystal','Poison'],
'Magical Sanctum':['Light','Psychic','Faerie','Bone'],
'Mythical Island':['Mythical','Dream'],
'Seasonal Shanty':['Spooktacle','Festival of Yay','Season of Love','Eggs-Travaganza','SummerSong'],
'Shugabush Island':['Plant','Legendary'],
'bbli$zard Island':['Cold','Legendary'],
}
zapping={'Wublin Island':'Electricity','Celestial Island':'Celestial','Amber Island':'Fire'}
teleport={'Magical Nexus','Ethereal Workshop','Paironormal Carnival','Plasma Islet','Shadow Islet','Mech Islet','Crystal Islet','Poison Islet','Gold Island','Tribal Island','Composer Island','The Colossingum'}
# These manual labels approximate playable mechanics. Some original islands have several
# methods of obtaining monsters (teleport + breeding, synthesis etc); one PRIMARY type is
# assigned only because the generator user interface requires exactly three categories.
secondary={'Ethereal Island':['teleporting'], 'Mythical Island':['teleporting'],
 'Magical Sanctum':['teleporting'], 'Seasonal Shanty':['teleporting'],
 'Shugabush Island':['teleporting'], 'Ethereal Workshop':['synthesis'],
 'Paironormal Carnival':['transposition'], 'Magical Nexus':['transposition'],
 'Plasma Islet':['teleporting','dish_harmonization'], 'Shadow Islet':['teleporting','dish_harmonization'],
 'Mech Islet':['teleporting','dish_harmonization'], 'Crystal Islet':['teleporting','dish_harmonization'],
 'Poison Islet':['teleporting','dish_harmonization'], 'The Colossingum':['level_transfer']}
# Gimmick names are derived from relevant species classes/subclasses in each island's
# placement roster, then supplemented with an island-variant flag. These are features
# present in the roster, not necessarily the island's unique headline mechanic.
gmap={
'Dipsters':lambda m:m['monster_class']=='Dipster',
'Wubboxes':lambda m:m['name']=='Wubbox',
'Mythicals':lambda m:m['monster_class']=='Mythical',
'Werdos':lambda m:m['subclass']=='Werdo',
'Legendaries':lambda m:m['monster_class']=='Legendary',
'Shugafam':lambda m:m['subclass']=='Shugafam',
'bb$quad':lambda m:m['subclass']=='bb$quad',
'Paironormals':lambda m:m['monster_class']=='Paironormal',
'Titansouls':lambda m:m['monster_class']=='Titansoul',
'Seasonals':lambda m:m['monster_class']=='Seasonal',
'Ethereals':lambda m:m['monster_class']=='Ethereal',
'Magicals':lambda m:m['monster_class']=='Magical',
'Fire monsters':lambda m:m['monster_class']=='Fire',
'Primordials':lambda m:m['monster_class']=='Primordial',
'Celestials':lambda m:m['monster_class']=='Celestial',
'Wublins':lambda m:m['subclass']=='Wublin',
}
GIMMICKS=sorted(list(gmap)+['Mirror forms'])
rows=[]
for island in islands:
 mirror=island.startswith('Mirror ')
 family=island.removeprefix('Mirror ')
 hypothetical=island=='Poison Islet'
 original=island
 kind='zapping' if family in zapping else 'teleporting' if family in teleport else 'breeding'
 els=pools.get(family,[])
 if kind=='zapping':els=[zapping[family]]
 rost=[x['name'] for x in placements if island in x['islands']]
 gimmicks=sorted([tag for tag,pred in gmap.items() if any(pred(look[n]) for n in rost)])
 if mirror:gimmicks.append('Mirror forms')
 if 'Paironormals' in gimmicks and 'Mirror forms' not in gimmicks:gimmicks.append('Mirror forms')
 rows.append(dict(name=island,group=family,type=kind,elements=els,roster=rost,gimmicks=sorted(gimmicks),
    secondary_mechanics=secondary.get(family,[]),mirror=mirror,status='projected' if hypothetical else 'source-backed-with-curation',
    eligible_for_training=not hypothetical))
assert len(rows)==41
assert {x['name'] for x in rows}==set(islands)
assert all(x['type'] in ['breeding','teleporting','zapping'] for x in rows)
assert [x['name'] for x in rows if x['type']=='zapping']==['Amber Island','Celestial Island','Wublin Island']
# Original non-mirror islands carry unique training evidence. Mirrored names are useful for
# learned spellings but should be downweighted and never split apart from their originals.
json.dump(rows,open(P/'island_corpus.json','w'),indent=2,ensure_ascii=False)
json.dump(GIMMICKS,open(P/'gimmicks.json','w'),indent=2)
json.dump(base_els,open(P/'basic_elements.json','w'),indent=2)
from csv import DictWriter
with open(P/'island_corpus.csv','w') as f:
 w=DictWriter(f,['name','type','elements','gimmicks','roster_size','status','mirror','secondary_mechanics']);w.writeheader()
 for r in rows:w.writerow({k:(' | '.join(r[k]) if isinstance(r[k],list) else r[k]) for k in ['name','type','elements','gimmicks','status','mirror','secondary_mechanics']}|{'roster_size':len(r['roster'])})
print('Built',len(rows),'rows:',collections.Counter(r['type'] for r in rows),'training rows:',sum(r['eligible_for_training'] for r in rows))
