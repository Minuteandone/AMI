"""Structural tests for AMI Islands v4. Run after building and training."""
import json, pathlib, torch
from predict_islands import IslandPredictor
p=pathlib.Path(__file__).resolve().parent
a=json.loads((p/'placements.json').read_text());islands=json.loads((p/'islands.json').read_text());orig=json.loads((p/'provenance.json').read_text());names={r['name']:r for r in a}
excluded={'Cave Island','Cloud Island','Party Island','Space Island','Starhenge','The Continent','Wanderer Island'}
assert excluded.isdisjoint(islands), excluded & set(islands)
assert len(islands)==41 and len(a)==263
assert len(names['Tunefor']['islands'])>=1 and 'Poison Islet' in names['Tunefor']['islands']
assert [r['name'] for r in a if 'Poison Islet' in r['islands']].__len__()==14
assert 'Poison Islet' in names['Oogiddy']['islands']
assert all('Poison Islet' not in names[x]['islands'] for x in ['Pentumbra','Rhysmuth','Vhenshun'])
assert names['Superno-Fa']['islands']==['Celestial Island']
assert names['Sol-stice']['islands']==['Celestial Island']
for special,original in orig['dipster_inheritance'].items():
    assert names[special]['islands']==names[original]['islands'],special
assert all(r['islands'] for r in a), 'Unexpected monster with zero labels'
m=IslandPredictor();assert m.names==islands
pred=m.predict('Gadla')[0];assert set(pred['on_off'])==set(islands)
print('All v4 main-game checks passed: 263 monsters; 41 islands; 14 predicted Poison Islet residents')
