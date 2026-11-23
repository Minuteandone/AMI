from pathlib import Path
import shutil,json
src=Path('/mnt/data/ami_islands_v3');dst=Path('/mnt/data/ami_islands_v4_main_game')
dst.mkdir(exist_ok=True)
files=['README.md','build_labels.py','train_islands.py','predict_islands.py','pipeline.py','smoke_test.py','monsters.json','solstice_corrected.pt']
for f in files: shutil.copy2(src/f,dst/f)
shutil.copytree(src/'ami_pure',dst/'ami_pure',dirs_exist_ok=True)
excluded=['Cave Island','Cloud Island','Party Island','Space Island','Starhenge','The Continent','Wanderer Island']
p=dst/'build_labels.py';s=p.read_text()
s=s.replace('for line in RAW.splitlines():\n        island,raw_names=line.split(\'|\',1)',
            'for line in RAW.splitlines():\n        island,raw_names=line.split(\'|\',1)\n        if island in DOF_ONLY_ISLANDS:\n            continue')
s=s.replace("ISLET_NAME='Poison Islet'", "ISLET_NAME='Poison Islet'\n# Standalone Dawn of Fire islands, not present in the original game.\nDOF_ONLY_ISLANDS="+repr(set(excluded)))
s=s.replace("'version':'v3'","'version':'v4-main-game'")
s=s.replace("'scope':'MSM + Dawn of Fire, nine mirror targets copied from base island labels'", "'scope':'Original My Singing Monsters only; 7 Dawn of Fire-only islands excluded; nine mirror targets copied from base island labels'")
s=s.replace("'changes':['Added Special Dipsters by matching base Dipster island rosters'", "'changes':['Excluded DoF-only islands: '+', '.join(sorted(DOF_ONLY_ISLANDS)), 'Added Special Dipsters by matching base Dipster island rosters'")
s=s.replace("'original_community_roster_transcription':str(SOURCE)","'original_community_roster_transcription':str(SOURCE), 'excluded_dawn_of_fire_islands':sorted(DOF_ONLY_ISLANDS)")
p.write_text(s)
# project training README and docs follow source
for filename in ['train_islands.py','pipeline.py','predict_islands.py']:
    p=dst/filename;s=p.read_text();s=s.replace('AMI Islands:', 'AMI Islands main-game:');p.write_text(s)
# create explicit tests independent of previous stale smoke test
p=dst/'smoke_test.py'
p.write_text('''"""Structural tests for AMI Islands v4. Run after building and training."""\nimport json, pathlib, torch\nfrom predict_islands import IslandPredictor\np=pathlib.Path(__file__).resolve().parent\na=json.loads((p/'placements.json').read_text());islands=json.loads((p/'islands.json').read_text());orig=json.loads((p/'provenance.json').read_text());names={r['name']:r for r in a}\nexcluded={'Cave Island','Cloud Island','Party Island','Space Island','Starhenge','The Continent','Wanderer Island'}\nassert excluded.isdisjoint(islands), excluded & set(islands)\nassert len(islands)==41 and len(a)==263\nassert len(names['Tunefor']['islands'])>=1 and 'Poison Islet' in names['Tunefor']['islands']\nassert [r['name'] for r in a if 'Poison Islet' in r['islands']].__len__()==14\nassert 'Poison Islet' in names['Oogiddy']['islands']\nassert all('Poison Islet' not in names[x]['islands'] for x in ['Pentumbra','Rhysmuth','Vhenshun'])\nassert names['Superno-Fa']['islands']==['Celestial Island']\nassert names['Sol-stice']['islands']==['Celestial Island']\nfor special,original in orig['dipster_inheritance'].items():\n    assert names[special]['islands']==names[original]['islands'],special\nassert all(r['islands'] for r in a), 'Unexpected monster with zero labels'\nm=IslandPredictor();assert m.names==islands\npred=m.predict('Gadla')[0];assert set(pred['on_off'])==set(islands)\nprint('All v4 main-game checks passed: 263 monsters; 41 islands; 14 predicted Poison Islet residents')\n''')
print('Created',dst,'excluding',excluded)
