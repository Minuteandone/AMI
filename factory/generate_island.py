"""Generate complete experimental MSM islands from small models and transparent rules.
Flow: sample type -> optional element(s) -> small name RNN -> gimmick classifier
-> deterministic breed / name-pair teleport classifier / generated-only zapping roster.

Existing official monsters are matched by exact, canonically defined official element sets.
All generated names are raw AMI Pure outputs, with no manually written additions.
"""
import os,sys,random,json,math,pathlib,argparse,itertools,collections
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import torch,joblib,numpy as np
from torch.nn import functional as F
from train_models import NameGRU
from inventory_model import for_island
P=pathlib.Path(__file__).resolve().parent
DATA=json.loads((P/'source/monsters.json').read_text());OFFICIAL={x['name'] for x in DATA}
NAME_R=next((P/'models').glob('island_name_gru.pt'))
BASIC=json.loads((P/'basic_elements.json').read_text())
GIMMICKS=json.loads((P/'gimmicks.json').read_text())
# Search all species by canonical official (not associated) elements. Eligibility limited
# to monster elemental classes, otherwise e.g. a Celestial's associated Plant would
# incorrectly be treated as a Plant single-element breeder.
BREEDABLE_CLASSES={'Natural','Fire','Magical','Ethereal','Paironormal'}
LOOK={tuple(sorted(r['official_elements'])):[] for r in DATA if r['monster_class'] in BREEDABLE_CLASSES}
for r in DATA:
 if r['monster_class'] in BREEDABLE_CLASSES:
  LOOK.setdefault(tuple(sorted(r['official_elements'])),[]).append(r)
TYPES=['breeding','teleporting','zapping']

class Factory:
 def __init__(self,seed=12,temperature=.85):
  torch.set_num_threads(1)
  self.rng=random.Random(seed);self.temperature=temperature;self.seed=seed
  ck=torch.load(NAME_R,map_location='cpu',weights_only=True)
  self.chars=ck['chars'];self.els=ck['elements'];self.types=ck['types'];self.namemodel=NameGRU(len(self.chars),len(self.types)+len(self.els))
  self.namemodel.load_state_dict(ck['state']);self.namemodel.eval()
  self.gimmickmodel=joblib.load(P/'models/gimmicks.joblib')
  self.teleportmodel=joblib.load(P/'models/teleport_pair.joblib')
  sys.path.insert(0,str(P/'source/ami_pure'))
  from train import Net,CONFIG,T
  from generate import generate
  self.pure_generate=generate;self.monster_tags=T
  raw=torch.load(P/'source/solstice_corrected.pt',map_location='cpu',weights_only=True)
  self.monstermodel=Net(CONFIG['contrast'],[])
  self.monstermodel.load_state_dict(raw['state']);self.monstermodel.eval()
  self.used=set()
 def name_for_island(self,kind,elements,temp=None):
  tmp=self.temperature if temp is None else temp
  cond=torch.zeros(1,len(self.types)+len(self.els));cond[0,self.types.index(kind)]=1
  for e in elements:
   if e in self.els:cond[0,len(self.types)+self.els.index(e)]=1
  tries=[]
  for attempt in range(75):
   g=torch.Generator().manual_seed(self.rng.randrange(2**31));x=torch.tensor([[1]])
   out=''
   for i in range(36):
    with torch.no_grad():log=self.namemodel(x,cond)[:,-1]/max(tmp,.05)
    log[:,0]=-1e8;log[:,1]=-1e8
    if len(out)<3:log[:,2]=-1e8
    idx=int(torch.multinomial(log.softmax(-1),1,generator=g))
    if idx==2:break
    out+=self.chars[idx];x=torch.cat([x,torch.tensor([[idx]])],1)
   tries.append(out)
   # No additions or spelling corrections. Do not simply replay an existing island.
   if 5<=len(out)<=28 and out not in self.used and out not in {r['name'] for r in json.loads((P/'island_corpus.json').read_text())} and sum(ch.isalpha() for ch in out)/len(out)>.77:
    self.used.add(out);return out,{'attempts':attempt+1,'raw_candidates_first_12':tries[:12],'backend':'tiny conditional GRU'}
  # The unfiltered fallback is STILL a true model output, never a fabricated island name.
  chosen=max(tries,key=lambda x:(sum(ch.isalpha() for ch in x),-abs(len(x)-12)))
  self.used.add(chosen)
  return chosen,{'attempts':75,'raw_candidates_first_12':tries[:12],'backend':'tiny conditional GRU','fallback':'longest-readable-sample'}
 def gimmicks_for_name(self,name):
  ck=self.gimmickmodel;v=ck['vectorizer'].transform([name]).toarray();pr=np.clip(ck['regression'].predict(v)[0],0,1)
  scores=dict(sorted(zip(ck['tags'],map(lambda x:round(float(x),4),pr)),key=lambda x:x[1],reverse=True))
  result=[g for g,p in scores.items() if p>=ck['threshold'] and g!='Mirror forms']
  # Global cap prevents short training corpus from choosing every gimmick at once.
  result=result[:5]
  if 'Paironormals' in result or scores.get('Mirror forms',0)>=.60:result.append('Mirror forms')
  return list(dict.fromkeys(result)),scores
 def new_monsters(self,elements,monster_class=None,subclass=None,n=1,temp=None):
  if monster_class is None:
   cls=set(elements)
   monster_class=('Paironormal' if cls&{'Control','Hoax','Ruin','Depths'} else
      'Ethereal' if cls&{'Plasma','Shadow','Mech','Crystal','Poison'} else
      'Magical' if cls&{'Light','Psychic','Faerie','Bone'} else
      'Fire' if 'Fire' in cls else 'Natural')
  else:monster_class=monster_class
  tags=['class:'+monster_class]+['element:'+e for e in elements]
  if monster_class=='Celestial':tags.append('element:Celestial')
  if monster_class=='Legendary':tags.append('element:Legendary')
  if monster_class=='Mythical':tags.append('element:Mythical')
  if monster_class=='Supernatural':tags.append('element:Electricity')
  if subclass:tags.append('subclass:'+subclass)
  tags=list(dict.fromkeys(t for t in tags if t in self.monster_tags))
  if not any(t.startswith('class:') for t in tags):tags=['class:Natural']+[t for t in tags if not t.startswith('class:')]
  # Generate in sizeable batches. Reject only exact copies/identical prior names and
  # blatantly invalid text; this is not a human-designed monster-name filter.
  result=[];attempts=0
  while len(result)<n and attempts<8:
   attemptseed=self.rng.randrange(2**31)
   batch=self.pure_generate(self.monstermodel,tags,n=max(24,(n-len(result))*6),seed=attemptseed,temp=self.temperature if temp is None else temp)
   for word in batch:
    if 3<=len(word)<=25 and word not in OFFICIAL and word not in self.used and any(ch.isalpha() for ch in word):
     self.used.add(word);result.append({'name':word,'origin':'generated','generator_conditions':tags})
     if len(result)==n:break
   attempts+=1
  if len(result)<n:raise RuntimeError(f'Monster generator provided only {len(result)} valid names for {n} slots')
  return result
 def predict_teleport(self,name,candidates):
  ck=self.teleportmodel
  text=[f'ISLAND {name} MONSTER {m}' for m in candidates]
  p=ck['classifier'].predict_proba(ck['vectorizer'].transform(text))[:,1]
  return [(c,float(q)) for c,q in zip(candidates,p)]
 def breeding_roster(self,els,gimmicks):
  core=[]
  for size in range(1,len(els)+1):
   for subset in itertools.combinations(els,size):
    k=tuple(sorted(subset));match=LOOK.get(k,[])
    if match:
     r=match[0]
     core.append({'name':r['name'],'elements':list(subset),'source':'existing_official_element_combo','monster_class':r['monster_class']})
    else:
     q=self.new_monsters(subset,n=1)[0]
     core.append({'name':q['name'],'elements':list(subset),'source':'generated_missing_combo','generator_conditions':q['generator_conditions']})
  extra=[]
  for gimmick in gimmicks:
   if gimmick=='Mirror forms':continue
   # Source-backed existing monsters from the selected gimmick class. No fabrication.
   def match(m):
    if gimmick=='Wubboxes':return m['name']=='Wubbox'
    if gimmick=='Dipsters':return m['monster_class']=='Dipster'
    if gimmick=='Mythicals':return m['monster_class']=='Mythical'
    if gimmick=='Legendaries':return m['monster_class']=='Legendary'
    if gimmick=='Werdos':return m.get('subclass')=='Werdo'
    if gimmick=='Shugafam':return m.get('subclass')=='Shugafam'
    if gimmick=='bb$quad':return m.get('subclass')=='bb$quad'
    if gimmick=='Paironormals':return m['monster_class']=='Paironormal'
    if gimmick=='Titansouls':return m['monster_class']=='Titansoul'
    if gimmick=='Seasonals':return m['monster_class']=='Seasonal'
    if gimmick=='Ethereals':return m['monster_class']=='Ethereal'
    if gimmick=='Magicals':return m['monster_class']=='Magical'
    if gimmick=='Fire monsters':return m['monster_class']=='Fire'
    if gimmick=='Primordials':return m['monster_class']=='Primordial'
    if gimmick=='Celestials':return m['monster_class']=='Celestial'
    if gimmick=='Wublins':return m.get('subclass')=='Wublin'
    return False
   candidates=[x for x in DATA if match(x) and x['name'] not in {r['name'] for r in core+extra}]
   self.rng.shuffle(candidates)
   for x in candidates[:min(len(candidates),2)]:extra.append({'name':x['name'],'source':'gimmick_existing_guest','gimmick':gimmick})
  return core,extra
 def teleporting_roster(self,name):
  # Train a name-pair score model; two quotas give both known and novel occupants.
  old=self.predict_teleport(name,[m['name'] for m in DATA]);old.sort(key=lambda x:x[1],reverse=True)
  # Names for invented monsters do not receive island metadata. The pair model only
  # sees each candidate's name and the new island's name.
  generated=[]
  for _ in range(5):
   size=self.rng.choice([1,2,3]);elems=self.rng.sample(BASIC,size)
   generated.extend(self.new_monsters(elems,n=6))
  new=self.predict_teleport(name,[m['name'] for m in generated]);new.sort(key=lambda x:x[1],reverse=True)
  # Model scores decide the ranking; deterministic quotas control roster size.
  chosen_old=old[:self.rng.randint(5,10)]
  chosen_new=new[:self.rng.randint(3,7)]
  extras={m['name']:m for m in generated}
  rows=([{'name':n,'source':'existing_via_pair_model','score':round(p,4)} for n,p in chosen_old]+
        [{'name':n,'source':'generated_via_pair_model','score':round(p,4),'generator_conditions':extras[n]['generator_conditions']} for n,p in chosen_new])
  return rows,{'candidate_old':len(old),'candidate_new':len(new),'threshold':float(self.teleportmodel['threshold']),'selection':'top-scoring names subject to a random roster-size quota'}
 def zapping_roster(self,element):
  n=self.rng.randint(6,14)
  # Celestial or Wublin-style naming is another stylistic decision, not an elemental rule.
  cls=self.rng.choice(['Celestial','Supernatural'])
  sub='Wublin' if cls=='Supernatural' else None
  rows=self.new_monsters([element],monster_class=cls,subclass=sub,n=n)
  return [{'name':x['name'],'source':'generated_only','seed_element':element,'generator_conditions':x['generator_conditions']} for x in rows],cls
 def create(self,kind=None,elements=None,temperature=None):
  kind=kind or self.rng.choice(TYPES)
  if kind not in TYPES:raise ValueError(f'Invalid type {kind}')
  if kind=='breeding':
   if elements is None:elements=self.rng.sample(BASIC,self.rng.choice([3,4,4,4,5]))
   else:elements=list(dict.fromkeys(elements))
   if not 1<=len(elements)<=5:raise ValueError('Breeding island needs 1–5 unique elements')
  elif kind=='zapping':
   elements=[self.rng.choice(BASIC)] if not elements else elements
   if len(elements)!=1:raise ValueError('Zapping requires one element')
  else:
   if elements:raise ValueError('Teleporting names are conditioned only on type; remove --elements')
   elements=[]
  if any(e not in BASIC for e in elements):raise ValueError('Unknown element in '+str(elements))
  name,trace=self.name_for_island(kind,elements,temperature)
  result={'island_name':name,'primary_type':kind,'seed_elements':elements,'naming_trace':trace,'generation_seed':self.seed,'monster_list':[],'gimmicks':[],'status':'entirely_ai_generated_experiment'}
  if kind=='breeding':
   g,score=self.gimmicks_for_name(name)
   core,extras=self.breeding_roster(elements,g)
   result.update(gimmicks=g,gimmick_scores=score,monster_list=core+extras,core_breeding_slots=len(core),
      supplementary_monsters=len(extras),mirror_form_enabled=('Mirror forms' in g))
  elif kind=='teleporting':
   roster,trace=self.teleporting_roster(name);result.update(monster_list=roster,roster_trace=trace)
  else:
   roster,style=self.zapping_roster(elements[0]);result.update(monster_list=roster,zap_generation_style=style)
  result['monster_count']=len(result['monster_list'])
  result['generated_monster_count']=sum('generated' in m['source'] for m in result['monster_list'])
  if kind=='zapping':
   result['egg_inventories']=for_island(result,seed=self.seed)
   result['egg_inventory_generator']='empirical_conditional_inventory_sampler_v1'
  return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description='AMI Island Factory: generate an island, no LLM')
 p.add_argument('--type',choices=TYPES,default=None);p.add_argument('--elements',nargs='+',default=None)
 p.add_argument('--seed',type=int,default=29);p.add_argument('--temperature',type=float,default=.8)
 p.add_argument('--output',default=None);args=p.parse_args()
 fac=Factory(seed=args.seed,temperature=args.temperature)
 obj=fac.create(args.type,args.elements)
 out=json.dumps(obj,indent=2,ensure_ascii=False)
 if args.output:pathlib.Path(args.output).write_text(out)
 print(out)
