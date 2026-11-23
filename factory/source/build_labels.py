"""Curated transcription of the By-Island 2026-10-29 roster.
Original source https://www.jetpunk.com/user-quizzes/1629255/all-monsters-in-my-singing-monsters-by-island
This is not a game-data export. No element/class info is provided to island model.
"""
import json,re,csv,collections,pathlib
P=pathlib.Path(__file__).resolve().parent
SOURCE='https://www.jetpunk.com/user-quizzes/1629255/all-monsters-in-my-singing-monsters-by-island'
# Each line has island_name | comma-separated exact monster names; commas not in names.
RAW=r'''
Plant Island|Potbelly,Shrubb,Oaktopus,Furcorn,Pummel,Clamble,Bowgart,Entbrat,Noggin,Toe Jammer,Mammott,Fwog,Drumpler,Maw,T-Rox,Ghazt,Shugabush,Punkleton,G'joob,Tawkerr,Parlsona,Do,Re,Mi,Fa,Sol,La,Ti,Glowbe,Wubbox
Cold Island|Mammott,Maw,Furcorn,Pango,Bowgart,Congle,Thumpies,Deedge,Toe Jammer,Potbelly,Tweedle,Oaktopus,Quibble,Dandidoo,Spunge,Grumpyre,bbli$zard,Yool,Strombonin,Tawkerr,Maggpi,Do,Re,Mi,Fa,Sol,La,Ti,Glowbe,Wubbox
Air Island|Tweedle,Cybop,Quibble,Pango,Scups,PomPom,Congle,Riff,Noggin,Toe Jammer,Mammott,Fwog,Drumpler,Maw,T-Rox,Reebro,Alcordion,Schmoochle,Hoola,Yawstrich,Parlsona,Stoowarb,Do,Re,Mi,Fa,Sol,La,Ti,Glowbe,Wubbox
Water Island|Toe Jammer,Fwog,Oaktopus,Quibble,Pummel,Scups,Spunge,Shellbeat,Noggin,Potbelly,Tweedle,Shrubb,Cybop,Dandidoo,Reedling,Jeeode,T-Pirainha,Blabbit,Anglow,Maggpi,Parlsona,Do,Re,Mi,Fa,Sol,La,Ti,Glowbe,Wubbox
Earth Island|Noggin,Drumpler,Shrubb,Cybop,Clamble,PomPom,Reedling,Quarrister,Mammott,Potbelly,Tweedle,Furcorn,Pango,Dandidoo,Thumpies,Humbug,Hoola,Hyehehe,Tawkerr,Stoowarb,Do,Re,Mi,Fa,Sol,La,Ti,Glowbe,Wubbox
Gold Island|Noggin,Toe Jammer,Mammott,Potbelly,Tweedle,Fwog,Drumpler,Shrubb,Cybop,Maw,Oaktopus,Quibble,Furcorn,Pango,Dandidoo,T-Rox,Pummel,Scups,Clamble,PomPom,Reedling,Bowgart,Congle,Spunge,Thumpies,Entbrat,Riff,Shellbeat,Quarrister,Deedge,Shugabush,Punkleton,Yool,Schmoochle,Blabbit,Hoola,Jam Boree,Wubbox
Fire Haven|Kayna,Stogg,Flowah,Glowl,Repatillo,Floogull,Barrb,Tring,Noggin,Potbelly,Tweedle,Shrubb,Cybop,Dandidoo,Reedling,Gobbleygourd,Buzzinga,Candelavra,Charrkoll,Do,Re,Mi,Fa,Sol,La,Ti,Glowbe,Wubbox
Fire Oasis|Kayna,Phangler,Boskus,Glowl,Wynq,Whaddle,Woolabee,Sneyser,Toe Jammer,Mammott,Tweedle,Maw,Quibble,Pango,Congle,Spurrit,Gobbleygourd,Cherubble,Mimic,Do,Re,Mi,Fa,Sol,La,Ti,Glowbe,Wubbox
Amber Island|Kayna,Stogg,Phangler,Boskus,Flowah,Glowl,Thrumble,Ziggurab,Repatillo,Floogull,Wynq,Rootitoot,Whaddle,Sooza,Woolabee,Barrb,Tiawa,Edamimi,Flum Ox,PongPing,Krillby,Tring,Incisaur,Sneyser,Bisonorus,Yelmut,Gnarls,Tuskski,Candelavra,Drummidary,Bowhead,Viveine,Glowbe
Tribal Island|Kayna,Noggin,Toe Jammer,Mammott,Potbelly,Tweedle,Fwog,Drumpler,Shrubb,Cybop,Maw,Oaktopus,Quibble,Furcorn,Pango,Dandidoo,T-Rox,Pummel,Scups,Clamble,PomPom,Reedling,Bowgart,Congle,Spunge,Thumpies,Entbrat,Riff,Shellbeat,Quarrister,Deedge
Wublin Island|Brump,Zynth,Poewk,Thwok,Dwumrohl,Zuuker,Screemu,Tympa,Dermit,Gheegur,Whajje,Creepuscule,Blipsqueak,Scargo,Astropod,Pixolotl,Bona-Petite,Maulch,Fleechwurm,Monculus,Glowbe,Wubbox
Light Island|Fluoress,Bulbo,Gob,Pluckbill,Spytrap,Fiddlement,TooToo,Blow't,Mammott,Potbelly,Kayna,Furcorn,Boskus,Flowah,Sooza,Shhimmer,Whiz-bang,Yelmut,Tiawa,Drummidary,Do,Re,Mi,Fa,Sol,La,Ti,Glowbe,Phosphoran Phlox
Psychic Island|Theremind,Poppette,Bonkers,Yuggler,Tapricorn,Periscorp,Rooba,Gloptic,Toe Jammer,Potbelly,Kayna,Oaktopus,Phangler,Flowah,Rootitoot,Hairionette,Boo'qwurm,Edamimi,Bisonorus,Bowhead,AnomaLee,Do,Re,Mi,Fa,Sol,La,Ti,Glowbe,Cruv'laaphtian Crocus
Faerie Island|Floot Fly,HippityHop,Squot,Wimmzies,Cantorell,Bridg-it,Clavi Gnat,Pladdie,Noggin,Mammott,Kayna,Drumpler,Stogg,Boskus,Ziggurab,Owlesque,Ffidyll,Krillby,PongPing,Tuskski,Do,Re,Mi,Fa,Sol,La,Ti,Glowbe,Faesoddoid Fungus
Bone Island|Clackula,Peckidna,Denchuhs,Hawlo,Withur,Uuduk,Banjaw,Plinkajou,Noggin,Toe Jammer,Kayna,Fwog,Stogg,Phangler,Thrumble,Arcorina,Clavavera,Flum Ox,Incisaur,Gnarls,Do,Re,Mi,Fa,Sol,La,Ti,Glowbe
Magical Sanctum|Fluoress,Theremind,Floot Fly,Clackula,Cahoot,Knucklehead,Roarick,Déjà-Jin,Xyster,Osstax,Frondley,G'day,Mushaboom,Larvaluss,Enchantling,Carillong,Do,Re,Mi,Fa,Sol,La,Ti,Glowbe
Magical Nexus|Fluoress,Theremind,Floot Fly,Clackula,Bulbo,Gob,Pluckbill,Poppette,Bonkers,Yuggler,HippityHop,Squot,Wimmzies,Peckidna,Denchuhs,Hawlo,Spytrap,Fiddlement,TooToo,Tapricorn,Periscorp,Rooba,Cantorell,Bridg-it,Clavi Gnat,Withur,Uuduk,Banjaw,Blow't,Gloptic,Pladdie,Plinkajou
Ethereal Island|Ghazt,Grumpyre,Reebro,Jeeode,Humbug,Whisp,Nebulob,Sox,Jellbilly,Arackulele,Boodoo,Kazilleon,Bellowfish,Dragong,Fung Pray,Monculus,Do,Re,Mi,Fa,Sol,La,Ti,Glowbe,Wubbox
Ethereal Workshop|Yooreek,Meebkin,Blarret,Gaddzooks,Auglur,Flasque,Nitebear,Piplash,X'rt,Teeter-Tauter,Whaill,Vhenshun,Pentumbra,Rhysmuth,Oogiddy,BeMeebEth,Glowbe
Plasma Islet|Ghazt,Whisp,Nebulob,Sox,Jellbilly,Yooreek,Meebkin,Blarret,Gaddzooks,Auglur,Flasque,Whaill,Lowb
Shadow Islet|Grumpyre,Whisp,Arackulele,Boodoo,Kazilleon,Yooreek,Meebkin,Blarret,Nitebear,Piplash,X'rt,Pentumbra,Fandhul
Mech Islet|Reebro,Nebulob,Arackulele,Bellowfish,Dragong,Yooreek,Gaddzooks,Auglur,Nitebear,Piplash,Teeter-Tauter,Vhenshun,Bogle
Crystal Islet|Jeeode,Sox,Boodoo,Bellowfish,Fung Pray,Meebkin,Gaddzooks,Flasque,Nitebear,X'rt,Teeter-Tauter,Rhysmuth,Droah
Paironormal Carnival|Hairionette,Owlesque,Arcorina,Shhimmer,Scallyrags,Raqsoun,eRmA gUrDy,Dakktyl,The Inflatterer,Vawk,Jerm,Galymph,Unklaw,Illoost,STRingmaster
Shugabush Island|Shugabush,Shugarock,Shugabass,Shugajo,Shugabeats,Shugabuzz,Shugitar,Shugavox,Mammott,Potbelly,Oaktopus,Quibble,Furcorn,PomPom,Deedge,Viveine,Glowbe
bbli$zard Island|bbli$zard,bb$enior,bbjunior$,bbdek$tr,bb$kinbash,bb$ynth,bb$inistr,bb$ervo,Mammott,Tweedle,Drumpler,Fwog,Pango,Spunge,Riff
Seasonal Shanty|Punkleton,Yool,Schmoochle,Blabbit,Hoola,Gobbleygourd,Clavavera,Viveine,Jam Boree,Carillong,Whiz-bang,Monculus,Ffidyll,Boo'qwurm,Spurrit
Mythical Island|Cataliszt,G'joob,Strombonin,Yawstrich,Anglow,Hyehehe,Buzzinga,Cherubble,Bleatnik,Cranchee,Sporerow,Pinghound,Wheezel,Knurv,shLep,Glowbe
Celestial Island|Hornacle,Furnoss,Glaishur,Blasoom,Syncopite,Vhamp,Galvana,Scaratar,Loodvigg,Torrt,Plixie,Attmoz,Superno-Fa,Glowbe
The Colossingum|Noggin,Toe Jammer,Mammott,Potbelly,Tweedle,Fwog,Drumpler,Shrubb,Cybop,Maw,Oaktopus,Quibble,Furcorn,Pango,Dandidoo,T-Rox,Pummel,Scups,Clamble,PomPom,Reedling,Bowgart,Congle,Spunge,Thumpies,Entbrat,Riff,Shellbeat,Quarrister,Deedge,Ghazt,Grumpyre,Reebro,Jeeode,Humbug,Whisp,Nebulob,Sox,Jellbilly,Arackulele,Boodoo,Kazilleon,Bellowfish,Dragong,Fung Pray,Punkleton,Yool,Schmoochle,Blabbit,Hoola
Composer Island|Noggin,Toe Jammer,Mammott,Potbelly,Tweedle,Fwog,Drumpler,Shrubb,Cybop,Maw,Oaktopus,Quibble,Furcorn,Pango,Dandidoo,T-Rox,Pummel,Scups,Clamble,PomPom,Reedling,Congle,Bowgart,Spunge,Thumpies,Entbrat,Riff,Shellbeat,Quarrister,Deedge,Ghazt,Glowbe
The Continent|Noggin,Toe Jammer,Mammott,Potbelly,Tweedle,Kayna,Fwog,Drumpler,Shrubb,Cybop,Stogg,Maw,Oaktopus,Quibble,Phangler,Furcorn,Pango,Boskus,Dandidoo,Flowah,Glowl,T-Rox,Pummel,Scups,Thrumble,Clamble,PomPom,Ziggurab,Reedling,Repatillo,Floogull,Bowgart,Congle,Wynq,Spunge,Rootitoot,Whaddle,Thumpies,Sooza,Woolabee,Barrb,Entbrat,Riff,Tiawa,Shellbeat,Edamimi,Flum Ox,Quarrister,PongPing,Krillby,Deedge,Tring,Incisaur,Sneyser,Bisonorus,Yelmut,Mimic,Gnarls,Tuskski,Candelavra,Drummidary,Bowhead,Punkleton,Yool,Schmoochle,Blabbit,Hoola,G'joob,Strombonin,Yawstrich,Anglow,Hyehehe,Buzzinga,Cherubble
Party Island|Noggin,Mammott,Potbelly,Kayna,Shrubb,Stogg,Maw,Oaktopus,Quibble,Furcorn,T-Rox,Scups,PomPom,Ziggurab,Edamimi,Quarrister,Candelavra
Space Island|Noggin,Toe Jammer,Potbelly,Drumpler,Cybop,Stogg,Furcorn,Flowah,Pummel,Clamble,PomPom,Repatillo,Bowgart,Congle,Spunge,Rootitoot,Sooza,Deedge,Tring,Sneyser,Candelavra,Bowhead
Cloud Island|Tweedle,Oaktopus,Phangler,Pango,Boskus,Dandidoo,Thrumble,PomPom,Ziggurab,Reedling,Whaddle,Woolabee,Entbrat,Tiawa,Shellbeat,Flum Ox,Krillby,Incisaur,Sneyser,Tuskski,Drummidary
Cave Island|Mammott,Kayna,Fwog,Quibble,Flowah,Glowl,Floogull,Bowgart,Wynq,Thumpies,Barrb,Riff,Shellbeat,PongPing,Krillby,Bisonorus,Yelmut,Mimic,Gnarls
Starhenge|Hornacle,Furnoss,Glaishur,Blasoom,Syncopite,Vhamp,Galvana,Scaratar,Loodvigg,Torrt,Plixie,Attmoz
Wanderer Island|Punkleton,Yool,Schmoochle,Blabbit,Hoola,G'joob,Strombonin,Yawstrich,Anglow,Hyehehe,Buzzinga,Cherubble
'''.strip()

def norm(s):
    return re.sub(r'[^a-z0-9]','',s.casefold())


# Rule-based extensions requested for v2. Keep provenance for each assignment.
DIPSTER_COUNTERPARTS={
    'Aquatic Fa':'Fa', 'Terrene Sol':'Sol', 'Waxen La':'La', 'Vitreous Ti':'Ti',
    'King Do':'Do', 'Queen Re':'Re', 'Bishop Mi':'Mi', 'Knight Fa':'Fa'
}
ASTRAL_DIPSTERS=('Superno-Fa','Sol-stice')
ISLET_NAME='Poison Islet'
# Standalone Dawn of Fire islands, not present in the original game.
DOF_ONLY_ISLANDS={'Party Island', 'Space Island', 'Cave Island', 'Starhenge', 'Cloud Island', 'The Continent', 'Wanderer Island'}

def build():
    source=json.loads((P/'monsters.json').read_text(encoding='utf-8'))
    lookup={norm(m['name']):m['name'] for m in source}
    src_by_name={m['name']:m for m in source}
    island_to_names={};unknown_source=collections.defaultdict(list)
    evidence=collections.defaultdict(dict)
    def add(island,name,kind):
        if name not in island_to_names.setdefault(island,[]):island_to_names[island].append(name)
        evidence[name][island]=kind
    for line in RAW.splitlines():
        island,raw_names=line.split('|',1)
        if island in DOF_ONLY_ISLANDS:
            continue
        for raw in raw_names.split(','):
            key=norm(raw.strip())
            if key not in lookup:
                unknown_source[island].append(raw.strip());continue
            add(island,lookup[key],'community_roster')
    # Existing mirror label simplification carried forward, independently marked.
    bases=['Plant Island','Cold Island','Air Island','Water Island','Earth Island',
           'Light Island','Psychic Island','Faerie Island','Bone Island']
    for base in bases:
        for name in island_to_names[base]:add('Mirror '+base,name,'mirror_derived_from_base')
    # User-specified placement rules. These are inferred rather than source-transcribed.
    for special,original in DIPSTER_COUNTERPARTS.items():
        assert special in src_by_name and original in src_by_name,(special,original)
        for island in list(island_to_names):
            if original in island_to_names[island]:
                add(island,special,'user_rule_regular_dipster_inheritance')
    for special in ASTRAL_DIPSTERS:
        assert special in src_by_name
        add('Celestial Island',special,'user_rule_astral_celestial' if special!='Superno-Fa' else evidence[special].get('Celestial Island','user_rule_astral_celestial'))
    # An *inferred future* roster, NOT a released island or verified placement.
    # Most Poison Ethereals are projected, but quads have dedicated Islets.
    # Oogiddy is Poison Islet's native quad; other Poison-element quads stay
    # on their own islets even though they also contain the Poison element.
    forbidden_quads={'Pentumbra','Rhysmuth','Vhenshun'}
    poison_names=[m['name'] for m in source if 'Poison' in m['official_elements']
                  and m['name'] not in forbidden_quads]
    assert len(poison_names)==13,poison_names
    assert {m['name'] for m in source if 'Poison' in m['official_elements']
            and len(m['official_elements'])==4}==forbidden_quads|{'Oogiddy'}
    for name in poison_names:
        add(ISLET_NAME,name, 'projected_unreleased_native_quad_rule' if name=='Oogiddy' else 'projected_unreleased_element_rule')
    add(ISLET_NAME,'Tunefor','projected_unreleased_primordial_rule')
    islands=sorted(island_to_names)
    rows=[]
    for m in source:
        name=m['name'];isl=[island for island in islands if name in island_to_names[island]]
        prov={i:evidence[name][i] for i in isl}
        any_verified=any(v in ('community_roster','mirror_derived_from_base') for v in prov.values())
        status=('source_plus_rules' if any_verified and len(prov)>sum(v in ('community_roster','mirror_derived_from_base') for v in prov.values())
                else 'source_transcribed' if any_verified else
                'inferred_dipster' if any(v.startswith('user_rule') for v in prov.values()) else
                'projected_only' if isl else 'unverified_no_island')
        rows.append({'name':name,'islands':isl,'island_vector':[int(i in isl) for i in islands],
          'label_status':status,'training_eligible':bool(isl),
          'island_provenance':prov,'source':SOURCE if any_verified else None})
    (P/'islands.json').write_text(json.dumps(islands,ensure_ascii=False,indent=2),encoding='utf-8')
    (P/'placements.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    with (P/'placements.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['name','status','island_count']+islands)
        for r in rows:w.writerow([r['name'],r['label_status'],len(r['islands'])]+r['island_vector'])
    with (P/'placement_evidence.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['monster','island','evidence','island_released'])
        for r in rows:
            for island in r['islands']:
                w.writerow([r['name'],island,r['island_provenance'][island],island!=ISLET_NAME])
    metadata={
        'version':'v4-main-game','source':SOURCE,'source_updated':'2026-10-29 as retrieved',
        'changes':['Excluded DoF-only islands: '+', '.join(sorted(DOF_ONLY_ISLANDS)), 'Added Special Dipsters by matching base Dipster island rosters',
         'Sol-stice and Superno-Fa on Celestial Island only',
         'Poison Islet projected roster excludes non-native quads Pentumbra, Rhysmuth, Vhenshun (Oogiddy only quad)'],
        'scope':'Original My Singing Monsters only; 7 Dawn of Fire-only islands excluded; nine mirror targets copied from base island labels',
        'islands':len(islands),'released_islands_in_labels':len(islands)-1,
        'monsters_total':len(rows),'monsters_with_any_labels':sum(bool(r['islands']) for r in rows),
        'source_supported_monsters':sum(r['label_status'] in ('source_transcribed','source_plus_rules') for r in rows),
        'inferred_only_monsters':[r['name'] for r in rows if r['label_status'] in ('inferred_dipster','projected_only')],
        'unknown_names':[r['name'] for r in rows if not r['islands']],
        'unknown_source_names':dict(unknown_source),'total_pairs_including_mirrors':sum(len(r['islands']) for r in rows),
        'poison_islet':{'release_status':'unreleased','roster_status':'predicted_not_confirmed',
                       'rule':'Poison-element monsters except the three non-native quads Pentumbra, Rhysmuth, Vhenshun, plus Tunefor; Oogiddy is the native quad',
                       'monster_count':len(island_to_names[ISLET_NAME]),'monsters':sorted(island_to_names[ISLET_NAME])},
        'dipster_inheritance':DIPSTER_COUNTERPARTS,
        'astral_dipsters':list(ASTRAL_DIPSTERS),
        'original_community_roster_transcription':str(SOURCE), 'excluded_dawn_of_fire_islands':sorted(DOF_ONLY_ISLANDS),
        'note':'Train with projected labels only when explicitly intended; these are not verified game placements.'
    }
    (P/'provenance.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))
    return metadata
if __name__=='__main__':build()
