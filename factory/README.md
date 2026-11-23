# AMI Island Factory v1 🏝️

A fully runnable **small-model** My Singing Monsters island generator. No API, no pretrained general-purpose language model, no hand-written island names. It reuses the user's original **AMI Pure monster name generator**, and trains three new components from the 41-island experimental main-game dataset.

## Quick start

```bash
pip install -r requirements.txt
python generate_island.py --seed 62
python generate_island.py --type breeding --elements Plant Cold Bone Fire --seed 8
python generate_island.py --type teleporting --seed 31
python generate_island.py --type zapping --seed 31
python smoke_tests.py
```

The output is a **JSON object** with the island name, type, elements, roster, learned gimmicks, origins for each monster, and confidence-like model scores. Use `--output island.json` to write it to disk. For each island, use the `--seed` option for reproducible results and `--temperature` for the name models.

The first command chooses the island type entirely at random; the later commands let you force a type for comparison. All models and training files ship in this ZIP. Works offline on CPU with Python 3.10+.

## The actual chain

1. **Random island type**: breeding / teleporting / zapping. The user can optionally force one.
2. **Seed elements**: breeding chooses 3–5 of 20 elements (with 4 favored), zapping chooses one, teleporting none.
3. **Island name**: a 10,420-parameter conditional GRU predicts a name using island type and optional elements, learned on 40 released-main-game island names (including Mirror names) and stopped at step 100. It does *not* append “Island” by hand; awkward names are possible. Only obvious invalid or duplicate names are rejected.
4. **Breeding-only gimmicks**: a name-only character n-gram Ridge classifier predicts any of 16 existing gimmicks, including Dipsters, Wubboxes, Mythicals, Legendaries, Seasonals, Titansouls, etc. If Paironormals are selected, `Mirror forms` is automatically enabled.
5. **Roster**:
   - **Breeding**: enumerate all `2^N - 1` nonempty elemental combinations. For an exact official element combination belonging to a breeding element family (Natural, Fire, Magical, Ethereal, Paironormal), reuse an official species; otherwise sample a **new** name from AMI Pure. Gimmicks optionally add a small selection of existing supporting monsters.
   - **Teleporting**: pairwise classifier takes **only island name + monster name**, scores all 263 existing monsters and 30 new monster-name candidates, and selects a small top-ranked subset of each. It does not receive element tags or classes, and has no ability to teleport actual game entities.
   - **Zapping**: chooses one seed element and a random 6–14 monsters, **all newly generated** using AMI Pure, with Celestial- or Wublin-style conditioning. The invented monster roster is experimental and does not imply actual zapping requirements.
6. **Output**: one reproducible imaginary island JSON, usable for a future sound/bio/design generator.

## Examples shipped in the project

| File | Type | Model-generated name | Input elements | Monster roster |
|---|---|---|---|---:|
| `demo_breeding.json` | Breeding | Mir Island | Plant, Cold, Bone, Fire | 24 (15 core + 9 gimmick guests) |
| `demo_teleporting.json` | Teleporting | Aal Island | — | 11 |
| `demo_zapping.json` | Zapping | Mirror WFanir Island | Plant | 7, all generated |
| `demo_automatic.json` | Randomly chosen (Zapping) | Wir Air Island | Fire | 9, all generated |

All names above came from trained AMI models, not manual replacements. The first model made bizarre decisions that are preserved rather than repaired. Each demo includes every monster and its source.

## Training and evaluation

```bash
python build_data.py
python train_models.py
python name_ngram_baseline.py
```

| Component | Training examples | Evaluation | Result |
|---|---:|---|---:|
| Name GRU | 40 names in 31 independent island families | 4-fold *family-grouped* held-out next-character loss | 1.944 nats/character, selected step 100 |
| Type-only name trigrams (baseline) | 40 names | same 4-fold family-grouped held-out loss | 2.344 nats/character |
| Gimmick classifier | 17 unique, released non-Mirror breeding islands | leave-one-island-out micro-F1 | 0.529 (prevalence baseline **0.599**) |
| Teleporting pair classifier | 2,893 pairs from 11 released teleporting islands; 267 positive pairs | group-held-out (entire islands) average precision | 0.256 (prevalence baseline 0.092) |

**Important:** Even though the name RNN beats an n-gram baseline, it has very little training data. The gimmick classifier **does not beat its simple prevalence baseline** on micro-F1, and its macro-F1 is only 0.208. This is a fun experimental machine, not a faithful predictor of island mechanics or gameplay. Teleporting results on unseen *real* islands do not guarantee generalization to invented names. The displayed scores are not calibrated probabilities.

## Source and metadata warnings

- Main-game island/monster placements originate from the prior AMI 41-island dataset, itself manually transcribed from the community roster [All Monsters by Island](https://www.jetpunk.com/user-quizzes/1629255/all-monsters-in-my-singing-monsters-by-island); it is not a verified game export.
- The `source/` directory contains the exact prior original-game dataset, monster-name generator code and weights, and island predictor materials needed to reproduce this version.
- `build_data.py` records *curated* island-type categories and element pools. Actual MSM islands can combine breeding, teleportation, synthesis, and other mechanics. We map them to **one primary type** for this experimental three-type generator, not as a canonical taxonomy.
- See official help-center descriptions of [teleporting islands](https://bigbluebubblesupport.helpshift.com/hc/en/4-my-singing-monsters/section/58-teleporting-islands/), [the Ethereal Islets](https://bigbluebubblesupport.helpshift.com/hc/en/4-my-singing-monsters/faq/612-how-do-i-unlock-ethereal-islets/), and [Paironormal transposition](https://bigbluebubblesupport.helpshift.com/hc/en/4-my-singing-monsters/faq/569-how-do-i-get-a-paironormal-monster/).
- Nine Mirror Island training names are variants of the originals; the cross-validation name splits keep related names in the same fold.
- The unreleased Poison Islet is bundled for context **but excluded as a positive training example**. Its previously inferred roster remains tagged as projected in `source/`.
- **Elements assigned by name generator != official elements**. Exact existing-monster matches use only `official_elements`, *never associated elements*. Unseen combination names are always tagged as generated.
- Sources for gimmick labels come from actual classes/subclasses in the transcribed rosters, not independently verified rules for every island.

## Files

- `island_corpus.json`/`.csv`: island metadata, types, elemental pools, gimmick labels and existing rosters.
- `models/island_name_gru.pt`: newly trained conditional name generator weights.
- `models/gimmicks.joblib`: name-only gimmick multi-label classifier.
- `models/teleport_pair.joblib`: island-name/monster-name pairwise membership classifier.
- `source/`: prior AMI source artifacts needed to generate new monster names.
- `train_models.py`, `build_data.py`, `name_ngram_baseline.py`: training/validation code.
- `generate_island.py`: runnable island generator.
- `smoke_tests.py`: generation invariants for all three island types.
- `name_cv.json`, `ngram_cv.json`, `gimmick_cv.json`, `teleport_cv.json`: actual held-out metrics.
- `demo_*.json`: raw model-generated islands.

No external vocabulary was added to the name generator. No bios were generated; those experiments remain separate.
