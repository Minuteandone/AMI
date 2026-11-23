# AMI — Artificial Monster Intelligence 🏝️🧬

Tiny, experimentally trained My Singing Monsters-inspired models. Most generated names, islands, and biographies are spectacular nonsense **on purpose**. No external language-model API is required.

## Install

Requires Python 3.10+ and CPU-capable PyTorch.

```bash
python -m pip install -r requirements.txt
```

## Quickstart: generate an entire island

```bash
python factory/generate_island.py --seed 20261122
python factory/generate_island.py --type breeding --elements Plant Cold Fire Bone --seed 8
python factory/generate_island.py --type teleporting --seed 31
python factory/generate_island.py --type zapping --seed 143 --output new_island.json
cd factory && python smoke_tests.py
```

The factory uses saved neural weights and fitted statistical models for island naming, gimmicks, teleporting rosters, and monster-name creation; the zapping inventory generator is empirical/statistical rather than a neural network. Generated islands are **not** real gameplay content.

## Predict the elements AND count from just a monster name

```bash
python element_predictor/predict_elements_count.py "Bumbleler"
python element_predictor/generate_then_count.py Fire Poison --class Mythical --n 5
```

The classifier's predictions are weak on unseen names. They can disagree with AMI's original name conditioning. Read `element_predictor/COUNT_AWARE_README.md` for held-out evaluations.

## Generate a monster biography

```bash
python bios/free_unigram/unigram_bio.py generate "Bumbleler" --dir complete_model --seed 9
python bios/eos_models/generate_bio.py --help
```

Bios are typically grammatical nonsense; these tiny models did not learn to write fluent English. We include trained weights and tokenizers but **do not redistribute the original copyrighted monster bio corpus**.

## Model weights and implementation

- `factory/models/island_name_gru.pt`: conditional island name generator
- `factory/models/gimmicks.joblib`: island-name gimmick classifier
- `factory/models/teleport_pair.joblib`: pairwise island-name and monster-name classifier
- `factory/source/ami_pure/models/contrast_creative.pt`: AMI Pure monster name generator
- `factory/source/island_model.pt`: name-only island membership predictor
- `element_predictor/count_models/elements_with_count.pt`: count-aware name-to-elements model
- `element_predictor/models/elements_cnn.pt`: earlier name-to-elements model
- `bios/free_unigram/complete_model/best.pt`: name-to-free-form-bio model
- `bios/eos_models/{free,template}/best.pt`: two previous bio model variants
- `data/inventory_model.json`: fitted zapping inventory probabilities

See model-specific READMEs for architectures, evaluation, provenance, and reproducible commands. *No weights need to be downloaded separately.*

## Original AI output versus the user's interpretation

- `examples/fon_islao_original.json`: original generated island name **Fon Islao**.
- `examples/hoax_cosmos_alternative.json`: user's alternate name **Hoax Cosmos**.

The two are separate interpretations of one generated island, not a replacement of the AI's original name.

## Data and attribution

- Prior island-monster placements are an **experimental community-transcribed** data table, not a verified live game export. Some islands and predicted rosters were provisional.
- The compact zapping recipe reference dataset comes from [getinski/msm-breeding-planner](https://github.com/getinski/msm-breeding-planner), MIT-licensed, snapshot commit `53aa0df3979e466932095506980560e4b019f6f3`. It is included for generating fictional egg requirements; this is not an authoritative live requirement list.
- Official MSM monster names appear as dataset labels for technical experimentation. This fan project is not affiliated with Big Blue Bubble.
- Official copyrighted monster biography text and binary localization files have **not** been published here. Retraining the bio models from scratch requires independently obtained training data.
- The models have very small datasets and validation metrics generally do **not** imply real generalization. In particular, the gimmick predictor performed worse than a prevalence baseline.

## Security note

PyTorch `.pt` and joblib `.joblib` checkpoints are **pickle-based** formats; only load files from trusted sources. These are our generated experiment checkpoints.

## Tests

```bash
cd factory && python smoke_tests.py
python tests.py
```
