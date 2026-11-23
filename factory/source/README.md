# AMI Islands v4 — original My Singing Monsters only

This is the two-model AI monster factory with **Dawn of Fire-specific islands removed** from the island-label dataset. It is an experimental name-conditioned multi-label classifier.

## What changed

- Removed 7 DoF islands: Cave Island, Cloud Island, Party Island, Space Island, Starhenge, The Continent, Wanderer Island. These are documented as DoF islands on the MSM Wiki.
- Kept **41** main-game locations, including nine Mirror Islands (copied from their corresponding originals), The Colossingum, Composer Island, the Ethereal Islets, and the **unreleased inferred Poison Islet**.
- Retained **263** monsters with at least one main-game island.
- Reduced positive monster/island entries from **1,404** to **1,227**.
- Retrained the 41-output character CNN and the 41-output logistic-regression n-gram baseline; did **not** retrain or alter the first name generator.
- Both island predictors receive **only Unicode characters from the name**; they cannot access elements, class, subclass, or the first model's context.
- Inferred Special Dipster placements and the user-supplied **14-monster projected Poison Islet roster** are preserved with provenance labels.

## Held-out evaluation

Three fixed-seed 80/20 splits (with an internal validation subset) measured on non-Mirror targets:

| Predictor | Mean micro-F1 | Mean average precision |
|---|---:|---:|
| Frequency baseline | 0.2339 | 0.1052 |
| CNN | 0.2256 | 0.2442 |
| Character n-gram linear | 0.2592 | 0.2390 |

The CNN does **not** outperform the frequency baseline on held-out micro-F1. These scores are weak, and the output scores are **not calibrated probabilities**. The linear baseline achieves the best micro-F1.

## New two-model generation

Generate a batch with Fire + Poison and Mythical conditioning:

```sh
python pipeline.py Fire Poison --class Mythical --n 6 --seed 17 --temperature 0.8
```

The **first** generated monster was **Bumbleler**. The trained 41-island CNN switched on: Magical Nexus, Poison Islet. These are model predictions, not canonical placements. The full unedited batch is in `new_monster_batch.json`.

The original previous example Gadla can also be run through the newly trained island predictor without regenerating its name:

```sh
python predict_islands.py Gadla
```

## Files and reproduce

- `build_labels.py`: rebuild island labels from bundled monster registry, transcribed community roster, and explicit user-provided rule extensions. It excludes DoF labels at the source parsing stage.
- `train_islands.py`: retrain CNN plus logistic regression; generate held-out results.
- `predict_islands.py`, `pipeline.py`: inference and two-model chain.
- `island_model.pt`: trained 41-output CNN.
- `validation_model.pt`: held-out-fold CNN.
- `linear_baseline.joblib`: trained character n-gram classifier.
- `solstice_corrected.pt`, `ami_pure/`: first model and its definition, unchanged from v3.
- `placements.json`, `placements.csv`, `placement_evidence.csv`, `islands.json`, `provenance.json`: labels, source categories, and projected status.
- `diff_from_v3.json`: exact deleted island/monster placements.
- `evaluation.json`, `test_predictions.json`, `train_history.json`, `new_monster_batch.json`: evaluation and examples.
- `smoke_test.py`: structural assertions for main-game-only targets.
- `make_ami_islands_v4.py`: the script used to create v4 from v3, included for reproducibility.

To rebuild locally after extracting:

```sh
python build_labels.py
python train_islands.py
python smoke_test.py
```

Requirements: Python 3.10+, `torch`, `numpy`, `scikit-learn`, and `joblib`.

## Source limitations

The underlying placements come mainly from a manually transcribed, community-maintained [by-island JetPunk roster](https://www.jetpunk.com/user-quizzes/1629255/all-monsters-in-my-singing-monsters-by-island), not a complete, current game export. Mirror island memberships are copied from corresponding originals rather than independently verified. The **Poison Islet roster remains inferred, not released**. This version changes game scope, but does not comprehensively revalidate every monster placement.
