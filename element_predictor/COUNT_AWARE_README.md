# AMI — Number-of-elements prediction

A new version of the experimental My Singing Monsters **name-only** element classifier.

## What changed

The original 42,924-parameter CNN actually included a supervised basic-element count head, but the inference script ignored it and used a fixed per-element threshold. This version trains a fresh classifier with:

* 46 independent element/tag scores;
* a six-way **basic element count** head for 0–5 of the 20 Natural/Fire/Magical/Ethereal/Paironormal/Electricity elements;
* a three-way **special tag count** head for 0–2 of the other 26 class-like elements (Legendary, Celestial, Seasonal, etc.).

It chooses both counts using the learned softmax heads, then selects exactly that many highest-ranked labels **within each group**. The island and name-generation models remain unchanged. No class, elements, or element count are supplied to the classifier. All 263 training examples are from the official-name dataset; there is no outside vocabulary or synthetic data.

**Important distinction:** These are counts of the project's **training element tags**, not necessarily the official number of in-game elemental icons. Associated elements were intentionally added to Celestials, Mythicals, Legendaries and others. For example, a Celestial can have the tags `Celestial + Fire`, although Fire is an associated element rather than a second official icon.

## Commands

From this directory (PyTorch, NumPy and scikit-learn required):

```bash
python predict_elements_count.py "Bumbleler"
python generate_then_count.py Fire Poison --class Mythical --seed 37 --n 7
python generate_then_count.py Plant --class Legendary --subclass Shugafam --n 10
python count_aware.py --train
```

`count_aware.py --train` repeats 3 class-stratified held-out folds, records out-of-fold predictions, and trains the final usable model on all 263 names. `count_models/elements_with_count.pt` contains its actual trained weights.

## Results (three-fold out-of-fold; all 263 names unseen in their respective fold)

| Metric | Count-aware neural model | Naive most-frequent count |
|---|---:|---:|
| Exact basic-element count | 31.9% | 40.7% |
| Exact special-element count | 66.5% | 65.4% |
| Exact total count | 30.8% | 42.6% |
| Exact complete set of element tags | 1.5% | Not measured |
| All-tag micro F1, count-aware top-K | 0.157 | — |
| All-tag micro F1, independent 0.25 threshold on same logits | 0.186 | — |

**The model is not good at guessing how many elements an unfamiliar monster really has.** It can **choose** a number, but its count prediction is worse than a majority-count baseline. The automatic count feature is meant for creative experiments, not faithful canonical classification. The model rarely predicts four or five elements, reflecting the small, imbalanced data.

All count-accuracy figures derive from the out-of-fold models. The supplied full-dataset final checkpoint is useful for generating fictional names, not for evaluating its accuracy on its own training names. Percentages attached to element scores are uncalibrated, not canonical probabilities.

## Files

* `count_aware.py` — model architecture, training, 3-fold evaluation, inference
* `predict_elements_count.py` — direct name-only CLI
* `generate_then_count.py` — original name generator -> automatic count/element model
* `count_models/elements_with_count.pt` — trained neural weights
* `count_cv_metrics.json` — folds, count histograms, comparative metrics
* `count_heldout_predictions.json` — all 263 genuine held-out predictions
* `count_examples.json` / `.csv` — familiar invented and official monster examples
* `count_chain_example.json` — raw samples from both models
* `compare_count_methods.py`, `count_method_exploration.json` — exploratory baselines (not independently tuned)
* `train_elements.py`, `predict_elements.py`, `generate_then_predict.py`, `models/` — original v1 classifier and its code
* `dataset.jsonl`, `monsters.json` — reference training dataset
* `original/ami_pure/` — existing AMI name model code

**Data caveat:** The original dataset's count labels include associated elements, as described above, and one monster (`Glowbe`) has zero element tags. Do not silently force it to one if you want to reproduce the experiment.
