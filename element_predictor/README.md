# AMI — Name → Elements (reverse monster factory)

This is a working new direction for the small AI My Singing Monsters experiments:

```
(user elements, optional class/subclass)
  -> existing AMI Pure name generator
  -> generated NAME ONLY
  -> (A) new trained 46-label name classifier OR (B) original generator run backwards
  -> element predictions
```

Model 2 does **not** receive the input element/class/subclass information. It only sees the generated name. The input conditions are printed in demonstrations solely to show disagreements.

## Input dataset

263 distinct official monster names and saved training tags from the corrected AMI dataset containing `Sol-stice` (lowercase s). These labels include both in-game elements and **additional associated elements** intentionally added for the earlier naming experiment (for example Plant for Shugafam, Celestial for Astral Dipsters). The 46 `element:*` tags include class-like special elements such as `Legendary` and `Mythical`, 20 basic elements (Natural, Fire, Magical, Ethereal, Paironormal, Electricity), seasonal labels, and Primordial labels. **They are not all comparable elemental categories.** No outside language, lore, or synthetic examples are used in training the classifier.

## Approach A: Standalone small neural classifier

- Character embedding, three 1D convolutions, max/mean pooling, two hidden layers, and 46 independent sigmoid outputs. A secondary head predicts basic-element count during training; outputs do not need an element count supplied at inference.
- 42,924 trainable parameters, trained from scratch. Each sigmoid score can be independently switched on using a configurable threshold.
- Three class-stratified outer cross-validation folds. Each name is withheld once; within each training fold a development split selects the stopping checkpoint. The final runnable model is retrained on all 263 names for the median selected number of steps.
- This classifier makes unrestricted multi-element guesses, including combinations that don't occur in the training set. The outputs are *not calibrated probabilities*.

Three-fold held-out metrics aggregated across all 263 names:

| Metric | New CNN | TF-IDF character kNN baseline |
|---|---:|---:|
| 20 basic elements micro average precision (↑) | 0.1361 | 0.1496 |
| 46 total tags micro average precision (↑) | 0.1290 | 0.1629 |
| 20 basic elements micro F1, 0.25 threshold (↑) | 0.2228 | 0.1692 |
| Basic-element top-ranked tag present in true set | 16.35% | 17.49% |

The naive prevalence baseline is approximately 0.0939 for the 20 basic labels, and 0.0491 across all 46. **The kNN baseline beats the neural classifier on ranking metrics**; don't interpret any of these scores as high accuracy. The fixed 0.25 F1 threshold shown above differs from the generator demo's adjustable default of 0.35. Single-example labels such as individual Seasonal events cannot meaningfully generalize with a dataset this size.

## Approach B: Reverse the existing name generator WITHOUT any new training

For each candidate set of training conditions, compute the old model's total negative log-likelihood of the name and reweight the 263 known conditioning templates. Average together the weights of templates that possess each element label. This provides a rough Bayes-like comparison of how compatible a name is with various known elemental combinations.

**Very important caveats:**

- The original generator was trained on **all 263 names**; you cannot fairly evaluate its element-recognition accuracy on those same official names.
- It can only put weight on combinations present in its original 263-example dataset. Novel combinations might never be recoverable exactly.
- The template prior and how probabilities are normalized affect the results. Marginal element weights need not sum to one, because monsters have multiple elements. They are NOT calibrated odds of possessing an element.
- The original generator has no unknown-character token, so its inversion cannot score names containing characters outside its 61-symbol vocabulary. The new classifier uses an `<UNK>` character for those.

One striking difference: `Shugabox` scored ~0.989 Plant under Approach B, while `bb$obbinn` scored ~0.999 Cold. The new CNN ranked Electricity higher for both. These are *inference examples*, not proof of reliable out-of-sample generalization.

## Demo of the complete chain

Requires Python 3, PyTorch, NumPy and scikit-learn.

```sh
cd ami_name_to_elements
python generate_then_predict.py Plant Cold --class Natural --n 5 --temperature 0.8
python generate_then_predict.py Fire Poison --class Mythical --n 5 --seed 2026
python generate_then_predict.py Plant --class Legendary --subclass Shugafam --n 5
```

Direct name classification (without generating first):

```sh
python predict_elements.py "Bumbleler"
python predict_elements.py "bb$obbinn" --threshold .3
python invert_generator.py "Shugabox"
```

Reproduce the CV experiment and final classifier, using only the supplied corrected name dataset:

```sh
python train_elements.py --train
python run_examples.py
```

`run_examples.py` creates a reproducible 36-name generation batch from 12 input families at temperature 0.8. `examples.json` records the original generation conditions, names, element classifier scores, and for the first sample of each family, generator-inversion results. `examples.csv` is a tabular view. Names have not been altered or manually chosen.

## Files

- `models/elements_cnn.pt`: trained new name-to-element model
- `models/original_solstice_corrected_generator.pt`: corrected existing original name model
- `train_elements.py`: model, dataset and 3-fold train/evaluate code
- `predict_elements.py`: on/off 46-label classifier CLI
- `invert_generator.py`: generator-backwards, no-new-weights element inference
- `generate_then_predict.py`: two-stage pipeline
- `run_examples.py`: fully reproducible demonstration batch
- `dataset.jsonl`: 263 name/conditioning pairs
- `cv_metrics.json`, `heldout_predictions.json`, `training.log`: reproducible evaluation results
- `examples.json`, `examples.csv`: unmodified raw outputs and predictions
- `original/ami_pure/train.py`, `original/ami_pure/generate.py`, `original/ami_pure/dataset.jsonl`: code/data required to restore and run the original name generator
- `monsters.json`, `islands.json`: retained source artifacts used during the project (islands are not inputs to this model)

The study is intentionally about what tiny models learn **from names alone**. It does not infer canonical elements reliably, especially for new or unusual names, and does not see the source class or elements at prediction time.
