# AMI Free Bio — name-enriched Unigram tokenizer

**Experiment:** original free-form MSM biography generator, trained with a **different tokenization algorithm**: SentencePiece **Unigram** with byte fallback and registered monster-name pieces. **No** `[monster name]` substitution is used. The model receives only a monster name and generates text until its learned `<end_bio>` token is sampled (with an emergency 450-token limit).

## What is special about this tokenizer?

The previous original model used `SentencePiece(model_type='bpe')`; the recent homemade template model used byte-level BPE. This model uses `SentencePiece(model_type='unigram')`, which fits a probabilistic inventory of pieces rather than learning a strict sequence of pair merges. During tokenizer training we include every **training-set** monster name in addition to the **training-set** official biographies (each name as a standalone line three times). Known single-piece names of at least four characters are registered as user-defined symbols. All other text, including names unseen during training, can be split into learned pieces or UTF-8 bytes. Monster names remain untouched in training output.

The held-out tokenizer is fit on **only the training fold**. It never receives held-out names explicitly or held-out bios. This is stricter than simply training a tokenizer on all biographies then splitting rows afterward. A held-out monster's name *can* still be mentioned by another monster's training biography, and that's a natural cross-reference caveat.

## Files

- `unigram_bio.py` — complete tokenizer trainer, model definition, training, validation, inference CLI.
- `prepare.py` — optional original source-file parser from the earlier AMI Bio project, to rebuild `bios_free.jsonl` using your own local `en.utf8` and `monsters.json`.
- `heldout_model/` — original held-out validation checkpoint and tokenizer (494 training / 86 held-out entries, split by 39 base species).
- `complete_model/` — model trained on all 580 official bio pairs, with all 263 original species names included in the tokenizer-training corpus, in addition to Rare/Epic titles.
- `*_train.log` — raw training logs.
- `tokenization_comparison.json` — calculations and caveats comparing the older Free model.
- `internal_tests/` — minimal tokenizer test files from development.

**Source corpus not redistributed:** the official `bios_free.jsonl`, the original `en.utf8`, and `monsters.json` aren't duplicated in this archive. To retrain, copy `bios_free.jsonl` from the earlier AMI Bio archive beside the script, and optionally copy `monsters.json` from `msm_name_dataset_v3.zip`. Alternatively regenerate with the included `prepare.py` using the original game data. The already-trained weights and tokenizers work without these files.

## Install and inference

```bash
pip install torch numpy sentencepiece
python unigram_bio.py generate "Bumbleler" --dir complete_model --seed 9
python unigram_bio.py generate "Ghazt" --dir complete_model --seed 28
python unigram_bio.py generate "Gadla" --dir heldout_model --seed 9
```

Use `--temperature 0.82` (default) and `--max-tokens 450` (safety fallback). The output reports whether generation actually stopped on the EOS token.

## Retraining

```bash
python unigram_bio.py train --steps 1200
python unigram_bio.py final-train --steps 950
```

All train/evaluation splits and RNG seeds are specified in the code. The command starts from random model weights, with no pretrained LLM and no synthetic biographies.

## Results

| Metric | Value |
|---|---:|
| Unigram vocabulary | 1,280 |
| Network parameters | 444,656 |
| Training biographies in held-out run | 494 |
| Validation biographies | 86 |
| Validation base species | 39 |
| Entire official biography corpus | 580 |
| Atomic monster-name tokens, held-out tokenizer | 196 |
| Atomic monster-name tokens, complete tokenizer | 234 |
| Best held-out checkpoint | 1,200 steps |
| Held-out bits per byte | 2.3330 |
| Wrong-vs-correct name loss difference | +0.00746 nats/token |
| Natural EOS endings in 18 sampled held-out generations | 18 |

**Comparison caveat:** the older Free Bio model with the original BPE tokenizer had roughly **2.169 bits/byte** on these same held-out text rows (computed from the saved full-bio loss and the older tokenizer's target token count). That is lower (better) than 2.333, so the new tokenizer did **not** improve prediction. However, the model architectures, total training steps, tokenizer corpora, and name registration strategy differ; this is not a perfectly controlled tokenizer-only comparison. Model outputs remain mainly gibberish, and the difference in correct-versus-wrong name loss is small. This experiment tests name-aware tokenization, not strong creative writing.

The `complete_model/` checkpoints use training loss to select a full-corpus snapshot, rather than the held-out loss used for the evaluation model; no held-out metric can be assigned to the complete model.
