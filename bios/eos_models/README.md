# AMI Bio v2 — real end-of-biography stopping 🧬

The **Free Bio** and **Template Bio** generators still accept *only a monster name*. This update makes their stopping behavior depend on a learned **true end-of-biography** marker, rather than training them to end after an artificial cutoff.

## What was already present, and what needed fixing

The previous version already defined `</s>` as the SentencePiece end token (ID 2), added it to targets, and stopped generation if sampled. However, `train.py` truncated each official biography to **at most 147 content tokens** and **then** appended the end token. This caused **493 of 580 biographies** to receive a false early ending. Generation also had a relatively short 180-token safety cap.

In v2, `train_eos.py` uses the **complete, unchanged** official biography text. It adds `</s>` only at the genuine final position. Training draws fixed-size windows from full examples for CPU efficiency, often sampling their actual endings; nonfinal windows do not include the end token. The two existing models are further trained, starting from the original checkpoints; they are not newly initialized. All target text comes from the originally supplied `en.utf8` file, and no extra invented vocabulary or bios were introduced.

The model's terminal token is still SentencePiece `</s>` (ID 2), **logically labeled END_BIO**. This is a *real dedicated model token*, not a literal punctuation mark embedded into the biography. We intentionally kept the vocabulary unchanged so existing weights were compatible.

`generate.py` now samples one token at a time until it predicts `</s>`, or until the **emergency** cap (`--max-tokens`, default 1024) is reached. Output reports `stop_reason`, `ended_naturally`, and the generated token count, so the two conditions cannot be confused.

## Data and architecture

- Original official monster biographies: **580 pairs**, covering base species and some Rare/Epic versions.
- Previous training procedure falsely truncated **493/580** examples. Full token lengths (not including BOS/EOS) ranged from **61–824**, with median **213.5**.
- The tokenizer is the same original 640-piece BPE, including the one-piece `[monster name]` symbol used by Template Bio.
- Both models remain the original ~380K-parameter name-conditioned GRU architecture.
- Original held-out split is preserved (by base species): **494 train, 86 validation**.
- Trained both variants for **500 additional steps** of true-ending continuation, starting from the previous best checkpoints.

## Measured held-out results

| Variant | Full-biography loss before | Full-biography loss after | END probability at real ending before | After |
| --- | ---: | ---: | ---: | ---: |
| Free | 4.0684 | 3.8585 | 14.87% | 39.96% |
| Template | 3.8952 | 3.7009 | 10.98% | 36.39% |

END probabilities above are evaluated across **36 withheld biographies with >148 tokens**, and represent mean predicted probability at the *true* final position. They are not probabilities that a generated paragraph is coherent. The full-biography loss uses all 86 held-out examples.

In an eight-prompt sampling audit per model and checkpoint (1000-token safety cap), **all 8/8 samples stopped by the end token**, both before and after this correction. The mean generated lengths were:

| Model | Original | Corrected |
| --- | ---: | ---: |
| Free | 145.4 | 187.0 tokens |
| Template | 169.0 | 121.5 tokens |

This is a small sample and does **not** indicate guaranteed desirable length. The corrected Template model still occasionally stops too early, while the Free model may ramble. This corrects the training signal, not the fundamental fluency problem of a tiny model trained on 580 biographies.

## Running the models

Install dependencies:

```bash
pip install -r requirements.txt
```

Free text:

```bash
python generate.py "Bumbleler" --model free --seed 9
```

Placeholder text (automatically rendered as the input monster name after generating):

```bash
python generate.py "Bumbleler" --model template --seed 9
```

The `raw_bio` field still retains the `[monster name]` placeholder. The `bio` field contains the rendered version. Both JSON results include `stop_reason: "end_of_bio_token"` when `</s>` is sampled. If a model loops for too long, `stop_reason` becomes `"max_tokens"` instead.

Full three-stage pipeline (name -> islands + both bios):

```bash
python full_chain.py Fire Poison --class Mythical --n 1 --seed 17
```

Repeat the true-ending fine-tuning (from the shipped original checkpoints):

```bash
python train_eos.py --variant both --steps 500
python audit_eos.py
```

## File guide

- `free/best.pt`, `template/best.pt`: the updated trained models
- `original_checkpoints/`: the previous two models, for comparisons
- `train_eos.py`: continuation trainer with true-end targets and full-text held-out validation
- `generate.py`, `generate_bio.py`: EOS-aware generation (the latter is named separately to avoid an import collision with the name model)
- `model.py`: the model architecture
- `audit_eos.py`, `eos_audit.json`: audit of stop behavior and probabilities with before/after samples
- `free/eos_metrics.json`, `template/eos_metrics.json`: training curves and evaluation metrics
- `bios_free.jsonl`, `bios_template.jsonl`, `msm_bio_sp.*`: original datasets and tokenizer
- `source/` and `prepare.py`: user-supplied localization and original extraction helper
- `previous_models/` and `full_chain.py`: complete name-and-island-to-bio pipeline
- `full_chain_eos_example.json`: verified demonstration

These experimental bios are **generated** text, not official canon. This is a small from-scratch language model; neither fluency nor lore accuracy is reliable yet.
