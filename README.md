# Tatar Text Detoxification

**Solo low-resource NLP project for toxic-to-neutral rewriting in Tatar.**

Originally developed for [ИИ-ЗАМАН Хак 2025](https://airi.net/ru/hackathon/aizaman-2025/), the project covers the full pipeline from data construction to model training, inference, baselines and retrospective evaluation.

## Highlights

- built **200,000 synthetic toxic → clean pairs** from a large clean Tatar news corpus;
- fine-tuned **`google/mt5-small`** for sequence-to-sequence detoxification;
- implemented lexical and **Tatar2Vec + XGBoost** baselines;
- explored hybrid neural + rule-based inference;
- recovered and evaluated historical outputs on **600 human-written Tatar detoxification references**;
- reconstructed the original hackathon work into an installable package with CLI tools, tests and CI on Python 3.10/3.12.

```text
clean Tatar corpus
      |
      v
sentence extraction + filtering
      |
      v
synthetic toxicity injection
      |
      v
200k toxic -> clean pairs
      |
      +---------------------------+
      |                           |
      v                           v
  mT5-small              lightweight baselines
      |                  lexical / Tatar2Vec+XGB
      v                           |
 neural generation               |
      +------------+--------------+
                   v
          evaluation / hybrid cleanup
```

## Problem

The task is to rewrite toxic or aggressive Tatar text into a neutral form while preserving as much of the original meaning as possible.

For a low-resource language, the main difficulty is not only model choice but also the lack of large paired detoxification datasets. The project therefore treats **data construction as a first-class part of the solution** rather than assuming ready-made training pairs.

## Synthetic training data

The recovered historical notebook used the `news_clean` split of [`veryrealtatarperson/tt-azatliq-crawl`](https://huggingface.co/datasets/veryrealtatarperson/tt-azatliq-crawl).

| Stage | Recovered count |
| --- | ---: |
| Source articles | 72,077 |
| Extracted sentences | 1,664,150 |
| Filtered clean sentences | 1,057,652 |
| Synthetic toxic/clean pairs | **200,000** |

Clean sentences were corrupted with a mixture of:

- toxic-word insertion and replacement;
- aggressive phrases;
- capitalization and punctuation noise;
- local word swaps;
- spacing noise and conversational artifacts.

The original clean sentence was used as the target.

This strategy made mT5 fine-tuning possible without a large manually aligned Tatar detoxification corpus.

## Neural model

The main neural approach fine-tunes **mT5-small** with a `detox:` task prefix.

Recovered historical configuration:

- maximum sequence length: 128;
- learning rate: `3e-4`;
- train batch size: 4;
- gradient accumulation: 8;
- effective batch size: 32;
- linear learning-rate schedule;
- 1,000 warmup steps;
- planned training: 3 epochs.

The historical run continued to roughly step 6,500 (~1.09 epochs). The best saved checkpoint was **step 2,000**, selected by validation loss **0.4895**.

That value is a training loss used for checkpoint selection, **not an official hackathon score**.

## Baselines and model behavior

Two lightweight alternatives were explored alongside mT5:

1. **Lexical cleaner** — targeted deletion/replacement of known toxic expressions.
2. **Tatar2Vec + XGBoost gate** — mean Word2Vec sentence embeddings classify whether a sentence should be passed through the lexical cleaner.

Retrospective evaluation showed a useful low-resource NLP trade-off rather than a single universally best method:

- neural generation substantially changes the input and can remove toxic content that is difficult to handle with exact string rules;
- conservative lexical methods preserve the source text very well when toxicity is predominantly lexical;
- raw seq2seq generation can over-edit, so a hybrid or gated approach is often preferable when meaning preservation matters.

That observation motivated keeping both neural and lightweight approaches in the reconstructed project instead of presenting one model in isolation.

## Post-hoc benchmark on 600 human references

After the hackathon, a public Tatar detoxification reference set became available containing the **same 600/600 toxic inputs** as the recovered development files. This allowed the surviving systems to be evaluated against human-written detoxifications.

| System | chrF++ ↑ | Token F1 to gold ↑ | Source similarity ↑ | Rows changed | Toxic-lexicon proxy ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Identity | 80.09 | 0.7647 | 1.0000 | 0.0% | 17.5% |
| Lexical only | 80.08 | **0.7692** | 0.9808 | 22.3% | **0.0%** |
| Raw mT5 | 69.98 | 0.7095 | 0.8728 | 88.5% | 5.2% |
| Tatar2Vec + XGBoost + rules | **80.09** | 0.7685 | **0.9863** | 17.2% | 4.3% |

The table should be read as a **behavior analysis**, not as an official leaderboard reconstruction. In particular:

- human detoxifications are conservative, so unchanged input already receives a high reference-overlap score;
- `Toxic-lexicon proxy` only checks the small recovered historical blacklist and is **not** the official TextDetox toxicity classifier;
- the raw mT5 output shown here is a recovered historical inference variant, not the corrected reconstructed training pipeline.

The main takeaway is that **model complexity alone was not enough**: in this low-resource setting, preserving meaning required balancing neural rewriting with conservative editing.

Full methodology: [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

## Reconstruction improvements

The public code is a cleaned reconstruction, not a raw dump of the hackathon directory. Several issues discovered while recovering the project are fixed in the current implementation:

- target sequences are dynamically padded and padding labels use `-100`, so padded positions do not contribute to seq2seq loss;
- paired examples are split before constructing the Word2Vec/XGBoost classification dataset, reducing pair leakage;
- training, inference, evaluation and baselines are separated into reusable modules;
- the lexical postprocessor no longer lowercases every generated sentence;
- competition inputs, generated submissions and model checkpoints are not committed to Git.

The historical experiment timeline and recovered checkpoint history are documented in [`docs/EXPERIMENT_HISTORY.md`](docs/EXPERIMENT_HISTORY.md).

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e .
```

## Build synthetic training data

```bash
tatar-detox-build \
  --output data/synthetic.jsonl \
  --sample-size 200000 \
  --seed 42
```

The reconstruction preserves the recovered corruption mechanism with a compact lexicon distilled from the historical lists. Exact historical intermediate datasets are not redistributed.

## Fine-tune mT5-small

```bash
tatar-detox-train \
  --train-data data/synthetic.jsonl \
  --output-dir checkpoints/mt5-detox \
  --epochs 3 \
  --learning-rate 3e-4 \
  --train-batch-size 4 \
  --gradient-accumulation-steps 8
```

If `--eval-data` is omitted, the script creates a deterministic 5% validation split. On compatible hardware, add `--bf16` to reproduce the historical precision choice.

## Run neural inference

```bash
tatar-detox-infer \
  --model checkpoints/mt5-detox \
  --input input.tsv \
  --output outputs/mt5.tsv
```

Optional hybrid post-processing:

```bash
tatar-detox-infer \
  --model checkpoints/mt5-detox \
  --input input.tsv \
  --output outputs/hybrid.tsv \
  --lexical-postprocess
```

## Run baselines

Lexical baseline:

```bash
tatar-detox-lexical \
  --input input.tsv \
  --output outputs/lexical.tsv
```

Tatar2Vec + XGBoost gate:

```bash
tatar-detox-w2v-xgb \
  --pairs data/synthetic.jsonl \
  --word2vec path/to/tatar2vec.model \
  --input input.tsv \
  --output outputs/w2v_xgb.tsv
```

The pretrained Tatar2Vec model is not distributed in this repository.

## Evaluate against paired references

```bash
tatar-detox-evaluate \
  --reference reference.tsv \
  --prediction outputs/mt5.tsv
```

The evaluator reports exact match, chrF++, character similarity, token F1, source similarity, changed-row fraction and mean output/input length ratio.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── docs/
│   ├── BENCHMARK.md
│   └── EXPERIMENT_HISTORY.md
├── src/tatar_detox/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── lexicons.py
│   ├── synthetic.py
│   ├── train.py
│   └── baselines/
│       ├── __init__.py
│       ├── lexical.py
│       └── word2vec_xgb.py
├── tests/
├── .gitignore
└── pyproject.toml
```

## Status

The historical experiment was reconstructed from surviving notebooks, scripts, checkpoints and generated TSV files. The current code is a reproducible cleanup rather than a claim that the original hackathon environment can be reproduced byte-for-byte.

No reliable official Codabench score or final leaderboard position was recovered, so the repository does not claim one.

## Author

Vladimir Garanin — Innopolis University
