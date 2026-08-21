# Tatar Text Detoxification

Low-resource NLP project for rewriting toxic Tatar text into a more neutral form while preserving its meaning.

The original system was developed independently for **ИИ-ЗАМАН Хак 2025**. This repository is a cleaned reconstruction of that work: historical experiments are documented separately, while the runnable code fixes issues discovered during recovery.

## Overview

The project explores several approaches to Tatar text detoxification:

- synthetic toxic/clean pair generation from a clean Tatar corpus;
- `google/mt5-small` fine-tuning for sequence-to-sequence detoxification;
- a lexical delete/replace baseline;
- a Tatar2Vec mean-embedding + XGBoost toxicity gate;
- hybrid mT5 inference followed by lexical post-processing.

```text
clean Tatar corpus
      |
      v
synthetic toxicity injection
      |
      v
200k toxic -> clean pairs
      |
      +-----------------------+
      |                       |
      v                       v
  mT5-small          Tatar2Vec + XGBoost
      |                       |
      v                       v
 generation            lexical cleaner
      |
      v
optional lexical post-processing
```

## Recovered historical experiment

The surviving notebook and local artifacts establish the following data pipeline:

| Stage | Recovered count |
| --- | ---: |
| Source articles | 72,077 |
| Extracted sentences | 1,664,150 |
| Filtered clean sentences | 1,057,652 |
| Synthetic toxic/clean pairs | 200,000 |

The clean source was the `news_clean` split of [`veryrealtatarperson/tt-azatliq-crawl`](https://huggingface.co/datasets/veryrealtatarperson/tt-azatliq-crawl). Synthetic corruption included toxic word insertion/replacement, aggressive phrases, capitalization, punctuation noise, local word swaps and spacing noise.

The recovered mT5 training run used:

- maximum sequence length: 128;
- learning rate: `3e-4`;
- train batch size: 4;
- gradient accumulation: 8;
- effective batch size: 32;
- linear LR schedule;
- 1,000 warmup steps;
- planned training: 3 epochs.

The run stopped at roughly step 6,500 (~1.09 epochs). The best saved checkpoint was **step 2,000**, selected by validation loss `0.4895`.

That value is a recovered training loss, **not an official hackathon score**.

## Post-hoc benchmark on 600 public Tatar references

A later-publicly available Tatar detoxification reference set contains the same **600/600 toxic inputs** as the recovered development files. This makes it possible to evaluate the surviving outputs retrospectively against human-written detoxifications.

| System | chrF++ ↑ | Token F1 to gold ↑ | Source similarity ↑ | Rows changed | Toxic-lexicon proxy ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Identity | 80.09 | 0.7647 | 1.0000 | 0.0% | 17.5% |
| Lexical only | 80.08 | **0.7692** | 0.9808 | 22.3% | **0.0%** |
| Raw mT5 | 69.98 | 0.7095 | 0.8728 | 88.5% | 5.2% |
| Tatar2Vec + XGBoost + rules | **80.09** | 0.7685 | **0.9863** | 17.2% | 4.3% |

The benchmark exposes the main engineering trade-off clearly. The recovered raw mT5 pipeline edits aggressively and removes much lexical toxicity, but it also changes far more text and agrees less with the human references. The small lexical baseline is much more conservative and, on these reference-based diagnostics, is surprisingly competitive.

The human references themselves are conservative edits, so the unchanged input already receives a high chrF++ score. **chrF++ is therefore not presented as a standalone detoxification score.** The `Toxic-lexicon proxy` is only the percentage of rows containing terms from the small recovered historical blacklist; it is not the official TextDetox toxicity classifier.

Full methodology and caveats: [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

## Why synthetic data?

A large paired Tatar detoxification corpus was not available during the hackathon. The workaround was to start from clean Tatar sentences and corrupt them synthetically, using the original sentence as the target.

This made fine-tuning possible but introduced important limitations:

- formal news text differs from toxic user-generated text;
- replacing an original word destroys information that the target still asks the model to reconstruct;
- synthetic toxicity is more lexical and predictable than real-world toxicity;
- dictionary-based cleaning can therefore be unusually competitive.

These limitations are documented rather than hidden because they materially affect the result.

## Clean reconstruction vs. historical code

The runnable implementation fixes several issues found during recovery:

- target sequences are dynamically padded and padding labels use `-100`, so padded positions do not contribute to seq2seq loss;
- paired examples are split before constructing the Word2Vec/XGBoost classification dataset, reducing pair leakage;
- training, inference, evaluation and baselines are separate modules;
- the lexical postprocessor no longer lowercases every generated sentence;
- competition inputs, generated submissions and model checkpoints are not committed to Git.

For the original experiment timeline and recovered checkpoint history, see [`docs/EXPERIMENT_HISTORY.md`](docs/EXPERIMENT_HISTORY.md).

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

The public reconstruction preserves the recovered corruption mechanism with a compact lexicon distilled from the historical lists. Exact historical intermediate datasets are not redistributed.

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
├── LICENSE
└── pyproject.toml
```

## Status

The historical experiment has been reconstructed from surviving notebooks, scripts, checkpoints and generated TSV files. The code here is a reproducible cleanup, not a claim that the original hackathon run can be reproduced byte-for-byte.

No reliable official Codabench score or final leaderboard position was recovered, so this repository does not claim one.

## Author

Vladimir Garanin — Innopolis University
