# Tatar Text Detoxification

Low-resource NLP project for rewriting toxic Tatar text into a more neutral form while preserving its meaning.

The original system was developed independently for **ИИ-ЗАМАН Хак 2025**. This repository is a cleaned reconstruction of that work: the historical experiments are documented, while the runnable code fixes issues discovered during recovery.

## What is in this project

The project explores several approaches to Tatar text detoxification:

- synthetic toxic/clean pair generation from a clean Tatar corpus;
- mT5-small fine-tuning for sequence-to-sequence detoxification;
- a lexical delete/replace baseline;
- a Tatar2Vec mean-embedding + XGBoost toxicity gate;
- hybrid mT5 inference followed by conservative lexical cleanup.

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

The surviving notebook and local artifacts establish the following historical pipeline:

| Stage | Recovered count |
| --- | ---: |
| Source articles | 72,077 |
| Extracted sentences | 1,664,150 |
| Filtered clean sentences | 1,057,652 |
| Synthetic toxic/clean pairs | 200,000 |

The clean source was the `news_clean` split of [`veryrealtatarperson/tt-azatliq-crawl`](https://huggingface.co/datasets/veryrealtatarperson/tt-azatliq-crawl). Toxicity was injected with a mixture of lexical substitutions/insertions, aggressive phrases, capitalization, punctuation noise, local word swaps and spacing noise.

The main neural model was `google/mt5-small`. The recovered training run used an effective batch size of 32, learning rate `3e-4`, a linear schedule with 1,000 warmup steps and a maximum sequence length of 128. Training was planned for three epochs but stopped at roughly step 6,500 (~1.09 epochs).

The best recovered checkpoint was **step 2,000**, selected by evaluation loss:

| Step | Eval loss |
| ---: | ---: |
| 500 | 2.2247 |
| 1000 | 0.6667 |
| 1500 | 0.5051 |
| 2000 | **0.4895** |
| 2500 | 0.4976 |
| 3000 | 0.4960 |
| 3500 | 0.5219 |
| 4000 | 0.5131 |
| 4500 | 0.5168 |
| 5000 | 0.5037 |
| 6000 | 0.5186 |
| 6500 | 0.5408 |

**This is a recovered training metric, not an official hackathon leaderboard score.** No reliable Codabench score or final rank was recovered, so this repository does not claim one.

More detail is available in [`docs/EXPERIMENT_HISTORY.md`](docs/EXPERIMENT_HISTORY.md).

## Why synthetic data?

A large paired Tatar detoxification corpus was not available during the hackathon. The practical workaround was to start from clean Tatar sentences and corrupt them synthetically.

The historical corruption process always inserted or replaced at least one token with a toxic expression, then probabilistically added conversational/aggressive noise. The original clean sentence became the target.

This makes training possible, but it also creates important limitations:

- news text is not the same domain as toxic user-generated text;
- replacing an original word destroys information that the target still asks the model to reconstruct;
- synthetic toxicity is more lexical and predictable than real-world toxicity;
- a dictionary-based cleaner can therefore be unusually competitive.

These limitations are part of the project rather than something hidden from the presentation.

## Clean reconstruction vs. historical code

The runnable code in this repository intentionally fixes several issues found in the recovered hackathon implementation:

- target padding is handled dynamically and ignored with label value `-100`;
- paired examples are split before constructing the Word2Vec/XGBoost classification dataset, avoiding pair leakage;
- training, inference and baselines are separate modules;
- the lexical postprocessor no longer lowercases every generated sentence;
- competition inputs, generated submissions and model checkpoints are not committed to Git.

For archaeology rather than current implementation details, see the experiment-history document.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e .
```

## 1. Build a synthetic dataset

```bash
tatar-detox-build --output data/synthetic.jsonl --sample-size 200000 --seed 42
```

The public reconstruction preserves the recovered corruption mechanism and a compact lexicon distilled from the historical lists. Exact historical intermediate datasets are not redistributed.

## 2. Fine-tune mT5-small

```bash
tatar-detox-train \
  --train-data data/synthetic.jsonl \
  --output-dir checkpoints/mt5-detox \
  --epochs 3 \
  --learning-rate 3e-4 \
  --train-batch-size 4 \
  --gradient-accumulation-steps 8
```

If `--eval-data` is omitted, the script creates a deterministic validation split from the synthetic pairs. An external paired evaluation JSONL can be passed explicitly when available.

To reproduce the historical mixed-precision choice on compatible hardware, add `--bf16`.

## 3. Run neural inference

The default TSV schema matches the recovered competition-style files: `ID`, `tat_toxic`, `tat_detox1`.

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

## 4. Run the lexical baseline

```bash
tatar-detox-lexical \
  --input input.tsv \
  --output outputs/lexical.tsv
```

## 5. Tatar2Vec + XGBoost experiment

A separate recovered branch used mean Word2Vec sentence embeddings as features for an XGBoost toxicity gate. The cleaned implementation trains on paired examples with pair-level splitting before classification examples are expanded.

```bash
tatar-detox-w2v-xgb \
  --pairs data/synthetic.jsonl \
  --word2vec path/to/tatar2vec.model \
  --input input.tsv \
  --output outputs/w2v_xgb.tsv
```

The pretrained Tatar2Vec model is not distributed in this repository.

## Repository structure

```text
.
├── docs/
│   └── EXPERIMENT_HISTORY.md
├── src/tatar_detox/
│   ├── __init__.py
│   ├── infer.py
│   ├── lexicons.py
│   ├── synthetic.py
│   ├── train.py
│   └── baselines/
│       ├── __init__.py
│       ├── lexical.py
│       └── word2vec_xgb.py
├── tests/
│   ├── test_lexical.py
│   └── test_synthetic.py
├── .gitignore
├── LICENSE
└── pyproject.toml
```

## Project status

The historical experiment has been reconstructed from surviving notebooks, scripts, checkpoints and generated TSV files. The code here is a reproducible cleanup, not a claim that the original hackathon run was fully reproducible byte-for-byte.

## Author

Vladimir Garanin — Innopolis University
