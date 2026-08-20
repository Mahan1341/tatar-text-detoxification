# Recovered Experiment History

This document records what could be established from the surviving AI-ZAMAN 2025 notebooks, scripts, checkpoints and generated TSV files.

It is intentionally separate from the cleaned implementation in `src/`.

## 1. Synthetic corpus construction

The recovered notebook loaded the `news_clean` split of `veryrealtatarperson/tt-azatliq-crawl`.

Recovered counts:

- source articles: 72,077;
- extracted sentences: 1,664,150;
- filtered sentences: 1,057,652;
- sampled clean sentences: 200,000;
- synthetic toxic/clean pairs: 200,000.

The historical filtering constrained sentence length, word count, Tatar-specific characters, noisy symbols, repeated content and some bureaucratic vocabulary.

The synthetic corruption process then created a toxic input from every sampled clean sentence. It could:

- insert or replace a token with a toxic expression;
- prepend or append an aggressive phrase;
- uppercase a token;
- append aggressive punctuation;
- swap adjacent words;
- insert a second toxic expression;
- distort spacing;
- add pseudo-noise such as repeated vowels;
- insert an aggressive phrase in the middle.

The untouched clean sentence was used as the target.

## 2. Historical mT5 training

Model: `google/mt5-small`.

Recovered final training-script configuration:

- maximum input/target length: 128;
- task prefix: `detox: `;
- planned epochs: 3;
- learning rate: `3e-4`;
- warmup: 1,000 steps;
- per-device train batch size: 4;
- gradient accumulation: 8;
- effective batch size: 32;
- linear learning-rate scheduler;
- AdamW;
- bf16 enabled;
- evaluation every 500 steps;
- checkpoint every 500 steps;
- best model selected by `eval_loss`.

Recovered evaluation losses:

| Step | Eval loss |
| ---: | ---: |
| 500 | 2.22466 |
| 1000 | 0.66667 |
| 1500 | 0.50510 |
| 2000 | **0.48946** |
| 2500 | 0.49757 |
| 3000 | 0.49603 |
| 3500 | 0.52193 |
| 4000 | 0.51306 |
| 4500 | 0.51677 |
| 5000 | 0.50373 |
| 6000 | 0.51855 |
| 6500 | 0.54079 |

The saved trainer state indicates that step 2,000 remained the best checkpoint. Training continued to at least step 6,500 (~1.09 epochs) and did not reach the planned three epochs.

The historical preprocessing padded targets to maximum length but did not replace existing target pad-token IDs by `-100`. That means padding likely contributed to the loss. The cleaned implementation uses dynamic sequence-to-sequence collation so padded label positions are ignored correctly.

## 3. Recovered development outputs

Several 600-row outputs on the same development input set were recovered.

### Early submission

A first output was saved in both two-column and three-column submission formats on November 29, 2025.

The exact generation implementation was not preserved.

### Raw neural output

A later output was present under several duplicate filenames/ZIP archives. The files were byte-identical. Qualitative inspection showed obvious generation artifacts, including repeated phrases.

### `output_final.tsv`

A distinct 600-row output was created shortly after the historical `infer_tsv.py` modification time. Its exact checkpoint/decoding configuration cannot be proven from the surviving source code, so the cleaned repository does not assign it a configuration that cannot be verified.

### Tatar2Vec + XGBoost + lexical cleaner

A separate branch used:

1. pretrained Tatar Word2Vec embeddings;
2. mean pooling over in-vocabulary word vectors;
3. XGBoost binary classification as a toxicity gate;
4. a lexical replace/delete cleaner when the classifier predicted toxicity.

The historical implementation constructed the binary classifier examples by concatenating toxic and detox texts before a random train/test split. Because paired examples were not kept together during splitting, this evaluation design was weak. The cleaned reconstruction instead splits pairs first and then expands each split into toxic/clean classification examples.

## 4. Recovered final test inference

A separate 701-row test set was recovered.

The surviving `detoxifier.py` shows a final historical pipeline using checkpoint 2,000 with beam-search generation followed by text normalization and an aggressive lexical blacklist.

Recovered output analysis:

- rows: 701;
- outputs changed relative to inputs: 642 (91.6%);
- unchanged: 59 (8.4%);
- empty outputs: 0.

Those numbers describe editing behavior only. They are not task-quality metrics because no gold targets for the test set were available locally.

## 5. What was not recovered

The following should not be claimed:

- an official Codabench score;
- an official final rank;
- the exact metric implementation used by the competition platform;
- a byte-for-byte reproducible copy of every historical submission;
- the exact implementation that produced `output_final.tsv`.

## 6. Main technical lessons

The reconstruction exposed several useful engineering lessons:

- low-resource NLP often requires synthetic supervision, but synthetic distributions can bias the task strongly toward lexical cues;
- corruption that replaces a meaningful source token can create impossible reconstruction targets and encourage hallucination;
- simple lexical systems are important baselines when toxicity is primarily injected through explicit markers;
- evaluation splits should preserve paired structure to prevent leakage;
- target padding must be masked properly in sequence-to-sequence training;
- aggressive post-processing can improve toxicity removal while harming semantic preservation.

These lessons are part of why the project is kept as a portfolio piece rather than presented only as a final model score.
