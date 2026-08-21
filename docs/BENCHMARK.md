# Post-hoc benchmark on public Tatar references

This benchmark evaluates recovered 600-row development outputs from the original hackathon work against a later-publicly available Tatar detoxification reference set.

The public reference file is `CLEF2025_dataset.tsv` from the Tatoxa repository. The recovered inputs match its `tat_toxic` column exactly for all **600/600** rows, which allows a retrospective comparison against human-written `tat_detox1` references.

This is **not** presented as the official AI-ZAMAN leaderboard metric. It is a post-hoc diagnostic used to understand the behavior of the recovered systems.

## Compared systems

- **Identity** — returns the toxic input unchanged.
- **Lexical only** — applies the recovered historical toxic-word delete/replace rules directly.
- **Raw mT5** — recovered generative mT5 output before the later hybrid post-processing branch.
- **Tatar2Vec + XGBoost + rules** — recovered lightweight branch that uses a Word2Vec/XGBoost toxicity gate and applies the lexical cleaner only when the gate fires.

## Results

| System | Exact gold | chrF++ ↑ | Char similarity to gold ↑ | Token F1 to gold ↑ | Source similarity ↑ | Rows changed | Toxic-lexicon proxy ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Identity | 0.17% | 80.09 | 0.8097 | 0.7647 | 1.0000 | 0.0% | 17.5% |
| Lexical only | 1.00% | 80.08 | 0.8116 | **0.7692** | 0.9808 | 22.3% | **0.0%** |
| Raw mT5 | **1.17%** | 69.98 | 0.7574 | 0.7095 | 0.8728 | 88.5% | 5.2% |
| Tatar2Vec + XGBoost + rules | 1.00% | **80.09** | **0.8118** | 0.7685 | **0.9863** | 17.2% | 4.3% |

## Interpretation

The human references are mostly conservative edits: even the unchanged toxic input obtains chrF++ 80.09 against the human detoxifications. Therefore **chrF++ alone is not a detoxification metric** here.

The useful signal is the trade-off between editing and preservation:

- the recovered **raw mT5** changes 88.5% of rows and removes many lexicon-detected toxic terms, but it also has the weakest reference similarity and exhibits the repetition artifacts visible in the recovered outputs;
- the **lexical-only baseline** changes only 22.3% of rows, removes all terms covered by the small historical toxic lexicon, and slightly improves reference similarity over the identity baseline;
- the **Tatar2Vec + XGBoost gate** is even more conservative, changing 17.2% of rows and preserving the source best among the non-identity systems, but its gate leaves some lexicon-covered toxicity untouched.

The result supports a practical lesson from the hackathon: on this small low-resource benchmark, model complexity did not automatically translate into a better detoxification/content-preservation trade-off. A simple targeted baseline was surprisingly competitive.

## Metric caveats

`Exact gold`, `chrF++`, character similarity and token F1 compare predictions with a single human reference. Detoxification can have many valid rewrites, so these should be interpreted as preservation/reference-agreement diagnostics rather than complete quality scores.

`Toxic-lexicon proxy` is the percentage of rows containing at least one substring from the small toxic lexicon recovered from the historical baseline. It is **not** the official TextDetox style-transfer classifier and must not be interpreted as true toxicity accuracy.

The original official Codabench score and leaderboard position were not recovered.
