# Held-out modality transfer

Frozen models; all training and model selection used ds003020. These are run-level reference matching scores, not recovered imagined words or time-aligned movie descriptions.

| Condition | Training labels | Seeds per arm | Random centered top-1 | SSL centered top-1 | Chance |
|---|---:|---:|---:|---:|---:|
| imagined_speech | 100% | 3/3 | 26.7% | 36.7% | 20.0% |
| imagined_speech | 25% | 3/3 | 36.7% | 26.7% | 20.0% |
| imagined_speech | 10% | 3/3 | 23.3% | 30.0% | 20.0% |
| imagined_speech | 5% | 3/3 | 16.7% | 16.7% | 20.0% |
| imagined_speech | 1% | 3/3 | 23.3% | 23.3% | 20.0% |
| perceived_movie | 100% | 3/3 | 44.4% | 33.3% | 33.3% |
| perceived_movie | 25% | 3/3 | 22.2% | 33.3% | 33.3% |
| perceived_movie | 10% | 3/3 | 22.2% | 33.3% | 33.3% |
| perceived_movie | 5% | 3/3 | 22.2% | 22.2% | 33.3% |
| perceived_movie | 1% | 3/3 | 33.3% | 33.3% | 33.3% |

Only five imagined narratives (two repeats each) and three scored films are available. Repeats move together in exact label permutations. Individual-model permutation p-values, raw-space retrieval, cosine, MRR and constant-training-mean controls are in per_seed.csv and the source JSON files. P-values are not averaged across seeds. Sintel is unscored because its reference transcript is absent. Error bars show seed ranges, not confidence intervals. This transfer is zero-shot with respect to neural training; external GPT2 corpus membership cannot be verified.
