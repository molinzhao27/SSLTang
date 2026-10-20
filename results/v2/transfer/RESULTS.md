# Protocol v2: ridge-decoder modality transfer (ds004510)

Decoder: each subject's Phase A benchmark (raw pooled BOLD ridge, t+1…t+3, 100% labels, alpha from ds003020 validation). Run vectors average predictions over every t with t+3 in the run; retrieval uses fixed whole-reference GPT-2 vectors. Narrative-macro scores with exact narrative-label permutation p-values (imagined: 120 permutations; films: 6, minimum p = 1/6). UTS01 is exploratory (its transfer data were viewed in the archived study); UTS02 and UTS03 follow the registered rule in [PROTOCOL_V2.md](../../../experiments/PROTOCOL_V2.md) §9.

| Subject | Condition | Centered top-1 (p) | Centered MRR (p) | Raw top-1 (p) | Constant-mean top-1 | Chance | Runs | TRs averaged per run |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| UTS01 | imagined_speech | 60% (0.017) | 0.728 (0.017) | 40% (0.150) | 20% | 20% | 10 | 27 |
| UTS01 | perceived_movie | 33% (1.000) | 0.611 (1.000) | 33% (1.000) | 33% | 33% | 3 | 124, 138, 172 |
| UTS02 | imagined_speech | 50% (0.067) | 0.673 (0.067) | 30% (0.333) | 20% | 20% | 10 | 27 |
| UTS02 | perceived_movie | 33% (1.000) | 0.611 (1.000) | 33% (1.000) | 33% | 33% | 3 | 124, 138, 172 |
| UTS03 | imagined_speech | 60% (0.033) | 0.753 (0.025) | 40% (0.150) | 20% | 20% | 10 | 27 |
| UTS03 | perceived_movie | 33% (0.667) | 0.611 (0.667) | 33% (1.000) | 33% | 33% | 3 | 124, 138, 172 |

Registered decision: imagined-speech transfer requires centered top-1 permutation p ≤ 0.05 in both UTS02 and UTS03. UTS02: not met, UTS03: met. Transfer is not established under the registered rule.

Archived UTS01 comparison (fine-tuned Transformers, 100% labels, mean of 3 seeds, [results/transfer_comparison](../../transfer_comparison/RESULTS.md)): centered imagined top-1 26.7% (random init) and 36.7% (SSL init); films 44.4% and 33.3%.
