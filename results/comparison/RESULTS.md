# Measured SSL comparison

Mean over held-out runs, then seeds. Full seed rows and paired effects are in CSV files.

| Subject | Labels | Seeds per arm | Random r | SSL r | Δ r | Δ retrieval percentile |
|---|---:|---:|---:|---:|---:|---:|
| UTS01 | 100% | 3/3 | 0.0529 | 0.0689 | +0.0160 | +0.0127 |
| UTS01 | 25% | 3/3 | 0.0121 | 0.0162 | +0.0041 | +0.0138 |
| UTS01 | 10% | 3/3 | 0.0083 | 0.0105 | +0.0022 | +0.0142 |
| UTS01 | 5% | 3/3 | 0.0040 | 0.0061 | +0.0021 | +0.0189 |
| UTS01 | 1% | 3/3 | 0.0008 | 0.0019 | +0.0012 | +0.0099 |

Missing experiments are omitted, never filled with simulated scores. Validation labels are fixed additional supervision beyond the displayed training budget. Raw GPT2 cosine is anisotropic; see constant-predictor control and multiple metrics. Only two held-out sessions in the initial split limit uncertainty estimates. These are within-subject results, not population-level evidence.
