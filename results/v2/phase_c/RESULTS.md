# Protocol v2 Phase C: label coverage and counted supervision (UTS01)

Method fixed to the Phase A benchmark (raw pooled BOLD ridge, t+1…t+3). Each budget is crossed with two allocations of the same target count and five subset seeds (2718, 11–14). "Validation-selected" reuses the 794 validation labels for alpha; "CV-selected" chooses alpha by leave-one-story-out CV inside the labeled subset, so counted supervision equals the training labels. Test r is the session-macro feature correlation; ranges are observed minima/maxima over subset seeds. See [PROTOCOL_V2.md](../../../experiments/PROTOCOL_V2.md) §6.

100% benchmark (one fit): test r 0.1221; CV-selected 0.1221 (alphas 3.16e+04 / 3.16e+04).

| Labels | Allocation | Stories (mean) | Sessions (mean) | Val-selected test r | CV-selected test r | Counted labels (val-sel / CV) | Share of 100% r (val-sel) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1130 (25%) | block64 | 46.4 | 11.0 | 0.0758 [0.0728, 0.0795] | 0.0761 [0.0731, 0.0795] (n=5) | 1924 / 1130 | 62% |
| 1130 (25%) | story8 | 61.0 | 11.0 | 0.0804 [0.0783, 0.0836] | 0.0793 [0.0764, 0.0815] (n=5) | 1924 / 1130 | 66% |
| 452 (10%) | block64 | 25.2 | 10.4 | 0.0542 [0.0511, 0.0594] | 0.0542 [0.0511, 0.0592] (n=5) | 1246 / 452 | 44% |
| 452 (10%) | story8 | 61.0 | 11.0 | 0.0548 [0.0479, 0.0588] | 0.0547 [0.0479, 0.0588] (n=5) | 1246 / 452 | 45% |
| 226 (5%) | block64 | 14.4 | 8.6 | 0.0398 [0.0358, 0.0426] | 0.0393 [0.0351, 0.0426] (n=5) | 1020 / 226 | 33% |
| 226 (5%) | story8 | 61.0 | 11.0 | 0.0398 [0.0343, 0.0447] | 0.0387 [0.0319, 0.0430] (n=5) | 1020 / 226 | 33% |
| 46 (1%) | block64 | 3.0 | 2.8 | 0.0151 [0.0094, 0.0191] | 0.0126 [0.0089, 0.0194] (n=5) | 840 / 46 | 12% |
| 46 (1%) | story8 | 24.2 | 10.8 | 0.0158 [0.0081, 0.0213] | 0.0155 [0.0084, 0.0215] (n=5) | 840 / 46 | 13% |

Useful label savings would require matching the 100% benchmark with fewer counted labels under a prespecified noninferiority margin; none was prespecified, so this table reports shares rather than a savings claim.
