# Protocol v2 Phase A: measured readout baselines (UTS01)

Exploratory benchmark extension on already-examined test stories; see [PROTOCOL_V2.md](../../../experiments/PROTOCOL_V2.md). Test r is the session-macro feature correlation; validation selected every option and alpha. Seed ranges are observed minima/maxima, not confidence intervals. B1/B2 are deterministic single fits. Every arm shares 4,518/794/802 train/val/test targets; 794 validation labels are reused at every budget.

## 6-second cutoff, 100% labels (4518 training targets)

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | fir | 1 | 0.1110 | 0.1221 | 0.1208 | 0.754 / 0.751 | 0.0071 | delays=[1, 2, 3] |
| B1 raw BOLD | endpoint | 1 | 0.0903 | 0.0980 | 0.0972 | 0.699 / 0.698 | 0.0073 | delays=[3] |
| B4 frozen recon encoder (best val) | fir | 3 | 0.0882 | 0.0984 [0.0970, 0.1001] | 0.0974 | 0.710 / 0.708 | 0.0072 | layer=1 |
| B4 frozen recon encoder (final) | fir | 3 | 0.0867 | 0.0978 [0.0962, 0.1010] | 0.0968 | 0.707 / 0.705 | 0.0072 | layer=1 |
| B3 frozen random encoder | fir | 3 | 0.0860 | 0.0944 [0.0909, 0.0965] | 0.0933 | 0.700 / 0.698 | 0.0072 | layer=1 |
| B4 frozen recon encoder (best val) | endpoint | 3 | 0.0729 | 0.0814 [0.0796, 0.0842] | 0.0809 | 0.668 / 0.667 | 0.0072 | layer=1 |
| B4 frozen recon encoder (final) | endpoint | 3 | 0.0717 | 0.0818 [0.0786, 0.0835] | 0.0814 | 0.667 / 0.666 | 0.0073 | layer=1 |
| B3 frozen random encoder | endpoint | 3 | 0.0669 | 0.0733 [0.0693, 0.0766] | 0.0728 | 0.655 / 0.654 | 0.0072 | layer=1 |
| B2 window PCA | window | 1 | 0.0574 | 0.0579 | 0.0572 | 0.655 / 0.654 | 0.0053 | k=2048,unwhitened |
| B5 fine-tuned, recon init | finetune | 3 | 0.0570 | 0.0658 [0.0426, 0.0790] | 0.0652 | 0.634 / 0.634 | 0.0062 | step 180, step 288 |
| B5 fine-tuned, random init | finetune | 3 | 0.0537 | 0.0638 [0.0403, 0.0787] | 0.0633 | 0.638 / 0.637 | 0.0060 | step 180, step 288 |

## 6-second cutoff, 10% labels (452 training targets)

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | fir | 1 | 0.0392 | 0.0528 | 0.0520 | 0.653 / 0.654 | 0.0054 | delays=[1, 2, 3] |
| B1 raw BOLD | endpoint | 1 | 0.0329 | 0.0442 | 0.0439 | 0.634 / 0.634 | 0.0051 | delays=[3] |
| B4 frozen recon encoder (final) | fir | 3 | 0.0312 | 0.0401 [0.0390, 0.0413] | 0.0394 | 0.610 / 0.610 | 0.0060 | layer=1 |
| B4 frozen recon encoder (best val) | fir | 3 | 0.0298 | 0.0394 [0.0377, 0.0409] | 0.0387 | 0.613 / 0.613 | 0.0057 | layer=1 |
| B3 frozen random encoder | fir | 3 | 0.0295 | 0.0395 [0.0369, 0.0408] | 0.0388 | 0.615 / 0.616 | 0.0057 | layer=1 |
| B4 frozen recon encoder (final) | endpoint | 3 | 0.0261 | 0.0365 [0.0312, 0.0403] | 0.0362 | 0.609 / 0.609 | 0.0055 | layer=1 |
| B4 frozen recon encoder (best val) | endpoint | 3 | 0.0260 | 0.0373 [0.0333, 0.0402] | 0.0369 | 0.611 / 0.611 | 0.0053 | layer=1 |
| B3 frozen random encoder | endpoint | 3 | 0.0230 | 0.0328 [0.0292, 0.0347] | 0.0326 | 0.603 / 0.603 | 0.0051 | layer=1 |
| B5 fine-tuned, recon init | finetune | 3 | 0.0091 | 0.0102 [0.0095, 0.0113] | 0.0101 | 0.563 / 0.563 | 0.0027 | step 36, step 40, step 44 |
| B2 window PCA | window | 1 | 0.0090 | 0.0160 | 0.0157 | 0.549 / 0.548 | 0.0052 | k=2048,whitened |
| B5 fine-tuned, random init | finetune | 3 | 0.0073 | 0.0083 [0.0076, 0.0089] | 0.0081 | 0.549 / 0.549 | 0.0025 | step 36 |

## 8-second cutoff, 100% labels (4518 training targets)

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | fir | 1 | 0.1078 | 0.1174 | 0.1161 | 0.747 / 0.744 | 0.0069 | delays=[1, 2, 3, 4] |
| B4 frozen recon encoder (best val) | fir | 3 | 0.0879 | 0.0949 [0.0932, 0.0974] | 0.0940 | 0.705 / 0.703 | 0.0071 | layer=1 |
| B4 frozen recon encoder (final) | fir | 3 | 0.0872 | 0.0951 [0.0934, 0.0985] | 0.0942 | 0.698 / 0.696 | 0.0075 | layer=1 |
| B3 frozen random encoder | fir | 3 | 0.0869 | 0.0930 [0.0887, 0.0954] | 0.0921 | 0.698 / 0.696 | 0.0070 | layer=1 |
| B2 window PCA | window | 1 | 0.0611 | 0.0594 | 0.0588 | 0.653 / 0.652 | 0.0057 | k=2048,unwhitened |
| B1 raw BOLD | endpoint | 1 | 0.0526 | 0.0554 | 0.0552 | 0.627 / 0.626 | 0.0066 | delays=[4] |
| B4 frozen recon encoder (best val) | endpoint | 3 | 0.0455 | 0.0454 [0.0432, 0.0481] | 0.0455 | 0.606 / 0.605 | 0.0060 | layer=1, layer=2 |
| B4 frozen recon encoder (final) | endpoint | 3 | 0.0448 | 0.0439 [0.0419, 0.0458] | 0.0440 | 0.600 / 0.599 | 0.0063 | layer=1 |
| B3 frozen random encoder | endpoint | 3 | 0.0424 | 0.0446 [0.0402, 0.0471] | 0.0445 | 0.606 / 0.606 | 0.0059 | layer=1 |
| B5 fine-tuned, recon init | finetune | 3 | 0.0211 | 0.0235 [0.0131, 0.0370] | 0.0235 | 0.569 / 0.569 | 0.0035 | step 144, step 180, step 288 |
| B5 fine-tuned, random init | finetune | 3 | 0.0202 | 0.0241 [0.0177, 0.0369] | 0.0240 | 0.571 / 0.572 | 0.0035 | step 180, step 288 |

## 8-second cutoff, 10% labels (452 training targets)

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | fir | 1 | 0.0383 | 0.0510 | 0.0503 | 0.649 / 0.649 | 0.0056 | delays=[1, 2, 3, 4] |
| B4 frozen recon encoder (final) | fir | 3 | 0.0312 | 0.0399 [0.0385, 0.0407] | 0.0393 | 0.611 / 0.611 | 0.0060 | layer=1 |
| B3 frozen random encoder | fir | 3 | 0.0306 | 0.0400 [0.0382, 0.0412] | 0.0395 | 0.617 / 0.618 | 0.0053 | layer=1 |
| B4 frozen recon encoder (best val) | fir | 3 | 0.0299 | 0.0388 [0.0360, 0.0406] | 0.0383 | 0.610 / 0.610 | 0.0057 | layer=1 |
| B1 raw BOLD | endpoint | 1 | 0.0163 | 0.0229 | 0.0230 | 0.575 / 0.575 | 0.0051 | delays=[4] |
| B4 frozen recon encoder (final) | endpoint | 3 | 0.0146 | 0.0184 [0.0181, 0.0188] | 0.0183 | 0.555 / 0.555 | 0.0052 | layer=1, layer=2 |
| B4 frozen recon encoder (best val) | endpoint | 3 | 0.0144 | 0.0203 [0.0192, 0.0213] | 0.0205 | 0.558 / 0.558 | 0.0049 | layer=1, layer=2, layer=4 |
| B3 frozen random encoder | endpoint | 3 | 0.0143 | 0.0197 [0.0180, 0.0207] | 0.0199 | 0.561 / 0.562 | 0.0048 | layer=1, layer=4 |
| B2 window PCA | window | 1 | 0.0100 | 0.0123 | 0.0120 | 0.541 / 0.542 | 0.0054 | k=2048,whitened |
| B5 fine-tuned, recon init | finetune | 3 | 0.0047 | 0.0054 [0.0038, 0.0066] | 0.0055 | 0.535 / 0.535 | 0.0026 | step 36, step 40, step 44 |
| B5 fine-tuned, random init | finetune | 3 | 0.0040 | 0.0050 [0.0041, 0.0065] | 0.0051 | 0.534 / 0.535 | 0.0024 | step 36 |

## Label budgets (6-second cutoff; test r, mean over fits)

| Arm | Readout | 100% | 25% | 10% | 5% | 1% |
|---|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | fir | 0.1221 | 0.0728 | 0.0528 | 0.0392 | 0.0178 |
| B2 window PCA | window | 0.0579 | 0.0321 | 0.0160 | 0.0141 | 0.0036 |
| B3 frozen random encoder | fir | 0.0944 | 0.0591 | 0.0395 | 0.0271 | 0.0134 |
| B4 frozen recon encoder (best val) | fir | 0.0984 | 0.0614 | 0.0394 | 0.0265 | 0.0136 |
| B4 frozen recon encoder (final) | fir | 0.0978 | 0.0618 | 0.0401 | 0.0265 | 0.0168 |
| B5 fine-tuned, random init | finetune | 0.0638 | — | 0.0083 | — | — |
| B5 fine-tuned, recon init | finetune | 0.0658 | — | 0.0102 | — | — |

## Paired story-level contrasts (6-second cutoff)

Story differences are averaged over paired seeds before an exact sign-flip test over the 11 test stories. Stories share two sessions, so p-values are descriptive.

| Contrast (a − b) | Labels | Pairs | Δ test r | Per pair | Per session | Stories a > b | Sign-flip p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen recon (final) − frozen random | 1 | 3 | +0.0035 | +0.0054, -0.0004, +0.0053 | 14: +0.0041, 4: +0.0028 | 9/11 | 0.047 |
| Frozen recon (final) − raw BOLD | 1 | 3 | -0.0243 | -0.0211, -0.0260, -0.0259 | 14: -0.0208, 4: -0.0278 | 0/11 | 0.001 |
| Frozen random − raw BOLD | 1 | 3 | -0.0278 | -0.0266, -0.0256, -0.0312 | 14: -0.0249, 4: -0.0306 | 0/11 | 0.001 |
| Fine-tuned recon init − random init | 1 | 3 | +0.0019 | +0.0064, -0.0029, +0.0023 | 14: +0.0019, 4: +0.0019 | 7/11 | 0.131 |
| Raw BOLD − fine-tuned recon init | 1 | 3 | +0.0564 | +0.0432, +0.0464, +0.0795 | 14: +0.0484, 4: +0.0644 | 11/11 | 0.001 |
| Raw BOLD FIR − endpoint | 1 | 1 | +0.0241 | +0.0241 | 14: +0.0187, 4: +0.0296 | 11/11 | 0.001 |
| Frozen recon (final) − frozen random | 0.1 | 3 | +0.0006 | -0.0018, -0.0006, +0.0044 | 14: +0.0002, 4: +0.0010 | 4/11 | 0.732 |
| Frozen recon (final) − raw BOLD | 0.1 | 3 | -0.0127 | -0.0138, -0.0127, -0.0115 | 14: -0.0116, 4: -0.0138 | 0/11 | 0.001 |
| Frozen random − raw BOLD | 0.1 | 3 | -0.0133 | -0.0120, -0.0121, -0.0159 | 14: -0.0118, 4: -0.0148 | 0/11 | 0.001 |
| Fine-tuned recon init − random init | 0.1 | 3 | +0.0019 | +0.0020, +0.0030, +0.0006 | 14: +0.0027, 4: +0.0011 | 8/11 | 0.064 |
| Raw BOLD − fine-tuned recon init | 0.1 | 3 | +0.0426 | +0.0432, +0.0415, +0.0433 | 14: +0.0351, 4: +0.0502 | 11/11 | 0.001 |
| Raw BOLD FIR − endpoint | 0.1 | 1 | +0.0086 | +0.0086 | 14: +0.0033, 4: +0.0138 | 10/11 | 0.027 |

