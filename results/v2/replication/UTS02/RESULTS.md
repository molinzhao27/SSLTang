# Protocol v2 Phase B: latent prediction versus reconstruction (UTS02)

Exploratory (UTS01 test stories were examined before). Both objectives share windows, masks, encoder, optimizer and 1,110 updates; each exports its final student encoder. The locked readout is the Phase A frozen ridge procedure at the 6-s cutoff with FIR states (layer and alpha chosen per encoder on validation). Test r is the session-macro feature correlation. See [PROTOCOL_V2.md](../../../experiments/PROTOCOL_V2.md) §5.

## Paired story-level contrasts

Seeds are paired by pretraining seed; story differences are averaged over seeds before an exact sign-flip test over 11 test stories (stories share two sessions: descriptive only).

| Contrast (a − b) | Labels | Pairs | Δ test r | Per pair | Per session | Stories a > b | Sign-flip p |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRIMARY Latent − reconstruction (frozen, FIR) | 1 | 3 | -0.0110 | -0.0108, -0.0126, -0.0095 | 14: -0.0092, 4: -0.0128 | 1/11 | 0.004 |
| Latent − reconstruction (frozen, FIR) | 0.1 | 3 | -0.0110 | -0.0145, -0.0079, -0.0106 | 14: -0.0089, 4: -0.0131 | 0/11 | 0.001 |
| Latent − reconstruction (frozen, FIR) | 0.25 | 3 | -0.0165 | -0.0178, -0.0172, -0.0145 | 14: -0.0130, 4: -0.0201 | 1/11 | 0.002 |
| Latent − reconstruction (frozen, FIR) | 0.05 | 3 | -0.0107 | -0.0154, -0.0045, -0.0120 | 14: -0.0134, 4: -0.0079 | 0/11 | 0.001 |
| Latent − reconstruction (frozen, FIR) | 0.01 | 3 | -0.0059 | -0.0054, -0.0073, -0.0051 | 14: -0.0053, 4: -0.0066 | 2/11 | 0.093 |
| Latent − reconstruction (frozen, 8-s FIR) | 1 | 3 | -0.0120 | -0.0141, -0.0108, -0.0110 | 14: -0.0103, 4: -0.0136 | 1/11 | 0.002 |
| Latent − reconstruction (frozen, endpoint) | 1 | 3 | -0.0113 | -0.0128, -0.0104, -0.0106 | 14: -0.0126, 4: -0.0099 | 1/11 | 0.002 |
| Latent − reconstruction (fine-tuned) | 1 | 3 | -0.0094 | -0.0222, -0.0266, +0.0207 | 14: -0.0090, 4: -0.0098 | 0/11 | 0.001 |
| Latent − frozen random encoder | 1 | 3 | -0.0143 | -0.0125, -0.0147, -0.0156 | 14: -0.0113, 4: -0.0172 | 1/11 | 0.002 |
| Reconstruction (v2) − frozen random encoder | 1 | 3 | -0.0033 | -0.0017, -0.0021, -0.0061 | 14: -0.0021, 4: -0.0045 | 2/11 | 0.204 |
| Latent − raw BOLD benchmark | 1 | 3 | -0.0352 | -0.0356, -0.0351, -0.0349 | 14: -0.0290, 4: -0.0415 | 0/11 | 0.001 |
| Reconstruction (v2) − raw BOLD benchmark | 1 | 3 | -0.0243 | -0.0249, -0.0225, -0.0254 | 14: -0.0198, 4: -0.0288 | 0/11 | 0.001 |
| Latent − reconstruction (frozen, 8-s FIR) | 0.1 | 3 | -0.0120 | -0.0159, -0.0102, -0.0101 | 14: -0.0103, 4: -0.0138 | 0/11 | 0.001 |
| Latent − reconstruction (frozen, endpoint) | 0.1 | 3 | -0.0113 | -0.0151, -0.0064, -0.0125 | 14: -0.0161, 4: -0.0066 | 1/11 | 0.002 |
| Latent − reconstruction (fine-tuned) | 0.1 | 3 | -0.0062 | -0.0054, -0.0058, -0.0074 | 14: -0.0073, 4: -0.0051 | 2/11 | 0.034 |
| Latent − frozen random encoder | 0.1 | 3 | -0.0129 | -0.0142, -0.0123, -0.0123 | 14: -0.0102, 4: -0.0157 | 0/11 | 0.001 |
| Reconstruction (v2) − frozen random encoder | 0.1 | 3 | -0.0020 | +0.0003, -0.0044, -0.0018 | 14: -0.0013, 4: -0.0026 | 5/11 | 0.367 |
| Latent − raw BOLD benchmark | 0.1 | 3 | -0.0258 | -0.0265, -0.0245, -0.0263 | 14: -0.0231, 4: -0.0286 | 0/11 | 0.001 |
| Reconstruction (v2) − raw BOLD benchmark | 0.1 | 3 | -0.0148 | -0.0121, -0.0166, -0.0158 | 14: -0.0142, 4: -0.0155 | 0/11 | 0.001 |

## 6-second cutoff, fir readout, 100% labels

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | fir | 1 | 0.1209 | 0.1371 | 0.1361 | 0.776 / 0.772 | 0.0072 | delays=[1, 2, 3] |
| B3 frozen random encoder | fir | 3 | 0.1007 | 0.1161 [0.1139, 0.1178] | 0.1154 | 0.736 / 0.733 | 0.0076 | layer=1 |
| R frozen recon (v2 retrain) | fir | 3 | 0.0970 | 0.1128 [0.1117, 0.1146] | 0.1122 | 0.732 / 0.728 | 0.0073 | layer=1 |
| L frozen latent (EMA teacher) | fir | 3 | 0.0882 | 0.1018 [0.1015, 0.1021] | 0.1014 | 0.718 / 0.715 | 0.0068 | layer=1 |
| B5 fine-tuned, random init | finetune | 3 | 0.0580 | 0.0678 [0.0469, 0.0800] | 0.0674 | 0.652 / 0.653 | 0.0063 | step 180, step 288 |
| fine-tuned, v2 recon init | finetune | 3 | 0.0540 | 0.0644 [0.0471, 0.0758] | 0.0644 | 0.648 / 0.647 | 0.0059 | step 180, step 288 |
| fine-tuned, latent init | finetune | 3 | 0.0472 | 0.0551 [0.0438, 0.0679] | 0.0551 | 0.629 / 0.628 | 0.0064 | step 1044, step 288, step 900 |

## 6-second cutoff, fir readout, 10% labels

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | fir | 1 | 0.0533 | 0.0694 | 0.0693 | 0.677 / 0.675 | 0.0061 | delays=[1, 2, 3] |
| R frozen recon (v2 retrain) | fir | 3 | 0.0441 | 0.0546 [0.0528, 0.0573] | 0.0545 | 0.643 / 0.641 | 0.0062 | layer=1 |
| B3 frozen random encoder | fir | 3 | 0.0435 | 0.0565 [0.0554, 0.0572] | 0.0564 | 0.650 / 0.648 | 0.0057 | layer=1 |
| L frozen latent (EMA teacher) | fir | 3 | 0.0368 | 0.0436 [0.0429, 0.0449] | 0.0437 | 0.619 / 0.618 | 0.0057 | layer=1 |
| fine-tuned, v2 recon init | finetune | 3 | 0.0059 | 0.0117 [0.0109, 0.0126] | 0.0117 | 0.564 / 0.563 | 0.0026 | step 40 |
| B5 fine-tuned, random init | finetune | 3 | 0.0045 | 0.0088 [0.0080, 0.0093] | 0.0088 | 0.551 / 0.549 | 0.0026 | step 32, step 36 |
| fine-tuned, latent init | finetune | 3 | 0.0032 | 0.0054 [0.0051, 0.0057] | 0.0053 | 0.524 / 0.526 | 0.0027 | step 48 |

## 6-second cutoff, endpoint readout, 100% labels

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | endpoint | 1 | 0.0821 | 0.1015 | 0.1010 | 0.712 / 0.709 | 0.0073 | delays=[3] |
| R frozen recon (v2 retrain) | endpoint | 3 | 0.0676 | 0.0858 [0.0854, 0.0862] | 0.0855 | 0.677 / 0.675 | 0.0071 | layer=1 |
| B3 frozen random encoder | endpoint | 3 | 0.0658 | 0.0812 [0.0779, 0.0831] | 0.0808 | 0.673 / 0.671 | 0.0075 | layer=1 |
| L frozen latent (EMA teacher) | endpoint | 3 | 0.0614 | 0.0745 [0.0734, 0.0751] | 0.0741 | 0.667 / 0.665 | 0.0064 | layer=1 |

## 6-second cutoff, endpoint readout, 10% labels

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | endpoint | 1 | 0.0336 | 0.0454 | 0.0452 | 0.622 / 0.620 | 0.0058 | delays=[3] |
| R frozen recon (v2 retrain) | endpoint | 3 | 0.0275 | 0.0376 [0.0360, 0.0400] | 0.0376 | 0.601 / 0.599 | 0.0055 | layer=1 |
| B3 frozen random encoder | endpoint | 3 | 0.0237 | 0.0338 [0.0327, 0.0344] | 0.0335 | 0.595 / 0.594 | 0.0056 | layer=1 |
| L frozen latent (EMA teacher) | endpoint | 3 | 0.0211 | 0.0262 [0.0242, 0.0296] | 0.0258 | 0.575 / 0.575 | 0.0054 | layer=1, layer=3 |

## 8-second cutoff, fir readout, 100% labels

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | fir | 1 | 0.1170 | 0.1306 | 0.1298 | 0.770 / 0.767 | 0.0071 | delays=[1, 2, 3, 4] |
| B3 frozen random encoder | fir | 3 | 0.0986 | 0.1119 [0.1098, 0.1130] | 0.1112 | 0.731 / 0.727 | 0.0076 | layer=1 |
| R frozen recon (v2 retrain) | fir | 3 | 0.0945 | 0.1087 [0.1084, 0.1092] | 0.1082 | 0.720 / 0.716 | 0.0076 | layer=1 |
| L frozen latent (EMA teacher) | fir | 3 | 0.0847 | 0.0968 [0.0951, 0.0977] | 0.0964 | 0.713 / 0.710 | 0.0066 | layer=1 |

## 8-second cutoff, fir readout, 10% labels

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | fir | 1 | 0.0510 | 0.0640 | 0.0640 | 0.667 / 0.665 | 0.0059 | delays=[1, 2, 3, 4] |
| R frozen recon (v2 retrain) | fir | 3 | 0.0422 | 0.0517 [0.0499, 0.0534] | 0.0517 | 0.637 / 0.636 | 0.0062 | layer=1 |
| B3 frozen random encoder | fir | 3 | 0.0421 | 0.0519 [0.0512, 0.0531] | 0.0518 | 0.641 / 0.639 | 0.0056 | layer=1 |
| L frozen latent (EMA teacher) | fir | 3 | 0.0338 | 0.0397 [0.0375, 0.0419] | 0.0398 | 0.612 / 0.611 | 0.0054 | layer=1 |

## Label budgets (6-s FIR; test r, mean over fits)

| Arm | 100% | 25% | 10% | 5% | 1% |
|---|---:|---:|---:|---:|---:|
| B1 raw BOLD | 0.1371 | 0.0949 | 0.0694 | 0.0467 | 0.0203 |
| B3 frozen random encoder | 0.1161 | 0.0801 | 0.0565 | 0.0400 | 0.0182 |
| L frozen latent (EMA teacher) | 0.1018 | 0.0624 | 0.0436 | 0.0278 | 0.0121 |
| R frozen recon (v2 retrain) | 0.1128 | 0.0790 | 0.0546 | 0.0385 | 0.0180 |

## Pretraining diagnostics and measured cost

Validation losses are in each objective's own units and are not comparable across objectives. Effective rank and pairwise cosine use fully observed validation windows at the final position (512 windows); content fraction is across-window variance at fixed positions over total variance. Seconds are logged training time on one RTX PRO 6000.

| Objective | Seed | Updates | Val loss first → final (best) | Train-mean control | Student rank | Content fraction | Pairwise cos | Teacher rank | Seconds | Examples/s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| latent | 17 | 1110 | 0.8496 → 0.5468 (0.5468) | — | 127.1 | 0.994 | 0.073 | 133.7 | 46.9 | 3149 | 0.800 |
| latent | 23 | 1110 | 0.8309 → 0.5259 (0.5254) | — | 125.7 | 0.996 | 0.092 | 133.5 | 48.3 | 3059 | 0.800 |
| latent | 47 | 1110 | 0.7991 → 0.5402 (0.5377) | — | 125.8 | 0.996 | 0.111 | 134.4 | 47.0 | 3148 | 0.800 |
| recon | 17 | 1110 | 0.9143 → 0.9000 (0.8961) | 0.9135 | 185.3 | 0.998 | 0.078 | — | 41.7 | 3472 | 0.774 |
| recon | 23 | 1110 | 0.9045 → 0.8927 (0.8894) | 0.9039 | 186.4 | 0.998 | 0.082 | — | 43.6 | 3306 | 0.774 |
| recon | 47 | 1110 | 0.9224 → 0.9080 (0.9066) | 0.9215 | 184.9 | 0.999 | 0.058 | — | 42.8 | 3384 | 0.774 |

