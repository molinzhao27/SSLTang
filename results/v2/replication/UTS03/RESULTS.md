# Protocol v2 Phase B: latent prediction versus reconstruction (UTS03)

Exploratory (UTS01 test stories were examined before). Both objectives share windows, masks, encoder, optimizer and 1,110 updates; each exports its final student encoder. The locked readout is the Phase A frozen ridge procedure at the 6-s cutoff with FIR states (layer and alpha chosen per encoder on validation). Test r is the session-macro feature correlation. See [PROTOCOL_V2.md](../../../experiments/PROTOCOL_V2.md) §5.

## Paired story-level contrasts

Seeds are paired by pretraining seed; story differences are averaged over seeds before an exact sign-flip test over 11 test stories (stories share two sessions: descriptive only).

| Contrast (a − b) | Labels | Pairs | Δ test r | Per pair | Per session | Stories a > b | Sign-flip p |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRIMARY Latent − reconstruction (frozen, FIR) | 1 | 3 | -0.0131 | -0.0149, -0.0130, -0.0113 | 14: -0.0117, 4: -0.0145 | 0/11 | 0.001 |
| Latent − reconstruction (frozen, FIR) | 0.1 | 3 | -0.0126 | -0.0137, -0.0148, -0.0094 | 14: -0.0147, 4: -0.0106 | 0/11 | 0.001 |
| Latent − reconstruction (frozen, FIR) | 0.25 | 3 | -0.0129 | -0.0148, -0.0152, -0.0086 | 14: -0.0161, 4: -0.0097 | 0/11 | 0.001 |
| Latent − reconstruction (frozen, FIR) | 0.05 | 3 | -0.0132 | -0.0125, -0.0135, -0.0135 | 14: -0.0164, 4: -0.0099 | 1/11 | 0.002 |
| Latent − reconstruction (frozen, FIR) | 0.01 | 3 | -0.0093 | -0.0065, -0.0089, -0.0125 | 14: -0.0082, 4: -0.0104 | 0/11 | 0.001 |
| Latent − reconstruction (frozen, 8-s FIR) | 1 | 3 | -0.0124 | -0.0153, -0.0109, -0.0110 | 14: -0.0101, 4: -0.0147 | 0/11 | 0.001 |
| Latent − reconstruction (frozen, endpoint) | 1 | 3 | -0.0094 | -0.0128, -0.0074, -0.0081 | 14: -0.0101, 4: -0.0088 | 0/11 | 0.001 |
| Latent − reconstruction (fine-tuned) | 1 | 3 | -0.0238 | -0.0181, -0.0216, -0.0318 | 14: -0.0231, 4: -0.0245 | 0/11 | 0.001 |
| Latent − frozen random encoder | 1 | 3 | -0.0123 | -0.0130, -0.0158, -0.0080 | 14: -0.0117, 4: -0.0128 | 0/11 | 0.001 |
| Reconstruction (v2) − frozen random encoder | 1 | 3 | +0.0008 | +0.0019, -0.0028, +0.0033 | 14: +0.0000, 4: +0.0017 | 6/11 | 0.551 |
| Latent − raw BOLD benchmark | 1 | 3 | -0.0403 | -0.0418, -0.0397, -0.0393 | 14: -0.0349, 4: -0.0456 | 0/11 | 0.001 |
| Reconstruction (v2) − raw BOLD benchmark | 1 | 3 | -0.0272 | -0.0269, -0.0267, -0.0280 | 14: -0.0233, 4: -0.0311 | 0/11 | 0.001 |
| Latent − reconstruction (frozen, 8-s FIR) | 0.1 | 3 | -0.0103 | -0.0124, -0.0131, -0.0055 | 14: -0.0114, 4: -0.0093 | 0/11 | 0.001 |
| Latent − reconstruction (frozen, endpoint) | 0.1 | 3 | -0.0130 | -0.0105, -0.0160, -0.0125 | 14: -0.0157, 4: -0.0103 | 0/11 | 0.001 |
| Latent − reconstruction (fine-tuned) | 0.1 | 3 | -0.0097 | -0.0054, -0.0109, -0.0128 | 14: -0.0083, 4: -0.0110 | 0/11 | 0.001 |
| Latent − frozen random encoder | 0.1 | 3 | -0.0111 | -0.0110, -0.0126, -0.0097 | 14: -0.0117, 4: -0.0105 | 0/11 | 0.001 |
| Reconstruction (v2) − frozen random encoder | 0.1 | 3 | +0.0016 | +0.0027, +0.0022, -0.0003 | 14: +0.0030, 4: +0.0001 | 7/11 | 0.398 |
| Latent − raw BOLD benchmark | 0.1 | 3 | -0.0237 | -0.0230, -0.0243, -0.0239 | 14: -0.0231, 4: -0.0244 | 0/11 | 0.001 |
| Reconstruction (v2) − raw BOLD benchmark | 0.1 | 3 | -0.0111 | -0.0093, -0.0096, -0.0145 | 14: -0.0084, 4: -0.0138 | 0/11 | 0.001 |

## 6-second cutoff, fir readout, 100% labels

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | fir | 1 | 0.1320 | 0.1431 | 0.1426 | 0.791 / 0.787 | 0.0067 | delays=[1, 2, 3] |
| R frozen recon (v2 retrain) | fir | 3 | 0.1077 | 0.1159 [0.1151, 0.1164] | 0.1158 | 0.744 / 0.740 | 0.0073 | layer=1 |
| B3 frozen random encoder | fir | 3 | 0.1052 | 0.1151 [0.1118, 0.1192] | 0.1150 | 0.740 / 0.736 | 0.0067 | layer=1 |
| L frozen latent (EMA teacher) | fir | 3 | 0.0945 | 0.1028 [0.1013, 0.1038] | 0.1028 | 0.726 / 0.721 | 0.0066 | layer=1 |
| fine-tuned, v2 recon init | finetune | 3 | 0.0789 | 0.0951 [0.0918, 0.0985] | 0.0956 | 0.688 / 0.686 | 0.0074 | step 252, step 288 |
| B5 fine-tuned, random init | finetune | 3 | 0.0778 | 0.0903 [0.0825, 0.0964] | 0.0905 | 0.686 / 0.683 | 0.0070 | step 216, step 252, step 288 |
| fine-tuned, latent init | finetune | 3 | 0.0600 | 0.0712 [0.0600, 0.0769] | 0.0718 | 0.654 / 0.651 | 0.0069 | step 252, step 720, step 936 |

## 6-second cutoff, fir readout, 10% labels

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | fir | 1 | 0.0659 | 0.0709 | 0.0707 | 0.688 / 0.684 | 0.0063 | delays=[1, 2, 3] |
| R frozen recon (v2 retrain) | fir | 3 | 0.0572 | 0.0598 [0.0564, 0.0616] | 0.0598 | 0.656 / 0.653 | 0.0066 | layer=1 |
| B3 frozen random encoder | fir | 3 | 0.0527 | 0.0582 [0.0567, 0.0591] | 0.0582 | 0.656 / 0.653 | 0.0063 | layer=1 |
| L frozen latent (EMA teacher) | fir | 3 | 0.0438 | 0.0472 [0.0466, 0.0479] | 0.0470 | 0.632 / 0.629 | 0.0064 | layer=1 |
| fine-tuned, v2 recon init | finetune | 3 | 0.0140 | 0.0165 [0.0130, 0.0191] | 0.0164 | 0.588 / 0.586 | 0.0033 | step 40, step 44 |
| B5 fine-tuned, random init | finetune | 3 | 0.0137 | 0.0160 [0.0127, 0.0181] | 0.0159 | 0.587 / 0.586 | 0.0031 | step 32, step 36 |
| fine-tuned, latent init | finetune | 3 | 0.0075 | 0.0069 [0.0064, 0.0076] | 0.0069 | 0.525 / 0.524 | 0.0043 | step 44, step 48 |

## 6-second cutoff, endpoint readout, 100% labels

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | endpoint | 1 | 0.1036 | 0.1284 | 0.1289 | 0.756 / 0.752 | 0.0077 | delays=[3] |
| R frozen recon (v2 retrain) | endpoint | 3 | 0.0870 | 0.1051 [0.1030, 0.1071] | 0.1057 | 0.715 / 0.711 | 0.0079 | layer=1 |
| B3 frozen random encoder | endpoint | 3 | 0.0820 | 0.1024 [0.1008, 0.1050] | 0.1030 | 0.715 / 0.711 | 0.0075 | layer=1 |
| L frozen latent (EMA teacher) | endpoint | 3 | 0.0809 | 0.0957 [0.0943, 0.0977] | 0.0962 | 0.713 / 0.709 | 0.0072 | layer=1 |

## 6-second cutoff, endpoint readout, 10% labels

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | endpoint | 1 | 0.0578 | 0.0698 | 0.0701 | 0.677 / 0.674 | 0.0065 | delays=[3] |
| R frozen recon (v2 retrain) | endpoint | 3 | 0.0498 | 0.0595 [0.0581, 0.0613] | 0.0599 | 0.655 / 0.652 | 0.0068 | layer=1 |
| B3 frozen random encoder | endpoint | 3 | 0.0460 | 0.0557 [0.0534, 0.0585] | 0.0558 | 0.651 / 0.648 | 0.0066 | layer=1 |
| L frozen latent (EMA teacher) | endpoint | 3 | 0.0428 | 0.0465 [0.0452, 0.0488] | 0.0467 | 0.632 / 0.630 | 0.0065 | layer=1 |

## 8-second cutoff, fir readout, 100% labels

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | fir | 1 | 0.1284 | 0.1404 | 0.1402 | 0.789 / 0.785 | 0.0067 | delays=[1, 2, 3, 4] |
| R frozen recon (v2 retrain) | fir | 3 | 0.1040 | 0.1150 [0.1131, 0.1162] | 0.1150 | 0.744 / 0.741 | 0.0071 | layer=1 |
| B3 frozen random encoder | fir | 3 | 0.1031 | 0.1173 [0.1147, 0.1209] | 0.1175 | 0.740 / 0.736 | 0.0066 | layer=1 |
| L frozen latent (EMA teacher) | fir | 3 | 0.0917 | 0.1025 [0.1009, 0.1047] | 0.1028 | 0.723 / 0.719 | 0.0067 | layer=1 |

## 8-second cutoff, fir readout, 10% labels

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | fir | 1 | 0.0632 | 0.0678 | 0.0676 | 0.679 / 0.675 | 0.0064 | delays=[1, 2, 3, 4] |
| R frozen recon (v2 retrain) | fir | 3 | 0.0560 | 0.0561 [0.0510, 0.0596] | 0.0562 | 0.646 / 0.643 | 0.0066 | layer=1 |
| B3 frozen random encoder | fir | 3 | 0.0513 | 0.0582 [0.0556, 0.0596] | 0.0581 | 0.656 / 0.653 | 0.0064 | layer=1 |
| L frozen latent (EMA teacher) | fir | 3 | 0.0412 | 0.0458 [0.0447, 0.0472] | 0.0457 | 0.631 / 0.628 | 0.0065 | layer=1 |

## Label budgets (6-s FIR; test r, mean over fits)

| Arm | 100% | 25% | 10% | 5% | 1% |
|---|---:|---:|---:|---:|---:|
| B1 raw BOLD | 0.1431 | 0.0977 | 0.0709 | 0.0572 | 0.0285 |
| B3 frozen random encoder | 0.1151 | 0.0814 | 0.0582 | 0.0481 | 0.0253 |
| L frozen latent (EMA teacher) | 0.1028 | 0.0670 | 0.0472 | 0.0369 | 0.0197 |
| R frozen recon (v2 retrain) | 0.1159 | 0.0798 | 0.0598 | 0.0501 | 0.0290 |

## Pretraining diagnostics and measured cost

Validation losses are in each objective's own units and are not comparable across objectives. Effective rank and pairwise cosine use fully observed validation windows at the final position (512 windows); content fraction is across-window variance at fixed positions over total variance. Seconds are logged training time on one RTX PRO 6000.

| Objective | Seed | Updates | Val loss first → final (best) | Train-mean control | Student rank | Content fraction | Pairwise cos | Teacher rank | Seconds | Examples/s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| latent | 17 | 1110 | 0.8285 → 0.5989 (0.5955) | — | 126.4 | 0.995 | 0.078 | 133.1 | 47.4 | 3106 | 0.800 |
| latent | 23 | 1110 | 0.8102 → 0.5582 (0.5525) | — | 129.5 | 0.992 | 0.090 | 134.0 | 48.4 | 3050 | 0.800 |
| latent | 47 | 1110 | 0.7805 → 0.5687 (0.5685) | — | 122.6 | 0.992 | 0.114 | 132.4 | 49.6 | 2969 | 0.800 |
| recon | 17 | 1110 | 0.8544 → 0.8537 (0.8475) | 0.8517 | 186.5 | 0.999 | 0.089 | — | 42.0 | 3442 | 0.774 |
| recon | 23 | 1110 | 0.8522 → 0.8466 (0.8435) | 0.8509 | 183.4 | 0.998 | 0.098 | — | 43.3 | 3350 | 0.774 |
| recon | 47 | 1110 | 0.8575 → 0.8548 (0.8490) | 0.8557 | 186.7 | 0.998 | 0.086 | — | 42.7 | 3390 | 0.774 |

