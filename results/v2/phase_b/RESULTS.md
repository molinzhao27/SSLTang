# Protocol v2 Phase B: latent prediction versus reconstruction (UTS01)

Exploratory (UTS01 test stories were examined before). Both objectives share windows, masks, encoder, optimizer and 1,110 updates; each exports its final student encoder. The locked readout is the Phase A frozen ridge procedure at the 6-s cutoff with FIR states (layer and alpha chosen per encoder on validation). Test r is the session-macro feature correlation. See [PROTOCOL_V2.md](../../../experiments/PROTOCOL_V2.md) §5.

## Paired story-level contrasts

Seeds are paired by pretraining seed; story differences are averaged over seeds before an exact sign-flip test over 11 test stories (stories share two sessions: descriptive only).

| Contrast (a − b) | Labels | Pairs | Δ test r | Per pair | Per session | Stories a > b | Sign-flip p |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRIMARY Latent − reconstruction (frozen, FIR) | 1 | 3 | -0.0144 | -0.0159, -0.0153, -0.0120 | 14: -0.0121, 4: -0.0166 | 0/11 | 0.001 |
| Latent − reconstruction (frozen, FIR) | 0.1 | 3 | -0.0088 | -0.0100, -0.0099, -0.0066 | 14: -0.0066, 4: -0.0111 | 2/11 | 0.011 |
| Latent − reconstruction (frozen, FIR) | 0.25 | 3 | -0.0165 | -0.0167, -0.0180, -0.0148 | 14: -0.0153, 4: -0.0177 | 1/11 | 0.003 |
| Latent − reconstruction (frozen, FIR) | 0.05 | 3 | -0.0089 | -0.0083, -0.0146, -0.0039 | 14: -0.0088, 4: -0.0090 | 0/11 | 0.001 |
| Latent − reconstruction (frozen, FIR) | 0.01 | 3 | -0.0029 | -0.0029, -0.0037, -0.0019 | 14: -0.0019, 4: -0.0038 | 4/11 | 0.240 |
| Latent − reconstruction (frozen, 8-s FIR) | 1 | 3 | -0.0137 | -0.0144, -0.0134, -0.0132 | 14: -0.0115, 4: -0.0158 | 1/11 | 0.002 |
| Latent − reconstruction (frozen, endpoint) | 1 | 3 | -0.0124 | -0.0118, -0.0133, -0.0122 | 14: -0.0122, 4: -0.0126 | 0/11 | 0.001 |
| Latent − reconstruction (fine-tuned) | 1 | 3 | -0.0140 | -0.0400, -0.0108, +0.0089 | 14: -0.0140, 4: -0.0140 | 0/11 | 0.001 |
| Latent − frozen random encoder | 1 | 3 | -0.0118 | -0.0161, -0.0121, -0.0071 | 14: -0.0098, 4: -0.0137 | 0/11 | 0.001 |
| Reconstruction (v2) − frozen random encoder | 1 | 3 | +0.0026 | -0.0003, +0.0032, +0.0049 | 14: +0.0023, 4: +0.0029 | 8/11 | 0.148 |
| Latent − raw BOLD benchmark | 1 | 3 | -0.0396 | -0.0427, -0.0377, -0.0382 | 14: -0.0347, 4: -0.0444 | 0/11 | 0.001 |
| Reconstruction (v2) − raw BOLD benchmark | 1 | 3 | -0.0252 | -0.0268, -0.0224, -0.0263 | 14: -0.0226, 4: -0.0277 | 0/11 | 0.001 |
| Latent − reconstruction (frozen, 8-s FIR) | 0.1 | 3 | -0.0091 | -0.0077, -0.0126, -0.0071 | 14: -0.0067, 4: -0.0116 | 2/11 | 0.014 |
| Latent − reconstruction (frozen, endpoint) | 0.1 | 3 | -0.0112 | -0.0141, -0.0097, -0.0098 | 14: -0.0075, 4: -0.0148 | 0/11 | 0.001 |
| Latent − reconstruction (fine-tuned) | 0.1 | 3 | -0.0036 | -0.0070, -0.0012, -0.0027 | 14: -0.0019, 4: -0.0054 | 2/11 | 0.058 |
| Latent − frozen random encoder | 0.1 | 3 | -0.0073 | -0.0131, -0.0065, -0.0024 | 14: -0.0062, 4: -0.0085 | 2/11 | 0.021 |
| Reconstruction (v2) − frozen random encoder | 0.1 | 3 | +0.0015 | -0.0031, +0.0034, +0.0042 | 14: +0.0004, 4: +0.0027 | 8/11 | 0.299 |
| Latent − raw BOLD benchmark | 0.1 | 3 | -0.0207 | -0.0251, -0.0186, -0.0183 | 14: -0.0180, 4: -0.0233 | 0/11 | 0.001 |
| Reconstruction (v2) − raw BOLD benchmark | 0.1 | 3 | -0.0118 | -0.0151, -0.0087, -0.0117 | 14: -0.0115, 4: -0.0122 | 0/11 | 0.001 |

## 6-second cutoff, fir readout, 100% labels

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | fir | 1 | 0.1110 | 0.1221 | 0.1208 | 0.754 / 0.751 | 0.0071 | delays=[1, 2, 3] |
| B4 frozen recon encoder (final) | fir | 3 | 0.0867 | 0.0978 [0.0962, 0.1010] | 0.0968 | 0.707 / 0.705 | 0.0072 | layer=1 |
| B3 frozen random encoder | fir | 3 | 0.0860 | 0.0944 [0.0909, 0.0965] | 0.0933 | 0.700 / 0.698 | 0.0072 | layer=1 |
| R frozen recon (v2 retrain) | fir | 3 | 0.0849 | 0.0970 [0.0953, 0.0997] | 0.0959 | 0.705 / 0.703 | 0.0073 | layer=1 |
| L frozen latent (EMA teacher) | fir | 3 | 0.0753 | 0.0826 [0.0795, 0.0844] | 0.0817 | 0.687 / 0.686 | 0.0063 | layer=1 |
| B5 fine-tuned, recon init | finetune | 3 | 0.0570 | 0.0658 [0.0426, 0.0790] | 0.0652 | 0.634 / 0.634 | 0.0062 | step 180, step 288 |
| B5 fine-tuned, random init | finetune | 3 | 0.0537 | 0.0638 [0.0403, 0.0787] | 0.0633 | 0.638 / 0.637 | 0.0060 | step 180, step 288 |
| fine-tuned, v2 recon init | finetune | 3 | 0.0532 | 0.0606 [0.0408, 0.0712] | 0.0602 | 0.627 / 0.627 | 0.0058 | step 180, step 288 |
| fine-tuned, latent init | finetune | 3 | 0.0442 | 0.0466 [0.0299, 0.0603] | 0.0462 | 0.607 / 0.607 | 0.0056 | step 1224, step 288 |

## 6-second cutoff, fir readout, 10% labels

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | fir | 1 | 0.0392 | 0.0528 | 0.0520 | 0.653 / 0.654 | 0.0054 | delays=[1, 2, 3] |
| R frozen recon (v2 retrain) | fir | 3 | 0.0322 | 0.0410 [0.0377, 0.0442] | 0.0402 | 0.611 / 0.611 | 0.0060 | layer=1 |
| B4 frozen recon encoder (final) | fir | 3 | 0.0312 | 0.0401 [0.0390, 0.0413] | 0.0394 | 0.610 / 0.610 | 0.0060 | layer=1 |
| B3 frozen random encoder | fir | 3 | 0.0295 | 0.0395 [0.0369, 0.0408] | 0.0388 | 0.615 / 0.616 | 0.0057 | layer=1 |
| L frozen latent (EMA teacher) | fir | 3 | 0.0228 | 0.0321 [0.0277, 0.0345] | 0.0316 | 0.596 / 0.596 | 0.0060 | layer=1 |
| B5 fine-tuned, recon init | finetune | 3 | 0.0091 | 0.0102 [0.0095, 0.0113] | 0.0101 | 0.563 / 0.563 | 0.0027 | step 36, step 40, step 44 |
| fine-tuned, v2 recon init | finetune | 3 | 0.0087 | 0.0089 [0.0087, 0.0091] | 0.0088 | 0.559 / 0.558 | 0.0026 | step 40, step 44 |
| B5 fine-tuned, random init | finetune | 3 | 0.0073 | 0.0083 [0.0076, 0.0089] | 0.0081 | 0.549 / 0.549 | 0.0025 | step 36 |
| fine-tuned, latent init | finetune | 3 | 0.0034 | 0.0053 [0.0016, 0.0079] | 0.0053 | 0.528 / 0.530 | 0.0028 | step 44, step 48 |

## 6-second cutoff, endpoint readout, 100% labels

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | endpoint | 1 | 0.0903 | 0.0980 | 0.0972 | 0.699 / 0.698 | 0.0073 | delays=[3] |
| B4 frozen recon encoder (final) | endpoint | 3 | 0.0717 | 0.0818 [0.0786, 0.0835] | 0.0814 | 0.667 / 0.666 | 0.0073 | layer=1 |
| R frozen recon (v2 retrain) | endpoint | 3 | 0.0695 | 0.0822 [0.0798, 0.0855] | 0.0817 | 0.667 / 0.666 | 0.0071 | layer=1 |
| B3 frozen random encoder | endpoint | 3 | 0.0669 | 0.0733 [0.0693, 0.0766] | 0.0728 | 0.655 / 0.654 | 0.0072 | layer=1 |
| L frozen latent (EMA teacher) | endpoint | 3 | 0.0639 | 0.0698 [0.0676, 0.0723] | 0.0693 | 0.655 / 0.654 | 0.0061 | layer=1 |

## 6-second cutoff, endpoint readout, 10% labels

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | endpoint | 1 | 0.0329 | 0.0442 | 0.0439 | 0.634 / 0.634 | 0.0051 | delays=[3] |
| B4 frozen recon encoder (final) | endpoint | 3 | 0.0261 | 0.0365 [0.0312, 0.0403] | 0.0362 | 0.609 / 0.609 | 0.0055 | layer=1 |
| R frozen recon (v2 retrain) | endpoint | 3 | 0.0258 | 0.0368 [0.0333, 0.0390] | 0.0363 | 0.608 / 0.607 | 0.0056 | layer=1 |
| B3 frozen random encoder | endpoint | 3 | 0.0230 | 0.0328 [0.0292, 0.0347] | 0.0326 | 0.603 / 0.603 | 0.0051 | layer=1 |
| L frozen latent (EMA teacher) | endpoint | 3 | 0.0197 | 0.0257 [0.0236, 0.0293] | 0.0255 | 0.579 / 0.579 | 0.0056 | layer=1, layer=4 |

## 8-second cutoff, fir readout, 100% labels

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | fir | 1 | 0.1078 | 0.1174 | 0.1161 | 0.747 / 0.744 | 0.0069 | delays=[1, 2, 3, 4] |
| B4 frozen recon encoder (final) | fir | 3 | 0.0872 | 0.0951 [0.0934, 0.0985] | 0.0942 | 0.698 / 0.696 | 0.0075 | layer=1 |
| B3 frozen random encoder | fir | 3 | 0.0869 | 0.0930 [0.0887, 0.0954] | 0.0921 | 0.698 / 0.696 | 0.0070 | layer=1 |
| R frozen recon (v2 retrain) | fir | 3 | 0.0853 | 0.0946 [0.0931, 0.0967] | 0.0936 | 0.696 / 0.694 | 0.0076 | layer=1 |
| L frozen latent (EMA teacher) | fir | 3 | 0.0750 | 0.0809 [0.0787, 0.0832] | 0.0801 | 0.684 / 0.682 | 0.0063 | layer=1 |

## 8-second cutoff, fir readout, 10% labels

| Arm | Readout | Fits | Val r | Test r (seed range) | Test run-macro r | Retrieval pct ±10 / ±32 TR | Shift-null q95 | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 raw BOLD | fir | 1 | 0.0383 | 0.0510 | 0.0503 | 0.649 / 0.649 | 0.0056 | delays=[1, 2, 3, 4] |
| R frozen recon (v2 retrain) | fir | 3 | 0.0317 | 0.0402 [0.0360, 0.0447] | 0.0396 | 0.608 / 0.609 | 0.0060 | layer=1 |
| B4 frozen recon encoder (final) | fir | 3 | 0.0312 | 0.0399 [0.0385, 0.0407] | 0.0393 | 0.611 / 0.611 | 0.0060 | layer=1 |
| B3 frozen random encoder | fir | 3 | 0.0306 | 0.0400 [0.0382, 0.0412] | 0.0395 | 0.617 / 0.618 | 0.0053 | layer=1 |
| L frozen latent (EMA teacher) | fir | 3 | 0.0225 | 0.0311 [0.0283, 0.0329] | 0.0307 | 0.593 / 0.592 | 0.0060 | layer=1 |

## Label budgets (6-s FIR; test r, mean over fits)

| Arm | 100% | 25% | 10% | 5% | 1% |
|---|---:|---:|---:|---:|---:|
| B1 raw BOLD | 0.1221 | 0.0728 | 0.0528 | 0.0392 | 0.0178 |
| B3 frozen random encoder | 0.0944 | 0.0591 | 0.0395 | 0.0271 | 0.0134 |
| B4 frozen recon encoder (final) | 0.0978 | 0.0618 | 0.0401 | 0.0265 | 0.0168 |
| L frozen latent (EMA teacher) | 0.0826 | 0.0465 | 0.0321 | 0.0200 | 0.0129 |
| R frozen recon (v2 retrain) | 0.0970 | 0.0630 | 0.0410 | 0.0290 | 0.0157 |

## Pretraining diagnostics and measured cost

Validation losses are in each objective's own units and are not comparable across objectives. Effective rank and pairwise cosine use fully observed validation windows at the final position (512 windows); content fraction is across-window variance at fixed positions over total variance. Seconds are logged training time on one RTX PRO 6000.

| Objective | Seed | Updates | Val loss first → final (best) | Train-mean control | Student rank | Content fraction | Pairwise cos | Teacher rank | Seconds | Examples/s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| latent | 17 | 1110 | 0.7867 → 0.6486 (0.6468) | — | 128.9 | 0.994 | 0.122 | 140.1 | 35.4 | 4149 | 0.756 |
| latent | 23 | 1110 | 0.8010 → 0.6303 (0.6133) | — | 129.5 | 0.996 | 0.104 | 137.9 | 38.2 | 3868 | 0.756 |
| latent | 47 | 1110 | 0.8349 → 0.6440 (0.6240) | — | 132.4 | 0.990 | 0.127 | 138.7 | 36.4 | 4071 | 0.756 |
| recon | 17 | 1110 | 0.8783 → 0.8773 (0.8739) | 0.8765 | 203.6 | 0.998 | 0.168 | — | 31.9 | 4526 | 0.715 |
| recon | 23 | 1110 | 0.8426 → 0.8399 (0.8368) | 0.8409 | 199.3 | 0.998 | 0.118 | — | 31.6 | 4568 | 0.715 |
| recon | 47 | 1110 | 0.8488 → 0.8457 (0.8428) | 0.8470 | 205.8 | 0.998 | 0.133 | — | 32.9 | 4412 | 0.715 |

