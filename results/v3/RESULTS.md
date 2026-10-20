# Protocol v3 results: structured voxel ridge (locked evaluation)

Advanced arm **V4**, locked 2026-10-03T21:42:59Z (code `c76b1c77eb38`); see [PROTOCOL_V3.md](../../experiments/PROTOCOL_V3.md) §9. Test r is the session-macro feature correlation on the exposed UTS01–03 test stories. Every arm chose its kernel and alpha by session-grouped CV on that budget's training labels only; the 794 validation labels were used once, for the development gates.

## Test r by arm and label budget

| Subject | Labels | V0 | V1 | V2 | V3 | V4 | V4 − V0 | V4 selection |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| UTS01 | 4518 | 0.1336 | 0.1313 | 0.1359 | 0.1351 | 0.1370 | 0.0035 | eta=0.1,tau=121, α=3.2e+05 |
| UTS02 | 4518 | 0.1481 | 0.1465 | 0.1504 | 0.1502 | 0.1527 | 0.0046 | eta=0.1,tau=110, α=1e+05 |
| UTS03 | 4518 | 0.1560 | 0.1553 | 0.1589 | 0.1591 | 0.1608 | 0.0048 | eta=0.1,tau=121, α=3.2e+05 |
| UTS01 | 1130 | 0.0824 | 0.0766 | 0.0851 | 0.0849 | 0.0878 | 0.0054 | eta=0.1,tau=121, α=1e+05 |
| UTS02 | 1130 | 0.0978 | 0.0940 | 0.1028 | 0.1011 | 0.1055 | 0.0077 | eta=0,tau=121, α=1e+05 |
| UTS03 | 1130 | 0.1070 | 0.1022 | 0.1092 | 0.1121 | 0.1116 | 0.0046 | eta=0.1,tau=121, α=1e+05 |
| UTS01 | 452 | 0.0579 | 0.0502 | 0.0610 | 0.0606 | 0.0608 | 0.0029 | eta=0,tau=121, α=1e+05 |
| UTS02 | 452 | 0.0702 | 0.0660 | 0.0737 | 0.0728 | 0.0763 | 0.0061 | eta=0.1,tau=121, α=1e+05 |
| UTS03 | 452 | 0.0785 | 0.0735 | 0.0778 | 0.0825 | 0.0826 | 0.0041 | eta=0.1,tau=011, α=3.2e+04 |
| UTS01 | 226 | 0.0444 | 0.0355 | 0.0451 | 0.0453 | 0.0396 | -0.0047 | eta=0,tau=110, α=0.1 |
| UTS02 | 226 | 0.0477 | 0.0437 | 0.0462 | 0.0449 | 0.0431 | -0.0046 | eta=10,tau=011, α=0.1 |
| UTS03 | 226 | 0.0618 | 0.0564 | 0.0616 | 0.0627 | 0.0642 | 0.0024 | eta=0.1,tau=011, α=3.2e+04 |
| UTS01 | 46 | 0.0161 | 0.0109 | 0.0149 | 0.0168 | 0.0164 | 0.0003 | eta=0,tau=121, α=3.2e+04 |
| UTS02 | 46 | 0.0207 | 0.0195 | 0.0198 | 0.0191 | 0.0241 | 0.0034 | eta=0,tau=110, α=0.1 |
| UTS03 | 46 | 0.0251 | 0.0261 | 0.0244 | 0.0259 | 0.0250 | -0.0001 | eta=10,tau=011, α=3.2e+06 |

## V4 − V0, paired by story (test, 11 stories per subject)

| Labels | Subject | Δ r | Per session | Stories V4 > V0 | Sign-flip p |
|---:|---|---:|---|---:|---:|
| 1 | UTS01 | +0.0035 | 14: +0.0029, 4: +0.0040 | 6/11 | 0.149 |
| 1 | UTS02 | +0.0046 | 14: +0.0036, 4: +0.0056 | 9/11 | 0.020 |
| 1 | UTS03 | +0.0048 | 14: +0.0068, 4: +0.0029 | 10/11 | 0.006 |
| 1 | **mean** | **+0.0043** | | | |
| 0.25 | UTS01 | +0.0054 | 14: +0.0056, 4: +0.0053 | 9/11 | 0.008 |
| 0.25 | UTS02 | +0.0077 | 14: +0.0095, 4: +0.0059 | 11/11 | 0.001 |
| 0.25 | UTS03 | +0.0046 | 14: +0.0008, 4: +0.0084 | 8/11 | 0.023 |
| 0.25 | **mean** | **+0.0059** | | | |
| 0.1 | UTS01 | +0.0029 | 14: +0.0014, 4: +0.0045 | 9/11 | 0.062 |
| 0.1 | UTS02 | +0.0061 | 14: +0.0063, 4: +0.0060 | 11/11 | 0.001 |
| 0.1 | UTS03 | +0.0041 | 14: +0.0033, 4: +0.0050 | 9/11 | 0.013 |
| 0.1 | **mean** | **+0.0044** | | | |
| 0.05 | UTS01 | -0.0047 | 14: -0.0023, 4: -0.0071 | 4/11 | 0.157 |
| 0.05 | UTS02 | -0.0046 | 14: -0.0033, 4: -0.0059 | 1/11 | 0.004 |
| 0.05 | UTS03 | +0.0024 | 14: +0.0017, 4: +0.0031 | 10/11 | 0.067 |
| 0.05 | **mean** | **-0.0023** | | | |
| 0.01 | UTS01 | +0.0003 | 14: -0.0007, 4: +0.0012 | 6/11 | 0.822 |
| 0.01 | UTS02 | +0.0034 | 14: +0.0039, 4: +0.0030 | 8/11 | 0.048 |
| 0.01 | UTS03 | -0.0001 | 14: -0.0016, 4: +0.0013 | 5/11 | 0.861 |
| 0.01 | **mean** | **+0.0012** | | | |

Stories share two sessions per subject and the same 11 stories recur across subjects; sign-flip p-values are descriptive.

## Alignment and brain-input controls (100% labels)

| Subject | V4 real | Shift-null q95 (p) | Disrupted training | Wrong-run input | Nuisance: position | Nuisance: oracle speech rate | Nuisance: both | Residualized V4 | Residualized V0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UTS01 | 0.1370 | 0.0071 (0.0005) | 0.0021 | -0.0070 | 0.0159 | 0.0944 | 0.0952 | 0.1269 | 0.1239 |
| UTS02 | 0.1527 | 0.0074 (0.0005) | 0.0018 | -0.0044 | 0.0159 | 0.0944 | 0.0952 | 0.1430 | 0.1383 |
| UTS03 | 0.1608 | 0.0071 (0.0005) | -0.0014 | -0.0064 | 0.0159 | 0.0944 | 0.0952 | 0.1533 | 0.1487 |

Residualized decoders score test targets minus a training-fit nuisance prediction; the speech-rate features use transcripts (an oracle unavailable to a deployed decoder).

## Supervision exposure per budget

Words consumed by target construction: words in selected bins plus their 20-word causal contexts (unique instances), GPT-2 tokens of those instances, and the union of their timestamped spans. Values are UTS01 / UTS02 / UTS03. Below 100% the hashed label blocks differ by subject (the v2 subset hash includes subject-specific run IDs), so counts differ slightly; within a subject every arm shares the same subset.

| Budget | Targets | Unique words | In-bin / context words (UTS01) | GPT-2 tokens | Transcript seconds | Stories | Sessions |
|---|---:|---:|---:|---:|---:|---:|---:|
| training 100% | 4518 | 97425 / 97425 / 97425 | 25379 / 72046 | 158651 / 158651 / 158651 | 26369 / 26369 / 26369 | 61 / 61 / 61 | 11 / 11 / 11 |
| training 25% | 1130 | 25148 / 23906 / 24828 | 6479 / 18669 | 40899 / 38985 / 40271 | 6750 / 6646 / 6644 | 50 / 51 / 49 | 11 / 11 / 11 |
| training 10% | 452 | 9924 / 9622 / 9945 | 2579 / 7345 | 16173 / 15779 / 16039 | 2708 / 2636 / 2667 | 27 / 29 / 30 | 9 / 11 / 11 |
| training 5% | 226 | 4740 / 4868 / 5098 | 1213 / 3527 | 7752 / 7970 / 8148 | 1348 / 1302 / 1326 | 13 / 19 / 16 | 8 / 10 / 9 |
| training 1% | 46 | 1037 / 906 / 962 | 264 / 773 | 1686 / 1464 / 1548 | 280 / 261 / 263 | 3 / 4 / 4 | 3 / 3 / 3 |
| validation (development exposure) | 794 | 17343 / 17343 / 17343 | 4715 / 12628 | 28045 / 28045 / 28045 | 4555 / 4555 / 4555 | 11 / 11 / 11 | 2 / 2 / 2 |
| dense training (§7) | 17993 | 102603 / 102603 / 102603 | 101383 / 1220 | 166774 / 166774 / 166774 | 27559 / 27559 / 27559 | 61 / 61 / 61 | 11 / 11 / 11 |

## Dense training clock (§7; a changed data budget)

| Subject | Training targets (dense) | V0 old | V0 dense | V4 old | V4 dense |
|---|---:|---:|---:|---:|---:|
| UTS01 | 17993 | 0.1336 | 0.1745 | 0.1370 | 0.1787 |
| UTS02 | 17993 | 0.1481 | 0.1954 | 0.1527 | 0.2003 |
| UTS03 | 17993 | 0.1560 | 0.1971 | 0.1608 | 0.2003 |

## Fresh-recording replication (§11; registered before download)

UTS09 and UTS10 were never used before this replication. The locked v3 procedure ran unchanged, with the §4 fold rule applied to each subject's own training sessions. A contrast replicates when its test difference has the predicted (positive) sign in both subjects.

| Contrast | UTS09 | UTS10 | Replicated |
|---|---:|---:|---|
| R1 dense-clock V0 − stride-4 V0 | +0.0315 | +0.0354 | yes |
| R2 V4 − V0 (stride-4, 100%) | +0.0077 (10/11 stories, p=0.003) | +0.0045 (10/11 stories, p=0.037) | yes |
| R3 V4 − V0 (dense clock) | +0.0083 | +0.0038 | yes |

| Subject | Labels | V0 | V1 | V2 | V3 | V4 | V0 dense | V4 dense |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UTS09 | 2418 | 0.0783 | 0.0758 | 0.0850 | 0.0801 | 0.0860 | 0.1098 | 0.1181 |
| UTS09 | 605 | 0.0475 | 0.0443 | 0.0486 | 0.0464 | 0.0479 | — | — |
| UTS09 | 242 | 0.0290 | 0.0266 | 0.0300 | 0.0271 | 0.0285 | — | — |
| UTS09 | 121 | 0.0139 | 0.0123 | 0.0183 | 0.0146 | 0.0168 | — | — |
| UTS09 | 25 | 0.0049 | 0.0043 | 0.0044 | 0.0058 | 0.0055 | — | — |
| UTS10 | 2418 | 0.0837 | 0.0822 | 0.0878 | 0.0869 | 0.0882 | 0.1191 | 0.1229 |
| UTS10 | 605 | 0.0487 | 0.0444 | 0.0505 | 0.0490 | 0.0506 | — | — |
| UTS10 | 242 | 0.0286 | 0.0239 | 0.0301 | 0.0313 | 0.0323 | — | — |
| UTS10 | 121 | 0.0169 | 0.0127 | 0.0187 | 0.0198 | 0.0214 | — | — |
| UTS10 | 25 | 0.0121 | 0.0130 | 0.0114 | 0.0124 | 0.0116 | — | — |

## Exploratory modality transfer with dense-clock decoders (§7.1)

ds004510 was scored in v2, so these are exploratory. The validation pseudo-split gate reproduced the standard predictions before scoring. Centered narrative-macro top-1 (exact permutation p); imagined speech is 5-way (chance 20%), films 3-way (chance 33%, minimum p = 1/6).

| Subject | Decoder | Consistency (rel. RMS) | Imagined top-1 (p) | Imagined MRR | Films top-1 (p) | v2 pooled imagined top-1 (p) |
|---|---|---:|---:|---:|---:|---:|
| UTS01 | V0 dense | 1e-15 | 70% (0.008) | 0.800 | 33% (1.000) | 60% (0.017) |
| UTS01 | V4 dense | 3e-15 | 60% (0.033) | 0.783 | 33% (1.000) | 60% (0.017) |
| UTS02 | V0 dense | 2e-15 | 70% (0.033) | 0.800 | 33% (1.000) | 50% (0.067) |
| UTS02 | V4 dense | 7e-15 | 60% (0.033) | 0.753 | 33% (1.000) | 50% (0.067) |
| UTS03 | V0 dense | 2e-15 | 70% (0.017) | 0.850 | 33% (1.000) | 60% (0.033) |
| UTS03 | V4 dense | 5e-15 | 80% (0.008) | 0.883 | 33% (1.000) | 60% (0.033) |
