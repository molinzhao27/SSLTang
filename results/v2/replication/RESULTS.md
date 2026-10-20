# Protocol v2 Phase D: registered replication in UTS02 and UTS03

Contrasts and predicted signs were registered before any UTS02/UTS03 processing ([PROTOCOL_V2.md](../../../experiments/PROTOCOL_V2.md) §7). Test session-macro r, 6-s cutoff, FIR readout. A contrast replicates when the seed-mean difference and every paired seed difference have the predicted sign. UTS01 values are the exploratory originals; D3 uses the v2 reconstruction initialization in every subject. Two test sessions per subject; no population inference.

## 100% labels

| Contrast | UTS01 | UTS02 | UTS03 | Replicated in new subjects |
|---|---:|---:|---:|---|
| D1 Frozen latent − frozen reconstruction | -0.0144 (-0.0159, -0.0153, -0.0120; 0/11 stories > 0) | -0.0110 (-0.0108, -0.0126, -0.0095; 1/11 stories > 0) | -0.0131 (-0.0149, -0.0130, -0.0113; 0/11 stories > 0) | UTS02: yes, UTS03: yes |
| D2 Raw BOLD ridge − frozen reconstruction | +0.0252 (+0.0268, +0.0224, +0.0263; 11/11 stories > 0) | +0.0243 (+0.0249, +0.0225, +0.0254; 11/11 stories > 0) | +0.0272 (+0.0269, +0.0267, +0.0280; 11/11 stories > 0) | UTS02: yes, UTS03: yes |
| D3 Raw BOLD ridge − fine-tuned reconstruction init | +0.0615 (+0.0522, +0.0510, +0.0813; 11/11 stories > 0) | +0.0726 (+0.0612, +0.0667, +0.0899; 11/11 stories > 0) | +0.0480 (+0.0482, +0.0446, +0.0512; 11/11 stories > 0) | UTS02: yes, UTS03: yes |
| D4 Frozen reconstruction − frozen random encoder | +0.0026 (-0.0003, +0.0032, +0.0049; 8/11 stories > 0) | -0.0033 (-0.0017, -0.0021, -0.0061; 2/11 stories > 0) | +0.0008 (+0.0019, -0.0028, +0.0033; 6/11 stories > 0) | report only |

## 10% labels

| Contrast | UTS01 | UTS02 | UTS03 | Replicated in new subjects |
|---|---:|---:|---:|---|
| D1 Frozen latent − frozen reconstruction | -0.0088 (-0.0100, -0.0099, -0.0066; 2/11 stories > 0) | -0.0110 (-0.0145, -0.0079, -0.0106; 0/11 stories > 0) | -0.0126 (-0.0137, -0.0148, -0.0094; 0/11 stories > 0) | UTS02: yes, UTS03: yes |
| D2 Raw BOLD ridge − frozen reconstruction | +0.0118 (+0.0151, +0.0087, +0.0117; 11/11 stories > 0) | +0.0148 (+0.0121, +0.0166, +0.0158; 11/11 stories > 0) | +0.0111 (+0.0093, +0.0096, +0.0145; 11/11 stories > 0) | UTS02: yes, UTS03: yes |
| D3 Raw BOLD ridge − fine-tuned reconstruction init | +0.0439 (+0.0441, +0.0437, +0.0439; 11/11 stories > 0) | +0.0577 (+0.0585, +0.0579, +0.0569; 11/11 stories > 0) | +0.0544 (+0.0579, +0.0535, +0.0518; 11/11 stories > 0) | UTS02: yes, UTS03: yes |
| D4 Frozen reconstruction − frozen random encoder | +0.0015 (-0.0031, +0.0034, +0.0042; 8/11 stories > 0) | -0.0020 (+0.0003, -0.0044, -0.0018; 5/11 stories > 0) | +0.0016 (+0.0027, +0.0022, -0.0003; 7/11 stories > 0) | report only |

## Arm scores (test r, mean [min, max] over fits)

| Arm | Labels | UTS01 | UTS02 | UTS03 |
|---|---:|---:|---:|---:|
| B1 raw BOLD | 100% | 0.1221 | 0.1371 | 0.1431 |
| B3 frozen random | 100% | 0.0944 [0.0909, 0.0965] | 0.1161 [0.1139, 0.1178] | 0.1151 [0.1118, 0.1192] |
| R frozen reconstruction | 100% | 0.0970 [0.0953, 0.0997] | 0.1128 [0.1117, 0.1146] | 0.1159 [0.1151, 0.1164] |
| L frozen latent | 100% | 0.0826 [0.0795, 0.0844] | 0.1018 [0.1015, 0.1021] | 0.1028 [0.1013, 0.1038] |
| Fine-tuned, random init | 100% | 0.0638 [0.0403, 0.0787] | 0.0678 [0.0469, 0.0800] | 0.0903 [0.0825, 0.0964] |
| Fine-tuned, reconstruction init | 100% | 0.0606 [0.0408, 0.0712] | 0.0644 [0.0471, 0.0758] | 0.0951 [0.0918, 0.0985] |
| Fine-tuned, latent init | 100% | 0.0466 [0.0299, 0.0603] | 0.0551 [0.0438, 0.0679] | 0.0712 [0.0600, 0.0769] |
| B1 raw BOLD | 10% | 0.0528 | 0.0694 | 0.0709 |
| B3 frozen random | 10% | 0.0395 [0.0369, 0.0408] | 0.0565 [0.0554, 0.0572] | 0.0582 [0.0567, 0.0591] |
| R frozen reconstruction | 10% | 0.0410 [0.0377, 0.0442] | 0.0546 [0.0528, 0.0573] | 0.0598 [0.0564, 0.0616] |
| L frozen latent | 10% | 0.0321 [0.0277, 0.0345] | 0.0436 [0.0429, 0.0449] | 0.0472 [0.0466, 0.0479] |
| Fine-tuned, random init | 10% | 0.0083 [0.0076, 0.0089] | 0.0088 [0.0080, 0.0093] | 0.0160 [0.0127, 0.0181] |
| Fine-tuned, reconstruction init | 10% | 0.0089 [0.0087, 0.0091] | 0.0117 [0.0109, 0.0126] | 0.0165 [0.0130, 0.0191] |
| Fine-tuned, latent init | 10% | 0.0053 [0.0016, 0.0079] | 0.0054 [0.0051, 0.0057] | 0.0069 [0.0064, 0.0076] |
