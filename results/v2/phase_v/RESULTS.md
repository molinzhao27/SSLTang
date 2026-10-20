# Protocol v2 Phase V (exploratory): spatial pooling versus all voxels

Same targets, labels, delays, standardization rule and validation-selected alpha as the pooled benchmark B1; only the input changes from 4-voxel cubes to every voxel of the subject's thick mask (kernel ridge; [scripts/run_voxel_ridge.py](../../../scripts/run_voxel_ridge.py)). Test session-macro r; the bracket is the centered retrieval rank percentile (±10 TR).

| Subject | Readout | Labels | Pooled cubes | All voxels | Δ r |
|---|---|---:|---:|---:|---:|
| UTS01 | fir | 4518 | 0.1221 [0.754] | 0.1336 [0.782] | +0.0114 |
| UTS01 | fir | 1130 | 0.0728 [0.682] | 0.0824 [0.709] | +0.0097 |
| UTS01 | fir | 452 | 0.0528 [0.653] | 0.0589 [0.668] | +0.0061 |
| UTS01 | fir | 226 | 0.0392 [0.623] | 0.0444 [0.633] | +0.0052 |
| UTS01 | fir | 46 | 0.0178 [0.552] | 0.0161 [0.548] | -0.0017 |
| UTS01 | endpoint | 4518 | 0.0980 [0.699] | 0.1091 [0.733] | +0.0111 |
| UTS01 | endpoint | 1130 | 0.0598 [0.646] | 0.0651 [0.669] | +0.0054 |
| UTS01 | endpoint | 452 | 0.0442 [0.634] | 0.0485 [0.646] | +0.0043 |
| UTS01 | endpoint | 226 | 0.0317 [0.596] | 0.0348 [0.610] | +0.0032 |
| UTS01 | endpoint | 46 | 0.0139 [0.545] | 0.0137 [0.542] | -0.0002 |
| UTS02 | fir | 4518 | 0.1371 [0.776] | 0.1481 [0.796] | +0.0110 |
| UTS02 | fir | 1130 | 0.0949 [0.717] | 0.0974 [0.730] | +0.0024 |
| UTS02 | fir | 452 | 0.0694 [0.677] | 0.0702 [0.680] | +0.0008 |
| UTS02 | fir | 226 | 0.0467 [0.629] | 0.0477 [0.630] | +0.0011 |
| UTS02 | fir | 46 | 0.0203 [0.569] | 0.0207 [0.566] | +0.0004 |
| UTS02 | endpoint | 4518 | 0.1015 [0.712] | 0.1110 [0.733] | +0.0095 |
| UTS02 | endpoint | 1130 | 0.0679 [0.667] | 0.0712 [0.673] | +0.0033 |
| UTS02 | endpoint | 452 | 0.0454 [0.622] | 0.0480 [0.640] | +0.0026 |
| UTS02 | endpoint | 226 | 0.0295 [0.583] | 0.0321 [0.598] | +0.0026 |
| UTS02 | endpoint | 46 | 0.0143 [0.542] | 0.0129 [0.531] | -0.0014 |
| UTS03 | fir | 4518 | 0.1431 [0.791] | 0.1560 [0.817] | +0.0129 |
| UTS03 | fir | 1130 | 0.0977 [0.735] | 0.1070 [0.755] | +0.0093 |
| UTS03 | fir | 452 | 0.0709 [0.688] | 0.0785 [0.704] | +0.0076 |
| UTS03 | fir | 226 | 0.0572 [0.649] | 0.0618 [0.663] | +0.0046 |
| UTS03 | fir | 46 | 0.0285 [0.576] | 0.0308 [0.584] | +0.0023 |
| UTS03 | endpoint | 4518 | 0.1284 [0.756] | 0.1413 [0.783] | +0.0129 |
| UTS03 | endpoint | 1130 | 0.0894 [0.717] | 0.1001 [0.733] | +0.0107 |
| UTS03 | endpoint | 452 | 0.0698 [0.677] | 0.0773 [0.698] | +0.0075 |
| UTS03 | endpoint | 226 | 0.0539 [0.651] | 0.0588 [0.668] | +0.0049 |
| UTS03 | endpoint | 46 | 0.0215 [0.571] | 0.0253 [0.582] | +0.0038 |
