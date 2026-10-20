# Protocol v3 (`ssltang-v3`): structured voxel ridge

**Written:** 2026-10-03, before any v3 model was fit or scored. **Plan:** FUTURE_WORK_V3.md.
**Base commit:** `76b3a52`. Sections 1–7 are frozen. Development results and the lock decision are appended in §9
before any v3 test score is produced. Deviations go in §10.

## 0. Evaluation exposure (recorded before v3)

- UTS01, UTS02 and UTS03 test stories (11 stories, sessions 4 and 14) were scored in v1 (UTS01) and v2 (all
  three). Validation stories (sessions 2 and 20) selected options in v1/v2.
- ds004510 transfer runs were scored for UTS01 (v1, v2) and UTS02/UTS03 (v2).
- Every v3 test score is therefore a comparison on an **exposed benchmark**, not a confirmatory test. Stronger
  generalization evidence needs new recordings (FUTURE_WORK_V3 §7.2), which are outside this protocol.

## 1. Fixed elements

- **Targets:** the v2 target index (window 32, stride 4, phase 0, cutoffs [3, 4]): 4,518 / 794 / 802
  train / validation / test targets in every subject. GPT-2 layer-9 768-d targets, unchanged.
- **Label budgets:** v2 nested hashed 64-TR blocks (seed 2718): 4,518 / 1,130 / 452 / 226 / 46 targets. These
  occupy 11 / 11 / 9 / 8 / 3 training sessions in UTS01. The same identities are used in every arm.
- **Input:** every voxel of the subject's thick mask at `t+1`, `t+2`, `t+3`, read from the published H5
  derivatives. Non-finite values become 0 (as in preprocessing); values are cached as float32.
- **Endpoints:** the v2 implementation ([endpoints.py](../src/evaluation/endpoints.py)).
  - Primary: session-macro feature correlation.
  - Also reported: per-story and per-session scores, normalized MSE, centered retrieval (±10 / ±32 TR), and the
    within-run circular-shift null (≥ 40 TR).

## 2. Features and kernels

- **Normalization.** Each (voxel, delay) column is z-scored with the mean and standard deviation of the target
  rows of the fitting partition's sessions, labeled or not. The near-zero threshold follows `Standardizer`.
  Because v2's per-voxel normalization is affine and per-column, this equals the Phase V two-stage scaling
  (checked in G0).
- **Spatial groups.** The groups come from the processed subject's `normalization.npz` membership. Subject,
  voxel count and membership SHA256 are checked and recorded. Here `Q[v,g] = 1/√n_g` for voxel `v` in occupied
  4×4×4 bin `g`.
- **Base kernels.** `C_d = X̃_d Q` and `R_d = X̃_d − C_d Qᵀ`. Six base kernels are computed directly in
  float64: `K_C,d = C_d C_dᵀ` and `K_R,d = R_d R_dᵀ`. Kernels are never subtracted, which avoids cancellation.
  Cross-kernels for evaluation rows use the same fitted statistics.
- **Candidate kernel.** `K(η, τ) = Σ_d τ_d (K_C,d + η K_R,d)` with raw coefficients. The v2 alpha grid
  `logspace(−1, 7, 17)` applies to this raw kernel. This is identical to FUTURE_WORK_V3's normalized-kernel
  formulation with `λ_normalized = λ_raw / S_a`. Trace scales `s_g` are saved. `η = 1`, `τ = (1,1,1)`
  reproduces all-voxel ridge exactly.
- **Grid (28 candidates).**
  - `η ∈ {0, 0.1, 1, 10}`.
  - `τ ∈ {(1,1,1), (2,1,1), (1,2,1), (1,1,2), (1,1,0), (0,1,1), (0,0,1)}`.
  - Declared order: the baseline first, then τ in the listed order, then η in the listed order.
- **Ridge.**
  - The intercept is unpenalized, implemented by centering kernels on the fitting rows.
  - Float64 eigendecomposition. Eigenvalues ≤ `n·ε·λ_max` are dropped.
  - Negative eigenvalues below `−n·ε·λ_max` are reported with their size relative to `λ_max`, never clipped
    silently.

## 3. Arms

| Arm | Candidates | Regularization choice |
|---|---|---|
| V0 | baseline only | one α |
| V1 | baseline only | one α per GPT-2 output (768). An output with no finite CV score uses V0's α |
| V2 | `η ∈ {0, 0.1, 1, 10}`, `τ = (1,1,1)` | candidate and α |
| V3 | `η = 1`, all seven τ | candidate and α |
| V4 | all 28 | candidate and α |

Every arm makes its choice by the grouped CV of §4. Ties within 1e−12 go first to the declared candidate
order, then to the larger α. V1 records its α distribution, its grid-edge frequency and its fallback count.

## 4. Grouped cross-validation (shared by both tracks)

- **Fixed folds.** Training sessions are sorted by `sha256("ssltang-v3-folds:<session>")` and assigned
  round-robin to five folds: {9, 7, 10}, {15, 6}, {18, 11}, {12, 8}, {5, 3}. Their target counts are
  1,225 / 814 / 813 / 825 / 841.
- **Folds per budget.** Let `S_b` be the set of sessions that contain labeled rows in budget `b`.
  - If `|S_b| ≥ 5`, use the fixed folds restricted to labeled rows. Drop folds with no labeled held-out rows;
    at least 2 folds must remain.
  - If `2 ≤ |S_b| < 5`, use leave-one-labeled-session-out.
  - If `|S_b| < 2`, the result is **not estimable**, with no fallback.
- **Inside each fold.**
  - Normalization is fit on all target rows of the inner sessions.
  - Centering, trace scales and fitting use the labeled inner rows only.
  - Held-out rows are transformed with the inner statistics.
- **CV score.** Out-of-fold predictions for every labeled training row are scored with the session-macro
  within-run feature correlation (the primary endpoint). Runs with fewer than 2 labeled rows are omitted. V1
  uses the same score computed per output.
- **Final fit.** All labeled training rows; normalization on all training target rows.

## 5. Tracks, gates and the lock

**Performance track (development decision).** Validation session-macro r of each arm's final fit at 100% and
10% labels, in every subject. The 794 validation targets are disclosed as development exposure. The gates were
fixed before any score:

| Gate | Rule |
|---|---|
| G0 numerical recovery | (a) Fixture identities: kernel and cross-kernel decomposition, centering, and prediction equality with `GramRidgePath`, to ≤ 1e−10 relative in float64. (b) UTS01 V0 with Phase V's α reproduces the Phase V 10% and 100% predictions. Max abs difference ≤ 2e−3 × prediction RMS (they were stored in float16); validation r equal to 1e−4. |
| G1 (100%) | Mean over subjects of (V4 − V0) validation r ≥ 0.005, and V4 − V0 > 0 in each subject |
| G2 (10%) | Mean (V4 − V0) ≥ 0, and no subject below −0.005 |
| G3 mechanism | Among V2, V3 and V4 arms passing G1 and G2, advance V4 unless V2 or V3 is within 0.002 of V4's mean validation r. In that case advance the simpler one (V2 before V3). V1 is reported as the output-wise shrinkage explanation. If V1 also passes G1/G2 and is within 0.002 of the advanced arm, attribute the gain to output-wise shrinkage. If nothing passes G1, voxel ridge (V0) stays the decoder and V1–V3 are reported. |

**Lock.** The advanced arm, grid, folds and tie rule are written into §9 before any v3 test prediction is made.

**Conditional supervision track.** The locked procedure, and every arm for the mechanism ablation, is applied
at all five budgets using only that budget's training labels for CV. Results are reported on test with the shared
development exposure (794 validation targets per subject) disclosed. Results that are not estimable are reported
as such.

## 6. Controls (locked evaluation, test)

- **Circular-shift null.** v2 rule: ≥ 40 TR, 2,000 draws.
- **Disrupted training correspondence.** The advanced arm's selected configuration is refit at 100% with training
  targets rotated within each training run by ⌊n/2⌋ rows (≥ 15 rows = 60 TR in every run).
- **Wrong-run input.** Each test run's targets are scored against predictions from the next test run in run-ID
  order, cycled to equal length.
- **Nuisance decoders.** These are ridge decoders using the same CV.
  - Elapsed position: `p`, `p²` and `log(1+t)`, with `p = t/(n_tr−1)`.
  - An **oracle** speech-rate model: the word count and mean word duration in the target bin and in bins
    `t+1…t+3`, taken from transcripts.
  - Both together.
  - Residualization: a nuisance→target ridge is fit on training targets, and the advanced arm is refit on the
    residual targets. Test is scored on residuals from the training-fitted nuisance model. This uses oracle
    transcripts for the speech-rate part.
- **Supervision exposure.** For each budget: the union of transcript words used by target construction (words
  in selected bins plus their 20-word causal contexts). Reported as unique word instances, GPT-2 tokens and the
  duration of the union of timestamped spans, with the counting convention.

## 7. Dense training (separate data-budget factor, after the locked evaluation)

- Training targets at every TR phase that is otherwise eligible. Validation and test targets, the target
  definition and the 6-s input cutoff stay unchanged. Actual target counts and transcript exposure are recorded.
- V0 and the advanced arm's structure are run at 100%. `(η, τ)` stays fixed at the old-clock 100% selection;
  α is re-selected by the same CV.
- A dense V0 fit is profiled before the structured fit is attempted.
- This is reported as a changed data budget, not a same-label improvement.

## 8. Planned artifacts

`configs/v3/structured_ridge.yaml` and `results/v3/<subject>/<stage>/` hold cell JSONs with every CV score,
selection, fold, eigenvalue diagnostic and exposure count. `results/v3/REPORT.md` holds the summary. Prediction
arrays stay out of git.

## 9. Development results and lock

Written 2026-10-03, after development and before any v3 test prediction.

- **Rerun under the final code.** Development was rerun after the independent audit's provenance corrections:
  cache and membership verification, lock enforcement, saved evidence, estimability rules and the V1 tie
  tolerance. All 30 development cells reproduced the first run exactly, with identical selections and validation
  r to 1e−9.
- **Membership.** The coordinates of every subject were verified against the 4×4×4 bins.

Validation session-macro r:

| Subject | V0 100% | V1 100% | V2 100% | V3 100% | V4 100% | V0 10% | V4 10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| UTS01 | 0.1193 | 0.1169 | 0.1240 | 0.1199 | 0.1264 | 0.0453 | 0.0479 |
| UTS02 | 0.1315 | 0.1300 | 0.1357 | 0.1344 | 0.1399 | 0.0560 | 0.0604 |
| UTS03 | 0.1476 | 0.1464 | 0.1505 | 0.1492 | 0.1512 | 0.0705 | 0.0735 |

**Gates.**

| Arm | Mean Δ vs V0 at 100% | Mean Δ vs V0 at 10% | G1 (100%) | G2 (10%) |
|---|---:|---:|:-:|:-:|
| V4 | +0.0063 (+0.0071, +0.0083, +0.0036) | +0.0033 | passed | passed |
| V2 | +0.0039 | +0.0026 | failed | — |
| V3 | +0.0017 | +0.0017 | failed | — |
| V1 | -0.0017 | -0.0055 | failed | — |

- G0 passed under the precision-aware decomposition (§10).
- **G3: V4 is advanced.** Neither V2 nor V3 passes G1. V1 is worse than V0 in every subject and at both budgets,
  so output-wise shrinkage does not explain the gain.
- **V4 selections:** UTS01 100%: eta=0.1,tau=121, α = 3.16e+05; UTS01 10%: eta=0,tau=121, α = 1e+05; UTS02 100%: eta=0.1,tau=110, α = 1e+05; UTS02 10%: eta=0.1,tau=121, α = 1e+05; UTS03 100%: eta=0.1,tau=121, α = 3.16e+05; UTS03 10%: eta=0.1,tau=011, α = 3.16e+04.

**Lock.** [`experiments/v3_lock.json`](v3_lock.json), created `2026-10-03T21:42:59.444541+00:00`. Its contents:
- advanced arm `V4`, the §2 grid, the §4 folds and the tie rule;
- config `332f3e9e0674`;
- code `c76b1c77eb38` over 10 files.

The evaluate stage verifies these fingerprints and refuses to produce a test score if they changed.

## 10. Deviations

- **G0(b) tolerance mis-specified (2026-10-03, before any development score).** As written, the UTS01 V0
  recovery failed its prediction tolerance (max |Δ| = 0.47% of RMS against the 0.2% allowed) and a follow-up
  float16-ulp rule (`results/v3/UTS01/recovery.json`). Validation r agreed with Phase V to 1e−8. The cause is
  Phase V's float16 storage: rounding grows with each value, and the largest GPT-2 outputs set the maximum.
  The near-zero outputs also carry Phase V's float32 arithmetic error. A decomposition diagnostic
  ([v3_recovery_diagnostic.py](../scripts/v3_recovery_diagnostic.py); `recovery_diagnostic.json`) resolves it:
  - The v3 kernel path matches the original feature solver in float64 on the same real data to 1.3e−8 (100%)
    and 1.0e−8 (10%) of RMS.
  - The float32 solver differs from the float64 one by ≤ 1.5e−5.
  - 99.9% of stored values lie within half a float16 ulp of the recomputed float32 predictions.

  G0 is therefore treated as passed on the precision-aware criterion that FUTURE_WORK_V3 §6.4 asks for. The
  originally written numeric threshold was not one.
- **Dense profile (§7).** A dense V0 CV fold was profiled before the structured dense fit, as §7 requires:
  13,127 fitting rows, 133 s, 38.8 GB peak GPU memory (`results/v3/UTS01/dense/profile.json`). The dense
  runs then built each fold's kernels once and scored both V0 and the locked V4 structure from them. This is
  computationally equivalent to separate runs, because kernels do not depend on the candidate.
- **Label subsets differ across subjects below 100%.** This is inherited v2 behavior: the subset hash includes
  each subject's run IDs. Within a subject, every arm and every control shares one subset, so the paired
  contrasts are unaffected. Exposure counts are therefore reported per subject.

## 11. Fresh-recording replication in UTS09 and UTS10 (registered before download)

Written 2026-10-03, after the locked v3 evaluation and before any UTS09/UTS10 neural data was downloaded,
processed or scored. This implements FUTURE_WORK_V3 §7.2.

**Why these subjects.**
- ds003020 contains UTS01–UTS10. This repository, its git history and the remote volume contain data only for
  UTS01–UTS03 (`data/processed`, `data/semantic`, `results/`).
- UTS09 and UTS10 have the most data among unused subjects: 54 stories across about 10 sessions each.
  UTS04–UTS08 have 26–27.
- Their recordings are new. Their story texts overlap the UTS01–03 stimulus set, so this tests new neural
  data, not unseen text.

**Fixed procedure.**
- **Preprocessing:** the subject's own thick mask in 4×4×4 bins, train-only normalization, and the archived
  whole-session split (`preprocess.py`, seed 42, 15% / 15% validation / test session groups).
- **Targets:** the same GPT-2 pin and the v2 target index.
- **Decoders:** the **locked v3 procedure, unchanged** (code `c76b1c77eb38` plus the replication runner below).
  - **V0:** all-voxel ridge.
  - **V4:** the 28-kernel coarse/detail × delay grid. Kernel and α are chosen by session-grouped CV inside
    training.
  - Arms V1–V3 and all five budgets are reported as secondary.
- **Folds:** the fixed-fold *rule* of §4, the same sha256 ordering and round-robin into five folds, applied to
  each subject's own training sessions. The derived folds are recorded. No development or validation gate is
  rerun: the procedure is already locked.
- **Dense clock (§7):** every eligible TR. V0, and V4 with its 100% stride-4 structure fixed, with α
  re-selected by CV.
- **Controls (§6):** unchanged, together with the exposure accounting.

**Confirmatory contrasts.** Test session-macro r, each subject separately. The predicted sign is taken from
UTS01–03.

| | Contrast | UTS01–03 effects | Predicted |
|---|---|---|---|
| R1 | Dense-clock V0 − stride-4 V0 (100%) | +0.041 / +0.047 / +0.041 | positive |
| R2 | V4 − V0, stride-4, 100% labels | +0.0035 / +0.0046 / +0.0048 | positive |
| R3 | V4 − V0, dense clock | +0.0042 / +0.0048 / +0.0032 | positive |

**Decision rule.**
- A contrast replicates in a subject when its test difference has the predicted sign.
- It replicates overall when it does so in both UTS09 and UTS10.
- Story-level sign-flip p-values and the residualized (nuisance-removed) contrasts are descriptive.
- With two new subjects and few test sessions each, there is no population inference. R2 and R3 are small
  effects and may fail by chance; such a failure is reported as a failure.
- All arms, budgets and controls are reported regardless of sign.

## 12. Encoding-predictability voxel weighting (FUTURE_WORK_V3 §8.1; registered before implementation)

Written 2026-10-03, after §11, before any weighting model was implemented, fit or scored.

**Question.** Does weighting each (voxel, delay) column by its training-derived *encoding predictability*
improve semantic decoding beyond uniform all-voxel ridge (V0) and the locked structured decoder (V4)?
Predictability means how well the GPT-2 target at bin t predicts that voxel at t+d.

**Encoding predictability.**
- For each delay d ∈ {1, 2, 3}, a ridge model predicts every voxel at t+d from the 768-d target at t, using
  only labeled training pairs.
- A single α comes from the v2 grid, chosen by mean out-of-fold r over voxels.
- The score `r_vd` is the out-of-fold Pearson r under the session-grouped folds of §4.
- Scores are **nested** inside every decoder-CV fold: computed from that fold's inner labeled rows only.
- In the final fit, scores come from out-of-fold predictions over all labeled training rows.
- Validation and test data never enter a score.

**Arms.** All use the v3 feature standardization and the v3 alpha grid. The weights scale standardized
columns before the all-voxel kernel.
- **W0:** uniform weights. This is V0, refit in this stage.
- **W1 soft:** `w_vd = max(r_vd, 0)^γ`, γ ∈ {0.5, 1, 2}.
- **W2 hard:** uniform weight on the top q of columns by `r_vd`, q ∈ {10%, 25%, 50%}.
- **Fallback:** if fewer than 1% of columns have `r_vd > 0`, weights are uniform. A fallback is recorded.
- Weights are rescaled so that the mean of `w²` is 1.
- The choice within W1 and W2 (γ or q, plus α) is made by the §4 grouped CV.
- V4 is reported from the locked v3 cells for reference.

**Development gate** (validation, UTS01–03, stride-4 clock, 100% and 10% labels).
- Advance the better of W1 and W2 when, at 100%, its mean validation gain over W0 is ≥ 0.005 and positive in
  each subject, and at 10% its mean is ≥ 0 with no subject below −0.005.
- Otherwise, report a negative result and stop.
- A lock records the advanced arm and the code fingerprint before any test score.

**Locked evaluation.**
- Test on UTS01–03 and UTS09/UTS10, all exposed, across five budgets.
- Residualized (speech-rate) endpoints.
- A dense-clock factor: the selected γ or q is fixed, and α is re-selected.

**Fresh confirmatory replication, if advanced.** UTS04–UTS08 have never been used. They have about 26 stories
each, and their data are not yet downloaded. Each runs the locked pipeline with the archived split procedure.
- Contrast: advanced arm − W0, stride-4 clock, 100% labels.
- Predicted: positive.
- Confirmation needs a positive difference in at least 4 of the 5 subjects (exact one-sided sign test
  p = 0.19 for 4/5, 0.03 for 5/5). It is reported with the dense clock and residualized contrasts.

**Implementation details fixed before any development result** (2026-10-04). An independent code audit
(GPT-6.1-Sol) found gaps in the first implementation. The first development launch was stopped, and its
outputs were deleted without being read. The details below were fixed before development was rerun:
- **Hard rule:** keeps exactly round(q·n) of all n (voxel, delay) columns, highest r first. Ties are broken by
  column order (delay, then voxel index).
- **Encoding α:** ties within 1e−12 of the best mean r go to the larger α, as in v3.
- **Not estimable:** if nesting leaves fewer than two encoding folds, or a fold lacks fitting or held-out
  labeled rows, W1 and W2 are not estimable at that budget. W0 never uses encoding scores. The 1% fallback
  applies only to measured scores.
- **Residualized endpoint:** as in the v3 controls (§6). The selected rule, α and weights are fixed, and the
  ridge is refit on training targets minus the training-fit nuisance prediction.
- **Lock:** requires all 18 development cells (UTS01–03 × W0–W2 × 100%/10%) from the current code and
  configuration, and records their hashes.
- **Other subjects:** UTS09/UTS10 and UTS04–UTS08 run through `scripts/run_v3w_replicate.py`. It applies the
  §4 fold rule to each subject's own sessions, as in §11, and is part of the §12 code fingerprint.

### 12.1 Outcome (written 2026-10-04)

**Development** (validation, stride-4 clock, nested grouped CV). W0 reproduced the v3 V0 cells to about 1e−16
in all six cells.

| Subject | W0 100% | W1 100% | W2 100% | W0 10% | W1 10% | W2 10% |
|---|---:|---:|---:|---:|---:|---:|
| UTS01 | 0.1193 | 0.1702 (γ=2) | 0.1624 (q=10%) | 0.0453 | 0.0668 (γ=2) | 0.0622 (q=10%) |
| UTS02 | 0.1315 | 0.1971 (γ=2) | 0.1807 (q=10%) | 0.0560 | 0.0860 (γ=2) | 0.0743 (q=10%) |
| UTS03 | 0.1476 | 0.2075 (γ=2) | 0.1915 (q=10%) | 0.0705 | 0.0944 (γ=2) | 0.0864 (q=10%) |

**Gates.** Both weighting arms passed:
- W1: mean Δ at 100% +0.0588, positive in all three subjects; at 10% +0.0251.
- W2: +0.0454 at 100% and +0.0170 at 10%.

The better arm, **W1**, was advanced. The lock is [`experiments/v3w_lock.json`](v3w_lock.json): code
`6825989e5ffd` and config `332f3e9e0674`, with the hashes of all 18 development cells. It was committed
(`d2fe833`) before any test score and before the UTS04–UTS08 download.

**Locked evaluation (exposed test data).**

| Contrast | UTS01 | UTS02 | UTS03 | UTS09 | UTS10 |
|---|---:|---:|---:|---:|---:|
| W1 − W0, stride-4, 100% | +0.0614 | +0.0641 | +0.0570 | +0.0550 | +0.0574 |
| Residualized | +0.0595 | +0.0628 | +0.0545 | +0.0519 | +0.0543 |
| Dense clock | +0.0552 | +0.0600 | +0.0511 | +0.0560 | +0.0570 |

W1 led in 11/11 test stories in every subject. Smaller budgets are reported in
[results/v3w/RESULTS.md](../results/v3w/RESULTS.md). The gain is about zero below roughly 250 labeled targets
and plateaus above roughly 1,100.

**Fresh confirmatory replication (UTS04–UTS08).**

| Contrast | UTS04 | UTS05 | UTS06 | UTS07 | UTS08 |
|---|---:|---:|---:|---:|---:|
| W1 − W0, stride-4, 100% (registered) | +0.0583 | +0.0546 | +0.0544 | +0.0371 | −0.0018 |
| Residualized | +0.0523 | +0.0515 | +0.0486 | +0.0357 | −0.0004 |
| Dense clock | +0.0694 | +0.0552 | +0.0719 | +0.0583 | +0.0345 |

The registered contrast was positive in 4/5 subjects, so the result is **confirmed** under the registered rule.
The exact one-sided sign test gives p = 0.19, as noted at registration.

**Notes.**
- UTS04's validation session (6) is near zero for every decoder, while its training-CV and test sessions are
  normal. No §12 selection used validation.
- Exploratory, unregistered controls were disrupted training and wrong-run input. Both gave |r| ≤ 0.013 in all
  ten subjects.
- An exploratory, validation-only sweep beyond the grid covered UTS01–03 at 100%. Soft weighting peaks at
  γ = 2–3; CV was best at γ = 2 in all three subjects. Hard selection improves down to q = 2% and then
  matches soft γ = 2 within ±0.004 on validation. This is hypothesis-generating only.
