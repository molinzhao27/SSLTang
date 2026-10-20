# Protocol v2 (`ssltang-v2`): strong readouts, then latent versus reconstruction pretraining

**Written:** 2026-10-03, before any v2 model was fit or scored.
**Source plan:** FUTURE_WORK.md §5 Phases A–B.
**Status:** exploratory benchmark extension. UTS01's test stories were already examined in the
archived study, so v2 test scores on them are not confirmatory. Archived v1 results are unchanged.

Everything not listed here is inherited from [PROTOCOL.md](PROTOCOL.md): UTS01, the same
semantic manifest and whole-session split (61/11/11 stories; test sessions 4 and 14), 1,695 pooled
spatial features with train-only normalization, frozen GPT-2 layer-9 targets with 20 words of
context, the 4-layer width-384 encoder, BF16 fitting, and seeds 17, 23, 47.

## 1. Target identity (shared by every arm)

Targets are enumerated first; neural inputs are derived from them
([`src/targets.py`](../src/targets.py)). A target TR `t` is eligible when `t % 4 == 0`, its GPT-2
bin is valid, and a complete in-run 32-TR window exists for **both** response cutoffs
`c ∈ {3, 4}` TRs: `28 ≤ t ≤ n_tr − 5`. Changing the cutoff moves the neural window
`[t+c−31, t+c]`; it never changes labels, held-out targets, or retrieval candidates.

| Split | v2 targets | Archived lag-3 targets | Difference |
|---|---:|---:|---|
| train | 4,518 | 4,528 | 10 run-final targets without a `t+4` window |
| val | 794 | 796 | 2 |
| test | 802 | 806 | 4 |

All v2 targets are archived targets. Label subsets use the archived procedure on the common
targets (hashed contiguous 64-TR blocks, subset seed 2718, ceil rounding, nested):

| Budget | Training targets | Stories | Sessions |
|---:|---:|---:|---:|
| 100% | 4,518 | 61 | 11 |
| 25% | 1,130 | 50 | 11 |
| 10% | 452 | 27 | 9 |
| 5% | 226 | 13 | 8 |
| 1% | 46 | 3 | 3 |

The 794 validation targets are reusable selection supervision at every budget (the
"training-target budget" protocol of FUTURE_WORK §5 Phase C), disclosed with every result.

## 2. Endpoints

**Primary:** test session-macro feature correlation: mean over the 768 GPT-2 dimensions of the
temporal Pearson correlation within each held-out story, averaged over stories within session,
then over the two sessions ([`src/evaluation/endpoints.py`](../src/evaluation/endpoints.py)).
The archived run-macro mean is reported under its own name.

**Secondary:** centered within-run retrieval rank percentile (selected-label mean subtracted)
with ±10-TR and ±32-TR temporal exclusions; a within-run circular-shift null for the primary
endpoint (every allowed rotation is ≥ 40 TR in both directions, i.e., beyond the 32-TR input
window plus haemodynamic delay; 2,000 draws, seed 0); normalized MSE with selected-label
statistics; selected-label mean (B0) control. Top-k and MRR now use expected values under
cosine ties, as the transfer scorer already did; correlation and rank percentile are unaffected.

## 3. Phase A arms

| Arm | Features (inputs no later than `t+c`) | Readout options selected on validation |
|---|---|---|
| B0 | Selected-label target mean | none |
| B1 | Pooled BOLD at `t+c` (endpoint) or `t+1…t+c` (FIR) | alpha |
| B2 | Train-only PCA of the flattened 32-TR window ending at `t+c` | k ∈ {128, 512, 2048} × {unwhitened, whitened}, alpha |
| B3 | Frozen random encoder (`seed_everything(seed)` then the fine-tuning arm's exact initialization) | layer ∈ {1,2,3,4}, alpha |
| B4 | Frozen archived reconstruction encoders: `encoder_best.pt` (best validation reconstruction) and `encoder_latest.pt` (final update, 1,110 steps) | layer ∈ {1,2,3,4}, alpha |
| B5 | Existing full fine-tuning (random and `encoder_best.pt` inits) on v2 targets | existing best-validation-MSE checkpoint |

Encoder readouts take the frozen states of the window ending at `t+c` at its final position
(endpoint) or final `c` positions concatenated (FIR, positions for `t+1…t+c`). Layer 4 includes
the encoder's final LayerNorm. Transformer states are bidirectionally contextualized within the
window, so FIR weights are predictive temporal pooling weights, not haemodynamic responses.
PCA is fit on the 4,672 unlabeled 32-TR training windows the SSL encoders were pretrained on.

**Ridge procedure** ([`src/readout.py`](../src/readout.py), [`src/probe.py`](../src/probe.py)):
features are z-scored with statistics from all training-split inputs (no labels; PCA options
carry their own scaling instead). Ridge with an unpenalized intercept is fit on the budget's
labeled rows only, for alpha ∈ `logspace(−1, 7, 17)`. Within each (arm, encoder, seed, cutoff,
readout, budget) cell, the (option, alpha) with the highest validation session-macro feature
correlation is selected; only that option's test predictions are computed. Selected alphas at a
grid edge are flagged. Ridge solutions are deterministic, so B1/B2 have one fit per cell.

**Budgets:** ridge arms are cheap, so all five budgets are fit and reported. B5 uses 100% and
10% (2 inits × 2 cutoffs × 2 budgets × 3 seeds = 24 fits).

## 4. Phase A decision rules (fixed now)

1. **Linear benchmark:** within each latency, the B1/B2 cell with the higher 100% validation
   score is the raw/PCA benchmark carried forward.
2. **Encoder readout lock for Phase B:** the primary latency is 6 s (cutoff 3), for continuity
   with the archive. The readout (endpoint or FIR) is the one with the higher mean validation
   score of B4 `encoder_latest.pt` (the fixed-update export type used in Phase B) over the three
   seeds at 100% labels. Layer and alpha remain per-encoder validation choices from the grids
   above. The 8-s condition and the other readout are secondary analyses in Phase B.
3. Interpretation follows the gates of FUTURE_WORK §6; latency effects are reported as a
   tradeoff within each condition, never as an SSL gain.

## 5. Phase B: latent prediction versus reconstruction (specified before Phase A results)

Both objectives use [`src/training.py`](../src/training.py) with the same 61 training runs,
32-TR stride-4 windows, encoder, AdamW (lr 3e-4, weight decay 0.01, cosine schedule), batch
128, 30 epochs (1,110 updates), BF16 and student dropout 0.1. Temporal masks are unchanged
(one 8-TR core with 4-TR guards on each side; loss on the core only) and are drawn from a
dedicated mask generator seeded from the training seed, so both objectives see identical
windows in identical order with identical masks.

- **Reconstruction:** the existing linear head predicting masked neural values.
- **Latent:** a teacher initialized as a copy of the student, without gradients, always in
  evaluation mode, receiving the uncorrupted window. Targets are the teacher's final states
  passed through a parameter-free per-token LayerNorm. A linear predictor (384→384) maps
  student states to targets; MSE is averaged over masked-core positions (guards excluded). The
  teacher is updated after every completed optimizer step, `θ̄ ← mθ̄ + (1−m)θ`, with `m` rising
  linearly from 0.996 to 1.0 over the 1,110 updates.
- **Export:** the student encoder at the final update (`encoder_final.pt`) for both objectives.
- **Primary contrast:** latent − reconstruction test session-macro correlation with the locked
  frozen ridge readout at 100% labels, per seed and averaged over seeds 17, 23, 47.
  **Key secondary:** the same at 10%; all five budgets; the 8-s condition; retrieval;
  fine-tuning (existing procedure) from each final student at 100% and 10%.
- **Diagnostics each epoch on fixed validation windows:** validation objective loss; student
  (and teacher) per-dimension standard deviation across windows at fixed positions; effective
  rank (exp entropy of normalized singular values) at the final position; mean pairwise cosine
  across windows; content fraction = across-window variance at fixed positions ÷ total variance
  (positional embeddings alone give near zero). Losses of the two objectives are in different
  units and are not compared.
- **Collapse rule:** if the latent student's final effective rank is < 10 or its content fraction
  is < 0.05, the only permitted stabilizer is a mixed objective (latent + reconstruction, weight
  1:1), reported as a separate arm. Its weight is not tuned on any decoding score.

Paired summaries report every seed, per-story and per-session differences, and a story-level
sign-flip test that treats stories as exchangeable; stories share sessions, so that p-value is
descriptive. Two test sessions cannot support population inference.

## 6. Phase C: label coverage and counted supervision (specified before running)

Written after Phase A test results were known and before any Phase C fit. The method is fixed
to the Phase A linear benchmark (ridge on pooled BOLD at `t+1…t+3`, 6-s cutoff), because no
encoder readout reached it at any budget. Budgets 25%, 10%, 5% and 1% are crossed with two
allocations of the same target count: the archived hashed 64-TR blocks, and story round-robin
8-TR blocks (two targets per block, spreading labels over more stories). Each uses subset seeds
2718, 11, 12, 13, 14; the 100% budget is one fit. Alpha is chosen two ways: on the 794 reusable
validation labels (the Phase A–B protocol), and by leave-one-story-out cross-validation inside
the labeled subset, so the counted supervision equals the training labels. Story coverage,
session coverage and counted labels are reported with every score. Useful label savings would
require matching the 100% benchmark; with no prespecified noninferiority margin, Phase C reports
the fraction of the 100% score reached rather than claiming savings.

## 7. Phase D: prospective replication in UTS02 and UTS03 (registered before processing)

Written after UTS01 Phases A–B were complete and before any UTS02/UTS03 preprocessing,
semantic-target construction, fitting or scoring. Only the raw ds003020 downloads exist
(83 story runs and a subject thick mask each; download manifests in `data/raw`). Neither
subject has been used in any experiment in this archive. Their BIDS sessions agree with the
author session map, so the archived split procedure (seed 42, whole sessions) is expected to
reproduce UTS01's assignment: the replication tests new neural recordings of the same
stories, not unseen text.

**Procedure, fixed in full:** the UTS01 pipeline is rerun without changes per subject, with
subject-specific pooling, normalization, targets, models and selection. Stages:
`preprocess.py` (4-voxel cubes of the subject's own thick mask, train-only normalization,
seed 42); `build_semantic_targets.py` (same GPT-2 pin); Phase A ridge arms B0–B3
(`configs/v2/phase_a.yaml`, all cutoffs, readouts and budgets, random encoders for seeds
17/23/47); Phase B pretraining of both objectives for seeds 17/23/47 and the locked frozen
readout; full fine-tuning (existing procedure) from random, reconstruction and latent
initializations at 100% and 10%, cutoff 3; reconstruction controls. Code is the v2 code at
registration time, including the audit corrections recorded in §8.

**Confirmatory contrasts** (test session-macro r, 6-s cutoff, FIR readout, 100% labels; the
direction predicted from UTS01 in brackets):

| | Contrast | UTS01 effect | Predicted |
|---|---|---:|---|
| D1 | Frozen latent − frozen reconstruction (Phase B primary) | −0.0144 | negative |
| D2 | Raw BOLD ridge − frozen reconstruction encoder (Phase A gate) | +0.0252 | positive |
| D3 | Raw BOLD ridge − fine-tuned reconstruction initialization | +0.0564 | positive |
| D4 | Frozen reconstruction − frozen random encoder | +0.0026 | no prediction (report) |

**Decision rule:** a contrast replicates in a subject when the seed-mean session-macro
difference has the predicted sign and every paired seed difference has that sign. Each
subject is reported separately, then the two subjects side by side; with two test sessions
per subject and two new subjects, no population-level inference is made. The 10% budget,
other budgets, cutoffs and readouts are secondary and reported in full. Voxel-resolution
ridge (Phase V) is exploratory and may be added without changing these contrasts.

## 8. Corrections and deviations recorded during execution

- An independent audit (2026-10-03) found no leakage or misalignment affecting Phase A–B
  results. Four corrections were made after Phase A–B and before Phase C finished and Phase
  D started: validation-alpha scoring now drops near-constant columns with the reporting
  metric's relative rule (no saved Phase A–B selection changes: only two 1%-label cells
  selected the top alpha, in a flat saturated region); test predictions are now computed only
  after the validation search ends; paired summaries refuse incomplete stochastic seeds; the
  Phase B readout stage now evaluates exactly the requested, hash-verified encoders. Phase C
  was already running with the earlier code; its single-option raw-feature fits are unaffected
  by the selection-order change and have no near-constant prediction columns.
- The implemented EMA ramp is `m_k = 0.996 + 0.004 (k−1)/1110` for update `k`, so the final
  update uses 0.9999964 rather than exactly 1.0 as written in §5. The completed runs used this
  schedule, which is retained for reproducibility.
- 161 of 180 Phase A frozen-encoder cells selected layer 1 (13 layer 2, 2 layer 3, 4 layer 4);
  every 100%-label 6-s encoder cell selected layer 1.

## 9. Ridge-decoder modality transfer (registered before any ridge transfer score)

Written before the ridge decoder was applied to any ds004510 run. UTS01 transfer data were
viewed in the archived study, so its ridge-transfer scores are exploratory. UTS02 and UTS03
transfer runs have never been scored. They were preprocessed only with each subject's own
training normalization (14 runs each; Sintel has no reference and is unscored).

**Decoder, fixed:** the subject's Phase A benchmark: ridge on pooled BOLD at `t+1…t+3`, 100%
of training labels, feature standardization from training inputs, and alpha chosen on that
subject's ds003020 validation stories. Nothing is fit, normalized or selected on ds004510.
For each transfer run, predictions are made at every `t` with `t+3` inside the run and
averaged into one run vector. This 3-TR input avoids the archived short-prefix mismatch
(FUTURE_WORK §3.3). No timing is inferred for imagined words or film scenes.

**Scoring, unchanged from the archived transfer protocol:** cosine retrieval against the
fixed whole-reference GPT-2 vectors of the 5 imagined narratives and of the 3 films with
references. Primary variant: selected-training-mean centered. Report narrative-macro top-1
and MRR with the exact narrative-label permutation null (repeats move together; 120
permutations for imagined speech, 6 for films, so the film p-value cannot fall below 1/6).
The raw and constant-training-mean variants are controls.

**Decision rule:** imagined-speech transfer is supported in a new subject only if its
centered narrative-macro top-1 has permutation p ≤ 0.05. A transfer claim requires both
UTS02 and UTS03. Films are reported descriptively. No direction is predicted; all scores are
reported, including null and below-chance results.
