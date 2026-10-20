# v4 implementation specification

This is the engineering contract for [PROTOCOL_V4.md](PROTOCOL_V4.md) and [FUTURE_WORK_V4.md](../FUTURE_WORK_V4.md)
(the design, §3–§9). Where the two differ, the protocol wins.

## Conventions

- **Language and numerics.** Python 3.12, numpy and torch. All regression and kernel arithmetic is **float64**, so
  the float32 criterion of G0(f) does not apply. A `device` argument (`'cpu'` or `'cuda'`) is passed through.
- **Determinism.** No unseeded randomness. Every hash order is ascending hexadecimal SHA-256 of the registered
  string (PROTOCOL_V4 §2).
- **Reuse, don't edit.** Locked v3 and §12 code is never edited: `src/encoding_weights.py`, `src/voxel_kernels.py`,
  `src/structured_ridge.py`, `src/multikernel_ridge.py`, `scripts/run_v3*.py`, `src/evaluation/*`. In particular,
  `src.encoding_weights.encoding_scores(X, Y, rows, row_groups, folds, alphas, device)` is the pooled out-of-fold
  ridge routine for every "nested OOF r" in v4.
  - X is the predictor matrix [N, F].
  - Y is a float64 torch tensor [N, C] of outputs on the device.
  - `row_groups[i]` is a hashable group label.
  - `folds` is a list of tuples of held-out group labels.
  - It returns (r per output column, alpha, mean-r curve).
  - The protocol requires alpha ties to go to the larger alpha with tolerance 1e−12, and `encoding_scores`
    already does this.
- **Tests.** Tests live in `tests/test_v4_*.py`. They must run on CPU in seconds with small synthetic fixtures and
  must not need data files.
- **Data roots.** On the remote, `data/raw` holds the ds003020 H5 files, at `data/raw/<run['source_key']>`, with
  per-run `data` arrays of shape [n_tr, n_voxels]. Subject manifests are at
  `data/semantic/<S>/semantic_manifest.json`; each run has `run_id`, `story`, `session`, `split`, `n_tr` and
  `source_key`. Voxel coordinates (`coords`, shape [n_voxels, 3]) are in each manifest's `normalization_path` npz.

## Module A: `src/budgets.py`, `src/provenance.py`, `scripts/audit_v4.py`

### `budgets.py`

- `SUBJECTS = ('UTS01', …, 'UTS10')`.
- `hash_order(items, prefix) -> list`.
- `shared_stories(manifests: dict[subject] -> manifest) -> sorted list`: the stories present in every subject.
- `ds003020_budgets(target_manifest, shared) -> dict` with keys `'S3'`, `'S6'`, `'S15'`, `'ALL'` (lists of
  stories).
  - S3 and S6 are prefixes of `hash_order(train_shared, 'ssltang-v4-budget:')`.
  - S15 is all of `train_shared`. `train_shared` means the target's training-split stories within `shared`.
  - ALL is every training-split story.
- `isp_stories(budget_stories, shared)` returns `[s for s in budget_stories if s in shared]`, in budget order. ISP
  at ALL therefore uses the S15 stories.
- `inner_folds(budget_name, stories) -> list[tuple]`:
  - S3: leave-one-story-out.
  - S6: 3 folds.
  - S15: 5 folds.
  - S6 and S15 are round-robin over the given (hash) order.
  - LPP budgets (`'L3'`, `'L5'`, `'L8'`): leave-one-section-out.
  - ALL is not handled here; the caller uses the v3 fixed-fold rule.
- `half_groups(stories_of_rows, trs_of_rows, n_tr_by_story) -> list[str]` returns `f'{story}:{0 or 1}'`, where 1
  means `tr >= n_tr // 2`.
- `lpp_partition(participant_ids) -> (dev[9], confirm[rest])`, using hash prefix
  `'ssltang-v4-lpp-subject:'`.
- `lpp_section_budgets(test_section, sections) -> {'L3', 'L5', 'L8'}`: prefixes of
  `hash_order([s != t], f'ssltang-v4-lpp-section:{t}:')`.
- `class Ledger`: a dict-backed record serialized into every cell. Fields:
  - `target_pairs_dense`, `target_pairs_stride4`;
  - `target_bold_minutes`;
  - `transcript_words` (unique word instances including the 20-word causal context, given word lists);
  - `references` (list), `reference_bold_minutes`;
  - `reference_labeled_pairs` and `reference_rule` (`'none' | 'TS' | 'SS'`).

### `provenance.py`

- `class Provenance(target, regime, budget, val_items, test_items, ts_allowed=None)`.
  - Items are `(subject, stimulus)` pairs or stimulus names, plus optional TR ranges.
  - `record(subject, stimulus, kind, purpose, trs=None)` logs one read. `kind` is `brain`, `transcript` or `audio`.
    `purpose` is `decoder_fit`, `decoder_score`, `standardize`, `own_encoding`, `isp_response`, `isp_predictor`,
    `pca_basis`, `reference_model`, `converter`, `nuisance` or `oracle`.
  - `check()` raises `ProvenanceError` naming the rule.
  - `to_dict()` returns the log for the cell JSON.
- **Rules (from FUTURE_WORK_V4 §3):**
  - **R1.** A target transcript or audio read is allowed only for budget stimuli. The exception is purpose
    `oracle`, which is allowed and flagged.
  - **R2** (regime S). Target BOLD outside the budget may be read only for purpose `decoder_score` on validation
    or test stimuli.
  - **R3.** Reference BOLD is allowed for `isp_predictor` and `converter` only at budget stimuli and times, and for
    `pca_basis` on any stimulus that is not the target's validation or test material.
  - **R4.** A reference transcript is allowed only for `reference_model`, and only if the stimulus is in
    `ts_allowed` (TS) or, under SS, not a target validation or test stimulus.
  - **R5.** Nothing derived from a target validation or test stimulus (any subject) may feed a prior, converter
    or reference model.
  - **R6** (LPP) is enforced by the target builder; it records context resets as `kind='transcript'`,
    `purpose='context_reset'`.
- **`scripts/audit_v4.py <cell.json ...>`** re-runs `check()` on saved logs and exits non-zero on any violation.
- **Tests (G0e):** planted leaks, each of which must raise:
  - a test-section reference response used for ISP;
  - an out-of-budget target transcript;
  - a reference model containing a target test story;
  - cross-section GPT-2 context;

  plus clean cases that must pass.

## Module B: `src/priors/isp.py` (with `src/priors/__init__.py`)

- `load_bold(raw_root, run) -> float32 [n_tr, V]`. Applies `nan_to_num`, then a per-run z-score over time
  (std floor 1e−6).
- `reference_basis(bold_runs: list[array], n_components=100, seed=0, device) -> (mean[V], comps[V, K])`.
  - PCA of the concatenated per-run z-scored BOLD.
  - Uses the exact SVD when rows ≤ 6000. Otherwise it uses `torch.svd_lowrank(q=K+20, niter=6)` with
    `torch.manual_seed(seed)`, and K is truncated afterwards.
  - Computed on GPU in float32 for the decomposition only.
- `lagged(z, lags=(-1, 0, 1))`: zero-padded within a run.
- `predictors(ref_runs_by_story: dict[ref] -> dict[story] -> array, bases, stories, lags) -> [T, n_refs*K*len(lags)]`
  concatenates over stories in the given order and over references in sorted order.
- `isp_scores(target_runs_by_story, predictors, stories, alphas=np.logspace(0, 10, 21), device, groups='half')
  -> (r[V], alpha, curve)`.
  - The response is the concatenated target BOLD over `stories`.
  - Groups are half-story (`budgets.half_groups`). With `groups='pilot'` they are `story_index % 5`, for the G0(c)
    pilot reproduction.
  - Uses `encoding_scores`.
- `misaligned(pred_runs)`: circularly shifts each reference's per-run predictor block by `n_run // 2` rows, before
  lagging.
- `ladder(p, f, seed)`: for `f > 0`, returns `p + N(0, Var(p)·(1/f − 1))` with `np.random.default_rng(seed)`; for
  `f == 0`, returns a permutation of p.
- `self_scores(target_runs_by_story, coords, stories, n_components=100, margin=4, alphas, device)`.
  - Octants come from median splits of x, y and z.
  - For each octant, the predictors are the top-K PCs (lags −1..1) of target voxels whose coordinates are outside
    the octant expanded by `margin` voxels.
  - The outputs are the octant's voxels, scored with half-story OOF r.
- **Tests:**
  - ISP agrees with a direct numpy ridge OOF reference on a tiny fixture.
  - Lag padding does not cross runs.
  - `misaligned` breaks alignment while keeping per-run content.
  - Ladder second moments: Var(p') = Var(p)/f in expectation, and corr(p, p')² ≈ f.
  - The octant mask excludes the margin.

## Module C: `src/priors/eb.py` and `scripts/v4_simulate.py`

- `fisher(r) = atanh(clip(r, -0.999, 0.999))`.
- `null_sigma2(targets, design, rows, groups, folds, alphas, device, run_ids, shifts=(1, 2, 3, 4)) -> float`.
  - Recomputes `encoding_scores` with the targets of `rows` circularly shifted within each run (in `run_ids` order)
    by `round(k * n_run_rows / 5)` rows.
  - Returns the mean over k of `(1.4826 * MAD(fisher(r0_k)))**2`.
  - It is called once per delay.
- `eb_posterior(z, p, sigma2) -> (m, info)`, with z and p of length C:
  - OLS of z on p gives (a, b). If b < 0, set b = 0 and a = mean(z).
  - V = Var(z − a − b·p); τ² = max(V − σ², 0.01·V); κ = τ² / (τ² + σ²).
  - m = a + b·p + κ·(z − a − b·p).
  - `info` holds a, b, τ², σ², κ and V.
- `eb_weights(m, gamma)` returns `max(tanh(m), 0) ** gamma` (unnormalized; the caller rescales).
- `mixcv_posterior(z, p, a, b, kappa)` gives the fixed-κ variant for MIX-CV.
- **`scripts/v4_simulate.py`** (G0d, run on CPU or GPU).
  - Grid: n ∈ {100, 500, 2000} rows, 250,000 columns, 2% informative columns with heavy-tailed true ρ (Student-t,
    df=3, scaled so that the 99th percentile of |ρ| is 0.3), prior reliability R_P ∈ {0.1, 0.3, 0.6}.
  - The observed own r is generated as sample correlations at n.
  - The prior is p = c·atanh(ρ) + noise, with noise chosen so the squared correlation of p with atanh(ρ) equals
    R_P.
  - σ² comes from a shifted-null analogue (independent noise columns at the same n).
  - Writes `results/v4/simulate.json`. It must show EB MSE ≤ min(raw MSE, prior-only MSE) in every cell and
    |κ̂ − κ_true| ≤ 0.05, where κ_true is from the known variances.
- **Tests:**
  - κ = 1 reproduces z, so W1's weights are reproduced (G0b).
  - With a = 0, b = 1 and κ = 0, it reproduces p, so PRIOR's weights are reproduced (G0b).
  - Negative-slope handling.
  - `null_sigma2` agrees with a hand computation on a fixture.

## Module D: `src/priors/acoustic.py` (P-AC)

- Per-TR low-level features for a story: a 32-band mel envelope (log, mean over each TR), RMS, F0 (median voiced
  F0 per TR, 0 if unvoiced; use `librosa.pyin` if available, else a deterministic autocorrelation F0), word rate,
  phoneme rate, and word-onset count.
  - Audio comes from `ds003020/stimuli/<story>.wav`, fetched by the caller.
  - Timing comes from TextGrids, read with `src.transcripts` and `src.semantic.features` helpers.
  - TR alignment uses the manifest's `tr_start_seconds` (11.0) and `tr_seconds` (2.0), the same as the semantic
    targets.
- `acoustic_targets(story_audio, textgrid, n_tr, tr_start, tr_seconds) -> [n_tr, F]`, z-scored per story.
- The nested out-of-fold encoding r for each design column uses `encoding_scores`, the same as W1.
- **Tests** use a synthetic tone and click track and a synthetic TextGrid.

## Module E: `scripts/summarize_v4.py` statistics core (`src/v4_stats.py`)

- `sign_flip_p(deltas, n=100000, seed=20261003, one_sided=True)`: p = (1 + #{T* ≥ T}) / (n + 1), with T the mean.
- `bca_ci(deltas, n=10000, seed=20261004, level=0.95)`.
- `iut(p1, p2, mean1, mean2, alpha=0.025)`.
- `fixed_sequence(results, order)` stops at the first failure.
- `crossed_lmm(df)` uses statsmodels MixedLM with a subject random intercept and a section variance component.
- `sesoi_tier(ci90_low, ci90_high, sesoi=0.005)`.
- `exact_binomial_ci`.
- **Tests:** against hand or scipy references.

## Module F: `scripts/lpp_prepare.py` (Le Petit Prince EN, DEV subjects only before the lock)

- **Data.** Downloads from OpenNeuro ds003643 only:
  - the 9 DEV participants (`budgets.lpp_partition` over the EN participant ids);
  - annotations (`annotation/EN/lppEN_word_information.csv`);
  - the EN audio (`stimuli`, for P-AC);
  - the participants file.
- **Verification (G4a).** Section mapping by sorted run number, verified against the TR counts
  282/298/340/303/265/343/325/292/368. Word counts per section are verified as well.
- **Mask.** An MNI gray-matter mask resampled to the 73×90×74 grid, intersected with voxels that are finite and
  non-constant in all DEV subjects. It is saved as `data/lpp/mask.npz` with coords.
- **Outputs**, per subject and section, in the ds003020-compatible layout:
  - `data/lpp/raw/<sub>/<section>.hf5` with `data` [n_tr, V_mask] float32;
  - a manifest `data/lpp/semantic/<sub>/semantic_manifest.json` whose runs carry `run_id`, `story=section`,
    `session=section`, `split='train'` (outer folds are handled by the runner), `n_tr`, `source_key`,
    `tr_seconds=2.0`, and `tr_start_seconds` set by the verified audio-onset convention;
  - GPT-2 layer-9 targets built by the existing `src.semantic.features` (`FrozenGPT2`, `downsample_words`), with
    the 20-word context **reset at each section start** (R6).
- **Picture exclusion.** A per-TR boolean `picture` array marks section 1, 0–80 s.
- **Provenance.** File checksums are written to `data/lpp/prepare_provenance.json`. Only DEV subjects are
  downloaded, and the script refuses CONFIRM subjects unless `--confirm` is given **and**
  `experiments/v4_lock.json` exists.

## Post-review notes (2026-10-04)

The modules were implemented and adversarially reviewed: two reviewers per module, then fixes. The final APIs
differ from the text above as follows. Each module's docstrings are authoritative.
- **`null_sigma2`** requires `tr` and clamps short-run shifts (PROTOCOL_V4 §4 item 6).
- **`isp.design`** builds the ISP design from cached component scores. `self_scores` returns `(r, info)`.
- **`acoustic.story_features(..., return_info=True)`** feeds `acoustic_rows` and `acoustic_scores`. F0 is
  computed by autocorrelation.
- **Provenance** logs carry the registered inner folds, and every per-fold read is tagged with
  `inner_fold(i)` (rule R7). ORACLE reads live in a separate `arm='ORACLE'` log. Secondary arms need their own
  logs with `reference_rule` TS or SS.
- **`lpp_prepare.py`**: a CONFIRM run needs `--lock-sha256` and a committed, pushed lock (PROTOCOL_V4 §4
  item 10).

## Phase 2 modules (written 2026-10-04 while ds003020 development runs)

**Frozen core.** The development cells carry `run_v4.code_sha256()` over `run_v4.CORE`:
- `src/budgets.py`, `src/provenance.py`, `src/priors/{__init__,isp,eb,acoustic}.py`, `src/v4_stats.py`;
- `src/windows.py`, `src/runtime.py`, `src/transcripts.py`;
- `scripts/run_v4.py`, `scripts/audit_v4.py`, `scripts/lpp_prepare.py`;
- the frozen v3 and §12 files.

**None of these may be edited from now on**, or every development cell becomes invalid. New code lives in new
files, which import the core (subclass or wrap it, never patch it). If a core change is unavoidable, stop and
report it.

### Module G: `scripts/run_v4_lpp.py` (Le Petit Prince EN; FUTURE_WORK_V4 §4.2, §5–§7, G4)

**Data.** As prepared by `scripts/lpp_prepare.py` on the remote:
- `data/lpp/semantic/<sub>/semantic_manifest.json`. Runs are the 9 sections, with `story=session='1'…'9'`.
  Each run has `source_key` (under the manifest's `raw_root`), `target_path`, `target_valid_path`,
  `stimulus_path` (TextGrid), `audio_path`, `picture_path` (a per-TR bool) and `tr_start_seconds`.
- The mask and coordinates are in `data/lpp/mask.npz` (`coords`, `membership`).
- `src.targets.TargetIndex` rejects these manifests (it accepts ds003020 only and is frozen).

**`LppIndex(manifest_path, stride, window=32, phase=0, cutoffs=(3, 4))`.** A duck-typed equivalent of
TargetIndex:
- `.split = 'all'`, `.runs`, `.items`, `.ids`, `.run_ids`, `.target_trs`, `.sessions`, `.targets(positions)`,
  `.signature()` and `len()`.
- Eligibility is identical to TargetIndex.
- Bins whose `picture` flag is true (section 1, 0–80 s) are excluded from training and scoring.
- `src.voxel_kernels.cache_delay_rows` accepts it. Caches go in `data/cache/v4_lpp/<sub>/{dense,stride4}`.

**Outer CV and budgets.**
- Outer CV: for each test section t (1…9), the budgets are `budgets.lpp_section_budgets(t, sections)`
  (L3, L5, L8).
- Inner CV: `inner_folds(name, B)` (leave-one-section-out).
- Scoring: the stride-4 bins of section t. The endpoint is `src.evaluation.endpoints.score`, with each section as
  its own session. The section-macro r is therefore the section's r.
- Reported per cell: `R_s(arm) = mean over the 9 outer folds`.

**Engine.** Reuse `run_v4`'s `Budget`, `Priors`, `References`, `run_cv`, `final_fit`, `fold_maps`,
`arm_weights`, `soft_weights` and `report` by subclassing and composition: an `LppTarget` with the attributes
`Target` exposes, eval rows restricted to section t, and `References` over LPP subjects.
- **References:**
  - DEV targets use the other 8 DEV subjects.
  - CONFIRM targets use all 48 other EN listeners (DEV and CONFIRM), brain only.
  - PCA bases per (reference, outer section t) come from the reference's sections ≠ t.
- **ISP** uses the budget sections. **SELF** uses `mask.npz` coordinates. **P_AC** uses the run's `audio_path`
  and TextGrid.
- **Provenance:**
  - Dataset `'lpp'` (inferred from the subject id), budget B, `val_items=[]`, `test_items=[t]`.
  - The registered folds, `budget_name` L3/L5/L8, and stage `develop` (DEV) or `evaluate` (CONFIRM).
  - Every target transcript read must be preceded by a `('transcript', 'context_reset')` record for that section
    (R6).
- **Tier-1 arms:** the same as ds003020, plus TH_SEL and XDEC (Module H) under the SS rule.

**Residualized endpoint (H2), per outer fold.**
- The nuisance model is position plus oracle speech rate, from TextGrid words via `src.transcripts.run_words`
  and `nuisance_features`, or their LppIndex-compatible equivalents. It is ridge-fit on B with the inner folds,
  and alpha is chosen as in `run_v3.feature_ridge_cv`.
- Residual targets are formed on train and test.
- The selected (arm, γ, α) and weights are kept fixed. The ridge is refit on the residual training targets and
  scored against the residual test targets.
- This is done for every arm in the confirmatory family (W0, W1, EB, PRIOR, EB_SELF, TH_SEL and the ladder).

**Stages.**
- `prepare-caches --subjects …`: builds the dense and stride-4 caches.
- `develop --subject <DEV>`: all outer folds, budgets, clocks and Tier-1 arms.
- `g4 --subjects <9 DEV>`:
  - (b) dense W0 at L8 beats its within-section circular-shift null q95 (≥ 40 TR, 2,000 draws, seed 0) in at
    least 6/9;
  - (c) every ISP alpha is interior.
- `confirm --subject <CONFIRM>`: refuses to run unless the lock exists and matches this file's fingerprint, the
  run_v4 core and the analysis script.

**Cells.** `results/v4_lpp/<sub>/<clock>/<stage>/cells/t<t>_<budget>_<arm>.json`, plus `residualized` blocks.

**Fingerprint.** `LPP_CODE = ('scripts/run_v4_lpp.py', 'src/priors/tanghuth.py')` + `run_v4.code_sha256()`.

### Module H: `src/priors/tanghuth.py` (published comparators, after Tang & Huth 2025)

**References' labels** follow rule TS or SS:
- **TS:** stories the target never heard; ds003020 targets UTS04–UTS10 with references UTS01–UTS03.
- **SS:** any material except the target's validation or test material; the LPP default.

**TH_SEL.**
1. For each reference, its top-10,000 voxels come from out-of-fold encoding r: W1's encoding model on the
   reference's TS/SS labeled rows.
2. A reverse ridge converter maps the reference's top-10k voxels (lags −1..+1) to all target voxels. It is fit
   on target and reference BOLD at B times, with alpha by grouped out-of-fold r.
3. Each reference's converter is applied to that reference's BOLD on material the target did not hear: TS stories
   in ds003020, or out-of-budget non-test sections in LPP.
4. The score per target voxel is the mean pairwise correlation of the references' predictions.
5. Weights are uniform on the top 15,000 voxels (ds003020) or the top 17% (LPP), at all delays.

**TH_SEL_rand** is the same-size random mask, with a seed from `sha256('ssltang-v4-thsel-rand:<subject>')`.

**XDEC (zero target labels).**
1. Each reference gets a W1 decoder (§12 rule) on its top-10k voxels, trained on its TS/SS labels.
2. A ridge converter maps target BOLD (all voxels, lags −1..+1) to those reference voxels. It is fit on B times,
   with alpha by grouped out-of-fold r.
3. Predictions for the target's eval rows are decoder(converter(target BOLD)), averaged over references.

**Provenance.** Each comparator writes its own log through
`Provenance(..., arm='TH_SEL' | 'XDEC', reference_rule='TS' | 'SS', ts_allowed=...)`, so reference transcripts
are confined to secondary arms (R4).

**`scripts/run_v4_tanghuth.py`** runs the ds003020 published-method block:
- targets UTS04–UTS10 at budgets S3, S6 and S15 (develop and evaluate);
- the TS rule, with references UTS01–UTS03;
- TH_SEL, TH_SEL_rand and XDEC.

It uses run_v4's engine for the decoder of TH_SEL. Cells go in `results/v4/<S>/<clock>/<stage>/cells/`
`<budget>_<arm>.json`.

### Module I: `scripts/summarize_v4.py` (primary analysis, hashed into the lock) and `scripts/v4_lock.py`

**`summarize_v4.py --confirm`** reads the LPP CONFIRM cells.
- **Primary contrast** at the locked primary budget (L3, or L5 after a G1 fallback), on the dense clock with the
  raw endpoint:
  - per subject, R_s(arm) averaged over the 9 outer folds;
  - Δ1 = EB − W1 and Δ2 = EB − W0. If PRIOR was advanced, PRIOR replaces EB.
- **Statistics,** using `src/v4_stats`:
  - a one-sided sign-flip test for each Δ;
  - the intersection-union decision (both p < 0.025 and both means > 0);
  - for each contrast: the mean, BCa 95% CI, d_z, number of positive subjects with an exact CI, and the oracle
    fraction;
  - SESOI tiers from the 90% CI;
  - the crossed LMM on per-(subject, section) Δ.
- **Fixed sequence,** run only after H1 is confirmed:
  - **H2:** the same intersection-union test on the residualized endpoint;
  - **H3:** the mean per-subject OLS slope of (EB[f] − W1) on f ∈ {0, 0.25, 0.5, 0.75, 1} (f = 1 is EB) > 0, by
    sign-flip;
  - **H4:** EB − EB_SELF > 0;
  - **H5:** EB − TH_SEL > 0;
  - **H6:** at L8, mean(EB − W1) > −0.005, by sign-flip on Δ + 0.005.
- **Outputs:** `results/v4/primary.json` and `results/v4/PRIMARY.md`. The script runs exactly once.
- **Descriptive modes:** `--dev` (LPP DEV) and `--ds003020` (budget curves).

**`v4_lock.py`** writes `experiments/v4_lock.json`. It refuses to write unless `results/v4/g0.json` and
`results/v4/gates.json` exist under the current core. It records:
- `advanced_arm`, `primary_budget` (ds003020 and LPP) and `drop_h6`, from gates.json;
- `code_sha256` (the run_v4 core), `lpp_runner_sha256`, `tanghuth_sha256`, `summarize_sha256` and
  `config_sha256`;
- the sha256 of every ds003020 and LPP DEV development cell;
- `lpp_partition`, `lpp_mask_sha256` and `lpp_code_sha256` (`lpp_prepare.code_fingerprint()`, checked against
  `data/lpp/prepare_provenance.json`);
- the development d_z and SD(Δ) at the primary budget (ds003020 dev and LPP DEV);
- the R_P(K) fit on DEV with its K = 48 prediction;
- the hash orders and seeds (by reference to PROTOCOL_V4 §2).
