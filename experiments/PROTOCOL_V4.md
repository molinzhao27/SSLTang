# Protocol v4: inter-subject and empirical-Bayes voxel priors for few-label semantic decoding

**Registered:** 2026-10-04. This was written before any v4 decoding score, before any Le Petit Prince (LPP) data was
downloaded, and before any v4 implementation beyond the feasibility pilot.

**Design document:** [FUTURE_WORK_V4.md](../FUTURE_WORK_V4.md), SHA-256 `ecc5c5ab69203f71fba52ee08bc602d812394f0981c43595b5b6060f9e4b9a40`.

## 0. Exposure before registration

- **ds003020 (UTS01–UTS10).** Development cells for UTS01–03 were scored in v3 and §12. **Every subject's test
  stories were scored** with W0 and W1 in §12. Test numbers in v4 are therefore an **exposed benchmark**, reported
  descriptively.
- **v4 feasibility pilot** ([results/v4_pilot](../results/v4_pilot/UTS01_isp_summary.json), commit `240f223`).
  - Data: UTS01, the 15 training stories shared by all ten subjects; training data only.
  - Results: the ISP map against the full-label encoding map gave Spearman 0.539 and 59% top-2% overlap.
  - No decoding endpoint was computed.
- **LPP (ds003643).** Never downloaded or analyzed by this project. The literature and dataset sweep read only
  metadata, file headers and the published descriptor.

## 1. Registered design

**Incorporated verbatim:** sections 3–9 of FUTURE_WORK_V4.md (as hashed above). These cover:
- budget currency and rules R1–R6;
- datasets, splits, budgets and the LPP DEV/CONFIRM partition;
- arms and the EB estimator;
- the primary contrast and decision rule (H1) and the fixed-sequence secondaries H2–H6;
- development gates G0–G4 and the advancement table;
- secondary analyses;
- controls.

The decisions listed at the top of FUTURE_WORK_V4.md (items 1–9) are part of this registration.

## 2. Constants fixed now

**Hash orders.** All orders are ascending hexadecimal SHA-256 of the given string:
- ds003020 story budgets: `ssltang-v4-budget:<story>`, over the target's training-split stories shared by all ten
  subjects. S3 and S6 are prefixes. S15 is all of them (13 for UTS09/UTS10). ALL is every training story.
- LPP subject partition: `ssltang-v4-lpp-subject:<participant_id>`. The first 9 are DEV and the remaining 40 are
  CONFIRM.
- LPP section budgets, within outer test section *t*: `ssltang-v4-lpp-section:<t>:<s>` over the 8 sections
  s ≠ t. S3, S5 and S8 are prefixes.

**Folds.**
- **ds003020 inner decoder CV:**
  - S3: leave-one-story-out;
  - S6: 3 folds;
  - S15: 5 folds;
  - in all three, folds are assigned round-robin over the budget's hash order;
  - ALL: the v3 §4 rule (UTS01–03) or the §11 rule (other subjects).
- **LPP:** the outer CV is 9-fold leave-one-section-out; the inner decoder CV is leave-one-section-out within B.
- **Nested encoding, ISP, SELF and P-AC scores:** out-of-fold over **half-story or half-section groups**. Each story
  or section in the fitting part is split at its middle TR, giving two groups per story or section.

**Grids.**
- Decoder α: logspace(−1, 7, 17).
- Soft-rule γ: {0.5, 1, 2}.
- Encoding α: the decoder grid.
- ISP, SELF and converter α: logspace(0, 10, 21).
- Ties go to the larger α, with tolerance 1e−12, as in §12.

**ISP predictors.**
- Each reference contributes its top-100 principal components, at lags −1, 0 and +1 TR, concatenated over
  references.
- **PCA basis for each (reference, target, outer fold):** the reference's BOLD on all material except the target's
  validation and test stories (ds003020) or the outer test section (LPP). It is z-scored per run.
- **Target responses:** z-scored per run, using only rows in the fitting part of B.

**EB null shifts.** The training targets are shifted circularly within each run by round(k·n_run/5) rows,
k = 1…4. Every shift is at least 40 TR in every ds003020 story and LPP section.

**Noise ladder.** Seed = int(sha256(`ssltang-v4-ladder:<subject>:<f>`)[:8], 16), for f ∈ {0.75, 0.5, 0.25}. For
f = 0, the voxels are permuted with seed = int(sha256(`ssltang-v4-perm:<subject>`)[:8], 16).

**Statistics.**
- Sign-flip: 100,000 Monte Carlo flips with seed 20261003.
- BCa bootstrap: 10,000 resamples with seed 20261004.
- Crossed LMM: statsmodels MixedLM with variance components for section and a subject random intercept. If it does
  not converge, use lme4 through rpy2 with Satterthwaite df. The choice is recorded.

**Endpoints.**
- **Primary:** the v2 session-macro feature correlation (`src/evaluation/endpoints.py`), with sections in place of
  sessions in LPP.
- **Exclusions:** the v3 retrieval exclusions apply. LPP bins overlapping 0–80 s of section 1 are excluded.

## 3. Code fingerprint and lock

- **Before development.** The v4 code (all files under `src/priors/`, plus `src/budgets.py`,
  `scripts/run_v4.py`, `scripts/audit_v4.py`, `scripts/lpp_prepare.py`, `scripts/summarize_v4.py`, and the v3
  and §12 files they import) is fingerprinted. An independent audit must be completed (G0g).
- **Development cells** record the fingerprint. The lock refuses cells from other code.
- **The lock** (`experiments/v4_lock.json`) is committed and pushed to main before any ds003020 test score under
  v4 and before any CONFIRM subject is downloaded. It contains everything listed in FUTURE_WORK_V4 §7.
- **The primary analysis** runs exactly once, from the hashed `scripts/summarize_v4.py`.

## 4. Deviations

Any departure from this registration is recorded here, with its date and reason, before the affected result is
read.

- **2026-10-04, clarifications before any v4 implementation result.** These resolve details the design left open
  or made computationally infeasible. No v4 decoding score existed.
  1. **The ALL budget.** It is used only for G3 and descriptive curves, and three things change there:
     - Nested own-encoding scores use the §12 rule: the decoder folds (v3 §4 or §11 session folds) minus the held
       fold. Half-story groups would require about 160 encoding fits per decoder fold.
     - ISP at ALL uses the budget's shared stories, the S15 set, because references did not hear the non-shared
       stories.
     - Only W0, W1, PRIOR and EB are run.
  2. **G0(a).** At ALL with the stride-4 clock, the v4 runner must reproduce the §12 development cells for W0 and
     W1. This works because ALL then uses the §12 fold and normalization rules. The S3, S6 and S15 budgets use the
     registered half-story groups.
  3. **Numerics.** All kernels and regressions are float64, so G0(f), which applies only to float32 kernels, is not
     applicable.
  4. **Inner decoder-CV scores** use the v3 statistic: the macro over sessions of per-run mean feature r. The §12
     tie rules apply over γ ∈ {0.5, 1, 2} in that order, then the larger α.
  5. **EB-MISALIGN shift.** The registered rule is "at least 40 TR". It is implemented as a circular shift of each
     reference's per-run predictors by n_run // 2 rows, which is at least 102 TR.

- **2026-10-04, decisions from the module reviews, before any v4 decoding score.**
  6. **EB null shifts.** The §2 sentence "every shift is at least 40 TR" is false for short runs. The registered
     shift round(k·m/5) falls below 40 TR on the dense clock for stories under about 230 TR (e.g. *inamoment*,
     205 TR, a shared story).
     - **Rule:** each run's shift is clamped into [lo, m − lo] rows, where lo = ⌈40 TR / TR-per-row⌉, so every
       shifted pairing, wrapped ones included, is at least 40 TR apart.
     - **Long runs:** shifts are unchanged.
     - **Impossible clamps:** a run too short for any valid shift raises an error rather than being silently
       accepted.
     - **Recording:** realized shifts and offsets are recorded per fold.
  7. **G0(d) synthetic recovery.** True ρ is heavy-tailed: Student-t (df = 3), scaled over all columns so that
     the 99th percentile of |ρ| is 0.3, and capped at |ρ| ≤ 0.9. The cap binds on 10.9% of informative columns
     and carries 55.5% of Var(atanh ρ).
     - **Gate reading:** the 'all-columns' scaling is the gate reading. It was chosen after both readings had been
       run, and both pass the registered prefix.
     - **Sensitivity block:** the informative-column reading and caps 0.99 and 0.999 are reported descriptively
       in `results/v4/simulate.json`.
  8. **SESOI tiers** use an 8-way partition of the 90% CI (`src/v4_stats.TIERS`):
     - meaningful;
     - positive but below SESOI;
     - positive, magnitude undetermined;
     - equivalent to zero;
     - the three negative mirrors;
     - inconclusive.
  9. **P-AC F0** is computed by deterministic autocorrelation (Boersma 1993; `F0_METHOD = 'autocorr'`), not
     pyin.
  10. **LPP CONFIRM download** requires all of the following:
      - `--confirm`;
      - `--lock-sha256` equal to the committed and pushed `experiments/v4_lock.json`;
      - a lock carrying `lpp_partition`, `lpp_mask_sha256` (content digest) and `lpp_code_sha256` (the
        preparation code).

      Post-lock DEV reruns may not change recorded outputs.
  11. **LPP DEV preparation facts** (G4(a), recorded in `data/lpp/prepare_provenance.json` on the remote):
      - All 9 DEV subjects have the registered section TR counts.
      - The mask holds 140,369 voxels (under 150k, so there is no block averaging).
      - Targets use the per-section context reset.
  12. **statsmodels** is a runtime dependency of the crossed LMM.
- **2026-10-04, after the G0 first pass and the G0(g) audit, before any v4 development score.**
  13. **G0(g) auditor substitution.** Delegation to the registered external auditor (GPT-6.1-Sol) was unavailable
      in this session: the orchestrator rejected the request. An independent multi-agent adversarial audit was run
      instead, with five lenses: leakage and provenance, protocol correctness, §12 reproduction, scale, and bugs.
      Each serious finding went to two skeptics who tried to refute it.
      - **Verdict:** no leakage. Four findings were confirmed (memory at dense ALL, story-level provenance logging,
        the ledger, G0(a) coverage), plus 12 minor items. All were fixed.
      - **Follow-up:** a Sol audit will be added before the lock if delegation becomes available.
  14. **G0(c) tolerance.** The production ISP module reproduces the pilot's Spearman to 8.5e−5, not 1e−6. Two
      steps traced the cause:
      - **Not the loader.** The pilot's own loader gives the identical value, so the constant-voxel fix is not the
        cause.
      - **The PCA step.** The pilot used an uncentered float32 SVD with scores u·s; the module uses a centered basis
        with float64 projection. G0(c) is therefore checked at 1e−6 with the pilot's PCA fed through the module
        pipeline (loader, lags, design, ridge, grouping, scoring). The production PCA difference is reported.
  15. **G0 coverage and run order.**
      - G0(a) covers UTS01–UTS03.
      - `results/v4/g0.json` records the code fingerprint.
      - The runner refuses any development run other than the G0(a) cells until G0 has passed under the same code.
  16. **Evaluate stage.** It reuses the develop selections (γ, α, CV curves), which are identical under the locked
      code, and recomputes only the final fit. Its provenance log records the reused inner-CV reads.
- **2026-10-04, phase-2 build decisions, before any LPP prior-arm result and before the lock.** At this point no
  LPP decoding score existed except one W0-only engineering smoke cell (sub-EN061, section 1, L8), which was
  kept outside the registered paths.
  17. **Reference pool.** "All other cohort listeners" means the other listeners of the partition (DEV: the DEV
      listeners; CONFIRM: all EN listeners) whose preparation record has exactly the registered section TR
      counts (`summarize_v4.registered_pool`). A listener with no such record has no prepared data and is
      listed with the reason in every cell's `reference_pool`.
  18. **LPP fitting rows (`run_v4_lpp.FIT_ROWS`).**
      - Label-free fits (PCA bases, ISP, SELF, TH_SEL/XDEC converters and selection material) use each section
        without its picture TRs, z-scored over the kept rows.
      - Their half-section groups split the kept rows at the middle (section 1: TR 161 of 282). The own-encoding,
        EB-null and ORACLE groups of the delay bins split at n_tr // 2 (TR 141).
      - XDEC's test-section input is z-scored over its kept rows.
  19. **Controls (FUTURE_WORK_V4 §9) in LPP**, for W0, W1, PRIOR and EB at the primary cell:
      - **Shift null:** within-section circular-shift null (≥ 40 TR, 2,000 draws, seed 0) per cell, and
        subject-level over the 9 sections.
      - **Wrong section:** each section's targets against the next section's prediction, cyclic, using the
        run_v3 wrong-run rule.
      - **Nuisance floors:** position, speech rate, and both.
      - **Rotation:** the full pipeline refit with every budget section's training targets rotated by half its
        rows.
  20. **Comparators.**
      - **TS labels in ds003020:** a reference's TS labels are all of its stories that the target never heard,
        in any of the reference's own splits. This includes UTS01–03 validation/test stories that targets
        UTS04–UTS10 never heard. It does not affect R5, which concerns the target's own validation/test
        material.
      - **Converter α:** chosen in kernel space, equal to `encoding_scores` to 1e−10; float64 only. The optional
        mixed-precision mode is not used.
      - **Two-reference sensitivity:**
        - ds003020: all 3 pairs of UTS01–03.
        - LPP: the first two references in sha256(`ssltang-v4-thsel-pair:<target>:<reference>`) order.
        - These cells are descriptive (layout `sensitivity/refs-<R1>-<R2>/cells/`).
      - **Not-estimable cells:** a comparator that is not estimable writes a not-estimable cell. The primary
        analysis reads it as "arm not estimable" (a recorded deviation), not as a missing file.
  21. **P-AC in LPP:** the LPP TextGrids have no phone tier, so phoneme rate is dropped (F = 36). P-AC is a
      control outside H1–H6.
  22. **K-subsets (§8.4, Tier 2).**
      - Levels k ∈ {1, 2, 4, 8, 16}, with 5 draws each.
      - References are the first k in ascending sha256(`ssltang-v4-lpp-refs:<subject>:<draw>:<reference>`)
        order.
      - R_P(K) is fitted on DEV for K ≤ 8, and its K = 48 prediction is locked.
  23. **Primary-analysis guards.**
      - A hypothesis with fewer than 2 analyzable subjects is reported as **NOT TESTABLE**, which stops the
        fixed sequence like a failure.
      - Any 'missing file' exclusion needs `--accept-missing REASON`.
      - Systematic omissions or never-run subjects need a recorded deviation.
      - All three are recorded in `primary.json`.
  24. **CONFIRM scope (registered, outcome-independent, fixed by cost).**
      - **Confirmatory cell set**, on the dense clock and raw plus residualized endpoints, for each of the
        40 CONFIRM listeners × 9 outer sections:
        - at L3: W0, W1, EB, EB_f{0.75, 0.5, 0.25, 0}, EB_SELF, TH_SEL, plus ORACLE, PRIOR, EB_MISALIGN,
          TH_SEL_rand and XDEC as reported secondaries;
        - at L8: W0, W1 and EB (H6);
        - the §9 controls of item 19.
        - These cover H1–H6 and the reported quantities.
      - **Descriptive only, run after the confirmatory set as GPU time allows:** L5, the stride-4 clock, the
        K-subsets, the two-reference pairs, and TH_SEL outside the primary cell.
      - **Reason:** the full default plan is estimated at about 450 GPU-hours for TH_SEL alone. The confirmatory
        set is about 40–60 GPU-hours.
      - **If PRIOR is advanced (substitution rule):** PRIOR and its ladder PRIOR_f* replace EB and EB_f* in this
        set, and H6 is dropped.
  25. **LPP DEV re-preparation.** The DEV data were re-prepared with the final `lpp_prepare.py`, which adds
      content digests and the code fingerprint. This happened before G4 and before any LPP decoding run; the
      earlier preparation predated those records.
