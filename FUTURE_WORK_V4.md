# Future work v4: what other listeners' brains are worth to a few-label semantic decoder

**Origin:** synthesized on 2026-10-04 by a research workflow (four literature sweeps, adversarial novelty checks of the
closest prior work, a dataset sweep, three independent designs and a judge). Evidence, verdicts and design scores are in
[experiments/v4_literature.json](experiments/v4_literature.json). The registration that freezes this plan is
[experiments/PROTOCOL_V4.md](experiments/PROTOCOL_V4.md).

**Decisions on the judge's open questions (2026-10-04, before any v4 decoding score):**
1. **Primary clock: dense.** Stride-4 is secondary.
2. **Budget currency: target scan material (regime S).** Regime L is secondary.
3. **LPP mask size, run-to-section mapping, picture intervals and motion exclusions** are checked on the 9 LPP DEV
   subjects before the lock, under the pre-specified W0-only adaptation rules (G4).
4. **Reference PCA bases** may use reference BOLD outside the target's budget (rule R3: reference-only,
   transcript-free, never the target's validation or test material). A budget-restricted basis is a sensitivity
   analysis.
5. **SSL-prior arm** is run only if the v1/v2 reconstruction checkpoints load. Otherwise it is dropped (it is
   exploratory; EB-SELF and tSNR remain).
6. **TH-SEL and XDEC** use public Huth-lab code where available, and otherwise faithful re-implementations from the
   papers' methods. They use the published parameters (reference top-10k, goal top-15k, a two-reference
   sensitivity analysis).
7. **Publication route:** a standard article with the git lock as the timestamped pre-registration. Posting the
   protocol to OSF or submitting a Registered Report Stage 1 is left to the author, because it is an external action.
8. **GIN story_listening fallback** stays as stated (exposed stories, downgraded claim).
9. **Power:** if the development d_z is well below 0.45, the outcome will still be reported under the SESOI tiers. n
   stays at 40.

**Status:** this is a proposal. The only v4 data seen so far is the training-only ISP map from the UTS01 feasibility pilot (`results/v4_pilot/UTS01_isp_summary.json`):
- Spearman 0.539 against the full-label encoding map;
- 59% overlap in the top 2% of voxels;
- the ISP ridge alpha landed on the grid edge at 1e6.

No v4 decoding score exists.

**Builds on:** PROTOCOL_V3 §12 and `results/v3w/REPORT.md`, which give the W1 encoding-predictability weighting. Its key numbers:
- Gain at full labels: +0.055 to +0.064 test r.
- Gain by label count:
  - ≤ 242 labeled pairs: about 0;
  - 452-605 pairs: +0.008 to +0.033;
  - above about 1,100 pairs: a plateau of +0.04 to +0.06.
- Dense training adds about 30%.

**Design lineage:** design A is the backbone (scan-matched EB and an intersection-union primary). From design B it takes the LPP DEV/CONFIRM split, the dense W1 baseline, the provenance auditor, and the misaligned and low-level controls. From design C it takes the noise ladder, the reliability model and the reference-count scaling.

---

## 1. Motivation and positioning

The v3 §12 result has a sharp limitation. Weighting columns by out-of-fold encoding predictability needs about 500 or more of the target subject's labeled (BOLD, GPT-2) pairs. Below about 250 pairs it gives nothing, because the target's own encoding scores are noise. The obvious remedy is to borrow the missing information from other people's brains.

That idea is **not new**, and the plan is framed accordingly:

- **Tang & Huth 2025, Current Biology.** Same dataset family (ds003020). They select a goal participant's 15k semantic voxels without goal linguistic labels, by reverse-converter agreement between two references, and they transfer decoders through converters. Cross-participant decoders beat within-participant decoders below about 2 h of goal story data.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC11903136/
  - https://doi.org/10.1016/j.cub.2025.01.024
- **Tang & Huth 2026, Imaging Neuroscience.** Transfers whole encoding models via converters fit on 12-24 min of alignment data. It **compares** cross- and within-participant models and never **combines** them.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC13543446/
- **Tang et al. 2026, aphasia bioRxiv.** Selects the top 10k voxels by the out-of-sample r of predicting goal voxels from the references' encoding-selected voxels. This is essentially our "semantic ISP".
  - https://www.biorxiv.org/content/10.64898/2026.04.07.717076v1.full
- **Wen et al. 2018, NeuroImage.** A Bayesian prior centered on other subjects' aligned encoding weights, with strength tuned by CV (vision encoding).
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC5976558
- **Shrinkage of individual estimates toward group priors, scaled by data quantity:**
  - Mejia 2018, functional connectivity: https://pmc.ncbi.nlm.nih.gov/articles/PMC5957759
  - Zhi 2025, parcellation: https://pmc.ncbi.nlm.nih.gov/articles/PMC12319954
  - Benson & Winawer 2018, retinotopy: https://pmc.ncbi.nlm.nih.gov/articles/PMC6340702
- **Co-data-adaptive penalties:**
  - GRridge: https://arxiv.org/abs/1411.3496
  - fwelnet: https://arxiv.org/abs/2006.01395
- **Tikhonov and banded ridge:**
  - https://gallantlab.org/publications/2019-voxelwise-encoding-models-non/
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC9807218/
- **Encoding-based voxel selection:**
  - Kay 2008: https://ideas.repec.org/a/nat/nature/v452y2008i7185d10.1038_nature06713.html
  - Tang 2023: https://www.nature.com/articles/s41593-023-01304-9
- **ISC masks and inter-subject ceilings:**
  - Vodrahalli 2018: https://arxiv.org/abs/1610.03914
  - Schrimpf 2021: https://pmc.ncbi.nlm.nih.gov/articles/8694052/
  - ISC tutorial: https://hassonlab.princeton.edu/publications/measuring-shared-responses-across-subjects-using-intersubject-correlation
- **Hyperalignment and individualized topography transfer:**
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC6048235
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC10666932
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC11501089
- **Ensemble-of-other-subjects encoders (Gu 2022):** https://pmc.ncbi.nlm.nih.gov/articles/PMC9759560
- **Functional alignment beats single-subject decoders below about 100 min:**
  - Thual: https://arxiv.org/abs/2312.06467
  - MindEye2: https://proceedings.mlr.press/v235/scotti24a.html
- **Jalouzot et al. 2025.** On the LeBel data, naive multi-subject decoder training did not help: https://arxiv.org/abs/2505.21304
- **Banville 2025.** Decoding scales mainly with data per subject: https://arxiv.org/abs/2501.15322

**What is left, and what this study claims** (to our knowledge; the literature sweeps ran out of search quota, so step 15 re-searches before submission):

1. **A label-count-adaptive empirical-Bayes (EB) combination** of the target's own noisy nested out-of-fold encoding scores with a transcript-free inter-subject-predictability (ISP) prior.
   - The weights are used as soft per-(voxel, delay) ridge penalties inside a within-subject decoder of continuous LM features.
   - The shrinkage factor comes from a null variance measured at the actual budget. The estimator therefore hands over from prior to own labels without tuning per budget.
   - This applies known EB logic (Mejia, GRridge, Wen) to a new object, decoder column weights. Its value is empirical, not methodological invention.
2. **A scan-time-matched, nested, label-accounted measurement of what an inter-subject voxel prior is worth to decoding.** Every arm sees the same target scan minutes. The comparisons include own-label weighting, an over-budget oracle, the Tang & Huth 2025 selection rule (whose decoding value they never ablated), and converter-based decoder transfer. Controls separate:
   - stimulus-locked shared signal from generic signal quality (other-listener vs own-brain priors, permuted and misaligned priors);
   - semantic content from low-level content (residualized endpoint, acoustic prior);
   - and test causally whether prior quality matters (a noise ladder with exactly known reliability).
3. **Pre-registered confirmation in 40 never-analyzed Le Petit Prince listeners** (different site, multi-echo, MNI space), with subject-level population inference. The v3 five-subject sign tests could not provide this.

**Explicitly not claimed:**
- label-free or transcript-free selection of semantic voxels from other listeners;
- ISP or semantic ISP as ideas;
- cross-participant transfer;
- Bayesian shrinkage toward other subjects;
- the reliability algebra in §8, which is standard best-linear-predictor algebra;
- "first many-listener test".

Using reference brain PCs without any reference labels is a minor implementation difference from Tang & Huth, not a contribution.

A clean null is publishable: "at matched scan time, other listeners add nothing to a within-subject decoder beyond the target's own labels". It would complement Jalouzot et al. at the level of voxel priors.

## 2. Research questions

- **RQ1.** At about 30 min of target scan time, all of it labeled, does EB weighting beat both own-label weighting (W1) and uniform ridge (W0)?
- **RQ2.** Is the gain semantic and stimulus-locked? Specifically:
  - does it survive speech-rate and position residualization;
  - does it scale with prior reliability (noise ladder);
  - does it exceed a within-subject self-predictability prior;
  - does it vanish with misaligned or permuted priors?
- **RQ3.** Inside the same decoder, does the soft EB prior beat the published hard selection rule? Where does zero-label converter decoding cross within-subject EB?
- **RQ4.** Does EB cost anything at full budget, and how does its shrinkage schedule (κ against pairs) behave across subjects, clocks and datasets?
- **RQ5 (descriptive).** How does prior quality scale with the number of reference listeners and with extra unlabeled shared-stimulus scan time?

## 3. Budget currency and label accounting

**Currency: target scan material (whole stories or sections).** In naturalistic listening every scanned story comes with a transcript, so the scarce resource is scan time. All arms read the same target material B.

Within B, **every** target-derived quantity is computed from B only, and re-computed inside each decoder-CV fold from that fold's training part of B:
- column standardization;
- own encoding scores;
- ISP and SELF;
- converters;
- EB hyperparameters.

This fixes design A's leak of ISP into inner validation.

**Rules** (enforced by the provenance auditor, §10 step 3):

- **R1.** A target transcript is paired with anything only if it lies in B.
- **R2.** Target BOLD outside B is never read in the primary regime (S). Target validation and test BOLD are used only as decoder input at scoring.
- **R3.** Reference BOLD may be read at the target's in-budget stimulus times (ISP, converters). For PCA bases, reference BOLD may be read on any material that is not the target's validation or test material; this is reference-only and transcript-free.
- **R4.** Reference transcripts are used only by the secondary arms (TH-SEL, XDEC, SISP, XENC, WEN, POP-anat), and only under one of two rules:
  - **TS (true-strict):** stories the target never heard. Available in ds003020 for targets UTS04-UTS10 with references UTS01-UTS03.
  - **SS (simulated-strict, flagged):** any material except the target's validation or test material. Out-of-budget text is reported as reference-only exposure; in-budget text adds no new transcript exposure.
- **R5.** Nothing from a target validation or test story or section enters any prior, converter, reference model, calibration or selection. In ds003020, reference models are refit per target to exclude that target's validation and test stories.
- **R6.** In LPP, GPT-2 context is reset at each section start, so no target feature ever conditions on another section's text. This mirrors the per-story context in ds003020.

**Exception, labeled as such:** ORACLE (§5) exceeds the budget by design and is never in a confirmatory contrast.

**Per-cell ledger**, stored in every cell JSON and extending the v3 exposure counter:
- target labeled pairs, on both the dense and stride-4 clocks;
- target BOLD minutes;
- unique transcript word instances, including the 20-word causal contexts, and GPT-2 tokens;
- number of references and reference BOLD minutes;
- reference labeled pairs, with the TS or SS rule.

**Secondary regime L** (not scan-matched, labeled as such): ISP may also use the target's unlabeled BOLD on all non-test shared material, while labels stay restricted to B. This quantifies the value of extra unlabeled shared-stimulus scans.

## 4. Datasets and splits

### 4.1 Development and exposed benchmark: ds003020 (UTS01-UTS10)

Archived v2/v3 splits are used unchanged. The 24 shared stories split train/validation/test as:
- UTS01-08: 15/5/4;
- UTS09/10: 13/11/0.

**Budgets** are nested sets of whole training stories:
- S3 and S6: the first 3 and 6 of the target's training-split shared stories, in `sha256("ssltang-v4-budget:<story>")` order;
- S15: all training shared stories (13 for UTS09/10);
- ALL: every training story.

UTS04-08 share the same training shared stories, so their S3 and S6 sets are identical across subjects. Approximate sizes:

| Budget | Dense pairs | Stride-4 pairs | Minutes |
|---|---|---|---|
| S3 | about 1,000 | about 250 | about 33 |
| S6 | about 2,000 | about 500 | |
| S15 | about 5,000 | about 1,250 | |

**Inner decoder CV:** leave-one-story-out at S3; 3 folds at S6; 5 story folds at S15; the v3 §4 or §11 fold rules at ALL. Nested encoding scores use story-half groups, so at least 4 groups exist at S3.

**Development** uses validation stories only. The 9 development subjects are UTS01-03 and UTS05-10. UTS04 is excluded from all gates because its validation session 6 is documented as unusable; it stays in test reporting.

**Test stories have all been scored before** (§12), so the ds003020 test is an **exposed benchmark, reported descriptively**. It also hosts the faithful deep-to-shallow published-method block: targets UTS04-UTS10, references UTS01-03, TS rule.

### 4.2 Confirmatory: Le Petit Prince English (OpenNeuro ds003643 v2.0.7, CC0)

Sources:
- https://openneuro.org/datasets/ds003643
- descriptor: https://pmc.ncbi.nlm.nih.gov/articles/PMC9424229/

**Cohort and data:**
- 49 EN listeners (EN057-EN115). Each has 9 sections with identical TR counts (282/298/340/303/265/343/325/292/368 = 2,816 TRs, 93.9 min).
- TR 2 s. ME-ICA-preprocessed data on a 2 mm MNI-Colin27 grid (73x90x74).
- Word timings from `annotation/EN/lppEN_word_information.csv`.
- Sections are mapped by sorted run number and verified against the TR counts.
- The cohort was used group-averaged by Bonnasse-Gahot & Pallier (https://arxiv.org/abs/2405.17992). This project has never analyzed it.

**Partition**, fixed now: sort subjects by `sha256("ssltang-v4-lpp-subject:"+id)`.
- The first 9 are **DEV**. Only these are downloaded before the lock.
- The remaining 40 are **CONFIRM**, downloaded only after the lock.

**Outer CV:** 9-fold leave-one-section-out (LOSO) for every subject, so each section is test exactly once.

**Budgets within each outer fold:** nested prefixes of the 8 non-test sections in `sha256("ssltang-v4-lpp-section:<testsec>:<sec>")` order, the same for all subjects.

| Budget | Dense pairs | Stride-4 pairs | Minutes | Role |
|---|---|---|---|---|
| S3 | about 900 | about 225 | about 31 | **primary** |
| S5 | | | | secondary |
| S8 | about 2,450 | about 610 | about 83 | secondary |

- **Inner decoder CV:** leave-one-section-out within B.
- **Nested encoding:** section-half groups.

**References:**
- DEV targets use only the other 8 DEV subjects.
- CONFIRM targets use all 48 other EN listeners, brain only. This is registered as "all other cohort listeners".
- No one's test-section BOLD is ever read for priors.

**Mask:** an MNI gray-matter mask resampled to the grid, intersected with voxels that are finite and non-constant in all DEV subjects. It is fixed in development from anatomy and signal presence only. Per subject, zero-variance columns are dropped. If the mask exceeds 150k voxels, 2x2x2 block averaging is used (a locked rule).

**Exclusions:** bins overlapping 0-80 s of section 1 (the pictures) are excluded from training and scoring.

**Targets:** the v2 FEATURE_CONTRACT definition (pinned GPT-2, layer 9, 8-s bins), with training on the dense or stride-4 clock and evaluation on stride-4 bins at t+1..t+3 TR.

### 4.3 Not used

- **Narratives:** too little data per subject. Optional only: the PNI group as an exploratory very-low-budget check (https://pmc.ncbi.nlm.nih.gov/articles/PMC8479122).
- **Berkeley GIN story_listening:** the same stories as ds003020, with possible participant overlap. It is the pre-registered **fallback** only if LPP fails G4, with a downgraded claim.
- **SMN4Lang:** Mandarin.
- **ds005717:** subjects already used. Exploratory silent-movie ISP only.

## 5. Arms

**Common decoder** for all within-subject arms (the locked §12 machinery):
- all-voxel linear kernel ridge on standardized (voxel, delay) columns at t+1..t+3 TR;
- columns scaled by weights w, rescaled to mean w² = 1;
- alpha from logspace(-1, 7, 17);
- gamma ∈ {0.5, 1, 2} for every soft rule;
- both chosen by the inner grouped CV over B, with the §12 tie rules, the 1% fallback, and the not-estimable rule (fewer than 2 encoding folds);
- **primary clock: dense**, which is the strongest established within-subject baseline at matched scan time; stride-4 is secondary.

**Tier 1** (all datasets, all budgets):

| Arm | Definition | Target data | Other subjects |
|---|---|---|---|
| **W0** | Uniform w = 1 | B labels | none |
| **W1** | Locked §12: w_vd = max(r_vd, 0)^gamma. r is the nested OOF Pearson r of ridge encoding (768-d target → column), computed in each fold from that fold's B rows | B labels | none |
| **PRIOR** | w_vd = max(ISP_v, 0)^gamma, the same at all delays (definition of ISP below) | B BOLD (unlabeled) for the prior; B labels for the decoder | reference BOLD, no transcripts |
| **EB** (proposed) | See the estimator below | B labels plus PRIOR's data | as PRIOR |
| **EB[f] ladder** | EB with p' = p + ε, ε ~ N(0, Var(p)(1/f - 1)), f ∈ {0.75, 0.5, 0.25}; f = 0 is p permuted across voxels. One locked seed per level, so R_P' = f·R_P in second moments | as EB | as EB |
| **EB-SELF** | EB with a within-subject self-predictability prior. The mask is split into octants at the median x, y and z. Each voxel is predicted from the top-100 PCs (lags -1..+1) of the target's own voxels outside its octant, with a 4-voxel margin. Story- or section-grouped OOF ridge over B. This is the linear stand-in for an SSL-predictability prior | B only | none |
| **EB-MISALIGN** | EB with ISP computed after circularly shifting every reference's time series by at least 40 TR within each run (not stimulus-locked) | as EB | as EB |
| **P-AC** | max(r_ac, 0)^gamma, where r_ac is the nested OOF r of a low-level encoding model on B: 32-band mel envelope, RMS, F0, word rate, phoneme rate, word-onset count | B audio and timing | none |
| **ORACLE** (descriptive only) | W1 weights from the target's full non-test training labels; decoder on B | over budget | none |

**Tier 1 published comparators.** Both run in LPP (SS rule) and in ds003020 targets UTS04-10 (TS rule, faithful setting):

- **TH-SEL** (Tang & Huth 2025 functional selection, adapted into our decoder):
  1. For each reference, a reverse ridge converter maps that reference's top-10k encoding voxels (lags -1..+1) to all target voxels. The reference voxels are selected from reference labels under TS/SS. The converter is trained on target and reference BOLD at B times.
  2. The converter is applied to reference BOLD on material the target did not hear: TS stories in ds003020, out-of-budget non-test sections in LPP.
  3. Score per target voxel: the mean pairwise correlation of the references' aligned predictions. The published two-reference version is run as a sensitivity analysis.
  4. Uniform weights on the top 15,000 voxels (ds003020), or the top 17% of mask voxels (LPP, the same fraction), at all delays.
  - Matched control: a random mask of the same size.
- **XDEC** (Tang & Huth 2025 converter decoder transfer, adapted to the feature-regression endpoint):
  1. For each reference, a §12 W1 decoder is trained on the reference's top-10k voxels using TS/SS labels.
  2. A ridge converter maps target BOLD (all voxels, lags -1..+1) to those reference voxels. It is fit on B times, with alpha chosen by grouped OOF r.
  3. Predictions are averaged over references.
  - It uses **zero target labels**. LPP references are shallow, and this adaptation is documented.

**ISP (prior source for PRIOR and EB):**
- **Response:** each target voxel's per-run z-scored BOLD at B times (minus the inner-held-out part when computed inside a fold).
- **Predictors:** each reference's top-100 PCs at lags -1, 0 and +1, concatenated across references. PCA bases come from the reference's own non-test material (R3).
- **Model:** ridge with alpha in logspace(0, 10, 21), which extends past the pilot's edge. Alpha is chosen by grouped OOF mean r; ISP_v is the OOF r with the same groups.
- **Transcripts:** none from anyone.

**EB estimator**, per (target, budget, decoder-CV fold, delay d), from that fold's labeled B rows only:
1. z_vd = atanh(clip(r_vd, ±0.999)) from W1's nested scores.
2. **Null variance.** Recompute step 1 with the 768-d targets circularly shifted within each run by four fixed shifts (k·n_run/5, k = 1..4, each at least 40 TR). Then σ²_d is the mean over shifts of (1.4826·MAD_v(z⁰_vd))². This captures label count and autocorrelation at this budget.
3. p_v = atanh(ISP_v), not standardized.
4. **Calibration.** OLS of z_·d on p across columns, giving (a_d, b_d), constrained to b_d ≥ 0. If the slope is negative, set b = 0 and a = mean z.
5. **Shrinkage.** V_d = Var(z − a − bp); τ²_d = max(V_d − σ²_d, 0.01·V_d); κ_d = τ²_d / (τ²_d + σ²_d).
6. **Posterior.** m_vd = a_d + b_d·p_v + κ_d·(z_vd − a_d − b_d·p_v).
7. **Weights.** w_vd = max(tanh(m_vd), 0)^gamma, rescaled. Fallbacks follow §12 and are recorded.

Identities used in G0:
- κ = 1 gives W1 exactly.
- κ = 0, a = 0, b = 1 gives PRIOR exactly.

**Tier 2** (secondary or exploratory; dropped first if time runs short; none affects the H1-H6 family):

- **Published-method block** (ds003020 UTS04-10 targets, TS rule, deep references UTS01-03):
  - **SISP / EB-SISP:** ISP with predictors restricted to each reference's top-10k TS-encoding voxels.
  - **XENC → EB-XENC:** the Tang & Huth 2026 transferred encoding model (reference encoding weights times reference-to-target converter). Its correlation on the in-budget rows is used as a prior.
  - **WEN:** target encoding weights fit with the penalty a‖β − β_XENC‖² + b‖β‖², with a and b chosen by inner CV, following Wen 2018. Its OOF r is then used through the W1 rule. This is the weight-space alternative to EB.
  - **HYB:** a convex mix of EB and XDEC predictions, λ ∈ {0, .25, .5, .75, 1}, chosen by inner CV.
- **LPP-only:**
  - **POP-anat / EB-POP:** the voxelwise mean of the references' own W1 maps on the shared MNI grid, using SS labels. Anatomical against functional prior.
- **Variants:**
  - **MIX-CV:** κ ∈ {0, .25, .5, .75, 1} chosen by inner CV. Tests whether the closed form matters.
  - **EB-HARD:** uniform weights on the top 2% of columns by m.
  - **TH-SEL-soft:** the TH-SEL score raised to gamma, as soft weights.
  - **tSNR prior.**
- **Regimes, clocks and scaling:**
  - Regime L versions of PRIOR, EB and TH-SEL.
  - The stride-4 clock.
  - Reference-count subsets K ∈ {1, 2, 4, 8, 16, 48} in LPP, with 5 hash-ordered draws each.
- **Exploratory:** an SSL-predictability prior from the project's reconstruction models (UTS01-03, if the checkpoints are intact).

## 6. Pre-registered primary contrast and decision rule

**Cell:** LPP CONFIRM (n = 40), primary regime S, budget **S3** (3 sections, about 31 min, about 900 dense / about 225 stride-4 labeled pairs), dense training clock, raw endpoint.

**Endpoint:** section-macro feature correlation.
- For each outer test section: the Pearson r between predicted and true targets over that section's stride-4 test bins, computed for each of the 768 dimensions, then averaged over dimensions. Picture bins are excluded.
- R_s(arm) is the mean over the 9 outer folds.
- This is the v2 session-macro endpoint (`src/evaluation/endpoints.py`), with sections in place of sessions.

**Contrasts:**
- Δ1_s = R_s(EB) − R_s(W1)
- Δ2_s = R_s(EB) − R_s(W0)

**Test:**
- For each contrast: a one-sided subject-level sign-flip permutation on the mean Δ, 100,000 Monte Carlo flips, seed 20261003, p = (1 + #{T* ≥ T}) / 100,001.
- **H1 is CONFIRMED iff p(Δ1) < 0.025 and p(Δ2) < 0.025, and both means are > 0.** This is an intersection-union test, so no adjustment is needed.
- Otherwise H1 is NOT CONFIRMED and is reported as such.

**Exclusions:** only by endpoint-blind QC, logged before scoring: missing or corrupt files, TR-count mismatch, or W0 not estimable. There is no performance-based exclusion.

**Reported with every contrast:**
- the mean;
- the BCa 95% subject-bootstrap CI (10,000 resamples);
- d_z;
- the number of subjects positive out of 40, with an exact CI;
- the fraction of the oracle gain recovered, (EB − W0)/(ORACLE − W0).

**SESOI = 0.005 r.** Tiers based on the 90% CI: "meaningful", "positive but below SESOI", "equivalent to zero".

**Stimulus generalization:** because everyone hears one book, a pre-registered crossed linear mixed model on per-(subject, section) Δ is run, with random intercepts for subject and section and Satterthwaite df. If H1 passes but the LMM does not, the claim is restricted to "across listeners of this book".

**Key secondaries.** Fixed sequence, each one-sided at α = 0.025, run only if H1 and all predecessors are confirmed, stopping at the first failure:
- **H2 (semantic):** the same intersection-union test on the residualized endpoint. Targets minus a training-fit position term plus an oracle speech-rate nuisance prediction, fit on B as in §6/§12.
- **H3 (prior quality is causal):** the mean over subjects of the per-subject OLS slope of (EB[f] − W1) on f ∈ {0, 0.25, 0.5, 0.75, 1} is > 0.
- **H4 (other listeners, not own signal quality):** EB − EB-SELF > 0.
- **H5 (vs the published rule, like for like):** EB − TH-SEL > 0. TH-SEL has access to reference labels and EB does not.
- **H6 (no regret):** at S8, mean(EB − W1) > −0.005, tested by sign-flip on Δ + 0.005.

**Substitution rule**, fixed now: if development advances PRIOR instead of EB (§7), PRIOR replaces EB in H1-H5, and H6 is dropped.

**Power:** n = 40 gives 80% power at d_z ≈ 0.45 (one-sided α = 0.025). The development d_z and SD(Δ) from ds003020 and LPP DEV are written into the lock. They cannot change n.

## 7. Development gates

All gates are fixed before any v4 decoding score. No hyperparameter is tuned on validation; the following are fixed now:
- 100 PCs and lags -1..+1;
- the ISP and decoder alpha grids;
- the gamma grid;
- the EB estimator;
- the ladder seeds.

**G0, integrity** (must pass before any validation score is read):
- (a) W0 and W1 under the v4 code reproduce the §12 development cells to ≤ 1e-10 relative, with identical alpha and gamma.
- (b) EB with κ = 1 equals W1; EB with κ = 0, a = 0, b = 1 equals PRIOR.
- (c) The UTS01 ISP, recomputed with the pilot settings, reproduces Spearman 0.539 to 1e-6.
- (d) **Synthetic recovery test.** 250k columns, 2% informative, heavy-tailed ρ, prior reliability R_P ∈ {0.1, 0.3, 0.6}, n ∈ {100, 500, 2,000}. EB's MSE in ρ must be ≤ the minimum of the raw-r and prior-only MSE at every n, and the estimated κ must be within 0.05 of the truth.
- (e) The provenance auditor catches all planted leaks:
  - a test-section reference response;
  - an out-of-budget target transcript;
  - a reference model containing a target test story;
  - cross-section GPT-2 context.
- (f) Float32 kernels with float64 accumulation match float64 to ≤ 1e-8 relative.
- (g) An independent code audit (Sol, as in §12) is completed, and its fixes are applied before the development launch. Outputs of any earlier launch are deleted unread and logged.

**G1, prior validity** (ds003020 training data only, no decoding):
- Requirement: Spearman(scan-matched S3 ISP, the full-label W1 encoding map max over delays) has a median ≥ 0.20 over the 9 development subjects and is > 0.10 in at least 7/9.
- If G1 fails at S3 but passes at S6, the **primary budget moves to S6 in ds003020 and S5 in LPP**. This is the only pre-specified budget adaptation.
- If it fails at S6, stop: "ISP is uninformative at matched scan time". Report this, and run Regime L descriptively.

**G-FEAS** (validation, dense, primary budget; W0, W1 and ORACLE only):
- Requirement: mean(ORACLE − W0) ≥ +0.005.
- If this fails, even a perfect prior cannot help at this scan time. Stop and report, and do not build the remaining arms.

**G2, efficacy** (validation, dense, primary budget, 9 subjects): all of the following:
- mean(EB − W1) ≥ +0.005;
- mean(EB − W0) ≥ +0.003;
- EB − W0 > 0 in at least 7/9.

**G3, no harm** (validation, dense, ALL):
- mean(EB − W1) ≥ −0.003;
- no subject below −0.010.

**Advancement:**

| Outcome | Action |
|---|---|
| G2 and G3 pass | Advance EB |
| EB passes G2, fails G3 | Advance EB and drop H6 |
| EB fails G2, PRIOR passes G2 (same thresholds) | Advance PRIOR (substitution rule) |
| Nothing passes | Report the development null and score the ds003020 test descriptively. LPP DEV results are reported descriptively. **CONFIRM stays untouched** and is reserved |

**G4, LPP feasibility** (9 DEV subjects, outer 9-fold, K = 8 DEV references):
- (a) Run lengths and word counts match the header-verified values.
- (b) Dense W0 at S8 exceeds its within-section circular-shift null q95 (at least 40 TR, 2,000 draws) in at least 6/9.
- (c) The ISP alpha is interior to its grid.
- Adaptation: if (b) fails, at most two revisions of the LPP adaptation (mask resolution 2 vs 3 mm, smoothing 0 or 4 mm) are allowed, chosen by **W0 CV score only**, before any prior arm is run on LPP.
- If (b) still fails, switch to the GIN fallback with a downgraded claim.
- LPP DEV efficacy contrasts are computed and reported but **do not gate**. They feed only the d_z estimate and the R_P(K) curve.

**Lock** (`experiments/v4_lock.json`) records:
- the advanced arm and the primary budget;
- code and config SHA256, including the LPP adapter and the hashed primary analysis script;
- all development cell hashes;
- seeds and hash orders;
- the DEV/CONFIRM partition;
- the LPP adaptation choices;
- d_z;
- the locked κ-schedule and reliability-model predictions (§8).

It is committed and pushed to main, and timestamped on OSF, **before** any ds003020 test score and before any CONFIRM download. The evaluate stage refuses to run if any fingerprint changes.

## 8. Secondary analyses

1. **Budget curves for every arm.**
   - ds003020: S3/S6/S15/ALL, exposed test, with a story-level sign-flip per subject and a mean across subjects.
   - LPP CONFIRM: S3/S5/S8.
   - Both clocks, both endpoints, and the fraction of the oracle gain recovered.
2. **Shrinkage schedule.**
   - κ_d, b_d, τ²_d and σ²_d plotted against labeled pairs, pooled over both datasets. This is the empirical "how much to trust your own labels" curve.
   - MIX-CV against the closed form.
3. **Reliability model (from design C)**, presented as a diagnostic built on standard best-linear-predictor algebra. It assumes errors between ISP and the own scores are independent.
   - Inputs, from in-budget data: own-map SNR λ = (Var z − σ²)/σ², and prior reliability R_P = b²Var(p)/(Var z − σ²).
   - Predictions:
     - R_own = λ/(1 + λ);
     - 1/(1 − R_EB) = 1/(1 − R_P) + λ;
     - the own-versus-prior crossover at λ = R_P/(1 − R_P);
     - the label-equivalent value of the prior, Δn*.
   - **Map-level check** (training data only, UTS01-03): L_n is drawn from non-shared stories, and a held-out encoding map comes from the remaining non-shared stories. Predicted and observed reliabilities are reported with a calibration slope.
   - Decoding-level crossover and Δn* are reported against observed values, within a factor of 2 or not.
   - **Error-correlation diagnostic:** shift the target BOLD jointly against both stimulus and references to estimate cov(ISP error, own-score error) under scan-matching.
4. **Reference-count scaling (LPP).**
   - R_P(K) for K ≤ 8 is fitted on DEV with a Spearman-Brown form, and its K = 48 prediction is locked.
   - It is then checked on CONFIRM, along with the EB and PRIOR decoding r across K.
5. **Published-method block** (ds003020 UTS04-10, TS rule, exposed test):
   - TH-SEL, TH-SEL-soft, the random-mask control, XDEC, HYB, SISP/EB-SISP, XENC/EB-XENC and WEN against EB, PRIOR, W1 and W0 at S3/S6/S15;
   - the budget at which EB overtakes zero-label XDEC;
   - whether HYB ≥ max(EB, XDEC).
   - This tests, at the voxel-prior level, the route that Jalouzot et al. found null for parameter pooling.
6. **Semantic specificity.**
   - Residualized minus raw gains for all arms; P-AC against PRIOR; EB-MISALIGN.
   - The partial Spearman correlation of ISP with the own full-label map, controlling for the P-AC map.
   - The share of EB weight mass in auditory cortex vs the language network (pycortex ROIs in ds003020; the LanA atlas in LPP, https://pmc.ncbi.nlm.nih.gov/articles/PMC9424256/).
7. **Anatomical vs functional population prior** (LPP): POP-anat and EB-POP against PRIOR and EB.
8. **Soft vs hard:** EB-HARD (top 2%) and TH-SEL-soft against their counterparts.
9. **Regime L:** the value of extra unlabeled shared-stimulus scans at fixed labels. Explicitly not scan-matched.
10. **Further endpoints:** retrieval rank (±10/±32 TR), normalized MSE and per-section scores.
11. **Exploratory:** the SSL prior (UTS01-03); a silent-movie ISP (ds005717); the Narratives PNI very-low-budget check.

## 9. Controls

- **Brain-input nulls on the locked EB and W1 decoders:**
  - within-section or within-run circular shift (at least 40 TR, 2,000 draws);
  - wrong-section input;
  - the full EB pipeline refit with training targets rotated by half a run, including the prior regression, to check that EB cannot manufacture signal.
- **Prior-identity nulls:**
  - EB[f = 0] (permuted prior), which gives the same weight distribution on the wrong voxels;
  - EB-MISALIGN, which keeps reference structure but is not stimulus-locked.
- **Generic signal quality:** EB-SELF (confirmatory H4), the tSNR prior, and the SSL prior (exploratory).
- **Low-level confounds:**
  - the P-AC prior;
  - the residualized endpoint (confirmatory H2);
  - nuisance-only decoders (position; oracle speech rate; both) as floors in each dataset.
- **Baseline strength:**
  - dense W1 with identical alpha/gamma grids and inner CV;
  - ORACLE as the upper reference;
  - every arm gets the same tuning budget (one gamma grid, or a fixed published k for TH-SEL; one 5-point grid for HYB and MIX-CV).
- **Leakage:**
  - the provenance auditor on every cell, which refuses to write a score on any violation;
  - per-target reference refits that exclude that target's validation and test stories (ds003020);
  - a shared LPP section split;
  - GPT-2 context reset per section;
  - the planted-leak unit tests.
- **Single-book dependence:** the crossed LMM, per-section results, and picture-bin exclusion.

## 10. Implementation (ordered)

1. **Days 1-2.** Write `experiments/PROTOCOL_V4.md` from this plan, in v3 style:
   - §0 exposure: the pilot, and the §12 test sets;
   - rules R1-R6;
   - arms and tiers;
   - budgets and hash orders;
   - the LPP partition and splits;
   - gates and the primary and fixed-sequence tests.
   Commit and push to main before any v4 code runs, and post to OSF.
2. **Days 2-4.** Write `src/budgets.py`:
   - story and section budget objects for regimes S and L;
   - per-target exclusion lists;
   - the TS/SS story filters, derived from the manifests;
   - the exposure ledger.
3. **Days 3-5.** Provenance logging in every loader: (subject, stimulus, time range, kind ∈ {brain, transcript, audio}, purpose). Write `scripts/audit_v4.py`, implementing R1-R6, and planted-leak tests under `tests/`.
4. **Days 3-6.** Write `src/priors/isp.py`, generalizing `scripts/v4_pilot_isp.py`:
   - fold- and budget-aware ISP with the extended alpha grid;
   - cached reference PCA bases;
   - K-subsets;
   - MISALIGN, the ladder, SELF (octant-out), tSNR and P-AC.
   Write `src/priors/eb.py` with the estimator in §5, reusing `src/encoding_weights.encoding_scores` for both own and null scores, plus MIX-CV and the reliability-model quantities.
5. **Days 5-7.** Write `scripts/run_v4.py`, extending `run_v3_weights.py` and `run_v3w_replicate.py`:
   - stages: simulate, develop, lock, evaluate, lpp-prepare, lpp-develop, lpp-confirm;
   - cell JSONs that carry the ledger.
   Add the G0 tests (a-f), including the synthetic recovery test.
6. **Day 8.** Independent audit (Sol). Fix the findings, log the deviations, and rerun G0.
7. **Days 9-10.** ds003020 development on the remote container:
   - G1 (training only);
   - G-FEAS;
   - then the full Tier 1 grid on validation: W0, W1, PRIOR, EB, ladder, SELF, MISALIGN, P-AC and ORACLE, at S3/S6/S15/ALL, dense and stride-4;
   - apply G2/G3;
   - run the reliability-model map-level check.
8. **Days 10-13.** Write `scripts/lpp_prepare.py`:
   - S3 download of **DEV only** (about 6 GB) plus annotations and audio;
   - section mapping by TR count;
   - the gray-matter mask;
   - GPT-2 layer-9 targets via an adapter to `scripts/build_semantic_targets.py`, with the context reset per section;
   - nuisance regressors and the low-level audio features;
   - picture-bin exclusion;
   - float32 caches.
   Then run G4 with W0-only adaptation, then all Tier 1 arms on DEV with K = 8. Fit R_P(K) and estimate d_z.
9. **Days 13-16.** Write `src/priors/tanghuth.py` (TH-SEL, XDEC, XENC, WEN, SISP) with TS/SS assertions. Run them on ds003020 validation as a sanity check (not gating) and on LPP DEV.
10. **Day 16.** Write and hash `scripts/summarize_v4.py` (sign-flip, BCa bootstrap, fixed sequence, crossed LMM). Write `experiments/v4_lock.json`, then commit, push and timestamp on OSF.
11. **Days 17-22.** Download the 40 CONFIRM subjects (about 26 GB). Run all Tier 1 arms at S3/S5/S8 on both clocks and both endpoints, with 48 references, plus the controls. The auditor must pass on every cell before scores are written.
12. **Day 22.** Run the locked primary analysis exactly once, and record the outcome in PROTOCOL_V4 §outcome.
13. **Days 22-26.** Exposed ds003020 test evaluation:
    - Tier 1 arms for all 10 subjects;
    - the published-method block for UTS04-10 (TS);
    - Tier 2 arms in LPP as time allows: POP-anat, HYB, K-scaling, Regime L, EB-HARD.
14. **Days 26-30.** `scripts/plot_v4.py`:
    - budget curves;
    - ladder dose-response;
    - κ schedule;
    - K scaling;
    - controls;
    - the reliability-model prediction against observation.
    Then `results/v4/REPORT.md`. Commit and push every milestone to main.
15. **Weeks 6-7.** Re-run the literature search for 2026 preprints on cross-subject voxel priors. Write the manuscript, positioned against Tang & Huth 2025/2026, Tang et al. 2026 and Wen 2018. Release the code and lock artifacts.

## 11. Compute

One RTX PRO 6000 (96 GB). Float32 kernels with float64 accumulation, validated in G0.

| Item | Size | Estimate |
|---|---|---|
| ds003020 development and exposed test, Tier 1 | 10 subjects x about 9 arm variants x 4 budgets x 2 clocks x 3 gamma. Dominated by dense ALL in UTS01-03, at about 5 min per fold for W0/W1/EB, with EB null shifts at 4x encoding cost | about 25-35 GPU-h |
| Published-method block (UTS04-10) | Reference TS encoders and decoders on about 15k TRs; converters on ≤ 5.7k rows | about 6-10 GPU-h |
| LPP decoders | 49 subjects x 9 outer folds x about 12 arm variants x 3 budgets x 2 clocks x (inner folds + final) x 3 gamma, at N ≤ 2,450 rows and ≤ about 450k columns, about 1-3 s each | about 30-45 GPU-h |
| LPP ISP | 14,400 predictors, ≤ about 150k outputs, per fold, inner fold, budget and ladder; 49 x 9 reference SVDs shared across targets | about 8-12 GPU-h |
| LPP TH-SEL and XDEC | Reference decoders are shared across targets per fold, about 430 | about 8-12 GPU-h |
| Tier 2 | | about 10-15 GPU-h |
| **Total** | | **about 90-130 GPU-h**, about 5-6 days of GPU wall-clock time interleaved with development |

**Disk:** about 32 GB for LPP EN derivatives plus about 80-100 GB of caches, against about 790 GB free.

**Personnel:** about 6 weeks to results plus 1-2 weeks of writing.

## 12. Risks and mitigations

- **Scan-matched ISP is too noisy at about 30 min.** The pilot used 176 min of shared material. Mitigations:
  - LPP has 48 references;
  - G1 is a training-only gate with a single pre-specified move to S6/S5;
  - Regime L quantifies the cost of scan-matching.
- **Dense W1 already works at about 30 min**, leaving little room. Mitigations: G-FEAS and G2 decide before any fresh data are touched; the stride-4 secondary connects to the §12 label currency; a null is pre-registered as publishable.
- **ISP rewards auditory or speech-rate cortex.** Mitigations:
  - H2 (residualized) is inside the confirmatory sequence;
  - P-AC, MISALIGN and the ROI weight-mass analysis;
  - EB's calibration on own semantic scores down-weights ISP-high but semantically empty columns once labels exist.
- **XDEC beats every within-subject arm at S3**, as Tang & Huth's crossover predicts. It is a secondary comparison with no registered direction, it is reported honestly, and HYB tests complementarity. The primary claim concerns weighting a within-subject decoder.
- **Correlated noise between ISP and own scores** under scan-matching inflates b. This affects only the shrinkage's optimality, not test validity. The error-correlation diagnostic is reported.
- **LPP floor effects or pipeline mismatch.** Mitigations: the DEV subjects and G4 with W0-only adaptation; the dense clock; paired contrasts; the GIN fallback.
- **Single continuous book.** Mitigations: paired contrasts, LOSO, context reset, the crossed LMM, and a restricted claim if the LMM fails.
- **Reviewer overlap** (Huth lab), and implementations judged unfaithful. Mitigations: the faithful TS deep-to-shallow block on their dataset; published parameters (reference top-10k, top-15k goal voxels, two-reference sensitivity); stating that our embedding endpoint replaces their beam-search decoder.
- **Forking paths.** Mitigations: everything is fixed now except one pre-specified budget fallback and the W0-only LPP adaptation; a single intersection-union primary; fixed-sequence secondaries; the hashed analysis script; the CONFIRM download only after the lock.
- **Scope creep.** Tier 2 is dropped first, and it does not affect H1-H6.
- **Novelty erosion.** The re-search before submission, plus claims limited to the EB combination, the scan-matched audited benchmark and the fresh-cohort confirmation.

## 13. Target venue

**Imaging Neuroscience** (the venue of Tang & Huth 2026). Preferred as a Registered Report: Stage 1 is this protocol plus the development results, Stage 2 is the confirmatory outcome. Otherwise a standard article with the OSF pre-registration and the git lock.

Fallbacks: NeuroImage or PLOS Computational Biology (if the reliability-model diagnostics are strong), or NeurIPS main track framed as label-count-adaptive empirical-Bayes feature priors with a strict evaluation protocol.
