# Protocol v3 report: structured voxel ridge and dense training

**Date:** 2026-10-03.
**Plan:** FUTURE_WORK_V3.md.
**Protocol, gates and deviations:** [experiments/PROTOCOL_V3.md](../../experiments/PROTOCOL_V3.md).
**Lock:** [experiments/v3_lock.json](../../experiments/v3_lock.json).
**All tables:** [RESULTS.md](RESULTS.md).

All fits ran on the remote RTX PRO 6000. The endpoint is the v2 session-macro feature correlation on held-out
test stories. UTS01–03 test stories were scored in earlier studies, so every v3 test number is an **exposed
benchmark**, not a confirmatory test. Within each subject, each arm is one deterministic fit.

## Main findings

1. **Structured shrinkage gives a small, consistent gain over all-voxel ridge.**
   - The locked V4 decoder fits separate weights for coarse 4×4×4-bin signal, within-bin voxel detail, and the
     three response delays. It reached test r **0.137 / 0.153 / 0.161** (UTS01/02/03), against **0.134 / 0.148 /
     0.156** for matched all-voxel ridge V0.
   - The gain was +0.0035 / +0.0046 / +0.0048 (mean **+0.0043**, about 3% relative). V4 was ahead in 6/11,
     9/11 and 10/11 stories.
   - It was also positive in every subject at 25% (mean +0.0059) and 10% (+0.0044) of labels.
   - It did **not** hold at 5% (−0.0023; at the smallest alpha the CV hit the grid edge in two subjects) or at
     1% (+0.0012, noise).
   - V4 was chosen by pre-registered validation gates before any test score.
2. **The mechanism is spatial-scale and delay regularization, not output-wise shrinkage.**
   - At 100%, every subject selected detail weight **η = 0.1**: voxel detail inside a bin is kept but shrunk 10×
     more than the bin means.
   - Two subjects up-weighted the 4-s delay (τ = 1,2,1); UTS02 dropped the 6-s delay (τ = 1,1,0).
   - On test, the spatial-only arm (+0.0025) and the delay-only arm (+0.0022) roughly add up to the joint gain
     (+0.0043).
   - A separate α per GPT-2 output (V1) was worse than one shared α in every subject at ≥ 5% labels (−0.0015 at
     100%, −0.0056 at 10%).
3. **Dense training is the largest lever found so far: about +30%.**
   - Using targets at every TR instead of every fourth raised test r to **0.175 / 0.195 / 0.197** for V0 and
     **0.179 / 0.200 / 0.200** for V4.
   - That is +0.041 to +0.048 in every subject, with within-story retrieval up from ≈0.80 to ≈0.84. V4's edge
     persisted (+0.0042 / +0.0048 / +0.0032).
   - Dense training is a changed data budget, not a same-label gain. It consumes only 5% more transcript words
     (102,603 vs 97,425), because each stride-4 target already embeds 20 words of context. But it makes every
     word an in-bin target word (4× the brain–target pairs). The archived stride-4 clock was discarding three
     quarters of the usable pairs.
4. **The brain signal is real, but the endpoint is partly sensitive to speech rate.**
   - Every brain control fails as it should. Rotating training targets within runs gives r ≈ 0.00. Predicting
     from the wrong test run gives ≈ −0.006. The circular-shift null reaches q95 ≈ 0.007.
   - However, an **oracle speech-rate model** (word counts and durations from transcripts, no brain data)
     predicts the GPT-2 targets at r = **0.094**. Elapsed position alone gives 0.016.
   - After removing that training-fit nuisance prediction from the targets, the brain decoders still reach r
     0.127 / 0.143 / 0.153 (V4), and V4 still leads V0.
   - So most decodable information lies beyond these nuisances, but future reports should include residualized
     scores.
5. **All three findings replicated on fresh recordings, registered before download.**
   - UTS09 and UTS10 had never been used in this project. The locked procedure ran unchanged on them.
   - Dense training gained +0.032 / +0.035. V4 − V0 was +0.0077 / +0.0045 (10/11 stories in each subject). V4
     − V0 under the dense clock was +0.0083 / +0.0038.
   - Every registered contrast had the predicted sign in both subjects. Detail weight η = 0.1 was selected in
     all five subjects, and delay weights τ = (1, 2, 1) in four of five.
   - Story texts overlap with UTS01–03, so this tests new neural data rather than unseen text.
6. **Supervision exposure was counted.**
   - "100% labels" (4,518 targets) consumes 97,425 transcript word instances (26,369 s), because of the 20-word
     context. The 794 validation targets add 17,343 words of development exposure.
   - Budgets below 100% use different story blocks per subject: the inherited v2 subset hash includes the
     subject's run IDs. Arms within a subject share their subset.

## Registered fresh-recording replication (PROTOCOL_V3 §11)

The registration was committed (`ee0f49e`) before the UTS09/UTS10 download. The runner was committed
(`98a1399`) before any of their outcomes. Each subject used its own whole-session split: six training sessions,
2,418 training targets, about half of UTS01–03's. The locked code and grid were unchanged. The §4 fold rule was
applied to each subject's own sessions.

| Contrast (test r) | UTS09 | UTS10 | Replicated |
|---|---:|---:|---|
| R1: dense-clock V0 − stride-4 V0 | +0.0315 | +0.0354 | yes |
| R2: V4 − V0, stride-4, 100% | +0.0077 (10/11 stories, sign-flip p = .003) | +0.0045 (10/11, p = .037) | yes |
| R3: V4 − V0, dense clock | +0.0083 | +0.0038 | yes |

- **Absolute test r:**
  - UTS09: V0 0.078, V4 0.086, dense V4 0.118.
  - UTS10: V0 0.084, V4 0.088, dense V4 0.123.
- **Controls:**
  - Disrupted training and wrong-run inputs give |r| ≤ 0.0042, and the shift-null 95th percentile is about 0.006.
  - The oracle speech-rate model reaches r = 0.0895 on these test stories, slightly above the stride-4 brain
    decoders' raw scores (0.078–0.088).
  - Residualized, V4 still beats V0: 0.076 vs 0.070, and 0.080 vs 0.076.
- **Interpretation:** a large share of the raw stride-4 endpoint is predictable from speech rate in this split.
  The brain-specific part, and the V4 and dense-clock gains, survive its removal.

## Exploratory modality transfer with the dense decoders (FUTURE_WORK_V3 §7.1)

A versioned voxel-kernel transfer path ([run_v3_transfer.py](../../scripts/run_v3_transfer.py)) applied the saved
dense V0 and V4 decoders to ds004510, with no re-selection.

**Consistency gate.** Before any score, the same path was applied to the ds003020 validation stories. It
reproduced the standard validation predictions to about 1e−15 relative RMS, and the saved validation r exactly.

**Imagined speech** (5-way, chance 20%), centered narrative-macro top-1 with exact permutation p:

| Subject | v2 pooled decoder | Dense V0 | Dense V4 |
|---|---:|---:|---:|
| UTS01 | 60% (.017) | 70% (.008) | 60% (.033) |
| UTS02 | 50% (.067) | 70% (.033) | 60% (.033) |
| UTS03 | 60% (.033) | 70% (.017) | 80% (.008) |

- Both dense decoders reach p ≤ .05 in all three subjects, and MRR rises to 0.75–0.88.
- Films stay at chance (33%).
- These transfer recordings were scored in v2, where the registered rule failed in UTS02. The dense decoders
  were built afterwards. This re-test is therefore **exploratory**: it is consistent with a real imagined-speech
  transfer signal, but cannot confirm one. Confirmation needs imagined-speech recordings that have not been
  scored.

## Gates (FUTURE_WORK_V3 §6.4), fixed before any score

| Stage | Outcome |
|---|---|
| Numerical recovery | Passed on the precision-aware criterion. The kernel path matches the original solver to 1e−8. Phase V's float16-stored predictions are reproduced to within half an ulp. The originally written tolerance was mis-specified and is recorded as a deviation. |
| Development, 100% | V4 − V0 (validation) +0.0071 / +0.0083 / +0.0036: mean +0.0063, above the 0.005 threshold and positive in every subject → advance |
| Development, 10% | +0.0026 / +0.0044 / +0.0030 → no drop → pass |
| Mechanism | V2 (+0.0039) and V3 (+0.0017) fail G1 alone; V1 is negative → V4 advanced; output-wise shrinkage does not explain the gain |
| Locked evaluation | Every subject, budget, arm and control reported. The test gain (+0.0043) is smaller than in development (+0.0063), as expected after selection on validation |

## Process notes

- **Audit.** An independent audit (GPT-6.1-Sol) confirmed no leakage and correct per-fold normalization,
  out-of-fold coverage and kernel algebra. It found enforcement gaps: a weak lock check, unverified cache and
  membership provenance, missing controls, under-saved evidence and edge-case estimability. All were fixed before
  the lock.
- **Rerun.** Development was rerun under the final code and reproduced all 30 cells exactly.
- **Measured cost.** Grouped CV for 28 kernels takes about 17 s per fold (N = 4,518). A dense CV fold takes about
  2 min at a 39–47 GB GPU peak (N ≈ 18,000).
- **Figures:** [arms by budget](figures/v3_arms_by_budget.png), [V4 − V0](figures/v3_v4_minus_v0.png),
  [controls](figures/v3_controls.png), [dense clock](figures/v3_dense.png),
  [five-subject replication](figures/v3_replication.png).

## Interpretation and limits

- Shrinking fine voxel detail more than bin means, and weighting the mid delay, improves an already strong linear
  decoder by a small amount. Separate penalties for feature groups are established methods (banded ridge), so
  this is evidence about these data, not a new algorithm.
- The much larger dense-training gain shows that data use, not model structure, was the main bottleneck of the
  earlier designs.
- The UTS01–03 results are exposed-benchmark comparisons. They are backed by a registered replication in two
  fresh subjects, which shared one split and one stimulus set. That replication holds the direction of all
  three effects. It is not population inference: five subjects, few test sessions each, and overlapping story
  texts.
- There is no label-savings claim, because no noninferiority margin was pre-registered.

## Recommended next steps

1. Re-baseline future work on **dense-clock V4**. It has now replicated in fresh recordings. Keep reporting
   dense V0 and the residualized (speech-rate-removed) endpoint.
2. Confirm imagined-speech transfer on unscored recordings. The exploratory dense-decoder re-test reached 60–80%
   top-1 (p ≤ .05 in all three subjects); ds004510's runs are now all exposed.
3. UTS04–UTS08, about 26 stories each, remain unused and are available for future confirmatory tests. Unseen
   stimulus text would need an external set such as ThinkAloud (ds006067) (FUTURE_WORK_V3 §7.2).
4. Then take up §8: semantic voxel weighting or a reduced-rank readout, a residual neural branch over the linear
   prediction, and controlled target-model comparisons. Evaluate all of them against dense V4 with residualized
   endpoints.

## Reproduction (inside the container, `/workspace/SSLTang`)

```bash
S=UTS01  # also UTS02, UTS03
python scripts/run_v3.py --stage cache    --set data.manifest=data/semantic/$S/semantic_manifest.json
python scripts/run_v3.py --stage recover  --set data.manifest=data/semantic/UTS01/semantic_manifest.json
python scripts/v3_recovery_diagnostic.py
python scripts/run_v3.py --stage develop  --set data.manifest=data/semantic/$S/semantic_manifest.json
python scripts/v3_gates.py                 # then: run_v3.py --stage lock --advanced V4 --gates '<json>'
python scripts/run_v3.py --stage evaluate --set data.manifest=data/semantic/$S/semantic_manifest.json
python scripts/run_v3.py --stage controls --set data.manifest=data/semantic/$S/semantic_manifest.json
python scripts/run_v3_dense.py            --set data.manifest=data/semantic/$S/semantic_manifest.json
python scripts/v3_exposure.py --config configs/v2/phase_a.yaml --set data.manifest=data/semantic/$S/semantic_manifest.json \
  --out results/v3/$S/exposure.json
python scripts/summarize_v3.py && python scripts/plot_v3.py    # locally, after pulling results
```
