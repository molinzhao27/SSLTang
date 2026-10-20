# Protocol v3 §12 report: encoding-predictability voxel weighting

**Date:** 2026-10-04.
**Plan:** FUTURE_WORK_V3.md §8.1.
**Protocol, gates and implementation details:** [experiments/PROTOCOL_V3.md §12](../../experiments/PROTOCOL_V3.md).
**Lock:** [experiments/v3w_lock.json](../../experiments/v3w_lock.json).
**All tables:** [RESULTS.md](RESULTS.md).

All fits ran on the remote RTX PRO 6000. The endpoint is the v2 session-macro feature correlation of predicted
GPT-2 layer-9 targets on held-out test stories.
- UTS01–03 and UTS09/UTS10 were scored in earlier studies, so their test numbers are **exposed benchmarks**.
- UTS04–UTS08 had never been used. Their replication was registered before implementation, and the data were
  downloaded only after the §12 lock. That replication is the confirmatory test.

## What was tested

For every voxel *v* and response delay *d* (t+1…t+3), a ridge **encoding** model predicts the standardized voxel
from the 768-d target. Its out-of-fold Pearson *r* over session-grouped folds of labeled training pairs is the
voxel's encoding predictability. The decoder is the v3 all-voxel kernel ridge with each column scaled by a
weight:

| Arm | Column weight |
|---|---|
| W0 | 1 (uniform; identical to v3 V0) |
| W1 | max(*r*, 0)^γ, γ ∈ {0.5, 1, 2} |
| W2 | 1 on the top *q* of columns, *q* ∈ {10%, 25%, 50%} |

- Weights are rescaled so that mean(w²) = 1.
- The scores are **nested**: inside every decoder-CV fold they are recomputed from that fold's labeled rows only.
- Validation and test data never inform a weight.
- The rule (γ or *q*) and α are chosen by the same grouped CV.
- Weighting a column by *w* is equivalent to giving that voxel its own ridge penalty λ/*w*². This is a
  data-derived per-voxel prior, the soft counterpart of standard encoding-based voxel selection.

## Main findings

1. **Encoding-predictability weighting is the largest gain found in this project.**
   - The locked W1 (γ = 2 in every 100% cell) raised stride-4 test r at 100% labels in every exposed subject:

     | Subject | W0 | W1 | Gain |
     |---|---:|---:|---:|
     | UTS01 | 0.134 | **0.195** | +46% |
     | UTS02 | 0.148 | **0.212** | +43% |
     | UTS03 | 0.156 | **0.213** | +37% |
     | UTS09 | 0.078 | **0.133** | +70% |
     | UTS10 | 0.084 | **0.141** | +69% |

   - The paired gains were **+0.055 to +0.064**. W1 was ahead in 11/11 test stories in each subject (sign-flip
     p = .001).
   - For scale, v3's structured decoder V4 gained +0.004 over the same baseline. Stride-4 W1 already beats v3's
     dense-clock V4 (0.179 / 0.200 / 0.200).
2. **Confirmed in fresh recordings, as registered.** W1 − W0 (stride-4, 100%, test) in UTS04–UTS08 was:

   | Subject | W1 − W0 |
   |---|---:|
   | UTS04 | +0.058 |
   | UTS05 | +0.055 |
   | UTS06 | +0.054 |
   | UTS07 | +0.037 |
   | UTS08 | −0.002 |

   - That is positive in 4/5 subjects, which meets the registered rule (at least 4 of 5), so the effect is
     **confirmed**.
   - As stated at registration, 4/5 alone gives an exact sign-test p of only 0.19.
   - UTS08 is the weakest subject (W0 0.045). Its encoding scores were at chance (48–50% of columns positive), so
     CV chose the mildest rule (γ = 0.5).
3. **The gain is not speech rate, alignment or drift.**
   - **Residualized endpoint:** after removing training-fit position and oracle speech-rate predictions from
     the targets, W1 − W0 stays at **+0.052 to +0.063** in the exposed subjects. In the fresh subjects it is
     +0.036 to +0.052 (UTS08 −0.0004).
   - **Brain-input controls:** the circular-shift null is at most 0.008 at the 95th percentile in the exposed
     subjects, and at most 0.011 in the fresh ones (five test stories each). Every observed W1 score has
     p = 0.0005, the smallest attainable with 2,000 draws. Two exploratory
     controls are near zero in all ten subjects: training targets rotated within runs (the whole weighting
     pipeline refit) and wrong-run inputs (|r| ≤ 0.013).
   - **Retrieval:** within-story retrieval rank rose too, e.g. UTS01 0.780 → 0.852 and UTS09 0.692 → 0.773.
4. **It adds to dense training.** Dense-clock W1 reached test r **0.230 / 0.255 / 0.248** in UTS01/02/03,
   against dense W0 0.175 / 0.195 / 0.197.
   - UTS09/UTS10: 0.166 / 0.176, against 0.110 / 0.119.
   - The two effects are nearly additive. In UTS01–03 the dense clock added +0.035 to +0.043 to W1, and
     +0.041 to +0.047 to W0.
   - Fresh subjects: dense W1 − W0 was positive in **all five** (+0.035 to +0.072), including UTS08.
   - In the fresh subjects, stride-4 labels are scarce (about 1,200 targets), and the dense clock helped W1
     more than W0 (+0.045 to +0.065, against +0.027 to +0.045). Four times as many labeled pairs appear to make
     the encoding scores informative.
   - In UTS08, stride-4 weighting failed, but dense W1 beat dense W0 by +0.035.
5. **It needs labels; it is not a label-efficiency method.** Test W1 − W0 by number of labeled targets, pooled
   over all budgets and subjects:

   | Labeled targets | W1 − W0 (test) |
   |---|---|
   | ≤ 242 | −0.004 to +0.022 (UTS09/10 at 1%: not estimable) |
   | 452–605 | +0.008 to +0.033 |
   | 1,130–4,518 | +0.037 to +0.064 (except UTS08), roughly flat above ~1,100 |

   - With few labeled pairs, the encoding scores are dominated by noise. Only about half of the columns have
     r > 0 even at 452 pairs, so the weights carry little information.
6. **Once tuned, soft and hard weighting converge, and the registered γ grid was not limiting.**
   - Within the registered grid, hard top-10% selection (W2) captured most of the gain: +0.043 / +0.048 /
     +0.044 / +0.033 / +0.035 in the exposed subjects, still below W1 everywhere.
   - An exploratory, validation-only sweep beyond the grid (UTS01–03, 100%;
     [table](RESULTS.md#exploratory-beyond-grid-sweep-validation-only-100-stride-4-not-registered)) found:
     - soft weighting peaks at γ = 2–3 (CV best at γ = 2 in all three subjects; γ = 4 and 6 are worse), so the
       locked γ = 2 was not truncated by the grid edge;
     - hard selection keeps improving down to the **top 2% of columns** (validation 0.172 / 0.195 / 0.204);
       that is within ±0.004 of soft γ = 2 (0.170 / 0.197 / 0.208).
   - The decodable signal is concentrated in a few thousand (voxel, delay) columns.
7. **Data note.** UTS04's validation session (session 6) is near zero for every decoder: all five stories
   ≤ 0.024, while UTS05's session 6 scores 0.06–0.11. UTS04's training-CV and test sessions are normal.
   Validation was not used for any §12 selection, but that recording session appears unusable.

## Gates (§12), fixed before implementation

| Arm | Δ validation at 100% (UTS01/02/03) | Mean | Δ at 10% | Mean | Passed |
|---|---|---:|---|---:|---|
| W1 | +0.0509 / +0.0656 / +0.0599 | +0.0588 | +0.0215 / +0.0300 / +0.0239 | +0.0251 | yes → **advanced** |
| W2 | +0.0431 / +0.0492 / +0.0438 | +0.0454 | +0.0169 / +0.0183 / +0.0158 | +0.0170 | yes |

- W0 reproduced the v3 V0 development cells to about 1e−16, with identical α.
- The lock (code `6825989e5ffd`) hashes all 18 development cells and was committed (`d2fe833`) before any test
  score and before the UTS04–08 download.

## Process notes

- **Audit before results.** An independent audit (GPT-6.1-Sol) of the first implementation found no leakage in
  the main paths. It did find enforcement and edge-case gaps:
  - the gates could be computed from a subset of subjects;
  - hard-rule boundary ties were unresolved;
  - an unregistered uniform fallback could occur when nesting left too few encoding folds;
  - the residualized variant leaked across encoding folds;
  - avoidable dense-clock memory use.

  All were fixed. The first development launch was stopped, and its outputs were deleted unread (recorded in
  §12).
- **Cost.** A stride-4 CV fold takes about 46 s at a 6.8 GiB peak. A dense-clock fold (N ≈ 18,000) takes about
  5 min at a 32–39 GiB peak.
- **Figures:**
  - [subjects](figures/v3w_subjects.png): W0 vs W1 and dense, for all ten subjects;
  - [label budgets](figures/v3w_budgets.png);
  - [raw vs residualized](figures/v3w_residualized.png).

## Interpretation and novelty

- **What the result is.** Uniform all-voxel ridge spreads its prior over about 250,000 (voxel, delay) columns.
  Most of them carry no recoverable semantic signal. Concentrating the prior on columns that a training-only
  encoding model can predict raises decoding r by 40–80%, once there are enough labeled pairs to find them.
- **What it is not.** Selecting voxels by encoding performance is standard practice. Examples are classic image
  identification (Kay et al. 2008), stability-based selection (Mitchell et al. 2008) and the Huth-lab language
  decoder (Tang et al. 2023). Per-feature ridge penalties are banded ridge or empirical Bayes. Our uniform
  baseline was weaker than published pipelines, which is why the relative gains look large.
- **What is new here.**
  - The effect size measured with strictly nested, label-accounted selection.
  - Its label threshold (about 500–1,000 pairs).
  - That it is additive with the dense clock and survives speech-rate removal.
  - A registered fresh-subject confirmation.

## Project conclusion (v1 → v3 §12)

The project asked whether self-supervised fMRI pretraining makes semantic decoding more label-efficient. On the
same endpoint (v2 onward), the answer is no:
- frozen and fine-tuned SSL encoders never beat ridge on raw voxels;
- latent prediction was worse than reconstruction;
- reconstruction pretraining matched random initialization.

The large, replicated gains came from **how the supervised signal is used**. UTS01's test r rose through these
steps:

| Decoder | Test r (UTS01) |
|---|---:|
| v2 ridge | 0.122 |
| v3 V4 structured ridge | 0.137 |
| v3 dense V4 | 0.179 |
| §12 W1 | 0.195 |
| dense W1 | **0.230** |

- Every step was registered and gated.
- Dense training and weighting were replicated in fresh subjects.
- The best decoder is a linear model: dense targets with an encoding-weighted voxel prior.

Any future representation-learning claim on these data should be measured against that baseline, with
residualized endpoints and nested label accounting.

## Recommended next steps (novelty)

1. **Label-free or transferred voxel priors.** Weighting fails below about 500 labeled pairs, which is the
   regime where label efficiency matters. Candidate sources of weights that need no semantic labels: other
   listeners' responses (inter-subject correlation in a shared surface space), other subjects' encoding maps,
   and self-supervised voxel predictability.
2. **A scaling law for voxel weighting.** Predict the gain from labeled pairs × encoding SNR × the fraction of
   informative voxels, and test it across subjects, budgets and clocks.
3. **Downstream text.** Test whether the better embedding decoder improves text reconstruction (word error
   rate, BERTScore), on unseen stimulus text.
4. **Reality check.** Compare published fMRI foundation models against dense W1 under this protocol.

## Reproduction (inside the container, `/workspace/SSLTang`)

```bash
for S in UTS01 UTS02 UTS03; do
  .venv/bin/python scripts/run_v3_weights.py --stage develop --set data.manifest=data/semantic/$S/semantic_manifest.json
done
.venv/bin/python scripts/run_v3_weights.py --stage lock
bash scripts/run_v3w_evaluation.sh                       # evaluate + residualized + dense: UTS01-03, UTS09/10
bash scripts/run_v3w_fresh.sh prepare UTS04 UTS05 UTS06 UTS07 UTS08
bash scripts/run_v3w_fresh.sh evaluate UTS04 UTS05 UTS06 UTS07 UTS08
for S in UTS01 UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08 UTS09 UTS10; do
  .venv/bin/python scripts/v3w_controls.py --subject $S   # exploratory
done
for S in UTS01 UTS02 UTS03; do .venv/bin/python scripts/v3w_explore_grid.py --subject $S; done   # exploratory
python scripts/summarize_v3w.py && python scripts/plot_v3w.py   # locally, after pulling results
```
