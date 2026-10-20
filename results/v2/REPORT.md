# Protocol v2 report: strong readouts, latent versus reconstruction pretraining, and replication

**Date:** 2026-10-03. **Plan:** FUTURE_WORK.md. **Protocol, decision rules and
corrections:** [experiments/PROTOCOL_V2.md](../../experiments/PROTOCOL_V2.md). Training and evaluation ran on the
remote RTX PRO 6000. Archived v1 results are unchanged.

The primary endpoint throughout is the **test session-macro feature correlation**: the mean over the 768 GPT-2
dimensions of the temporal Pearson r within each held-out story, averaged within session and then over the two
test sessions. Every option, alpha and checkpoint was chosen on ds003020 validation stories. UTS01 results are
exploratory, because its test stories were examined in the archived study. UTS02/UTS03 contrasts and the transfer
rule were registered before those subjects were processed or scored.

## Main findings

1. **Plain ridge on raw BOLD is the strongest decoder in every subject.** It concatenates pooled BOLD at t+1, t+2,
   t+3 (2–6 s after the word bin) and reaches test r **0.122 / 0.137 / 0.143** (UTS01/02/03). That is 1.5–2.1 times
   the archived approach, a fine-tuned Transformer: 0.061–0.095 with reconstruction initialization. Ridge also beats
   every frozen-encoder readout, at every label budget.
2. **The plan's central hypothesis is not supported.** EMA latent prediction was *worse* than reconstruction under
   the locked frozen readout in all three subjects: Δr **−0.0144 / −0.0110 / −0.0131**, all 9 seed pairs negative.
   It was also worse at every budget, at both latencies, with both readouts, and under fine-tuning. It did not
   collapse. Instead it compressed the representation (effective rank ≈ 123–132 vs 183–206 for reconstruction) and
   lowered decodability at every layer.
3. **Masked temporal reconstruction learns almost nothing that generalizes.** On the exact SSL validation masks,
   predicting the training mean scores within 1.5% of the trained model's MSE. Frozen reconstruction encoders were
   no better than frozen *random* encoders: Δr +0.0026 / −0.0033 / +0.0008. Both were below raw input.
4. **Counted supervision can shrink.** Choosing ridge alpha by leave-one-story-out CV inside the labeled subset
   matches validation selection at every budget, so the 794 validation labels are unnecessary for this decoder.
   Spreading labels over more stories helps little. With 10% of labels the benchmark keeps about 44% of its
   full-label r.
5. **Spatial pooling costs about 8%.** Kernel ridge on all 81–96k voxels adds **+0.011 / +0.011 / +0.013** r at
   100% labels (to 0.134 / 0.148 / 0.156). This analysis is exploratory.
6. **Imagined-speech transfer is directional but not established.** The ridge decoder ranks the correct imagined
   narrative first **60% / 50% / 60%** of the time (5-way, chance 20%; exact permutation p = .017 / .067 / .033). The
   registered rule needs p ≤ .05 in both new subjects, so transfer is *not established*. Films stayed at chance.
   For comparison, the archived fine-tuned models reached 27–37% on UTS01.

## Decisions under the plan's gates (FUTURE_WORK §6)

| Gate | Observation | Decision |
|---|---|---|
| Frozen SSL + ridge outperforms raw/PCA/random features | Beats window PCA. Ties random features. Loses to raw BOLD by 0.024–0.027 r in all subjects. | No evidence that reconstruction pretraining makes semantic information more accessible than the input |
| A readout/timing change helps every method similarly | FIR pooling over t+1…t+3 helps raw and encoder features alike (+0.013 to +0.024 r). The 8-s cutoff adds nothing. | Credit the readout, not SSL |
| Latent prediction improves semantic endpoints | It is worse than reconstruction in 3/3 subjects and 9/9 seed pairs | Keep the simpler method; the semantic hypothesis is unsupported in this configuration |
| Aligned predictions exceed misaligned controls | Every 100%-label arm is far above the within-run circular-shift null (q95 ≈ 0.007, p = 0.0005) | Effects are stimulus-aligned |

## Phase results

**Phase A — readout baselines** ([table](phase_a/RESULTS.md)). UTS01, 6-s cutoff, test r at 100% / 10% labels:

| Arm | 100% | 10% |
|---|---:|---:|
| Raw BOLD, ridge at t+1…t+3 | 0.1221 | 0.0528 |
| Raw BOLD, ridge at t+3 only | 0.0980 | 0.0442 |
| Frozen reconstruction encoder, ridge (final / best-val checkpoint) | 0.0978 / 0.0984 | 0.0401 / 0.0394 |
| Frozen random encoder, ridge | 0.0944 | 0.0395 |
| Window PCA, ridge | 0.0579 | 0.0160 |
| Full fine-tuning, reconstruction / random init (seed ranges 0.04–0.08) | 0.0658 / 0.0638 | 0.0102 / 0.0083 |

Validation r falls with encoder depth in every encoder. 161 of 180 encoder cells selected layer 1. Pretraining
slows this decline (layer 4: 0.077 vs 0.064 for random) but never lifts any layer above the raw input.

**Phase B — objectives** ([table](phase_b/RESULTS.md)). The two objectives shared windows, masks (a dedicated mask
generator), encoder, optimizer and 1,110 updates. Each exported its final student encoder. The latent arm cost
35–38 s and 0.76 GiB peak memory per fit, against 32–33 s and 0.72 GiB for reconstruction.

| UTS01, 6-s FIR, test r | 100% | 10% |
|---|---:|---:|
| Reconstruction, frozen | 0.0970 [0.0953, 0.0997] | 0.0410 |
| Latent, frozen | 0.0826 [0.0795, 0.0844] | 0.0321 |
| Latent − reconstruction (seed pairs) | −0.0144 (−0.016, −0.015, −0.012) | −0.0088 |

**Phase C — label coverage** ([table](phase_c/RESULTS.md)). Raw-BOLD ridge, 5 subset seeds per cell.

**Phase V — all voxels** ([table](phase_v/RESULTS.md)). Same targets and selection as the raw benchmark.

**Phase D — registered replication** ([table](replication/RESULTS.md); per subject:
[UTS02](replication/UTS02/RESULTS.md), [UTS03](replication/UTS03/RESULTS.md)). Values are test r at 100% labels.

| Contrast | UTS01 (exploratory) | UTS02 | UTS03 | Replicated |
|---|---:|---:|---:|---|
| D1 frozen latent − frozen reconstruction | −0.0144 | −0.0110 | −0.0131 | yes, both |
| D2 raw BOLD ridge − frozen reconstruction | +0.0252 | +0.0243 | +0.0272 | yes, both |
| D3 raw BOLD ridge − fine-tuned reconstruction init | +0.0615 | +0.0726 | +0.0480 | yes, both |
| D4 frozen reconstruction − frozen random (no prediction) | +0.0026 | −0.0033 | +0.0008 | — |

All three directional contrasts also hold at 10% labels.

**Transfer** ([table](transfer/RESULTS.md)). See finding 6. The 3-TR ridge input avoids the archived
short-context mismatch (FUTURE_WORK §3.3).

Figures: [Phase A label efficiency](figures/phase_a_label_efficiency.png),
[objectives](figures/phase_b_objectives.png), [coverage](figures/phase_c_coverage.png),
[pretraining diagnostics](figures/ssl_diagnostics.png), [replication](figures/replication_summary.png).

## Interpretation

Semantic information about a word bin sits mainly in the BOLD signal 2–6 s later. Ridge reads it directly from
all spatial features. The encoder instead projects 1,695–1,977 pooled features to 384 dimensions per TR and mixes
32 TRs. Its frozen states lose information, and deeper layers lose more.

The pretext task supplies little to recover that loss. With whole-time-point 8-TR cores and 4-TR guards, held-out
reconstruction is near the mean-predictor floor. Latent targets drop dimensions the semantic readout needed. The
unlabeled data are the same subject's ~10 h of training runs that ridge already uses, so pretraining has no new
information source. The gains reported for fMRI foundation models come from thousands of multi-subject hours,
which is outside this design.

## Limitations

- One whole-session split. Two test sessions per subject. The same 11 test stories in all subjects: the
  replication used new recordings, not unseen text. Seeds are not independent participants. Story sign-flip
  p-values are descriptive.
- One SSL configuration: 4 layers, width 384, temporal 8-TR masks, 1,110 updates, a linear predictor, and EMA
  ramping 0.996 → 0.9999964. Other masks, predictors, scales or pooled multi-subject pretraining were not tested.
- Frozen readouts are linear, and feature counts differ (5,085–5,931 raw vs 1,152 encoder). Selection grids
  differ in size, but selection never used test data.
- Targets are GPT-2 layer-9 embeddings; there is no text generation. Transfer is coarse run-level retrieval over
  5 narratives (2 repeats each) and 3 films.

## Recommended next steps

1. Report raw-BOLD and voxel ridge as the benchmark for any future SSL claim. An SSL method must beat them, not
   only random initialization.
2. Before any semantic test, require a pretext task that is learnable on held-out data: the reconstruction
   controls must show clear gains over the mean and interpolation. Options include spatial or joint masks with
   interpolation controls, wider encoders without the 384-dimension bottleneck, and multi-subject pretraining with
   functional alignment (FUTURE_WORK §8).
3. The imagined-speech retrieval (50–60% vs 20%) merits a larger registered transfer test: more narratives,
   subjects and repeats, using the voxel decoder. Adding seeds will not help.
4. Use voxel-level decoders, or encoding-model approaches (Tang et al., 2023), when decoding performance is the
   goal.

## Reproduction

Inside the container (`/workspace/SSLTang`). Logs are in `results/logs/v2-*.log`.

```bash
.venv/bin/python scripts/run_phase_a.py                        # Phase A ridge arms (UTS01)
.venv/bin/python scripts/run_v2_finetune.py --root results/v2/finetune_a --arm random=random \
  --arm 'ssl_best=results/runs/ssl_UTS01_seed{seed}/encoder_best.pt' --cutoffs 3 4 --fractions 1 0.1
.venv/bin/python scripts/run_phase_b.py --stage pretrain      # then --stage readout, --stage finetune
.venv/bin/python scripts/run_phase_c.py
.venv/bin/python scripts/run_voxel_ridge.py                    # exploratory
.venv/bin/python scripts/reconstruction_controls.py
bash scripts/run_replication.sh UTS02                          # and UTS03 (after downloads)
.venv/bin/python scripts/run_ridge_transfer.py --set data.manifest=data/semantic/UTS01/semantic_manifest.json \
  --transfer-manifest data/transfer/UTS01/manifest.json --out results/v2/transfer/UTS01
```

Locally, `scripts/pull_v2.sh` fetches results. `scripts/summarize_v2.py --report {phase_a,phase_b,phase_c,voxel,transfer,replication}`
and `scripts/plot_v2.py` regenerate every table and figure from the versioned JSON files.
