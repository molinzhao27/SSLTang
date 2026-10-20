# Temporal neural encoder and masked reconstruction

`src.models.EncoderConfig(n_features=1024, d_model=384, n_layers=4,
n_heads=8, max_length=128)` defines the shared encoder. `FmriEncoder(config)`
also accepts a dictionary. Its input is floating point `[B,T,F]`; output is
`[B,T,D]`. The optional boolean observation mask has input shape and `True`
means observed. Hidden values are zeroed before the linear embedding, and a
separate learned projection encodes missingness. Even NaN hidden values cannot
enter the encoder. Learned positional embeddings support lengths up to
`max_length`. Random and SSL conditions must instantiate this exact encoder
configuration; the reconstruction head is discarded for downstream fitting.

`src.ssl.MaskedNeuralModel(encoder, mask_config)` owns an independent linear
reconstruction head. `model(x, generator=cpu_generator)` returns a dictionary
with `loss`, `prediction`, `latent`, `target_mask`, and `observation_mask`.
Call `result['loss'].backward()` in the training loop. MaskConfig can be a dict.
For repeatable validation, call `generate_mask(x.shape, config,
device=x.device, generator=torch.Generator().manual_seed(seed))` once and pass
the resulting `MaskBatch` as `model(x, mask)`. Generator state advances normally;
no global seed is reset. Masks generated on CPU are transferred to the input
device. Inputs/targets should be finite and normalized using training-only
statistics before masking. Reconstruction MSE casts to float32 before arithmetic,
which supports CUDA BF16 autocast while keeping loss accumulation stable.

Default temporal masks reconstruct contiguous 8-TR cores, with an additional
4-TR hidden guard on each side, clipped at boundaries. Only cores contribute
to loss. The requested target fraction is approximate because spans/groups are
indivisible and can overlap. Spatial mode hides whole contiguous index groups
for the full sequence. Joint mode hides temporal cores within index groups,
with temporal guards for those groups. Contiguous feature indices are anatomical
groups only when the feature ordering supplies that mapping; PCA components do
not have anatomical adjacency. Guard bands reduce adjacent-TR shortcuts but do
not establish independence beyond the chosen lag. Short sequences or dense
masks may hide most context; select mask settings for the temporal resolution
and sequence length and compare guard lengths in evaluation.

There are no text inputs, tokenizers, pretrained language models, or text
outputs. The semantic/readout module is external to this contract.

## Training and checkpoint interface

`python scripts/pretrain_ssl.py --config CONFIG.yaml` and
`python scripts/train_semantic.py --config CONFIG.yaml` accept repeated
`--set dotted.key=VALUE` overrides and `--resume latest` or a checkpoint path.
Only ds003020 training runs are fit; its validation partition selects the best
checkpoint. Each SSL validation batch gets a fixed mask seeded independently
from training. Semantic targets are standardized using only the selected
training examples; `target_mean` and `target_std` in checkpoints reverse that
normalization for evaluation. Semantic heads receive identical initialization
seeds across random/SSL arms. Both arms use the same optimizer/schedule and
step count when configuration, label fraction, and seed agree.

Full `best.pt` / `latest.pt` checkpoints save model, optimizer, scheduler,
scaler, global RNG states, epoch/batch cursor, data-loader generator state,
configuration, manifest SHA256, and git commit. Cursor checkpoints occur at
optimizer boundaries, including on SIGTERM. The epoch-specific shuffle seed
reconstructs the order when skipped deterministic windows are resumed; global
training RNG is restored without replaying their forward passes. Configuration
and manifest must remain identical on resume. `training.max_steps` supplies a
bounded pilot budget, and `training.compile` enables torch.compile. BF16 is
the default CUDA precision; reconstruction/regression arithmetic uses FP32.
TF32 defaults to enabled and can be disabled via `training.tf32=false`.

SSL also exports `encoder_best.pt` / `encoder_latest.pt`. These exclude the
reconstruction head and include an encoder configuration/state plus task and
neural signature provenance. SSL initialization rejects architecture or neural
run/representation provenance differences. Target enrichment of the same
neural manifest is supported. The semantic head maps pooled temporal latents
to continuous target embeddings and never generates text.
