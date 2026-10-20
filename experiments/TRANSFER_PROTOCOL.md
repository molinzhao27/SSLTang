# Held-out ds004510 evaluation

This evaluator measures coarse within-subject narrative matching with an already
trained semantic checkpoint. It performs no training, fine-tuning, checkpoint
selection, normalization fit, projection fit, or text generation. It does not
compute the original paper's BERTScore. Run transfer evaluation only after the
ordinary ds003020 results and checkpoint choice are reliable. The command itself
has no acknowledgment or frozen-gate requirement.

## Command and inputs

From the repository root, use the existing `configs/` directory:

```bash
python scripts/evaluate_transfer.py --config configs/transfer_uts01.yaml
```

Set `evaluation.manifest` to the actual **neural transfer manifest** produced by
`preprocess.py --mode transfer`. The shared YAML now names the neural manifest;
use `--set evaluation.manifest=PATH` if artifacts live elsewhere. Do not build
timed imagined targets to satisfy this command. A semantic wrapper is accepted
only when it retains every original transfer run and verifies its source hash.
The timed-target builder intentionally drops imagined runs; that output is
rejected if it omits runs. There is no silent fallback to another manifest.

The required evaluation configuration is:

```yaml
evaluation:
  manifest: data/transfer/UTS01/manifest.json
  checkpoint: results/runs/random_UTS01_fraction1_seed17/best.pt
  output_dir: results/transfer
  exclusion_tr: 10
```

Paths in configuration follow the repository's working-directory convention.
Paths stored inside manifests resolve against their owning manifest directory.
Optional settings are `device` (otherwise inherited from `training.device`),
`reference_device` (otherwise the inference device), `batch_size: 64`,
`feature_batch_size: 16`, `reference_dir`, and `split: transfer`. For CPU, pass
`--set evaluation.device=cpu`. Only `protocol: run_level` is supported.
CLI `--set dotted.key=value` overrides can be repeated. Importing the module or
running `--help` does not instantiate/download the language model or open neural
arrays. Frozen GPT2 is instantiated lazily when an uncached reference is encoded.

## Checkpoint and neural representation

Load the complete semantic checkpoint with its saved `encoder_config` and
`model_state_dict` into `FmriEncoder` and `SemanticRegressor`. The contracted
architecture is 384 dimensions, four layers, maximum sequence length 128, with
end pooling; the saved training window must be 32 TR and semantic lag 3 TR. Other encoder widths/depths load from their saved configuration unchanged.
Inference uses evaluation mode, disabled gradients, and FP32. A constant saved-training-mean vector is scored as an additional baseline. Each output is
inverse-scaled with the checkpoint's saved `target_std` and `target_mean` before
run averaging. No transfer statistics affect those values.

The training semantic manifest must match the checkpoint's SHA256 and identify
only ds003020 runs. Its source neural manifest, when supplied, must match the
recorded source hash. Transfer must contain only ds004510 runs with split
`transfer`, the same subject, and exactly the checkpoint feature count. Semantic
enrichment must preserve neural run identities and paths.

The evaluator verifies `normalized: true`, matching explicit feature order, and
the training/transfer `normalization.npz`: either its entire file SHA256 matches,
or all saved arrays match exactly, including dtype and shape. Checks cover
mean, std, spatial membership, coordinates, subject, fit run IDs, fitted TR
count, and block size. Fit IDs/count must equal the source ds003020 training
partition. The source manifests' normalizations must agree too. No extra
normalization is applied to the already normalized transfer arrays.

This provenance supports the authors' shared within-subject column convention;
the published H5 files carry no independent per-column coordinate map. Equal
feature counts alone do not establish alignment. The results retain this caveat.

## Prefix windows and averaging

For each run independently, select every zero-based neural endpoint
`end_tr = 7, 8, ..., n_tr-1`. Use rows
`max(0,end_tr-31):end_tr+1`, so lengths grow from 8 to 32 and then remain 32.
There is no padding and no window crosses a run boundary. The checkpoint's model
output corresponds to `model_target_tr = end_tr-3`; retain only target indices
inside that run. A 30-TR imagined repeat has 23 proposed windows, lengths 8–30,
neural endpoints 7–29 and model target indices 4–26. Reject a window with
nonfinite neural values or a nonfinite inverse-scaled prediction. Average all
remaining output vectors with equal endpoint weight to obtain one run vector.

Prefix inference is deliberately coarser than the 32-TR supervised training
context. `model_target_tr` documents the learned lag, not observed imagined-word
timing. Overlapping windows do not become independent statistical trials.
`exclusion_tr: 10` is recorded but has no application to this run-level candidate
pool: candidates have no TR clock, and no ten-TR trim or temporal-negative
exclusion is performed.

## Imagined speech: five unique references

The candidate identities are **alpha, bravo, charlie, delta, echo**. All repeats
of a narrative share its single candidate vector. Clean words from the supplied
TextGrid's explicitly named word tier, discard timestamps, join them in supplied
order, and call `features.run_reference_embedding`. This averages causal frozen
word embeddings over the **whole cue/overt reference transcript**. It does not
construct synchronized imagined semantic targets or establish which words the
participant imagined at any TR. References supplied for repeats must agree.

The frozen encoder is `openai-community/gpt2`, immutable revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, hidden state 9, 768 dimensions,
20 preceding words plus the current word, last-subword pooling. Training semantic
metadata must verify this same pin and representation. Text/reference features
are encoded separately from the neural model, only during explicit evaluation.

## Perceived movies: conservative reference proxy

The fixed candidates are **laluna, partlycloudy, presto**, each represented by
the mean frozen embedding of its whole supplied description/reference TextGrid.
Sintel is explicitly skipped because no published reference TextGrid exists.
No captions are invented. The current evaluator supports only the run-level
caption/reference proxy; it does not infer time-resolved movie scene targets.

Read-only inspection of the public TextGrid metadata found exactly one interval
tier named `word` for all five imagined references and all three movie references.
This satisfies the existing `features.read_word_intervals` parser requirement.
Word intervals and a dataset description of aligned transcripts do not establish
alignment of those descriptions to movie scenes or imagined words. The checked-in
Tang/Huth TextGrid parser supplies parsing, without importing its decoding models.
Sources: [ds004510 README](https://s3.amazonaws.com/openneuro.org/ds004510/README),
[public TextGrid listing](https://s3.amazonaws.com/openneuro.org?list-type=2&prefix=ds004510/derivative/TextGrids/),
and [reference repository](https://github.com/HuthLab/semantic-decoding).

## Metrics and clustered chance

Compute cosine against the fixed unique candidates separately in two spaces:
raw inverse-scaled embeddings, and embeddings with the checkpoint's training
target mean subtracted from **both** predictions and references. Centering uses
saved training statistics and fits nothing to transfer. Report each run's true
candidate cosine, top-1 accuracy, reciprocal rank, rank, and all candidate cosines.
Top-1 random-label chance is **20%** for imagined speech and **1/3** for movies;
MRR chance is the harmonic mean reciprocal rank (about 0.4567 and 0.6111).
Cosine ties use expected metrics under uniform ordering within the tied group;
the absolute tie tolerance is `1e-8`.

Report run-weighted metrics and equal-narrative macro metrics, with per-narrative
repeat summaries and run IDs. The exact null enumerates all 120 imagined label
permutations or all six movie label permutations, keeping predictions and the
candidate pool fixed. Every repeat of a narrative moves together. Null statistics
are equal-narrative macro top-1, MRR, and mean true-candidate cosine. Record null
means, all permutation values, and exact one-sided `p_greater_equal`, including
the identity permutation. This small narrative count yields coarse p-values;
there is no claim of independent TR/window or repeat observations.

A missing or invalid reference prevents scoring that condition with its fixed
pool; the evaluator reports reasons instead of shrinking candidate count.
Missing neural files and runs without valid endpoint predictions are explicitly
skipped. Undefined cosine vectors are marked unscorable. Available runs can still
be summarized, with incomplete coverage recorded and fixed chance unchanged.

## Artifacts and validation boundary

The separate evaluation output contains `results.json`, per-condition prediction
NPZs, and content-addressed `reference_targets/` NPY/JSON caches. Each cache records
condition, narrative, run IDs, source paths/hashes, cleaned-text identity, frozen
model pin, parser/features code hashes, feature batch size, and its vector hash.
Only finite float32 vectors with matching metadata and file hash are reused.
Results include protocols, missing reasons, selected endpoints/lengths and rejected
windows, raw/centered retrieval, repeat-clustered summaries/permutations, checkpoint
metadata/hash, training/source/transfer manifest hashes, normalization file and
array hashes, reference/cache hashes, evaluated neural file hashes, prediction
artifact hashes, and evaluator/model/features code hashes. JSON is strict (no
NaN), and results are written atomically. Source/checkpoint/normalization changes
during evaluation abort publication. The output directory must be separate from
all input artifacts. Source arrays, manifests, TextGrids, and checkpoints are
never modified.

Implementation verification uses synthetic fixtures and CLI help only. No real
transfer inference, neural-array inspection, label-based tuning, or GPT2 loading
is authorized during implementation; the explicit evaluation command performs
that work later with the chosen checkpoint.
