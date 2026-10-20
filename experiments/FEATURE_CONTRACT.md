# Frozen semantic targets

This first experiment predicts embeddings from fMRI; it does not generate text.
The frozen encoder is Hugging Face `openai-community/gpt2`, revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, with `hidden_states[9]`
(the output of transformer block 9), dimension 768. Each word is embedded
with up to 20 preceding words plus the current word; its final subword token
provides the vector. GPT2 runs in evaluation mode with gradients disabled.
The tokenizer and checkpoint use the same immutable revision; custom models
must supply an immutable 40-character revision. Transformers downloads cache
the checkpoint. No corpus-dependent fitting or normalization happens here.

This is inspired by Tang/Huth, **not their exact GPT checkpoint or pipeline**.
The inspected reference `config.py` sets layer 9 and `GPT_WORDS=5`; our 20-word
causal context is a deliberate choice. Their `utils_stim.py` uses Lanczos
resampling and FIR delays [1,2,3,4] TRs. Here word midpoint vectors are averaged
within each centered TR interval. Empty bins are zero and explicitly invalid;
zeros must never become training labels. This simpler resampling is documented
rather than represented as an exact replication.

`build_semantic_targets.py --manifest PATH --output-dir DIR` reads neural
manifest runs and writes a separate `semantic_manifest.json`. Each included run
adds `target_path` ([T,768] float32), `target_valid_path` ([T] bool), and immutable
encoder metadata. Paths in the input resolve against its manifest directory.
The output is outside SSL inputs; neural manifests are never modified.
Content-addressed per-run caches include text bytes, metadata, run fields, and
actual TR centers. Partial or malformed caches are rebuilt.

The run must provide `stimulus_path` to a verified word TextGrid, plus
`tr_times_path` (a length-T NumPy array of acquisition centers relative to
stimulus onset) or `tr_start_seconds` (the first such center) and `tr_seconds`.
No timeline is inferred from row count. The reference `load_simulated_trfiles`
uses TR=2 s, sound start=10 s, and 5 trailing padding rows;
`DataSequence.from_grid` adds half a TR, and `utils_stim` slices `[10:-5]`.
Thus matching retained Huth training rows start at stimulus time **11 s**.
The preprocessing component owns this verification and supplies the timing.
Transfer audible-speech centers start at 1 s only when its acquisition/stimulus
alignment is independently verified; a TextGrid alone does not establish it.

Targets are **unlagged** stimulus features. Training pairs neural rows at
stimulus time +4,+6,+8 seconds with these targets; training owns lagging and
mask propagation, including run boundaries. Target standardization and any
projection must be fit exclusively on training runs. Validation/test text is
used only to construct evaluation targets from the same frozen encoder.

Runs without verified timed text are listed in `skipped_runs`. Imagined speech
is skipped even if an overt-speech cue transcript exists: those times are not
imagined-word timestamps. Video receives no invented captions. ds004510 must
be processed in a separate output directory at the authorized evaluation stage.

The semantic manifest preserves the source manifest header, including dataset,
subject, mode/schema, and neural normalization identity. It additionally records
`neural_manifest_path` and `neural_manifest_hash` for the immutable source file.
`--reference-dir` selects the checked-in Tang/Huth parser root explicitly.

`run_reference_embedding(text, encoder)` is a separate coarse reference helper:
it averages causal word features across a supplied whole-story cue/reference.
Its [768] output has no timestamps and must not be represented as timed imagined
speech targets. A transfer evaluation caller must maintain this distinction.

Validation on 2026-10-03 used the remote CUDA environment (torch 2.14,
transformers 4.57.6): the pinned GPT2 encoded five actual input words into finite
float32 [5,768] vectors. Batch sizes 2 and 5 differed by at most 0.0002442 due
to floating-point kernel batching. The run-level helper returned [768]. The
actual adollshouse TextGrid parsed into 1,654 cleaned timed words. Five local
alignment/cache tests passed. Full-corpus feature extraction is not yet run.
