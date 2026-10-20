#!/usr/bin/env bash
# Phase D (experiments/PROTOCOL_V2.md §7): rerun the fixed UTS01 v2 pipeline for one
# new subject. Every stage resumes or skips completed work. Run inside the container.
set -euo pipefail
cd "$(dirname "$0")/.."
subject="${1:?Usage: run_replication.sh SUBJECT}"
py="${SSLTANG_PYTHON:-.venv/bin/python}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HOME="$PWD/.cache/huggingface" TOKENIZERS_PARALLELISM=false
processed="data/processed/$subject"
semantic="data/semantic/$subject"
manifest="$semantic/semantic_manifest.json"
root="results/v2/replication/$subject"
[[ -f "$processed/manifest.json" ]] || "$py" scripts/preprocess.py --root data/raw --out "$processed" \
  --mode train --subject "$subject" --block-size 4 --session-map configs/huth_session_map.json --seed 42
[[ -f "$manifest" ]] || "$py" scripts/build_semantic_targets.py --manifest "$processed/manifest.json" \
  --output-dir "$semantic" --device cuda --batch-size 256
# Phase A ridge arms B0-B3; this subject has no archived SSL encoders.
"$py" scripts/run_phase_a.py --config configs/v2/phase_a.yaml --set "data.manifest=$manifest" \
  --set "output_dir=$root/phase_a" --set 'encoders.sources={}' --set encoders.random_arm=B3
# Phase B: both objectives, then the locked frozen readout and fine-tuning.
"$py" scripts/run_phase_b.py --stage pretrain --subject "$subject" --root "$root/ssl" \
  --set "data.manifest=$processed/manifest.json"
"$py" scripts/run_phase_b.py --stage readout --subject "$subject" --root "$root/ssl" \
  --readout-out "$root/phase_b" --readout-set "data.manifest=$manifest"
"$py" scripts/run_v2_finetune.py --root "$root/finetune" --arm random=random \
  --arm "recon=$root/ssl/recon_${subject}_seed{seed}/encoder_latest.pt" \
  --arm "latent=$root/ssl/latent_${subject}_seed{seed}/encoder_latest.pt" \
  --cutoffs 3 --fractions 1 0.1 --parallel 4 --set "data.manifest=$manifest"
"$py" scripts/reconstruction_controls.py --set "data.manifest=$processed/manifest.json" \
  --out "$root/reconstruction_controls.json"
echo "replication complete: $subject"
