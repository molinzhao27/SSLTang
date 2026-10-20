#!/usr/bin/env bash
# PROTOCOL_V3 §12 fresh confirmatory replication (UTS04-UTS08), run inside the container after the §12 lock.
#   run_v3w_fresh.sh prepare  SUBJECT...   download, preprocess (archived split procedure) and build targets
#   run_v3w_fresh.sh evaluate SUBJECT...   locked §12 evaluate (+ residualized) and dense stages
set -euo pipefail
cd "$(dirname "$0")/.."
py="${SSLTANG_PYTHON:-.venv/bin/python}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HOME="$PWD/.cache/huggingface" TOKENIZERS_PARALLELISM=false
mode="${1:?Usage: run_v3w_fresh.sh prepare|evaluate SUBJECT...}"
shift
for subject in "$@"; do
  processed="data/processed/$subject"
  semantic="data/semantic/$subject"
  if [[ "$mode" == prepare ]]; then
    "$py" scripts/download_data.py --root data/raw --subject "$subject" --mode train
    [[ -f "$processed/manifest.json" ]] || "$py" scripts/preprocess.py --root data/raw --out "$processed" \
      --mode train --subject "$subject" --block-size 4 --session-map configs/huth_session_map.json --seed 42
    [[ -f "$semantic/semantic_manifest.json" ]] || "$py" scripts/build_semantic_targets.py \
      --manifest "$processed/manifest.json" --output-dir "$semantic" --device cuda --batch-size 256
  else
    "$py" scripts/run_v3w_replicate.py --stage evaluate --subject "$subject"
    "$py" scripts/run_v3w_replicate.py --stage dense --subject "$subject"
  fi
  echo "$mode complete: $subject"
done
