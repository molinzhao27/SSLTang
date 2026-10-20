#!/usr/bin/env bash
# PROTOCOL_V3 §11: process one fresh subject and run the locked v3 procedure.
# Every stage resumes or skips completed work. Run inside the container.
set -euo pipefail
cd "$(dirname "$0")/.."
subject="${1:?Usage: run_v3_replication.sh UTS09|UTS10}"
py="${SSLTANG_PYTHON:-.venv/bin/python}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HOME="$PWD/.cache/huggingface" TOKENIZERS_PARALLELISM=false
processed="data/processed/$subject"
semantic="data/semantic/$subject"
[[ -f "$processed/manifest.json" ]] || "$py" scripts/preprocess.py --root data/raw --out "$processed" \
  --mode train --subject "$subject" --block-size 4 --session-map configs/huth_session_map.json --seed 42
[[ -f "$semantic/semantic_manifest.json" ]] || "$py" scripts/build_semantic_targets.py \
  --manifest "$processed/manifest.json" --output-dir "$semantic" --device cuda --batch-size 256
for stage in cache evaluate controls dense; do
  "$py" scripts/run_v3_replicate.py --stage "$stage" --subject "$subject"
done
"$py" scripts/v3_exposure.py --config configs/v2/phase_a.yaml --set "data.manifest=$semantic/semantic_manifest.json" \
  --out "results/v3/$subject/exposure.json" --tokenizer-model openai-community/gpt2 \
  --tokenizer-revision 607a30d783dfa663caf39e06633721c8d4cfcd7e
echo "v3 replication complete: $subject"
