#!/usr/bin/env bash
# PROTOCOL_V3 §12 locked evaluation queue (run inside the container after the §12 lock):
# five budgets + residualized endpoint, then the dense-clock factor, for UTS01-03 and UTS09/UTS10.
set -euo pipefail
cd "$(dirname "$0")/.."
py="${SSLTANG_PYTHON:-.venv/bin/python}"
for s in UTS01 UTS02 UTS03; do
  "$py" scripts/run_v3_weights.py --stage evaluate --set "data.manifest=data/semantic/$s/semantic_manifest.json"
done
for s in UTS09 UTS10; do "$py" scripts/run_v3w_replicate.py --stage evaluate --subject "$s"; done
for s in UTS01 UTS02 UTS03; do
  "$py" scripts/run_v3_weights.py --stage dense --set "data.manifest=data/semantic/$s/semantic_manifest.json"
done
for s in UTS09 UTS10; do "$py" scripts/run_v3w_replicate.py --stage dense --subject "$s"; done
echo "v3w evaluation queue complete"
