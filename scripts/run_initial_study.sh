#!/usr/bin/env bash
# Run inside Linux project root after download/preprocessing/targets.
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${SSLTANG_PYTHON:-.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
"$python_bin" scripts/run_experiments.py --stage ssl --seeds 17 23 47
"$python_bin" scripts/run_experiments.py --stage milestone --seeds 17 23 47
"$python_bin" scripts/run_experiments.py --stage labels --seeds 17 23 47
"$python_bin" scripts/plot_results.py
