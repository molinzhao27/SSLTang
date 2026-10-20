#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reference
fetch() {
  local name="$1" commit="$2"
  if [[ ! -d "reference/$name/.git" ]]; then git clone "https://github.com/HuthLab/$name.git" "reference/$name"; fi
  git -C "reference/$name" fetch origin "$commit"
  git -C "reference/$name" checkout --detach "$commit"
}
fetch semantic-decoding 6b25dd5720ced1c4199cd9b9149061683b77ba6b
fetch deep-fMRI-dataset eaaa5cd186e0222c374f58adf29ed13ab66cc02a
