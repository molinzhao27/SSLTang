#!/usr/bin/env bash
# PROTOCOL_V4 sequential queue after G4 (one GPU job at a time; resumes ds003020 development where cells are missing
# under the current core, then the efficacy gates, then LPP DEV development (L3 for all, then L5/L8), then
# descriptive stride-4 cells). Steps that are already complete are skipped, so a rerun resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
py="${SSLTANG_PYTHON:-.venv/bin/python}"
dev=(UTS01 UTS02 UTS03 UTS05 UTS06 UTS07 UTS08 UTS09 UTS10)
budgets=(${V4_BUDGETS:-S3 S6 S15 ALL})  # V4_BUDGETS="S3 S6 S15" runs only the low-memory budgets
only_dense=${V4_ONLY_DENSE:-0}  # 1: stop after the dense ds003020 budgets (no gates, LPP or stride-4)
lpp_clocks=${V4_LPP_CLOCKS:-dense}  # LPP DEV develop clocks (trimmed plan: dense only; every registered cell is dense)
for w in ${V4_WAIT_SESSIONS:-}; do  # wait for earlier GPU jobs (tmux sessions) so the ALL budgets run alone
  while tmux has-session -t "$w" 2>/dev/null; do sleep 300; done
done
missing() {  # missing <subject> <clock> <budget>: exit 0 if any registered cell of the budget is absent or stale
  "$py" - "$@" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import run_v4
subject, clock, budget = sys.argv[1:4]
arms = run_v4.ALL_ARMS if budget == 'ALL' else run_v4.TIER1
code = run_v4.code_sha256()
for arm in arms:
    p = Path(f'results/v4/{subject}/{clock}/develop/cells/{budget}_{arm}.json')
    if not p.exists() or json.loads(p.read_text()).get('code_sha256') != code:
        sys.exit(0)
sys.exit(1)
PY
}
for s in "${dev[@]}"; do
  for b in "${budgets[@]}"; do
    if missing "$s" dense "$b"; then "$py" scripts/run_v4.py --stage develop --subject "$s" --clock dense --budgets "$b"; fi
  done
done
[[ "$only_dense" == 1 ]] && { echo "dense budgets ${budgets[*]} complete"; exit 0; }
"$py" scripts/v4_gates.py --stage efficacy
if [[ "$("$py" -c "import json; print(json.load(open('results/v4/gates.json')).get('advanced'))")" == None ]]; then
  echo "development null: CONFIRM stays untouched; stopping before LPP DEV develop"; exit 0
fi
lppdev=$("$py" -c "import json; print(' '.join(json.load(open('data/lpp/prepare_provenance.json'))['partition']['dev']))")
# The primary budget first for every DEV listener (everything the lock reads), then the descriptive budgets
# (L5/L8; develop is refused once the lock exists, so these must finish before it or be dropped).
for s in $lppdev; do "$py" scripts/run_v4_lpp.py develop --subject "$s" --clocks $lpp_clocks --budgets L3; done
echo "LPP DEV primary budget (L3) complete"
[[ "${V4_LPP_PRIMARY_ONLY:-0}" == 1 ]] && { echo "stopping before the descriptive LPP budgets"; exit 0; }
for s in $lppdev; do "$py" scripts/run_v4_lpp.py develop --subject "$s" --clocks $lpp_clocks --budgets ${V4_LPP_EXTRA_BUDGETS:-L5 L8}; done
for s in "${dev[@]}" UTS04; do
  for b in S3 S6 S15 ALL; do
    if missing "$s" stride4 "$b"; then "$py" scripts/run_v4.py --stage develop --subject "$s" --clock stride4 --budgets "$b"; fi
  done
done
for b in S3 S6 S15 ALL; do
  if missing UTS04 dense "$b"; then "$py" scripts/run_v4.py --stage develop --subject UTS04 --clock dense --budgets "$b"; fi
done
echo "v4 development complete"
