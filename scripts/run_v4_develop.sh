#!/usr/bin/env bash
# PROTOCOL_V4 ds003020 development queue (run inside the container after G0 has passed under the current code).
# G1 (training only) -> G-FEAS (W0, W1, ORACLE at the primary budget) -> full Tier-1 development -> G2/G3.
set -euo pipefail
cd "$(dirname "$0")/.."
py="${SSLTANG_PYTHON:-.venv/bin/python}"
dev=(UTS01 UTS02 UTS03 UTS05 UTS06 UTS07 UTS08 UTS09 UTS10)
for s in "${dev[@]}"; do "$py" scripts/run_v4.py --stage g1 --subject "$s"; done
"$py" scripts/v4_gates.py --stage g1
primary=$("$py" -c "import json; print(json.load(open('results/v4/gates.json'))['G1']['primary'] or '')")
[[ -n "$primary" ]] || { echo "G1 failed at S3 and S6: stop (report ISP uninformative; regime L descriptive)"; exit 0; }
for s in "${dev[@]}"; do "$py" scripts/run_v4.py --stage develop --subject "$s" --clock dense --budgets "$primary" --arms W0 W1 ORACLE; done
"$py" scripts/v4_gates.py --stage feas
feas=$("$py" -c "import json; print(json.load(open('results/v4/gates.json'))['G_FEAS']['passed'])")
[[ "$feas" == "True" ]] || { echo "G-FEAS failed: stop"; exit 0; }
for s in "${dev[@]}"; do "$py" scripts/run_v4.py --stage develop --subject "$s" --clock dense; done
"$py" scripts/v4_gates.py --stage efficacy
for s in "${dev[@]}" UTS04; do "$py" scripts/run_v4.py --stage develop --subject "$s" --clock stride4; done
"$py" scripts/run_v4.py --stage develop --subject UTS04 --clock dense
echo "v4 ds003020 development complete"
