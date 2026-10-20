#!/usr/bin/env python3
"""PROTOCOL_V4 development gates on ds003020 (FUTURE_WORK_V4 §7), computed mechanically from saved cells.

  G1      prior validity (training only): median over the 9 development subjects of Spearman(ISP at S3, full-label
          W1 map) >= 0.20 and > 0.10 in at least 7/9; else the same at S6 moves the primary budget to S6 (LPP L5);
          else stop.
  G-FEAS  validation, dense, primary budget: mean(ORACLE - W0) >= +0.005.
  G2      validation, dense, primary budget: mean(EB - W1) >= +0.005, mean(EB - W0) >= +0.003, EB - W0 > 0 in >= 7/9.
  G3      validation, dense, ALL: mean(EB - W1) >= -0.003 and no subject below -0.010.
Advancement follows the registered table (EB; EB without H6; PRIOR by substitution; or nothing).
Every cell must carry the current run_v4 code fingerprint. Writes results/v4/gates.json.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v4

OUT = Path('results/v4')
VAL = 'session_macro_feature_correlation'


def cell(subject, budget, arm, clock='dense'):
    path = OUT / subject / clock / 'develop' / 'cells' / f'{budget}_{arm}.json'
    c = json.loads(path.read_text())
    if c.get('code_sha256') != run_v4.code_sha256():
        raise SystemExit(f'{path} is not from the current code')
    if not c.get('estimable') or 'val' not in c:
        raise SystemExit(f'{path} is not estimable; the registered gate cannot be computed')
    return c['val'][VAL]


def g1(subjects):
    out = {}
    for name in ('S3', 'S6'):
        rho = []
        for s in subjects:
            g = json.loads((OUT / s / 'g1.json').read_text())
            if g.get('code_sha256') != run_v4.code_sha256():
                raise SystemExit(f'{s} g1.json is not from the current code')
            rho.append(g['budgets'][name]['spearman'])
        rho = np.array(rho)
        out[name] = dict(spearman=rho.round(4).tolist(), median=float(np.median(rho)), above_010=int((rho > 0.10).sum()),
                         passed=bool(np.median(rho) >= 0.20 and (rho > 0.10).sum() >= 7))
    primary = 'S3' if out['S3']['passed'] else 'S6' if out['S6']['passed'] else None
    return dict(out, primary=primary, lpp_primary={'S3': 'L3', 'S6': 'L5'}.get(primary),
                decision=('primary budget ' + primary) if primary else 'stop: ISP is uninformative at matched scan time')


def deltas(subjects, budget, a, b):
    return np.array([cell(s, budget, a) - cell(s, budget, b) for s in subjects])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['g1', 'feas', 'efficacy'], required=True)
    p.add_argument('--subjects', nargs='+', default=list(run_v4.DEV_SUBJECTS))
    a = p.parse_args()
    if sorted(a.subjects) != sorted(run_v4.DEV_SUBJECTS):
        raise SystemExit(f'The gates use exactly the 9 development subjects {run_v4.DEV_SUBJECTS}')
    path = OUT / 'gates.json'
    gates = json.loads(path.read_text()) if path.exists() else {}
    gates['code_sha256'] = run_v4.code_sha256()
    if a.stage == 'g1':
        gates['G1'] = g1(a.subjects)
    else:
        primary = gates.get('G1', {}).get('primary')
        if primary is None:
            raise SystemExit('G1 has not set a primary budget')
        if a.stage == 'feas':
            d = deltas(a.subjects, primary, 'ORACLE', 'W0')
            gates['G_FEAS'] = dict(budget=primary, delta=d.round(5).tolist(), mean=float(d.mean()),
                                   passed=bool(d.mean() >= 0.005))
        else:
            res = {}
            for arm in ('EB', 'PRIOR'):
                d1, d2 = deltas(a.subjects, primary, arm, 'W1'), deltas(a.subjects, primary, arm, 'W0')
                res[arm] = dict(minus_w1=d1.round(5).tolist(), mean_minus_w1=float(d1.mean()),
                                minus_w0=d2.round(5).tolist(), mean_minus_w0=float(d2.mean()),
                                positive_vs_w0=int((d2 > 0).sum()),
                                passed=bool(d1.mean() >= 0.005 and d2.mean() >= 0.003 and (d2 > 0).sum() >= 7))
            d3 = deltas(a.subjects, 'ALL', 'EB', 'W1')
            g3 = dict(minus_w1=d3.round(5).tolist(), mean=float(d3.mean()), worst=float(d3.min()),
                      passed=bool(d3.mean() >= -0.003 and d3.min() >= -0.010))
            if res['EB']['passed']:
                advanced, drop_h6 = 'EB', not g3['passed']
            elif res['PRIOR']['passed']:
                advanced, drop_h6 = 'PRIOR', True
            else:
                advanced, drop_h6 = None, None
            gates.update(G2=res, G3=g3, advanced=advanced, drop_h6=drop_h6,
                         decision=('advance ' + advanced + (' (H6 dropped)' if drop_h6 else '')) if advanced else
                         'development null: CONFIRM stays untouched; ds003020 test reported descriptively')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gates, indent=1) + '\n')
    print(json.dumps(gates, indent=1))


if __name__ == '__main__':
    main()
