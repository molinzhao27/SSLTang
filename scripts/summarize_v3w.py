#!/usr/bin/env python3
"""Tables for PROTOCOL_V3 §12 (encoding-predictability voxel weighting) from saved cells.

Only measured values are reported. W0 is checked against the v3 V0 cells it must reproduce.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_v3 import PRIMARY, fmt, paired

ARMS = ('W0', 'W1', 'W2')
FRACTIONS = ('1', '0.25', '0.1', '0.05', '0.01')


def load(root, subject, stage):
    cells = {}
    for path in sorted((Path(root) / subject / stage / 'cells').glob('*.json')):
        cell = json.loads(path.read_text())
        cells[(cell['arm'], f"{cell['fraction']:g}")] = cell
    return cells


def value(cell, split):
    return cell[split][PRIMARY] if cell and cell.get(split) else None


def develop_section(root, v3_root, subjects):
    lines = ['## Development (validation, stride-4 clock)', '',
             '| Subject | Labels | Arm | Rule | alpha | CV r | Val r | Δ vs W0 |', '|---|---:|---|---|---:|---:|---:|---:|']
    checks = ['', '**W0 reproduces v3 V0** (same standardization and all-voxel kernel):', '',
              '| Subject | Labels | W0 alpha | V0 alpha | W0 val | V0 val | Δ |', '|---|---:|---:|---:|---:|---:|---:|']
    weights = ['', '**Encoding predictability in the final development fit** (all labeled training rows):', '',
               '| Subject | Labels | Encoding alpha (d=1,2,3) | Columns with r > 0 (d=1,2,3) | Selected-rule nonzero |',
               '|---|---:|---|---|---|']
    for s in subjects:
        cells = load(root, s, 'develop')
        v3 = load(v3_root, s, 'develop')
        for b in ('1', '0.1'):
            base = value(cells.get(('W0', b)), 'val')
            for arm in ARMS:
                cell = cells.get((arm, b))
                if not cell:
                    continue
                if not cell.get('selection'):
                    lines.append(f'| {s} | {float(b):.0%} | {arm} | not estimable | | | | |')
                    continue
                sel, val = cell['selection'], value(cell, 'val')
                delta = '' if arm == 'W0' else f'{val - base:+.4f}'
                lines.append(f"| {s} | {float(b):.0%} | {arm} | {sel['label']} | {sel['alpha']:.3g} | "
                             f"{sel['cv_score']:.4f} | {val:.4f} | {delta} |")
            v4 = v3.get(('V4', b))
            if v4 and v4.get('selection') and base is not None:
                lines.append(f"| {s} | {float(b):.0%} | V4 (v3, reference) | {v4['selection']['label']} | "
                             f"{v4['selection']['alpha']:.3g} | {v4['selection']['cv_score']:.4f} | "
                             f"{value(v4, 'val'):.4f} | {value(v4, 'val') - base:+.4f} |")
            w0, v0 = cells.get(('W0', b)), v3.get(('V0', b))
            if w0 and v0 and w0.get('selection') and v0.get('selection'):
                checks.append(f"| {s} | {float(b):.0%} | {w0['selection']['alpha']:.3g} | {v0['selection']['alpha']:.3g} | "
                              f"{value(w0, 'val'):.6f} | {value(v0, 'val'):.6f} | "
                              f"{value(w0, 'val') - value(v0, 'val'):+.1e} |")
            info = (cells.get(('W1', b)) or {}).get('final_fit_weights') or {}
            if info:
                alphas = ', '.join(fmt(info['encoding_alpha'][d], 3) if info['encoding_alpha'][d] else '—'
                                   for d in sorted(info['encoding_alpha']))
                positive = ', '.join(f"{info['positive_fraction'][d]:.0%}" for d in sorted(info['positive_fraction']))
                chosen = []
                for arm in ('W1', 'W2'):
                    sel = (cells.get((arm, b)) or {}).get('selection')
                    if sel:
                        nz = info['weights'][sel['label']]['nonzero_fraction']
                        chosen.append(f"{arm} {sel['label']}: {nz:.0%}")
                weights.append(f"| {s} | {float(b):.0%} | {alphas} | {positive} | {'; '.join(chosen)} |")
    return lines + checks + weights


def gate_section(lock_path):
    if not Path(lock_path).exists():
        return ['## Gates', '', 'Not yet applied.']
    lock = json.loads(Path(lock_path).read_text())
    lines = ['## Gates (§12, fixed before implementation)', '',
             '| Arm | Δ val at 100% (per subject) | Mean | Δ val at 10% (per subject) | Mean | Passed |',
             '|---|---|---:|---|---:|---|']
    for arm, g in lock['gates'].items():
        lines.append(f"| {arm} | {', '.join(f'{x:+.4f}' for x in g['delta_100'])} | {g['mean_100']:+.4f} | "
                     f"{', '.join(f'{x:+.4f}' for x in g['delta_10'])} | {g['mean_10']:+.4f} | "
                     f"{'yes' if g['passed'] else 'no'} |")
    lines += ['', f"**Decision:** {lock['decision']}. Code fingerprint `{lock['code_sha256'][:12]}`."]
    return lines


def evaluate_section(root, v3_root, subjects, advanced):
    lines = ['## Locked evaluation (test; exposed benchmark)', '',
             '| Subject | Labels | W0 | ' + f'{advanced} (rule) | Δ | Stories + | Sign-flip p | v3 V4 |',
             '|---|---:|---:|---|---:|---:|---:|---:|']
    for s in subjects:
        cells, v3 = load(root, s, 'evaluate'), load(v3_root, s, 'evaluate')
        for b in FRACTIONS:
            w0, adv = cells.get(('W0', b)), cells.get((advanced, b))
            if w0 and w0.get('test') and adv and not adv.get('estimable'):
                lines.append(f"| {s} | {float(b):.0%} | {value(w0, 'test'):.4f} | not estimable | | | | "
                             f"{fmt(value(v3.get(('V4', b)), 'test'))} |")
                continue
            if not (w0 and adv and w0.get('test') and adv.get('test')):
                continue
            p = paired(adv, w0)
            lines.append(f"| {s} | {float(b):.0%} | {value(w0, 'test'):.4f} | {value(adv, 'test'):.4f} "
                         f"({adv['selection']['label']}) | {p['delta']:+.4f} | {p['stories_positive']}/{p['n_stories']} | "
                         f"{p['sign_flip_p']:.3f} | {fmt(value(v3.get(('V4', b)), 'test'))} |")
    res = ['', '**Residualized test endpoint, 100%** (targets minus training-fit position + oracle speech rate):', '',
           '| Subject | W0 | ' + advanced + ' | Δ |', '|---|---:|---:|---:|']
    dense = ['', '**Dense training clock** (rule fixed from the stride-4 100% fit, alpha re-selected):', '',
             '| Subject | W0 val | W0 test | ' + f'{advanced} val | {advanced} test | Δ test |', '|---|---:|---:|---:|---:|---:|']
    for s in subjects:
        path = Path(root) / s / 'evaluate' / 'residualized.json'
        if path.exists():
            v = json.loads(path.read_text())['test']
            res.append(f"| {s} | {v['W0']:.4f} | {v[advanced]:.4f} | {v[advanced] - v['W0']:+.4f} |")
        cells = {a: Path(root) / s / 'dense' / f'{a}_dense.json' for a in ('W0', advanced)}
        if all(p.exists() for p in cells.values()):
            w0, adv = (json.loads(cells[a].read_text()) for a in ('W0', advanced))
            dense.append(f"| {s} | {value(w0, 'val'):.4f} | {value(w0, 'test'):.4f} | {value(adv, 'val'):.4f} | "
                         f"{value(adv, 'test'):.4f} | {value(adv, 'test') - value(w0, 'test'):+.4f} |")
    controls = ['', '**Brain-input controls, 100%** (shift null registered; the other two exploratory, after the lock):', '',
                f'| Subject | {advanced} test | Shift-null q95 (p) | Disrupted training | Wrong-run input | '
                'Oracle speech-rate model (v3) |', '|---|---:|---:|---:|---:|---:|']
    for s in subjects:
        adv = load(root, s, 'evaluate').get((advanced, '1'))
        if not (adv and adv.get('test')):
            continue
        null = adv.get('test_shift_null', {})
        path, v3c = Path(root) / s / 'controls.json', Path(v3_root) / s / 'controls.json'
        c = json.loads(path.read_text()) if path.exists() else {}
        oracle = json.loads(v3c.read_text()).get('nuisance_both', {}).get('test') if v3c.exists() else None
        controls.append(f"| {s} | {value(adv, 'test'):.4f} | {fmt(null.get('null_q95'))} ({fmt(null.get('p_value'))}) | "
                        f"{fmt(c.get('disrupted_training', {}).get('test'))} | "
                        f"{fmt(c.get('wrong_run_input', {}).get('test'))} | {fmt(oracle)} |")
    return lines + res + dense + controls


def fresh_section(root, subjects, advanced):
    """Registered confirmatory contrast (§12): advanced arm − W0, stride-4, 100%, test; >= 4 of 5 positive."""
    from math import comb
    lines = ['## Fresh confirmatory replication (UTS04–UTS08; registered before download)', '',
             f'| Subject | W0 test | {advanced} test (rule) | Δ | Stories + | Sign-flip p | Residualized Δ | Dense Δ |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    deltas = []
    for s in subjects:
        cells = load(root, s, 'evaluate')
        w0, adv = cells.get(('W0', '1')), cells.get((advanced, '1'))
        if not (w0 and adv and w0.get('test') and adv.get('test')):
            lines.append(f'| {s} | missing | | | | | | |')
            continue
        p = paired(adv, w0)
        deltas.append(p['delta'])
        res_path = Path(root) / s / 'evaluate' / 'residualized.json'
        res = json.loads(res_path.read_text())['test'] if res_path.exists() else None
        dense = {a: Path(root) / s / 'dense' / f'{a}_dense.json' for a in ('W0', advanced)}
        dense_delta = None
        if all(q.exists() for q in dense.values()):
            d0, d1 = (json.loads(dense[a].read_text()) for a in ('W0', advanced))
            dense_delta = value(d1, 'test') - value(d0, 'test')
        res_text = '—' if res is None else f'{res[advanced] - res["W0"]:+.4f}'
        dense_text = '—' if dense_delta is None else f'{dense_delta:+.4f}'
        lines.append(f"| {s} | {value(w0, 'test'):.4f} | {value(adv, 'test'):.4f} ({adv['selection']['label']}) | "
                     f"{p['delta']:+.4f} | {p['stories_positive']}/{p['n_stories']} | {p['sign_flip_p']:.3f} | "
                     f"{res_text} | {dense_text} |")
    k, n = sum(d > 0 for d in deltas), len(deltas)
    p_sign = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else float('nan')
    verdict = 'confirmed' if n == len(subjects) and k >= 4 else 'not confirmed'
    lines += ['', f'**Registered decision:** positive in {k}/{n} subjects (rule: at least 4 of 5) → **{verdict}**. '
              f'Exact one-sided sign test p = {p_sign:.3f}.']
    return lines


def sweep_section(root, subjects):
    """Exploratory, validation-only sweep beyond the registered grid (100%, stride-4)."""
    found = {s: Path(root) / s / 'explore_grid.json' for s in subjects}
    found = {s: json.loads(p.read_text()) for s, p in found.items() if p.exists()}
    if not found:
        return []
    labels = list(next(iter(found.values()))['rules'])
    lines = ['## Exploratory beyond-grid sweep (validation only; 100%, stride-4; not registered)', '',
             '| Rule | ' + ' | '.join(f'{s} CV | {s} val' for s in found) + ' |',
             '|---|' + '---:|---:|' * len(found)]
    for label in labels:
        cells = []
        for s, data in found.items():
            rule = data['rules'].get(label)
            cells += [fmt(rule and rule['cv_score']), fmt(rule and rule['val'])]
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='results/v3w')
    p.add_argument('--v3-root', default='results/v3')
    p.add_argument('--lock', default='experiments/v3w_lock.json')
    p.add_argument('--subjects', nargs='+', default=['UTS01', 'UTS02', 'UTS03'])
    p.add_argument('--exposed', nargs='+', default=['UTS01', 'UTS02', 'UTS03', 'UTS09', 'UTS10'])
    p.add_argument('--fresh', nargs='+', default=['UTS04', 'UTS05', 'UTS06', 'UTS07', 'UTS08'])
    p.add_argument('--out', default='results/v3w/RESULTS.md')
    a = p.parse_args()
    lines = ['# Protocol v3 §12 results: encoding-predictability voxel weighting', '',
             'Generated by `scripts/summarize_v3w.py` from saved cells. Endpoint: session-macro feature correlation.',
             '']
    lines += develop_section(a.root, a.v3_root, a.subjects) + [''] + gate_section(a.lock)
    lock = json.loads(Path(a.lock).read_text()) if Path(a.lock).exists() else {}
    if lock.get('advanced_arm'):
        lines += [''] + evaluate_section(a.root, a.v3_root, a.exposed, lock['advanced_arm'])
        lines += [''] + fresh_section(a.root, a.fresh, lock['advanced_arm'])
    lines += [''] + sweep_section(a.root, a.subjects)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
