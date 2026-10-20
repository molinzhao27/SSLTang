#!/usr/bin/env python3
"""Tables for protocol v3 (structured voxel ridge) from saved cell JSONs.

Only measured values are reported; non-estimable cells are listed as such.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np

PRIMARY = 'session_macro_feature_correlation'
ARMS = ('V0', 'V1', 'V2', 'V3', 'V4')
FRACTIONS = ('1', '0.25', '0.1', '0.05', '0.01')


def load(root, subject, stage):
    cells = {}
    for path in sorted((Path(root) / subject / stage / 'cells').glob('*.json')):
        cell = json.loads(path.read_text())
        cells[(cell['arm'], f"{cell['fraction']:g}")] = cell
    return cells


def sign_flip(d):
    d = np.asarray(d, dtype=float)
    flips = np.array(list(itertools.product([-1, 1], repeat=len(d))))
    return float(np.mean(np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-15))


def paired(a, b, split='test'):
    runs = sorted(a[split]['by_run_feature_correlation'])
    d = np.array([a[split]['by_run_feature_correlation'][r] - b[split]['by_run_feature_correlation'][r] for r in runs])
    sa, sb = a[split]['by_session_feature_correlation'], b[split]['by_session_feature_correlation']
    return dict(delta=a[split][PRIMARY] - b[split][PRIMARY], sessions={k: sa[k] - sb[k] for k in sorted(sa)},
                stories_positive=int((d > 0).sum()), n_stories=len(d), sign_flip_p=sign_flip(d))


def fmt(x, digits=4):
    return '—' if x is None else f'{x:.{digits}f}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='results/v3')
    p.add_argument('--subjects', nargs='+', default=['UTS01', 'UTS02', 'UTS03'])
    p.add_argument('--lock', default='experiments/v3_lock.json')
    p.add_argument('--replication', nargs='*', default=['UTS09', 'UTS10'])
    a = p.parse_args()
    lock = json.loads(Path(a.lock).read_text())
    ev = {s: load(a.root, s, 'evaluate') for s in a.subjects}
    lines = ['# Protocol v3 results: structured voxel ridge (locked evaluation)', '',
             f"Advanced arm **{lock['advanced_arm']}**, locked {lock['created_at'][:19]}Z (code "
             f"`{lock['code_sha256'][:12]}`); see [PROTOCOL_V3.md](../../experiments/PROTOCOL_V3.md) §9. "
             'Test r is the session-macro feature correlation on the exposed UTS01–03 test stories. Every '
             'arm chose its kernel and alpha by session-grouped CV on that budget\'s training labels only; '
             'the 794 validation labels were used once, for the development gates.', '']
    lines += ['## Test r by arm and label budget', '',
              '| Subject | Labels | ' + ' | '.join(ARMS) + ' | V4 − V0 | V4 selection |',
              '|---|---:|' + '---:|' * len(ARMS) + '---:|---|']
    for f in FRACTIONS:
        for s in a.subjects:
            cells = ev[s]
            row = []
            for arm in ARMS:
                c = cells.get((arm, f))
                row.append(fmt(c['test'][PRIMARY]) if c and c.get('estimable') else 'n.e.')
            v4, v0 = cells.get(('V4', f)), cells.get(('V0', f))
            delta = (fmt(v4['test'][PRIMARY] - v0['test'][PRIMARY]) if v4 and v0 and v4.get('estimable')
                     and v0.get('estimable') else '—')
            sel = (f"{v4['selection']['label']}, α={v4['selection']['alpha']:.2g}" if v4 and v4.get('estimable') else
                   (v4 or {}).get('reason', '—'))
            labels = (v0 or v4 or {}).get('labels', {}).get('selected_examples', '—')
            lines.append(f'| {s} | {labels} | ' + ' | '.join(row) + f' | {delta} | {sel} |')
    lines += ['', '## V4 − V0, paired by story (test, 11 stories per subject)', '',
              '| Labels | Subject | Δ r | Per session | Stories V4 > V0 | Sign-flip p |', '|---:|---|---:|---|---:|---:|']
    for f in FRACTIONS:
        means = []
        for s in a.subjects:
            v4, v0 = ev[s].get(('V4', f)), ev[s].get(('V0', f))
            if not (v4 and v0 and v4.get('estimable') and v0.get('estimable')):
                continue
            r = paired(v4, v0)
            means.append(r['delta'])
            lines.append(f"| {f} | {s} | {r['delta']:+.4f} | " +
                         ', '.join(f'{k}: {v:+.4f}' for k, v in r['sessions'].items()) +
                         f" | {r['stories_positive']}/{r['n_stories']} | {r['sign_flip_p']:.3f} |")
        if means:
            lines.append(f'| {f} | **mean** | **{np.mean(means):+.4f}** | | | |')
    lines += ['', 'Stories share two sessions per subject and the same 11 stories recur across subjects; '
              'sign-flip p-values are descriptive.', '']
    lines += ['## Alignment and brain-input controls (100% labels)', '',
              '| Subject | V4 real | Shift-null q95 (p) | Disrupted training | Wrong-run input | '
              'Nuisance: position | Nuisance: oracle speech rate | Nuisance: both | Residualized V4 | Residualized V0 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in a.subjects:
        path = Path(a.root) / s / 'controls.json'
        if not path.exists():
            continue
        c = json.loads(path.read_text())
        null = ev[s][('V4', '1')]['test_shift_null']
        lines.append(f"| {s} | {c['real_test']:.4f} | {null['null_q95']:.4f} ({null['p_value']:.4f}) | "
                     f"{c['disrupted_training']['test']:.4f} | {c['wrong_run_input']['test']:.4f} | "
                     f"{c['nuisance_position']['test']:.4f} | {c['nuisance_speech_rate']['test']:.4f} | "
                     f"{c['nuisance_both']['test']:.4f} | {c['residualized_advanced']['test']:.4f} | "
                     f"{c['residualized_V0']['test']:.4f} |")
    lines += ['', 'Residualized decoders score test targets minus a training-fit nuisance prediction; the '
              'speech-rate features use transcripts (an oracle unavailable to a deployed decoder).', '']
    lines += ['## Supervision exposure per budget', '',
              'Words consumed by target construction: words in selected bins plus their 20-word causal contexts '
              '(unique instances), GPT-2 tokens of those instances, and the union of their timestamped spans. '
              'Values are UTS01 / UTS02 / UTS03. Below 100% the hashed label blocks differ by subject (the v2 '
              'subset hash includes subject-specific run IDs), so counts differ slightly; within a subject '
              'every arm shares the same subset.', '',
              '| Budget | Targets | Unique words | In-bin / context words (UTS01) | GPT-2 tokens | Transcript seconds | Stories | Sessions |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    exposures = {s: json.loads((Path(a.root) / s / 'exposure.json').read_text()) for s in a.subjects
                 if (Path(a.root) / s / 'exposure.json').exists()}
    if exposures:
        def entries(e):
            named = [(f'training {float(k):.0%}', v) for k, v in e['budgets'].items()]
            return named + [('validation (development exposure)', e['validation']), ('dense training (§7)', e['dense_training'])]
        per = {s: entries(e) for s, e in exposures.items()}
        first = next(iter(per.values()))
        for i, (name, summary) in enumerate(first):
            join = lambda k, f='{}': ' / '.join(f.format(per[s][i][1][k]) for s in per)
            lines.append(f"| {name} | {summary['selected_targets']} | {join('unique_word_instances')} | "
                         f"{summary['in_bin_word_instances']} / {summary['context_only_word_instances']} | "
                         f"{join('unique_gpt2_tokens')} | {join('duration_seconds', '{:.0f}')} | "
                         f"{join('runs_touched')} | {join('sessions_touched')} |")
    dense = {s: {arm: json.loads(p.read_text()) for arm in ('V0', lock['advanced_arm'])
                 for p in [Path(a.root) / s / 'dense' / f'{arm}_dense.json'] if p.exists()} for s in a.subjects}
    if any(dense.values()):
        lines += ['', '## Dense training clock (§7; a changed data budget)', '',
                  '| Subject | Training targets (dense) | V0 old | V0 dense | V4 old | V4 dense |', '|---|---:|---:|---:|---:|---:|']
        for s in a.subjects:
            if len(dense[s]) < 2:
                continue
            d0, d4 = dense[s]['V0'], dense[s][lock['advanced_arm']]
            lines.append(f"| {s} | {d0['n_train_targets']} | {ev[s][('V0', '1')]['test'][PRIMARY]:.4f} | "
                         f"{d0['test'][PRIMARY]:.4f} | {ev[s][('V4', '1')]['test'][PRIMARY]:.4f} | {d4['test'][PRIMARY]:.4f} |")
    fresh = [s for s in a.replication if (Path(a.root) / s / 'evaluate' / 'cells').exists()]
    if fresh:
        lines += ['', '## Fresh-recording replication (§11; registered before download)', '',
                  'UTS09 and UTS10 were never used before this replication. The locked v3 procedure ran '
                  'unchanged, with the §4 fold rule applied to each subject\'s own training sessions. A contrast '
                  'replicates when its test difference has the predicted (positive) sign in both subjects.', '',
                  '| Contrast | ' + ' | '.join(fresh) + ' | Replicated |', '|---|' + '---:|' * len(fresh) + '---|']
        rows = {}
        for s in fresh:
            cells = load(a.root, s, 'evaluate')
            d0p = Path(a.root) / s / 'dense' / 'V0_dense.json'
            d4p = Path(a.root) / s / 'dense' / f"{lock['advanced_arm']}_dense.json"
            v0, v4 = cells.get(('V0', '1')), cells.get(('V4', '1'))
            r = {}
            if v0 and v4 and v0.get('estimable') and v4.get('estimable'):
                pr = paired(v4, v0)
                r['R2'] = (pr['delta'], f"{pr['stories_positive']}/{pr['n_stories']}", pr['sign_flip_p'])
            if d0p.exists() and v0:
                d0 = json.loads(d0p.read_text())
                r['R1'] = (d0['test'][PRIMARY] - v0['test'][PRIMARY], None, None)
                if d4p.exists():
                    d4 = json.loads(d4p.read_text())
                    r['R3'] = (d4['test'][PRIMARY] - d0['test'][PRIMARY], None, None)
            rows[s] = r
        names = {'R1': 'R1 dense-clock V0 − stride-4 V0', 'R2': 'R2 V4 − V0 (stride-4, 100%)', 'R3': 'R3 V4 − V0 (dense clock)'}
        for key, name in names.items():
            values = [rows[s].get(key) for s in fresh]
            cells_txt = []
            for v in values:
                if v is None:
                    cells_txt.append('—')
                else:
                    extra = f' ({v[1]} stories, p={v[2]:.3f})' if v[1] else ''
                    cells_txt.append(f'{v[0]:+.4f}{extra}')
            done = all(v is not None for v in values) and len(values) == len(a.replication)
            verdict = ('yes' if all(v[0] > 0 for v in values) else 'no') if done else 'pending'
            lines.append(f'| {name} | ' + ' | '.join(cells_txt) + f' | {verdict} |')
        lines += ['', '| Subject | Labels | ' + ' | '.join(ARMS) + ' | V0 dense | V4 dense |',
                  '|---|---:|' + '---:|' * len(ARMS) + '---:|---:|']
        for s in fresh:
            cells = load(a.root, s, 'evaluate')
            for f in FRACTIONS:
                row = [fmt(cells[(arm, f)]['test'][PRIMARY]) if cells.get((arm, f), {}).get('estimable') else 'n.e.'
                       for arm in ARMS]
                dense_txt = ['—', '—']
                if f == '1':
                    for k, arm in enumerate(('V0', lock['advanced_arm'])):
                        path = Path(a.root) / s / 'dense' / f'{arm}_dense.json'
                        if path.exists():
                            dense_txt[k] = fmt(json.loads(path.read_text())['test'][PRIMARY])
                labels = (cells.get(('V0', f)) or {}).get('labels', {}).get('selected_examples', '—')
                lines.append(f'| {s} | {labels} | ' + ' | '.join(row) + f' | {dense_txt[0]} | {dense_txt[1]} |')
    transfer = {s: json.loads((Path(a.root) / 'transfer' / s / 'results.json').read_text()) for s in a.subjects
                if (Path(a.root) / 'transfer' / s / 'results.json').exists()}
    if transfer:
        lines += ['', '## Exploratory modality transfer with dense-clock decoders (§7.1)', '',
                  'ds004510 was scored in v2, so these are exploratory. The validation pseudo-split gate reproduced the '
                  'standard predictions before scoring. Centered narrative-macro top-1 (exact permutation p); '
                  'imagined speech is 5-way (chance 20%), films 3-way (chance 33%, minimum p = 1/6).', '',
                  '| Subject | Decoder | Consistency (rel. RMS) | Imagined top-1 (p) | Imagined MRR | Films top-1 (p) | v2 pooled imagined top-1 (p) |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for s, r in transfer.items():
            v2_path = Path('results/v2/transfer') / s / 'results.json'
            old = None
            if v2_path.exists():
                c = json.loads(v2_path.read_text())['conditions']['imagined_speech']['training_mean_centered']
                old = f"{c['narrative_macro']['top1']:.0%} ({c['permutation']['metrics']['top1']['p_greater_equal']:.3f})"
            for arm, d in r['decoders'].items():
                cells = []
                for cond in ('imagined_speech', 'perceived_movie'):
                    c = d['conditions'][cond]['training_mean_centered']
                    cells.append((c['narrative_macro'], c['permutation']['metrics']))
                (im, imp), (mv, mvp) = cells
                check = r['consistency_check'][arm]['prediction_relative_rms_error']
                lines.append(f"| {s} | {arm} dense | {check:.0e} | {im['top1']:.0%} ({imp['top1']['p_greater_equal']:.3f}) | "
                             f"{im['mrr']:.3f} | {mv['top1']:.0%} ({mvp['top1']['p_greater_equal']:.3f}) | {old or '—'} |")
    out = Path(a.root) / 'RESULTS.md'
    out.write_text('\n'.join(lines) + '\n')
    print(out)


if __name__ == '__main__':
    main()
