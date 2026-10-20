#!/usr/bin/env python3
"""Tables and paired story-level contrasts for v2 ridge and fine-tuning cells.

Values are measured scores read from cell JSONs; nothing is simulated or
filled in. Seeds are averaged only after per-seed scores are recorded.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

PRIMARY = 'session_macro_feature_correlation'


def load_ridge(root):
    rows = []
    for path in sorted(Path(root).glob('cells/*.json')):
        cell = json.loads(path.read_text())
        if cell['arm'] == 'B0':
            continue
        rows.append(dict(
            arm=cell['arm'], features=cell['features'], seed=cell['seed'], cutoff=cell['cutoff_tr'],
            readout=cell['readout'], fraction=cell['fraction'], labels=cell['labels']['selected_examples'],
            val=cell['val'][PRIMARY], test=cell['test'][PRIMARY],
            test_run_macro=cell['test']['run_macro_feature_correlation'],
            retrieval10=cell['test']['centered_retrieval']['10']['session_macro_rank_percentile'],
            retrieval32=cell['test']['centered_retrieval']['32']['session_macro_rank_percentile'],
            null_mean=cell['test_shift_null']['null_mean'], null_q95=cell['test_shift_null']['null_q95'],
            null_p=cell['test_shift_null']['p_value'], option=cell['selection']['option'],
            alpha=cell['selection']['alpha'], edge=cell['selection']['alpha_at_grid_edge'],
            by_run=cell['test']['by_run_feature_correlation'],
            sessions=cell['test']['by_session_feature_correlation'], source=str(path)))
    return rows


def load_finetuned(root, arm='B5'):
    rows = []
    for path in sorted(Path(root).glob('*/evaluation_v2/metrics.json')):
        m = json.loads(path.read_text())
        labels = json.loads((path.parents[1]/'label_subset.json').read_text())['label_metadata']
        name = path.parents[1].name
        rows.append(dict(
            arm=arm, features=name.split('_c')[0], seed=m['seed'], cutoff=m['cutoff_tr'], readout='finetune',
            fraction=float(m['fraction']), labels=labels['selected_examples'], val=m['val'][PRIMARY],
            test=m['test'][PRIMARY], test_run_macro=m['test']['run_macro_feature_correlation'],
            retrieval10=m['test']['centered_retrieval']['10']['session_macro_rank_percentile'],
            retrieval32=m['test']['centered_retrieval']['32']['session_macro_rank_percentile'],
            null_mean=m['test_shift_null']['null_mean'], null_q95=m['test_shift_null']['null_q95'],
            null_p=m['test_shift_null']['p_value'], option=f"step {m['global_step']}", alpha=None, edge=False,
            by_run=m['test']['by_run_feature_correlation'],
            sessions=m['test']['by_session_feature_correlation'], source=str(path)))
    return rows


def grouped(df):
    keys = ['arm', 'features', 'cutoff', 'readout', 'fraction']
    agg = df.groupby(keys).agg(
        seeds=('seed', 'nunique'), labels=('labels', 'first'), val=('val', 'mean'),
        test=('test', 'mean'), test_min=('test', 'min'), test_max=('test', 'max'),
        test_run_macro=('test_run_macro', 'mean'), retrieval10=('retrieval10', 'mean'),
        retrieval32=('retrieval32', 'mean'), null_q95=('null_q95', 'mean'), null_p_max=('null_p', 'max'),
        options=('option', lambda v: ', '.join(sorted(set(map(str, v))))),
        edges=('edge', 'sum')).reset_index()
    return agg


def sign_flip(differences):
    """Exact two-sided sign-flip p-value for the mean of story differences."""
    d = np.asarray(differences, dtype=float)
    observed = abs(d.mean())
    flips = np.array(list(itertools.product([-1, 1], repeat=len(d))))
    null = np.abs((flips * d).mean(1))
    return float(np.mean(null >= observed - 1e-15))


def paired(df, a, b, sessions):
    """Story-level contrast a − b; filters are dicts of column values.

    Seeds are paired when both arms have them; a deterministic arm (one fit)
    is compared with every seed of the other. Story differences are averaged
    over seeds before the sign-flip test (stories treated as exchangeable).
    """
    left = df[np.logical_and.reduce([df[k] == v for k, v in a.items()])]
    right = df[np.logical_and.reduce([df[k] == v for k, v in b.items()])]
    if left.empty or right.empty:
        return None
    for side in (left, right):
        if side.seed.duplicated().any():
            raise ValueError(f'More than one row per seed for {a} or {b}')
    # Seed 0 marks a deterministic fit, compared with every seed of the other arm.
    deterministic = [set(side.seed) == {0} for side in (left, right)]
    if not any(deterministic) and set(left.seed) != set(right.seed):
        raise ValueError(f'Unpaired seeds {sorted(set(left.seed) ^ set(right.seed))} for {a} vs {b}')
    pairs = []
    for _, x in left.iterrows():
        match = right if any(deterministic) else right[right.seed == x.seed]
        for _, y in match.iterrows():
            pairs.append({run: x.by_run[run] - y.by_run[run] for run in x.by_run})
    if not pairs:
        return None
    stories = sorted(pairs[0])
    story_diff = np.array([np.mean([p[s] for p in pairs]) for s in stories])
    by_session = {}
    for story, value in zip(stories, story_diff):
        by_session.setdefault(sessions[story], []).append(value)
    per_seed = [float(np.mean([np.mean([p[s] for s in stories if sessions[s] == ses])
                               for ses in sorted(by_session)])) for p in pairs]
    return dict(n_pairs=len(pairs), session_macro_delta=float(np.mean([np.mean(v) for v in by_session.values()])),
                per_pair_delta=per_seed, per_session_delta={k: float(np.mean(v)) for k, v in sorted(by_session.items())},
                stories_positive=int((story_diff > 0).sum()), n_stories=len(stories),
                story_sign_flip_p=sign_flip(story_diff))


def fmt(value, digits=4):
    return '—' if value is None or (isinstance(value, float) and not np.isfinite(value)) else f'{value:.{digits}f}'


def test_sessions(root):
    """run_id -> session for the held-out test stories, from the v2 target IDs."""
    provenance = json.loads((Path(root)/'provenance.json').read_text())
    if 'sessions' in provenance:
        return provenance['sessions']['test']
    manifest = json.loads(Path(provenance['config']['data']['manifest']).read_text()) \
        if Path(provenance['config']['data']['manifest']).exists() else None
    if manifest:
        return {r['run_id']: r.get('session') for r in manifest['runs'] if r['split'] == 'test'}
    archived = sorted(Path('results/runs').glob('*/evaluation_test/metrics.json'))
    return json.loads(archived[0].read_text())['sessions']


def table(rows, columns):
    lines = ['| ' + ' | '.join(name for name, _ in columns) + ' |',
             '|' + '|'.join('---' if i == 0 else '---:' for i in range(len(columns))) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(render(row) for _, render in columns) + ' |')
    return lines


LABELS = {('B1', 'bold'): 'B1 raw BOLD', ('B2', 'window_pca'): 'B2 window PCA',
          ('B3', 'random'): 'B3 frozen random encoder', ('B4', 'ssl_best'): 'B4 frozen recon encoder (best val)',
          ('B4', 'ssl_final'): 'B4 frozen recon encoder (final)', ('B5', 'random'): 'B5 fine-tuned, random init',
          ('B5', 'ssl_best'): 'B5 fine-tuned, recon init', ('R', 'recon_final'): 'R frozen recon (v2 retrain)',
          ('L', 'latent_final'): 'L frozen latent (EMA teacher)', ('B5', 'recon'): 'fine-tuned, v2 recon init',
          ('B5', 'latent'): 'fine-tuned, latent init'}


def label(row):
    return LABELS.get((row['arm'], row['features']), f"{row['arm']} {row['features']}")


def score_range(row):
    if row['seeds'] > 1:
        return f"{row['test']:.4f} [{row['test_min']:.4f}, {row['test_max']:.4f}]"
    return f"{row['test']:.4f}"


MAIN_COLUMNS = [('Arm', label), ('Readout', lambda r: r['readout']), ('Fits', lambda r: str(r['seeds'])),
                ('Val r', lambda r: fmt(r['val'])), ('Test r (seed range)', score_range),
                ('Test run-macro r', lambda r: fmt(r['test_run_macro'])),
                ('Retrieval pct ±10 / ±32 TR', lambda r: f"{fmt(r['retrieval10'], 3)} / {fmt(r['retrieval32'], 3)}"),
                ('Shift-null q95', lambda r: fmt(r['null_q95'])), ('Selected', lambda r: r['options'])]


def contrast_lines(df, specs, sessions):
    lines = table([], [('Contrast (a − b)', None), ('Labels', None), ('Pairs', None), ('Δ test r', None),
                       ('Per pair', None), ('Per session', None), ('Stories a > b', None), ('Sign-flip p', None)])
    for name, fraction, a, b in specs:
        result = paired(df[df.fraction == fraction], a, b, sessions)
        if result is None:
            continue
        lines.append(f"| {name} | {fraction:g} | {result['n_pairs']} | {result['session_macro_delta']:+.4f} | "
                     + ', '.join(f'{v:+.4f}' for v in result['per_pair_delta']) + ' | '
                     + ', '.join(f'{k}: {v:+.4f}' for k, v in result['per_session_delta'].items())
                     + f" | {result['stories_positive']}/{result['n_stories']} | {result['story_sign_flip_p']:.3f} |")
    return lines


def phase_a_report(df, agg, sessions, out):
    lines = ['# Protocol v2 Phase A: measured readout baselines (UTS01)', '',
             'Exploratory benchmark extension on already-examined test stories; see '
             '[PROTOCOL_V2.md](../../../experiments/PROTOCOL_V2.md). Test r is the session-macro '
             'feature correlation; validation selected every option and alpha. Seed ranges are '
             'observed minima/maxima, not confidence intervals. B1/B2 are deterministic single fits. '
             'Every arm shares 4,518/794/802 train/val/test targets; 794 validation labels are reused '
             'at every budget.', '']
    for cutoff in (3, 4):
        for fraction in (1.0, 0.1):
            rows = agg[(agg.cutoff == cutoff) & (agg.fraction == fraction)].sort_values('val', ascending=False)
            lines += [f'## {2*cutoff}-second cutoff, {fraction:.0%} labels '
                      f'({int(rows.labels.iloc[0])} training targets)', '']
            lines += table([r for _, r in rows.iterrows()], MAIN_COLUMNS) + ['']
    lines += ['## Label budgets (6-second cutoff; test r, mean over fits)', '']
    budget = agg[(agg.cutoff == 3) & agg.readout.isin(['fir', 'window', 'finetune'])]
    fractions = sorted(budget.fraction.unique(), reverse=True)
    rows = []
    for (arm, features, readout), group in budget.groupby(['arm', 'features', 'readout']):
        values = {f: group[group.fraction == f].test.mean() for f in fractions}
        rows.append(dict(arm=arm, features=features, readout=readout, **{str(f): values[f] for f in fractions}))
    lines += table(rows, [('Arm', label), ('Readout', lambda r: r['readout'])] +
                   [(f'{f:.0%}', (lambda f: lambda r: fmt(r[str(f)]))(f)) for f in fractions]) + ['']
    specs = []
    for fraction in (1.0, 0.1):
        fir = dict(cutoff=3, readout='fir')
        specs += [('Frozen recon (final) − frozen random', fraction, dict(arm='B4', features='ssl_final', **fir),
                   dict(arm='B3', features='random', **fir)),
                  ('Frozen recon (final) − raw BOLD', fraction, dict(arm='B4', features='ssl_final', **fir),
                   dict(arm='B1', features='bold', **fir)),
                  ('Frozen random − raw BOLD', fraction, dict(arm='B3', features='random', **fir),
                   dict(arm='B1', features='bold', **fir)),
                  ('Fine-tuned recon init − random init', fraction,
                   dict(arm='B5', features='ssl_best', cutoff=3), dict(arm='B5', features='random', cutoff=3)),
                  ('Raw BOLD − fine-tuned recon init', fraction, dict(arm='B1', features='bold', **fir),
                   dict(arm='B5', features='ssl_best', cutoff=3)),
                  ('Raw BOLD FIR − endpoint', fraction, dict(arm='B1', features='bold', **fir),
                   dict(arm='B1', features='bold', cutoff=3, readout='endpoint'))]
    lines += ['## Paired story-level contrasts (6-second cutoff)', '',
              'Story differences are averaged over paired seeds before an exact sign-flip test over '
              'the 11 test stories. Stories share two sessions, so p-values are descriptive.', '']
    lines += contrast_lines(df, specs, sessions) + ['']
    (Path(out)/'RESULTS.md').write_text('\n'.join(lines) + '\n')


def ssl_runs(root):
    """Final-epoch diagnostics and measured cost of each Phase B pretraining run."""
    rows = []
    for path in sorted(Path(root).glob('*/metrics.jsonl')):
        logs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        last = logs[-1]
        student = last.get('diagnostics', {}).get('student', {})
        teacher = last.get('diagnostics', {}).get('teacher', {})
        objective, rest = path.parent.name.split('_', 1)
        seed = rest.rsplit('_seed', 1)[1]
        rows.append(dict(objective=objective, seed=int(seed), steps=last['global_step'],
                         first_val=logs[0]['val_loss'], final_val=last['val_loss'],
                         best_val=min(r['val_loss'] for r in logs), seconds=last['wall_seconds'],
                         examples_per_second=float(np.mean([r['examples_per_second'] for r in logs])),
                         peak_gib=max(r['peak_gpu_bytes'] for r in logs) / 2**30,
                         rank=student.get('effective_rank_last'), content=student.get('content_fraction'),
                         cosine=student.get('mean_pairwise_cosine_last'),
                         teacher_rank=teacher.get('effective_rank_last'),
                         teacher_cosine=teacher.get('mean_pairwise_cosine_last')))
    return rows


def phase_b_report(df, agg, sessions, out, ssl_root, controls_path):
    lines = ['# Protocol v2 Phase B: latent prediction versus reconstruction (UTS01)', '',
             'Exploratory (UTS01 test stories were examined before). Both objectives share windows, '
             'masks, encoder, optimizer and 1,110 updates; each exports its final student encoder. The '
             'locked readout is the Phase A frozen ridge procedure at the 6-s cutoff with FIR states '
             '(layer and alpha chosen per encoder on validation). Test r is the session-macro feature '
             'correlation. See [PROTOCOL_V2.md](../../../experiments/PROTOCOL_V2.md) §5.', '']
    fir = dict(cutoff=3, readout='fir')
    specs = []
    for fraction in (1.0, 0.1, 0.25, 0.05, 0.01):
        specs.append((('PRIMARY ' if fraction == 1.0 else '') + 'Latent − reconstruction (frozen, FIR)', fraction,
                      dict(arm='L', features='latent_final', **fir), dict(arm='R', features='recon_final', **fir)))
    for fraction in (1.0, 0.1):
        specs += [('Latent − reconstruction (frozen, 8-s FIR)', fraction,
                   dict(arm='L', features='latent_final', cutoff=4, readout='fir'),
                   dict(arm='R', features='recon_final', cutoff=4, readout='fir')),
                  ('Latent − reconstruction (frozen, endpoint)', fraction,
                   dict(arm='L', features='latent_final', cutoff=3, readout='endpoint'),
                   dict(arm='R', features='recon_final', cutoff=3, readout='endpoint')),
                  ('Latent − reconstruction (fine-tuned)', fraction,
                   dict(arm='B5', features='latent', cutoff=3), dict(arm='B5', features='recon', cutoff=3)),
                  ('Latent − frozen random encoder', fraction, dict(arm='L', features='latent_final', **fir),
                   dict(arm='B3', features='random', **fir)),
                  ('Reconstruction (v2) − frozen random encoder', fraction,
                   dict(arm='R', features='recon_final', **fir), dict(arm='B3', features='random', **fir)),
                  ('Latent − raw BOLD benchmark', fraction, dict(arm='L', features='latent_final', **fir),
                   dict(arm='B1', features='bold', **fir)),
                  ('Reconstruction (v2) − raw BOLD benchmark', fraction,
                   dict(arm='R', features='recon_final', **fir), dict(arm='B1', features='bold', **fir))]
    lines += ['## Paired story-level contrasts', '',
              'Seeds are paired by pretraining seed; story differences are averaged over seeds before an '
              'exact sign-flip test over 11 test stories (stories share two sessions: descriptive only).', '']
    lines += contrast_lines(df, specs, sessions) + ['']
    for cutoff, readout in ((3, 'fir'), (3, 'endpoint'), (4, 'fir')):
        rows = agg[(agg.cutoff == cutoff) & ((agg.readout == readout) | (agg.readout == 'finetune') & (cutoff == 3)
                                             & (readout == 'fir'))
                   & agg.arm.isin(['L', 'R', 'B1', 'B3', 'B4', 'B5'])
                   & ~((agg.arm == 'B4') & (agg.features == 'ssl_best'))]
        for fraction in (1.0, 0.1):
            subset = rows[rows.fraction == fraction].sort_values('val', ascending=False)
            if subset.empty:
                continue
            lines += [f'## {2*cutoff}-second cutoff, {readout} readout, {fraction:.0%} labels', '']
            lines += table([r for _, r in subset.iterrows()], MAIN_COLUMNS) + ['']
    budget = agg[(agg.cutoff == 3) & (agg.readout == 'fir') & agg.arm.isin(['L', 'R', 'B1', 'B3', 'B4'])
                 & ~((agg.arm == 'B4') & (agg.features == 'ssl_best'))]
    fractions = sorted(budget.fraction.unique(), reverse=True)
    rows = []
    for (arm, features, readout), group in budget.groupby(['arm', 'features', 'readout']):
        rows.append(dict(arm=arm, features=features, readout=readout,
                         **{str(f): group[group.fraction == f].test.mean() for f in fractions}))
    lines += ['## Label budgets (6-s FIR; test r, mean over fits)', '']
    lines += table(rows, [('Arm', label)] + [(f'{f:.0%}', (lambda f: lambda r: fmt(r[str(f)]))(f))
                                             for f in fractions]) + ['']
    runs = ssl_runs(ssl_root)
    controls = json.loads(Path(controls_path).read_text()) if Path(controls_path).exists() else None
    lines += ['## Pretraining diagnostics and measured cost', '',
              'Validation losses are in each objective\'s own units and are not comparable across '
              'objectives. Effective rank and pairwise cosine use fully observed validation windows at '
              'the final position (512 windows); content fraction is across-window variance at fixed '
              'positions over total variance. Seconds are logged training time on one RTX PRO 6000.', '']
    columns = [('Objective', lambda r: r['objective']), ('Seed', lambda r: str(r['seed'])),
               ('Updates', lambda r: str(r['steps'])),
               ('Val loss first → final (best)', lambda r: f"{r['first_val']:.4f} → {r['final_val']:.4f} ({r['best_val']:.4f})"),
               ('Train-mean control', lambda r: fmt(controls['by_seed'][str(r['seed'])]['train_mean'])
                if controls and r['objective'] == 'recon' else '—'),
               ('Student rank', lambda r: fmt(r['rank'], 1)), ('Content fraction', lambda r: fmt(r['content'], 3)),
               ('Pairwise cos', lambda r: fmt(r['cosine'], 3)), ('Teacher rank', lambda r: fmt(r['teacher_rank'], 1)),
               ('Seconds', lambda r: fmt(r['seconds'], 1)), ('Examples/s', lambda r: fmt(r['examples_per_second'], 0)),
               ('Peak GiB', lambda r: fmt(r['peak_gib'], 3))]
    lines += table(runs, columns) + ['']
    (Path(out)/'RESULTS.md').write_text('\n'.join(lines) + '\n')


def phase_c_report(root, out):
    """Label coverage and counted supervision for the raw-BOLD benchmark."""
    cells = [json.loads(path.read_text()) for path in sorted(Path(root).glob('cells/*.json'))]
    full = [c for c in cells if c['fraction'] == 1.0]
    reference = full[0]['validation_selected']['test'][PRIMARY] if full else None
    lines = ['# Protocol v2 Phase C: label coverage and counted supervision (UTS01)', '',
             'Method fixed to the Phase A benchmark (raw pooled BOLD ridge, t+1…t+3). Each budget is '
             'crossed with two allocations of the same target count and five subset seeds (2718, 11–14). '
             '"Validation-selected" reuses the 794 validation labels for alpha; "CV-selected" chooses alpha '
             'by leave-one-story-out CV inside the labeled subset, so counted supervision equals the '
             'training labels. Test r is the session-macro feature correlation; ranges are observed '
             'minima/maxima over subset seeds. See [PROTOCOL_V2.md](../../../experiments/PROTOCOL_V2.md) §6.', '']
    if reference is not None:
        lines += [f'100% benchmark (one fit): test r {reference:.4f}; CV-selected '
                  f"{full[0].get('cv_selected', {}).get('test', {}).get(PRIMARY, float('nan')):.4f} "
                  f"(alphas {full[0]['validation_selected']['alpha']:.3g} / {full[0].get('cv_selected', {}).get('alpha', float('nan')):.3g}).", '']
    columns = [('Labels', None), ('Allocation', None), ('Stories (mean)', None), ('Sessions (mean)', None),
               ('Val-selected test r', None), ('CV-selected test r', None),
               ('Counted labels (val-sel / CV)', None), ('Share of 100% r (val-sel)', None)]
    lines += table([], columns)
    for fraction in sorted({c['fraction'] for c in cells if c['fraction'] < 1}, reverse=True):
        for allocation in ('block', 'story'):
            group = [c for c in cells if c['fraction'] == fraction and c['allocation'] == allocation]
            if not group:
                continue
            val = np.array([c['validation_selected']['test'][PRIMARY] for c in group])
            cv = np.array([c['cv_selected']['test'][PRIMARY] for c in group if 'cv_selected' in c])
            stories = np.mean([c['labels']['labeled_stories'] for c in group])
            sessions = np.mean([c['labels']['labeled_sessions'] for c in group])
            counted = (group[0]['validation_selected']['counted_supervision'],
                       group[0].get('cv_selected', {}).get('counted_supervision'))
            cv_text = (f'{cv.mean():.4f} [{cv.min():.4f}, {cv.max():.4f}] (n={len(cv)})' if len(cv) else '—')
            share = f'{val.mean() / reference:.0%}' if reference else '—'
            lines.append(f"| {group[0]['labels']['selected_examples']} ({fraction:.0%}) | {allocation}"
                         f"{group[0]['block_tr']} | {stories:.1f} | {sessions:.1f} | "
                         f'{val.mean():.4f} [{val.min():.4f}, {val.max():.4f}] | {cv_text} | '
                         f'{counted[0]} / {counted[1]} | {share} |')
    lines += ['', 'Useful label savings would require matching the 100% benchmark with fewer counted '
              'labels under a prespecified noninferiority margin; none was prespecified, so this table '
              'reports shares rather than a savings claim.']
    (Path(out)/'RESULTS.md').write_text('\n'.join(lines) + '\n')


REPLICATION = {  # Registered contrasts, PROTOCOL_V2.md §7: (name, a, b, predicted sign).
    'D1': ('Frozen latent − frozen reconstruction', dict(arm='L', features='latent_final', cutoff=3, readout='fir'),
           dict(arm='R', features='recon_final', cutoff=3, readout='fir'), -1),
    'D2': ('Raw BOLD ridge − frozen reconstruction', dict(arm='B1', features='bold', cutoff=3, readout='fir'),
           dict(arm='R', features='recon_final', cutoff=3, readout='fir'), +1),
    'D3': ('Raw BOLD ridge − fine-tuned reconstruction init', dict(arm='B1', features='bold', cutoff=3, readout='fir'),
           dict(arm='B5', features='recon', cutoff=3), +1),
    'D4': ('Frozen reconstruction − frozen random encoder', dict(arm='R', features='recon_final', cutoff=3, readout='fir'),
           dict(arm='B3', features='random', cutoff=3, readout='fir'), 0),
}


def replication_report(subjects, out):
    """Registered D1–D4 contrasts per subject; ``subjects`` maps name -> (ridge roots, finetune roots)."""
    frames, sessions = {}, {}
    for subject, (ridge, finetuned) in subjects.items():
        rows = [r for root in ridge for r in load_ridge(root)]
        rows += [r for root in finetuned for r in load_finetuned(root)]
        frames[subject] = pd.DataFrame(rows)
        sessions[subject] = test_sessions(ridge[0])
    lines = ['# Protocol v2 Phase D: registered replication in UTS02 and UTS03', '',
             'Contrasts and predicted signs were registered before any UTS02/UTS03 processing '
             '([PROTOCOL_V2.md](../../../experiments/PROTOCOL_V2.md) §7). Test session-macro r, 6-s cutoff, '
             'FIR readout. A contrast replicates when the seed-mean difference and every paired seed '
             'difference have the predicted sign. UTS01 values are the exploratory originals; D3 uses '
             'the v2 reconstruction initialization in every subject. Two test sessions per subject; no '
             'population inference.', '']
    for fraction in (1.0, 0.1):
        lines += [f'## {fraction:.0%} labels', '',
                  '| Contrast | ' + ' | '.join(subjects) + ' | Replicated in new subjects |',
                  '|---|' + '---:|' * len(subjects) + '---|']
        for key, (name, a, b, sign) in REPLICATION.items():
            cells, verdicts = [], []
            for subject, df in frames.items():
                result = paired(df[df.fraction == fraction], a, b, sessions[subject])
                if result is None:
                    cells.append('—')
                    continue
                pairs = result['per_pair_delta']
                cells.append(f"{result['session_macro_delta']:+.4f} ({', '.join(f'{v:+.4f}' for v in pairs)}; "
                             f"{result['stories_positive']}/{result['n_stories']} stories > 0)")
                if subject != 'UTS01' and sign:
                    ok = np.sign(result['session_macro_delta']) == sign and all(np.sign(v) == sign for v in pairs)
                    verdicts.append(f"{subject}: {'yes' if ok else 'no'}")
            label = 'report only' if not sign else ', '.join(verdicts) or '—'
            lines.append(f'| {key} {name} | ' + ' | '.join(cells) + f' | {label} |')
        lines += ['']
    lines += ['## Arm scores (test r, mean [min, max] over fits)', '',
              '| Arm | Labels | ' + ' | '.join(subjects) + ' |', '|---|---:|' + '---:|' * len(subjects)]
    arms = [('B1 raw BOLD', dict(arm='B1', features='bold', cutoff=3, readout='fir')),
            ('B3 frozen random', dict(arm='B3', features='random', cutoff=3, readout='fir')),
            ('R frozen reconstruction', dict(arm='R', features='recon_final', cutoff=3, readout='fir')),
            ('L frozen latent', dict(arm='L', features='latent_final', cutoff=3, readout='fir')),
            ('Fine-tuned, random init', dict(arm='B5', features='random', cutoff=3)),
            ('Fine-tuned, reconstruction init', dict(arm='B5', features='recon', cutoff=3)),
            ('Fine-tuned, latent init', dict(arm='B5', features='latent', cutoff=3))]
    for fraction in (1.0, 0.1):
        for name, spec in arms:
            cells = []
            for subject, df in frames.items():
                sel = df[(df.fraction == fraction) & np.logical_and.reduce([df[k] == v for k, v in spec.items()])]
                cells.append('—' if sel.empty else (f'{sel.test.mean():.4f}' if len(sel) == 1 else
                             f'{sel.test.mean():.4f} [{sel.test.min():.4f}, {sel.test.max():.4f}]'))
            lines.append(f'| {name} | {fraction:.0%} | ' + ' | '.join(cells) + ' |')
    (Path(out)/'RESULTS.md').write_text('\n'.join(lines) + '\n')


def transfer_report(root, out):
    """Ridge-decoder modality transfer (PROTOCOL_V2.md §9), one row per subject and condition."""
    lines = ['# Protocol v2: ridge-decoder modality transfer (ds004510)', '',
             'Decoder: each subject\'s Phase A benchmark (raw pooled BOLD ridge, t+1…t+3, 100% labels, '
             'alpha from ds003020 validation). Run vectors average predictions over every t with t+3 in the '
             'run; retrieval uses fixed whole-reference GPT-2 vectors. Narrative-macro scores with exact '
             'narrative-label permutation p-values (imagined: 120 permutations; films: 6, minimum p = 1/6). '
             'UTS01 is exploratory (its transfer data were viewed in the archived study); UTS02 and UTS03 '
             'follow the registered rule in [PROTOCOL_V2.md](../../../experiments/PROTOCOL_V2.md) §9.', '',
             '| Subject | Condition | Centered top-1 (p) | Centered MRR (p) | Raw top-1 (p) | Constant-mean top-1 | Chance | Runs | TRs averaged per run |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    verdicts = {}
    for path in sorted(Path(root).glob('*/results.json')):
        result = json.loads(path.read_text())
        subject = result['subject']
        for condition, report in result['conditions'].items():
            if report.get('status') != 'scored':
                lines.append(f'| {subject} | {condition} | {report.get("status")} | | | | | | |')
                continue
            c, r, k = (report[v] for v in ('training_mean_centered', 'raw', 'constant_training_mean'))
            perm = c['permutation']['metrics']
            counts = sorted({run['n_tr_averaged'] for run in report['runs']})
            lines.append(f"| {subject} | {condition} | {c['narrative_macro']['top1']:.0%} ({perm['top1']['p_greater_equal']:.3f}) | "
                         f"{c['narrative_macro']['mrr']:.3f} ({perm['mrr']['p_greater_equal']:.3f}) | "
                         f"{r['narrative_macro']['top1']:.0%} ({r['permutation']['metrics']['top1']['p_greater_equal']:.3f}) | "
                         f"{k['narrative_macro']['top1']:.0%} | {c['top1_chance']:.0%} | {c['n_queries']} | "
                         f"{', '.join(map(str, counts))} |")
            if condition == 'imagined_speech' and subject != 'UTS01':
                verdicts[subject] = perm['top1']['p_greater_equal'] <= 0.05
    if verdicts:
        supported = all(verdicts.values()) and len(verdicts) == 2
        lines += ['', 'Registered decision: imagined-speech transfer requires centered top-1 permutation '
                  'p ≤ 0.05 in both UTS02 and UTS03. ' + ', '.join(f"{k}: {'met' if v else 'not met'}"
                                                                   for k, v in sorted(verdicts.items()))
                  + f". Transfer is {'supported' if supported else 'not established'} under the registered rule."]
    lines += ['', 'Archived UTS01 comparison (fine-tuned Transformers, 100% labels, mean of 3 seeds, '
              '[results/transfer_comparison](../../transfer_comparison/RESULTS.md)): centered imagined '
              'top-1 26.7% (random init) and 36.7% (SSL init); films 44.4% and 33.3%.']
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out)/'RESULTS.md').write_text('\n'.join(lines) + '\n')


def voxel_report(subjects, out):
    """Exploratory Phase V: pooled-cube versus all-voxel ridge, same targets and selection."""
    lines = ['# Protocol v2 Phase V (exploratory): spatial pooling versus all voxels', '',
             'Same targets, labels, delays, standardization rule and validation-selected alpha as the pooled '
             'benchmark B1; only the input changes from 4-voxel cubes to every voxel of the subject\'s thick '
             'mask (kernel ridge; [scripts/run_voxel_ridge.py](../../../scripts/run_voxel_ridge.py)). '
             'Test session-macro r; the bracket is the centered retrieval rank percentile (±10 TR).', '',
             '| Subject | Readout | Labels | Pooled cubes | All voxels | Δ r |', '|---|---|---:|---:|---:|---:|']
    for subject, (pooled_root, voxel_root) in subjects.items():
        for readout in ('fir', 'endpoint'):
            for path in sorted(Path(voxel_root).glob(f'cells/V1_voxels_seed0_c3_{readout}_f*.json'),
                               key=lambda q: -float(q.stem.rsplit('_f', 1)[1])):
                voxel = json.loads(path.read_text())
                budget = path.stem.rsplit('_f', 1)[1]
                pooled_path = Path(pooled_root)/'cells'/f'B1_bold_seed0_c3_{readout}_f{budget}.json'
                if not pooled_path.exists():
                    continue
                pooled = json.loads(pooled_path.read_text())
                a, b = (c['test'][PRIMARY] for c in (pooled, voxel))
                ra, rb = (c['test']['centered_retrieval']['10']['session_macro_rank_percentile'] for c in (pooled, voxel))
                lines.append(f"| {subject} | {readout} | {voxel['labels']['selected_examples']} | {a:.4f} [{ra:.3f}] | "
                             f'{b:.4f} [{rb:.3f}] | {b - a:+.4f} |')
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out)/'RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ridge', nargs='+', default=['results/v2/phase_a'])
    p.add_argument('--finetuned', nargs='*', default=['results/v2/finetune_a'])
    p.add_argument('--out', default='results/v2/phase_a')
    p.add_argument('--report', choices=['phase_a', 'phase_b', 'phase_c', 'replication', 'transfer', 'voxel', 'none'], default='phase_a')
    p.add_argument('--ssl-root', default='results/v2/ssl')
    p.add_argument('--controls', default='results/v2/reconstruction_controls.json')
    a = p.parse_args()
    if a.report == 'phase_c':
        phase_c_report(a.ridge[0], a.out)
        print(f'wrote {a.out}')
        return
    if a.report == 'voxel':
        subjects = {'UTS01': ('results/v2/phase_a', 'results/v2/phase_v')}
        for subject in ('UTS02', 'UTS03'):
            root = Path('results/v2/replication')/subject
            if (root/'phase_v').exists():
                subjects[subject] = (str(root/'phase_a'), str(root/'phase_v'))
        voxel_report(subjects, a.out)
        print(f'wrote {a.out}')
        return
    if a.report == 'transfer':
        transfer_report(a.ridge[0], a.out)
        print(f'wrote {a.out}')
        return
    if a.report == 'replication':
        subjects = {'UTS01': (['results/v2/phase_a', 'results/v2/phase_b'],
                              ['results/v2/finetune_a', 'results/v2/finetune_b'])}
        for subject in ('UTS02', 'UTS03'):
            root = Path('results/v2/replication')/subject
            subjects[subject] = ([str(root/'phase_a'), str(root/'phase_b')], [str(root/'finetune')])
        Path(a.out).mkdir(parents=True, exist_ok=True)
        replication_report(subjects, a.out)
        print(f'wrote {a.out}')
        return
    rows = [r for root in a.ridge for r in load_ridge(root)]
    rows += [r for root in a.finetuned for r in load_finetuned(root)]
    df = pd.DataFrame(rows)
    out = Path(a.out)
    df.drop(columns=['by_run', 'sessions']).to_csv(out/'cells.csv', index=False)
    agg = grouped(df)
    agg.to_csv(out/'summary.csv', index=False)
    if a.report == 'phase_a':
        phase_a_report(df, agg, test_sessions(a.ridge[0]), out)
    elif a.report == 'phase_b':
        phase_b_report(df, agg, test_sessions(a.ridge[0]), out, a.ssl_root, a.controls)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
