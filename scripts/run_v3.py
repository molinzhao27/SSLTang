#!/usr/bin/env python3
"""Protocol v3 structured voxel ridge (experiments/PROTOCOL_V3.md).

Stages:
  cache     cache raw H5 rows t+1..t+3 for every train/val/test target, with provenance
  recover   G0(b): V0 at Phase V's alphas against Phase V predictions
  develop   grouped-CV selection for V0-V4 at 100% and 10%; validation scores only
  lock      write experiments/v3_lock.json from all subjects' development results
  evaluate  after a verified lock: all budgets, validation and test scores, shift nulls
  controls  after evaluate: registered brain-input and nuisance controls (§6)
Test targets are never scored before a lock whose fingerprints match this code.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.endpoints import score, shift_null
from src.multikernel_ridge import KernelRidgePath, combine, session_macro_scores
from src.probe import Split
from src.readout import RidgePath, Standardizer, run_codes
from src.runtime import file_sha256, load_config, save_json
from src.structured_ridge import (ARMS, BASELINE, ETAS, TAUS, TIE, aggregate, budget_folds, coefficients,
                                  fixed_folds, label, score_fold, select, select_per_output, _block)
from src.targets import TargetIndex
from src.voxel_kernels import DELAYS, KernelBuilder, cache_delay_rows, membership_groups, verify_cache

PROTOCOL_FOLDS = [['9', '7', '10'], ['15', '6'], ['18', '11'], ['12', '8'], ['5', '3']]
ROOT = Path(__file__).resolve().parents[1]
LOCKED_CODE = ('src/multikernel_ridge.py', 'src/voxel_kernels.py', 'src/structured_ridge.py', 'scripts/run_v3.py',
               'src/targets.py', 'src/evaluation/endpoints.py', 'src/evaluation/metrics.py',
               'src/transcripts.py', 'src/readout.py', 'src/semantic/features.py')
ARM_NAMES = ('V0', 'V1', 'V2', 'V3', 'V4')


def code_sha256():
    digest = hashlib.sha256()
    for name in LOCKED_CODE:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def config_sha256(cfg):
    shared = json.loads(json.dumps(cfg))
    shared['data'].pop('manifest', None)  # Subject-specific; everything else is shared.
    return hashlib.sha256(json.dumps(shared, sort_keys=True).encode()).hexdigest()


class Context:
    def __init__(self, cfg, device, build_cache=False):
        data = cfg['data']
        self.cfg, self.device = cfg, torch.device(device)
        self.manifest_path = data['manifest']
        self.index = {s: TargetIndex(data['manifest'], s, window=data['window'], stride=data['stride'],
                                     phase=data['target_phase'], cutoffs=data['cutoffs'])
                      for s in ('train', 'val', 'test')}
        self.subject = self.index['train'].runs[0]['subject']
        self.split = {s: Split(self.index[s], 'cpu') for s in self.index}
        cache_root = Path(cfg['cache_root']) / self.subject
        self.cache = {}
        for s, index in self.index.items():
            make = cache_delay_rows if build_cache else verify_cache
            self.cache[s], self.n_voxels = make(index, cfg['raw_root'], cache_root)
        self.groups = membership_groups(self.index['train'].manifest, self.manifest_path, self.n_voxels)
        self.builder = KernelBuilder(self.cache, self.groups, device)
        train = self.index['train']
        self.row_sessions = [train.sessions[r] for r in train.run_ids]
        self.run_sessions = dict(train.sessions)
        self.fixed = fixed_folds(self.row_sessions)
        if [sorted(f, key=int) for f in self.fixed] != [sorted(f, key=int) for f in PROTOCOL_FOLDS]:
            raise ValueError(f'Fold assignment {self.fixed} differs from the protocol')
        labels = cfg['labels']
        self.budgets, self.subsets = {}, {}
        for fraction in labels['fractions']:
            rows, meta = train.label_subset(fraction, labels['subset_seed'], labels['block_tr'], labels['allocation'])
            self.budgets[f'{fraction:g}'], self.subsets[f'{fraction:g}'] = rows, meta
        self.alphas = np.logspace(*cfg['alphas'])
        self.out = Path(cfg['output_root']) / self.subject

    def provenance(self, stage):
        return dict(protocol='ssltang-v3', stage=stage, subject=self.subject, config=self.cfg,
                    config_sha256=config_sha256(self.cfg), code_sha256=code_sha256(),
                    manifest_sha256=file_sha256(self.manifest_path), n_voxels=self.n_voxels,
                    groups={k: v for k, v in self.groups.items() if k not in ('membership', 'sizes')},
                    folds=self.fixed, alphas=self.alphas.tolist(),
                    targets={s: i.signature() for s, i in self.index.items()})


def cross_validate(ctx, budget_names):
    """Out-of-fold CV for every candidate (and V1 outputs) at the named budgets."""
    plans = {b: budget_folds(ctx.row_sessions, ctx.budgets[b], ctx.fixed) for b in budget_names}
    held_sets = list(dict.fromkeys(f for folds, _ in plans.values() if folds for f in folds))
    results = {b: [] for b in budget_names}
    fold_log = {b: [] for b in budget_names}
    y = ctx.split['train'].targets
    run_ids = ctx.index['train'].run_ids
    for held in held_sets:
        start = time.monotonic()
        held_set = set(held)
        stats_rows = [i for i, s in enumerate(ctx.row_sessions) if s not in held_set]
        kernels, _, _ = ctx.builder.build(stats_rows)
        for b in budget_names:
            folds, _ = plans[b]
            if not folds or held not in folds:
                continue
            rows = ctx.budgets[b]
            fit = [i for i in rows if ctx.row_sessions[i] not in held_set]
            held_rows = [i for i in rows if ctx.row_sessions[i] in held_set]
            if len(fit) < 2 or not held_rows:
                fold_log[b].append(dict(held_sessions=list(held), skipped='too few fitting or held rows'))
                continue
            held_runs = list(dict.fromkeys(run_ids[i] for i in held_rows))
            codes = np.array([held_runs.index(run_ids[i]) for i in held_rows])
            out = score_fold(kernels, fit, held_rows, y, codes, len(held_runs), ctx.alphas, ctx.device)
            results[b].append((held_runs, out))
            fold_log[b].append(dict(held_sessions=list(held), n_fit=len(fit), n_held=len(held_rows),
                                    held_runs=len(held_runs), covered_rows=len(held_rows),
                                    eigen={label(c): v['diagnostics'] for c, v in out.items()}))
        del kernels
        torch.cuda.empty_cache()
        print(json.dumps(dict(subject=ctx.subject, held=list(held), seconds=round(time.monotonic() - start, 1))),
              flush=True)
    cv = {}
    for b in budget_names:
        folds, rule = plans[b]
        covered = sum(f.get('covered_rows', 0) for f in fold_log[b])
        reason = None
        if not folds:
            reason = rule
        elif any('skipped' in f for f in fold_log[b]):
            reason = 'not estimable: a fold had too few fitting or held-out rows'
        elif covered != len(ctx.budgets[b]):
            reason = f'not estimable: out-of-fold coverage {covered} of {len(ctx.budgets[b])} labeled rows'
        if reason:
            cv[b] = dict(estimable=False, rule=rule, reason=reason, folds=fold_log[b])
            continue
        scores, outputs = aggregate(results[b], ctx.run_sessions, ctx.alphas)
        cv[b] = dict(estimable=True, rule=rule, folds=fold_log[b], scores=scores, output_scores=outputs,
                     coverage=covered)
    return cv


def selections(cv_b, alphas):
    """Per-arm selections; an arm without any finite CV score is not estimable."""
    chosen = {arm: select(arm, cv_b['scores'], alphas) for arm in ARMS}
    v0 = chosen['V0']
    if v0 is not None and cv_b['output_scores'] is not None:
        per_output, summary = select_per_output(cv_b['output_scores'], v0['alpha'], alphas)
        chosen['V1'] = dict(candidate=BASELINE, label=label(BASELINE), alphas_per_output=per_output.tolist(),
                            cv_score=float(np.nanmean(np.nanmax(cv_b['output_scores'], axis=0))), **summary)
    else:
        chosen['V1'] = None
    return chosen


def fit_predict(ctx, kernels, cross, rows, selection, splits, targets=None):
    rows_t = torch.as_tensor(np.asarray(rows), device=ctx.device)
    coef = coefficients(*selection['candidate'])
    blocks = _block(kernels, rows_t, rows_t)
    y = ctx.split['train'].targets if targets is None else targets
    path = KernelRidgePath(combine(blocks, coef), y[rows], ctx.device)
    out = {}
    for s in splits:
        projected = path.project(combine({k: v.index_select(1, rows_t) for k, v in cross[s].items()}, coef))
        prediction = (path.predict_per_output(projected, selection['alphas_per_output'])
                      if 'alphas_per_output' in selection else path.predict(projected, selection['alpha']))
        out[s] = prediction.cpu().numpy()
    n = len(rows)
    scales = {f'{g}{d}': float((k.diagonal().sum() - k.sum() / n) / n) for (g, d), k in blocks.items()}
    return out, dict(path.diagnostics, base_kernel_trace_scales=scales)


def report(ctx, b, prediction, split, targets=None):
    y = ctx.split['train'].targets[ctx.budgets[b]].astype(np.float64)
    mean, std = y.mean(0), np.maximum(y.std(0), 1e-6)
    sp = ctx.split[split]
    truth = sp.targets if targets is None else targets
    return score(prediction, truth, sp.run_ids, sp.target_trs, sp.sessions, mean, std,
                 ctx.cfg['evaluation']['exclusions'])


def save_cv(ctx, cv, stage, budget_names):
    folder = ctx.out / stage
    folder.mkdir(parents=True, exist_ok=True)
    summary, arrays = {}, {}
    for b in budget_names:
        entry = {k: v for k, v in cv[b].items() if k not in ('scores', 'output_scores')}
        if cv[b]['estimable']:
            entry['scores'] = {label(c): [None if not np.isfinite(x) else float(x) for x in v]
                               for c, v in cv[b]['scores'].items()}
            arrays[f'output_scores_f{b}'] = cv[b]['output_scores']
        summary[b] = entry
    save_json(dict(provenance=ctx.provenance(stage), alphas=ctx.alphas.tolist(), cv=summary), folder / 'cv.json')
    np.savez_compressed(folder / 'cv_output_scores.npz', **arrays)


def stage_recover(ctx):
    """V0 at Phase V's selected alphas against Phase V validation predictions (see §10)."""
    kernels, cross, _ = ctx.builder.build(list(range(len(ctx.row_sessions))), ('val',))
    rows = {}
    for b in ('1', '0.1'):
        cell = json.loads(Path(f'results/v2/phase_v/cells/V1_voxels_seed0_c3_fir_f{b}.json').read_text())
        saved = np.load(f'results/v2/phase_v/predictions/V1_voxels_seed0_c3_fir_f{b}.npz')['val'].astype(np.float64)
        pred, diag = fit_predict(ctx, kernels, cross, ctx.budgets[b],
                                 dict(candidate=BASELINE, alpha=cell['selection']['alpha']), ('val',))
        rms = float(np.sqrt(np.mean(pred['val'] ** 2)))
        diff = float(np.abs(pred['val'] - saved).max())
        mine = report(ctx, b, pred['val'], 'val')['session_macro_feature_correlation']
        theirs = cell['val']['session_macro_feature_correlation']
        rows[b] = dict(alpha=cell['selection']['alpha'], max_abs_diff=diff, prediction_rms=rms,
                       relative=diff / rms, val_r_v3=mine, val_r_phase_v=theirs, val_r_diff=mine - theirs,
                       eigen=diag, passed_as_written=bool(diff <= 2e-3 * rms and abs(mine - theirs) <= 1e-4))
        print(b, json.dumps({k: v for k, v in rows[b].items() if k != 'eigen'}), flush=True)
    save_json(dict(provenance=ctx.provenance('recover'), gate='G0b', results=rows,
                   note='Written tolerance mis-specified for float16 storage; see PROTOCOL_V3 §10 and '
                        'recovery_diagnostic.json'), ctx.out / 'recovery.json')


def stage_fit(ctx, budget_names, splits, stage):
    cv = cross_validate(ctx, budget_names)
    save_cv(ctx, cv, stage, budget_names)
    kernels, cross, info = ctx.builder.build(list(range(len(ctx.row_sessions))), splits)
    folder = ctx.out / stage
    (folder / 'cells').mkdir(parents=True, exist_ok=True)
    (folder / 'predictions').mkdir(parents=True, exist_ok=True)
    norm_arrays = {f'{k}_d{d}': v for d, item in info.items() for k, v in item.items() if k in ('mean', 'std')}
    np.savez_compressed(folder / 'normalization_state.npz', **norm_arrays)
    for b in budget_names:
        chosen = selections(cv[b], ctx.alphas) if cv[b]['estimable'] else {arm: None for arm in ARM_NAMES}
        for arm in ARM_NAMES:
            selection = chosen.get(arm)
            base = dict(protocol='ssltang-v3', stage=stage, subject=ctx.subject, arm=arm, fraction=float(b),
                        labels={k: v for k, v in ctx.subsets[b].items() if k != 'label_ids'},
                        validation_labels=len(ctx.split['val'].targets), cv_rule=cv[b]['rule'])
            if selection is None:
                save_json(dict(base, estimable=False, reason=cv[b].get('reason', 'no finite CV score'),
                               folds=cv[b]['folds']), folder / 'cells' / f'{arm}_f{b}.json')
                continue
            start = time.monotonic()
            predictions, diag = fit_predict(ctx, kernels, cross, ctx.budgets[b], selection, splits)
            cell = dict(base, estimable=True,
                        selection={k: (list(v) if k == 'candidate' else v) for k, v in selection.items()
                                   if k != 'alphas_per_output'},
                        arm_candidates=[label(c) for c in ARMS.get(arm, [BASELINE])],
                        final_fit=diag, normalization_rows='all training targets',
                        normalization_state_sha256=file_sha256(folder / 'normalization_state.npz'),
                        seconds=time.monotonic() - start)
            for s in splits:
                cell[s] = report(ctx, b, predictions[s], s)
            if 'test' in splits:
                ev, sp = ctx.cfg['evaluation'], ctx.split['test']
                cell['test_shift_null'] = shift_null(predictions['test'], sp.targets, sp.run_ids, sp.sessions,
                                                     ctx.cfg['data']['stride'], ev['shift_min_tr'],
                                                     ev['shift_draws'], ev['shift_seed'])
            save_json(cell, folder / 'cells' / f'{arm}_f{b}.json')
            arrays = {s: v.astype(np.float16) for s, v in predictions.items()}
            if 'alphas_per_output' in selection:
                arrays['alphas_per_output'] = np.asarray(selection['alphas_per_output'])
            np.savez_compressed(folder / 'predictions' / f'{arm}_f{b}.npz', **arrays)
            shown = {s: round(cell[s]['session_macro_feature_correlation'], 4) for s in splits}
            print(f"{ctx.subject} {stage} {arm} f{b}: {selection['label']} "
                  f"alpha={selection.get('alpha', 'per-output')} cv={selection['cv_score']} {shown}", flush=True)
    save_json(ctx.provenance(stage), folder / 'provenance.json')


def stage_lock(cfg, subjects, advanced, gates):
    """Bind the advanced arm, grid, folds, ties, config and code into the lock file."""
    developed = {}
    for subject in subjects:
        prov = json.loads((Path(cfg['output_root']) / subject / 'develop' / 'provenance.json').read_text())
        if prov['code_sha256'] != code_sha256() or prov['config_sha256'] != config_sha256(cfg):
            raise SystemExit(f'{subject} development used different code or configuration; rerun develop')
        developed[subject] = {p.stem: json.loads(p.read_text()).get('val', {}).get('session_macro_feature_correlation')
                              for p in sorted((Path(cfg['output_root']) / subject / 'develop' / 'cells').glob('*.json'))}
    lock = dict(protocol='ssltang-v3', created_at=datetime.now(timezone.utc).isoformat(), advanced_arm=advanced,
                grid=dict(etas=list(ETAS), taus=[list(t) for t in TAUS], baseline=[BASELINE[0], list(BASELINE[1])]),
                folds=PROTOCOL_FOLDS, tie_rule=f'declared candidate order, then larger alpha; tolerance {TIE}',
                alphas=np.logspace(*cfg['alphas']).tolist(), config_sha256=config_sha256(cfg),
                code_sha256=code_sha256(), locked_code=list(LOCKED_CODE), subjects=subjects,
                development_validation_r=developed, gates=gates)
    save_json(lock, cfg['lock_file'])
    print(json.dumps(lock, indent=1)[:2000])


def verify_lock(cfg):
    path = Path(cfg['lock_file'])
    if not path.exists():
        raise SystemExit('No v3 lock file; run the lock stage before any test score')
    lock = json.loads(path.read_text())
    required = ('advanced_arm', 'grid', 'folds', 'tie_rule', 'alphas', 'config_sha256', 'code_sha256', 'gates')
    missing = [k for k in required if k not in lock]
    if missing or lock['advanced_arm'] not in ARM_NAMES:
        raise SystemExit(f'Lock file incomplete: {missing or lock["advanced_arm"]}')
    if lock['config_sha256'] != config_sha256(cfg) or lock['code_sha256'] != code_sha256():
        raise SystemExit('Configuration or locked code changed since the lock; re-lock with a recorded deviation')
    if lock['folds'] != PROTOCOL_FOLDS or not np.allclose(lock['alphas'], np.logspace(*cfg['alphas'])):
        raise SystemExit('Lock grid/folds differ from this run')
    return lock


def feature_ridge_cv(ctx, x_train, rows, y, splits_x):
    """Ridge on small nuisance features with the same session-grouped CV (§6)."""
    folds, rule = budget_folds(ctx.row_sessions, rows, ctx.fixed)
    run_ids = ctx.index['train'].run_ids
    per_run, runs = [], []
    for held in folds:
        fit = [i for i in rows if ctx.row_sessions[i] not in held]
        held_rows = [i for i in rows if ctx.row_sessions[i] in held]
        inner = [i for i, s in enumerate(ctx.row_sessions) if s not in held]
        scale = Standardizer(x_train[inner])
        ridge = RidgePath(scale(x_train[fit]), y[fit], 'cpu')
        projected = ridge.project(scale(x_train[held_rows]))
        names, codes = run_codes(run_ids[held_rows])
        target = torch.as_tensor(y[held_rows], dtype=torch.float64)
        from src.multikernel_ridge import run_feature_r
        preds = torch.stack([ridge.predict(projected, a) for a in ctx.alphas])
        r = run_feature_r(preds, target, torch.as_tensor(codes), len(names))
        per_run.append(torch.nanmean(r, dim=-1).numpy())
        runs.extend(names)
    scores = session_macro_scores(np.concatenate(per_run, axis=-1), [ctx.run_sessions[r] for r in runs])
    k = int(np.flatnonzero(scores >= np.nanmax(scores) - TIE)[-1])
    scale = Standardizer(x_train)
    ridge = RidgePath(scale(x_train[rows]), y[rows], 'cpu')
    preds = {s: ridge.predict(ridge.project(scale(x)), ctx.alphas[k]).numpy() for s, x in splits_x.items()}
    return preds, dict(alpha=float(ctx.alphas[k]), cv_score=float(scores[k]), rule=rule)


def stage_controls(ctx, lock):
    """Registered controls for the advanced arm at 100% labels (PROTOCOL_V3 §6)."""
    from src.transcripts import nuisance_features, run_words
    arm, b = lock['advanced_arm'], '1'
    cell = json.loads((ctx.out / 'evaluate' / 'cells' / f'{arm}_f{b}.json').read_text())
    v0 = json.loads((ctx.out / 'evaluate' / 'cells' / f'V0_f{b}.json').read_text())
    sel = lambda c: dict(candidate=(c['selection']['candidate'][0], tuple(c['selection']['candidate'][1])),
                         alpha=c['selection']['alpha'])
    chosen, baseline = sel(cell), sel(v0)
    kernels, cross, _ = ctx.builder.build(list(range(len(ctx.row_sessions))), ('test',))
    rows = ctx.budgets[b]
    y = ctx.split['train'].targets
    test = ctx.split['test']
    out = dict(provenance=ctx.provenance('controls'), advanced_arm=arm, selection=cell['selection'],
               real_test=cell['test']['session_macro_feature_correlation'])
    # (a) Training targets rotated within each training run by floor(n/2) rows.
    rotated = y.copy()
    run_ids = ctx.index['train'].run_ids
    shifts = {}
    for run in dict.fromkeys(run_ids.tolist()):
        idx = np.flatnonzero(run_ids == run)
        shifts[run] = len(idx) // 2
        rotated[idx] = np.roll(y[idx], shifts[run], axis=0)
    pred, _ = fit_predict(ctx, kernels, cross, rows, chosen, ('test',), targets=rotated)
    out['disrupted_training'] = dict(min_shift_rows=min(shifts.values()),
                                     test=report(ctx, b, pred['test'], 'test')['session_macro_feature_correlation'])
    # (b) Wrong-run input: each test run scored against the next run's predictions.
    real = np.load(ctx.out / 'evaluate' / 'predictions' / f'{arm}_f{b}.npz')['test'].astype(np.float64)
    names = list(dict.fromkeys(test.run_ids.tolist()))
    wrong = np.empty_like(real)
    for i, run in enumerate(names):
        mine, other = np.flatnonzero(test.run_ids == run), np.flatnonzero(test.run_ids == names[(i + 1) % len(names)])
        wrong[mine] = real[other][np.arange(len(mine)) % len(other)]
    out['wrong_run_input'] = dict(test=report(ctx, b, wrong, 'test')['session_macro_feature_correlation'],
                                  rule='next test run in run-ID order, cycled to equal length')
    # (c) Nuisance decoders and (d) residualization, all fit on training rows only.
    words = {s: {r['run_id']: run_words(r, ctx.manifest_path) for r in ctx.index[s].runs} for s in ctx.index}
    feats = {}
    for group in ('position', 'speech_rate', ('position', 'speech_rate')):
        x = {s: nuisance_features(ctx.index[s], words[s], groups=group)[0] for s in ('train', 'test')}
        preds, info = feature_ridge_cv(ctx, x['train'], rows, y, {'train': x['train'], 'test': x['test']})
        name = group if isinstance(group, str) else 'both'
        feats[name] = preds
        out[f'nuisance_{name}'] = dict(info, test=report(ctx, b, preds['test'], 'test')
                                       ['session_macro_feature_correlation'],
                                       oracle='speech_rate' in group)
    residual_train = y - feats['both']['train']
    residual_test = test.targets - feats['both']['test']
    for name, selection in (('advanced', chosen), ('V0', baseline)):
        pred, _ = fit_predict(ctx, kernels, cross, rows, selection, ('test',), targets=residual_train)
        out[f'residualized_{name}'] = dict(test=report(ctx, b, pred['test'], 'test', targets=residual_test)
                                           ['session_macro_feature_correlation'],
                                           note='targets minus training-fit nuisance (position + oracle speech rate)')
    save_json(out, ctx.out / 'controls.json')
    print(json.dumps({k: v for k, v in out.items() if k != 'provenance'}, indent=1), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['cache', 'recover', 'develop', 'lock', 'evaluate', 'controls'], required=True)
    p.add_argument('--config', default='configs/v3/structured_ridge.yaml')
    p.add_argument('--set', action='append', default=[])
    p.add_argument('--device', default='cuda')
    p.add_argument('--subjects', nargs='+', default=['UTS01', 'UTS02', 'UTS03'])
    p.add_argument('--advanced', choices=ARM_NAMES)
    p.add_argument('--gates', help='JSON with the recorded gate outcomes (lock stage)')
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if a.stage == 'lock':
        if not a.advanced or not a.gates:
            raise SystemExit('--advanced and --gates are required to lock')
        stage_lock(cfg, a.subjects, a.advanced, json.loads(a.gates))
        return
    lock = verify_lock(cfg) if a.stage in ('evaluate', 'controls') else None
    ctx = Context(cfg, a.device, build_cache=a.stage == 'cache')
    ctx.out.mkdir(parents=True, exist_ok=True)
    if a.stage == 'cache':
        print(json.dumps(dict(subject=ctx.subject, n_voxels=ctx.n_voxels, groups=ctx.groups['n_groups'],
                              coordinates_verified=ctx.groups['coordinates_verified'])))
    elif a.stage == 'recover':
        stage_recover(ctx)
    elif a.stage == 'develop':
        stage_fit(ctx, cfg['develop_fractions'], ('val',), 'develop')
    elif a.stage == 'evaluate':
        stage_fit(ctx, [f'{f:g}' for f in cfg['labels']['fractions']], ('val', 'test'), 'evaluate')
    else:
        stage_controls(ctx, lock)


if __name__ == '__main__':
    main()
