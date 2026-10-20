#!/usr/bin/env python3
"""PROTOCOL_V3 §12: encoding-predictability voxel weighting.

Stages:
  develop   nested grouped CV for W0/W1/W2 at 100% and 10% (stride-4); validation only
  lock      apply the §12 gates to all UTS01-03 development cells; write experiments/v3w_lock.json
  evaluate  after a matching lock: five budgets, validation and test, shift nulls,
            residualized (speech-rate) endpoint at 100%
  dense     dense training clock: the selected rule fixed, alpha re-selected
Encoding scores are always computed from labeled training rows of the fitting
partition only (nested inside every decoder-CV fold). When nesting leaves fewer
than two encoding folds, W1/W2 are not estimable at that budget; W0 never uses
encoding scores.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v3 as v3
from src.encoding_weights import ARMS, RULES, TIE, column_weights, encoding_scores, rule_label
from src.evaluation.endpoints import score, shift_null
from src.multikernel_ridge import KernelRidgePath, run_feature_r, session_macro_scores
from src.runtime import file_sha256, load_config, save_json
from src.structured_ridge import budget_folds
from src.voxel_kernels import DELAYS, column_stats

ROOT = Path(__file__).resolve().parents[1]
CODE = ('src/encoding_weights.py', 'scripts/run_v3_weights.py', 'scripts/run_v3w_replicate.py',
        'scripts/run_v3_dense.py', 'src/probe.py')
ARM_ORDER = ('W0', 'W1', 'W2')
REGISTERED = ('UTS01', 'UTS02', 'UTS03')
GATE_FRACTIONS = ('1', '0.1')
LOCK = Path('experiments/v3w_lock.json')
CHUNK = 16384


def code_sha256():
    digest = hashlib.sha256(v3.code_sha256().encode())
    for name in CODE:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def load(cache, device):
    array = np.array(np.load(cache, mmap_mode='r'), dtype=np.float32)
    return torch.from_numpy(array).to(device, torch.float64)


class Designs:
    """Per-delay standardized designs, streamed one delay at a time."""

    def __init__(self, cache, stats_rows, device):
        self.cache, self.device = cache, torch.device(device)
        self.stats_rows = torch.as_tensor(np.asarray(stats_rows), dtype=torch.long, device=self.device)
        self.moments = {}

    def train(self, d):
        x = load(self.cache['train'][d], self.device)
        if d not in self.moments:
            self.moments[d] = column_stats(x, self.stats_rows)
        mean, std = self.moments[d]
        return x.sub_(mean).div_(std)

    def other(self, split, d):
        mean, std = self.moments[d]
        return load(self.cache[split][d], self.device).sub_(mean).div_(std)


def weighted_kernels(designs, y, fit_rows, row_sessions, enc_folds, rules, alphas, eval_splits=(), kernel_rows=None):
    """Nested encoding scores from ``fit_rows`` and a weighted linear kernel per estimable rule.

    Kernels cover ``kernel_rows`` (default: every training row) in that order;
    cross-kernels are [eval rows, kernel rows]. Non-uniform rules are absent
    from the result when encoding scores are not estimable (fewer than two
    encoding folds, or a fold without fitting/held-out labeled rows).
    """
    scores, enc_alpha, enc_curve, reason = {}, {}, {}, None
    if any(kind != 'uniform' for kind, _ in rules):
        if len(enc_folds) < 2:
            reason = 'fewer than two encoding folds'
        for d in DELAYS if reason is None else ():
            x = designs.train(d)
            try:
                scores[d], enc_alpha[d], curve = encoding_scores(y, x, fit_rows, row_sessions, enc_folds, alphas,
                                                                 x.device)
                enc_curve[d] = curve.tolist()
            except ValueError as error:
                reason = str(error)
            del x
            if reason:
                break
    usable = [r for r in rules if r[0] == 'uniform' or reason is None]
    weights = {r: (None, dict(rule='uniform', fallback=False, nonzero_fraction=1.0)) if r[0] == 'uniform'
               else column_weights(scores, r) for r in usable}
    rows_t = None if kernel_rows is None else torch.as_tensor(np.asarray(kernel_rows), dtype=torch.long,
                                                              device=designs.device)
    kernels = {r: None for r in usable}
    cross = {r: {s: None for s in eval_splits} for r in usable}
    for d in DELAYS:
        x = designs.train(d)
        evals = {s: designs.other(s, d) for s in eval_splits}
        for rule in usable:
            w = None if weights[rule][0] is None else torch.as_tensor(weights[rule][0][d], dtype=x.dtype,
                                                                      device=x.device)
            keep = torch.arange(x.shape[1], device=x.device) if w is None else torch.nonzero(w).squeeze(1)
            for start in range(0, len(keep), CHUNK):  # Zero-weight columns contribute nothing.
                cols = keep[start:start + CHUNK]
                xb = x.index_select(1, cols)
                if rows_t is not None:
                    xb = xb.index_select(0, rows_t)
                scale = None if w is None else w.index_select(0, cols)
                if scale is not None:
                    xb.mul_(scale)
                term = xb @ xb.T
                kernels[rule] = term if kernels[rule] is None else kernels[rule].add_(term)
                for s, e in evals.items():
                    eb = e.index_select(1, cols)
                    if scale is not None:
                        eb.mul_(scale)
                    c = eb @ xb.T
                    cross[rule][s] = c if cross[rule][s] is None else cross[rule][s].add_(c)
                del xb, term
        del x, evals
    for rule in usable:
        kernels[rule] = (kernels[rule] + kernels[rule].T) * 0.5
    info = dict(encoding_reason=reason, encoding_alpha=enc_alpha, encoding_alpha_curve=enc_curve,
                positive_fraction={d: float((scores[d] > 0).mean()) for d in scores},
                weights={rule_label(r): weights[r][1] for r in usable},
                not_estimable=[rule_label(r) for r in rules if r not in usable])
    return kernels, cross, info


def cv_select(cache, rows, rules, alphas, y, row_sessions, run_ids, run_sessions, fixed, device):
    """Nested grouped CV over rules and alphas; per-rule session-macro scores (NaN when not estimable)."""
    folds, rule_text = budget_folds(row_sessions, rows, fixed)
    if not folds:
        return None, rule_text, []
    position = {r: i for i, r in enumerate(rows)}
    y_t = torch.as_tensor(y, dtype=torch.float64, device=device)
    results, log, dropped = {r: [] for r in rules}, [], set()
    for held in folds:
        start = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        held_set = set(held)
        stats_rows = [i for i, s in enumerate(row_sessions) if s not in held_set]
        fit = [i for i in rows if row_sessions[i] not in held_set]
        hold = [i for i in rows if row_sessions[i] in held_set]
        if len(fit) < 2 or not hold:
            return None, 'not estimable: a fold had too few fitting or held-out rows', log
        enc_folds = [f for f in folds if f != held]
        kernels, _, info = weighted_kernels(Designs(cache, stats_rows, device), y, fit, row_sessions, enc_folds,
                                            rules, alphas, kernel_rows=rows)
        dropped.update(r for r in rules if r not in kernels)
        fit_p, hold_p = (torch.as_tensor([position[i] for i in part], device=device) for part in (fit, hold))
        target_fit, target_hold = (y_t.index_select(0, torch.as_tensor(part, device=device)) for part in (fit, hold))
        held_runs = list(dict.fromkeys(run_ids[i] for i in hold))
        codes = torch.as_tensor([held_runs.index(run_ids[i]) for i in hold], device=device)
        for rule, k in kernels.items():
            path = KernelRidgePath(k.index_select(0, fit_p).index_select(1, fit_p), target_fit, device)
            proj = path.project(k.index_select(0, hold_p).index_select(1, fit_p))
            r = run_feature_r(path.predict_path(proj, alphas), target_hold, codes, len(held_runs))
            results[rule].append((held_runs, torch.nanmean(r, dim=-1).cpu().numpy()))
        log.append(dict(held_sessions=list(held), n_fit=len(fit), n_held=len(hold), encoding_folds=len(enc_folds),
                        seconds=time.monotonic() - start,
                        peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
                        **info))
        print(json.dumps(dict(held=list(held), seconds=round(log[-1]['seconds'], 1),
                              peak_gpu_gib=log[-1]['peak_gpu_gib'], not_estimable=info['not_estimable'])), flush=True)
        del kernels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    scores = {}
    for rule in rules:
        if rule in dropped:  # Nested encoding scores were unavailable in at least one decoder fold.
            scores[rule] = np.full(len(alphas), np.nan)
            continue
        runs = [r for held_runs, _ in results[rule] for r in held_runs]
        per_run = np.concatenate([v for _, v in results[rule]], axis=-1)
        scores[rule] = session_macro_scores(per_run, [run_sessions[r] for r in runs])
    return scores, rule_text, log


def select(rules, scores, alphas):
    """Best (rule, alpha): declared rule order, then the larger alpha, within tolerance TIE."""
    best = None
    for rule in rules:
        for k in range(len(alphas) - 1, -1, -1):
            value = scores[rule][k]
            if np.isfinite(value) and (best is None or value > best[0] + TIE):
                best = (value, rule, k)
    if best is None:
        return None
    return dict(rule=list(best[1]), label=rule_label(best[1]), alpha=float(alphas[best[2]]), cv_score=float(best[0]),
                alpha_at_grid_edge=best[2] in (0, len(alphas) - 1))


def final_fit(cache, rows, selections, y, row_sessions, fixed, alphas, splits, device, ridge_targets=None):
    """Encoding scores from all labeled rows; kernels with training statistics; predictions.

    ``y`` informs the encoding scores; the ridge fits ``ridge_targets`` (default ``y``).
    """
    folds, _ = budget_folds(row_sessions, rows, fixed)
    rules = list(dict.fromkeys(tuple(s['rule']) for s in selections.values() if s))
    designs = Designs(cache, list(range(len(row_sessions))), device)
    kernels, cross, info = weighted_kernels(designs, y, rows, row_sessions, folds or [], rules, alphas, splits,
                                            kernel_rows=rows)
    target = torch.as_tensor((y if ridge_targets is None else ridge_targets)[rows], dtype=torch.float64,
                             device=device)
    out = {}
    for arm, sel in selections.items():
        if sel is None:
            continue
        rule = tuple(sel['rule'])
        if rule not in kernels:
            raise RuntimeError(f"{arm}: final-fit encoding scores not estimable ({info['encoding_reason']})")
        path = KernelRidgePath(kernels[rule], target, device)
        out[arm] = ({s: path.predict(path.project(cross[rule][s]), sel['alpha']).cpu().numpy() for s in splits},
                    path.diagnostics)
    return out, info


def report(ctx, b, prediction, split, targets=None):
    y = ctx.split['train'].targets[ctx.budgets[b]].astype(np.float64)
    mean, std = y.mean(0), np.maximum(y.std(0), 1e-6)
    sp = ctx.split[split]
    return score(prediction, sp.targets if targets is None else targets, sp.run_ids, sp.target_trs, sp.sessions,
                 mean, std, ctx.cfg['evaluation']['exclusions'])


def source_cell(cfg, subject, arm, stage='evaluate', fraction='1'):
    """A saved cell this stage builds on, checked against the current code and configuration."""
    path = Path(cfg['output_root']).parent / 'v3w' / subject / stage / 'cells' / f'{arm}_f{fraction}.json'
    cell = json.loads(path.read_text())
    expected = dict(protocol='ssltang-v3-weights', stage=stage, subject=subject, arm=arm, code_sha256=code_sha256(),
                    config_sha256=v3.config_sha256(cfg))
    if any(cell.get(k) != v for k, v in expected.items()) or f"{cell['fraction']:g}" != fraction:
        raise SystemExit(f'{path} is not the expected cell from this code and configuration')
    if not cell.get('estimable') or not cell.get('selection'):
        raise SystemExit(f'{path} is not estimable')
    return cell, file_sha256(path)


def stage_fit(ctx, budgets, splits, stage, device):
    folder = Path(ctx.cfg['output_root']).parent / 'v3w' / ctx.subject / stage
    (folder / 'cells').mkdir(parents=True, exist_ok=True)
    (folder / 'predictions').mkdir(parents=True, exist_ok=True)
    y = ctx.split['train'].targets
    run_ids = ctx.index['train'].run_ids
    for b in budgets:
        rows = ctx.budgets[b]
        scores, rule_text, log = cv_select(ctx.cache, rows, RULES, ctx.alphas, y, ctx.row_sessions, run_ids,
                                           ctx.run_sessions, ctx.fixed, device)
        selections = {arm: (select(ARMS[arm], scores, ctx.alphas) if scores else None) for arm in ARM_ORDER}
        predictions, info = (final_fit(ctx.cache, rows, selections, y, ctx.row_sessions, ctx.fixed, ctx.alphas,
                                       splits, device) if any(selections.values()) else ({}, {}))
        for arm in ARM_ORDER:
            cell = dict(protocol='ssltang-v3-weights', stage=stage, subject=ctx.subject, arm=arm, fraction=float(b),
                        labels={k: v for k, v in ctx.subsets[b].items() if k != 'label_ids'}, cv_rule=rule_text,
                        folds=log, estimable=bool(selections[arm]), selection=selections[arm],
                        cv_scores={rule_label(r): [None if not np.isfinite(x) else float(x) for x in v]
                                   for r, v in (scores or {}).items() if r in ARMS[arm]},
                        final_fit_weights=info, code_sha256=code_sha256(), config_sha256=v3.config_sha256(ctx.cfg))
            if not selections[arm]:
                cell['reason'] = (rule_text if scores is None else
                                  'nested encoding scores not estimable in at least one decoder fold')
            if arm in predictions:
                pred, diag = predictions[arm]
                cell['final_fit'] = diag
                for s in splits:
                    cell[s] = report(ctx, b, pred[s], s)
                if 'test' in splits:
                    ev, sp = ctx.cfg['evaluation'], ctx.split['test']
                    cell['test_shift_null'] = shift_null(pred['test'], sp.targets, sp.run_ids, sp.sessions,
                                                         ctx.cfg['data']['stride'], ev['shift_min_tr'],
                                                         ev['shift_draws'], ev['shift_seed'])
                np.savez_compressed(folder / 'predictions' / f'{arm}_f{b}.npz',
                                    **{s: v.astype(np.float16) for s, v in pred.items()})
                print(f"{ctx.subject} {stage} {arm} f{b}: {selections[arm]['label']} alpha={selections[arm]['alpha']:.3g} "
                      f"cv={selections[arm]['cv_score']:.4f} " +
                      ' '.join(f"{s}={cell[s]['session_macro_feature_correlation']:.4f}" for s in splits), flush=True)
            save_json(cell, folder / 'cells' / f'{arm}_f{b}.json')
    save_json(dict(ctx.provenance(stage), weights_code_sha256=code_sha256()), folder / 'provenance.json')
    return folder


def stage_residualized(ctx, lock, device):
    """Speech-rate-residualized test endpoint at 100% for W0 and the advanced arm (§12).

    As in the v3 controls: the selected rule, alpha and weights (encoding scores
    from the original training targets) are fixed; the ridge is refit on
    training targets minus the training-fit nuisance prediction.
    """
    from src.transcripts import nuisance_features, run_words
    folder = Path(ctx.cfg['output_root']).parent / 'v3w' / ctx.subject / 'evaluate'
    rows, y = ctx.budgets['1'], ctx.split['train'].targets
    sources = {arm: source_cell(ctx.cfg, ctx.subject, arm) for arm in ('W0', lock['advanced_arm'])}
    words = {s: {r['run_id']: run_words(r, ctx.manifest_path) for r in ctx.index[s].runs} for s in ('train', 'test')}
    x = {s: nuisance_features(ctx.index[s], words[s])[0] for s in ('train', 'test')}
    nuisance, info = v3.feature_ridge_cv(ctx, x['train'], rows, y, {'train': x['train'], 'test': x['test']})
    residual_train, residual_test = y - nuisance['train'], ctx.split['test'].targets - nuisance['test']
    selections = {arm: cell['selection'] for arm, (cell, _) in sources.items()}
    predictions, _ = final_fit(ctx.cache, rows, selections, y, ctx.row_sessions, ctx.fixed, ctx.alphas, ('test',),
                               device, ridge_targets=residual_train)
    out = dict(protocol='ssltang-v3-weights', stage='residualized', subject=ctx.subject, nuisance=info,
               selections=selections, source_cells={arm: sha for arm, (_, sha) in sources.items()},
               rule='targets minus training-fit nuisance (position + oracle speech rate); selected rule, alpha and '
                    'weights fixed; ridge refit on residual training targets',
               code_sha256=code_sha256(), config_sha256=v3.config_sha256(ctx.cfg),
               test={arm: report(ctx, '1', pred['test'], 'test', targets=residual_test)
                     ['session_macro_feature_correlation'] for arm, (pred, _) in predictions.items()})
    save_json(out, folder / 'residualized.json')
    print(json.dumps(dict(subject=ctx.subject, residualized_test=out['test'])), flush=True)


def stage_dense(cfg, lock, device):
    """Dense clock: W0 and the advanced arm's selected rule fixed; alpha re-selected by nested CV."""
    import run_v3_dense as dense
    ctx = dense.DenseContext(cfg, device)
    source, source_sha = source_cell(cfg, ctx.subject, lock['advanced_arm'])
    fixed_rule = tuple(source['selection']['rule'])
    rules = [('uniform', None), fixed_rule]
    rows = list(range(len(ctx.index['train'])))
    y = ctx.split['train'].targets
    scores, rule_text, log = cv_select(ctx.cache, rows, rules, ctx.alphas, y, ctx.row_sessions,
                                       ctx.index['train'].run_ids, ctx.run_sessions, ctx.fixed, device)
    picks = {name: select([rule], scores, ctx.alphas) if scores else None
             for name, rule in (('W0', rules[0]), (lock['advanced_arm'], fixed_rule))}
    if not all(picks.values()):
        raise SystemExit(f'Dense CV not estimable: {rule_text}')
    predictions, info = final_fit(ctx.cache, rows, picks, y, ctx.row_sessions, ctx.fixed, ctx.alphas, ('val', 'test'),
                                  device)
    folder = Path(cfg['output_root']).parent / 'v3w' / ctx.subject / 'dense'
    folder.mkdir(parents=True, exist_ok=True)
    mean, std = y.mean(0), np.maximum(y.std(0), 1e-6)
    for name, (pred, diag) in predictions.items():
        rule = tuple(picks[name]['rule'])
        cell = dict(protocol='ssltang-v3-weights', stage='dense', subject=ctx.subject, arm=name, selection=picks[name],
                    source_cell=dict(arm=lock['advanced_arm'], sha256=source_sha), cv_rule=rule_text,
                    cv_scores=[None if not np.isfinite(x) else float(x) for x in scores[rule]], folds=log,
                    final_fit=diag, final_fit_weights=info, n_train_targets=len(rows), code_sha256=code_sha256(),
                    config_sha256=v3.config_sha256(cfg),
                    targets={s: i.signature() for s, i in ctx.index.items()})
        for sp_name in ('val', 'test'):
            sp = ctx.split[sp_name]
            cell[sp_name] = score(pred[sp_name], sp.targets, sp.run_ids, sp.target_trs, sp.sessions, mean, std,
                                  cfg['evaluation']['exclusions'])
        ev, sp = cfg['evaluation'], ctx.split['test']
        cell['test_shift_null'] = shift_null(pred['test'], sp.targets, sp.run_ids, sp.sessions, cfg['data']['stride'],
                                             ev['shift_min_tr'], ev['shift_draws'], ev['shift_seed'])
        save_json(cell, folder / f'{name}_dense.json')
        print(name, picks[name], {k: round(cell[k]['session_macro_feature_correlation'], 4) for k in ('val', 'test')},
              flush=True)


def stage_lock(cfg):
    """Apply the §12 development gates to every registered cell; refuse incomplete or stale evidence."""
    root = Path(cfg['output_root']).parent / 'v3w'
    val, evidence = {}, {}
    for s in REGISTERED:
        for arm in ARM_ORDER:
            for b in GATE_FRACTIONS:
                path = root / s / 'develop' / 'cells' / f'{arm}_f{b}.json'
                if not path.exists():
                    raise SystemExit(f'Missing development cell {path}')
                try:
                    cell, sha = source_cell(cfg, s, arm, stage='develop', fraction=b)
                except SystemExit as error:
                    raise SystemExit(f'{error}; rerun develop') from None
                value = (cell.get('val') or {}).get('session_macro_feature_correlation')
                if value is None or not np.isfinite(value):
                    raise SystemExit(f'{path} has no finite validation score; the registered gate cannot be computed')
                val[(s, arm, b)] = float(value)
                evidence[f'{s}/{arm}_f{b}'] = sha
    gates = {}
    for arm in ('W1', 'W2'):
        d1 = np.array([val[(s, arm, '1')] - val[(s, 'W0', '1')] for s in REGISTERED])
        d2 = np.array([val[(s, arm, '0.1')] - val[(s, 'W0', '0.1')] for s in REGISTERED])
        gates[arm] = dict(delta_100=d1.round(5).tolist(), mean_100=float(d1.mean()), delta_10=d2.round(5).tolist(),
                          mean_10=float(d2.mean()), passed=bool(d1.mean() >= 0.005 and (d1 > 0).all()
                                                                and d2.mean() >= 0 and (d2 >= -0.005).all()))
    passing = [a for a in ('W1', 'W2') if gates[a]['passed']]
    advanced = max(passing, key=lambda a: gates[a]['mean_100']) if passing else None
    lock = dict(protocol='ssltang-v3-weights', created_at=datetime.now(timezone.utc).isoformat(),
                advanced_arm=advanced, gates=gates, code_sha256=code_sha256(), config_sha256=v3.config_sha256(cfg),
                v3_lock_code_sha256=v3.code_sha256(), subjects=list(REGISTERED), evidence=evidence,
                decision=('advance ' + advanced) if advanced else 'negative result: no weighting arm passed; stop')
    save_json(lock, LOCK)
    print(json.dumps(lock, indent=1))
    return lock


def verify_lock(cfg):
    if not LOCK.exists():
        raise SystemExit('No §12 lock; run the lock stage first')
    lock = json.loads(LOCK.read_text())
    if lock['code_sha256'] != code_sha256() or lock['config_sha256'] != v3.config_sha256(cfg):
        raise SystemExit('Code or configuration changed since the §12 lock')
    if not lock['advanced_arm']:
        raise SystemExit('§12 gates did not advance a weighting arm; no locked evaluation is registered')
    return lock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['develop', 'lock', 'evaluate', 'dense'], required=True)
    p.add_argument('--config', default='configs/v3/structured_ridge.yaml')
    p.add_argument('--set', action='append', default=[])
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    torch.backends.cuda.matmul.allow_tf32 = False
    v3.verify_lock(cfg)  # The v3 code this stage builds on is unchanged.
    if a.stage == 'lock':
        stage_lock(cfg)
        return
    lock = verify_lock(cfg) if a.stage in ('evaluate', 'dense') else None
    if a.stage == 'dense':
        stage_dense(cfg, lock, a.device)
        return
    ctx = v3.Context(cfg, a.device)
    if a.stage == 'develop':
        stage_fit(ctx, cfg['develop_fractions'], ('val',), 'develop', a.device)
    else:
        stage_fit(ctx, [f'{f:g}' for f in cfg['labels']['fractions']], ('val', 'test'), 'evaluate', a.device)
        stage_residualized(ctx, lock, a.device)


if __name__ == '__main__':
    main()
