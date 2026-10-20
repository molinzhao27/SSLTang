#!/usr/bin/env python3
"""Protocol v3 §7: dense training clock as a separate data-budget factor.

Training targets at every eligible TR phase (stride 1, same cutoff
eligibility); validation and test targets, target definition and the 6-s
input cutoff are unchanged. V0 and the advanced arm's locked structure are
refit with alpha re-selected by the same session-grouped CV. Runs only after
the locked evaluation exists, and never changes the locked code.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v3 as v3
from src.evaluation.endpoints import shift_null
from src.multikernel_ridge import KernelRidgePath, combine
from src.probe import Split
from src.runtime import load_config, save_json
from src.structured_ridge import BASELINE, aggregate, budget_folds, coefficients, label, score_fold, select, _block
from src.targets import TargetIndex
from src.voxel_kernels import KernelBuilder, cache_delay_rows, membership_groups, verify_cache


class DenseContext:
    """Like run_v3.Context, with a stride-1 training index and its own cache."""

    def __init__(self, cfg, device):
        data = cfg['data']
        self.cfg, self.device = cfg, torch.device(device)
        self.manifest_path = data['manifest']
        common = dict(window=data['window'], phase=data['target_phase'], cutoffs=data['cutoffs'])
        self.index = {'train': TargetIndex(data['manifest'], 'train', stride=1, phase=0,
                                           window=data['window'], cutoffs=data['cutoffs'])}
        for s in ('val', 'test'):
            self.index[s] = TargetIndex(data['manifest'], s, stride=data['stride'], **common)
        self.subject = self.index['train'].runs[0]['subject']
        self.split = {s: Split(self.index[s], 'cpu') for s in self.index}
        root = Path(cfg['cache_root'])
        train_paths, n_voxels = cache_delay_rows(self.index['train'], cfg['raw_root'],
                                                 root.parent / (root.name + '_dense') / self.subject)
        self.cache = {'train': train_paths}
        for s in ('val', 'test'):
            self.cache[s], _ = verify_cache(self.index[s], cfg['raw_root'], root / self.subject)
        self.n_voxels = n_voxels
        self.groups = membership_groups(self.index['train'].manifest, self.manifest_path, n_voxels)
        self.builder = KernelBuilder(self.cache, self.groups, device)
        train = self.index['train']
        self.row_sessions = [train.sessions[r] for r in train.run_ids]
        self.run_sessions = dict(train.sessions)
        self.fixed = v3.fixed_folds(self.row_sessions)
        self.alphas = np.logspace(*cfg['alphas'])
        self.out = Path(cfg['output_root']) / self.subject / 'dense'


def dense_fit(ctx, arms, profile):
    """CV (kernels built once per fold) and final fits for {arm: candidate}."""
    rows = list(range(len(ctx.index['train'])))
    folds, rule = budget_folds(ctx.row_sessions, rows, ctx.fixed)
    run_ids = ctx.index['train'].run_ids
    y = ctx.split['train'].targets
    candidates = list(dict.fromkeys(arms.values()))
    results, log = [], []
    for held in folds:
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        held_set = set(held)
        fit = [i for i in rows if ctx.row_sessions[i] not in held_set]
        held_rows = [i for i in rows if ctx.row_sessions[i] in held_set]
        kernels, _, _ = ctx.builder.build(fit)  # Inner-session statistics: every inner row.
        held_runs = list(dict.fromkeys(run_ids[i] for i in held_rows))
        codes = np.array([held_runs.index(run_ids[i]) for i in held_rows])
        out = score_fold(kernels, fit, held_rows, y, codes, len(held_runs), ctx.alphas, ctx.device, candidates)
        results.append((held_runs, {c: {k: v for k, v in out[c].items() if k != 'per_output'} for c in candidates}))
        log.append(dict(held_sessions=list(held), n_fit=len(fit), n_held=len(held_rows),
                        seconds=time.monotonic() - start, peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                        eigen={label(c): out[c]['diagnostics'] for c in candidates}))
        print(json.dumps(dict(subject=ctx.subject, **{k: v for k, v in log[-1].items() if k != 'eigen'})), flush=True)
        del kernels
        torch.cuda.empty_cache()
        if profile:
            return dict(profile=log)
    scores, _ = aggregate(results, ctx.run_sessions, ctx.alphas)
    kernels, cross, _ = ctx.builder.build(rows, ('val', 'test'))
    rows_t = torch.as_tensor(np.asarray(rows), device=ctx.device)
    cells = {}
    for arm, candidate in arms.items():
        best = None
        for k in range(len(ctx.alphas) - 1, -1, -1):  # Ties to the larger alpha.
            value = scores[candidate][k]
            if np.isfinite(value) and (best is None or value > best[0] + v3.TIE):
                best = (value, k)
        alpha = float(ctx.alphas[best[1]])
        coef = coefficients(*candidate)
        path = KernelRidgePath(combine(_block(kernels, rows_t, rows_t), coef), y, ctx.device)
        cell = dict(protocol='ssltang-v3', stage='dense', subject=ctx.subject, arm=arm, candidate=label(candidate),
                    alpha=alpha, cv_score=float(best[0]), cv_rule=rule, folds=log, n_train_targets=len(rows),
                    cv_scores=[None if not np.isfinite(x) else float(x) for x in scores[candidate]],
                    final_fit=path.diagnostics)
        mean, std = y.mean(0), np.maximum(y.std(0), 1e-6)
        from src.evaluation.endpoints import score
        for s in ('val', 'test'):
            projected = path.project(combine({k: v.index_select(1, rows_t) for k, v in cross[s].items()}, coef))
            prediction = path.predict(projected, alpha).cpu().numpy()
            sp = ctx.split[s]
            cell[s] = score(prediction, sp.targets, sp.run_ids, sp.target_trs, sp.sessions, mean, std,
                            ctx.cfg['evaluation']['exclusions'])
            if s == 'test':
                ev = ctx.cfg['evaluation']
                cell['test_shift_null'] = shift_null(prediction, sp.targets, sp.run_ids, sp.sessions,
                                                     ctx.cfg['data']['stride'], ev['shift_min_tr'],
                                                     ev['shift_draws'], ev['shift_seed'])
        cells[arm] = cell
        del path
    return cells


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/v3/structured_ridge.yaml')
    p.add_argument('--set', action='append', default=[])
    p.add_argument('--device', default='cuda')
    p.add_argument('--profile', action='store_true', help='Time one CV fold of dense V0 and stop')
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    torch.backends.cuda.matmul.allow_tf32 = False
    lock = v3.verify_lock(cfg)
    ctx = DenseContext(cfg, a.device)
    ctx.out.mkdir(parents=True, exist_ok=True)
    evaluate = Path(cfg['output_root']) / ctx.subject / 'evaluate' / 'cells'
    if not (evaluate / f"{lock['advanced_arm']}_f1.json").exists():
        raise SystemExit('Run the locked evaluation before the dense factor')
    if a.profile:
        save_json(dense_fit(ctx, {'V0': BASELINE}, True), ctx.out / 'profile.json')
        return
    advanced = json.loads((evaluate / f"{lock['advanced_arm']}_f1.json").read_text())['selection']['candidate']
    structure = (float(advanced[0]), tuple(advanced[1]))
    for name, cell in dense_fit(ctx, {'V0': BASELINE, lock['advanced_arm']: structure}, False).items():
        save_json(cell, ctx.out / f'{name}_dense.json')
        print(name, cell['candidate'], cell['alpha'], {s: round(cell[s]['session_macro_feature_correlation'], 4)
                                                       for s in ('val', 'test')}, flush=True)


if __name__ == '__main__':
    main()
