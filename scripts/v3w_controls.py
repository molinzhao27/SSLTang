#!/usr/bin/env python3
"""Exploratory brain-input controls for the locked §12 advanced arm (after the locked evaluation).

Not part of the §12 registration or its code fingerprint; mirrors the v3 §6 controls:
  (a) disrupted training: training targets rotated within each run by floor(n/2) rows, then the
      whole advanced-arm pipeline (encoding scores, weights, ridge) refit with its locked rule and alpha;
  (b) wrong-run input: each test run scored against the next test run's saved predictions.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v3 as v3
import run_v3_weights as w
from src.runtime import load_config, save_json
from src.structured_ridge import fixed_folds
from src.targets import TargetIndex


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', required=True)
    p.add_argument('--config', default='configs/v3/structured_ridge.yaml')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(a.config, [f'data.manifest=data/semantic/{a.subject}/semantic_manifest.json'])
    torch.backends.cuda.matmul.allow_tf32 = False
    v3.verify_lock(cfg)
    lock = w.verify_lock(cfg)
    if a.subject not in w.REGISTERED:  # §11 fold rule for other subjects, as in run_v3w_replicate.py.
        data = cfg['data']
        train = TargetIndex(data['manifest'], 'train', window=data['window'], stride=data['stride'],
                            phase=data['target_phase'], cutoffs=data['cutoffs'])
        v3.PROTOCOL_FOLDS = fixed_folds([train.sessions[r] for r in train.run_ids])
    ctx = v3.Context(cfg, a.device)
    arm, b = lock['advanced_arm'], '1'
    folder = Path(cfg['output_root']).parent / 'v3w' / ctx.subject
    cell, sha = w.source_cell(cfg, ctx.subject, arm)
    rows, y = ctx.budgets[b], ctx.split['train'].targets
    out = dict(protocol='ssltang-v3-weights', stage='exploratory-controls', subject=ctx.subject, arm=arm,
               selection=cell['selection'], source_cell=sha, real_test=cell['test']['session_macro_feature_correlation'],
               note='exploratory, after the locked evaluation; not part of the §12 registration')
    rotated, shifts = y.copy(), {}
    run_ids = ctx.index['train'].run_ids
    for run in dict.fromkeys(run_ids.tolist()):
        idx = np.flatnonzero(run_ids == run)
        shifts[run] = len(idx) // 2
        rotated[idx] = np.roll(y[idx], shifts[run], axis=0)
    predictions, info = w.final_fit(ctx.cache, rows, {arm: cell['selection']}, rotated, ctx.row_sessions, ctx.fixed,
                                    ctx.alphas, ('test',), a.device)
    out['disrupted_training'] = dict(min_shift_rows=min(shifts.values()), weights=info['weights'],
                                     positive_fraction=info['positive_fraction'],
                                     test=w.report(ctx, b, predictions[arm][0]['test'], 'test')
                                     ['session_macro_feature_correlation'])
    real = np.load(folder / 'evaluate' / 'predictions' / f'{arm}_f{b}.npz')['test'].astype(np.float64)
    test = ctx.split['test']
    names = list(dict.fromkeys(test.run_ids.tolist()))
    wrong = np.empty_like(real)
    for i, run in enumerate(names):
        mine, other = np.flatnonzero(test.run_ids == run), np.flatnonzero(test.run_ids == names[(i + 1) % len(names)])
        wrong[mine] = real[other][np.arange(len(mine)) % len(other)]
    out['wrong_run_input'] = dict(test=w.report(ctx, b, wrong, 'test')['session_macro_feature_correlation'],
                                  rule='next test run in run-ID order, cycled to equal length')
    save_json(out, folder / 'controls.json')
    print(json.dumps({k: v for k, v in out.items() if k != 'selection'}, indent=1), flush=True)


if __name__ == '__main__':
    main()
