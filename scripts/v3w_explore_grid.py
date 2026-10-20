#!/usr/bin/env python3
"""Exploratory, validation-only sweep beyond the registered §12 weighting grid.

The locked W1 selected the grid edge (gamma = 2) in nearly every cell and W2 the
smallest q (10%). This asks whether more concentrated weighting would help,
using the same nested grouped CV and final-fit code paths on the stride-4 clock
at 100% labels. Validation only: no test score is computed. Results are
hypothesis-generating for a future registration, not part of §12.
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
from src.encoding_weights import rule_label
from src.runtime import load_config, save_json

RULES = (('uniform', None), ('soft', 1.0), ('soft', 2.0), ('soft', 3.0), ('soft', 4.0), ('soft', 6.0),
         ('hard', 0.10), ('hard', 0.05), ('hard', 0.02), ('hard', 0.01))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', required=True, choices=w.REGISTERED)
    p.add_argument('--config', default='configs/v3/structured_ridge.yaml')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(a.config, [f'data.manifest=data/semantic/{a.subject}/semantic_manifest.json'])
    torch.backends.cuda.matmul.allow_tf32 = False
    v3.verify_lock(cfg)
    w.verify_lock(cfg)  # The shared code paths are the locked ones.
    ctx = v3.Context(cfg, a.device)
    rows, y = ctx.budgets['1'], ctx.split['train'].targets
    scores, rule_text, log = w.cv_select(ctx.cache, rows, list(RULES), ctx.alphas, y, ctx.row_sessions,
                                         ctx.index['train'].run_ids, ctx.run_sessions, ctx.fixed, a.device)
    picks = {rule_label(r): w.select([r], scores, ctx.alphas) for r in RULES}
    predictions, info = w.final_fit(ctx.cache, rows, picks, y, ctx.row_sessions, ctx.fixed, ctx.alphas, ('val',),
                                    a.device)
    out = dict(protocol='ssltang-v3-weights', stage='exploratory-grid', subject=ctx.subject, fraction=1.0,
               note='exploratory, validation only; not part of the §12 registration', cv_rule=rule_text,
               weights=info['weights'], encoding_alpha=info['encoding_alpha'], folds=log, rules={})
    for label, pick in picks.items():
        val = w.report(ctx, '1', predictions[label][0]['val'], 'val')
        out['rules'][label] = dict(pick, val=val['session_macro_feature_correlation'],
                                   cv_curve=[None if not np.isfinite(x) else float(x)
                                             for x in scores[tuple(pick['rule'])]])
        print(f"{ctx.subject} {label}: alpha={pick['alpha']:.3g} cv={pick['cv_score']:.4f} "
              f"val={out['rules'][label]['val']:.4f}", flush=True)
    save_json(out, Path(cfg['output_root']).parent / 'v3w' / ctx.subject / 'explore_grid.json')


if __name__ == '__main__':
    main()
