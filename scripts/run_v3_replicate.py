#!/usr/bin/env python3
"""PROTOCOL_V3 §11: the locked v3 procedure on fresh subjects (UTS09, UTS10).

The lock is verified against the unchanged v3 code first. The registered fold
*rule* (sha256 order, round-robin into five folds) is then applied to the new
subject's own training sessions; the derived folds are recorded. Stages run the
locked evaluate, controls and dense code paths without modification.
"""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v3 as v3
import run_v3_dense as dense
from src.runtime import load_config, save_json
from src.structured_ridge import BASELINE, fixed_folds
from src.targets import TargetIndex

REGISTERED = ('UTS09', 'UTS10')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['cache', 'evaluate', 'controls', 'dense'], required=True)
    p.add_argument('--subject', choices=REGISTERED, required=True)
    p.add_argument('--config', default='configs/v3/structured_ridge.yaml')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(a.config, [f'data.manifest=data/semantic/{a.subject}/semantic_manifest.json'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lock = v3.verify_lock(cfg)  # Unchanged locked code and configuration.
    data = cfg['data']
    train = TargetIndex(data['manifest'], 'train', window=data['window'], stride=data['stride'],
                        phase=data['target_phase'], cutoffs=data['cutoffs'])
    derived = fixed_folds([train.sessions[r] for r in train.run_ids])
    v3.PROTOCOL_FOLDS = derived  # Same rule as §4, applied to this subject's sessions.
    out = Path(cfg['output_root']) / a.subject
    out.mkdir(parents=True, exist_ok=True)
    save_json(dict(subject=a.subject, rule='sha256("ssltang-v3-folds:<session>") order, round-robin into 5',
                   folds=derived, lock_code_sha256=lock['code_sha256']), out / 'replication_folds.json')
    if a.stage == 'dense':
        ctx = dense.DenseContext(cfg, a.device)
        ctx.out.mkdir(parents=True, exist_ok=True)
        cell = json.loads((out / 'evaluate' / 'cells' / f"{lock['advanced_arm']}_f1.json").read_text())
        structure = (float(cell['selection']['candidate'][0]), tuple(cell['selection']['candidate'][1]))
        for name, result in dense.dense_fit(ctx, {'V0': BASELINE, lock['advanced_arm']: structure}, False).items():
            save_json(result, ctx.out / f'{name}_dense.json')
            print(name, result['candidate'], result['alpha'],
                  {s: round(result[s]['session_macro_feature_correlation'], 4) for s in ('val', 'test')}, flush=True)
        return
    ctx = v3.Context(cfg, a.device, build_cache=a.stage == 'cache')
    if a.stage == 'cache':
        print(json.dumps(dict(subject=ctx.subject, n_voxels=ctx.n_voxels, groups=ctx.groups['n_groups'],
                              coordinates_verified=ctx.groups['coordinates_verified'], folds=derived)))
    elif a.stage == 'evaluate':
        v3.stage_fit(ctx, [f'{f:g}' for f in cfg['labels']['fractions']], ('val', 'test'), 'evaluate')
    else:
        v3.stage_controls(ctx, lock)


if __name__ == '__main__':
    main()
