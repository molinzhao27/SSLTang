#!/usr/bin/env python3
"""PROTOCOL_V3 §12 locked evaluation in subjects outside UTS01–03.

UTS09/UTS10 are exposed (scored in §11); UTS04–UTS08 are the registered fresh
confirmatory subjects. Both locks (v3 and §12) are verified against unchanged
code first. The §4 fold *rule* is then applied to the subject's own training
sessions, as in §11, and the derived folds are recorded. The locked §12 stage
functions run unmodified.
"""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v3 as v3
import run_v3_weights as w
from src.runtime import load_config, save_json
from src.structured_ridge import fixed_folds
from src.targets import TargetIndex

EXPOSED = ('UTS09', 'UTS10')
FRESH = ('UTS04', 'UTS05', 'UTS06', 'UTS07', 'UTS08')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['evaluate', 'dense'], required=True)
    p.add_argument('--subject', choices=EXPOSED + FRESH, required=True)
    p.add_argument('--config', default='configs/v3/structured_ridge.yaml')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(a.config, [f'data.manifest=data/semantic/{a.subject}/semantic_manifest.json'])
    torch.backends.cuda.matmul.allow_tf32 = False
    v3_lock = v3.verify_lock(cfg)  # Before the fold rule is applied to this subject.
    lock = w.verify_lock(cfg)
    data = cfg['data']
    train = TargetIndex(data['manifest'], 'train', window=data['window'], stride=data['stride'],
                        phase=data['target_phase'], cutoffs=data['cutoffs'])
    derived = fixed_folds([train.sessions[r] for r in train.run_ids])
    v3.PROTOCOL_FOLDS = derived  # Same rule as §4, applied to this subject's sessions.
    out = Path(cfg['output_root']).parent / 'v3w' / a.subject
    out.mkdir(parents=True, exist_ok=True)
    save_json(dict(subject=a.subject, role='fresh confirmatory' if a.subject in FRESH else 'exposed',
                   rule='sha256("ssltang-v3-folds:<session>") order, round-robin into 5', folds=derived,
                   v3_lock_code_sha256=v3_lock['code_sha256'], v3w_lock_code_sha256=lock['code_sha256']),
              out / 'replication_folds.json')
    if a.stage == 'dense':
        w.stage_dense(cfg, lock, a.device)
        return
    build = not (Path(cfg['cache_root']) / a.subject / 'train.meta.json').exists()
    ctx = v3.Context(cfg, a.device, build_cache=build)
    w.stage_fit(ctx, [f'{f:g}' for f in cfg['labels']['fractions']], ('val', 'test'), 'evaluate', a.device)
    w.stage_residualized(ctx, lock, a.device)
    print(json.dumps(dict(subject=a.subject, folds=derived, cache_built=build)), flush=True)


if __name__ == '__main__':
    main()
