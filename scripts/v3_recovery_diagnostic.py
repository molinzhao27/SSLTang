#!/usr/bin/env python3
"""G0(b) diagnostic: separate float16 storage and float32 arithmetic in Phase V.

Recomputes Phase V's all-voxel predictions with the original feature solver
(GramRidgePath) in float32 and float64 from the v3 cache, and compares them
with the v3 kernel path and with Phase V's float16-stored predictions.
"""
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v3 as v3
from src.kernel_ridge import GramRidgePath
from src.runtime import load_config, save_json


def main():
    cfg = load_config('configs/v3/structured_ridge.yaml', ['data.manifest=data/semantic/UTS01/semantic_manifest.json'])
    torch.backends.cuda.matmul.allow_tf32 = False
    ctx = v3.Context(cfg, 'cuda')
    kernels, cross, _ = ctx.builder.build(list(range(len(ctx.row_sessions))), ('val',))
    load = lambda split: np.concatenate([np.asarray(np.load(ctx.cache[split][d], mmap_mode='r'), dtype=np.float32)
                                         for d in (1, 2, 3)], axis=1)
    train_all, val_x = load('train'), load('val')
    mean, std = train_all.mean(0, dtype=np.float64), train_all.std(0, dtype=np.float64)
    std = np.where(std > 1e-8 * max(1.0, float(np.abs(train_all).max())), std, 1.0)
    train_all = ((train_all - mean) / std).astype(np.float32)
    val_x = ((val_x - mean) / std).astype(np.float32)
    report = {}
    for b in ('1', '0.1'):
        cell = json.loads(Path(f'results/v2/phase_v/cells/V1_voxels_seed0_c3_fir_f{b}.json').read_text())
        saved = np.load(f'results/v2/phase_v/predictions/V1_voxels_seed0_c3_fir_f{b}.npz')['val'].astype(np.float64)
        alpha, rows = cell['selection']['alpha'], ctx.budgets[b]
        ours = v3.fit_predict(ctx, kernels, cross, rows, dict(candidate=v3.BASELINE, alpha=alpha), ('val',))[0]['val']
        rms = float(np.sqrt(np.mean(ours ** 2)))
        refs = {}
        for name, dtype in (('float32', torch.float32), ('float64', torch.float64)):
            g = GramRidgePath(torch.as_tensor(train_all[rows], dtype=dtype, device='cuda'),
                              ctx.split['train'].targets[rows], 'cuda')
            refs[name] = g.predict(g.project(torch.as_tensor(val_x, dtype=dtype, device='cuda')), alpha).cpu().numpy()
            del g
            torch.cuda.empty_cache()
        half_ulp = lambda v: 0.5 * np.spacing(np.abs(v).astype(np.float16)).astype(np.float64)
        # Phase V's own path: per-voxel normalization over all training TRs, then
        # target-row standardization, float32 features and float32 solver.
        if 'two_stage' not in locals():
            vn = np.load('results/v2/phase_v/voxel_normalization.npz')
            stage1 = lambda split: np.concatenate([((np.asarray(np.load(ctx.cache[split][d], mmap_mode='r'),
                                                                    dtype=np.float32) - vn['mean']) / vn['std'])
                                                   .astype(np.float32) for d in (1, 2, 3)], axis=1)
            t1, v1 = stage1('train'), stage1('val')
            m2, s2 = t1.mean(0, dtype=np.float64), t1.std(0, dtype=np.float64)
            s2 = np.where(s2 > 1e-8 * max(1.0, float(np.abs(t1).max())), s2, 1.0)
            two_stage = (((t1 - m2) / s2).astype(np.float32), ((v1 - m2) / s2).astype(np.float32))
            del t1, v1
        g = GramRidgePath(torch.as_tensor(two_stage[0][rows], device='cuda'), ctx.split['train'].targets[rows], 'cuda')
        original = g.predict(g.project(torch.as_tensor(two_stage[1], device='cuda')), alpha).cpu().numpy()
        del g
        torch.cuda.empty_cache()
        spacing = np.spacing(np.abs(original).astype(np.float16)).astype(np.float64)
        ulps = np.abs(saved - original) / spacing
        worst = np.unravel_index(int(np.argmax(ulps)), ulps.shape)
        report[b] = dict(
            phase_v_two_stage_float32_vs_stored_within_half_ulp=float((ulps <= 0.5 * 1.0001).mean()),
            phase_v_two_stage_float32_vs_stored_max_ulps=float(ulps.max()),
            worst_entry=dict(target_row=int(worst[0]), output=int(worst[1]), stored=float(saved[worst]),
                             recomputed=float(original[worst])),
            v3_vs_phase_v_two_stage_max_rel=float(np.abs(ours - original).max() / rms),
            prediction_rms=rms,
            v3_vs_feature_float64_max_rel=float(np.abs(ours - refs['float64']).max() / rms),
            v3_vs_feature_float32_max_rel=float(np.abs(ours - refs['float32']).max() / rms),
            stored_float16_vs_feature_float32_within_half_ulp=float(
                (np.abs(saved - refs['float32']) <= half_ulp(refs['float32']) * 1.0001).mean()),
            v3_vs_stored_max_rel=float(np.abs(ours - saved).max() / rms))
        print(b, json.dumps(report[b]), flush=True)
    save_json(report, 'results/v3/UTS01/recovery_diagnostic.json')


if __name__ == '__main__':
    main()
