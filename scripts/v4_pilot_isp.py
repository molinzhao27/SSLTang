#!/usr/bin/env python3
"""Feasibility pilot (no decoding endpoint): label-free inter-subject predictability (ISP) maps.

For a target subject, each voxel's response to the target's TRAINING stories shared by all ten
ds003020 subjects is predicted from the other subjects' responses at the same story time
(top-K principal components per reference subject, lags -1..+1 TR), with story-grouped
out-of-fold ridge. No transcript is used. The ISP map is then compared with the target's
label-derived encoding-predictability map (the §12 W1 scores from all labeled training pairs).
Only training data are read; validation and test stories are never loaded.
"""
import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.runtime import load_config, save_json

SUBJECTS = [f'UTS{i:02d}' for i in range(1, 11)]


def runs_by_story(subject):
    m = json.loads(Path(f'data/semantic/{subject}/semantic_manifest.json').read_text())
    return {r['story']: r for r in m['runs']}


def load_story(run, raw_root):
    with h5py.File(Path(raw_root) / run['source_key']) as handle:
        x = np.nan_to_num(np.asarray(handle['data'], dtype=np.float32))
    x -= x.mean(0)
    x /= np.maximum(x.std(0), 1e-6)  # Per-run z-score: unsupervised, within run.
    return x


def lagged(z, lags):
    out = []
    for lag in lags:
        shifted = np.zeros_like(z)
        if lag > 0:
            shifted[lag:] = z[:-lag]
        elif lag < 0:
            shifted[:lag] = z[-lag:]
        else:
            shifted = z
        out.append(shifted)
    return np.concatenate(out, axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', default='UTS01')
    p.add_argument('--components', type=int, default=100)
    p.add_argument('--config', default='configs/v3/structured_ridge.yaml')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(a.config, [f'data.manifest=data/semantic/{a.target}/semantic_manifest.json'])
    runs = {s: runs_by_story(s) for s in SUBJECTS}
    shared = sorted(set.intersection(*(set(r) for r in runs.values())))
    stories = [st for st in shared if runs[a.target][st]['split'] == 'train']
    refs = [s for s in SUBJECTS if s != a.target]
    print(json.dumps(dict(target=a.target, training_shared_stories=len(stories))), flush=True)
    target = [load_story(runs[a.target][st], cfg['raw_root']) for st in stories]
    groups = np.concatenate([np.full(len(x), i % 5) for i, x in enumerate(target)])
    y = np.concatenate(target)
    feats = []
    for s in refs:
        z = np.concatenate([load_story(runs[s][st], cfg['raw_root']) for st in stories])
        zt = torch.as_tensor(z, dtype=torch.float32, device=a.device)
        u, sv, _ = torch.linalg.svd(zt, full_matrices=False)  # Reference-only, label-free.
        pcs = (u[:, :a.components] * sv[:a.components]).cpu().numpy()
        bounds = np.cumsum([0] + [len(x) for x in target])
        feats.append(np.concatenate([lagged(pcs[bounds[i]:bounds[i + 1]], (-1, 0, 1)) for i in range(len(target))]))
        del zt, u
        print(json.dumps(dict(reference=s, rows=len(z))), flush=True)
    features = np.concatenate(feats, axis=1)
    from src.encoding_weights import encoding_scores
    alphas = np.logspace(0, 6, 13)
    labels = [str(g) for g in groups]
    design = torch.as_tensor(y, dtype=torch.float64, device=a.device)
    isp, alpha, curve = encoding_scores(features, design, list(range(len(y))), labels,
                                        [(str(g),) for g in range(5)], alphas, a.device)
    del design
    out = dict(target=a.target, stories=stories, n_tr=int(len(y)), n_voxels=int(y.shape[1]), components=a.components,
               references=refs, alpha=alpha, alpha_curve=curve.tolist(),
               isp_quantiles={q: float(np.quantile(isp, q)) for q in (0.5, 0.9, 0.98, 0.99)},
               isp_positive=float((isp > 0).mean()))
    folder = Path('results/v4_pilot')
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f'{a.target}_isp.npy', isp.astype(np.float32))
    # Label-derived encoding scores from all labeled training pairs (§12 code path), training data only.
    import run_v3 as v3
    import run_v3_weights as w
    from src.structured_ridge import budget_folds
    ctx = v3.Context(cfg, a.device)
    rows = ctx.budgets['1']
    folds, _ = budget_folds(ctx.row_sessions, rows, ctx.fixed)
    designs = w.Designs(ctx.cache, list(range(len(ctx.row_sessions))), a.device)
    enc = {}
    for d in (1, 2, 3):
        x = designs.train(d)
        enc[d], _, _ = encoding_scores(ctx.split['train'].targets, x, rows, ctx.row_sessions, folds, ctx.alphas, a.device)
        del x
    enc_max = np.max(np.stack([enc[d] for d in (1, 2, 3)]), axis=0)
    from scipy.stats import spearmanr
    out['spearman_isp_vs_encoding_max'] = float(spearmanr(isp, enc_max).correlation)
    top = lambda v, q: set(np.argsort(-v)[:int(round(q * len(v)))].tolist())
    out['top_overlap'] = {f'{q:g}': len(top(isp, q) & top(enc_max, q)) / max(1, int(round(q * len(isp))))
                          for q in (0.02, 0.05, 0.10)}
    out['note'] = 'feasibility pilot; training data only; no decoding endpoint computed'
    save_json(out, Path('results/v4_pilot') / f'{a.target}_isp_summary.json')
    print(json.dumps({k: v for k, v in out.items() if k not in ('stories', 'alpha_curve')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
