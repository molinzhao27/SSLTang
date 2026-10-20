"""Encoding-predictability column weights for all-voxel kernels (PROTOCOL_V3 §12).

For each delay d, a ridge encoding model predicts every standardized voxel
column at t+d from the 768-d target at t. Its out-of-fold Pearson r per column,
pooled over session-grouped folds of labeled rows only, sets that column's
weight. Weighted columns form one all-voxel linear kernel.
"""
from __future__ import annotations
import numpy as np
import torch

RULES = (('uniform', None), ('soft', 0.5), ('soft', 1.0), ('soft', 2.0),
         ('hard', 0.10), ('hard', 0.25), ('hard', 0.50))
ARMS = {'W0': [('uniform', None)],
        'W1': [('soft', 0.5), ('soft', 1.0), ('soft', 2.0)],
        'W2': [('hard', 0.10), ('hard', 0.25), ('hard', 0.50)]}
FALLBACK_FRACTION = 0.01
TIE = 1e-12


def rule_label(rule):
    kind, value = rule
    return 'uniform' if kind == 'uniform' else f'{kind}={value:g}'


def best_index(curve, tie=TIE):
    """Index of the best finite value; ties within ``tie`` go to the later (larger-alpha) entry."""
    curve = np.asarray(curve, dtype=np.float64)
    finite = np.isfinite(curve)
    if not finite.any():
        return None
    best = curve[finite].max()
    return int(np.flatnonzero(finite & (curve >= best - tie))[-1])


@torch.no_grad()
def encoding_scores(targets, design, rows, row_sessions, folds, alphas, device, chunk=8192):
    """Pooled out-of-fold Pearson r per column of ``design``, the chosen alpha and the alpha curve.

    ``design`` is a float64 tensor [N, V] of standardized columns on ``device``;
    ``rows`` are the labeled rows allowed to inform the scores; ``folds`` are
    held-out session tuples. Each labeled row is predicted once, by the model
    fit without its sessions. One alpha (best mean r over columns; ties to the
    larger alpha) is returned. Raises ValueError when a fold lacks fitting or
    held-out labeled rows.
    """
    device = torch.device(device)
    y = torch.as_tensor(np.asarray(targets), dtype=torch.float64, device=device)
    n_cols, n_alpha = design.shape[1], len(alphas)
    stats = {k: torch.zeros(n_alpha, n_cols, dtype=torch.float64, device=device)
             for k in ('p', 'pp', 'v', 'vv', 'pv')}
    count = 0
    for held in folds:
        held = set(held)
        fit = torch.as_tensor([i for i in rows if row_sessions[i] not in held], dtype=torch.long, device=device)
        hold = torch.as_tensor([i for i in rows if row_sessions[i] in held], dtype=torch.long, device=device)
        if len(fit) < 2 or len(hold) == 0:
            raise ValueError('Encoding folds need fitting and held-out labeled rows')
        x_fit, x_hold = y.index_select(0, fit), y.index_select(0, hold)
        mean_x = x_fit.mean(0)
        u, s, vh = torch.linalg.svd(x_fit - mean_x, full_matrices=False)
        projected = (x_hold - mean_x) @ vh.T
        shrink = torch.stack([s / (s.square() + float(a)) for a in alphas])  # [alphas, rank]
        count += len(hold)
        for start in range(0, n_cols, chunk):
            cols = slice(start, start + chunk)
            block = design[:, cols]
            cf, actual = block.index_select(0, fit), block.index_select(0, hold)  # One column chunk at a time.
            mean_c = cf.mean(0)
            uty = u.T @ (cf - mean_c)
            del cf
            for k in range(n_alpha):
                pred = (projected * shrink[k]) @ uty + mean_c
                stats['p'][k, cols] += pred.sum(0)
                stats['pp'][k, cols] += pred.square().sum(0)
                stats['v'][k, cols] += actual.sum(0)
                stats['vv'][k, cols] += actual.square().sum(0)
                stats['pv'][k, cols] += (pred * actual).sum(0)
    n = float(count)
    cov = n * stats['pv'] - stats['p'] * stats['v']
    var = (n * stats['pp'] - stats['p'].square()) * (n * stats['vv'] - stats['v'].square())
    r = torch.where(var > 0, cov / var.clamp_min(1e-300).sqrt(), torch.zeros_like(cov))
    mean_r = r.mean(1).cpu().numpy()
    k = best_index(mean_r)
    return r[k].cpu().numpy(), float(alphas[k]), mean_r


def column_weights(scores, rule):
    """Weights per column (pooled over delays) with mean(w^2) = 1 and a uniform fallback.

    ``scores`` maps delay -> r array. Hard rules keep exactly round(q * n) columns:
    the highest r, ties broken by column order (delay, then voxel index).
    Returns (weights by delay, info).
    """
    kind, value = rule
    delays = sorted(scores)
    flat = np.concatenate([np.asarray(scores[d], dtype=np.float64) for d in delays])
    positive = float((flat > 0).mean())
    fallback = kind != 'uniform' and positive < FALLBACK_FRACTION
    if kind == 'uniform' or fallback:
        w = np.ones_like(flat)
    elif kind == 'soft':
        w = np.maximum(flat, 0.0) ** float(value)
    elif kind == 'hard':
        keep = max(1, int(round(float(value) * len(flat))))
        w = np.zeros_like(flat)
        w[np.argsort(-flat, kind='stable')[:keep]] = 1.0
    else:
        raise ValueError(f'Unknown weighting rule {rule}')
    w = w / np.sqrt(np.mean(np.square(w)))
    out, start = {}, 0
    for d in delays:
        size = len(scores[d])
        out[d] = w[start:start + size]
        start += size
    info = dict(rule=rule_label(rule), fallback=bool(fallback), positive_fraction=positive,
                nonzero_fraction=float((w > 0).mean()))
    return out, info
