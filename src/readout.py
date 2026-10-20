"""Frozen neural features and closed-form ridge readouts for the v2 baselines.

Feature extractors never see semantic targets. Ridge is fit only on the labeled
training subset; feature standardization uses all training-split neural inputs,
which are available without labels at every budget.
"""
from __future__ import annotations
import numpy as np
import torch
from .targets import TargetIndex

ALPHAS = tuple(float(a) for a in np.logspace(-1, 7, 17))


def _device(device):
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return device


def raw_features(index: TargetIndex, cutoff, delays):
    """Concatenate pooled BOLD rows at ``target_tr + delay`` (delay <= cutoff)."""
    delays = [int(d) for d in delays]
    start, _ = index.window_bounds(cutoff)
    if min(delays) < 0 or max(delays) > int(cutoff):
        raise ValueError('Delays must lie within the response cutoff')
    out = np.empty((len(index), len(delays) * index.n_features), np.float32)
    for row, (ri, t) in enumerate(index.items):
        x = index.neural(ri)
        out[row] = np.concatenate([x[t + d] for d in delays])
    return out


def window_matrix(index: TargetIndex, cutoff, rows=None):
    """Flattened [window, features] inputs ending at ``target_tr + cutoff``."""
    rows = range(len(index)) if rows is None else rows
    rows = list(rows)
    out = np.empty((len(rows), index.window * index.n_features), np.float32)
    for k, i in enumerate(rows):
        ri, t = index.items[i]
        end = t + int(cutoff)
        out[k] = np.asarray(index.neural(ri)[end - index.window + 1:end + 1]).reshape(-1)
    return out


class WindowPCA:
    """Train-only PCA of flattened neural windows (no semantic labels)."""

    def __init__(self, windows, n_components, device='cuda'):
        device = _device(device)
        x = torch.as_tensor(windows, dtype=torch.float64, device=device)
        self.mean = x.mean(0)
        x = x - self.mean
        gram = x @ x.T
        values, vectors = torch.linalg.eigh(gram)
        order = torch.argsort(values, descending=True)[:int(n_components)]
        values, vectors = values[order].clamp_min(1e-12), vectors[:, order]
        self.components = (x.T @ vectors) / values.sqrt()  # [features, k], orthonormal columns
        self.explained_variance = (values / (len(x) - 1)).cpu().numpy()
        self.total_variance = float(x.square().sum().item() / (len(x) - 1))
        self.device = device

    def transform(self, windows, n_components=None, batch_size=1024):
        k = self.components.shape[1] if n_components is None else int(n_components)
        parts = []
        for start in range(0, len(windows), batch_size):
            x = torch.as_tensor(windows[start:start + batch_size], dtype=torch.float64, device=self.device)
            parts.append(((x - self.mean) @ self.components[:, :k]).float().cpu().numpy())
        return np.concatenate(parts)


@torch.no_grad()
def encoder_states(encoder, x):
    """All block outputs [layers, B, T, D]; the last includes the encoder's final norm.

    Identical to ``encoder(x)`` for the final block with every value observed.
    """
    hidden = (encoder.value_embedding(x) + encoder.missing_embedding(torch.zeros_like(x))
              + encoder.position_embedding[:, :x.shape[1]])
    blocks = encoder.transformer.layers
    states = []
    for number, block in enumerate(blocks, 1):
        hidden = block(hidden)
        last = number == len(blocks) and encoder.transformer.norm is not None
        states.append(encoder.transformer.norm(hidden) if last else hidden)
    return torch.stack(states)


@torch.no_grad()
def encoder_features(encoder, index: TargetIndex, cutoff, n_positions, device='cuda', batch_size=256):
    """Frozen states at the final ``n_positions`` window positions: [N, layers, positions, D]."""
    device = _device(device)
    encoder = encoder.to(device).eval()
    out = []
    for start in range(0, len(index), batch_size):
        rows = range(start, min(len(index), start + batch_size))
        x = torch.from_numpy(window_matrix(index, cutoff, rows)).view(len(rows), index.window, index.n_features)
        states = encoder_states(encoder, x.to(device))[:, :, -int(n_positions):]
        out.append(states.permute(1, 0, 2, 3).float().cpu().numpy())
    return np.concatenate(out)


class Standardizer:
    """Per-feature z-scoring fit on unlabeled training-split features."""

    def __init__(self, train_features):
        x = np.asarray(train_features, dtype=np.float64)
        self.mean = x.mean(0)
        std = x.std(0)
        self.std = np.where(std > 1e-8 * max(1.0, float(np.abs(x).max())), std, 1.0)

    def __call__(self, x):
        return ((np.asarray(x, dtype=np.float64) - self.mean) / self.std).astype(np.float32)


class RidgePath:
    """Ridge with an unpenalized intercept, solved once by SVD for many alphas."""

    def __init__(self, x, y, device='cuda'):
        self.device = _device(device)
        x = torch.as_tensor(x, dtype=torch.float64, device=self.device)
        y = torch.as_tensor(y, dtype=torch.float64, device=self.device)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError('Ridge requires paired 2-D designs with at least two rows')
        self.x_mean, self.y_mean = x.mean(0), y.mean(0)
        u, self.s, self.vh = torch.linalg.svd(x - self.x_mean, full_matrices=False)
        self.uty = u.T @ (y - self.y_mean)

    def project(self, x):
        x = torch.as_tensor(x, dtype=torch.float64, device=self.device)
        return (x - self.x_mean) @ self.vh.T

    def predict(self, projected, alpha):
        shrink = self.s / (self.s.square() + float(alpha))
        return (projected * shrink) @ self.uty + self.y_mean


def run_codes(run_ids):
    names = list(dict.fromkeys(np.asarray(run_ids).tolist()))
    lookup = {name: i for i, name in enumerate(names)}
    return names, np.array([lookup[r] for r in np.asarray(run_ids).tolist()], dtype=np.int64)


def run_correlations(prediction, target, codes, n_runs):
    """Mean per-feature temporal Pearson r within each run (torch, any device).

    Matches the reported metric (``metrics._feature_correlation``): population
    standard deviations, and columns whose deviation is below 1e-8 times their
    RMS are omitted, so selection and reporting score the same columns.
    """
    out = torch.full((n_runs,), float('nan'), dtype=torch.float64)
    for r in range(n_runs):
        mask = codes == r
        if int(mask.sum()) < 2:
            continue
        raw_p, raw_t = prediction[mask], target[mask]
        p, t = raw_p - raw_p.mean(0), raw_t - raw_t.mean(0)
        sp, st = p.square().mean(0).sqrt(), t.square().mean(0).sqrt()
        rms_p = raw_p.square().mean(0).sqrt().clamp_min(1e-12)
        rms_t = raw_t.square().mean(0).sqrt().clamp_min(1e-12)
        ok = (sp > 1e-8 * rms_p) & (st > 1e-8 * rms_t)
        if ok.any():
            out[r] = ((p * t).mean(0)[ok] / (sp[ok] * st[ok])).mean().cpu()
    return out


def session_macro(values, run_names, sessions):
    """Average runs within session, then sessions (NaN runs omitted)."""
    groups = {}
    for name, value in zip(run_names, np.asarray(values, dtype=np.float64)):
        if np.isfinite(value):
            groups.setdefault(sessions[name], []).append(value)
    if not groups:
        return float('nan')
    return float(np.mean([np.mean(v) for v in groups.values()]))
