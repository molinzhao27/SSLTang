"""Ridge on precomputed linear kernels, with an unpenalized intercept (v3).

Inputs are uncentered Gram matrices between fitting rows and cross-kernels from
evaluation rows to fitting rows. Centering both on the fitting rows' feature
mean implements the intercept exactly for linear kernels. A Gram matrix must not
be handed to a feature-based solver such as RidgePath: that would square it.
"""
from __future__ import annotations
import warnings
import numpy as np
import torch

EPS64 = float(np.finfo(np.float64).eps)


def combine(kernels, coefficients):
    """Σ_g a_g K_g over the groups with nonzero coefficients."""
    total = None
    for key, weight in coefficients.items():
        if weight:
            term = kernels[key] * float(weight)
            total = term if total is None else total + term
    if total is None:
        raise ValueError('Candidate kernel has no nonzero coefficient')
    return total


class KernelRidgePath:
    """Centered dual ridge, solved once by eigendecomposition for many alphas.

    Eigenvalues at or below ``n * eps * lambda_max`` are dropped. More negative
    eigenvalues are counted and reported relative to lambda_max, never clipped
    silently into the fit.
    """

    def __init__(self, kernel, y, device='cuda'):
        self.device = torch.device(device)
        k = torch.as_tensor(kernel, dtype=torch.float64, device=self.device)
        y = torch.as_tensor(y, dtype=torch.float64, device=self.device)
        n = k.shape[0]
        if k.ndim != 2 or k.shape != (n, n) or y.ndim != 2 or len(y) != n or n < 2:
            raise ValueError('Require a square n-by-n kernel and n paired target rows, n >= 2')
        self.col, self.total = k.mean(0), k.mean()
        centered = k - self.col[None, :] - self.col[:, None] + self.total
        centered = (centered + centered.T) * 0.5
        values, vectors = torch.linalg.eigh(centered)
        lam_max = float(values[-1].clamp_min(0))
        tolerance = n * EPS64 * lam_max
        negative = values < -tolerance
        self.diagnostics = dict(n=n, lambda_max=lam_max, tolerance=tolerance,
                                trace=float(centered.diagonal().sum()),
                                negative_count=int(negative.sum()),
                                most_negative_relative=float(values.min() / lam_max) if lam_max > 0 else 0.0)
        keep = values > tolerance
        self.lam, self.u = values[keep], vectors[:, keep]
        self.diagnostics['rank'] = int(keep.sum())
        self.y_mean = y.mean(0)
        self.uty = self.u.T @ (y - self.y_mean)

    def project(self, cross):
        """Centered cross-kernel times eigenvectors for evaluation rows."""
        k = torch.as_tensor(cross, dtype=torch.float64, device=self.device)
        if k.ndim != 2 or k.shape[1] != len(self.col):
            raise ValueError('Cross-kernel must be evaluation-rows by fitting-rows')
        centered = k - k.mean(1, keepdim=True) - self.col[None, :] + self.total
        return centered @ self.u

    def predict(self, projected, alpha):
        return (projected / (self.lam + float(alpha))) @ self.uty + self.y_mean

    def predict_path(self, projected, alphas):
        """Predictions for every alpha: [n_alpha, rows, outputs]."""
        return torch.stack([self.predict(projected, a) for a in alphas])

    def predict_per_output(self, projected, alphas):
        """One alpha per output column (``alphas`` has one entry per output)."""
        a = torch.as_tensor(np.asarray(alphas, dtype=np.float64), device=self.device)
        if a.shape != (self.uty.shape[1],):
            raise ValueError('Need exactly one alpha per output')
        return projected @ (self.uty / (self.lam[:, None] + a[None, :])) + self.y_mean


def run_feature_r(prediction, target, codes, n_runs):
    """Per-run, per-output temporal Pearson r with the reporting metric's rule.

    ``prediction`` is [..., rows, outputs]; returns [..., n_runs, outputs] with
    NaN for runs with fewer than two rows and for columns whose standard
    deviation is below 1e-8 times their RMS (as ``metrics._feature_correlation``).
    """
    shape = prediction.shape[:-2] + (n_runs, prediction.shape[-1])
    out = torch.full(shape, float('nan'), dtype=torch.float64, device=prediction.device)
    for r in range(n_runs):
        mask = codes == r
        if int(mask.sum()) < 2:
            continue
        p, t = prediction[..., mask, :], target[mask]
        pc, tc = p - p.mean(-2, keepdim=True), t - t.mean(0, keepdim=True)
        sp, st = pc.square().mean(-2).sqrt(), tc.square().mean(0).sqrt()
        rms_p = p.square().mean(-2).sqrt().clamp_min(1e-12)
        rms_t = t.square().mean(0).sqrt().clamp_min(1e-12)
        ok = (sp > 1e-8 * rms_p) & (st > 1e-8 * rms_t)[None, ...]
        r_value = (pc * tc).mean(-2) / (sp * st)
        out[..., r, :] = torch.where(ok, r_value, torch.full_like(r_value, float('nan')))
    return out


def session_macro_scores(per_run, run_sessions):
    """Session-macro mean over runs of ``per_run`` [..., n_runs] (NaN runs omitted)."""
    per_run = np.asarray(per_run, dtype=np.float64)
    sessions = np.asarray(run_sessions)
    with warnings.catch_warnings():  # All-NaN sessions are omitted, not errors.
        warnings.simplefilter('ignore', RuntimeWarning)
        means = [np.nanmean(per_run[..., sessions == s], axis=-1) for s in dict.fromkeys(sessions.tolist())]
        return np.nanmean(np.stack(means, axis=-1), axis=-1)
