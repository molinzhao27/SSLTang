"""Dual ridge paths for voxel designs with many more columns than rows."""
from __future__ import annotations

import torch

from .readout import _device


class GramRidgePath:
    """RidgePath's interface with an n-by-n Gram eigendecomposition.

    Float32 inputs stay float32 for the large centered design and the feature
    matmuls (including on CUDA). Only the small Gram/eigenvector algebra and
    targets use float64; TF32 should be disabled by the caller. Explicit
    float64 inputs retain double precision, useful for reference checks. An
    on-device tensor is borrowed without conversion before centering, and
    neither fitting nor projection modifies the caller's array.

    Eigenvalues below a relative roundoff tolerance are discarded: 10 times
    float32 epsilon for single-precision Gram products, or max(n, p) times
    float64 epsilon for double-precision products. This removes the centered
    null direction and numerical negative eigenvalues, without dividing by
    their square roots. The intercept is fit on labeled rows and unpenalized.
    """

    @torch.no_grad()
    def __init__(self, x, y, device='cuda'):
        self.device = _device(device)
        x = torch.as_tensor(x, device=self.device)
        if x.dtype not in (torch.float32, torch.float64):
            x = x.float()
        y = torch.as_tensor(y, dtype=torch.float64, device=self.device)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError('Ridge requires paired 2-D designs with at least two rows')
        self.x_mean, self.y_mean = x.mean(0), y.mean(0)
        self._xc = x - self.x_mean
        gram = (self._xc @ self._xc.T).double()
        # Symmetrize any BLAS roundoff before the Hermitian decomposition.
        gram = (gram + gram.T) * 0.5
        values, vectors = torch.linalg.eigh(gram)
        eps = torch.finfo(x.dtype).eps
        self.relative_tolerance = 10 * eps if x.dtype == torch.float32 else max(x.shape) * eps
        keep = values > values[-1].clamp_min(0) * self.relative_tolerance
        self.s = values[keep].flip(0).sqrt()
        self._u = vectors[:, keep].flip(1)
        self.uty = self._u.T @ (y - self.y_mean)

    @torch.no_grad()
    def project(self, x_eval):
        """Return (x_eval - x_mean) Xc.T U / s, with no p-by-r matrix."""
        x = torch.as_tensor(x_eval, dtype=self._xc.dtype, device=self.device)
        if x.ndim != 2 or x.shape[1] != self._xc.shape[1]:
            raise ValueError('Evaluation features must match the fitted design')
        cross_gram = ((x - self.x_mean) @ self._xc.T).double()
        return (cross_gram @ self._u) / self.s

    @torch.no_grad()
    def predict(self, projected, alpha):
        shrink = self.s / (self.s.square() + float(alpha))
        return (projected * shrink) @ self.uty + self.y_mean
