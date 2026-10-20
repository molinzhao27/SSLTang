"""PROTOCOL_V4 inter-subject predictability (ISP) and self-predictability (SELF) voxel priors.

ISP: each target voxel's per-run z-scored BOLD at budget times is predicted from other
listeners' BOLD at the same stimulus times. Every reference contributes the top-K principal
components of its own per-run z-scored BOLD (a reference-only, transcript-free basis, rule R3)
at lags -1, 0, +1 TR, concatenated over references in sorted order. ISP_v is the pooled
out-of-fold Pearson r of ridge (alpha in logspace(0, 10, 21), ties to the larger alpha) over
half-story groups, computed with the §12 routine ``encoding_scores``. No transcript is read.
Callers that cache projected components build the design with ``design`` (``predictors`` is the
same path from raw BOLD) and score it with ``isp_scores``.

Controls: MISALIGN circularly shifts every reference's per-run components by n_run // 2 rows
before lagging; the noise ladder degrades the prior to reliability f (f = 0 permutes it); SELF
predicts each octant of the target's own voxels from the top-K components of its voxels
outside that octant dilated by a margin. ``groups='pilot'`` reproduces the feasibility pilot
(scripts/v4_pilot_isp.py: story index mod 5 groups) for gate G0(c).
"""
from __future__ import annotations
import hashlib
import h5py
import numpy as np
import torch
from ..budgets import half_groups
from ..encoding_weights import encoding_scores
from ..voxel_kernels import _h5_path

LAGS = (-1, 0, 1)
N_COMPONENTS = 100
ALPHAS = np.logspace(0, 10, 21)
PILOT_ALPHAS = np.logspace(0, 6, 13)
PILOT_FOLDS = 5
EXACT_ROWS = 6000
STD_FLOOR = 1e-6


def load_bold(raw_root, run):
    """[n_tr, V] float32 BOLD of one run: non-finite values set to 0, then z-scored over time.

    Columns whose centred std is at most ``STD_FLOOR`` (1e-6) are set to exactly 0: a constant
    column that float32 cannot represent keeps a mean-rounding residue that the floor would
    otherwise inflate into a per-run offset. Every other column is bitwise the pilot's z-score.
    """
    with h5py.File(_h5_path(run, raw_root), 'r') as handle:
        x = np.nan_to_num(np.asarray(handle['data'], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if 'n_tr' in run and len(x) != int(run['n_tr']):
        raise ValueError(f"{run['source_key']}: {len(x)} TRs, manifest says {run['n_tr']}")
    x -= x.mean(0)
    std = x.std(0)
    x /= np.maximum(std, STD_FLOOR)  # Per-run z-score: unsupervised, within run.
    x[:, std <= STD_FLOOR] = 0
    return x


@torch.no_grad()
def reference_basis(bold_runs, n_components=N_COMPONENTS, seed=0, device='cpu'):
    """PCA basis (mean[V], comps[V, K], float64) of one subject's concatenated per-run z-scored runs.

    Exact SVD up to ``EXACT_ROWS`` rows; otherwise ``torch.svd_lowrank(q=K+20, niter=6)`` after
    ``torch.manual_seed(seed)``. Only the decomposition runs in float32 on ``device``; K is
    truncated to the available rank afterwards.
    """
    z = np.concatenate([np.asarray(x, dtype=np.float32) for x in bold_runs])
    mean = z.mean(0, dtype=np.float64)
    z -= mean.astype(np.float32)
    zt = torch.as_tensor(z, device=device)
    k = min(int(n_components), *zt.shape)
    if len(zt) <= EXACT_ROWS:
        comps = torch.linalg.svd(zt, full_matrices=False)[2][:k].T
    else:
        torch.manual_seed(seed)
        comps = torch.svd_lowrank(zt, q=min(k + 20, *zt.shape), niter=6)[2][:, :k]
    return mean, comps.double().cpu().numpy()


@torch.no_grad()
def project(z, basis, device='cpu'):
    """Component scores [n_tr, K] (float64) of z-scored BOLD on a basis from ``reference_basis``."""
    mean, comps = (torch.as_tensor(a, dtype=torch.float64, device=device) for a in basis)
    x = torch.as_tensor(np.asarray(z), device=device).double()
    return ((x - mean) @ comps).cpu().numpy()


def lagged(z, lags=LAGS):
    """[n_tr, F * len(lags)]: row t holds z[t - lag] for each lag, zero-padded within this run."""
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


def misaligned(pred_runs):
    """Each per-run block circularly shifted by n_run // 2 rows (row t moves to t + n_run // 2).

    Applied to a reference's per-run components before lagging: per-run content and the
    references' mutual alignment are kept; stimulus locking to the target is broken.
    """
    return [np.roll(p, len(p) // 2, axis=0) for p in pred_runs]


def design(scores_by_ref_story, stories, lags=LAGS, misalign=False):
    """ISP design [T, n_refs * K * len(lags)] (float64) from per-run component scores.

    ``scores_by_ref_story[ref][story]`` is a reference's [n_tr, K] scores at one story (e.g. a
    cached ``project`` output). References enter in sorted order, each as a lag-major block
    (``lagged``: lag -1, 0, +1 of its K components); rows are ``stories`` concatenated in the given
    order. Lags are zero-padded within each run; ``misalign`` shifts every per-run block by half a
    run first (``misaligned``, EB-MISALIGN).
    """
    blocks, lengths = [], None
    for ref in sorted(scores_by_ref_story):
        pcs = [np.asarray(scores_by_ref_story[ref][st], dtype=np.float64) for st in stories]
        if lengths is None:
            lengths = [len(p) for p in pcs]
        elif [len(p) for p in pcs] != lengths:
            raise ValueError(f'{ref}: story lengths differ from the other references')
        if misalign:
            pcs = misaligned(pcs)
        blocks.append(np.concatenate([lagged(p, lags) for p in pcs]))
    if not blocks:
        raise ValueError('ISP needs at least one reference')
    return np.concatenate(blocks, axis=1)


def predictors(ref_runs_by_story, bases, stories, lags=LAGS, misalign=False, device='cpu'):
    """``design`` from raw BOLD: ``ref_runs_by_story[ref][story]`` (per-run z-scored) projected on ``bases[ref]``."""
    return design({ref: {st: project(runs[st], bases[ref], device) for st in stories}
                   for ref, runs in ref_runs_by_story.items()}, stories, lags, misalign)


def row_groups(stories, lengths, groups='half'):
    """Row group labels and held-out folds for ``stories`` of the given lengths.

    ``'half'``: two groups per story split at its middle TR (``budgets.half_groups``), each held
    out once. ``'pilot'``: story index mod 5, five folds, as in scripts/v4_pilot_isp.py.
    """
    if len(set(stories)) != len(stories):
        raise ValueError('Stories must be unique')
    if groups == 'half':
        rows = [(st, t) for st, n in zip(stories, lengths) for t in range(n)]
        labels = half_groups([s for s, _ in rows], [t for _, t in rows], dict(zip(stories, lengths)))
        return labels, [(g,) for g in dict.fromkeys(labels)]
    if groups == 'pilot':
        labels = [str(i % PILOT_FOLDS) for i, n in enumerate(lengths) for _ in range(n)]
        return labels, [(str(g),) for g in range(PILOT_FOLDS)]
    raise ValueError(f'Unknown ISP grouping {groups!r}')


def isp_scores(target_runs_by_story, predictors, stories, alphas=ALPHAS, device='cpu', groups='half'):
    """(r[V], alpha, mean-r curve): pooled out-of-fold r of ridge from ``predictors`` to target voxels.

    The response is the target's per-run z-scored BOLD over ``stories``, concatenated in the given
    order (the row order of ``predictors``). One alpha is chosen for all voxels by mean r.
    """
    runs = [np.asarray(target_runs_by_story[st]) for st in stories]
    labels, folds = row_groups(stories, [len(x) for x in runs], groups)
    x = np.asarray(predictors, dtype=np.float64)
    if len(x) != len(labels):
        raise ValueError(f'Predictors have {len(x)} rows; target stories have {len(labels)}')
    y = torch.as_tensor(np.concatenate(runs), dtype=torch.float64, device=device)
    return encoding_scores(x, y, list(range(len(y))), labels, folds, np.asarray(alphas, dtype=np.float64), device)


def ladder_seed(subject, f):
    """Registered seed: sha256('ssltang-v4-ladder:<subject>:<f>'), or 'ssltang-v4-perm:<subject>' at f = 0."""
    text = f'ssltang-v4-perm:{subject}' if f == 0 else f'ssltang-v4-ladder:{subject}:{f:g}'
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def ladder(p, f, seed):
    """Prior degraded to reliability ``f``: p + N(0, Var(p)(1/f - 1)) for f > 0; p permuted for f = 0."""
    p = np.asarray(p, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if f == 0:
        return rng.permutation(p)
    if not 0 < f <= 1:
        raise ValueError(f'Ladder level must lie in [0, 1], got {f}')
    return p + rng.normal(0.0, np.sqrt(p.var() * (1.0 / f - 1.0)), size=p.shape)


def octant_masks(coords, margin=4):
    """Per octant: (its voxels, the predictor voxels outside it dilated by ``margin`` voxels).

    Octants come from median splits of x, y and z (above the median is the upper half; octant
    id = 4 x_upper + 2 y_upper + z_upper). ``coords`` are integer voxel indices [V, 3]; a voxel is
    in the dilated octant when on every axis it lies on the octant's side or at most ``margin``
    voxels beyond its boundary, i.e. within Chebyshev distance ``margin`` of the octant.
    """
    c = np.asarray(coords)
    med = np.median(c, axis=0)
    upper = c > med
    octant = upper.astype(np.int64) @ np.array([4, 2, 1])
    out = []
    for o in range(8):
        near = np.ones(len(c), dtype=bool)
        for a in range(3):
            near &= (c[:, a] > med[a] - margin) if (o >> (2 - a)) & 1 else (c[:, a] <= med[a] + margin)
        out.append((octant == o, ~near))
    return out


def self_scores(target_runs_by_story, coords, stories, n_components=N_COMPONENTS, margin=4, alphas=ALPHAS,
                device='cpu', lags=LAGS, seed=0):
    """(r[V], info): SELF prior, each octant's voxels scored from the target's own voxels outside it.

    Predictors are the top-K components (lags -1..+1) of the target's voxels outside the octant
    dilated by ``margin``, with the basis from the same ``stories``; outputs are the octant's
    voxels, scored with half-story out-of-fold r. Each octant chooses its own alpha.
    """
    runs = [np.asarray(target_runs_by_story[st], dtype=np.float32) for st in stories]
    r = np.zeros(runs[0].shape[1])
    if np.shape(coords) != (len(r), 3):
        raise ValueError(f'Coordinates {np.shape(coords)} do not match {len(r)} voxels')
    info = []
    for o, (out, pred) in enumerate(octant_masks(coords, margin)):
        if not out.any():
            continue
        if not pred.any():
            raise ValueError(f'Octant {o}: no predictor voxels outside the margin')
        sources = [x[:, pred] for x in runs]
        basis = reference_basis(sources, n_components, seed, device)
        design = np.concatenate([lagged(project(x, basis, device), lags) for x in sources])
        del sources
        r[out], alpha, curve = isp_scores({st: x[:, out] for st, x in zip(stories, runs)}, design, stories, alphas,
                                          device)
        info.append(dict(octant=o, n_outputs=int(out.sum()), n_predictor_voxels=int(pred.sum()),
                         components=int(basis[1].shape[1]), alpha=alpha, alpha_curve=curve.tolist(),
                         alpha_at_grid_edge=alpha in (float(alphas[0]), float(alphas[-1]))))
    return r, info
