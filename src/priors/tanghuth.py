"""PROTOCOL_V4 published comparators after Tang & Huth 2025 (FUTURE_WORK_V4 §5; V4_IMPLEMENTATION Module H).

Reference labels follow rule TS or SS (FUTURE_WORK_V4 §3, R4; ``rule_for``: TS in ds003020, SS in LPP):
  TS  stories the target never heard: ds003020 targets UTS04-UTS10 with references UTS01-UTS03; a reference's
      labeled set is all its TS stories, in any of its own splits (its validation/test stories are not the
      target's material; the runner builds their delay rows at the cell's clock).
  SS  any material except the target's validation or test material (the LPP default, flagged by the auditor).

TH_SEL (functional voxel selection; the decoder is the common v4 decoder):
  1. Each reference's top-10,000 voxels by out-of-fold encoding r: W1's encoding model (768-d target -> each
     standardized (voxel, delay) column, ``encoding_scores``) on the reference's TS/SS labeled rows; a voxel's
     score is its maximum over the delays 1..3 (the G1 convention for a per-voxel W1 map).
  2. A reverse ridge converter per reference maps its top-10k voxels (per-run z-scored, lags -1..+1) to every
     target voxel (per-run z-scored), fit on the fold's fitting stories of B; alpha in logspace(0, 10, 21) by
     pooled out-of-fold r over half-story (half-section) groups, ties to the larger alpha.
  3. Each converter is applied to its reference's BOLD on material the target did not hear (ds003020: the TS
     stories every reference heard, in any of the references' splits, BOLD only; LPP: out-of-budget non-test
     sections, none at L8: not estimable, no fallback).
  4. Score per target voxel: the mean pairwise correlation of the references' predictions (>= 2 references).
  5. Weights: uniform on the top 15,000 voxels (ds003020) or the top 17% of mask voxels (LPP), at all delays.
TH_SEL_rand: a uniform random mask of the same size, seed int(sha256('ssltang-v4-thsel-rand:<subject>')[:8], 16).
XDEC (zero target labels):
  1. Each reference's W1 decoder (§12 rule, run_v4's engine) on its top-10k voxels and its TS/SS labels,
     represented in primal form (``LinearDecoder``) so that it can be applied to any input rows.
  2. A ridge converter maps target BOLD (all voxels, per-run z-scored, lags -1..+1) to that reference's
     top-10k voxels in the decoder's input representation (the cached H5 rows: raw values, non-finite -> 0),
     fit on B times with alpha by grouped out-of-fold r as in TH_SEL.
  3. Prediction for each target evaluation row (target bin at TR t): decoder(converted BOLD at t+1..t+3),
     averaged over references.

Wide converters are solved in the dual: the linear kernel of the lagged design is accumulated in feature
chunks. Ridge with an intercept depends on the design only through its Gram matrix, so the converter alpha
(``kernel_alpha_scores``) is ``encoding_scores`` (the §12 pooled out-of-fold routine: same folds, statistics,
grid and tie rule) computed in kernel space: each held-out group's fitting rows get run_v4's centered dual
ridge eigen-solve (``KernelRidgePath``) and the 21 smoother matrices [n_held, n_fit] act on the outputs
directly. That equals ``encoding_scores(gram_factor(kernel), ...)`` to rounding (tested at 1e-10) without
its two per-group costs that scale with the outputs (U^T Y over all target voxels and an SVD of the N-wide
factor). ``precision='mixed'`` (not the registered default; only if PROTOCOL_V4 §4 records it) computes the
alpha curve in float32 and rechecks the chosen alpha and its neighbours in float64 (the window grows until
the float64 maximum is interior). ``converter_flops`` / ``thsel_flops`` are the cost model of this routine.
Final fits use run_v4's centered dual ridge (``KernelRidgePath``).

Provenance (R3/R4): converters read target and reference BOLD at budget stimuli (purpose 'converter');
reference models (reference BOLD and transcripts at their labeled stimuli) and the selection material
(reference BOLD the target did not hear) are purpose 'reference_model'. ``*_reads`` return these records
(derived from the stimuli of the arrays actually passed) and ``log_reads`` writes them into a
``Provenance(..., arm='TH_SEL' | 'XDEC', reference_rule='TS' | 'SS')``. ``ThSelector`` logs its reads
itself, so inside run_v4's ``prov.inner_fold`` they carry the inner fold tag (R7). ``scope_errors`` /
``audit_errors`` add each arm's own registered-scope checks to ``scripts/audit_v4.py``: TH_SEL must show its
selection refit inside every inner fold (tagged target and reference converter reads over the fold's fitting
stories) and on the whole budget; TH_SEL_rand reads no other subject; XDEC fits nothing on target labels and
declares no inner folds, so the two target-decoder completeness checks of the frozen audit are replaced by
zero-target-label checks (log and ledger). TH_SEL and XDEC also carry their dataset's registered rule
(``rule_for``), in ds003020 read reference models only on stories the target never heard (its whole
manifest), and TH_SEL's selection material (the tagged 'reference_model' reads of every inner fold, and the
cell's ``selection_material`` / ``unheard`` when given) is the registered one: ds003020 the stories every
reference heard and the target did not, LPP ``lpp_unheard``. ``audit_file`` is the matching drop-in for
``audit_v4.audit_file`` (identical for every non-Module-H cell; the re-audit for all of ``ARMS``).

Two-reference sensitivity (FUTURE_WORK_V4 §5 TH-SEL step 3, decision 6; ``sensitivity_pairs``): ds003020 every
pair of UTS01-UTS03; LPP the first two of the target's references in ``ssltang-v4-thsel-pair:<target>:<ref>``
hash order. ``thsel_scores(alpha_memo=...)`` reuses a reference's converter alpha across reference sets (it
depends only on that reference and the target), so a pair costs only its final applications.

Engine adapters: ``engine()`` imports the frozen ``scripts/run_v4.py`` (never modified here);
``selection_priors`` is a run_v4 ``Priors`` whose ISP slot returns a selection mask, so run_v4's arm
'PRIOR' fits the TH_SEL / TH_SEL_rand decoder (max(mask, 0)^gamma = mask at every gamma and delay; the three
gamma rules tie and the engine keeps gamma 0.5).
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
from dataclasses import dataclass, field
import hashlib
from itertools import combinations
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from ..budgets import LPP_SECTIONS, dataset_of, hash_order, inner_folds
from ..encoding_weights import best_index
from ..multikernel_ridge import EPS64, KernelRidgePath
from ..provenance import ANY_FOLD, Provenance, _covered, _item
from ..structured_ridge import budget_folds, fixed_folds
from ..voxel_kernels import DELAYS
from .isp import ALPHAS as ISP_ALPHAS
from .isp import LAGS, row_groups

ROOT = Path(__file__).resolve().parents[2]
CODE = ('src/priors/tanghuth.py', 'scripts/run_v4_tanghuth.py')  # ``code_sha256`` (the lock's tanghuth_sha256).
ARMS = ('TH_SEL', 'TH_SEL_rand', 'XDEC')
REFERENCE_ARMS = ('TH_SEL', 'XDEC')  # Arms that read reference data (TH_SEL_rand reads none).
ENGINE_ARM = 'PRIOR'  # The run_v4 arm slot that carries a selection mask (``selection_priors``).
RULES = {'ds003020': 'TS', 'lpp': 'SS'}
REF_TOP = 10_000
GOAL_TOP = 15_000
GOAL_FRACTION = 0.17
ALPHAS = ISP_ALPHAS  # Converter alpha grid: logspace(0, 10, 21).
RAND_PREFIX = 'ssltang-v4-thsel-rand:'
PAIR_PREFIX = 'ssltang-v4-thsel-pair:'  # LPP two-reference sensitivity: hash order of the target's references.
CHUNK = 4096  # Features (voxels) per Gram accumulation step and outputs per alpha-curve / TH_SEL scoring step.
PRECISIONS = ('float64', 'mixed')  # Converter alpha curves: registered float64, or float32 + float64 recheck.
XDEC_OUTPUTS = 65536  # XDEC: reference outputs per shared alpha-curve pass (bounds the stacked outputs' memory).
EIGH_FLOP = 9.0  # Cost model: FLOP of one n x n symmetric eigendecomposition with vectors, per n^3.


class NotEstimable(ValueError):
    """The comparator cannot be computed in this cell; recorded as not estimable, never silently replaced."""


def code_sha256(root=ROOT):
    """SHA-256 over the Module H files (name and bytes, in ``CODE`` order)."""
    digest = hashlib.sha256()
    for name in CODE:
        digest.update(name.encode())
        digest.update((Path(root) / name).read_bytes())
    return digest.hexdigest()


def engine(name='run_v4'):
    """The frozen engine module ``scripts/<name>.py`` (imported, never modified)."""
    scripts = str(ROOT / 'scripts')
    if name not in sys.modules and scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module(name)


# ------------------------------------------------------------------------------------- rules and sizes
def rule_for(dataset):
    """The registered reference-label rule: 'TS' in ds003020, 'SS' in LPP."""
    return RULES[dataset]


def labeled_stimuli(rule, reference_stimuli, heard=(), held=()):
    """Reference stimuli whose transcripts the rule allows, sorted.

    TS: never heard by the target (``heard`` is everything in the target's manifest). SS: not the target's
    validation or test material (``held``).
    """
    if rule == 'TS':
        drop = {str(s) for s in heard}
    elif rule == 'SS':
        drop = {str(s) for s in held}
    else:
        raise ValueError(f'Reference labels need rule TS or SS, not {rule!r}')
    return sorted({str(s) for s in reference_stimuli} - drop)


def lpp_unheard(sections, budget, test):
    """LPP selection material: the out-of-budget sections other than the outer test section (none at L8)."""
    budget = {str(s) for s in budget}
    return [str(s) for s in sections if str(s) not in budget and str(s) != str(test)]


def ds003020_selection(manifests, target, references):
    """ds003020 selection material: the stories every reference heard (any of its splits) and the target never
    heard (``manifests``: subject -> manifest), sorted."""
    heard = {r['story'] for r in manifests[target]['runs']}
    common = set.intersection(*({r['story'] for r in manifests[ref]['runs']} for ref in references))
    return sorted(common - heard)


def sensitivity_pairs(references, dataset, target=None):
    """The registered two-reference sensitivity sets (sorted pairs).

    ds003020: every pair of the references (UTS01-UTS03: three pairs). LPP: one pair per target, the first two
    of its references in ``ssltang-v4-thsel-pair:<target>:<reference>`` hash order.
    """
    refs = sorted(str(r) for r in references)
    if len(refs) < 2:
        raise ValueError(f'A two-reference sensitivity needs at least two references, not {refs}')
    if dataset == 'ds003020':
        return [tuple(p) for p in combinations(refs, 2)]
    if dataset == 'lpp':
        if target is None:
            raise ValueError('The LPP pair is drawn per target')
        return [tuple(sorted(hash_order(refs, f'{PAIR_PREFIX}{target}:')[:2]))]
    raise ValueError(f'Unknown dataset {dataset!r}')


def goal_size(n_voxels, dataset):
    """Selected target voxels: 15,000 in ds003020 (at most all), round(17%) of the mask in LPP."""
    n_voxels = int(n_voxels)
    if dataset == 'ds003020':
        return min(GOAL_TOP, n_voxels)
    if dataset == 'lpp':
        return max(1, int(round(GOAL_FRACTION * n_voxels)))
    raise ValueError(f'Unknown dataset {dataset!r}')


def top_k(scores, k):
    """Ascending indices of the ``k`` highest scores (non-finite scores last; ties to the lower index)."""
    s = np.asarray(scores, dtype=np.float64)
    k = int(k)
    if s.ndim != 1 or not 0 < k <= len(s):
        raise ValueError(f'Cannot take the top {k} of {s.shape} scores')
    key = np.where(np.isfinite(s), s, -np.inf)
    return np.sort(np.argsort(-key, kind='stable')[:k])


def selection_mask(scores, k):
    mask = np.zeros(len(scores), dtype=bool)
    mask[top_k(scores, k)] = True
    return mask


def rand_seed(subject):
    """Registered TH_SEL_rand seed: int(sha256('ssltang-v4-thsel-rand:<subject>')[:8], 16)."""
    return int(hashlib.sha256(f'{RAND_PREFIX}{subject}'.encode()).hexdigest()[:8], 16)


def random_mask(n_voxels, k, subject):
    """TH_SEL_rand: ``k`` voxels drawn uniformly without replacement with the subject's registered seed."""
    if not 0 < int(k) <= int(n_voxels):
        raise ValueError(f'Cannot draw {k} of {n_voxels} voxels')
    mask = np.zeros(int(n_voxels), dtype=bool)
    mask[np.random.default_rng(rand_seed(subject)).choice(int(n_voxels), size=int(k), replace=False)] = True
    return mask


def mask_weights(mask, delays=DELAYS):
    """{d: w}: uniform on the mask at every delay, zero elsewhere, mean w^2 = 1 over all (voxel, delay) columns.

    Equals run_v4's ``soft_weights(mask, gamma, broadcast=True)`` for every gamma (the engine's PRIOR slot).
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        raise ValueError('Empty selection mask')
    w = m / np.sqrt(m.mean())
    return {d: w.copy() for d in delays}


# ------------------------------------------------------------------------------------- provenance reads
def read(subject, stimulus, kind, purpose):
    return dict(subject=str(subject), stimulus=str(stimulus), kind=kind, purpose=purpose)


def log_reads(prov, reads):
    """Record ``reads`` on a Provenance (inside ``prov.inner_fold`` they carry that fold's tag)."""
    for r in reads:
        prov.record(r['subject'], r['stimulus'], r['kind'], r['purpose'])
    return prov


def thsel_reads(target, references, stories, unheard):
    """TH_SEL reads at one fit: target and reference BOLD at the fitting stories ('converter'), and every
    reference's BOLD on the selection material ('reference_model')."""
    out = [read(target, s, 'brain', 'converter') for s in stories]
    for ref in sorted(references):
        out += [read(ref, s, 'brain', 'converter') for s in stories]
        out += [read(ref, u, 'brain', 'reference_model') for u in unheard]
    return out


def xdec_reads(target, references, stories, eval_stimuli, scored=True, context_reset=None):
    """XDEC reads: target and reference BOLD at the budget stimuli ('converter'), target BOLD at the evaluation
    stimuli ('decoder_score'; the decoder input) and, with ``scored``, the target transcripts that score them.
    LPP targets (``context_reset`` default) also get the per-section context reset of their GPT-2 targets (R6)."""
    out = [read(target, s, 'brain', 'converter') for s in stories]
    out += [read(ref, s, 'brain', 'converter') for ref in sorted(references) for s in stories]
    if context_reset is None:
        context_reset = dataset_of(target) == 'lpp'
    for e in eval_stimuli:
        out.append(read(target, e, 'brain', 'decoder_score'))
        if scored:
            if context_reset:
                out.append(read(target, e, 'transcript', 'context_reset'))
            out.append(read(target, e, 'transcript', 'decoder_score'))
    return out


def model_reads(model):
    """A reference model's reads (its BOLD and transcripts at its labeled stimuli) as 'reference_model' reads."""
    return [read(s, st, kind, 'reference_model') for s, st, kind in model.reads]


class Recorder:
    """Read log for a reference's own engine run.

    run_v4 records reads with the engine's purposes (standardize, own_encoding, decoder_fit, ...) for the
    reference as its subject; the target cell logs every such (subject, stimulus, kind) once, as purpose
    'reference_model' (``model_reads``).
    """

    def __init__(self):
        self.reads, self.current_fold = {}, None

    def record(self, subject, stimulus, kind, purpose, trs=None, context=None, n=1, fold=None):
        self.reads.setdefault((str(subject), str(stimulus), str(kind)), set()).add(str(purpose))
        return self

    @contextmanager
    def inner_fold(self, fold):
        previous, self.current_fold = self.current_fold, fold
        try:
            yield self
        finally:
            self.current_fold = previous

    def keys(self):
        return sorted(self.reads)


# --------------------------------------------------------------------------------- lagged dual ridge
def _lag(z, lags):
    """Torch mirror of ``isp.lagged``: row t holds z[t - lag] for each lag, zero-padded within this run."""
    n, out = z.shape[0], []
    for lag in lags:
        if lag == 0:
            out.append(z)
            continue
        shifted = torch.zeros_like(z)
        if abs(lag) < n:
            if lag > 0:
                shifted[lag:] = z[:-lag]
            else:
                shifted[:lag] = z[-lag:]
        out.append(shifted)
    return torch.cat(out, dim=1)


def lag_rows(runs, lags=LAGS, cols=None, device='cpu'):
    """[sum n, F * len(lags)] float64: each run's columns ``cols`` lagged within the run, runs concatenated."""
    blocks = []
    for z in runs:
        z = np.asarray(z)
        part = z if cols is None else z[:, cols]
        blocks.append(_lag(torch.as_tensor(part, device=device).to(torch.float64), lags))
    return torch.cat(blocks)


def _width(runs):
    widths = {int(np.shape(z)[1]) for z in runs}
    if len(widths) != 1:
        raise ValueError(f'Runs have different widths {sorted(widths)}')
    return widths.pop()


@torch.no_grad()
def gram(runs, other=None, lags=LAGS, device='cpu', chunk=CHUNK):
    """Linear kernel [N_runs, N_other] (float64) between the lagged designs of two run lists (``other=None``:
    ``runs`` with itself, symmetrized). The lagged design is never materialized beyond ``chunk`` features."""
    width = _width(runs)
    if other is not None and _width(other) != width:
        raise ValueError('Kernel sides have different widths')
    kernel = None
    for start in range(0, width, chunk):
        cols = slice(start, min(start + chunk, width))
        a = lag_rows(runs, lags, cols, device)
        b = a if other is None else lag_rows(other, lags, cols, device)
        term = a @ b.T
        kernel = term if kernel is None else kernel.add_(term)
        del a, b, term
    if other is None:
        kernel = (kernel + kernel.T) * 0.5
    return kernel


@torch.no_grad()
def gram_factor(kernel):
    """F [N, r] with F F^T = kernel (eigenvalues at or below N eps lambda_max dropped).

    Ridge with an intercept depends on its design only through the Gram matrix, so ``encoding_scores`` on F
    equals ``encoding_scores`` on any design whose linear kernel is ``kernel``.
    """
    k = torch.as_tensor(kernel, dtype=torch.float64)
    k = (k + k.T) * 0.5
    values, vectors = torch.linalg.eigh(k)
    keep = values > k.shape[0] * EPS64 * max(float(values[-1]), 0.0)
    return vectors[:, keep] * values[keep].sqrt()


class DualRidge:
    """Centered dual ridge on one kernel (run_v4's ``KernelRidgePath`` eigen-solve), for any outputs and alpha."""

    def __init__(self, kernel, device='cpu'):
        n = kernel.shape[0]
        self.path = KernelRidgePath(kernel, torch.zeros(n, 1, dtype=torch.float64, device=device), device)
        self.diagnostics = dict(self.path.diagnostics)

    def project(self, cross):
        """Centered cross-kernel [rows, N] times the eigenvectors."""
        return self.path.project(cross)

    def scaled(self, projected, alpha):
        """Projected rows divided by (eigenvalue + alpha): reusable across output chunks."""
        return projected / (self.path.lam + float(alpha))

    def apply(self, scaled, y, cols=None):
        """Predictions of the outputs ``y[:, cols]`` (fit rows x outputs, float64 tensor) from ``scaled`` rows."""
        y = y if cols is None else y[:, cols]
        mean = y.mean(0)
        return scaled @ (self.path.u.T @ (y - mean)) + mean

    def predict(self, projected, y, alpha, cols=None):
        """Ridge predictions of the outputs ``y[:, cols]`` at ``alpha``."""
        return self.apply(self.scaled(projected, alpha), y, cols)


def fold_rows(labels, folds, device='cpu'):
    """[(fit rows, held rows)] per fold (held-out group labels), as ``encoding_scores`` forms them."""
    labels = np.asarray([str(g) for g in labels])
    out = []
    for held in folds:
        mask = np.isin(labels, [str(g) for g in held])
        fit, hold = np.flatnonzero(~mask), np.flatnonzero(mask)
        if len(fit) < 2 or len(hold) == 0:
            raise ValueError('Encoding folds need fitting and held-out labeled rows')
        out.append(tuple(torch.as_tensor(r, device=device) for r in (fit, hold)))
    return out


@torch.no_grad()
def alpha_r(kernel, y, rows, alphas, device='cpu', chunk=CHUNK, dtype=torch.float64):
    """[len(alphas), C] pooled out-of-fold Pearson r of every output column of ``y`` at each alpha.

    ``encoding_scores`` in kernel space: per fold (``rows``: fit and held row tensors) the fitting rows' kernel
    gets run_v4's centered dual eigen-solve (``KernelRidgePath``: eigenvalues at or below n eps lambda_max
    dropped) and the held rows' predictions at alpha are S_a (Y_fit - mean) + mean with S_a = (centered K_HF U)
    diag(1 / (lambda + alpha)) U^T, so nothing per fold scales with U^T Y. ``dtype`` is the precision of the
    S_a Y products only (smoothers, statistics and r are float64). Same statistics as ``encoding_scores``.
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    n_cols, n_alpha = y.shape[1], len(alphas)
    stats = {k: torch.zeros(n_alpha, n_cols, dtype=torch.float64, device=device) for k in ('p', 'pp', 'pv')}
    v = torch.zeros(n_cols, dtype=torch.float64, device=device)
    vv = torch.zeros_like(v)
    count = 0
    for fit, hold in rows:
        path = KernelRidgePath(kernel.index_select(0, fit).index_select(1, fit).contiguous(),
                               torch.zeros(len(fit), 1, dtype=torch.float64, device=device), device)
        projected = path.project(kernel.index_select(0, hold).index_select(1, fit))
        smooth = torch.cat([(projected / (path.lam + float(a))) @ path.u.T for a in alphas]).to(dtype)
        n_hold = len(hold)
        count += n_hold
        del path, projected
        for start in range(0, n_cols, chunk):
            cols = slice(start, min(start + chunk, n_cols))
            block = y[:, cols]
            cf, actual = block.index_select(0, fit), block.index_select(0, hold)
            mean_c = cf.mean(0)
            pred = (smooth @ (cf - mean_c).to(dtype)).to(torch.float64).view(n_alpha, n_hold, -1).add_(mean_c)
            del cf
            stats['p'][:, cols] += pred.sum(1)
            stats['pp'][:, cols] += pred.square().sum(1)
            stats['pv'][:, cols] += (pred * actual).sum(1)
            v[cols] += actual.sum(0)
            vv[cols] += actual.square().sum(0)
            del pred
        del smooth
    n = float(count)
    cov = n * stats['pv'] - stats['p'] * v
    var = (n * stats['pp'] - stats['p'].square()) * (n * vv - v.square())
    return torch.where(var > 0, cov / var.clamp_min(1e-300).sqrt(), torch.zeros_like(cov))


def _best(curve):
    k = best_index(curve)
    if k is None:
        raise ValueError('No finite value on the alpha curve')
    return k


@torch.no_grad()
def kernel_alpha_scores(kernel, y, labels, folds, alphas=ALPHAS, device='cpu', chunk=CHUNK, precision='float64',
                        info=None):
    """(r per output, alpha, mean-r curve) = ``encoding_scores(gram_factor(kernel), y, all rows, labels, folds,
    alphas)`` to rounding, computed in kernel space (``alpha_r``); ties go to the larger alpha (1e-12).

    ``precision='mixed'``: the curve in float32 products, then float64 at the float32 choice and its two
    neighbours, the window growing by one alpha while the float64 maximum sits on a window edge that is not a
    grid edge; the alpha is the float64 maximum of the window and r its float64 r. ``info`` (a dict) receives
    the precision and, when mixed, the alpha indices computed in float64 and the float32 curve and choice.
    """
    if precision not in PRECISIONS:
        raise ValueError(f'precision must be one of {PRECISIONS}, not {precision!r}')
    alphas = np.asarray(alphas, dtype=np.float64)
    k = torch.as_tensor(kernel, dtype=torch.float64, device=device)
    k = (k + k.T) * 0.5
    y = torch.as_tensor(y, dtype=torch.float64, device=device)
    if k.shape != (len(y), len(y)) or len(labels) != len(y):
        raise ValueError(f'Kernel {tuple(k.shape)}, {len(labels)} labels and {len(y)} output rows disagree')
    rows = fold_rows(labels, folds, device)
    if precision == 'float64':
        r = alpha_r(k, y, rows, alphas, device, chunk)
        curve = r.mean(1).cpu().numpy()
        best = _best(curve)
        if info is not None:
            info.update(precision='float64')
        return r[best].cpu().numpy(), float(alphas[best]), curve
    curve = alpha_r(k, y, rows, alphas, device, chunk, torch.float32).mean(1).cpu().numpy()
    approx = curve.copy()
    first = _best(curve)
    lo, hi, exact = max(first - 1, 0), min(first + 1, len(alphas) - 1), {}
    while True:
        todo = [i for i in range(lo, hi + 1) if i not in exact]
        if todo:
            r64 = alpha_r(k, y, rows, alphas[todo], device, chunk)
            exact.update({i: r64[j] for j, i in enumerate(todo)})
            curve[todo] = r64.mean(1).cpu().numpy()
        best = lo + _best(curve[lo:hi + 1])
        if best == lo and lo > 0:
            lo -= 1
        elif best == hi and hi < len(alphas) - 1:
            hi += 1
        else:
            break
    if info is not None:
        info.update(precision='mixed', float64=sorted(exact), float32_choice=int(first),
                    float32_curve=[float(c) for c in approx])
    return exact[best].cpu().numpy(), float(alphas[best]), curve


def converter_alpha(kernel, y, stories, lengths, alphas=ALPHAS, device='cpu', precision='float64', info=None,
                    chunk=CHUNK):
    """(r per output, alpha, mean-r curve): pooled out-of-fold r over half-story groups (``kernel_alpha_scores``)."""
    labels, folds = row_groups(list(stories), list(lengths), 'half')
    if len(labels) != y.shape[0]:
        raise ValueError(f'{y.shape[0]} output rows for {len(labels)} story rows')
    return kernel_alpha_scores(kernel, y, labels, folds, alphas, device, chunk, precision, info)


# ------------------------------------------------------------------------------------------ cost model
def _halves(lengths):
    return [h for n in lengths for h in (int(n) // 2, int(n) - int(n) // 2)]


def converter_flops(lengths, n_outputs, n_alphas=len(ALPHAS)):
    """FLOP of one ``converter_alpha`` call over stories of ``lengths`` rows and ``n_outputs`` outputs, by term.

    Per half-story group (h held, f = N - h fitting rows): the alpha loop 2 A h f V (A alphas, V outputs), the
    smoothers 2 A h f^2 + 2 h f^2 and the eigen-solve EIGH_FLOP f^3. (The factor route, ``encoding_scores`` on
    ``gram_factor``, adds 2 f^2 V per group for U^T Y and an SVD of the f x N factor, and is not used.)
    """
    total, out = sum(int(n) for n in lengths), dict(alpha_loop=0.0, smoothers=0.0, eigh=0.0)
    for h in _halves(lengths):
        f = total - h
        out['alpha_loop'] += 2.0 * n_alphas * h * f * n_outputs
        out['smoothers'] += 2.0 * (n_alphas + 1) * h * f * f
        out['eigh'] += EIGH_FLOP * f ** 3
    out['total'] = sum(out.values())
    return out


def thsel_flops(calls, n_unheard, n_outputs, n_refs, n_inputs=REF_TOP * len(LAGS), n_alphas=len(ALPHAS),
                alpha_refs=None):
    """FLOP of TH_SEL's selections in one cell, by term: ``calls`` lists each selection's fitting-story lengths
    (every inner fold's fitting part and the whole budget), ``n_unheard`` the selection-material rows, ``n_refs``
    references with ``n_inputs`` lagged inputs each, ``n_outputs`` target voxels. ``alpha_refs`` (default
    ``n_refs``) is how many references need their converter alpha (fewer when ``alpha_memo`` reuses them)."""
    alpha_refs = n_refs if alpha_refs is None else alpha_refs
    out = dict(alpha=0.0, gram=0.0, final=0.0)
    for lengths in calls:
        n = float(sum(int(x) for x in lengths))
        out['alpha'] += alpha_refs * converter_flops(lengths, n_outputs, n_alphas)['total']
        out['gram'] += n_refs * 2.0 * (n * n + n_unheard * n) * n_inputs
        out['final'] += n_refs * (EIGH_FLOP * n ** 3 + 2.0 * n * n * n_outputs + 2.0 * n_unheard * n * n_outputs)
    out['total'] = sum(out.values())
    return out


def _stack(runs, device):
    return torch.cat([torch.as_tensor(np.asarray(z), device=device).to(torch.float64) for z in runs])


def _alpha_info(alpha, curve, alphas):
    curve = np.asarray(curve, dtype=np.float64)
    k = int(np.flatnonzero(np.asarray(alphas) == alpha)[0])  # The chosen alpha is a grid value.
    return dict(alpha=float(alpha), alpha_at_grid_edge=k in (0, len(alphas) - 1), oof_mean_r=float(curve[k]),
                curve=[float(v) for v in curve])


def _zcols(p):
    """Columns centered and scaled to unit population SD; columns with SD <= 1e-8 RMS become 0."""
    c = p - p.mean(0)
    sd = c.square().mean(0).sqrt()
    rms = p.square().mean(0).sqrt().clamp_min(1e-12)
    ok = sd > 1e-8 * rms
    return torch.where(ok, c / torch.where(ok, sd, torch.ones_like(sd)), torch.zeros_like(c))


# ------------------------------------------------------------------------------------------- TH_SEL
def digest(arrays):
    """Content digest of a sequence of arrays (shape, dtype and bytes), for ``alpha_memo`` keys."""
    h = hashlib.blake2b(digest_size=20)
    for a in arrays:
        a = np.ascontiguousarray(a.cpu().numpy() if torch.is_tensor(a) else np.asarray(a))
        h.update(json.dumps([list(a.shape), a.dtype.str]).encode())
        h.update(memoryview(a).cast('B'))
    return h.hexdigest()


@torch.no_grad()
def thsel_scores(target_runs, reference_runs, reference_unheard, stories, unheard, alphas=ALPHAS, device='cpu',
                 lags=LAGS, chunk=CHUNK, precision='float64', alpha_memo=None):
    """(score[V], info): reverse-converter agreement of every target voxel (TH_SEL steps 2-4).

    ``target_runs[story]``: [n, V] per-run z-scored target BOLD at the fitting stories (all voxels).
    ``reference_runs[ref][story]``: [n, Vk] per-run z-scored BOLD of the reference's selected voxels, same stories.
    ``reference_unheard[ref][stim]``: [m, Vk] the same voxels on material the target did not hear.
    Rows follow ``stories`` and ``unheard`` order. The score is the mean over reference pairs of the Pearson r
    of their predicted responses on the unheard material, (|sum_r z_r|^2 - sum_r |z_r|^2) / (T R (R - 1)) with
    z_r the column-standardized predictions (a constant prediction contributes r = 0).
    ``alpha_memo`` (a dict shared across calls, e.g. the reference sets of one run) keeps each converter's
    (alpha, curve, details) under the content digest of its fitting arrays, alphas, lags and precision: a
    reference's converter alpha depends only on that reference and the target, not on the other references.
    """
    refs, stories, unheard = sorted(reference_runs), list(stories), list(unheard)
    alphas = np.asarray(alphas, dtype=np.float64)
    if len(refs) < 2:
        raise NotEstimable('TH_SEL needs at least two references (mean pairwise correlation)')
    if not unheard:
        raise NotEstimable('TH_SEL needs material the target did not hear; this cell has none')
    if not stories:
        raise NotEstimable('TH_SEL needs fitting stories')
    if sorted(reference_unheard) != refs:
        raise ValueError('Every reference needs its selection material')
    if set(unheard) & set(stories):
        raise ValueError(f'Selection material {sorted(set(unheard) & set(stories))} is converter material')
    lengths = [len(target_runs[s]) for s in stories]
    u_lengths = [len(reference_unheard[refs[0]][u]) for u in unheard]
    for ref in refs:
        if [len(reference_runs[ref][s]) for s in stories] != lengths:
            raise ValueError(f'{ref}: story lengths differ from the target')
        if [len(reference_unheard[ref][u]) for u in unheard] != u_lengths:
            raise ValueError(f'{ref}: selection material lengths differ between references')
    y = _stack([target_runs[s] for s in stories], device)
    n_voxels, total = y.shape[1], int(sum(u_lengths))
    fits, info = {}, dict(references=refs, stories=stories, unheard=unheard, n_fit=int(len(y)), n_unheard=total,
                          precision=precision, converters={})
    target_key = None if alpha_memo is None else digest([target_runs[s] for s in stories])
    for ref in refs:
        fit_runs = [reference_runs[ref][s] for s in stories]
        kernel = gram(fit_runs, lags=lags, device=device, chunk=chunk)
        key = None if alpha_memo is None else (target_key, digest(fit_runs), tuple(stories), tuple(lengths),
                                               tuple(float(a) for a in alphas), tuple(lags), precision)
        if key is not None and key in alpha_memo:
            alpha, curve, details = alpha_memo[key]
            details = dict(details, memo=True)
        else:
            details = {}
            _, alpha, curve = converter_alpha(kernel, y, stories, lengths, alphas, device, precision, details, chunk)
            details = {k: v for k, v in details.items() if k != 'float32_curve'}
            if key is not None:
                alpha_memo[key] = (alpha, curve, details)
        ridge = DualRidge(kernel, device)
        cross = gram([reference_unheard[ref][u] for u in unheard], fit_runs, lags, device, chunk)
        fits[ref] = (ridge, ridge.scaled(ridge.project(cross), alpha))
        info['converters'][ref] = dict(_alpha_info(alpha, curve, alphas), n_inputs=int(np.shape(fit_runs[0])[1]),
                                       rank=int(ridge.diagnostics['rank']), **details)
        del kernel, cross
    pairs = len(refs) * (len(refs) - 1)
    score = np.empty(n_voxels)
    for start in range(0, n_voxels, chunk):
        cols = slice(start, min(start + chunk, n_voxels))
        total_z, squares = None, None
        for ref in refs:
            ridge, scaled = fits[ref]
            z = _zcols(ridge.apply(scaled, y, cols))
            total_z = z if total_z is None else total_z.add_(z)
            sq = z.square().sum(0)
            squares = sq if squares is None else squares.add_(sq)
            del z
        score[cols] = ((total_z.square().sum(0) - squares) / (total * pairs)).cpu().numpy()
    info.update(score_mean=float(score.mean()), score_max=float(score.max()))
    return score, info


class ThSelector:
    """Per-fit TH_SEL selection for run_v4's ISP slot (``selection_priors``).

    Each call refits the converters from the given fitting stories (the fold's fitting part of B inside the
    engine's inner CV, all of B at the final fit), logs the reads on ``prov`` (tagged with the current inner
    fold) and returns (mask as 0/1 floats over target voxels, info). ``target_bold(stimulus)`` returns per-run
    z-scored target BOLD [n, V]; ``reference_bold(ref, stimulus)`` per-run z-scored BOLD of the reference's
    selected voxels [n, Vk]; ``unheard`` is the selection material; ``k`` the number of selected voxels.
    ``precision`` and ``alpha_memo`` go to ``thsel_scores`` (pass one memo dict to the selectors of several
    reference sets of the same target, clock and budget to fit each reference's converter alpha once).
    """

    def __init__(self, target, target_bold, references, reference_bold, unheard, k, prov, alphas=ALPHAS,
                 device='cpu', chunk=CHUNK, precision='float64', alpha_memo=None):
        if precision not in PRECISIONS:
            raise ValueError(f'precision must be one of {PRECISIONS}, not {precision!r}')
        self.target, self.target_bold, self.references = str(target), target_bold, sorted(references)
        self.reference_bold, self.unheard, self.k, self.prov = reference_bold, list(unheard), int(k), prov
        self.alphas, self.device, self.chunk = np.asarray(alphas, dtype=np.float64), device, chunk
        self.precision, self.alpha_memo = precision, alpha_memo

    def scores(self, stories):
        target = {s: self.target_bold(s) for s in stories}
        refs = {ref: {s: self.reference_bold(ref, s) for s in stories} for ref in self.references}
        unheard = {ref: {u: self.reference_bold(ref, u) for u in self.unheard} for ref in self.references}
        log_reads(self.prov, thsel_reads(self.target, list(refs), list(target), self.unheard))
        return thsel_scores(target, refs, unheard, list(target), self.unheard, self.alphas, self.device,
                            chunk=self.chunk, precision=self.precision, alpha_memo=self.alpha_memo)

    def __call__(self, stories, misaligned=False):
        if misaligned:
            raise ValueError('TH_SEL has no misaligned variant')
        score, info = self.scores(list(stories))
        mask = selection_mask(score, self.k)
        info.update(rule='uniform on the top-k TH_SEL voxels at every delay', k=self.k, n_voxels=int(len(mask)),
                    threshold=float(score[mask].min()))
        return mask.astype(np.float64), info


class RandomSelector:
    """TH_SEL_rand for run_v4's ISP slot: the subject's registered random mask; it reads nothing."""

    def __init__(self, n_voxels, k, subject):
        self.mask, self.subject = random_mask(n_voxels, k, subject), str(subject)

    def __call__(self, stories, misaligned=False):
        if misaligned:
            raise ValueError('TH_SEL_rand has no misaligned variant')
        return self.mask.astype(np.float64), dict(rule='uniform on a random mask at every delay', seed=rand_seed(
            self.subject), k=int(self.mask.sum()), n_voxels=int(len(self.mask)))


_CLASSES = {}


def selection_priors(tgt, prov, device, selector):
    """A run_v4 ``Priors`` whose ISP slot returns ``selector(stories)`` (a subclass; run_v4 is not modified).

    With arm 'PRIOR' (``ENGINE_ARM``), run_v4's ``run_cv`` and ``final_fit`` then fit the selection-mask
    decoder exactly as every other arm, recomputing the selection from each fold's fitting stories.
    """
    if 'priors' not in _CLASSES:
        base = engine().Priors

        class SelectionPriors(base):
            def __init__(self, tgt, prov, device, selector):
                super().__init__(tgt, None, prov, device, ())
                self.selector = selector

            def isp(self, stories, misaligned=False):
                return self.selector(list(stories), misaligned=misaligned)

        _CLASSES['priors'] = SelectionPriors
    return _CLASSES['priors'](tgt, prov, device, selector)


# ----------------------------------------------------------------------------------- reference models
def target_like(subject, alphas, device, run_sessions, n_tr, **extra):
    """The Target attributes run_v4's ``run_cv`` / ``fold_maps`` read (subject, alphas, device, run sessions,
    TRs per story), for a reference held in memory."""
    return SimpleNamespace(subject=str(subject), alphas=np.asarray(alphas, dtype=np.float64),
                           device=torch.device(device), run_sessions=dict(run_sessions),
                           n_tr={str(k): int(v) for k, v in n_tr.items()}, **extra)


def budget_from_arrays(tgt, name, stories, raw, y, story, tr, run, session, eval_raw=None):
    """A genuine run_v4 ``Budget`` over in-memory rows (its file-loading constructor bypassed).

    ``raw[d]``: [N, V] delay-d design rows (the cache representation: raw BOLD at t + d, non-finite -> 0), or
    ``raw(d)`` returning them (loaded one delay at a time, as run_v4's Budget does); ``y``: [N, D] targets;
    ``story``/``tr``/``run``/``session``: per-row stimulus, target TR, run id and session;
    ``eval_raw[split][d]``: optional evaluation rows (for ``final_fit``).
    """
    v4 = engine()
    n = len(y)
    if any(len(a) != n for a in (story, tr, run, session)):
        raise ValueError('Budget arrays need one entry per row')
    load = raw if callable(raw) else raw.__getitem__
    bd = v4.Budget.__new__(v4.Budget)
    bd.tgt, bd.name, bd.stories = tgt, name, [str(s) for s in stories]
    bd.rows = np.arange(n)
    bd.raw = {}
    for d in DELAYS:
        x = load(d)
        x = x.to(device=tgt.device, dtype=torch.float32) if torch.is_tensor(x) else torch.as_tensor(
            np.asarray(x, dtype=np.float32), device=tgt.device)
        if x.ndim != 2 or x.shape[0] != n:
            raise ValueError(f'Delay {d} rows have shape {tuple(x.shape)}; the budget has {n} rows')
        bd.raw[d] = x
        del x
    bd.eval_raw = {s: {d: torch.as_tensor(np.asarray(v[d], dtype=np.float32), device=tgt.device) for d in DELAYS}
                   for s, v in (eval_raw or {}).items()}
    bd.y = np.asarray(y, dtype=np.float64)
    bd.story, bd.tr, bd.run = np.asarray(story).astype(str), np.asarray(tr, dtype=np.int64), np.asarray(run)
    bd.session = list(session)
    missing = sorted(set(bd.story) - set(bd.stories))
    if missing:
        raise ValueError(f'Rows of stories {missing} outside the budget')
    return bd


def budget_columns(bd, voxels):
    """Shallow copy of a run_v4 Budget(-like) keeping only ``voxels`` at every delay (no evaluation rows)."""
    out = copy.copy(bd)
    idx = torch.as_tensor(np.asarray(voxels, dtype=np.int64), device=bd.raw[DELAYS[0]].device)
    out.raw = {d: bd.raw[d].index_select(1, idx) for d in DELAYS}
    out.eval_raw = {}
    return out


def session_rule_folds(bd, all_sessions=None):
    """§12 decoder folds over a budget's rows: (held story tuples, held session tuples).

    ``budget_folds`` over the fixed five-fold session partition of ``all_sessions`` (default: the budget's
    sessions; run_v4 uses the subject's training sessions), as run_v4's ALL rule.
    """
    sessions = list(bd.session)
    fixed = fixed_folds(list(all_sessions) if all_sessions is not None else sessions)
    folds, rule = budget_folds(sessions, range(len(sessions)), fixed)
    if folds is None:
        raise NotEstimable(f'reference decoder: {rule}')
    story_folds = [tuple(sorted({str(bd.story[i]) for i, s in enumerate(sessions) if s in set(f)})) for f in folds]
    return story_folds, [tuple(f) for f in folds]


@dataclass
class LinearDecoder:
    """A run_v4 W1 decoder in primal form: prediction = sum_d (((x_d - mean_d) / std_d) w_d - center_d) beta_d
    + y_mean, identical to the engine's centered dual kernel ridge on the same rows, weights and alpha."""
    subject: str
    voxels: np.ndarray
    mean: dict
    std: dict
    weight: dict
    center: dict
    beta: dict
    y_mean: np.ndarray
    y_std: np.ndarray
    selection: dict
    info: dict = field(default_factory=dict)
    delays: tuple = DELAYS

    @torch.no_grad()
    def predict(self, rows, device='cpu'):
        """[M, D] predictions from input rows ``{d: [M, len(voxels)]}`` in the decoder's input representation."""
        out = None
        for d in self.delays:
            x = rows[d] if torch.is_tensor(rows[d]) else torch.as_tensor(np.asarray(rows[d]))
            x = x.to(device=device, dtype=torch.float64)
            if x.ndim != 2 or x.shape[1] != len(self.voxels):
                raise ValueError(f'Delay {d} rows have shape {tuple(x.shape)}; the decoder has {len(self.voxels)} '
                                 'voxels')
            t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
            e = ((x - t(self.mean[d])) / t(self.std[d])) * t(self.weight[d]) - t(self.center[d])
            term = e @ t(self.beta[d])
            out = term if out is None else out.add_(term)
        return (out + torch.as_tensor(self.y_mean, dtype=torch.float64, device=device)).cpu().numpy()

    def arrays(self, prefix='dec_'):
        out = {f'{prefix}voxels': np.asarray(self.voxels), f'{prefix}y_mean': self.y_mean,
               f'{prefix}y_std': self.y_std}
        for d in self.delays:
            for name in ('mean', 'std', 'weight', 'center', 'beta'):
                out[f'{prefix}{name}_{d}'] = np.asarray(getattr(self, name)[d])
        return out

    def meta(self):
        return dict(subject=self.subject, selection=self.selection, info=self.info, delays=list(self.delays))

    @classmethod
    def from_arrays(cls, arrays, meta, prefix='dec_'):
        delays = tuple(int(d) for d in meta['delays'])
        get = lambda name: {d: np.asarray(arrays[f'{prefix}{name}_{d}']) for d in delays}
        return cls(meta['subject'], np.asarray(arrays[f'{prefix}voxels']), get('mean'), get('std'), get('weight'),
                   get('center'), get('beta'), np.asarray(arrays[f'{prefix}y_mean']),
                   np.asarray(arrays[f'{prefix}y_std']), meta['selection'], meta.get('info', {}), delays)


@torch.no_grad()
def linear_decoder(tgt, bd, name, session_folds, selection, pri, voxels=None):
    """The engine's final W1 fit (``fold_maps`` -> ``arm_weights`` -> ``build_kernels`` -> ``KernelRidgePath``)
    over all of ``bd`` at ``selection`` (gamma, alpha), returned in primal form."""
    v4 = engine()
    all_pos = np.arange(len(bd.rows))
    maps, _ = v4.fold_maps(tgt, bd, name, all_pos, session_folds or [], None, ['W1'], pri, None)
    ws = v4.arm_weights('W1', maps, tgt.subject)
    if ws is None or selection['gamma'] not in ws:
        raise NotEstimable('reference decoder: W1 weights not estimable on the whole labeled set')
    w, meta = ws[selection['gamma']]
    kernels, _ = v4.build_kernels(bd, all_pos, {'W1': w}, all_pos, prov=pri.prov)
    y = torch.as_tensor(bd.y, dtype=torch.float64, device=tgt.device)
    path = KernelRidgePath(kernels.pop('W1'), y, tgt.device)
    coef = path.u @ (path.uty / (path.lam + float(selection['alpha']))[:, None])  # [N, D] centered dual weights
    parts = {k: {} for k in ('mean', 'std', 'weight', 'center', 'beta')}
    for d in DELAYS:
        x, (mean, std) = bd.standardized(d, all_pos, pri.prov)
        wd = torch.as_tensor(w[d], dtype=torch.float64, device=x.device)
        x.mul_(wd[None, :])
        center = x.mean(0)
        parts['beta'][d] = ((x - center).T @ coef).cpu().numpy()
        parts['mean'][d], parts['std'][d] = mean.cpu().numpy(), std.cpu().numpy()
        parts['weight'][d], parts['center'][d] = np.asarray(w[d], dtype=np.float64), center.cpu().numpy()
        del x
    n_vox = parts['mean'][DELAYS[0]].shape[0]
    return LinearDecoder(str(tgt.subject), np.arange(n_vox) if voxels is None else np.asarray(voxels), parts['mean'],
                         parts['std'], parts['weight'], parts['center'], parts['beta'],
                         path.y_mean.cpu().numpy(), np.maximum(bd.y.std(0), 1e-6), dict(selection),
                         dict(weights=meta, ridge=dict(path.diagnostics)))


@dataclass
class ReferenceModel:
    """One reference's TS/SS model: its top voxels (and optionally its W1 decoder) with the reads behind them."""
    subject: str
    stimuli: list
    voxels: np.ndarray
    scores: np.ndarray
    info: dict
    reads: list
    decoder: LinearDecoder | None = None

    def save(self, path):
        meta = dict(subject=self.subject, stimuli=self.stimuli, info=self.info, reads=[list(r) for r in self.reads],
                    decoder=None if self.decoder is None else self.decoder.meta())
        arrays = dict(meta=np.array(json.dumps(meta)), voxels=self.voxels, scores=self.scores)
        if self.decoder is not None:
            arrays.update(self.decoder.arrays())
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + '.tmp.npz')
        np.savez(tmp, **arrays)
        tmp.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            meta = json.loads(str(z['meta']))
            decoder = None if meta['decoder'] is None else LinearDecoder.from_arrays(z, meta['decoder'])
            return cls(meta['subject'], meta['stimuli'], z['voxels'], z['scores'], meta['info'],
                       [tuple(r) for r in meta['reads']], decoder)

    def summary(self):
        out = dict(subject=self.subject, stimuli=len(self.stimuli), labeled_rows=self.info.get('labeled_rows'),
                   voxels=int(len(self.voxels)), score_threshold=float(self.scores[self.voxels].min()),
                   encoding_alpha=self.info.get('encoding_alpha'))
        if 'splits' in self.info:  # The reference's own split of each labeled story (ds003020 TS labels).
            out['splits'] = self.info['splits']
        if self.decoder is not None:
            out['decoder_selection'] = self.decoder.selection
        return out


@torch.no_grad()
def fit_reference_model(tgt, bd, name='ALL', folds=None, k=REF_TOP, decoder=True, log_fold=None, all_sessions=None):
    """A reference's TS/SS model from its labeled rows with run_v4's engine.

    ``tgt``: the reference's Target(-like); ``bd``: a run_v4 Budget(-like) over its labeled rows (all voxels).
    1. ``fold_maps`` (arm W1) over all rows gives the nested out-of-fold encoding r per (voxel, delay); the
       voxel score is the maximum over delays and the top ``k`` voxels are kept.
    2. With ``decoder``, ``run_cv`` selects (gamma, alpha) of a W1 decoder on those voxels and
       ``linear_decoder`` refits it on all rows. ``name`` 'ALL' (default) is the §12 rule: session folds
       (``folds`` = (story tuples, session tuples); default ``session_rule_folds``) with nested encoding over
       the remaining session folds; an LPP budget name ('L8') uses leave-one-section-out decoder folds and
       half-section nested groups instead (``folds`` = (story tuples, None)).
    Every engine read is kept (``Recorder``) for the target cell to log as 'reference_model'.
    """
    v4 = engine()
    rec = Recorder()
    pri = v4.Priors(tgt, None, rec, tgt.device, ())
    if folds is None:
        folds = session_rule_folds(bd, all_sessions) if name == 'ALL' else (
            [tuple(f) for f in inner_folds(name, bd.stories)], None)
    story_folds, session_folds = folds
    session_folds = session_folds if name == 'ALL' else None
    all_pos = np.arange(len(bd.rows))
    maps, info = v4.fold_maps(tgt, bd, name, all_pos, session_folds or [], None, ['W1'], pri, None)
    if 'own' not in maps:
        raise NotEstimable('reference encoding: fewer than two encoding folds')
    scores = np.max(np.stack([np.asarray(maps['own'][d], dtype=np.float64) for d in DELAYS]), axis=0)
    voxels = top_k(scores, min(int(k), len(scores)))
    model_info = dict(labeled_rows=int(len(bd.rows)), rule=name, encoding_alpha={str(d): float(a) for d, a in
                                                                                  info['own_alpha'].items()},
                      voxel_score='max over delays of the nested out-of-fold W1 encoding r', k=int(len(voxels)),
                      decoder_folds=[list(f) for f in story_folds])
    dec = None
    if decoder:
        sub = budget_columns(bd, voxels)
        selections, curves, logs = v4.run_cv(tgt, sub, name, ['W1'], pri, rec, None, story_folds, session_folds,
                                             log_fold or (lambda entry: None))
        if selections.get('W1') is None:
            raise NotEstimable('reference decoder: no estimable W1 selection')
        dec = linear_decoder(tgt, sub, name, session_folds, selections['W1'], pri, voxels)
        kept = ('held', 'n_fit', 'n_held', 'seconds', 'estimable')
        dec.info.update(cv_curves=curves['W1'], folds=[{key: e[key] for key in kept if key in e} for e in logs])
        del sub
    return ReferenceModel(str(tgt.subject), [str(s) for s in bd.stories], voxels, scores, model_info, rec.keys(), dec)


# --------------------------------------------------------------------------------------------- XDEC
def delay_rows(series, trs, delays=DELAYS):
    """{d: series[trs + d]}: the decoder input rows of target bins at TRs ``trs`` of one run."""
    trs = np.asarray(trs, dtype=np.int64)
    if len(trs) and (trs.min() + min(delays) < 0 or trs.max() + max(delays) >= len(series)):
        raise ValueError(f'Delayed rows outside a run of {len(series)} TRs')
    idx = torch.as_tensor(trs, device=series.device)
    return {d: series.index_select(0, idx + d) for d in delays}


@torch.no_grad()
def xdec_predict(target_runs, reference_runs, decoders, stories, eval_runs, eval_stimulus, eval_tr, alphas=ALPHAS,
                 device='cpu', lags=LAGS, chunk=CHUNK):
    """(prediction [M, D], info): XDEC = mean over references of decoder_r(converter_r(target BOLD)).

    ``target_runs[story]``: [n, V] per-run z-scored target BOLD at the budget stories (all voxels).
    ``reference_runs[ref][story]``: [n, Vk] the reference's decoder voxels (``decoders[ref].voxels`` order) at
    the same stories, in the decoder's input representation. ``eval_runs[stim]``: [n_e, V] per-run z-scored
    target BOLD of each evaluation stimulus; ``eval_stimulus``/``eval_tr``: per evaluation row, its stimulus
    and target TR (the decoder reads converted BOLD at TR + 1..3). No target label is used.
    """
    refs, stories = sorted(decoders), list(stories)
    alphas = np.asarray(alphas, dtype=np.float64)
    if not refs:
        raise NotEstimable('XDEC needs at least one reference decoder')
    if sorted(reference_runs) != refs:
        raise ValueError('Every reference decoder needs its converter outputs')
    eval_stimulus = np.asarray(eval_stimulus).astype(str)
    eval_tr = np.asarray(eval_tr, dtype=np.int64)
    order = list(dict.fromkeys(eval_stimulus.tolist()))
    if set(order) & set(stories):
        raise ValueError(f'Evaluation material {sorted(set(order) & set(stories))} is inside the budget')
    missing = [e for e in order if e not in eval_runs]
    if missing:
        raise ValueError(f'No target BOLD for evaluation stimuli {missing}')
    lengths = [len(target_runs[s]) for s in stories]
    fit_runs = [target_runs[s] for s in stories]
    for ref in refs:
        if [len(reference_runs[ref][s]) for s in stories] != lengths:
            raise ValueError(f'{ref}: story lengths differ from the target')
        width = {int(np.shape(reference_runs[ref][s])[1]) for s in stories}
        if width != {len(decoders[ref].voxels)}:
            raise ValueError(f'{ref}: {sorted(width)} converter outputs for a decoder on {len(decoders[ref].voxels)} '
                             'voxels')
    kernel = gram(fit_runs, lags=lags, device=device, chunk=chunk)
    ridge = DualRidge(kernel, device)
    projected = {e: ridge.project(gram([eval_runs[e]], fit_runs, lags, device, chunk)) for e in order}
    positions = {e: np.flatnonzero(eval_stimulus == e) for e in order}
    labels, folds = row_groups(stories, lengths, 'half')
    rows_order = np.concatenate([positions[e] for e in order])
    info = dict(references=refs, stories=stories, eval_stimuli=order, n_fit=int(sum(lengths)),
                rank=int(ridge.diagnostics['rank']), precision='float64', converters={})
    # One kernel for every reference: the alpha curves of a group of references (at most XDEC_OUTPUTS stacked
    # outputs) share the per-group eigen-solves; each reference's curve is the mean r over its own outputs.
    groups = []
    for ref in refs:
        if not groups or sum(len(decoders[r].voxels) for r in groups[-1]) + len(decoders[ref].voxels) > XDEC_OUTPUTS:
            groups.append([])
        groups[-1].append(ref)
    rows_by_fold, total = fold_rows(labels, folds, device), None
    for group in groups:
        y_all = torch.cat([_stack([reference_runs[ref][s] for s in stories], device) for ref in group], dim=1)
        r_all = alpha_r(kernel, y_all, rows_by_fold, alphas, device, chunk)
        start = 0
        for ref in group:
            dec = decoders[ref]
            cols = slice(start, start + len(dec.voxels))
            start += len(dec.voxels)
            y = y_all[:, cols]
            curve = r_all[:, cols].mean(1).cpu().numpy()
            alpha = float(alphas[_best(curve)])
            rows = {d: [] for d in dec.delays}
            for e in order:
                series = ridge.predict(projected[e], y, alpha)
                for d, block in delay_rows(series, eval_tr[positions[e]], dec.delays).items():
                    rows[d].append(block)
            pred = dec.predict({d: torch.cat(v) for d, v in rows.items()}, device)
            total = pred if total is None else total + pred
            info['converters'][ref] = dict(_alpha_info(alpha, curve, alphas), n_outputs=int(y.shape[1]))
            del y, rows
        del y_all, r_all
    out = np.empty_like(total)
    out[rows_order] = total / len(refs)
    return out, info


def scoring_stats(decoders):
    """Label mean and SD for the secondary endpoints of an XDEC cell: the references' label statistics averaged
    over references (no target label is read)."""
    decs = [decoders[r] for r in sorted(decoders)]
    return np.mean([d.y_mean for d in decs], axis=0), np.mean([d.y_std for d in decs], axis=0)


# ------------------------------------------------------------------------------------------- audit
XDEC_EXEMPT = ('no untagged (final) target brain decoder_fit read', 'no untagged (final) target transcript decoder_fit '
               'read', 'inner folds None are not the registered')
XDEC_ZERO = ('target_pairs_dense', 'target_pairs_stride4', 'transcript_words')  # Ledger counts XDEC keeps at 0.


def reads_uncovered(p, subject, kind, purpose, items, fold=ANY_FOLD):
    """``Provenance.uncovered`` for the reads of any ``subject`` (a reference too): the ``items`` not covered by
    ``subject``'s reads of ``kind`` for ``purpose`` (``fold``: that inner fold's tagged reads; None: untagged)."""
    spans, fold = {}, fold if fold in (ANY_FOLD, None) else str(fold)
    for r in p.records:
        if r.subject == str(subject) and r.kind == kind and r.purpose == purpose and (fold is ANY_FOLD
                                                                                      or r.fold == fold):
            spans.setdefault(r.stimulus, []).append(r.trs)
    return [i for i in map(_item, items) if not _covered(spans.get(i.stimulus, []), i.trs)]


def model_subjects(p):
    """Sorted references with reference-model reads in log ``p``."""
    return sorted({r.subject for r in p.records if r.subject != p.target and r.purpose == 'reference_model'})


def converter_errors(p, subjects, items, fold, where):
    """Missing brain 'converter' reads of ``subjects`` over ``items`` at ``fold`` (None: untagged)."""
    out = []
    for subject in subjects:
        missing = [i.stimulus for i in reads_uncovered(p, subject, 'brain', 'converter', items, fold)]
        if missing:
            who = 'target' if subject == p.target else f'reference {subject}'
            out.append(f'{where}: no {"untagged" if fold is None else "tagged"} {who} brain converter read for '
                       f'{missing}')
    return out


def thsel_errors(cell, p):
    """TH_SEL: at least two reference models (those the cell lists), and the selection refit inside every
    registered inner fold from the fold's fitting stories (tagged target and reference converter reads over
    them; R7 already forbids any on the held-out part) and on the whole budget for the final fit (untagged).
    A selection computed once on all of B and reused in every fold leaves no tagged converter read and fails."""
    errors, refs = [], model_subjects(p)
    if len(refs) < 2:
        errors.append(f'TH_SEL needs reference models of at least two references, the log has {refs}')
    listed = cell.get('references')
    if isinstance(listed, list) and sorted(str(r) for r in listed) != refs:
        errors.append(f'cell references {sorted(listed)} are not the log reference models {refs}')
    if (cell.get('stage') or p.stage) not in ('develop', 'evaluate'):
        return errors
    subjects = [p.target] + refs
    errors += converter_errors(p, subjects, p.budget, None, 'final TH_SEL selection')
    for k, held in (p.folds or {}).items():
        out = {i.stimulus for i in held}
        errors += converter_errors(p, subjects, [i for i in p.budget if i.stimulus not in out], k,
                                   f'inner fold {k} TH_SEL selection')
    return errors


def xdec_errors(cell, p):
    """XDEC (zero target labels): no inner folds and no fold-tagged read; reference models (those the cell lists);
    target and every reference's BOLD read for 'converter' over the whole budget; target BOLD read only as
    converter input or decoder input
    ('decoder_score'); every target transcript/audio read a context reset or the scoring of a scored item; and a
    ledger with zero target labeled pairs and transcript words."""
    errors = []
    if p.folds is not None:
        errors.append('an XDEC log declares inner folds, but XDEC fits nothing on target labels')
    refs = model_subjects(p)
    if not refs:
        errors.append('an XDEC log has no reference model')
    listed = cell.get('references')
    if isinstance(listed, list) and sorted(str(r) for r in listed) != refs:
        errors.append(f'cell references {sorted(listed)} are not the log reference models {refs}')
    errors += converter_errors(p, [p.target] + refs, p.budget, None, 'XDEC converters')
    brain = sorted({(r.stimulus, r.purpose) for r in p.records if r.subject == p.target and r.kind == 'brain'
                    and r.purpose not in ('converter', 'decoder_score')})
    if brain:
        errors.append(f'XDEC read target BOLD beyond its converters and decoder input: {brain}')
    labels = sorted({r.stimulus for r in p.records if r.subject == p.target and r.kind != 'brain'
                     and r.purpose != 'context_reset' and not (
                         r.purpose == 'decoder_score' and _covered([i.trs for i in p.score_items
                                                                    if i.stimulus == r.stimulus], r.trs))})
    if labels:
        errors.append(f'XDEC read target transcripts/audio beyond scoring the scored items at {labels}')
    if any(r.fold is not None for r in p.records):
        errors.append('XDEC reads tagged with an inner fold')
    ledger = cell.get('ledger')
    if not isinstance(ledger, dict):
        errors.append('an XDEC cell needs its ledger (zero target labels)')
    else:
        counted = {k: ledger.get(k) for k in XDEC_ZERO if ledger.get(k) != 0}
        if ledger.get('transcript_tokens') not in (None, 0):
            counted['transcript_tokens'] = ledger['transcript_tokens']
        if counted:
            errors.append(f'the XDEC ledger counts target labels {counted}')
    if cell.get('target_labels', 0) != 0:
        errors.append(f"an XDEC cell with target_labels={cell.get('target_labels')!r}")
    return errors


def rand_errors(p):
    """TH_SEL_rand: the registered random mask reads nothing from any other subject."""
    foreign = sorted({r.subject for r in p.records if r.subject != p.target})
    return [f'TH_SEL_rand read data of {foreign}'] if foreign else []


def registered_selection(p, root=None):
    """TH_SEL's registered selection material for log ``p`` (sorted): ds003020 the stories every reference model
    of the log heard and the target never heard (``ds003020_selection`` over the manifests under ``root``),
    LPP ``lpp_unheard`` of the log's budget and single outer test section."""
    if p.dataset == 'ds003020':
        audit = engine('audit_v4')
        refs = model_subjects(p)
        return ds003020_selection(audit.manifests(str(audit.MANIFESTS if root is None else root)), p.target,
                                  refs) if refs else []
    test = sorted({i.stimulus for i in p.test_items})
    if len(test) != 1:
        raise ValueError(f'LPP test items {test} are not one outer section')
    return sorted(lpp_unheard(LPP_SECTIONS, [i.stimulus for i in p.budget], test[0]))


def material_errors(cell, p, root=None):
    """TH_SEL / XDEC: the dataset's registered reference rule (``rule_for``: TS in ds003020, SS in LPP); in
    ds003020 no reference-model read (labels or selection material, any kind) on a story the target heard (its
    whole manifest under ``root``); for TH_SEL the registered selection material (``registered_selection``) in
    every inner fold (the fold's tagged reference-model BOLD reads of every reference) and in the cell's
    ``selection_material`` (ds003020 runner) or ``unheard`` (LPP runner) field when given."""
    errors, arm = [], cell.get('arm')
    want = RULES.get(p.dataset)
    if p.reference_rule != want:
        errors.append(f'{arm} in {p.dataset} carries reference rule {p.reference_rule}; the registered rule is {want}')
    try:
        if p.dataset == 'ds003020':
            audit = engine('audit_v4')
            heard = {r['story'] for r in audit.manifests(str(audit.MANIFESTS if root is None else root))[p.target]
                     ['runs']}
            bad = sorted({(r.subject, r.stimulus) for r in p.records
                          if r.subject != p.target and r.purpose == 'reference_model' and r.stimulus in heard})
            if bad:
                errors.append(f'reference-model reads on stories the target heard: {bad}')
        if arm != 'TH_SEL':
            return errors
        want = registered_selection(p, root)
    except (OSError, KeyError, ValueError) as error:
        return errors + [f'registered selection material unavailable: {error!r}']
    if not want:
        return errors + ['TH_SEL has no registered selection material in this cell (not estimable)']
    for k in sorted(p.folds or {}):
        for ref in model_subjects(p):
            got = sorted({r.stimulus for r in p.records if r.subject == ref and r.kind == 'brain'
                          and r.purpose == 'reference_model' and r.fold == k})
            if got != want:
                errors.append(f'inner fold {k}: reference {ref} selection material {got} is not the registered {want}')
    for key in ('selection_material', 'unheard'):
        declared = cell.get(key)
        if declared is not None and (not isinstance(declared, list) or sorted(str(x) for x in declared) != want):
            errors.append(f'cell {key} {declared} is not the registered selection material {want}')
    return errors


def scope_errors(cell, p, root=None):
    """``audit_v4.scope_errors`` of one rebuilt log of a Module H cell plus its arm's own checks (empty if it
    passes).

    TH_SEL: the frozen checks plus ``thsel_errors`` (the selection refit inside every inner fold, the final
    selection on the whole budget, at least two reference models). TH_SEL_rand: plus ``rand_errors``. XDEC has
    no target decoder: the frozen audit's checks for a final target decoder_fit read and for the registered
    inner folds (``XDEC_EXEMPT``) cannot apply and are replaced by ``xdec_errors``. TH_SEL and XDEC: plus
    ``material_errors`` (registered rule, unheard reference-model material, registered selection material;
    ``root`` is the ds003020 manifest root, default ``audit_v4.MANIFESTS``). Any other arm: unchanged.
    """
    kwargs = {} if root is None else dict(root=root)
    errors = engine('audit_v4').scope_errors(cell, p, **kwargs)
    cell = cell if isinstance(cell, dict) else {}
    arm = cell.get('arm')
    if arm == 'XDEC':
        return [e for e in errors if not e.startswith(XDEC_EXEMPT)] + xdec_errors(cell, p) + material_errors(
            cell, p, root)
    if arm == 'TH_SEL':
        return errors + thsel_errors(cell, p) + material_errors(cell, p, root)
    if arm == 'TH_SEL_rand':
        return errors + rand_errors(p)
    return errors


def audit_errors(cell, p, root=None):
    """Cell-field and registered-scope errors of a Module H cell with log ``p`` (``audit_v4.cell_errors`` +
    ``scope_errors``); the runners write a cell only when this is empty."""
    return engine('audit_v4').cell_errors(cell) + scope_errors(cell, p, root)


def audit_file(path, root=None):
    """Drop-in for ``audit_v4.audit_file`` (same result dict) that audits Module H cells with ``scope_errors``.

    Identical to the frozen audit for any non-Module-H cell. For TH_SEL and TH_SEL_rand it is strictly stronger
    (the frozen audit does not see a selection hoisted out of the inner folds); XDEC cells (which read no target
    label) are re-audited rather than refused. The lock and the primary analysis should therefore pass it as
    their audit callable for every arm in ``ARMS``, not only XDEC.
    """
    audit = engine('audit_v4')
    out = dict(file=str(path), logs=0, violations=[], flags=0)
    try:
        cell = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        return dict(out, passed=False, error=f'unreadable: {error}')
    out['violations'] += [dict(rule='scope', path='$', reason=r) for r in audit.cell_errors(cell)]
    for where, log in audit.logs(cell):
        out['logs'] += 1
        try:
            p = Provenance.from_dict(log)
        except (KeyError, TypeError, ValueError) as error:
            out['violations'].append(dict(rule='invalid', path=where, reason=f'unreadable log: {error!r}'))
            continue
        result = p.audit()
        out['flags'] += len(result['flags'])
        out['violations'] += [dict(v, path=where) for v in result['violations']]
        out['violations'] += [dict(rule='scope', path=where, reason=r) for r in scope_errors(cell, p, root)]
    if not out['logs']:
        out['error'] = 'no provenance log'
    out['rules'] = sorted({v['rule'] for v in out['violations']})
    out['passed'] = bool(out['logs']) and not out['violations']
    return out
