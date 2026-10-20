"""PROTOCOL_V4 empirical-Bayes voxel weights (FUTURE_WORK_V4 §5, EB estimator; V4_IMPLEMENTATION Module C).

Per (target, budget, decoder-CV fold, delay), from that fold's labeled rows of B only:
  1. z = fisher(r), with r the W1 nested out-of-fold encoding scores;
  2. sigma^2 = mean over shifts k = 1..4 of (1.4826 * MAD(fisher(r0_k)))^2, where r0_k are the same scores
     recomputed with the 768-d targets circularly shifted within each run by round(k * n_run / 5) rows, n_run the
     run's labeled rows; every row must sit >= 40 TR from the row whose target it gets (PROTOCOL_V4 §2). The check
     is mandatory (``null_sigma2`` requires the rows' target TRs) because the registered formula can break it: on
     a contiguous clock with stride s, k = 1 gives round(n_run / 5) * s TR, below 40 when n_run < 198 dense rows
     (n_tr < 230 under TargetIndex's TRs 28..n_tr - 5) or n_run < 48 stride-4 rows (n_tr < 221).
     ``null_shift_offsets`` runs the same check without fitting, as a preflight over a whole budget;
  3. p = fisher(ISP), not standardized;
  4. calibration: OLS of z on p across columns gives (a, b); a negative slope sets b = 0, a = mean(z);
  5. shrinkage: V = Var(z - a - b p), tau^2 = max(V - sigma^2, 0.01 V), kappa = tau^2 / (tau^2 + sigma^2);
  6. posterior: m = a + b p + kappa (z - a - b p), evaluated as kappa z + (1 - kappa)(a + b p);
  7. weights: max(tanh(m), 0)^gamma, rescaled by the caller (``eb_column_weights`` applies the §12 rescaling and
     1% fallback of W1).
The convex form in step 6 is algebraically identical and makes the G0(b) identities exact in floating point:
kappa = 1 returns z itself (W1), and kappa = 0, a = 0, b = 1 returns p itself (PRIOR). Variances are population
variances (ddof = 0). MIX-CV replaces the closed-form kappa by one chosen by inner CV from ``MIXCV_KAPPAS``.
``reliability`` gives the §8.3 diagnostic quantities from an ``eb_posterior`` info record.
"""
from __future__ import annotations
import numpy as np
import torch
from ..encoding_weights import column_weights, encoding_scores

CLIP = 0.999
MAD_SCALE = 1.4826
SHIFTS = (1, 2, 3, 4)
SHIFT_PARTS = 5
MIN_SHIFT_TR = 40
TAU2_FLOOR = 0.01
MIXCV_KAPPAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def fisher(r):
    """atanh(clip(r, -0.999, 0.999)) in float64."""
    return np.arctanh(np.clip(np.asarray(r, dtype=np.float64), -CLIP, CLIP))


def mad_sigma2(r0):
    """(1.4826 * MAD(fisher(r0)))^2: robust noise variance of one set of null scores.

    Registered as is. Null atanh(r) (density proportional to sech^(n-2)) has positive excess kurtosis, so this is
    biased low at small n: -0.43% of Var at n = 100, -0.09% at 500, -0.02% at 2,000.
    """
    f = fisher(r0)
    return float((MAD_SCALE * np.median(np.abs(f - np.median(f)))) ** 2)


def clamp_shift(shift, members, tr, min_tr):
    """PROTOCOL_V4 §4 clarification: clamp a row shift into [lo, m - lo], lo = ceil(min_tr / TR-per-row), so that
    every shifted pairing (wrapped ones included) is at least ``min_tr`` TR apart; raises if no shift can be."""
    m = len(members)
    step = max(int(np.min(np.diff(np.asarray(tr)[members]))), 1) if m > 1 else 1
    lo = -(-int(min_tr) // step)
    if lo > m - lo:
        raise ValueError(f'null shifts below {min_tr} TR are unavoidable for a run of {m} rows ({step} TR per row)')
    return min(max(int(shift), lo), m - lo)


def shift_plan(rows, run_ids, k, parts=SHIFT_PARTS, tr=None, min_tr=None):
    """[(run, member rows in increasing index order, shift in rows = round(k * m / parts))], runs in order of first
    appearance in ``run_ids[rows]``; m is the run's number of rows in ``rows``. With ``tr`` and ``min_tr`` the shift is
    clamped (``clamp_shift``)."""
    run_ids, rows = np.asarray(run_ids), np.asarray(rows, dtype=np.int64)
    plan = []
    for run in dict.fromkeys(run_ids[rows].tolist()):
        members = np.sort(rows[run_ids[rows] == run])
        shift = int(round(k * len(members) / parts))
        if tr is not None and min_tr is not None:
            shift = clamp_shift(shift, members, tr, min_tr)
        plan.append((run, members, shift))
    return plan


def shifted_targets(targets, rows, run_ids, k, parts=SHIFT_PARTS, tr=None, min_tr=None):
    """Copy of ``targets`` (float64) with the rows of ``rows`` circularly shifted within each run.

    Within a run (``shift_plan``), the i-th of its m rows receives the target of row i - shift (``np.roll``). Rows
    outside ``rows`` are unchanged. Returns (shifted targets, {run: shift in rows}).
    """
    source = np.array(targets.detach().cpu().numpy() if torch.is_tensor(targets) else targets, dtype=np.float64)
    out, applied = source.copy(), {}
    for run, members, shift in shift_plan(rows, run_ids, k, parts, tr, min_tr):
        out[members] = source[np.roll(members, shift)]
        applied[run] = shift
    return out, applied


def shift_tr_offsets(rows, run_ids, tr, k, parts=SHIFT_PARTS, clamp_min_tr=None):
    """{run: smallest |tr[i] - tr[j]|} over the rows i of ``rows``, j the row whose target i receives under
    ``shifted_targets`` (wrapped rows included). ``tr`` holds each row's target TR, indexed like ``run_ids``."""
    tr = np.asarray(tr, dtype=np.int64)
    return {run: int(np.abs(tr[members] - tr[np.roll(members, shift)]).min())
            for run, members, shift in shift_plan(rows, run_ids, k, parts, tr, clamp_min_tr)}


def null_shift_offsets(rows, run_ids, tr, shifts=SHIFTS, min_tr=MIN_SHIFT_TR, clamp=False):
    """[``shift_tr_offsets`` for k in ``shifts``]; raises ValueError naming every (k, run, TR) below ``min_tr``
    (PROTOCOL_V4 §2: every shift is at least 40 TR). ``min_tr=None`` only reports. The shift of a run depends only on
    its rows in ``rows``, so a preflight over a budget's full runs gives the offsets of every fold that keeps those
    runs whole."""
    if tr is None or len(np.asarray(tr)) != len(np.asarray(run_ids)):
        raise ValueError('tr (target TR per row) is required and must be indexed like run_ids')
    offsets = [shift_tr_offsets(rows, run_ids, tr, k, clamp_min_tr=min_tr if clamp else None) for k in shifts]
    if min_tr is not None:
        short = [(k, run, o) for k, off in zip(shifts, offsets) for run, o in off.items() if o < min_tr]
        if short:
            raise ValueError(f'null shifts below {min_tr} TR (k, run, TR): {short}')
    return offsets


def null_sigma2(targets, design, rows, groups, folds, alphas, device, run_ids, tr, shifts=SHIFTS, info=None,
                min_tr=MIN_SHIFT_TR, clamp=True):
    """Shifted-null noise variance of Fisher-z encoding scores for one delay (§5 EB step 2).

    For each k in ``shifts``, ``encoding_scores`` is recomputed with the same ``design`` [N, V], ``rows``,
    ``groups``, ``folds`` and alpha grid (alpha re-selected per shift), with the targets of ``rows`` shifted by
    ``shifted_targets``. Returns the mean over k of ``mad_sigma2``. ``tr`` (target TR per row, indexed like
    ``run_ids``) is required: ``null_shift_offsets`` checks the realized shifts before any fit and raises on any run
    below ``min_tr`` TR; ``min_tr=None`` only records them. When ``info`` is a dict it receives the per-shift
    sigma^2, encoding alpha, row shifts by run, smallest TR offset by run and the overall smallest offset.
    """
    clamp = bool(clamp and min_tr is not None)  # PROTOCOL_V4 §4: shifts clamped to >= 40 TR in short runs.
    offsets = null_shift_offsets(rows, run_ids, tr, shifts, min_tr, clamp=clamp)
    per, alpha, applied = [], [], []
    for k in shifts:
        y, shift = shifted_targets(targets, rows, run_ids, k, tr=tr if clamp else None, min_tr=min_tr if clamp else None)
        r0, a0, _ = encoding_scores(y, design, rows, groups, folds, alphas, device)
        per.append(mad_sigma2(r0))
        alpha.append(a0)
        applied.append(shift)
    if info is not None:
        info.update(shifts=list(shifts), sigma2_by_shift=per, alpha_by_shift=alpha, shift_rows=applied,
                    min_tr_offset=offsets, min_tr_offset_overall=min((o for off in offsets for o in off.values()),
                                                                     default=None), min_tr_required=min_tr)
    return float(np.mean(per))


def eb_posterior(z, p, sigma2):
    """Closed-form EB posterior m (length C) and its hyperparameters.

    ``info`` holds a, b (slope after the b >= 0 rule), b_ols (unconstrained slope; 0 for a constant prior),
    slope_clipped, tau2, tau2_floored, sigma2, kappa, V, var_z, var_p and n_columns.
    """
    z, p = np.asarray(z, dtype=np.float64), np.asarray(p, dtype=np.float64)
    if z.ndim != 1 or z.shape != p.shape:
        raise ValueError('z and p must be 1-d arrays of equal length')
    sigma2 = float(sigma2)
    if not sigma2 >= 0:
        raise ValueError('sigma2 must be non-negative')
    var_p = float(np.var(p))
    b_ols = float(np.mean((p - p.mean()) * (z - z.mean())) / var_p) if var_p > 0 else 0.0
    b = max(b_ols, 0.0)
    a = float(z.mean() - b * p.mean())  # b = 0 gives a = mean(z).
    prior = a + b * p
    v = float(np.var(z - prior))
    tau2 = max(v - sigma2, TAU2_FLOOR * v)
    kappa = tau2 / (tau2 + sigma2) if tau2 + sigma2 > 0 else 1.0  # V = sigma2 = 0: z = a + b p, any kappa.
    m = kappa * z + (1.0 - kappa) * prior
    info = dict(a=a, b=b, b_ols=b_ols, slope_clipped=b_ols < 0, tau2=tau2, tau2_floored=v - sigma2 < TAU2_FLOOR * v,
                sigma2=sigma2, kappa=kappa, V=v, var_z=float(np.var(z)), var_p=var_p, n_columns=int(len(z)))
    return m, info


def mixcv_posterior(z, p, a, b, kappa):
    """MIX-CV posterior with a fixed kappa (chosen by inner CV from ``MIXCV_KAPPAS``): kappa z + (1 - kappa)(a + b p).

    (a, b) are the calibration of ``eb_posterior`` on the same fold. kappa = 1 returns z; kappa = 0, a = 0, b = 1
    returns p.
    """
    kappa = float(kappa)
    if not 0.0 <= kappa <= 1.0:
        raise ValueError('kappa must lie in [0, 1]')
    z, p = np.asarray(z, dtype=np.float64), np.asarray(p, dtype=np.float64)
    return kappa * z + (1.0 - kappa) * (float(a) + float(b) * p)


def eb_weights(m, gamma):
    """Unnormalized EB weights max(tanh(m), 0)^gamma; the caller rescales."""
    return np.maximum(np.tanh(np.asarray(m, dtype=np.float64)), 0.0) ** float(gamma)


def eb_column_weights(m_by_delay, gamma):
    """EB weights through the locked §12 W1 path: ``column_weights`` of tanh(m) under ('soft', gamma).

    Pooled over delays: mean(w^2) = 1, and the uniform fallback when fewer than 1% of columns have m > 0.
    Returns (weights by delay, info), as ``column_weights``.
    """
    return column_weights({d: np.tanh(np.asarray(m, dtype=np.float64)) for d, m in m_by_delay.items()},
                          ('soft', float(gamma)))


def reliability(info, n_pairs=None):
    """§8.3 reliability-model quantities from an ``eb_posterior`` info record (assumes independent errors).

    lambda = (Var z - sigma^2) / sigma^2 (own-map SNR), R_P = b^2 Var(p) / (Var z - sigma^2), R_own = lambda / (1 +
    lambda), 1 / (1 - R_EB) = 1 / (1 - R_P) + lambda, crossover lambda* = R_P / (1 - R_P), and with ``n_pairs``
    labeled pairs the label-equivalent value of the prior dn* = n_pairs * lambda* / lambda (lambda proportional to
    n). Undefined quantities are None.
    """
    signal, sigma2 = info['var_z'] - info['sigma2'], info['sigma2']
    out = dict(lambda_own=None, R_P=None, R_own=None, R_EB=None, crossover_lambda=None, delta_n=None)
    if signal <= 0 or sigma2 <= 0:
        return out
    lam = signal / sigma2
    r_p = info['b'] ** 2 * info['var_p'] / signal
    out.update(lambda_own=lam, R_P=r_p, R_own=lam / (1 + lam))
    if r_p < 1:
        cross = r_p / (1 - r_p)
        out.update(R_EB=1 - 1 / (1 / (1 - r_p) + lam), crossover_lambda=cross,
                   delta_n=None if n_pairs is None else float(n_pairs) * cross / lam)
    return out
