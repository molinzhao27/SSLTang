"""Grouped-CV selection of structured voxel kernels (PROTOCOL_V3.md §2-4).

Candidates combine coarse (C) and detail (R) base kernels across delays with
raw coefficients a_(C,d) = tau_d and a_(R,d) = eta * tau_d. Every arm chooses
its candidate and alpha by out-of-fold, session-grouped CV on labeled training
rows only; normalization is refit within each fold on the inner sessions.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from .multikernel_ridge import KernelRidgePath, combine, run_feature_r, session_macro_scores
from .voxel_kernels import DELAYS

ETAS = (0.0, 0.1, 1.0, 10.0)
TAUS = ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (1, 1, 0), (0, 1, 1), (0, 0, 1))
BASELINE = (1.0, (1, 1, 1))
FOLD_SALT = 'ssltang-v3-folds'
TIE = 1e-12


def candidates():
    """Declared order: baseline first, then tau in listed order, then eta."""
    rest = [(eta, tau) for tau in TAUS for eta in ETAS if (eta, tau) != BASELINE]
    return [BASELINE] + rest


ORDER = {c: i for i, c in enumerate(candidates())}
ARMS = {
    'V0': [BASELINE],
    'V2': sorted([(eta, (1, 1, 1)) for eta in ETAS], key=ORDER.get),
    'V3': sorted([(1.0, tau) for tau in TAUS], key=ORDER.get),
    'V4': candidates(),
}


def coefficients(eta, tau):
    coef = {('C', d): float(t) for d, t in zip(DELAYS, tau)}
    coef.update({('R', d): float(eta) * float(t) for d, t in zip(DELAYS, tau)})
    return coef


def label(candidate):
    eta, tau = candidate
    return f"eta={eta:g},tau={''.join(map(str, tau))}"


def fixed_folds(sessions, k=5, salt=FOLD_SALT):
    order = sorted(set(sessions), key=lambda s: hashlib.sha256(f'{salt}:{s}'.encode()).hexdigest())
    return [[s for i, s in enumerate(order) if i % k == f] for f in range(k)]


def budget_folds(row_sessions, labeled_rows, fixed):
    """Held-out session sets for one budget, or (None, reason) when not estimable."""
    labeled = {row_sessions[i] for i in labeled_rows}
    if len(labeled) < 2:
        return None, 'not estimable: fewer than two labeled sessions'
    if len(labeled) >= 5:
        folds = [tuple(f) for f in fixed if labeled & set(f)]
        rule = 'fixed five-fold session partition; folds without labeled held-out rows dropped'
    else:
        folds = [(s,) for s in sorted(labeled, key=lambda s: (len(s), s))]
        rule = 'leave-one-labeled-session-out'
    if len(folds) < 2:
        return None, 'not estimable: fewer than two folds with labeled held-out rows'
    return folds, rule


def _block(kernels, rows, cols):
    return {k: v.index_select(0, rows).index_select(1, cols) for k, v in kernels.items()}


def score_fold(kernels, fit, held, y, held_codes, n_runs, alphas, device, candidate_list=None):
    """Per-run mean r [alphas, runs] for each candidate; per-output r for the baseline."""
    fit_t, held_t = (torch.as_tensor(np.asarray(r), device=device) for r in (fit, held))
    ff, hf = _block(kernels, fit_t, fit_t), _block(kernels, held_t, fit_t)
    y_t = torch.as_tensor(y, dtype=torch.float64, device=device)
    y_fit, y_held = y_t.index_select(0, fit_t), y_t.index_select(0, held_t)
    codes = torch.as_tensor(held_codes, device=device)
    out = {}
    for candidate in candidate_list or candidates():
        coef = coefficients(*candidate)
        path = KernelRidgePath(combine(ff, coef), y_fit, device)
        r = run_feature_r(path.predict_path(path.project(combine(hf, coef)), alphas), y_held, codes, n_runs)
        out[candidate] = dict(per_run=torch.nanmean(r, dim=-1).cpu().numpy(), diagnostics=path.diagnostics)
        if candidate == BASELINE:
            out[candidate]['per_output'] = r.cpu().numpy()
    return out


def select(arm, cv_scores, alphas):
    """Highest CV score; ties to declared candidate order, then larger alpha."""
    best = None
    for candidate in ARMS[arm]:
        for k in range(len(alphas) - 1, -1, -1):
            value = cv_scores[candidate][k]
            if not np.isfinite(value):
                continue
            if best is None or value > best[0] + TIE:
                best = (value, candidate, k)
    if best is None:
        return None
    value, candidate, k = best
    return dict(candidate=candidate, label=label(candidate), alpha=float(alphas[k]), alpha_index=k,
                cv_score=float(value), alpha_at_grid_edge=k in (0, len(alphas) - 1))


def select_per_output(per_output_scores, fallback_alpha, alphas):
    """V1: per-output alpha [outputs]; ties to the larger alpha; NaN outputs fall back."""
    scores = np.asarray(per_output_scores, dtype=np.float64)  # [alphas, outputs]
    chosen = np.full(scores.shape[1], float(fallback_alpha))
    fallback = 0
    for j in range(scores.shape[1]):
        column = scores[:, j]
        finite = np.isfinite(column)
        if not finite.any():
            fallback += 1
            continue
        near = np.flatnonzero(finite & (column >= np.nanmax(column) - TIE))
        chosen[j] = alphas[near[-1]]  # Ties within TIE go to the larger alpha.
    edges = int(np.isin(chosen, [alphas[0], alphas[-1]]).sum())
    return chosen, dict(fallback_outputs=fallback, alpha_at_grid_edge_outputs=edges,
                        alpha_quantiles={q: float(np.quantile(chosen, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)})


def aggregate(fold_results, run_sessions, alphas):
    """Session-macro CV scores from out-of-fold per-run results.

    ``fold_results`` is a list of (held_run_ids, score_fold output). Returns
    {candidate: [alphas]} and the baseline per-output scores [alphas, outputs].
    """
    runs, per_candidate, per_output = [], {}, []
    for held_runs, result in fold_results:
        runs.extend(held_runs)
        for candidate, value in result.items():
            per_candidate.setdefault(candidate, []).append(value['per_run'])
            if 'per_output' in value:
                per_output.append(value['per_output'])
    sessions = [run_sessions[r] for r in runs]
    scores = {c: session_macro_scores(np.concatenate(v, axis=-1), sessions) for c, v in per_candidate.items()}
    output_scores = None
    if per_output:
        stacked = np.concatenate(per_output, axis=-2)  # [alphas, runs, outputs]
        output_scores = session_macro_scores(np.moveaxis(stacked, -2, -1), sessions)
    return scores, output_scores
