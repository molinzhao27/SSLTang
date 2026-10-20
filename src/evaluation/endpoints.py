"""v2 held-out endpoints: session-macro correlation, retrieval and temporal nulls.

The primary endpoint is the mean per-feature temporal Pearson correlation
within each held-out story, averaged over stories within session and then over
sessions. The archived run-macro summary is retained under its own name.
"""
from __future__ import annotations
import numpy as np
from .metrics import evaluate_embeddings, _feature_correlation


def _session_macro(by_run, sessions):
    groups = {}
    for run, value in by_run.items():
        if value is not None and np.isfinite(value):
            groups.setdefault(sessions[run], []).append(value)
    by_session = {str(s): float(np.mean(v)) for s, v in sorted(groups.items())}
    return (float(np.mean(list(by_session.values()))) if by_session else None), by_session


def score(prediction, target, run_ids, target_trs, sessions, train_mean, train_std=None,
          exclusions=(10, 32)):
    """Score one prediction set. ``sessions`` maps run_id to acquisition session."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mean = np.asarray(train_mean, dtype=np.float64)
    run_ids, times = np.asarray(run_ids), np.asarray(target_trs, dtype=np.float64)
    raw = evaluate_embeddings(prediction, target, run_ids, times, exclusion_tr=exclusions[0])
    by_run = {run: entry['feature_correlation'] for run, entry in raw['by_run'].items()}
    primary, by_session = _session_macro(by_run, sessions)
    report = {
        'session_macro_feature_correlation': primary,
        'run_macro_feature_correlation': raw['macro_mean']['feature_correlation'],
        'pooled_feature_correlation': raw['feature_correlation'],
        'by_session_feature_correlation': by_session,
        'by_run_feature_correlation': by_run,
        'raw_mean_cosine': raw['mean_cosine'],
        'n_samples': raw['n_samples'], 'n_runs': len(by_run),
        'centered_retrieval': {},
    }
    for exclusion in exclusions:
        centered = evaluate_embeddings(prediction - mean, target - mean, run_ids, times,
                                       exclusion_tr=exclusion)
        percentile = {run: (entry['retrieval'] or {}).get('rank_percentile')
                      for run, entry in centered['by_run'].items()}
        pools = [entry['retrieval']['mean_n_candidates'] for entry in centered['by_run'].values()
                 if entry['retrieval'] and entry['retrieval']['mean_n_candidates'] is not None]
        report['centered_retrieval'][str(exclusion)] = {
            'session_macro_rank_percentile': _session_macro(percentile, sessions)[0],
            'run_macro': centered['macro_mean']['retrieval'],
            'run_macro_centered_cosine': centered['macro_mean']['mean_cosine'],
            # A constant centered prediction is a zero vector: no usable queries.
            'mean_candidates_per_query': float(np.mean(pools)) if pools else None,
        }
    if train_std is not None:
        std = np.asarray(train_std, dtype=np.float64)
        report['normalized_mse'] = float(np.mean(((prediction - target) / std) ** 2))
        report['constant_normalized_mse'] = float(np.mean(((mean - target) / std) ** 2))
    return report


def shift_null(prediction, target, run_ids, sessions, stride=4, min_shift_tr=40, draws=2000, seed=0):
    """Within-run circular-shift null for the session-macro correlation.

    Each run's predictions are rotated against its own targets by at least
    ``min_shift_tr`` TRs in both directions (wrapping inside the run), which
    keeps each sequence's autocorrelation and removes stimulus alignment.
    Runs too short for any allowed shift are omitted with a recorded reason.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    run_ids = np.asarray(run_ids)
    minimum = int(np.ceil(min_shift_tr / stride))
    observed, shifted, omitted = {}, {}, {}
    for run in dict.fromkeys(run_ids.tolist()):
        rows = np.flatnonzero(run_ids == run)
        n = len(rows)
        allowed = [s for s in range(1, n) if min(s, n - s) >= minimum]
        observed[run] = _feature_correlation(prediction[rows], target[rows])[0]
        if not allowed:
            omitted[run] = f'{n} samples cannot be shifted by >= {min_shift_tr} TR both ways'
            continue
        shifted[run] = np.array([_feature_correlation(np.roll(prediction[rows], s, axis=0),
                                                      target[rows])[0] for s in allowed])
    usable = list(shifted)
    report = {'method': 'within-run circular shift, uniform over allowed shifts per run',
              'min_shift_tr': min_shift_tr, 'stride_tr': stride, 'draws': draws, 'seed': seed,
              'runs_used': len(usable), 'omitted_runs': omitted}
    if not usable:
        return report
    obs, _ = _session_macro({run: observed[run] for run in usable}, sessions)
    rng = np.random.default_rng(seed)
    null = np.empty(draws)
    for d in range(draws):
        sample = {run: shifted[run][rng.integers(len(shifted[run]))] for run in usable}
        null[d] = _session_macro(sample, sessions)[0]
    report.update(observed_session_macro=obs, null_mean=float(null.mean()),
                  null_q95=float(np.quantile(null, .95)), null_sd=float(null.std(ddof=1)),
                  p_value=float((1 + np.sum(null >= obs)) / (1 + draws)),
                  mean_shifted_by_run={run: float(v.mean()) for run, v in shifted.items()})
    return report
