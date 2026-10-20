"""Paired embedding scores for semantic fMRI decoding.

The scores in this module compare a predicted embedding matrix with a target
embedding matrix. They are cosine similarity, column-wise Pearson correlation,
and chunked cosine retrieval (top-1, top-5, mean reciprocal rank, rank
percentile) with temporally neighboring negatives removed.

Generated-word scores used by Tang et al. (WER, BLEU, METEOR, BERTScore) are a
separate family. This module does not compute them.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]

# Cosines within this absolute tolerance count as ties.
_TIE_ATOL = 1e-8


def evaluate_embeddings(
    prediction: ArrayLike,
    target: ArrayLike,
    run_ids: Optional[ArrayLike] = None,
    times: Optional[ArrayLike] = None,
    exclusion_tr: float = 10,
    retrieval: bool = True,
    chunk_size: int = 256,
    topk: Sequence[int] = (1, 5),
    uncertainty: Optional[str] = None,
    block_length: int = 10,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> Dict[str, Any]:
    """Score paired embeddings with cosine, feature correlation, and retrieval.

    Parameters
    ----------
    prediction, target :
        Arrays shaped (n_samples, n_features). Row ``i`` of ``prediction`` is
        scored against row ``i`` of ``target``.
    run_ids :
        Length-``n_samples`` labels. Temporal neighbors are defined only inside
        a run. ``None`` treats every row as one run, in row order.
    times :
        Length-``n_samples`` clock in TR units. A negative is excluded when it
        shares ``run_ids`` and ``abs(times[i] - times[j]) <= exclusion_tr``.
        ``None`` uses the row order inside each run as the TR index (0, 1, ...).
    exclusion_tr :
        Radius, in the units of ``times``, of negatives left out of retrieval.
        The paired positive is always kept. Default 10 TRs (20 s at TR = 2 s),
        outside one Tang et al. evaluation window.
    retrieval :
        When False, retrieval scores are omitted.
    chunk_size :
        Query and candidate block size. Similarities are never materialized as
        a full ``n_samples x n_samples`` matrix.
    topk :
        Extra cutoffs reported under ``retrieval["at_k"]``. Top-1 and top-5
        are always reported when retrieval is on.
    uncertainty :
        ``None``, ``"block"`` (resample contiguous within-run blocks), or
        ``"run"`` (standard error across run-level scores).
    block_length, n_boot, seed, ci :
        Block-bootstrap settings. Blocks follow time order inside each run.
        The last block in a run may be shorter than ``block_length``.

    Returns
    -------
    dict
        JSON-serializable. Non-finite summaries are ``None``.

        ``mean_cosine`` is the mean of row-wise cosines.
        ``feature_correlation`` is the mean, across features, of the
        column-wise Pearson correlation used by Huth ``npp.mcorr`` (population
        standard deviation). Zero-variance columns are omitted.
        ``retrieval`` is chunked cosine embedding retrieval, not a
        generated-text score. Rank percentile is 1 when the positive is the
        unique best candidate and 0 when it is the unique worst. Queries whose
        candidate pool has size 1 are omitted. Top-k and MRR are expectations
        under a uniformly random order within cosine ties, as in the transfer
        scorer; rank percentile uses the equivalent tie midrank.

        ``by_run`` repeats the scores inside each run (retrieval candidates
        stay inside that run). ``macro_mean`` is the unweighted mean of those
        run scores. The top-level scores pool every row, and the top-level
        retrieval pool is every run.
    """
    pred, tgt, runs, clock = _align_inputs(prediction, target, run_ids, times)
    _check_retrieval_args(exclusion_tr, chunk_size, topk, uncertainty, block_length, n_boot, ci)
    ks = _topk_tuple(topk)

    cosine = _row_cosine(pred, tgt)
    feature_r, n_feature = _feature_correlation(pred, tgt)
    result: Dict[str, Any] = {
        "metric_family": "paired_embedding_retrieval",
        "description": (
            "Paired embedding cosine, column-wise Pearson correlation, and "
            "chunked cosine retrieval with temporally neighboring negatives "
            "removed. Generated-word scores (WER, BLEU, METEOR, BERTScore) "
            "are a separate family and are not computed."
        ),
        "n_samples": int(pred.shape[0]),
        "n_features": int(pred.shape[1]),
        "exclusion_tr": float(exclusion_tr),
        "mean_cosine": _mean_finite(cosine),
        "n_cosine": int(np.isfinite(cosine).sum()),
        "feature_correlation": _finite_or_none(feature_r),
        "n_feature_correlation": int(n_feature),
        "retrieval": None,
        "by_run": {},
        "macro_mean": {},
        "uncertainty": None,
    }

    query_stats = None
    if retrieval:
        query_stats = _retrieval_query_stats(
            pred, tgt, runs, clock, exclusion_tr, chunk_size
        )
        result["retrieval"] = _summarize_retrieval(query_stats, ks)

    by_run: Dict[str, Any] = {}
    for run in _unique_in_order(runs):
        mask = runs == run
        run_pred = pred[mask]
        run_tgt = tgt[mask]
        run_clock = clock[mask]
        run_cosine = _row_cosine(run_pred, run_tgt)
        run_feature, run_n_feature = _feature_correlation(run_pred, run_tgt)
        entry: Dict[str, Any] = {
            "n_samples": int(mask.sum()),
            "mean_cosine": _mean_finite(run_cosine),
            "n_cosine": int(np.isfinite(run_cosine).sum()),
            "feature_correlation": _finite_or_none(run_feature),
            "n_feature_correlation": int(run_n_feature),
            "retrieval": None,
        }
        if retrieval:
            run_ids_local = np.zeros(run_pred.shape[0], dtype=np.int64)
            run_stats = _retrieval_query_stats(
                run_pred,
                run_tgt,
                run_ids_local,
                run_clock,
                exclusion_tr,
                chunk_size,
            )
            entry["retrieval"] = _summarize_retrieval(run_stats, ks)
        by_run[str(run)] = entry
    result["by_run"] = by_run
    result["macro_mean"] = _macro_mean(by_run, ks if retrieval else ())

    if uncertainty is not None:
        result["uncertainty"] = _uncertainty(
            uncertainty,
            pred,
            tgt,
            runs,
            clock,
            cosine,
            query_stats,
            by_run,
            ks,
            exclusion_tr,
            block_length,
            n_boot,
            seed,
            ci,
            retrieval,
        )
    return result


def common_semantic_prediction(
    target: ArrayLike,
    vector: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Broadcast one semantic vector to every time point.

    ``vector=None`` uses the column mean of ``target``, skipping non-finite
    rows. This is the constant-embedding baseline: it keeps the shared
    direction of the embedding space and removes all time-varying meaning.
    """
    tgt = np.asarray(target, dtype=np.float64)
    if tgt.ndim != 2:
        raise ValueError("target must have shape (n_samples, n_features)")
    if vector is None:
        finite = np.isfinite(tgt).all(axis=1)
        if not np.any(finite):
            raise ValueError("target has no finite rows to average")
        vector_arr = tgt[finite].mean(axis=0)
    else:
        vector_arr = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vector_arr.shape != (tgt.shape[1],):
            raise ValueError("vector length must equal n_features")
    return np.broadcast_to(vector_arr, tgt.shape).copy()


def temporal_nearest_prediction(
    target: ArrayLike,
    labeled_mask: ArrayLike,
    run_ids: Optional[ArrayLike] = None,
    times: Optional[ArrayLike] = None,
    method: str = "nearest",
) -> np.ndarray:
    """Build the temporal-label SSL control prediction.

    Each row is replaced by an embedding taken from labeled rows in the same
    run. Labeled rows are copied through unchanged.

    ``method="nearest"`` copies the labeled row with the smallest time
    distance. Ties keep the earlier row in time, then the earlier index.
    ``method="linear"`` blends the nearest labeled row on each side by time
    and copies the single available side when the row is outside the labeled
    span. A run with no labeled row is filled with NaN.
    """
    if method not in ("nearest", "linear"):
        raise ValueError("method must be 'nearest' or 'linear'")
    tgt = np.asarray(target, dtype=np.float64)
    if tgt.ndim != 2:
        raise ValueError("target must have shape (n_samples, n_features)")
    mask = np.asarray(labeled_mask, dtype=bool)
    if mask.shape != (tgt.shape[0],):
        raise ValueError("labeled_mask must have shape (n_samples,)")
    if run_ids is None:
        runs = np.zeros(tgt.shape[0], dtype=np.int64)
    else:
        runs = np.asarray(run_ids)
        if runs.shape != (tgt.shape[0],):
            raise ValueError("run_ids must have shape (n_samples,)")
    if times is None:
        clock = _order_index(runs)
    else:
        clock = np.asarray(times, dtype=np.float64)
        if clock.shape != (tgt.shape[0],):
            raise ValueError("times must have shape (n_samples,)")

    out = np.full(tgt.shape, np.nan, dtype=np.float64)
    for run in _unique_in_order(runs):
        idx = np.flatnonzero(runs == run)
        labeled = idx[mask[idx]]
        if labeled.size == 0:
            continue
        out[labeled] = tgt[labeled]
        unlabeled = idx[~mask[idx]]
        if unlabeled.size == 0:
            continue
        if method == "nearest":
            out[unlabeled] = _nearest_rows(tgt, clock, labeled, unlabeled)
        else:
            out[unlabeled] = _linear_rows(tgt, clock, labeled, unlabeled)
    return out


def _align_inputs(
    prediction: ArrayLike,
    target: ArrayLike,
    run_ids: Optional[ArrayLike],
    times: Optional[ArrayLike],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if pred.ndim != 2 or tgt.ndim != 2:
        raise ValueError("prediction and target must have shape (n_samples, n_features)")
    if pred.shape != tgt.shape:
        raise ValueError(
            "prediction and target must share a shape, got %s and %s" % (pred.shape, tgt.shape)
        )
    if pred.shape[0] == 0:
        raise ValueError("prediction and target must contain at least one sample")
    n = pred.shape[0]
    if run_ids is None:
        runs = np.zeros(n, dtype=np.int64)
    else:
        runs = np.asarray(run_ids)
        if runs.shape != (n,):
            raise ValueError("run_ids must have shape (n_samples,)")
    if times is None:
        clock = _order_index(runs)
    else:
        clock = np.asarray(times, dtype=np.float64)
        if clock.shape != (n,):
            raise ValueError("times must have shape (n_samples,)")
        if not np.isfinite(clock).all():
            raise ValueError("times must be finite")
    return pred, tgt, runs, clock


def _order_index(runs: np.ndarray) -> np.ndarray:
    clock = np.empty(runs.shape[0], dtype=np.float64)
    for run in _unique_in_order(runs):
        idx = np.flatnonzero(runs == run)
        clock[idx] = np.arange(idx.size, dtype=np.float64)
    return clock


def _unique_in_order(runs: np.ndarray) -> list:
    seen = set()
    ordered = []
    for value in runs.tolist():
        key = value if not isinstance(value, float) else value
        # NumPy scalars and plain Python values should collapse to one key.
        marker = value.item() if hasattr(value, "item") else value
        if marker not in seen:
            seen.add(marker)
            ordered.append(marker)
    return ordered


def _check_retrieval_args(
    exclusion_tr: float,
    chunk_size: int,
    topk: Sequence[int],
    uncertainty: Optional[str],
    block_length: int,
    n_boot: int,
    ci: float,
) -> None:
    if not np.isfinite(exclusion_tr) or exclusion_tr < 0:
        raise ValueError("exclusion_tr must be a finite value >= 0")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be >= 1")
    if uncertainty not in (None, "block", "run"):
        raise ValueError("uncertainty must be None, 'block', or 'run'")
    if int(block_length) < 1:
        raise ValueError("block_length must be >= 1")
    if int(n_boot) < 1:
        raise ValueError("n_boot must be >= 1")
    if not (0 < float(ci) < 1):
        raise ValueError("ci must be between 0 and 1")
    for k in topk:
        if int(k) < 1:
            raise ValueError("topk entries must be >= 1")


def _topk_tuple(topk: Sequence[int]) -> Tuple[int, ...]:
    values = sorted({int(k) for k in topk} | {1, 5})
    return tuple(values)


def _row_cosine(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    pred_n, pred_ok = _normalize_rows(pred)
    tgt_n, tgt_ok = _normalize_rows(tgt)
    cosine = np.sum(pred_n * tgt_n, axis=1)
    cosine[~(pred_ok & tgt_ok)] = np.nan
    return cosine


def _normalize_rows(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(data).all(axis=1)
    norms = np.linalg.norm(data, axis=1)
    ok = finite & (norms > 0)
    out = np.zeros(data.shape, dtype=np.float64)
    out[ok] = data[ok] / norms[ok, None]
    return out, ok


def _feature_correlation(pred: np.ndarray, tgt: np.ndarray) -> Tuple[float, int]:
    """Mean column-wise Pearson correlation with population standard deviation.

    This matches ``ridge_utils.npp.mcorr`` wherever both columns have nonzero
    variance. Columns with a zero variance, or a non-finite pair, are left
    out of the mean rather than scaled as a zero vector.
    """
    finite = np.isfinite(pred).all(axis=1) & np.isfinite(tgt).all(axis=1)
    if int(finite.sum()) < 2:
        return math.nan, 0
    raw_a = pred[finite]
    raw_b = tgt[finite]
    a = raw_a - raw_a.mean(axis=0)
    b = raw_b - raw_b.mean(axis=0)
    sa = np.sqrt(np.mean(a * a, axis=0))
    sb = np.sqrt(np.mean(b * b, axis=0))
    # A repeated semantic vector is constant, but float64 centering can leave
    # an RMS near 1e-15. Treat that residual as zero variance.
    scale_a = np.maximum(np.sqrt(np.mean(raw_a * raw_a, axis=0)), 1e-12)
    scale_b = np.maximum(np.sqrt(np.mean(raw_b * raw_b, axis=0)), 1e-12)
    varying = (sa > 1e-8 * scale_a) & (sb > 1e-8 * scale_b)
    numer = np.mean(a * b, axis=0)
    corr = np.full(pred.shape[1], np.nan, dtype=np.float64)
    corr[varying] = numer[varying] / (sa[varying] * sb[varying])
    valid = varying
    n_valid = int(valid.sum())
    if n_valid == 0:
        return math.nan, 0
    return float(np.mean(corr[valid])), n_valid


def _retrieval_query_stats(
    pred: np.ndarray,
    tgt: np.ndarray,
    runs: np.ndarray,
    clock: np.ndarray,
    exclusion_tr: float,
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    """Per-query rank statistics without an ``n x n`` similarity matrix."""
    n = pred.shape[0]
    pred_n, pred_ok = _normalize_rows(pred)
    tgt_n, tgt_ok = _normalize_rows(tgt)
    positive = np.sum(pred_n * tgt_n, axis=1)
    positive[~(pred_ok & tgt_ok)] = np.nan

    n_better = np.zeros(n, dtype=np.float64)
    n_equal = np.zeros(n, dtype=np.float64)
    n_valid = np.zeros(n, dtype=np.float64)
    chunk = int(chunk_size)

    for q0 in range(0, n, chunk):
        q1 = min(n, q0 + chunk)
        q_ok = pred_ok[q0:q1] & tgt_ok[q0:q1]
        if not np.any(q_ok):
            continue
        pos = positive[q0:q1]
        for c0 in range(0, n, chunk):
            c1 = min(n, c0 + chunk)
            sims = pred_n[q0:q1] @ tgt_n[c0:c1].T
            cand_ok = tgt_ok[c0:c1]
            q_index = np.arange(q0, q1)[:, None]
            c_index = np.arange(c0, c1)[None, :]
            same_run = runs[q0:q1, None] == runs[c0:c1][None, :]
            # Compare in TR units. The paired positive is never a negative.
            neighbor = same_run & (np.abs(clock[q0:q1, None] - clock[c0:c1][None, :]) <= exclusion_tr)
            is_positive = q_index == c_index
            keep = cand_ok[None, :] & (~neighbor | is_positive) & q_ok[:, None]
            kept_sims = np.where(keep, sims, np.nan)
            n_valid[q0:q1] += np.sum(keep, axis=1)
            compared = keep & ~is_positive
            n_better[q0:q1] += np.sum(compared & (kept_sims > pos[:, None] + _TIE_ATOL), axis=1)
            n_equal[q0:q1] += np.sum(
                compared & (np.abs(kept_sims - pos[:, None]) <= _TIE_ATOL),
                axis=1,
            )

    usable = pred_ok & tgt_ok & (n_valid >= 2)
    rank = np.full(n, np.nan, dtype=np.float64)
    percentile = np.full(n, np.nan, dtype=np.float64)
    reciprocal = np.full(n, np.nan, dtype=np.float64)
    # Average rank of the tied group that contains the positive.
    # n_equal counts other candidates tied with the positive, not the positive.
    rank[usable] = n_better[usable] + 1.0 + 0.5 * n_equal[usable]
    percentile[usable] = (n_valid[usable] - rank[usable]) / (n_valid[usable] - 1.0)
    # Expected reciprocal rank under a uniformly random order within ties.
    reciprocal[usable] = 1.0 / (n_better[usable] + 1.0)
    for i in np.flatnonzero(usable & (n_equal > 0)):
        first = n_better[i] + 1.0
        reciprocal[i] = np.mean(1.0 / np.arange(first, first + n_equal[i] + 1.0))
    return {
        "rank": rank,
        "percentile": percentile,
        "reciprocal_rank": reciprocal,
        "n_better": n_better,
        "n_equal": n_equal,
        "n_candidates": np.where(usable, n_valid, np.nan),
        "usable": usable,
    }


def _expected_at_k(stats: Dict[str, np.ndarray], rows: np.ndarray, k: int) -> np.ndarray:
    """P(rank <= k) when the positive's tied group is ordered uniformly at random."""
    better, equal = stats["n_better"][rows], stats["n_equal"][rows]
    return np.clip((k - better) / (equal + 1.0), 0.0, 1.0)


def _summarize_retrieval(stats: Dict[str, np.ndarray], ks: Tuple[int, ...]) -> Dict[str, Any]:
    usable = stats["usable"]
    n_queries = int(np.sum(usable))
    if n_queries == 0:
        at_k = {str(k): None for k in ks}
        return {
            "name": "chunked_cosine_embedding_retrieval",
            "top1": None,
            "top5": None,
            "mrr": None,
            "rank_percentile": None,
            "n_queries": 0,
            "mean_n_candidates": None,
            "at_k": at_k,
        }
    rows = np.flatnonzero(usable)
    at_k = {str(k): float(np.mean(_expected_at_k(stats, rows, k))) for k in ks}
    return {
        "name": "chunked_cosine_embedding_retrieval",
        "top1": at_k["1"],
        "top5": at_k["5"],
        "mrr": float(np.mean(stats["reciprocal_rank"][rows])),
        "rank_percentile": float(np.mean(stats["percentile"][usable])),
        "n_queries": n_queries,
        "mean_n_candidates": float(np.mean(stats["n_candidates"][usable])),
        "at_k": at_k,
    }


def _macro_mean(by_run: Mapping[str, Mapping[str, Any]], ks: Tuple[int, ...]) -> Dict[str, Any]:
    def collect(key: str) -> Any:
        values = [entry[key] for entry in by_run.values() if entry[key] is not None]
        return _mean_or_none(values)

    retrieval: Dict[str, Any] = {}
    if ks:
        for name in ("top1", "top5", "mrr", "rank_percentile"):
            values = []
            for entry in by_run.values():
                block = entry["retrieval"]
                if block is not None and block[name] is not None:
                    values.append(block[name])
            retrieval[name] = _mean_or_none(values)
        at_k = {}
        for k in ks:
            values = []
            for entry in by_run.values():
                block = entry["retrieval"]
                if block is not None and block["at_k"][str(k)] is not None:
                    values.append(block["at_k"][str(k)])
            at_k[str(k)] = _mean_or_none(values)
        retrieval["at_k"] = at_k
    return {
        "mean_cosine": collect("mean_cosine"),
        "feature_correlation": collect("feature_correlation"),
        "retrieval": retrieval or None,
        "n_runs": len(by_run),
    }


def _uncertainty(
    method: str,
    pred: np.ndarray,
    tgt: np.ndarray,
    runs: np.ndarray,
    clock: np.ndarray,
    cosine: np.ndarray,
    query_stats: Optional[Dict[str, np.ndarray]],
    by_run: Mapping[str, Mapping[str, Any]],
    ks: Tuple[int, ...],
    exclusion_tr: float,
    block_length: int,
    n_boot: int,
    seed: int,
    ci: float,
    retrieval: bool,
) -> Dict[str, Any]:
    if method == "run":
        return _run_uncertainty(by_run, ks, retrieval)
    return _block_uncertainty(
        pred,
        tgt,
        runs,
        clock,
        cosine,
        query_stats,
        ks,
        block_length,
        n_boot,
        seed,
        ci,
        retrieval,
    )


def _run_uncertainty(
    by_run: Mapping[str, Mapping[str, Any]],
    ks: Tuple[int, ...],
    retrieval: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "method": "run",
        "n_runs": len(by_run),
        "mean_cosine": _dispersion([entry["mean_cosine"] for entry in by_run.values()]),
        "feature_correlation": _dispersion(
            [entry["feature_correlation"] for entry in by_run.values()]
        ),
    }
    if retrieval:
        out["retrieval"] = {
            "name": "chunked_cosine_embedding_retrieval",
            "top1": _dispersion(_retrieval_values(by_run, "top1")),
            "top5": _dispersion(_retrieval_values(by_run, "top5")),
            "mrr": _dispersion(_retrieval_values(by_run, "mrr")),
            "rank_percentile": _dispersion(_retrieval_values(by_run, "rank_percentile")),
        }
        for k in ks:
            out["retrieval"]["at_%d" % k] = _dispersion(_retrieval_at_k(by_run, k))
    return out


def _retrieval_values(by_run: Mapping[str, Mapping[str, Any]], name: str) -> list:
    values = []
    for entry in by_run.values():
        block = entry["retrieval"]
        values.append(None if block is None else block[name])
    return values


def _retrieval_at_k(by_run: Mapping[str, Mapping[str, Any]], k: int) -> list:
    values = []
    for entry in by_run.values():
        block = entry["retrieval"]
        values.append(None if block is None else block["at_k"][str(k)])
    return values


def _dispersion(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    summary: Dict[str, Any] = {"n": len(finite), "mean": _mean_or_none(finite), "std": None, "sem": None}
    if len(finite) >= 2:
        std = float(np.std(finite, ddof=1))
        summary["std"] = std
        summary["sem"] = float(std / math.sqrt(len(finite)))
    return summary


def _block_uncertainty(
    pred: np.ndarray,
    tgt: np.ndarray,
    runs: np.ndarray,
    clock: np.ndarray,
    cosine: np.ndarray,
    query_stats: Optional[Dict[str, np.ndarray]],
    ks: Tuple[int, ...],
    block_length: int,
    n_boot: int,
    seed: int,
    ci: float,
    retrieval: bool,
) -> Dict[str, Any]:
    groups = _blocks_by_run(runs, clock, int(block_length))
    rng = np.random.default_rng(int(seed))
    cosine_stats = []
    feature_stats = []
    rank_stats = {k: [] for k in ks}
    mrr_stats = []
    percentile_stats = []
    for _ in range(int(n_boot)):
        chosen = _resample_blocks(groups, rng)
        if chosen.size < 2:
            continue
        cosine_stats.append(_mean_finite(cosine[chosen]))
        feature_value, _ = _feature_correlation(pred[chosen], tgt[chosen])
        feature_stats.append(_finite_or_none(feature_value))
        if retrieval and query_stats is not None:
            usable = query_stats["usable"][chosen]
            if int(np.sum(usable)) == 0:
                mrr_stats.append(None)
                percentile_stats.append(None)
                for k in ks:
                    rank_stats[k].append(None)
                continue
            rows = chosen[usable]
            mrr_stats.append(float(np.mean(query_stats["reciprocal_rank"][rows])))
            percentile_stats.append(float(np.mean(query_stats["percentile"][rows])))
            for k in ks:
                rank_stats[k].append(float(np.mean(_expected_at_k(query_stats, rows, k))))
    alpha = (1.0 - float(ci)) / 2.0
    out: Dict[str, Any] = {
        "method": "block",
        "block_length": int(block_length),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "ci": float(ci),
        "note": (
            "Blocks are resampled inside each run. Cosine and feature "
            "correlation are recomputed on the resampled pairs. Retrieval "
            "resamples precomputed query ranks and keeps the candidate pool "
            "fixed."
        ),
        "mean_cosine": _bootstrap_summary(cosine_stats, alpha),
        "feature_correlation": _bootstrap_summary(feature_stats, alpha),
    }
    if retrieval:
        out["retrieval"] = {
            "name": "chunked_cosine_embedding_retrieval",
            "top1": _bootstrap_summary(rank_stats[1], alpha),
            "top5": _bootstrap_summary(rank_stats[5], alpha),
            "mrr": _bootstrap_summary(mrr_stats, alpha),
            "rank_percentile": _bootstrap_summary(percentile_stats, alpha),
        }
    return out


def _blocks_by_run(runs: np.ndarray, clock: np.ndarray, block_length: int) -> list:
    groups = []
    for run in _unique_in_order(runs):
        idx = np.flatnonzero(runs == run)
        order = idx[np.argsort(clock[idx], kind="mergesort")]
        blocks = [order[start : start + block_length] for start in range(0, order.size, block_length)]
        groups.append(blocks)
    return groups


def _resample_blocks(groups: list, rng: np.random.Generator) -> np.ndarray:
    chosen = []
    for blocks in groups:
        if not blocks:
            continue
        draw = rng.integers(0, len(blocks), size=len(blocks))
        for block_index in draw:
            chosen.append(blocks[int(block_index)])
    if not chosen:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(chosen)


def _bootstrap_summary(values: Sequence[Optional[float]], alpha: float) -> Dict[str, Any]:
    finite = np.array([float(v) for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    summary: Dict[str, Any] = {
        "n": int(finite.size),
        "mean": None,
        "std": None,
        "low": None,
        "high": None,
    }
    if finite.size == 0:
        return summary
    summary["mean"] = float(np.mean(finite))
    if finite.size >= 2:
        summary["std"] = float(np.std(finite, ddof=1))
        summary["low"] = float(np.quantile(finite, alpha))
        summary["high"] = float(np.quantile(finite, 1.0 - alpha))
    return summary


def _nearest_rows(
    tgt: np.ndarray,
    clock: np.ndarray,
    labeled: np.ndarray,
    unlabeled: np.ndarray,
) -> np.ndarray:
    distance = np.abs(clock[unlabeled, None] - clock[labeled][None, :])
    # Earlier time wins a distance tie; labeled is then sorted by index.
    order = np.lexsort((labeled, clock[labeled]))
    labeled = labeled[order]
    distance = distance[:, order]
    choice = np.argmin(distance, axis=1)
    return tgt[labeled[choice]]


def _linear_rows(
    tgt: np.ndarray,
    clock: np.ndarray,
    labeled: np.ndarray,
    unlabeled: np.ndarray,
) -> np.ndarray:
    order = np.argsort(clock[labeled], kind="mergesort")
    labeled = labeled[order]
    lab_t = clock[labeled]
    out = np.empty((unlabeled.size, tgt.shape[1]), dtype=np.float64)
    for row, index in enumerate(unlabeled):
        t = clock[index]
        after = np.flatnonzero(lab_t >= t)
        before = np.flatnonzero(lab_t <= t)
        if before.size == 0:
            out[row] = tgt[labeled[after[0]]]
        elif after.size == 0:
            out[row] = tgt[labeled[before[-1]]]
        else:
            left = labeled[before[-1]]
            right = labeled[after[0]]
            span = clock[right] - clock[left]
            if span == 0 or left == right:
                out[row] = tgt[left]
            else:
                weight = (t - clock[left]) / span
                out[row] = (1.0 - weight) * tgt[left] + weight * tgt[right]
    return out


def _mean_finite(values: np.ndarray) -> Optional[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    if len(values) == 0:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _finite_or_none(value: float) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    return float(value)
