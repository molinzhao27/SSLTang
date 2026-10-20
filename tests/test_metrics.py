"""Tests for paired embedding retrieval scores."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import (  # noqa: E402
    common_semantic_prediction,
    evaluate_embeddings,
    temporal_nearest_prediction,
)


def _naive_retrieval(prediction, target, run_ids, times, exclusion_tr):
    """Independent full-matrix retrieval used only at test scale."""
    pred = np.asarray(prediction, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    pred_n = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    tgt_n = tgt / np.linalg.norm(tgt, axis=1, keepdims=True)
    sims = pred_n @ tgt_n.T
    n = pred.shape[0]
    ranks = []
    percentiles = []
    for i in range(n):
        keep = []
        for j in range(n):
            neighbor = run_ids[i] == run_ids[j] and abs(times[i] - times[j]) <= exclusion_tr
            if j != i and neighbor:
                continue
            keep.append(j)
        if len(keep) < 2:
            continue
        pos = sims[i, i]
        others = [j for j in keep if j != i]
        n_better = sum(sims[i, j] > pos + 1e-8 for j in others)
        n_equal = sum(abs(sims[i, j] - pos) <= 1e-8 for j in others)
        rank = n_better + 1.0 + 0.5 * n_equal
        # Every order of the tied group is equally likely.
        ranks.append(np.arange(n_better + 1, n_better + n_equal + 2, dtype=float))
        percentiles.append((len(keep) - rank) / (len(keep) - 1.0))
    return {
        "top1": float(np.mean([np.mean(r <= 1) for r in ranks])),
        "top5": float(np.mean([np.mean(r <= 5) for r in ranks])),
        "mrr": float(np.mean([np.mean(1.0 / r) for r in ranks])),
        "rank_percentile": float(np.mean(percentiles)),
        "n_queries": len(ranks),
    }


class EvaluateEmbeddingsTest(unittest.TestCase):
    def test_result_is_strict_json(self):
        rng = np.random.default_rng(0)
        target = rng.normal(size=(12, 4))
        prediction = target + 0.05 * rng.normal(size=target.shape)
        result = evaluate_embeddings(
            prediction,
            target,
            run_ids=np.array(["a"] * 6 + ["b"] * 6),
            uncertainty="block",
            n_boot=8,
            block_length=3,
            chunk_size=5,
            seed=1,
        )
        encoded = json.dumps(result, allow_nan=False)
        self.assertIn("paired_embedding_retrieval", encoded)
        self.assertEqual(result["retrieval"]["name"], "chunked_cosine_embedding_retrieval")
        self.assertIn("BERTScore", result["description"])

    def test_identical_distinct_rows_are_perfect(self):
        rng = np.random.default_rng(1)
        target = rng.normal(size=(9, 5))
        result = evaluate_embeddings(target, target, exclusion_tr=1, chunk_size=4)
        self.assertAlmostEqual(result["mean_cosine"], 1.0)
        self.assertAlmostEqual(result["feature_correlation"], 1.0)
        self.assertAlmostEqual(result["retrieval"]["top1"], 1.0)
        self.assertAlmostEqual(result["retrieval"]["top5"], 1.0)
        self.assertAlmostEqual(result["retrieval"]["mrr"], 1.0)
        self.assertAlmostEqual(result["retrieval"]["rank_percentile"], 1.0)

    def test_chunk_size_matches_naive_retrieval(self):
        rng = np.random.default_rng(2)
        target = rng.normal(size=(11, 3))
        prediction = rng.normal(size=target.shape)
        runs = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        times = np.array([0, 1, 2, 3, 0, 1, 2, 10, 0, 1, 2], dtype=float)
        for exclusion in (0, 1, 2):
            naive = _naive_retrieval(prediction, target, runs, times, exclusion)
            for chunk_size in (1, 3, 11):
                result = evaluate_embeddings(
                    prediction,
                    target,
                    run_ids=runs,
                    times=times,
                    exclusion_tr=exclusion,
                    chunk_size=chunk_size,
                    retrieval=True,
                )
                got = result["retrieval"]
                self.assertEqual(got["n_queries"], naive["n_queries"])
                self.assertAlmostEqual(got["top1"], naive["top1"])
                self.assertAlmostEqual(got["top5"], naive["top5"])
                self.assertAlmostEqual(got["mrr"], naive["mrr"])
                self.assertAlmostEqual(got["rank_percentile"], naive["rank_percentile"])

    def test_temporal_neighbors_are_excluded_and_other_runs_are_kept(self):
        target = np.array(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        )
        prediction = np.array(
            [
                [0.96, 0.02],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )
        runs = np.array([0, 0, 0, 1])
        times = np.array([0, 1, 2, 1], dtype=float)
        kept = evaluate_embeddings(
            prediction, target, run_ids=runs, times=times, exclusion_tr=1, chunk_size=2
        )
        open_pool = evaluate_embeddings(
            prediction, target, run_ids=runs, times=times, exclusion_tr=0, chunk_size=2
        )
        naive_kept = _naive_retrieval(prediction, target, runs, times, 1)
        naive_open = _naive_retrieval(prediction, target, runs, times, 0)
        self.assertAlmostEqual(kept["retrieval"]["top1"], naive_kept["top1"])
        self.assertAlmostEqual(open_pool["retrieval"]["top1"], naive_open["top1"])
        self.assertGreater(kept["retrieval"]["top1"], open_pool["retrieval"]["top1"])
        # A sample in another run stays in the candidate pool even when its
        # time index matches a within-run neighbor.
        self.assertGreater(open_pool["retrieval"]["mean_n_candidates"], 2)

    def test_feature_correlation_skips_constant_columns(self):
        target = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
        prediction = np.array([[1.0, 9.0], [2.0, 9.0], [3.0, 9.0], [4.0, 9.0]])
        result = evaluate_embeddings(prediction, target, retrieval=False)
        self.assertAlmostEqual(result["feature_correlation"], 1.0)
        self.assertEqual(result["n_feature_correlation"], 1)
        self.assertIsNone(result["retrieval"])

    def test_zero_vector_is_omitted_from_cosine(self):
        target = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        prediction = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
        result = evaluate_embeddings(prediction, target, exclusion_tr=0)
        self.assertAlmostEqual(result["mean_cosine"], 1.0)
        self.assertEqual(result["n_cosine"], 2)

    def test_run_macro_mean_is_unweighted(self):
        target = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        prediction = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]])
        runs = np.array(["story"] * 2 + ["other"] * 3)
        result = evaluate_embeddings(prediction, target, run_ids=runs, exclusion_tr=0)
        self.assertAlmostEqual(result["by_run"]["story"]["mean_cosine"], 1.0)
        self.assertAlmostEqual(result["by_run"]["other"]["mean_cosine"], -1.0)
        self.assertAlmostEqual(result["macro_mean"]["mean_cosine"], 0.0)
        self.assertAlmostEqual(result["mean_cosine"], -0.2)
        self.assertEqual(result["by_run"]["story"]["retrieval"]["n_queries"], 2)

    def test_common_semantic_vector_has_no_feature_correlation(self):
        rng = np.random.default_rng(3)
        target = rng.normal(size=(16, 4)) + np.array([3.0, -2.0, 0.5, 1.0])
        prediction = common_semantic_prediction(target)
        self.assertEqual(prediction.shape, target.shape)
        self.assertTrue(np.allclose(prediction, prediction[:1]))
        result = evaluate_embeddings(prediction, target, retrieval=False)
        self.assertIsNotNone(result["mean_cosine"])
        self.assertGreater(result["mean_cosine"], 0.5)
        self.assertIsNone(result["feature_correlation"])

    def test_temporal_nearest_and_linear_controls(self):
        target = np.array(
            [
                [0.0, 0.0],
                [9.0, 9.0],
                [2.0, 0.0],
                [4.0, 4.0],
                [8.0, 0.0],
            ],
            dtype=float,
        )
        labeled = np.array([True, False, True, True, False])
        runs = np.array([0, 0, 0, 1, 1])
        times = np.array([0.0, 1.0, 2.0, 0.0, 5.0])
        nearest = temporal_nearest_prediction(target, labeled, runs, times, method="nearest")
        linear = temporal_nearest_prediction(target, labeled, runs, times, method="linear")
        self.assertTrue(np.allclose(nearest[1], target[0]))
        self.assertTrue(np.allclose(linear[1], np.array([1.0, 0.0])))
        self.assertTrue(np.allclose(nearest[4], target[3]))
        self.assertTrue(np.allclose(linear[4], target[3]))
        # A run with no labels stays missing.
        unlabeled_run = temporal_nearest_prediction(
            target[:2],
            np.array([False, False]),
            method="nearest",
        )
        self.assertTrue(np.isnan(unlabeled_run).all())

    def test_block_uncertainty_is_deterministic_and_run_uncertainty_reports_sem(self):
        rng = np.random.default_rng(4)
        target = rng.normal(size=(20, 3))
        prediction = target + 0.1 * rng.normal(size=target.shape)
        runs = np.repeat(np.arange(4), 5)
        kwargs = dict(
            prediction=prediction,
            target=target,
            run_ids=runs,
            exclusion_tr=1,
            chunk_size=7,
            uncertainty="block",
            block_length=2,
            n_boot=12,
            seed=9,
        )
        first = evaluate_embeddings(**kwargs)
        second = evaluate_embeddings(**kwargs)
        self.assertEqual(first["uncertainty"], second["uncertainty"])
        self.assertEqual(first["uncertainty"]["method"], "block")
        self.assertLessEqual(
            first["uncertainty"]["mean_cosine"]["low"],
            first["uncertainty"]["mean_cosine"]["high"],
        )
        run_level = evaluate_embeddings(
            prediction,
            target,
            run_ids=runs,
            uncertainty="run",
            retrieval=True,
            exclusion_tr=1,
        )
        self.assertEqual(run_level["uncertainty"]["n_runs"], 4)
        self.assertIsNotNone(run_level["uncertainty"]["mean_cosine"]["sem"])
        self.assertIsNotNone(run_level["uncertainty"]["retrieval"]["rank_percentile"]["sem"])

    def test_ties_use_expected_top_k_and_reciprocal_rank(self):
        # Each query's positive ties with the one other candidate.
        prediction = np.array([[1.0, 0.0], [1.0, 0.0]])
        target = np.array([[1.0, 1.0], [1.0, -1.0]])
        result = evaluate_embeddings(prediction, target, times=[0, 10], exclusion_tr=0)
        self.assertAlmostEqual(result["retrieval"]["top1"], 0.5)
        self.assertAlmostEqual(result["retrieval"]["mrr"], 0.75)
        self.assertAlmostEqual(result["retrieval"]["rank_percentile"], 0.5)
        boot = evaluate_embeddings(prediction, target, times=[0, 10], exclusion_tr=0,
                                   uncertainty="block", block_length=1, n_boot=20)
        self.assertAlmostEqual(boot["uncertainty"]["retrieval"]["top1"]["mean"], 0.5)
        self.assertAlmostEqual(boot["uncertainty"]["retrieval"]["mrr"]["mean"], 0.75)

    def test_singleton_candidate_pool_is_not_counted_as_retrieval_success(self):
        target = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = evaluate_embeddings(target, target, exclusion_tr=10)
        self.assertEqual(result["retrieval"]["n_queries"], 0)
        self.assertIsNone(result["retrieval"]["top1"])

    def test_shape_errors(self):
        with self.assertRaises(ValueError):
            evaluate_embeddings(np.ones((3, 2)), np.ones((4, 2)))
        with self.assertRaises(ValueError):
            evaluate_embeddings(np.ones((3,)), np.ones((3,)))
        with self.assertRaises(ValueError):
            evaluate_embeddings(np.ones((2, 2)), np.ones((2, 2)), uncertainty="permute")


if __name__ == "__main__":
    unittest.main()
