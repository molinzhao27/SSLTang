"""PROTOCOL_V3 §12 encoding-predictability weights: scores, weights, kernels, nesting and the lock."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_v3_weights as w  # noqa: E402
from src.encoding_weights import best_index, column_weights, encoding_scores  # noqa: E402
from src.multikernel_ridge import combine  # noqa: E402
from src.structured_ridge import BASELINE, coefficients, fixed_folds  # noqa: E402
from src.voxel_kernels import DELAYS, KernelBuilder, column_stats  # noqa: E402

ALPHAS = np.logspace(-1, 3, 5)


def reference_scores(y, x, rows, sessions, folds, alphas):
    """Direct numpy out-of-fold ridge with intercept; pooled Pearson r per column and alpha."""
    position = {row: i for i, row in enumerate(rows)}
    preds = np.zeros((len(alphas), len(rows), x.shape[1]))
    for held in folds:
        fit = [i for i in rows if sessions[i] not in held]
        hold = [i for i in rows if sessions[i] in held]
        mean_y, mean_x = y[fit].mean(0), x[fit].mean(0)
        a, b = y[fit] - mean_y, x[fit] - mean_x
        for k, alpha in enumerate(alphas):
            beta = np.linalg.solve(a.T @ a + alpha * np.eye(y.shape[1]), a.T @ b)
            preds[k, [position[i] for i in hold]] = (y[hold] - mean_y) @ beta + mean_x
    actual = x[rows]
    r = np.zeros((len(alphas), x.shape[1]))
    for k in range(len(alphas)):
        for j in range(x.shape[1]):
            if actual[:, j].std() > 0:
                r[k, j] = np.corrcoef(preds[k, :, j], actual[:, j])[0, 1]
    return r


def fixture(n_sessions=6, per_session=10, n_targets=5, n_cols=9, seed=0):
    rng = np.random.default_rng(seed)
    sessions = [f's{i // per_session}' for i in range(n_sessions * per_session)]
    y = rng.normal(size=(len(sessions), n_targets))
    x = rng.normal(size=(len(sessions), n_cols))
    x[:, :4] += y @ rng.normal(size=(n_targets, 4))  # Four predictable columns.
    x[:, -1] = 3.0  # Constant column: r is defined as 0.
    return rng, sessions, y, x


def test_encoding_scores_match_numpy_out_of_fold_ridge():
    _, sessions, y, x = fixture()
    rows = [i for i in range(len(sessions)) if i % 3]  # Unlabeled rows must not inform the scores.
    folds = [('s0', 's1'), ('s2', 's3'), ('s4', 's5')]
    alphas = np.array([0.01, 1.0, 100.0])
    r, alpha, mean_r = encoding_scores(y, torch.as_tensor(x), rows, sessions, folds, alphas, 'cpu', chunk=4)
    expected = reference_scores(y, x, rows, sessions, folds, alphas)
    np.testing.assert_allclose(mean_r, expected.mean(1), rtol=1e-9, atol=1e-12)
    k = int(np.argmax(expected.mean(1)))
    assert alpha == alphas[k]
    np.testing.assert_allclose(r, expected[k], rtol=1e-9, atol=1e-12)
    assert r[-1] == 0.0 and (r[:4] > 0.5).all()
    changed = y.copy()
    changed[[i for i in range(len(sessions)) if i not in rows]] += 100.0
    np.testing.assert_allclose(encoding_scores(changed, torch.as_tensor(x), rows, sessions, folds, alphas, 'cpu')[0],
                               r, rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError):  # A fold without held-out labeled rows.
        encoding_scores(y, torch.as_tensor(x), rows, sessions, [('s0',), ('s9',)], alphas, 'cpu')


def test_best_index_prefers_larger_alpha_within_tolerance():
    assert best_index([0.1, 0.3, 0.3 + 1e-13, 0.2]) == 2
    assert best_index([0.3, 0.3, 0.1]) == 1
    assert best_index([0.3, 0.3 - 1e-9]) == 0
    assert best_index([np.nan, 0.2, np.nan]) == 1
    assert best_index([np.nan, np.nan]) is None


def test_column_weights_rules_normalization_and_fallback():
    scores = {1: np.array([0.4, -0.1, 0.2, 0.0]), 2: np.array([0.1, 0.3, -0.2, 0.05])}
    flat = np.concatenate([scores[1], scores[2]])
    top2 = np.zeros(8)
    top2[[0, 5]] = 1.0  # r = 0.4 and 0.3.
    for rule, raw in ((('uniform', None), np.ones(8)),
                      (('soft', 1.0), np.maximum(flat, 0)),
                      (('soft', 2.0), np.maximum(flat, 0) ** 2),
                      (('hard', 0.25), top2)):
        weights, info = column_weights(scores, rule)
        got = np.concatenate([weights[1], weights[2]])
        np.testing.assert_allclose(got, raw / np.sqrt(np.mean(raw ** 2)), rtol=1e-12)
        np.testing.assert_allclose(np.mean(got ** 2), 1.0, rtol=1e-12)
        assert not info['fallback']
    assert column_weights(scores, ('hard', 0.25))[1]['nonzero_fraction'] == 0.25
    bad = {1: np.full(300, -0.1), 2: np.r_[np.full(299, -0.1), 0.2]}  # 1/600 positive < 1%.
    weights, info = column_weights(bad, ('soft', 1.0))
    assert info['fallback'] and all(np.array_equal(v, np.ones_like(v)) for v in weights.values())


def test_hard_rule_keeps_exact_count_with_ties_in_column_order():
    tied = {1: np.full(10, 0.2), 2: np.full(10, 0.2)}
    weights, info = column_weights(tied, ('hard', 0.10))
    assert info['nonzero_fraction'] == 0.10
    assert np.flatnonzero(weights[1]).tolist() == [0, 1] and not weights[2].any()
    mixed = {1: np.array([0.0, 0.0, 0.5, 0.0]), 2: np.array([0.0, 0.5, 0.0, 0.0])}
    weights, _ = column_weights(mixed, ('hard', 0.50))
    # Keep 4 of 8: both r = 0.5 columns, then zero-r ties in column order (delay 1 first).
    assert np.flatnonzero(weights[1]).tolist() == [0, 1, 2] and np.flatnonzero(weights[2]).tolist() == [1]


def write_cache(tmp_path, rng, n_rows, n_val, n_cols, y):
    cache = {'train': {}, 'val': {}}
    for d in DELAYS:
        x = rng.normal(size=(n_rows, n_cols)) * rng.uniform(0.5, 3.0, size=n_cols) + rng.normal(size=n_cols)
        x[:, :6] += y @ rng.normal(size=(y.shape[1], 6)) * d
        for split, values in (('train', x), ('val', rng.normal(size=(n_val, n_cols)))):
            cache[split][d] = tmp_path / f'{split}_d{d}.npy'
            np.save(cache[split][d], values.astype(np.float32))
    return cache


def test_weighted_kernels_match_reference_and_uniform_is_v0(tmp_path, monkeypatch):
    monkeypatch.setattr(w, 'CHUNK', 7)  # Exercise column-block accumulation.
    rng = np.random.default_rng(3)
    n_rows, n_cols = 48, 20
    sessions = [f's{i // 8}' for i in range(n_rows)]
    y = rng.normal(size=(n_rows, 4))
    cache = write_cache(tmp_path, rng, n_rows, 7, n_cols, y)
    stats_rows = [i for i in range(n_rows) if sessions[i] != 's5']
    fit = [i for i in stats_rows if i % 2]
    enc_folds = [('s0', 's1'), ('s2', 's3'), ('s4',)]
    rules = [('uniform', None), ('soft', 1.0), ('hard', 0.25)]
    kernels, cross, info = w.weighted_kernels(w.Designs(cache, stats_rows, 'cpu'), y, fit, sessions, enc_folds,
                                              rules, ALPHAS, ('val',))
    membership = np.arange(n_cols) // 5
    groups = dict(membership=membership, n_groups=4, sizes=np.bincount(membership))
    base, base_cross, _ = KernelBuilder(cache, groups, 'cpu', chunk=7).build(stats_rows, ('val',))
    coef = coefficients(*BASELINE)
    np.testing.assert_allclose(kernels[rules[0]].numpy(), combine(base, coef).numpy(), rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(cross[rules[0]]['val'].numpy(), combine(base_cross['val'], coef).numpy(),
                               rtol=1e-10, atol=1e-9)
    designs, evals, scores = {}, {}, {}
    for d in DELAYS:
        x = torch.as_tensor(np.load(cache['train'][d]).astype(np.float64))
        mean, std = column_stats(x, torch.as_tensor(stats_rows))
        designs[d] = ((x - mean) / std).numpy()
        evals[d] = ((torch.as_tensor(np.load(cache['val'][d]).astype(np.float64)) - mean) / std).numpy()
        curve = reference_scores(y, designs[d], fit, sessions, enc_folds, ALPHAS)
        scores[d] = curve[best_index(curve.mean(1))]
        assert info['encoding_alpha'][d] == ALPHAS[best_index(curve.mean(1))]
    for rule in rules[1:]:
        weights, _ = column_weights(scores, rule)
        expected = sum((designs[d] * weights[d]) @ (designs[d] * weights[d]).T for d in DELAYS)
        expected_cross = sum((evals[d] * weights[d]) @ (designs[d] * weights[d]).T for d in DELAYS)
        np.testing.assert_allclose(kernels[rule].numpy(), expected, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(cross[rule]['val'].numpy(), expected_cross, rtol=1e-8, atol=1e-8)
    assert info['weights']['hard=0.25']['nonzero_fraction'] == 0.25 and not info['not_estimable']
    # Restricting kernel rows gives the corresponding block, in the given order.
    subset = [40, 3, 17, 8, 22]
    part, part_cross, _ = w.weighted_kernels(w.Designs(cache, stats_rows, 'cpu'), y, fit, sessions, enc_folds,
                                             rules, ALPHAS, ('val',), kernel_rows=subset)
    for rule in rules:
        np.testing.assert_allclose(part[rule].numpy(), kernels[rule].numpy()[np.ix_(subset, subset)],
                                   rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(part_cross[rule]['val'].numpy(), cross[rule]['val'].numpy()[:, subset],
                                   rtol=1e-10, atol=1e-10)
    # Targets outside the fitting rows (held sessions, unlabeled rows) cannot change any kernel.
    changed = y.copy()
    changed[[i for i in range(n_rows) if i not in fit]] *= -7.0
    again, _, _ = w.weighted_kernels(w.Designs(cache, stats_rows, 'cpu'), changed, fit, sessions, enc_folds,
                                     rules, ALPHAS)
    for rule in rules:
        np.testing.assert_allclose(again[rule].numpy(), kernels[rule].numpy(), rtol=1e-12, atol=1e-12)
    # Fewer than two encoding folds: non-uniform rules are not estimable; uniform is unaffected.
    single, _, missing = w.weighted_kernels(w.Designs(cache, stats_rows, 'cpu'), y, fit, sessions, enc_folds[:1],
                                            rules, ALPHAS)
    assert list(single) == [rules[0]] and missing['not_estimable'] == ['soft=1', 'hard=0.25']
    np.testing.assert_allclose(single[rules[0]].numpy(), kernels[rules[0]].numpy(), rtol=1e-12, atol=1e-12)


def cv_fixture(tmp_path, seed, n_sessions=10):
    rng = np.random.default_rng(seed)
    n_rows = 12 * n_sessions
    sessions = [str(3 + i // 12) for i in range(n_rows)]
    run_ids = np.array([f'run{i // 6}' for i in range(n_rows)])
    run_sessions = {f'run{i // 6}': sessions[i] for i in range(n_rows)}
    y = rng.normal(size=(n_rows, 4))
    cache = write_cache(tmp_path, rng, n_rows, 5, 16, y)
    return sessions, run_ids, run_sessions, y, cache


def test_cv_select_nests_scores_inside_each_decoder_fold(tmp_path, monkeypatch):
    sessions, run_ids, run_sessions, y, cache = cv_fixture(tmp_path, 4)
    fixed = fixed_folds(sessions)
    rows = [i for i in range(len(sessions)) if i % 4]
    calls, original = [], w.weighted_kernels

    def spy(designs, y_, fit, row_sessions, enc_folds, *args, **kwargs):
        calls.append(([row_sessions[i] for i in designs.stats_rows.tolist()], fit, list(enc_folds),
                      kwargs.get('kernel_rows')))
        return original(designs, y_, fit, row_sessions, enc_folds, *args, **kwargs)

    monkeypatch.setattr(w, 'weighted_kernels', spy)
    scores, _, log = w.cv_select(cache, rows, list(w.RULES), ALPHAS, y, sessions, run_ids, run_sessions, fixed, 'cpu')
    assert len(calls) == len(fixed) == len(log)
    for held, (stats_sessions, fit, enc_folds, kernel_rows) in zip(fixed, calls):
        held = set(held)
        assert not held & set(stats_sessions)
        assert not held & {sessions[i] for i in fit} and set(fit) <= set(rows)
        assert len(enc_folds) == len(fixed) - 1 and all(not held & set(f) for f in enc_folds)
        assert kernel_rows == rows
    assert set(scores) == set(w.RULES) and all(np.isfinite(v).all() for v in scores.values())
    best = w.select(w.ARMS['W1'], scores, ALPHAS)
    assert tuple(best['rule']) in w.ARMS['W1']


def test_cv_select_marks_weighting_not_estimable_with_two_decoder_folds(tmp_path):
    sessions, run_ids, run_sessions, y, cache = cv_fixture(tmp_path, 6)
    rows = [i for i in range(len(sessions)) if sessions[i] in ('4', '9')]  # Two labeled sessions.
    scores, rule_text, log = w.cv_select(cache, rows, list(w.RULES), ALPHAS, y, sessions, run_ids, run_sessions,
                                         fixed_folds(sessions), 'cpu')
    assert rule_text == 'leave-one-labeled-session-out' and len(log) == 2
    assert np.isfinite(scores[('uniform', None)]).all()
    assert all(np.isnan(v).all() for r, v in scores.items() if r[0] != 'uniform')
    assert w.select(w.ARMS['W1'], scores, ALPHAS) is None and w.select(w.ARMS['W0'], scores, ALPHAS)
    three = [i for i in range(len(sessions)) if sessions[i] in ('4', '9', '11')]
    scores, _, _ = w.cv_select(cache, three, list(w.RULES), ALPHAS, y, sessions, run_ids, run_sessions,
                               fixed_folds(sessions), 'cpu')
    assert all(np.isfinite(v).all() for v in scores.values())


def test_select_ties_prefer_declared_rule_then_larger_alpha():
    alphas = np.array([1.0, 10.0, 100.0])
    scores = {('soft', 0.5): np.array([0.1, 0.2, 0.2]), ('soft', 1.0): np.array([0.2, 0.2, 0.2 + 1e-13]),
              ('soft', 2.0): np.array([0.0, 0.0, 0.0])}
    best = w.select(w.ARMS['W1'], scores, alphas)
    assert best['rule'] == ['soft', 0.5] and best['alpha'] == 100.0 and best['alpha_at_grid_edge']


def write_develop_cells(root, cfg, values, code=None):
    for (s, arm, b), value in values.items():
        folder = root / 'v3w' / s / 'develop' / 'cells'
        folder.mkdir(parents=True, exist_ok=True)
        cell = dict(protocol='ssltang-v3-weights', stage='develop', subject=s, arm=arm, fraction=float(b),
                    estimable=True, selection=dict(rule=['uniform', None], alpha=1.0),
                    val=dict(session_macro_feature_correlation=value), code_sha256=code or w.code_sha256(),
                    config_sha256=w.v3.config_sha256(cfg))
        (folder / f'{arm}_f{b}.json').write_text(json.dumps(cell))


def test_lock_requires_every_registered_cell_and_applies_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(w, 'LOCK', tmp_path / 'lock.json')
    cfg = dict(data=dict(manifest='m.json'), output_root=str(tmp_path / 'v3'))
    gain = {'W0': 0.0, 'W1': 0.006, 'W2': 0.004}
    values = {(s, arm, b): 0.1 + gain[arm] for s in w.REGISTERED for arm in w.ARM_ORDER for b in w.GATE_FRACTIONS}
    write_develop_cells(tmp_path, cfg, {k: v for k, v in values.items() if k[0] != 'UTS03'})
    with pytest.raises(SystemExit, match='Missing development cell'):
        w.stage_lock(cfg)
    write_develop_cells(tmp_path, cfg, {k: v for k, v in values.items() if k[0] == 'UTS03'}, code='stale')
    with pytest.raises(SystemExit, match='rerun develop'):
        w.stage_lock(cfg)
    write_develop_cells(tmp_path, cfg, values)
    lock = w.stage_lock(cfg)
    assert lock['advanced_arm'] == 'W1' and not lock['gates']['W2']['passed'] and len(lock['evidence']) == 18
    assert json.loads((tmp_path / 'lock.json').read_text())['decision'] == 'advance W1'
    values[('UTS02', 'W1', '0.1')] = 0.1 - 0.006  # One subject below -0.005 at 10% fails the gate.
    write_develop_cells(tmp_path, cfg, values)
    lock = w.stage_lock(cfg)
    assert lock['advanced_arm'] is None and lock['decision'].startswith('negative result')
