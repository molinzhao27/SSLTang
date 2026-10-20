"""G0(a) numerical recovery and selection rules for v3 structured voxel ridge."""
import json

import h5py
import numpy as np
import pytest
import torch

from src.evaluation.metrics import _feature_correlation
from src.kernel_ridge import GramRidgePath
from src.multikernel_ridge import KernelRidgePath, combine, run_feature_r, session_macro_scores
from src.structured_ridge import (ARMS, BASELINE, aggregate, budget_folds, candidates, coefficients,
                                  fixed_folds, score_fold, select, select_per_output)
from src.voxel_kernels import KernelBuilder, cache_delay_rows, column_stats, membership_groups

SESSIONS = ['3', '5', '6', '7', '8', '9', '10', '11', '12', '15', '18']


def decompose(x, membership):
    sizes = np.bincount(membership)
    q = np.zeros((x.shape[1], len(sizes)))
    q[np.arange(x.shape[1]), membership] = 1 / np.sqrt(sizes[membership])
    c = x @ q
    return c, x - c @ q.T, q


def test_coarse_detail_kernels_recover_all_voxel_kernel():
    rng = np.random.default_rng(0)
    membership = rng.integers(0, 7, size=40)
    membership[:7] = np.arange(7)
    x, e = rng.normal(size=(12, 40)), rng.normal(size=(5, 40))
    c, r, q = decompose(x, membership)
    ce, re, _ = decompose(e, membership)
    np.testing.assert_allclose(q.T @ q, np.eye(7), atol=1e-12)
    np.testing.assert_allclose(c @ c.T + r @ r.T, x @ x.T, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ce @ c.T + re @ r.T, e @ x.T, rtol=1e-10, atol=1e-10)


def test_kernel_ridge_matches_feature_ridge_and_normalization_invariance():
    rng = np.random.default_rng(1)
    x, y, e = rng.normal(size=(30, 80)), rng.normal(size=(30, 4)), rng.normal(size=(9, 80))
    path = KernelRidgePath(x @ x.T, y, 'cpu')
    reference = GramRidgePath(x, y, 'cpu')
    for alpha in (0.1, 3.0, 1e3):
        ours = path.predict(path.project(e @ x.T), alpha).numpy()
        theirs = reference.predict(reference.project(e), alpha).numpy()
        np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-10)
    # Raw coefficients a_g with lambda equal normalized kernels sum(beta_g K_g / s_g) with lambda / S_a.
    k1, k2 = x[:, :50] @ x[:, :50].T, x[:, 50:] @ x[:, 50:].T
    c1, c2 = e[:, :50] @ x[:, :50].T, e[:, 50:] @ x[:, 50:].T
    a = {'g1': 2.0, 'g2': 0.1}
    raw = KernelRidgePath(combine({'g1': k1, 'g2': k2}, a), y, 'cpu')
    s = {'g1': np.trace(k1) / 30, 'g2': np.trace(k2) / 30}
    total = sum(a[g] * s[g] for g in a)
    beta = {g: a[g] * s[g] / total for g in a}
    normalized = KernelRidgePath(beta['g1'] * k1 / s['g1'] + beta['g2'] * k2 / s['g2'], y, 'cpu')
    lam = 5.0
    np.testing.assert_allclose(
        raw.predict(raw.project(combine({'g1': c1, 'g2': c2}, a)), lam).numpy(),
        normalized.predict(normalized.project(beta['g1'] * c1 / s['g1'] + beta['g2'] * c2 / s['g2']),
                           lam / total).numpy(), rtol=1e-9, atol=1e-10)


def test_per_output_alpha_equals_columnwise_scalar_fits():
    rng = np.random.default_rng(2)
    x, y, e = rng.normal(size=(25, 30)), rng.normal(size=(25, 3)), rng.normal(size=(6, 30))
    path = KernelRidgePath(x @ x.T, y, 'cpu')
    projected = path.project(e @ x.T)
    alphas = [0.5, 10.0, 200.0]
    joint = path.predict_per_output(projected, alphas).numpy()
    for j, alpha in enumerate(alphas):
        np.testing.assert_allclose(joint[:, j], path.predict(projected, alpha).numpy()[:, j], rtol=1e-12)


def test_run_feature_r_and_session_macro_match_reporting_metric():
    rng = np.random.default_rng(3)
    p, t = rng.normal(size=(2, 20, 5)), rng.normal(size=(20, 5))
    codes = np.repeat([0, 1], 10)
    r = run_feature_r(torch.as_tensor(p), torch.as_tensor(t), torch.as_tensor(codes), 2).numpy()
    for a in range(2):
        for run in range(2):
            rows = codes == run
            expected = [_feature_correlation(p[a, rows][:, [j]], t[rows][:, [j]])[0] for j in range(5)]
            np.testing.assert_allclose(r[a, run], expected, rtol=1e-10)
    values = np.array([[0.1, 0.3, 0.5, np.nan]])
    assert session_macro_scores(values, ['a', 'a', 'b', 'c'])[0] == pytest.approx((0.2 + 0.5) / 2)


def test_folds_follow_protocol_and_budget_rules():
    assert fixed_folds(SESSIONS) == [['9', '7', '10'], ['15', '6'], ['18', '11'], ['12', '8'], ['5', '3']]
    fixed = fixed_folds(SESSIONS)
    rows = [s for s in SESSIONS for _ in range(3)]
    folds, rule = budget_folds(rows, list(range(len(rows))), fixed)
    assert len(folds) == 5 and rule.startswith('fixed')
    three = [i for i, s in enumerate(rows) if s in ('3', '9', '15')]
    folds, rule = budget_folds(rows, three, fixed)
    assert sorted(folds) == [('15',), ('3',), ('9',)] and rule == 'leave-one-labeled-session-out'
    assert budget_folds(rows, [0, 1], fixed)[0] is None


def test_selection_tie_rules_and_v1_fallback():
    alphas = np.array([1.0, 10.0, 100.0])
    scores = {c: np.array([0.1, 0.2, 0.2]) for c in candidates()}
    chosen = select('V4', scores, alphas)
    assert chosen['candidate'] == BASELINE and chosen['alpha'] == 100.0  # Declared order, then larger alpha.
    better = dict(scores)
    better[(10.0, (0, 0, 1))] = np.array([0.0, 0.3, 0.0])
    assert select('V4', better, alphas)['candidate'] == (10.0, (0, 0, 1))
    assert select('V2', better, alphas)['candidate'] == BASELINE
    per_output = np.array([[0.1, np.nan], [0.3, np.nan], [0.3, np.nan]])
    alpha_vec, info = select_per_output(per_output, 10.0, alphas)
    assert list(alpha_vec) == [100.0, 10.0] and info['fallback_outputs'] == 1
    assert set(ARMS) == {'V0', 'V2', 'V3', 'V4'} and len(candidates()) == 28


def test_cv_scoring_reproduces_manual_out_of_fold_predictions():
    rng = np.random.default_rng(4)
    n, d = 24, 3
    x = {key: rng.normal(size=(n, 10)) for key in [('C', 1), ('R', 1), ('C', 2), ('R', 2), ('C', 3), ('R', 3)]}
    kernels = {k: torch.as_tensor(v @ v.T) for k, v in x.items()}
    y = rng.normal(size=(n, d))
    fit, held = list(range(16)), list(range(16, 24))
    codes = np.repeat([0, 1], 4)
    alphas = np.array([1.0, 10.0])
    out = score_fold(kernels, fit, held, y, codes, 2, alphas, 'cpu', [BASELINE])
    coef = coefficients(*BASELINE)
    k = sum(coef[g] * (x[g] @ x[g].T) for g in coef)
    path = KernelRidgePath(k[np.ix_(fit, fit)], y[fit], 'cpu')
    for a, alpha in enumerate(alphas):
        pred = path.predict(path.project(k[np.ix_(held, fit)]), alpha).numpy()
        for run in range(2):
            rows = codes == run
            expected = _feature_correlation(pred[rows], y[held][rows])[0]
            assert out[BASELINE]['per_run'][a, run] == pytest.approx(expected, rel=1e-9)
    scores, outputs = aggregate([(['r1', 'r2'], out)], {'r1': 's1', 'r2': 's2'}, alphas)
    assert scores[BASELINE].shape == (2,) and outputs.shape == (2, d)


def test_cache_and_builder_match_numpy_reference(tmp_path):
    rng = np.random.default_rng(5)
    voxels, raw = 12, tmp_path / 'raw'
    membership = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3])
    runs = []
    for i, split in enumerate(('train', 'train', 'val')):
        key = f'ds003020/derivatives/preprocessed_data/UTS01/run{i}.hf5'
        (raw / key).parent.mkdir(parents=True, exist_ok=True)
        data = rng.normal(size=(40, voxels))
        data[3, 2] = np.nan
        with h5py.File(raw / key, 'w') as handle:
            handle['data'] = data
        np.save(tmp_path / f'run{i}.npy', rng.normal(size=(40, 4)).astype(np.float32))
        np.save(tmp_path / f'run{i}.t.npy', rng.normal(size=(40, 3)).astype(np.float32))
        np.save(tmp_path / f'run{i}.v.npy', np.ones(40, dtype=bool))
        runs.append(dict(run_id=f'run{i}', dataset='ds003020', subject='UTS01', story=f'run{i}',
                         session=f's{i}', split=split, n_tr=40, n_features=4, source_key=key,
                         neural_path=str(tmp_path / f'run{i}.npy'), target_path=str(tmp_path / f'run{i}.t.npy'),
                         target_valid_path=str(tmp_path / f'run{i}.v.npy')))
    np.savez(tmp_path / 'norm.npz', subject='UTS01', membership=membership, fit_run_ids=np.array(['run0', 'run1']))
    manifest = dict(subject='UTS01', normalization_path=str(tmp_path / 'norm.npz'), runs=runs)
    (tmp_path / 'm.json').write_text(json.dumps(manifest))
    from src.targets import TargetIndex
    index = {s: TargetIndex(tmp_path / 'm.json', s, window=8, stride=4, cutoffs=(3, 4)) for s in ('train', 'val')}
    cache = {s: cache_delay_rows(index[s], raw, tmp_path / 'cache')[0] for s in index}
    ri, t = index['train'].items[1]
    with h5py.File(raw / runs[ri]['source_key']) as handle:
        expected = np.nan_to_num(handle['data'][t + 2]).astype(np.float32)
    np.testing.assert_array_equal(np.load(cache['train'][2])[1], expected)
    groups = membership_groups(manifest, tmp_path / 'm.json', voxels)
    stats_rows = list(range(len(index['train'])))[:5]
    kernels, cross, _ = KernelBuilder(cache, groups, 'cpu', chunk=5).build(stats_rows, ('val',))
    for d in (1, 2, 3):
        x = np.load(cache['train'][d]).astype(np.float64)
        e = np.load(cache['val'][d]).astype(np.float64)
        mean, std = column_stats(torch.as_tensor(x), torch.as_tensor(stats_rows))
        x, e = (x - mean.numpy()) / std.numpy(), (e - mean.numpy()) / std.numpy()
        c, r, _ = decompose(x, membership)
        ce, re, _ = decompose(e, membership)
        np.testing.assert_allclose(kernels[('C', d)].numpy(), c @ c.T, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(kernels[('R', d)].numpy(), r @ r.T, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(cross['val'][('R', d)].numpy(), re @ r.T, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose((kernels[('C', d)] + kernels[('R', d)]).numpy(), x @ x.T, rtol=1e-10, atol=1e-9)
    with pytest.raises(ValueError):
        membership_groups(dict(manifest, subject='UTS02'), tmp_path / 'm.json', voxels)


def test_cache_provenance_rejects_changed_targets(tmp_path):
    from src.voxel_kernels import verify_cache
    from src.targets import TargetIndex
    rng = np.random.default_rng(6)
    raw = tmp_path / 'raw'
    runs = []
    for i, split in enumerate(('train', 'val')):
        key = f'ds003020/derivatives/preprocessed_data/UTS01/run{i}.hf5'
        (raw / key).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(raw / key, 'w') as handle:
            handle['data'] = rng.normal(size=(40, 6))
        for suffix, value in (('npy', rng.normal(size=(40, 2)).astype(np.float32)),
                              ('t.npy', rng.normal(size=(40, 3)).astype(np.float32)),
                              ('v.npy', np.ones(40, dtype=bool))):
            np.save(tmp_path / f'run{i}.{suffix}', value)
        runs.append(dict(run_id=f'run{i}', dataset='ds003020', subject='UTS01', story=f'run{i}', session=f's{i}',
                         split=split, n_tr=40, n_features=2, source_key=key, neural_path=str(tmp_path / f'run{i}.npy'),
                         target_path=str(tmp_path / f'run{i}.t.npy'), target_valid_path=str(tmp_path / f'run{i}.v.npy')))
    (tmp_path / 'm.json').write_text(json.dumps(dict(subject='UTS01', runs=runs)))
    index = TargetIndex(tmp_path / 'm.json', 'train', window=8, stride=4, cutoffs=(3, 4))
    cache_delay_rows(index, raw, tmp_path / 'cache')
    verify_cache(index, raw, tmp_path / 'cache')
    valid = np.ones(40, dtype=bool)
    valid[12] = False  # Same shape after another target is dropped? No: identities differ.
    np.save(tmp_path / 'run0.v.npy', valid)
    changed = TargetIndex(tmp_path / 'm.json', 'train', window=8, stride=4, cutoffs=(3, 4))
    with pytest.raises(ValueError):
        verify_cache(changed, raw, tmp_path / 'cache')


def test_v1_ties_within_tolerance_prefer_larger_alpha():
    alphas = np.array([1.0, 10.0])
    scores = np.array([[0.5], [0.5 - 5e-13]])
    chosen, _ = select_per_output(scores, 1.0, alphas)
    assert chosen[0] == 10.0
