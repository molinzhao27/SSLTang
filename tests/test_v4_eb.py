"""PROTOCOL_V4 empirical-Bayes priors (src/priors/eb.py): G0(b) identities, calibration, shifted null, G0(d) sim."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import v4_simulate as sim  # noqa: E402
from src.encoding_weights import best_index, column_weights  # noqa: E402
from src.priors.eb import (MIXCV_KAPPAS, eb_column_weights, eb_posterior, eb_weights, fisher, mad_sigma2,  # noqa: E402
                           mixcv_posterior, null_shift_offsets, shift_plan, null_sigma2, reliability, shift_tr_offsets,
                           shifted_targets)
from src.voxel_kernels import DELAYS  # noqa: E402

GAMMAS = (0.5, 1.0, 2.0)


def test_fisher_clips_at_0999():
    np.testing.assert_array_equal(fisher([1.0, -2.0, 0.999, 0.5, 0.0]),
                                  np.arctanh([0.999, -0.999, 0.999, 0.5, 0.0]))
    assert fisher(np.float32(0.25)).dtype == np.float64


def test_kappa_one_reproduces_w1_weights():
    rng = np.random.default_rng(0)
    r = {d: rng.uniform(-0.4, 0.8, size=500) for d in DELAYS}
    p = rng.normal(size=500)
    for d in DELAYS:
        z = fisher(r[d])
        m, info = eb_posterior(z, p, 0.0)  # sigma^2 = 0: kappa = 1.
        assert info['kappa'] == 1.0 and np.array_equal(m, z)
        assert np.array_equal(mixcv_posterior(z, p, info['a'], info['b'], 1.0), z)
        for g in GAMMAS:
            np.testing.assert_allclose(eb_weights(m, g), np.maximum(r[d], 0) ** g, rtol=1e-12, atol=0)
    for g in GAMMAS:  # Through the locked §12 rescaling: mean(w^2) = 1 pooled over delays.
        got, got_info = eb_column_weights({d: fisher(r[d]) for d in DELAYS}, g)
        want, want_info = column_weights(r, ('soft', g))
        for d in DELAYS:
            np.testing.assert_allclose(got[d], want[d], rtol=1e-12, atol=0)
        assert got_info['fallback'] == want_info['fallback'] is False


def test_kappa_zero_unit_calibration_reproduces_prior_weights():
    rng = np.random.default_rng(1)
    isp = rng.uniform(-0.2, 0.6, size=400)
    p, z = fisher(isp), rng.normal(size=400)
    m = mixcv_posterior(z, p, 0.0, 1.0, 0.0)
    assert np.array_equal(m, p)
    for g in GAMMAS:
        np.testing.assert_allclose(eb_weights(m, g), np.maximum(isp, 0) ** g, rtol=1e-12, atol=0)
        got, _ = eb_column_weights({d: m for d in DELAYS}, g)  # PRIOR: the same weights at every delay.
        want, _ = column_weights({d: isp for d in DELAYS}, ('soft', g))
        for d in DELAYS:
            np.testing.assert_allclose(got[d], want[d], rtol=1e-12, atol=0)


def test_eb_posterior_matches_hand_computation_and_floor():
    rng = np.random.default_rng(2)
    p = 0.3 * rng.normal(size=300)
    z = 0.1 + 0.8 * p + 0.2 * rng.normal(size=300)
    for sigma2 in (0.01, 10.0):
        m, info = eb_posterior(z, p, sigma2)
        b, a = np.polyfit(p, z, 1)
        v = np.var(z - a - b * p)
        tau2 = max(v - sigma2, 0.01 * v)
        kappa = tau2 / (tau2 + sigma2)
        np.testing.assert_allclose([info[k] for k in ('a', 'b', 'V', 'tau2', 'kappa')], [a, b, v, tau2, kappa],
                                   rtol=1e-10)
        np.testing.assert_allclose(m, a + b * p + kappa * (z - a - b * p), rtol=1e-12, atol=1e-15)
        assert info['sigma2'] == sigma2 and info['tau2_floored'] == (sigma2 == 10.0)
        assert not info['slope_clipped'] and info['n_columns'] == 300
        assert np.array_equal(m, mixcv_posterior(z, p, info['a'], info['b'], info['kappa']))
    for kappa in MIXCV_KAPPAS:
        np.testing.assert_allclose(mixcv_posterior(z, p, 0.2, 0.5, kappa), 0.2 + 0.5 * p + kappa * (z - 0.2 - 0.5 * p),
                                   rtol=1e-12, atol=1e-15)
    with pytest.raises(ValueError):
        mixcv_posterior(z, p, 0.0, 1.0, 1.5)
    with pytest.raises(ValueError):
        eb_posterior(z, p[:-1], 0.01)


def test_negative_slope_sets_b_zero_and_a_mean():
    rng = np.random.default_rng(3)
    p = rng.normal(size=200)
    z = 0.05 - 0.7 * p + 0.1 * rng.normal(size=200)
    m, info = eb_posterior(z, p, 0.02)
    assert info['slope_clipped'] and info['b_ols'] < 0 and info['b'] == 0.0
    assert info['a'] == pytest.approx(z.mean(), rel=1e-12)
    assert info['V'] == pytest.approx(np.var(z), rel=1e-12)
    tau2 = max(np.var(z) - 0.02, 0.01 * np.var(z))
    kappa = tau2 / (tau2 + 0.02)
    np.testing.assert_allclose(m, z.mean() + kappa * (z - z.mean()), rtol=1e-12)
    _, flat = eb_posterior(z, np.full(200, 0.3), 0.02)  # A constant prior has no slope.
    assert flat['b'] == 0.0 and flat['a'] == pytest.approx(z.mean(), rel=1e-12)


def test_reliability_model_quantities():
    out = reliability(dict(var_z=0.02, sigma2=0.01, b=1.0, var_p=0.005), n_pairs=1000)
    expected = dict(lambda_own=1.0, R_P=0.5, R_own=0.5, R_EB=2 / 3, crossover_lambda=1.0, delta_n=1000.0)
    for k, v in expected.items():
        assert out[k] == pytest.approx(v, rel=1e-12)
    assert reliability(dict(var_z=0.01, sigma2=0.02, b=1.0, var_p=0.1))['lambda_own'] is None


def ridge_oof_r(y, x, rows, groups, folds, alphas):
    """Direct numpy out-of-fold ridge with intercept; pooled Pearson r per column and alpha."""
    position = {row: i for i, row in enumerate(rows)}
    pred = np.zeros((len(alphas), len(rows), x.shape[1]))
    for held in folds:
        fit = [i for i in rows if groups[i] not in held]
        hold = [i for i in rows if groups[i] in held]
        my, mx = y[fit].mean(0), x[fit].mean(0)
        a, b = y[fit] - my, x[fit] - mx
        for k, alpha in enumerate(alphas):
            beta = np.linalg.solve(a.T @ a + alpha * np.eye(y.shape[1]), a.T @ b)
            pred[k, [position[i] for i in hold]] = (y[hold] - my) @ beta + mx
    actual = x[rows]
    return np.array([[np.corrcoef(pred[k, :, j], actual[:, j])[0, 1] for j in range(x.shape[1])]
                     for k in range(len(alphas))])


def null_fixture():
    rng = np.random.default_rng(4)
    run_ids = np.repeat([f'run{i}' for i in range(4)], 14)
    n = len(run_ids)
    rows = [i for i in range(n) if i % 14 not in (0, 9) and i != 30]  # Unlabeled rows; uneven runs.
    half = {i: f'{run_ids[i]}:{int(i % 14 >= 7)}' for i in range(n)}
    groups = [half[i] for i in range(n)]
    folds = [(f'run{i}:0', f'run{i}:1') for i in range(4)]
    targets = rng.normal(size=(n, 3))
    x = rng.normal(size=(n, 6))
    x[:, :3] += targets @ rng.normal(size=(3, 3))
    return run_ids, rows, groups, folds, targets, x


def fixture_tr(n):
    """40 TR between consecutive rows of a 14-row run: every null shift of the fixture is >= 80 TR."""
    return 40 * (np.arange(n) % 14)


def test_shifted_targets_roll_rows_within_runs():
    run_ids, rows, _, _, targets, _ = null_fixture()
    out, applied = shifted_targets(targets, rows, run_ids, 2)
    for run in ('run0', 'run1', 'run2', 'run3'):
        members = [i for i in rows if run_ids[i] == run]
        assert applied[run] == round(2 * len(members) / 5)
        np.testing.assert_array_equal(out[members], np.roll(targets[members], applied[run], axis=0))
    outside = [i for i in range(len(run_ids)) if i not in rows]
    np.testing.assert_array_equal(out[outside], targets[outside])
    assert applied == {'run0': 5, 'run1': 5, 'run2': 4, 'run3': 5}  # m = 12, 12, 11, 12.


def test_null_sigma2_matches_hand_computation():
    run_ids, rows, groups, folds, targets, x = null_fixture()
    alphas = np.array([0.1, 1.0, 10.0, 100.0])
    expected = []
    for k in (1, 2, 3, 4):
        y = targets.copy()
        for run in dict.fromkeys(run_ids):
            members = [i for i in rows if run_ids[i] == run]
            shift = round(k * len(members) / 5)
            for j, i in enumerate(members):
                y[i] = targets[members[(j - shift) % len(members)]]
        curve = ridge_oof_r(y, x, rows, groups, folds, alphas)
        f = np.arctanh(np.clip(curve[best_index(curve.mean(1))], -0.999, 0.999))
        expected.append((1.4826 * np.median(np.abs(f - np.median(f)))) ** 2)
    info, tr = {}, fixture_tr(len(run_ids))
    got = null_sigma2(targets, torch.as_tensor(x), rows, groups, folds, alphas, 'cpu', run_ids, tr, info=info)
    assert got == pytest.approx(np.mean(expected), rel=1e-9)
    np.testing.assert_allclose(info['sigma2_by_shift'], expected, rtol=1e-9)
    assert info['shift_rows'][0] == {'run0': 2, 'run1': 2, 'run2': 2, 'run3': 2}
    assert info['min_tr_offset_overall'] == 80 and info['min_tr_required'] == 40
    changed = targets.copy()
    changed[[i for i in range(len(run_ids)) if i not in rows]] += 100.0  # Unlabeled rows never enter.
    again = null_sigma2(torch.as_tensor(changed), torch.as_tensor(x), rows, groups, folds, alphas, 'cpu', run_ids, tr)
    assert again == pytest.approx(got, rel=1e-12)
    assert mad_sigma2([0.0, 0.1, -0.1, 0.2]) == pytest.approx(
        (1.4826 * np.median(np.abs(np.arctanh([0.0, 0.1, -0.1, 0.2]) - np.arctanh(0.05)))) ** 2, rel=1e-12)


def test_null_shift_tr_offsets_are_checked_and_recorded():
    run_ids, rows, groups, folds, targets, x = null_fixture()
    tr = 40 * (np.arange(len(run_ids)) % 14)  # 40 TR between consecutive rows of a run.
    tr[[5, 19]] = 195  # A silent-TR-like irregularity: the 5th row of run0 and run1 sits 5 TR early.
    # run0 rows 1-8, 10-13 (m = 12, k = 1 shift 2): smallest |offset| is tr[5] - tr[3] = 75; wrapped rows are far.
    got = shift_tr_offsets(rows, run_ids, tr, 1)
    assert got == {'run0': 75, 'run1': 75, 'run2': 80, 'run3': 80}
    for k in (1, 2, 3, 4):
        for run, members in ((r, [i for i in rows if run_ids[i] == r]) for r in ('run0', 'run2')):
            shift = round(k * len(members) / 5)
            want = min(abs(tr[members[j]] - tr[members[(j - shift) % len(members)]]) for j in range(len(members)))
            assert shift_tr_offsets(rows, run_ids, tr, k)[run] == want
    alphas = np.array([1.0, 100.0])
    info = {}
    sigma2 = null_sigma2(targets, torch.as_tensor(x), rows, groups, folds, alphas, 'cpu', run_ids, info=info, tr=tr)
    assert info['min_tr_offset'][0] == got and info['min_tr_required'] == 40 and info['min_tr_offset_overall'] == 75
    assert info['min_tr_offset'] == null_shift_offsets(rows, run_ids, tr)
    assert sigma2 == null_sigma2(targets, torch.as_tensor(x), rows, groups, folds, alphas, 'cpu', run_ids,
                                 fixture_tr(len(run_ids)))  # The check never changes sigma^2.
    close = 4 * (np.arange(len(run_ids)) % 14)  # Stride-4 rows: k = 1 shifts 2 rows = 8 TR < 40.
    with pytest.raises(ValueError, match='below 40 TR'):
        null_sigma2(targets, torch.as_tensor(x), rows, groups, folds, alphas, 'cpu', run_ids, tr=close)
    info = {}
    null_sigma2(targets, torch.as_tensor(x), rows, groups, folds, alphas, 'cpu', run_ids, info=info, tr=close,
                min_tr=None)
    assert min(min(o.values()) for o in info['min_tr_offset']) == 8 and info['min_tr_offset_overall'] == 8


def test_null_sigma2_requires_target_trs_and_checks_before_fitting():
    # Regression: tr defaulted to None and the 40 TR rule was then skipped silently.
    run_ids, rows, groups, folds, targets, x = null_fixture()
    with pytest.raises(TypeError):
        null_sigma2(targets, torch.as_tensor(x), rows, groups, folds, np.array([1.0]), 'cpu', run_ids)
    close = 4 * (np.arange(len(run_ids)) % 14)
    with pytest.raises(ValueError, match='below 40 TR'):  # design=None: the check runs before any encoding fit.
        null_sigma2(targets, None, rows, groups, folds, np.array([1.0]), 'cpu', run_ids, close)
    for bad in (close[:-1], None):
        with pytest.raises(ValueError, match='indexed like run_ids'):
            null_sigma2(targets, None, rows, groups, folds, np.array([1.0]), 'cpu', run_ids, bad)


@pytest.mark.parametrize('stride, shortest_ok', [(1, 230), (4, 221)])
def test_registered_shift_breaks_40_tr_on_short_runs(stride, shortest_ok):
    # TargetIndex rows of one run: target TRs 28..n_tr - 5 on the clock. round(k m / 5) rows stays >= 40 TR for all
    # k only from n_tr = 230 (dense) or 221 (stride 4); a 204-TR story gives 34 TR (dense) or 36 TR (stride 4).
    def offsets(n_tr):
        tr = np.array([t for t in range(28, n_tr - 4) if t % stride == 0])
        return null_shift_offsets(np.arange(len(tr)), np.zeros(len(tr), dtype=int), tr, min_tr=None)
    assert min(min(o.values()) for o in offsets(204)) == {1: 34, 4: 36}[stride]
    for n_tr in (shortest_ok - 1, 204):
        tr = np.array([t for t in range(28, n_tr - 4) if t % stride == 0])
        with pytest.raises(ValueError, match=r'\(1, 0, '):
            null_shift_offsets(np.arange(len(tr)), np.zeros(len(tr), dtype=int), tr)
    for n_tr in (shortest_ok, shortest_ok + 1, 400):
        assert min(min(o.values()) for o in offsets(n_tr)) >= 40


def test_bartlett_sampler_matches_sample_correlations():
    for rho, n in ((0.0, 40), (0.35, 40), (-0.6, 120)):
        values = np.full(20000, rho)
        exact = sim.sample_r(values, n, np.random.default_rng(5))
        literal = sim.sample_r(values, n, np.random.default_rng(6), literal=True)
        assert ks_2samp(exact, literal).pvalue > 1e-3
        for r in (exact, literal):
            z = fisher(r)
            assert abs(z.mean() - (np.arctanh(rho) + rho / (2 * (n - 1)))) < 4 * np.sqrt(1 / (n - 3) / len(z))
            assert z.var() == pytest.approx(1 / (n - 3), rel=0.05)


def test_simulated_truth_and_prior_reliability():
    k = int(round(0.02 * 10000))
    rho, info = sim.true_rho(10000)  # Default reading: 99th percentile of |rho| over all columns.
    assert info['scale_over'] == 'all' and info['informative'] == k and not rho[k:].any()
    assert np.abs(rho).max() <= 0.9 and np.quantile(np.abs(rho), 0.99) == pytest.approx(0.3, rel=1e-12)
    cap = np.abs(rho) >= 0.9  # The cap's footprint is recorded: count, fraction and share of Var(zeta).
    dev = np.arctanh(rho) - np.arctanh(rho).mean()
    assert info['capped'] == cap.sum() > 0 and info['capped_fraction'] == pytest.approx(cap.sum() / k, rel=1e-12)
    assert info['capped_var_share'] == pytest.approx((dev[cap] ** 2).sum() / (dev ** 2).sum(), rel=1e-12)
    loose, loose_info = sim.true_rho(10000, rho_max=0.999)  # Same draws; only the cap moves.
    np.testing.assert_array_equal(loose[~cap], rho[~cap])
    assert loose_info['rho_max'] == 0.999 and loose_info['capped'] < info['capped'] and np.abs(loose).max() > 0.9
    narrow, _ = sim.true_rho(10000, 'informative')
    assert np.quantile(np.abs(narrow[:k]), 0.99) == pytest.approx(0.3, rel=1e-12)
    assert sim.true_rho(10000, prefix='other')[0][:k] != pytest.approx(rho[:k])
    zeta = np.arctanh(rho)
    z = fisher(sim.sample_r(rho, 500, np.random.default_rng(7)))
    for rp in (0.1, 0.3, 0.6):
        p1 = sim.prior(zeta, rp, 1.0, np.random.default_rng(8))
        assert np.corrcoef(p1, zeta)[0, 1] ** 2 == pytest.approx(rp, rel=1e-10)
        p2 = sim.prior(zeta, rp, 0.5, np.random.default_rng(8))  # EB is invariant to the prior's scale.
        np.testing.assert_allclose(eb_posterior(z, p2, 0.002)[0], eb_posterior(z, p1, 0.002)[0], rtol=1e-10)


def test_simulate_quick_passes_gate():
    out = sim.simulate(columns=20000, verbose=False)
    assert len(out['cells']) == 9 and out['settings']['sampler'] == 'bartlett' and out['passed']
    assert out['settings']['scale_over'] == 'all' and out['settings']['numpy'] == np.__version__
    for cell in out['cells']:
        assert abs(cell['kappa_hat'] - cell['kappa_true']) <= 0.05 and cell['kappa_criterion']
        assert cell['mse_rho']['eb'] <= min(cell['mse_rho']['raw'], cell['mse_rho']['prior']) and cell['mse_criterion']
        assert cell['realized_R_P'] == pytest.approx(cell['R_P'], rel=1e-10)
        assert cell['sigma2_hat'] == pytest.approx(1 / (cell['n'] - 3), rel=0.05)
    worst = max(out['cells'], key=lambda c: c['mse_ratio_eb_to_best'])  # The maximum over cells, not the last cell.
    assert out['summary']['max_mse_ratio'] == dict(value=worst['mse_ratio_eb_to_best'], n=worst['n'],
                                                   R_P=worst['R_P'])
    assert out['summary']['max_abs_kappa_error']['value'] == max(abs(c['kappa_error']) for c in out['cells'])
    rep = sim.replicates(3, columns=20000)
    assert rep['n'] == 3 and rep['passed'] == 3 and len(rep['cells']) == 9 and rep['rho_max'] == 0.9
    assert all(c['pass_rate'] == 1.0 for c in rep['cells']) and rep['summary']['min_pass_rate'] == 1.0
    assert rep['summary']['max_mse_ratio']['value'] == max(c['max_mse_ratio'] for c in rep['cells'])


def test_sensitivity_records_reading_and_cap_variants():
    out = sim.sensitivity(1, columns=20000)
    assert [(v['scale_over'], v['rho_max']) for v in out['variants']] == [('informative', 0.9), ('all', 0.99),
                                                                          ('all', 0.999)]
    for v in out['variants']:
        assert v['rho']['rho_max'] == v['rho_max'] == v['replicates']['rho_max']
        assert v['replicates']['scale_over'] == v['scale_over'] and v['replicates']['n'] == 1
        assert len(v['registered_prefix']['cells']) == 9 and 'not the gate' in out['note']
    gate = sim.simulate(columns=20000, verbose=False, rho_max=0.999)['cells']
    assert [c['mse_ratio_eb_to_best'] for c in out['variants'][2]['registered_prefix']['cells']] == \
        [c['mse_ratio_eb_to_best'] for c in gate]


def test_mad_noise_estimate_is_biased_low_at_small_n():
    # Null z = atanh(r) has density proportional to sech(z)^(n - 2). Its (1.4826 MAD)^2 (MAD = median of |z|) falls
    # short of Var(z) by about 0.43% at n = 100: the population bias of the registered sigma^2 reduction.
    for n, low, high in ((100, 0.9950, 0.9962), (2000, 0.9996, 0.99995)):
        grid = np.linspace(0.0, 15 / np.sqrt(n), 200_001)
        density = np.cosh(grid) ** -(n - 2.0)
        mass = np.r_[0.0, np.cumsum(density[1:] + density[:-1])]
        median_abs = np.interp(0.5, mass / mass[-1], grid)
        var = np.trapezoid(grid ** 2 * density, grid) / np.trapezoid(density, grid)
        assert var == pytest.approx(1 / (n - 3), rel=1e-3)
        assert low < (1.4826 * median_abs) ** 2 / var < high


def test_main_exit_status_follows_gate(monkeypatch, tmp_path):
    import json
    sensitivity = dict(note='descriptive', variants=[dict(scale_over='all', rho_max=0.999, replicates=dict(passed=0),
                                                          registered_prefix=dict(passed=False))])
    monkeypatch.setattr(sim, 'replicates', lambda *a, **k: dict(passed=0))
    monkeypatch.setattr(sim, 'sensitivity', lambda *a, **k: sensitivity)
    for passed, code, flags in ((True, 0, []), (False, 1, []), (True, 0, ['--no-sensitivity'])):
        fake = dict(passed=passed, seconds=0.0)
        monkeypatch.setattr(sim, 'simulate', lambda *a, **k: dict(fake))
        path = tmp_path / f'sim{code}{len(flags)}.json'
        monkeypatch.setattr(sys, 'argv', ['v4_simulate.py', '--quick', '--out', str(path)] + flags)
        with pytest.raises(SystemExit) as stop:  # A failing sensitivity variant never changes the exit status.
            sim.main()
        assert stop.value.code == code
        assert ('sensitivity' in json.loads(path.read_text())) == (not flags)


def test_runner_calls_match_the_module_api():
    """Every ``ebm.<name>(...)`` call in scripts/run_v4.py names a function here and binds to its signature; in
    particular ``null_sigma2`` gets the target TRs (``tr``), so the 40 TR check cannot be skipped."""
    import ast
    import inspect
    from src.priors import eb
    runner = Path(__file__).resolve().parents[1] / 'scripts' / 'run_v4.py'
    if not runner.is_file():
        pytest.skip('scripts/run_v4.py is absent')
    calls = [n for n in ast.walk(ast.parse(runner.read_text())) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'ebm']
    for call in calls:
        assert hasattr(eb, call.func.attr), f'run_v4.py calls missing ebm.{call.func.attr}'
        if not any(isinstance(a, ast.Starred) for a in call.args) and all(k.arg for k in call.keywords):
            signature = inspect.signature(getattr(eb, call.func.attr))
            try:
                signature.bind(*call.args, **{k.arg: k.value for k in call.keywords})
            except TypeError as e:
                pytest.fail(f'run_v4.py line {call.lineno}: ebm.{call.func.attr}: {e}')


def test_null_shifts_are_clamped_to_40_tr_in_short_runs():
    # PROTOCOL_V4 §4 clarification: a 204-TR story (dense rows at TRs 28..199) keeps every shift >= 40 TR.
    for stride in (1, 4):
        tr = np.array([t for t in range(28, 204 - 4) if t % stride == 0])
        rows, run_ids = np.arange(len(tr)), np.zeros(len(tr), dtype=int)
        offsets = null_shift_offsets(rows, run_ids, tr, clamp=True)
        assert min(min(o.values()) for o in offsets) >= 40
        raw = [s for _, _, s in shift_plan(rows, run_ids, 1)]
        clamped = [s for _, _, s in shift_plan(rows, run_ids, 1, tr=tr, min_tr=40)]
        assert clamped[0] >= raw[0] and clamped[0] * stride >= 40
    long_tr = np.arange(28, 396)  # A long run: the clamp never changes the registered shifts.
    rows, run_ids = np.arange(len(long_tr)), np.zeros(len(long_tr), dtype=int)
    for k in (1, 2, 3, 4):
        assert ([s for _, _, s in shift_plan(rows, run_ids, k)]
                == [s for _, _, s in shift_plan(rows, run_ids, k, tr=long_tr, min_tr=40)])
