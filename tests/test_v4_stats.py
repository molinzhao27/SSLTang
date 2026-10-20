"""PROTOCOL_V4 §6 statistics core: sign-flip, BCa, IUT, fixed sequence, SESOI tiers, exact CI, crossed LMM."""
import itertools
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import v4_stats as st


def exact_flip(d, two_sided=False):
    """Exact sign-flip p by enumerating all 2^m sign patterns (identity included)."""
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=len(d))))
    null, observed = signs @ d, d.sum()
    hits = np.abs(null) >= abs(observed) - 1e-12 if two_sided else null >= observed - 1e-12
    return hits.mean()


def anova(y):
    """Balanced two-way (subject x section, one observation per cell) mean squares."""
    i, j = y.shape
    g = y.mean()
    ms_s = j * ((y.mean(1) - g) ** 2).sum() / (i - 1)
    ms_t = i * ((y.mean(0) - g) ** 2).sum() / (j - 1)
    resid = y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + g
    return ms_s, ms_t, (resid ** 2).sum() / ((i - 1) * (j - 1))


def frame(y, drop=()):
    rows = [(f'S{i:02d}', f'sec{j}', y[i, j]) for i in range(y.shape[0]) for j in range(y.shape[1])
            if (i, j) not in drop]
    return pd.DataFrame(rows, columns=['subject', 'section', 'delta'])


def crossed(i=12, j=9, sd_subject=1.0, sd_section=0.6, seed=0, scale=1e-3):
    rng = np.random.default_rng(seed)
    y = 0.4 + rng.normal(0, sd_subject, (i, 1)) + rng.normal(0, sd_section, (1, j)) + rng.normal(0, 1, (i, j))
    return y * scale


def test_sign_flip_p_is_the_registered_monte_carlo_formula():
    d = np.array([0.012, -0.004, 0.007, 0.001, 0.009, -0.002, 0.005, 0.003, -0.006, 0.008])
    signs = 1 - 2 * np.random.default_rng(20261003).integers(0, 2, (100_000, len(d)))  # As documented: int64.
    count = int(np.sum(signs @ d >= d.sum() - 1e-12))
    assert st.sign_flip_p(d) == (1 + count) / 100_001
    assert st.sign_flip_p(d) == 6444 / 100_001  # Hard-coded: any change to the draw (e.g. dtype=int8) breaks this.
    assert st.sign_flip_p(d) == st.sign_flip_p(d.tolist())  # Deterministic, no hidden state.
    assert st.sign_flip_p(np.zeros(5)) == 1.0  # Every flip ties with T = 0.


@pytest.mark.parametrize('two_sided', [False, True])
def test_sign_flip_p_agrees_with_exact_enumeration(two_sided):
    d = np.array([0.012, -0.004, 0.007, 0.001, 0.009, -0.002, 0.005, 0.003, -0.006, 0.008])
    exact = exact_flip(d, two_sided)
    p = st.sign_flip_p(d, one_sided=not two_sided)
    assert abs(p - exact) < 4 * np.sqrt(exact * (1 - exact) / 100_000) + 1e-5
    if not two_sided:  # scipy's exact one-sample sign-flip permutation test as a second reference.
        ref = stats.permutation_test((d,), np.mean, permutation_type='samples', alternative='greater',
                                     n_resamples=np.inf).pvalue
        assert ref == pytest.approx(exact, abs=1e-12)


def test_bca_ci_matches_scipy_bootstrap_with_the_same_generator():
    rng = np.random.default_rng(7)
    d = 0.004 * rng.standard_exponential(40) - 0.002  # Skewed, so bias and acceleration matter.
    for level in (0.95, 0.90):
        ref = stats.bootstrap((d,), np.mean, n_resamples=10_000, method='BCa', confidence_level=level,
                              rng=np.random.default_rng(20261004), vectorized=True).confidence_interval
        np.testing.assert_allclose(st.bca_ci(d, level=level), (ref.low, ref.high), rtol=1e-12, atol=0)
    boot = d[np.random.default_rng(20261004).integers(0, 40, (10_000, 40))].mean(1)
    assert st.bca_ci(d) != pytest.approx(tuple(np.quantile(boot, [0.025, 0.975])), abs=1e-7)  # Not percentile.
    assert st.bca_ci([0.01] * 6) == (0.01, 0.01)


def test_iut_needs_both_contrasts_significant_and_positive():
    assert st.iut(0.001, 0.02, 0.004, 0.002)
    assert not st.iut(0.001, 0.025, 0.004, 0.002)  # p must be strictly below alpha.
    assert not st.iut(0.03, 0.001, 0.004, 0.002)
    assert not st.iut(0.001, 0.001, 0.004, 0.0)
    assert st.iut(0.04, 0.04, 1, 1, alpha=0.05)


def test_fixed_sequence_stops_at_first_failure_without_evaluating_later_hypotheses():
    def never():
        raise AssertionError('evaluated after a failure')

    order = ('H1', 'H2', 'H3', 'H4', 'H5', 'H6')
    results = {'H1': True, 'H2': {'confirmed': True}, 'H3': lambda: False, 'H4': never}  # H5, H6 absent.
    assert st.fixed_sequence(results, order) == {'H1': 'CONFIRMED', 'H2': 'CONFIRMED', 'H3': 'NOT CONFIRMED',
                                                 'H4': 'NOT TESTED', 'H5': 'NOT TESTED', 'H6': 'NOT TESTED'}
    assert st.fixed_sequence({'H1': False}, order)['H2'] == 'NOT TESTED'
    assert set(st.fixed_sequence({h: True for h in order[:5]}, order[:5]).values()) == {'CONFIRMED'}  # H6 dropped.
    assert st.fixed_sequence({'H1': np.bool_(True), 'H2': {'confirmed': np.bool_(False)}}, order[:2]) == {
        'H1': 'CONFIRMED', 'H2': 'NOT CONFIRMED'}


@pytest.mark.parametrize('verdict', ['NOT CONFIRMED', 'False', 0.3, np.float64(0.02), 1, None])
def test_fixed_sequence_rejects_non_bool_verdicts(verdict):
    for value in (verdict, {'confirmed': verdict}, lambda: verdict):
        with pytest.raises(TypeError):
            st.fixed_sequence({'H1': True, 'H2': value}, ('H1', 'H2'))


@pytest.mark.parametrize('ci, tier', [
    ((0.005, 0.02), 'meaningful'),
    ((0.0051, 0.02), 'meaningful'),
    ((0.001, 0.004), 'positive but below SESOI'),
    ((0.001, 0.0049), 'positive but below SESOI'),
    ((0.001, 0.005), 'positive, magnitude undetermined'),  # Upper bound reaches SESOI: not shown to be below it.
    ((0.001, 0.02), 'positive, magnitude undetermined'),
    ((0.0034, 0.0086), 'positive, magnitude undetermined'),  # Point estimate 0.006 > SESOI.
    ((-0.004, 0.0049), 'equivalent to zero'),
    ((0.0, 0.004), 'equivalent to zero'),
    ((-0.004, 0.005), 'inconclusive'),
    ((-0.006, 0.004), 'inconclusive'),
    ((-0.004, -0.001), 'negative but above -SESOI'),
    ((-0.0049, -0.001), 'negative but above -SESOI'),
    ((-0.005, -0.001), 'negative, magnitude undetermined'),
    ((-0.02, -0.001), 'negative, magnitude undetermined'),
    ((-0.02, -0.005), 'meaningfully negative'),
])
def test_sesoi_tiers(ci, tier):
    assert st.sesoi_tier(*ci) == tier


def test_sesoi_tiers_are_a_complete_mirrored_partition():
    flip = {'meaningful': 'meaningfully negative', 'positive but below SESOI': 'negative but above -SESOI',
            'positive, magnitude undetermined': 'negative, magnitude undetermined'}
    flip.update({v: k for k, v in flip.items()})
    grid = np.round(np.arange(-0.012, 0.0121, 0.0005), 6)
    seen = set()
    for low, high in itertools.combinations_with_replacement(grid, 2):
        tier = st.sesoi_tier(low, high)
        seen.add(tier)
        assert st.sesoi_tier(-high, -low) == flip.get(tier, tier), (low, high, tier)
    assert seen == set(st.TIERS)


def test_sesoi_tier_rejects_inverted_ci():
    with pytest.raises(ValueError):
        st.sesoi_tier(0.01, 0.0)


@pytest.mark.parametrize('k', [0, 1, 7, 20, 33, 39, 40])
def test_exact_binomial_ci_is_clopper_pearson(k):
    ref = stats.binomtest(k, 40).proportion_ci(confidence_level=0.95, method='exact')
    np.testing.assert_allclose(st.exact_binomial_ci(k, 40), (ref.low, ref.high), rtol=1e-10, atol=1e-14)


def test_exact_binomial_ci_hand_values():
    assert st.exact_binomial_ci(0, 40) == (0.0, pytest.approx(1 - 0.025 ** (1 / 40), rel=1e-12))
    assert st.exact_binomial_ci(40, 40) == (pytest.approx(0.025 ** (1 / 40), rel=1e-12), 1.0)
    with pytest.raises(ValueError):
        st.exact_binomial_ci(5, 4)


def test_effect_size_slopes_and_oracle_fraction():
    d = np.array([0.01, 0.02, -0.005, 0.015])
    assert st.dz(d) == pytest.approx(d.mean() / d.std(ddof=1), rel=1e-14)
    assert np.isnan(st.dz([0.01, 0.01]))
    f = np.array([0, 0.25, 0.5, 0.75, 1.0])
    y = np.random.default_rng(1).normal(size=(6, 5))
    np.testing.assert_allclose(st.ols_slopes(f, y), [np.polyfit(f, row, 1)[0] for row in y], rtol=1e-12)
    assert st.oracle_fraction([0.3, 0.4], [0.2, 0.2], [0.4, 0.6]) == pytest.approx(0.5)
    assert np.isnan(st.oracle_fraction([0.3], [0.2], [0.2]))


def test_contrast_summary_is_consistent_with_its_parts():
    d = np.random.default_rng(3).normal(0.004, 0.006, 40)
    s = st.contrast_summary(d)
    assert s['p_one_sided'] == st.sign_flip_p(d)
    assert s['ci95_bca'] == st.bca_ci(d) and s['ci90_bca'] == st.bca_ci(d, level=0.90)
    assert s['ci95_bca'][0] < s['ci90_bca'][0] < s['mean'] < s['ci90_bca'][1] < s['ci95_bca'][1]
    assert s['sesoi_tier'] == st.sesoi_tier(*s['ci90_bca'])
    assert s['n_positive'] == int((d > 0).sum()) and s['positive_ci95'] == st.exact_binomial_ci(s['n_positive'], 40)
    h6 = st.contrast_summary(d - 0.006, null=-0.005)  # H6: mean(Δ) > -0.005 via a sign-flip on Δ + 0.005.
    assert h6['p_one_sided'] == st.sign_flip_p(d - 0.006 + 0.005)


def test_crossed_lmm_balanced_interior_matches_anova_reml_and_satterthwaite():
    y = crossed()
    ms_s, ms_t, ms_e = anova(y)
    i, j = y.shape
    assert ms_s > ms_e and ms_t > ms_e  # Interior REML solution: REML = ANOVA estimates.
    r = st.crossed_lmm(frame(y))
    assert r['engine'] == 'statsmodels' and r['converged'] and r['boundary'] == []
    assert r['estimate'] == pytest.approx(y.mean(), rel=1e-10)
    assert r['var_subject'] == pytest.approx((ms_s - ms_e) / j, rel=1e-8)
    assert r['var_section'] == pytest.approx((ms_t - ms_e) / i, rel=1e-8)
    assert r['var_residual'] == pytest.approx(ms_e, rel=1e-8)
    assert r['se'] == pytest.approx(np.sqrt((ms_s + ms_t - ms_e) / (i * j)), rel=1e-8)
    df = (ms_s + ms_t - ms_e) ** 2 / (ms_s ** 2 / (i - 1) + ms_t ** 2 / (j - 1) + ms_e ** 2 / ((i - 1) * (j - 1)))
    assert r['df'] == pytest.approx(df, rel=1e-7)
    assert r['p_one_sided'] == pytest.approx(stats.t.sf(r['estimate'] / r['se'], df), rel=1e-6)
    assert np.max(np.abs(r['reml_score'])) < 1e-6  # The polished REML optimum is stationary.
    assert all(r['reml_loglik'] >= f['reml_loglik'] - 1e-10 for f in r['statsmodels'] if f['converged'])


def test_crossed_lmm_boundary_section_variance_is_exactly_zero():
    y = crossed(sd_section=0.0, seed=4)
    y = y - y.mean(0, keepdims=True) + y.mean()  # Identical section means: MS_section = 0 < MS_error.
    i, j = y.shape
    ms_s = anova(y)[0]
    within = ((y - y.mean(1, keepdims=True)) ** 2).sum() / (i * (j - 1))  # One-way REML after pooling.
    r = st.crossed_lmm(frame(y))
    assert r['boundary'] == ['section'] and r['var_section'] == 0.0
    assert r['var_residual'] == pytest.approx(within, rel=1e-8)
    assert r['var_subject'] == pytest.approx((ms_s - within) / j, rel=1e-8)
    assert r['se'] == pytest.approx(np.sqrt(ms_s / (i * j)), rel=1e-8)
    assert r['df'] == pytest.approx(i - 1, rel=1e-7)
    assert r['reml_score'][1] < 0 and max(abs(r['reml_score'][0]), abs(r['reml_score'][2])) < 1e-6


def test_satterthwaite_matches_numerical_derivatives_on_unbalanced_data():
    y = crossed(seed=2, scale=1.0)
    data = frame(y, drop={(0, 0), (1, 3), (2, 3), (5, 8), (7, 1)})
    r = st.crossed_lmm(data)
    labels = [data['subject'].tolist(), data['section'].tolist()]
    z = [st._indicators(v) for v in labels]
    grams = [a @ a.T for a in z] + [np.eye(len(data))]
    yy, x = data['delta'].to_numpy(), np.ones((len(data), 1))
    psi = np.array([r['var_subject'], r['var_section'], r['var_residual']])
    assert r['boundary'] == [] and np.max(np.abs(r['reml_score'])) < 1e-6
    estimate, se, df, info = st.satterthwaite(yy, x, grams, psi)
    assert (estimate, se, df) == pytest.approx((r['estimate'], r['se'], r['df']), rel=1e-9)
    assert info['information'] == 'observed'

    def var_beta(p):
        return st._gls(yy, x, grams, p)[1][0, 0]

    h = 1e-4 * psi
    eye = np.eye(3)
    grad = np.array([(var_beta(psi + h[k] * eye[k]) - var_beta(psi - h[k] * eye[k])) / (2 * h[k]) for k in range(3)])
    hess = np.zeros((3, 3))
    for k in range(3):
        for m in range(3):
            def f(a, b):
                return st.reml_loglik(yy, x, grams, psi + a * h[k] * eye[k] + b * h[m] * eye[m])
            hess[k, m] = -(f(1, 1) - f(1, -1) - f(-1, 1) + f(-1, -1)) / (4 * h[k] * h[m])
    numeric = 2 * var_beta(psi) ** 2 / (grad @ np.linalg.solve(hess, grad))
    assert df == pytest.approx(numeric, rel=1e-4)
    assert se == pytest.approx(np.sqrt(var_beta(psi)), rel=1e-12)


def test_crossed_lmm_records_the_registered_fallback(monkeypatch):
    y = crossed(seed=1)
    failed = ([dict(optimizer=m, converged=False, psi=np.ones(3), estimate=0.0, se=1.0) for m in st.OPTIMIZERS],
              ['did not converge'])
    monkeypatch.setattr(st, '_mixedlm', lambda data: failed)

    def no_r(data, scale):
        raise ImportError('rpy2 missing')

    monkeypatch.setattr(st, '_lmer', no_r)
    with pytest.raises(RuntimeError, match='rpy2 missing'):  # Neither engine: an error, never a silent no-result.
        st.crossed_lmm(frame(y))
    monkeypatch.setattr(st, '_lmer', lambda data, scale: dict(engine='lme4/lmerTest (rpy2)', converged=True))
    r = st.crossed_lmm(frame(y))
    assert r['engine'] == 'lme4/lmerTest (rpy2)' and [f['optimizer'] for f in r['statsmodels']] == list(st.OPTIMIZERS)


def test_crossed_lmm_missing_statsmodels_raises_instead_of_falling_back(monkeypatch):
    for name in [m for m in sys.modules if m == 'statsmodels' or m.startswith('statsmodels.')] + ['statsmodels']:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(st, '_lmer', lambda data, scale: pytest.fail('fallback used for an import error'))
    with pytest.raises(ImportError):
        st.crossed_lmm(frame(crossed(seed=1)))


def test_crossed_lmm_one_optimizer_raising_keeps_the_other_fits(monkeypatch):
    mixed = pytest.importorskip('statsmodels.regression.mixed_linear_model')
    data = frame(crossed())
    reference = st.crossed_lmm(data)
    fit = mixed.MixedLM.fit

    def flaky(self, *args, **kwargs):
        if kwargs.get('method') == 'powell':
            raise np.linalg.LinAlgError('Singular matrix')
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(mixed.MixedLM, 'fit', flaky)
    monkeypatch.setattr(st, '_lmer', lambda data, scale: pytest.fail('fallback used although two optimizers ran'))
    r = st.crossed_lmm(data)
    record = {f['optimizer']: f for f in r['statsmodels']}
    assert r['engine'] == 'statsmodels' and r['converged'] and r['optimizer'] != 'powell'
    assert not record['powell']['converged'] and 'LinAlgError' in record['powell']['error']
    assert record['lbfgs']['converged'] and record['lbfgs']['error'] is None
    for key in ('estimate', 'se', 'df', 'var_subject', 'var_section', 'var_residual'):
        assert r[key] == pytest.approx(reference[key], rel=1e-7)

    def broken(self, *args, **kwargs):
        raise np.linalg.LinAlgError('Singular matrix')

    monkeypatch.setattr(mixed.MixedLM, 'fit', broken)  # Every optimizer ran and failed: now the fallback is used.
    monkeypatch.setattr(st, '_lmer', lambda data, scale: dict(engine='lme4/lmerTest (rpy2)', converged=True))
    r = st.crossed_lmm(data)
    assert r['engine'] == 'lme4/lmerTest (rpy2)'
    assert all('LinAlgError' in f['error'] for f in r['statsmodels'])

    def bug(self, *args, **kwargs):
        raise KeyError('programming error')

    monkeypatch.setattr(mixed.MixedLM, 'fit', bug)  # Not numerical: propagates.
    with pytest.raises(KeyError):
        st.crossed_lmm(data)


@pytest.mark.parametrize('corrupt', ['nan', 'inf', 'subject_nan', 'duplicate'])
def test_crossed_lmm_rejects_invalid_input_instead_of_dropping_it(corrupt):
    data = frame(crossed())
    if corrupt == 'nan':
        data.loc[3, 'delta'] = np.nan
    elif corrupt == 'inf':
        data.loc[3, 'delta'] = np.inf
    elif corrupt == 'subject_nan':
        data.loc[3, 'subject'] = None
    else:
        data = pd.concat([data, data.iloc[:2]], ignore_index=True)
    with pytest.raises(ValueError):
        st.crossed_lmm(data)


def test_crossed_lmm_agrees_with_lmertest_when_available():
    ro = pytest.importorskip('rpy2.robjects')
    if not ro.r('requireNamespace("lmerTest", quietly = TRUE)')[0]:
        pytest.skip('lmerTest is not installed in R')
    y = crossed(seed=2)
    data = frame(y, drop={(0, 0), (1, 3), (2, 3), (5, 8), (7, 1)})
    ours = st.crossed_lmm(data)
    scale = ours['response_scale']
    theirs = st._lmer(pd.DataFrame(dict(y=data['delta'] / scale, subject=data['subject'], section=data['section'])),
                      scale)
    for key in ('estimate', 'se', 'df', 'var_subject', 'var_section', 'var_residual'):
        assert ours[key] == pytest.approx(theirs[key], rel=1e-4)
