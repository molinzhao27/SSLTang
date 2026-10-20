"""Statistics core for PROTOCOL_V4 §6 (used by scripts/summarize_v4.py).

Every contrast is a vector of per-subject differences Δ_s (arm minus comparator).
  sign_flip_p       one-sided (or two-sided) Monte Carlo sign-flip test on mean Δ, p = (1 + #{T* >= T}) / (n + 1)
  bca_ci            BCa subject-bootstrap CI of mean Δ (Efron & Tibshirani 14.3; matches scipy.stats.bootstrap)
  iut               H1 intersection-union rule: both p < alpha and both means > 0
  fixed_sequence    H1 -> H2 -> ... each tested only if all predecessors are confirmed
  crossed_lmm       per-(subject, section) Δ ~ 1, crossed subject and section random intercepts, Satterthwaite df
  sesoi_tier        90% CI tiers against SESOI = 0.005 r (TIERS)
  exact_binomial_ci Clopper-Pearson CI for the number of subjects with Δ > 0
No unseeded randomness: the seeds are the registered ones (PROTOCOL_V4 §2).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import special, stats

FLIP_SEED, N_FLIPS = 20261003, 100_000
BOOT_SEED, N_BOOT = 20261004, 10_000
ALPHA = 0.025
SESOI = 0.005
TIE = 1e-12  # Relative tolerance (to mean |Δ|) for counting T* = T as T* >= T.
COMPONENTS = ('subject', 'section')
OPTIMIZERS = ('lbfgs', 'powell', 'nm')  # statsmodels MixedLM.fit methods; each is run.
NUMERICAL = (np.linalg.LinAlgError, ValueError, FloatingPointError)  # Recorded as that optimizer's failure.
TIERS = ('meaningful', 'positive but below SESOI', 'positive, magnitude undetermined', 'equivalent to zero',
         'negative but above -SESOI', 'negative, magnitude undetermined', 'meaningfully negative', 'inconclusive')


def _deltas(deltas, minimum=1):
    d = np.asarray(deltas, dtype=np.float64).ravel()
    if len(d) < minimum or not np.all(np.isfinite(d)):
        raise ValueError(f'Need at least {minimum} finite differences, got {d.tolist()}')
    return d


def sign_flip_p(deltas, n=N_FLIPS, seed=FLIP_SEED, one_sided=True):
    """Monte Carlo sign-flip p for the mean of ``deltas``: (1 + #{T* >= T}) / (n + 1).

    One-sided tests mean > 0 (T = mean); two-sided uses |T*| >= |T|. Signs are
    ``1 - 2 * default_rng(seed).integers(0, 2, (n, m))`` (the default int64 draw; a
    different dtype draws a different stream), so the result is a pure function of the
    inputs. Ties within TIE * mean|Δ| count as exceedances (conservative).
    """
    d = _deltas(deltas)
    signs = 1 - 2 * np.random.default_rng(seed).integers(0, 2, (int(n), len(d)))
    total, tol = d.sum(), TIE * len(d) * max(np.abs(d).mean(), np.finfo(float).tiny)
    count = 0
    for start in range(0, int(n), 10_000):
        null = signs[start:start + 10_000].astype(np.float64) @ d
        count += int(np.sum(null >= total - tol) if one_sided else np.sum(np.abs(null) >= abs(total) - tol))
    return (1 + count) / (int(n) + 1)


def bca_ci(deltas, n=N_BOOT, seed=BOOT_SEED, level=0.95):
    """Two-sided BCa bootstrap CI (low, high) for the mean of ``deltas`` (resampling subjects).

    Draws are ``default_rng(seed).integers(0, m, (n, m))``, bias correction uses the
    mid-rank of the observed mean among the bootstrap means, acceleration the
    jackknife, and the interval the linear-interpolation quantiles; this is exactly
    scipy.stats.bootstrap(method='BCa') with the same generator. Constant input
    gives (mean, mean); a degenerate bootstrap gives (nan, nan).
    """
    d = _deltas(deltas, minimum=2)
    theta = d.mean()
    if np.ptp(d) == 0:
        return float(theta), float(theta)
    boot = d[np.random.default_rng(seed).integers(0, len(d), (int(n), len(d)))].mean(-1)
    z0 = special.ndtri(((boot < theta).sum() + (boot <= theta).sum()) / (2 * len(boot)))
    jack = (d.sum() - d) / (len(d) - 1)
    u = jack.mean() - jack
    accel = (u ** 3).sum() / (6 * (u ** 2).sum() ** 1.5)
    z = special.ndtri((1 - level) / 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        q = [special.ndtr(z0 + (z0 + s) / (1 - accel * (z0 + s))) for s in (z, -z)]
    if not np.all(np.isfinite(q)):
        return float('nan'), float('nan')
    low, high = np.quantile(boot, q)
    return float(low), float(high)


def iut(p1, p2, mean1, mean2, alpha=ALPHA):
    """Intersection-union rule of H1: CONFIRMED iff p1 < alpha, p2 < alpha, mean1 > 0 and mean2 > 0."""
    return bool(p1 < alpha and p2 < alpha and mean1 > 0 and mean2 > 0)


def fixed_sequence(results, order):
    """Fixed-sequence testing: {name: 'CONFIRMED' | 'NOT CONFIRMED' | 'NOT TESTED'} in ``order``.

    ``results[name]`` is a bool, a dict with key 'confirmed', or a zero-argument callable
    returning either; it is read (or called) only when every predecessor is confirmed, so
    hypotheses after the first failure need not be present. Only bool or np.bool_ verdicts
    are accepted (TypeError otherwise), so a p-value, mean or status string is never truthy.
    """
    out, open_ = {}, True
    for name in order:
        if not open_:
            out[name] = 'NOT TESTED'
            continue
        value = results[name]
        value = value() if callable(value) else value
        value = value['confirmed'] if isinstance(value, dict) else value
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f'{name}: the verdict must be a bool, got {type(value).__name__} {value!r}')
        confirmed = bool(value)
        out[name] = 'CONFIRMED' if confirmed else 'NOT CONFIRMED'
        open_ = confirmed
    return out


def sesoi_tier(ci90_low, ci90_high, sesoi=SESOI):
    """Tier of a contrast from its 90% CI [low, high]; a complete partition, checked in this order.

    meaningful: low >= SESOI. positive but below SESOI: 0 < low and high < SESOI.
    positive, magnitude undetermined: 0 < low < SESOI <= high. meaningfully negative:
    high <= -SESOI. negative but above -SESOI: -SESOI < low and high < 0. negative,
    magnitude undetermined: low <= -SESOI < high < 0. equivalent to zero (TOST at 5%):
    -SESOI < low <= 0 <= high < SESOI. inconclusive: otherwise.
    """
    low, high = float(ci90_low), float(ci90_high)
    if not (np.isfinite(low) and np.isfinite(high)) or low > high:
        raise ValueError(f'Invalid CI ({low}, {high})')
    if low >= sesoi:
        return TIERS[0]
    if low > 0:
        return TIERS[1] if high < sesoi else TIERS[2]
    if high <= -sesoi:
        return TIERS[6]
    if high < 0:
        return TIERS[4] if low > -sesoi else TIERS[5]
    if low > -sesoi and high < sesoi:
        return TIERS[3]
    return TIERS[7]


def exact_binomial_ci(k, n, level=0.95):
    """Clopper-Pearson CI (low, high) for a proportion with k successes out of n."""
    k, n = int(k), int(n)
    if n < 1 or not 0 <= k <= n:
        raise ValueError(f'Need 0 <= k <= n and n >= 1, got k={k}, n={n}')
    tail = (1 - level) / 2
    low = 0.0 if k == 0 else float(stats.beta.ppf(tail, k, n - k + 1))
    high = 1.0 if k == n else float(stats.beta.ppf(1 - tail, k + 1, n - k))
    return low, high


def dz(deltas):
    """Cohen's d_z = mean / SD (ddof=1); nan when SD is 0."""
    d = _deltas(deltas, minimum=2)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float('nan')


def ols_slopes(x, y):
    """OLS slope of each row of ``y`` [..., F] on ``x`` [F] (H3: per-subject slope of EB[f] - W1 on f)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc = x - x.mean()
    return (y - y.mean(-1, keepdims=True)) @ xc / (xc @ xc)


def oracle_fraction(arm, w0, oracle):
    """Fraction of the oracle gain recovered, mean(arm - W0) / mean(ORACLE - W0); nan if the oracle gain is <= 0."""
    gain = np.mean(np.asarray(arm, dtype=np.float64) - np.asarray(w0, dtype=np.float64))
    full = np.mean(np.asarray(oracle, dtype=np.float64) - np.asarray(w0, dtype=np.float64))
    return float(gain / full) if full > 0 else float('nan')


def contrast_summary(deltas, null=0.0, sesoi=SESOI, n_flips=N_FLIPS, flip_seed=FLIP_SEED, n_boot=N_BOOT,
                     boot_seed=BOOT_SEED):
    """Everything §6 reports with a contrast. ``p_one_sided`` tests mean(Δ) > ``null`` (H6: null = -0.005)."""
    d = _deltas(deltas, minimum=2)
    positive = int((d > 0).sum())
    ci95, ci90 = bca_ci(d, n_boot, boot_seed, 0.95), bca_ci(d, n_boot, boot_seed, 0.90)
    return dict(n=len(d), mean=float(d.mean()), sd=float(d.std(ddof=1)), dz=dz(d), null=float(null),
                p_one_sided=sign_flip_p(d - null, n_flips, flip_seed, True), n_positive=positive,
                positive_ci95=exact_binomial_ci(positive, len(d)), ci95_bca=ci95, ci90_bca=ci90,
                sesoi=float(sesoi), sesoi_tier=sesoi_tier(*ci90, sesoi) if np.all(np.isfinite(ci90)) else None)


def _indicators(labels):
    levels = sorted(set(labels))
    index = {level: i for i, level in enumerate(levels)}
    z = np.zeros((len(labels), len(levels)))
    z[np.arange(len(labels)), [index[v] for v in labels]] = 1.0
    return z


def _gls(y, x, grams, psi):
    """V^-1, Cov(beta), beta and the REML projection P at variance components ``psi``."""
    vi = np.linalg.inv(sum(p * g for p, g in zip(psi, grams)))
    cov_beta = np.linalg.inv(x.T @ vi @ x)
    return vi, cov_beta, cov_beta @ x.T @ vi @ y, vi - vi @ x @ cov_beta @ x.T @ vi


def reml_loglik(y, x, grams, psi):
    """Restricted log-likelihood without its constant: -(log|V| + log|X'V^-1 X| + y'Py) / 2."""
    _, cov_beta, _, proj = _gls(y, x, grams, psi)
    v = sum(p * g for p, g in zip(psi, grams))
    logdet = [2 * np.log(np.diag(np.linalg.cholesky(a))).sum() for a in (v, cov_beta)]  # slogdet warns spuriously
    return float(-0.5 * (logdet[0] - logdet[1] + y @ proj @ y))


def satterthwaite(y, x, grams, psi, contrast=None):
    """Satterthwaite inference for c'beta in y = X beta + e, Cov(y) = sum_k psi_k G_k, at REML estimates psi.

    ``grams`` are the G_k (the last is usually the identity, for the residual).
    Var(psi-hat) is the inverse observed REML information (the expected one if the
    observed is not positive definite) over the components with psi_k > 0;
    components at 0 are held fixed, as lmerTest's zero gradient on the SD scale
    implies. Returns (estimate, se, df, info) with the REML score per component.
    """
    y, x = np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64).reshape(len(y), -1)
    psi = np.asarray(psi, dtype=np.float64)
    c = np.zeros(x.shape[1]) if contrast is None else np.asarray(contrast, dtype=np.float64)
    if contrast is None:
        c[0] = 1.0
    vi, cov_beta, beta, proj = _gls(y, x, grams, psi)
    variance = float(c @ cov_beta @ c)
    active = [k for k in range(len(psi)) if psi[k] > 0]
    pg = [proj @ grams[k] for k in active]
    gpy = [grams[k] @ proj @ y for k in active]
    expected = np.array([[0.5 * np.sum(a * b.T) for b in pg] for a in pg])  # tr(P G_k P G_l) / 2
    observed = np.array([[a @ proj @ b for b in gpy] for a in gpy]) - expected  # y'P G_k P G_l P y - expected
    information, kind = observed, 'observed'
    if np.linalg.eigvalsh(observed).min() <= 0:
        information, kind = expected, 'expected'
    w = vi @ x @ cov_beta @ c
    grad = np.array([w @ grams[k] @ w for k in active])  # d(c'Cov(beta)c) / d psi_k
    df = 2 * variance ** 2 / float(grad @ np.linalg.solve(information, grad))
    py = proj @ y  # REML score: ~0 for interior components, <= 0 for components at 0.
    score = [float(0.5 * (py @ g @ py - np.sum(proj * g))) for g in grams]
    return float(c @ beta), float(np.sqrt(variance)), float(df), dict(information=kind, active=active, score=score)


def reml_polish(y, x, grams, psi, iterations=200, tol=1e-12):
    """Projected Fisher scoring on the REML likelihood from ``psi`` (all psi_k >= 0, the last > 0).

    Components at 0 with a non-positive score stay at 0; steps are halved until the
    restricted log-likelihood does not decrease. Stops when no component moves by more
    than tol * sum(psi). Returns (psi, restricted log-likelihood, iterations).
    """
    y, x = np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64).reshape(len(y), -1)
    floor = np.r_[np.zeros(len(grams) - 1), 1e-10 * y.var()]
    psi = np.maximum(np.asarray(psi, dtype=np.float64), floor)
    llf = reml_loglik(y, x, grams, psi)
    for iteration in range(1, iterations + 1):
        proj = _gls(y, x, grams, psi)[3]
        py, pg = proj @ y, [proj @ g for g in grams]
        score = np.array([0.5 * (py @ g @ py - np.trace(a)) for g, a in zip(grams, pg)])
        free = np.flatnonzero((psi > floor) | (score > 0))
        step = np.zeros_like(psi)
        step[free] = np.linalg.solve([[0.5 * np.sum(pg[k] * pg[j].T) for j in free] for k in free], score[free])
        for _ in range(60):
            trial = np.maximum(psi + step, floor)
            new = reml_loglik(y, x, grams, trial)
            if new >= llf - 1e-13 * abs(llf):
                break
            step /= 2
        else:
            break
        moved = np.abs(trial - psi).max()
        psi, llf = trial, new
        if moved <= tol * psi.sum():
            break
    return psi, llf, iteration


def _mixedlm(frame):
    """statsmodels MixedLM (REML) fits of y ~ 1 with crossed subject and section variance components, per optimizer.

    The import is outside any try: a missing statsmodels raises instead of reading as
    non-convergence. Only NUMERICAL errors inside one optimizer's fit are caught, and
    are recorded as that optimizer's failure (``error``) without discarding the others.
    """
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    fits = []
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        model = smf.mixedlm('y ~ 1', frame, groups='group', re_formula='0',
                            vc_formula={k: f'0 + C({k})' for k in COMPONENTS})
        for method in OPTIMIZERS:
            try:
                fit = model.fit(reml=True, method=method)
            except NUMERICAL as error:
                fits.append(dict(optimizer=method, converged=False, psi=np.full(len(COMPONENTS) + 1, np.nan),
                                 estimate=float('nan'), se=float('nan'), error=f'{type(error).__name__}: {error}'))
                continue
            vcomp = dict(zip(model.exog_vc.names, np.asarray(fit.vcomp, dtype=np.float64)))
            psi = np.array([vcomp[k] for k in COMPONENTS] + [float(fit.scale)])
            ok = bool(fit.converged and np.all(np.isfinite(psi)) and np.all(np.isfinite(fit.fe_params)))
            fits.append(dict(optimizer=method, converged=ok, psi=psi, estimate=float(fit.fe_params.iloc[0]),
                             se=float(fit.bse_fe.iloc[0]), error=None))
    caught = sorted({str(w.message) for w in record if issubclass(w.category, ConvergenceWarning)})
    return fits, caught


def crossed_lmm(df, response='delta', subject='subject', section='section'):
    """Crossed LMM Δ ~ 1 + (1 | subject) + (1 | section) on per-(subject, section) differences (REML).

    statsmodels MixedLM with one all-data group and variance components for subject and
    section (its documented form for crossed effects; grouping by subject would nest
    section within subject), fit with each of OPTIMIZERS. The response is divided by its
    SD for fitting and every output is mapped back. statsmodels' optimizers can report
    convergence short of the optimum and stop near, not at, a zero variance, so the
    converged fit with the highest restricted likelihood is refined by ``reml_polish``
    (the same estimator, to numerical precision; a zero variance is then exactly 0, as in
    lme4). Intercept inference: t with Satterthwaite df. Only when every optimizer ran and
    none converged is the registered fallback lme4/lmerTest via rpy2 used (``engine``
    records the choice); if that fails too, RuntimeError, never a silent no-result.
    Missing statsmodels raises ImportError. Exclusions belong to the endpoint-blind QC
    upstream, so a non-finite response, a missing label or a repeated (subject, section)
    pair raises ValueError instead of being dropped or counted twice.
    """
    data = df[[response, subject, section]]
    y = data[response].to_numpy(np.float64)
    if not len(y) or not np.all(np.isfinite(y)) or data[[subject, section]].isna().to_numpy().any():
        raise ValueError(f'The crossed LMM needs finite {response!r} and non-missing labels in every row')
    labels = {'subject': data[subject].astype(str).tolist(), 'section': data[section].astype(str).tolist()}
    pairs = list(zip(labels['subject'], labels['section']))
    if len(set(pairs)) < len(pairs):
        raise ValueError(f'Repeated (subject, section) rows: {sorted({p for p in pairs if pairs.count(p) > 1})}')
    scale = float(y.std(ddof=1)) if len(y) > 1 and y.std(ddof=1) > 0 else 1.0
    out = dict(n=len(y), n_subjects=len(set(labels['subject'])), n_sections=len(set(labels['section'])),
               response_scale=scale)
    if min(out['n_subjects'], out['n_sections']) < 2:
        raise ValueError('The crossed LMM needs at least two subjects and two sections')
    frame = pd.DataFrame(dict(y=y / scale, group=1, **labels))
    fits, caught = _mixedlm(frame)
    names = COMPONENTS + ('residual',)
    out['warnings'] = caught
    if not any(f['converged'] for f in fits):
        record = [dict(optimizer=f['optimizer'], converged=False, error=f.get('error')) for f in fits]
        try:
            fallback = _lmer(frame, scale)
        except Exception as error:
            raise RuntimeError(f'No statsmodels optimizer converged ({record}) and the registered lme4/lmerTest '
                               f'fallback failed: {type(error).__name__}: {error}') from error
        return {**out, **fallback, 'statsmodels': record}
    z = {k: _indicators(labels[k]) for k in COMPONENTS}
    grams = [z[k] @ z[k].T for k in COMPONENTS] + [np.eye(len(y))]
    x, ys = np.ones((len(y), 1)), frame['y'].to_numpy()
    for f in fits:
        f['reml_loglik'] = reml_loglik(ys, x, grams, f['psi']) if f['converged'] else None
    best = max((f for f in fits if f['converged']), key=lambda f: f['reml_loglik'])
    psi, llf, iterations = reml_polish(ys, x, grams, best['psi'])
    estimate, se, dof, info = satterthwaite(ys, x, grams, psi)
    t = estimate / se
    return {**out, 'engine': 'statsmodels', 'converged': True, 'optimizer': best['optimizer'],
            'estimate': estimate * scale, 'se': se * scale, 'df': dof, 't': t,
            'p_one_sided': float(stats.t.sf(t, dof)), 'p_two_sided': float(2 * stats.t.sf(abs(t), dof)),
            **{f'var_{k}': float(v * scale ** 2) for k, v in zip(names, psi)},
            'boundary': [k for i, k in enumerate(COMPONENTS) if psi[i] == 0], 'information': info['information'],
            'reml_loglik': llf, 'reml_score': info['score'], 'polish_iterations': iterations,
            'statsmodels': [dict(optimizer=f['optimizer'], converged=f['converged'], reml_loglik=f['reml_loglik'],
                                 error=f.get('error'), estimate=f['estimate'] * scale, se=f['se'] * scale,
                                 **{f'var_{k}': float(v * scale ** 2) for k, v in zip(names, f['psi'])})
                            for f in fits]}


def _lmer(frame, scale):
    """Registered fallback: lmerTest::lmer(y ~ 1 + (1|subject) + (1|section), REML) with Satterthwaite df via rpy2."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    ro.r('suppressPackageStartupMessages(library(lmerTest))')
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv['v4_lmm_data'] = frame[['y', 'subject', 'section']]
    result = ro.r('local({fit <- lmer(y ~ 1 + (1 | subject) + (1 | section), data = v4_lmm_data, REML = TRUE); '
                  's <- coef(summary(fit, ddf = "Satterthwaite")); v <- as.data.frame(VarCorr(fit)); '
                  'c(estimate = s[1, 1], se = s[1, 2], df = s[1, 3], t = s[1, 4], '
                  'singular = as.numeric(isSingular(fit)), setNames(v$vcov, paste0("var_", v$grp)))})')
    values = dict(zip(result.names, (float(v) for v in result)))
    t, dof = values['t'], values['df']
    return dict(engine='lme4/lmerTest (rpy2)', converged=True, estimate=values['estimate'] * scale,
                se=values['se'] * scale, df=dof, t=t, p_one_sided=float(stats.t.sf(t, dof)),
                p_two_sided=float(2 * stats.t.sf(abs(t), dof)), var_subject=values['var_subject'] * scale ** 2,
                var_section=values['var_section'] * scale ** 2, var_residual=values['var_Residual'] * scale ** 2,
                singular=bool(values['singular']))
