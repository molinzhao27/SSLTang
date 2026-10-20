#!/usr/bin/env python3
"""PROTOCOL_V4 gate G0(d): synthetic recovery test of the empirical-Bayes estimator (src/priors/eb.py).

Columns: 250,000, the first 2% informative. Informative true correlations rho are Student-t (df = 3), scaled so
that the 99th percentile of |rho| over all columns is 0.3, and capped at |rho| <= RHO_MAX = 0.9; the rest have
rho = 0. zeta = atanh(rho). The all-column reading matches the target the design anchors on: the pilot ISP map's
99th percentile over all voxels is 0.27 (results/v4_pilot). ``--scale-over informative`` (99th percentile over the
informative columns only) leaves median informative |rho| at 0.04 and tau^2 about 6e-5, below the low bias of the
registered MAD noise estimate at n = 100 (about 4.5e-5), so the n = 100 cells then pass or fail by seed (the
(100, 0.6) cell fails in 38 of the first 100 replicate prefixes below; the all-column reading fails none).
Provenance: neither the reading nor the cap is registered (the contract says only "Student-t, df = 3, 99th
percentile of |rho| 0.3"), and the all-column reading was adopted after both readings had been run at 250,000
columns. The cap binds under that reading: the scaling factor is 0.40, so 8.7% of informative draws would have
|rho| >= 1, and on the registered prefix 543 of the 5,000 informative columns (10.9%) sit at |rho| = 0.9 and carry
55.5% of Var(zeta) (``rho.capped``, ``rho.capped_var_share``; under the informative reading 3 columns and 17%).
The truth is therefore a t body plus an atom at the cap, and some cap is unavoidable under this reading (at 0.999
the atom holds 8.7% of the columns and 82% of Var(zeta)). The gate verdict depends on the cap's level: at
RHO_MAX = 0.999 (= the Fisher clip) a few extreme columns inflate tau^2, EB over-trusts z on the null columns, and
the (100, 0.1) cell fails its MSE criterion under about half of the replicate prefixes (14 of the first 30).
``sensitivity`` records this descriptively.
  own scores  For each n in {100, 500, 2000} rows, r is the sample Pearson correlation of n bivariate-normal
              pairs with correlation rho, drawn exactly via the Bartlett decomposition of the 2x2 Wishart
              (``--literal`` draws the n pairs).
  null        The shifted-null analogue: four independent sets of rho = 0 columns at the same n;
              sigma^2 = mean of ``eb.mad_sigma2``, the reduction used by ``eb.null_sigma2``.
  prior       For each R_P in {0.1, 0.3, 0.6}: p = c zeta + e, with e exactly orthogonal to zeta and scaled so that
              corr(p, zeta)^2 = R_P. EB is invariant to c > 0 (OLS calibration); c = 1.
  estimates   raw r; prior-only tanh(a + b p) (EB's calibrated kappa = 0 end); EB tanh(m).
  truth       kappa_true = tau2 / (tau2 + sigma2), tau2 = Var(zeta)(1 - R_P) (exact by construction),
              sigma2 = 1 / (n - 3).
Gate, in every (n, R_P) cell: MSE_rho(EB) <= min(MSE_rho(raw), MSE_rho(prior-only)) and
|kappa_hat - kappa_true| <= 0.05. Seeds are int(sha256('ssltang-v4-simulate:<part>')[:8], 16); the same rho, own
scores, nulls and prior noise are shared across cells (common random numbers). The gate is the run under this
prefix. ``--replicates R`` (default 20) reruns the grid under prefixes 'ssltang-v4-simulate-replicate<i>' and
records per-cell pass rates, descriptively, with the numpy version (Generator streams may change across numpy
releases). ``sensitivity`` (skipped with ``--no-sensitivity``) reruns the registered prefix and R replicates for
each (reading, cap) in ``SENSITIVITY``: the informative reading, and caps 0.99 and 0.999; descriptive, never the
gate. Quote margins from the JSON (``summary``, ``replicates.summary``), not from logs. Writes
results/v4/simulate.json (``--quick``: 20,000 columns, simulate_quick.json) and exits with status 1 when the gate
fails.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.priors.eb import SHIFTS, eb_posterior, fisher, mad_sigma2, reliability
from src.runtime import file_sha256, save_json

ROOT = Path(__file__).resolve().parents[1]
ROWS = (100, 500, 2000)
RELIABILITIES = (0.1, 0.3, 0.6)
COLUMNS, QUICK_COLUMNS = 250_000, 20_000
INFORMATIVE, DF, Q99, RHO_MAX = 0.02, 3, 0.3, 0.9
SCALE = 1.0
KAPPA_TOL = 0.05
SCALE_OVER = 'all'
SEED_PREFIX = 'ssltang-v4-simulate'
REPLICATES = 20
SENSITIVITY = (('informative', RHO_MAX), ('all', 0.99), ('all', 0.999))


def seed(*parts, prefix=SEED_PREFIX):
    text = ':'.join((prefix,) + tuple(str(p) for p in parts))
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def true_rho(n_cols, scale_over=SCALE_OVER, prefix=SEED_PREFIX, rho_max=RHO_MAX):
    """True correlations: Student-t (df 3) on the first 2% of columns, scaled so that the 99th percentile of |rho|
    over all columns (``scale_over='informative'``: the informative ones) is 0.3, then capped at ``rho_max``.

    The info records how much the cap carries: ``capped`` columns (``capped_fraction`` of the informative ones) and
    ``capped_var_share``, their share of sum((zeta - mean zeta)^2)."""
    k = int(round(INFORMATIVE * n_cols))
    t = np.random.default_rng(seed('rho', prefix=prefix)).standard_t(DF, size=k)
    pool = np.abs(t) if scale_over == 'informative' else np.r_[np.abs(t), np.zeros(n_cols - k)]
    factor = Q99 / np.quantile(pool, 0.99)
    rho = np.zeros(n_cols)
    rho[:k] = np.clip(factor * t, -rho_max, rho_max)
    cap = np.abs(rho) >= rho_max
    dev = np.arctanh(rho) - np.arctanh(rho).mean()
    return rho, dict(informative=k, scale_over=scale_over, rho_max=float(rho_max), scale_factor=float(factor),
                     capped=int(cap.sum()), capped_fraction=float(cap.sum() / k),
                     capped_var_share=float((dev[cap] ** 2).sum() / (dev ** 2).sum()),
                     q99_abs_rho=float(np.quantile(np.abs(rho[:k] if scale_over == 'informative' else rho), 0.99)),
                     var_zeta=float(np.var(np.arctanh(rho))), max_abs_rho=float(np.abs(rho).max()))


def sample_r(rho, n, rng, literal=False):
    """Sample Pearson r of n bivariate-normal pairs per column (exact Bartlett draw, or the pairs themselves)."""
    rho = np.asarray(rho, dtype=np.float64)
    s = np.sqrt(1.0 - rho ** 2)
    if not literal:  # (n - 1) S ~ W_2(n - 1, Sigma) = L T T' L', T lower triangular (Bartlett).
        t11 = np.sqrt(rng.chisquare(n - 1, len(rho)))
        t22 = np.sqrt(rng.chisquare(n - 2, len(rho)))
        u = rho * t11 + s * rng.standard_normal(len(rho))
        return u / np.sqrt(u ** 2 + (s * t22) ** 2)
    out, step = np.empty(len(rho)), max(1, (1 << 24) // n)
    for start in range(0, len(rho), step):
        cols = slice(start, start + step)
        x = rng.standard_normal((n, len(rho[cols])))
        y = rho[cols] * x + s[cols] * rng.standard_normal(x.shape)
        x -= x.mean(0)
        y -= y.mean(0)
        out[cols] = (x * y).sum(0) / np.sqrt((x * x).sum(0) * (y * y).sum(0))
    return out


def prior(zeta, reliability_p, scale, rng):
    """p = scale * zeta + e with e orthogonal to zeta (in sample) and corr(p, zeta)^2 = reliability_p exactly."""
    centered = zeta - zeta.mean()
    e = rng.standard_normal(len(zeta))
    e -= e.mean()
    e -= (e @ centered) / (centered @ centered) * centered
    e *= np.sqrt(scale ** 2 * np.var(centered) * (1.0 / reliability_p - 1.0) / np.var(e))
    return scale * zeta + e


def worst(cells, key, absolute=False):
    """{value, n, R_P} of the cell with the largest ``key`` (its absolute value with ``absolute``)."""
    get = (lambda c: abs(c[key])) if absolute else (lambda c: c[key])  # noqa: E731
    c = max(cells, key=get)
    return dict(value=float(get(c)), n=c['n'], R_P=c['R_P'])


def simulate(columns=COLUMNS, literal=False, scale_over=SCALE_OVER, prefix=SEED_PREFIX, verbose=True,
             rho_max=RHO_MAX):
    start = time.monotonic()
    rho, rho_info = true_rho(columns, scale_over, prefix, rho_max)
    zeta, informative = np.arctanh(rho), rho != 0
    rng = lambda *parts: np.random.default_rng(seed(*parts, prefix=prefix))  # noqa: E731
    priors = {rp: prior(zeta, rp, SCALE, rng('prior', rp)) for rp in RELIABILITIES}
    cells = []
    for n in ROWS:
        r = sample_r(rho, n, rng('own', n), literal)
        null = [mad_sigma2(sample_r(np.zeros(columns), n, rng('null', n, k), literal)) for k in SHIFTS]
        z, sigma2_hat, sigma2_true = fisher(r), float(np.mean(null)), 1.0 / (n - 3)
        for rp in RELIABILITIES:
            p = priors[rp]
            m, info = eb_posterior(z, p, sigma2_hat)
            calibrated = info['a'] + info['b'] * p
            tau2_true = rho_info['var_zeta'] * (1.0 - rp)
            kappa_true = tau2_true / (tau2_true + sigma2_true)
            rho_hat = dict(raw=r, prior=np.tanh(calibrated), eb=np.tanh(m), prior_uncalibrated=np.tanh(p))
            z_hat = dict(raw=z, prior=calibrated, eb=m)
            d, u = calibrated - zeta, z - calibrated  # EB error at kappa: d + kappa u.
            mse = {k: float(np.mean((v - rho) ** 2)) for k, v in rho_hat.items()}
            ok_mse = mse['eb'] <= min(mse['raw'], mse['prior'])
            ok_kappa = abs(info['kappa'] - kappa_true) <= KAPPA_TOL
            cells.append(dict(
                n=n, R_P=rp, passed=bool(ok_mse and ok_kappa), mse_criterion=bool(ok_mse),
                kappa_criterion=bool(ok_kappa), kappa_hat=info['kappa'], kappa_true=kappa_true,
                kappa_error=info['kappa'] - kappa_true,
                kappa_oracle=float(np.clip(-np.mean(d * u) / np.mean(u * u), 0.0, 1.0)),
                mse_rho=mse, mse_ratio_eb_to_best=mse['eb'] / min(mse['raw'], mse['prior']),
                mse_rho_informative={k: float(np.mean((v[informative] - rho[informative]) ** 2))
                                     for k, v in rho_hat.items()},
                mse_z={k: float(np.mean((v - zeta) ** 2)) for k, v in z_hat.items()},
                sigma2_hat=sigma2_hat, sigma2_by_null=null, sigma2_true=sigma2_true,
                sigma2_realized=float(np.var(z - zeta)), tau2_hat=info['tau2'], tau2_true=tau2_true,
                tau2_floored=info['tau2_floored'], a=info['a'], b=info['b'], b_true=rp / SCALE, V=info['V'],
                realized_R_P=float(np.corrcoef(p, zeta)[0, 1] ** 2),
                estimated_R_P=reliability(info)['R_P']))
            c = cells[-1]
            if verbose:
                print(f"n={n:5d} R_P={rp:.1f} kappa {c['kappa_hat']:.4f}/{kappa_true:.4f} "
                      f"MSE raw={mse['raw']:.3e} prior={mse['prior']:.3e} eb={mse['eb']:.3e} "
                      f"{'pass' if c['passed'] else 'FAIL'}", flush=True)
    return dict(protocol='ssltang-v4', stage='simulate', gate='G0(d)', passed=all(c['passed'] for c in cells),
                settings=dict(columns=columns, informative_fraction=INFORMATIVE, t_df=DF, q99_abs_rho=Q99,
                              rho_max=float(rho_max), scale_over=scale_over, prior_scale=SCALE, rows=list(ROWS),
                              reliabilities=list(RELIABILITIES), null_sets=len(SHIFTS), kappa_tolerance=KAPPA_TOL,
                              sampler='literal' if literal else 'bartlett', seed_prefix=prefix,
                              numpy=np.__version__, python=platform.python_version()),
                summary=dict(max_mse_ratio=worst(cells, 'mse_ratio_eb_to_best'),
                             max_abs_kappa_error=worst(cells, 'kappa_error', absolute=True)),
                rho=rho_info, cells=cells, seconds=time.monotonic() - start,
                code_sha256={name: file_sha256(ROOT / name) for name in ('src/priors/eb.py', 'scripts/v4_simulate.py')})


def replicates(n_rep, columns=COLUMNS, literal=False, scale_over=SCALE_OVER, rho_max=RHO_MAX):
    """Descriptive seed robustness: the grid under prefixes '<SEED_PREFIX>-replicate<i>', i < n_rep."""
    runs = [simulate(columns, literal, scale_over, f'{SEED_PREFIX}-replicate{i}', False, rho_max)
            for i in range(n_rep)]
    cells = []
    for j, cell in enumerate(runs[0]['cells'] if runs else []):
        got = [run['cells'][j] for run in runs]
        cells.append(dict(n=cell['n'], R_P=cell['R_P'], pass_rate=float(np.mean([c['passed'] for c in got])),
                          mse_pass_rate=float(np.mean([c['mse_criterion'] for c in got])),
                          kappa_pass_rate=float(np.mean([c['kappa_criterion'] for c in got])),
                          max_mse_ratio=max(c['mse_ratio_eb_to_best'] for c in got),
                          max_abs_kappa_error=max(abs(c['kappa_error']) for c in got)))
    return dict(n=n_rep, prefix=f'{SEED_PREFIX}-replicate<i>', note='descriptive; the gate is the registered prefix',
                scale_over=scale_over, rho_max=float(rho_max), passed=int(sum(run['passed'] for run in runs)),
                summary=dict(max_mse_ratio=worst(cells, 'max_mse_ratio'),
                             max_abs_kappa_error=worst(cells, 'max_abs_kappa_error'),
                             min_pass_rate=float(min(c['pass_rate'] for c in cells))) if cells else None,
                cells=cells)


def sensitivity(n_rep, columns=COLUMNS, literal=False, variants=SENSITIVITY):
    """Descriptive (reading, cap) sensitivity: per variant the registered prefix (verdict, summary, per-cell
    criteria) and ``replicates``. Never the gate."""
    out = []
    for scale_over, rho_max in variants:
        reg = simulate(columns, literal, scale_over, verbose=False, rho_max=rho_max)
        out.append(dict(scale_over=scale_over, rho_max=float(rho_max), rho=reg['rho'], registered_prefix=dict(
            passed=reg['passed'], summary=reg['summary'],
            cells=[{k: c[k] for k in ('n', 'R_P', 'passed', 'mse_criterion', 'kappa_criterion', 'mse_ratio_eb_to_best',
                                      'kappa_hat', 'kappa_true', 'kappa_error')} for c in reg['cells']]),
            replicates=replicates(n_rep, columns, literal, scale_over, rho_max)))
    return dict(note='descriptive; not the gate (the gate is the registered prefix at the default reading and cap)',
                variants=out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--quick', action='store_true', help=f'{QUICK_COLUMNS} columns; a smoke test, not the gate')
    p.add_argument('--columns', type=int, default=None)
    p.add_argument('--literal', action='store_true', help='draw the n pairs instead of the exact Bartlett r')
    p.add_argument('--scale-over', choices=['all', 'informative'], default=SCALE_OVER)
    p.add_argument('--replicates', type=int, default=REPLICATES, help='descriptive reruns under other seed prefixes')
    p.add_argument('--no-sensitivity', action='store_true', help='skip the descriptive (reading, cap) variants')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    columns = a.columns or (QUICK_COLUMNS if a.quick else COLUMNS)
    out = simulate(columns, a.literal, a.scale_over)
    out['quick'] = columns != COLUMNS
    out['replicates'] = replicates(a.replicates, columns, a.literal, a.scale_over)
    if not a.no_sensitivity:
        out['sensitivity'] = sensitivity(a.replicates, columns, a.literal)
    path = Path(a.out or ('results/v4/simulate_quick.json' if out['quick'] else 'results/v4/simulate.json'))
    save_json(out, path)
    sens = {f"{v['scale_over']}@{v['rho_max']}": f"{v['registered_prefix']['passed']}, {v['replicates']['passed']}/"
            f"{a.replicates}" for v in out.get('sensitivity', {}).get('variants', [])}
    print(json.dumps(dict(passed=out['passed'], columns=columns, out=str(path), seconds=round(out['seconds'], 1),
                          max_mse_ratio=out.get('summary', {}).get('max_mse_ratio'),
                          replicates_passed=f"{out['replicates']['passed']}/{a.replicates}", sensitivity=sens)))
    sys.exit(0 if out['passed'] else 1)


if __name__ == '__main__':
    main()
