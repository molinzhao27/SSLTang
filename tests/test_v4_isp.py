"""PROTOCOL_V4 ISP/SELF priors: loading, bases, lags, ISP scores, pilot reproduction, MISALIGN, ladder, octants."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types

import h5py
import numpy as np
import pytest

import src.encoding_weights as ew
from src.priors import isp

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = np.logspace(-1, 3, 5)


def reference_scores(x, y, labels, folds, alphas):
    """Direct numpy out-of-fold ridge with intercept from predictors x to outputs y; pooled r per alpha and column."""
    preds = np.zeros((len(alphas),) + y.shape)
    for held in folds:
        fit = np.array([g not in held for g in labels])
        mean_x, mean_y = x[fit].mean(0), y[fit].mean(0)
        a, b = x[fit] - mean_x, y[fit] - mean_y
        for k, alpha in enumerate(alphas):
            beta = np.linalg.solve(a.T @ a + alpha * np.eye(x.shape[1]), a.T @ b)
            preds[k, ~fit] = (x[~fit] - mean_x) @ beta + mean_y
    return np.array([[np.corrcoef(preds[k, :, j], y[:, j])[0, 1] for j in range(y.shape[1])]
                     for k in range(len(alphas))])


def cohort(subjects, lengths, n_voxels, seed=0, noise=1.0, latent=3):
    """Per-run z-scored BOLD {subject: {story: [n, V_s]}} driven by a shared stimulus-locked latent."""
    rng = np.random.default_rng(seed)
    shared = {st: rng.normal(size=(n, latent)) for st, n in lengths.items()}
    out = {}
    for i, s in enumerate(subjects):
        w = rng.normal(size=(latent, n_voxels + i)) * np.linspace(0, 2, n_voxels + i)  # Voxel 0 is pure noise.
        out[s] = {st: (z @ w + noise * rng.normal(size=(len(z), n_voxels + i))).astype(np.float32)
                  for st, z in shared.items()}
        for st, x in out[s].items():
            out[s][st] = ((x - x.mean(0)) / x.std(0)).astype(np.float32)
    return out


def write_h5(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as handle:
        handle['data'] = data


def test_load_bold_zscores_within_run_and_checks_length(tmp_path):
    rng = np.random.default_rng(0)
    data = (rng.normal(size=(300, 7)) * 3 + 5).astype(np.float32)
    data[3, 1], data[7, 2] = np.nan, np.inf  # Non-finite entries.
    data[:, 4], data[:, 6] = 1234.567, 2.1  # Constant columns float32 cannot represent exactly.
    write_h5(tmp_path / 'UTS01' / 'a.hf5', data)
    run = dict(source_key='UTS01/a.hf5', n_tr=300)
    z = isp.load_bold(tmp_path, run)
    assert z.dtype == np.float32 and z.shape == (300, 7) and np.isfinite(z).all()
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-5)
    np.testing.assert_allclose(z.std(0)[[0, 3, 5]], 1, rtol=1e-5)
    assert np.all(z[:, [4, 6]] == 0)
    pilot = np.nan_to_num(data.copy(), nan=0.0, posinf=0.0, neginf=0.0)  # scripts/v4_pilot_isp.py:load_story.
    pilot -= pilot.mean(0)
    pilot /= np.maximum(pilot.std(0), 1e-6)
    assert np.abs(pilot[:, 4]).min() > 1  # The floor alone inflates the mean-rounding residue.
    np.testing.assert_array_equal(z[:, [0, 1, 2, 3, 5]], pilot[:, [0, 1, 2, 3, 5]])  # Bitwise the pilot elsewhere.
    with pytest.raises(ValueError):
        isp.load_bold(tmp_path, dict(source_key='UTS01/a.hf5', n_tr=301))
    with pytest.raises(ValueError):
        isp.load_bold(tmp_path, dict(source_key='../a.hf5'))


def test_constant_voxel_gets_no_reference_loading(tmp_path):
    rng = np.random.default_rng(11)
    runs = []
    for i, n in enumerate((300, 350)):
        data = (rng.normal(size=(n, 5)) * [1, 2, 3, 4, 5] + 100).astype(np.float32)
        data[:, 2] = 812.37
        write_h5(tmp_path / 'UTS02' / f'{i}.hf5', data)
        runs.append(isp.load_bold(tmp_path, dict(source_key=f'UTS02/{i}.hf5', n_tr=n)))
    assert all(np.all(x[:, 2] == 0) for x in runs)
    mean, comps = isp.reference_basis(runs, n_components=4)
    assert mean[2] == 0 and np.abs(comps[2]).max() < 1e-6
    assert np.abs(isp.project(runs[1], (mean, comps))).max() < 10


def test_reference_basis_exact_and_lowrank(monkeypatch):
    rng = np.random.default_rng(1)
    w = rng.normal(size=(4, 30))
    runs = [((rng.normal(size=(n, 4)) * [8, 6, 4, 2]) @ w + 0.1 * rng.normal(size=(n, 30)) + 7).astype(np.float32)
            for n in (50, 70)]
    mean, comps = isp.reference_basis(runs, n_components=4)
    z = np.concatenate(runs).astype(np.float64)
    np.testing.assert_allclose(mean, z.mean(0), rtol=1e-6)
    np.testing.assert_allclose(comps.T @ comps, np.eye(4), atol=1e-5)
    top = np.linalg.svd(z - z.mean(0), full_matrices=False)[2][:4].T
    np.testing.assert_allclose(comps @ comps.T, top @ top.T, atol=1e-4)  # Same subspace.
    assert isp.reference_basis(runs, n_components=500)[1].shape == (30, 30)  # K truncated to the rank.
    monkeypatch.setattr(isp, 'EXACT_ROWS', 10)
    low = isp.reference_basis(runs, n_components=4, seed=3)[1]
    np.testing.assert_allclose(low @ low.T, top @ top.T, atol=1e-3)
    np.testing.assert_array_equal(low, isp.reference_basis(runs, n_components=4, seed=3)[1])  # Seeded.


def test_registered_constants_and_defaults():
    np.testing.assert_array_equal(isp.ALPHAS, np.logspace(0, 10, 21))
    np.testing.assert_array_equal(isp.PILOT_ALPHAS, np.logspace(0, 6, 13))
    assert isp.N_COMPONENTS == 100 and isp.LAGS == (-1, 0, 1) and isp.PILOT_FOLDS == 5
    assert isp.EXACT_ROWS == 6000 and isp.STD_FLOOR == 1e-6

    def defaults(fn):
        return {k: v.default for k, v in inspect.signature(fn).parameters.items() if v.default is not v.empty}

    assert defaults(isp.reference_basis) == dict(n_components=100, seed=0, device='cpu')
    assert defaults(isp.lagged) == dict(lags=(-1, 0, 1)) and defaults(isp.octant_masks) == dict(margin=4)
    assert defaults(isp.design) == dict(lags=(-1, 0, 1), misalign=False)
    assert defaults(isp.predictors) == dict(lags=(-1, 0, 1), misalign=False, device='cpu')
    scores = defaults(isp.isp_scores)
    assert scores.pop('groups') == 'half' and scores.pop('device') == 'cpu'
    np.testing.assert_array_equal(scores.pop('alphas'), np.logspace(0, 10, 21))
    own = defaults(isp.self_scores)
    np.testing.assert_array_equal(own.pop('alphas'), np.logspace(0, 10, 21))
    assert own == dict(n_components=100, margin=4, device='cpu', lags=(-1, 0, 1), seed=0)
    # PROTOCOL_V4: int(sha256('ssltang-v4-ladder:<subject>:<f>')[:8], 16); f = 0: 'ssltang-v4-perm:<subject>'.
    seeds = {0.75: 2542373576, 0.5: 3918455964, 0.25: 1999276853, 0.0: 86696195}
    for f, seed in seeds.items():
        for value in (f, np.float64(f), np.float32(f)):
            assert isp.ladder_seed('UTS01', value) == seed
    assert isp.ladder_seed('UTS01', 0) == seeds[0.0] and len(set(seeds.values())) == 4
    assert isp.ladder_seed('UTS02', 0.5) != seeds[0.5]


def test_design_from_cached_scores_is_the_predictors_path():
    lengths = {'s0': 21, 's1': 26, 's2': 18}
    bold = cohort(['UTS01', 'UTS02', 'UTS03', 'UTS04'], lengths, 5, seed=12)
    stories = ['s1', 's2', 's0']
    refs = {s: bold[s] for s in ('UTS04', 'UTS02', 'UTS03')}
    bases = {s: isp.reference_basis([refs[s][st] for st in stories], 3) for s in refs}
    scores = {s: {st: isp.project(refs[s][st], bases[s]) for st in lengths} for s in refs}
    for misalign in (False, True):
        x = isp.design(scores, stories, misalign=misalign)
        assert x.dtype == np.float64 and x.shape == (65, 3 * 3 * 3)
        np.testing.assert_array_equal(x, isp.predictors(refs, bases, stories, misalign=misalign))
        # Independent layout: references sorted; per reference lag-major K columns; per-run roll before lagging.
        manual = np.zeros_like(x)
        for j, s in enumerate(sorted(refs)):
            start = 0
            for st in stories:
                z = np.roll(scores[s][st], lengths[st] // 2 if misalign else 0, axis=0)
                for t in range(lengths[st]):
                    for k, lag in enumerate((-1, 0, 1)):  # Lag L at row t holds z[t - L] within the run.
                        if 0 <= t - lag < lengths[st]:
                            manual[start + t, (3 * j + k) * 3:(3 * j + k + 1) * 3] = z[t - lag]
                start += lengths[st]
        np.testing.assert_array_equal(x, manual)
    with pytest.raises(ValueError):
        isp.design({}, stories)
    with pytest.raises(ValueError):
        isp.design({'UTS02': {'a': np.zeros((5, 2))}, 'UTS03': {'a': np.zeros((4, 2))}}, ['a'])


def test_lag_padding_does_not_cross_runs_and_references_are_sorted():
    lengths = {'b': 5, 'a': 4}
    refs = {s: {st: np.full((n, 2), 10 * k + (st == 'b') + 1, np.float32) + np.arange(n)[:, None] / 100
                for st, n in lengths.items()} for k, s in enumerate(('UTS03', 'UTS02'))}
    identity = (np.zeros(2), np.eye(2))
    x = isp.predictors(refs, {s: identity for s in refs}, ['b', 'a'])
    assert x.shape == (9, 2 * 2 * 3)
    first = x[:, :6]  # UTS02 (sorted first): lags -1, 0, +1 blocks of two components each.
    b, a = refs['UTS02']['b'], refs['UTS02']['a']
    np.testing.assert_allclose(first[:5, 2:4], b)
    np.testing.assert_allclose(first[5:, 2:4], a)
    np.testing.assert_allclose(first[:4, 0:2], b[1:])  # Lag -1: the next TR of the same run.
    assert np.all(first[4, 0:2] == 0) and np.all(first[8, 0:2] == 0)  # Not the first TR of the next run.
    np.testing.assert_allclose(first[6:, 4:6], a[:-1])  # Lag +1: the previous TR of the same run.
    assert np.all(first[0, 4:6] == 0) and np.all(first[5, 4:6] == 0)
    np.testing.assert_allclose(x[:5, 8:10], refs['UTS03']['b'])
    np.testing.assert_array_equal(isp.lagged(b, (0,)), b)
    with pytest.raises(ValueError):
        isp.predictors({'UTS02': {'a': a}, 'UTS03': {'a': a[:3]}}, {'UTS02': identity, 'UTS03': identity}, ['a'])


def test_half_groups_and_pilot_groups():
    labels, folds = isp.row_groups(['s', 't'], [5, 4], 'half')
    assert labels == ['s:0'] * 2 + ['s:1'] * 3 + ['t:0'] * 2 + ['t:1'] * 2
    assert folds == [('s:0',), ('s:1',), ('t:0',), ('t:1',)]
    labels, folds = isp.row_groups([f's{i}' for i in range(7)], [2] * 7, 'pilot')
    assert labels == [str(i % 5) for i in range(7) for _ in range(2)]
    assert folds == [(str(g),) for g in range(5)]
    with pytest.raises(ValueError):
        isp.row_groups(['s', 's'], [4, 4])
    with pytest.raises(ValueError):
        isp.row_groups(['s'], [4], 'story')


def test_isp_scores_match_numpy_out_of_fold_ridge():
    lengths = {'s0': 23, 's1': 30, 's2': 27}
    bold = cohort(['UTS01', 'UTS02', 'UTS03'], lengths, 6, seed=2)
    stories = ['s2', 's0', 's1']
    refs = {s: bold[s] for s in ('UTS02', 'UTS03')}
    bases = {s: isp.reference_basis([refs[s][st] for st in stories], 3) for s in refs}
    x = isp.predictors(refs, bases, stories)
    assert x.shape == (80, 2 * 3 * 3)
    r, alpha, curve = isp.isp_scores(bold['UTS01'], x, stories, ALPHAS)
    labels, folds = isp.row_groups(stories, [lengths[st] for st in stories])
    y = np.concatenate([bold['UTS01'][st] for st in stories]).astype(np.float64)
    expected = reference_scores(x, y, labels, folds, ALPHAS)
    np.testing.assert_allclose(curve, expected.mean(1), rtol=1e-9, atol=1e-12)
    k = int(np.argmax(expected.mean(1)))
    assert alpha == ALPHAS[k]
    np.testing.assert_allclose(r, expected[k], rtol=1e-9, atol=1e-12)
    assert r[-1] > 0.5 and r[-1] > r[0]  # Stimulus-driven voxels are predictable; the pure-noise voxel is not.
    with pytest.raises(ValueError):
        isp.isp_scores(bold['UTS01'], x[:-1], stories, ALPHAS)


def test_isp_pilot_grouping_reproduces_pilot_script(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('v4_pilot_isp', ROOT / 'scripts' / 'v4_pilot_isp.py')
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    rng = np.random.default_rng(5)
    lengths = {f'story{i}': int(n) for i, n in enumerate(rng.integers(30, 45, size=8))}
    latent = {st: rng.normal(size=(n, 3)) for st, n in lengths.items()}
    for i, s in enumerate(pilot.SUBJECTS):  # Seven shared training stories; one shared validation story.
        runs, w = [], rng.normal(size=(3, 12 + i)) * np.linspace(0, 2, 12 + i)
        for st, n in lengths.items():
            data = (latent[st] @ w + rng.normal(size=(n, 12 + i)) + 3).astype(np.float32)
            data[rng.integers(n), rng.integers(12 + i)] = np.nan
            write_h5(tmp_path / 'raw' / s / f'{st}.hf5', data)
            runs.append(dict(run_id=f'{s}_{st}', story=st, n_tr=n, source_key=f'{s}/{st}.hf5',
                             split='validation' if st == 'story3' and s == 'UTS01' else 'train'))
        if s == 'UTS04':  # A story only one reference heard is not shared.
            write_h5(tmp_path / 'raw' / s / 'solo.hf5', rng.normal(size=(30, 15)).astype(np.float32))
            runs.append(dict(run_id=f'{s}_solo', story='solo', n_tr=30, source_key=f'{s}/solo.hf5', split='train'))
        manifest = tmp_path / 'data' / 'semantic' / s / 'semantic_manifest.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps(dict(subject=s, runs=runs)))

    class Stop(Exception):
        pass

    def stop(*args, **kwargs):
        raise Stop

    calls, original = [], ew.encoding_scores

    def recording(*args, **kwargs):
        calls.append((args, original(*args, **kwargs)))
        return calls[-1][1]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['v4_pilot_isp.py', '--target', 'UTS01', '--components', '4', '--device', 'cpu'])
    monkeypatch.setattr(pilot, 'load_config', lambda path, overrides: dict(raw_root=str(tmp_path / 'raw')))
    monkeypatch.setattr(ew, 'encoding_scores', recording)
    monkeypatch.setitem(sys.modules, 'run_v3', types.SimpleNamespace(Context=stop))  # Stop before label scores.
    monkeypatch.setitem(sys.modules, 'run_v3_weights', types.SimpleNamespace())
    with pytest.raises(Stop):
        pilot.main()
    assert len(calls) == 1
    (features, design, rows, labels, folds, alphas, _), (r_pilot, alpha_pilot, curve_pilot) = calls[0]

    stories = [st for st in sorted(lengths) if st != 'story3']
    runs = {s: pilot.runs_by_story(s) for s in pilot.SUBJECTS}
    bold = {s: {st: isp.load_bold(tmp_path / 'raw', runs[s][st]) for st in stories} for s in pilot.SUBJECTS}
    for st in stories:
        np.testing.assert_array_equal(bold['UTS01'][st], pilot.load_story(runs['UTS01'][st], tmp_path / 'raw'))
    refs = {s: bold[s] for s in reversed(pilot.SUBJECTS[1:])}  # Insertion order must not matter.
    bases = {s: isp.reference_basis([refs[s][st] for st in stories], 4) for s in refs}
    x = isp.predictors(refs, bases, stories)
    r, alpha, curve = isp.isp_scores(bold['UTS01'], x, stories, isp.PILOT_ALPHAS, 'cpu', groups='pilot')

    assert (labels, folds) == isp.row_groups(stories, [lengths[st] for st in stories], 'pilot')
    assert rows == list(range(len(x))) and np.array_equal(alphas, isp.PILOT_ALPHAS)
    np.testing.assert_array_equal(design.numpy(), np.concatenate([bold['UTS01'][st] for st in stories]))
    gram = features.astype(np.float64) @ features.T.astype(np.float64)  # Invariant to component sign/rotation.
    np.testing.assert_allclose(x @ x.T, gram, rtol=0, atol=1e-5 * np.abs(gram).max())
    assert alpha == alpha_pilot and alpha not in (isp.PILOT_ALPHAS[0], isp.PILOT_ALPHAS[-1])
    np.testing.assert_allclose(curve, curve_pilot, rtol=0, atol=1e-6)  # The pilot's components are float32.
    np.testing.assert_allclose(r, r_pilot, rtol=0, atol=1e-6)
    np.testing.assert_allclose(r, np.load(tmp_path / 'results' / 'v4_pilot' / 'UTS01_isp.npy'), atol=1e-6)


def test_misaligned_breaks_alignment_but_keeps_run_content():
    rng = np.random.default_rng(6)
    blocks = [rng.normal(size=(n, 3)) for n in (9, 12)]
    shifted = isp.misaligned(blocks)
    for p, q in zip(blocks, shifted):
        np.testing.assert_array_equal(q[len(p) // 2:], p[:len(p) - len(p) // 2])
        np.testing.assert_array_equal(np.sort(q, axis=0), np.sort(p, axis=0))  # Same rows, same run.
        assert not np.any(np.all(q == p, axis=1))
    lengths = {f's{i}': 60 for i in range(4)}
    bold = cohort(['UTS01', 'UTS02', 'UTS03'], lengths, 8, seed=7, noise=0.5)
    stories = list(lengths)
    refs = {s: bold[s] for s in ('UTS02', 'UTS03')}
    bases = {s: isp.reference_basis([refs[s][st] for st in stories], 3) for s in refs}
    aligned = isp.isp_scores(bold['UTS01'], isp.predictors(refs, bases, stories), stories, ALPHAS)[0]
    broken = isp.isp_scores(bold['UTS01'], isp.predictors(refs, bases, stories, misalign=True), stories, ALPHAS)[0]
    assert aligned[1:].mean() > 0.6 and abs(broken[1:].mean()) < 0.15


def test_ladder_second_moments_and_seeds():
    p = np.random.default_rng(8).standard_t(5, size=400_000) * 0.3 + 0.1
    for f in (0.75, 0.5, 0.25):
        q = isp.ladder(p, f, isp.ladder_seed('UTS01', f))
        assert abs(q.var() / (p.var() / f) - 1) < 0.01
        assert abs(np.corrcoef(p, q)[0, 1] ** 2 - f) < 0.005
        assert abs(q.mean() - p.mean()) < 0.01
        np.testing.assert_array_equal(q, isp.ladder(p, f, isp.ladder_seed('UTS01', f)))
    np.testing.assert_array_equal(isp.ladder(p, 1.0, 0), p)
    perm = isp.ladder(p, 0, isp.ladder_seed('UTS01', 0))
    np.testing.assert_array_equal(np.sort(perm), np.sort(p))
    assert abs(np.corrcoef(p, perm)[0, 1]) < 0.01
    assert isp.ladder_seed('UTS01', 0.5) == int(hashlib.sha256(b'ssltang-v4-ladder:UTS01:0.5').hexdigest()[:8], 16)
    assert isp.ladder_seed('UTS01', 0) == int(hashlib.sha256(b'ssltang-v4-perm:UTS01').hexdigest()[:8], 16)
    with pytest.raises(ValueError):
        isp.ladder(p, 1.5, 0)


@pytest.mark.parametrize('margin', [0, 2, 4])
def test_octant_mask_excludes_margin(margin):
    rng = np.random.default_rng(9)
    grid = np.argwhere(np.ones((11, 12, 9), dtype=bool))
    coords = grid[rng.random(len(grid)) < 0.7]  # An irregular mask on integer voxel coordinates.
    masks = isp.octant_masks(coords, margin)
    owners = np.sum([out for out, _ in masks], axis=0)
    assert np.all(owners == 1)  # Every voxel is an output of exactly one octant.
    med = np.median(coords, axis=0)
    for o, (out, pred) in enumerate(masks):
        bits = np.array([(o >> 2) & 1, (o >> 1) & 1, o & 1], dtype=bool)
        np.testing.assert_array_equal(out, ((coords > med) == bits).all(1))
        # Brute force: Chebyshev distance to the nearest grid point on the octant's side of every median.
        box = grid[((grid > med) == bits).all(1)]
        dist = np.array([np.abs(box - c).max(1).min() for c in coords])
        np.testing.assert_array_equal(pred, dist > margin)
        assert not np.any(pred & out)


def test_self_scores_use_only_voxels_beyond_the_margin():
    grid = np.argwhere(np.ones((8, 8, 8), dtype=bool))  # Median 3.5: octant 0 is coordinates <= 3.
    index = {tuple(c): i for i, c in enumerate(grid)}
    rng = np.random.default_rng(10)
    stories = [f's{i}' for i in range(4)]
    runs = {}
    for st in stories:
        x = rng.normal(size=(60, len(grid)))
        a, b = rng.normal(size=60), rng.normal(size=60)
        x[:, index[0, 0, 0]] += 3 * a
        x[:, index[1, 1, 1]] += 3 * b
        x[:, [index[7, 7, z] for z in range(8)]] += 6 * a[:, None]  # Far from octant 0.
        x[:, [index[i, j, k] for i in (4, 5) for j in (4, 5) for k in (4, 5)]] += 6 * b[:, None]  # In its margin.
        runs[st] = ((x - x.mean(0)) / x.std(0)).astype(np.float32)
    r, info = isp.self_scores(runs, grid, stories, n_components=5, margin=2, alphas=ALPHAS)
    assert len(info) == 8 and all(d['n_outputs'] == 64 for d in info)
    assert r[index[0, 0, 0]] > 0.6 and r[index[1, 1, 1]] < 0.3
    r0 = isp.self_scores(runs, grid, stories, n_components=5, margin=0, alphas=ALPHAS)[0]
    assert r0[index[1, 1, 1]] > 0.6  # Without the margin the adjacent signal leaks in.
    out, pred = isp.octant_masks(grid, 2)[0]
    sources = [runs[st][:, pred] for st in stories]
    basis = isp.reference_basis(sources, 5)
    design = np.concatenate([isp.lagged(isp.project(x, basis)) for x in sources])
    manual = isp.isp_scores({st: runs[st][:, out] for st in stories}, design, stories, ALPHAS)[0]
    np.testing.assert_allclose(r[out], manual)
