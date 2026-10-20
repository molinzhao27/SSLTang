"""v4 Module H (Tang & Huth comparators): TH_SEL, TH_SEL_rand and XDEC on synthetic fixtures, against direct
numpy references and run_v4's engine, with their provenance (TS / SS rules) and audits. CPU, no data files."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_v4  # noqa: E402
import run_v4 as v4  # noqa: E402
import run_v4_tanghuth as rt  # noqa: E402
from src import budgets as bud  # noqa: E402
from src.encoding_weights import encoding_scores  # noqa: E402
from src.multikernel_ridge import KernelRidgePath  # noqa: E402
from src.priors import isp  # noqa: E402
from src.priors import tanghuth as th  # noqa: E402
from src.provenance import Provenance, ProvenanceError  # noqa: E402
from src.voxel_kernels import DELAYS, column_stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = np.logspace(0, 6, 7)  # Converter grid for the fixtures (the registered grid is th.ALPHAS).
DECODER_ALPHAS = np.logspace(-1, 7, 17)
SECTIONS = list(bud.LPP_SECTIONS)
N_TR = {s: 22 + 2 * int(s) for s in SECTIONS}
TARGET, REFS, TEST, THIRD = 'sub-EN057', ('sub-EN058', 'sub-EN059'), '4', 'sub-EN060'
DIM = 4


def ridge_numpy(x, y, new, alpha):
    """Primal ridge with an intercept: fit on (x, y), predict ``new``."""
    mx, my = x.mean(0), y.mean(0)
    beta = np.linalg.solve((x - mx).T @ (x - mx) + alpha * np.eye(x.shape[1]), (x - mx).T @ (y - my))
    return (new - mx) @ beta + my


def zscore(x):
    return ((x - x.mean(0)) / x.std(0)).astype(np.float32)


def lagged_runs(runs):
    return np.concatenate([isp.lagged(np.asarray(z, dtype=np.float64)) for z in runs])


# ------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope='module')
def lpp():
    """LPP-like cohort: a shared stimulus latent drives part of every listener's BOLD 2 TRs later and the
    GPT-like features at the same TR; target voxels 0-9 carry signal, 10-23 are noise."""
    rng = np.random.default_rng(7)
    latent = {s: rng.normal(size=(n, 3)) for s, n in N_TR.items()}
    mix = rng.normal(size=(3, DIM))
    feats = {s: latent[s] @ mix + 0.3 * rng.normal(size=(n, DIM)) for s, n in N_TR.items()}

    def listener(n_voxels, signal):
        w = rng.normal(size=(3, n_voxels))
        w[:, signal:] = 0
        out = {}
        for s, n in N_TR.items():
            lagged = np.zeros_like(latent[s])
            lagged[2:] = latent[s][:-2]
            out[s] = zscore(1.5 * lagged @ w + rng.normal(size=(n, n_voxels)))
        return out
    bold = {TARGET: listener(24, 10)} | {r: listener(16, 16) for r in REFS + (THIRD,)}
    return dict(feats=feats, bold=bold)


def delay_budget(subject, bold, feats, sections, alphas=DECODER_ALPHAS, split_stories=None, eval_sections=(),
                 name='ALL'):
    """(Target-like, run_v4 Budget): dense rows t (t + 3 inside the section) with delay rows BOLD[t + d]."""
    rows = {s: np.arange(0, N_TR[s] - 4) for s in list(sections) + list(eval_sections)}
    raw = {d: np.concatenate([bold[s][rows[s] + d] for s in sections]) for d in DELAYS}
    y = np.concatenate([feats[s][rows[s]] for s in sections])
    story = np.concatenate([[s] * len(rows[s]) for s in sections])
    run = np.array([f'{subject}-{s}' for s in story])
    evals = {'test': {d: np.concatenate([bold[s][rows[s] + d] for s in eval_sections]) for d in DELAYS}} \
        if eval_sections else None
    tgt = th.target_like(subject, alphas, 'cpu', {f'{subject}-{s}': s for s in SECTIONS}, N_TR,
                         split_stories=split_stories or {})
    bd = th.budget_from_arrays(tgt, name, sections, raw, y, story,
                               np.concatenate([rows[s] for s in sections]), run, list(story), eval_raw=evals)
    return tgt, bd


@pytest.fixture(scope='module')
def ref_models(lpp):
    """SS reference models (top 8 of 16 voxels, W1 decoders) on every section but the target's test section."""
    out = {}
    for ref in REFS:
        labeled = th.labeled_stimuli('SS', SECTIONS, held=[TEST])
        tgt, bd = delay_budget(ref, lpp['bold'][ref], lpp['feats'], labeled)
        out[ref] = th.fit_reference_model(tgt, bd, 'ALL', k=8, decoder=True)
    return out


def lpp_target(lpp, budget):
    return delay_budget(TARGET, lpp['bold'][TARGET], lpp['feats'], budget, split_stories={'test': [TEST]},
                        eval_sections=[TEST], name='L3')


def lpp_prov(arm, budget, folds=None, **kwargs):
    p = Provenance(TARGET, 'S', budget, [], [TEST], reference_rule='SS', folds=folds, arm=arm, budget_name='L3',
                   stage='develop', **kwargs)
    for s in list(budget) + [TEST]:  # The LPP target builder resets GPT-2 context per section (R6).
        p.record(TARGET, s, 'transcript', 'context_reset')
    return p


def cell_of(arm, rule='SS', subject=TARGET, budget='L3'):
    ledger = dict(reference_rule=rule)
    if arm == 'XDEC':  # Zero target labels (the XDEC ledger check).
        ledger.update(target_pairs_dense=0, target_pairs_stride4=0, transcript_words=0, transcript_tokens=None)
    return dict(subject=subject, stage='develop', arm=arm, budget=budget, ledger=ledger)


# ------------------------------------------------------------------------------------- rules and sizes
def test_random_mask_uses_the_registered_seed():
    seed = int(hashlib.sha256(b'ssltang-v4-thsel-rand:UTS05').hexdigest()[:8], 16)
    assert th.rand_seed('UTS05') == seed
    mask = th.random_mask(100, 17, 'UTS05')
    expected = np.zeros(100, dtype=bool)
    expected[np.random.default_rng(seed).choice(100, size=17, replace=False)] = True
    assert mask.sum() == 17 and np.array_equal(mask, expected)
    assert np.array_equal(mask, th.random_mask(100, 17, 'UTS05'))
    assert not np.array_equal(mask, th.random_mask(100, 17, 'UTS06'))
    selector = th.RandomSelector(100, 17, 'UTS05')
    weights, info = selector(['a', 'b'])
    assert np.array_equal(weights, mask.astype(float)) and info['seed'] == seed and info['k'] == 17
    with pytest.raises(ValueError):
        th.random_mask(10, 11, 'UTS05')


def test_top_k_goal_sizes_and_mask_weights_equal_the_engine_prior_slot():
    s = np.array([0.1, np.nan, 0.5, 0.5, -1.0, 0.3])
    assert th.top_k(s, 3).tolist() == [2, 3, 5]
    assert th.top_k(s, 2).tolist() == [2, 3]  # Ties go to the lower index.
    assert 1 in th.top_k(s, 6) and th.top_k(s, 5).tolist() == [0, 2, 3, 4, 5]  # NaN is last.
    with pytest.raises(ValueError):
        th.top_k(s, 0)
    assert th.goal_size(90_000, 'ds003020') == 15_000 and th.goal_size(9_000, 'ds003020') == 9_000
    assert th.goal_size(140_369, 'lpp') == round(0.17 * 140_369)
    mask = th.selection_mask(np.random.default_rng(0).normal(size=50), 9)
    w = th.mask_weights(mask)
    assert np.isclose(np.mean(np.concatenate([w[d] for d in DELAYS]) ** 2), 1.0)
    for gamma in v4.GAMMAS:
        soft, info = v4.soft_weights(mask.astype(float), gamma, broadcast=True)
        assert not info['fallback']
        for d in DELAYS:
            np.testing.assert_allclose(soft[d], w[d], rtol=1e-12)
            assert np.array_equal(soft[d] > 0, mask)


def test_ts_and_ss_reference_labels():
    heard = ['a', 'b', 'c', 'v1', 't1']
    assert th.labeled_stimuli('TS', ['x2', 'a', 'x1', 't1'], heard=heard) == ['x1', 'x2']
    assert th.labeled_stimuli('SS', SECTIONS, held=[TEST]) == [s for s in SECTIONS if s != TEST]
    with pytest.raises(ValueError):
        th.labeled_stimuli('none', SECTIONS)
    assert th.rule_for('ds003020') == 'TS' and th.rule_for('lpp') == 'SS'
    budget = bud.lpp_section_budgets(TEST, SECTIONS)
    assert th.lpp_unheard(SECTIONS, budget['L3'], TEST) == [s for s in SECTIONS if s not in budget['L3'] + [TEST]]
    assert th.lpp_unheard(SECTIONS, budget['L8'], TEST) == []  # TH_SEL is not estimable at L8.


def test_two_reference_sensitivity_sets_are_registered_per_dataset():
    assert th.sensitivity_pairs(['UTS03', 'UTS01', 'UTS02'], 'ds003020') == [
        ('UTS01', 'UTS02'), ('UTS01', 'UTS03'), ('UTS02', 'UTS03')] == list(rt.SENSITIVITY_PAIRS)
    refs = [f'sub-EN{i:03d}' for i in range(58, 106)]  # 48 references of one LPP target.
    order = sorted(refs, key=lambda r: hashlib.sha256(f'ssltang-v4-thsel-pair:sub-EN057:{r}'.encode()).hexdigest())
    assert th.sensitivity_pairs(refs[::-1], 'lpp', 'sub-EN057') == [tuple(sorted(order[:2]))]
    for args in ((['UTS01'], 'ds003020'), (refs, 'lpp'), (refs, 'other', 'sub-EN057')):
        with pytest.raises(ValueError):
            th.sensitivity_pairs(*args)


# ------------------------------------------------------------------------------------ lagged dual ridge
def test_lag_rows_and_gram_match_isp_lagged_within_runs():
    rng = np.random.default_rng(1)
    runs = [rng.normal(size=(n, 7)) for n in (9, 5, 1)]
    explicit = lagged_runs(runs)
    np.testing.assert_array_equal(th.lag_rows(runs).numpy(), explicit)
    np.testing.assert_allclose(th.gram(runs, chunk=3).numpy(), explicit @ explicit.T, rtol=1e-12, atol=1e-12)
    other = [rng.normal(size=(4, 7))]
    np.testing.assert_allclose(th.gram(other, runs, chunk=2).numpy(), isp.lagged(other[0]) @ explicit.T,
                               rtol=1e-12, atol=1e-12)
    assert np.all(explicit[9, :7] == runs[1][1]) and np.all(explicit[9, 14:] == 0)  # Lag +1 does not cross runs.


def test_gram_factor_reproduces_encoding_scores_and_dual_ridge_matches_primal_ridge():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(40, 120))  # Wider than tall, like the converters.
    y = x[:, :5] @ rng.normal(size=(5, 6)) + rng.normal(size=(40, 6))
    yt = torch.as_tensor(y)
    labels, folds = isp.row_groups(['a', 'b', 'c'], [15, 10, 15], 'half')
    r0, a0, c0 = encoding_scores(x, yt, list(range(40)), labels, folds, ALPHAS, 'cpu')
    k = torch.as_tensor(x @ x.T)
    r1, a1, c1 = encoding_scores(th.gram_factor(k).numpy(), yt, list(range(40)), labels, folds, ALPHAS, 'cpu')
    r2, a2, c2 = th.converter_alpha(k, yt, ['a', 'b', 'c'], [15, 10, 15], ALPHAS)
    assert a0 == a1 == a2
    np.testing.assert_allclose(r1, r0, atol=1e-9)
    np.testing.assert_allclose(r2, r0, atol=1e-9)
    np.testing.assert_allclose(c1, c0, atol=1e-9)
    new = rng.normal(size=(7, 120))
    cross = torch.as_tensor(new @ x.T)
    ridge = th.DualRidge(k)
    pred = ridge.predict(ridge.project(cross), yt, 10.0).numpy()
    np.testing.assert_allclose(pred, ridge_numpy(x, y, new, 10.0), rtol=1e-8, atol=1e-10)
    path = KernelRidgePath(k, yt, 'cpu')
    np.testing.assert_allclose(path.predict(path.project(cross), 10.0).numpy(), pred, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ridge.predict(ridge.project(cross), yt, 10.0, slice(2, 5)).numpy(), pred[:, 2:5],
                               rtol=1e-12, atol=1e-12)


def test_kernel_alpha_scores_equal_encoding_scores_on_the_factor_and_on_the_design():
    rng = np.random.default_rng(5)
    stories, lengths = ['a', 'b', 'c', 'd'], [23, 18, 27, 20]
    n = sum(lengths)
    x = rng.normal(size=(n, 150))  # Wider than tall, like the converters.
    y = x[:, :6] @ rng.normal(size=(6, 37)) + 2 * rng.normal(size=(n, 37))
    yt = torch.as_tensor(y)
    labels, folds = isp.row_groups(stories, lengths, 'half')
    k = torch.as_tensor(x @ x.T)
    r0, a0, c0 = encoding_scores(th.gram_factor(k).numpy(), yt, list(range(n)), labels, folds, th.ALPHAS, 'cpu')
    r1, a1, c1 = th.kernel_alpha_scores(k, yt, labels, folds, th.ALPHAS, chunk=8)  # Several output chunks.
    r2, a2, c2 = encoding_scores(x, yt, list(range(n)), labels, folds, th.ALPHAS, 'cpu')
    assert a0 == a1 == a2
    np.testing.assert_allclose(r1, r0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(c1, c0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(r1, r2, rtol=0, atol=1e-10)
    r3, a3, c3 = th.converter_alpha(k, yt, stories, lengths, th.ALPHAS)
    assert a3 == a1 and np.array_equal(r3, r1) and np.array_equal(c3, c1)
    # Every group of the per-row labels is held out once; a fold without fitting rows is refused.
    rows = th.fold_rows(labels, folds)
    assert len(rows) == 8 and sorted(int(i) for _, h in rows for i in h) == list(range(n))
    with pytest.raises(ValueError):
        th.kernel_alpha_scores(k, yt, labels, [tuple(dict.fromkeys(labels))], th.ALPHAS)
    with pytest.raises(ValueError):
        th.kernel_alpha_scores(k, yt, labels, folds, th.ALPHAS, precision='float16')
    # Mixed precision: the float64 recheck window holds the chosen alpha, its value and r are float64.
    info = {}
    r4, a4, c4 = th.kernel_alpha_scores(k, yt, labels, folds, th.ALPHAS, precision='mixed', info=info)
    best = int(np.flatnonzero(th.ALPHAS == a1)[0])
    assert a4 == a1 and info['precision'] == 'mixed' and {best - 1, best, best + 1} <= set(info['float64'])
    np.testing.assert_allclose(r4, r1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(c4[info['float64']], c1[info['float64']], rtol=0, atol=1e-12)
    np.testing.assert_allclose(c4, c1, rtol=0, atol=1e-5)
    np.testing.assert_allclose(info['float32_curve'], c1, rtol=0, atol=1e-5)


def test_mixed_precision_window_grows_until_the_float64_maximum_is_interior(monkeypatch):
    alphas = np.logspace(0, 6, 7)
    curves = {torch.float32: np.array([0.0, 0.1, 0.2, 0.5, 0.3, 0.2, 0.1]),  # float32 maximum at 3 ...
              torch.float64: np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])}  # ... float64 rises to the grid edge.

    def fake(kernel, y, rows, grid, device='cpu', chunk=th.CHUNK, dtype=torch.float64):
        idx = [int(np.flatnonzero(alphas == a)[0]) for a in grid]
        return torch.as_tensor(np.repeat(curves[dtype][idx][:, None], 3, axis=1))
    monkeypatch.setattr(th, 'alpha_r', fake)
    labels, folds = isp.row_groups(['a', 'b'], [4, 4], 'half')
    call = lambda: th.kernel_alpha_scores(torch.eye(8), torch.zeros(8, 3), labels, folds, alphas, precision='mixed',
                                          info=info)
    info = {}
    _, alpha, curve = call()
    assert alpha == alphas[6] and info['float64'] == [2, 3, 4, 5, 6] and info['float32_choice'] == 3
    np.testing.assert_allclose(curve, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], rtol=0, atol=1e-15)
    curves[torch.float64] = np.array([0.0, 0.1, 0.25, 0.3, 0.3, 0.2, 0.1])  # A float64 tie: the larger alpha.
    info = {}
    _, alpha, _ = call()
    assert alpha == alphas[4] and info['float64'] == [2, 3, 4, 5]
    curves[torch.float64] = curves[torch.float32].copy()  # Interior at once: three float64 alphas.
    info = {}
    _, alpha, _ = call()
    assert alpha == alphas[3] and info['float64'] == [2, 3, 4]


def test_cost_model_counts_each_term():
    out = th.converter_flops([10, 7], 5, n_alphas=3)  # Half groups 5, 5, 3, 4 of N = 17 rows.
    groups = [(5, 12), (5, 12), (3, 14), (4, 13)]
    assert out['alpha_loop'] == sum(2 * 3 * h * f * 5 for h, f in groups)
    assert out['smoothers'] == sum(2 * 4 * h * f * f for h, f in groups)
    assert out['eigh'] == sum(th.EIGH_FLOP * f ** 3 for _, f in groups)
    assert out['total'] == out['alpha_loop'] + out['smoothers'] + out['eigh']
    calls = [[10], [7], [10, 7]]
    cost = th.thsel_flops(calls, 30, 5, 4, n_inputs=6, n_alphas=3)
    assert cost['alpha'] == 4 * sum(th.converter_flops(c, 5, 3)['total'] for c in calls)
    assert cost['gram'] == sum(4 * 2.0 * (n * n + 30 * n) * 6 for n in (10, 7, 17))
    assert cost['final'] == sum(4 * (th.EIGH_FLOP * n ** 3 + 2.0 * n * n * 5 + 2.0 * 30 * n * 5) for n in (10, 7, 17))
    shared = th.thsel_flops(calls, 30, 5, 4, n_inputs=6, n_alphas=3, alpha_refs=2)  # alpha_memo: two alphas fit.
    assert shared['alpha'] == cost['alpha'] / 2 and shared['final'] == cost['final']


# ---------------------------------------------------------------------------------------------- TH_SEL
def test_thsel_scores_match_a_direct_numpy_pipeline(lpp):
    bold, stories, unheard = lpp['bold'], ['1', '2', '3'], ['5', '6']
    target = {s: bold[TARGET][s] for s in stories}
    refs = {r: {s: bold[r][s][:, :8] for s in stories} for r in REFS + (THIRD,)}  # Three pairs.
    unh = {r: {u: bold[r][u][:, :8] for u in unheard} for r in refs}
    score, info = th.thsel_scores(target, refs, unh, stories, unheard, ALPHAS, chunk=5)
    y = np.concatenate([target[s] for s in stories]).astype(np.float64)
    labels, folds = isp.row_groups(stories, [len(target[s]) for s in stories], 'half')
    preds = {}
    for r in refs:
        x = lagged_runs([refs[r][s] for s in stories])
        _, alpha, _ = encoding_scores(x, torch.as_tensor(y), list(range(len(y))), labels, folds, ALPHAS, 'cpu')
        assert info['converters'][r]['alpha'] == alpha
        preds[r] = ridge_numpy(x, y, lagged_runs([unh[r][u] for u in unheard]), alpha)
    pairs = [(a, b) for i, a in enumerate(refs) for b in list(refs)[i + 1:]]
    direct = np.array([np.mean([np.corrcoef(preds[a][:, v], preds[b][:, v])[0, 1] for a, b in pairs])
                       for v in range(y.shape[1])])
    np.testing.assert_allclose(score, direct, rtol=1e-7, atol=1e-9)
    assert score[:10].mean() > score[10:].mean() + 0.3  # Stimulus-driven voxels agree across references.
    # alpha_memo: a reference's converter alpha depends on that reference and the target only, so another
    # reference set reuses it; the pair's score is the one computed without the memo.
    memo = {}
    _, first = th.thsel_scores(target, refs, unh, stories, unheard, ALPHAS, chunk=5, alpha_memo=memo)
    assert len(memo) == 3 and not any(c.get('memo') for c in first['converters'].values())
    pair = list(REFS)
    two, info2 = th.thsel_scores(target, {r: refs[r] for r in pair}, {r: unh[r] for r in pair}, stories, unheard,
                                 ALPHAS, chunk=5, alpha_memo=memo)
    assert len(memo) == 3 and all(info2['converters'][r]['memo'] for r in pair)
    assert all(info2['converters'][r]['alpha'] == info['converters'][r]['alpha'] for r in pair)
    plain, _ = th.thsel_scores(target, {r: refs[r] for r in pair}, {r: unh[r] for r in pair}, stories, unheard,
                               ALPHAS, chunk=5)
    np.testing.assert_array_equal(two, plain)
    moved = {r: {s: v[::-1].copy() for s, v in refs[r].items()} for r in pair}  # Other data: refit, new keys.
    _, info3 = th.thsel_scores(target, moved, {r: unh[r] for r in pair}, stories, unheard, ALPHAS, chunk=5,
                               alpha_memo=memo)
    assert len(memo) == 5 and not any(info3['converters'][r].get('memo') for r in pair)


def test_thsel_is_not_estimable_without_two_references_or_unheard_material(lpp):
    bold, stories = lpp['bold'], ['1', '2']
    target = {s: bold[TARGET][s] for s in stories}
    refs = {r: {s: bold[r][s] for s in stories} for r in REFS}
    unh = {r: {'5': bold[r]['5']} for r in REFS}
    with pytest.raises(th.NotEstimable):
        th.thsel_scores(target, {REFS[0]: refs[REFS[0]]}, {REFS[0]: unh[REFS[0]]}, stories, ['5'])
    with pytest.raises(th.NotEstimable):
        th.thsel_scores(target, refs, {r: {} for r in REFS}, stories, [])
    with pytest.raises(ValueError):
        th.thsel_scores(target, refs, {r: {'1': bold[r]['1']} for r in REFS}, stories, ['1'])
    short = {r: {s: v[:-1] for s, v in refs[r].items()} for r in REFS}
    with pytest.raises(ValueError):
        th.thsel_scores(target, short, unh, stories, ['5'])


# ------------------------------------------------------------------------------- reference models
def test_reference_model_matches_the_engine_and_round_trips(ref_models, lpp, tmp_path):
    ref = REFS[0]
    model = ref_models[ref]
    labeled = th.labeled_stimuli('SS', SECTIONS, held=[TEST])
    tgt, bd = delay_budget(ref, lpp['bold'][ref], lpp['feats'], labeled, split_stories={'ev': []})
    story_folds, session_folds = th.session_rule_folds(bd)
    assert len(session_folds) == 5 and sorted(s for f in story_folds for s in f) == labeled
    # Voxels: top 8 of the max over delays of the nested W1 encoding r (§12 session folds).
    rows = torch.arange(len(bd.rows))
    r = []
    for d in DELAYS:
        x = bd.raw[d].double()
        mean, std = column_stats(x, rows)
        r.append(encoding_scores(bd.y, (x - mean) / std, list(range(len(bd.rows))), bd.session, session_folds,
                                 tgt.alphas, 'cpu')[0])
    np.testing.assert_allclose(model.scores, np.max(np.stack(r), axis=0), atol=1e-12)
    assert np.array_equal(model.voxels, th.top_k(np.max(np.stack(r), axis=0), 8))
    # Decoder: the engine's selection and final fit, here in primal form.
    sub = th.budget_columns(bd, model.voxels)
    rec = th.Recorder()
    pri = v4.Priors(tgt, None, rec, tgt.device, ())
    selections, _, _ = v4.run_cv(tgt, sub, 'ALL', ['W1'], pri, rec, None, story_folds, session_folds, lambda e: None)
    assert selections['W1'] == model.decoder.selection
    rng = np.random.default_rng(3)
    ev = {d: rng.normal(size=(9, 8)).astype(np.float32) for d in DELAYS}
    sub.eval_raw = {'ev': {d: torch.as_tensor(ev[d]) for d in DELAYS}}
    preds, _ = v4.final_fit(tgt, sub, 'ALL', ['W1'], selections, ('ev',), pri, rec, None, session_folds)
    np.testing.assert_allclose(model.decoder.predict(ev), preds['W1'][0]['ev'], rtol=1e-8, atol=1e-10)
    assert set(model.reads) == {(ref, s, k) for s in labeled for k in ('brain', 'transcript')}
    assert {r['purpose'] for r in th.model_reads(model)} == {'reference_model'}
    model.save(tmp_path / 'm.npz')
    back = th.ReferenceModel.load(tmp_path / 'm.npz')
    assert back.reads == model.reads and back.stimuli == model.stimuli and np.array_equal(back.voxels, model.voxels)
    np.testing.assert_array_equal(back.decoder.predict(ev), model.decoder.predict(ev))
    assert back.summary()['decoder_selection'] == model.decoder.selection


# -------------------------------------------------------------------------------------------- XDEC
def hand_decoder(rng, n_voxels, dim):
    arrays = lambda f: {d: f(d) for d in DELAYS}
    return th.LinearDecoder('ref', np.arange(n_voxels), arrays(lambda d: rng.normal(size=n_voxels)),
                            arrays(lambda d: rng.uniform(0.5, 2, size=n_voxels)),
                            arrays(lambda d: rng.uniform(0, 1, size=n_voxels)),
                            arrays(lambda d: rng.normal(size=n_voxels) * 0.1),
                            arrays(lambda d: rng.normal(size=(n_voxels, dim))), rng.normal(size=dim),
                            np.ones(dim), dict(gamma=1.0, alpha=1.0))


def test_xdec_predict_matches_converter_then_decoder_composition(monkeypatch):
    rng = np.random.default_rng(4)
    stories, lengths = ['a', 'b', 'c'], [18, 14, 16]
    evals = {'v1': 20, 'v2': 18}
    target = {s: zscore(rng.normal(size=(n, 15))) for s, n in zip(stories, lengths)}
    eval_runs = {e: zscore(rng.normal(size=(n, 15))) for e, n in evals.items()}
    refs = {r: {s: target[s][:, :6] @ rng.normal(size=(6, 6)) + rng.normal(size=(n, 6)) for s, n in
                zip(stories, lengths)} for r in ('R1', 'R2')}
    decoders = {r: hand_decoder(rng, 6, 3) for r in refs}
    eval_stimulus = np.array(['v1', 'v2', 'v1', 'v2', 'v1', 'v2', 'v1'])
    eval_tr = np.array([0, 2, 3, 5, 7, 14, 16])
    pred, info = th.xdec_predict(target, refs, decoders, stories, eval_runs, eval_stimulus, eval_tr, ALPHAS, chunk=4)
    x = lagged_runs([target[s] for s in stories])
    labels, folds = isp.row_groups(stories, lengths, 'half')
    direct = np.zeros((len(eval_tr), 3))
    for r in refs:
        y = np.concatenate([refs[r][s] for s in stories])
        _, alpha, _ = encoding_scores(x, torch.as_tensor(y), list(range(len(y))), labels, folds, ALPHAS, 'cpu')
        assert info['converters'][r]['alpha'] == alpha
        for e in evals:
            pos = np.flatnonzero(eval_stimulus == e)
            series = ridge_numpy(x, y, isp.lagged(eval_runs[e].astype(np.float64)), alpha)
            direct[pos] += decoders[r].predict({d: series[eval_tr[pos] + d] for d in DELAYS}) / len(refs)
    np.testing.assert_allclose(pred, direct, rtol=1e-7, atol=1e-9)
    monkeypatch.setattr(th, 'XDEC_OUTPUTS', 6)  # One reference per shared alpha-curve pass: the same result.
    alone, info_alone = th.xdec_predict(target, refs, decoders, stories, eval_runs, eval_stimulus, eval_tr, ALPHAS,
                                        chunk=4)
    np.testing.assert_allclose(alone, pred, rtol=1e-12, atol=1e-12)
    assert {r: c['alpha'] for r, c in info_alone['converters'].items()} == {
        r: c['alpha'] for r, c in info['converters'].items()}
    with pytest.raises(ValueError):  # Delayed rows must stay inside the evaluation run.
        th.xdec_predict(target, refs, decoders, stories, eval_runs, ['v2'], [15], ALPHAS)
    with pytest.raises(ValueError):  # Evaluation material may not be budget material.
        th.xdec_predict(target, refs, decoders, stories, dict(eval_runs, a=target['a']), ['a'], [0], ALPHAS)
    mean, std = th.scoring_stats(decoders)
    np.testing.assert_allclose(mean, (decoders['R1'].y_mean + decoders['R2'].y_mean) / 2)


# ------------------------------------------------------------------------- LPP (SS) through the engine
def test_lpp_thsel_through_the_engine_passes_provenance_and_the_registered_audit(lpp, ref_models, tmp_path):
    budget = bud.lpp_section_budgets(TEST, SECTIONS)['L3']
    folds = [tuple(f) for f in bud.inner_folds('L3', budget)]
    unheard = th.lpp_unheard(SECTIONS, budget, TEST)
    tgt, bd = lpp_target(lpp, budget)
    k = th.goal_size(24, 'lpp')
    prov = lpp_prov('TH_SEL', budget, folds)
    for ref in REFS:
        th.log_reads(prov, th.model_reads(ref_models[ref]))
    ref_bold = lambda ref, s: lpp['bold'][ref][s][:, ref_models[ref].voxels]
    selector = th.ThSelector(TARGET, lambda s: lpp['bold'][TARGET][s], REFS, ref_bold, unheard, k, prov, ALPHAS)
    pri = th.selection_priors(tgt, prov, 'cpu', selector)
    selections, curves, logs = v4.run_cv(tgt, bd, 'L3', [th.ENGINE_ARM], pri, prov, None, folds, None,
                                         lambda e: None)
    predictions, final = v4.final_fit(tgt, bd, 'L3', [th.ENGINE_ARM], selections, ('test',), pri, prov, None, None)
    assert th.ENGINE_ARM in predictions and predictions[th.ENGINE_ARM][0]['test'].shape[1] == DIM
    flags = prov.check()
    assert any(f['rule'] == 'R4' for f in flags)  # SS reference transcripts are flagged.
    assert np.isclose(final['weights'][th.ENGINE_ARM]['nonzero_fraction'], k / 24)
    assert final['isp']['stories'] == budget and final['isp']['k'] == k
    curve = curves[th.ENGINE_ARM]
    assert curve['0.5'] == curve['1'] == curve['2']  # A binary mask: the gamma rules tie.
    for i, held in enumerate(folds):  # Converters were refit inside every inner fold from its fitting part.
        fit = [s for s in budget if s not in held]
        assert prov.uncovered('brain', 'converter', fit, fold=i) == []
        assert [i_.stimulus for i_ in prov.uncovered('brain', 'converter', list(held), fold=i)] == list(held)
        assert logs[i]['priors']['isp']['stories'] == fit
    score, _ = th.thsel_scores({s: lpp['bold'][TARGET][s] for s in budget},
                               {r: {s: ref_bold(r, s) for s in budget} for r in REFS},
                               {r: {u: ref_bold(r, u) for u in unheard} for r in REFS}, budget, unheard, ALPHAS)
    assert final['isp']['threshold'] == float(score[th.selection_mask(score, k)].min())
    cell = cell_of('TH_SEL')
    assert th.audit_errors(cell, prov) == [] and audit_v4.scope_errors(cell, prov) == []
    path = tmp_path / 't4_L3_TH_SEL.json'
    path.write_text(json.dumps(dict(cell, provenance=prov.to_dict())))
    assert th.audit_file(path) == audit_v4.audit_file(path) and th.audit_file(path)['passed']
    back = Provenance.from_dict(prov.to_dict())
    assert back.audit()['violations'] == []
    # Planted leak: one selection computed (untagged) on all of B and reused in every inner fold. R1-R7 and the
    # frozen audit pass; Module H's scope audit must not.
    hoist = lpp_prov('TH_SEL', budget, folds)
    for ref in REFS:
        th.log_reads(hoist, th.model_reads(ref_models[ref]))
    mask, info = th.ThSelector(TARGET, lambda s: lpp['bold'][TARGET][s], REFS, ref_bold, unheard, k, hoist,
                               ALPHAS)(budget)
    pri = th.selection_priors(tgt, hoist, 'cpu', lambda stories, misaligned=False: (mask, dict(info)))
    sel_h, _, _ = v4.run_cv(tgt, bd, 'L3', [th.ENGINE_ARM], pri, hoist, None, folds, None, lambda e: None)
    v4.final_fit(tgt, bd, 'L3', [th.ENGINE_ARM], sel_h, ('test',), pri, hoist, None, None)
    hoist.check()
    assert audit_v4.scope_errors(cell, hoist) == []  # The frozen audit cannot see it ...
    errors = th.audit_errors(cell, hoist)
    converter = [e for e in errors if 'TH_SEL selection: no tagged' in e]
    material = [e for e in errors if 'selection material [] is not the registered' in e]
    assert len(converter) == len(folds) * 3 and len(material) == len(folds) * 2, errors  # 3 subjects, 2 references
    assert len(errors) == len(converter) + len(material)
    path = tmp_path / 't4_L3_TH_SEL_hoisted.json'
    path.write_text(json.dumps(dict(cell, provenance=hoist.to_dict())))
    assert audit_v4.audit_file(path)['passed'] and not th.audit_file(path)['passed']  # ... Module H's does.
    # A selection without its final refit on the whole budget, or with a single reference, fails too.
    partial = lpp_prov('TH_SEL', budget, folds)
    for read, n in prov.records.items():
        if read.fold is not None or read.purpose != 'converter':
            partial.records[read] = n
    assert any(e.startswith('final TH_SEL selection: no untagged target') for e in th.audit_errors(cell, partial))
    alone = lpp_prov('TH_SEL', budget, folds)
    alone.records = {r: n for r, n in prov.records.items() if r.subject != REFS[1]}
    assert any('at least two references' in e for e in th.audit_errors(cell, alone))
    assert any('not the log reference models' in e for e in th.audit_errors(dict(cell, references=[REFS[0]]), prov))
    # The registered selection material (lpp_unheard) in every inner fold and in the cell's 'unheard' field.
    assert th.registered_selection(prov) == sorted(unheard)
    assert th.audit_errors(dict(cell, unheard=unheard[::-1]), prov) == []
    assert th.audit_errors(dict(cell, unheard=unheard[:-1]), prov) == [
        f"cell unheard {unheard[:-1]} is not the registered selection material {sorted(unheard)}"]
    heard = lpp_prov('TH_SEL', budget, folds)  # A budget section as selection material inside inner fold 0.
    heard.records = dict(prov.records)
    with heard.inner_fold(0):
        th.log_reads(heard, [th.read(REFS[0], budget[-1], 'brain', 'reference_model')])
    heard.check()
    assert audit_v4.scope_errors(cell, heard) == []
    assert th.audit_errors(cell, heard) == [f"inner fold 0: reference {REFS[0]} selection material "
                                            f"{sorted(unheard + [budget[-1]])} is not the registered {sorted(unheard)}"]
    full = bud.lpp_section_budgets(TEST, SECTIONS)['L8']  # No selection material at L8: no TH_SEL cell.
    l8 = Provenance(TARGET, 'S', full, [], [TEST], reference_rule='SS', arm='TH_SEL', budget_name='L8',
                    stage='develop')
    assert th.registered_selection(l8) == [] and th.material_errors(dict(arm='TH_SEL'), l8) == [
        'TH_SEL has no registered selection material in this cell (not estimable)']
    # Planted leak: selection material containing the outer test section.
    leak = lpp_prov('TH_SEL', budget, folds)
    th.ThSelector(TARGET, lambda s: lpp['bold'][TARGET][s], REFS, ref_bold, unheard + [TEST], k, leak, ALPHAS)(budget)
    with pytest.raises(ProvenanceError) as error:
        leak.check()
    assert 'R3' in error.value.rules
    # Planted leak: a reference model whose labels include the target's test section.
    leak = lpp_prov('TH_SEL', budget, folds)
    th.log_reads(leak, [th.read(REFS[0], TEST, k_, 'reference_model') for k_ in ('brain', 'transcript')])
    with pytest.raises(ProvenanceError) as error:
        leak.check()
    assert {'R3', 'R4', 'R5'} <= set(error.value.rules)


def test_lpp_thsel_rand_reads_nothing_from_references(lpp):
    budget = bud.lpp_section_budgets(TEST, SECTIONS)['L3']
    folds = [tuple(f) for f in bud.inner_folds('L3', budget)]
    tgt, bd = lpp_target(lpp, budget)
    prov = Provenance(TARGET, 'S', budget, [], [TEST], folds=folds, arm='TH_SEL_rand', budget_name='L3',
                      stage='develop')
    for s in budget + [TEST]:
        prov.record(TARGET, s, 'transcript', 'context_reset')
    pri = th.selection_priors(tgt, prov, 'cpu', th.RandomSelector(24, 4, TARGET))
    selections, _, _ = v4.run_cv(tgt, bd, 'L3', [th.ENGINE_ARM], pri, prov, None, folds, None, lambda e: None)
    _, final = v4.final_fit(tgt, bd, 'L3', [th.ENGINE_ARM], selections, ('test',), pri, prov, None, None)
    prov.check()
    assert {r.subject for r in prov.records} == {TARGET} and prov.reference_rule == 'none'
    assert np.isclose(final['weights'][th.ENGINE_ARM]['nonzero_fraction'], 4 / 24)
    assert th.audit_errors(cell_of('TH_SEL_rand', rule='none'), prov) == []
    prov.record(REFS[0], budget[0], 'brain', 'converter')  # Allowed by R3, but TH_SEL_rand reads no reference.
    prov.check()
    assert th.audit_errors(cell_of('TH_SEL_rand', rule='none'), prov) == [f"TH_SEL_rand read data of ['{REFS[0]}']"]
    with pytest.raises(ValueError):  # TH_SEL_rand is no secondary arm: it may not carry a reference rule.
        Provenance(TARGET, 'S', budget, [], [TEST], reference_rule='SS', arm='TH_SEL_rand')


def test_lpp_xdec_predictions_provenance_and_audit(lpp, ref_models, tmp_path):
    budget = bud.lpp_section_budgets(TEST, SECTIONS)['L3']
    bold = lpp['bold']
    trs = np.arange(0, N_TR[TEST] - 4, 2)
    target_runs = {s: bold[TARGET][s] for s in budget}
    reference_runs = {r: {s: bold[r][s][:, ref_models[r].voxels] for s in budget} for r in REFS}
    decoders = {r: ref_models[r].decoder for r in REFS}
    pred, info = th.xdec_predict(target_runs, reference_runs, decoders, budget, {TEST: bold[TARGET][TEST]},
                                 [TEST] * len(trs), trs, ALPHAS)
    truth = lpp['feats'][TEST][trs]
    assert pred.shape == truth.shape and np.isfinite(pred).all()
    r = [np.corrcoef(pred[:, j], truth[:, j])[0, 1] for j in range(DIM)]
    assert np.mean(r) > 0.2  # Zero target labels, yet stimulus-locked through the references.
    prov = lpp_prov('XDEC', budget)
    for ref in REFS:
        th.log_reads(prov, th.model_reads(ref_models[ref]))
    th.log_reads(prov, th.xdec_reads(TARGET, REFS, list(target_runs), [TEST]))
    prov.check()
    cell = cell_of('XDEC')
    assert th.audit_errors(cell, prov) == []
    frozen = audit_v4.scope_errors(cell, prov)  # The frozen audit assumes a target-trained decoder.
    assert frozen and all(e.startswith(th.XDEC_EXEMPT) for e in frozen)
    path = tmp_path / 't4_L3_XDEC.json'
    path.write_text(json.dumps(dict(cell, provenance=prov.to_dict())))
    assert th.audit_file(path)['passed'] and not audit_v4.audit_file(path)['passed']
    leak = lpp_prov('XDEC', budget)
    th.log_reads(leak, th.xdec_reads(TARGET, REFS, budget, [TEST]))
    leak.record(TARGET, budget[0], 'transcript', 'decoder_fit')
    assert any('beyond scoring' in e for e in th.audit_errors(cell, leak))
    # Planted leak: target labels in B read as 'decoder_score' (calibration/tuning). R1-R7 allow it (in budget).
    for ref in REFS:
        th.log_reads(prov, th.model_reads(ref_models[ref]))
    tuned = lpp_prov('XDEC', budget)
    tuned.records = dict(prov.records)
    for s in budget:
        tuned.record(TARGET, s, 'transcript', 'decoder_score')
    tuned.check()
    assert [e for e in th.audit_errors(cell, tuned) if 'beyond scoring the scored items' in e] == [
        f'XDEC read target transcripts/audio beyond scoring the scored items at {sorted(budget)}']
    # A target decoder fit on B BOLD, a ledger counting target labels, a missing ledger, or no reference model.
    fit = lpp_prov('XDEC', budget)
    fit.records = dict(prov.records)
    fit.record(TARGET, budget[0], 'brain', 'decoder_fit')
    assert any('beyond its converters and decoder input' in e for e in th.audit_errors(cell, fit))
    labeled = dict(cell, ledger=dict(cell['ledger'], target_pairs_dense=120))
    assert any('ledger counts target labels' in e for e in th.audit_errors(labeled, prov))
    assert any('needs its ledger' in e for e in th.audit_errors({k: v for k, v in cell.items() if k != 'ledger'},
                                                                 prov))
    assert any('target_labels' in e for e in th.audit_errors(dict(cell, target_labels=5), prov))
    assert th.audit_errors(dict(cell, references=list(REFS), target_labels=0), prov) == []
    assert any('not the log reference models' in e for e in th.audit_errors(dict(cell, references=[REFS[0]]), prov))
    bare = lpp_prov('XDEC', budget)
    bare.records = {r: n for r, n in prov.records.items() if r.purpose != 'reference_model'}
    assert any('no reference model' in e for e in th.audit_errors(cell, bare))
    no_ref_converter = lpp_prov('XDEC', budget)
    no_ref_converter.records = {r: n for r, n in prov.records.items()
                                if not (r.subject == REFS[0] and r.purpose == 'converter')}
    assert any(f'reference {REFS[0]} brain converter' in e for e in th.audit_errors(cell, no_ref_converter))
    outside = th.lpp_unheard(SECTIONS, budget, TEST)[0]
    leak = lpp_prov('XDEC', budget)
    th.log_reads(leak, th.xdec_reads(TARGET, REFS, budget + [outside], [TEST]))
    with pytest.raises(ProvenanceError) as error:
        leak.check()
    assert {'R2', 'R3'} <= set(error.value.rules)


def test_ds003020_ts_rule_confines_reference_transcripts_to_unheard_stories():
    budget, folds = ['a', 'b', 'c'], [('a',), ('b',), ('c',)]
    prov = Provenance('UTS05', 'S', budget, ['v1'], ['t1'], ts_allowed=['x1', 'x2'], folds=folds, arm='TH_SEL',
                      budget_name='S3', stage='develop')
    th.log_reads(prov, [th.read('UTS01', s, k, 'reference_model') for s in ('x1', 'x2') for k in ('brain',
                                                                                                    'transcript')])
    with prov.inner_fold(0):
        th.log_reads(prov, th.thsel_reads('UTS05', ['UTS01', 'UTS02'], ['b', 'c'], ['x1']))
    prov.check()
    prov.record('UTS01', 'a', 'transcript', 'reference_model')  # A story the target heard (its budget).
    with pytest.raises(ProvenanceError) as error:
        prov.check()
    assert error.value.rules == ['R4']
    leaky = Provenance('UTS05', 'S', budget, ['v1'], ['t1'], ts_allowed=['x1'], folds=folds, arm='TH_SEL')
    with leaky.inner_fold(0):  # A converter fit on the inner fold's held-out story.
        th.log_reads(leaky, th.thsel_reads('UTS05', ['UTS01', 'UTS02'], ['a', 'b'], ['x1']))
    with pytest.raises(ProvenanceError) as error:
        leaky.check()
    assert 'R7' in error.value.rules
    with pytest.raises(ValueError):  # TS stimuli must be unheard by the target.
        Provenance('UTS05', 'S', budget, ['v1'], ['t1'], ts_allowed=['a'], arm='TH_SEL')
    reads = th.xdec_reads('UTS05', ['UTS01'], budget, ['v1'])
    assert {(r['kind'], r['purpose']) for r in reads if r['stimulus'] == 'v1'} == {
        ('brain', 'decoder_score'), ('transcript', 'decoder_score')}  # No context-reset record in ds003020.


DS_REFS = ['UTS01', 'UTS02', 'UTS03']


def ds_manifests(root):
    """JSON-only manifests of UTS01-UTS10 (enough for the audits): shared training stories sh1-sh6, validation
    va1/va2, test te1; UTS01-UTS03 also heard x1-x4 (x4 in UTS01's validation split) and UTS05 also o1."""
    for subject in bud.SUBJECTS:
        stories = [(f'sh{i}', 'train') for i in range(1, 7)] + [('va1', 'val'), ('va2', 'val'), ('te1', 'test')]
        if subject in DS_REFS:
            stories += [(f'x{i}', 'val' if (subject, i) == ('UTS01', 4) else 'train') for i in range(1, 5)]
        if subject == 'UTS05':
            stories.append(('o1', 'train'))
        path = root / subject / 'semantic_manifest.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(subject=subject, runs=[dict(story=s, split=sp, session=f'ses-{s}', n_tr=300)
                                                                   for s, sp in stories])))
    return root


def ds_log(root, arm='TH_SEL', rule='TS', material=None, extra=()):
    """A complete UTS05 S3 develop log of TH_SEL (or XDEC) with references UTS01-UTS03 under ``rule``; reference
    models read ``extra`` stories too and the selections read ``material`` (default: the registered one)."""
    ms = audit_v4.manifests(str(root))
    budget = bud.ds003020_budgets(ms['UTS05'], bud.shared_stories(ms))['S3']
    ts = sorted(set().union(*(bud.ts_stories(ms['UTS05'], ms[r]) for r in DS_REFS)))
    material = th.ds003020_selection(ms, 'UTS05', DS_REFS) if material is None else material
    folds = [tuple(f) for f in bud.inner_folds('S3', budget)] if arm == 'TH_SEL' else None
    p = Provenance('UTS05', 'S', budget, ['va1', 'va2'], ['te1'], ts_allowed=ts if rule == 'TS' else None,
                   reference_rule=rule, folds=folds, arm=arm, budget_name='S3', stage='develop')
    th.log_reads(p, [th.read(r, s, k, 'reference_model') for r in DS_REFS for s in ts + list(extra)
                     for k in ('brain', 'transcript')])
    if arm == 'XDEC':
        th.log_reads(p, th.xdec_reads('UTS05', DS_REFS, budget, ['va1', 'va2']))
        return p
    for i, held in enumerate(folds):
        with p.inner_fold(i):
            fit = [s for s in budget if s not in held]
            for kind, purpose, _ in audit_v4.fold_reads(arm):
                for s in fit:
                    p.record('UTS05', s, kind, purpose)
            th.log_reads(p, th.thsel_reads('UTS05', DS_REFS, fit, material))
            for s in held:
                p.record('UTS05', s, 'brain', 'decoder_score')
                p.record('UTS05', s, 'transcript', 'decoder_score')
    th.log_reads(p, th.thsel_reads('UTS05', DS_REFS, budget, material))
    for s in budget:
        p.record('UTS05', s, 'brain', 'decoder_fit')
        p.record('UTS05', s, 'transcript', 'decoder_fit')
    for s in ('va1', 'va2'):
        p.record('UTS05', s, 'brain', 'decoder_score')
        p.record('UTS05', s, 'transcript', 'decoder_score')
    return p


def test_registered_rule_and_selection_material_are_audited_in_ds003020(tmp_path):
    root = ds_manifests(tmp_path / 'semantic')
    ms = audit_v4.manifests(str(root))
    assert th.ds003020_selection(ms, 'UTS05', DS_REFS) == ['x1', 'x2', 'x3', 'x4']  # Any reference split.
    heard = bud.ds003020_budgets(ms['UTS05'], bud.shared_stories(ms))['S6'][-1]  # Outside S3, heard in training.
    cell = lambda rule='TS', **kw: dict(dict(subject='UTS05', stage='develop', arm='TH_SEL', budget='S3',
                                             references=DS_REFS, selection_material=['x1', 'x2', 'x3', 'x4'],
                                             ledger=dict(reference_rule=rule)), **kw)
    clean = ds_log(root)
    clean.check()
    assert th.audit_errors(cell(), clean, root) == [] and audit_v4.scope_errors(cell(), clean, root) == []
    # The reviewer's probe: rule SS in ds003020, reference models also on a story the target heard (flagged by
    # R4 only); R1-R7 and the frozen audit pass, Module H's audit does not.
    probe = ds_log(root, rule='SS', extra=[heard])
    assert {f['rule'] for f in probe.check()} == {'R4'} and audit_v4.scope_errors(cell('SS'), probe, root) == []
    errors = th.audit_errors(cell('SS'), probe, root)
    assert any('carries reference rule SS; the registered rule is TS' in e for e in errors)
    assert any(f"reference-model reads on stories the target heard: [('UTS01', '{heard}')" in e for e in errors)
    # Selection material the target heard (TS rule, BOLD only: R3 allows it as reference-model material).
    leak = ds_log(root, material=['x1', 'x2', 'x3', 'x4', heard])
    leak.check()
    assert audit_v4.scope_errors(cell(), leak, root) == []
    errors = th.audit_errors(cell(selection_material=['x1', 'x2', 'x3', 'x4', heard]), leak, root)
    assert any('stories the target heard' in e for e in errors)
    assert sum('selection material' in e and 'is not the registered' in e for e in errors) == 3 * 3 + 1
    # Material short of the registered set, and a cell that misstates it.
    short = ds_log(root, material=['x1', 'x2'])
    assert sum('is not the registered' in e for e in th.audit_errors(cell(), short, root)) == 3 * 3
    assert th.audit_errors(cell(selection_material=['x1']), clean, root) == [
        "cell selection_material ['x1'] is not the registered selection material ['x1', 'x2', 'x3', 'x4']"]
    # XDEC carries the registered rule too, and its reference models never read heard stories.
    xcell = dict(cell(), arm='XDEC', target_labels=0, selection_material=None,
                 ledger=dict(reference_rule='TS', target_pairs_dense=0, target_pairs_stride4=0, transcript_words=0))
    assert th.audit_errors(xcell, ds_log(root, 'XDEC'), root) == []
    xss = dict(xcell, ledger=dict(xcell['ledger'], reference_rule='SS'))
    errors = th.audit_errors(xss, ds_log(root, 'XDEC', rule='SS', extra=[heard]), root)
    assert any('rule SS; the registered rule is TS' in e for e in errors) and any('target heard' in e for e in errors)


# -------------------------------------------------------------------------------------------- script
def test_script_cell_names_ts_material_and_ledger(tmp_path):
    assert rt.cell_path('UTS05', 'dense', 'develop', 'S3', 'TH_SEL') == Path(
        'results/v4/UTS05/dense/develop/cells/S3_TH_SEL.json')
    two = ('UTS01', 'UTS02')  # The two-reference sensitivity stays out of the locked cells/ folder, for any arm.
    assert rt.cell_path('UTS05', 'dense', 'develop', 'S6', 'XDEC', two) == Path(
        'results/v4/UTS05/dense/develop/sensitivity/refs-UTS01-UTS02/cells/S6_XDEC.json')
    assert rt.cell_path('UTS05', 'stride4', 'evaluate', 'S6', 'TH_SEL_rand', two) == Path(
        'results/v4/UTS05/stride4/evaluate/sensitivity/refs-UTS01-UTS02/cells/S6_TH_SEL_rand.json')
    # The registered sensitivity: every pair of UTS01-UTS03, TH_SEL and XDEC only.
    assert rt.SENSITIVITY_PAIRS == (('UTS01', 'UTS02'), ('UTS01', 'UTS03'), ('UTS02', 'UTS03'))
    assert rt.check_refs(rt.REFERENCES, th.ARMS) == rt.REFERENCES
    assert rt.check_refs(['UTS02', 'UTS03'], ['TH_SEL', 'XDEC']) == ('UTS02', 'UTS03')
    for refs, arms in ((['UTS01'], ['XDEC']), (['UTS02', 'UTS01'], ['XDEC']), (two, th.ARMS), (two, ['TH_SEL_rand'])):
        with pytest.raises(SystemExit):
            rt.check_refs(refs, arms)
    # TS material: a reference's labeled stories are all its TS stories, in any split (x3 is UTS01's validation
    # story, x4 UTS02's test story); the selection material is every TS story all references heard.
    runs = lambda stories, held=(): dict(runs=[dict(story=s, n_tr=300 + i, split=dict(held).get(s, 'train'))
                                               for i, s in enumerate(stories)])
    manifests = dict(UTS05=runs(['a', 'b', 'v1', 't1'], [('v1', 'val'), ('t1', 'test')]),
                     UTS01=runs(['a', 'b', 'x1', 'x2', 'x3', 'x4'], [('b', 'val'), ('x3', 'val')]),
                     UTS02=runs(['a', 'x1', 'x2', 'x3', 'x4'], [('x4', 'test')]))
    by_ref, union, common = rt.ts_material(manifests, 'UTS05', ['UTS01', 'UTS02'])
    assert by_ref == dict(UTS01=['x1', 'x2', 'x3', 'x4'], UTS02=['x1', 'x2', 'x3', 'x4'])
    assert union == ['x1', 'x2', 'x3', 'x4'] and common == ['x1', 'x2', 'x3', 'x4']
    models = {r: SimpleNamespace(stimuli=['x1', 'x2'], info=dict(labeled_rows=100)) for r in by_ref}
    base = bud.Ledger().target(10, 3, 30.0).to_dict()
    ledger = rt.reference_ledger(base, manifests, ['UTS01', 'UTS02'], models, ['a'])
    assert ledger['reference_rule'] == 'TS' and ledger['reference_labeled_pairs'] == 200
    assert ledger['references'] == ['UTS01', 'UTS02'] and ledger['target_pairs_dense'] == 10
    # Only the stories read: converter story a and the labeled stories (no selection material here) ...
    expected = bud.bold_minutes([300, 302, 303]) + bud.bold_minutes([300, 301, 302])
    assert np.isclose(ledger['reference_bold_minutes'], expected)
    # ... plus the selection material (x3 and x4: BOLD only).
    ledger = rt.reference_ledger(base, manifests, ['UTS01', 'UTS02'], models, ['a'], common)
    expected = bud.bold_minutes([300, 302, 303, 304, 305]) + bud.bold_minutes([300, 301, 302, 303, 304])
    assert np.isclose(ledger['reference_bold_minutes'], expected)
    # After the lock, an existing develop cell is never rewritten (evaluate cells and new files are not blocked).
    lock, cell = tmp_path / 'lock.json', tmp_path / 'S3_TH_SEL_rand.json'
    rt.check_writable(cell, 'develop', lock)
    cell.write_text('{}')
    rt.check_writable(cell, 'develop', lock)
    lock.write_text('{}')
    rt.check_writable(cell, 'evaluate', lock)
    rt.check_writable(tmp_path / 'new.json', 'develop', lock)
    with pytest.raises(SystemExit):
        rt.check_writable(cell, 'develop', lock)
    data = np.arange(12, dtype=np.float32).reshape(4, 3)
    data[1, 2] = np.nan
    (tmp_path / 'UTS01').mkdir()
    with h5py.File(tmp_path / 'UTS01' / 'a.hf5', 'w') as handle:
        handle['data'] = data
    raw = rt.load_raw(tmp_path, dict(source_key='UTS01/a.hf5', n_tr=4))
    assert raw[1, 2] == 0 and raw[3, 1] == data[3, 1]
    with pytest.raises(ValueError):
        rt.load_raw(tmp_path, dict(source_key='UTS01/a.hf5', n_tr=5))
    with pytest.raises(SystemExit):  # No free reference subset: only the registered pairs (refused before any data).
        rt.main(['--stage', 'develop', '--subject', 'UTS05', '--references', 'UTS01', 'UTS02'])
    with pytest.raises(SystemExit):  # The sensitivity does not rerun TH_SEL_rand.
        rt.main(['--stage', 'develop', '--subject', 'UTS05', '--sensitivity', '--arms', 'TH_SEL', 'TH_SEL_rand'])
    with pytest.raises(SystemExit):  # UTS01-UTS03 are references, never targets of this block.
        rt.main(['--stage', 'develop', '--subject', 'UTS01'])
    with pytest.raises(SystemExit):
        rt.check_lock(tmp_path / 'missing.json')


def test_code_fingerprints_cover_module_script_and_core(tmp_path):
    digest = hashlib.sha256()
    for name in ('src/priors/tanghuth.py', 'scripts/run_v4_tanghuth.py'):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    assert th.code_sha256() == digest.hexdigest()
    assert rt.code_sha256() == hashlib.sha256((v4.code_sha256() + th.code_sha256()).encode()).hexdigest()
    assert 'src/priors/tanghuth.py' not in v4.CORE and 'scripts/run_v4_tanghuth.py' not in v4.CORE
    lock = dict(code_sha256=v4.code_sha256(), tanghuth_sha256=v4.file_sha('src/priors/tanghuth.py'),
                tanghuth_cell_code_sha256=rt.code_sha256())
    (tmp_path / 'lock.json').write_text(json.dumps(lock))
    rt.check_lock(tmp_path / 'lock.json')  # The fields scripts/v4_lock.py pins for this block.
    for key in lock:
        (tmp_path / 'bad.json').write_text(json.dumps(dict(lock, **{key: '0' * 64})))
        with pytest.raises(SystemExit):
            rt.check_lock(tmp_path / 'bad.json')


# ------------------------------------------------------------------- script end to end (synthetic ds003020)
PARSER = ROOT / 'reference/semantic-decoding/decoding/utils_ridge/textgrid.py'


def textgrid(intervals):
    end = max(b for _, b, _ in intervals)
    head = (f'File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = {end}\ntiers? <exists>\n'
            f'size = 1\nitem []:\n    item [1]:\n        class = "IntervalTier"\n        name = "word"\n'
            f'        xmin = 0\n        xmax = {end}\n        intervals: size = {len(intervals)}\n')
    return head + ''.join(f'        intervals [{i}]:\n            xmin = {a}\n            xmax = {b}\n'
                          f'            text = "{w}"\n' for i, (a, b, w) in enumerate(intervals, 1))


REF_HELD = {('UTS01', 'x6'): 'val', ('UTS02', 'x5'): 'test'}  # TS stories in a reference's val/test split.


def build_ds003020(root):
    """Ten subjects on shared stories sh1-sh6 (train), va1-va2 (val), te1 (test); UTS01-UTS03 also heard x1-x6
    (the TS stories), all in their training split except ``REF_HELD`` (as UTS01-03's own validation and test
    stories are TS stories of UTS04-10). Each run is its own session; caches follow run_v4's layout (window 8)."""
    from src.targets import TargetIndex
    from src.voxel_kernels import cache_delay_rows
    rng = np.random.default_rng(11)
    shared, val, test, ts = [f'sh{i}' for i in range(1, 7)], ['va1', 'va2'], ['te1'], [f'x{i}' for i in range(1, 7)]
    n_tr = {s: 40 + 2 * i for i, s in enumerate(shared + val + test + ts)}
    latent = {s: rng.normal(size=(n, 3)) for s, n in n_tr.items()}
    mix = rng.normal(size=(3, DIM))
    stim, raw_root = root / 'stimuli', root / 'data' / 'raw'
    stim.mkdir(parents=True)
    for s, n in n_tr.items():
        np.save(stim / f'{s}.npy', (latent[s] @ mix + 0.3 * rng.normal(size=(n, DIM))).astype(np.float32))
        np.save(stim / f'{s}.valid.npy', np.ones(n, dtype=bool))
        (stim / f'{s}.TextGrid').write_text(textgrid([(11 + 2 * t - 0.4, 11 + 2 * t + 0.4, f'w{t}') for t in
                                                      range(n)]))
    for subject in bud.SUBJECTS:
        w = rng.normal(size=(3, 30))
        w[:, 20:] = 0
        runs = []
        for s in shared + val + test + (ts if subject in rt.REFERENCES else []):
            lagged = np.zeros_like(latent[s])
            lagged[2:] = latent[s][:-2]
            x = zscore(1.5 * lagged @ w + rng.normal(size=(n_tr[s], 30)))
            (raw_root / subject).mkdir(parents=True, exist_ok=True)
            with h5py.File(raw_root / subject / f'{s}.hf5', 'w') as handle:
                handle['data'] = x
            np.save(raw_root / subject / f'{s}.npy', x)
            runs.append(dict(run_id=f'{subject}-{s}', dataset='ds003020', subject=subject, story=s,
                             session=f'sess-{s}', split='val' if s in val else 'test' if s in test else
                             REF_HELD.get((subject, s), 'train'),
                             n_tr=n_tr[s], n_features=30, neural_path=str(raw_root / subject / f'{s}.npy'),
                             target_path=str(stim / f'{s}.npy'), target_valid_path=str(stim / f'{s}.valid.npy'),
                             source_key=f'{subject}/{s}.hf5', stimulus_path=str(stim / f'{s}.TextGrid'),
                             tr_seconds=2.0, tr_start_seconds=11.0))
        path = root / 'data' / 'semantic' / subject / 'semantic_manifest.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(subject=subject, runs=runs)))
        common = dict(window=8, phase=0, cutoffs=(3, 4))
        cache_delay_rows(TargetIndex(path, 'train', stride=1, **common), raw_root,
                         root / 'data' / 'cache' / 'v3_dense' / subject)
        for split in ('train', 'val', 'test'):
            cache_delay_rows(TargetIndex(path, split, stride=4, **common), raw_root,
                             root / 'data' / 'cache' / 'v3' / subject)
    return dict(protocol='ssltang-v3', data=dict(manifest='data/semantic/UTS05/semantic_manifest.json', window=8,
                                                 stride=4, target_phase=0, cutoffs=[3, 4]),
                alphas=[-1, 7, 17], raw_root=str(raw_root), cache_root='data/cache/v3',
                evaluation=dict(exclusions=[10, 32], shift_min_tr=40, shift_draws=50, shift_seed=0))


@pytest.mark.skipif(not PARSER.is_file(), reason='Tang/Huth TextGrid parser not checked out')
def test_script_runs_the_published_method_block_on_a_synthetic_ds003020_tree(tmp_path, monkeypatch):
    cfg = build_ds003020(tmp_path)
    manifests = tmp_path / 'data' / 'semantic'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(th, 'REF_TOP', 10)
    monkeypatch.setattr(th, 'GOAL_TOP', 12)
    audited = th.audit_errors
    monkeypatch.setattr(th, 'audit_errors', lambda cell, p, root=None: audited(cell, p, manifests))
    rt.stage_fit(cfg, 'UTS05', 'dense', ['S3', 'S6'], list(th.ARMS), 'develop', 'cpu')
    cells = {}
    for name in ('S3', 'S6'):
        for arm in th.ARMS:
            path = rt.cell_path('UTS05', 'dense', 'develop', name, arm)
            cells[name, arm] = json.loads(path.read_text())
            assert th.audit_file(path, manifests)['passed'], th.audit_file(path, manifests)
            assert np.isfinite(cells[name, arm]['val']['session_macro_feature_correlation'])
            assert cells[name, arm]['code_sha256'] == rt.code_sha256() and 'test' not in cells[name, arm]
    sel, rand, xdec = (cells['S3', a] for a in th.ARMS)
    assert np.isclose(sel['final_weights']['nonzero_fraction'], 12 / 30)
    assert np.isclose(rand['final_weights']['nonzero_fraction'], 12 / 30)
    assert sel['ledger']['reference_rule'] == 'TS' and sel['ledger']['references'] == list(rt.REFERENCES)
    assert rand['ledger']['reference_rule'] == 'none' and rand['ledger']['references'] == []
    assert xdec['ledger']['target_pairs_dense'] == 0 and xdec['ledger']['transcript_words'] == 0
    assert sel['ledger']['target_pairs_dense'] > 0 and sel['ledger']['reference_labeled_pairs'] > 0
    assert sel['provenance']['ts_allowed'] == [dict(subject=None, stimulus=f'x{i}', trs=None) for i in range(1, 7)]
    # Labels from every TS story of each reference, in any of its splits (x6 is UTS01's validation story, x5
    # UTS02's test story: their dense rows come from this block's own caches); the same selection material.
    labeled = {r: [f'x{i}' for i in range(1, 7)] for r in rt.REFERENCES}
    assert {r: m['stimuli'] for r, m in sel['reference_models'].items()} == {r: len(v) for r, v in labeled.items()}
    assert sel['reference_models']['UTS01']['splits'] == dict(train=['x1', 'x2', 'x3', 'x4', 'x5'], val=['x6'])
    assert sel['reference_models']['UTS02']['splits'] == dict(train=['x1', 'x2', 'x3', 'x4', 'x6'], test=['x5'])
    assert sel['reference_models']['UTS03']['splits'] == dict(train=labeled['UTS03'])
    assert sorted(p.name for p in (rt.ROWS / 'dense').glob('*/*.meta.json')) == ['test.meta.json', 'val.meta.json']
    n_tr = lambda s: 56 + 2 * int(s[1:])  # x1..x6 (build_ds003020); eligible TRs 4..n-5 (window 8, cutoffs 3/4).
    assert sel['reference_models']['UTS01']['labeled_rows'] == sum(n_tr(s) - 8 for s in labeled['UTS01'])
    assert sel['selection_material'] == [f'x{i}' for i in range(1, 7)]
    # Stride-4 clock: validation/test rows come from the verified v3 stride-4 caches.
    parts = rt.reference_rows(cfg, v4.Target(cfg, 'UTS01', 'stride4', 'cpu'), ['x1', 'x6'])
    assert [(sp, len(pos)) for sp, _, _, pos in parts] == [('train', len(range(4, n_tr('x1') - 4, 4))),
                                                           ('val', len(range(4, n_tr('x6') - 4, 4)))]
    assert Path(parts[1][2][1]) == Path(cfg['cache_root']) / 'UTS01' / 'val_d1.npy'
    log = Provenance.from_dict(sel['provenance'])
    for ref, stories in labeled.items():
        assert sorted({r.stimulus for r in log.records if r.subject == ref and r.kind == 'transcript'}) == stories
        assert sorted({r.stimulus for r in log.records if r.subject == ref and r.purpose == 'reference_model'
                       and r.kind == 'brain'}) == [f'x{i}' for i in range(1, 7)]
    assert len(list(rt.CACHE.glob('references/*.npz'))) == 3  # One model per reference, shared across budgets.
    rt.stage_fit(cfg, 'UTS05', 'dense', ['S3'], list(th.ARMS), 'evaluate', 'cpu')
    for arm in th.ARMS:
        path = rt.cell_path('UTS05', 'dense', 'evaluate', 'S3', arm)
        cell = json.loads(path.read_text())
        assert th.audit_file(path, manifests)['passed'] and 'test' in cell and 'test_shift_null' in cell
        assert cell.get('selection') == cells['S3', arm].get('selection')  # Reused (XDEC selects nothing).
        np.testing.assert_allclose(cell['val']['session_macro_feature_correlation'],
                                   cells['S3', arm]['val']['session_macro_feature_correlation'], rtol=1e-9)
    main = {arm: rt.cell_path('UTS05', 'dense', 'develop', 'S3', arm).read_bytes() for arm in th.ARMS}
    with pytest.raises(SystemExit):  # TH_SEL_rand is not part of the sensitivity analysis.
        rt.stage_fit(cfg, 'UTS05', 'dense', ['S3'], list(th.ARMS), 'develop', 'cpu', refs=['UTS01', 'UTS02'])
    memo = {}  # The pairs of one call share their converter alphas (content-keyed).
    rt.stage_fit(cfg, 'UTS05', 'dense', ['S3'], ['TH_SEL', 'XDEC'], 'develop', 'cpu', refs=['UTS01', 'UTS02'],
                 alpha_memo=memo)
    for arm in ('TH_SEL', 'XDEC'):
        two = rt.cell_path('UTS05', 'dense', 'develop', 'S3', arm, ('UTS01', 'UTS02'))
        assert 'sensitivity' in two.parts and json.loads(two.read_text())['references'] == ['UTS01', 'UTS02']
        assert th.audit_file(two, manifests)['passed']
    rt.stage_fit(cfg, 'UTS05', 'dense', ['S3'], ['TH_SEL'], 'develop', 'cpu', refs=['UTS01', 'UTS03'], alpha_memo=memo)
    first, second = (json.loads(rt.cell_path('UTS05', 'dense', 'develop', 'S3', 'TH_SEL', pair).read_text())
                     for pair in (('UTS01', 'UTS02'), ('UTS01', 'UTS03')))
    a, b = first['final_selection']['converters'], second['final_selection']['converters']
    assert b['UTS01'].get('memo') and not b['UTS03'].get('memo') and not a['UTS01'].get('memo')
    assert b['UTS01']['alpha'] == a['UTS01']['alpha'] == sel['final_selection']['converters']['UTS01']['alpha']
    assert all(f['priors']['isp']['converters']['UTS01'].get('memo') for f in second['folds'])
    assert {arm: rt.cell_path('UTS05', 'dense', 'develop', 'S3', arm).read_bytes() for arm in th.ARMS} == main
    # After the lock, rerunning a develop cell is refused before any fit; its bytes stay those the lock pinned.
    Path('experiments').mkdir()
    Path('experiments/v4_lock.json').write_text('{}')
    with pytest.raises(SystemExit):
        rt.stage_fit(cfg, 'UTS05', 'dense', ['S3'], ['TH_SEL_rand'], 'develop', 'cpu')
    assert {arm: rt.cell_path('UTS05', 'dense', 'develop', 'S3', arm).read_bytes() for arm in th.ARMS} == main
