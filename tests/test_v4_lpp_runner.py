"""v4 Module G (LPP runner, scripts/run_v4_lpp.py) on a synthetic Le Petit Prince cohort: index and caches, data
binding, outer folds and budgets, Tier-1 and K-subset arms through run_v4's engine, picture TRs in label-free fits,
the residualized endpoint, the Module H comparators, G4, write-once cells, the gates and the lock and plan checks,
plus independent numpy references (W0 ridge, nuisance LOSO ridge, G4(b) shift null) and the Module I readers.
CPU, seconds, no data files (the fixture writes a tiny LPP-shaped dataset with its preparation record)."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_v4  # noqa: E402
import lpp_prepare  # noqa: E402
import run_v3 as v3  # noqa: E402
import run_v4 as v4  # noqa: E402
import run_v4_lpp as rl  # noqa: E402
from src import budgets as bud  # noqa: E402
from src.priors import eb as ebm  # noqa: E402
from src.priors import isp as ispm  # noqa: E402
from src.provenance import Provenance, ProvenanceError  # noqa: E402
from src.voxel_kernels import DELAYS  # noqa: E402

PARSER = ROOT / 'reference/semantic-decoding/decoding/utils_ridge/textgrid.py'
pytestmark = pytest.mark.skipif(not PARSER.is_file(), reason='Tang/Huth TextGrid parser not checked out')
SECTIONS = list(bud.LPP_SECTIONS)
N_TR, V, DIM = 130, 60, 6
TARGET, REFS = 'sub-EN001', ('sub-EN002', 'sub-EN003', 'sub-EN004')
TEST = '4'  # Its budgets: L3 = 5, 7, 2; L8 adds 8, 1, 6, 9, 3.
INVALID = ('2', 70)  # One invalid target TR.
MASK = 'a' * 64
CFG = dict(data=dict(window=32, stride=4, target_phase=0, cutoffs=[3, 4]), alphas=[-1, 7, 17],
           evaluation=dict(exclusions=[10, 32], shift_min_tr=40, shift_draws=200, shift_seed=0))
TIER1_TEST = ['W0', 'W1', 'EB', 'PRIOR', 'EB_f0.5', 'EB_SELF', 'EB_MISALIGN', 'ORACLE', 'PRIOR_f0.5']


# ------------------------------------------------------------------------------------------ fixtures
def write_record(root, subjects=(TARGET,) + REFS, drop=()):
    """lpp_prepare's preparation record with the output_content digests (``drop``: outputs left out), written after
    every data file."""
    root = Path(root)
    content = {}
    for s in SECTIONS:
        for kind in ('targets', 'valid'):
            rel = f'semantic/targets/{s}.{kind}.npy'
            content[rel] = lpp_prepare.array_digest(np.load(root / rel))
        for subject in subjects:
            rel = f'raw/{subject}/{s}.hf5'
            with h5py.File(root / rel, 'r') as handle:
                content[rel] = lpp_prepare.array_digest(np.asarray(handle['data']))
    content = {k: v for k, v in content.items() if k not in drop}
    (root / 'prepare_provenance.json').write_text(json.dumps(dict(mode='DEV', output_content=content,
                                                                  g4a=g4a(subjects))))
    return root / 'prepare_provenance.json'


def g4a(subjects, trs=None):
    """lpp_prepare's G4(a) run lengths of ``subjects`` (default the fixture's N_TR per section)."""
    trs = trs or (N_TR,) * len(SECTIONS)
    return dict(passed=True, runs={s: [dict(section=sec, run_number=i + 1, n_tr=n) for i, (sec, n) in
                                       enumerate(zip(SECTIONS, trs))] for s in subjects})


def write_cohort(root, perturb=None, context_reset='section_start'):
    """A shared stimulus latent drives every listener's BOLD 2 TRs later and the GPT-like targets at the same TR.

    ``perturb``: a section whose target-listener BOLD and (shared) targets are replaced by noise.
    """
    rng = np.random.default_rng(11)
    latent = {s: rng.normal(size=(N_TR, 3)) for s in SECTIONS}
    mix = rng.normal(size=(3, DIM))
    root = Path(root)
    targets = root / 'semantic' / 'targets'
    stimuli = root / 'stimuli'
    targets.mkdir(parents=True)
    stimuli.mkdir(parents=True)
    coords = rng.integers(0, 20, size=(V, 3))
    np.savez(root / 'mask.npz', coords=coords, membership=np.arange(V))
    paths = {}
    for s in SECTIONS:
        y = latent[s] @ mix + 0.3 * rng.normal(size=(N_TR, DIM))
        valid = np.ones(N_TR, dtype=bool)
        if s == INVALID[0]:
            valid[INVALID[1]] = False
        if s == perturb:
            y = np.random.default_rng(99).normal(size=y.shape)
        np.save(targets / f'{s}.targets.npy', y.astype(np.float32))
        np.save(targets / f'{s}.valid.npy', valid)
        np.save(stimuli / f'picture_section-{s}.npy', lpp_prepare.picture_mask(N_TR, s))
        words = [(0.3 + 0.5 * i, 0.45 + 0.5 * i + 0.3 * ((i * 37) % 11) / 11, f'w{i % 7}')  # Varying durations.
                 for i in range(int(N_TR * 2 / 0.5) - 2)]
        (stimuli / f'section-{s}.TextGrid').write_text(lpp_prepare.textgrid_text(words, N_TR * 2.0))
        paths[s] = dict(target_path=str(targets / f'{s}.targets.npy'),
                        target_valid_path=str(targets / f'{s}.valid.npy'),
                        picture_path=str(stimuli / f'picture_section-{s}.npy'),
                        stimulus_path=str(stimuli / f'section-{s}.TextGrid'),
                        audio_path=str(stimuli / f'section-{s}.wav'),
                        semantic_metadata=dict(context_reset=context_reset, context_words=20, layer=9,
                                               cache_key=f'key-{s}'))
    for k, subject in enumerate((TARGET,) + REFS):
        w = rng.normal(size=(3, V))
        w[:, 40 + 5 * k:] = 0  # Each listener has a few silent voxels.
        folder = root / 'raw' / subject
        folder.mkdir(parents=True)
        runs = []
        for s in SECTIONS:
            lagged = np.zeros_like(latent[s])
            lagged[2:] = latent[s][:-2]
            x = 1.5 * lagged @ w + rng.normal(size=(N_TR, V))
            if subject == TARGET and s == perturb:
                x = np.random.default_rng(98).normal(size=x.shape)
            with h5py.File(folder / f'{s}.hf5', 'w') as handle:
                handle.create_dataset('data', data=((x - x.mean(0)) / x.std(0)).astype(np.float32))
            runs.append(dict(run_id=f'ds003643_{subject}_section-{s}', dataset='ds003643', subject=subject,
                             condition='perceived_speech', session=s, split='train', story=s, section=int(s),
                             n_tr=N_TR, n_features=V, tr_seconds=2.0, tr_start_seconds=1.0,
                             source_key=f'{subject}/{s}.hf5', **paths[s]))
        manifest = dict(dataset='ds003643', subject=subject, mode='lpp', raw_root=str((root / 'raw').resolve()),
                        mask_path=str((root / 'mask.npz').resolve()),
                        normalization_path=str((root / 'mask.npz').resolve()), mask_content_sha256=MASK, runs=runs)
        (root / 'semantic' / subject).mkdir(parents=True)
        (root / 'semantic' / subject / 'semantic_manifest.json').write_text(json.dumps(manifest))
    write_record(root)
    return root


@pytest.fixture(scope='module')
def cohort(tmp_path_factory):
    return write_cohort(tmp_path_factory.mktemp('lpp') / 'lpp')


@pytest.fixture(scope='module')
def caches(cohort, tmp_path_factory):
    """Delay-row caches of every listener on both clocks (prepare-caches)."""
    cache = tmp_path_factory.mktemp('cache')
    ctx = context(cohort, cache, tmp_path_factory.mktemp('out'))
    rl.stage_prepare(ctx, (TARGET,) + REFS, rl.CLOCKS)
    return cache


def context(lpp_root, cache_root, out_root, **kwargs):
    kwargs.setdefault('code', 'lpp-code-test')
    kwargs.setdefault('lock_path', Path(out_root) / 'absent_v4_lock.json')
    return rl.Context(copy.deepcopy(CFG), 'cpu', lpp_root=lpp_root, cache_root=cache_root, out_root=out_root,
                      core='core-test', **kwargs)


def target(ctx, clock='dense'):
    return rl.LppTarget(ctx, TARGET, clock)


def cells(folder):
    return {p.name: json.loads(p.read_text()) for p in sorted((Path(folder) / 'cells').glob('*.json'))}


def reads(log):
    return [(r['subject'], r['stimulus'], r['kind'], r['purpose'], r['fold']) for r in log['records']]


def h5(root, subject, section):
    with h5py.File(Path(root) / 'raw' / subject / f'{section}.hf5', 'r') as handle:
        return np.asarray(handle['data'])


def feature_r(pred, tgt):
    """Mean over columns of the population Pearson r (an independent numpy section r)."""
    a, b = pred - pred.mean(0), tgt - tgt.mean(0)
    return float(np.mean((a * b).mean(0) / np.sqrt((a * a).mean(0) * (b * b).mean(0))))


# ------------------------------------------------------------------------------------- index and caches
def test_index_has_target_index_eligibility_minus_picture_bins(cohort):
    manifest = cohort / 'semantic' / TARGET / 'semantic_manifest.json'
    dense, stride4 = rl.LppIndex(manifest, 1), rl.LppIndex(manifest, 4)
    picture = lpp_prepare.picture_mask(N_TR, '1')
    want = [(s, t) for s in SECTIONS for t in range(28, N_TR - 4)
            if (s, t) != INVALID and not (s == '1' and picture[t])]
    got = [(dense.runs[ri]['story'], t) for ri, t in dense.items]
    assert got == want and dense.picture_excluded == int(picture[28:N_TR - 4].sum()) == 12
    assert [(stride4.runs[ri]['story'], t) for ri, t in stride4.items] == [p for p in want if p[1] % 4 == 0]
    assert dense.split == 'all' and len(dense) == len(want) and dense.target_dim == DIM
    sig = dense.signature()
    assert sig['protocol'] == rl.PROTOCOL_ID and sig['n_targets'] == len(want) and sig['picture_excluded'] == 12
    assert dense.sessions[dense.run_ids[0]] == '1'
    np.testing.assert_array_equal(dense.targets([0]), np.load(cohort / 'semantic/targets/1.targets.npy')[[40]])
    assert dense.target_context['context_reset'] == 'section_start' and dense.target_context['context_words'] == 20
    path = cohort.parent / 'bad.json'
    for edit in (lambda m: m['runs'][3].update(dataset='ds003020'), lambda m: m.update(runs=m['runs'][:8])):
        bad = json.loads(manifest.read_text())
        edit(bad)
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            rl.LppIndex(path, 1)
    for edit in (lambda m: m['runs'][5]['semantic_metadata'].update(context_reset='story_start'),
                 lambda m: m['runs'][2]['semantic_metadata'].update(context_words=40),
                 lambda m: m['runs'][0].pop('semantic_metadata')):
        bad = json.loads(manifest.read_text())
        edit(bad)
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match='R6'):  # Cross-section GPT-2 context is refused from the record.
            rl.LppIndex(path, 1)


def test_caches_hold_raw_h5_rows_and_outer_views_score_stride4_bins(cohort, caches, tmp_path):
    ctx = context(cohort, caches, tmp_path)
    dense, stride4 = target(ctx, 'dense'), target(ctx, 'stride4')
    assert (caches / TARGET / 'dense' / 'all.meta.json').is_file()
    sidecar = json.loads((caches / TARGET / 'dense' / rl.CONTENT_FILE).read_text())
    assert sidecar['bold'] == {s: ctx.content(TARGET)[s][0] for s in SECTIONS}
    data = h5(cohort, TARGET, '2')
    rows = dense.rows_of(['2'])
    for d in DELAYS:
        np.testing.assert_array_equal(np.load(dense.cache['train'][d])[rows], data[dense.row_tr[rows] + d])
    view = dense.outer(TEST)
    assert view.split_stories == {'train': [s for s in SECTIONS if s != TEST], 'val': [], 'test': [TEST]}
    sp = view.split['test']
    assert list(sp.target_trs) == list(range(28, N_TR - 4, 4)) and set(sp.run_ids) == {f'ds003643_{TARGET}_section-4'}
    assert dense.split_stories['test'] == [] and dense.test_section is None  # The base target is untouched.
    budget = bud.lpp_section_budgets(TEST, SECTIONS)['L3']
    bd = rl.LppBudget(view, 'L3', budget)
    bd4 = rl.LppBudget(stride4.outer(TEST), 'L3', budget)
    assert set(bd.story) == set(budget) and TEST not in set(bd.story)
    data = h5(cohort, TARGET, TEST)
    for d in DELAYS:  # Both clocks score the same stride-4 bins of the test section.
        np.testing.assert_array_equal(bd.eval_raw['test'][d].numpy(), data[sp.target_trs + d])
        np.testing.assert_array_equal(bd4.eval_raw['test'][d].numpy(), data[sp.target_trs + d])
    assert len(bd4.rows) * 4 == pytest.approx(len(bd.rows), rel=0.05)
    with pytest.raises(ValueError):
        rl.LppBudget(view, 'L3', budget + [TEST])
    with pytest.raises(ValueError):
        rl.LppBudget(dense, 'L3', budget)  # No outer view: no test section.
    assert view.coords().shape == (V, 3)


def test_caches_maps_and_records_are_bound_to_the_recorded_data_content(tmp_path):
    """A same-shape re-preparation (same H5 file size) invalidates the delay rows, the PCA bases and the data
    fingerprint; unfinished preparations and outputs without a recorded digest are refused."""
    root = write_cohort(tmp_path / 'lpp')
    ctx = context(root, tmp_path / 'cache', tmp_path / 'out')
    rl.stage_prepare(ctx, (TARGET, REFS[0]), ['dense'])
    old = rl.LppStore(ctx, rl.LppTarget(ctx, TARGET, 'dense'), (REFS[0],))
    key, data = old.basis(REFS[0], TEST)['key'], ctx.data_sha256([TARGET, REFS[0]])
    new = np.random.default_rng(7).normal(size=(N_TR, V)).astype(np.float32)
    for subject in (TARGET, REFS[0]):
        path = root / 'raw' / subject / '2.hf5'
        size = path.stat().st_size
        with h5py.File(path, 'r+') as handle:
            handle['data'][...] = new
        assert path.stat().st_size == size  # The frozen verify_cache (key + size) cannot see this.
    with pytest.raises(SystemExit, match='newer'):  # Rewritten after its record: an unfinished preparation.
        context(root, tmp_path / 'cache', tmp_path / 'out').content(TARGET)
    record = write_record(root)
    fresh = context(root, tmp_path / 'cache', tmp_path / 'out')
    assert fresh.data_sha256([TARGET, REFS[0]]) != data
    with pytest.raises(ValueError, match='prepare-caches'):  # The old cache is not served for the new data.
        rl.LppTarget(fresh, TARGET, 'dense')
    rl.stage_prepare(fresh, (TARGET, REFS[0]), ['dense'])
    tgt = rl.LppTarget(fresh, TARGET, 'dense')
    rows = tgt.rows_of(['2'])
    for d in DELAYS:
        np.testing.assert_array_equal(np.load(tgt.cache['train'][d])[rows], new[tgt.row_tr[rows] + d])
    store = rl.LppStore(fresh, tgt, (REFS[0],))
    assert store.basis(REFS[0], TEST)['key'] != key
    np.testing.assert_array_equal(store.target_bold('2'), rl.zscore(new))  # Live reads agree with the cache.
    write_record(root, drop=(f'raw/{REFS[0]}/5.hf5',))
    with pytest.raises(SystemExit, match='no recorded content digest'):
        context(root, tmp_path / 'cache', tmp_path / 'out').content(REFS[0])
    assert record.is_file()


def test_content_is_verified_not_only_timestamps(tmp_path):
    """Content restored with old timestamps (cp -p, rsync -a, tar x) passes the mtime pre-filter but not the content
    checks: live H5 reads, delay-cache builds and the target arrays are refused against the record."""
    root = write_cohort(tmp_path / 'lpp')
    ctx = context(root, tmp_path / 'cache', tmp_path / 'out')
    rl.stage_prepare(ctx, (TARGET,), ['dense'])
    sidecar = json.loads((tmp_path / 'cache' / TARGET / 'dense' / rl.CONTENT_FILE).read_text())
    assert sidecar['verified'] and sidecar['bold'] == {s: ctx.content(TARGET)[s][0] for s in SECTIONS}
    path = root / 'raw' / TARGET / '2.hf5'
    stat = path.stat()
    with h5py.File(path, 'r+') as handle:
        handle['data'][...] = np.random.default_rng(3).normal(size=(N_TR, V)).astype(np.float32)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # Old timestamp: same size, same mtime.
    fresh = context(root, tmp_path / 'cache', tmp_path / 'out')
    fresh.content(TARGET)  # The mtime pre-filter cannot see it ...
    tgt = rl.LppTarget(fresh, TARGET, 'dense')  # ... nor the bound delay cache, built from verified content.
    store = rl.LppStore(fresh, tgt, ())
    np.testing.assert_array_equal(store.target_bold('3'), rl.zscore(h5(root, TARGET, '3')))
    with pytest.raises(SystemExit, match='content digest'):  # ... but every live read checks the content.
        store.target_bold('2')
    with pytest.raises(SystemExit, match='content digest'):
        store.reference_runs('raw', TARGET, '2', [0, 1], 'k')
    (tmp_path / 'cache' / TARGET / 'dense' / rl.CONTENT_FILE).unlink()
    with pytest.raises(SystemExit, match='content digest'):  # A rebuild verifies every H5 first.
        rl.stage_prepare(fresh, (TARGET,), ['dense'])
    target = root / 'semantic' / 'targets' / '5.targets.npy'
    stat, y = target.stat(), np.load(target)
    np.save(target, (y + 1).astype(y.dtype))
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(SystemExit, match='content digest'):
        context(root, tmp_path / 'cache', tmp_path / 'out').content(REFS[0])
    data = rl.PreparedData(root)  # One content check per file state: unchanged files are not re-hashed.
    rel = 'semantic/targets/6.targets.npy'
    calls = []
    reader = lambda p: calls.append(p) or np.load(p)  # noqa: E731
    data.verified(rel, root / rel, reader)
    data.verified(rel, root / rel, reader)
    assert len(calls) == 1


def test_bases_store_only_what_is_read_back_and_disk_check(cohort, tier1, tmp_path):
    ctx, store, _, _ = tier1
    entry = store.basis(REFS[1], TEST)
    with np.load(next((ctx.cache_root / 'bases').glob(f'{REFS[1]}_t{TEST}_{entry["key"]}.npz'))) as z:
        assert set(z.files) == {'sections'} | {f'proj_{s}' for s in entry['sections']}  # No 113 MB comps/mean.
    rl.disk_check([tmp_path / 'not' / 'yet'], 0)
    with pytest.raises(SystemExit, match='min-free-gb'):
        rl.disk_check([tmp_path], 1e12)


def test_cache_keys_hold_the_code_fingerprint(cohort, caches, tier1):
    """Maps computed by other runner code are never served: bases, ISP maps and reference voxel slices are keyed
    by the code fingerprint."""
    ctx, store, _, _ = tier1
    other = context(cohort, caches, ctx.out_root, code='other-lpp-code')
    store2 = rl.LppStore(other, rl.LppTarget(other, TARGET, 'dense'), REFS)
    assert store2.basis(REFS[0], TEST)['key'] != store.basis(REFS[0], TEST)['key']
    view = store2.tgt.outer(TEST)
    budget = bud.lpp_section_budgets(TEST, SECTIONS)['L3']
    before = set((caches / 'isp').glob('*.npz'))
    prov = Provenance(TARGET, 'S', budget, [], [TEST])
    rl.LppPriors(view, rl.LppReferences(view.cfg, view, prov, 'cpu', store2), prov, 'cpu', store2).isp(budget)
    assert set((caches / 'isp').glob('*.npz')) - before  # A new map, not the one cached under 'lpp-code-test'.


# ------------------------------------------------------------------------------- picture TRs and priors
def test_label_free_fits_drop_the_picture_trs(cohort, tier1):
    ctx, store, _, _ = tier1
    picture = lpp_prepare.picture_mask(N_TR, '1')
    keep = np.flatnonzero(~picture)
    assert 0 < picture.sum() and np.array_equal(store.kept(TARGET, '1'), keep)
    assert len(store.kept(TARGET, '2')) == N_TR
    raw = h5(cohort, TARGET, '1')
    np.testing.assert_array_equal(store.target_bold('1'), rl.zscore(raw[keep]))  # z-scored over the kept rows.
    assert np.abs(store.target_bold('1').mean(0)).max() < 1e-5
    run2 = next(r for r in store.tgt.manifest['runs'] if r['story'] == '2')
    np.testing.assert_array_equal(store.target_bold('2'), ispm.load_bold(store.raw_root(TARGET), run2))
    evaluated = store.target_eval_bold('1')  # XDEC's test input: z-scored over the kept rows, as its converters'.
    assert evaluated.shape == (N_TR, V) and not evaluated[picture].any()
    np.testing.assert_array_equal(evaluated[keep], store.target_bold('1'))
    np.testing.assert_array_equal(store.target_eval_bold('2'), ispm.load_bold(store.raw_root(TARGET), run2))
    view1 = store.tgt.outer('1')  # No scored bin of section 1 reads a picture row through its delays and lags.
    reach = view1.split['test'].target_trs[:, None] + np.array(DELAYS)[None, :]
    assert (reach.min() - 1) >= keep.min() and not picture[reach.ravel()].any()
    labels, _ = ispm.row_groups(['1'], [len(keep)])  # FIT_ROWS documents both section-1 splits.
    split_isp = int(keep[labels.index('1:1')])
    split_own = bud.half_groups(['1'] * N_TR, list(range(N_TR)), {'1': N_TR}).index('1:1')
    assert (split_isp, split_own) == (int(keep[len(keep) // 2]), N_TR // 2) and split_isp != split_own
    assert 'TR 141' in rl.FIT_ROWS and 'TR 40 + 242 // 2 = 161' in rl.FIT_ROWS and 'XDEC' in rl.FIT_ROWS
    assert store.basis(REFS[0], TEST)['proj']['1'].shape[0] == len(keep)
    zref = store.reference_runs('z', REFS[0], '1', [3, 0, 5], 'picture-test')
    np.testing.assert_array_equal(zref, rl.zscore(h5(cohort, REFS[0], '1')[keep])[:, [3, 0, 5]])
    raw_ref = store.reference_runs('raw', REFS[0], '1', [3, 0, 5], 'picture-test')
    np.testing.assert_array_equal(raw_ref, h5(cohort, REFS[0], '1')[keep][:, [3, 0, 5]])
    store.release()
    np.testing.assert_array_equal(store.reference_runs('z', REFS[0], '1', [3, 0, 5], 'again'), zref)  # From disk.
    view = store.tgt.outer(TEST)
    prov = Provenance(TARGET, 'S', SECTIONS[4:], [], [TEST])  # Any material without the test section.
    pri = rl.LppPriors(view, rl.LppReferences(view.cfg, view, prov, 'cpu', store), prov, 'cpu', store)
    r, info = pri.isp(['1', '2'])  # ISP over section 1 aligns kept target and reference rows.
    assert r.shape == (V,) and info['stories'] == ['1', '2']


# --------------------------------------------------------------------------------------- Tier-1 arms
def run_tier1(cohort, cache, out, arms=TIER1_TEST, name='L3', clock='dense'):
    ctx = context(cohort, cache, out)
    tgt = target(ctx, clock)
    store = rl.LppStore(ctx, tgt, REFS)
    folder = out / TARGET / clock / 'develop'
    preds = rl.run_budget(ctx, tgt.outer(TEST), store, name, arms, 'develop', folder)
    return ctx, store, folder, preds


@pytest.fixture(scope='module')
def tier1(cohort, caches, tmp_path_factory):
    out = tmp_path_factory.mktemp('out')
    return run_tier1(cohort, caches, out)


def test_tier1_cells_pass_the_registered_audit_and_never_fit_on_the_test_section(tier1):
    ctx, store, folder, preds = tier1
    got = cells(folder)
    budget = bud.lpp_section_budgets(TEST, SECTIONS)['L3']
    assert sorted(got) == sorted(f't{TEST}_L3_{a}.json' for a in TIER1_TEST)
    for name, cell in got.items():
        result = audit_v4.audit_file(folder / 'cells' / name)
        assert result['passed'], (name, result)
        assert cell['code_sha256'] == ctx.code and cell['core_code_sha256'] == ctx.core
        assert cell['data_sha256'] == ctx.data_sha256((TARGET,) + REFS) and cell['fit_rows'] == rl.FIT_ROWS
        assert cell['targets']['context_reset'] == 'section_start' and 'lock' not in cell
        assert (cell['stage'], cell['dataset'], cell['test_section'], cell['budget']) == ('develop', 'lpp', TEST, 'L3')
        assert cell['estimable'] and np.isfinite(cell['test'][rl.VAL])
        assert cell['test']['by_session_feature_correlation'].keys() == {TEST}  # The section is its own session.
        assert np.isfinite(cell['residualized']['test'][rl.VAL])
        null = cell['test_shift_null']  # §9: the within-section circular-shift null of every scored cell.
        assert (null['runs_used'], null['min_shift_tr'], null['draws'], null['seed']) == (1, 40, CFG['evaluation'][
            'shift_draws'], 0) and null['observed_session_macro'] == pytest.approx(cell['test'][rl.VAL], abs=1e-12)
        assert set(cell['nuisance_floors']['models']) == {'position', 'speech_rate', 'both'}
        assert 'controls' not in cell  # Only at the primary cell (stage_fit's ``primary``).
        log = cell['provenance']
        assert log['budget'][0]['stimulus'] in budget and [i['stimulus'] for i in log['test_items']] == [TEST]
        assert log['folds'] == {str(i): [dict(subject=None, stimulus=s, trs=None)] for i, s in enumerate(budget)}
        for subject, stimulus, kind, purpose, fold in reads(log):
            if stimulus == TEST:  # The outer test section is only ever scored.
                assert subject == TARGET and fold is None, (name, purpose)
                assert purpose in ('decoder_score', 'nuisance_apply', 'context_reset'), (name, kind, purpose)
        sections = 8 if cell['arm'] == 'ORACLE' else 3  # ORACLE: all non-test labels (over budget by design).
        assert cell['ledger']['target_bold_minutes'] == pytest.approx(sections * N_TR * 2 / 60)
    w0 = got[f't{TEST}_L3_W0.json']
    assert w0['ledger']['references'] == [] and w0['ledger']['target_pairs_dense'] == len(
        target(ctx).rows_of(budget))
    eb = got[f't{TEST}_L3_EB.json']
    assert eb['ledger']['references'] == list(REFS) and eb['reference_bases'].keys() == set(REFS)
    assert set(eb['final_weights']['eb']) == {str(d) for d in DELAYS}
    assert [f['priors']['isp']['stories'] for f in eb['folds']] == [[s for s in budget if s != h] for h in budget]
    oracle = got[f't{TEST}_L3_ORACLE.json']
    flagged = {(r['stimulus'], r['kind']) for r in oracle['provenance']['records'] if r['purpose'] == 'oracle'}
    assert flagged == {(s, k) for s in SECTIONS if s != TEST for k in ('brain', 'transcript')}
    assert oracle['provenance']['flags'] and not oracle['provenance']['violations']
    ladder = got[f't{TEST}_L3_PRIOR_f0.5.json']
    assert ladder['engine_arm'] == 'PRIOR' and 'lpp_note' in ladder and ladder['selection']['gamma'] in v4.GAMMAS
    for arm in TIER1_TEST:  # The predictions are the cells' (float16 copies on disk).
        saved = np.load(folder / 'predictions' / f't{TEST}_L3_{arm}.npz')
        np.testing.assert_allclose(saved['test'], preds[arm]['test'], rtol=1e-2, atol=1e-3)


def test_every_target_transcript_read_follows_a_section_context_reset(tier1):
    _, _, folder, _ = tier1
    for name, cell in cells(folder).items():
        seen = set()
        for subject, stimulus, kind, purpose, _ in reads(cell['provenance']):
            if subject != TARGET or kind != 'transcript':
                continue
            if purpose == 'context_reset':
                seen.add(stimulus)
            elif purpose not in ('nuisance_fit', 'nuisance_apply'):
                assert stimulus in seen, (name, stimulus, purpose)
    log = Provenance.from_dict(cells(folder)[f't{TEST}_L3_W1.json']['provenance'])
    planted = Provenance.from_dict(log.to_dict())
    planted.records = {r: n for r, n in planted.records.items() if r.purpose != 'context_reset'}
    with pytest.raises(ProvenanceError) as error:  # Without the resets, R6 refuses the cell.
        planted.check()
    assert 'R6' in error.value.rules


def test_outer_test_section_never_influences_selection_or_weights(cohort, caches, tier1, tmp_path):
    """Replace the test section's target BOLD and (shared) targets by noise: every selection, CV curve, prior map
    and weight is unchanged; only the test scores move."""
    _, _, folder, preds = tier1
    other = write_cohort(tmp_path / 'lpp', perturb=TEST)
    ctx = context(other, tmp_path / 'cache', tmp_path / 'out')
    rl.stage_prepare(ctx, (TARGET,) + REFS, ['dense'])
    _, _, folder2, preds2 = run_tier1(other, tmp_path / 'cache', tmp_path / 'out')
    a, b = cells(folder), cells(folder2)
    untimed = lambda folds: [{k: v for k, v in f.items() if k not in ('seconds', 'peak_gpu_gib')} for f in folds]
    for name in a:
        for key in ('selection', 'cv_curves', 'final_weights', 'final_priors', 'ledger'):
            assert a[name].get(key) == b[name].get(key), (name, key)
        assert untimed(a[name]['folds']) == untimed(b[name]['folds']), name
    arm = 'W0'
    assert a[f't{TEST}_L3_{arm}.json']['test'][rl.VAL] != b[f't{TEST}_L3_{arm}.json']['test'][rl.VAL]
    assert not np.allclose(preds[arm]['test'], preds2[arm]['test'])


def test_residualized_endpoint_refits_the_selected_ridge_on_residual_targets(cohort, caches, tmp_path, monkeypatch):
    """With a zero nuisance the residualized block equals the raw endpoint; the nuisance CV folds are the
    registered inner folds and its features come from B and the test section only."""
    seen = {}
    original = v3.feature_ridge_cv

    def zero_nuisance(shim, x_train, rows, y, splits_x):
        from src.structured_ridge import budget_folds
        seen['folds'] = budget_folds(shim.row_sessions, rows, shim.fixed)[0]
        seen['shape'] = (x_train.shape, splits_x['test'].shape)
        preds, info = original(shim, x_train, rows, y, splits_x)
        seen['alpha'] = info['alpha']
        return {k: np.zeros_like(v) for k, v in preds.items()}, info
    monkeypatch.setattr(v3, 'feature_ridge_cv', zero_nuisance)
    _, _, folder, preds = run_tier1(cohort, caches, tmp_path, arms=['W0'])
    budget = bud.lpp_section_budgets(TEST, SECTIONS)['L3']
    assert sorted(seen['folds']) == sorted((s,) for s in budget)
    n_rows = sum(N_TR - 32 - (s == INVALID[0]) for s in budget)
    assert seen['shape'] == ((n_rows, 11), (len(range(28, N_TR - 4, 4)), 11))
    np.testing.assert_allclose(preds['W0']['residualized'], preds['W0']['test'], rtol=1e-10, atol=1e-12)
    cell = cells(folder)[f't{TEST}_L3_W0.json']
    assert cell['residualized']['test'][rl.VAL] == pytest.approx(cell['test'][rl.VAL], abs=1e-10)
    assert cell['residualized']['nuisance']['alpha'] == seen['alpha']
    nuisance = {(r[1], r[3], r[4]) for r in reads(cell['provenance']) if r[3].startswith('nuisance')}
    assert {s for s, _, _ in nuisance} == set(budget) | {TEST}
    assert {(s, p) for s, p, f in nuisance if f is None and s == TEST} == {(TEST, 'nuisance_apply')}
    for i, s in enumerate(budget):  # Inside inner fold i the held section is only ever applied to.
        assert (s, 'nuisance_fit', str(i)) not in nuisance and (s, 'nuisance_apply', str(i)) in nuisance


def test_final_fit_matches_a_direct_kernel_ridge_refit(cohort, caches, tmp_path):
    """W0 through ``final_fit``: raw and residualized predictions are KernelRidgePath on the same kernel."""
    ctx = context(cohort, caches, tmp_path)
    view = target(ctx).outer(TEST)
    budget = bud.lpp_section_budgets(TEST, SECTIONS)['L3']
    bd = rl.LppBudget(view, 'L3', budget)
    prov = Provenance(TARGET, 'S', budget, [], [TEST], folds=bud.inner_folds('L3', budget), budget_name='L3',
                      stage='develop')
    pri = rl.LppPriors(view, None, prov, 'cpu', rl.LppStore(ctx, view, ()))
    residual = np.random.default_rng(0).normal(size=bd.y.shape)
    sel = dict(W0=dict(gamma=1.0, alpha=10.0))
    preds, _ = rl.final_fit(view, bd, 'L3', ['W0'], sel, pri, prov, None, residual)
    kernels, cross = v4.build_kernels(bd, np.arange(len(bd.rows)), {'W0': None}, np.arange(len(bd.rows)), ('test',))
    for y, key in ((bd.y, 'test'), (residual, 'residualized')):
        path = v4.KernelRidgePath(kernels['W0'], torch.as_tensor(y), 'cpu')
        np.testing.assert_allclose(preds['W0'][0][key], path.predict(path.project(cross['W0']['test']), 10.0).numpy(),
                                   rtol=1e-10, atol=1e-12)


def test_w0_prediction_and_section_r_match_a_numpy_primal_ridge(cohort, tier1):
    """Independent reference: the W0 outer prediction is the primal ridge (unpenalized intercept) on the raw H5
    rows t+1..t+3, standardized with the budget rows' statistics, at the selected alpha; its section r is the mean
    per-dimension Pearson r over the test section's stride-4 bins."""
    _, _, folder, preds = tier1
    cell = cells(folder)[f't{TEST}_L3_W0.json']
    alpha = cell['selection']['alpha']
    budget = sorted(bud.lpp_section_budgets(TEST, SECTIONS)['L3'])
    data = {s: h5(cohort, TARGET, s).astype(np.float64) for s in budget + [TEST]}
    ys = {s: np.load(cohort / 'semantic' / 'targets' / f'{s}.targets.npy').astype(np.float64) for s in budget + [TEST]}
    train = [(s, t) for s in budget for t in range(28, N_TR - 4) if (s, t) != INVALID]  # No picture section in B.
    test = [(TEST, t) for t in range(28, N_TR - 4, 4)]

    def design(items):
        return np.concatenate([np.stack([data[s][t + d] for s, t in items]) for d in DELAYS], axis=1)
    x, xt = design(train), design(test)
    mean, std = x.mean(0), x.std(0)
    assert std.min() > 1e-3
    z, zt = (x - mean) / std, (xt - mean) / std
    y = np.stack([ys[s][t] for s, t in train])
    beta = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - y.mean(0)))
    pred = zt @ beta + y.mean(0)
    np.testing.assert_allclose(preds['W0']['test'], pred, rtol=1e-9, atol=1e-11)
    truth = np.stack([ys[TEST][t] for _, t in test])
    assert cell['test'][rl.VAL] == pytest.approx(feature_r(pred, truth), abs=1e-10)


def test_nuisance_residuals_match_a_numpy_leave_one_section_out_ridge(cohort, caches, tmp_path):
    """Independent reference for H2's nuisance model: ridge with intercept on the 11 position + speech-rate
    features, standardized per fold (float32, as the v3 Standardizer), alpha by leave-one-section-out mean section
    r (ties to the larger alpha), refit on B; residuals = targets minus its predictions."""
    from src.transcripts import nuisance_features, run_words
    ctx = context(cohort, caches, tmp_path)
    tgt = target(ctx)
    view = tgt.outer(TEST)
    budget = sorted(bud.lpp_section_budgets(TEST, SECTIONS)['L3'])
    bd = rl.LppBudget(view, 'L3', bud.lpp_section_budgets(TEST, SECTIONS)['L3'])
    got = rl.nuisance(view, bd, rl.LppStore(ctx, tgt, ()))
    manifest_path = cohort / 'semantic' / TARGET / 'semantic_manifest.json'
    runs = {r['story']: r for r in json.loads(manifest_path.read_text())['runs']}
    words = {runs[s]['run_id']: run_words(runs[s], manifest_path) for s in budget + [TEST]}
    train = [(s, t) for s in budget for t in range(28, N_TR - 4) if (s, t) != INVALID]
    test_trs = list(range(28, N_TR - 4, 4))
    x, _ = nuisance_features(SimpleNamespace(runs=[runs[s] for s in budget],
                                             items=[(budget.index(s), t) for s, t in train]), words)
    xt, _ = nuisance_features(SimpleNamespace(runs=[runs[TEST]], items=[(0, t) for t in test_trs]), words)
    ys = {s: np.load(cohort / 'semantic' / 'targets' / f'{s}.targets.npy').astype(np.float64) for s in budget + [TEST]}
    y, yt = np.stack([ys[s][t] for s, t in train]), np.stack([ys[TEST][t] for t in test_trs])
    section = np.array([s for s, _ in train])

    def standardize(fit, *xs):
        mean, std = fit.mean(0), fit.std(0)
        std = np.where(std > 1e-8 * max(1.0, np.abs(fit).max()), std, 1.0)
        return [((a - mean) / std).astype(np.float32).astype(np.float64) for a in xs]

    def ridge(z, yy, zt, alpha):
        zm, ym = z.mean(0), yy.mean(0)
        beta = np.linalg.solve((z - zm).T @ (z - zm) + alpha * np.eye(z.shape[1]), (z - zm).T @ (yy - ym))
        return (zt - zm) @ beta + ym
    alphas = view.alphas

    def fit_model(cols):
        scores = np.zeros(len(alphas))
        for s in budget:
            fit, held = section != s, section == s
            zf, zh = standardize(x[fit][:, cols], x[fit][:, cols], x[held][:, cols])
            scores += np.array([feature_r(ridge(zf, y[fit], zh, a), y[held]) for a in alphas]) / len(budget)
        k = int(np.flatnonzero(scores >= scores.max() - 1e-12)[-1])
        z, zt = standardize(x[:, cols], x[:, cols], xt[:, cols])
        return alphas[k], ridge(z, y, z, alphas[k]), ridge(z, y, zt, alphas[k])
    alpha, train, test = fit_model(slice(None))
    assert got['info']['alpha'] == pytest.approx(alpha)
    np.testing.assert_allclose(got['train'], y - train, rtol=0, atol=1e-9)
    np.testing.assert_allclose(got['test'], yt - test, rtol=0, atol=1e-9)
    floors = got['floors']['models']  # §9 nuisance-only decoders: position (3 columns), speech rate (8), both.
    for name, cols in (('position', slice(0, 3)), ('speech_rate', slice(3, 11)), ('both', slice(None))):
        alpha, _, test = fit_model(cols)
        assert floors[name]['estimable'] and floors[name]['alpha'] == pytest.approx(alpha)
        assert floors[name]['test'][rl.VAL] == pytest.approx(feature_r(test, yt), abs=1e-9)
        assert floors[name]['oracle'] == (name != 'position')


def test_ladder_priors_degrade_the_isp_map_with_the_registered_seed(tier1):
    ctx, store, _, _ = tier1
    view = store.tgt.outer(TEST)
    budget = bud.lpp_section_budgets(TEST, SECTIONS)['L3']
    prov = Provenance(TARGET, 'S', budget, [], [TEST])
    refs = rl.LppReferences(view.cfg, view, prov, 'cpu', store)
    base = rl.LppPriors(view, refs, prov, 'cpu', store)
    r, info = base.isp(budget)
    for f in (0.5, 0.0):
        got, linfo = rl.LadderPriors(view, refs, prov, 'cpu', store, f).isp(budget)
        want = np.tanh(ispm.ladder(ebm.fisher(r), f, ispm.ladder_seed(TARGET, f)))
        np.testing.assert_array_equal(got, want)
        assert linfo['ladder'] == f and linfo['alpha'] == info['alpha']
    assert {r_.stimulus for r_ in prov.records if r_.purpose == 'pca_basis'} == set(SECTIONS) - {TEST}


def test_reference_bases_exclude_the_test_section_and_are_shared_across_targets(tier1, tmp_path):
    ctx, store, _, _ = tier1
    entry = store.basis(REFS[0], TEST)
    assert entry['sections'] == [s for s in SECTIONS if s != TEST] and TEST not in entry['proj']
    path = next((ctx.cache_root / 'bases').glob(f'{REFS[0]}_t{TEST}_{entry["key"]}.npz'))
    with np.load(path) as z:
        assert list(z['sections']) == entry['sections']
    other = rl.LppStore(ctx, rl.LppTarget(ctx, REFS[1], 'dense'), [TARGET, REFS[0]])
    assert other.basis(REFS[0], TEST)['key'] == entry['key']  # Bases depend on (reference, t) only.


def test_pac_reads_audio_and_word_timing_without_phones(tier1, monkeypatch):
    from src.priors import acoustic as acm
    ctx, store, _, _ = tier1
    calls = []

    def story_features(story, run, manifest_path, **kw):
        calls.append((story, kw))
        feats = np.random.default_rng(int(story)).normal(size=(N_TR, 36))
        return feats, dict(phone_source='none', story=story)
    monkeypatch.setattr(acm, 'story_features', story_features)
    fresh = rl.LppStore(ctx, store.tgt, REFS)
    view = store.tgt.outer(TEST)
    budget = bud.lpp_section_budgets(TEST, SECTIONS)['L3']
    prov = Provenance(TARGET, 'S', budget, [], [TEST], folds=bud.inner_folds('L3', budget), budget_name='L3',
                      stage='develop')
    bd = rl.LppBudget(view, 'L3', budget)
    pri = rl.LppPriors(view, None, prov, 'cpu', fresh)
    fit = bd.positions(budget[1:])
    with prov.inner_fold(0):
        feats, info = pri.acoustic_features(bd, fit)
    assert sorted(s for s, _ in calls) == sorted(budget[1:]) and all(kw['phones'] is None for _, kw in calls)
    assert feats.shape == (len(bd.rows), 36) and not feats[bd.positions(budget[:1])].any()
    kinds = {(r.stimulus, r.kind, r.purpose) for r in prov.records}
    assert {(s, 'audio', 'own_encoding') for s in budget[1:]} <= kinds
    assert {(s, 'transcript', 'context_reset') for s in budget[1:]} <= kinds
    prov.check()


# ------------------------------------------------------------------------------------ K-subset arms
def test_reference_subsets_follow_the_hash_order_and_nest():
    refs = [f'sub-EN{i:03d}' for i in range(100, 148)]
    for d in range(rl.SUBSET_DRAWS):
        order = sorted(refs, key=lambda r: hashlib.sha256(f'ssltang-v4-lpp-refs:{TARGET}:{d}:{r}'.encode()).hexdigest())
        previous = []
        for k in rl.SUBSET_K['CONFIRM']:
            got = rl.reference_subset(TARGET, refs[::-1], k, d)
            assert got == sorted(order[:k]) and set(previous) <= set(got)
            previous = got
    assert len({tuple(rl.reference_subset(TARGET, refs, 4, d)) for d in range(rl.SUBSET_DRAWS)}) > 1
    assert rl.reference_subset(TARGET, refs, 4, 0) != rl.reference_subset(REFS[0], refs, 4, 0)  # Per target.
    for k in (0, 48):
        with pytest.raises(ValueError):
            rl.reference_subset(TARGET, refs, k, 0)
    assert rl.subset_arms('DEV', 8) == [f'{b}_K{k}_d{j}' for b in ('EB', 'PRIOR') for k in (1, 2, 4) for j in range(5)]
    assert len(rl.subset_arms('CONFIRM', 48)) == 2 * 5 * 5 and set(rl.subset_arms('CONFIRM', 48)) <= set(rl.ARMS)
    assert rl.expand_arms(['W0', 'EB_K', 'W0'], 'DEV', 8) == ['W0'] + [f'EB_K{k}_d{j}' for k in (1, 2, 4)
                                                                      for j in range(5)]
    assert rl.subset_key('PRIOR_K16_d3') == 'K16_d3' and rl.subset_key('EB_f0.5') is None


def test_subset_weights_are_the_base_rule_on_the_subset_map_and_the_engine_is_restored():
    rng = np.random.default_rng(3)
    maps = dict(own={d: np.tanh(rng.normal(scale=0.2, size=V)) for d in DELAYS}, sigma2={d: 0.01 for d in DELAYS},
                isp=np.tanh(rng.normal(scale=0.2, size=V)))
    for base in ('EB', 'PRIOR'):
        want = v4.arm_weights(base, maps, TARGET)
        got = rl.lpp_arm_weights(f'{base}_K2_d1', dict(maps, isp_K2_d1=maps['isp'], isp=None), TARGET)
        assert want.keys() == got.keys()
        for g in v4.GAMMAS:
            for d in DELAYS:
                np.testing.assert_array_equal(want[g][0][d], got[g][0][d])
    assert rl.lpp_arm_weights('EB_K2_d1', maps, TARGET) is None  # No subset map: not estimable here.
    with pytest.raises(RuntimeError):
        with rl.engine_extension():
            assert v4.fold_maps is rl.lpp_fold_maps and v4.arm_weights is rl.lpp_arm_weights
            raise RuntimeError('fit failed')
    assert v4.fold_maps is rl.ENGINE['fold_maps'] and v4.arm_weights is rl.ENGINE['arm_weights']


def test_k_subset_arms_share_eb_maps_and_carry_their_reference_subset(cohort, caches, tier1, tmp_path, monkeypatch):
    calls = []
    original = v4.encoding_scores

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(v4, 'encoding_scores', counting)  # run_v4.fold_maps' own-encoding fits.
    _, _, folder, preds = run_tier1(cohort, caches, tmp_path / 'a', arms=['EB'])
    alone = len(calls)
    subsets = ['EB_K1_d0', 'EB_K2_d3', 'PRIOR_K2_d3']
    _, store, folder2, preds2 = run_tier1(cohort, caches, tmp_path / 'b', arms=['EB'] + subsets)
    assert len(calls) == 2 * alone  # The subsets reuse EB's own-encoding maps in every fold.
    a, b = cells(folder)[f't{TEST}_L3_EB.json'], cells(folder2)[f't{TEST}_L3_EB.json']
    assert a['selection'] == b['selection'] and a['final_weights'] == b['final_weights']
    np.testing.assert_allclose(preds['EB']['test'], preds2['EB']['test'], rtol=1e-12, atol=1e-14)
    got = cells(folder2)
    for arm in subsets:
        cell = got[f't{TEST}_L3_{arm}.json']
        m = rl.SUBSET_ARM.match(arm)
        want = rl.reference_subset(TARGET, REFS, int(m['k']), int(m['draw']))
        assert audit_v4.audit_file(folder2 / 'cells' / f't{TEST}_L3_{arm}.json')['passed']
        assert cell['estimable'] and cell['base_arm'] == m['base'] and cell['lpp_note'] == rl.SUBSET_NOTE
        assert cell['ledger']['references'] == want == cell['reference_subset']['references']
        assert sorted(cell['reference_bases']) == want and cell['reference_subset']['pool'] == list(REFS)
        key = f"isp_{rl.subset_key(arm)}"
        assert cell['final_priors'][key]['references'] == want
        assert all(f['priors'][key]['references'] == want for f in cell['folds'])
        isp_reads = {r[0] for r in reads(cell['provenance']) if r[3] == 'isp_predictor'}
        assert set(want) <= isp_reads
    assert got[f't{TEST}_L3_EB_K1_d0.json']['final_weights']['eb']  # EB_K: EB diagnostics for R_P(K).
    assert got[f't{TEST}_L3_EB_K2_d3.json']['final_priors'] == got[f't{TEST}_L3_PRIOR_K2_d3.json']['final_priors']
    assert v4.fold_maps is rl.ENGINE['fold_maps']


# ------------------------------------------------------------------------------------- comparators
@pytest.fixture(scope='module')
def comparators(cohort, caches, tmp_path_factory):
    out = tmp_path_factory.mktemp('out')
    ctx = context(cohort, caches, out)
    tgt = target(ctx)
    store = rl.LppStore(ctx, tgt, REFS)
    folder = out / TARGET / 'dense' / 'develop'
    preds = rl.run_budget(ctx, tgt.outer(TEST), store, 'L3', list(rl.COMPARATORS + rl.PAIR_ARMS), 'develop', folder)
    return ctx, store, folder, preds


def test_th_sel_and_its_random_control_run_through_the_engine_and_pass_the_audit(comparators):
    ctx, store, folder, preds = comparators
    got = cells(folder)
    budget = bud.lpp_section_budgets(TEST, SECTIONS)['L3']
    unheard = [s for s in SECTIONS if s not in budget and s != TEST]
    th_sel = got[f't{TEST}_L3_TH_SEL.json']
    assert audit_v4.audit_file(folder / 'cells' / f't{TEST}_L3_TH_SEL.json')['passed']
    assert th_sel['estimable'] and th_sel['engine_arm'] == 'PRIOR' and th_sel['unheard'] == unheard
    assert th_sel['provenance']['arm'] == 'TH_SEL' and th_sel['provenance']['reference_rule'] == 'SS'
    assert any(f['rule'] == 'R4' for f in th_sel['provenance']['flags'])  # SS reference transcripts, flagged.
    assert th_sel['ledger']['reference_rule'] == 'SS' and th_sel['ledger']['references'] == list(REFS)
    assert th_sel['ledger']['reference_labeled_pairs'] > 0 and th_sel['final_selection']['k'] == round(0.17 * V)
    assert np.isfinite(th_sel['residualized']['test'][rl.VAL])
    tagged = {(r[0], r[1], r[4]) for r in reads(th_sel['provenance']) if r[3] == 'converter'}
    for i, held in enumerate(budget):  # Converters were refit inside every inner fold from its fitting part.
        fit = [s for s in budget if s != held]
        assert {(TARGET, s, str(i)) for s in fit} <= tagged and (TARGET, held, str(i)) not in tagged
    labels = {r[1] for r in reads(th_sel['provenance']) if r[0] in REFS and r[2] == 'transcript'}
    assert labels == set(SECTIONS) - {TEST}  # SS: every reference section except the target's test section.
    assert set((ctx.cache_root / 'refsub').glob(f'{REFS[0]}_s*_z_*.npy'))  # Reference voxel slices cached.
    rand = got[f't{TEST}_L3_TH_SEL_rand.json']
    assert audit_v4.audit_file(folder / 'cells' / f't{TEST}_L3_TH_SEL_rand.json')['passed']
    assert {r[0] for r in reads(rand['provenance'])} == {TARGET} and rand['ledger']['references'] == []
    assert rand['final_weights']['nonzero_fraction'] == pytest.approx(round(0.17 * V) / V)


def test_xdec_uses_zero_target_labels_and_passes_the_module_h_audit(comparators):
    ctx, store, folder, preds = comparators
    cell = cells(folder)[f't{TEST}_L3_XDEC.json']
    p = Provenance.from_dict(cell['provenance'])
    assert ctx.th.audit_errors(cell, p) == [] and p.folds is None
    assert cell['estimable'] and cell['target_labels'] == 0 and np.isfinite(cell['test'][rl.VAL])
    target_transcripts = {r.purpose for r in p.records if r.subject == TARGET and r.kind != 'brain'}
    assert target_transcripts <= {'decoder_score', 'context_reset'}
    assert cell['ledger']['target_pairs_dense'] == 0 and cell['ledger']['transcript_words'] == 0
    assert preds['XDEC']['test'].shape == (len(range(28, N_TR - 4, 4)), DIM)
    models = list((ctx.cache_root / 'tanghuth' / 'references').glob(f'*_dense_t{TEST}_*.npz'))
    assert len(models) >= len(REFS)  # Cached across targets: a second store reuses them.
    again = rl.LppStore(ctx, store.tgt, REFS)
    _, info = again.reference_model(REFS[0], TEST, 'dense', decoder=True)
    assert info['cached']


def test_two_reference_sensitivity_uses_the_registered_pair_and_tth_sel_alphas(comparators):
    """FUTURE_WORK_V4 §5 TH-SEL step 3: TH_SEL / XDEC with the target's hash-ordered pair, in Module H's sensitivity
    layout (outside cells/), passing Module H's audit; the pair reuses TH_SEL's converter alphas."""
    from src.priors import tanghuth as th
    sv = pytest.importorskip('summarize_v4')
    ctx, store, folder, preds = comparators
    order = sorted(REFS, key=lambda r: hashlib.sha256(f'ssltang-v4-thsel-pair:{TARGET}:{r}'.encode()).hexdigest())
    pair = sorted(order[:2])
    assert store.pair == pair == list(th.sensitivity_pairs(REFS, 'lpp', TARGET)[0])
    sens = folder / 'sensitivity' / ('refs-' + '-'.join(pair))
    names = sorted(p.name for p in (sens / 'cells').glob('*.json'))
    assert names == [f't{TEST}_L3_{a}.json' for a in ('TH_SEL', 'XDEC')]
    assert {'TH_SEL__pair', 'XDEC__pair', 'TH_SEL', 'XDEC'} <= set(preds)
    assert sorted(cells(folder)) == sorted(f't{TEST}_L3_{a}.json' for a in rl.COMPARATORS)  # cells/ untouched.
    main = cells(folder)[f't{TEST}_L3_TH_SEL.json']
    for arm in ('TH_SEL', 'XDEC'):
        path = sens / 'cells' / f't{TEST}_L3_{arm}.json'
        cell = json.loads(path.read_text())
        assert sv.default_audit(path)['passed'] and th.audit_file(path)['passed']
        assert cell['arm'] == arm and cell['references'] == pair and cell['ledger']['references'] == pair
        assert cell['sensitivity']['pair'] == pair and cell['sensitivity']['pool'] == list(REFS)
        assert cell['estimable'] and np.isfinite(cell['test'][rl.VAL]) and 'test_shift_null' in cell
        assert (sens / 'predictions' / f't{TEST}_L3_{arm}.npz').is_file()
    pair_cell = json.loads((sens / 'cells' / f't{TEST}_L3_TH_SEL.json').read_text())
    for ref in pair:  # The final selection's converters come from TH_SEL's memo (same reference, same target).
        got, want = pair_cell['final_selection']['converters'][ref], main['final_selection']['converters'][ref]
        assert got['memo'] is True and got['alpha'] == want['alpha']
    assert np.isfinite(pair_cell['residualized']['test'][rl.VAL])
    rerun = rl.run_budget(ctx, store.tgt.outer(TEST), store, 'L3', list(rl.PAIR_ARMS), 'develop', folder)
    assert rerun == {}  # Current pair cells are skipped like any other.


def test_structurally_not_estimable_comparators_write_no_cell(cohort, caches, tmp_path, capsys):
    """TH_SEL at L8 has no selection material: logged, no cell. A Module H NotEstimable (here XDEC, and the XDEC
    pair) is logged, never fatal, and leaves a not-estimable cell that the primary analysis's QC reads as 'arm not
    estimable' (not a missing file) and that a rerun skips."""
    from src.priors import tanghuth as th
    sv = pytest.importorskip('summarize_v4')

    def broken(*args, **kwargs):
        raise th.NotEstimable('XDEC needs at least one reference decoder')
    stub = SimpleNamespace(**{k: getattr(th, k) for k in dir(th) if not k.startswith('__')})
    stub.xdec_predict = broken
    ctx = context(cohort, caches, tmp_path, tanghuth=stub)
    tgt = target(ctx)
    store = rl.LppStore(ctx, tgt, REFS)
    folder = tmp_path / 'cells'
    arms = ['TH_SEL', 'TH_SEL_rand', 'XDEC', 'XDEC__pair']
    out = rl.run_budget(ctx, tgt.outer(TEST), store, 'L8', arms, 'develop', folder, residualize=False)
    assert set(out) == {'TH_SEL_rand'}
    assert sorted(cells(folder)) == [f't{TEST}_L8_{a}.json' for a in ('TH_SEL_rand', 'XDEC')]
    logged = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith('{')]
    assert {x['arm'] for x in logged if 'not_estimable' in x} == {'TH_SEL', 'XDEC', 'XDEC__pair'}
    pair = sorted(pair_of(TARGET, REFS))
    for path, refs in ((rl.cell_path(folder, TEST, 'L8', 'XDEC'), list(REFS)),
                       (rl.cell_path(rl.pair_folder(folder, pair), TEST, 'L8', 'XDEC'), pair)):
        cell = json.loads(path.read_text())
        assert cell['estimable'] is False and 'reference decoder' in cell['not_estimable'] and cell['scope_errors']
        assert cell['references'] == refs == cell['ledger']['references'] and 'test' not in cell
        expect = dict(subject=TARGET, clock='dense', stage='develop', section=TEST, budget='L8', arm='XDEC')
        problem, _, _ = sv.inspect_cell(path, expect, ['raw'], ctx.code, ctx.config, sv.default_audit,
                                        [TARGET] + list(REFS) if refs == list(REFS) else None)
        assert problem['category'] == 'arm not estimable', problem
    capsys.readouterr()
    assert rl.run_budget(ctx, tgt.outer(TEST), store, 'L8', ['XDEC', 'XDEC__pair'], 'develop', folder) == {}
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['current'] == ['XDEC', 'XDEC__pair']


# ------------------------------------------------------------------------------------------- write-once
def test_cells_are_written_once_and_gates_close_after_the_lock(cohort, caches, tmp_path, capsys):
    out = tmp_path / 'results' / 'v4_lpp'
    ctx = context(cohort, caches, out)
    tgt = target(ctx)
    view, store = tgt.outer(TEST), rl.LppStore(ctx, tgt, REFS)
    folder = out / TARGET / 'dense' / 'develop'
    rl.run_budget(ctx, view, store, 'L3', ['W0'], 'develop', folder)
    path = rl.cell_path(folder, TEST, 'L3', 'W0')
    before, mtime = path.read_bytes(), path.stat().st_mtime_ns
    capsys.readouterr()
    assert rl.run_budget(ctx, view, store, 'L3', ['W0'], 'develop', folder) == {}  # Current: skipped.
    assert path.read_bytes() == before and path.stat().st_mtime_ns == mtime
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['current'] == ['W0']
    other = context(cohort, caches, out, code='edited-lpp-code')  # Stale under other code: recomputed in develop.
    otgt = rl.LppTarget(other, TARGET, 'dense')
    assert 'W0' in rl.run_budget(other, otgt.outer(TEST), rl.LppStore(other, otgt, REFS), 'L3', ['W0'], 'develop',
                                 folder)
    assert json.loads(path.read_text())['code_sha256'] == 'edited-lpp-code'
    evaluate = out / TARGET / 'dense' / 'evaluate'
    rl.save_json(dict(json.loads(before), stage='evaluate', code_sha256='old-code'),
                 rl.cell_path(evaluate, TEST, 'L3', 'W0'))
    with pytest.raises(SystemExit, match='written once'):  # A stale evaluate cell is never replaced.
        rl.run_budget(ctx, view, store, 'L3', ['W0'], 'evaluate', evaluate)
    lock = tmp_path / 'v4_lock.json'
    lock.write_text('{}')
    locked = context(cohort, caches, out, lock_path=lock)
    with pytest.raises(SystemExit, match='exists'):
        rl.check_develop(locked, TARGET, ['W0'], [TARGET], g0=True)
    with pytest.raises(SystemExit, match='exists'):
        rl.stage_g4(locked, [TARGET], [TARGET], [])
    started = rl.started_path(ctx)
    assert started == tmp_path / 'results' / 'v4' / 'primary_started.json'
    started.parent.mkdir(parents=True)
    started.write_text('{}')
    with pytest.raises(SystemExit, match='started'):
        rl.check_confirm(ctx, dict(lpp_mask_sha256=MASK), TARGET, REFS)


def test_evaluate_cells_record_the_lock_and_print_no_score(cohort, caches, tmp_path, capsys):
    ctx = context(cohort, caches, tmp_path)
    ctx.lock_record = dict(sha256='f' * 64, git=None, git_check='skipped (--no-git-check)')
    tgt = target(ctx)
    folder = tmp_path / TARGET / 'dense' / 'evaluate'
    capsys.readouterr()
    rl.run_budget(ctx, tgt.outer(TEST), rl.LppStore(ctx, tgt, REFS), 'L3', ['W0', 'EB'], 'evaluate', folder)
    printed = capsys.readouterr().out
    assert 'estimable=True' in printed and 'test=' not in printed and 'residualized=' not in printed
    assert 'cv=' not in printed and '"test"' not in printed
    for name, cell in cells(folder).items():
        assert cell['stage'] == 'evaluate' and cell['lock'] == ctx.lock_record and np.isfinite(cell['test'][rl.VAL])
        assert audit_v4.audit_file(folder / 'cells' / name)['passed']


# ------------------------------------------------------------------------------------------------ G4
def test_g4b_shift_null_and_verdict(cohort, caches, tmp_path, monkeypatch):
    ctx = context(cohort, caches, tmp_path)
    record = rl.g4_subject_b(ctx, TARGET)
    assert record['passed'] and record['observed'] > record['null_q95']
    assert record['data_sha256'] == ctx.data_sha256([TARGET])
    assert record['shift_null']['draws'] == 2000 and record['shift_null']['min_shift_tr'] == 40
    assert record['shift_null']['runs_used'] == 9
    got = cells(tmp_path / TARGET / 'dense' / 'g4')
    assert sorted(got) == sorted(f't{t}_L8_W0.json' for t in SECTIONS)
    assert np.mean([c['test'][rl.VAL] for c in got.values()]) == pytest.approx(record['observed'])
    assert all(audit_v4.audit_file(tmp_path / TARGET / 'dense' / 'g4' / 'cells' / n)['passed'] for n in got)
    monkeypatch.setattr(rl, 'run_budget', lambda *a, **k: pytest.fail('a recorded G4(b) is reused'))
    assert rl.g4_subject_b(ctx, TARGET) == json.loads((tmp_path / TARGET / 'dense' / 'g4' / 'g4b.json').read_text())
    monkeypatch.setattr(rl.Context, 'data_sha256', lambda self, subjects: 'other data')
    with pytest.raises(pytest.fail.Exception):  # A G4(b) record of other data is recomputed.
        rl.g4_subject_b(ctx, TARGET)
    monkeypatch.undo()
    tgt = target(ctx)
    perfect = {t: tgt.outer(t).split['test'].targets for t in SECTIONS}
    assert rl.shift_record(tgt.outer, perfect)['observed'] == pytest.approx(1.0)
    b = {f's{i}': dict(observed=0.1, null_q95=0.0, passed=i < 6) for i in range(9)}
    assert rl.g4_verdict(b)['b']['passed'] and not rl.g4_verdict(b)['passed']  # (c) not run yet.
    c = {f's{i}': dict(n_fits=19, n_at_edge=0, passed=True) for i in range(9)}
    assert rl.g4_verdict(b, c)['passed']
    assert not rl.g4_verdict(dict(b, s5=dict(b['s5'], passed=False)), c)['passed']  # 5/9
    assert not rl.g4_verdict(b, dict(c, s0=dict(n_fits=19, n_at_edge=1, passed=False)))['passed']


def test_g4b_shift_null_matches_a_numpy_circular_shift(cohort, caches, tmp_path):
    """Independent reference for G4(b): per section, the mean per-dimension r of the predictions rolled by every
    shift of >= 40 TR (10 stride-4 rows) both ways; 2,000 draws with default_rng(0), one uniform shift per section
    in section order; q95 of the mean over the 9 sections."""
    ctx = context(cohort, caches, tmp_path)
    tgt = target(ctx)
    rng = np.random.default_rng(5)
    truth = {t: tgt.outer(t).split['test'].targets.astype(np.float64) for t in SECTIONS}
    preds = {t: truth[t] + rng.normal(scale=2.0, size=truth[t].shape) for t in SECTIONS}
    record = rl.shift_record(tgt.outer, preds)
    observed, shifted = [], []
    for t in SECTIONS:
        n = len(preds[t])
        allowed = [s for s in range(1, n) if min(s, n - s) >= 10]
        observed.append(feature_r(preds[t], truth[t]))
        shifted.append(np.array([feature_r(np.roll(preds[t], s, axis=0), truth[t]) for s in allowed]))
    draw = np.random.default_rng(0)
    null = np.array([np.mean([sh[draw.integers(len(sh))] for sh in shifted]) for _ in range(2000)])
    assert record['observed'] == pytest.approx(np.mean(observed), abs=1e-12)
    assert record['null_q95'] == pytest.approx(np.quantile(null, 0.95), abs=1e-12)
    assert record['passed'] == bool(np.mean(observed) > np.quantile(null, 0.95))


def test_g4c_records_every_isp_alpha_and_feeds_develop(tier1):
    ctx, store, _, _ = tier1
    record = rl.g4_subject_c(ctx, TARGET, REFS, sections=[TEST])  # One outer fold keeps the test fast.
    assert record['n_fits'] == 4 + 6 + 9 and record['passed'] == (record['n_at_edge'] == 0)
    assert record['data_sha256'] == ctx.data_sha256((TARGET,) + REFS)
    assert rl.current_records(ctx, [TARGET], Path('g4c.json'),
                              lambda s: ctx.data_sha256([s] + list(REFS))) is None  # A partial (c) never counts.
    assert {(f['budget'], f['fold']) for f in record['fits']} == {
        (n, i) for n, k in bud.LPP_SIZES.items() for i in list(range(k)) + [None]}
    l3 = [f for f in record['fits'] if f['budget'] == 'L3' and f['fold'] is None][0]
    eb = json.loads((ctx.out_root / TARGET / 'dense' / 'develop' / 'cells' / f't{TEST}_L3_EB.json').read_text())
    assert eb['final_priors']['isp']['alpha'] == l3['alpha']  # Same cached map as develop's.


def test_stage_g4_waits_for_every_dev_listener_and_runs_c_only_after_b(tmp_path, monkeypatch):
    ctx = context(tmp_path, tmp_path, tmp_path)
    monkeypatch.setattr(rl.Context, 'data_sha256', lambda self, subjects: 'data:' + ','.join(sorted(subjects)))
    dev = [f's{i}' for i in range(9)]
    record = json.dumps(dict(g4a=g4a(dev, lpp_prepare.SECTION_TRS)))  # Every DEV listener prepared (G4(a)).
    for folder in (tmp_path, tmp_path / 'f'):
        folder.mkdir(exist_ok=True)
        (folder / 'prepare_provenance.json').write_text(record)
    calls, failing_b = [], set()

    def fake_b(ctx_, s):
        calls.append(('b', s))
        rec = dict(code_sha256=ctx_.code, data_sha256=ctx_.data_sha256([s]), observed=0.2, null_q95=0.1,
                   passed=s != 's8' and s not in failing_b)
        path = ctx_.out_root / s / 'dense' / 'g4' / 'g4b.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec))
        return rec

    def fake_c(ctx_, s, refs):
        calls.append(('c', s))
        rec = dict(code_sha256=ctx_.code, data_sha256=ctx_.data_sha256([s] + list(refs)), n_fits=171, n_at_edge=0,
                   passed=True)
        (ctx_.out_root / s / 'g4c.json').write_text(json.dumps(rec))
        return rec
    monkeypatch.setattr(rl, 'g4_subject_b', fake_b)
    monkeypatch.setattr(rl, 'g4_subject_c', fake_c)
    assert rl.stage_g4(ctx, dev[:4], dev, []) is None and not (tmp_path / 'g4.json').exists()
    assert not [c for c in calls if c[0] == 'c']  # (c) never runs before every (b) is in.
    out = rl.stage_g4(ctx, dev[4:], dev, [])
    assert out is None and [c for c in calls if c[0] == 'c'] == [('c', s) for s in dev[4:]]
    out = rl.stage_g4(ctx, dev[:4], dev, [])
    assert out['passed'] and out['b']['n_passed'] == 8 and out['code_sha256'] == ctx.code
    assert out['data_sha256'] == ctx.data_sha256(dev) and rl.g4_passed(ctx)
    monkeypatch.setattr(rl.Context, 'data_sha256', lambda self, subjects: 're-prepared')
    assert not rl.g4_passed(ctx)  # A G4 verdict counts only on the data it was computed on.
    monkeypatch.setattr(rl.Context, 'data_sha256', lambda self, subjects: 'data:' + ','.join(sorted(subjects)))
    with pytest.raises(SystemExit):
        rl.stage_g4(ctx, ['sub-EN999'], dev, [])
    failing = context(tmp_path / 'f', tmp_path / 'f', tmp_path / 'f')
    failing_b.update(dev[:4])  # 4/9 pass: (b) fails and (c) never runs.
    calls.clear()
    out = rl.stage_g4(failing, dev, dev, [])
    assert not out['passed'] and out['b']['n_passed'] == 4 and out['c'] is None
    assert not [c for c in calls if c[0] == 'c'] and 'W0 CV score only' in out['note']
    assert not rl.g4_passed(failing)


# ---------------------------------------------------------------------------------- gates and lock
def test_develop_order_g0_then_w0_then_g4(tmp_path, monkeypatch):
    ctx = context(tmp_path, tmp_path, tmp_path)
    monkeypatch.setattr(rl.Context, 'data_sha256', lambda self, subjects: 'dev-data')
    dev = [TARGET] + list(REFS)
    with pytest.raises(SystemExit):
        rl.check_develop(ctx, 'sub-EN999', ['W0'], dev, g0=True)  # Not DEV.
    with pytest.raises(SystemExit):
        rl.check_develop(ctx, TARGET, ['W0'], dev, g0=False)  # G0 first.
    rl.check_develop(ctx, TARGET, ['W0'], dev, g0=True)  # W0 before G4.
    with pytest.raises(SystemExit, match='G4'):
        rl.check_develop(ctx, TARGET, ['W0', 'EB'], dev, g0=True)
    good = dict(passed=True, code_sha256=ctx.code, data_sha256='dev-data', dev=dev)
    for bad in (dict(good, code_sha256='other code'), dict(good, data_sha256='old data'), dict(good, dev=[])):
        (tmp_path / 'g4.json').write_text(json.dumps(bad))
        with pytest.raises(SystemExit, match='G4'):
            rl.check_develop(ctx, TARGET, ['EB'], dev, g0=True)
    (tmp_path / 'g4.json').write_text(json.dumps(good))
    rl.check_develop(ctx, TARGET, rl.expand_arms(rl.DEVELOP_DEFAULT, 'DEV', 8), dev, g0=True)


def lock_root(tmp_path):
    root = tmp_path / 'repo'
    for name in rl.LPP_CODE + (rl.ANALYSIS,):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, root / name)
    return root


def test_code_fingerprint_covers_the_lpp_code_and_the_core(tmp_path):
    root = lock_root(tmp_path)
    first = rl.code_sha256(root)
    assert first == rl.code_sha256(root) and first != v4.code_sha256()
    (root / rl.LPP_CODE[1]).write_bytes((root / rl.LPP_CODE[1]).read_bytes() + b'\n')
    assert rl.code_sha256(root) != first
    (root / rl.LPP_CODE[1]).unlink()
    with pytest.raises(FileNotFoundError):
        rl.code_sha256(root)
    digest = hashlib.sha256(v4.code_sha256().encode())
    for name in rl.LPP_CODE:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    assert rl.code_sha256() == digest.hexdigest()


def test_confirm_needs_the_matching_committed_lock(tmp_path, monkeypatch):
    root = lock_root(tmp_path)
    lock = tmp_path / 'v4_lock.json'
    ctx = rl.Context(copy.deepcopy(CFG), 'cpu', root=root, lock_path=lock)
    confirm, dev = ['sub-EN050', 'sub-EN051'], ['sub-EN040']
    with pytest.raises(SystemExit, match='does not exist'):
        rl.check_lock(ctx, confirm[0], git_state=None)
    good = dict(rl.fingerprints(ctx), lpp_partition=dict(dev=dev, confirm=confirm),
                confirmatory_plan=dict(hypotheses=dict(H1={})))
    lock.write_text(json.dumps(good))
    assert rl.check_lock(ctx, confirm[0], git_state=None) == good
    for key in ('code_sha256', 'lpp_cell_code_sha256', 'lpp_runner_sha256', 'tanghuth_sha256', 'summarize_sha256',
                'config_sha256'):
        lock.write_text(json.dumps(dict(good, **{key: '0' * 64})))
        with pytest.raises(SystemExit, match=key):
            rl.check_lock(ctx, confirm[0], git_state=None)
    lock.write_text(json.dumps({k: v for k, v in good.items() if k != 'confirmatory_plan'}))
    with pytest.raises(SystemExit, match='plan'):
        rl.check_lock(ctx, confirm[0], git_state=None)
    lock.write_text(json.dumps(good))
    with pytest.raises(SystemExit, match='CONFIRM'):
        rl.check_lock(ctx, dev[0], git_state=None)
    with pytest.raises(SystemExit, match='lock-sha256'):
        rl.check_lock(ctx, confirm[0], git_state=None, lock_sha256='0' * 64)
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match='pushed'):
        rl.check_lock(ctx, confirm[0], git_state=dict(tracked=True, clean=True, pushed=False, commit='abc'))
    state = dict(tracked=True, clean=True, pushed=True, commit='abc')
    rl.check_lock(ctx, confirm[0], git_state=state, lock_sha256=digest)
    assert ctx.lock_record == dict(path=str(lock), sha256=digest, git=state,
                                   git_check='not checked here (the caller verified git)')
    monkeypatch.setattr(lpp_prepare, 'lock_git_state', lambda path: None)  # No git: refused unless opted out.
    with pytest.raises(SystemExit, match='git is unavailable'):
        rl.check_lock(ctx, confirm[0], lock_sha256=digest)
    rl.check_lock(ctx, confirm[0], lock_sha256=digest, no_git_check=True)
    assert ctx.lock_record['git_check'] == 'skipped (--no-git-check)' and ctx.lock_record['git'] is None
    monkeypatch.setattr(lpp_prepare, 'lock_git_state', lambda path: state)
    rl.check_lock(ctx, confirm[0], lock_sha256=digest)
    assert ctx.lock_record['git_check'] == 'verified'
    (root / rl.ANALYSIS).write_bytes(b'# changed\n')
    with pytest.raises(SystemExit, match='summarize_sha256'):
        rl.check_lock(rl.Context(copy.deepcopy(CFG), 'cpu', root=root, lock_path=lock), confirm[0], git_state=None)


def test_confirm_data_must_be_the_locked_preparation(cohort, tmp_path):
    ctx = context(cohort, tmp_path, tmp_path / 'results' / 'v4_lpp')
    record = hashlib.sha256((cohort / 'prepare_provenance.json').read_bytes()).hexdigest()
    lock = dict(lpp_mask_sha256=MASK, lpp_preparation=dict(sha256=record))
    assert rl.check_confirm(ctx, lock, TARGET, REFS) == ctx.data_sha256((TARGET,) + REFS)
    with pytest.raises(SystemExit, match='mask'):
        rl.check_confirm(ctx, dict(lock, lpp_mask_sha256='b' * 64), TARGET, REFS)
    with pytest.raises(SystemExit, match='DEV record'):
        rl.check_confirm(ctx, dict(lock, lpp_preparation=dict(sha256='c' * 64)), TARGET, REFS)
    with pytest.raises(SystemExit, match='no manifest'):  # A pool member without its manifest: a clean refusal.
        rl.check_confirm(ctx, lock, TARGET, list(REFS) + ['sub-EN999'])


def test_confirm_covers_every_cell_the_locked_plan_reads(cohort, tmp_path):
    sv = pytest.importorskip('summarize_v4')
    prior = rl.plan_cells(dict(confirmatory_plan=sv.plan('PRIOR', False, 'L3')))
    assert set(prior) == {('dense', 'L3')}  # H6 is dropped under the substitution rule.
    assert prior[('dense', 'L3')] == {'PRIOR', 'W1', 'W0', 'EB_SELF', 'TH_SEL', 'ORACLE'} | set(rl.PRIOR_LADDER)
    eb = rl.plan_cells(dict(confirmatory_plan=sv.plan('EB', False, 'L3')))
    assert eb[('dense', 'L8')] == {'EB', 'W1', 'W0', 'ORACLE'}
    assert {f'EB_f{f:g}' for f in v4.LADDER} <= eb[('dense', 'L3')]
    # The default confirm request adds the plan's cells at their own (clock, budget) only.
    arms = rl.expand_arms(rl.CONFIRM_DEFAULT, 'CONFIRM', 48)
    plan = rl.fit_plan(list(rl.CLOCKS), list(rl.BUDGETS), arms, ('dense', 'L3'), extra=prior)
    assert set(rl.PRIOR_LADDER) <= set(plan[('dense', 'L3')])
    assert not set(rl.PRIOR_LADDER) & (set(plan[('dense', 'L5')]) | set(plan[('stride4', 'L3')]))
    assert {'TH_SEL', 'XDEC', 'EB_K16_d4', 'PRIOR_K1_d0'} <= set(plan[('dense', 'L3')])
    assert not set(rl.SCOPED) & (set(plan[('dense', 'L8')]) | set(plan[('stride4', 'L3')]))
    assert set(rl.SCOPED) & set(rl.fit_plan(['stride4'], ['L8'], arms)[('stride4', 'L8')])  # --scope all.
    ctx = context(cohort, tmp_path, tmp_path / 'results' / 'v4_lpp')
    assert rl.uncovered(ctx, TARGET, 'data', prior, plan, SECTIONS) == []
    explicit = rl.fit_plan(['dense'], ['L3'], list(rl.DEFAULT_ARMS), ('dense', 'L3'))
    missing = rl.uncovered(ctx, TARGET, 'data', prior, explicit, SECTIONS)
    assert len(missing) == 4 * 9 and all('PRIOR_f' in m for m in missing)  # Refused: H3 would lose every subject.
    assert len(rl.uncovered(ctx, TARGET, 'data', prior, explicit, ['1', '2'])) == 4 * 2  # Section-parallel runs.
    folder = ctx.out_root / TARGET / 'dense' / 'evaluate'  # Current evaluate cells on disk cover their plan cells.
    for arm in rl.PRIOR_LADDER:
        for t in SECTIONS:
            rl.save_json(dict(code_sha256=ctx.code, config_sha256=ctx.config, data_sha256='data', stage='evaluate'),
                         rl.cell_path(folder, t, 'L3', arm))
    assert rl.uncovered(ctx, TARGET, 'data', prior, explicit, SECTIONS) == []
    assert len(rl.uncovered(ctx, TARGET, 'other data', prior, explicit, SECTIONS)) == 4 * 9


def test_partition_references_and_confirm_caches(tmp_path):
    ids = [f'sub-EN{i:03d}' for i in range(30, 44)]
    dev, confirm = bud.lpp_partition(ids)
    (tmp_path / 'prepare_provenance.json').write_text(json.dumps(dict(partition=dict(dev=dev, confirm=confirm))))
    ctx = rl.Context(copy.deepcopy(CFG), 'cpu', lpp_root=tmp_path, lock_path=tmp_path / 'no_lock.json')
    assert rl.read_partition(ctx) == (list(dev), list(confirm))
    assert rl.references_of(dev[0], dev, confirm) == list(dev[1:])
    assert rl.references_of(confirm[0], dev, confirm) == [s for s in list(dev) + list(confirm) if s != confirm[0]]
    with pytest.raises(SystemExit):
        rl.references_of('sub-EN999', dev, confirm)
    (tmp_path / 'prepare_provenance.json').write_text(json.dumps(dict(partition=dict(dev=confirm[:9],
                                                                                    confirm=dev + confirm[9:]))))
    with pytest.raises(SystemExit):
        rl.read_partition(ctx)
    with pytest.raises(SystemExit, match='CONFIRM'):
        rl.stage_prepare(ctx, [confirm[0]], ['dense'], confirm)


# ------------------------------------------------------------------------------------ stage and readers
def test_stage_fit_writes_the_cells_summarize_reads(cohort, caches, tmp_path):
    """develop over both clocks: cells at results/v4_lpp/<sub>/<clock>/develop/cells/t<t>_<budget>_<arm>.json with
    the raw and residualized blocks the primary analysis reads, each passing the registered audit."""
    summarize = pytest.importorskip('summarize_v4')
    ctx = context(cohort, caches, tmp_path / 'results' / 'v4_lpp')
    plan = rl.fit_plan(list(rl.CLOCKS), ['L3'], ['W0', 'EB', 'TH_SEL_rand'])
    rl.stage_fit(ctx, TARGET, REFS, plan, [TEST], 'develop')
    for clock in rl.CLOCKS:
        for arm in ('W0', 'EB', 'TH_SEL_rand'):
            path = summarize.lpp_cell(tmp_path, TARGET, clock, 'develop', TEST, 'L3', arm)
            cell = json.loads(path.read_text())
            assert cell['clock'] == clock and cell['code_sha256'] == ctx.code
            assert summarize.endpoint_value(cell, 'raw') == cell['test'][rl.VAL]
            assert summarize.endpoint_value(cell, 'residualized') == cell['residualized']['test'][rl.VAL]
            assert audit_v4.audit_file(path)['passed']
    dense = json.loads(summarize.lpp_cell(tmp_path, TARGET, 'dense', 'develop', TEST, 'L3', 'EB').read_text())
    sparse = json.loads(summarize.lpp_cell(tmp_path, TARGET, 'stride4', 'develop', TEST, 'L3', 'EB').read_text())
    assert dense['ledger']['target_pairs_dense'] == sparse['ledger']['target_pairs_dense']
    assert dense['ledger']['target_pairs_stride4'] == sparse['ledger']['target_pairs_stride4']
    assert dense['test']['n_samples'] == sparse['test']['n_samples']  # Both clocks score the stride-4 bins.
    assert dense['final_priors']['isp'] == sparse['final_priors']['isp']  # ISP is clock-independent (cached).


@pytest.fixture(scope='module')
def dev_cells(cohort, caches, tmp_path_factory):
    """Runner-written DEV cells of every cell the EB family reads at L3 (H1-H5; all 9 outer sections, dense
    clock), PRIOR (the fourth control arm), two K-subset arms, the two-reference sensitivity and the target's G4(b)
    cells, under <root>/results/v4_lpp, with the primary cell (dense, L3) carrying the §9 controls and the
    reference pool of ``reference_pool``. H6's L8 cells (8 inner folds with the EB null) are written for the outer
    sections L8_SECTIONS only, to keep the test short: t4 has section 1's picture rows in B, t1 scores the picture
    section."""
    sv = pytest.importorskip('summarize_v4')
    root = tmp_path_factory.mktemp('root')
    ctx = context(cohort, caches, root / 'results' / 'v4_lpp')
    pl = dict(sv.plan('EB', False, 'L3'), cell_stage='develop')
    plan = {k: sorted(v) for k, v in rl.plan_cells(dict(confirmatory_plan=pl)).items()}
    plan[('dense', 'L3')] += ['PRIOR', 'EB_K1_d0', 'EB_K2_d0'] + list(rl.PAIR_ARMS)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sv, 'SECTION_TRS', (N_TR,) * len(SECTIONS))  # The fixture's sections are N_TR long.
        refs, pool = rl.reference_pool(ctx, TARGET, [TARGET] + list(REFS), [])
    assert refs == list(REFS) and pool['dropped'] == {}
    l3 = {k: v for k, v in plan.items() if k[1] == 'L3'}
    rl.stage_fit(ctx, TARGET, refs, l3, SECTIONS, 'develop', pool, ('dense', 'L3'))
    rl.stage_fit(ctx, TARGET, refs, {k: v for k, v in plan.items() if k[1] == 'L8'}, L8_SECTIONS, 'develop', pool,
                 ('dense', 'L3'))
    rl.g4_subject_b(ctx, TARGET)
    return root, ctx, pl, plan


L8_SECTIONS = ['4', '1']


def test_runner_cells_pass_the_primary_qc_the_lock_checks_and_the_rp_fit(dev_cells):
    """The Module I readers on runner-written cells with their real fingerprints: summarize_v4.qc excludes and
    refuses nothing, v4_lock.development_cells locks every DEV and G4 cell, and the R_P(K) fit sees three K
    levels."""
    sv = pytest.importorskip('summarize_v4')
    vl = pytest.importorskip('v4_lock')
    root, ctx, pl, plan = dev_cells
    pool = [TARGET] + list(REFS)
    prepared = {s: {sv.SECTION_TRS} for s in pool}  # Every listener passes the subject-level QC.
    rec = sv.qc(root, [TARGET], pl, ctx.code, prepared, ctx.config, reference_pool=pool)
    assert rec['refusals'] == [] and rec['included'] == [TARGET] and rec['unregistered_exclusions'] == []
    assert all(v['excluded'] == {} and v['included'] == [TARGET] for h, v in rec['samples'].items() if h != 'H6')
    assert rec['oracle_subjects'] == {'L3': [TARGET], 'L8': []}
    reasons = rec['samples']['H6']['excluded'][TARGET]  # H6 (L8): only the sections this test did not run.
    want = {f't{t}_L8_{a}.json' for t in SECTIONS if t not in L8_SECTIONS for a in ('EB', 'W1')}
    assert {r['category'] for r in reasons} == {'missing file'} and {r['detail'] for r in reasons} == want
    for t in L8_SECTIONS:  # The written L8 cells pass every QC check, references included.
        for arm in sorted(plan[('dense', 'L8')]):
            path = sv.lpp_cell(root, TARGET, 'dense', 'develop', t, 'L8', arm)
            expect = dict(subject=TARGET, clock='dense', stage='develop', section=t, budget='L8', arm=arm)
            assert sv.inspect_cell(path, expect, ['raw'], ctx.code, ctx.config, sv.default_audit, pool)[:2] == (
                None, {}), (t, arm)
    eb8 = json.loads(sv.lpp_cell(root, TARGET, 'dense', 'develop', '4', 'L8', 'EB').read_text())
    assert len(eb8['provenance']['folds']) == 8 == len(eb8['folds'])  # Section 1 (picture rows) is in B here.
    assert {f['stimulus'] for v in eb8['provenance']['folds'].values() for f in v} == set(SECTIONS) - {'4'}
    assert all(set(f['priors']['null']) == {str(d) for d in DELAYS} for f in eb8['folds'])  # EB null, 8 folds.
    ds = root / 'results' / 'v4' / 'UTS01' / 'dense' / 'develop' / 'cells' / 'S3_W0.json'  # v4_lock needs one.
    rl.save_json(dict(subject='UTS01', stage='develop', clock='dense', budget='S3', arm='W0', code_sha256=ctx.core,
                      config_sha256=ctx.config), ds)

    def audit(path):  # The real audits for LPP cells; the stand-in ds003020 cell has no log.
        return dict(passed=True) if '/results/v4/' in Path(path).as_posix() else sv.default_audit(path)
    fp = dict(code_sha256=ctx.core, lpp_cell_code_sha256=ctx.code, config_sha256=ctx.config)
    locked = vl.development_cells(root, fp, [TARGET], audit, reference_pool=pool)
    written = {p.relative_to(root).as_posix() for p in (root / 'results' / 'v4_lpp').rglob('cells/*.json')
               if '/sensitivity/' not in p.as_posix()}  # Pair cells: Module H's layout, which v4_lock does not walk.
    assert set(locked['lpp_dev']) | set(locked['lpp_g4']) == written and len(locked['lpp_g4']) == 9
    n_cells = sum(len([a for a in v if a not in rl.PAIR_ARMS]) * (9 if b == 'L3' else len(L8_SECTIONS))
                  for (_, b), v in plan.items())
    assert len(locked['lpp_dev']) == n_cells and locked['lpp_config'] == ctx.config
    table, eb, skipped = sv.lpp_table(root, [TARGET], 'develop', ctx.code, clocks=('dense',))
    assert ('dense', 'L3', 'EB_K1_d0') in table  # Only the partial L8 groups (L8_SECTIONS) are skipped.
    assert skipped and all(x['budget'] == 'L8' and x['reason'].startswith('folds') for x in skipped)
    fit = sv.reliability_fit(eb, 'dense', 'L3')
    assert fit['n_k_levels'] == 3 and set(fit['by_k']) == {'1', '2', '3'} and fit['note'] is None


def test_primary_cell_controls_and_subject_records(dev_cells):
    """FUTURE_WORK_V4 §9 at the primary cell: per section the rotated-target refit of the full pipeline (own audited
    log, embedded in the cell) and the float64 prediction; per subject the within-section shift null of R_s, the
    wrong-section input control and the nuisance floors, checked against independent numpy computations."""
    sv = pytest.importorskip('summarize_v4')
    root, ctx, pl, plan = dev_cells
    folder = ctx.out_root / TARGET / 'dense' / 'develop'
    views = rl.scoring_views(ctx, TARGET)
    truth = {t: views[t].split['test'].targets.astype(np.float64) for t in SECTIONS}
    row_story = rl.LppTarget(ctx, TARGET, 'dense').row_story
    for arm in rl.CONTROL_ARMS:
        cells_ = {t: json.loads(rl.cell_path(folder, t, 'L3', arm).read_text()) for t in SECTIONS}
        preds = {}
        for t, cell in cells_.items():
            block = cell['controls']
            rot = block['rotated_training']
            assert rot['estimable'] and rot['rule'] == rl.ROTATION_RULE and np.isfinite(rot['test'][rl.VAL])
            budget = bud.lpp_section_budgets(t, SECTIONS)['L3']
            assert rot['shift_rows'] == {s: int(np.sum(row_story == s)) // 2 for s in budget}
            log = Provenance.from_dict(rot['provenance'])  # Its own complete, registered-scope log.
            log.check()
            assert audit_v4.scope_errors(cell, log) == [] and log.test_items[0].stimulus == t
            result = audit_v4.audit_file(rl.cell_path(folder, t, 'L3', arm))  # Both logs of the cell re-audited.
            assert result['passed'] and result['logs'] == 2, result
            preds[t] = np.load(folder / block['prediction']['path'])
            assert lpp_prepare.array_digest(preds[t]) == block['prediction']['array_digest']
            saved = np.load(folder / 'predictions' / f't{t}_L3_{arm}.npz')['test']
            np.testing.assert_allclose(preds[t], saved, rtol=1e-2, atol=1e-3)
            assert cell['test'][rl.VAL] == pytest.approx(feature_r(preds[t], truth[t]), abs=1e-10)
            assert cell['test_shift_null']['observed_session_macro'] == pytest.approx(cell['test'][rl.VAL], abs=1e-12)
        record = json.loads((folder / 'controls' / f'L3_{arm}.json').read_text())
        assert record['code_sha256'] == ctx.code and record['data_sha256'] == ctx.data_sha256((TARGET,) + REFS)
        assert set(record['inputs']) == {f'{TARGET}/dense/develop/cells/t{t}_L3_{arm}.json' for t in SECTIONS}
        assert record['observed'] == pytest.approx(np.mean([feature_r(preds[t], truth[t]) for t in SECTIONS]))
        null = record['shift_null']  # Independent numpy: one uniform allowed shift (>= 10 rows) per section.
        shifted = []
        for t in SECTIONS:
            n = len(preds[t])
            allowed = [k for k in range(1, n) if min(k, n - k) >= 10]
            shifted.append(np.array([feature_r(np.roll(preds[t], k, axis=0), truth[t]) for k in allowed]))
        draw = np.random.default_rng(0)
        want = np.array([np.mean([sh[draw.integers(len(sh))] for sh in shifted])
                         for _ in range(CFG['evaluation']['shift_draws'])])
        assert null['null_q95'] == pytest.approx(np.quantile(want, 0.95), abs=1e-12)
        assert null['observed'] == pytest.approx(record['observed'], abs=1e-12) and null['passed']
        wrong = [feature_r(preds[SECTIONS[(i + 1) % 9]][np.arange(len(truth[t])) % len(preds[SECTIONS[(i + 1) % 9]])],
                           truth[t]) for i, t in enumerate(SECTIONS)]
        assert record['wrong_section']['mean'] == pytest.approx(np.mean(wrong), abs=1e-12)
        assert abs(record['wrong_section']['mean']) < 0.5 < record['observed']  # Stimulus-locked signal only.
        rotated = record['rotated_training']
        assert rotated['estimable'] and rotated['mean'] == pytest.approx(np.mean(
            [cells_[t]['controls']['rotated_training']['test'][rl.VAL] for t in SECTIONS]))
        assert rotated['mean'] < record['observed'] - 0.3  # Rotated labels cannot manufacture the signal.
        floors = record['nuisance_floors']
        assert set(floors) == {'position', 'speech_rate', 'both'} and all(
            f['mean'] is not None for f in floors.values())
        assert floors['both']['by_section'] == {t: cells_[t]['nuisance_floors']['models']['both']['test'][rl.VAL]
                                                for t in SECTIONS}
    before = {p: p.read_bytes() for p in (folder / 'controls').glob('*.json')}
    assert rl.subject_controls(ctx, TARGET, 'dense', 'develop', 'L3', ctx.data_sha256((TARGET,) + REFS))
    assert {p: p.read_bytes() for p in (folder / 'controls').glob('*.json')} == before  # Same inputs: kept.
    eb = json.loads(rl.cell_path(folder, '4', 'L3', 'EB').read_text())
    assert eb['reference_pool']['references'] == list(REFS) and eb['reference_pool']['rule'] == rl.POOL_RULE
    assert 'controls' not in json.loads(rl.cell_path(folder, '4', 'L3', 'EB_SELF').read_text())
    assert 'controls' not in json.loads(rl.cell_path(folder, '4', 'L8', 'EB').read_text())  # Not the primary cell.
    assert not sv.audit_problem(rl.cell_path(folder, '4', 'L3', 'EB'), eb, sv.default_audit)


def test_module_i_locks_and_reads_the_pair_cells_and_control_records(dev_cells, tmp_path):
    """Integration (Modules G and I): the runner-written two-reference sensitivity cells and §9 control records are
    locked by v4_lock.development_cells (pair cells: the registered pair of the target's pool; control records: from
    the current cells) and summarized by summarize_v4.lpp_summary; a pair cell with other references or a control
    record whose cells changed refuses the lock."""
    sv = pytest.importorskip('summarize_v4')
    vl = pytest.importorskip('v4_lock')
    src, ctx, pl, plan = dev_cells
    root = tmp_path / 'root'
    shutil.copytree(src / 'results', root / 'results')
    pool = [TARGET] + list(REFS)
    pair = sorted(pair_of(TARGET, REFS))
    folder = root / 'results' / 'v4_lpp' / TARGET / 'dense' / 'develop'
    sens = rl.pair_folder(folder, pair)
    ds = root / 'results' / 'v4' / 'UTS01' / 'dense' / 'develop' / 'cells' / 'S3_W0.json'  # v4_lock needs one.
    rl.save_json(dict(subject='UTS01', stage='develop', clock='dense', budget='S3', arm='W0', code_sha256=ctx.core,
                      config_sha256=ctx.config), ds)

    def audit(path):
        return dict(passed=True) if '/results/v4/' in Path(path).as_posix() else sv.default_audit(path)
    fp = dict(code_sha256=ctx.core, lpp_cell_code_sha256=ctx.code, config_sha256=ctx.config)
    locked = vl.development_cells(root, fp, [TARGET], audit, reference_pool=pool)
    want = {rl.cell_path(sens, t, 'L3', a).relative_to(root).as_posix() for t in SECTIONS for a in ('TH_SEL', 'XDEC')}
    assert set(locked['lpp_dev_sensitivity']) == want and not set(want) & set(locked['lpp_dev'])
    assert set(locked['lpp_dev_controls']) == {(folder / 'controls' / f'L3_{a}.json').relative_to(root).as_posix()
                                               for a in rl.CONTROL_ARMS}
    out = sv.lpp_summary(root, [TARGET], 'develop', ctx.code, None, 'DEV', pools={TARGET: pool})
    assert out['lpp_sensitivity_skipped'] == [] and out['controls_skipped'] == []
    rows = {(r['clock'], r['budget'], r['arm']): r for r in out['lpp_sensitivity']}
    assert set(rows) == {('dense', 'L3', 'TH_SEL'), ('dense', 'L3', 'XDEC')}
    for arm in ('TH_SEL', 'XDEC'):
        r_pair = np.mean([json.loads(rl.cell_path(sens, t, 'L3', arm).read_text())['test'][rl.VAL] for t in SECTIONS])
        row = rows[('dense', 'L3', arm)]
        assert row['mean'] == pytest.approx(r_pair) and row['pairs'] == {TARGET: pair} and 'minus_W1' in row
    r_pool = np.mean([json.loads(rl.cell_path(folder, t, 'L3', 'TH_SEL').read_text())['test'][rl.VAL]
                      for t in SECTIONS])  # The fixture ran XDEC with the pair only (XDEC__pair).
    assert rows[('dense', 'L3', 'TH_SEL')]['minus_whole_pool']['mean'] == pytest.approx(
        rows[('dense', 'L3', 'TH_SEL')]['mean'] - r_pool)
    assert 'minus_whole_pool' not in rows[('dense', 'L3', 'XDEC')]
    controls = {r['arm']: r for r in out['controls']}
    assert set(controls) == set(rl.CONTROL_ARMS)
    record = json.loads((folder / 'controls' / 'L3_EB.json').read_text())
    assert controls['EB']['observed']['mean'] == pytest.approx(record['observed'])
    assert controls['EB']['rotated_training']['mean'] == pytest.approx(record['rotated_training']['mean'])
    text = sv.curves_markdown('LPP DEV', sv.clean(out), sv.LPP_BUDGETS)
    assert '## Two-reference sensitivity' in text and '## §9 controls' in text
    # Another pair than the registered one is refused (and skipped by the reader).
    other = sorted(set(REFS) - {pair[0]})
    moved = rl.pair_folder(folder, other)
    sens.rename(moved)
    for path in (moved / 'cells').glob('*.json'):
        cell = json.loads(path.read_text())
        cell['references'] = other
        cell['ledger']['references'] = other
        path.write_text(json.dumps(cell))
    with pytest.raises(vl.Refused, match='registered pair'):
        vl.development_cells(root, fp, [TARGET], lambda p: dict(passed=True), reference_pool=pool)
    skipped = sv.lpp_sensitivity(root, [TARGET], 'develop', ctx.code, pools={TARGET: pool})[1]
    assert skipped and all('registered' in s['reason'] for s in skipped)
    shutil.rmtree(moved)
    # A control record computed from cells that changed since is refused.
    cell = rl.cell_path(folder, '1', 'L3', 'W0')
    cell.write_text(cell.read_text().replace('"estimable": true', '"estimable": true ', 1))
    with pytest.raises(vl.Refused, match='changed since'):
        vl.development_cells(root, fp, [TARGET], lambda p: dict(passed=True), reference_pool=pool)
    assert {s['arm'] for s in sv.lpp_controls(root, [TARGET], 'develop', ctx.code)[1]} == {'W0'}


def test_rotated_budget_rotates_each_budget_section_by_half(cohort, caches, tmp_path):
    ctx = context(cohort, caches, tmp_path)
    view = target(ctx).outer(TEST)
    bd = rl.LppBudget(view, 'L3', bud.lpp_section_budgets(TEST, SECTIONS)['L3'])
    rot, shifts = rl.rotated_budget(bd)
    for s in bd.stories:
        idx = np.flatnonzero(bd.story == s)
        assert shifts[s] == len(idx) // 2 and len(idx) // 2 >= 40
        np.testing.assert_array_equal(rot.y[idx], np.roll(bd.y[idx], len(idx) // 2, axis=0))
    assert rot.raw is bd.raw and rot.y is not bd.y and not np.array_equal(rot.y, bd.y)
    np.testing.assert_array_equal(np.sort(rot.y, axis=0), np.sort(bd.y, axis=0))


def write_g4a_record(root, name, subjects, trs):
    (Path(root) / name).write_text(json.dumps(dict(g4a=g4a(subjects, trs))))


def test_reference_pool_is_the_primary_qc_pool(tmp_path):
    """The runner's pool is summarize_v4.registered_pool of the preparation records: unprepared listeners and
    listeners with other TR counts are dropped (recorded); a target without prepared data is refused."""
    sv = pytest.importorskip('summarize_v4')
    ids = [f'sub-EN{i:03d}' for i in range(57, 77)]
    dev, confirm = bud.lpp_partition(ids)
    registered = lpp_prepare.SECTION_TRS
    write_g4a_record(tmp_path, 'prepare_provenance.json', dev, registered)
    write_g4a_record(tmp_path, 'prepare_provenance_confirm_1.json', confirm[:7], registered)
    write_g4a_record(tmp_path, 'prepare_provenance_confirm_2.json', confirm[7:9], (300,) + registered[1:])
    ctx = rl.Context(copy.deepcopy(CFG), 'cpu', lpp_root=tmp_path, lock_path=tmp_path / 'no_lock.json')
    target_ = confirm[0]
    refs, record = rl.reference_pool(ctx, target_, dev, confirm)
    pool = sv.registered_pool(list(dev) + list(confirm), sv.prepared_trs(tmp_path))
    assert refs == sorted(s for s in pool if s != target_) == sorted(s for s in dev + confirm[:7] if s != target_)
    assert set(record['dropped']) == set(confirm[7:]) and record['references'] == refs
    assert record['dropped'][confirm[-1]] == 'no preparation record' and record['partition'] == sorted(ids)
    assert record['dropped'][confirm[7]] == [[300] + list(registered[1:])]
    assert rl.reference_pool(ctx, dev[0], dev, confirm)[0] == sorted(dev[1:])  # DEV: the DEV listeners only.
    for bad in (confirm[7], confirm[-1]):  # Other TR counts / no record: the target itself is refused.
        with pytest.raises(SystemExit, match='no prepared data'):
            rl.reference_pool(ctx, bad, dev, confirm)


@pytest.fixture
def confirm_repo(cohort, caches, tmp_path, monkeypatch):
    """A working tree for ``main(['confirm', ...])``: data/lpp = the synthetic cohort, its caches, a committed
    stand-in lock for the EB plan (H6 dropped, so the plan reads L3 only) whose partition makes TARGET a CONFIRM
    listener and holds unprepared listeners (no manifest, no record) in both partitions."""
    sv = pytest.importorskip('summarize_v4')
    monkeypatch.setattr(sv, 'SECTION_TRS', (N_TR,) * len(SECTIONS))
    monkeypatch.setattr(rl, 'load_config', lambda path, sets: copy.deepcopy(CFG))
    monkeypatch.setattr(v3, 'verify_lock', lambda cfg: None)
    monkeypatch.setattr(lpp_prepare, 'lock_git_state', lambda path: None)
    key = lambda s: hashlib.sha256(f'ssltang-v4-lpp-subject:{s}'.encode()).hexdigest()  # noqa: E731
    below = [f'sub-EN{i:03d}' for i in range(200, 400) if key(f'sub-EN{i:03d}') < key(TARGET)][:9]
    above = [f'sub-EN{i:03d}' for i in range(400, 600) if key(f'sub-EN{i:03d}') > key(TARGET)][:1]
    ids = [TARGET] + list(REFS) + below + above
    dev, confirm = bud.lpp_partition(ids)
    assert TARGET in confirm and set(below + above) & set(dev) and set(below + above) & set(confirm)
    (tmp_path / 'data' / 'cache').mkdir(parents=True)
    (tmp_path / 'data' / 'lpp').symlink_to(cohort)
    (tmp_path / 'data' / 'cache' / 'v4_lpp').symlink_to(caches)
    monkeypatch.chdir(tmp_path)
    ctx = rl.Context(copy.deepcopy(CFG), 'cpu')
    pl = sv.plan('EB', True, 'L3')
    lock = dict(rl.fingerprints(ctx), lpp_partition=dict(dev=list(dev), confirm=list(confirm)),
                confirmatory_plan=pl, lpp_mask_sha256=MASK, lpp_config_sha256=ctx.config,
                lpp_preparation=dict(sha256=hashlib.sha256((cohort / 'prepare_provenance.json').read_bytes()
                                                           ).hexdigest()))
    path = tmp_path / 'experiments' / 'v4_lock.json'
    path.parent.mkdir()
    path.write_text(json.dumps(lock))
    return tmp_path, ctx, lock, hashlib.sha256(path.read_bytes()).hexdigest(), below + above


def test_confirm_runs_on_the_prepared_pool_and_writes_every_plan_cell(confirm_repo):
    """main(['confirm', ...]) end to end: unprepared partition members are dropped from the pool (recorded), the
    default arms write every plan cell of the requested section, each passing the primary analysis's QC with
    the registered pool (no 'wrong references'), plus the pair and the controls; an explicit --arms request
    that leaves out a plan cell is refused."""
    sv = pytest.importorskip('summarize_v4')
    root, ctx, lock, digest, unprepared = confirm_repo
    common = ['--subject', TARGET, '--lock-sha256', digest, '--no-git-check', '--device', 'cpu', '--min-free-gb', '0']
    rl.main(['confirm'] + common + ['--sections', TEST, '--clocks', 'dense', '--budgets', 'L3'])
    folder = root / 'results' / 'v4_lpp' / TARGET / 'dense' / 'evaluate'
    need = rl.plan_cells(lock)
    assert set(need) == {('dense', 'L3')}
    prepared = sv.prepared_trs(root / 'data' / 'lpp')
    pool = sv.registered_pool(lock['lpp_partition']['dev'] + lock['lpp_partition']['confirm'], prepared)
    assert sorted(pool) == sorted((TARGET,) + REFS)
    for arm in sorted(need[('dense', 'L3')]):
        path = rl.cell_path(folder, TEST, 'L3', arm)
        cell = json.loads(path.read_text())
        expect = dict(subject=TARGET, clock='dense', stage='evaluate', section=TEST, budget='L3', arm=arm)
        endpoints = sv.required_cells(lock['confirmatory_plan']).get(('L3', arm), ['raw'])
        assert sv.inspect_cell(path, expect, endpoints, lock['lpp_cell_code_sha256'], lock['lpp_config_sha256'],
                               sv.default_audit, pool)[:2] == (None, {}), arm
        assert cell['lock']['sha256'] == digest and cell['lock']['git_check'] == 'skipped (--no-git-check)'
        assert cell['reference_pool']['references'] == list(REFS)
        assert sorted(cell['reference_pool']['dropped']) == sorted(unprepared)
        if sv.reads_references(arm):
            assert cell['ledger']['references'] == list(REFS)
    rec = sv.qc(root, [TARGET], dict(lock['confirmatory_plan']), lock['lpp_cell_code_sha256'],
                {s: {sv.SECTION_TRS} for s in pool}, lock['lpp_config_sha256'],
                reference_pool=lock['lpp_partition']['dev'] + lock['lpp_partition']['confirm'])
    assert rec['refusals'] == [] and rec['reference_pool']['used'] == sorted(pool, key=(
        lock['lpp_partition']['dev'] + lock['lpp_partition']['confirm']).index)
    assert {r['category'] for r in rec['excluded'][TARGET]} == {'missing file'}  # Only the 8 sections not run.
    for arm in rl.CONTROL_ARMS:
        assert 'rotated_training' in json.loads(rl.cell_path(folder, TEST, 'L3', arm).read_text())['controls']
    pair = sorted(pair_of(TARGET, REFS))
    for arm in ('TH_SEL', 'XDEC'):
        assert rl.cell_path(rl.pair_folder(folder, pair), TEST, 'L3', arm).is_file()
    assert {f't{TEST}_L3_{a}.json' for a in ('EB_K1_d0', 'PRIOR_K2_d4', 'TH_SEL', 'XDEC')} <= {
        p.name for p in (folder / 'cells').glob('*.json')}
    with pytest.raises(SystemExit, match='leaves out'):  # Explicit arms that miss plan cells of a new section.
        rl.main(['confirm'] + common + ['--sections', '5', '--clocks', 'dense', '--budgets', 'L3', '--arms', 'W0'])
    assert not list((folder / 'cells').glob('t5_*'))
    rl.main(['confirm'] + common + ['--sections', TEST, '--clocks', 'dense', '--budgets', 'L3', '--arms', 'W0'])


def pair_of(subject, refs):
    order = sorted(refs, key=lambda r: hashlib.sha256(f'ssltang-v4-thsel-pair:{subject}:{r}'.encode()).hexdigest())
    return order[:2]


def test_cli_rejects_unknown_arms_and_stages():
    with pytest.raises(SystemExit):
        rl.main(['develop', '--subject', TARGET, '--arms', 'BOGUS'])
    with pytest.raises(SystemExit):
        rl.main(['evaluate', '--subject', TARGET])
    with pytest.raises(SystemExit):
        rl.main(['confirm', '--subject', TARGET, '--budgets', 'S3'])
    with pytest.raises(SystemExit):  # confirm needs --lock-sha256.
        rl.main(['confirm', '--subject', TARGET])
