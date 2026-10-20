"""Tiny CPU voxel transfer checks; the GPT-2 encoder is replaced by a fake."""
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from scripts import run_v3_transfer as transfer
from src.evaluation.endpoints import score
from src.multikernel_ridge import KernelRidgePath, combine
from src.runtime import load_config
from src.structured_ridge import BASELINE, coefficients, label
from src.targets import TargetIndex
from src.voxel_kernels import KernelBuilder, cache_delay_rows
from test_ridge_transfer import inputs as ridge_inputs  # Includes a mock FrozenGPT2.


def raw_run(root, run_id, values, dataset='ds004510'):
    key = f'{dataset}/derivative/preprocessed_data/UTS01/{run_id}.hf5'
    path = root / key
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as handle:
        handle['data'] = values
    return dict(run_id=run_id, source_key=key, n_tr=len(values), story=run_id, repeat=1)


def test_pseudo_split_row_identities_and_delay_rows(tmp_path):
    a = np.arange(32, dtype=np.float64).reshape(8, 4)
    a[2, :3] = [np.nan, np.inf, -np.inf]
    b = np.arange(20, dtype=np.float64).reshape(5, 4) + 100
    runs = [raw_run(tmp_path / 'raw', 'a', a), raw_run(tmp_path / 'raw', 'b', b)]
    paths, identities, meta = transfer.cache_pseudo_split(runs, tmp_path / 'raw', tmp_path / 'cache', 4,
                                                        block_rows=2)
    expected_ids = [('a', t) for t in range(5)] + [('b', t) for t in range(2)]
    assert identities == expected_ids
    assert json.loads((tmp_path / 'cache/transfer.meta.json').read_text())['rows'] == [list(x) for x in expected_ids]
    assert meta['n_rows'] == 7
    for d in (1, 2, 3):
        expected = np.stack([dict(a=a, b=b)[run_id][t + d] for run_id, t in identities])
        expected = np.nan_to_num(expected, nan=0, posinf=0, neginf=0).astype(np.float32)
        actual = np.load(paths[d])
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match='voxel count'):
        transfer.cache_pseudo_split(runs, tmp_path / 'raw', tmp_path / 'bad', 5)
    with pytest.raises(ValueError, match='relative'):
        transfer.cache_pseudo_split([dict(runs[0], source_key='../escape.hf5')], tmp_path / 'raw', tmp_path / 'bad', 4)


def test_pseudo_split_cross_kernel_and_prediction_match_direct_numpy(tmp_path):
    rng = np.random.default_rng(31)
    runs = [raw_run(tmp_path / 'raw', name, rng.normal(size=(n, 6)))
            for name, n in [('train', 10), ('transfer', 8)]]
    train, _, _ = transfer.cache_pseudo_split(runs[:1], tmp_path / 'raw', tmp_path / 'cache', 6, 'train')
    pseudo, _, _ = transfer.cache_pseudo_split(runs[1:], tmp_path / 'raw', tmp_path / 'cache', 6)
    membership = np.array([0, 0, 1, 1, 1, 2])
    groups = dict(membership=membership, sizes=np.bincount(membership), n_groups=3)
    builder = KernelBuilder(dict(train=train, transfer=pseudo), groups, 'cpu', chunk=2)
    kernels, cross, _ = builder.build(list(range(7)), ('transfer',))
    q = np.zeros((6, 3))
    q[np.arange(6), membership] = 1 / np.sqrt(groups['sizes'][membership])
    candidate = (.1, (1, 2, 1))
    direct_train, direct_transfer = [], []
    for d, weight in zip((1, 2, 3), candidate[1]):
        x, e = np.load(train[d]).astype(np.float64), np.load(pseudo[d]).astype(np.float64)
        mean, std = x.mean(0), x.std(0)
        x, e = (x - mean) / std, (e - mean) / std
        c, ce = x @ q, e @ q
        r, re = x - c @ q.T, e - ce @ q.T
        for key, a, b in [('C', c, ce), ('R', r, re)]:
            np.testing.assert_allclose(kernels[(key, d)], a @ a.T, atol=1e-12)
            np.testing.assert_allclose(cross['transfer'][(key, d)], b @ a.T, atol=1e-12)
        direct_train.extend([np.sqrt(weight) * c, np.sqrt(weight * candidate[0]) * r])
        direct_transfer.extend([np.sqrt(weight) * ce, np.sqrt(weight * candidate[0]) * re])
    x, e = np.concatenate(direct_train, 1), np.concatenate(direct_transfer, 1)
    coef = coefficients(*candidate)
    np.testing.assert_allclose(combine(cross['transfer'], coef), e @ x.T, atol=1e-12)
    y, alpha = rng.normal(size=(len(x), 4)), 7.0
    path = KernelRidgePath(combine(kernels, coef), y, 'cpu')
    actual = path.predict(path.project(combine(cross['transfer'], coef)), alpha).numpy()
    xc, ec, yc = x - x.mean(0), e - x.mean(0), y - y.mean(0)
    expected = ec @ xc.T @ np.linalg.solve(xc @ xc.T + alpha * np.eye(len(x)), yc) + y.mean(0)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_validation_consistency_aligns_identities_and_rejects_misaligned_rows():
    rng = np.random.default_rng(32)
    pseudo = rng.normal(size=(12, 4))
    identities = [('a', t) for t in range(6)] + [('b', t) for t in range(6)]
    rows = [1, 3, 5, 6, 8, 10]
    standard = pseudo[rows]
    validation = SimpleNamespace(targets=standard + rng.normal(scale=.1, size=standard.shape),
        run_ids=np.array(['a'] * 3 + ['b'] * 3), target_trs=np.array([1, 3, 5, 0, 2, 4]),
        sessions={'a': 's1', 'b': 's2'})
    mean, std = np.zeros(4), np.ones(4)
    expected_r = score(standard, validation.targets, validation.run_ids, validation.target_trs,
                       validation.sessions, mean, std)['session_macro_feature_correlation']
    # Permuting the cache together with its row IDs is harmless.
    order = rng.permutation(12)
    check = transfer.validation_consistency(standard, pseudo[order], [identities[i] for i in order],
                                             validation, expected_r, mean, std)
    assert check['passed'] and check['prediction_relative_rms_error'] == 0
    with pytest.raises(ValueError, match='prediction consistency failed'):
        transfer.validation_consistency(standard, np.roll(pseudo, 1, axis=0), identities,
                                        validation, expected_r, mean, std)
    with pytest.raises(ValueError, match='dense-cell consistency failed'):
        transfer.validation_consistency(standard, pseudo, identities, validation, expected_r + .001, mean, std)


def test_run_vectors_mean_over_all_eligible_t():
    predictions = np.arange(28, dtype=np.float64).reshape(7, 4)
    identities = [('a', t) for t in range(5)] + [('b', t) for t in range(2)]
    runs = [dict(run_id='a', story='a', source_key='a.hf5', n_tr=8),
            dict(run_id='b', story='b', source_key='b.hf5', n_tr=5),
            dict(run_id='short', story='short', source_key='short.hf5', n_tr=3)]
    vectors = transfer.mean_run_predictions(predictions, identities, runs)
    np.testing.assert_array_equal(vectors['a'][0], predictions[:5].mean(0).astype(np.float32))
    np.testing.assert_array_equal(vectors['b'][0], predictions[5:].mean(0).astype(np.float32))
    assert vectors['a'][1]['target_trs_averaged'] == list(range(5))
    assert vectors['short'][0] is None
    predictions[2] = np.nan
    vectors = transfer.mean_run_predictions(predictions, identities, runs)
    np.testing.assert_array_equal(vectors['a'][0], predictions[[0, 1, 3, 4]].mean(0).astype(np.float32))
    assert vectors['a'][1]['skipped_nonfinite_prediction_trs'] == [2]
    assert vectors['a'][1]['n_tr_averaged'] == 4


@pytest.fixture
def dense_inputs(ridge_inputs, tmp_path):
    inputs = ridge_inputs
    rng = np.random.default_rng(33)
    raw = tmp_path / 'raw'
    for manifest_path, dataset in [(inputs['training_path'], 'ds003020'),
                                    (inputs['transfer_path'], 'ds004510')]:
        document = json.loads(manifest_path.read_text())
        for run in document['runs']:
            if run['run_id'] == 'sintel':
                continue  # It must never need an H5 or neural array.
            source = raw_run(raw, run['run_id'], rng.normal(size=(run['n_tr'], 3)), dataset)
            run['source_key'] = source['source_key']
        manifest_path.write_text(json.dumps(document))
    cfg = load_config('configs/v3/structured_ridge.yaml', [])
    cfg.update(raw_root=str(raw), cache_root=str(tmp_path / 'cache'), output_root=str(tmp_path / 'cells'),
               lock_file=str(tmp_path / 'lock.json'))
    cfg['data']['manifest'] = str(inputs['training_path'])
    lock = json.loads(Path('experiments/v3_lock.json').read_text())
    lock['config_sha256'] = transfer.v3.config_sha256(cfg)
    Path(cfg['lock_file']).write_text(json.dumps(lock))
    for split in ('val', 'test'):
        index = TargetIndex(inputs['training_path'], split, window=32, stride=4, phase=0, cutoffs=(3, 4))
        cache_delay_rows(index, raw, Path(cfg['cache_root']) / 'UTS01')
    ctx = transfer.DenseContext(cfg, 'cpu')
    kernels, cross, _ = ctx.builder.build(list(range(len(ctx.index['train']))), ('val',))
    cell_root = Path(cfg['output_root']) / 'UTS01'
    (cell_root / 'dense').mkdir(parents=True)
    (cell_root / 'evaluate/cells').mkdir(parents=True)
    structure = (.1, (1, 2, 1))
    (cell_root / 'evaluate/cells/V4_f1.json').write_text(json.dumps(dict(protocol='ssltang-v3',
        stage='evaluate', subject='UTS01', arm='V4', selection=dict(candidate=structure))))
    mean = ctx.split['train'].targets.astype(np.float64).mean(0)
    std = np.maximum(ctx.split['train'].targets.std(0), 1e-6)
    for arm, candidate in [('V0', BASELINE), ('V4', structure)]:
        coef, alpha = coefficients(*candidate), 10.0
        path = KernelRidgePath(combine(kernels, coef), ctx.split['train'].targets, 'cpu')
        prediction = path.predict(path.project(combine(cross['val'], coef)), alpha).numpy()
        sp = ctx.split['val']
        val = score(prediction, sp.targets, sp.run_ids, sp.target_trs, sp.sessions, mean, std)
        (cell_root / f'dense/{arm}_dense.json').write_text(json.dumps(dict(protocol='ssltang-v3',
            stage='dense', subject='UTS01', arm=arm, candidate=label(candidate), alpha=alpha,
            n_train_targets=len(ctx.index['train']), val=val)))
    inputs.update(config=cfg, mean=mean)
    return inputs


def test_end_to_end_fixed_parameters_variants_cache_and_artifacts(dense_inputs, tmp_path, monkeypatch):
    calls, original = [], transfer.retrieval_summary

    def summary(predictions, references, runs, candidates):
        calls.append((np.array(predictions), np.array(references), candidates))
        return original(predictions, references, runs, candidates)

    monkeypatch.setattr(transfer, 'retrieval_summary', summary)
    out = tmp_path / 'out'
    destination = transfer.run_evaluation(dense_inputs['config'], dense_inputs['transfer_path'], out, 'cpu')
    result = json.loads(destination.read_text())
    assert result['protocol'] == 'ssltang-v3-voxel-transfer' and result['exploratory'] is True
    assert 'v2' in result['exploratory_explanation']
    assert result['provenance']['lock_code_sha256'] == transfer.v3.code_sha256()
    assert len(calls) == 12  # Two arms, two conditions, three variants.
    for i in range(0, 12, 3):
        raw, constant, centered = calls[i:i + 3]
        np.testing.assert_array_equal(constant[0], np.broadcast_to(dense_inputs['mean'], raw[0].shape))
        np.testing.assert_allclose(centered[0], raw[0] - dense_inputs['mean'])
        np.testing.assert_allclose(centered[1], raw[1] - dense_inputs['mean'])
        assert raw[2] == constant[2] == centered[2]
    for arm in ('V0', 'V4'):
        assert result['consistency_check'][arm]['passed']
        decoder = result['decoders'][arm]
        assert decoder['hyperparameters']['alpha'] == 10.0
        for condition, candidates in transfer.CANDIDATES.items():
            report = decoder['conditions'][condition]
            assert report['status'] == 'scored'
            assert report['raw']['candidate_ids'] == list(candidates)
            assert all(run['n_tr_averaged'] == 5 for run in report['runs'])
            assert report['raw']['permutation']['n_permutations'] == (120 if condition == 'imagined_speech' else 6)
    with np.load(out / 'predictions.npz', allow_pickle=False) as saved:
        assert saved['predictions'].shape == (26, 768) and saved['predictions'].dtype == np.float16
        assert set(saved['decoders']) == {'V0', 'V4'}
        assert np.all(saved['n_tr_averaged'] == 5)
    assert len(dense_inputs['factory_calls']) == 1
    transfer.run_evaluation(dense_inputs['config'], dense_inputs['transfer_path'], out, 'cpu')
    assert len(dense_inputs['factory_calls']) == 1  # Whole-reference content-addressed caching.


def test_failed_consistency_publishes_no_scores(dense_inputs, tmp_path, monkeypatch):
    original = transfer.cache_pseudo_split

    def misaligned(*args, **kwargs):
        paths, identities, meta = original(*args, **kwargs)
        if meta['split'] == 'validation_pseudo':
            for path in paths.values():
                values = np.load(path)
                np.save(path, np.roll(values, 1, axis=0))  # Keep IDs, break neural alignment.
        return paths, identities, meta

    def forbidden_score(*args, **kwargs):
        pytest.fail('Retrieval must never run after a failed validation gate')

    monkeypatch.setattr(transfer, 'cache_pseudo_split', misaligned)
    monkeypatch.setattr(transfer, 'retrieval_summary', forbidden_score)
    out = tmp_path / 'bad_out'
    with pytest.raises(ValueError, match='prediction consistency failed'):
        transfer.run_evaluation(dense_inputs['config'], dense_inputs['transfer_path'], out, 'cpu')
    assert not (out / 'results.json').exists()
    assert not (out / 'predictions.npz').exists()


def test_missing_candidate_preserves_fixed_pool_and_skips_condition_h5(dense_inputs, tmp_path, monkeypatch):
    path = dense_inputs['transfer_path']
    document = json.loads(path.read_text())
    missing = transfer.CANDIDATES['imagined_speech'][0]
    document['runs'] = [r for r in document['runs'] if r['story'] != missing]
    # Delete the remaining imagined H5s: no missing-candidate condition may open them.
    for run in document['runs']:
        if run['condition'] == 'imagined_speech':
            transfer._h5_path(run, dense_inputs['config']['raw_root']).unlink()
    path.write_text(json.dumps(document))
    result = json.loads(transfer.run_evaluation(dense_inputs['config'], path, tmp_path / 'missing', 'cpu').read_text())
    for arm in ('V0', 'V4'):
        condition = result['decoders'][arm]['conditions']['imagined_speech']
        assert condition['status'] == 'unscored_missing_fixed_candidates'
        assert condition['raw'] is None
        assert condition['candidate_ids'] == list(transfer.CANDIDATES['imagined_speech'])
        assert condition['top1_chance'] == .2
