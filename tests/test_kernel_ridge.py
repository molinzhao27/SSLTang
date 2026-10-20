"""CPU reference, H5 alignment and output/resume checks for voxel ridge."""
import json

import h5py
import numpy as np
import pytest
import torch
import yaml

from scripts.run_voxel_ridge import (
    load_voxel_normalization, load_voxel_rows, main, standardize_features_inplace, voxel_features,
)
from src.kernel_ridge import GramRidgePath
from src.readout import ALPHAS, RidgePath, Standardizer
from src.targets import TargetIndex


@pytest.mark.parametrize('n,p', [(13, 31), (31, 13)])
@pytest.mark.parametrize('tensor', [False, True])
def test_gram_ridge_matches_svd_float64(n, p, tensor):
    rng = np.random.default_rng(101)
    x, y, z = rng.normal(size=(n, p)), rng.normal(size=(n, 4)), rng.normal(size=(7, p))
    # Nonzero feature/target means exercise the unpenalized intercept.
    x += np.arange(p)
    z += np.arange(p)
    y += 12
    original = x.copy()
    reference = RidgePath(x, y, 'cpu')
    ridge = GramRidgePath(torch.from_numpy(x) if tensor else x,
                          torch.from_numpy(y) if tensor else y, 'cpu')
    projected = ridge.project(torch.from_numpy(z) if tensor else z)
    assert ridge.s.dtype == torch.float64
    for alpha in ALPHAS:
        np.testing.assert_allclose(ridge.predict(projected, alpha).numpy(),
                                   reference.predict(reference.project(z), alpha).numpy(),
                                   rtol=1e-11, atol=1e-11)
    np.testing.assert_array_equal(x, original)
    torch.testing.assert_close(ridge.x_mean, reference.x_mean)
    torch.testing.assert_close(ridge.y_mean, reference.y_mean)


def test_float32_design_stays_float32_and_matches_reference():
    rng = np.random.default_rng(12)
    x = torch.from_numpy(rng.normal(size=(17, 70)).astype(np.float32))
    y = rng.normal(size=(17, 3)).astype(np.float32)
    z = rng.normal(size=(9, 70)).astype(np.float32)
    original = x.clone()
    ridge = GramRidgePath(x, y, 'cpu')
    reference = RidgePath(x, y, 'cpu')
    assert ridge._xc.dtype == ridge.x_mean.dtype == torch.float32
    assert ridge._u.dtype == ridge.s.dtype == torch.float64
    assert len(ridge.s) == len(x) - 1
    np.testing.assert_allclose(ridge.predict(ridge.project(z), 0.1).numpy(),
                               reference.predict(reference.project(z), 0.1).numpy(),
                               rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


def test_degenerate_and_invalid_designs():
    x, y = np.ones((4, 7)), np.arange(12).reshape(4, 3)
    ridge = GramRidgePath(x, y, 'cpu')
    assert ridge.s.numel() == 0
    np.testing.assert_array_equal(ridge.predict(ridge.project(x), 1).numpy(),
                                  np.broadcast_to(y.mean(0), y.shape))
    for bad_x, bad_y in [(x[:1], y[:1]), (x, y[:3]), (x[0], y)]:
        with pytest.raises(ValueError, match='paired 2-D'):
            GramRidgePath(bad_x, bad_y, 'cpu')
    with pytest.raises(ValueError, match='Evaluation features'):
        ridge.project(np.ones((4, 6)))


def voxel_manifest(root):
    """The tests/test_v2.py manifest layout, with raw H5 source keys added."""
    rng = np.random.default_rng(4)
    runs, raw = [], {}
    raw_root = root / 'raw'
    for i, (length, split) in enumerate(zip((41, 38, 45, 40, 44, 39),
                                          ('train', 'train', 'val', 'val', 'test', 'test'))):
        run_id = f'run{i}'
        # Pooled dimension deliberately differs from the voxel count.
        np.save(root / f'{run_id}.npy', rng.normal(size=(length, 2)).astype(np.float32))
        np.save(root / f'{run_id}.targets.npy', rng.normal(size=(length, 3)).astype(np.float32))
        valid = np.ones(length, dtype=bool)
        valid[length // 2] = False
        np.save(root / f'{run_id}.valid.npy', valid)
        key = f'ds003020/derivatives/preprocessed_data/UTS01/{run_id}.hf5'
        path = raw_root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        values = rng.normal(size=(length, 5)) + (0 if split == 'train' else 20)
        values[:, -1] = 6  # Constant voxels must use unit std.
        values[0, :3] = [np.nan, np.inf, -np.inf]
        raw[run_id] = values
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('data', data=values)
        runs.append(dict(run_id=run_id, dataset='ds003020', subject='UTS01', story=run_id,
                         session=f's{i // 2}', split=split, n_tr=length, n_features=2,
                         neural_path=str(root / f'{run_id}.npy'),
                         target_path=str(root / f'{run_id}.targets.npy'),
                         target_valid_path=str(root / f'{run_id}.valid.npy'), source_key=key))
    path = root / 'manifest.json'
    path.write_text(json.dumps(dict(runs=runs)))
    return path, raw_root, raw


@pytest.mark.parametrize('delays', [[3], [1, 2, 3]])
def test_h5_feature_alignment_and_train_only_normalization(tmp_path, delays):
    path, raw_root, raw = voxel_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    norm = load_voxel_normalization(manifest, raw_root, block_rows=3)
    train_raw = np.concatenate([np.nan_to_num(raw[r['run_id']], nan=0, posinf=0, neginf=0)
                                .astype(np.float32) for r in manifest['runs'] if r['split'] == 'train'])
    expected_mean = train_raw.astype(np.float64).mean(0).astype(np.float32)
    expected_std = train_raw.astype(np.float64).std(0)
    expected_std[expected_std < 1e-8] = 1
    np.testing.assert_allclose(norm.mean, expected_mean, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(norm.std, expected_std.astype(np.float32), rtol=1e-6, atol=1e-7)
    assert norm.n_fit_tr == 79 and norm.fit_run_ids == ['run0', 'run1']
    assert norm.mean.dtype == norm.std.dtype == np.float32 and norm.std[-1] == 1
    arrays = []
    for split in ('train', 'val', 'test'):
        index = TargetIndex(path, split, window=8, stride=2, cutoffs=(3, 4))
        features = voxel_features(index, 3, delays, raw_root, norm)
        expected = []
        for ri, t in index.items:
            clean = np.nan_to_num(raw[index.runs[ri]['run_id']], nan=0, posinf=0, neginf=0).astype(np.float32)
            clean = (clean - norm.mean) / norm.std
            expected.append(np.concatenate([clean[t + d] for d in delays]))
        np.testing.assert_array_equal(features, np.asarray(expected))
        assert features.dtype == np.float32 and features.shape[1] == 5 * len(delays)
        arrays.append(features)
    reference = Standardizer(arrays[0])
    expected = [reference(x) for x in arrays]
    mean, std = standardize_features_inplace(*arrays, block_columns=2, block_rows=3)
    np.testing.assert_allclose(mean, reference.mean, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(std, reference.std, rtol=1e-14, atol=1e-14)
    for actual, wanted in zip(arrays, expected):
        np.testing.assert_array_equal(actual, wanted)
    index = TargetIndex(path, 'train', window=8, stride=2, cutoffs=(3, 4))
    with pytest.raises(ValueError, match='Delays'):
        voxel_features(index, 3, [4], raw_root, norm)


def test_loader_unsorted_rows_and_shape_checks(tmp_path):
    path, raw_root, raw = voxel_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    norm = load_voxel_normalization(manifest, raw_root)
    run = manifest['runs'][0]
    rows = [4, 0, 4, 2]
    actual = load_voxel_rows(run, raw_root, norm, rows, block_rows=2)
    clean = np.nan_to_num(raw[run['run_id']], nan=0, posinf=0, neginf=0).astype(np.float32)
    np.testing.assert_array_equal(actual, ((clean - norm.mean) / norm.std)[rows])
    with pytest.raises(ValueError, match='within this run'):
        load_voxel_rows(run, raw_root, norm, [-1])
    with pytest.raises(ValueError, match='H5 shape differs'):
        load_voxel_rows(dict(run, n_tr=run['n_tr'] + 1), raw_root, norm)
    with h5py.File(raw_root / manifest['runs'][-1]['source_key'], 'w') as handle:
        handle.create_dataset('data', data=np.zeros((39, 4)))
    with pytest.raises(ValueError, match='voxel count differs'):
        load_voxel_normalization(manifest, raw_root)


def test_standardizer_global_threshold():
    # Standardizer uses the *global* maximum, even for a low-amplitude block.
    train = np.array([[1e8, 1], [-1e8, -1], [1e8, 0]], dtype=np.float32)
    expected = Standardizer(train)(train)
    standardize_features_inplace(train, block_columns=1, block_rows=1)
    np.testing.assert_array_equal(train, expected)


def test_tiny_runner_schema_predictions_and_resume(tmp_path, capsys):
    path, raw_root, _ = voxel_manifest(tmp_path)
    output = tmp_path / 'out'
    cfg = dict(data=dict(manifest=str(path), window=8, stride=2, target_phase=0, cutoffs=[3, 4]),
               labels=dict(fractions=[1.0, 0.25], subset_seed=2718, block_tr=64, allocation='block'),
               evaluation=dict(exclusions=[2, 4], shift_min_tr=4, shift_draws=10, shift_seed=0))
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    args = ['--config', str(config), '--raw-root', str(raw_root), '--out', str(output), '--device', 'cpu']
    main(args)
    required = {'protocol', 'arm', 'features', 'seed', 'cutoff_tr', 'latency_seconds', 'readout',
                'input_support_tr', 'fraction', 'labels', 'validation_labels', 'selection',
                'validation_grid', 'val', 'test', 'test_shift_null', 'fit_seconds'}
    cells = sorted((output / 'cells').glob('*.json'))
    assert len(cells) == 4
    before = {cell.name: cell.read_bytes() for cell in cells}
    for cell in cells:
        result = json.loads(cell.read_text())
        assert set(result) == required
        assert (result['arm'], result['features']) == ('V1', 'voxels')
        assert result['selection']['n_features'] == (15 if result['readout'] == 'fir' else 5)
        assert result['selection']['alpha'] in ALPHAS
        assert len(next(iter(result['validation_grid'].values()))) == len(ALPHAS)
        assert result['test_shift_null']['draws'] == 10
        with np.load(output / 'predictions' / f'{cell.stem}.npz') as prediction:
            assert set(prediction.files) == {'val', 'test'}
            for split in ('val', 'test'):
                assert prediction[split].dtype == np.float16
                assert prediction[split].shape == (result[split]['n_samples'], 3)
    provenance = json.loads((output / 'provenance.json').read_text())
    assert provenance['config'] == cfg and provenance['voxel_count'] == 5
    assert len(provenance['h5_source_keys']) == 6
    assert len(provenance['manifest_sha256']) == len(provenance['code_sha256']) == 64
    # Exercise a genuinely partial resume before testing the all-complete exit.
    missing = cells[-1]
    missing.unlink()
    capsys.readouterr()
    main(args)
    assert capsys.readouterr().out.count('skip existing cell') == 3
    for cell in cells[:-1]:
        assert cell.read_bytes() == before[cell.name]
    main(args)
    assert capsys.readouterr().out.count('skip existing cell') == 4
