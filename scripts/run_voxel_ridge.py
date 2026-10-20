#!/usr/bin/env python3
"""Exploratory voxel-resolution B1 ridge readout, with Phase A cell outputs.

H5 rows are streamed, sanitized as in preprocess.py, and normalized using all
training TRs. Feature z-scoring then uses every eligible training target (no
labels), exactly as B1. One readout is resident at a time: the full FIR design
is about 4.4 GB in float32, never expanded to a full float64 design. CUDA uses
float32 feature matmuls with TF32 disabled and float64 Gram eigendecomposition.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.endpoints import score, shift_null
from src.kernel_ridge import GramRidgePath
from src.probe import Split, code_fingerprint
from src.readout import ALPHAS
from src.runtime import file_sha256, load_config, save_json
from src.targets import PROTOCOL_ID, TargetIndex


@dataclass
class VoxelNormalization:
    mean: np.ndarray
    std: np.ndarray
    n_fit_tr: int
    fit_run_ids: list[str]

    @property
    def n_voxels(self):
        return len(self.mean)


def _source_path(run, raw_root):
    key = Path(run['source_key'])
    if key.is_absolute() or '..' in key.parts:
        raise ValueError(f'H5 source_key must be relative to raw-root: {key}')
    return Path(raw_root) / key


def _checked_data(handle, run, n_voxels=None):
    data = handle['data']
    if data.ndim != 2 or data.shape[0] != int(run['n_tr']) or data.shape[1] < 1:
        raise ValueError(f"H5 shape differs from manifest for {run['run_id']}: {data.shape}")
    if n_voxels is not None and data.shape[1] != n_voxels:
        raise ValueError(f"H5 voxel count differs for {run['run_id']}: {data.shape[1]} != {n_voxels}")
    return data


def _clean_float32(values):
    # Sanitize in the source dtype before conversion, as preprocess.pooled does.
    np.nan_to_num(values, copy=False, nan=0, posinf=0, neginf=0)
    return values.astype(np.float32, copy=False)


def load_voxel_normalization(manifest, raw_root, block_rows=64):
    """Check every source and stream train-only moments over *all* H5 TRs.

    ``manifest`` is a parsed manifest dictionary. Moments accumulate in
    float64 over float32 voxel values (the unpooled analogue of preprocess's
    float32 pooled arrays); saved/applied normalization parameters are float32.
    No held-out values enter these moments.
    """
    if block_rows < 1:
        raise ValueError('block_rows must be positive')
    total = squares = None
    count, n_voxels, fit_ids = 0, None, []
    for run in manifest['runs']:
        with h5py.File(_source_path(run, raw_root), 'r') as handle:
            data = _checked_data(handle, run, n_voxels)
            if n_voxels is None:
                n_voxels = data.shape[1]
                total = np.zeros(n_voxels, np.float64)
                squares = np.zeros(n_voxels, np.float64)
            if run['split'] != 'train':
                continue
            fit_ids.append(run['run_id'])
            for start in range(0, len(data), block_rows):
                block = _clean_float32(data[start:start + block_rows]).astype(np.float64)
                total += block.sum(0)
                np.square(block, out=block)
                squares += block.sum(0)
                count += len(block)
    if count == 0:
        raise ValueError('No training H5 rows for voxel normalization')
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 0))
    std[std < 1e-8] = 1
    return VoxelNormalization(mean.astype(np.float32), std.astype(np.float32), count, fit_ids)


def load_voxel_rows(run, raw_root, normalization, rows=None, block_rows=64):
    """Load only requested H5 rows, sanitized and normalized, in their order.

    Duplicate and unsorted requested rows are supported. No full run is read
    unless ``rows`` is omitted; temporary reads are bounded by ``block_rows``.
    """
    if block_rows < 1:
        raise ValueError('block_rows must be positive')
    rows = np.arange(int(run['n_tr'])) if rows is None else np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= int(run['n_tr'])):
        raise ValueError('Requested H5 rows must lie within this run')
    unique, inverse = np.unique(rows, return_inverse=True)
    values = np.empty((len(unique), normalization.n_voxels), np.float32)
    with h5py.File(_source_path(run, raw_root), 'r') as handle:
        data = _checked_data(handle, run, normalization.n_voxels)
        for start in range(0, len(unique), block_rows):
            selected = unique[start:start + block_rows]
            block = _clean_float32(data[selected])
            block -= normalization.mean
            block /= normalization.std
            values[start:start + len(selected)] = block
    # Avoid duplicating a large sorted-unique selection, the feature-builder case.
    if np.array_equal(rows, unique):
        return values
    return values[inverse]


def voxel_features(index, cutoff, delays, raw_root, normalization):
    """Concatenate normalized H5 rows t+d in TargetIndex order, as B1 does."""
    delays = [int(d) for d in delays]
    index.window_bounds(cutoff)
    if not delays or min(delays) < 0 or max(delays) > int(cutoff):
        raise ValueError('Delays must lie within the response cutoff')
    width = normalization.n_voxels
    out = np.empty((len(index), width * len(delays)), np.float32)
    # Keep at most one run's required rows in RAM, not all 83 voxel time series.
    run_positions = {}
    for position, (ri, t) in enumerate(index.items):
        run_positions.setdefault(ri, []).append((position, t))
    for ri, items in run_positions.items():
        positions, times = np.asarray(items, dtype=np.int64).T
        requested = times[:, None] + np.asarray(delays)[None, :]
        unique, inverse = np.unique(requested, return_inverse=True)
        values = load_voxel_rows(index.runs[ri], raw_root, normalization, unique)
        inverse = inverse.reshape(requested.shape)
        for column in range(len(delays)):
            out[positions, column * width:(column + 1) * width] = values[inverse[:, column]]
        del values
    return out


def standardize_features_inplace(train, *heldout, block_columns=1024, block_rows=32):
    """Standardizer's train-only statistics and threshold, with bounded scratch.

    The resident designs remain float32. Float64 column/row blocks reproduce
    Standardizer's moments and final rounding without a full float64 copy.
    Returns (mean, std) so the fitted scaling is inspectable in small tests.
    """
    if block_columns < 1 or block_rows < 1:
        raise ValueError('Block sizes must be positive')
    arrays = (train, *heldout)
    if any(x.ndim != 2 or x.dtype != np.float32 or x.shape[1] != train.shape[1] for x in arrays):
        raise ValueError('Standardization requires matching float32 2-D designs')
    mean, std = np.empty(train.shape[1]), np.empty(train.shape[1])
    maximum = 0.0
    for start in range(0, train.shape[1], block_columns):
        block = train[:, start:start + block_columns].astype(np.float64)
        mean[start:start + block_columns] = block.mean(0)
        std[start:start + block_columns] = block.std(0)
        maximum = max(maximum, float(np.abs(block).max()))
    std = np.where(std > 1e-8 * max(1.0, maximum), std, 1.0)
    for array in arrays:
        for start in range(0, len(array), block_rows):
            block = array[start:start + block_rows].astype(np.float64)
            block -= mean
            block /= std
            array[start:start + block_rows] = block
    return mean, std


def cell_name(cutoff, readout, budget):
    return f'V1_voxels_seed0_c{cutoff}_{readout}_f{budget}'


def record_cell(out, cutoff, readout, delays, budget, rows, subset, result, train, val, test, cfg):
    """Phase A record() schema, with V1/voxels identity and the same endpoints."""
    ev, data = cfg['evaluation'], cfg['data']
    y = train.targets[rows].astype(np.float64)
    mean, std = y.mean(0), np.maximum(y.std(0), 1e-6)
    name = cell_name(cutoff, readout, budget)
    cell = {
        'protocol': PROTOCOL_ID, 'arm': 'V1', 'features': 'voxels', 'seed': 0,
        'cutoff_tr': cutoff, 'latency_seconds': 2 * cutoff, 'readout': readout,
        'input_support_tr': [f't+{d}' for d in delays], 'fraction': float(budget),
        'labels': {k: v for k, v in subset.items() if k != 'label_ids'},
        'validation_labels': len(val.targets),
        'selection': {k: result[k] for k in ('option', 'alpha', 'score', 'alpha_at_grid_edge', 'n_features')},
        'validation_grid': result['grid'],
        'val': score(result['prediction_val'], val.targets, val.run_ids, val.target_trs,
                     val.sessions, mean, std, ev['exclusions']),
        'test': score(result['prediction_test'], test.targets, test.run_ids, test.target_trs,
                      test.sessions, mean, std, ev['exclusions']),
        'test_shift_null': shift_null(result['prediction_test'], test.targets, test.run_ids,
                                      test.sessions, data['stride'], ev['shift_min_tr'],
                                      ev['shift_draws'], ev['shift_seed']),
        'fit_seconds': result['fit_seconds'],
    }
    # JSON is the completion marker; publish it after the predictions are safe.
    destination = out / 'predictions' / f'{name}.npz'
    temporary = destination.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, val=result['prediction_val'].astype(np.float16),
                            test=result['prediction_test'].astype(np.float16))
    temporary.replace(destination)
    save_json(cell, out / 'cells' / f'{name}.json')
    print(f"{name}: val {result['score']:.4f} test {cell['test']['session_macro_feature_correlation']:.4f} "
          f"[{result['option']}, alpha={result['alpha']:.3g}]", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/v2/phase_a.yaml')
    parser.add_argument('--raw-root', default='data/raw')
    parser.add_argument('--out', default='results/v2/phase_v')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cutoff', type=int, default=3)
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    (out / 'cells').mkdir(parents=True, exist_ok=True)
    (out / 'predictions').mkdir(parents=True, exist_ok=True)
    data, labels = cfg['data'], cfg['labels']
    index = {split: TargetIndex(data['manifest'], split, window=data['window'], stride=data['stride'],
                                phase=data['target_phase'], cutoffs=data['cutoffs'])
             for split in ('train', 'val', 'test')}
    for item in index.values():
        item.window_bounds(args.cutoff)
    train, val, test = (Split(index[s], args.device) for s in ('train', 'val', 'test'))
    budgets, subsets = {}, {}
    for fraction in labels['fractions']:
        budget = f'{fraction:g}'
        budgets[budget], subsets[budget] = index['train'].label_subset(
            fraction, labels['subset_seed'], labels['block_tr'], labels['allocation'])
    readouts = [('endpoint', [args.cutoff]), ('fir', list(range(1, args.cutoff + 1)))]
    pending = {}
    for readout, _ in readouts:
        pending[readout] = {}
        for budget, rows in budgets.items():
            name = cell_name(args.cutoff, readout, budget)
            if (out / 'cells' / f'{name}.json').exists():
                print(f'{name}: skip existing cell', flush=True)
            else:
                pending[readout][budget] = rows
    if not any(pending.values()):
        return
    normalization = load_voxel_normalization(index['train'].manifest, args.raw_root)
    provenance = {
        'protocol': PROTOCOL_ID, 'phase': 'V', 'config': cfg,
        'manifest_sha256': file_sha256(data['manifest']), 'code_sha256': code_fingerprint(),
        'alphas': list(ALPHAS), 'targets': {s: index[s].signature() for s in index},
        'voxel_count': normalization.n_voxels,
        'h5_source_keys': [r['source_key'] for r in index['train'].manifest['runs']],
        'raw_root': str(args.raw_root), 'output_dir': str(out), 'cutoff_tr': args.cutoff,
        'device': args.device, 'exploratory': True,
        'precision': {'design_and_feature_matmul': 'float32', 'gram_eigh_and_prediction': 'float64',
                      'tf32': False, 'float32_eigenvalue_relative_tolerance': 10 * torch.finfo(torch.float32).eps},
        'normalization': {'n_fit_tr': normalization.n_fit_tr, 'fit_run_ids': normalization.fit_run_ids},
    }
    provenance_path = out / 'provenance.json'
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        # Never mix completed cells with a different protocol, source or implementation.
        for key in ('config', 'manifest_sha256', 'code_sha256', 'voxel_count', 'h5_source_keys', 'cutoff_tr'):
            if previous.get(key) != provenance[key]:
                raise ValueError(f'Resume provenance differs at {key}; choose a fresh --out')
    save_json(provenance, provenance_path)
    save_json({s: index[s].ids for s in ('val', 'test')}, out / 'heldout_target_ids.json')
    save_json(subsets, out / 'label_subsets.json')
    np.savez_compressed(out / 'voxel_normalization.npz', mean=normalization.mean, std=normalization.std,
                        fit_run_ids=np.asarray(normalization.fit_run_ids), n_fit_tr=normalization.n_fit_tr)
    print(json.dumps({'voxel_count': normalization.n_voxels,
                      'targets': {s: len(index[s]) for s in index}}), flush=True)
    for readout, delays in readouts:
        if not pending[readout]:
            continue
        features = [voxel_features(index[s], args.cutoff, delays, args.raw_root, normalization)
                    for s in ('train', 'val', 'test')]
        standardize_features_inplace(*features)
        # Transfer once per readout. Budget fitting borrows this full design or
        # gathers only labeled rows; its one centered copy stays in float32.
        x_train, x_val, x_test = [torch.as_tensor(x, device=args.device) for x in features]
        del features
        option = f'delays={delays}'
        for budget, rows in pending[readout].items():
            start = time.monotonic()
            full = len(rows) == len(x_train)
            x_fit = x_train if full else x_train[rows]
            ridge = GramRidgePath(x_fit, train.targets[rows], args.device)
            del x_fit
            projected = ridge.project(x_val)
            scores = val.score_alpha_path(ridge, projected, ALPHAS)
            finite = np.isfinite(scores)
            if not finite.any():
                raise ValueError(f'No finite validation scores for {readout}, budget {budget}')
            best = int(np.nanargmax(np.where(finite, scores, np.nan)))
            alpha = ALPHAS[best]
            result = dict(option=option, alpha=alpha, score=float(scores[best]),
                          alpha_at_grid_edge=best in (0, len(ALPHAS) - 1), n_features=int(x_train.shape[1]),
                          grid={option: scores}, prediction_val=ridge.predict(projected, alpha).cpu().numpy(),
                          prediction_test=ridge.predict(ridge.project(x_test), alpha).cpu().numpy(),
                          fit_seconds=time.monotonic() - start)
            del ridge, projected
            record_cell(out, args.cutoff, readout, delays, budget, rows, subsets[budget],
                        result, train, val, test, cfg)
            del result
        del x_train, x_val, x_test


if __name__ == '__main__':
    main()
