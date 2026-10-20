"""All-voxel delay designs and coarse/detail base kernels for v3 structured ridge.

Raw H5 rows are cached once per subject (float32, non-finite values set to 0 as
in preprocessing). Each kernel set z-scores every (voxel, delay) column with the
statistics of a declared row set, splits it into scaled spatial-group means
``C = X Q`` and residual detail ``R = X - C Q^T``, and forms float64 Gram
matrices directly: K_X = K_C + K_R holds without subtracting kernels.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from .windows import resolve_path

DELAYS = (1, 2, 3)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def membership_groups(manifest, manifest_path, n_voxels):
    """Spatial bins saved by preprocessing, verified against their own provenance.

    The artifact must match the manifest's ``normalization_sha256``, belong to
    the manifest subject, cover every H5 voxel, and equal the occupied
    block-size bins recomputed from its saved voxel coordinates.
    """
    path = resolve_path(manifest['normalization_path'], manifest_path)
    expected = manifest.get('normalization_sha256')
    if expected and _file_sha256(path) != expected:
        raise ValueError(f'Normalization artifact {path} differs from the manifest hash')
    with np.load(path) as state:
        subject, membership = str(state['subject']), state['membership'].astype(np.int64)
        coords = state['coords'] if 'coords' in state else None
        block = int(state['block_size']) if 'block_size' in state else None
    if subject != manifest['subject']:
        raise ValueError(f'Normalization subject {subject} differs from manifest {manifest["subject"]}')
    if membership.shape != (n_voxels,):
        raise ValueError(f'Membership covers {len(membership)} voxels; H5 has {n_voxels}')
    if coords is not None and block is not None:
        _, recomputed = np.unique(np.asarray(coords) // block, axis=0, return_inverse=True)
        if not np.array_equal(recomputed.reshape(-1), membership):
            raise ValueError('Saved membership differs from bins recomputed from voxel coordinates')
    sizes = np.bincount(membership)
    if (sizes == 0).any():
        raise ValueError('Empty spatial group in membership')
    return dict(membership=membership, sizes=sizes, path=str(path), block_size=block,
                normalization_sha256=expected, coordinates_verified=coords is not None and block is not None,
                sha256=hashlib.sha256(membership.tobytes()).hexdigest(), n_groups=len(sizes))


def _h5_path(run, raw_root):
    key = Path(run['source_key'])
    if key.is_absolute() or '..' in key.parts:
        raise ValueError(f'source_key must be relative: {key}')
    return Path(raw_root) / key


def source_fingerprint(index, raw_root):
    """SHA256 over each run's H5 key, size and download ETag (when recorded)."""
    manifest = Path(raw_root) / f"download_ds003020_{index.runs[0]['subject']}.json"
    objects = {}
    if manifest.exists():
        objects = {o['key']: (o['size'], o['etag']) for o in json.loads(manifest.read_text())['objects']}
    rows = []
    for run in index.runs:
        path = _h5_path(run, raw_root)
        size, etag = objects.get(run['source_key'], (path.stat().st_size, None))
        if path.stat().st_size != size:
            raise ValueError(f'{path} size differs from its download record')
        rows.append([run['source_key'], size, etag])
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def cache_metadata(index, raw_root, n_voxels, delays):
    return dict(subject=index.runs[0]['subject'], split=index.split, targets=index.signature(),
                delays=list(delays), n_voxels=int(n_voxels), dtype='float32',
                source_sha256=source_fingerprint(index, raw_root),
                rule='raw H5 rows t+d; non-finite values set to 0')


def verify_cache(index, raw_root, cache_dir, delays=DELAYS):
    """Paths of a cache whose metadata matches this target index and its sources."""
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / f'{index.split}.meta.json'
    if not meta_path.exists():
        raise ValueError(f'No cache metadata at {meta_path}; rebuild the cache')
    saved = json.loads(meta_path.read_text())
    expected = cache_metadata(index, raw_root, saved.get('n_voxels', -1), delays)
    if saved != expected:
        raise ValueError(f'Cache {meta_path} does not match the current targets, delays or sources')
    paths = {d: cache_dir / f'{index.split}_d{d}.npy' for d in delays}
    for p in paths.values():
        array = np.load(p, mmap_mode='r')
        if array.shape != (len(index), saved['n_voxels']) or array.dtype != np.float32:
            raise ValueError(f'Cache array {p} has shape {array.shape} / dtype {array.dtype}')
    return paths, saved['n_voxels']


def cache_delay_rows(index, raw_root, cache_dir, delays=DELAYS, block_rows=64):
    """Write ``<split>_d<d>.npy`` [targets, voxels] holding raw H5 rows t+d, plus metadata."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {d: cache_dir / f'{index.split}_d{d}.npy' for d in delays}
    with h5py.File(_h5_path(index.runs[0], raw_root), 'r') as handle:
        n_voxels = handle['data'].shape[1]
    try:
        return verify_cache(index, raw_root, cache_dir, delays)
    except ValueError:
        pass
    temporary = {d: p.with_suffix('.tmp.npy') for d, p in paths.items()}
    out = {d: np.lib.format.open_memmap(temporary[d], mode='w+', dtype=np.float32,
                                        shape=(len(index), n_voxels)) for d in delays}
    by_run = {}
    for position, (ri, t) in enumerate(index.items):
        by_run.setdefault(ri, []).append((position, t))
    for ri, items in by_run.items():
        run = index.runs[ri]
        positions, times = np.asarray(items, dtype=np.int64).T
        needed = np.unique(np.concatenate([times + d for d in delays]))
        with h5py.File(_h5_path(run, raw_root), 'r') as handle:
            data = handle['data']
            if data.shape != (int(run['n_tr']), n_voxels):
                raise ValueError(f"H5 shape {data.shape} differs from manifest for {run['run_id']}")
            values = np.empty((len(needed), n_voxels), np.float32)
            for start in range(0, len(needed), block_rows):
                block = data[needed[start:start + block_rows]]
                np.nan_to_num(block, copy=False, nan=0, posinf=0, neginf=0)
                values[start:start + len(block)] = block
        lookup = {int(row): k for k, row in enumerate(needed)}
        for d in delays:
            out[d][positions] = values[[lookup[int(t + d)] for t in times]]
    for d in delays:
        out[d].flush()
        del out[d]
        temporary[d].replace(paths[d])
    meta = cache_dir / f'{index.split}.meta.json'
    meta.write_text(json.dumps(cache_metadata(index, raw_root, n_voxels, delays), indent=2) + '\n')
    return paths, n_voxels


def column_stats(x, rows, return_count=False):
    """Column mean/std over ``rows`` with Standardizer's near-zero rule."""
    sub = x.index_select(0, rows)
    mean, std = sub.mean(0), sub.std(0, unbiased=False)
    limit = 1e-8 * max(1.0, float(sub.abs().max()))
    del sub
    small = std <= limit
    std = torch.where(small, torch.ones_like(std), std)
    return (mean, std, int(small.sum())) if return_count else (mean, std)


class KernelBuilder:
    """Base kernels C_d, R_d (d = 1..3) for cached delay designs on one device."""

    def __init__(self, cache, groups, device='cuda', chunk=32768):
        self.cache, self.device, self.chunk = cache, torch.device(device), int(chunk)
        self.member = torch.as_tensor(groups['membership'], device=self.device)
        self.n_groups = groups['n_groups']
        self.inv_sqrt = torch.as_tensor(1.0 / np.sqrt(groups['sizes']), dtype=torch.float64, device=self.device)

    def _load(self, split, delay):
        array = np.array(np.load(self.cache[split][delay], mmap_mode='r'), dtype=np.float32)  # writable copy
        return torch.from_numpy(array).to(self.device, torch.float64)

    def _coarse(self, x):
        c = torch.zeros(x.shape[0], self.n_groups, dtype=torch.float64, device=self.device)
        c.index_add_(1, self.member, x)
        return c * self.inv_sqrt[None, :]

    @torch.no_grad()
    def build(self, stats_rows, eval_splits=(), delays=DELAYS):
        """Training kernels [N, N] and cross-kernels [M_split, N], keyed ('C'|'R', d).

        ``stats_rows`` indexes the training rows whose statistics standardize
        every split (all target rows of the fitting partition's sessions).
        """
        stats_rows = torch.as_tensor(np.asarray(stats_rows), device=self.device)
        kernels, cross, info = {}, {s: {} for s in eval_splits}, {}
        for d in delays:
            x = self._load('train', d)
            mean, std, thresholded = column_stats(x, stats_rows, return_count=True)
            x.sub_(mean).div_(std)
            c = self._coarse(x)
            kernels[('C', d)] = c @ c.T
            evals = {}
            for split in eval_splits:
                e = self._load(split, d)
                e.sub_(mean).div_(std)
                evals[split] = (e, self._coarse(e))
                cross[split][('C', d)] = evals[split][1] @ c.T
                cross[split][('R', d)] = torch.zeros(len(e), len(x), dtype=torch.float64, device=self.device)
            k_r = torch.zeros(len(x), len(x), dtype=torch.float64, device=self.device)
            for start in range(0, x.shape[1], self.chunk):
                cols = slice(start, start + self.chunk)
                scale = (self.inv_sqrt[self.member[cols]])[None, :]
                r = x[:, cols] - c[:, self.member[cols]] * scale
                k_r += r @ r.T
                for split, (e, ce) in evals.items():
                    cross[split][('R', d)] += (e[:, cols] - ce[:, self.member[cols]] * scale) @ r.T
            kernels[('R', d)] = k_r
            info[d] = dict(thresholded_columns=thresholded, mean=mean.cpu().numpy(), std=std.cpu().numpy())
            del x, c, evals
        for key, value in kernels.items():
            kernels[key] = (value + value.T) * 0.5
        return kernels, cross, info
