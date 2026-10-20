"""Target-first indexing for the v2 protocol.

Every timing, readout and pretraining arm enumerates the same semantic target
TRs first and only then derives neural inputs from them. Changing the response
cutoff therefore changes the neural window, never the target identities,
labels or candidate pools. A target is eligible only when every compared cutoff
has a complete in-run window.
"""
from __future__ import annotations
import hashlib
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from .windows import NeuralWindows

PROTOCOL_ID = 'ssltang-v2'


def _sha256(lines):
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


class TargetIndex:
    """Common eligible semantic targets of one split.

    ``phase`` fixes the target clock: ``target_tr % stride == phase``. Phase 0
    with window 32 and cutoff 3 reproduces the stride-4 targets of the archived
    lag-3 protocol, apart from targets lacking a window for the latest cutoff.
    """

    def __init__(self, manifest_path, split, *, window=32, stride=4, phase=0, cutoffs=(3, 4)):
        # NeuralWindows enforces split identity, subject and normalization provenance.
        base = NeuralWindows(manifest_path, split, window=window, stride=stride)
        self.manifest_path, self.manifest, self.runs = base.manifest_path, base.manifest, base.runs
        self.split, self.window, self.stride, self.phase = split, int(window), int(stride), int(phase)
        self.cutoffs = tuple(sorted({int(c) for c in cutoffs}))
        if not self.cutoffs or self.cutoffs[0] < 0 or self.cutoffs[-1] >= self.window:
            raise ValueError('Cutoffs must be nonnegative and shorter than the window')
        if not 0 <= self.phase < self.stride:
            raise ValueError('phase must lie in [0, stride)')
        self.n_features = base.n_features
        self._base = base
        for r in self.runs:
            if 'target_path' not in r or 'target_valid_path' not in r:
                raise ValueError(f"Run lacks verified semantic targets: {r['run_id']}")
        self.target_dim = int(base._array(0, 'target_path').shape[-1])
        first = self.window - 1 - self.cutoffs[0]
        items = []
        for ri, run in enumerate(self.runs):
            if base._array(ri, 'target_path').shape[1] != self.target_dim:
                raise ValueError('All runs must share the same semantic target dimension')
            valid = base._array(ri, 'target_valid_path')
            last = int(run['n_tr']) - 1 - self.cutoffs[-1]
            items.extend((ri, t) for t in range(first, last + 1)
                         if t % self.stride == self.phase and bool(valid[t]))
        if not items:
            raise ValueError(f'No common eligible targets in split {split}')
        self.items = items
        self.ids = [f"{self.runs[ri]['run_id']}:{t}" for ri, t in items]
        self.run_ids = np.array([self.runs[ri]['run_id'] for ri, _ in items])
        self.target_trs = np.array([t for _, t in items], dtype=np.int64)
        self.sessions = {r['run_id']: r.get('session') for r in self.runs}

    def __len__(self):
        return len(self.items)

    def neural(self, run_index):
        return self._base._array(run_index)

    def targets(self, positions=None):
        positions = range(len(self.items)) if positions is None else positions
        return np.stack([np.asarray(self._base._array(self.items[i][0], 'target_path')[self.items[i][1]],
                                    dtype=np.float32) for i in positions])

    def window_bounds(self, cutoff):
        """Inclusive [start, end] neural TRs supplied for each target at this cutoff."""
        cutoff = int(cutoff)
        if cutoff not in self.cutoffs:
            raise ValueError(f'Cutoff {cutoff} was not part of common eligibility {self.cutoffs}')
        end = self.target_trs + cutoff
        return end - self.window + 1, end

    def label_subset(self, fraction, subset_seed=2718, block_tr=64, allocation='block'):
        """Nested, arm-independent labeled positions into ``items``.

        ``block`` orders contiguous ``block_tr`` target blocks by a seeded hash,
        as in the archived protocol. ``story`` visits the same hashed blocks
        round-robin across stories, spreading a count over more stories.
        """
        if self.split != 'train':
            raise ValueError('Only the training split has label budgets')
        if not 0 < fraction <= 1:
            raise ValueError('fraction must be in (0, 1]')
        block_tr = int(block_tr)
        if block_tr < 1:
            raise ValueError('block_tr must be positive')
        blocks = {}
        for position, (ri, t) in enumerate(self.items):
            blocks.setdefault((self.runs[ri]['run_id'], t // block_tr), []).append(position)
        digest = lambda key: hashlib.sha256(f'{subset_seed}:{key[0]}:{key[1]}'.encode()).hexdigest()
        ordered_keys = sorted(blocks, key=digest)
        if allocation == 'story':
            by_story = {}
            for key in ordered_keys:
                by_story.setdefault(key[0], []).append(key)
            story_order = sorted(by_story, key=lambda run: hashlib.sha256(
                f'{subset_seed}:story:{run}'.encode()).hexdigest())
            ordered_keys = [by_story[run][depth] for depth in range(max(map(len, by_story.values())))
                            for run in story_order if depth < len(by_story[run])]
        elif allocation != 'block':
            raise ValueError('allocation must be block or story')
        ordered = [position for key in ordered_keys for position in blocks[key]]
        count = max(1, math.ceil(len(ordered) * float(fraction)))
        selected = sorted(ordered[:count])
        ids = [self.ids[i] for i in selected]
        runs = {self.run_ids[i] for i in selected}
        return selected, {
            'protocol': PROTOCOL_ID, 'requested_fraction': float(fraction),
            'selected_examples': len(selected), 'eligible_examples': len(ordered),
            'realized_fraction': len(selected) / len(ordered), 'subset_seed': subset_seed,
            'block_tr': block_tr, 'allocation': allocation,
            'labeled_stories': len(runs),
            'labeled_sessions': len({self.sessions[r] for r in runs}),
            'indices_sha256': _sha256(ids), 'label_ids': ids,
        }

    def signature(self):
        return {'protocol': PROTOCOL_ID, 'split': self.split, 'window': self.window,
                'stride': self.stride, 'phase': self.phase, 'cutoffs': list(self.cutoffs),
                'n_targets': len(self.items), 'target_ids_sha256': _sha256(self.ids)}


class TargetWindows(Dataset):
    """Neural windows ending at ``target_tr + cutoff_tr`` for fixed targets.

    Exposes the attributes the training loop expects from SemanticWindows.
    """

    def __init__(self, index: TargetIndex, cutoff_tr, label_fraction=1.0, subset_seed=2718,
                 block_tr=64, allocation='block'):
        self.index, self.cutoff = index, int(cutoff_tr)
        index.window_bounds(self.cutoff)  # Validates the cutoff.
        self.manifest, self.runs = index.manifest, index.runs
        self.window, self.n_features, self.target_dim = index.window, index.n_features, index.target_dim
        if index.split == 'train':
            self.positions, self.label_metadata = index.label_subset(label_fraction, subset_seed, block_tr, allocation)
        else:
            self.positions = list(range(len(index)))
            self.label_metadata = {'protocol': PROTOCOL_ID, 'requested_fraction': 1.0,
                                   'selected_examples': len(self.positions),
                                   'eligible_examples': len(self.positions),
                                   'indices_sha256': _sha256(index.ids), 'label_ids': list(index.ids)}
        self.label_metadata.update(cutoff_tr=self.cutoff, target_index=index.signature())
        # (run index, window end) pairs, mirroring SemanticWindows.indices.
        self.indices = [(index.items[p][0], index.items[p][1] + self.cutoff) for p in self.positions]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        ri, t = self.index.items[self.positions[i]]
        end = t + self.cutoff
        x = np.array(self.index.neural(ri)[end - self.window + 1:end + 1], dtype=np.float32, copy=True)
        y = np.array(self.index._base._array(ri, 'target_path')[t], dtype=np.float32, copy=True)
        return {'x': torch.from_numpy(x), 'y': torch.from_numpy(y), 'run_index': ri,
                'end_tr': end, 'target_tr': t, 'run_id': self.runs[ri]['run_id']}
