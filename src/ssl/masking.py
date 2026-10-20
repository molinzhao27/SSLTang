"""Contiguous reconstruction cores with extra hidden temporal guard bands."""
from dataclasses import dataclass
from typing import Mapping
import math
import torch


@dataclass
class MaskConfig:
    mode: str = 'temporal'  # temporal, spatial, joint
    target_fraction: float = 0.25
    span_length: int = 8
    guard_length: int = 4
    feature_group_size: int = 64

    def __post_init__(self):
        if self.mode not in ('temporal', 'spatial', 'joint'):
            raise ValueError('mode must be temporal, spatial, or joint')
        if not 0 < self.target_fraction < 1:
            raise ValueError('target_fraction must lie between zero and one')
        if self.span_length < 1 or self.guard_length < 0 or self.feature_group_size < 1:
            raise ValueError('Invalid span, guard, or group size')


@dataclass
class MaskBatch:
    target_mask: torch.Tensor
    observation_mask: torch.Tensor


def generate_mask(shape, config: MaskConfig | Mapping, *, device=None, generator=None):
    """Generator should be CPU; masks are sampled on CPU then moved to device.

    Spatial groups are contiguous feature indices; anatomical adjacency requires
    the caller to order features anatomically before constructing input tensors.
    Temporal/joint cores have hidden guard_length TRs on both sides (clipped at
    sequence boundaries). Guards are excluded from reconstruction loss.
    """
    c = MaskConfig(**config) if isinstance(config, Mapping) else config
    b, t, f = shape
    if min(b, t, f) < 1:
        raise ValueError('Mask dimensions must be positive')
    target = torch.zeros(shape, dtype=torch.bool)
    hidden = torch.zeros_like(target)
    groups = math.ceil(f / c.feature_group_size)
    for batch in range(b):
        if c.mode == 'spatial':
            count = min(groups, max(1, math.ceil(groups * c.target_fraction)))
            for group in torch.randperm(groups, generator=generator)[:count].tolist():
                start = group * c.feature_group_size
                target[batch, :, start:start+c.feature_group_size] = True
            hidden[batch] = target[batch]
        else:
            span = min(t, c.span_length)
            # Distinct shuffled start locations; overlapping cores are merged.
            goal = max(1, math.ceil(t * f * c.target_fraction))
            n_starts = t - span + 1
            candidates = n_starts * groups if c.mode == 'joint' else n_starts
            for candidate in torch.randperm(candidates, generator=generator).tolist():
                start = candidate % n_starts
                if c.mode == 'joint':
                    group = candidate // n_starts
                    lo, hi = group*c.feature_group_size, min(f, (group+1)*c.feature_group_size)
                else:
                    lo, hi = 0, f
                target[batch, start:start+span, lo:hi] = True
                hidden[batch, max(0,start-c.guard_length):min(t,start+span+c.guard_length), lo:hi] = True
                if int(target[batch].sum()) >= goal:
                    break
    return MaskBatch(target.to(device=device), (~hidden).to(device=device))
