"""EMA-teacher latent prediction for masked neural windows (no language inputs).

A compact adaptation of data2vec/JEPA-style latent targets, not a reproduction
of Brain-JEPA. The student encodes the masked window exactly as the
reconstruction objective does; a linear predictor maps its states at hidden
core positions to the teacher's states for the uncorrupted window. The teacher
is an exponential moving average of the student, never receives gradients and
always runs without dropout. Hidden values reach only the teacher target.
"""
import copy
from typing import Mapping
import torch
from torch import nn
from torch.nn import functional as F
from .masking import MaskBatch, MaskConfig, generate_mask


class LatentPredictionModel(nn.Module):
    def __init__(self, encoder, mask_config=None, ema_start=0.996, ema_end=1.0):
        super().__init__()
        self.encoder = encoder
        self.mask_config = (MaskConfig(**mask_config) if isinstance(mask_config, Mapping)
                            else mask_config or MaskConfig())
        if self.mask_config.mode != 'temporal':
            raise ValueError('Latent targets are defined per time step; use temporal masks')
        if not 0 <= ema_start <= ema_end <= 1:
            raise ValueError('Require 0 <= ema_start <= ema_end <= 1')
        self.ema_start, self.ema_end = float(ema_start), float(ema_end)
        d = encoder.config.d_model
        self.predictor = nn.Linear(d, d)
        self.teacher = copy.deepcopy(encoder)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()  # The target encoder never uses dropout.
        return self

    def momentum(self, update, total_updates):
        """Momentum for the ``update``-th optimizer step (1-based).

        m_k = start + (end - start) * (k - 1) / total for k = 1..total, so the
        final update uses start + (end - start) * (total - 1) / total (0.9999964
        for 0.996 -> 1.0 over 1,110 updates), as in the Phase B runs.
        """
        fraction = min(1.0, max(0, update - 1) / max(1, total_updates))
        return self.ema_start + (self.ema_end - self.ema_start) * fraction

    @torch.no_grad()
    def update_teacher(self, momentum):
        for teacher, student in zip(self.teacher.parameters(), self.encoder.parameters()):
            teacher.lerp_(student.detach().to(teacher.dtype), 1.0 - momentum)
        for teacher, student in zip(self.teacher.buffers(), self.encoder.buffers()):
            teacher.copy_(student)

    @torch.no_grad()
    def targets(self, x):
        latent = self.teacher(x)
        return F.layer_norm(latent.float(), latent.shape[-1:])

    def forward(self, x, mask=None, *, generator=None):
        if mask is None:
            mask = generate_mask(x.shape, self.mask_config, device=x.device, generator=generator)
        if not isinstance(mask, MaskBatch):
            raise TypeError('mask must be a MaskBatch')
        target, observed = mask.target_mask, mask.observation_mask
        if (target.shape != x.shape or observed.shape != x.shape or
                target.dtype != torch.bool or observed.dtype != torch.bool):
            raise ValueError('Mask tensors must be bool and match x shape')
        if not target.any() or (target & observed).any():
            raise ValueError('Require nonempty targets and no target visible to encoder')
        core = target.any(-1)
        if not torch.equal(core, target.all(-1)):
            raise ValueError('Temporal cores must hide every feature of a time step')
        latent = self.encoder(x, observed)
        prediction = self.predictor(latent)
        goal = self.targets(x)
        difference = prediction[core].float() - goal[core]
        loss = difference.square().mean()
        return dict(loss=loss, prediction=prediction, latent=latent, target_mask=target,
                    observation_mask=observed, core=core,
                    n_units=core.sum() * latent.shape[-1])
