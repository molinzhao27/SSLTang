"""Masked neural modeling with no language inputs or language model."""
from typing import Mapping
import torch
from torch import nn
from .masking import MaskBatch, MaskConfig, generate_mask


class MaskedNeuralModel(nn.Module):
    def __init__(self, encoder, mask_config=None):
        super().__init__()
        self.encoder = encoder
        self.mask_config = (MaskConfig(**mask_config) if isinstance(mask_config, Mapping)
                            else mask_config or MaskConfig())
        self.reconstruction_head = nn.Linear(encoder.config.d_model, encoder.config.n_features)

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
        latent = self.encoder(x, observed)
        prediction = self.reconstruction_head(latent)
        # Cast before subtraction/squaring, including under CUDA BF16 autocast.
        difference = prediction[target].float() - x[target].float()
        loss = difference.square().mean()
        return dict(loss=loss, prediction=prediction, latent=latent,
                    target_mask=target, observation_mask=observed)
