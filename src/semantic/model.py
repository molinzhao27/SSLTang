"""Continuous semantic embedding regression; does not generate text."""
import torch
from torch import nn


class SemanticRegressor(nn.Module):
    def __init__(self, encoder, target_dim, pooling='end'):
        super().__init__()
        if pooling not in ('end', 'mean'):
            raise ValueError('pooling must be end or mean')
        self.encoder = encoder
        self.pooling = pooling
        self.head = nn.Linear(encoder.config.d_model, target_dim)

    def forward(self, x):
        latent = self.encoder(x)
        pooled = latent[:, -1] if self.pooling == 'end' else latent.mean(dim=1)
        return self.head(pooled)
