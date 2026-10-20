"""The identical temporal encoder used for random and SSL initializations."""
from dataclasses import dataclass
from typing import Mapping
import torch
from torch import nn


@dataclass
class EncoderConfig:
    n_features: int
    d_model: int = 384
    n_layers: int = 4
    n_heads: int = 8
    max_length: int = 128
    dropout: float = 0.1
    ff_multiplier: int = 4


class FmriEncoder(nn.Module):
    def __init__(self, config: EncoderConfig | Mapping):
        super().__init__()
        self.config = EncoderConfig(**config) if isinstance(config, Mapping) else config
        c = self.config
        if c.n_features <= 0 or c.max_length <= 0 or c.d_model % c.n_heads:
            raise ValueError('Require positive feature/length sizes and d_model divisible by n_heads')
        self.value_embedding = nn.Linear(c.n_features, c.d_model)
        # Encodes which channels are absent, never the hidden channel values.
        self.missing_embedding = nn.Linear(c.n_features, c.d_model, bias=False)
        self.position_embedding = nn.Parameter(torch.empty(1, c.max_length, c.d_model))
        nn.init.normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(c.d_model, c.n_heads,
            c.d_model * c.ff_multiplier, c.dropout, batch_first=True,
            norm_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(layer, c.n_layers,
            norm=nn.LayerNorm(c.d_model), enable_nested_tensor=False)

    def forward(self, x, observation_mask=None):
        if x.ndim != 3 or x.shape[-1] != self.config.n_features:
            raise ValueError('x must have shape [batch, time, n_features]')
        if x.shape[1] > self.config.max_length:
            raise ValueError('Sequence exceeds configured max_length')
        if observation_mask is None:
            observation_mask = torch.ones_like(x, dtype=torch.bool)
        if observation_mask.shape != x.shape or observation_mask.dtype != torch.bool:
            raise ValueError('observation_mask must be bool with the same shape as x')
        values = x.masked_fill(~observation_mask, 0)
        missing = (~observation_mask).to(x.dtype)
        hidden = self.value_embedding(values) + self.missing_embedding(missing)
        hidden = hidden + self.position_embedding[:, :x.shape[1]]
        return self.transformer(hidden)
