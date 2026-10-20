"""Collapse and input-dependence diagnostics for frozen-input encoder states."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def representation_diagnostics(encoder, x):
    """Summaries of ``encoder(x)`` for fixed, fully observed windows [N, T, F].

    content_fraction is the across-window variance at fixed positions divided
    by the total variance; states driven only by positional embeddings give
    values near zero. Effective rank is exp(entropy) of the normalized singular
    values of the centered final-position states.
    """
    was_training = encoder.training
    encoder.eval()
    h = encoder(x).float()
    encoder.train(was_training)
    n = h.shape[0]
    within = (h - h.mean(0, keepdim=True)).square().mean()
    total = (h - h.mean((0, 1), keepdim=True)).square().mean().clamp_min(1e-12)
    report = {'content_fraction': float(within / total)}
    for name, index in (('last', -1), ('mid', h.shape[1] // 2)):
        states = h[:, index]
        centered = states - states.mean(0)
        singular = torch.linalg.svdvals(centered)
        p = singular / singular.sum().clamp_min(1e-12)
        unit = F.normalize(states, dim=1)
        cosine = unit @ unit.T
        report.update({
            f'effective_rank_{name}': float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum())),
            f'std_{name}': float(centered.std(0).mean()),
            f'mean_pairwise_cosine_{name}': float((cosine.sum() - n) / max(1, n * (n - 1))),
        })
    return report
