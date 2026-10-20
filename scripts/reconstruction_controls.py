#!/usr/bin/env python3
"""Neural reconstruction controls on the exact SSL validation masks.

Reproduces the training loop's fixed validation masks (sequential batches,
generator seed + 100000 + batch index) and scores simple predictors of the
hidden core that use only observed values of the same window: the training
mean (zero after train-only normalization), the observed-window mean, and
linear interpolation between the nearest observed frames on each side.
Winning reconstruction is a diagnostic, not a semantic endpoint.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import load_config, save_json
from src.ssl import MaskConfig, generate_mask
from src.windows import NeuralWindows


def interpolate(x, observed_time):
    """Per window, linear interpolation over time from observed frames."""
    out = np.empty_like(x)
    steps = np.arange(x.shape[0])
    seen = np.flatnonzero(observed_time)
    for f in range(x.shape[1]):
        out[:, f] = np.interp(steps, seen, x[seen, f])  # Flat beyond the observed range.
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/ssl_uts01.yaml')
    p.add_argument('--seeds', type=int, nargs='+', default=[17, 23, 47])
    p.add_argument('--out', default='results/v2/reconstruction_controls.json')
    p.add_argument('--set', action='append', default=[])
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    data = cfg['data']
    windows = NeuralWindows(data['manifest'], 'val', data['window'], data['stride'])
    mask_cfg = MaskConfig(**{k: v for k, v in cfg['ssl'].items() if k in MaskConfig.__dataclass_fields__})
    batch = int(cfg['training']['batch_size'])
    x_all = np.stack([windows[i]['x'].numpy() for i in range(len(windows))])
    report = {'windows': len(windows), 'mask': vars(mask_cfg), 'by_seed': {}}
    for seed in a.seeds:
        sums = {'train_mean': 0.0, 'window_mean': 0.0, 'interpolation': 0.0}
        count = 0
        for index, start in enumerate(range(0, len(x_all), batch)):
            x = x_all[start:start + batch]
            mask = generate_mask(x.shape, mask_cfg, generator=torch.Generator().manual_seed(seed + 100000 + index))
            target, observed = mask.target_mask.numpy(), mask.observation_mask.numpy()
            for w in range(len(x)):
                core = target[w]
                observed_time = observed[w].all(-1)
                values = x[w][core]
                window_mean = x[w][observed_time].mean(0)
                sums['train_mean'] += float(np.square(values).sum())
                sums['window_mean'] += float(np.square(np.broadcast_to(window_mean, x[w].shape)[core] - values).sum())
                sums['interpolation'] += float(np.square(interpolate(x[w], observed_time)[core] - values).sum())
                count += int(core.sum())
        report['by_seed'][str(seed)] = {k: v / count for k, v in sums.items()} | {'core_elements': count}
    for name in ('train_mean', 'window_mean', 'interpolation'):
        report[f'mean_{name}'] = float(np.mean([s[name] for s in report['by_seed'].values()]))
    save_json(report, a.out)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
