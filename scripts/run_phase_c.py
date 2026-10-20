#!/usr/bin/env python3
"""Protocol v2 Phase C: label coverage and counted supervision for the benchmark.

Method: the Phase A benchmark, ridge on pooled BOLD at t+1..t+3 (6-s cutoff).
For each budget, allocation (hashed 64-TR blocks or story round-robin 8-TR
blocks) and subset seed, alpha is chosen two ways: (1) on the 794 reusable
validation labels, as in Phases A-B; (2) by leave-one-story-out CV inside the
labeled subset, so the training labels are the only semantic supervision.
"""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.endpoints import score
from src.probe import Split, code_fingerprint, grouped_cv_alpha, select_ridge
from src.readout import ALPHAS, RidgePath, Standardizer, raw_features
from src.runtime import file_sha256, load_config, save_json
from src.targets import PROTOCOL_ID, TargetIndex


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/v2/phase_a.yaml')
    p.add_argument('--out', default='results/v2/phase_c')
    p.add_argument('--device', default='cuda')
    p.add_argument('--subset-seeds', type=int, nargs='+', default=[2718, 11, 12, 13, 14])
    p.add_argument('--fractions', type=float, nargs='+', default=[0.25, 0.1, 0.05, 0.01])
    a = p.parse_args()
    cfg = load_config(a.config)
    torch.backends.cuda.matmul.allow_tf32 = False
    data, ev = cfg['data'], cfg['evaluation']
    out = Path(a.out)
    (out/'cells').mkdir(parents=True, exist_ok=True)
    index = {s: TargetIndex(data['manifest'], s, window=data['window'], stride=data['stride'],
                            phase=data['target_phase'], cutoffs=data['cutoffs']) for s in ('train', 'val', 'test')}
    train, val, test = (Split(index[s], a.device) for s in ('train', 'val', 'test'))
    delays = [1, 2, 3]
    features = {s: raw_features(index[s], 3, delays) for s in index}
    standardize = Standardizer(features['train'])  # All training inputs; label-free.
    features = {s: standardize(x) for s, x in features.items()}
    save_json({'protocol': PROTOCOL_ID, 'phase': 'C', 'method': 'B1 raw BOLD ridge, delays t+1..t+3',
               'manifest_sha256': file_sha256(data['manifest']), 'code_sha256': code_fingerprint(),
               'alphas': list(ALPHAS), 'subset_seeds': a.subset_seeds, 'fractions': a.fractions},
              out/'provenance.json')
    designs = [('block', 64, 2718, 1.0)]
    designs += [(allocation, block, seed, fraction) for allocation, block in (('block', 64), ('story', 8))
                for seed in a.subset_seeds for fraction in a.fractions]
    for allocation, block, seed, fraction in designs:
        name = f'{allocation}{block}_seed{seed}_f{fraction:g}'
        path = out/'cells'/f'{name}.json'
        if path.exists():
            continue
        start = time.monotonic()
        rows, meta = index['train'].label_subset(fraction, seed, block, allocation)
        y = train.targets[rows].astype(np.float64)
        mean, std = y.mean(0), np.maximum(y.std(0), 1e-6)
        selected = select_ridge({'fir': (features['train'], features['val'], features['test'])}, {'f': rows},
                                train, val, test, ALPHAS, a.device, standardize=False)['f']
        cell = {'protocol': PROTOCOL_ID, 'allocation': allocation, 'block_tr': block, 'subset_seed': seed,
                'fraction': fraction, 'labels': {k: v for k, v in meta.items() if k != 'label_ids'},
                'validation_selected': {
                    'alpha': selected['alpha'], 'val': selected['score'],
                    'counted_supervision': len(rows) + len(val.targets),
                    'test': score(selected['prediction_test'], test.targets, test.run_ids, test.target_trs,
                                  test.sessions, mean, std, ev['exclusions'])}}
        groups = train.run_ids[rows]
        if len(set(groups.tolist())) >= 2:
            alpha, curve, folds = grouped_cv_alpha(features['train'][rows], train.targets[rows], groups,
                                                   ALPHAS, a.device)
            ridge = RidgePath(features['train'][rows], train.targets[rows], a.device)
            prediction = ridge.predict(ridge.project(features['test']), alpha).cpu().numpy()
            cell['cv_selected'] = {
                'alpha': alpha, 'cv_curve': curve, 'folds': folds, 'counted_supervision': len(rows),
                'test': score(prediction, test.targets, test.run_ids, test.target_trs, test.sessions,
                              mean, std, ev['exclusions'])}
        cell['seconds'] = time.monotonic() - start
        save_json(cell, path)
        cv = cell.get('cv_selected', {}).get('test', {}).get('session_macro_feature_correlation')
        print(f"{name}: stories {meta['labeled_stories']} val-selected test "
              f"{cell['validation_selected']['test']['session_macro_feature_correlation']:.4f} "
              f"cv-selected test {cv if cv is None else round(cv, 4)}", flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
