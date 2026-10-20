#!/usr/bin/env python3
"""Protocol v2 Phase A: ridge baselines on raw, PCA and frozen-encoder features.

Arms: B0 constant selected-label mean; B1 pooled BOLD at explicit delays;
B2 train-only window PCA; B3 frozen random encoders; B4 frozen reconstruction
encoders (best-validation and final checkpoints). All arms share target IDs,
label subsets, alpha grid and the validation selection rule. Test scores never
select anything. Writes one JSON per cell; predictions stay out of git.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.endpoints import score, shift_null
from src.models import FmriEncoder
from src.probe import Split, code_fingerprint, select_ridge
from src.readout import ALPHAS, WindowPCA, encoder_features, raw_features, window_matrix
from src.runtime import file_sha256, load_config, save_json, seed_everything
from src.targets import PROTOCOL_ID, TargetIndex
from src.windows import NeuralWindows, SemanticWindows


def archived_comparison(manifest, index, runs_root, fractions, subsets):
    """How v2 targets and label subsets relate to the archived lag-3 protocol."""
    report = {}
    for split in ('train', 'val', 'test'):
        old = SemanticWindows(manifest, split, 32, 4, lag_tr=3)
        old_ids = set(old.label_metadata['label_ids'])
        new_ids = set(index[split].ids)
        report[split] = {'archived': len(old_ids), 'v2': len(new_ids), 'shared': len(old_ids & new_ids),
                         'only_archived': sorted(old_ids - new_ids), 'only_v2': sorted(new_ids - old_ids)}
    subject = index['train'].runs[0]['subject']
    for fraction in fractions:
        path = Path(runs_root)/f'random_{subject}_fraction{fraction:g}_seed17'/'label_subset.json'
        if path.exists():
            old = set(json.loads(path.read_text())['label_metadata']['label_ids'])
            new = set(subsets[f'{fraction:g}']['label_ids'])
            report[f'labels_{fraction:g}'] = {'archived': len(old), 'v2': len(new), 'shared': len(old & new)}
    return report


def load_encoder(path, model_cfg, n_features, manifest):
    """Load an SSL encoder after the provenance checks used by semantic fine-tuning."""
    from src.training import _neural_signature
    state = torch.load(path, map_location='cpu', weights_only=False)
    source = manifest.get('neural_manifest_hash')
    matches = (state.get('neural_signature') == _neural_signature(manifest)
               or (bool(source) and source == state.get('manifest_sha256')))
    if state.get('task') != 'ssl' or not matches:
        raise ValueError(f'SSL encoder provenance differs from this manifest: {path}')
    expected = dict(model_cfg, n_features=n_features)
    if any(state['encoder_config'].get(k) != v for k, v in expected.items()):
        raise ValueError(f'Encoder architecture differs: {path}')
    encoder = FmriEncoder(state['encoder_config'])
    encoder.load_state_dict(state['encoder_state_dict'])
    return encoder, {'path': str(path), 'sha256': file_sha256(path), 'global_step': state.get('global_step')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/v2/phase_a.yaml')
    p.add_argument('--device', default='cuda')
    p.add_argument('--only', nargs='*', default=None, help='Subset of arms: raw pca encoders constant')
    p.add_argument('--set', action='append', default=[])
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(cfg['output_dir'])
    (out/'cells').mkdir(parents=True, exist_ok=True)
    (out/'predictions').mkdir(parents=True, exist_ok=True)
    data, labels, readout, ev = cfg['data'], cfg['labels'], cfg['readout'], cfg['evaluation']
    manifest = data['manifest']
    cutoffs = [int(c) for c in data['cutoffs']]
    index = {split: TargetIndex(manifest, split, window=data['window'], stride=data['stride'],
                                phase=data['target_phase'], cutoffs=cutoffs)
             for split in ('train', 'val', 'test')}
    train, val, test = (Split(index[s], a.device) for s in ('train', 'val', 'test'))
    budgets, subsets = {}, {}
    for fraction in labels['fractions']:
        rows, meta = index['train'].label_subset(fraction, labels['subset_seed'], labels['block_tr'],
                                                 labels['allocation'])
        budgets[f'{fraction:g}'], subsets[f'{fraction:g}'] = rows, meta
    provenance = {
        'protocol': PROTOCOL_ID, 'phase': 'A', 'config': cfg, 'manifest_sha256': file_sha256(manifest),
        'code_sha256': code_fingerprint(), 'alphas': list(ALPHAS),
        'targets': {s: index[s].signature() for s in index},
        'sessions': {s: index[s].sessions for s in ('val', 'test')},
        'archived_comparison': archived_comparison(manifest, index, 'results/runs', labels['fractions'], subsets),
    }
    save_json(provenance, out/'provenance.json')
    save_json({s: index[s].ids for s in ('val', 'test')}, out/'heldout_target_ids.json')
    save_json(subsets, out/'label_subsets.json')
    print(json.dumps({s: index[s].signature()['n_targets'] for s in index}), flush=True)

    def record(arm, tag, seed, cutoff, readout_name, support, results, extra=None):
        for budget, result in results.items():
            rows = budgets[budget]
            y = train.targets[rows].astype(np.float64)
            mean, std = y.mean(0), np.maximum(y.std(0), 1e-6)
            name = f'{arm}_{tag}_seed{seed}_c{cutoff}_{readout_name}_f{budget}'
            cell = {
                'protocol': PROTOCOL_ID, 'arm': arm, 'features': tag, 'seed': seed,
                'cutoff_tr': cutoff, 'latency_seconds': 2*cutoff, 'readout': readout_name,
                'input_support_tr': support, 'fraction': float(budget),
                'labels': {k: v for k, v in subsets[budget].items() if k != 'label_ids'},
                'validation_labels': len(val.targets),
                'selection': {k: result[k] for k in ('option', 'alpha', 'score', 'alpha_at_grid_edge', 'n_features')},
                'validation_grid': result['grid'],
                'val': score(result['prediction_val'], val.targets, val.run_ids, val.target_trs,
                             val.sessions, mean, std, ev['exclusions']),
                'test': score(result['prediction_test'], test.targets, test.run_ids, test.target_trs,
                              test.sessions, mean, std, ev['exclusions']),
                'test_shift_null': shift_null(result['prediction_test'], test.targets, test.run_ids,
                                              test.sessions, data['stride'], ev['shift_min_tr'],
                                              ev['shift_draws'], ev['shift_seed']),
            }
            cell.update(extra or {})
            save_json(cell, out/'cells'/f'{name}.json')
            np.savez_compressed(out/'predictions'/f'{name}.npz',
                                val=result['prediction_val'].astype(np.float16),
                                test=result['prediction_test'].astype(np.float16))
            print(f"{name}: val {result['score']:.4f} test {cell['test']['session_macro_feature_correlation']:.4f} "
                  f"[{result['option']}, alpha={result['alpha']:.3g}]", flush=True)

    def family(arm, tag, seed, cutoff, readout_name, support, options, extra=None, standardize=True):
        start = time.monotonic()
        results = select_ridge(options, budgets, train, val, test, ALPHAS, a.device, standardize)
        record(arm, tag, seed, cutoff, readout_name, support, results,
               dict(extra or {}, fit_seconds=time.monotonic()-start))

    only = set(a.only or ['constant', 'raw', 'pca', 'encoders'])
    if 'constant' in only:
        for budget, rows in budgets.items():
            y = train.targets[rows].astype(np.float64)
            mean, std = y.mean(0), np.maximum(y.std(0), 1e-6)
            constant = {split.index.split: score(np.broadcast_to(mean, split.targets.shape), split.targets,
                                                 split.run_ids, split.target_trs, split.sessions, mean, std,
                                                 ev['exclusions']) for split in (val, test)}
            save_json({'protocol': PROTOCOL_ID, 'arm': 'B0', 'features': 'constant_selected_mean',
                       'fraction': float(budget), **constant}, out/'cells'/f'B0_constant_f{budget}.json')
    if 'raw' in only:
        for cutoff in cutoffs:
            for readout_name, delays in (('endpoint', [cutoff]), ('fir', list(range(1, cutoff+1)))):
                options = {f'delays={delays}': tuple(raw_features(index[s], cutoff, delays)
                                                    for s in ('train', 'val', 'test'))}
                family('B1', 'bold', 0, cutoff, readout_name, [f't+{d}' for d in delays], options)
    if 'pca' in only:
        windows = NeuralWindows(manifest, 'train', data['window'], data['stride'])
        matrix = np.stack([windows[i]['x'].numpy().reshape(-1) for i in range(len(windows))])
        pca = WindowPCA(matrix, max(readout['pca_components']), a.device)
        del matrix
        for cutoff in cutoffs:
            scores = {s: pca.transform(window_matrix(index[s], cutoff)) for s in ('train', 'val', 'test')}
            options = {}
            for k in readout['pca_components']:
                for scaling in readout['pca_scalings']:
                    if scaling == 'whitened':
                        scale = np.sqrt(pca.explained_variance[:k])
                    else:  # One global scale keeps relative component variances.
                        scale = np.sqrt(pca.explained_variance[:k].mean())
                    options[f'k={k},{scaling}'] = tuple(scores[s][:, :k] / scale for s in ('train', 'val', 'test'))
            family('B2', 'window_pca', 0, cutoff, 'window', [f't+{cutoff-data["window"]+1}', f't+{cutoff}'],
                   options, {'pca_fit_windows': len(windows), 'pca_variance_explained':
                             {str(k): float(pca.explained_variance[:k].sum()/pca.total_variance)
                              for k in readout['pca_components']},
                             'scaling': 'unwhitened: one global scale; whitened: per-component unit variance'},
                   standardize=False)
        del pca
    if 'encoders' in only:
        model_cfg = load_config(cfg['encoders']['model_config'])['model']
        manifest_data = json.loads(Path(manifest).read_text())
        settings = cfg['encoders']
        if 'sources' in settings:  # tag: [arm, checkpoint path template with {seed}]
            sources = {tag: tuple(value) for tag, value in settings['sources'].items()}
        else:  # Phase A form: one SSL run directory, several checkpoint files.
            sources = {tag: ('B4', str(Path(settings['ssl_run_template'])/name))
                       for tag, name in settings['ssl_checkpoints'].items()}
        random_arm = settings.get('random_arm', 'B3')
        for seed in settings['seeds']:
            encoders = {}
            if random_arm:
                seed_everything(seed)  # Identical to the random arm's initialization.
                encoders['random'] = (random_arm, FmriEncoder(dict(model_cfg, n_features=train.index.n_features)),
                                      {'init': 'seed_everything(seed); FmriEncoder(model)'})
            for tag, (arm, template) in sources.items():
                encoders[tag] = (arm, *load_encoder(Path(template.format(seed=seed)), model_cfg,
                                                    train.index.n_features, manifest_data))
            for tag, (arm, encoder, source) in encoders.items():
                for cutoff in cutoffs:
                    states = {s: encoder_features(encoder, index[s], cutoff, cutoff, a.device)
                              for s in ('train', 'val', 'test')}
                    for readout_name, positions in (('endpoint', slice(-1, None)), ('fir', slice(None))):
                        options = {f'layer={layer}': tuple(
                            states[s][:, layer-1, positions].reshape(len(states[s]), -1)
                            for s in ('train', 'val', 'test')) for layer in readout['encoder_layers']}
                        support = ([f't+{cutoff}'] if readout_name == 'endpoint'
                                   else [f't+{d}' for d in range(1, cutoff+1)])
                        family(arm, tag, seed, cutoff, readout_name,
                               {'states_at': support, 'window': [f't+{cutoff-data["window"]+1}', f't+{cutoff}']},
                               options, {'encoder_source': source})
                del states
    print('done', flush=True)


if __name__ == '__main__':
    main()
