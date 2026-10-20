#!/usr/bin/env python3
"""Fit the locked v2 B1 ridge decoder, then evaluate run-level ds004510 transfer.

No transfer neural array is opened until the training-only decoder is fully fit.
Reference TextGrid times are discarded; retrieval measures narrative matching.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.transfer import (
    CANDIDATES, _array_hash, _cached_reference, _frozen_pin, _narrative,
    _normalization, _path, _read_manifest, _source_manifest, retrieval_summary,
)
from src.probe import Split, code_fingerprint, select_ridge
from src.readout import ALPHAS, RidgePath, Standardizer, raw_features
from src.runtime import file_sha256, load_config, save_json
from src.semantic import features
from src.targets import TargetIndex

PROTOCOL_ID = 'ssltang-v2-ridge-transfer'
DELAYS = (1, 2, 3)


@dataclass(frozen=True)
class FittedDecoder:
    """Training-only fitted state; inference has no fitting or selection path."""

    standardizer: Standardizer
    weights: torch.Tensor
    x_mean: torch.Tensor
    target_mean: np.ndarray
    alpha: float
    validation_score: float
    description: dict

    def predict(self, x):
        x = torch.as_tensor(self.standardizer(x), dtype=torch.float64,
                            device=self.weights.device)
        with torch.inference_mode():
            mean = torch.as_tensor(self.target_mean, dtype=torch.float64,
                                   device=self.weights.device)
            return ((x - self.x_mean) @ self.weights + mean).cpu().numpy()


def fit_decoder(config, device='cuda'):
    """Reproduce the single-option, 100%-label Phase A B1 cell and refit it.

    This API deliberately accepts no transfer manifest, arrays, or references.
    The required test split is passed to select_ridge, but its scores are unused.
    """
    data = config['data']
    timing = (int(data.get('window', 32)), int(data.get('stride', 4)),
              int(data.get('target_phase', 0)), tuple(data.get('cutoffs', (3, 4))))
    if timing != (32, 4, 0, (3, 4)):
        raise ValueError('Locked B1 requires window=32, stride=4, phase=0, cutoffs=(3,4)')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    indices = {s: TargetIndex(data['manifest'], s, window=32, stride=4,
                              phase=0, cutoffs=(3, 4)) for s in ('train', 'val', 'test')}
    if any(i.target_dim != 768 for i in indices.values()):
        raise ValueError('Locked B1 requires 768-D GPT-2 targets in every split')
    train, val, test = (Split(indices[s], device) for s in ('train', 'val', 'test'))
    x_train, x_val, x_test = (raw_features(indices[s], 3, list(DELAYS))
                            for s in ('train', 'val', 'test'))
    if not all(np.isfinite(x).all() for x in (x_train, x_val, x_test, train.targets,
                                             val.targets, test.targets)):
        raise ValueError('Training/development features and targets must be finite')
    scale = Standardizer(x_train)  # All training inputs; no validation or transfer statistics.
    x_train, x_val, x_test = (scale(x) for x in (x_train, x_val, x_test))
    rows = list(range(len(train.targets)))
    option = 'delays=[1, 2, 3]'
    selected = select_ridge({option: (x_train, x_val, x_test)}, {'1': rows},
                            train, val, test, ALPHAS, device, standardize=False)['1']
    if selected['alpha'] is None:
        raise ValueError('No finite validation score on the fixed alpha grid')
    alpha = float(selected['alpha'])
    # Refit on every training row (4,518 for the contracted UTS01 manifest).
    ridge = RidgePath(x_train, train.targets, device)
    shrink = ridge.s / (ridge.s.square() + alpha)
    weights = ridge.vh.T @ (shrink[:, None] * ridge.uty)
    mean = train.targets.astype(np.float64).mean(0)
    description = dict(arm='B1', features='concatenated pooled BOLD rows t+1, t+2, t+3',
        delays=list(DELAYS), cutoff_tr=3, window=32, stride=4, phase=0, cutoffs=[3, 4],
        label_fraction=1.0, alpha=alpha, validation_score=selected['score'],
        validation_metric='session_macro_feature_correlation', n_train=len(rows),
        n_validation=len(val.targets), n_features=x_train.shape[1],
        alpha_grid=list(ALPHAS), alpha_at_grid_edge=selected['alpha_at_grid_edge'],
        validation_grid=selected['grid'], targets={s: i.signature() for s, i in indices.items()},
        standardization='Standardizer fit on all training-split feature rows',
        intercept='unpenalized', precision='float64 ridge; float32 standardized features',
        weights_sha256=_array_hash(weights.cpu().numpy()),
        standardizer_mean_sha256=_array_hash(scale.mean),
        standardizer_std_sha256=_array_hash(scale.std), target_mean_sha256=_array_hash(mean))
    return FittedDecoder(scale, weights, ridge.x_mean, mean, alpha, selected['score'], description)


def transfer_features(values):
    """Every eligible t, concatenated in the same delay-major order as raw_features."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError('Transfer neural array must be [n_tr,n_features]')
    n = max(0, len(values) - 3)
    return np.concatenate([values[d:d + n] for d in DELAYS], axis=1)


def predict_run(decoder, run, manifest_path):
    if not isinstance(decoder, FittedDecoder):
        raise TypeError('Transfer arrays require a fully fitted training-only decoder')
    path = _path(run['neural_path'], manifest_path)
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    if values.shape != (int(run['n_tr']), int(run['n_features'])) or values.dtype != np.float32:
        raise ValueError(f"{run['run_id']}: expected normalized float32 [n_tr,n_features]")
    x = transfer_features(values)
    finite = np.isfinite(x).all(1)
    times = np.flatnonzero(finite)
    predictions = decoder.predict(x[finite]) if finite.any() else np.empty((0, len(decoder.target_mean)))
    finite_prediction = np.isfinite(predictions).all(1)
    kept = times[finite_prediction]
    record = dict(run_id=run['run_id'], narrative=_narrative(run), repeat=run.get('repeat'),
        neural_path=str(path), neural_sha256=file_sha256(path), n_tr=len(values),
        n_eligible_tr=len(x), n_tr_averaged=len(kept), target_trs_averaged=kept.tolist(),
        skipped_nonfinite_feature_trs=np.flatnonzero(~finite).tolist(),
        skipped_nonfinite_prediction_trs=times[~finite_prediction].tolist())
    if not len(kept):
        return None, record
    return predictions[finite_prediction].mean(0, dtype=np.float64).astype(np.float32), record


def validate_manifests(training_path, transfer_path):
    """Reuse archived checks, without opening neural arrays or instantiating GPT-2."""
    training, transfer = _read_manifest(training_path), _read_manifest(transfer_path)
    source_train, source_train_path, _ = _source_manifest(training, training_path)
    source_transfer, source_transfer_path, _ = _source_manifest(transfer, transfer_path)
    manifests = (training, source_train, transfer, source_transfer)
    runs = [r for m in manifests for r in m['runs']]
    if any(r.get('dataset') != 'ds003020' or r.get('split') not in ('train', 'val', 'test')
           for m in (training, source_train) for r in m['runs']):
        raise ValueError('Training manifests must contain only ds003020 train/val/test runs')
    if any(r.get('dataset') != 'ds004510' or r.get('split') != 'transfer'
           for m in (transfer, source_transfer) for r in m['runs']):
        raise ValueError('Transfer manifests must contain only held-out ds004510 transfer runs')
    if {r['run_id'] for r in transfer['runs']} != {r['run_id'] for r in source_transfer['runs']}:
        raise ValueError('Transfer semantic manifest omits source runs; use the neural transfer manifest')
    subjects = {r['subject'] for r in runs}
    if len(subjects) != 1:
        raise ValueError('Training/transfer must share one subject')
    subject = next(iter(subjects))
    n_features = int(runs[0]['n_features'])
    if any(int(r['n_features']) != n_features for r in runs):
        raise ValueError('Training/transfer feature dimensions differ')
    if any(m.get('subject', subject) != subject for m in manifests):
        raise ValueError('Manifest subject differs from run subject')
    if not training.get('feature_order') or any(m.get('feature_order') != training['feature_order']
                                              for m in manifests):
        raise ValueError('Explicit training/transfer feature order provenance must agree')
    for enriched, original, owner, original_owner in (
        (training, source_train, training_path, source_train_path),
        (transfer, source_transfer, transfer_path, source_transfer_path),
    ):
        original_runs = {r['run_id']: r for r in original['runs']}
        for run in enriched['runs']:
            old = original_runs.get(run['run_id'])
            if old is None or any(run.get(k) != old.get(k) for k in (
                'dataset', 'subject', 'split', 'n_features', 'n_tr', 'condition', 'story', 'repeat')):
                raise ValueError('Semantic enrichment changed neural run identity')
            if _path(run['neural_path'], owner) != _path(old['neural_path'], original_owner):
                raise ValueError('Semantic enrichment changed neural array provenance')
    norms = {}
    paths = (training_path, source_train_path, transfer_path, source_transfer_path)
    for name, manifest, path in zip(('training', 'source_training', 'transfer', 'source_transfer'), manifests, paths):
        _, norms[name] = _normalization(manifest, path, subject, n_features, source_train['runs'])
        if norms[name]['array_sha256'] != norms['training']['array_sha256']:
            raise ValueError('Transfer/source normalization differs from the training preprocessor')
    norms.update(match='exact_file_sha256' if norms['training']['sha256'] == norms['transfer']['sha256']
                 else 'exact_saved_arrays', fit_during_evaluation=False)
    provenance = dict(subject=subject, n_features=n_features, feature_order=training['feature_order'],
        normalization=norms, frozen_reference_encoder=_frozen_pin(training, 768),
        code_sha256=code_fingerprint(),
        coordinate_identity_evidence=transfer.get('coordinate_identity_evidence',
            'Published within-subject column convention; no independently embedded H5 coordinate map'))
    for name, path in zip(('training', 'training_neural', 'transfer', 'transfer_neural'), paths):
        provenance[f'{name}_manifest_path'] = str(Path(path).resolve())
        provenance[f'{name}_manifest_sha256'] = file_sha256(path)
    # Include every immutable source to prevent output directories swallowing inputs.
    sources = list(map(Path, paths)) + [Path(v['path']) for v in norms.values() if isinstance(v, dict)]
    for manifest, path in zip(manifests, paths):
        sources.extend(_path(r[k], path) for r in manifest['runs']
                       for k in ('neural_path', 'target_path', 'target_valid_path', 'stimulus_path') if r.get(k))
    return transfer, provenance, sources


def run_evaluation(config, transfer_manifest, output, device='cuda', feature_batch_size=16):
    if feature_batch_size < 1:
        raise ValueError('feature_batch_size must be positive')
    training_path, transfer_path = Path(config['data']['manifest']).resolve(), Path(transfer_manifest).resolve()
    output = Path(output).resolve()
    manifest, provenance, sources = validate_manifests(training_path, transfer_path)
    if any(p.resolve() == output or output in p.resolve().parents for p in sources):
        raise ValueError('Evaluation output must be separate from immutable input artifacts')
    # Structural boundary: all fitting, scaling and validation selection finish here.
    decoder = fit_decoder(config, device)
    output.mkdir(parents=True, exist_ok=True)
    pin, encoder = provenance['frozen_reference_encoder'], None

    def get_encoder():
        nonlocal encoder
        if encoder is None:
            encoder = features.FrozenGPT2(model=pin['model'], revision=pin['revision'],
                layer=pin['layer'], context_words=pin['context_words'], device=device)
        return encoder

    result = dict(schema_version=1, protocol=PROTOCOL_ID, dataset='ds004510', subject=provenance['subject'],
        created_at=datetime.now(timezone.utc).isoformat(), config=config, decoder=decoder.description,
        provenance=provenance, inference=dict(device=str(device), feature_batch_size=feature_batch_size,
            fitting_complete_before_transfer_arrays_opened=True), conditions={}, skipped_runs=[],
        limitations=['Run-level retrieval only.', 'No verified imagined-word timing.',
                     'Whole-reference embeddings are reference proxies.',
                     'Repeats are not independent observations.', '3-film null minimum p = 1/6.'])
    all_predictions, all_identities, counts = [], [], []
    for condition, candidate_ids in CANDIDATES.items():
        report = dict(candidate_ids=list(candidate_ids), top1_chance=1 / len(candidate_ids),
            reference_artifacts={}, runs=[], skipped_runs=[], missing_reasons=[], raw=None,
            constant_training_mean=None, training_mean_centered=None,
            aggregation='Mean of finite predictions for every t=0..n_tr-4; inputs t+1,t+2,t+3',
            centering='Subtract the 100% training-label target mean from predictions and references; no transfer fit')
        result['conditions'][condition] = report
        runs = []
        for run in manifest['runs']:
            if run.get('condition') != condition:
                continue
            narrative = _narrative(run)
            reason = None
            if narrative == 'sintel' and condition == 'perceived_movie':
                reason = 'Sintel has no reference TextGrid; excluded from the fixed three-film pool'
            elif narrative not in candidate_ids:
                reason = f'Unknown narrative: {narrative}'
            elif not run.get('stimulus_path'):
                reason = 'Run has no reference TextGrid'
            if reason:
                report['skipped_runs'].append(dict(run_id=run['run_id'], reason=reason))
            else:
                runs.append(run)
        references = []
        for narrative in candidate_ids:
            try:
                vector, artifact = _cached_reference(condition, narrative,
                    [r for r in runs if _narrative(r) == narrative], transfer_path, output, pin,
                    get_encoder, config.get('evaluation', {}).get('reference_dir'), feature_batch_size)
            except (FileNotFoundError, ValueError, IndexError) as error:
                report['missing_reasons'].append(dict(narrative=narrative, reason=str(error)))
                continue
            references.append(vector)
            report['reference_artifacts'][narrative] = artifact
        if report['missing_reasons']:
            report['status'] = 'unscored_missing_fixed_candidates'
            continue  # Never shrink the fixed pool or open unscorable neural runs.
        predictions, identities = [], []
        for run in runs:
            if not _path(run['neural_path'], transfer_path).is_file():
                report['skipped_runs'].append(dict(run_id=run['run_id'], reason='Normalized neural array is missing'))
                continue
            vector, record = predict_run(decoder, run, transfer_path)
            report['runs'].append(record)
            if vector is None:
                report['skipped_runs'].append(dict(run_id=run['run_id'], reason='No finite eligible predictions'))
                continue
            identity = dict(run_id=run['run_id'], narrative=_narrative(run), repeat=run.get('repeat'))
            predictions.append(vector)
            identities.append(identity)
            all_predictions.append(vector)
            all_identities.append(dict(identity, condition=condition))
            counts.append(record['n_tr_averaged'])
        report['complete_run_coverage'] = len(predictions) == len(runs) and not report['skipped_runs']
        if not predictions:
            report['status'] = 'unscored_no_valid_predictions'
            continue
        predictions, references = np.stack(predictions), np.stack(references)
        report['raw'] = retrieval_summary(predictions, references, identities, candidate_ids)
        report['constant_training_mean'] = retrieval_summary(
            np.broadcast_to(decoder.target_mean, predictions.shape), references, identities, candidate_ids)
        report['training_mean_centered'] = retrieval_summary(
            predictions - decoder.target_mean, references - decoder.target_mean, identities, candidate_ids)
        report['status'] = 'scored' if not report['raw']['missing_reason'] else 'unscored_undefined_cosine'
    result['skipped_runs'] = [r for report in result['conditions'].values() for r in report['skipped_runs']]
    result['skipped_runs'].extend(dict(run_id=r['run_id'], reason=f"Unsupported condition: {r.get('condition')}")
                                  for r in manifest['runs'] if r.get('condition') not in CANDIDATES)
    for name in ('training', 'training_neural', 'transfer', 'transfer_neural'):
        if file_sha256(provenance[f'{name}_manifest_path']) != provenance[f'{name}_manifest_sha256']:
            raise ValueError('Input manifest changed during evaluation; results not published')
    for norm in provenance['normalization'].values():
        if isinstance(norm, dict) and file_sha256(norm['path']) != norm['sha256']:
            raise ValueError('Normalization changed during evaluation; results not published')
    prediction_path = output / 'predictions.npz'
    temporary = prediction_path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle,
            predictions=np.asarray(all_predictions, dtype=np.float16).reshape(-1, 768),
            run_ids=np.array([r['run_id'] for r in all_identities], dtype=str),
            narratives=np.array([r['narrative'] for r in all_identities], dtype=str),
            conditions=np.array([r['condition'] for r in all_identities], dtype=str),
            n_tr_averaged=np.array(counts, dtype=np.int64))
    temporary.replace(prediction_path)
    result['prediction_artifact'] = dict(path=str(prediction_path), sha256=file_sha256(prediction_path),
                                         dtype='float16', scoring_precision='float32 run vectors')
    destination = output / 'results.json'
    save_json(result, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/v2/phase_a.yaml')
    parser.add_argument('--set', action='append', default=[], help='Repeatable key=value config override')
    parser.add_argument('--transfer-manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--feature-batch-size', type=int, default=16)
    args = parser.parse_args()
    destination = run_evaluation(load_config(args.config, args.set), args.transfer_manifest,
                                 args.out, args.device, args.feature_batch_size)
    print(destination, flush=True)


if __name__ == '__main__':
    main()
