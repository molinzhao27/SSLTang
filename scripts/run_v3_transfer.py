#!/usr/bin/env python3
"""Exploratory ds004510 retrieval with the saved dense-clock V0/V4 decoders.

Transfer recordings were already exposed in v2. No parameters are selected here.
Raw voxel rows use the dense final fit's training statistics and kernel geometry.
Both validation checks must pass for both decoders before retrieval is scored.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v3 as v3
from run_v3_dense import DenseContext
from scripts.run_ridge_transfer import validate_manifests
from src.evaluation.endpoints import score
from src.evaluation.transfer import CANDIDATES, _array_hash, _cached_reference, _narrative, _path, retrieval_summary
from src.multikernel_ridge import KernelRidgePath, combine
from src.runtime import file_sha256, load_config, save_json
from src.semantic import features
from src.structured_ridge import BASELINE, candidates, coefficients, label
from src.voxel_kernels import DELAYS, _h5_path

PROTOCOL_ID = 'ssltang-v3-voxel-transfer'
EXPLORATORY_REASON = ('ds004510 imagined-speech and film retrieval data were already scored in v2; '
                      'this is an exploratory re-evaluation, not a confirmatory test.')


def cache_pseudo_split(runs, raw_root, cache_dir, n_voxels, split='transfer', block_rows=64):
    """Cache every eligible t using H5 rows t+d, without targets or pooled BOLD.

    Stream small contiguous blocks into disk-backed arrays; never materialize a
    full voxel design in host memory. Metadata records the exact (run_id, t)
    ordering consumed by KernelBuilder. Rebuild to avoid stale inference caches.
    """
    if block_rows < 1 or n_voxels < 1:
        raise ValueError('Positive block size and training voxel count required')
    if len({r['run_id'] for r in runs}) != len(runs):
        raise ValueError('Pseudo-split run IDs must be unique')
    identities, sources = [], []
    for run in runs:
        path = _h5_path(run, raw_root)
        with h5py.File(path, 'r') as handle:
            if handle['data'].shape != (int(run['n_tr']), n_voxels):
                raise ValueError(f"{run['run_id']}: H5 shape differs from n_tr/training voxel count")
        stat = path.stat()
        sources.append(dict(run_id=run['run_id'], source_key=run['source_key'], path=str(path.resolve()),
                            size=stat.st_size, mtime_ns=stat.st_mtime_ns))
        identities.extend((run['run_id'], t) for t in range(max(0, int(run['n_tr']) - max(DELAYS))))
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {d: cache_dir / f'{split}_d{d}.npy' for d in DELAYS}
    temporary = {d: p.with_suffix('.tmp.npy') for d, p in paths.items()}
    arrays = {d: np.lib.format.open_memmap(temporary[d], mode='w+', dtype=np.float32,
                                          shape=(len(identities), n_voxels)) for d in DELAYS}
    offset = 0
    for run in runs:
        n = max(0, int(run['n_tr']) - max(DELAYS))
        with h5py.File(_h5_path(run, raw_root), 'r') as handle:
            for start in range(0, n, block_rows):
                stop = min(n, start + block_rows)
                for d in DELAYS:
                    block = handle['data'][start + d:stop + d]
                    np.nan_to_num(block, copy=False, nan=0, posinf=0, neginf=0)
                    arrays[d][offset + start:offset + stop] = block.astype(np.float32)
        offset += n
    for d in DELAYS:
        arrays[d].flush()
        del arrays[d]
        temporary[d].replace(paths[d])
    metadata = dict(split=split, delays=list(DELAYS), n_voxels=int(n_voxels), dtype='float32',
                    n_rows=len(identities), rows=identities, sources=sources,
                    rule='every t=0..n_tr-4; raw H5 t+d; non-finite values set to 0')
    save_json(metadata, cache_dir / f'{split}.meta.json')
    return paths, identities, metadata


def fixed_decoders(ctx):
    """Read archived structure/alpha only; reject incompatible dense cells."""
    root = Path(ctx.cfg['output_root']) / ctx.subject
    structure_path = root / 'evaluate' / 'cells' / 'V4_f1.json'
    advanced = json.loads(structure_path.read_text())
    if any(advanced.get(k) != v for k, v in dict(protocol='ssltang-v3', stage='evaluate',
                                               subject=ctx.subject, arm='V4').items()):
        raise ValueError('V4 structure cell has incompatible identity')
    eta, tau = advanced['selection']['candidate']
    structure = (float(eta), tuple(tau))
    if structure not in candidates():
        raise ValueError('V4 structure is outside the locked grid')
    out = {}
    for arm, candidate in (('V0', BASELINE), ('V4', structure)):
        path = root / 'dense' / f'{arm}_dense.json'
        cell = json.loads(path.read_text())
        expected = dict(protocol='ssltang-v3', stage='dense', subject=ctx.subject, arm=arm,
                        candidate=label(candidate), n_train_targets=len(ctx.index['train']))
        if any(cell.get(k) != v for k, v in expected.items()):
            raise ValueError(f'{arm}: dense cell does not match the current dense training design')
        alpha = float(cell['alpha'])
        validation_r = cell['val']['session_macro_feature_correlation']
        if (not np.isfinite(alpha) or alpha <= 0 or validation_r is None
                or not np.isfinite(validation_r)):
            raise ValueError(f'{arm}: nonfinite saved alpha/validation score')
        sources = dict(alpha=dict(path=str(path.resolve()), sha256=file_sha256(path)),
                       candidate=dict(path=str((structure_path if arm == 'V4' else path).resolve()),
                                      sha256=file_sha256(structure_path if arm == 'V4' else path)))
        out[arm] = dict(candidate=candidate, eta=candidate[0], tau=list(candidate[1]), alpha=alpha,
                        saved_validation_r=float(validation_r), sources=sources, selection='none; archived parameters')
    return out


def validation_consistency(standard, pseudo, identities, validation, expected_r,
                           target_mean, target_std, exclusions=(10, 32)):
    """Align by identities, then gate on prediction RMS error and saved macro r."""
    standard, pseudo = np.asarray(standard), np.asarray(pseudo)
    lookup = {(str(run), int(t)): i for i, (run, t) in enumerate(identities)}
    if len(lookup) != len(identities):
        raise ValueError('Duplicate validation pseudo-split identities')
    try:
        rows = [lookup[(str(run), int(t))] for run, t in zip(validation.run_ids, validation.target_trs)]
    except KeyError as error:
        raise ValueError(f'Missing validation pseudo-split target: {error}') from error
    aligned = pseudo[rows]
    if (aligned.shape != standard.shape or not np.isfinite(aligned).all()
            or not np.isfinite(standard).all()):
        raise ValueError('Validation prediction shape/finite consistency failed')
    rms = float(np.sqrt(np.mean(np.square(standard))))
    error_rms = float(np.sqrt(np.mean(np.square(aligned - standard))))
    relative = error_rms / max(rms, np.finfo(np.float64).tiny)
    if relative > 1e-8:
        raise ValueError(f'Validation pseudo-split prediction consistency failed: relative RMS {relative:g} > 1e-8')
    actual_r = score(aligned, validation.targets, validation.run_ids, validation.target_trs,
                     validation.sessions, target_mean, target_std, exclusions)['session_macro_feature_correlation']
    if (actual_r is None or expected_r is None or not np.isfinite(actual_r)
            or not np.isfinite(expected_r) or abs(actual_r - expected_r) > 1e-6):
        raise ValueError(f'Validation dense-cell consistency failed: {actual_r} versus {expected_r}')
    return dict(passed=True, n_validation_targets=len(rows), prediction_rms=rms,
                prediction_error_rms=error_rms, prediction_relative_rms_error=relative,
                prediction_relative_rms_tolerance=1e-8, validation_session_macro_r=float(actual_r),
                saved_validation_session_macro_r=float(expected_r),
                validation_r_absolute_error=float(abs(actual_r - expected_r)), validation_r_tolerance=1e-6)


def mean_run_predictions(predictions, identities, runs):
    """Mean finite predicted vectors over eligible t, in each manifest run."""
    predictions = np.asarray(predictions)
    if predictions.ndim != 2 or len(predictions) != len(identities):
        raise ValueError('Predictions must align with pseudo-split row identities')
    by_run = {}
    for position, (run_id, t) in enumerate(identities):
        by_run.setdefault(run_id, []).append((position, int(t)))
    out = {}
    for run in runs:
        items = by_run.get(run['run_id'], [])
        positions = [i for i, _ in items]
        values = predictions[positions]
        finite = np.isfinite(values).all(1)
        kept = [t for (_, t), ok in zip(items, finite) if ok]
        record = dict(run_id=run['run_id'], narrative=_narrative(run), repeat=run.get('repeat'),
                      source_key=run['source_key'], n_tr=int(run['n_tr']), n_eligible_tr=len(items),
                      n_tr_averaged=len(kept), target_trs_averaged=kept,
                      skipped_nonfinite_prediction_trs=[t for (_, t), ok in zip(items, finite) if not ok])
        vector = values[finite].mean(0, dtype=np.float64).astype(np.float32) if kept else None
        out[run['run_id']] = (vector, record)
    return out


def prepare_conditions(manifest, transfer_path, output, provenance, config, device, batch_size):
    """Reuse the v2 fixed pools and content-addressed whole-reference targets."""
    pin, encoder = provenance['frozen_reference_encoder'], None

    def get_encoder():
        nonlocal encoder
        if encoder is None:
            encoder = features.FrozenGPT2(model=pin['model'], revision=pin['revision'], layer=pin['layer'],
                                         context_words=pin['context_words'], device=device)
        return encoder

    conditions, scorable = {}, []
    for condition, candidate_ids in CANDIDATES.items():
        report = dict(candidate_ids=list(candidate_ids), top1_chance=1 / len(candidate_ids),
                      reference_artifacts={}, runs=[], skipped_runs=[], missing_reasons=[], raw=None,
                      constant_training_mean=None, training_mean_centered=None,
                      aggregation='Mean finite predictions for every t=0..n_tr-4; voxel inputs t+1,t+2,t+3',
                      centering='Subtract the dense training-target mean from predictions and references')
        runs = []
        for run in manifest['runs']:
            if run.get('condition') != condition:
                continue
            narrative, reason = _narrative(run), None
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
                    [r for r in runs if _narrative(r) == narrative], transfer_path, output, pin, get_encoder,
                    config.get('evaluation', {}).get('reference_dir'), batch_size)
            except (FileNotFoundError, ValueError, IndexError) as error:
                report['missing_reasons'].append(dict(narrative=narrative, reason=str(error)))
                continue
            references.append(vector)
            artifact['reused_cache'] = bool(artifact['reused_cache'])
            report['reference_artifacts'][narrative] = artifact
        if report['missing_reasons']:
            report['status'] = 'unscored_missing_fixed_candidates'
            runs = []  # Never shrink the candidate pool or open this condition's voxel arrays.
        else:
            report['status'] = 'pending_consistency_check'
            available = []
            for run in runs:
                if not _h5_path(run, config['raw_root']).is_file():
                    report['skipped_runs'].append(dict(run_id=run['run_id'], reason='Raw voxel H5 is missing'))
                else:
                    available.append(run)
            scorable.extend(available)
        conditions[condition] = (report, runs, references)
    return conditions, scorable


def run_evaluation(config, transfer_manifest, output, device='cuda', feature_batch_size=16):
    if feature_batch_size < 1:
        raise ValueError('feature_batch_size must be positive')
    lock = v3.verify_lock(config)
    if lock['advanced_arm'] != 'V4':
        raise ValueError('This exploratory runner requires the locked V4 advanced arm')
    training_path = Path(config['data']['manifest']).resolve()
    transfer_path, output = Path(transfer_manifest).resolve(), Path(output).resolve()
    manifest, provenance, sources = validate_manifests(training_path, transfer_path)
    ctx = DenseContext(config, device)
    if ctx.subject != provenance['subject'] or any(i.target_dim != 768 for i in ctx.index.values()):
        raise ValueError('Dense context subject/target space differs from transfer provenance')
    parameters = fixed_decoders(ctx)
    for item in parameters.values():
        sources.extend(Path(source['path']) for source in item['sources'].values())
    sources.append(Path(config['lock_file']))
    sources.extend(_h5_path(run, config['raw_root']) for run in manifest['runs'] if run.get('source_key'))
    if any(p.resolve() == output or output in p.resolve().parents for p in sources):
        raise ValueError('Evaluation output must be separate from immutable input artifacts')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prepared, runs = prepare_conditions(manifest, transfer_path, output, provenance, config,
                                         device, feature_batch_size)
    root = Path(config['cache_root'])
    cache_dir = root.parent / (root.name + '_transfer') / ctx.subject
    paths, identities, transfer_meta = cache_pseudo_split(runs, config['raw_root'], cache_dir, ctx.n_voxels)
    val_paths, val_identities, val_meta = cache_pseudo_split(ctx.index['val'].runs, config['raw_root'],
                                                          cache_dir, ctx.n_voxels, 'validation_pseudo')
    ctx.builder.cache['transfer'] = paths
    ctx.builder.cache['validation_pseudo'] = val_paths
    rows = list(range(len(ctx.index['train'])))
    # One build and one eigensolve per arm. The additional cross-kernel checks
    # the independent all-t validation cache with the identical final-fit stats.
    kernels, cross, _ = ctx.builder.build(rows, ('val', 'transfer', 'validation_pseudo'))
    y = ctx.split['train'].targets
    if not np.isfinite(y).all():
        raise ValueError('Dense training targets must be finite')
    mean = y.astype(np.float64).mean(0)
    std = np.maximum(y.std(0), 1e-6)
    predicted, checks, diagnostics = {}, {}, {}
    for arm, item in parameters.items():
        coef = coefficients(*item['candidate'])
        path = KernelRidgePath(combine(kernels, coef), y, ctx.device)
        prediction = {s: path.predict(path.project(combine(cross[s], coef)), item['alpha']).cpu().numpy()
                      for s in ('val', 'validation_pseudo', 'transfer')}
        checks[arm] = validation_consistency(prediction['val'], prediction['validation_pseudo'], val_identities,
            ctx.split['val'], item['saved_validation_r'], mean, std, config['evaluation']['exclusions'])
        predicted[arm], diagnostics[arm] = prediction['transfer'], path.diagnostics
        del path, prediction
    del kernels, cross
    # Neither decoder may write retrieval scores until every consistency gate passes.
    provenance.update(lock_code_sha256=lock['code_sha256'], lock_file_sha256=file_sha256(config['lock_file']),
                      runner_sha256=file_sha256(__file__),
                      dense_runner_sha256=file_sha256(Path(__file__).with_name('run_v3_dense.py')),
                      target_mean_sha256=_array_hash(mean), n_voxels=ctx.n_voxels,
                      spatial_groups={k: v for k, v in ctx.groups.items() if k not in ('membership', 'sizes')},
                      targets={s: i.signature() for s, i in ctx.index.items()})
    result = dict(schema_version=1, protocol=PROTOCOL_ID, exploratory=True,
                  exploratory_explanation=EXPLORATORY_REASON, dataset='ds004510', subject=ctx.subject,
                  created_at=datetime.now(timezone.utc).isoformat(), config=config, provenance=provenance,
                  consistency_check=checks, row_counts=dict(dense_train=len(rows), validation=len(ctx.index['val']),
                      validation_pseudo=len(val_identities), transfer=len(identities)),
                  cache=dict(directory=str(cache_dir.resolve()), transfer=transfer_meta, validation_pseudo=val_meta),
                  inference=dict(device=str(device), feature_batch_size=feature_batch_size,
                                 standardization='All dense training rows; no transfer statistics', selection='none'),
                  decoders={}, skipped_runs=[], limitations=['Run-level retrieval only.',
                      'No verified imagined-word timing.', 'Whole-reference embeddings are reference proxies.',
                      'Repeats are not independent observations.', '3-film null minimum p = 1/6.'])
    vectors, saved_ids, counts, saved_arms = [], [], [], []
    for arm, item in parameters.items():
        reports = {}
        result['decoders'][arm] = dict(hyperparameters=item, final_fit=diagnostics[arm], conditions=reports)
        run_vectors = mean_run_predictions(predicted[arm], identities, runs)
        for condition, (template, condition_runs, references) in prepared.items():
            report = deepcopy(template)
            reports[condition] = report
            if report['status'] == 'unscored_missing_fixed_candidates':
                continue
            predictions, run_ids = [], []
            for run in condition_runs:
                if run['run_id'] not in run_vectors:
                    continue
                vector, record = run_vectors[run['run_id']]
                report['runs'].append(record)
                if vector is None:
                    report['skipped_runs'].append(dict(run_id=run['run_id'], reason='No finite eligible predictions'))
                    continue
                identity = dict(run_id=run['run_id'], narrative=_narrative(run), repeat=run.get('repeat'))
                predictions.append(vector)
                run_ids.append(identity)
                vectors.append(vector)
                saved_ids.append(dict(identity, condition=condition))
                counts.append(record['n_tr_averaged'])
                saved_arms.append(arm)
            report['complete_run_coverage'] = len(predictions) == len(condition_runs) and not report['skipped_runs']
            if not predictions:
                report['status'] = 'unscored_no_valid_predictions'
                continue
            predictions, references_array = np.stack(predictions), np.stack(references)
            candidates_for_condition = CANDIDATES[condition]
            report['raw'] = retrieval_summary(predictions, references_array, run_ids, candidates_for_condition)
            report['constant_training_mean'] = retrieval_summary(np.broadcast_to(mean, predictions.shape),
                references_array, run_ids, candidates_for_condition)
            report['training_mean_centered'] = retrieval_summary(predictions - mean, references_array - mean,
                                                                 run_ids, candidates_for_condition)
            report['status'] = 'scored' if not report['raw']['missing_reason'] else 'unscored_undefined_cosine'
        result['skipped_runs'].extend(dict(r, decoder=arm) for report in reports.values() for r in report['skipped_runs'])
    result['skipped_runs'].extend(dict(run_id=r['run_id'], reason=f"Unsupported condition: {r.get('condition')}")
                                 for r in manifest['runs'] if r.get('condition') not in CANDIDATES)
    for name in ('training', 'training_neural', 'transfer', 'transfer_neural'):
        if file_sha256(provenance[f'{name}_manifest_path']) != provenance[f'{name}_manifest_sha256']:
            raise ValueError('Input manifest changed during evaluation; results not published')
    for norm in provenance['normalization'].values():
        if isinstance(norm, dict) and file_sha256(norm['path']) != norm['sha256']:
            raise ValueError('Normalization changed during evaluation; results not published')
    for item in parameters.values():
        for source in item['sources'].values():
            if file_sha256(source['path']) != source['sha256']:
                raise ValueError('Saved decoder cell changed during evaluation; results not published')
    v3.verify_lock(config)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / 'predictions.npz'
    temporary = prediction_path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, predictions=np.asarray(vectors, dtype=np.float16).reshape(-1, 768),
            decoders=np.array(saved_arms, dtype=str), run_ids=np.array([r['run_id'] for r in saved_ids], dtype=str),
            narratives=np.array([r['narrative'] for r in saved_ids], dtype=str),
            conditions=np.array([r['condition'] for r in saved_ids], dtype=str), n_tr_averaged=np.array(counts, np.int64))
    temporary.replace(prediction_path)
    result['prediction_artifact'] = dict(path=str(prediction_path), sha256=file_sha256(prediction_path),
                                         dtype='float16', scoring_precision='float32 run vectors')
    destination = output / 'results.json'
    save_json(result, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/v3/structured_ridge.yaml')
    parser.add_argument('--set', action='append', default=[], help='Repeatable key=value config override')
    parser.add_argument('--transfer-manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--feature-batch-size', type=int, default=16)
    args = parser.parse_args()
    print(run_evaluation(load_config(args.config, args.set), args.transfer_manifest,
                         args.out, args.device, args.feature_batch_size), flush=True)


if __name__ == '__main__':
    main()
