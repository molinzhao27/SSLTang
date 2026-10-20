"""Held-out, within-subject narrative matching; no fitting or text generation.

Reference TextGrid intervals establish word order, not a transfer target clock.
Transfer arrays are opened only by the explicit evaluation entry point.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from ..models import FmriEncoder
from ..runtime import file_sha256, git_commit, save_json
from ..semantic import features
from ..semantic.model import SemanticRegressor


CANDIDATES = {
    'imagined_speech': ('alpha', 'bravo', 'charlie', 'delta', 'echo'),
    'perceived_movie': ('laluna', 'partlycloudy', 'presto'),
}
NORMALIZATION_KEYS = ('mean', 'std', 'membership', 'coords', 'subject',
                      'fit_run_ids', 'n_fit_tr', 'block_size')
PIN_KEYS = ('model', 'revision', 'layer', 'context_words', 'token_pool', 'frozen')
TIE_ATOL = 1e-8


def _path(value, owner):
    """Manifest paths belong to the manifest, independent of the current CWD."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(owner).parent / path).resolve()


def _array_hash(values):
    import hashlib
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(json.dumps(list(values.shape)).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _read_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if not isinstance(manifest, dict) or not isinstance(manifest.get('runs'), list):
        raise ValueError(f'{path}: require a neural/semantic manifest object with runs')
    if not manifest['runs']:
        raise ValueError(f'{path}: no neural runs; use the original transfer neural manifest')
    ids = [r['run_id'] for r in manifest['runs']]
    if len(set(ids)) != len(ids):
        raise ValueError(f'{path}: duplicate run IDs')
    return manifest


def _source_manifest(manifest, path):
    if not manifest.get('neural_manifest_path'):
        return manifest, Path(path), None
    source = _path(manifest['neural_manifest_path'], path)
    actual = file_sha256(source)
    if actual != manifest.get('neural_manifest_hash'):
        raise ValueError(f'{path}: source neural manifest hash does not match')
    return _read_manifest(source), source, actual


def _normalization(manifest, path, subject, n_features, training_runs):
    if manifest.get('normalized') is not True or not manifest.get('normalization_path'):
        raise ValueError(f'{path}: normalized arrays and training normalization provenance required')
    norm_path = _path(manifest['normalization_path'], path)
    with np.load(norm_path, allow_pickle=False) as state:
        if any(key not in state for key in NORMALIZATION_KEYS):
            raise ValueError(f'{norm_path}: incomplete preprocessor normalization state')
        arrays = {key: np.array(state[key], copy=True) for key in NORMALIZATION_KEYS}
    if arrays['subject'].shape != () or str(arrays['subject'].item()) != subject:
        raise ValueError('Normalization subject differs from checkpoint training subject')
    for key in ('mean', 'std'):
        if arrays[key].shape != (n_features,) or not np.isfinite(arrays[key]).all():
            raise ValueError(f'Normalization {key} must match the checkpoint feature dimension')
    if np.any(arrays['std'] <= 0):
        raise ValueError('Normalization std must be positive')
    membership, coords = arrays['membership'], arrays['coords']
    if (membership.ndim != 1 or not np.issubdtype(membership.dtype, np.integer)
            or coords.shape != (len(membership), 3) or not np.isfinite(coords).all()
            or not np.array_equal(np.unique(membership), np.arange(n_features))):
        raise ValueError('Normalization spatial grouping is malformed')
    fit_ids = list(map(str, arrays['fit_run_ids'].reshape(-1)))
    train_ids = {r['run_id'] for r in training_runs if r['split'] == 'train'}
    if not fit_ids or len(set(fit_ids)) != len(fit_ids) or set(fit_ids) != train_ids:
        raise ValueError('Normalization must be fit on exactly the source ds003020 training runs')
    expected_tr = sum(int(r['n_tr']) for r in training_runs if r['run_id'] in fit_ids)
    if arrays['n_fit_tr'].shape != () or int(arrays['n_fit_tr']) != expected_tr:
        raise ValueError('Normalization fitted TR count differs from training provenance')
    if arrays['block_size'].shape != () or int(arrays['block_size']) < 1:
        raise ValueError('Invalid spatial block size')
    return arrays, {'path': str(norm_path), 'sha256': file_sha256(norm_path),
                    'array_sha256': {k: _array_hash(v) for k, v in arrays.items()},
                    'fit_run_ids': fit_ids, 'n_fit_tr': expected_tr,
                    'block_size': int(arrays['block_size'])}


def _frozen_pin(manifest, target_dim):
    header = manifest.get('semantic_metadata', {})
    metadata = [r.get('semantic_metadata', header) for r in manifest['runs']]
    if not metadata or any(any(k not in item for k in PIN_KEYS) for item in metadata):
        raise ValueError('Training semantic manifest lacks frozen language encoder provenance')
    pin = {k: metadata[0][k] for k in PIN_KEYS}
    if any({k: item[k] for k in PIN_KEYS} != pin for item in metadata):
        raise ValueError('Training targets use inconsistent frozen encoders')
    if header and any(header.get(k) != pin[k] for k in PIN_KEYS):
        raise ValueError('Training semantic header and run metadata disagree')
    expected = dict(model=features.MODEL, revision=features.REVISION, layer=9,
                    context_words=20, token_pool='last subword of current word', frozen=True)
    if pin != expected or target_dim != 768:
        raise ValueError('Transfer protocol requires the contracted pinned GPT2 layer-9, 768-D targets')
    if any(int(item.get('embedding_dim', target_dim)) != target_dim for item in metadata):
        raise ValueError('Training target metadata dimension differs from checkpoint')
    return dict(pin, embedding_dim=target_dim,
                reference_pooling='whole-reference mean of causal word features')


def load_transfer_checkpoint(checkpoint_path, transfer_manifest_path, device='cpu'):
    """Validate provenance without opening any training or transfer neural array."""
    checkpoint_path, transfer_manifest_path = Path(checkpoint_path).resolve(), Path(transfer_manifest_path).resolve()
    checkpoint_hash = file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('task') != 'semantic':
        raise ValueError('A complete semantic training checkpoint is required')
    cfg = checkpoint['config']
    # Config paths, unlike manifest paths, follow the repository's CWD contract.
    training_path = Path(cfg['data']['manifest']).resolve()
    training_hash = file_sha256(training_path)
    if training_hash != checkpoint.get('manifest_sha256'):
        raise ValueError('Training semantic manifest differs from checkpoint SHA256')
    training = _read_manifest(training_path)
    source_training, source_training_path, source_training_hash = _source_manifest(training, training_path)
    transfer = _read_manifest(transfer_manifest_path)
    source_transfer, source_transfer_path, source_transfer_hash = _source_manifest(transfer, transfer_manifest_path)
    if any(r.get('dataset') != 'ds003020' for r in source_training['runs'] + training['runs']):
        raise ValueError('Checkpoint training provenance must contain only ds003020')
    if any(r.get('dataset') != 'ds004510' or r.get('split') != 'transfer'
           for r in transfer['runs'] + source_transfer['runs']):
        raise ValueError('Transfer manifest must contain only held-out ds004510 transfer runs')
    # Semantic enrichment can drop imagined runs. Do not silently lose repeats.
    if {r['run_id'] for r in transfer['runs']} != {r['run_id'] for r in source_transfer['runs']}:
        raise ValueError('Transfer semantic manifest omits source runs; use the neural transfer manifest')
    subjects = {r['subject'] for r in training['runs'] + source_training['runs'] + transfer['runs']}
    if len(subjects) != 1:
        raise ValueError('Transfer and training must use the same single subject')
    subject = next(iter(subjects))
    encoder_cfg = checkpoint['encoder_config']
    n_features = int(encoder_cfg['n_features'])
    if any(int(r['n_features']) != n_features for r in training['runs'] + source_training['runs'] + transfer['runs']):
        raise ValueError('Training/transfer feature dimensions differ from checkpoint')
    for manifest in (training, source_training, transfer, source_transfer):
        if manifest.get('subject', subject) != subject:
            raise ValueError('Manifest header subject differs from runs')
    if checkpoint.get('target_mean') is None or checkpoint.get('target_std') is None:
        raise ValueError('Semantic checkpoint must include training target mean/std')
    mean = torch.as_tensor(checkpoint['target_mean'], dtype=torch.float32).cpu()
    std = torch.as_tensor(checkpoint['target_std'], dtype=torch.float32).cpu()
    if (mean.ndim != 1 or std.shape != mean.shape or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or not (std > 0).all()):
        raise ValueError('Checkpoint target mean/std must be finite vectors with positive std')
    semantic = cfg.get('semantic', {})
    if (int(cfg['data'].get('window', 32)) != 32 or int(semantic.get('lag_tr', 3)) != 3
            or semantic.get('pooling', 'end') != 'end'):
        raise ValueError('This protocol requires training window=32, lag_tr=3 and end pooling')
    encoder = FmriEncoder(encoder_cfg)
    if encoder.config.max_length < 32:
        raise ValueError('Encoder must support the contracted 32-TR context')
    expected_cfg = asdict(encoder.config)
    for key, value in cfg.get('model', {}).items():
        if expected_cfg.get(key) != value:
            raise ValueError(f'Checkpoint encoder/config mismatch: {key}')
    pin = _frozen_pin(training, len(mean))
    _, train_norm_info = _normalization(training, training_path, subject, n_features, source_training['runs'])
    _, transfer_norm_info = _normalization(transfer, transfer_manifest_path, subject, n_features, source_training['runs'])
    exact_file = train_norm_info['sha256'] == transfer_norm_info['sha256']
    if not exact_file and transfer_norm_info['array_sha256'] != train_norm_info['array_sha256']:
        raise ValueError('Transfer normalization/grouping differs from the saved training preprocessor')
    for manifest, path in ((source_training, source_training_path), (source_transfer, source_transfer_path)):
        _, source_norm_info = _normalization(manifest, path, subject, n_features, source_training['runs'])
        if source_norm_info['array_sha256'] != train_norm_info['array_sha256']:
            raise ValueError('Source neural manifest normalization differs from semantic manifest provenance')
    if not training.get('feature_order') or any(
            m.get('feature_order') != training['feature_order']
            for m in (source_training, transfer, source_transfer)):
        raise ValueError('Explicit training/transfer feature order provenance must agree')
    # Enriched manifests must retain all neural run identity fields unchanged.
    for enriched, original, owner, original_owner in (
            (training, source_training, training_path, source_training_path),
            (transfer, source_transfer, transfer_manifest_path, source_transfer_path)):
        original_runs = {r['run_id']: r for r in original['runs']}
        for run in enriched['runs']:
            old = original_runs.get(run['run_id'])
            if old is None or any(run.get(k) != old.get(k) for k in
                                  ('dataset', 'subject', 'split', 'n_features', 'n_tr', 'condition', 'story', 'repeat')):
                raise ValueError('Semantic enrichment changed neural run identity')
            if _path(run['neural_path'], owner) != _path(old['neural_path'], original_owner):
                raise ValueError('Semantic enrichment changed neural array provenance')
    model = SemanticRegressor(encoder, len(mean), pooling='end')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device).eval().requires_grad_(False)
    provenance = dict(checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
        task=checkpoint['task'], global_step=checkpoint.get('global_step'), seed=checkpoint.get('seed'),
        training_git_commit=checkpoint.get('git_commit'), training_config=cfg,
        training_manifest_path=str(training_path), training_manifest_sha256=training_hash,
        training_neural_manifest_path=str(source_training_path), training_neural_manifest_sha256=source_training_hash or training_hash,
        transfer_manifest_path=str(transfer_manifest_path), transfer_manifest_sha256=file_sha256(transfer_manifest_path),
        transfer_neural_manifest_path=str(source_transfer_path),
        transfer_neural_manifest_sha256=source_transfer_hash or file_sha256(transfer_manifest_path),
        subject=subject, n_features=n_features, encoder_config=expected_cfg,
        frozen_reference_encoder=pin, normalization=dict(training=train_norm_info, transfer=transfer_norm_info,
            match='exact_file_sha256' if exact_file else 'exact_saved_arrays', fit_during_evaluation=False),
        target_mean_sha256=_array_hash(mean.numpy()), target_std_sha256=_array_hash(std.numpy()),
        feature_order=training['feature_order'],
        coordinate_identity_evidence=transfer.get('coordinate_identity_evidence',
            'Published within-subject column convention; no independently embedded H5 coordinate map'))
    return model, mean.numpy(), std.numpy(), transfer, provenance


def prefix_windows(n_tr, min_window=8, max_window=32, lag_tr=3):
    """One run-local window per endpoint; target clock is end minus saved lag."""
    if n_tr < 1 or min_window < 1 or max_window < min_window or lag_tr < 0:
        raise ValueError('Invalid prefix window parameters')
    return [(max(0, end-max_window+1), end, end-lag_tr)
            for end in range(min_window-1, n_tr) if 0 <= end-lag_tr < n_tr]


def predict_run(model, run, manifest_path, target_mean, target_std, device, batch_size=64):
    """Inverse target scaling, then average all finite endpoint predictions."""
    path = _path(run['neural_path'], manifest_path)
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    if values.shape != (int(run['n_tr']), int(run['n_features'])) or values.dtype != np.float32:
        raise ValueError(f"{run['run_id']}: expected normalized float32 [n_tr,n_features]")
    windows = prefix_windows(len(values))
    groups = {}
    for start, end, target in windows:
        groups.setdefault(end-start+1, []).append((start, end, target))
    selected, skipped, predictions = [], [], []
    with torch.inference_mode():
        for length in sorted(groups):
            group = groups[length]
            for first in range(0, len(group), batch_size):
                batch, identities = [], []
                for start, end, target in group[first:first+batch_size]:
                    x = np.array(values[start:end+1], dtype=np.float32, copy=True)
                    if not np.isfinite(x).all():
                        skipped.append(dict(end_tr=end, reason='Nonfinite normalized neural window'))
                        continue
                    batch.append(x)
                    identities.append(dict(start_tr=start, end_tr=end, model_target_tr=target, window_tr=length))
                if not batch:
                    continue
                y = model(torch.from_numpy(np.stack(batch)).to(device)).float().cpu().numpy()
                y = y * target_std + target_mean
                for identity, prediction in zip(identities, y):
                    if not np.isfinite(prediction).all():
                        skipped.append(dict(end_tr=identity['end_tr'], reason='Nonfinite model prediction'))
                        continue
                    selected.append(identity)
                    predictions.append(prediction)
    record = dict(run_id=run['run_id'], neural_path=str(path), neural_sha256=file_sha256(path),
                  n_tr=len(values), n_proposed_windows=len(windows), n_selected_windows=len(selected),
                  selected_windows=selected, skipped_windows=skipped)
    if not predictions:
        return None, record
    prediction = np.mean(np.stack(predictions).astype(np.float64), axis=0).astype(np.float32)
    return prediction, record


def _narrative(run):
    story = run.get('story') or run.get('stimulus_id')
    if not story:
        story = Path(run.get('source_key', run['run_id'])).stem
    return str(story).split('_repeat-')[0].lower()


def _reference_text(path, reference_dir):
    if path.suffix.lower() != '.textgrid':
        raise ValueError('Whole-reference proxy requires a supplied word TextGrid')
    intervals = features.read_word_intervals(path, reference_dir)
    # Times are deliberately discarded: only supplied word order is used.
    text = ' '.join(word for _, _, word in intervals)
    return text, len(text.split())


def _cached_reference(condition, narrative, runs, manifest_path, output, pin, encoder_factory,
                      reference_dir, feature_batch_size):
    sources = []
    texts = []
    for run in runs:
        if not run.get('stimulus_path'):
            continue
        path = _path(run['stimulus_path'], manifest_path)
        if not path.is_file():
            raise FileNotFoundError(f'Supplied reference TextGrid is missing: {path}')
        text, word_count = _reference_text(path, reference_dir)
        texts.append(text)
        sources.append(dict(run_id=run['run_id'], path=str(path), sha256=file_sha256(path), word_count=word_count))
    if not texts:
        raise ValueError('No supplied reference TextGrid for this narrative')
    if any(text != texts[0] for text in texts):
        raise ValueError('Repeat references disagree; cannot choose one narrative vector')
    parser_root = Path(reference_dir) if reference_dir else Path(__file__).resolve().parents[2] / 'reference/semantic-decoding'
    parser_path = parser_root / 'decoding/utils_ridge/textgrid.py'
    payload = dict(schema_version=1, condition=condition, narrative=narrative,
                   run_ids=sorted(r['run_id'] for r in runs), reference_sources=sources,
                   text_sha256=features.fingerprint({'text': texts[0]}), frozen_encoder=pin,
                   parser_sha256=file_sha256(parser_path), features_sha256=file_sha256(features.__file__),
                   feature_batch_size=feature_batch_size, representation='run_reference_embedding')
    cache_key = features.fingerprint(payload)
    cache_path = output / 'reference_targets' / f'{cache_key}.npy'
    metadata_path = cache_path.with_suffix('.json')
    cached = False
    if cache_path.is_file() and metadata_path.is_file():
        try:
            saved = json.loads(metadata_path.read_text())
            vector = np.load(cache_path, allow_pickle=False)
            cached = (isinstance(saved, dict) and isinstance(vector, np.ndarray)
                      and saved.get('payload') == payload and saved.get('vector_sha256') == file_sha256(cache_path)
                      and vector.shape == (pin['embedding_dim'],) and vector.dtype == np.float32
                      and np.isfinite(vector).all())
        except (OSError, ValueError, KeyError, EOFError):
            cached = False
    if not cached:
        vector = features.run_reference_embedding(texts[0], encoder_factory(), feature_batch_size)
        if vector.shape != (pin['embedding_dim'],) or vector.dtype != np.float32 or not np.isfinite(vector).all():
            raise ValueError('Frozen whole-reference embedding shape/dtype/values invalid')
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix('.npy.tmp')
        with temporary.open('wb') as handle:
            np.save(handle, vector, allow_pickle=False)
        temporary.replace(cache_path)
        save_json(dict(payload=payload, vector_sha256=file_sha256(cache_path)), metadata_path)
    return vector, dict(cache_key=cache_key, cache_path=str(cache_path), cache_sha256=file_sha256(cache_path),
                        cache_metadata_path=str(metadata_path), cache_metadata_sha256=file_sha256(metadata_path),
                        reused_cache=cached, **payload)


def cosine_matrix(predictions, references):
    p, r = np.asarray(predictions, dtype=np.float64), np.asarray(references, dtype=np.float64)
    if p.ndim != 2 or r.ndim != 2 or p.shape[1] != r.shape[1]:
        raise ValueError('Cosine requires two matrices with matching embedding dimensions')
    pn, rn = np.linalg.norm(p, axis=1), np.linalg.norm(r, axis=1)
    usable = np.isfinite(p).all(1) & (pn > 1e-12)
    candidate_ok = np.isfinite(r).all(1) & (rn > 1e-12)
    if not candidate_ok.all():
        # Preserve the predetermined candidate count/chance; do not drop vectors.
        return None, usable, 'At least one fixed candidate has an undefined cosine'
    normalized = np.divide(p, pn[:, None], out=np.zeros_like(p), where=pn[:, None] > 1e-12)
    similarities = normalized @ (r / rn[:, None]).T
    similarities[~usable] = np.nan
    return np.clip(similarities, -1, 1), usable, None


def _query_metrics(similarity, positive):
    score = float(similarity[positive])
    better = int(np.sum(similarity > score + TIE_ATOL))
    tied = int(np.sum(np.abs(similarity-score) <= TIE_ATOL))
    return dict(top1=1.0/tied if better == 0 else 0.0,
                mrr=float(np.mean(1.0/np.arange(better+1, better+tied+1))),
                rank=float(better + (tied+1)/2), mean_cosine=score)


def _mean_metrics(rows):
    return {key: float(np.mean([r[key] for r in rows])) if rows else None
            for key in ('top1', 'mrr', 'mean_cosine')}


def retrieval_summary(predictions, references, runs, candidate_ids):
    """Exact narrative-label null, moving every repeat of a narrative together."""
    similarities, usable, reason = cosine_matrix(predictions, references)
    n_candidates = len(candidate_ids)
    result = dict(candidate_ids=list(candidate_ids), n_candidates=n_candidates,
        top1_chance=1.0/n_candidates, mrr_chance=float(np.mean(1.0/np.arange(1, n_candidates+1))),
        tie_policy='uniform random ordering within cosine ties (absolute tolerance 1e-8)',
        missing_reason=reason, per_run={}, per_narrative={}, micro={}, narrative_macro={}, permutation=None)
    if reason:
        return result
    labels = [candidate_ids.index(r['narrative']) for r in runs]
    clusters = {}
    for i, run in enumerate(runs):
        rid, narrative = run['run_id'], run['narrative']
        if not usable[i]:
            result['per_run'][rid] = dict(missing_reason='Prediction has an undefined cosine')
            continue
        row = _query_metrics(similarities[i], labels[i])
        result['per_run'][rid] = dict(row, repeat=run.get('repeat'), narrative=narrative,
                                     cosine_by_candidate=dict(zip(candidate_ids, map(float, similarities[i]))))
        clusters.setdefault(narrative, []).append(i)
    result['n_queries'] = int(usable.sum())
    result['micro'] = _mean_metrics([result['per_run'][r['run_id']] for i, r in enumerate(runs) if usable[i]])
    for narrative, indices in clusters.items():
        rows = [result['per_run'][runs[i]['run_id']] for i in indices]
        result['per_narrative'][narrative] = dict(_mean_metrics(rows), n_repeats=len(indices),
                                                  run_ids=[runs[i]['run_id'] for i in indices])
    result['narrative_macro'] = _mean_metrics(list(result['per_narrative'].values()))
    result['narrative_macro']['n_narratives'] = len(clusters)
    if not clusters:
        result['missing_reason'] = 'No finite nonzero prediction vectors'
        return result
    null = {key: [] for key in ('top1', 'mrr', 'mean_cosine')}
    for permutation in itertools.permutations(range(n_candidates)):
        cluster_rows = []
        for indices in clusters.values():
            rows = [_query_metrics(similarities[i], permutation[labels[i]]) for i in indices]
            cluster_rows.append(_mean_metrics(rows))
        macro = _mean_metrics(cluster_rows)
        for key in null:
            null[key].append(macro[key])
    result['permutation'] = dict(method='exact permutation of narrative labels; repeats move together',
        statistic='equal-narrative macro mean; fixed candidate pool and predictions',
        n_permutations=len(null['top1']), includes_identity=True,
        metrics={key: dict(observed=result['narrative_macro'][key], null_mean=float(np.mean(values)),
                          p_greater_equal=float(np.mean(np.asarray(values) >= result['narrative_macro'][key]-TIE_ATOL)),
                          null_values=values) for key, values in null.items()})
    return result


def _protocol(condition, exclusion_tr):
    imagined = condition == 'imagined_speech'
    return dict(name='coarse_narrative_matching' if imagined else 'run_level_caption_reference_proxy',
        reference='whole-story cue/overt reference transcript mean' if imagined else 'whole supplied movie-description reference mean',
        time_resolved=False, imagined_word_timing_verified=False if imagined else None,
        movie_scene_timing_verified=False if not imagined else None,
        window=dict(min_tr=8, max_tr=32, stride_tr=1, prefix_rule='length=min(end_tr+1,32)',
                    pooling='end', lag_tr=3, model_target_tr='end_tr-3 (zero based)',
                    aggregation='mean of all finite selected inverse-scaled endpoint vectors',
                    valid_selection='full window finite; prediction finite; model_target_tr within run'),
        exclusion_tr=exclusion_tr,
        exclusion_tr_application='not applicable: run-level candidates have no TR clock; no TRs trimmed',
        candidate_unit='unique narrative, shared by all repeats',
        fitting=False, text_generation=False, original_bertscore=False,
        limitations=['Short prefix windows differ from the 32-TR training context.',
                     'Overlapping endpoints are aggregated into one run observation, not independent trials.',
                     'Reference similarity does not establish imagined words or scene-synchronized reconstruction.'])


def run_transfer_evaluation(config):
    """Explicit evaluation command. No acknowledgment gate and no adaptation."""
    cfg = config['evaluation']
    manifest_path, checkpoint_path = Path(cfg['manifest']).resolve(), Path(cfg['checkpoint']).resolve()
    output = Path(cfg['output_dir']).resolve()
    device = torch.device(cfg.get('device', config.get('training', {}).get('device', 'cpu')))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; set evaluation.device=cpu')
    batch_size, feature_batch_size = int(cfg.get('batch_size', 64)), int(cfg.get('feature_batch_size', 16))
    exclusion_tr = int(cfg.get('exclusion_tr', 10))
    if batch_size < 1 or feature_batch_size < 1 or exclusion_tr < 0:
        raise ValueError('Batch sizes must be positive and exclusion_tr nonnegative')
    if cfg.get('split', 'transfer') != 'transfer':
        raise ValueError('Transfer evaluation requires split=transfer')
    if cfg.get('protocol', 'run_level') != 'run_level':
        raise ValueError('Only the conservative run-level reference protocol is verified')
    model, mean, std, manifest, provenance = load_transfer_checkpoint(checkpoint_path, manifest_path, device)
    # Evaluation artifacts must never overwrite a source, even with a bad config.
    sources = [manifest_path, checkpoint_path, Path(provenance['training_manifest_path']),
               Path(provenance['training_neural_manifest_path']), Path(provenance['transfer_neural_manifest_path']),
               Path(provenance['normalization']['training']['path']), Path(provenance['normalization']['transfer']['path'])]
    for run in manifest['runs']:
        sources.append(_path(run['neural_path'], manifest_path))
        if run.get('stimulus_path'):
            sources.append(_path(run['stimulus_path'], manifest_path))
    if any(p == output or output in p.parents for p in sources):
        raise ValueError('Evaluation output must be separate from all immutable input artifacts')
    output.mkdir(parents=True, exist_ok=True)
    encoder = None
    pin = provenance['frozen_reference_encoder']

    def get_encoder():
        nonlocal encoder
        if encoder is None:
            encoder = features.FrozenGPT2(model=pin['model'], revision=pin['revision'],
                layer=pin['layer'], context_words=pin['context_words'],
                device=cfg.get('reference_device', str(device)))
        return encoder

    result = dict(schema_version=1, dataset='ds004510', subject=provenance['subject'],
        created_at=datetime.now(timezone.utc).isoformat(), evaluation_git_commit=git_commit(),
        evaluation_config=cfg, provenance=provenance,
        code_sha256=dict(transfer=file_sha256(__file__), features=file_sha256(features.__file__),
                        encoder=file_sha256(Path(__file__).parents[1] / 'models/encoder.py'),
                        semantic_model=file_sha256(Path(__file__).parents[1] / 'semantic/model.py')),
        inference=dict(device=str(device), precision='fp32', batch_size=batch_size,
                       feature_batch_size=feature_batch_size), conditions={}, skipped_runs=[])
    runs_by_condition = {condition: [] for condition in CANDIDATES}
    skipped_by_condition = {condition: [] for condition in CANDIDATES}
    for run in manifest['runs']:
        condition = run.get('condition')
        if condition not in CANDIDATES:
            result['skipped_runs'].append(dict(run_id=run['run_id'], reason=f'Unsupported condition: {condition}'))
        elif condition == 'perceived_movie' and _narrative(run) == 'sintel':
            skipped_by_condition[condition].append(dict(run_id=run['run_id'], condition=condition, narrative='sintel',
                reason='Sintel has no published reference TextGrid; excluded from the fixed three-candidate proxy'))
        elif _narrative(run) not in CANDIDATES[condition]:
            skipped_by_condition[condition].append(dict(run_id=run['run_id'], condition=condition,
                reason=f'Unknown narrative: {_narrative(run)}'))
        else:
            runs_by_condition[condition].append(run)
    for condition, candidate_ids in CANDIDATES.items():
        runs = runs_by_condition[condition]
        report = dict(protocol=_protocol(condition, exclusion_tr), candidate_ids=list(candidate_ids),
                      top1_chance=1/len(candidate_ids), reference_artifacts={}, runs=[],
                      skipped_runs=skipped_by_condition[condition],
                      missing_reasons=[], raw=None, training_mean_centered=None)
        result['conditions'][condition] = report
        references = []
        for narrative in candidate_ids:
            members = [r for r in runs if _narrative(r) == narrative]
            try:
                vector, artifact = _cached_reference(condition, narrative, members, manifest_path, output,
                    pin, get_encoder, cfg.get('reference_dir'), feature_batch_size)
            except (FileNotFoundError, ValueError, IndexError) as error:
                report['missing_reasons'].append(dict(narrative=narrative, reason=str(error)))
                continue
            references.append(vector)
            report['reference_artifacts'][narrative] = artifact
        if report['missing_reasons']:
            report['status'] = 'unscored_missing_fixed_candidates'
            # Do not reduce candidate count or inspect neural data for an unscorable condition.
            continue
        predictions, identities = [], []
        for run in runs:
            if not _path(run['neural_path'], manifest_path).is_file():
                report['skipped_runs'].append(dict(run_id=run['run_id'], reason='Normalized neural array is missing'))
                continue
            prediction, record = predict_run(model, run, manifest_path, mean, std, device, batch_size)
            record.update(narrative=_narrative(run), repeat=run.get('repeat'))
            report['runs'].append(record)
            if prediction is None:
                report['skipped_runs'].append(dict(run_id=run['run_id'], reason='No valid prefix windows/predictions'))
                continue
            predictions.append(prediction)
            identities.append(dict(run_id=run['run_id'], narrative=_narrative(run), repeat=run.get('repeat')))
        if not predictions:
            report['status'] = 'unscored_no_valid_predictions'
            report['missing_reasons'].append(dict(reason='No valid run-level neural predictions'))
            continue
        predictions, references = np.stack(predictions), np.stack(references)
        prediction_path = output / f'{condition}.predictions.npz'
        temporary = prediction_path.with_suffix('.npz.tmp')
        with temporary.open('wb') as handle:
            np.savez(handle, predictions=predictions, references=references,
                     run_ids=np.array([r['run_id'] for r in identities]),
                     narratives=np.array([r['narrative'] for r in identities]),
                     candidate_ids=np.array(candidate_ids))
        temporary.replace(prediction_path)
        report['prediction_artifact'] = dict(path=str(prediction_path), sha256=file_sha256(prediction_path))
        report['raw'] = retrieval_summary(predictions, references, identities, candidate_ids)
        report['constant_training_mean'] = retrieval_summary(np.broadcast_to(mean, predictions.shape), references, identities, candidate_ids)
        report['training_mean_centered'] = retrieval_summary(predictions-mean, references-mean, identities, candidate_ids)
        report['centering'] = 'Subtract checkpoint target_mean from both prediction and reference; no transfer fit'
        report['status'] = 'scored' if not report['raw']['missing_reason'] else 'unscored_undefined_cosine'
        report['complete_run_coverage'] = len(predictions) == len(runs)
    result['skipped_runs'].extend(r for report in result['conditions'].values() for r in report['skipped_runs'])
    # Detect source/checkpoint replacement during a long evaluation before publishing results.
    if file_sha256(checkpoint_path) != provenance['checkpoint_sha256']:
        raise ValueError('Checkpoint changed during evaluation; results not published')
    if file_sha256(manifest_path) != provenance['transfer_manifest_sha256']:
        raise ValueError('Transfer manifest changed during evaluation; results not published')
    for path_key, hash_key in (
            ('training_manifest_path', 'training_manifest_sha256'),
            ('training_neural_manifest_path', 'training_neural_manifest_sha256'),
            ('transfer_neural_manifest_path', 'transfer_neural_manifest_sha256')):
        if file_sha256(provenance[path_key]) != provenance[hash_key]:
            raise ValueError('Input manifest changed during evaluation; results not published')
    for norm in provenance['normalization'].values():
        if isinstance(norm, dict) and 'path' in norm and file_sha256(norm['path']) != norm['sha256']:
            raise ValueError('Normalization changed during evaluation; results not published')
    destination = output / 'results.json'
    save_json(result, destination)
    return destination
