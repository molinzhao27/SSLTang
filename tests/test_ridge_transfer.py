"""CPU synthetic checks for the locked B1 transfer evaluator; no model downloads."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import run_ridge_transfer as transfer
from src.readout import ALPHAS, Standardizer, raw_features
from src.targets import TargetIndex
from test_v2 import manifest


def textgrid(word):
    return f'''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "word"
        xmin = 0
        xmax = 2
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 2
            text = "{word}"
'''


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    train_root, transfer_root = tmp_path / 'train', tmp_path / 'transfer'
    train_root.mkdir()
    transfer_root.mkdir()
    training_path = manifest(train_root, n_tr=(57, 54, 57, 54, 57, 54),
                             n_features=3, target_dim=768)
    training = json.loads(training_path.read_text())
    pin = dict(model=transfer.features.MODEL, revision=transfer.features.REVISION, layer=9,
               context_words=20, token_pool='last subword of current word', frozen=True,
               embedding_dim=768)
    norm_path = train_root / 'normalization.npz'
    train_runs = [r for r in training['runs'] if r['split'] == 'train']
    np.savez(norm_path, mean=np.zeros(3, np.float32), std=np.ones(3, np.float32),
             membership=np.arange(3), coords=np.arange(9).reshape(3, 3), subject=np.array('UTS01'),
             fit_run_ids=np.array([r['run_id'] for r in train_runs]),
             n_fit_tr=np.array(sum(r['n_tr'] for r in train_runs)), block_size=np.array(4))
    common = dict(subject='UTS01', normalized=True, normalization_path=str(norm_path),
                  feature_order='training block membership order')
    training.update(common, semantic_metadata=pin)
    training_path.write_text(json.dumps(training))
    rng = np.random.default_rng(8)
    runs = []
    for condition, candidates in transfer.CANDIDATES.items():
        for narrative in reversed(candidates):  # Manifest order is deliberately not candidate order.
            for repeat in range(1, 3 if condition == 'imagined_speech' else 2):
                run_id = f'{condition}_{narrative}_repeat-{repeat}'
                neural_path = transfer_root / f'{run_id}.npy'
                stimulus_path = transfer_root / f'{run_id}.TextGrid'
                np.save(neural_path, rng.normal(size=(8, 3)).astype(np.float32))
                stimulus_path.write_text(textgrid(narrative))
                runs.append(dict(run_id=run_id, dataset='ds004510', split='transfer', subject='UTS01',
                    condition=condition, story=narrative, repeat=repeat, n_tr=8, n_features=3,
                    neural_path=neural_path.name, stimulus_path=stimulus_path.name))
    # The unscored movie has no reference and intentionally no neural file.
    runs.append(dict(run_id='sintel', dataset='ds004510', split='transfer', subject='UTS01',
        condition='perceived_movie', story='sintel', repeat=1, n_tr=8, n_features=3,
        neural_path='never_open_sintel.npy', stimulus_path=None))
    transfer_path = transfer_root / 'manifest.json'
    transfer_path.write_text(json.dumps(dict(common, runs=runs)))
    words = [n for candidates in transfer.CANDIDATES.values() for n in candidates]
    factory_calls = []

    class FakeGPT2:
        def __init__(self, **kwargs):
            factory_calls.append(kwargs)

        def encode(self, tokens, batch_size=16):
            vectors = np.zeros((len(tokens), 768), np.float32)
            for i, token in enumerate(tokens):
                vectors[i, words.index(token)] = 1
            return vectors

    monkeypatch.setattr(transfer.features, 'FrozenGPT2', FakeGPT2)
    return dict(config=dict(data=dict(manifest=str(training_path), window=32, stride=4,
        target_phase=0, cutoffs=[3, 4])), training_path=training_path, transfer_path=transfer_path,
        runs=runs, factory_calls=factory_calls)


def evaluate(inputs, out):
    path = transfer.run_evaluation(inputs['config'], inputs['transfer_path'], out, 'cpu', 16)
    return json.loads(path.read_text())


def test_feature_rows_match_delay_major_b1(inputs):
    values = np.arange(24, dtype=np.float32).reshape(8, 3)
    actual = transfer.transfer_features(values)
    expected = np.stack([values[t + 1:t + 4].reshape(-1) for t in range(5)])
    np.testing.assert_array_equal(actual, expected)
    for length in (0, 1, 2, 3):
        assert transfer.transfer_features(values[:length]).shape == (0, 9)
    index = TargetIndex(inputs['training_path'], 'train', window=32, stride=4, phase=0, cutoffs=(3, 4))
    raw = raw_features(index, 3, [1, 2, 3])
    for row, (ri, t) in enumerate(index.items):
        np.testing.assert_array_equal(raw[row], transfer.transfer_features(index.neural(ri))[t])


def test_fit_reproduces_phase_a_selection(inputs):
    from src.probe import Split, select_ridge
    indices = {s: TargetIndex(inputs['training_path'], s, window=32, stride=4,
                              phase=0, cutoffs=(3, 4)) for s in ('train', 'val', 'test')}
    splits = [Split(indices[s], 'cpu') for s in ('train', 'val', 'test')]
    options = {'B1': tuple(raw_features(indices[s], 3, [1, 2, 3]) for s in ('train', 'val', 'test'))}
    expected = select_ridge(options, {'1': list(range(len(indices['train'])))}, *splits,
                            ALPHAS, 'cpu')['1']
    decoder = transfer.fit_decoder(inputs['config'], 'cpu')
    assert decoder.alpha == expected['alpha']
    assert decoder.validation_score == expected['score']
    np.testing.assert_allclose(decoder.predict(options['B1'][1]), expected['prediction_val'], atol=1e-12)
    np.testing.assert_array_equal(decoder.target_mean, splits[0].targets.astype(np.float64).mean(0))
    scale = Standardizer(options['B1'][0])
    np.testing.assert_array_equal(decoder.standardizer.mean, scale.mean)
    np.testing.assert_array_equal(decoder.standardizer.std, scale.std)


def test_transfer_mutation_cannot_change_fit_and_arrays_open_after_fit(inputs, tmp_path, monkeypatch):
    original_fit, original_load = transfer.fit_decoder, np.load
    fitted, completed = [], [False]
    transfer_arrays = {str(transfer._path(r['neural_path'], inputs['transfer_path'])) for r in inputs['runs']}

    def fit(config, device):
        completed[0] = False
        decoder = original_fit(config, device)
        fitted.append(decoder)
        completed[0] = True
        return decoder

    def load(path, *args, **kwargs):
        if str(Path(path).resolve()) in transfer_arrays:
            assert completed[0], 'Transfer neural array opened before decoder fit completed'
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(transfer, 'fit_decoder', fit)
    monkeypatch.setattr(np, 'load', load)
    evaluate(inputs, tmp_path / 'first')
    for run in inputs['runs'][:-1]:
        path = transfer._path(run['neural_path'], inputs['transfer_path'])
        np.save(path, original_load(path) * -13 + 100)
    evaluate(inputs, tmp_path / 'second')
    assert len(fitted) == 2 and fitted[0].alpha == fitted[1].alpha
    assert fitted[0].validation_score == fitted[1].validation_score
    torch.testing.assert_close(fitted[0].weights, fitted[1].weights, rtol=0, atol=0)
    np.testing.assert_array_equal(fitted[0].standardizer.mean, fitted[1].standardizer.mean)
    np.testing.assert_array_equal(fitted[0].standardizer.std, fitted[1].standardizer.std)
    np.testing.assert_array_equal(fitted[0].target_mean, fitted[1].target_mean)
    with original_load(tmp_path / 'first' / 'predictions.npz') as a, original_load(tmp_path / 'second' / 'predictions.npz') as b:
        assert not np.array_equal(a['predictions'], b['predictions'])


def test_run_vectors_average_every_eligible_finite_prediction(inputs):
    decoder = transfer.fit_decoder(inputs['config'], 'cpu')
    run = inputs['runs'][0]
    path = transfer._path(run['neural_path'], inputs['transfer_path'])
    values = np.load(path)
    vector, record = transfer.predict_run(decoder, run, inputs['transfer_path'])
    expected = decoder.predict(np.stack([values[t + 1:t + 4].reshape(-1) for t in range(5)]))
    np.testing.assert_array_equal(vector, expected.mean(0).astype(np.float32))
    assert record['n_tr_averaged'] == 5 and record['target_trs_averaged'] == list(range(5))
    values[3, 0] = np.nan
    np.save(path, values)
    vector, record = transfer.predict_run(decoder, run, inputs['transfer_path'])
    expected = decoder.predict(np.stack([values[t + 1:t + 4].reshape(-1) for t in (3, 4)]))
    np.testing.assert_array_equal(vector, expected.mean(0).astype(np.float32))
    assert record['target_trs_averaged'] == [3, 4] and record['n_tr_averaged'] == 2
    assert record['skipped_nonfinite_feature_trs'] == [0, 1, 2]
    np.save(path, np.full_like(values, np.nan))
    vector, record = transfer.predict_run(decoder, run, inputs['transfer_path'])
    assert vector is None and record['n_tr_averaged'] == 0


def test_fixed_candidate_order_variants_clustered_null_and_artifacts(inputs, tmp_path, monkeypatch):
    original = transfer.retrieval_summary
    calls = []

    def summary(predictions, references, runs, candidates):
        calls.append((np.array(predictions), np.array(references), runs, candidates))
        return original(predictions, references, runs, candidates)

    monkeypatch.setattr(transfer, 'retrieval_summary', summary)
    report = evaluate(inputs, tmp_path / 'out')
    mean = transfer.fit_decoder(inputs['config'], 'cpu').target_mean
    assert report['protocol'] == 'ssltang-v2-ridge-transfer' and report['subject'] == 'UTS01'
    assert report['decoder']['n_train'] == len(TargetIndex(inputs['training_path'], 'train'))
    assert len(calls) == 6
    for i, (condition, candidates) in enumerate(transfer.CANDIDATES.items()):
        raw, constant, centered = calls[i * 3:i * 3 + 3]
        for call in (raw, constant, centered):
            assert call[3] == candidates
        expected_reference = np.zeros((len(candidates), 768), np.float32)
        offset = 0 if condition == 'imagined_speech' else 5
        expected_reference[np.arange(len(candidates)), offset + np.arange(len(candidates))] = 1
        np.testing.assert_array_equal(raw[1], expected_reference)
        np.testing.assert_array_equal(constant[0], np.broadcast_to(mean, raw[0].shape))
        np.testing.assert_array_equal(centered[0], raw[0] - mean)
        np.testing.assert_array_equal(centered[1], raw[1] - mean)
        condition_report = report['conditions'][condition]
        assert condition_report['status'] == 'scored'
        for variant in ('raw', 'constant_training_mean', 'training_mean_centered'):
            assert condition_report[variant]['candidate_ids'] == list(candidates)
        assert condition_report['raw']['permutation']['n_permutations'] == (120 if i == 0 else 6)
        assert all(r['n_tr_averaged'] == 5 for r in condition_report['runs'])
    imagined = report['conditions']['imagined_speech']['raw']['per_narrative']
    assert all(r['n_repeats'] == 2 for r in imagined.values())
    assert any(r['run_id'] == 'sintel' for r in report['skipped_runs'])
    provenance = report['provenance']
    assert provenance['training_manifest_sha256'] == transfer.file_sha256(inputs['training_path'])
    assert provenance['transfer_manifest_sha256'] == transfer.file_sha256(inputs['transfer_path'])
    assert provenance['normalization']['fit_during_evaluation'] is False
    assert len(provenance['code_sha256']) == 64
    assert len(inputs['factory_calls']) == 1
    assert inputs['factory_calls'][0]['revision'] == transfer.features.REVISION
    with np.load(tmp_path / 'out' / 'predictions.npz', allow_pickle=False) as saved:
        assert saved['predictions'].shape == (13, 768) and saved['predictions'].dtype == np.float16
        assert np.all(saved['n_tr_averaged'] == 5)
    evaluate(inputs, tmp_path / 'out')
    assert len(inputs['factory_calls']) == 1  # Existing content-addressed references reused.


def test_missing_candidate_never_shrinks_pool_or_opens_condition_arrays(inputs, tmp_path, monkeypatch):
    path = inputs['transfer_path']
    document = json.loads(path.read_text())
    document['runs'] = [r for r in document['runs'] if r['story'] != 'alpha']
    path.write_text(json.dumps(document))
    original_load = np.load
    forbidden = {str(transfer._path(r['neural_path'], path)) for r in document['runs']
                 if r['condition'] == 'imagined_speech'}

    def load(path, *args, **kwargs):
        assert str(Path(path).resolve()) not in forbidden
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, 'load', load)
    report = evaluate(inputs, tmp_path / 'out')['conditions']['imagined_speech']
    assert report['status'] == 'unscored_missing_fixed_candidates'
    assert report['candidate_ids'] == list(transfer.CANDIDATES['imagined_speech'])
    assert report['top1_chance'] == .2 and report['raw'] is None


@pytest.mark.parametrize('change', ['feature_order', 'subject', 'normalization'])
def test_provenance_mismatch_rejected_before_fit(inputs, tmp_path, monkeypatch, change):
    path = inputs['transfer_path']
    document = json.loads(path.read_text())
    if change == 'feature_order':
        document['feature_order'] = 'different columns'
    elif change == 'subject':
        document['runs'][0]['subject'] = 'UTS02'
    else:
        with np.load(document['normalization_path']) as state:
            arrays = {k: state[k] for k in state.files}
        arrays['mean'] = arrays['mean'] + 1
        changed = tmp_path / 'different_normalization.npz'
        np.savez(changed, **arrays)
        document['normalization_path'] = str(changed)
    path.write_text(json.dumps(document))

    def forbidden_fit(*args, **kwargs):
        pytest.fail('Decoder fit should not start with invalid provenance')

    monkeypatch.setattr(transfer, 'fit_decoder', forbidden_fit)
    with pytest.raises(ValueError):
        evaluate(inputs, tmp_path / 'out')
