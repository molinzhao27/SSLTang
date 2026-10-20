"""CPU checks for v2 target-first indexing, ridge readouts and endpoints."""
import json

import numpy as np
import pytest
import torch

from src.evaluation.endpoints import score, shift_null
from src.models import FmriEncoder
from src.probe import Split, select_ridge
from src.readout import RidgePath, encoder_states, raw_features, window_matrix
from src.targets import TargetIndex, TargetWindows
from src.windows import SemanticWindows


def manifest(root, n_tr=(41, 38, 45, 40, 44, 39), splits=('train', 'train', 'val', 'val', 'test', 'test'),
             n_features=5, target_dim=3):
    rng = np.random.default_rng(4)
    runs = []
    for index, (length, split) in enumerate(zip(n_tr, splits)):
        run_id = f'run{index}'
        np.save(root/f'{run_id}.npy', rng.normal(size=(length, n_features)).astype(np.float32))
        np.save(root/f'{run_id}.targets.npy', rng.normal(size=(length, target_dim)).astype(np.float32))
        valid = np.ones(length, dtype=bool)
        valid[length // 2] = False
        np.save(root/f'{run_id}.valid.npy', valid)
        runs.append(dict(run_id=run_id, dataset='ds003020', subject='UTS01', story=run_id,
                         session=f's{index // 2}', split=split, n_tr=length, n_features=n_features,
                         neural_path=str(root/f'{run_id}.npy'), target_path=str(root/f'{run_id}.targets.npy'),
                         target_valid_path=str(root/f'{run_id}.valid.npy')))
    path = root/'manifest.json'
    path.write_text(json.dumps(dict(runs=runs)))
    return path


def test_targets_are_common_across_cutoffs_and_match_archived_clock(tmp_path):
    path = manifest(tmp_path)
    index = TargetIndex(path, 'train', window=8, stride=4, phase=0, cutoffs=(3, 4))
    for (ri, t), name in zip(index.items, index.ids):
        n_tr = index.runs[ri]['n_tr']
        assert t % 4 == 0 and t >= 8 - 1 - 3 and t + 4 <= n_tr - 1 and name.endswith(f':{t}')
    # Every common target is an archived lag-3 target; archived extras lack a cutoff-4 window.
    archived = set(SemanticWindows(path, 'train', 8, 4, lag_tr=3).label_metadata['label_ids'])
    assert set(index.ids) <= archived
    for extra in archived - set(index.ids):
        run, t = extra.rsplit(':', 1)
        n_tr = next(r['n_tr'] for r in index.runs if r['run_id'] == run)
        assert int(t) + 4 > n_tr - 1
    for cutoff in (3, 4):
        windows = TargetWindows(index, cutoff)
        assert [f"{index.runs[ri]['run_id']}:{end - cutoff}" for ri, end in windows.indices] == index.ids
        item = windows[2]
        ri, t = index.items[2]
        np.testing.assert_array_equal(item['x'].numpy(), index.neural(ri)[t + cutoff - 7:t + cutoff + 1])
        assert item['target_tr'] == t and item['end_tr'] == t + cutoff
    with pytest.raises(ValueError):
        TargetWindows(index, 2)


@pytest.mark.parametrize('allocation,block', [('block', 8), ('story', 4)])
def test_label_subsets_nested_and_deterministic(tmp_path, allocation, block):
    index = TargetIndex(manifest(tmp_path), 'train', window=8, stride=2, cutoffs=(3, 4))
    previous = None
    for fraction in (0.1, 0.5, 1.0):
        rows, meta = index.label_subset(fraction, 7, block, allocation)
        assert rows == index.label_subset(fraction, 7, block, allocation)[0]
        assert meta['selected_examples'] == len(rows) == max(1, int(np.ceil(len(index) * fraction)))
        if previous is not None:
            assert set(previous) <= set(rows)
        previous = rows
    with pytest.raises(ValueError):
        TargetIndex(manifest(tmp_path), 'val', window=8, stride=2).label_subset(0.5)


def test_story_allocation_covers_more_stories(tmp_path):
    index = TargetIndex(manifest(tmp_path), 'train', window=8, stride=1, cutoffs=(3,))
    _, block = index.label_subset(0.2, 3, 64, 'block')
    _, story = index.label_subset(0.2, 3, 4, 'story')
    assert story['labeled_stories'] >= block['labeled_stories'] and story['labeled_stories'] == 2


def test_raw_features_and_windows_respect_cutoff(tmp_path):
    index = TargetIndex(manifest(tmp_path), 'val', window=8, stride=4, cutoffs=(3, 4))
    features = raw_features(index, 4, [1, 4])
    windows = window_matrix(index, 4).reshape(len(index), 8, -1)
    np.testing.assert_array_equal(features[:, :5], windows[:, 4])
    np.testing.assert_array_equal(features[:, 5:], windows[:, 7])
    with pytest.raises(ValueError):
        raw_features(index, 3, [4])


def test_encoder_states_match_forward():
    torch.manual_seed(0)
    encoder = FmriEncoder(dict(n_features=5, d_model=8, n_layers=3, n_heads=2, max_length=16)).eval()
    x = torch.randn(4, 10, 5)
    states = encoder_states(encoder, x)
    assert states.shape == (3, 4, 10, 8)
    torch.testing.assert_close(states[-1], encoder(x))


def test_ridge_path_matches_closed_form():
    rng = np.random.default_rng(1)
    x, y, z = rng.normal(size=(30, 6)), rng.normal(size=(30, 2)), rng.normal(size=(7, 6))
    ridge = RidgePath(x, y, 'cpu')
    for alpha in (0.1, 10.0):
        xc, yc = x - x.mean(0), y - y.mean(0)
        beta = np.linalg.solve(xc.T @ xc + alpha * np.eye(6), xc.T @ yc)
        expected = (z - x.mean(0)) @ beta + y.mean(0)
        np.testing.assert_allclose(ridge.predict(ridge.project(z), alpha).numpy(), expected, atol=1e-10)


def test_selection_ignores_test_targets(tmp_path):
    path = manifest(tmp_path)
    splits = {s: Split(TargetIndex(path, s, window=8, stride=1, cutoffs=(3,)), 'cpu') for s in ('train', 'val', 'test')}
    feature = lambda s: raw_features(splits[s].index, 3, [1, 2, 3])
    options = {'a': tuple(feature(s) for s in ('train', 'val', 'test')),
               'b': tuple(feature(s)[:, :5] for s in ('train', 'val', 'test'))}
    budgets = {'1': list(range(len(splits['train'].targets)))}
    first = select_ridge(options, budgets, splits['train'], splits['val'], splits['test'], device='cpu')['1']
    splits['test'].targets = -splits['test'].targets
    second = select_ridge(options, budgets, splits['train'], splits['val'], splits['test'], device='cpu')['1']
    assert (first['option'], first['alpha']) == (second['option'], second['alpha'])
    np.testing.assert_array_equal(first['prediction_test'], second['prediction_test'])


def test_endpoints_session_macro_and_shift_null():
    rng = np.random.default_rng(3)
    runs = np.repeat(['a', 'b', 'c'], 40)
    times = np.tile(np.arange(40) * 4, 3)
    target = np.cumsum(rng.normal(size=(120, 4)), axis=0)
    sessions = {'a': 1, 'b': 1, 'c': 2}
    report = score(target, target, runs, times, sessions, target.mean(0), target.std(0))
    assert report['session_macro_feature_correlation'] == pytest.approx(1.0)
    noisy = target + rng.normal(scale=3, size=target.shape)
    report = score(noisy, target, runs, times, sessions, target.mean(0))
    by_run = report['by_run_feature_correlation']
    expected = np.mean([(by_run['a'] + by_run['b']) / 2, by_run['c']])
    assert report['session_macro_feature_correlation'] == pytest.approx(expected)
    null = shift_null(noisy, target, runs, sessions, stride=4, min_shift_tr=40, draws=200)
    assert null['runs_used'] == 3 and null['observed_session_macro'] == pytest.approx(expected)
    assert 0 < null['p_value'] <= 1
    short = shift_null(noisy[:12], target[:12], runs[:12], sessions, stride=4, min_shift_tr=40, draws=10)
    assert short['runs_used'] == 0 and 'a' in short['omitted_runs']


def v2_config(path, output, cutoff):
    return dict(data=dict(manifest=str(path), window=8, stride=2, num_workers=0),
                model=dict(d_model=8, n_layers=1, n_heads=2, max_length=8, dropout=0.0),
                training=dict(seed=17, epochs=2, batch_size=4, device='cpu', precision='fp32',
                              checkpoint_every_steps=100, lr=.001),
                semantic=dict(protocol='ssltang-v2', common_cutoffs=[3, 4], target_phase=0, lag_tr=cutoff,
                              label_fraction=.5, subset_seed=23, block_tr=4, init='random'),
                output_dir=str(output))


def test_v2_finetuning_labels_do_not_depend_on_cutoff(tmp_path):
    from src import training
    from src.evaluation.finetuned import evaluate_checkpoint
    path = manifest(tmp_path)
    subsets = {}
    for cutoff in (3, 4):
        out = tmp_path/f'c{cutoff}'
        training.run_training(v2_config(path, out, cutoff), 'semantic')
        labels = json.loads((out/'label_subset.json').read_text())
        subsets[cutoff] = labels['label_metadata']['label_ids']
        assert all(end - target == cutoff for (_, end), target in
                   zip(labels['indices'], [int(i.rsplit(':', 1)[1]) for i in subsets[cutoff]]))
        report = evaluate_checkpoint(out/'best.pt', device='cpu', num_workers=0, shift_min_tr=4, shift_draws=20)
        index = TargetIndex(path, 'test', window=8, stride=2, cutoffs=(3, 4))
        assert report['test']['n_samples'] == len(index) and report['cutoff_tr'] == cutoff
        assert report['test_targets'] == index.signature()
    assert subsets[3] == subsets[4]
    expected = TargetIndex(path, 'train', window=8, stride=2, cutoffs=(3, 4)).label_subset(.5, 23, 4)[1]
    assert subsets[3] == expected['label_ids']


def ssl_config(path, output, objective):
    return dict(data=dict(manifest=str(path), window=8, stride=2, num_workers=0),
                model=dict(d_model=8, n_layers=1, n_heads=2, max_length=8, dropout=.2),
                training=dict(seed=17, epochs=3, max_steps=6, batch_size=3, grad_accumulation=1,
                              device='cpu', precision='fp32', checkpoint_every_steps=1, lr=.001),
                ssl=dict(mode='temporal', target_fraction=.25, span_length=2, guard_length=1,
                         objective=objective, mask_rng='dedicated', diagnostics=True),
                output_dir=str(output))


def test_latent_targets_hide_masked_values_from_student():
    from src.ssl.latent import LatentPredictionModel
    from src.ssl import MaskConfig, generate_mask
    torch.manual_seed(0)
    encoder = FmriEncoder(dict(n_features=5, d_model=8, n_layers=2, n_heads=2, max_length=12, dropout=0.0))
    model = LatentPredictionModel(encoder, MaskConfig(span_length=3, guard_length=1)).train()
    assert not model.teacher.training and not any(p.requires_grad for p in model.teacher.parameters())
    x = torch.randn(3, 12, 5)
    mask = generate_mask(x.shape, model.mask_config, generator=torch.Generator().manual_seed(1))
    changed = x.clone()
    changed[~mask.observation_mask] = 100.0
    first, second = model(x, mask), model(changed, mask)
    torch.testing.assert_close(first['latent'], second['latent'])   # Student input unchanged.
    assert not torch.allclose(first['loss'], second['loss'])       # Only the teacher target sees it.
    first['loss'].backward()
    assert all(p.grad is None for p in model.teacher.parameters())
    assert encoder.value_embedding.weight.grad is not None
    before = [p.detach().clone() for p in model.teacher.parameters()]
    with torch.no_grad():
        for p in encoder.parameters():
            p.add_(1.0)
    model.update_teacher(0.75)
    for old, new, student in zip(before, model.teacher.parameters(), encoder.parameters()):
        torch.testing.assert_close(new, 0.75 * old + 0.25 * student)
    assert model.momentum(1, 10) == pytest.approx(0.996) and model.momentum(11, 10) == pytest.approx(1.0)


def test_objectives_share_masks_and_latent_resume_is_exact(tmp_path, monkeypatch):
    import copy, os, signal
    from pathlib import Path
    from src import training
    from src.ssl import generate_mask
    path = manifest(tmp_path)
    seen = {}
    original = training.generate_mask

    def recording(shape, config, *, device=None, generator=None):
        mask = original(shape, config, device=device, generator=generator)
        seen.setdefault(current[0], []).append(mask.target_mask.clone())
        return mask

    monkeypatch.setattr(training, 'generate_mask', recording)
    current = [None]
    for objective in ('reconstruction', 'latent'):
        current[0] = objective
        training.run_training(ssl_config(path, tmp_path/objective, objective), 'ssl')
    assert len(seen['latent']) == len(seen['reconstruction'])
    for a, b in zip(seen['latent'], seen['reconstruction']):
        assert torch.equal(a, b)
    monkeypatch.setattr(training, 'generate_mask', original)
    logs = [json.loads(line) for line in (tmp_path/'latent'/'metrics.jsonl').read_text().splitlines()]
    assert {'student', 'teacher'} <= set(logs[-1]['diagnostics'])
    export = torch.load(tmp_path/'latent'/'encoder_latest.pt', weights_only=False)
    assert 'teacher_state_dict' in export and export['global_step'] == 6
    # Interrupt after the second update and resume: identical final state.
    expected = torch.load(tmp_path/'latent'/'latest.pt', weights_only=False)
    cfg = ssl_config(path, tmp_path/'interrupted', 'latent')
    save = training.atomic_torch_save

    def stop_after_save(state, destination):
        save(state, destination)
        if Path(destination).name == 'latest.pt' and state['global_step'] == 2:
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(training, 'atomic_torch_save', stop_after_save)
    training.run_training(cfg, 'ssl')
    monkeypatch.setattr(training, 'atomic_torch_save', save)
    training.run_training(cfg, 'ssl', resume='latest')
    actual = torch.load(tmp_path/'interrupted'/'latest.pt', weights_only=False)
    for key in ('model_state_dict', 'optimizer_state_dict', 'mask_generator_state', 'global_step'):
        a, b = expected[key], actual[key]
        if isinstance(a, dict):
            for k in a:
                if isinstance(a[k], torch.Tensor):
                    torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
        elif isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        else:
            assert a == b


def test_grouped_cv_alpha_uses_only_given_rows():
    from src.probe import grouped_cv_alpha
    rng = np.random.default_rng(5)
    x = rng.normal(size=(60, 8))
    y = x[:, :3] + 0.5 * rng.normal(size=(60, 3))
    groups = np.repeat(['s1', 's2', 's3', 's4'], 15)
    alpha, curve, folds = grouped_cv_alpha(x, y, groups, alphas=(0.1, 1.0, 1e6), device='cpu')
    assert folds == 4 and len(curve) == 3 and alpha in (0.1, 1.0)
    with pytest.raises(ValueError):
        grouped_cv_alpha(x, y, np.repeat('one', 60), device='cpu')


def test_selection_and_reporting_share_varying_column_rule():
    from src.evaluation.metrics import _feature_correlation
    from src.readout import run_correlations
    z = np.random.default_rng(6).normal(size=40)
    prediction = np.column_stack([100 + 1e-8 * z, -z])
    target = np.column_stack([z, z])
    reported = _feature_correlation(prediction, target)[0]
    selected = run_correlations(torch.as_tensor(prediction), torch.as_tensor(target),
                                torch.zeros(40, dtype=torch.long), 1)[0].item()
    assert reported == pytest.approx(-1.0) and selected == pytest.approx(reported)


def test_paired_contrasts_require_matching_seeds():
    import sys
    from pathlib import Path
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from summarize_v2 import paired
    sessions = {'s1': 1, 's2': 2}
    row = lambda arm, seed, value: dict(arm=arm, seed=seed, fraction=1.0, by_run={'s1': value, 's2': value})
    df = pd.DataFrame([row('A', 17, 0.2), row('B', 17, 0.1), row('B', 23, 0.9), row('C', 0, 0.0)])
    with pytest.raises(ValueError):
        paired(df, dict(arm='A'), dict(arm='B'), sessions)
    result = paired(df, dict(arm='B'), dict(arm='C'), sessions)  # Deterministic C pairs with every B seed.
    assert result['n_pairs'] == 2 and result['session_macro_delta'] == pytest.approx(0.5)
