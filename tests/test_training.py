"""Small CPU regressions for scientific fitting and checkpoint behavior."""
import copy
import json
import os
from pathlib import Path
import signal

import numpy as np
import pytest
import torch

from src import training
from src.ssl import MaskedNeuralModel
from src.windows import NeuralWindows, SemanticWindows


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def manifest(root, n_tr=16):
    runs = []
    rng = np.random.default_rng(2)
    for index, split in enumerate(('train', 'train', 'val')):
        run_id = f'run{index}'
        x = rng.normal(size=(n_tr, 4)).astype(np.float32)
        y = np.column_stack((np.arange(n_tr), np.arange(n_tr)**2)).astype(np.float32)
        if split == 'val':
            y += 10000  # Validation must not influence target standardization.
        np.save(root/f'{run_id}.npy', x)
        np.save(root/f'{run_id}.targets.npy', y)
        np.save(root/f'{run_id}.valid.npy', np.ones(n_tr, dtype=bool))
        runs.append(dict(run_id=run_id, dataset='ds003020', subject='UTS01',
                         story=run_id, session=f'session{index}', split=split,
                         n_tr=n_tr, n_features=4, neural_path=str(root/f'{run_id}.npy'),
                         target_path=str(root/f'{run_id}.targets.npy'),
                         target_valid_path=str(root/f'{run_id}.valid.npy')))
    path = root/'manifest.json'
    path.write_text(json.dumps(dict(runs=runs)))
    return path


def config(path, output):
    return dict(data=dict(manifest=str(path), window=4, stride=2, num_workers=0),
                model=dict(d_model=8, n_layers=1, n_heads=2, max_length=4, dropout=.2),
                training=dict(seed=17, epochs=3, max_steps=6, batch_size=3,
                              grad_accumulation=2, device='cpu', precision='fp32',
                              checkpoint_every_steps=1, lr=.001, max_grad_norm=100),
                ssl=dict(mode='joint', target_fraction=.3, span_length=1,
                         guard_length=1, feature_group_size=3),
                semantic=dict(lag_tr=1, label_fraction=.5, subset_seed=23, block_tr=4),
                output_dir=str(output))


def assert_same(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_same(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_same(a, b)
    else:
        assert first == second


@pytest.mark.parametrize('task,stop_step,workers', [('ssl', 1, 0), ('ssl', 6, 0),
                                                  ('semantic', 1, 0), ('ssl', 1, 2)])
def test_resume_matches_uninterrupted(tmp_path, monkeypatch, task, stop_step, workers):
    path = manifest(tmp_path)
    baseline = config(path, tmp_path/'baseline')
    baseline['data']['num_workers'] = workers
    training.run_training(baseline, task)
    expected = torch.load(tmp_path/'baseline/latest.pt', weights_only=False)
    interrupted = copy.deepcopy(baseline)
    interrupted['output_dir'] = str(tmp_path/'interrupted')
    save = training.atomic_torch_save

    def stop_after_save(state, destination):
        save(state, destination)
        if Path(destination).name == 'latest.pt' and state['global_step'] == stop_step:
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(training, 'atomic_torch_save', stop_after_save)
    training.run_training(interrupted, task)
    monkeypatch.setattr(training, 'atomic_torch_save', save)
    training.run_training(interrupted, task, resume='latest')
    actual = torch.load(tmp_path/'interrupted/latest.pt', weights_only=False)
    for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
                'scaler_state_dict', 'rng_state', 'loader_generator_state',
                'epoch', 'next_batch', 'global_step', 'best_val_loss', 'target_mean', 'target_std'):
        assert_same(expected[key], actual[key])
    assert (tmp_path/'interrupted/best.pt').exists()


def test_accumulation_weights_uneven_batches(tmp_path):
    path = manifest(tmp_path, n_tr=7)  # Four train windows: microbatches of 3 and 1.
    large = config(path, tmp_path/'large')
    large['model']['dropout'] = 0
    large['training'].update(epochs=1, max_steps=1, batch_size=4, grad_accumulation=1)
    small = copy.deepcopy(large)
    small['output_dir'] = str(tmp_path/'small')
    small['training'].update(batch_size=3, grad_accumulation=2)
    training.run_training(large, 'ssl')
    training.run_training(small, 'ssl')
    a = torch.load(tmp_path/'large/latest.pt', weights_only=False)
    b = torch.load(tmp_path/'small/latest.pt', weights_only=False)
    # Compare Adam's first moments: updates alone can hide magnitude errors.
    for key, state in a['optimizer_state_dict']['state'].items():
        torch.testing.assert_close(state['exp_avg'], b['optimizer_state_dict']['state'][key]['exp_avg'],
                                   rtol=2e-5, atol=1e-8)


def test_validation_weights_masked_elements(tmp_path):
    path = manifest(tmp_path)
    cfg = config(path, tmp_path/'run')
    cfg['training'].update(epochs=1, max_steps=1)
    training.run_training(cfg, 'ssl')
    state = torch.load(tmp_path/'run/latest.pt', weights_only=False)
    model = MaskedNeuralModel(training.FmriEncoder(state['encoder_config']), cfg['ssl']).eval()
    model.load_state_dict(state['model_state_dict'])
    val = NeuralWindows(path, 'val', window=4, stride=2)
    loader, _ = training._loader(val, cfg['training'], 0, False)
    total = count = 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            mask = training.generate_mask(batch['x'].shape, model.mask_config,
                                          generator=torch.Generator().manual_seed(100017+index))
            result = model(batch['x'], mask)
            difference = result['prediction'][mask.target_mask] - batch['x'][mask.target_mask]
            total += difference.double().square().sum().item()
            count += int(mask.target_mask.sum())
    metric = json.loads((tmp_path/'run/metrics.jsonl').read_text().splitlines()[0])
    assert metric['val_loss'] == pytest.approx(total/count, rel=1e-6)


def test_labels_nested_aligned_and_train_statistics_only(tmp_path):
    path = manifest(tmp_path)
    options = dict(window=4, stride=1, lag_tr=2, subset_seed=9, block_tr=4)
    small = SemanticWindows(path, 'train', label_fraction=.25, **options)
    medium = SemanticWindows(path, 'train', label_fraction=.5, **options)
    full = SemanticWindows(path, 'train', **options)
    assert set(small.indices) <= set(medium.indices) <= set(full.indices)
    assert small.indices == SemanticWindows(path, 'train', label_fraction=.25, **options).indices
    val = SemanticWindows(path, 'val', label_fraction=.01, **options)
    assert len(val) == len(SemanticWindows(path, 'val', **options))
    for index in range(len(small)):
        example = small[index]
        ri, end = small.indices[index]
        assert example['target_tr'] == end-2
        np.testing.assert_array_equal(example['x'], np.load(small.runs[ri]['neural_path'])[end-3:end+1])
        np.testing.assert_array_equal(example['y'], np.load(small.runs[ri]['target_path'])[end-2])
    mean, std = training._target_statistics(small)
    targets = torch.stack([small[i]['y'] for i in range(len(small))])
    torch.testing.assert_close(mean, targets.mean(0))
    torch.testing.assert_close(std, targets.std(0, correction=0).clamp_min(1e-6))
    assert mean.max() < 1000


@pytest.mark.parametrize('identity', ['story', 'run_id', 'session', 'neural_path'])
def test_cross_split_identity_rejected(tmp_path, identity):
    path = manifest(tmp_path)
    document = json.loads(path.read_text())
    document['runs'][2][identity] = document['runs'][0][identity]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='crosses splits'):
        NeuralWindows(path, 'train', window=4)


def test_held_out_normalization_rejected(tmp_path):
    path = manifest(tmp_path)
    document = json.loads(path.read_text())
    norm = tmp_path/'normalization.npz'
    np.savez(norm, fit_run_ids=['run0', 'run2'])
    document['normalization_path'] = str(norm)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='training runs only'):
        NeuralWindows(path, 'train', window=4)


def test_misaligned_targets_rejected(tmp_path):
    path = manifest(tmp_path)
    np.save(tmp_path/'run0.valid.npy', np.ones(15, dtype=bool))
    with pytest.raises(ValueError, match='one boolean per TR'):
        SemanticWindows(path, 'train', window=4)


def test_semantic_heads_identical_across_initializations(tmp_path, monkeypatch):
    path = manifest(tmp_path)
    ssl = config(path, tmp_path/'pretrained')
    ssl['training'].update(epochs=1, max_steps=1)
    training.run_training(ssl, 'ssl')
    heads = []
    regressor = training.SemanticRegressor

    def record_head(*args, **kwargs):
        model = regressor(*args, **kwargs)
        heads.append(copy.deepcopy(model.head.state_dict()))
        return model

    monkeypatch.setattr(training, 'SemanticRegressor', record_head)
    for initialization in ('random', 'ssl'):
        cfg = config(path, tmp_path/initialization)
        cfg['training'].update(epochs=1, max_steps=1)
        cfg['semantic'].update(init=initialization,
                               pretrained_encoder=str(tmp_path/'pretrained/encoder_best.pt'))
        training.run_training(cfg, 'semantic')
    assert_same(heads[0], heads[1])


def test_missing_ssl_provenance_rejected(tmp_path):
    path = manifest(tmp_path)
    cfg = config(path, tmp_path/'run')
    encoder = training.FmriEncoder(dict(cfg['model'], n_features=4))
    pretrained = tmp_path/'unverified.pt'
    torch.save(dict(task='ssl', encoder_config=vars(encoder.config),
                    encoder_state_dict=encoder.state_dict()), pretrained)
    cfg['semantic'].update(init='ssl', pretrained_encoder=str(pretrained))
    with pytest.raises(ValueError, match='provenance'):
        training.run_training(cfg, 'semantic')


def test_overflow_skips_optimizer_schedule_and_budget(tmp_path, monkeypatch):
    class OverflowOnce:
        def __init__(self, *args, **kwargs):
            self.current_scale = 2.0
            self.skip = True

        def is_enabled(self):
            return True

        def get_scale(self):
            return self.current_scale

        def scale(self, loss):
            return loss*self.current_scale

        def unscale_(self, optimizer):
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    if parameter.grad is not None:
                        parameter.grad.div_(self.current_scale)

        def step(self, optimizer):
            if not self.skip:
                optimizer.step()

        def update(self):
            if self.skip:
                self.current_scale /= 2
                self.skip = False

        def state_dict(self):
            return dict(scale=self.current_scale)

    monkeypatch.setattr(torch.amp, 'GradScaler', OverflowOnce)
    path = manifest(tmp_path, n_tr=7)
    cfg = config(path, tmp_path/'run')
    cfg['training'].update(epochs=1, max_steps=1, batch_size=2, grad_accumulation=1)
    training.run_training(cfg, 'ssl')
    state = torch.load(tmp_path/'run/latest.pt', weights_only=False)
    assert state['global_step'] == 1
    assert state['scheduler_state_dict']['last_epoch'] == 1
    assert state['optimizer_state_dict']['state']
    assert all(item['step'] == 1 for item in state['optimizer_state_dict']['state'].values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA smoke test')
@pytest.mark.parametrize('task', ['ssl', 'semantic'])
@pytest.mark.parametrize('tf32', [False, True])
def test_cuda_bf16_training(tmp_path, task, tf32):
    if not torch.cuda.is_bf16_supported():
        pytest.skip('GPU does not support BF16')
    path = manifest(tmp_path)
    cfg = config(path, tmp_path/'run')
    cfg['training'].update(epochs=1, max_steps=1, device='cuda', precision='bf16', tf32=tf32)
    training.run_training(cfg, task)
    state = torch.load(tmp_path/'run/latest.pt', map_location='cpu', weights_only=False)
    assert state['global_step'] == 1
    assert all(torch.isfinite(value).all() for value in state['model_state_dict'].values())
    assert torch.backends.cuda.matmul.allow_tf32 == tf32
    assert torch.backends.cudnn.allow_tf32 == tf32
    record = json.loads((tmp_path/'run/metrics.jsonl').read_text().splitlines()[0])
    assert np.isfinite(record['train_loss']) and np.isfinite(record['val_loss'])
