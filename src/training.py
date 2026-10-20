"""Resumable neural-only SSL and matched-budget semantic training."""
from dataclasses import asdict
from pathlib import Path
import json
import hashlib
import math
import signal
import time
import torch
from torch.utils.data import DataLoader
from .models import FmriEncoder
from .ssl import MaskedNeuralModel, generate_mask
from .ssl.diagnostics import representation_diagnostics
from .ssl.latent import LatentPredictionModel
from .ssl.masking import MaskConfig
from .semantic.model import SemanticRegressor
from .windows import NeuralWindows, SemanticWindows
from .targets import PROTOCOL_ID, TargetIndex, TargetWindows
from .runtime import (seed_everything, atomic_torch_save, save_json,
                      capture_rng, restore_rng, git_commit, file_sha256)


def _loader(dataset, cfg, epoch, shuffle):
    generator = torch.Generator().manual_seed(int(cfg.get('seed', 17)) + epoch)
    return DataLoader(dataset, batch_size=int(cfg.get('batch_size', 128)),
        shuffle=shuffle, generator=generator, num_workers=int(cfg.get('num_workers', 0)),
        pin_memory=cfg.get('device', 'cuda').startswith('cuda'), drop_last=False), generator


def _target_statistics(dataset):
    total = torch.zeros(dataset.target_dim, dtype=torch.float64)
    squares = torch.zeros_like(total)
    count = 0
    for batch in DataLoader(dataset, batch_size=512):
        values = batch['y'].double()
        total += values.sum(0)
        squares += values.square().sum(0)
        count += len(values)
    if not count:
        raise ValueError('No selected training targets')
    mean = total / count
    std = (squares/count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    return mean.float(), std.float()


def _neural_signature(manifest):
    fields = ('dataset', 'subject', 'run_id', 'split', 'neural_path', 'n_features', 'n_tr')
    payload = {'runs': sorted([{k: r.get(k) for k in fields} for r in manifest['runs']],
                             key=lambda r: str(r['run_id'])),
               'transforms': {k: v for k, v in manifest.items()
                              if any(word in k.lower() for word in ('reduction', 'transform', 'pca', 'normalization'))}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_training(config, task, resume=None):
    if task not in ('ssl', 'semantic'):
        raise ValueError('Unknown task')
    data = config['data']
    train_cfg = dict(config['training'])
    train_cfg['num_workers'] = data.get('num_workers', 8)
    seed = int(train_cfg.get('seed', 17))
    seed_everything(seed)
    device = torch.device(train_cfg.get('device', 'cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; set training.device=cpu for a smoke test')
    torch.set_float32_matmul_precision('high' if train_cfg.get('tf32', True) else 'highest')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = bool(train_cfg.get('tf32', True))
        torch.backends.cudnn.allow_tf32 = bool(train_cfg.get('tf32', True))
    precision = train_cfg.get('precision', 'bf16')
    if precision not in ('bf16', 'fp16', 'fp32'):
        raise ValueError('precision must be bf16, fp16, or fp32')
    if device.type == 'cuda' and precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 is unsupported on this GPU')
    out = Path(config['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    kwargs = dict(manifest_path=data['manifest'], window=data.get('window', 32), stride=data.get('stride', 4))
    semantic_cfg = config.get('semantic', {})
    if task == 'ssl':
        train = NeuralWindows(split='train', **kwargs)
        val = NeuralWindows(split='val', **kwargs)
    elif semantic_cfg.get('protocol') == PROTOCOL_ID:
        # Target-first v2 indexing: lag_tr is the response cutoff; targets are
        # common to every cutoff in common_cutoffs, so lag never changes labels.
        index = dict(window=kwargs['window'], stride=kwargs['stride'],
                     phase=semantic_cfg.get('target_phase', 0),
                     cutoffs=semantic_cfg.get('common_cutoffs', [3, 4]))
        cutoff = semantic_cfg.get('lag_tr', 3)
        train = TargetWindows(TargetIndex(data['manifest'], 'train', **index), cutoff,
                              label_fraction=semantic_cfg.get('label_fraction', 1),
                              subset_seed=semantic_cfg.get('subset_seed', 2718),
                              block_tr=semantic_cfg.get('block_tr', 64),
                              allocation=semantic_cfg.get('allocation', 'block'))
        val = TargetWindows(TargetIndex(data['manifest'], 'val', **index), cutoff)
    else:
        options = dict(label_fraction=semantic_cfg.get('label_fraction', 1),
                       subset_seed=semantic_cfg.get('subset_seed', 2718), lag_tr=semantic_cfg.get('lag_tr', 3),
                       block_tr=semantic_cfg.get('block_tr', 64))
        train = SemanticWindows(split='train', **kwargs, **options)
        val = SemanticWindows(split='val', **kwargs, **dict(options, label_fraction=1))
    # Refuse the held-out dataset anywhere in the fitting manifest.
    if any(r.get('dataset') != 'ds003020' for r in train.manifest['runs']):
        raise ValueError('Fitting manifests must contain only ds003020 runs')
    neural_signature = _neural_signature(train.manifest)
    if not len(train) or not len(val):
        raise ValueError('Training and validation windows must be nonempty')
    encoder_cfg = dict(config.get('model', {}))
    encoder_cfg['n_features'] = train.n_features
    encoder = FmriEncoder(encoder_cfg)
    resume_path = out/'latest.pt' if resume == 'latest' else Path(resume) if resume else None
    checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False) if resume_path else None
    if checkpoint and checkpoint.get('task') != task:
        raise ValueError('Resume task differs from checkpoint')
    if task == 'semantic':
        if semantic_cfg.get('loss', 'mse') != 'mse':
            raise ValueError('Only continuous embedding MSE is supported')
        if semantic_cfg.get('init', 'random') not in ('random', 'ssl'):
            raise ValueError('semantic.init must be random or ssl')
        if not checkpoint and semantic_cfg.get('init', 'random') == 'ssl':
            pretrained_path = semantic_cfg['pretrained_encoder']
            pretrained = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            signature_matches = pretrained.get('neural_signature') == neural_signature
            source_hash = train.manifest.get('neural_manifest_hash')
            source_matches = bool(source_hash) and source_hash == pretrained.get('manifest_sha256')
            if pretrained.get('task') != 'ssl' or not (signature_matches or source_matches):
                raise ValueError('SSL provenance must match ds003020 neural runs and spatial transformation')
            if pretrained['encoder_config'] != asdict(encoder.config):
                raise ValueError('Pretrained encoder architecture differs from random condition')
            encoder.load_state_dict(pretrained['encoder_state_dict'])
        seed_everything(seed)  # Identical semantic head initialization for both conditions.
        model = SemanticRegressor(encoder, train.target_dim, semantic_cfg.get('pooling', 'end'))
        target_mean, target_std = (_target_statistics(train) if checkpoint is None else
                                  (checkpoint['target_mean'], checkpoint['target_std']))
    else:
        ssl_cfg = dict(config.get('ssl', {}))
        objective = ssl_cfg.pop('objective', 'reconstruction')
        ema = (ssl_cfg.pop('ema_start', 0.996), ssl_cfg.pop('ema_end', 1.0))
        ssl_cfg.pop('mask_rng', None)
        ssl_cfg.pop('diagnostics', None)
        unknown = set(ssl_cfg) - set(MaskConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f'Unknown ssl options: {sorted(unknown)}')
        if objective == 'reconstruction':
            model = MaskedNeuralModel(encoder, ssl_cfg)
        elif objective == 'latent':
            model = LatentPredictionModel(encoder, ssl_cfg, *ema)
        else:
            raise ValueError('ssl.objective must be reconstruction or latent')
        target_mean = target_std = None
    seed_everything(seed)  # Initialization does not change optimization RNG budget.
    model.to(device)
    if target_mean is not None:
        target_mean, target_std = target_mean.to(device), target_std.to(device)
    epochs = int(train_cfg.get('epochs', 30))
    accum = int(train_cfg.get('grad_accumulation', 1))
    if accum < 1:
        raise ValueError('grad_accumulation must be positive')
    batch_size = int(train_cfg.get('batch_size', 128))
    checkpoint_every = int(train_cfg.get('checkpoint_every_steps', 200))
    if batch_size < 1 or checkpoint_every < 1:
        raise ValueError('batch_size and checkpoint_every_steps must be positive')
    steps_per_epoch = math.ceil(math.ceil(len(train)/batch_size)/accum)
    max_steps = int(train_cfg.get('max_steps', epochs*steps_per_epoch))
    # Keep backward losses near the batch-mean scale, then normalize the
    # accumulated gradients by the actual number of supervised elements.
    latent = isinstance(model, LatentPredictionModel)
    unit_width = encoder.config.d_model if latent else train.n_features
    reference_units = batch_size * (train.window * unit_width if task == 'ssl' else train.target_dim)
    # The EMA teacher is updated outside the optimizer and never receives gradients.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=train_cfg.get('lr', .0003),
                                  weight_decay=train_cfg.get('weight_decay', .01))
    # A dedicated mask stream makes masks identical across SSL objectives,
    # whose other RNG use (for example teacher forward passes) may differ.
    mask_generator = None
    if task == 'ssl' and config.get('ssl', {}).get('mask_rng', 'global') == 'dedicated':
        mask_generator = torch.Generator().manual_seed(seed + 200000)
    elif task == 'ssl' and config.get('ssl', {}).get('mask_rng', 'global') != 'global':
        raise ValueError('ssl.mask_rng must be global or dedicated')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, max_steps))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and precision == 'fp16')
    epoch = next_batch = global_step = 0
    best = float('inf')
    manifest_hash = file_sha256(data['manifest'])
    if checkpoint:
        if checkpoint['manifest_sha256'] != manifest_hash:
            raise ValueError('Resume manifest differs from checkpoint')
        if checkpoint['config'] != config:
            raise ValueError('Resume configuration differs from checkpoint')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        epoch, next_batch, global_step = checkpoint['epoch'], checkpoint['next_batch'], checkpoint['global_step']
        batches = math.ceil(len(train)/batch_size)
        if not 0 <= next_batch <= batches or (next_batch != batches and next_batch % accum):
            raise ValueError('Resume cursor must be at an optimizer boundary')
        best = checkpoint['best_val_loss']
        restore_rng(checkpoint['rng_state'])
        if mask_generator is not None:
            mask_generator.set_state(checkpoint['mask_generator_state'])
    executable = torch.compile(model) if train_cfg.get('compile', False) else model
    stopping = [False]
    old_handler = signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    loader_generator = None
    diagnostic_windows = None
    if task == 'ssl' and config.get('ssl', {}).get('diagnostics', False):
        count = min(512, len(val))
        diagnostic_windows = torch.stack([val[i]['x'] for i in range(count)]).to(device)
    start_time = time.monotonic()
    save_json(dict(config=config, manifest_sha256=manifest_hash, git_commit=git_commit(), task=task,
                   train_windows=len(train), val_windows=len(val)), out/'run_metadata.json')
    if task == 'semantic':
        save_json(dict(indices=[list(map(int, pair)) for pair in train.indices], label_metadata=train.label_metadata), out/'label_subset.json')

    def save(name, current_epoch, batch_cursor):
        state = dict(model_state_dict=model.state_dict(), encoder_config=asdict(encoder.config),
            encoder_state_dict=encoder.state_dict(), optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(), scaler_state_dict=scaler.state_dict(),
            rng_state=capture_rng(), loader_generator_state=loader_generator.get_state() if loader_generator else None,
            epoch=current_epoch, next_batch=batch_cursor, global_step=global_step, best_val_loss=best,
            config=config, manifest_sha256=manifest_hash, seed=seed, git_commit=git_commit(),
            target_mean=target_mean.detach().cpu() if target_mean is not None else None,
            target_std=target_std.detach().cpu() if target_std is not None else None, task=task,
            neural_signature=neural_signature,
            mask_generator_state=mask_generator.get_state() if mask_generator is not None else None)
        atomic_torch_save(state, out/f'{name}.pt')
        if task == 'ssl':
            export = {k: state[k] for k in ('encoder_config', 'encoder_state_dict',
                'manifest_sha256', 'seed', 'git_commit', 'global_step', 'config', 'task', 'neural_signature')}
            if latent:
                export['teacher_state_dict'] = model.teacher.state_dict()
            atomic_torch_save(export, out/f'encoder_{name}.pt')

    def autocast():
        return torch.autocast(device.type, dtype=torch.bfloat16 if precision == 'bf16' else torch.float16,
                              enabled=precision != 'fp32' and device.type == 'cuda')

    def loss_for(batch, validation=False, index=0):
        x = batch['x'].to(device, non_blocking=True)
        if task == 'ssl':
            # Fixed validation masks independent of optimization RNG and epoch.
            if validation:
                mask = generate_mask(x.shape, model.mask_config, device=device,
                    generator=torch.Generator().manual_seed(seed+100000+index))
            elif mask_generator is not None:
                mask = generate_mask(x.shape, model.mask_config, device=device, generator=mask_generator)
            else:
                mask = None
            result = executable(x, mask)
            units = result['n_units'] if 'n_units' in result else result['target_mask'].sum()
            return result['loss'], int(units.item()), len(x)
        y = batch['y'].to(device, non_blocking=True)
        prediction = executable(x)
        loss = ((prediction.float() - (y.float()-target_mean)/target_std).square()).mean()
        return loss, y.numel(), len(x)

    try:
        while epoch < epochs and (global_step < max_steps or next_batch > 0):
            epoch_start_time = time.monotonic()
            loader, loader_generator = _loader(train, train_cfg, epoch, True)
            # Iterator construction consumes torch RNG for worker seeds. Isolate it
            # with a dedicated loader generator, including when skipping batches.
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = total_units = total_examples = group_units = 0
            last_cursor = next_batch
            for batch_index, batch in enumerate(loader):
                if batch_index < next_batch:
                    continue
                if global_step >= max_steps:
                    break
                with autocast():
                    loss, units, count = loss_for(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                scaler.scale(loss*(units/reference_units)).backward()
                group_units += units
                epoch_loss += loss.detach().item()*units
                total_units += units
                total_examples += count
                last_cursor = batch_index+1
                if last_cursor % accum == 0 or last_cursor == len(loader):
                    scaler.unscale_(optimizer)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(reference_units/group_units)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get('max_grad_norm', 1.0),
                                                  error_if_nonfinite=not scaler.is_enabled())
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    group_units = 0
                    # FP16 overflow skips the optimizer update; it must also
                    # skip the schedule and matched optimizer-step budget.
                    if scaler.get_scale() >= previous_scale:
                        scheduler.step()
                        global_step += 1
                        if latent:
                            model.update_teacher(model.momentum(global_step, max_steps))
                    if global_step % checkpoint_every == 0 or stopping[0]:
                        save('latest', epoch, last_cursor)
                    if stopping[0] or global_step >= max_steps:
                        break
            if stopping[0]:
                break
            model.eval()
            val_loader, _ = _loader(val, train_cfg, 0, False)
            val_loss = val_units = 0
            with torch.no_grad():
                for index, batch in enumerate(val_loader):
                    with autocast():
                        loss, units, _ = loss_for(batch, True, index)
                    val_loss += loss.item()*units
                    val_units += units
            score = val_loss/val_units
            if not math.isfinite(score):
                raise FloatingPointError('Nonfinite validation loss')
            completed = last_cursor == len(loader)
            saved_epoch, saved_cursor = (epoch+1, 0) if completed else (epoch, last_cursor)
            if score < best:
                best = score
                save('best', saved_epoch, saved_cursor)
            save('latest', saved_epoch, saved_cursor)
            elapsed = time.monotonic()-start_time
            record = dict(epoch=epoch, global_step=global_step, train_loss=epoch_loss/max(1,total_units),
                val_loss=score, wall_seconds=elapsed, train_examples=total_examples,
                examples_per_second=total_examples/max(1e-9, time.monotonic()-epoch_start_time), lr=optimizer.param_groups[0]['lr'],
                peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
            if diagnostic_windows is not None:
                record['diagnostics'] = {'student': representation_diagnostics(encoder, diagnostic_windows)}
                if latent:
                    record['diagnostics']['teacher'] = representation_diagnostics(model.teacher, diagnostic_windows)
                    record['ema_momentum'] = model.momentum(global_step, max_steps)
            with (out/'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
            epoch, next_batch = saved_epoch, saved_cursor
            if global_step >= max_steps:
                break
    finally:
        signal.signal(signal.SIGTERM, old_handler)
    return out
