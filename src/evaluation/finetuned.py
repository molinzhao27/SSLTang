"""Score fine-tuned v2 semantic checkpoints on the common held-out targets."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..models import FmriEncoder
from ..runtime import file_sha256, save_json
from ..semantic.model import SemanticRegressor
from ..targets import PROTOCOL_ID, TargetIndex, TargetWindows
from .endpoints import score, shift_null


def evaluate_checkpoint(checkpoint_path, output_dir=None, device='cuda', batch_size=256, num_workers=4,
                        exclusions=(10, 32), shift_min_tr=40, shift_draws=2000, shift_seed=0):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    semantic = config['semantic']
    if checkpoint.get('task') != 'semantic' or semantic.get('protocol') != PROTOCOL_ID:
        raise ValueError('Expected a v2 semantic checkpoint')
    manifest = config['data']['manifest']
    if file_sha256(manifest) != checkpoint['manifest_sha256']:
        raise ValueError('Evaluation must use the training/split manifest')
    index_kwargs = dict(window=config['data']['window'], stride=config['data']['stride'],
                        phase=semantic.get('target_phase', 0), cutoffs=semantic.get('common_cutoffs', [3, 4]))
    cutoff = semantic.get('lag_tr', 3)
    mean, std = checkpoint['target_mean'].numpy(), checkpoint['target_std'].numpy()
    precision = config['training'].get('precision', 'bf16')
    model = None
    report = {'protocol': PROTOCOL_ID, 'checkpoint': str(Path(checkpoint_path).resolve()),
              'checkpoint_sha256': file_sha256(checkpoint_path), 'init': semantic['init'],
              'seed': config['training']['seed'], 'fraction': semantic['label_fraction'],
              'cutoff_tr': cutoff, 'global_step': checkpoint['global_step']}
    out = Path(output_dir) if output_dir else Path(checkpoint_path).parent/'evaluation_v2'
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for split in ('val', 'test'):
        index = TargetIndex(manifest, split, **index_kwargs)
        dataset = TargetWindows(index, cutoff)
        if model is None:
            model = SemanticRegressor(FmriEncoder(checkpoint['encoder_config']), dataset.target_dim,
                                      semantic.get('pooling', 'end'))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device).eval()
        predictions = []
        with torch.inference_mode():
            for batch in DataLoader(dataset, batch_size=batch_size, num_workers=num_workers):
                with torch.autocast('cuda', dtype=torch.bfloat16 if precision == 'bf16' else torch.float16,
                                    enabled=str(device).startswith('cuda') and precision != 'fp32'):
                    prediction = model(batch['x'].to(device, non_blocking=True))
                predictions.append(prediction.float().cpu().numpy() * std + mean)
        prediction, target = np.concatenate(predictions), index.targets()
        report[split] = score(prediction, target, index.run_ids, index.target_trs, index.sessions,
                              mean, std, exclusions)
        report[f'{split}_targets'] = index.signature()
        arrays[split] = prediction.astype(np.float16)
        if split == 'test':
            report['test_shift_null'] = shift_null(prediction, target, index.run_ids, index.sessions,
                                                   index.stride, shift_min_tr, shift_draws, shift_seed)
    np.savez_compressed(out/'predictions.npz', **arrays)
    save_json(report, out/'metrics.json')
    return report
