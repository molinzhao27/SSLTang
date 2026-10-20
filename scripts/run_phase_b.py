#!/usr/bin/env python3
"""Protocol v2 Phase B: reconstruction versus EMA latent-prediction pretraining.

Stage ``pretrain`` fits both objectives per seed, sequentially so measured cost
is comparable, and marks a run complete only at the full update budget. Stage
``readout`` applies the locked Phase A ridge procedure to each final student;
stage ``finetune`` runs the existing full fine-tuning from each final student.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import load_config, save_json, file_sha256

OBJECTIVES = {'recon': 'configs/v2/ssl_recon.yaml', 'latent': 'configs/v2/ssl_latent.yaml'}
ARMS = {'recon': 'R', 'latent': 'L'}


def pretrain(seeds, root, objectives, overrides, subject):
    import torch
    for seed in seeds:
        for objective in objectives:
            cfg = load_config(OBJECTIVES[objective], overrides)
            cfg['training']['seed'] = seed
            out = Path(root)/f'{objective}_{subject}_seed{seed}'
            cfg['output_dir'] = str(out)
            out.mkdir(parents=True, exist_ok=True)
            config_path = out/'config.json'
            if config_path.exists() and json.loads(config_path.read_text()) != cfg:
                raise ValueError(f'Existing run has a different config: {out}')
            save_json(cfg, config_path)
            if (out/'completed.json').exists():
                continue
            command = [sys.executable, 'scripts/pretrain_ssl.py', '--config', str(config_path)]
            if (out/'latest.pt').exists():
                command += ['--resume', 'latest']
            subprocess.run(command, check=True)
            state = torch.load(out/'latest.pt', map_location='cpu', weights_only=False)
            if state['epoch'] < cfg['training']['epochs']:
                raise SystemExit(f'Interrupted: rerun to resume {out}')
            save_json({'encoder': str(out/'encoder_latest.pt'), 'global_step': state['global_step'],
                       'encoder_sha256': file_sha256(out/'encoder_latest.pt')}, out/'completed.json')
            print('pretrained', out, state['global_step'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['pretrain', 'readout', 'finetune'], required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[17, 23, 47])
    p.add_argument('--root', default='results/v2/ssl')
    p.add_argument('--subject', default='UTS01')
    p.add_argument('--finetune-root', default='results/v2/finetune_b')
    p.add_argument('--finetune-set', action='append', default=[], help='Fine-tuning config overrides')
    p.add_argument('--objectives', nargs='+', default=list(OBJECTIVES))
    p.add_argument('--readout-config', default='configs/v2/phase_b.yaml')
    p.add_argument('--readout-out', default='results/v2/phase_b')
    p.add_argument('--set', action='append', default=[], help='Pretraining config overrides')
    p.add_argument('--readout-set', action='append', default=[], help='Readout config overrides')
    a = p.parse_args()
    if a.stage == 'pretrain':
        pretrain(a.seeds, a.root, a.objectives, a.set, a.subject)
    else:
        for seed in a.seeds:  # Never read an encoder whose budget did not complete.
            for objective in a.objectives:
                run = Path(a.root)/f'{objective}_{a.subject}_seed{seed}'
                if not (run/'completed.json').exists():
                    raise SystemExit(f'Pretraining incomplete: {objective} seed {seed}')
                record = json.loads((run/'completed.json').read_text())
                if file_sha256(run/'encoder_latest.pt') != record['encoder_sha256']:
                    raise SystemExit(f'Encoder changed after completion: {run}')
        if a.stage == 'readout':
            sources = {f'{o}_final': [ARMS[o], f'{a.root}/{o}_{a.subject}_seed{{seed}}/encoder_latest.pt']
                       for o in a.objectives}
            overrides = [f'encoders.seeds={json.dumps(a.seeds)}', f'encoders.sources={json.dumps(sources)}',
                         f'output_dir={a.readout_out}', *a.readout_set]
            subprocess.run([sys.executable, 'scripts/run_phase_a.py', '--config', a.readout_config,
                            '--only', 'encoders', *[x for o in overrides for x in ('--set', o)]], check=True)
        else:
            arms = [f'--arm={o}={a.root}/{o}_{a.subject}_seed{{seed}}/encoder_latest.pt' for o in a.objectives]
            subprocess.run([sys.executable, 'scripts/run_v2_finetune.py', '--root', a.finetune_root,
                            *arms, '--seeds', *map(str, a.seeds), '--cutoffs', '3',
                            '--fractions', '1', '0.1', '--parallel', '4',
                            *[x for o in a.finetune_set for x in ('--set', o)]], check=True)


if __name__ == '__main__':
    main()
