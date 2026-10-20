#!/usr/bin/env python3
"""Full fine-tuning arms on v2 common targets, several small jobs per GPU.

Each --arm is NAME=random or NAME=PATH_TEMPLATE (formatted with {seed}) for an
SSL encoder. Jobs resume from latest.pt, are marked complete only after the
full epoch budget, and are then scored with the v2 endpoints on best.pt.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import load_config, save_json
from src.targets import PROTOCOL_ID


def run_job(job, python):
    out = Path(job['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    config_path = out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != job:
        raise ValueError(f'Existing job has a different config: {out}')
    save_json(job, config_path)
    log = (out/'job.log').open('a')
    if not (out/'completed.json').exists():
        command = [python, 'scripts/train_semantic.py', '--config', str(config_path)]
        if (out/'latest.pt').exists():
            command += ['--resume', 'latest']
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        import torch
        state = torch.load(out/'latest.pt', map_location='cpu', weights_only=False)
        if state['epoch'] < job['training']['epochs']:
            raise SystemExit(f'Interrupted: rerun to resume {out}')
        save_json({'checkpoint': str(out/'best.pt'), 'global_step': state['global_step']}, out/'completed.json')
    if not (out/'evaluation_v2'/'metrics.json').exists():
        subprocess.run([python, '-c', 'import sys; from src.evaluation.finetuned import evaluate_checkpoint; '
                        'evaluate_checkpoint(sys.argv[1])', str(out/'best.pt')],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    return out.name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/semantic_uts01.yaml')
    p.add_argument('--root', required=True)
    p.add_argument('--arm', action='append', required=True, help='NAME=random or NAME=encoder path template')
    p.add_argument('--seeds', type=int, nargs='+', default=[17, 23, 47])
    p.add_argument('--cutoffs', type=int, nargs='+', default=[3])
    p.add_argument('--fractions', type=float, nargs='+', default=[1.0, 0.1])
    p.add_argument('--parallel', type=int, default=4)
    p.add_argument('--set', action='append', default=[])
    a = p.parse_args()
    jobs = []
    for seed in a.seeds:
        for cutoff in a.cutoffs:
            for fraction in a.fractions:
                for arm in a.arm:
                    name, source = arm.split('=', 1)
                    cfg = load_config(a.config, a.set)
                    cfg['training']['seed'] = seed
                    cfg['data']['num_workers'] = 4
                    cfg['semantic'].update(protocol=PROTOCOL_ID, common_cutoffs=[3, 4], target_phase=0,
                                           lag_tr=cutoff, label_fraction=fraction,
                                           init='random' if source == 'random' else 'ssl')
                    if source == 'random':
                        cfg['semantic'].pop('pretrained_encoder', None)
                    else:
                        cfg['semantic']['pretrained_encoder'] = source.format(seed=seed)
                    cfg['output_dir'] = str(Path(a.root)/f'{name}_c{cutoff}_f{fraction:g}_seed{seed}')
                    jobs.append(cfg)
    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        for name in pool.map(lambda job: run_job(job, sys.executable), jobs):
            print('finished', name, flush=True)


if __name__ == '__main__':
    main()
