#!/usr/bin/env python3
"""Audit paired saved artifacts without loading data or fitting models."""
import argparse
import copy
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', default='results/runs')
    p.add_argument('--transfer', default='results/transfer')
    p.add_argument('--subject', default='UTS01')
    p.add_argument('--seeds', type=int, nargs='+', default=[17, 23, 47])
    p.add_argument('--fractions', type=float, nargs='+', default=[1, .25, .1, .05, .01])
    p.add_argument('--out', default='experiments/study_validation.json')
    a = p.parse_args()
    root = Path(a.runs)
    jobs, subsets, commits, manifests = [], {}, set(), set()
    for seed in a.seeds:
        names = [f'ssl_{a.subject}_seed{seed}']
        for fraction in a.fractions:
            pair = []
            for arm in ['random', 'ssl']:
                name = f'{arm}_{a.subject}_fraction{fraction:g}_seed{seed}'
                run = root / name
                cfg = read(run/'config.json')
                labels = read(run/'label_subset.json')['label_metadata']
                ids = set(labels['label_ids'])
                assert len(ids) == labels['selected_examples'], name
                assert ids == subsets.setdefault(fraction, ids), 'Labels differ across arms/seeds'
                score = read(run/'evaluation_test/metrics.json')
                transfer = read(Path(a.transfer)/name/'results.json')
                assert score['checkpoint_sha256'] == transfer['provenance']['checkpoint_sha256'], name
                assert score['split'] == 'test' and transfer['dataset'] == 'ds004510', name
                assert transfer['provenance']['normalization']['fit_during_evaluation'] is False
                assert cfg['training']['seed'] == seed and cfg['semantic']['init'] == arm
                assert cfg['semantic']['label_fraction'] == fraction
                comparison = copy.deepcopy(cfg)
                comparison.pop('output_dir')
                comparison['semantic'].pop('init')
                comparison['semantic'].pop('pretrained_encoder', None)
                pair.append(comparison)
                manifests.add(score['manifest_sha256'])
                names.append(name)
            assert pair[0] == pair[1], f'Optimization or architecture differs: seed={seed}, fraction={fraction}'
        for name in names:
            run = root/name
            done = read(run/'completed.json')
            metadata = read(run/'run_metadata.json')
            logs = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            assert logs and logs[-1]['global_step'] == done['global_step'], name
            assert all(math.isfinite(r[k]) for r in logs for k in ['train_loss', 'val_loss']), name
            commits.add(metadata['git_commit'])
            jobs.append({'name': name, 'steps': done['global_step'],
                         'train_windows': metadata['train_windows'], 'validation_windows': metadata['val_windows'],
                         'logged_wall_seconds': logs[-1]['wall_seconds'],
                         'peak_gpu_bytes': max(r['peak_gpu_bytes'] for r in logs)})
    fractions = sorted(subsets)
    assert all(subsets[x] <= subsets[y] for x, y in zip(fractions, fractions[1:])), 'Subsets are not nested'
    assert len(manifests) == 1, 'Semantic manifests differ'
    report = {'subject': a.subject, 'seeds': a.seeds, 'fractions': a.fractions,
              'completed_training_jobs': len(jobs),
              'completed_story_and_transfer_evaluations_each': len(a.seeds)*len(a.fractions)*2,
              'identical_paired_configs_except_initialization': True,
              'identical_and_nested_labels_across_all_arms_and_seeds': True,
              'same_checkpoint_for_story_and_transfer': True,
              'semantic_manifest_sha256': next(iter(manifests)),
              'training_git_commits': sorted(commits),
              'total_logged_training_wall_seconds': sum(r['logged_wall_seconds'] for r in jobs),
              'peak_allocated_gpu_bytes': max(r['peak_gpu_bytes'] for r in jobs),
              'jobs': jobs}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Validated {len(jobs)} training jobs and {report["completed_story_and_transfer_evaluations_each"]} paired story/transfer records: {out}')


if __name__ == '__main__':
    main()
