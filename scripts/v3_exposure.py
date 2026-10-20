#!/usr/bin/env python3
"""Account for v3 transcript supervision and disclosed development exposure."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.probe import code_fingerprint
from src.runtime import file_sha256, load_config, save_json
from src.semantic.features import MODEL, REVISION
from src.targets import TargetIndex
from src.transcripts import exposure_summary, run_words


FRACTIONS = (1.0, 0.25, 0.1, 0.05, 0.01)


def target_pairs(index, positions=None):
    """Convert index positions to public (run_id, TR) transcript identities."""
    positions = range(len(index.items)) if positions is None else positions
    return [(index.runs[index.items[i][0]]['run_id'], index.items[i][1]) for i in positions]


def dense_training_index(manifest_path):
    """Reuse all TargetIndex validation and cutoffs, removing only stride phase.

    Window 32 and cutoffs (3, 4) give 28 <= t <= n_tr - 5, with valid[t].
    Stride 1 includes every eligible phase without changing src/targets.py.
    """
    return TargetIndex(manifest_path, 'train', window=32, stride=1, phase=0, cutoffs=(3, 4))


def build_report(config, tokenizer, *, tokenizer_model=MODEL, tokenizer_revision=REVISION):
    """Build the report using an injected tokenizer (no model is instantiated)."""
    manifest_path = config['data']['manifest']
    indices = {split: TargetIndex(manifest_path, split, window=32, stride=4, phase=0,
                                  cutoffs=(3, 4)) for split in ('train', 'val', 'test')}
    dense = dense_training_index(manifest_path)
    # Test identities are validated and disclosed, but test transcripts are not
    # needed for training/development accounting.
    runs = indices['train'].runs + indices['val'].runs
    if any(float(run['tr_seconds']) != 2.0 for run in runs):
        raise ValueError('The fixed v3 exposure protocol requires two-second TRs')
    words = {run['run_id']: run_words(run, manifest_path) for run in runs}
    sessions = {run['run_id']: run.get('session') for run in runs}

    def summarize(index, positions=None):
        return exposure_summary(words, target_pairs(index, positions), tokenizer, sessions)

    budgets = {}
    for fraction in FRACTIONS:
        positions, subset = indices['train'].label_subset(fraction, 2718, 64, 'block')
        budgets[f'{fraction:g}'] = dict(summarize(indices['train'], positions), labels=subset)
    return {
        'protocol': 'ssltang-v3',
        'provenance': {
            'config': config, 'manifest_path': str(Path(manifest_path).resolve()),
            'manifest_sha256': file_sha256(manifest_path), 'code_sha256': code_fingerprint(),
            'tokenizer': {'model': tokenizer_model, 'revision': tokenizer_revision},
            'targets': {split: index.signature() for split, index in indices.items()},
        },
        'budgets': budgets,
        'validation': dict(summarize(indices['val']),
                           disclosure='Shared full-label development exposure beyond every training budget.'),
        'dense_training': dict(summarize(dense), target_index=dense.signature(),
                               target_ids=dense.ids,
                               disclosure='Separate changed training data budget; all eligible TR phases.'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/v2/phase_a.yaml')
    parser.add_argument('--set', action='append', default=[])
    parser.add_argument('--out', required=True, help='results/v3/<subject>/exposure.json')
    parser.add_argument('--tokenizer-model', default=MODEL)
    parser.add_argument('--tokenizer-revision', default=REVISION)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[0-9a-f]{40}', args.tokenizer_revision):
        parser.error('--tokenizer-revision must be an immutable 40-character commit SHA')
    config = load_config(args.config, args.set)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, revision=args.tokenizer_revision)
    report = build_report(config, tokenizer, tokenizer_model=args.tokenizer_model,
                          tokenizer_revision=args.tokenizer_revision)
    save_json(report, args.out)
    print(args.out, flush=True)


if __name__ == '__main__':
    main()
