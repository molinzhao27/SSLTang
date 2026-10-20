#!/usr/bin/env python3
"""Produce a separate, language-bearing semantic manifest; never modify SSL inputs."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.semantic.features import (MODEL, REVISION, FrozenGPT2, downsample_words,
                                   fingerprint, read_word_intervals, tr_centers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--reference-dir', default=str(Path(__file__).resolve().parents[1] / 'reference/semantic-decoding'))
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be an immutable 40-character commit SHA")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    source = Path(args.manifest).resolve()
    manifest = json.loads(source.read_text())
    runs = manifest['runs'] if isinstance(manifest,dict) else manifest
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(model=args.model, revision=args.revision, layer=9, context_words=20,
                    token_pool='last subword of current word', frozen=True,
                    downsampling='word midpoint mean in centered TR interval',
                    target_lag_seconds=0, exact_tang_replication=False)
    encoder = None
    result, skipped = [], []
    for run in runs:
        if not run.get('stimulus_path'):
            skipped.append(dict(run_id=run['run_id'], reason='No verified timed reference text'))
            continue
        if run.get('dataset') == 'ds004510' and not run.get('stimulus_timing_verified', False):
            skipped.append(dict(run_id=run['run_id'],reason='Transfer reference timing is unverified; use run-level evaluation'))
            continue
        if 'imag' in str(run.get('condition','')).lower():
            skipped.append(dict(run_id=run['run_id'],reason='Imagined speech has no verified word timeline'))
            continue
        path = Path(run['stimulus_path'])
        if not path.is_absolute(): path = source.parent / path
        if path.suffix.lower() != '.textgrid':
            skipped.append(dict(run_id=run['run_id'],reason='Verified word TextGrid required'))
            continue
        intervals = read_word_intervals(path, args.reference_dir)
        centers = tr_centers(run, source.parent)
        cache_key = fingerprint(dict(metadata=metadata, run=run,
                          stimulus_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          centers=centers.tolist()))
        target_path = output / f'{cache_key}.targets.npy'
        valid_path = output / f'{cache_key}.valid.npy'
        resume = target_path.exists() and valid_path.exists()
        if resume:
            targets,valid = np.load(target_path),np.load(valid_path)
            resume = targets.ndim == 2 and targets.shape[0] == len(centers) and valid.shape == (len(centers),) and valid.dtype == np.bool_ and np.isfinite(targets).all()
        if not resume:
            if encoder is None: encoder = FrozenGPT2(args.model,args.revision,args.device)
            words = encoder.encode([w for _,_,w in intervals],args.batch_size)
            targets,valid = downsample_words(intervals, words, centers,float(run['tr_seconds']))
            np.save(target_path, targets)
            np.save(valid_path,valid)
        result.append(dict(run, target_path=str(target_path), target_valid_path=str(valid_path),
                           semantic_metadata=dict(metadata, embedding_dim=targets.shape[1],
                                                   cache_key=cache_key)))
        print(f"{run['run_id']}: {valid.sum()}/{len(valid)} valid TR targets", flush=True)
    destination = output / 'semantic_manifest.json'
    header = {key: value for key,value in manifest.items() if key != 'runs'} if isinstance(manifest,dict) else {}
    header.update(runs=result, skipped_runs=skipped, semantic_metadata=metadata,
                  neural_manifest_path=str(source),
                  neural_manifest_hash=hashlib.sha256(source.read_bytes()).hexdigest())
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(header,indent=2,allow_nan=False)+'\n')
    temporary.replace(destination)
    print(destination)

if __name__ == '__main__': main()
