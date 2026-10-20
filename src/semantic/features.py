"""Frozen causal word features and explicit stimulus-time alignment."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

MODEL = 'openai-community/gpt2'
REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
BAD_WORDS = {'sentence_start', 'sentence_end', 'br', 'lg', 'ls', 'ns', 'sp', ''}


def read_word_intervals(path, reference_dir=None):
    """Use the checked-in Tang/Huth TextGrid parser, without importing its pipeline."""
    root = Path(reference_dir) if reference_dir else Path(__file__).resolve().parents[2] / 'reference/semantic-decoding'
    parser = root / 'decoding/utils_ridge/textgrid.py'
    if not parser.is_file():
        raise FileNotFoundError(f'Tang/Huth TextGrid parser not found: {parser}')
    spec = importlib.util.spec_from_file_location('_semantic_textgrid', parser)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grid = module.TextGrid(Path(path).read_text())
    tiers = [tier for tier in grid.tiers if 'word' in tier.tier_name().lower()]
    if len(tiers) != 1:
        raise ValueError(f'{path}: expected exactly one explicitly named word tier')
    intervals = [(float(a), float(b), str(w).strip().strip('{}').strip())
                 for a, b, w in tiers[0].make_simple_transcript()]
    intervals = [(a, b, w) for a, b, w in intervals if w.lower() not in BAD_WORDS]
    if not intervals or any(not np.isfinite([a,b]).all() or b <= a for a,b,_ in intervals):
        raise ValueError(f'{path}: missing or invalid word intervals')
    if any(intervals[i][0] < intervals[i-1][0] for i in range(1,len(intervals))):
        raise ValueError('Word intervals must be time ordered')
    return intervals


def tr_centers(run, manifest_dir):
    """Require verified acquisition centers in seconds relative to stimulus onset."""
    n = int(run['n_tr'])
    if run.get('tr_times_path'):
        path = Path(run['tr_times_path'])
        centers = np.load(path if path.is_absolute() else Path(manifest_dir) / path)
    elif run.get('tr_start_seconds') is not None:
        centers = float(run['tr_start_seconds']) + np.arange(n) * float(run['tr_seconds'])
    else:
        raise ValueError(f"{run['run_id']}: explicit stimulus-relative TR center times required")
    centers = np.asarray(centers, dtype=float)
    if centers.shape != (n,) or not np.isfinite(centers).all() or np.any(np.diff(centers) <= 0):
        raise ValueError('TR centers must be finite, increasing, and match neural rows')
    return centers


def downsample_words(intervals, vectors, centers, tr_seconds):
    """Average word-midpoint features inside [center-TR/2, center+TR/2)."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(intervals) or not np.isfinite(vectors).all():
        raise ValueError('Invalid word feature matrix')
    midpoint = np.array([(a+b)/2 for a,b,_ in intervals])
    output = np.zeros((len(centers), vectors.shape[1]), dtype=np.float32)
    valid = np.zeros(len(centers), dtype=bool)
    for i, center in enumerate(centers):
        use = (midpoint >= center-tr_seconds/2) & (midpoint < center+tr_seconds/2)
        if use.any():
            output[i] = vectors[use].mean(0)
            valid[i] = True
    return output, valid


class FrozenGPT2:
    def __init__(self, model=MODEL, revision=REVISION, device='cpu', layer=9, context_words=20):
        import re
        import torch
        from transformers import AutoModel, AutoTokenizer
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('Model revision must be a pinned 40-character commit SHA')
        self.torch, self.device = torch, device
        self.layer, self.context_words = layer, context_words
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model, revision=revision).to(device).eval()
        self.model.requires_grad_(False)
        if layer > self.model.config.num_hidden_layers:
            raise ValueError('Requested hidden layer does not exist')

    def encode(self, words, batch_size=16):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        outputs = []
        for start in range(0,len(words),batch_size):
            texts = [' '.join(words[max(0,i-self.context_words):i+1])
                     for i in range(start,min(start+batch_size,len(words)))]
            tokens = self.tokenizer(texts, padding=True, truncation=True,
                                    max_length=self.model.config.n_positions, return_tensors='pt')
            tokens = {key: value.to(self.device) for key,value in tokens.items()}
            with self.torch.inference_mode():
                hidden = self.model(**tokens, output_hidden_states=True).hidden_states[self.layer]
                indices = tokens['attention_mask'].sum(1)-1
                last = hidden[self.torch.arange(len(texts),device=self.device),indices]
            outputs.append(last.cpu().float().numpy())
        return np.concatenate(outputs)


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def run_reference_embedding(text, encoder, batch_size=16):
    """Coarse cue/reference feature only; carries no word timing claim.

    Average frozen causal word features over the supplied whole reference text.
    This represents a run's known cue/reference, not the imagined speech itself.
    """
    words = text.split()
    if not words:
        raise ValueError('Run-level reference text is empty')
    vector = encoder.encode(words, batch_size).mean(axis=0).astype(np.float32)
    if not np.isfinite(vector).all():
        raise ValueError('Run-level reference embedding is nonfinite')
    return vector
