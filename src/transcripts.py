"""Transcript exposure and oracle nuisance features for the fixed v3 targets.

Word indices refer to the complete BAD_WORDS-filtered, ordered transcript that
``build_semantic_targets.py`` passes to FrozenGPT2, including unbinned words.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .semantic.features import read_word_intervals, tr_centers


POSITION_NAMES = ('p', 'p_squared', 'log1p_t')
SPEECH_RATE_NAMES = tuple(name for delay in range(4) for name in
                          (f'word_count_t_plus_{delay}', f'mean_word_duration_t_plus_{delay}'))
NUISANCE_GROUPS = {'position': POSITION_NAMES, 'speech_rate': SPEECH_RATE_NAMES}


def run_words(run, manifest_path, reference_dir=None):
    """Return ordered dict rows: word, start, end, midpoint, and bin_index.

    Relative stimulus/timeline paths resolve against the manifest directory,
    exactly as in the target builder. Bins are half-open centered TR intervals;
    words outside all bins retain ``bin_index=None`` and can supply context.
    Overlapping TR intervals are rejected because a single bin index would not
    faithfully represent ``downsample_words`` in that case.
    """
    manifest_dir = Path(manifest_path).resolve().parent
    # build_semantic_targets resolved relative paths against its input (neural)
    # manifest and copied them unchanged into the semantic manifest.
    import json as _json
    source = (_json.loads(Path(manifest_path).read_text()).get('neural_manifest_path')
              if Path(manifest_path).is_file() else None)
    if source:
        manifest_dir = Path(source).resolve().parent
    path = Path(run['stimulus_path'])
    if not path.is_absolute():
        path = manifest_dir / path
    intervals = read_word_intervals(path, reference_dir)
    centers = tr_centers(run, manifest_dir)
    tr_seconds = float(run['tr_seconds'])
    if not np.isfinite(tr_seconds) or tr_seconds <= 0:
        raise ValueError('TR duration must be finite and positive')
    left, right = centers - tr_seconds / 2, centers + tr_seconds / 2
    if np.any(left[1:] < right[:-1]):
        raise ValueError('Overlapping TR bins cannot have a single word bin index')
    rows = []
    for start, end, word in intervals:
        midpoint = (start + end) / 2
        t = int(np.searchsorted(left, midpoint, side='right') - 1)
        bin_index = t if t >= 0 and midpoint < right[t] else None
        rows.append(dict(word=word, start=start, end=end, midpoint=midpoint,
                         bin_index=bin_index))
    return rows


def consumed_words(run_words_by_run, targets):
    """Return sets ``in_bin``, ``context_only`` and ``all`` of word instances.

    ``targets`` contains (run_id, target_TR) pairs, not TargetIndex run positions.
    Each in-bin word consumes itself and up to 20 preceding filtered words in
    that run. Context shared between targets is counted once; a word in any
    selected bin is classified as in-bin even when another target uses it as
    context. Duplicate target pairs do not multiply exposure.
    """
    bins = {}
    for run_id, t in targets:
        if run_id not in run_words_by_run:
            raise KeyError(f'No transcript for run {run_id}')
        if int(t) != t or t < 0:
            raise ValueError('Target TRs must be nonnegative integers')
        bins.setdefault(run_id, set()).add(int(t))
    in_bin, all_words = set(), set()
    for run_id, selected in bins.items():
        for i, row in enumerate(run_words_by_run[run_id]):
            if row['bin_index'] in selected:
                in_bin.add((run_id, i))
                all_words.update((run_id, j) for j in range(max(0, i - 20), i + 1))
    return {'in_bin': in_bin, 'context_only': all_words - in_bin, 'all': all_words}


def _span_union(intervals):
    """Merge overlapping/touching intervals, preserving gaps between words."""
    merged = []
    for start, end in sorted(intervals):
        start, end = float(start), float(end)
        if not np.isfinite([start, end]).all() or end <= start:
            raise ValueError('Consumed word spans must be finite and positive')
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def exposure_summary(run_words_by_run, targets, tokenizer, sessions, *, tr_seconds=2.0):
    """Return JSON-ready exposure counts and inspectable word/span unions.

    ``sessions`` maps run IDs to acquisition session IDs (as TargetIndex.sessions).
    Tokens are the sum of standalone ``tokenizer.encode(' ' + word,
    add_special_tokens=False)`` lengths over unique *instances*, not vocabulary
    types, repeated context windows, or the original concatenated LM inputs.
    ``target_seconds`` is a nominal clock count, distinct from span exposure.
    """
    if not np.isfinite(tr_seconds) or tr_seconds <= 0:
        raise ValueError('TR duration must be finite and positive')
    targets = sorted(set(targets))
    consumed = consumed_words(run_words_by_run, targets)
    touched = sorted({run_id for run_id, _ in targets})
    session_ids = set()
    for run_id in touched:
        session = sessions[run_id]
        if session is None:
            raise ValueError(f'Missing acquisition session for run {run_id}')
        # Accept the comma-separated session convention used by NeuralWindows.
        values = session.split(',') if isinstance(session, str) else (
            session if isinstance(session, (list, tuple, set)) else [session])
        session_ids.update(str(value).strip() for value in values if str(value).strip())
    token_counts = {'in_bin': 0, 'context_only': 0}
    by_run = {}
    for run_id in touched:
        rows = run_words_by_run[run_id]
        groups = {group: sorted(i for r, i in consumed[group] if r == run_id)
                  for group in token_counts}
        indices = sorted(groups['in_bin'] + groups['context_only'])
        spans = _span_union((rows[i]['start'], rows[i]['end']) for i in indices)
        counts = {group: sum(len(tokenizer.encode(' ' + rows[i]['word'],
                                                 add_special_tokens=False)) for i in ids)
                  for group, ids in groups.items()}
        for group, count in counts.items():
            token_counts[group] += count
        by_run[run_id] = {
            'in_bin_word_indices': groups['in_bin'],
            'context_only_word_indices': groups['context_only'],
            'consumed_span_union': spans,
            'duration_seconds': sum(end - start for start, end in spans),
            'unique_gpt2_tokens': sum(counts.values()),
        }
    return {
        'selected_targets': len(targets), 'target_seconds': len(targets) * float(tr_seconds),
        'unique_word_instances': len(consumed['all']),
        'in_bin_word_instances': len(consumed['in_bin']),
        'context_only_word_instances': len(consumed['context_only']),
        'unique_gpt2_tokens': sum(token_counts.values()),
        'in_bin_gpt2_tokens': token_counts['in_bin'],
        'context_only_gpt2_tokens': token_counts['context_only'],
        'duration_seconds': sum(row['duration_seconds'] for row in by_run.values()),
        'runs_touched': len(touched), 'sessions_touched': len(session_ids),
        'counting_convention': {
            'targets': 'Unique (run_id, TR) pairs; nominal duration is count times TR seconds.',
            'words': 'Unique (run_id, index in full BAD_WORDS-filtered transcript); '
                     'selected-bin words plus up to 20 preceding words per in-bin word. '
                     'In-bin classification takes precedence over context-only.',
            'tokens': 'Sum of GPT-2 token counts for each unique word instance independently '
                      'encoded as " " + word, without special tokens; repeated lexical words '
                      'in different instances count separately. Not distinct token IDs or '
                      'the exact tokenization of concatenated contextual LM inputs.',
            'duration': 'Union of consumed [start, end] word spans separately per run, '
                        'then summed across runs; silence gaps are excluded.',
            'sessions': 'Distinct acquisition session IDs of selected-target runs.',
        },
        'by_run': by_run,
    }


def nuisance_features(index, run_words_by_run, *, groups=('position', 'speech_rate')):
    """Return (float64 matrix, feature-name list) in ``index.items`` order.

    Select ``groups='position'``, ``groups='speech_rate'``, or both. Columns
    always use canonical position-then-speech order. Speech columns are oracle
    transcript counts and mean durations, paired for bins t, t+1, t+2, t+3;
    empty (including out-of-run) bins have count and duration zero.
    """
    groups = {groups} if isinstance(groups, str) else set(groups)
    if not groups or groups - NUISANCE_GROUPS.keys():
        raise ValueError('Select nuisance groups position and/or speech_rate')
    names = [name for group, columns in NUISANCE_GROUPS.items() if group in groups
             for name in columns]
    rates = {}
    if 'speech_rate' in groups:
        for ri, run in enumerate(index.runs):
            n_tr = int(run['n_tr'])
            counts, durations = np.zeros(n_tr), np.zeros(n_tr)
            for row in run_words_by_run[run['run_id']]:
                t = row['bin_index']
                if t is not None and 0 <= t < n_tr:
                    counts[t] += 1
                    durations[t] += row['end'] - row['start']
            means = np.divide(durations, counts, out=np.zeros(n_tr), where=counts > 0)
            rates[ri] = np.column_stack((counts, means))
    features = np.zeros((len(index.items), len(names)), dtype=np.float64)
    for i, (ri, t) in enumerate(index.items):
        values = []
        n_tr = int(index.runs[ri]['n_tr'])
        if 'position' in groups:
            p = t / (n_tr - 1) if n_tr > 1 else 0.0
            values.extend((p, p**2, np.log1p(t)))
        if 'speech_rate' in groups:
            for delay in range(4):
                values.extend(rates[ri][t + delay] if 0 <= t + delay < n_tr else (0.0, 0.0))
        features[i] = values
    return features, names
