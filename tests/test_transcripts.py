"""Synthetic transcript accounting; tokenizer calls are always mocked."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import v3_exposure
from src import transcripts
from src.semantic.features import MODEL, REVISION, downsample_words
from src.targets import TargetIndex
from test_v2 import manifest


def word(word, start, end, bin_index):
    return dict(word=word, start=start, end=end, midpoint=(start + end) / 2,
                bin_index=bin_index)


class MockTokenizer:
    """Repeated token IDs still count once per token occurrence/word instance."""
    def __init__(self):
        self.calls = []

    def encode(self, text, *, add_special_tokens):
        assert text.startswith(' ') and add_special_tokens is False
        self.calls.append(text)
        return [7] * len(text.strip().split('-'))


def textgrid(intervals):
    end = max(b for _, b, _ in intervals)
    header = f'''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = {end}
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "word"
        xmin = 0
        xmax = {end}
        intervals: size = {len(intervals)}
'''
    return header + ''.join(f'''        intervals [{i}]:
            xmin = {a}
            xmax = {b}
            text = "{w}"
''' for i, (a, b, w) in enumerate(intervals, 1))


@pytest.fixture
def reference_parser():
    root = Path(__file__).resolve().parents[1] / 'reference/semantic-decoding'
    if not (root / 'decoding/utils_ridge/textgrid.py').is_file():
        pytest.skip('Parser-dependent test: local Tang/Huth TextGrid parser is unavailable')
    return root


def test_bin_assignment_matches_downsampling_without_parser(tmp_path, monkeypatch):
    # ds003020's center offset is 11 seconds. Boundaries 10,12,14,16 are
    # assigned to the bin on their right; outside words remain in the table.
    intervals = [(9.25, 9.75, 'pre'), (9.5, 10.5, 'left'), (10.5, 11.5, 'inside'),
                 (11.5, 12.5, 'next'), (13.5, 14.5, 'third'), (15.5, 16.5, 'post')]
    monkeypatch.setattr(transcripts, 'read_word_intervals', lambda *_: intervals)
    run = dict(run_id='r', stimulus_path='words.TextGrid', n_tr=3,
               tr_start_seconds=11, tr_seconds=2)
    rows = transcripts.run_words(run, tmp_path / 'manifest.json')
    assert [row['bin_index'] for row in rows] == [None, 0, 0, 1, 2, None]
    vectors = np.arange(len(rows), dtype=np.float32)[:, None]
    targets, valid = downsample_words(intervals, vectors, np.array([11, 13, 15]), 2)
    for t in range(3):
        indices = [i for i, row in enumerate(rows) if row['bin_index'] == t]
        assert bool(indices) == bool(valid[t])
        np.testing.assert_array_equal(targets[t], vectors[indices].mean(0))


def test_textgrid_filters_bad_words_and_keeps_unbinned_context(tmp_path, reference_parser):
    intervals = [(0, 1, 'sentence_start'), (1, 2, '{before}'), (2, 3, 'SP'),
                 (9.5, 10.5, 'first'), (11.5, 12.5, 'second'), (13, 14, 'sentence_end')]
    (tmp_path / 'words.TextGrid').write_text(textgrid(intervals))
    run = dict(run_id='r', stimulus_path='words.TextGrid', n_tr=3,
               tr_start_seconds=11, tr_seconds=2, session='s1')
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'runs': [run]}))
    rows = transcripts.run_words(run, path, reference_parser)
    assert [row['word'] for row in rows] == ['before', 'first', 'second']
    assert [row['bin_index'] for row in rows] == [None, 0, 1]
    assert rows[1]['midpoint'] == 10
    used = transcripts.consumed_words({'r': rows}, [('r', 0)])
    assert used['context_only'] == {('r', 0)}


def test_explicit_timeline_and_silent_bins(tmp_path, monkeypatch):
    intervals = [(0.5, 1.5, 'a'), (2, 3, 'gap'), (4.5, 5.5, 'b')]
    monkeypatch.setattr(transcripts, 'read_word_intervals', lambda *_: intervals)
    np.save(tmp_path / 'times.npy', [1, 5, 9])
    run = dict(run_id='r', stimulus_path='x.TextGrid', n_tr=3,
               tr_times_path='times.npy', tr_seconds=2)
    rows = transcripts.run_words(run, tmp_path / 'manifest.json')
    assert [row['bin_index'] for row in rows] == [0, None, 1]
    _, valid = downsample_words(intervals, np.ones((3, 1)), np.array([1, 5, 9]), 2)
    np.testing.assert_array_equal(valid, [True, True, False])
    run.update(tr_times_path=None, tr_start_seconds=0, tr_seconds=0)
    with pytest.raises(ValueError):
        transcripts.run_words(run, tmp_path / 'manifest.json')


def test_context_clips_at_start_and_has_exactly_20_preceding_words():
    rows = [word(str(i), i, i + 1, i) for i in range(30)]
    tables = {'r': rows, 'other': rows}
    used = transcripts.consumed_words(tables, [('r', 0), ('r', 2)])
    assert used['in_bin'] == {('r', 0), ('r', 2)}
    assert used['context_only'] == {('r', 1)}
    used = transcripts.consumed_words(tables, [('r', 25)])
    assert used['all'] == {('r', i) for i in range(5, 26)}
    assert len(used['context_only']) == 20


def test_union_counts_shared_context_once_and_tokens_are_instance_counts():
    tables = {
        'r': [word('same', 0, 1, None), word('two-parts', 0.5, 2, 0),
              word('same', 4, 5, 1), word('late', 5, 7, 2)],
        's': [word('same', 0, 3, 0)],
    }
    tokenizer = MockTokenizer()
    summary = transcripts.exposure_summary(tables, [('r', 0), ('r', 1), ('r', 1), ('s', 0)],
                                           tokenizer, {'r': 'session1', 's': 'session1'})
    assert summary['selected_targets'] == 3 and summary['target_seconds'] == 6
    assert summary['unique_word_instances'] == 4
    assert summary['in_bin_word_instances'] == 3
    assert summary['context_only_word_instances'] == 1
    assert summary['unique_gpt2_tokens'] == 5
    assert summary['in_bin_gpt2_tokens'] == 4 and summary['context_only_gpt2_tokens'] == 1
    assert sorted(tokenizer.calls) == [' same', ' same', ' same', ' two-parts']
    assert summary['duration_seconds'] == 6
    assert summary['runs_touched'] == 2 and summary['sessions_touched'] == 1
    assert summary['by_run']['r']['context_only_word_indices'] == [0]
    assert 'unique word instance' in summary['counting_convention']['tokens']


def test_span_union_arithmetic_and_empty_exposure():
    rows = [word('x', start, end, 0) for start, end in [(4, 5), (0, 2), (1, 3), (5, 6)]]
    summary = transcripts.exposure_summary({'r': rows}, [('r', 0)], MockTokenizer(), {'r': 's'})
    assert summary['duration_seconds'] == 5
    assert summary['by_run']['r']['consumed_span_union'] == [[0.0, 3.0], [4.0, 6.0]]
    empty = transcripts.exposure_summary({}, [], MockTokenizer(), {})
    assert empty['selected_targets'] == empty['unique_word_instances'] == 0
    assert empty['unique_gpt2_tokens'] == empty['duration_seconds'] == 0
    assert empty['runs_touched'] == empty['sessions_touched'] == 0


def test_nuisance_column_order_values_and_group_selection():
    index = SimpleNamespace(runs=[dict(run_id='r', n_tr=5), dict(run_id='s', n_tr=1)],
                            items=[(0, 1), (0, 0), (0, 4), (1, 0)])
    tables = {'r': [word('a', 0, 1, 0), word('b', 1, 3, 1), word('c', 3, 7, 1),
                    word('d', 7, 8, 3), word('outside', 20, 30, None)], 's': []}
    values, names = transcripts.nuisance_features(index, tables)
    assert names == ['p', 'p_squared', 'log1p_t',
                     'word_count_t_plus_0', 'mean_word_duration_t_plus_0',
                     'word_count_t_plus_1', 'mean_word_duration_t_plus_1',
                     'word_count_t_plus_2', 'mean_word_duration_t_plus_2',
                     'word_count_t_plus_3', 'mean_word_duration_t_plus_3']
    assert values.dtype == np.float64 and values.shape == (4, 11)
    np.testing.assert_allclose(values[0], [.25, .0625, np.log(2), 2, 3, 0, 0, 1, 1, 0, 0])
    np.testing.assert_allclose(values[1], [0, 0, 0, 1, 1, 2, 3, 0, 0, 1, 1])
    np.testing.assert_allclose(values[2], [1, 1, np.log(5), 0, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(values[3], np.zeros(11))
    for group, columns in [('position', slice(0, 3)), ('speech_rate', slice(3, 11))]:
        subset, subset_names = transcripts.nuisance_features(index, tables, groups=group)
        np.testing.assert_array_equal(subset, values[:, columns])
        assert subset_names == names[columns]
    position, _ = transcripts.nuisance_features(index, {}, groups='position')
    np.testing.assert_array_equal(position, values[:, :3])
    with pytest.raises(ValueError):
        transcripts.nuisance_features(index, tables, groups='unknown')


def test_dense_clock_report_and_cli_mock_tokenizer(tmp_path, monkeypatch):
    path = manifest(tmp_path, n_tr=(57, 54, 57, 54, 57, 54))
    content = json.loads(path.read_text())
    for run in content['runs']:
        run.update(tr_start_seconds=11, tr_seconds=2, stimulus_path=f"{run['run_id']}.TextGrid")
        # Files are synthetic; parser coverage is independent of this CLI test.
        (tmp_path / run['stimulus_path']).write_text(textgrid([(10, 12, 'toy')]))
    path.write_text(json.dumps(content))
    read_runs = []

    def toy_words(run, manifest_path):
        assert Path(manifest_path) == path
        read_runs.append(run['run_id'])
        return [word('same', 10 + 2 * t, 12 + 2 * t, t) for t in range(run['n_tr'])]

    monkeypatch.setattr(v3_exposure, 'run_words', toy_words)
    tokenizer = MockTokenizer()
    pin_calls = []

    def from_pretrained(model, *, revision):
        pin_calls.append((model, revision))
        return tokenizer

    monkeypatch.setitem(sys.modules, 'transformers',
                        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained)))
    cfg = tmp_path / 'config.yaml'
    cfg.write_text('data:\n  manifest: unused.json\n')
    out = tmp_path / 'results/v3/UTS01/exposure.json'
    v3_exposure.main(['--config', str(cfg), '--set', f'data.manifest={path}', '--out', str(out)])
    report = json.loads(out.read_text())
    assert pin_calls == [(MODEL, REVISION)]
    assert set(report['budgets']) == {'1', '0.25', '0.1', '0.05', '0.01'}
    old = TargetIndex(path, 'train', window=32, stride=4, phase=0, cutoffs=(3, 4))
    for fraction in v3_exposure.FRACTIONS:
        positions, labels = old.label_subset(fraction, 2718, 64, 'block')
        budget = report['budgets'][f'{fraction:g}']
        assert budget['selected_targets'] == len(positions)
        assert budget['labels'] == labels
    dense = v3_exposure.dense_training_index(path)
    expected = [(ri, t) for ri, run in enumerate(dense.runs)
                for t in range(28, run['n_tr'] - 4)
                if np.load(run['target_valid_path'])[t]]
    assert dense.items == expected
    assert set(old.items) <= set(dense.items)
    assert any(t % 4 != 0 for _, t in dense.items)
    assert report['dense_training']['selected_targets'] == len(expected)
    assert report['dense_training']['target_ids'] == dense.ids
    assert report['validation']['selected_targets'] == len(TargetIndex(path, 'val'))
    assert set(read_runs) == {r['run_id'] for r in content['runs'] if r['split'] in ('train', 'val')}
    provenance = report['provenance']
    assert provenance['manifest_sha256'] == v3_exposure.file_sha256(path)
    assert len(provenance['code_sha256']) == 64
    assert provenance['tokenizer'] == {'model': MODEL, 'revision': REVISION}
    for split in ('train', 'val', 'test'):
        assert provenance['targets'][split] == TargetIndex(path, split).signature()
