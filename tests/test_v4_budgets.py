"""v4 budgets: registered hash orders, nested story/section budgets, folds, half-story groups and the ledger."""
import hashlib
import json

import numpy as np
import pytest

from src import budgets as bud

SHARED = [f'story{i:02d}' for i in range(24)]


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def manifest(train, val, test, extra_train=()):
    runs = [dict(run_id=f'{split}-{s}', story=s, split=split, session='1', n_tr=300)
            for split, stories in (('train', list(train) + list(extra_train)), ('val', val), ('test', test))
            for s in stories]
    return dict(runs=runs)


def subjects():
    """UTS01-08-like split 15/5/4 (plus non-shared training stories) and a UTS09/10-like split 13/11/0."""
    a = manifest(SHARED[:15], SHARED[15:20], SHARED[20:], extra_train=['onlyA1', 'onlyA2'])
    b = manifest(SHARED[:13], SHARED[13:], [], extra_train=['onlyB'])
    return {'UTS01': a, 'UTS09': b}


def test_hash_order_is_sha256_of_the_registered_string_and_restricts_to_subsets():
    items = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    order = bud.hash_order(items, 'ssltang-v4-budget:')
    assert order == sorted(items, key=lambda s: sha(f'ssltang-v4-budget:{s}'))
    assert bud.hash_order(['delta', 'alpha', 'gamma'], 'ssltang-v4-budget:') == [s for s in order
                                                                                if s in ('delta', 'alpha', 'gamma')]
    assert bud.hash_order(items[::-1], 'ssltang-v4-budget:') == order
    with pytest.raises(ValueError):
        bud.hash_order(['a', 'a'], 'x:')


def test_subjects_constant():
    assert bud.SUBJECTS == tuple(f'UTS{i:02d}' for i in range(1, 11))
    assert bud.LPP_SECTIONS == tuple('123456789')
    assert [bud.dataset_of(s) for s in ('UTS01', 'UTS10', 'sub-EN057', 'UTS11')] == ['ds003020', 'ds003020',
                                                                                    'lpp', 'lpp']


def test_shared_stories_and_ds003020_budgets_are_nested_hash_prefixes():
    m = subjects()
    shared = bud.shared_stories(m)
    assert shared == sorted(SHARED)
    out = bud.ds003020_budgets(m['UTS01'], shared)
    assert list(out) == ['S3', 'S6', 'S15', 'ALL']
    expected = sorted(SHARED[:15], key=lambda s: sha(f'ssltang-v4-budget:{s}'))
    assert out['S15'] == expected
    assert out['S3'] == expected[:3] and out['S6'] == expected[:6]
    assert out['ALL'] == sorted(SHARED[:15] + ['onlyA1', 'onlyA2'], key=lambda s: sha(f'ssltang-v4-budget:{s}'))
    assert set(out['S15']) < set(out['ALL'])
    assert bud.isp_stories(out['ALL'], shared) == out['S15']  # ISP at ALL uses the S15 stories, same order.
    assert bud.isp_stories(out['S3'], shared) == out['S3']
    nine = bud.ds003020_budgets(m['UTS09'], shared)
    assert len(nine['S15']) == 13 and nine['S6'] == nine['S15'][:6]
    assert bud.split_stories(m['UTS09'])['val'] == sorted(SHARED[13:])


def test_budgets_reject_inconsistent_manifests():
    m = subjects()
    with pytest.raises(ValueError, match='absent'):
        bud.ds003020_budgets(m['UTS01'], SHARED + ['nowhere'])
    bad = manifest(SHARED[:15], SHARED[15:20], SHARED[20:])
    bad['runs'].append(dict(run_id='dup', story=SHARED[0], split='val', session='2', n_tr=300))
    with pytest.raises(ValueError, match='more than one split'):
        bud.ds003020_budgets(bad, SHARED)
    with pytest.raises(ValueError):
        bud.ds003020_budgets(manifest(SHARED[:5], SHARED[5:], []), SHARED)


def test_ts_stories_are_reference_stories_the_target_never_heard():
    target = manifest(['a', 'b'], ['v'], ['t'])
    reference = manifest(['a', 'x', 'y'], ['v'], ['t'])
    assert bud.ts_stories(target, reference) == ['x', 'y']


def test_inner_folds():
    s3 = ['a', 'b', 'c']
    assert bud.inner_folds('S3', s3) == [('a',), ('b',), ('c',)]
    s6 = list('abcdef')
    assert bud.inner_folds('S6', s6) == [('a', 'd'), ('b', 'e'), ('c', 'f')]
    s15 = [f's{i}' for i in range(15)]
    folds = bud.inner_folds('S15', s15)
    assert len(folds) == 5 and folds[0] == ('s0', 's5', 's10') and folds[4] == ('s4', 's9', 's14')
    s13 = s15[:13]
    folds = bud.inner_folds('S15', s13)
    assert [len(f) for f in folds] == [3, 3, 3, 2, 2]
    assert sorted(s for f in folds for s in f) == sorted(s13)  # Each story held out exactly once.
    for name, n in (('L3', 3), ('L5', 5), ('L8', 8)):
        sections = [str(i) for i in range(n)]
        assert bud.inner_folds(name, sections) == [(s,) for s in sections]
    with pytest.raises(ValueError, match='fixed-fold'):
        bud.inner_folds('ALL', s15)
    with pytest.raises(ValueError):
        bud.inner_folds('S7', s6)
    with pytest.raises(ValueError):
        bud.inner_folds('S6', ['a', 'b'])
    with pytest.raises(ValueError):
        bud.inner_folds('S3', ['a', 'a', 'b'])


def test_all_folds_are_the_v3_session_folds_over_the_training_stories():
    def sessions_manifest(n_sessions, per=2):
        train = [dict(run_id=f'r{i}', story=f't{i:02d}', split='train', session=f'{i // per}', n_tr=300)
                 for i in range(n_sessions * per)]
        return dict(runs=train + [dict(run_id='v', story='v', split='val', session='0', n_tr=300)])
    m = sessions_manifest(7)
    order = sorted({str(i) for i in range(7)}, key=lambda s: sha(f'ssltang-v3-folds:{s}'))
    expected = [tuple(sorted(f't{i:02d}' for s in order[f::5] for i in (2 * int(s), 2 * int(s) + 1)))
                for f in range(5)]  # Fixed five-fold session partition, round-robin over the hash order.
    assert bud.all_folds(m) == expected
    assert sorted(s for f in expected for s in f) == [f't{i:02d}' for i in range(14)]  # A partition of ALL.
    assert bud.all_folds(sessions_manifest(3)) == [('t00', 't01'), ('t02', 't03'), ('t04', 't05')]  # LOSO.
    assert bud.all_folds(sessions_manifest(1)) is None  # Not estimable.


def test_half_groups_split_each_story_at_its_middle_tr():
    stories = ['odd'] * 7 + ['even'] * 8
    trs = list(range(7)) + list(range(8))
    groups = bud.half_groups(stories, np.array(trs), {'odd': 7, 'even': 8})
    assert groups == ['odd:0'] * 3 + ['odd:1'] * 4 + ['even:0'] * 4 + ['even:1'] * 4
    assert bud.half_folds(['odd', 'even']) == [('odd:0',), ('odd:1',), ('even:0',), ('even:1',)]
    with pytest.raises(ValueError):
        bud.half_groups(['odd'], [7], {'odd': 7})
    with pytest.raises(ValueError):
        bud.half_groups(['odd', 'odd'], [0], {'odd': 7})


def test_lpp_partition_is_the_registered_hash_split():
    ids = [f'EN{i:03d}' for i in range(57, 57 + 49)]
    dev, confirm = bud.lpp_partition(ids)
    assert len(dev) == 9 and len(confirm) == 40 and not set(dev) & set(confirm)
    assert dev + confirm == sorted(ids, key=lambda i: sha(f'ssltang-v4-lpp-subject:{i}'))
    assert bud.lpp_partition(ids[::-1]) == (dev, confirm)
    with pytest.raises(ValueError, match='mix'):
        bud.lpp_partition(ids[:-1] + ['sub-EN999'])
    with pytest.raises(ValueError):
        bud.lpp_partition(ids[:9])


def test_lpp_section_budgets_are_nested_prefixes_of_the_non_test_sections():
    sections = list(range(1, 10))
    for t in sections:
        out = bud.lpp_section_budgets(t, sections)
        order = sorted([s for s in sections if s != t], key=lambda s: sha(f'ssltang-v4-lpp-section:{t}:{s}'))
        assert out == {'L3': order[:3], 'L5': order[:5], 'L8': order}
        assert t not in out['L8']
    assert bud.lpp_section_budgets('4', [str(s) for s in sections])['L5'] == [
        str(s) for s in bud.lpp_section_budgets(4, sections)['L5']]
    with pytest.raises(ValueError):
        bud.lpp_section_budgets(10, sections)
    with pytest.raises(ValueError):
        bud.lpp_section_budgets(1, sections[:8])


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert text.startswith(' ') and not add_special_tokens
        return [0, 1] if len(text.strip()) > 4 else [0]


def test_ledger_counts_words_with_the_20_word_causal_context():
    words = [dict(word=('longword' if i % 2 else 'w'), bin_index=i // 3) for i in range(30)]  # Bins 0..9.
    words.append(dict(word='tail', bin_index=None))  # Unbinned words supply context only.
    ledger = bud.Ledger()
    ledger.transcript({'r1': words}, [('r1', 9)], FakeTokenizer())
    assert ledger['transcript_words'] == 23  # Words 27-29 and their 20-word context back to word 7.
    expected_tokens = sum(2 if i % 2 else 1 for i in range(7, 30))
    assert ledger['transcript_tokens'] == expected_tokens
    ledger.transcript({'r1': words}, [('r1', 0), ('r1', 9), ('r1', 9)])
    assert ledger['transcript_words'] == 26 and ledger['transcript_tokens'] is None


def test_ledger_fields_validation_and_json_round_trip():
    ledger = bud.Ledger()
    assert ledger == dict(target_pairs_dense=0, target_pairs_stride4=0, target_bold_minutes=0.0, transcript_words=0,
                          transcript_tokens=None, references=[], reference_bold_minutes=0.0,
                          reference_labeled_pairs=0, reference_rule='none')
    ledger.target(np.int64(1010), 252, bud.bold_minutes([300, 330, 360]))
    assert ledger['target_bold_minutes'] == pytest.approx(33.0) and type(ledger['target_pairs_dense']) is int
    ledger.reference(['UTS03', 'UTS02'], 66.0, labeled_pairs=4000, rule='TS')
    assert ledger['references'] == ['UTS02', 'UTS03']
    again = bud.Ledger.from_dict(json.loads(json.dumps(ledger.to_dict())))
    assert again == ledger
    with pytest.raises(ValueError):
        bud.Ledger(colour='red')
    with pytest.raises(ValueError):
        bud.Ledger(target_pairs_dense=-1)
    with pytest.raises(ValueError):
        bud.Ledger(target_bold_minutes=float('nan'))
    with pytest.raises(ValueError):
        bud.Ledger(references=['UTS02'], reference_labeled_pairs=10, reference_rule='none')
    with pytest.raises(ValueError):
        bud.Ledger(reference_bold_minutes=5.0)
    with pytest.raises(ValueError):
        bud.Ledger(references=['UTS02'], reference_rule='XX')
