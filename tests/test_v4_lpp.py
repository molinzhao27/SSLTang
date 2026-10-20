"""v4 Module F (Le Petit Prince preparation): pure-logic checks on synthetic fixtures; no data files."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import lpp_prepare as lpp  # noqa: E402
from src.semantic.features import downsample_words, read_word_intervals, tr_centers  # noqa: E402

EN_IDS = ['sub-EN057', 'sub-EN058', 'sub-EN059', 'sub-EN061', 'sub-EN062', 'sub-EN063', 'sub-EN064', 'sub-EN065',
          'sub-EN067', 'sub-EN068', 'sub-EN069', 'sub-EN070', 'sub-EN072', 'sub-EN073', 'sub-EN074', 'sub-EN075',
          'sub-EN076', 'sub-EN077', 'sub-EN078', 'sub-EN079', 'sub-EN081', 'sub-EN082', 'sub-EN083', 'sub-EN084',
          'sub-EN086', 'sub-EN087', 'sub-EN088', 'sub-EN089', 'sub-EN091', 'sub-EN092', 'sub-EN093', 'sub-EN094',
          'sub-EN095', 'sub-EN096', 'sub-EN097', 'sub-EN098', 'sub-EN099', 'sub-EN100', 'sub-EN101', 'sub-EN103',
          'sub-EN104', 'sub-EN105', 'sub-EN106', 'sub-EN108', 'sub-EN109', 'sub-EN110', 'sub-EN113', 'sub-EN114',
          'sub-EN115']
PARSER = Path(__file__).resolve().parents[1] / 'reference/semantic-decoding/decoding/utils_ridge/textgrid.py'
needs_parser = pytest.mark.skipif(not PARSER.is_file(), reason='Tang/Huth TextGrid parser not checked out')


# ----------------------------------------------------------------------------------------- partition

def test_partition_is_the_registered_hash_order():
    ids = [f'sub-EN{i:03d}' for i in range(20)]
    dev, confirm = lpp.lpp_partition(list(reversed(ids)))
    digest = lambda s: hashlib.sha256(f'ssltang-v4-lpp-subject:{s}'.encode()).hexdigest()
    assert dev + confirm == sorted(ids, key=digest)
    assert len(dev) == 9 and not set(dev) & set(confirm) and set(dev) | set(confirm) == set(ids)
    assert lpp.lpp_partition(ids) == (dev, confirm)  # Input order is irrelevant.
    with pytest.raises(ValueError):
        lpp.lpp_partition(ids[:5] + ids[:5])


def test_partition_on_the_49_en_ids_is_pinned():
    dev, confirm = lpp.lpp_partition(EN_IDS)
    assert dev == ['sub-EN061', 'sub-EN083', 'sub-EN096', 'sub-EN101', 'sub-EN100', 'sub-EN105', 'sub-EN089',
                   'sub-EN063', 'sub-EN059']
    assert len(confirm) == 40


def test_partition_agrees_with_budgets_module_when_present():
    budgets = pytest.importorskip('src.budgets')
    dev, confirm = budgets.lpp_partition(EN_IDS)
    assert list(dev) == lpp.hash_order(EN_IDS, lpp.SUBJECT_PREFIX)[:9] and set(confirm) == set(EN_IDS) - set(dev)


def test_en_participants_reads_bids_tsv_with_bom():
    text = '﻿participant_id\tage\nsub-CN001\t18\nsub-EN058\t22\nsub-FR001\t20\nsub-EN057\tn/a\n'
    assert lpp.en_participants(text) == ['sub-EN057', 'sub-EN058']


def test_confirm_subjects_are_refused_without_flag_and_lock():
    dev, confirm = lpp.lpp_partition(EN_IDS)
    assert lpp.check_access(dev, dev, confirm, False, False) == 'DEV'
    assert lpp.check_access(dev, dev, confirm, True, True) == 'DEV'
    for flag, lock in ((False, False), (True, False), (False, True)):
        with pytest.raises(PermissionError):
            lpp.check_access(dev + confirm[:1], dev, confirm, flag, lock)
    assert lpp.check_access(confirm[:3], dev, confirm, True, True) == 'CONFIRM'
    with pytest.raises(ValueError):
        lpp.check_access(['sub-CN001'], dev, confirm, True, True)
    with pytest.raises(ValueError):  # A CONFIRM batch never carries (and so never rewrites) DEV subjects.
        lpp.check_access(dev[:1] + confirm[:3], dev, confirm, True, True)


# ------------------------------------------------------------------------------------------ selection

def listing(subjects, runs=None):
    runs = runs or (9, 11, 12, 13, 14, 19, 20, 21, 22)
    keys = [f'ds003643/{name}' for name in lpp.META]
    keys += [f'ds003643/annotation/EN/lppEN_section{s}.TextGrid' for s in lpp.SECTIONS]
    keys += [f'ds003643/stimuli/task-lppEN_section-{s}.wav' for s in lpp.SECTIONS]
    keys += ['ds003643/annotation/EN/lppEN_prosody.csv', 'ds003643/stimuli/task-lppFR_section_1.wav']
    for sub in subjects:
        keys += [f'ds003643/derivatives/{sub}/func/{sub}_task-lppEN_run-{r:02d}_space-MNIColin27_desc-preproc_bold.nii.gz'
                 for r in runs]
        keys.append(f'ds003643/{sub}/func/{sub}_task-lppEN_run-{runs[0]:02d}_echo-1_bold.nii.gz')
    keys += ['ds003643/derivatives/sub-FR001/func/sub-FR001_task-lppFR_run-01_space-MNIColin27_desc-preproc_bold.nii.gz']
    return [dict(key=k, size=1, etag='x') for k in keys]


def test_selection_downloads_only_requested_subjects():
    objects = listing(['sub-EN061', 'sub-EN057', 'sub-EN058'])
    selected = [x['key'] for x in lpp.select_objects(objects, ['sub-EN061'])]
    derivatives = [k for k in selected if '/derivatives/' in k]
    assert len(derivatives) == 9 and all('sub-EN061' in k for k in derivatives)
    assert not any('sub-EN057' in k or 'sub-EN058' in k or 'echo-' in k or 'FR' in k for k in selected)
    assert 'ds003643/annotation/EN/lppEN_word_information.csv' in selected
    assert sum(k.endswith('.wav') for k in selected) == 9 and 'ds003643/participants.tsv' in selected
    assert sorted(lpp.derivative_runs(objects)) == ['sub-EN057', 'sub-EN058', 'sub-EN061']
    with pytest.raises(ValueError):  # A subject without nine runs is refused.
        lpp.select_objects(listing(['sub-EN061'], runs=(1, 2, 3)), ['sub-EN061'])


def test_sections_follow_sorted_run_numbers_with_gaps():
    keys = [x['key'] for x in listing(['sub-EN061']) if '/derivatives/sub-EN061/' in x['key']]
    mapped = lpp.section_runs(list(reversed(keys)))
    assert [s for s, _, _ in mapped] == list(lpp.SECTIONS)
    assert [n for _, n, _ in mapped] == [9, 11, 12, 13, 14, 19, 20, 21, 22]
    lpp.verify_lengths(lpp.SECTION_TRS)
    with pytest.raises(ValueError):
        lpp.verify_lengths((298, 282) + lpp.SECTION_TRS[2:])
    with pytest.raises(ValueError):
        lpp.section_runs(keys[:8])


def test_nifti_dims_from_gzipped_header_prefix():
    header = bytearray(352)
    header[:4] = struct.pack('<i', 348)
    header[40:56] = struct.pack('<8h', 4, 64, 64, 33, 286, 1, 1, 1)
    blob = gzip.compress(bytes(header) + np.random.default_rng(0).bytes(1 << 16))
    assert len(blob) > 4096 and lpp.nifti_dims(blob[:4096]) == (64, 64, 33, 286)  # A prefix suffices.


# ---------------------------------------------------------------------------------------------- words

def word_csv(rows):
    head = ',word,lemma,onset,offset,logfreq,pos,section,top_down,bottom_up,left_corner\n'
    return head + ''.join(f'{i},{w},{w},{a},{b},1.0,X,{s},1,1,1\n' for i, (s, a, b, w) in enumerate(rows))


def synthetic_rows(n=lpp.N_WORDS):
    per = n // 9
    rows = []
    for k, s in enumerate(lpp.SECTIONS):
        count = per + (n - 9 * per if k == 8 else 0)
        rows += [(s, round(0.5 + 0.3 * i, 3), round(0.7 + 0.3 * i, 3), 'w') for i in range(count)]
    return rows


def test_read_words_keeps_literal_tokens_and_validates():
    rows = synthetic_rows()
    rows[0], rows[1], rows[2] = ('1', 0.1, 0.3, 'null'), ('1', 0.3, 0.4, ''), ('1', 0.4, 0.5, 'NA')
    words = lpp.read_words(word_csv(rows))
    assert [w for _, _, w in words['1'][:3]] == ['null', '', 'NA']  # No NaN coercion.
    assert sum(len(v) for v in words.values()) == lpp.N_WORDS
    assert [w for _, _, w in lpp.used_words(words['1'][:3])] == ['null', 'NA']
    with pytest.raises(ValueError):  # Wrong total.
        lpp.read_words(word_csv(rows[:-1]))
    bad = list(rows)
    bad[5] = ('1', bad[4][1], bad[4][2] + 0.5, 'w')  # Overlaps its successor.
    with pytest.raises(ValueError):
        lpp.read_words(word_csv(bad))
    late = list(rows)
    late[-1] = ('9', 900.0, 901.0, 'w')  # Past the 736 s run of section 9.
    with pytest.raises(ValueError):
        lpp.read_words(word_csv(late))


@needs_parser
def test_textgrid_round_trip_matches_filtered_words(tmp_path):
    intervals = [(0.113, 0.728, 'once'), (0.728, 0.919, 'didnt'), (0.919, 1.025, ''), (1.5, 2.25, 'three two five'),
                 (2.25, 2.5, 'na\\ive'), (3.0, 3.1, 'sp')]
    path = tmp_path / 'section-1.TextGrid'
    path.write_text(lpp.textgrid_text(intervals, 10.0))
    parsed = read_word_intervals(path)
    assert [(round(a, 6), round(b, 6), w) for a, b, w in parsed] == lpp.used_words(intervals)
    assert [w for _, _, w in parsed] == ['once', 'didnt', 'three two five', 'na\\ive']
    with pytest.raises(ValueError):
        lpp.textgrid_text([(0.0, 1.0, 'a'), (0.5, 1.5, 'b')], 10.0)
    with pytest.raises(ValueError):
        lpp.textgrid_text([(0.0, 11.0, 'a')], 10.0)


def test_tr_centers_follow_the_audio_onset_convention():
    run = dict(run_id='x', n_tr=5, tr_seconds=lpp.TR, tr_start_seconds=lpp.TR_START)
    centers = tr_centers(run, '.')
    assert np.allclose(centers, [1, 3, 5, 7, 9])
    intervals = [(0.1, 0.5, 'a'), (1.9, 2.3, 'b'), (8.5, 9.9, 'c')]  # Midpoints 0.3, 2.1, 9.2.
    _, valid = downsample_words(intervals, np.eye(3, dtype=np.float32), centers, lpp.TR)
    assert valid.tolist() == [True, True, False, False, True]  # Volume k covers audio [2k, 2k + 2).


def test_picture_mask_covers_section_one_first_80_seconds():
    mask = lpp.picture_mask(282, '1')
    assert mask[:40].all() and not mask[40:].any()  # TR 39 spans [78, 80); TR 40 starts at 80 s.
    assert not lpp.picture_mask(298, '2').any()
    assert lpp.picture_mask(282, '1', tr_start=0.0)[:41].sum() == 41  # Window bound is applied in audio time.


def test_encoding_is_one_call_per_section():
    class Recorder:
        def __init__(self):
            self.calls = []

        def encode(self, words, batch_size):
            self.calls.append(list(words))
            return np.zeros((len(words), 2), dtype=np.float32)

    encoder = Recorder()
    out = lpp.encode_sections(encoder, {'1': ['a', 'b'], '2': ['c'], '3': ['d', 'e', 'f']})
    assert encoder.calls == [['a', 'b'], ['c'], ['d', 'e', 'f']]  # No section sees another's words.
    assert {s: v.shape[0] for s, v in out.items()} == {'1': 2, '2': 1, '3': 3}


# ----------------------------------------------------------------------------------------------- mask

def test_block_rule_threshold():
    assert lpp.choose_block(150_000) == 1 and lpp.choose_block(150_001) == 2


def test_block_columns_and_pooling():
    grid = np.zeros((4, 4, 2), dtype=bool)
    grid[0, 0, 0] = grid[1, 1, 1] = grid[2, 0, 0] = grid[3, 3, 1] = grid[0, 3, 0] = True
    ijk = np.argwhere(grid)
    membership, coords = lpp.block_columns(ijk, 1)
    assert np.array_equal(coords, ijk) and np.array_equal(membership, np.arange(len(ijk)))
    membership, coords = lpp.block_columns(ijk, 2)
    assert coords.tolist() == [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
    assert membership.tolist() == [0, 1, 0, 2, 3]  # ijk order: 000, 031, 111, 200, 331.
    x = np.arange(15, dtype=np.float32).reshape(3, 5)
    pooled = lpp.pool(x, membership, len(coords))
    expected = np.stack([x[:, [0, 2]].mean(1), x[:, 1], x[:, 3], x[:, 4]], axis=1)
    assert pooled.dtype == np.float32 and np.allclose(pooled, expected)
    assert np.array_equal(lpp.pool(x, np.arange(5), 5), x)
    with pytest.raises(ValueError):
        lpp.pool(x, np.array([0, 0, 2, 2, 2]), 3)


# ------------------------------------------------------------------------------------- normalization

def test_zscore_run_normalizes_each_column_within_the_run():
    rng = np.random.default_rng(0)
    raw = (18000 + 40 * rng.standard_normal((300, 6)) + np.linspace(0, 20, 300)[:, None]).astype(np.float32)
    raw[:, 4] = 9000.0  # Constant column.
    raw[[3, 7], 5] = [np.nan, np.inf]
    z, floored = lpp.zscore_run(raw)
    assert z.dtype == np.float32 and floored == 1 and np.isfinite(z).all()
    live = z[:, [0, 1, 2, 3]].astype(np.float64)
    assert np.allclose(live.mean(0), 0, atol=1e-6) and np.allclose(live.std(0), 1, atol=1e-5)
    assert not z[:, 4].any() and z[3, 5] == 0 and z[7, 5] == 0  # Floored column and non-finite entries are 0.
    finite = np.delete(raw[:, 5].astype(np.float64), [3, 7])
    assert np.allclose(np.delete(z[:, 5], [3, 7]), (finite - finite.mean()) / finite.std(), atol=1e-5)
    assert np.allclose(lpp.zscore_run(raw[:, :2])[0], z[:, :2], rtol=0, atol=1e-6)  # No cross-column statistic.


def test_zscored_h5_is_a_fixed_point_of_module_b_load_bold(tmp_path):
    isp = pytest.importorskip('src.priors.isp')
    h5py = pytest.importorskip('h5py')
    rng = np.random.default_rng(1)
    raw = (20000 + 30 * rng.standard_normal((120, 5))).astype(np.float32)
    raw[:, 2] = 7.0
    z, _ = lpp.zscore_run(raw)
    (tmp_path / 'sub').mkdir()
    with h5py.File(tmp_path / 'sub/1.hf5', 'w') as handle:
        handle.create_dataset('data', data=z)
    again = isp.load_bold(tmp_path, dict(source_key='sub/1.hf5', n_tr=120))
    assert np.allclose(again, z, atol=1e-5)  # ISP/PCA and the decoder see the same per-run normalization.


def test_dvars_flags_a_volume_jump():
    z = lpp.zscore_run(np.random.default_rng(2).standard_normal((400, 500)))[0]
    base = lpp.dvars(z)
    assert base.shape == (399,) and abs(np.median(base) - np.sqrt(2)) < 0.05  # White noise: sqrt(2).
    z[200] += 5.0
    assert np.argmax(lpp.dvars(z)) in (199, 200)


# ------------------------------------------------------------------------------------- lock integrity

def locked_state(affine=None, dev=None):
    mask = np.zeros(lpp.GRID, dtype=bool)
    mask[30:33, 40:42, 20] = True
    membership, coords = lpp.block_columns(np.argwhere(mask), 1)
    affine = np.diag([-2.0, 2.0, 2.0, 1.0]) if affine is None else affine
    affine[:3, 3] = [72.0, -126.0, -72.0]
    return dict(mask=mask, membership=membership, coords=coords, affine=affine, block_size=np.int64(1),
                grid_shape=np.array(lpp.GRID), dev_subjects=np.array(dev or lpp.lpp_partition(EN_IDS)[0]))


CODE = {name: hashlib.sha256(name.encode()).hexdigest() for name in lpp.CODE_FILES}


def test_lock_requires_partition_and_mask_digest():
    dev, confirm = lpp.lpp_partition(EN_IDS)
    digest = lpp.state_digest(locked_state())
    good = dict(lpp_partition=dict(dev=dev, confirm=confirm), lpp_mask_sha256=digest, lpp_code_sha256=CODE)
    assert lpp.check_lock(good, dev, confirm, CODE) == digest
    assert lpp.check_lock(dict(good, lpp_dataset_doi=lpp.DOI), dev, confirm, CODE) == digest
    bad = [dict(lpp_partition=good['lpp_partition'], lpp_code_sha256=CODE),  # No mask digest: no longer optional.
           dict(lpp_mask_sha256=digest, lpp_code_sha256=CODE),
           dict(good, lpp_partition=dict(dev=dev[::-1], confirm=confirm)),
           dict(good, lpp_partition=dict(dev=dev, confirm=confirm[:-1])),
           dict(good, lpp_mask_sha256='0' * 63),
           dict(good, lpp_dataset_doi='doi:10.18112/openneuro.ds003643.v2.0.8')]
    for lock in bad:
        with pytest.raises(RuntimeError):
            lpp.check_lock(lock, dev, confirm, CODE)


def test_lock_pins_the_preparation_code():
    dev, confirm = lpp.lpp_partition(EN_IDS)
    good = dict(lpp_partition=dict(dev=dev, confirm=confirm), lpp_mask_sha256='a' * 64, lpp_code_sha256=CODE)
    edited = dict(CODE, **{'scripts/lpp_prepare.py': '0' * 64})  # e.g. TR_START or STD_FLOOR changed after the lock.
    for lock in (dict(good, lpp_code_sha256=edited), {k: v for k, v in good.items() if k != 'lpp_code_sha256'},
                 dict(good, lpp_code_sha256='0' * 64), dict(good, lpp_code_sha256=dict(list(CODE.items())[1:]))):
        with pytest.raises(RuntimeError):
            lpp.check_lock(lock, dev, confirm, CODE)
    with pytest.raises(RuntimeError, match='lpp_prepare'):
        lpp.check_lock(good, dev, confirm, edited)
    assert set(lpp.CODE_FILES) >= {'scripts/lpp_prepare.py', 'src/semantic/features.py', 'src/data/openneuro.py'}


def test_dev_record_must_come_from_the_locked_code_and_versions():
    versions = dict(numpy='2.1.0', torch='2.5.1')
    record = dict(inputs={}, outputs={}, output_content={}, code=CODE, versions=versions)
    assert lpp.check_dev_record(record, CODE, versions) is None
    with pytest.raises(RuntimeError):  # DEV outputs written by other code than the locked code.
        lpp.check_dev_record(dict(record, code=dict(CODE, **{'src/data/openneuro.py': '1' * 64})), CODE, versions)
    with pytest.raises(RuntimeError):  # Written before content digests were recorded.
        lpp.check_dev_record({k: v for k, v in record.items() if k != 'output_content'}, CODE, versions)
    upgraded = dict(versions, numpy='2.2.0')
    with pytest.raises(RuntimeError):
        lpp.check_dev_record(record, CODE, upgraded)
    assert lpp.check_dev_record(record, CODE, upgraded, 'container rebuilt') == dict(
        changed={'numpy': ['2.1.0', '2.2.0']}, reason='container rebuilt')
    with pytest.raises(RuntimeError):  # A library missing on either side is a change too.
        lpp.check_versions(dict(numpy='2.1.0'), versions)


def test_confirm_needs_the_committed_lock_sha():
    sha = 'c' * 64
    good = dict(tracked=True, clean=True, pushed=True, commit='d' * 40)
    lpp.check_lock_commit(sha, sha, good)
    lpp.check_lock_commit(sha, sha, None)  # No git: the --lock-sha256 match is the record.
    for given, state in ((None, good), ('e' * 64, good), (sha, dict(good, pushed=False)),
                         (sha, dict(good, clean=False)), (sha, dict(good, tracked=False, clean=False, pushed=False))):
        with pytest.raises(PermissionError):
            lpp.check_lock_commit(sha, given, state)


@pytest.mark.skipif(shutil.which('git') is None, reason='git not installed')
def test_lock_git_state_tracks_commit_cleanliness_and_push(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()

    def git(*args):
        subprocess.run(['git', '-C', str(repo), '-c', 'user.email=t@t', '-c', 'user.name=t', '-c',
                        'commit.gpgsign=false', *args], check=True, capture_output=True)
    git('init', '-q')
    lock = repo / 'v4_lock.json'
    lock.write_text('{}')
    assert lpp.lock_git_state(lock) == dict(tracked=False, clean=False, pushed=False, commit=None)  # Draft.
    git('add', 'v4_lock.json')
    git('commit', '-qm', 'lock')
    state = lpp.lock_git_state(lock)
    assert state['tracked'] and state['clean'] and not state['pushed'] and len(state['commit']) == 40
    git('update-ref', 'refs/remotes/origin/main', 'HEAD')
    assert lpp.lock_git_state(lock)['pushed']
    lock.write_text('{"edited": 1}')
    assert not lpp.lock_git_state(lock)['clean']
    plain = tmp_path / 'plain'
    plain.mkdir()
    if subprocess.run(['git', '-C', str(plain), 'rev-parse'], capture_output=True).returncode:
        assert lpp.lock_git_state(plain / 'v4_lock.json') is None


def test_mask_digest_is_content_based_and_sensitive(tmp_path):
    state = locked_state()
    digest = lpp.state_digest(state)
    for name in ('a.npz', 'b.npz'):
        np.savez(tmp_path / name, **state)
        with np.load(tmp_path / name) as handle:
            assert lpp.state_digest(handle) == digest  # Independent of the zip container bytes.
    nudged = dict(state, affine=state['affine'] + 1e-7)
    assert lpp.state_digest(nudged) == digest
    changed = [dict(state, mask=np.roll(state['mask'], 1, axis=0)), dict(state, coords=state['coords'] + 1),
               dict(state, membership=state['membership'][::-1].copy()), dict(state, block_size=np.int64(2))]
    shifted = state['affine'].copy()
    shifted[0, 3] += 2.0
    changed.append(dict(state, affine=shifted))
    assert all(lpp.state_digest(c) != digest for c in changed)


def test_confirm_mask_state_and_dev_mask_rewrite_rules():
    dev, _ = lpp.lpp_partition(EN_IDS)
    state = locked_state()
    digest = lpp.state_digest(state)
    lpp.check_mask_state(state, digest, dev)
    with pytest.raises(RuntimeError):  # A rebuilt (different) mask on disk.
        lpp.check_mask_state(dict(state, mask=~state['mask']), digest, dev)
    with pytest.raises(RuntimeError):
        lpp.check_mask_state(state, digest, dev[::-1])
    assert lpp.mask_action('a' * 64, None, None) == 'write'
    assert lpp.mask_action('a' * 64, 'b' * 64, None) == 'write'  # Before the lock a DEV rebuild may change it.
    assert lpp.mask_action('a' * 64, 'a' * 64, 'a' * 64) == 'keep'
    assert lpp.mask_action('a' * 64, 'b' * 64, 'a' * 64) == 'write'  # Restores the locked mask.
    with pytest.raises(RuntimeError):  # After the lock a different rebuilt mask is never written.
        lpp.mask_action('b' * 64, 'a' * 64, 'a' * 64)


def test_affine_must_match_the_mask_grid():
    affine = locked_state()['affine']
    lpp.check_affine(affine + 1e-6, affine, 'run')
    origin = affine.copy()
    origin[1, 3] += 2.0
    flipped = affine.copy()
    flipped[0, 0] *= -1
    for other in (origin, flipped):
        with pytest.raises(ValueError):
            lpp.check_affine(other, affine, 'run')


def test_inputs_must_match_the_dev_record():
    words, wav = lpp.WORDS_KEY, 'ds003643/stimuli/task-lppEN_section-1.wav'
    derivative = 'ds003643/derivatives/sub-EN061/func/sub-EN061_task-lppEN_run-09_space-MNIColin27_desc-preproc_bold.nii.gz'
    recorded = {words: dict(sha256='a'), wav: dict(sha256='b'), derivative: dict(sha256='c')}
    lpp.check_inputs({words: dict(sha256='a'), wav: dict(sha256='b')}, recorded)  # DEV derivatives may be absent.
    with pytest.raises(RuntimeError):
        lpp.check_inputs({words: dict(sha256='x'), wav: dict(sha256='b')}, recorded)
    with pytest.raises(RuntimeError):
        lpp.check_inputs({words: dict(sha256='a')}, recorded)
    lpp.check_inputs({words: dict(sha256='a')}, recorded, complete=False)
    with pytest.raises(RuntimeError):
        lpp.check_inputs({derivative: dict(sha256='y')}, recorded, complete=False)


def test_frozen_stimuli_cannot_change(tmp_path):
    path = tmp_path / 'section-1.TextGrid'
    assert lpp.write_frozen(path, b'one', frozen=True)  # A missing file may be created.
    assert not lpp.write_frozen(path, b'one', frozen=True)
    with pytest.raises(RuntimeError):
        lpp.write_frozen(path, b'two', frozen=True)
    assert lpp.write_frozen(path, b'two', frozen=False) and path.read_bytes() == b'two'


def test_confirm_batches_get_distinct_records():
    _, confirm = lpp.lpp_partition(EN_IDS)
    first, second = lpp.record_suffix('CONFIRM', confirm[:20]), lpp.record_suffix('CONFIRM', confirm[20:])
    assert first != second and first.startswith('confirm_') and len(first) == len('confirm_') + 12
    assert lpp.record_suffix('CONFIRM', confirm[:20][::-1]) == first and lpp.record_suffix('DEV', []) == 'dev'
    assert lpp.record_names('dev') == ('download_ds003643_dev.json', 'prepare_provenance.json')
    rerun = lpp.record_suffix('DEV', [], '20261005T120000Z')
    assert rerun == 'dev_rerun_20261005T120000Z'  # A post-lock DEV re-run never replaces the DEV record.
    assert lpp.record_names(rerun)[1] == 'prepare_provenance_dev_rerun_20261005T120000Z.json'
    assert lpp.record_names(first)[1] != lpp.record_names(second)[1]


def test_array_digest_and_post_lock_output_check():
    y = np.arange(12, dtype=np.float32).reshape(3, 4)
    digest = lpp.array_digest(y)
    assert digest == lpp.array_digest(np.asfortranarray(y)) == lpp.array_digest(y.copy())
    assert len({digest, lpp.array_digest(y.astype(np.float64)), lpp.array_digest(y.reshape(4, 3)),
                lpp.array_digest(y + np.float32(1e-6))}) == 4
    lpp.check_output('raw/sub-EN061/1.hf5', 'x', None)  # Before the lock anything may be written.
    lpp.check_output('raw/sub-EN061/1.hf5', digest, {'raw/sub-EN061/1.hf5': digest})
    for recorded in ({'raw/sub-EN061/1.hf5': 'other'}, {}):
        with pytest.raises(RuntimeError):
            lpp.check_output('raw/sub-EN061/1.hf5', digest, recorded)


def test_h5_digest_reads_the_data_array(tmp_path):
    h5py = pytest.importorskip('h5py')
    y = np.random.default_rng(3).standard_normal((5, 7)).astype(np.float32)
    for name in ('a.hf5', 'b.hf5'):
        with h5py.File(tmp_path / name, 'w') as handle:
            handle.create_dataset('data', data=y)
            handle.attrs['name'] = name
        assert lpp.h5_digest(tmp_path / name) == lpp.array_digest(y)  # Independent of attributes and file bytes.
    assert lpp.h5_digest(tmp_path / 'missing.hf5') is None


def test_post_lock_h5_write_must_reproduce_the_dev_record(tmp_path):
    h5py = pytest.importorskip('h5py')
    y = np.random.default_rng(4).standard_normal((6, 3)).astype(np.float32)
    path, rel = tmp_path / '1.hf5', 'raw/sub-EN061/1.hf5'
    recorded = {rel: lpp.write_h5(path, y, dict(section='1'), rel)}  # DEV, before the lock.
    stamp = path.stat().st_mtime_ns
    assert lpp.write_h5(path, y, dict(section='1'), rel, recorded) == recorded[rel]
    assert path.stat().st_mtime_ns == stamp  # Equal content: the locked file is kept.
    with pytest.raises(RuntimeError):  # e.g. a changed z-score rule or library after the lock.
        lpp.write_h5(path, y + 1, dict(section='1'), rel, recorded)
    with pytest.raises(RuntimeError):  # Not in the DEV record.
        lpp.write_h5(tmp_path / '2.hf5', y, {}, 'raw/sub-EN061/2.hf5', recorded)
    assert path.stat().st_mtime_ns == stamp and not (tmp_path / '2.hf5').exists()
    path.unlink()
    lpp.write_h5(path, y, dict(section='1'), rel, recorded)  # A missing locked file is restored.
    with h5py.File(path, 'r') as handle:
        assert np.array_equal(handle['data'][()], y) and handle.attrs['section'] == '1'


@needs_parser
def test_targets_after_the_lock_must_reproduce_the_dev_record(tmp_path, monkeypatch):
    import src.semantic.features as features

    class Encoder:
        scale = 1.0

        def __init__(self, *args):
            pass

        def encode(self, words, batch_size):
            return Encoder.scale * np.arange(2 * len(words), dtype=np.float32).reshape(-1, 2)

    monkeypatch.setattr(features, 'FrozenGPT2', Encoder)
    textgrids = {}
    for s, n_tr in zip(lpp.SECTIONS, lpp.SECTION_TRS):
        textgrids[s] = tmp_path / f'section-{s}.TextGrid'
        textgrids[s].write_text(lpp.textgrid_text([(1.0, 1.5, 'a'), (3.2, 3.9, 'b'), (10.0, 10.4, 'c')], n_tr * lpp.TR))
    out = tmp_path / 'lpp'
    _, first = lpp.build_targets(textgrids, out, 'cpu', 4)
    recorded = {k: v for t in first.values() for k, v in t['content_sha256'].items()}
    assert len(recorded) == 18 and all(t['recomputed'] for t in first.values())
    _, again = lpp.build_targets(textgrids, out, 'cpu', 4, recorded)
    assert not any(t['recomputed'] for t in again.values())  # Cached and equal to the DEV record.
    target = Path(first['3']['target_path'])
    np.save(target, np.zeros_like(np.load(target)))  # A tampered cache file is recomputed and restored.
    _, restored = lpp.build_targets(textgrids, out, 'cpu', 4, recorded)
    assert restored['3']['recomputed'] and lpp.array_digest(np.load(target)) in recorded.values()
    target.unlink()
    Encoder.scale = 2.0  # Recomputed targets that differ (library, device) are refused before saving.
    with pytest.raises(RuntimeError, match='differs from the DEV record'):
        lpp.build_targets(textgrids, out, 'cpu', 4, recorded)
    assert not target.exists()
    Encoder.scale = 1.0
    textgrids['1'].write_text(lpp.textgrid_text([(1.0, 1.6, 'a')], lpp.SECTION_TRS[0] * lpp.TR))
    with pytest.raises(RuntimeError, match='not in the DEV record'):  # Changed stimulus, so a new cache key.
        lpp.build_targets(textgrids, out, 'cpu', 4, recorded)


def test_fetch_refetches_a_stale_same_size_file(tmp_path, monkeypatch):
    from src.data import openneuro
    good, stale = b'new release bytes', b'old release bytes'
    served = [stale, good]

    def fake(item, root):
        path = Path(root) / item['key']
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(served.pop(0))
        return path

    monkeypatch.setattr(openneuro, 'download', fake)
    item = dict(key='ds003643/README', size=len(good), etag=hashlib.md5(good).hexdigest())
    record = lpp.fetch(item, tmp_path)
    assert record['sha256'] == hashlib.sha256(good).hexdigest() and record['md5'] == item['etag'] and not served
    served[:] = [stale, stale]
    (tmp_path / item['key']).unlink()
    with pytest.raises(RuntimeError):
        lpp.fetch(item, tmp_path)
    served[:] = [stale]
    (tmp_path / item['key']).unlink(missing_ok=True)
    assert lpp.fetch(dict(item, etag='abc-3'), tmp_path)['sha256'] == hashlib.sha256(stale).hexdigest()  # Multipart.


def test_motion_record_reports_files_and_statements():
    keys = [f'ds003643/derivatives/sub-EN061/func/sub-EN061_run-{r}_bold.nii.gz' for r in (9, 11)]
    keys += ['ds003643/sub-EN083/func/sub-EN083_run-09_motion.1D', 'ds003643/sub-EN096/anat/sub-EN096_T1w.nii.gz']
    readme = ('Participants were told not to move.\nThe first 4 volumes were excluded. ME-ICA was used to\n'
              'denoise data for motion. Images were resampled.')
    record = lpp.motion_record(keys, ['sub-EN061', 'sub-EN083'], readme, '{"age": {}, "sex": {}}')
    assert record['motion_files'] == {'sub-EN061': [], 'sub-EN083': ['ds003643/sub-EN083/func/sub-EN083_run-09_motion.1D']}
    assert record['motion_parameters_released'] and record['participants_columns'] == ['age', 'sex']
    assert record['readme_statements'] == ['Participants were told not to move.', 'The first 4 volumes were excluded.',
                                           'ME-ICA was used to denoise data for motion.']


# --------------------------------------------------------------------------------------- main() wiring

class Reached(Exception):
    """fetch_all was called: every pre-download check passed."""


@pytest.fixture
def stub(tmp_path, monkeypatch):
    """main() on a synthetic listing with a DEV record, mask.npz and lock in tmp_path; fetch_all raises Reached."""
    from src.data import openneuro
    from src.runtime import file_sha256, save_json
    out = tmp_path / 'lpp'
    meta = out / 'openneuro/ds003643'
    meta.mkdir(parents=True)
    rows = ''.join(f'{s}\t20\n' for s in EN_IDS + ['sub-FR001'])
    (meta / 'participants.tsv').write_text('participant_id\tage\n' + rows)
    (meta / 'dataset_description.json').write_text(json.dumps(dict(DatasetDOI=lpp.DOI)))
    dev, confirm = lpp.lpp_partition(EN_IDS)
    s = SimpleNamespace(out=out, dev=dev, confirm=confirm, lock=tmp_path / 'v4_lock.json', calls=[], returns=None,
                        versions=dict(numpy='1.0'), git=None)

    def fetch(item, root):
        return dict(size=item['size'], etag=item['etag'], sha256=file_sha256(Path(root) / item['key']), md5='')

    def fetch_all(selected, root, workers):
        s.calls.append([x['key'] for x in selected])
        if s.returns is None:
            raise Reached
        return s.returns(selected)

    monkeypatch.setattr(openneuro, 'discover', lambda dataset: listing(EN_IDS))
    monkeypatch.setattr(lpp, 'fetch', fetch)
    monkeypatch.setattr(lpp, 'fetch_all', fetch_all)
    monkeypatch.setattr(lpp, 'LOCK', s.lock)
    monkeypatch.setattr(lpp, 'library_versions', lambda: dict(s.versions))
    monkeypatch.setattr(lpp, 'lock_git_state', lambda path: s.git)
    state = locked_state(dev=dev)
    np.savez(out / 'mask.npz', **state)
    s.record = dict(inputs={k: fetch(dict(key=k, size=1, etag='x'), out / 'openneuro')
                            for k in ('ds003643/participants.tsv', 'ds003643/dataset_description.json')},
                    outputs={}, output_content={}, code=lpp.code_fingerprint(), versions=dict(s.versions))
    s.save_record = lambda: save_json(s.record, out / 'prepare_provenance.json')
    s.save_record()

    def write_lock(**changes):
        lock = dict(dict(lpp_partition=dict(dev=dev, confirm=confirm), lpp_mask_sha256=lpp.state_digest(state),
                         lpp_code_sha256=lpp.code_fingerprint()), **changes)
        s.lock.write_text(json.dumps({k: v for k, v in lock.items() if v is not None}))
        return file_sha256(s.lock)

    s.write_lock = write_lock
    s.run = lambda *args: lpp.main(['--out', str(out), *args])
    s.download = lambda: {p.name: json.loads(p.read_text()) for p in (out / 'openneuro').glob('download_*.json')}
    return s


@needs_parser
def test_main_dev_before_the_lock_downloads_dev_subjects_only(stub):
    with pytest.raises(Reached):
        stub.run()
    derivatives = [k for k in stub.calls[0] if '/derivatives/' in k]
    assert len(derivatives) == 81 and {k.split('/')[2] for k in derivatives} == set(stub.dev)
    record = stub.download()['download_ds003643_dev.json']
    assert record['provenance'] == 'prepare_provenance.json' and not record['lock']['present']
    with pytest.raises(PermissionError):  # CONFIRM without the lock.
        stub.run('--confirm')
    assert len(stub.calls) == 1


@needs_parser
def test_main_confirm_checks_lock_code_mask_and_commit_before_download(stub):
    batch = ['--confirm', '--subjects', *stub.confirm[:20]]
    sha = stub.write_lock()
    with pytest.raises(Reached):
        stub.run(*batch, '--lock-sha256', sha)
    assert not any(f'/{d}/' in k for k in stub.calls[0] for d in stub.dev)
    record = next(iter(stub.download().values()))
    assert record['lock']['file_sha256'] == sha and record['code'] == lpp.code_fingerprint()
    other = dict(lpp.code_fingerprint(), **{'scripts/lpp_prepare.py': '0' * 64})
    refused = [  # (lock changes, extra args, record/state edits, exception)
        (dict(lpp_mask_sha256=None), (), None, RuntimeError),  # Lock lacks the mask digest.
        (dict(lpp_partition=dict(dev=stub.dev[::-1], confirm=stub.confirm)), (), None, RuntimeError),
        (dict(lpp_code_sha256=other), (), None, RuntimeError),  # Lock pins other preparation code.
        ({}, ('--lock-sha256', 'f' * 64), None, PermissionError),  # Not the committed lock.
        ({}, (), 'git', PermissionError),  # Lock not pushed.
        ({}, (), 'mask', RuntimeError),  # mask.npz differs from the locked digest.
        ({}, (), 'code', RuntimeError),  # DEV record written by other code.
        ({}, (), 'versions', RuntimeError),  # Library versions changed without a logged deviation.
    ]
    for changes, args, edit, error in refused:
        sha = stub.write_lock(**changes)
        stub.git, versions = None, dict(stub.versions)
        if edit == 'git':
            stub.git = dict(tracked=True, clean=True, pushed=False, commit='a' * 40)
        if edit == 'mask':
            np.savez(stub.out / 'mask.npz', **dict(locked_state(dev=stub.dev), mask=np.zeros(lpp.GRID, dtype=bool)))
        if edit == 'code':
            stub.record['code'] = other
            stub.save_record()
        if edit == 'versions':
            stub.versions['numpy'] = '2.0'
        with pytest.raises(error):
            stub.run(*batch, *(args or ('--lock-sha256', sha)))
        assert len(stub.calls) == 1, (changes, args, edit)  # Refused before fetch_all.
        np.savez(stub.out / 'mask.npz', **locked_state(dev=stub.dev))
        stub.record['code'] = lpp.code_fingerprint()
        stub.save_record()
        stub.versions = versions
    with pytest.raises(PermissionError):  # Without --lock-sha256.
        stub.run(*batch)
    stub.versions['numpy'] = '2.0'
    sha = stub.write_lock()
    with pytest.raises(Reached):  # A logged version deviation proceeds and is recorded.
        stub.run(*batch, '--lock-sha256', sha, '--version-deviation', 'container rebuilt')
    deviation = [r['lock']['version_deviation'] for r in stub.download().values()][0]
    assert deviation == dict(changed={'numpy': ['1.0', '2.0']}, reason='container rebuilt')


@needs_parser
def test_main_confirm_batches_write_distinct_records(stub):
    sha = stub.write_lock()
    for batch in (stub.confirm[:20], stub.confirm[20:]):
        with pytest.raises(Reached):
            stub.run('--confirm', '--lock-sha256', sha, '--subjects', *batch)
    records = stub.download()
    assert len(records) == 2 and len({r['provenance'] for r in records.values()}) == 2
    assert all(r['provenance'].startswith('prepare_provenance_confirm_') for r in records.values())
    with pytest.raises(ValueError):  # Mixed DEV/CONFIRM batch.
        stub.run('--confirm', '--lock-sha256', sha, '--subjects', stub.dev[0], *stub.confirm[:2])
    assert len(stub.calls) == 2


@needs_parser
def test_main_dev_after_the_lock_checks_inputs_and_keeps_the_dev_record(stub):
    stub.record['inputs'][lpp.WORDS_KEY] = dict(sha256='a' * 64)
    stub.save_record()
    stub.write_lock()
    stub.returns = lambda selected: {x['key']: dict(stub.record['inputs'].get(x['key'], dict(sha256='n')))
                                     for x in selected}
    with pytest.raises(FileNotFoundError):  # Inputs equal the DEV record: it proceeds to read the words.
        stub.run()
    record = next(iter(stub.download().values()))
    assert record['provenance'].startswith('prepare_provenance_dev_rerun_') and record['lock']['present']
    changed = stub.returns
    stub.returns = lambda selected: dict(changed(selected), **{lpp.WORDS_KEY: dict(sha256='b' * 64)})
    with pytest.raises(RuntimeError, match='Inputs differ'):  # A changed shared input after download.
        stub.run()
    stub.record['inputs']['ds003643/participants.tsv']['sha256'] = '0' * 64
    stub.save_record()
    calls = len(stub.calls)
    with pytest.raises(RuntimeError, match='Inputs differ'):  # A changed early input, before any download.
        stub.run()
    assert len(stub.calls) == calls
