#!/usr/bin/env python3
"""v4 Module F: Le Petit Prince English (OpenNeuro ds003643) preparation; DEV subjects only before the lock.

Stages (no decoder is fit and no decoding score is computed):
  partition  EN participant_ids from participants.tsv, ordered by sha256('ssltang-v4-lpp-subject:' + id);
             the first 9 are DEV, the remaining 40 CONFIRM. CONFIRM subjects are refused unless --confirm
             is given AND experiments/v4_lock.json exists.
  download   selected derivatives (9 runs per subject), annotation/EN (word CSV + section TextGrids),
             the 9 EN section WAVs, participants and dataset metadata from the registered v2.0.7 release
             (DatasetDOI checked first); S3 size-checked, and MD5-checked against single-part ETags so a
             stale same-size local file is fetched again.
  verify     G4(a): sections are the derivative runs in ascending run number, with TR counts
             282/298/340/303/265/343/325/292/368; raw echo-1 headers carry 4 more volumes than each
             derivative (the excluded T1-equilibration volumes); the word CSV has 15,376 words, every
             word inside its section's run, and the official section TextGrids span n_tr * 2 s.
             Motion (registered decision 3, descriptor route): the release has no motion parameters, so
             the README/participants.json statements, any motion files in the listing and per-run DVARS
             of the normalized columns are recorded; no subject is excluded by this script.
  mask       nilearn MNI152 gray-matter mask resampled (nearest) to the 73x90x74 derivative grid,
             intersected with voxels finite and non-constant in every run of every DEV subject. Above
             150,000 voxels, columns become means over 2x2x2 voxel blocks (locked rule).
  normalize  every run's columns are z-scored over that run's own TRs (float64, finite values only;
             SD <= 1e-6 or non-finite -> 0), as in the ds003020 H5s; no detrending, no cross-run statistic.
  write      data/lpp/raw/<sub>/<section>.hf5 ('data' [n_tr, V] float32), a ds003020-style manifest
             data/lpp/semantic/<sub>/semantic_manifest.json, per-section word TextGrids and picture masks.
  targets    pinned GPT-2 layer-9 word features (src.semantic.features), one encoder call per section,
             so the 20-word context resets at every section start (R6); centered-TR midpoint averaging.

Lock integrity. Once experiments/v4_lock.json exists it must hold 'lpp_partition' ({'dev': [...],
'confirm': [...]} in hash order, equal to the computed partition), 'lpp_mask_sha256' (the mask content
digest, prepare_provenance.json mask.content_sha256) and 'lpp_code_sha256' (prepare_provenance.json code:
{file: sha256} over CODE_FILES, equal to the running code). Before anything is downloaded: the lock entries,
the DEV record (written by the same code; library versions equal unless --version-deviation REASON is
given and logged) and the shared inputs recorded there are checked; a CONFIRM run also needs --lock-sha256
equal to the lock file's sha256, the lock committed, clean and on origin/main (when git is available), and
mask.npz equal to the locked digest. After the lock a DEV re-run is verification only: mask.npz, stimuli,
manifests, the H5 'data' arrays and the GPT-2 target arrays must reproduce the DEV record (content digests)
before any file is replaced, and its record is prepare_provenance_dev_rerun_<utc>.json (the DEV record is
never overwritten). CONFIRM batches contain CONFIRM subjects only, verify the mask affine for every run and
write one record per batch (prepare_provenance_confirm_<batch>.json).

Loader. The layout mirrors ds003020 (H5 'data' arrays, manifest runs), but src.targets.TargetIndex /
src.windows.NeuralWindows reject it (they accept only ds003020 runs with neural_path and a fit_run_ids
normalization file); the LPP runner needs its own index and delay-row cache.

Audio-onset convention. The descriptor (Li et al. 2022) states a trigger at the start of each section, 8 s
(4 TRs) before audio onset, and that the first 4 volumes of every run were excluded. Raw echo-1 headers
have exactly 4 more volumes than the derivatives, and the official section TextGrids span exactly
n_tr * 2 s, so derivative volume k covers audio time [2k, 2k + 2) s. Following the ds003020 convention
(TR centers relative to stimulus onset, FEATURE_CONTRACT.md), tr_start_seconds = 1.0.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import urllib.request
import zlib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import file_sha256, save_json

ROOT = Path(__file__).resolve().parents[1]
DATASET = 'ds003643'
BASE = 'https://s3.amazonaws.com/openneuro.org'
SECTIONS = tuple(str(k) for k in range(1, 10))  # Section ids: '1'..'9' (also the manifest story/session).
SECTION_TRS = (282, 298, 340, 303, 265, 343, 325, 292, 368)
N_EN, N_DEV, N_WORDS = 49, 9, 15376
GRID, VOXEL_MM = (73, 90, 74), 2.0
TR, TR_START, DISCARDED = 2.0, 1.0, 4
PICTURE_SECTION, PICTURE_SECONDS = '1', (0.0, 80.0)
MAX_VOXELS, BLOCK = 150_000, 2
SUBJECT_PREFIX = 'ssltang-v4-lpp-subject:'
DOI = 'doi:10.18112/openneuro.ds003643.v2.0.7'  # Registered release (FUTURE_WORK_V4 §4.2).
LOCK = ROOT / 'experiments/v4_lock.json'
LOCK_KEYS = ('lpp_partition', 'lpp_mask_sha256', 'lpp_code_sha256')
# Code that determines the prepared data (PROTOCOL_V4 §3 fingerprint; the textgrid parser is the pinned checkout).
CODE_FILES = ('scripts/lpp_prepare.py', 'src/budgets.py', 'src/data/openneuro.py', 'src/semantic/features.py',
              'reference/semantic-decoding/decoding/utils_ridge/textgrid.py')
DEV_RECORD_KEYS = ('inputs', 'outputs', 'output_content', 'code', 'versions')
STD_FLOOR = 1e-6  # As src.priors.isp.load_bold.
NORMALIZATION = dict(
    bold_normalization='per-run z-score', detrend=None,
    rule="per run and column: (x - mean) / SD over that run's TRs, float64, finite values only; "
         "SD <= 1e-6 or non-finite values -> 0; applied after masking and block averaging")
LOADER_NOTE = ('Not loadable by src.targets.TargetIndex / src.windows.NeuralWindows (ds003020 runs with '
               'neural_path and fit_run_ids only); use an LPP index and delay-row cache.')
MOTION_FILE = re.compile(r'motion|confound|movement|framewise|_mcf|\.1D$|\.par$', re.I)
META = ('participants.tsv', 'participants.json', 'dataset_description.json', 'README', 'CHANGES',
        'code/lppEN_presentation.py', 'annotation/EN/lppEN_word_information.csv')
WORDS_KEY = f'{DATASET}/annotation/EN/lppEN_word_information.csv'
DERIVATIVE = re.compile(rf'^{DATASET}/derivatives/(sub-EN\d+)/func/\1_task-lppEN_run-(\d+)'
                        r'_space-MNIColin27_desc-preproc_bold\.nii\.gz$')
GM_MASK = dict(resolution=2, threshold=0.2, n_iter=2)
TARGET_METADATA = dict(model=None, revision=None, layer=9, context_words=20, context_reset='section_start',
                       token_pool='last subword of current word', frozen=True,
                       downsampling='word midpoint mean in centered TR interval', target_lag_seconds=0,
                       word_source=WORDS_KEY, exact_tang_replication=False)


# ----------------------------------------------------------------------------------------- partition

def hash_order(items, prefix):
    """Ascending hexadecimal sha256(prefix + item) order (PROTOCOL_V4 §2)."""
    return sorted(items, key=lambda item: hashlib.sha256(f'{prefix}{item}'.encode()).hexdigest())


def lpp_partition(participant_ids):
    """(dev[9], confirm[rest]) in hash order; cross-checked against src.budgets when it exists.

    ``participant_ids`` are participants.tsv ``participant_id`` values used verbatim ('sub-EN057').
    """
    ids = list(participant_ids)
    if len(set(ids)) != len(ids) or len(ids) <= N_DEV:
        raise ValueError('Participant ids must be unique and more than the DEV count')
    ordered = hash_order(ids, SUBJECT_PREFIX)
    dev, confirm = ordered[:N_DEV], ordered[N_DEV:]
    try:
        from src.budgets import lpp_partition as shared
    except ImportError:
        return dev, confirm
    other = shared(ids)
    if list(other[0]) != dev or set(other[1]) != set(confirm):
        raise RuntimeError('src.budgets.lpp_partition disagrees with the registered hash rule')
    return dev, confirm


def partition_source():
    try:
        from src.budgets import lpp_partition  # noqa: F401
        return 'src.budgets.lpp_partition (cross-checked against the inline rule)'
    except ImportError:
        return 'inline hash rule in scripts/lpp_prepare.py (src.budgets not present)'


def en_participants(tsv_text):
    """Sorted EN participant_ids from participants.tsv text (BOM tolerated)."""
    rows = csv.DictReader(io.StringIO(tsv_text.lstrip('﻿')), delimiter='\t')
    return sorted(r['participant_id'].strip() for r in rows if r['participant_id'].strip().startswith('sub-EN'))


def check_access(subjects, dev, confirm, confirm_flag, lock_exists):
    """Refuse unknown subjects, any CONFIRM subject unless --confirm and the lock file both hold, and batches that
    mix DEV and CONFIRM subjects (a CONFIRM run never rewrites DEV outputs)."""
    unknown = sorted(set(subjects) - set(dev) - set(confirm))
    if unknown:
        raise ValueError(f'Not EN participants: {unknown}')
    blocked = sorted(set(subjects) & set(confirm))
    if blocked and not (confirm_flag and lock_exists):
        raise PermissionError(f'Refusing CONFIRM subjects {blocked}: requires --confirm and {LOCK.name}')
    if blocked and set(subjects) & set(dev):
        raise ValueError('A CONFIRM batch may not contain DEV subjects')
    return 'CONFIRM' if blocked else 'DEV'


# ------------------------------------------------------------------------------------- lock integrity

def code_fingerprint(root=ROOT):
    """{file: sha256} over CODE_FILES: the preparation code the lock pins (lock key ``lpp_code_sha256``)."""
    return {name: file_sha256(Path(root) / name) for name in CODE_FILES}


def check_lock(lock, dev, confirm, code):
    """Locked mask digest after validating the lock entries Module F relies on (before any download).

    Requires ``lpp_partition`` equal to the computed (dev, confirm) lists, ``lpp_mask_sha256`` as a hex
    digest and ``lpp_code_sha256`` equal to ``code`` (code_fingerprint() of the running code);
    ``lpp_dataset_doi``, when recorded, must be the registered release.
    """
    missing = [k for k in LOCK_KEYS if k not in lock]
    if missing:
        raise RuntimeError(f'{LOCK.name} lacks {missing}')
    part = lock['lpp_partition']
    if not isinstance(part, dict) or list(part.get('dev', ())) != list(dev) or \
            list(part.get('confirm', ())) != list(confirm):
        raise RuntimeError(f'{LOCK.name} lpp_partition differs from the registered hash partition')
    if lock.get('lpp_dataset_doi', DOI) != DOI:
        raise RuntimeError(f"{LOCK.name} records {lock['lpp_dataset_doi']}, the script pins {DOI}")
    digest = str(lock['lpp_mask_sha256'])
    if not re.fullmatch(r'[0-9a-f]{64}', digest):
        raise RuntimeError(f'{LOCK.name} lpp_mask_sha256 is not a SHA-256 hex digest')
    locked_code = lock['lpp_code_sha256']
    if not isinstance(locked_code, dict):
        raise RuntimeError(f'{LOCK.name} lpp_code_sha256 is not a {{file: sha256}} map')
    changed = sorted(k for k in set(code) | set(locked_code) if code.get(k) != locked_code.get(k))
    if changed:
        raise RuntimeError(f'{LOCK.name} lpp_code_sha256 differs from the running code: {changed}')
    return digest


def check_versions(current, recorded, reason=None):
    """Library versions equal to the DEV record, or the changes with a deviation ``reason`` (else refused)."""
    changed = {k: [recorded.get(k), current.get(k)] for k in sorted(set(current) | set(recorded))
               if recorded.get(k) != current.get(k)}
    if changed and not reason:
        raise RuntimeError(f'Library versions differ from the DEV record {changed}; pass --version-deviation '
                           'REASON (and record it in PROTOCOL_V4 §4) to proceed')
    return dict(changed=changed, reason=reason) if changed else None


def check_dev_record(record, code, versions, reason=None):
    """After the lock: the DEV record carries content digests and was written by the running (locked) code.

    Returns the logged library-version deviation, or None when the versions equal the DEV record.
    """
    missing = [k for k in DEV_RECORD_KEYS if k not in record]
    if missing:
        raise RuntimeError(f'prepare_provenance.json lacks {missing}; re-run DEV preparation before the lock')
    if record['code'] != code:
        changed = sorted(k for k in set(code) | set(record['code']) if code.get(k) != record['code'].get(k))
        raise RuntimeError(f'The DEV outputs were prepared by other code than the locked code: {changed}')
    return check_versions(versions, record['versions'], reason)


def lock_git_state(path=LOCK):
    """dict(tracked, clean, pushed, commit) of the lock file in its git repository; None without git."""
    path = Path(path)

    def git(*args):
        return subprocess.run(['git', '-C', str(path.parent), *args], capture_output=True, text=True)
    try:
        if git('rev-parse', '--show-toplevel').returncode:
            return None
    except OSError:
        return None
    tracked = git('ls-files', '--error-unmatch', '--', path.name).returncode == 0
    clean = tracked and git('diff', '--quiet', 'HEAD', '--', path.name).returncode == 0
    commit = git('log', '-1', '--format=%H', '--', path.name).stdout.strip() if tracked else ''
    pushed = bool(commit) and git('merge-base', '--is-ancestor', commit, 'origin/main').returncode == 0
    return dict(tracked=tracked, clean=clean, pushed=pushed, commit=commit or None)


def check_lock_commit(lock_sha256, given, git_state):
    """CONFIRM authorization: ``given`` (--lock-sha256, copied from the pushed commit) names this lock file, and
    with git the file is tracked, unmodified and its last commit is on origin/main (PROTOCOL_V4 §3)."""
    if given != lock_sha256:
        raise PermissionError(f'--lock-sha256 {given} is not the sha256 of {LOCK.name} ({lock_sha256})')
    if git_state is not None and not (git_state['tracked'] and git_state['clean'] and git_state['pushed']):
        raise PermissionError(f'{LOCK.name} must be committed, unmodified and pushed to origin/main: {git_state}')


def array_digest(value):
    """SHA-256 over dtype, shape and C-order bytes (H5 and npz container bytes are not reproducible)."""
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(f'{value.dtype.str}:{value.shape};'.encode())
    digest.update(value.reshape(-1).view(np.uint8))
    return digest.hexdigest()


def check_output(rel, digest, recorded):
    """After the lock: an output's content digest must equal the DEV record before any file is replaced.

    ``recorded`` is the DEV record's output_content (None before the lock: anything may be written).
    """
    if recorded is not None and recorded.get(rel) != digest:
        raise RuntimeError(f'{rel}: content differs from the DEV record; refusing to write it after the lock')


def mask_digest(mask, membership, coords, affine, block):
    """SHA-256 over the arrays that define the column space (an .npz file hash is not reproducible)."""
    digest = hashlib.sha256()
    for name, value in (('mask', np.asarray(mask, dtype=bool)), ('membership', np.asarray(membership, dtype=np.int64)),
                        ('coords', np.asarray(coords, dtype=np.int64)),
                        ('affine', np.round(np.asarray(affine, dtype=np.float64), 4) + 0.0),
                        ('block', np.asarray(block, dtype=np.int64))):
        value = np.ascontiguousarray(value)
        digest.update(f'{name}:{value.dtype.str}:{value.shape};'.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def state_digest(state):
    return mask_digest(state['mask'], state['membership'], state['coords'], state['affine'], state['block_size'])


def check_mask_state(state, locked, dev):
    """CONFIRM reuse of mask.npz: content digest equals the lock; grid and DEV subjects are the registered ones."""
    if state_digest(state) != locked:
        raise RuntimeError('data/lpp/mask.npz differs from the locked lpp_mask_sha256')
    if tuple(int(n) for n in state['grid_shape']) != GRID or [str(s) for s in state['dev_subjects']] != list(dev):
        raise RuntimeError('mask.npz grid or DEV subjects differ from the registered ones')


def mask_action(rebuilt, existing, locked):
    """'keep' or 'write' for the DEV mask.npz; refuses any change of a locked mask."""
    if locked is not None and rebuilt != locked:
        raise RuntimeError('The rebuilt DEV mask differs from the locked lpp_mask_sha256; refusing to overwrite')
    return 'keep' if existing == rebuilt else 'write'


def check_affine(affine, reference, where):
    if not np.allclose(affine, reference, atol=1e-4):
        raise ValueError(f'{where}: affine differs from the reference grid')


def check_inputs(inputs, recorded, complete=True):
    """Refuse inputs that changed relative to the DEV record, and (``complete``) missing recorded shared inputs.

    Both map key -> dict(sha256=...). Every key in both must agree; shared means non-derivative (metadata,
    words, audio), so DEV derivatives may be absent from a CONFIRM batch.
    """
    changed = sorted(k for k in inputs.keys() & recorded.keys() if inputs[k]['sha256'] != recorded[k]['sha256'])
    missing = sorted(k for k in recorded if complete and not DERIVATIVE.match(k) and k not in inputs)
    if changed or missing:
        raise RuntimeError(f'Inputs differ from the DEV record: changed {changed}, missing {missing}')


def write_frozen(path, data, frozen):
    """Write bytes unless identical; once frozen (the lock exists) an existing file may not change."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() == data:
            return False
        if frozen:
            raise RuntimeError(f'{path} would change after the lock')
    path.write_bytes(data)
    return True


def record_suffix(mode, subjects, rerun=None):
    """'dev'; 'dev_rerun_<rerun>' for a post-lock DEV re-run (``rerun``: a UTC stamp), which never replaces the
    DEV record; or 'confirm_<12 hex>' keyed by the sorted CONFIRM batch so batches never overwrite each other."""
    if mode == 'DEV':
        return 'dev' if rerun is None else f'dev_rerun_{rerun}'
    return 'confirm_' + hashlib.sha256('\n'.join(sorted(subjects)).encode()).hexdigest()[:12]


def record_names(suffix):
    """(download record, provenance record) file names for one run."""
    return (f'download_{DATASET}_{suffix}.json',
            'prepare_provenance.json' if suffix == 'dev' else f'prepare_provenance_{suffix}.json')


# ------------------------------------------------------------------------------------------ selection

def derivative_runs(objects):
    """{subject: [derivative keys]} for every EN subject in an S3 listing."""
    runs = {}
    for item in objects:
        match = DERIVATIVE.match(item['key'])
        if match:
            runs.setdefault(match[1], []).append(item['key'])
    return runs


def run_number(key):
    return int(re.search(r'_run-(\d+)_', key)[1])


def section_runs(keys):
    """[(section, run number, key)]: sections 1..9 are the runs in ascending run number."""
    keys = sorted(keys, key=run_number)
    numbers = [run_number(k) for k in keys]
    if len(keys) != len(SECTIONS) or len(set(numbers)) != len(numbers):
        raise ValueError(f'Expected {len(SECTIONS)} distinct runs, got {numbers}')
    return [(s, n, k) for s, n, k in zip(SECTIONS, numbers, keys)]


def verify_lengths(n_trs):
    if tuple(int(n) for n in n_trs) != SECTION_TRS:
        raise ValueError(f'TR counts {list(n_trs)} differ from the verified {list(SECTION_TRS)}')


def select_objects(objects, subjects):
    """Listing items to download: metadata, EN annotation and audio, and the subjects' derivatives only."""
    by_key = {item['key']: item for item in objects}
    want = {f'{DATASET}/{name}' for name in META}
    want |= {f'{DATASET}/annotation/EN/lppEN_section{s}.TextGrid' for s in SECTIONS}
    want |= {f'{DATASET}/stimuli/task-lppEN_section-{s}.wav' for s in SECTIONS}
    missing = sorted(want - by_key.keys())
    if missing:
        raise ValueError(f'Missing dataset objects: {missing}')
    runs = derivative_runs(objects)
    for subject in subjects:
        section_runs(runs.get(subject, []))
        want |= set(runs[subject])
    selected = [by_key[k] for k in sorted(want)]
    stray = {m[1] for m in (DERIVATIVE.match(x['key']) for x in selected) if m} - set(subjects)
    if stray:
        raise RuntimeError(f'Selection contains unrequested subjects {sorted(stray)}')
    return selected


# ---------------------------------------------------------------------------------------------- words

def read_words(csv_text):
    """{section: [(onset, offset, word)]} from lppEN_word_information.csv, read without NaN coercion.

    Empty words (the split-off 'not' of contractions) are kept here and dropped downstream exactly
    like every BAD_WORDS token. Raises unless the total is 15,376, sections are 1..9, onsets are
    ordered, durations positive, words do not overlap, and every word lies inside its section's run.
    """
    rows = list(csv.DictReader(io.StringIO(csv_text.lstrip('﻿'))))
    if len(rows) != N_WORDS:
        raise ValueError(f'{len(rows)} annotated words, expected {N_WORDS}')
    words = {s: [] for s in SECTIONS}
    for r in rows:
        section = str(int(r['section']))
        if section not in words:
            raise ValueError(f'Unknown section {section}')
        words[section].append((float(r['onset']), float(r['offset']), r['word'].strip()))
    for s, n_tr in zip(SECTIONS, SECTION_TRS):
        on = np.array([a for a, _, _ in words[s]])
        off = np.array([b for _, b, _ in words[s]])
        if not len(on) or not np.isfinite(np.r_[on, off]).all():
            raise ValueError(f'Section {s}: missing or non-finite timings')
        if np.any(np.diff(on) < 0) or np.any(off <= on) or np.any(on[1:] < off[:-1] - 1e-6):
            raise ValueError(f'Section {s}: words must be ordered, positive and non-overlapping')
        if on[0] < 0 or off[-1] > n_tr * TR:
            raise ValueError(f'Section {s}: words outside the {n_tr * TR:g} s run')
    return words


def textgrid_text(intervals, xmax, tier='words'):
    """Praat ooTextFile with one IntervalTier; gaps are empty intervals (dropped as BAD_WORDS)."""
    items, t = [], 0.0
    for a, b, w in intervals:
        a, b = round(float(a), 6), round(float(b), 6)
        if '"' in w or a < t - 1e-9 or b <= a:
            raise ValueError(f'Invalid word interval {(a, b, w)}')
        if a > t:
            items.append((t, a, ''))
        items.append((max(a, t), b, w))
        t = b
    if xmax < t:
        raise ValueError('Words extend past xmax')
    if xmax > t:
        items.append((t, float(xmax), ''))
    head = (f'File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = {xmax:.6f}\n'
            f'tiers? <exists>\nsize = 1\nitem []:\n    item [1]:\n        class = "IntervalTier"\n'
            f'        name = "{tier}"\n        xmin = 0\n        xmax = {xmax:.6f}\n'
            f'        intervals: size = {len(items)}\n')
    return head + ''.join(f'        intervals [{i}]:\n            xmin = {a:.6f}\n            xmax = {b:.6f}\n'
                          f'            text = "{w}"\n' for i, (a, b, w) in enumerate(items, 1))


def used_words(intervals):
    from src.semantic.features import BAD_WORDS
    return [(a, b, w) for a, b, w in intervals if w.lower() not in BAD_WORDS]


def picture_mask(n_tr, section, tr_start=TR_START, tr_seconds=TR, window=PICTURE_SECONDS):
    """Per-TR bool: the TR's centered interval overlaps the section-1 picture window (audio time)."""
    centers = tr_start + np.arange(int(n_tr)) * tr_seconds
    if str(section) != PICTURE_SECTION:
        return np.zeros(int(n_tr), dtype=bool)
    return (centers - tr_seconds / 2 < window[1]) & (centers + tr_seconds / 2 > window[0])


def encode_sections(encoder, words_by_section, batch_size=16):
    """One encoder call per section, so no word's causal context crosses a section start (R6)."""
    return {s: encoder.encode(list(words), batch_size) for s, words in words_by_section.items()}


def motion_record(keys, subjects, readme, participants_json):
    """Descriptor-route motion check: released motion files per subject and the descriptor's statements."""
    files = {s: sorted(k for k in keys if f'/{s}/' in k and MOTION_FILE.search(k.rsplit('/', 1)[-1]))
             for s in subjects}
    sentences = re.split(r'(?<=[.!?])\s+', ' '.join(readme.split()))
    return dict(route='descriptor (no raw-echo framewise displacement computed)',
                motion_parameters_released=any(files.values()), motion_files=files,
                readme_statements=[t for t in sentences if re.search(r'motion|movement|move\b|exclu', t, re.I)],
                participants_columns=sorted(json.loads(participants_json)), excluded_subjects=[])


# ----------------------------------------------------------------------------------------------- mask

def choose_block(n_voxels):
    return BLOCK if int(n_voxels) > MAX_VOXELS else 1


def block_columns(voxel_ijk, block):
    """(membership [n_vox], coords [V, 3]): columns are block ijk in lexicographic order.

    With block 1 this is the identity in C order over the grid (np.argwhere order).
    """
    blocks = np.asarray(voxel_ijk, dtype=np.int64) // int(block)
    coords, membership = np.unique(blocks, axis=0, return_inverse=True)
    return membership.reshape(-1), coords


def pool(x, membership, n_columns):
    """Column means of x [T, n_vox] over membership groups, float32 [T, n_columns]."""
    membership = np.asarray(membership)
    if n_columns == len(membership) and np.array_equal(membership, np.arange(n_columns)):
        return np.ascontiguousarray(x, dtype=np.float32)
    counts = np.bincount(membership, minlength=n_columns)
    if np.any(counts == 0):
        raise ValueError('Every column needs at least one voxel')
    order = np.argsort(membership, kind='stable')
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    sums = np.add.reduceat(np.asarray(x, dtype=np.float64)[:, order], starts, axis=1)
    return (sums / counts).astype(np.float32)


def zscore_run(y, floor=STD_FLOOR):
    """(float32 [T, V], #floored columns): one run's columns z-scored over its own TRs.

    float64 arithmetic over finite values only; columns with SD <= floor and non-finite entries become 0.
    Uses nothing but this run's BOLD (no labels, no other run), as the ds003020 H5s were normalized.
    """
    z = np.array(y, dtype=np.float64)
    finite = np.isfinite(z)
    z[~finite] = 0.0
    n = np.maximum(finite.sum(0), 1)
    z -= z.sum(0) / n
    z[~finite] = 0.0
    std = np.sqrt(np.einsum('ij,ij->j', z, z) / n)
    small = std <= floor
    z /= np.where(small, 1.0, std)
    z[:, small] = 0.0
    return z.astype(np.float32), int(small.sum())


def dvars(z):
    """Per-volume RMS over columns of the temporal difference of z-scored data [T - 1] (motion QC proxy)."""
    return np.sqrt(np.mean(np.square(np.diff(np.asarray(z, dtype=np.float64), axis=0)), axis=1))


def nifti_dims(gz_prefix):
    """Image dimensions from the first bytes of a gzipped NIfTI-1 file."""
    raw = zlib.decompressobj(31).decompress(gz_prefix)
    if len(raw) < 348:
        raise ValueError('Truncated NIfTI header')
    endian = '<' if struct.unpack('<i', raw[:4])[0] == 348 else '>'
    dim = struct.unpack(endian + '8h', raw[40:56])
    return tuple(int(d) for d in dim[1:dim[0] + 1])


def raw_volumes(key):
    request = urllib.request.Request(f'{BASE}/{key}', headers={'Range': 'bytes=0-65535'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return nifti_dims(response.read())[-1]


def load_run(path):
    import nibabel as nib
    image = nib.load(str(path))
    if image.shape[:3] != GRID or len(image.shape) != 4:
        raise ValueError(f'{path}: grid {image.shape}, expected {GRID} x time')
    zooms = tuple(float(z) for z in image.header.get_zooms())
    if not np.allclose(zooms, (VOXEL_MM,) * 3 + (TR,)):
        raise ValueError(f'{path}: voxel/TR sizes {zooms}')
    return image, np.asarray(image.dataobj, dtype=np.float32)


def gray_matter(affine):
    """MNI152 gray-matter mask (nilearn) resampled with nearest neighbour to the derivative grid."""
    import nibabel as nib
    from nilearn import datasets, image
    gm = datasets.load_mni152_gm_mask(**GM_MASK)
    reference = nib.Nifti1Image(np.zeros(GRID, dtype=np.uint8), affine)
    resampled = image.resample_to_img(gm, reference, interpolation='nearest', force_resample=True,
                                      copy_header=True)
    return np.asarray(resampled.get_fdata()) > 0.5


# ---------------------------------------------------------------------------------------------- main

def file_digests(path):
    """(sha256, md5) hex digests in one pass."""
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def fetch(item, root):
    """Input record dict(size, etag, sha256, md5) of one listing item.

    download() keeps any same-size local file, so its MD5 is checked against a single-part ETag and a stale
    file is fetched once more; multipart ETags are not checksums and stay size-checked only.
    """
    from src.data import openneuro
    for _ in range(2):
        path = openneuro.download(item, root)
        sha, md5 = file_digests(path)
        if '-' in item['etag'] or md5 == item['etag']:
            return dict(size=item['size'], etag=item['etag'], sha256=sha, md5=md5)
        path.unlink()
    raise RuntimeError(f"{item['key']}: local content differs from the listing ETag")


def fetch_all(selected, root, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool_:
        return dict(zip((x['key'] for x in selected), pool_.map(lambda item: fetch(item, root), selected)))


def build_targets(textgrids, out, device, batch_size, expected=None):
    """Shared per-section GPT-2 targets (subject-independent); content-addressed cache.

    ``expected`` (after the lock: the DEV record's output_content) maps target files relative to ``out`` to
    array digests. A section whose cache key is not recorded is refused; a cached file that differs is
    recomputed, and recomputed arrays that differ are refused before anything is saved.
    """
    from src.semantic.features import (MODEL, REVISION, FrozenGPT2, downsample_words, fingerprint,
                                       read_word_intervals, tr_centers)
    metadata = dict(TARGET_METADATA, model=MODEL, revision=REVISION)
    out = Path(out)
    folder = out / 'semantic/targets'
    folder.mkdir(parents=True, exist_ok=True)
    plan, todo = {}, {}
    for s, n_tr in zip(SECTIONS, SECTION_TRS):
        intervals = read_word_intervals(textgrids[s])
        run = dict(run_id=f'lppEN_section-{s}', n_tr=n_tr, tr_seconds=TR, tr_start_seconds=TR_START)
        centers = tr_centers(run, out)
        key = fingerprint(dict(metadata=metadata, section=s, run=run, centers=centers.tolist(),
                               stimulus_sha256=file_sha256(textgrids[s])))
        paths = folder / f'{key}.targets.npy', folder / f'{key}.valid.npy'
        rels = [str(p.relative_to(out)) for p in paths]
        if expected is not None and any(r not in expected for r in rels):
            raise RuntimeError(f'Section {s}: target cache key {key} is not in the DEV record (changed after the lock)')
        plan[s] = (intervals, centers, key, paths, rels)
        ok = all(p.exists() for p in paths)
        if ok:
            y, v = np.load(paths[0]), np.load(paths[1])
            ok = y.shape[0] == n_tr and v.shape == (n_tr,) and v.dtype == np.bool_ and np.isfinite(y).all()
            ok = ok and (expected is None or [array_digest(y), array_digest(v)] == [expected[r] for r in rels])
        if not ok:
            todo[s] = [w for _, _, w in intervals]
    if todo:
        vectors = encode_sections(FrozenGPT2(MODEL, REVISION, device), todo, batch_size)
        for s, vec in vectors.items():
            intervals, centers, _, paths, rels = plan[s]
            y, v = downsample_words(intervals, vec, centers, TR)
            for rel, value in zip(rels, (y, v)):
                check_output(rel, array_digest(value), expected)
            np.save(paths[0], y)
            np.save(paths[1], v)
    result = {}
    for s, (intervals, _, key, paths, rels) in plan.items():
        y, v = np.load(paths[0]), np.load(paths[1])
        result[s] = dict(cache_key=key, target_path=str(paths[0].resolve()), target_valid_path=str(paths[1].resolve()),
                         embedding_dim=int(y.shape[1]), words=len(intervals), valid_trs=int(v.sum()), n_tr=len(v),
                         recomputed=s in todo, content_sha256={rels[0]: array_digest(y), rels[1]: array_digest(v)})
    return metadata, result


def library_versions():
    import h5py
    import nibabel
    import nilearn
    import torch
    import transformers
    return dict(python=platform.python_version(), numpy=np.__version__, nibabel=nibabel.__version__,
                nilearn=nilearn.__version__, h5py=h5py.__version__, torch=torch.__version__,
                transformers=transformers.__version__)


def write_h5(path, y, attrs, rel, expected=None):
    """Atomically write one run's H5 'data'; returns its array digest.

    With ``expected`` (a post-lock DEV re-run: the DEV record's output_content) the content must equal the
    record before anything is written, and an existing file with equal content is kept.
    """
    import h5py
    path, digest = Path(path), array_digest(y)
    check_output(rel, digest, expected)
    if expected is None or h5_digest(path) != digest:
        temporary = path.with_suffix('.hf5.tmp')
        with h5py.File(temporary, 'w') as handle:
            handle.create_dataset('data', data=y)
            handle.attrs.update(attrs)
        temporary.replace(path)
    return digest


def h5_digest(path):
    """array_digest of an H5 file's 'data' array, or None when the file is missing or unreadable."""
    import h5py
    try:
        with h5py.File(path, 'r') as handle:
            return array_digest(handle['data'][()])
    except (OSError, KeyError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='data/lpp')
    parser.add_argument('--subjects', nargs='*', help='Default: the 9 DEV subjects (CONFIRM: all 40 with --confirm)')
    parser.add_argument('--confirm', action='store_true', help='Allow CONFIRM subjects (also needs the lock file)')
    parser.add_argument('--lock-sha256', help='CONFIRM: sha256 of the committed and pushed experiments/v4_lock.json')
    parser.add_argument('--version-deviation', help='After the lock: reason for library versions that differ from '
                                                    'the DEV record (logged; record it in PROTOCOL_V4 §4)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true', help='Partition and selection only; no data download')
    a = parser.parse_args(argv)
    from src.data.openneuro import discover
    out = Path(a.out).resolve()
    root = out / 'openneuro'
    code = code_fingerprint()
    objects = discover(DATASET)
    listing = {x['key']: x for x in objects}
    early = {k: fetch(listing[k], root) for k in (f'{DATASET}/participants.tsv', f'{DATASET}/dataset_description.json')}
    description = json.loads((root / DATASET / 'dataset_description.json').read_text())
    if description.get('DatasetDOI') != DOI:
        raise RuntimeError(f"OpenNeuro serves {description.get('DatasetDOI')}; the registered release is {DOI}")
    participants = en_participants((root / DATASET / 'participants.tsv').read_text(encoding='utf-8'))
    with_runs = sorted(s for s, keys in derivative_runs(objects).items() if len(keys) == len(SECTIONS))
    if participants != with_runs or len(participants) != N_EN:
        raise ValueError(f'participants.tsv EN ids ({len(participants)}) differ from EN derivatives ({len(with_runs)})')
    dev, confirm = lpp_partition(participants)
    subjects = list(a.subjects) if a.subjects else (confirm if a.confirm else dev)
    mode = check_access(subjects, dev, confirm, a.confirm, LOCK.exists())
    if mode == 'DEV' and sorted(subjects) != sorted(dev):
        raise ValueError('DEV preparation processes all 9 DEV subjects together (the mask needs all of them)')

    # Lock integrity, before anything else is downloaded or read: lock entries (partition, mask digest, code),
    # the DEV record (same code, versions, shared inputs) and, for CONFIRM, the committed lock and mask.npz.
    mask_path, dev_record = out / 'mask.npz', out / 'prepare_provenance.json'
    frozen = LOCK.exists()
    locked = lock_sha256 = git_state = deviation = recorded = locked_outputs = state = None
    if frozen:
        locked = check_lock(json.loads(LOCK.read_text()), dev, confirm, code)
        lock_sha256 = file_sha256(LOCK)
        if not dev_record.exists():
            raise RuntimeError(f'{dev_record} is required once {LOCK.name} exists')
        record = json.loads(dev_record.read_text())
        deviation = check_dev_record(record, code, library_versions(), a.version_deviation)
        recorded, locked_outputs = record['inputs'], record['output_content']
        check_inputs(early, recorded, complete=False)
    if mode == 'CONFIRM':
        git_state = lock_git_state(LOCK)
        check_lock_commit(lock_sha256, a.lock_sha256, git_state)
        with np.load(mask_path) as handle:
            state = {k: handle[k] for k in handle.files}
        check_mask_state(state, locked, dev)

    selected = select_objects(objects, subjects)
    suffix = record_suffix(mode, subjects, datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                           if frozen and mode == 'DEV' else None)
    download_name, provenance_name = record_names(suffix)
    lock_record = dict(present=frozen, file_sha256=lock_sha256, git=git_state, lpp_mask_sha256=locked,
                       code_checked=frozen, partition_checked=frozen, inputs_checked=frozen,
                       outputs_checked=frozen, version_deviation=deviation)
    save_json(dict(dataset=DATASET, dataset_doi=DOI, mode=mode, subjects=subjects, source=BASE, lock=lock_record,
                   code=code, provenance=provenance_name, objects=selected,
                   total_bytes=sum(x['size'] for x in selected)), root / download_name)
    print(json.dumps(dict(mode=mode, dev=dev, subjects=subjects, objects=len(selected), lock=frozen,
                          gigabytes=round(sum(x['size'] for x in selected) / 1e9, 3))), flush=True)
    if a.dry_run:
        return
    inputs = fetch_all(selected, root, a.workers)
    if recorded is not None:
        check_inputs(inputs, recorded)

    # Words, TextGrids and picture masks (subject-independent; frozen once the lock exists).
    words = read_words((root / WORDS_KEY).read_text(encoding='utf-8'))
    from src.semantic.features import read_word_intervals
    stimuli = out / 'stimuli'
    stimuli.mkdir(parents=True, exist_ok=True)
    textgrids, pictures, word_check = {}, {}, {}
    import nibabel as nib  # noqa: F401  (fail early if missing)
    for s, n_tr in zip(SECTIONS, SECTION_TRS):
        textgrids[s] = stimuli / f'section-{s}.TextGrid'
        write_frozen(textgrids[s], textgrid_text(words[s], n_tr * TR).encode('utf-8'), frozen)
        parsed = read_word_intervals(textgrids[s])
        expected = used_words(words[s])
        if len(parsed) != len(expected) or any(abs(p[0] - e[0]) > 1e-5 or p[2] != e[2] for p, e in zip(parsed, expected)):
            raise RuntimeError(f'Section {s}: TextGrid round trip changed the words')
        pictures[s] = stimuli / f'picture_section-{s}.npy'
        buffer = io.BytesIO()
        np.save(buffer, picture_mask(n_tr, s))
        write_frozen(pictures[s], buffer.getvalue(), frozen)
        official = (root / DATASET / f'annotation/EN/lppEN_section{s}.TextGrid').read_bytes()
        text = official.decode('utf-16' if official[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8')
        xmax = float(re.search(r'xmax = ([\d.]+)', text)[1])
        with open(root / DATASET / f'stimuli/task-lppEN_section-{s}.wav', 'rb') as handle:
            wav = handle.read(4096)
        channels, rate = struct.unpack('<HI', wav[22:28])
        bits = struct.unpack('<H', wav[34:36])[0]
        data_at = wav.find(b'data')
        audio_seconds = struct.unpack('<I', wav[data_at + 4:data_at + 8])[0] / (rate * channels * bits // 8)
        if abs(xmax - n_tr * TR) > 1e-6 or not 0 < n_tr * TR - audio_seconds < 2 * DISCARDED * TR:
            raise ValueError(f'Section {s}: official TextGrid ({xmax}) or audio ({audio_seconds:.3f} s) '
                             f'does not fit the {n_tr * TR:g} s run')
        word_check[s] = dict(csv_words=len(words[s]), empty_words=sum(w == '' for _, _, w in words[s]),
                             used_words=len(expected), first_onset=words[s][0][0], last_offset=words[s][-1][1],
                             run_seconds=n_tr * TR, official_textgrid_xmax=xmax, audio_seconds=round(audio_seconds, 4),
                             picture_trs=int(np.load(pictures[s]).sum()))

    # Pass 1: run mapping, G4(a) lengths, raw-header offsets and per-subject validity. CONFIRM runs must sit
    # on the locked mask's grid; DEV runs on the first DEV run's grid.
    listing_runs = derivative_runs(objects)
    runs, lengths, offsets, valid = {}, {}, {}, {}
    affine = None if state is None else np.asarray(state['affine'], dtype=np.float64)
    for subject in subjects:
        runs[subject] = section_runs(listing_runs[subject])
        valid[subject], lengths[subject] = np.ones(GRID, dtype=bool), []
        for s, number, key in runs[subject]:
            image, x = load_run(root / key)
            if affine is None:
                affine = image.affine
            check_affine(image.affine, affine, key)
            lengths[subject].append(int(x.shape[3]))
            valid[subject] &= np.isfinite(x).all(-1) & (x.max(-1) > x.min(-1))
            del x
        verify_lengths(lengths[subject])
        raw_keys = [f'{DATASET}/{subject}/func/{subject}_task-lppEN_run-{number:02d}_echo-1_bold.nii.gz'
                    for _, number, _ in runs[subject]]
        offsets[subject] = [raw_volumes(k) - n for k, n in zip(raw_keys, lengths[subject])]
        if any(d != DISCARDED for d in offsets[subject]):
            raise ValueError(f'{subject}: raw minus derivative volumes {offsets[subject]}, expected {DISCARDED}')
        print(json.dumps(dict(subject=subject, runs=[n for _, n, _ in runs[subject]], n_tr=lengths[subject],
                              raw_minus_derivative=offsets[subject], valid_voxels=int(valid[subject].sum()))), flush=True)

    # Mask: fixed from DEV subjects only; once locked, a DEV re-run may not change it and CONFIRM reuses it.
    if mode == 'DEV':
        gm = gray_matter(affine)
        valid_all = np.logical_and.reduce([valid[s] for s in dev])
        mask = gm & valid_all
        voxel_ijk = np.argwhere(mask)
        block = choose_block(len(voxel_ijk))
        membership, coords = block_columns(voxel_ijk, block)
        existing = None
        if mask_path.exists():
            with np.load(mask_path) as handle:
                existing = state_digest(handle)
        if mask_action(mask_digest(mask, membership, coords, affine, block), existing, locked) == 'write':
            centers = coords * block + (block - 1) / 2
            world = (affine @ np.c_[centers, np.ones(len(centers))].T).T[:, :3]
            np.savez(mask_path, mask=mask, gray_matter=gm, valid_all_dev=valid_all, voxel_ijk=voxel_ijk,
                     membership=membership, coords=coords, world_mm=world, affine=affine, grid_shape=np.array(GRID),
                     block_size=np.int64(block), max_voxels=np.int64(MAX_VOXELS), dev_subjects=np.array(dev),
                     coord_unit=np.array('voxel' if block == 1 else f'{block}x{block}x{block} voxel block'))
    with np.load(mask_path) as handle:
        state = {k: handle[k] for k in handle.files}
    if locked is not None:
        check_mask_state(state, locked, dev)
    mask, membership, coords = state['mask'], state['membership'], state['coords']
    block, gm_voxels = int(state['block_size']), int(state['gray_matter'].sum())
    valid_voxels, content_sha256 = int(state['valid_all_dev'].sum()), state_digest(state)
    n_columns = len(coords)

    # Pass 2: masked (and block-averaged when required) columns, z-scored within each run; one H5 per section.
    # After the lock a DEV re-run must reproduce every recorded 'data' array; an equal existing file is kept.
    outputs, content, qc = {}, {'mask.npz': content_sha256}, {}
    h5_expected = locked_outputs if mode == 'DEV' else None
    for subject in subjects:
        folder = out / 'raw' / subject
        folder.mkdir(parents=True, exist_ok=True)
        qc[subject] = dict(nonfinite_column_runs=0, zero_variance_column_runs=0, dvars={})
        for s, number, key in runs[subject]:
            _, x = load_run(root / key)
            y = pool(x[mask].T, membership, n_columns)
            del x
            qc[subject]['nonfinite_column_runs'] += int((~np.isfinite(y).all(0)).sum())
            y, floored = zscore_run(y)
            qc[subject]['zero_variance_column_runs'] += floored
            d = dvars(y)
            qc[subject]['dvars'][s] = dict(median=round(float(np.median(d)), 4),
                                           p99=round(float(np.percentile(d, 99)), 4), max=round(float(d.max()), 4))
            path = folder / f'{s}.hf5'
            rel = str(path.relative_to(out))
            content[rel] = write_h5(path, y, dict(subject=subject, section=s, run_number=number, source_key=key,
                                                  bold_normalization=NORMALIZATION['bold_normalization']),
                                    rel, h5_expected)
            outputs[rel] = file_sha256(path)
        print(json.dumps(dict(subject=subject, columns=n_columns,
                              **{k: v for k, v in qc[subject].items() if k != 'dvars'})), flush=True)

    # Targets (checked against the DEV record once locked) and manifests (frozen once locked).
    metadata, targets = build_targets(textgrids, out, a.device, a.batch_size, locked_outputs)
    for s, t in targets.items():
        content.update(t['content_sha256'])
        print(json.dumps(dict(section=s, words=t['words'], valid_trs=t['valid_trs'], n_tr=t['n_tr'])), flush=True)
    for subject in subjects:
        result = []
        for (s, number, key), n_tr in zip(runs[subject], lengths[subject]):
            t = targets[s]
            result.append(dict(
                run_id=f'{DATASET}_{subject}_section-{s}', dataset=DATASET, subject=subject,
                condition='perceived_speech', session=s, split='train', story=s, section=int(s), n_tr=n_tr,
                n_features=n_columns, tr_seconds=TR, tr_start_seconds=TR_START, source_key=f'{subject}/{s}.hf5',
                source_object=key, source_run_number=number, repeat=None,
                stimulus_path=str(textgrids[s].resolve()), stimulus_timing_verified=True,
                audio_path=str((root / DATASET / f'stimuli/task-lppEN_section-{s}.wav').resolve()),
                picture_path=str(pictures[s].resolve()),
                picture_seconds=list(PICTURE_SECONDS) if s == PICTURE_SECTION else None,
                target_path=t['target_path'], target_valid_path=t['target_valid_path'],
                semantic_metadata=dict(metadata, embedding_dim=t['embedding_dim'], cache_key=t['cache_key'])))
        manifest = dict(schema_version=1, dataset=DATASET, dataset_doi=DOI, loader='lpp', loader_note=LOADER_NOTE,
                        subject=subject, mode='lpp', partition='CONFIRM' if subject in confirm else 'DEV',
                        raw_root=str((out / 'raw').resolve()), mask_path=str(mask_path),
                        normalization_path=str(mask_path), **NORMALIZATION,
                        normalization_note='The H5 data are z-scored within each run (no statistic is fit across '
                                           'runs); this npz holds the mask, membership and coords',
                        mask_content_sha256=content_sha256, feature_order=(
                            'mask voxels in C order over the 73x90x74 derivative grid' if block == 1 else
                            f'means over {block}x{block}x{block} voxel blocks, blocks in lexicographic ijk order'),
                        runs=result, skipped_runs=[], semantic_metadata=metadata)
        path = out / 'semantic' / subject / 'semantic_manifest.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        write_frozen(path, (json.dumps(manifest, indent=2, allow_nan=False, default=str) + '\n').encode(), frozen)
        outputs[str(path.relative_to(out))] = file_sha256(path)
    for s in SECTIONS:
        for path in (textgrids[s], pictures[s], Path(targets[s]['target_path']), Path(targets[s]['target_valid_path'])):
            outputs[str(path.relative_to(out))] = file_sha256(path)
    outputs['mask.npz'] = file_sha256(mask_path)

    import nilearn
    audio_gap = [round(v['run_seconds'] - v['audio_seconds'], 1) for v in word_check.values()]
    report = dict(
        module='F', script='scripts/lpp_prepare.py', script_sha256=code['scripts/lpp_prepare.py'], code=code,
        created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'), dataset=DATASET,
        dataset_doi=DOI, source=BASE, mode=mode, no_decoding_score=True, lock=lock_record,
        partition=dict(rule="ascending sha256('ssltang-v4-lpp-subject:' + participant_id); first 9 DEV",
                       id_convention="participants.tsv participant_id verbatim, e.g. 'sub-EN057'",
                       implementation=partition_source(), en_participants=len(participants), dev=dev,
                       confirm=confirm),
        subjects=subjects,
        g4a=dict(passed=True, section_trs_expected=list(SECTION_TRS), section_mapping='ascending derivative run number',
                 runs={s: [dict(section=sec, run_number=n, n_tr=t) for (sec, n, _), t in zip(runs[s], lengths[s])]
                       for s in subjects},
                 raw_minus_derivative_volumes=offsets, words_total=sum(v['csv_words'] for v in word_check.values()),
                 words_expected=N_WORDS, words_used_total=sum(v['used_words'] for v in word_check.values()),
                 words=word_check,
                 motion=dict(motion_record(listing, subjects, (root / DATASET / 'README').read_text(encoding='utf-8'),
                                           (root / DATASET / 'participants.json').read_text(encoding='utf-8')),
                             dvars={s: qc[s]['dvars'] for s in subjects},
                             dvars_rule='RMS over columns of the TR-to-TR difference of the per-run z-scored H5 data')),
        audio_onset=dict(
            tr_start_seconds=TR_START, tr_seconds=TR,
            convention='TR centers relative to audio onset (as ds003020: 11.0); derivative volume k covers '
                       '[2k, 2k+2) s of audio time, center 2k+1',
            evidence=['descriptor: trigger at section start, 8 s (4 TRs) delay to audio onset; first 4 volumes excluded',
                      'code/lppEN_presentation.py (released with debug = 1, which selects beforeafter = 2 s and '
                      'tempsounds/ audio; the non-debug branch uses beforeafter = 8 s, matching the descriptor): '
                      'trigger, pre-audio wait, then play audio; not used as evidence for the 8 s value',
                      f'raw echo-1 headers minus derivatives = {DISCARDED} volumes for every {mode} run',
                      'official section TextGrids span exactly n_tr * 2 s (time 0 = derivative volume 0)',
                      f'every annotated word lies in [0, n_tr * 2) s; audio ends {min(audio_gap):g}-'
                      f'{max(audio_gap):g} s before run end',
                      'AFNI slice-timing correction aligns to the mean slice time (about 0.98 s), i.e. TR center']),
        picture=dict(section=PICTURE_SECTION, seconds=list(PICTURE_SECONDS), trs=word_check[PICTURE_SECTION]['picture_trs'],
                     rule='TR centered interval overlaps [0, 80) s of section-1 audio time'),
        mask=dict(gray_matter=f'nilearn {nilearn.__version__} load_mni152_gm_mask({GM_MASK}), nearest resample_to_img',
                  gray_matter_voxels_on_grid=gm_voxels, valid_all_dev_voxels=valid_voxels,
                  mask_voxels=int(mask.sum()), max_voxels=MAX_VOXELS, block_averaging=block > 1, block_size=block,
                  columns=n_columns, content_sha256=content_sha256,
                  content_sha256_rule='lpp_prepare.mask_digest over mask, membership, coords, affine (1e-4), block size',
                  qc={s: {k: v for k, v in q.items() if k != 'dvars'} for s, q in qc.items()}),
        normalization=NORMALIZATION, loader=dict(loader='lpp', note=LOADER_NOTE),
        targets=dict(metadata=metadata, sections=targets,
                     context_resets=[dict(stimulus=s, kind='transcript', purpose='context_reset', word_index=0)
                                     for s in SECTIONS]),
        inputs=inputs, outputs=outputs, output_content=content,
        output_content_rule='lpp_prepare.array_digest of H5 data and target arrays; mask.npz: mask content_sha256',
        versions=library_versions(), deviations=[deviation] if deviation else [])
    save_json(report, out / provenance_name)
    print(json.dumps(dict(provenance=str(out / provenance_name), mask_voxels=int(mask.sum()), columns=n_columns,
                          block_size=block), indent=1), flush=True)


if __name__ == '__main__':
    main()
