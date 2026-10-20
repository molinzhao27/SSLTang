#!/usr/bin/env python3
"""PROTOCOL_V4 on Le Petit Prince EN (OpenNeuro ds003643): the LPP runner (V4_IMPLEMENTATION Module G).

Stages (sub-commands; one target listener per develop/confirm call):
  prepare-caches --subjects S..  delay-row caches (raw H5 rows t+1..t+3) of every eligible bin on the dense and
                                 stride-4 clocks, data/cache/v4_lpp/<sub>/{dense,stride4}, bound to the recorded
                                 content of the listener's H5 files (lpp_content.json; a cache whose content
                                 changed is rebuilt, after every H5 is checked against its recorded digest).
                                 CONFIRM listeners need the lock file.
  g4 --subjects <DEV..>          G4 in the registered order, W0 ONLY before any prior arm touches LPP:
                                 (b) dense W0 at L8 in all 9 outer folds against its within-section circular-shift
                                     null (>= 40 TR, 2,000 draws, seed 0); passes in >= 6/9 DEV listeners;
                                 (c) only after (b) passed: every ISP alpha (final and inner-fold fits, L3/L5/L8, all
                                     9 outer folds; label-free, no decoder) is interior to logspace(0, 10, 21).
                                 Per-listener records are resumable under the same code and data; results/v4_lpp/
                                 g4.json is written once every DEV listener has them. (a) is lpp_prepare's G4(a)
                                 record. Refused once the lock exists.
  develop --subject <DEV>        every outer fold, budget (L3/L5/L8), clock and arm; cells at stage 'develop'. Needs
                                 G0 (results/v4/g0.json) under the run_v4 core; any arm but W0 needs G4 passed under
                                 this code and data. Default arms DEVELOP_DEFAULT: the Tier-1 arms but P_AC, the
                                 comparators with their two-reference sensitivity and the EB_K subsets (R_P(K) on
                                 DEV), the last three at the primary cell (--primary-budget, dense), where the §9
                                 controls are also computed. Refused once experiments/v4_lock.json exists.
  confirm --subject <CONFIRM> --lock-sha256 <sha256 of the lock file>
                                 the same at stage 'evaluate', only under experiments/v4_lock.json whose sha256 is the
                                 given one, whose fingerprints equal this code (code_sha256 = run_v4 core,
                                 lpp_cell_code_sha256, lpp_runner_sha256, tanghuth_sha256, summarize_sha256,
                                 config_sha256), which carries a confirmatory plan, lists the listener as CONFIRM and
                                 is committed, unmodified and pushed (where git is unavailable only with
                                 --no-git-check, recorded in every evaluate cell), whose lpp_mask_sha256 is every
                                 manifest's mask digest and whose DEV preparation record is the one on disk. Refused
                                 once results/v4/primary_started.json exists. The default arms are CONFIRM_DEFAULT
                                 plus every cell the locked plan reads (summarize_v4.required_cells and the ORACLE/
                                 W0 cells of its oracle fractions); an explicit --arms/--budgets/--clocks request
                                 that leaves out a plan cell of a requested section not already on disk is refused.
                                 Scores are not printed at this stage (endpoint-blind QC comes first).
  develop/confirm/prepare-caches refuse to start with less than --min-free-gb (default 50) free under the cache and
                                 results roots. Disk budget at CONFIRM size (49 listeners, 140,369 voxels): about
                                 260 GB of delay-row caches (both clocks), 60 GB of ISP/SELF maps, 57 GB of reference
                                 voxel slices and 82 GB of reference models (Module H, float64), about 0.45 TB.

Reference pool (``reference_pool``): DEV targets use the other DEV listeners, CONFIRM targets all other EN listeners
of the partition, in both cases only those that pass the subject-level QC of the primary analysis
(``summarize_v4.registered_pool``: a preparation record, data/lpp/prepare_provenance*.json g4a, with exactly the
registered section TR counts). A listener without one has no prepared data and no target uses it; the target itself
must pass. Every cell records the pool (``reference_pool``: partition members, references, dropped listeners). This
reading of "all other cohort listeners" needs its PROTOCOL_V4 §4 entry before the lock.

Design (FUTURE_WORK_V4 §4.2, §5-§7):
- Outer CV: test section t = 1..9; budgets ``budgets.lpp_section_budgets(t)`` (L3 c L5 c L8 over the other 8
  sections); inner decoder CV ``budgets.inner_folds`` (leave one section of B out); nested encoding, ISP, SELF and
  P-AC over half-section groups (run_v4.fold_maps). Scoring: the stride-4 bins of section t (picture bins of
  section 1, 0-80 s, are excluded from training and scoring) with ``src.evaluation.endpoints.score``, the section
  its own session, so the section-macro r is the section's r. R_s(arm) is the mean over the 9 cells.
- Engine: run_v4's Budget, Priors, References, run_cv, fold_maps, arm_weights, soft_weights, build_kernels and
  report, through ``LppIndex`` (TargetIndex's eligibility plus the picture exclusion and the R6 check of the
  targets' recorded context reset), ``LppTarget`` (Target's attributes; ``outer(t)`` makes the fold view),
  ``LppBudget``, ``LppReferences`` (the target's reference pool, above; PCA bases per (reference, t) from the
  reference's sections != t, R3) and ``LppPriors`` (ISP on budget sections, SELF on
  mask.npz coordinates, P-AC from audio_path + TextGrid). ``final_fit`` is run_v4.final_fit's composition plus the
  residualized refit on the same kernels.
- Picture TRs in label-free fits (FIT_ROWS): the PCA bases, ISP (response and reference predictors), SELF, the
  TH_SEL converters and selection material and the XDEC converters use section 1 without its picture TRs, z-scored
  over the kept rows, so no prior, converter or selection is fit on the shared picture viewing. Two consequences
  are part of the rule: (i) these fits derive their half-section groups from the kept rows (``isp.row_groups``,
  ``tanghuth`` converters), so section 1 splits at the middle of its kept rows (TR 40 + 242 // 2 = 161 of 282),
  while the own-encoding, EB-null and ORACLE groups of the delay bins split at n_tr // 2 = TR 141
  (``budgets.half_groups``); (ii) XDEC's test-section decoder input is z-scored over the kept rows as well (the
  picture rows are set to 0; no scored bin reads them through its delays and lags). This reading of the
  registered exclusion needs its PROTOCOL_V4 §4 entry before any prior result is read.
- Residualized endpoint (H2) per outer fold: position + oracle speech-rate nuisance (src.transcripts) ridge-fit on B
  with the registered inner folds and run_v3.feature_ridge_cv's alpha rule; with the selected (arm, gamma, alpha)
  and weights fixed, every decoder arm's ridge is refit on residual B targets (same kernel) and scored against the
  residual test targets (cell block ``residualized``). The nuisance-only decoders (position; oracle speech rate;
  both) are scored as floors (FUTURE_WORK_V4 §9; cell block ``nuisance_floors``, run_v3 stage_controls' rule).
- Controls (FUTURE_WORK_V4 §9, brain-input nulls on the locked decoders): every scored cell carries
  ``test_shift_null`` (``endpoints.shift_null`` of its section: within-section circular shifts >= 40 TR, 2,000
  draws, seed 0, as run_v4's evaluate cells). At the primary cell (dense clock, primary budget) the CONTROL_ARMS
  W0, W1, PRIOR and EB also carry ``controls``: their float64 test prediction (controls/predictions/, digest in
  the cell) and ``rotated_training``, the arm's full pipeline (own encoding, EB null sigma^2, prior calibration and
  shrinkage, weights, inner CV, final fit; the label-free ISP is unchanged) refit with every budget section's
  training targets rotated by half its rows (run_v3 stage_controls' rule) and scored against the true test
  targets, with its own audited provenance log. Once all 9 outer sections of an arm are current,
  ``subject_controls`` writes <stage>/controls/<budget>_<arm>.json: the subject-level within-section shift null of
  R_s (``shift_record``), the wrong-section input control (each section's targets against the next section's
  prediction in section order, cycled to equal length: run_v3's wrong-run rule), the rotated-target R_s and the
  nuisance floors.
- Comparators (src/priors/tanghuth.py, rule SS, own provenance logs): reference models (top-10k voxels by nested W1
  encoding r and, for XDEC, a W1 decoder) are fit with ``tanghuth.fit_reference_model`` on each reference's sections
  != t (the L8 rule) and cached across targets, as are the references' top-voxel section slices. TH_SEL /
  TH_SEL_rand run through run_v4's PRIOR slot (``tanghuth.selection_priors``; ThSelector refits the converters
  inside every inner fold; selection material = ``tanghuth.lpp_unheard``, none at L8, where TH_SEL is not estimable
  and no cell is written). XDEC is ``tanghuth.xdec_predict`` (zero target labels, no inner folds) audited by
  ``tanghuth.audit_errors``. A comparator that raises Module H's ``NotEstimable`` gets a not-estimable cell
  (``not_estimable_cell``), which the primary analysis reads as 'arm not estimable' rather than a missing file and
  which reruns skip. The registered two-reference sensitivity (FUTURE_WORK_V4 §5 TH-SEL step 3, decision
  6) runs as TH_SEL__pair / XDEC__pair (default, scoped arms): TH_SEL / XDEC with the target's pair
  ``tanghuth.sensitivity_pairs(references, 'lpp', target)`` (the first two references in
  sha256('ssltang-v4-thsel-pair:<target>:<reference>') order), sharing TH_SEL's converter alphas (``alpha_memo``);
  their cells (arm TH_SEL / XDEC, ``references`` = the pair, block ``sensitivity``) go to Module H's layout
  <stage>/sensitivity/refs-<R1>-<R2>/cells/t<t>_<budget>_<arm>.json, outside the cells/ folder the primary
  analysis reads.
- K-subset arms (FUTURE_WORK_V4 §5 Tier 2 and §8.4, the R_P(K) curve): ``EB_K<k>_d<j>`` / ``PRIOR_K<k>_d<j>`` are
  EB / PRIOR with ISP from k of the target's references, draw j = 0..4: the first k references in ascending
  sha256('ssltang-v4-lpp-refs:<subject>:<j>:<reference>') (nested over k within a draw; ``reference_subset``).
  DEV k in {1, 2, 4} (8 = EB itself), CONFIRM k in {1, 2, 4, 8, 16} (the whole pool = EB). Their cells carry the
  subset in the ledger, so ``summarize_v4.reliability_fit`` reads K = len(ledger references). They run through
  run_v4.run_cv with every Tier-1 arm of the cell, sharing EB's own-encoding and null maps (``engine_extension``:
  run_v4's module-level fold_maps / arm_weights are swapped for ``lpp_fold_maps`` / ``lpp_arm_weights``, which pass
  every other arm to run_v4 unchanged, for the duration of the call; run_v4's file is not modified). The draw rule
  needs its PROTOCOL_V4 §4 entry before the lock. Aliases on the command line: EB_K and PRIOR_K (the
  partition's registered set).
- Scope: the K-subset arms, the comparators and their pairs (SCOPED) run only at the primary cell (dense clock and
  the primary budget: --primary-budget for develop, the locked plan's for confirm) unless --scope all; the Tier-1
  arms run at every requested clock and budget. The §9 controls are computed at the primary cell only.
- Data binding: every run reads lpp_prepare's records (data/lpp/prepare_provenance*.json). Each H5 and target array
  is identified by its recorded content digest (``output_content``); a file newer than its record, or one without
  a digest, is refused (re-run lpp_prepare). Content is verified, not only timestamps: the target arrays when a
  listener's content is first used, every live H5 read (``LppStore.h5``: bases, ISP/SELF responses, reference
  slices, XDEC input) and every H5 before a delay-row cache is built (``lpp_prepare.array_digest`` equal to the
  record; one check per file state and process). The delay-row caches carry lpp_content.json; PCA bases (their
  projections only), ISP/SELF/ORACLE maps, reference models and reference voxel slices are keyed by the content
  digests and this module's ``code_sha256()``. Cells, g4b/g4c records and g4.json carry ``data_sha256`` (the
  digests of the target and its references), and a G4 verdict counts only under the same code and data.
- Write-once: develop/confirm skip a current cell (same code, configuration, data and stage; at the primary cell a
  CONTROL_ARMS cell must also carry its ``controls``) instead of rewriting it; develop recomputes a stale cell
  (pre-lock only); confirm refuses a stale one.
- Provenance: one log per (t, budget) shared by the Tier-1 and K-subset arms (merged per arm as in run_v4), a
  separate ORACLE log, one log per comparator arm; dataset 'lpp', val_items [], test_items [t], the registered
  folds, budget L3/L5/L8, stage develop/evaluate; every target transcript read is preceded by a ('transcript',
  'context_reset') record for its section (R6). R1-R7 and the registered-scope audit must pass before a cell is
  written.
- Cells: results/v4_lpp/<sub>/<clock>/<stage>/cells/t<t>_<budget>_<arm>.json (+ predictions/), carrying
  ``code_sha256`` = this module's ``code_sha256()`` (LPP_CODE + run_v4.code_sha256()), ``data_sha256``,
  ``reference_pool`` and, at stage evaluate, the lock record (sha256 and git state). Beside them: the pair cells
  (sensitivity/refs-<R1>-<R2>/), the control predictions and per-subject control records (controls/).

Opt-in arms beyond the ds003020 set (see the ``lpp_note`` of their cells):
  PRIOR_f<f>  the noise ladder applied to PRIOR through the engine's PRIOR slot (needed for H3 if PRIOR is advanced;
              confirm adds it whenever the locked plan reads it).
  P_AC        LPP TextGrids carry no phone tier: phoneme rate is dropped (F = 36), which needs a registered
              PROTOCOL_V4 §4 deviation before any P-AC result is read; P_AC is no default arm and its scores are not
              printed.
"""
import argparse
from collections import OrderedDict
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace
import h5py
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_v4  # noqa: E402
import lpp_prepare  # noqa: E402
import run_v3 as v3  # noqa: E402
import run_v4 as v4  # noqa: E402
from src import budgets as bud  # noqa: E402
from src.encoding_weights import encoding_scores  # noqa: E402
from src.evaluation.endpoints import score, shift_null  # noqa: E402
from src.evaluation.metrics import _feature_correlation  # noqa: E402
from src.multikernel_ridge import KernelRidgePath  # noqa: E402
from src.priors import eb as ebm  # noqa: E402
from src.priors import isp as ispm  # noqa: E402
from src.provenance import Provenance  # noqa: E402
from src.runtime import load_config, save_json  # noqa: E402
from src.voxel_kernels import DELAYS, _h5_path, cache_delay_rows, column_stats, verify_cache  # noqa: E402
from src.windows import resolve_path  # noqa: E402

LPP_CODE = ('scripts/run_v4_lpp.py', 'src/priors/tanghuth.py')
ANALYSIS = 'scripts/summarize_v4.py'
LPP_ROOT = Path('data/lpp')
CACHE = Path('data/cache/v4_lpp')
OUT = Path('results/v4_lpp')
LOCK = Path('experiments/v4_lock.json')
PROTOCOL_ID = 'ssltang-v4-lpp'
DATASET = 'ds003643'
VAL = 'session_macro_feature_correlation'
SECTIONS = bud.LPP_SECTIONS
BUDGETS = tuple(bud.LPP_SIZES)
CLOCKS = ('dense', 'stride4')
PRIMARY_CLOCK = 'dense'
CELL_STAGE = {'develop': 'develop', 'confirm': 'evaluate'}
ENGINE_ARM = 'PRIOR'  # run_v4's arm slot carrying a voxel map (TH_SEL masks, the PRIOR ladder).
SELECTION_ARMS = ('TH_SEL', 'TH_SEL_rand')
COMPARATORS = SELECTION_ARMS + ('XDEC',)
PRIOR_LADDER = tuple(f'PRIOR_f{f:g}' for f in v4.LADDER)
SUBSET_PREFIX = 'ssltang-v4-lpp-refs:'
SUBSET_DRAWS = 5
SUBSET_K = {'DEV': (1, 2, 4), 'CONFIRM': (1, 2, 4, 8, 16)}  # Proper subsets: 8 (DEV) and 48 (CONFIRM) are EB/PRIOR.
SUBSET_BASES = ('EB', 'PRIOR')
SUBSET_ARM = re.compile(r'^(?P<base>EB|PRIOR)_K(?P<k>[1-9][0-9]*)_d(?P<draw>[0-9]+)$')
SUBSET_LEVELS = tuple(sorted(set(SUBSET_K['DEV']) | set(SUBSET_K['CONFIRM'])))
SUBSET_ARMS = tuple(f'{b}_K{k}_d{j}' for b in SUBSET_BASES for k in SUBSET_LEVELS for j in range(SUBSET_DRAWS))
SUBSET_ALIASES = {f'{b}_K': b for b in SUBSET_BASES}  # Command-line aliases for the partition's registered set.
ENGINE_ARMS = v4.TIER1 + SUBSET_ARMS  # Arms fit through run_v4.run_cv on the shared Tier-1 log.
PAIR_SUFFIX = '__pair'  # Two-reference sensitivity of TH_SEL / XDEC (cells: arm TH_SEL / XDEC, sensitivity folder).
PAIR_ARMS = tuple(f'{a}{PAIR_SUFFIX}' for a in ('TH_SEL', 'XDEC'))
ARMS = v4.TIER1 + PRIOR_LADDER + SUBSET_ARMS + COMPARATORS + PAIR_ARMS
SCOPED = SUBSET_ARMS + COMPARATORS + PAIR_ARMS  # Run only at the primary cell unless --scope all.
DEFAULT_ARMS = tuple(a for a in v4.TIER1 if a != 'P_AC') + COMPARATORS + PAIR_ARMS
CONTROL_ARMS = ('W0', 'W1', 'PRIOR', 'EB')  # §9 brain-input controls at the primary cell (EB or PRIOR is advanced).
DEVELOP_DEFAULT = DEFAULT_ARMS + ('EB_K',)
CONFIRM_DEFAULT = DEFAULT_ARMS + ('EB_K', 'PRIOR_K')  # Plus every arm the locked plan reads.
REFERENCE_ARMS = v4.REFERENCE_ARMS + PRIOR_LADDER  # Arms reading all references' BOLD through ISP.
FINAL_PRIORS = ('isp', 'isp_mis', 'own_alpha', 'null', 'self', 'pac')
NUISANCE_PURPOSES = ('nuisance_fit', 'nuisance_apply', 'context_reset')
G4_ARM, G4_BUDGET, G4_CLOCK = 'W0', 'L8', 'dense'
G4_MIN_SHIFT_TR, G4_DRAWS, G4_SEED, G4_MIN_PASS = 40, 2000, 0, 6
CONTEXT = {k: lpp_prepare.TARGET_METADATA[k] for k in ('context_reset', 'context_words')}  # R6, as prepared.
CONTENT_FILE = 'lpp_content.json'
FIT_ROWS = ('label-free fits (PCA bases, ISP, SELF, TH_SEL/XDEC converters and TH_SEL selection material) use each '
            'section without its picture TRs (section 1, 0-80 s), z-scored over the kept rows; their half-section '
            'groups split the kept rows at their middle (section 1: TR 40 + 242 // 2 = 161 of 282), while the '
            'own-encoding, EB-null and ORACLE groups of the delay bins split at n_tr // 2 (TR 141); XDEC\'s '
            'test-section decoder input is z-scored over its kept rows (picture rows 0, read by no scored bin)')
POOL_RULE = ("the other listeners of the partition (DEV targets: the DEV listeners; CONFIRM targets: all EN "
             "listeners) that pass the primary analysis's subject-level QC (summarize_v4.registered_pool: a "
             "preparation record, prepare_provenance*.json g4a, with exactly the registered section TR counts); a "
             "listener without one has no prepared data")
PAIR_RULE = ("FUTURE_WORK_V4 §5 TH-SEL step 3 (decision 6): the first two of the target's references in ascending "
             "sha256('ssltang-v4-thsel-pair:<target>:<reference>') (tanghuth.sensitivity_pairs)")
ROTATION_RULE = ('FUTURE_WORK_V4 §9: the full pipeline (own encoding, EB null, prior calibration and shrinkage, '
                 'weights, inner CV, final fit) refit with every budget section\'s training targets rotated by half '
                 'its rows (np.roll by n // 2, run_v3 stage_controls), scored against the true test targets')
WRONG_SECTION_RULE = ("FUTURE_WORK_V4 §9 wrong-section input: each section's test targets against the next section's "
                      "test prediction (section order, cyclic), cycled to equal length (run_v3 stage_controls' "
                      "wrong-run rule); R = mean over sections of the per-section feature r")
NUISANCE_MODELS = (('position', 'position'), ('speech_rate', 'speech_rate'), ('both', ('position', 'speech_rate')))
PAC_NOTE = ('LPP TextGrids have no phone tier: phoneme rate dropped (F = 36, phones=None); a registered '
            'PROTOCOL_V4 §4 deviation is required before any P-AC result is read')
PRIOR_LADDER_NOTE = ('PRIOR[f] = run_v4 PRIOR on tanh(ladder(atanh(ISP), f, ladder_seed(subject, f))): the EB[f] '
                     'ladder applied to PRIOR (H3 under the substitution rule)')
SUBSET_NOTE = ("K-subset arm (FUTURE_WORK_V4 §5 Tier 2, §8.4): the base arm's rule with ISP from the first k "
               "references in ascending sha256('ssltang-v4-lpp-refs:<subject>:<draw>:<reference>')")


# ----------------------------------------------------------------------------------------------- utils
def jsonable(obj):
    """JSON-ready copy: numpy/torch values to Python, tuples to lists, non-finite floats to None."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if torch.is_tensor(obj):
        return jsonable(obj.detach().cpu().numpy())
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(root, name):
    return sha256_bytes((Path(root) / name).read_bytes())


def code_sha256(root=ROOT):
    """Fingerprint every LPP cell carries: run_v4.code_sha256() (the frozen core) plus the LPP_CODE files."""
    digest = hashlib.sha256(v4.code_sha256().encode())
    for name in LPP_CODE:
        path = Path(root) / name
        if not path.is_file():
            raise FileNotFoundError(f'{name} is fingerprinted into every LPP cell (LPP_CODE) but does not exist')
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fingerprints(ctx):
    """The lock fingerprints this runner must match (summarize_v4.FINGERPRINTS)."""
    return dict(code_sha256=ctx.core, lpp_cell_code_sha256=ctx.code,
                lpp_runner_sha256=file_sha(ctx.root, LPP_CODE[0]), tanghuth_sha256=file_sha(ctx.root, LPP_CODE[1]),
                summarize_sha256=file_sha(ctx.root, ANALYSIS), config_sha256=ctx.config)


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def log_json(**fields):
    print(json.dumps(jsonable(fields)), flush=True)


def fmt(value, spec='.4f'):
    return '-' if value is None or not math.isfinite(float(value)) else format(float(value), spec)


def atomic_save(path, array):
    """np.save through a temporary file and an atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npy')
    np.save(tmp, array)
    os.replace(tmp, path)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(jsonable(obj), indent=1, sort_keys=True) + '\n')
    os.replace(tmp, path)


def read_h5(path):
    """An H5 file's stored 'data' array (the array ``lpp_prepare.array_digest`` recorded)."""
    with h5py.File(path, 'r') as handle:
        return handle['data'][()]


def load_raw(raw_root, run, check=None):
    """[n_tr, V] float32 BOLD in the delay-cache representation (raw H5 rows, non-finite values set to 0).

    ``check(stored)`` (``PreparedData.check``) verifies the stored array against its recorded digest first."""
    stored = read_h5(_h5_path(run, raw_root))
    if check is not None:
        check(stored)
    x = np.nan_to_num(np.asarray(stored, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    del stored
    if len(x) != int(run['n_tr']):
        raise ValueError(f"{run['source_key']}: {len(x)} TRs, manifest says {run['n_tr']}")
    return x


def zscore(x):
    """``isp.load_bold``'s per-run z-score of the rows ``x`` (float32; std floor; constant columns exactly 0)."""
    x = np.array(x, dtype=np.float32)
    x -= x.mean(0)
    std = x.std(0)
    x /= np.maximum(std, ispm.STD_FLOOR)
    x[:, std <= ispm.STD_FLOOR] = 0
    return x


# ----------------------------------------------------------------------------------------------- data
class PreparedData:
    """Recorded content of the prepared LPP data: ``output_content`` of lpp_prepare's records
    (``<lpp_root>/prepare_provenance*.json``: the DEV record, DEV re-run and CONFIRM batch records).

    ``digest(rel, path)`` is the content digest of output ``rel`` (e.g. 'raw/<sub>/<s>.hf5'). Records that disagree
    on an output, an output without a digest, or a file newer than the newest record listing it (a preparation
    that did not finish) are refused: such data cannot be tied to the record the lock certifies. The timestamp
    test is only a cheap pre-filter: ``check(rel, path, stored)`` verifies an array read from the file against
    the digest (``lpp_prepare.array_digest``) and ``verified(rel, path, read)`` reads and verifies a file, both
    once per file state (path, size, mtime, ctime, inode; a copy that keeps the old mtime still changes the
    ctime or inode) in this process. Content that differs from the record is refused.
    """

    def __init__(self, lpp_root):
        self.root = Path(lpp_root)
        self.content, self.newest, self.records, self._checked = {}, {}, {}, {}
        for path in sorted(self.root.glob('prepare_provenance*.json')):
            record = json.loads(path.read_text())
            self.records[path.name] = sha256_bytes(path.read_bytes())
            mtime = path.stat().st_mtime_ns
            for rel, digest in (record.get('output_content') or {}).items():
                if self.content.setdefault(rel, digest) != digest:
                    raise SystemExit(f'Preparation records disagree on the content of {rel} ({path.name})')
                self.newest[rel] = max(self.newest.get(rel, mtime), mtime)

    def digest(self, rel, path):
        if rel not in self.content:
            raise SystemExit(f'{rel} has no recorded content digest in {self.root}/prepare_provenance*.json: run '
                             'lpp_prepare with the current code (its record carries output_content) before any LPP run')
        if Path(path).stat().st_mtime_ns > self.newest[rel]:
            raise SystemExit(f'{path} is newer than the preparation record that lists it: the preparation did not '
                             'finish or the file changed since; re-run lpp_prepare')
        return self.content[rel]

    @staticmethod
    def state(path):
        st = Path(path).stat()
        return str(Path(path).resolve()), st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino

    def check(self, rel, path, stored):
        """The recorded digest of ``rel``, once ``stored`` (the array just read from ``path``) is shown to have it."""
        want, key = self.digest(rel, path), self.state(path)
        if self._checked.get(key) != want:
            got = lpp_prepare.array_digest(stored)
            if got != want:
                raise SystemExit(f'{path}: content digest {got} differs from the preparation record ({want}); the '
                                 'file was replaced or restored after its preparation: re-run lpp_prepare')
            self._checked[key] = want
        return want

    def verified(self, rel, path, read):
        """The recorded digest of ``rel`` once ``read(path)`` (the stored array) has been verified against it."""
        want, key = self.digest(rel, path), self.state(path)
        if self._checked.get(key) != want:
            self.check(rel, path, read(path))
        return want


class Context:
    """Configuration, paths and fingerprints of one runner invocation (tests pass temporary roots and codes)."""

    def __init__(self, cfg, device='cuda', lpp_root=LPP_ROOT, cache_root=CACHE, out_root=OUT, root=ROOT,
                 code=None, core=None, tanghuth=None, lock_path=LOCK):
        self.cfg, self.device = cfg, device
        self.lpp_root, self.cache_root, self.out_root = Path(lpp_root), Path(cache_root), Path(out_root)
        self.root, self.lock_path = Path(root), Path(lock_path)
        self._code, self._core, self._tanghuth = code, core, tanghuth
        self.config = v3.config_sha256(cfg)
        self._data, self._content = None, {}
        self.lock_record = None  # confirm: the lock's sha256 and git state, recorded in every evaluate cell.

    @property
    def core(self):
        if self._core is None:
            self._core = v4.code_sha256()
        return self._core

    @property
    def code(self):
        if self._code is None:
            self._code = code_sha256(self.root)
        return self._code

    @property
    def th(self):
        """src.priors.tanghuth (Module H), imported on first use."""
        if self._tanghuth is None:
            from src.priors import tanghuth
            self._tanghuth = tanghuth
        return self._tanghuth

    @property
    def data(self):
        if self._data is None:
            self._data = PreparedData(self.lpp_root)
        return self._data

    def manifest_path(self, subject):
        return self.lpp_root / 'semantic' / str(subject) / 'semantic_manifest.json'

    def content(self, subject):
        """{section: [H5, targets, valid] content digests} of one listener, from the preparation records. The target
        arrays are verified here; an H5's content is verified at every read (``LppStore.h5``) and before a delay-row
        cache is built from it (``LppTarget``)."""
        subject = str(subject)
        if subject not in self._content:
            path = self.manifest_path(subject)
            manifest = json.loads(path.read_text())
            raw_root = raw_root_of(manifest, self.lpp_root)
            out = {}
            for run in manifest['runs']:
                digests = [self.data.digest('raw/' + run['source_key'], raw_root / run['source_key'])]
                for k in ('target_path', 'target_valid_path'):
                    target = Path(resolve_path(run[k], path))
                    digests.append(self.data.verified('semantic/targets/' + target.name, target, np.load))
                out[str(run['story'])] = digests
            self._content[subject] = out
        return self._content[subject]

    def data_sha256(self, subjects):
        """Digest of the recorded content of every listener in ``subjects`` (cells, G4 records, cache keys)."""
        return v4.sha(json.dumps([[s, self.content(s)] for s in sorted({str(x) for x in subjects})]))


def raw_root_of(manifest, lpp_root):
    path = Path(manifest.get('raw_root') or Path(lpp_root) / 'raw')
    return path if path.is_absolute() else Path(lpp_root) / 'raw'


def mask_path_of(manifest, manifest_path, lpp_root):
    for key in ('mask_path', 'normalization_path'):
        if manifest.get(key):
            path = Path(resolve_path(manifest[key], manifest_path))
            if path.is_file():
                return path
    return Path(lpp_root) / 'mask.npz'


class LppIndex:
    """Duck-typed ``src.targets.TargetIndex`` over all 9 sections of one LPP listener (split 'all').

    Eligibility is TargetIndex's: ``t % stride == phase``, a complete window for every cutoff
    (window - 1 - min(cutoffs) <= t <= n_tr - 1 - max(cutoffs)) and a valid target. In addition every bin whose
    per-TR ``picture`` flag is set (section 1, 0-80 s of audio) is excluded from training and scoring. Runs are the
    sections '1'..'9' in order; each needs target_path, target_valid_path and picture_path, and its recorded
    ``semantic_metadata`` must show the registered GPT-2 context reset at the section start (R6: context_reset
    'section_start', context_words 20); ``target_context`` records it.
    ``src.voxel_kernels.cache_delay_rows`` accepts it (cache files ``all_d<d>.npy``, ``all.meta.json``).
    """

    def __init__(self, manifest_path, stride, window=32, phase=0, cutoffs=(3, 4)):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.split, self.window, self.stride, self.phase = 'all', int(window), int(stride), int(phase)
        self.cutoffs = tuple(sorted({int(c) for c in cutoffs}))
        if self.window < 1 or self.stride < 1:
            raise ValueError('Window and stride must be positive')
        if not self.cutoffs or self.cutoffs[0] < 0 or self.cutoffs[-1] >= self.window:
            raise ValueError('Cutoffs must be nonnegative and shorter than the window')
        if not 0 <= self.phase < self.stride:
            raise ValueError('phase must lie in [0, stride)')
        runs = list(self.manifest['runs'])
        stories = sorted(str(r['story']) for r in runs)
        if stories != sorted(SECTIONS) or any(str(r.get('session')) != str(r['story']) for r in runs):
            raise ValueError(f'{manifest_path}: runs must be the sections {list(SECTIONS)} (story = session)')
        if any(r.get('dataset') != DATASET for r in runs) or len({r['subject'] for r in runs}) != 1:
            raise ValueError(f'{manifest_path}: one {DATASET} listener per manifest')
        self.runs = sorted(runs, key=lambda r: SECTIONS.index(str(r['story'])))
        for run in self.runs:  # R6 from the targets' own record, not the runner's claim.
            meta = run.get('semantic_metadata') or {}
            got = {k: meta.get(k) for k in CONTEXT}
            if got != CONTEXT:
                raise ValueError(f"{run['run_id']}: targets must reset the GPT-2 context at every section start (R6, "
                                 f'{CONTEXT}); its semantic_metadata records {got}')
        self.target_context = dict(CONTEXT, cache_keys={str(r['story']): (r.get('semantic_metadata') or {}).get(
            'cache_key') for r in self.runs})
        self.n_features = int(self.runs[0]['n_features'])
        if any(int(r['n_features']) != self.n_features for r in self.runs):
            raise ValueError('All sections must share the same columns')
        self._cache = {}
        self.target_dim = int(self._array(0, 'target_path').shape[1])
        first = self.window - 1 - self.cutoffs[0]
        items, dropped = [], 0
        for ri, run in enumerate(self.runs):
            if self._array(ri, 'target_path').shape[1] != self.target_dim:
                raise ValueError('All sections must share the same semantic target dimension')
            valid, picture = self._array(ri, 'target_valid_path'), self._array(ri, 'picture_path')
            last = int(run['n_tr']) - 1 - self.cutoffs[-1]
            for t in range(first, last + 1):
                if t % self.stride != self.phase or not bool(valid[t]):
                    continue
                if bool(picture[t]):
                    dropped += 1
                    continue
                items.append((ri, t))
        if not items:
            raise ValueError(f'No eligible targets in {manifest_path}')
        self.items, self.picture_excluded = items, dropped
        self.ids = [f"{self.runs[ri]['run_id']}:{t}" for ri, t in items]
        self.run_ids = np.array([self.runs[ri]['run_id'] for ri, _ in items])
        self.target_trs = np.array([t for _, t in items], dtype=np.int64)
        self.sessions = {r['run_id']: str(r['session']) for r in self.runs}

    def _array(self, run_index, kind):
        key = (run_index, kind)
        if key not in self._cache:
            run = self.runs[run_index]
            if not run.get(kind):
                raise ValueError(f"{run['run_id']} lacks {kind}")
            values = np.load(resolve_path(run[kind], self.manifest_path), mmap_mode='r')
            n = int(run['n_tr'])
            if kind == 'target_path':
                if values.ndim != 2 or values.shape[0] != n or values.shape[1] < 1:
                    raise ValueError(f"Targets must align with run TRs: {run['run_id']}")
            elif values.shape != (n,) or values.dtype != np.bool_:
                raise ValueError(f"{kind} must be one boolean per TR: {run['run_id']}")
            self._cache[key] = values
        return self._cache[key]

    def __len__(self):
        return len(self.items)

    def targets(self, positions=None):
        positions = range(len(self.items)) if positions is None else positions
        return np.stack([np.asarray(self._array(self.items[i][0], 'target_path')[self.items[i][1]], dtype=np.float32)
                         for i in positions])

    def signature(self):
        return {'protocol': PROTOCOL_ID, 'split': self.split, 'window': self.window, 'stride': self.stride,
                'phase': self.phase, 'cutoffs': list(self.cutoffs), 'n_targets': len(self.items),
                'picture_excluded': int(self.picture_excluded),
                'target_ids_sha256': hashlib.sha256('\n'.join(self.ids).encode()).hexdigest()}


def bind_cache(index, raw_root, cache_dir, bold, build, verify=None):
    """Delay-row cache paths of ``index`` bound to the recorded H5 content ``bold`` ({section: digest}).

    ``build`` (prepare-caches) rebuilds a cache whose ``lpp_content.json`` differs from ``bold`` (the frozen
    ``verify_cache`` identifies the sources by key and size only), after ``verify()`` has checked every H5's content
    against the record; otherwise such a cache is refused.
    """
    cache_dir = Path(cache_dir)
    sidecar = cache_dir / CONTENT_FILE
    want = dict(rule='src.voxel_kernels delay rows of the H5 files with these recorded contents (lpp_prepare '
                     'output_content)', bold=dict(bold),
                verified='every H5 array_digest equal to its recorded digest before the build')
    bound = sidecar.is_file() and json.loads(sidecar.read_text()) == jsonable(want)
    if build:
        if not bound:
            sidecar.unlink(missing_ok=True)
            (cache_dir / f'{index.split}.meta.json').unlink(missing_ok=True)  # Forces the rebuild.
            if verify is not None:
                verify()
        paths = cache_delay_rows(index, raw_root, cache_dir)[0]
        if not bound:
            atomic_json(sidecar, want)
        return paths
    if not bound:
        raise ValueError(f'{cache_dir}: the delay-row cache is not bound to the recorded H5 content '
                         f'({CONTENT_FILE}); rebuild it with prepare-caches')
    return verify_cache(index, raw_root, cache_dir)[0]


class SectionSplit:
    """``src.probe.Split``'s scoring fields for one outer test section; targets are read on first use."""

    def __init__(self, index, positions):
        self.index, self.positions = index, np.asarray(positions, dtype=np.int64)
        self.run_ids, self.target_trs = index.run_ids[self.positions], index.target_trs[self.positions]
        self.sessions = dict(index.sessions)
        self._targets = None

    @property
    def targets(self):
        if self._targets is None:
            self._targets = self.index.targets(self.positions)
        return self._targets


class LppTarget:
    """run_v4.Target's attributes for one LPP listener on one training clock (all 9 sections).

    ``outer(t)`` returns the view of outer fold t: ``split['test']`` is section t's stride-4 bins, ``eval_pos``
    their rows in this clock's cache, ``split_stories`` = {'train': the other 8 sections, 'val': [], 'test': [t]}.
    """

    def __init__(self, ctx, subject, clock, build_cache=False):
        if clock not in CLOCKS:
            raise ValueError(f'clock must be one of {CLOCKS}')
        data = ctx.cfg['data']
        self.ctx, self.subject, self.clock, self.device = ctx, str(subject), clock, torch.device(ctx.device)
        self.manifest_path = ctx.manifest_path(self.subject)
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get('subject') != self.subject:
            raise ValueError(f'{self.manifest_path} is not the manifest of {self.subject}')
        self.raw_root = raw_root_of(self.manifest, ctx.lpp_root)
        self.cfg = dict(ctx.cfg, raw_root=str(self.raw_root))
        self.stride, self.phase = int(data['stride']), int(data['target_phase'])
        common = dict(window=data['window'], phase=self.phase, cutoffs=data['cutoffs'])
        train = LppIndex(self.manifest_path, 1 if clock == 'dense' else self.stride, **common)
        self.index = {'train': train}
        self.scoring_index = train if clock == 'stride4' else LppIndex(self.manifest_path, self.stride, **common)
        self.dense_index = train if clock == 'dense' else LppIndex(self.manifest_path, 1, **common)
        self.train_dir = ctx.cache_root / self.subject / clock
        self.content = ctx.content(self.subject)

        def verify():  # Every H5 the cache is built from has its recorded content.
            for run in train.runs:
                ctx.data.verified('raw/' + run['source_key'], _h5_path(run, self.raw_root), read_h5)
        self.cache = {'train': bind_cache(train, self.raw_root, self.train_dir,
                                          {s: c[0] for s, c in sorted(self.content.items())}, build_cache, verify)}
        self.row_story = np.array([train.runs[ri]['story'] for ri, _ in train.items])
        self.row_tr = train.target_trs
        self.row_run = train.run_ids
        self.row_session = [train.sessions[r] for r in train.run_ids]
        self.run_sessions = dict(train.sessions)
        self.runs_by_story = {r['story']: r for r in train.runs}
        self.n_tr = {st: int(r['n_tr']) for st, r in self.runs_by_story.items()}
        self.alphas = np.logspace(*ctx.cfg['alphas'])
        self.split_stories = {'train': list(SECTIONS), 'val': [], 'test': []}
        self.split, self.test_section, self.eval_pos = {}, None, None
        lookup = {(r, int(t)): i for i, (r, t) in enumerate(zip(train.run_ids, train.target_trs))}
        scoring_story = np.array([self.scoring_index.runs[ri]['story'] for ri, _ in self.scoring_index.items])
        self._eval = {}
        for s in SECTIONS:
            spos = np.flatnonzero(scoring_story == s)
            tpos = np.array([lookup[(self.scoring_index.run_ids[i], int(self.scoring_index.target_trs[i]))]
                             for i in spos], dtype=np.int64)
            self._eval[s] = (spos, tpos)
        self.mask_sha256 = self.manifest.get('mask_content_sha256') or self.manifest.get('mask_sha256')

    def rows_of(self, stories):
        return np.flatnonzero(np.isin(self.row_story, list(set(stories))))

    def outer(self, t):
        t = str(t)
        if t not in SECTIONS:
            raise ValueError(f'Unknown outer section {t!r}')
        view = copy.copy(self)
        spos, tpos = self._eval[t]
        if not len(spos):
            raise ValueError(f'Section {t} has no scoring bins')
        view.test_section, view.eval_pos = t, tpos
        view.split_stories = {'train': [s for s in SECTIONS if s != t], 'val': [], 'test': [t]}
        view.split = {'test': SectionSplit(self.scoring_index, spos)}
        return view

    def coords(self):
        with np.load(mask_path_of(self.manifest, self.manifest_path, self.ctx.lpp_root)) as z:
            coords = np.asarray(z['coords'])
        if coords.shape != (self.index['train'].n_features, 3):
            raise ValueError(f'mask coords {coords.shape} do not match {self.index["train"].n_features} columns')
        return coords


class LppBudget(v4.Budget):
    """run_v4.Budget on budget sections B of one outer fold; the eval rows are the test section's stride-4 bins.

    ``log`` writes a ('transcript', 'context_reset') record for a section before each of its target transcript
    reads (R6: LppIndex checked that the targets were built with the 20-word GPT-2 context reset at every section
    start).
    """

    def __init__(self, tgt, name, stories, eval_splits=('test',)):
        self.tgt, self.name, self.stories = tgt, name, [str(s) for s in stories]
        if tgt.test_section is None or tgt.test_section in self.stories:
            raise ValueError('An LPP budget needs an outer view whose test section lies outside the budget')
        self.rows = tgt.rows_of(self.stories)
        dev = tgt.device

        def load(pos):
            return {d: torch.from_numpy(np.array(np.load(tgt.cache['train'][d], mmap_mode='r')[pos],
                                                 dtype=np.float32)).to(dev) for d in DELAYS}
        self.raw = load(self.rows)
        self.eval_raw = {s: load(tgt.eval_pos) for s in eval_splits}
        self.y = tgt.index['train'].targets(self.rows).astype(np.float64)
        self.story = tgt.row_story[self.rows]
        self.tr = tgt.row_tr[self.rows]
        self.run = tgt.row_run[self.rows]
        self.session = [tgt.row_session[i] for i in self.rows]

    def log(self, prov, pos, kind, purpose):
        for st in sorted(set(self.story[np.asarray(pos, dtype=np.int64)])):
            if kind == 'transcript' and purpose not in NUISANCE_PURPOSES:
                prov.record(self.tgt.subject, st, 'transcript', 'context_reset')
            prov.record(self.tgt.subject, st, kind, purpose)


class LppStore:
    """Caches of one target run shared by every outer fold and budget (memory), and across targets and clocks
    (disk, under the cache root): target BOLD, reference PCA bases with their section projections, ISP / SELF /
    ORACLE maps, reference models, reference voxel slices, P-AC features and transcripts. Callers log the reads,
    never the store. Every disk key holds the recorded content digests and the code fingerprint (``cached``)."""

    def __init__(self, ctx, tgt, references=(), pool=None):
        self.ctx, self.tgt, self.pool = ctx, tgt, pool  # ``pool``: the reference-pool record (``reference_pool``).
        self.references = sorted(str(r) for r in references)
        if tgt.subject in self.references:
            raise ValueError('A target is never its own reference')
        self.bold, self.bases, self.memo, self.words_, self.pac_, self.models = {}, {}, {}, {}, {}, {}
        self.manifests, self.targets_, self.ref_runs, self.keep_ = {}, {}, OrderedDict(), {}
        self._coords = None

    def manifest(self, subject):
        if subject == self.tgt.subject:
            return self.tgt.manifest
        if subject not in self.manifests:
            self.manifests[subject] = json.loads(self.ctx.manifest_path(subject).read_text())
        return self.manifests[subject]

    def run(self, subject, section):
        return next(r for r in self.manifest(subject)['runs'] if str(r['story']) == str(section))

    def raw_root(self, subject):
        return raw_root_of(self.manifest(subject), self.ctx.lpp_root)

    def source(self, subject, section):
        """The recorded content digest of one listener's section H5."""
        return self.ctx.content(subject)[str(section)][0]

    @property
    def pair(self):
        """The registered two-reference sensitivity pair of the target (``tanghuth.sensitivity_pairs``), or None."""
        if len(self.references) < 2:
            return None
        return list(self.ctx.th.sensitivity_pairs(self.references, 'lpp', self.tgt.subject)[0])

    def h5(self, subject, section):
        """A listener's section in the delay-cache representation [n_tr, V], its content verified against the
        preparation record (``PreparedData.check``)."""
        run, raw_root = self.run(subject, section), self.raw_root(subject)
        rel, path = 'raw/' + run['source_key'], _h5_path(run, raw_root)
        return load_raw(raw_root, run, check=lambda stored: self.ctx.data.check(rel, path, stored))

    @property
    def coords(self):
        if self._coords is None:
            self._coords = self.tgt.coords()
        return self._coords

    def kept(self, subject, section):
        """Rows of a section that label-free fits use (FIT_ROWS): the TRs without the per-TR picture flag."""
        key = (str(subject), str(section))
        if key not in self.keep_:
            run = self.run(subject, section)
            picture = np.load(resolve_path(run['picture_path'], self.ctx.manifest_path(subject)))
            if picture.shape != (int(run['n_tr']),) or picture.dtype != np.bool_:
                raise ValueError(f"picture_path must be one boolean per TR: {run['run_id']}")
            self.keep_[key] = np.flatnonzero(~picture)
        return self.keep_[key]

    def zrun(self, subject, section):
        """A listener's per-run z-scored BOLD of one section over its kept rows [n_kept, V] (FIT_ROWS; without
        picture TRs bitwise ``isp.load_bold``)."""
        x, keep = self.h5(subject, section), self.kept(subject, section)
        return zscore(x if len(keep) == len(x) else x[keep])

    def rawrun(self, subject, section):
        """A listener's section in the delay-cache representation over its kept rows (XDEC converter outputs)."""
        return self.h5(subject, section)[self.kept(subject, section)]

    def target_bold(self, section):
        """The target's per-run z-scored BOLD of one section over its kept rows (ISP, SELF, converters)."""
        section = str(section)
        if section not in self.bold:
            self.bold[section] = self.zrun(self.tgt.subject, section)
        return self.bold[section]

    def target_eval_bold(self, section):
        """XDEC's decoder input at scoring [n_tr, V]: the target's test section z-scored over its kept rows as its
        converters' inputs are (FIT_ROWS; picture rows 0, read by no scored bin's delays and lags)."""
        section = str(section)
        keep = self.kept(self.tgt.subject, section)
        z = self.target_bold(section)
        out = np.zeros((int(self.run(self.tgt.subject, section)['n_tr']), z.shape[1]), dtype=z.dtype)
        out[keep] = z
        return out

    def reference_runs(self, kind, ref, section, voxels, key):
        """A reference section's kept rows restricted to ``voxels``: per-run z-scored (``kind`` 'z') or in the cache
        representation ('raw'); cached on disk by (content, voxels, code) and memoized per (kind, ref, section, key)
        until ``release``."""
        tag = (kind, ref, str(section), key)
        if tag not in self.ref_runs:
            voxels = np.ascontiguousarray(voxels, dtype=np.int64)
            name = v4.sha(json.dumps([kind, ref, str(section), sha256_bytes(voxels.tobytes()),
                                      self.source(ref, section), self.ctx.code, FIT_ROWS]))[:24]
            path = self.ctx.cache_root / 'refsub' / f'{ref}_s{section}_{kind}_{name}.npy'
            if path.exists():
                x = np.load(path)
            else:
                full = self.zrun(ref, section) if kind == 'z' else self.rawrun(ref, section)
                x = np.ascontiguousarray(full[:, voxels])
                del full
                atomic_save(path, x)
            self.ref_runs[tag] = x
        return self.ref_runs[tag]

    def release(self):
        """Drop what one outer section needed (reference runs and models, memoized maps; all cached on disk)."""
        self.ref_runs.clear()
        self.models.clear()
        self.memo.clear()

    def reference_target(self, ref, clock):
        if (ref, clock) not in self.targets_:
            self.targets_[(ref, clock)] = LppTarget(self.ctx, ref, clock)
        return self.targets_[(ref, clock)]

    def basis(self, ref, t):
        """Reference ``ref``'s PCA basis from its sections != t (rule R3; kept rows, FIT_ROWS) and the projections
        of those sections."""
        if (ref, t) in self.bases:
            return self.bases[(ref, t)]
        allowed = [s for s in SECTIONS if s != t]
        sources = [self.source(ref, s) for s in allowed]
        key = v4.sha(json.dumps([ref, allowed, sources, v4.N_COMPONENTS, 0, self.ctx.code, FIT_ROWS]))[:20]
        path = self.ctx.cache_root / 'bases' / f'{ref}_t{t}_{key}.npz'
        if path.exists():
            with np.load(path) as z:
                proj = {s: np.asarray(z[f'proj_{s}']) for s in allowed}
        else:
            runs = [self.zrun(ref, s) for s in allowed]
            mean, comps = ispm.reference_basis(runs, n_components=v4.N_COMPONENTS, seed=0, device=self.ctx.device)
            proj = {s: ispm.project(z, (mean, comps), self.ctx.device) for s, z in zip(allowed, runs)}
            del runs, mean, comps  # Only the projections are ever read back (about 113 MB saved per basis).
            v4.atomic_savez(path, sections=np.array(allowed), **{f'proj_{s}': p for s, p in proj.items()})
        entry = dict(key=key, sections=allowed, proj=proj,
                     minutes=bud.bold_minutes([int(self.run(ref, s)['n_tr']) for s in allowed]))
        self.bases[(ref, t)] = entry
        return entry

    def cached(self, kind, parts, compute):
        """(arrays, info) of a map keyed by ``parts`` (plus the code fingerprint and FIT_ROWS): memory, then
        ``<cache>/<kind>/<subject>_<key>.npz``, else ``compute()`` -> (dict of arrays, JSON-able info), saved."""
        key = v4.sha(json.dumps(jsonable([parts, self.ctx.code, FIT_ROWS]), sort_keys=True))[:24]
        if (kind, key) in self.memo:
            return self.memo[(kind, key)]
        path = self.ctx.cache_root / kind / f'{self.tgt.subject}_{key}.npz'
        if path.exists():
            with np.load(path) as z:
                arrays = {k: np.asarray(z[k]) for k in z.files if k != 'info'}
                info = json.loads(str(z['info']))
        else:
            arrays, info = compute()
            info = jsonable(info)
            v4.atomic_savez(path, info=np.array(json.dumps(info)), **arrays)
        self.memo[(kind, key)] = (arrays, info)
        return arrays, info

    def reference_model(self, ref, t, clock, decoder):
        """(model, info): reference ``ref``'s SS model on its sections != t at ``clock`` (``tanghuth.ReferenceModel``:
        top-10k voxels by nested W1 encoding r; with ``decoder`` its W1 decoder), fit with run_v4's engine under
        the L8 rule (leave-one-section-out decoder folds, half-section nested groups) and cached across targets
        (keyed by the reference's recorded content and this module's code fingerprint)."""
        th = self.ctx.th
        if (ref, t, clock) in self.models:
            model, info = self.models[(ref, t, clock)]
            if model.decoder is not None or not decoder:
                return model, info
        rt = self.reference_target(ref, clock)
        view = rt.outer(t)
        sections = view.split_stories['train']
        meta = json.loads((rt.train_dir / f"{rt.index['train'].split}.meta.json").read_text())

        def path(dec):
            key = v4.sha(json.dumps([ref, clock, t, sections, list(rt.alphas), th.REF_TOP, dec, meta,
                                     self.ctx.data_sha256([ref]), self.ctx.code, self.ctx.config]))[:20]
            return self.ctx.cache_root / 'tanghuth' / 'references' / f'{ref}_{clock}_t{t}_{key}.npz'
        for dec in ((True,) if decoder else (True, False)):
            if path(dec).exists():
                out = th.ReferenceModel.load(path(dec)), dict(cache=path(dec).name, cached=True)
                self.models[(ref, t, clock)] = out
                return out
        bd = LppBudget(view, 'L8', sections, eval_splits=())
        log_json(reference=ref, clock=clock, test_section=t, sections=sections, rows=len(bd.rows), decoder=decoder)
        model = th.fit_reference_model(view, bd, 'L8', decoder=decoder, log_fold=lambda e: log_json(
            reference=ref, test_section=t, held=e['held'], seconds=round(e['seconds'], 1)))
        model.info.update(clock=clock, test_section=t, fold_rule='L8: leave one section out, half-section groups',
                          minutes=bud.bold_minutes([rt.n_tr[s] for s in sections]))
        model.save(path(decoder))
        del bd
        torch.cuda.empty_cache()
        out = model, dict(cache=path(decoder).name, cached=False)
        self.models[(ref, t, clock)] = out
        return out

    def words(self, section):
        from src.transcripts import run_words
        if section not in self.words_:
            self.words_[section] = run_words(self.tgt.runs_by_story[section], self.tgt.manifest_path)
        return self.words_[section]

    def pac(self, section):
        from src.priors import acoustic as acm
        if section not in self.pac_:
            self.pac_[section] = acm.story_features(section, self.tgt.runs_by_story[section], self.tgt.manifest_path,
                                                    phones=None, return_info=True)
        return self.pac_[section]


# --------------------------------------------------------------------------------------------- priors
def reference_subset(subject, references, k, draw):
    """Draw ``draw`` of k references for ``subject``: the first k of ``references`` in ascending
    sha256('ssltang-v4-lpp-refs:<subject>:<draw>:<reference>') (nested over k within a draw), sorted."""
    refs = sorted({str(r) for r in references})
    if not 1 <= int(k) < len(refs):
        raise ValueError(f'A K-subset needs 1 <= k < {len(refs)} references (k = {len(refs)} is the full arm), not {k}')
    return sorted(bud.hash_order(refs, f'{SUBSET_PREFIX}{subject}:{int(draw)}:')[:int(k)])


def subset_key(arm):
    """'K<k>_d<draw>' of a K-subset arm, else None."""
    m = SUBSET_ARM.match(str(arm))
    return None if m is None else f"K{int(m['k'])}_d{int(m['draw'])}"


def subset_arms(partition, n_references, bases=SUBSET_BASES):
    """The registered K-subset arms of a DEV or CONFIRM target (k below its number of references)."""
    return [f'{b}_K{k}_d{j}' for b in bases for k in SUBSET_K[partition] if k < n_references
            for j in range(SUBSET_DRAWS)]


def expand_arms(arms, partition, n_references):
    """``arms`` with the aliases EB_K / PRIOR_K replaced by the partition's registered K-subset arms, deduplicated."""
    out = []
    for arm in arms:
        out += subset_arms(partition, n_references, (SUBSET_ALIASES[arm],)) if arm in SUBSET_ALIASES else [arm]
    return list(dict.fromkeys(out))


class LppReferences(v4.References):
    """run_v4.References over LPP listeners (``subjects``: default every reference of the store): bases per
    (reference, outer section t) from the reference's sections != t (rule R3, logged as ``pca_basis``); projections
    at budget sections logged as ``isp_predictor``."""

    def __init__(self, cfg, target, prov, device, store, subjects=None):
        self.cfg, self.target, self.prov, self.device, self.store = cfg, target, prov, device, store
        self.subjects = sorted(store.references if subjects is None else (str(s) for s in subjects))
        if not self.subjects:
            raise ValueError('ISP needs at least one reference listener')
        if set(self.subjects) - set(store.references):
            raise ValueError(f'{sorted(set(self.subjects) - set(store.references))} are not references of the store')
        self.manifests = {s: store.manifest(s) for s in self.subjects}
        self.excluded = [target.test_section]
        self.proj, self.basis_info = {}, {}
        self.bases = {s: self._basis(s) for s in self.subjects}

    def _basis(self, ref):
        entry = self.store.basis(ref, self.target.test_section)
        for st in entry['sections']:
            self.prov.record(ref, st, 'brain', 'pca_basis')
        self.basis_info[ref] = dict(key=entry['key'], stories=len(entry['sections']), minutes=entry['minutes'])
        return entry['key']

    def projection(self, ref, story):
        self.prov.record(ref, story, 'brain', 'isp_predictor')
        return self.store.basis(ref, self.target.test_section)['proj'][story]


class LppPriors(v4.Priors):
    """run_v4.Priors for an LPP outer fold: all 9 sections are heard by every listener; ISP and SELF maps are cached
    on disk by their inputs (they are clock-independent); P-AC uses phones=None (LPP TextGrids: one word tier).
    ``subsets`` {key: LppReferences} gives the K-subset arms' ISP (``subset_isp``)."""

    def __init__(self, tgt, refs, prov, device, store, subsets=None):
        super().__init__(tgt, refs, prov, device, SECTIONS)
        self.store, self.subsets = store, dict(subsets or {})

    def target_bold(self, story, purpose):
        self.prov.record(self.tgt.subject, story, 'brain', purpose)
        return self.store.target_bold(story)

    def isp(self, stories, misaligned=False, refs=None):
        stories = [s for s in stories if s in self.shared]
        refs = self.refs if refs is None else refs
        subject, t = self.tgt.subject, self.tgt.test_section
        for ref in refs.subjects:
            for s in stories:
                self.prov.record(ref, s, 'brain', 'isp_predictor')
        for s in stories:
            self.prov.record(subject, s, 'brain', 'isp_response')
        parts = ['isp', subject, t, refs.subjects, [refs.basis_info[r]['key'] for r in refs.subjects], stories,
                 bool(misaligned), v4.LAGS, v4.ISP_ALPHAS, [self.store.source(subject, s) for s in stories]]

        def compute():
            x = ispm.design({ref: {s: self.store.basis(ref, t)['proj'][s] for s in stories} for ref in refs.subjects},
                            stories, v4.LAGS, misaligned)
            r, alpha, curve = ispm.isp_scores({s: self.store.target_bold(s) for s in stories}, x, stories,
                                              v4.ISP_ALPHAS, self.device)
            r = np.asarray(r)
            return dict(r=r), dict(alpha=float(alpha), alpha_at_edge=bool(alpha in (v4.ISP_ALPHAS[0],
                                                                                     v4.ISP_ALPHAS[-1])),
                                   stories=stories, references=list(refs.subjects), positive=float((r > 0).mean()),
                                   curve=np.asarray(curve).tolist())
        arrays, info = self.store.cached('isp', parts, compute)
        return arrays['r'], info

    def subset_isp(self, key, stories):
        """ISP from the references of K-subset ``key`` ('K<k>_d<draw>')."""
        return self.isp(stories, refs=self.subsets[key])

    def self_prior(self, stories):
        stories, subject = list(stories), self.tgt.subject
        for s in stories:
            self.prov.record(subject, s, 'brain', 'isp_response')
        coords = self.store.coords
        parts = ['self', subject, stories, v4.N_COMPONENTS, 4, v4.LAGS, v4.ISP_ALPHAS,
                 hashlib.sha256(np.ascontiguousarray(coords, dtype=np.int64).tobytes()).hexdigest(),
                 [self.store.source(subject, s) for s in stories]]

        def compute():
            r, info = ispm.self_scores({s: self.store.target_bold(s) for s in stories}, coords, stories,
                                       n_components=v4.N_COMPONENTS, margin=4, alphas=v4.ISP_ALPHAS, device=self.device)
            return dict(r=np.asarray(r)), dict(stories=stories, octants=info)
        arrays, info = self.store.cached('self', parts, compute)
        return arrays['r'], info

    def acoustic_features(self, bd, fit_pos):
        from src.priors import acoustic as acm
        per_story, info = {}, {}
        for st in sorted(set(bd.story[fit_pos])):
            self.prov.record(self.tgt.subject, st, 'audio', 'own_encoding')
            self.prov.record(self.tgt.subject, st, 'transcript', 'context_reset')
            self.prov.record(self.tgt.subject, st, 'transcript', 'own_encoding')  # TextGrid word timing.
            per_story[st], info[st] = self.store.pac(st)
        feats = acm.acoustic_rows(per_story, list(bd.story), bd.tr, rows=fit_pos)
        return feats, {s: {k: v for k, v in i.items() if isinstance(v, (str, int, float))} for s, i in info.items()}


class LadderPriors(LppPriors):
    """PRIOR[f] through run_v4's PRIOR slot: each fit's ISP map with the registered ladder at reliability f
    (``isp.ladder`` on the Fisher scale, seed ``isp.ladder_seed(subject, f)``), returned on the r scale."""

    def __init__(self, tgt, refs, prov, device, store, f):
        super().__init__(tgt, refs, prov, device, store)
        self.f = float(f)

    def isp(self, stories, misaligned=False, refs=None):
        r, info = super().isp(stories, misaligned, refs)
        p = ispm.ladder(ebm.fisher(r), self.f, ispm.ladder_seed(self.tgt.subject, self.f))
        return np.tanh(p), dict(info, ladder=self.f)


def oracle_scores(view, store, prov):
    """ORACLE (descriptive): W1 nested out-of-fold r from all 8 non-test sections (half-section groups); the reads
    are 'oracle' (over budget by design, flagged) and never touch the outer test section."""
    sections, subject = view.split_stories['train'], view.subject
    for st in sections:
        prov.record(subject, st, 'transcript', 'context_reset')
        prov.record(subject, st, 'transcript', 'oracle')
        prov.record(subject, st, 'brain', 'oracle')
    meta = json.loads((view.train_dir / f"{view.index['train'].split}.meta.json").read_text())
    parts = ['oracle', subject, view.clock, view.test_section, sections, view.alphas, meta,
             store.ctx.data_sha256([subject])]

    def compute():
        rows = view.rows_of(sections)
        groups = bud.half_groups(view.row_story[rows], view.row_tr[rows], view.n_tr)
        folds = [(g,) for g in sorted(set(groups))]
        y = view.index['train'].targets(rows).astype(np.float64)
        out, alphas = {}, {}
        for d in DELAYS:
            x = torch.from_numpy(np.array(np.load(view.cache['train'][d], mmap_mode='r')[rows],
                                          dtype=np.float32)).to(view.device, torch.float64)
            mean, std = column_stats(x, torch.arange(len(rows), device=x.device))
            x.sub_(mean).div_(std)
            r, alpha, _ = encoding_scores(y, x, list(range(len(rows))), groups, folds, view.alphas, view.device)
            out[f'd{d}'], alphas[str(d)] = np.asarray(r), float(alpha)
            del x
            torch.cuda.empty_cache()
        return out, dict(sections=sections, rows=len(rows), groups='half-section', alpha=alphas)
    arrays, _ = store.cached('oracle', parts, compute)
    return {d: arrays[f'd{d}'] for d in DELAYS}


# --------------------------------------------------------------------------------------------- engine
ENGINE = dict(fold_maps=v4.fold_maps, arm_weights=v4.arm_weights)  # run_v4's own functions, captured at import.


def lpp_fold_maps(tgt, bd, name, fit_pos, session_folds, held_session, arms, pri, oracle):
    """run_v4.fold_maps plus one ISP map per K-subset of ``arms`` (``isp_K<k>_d<j>``, from ``pri.subset_isp``).

    run_v4.fold_maps reads K-subset names only through its ``startswith('EB')`` test, so an EB_K arm gets the
    own-encoding and null maps (shared with EB and W1) and nothing else; every other map is run_v4's.
    """
    maps, info = ENGINE['fold_maps'](tgt, bd, name, fit_pos, session_folds, held_session, arms, pri, oracle)
    keys = list(dict.fromkeys(k for k in map(subset_key, arms) if k))
    if keys:
        fit_stories = [s for s in bd.stories if s in set(bd.story[fit_pos])]
        for key in keys:
            maps[f'isp_{key}'], info[f'isp_{key}'] = pri.subset_isp(key, fit_stories)
    return maps, info


def lpp_arm_weights(arm, maps, subject):
    """run_v4.arm_weights; a K-subset arm is its base arm (EB or PRIOR) on its subset's ISP map."""
    m = SUBSET_ARM.match(str(arm))
    if m is None:
        return ENGINE['arm_weights'](arm, maps, subject)
    src = maps.get(f'isp_{subset_key(arm)}')
    return None if src is None else ENGINE['arm_weights'](m['base'], dict(maps, isp=src), subject)


@contextmanager
def engine_extension():
    """run_v4.run_cv with the K-subset arms: run_v4's module-level ``fold_maps`` and ``arm_weights`` (looked up by
    run_cv at call time) are ``lpp_fold_maps`` / ``lpp_arm_weights`` inside the block and run_v4's own functions
    again after it, also on error. run_v4's file is not modified (its fingerprint is unchanged)."""
    v4.fold_maps, v4.arm_weights = lpp_fold_maps, lpp_arm_weights
    try:
        yield
    finally:
        v4.fold_maps, v4.arm_weights = ENGINE['fold_maps'], ENGINE['arm_weights']


def final_fit(view, bd, name, arms, selections, pri, prov, oracle=None, residual=None):
    """run_v4.final_fit on the outer test section (same fold_maps, arm_weights, build_kernels, KernelRidgePath),
    plus the residualized refit: with ``residual`` (residual budget targets) every arm's ridge is also fit on them
    with the same kernel at its selected alpha. Returns ({arm: ({'test': pred[, 'residualized': pred]},
    diagnostics)}, info with 'weights')."""
    all_pos = np.arange(len(bd.rows))
    maps, info = lpp_fold_maps(view, bd, name, all_pos, [], None, arms, pri, oracle)
    sets, meta = {}, {}
    for arm in arms:
        sel = selections.get(arm)
        if sel is None:
            continue
        ws = lpp_arm_weights(arm, maps, view.subject)
        if ws is None or sel['gamma'] not in ws:
            continue
        sets[arm], meta[arm] = ws[sel['gamma']]
        if 'eb' in ws:
            meta[arm] = dict(meta[arm], eb=ws['eb'])
    y = torch.as_tensor(bd.y, dtype=torch.float64, device=view.device)
    yr = None if residual is None else torch.as_tensor(residual, dtype=torch.float64, device=view.device)
    predictions, keys = {}, list(sets)
    for g0 in range(0, len(keys), v4.GROUP):
        group = {k: sets[k] for k in keys[g0:g0 + v4.GROUP]}
        kernels, cross = v4.build_kernels(bd, all_pos, group, all_pos, ('test',), prov=prov)
        bd.log(prov, all_pos, 'brain', 'decoder_fit')
        bd.log(prov, all_pos, 'transcript', 'decoder_fit')
        for arm in list(kernels):
            k, c, alpha = kernels.pop(arm), cross[arm]['test'], selections[arm]['alpha']
            path = KernelRidgePath(k, y, view.device)
            entry = dict(test=path.predict(path.project(c), alpha).cpu().numpy())
            if yr is not None:
                rpath = KernelRidgePath(k, yr, view.device)
                entry['residualized'] = rpath.predict(rpath.project(c), alpha).cpu().numpy()
                del rpath
            predictions[arm] = (entry, path.diagnostics)
            del k, path
        del kernels, cross
        torch.cuda.empty_cache()
    log_scoring(prov, view)
    return predictions, dict(info, weights=meta)


def log_scoring(prov, view):
    t = view.test_section
    prov.record(view.subject, t, 'brain', 'decoder_score')
    prov.record(view.subject, t, 'transcript', 'context_reset')
    prov.record(view.subject, t, 'transcript', 'decoder_score')


def report_against(view, y_train, prediction, targets):
    """v2 endpoints of ``prediction`` against ``targets`` on the test section (train statistics of ``y_train``)."""
    mean, std = y_train.mean(0), np.maximum(y_train.std(0), 1e-6)
    sp = view.split['test']
    return score(prediction, targets, sp.run_ids, sp.target_trs, sp.sessions, mean, std,
                 view.cfg['evaluation']['exclusions'])


# ------------------------------------------------------------------------------------- residualization
def sub_index(index, positions):
    """``nuisance_features``' view (runs, items) of ``index`` restricted to ``positions``, in that order."""
    positions = np.asarray(positions, dtype=np.int64)
    used = list(dict.fromkeys(int(index.items[i][0]) for i in positions))
    local = {ri: k for k, ri in enumerate(used)}
    return SimpleNamespace(runs=[index.runs[ri] for ri in used],
                           items=[(local[int(index.items[i][0])], int(index.items[i][1])) for i in positions])


def nuisance(view, bd, store):
    """Position + oracle speech-rate nuisance predictions (FUTURE_WORK_V4 §6 H2; PROTOCOL_V3 §6/§12) and the
    nuisance-only decoder floors (FUTURE_WORK_V4 §9).

    ``src.transcripts.nuisance_features`` of the budget rows and the test bins (transcripts of B and t only);
    ``run_v3.feature_ridge_cv`` on the budget rows alone, whose CV folds are then the registered inner folds (one
    section each) and whose standardizers see only each fold's fitting rows; one model per NUISANCE_MODELS group
    (position; oracle speech rate; both). Returns dict(train, test, info, floors): the residual budget and test
    targets of the 'both' model, its fit record, and every model's test score as a decoder (``floors``; run_v3
    stage_controls' nuisance decoders). Reads: ``log_nuisance``.
    """
    from src.transcripts import nuisance_features
    t, test = view.test_section, view.split['test']
    words = {view.runs_by_story[s]['run_id']: store.words(s) for s in list(bd.stories) + [t]}
    shim = SimpleNamespace(row_sessions=list(bd.session), fixed=[[s] for s in bd.stories],
                           index={'train': SimpleNamespace(run_ids=np.asarray(bd.run))}, alphas=view.alphas,
                           run_sessions=view.run_sessions)
    targets = np.asarray(test.targets, dtype=np.float64)
    fits, floors = {}, {}
    for name, groups in NUISANCE_MODELS:  # 'both' last: its fit is the residualization.
        x_train, names = nuisance_features(sub_index(view.index['train'], bd.rows), words, groups=groups)
        x_test, _ = nuisance_features(sub_index(view.scoring_index, test.positions), words, groups=groups)
        try:
            preds, info = v3.feature_ridge_cv(shim, x_train, list(range(len(bd.rows))), bd.y,
                                              {'train': x_train, 'test': x_test})
        except IndexError:  # Every CV score undefined (e.g. constant features): a floor is then not estimable.
            if name == 'both':
                raise
            floors[name] = dict(estimable=False, features=list(names), oracle='speech_rate' in groups,
                                reason='no finite inner-CV score (constant predictions)')
            continue
        fits[name] = (preds, info, list(names))
        floors[name] = dict(info, estimable=True, features=list(names), oracle='speech_rate' in groups,
                            test=report_against(view, bd.y, preds['test'], targets))
    preds, info, names = fits['both']
    return dict(train=bd.y - preds['train'], test=targets - preds['test'],
                info=dict(info, rule='run_v3.feature_ridge_cv on B; CV folds = the registered inner folds',
                          folds=[[s] for s in bd.stories], features=names),
                floors=dict(rule='nuisance-only decoders (FUTURE_WORK_V4 §9 floors): run_v3.feature_ridge_cv on B '
                                 'per feature group, scored against the raw test targets', models=floors))


def log_nuisance(prov, view, bd, story_folds):
    """Transcript reads of the nuisance model: per inner fold (fit on its training part, applied to its held part),
    then the final fit on B, applied to B and to the test section."""
    subject, t = view.subject, view.test_section
    for i, held in enumerate(story_folds):
        with prov.inner_fold(i):
            for s in bd.stories:
                prov.record(subject, s, 'transcript', 'nuisance_apply' if s in held else 'nuisance_fit')
    for s in bd.stories:
        prov.record(subject, s, 'transcript', 'nuisance_fit')
        prov.record(subject, s, 'transcript', 'nuisance_apply')
    prov.record(subject, t, 'transcript', 'nuisance_apply')


# ------------------------------------------------------------------------------------------------ cells
def reads_references(arm):
    """Arms whose ISP reads reference BOLD: the registered reference arms (all references) and the K-subsets."""
    return arm in REFERENCE_ARMS or subset_key(arm) is not None


def lpp_ledger(view, bd, store, arm, refs=None, comparator=None):
    """FUTURE_WORK_V4 §3 ledger of one LPP cell (run_v4.ledger_for on the LPP index). ``refs`` are the references
    the arm's ISP used (a K-subset arm: its subset); ``comparator`` = dict(references, minutes, labeled_pairs) adds
    the SS reference exposure of TH_SEL / XDEC."""
    data = view.cfg['data']
    stride, phase = int(data['stride']), int(data['target_phase'])
    stories = view.split_stories['train'] if arm == 'ORACLE' else list(bd.stories)
    keep, dense = set(stories), view.dense_index
    minutes = bud.bold_minutes([view.n_tr[s] for s in stories])
    ledger = bud.Ledger()
    if arm == 'XDEC':  # Zero target labels: target BOLD at B (converter) and the test section only.
        ledger.target(0, 0, minutes)
    else:
        pairs = [(dense.runs[ri]['run_id'], int(t)) for ri, t in dense.items if dense.runs[ri]['story'] in keep]
        own = pairs if view.clock == 'dense' else [p for p in pairs if p[1] % stride == phase]
        ledger.target(len(pairs), int(sum(p[1] % stride == phase for p in pairs)), minutes)
        ledger.transcript({view.runs_by_story[s]['run_id']: store.words(s) for s in stories}, own)
    if refs is not None and reads_references(arm):
        isp_minutes = len(refs.subjects) * bud.bold_minutes([view.n_tr[s] for s in bd.stories])
        ledger.reference(refs.subjects, isp_minutes + sum(v['minutes'] for v in refs.basis_info.values()))
    if comparator is not None:
        ledger.reference(comparator['references'], comparator['minutes'], comparator['labeled_pairs'], 'SS')
    return ledger.to_dict()


def comparator_exposure(view, models, read_sections):
    """Reference exposure of TH_SEL / XDEC: each reference's BOLD minutes over its model's labeled sections and
    ``read_sections`` (converter and selection material), and the models' labeled rows."""
    minutes = sum(bud.bold_minutes([view.n_tr[s] for s in sorted(set(m.stimuli) | set(read_sections))])
                  for m in models.values())
    return dict(references=sorted(models), minutes=minutes,
                labeled_pairs=int(sum(int(m.info['labeled_rows']) for m in models.values())))


def base_cell(c, arm, **fields):
    view = c.view
    cell = dict(protocol='ssltang-v4', dataset='lpp', stage=c.stage, subject=view.subject, clock=view.clock,
                test_section=view.test_section, budget=c.name, arm=arm, regime='S', estimable=False,
                code_sha256=c.ctx.code, core_code_sha256=c.ctx.core, config_sha256=c.ctx.config,
                data_sha256=c.data, mask_sha256=view.mask_sha256, targets=view.index['train'].target_context,
                fit_rows=FIT_ROWS)
    if c.store.pool is not None:
        cell['reference_pool'] = c.store.pool
    if c.stage == 'evaluate':
        cell['lock'] = c.ctx.lock_record
    cell.update(fields)
    if arm == 'P_AC':
        cell['lpp_note'] = PAC_NOTE
    elif arm in PRIOR_LADDER:
        cell['lpp_note'] = PRIOR_LADDER_NOTE
    elif subset_key(arm) is not None:
        cell['lpp_note'] = SUBSET_NOTE
    return cell


def section_shift_null(view, prediction):
    """``endpoints.shift_null`` of a test section's prediction (within-section circular shifts of at least
    evaluation.shift_min_tr TR, shift_draws draws, shift_seed; the registered 40 / 2,000 / 0), as run_v4 records
    ``test_shift_null`` in its evaluate cells."""
    ev, sp = view.cfg['evaluation'], view.split['test']
    return shift_null(prediction, sp.targets, sp.run_ids, sp.sessions, int(view.cfg['data']['stride']),
                      ev['shift_min_tr'], ev['shift_draws'], ev['shift_seed'])


def add_scores(view, bd, cell, prediction, residual):
    entry, diag = prediction
    cell.update(estimable=True, final_fit=diag, test=v4.report(view, bd, entry['test'], 'test'),
                test_shift_null=section_shift_null(view, entry['test']))
    if residual is not None:
        cell['nuisance_floors'] = residual['floors']
    if residual is not None and 'residualized' in entry:
        cell['residualized'] = dict(test=report_against(view, residual['train'], entry['residualized'],
                                                        residual['test']),
                                    nuisance=residual['info'],
                                    rule='targets minus the B-fit nuisance (position + oracle speech rate); selected '
                                         '(arm, gamma, alpha) and weights fixed; ridge refit on residual B targets')
    return cell


def scope_audit(cell, log):
    return audit_v4.cell_errors(cell) + audit_v4.scope_errors(cell, log)


def cell_path(folder, t, name, arm):
    return Path(folder) / 'cells' / f't{t}_{name}_{arm}.json'


def pair_folder(folder, pair):
    """Module H's sensitivity layout: <stage folder>/sensitivity/refs-<R1>-<R2>."""
    return Path(folder) / 'sensitivity' / ('refs-' + '-'.join(pair))


def cell_state(ctx, path, data, stage, require=()):
    """'missing', 'current' (same code, configuration, data and stage, and every ``require`` key present: never
    rewritten) or 'stale'."""
    path = Path(path)
    if not path.is_file():
        return 'missing'
    try:
        cell = json.loads(path.read_text())
    except ValueError:
        return 'stale'
    same = (cell.get('code_sha256') == ctx.code and cell.get('config_sha256') == ctx.config
            and cell.get('data_sha256') == data and cell.get('stage') == stage and all(k in cell for k in require))
    return 'current' if same else 'stale'


def finish(cell, log, folder, name, arm, prediction=None, auditor=scope_audit):
    """R1-R7 (always) and the registered-scope audit, then write the cell (and its float16 predictions).

    Any audit error refuses a cell that carries a score; a cell without a score records its errors instead. At
    stage evaluate nothing about the scores is printed (endpoint-blind QC precedes any CONFIRM endpoint).
    """
    log.check()
    cell['provenance'] = log.to_dict()  # Every read up to now, scoring included.
    errors = auditor(cell, log)
    if errors and cell.get('estimable'):
        raise SystemExit(f"Registered-scope audit failed for t{cell['test_section']} {name} {arm}: {errors}")
    if errors:
        cell['scope_errors'] = errors  # Not estimable: no score is written.
    t = cell['test_section']
    if prediction is not None:
        (folder / 'predictions').mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'predictions' / f't{t}_{name}_{arm}.npz',
                            **{k: np.asarray(v).astype(np.float16) for k, v in prediction.items()})
    save_json(jsonable(cell), cell_path(folder, t, name, arm))
    sel = cell.get('selection') or {}
    head = f"{cell['subject']} {cell['clock']} t{t} {name} {arm}:"
    if cell.get('stage') == 'evaluate':
        print(f"{head} estimable={bool(cell.get('estimable'))} gamma={sel.get('gamma', '-')} "
              f"alpha={fmt(sel.get('alpha'), '.3g')}", flush=True)
    elif cell.get('estimable') and arm != 'P_AC':  # P-AC needs a registered deviation before it is read.
        shown = [f"test={fmt(cell['test'].get(VAL))}"]
        if 'residualized' in cell:
            shown.append(f"residualized={fmt(cell['residualized']['test'].get(VAL))}")
        print(f"{head} gamma={sel.get('gamma', '-')} alpha={fmt(sel.get('alpha'), '.3g')} "
              f"cv={fmt(sel.get('cv_score'))} {' '.join(shown)}", flush=True)
    return cell


class Cell:
    """The (outer section, budget) context shared by every arm's fit. ``controls``: this is the primary cell, where
    the CONTROL_ARMS carry the §9 controls; ``alpha_memo`` is shared by every TH_SEL selector of the cell (the
    sensitivity pair reuses TH_SEL's converter alphas)."""

    def __init__(self, ctx, view, store, name, stage, folder, controls=False):
        self.ctx, self.view, self.store, self.name, self.stage, self.folder = ctx, view, store, name, stage, folder
        self.subject, self.t, self.controls, self.alpha_memo = view.subject, view.test_section, bool(controls), {}
        self.budget = bud.lpp_section_budgets(self.t, SECTIONS)[name]
        self.folds = [tuple(f) for f in bud.inner_folds(name, self.budget)]
        self.scope = (self.subject, 'S', self.budget, [], [self.t])
        self.data = ctx.data_sha256([self.subject] + list(store.references))

    def path(self, arm):
        """Where ``arm``'s cell goes: cells/ or, for a pair arm, the pair's sensitivity folder (cell arm without
        the suffix); None for a pair arm without a pair."""
        if arm in PAIR_ARMS:
            pair = self.store.pair
            return None if pair is None else cell_path(pair_folder(self.folder, pair), self.t, self.name,
                                                       arm[:-len(PAIR_SUFFIX)])
        return cell_path(self.folder, self.t, self.name, arm)

    def require(self, arm):
        """Keys a current cell of ``arm`` must carry here (the controls of a CONTROL_ARMS cell at the primary cell)."""
        return ('controls',) if self.controls and arm in CONTROL_ARMS else ()

    def prov(self, arm=None, reference_rule=None, folds=True):
        return Provenance(*self.scope, folds=self.folds if folds else None, budget_name=self.name, stage=self.stage,
                          arm=arm, reference_rule=reference_rule)

    def log_fold(self, entry):
        log_json(test_section=self.t, budget=self.name, held=entry['held'], seconds=round(entry['seconds'], 1))


def run_budget(ctx, view, store, name, arms, stage, folder, residualize=True, overwrite=False, controls=False):
    """Every requested arm at one (outer section, budget): inner CV, final fit, scores and audited cells.

    A current cell (``cell_state``) is skipped, never rewritten; a stale one is recomputed at stage develop and
    refused at stage evaluate. ``overwrite`` (G4(b), which needs the predictions) recomputes every arm.
    ``controls`` (the primary cell): the CONTROL_ARMS cells carry the §9 controls (``fit_controls``).
    Returns {arm: prediction entry} of the estimable arms fit here (``'test'`` and, when residualized,
    ``'residualized'``; pair arms under their own names).
    """
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise ValueError(f'Unknown LPP arms {unknown}')
    c = Cell(ctx, view, store, name, stage, folder, controls)
    if not overwrite:
        state = {a: 'missing' if c.path(a) is None else cell_state(ctx, c.path(a), c.data, stage, c.require(a))
                 for a in arms}
        stale = [a for a, s in state.items() if s == 'stale']
        if stale and stage == 'evaluate':
            raise SystemExit(f'{c.subject} t{c.t} {name}: evaluate cells {stale} exist from other code, configuration '
                             'or data (or without their controls); evaluate cells are written once')
        done = [a for a, s in state.items() if s == 'current']
        if done:
            log_json(subject=c.subject, clock=view.clock, test_section=c.t, budget=name, current=done)
        arms = [a for a in arms if state[a] != 'current']
        if not arms:
            return {}
    engine = [a for a in ENGINE_ARMS if a in arms]
    ladder = [a for a in PRIOR_LADDER if a in arms]
    selection = [a for a in SELECTION_ARMS if a in arms]
    log_json(subject=c.subject, clock=view.clock, test_section=c.t, budget=name, sections=c.budget, arms=list(arms))
    bd = LppBudget(view, name, c.budget)
    residual = nuisance(view, bd, store) if residualize and (engine or ladder or selection or
                                                             'TH_SEL__pair' in arms) else None
    out = {}
    if engine or ladder:
        prov = c.prov()
        refs = LppReferences(view.cfg, view, prov, ctx.device, store) if set(arms) & set(REFERENCE_ARMS) else None
        subsets = {}
        for arm in engine:
            key, m = subset_key(arm), SUBSET_ARM.match(arm)
            if key is not None and key not in subsets:
                subsets[key] = LppReferences(view.cfg, view, prov, ctx.device, store, subjects=reference_subset(
                    c.subject, store.references, int(m['k']), int(m['draw'])))
        if residual is not None:
            log_nuisance(prov, view, bd, c.folds)
        if engine:
            out.update(fit_engine(c, bd, engine, prov, refs, subsets, residual))
        for arm in ladder:
            out.update(fit_ladder(c, bd, arm, prov, refs, residual))
    decoder = 'XDEC' in arms or 'XDEC__pair' in arms  # Reference models fit once, with their decoders.
    comparators = [(a, lambda a=a: fit_selection(c, bd, a, residual, decoder=decoder)) for a in selection]
    comparators += [('XDEC', lambda: fit_xdec(c, bd))] if 'XDEC' in arms else []
    comparators += [('TH_SEL__pair', lambda: fit_selection(c, bd, 'TH_SEL', residual, decoder=decoder, pair=True))
                    ] if 'TH_SEL__pair' in arms else []
    comparators += [('XDEC__pair', lambda: fit_xdec(c, bd, pair=True))] if 'XDEC__pair' in arms else []
    for arm, fit in comparators:  # TH_SEL before its pair: the pair reuses its converter alphas (c.alpha_memo).
        try:
            out.update(fit())
        except ctx.th.NotEstimable as error:  # Structural (Module H): logged, a not-estimable cell, the run goes on.
            log_json(test_section=c.t, budget=name, arm=arm, not_estimable=str(error))
            not_estimable_cell(c, bd, arm, error)
    torch.cuda.empty_cache()
    return out


def not_estimable_cell(c, bd, arm, error):
    """The cell of a comparator that raised Module H's ``NotEstimable`` here: estimable False with the reason, the
    arm's references in its ledger and an (empty) log of its own whose registered-scope errors are recorded
    (``finish``). The primary analysis's QC then reads 'arm not estimable' (an unregistered exclusion that needs a
    recorded PROTOCOL_V4 §4 deviation) rather than a 'missing file', and reruns skip the cell like any current cell
    instead of refitting the comparator. Structural absences that write no cell by design (TH_SEL without selection
    material, e.g. at L8, or with fewer than 2 references) never raise and are unaffected."""
    base = arm[:-len(PAIR_SUFFIX)] if arm in PAIR_ARMS else arm
    path = c.path(arm)
    if path is None:
        return None
    references = (c.store.pair if arm in PAIR_ARMS else c.store.references) if base in ('TH_SEL', 'XDEC') else []
    rule = 'SS' if references else None
    ledger = bud.Ledger(references=list(references), reference_rule=rule or 'none')
    if base == 'XDEC':
        ledger.target(0, 0, bud.bold_minutes([c.view.n_tr[s] for s in bd.stories]))
    extra = dict(references=list(references), reference_rule=rule) if references else {}
    if arm in PAIR_ARMS:
        extra['sensitivity'] = dict(rule=PAIR_RULE, pair=list(references), pool=list(c.store.references))
    cell = base_cell(c, base, not_estimable=str(error), ledger=ledger.to_dict(), **extra)
    if base == 'XDEC':
        cell['target_labels'] = 0
    prov = c.prov(arm=base, reference_rule=rule, folds=base != 'XDEC')
    return finish(cell, prov, path.parent.parent, c.name, base, auditor=c.ctx.th.audit_errors)


def fit_engine(c, bd, arms, prov, refs, subsets, residual):
    """run_v4's Tier-1 arms and the K-subset arms in one run_v4.run_cv on the shared log (``engine_extension``;
    ORACLE with its own 'oracle' log merged in)."""
    view, store, ctx = c.view, c.store, c.ctx
    pri = LppPriors(view, refs, prov, ctx.device, store, subsets)
    oracle_prov = c.prov(arm='ORACLE') if 'ORACLE' in arms else None
    oracle = oracle_scores(view, store, oracle_prov) if oracle_prov is not None else None
    with engine_extension():
        selections, curves, logs = v4.run_cv(view, bd, c.name, arms, pri, prov, oracle, c.folds, None, c.log_fold)
    predictions, final = final_fit(view, bd, c.name, arms, selections, pri, prov, oracle,
                                   None if residual is None else residual['train'])
    control = [a for a in arms if a in CONTROL_ARMS] if c.controls else []
    controls = fit_controls(c, bd, control, predictions) if control else {}
    out = {}
    for arm in arms:
        log = prov.merged(oracle_prov, arm='ORACLE') if arm == 'ORACLE' else prov.merged(arm=arm)
        key = subset_key(arm)
        arm_refs = subsets[key] if key is not None else refs if arm in REFERENCE_ARMS else None
        extra = {}
        if key is not None:
            m = SUBSET_ARM.match(arm)
            extra = dict(base_arm=m['base'], reference_subset=dict(
                k=int(m['k']), draw=int(m['draw']), references=list(arm_refs.subjects), pool=list(store.references),
                rule=f"first k of the pool in ascending sha256('{SUBSET_PREFIX}<subject>:<draw>:<reference>')"))
        priors = {k: v for k, v in final.items() if k in FINAL_PRIORS or (key is not None and k == f'isp_{key}')}
        cell = base_cell(c, arm, selection=selections.get(arm), cv_curves=curves.get(arm), folds=logs,
                         final_priors=priors, final_weights=final['weights'].get(arm),
                         ledger=lpp_ledger(view, bd, store, arm, refs=arm_refs),
                         reference_bases=None if arm_refs is None else arm_refs.basis_info, **extra)
        if arm in predictions:
            add_scores(view, bd, cell, predictions[arm], residual)
            out[arm] = predictions[arm][0]
        if arm in controls:
            cell['controls'] = controls[arm]
        finish(cell, log, c.folder, c.name, arm, out.get(arm))
    return out


def rotated_budget(bd):
    """(copy of ``bd`` whose training targets are rotated within every budget section by half its rows, {section:
    rows rotated}); the design rows are shared, not copied (run_v3 stage_controls' disrupted-training rule)."""
    rot, y, shifts = copy.copy(bd), np.array(bd.y, dtype=np.float64, copy=True), {}
    for st in bd.stories:
        idx = np.flatnonzero(bd.story == st)
        shifts[st] = int(len(idx) // 2)
        y[idx] = np.roll(bd.y[idx], shifts[st], axis=0)
    rot.y = y
    return rot, shifts


def fit_controls(c, bd, arms, predictions):
    """§9 brain-input controls of the CONTROL_ARMS ``arms`` at the primary cell: {arm: controls block}.

    ``rotated_training``: the arms' full pipeline (run_v4.run_cv and ``final_fit`` with fresh priors: own encoding,
    EB null sigma^2, calibration and shrinkage, weights, inner CV, final fit; the label-free ISP map is unchanged)
    on ``rotated_budget(bd)``, scored against the true test targets, with its own log (R1-R7 and the registered
    scope must pass for a scored control; its log is embedded so that every re-audit of the cell checks it; a
    control that is not estimable records its scope errors and log digest instead). ``prediction``: the
    arm's real test prediction in float64 under controls/predictions/ with its digest, read by
    ``subject_controls`` (shift null and wrong-section input over the 9 outer sections).
    """
    view, store, ctx = c.view, c.store, c.ctx
    rot, shifts = rotated_budget(bd)
    prov = c.prov()
    refs = LppReferences(view.cfg, view, prov, ctx.device, store) if set(arms) & set(REFERENCE_ARMS) else None
    pri = LppPriors(view, refs, prov, ctx.device, store)
    log_json(test_section=c.t, budget=c.name, controls=list(arms), rotated_rows=shifts)
    selections, curves, _ = v4.run_cv(view, rot, c.name, arms, pri, prov, None, c.folds, None, c.log_fold)
    preds, _ = final_fit(view, rot, c.name, arms, selections, pri, prov)
    out = {}
    for arm in arms:
        log = prov.merged(arm=arm)
        log.check()
        errors = audit_v4.scope_errors(dict(subject=c.subject, arm=arm, stage=c.stage, budget=c.name, regime='S'),
                                       log)
        if errors and arm in preds:
            raise SystemExit(f'Registered-scope audit failed for the rotated-target control t{c.t} {c.name} {arm}: '
                             f'{errors}')
        rotated = dict(rule=ROTATION_RULE, shift_rows=shifts, estimable=arm in preds, selection=selections.get(arm),
                       cv_curves=curves.get(arm), provenance=None if errors else log.to_dict())
        if errors:  # Not estimable: the log is not embedded, so the control can never fail the cell's re-audit.
            rotated.update(scope_errors=errors, provenance_sha256=v4.sha(json.dumps(jsonable(log.to_dict()),
                                                                                    sort_keys=True)))
        if arm in preds:
            rotated['test'] = v4.report(view, rot, preds[arm][0]['test'], 'test')
        block = dict(rotated_training=rotated, prediction=None)
        if arm in predictions:
            pred = np.asarray(predictions[arm][0]['test'], dtype=np.float64)
            rel = Path('controls') / 'predictions' / f't{c.t}_{c.name}_{arm}.npy'
            atomic_save(c.folder / rel, pred)
            block['prediction'] = dict(path=rel.as_posix(), array_digest=lpp_prepare.array_digest(pred))
        out[arm] = block
    del rot
    return out


def fit_ladder(c, bd, arm, prov, refs, residual):
    """PRIOR[f] through the engine's PRIOR slot (``LadderPriors``), on the shared log."""
    view, store, ctx = c.view, c.store, c.ctx
    pri = LadderPriors(view, refs, prov, ctx.device, store, float(arm[len('PRIOR_f'):]))
    selections, curves, logs = v4.run_cv(view, bd, c.name, [ENGINE_ARM], pri, prov, None, c.folds, None, c.log_fold)
    predictions, final = final_fit(view, bd, c.name, [ENGINE_ARM], selections, pri, prov, None,
                                   None if residual is None else residual['train'])
    cell = base_cell(c, arm, engine_arm=ENGINE_ARM, selection=selections.get(ENGINE_ARM),
                     cv_curves=curves.get(ENGINE_ARM), folds=logs,
                     final_priors={k: v for k, v in final.items() if k in FINAL_PRIORS},
                     final_weights=final['weights'].get(ENGINE_ARM),
                     ledger=lpp_ledger(view, bd, store, arm, refs=refs), reference_bases=refs.basis_info)
    out = {}
    if ENGINE_ARM in predictions:
        add_scores(view, bd, cell, predictions[ENGINE_ARM], residual)
        out[arm] = predictions[ENGINE_ARM][0]
    finish(cell, prov.merged(arm=arm), c.folder, c.name, arm, out.get(arm))
    return out


def pair_fields(c):
    """(references, cell folder, extra cell fields) of TH_SEL / XDEC with the target's registered two-reference
    sensitivity pair ((None, None, {}) with fewer than two references)."""
    pair = c.store.pair
    if pair is None:
        return None, None, {}
    return pair, pair_folder(c.folder, pair), dict(sensitivity=dict(rule=PAIR_RULE, pair=pair,
                                                                    pool=list(c.store.references)))


def fit_selection(c, bd, arm, residual, decoder=False, pair=False):
    """TH_SEL (rule SS) or TH_SEL_rand through the engine's PRIOR slot (``tanghuth.selection_priors``), own log.

    TH_SEL needs out-of-budget non-test material for its selection (``tanghuth.lpp_unheard``); at L8 there is none:
    not estimable, logged, no cell. ``decoder``: fit the reference models with their W1 decoders (XDEC will need
    them; the selected voxels are the same either way). ``pair``: TH_SEL with the target's sensitivity pair only
    (``LppStore.pair``; cell in the pair's folder, returned under 'TH_SEL__pair'); its selectors share the cell's
    converter-alpha memo with TH_SEL's.
    """
    view, store, ctx, th = c.view, c.store, c.ctx, c.ctx.th
    n_voxels = view.index['train'].n_features
    k = th.goal_size(n_voxels, 'lpp')
    models, model_info, unheard = {}, {}, []
    references, folder, extra = (store.references, c.folder, {}) if not pair else pair_fields(c)
    key = arm + (PAIR_SUFFIX if pair else '')
    if arm == 'TH_SEL':
        unheard = th.lpp_unheard(SECTIONS, c.budget, c.t)
        if not unheard or len(references or ()) < 2:
            log_json(test_section=c.t, budget=c.name, arm=key, not_estimable=(
                'no out-of-budget non-test section (selection material)' if not unheard else 'fewer than 2 references'))
            return {}
        prov = c.prov(arm=arm, reference_rule='SS')
        for ref in references:
            models[ref], model_info[ref] = store.reference_model(ref, c.t, view.clock, decoder=decoder)
            th.log_reads(prov, th.model_reads(models[ref]))
        selector = th.ThSelector(c.subject, store.target_bold, list(models), lambda ref, st: store.reference_runs(
            'z', ref, st, models[ref].voxels, ('model', c.t, view.clock)), unheard, k, prov, device=ctx.device,
            alpha_memo=c.alpha_memo)
    elif pair:
        raise ValueError('The two-reference sensitivity runs TH_SEL and XDEC only')
    else:
        prov = c.prov(arm=arm)
        selector = th.RandomSelector(n_voxels, k, c.subject)
    pri = th.selection_priors(view, prov, ctx.device, selector)
    if residual is not None:
        log_nuisance(prov, view, bd, c.folds)
    selections, curves, logs = v4.run_cv(view, bd, c.name, [ENGINE_ARM], pri, prov, None, c.folds, None, c.log_fold)
    predictions, final = final_fit(view, bd, c.name, [ENGINE_ARM], selections, pri, prov, None,
                                   None if residual is None else residual['train'])
    exposure = comparator_exposure(view, models, list(c.budget) + unheard) if models else None
    cell = base_cell(c, arm, engine_arm=ENGINE_ARM, references=sorted(models),
                     reference_rule=prov.reference_rule, selection=selections.get(ENGINE_ARM),
                     selection_note='binary mask: the engine gamma rules tie (weights equal at every gamma)',
                     cv_curves=curves.get(ENGINE_ARM), folds=logs, final_selection=final.get('isp'),
                     final_weights=final['weights'].get(ENGINE_ARM), unheard=unheard, goal_size=k,
                     ledger=lpp_ledger(view, bd, store, arm, comparator=exposure),
                     reference_models={r: dict(models[r].summary(), **model_info[r]) for r in models} or None, **extra)
    out = {}
    if ENGINE_ARM in predictions:
        add_scores(view, bd, cell, predictions[ENGINE_ARM], residual)
        out[key] = predictions[ENGINE_ARM][0]
    finish(cell, prov, folder, c.name, arm, out.get(key), auditor=th.audit_errors)
    return out


def fit_xdec(c, bd, pair=False):
    """XDEC (rule SS, own log without inner folds, zero target labels): ``tanghuth.xdec_predict`` for the test bins
    from the references' W1 decoders and converters fit on B; scored with the references' label statistics.
    ``pair``: with the target's sensitivity pair only (cell in the pair's folder, returned under 'XDEC__pair')."""
    view, store, ctx, th = c.view, c.store, c.ctx, c.ctx.th
    references, folder, extra = (store.references, c.folder, {}) if not pair else pair_fields(c)
    key = 'XDEC' + (PAIR_SUFFIX if pair else '')
    if not references:
        log_json(test_section=c.t, budget=c.name, arm=key, not_estimable='no reference pair' if pair else
                 'no references')
        return {}
    prov = c.prov(arm='XDEC', reference_rule='SS', folds=False)
    models, model_info = {}, {}
    for ref in references:
        models[ref], model_info[ref] = store.reference_model(ref, c.t, view.clock, decoder=True)
        th.log_reads(prov, th.model_reads(models[ref]))
    budget, sp = list(c.budget), view.split['test']
    decoders = {ref: m.decoder for ref, m in models.items()}
    target_runs = {s: store.target_bold(s) for s in budget}
    reference_runs = {ref: {s: store.reference_runs('raw', ref, s, decoders[ref].voxels, ('decoder', c.t, view.clock))
                            for s in budget} for ref in models}
    th.log_reads(prov, th.xdec_reads(c.subject, list(models), budget, [c.t]))
    pred, info = th.xdec_predict(target_runs, reference_runs, decoders, budget, {c.t: store.target_eval_bold(c.t)},
                                 np.full(len(sp.target_trs), c.t), sp.target_trs, device=ctx.device)
    mean, std = th.scoring_stats(decoders)
    cell = base_cell(c, 'XDEC', references=sorted(models), reference_rule='SS',
                     estimable=True, converters=info, target_labels=0, selection=None,
                     score_normalization='reference label mean/SD averaged over references (no target label)',
                     test=score(pred, sp.targets, sp.run_ids, sp.target_trs, sp.sessions, mean, std,
                                view.cfg['evaluation']['exclusions']),
                     test_shift_null=section_shift_null(view, pred),
                     ledger=lpp_ledger(view, bd, store, 'XDEC', comparator=comparator_exposure(view, models, budget)),
                     reference_models={r: dict(models[r].summary(), **model_info[r]) for r in models}, **extra)
    finish(cell, prov, folder, c.name, 'XDEC', dict(test=pred), auditor=th.audit_errors)
    return {key: dict(test=pred)}


# --------------------------------------------------------------------------------------------- stages
def read_partition(ctx):
    """(DEV, CONFIRM) listener ids: the lock's lpp_partition, else data/lpp/prepare_provenance.json, checked against
    the registered hash order (budgets.lpp_partition)."""
    part = None
    if ctx.lock_path.is_file():
        part = json.loads(ctx.lock_path.read_text()).get('lpp_partition')
    if part is None:
        part = json.loads((ctx.lpp_root / 'prepare_provenance.json').read_text())['partition']
    dev, confirm = [str(s) for s in part['dev']], [str(s) for s in part['confirm']]
    want_dev, want_confirm = bud.lpp_partition(dev + confirm)
    if list(want_dev) != dev or set(want_confirm) != set(confirm):
        raise SystemExit('The LPP partition differs from the registered hash order')
    return dev, confirm


def references_of(subject, dev, confirm):
    """The partition's other listeners: DEV targets the other 8 DEV listeners; CONFIRM targets all 48 other EN
    listeners (before the preparation filter of ``reference_pool``)."""
    if subject in dev:
        return [s for s in dev if s != subject]
    if subject in confirm:
        return [s for s in dev + confirm if s != subject]
    raise SystemExit(f'{subject} is not an EN listener of the partition')


def reference_pool(ctx, subject, dev, confirm):
    """(references, record) of a target: ``references_of`` restricted to the listeners that pass the primary
    analysis's subject-level QC (POOL_RULE; ``summarize_v4.prepared_trs`` and ``registered_pool`` themselves, so
    the runner and the QC agree on the pool). Refused when the target itself does not pass. ``record`` goes into
    every cell (``reference_pool``)."""
    import summarize_v4 as sv
    partition = references_of(subject, dev, confirm)
    try:
        prepared = sv.prepared_trs(ctx.lpp_root)
    except sv.Refused as error:
        raise SystemExit(f'reference pool: {error}') from error
    if not sv.prepared_ok(prepared, subject):
        raise SystemExit(f'{subject} has no preparation record with the registered section TR counts under '
                         f'{ctx.lpp_root} ({sorted(prepared.get(subject, ()))}): it has no prepared data')
    references = sorted(str(s) for s in sv.registered_pool(partition, prepared))
    dropped = [str(s) for s in partition if str(s) not in references]
    record = dict(rule=POOL_RULE, partition=sorted({str(s) for s in [subject] + list(partition)}),
                  references=references, dropped={s: [list(x) for x in sorted(prepared.get(s, ()))] or
                                                  'no preparation record' for s in dropped})
    return references, record


def started_path(ctx):
    """results/v4/primary_started.json (summarize_v4.STARTED) next to this runner's results root."""
    return ctx.out_root.parent / 'v4' / 'primary_started.json'


def g4_passed(ctx):
    """G4 passed under this code and on the data its records were computed on."""
    path = ctx.out_root / 'g4.json'
    if not path.is_file():
        return False
    g4 = json.loads(path.read_text())
    if not (g4.get('passed') is True and g4.get('code_sha256') == ctx.code and g4.get('dev')):
        return False
    return g4.get('data_sha256') == ctx.data_sha256(g4['dev'])


def check_develop(ctx, subject, arms, dev, g0=None):
    """Registered order: G0 under the run_v4 core; W0 before G4; every other arm only after G4 passed here; no
    develop run once the lock exists (post-lock DEV reruns may not change recorded outputs)."""
    if ctx.lock_path.is_file():
        raise SystemExit(f'develop refused: {ctx.lock_path} exists; the locked DEV cells are never rewritten')
    if subject not in dev:
        raise SystemExit(f'develop runs LPP DEV listeners only; {subject} is not DEV')
    if not (v4.g0_passed() if g0 is None else g0):
        raise SystemExit('G0 has not passed under the run_v4 core (results/v4/g0.json)')
    priors = sorted(set(arms) - {G4_ARM})
    if priors and not g4_passed(ctx):
        raise SystemExit(f'G4 has not passed under this code and data ({ctx.out_root / "g4.json"}); only W0 may touch '
                         f'LPP before it, not {priors}')


def check_lock(ctx, subject, git_state='auto', no_git_check=False, lock_sha256=None):
    """CONFIRM authorization: the lock exists (with ``lock_sha256``: its sha256 is that value), its fingerprints
    equal this code, it carries a confirmatory plan and lists the listener as CONFIRM, and it is committed,
    unmodified and pushed to origin/main. ``git_state`` 'auto' asks git; where git is unavailable the lock is
    refused unless ``no_git_check``. A caller passing ``git_state=None`` has checked git itself (scripts/v4_lock.py).
    Sets ``ctx.lock_record`` (recorded in every evaluate cell) and returns the lock."""
    if not ctx.lock_path.is_file():
        raise SystemExit(f'confirm refused: {ctx.lock_path} does not exist')
    raw = ctx.lock_path.read_bytes()
    digest = sha256_bytes(raw)
    if lock_sha256 is not None and lock_sha256 != digest:
        raise SystemExit(f'confirm refused: --lock-sha256 {lock_sha256} is not the sha256 of {ctx.lock_path} '
                         f'({digest})')
    lock = json.loads(raw)
    want = fingerprints(ctx)
    bad = sorted(k for k, v in want.items() if lock.get(k) != v)
    if bad:
        raise SystemExit(f'confirm refused: lock fingerprints {bad} differ from this code')
    if subject not in (lock.get('lpp_partition') or {}).get('confirm', ()):
        raise SystemExit(f'confirm refused: {subject} is not a CONFIRM listener of the lock')
    if not isinstance(lock.get('confirmatory_plan'), dict) or not lock['confirmatory_plan'].get('hypotheses'):
        raise SystemExit('confirm refused: the lock carries no confirmatory plan')
    check = 'not checked here (the caller verified git)'
    if git_state == 'auto':
        git_state = lpp_prepare.lock_git_state(ctx.lock_path)
        if git_state is None and not no_git_check:
            raise SystemExit(f'confirm refused: git is unavailable for {ctx.lock_path}, so it cannot be verified as '
                             'committed and pushed; run where git is available, or pass --no-git-check (recorded '
                             'in every evaluate cell)')
        check = 'verified' if git_state is not None else 'skipped (--no-git-check)'
    if git_state is not None and not (git_state['tracked'] and git_state['clean'] and git_state['pushed']):
        raise SystemExit(f'confirm refused: the lock must be committed, unmodified and pushed: {git_state}')
    ctx.lock_record = dict(path=str(ctx.lock_path), sha256=digest, git=git_state, git_check=check)
    return lock


def check_confirm(ctx, lock, subject, references):
    """Before any CONFIRM fit: the primary analysis has not started, the manifest of the target and of every
    listener of its reference pool (``reference_pool``) exists and carries the locked ``lpp_mask_sha256``, the DEV
    preparation record is the one the lock certified and their recorded content is available
    (``Context.content``)."""
    if started_path(ctx).exists():
        raise SystemExit(f'confirm refused: {started_path(ctx)} exists; the primary analysis has started and its QC '
                         'hashed the CONFIRM cells')
    mask = lock.get('lpp_mask_sha256')
    absent = [s for s in [subject] + list(references) if not ctx.manifest_path(s).is_file()]
    if absent:
        raise SystemExit(f'confirm refused: {absent} pass the preparation QC but have no manifest under '
                         f'{ctx.lpp_root}/semantic: the preparation is incomplete')
    wrong = [s for s in [subject] + list(references)
             if json.loads(ctx.manifest_path(s).read_text()).get('mask_content_sha256') != mask]
    if not mask or wrong:
        raise SystemExit(f'confirm refused: the manifests of {wrong} do not carry the locked mask digest {mask}')
    recorded = (lock.get('lpp_preparation') or {}).get('sha256')
    if recorded and ctx.data.records.get('prepare_provenance.json') != recorded:
        raise SystemExit('confirm refused: data/lpp/prepare_provenance.json is not the DEV record the lock certified')
    return ctx.data_sha256([subject] + list(references))


def plan_cells(lock):
    """{(clock, budget): set of arms}: every cell the locked confirmatory plan reads (summarize_v4.required_cells)
    plus the ORACLE and W0 cells of its oracle fractions (summarize_v4.oracle_budgets)."""
    import summarize_v4 as sv
    pl = lock['confirmatory_plan']
    need = {}
    for budget, arm in sv.required_cells(pl):
        need.setdefault((pl['clock'], budget), set()).add(arm)
    for budget in sv.oracle_budgets(pl):
        need.setdefault((pl['clock'], budget), set()).update(('ORACLE', 'W0'))
    unknown = sorted({a for arms in need.values() for a in arms} - set(ARMS))
    if unknown:
        raise SystemExit(f'confirm refused: the locked plan reads arms this runner cannot write: {unknown}')
    return need


def arms_at(arms, clock, budget, primary=None):
    """``arms`` to run at (clock, budget): the SCOPED arms only at ``primary`` (when given)."""
    return [a for a in arms if a not in SCOPED or primary is None or (clock, budget) == tuple(primary)]


def fit_plan(clocks, budgets, arms, primary=None, extra=None):
    """{(clock, budget): arms} for ``stage_fit``: ``arms`` at every requested clock and budget (``arms_at``), plus
    the ``extra`` {(clock, budget): arms} that lie among the requested clocks and budgets."""
    out = {}
    for clock in clocks:
        for budget in budgets:
            here = arms_at(arms, clock, budget, primary) + sorted((extra or {}).get((clock, budget), ()))
            here = list(dict.fromkeys(here))
            if here:
                out[(clock, budget)] = here
    return out


def uncovered(ctx, subject, data, need, plan, sections):
    """Plan cells (``need``) of the requested outer ``sections`` that this request (``plan``) does not write and that
    do not exist as current cells (sections may be run in separate calls; arms, budgets and clocks may not be
    left out of a section)."""
    missing = []
    for (clock, budget), arms in sorted(need.items()):
        for arm in sorted(arms):
            for t in sections:
                if arm in plan.get((clock, budget), ()):
                    continue
                path = cell_path(ctx.out_root / subject / clock / 'evaluate', t, budget, arm)
                if cell_state(ctx, path, data, 'evaluate') != 'current':
                    missing.append(f'{clock} t{t} {budget} {arm}')
    return missing


def stage_prepare(ctx, subjects, clocks, confirm=()):
    for subject in subjects:
        if subject in confirm and not ctx.lock_path.is_file():
            raise SystemExit(f'{subject} is a CONFIRM listener: caches only after {ctx.lock_path} exists')
        for clock in clocks:
            tgt = LppTarget(ctx, subject, clock, build_cache=True)
            index = tgt.index['train']
            log_json(subject=subject, clock=clock, rows=len(index), voxels=index.n_features,
                     picture_excluded=index.picture_excluded, cache=str(tgt.train_dir))


def stage_fit(ctx, subject, references, plan, sections, stage, pool=None, primary=None):
    """develop / evaluate cells of one target listener: ``plan`` {(clock, budget): arms} (``fit_plan``) x outer
    ``sections``. ``pool``: the reference-pool record of every cell; ``primary`` (clock, budget): the primary cell,
    where the CONTROL_ARMS carry the §9 controls and ``subject_controls`` runs once all 9 sections are in."""
    arms = {a for v in plan.values() for a in v}
    if arms & set(COMPARATORS + PAIR_ARMS) and ctx.th is None:  # Imports Module H before any compute.
        raise SystemExit('The comparators need src/priors/tanghuth.py')
    primary = None if primary is None else tuple(primary)
    for clock in CLOCKS:
        names = [b for b in BUDGETS if (clock, b) in plan]
        if not names:
            continue
        tgt = LppTarget(ctx, subject, clock)
        store = LppStore(ctx, tgt, references, pool)
        folder = ctx.out_root / subject / clock / stage
        for t in sections:
            view = tgt.outer(t)
            for name in names:
                run_budget(ctx, view, store, name, plan[(clock, name)], stage, folder,
                           controls=(clock, name) == primary)
            store.release()
        for name in names:
            if (clock, name) == primary:
                subject_controls(ctx, subject, clock, stage, name, ctx.data_sha256([subject] + list(references)))


def shift_record(view_of, predictions, min_tr=G4_MIN_SHIFT_TR, draws=G4_DRAWS, seed=G4_SEED):
    """G4(b) record of one listener: the section r's and the within-section circular-shift null of their mean
    (``endpoints.shift_null`` over the 9 outer sections, each its own session)."""
    pred, target, run_ids = [], [], []
    for t in SECTIONS:
        sp = view_of(t).split['test']
        pred.append(predictions[t])
        target.append(sp.targets)
        run_ids.append(sp.run_ids)
    first = view_of(SECTIONS[0])
    null = shift_null(np.concatenate(pred), np.concatenate(target), np.concatenate(run_ids),
                      first.split['test'].sessions, first.stride, min_tr, draws, seed)
    observed, q95 = null.get('observed_session_macro'), null.get('null_q95')
    passed = observed is not None and q95 is not None and observed > q95
    return dict(observed=observed, null_q95=q95, passed=bool(passed), shift_null=null)


def scoring_views(ctx, subject):
    """{t: view with ``split['test']`` and ``stride``}: the stride-4 scoring bins of every outer section (as
    ``LppTarget.outer``), from the manifest alone (no cache)."""
    data = ctx.cfg['data']
    stride = int(data['stride'])
    index = LppIndex(ctx.manifest_path(subject), stride, window=data['window'], phase=int(data['target_phase']),
                     cutoffs=data['cutoffs'])
    story = np.array([index.runs[ri]['story'] for ri, _ in index.items])
    return {t: SimpleNamespace(split={'test': SectionSplit(index, np.flatnonzero(story == t))}, stride=stride)
            for t in SECTIONS}


def wrong_section(views, predictions):
    """§9 wrong-section input (WRONG_SECTION_RULE): {by_section: r, mean} of each section's targets against the
    next section's prediction (section order, cyclic), cycled to the section's length."""
    by_section = {}
    for i, t in enumerate(SECTIONS):
        other = predictions[SECTIONS[(i + 1) % len(SECTIONS)]]
        targets = np.asarray(views[t].split['test'].targets, dtype=np.float64)
        wrong = np.asarray(other, dtype=np.float64)[np.arange(len(targets)) % len(other)]
        by_section[t] = float(_feature_correlation(wrong, targets)[0])
    return dict(rule=WRONG_SECTION_RULE, by_section=by_section, mean=float(np.mean(list(by_section.values()))))


def subject_controls(ctx, subject, clock, stage, budget, data, arms=CONTROL_ARMS):
    """§9 controls of one listener at the primary cell, per arm whose 9 outer-section cells are current and carry
    their controls: <stage folder>/controls/<budget>_<arm>.json with R_s, the subject-level within-section
    circular-shift null (``shift_record`` at the registered evaluation.shift_* settings, on the float64 predictions
    verified against the cells' digests), the wrong-section input control (``wrong_section``), the rotated-target
    refit (per section and R_s) and the nuisance-only floors. Deterministic from the cells: a record with the same
    inputs is kept; at stage evaluate a record from other inputs is refused (develop rewrites it). Nothing about
    the values is printed at stage evaluate. Returns {arm: record} of the arms written or kept."""
    folder = ctx.out_root / subject / clock / stage
    ev = ctx.cfg['evaluation']
    views, out = None, {}
    for arm in arms:
        paths = {t: cell_path(folder, t, budget, arm) for t in SECTIONS}
        if any(cell_state(ctx, q, data, stage, ('controls',)) != 'current' for q in paths.values()):
            continue
        cells = {t: json.loads(q.read_text()) for t, q in paths.items()}
        if not all(c_.get('estimable') and (c_['controls'] or {}).get('prediction') for c_ in cells.values()):
            log_json(subject=subject, clock=clock, budget=budget, arm=arm, controls='not estimable in every section')
            continue
        inputs = {q.relative_to(ctx.out_root).as_posix(): sha256_bytes(q.read_bytes()) for q in paths.values()}
        target = folder / 'controls' / f'{budget}_{arm}.json'
        if target.is_file():
            old = json.loads(target.read_text())
            if old.get('inputs') == inputs and old.get('code_sha256') == ctx.code:
                out[arm] = old
                continue
            if stage == 'evaluate':
                raise SystemExit(f'{target} exists from other cells or code; evaluate control records are written once')
        predictions = {}
        for t, cell in cells.items():
            meta = cell['controls']['prediction']
            pred = np.load(folder / meta['path'])
            if lpp_prepare.array_digest(pred) != meta['array_digest']:
                raise SystemExit(f"{folder / meta['path']}: not the prediction cell t{t} {budget} {arm} recorded")
            predictions[t] = pred
        views = views or scoring_views(ctx, subject)
        rotated = {t: cell['controls']['rotated_training'] for t, cell in cells.items()}
        rotated_r = {t: (r.get('test') or {}).get(VAL) for t, r in rotated.items()}
        floors = {}
        for t, cell in cells.items():
            for name, model in ((cell.get('nuisance_floors') or {}).get('models') or {}).items():
                floors.setdefault(name, {})[t] = (model.get('test') or {}).get(VAL)
        record = dict(protocol='ssltang-v4', dataset='lpp', stage=stage, kind='controls', subject=subject, clock=clock,
                      budget=budget, arm=arm, code_sha256=ctx.code, config_sha256=ctx.config, data_sha256=data,
                      created_utc=now_utc(), inputs=inputs,
                      observed=float(np.mean([cells[t]['test'][VAL] for t in SECTIONS])),
                      shift_null=dict(shift_record(views.get, predictions, ev['shift_min_tr'], ev['shift_draws'],
                                                   ev['shift_seed']),
                                      rule='FUTURE_WORK_V4 §9 within-section circular shift of R_s (>= '
                                           f"{ev['shift_min_tr']} TR, {ev['shift_draws']} draws, seed "
                                           f"{ev['shift_seed']}; endpoints.shift_null over the 9 sections)"),
                      wrong_section=wrong_section(views, predictions),
                      rotated_training=dict(rule=ROTATION_RULE, by_section=rotated_r,
                                            estimable=all(r.get('estimable') for r in rotated.values()),
                                            mean=None if None in rotated_r.values() else
                                            float(np.mean(list(rotated_r.values())))),
                      nuisance_floors={name: dict(by_section=v, mean=None if len(v) < len(SECTIONS) or None in
                                                  v.values() else float(np.mean(list(v.values()))))
                                       for name, v in floors.items()} or None)
        atomic_json(target, record)
        out[arm] = record
        shown = {} if stage == 'evaluate' else dict(observed=record['observed'],
                                                    null_q95=record['shift_null']['null_q95'],
                                                    wrong_section=record['wrong_section']['mean'],
                                                    rotated=record['rotated_training']['mean'])
        log_json(subject=subject, clock=clock, budget=budget, arm=arm, controls=str(target), **shown)
    return out


def g4_subject_b(ctx, subject):
    """G4(b) for one DEV listener (resumable under the same code and data): W0 dense L8 cells for the 9 outer
    sections and the shift null."""
    folder = ctx.out_root / subject / G4_CLOCK / 'g4'
    path = folder / 'g4b.json'
    data = ctx.data_sha256([subject])
    if path.is_file():
        record = json.loads(path.read_text())
        if record.get('code_sha256') == ctx.code and record.get('data_sha256') == data:
            return record
    tgt = LppTarget(ctx, subject, G4_CLOCK)
    store = LppStore(ctx, tgt, ())
    predictions = {}
    for t in SECTIONS:
        out = run_budget(ctx, tgt.outer(t), store, G4_BUDGET, [G4_ARM], 'develop', folder, residualize=False,
                         overwrite=True)
        if G4_ARM not in out:
            raise SystemExit(f'{subject} t{t}: W0 not estimable at {G4_BUDGET}')
        predictions[t] = out[G4_ARM]['test']
    record = dict(protocol='ssltang-v4', stage='g4b', subject=subject, clock=G4_CLOCK, budget=G4_BUDGET, arm=G4_ARM,
                  code_sha256=ctx.code, data_sha256=data, created_utc=now_utc(),
                  rule=f'R_s(W0) > q95 of the within-section circular-shift null (>= {G4_MIN_SHIFT_TR} TR, '
                       f'{G4_DRAWS} draws, seed {G4_SEED})',
                  **shift_record(tgt.outer, predictions))
    save_json(jsonable(record), path)
    log_json(subject=subject, g4b=dict(observed=record['observed'], null_q95=record['null_q95'],
                                       passed=record['passed']))
    return record


def g4_subject_c(ctx, subject, references, sections=SECTIONS):
    """G4(c) for one DEV listener (resumable under the same code and data): the alpha of every ISP fit develop will
    use (no decoder, no labels) over the outer ``sections`` (the gate needs all 9). The maps are cached, so develop
    reuses them."""
    path = ctx.out_root / subject / 'g4c.json'
    sections = [str(s) for s in sections]
    data = ctx.data_sha256([subject] + list(references))
    if path.is_file():
        record = json.loads(path.read_text())
        if record.get('code_sha256') == ctx.code and record.get('data_sha256') == data and \
                record.get('sections') == sections:
            return record
    tgt = LppTarget(ctx, subject, G4_CLOCK)
    store = LppStore(ctx, tgt, references)
    fits = []
    for t in sections:
        view = tgt.outer(t)
        for name, budget in bud.lpp_section_budgets(t, SECTIONS).items():
            folds = [tuple(f) for f in bud.inner_folds(name, budget)]
            prov = Provenance(subject, 'S', budget, [], [t], folds=folds, budget_name=name, stage='develop')
            pri = LppPriors(view, LppReferences(view.cfg, view, prov, ctx.device, store), prov, ctx.device, store)
            for i, held in enumerate(folds):
                with prov.inner_fold(i):
                    _, info = pri.isp([s for s in budget if s not in held])
                fits.append(dict(test_section=t, budget=name, fold=i, alpha=info['alpha'],
                                 at_edge=info['alpha_at_edge']))
            _, info = pri.isp(budget)
            fits.append(dict(test_section=t, budget=name, fold=None, alpha=info['alpha'],
                             at_edge=info['alpha_at_edge']))
            prov.check()
    edges = [f for f in fits if f['at_edge']]
    record = dict(protocol='ssltang-v4', stage='g4c', subject=subject, code_sha256=ctx.code, data_sha256=data,
                  created_utc=now_utc(), references=list(store.references), sections=sections, n_fits=len(fits),
                  n_at_edge=len(edges), passed=not edges, fits=fits)
    save_json(jsonable(record), path)
    log_json(subject=subject, g4c=dict(n_fits=len(fits), n_at_edge=len(edges)))
    return record


def g4_verdict(b_records, c_records=None):
    """G4: (b) passes in >= 6 of the DEV listeners; (c) every recorded ISP alpha interior (None: not run)."""
    n = sum(bool(r['passed']) for r in b_records.values())
    b = dict(subjects={s: dict(observed=r['observed'], null_q95=r['null_q95'], passed=r['passed'])
                       for s, r in b_records.items()}, n_passed=n, required=G4_MIN_PASS, passed=n >= G4_MIN_PASS)
    c = None
    if c_records is not None:
        c = dict(subjects={s: dict(n_fits=r['n_fits'], n_at_edge=r['n_at_edge'], passed=r['passed'])
                           for s, r in c_records.items()}, passed=all(r['passed'] for r in c_records.values()))
    return dict(b=b, c=c, passed=bool(b['passed'] and c is not None and c['passed']))


def current_records(ctx, dev, relative, data_of):
    """{subject: record} of every DEV listener whose record exists under this code and its current data
    (``data_of(subject)``) over all 9 outer sections, or None if one is missing."""
    out = {}
    for s in dev:
        path = ctx.out_root / s / relative
        record = json.loads(path.read_text()) if path.is_file() else None
        if record is None or record.get('code_sha256') != ctx.code or record.get('data_sha256') != data_of(s) or \
                record.get('sections', list(SECTIONS)) != list(SECTIONS):
            log_json(g4='pending', waiting_for=str(path))
            return None
        out[s] = record
    return out


def stage_g4(ctx, subjects, dev, confirm):
    """G4(b) for ``subjects``; once every DEV listener has (b): the verdict, then (c) only if (b) passed."""
    if ctx.lock_path.is_file():
        raise SystemExit(f'G4 refused: {ctx.lock_path} exists; the locked G4 records and cells are never rewritten')
    unknown = sorted(set(subjects) - set(dev))
    if unknown:
        raise SystemExit(f'G4 runs on LPP DEV listeners only: {unknown}')
    for s in subjects:
        g4_subject_b(ctx, s)
    b = current_records(ctx, dev, Path(G4_CLOCK) / 'g4' / 'g4b.json', lambda s: ctx.data_sha256([s]))
    if b is None:
        return None
    head = dict(protocol='ssltang-v4', stage='g4', code_sha256=ctx.code, data_sha256=ctx.data_sha256(dev),
                dev=list(dev))
    if not g4_verdict(b)['b']['passed']:
        out = dict(head, created_utc=now_utc(), **g4_verdict(b),
                   note='G4(b) failed: at most two revisions of the LPP adaptation (mask resolution 2 vs 3 mm, '
                        'smoothing 0 or 4 mm), chosen by W0 CV score only, before any prior arm touches LPP')
    else:
        pools = {}

        def refs(s):  # The DEV target's reference pool (``reference_pool``), as develop uses it.
            if s not in pools:
                pools[s] = reference_pool(ctx, s, dev, confirm)[0]
            return pools[s]
        for s in subjects:
            g4_subject_c(ctx, s, refs(s))
        c = current_records(ctx, dev, Path('g4c.json'), lambda s: ctx.data_sha256([s] + refs(s)))
        if c is None:
            return None
        out = dict(head, created_utc=now_utc(), **g4_verdict(b, c))
    save_json(jsonable(out), ctx.out_root / 'g4.json')
    log_json(g4=dict(passed=out['passed'], b=out['b']['n_passed'], c=None if out['c'] is None else out['c']['passed']))
    return out


def disk_check(paths, min_free_gb):
    """Refuse to start with less than ``min_free_gb`` GiB free on the file system of any of ``paths`` (or their
    nearest existing parent): a cache or cell write that fails part-way stops a long run."""
    for path in paths:
        here = Path(path).resolve()
        while not here.exists() and here != here.parent:
            here = here.parent
        free = shutil.disk_usage(here).free / 2 ** 30
        if free < float(min_free_gb):
            raise SystemExit(f'{free:.0f} GiB free under {here}, less than --min-free-gb {min_free_gb} (the CONFIRM '
                             'caches and cells need about 0.45 TB in all; see the module docstring)')


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--config', default='configs/v3/structured_ridge.yaml')
    common.add_argument('--device', default='cuda')
    common.add_argument('--min-free-gb', type=float, default=50.0,
                        help='refuse to start with less free disk space under the cache and results roots')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='stage', required=True)
    prep = sub.add_parser('prepare-caches', parents=[common])
    prep.add_argument('--subjects', nargs='+', required=True)
    prep.add_argument('--clocks', nargs='+', default=list(CLOCKS), choices=CLOCKS)
    g4 = sub.add_parser('g4', parents=[common])
    g4.add_argument('--subjects', nargs='+', help='DEV listeners (default: all 9)')
    choices = list(ARMS) + list(SUBSET_ALIASES)
    for name in ('develop', 'confirm'):
        q = sub.add_parser(name, parents=[common])
        q.add_argument('--subject', required=True)
        q.add_argument('--clocks', nargs='+', default=list(CLOCKS), choices=CLOCKS)
        q.add_argument('--budgets', nargs='+', default=list(BUDGETS), choices=BUDGETS)
        q.add_argument('--arms', nargs='+', default=None, choices=choices,
                       help=f"default: {' '.join(DEVELOP_DEFAULT if name == 'develop' else CONFIRM_DEFAULT)}"
                            + ('' if name == 'develop' else ' and every arm the locked plan reads'))
        q.add_argument('--sections', nargs='+', default=list(SECTIONS), choices=SECTIONS)
        q.add_argument('--scope', default='primary', choices=('primary', 'all'),
                       help=f"where the scoped arms ({', '.join(SUBSET_ALIASES)}, {', '.join(COMPARATORS)}) run: "
                            'the primary cell (dense clock, primary budget) or every requested clock and budget')
        if name == 'develop':
            q.add_argument('--primary-budget', default='L3', choices=('L3', 'L5'))
        else:
            q.add_argument('--lock-sha256', required=True, help='sha256 of the committed, pushed lock file')
            q.add_argument('--no-git-check', action='store_true',
                           help='allow a run where git is unavailable (recorded in every evaluate cell)')
    a = p.parse_args(argv)
    cfg = load_config(a.config, [])
    torch.backends.cuda.matmul.allow_tf32 = False
    v3.verify_lock(cfg)
    ctx = Context(cfg, a.device)
    dev, confirm = read_partition(ctx)
    if a.stage != 'g4':
        disk_check((ctx.cache_root, ctx.out_root), a.min_free_gb)
    if a.stage == 'prepare-caches':
        stage_prepare(ctx, a.subjects, a.clocks, confirm)
        return
    if a.stage == 'g4':
        if not v4.g0_passed():
            raise SystemExit('G0 has not passed under the run_v4 core (results/v4/g0.json)')
        stage_g4(ctx, a.subjects or dev, dev, confirm)
        return
    references, pool = reference_pool(ctx, a.subject, dev, confirm)
    if a.stage == 'develop':
        arms = expand_arms(a.arms or DEVELOP_DEFAULT, 'DEV', len(references))
        check_develop(ctx, a.subject, arms, dev)
        primary = (PRIMARY_CLOCK, a.primary_budget)
        plan = fit_plan(a.clocks, a.budgets, arms, primary if a.scope == 'primary' else None)
        stage_fit(ctx, a.subject, references, plan, a.sections, CELL_STAGE['develop'], pool, primary)
        return
    lock = check_lock(ctx, a.subject, no_git_check=a.no_git_check, lock_sha256=a.lock_sha256)
    data = check_confirm(ctx, lock, a.subject, references)
    need = plan_cells(lock)
    pl = lock['confirmatory_plan']
    primary = (pl['clock'], pl['budget'])
    arms = expand_arms(a.arms or CONFIRM_DEFAULT, 'CONFIRM', len(references))
    plan = fit_plan(a.clocks, a.budgets, arms, primary if a.scope == 'primary' else None,
                    extra=None if a.arms else need)
    missing = uncovered(ctx, a.subject, data, need, plan, a.sections)
    if missing:
        raise SystemExit(f'confirm refused: the request leaves out {len(missing)} cells the locked plan reads and that '
                         f'are not on disk: {missing[:12]}')
    stage_fit(ctx, a.subject, references, plan, a.sections, CELL_STAGE['confirm'], pool, primary)


if __name__ == '__main__':
    main()
