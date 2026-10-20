#!/usr/bin/env python3
"""PROTOCOL_V4 primary analysis (FUTURE_WORK_V4 §6) and descriptive summaries; hashed into experiments/v4_lock.json.

Modes (one per call):
  --confirm         The registered primary analysis on the LPP CONFIRM cells. It runs exactly once: it refuses
                    when results/v4/primary.json, PRIMARY.md or the start marker results/v4/primary_started.json
                    exists, and it runs only under a confirmatory lock (a lock that reserves CONFIRM is
                    refused) whose fingerprints (FINGERPRINTS) equal the running code, whose confirmatory plan
                    equals ``plan(...)`` of this script, and which is committed, unmodified and pushed to
                    origin/main (where git is unavailable only with --no-git-check, recorded in primary.json).
                    The locked code must be the committed, pushed tree (``git_tree``: every file the lock pins,
                    at the lock's commit on origin/main, and every pinned code file as it runs here); where git
                    is unavailable this is accepted only from a lock whose own git check was verified, never
                    from a lock written with --no-git-check (verify such a lock with scripts/v4_lock.py
                    --verify and run --confirm where git is available).
                    1. Endpoint-blind QC (``qc``) of every cell the confirmatory family reads, recorded in
                       results/v4/primary_qc.json before any endpoint value is used. Each hypothesis has its own
                       sample: a subject is excluded from H_k only for a problem in a cell H_k reads (at H_k's
                       endpoint) or a subject-level problem (TR counts, no preparation record), so a failed
                       secondary cell never changes the H1 sample. Registered exclusions (§6): missing or
                       corrupt files (no score block, fields or provenance section differing from the path),
                       TR-count mismatch, W0 not estimable. Unregistered (endpoint-blind, but not in the §6
                       list): another arm not estimable, or a present but non-finite score; any of these refuses
                       the run unless --deviation names the PROTOCOL_V4 §4 entry recorded for it. A cell from
                       other code or configuration, one whose reference arm did not use the registered
                       references, or one failing its writer's provenance audit (scripts/audit_v4.py;
                       src/priors/tanghuth.py for Module H arms) refuses the run. The registered references
                       of a target are the other EN listeners of the partition that pass the subject-level QC
                       (a preparation record with the registered section TR counts): a listener excluded for
                       its TR counts has no prepared data, so no target can use it.
                       Completeness (``qc``'s ``completeness``), before the start marker:
                       - any 'missing file' exclusion, although registered, is applied only with
                         --accept-missing REASON (recorded in primary.json and the start marker), so a premature
                         launch or an unsynced results folder cannot silently shrink a sample;
                       - a systematic omission (a cell the family reads, missing at some section for every
                         subject that passes the subject-level QC), a subject that passes it but has none of the
                         cells (never run), or a hypothesis whose QC sample is below 2 (not testable) points to a
                         run that did not happen, not to a data defect: refused unless --deviation names the
                         PROTOCOL_V4 §4 entry recorded for it;
                       - H1 with fewer than 2 subjects is always refused.
                    2. The start marker is created (exclusively) and only then are endpoint values read. A run
                       that fails after this point is not rerun silently: removing the marker is a deliberate
                       act to be recorded as a PROTOCOL_V4 §4 deviation first.
                    3. H1 on its sample at the locked primary budget (L3, or L5 after the G1 fallback), dense
                       clock, raw endpoint. R_s(arm) is the mean over the 9 outer folds of the test section's
                       session-macro
                       r (each section is its own session). Δ1 = A − W1 and Δ2 = A − W0, with A the advanced
                       arm (EB; PRIOR under the substitution rule). One-sided sign-flip p per Δ; CONFIRMED iff
                       both p < 0.025 and both means > 0 (``v4_stats.iut``). Each contrast reports the mean,
                       BCa 95% CI, d_z, positives with a Clopper-Pearson CI, the oracle fraction
                       (A − W0)/(ORACLE − W0) and the SESOI tier of the 90% BCa CI. The crossed LMM runs on the
                       per-(subject, section) Δ of both contrasts; if H1 passes and the LMM does not, the claim
                       is restricted to "across listeners of this book".
                    4. Fixed sequence (``v4_stats.fixed_sequence``); a step's cells are read only when every
                       predecessor is confirmed, on that step's own QC sample (fewer than 2 subjects, possible
                       only under --deviation: NOT TESTABLE, which stops the sequence like a failure):
                       H2  the H1 intersection-union test on the residualized endpoint;
                       H3  the mean per-subject OLS slope of (A[f] − W1) on f ∈ {0, .25, .5, .75, 1} > 0
                           (A[1] = A; ladder arms ``A_f<f>``, run_v4 naming), by sign-flip;
                       H4  A − EB_SELF > 0;
                       H5  A − TH_SEL > 0;
                       H6  at L8, mean(A − W1) > −0.005, by sign-flip on Δ + 0.005; dropped when the lock drops
                           it (G3 failed) or PRIOR was advanced.
                    Writes results/v4/primary.json and results/v4/PRIMARY.md.
  --qc              Step 1 alone, as a pre-check (no endpoint value is used); refused once the primary started.
  --dev             Descriptive LPP DEV summary: budget curves (L3/L5/L8, both clocks, raw and residualized),
                    contrasts against W0 and W1, oracle fractions, the development d_z and SD(Δ), the κ schedule
                    and the R_P(K) reliability fit. Writes results/v4/lpp_dev_summary.json and LPP_DEV.md.
  --ds003020        Descriptive ds003020 budget curves (S3/S6/S15/ALL, both clocks) at --stage develop
                    (validation) or evaluate (exposed test; only once the lock exists), with per-subject
                    story-level sign-flips, oracle fractions and the κ schedule, and Module H's registered
                    two-reference sensitivity analysis (FUTURE_WORK_V4 §5 TH-SEL step 3, §8.5;
                    ``ds_sensitivity``): TH_SEL / XDEC per reference pair against the three-reference cells
                    (with per-subject story-level sign-flips), W0 and W1. Writes
                    results/v4/ds003020_<stage>_summary.json and DS003020_<STAGE>.md.
  --confirm-curves  Descriptive LPP CONFIRM budget curves, only after the primary analysis has run.
Cell layouts: LPP results/v4_lpp/<sub>/<clock>/<stage>/cells/t<t>_<budget>_<arm>.json with a ``test`` score
block and a ``residualized`` block (a score dict, or one under ``test``), Module G's two-reference sensitivity
cells under <stage>/sensitivity/refs-<R1>-<R2>/cells/ (``lpp_sensitivity``) and its §9 control records
<stage>/controls/<budget>_<arm>.json (``lpp_controls``), both descriptive (--dev, --confirm-curves); ds003020
results/v4/<S>/<clock>/<stage>/cells/<budget>_<arm>.json with ``val`` (develop) or ``val`` and ``test``
(evaluate); Module H's two-reference sensitivity cells under <stage>/sensitivity/refs-<R1>-<R2>/cells/ (or
named ``<budget>_<arm>__refs-<R1>-<R2>.json``) are summarized apart from the budget curves. Every statistic comes from
src/v4_stats.py with its registered seeds. Non-finite numbers are written as null.
Lock fingerprints (FINGERPRINTS; one schema shared with the readers of the lock):
  code_sha256                run_v4.code_sha256() (the frozen core; run_v4 ds003020 cells carry it)
  config_sha256              run_v3.config_sha256 of configs/v3/structured_ridge.yaml
  lpp_cell_code_sha256       run_v4_lpp.code_sha256() (LPP cells and results/v4_lpp/g4.json carry it)
  lpp_runner_sha256          sha256 of scripts/run_v4_lpp.py
  tanghuth_sha256            sha256 of src/priors/tanghuth.py
  tanghuth_runner_sha256     sha256 of scripts/run_v4_tanghuth.py
  tanghuth_cell_code_sha256  run_v4_tanghuth.code_sha256() (Module H ds003020 cells carry it)
  summarize_sha256           sha256 of this script
all computed from the running code (never from --root, which only locates results, data and the lock).
run_v4_lpp.py confirm and run_v4_tanghuth.py evaluate check their subsets; scripts/v4_lock.py runs both
checks on the lock before writing it.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import lpp_prepare  # noqa: E402
from src import budgets as bud  # noqa: E402
from src import v4_stats as st  # noqa: E402
from src.provenance import SCHEMA  # noqa: E402
from src.runtime import save_json  # noqa: E402

VAL = 'session_macro_feature_correlation'
CONFIG = Path('configs/v3/structured_ridge.yaml')
LOCK = Path('experiments/v4_lock.json')
OUT = Path('results/v4')
PRIMARY_JSON, PRIMARY_MD = OUT / 'primary.json', OUT / 'PRIMARY.md'
QC_JSON, STARTED = OUT / 'primary_qc.json', OUT / 'primary_started.json'
LPP_DATA = Path('data/lpp')
SECTIONS = bud.LPP_SECTIONS
SECTION_TRS = tuple(lpp_prepare.SECTION_TRS)
CLOCK = 'dense'  # Primary clock (registered decision 1).
CLOCKS = ('dense', 'stride4')
STAGE_DIRS = {'DEV': 'develop', 'CONFIRM': 'evaluate'}  # LPP cell folders = the cells' provenance stage.
LPP_BUDGETS = tuple(bud.LPP_SIZES)
DS_BUDGETS = bud.DS003020_BUDGETS
DS_DEV_SUBJECTS = ('UTS01', 'UTS02', 'UTS03', 'UTS05', 'UTS06', 'UTS07', 'UTS08', 'UTS09', 'UTS10')  # run_v4
ADVANCEABLE = ('EB', 'PRIOR')
LADDER_F = (0.0, 0.25, 0.5, 0.75, 1.0)
H6_BUDGET, H6_MARGIN = 'L8', 0.005
ORACLE = 'ORACLE'
RELIABILITY_K = 48  # CONFIRM targets use all 48 other EN listeners.
FINGERPRINTS = ('code_sha256', 'config_sha256', 'lpp_cell_code_sha256', 'lpp_runner_sha256', 'tanghuth_sha256',
                'tanghuth_runner_sha256', 'tanghuth_cell_code_sha256', 'summarize_sha256')  # Module docstring.
TANGHUTH_ARMS = ('TH_SEL', 'TH_SEL_rand', 'XDEC')  # src.priors.tanghuth.ARMS: Module H cells.
REFERENCE_ARMS = ('PRIOR', 'EB', 'EB_MISALIGN', 'TH_SEL', 'XDEC')  # Plus the EB_f / PRIOR_f ladders.
MISSING = 'missing file'
REGISTERED_EXCLUSIONS = (MISSING, 'corrupt file', 'TR-count mismatch', 'W0 not estimable')
UNREGISTERED_EXCLUSIONS = ('arm not estimable', 'non-finite score')  # Endpoint-blind, not in §6: --deviation.
REFUSALS = ('other code', 'audit failed', 'wrong references')
NOT_ESTIMABLE = ('W0 not estimable', 'arm not estimable')
SCORE_KEYS = ('val', 'test', 'test_shift_null', 'residualized')
LPP_NAME = re.compile(r'^t(?P<section>[1-9])_(?P<budget>L3|L5|L8)_(?P<arm>.+)\.json$')
DS_NAME = re.compile(r'^(?P<budget>S3|S6|S15|ALL)_(?P<arm>[^_].*?)(?:__refs-(?P<refs>[A-Za-z0-9-]+))?\.json$')
HEX64 = re.compile(r'^[0-9a-f]{64}$')


class Refused(SystemExit):
    """The run is refused (exit status 1 with the reason); nothing past the refusal is read or written."""

    def __init__(self, reason):
        super().__init__(f'summarize_v4 refused: {reason}')
        self.reason = reason


# ------------------------------------------------------------------------------------------- utilities
def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def clean(obj):
    """JSON-ready copy: numpy scalars and arrays to Python, tuples to lists, non-finite floats to None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in (sorted(obj) if isinstance(obj, set) else obj)]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    return obj


def write_json(obj, path):
    save_json(clean(obj), path)


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def fmt(value, digits=4, sign=False):
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return '—'
    return f'{value:+.{digits}f}' if sign else f'{value:.{digits}f}'


def lpp_cell(root, subject, clock, stage, section, budget, arm):
    """results/v4_lpp/<sub>/<clock>/<stage>/cells/t<t>_<budget>_<arm>.json (V4_IMPLEMENTATION Module G)."""
    return Path(root) / 'results' / 'v4_lpp' / subject / clock / stage / 'cells' / f't{section}_{budget}_{arm}.json'


def ds_cell(root, subject, clock, stage, budget, arm):
    """results/v4/<S>/<clock>/<stage>/cells/<budget>_<arm>.json (scripts/run_v4.py, run_v4_tanghuth.py)."""
    return Path(root) / 'results' / 'v4' / subject / clock / stage / 'cells' / f'{budget}_{arm}.json'


def score_block(cell, endpoint):
    """The score dict of an LPP cell: ``test`` (raw) or ``residualized`` (a score dict or one under ``test``)."""
    if endpoint == 'raw':
        block = cell.get('test')
    elif endpoint == 'residualized':
        block = cell.get('residualized')
        if isinstance(block, dict) and VAL not in block and isinstance(block.get('test'), dict):
            block = block['test']
    else:
        raise ValueError(f'Unknown endpoint {endpoint!r}')
    return block if isinstance(block, dict) else None


def finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def endpoint_value(cell, endpoint):
    """The section-macro r of an LPP cell at ``endpoint``; ValueError when it is missing or not finite."""
    block = score_block(cell, endpoint)
    value = None if block is None else block.get(VAL)
    if not finite_number(value):
        raise ValueError(f'No finite {endpoint} {VAL} in the cell')
    return float(value)


# ----------------------------------------------------------------------------------------- fingerprints
def file_fingerprint(name):
    """sha256 of a file of the running code tree (ROOT, the tree this script was loaded from)."""
    path = ROOT / name
    if not path.is_file():
        raise Refused(f'{name} does not exist; it is fingerprinted into the lock')
    return sha256_file(path)


def module_fingerprint(module_name):
    """``<module>.code_sha256()`` of a runner script of the running tree (the value its cells carry)."""
    try:
        module = importlib.import_module(module_name)
        value = module.code_sha256()
    except (ImportError, AttributeError, OSError) as error:
        raise Refused(f'scripts/{module_name}.py has no usable code_sha256(): {error!r}') from error
    where = Path(module.__file__).resolve()
    if where != ROOT / 'scripts' / f'{module_name}.py':
        raise Refused(f'{module_name} was imported from {where}, not from this code tree ({ROOT})')
    return value


def current_fingerprints():
    """Every fingerprint the lock pins (FINGERPRINTS, module docstring), computed from the running code.

    Code identity never comes from ``--root``: a pristine tree holding results must not vouch for an edited
    script that is actually executing.
    """
    import run_v3
    import run_v4
    from src.runtime import load_config
    fp = dict(code_sha256=run_v4.code_sha256(),
              config_sha256=run_v3.config_sha256(load_config(ROOT / CONFIG, [])),
              lpp_cell_code_sha256=module_fingerprint('run_v4_lpp'),
              lpp_runner_sha256=file_fingerprint('scripts/run_v4_lpp.py'),
              tanghuth_sha256=file_fingerprint('src/priors/tanghuth.py'),
              tanghuth_runner_sha256=file_fingerprint('scripts/run_v4_tanghuth.py'),
              tanghuth_cell_code_sha256=module_fingerprint('run_v4_tanghuth'),
              summarize_sha256=sha256_file(Path(__file__).resolve()))
    return fp


def lock_mismatches(lock, fingerprints):
    """FINGERPRINTS keys whose locked value is absent or differs from the current one."""
    return sorted(k for k in FINGERPRINTS if lock.get(k) is None or lock.get(k) != fingerprints.get(k))


def ds_code_problem(cell, fp):
    """None if a ds003020 cell is from the current code and configuration, else why not.

    Module H cells (``block`` 'published-method' or a TANGHUTH_ARMS arm) carry ``tanghuth_cell_code_sha256`` as
    ``code_sha256`` plus the core and Module H file hashes they were written under; every other cell carries the
    run_v4 core. A Module H arm under the bare core fingerprint is therefore refused as stale.
    """
    module_h = cell.get('block') == 'published-method' or cell.get('arm') in TANGHUTH_ARMS
    want, what = ((fp['tanghuth_cell_code_sha256'], 'Module H') if module_h else (fp['code_sha256'], 'run_v4 core'))
    if cell.get('code_sha256') != want:
        return f"code_sha256 {cell.get('code_sha256')} is not the current {what} fingerprint"
    if module_h:
        stale = [k for k, f in (('core_sha256', 'code_sha256'), ('tanghuth_sha256', 'tanghuth_sha256'),
                                ('tanghuth_runner_sha256', 'tanghuth_runner_sha256')) if cell.get(k) != fp[f]]
        if stale:
            return f'Module H fields {stale} are not the current code'
    if cell.get('config_sha256') != fp['config_sha256']:
        return f"config_sha256 {cell.get('config_sha256')} is not the current configuration"
    return None


def reads_references(arm):
    """True for the arms whose ledger lists reference listeners (ISP/EB arms, their ladders, TH_SEL, XDEC)."""
    return str(arm) in REFERENCE_ARMS or re.fullmatch(r'(EB|PRIOR)_f[0-9.]+', str(arm)) is not None


def embedded_logs(obj):
    """Every provenance log (``Provenance.to_dict``) embedded in a cell, depth-first."""
    if isinstance(obj, dict):
        if obj.get('schema') == SCHEMA:
            yield obj
            return
        for value in obj.values():
            yield from embedded_logs(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from embedded_logs(value)


def logged_sections(cell):
    """The outer test sections the cell's provenance logs scored (``test_items``), or None without a log."""
    logs = list(embedded_logs(cell))
    if not logs:
        return None
    return {str(i.get('stimulus')) for log in logs for i in (log.get('test_items') or []) if isinstance(i, dict)}


def audit_problem(path, cell, audit):
    """None if the cell passes its writer's audit, else the failed rules.

    A cell without any score (not estimable; no val/test/residualized block) may fail only on the scope errors its
    writer recorded in ``scope_errors`` (Module G writes such cells instead of a score); any other violation, or
    any violation of a scored cell, is a failure.
    """
    result = audit(path)
    if result.get('passed'):
        return None
    violations = result.get('violations') or []
    recorded = set(cell.get('scope_errors') or [])
    unscored = cell.get('estimable') is not True and not any(k in cell for k in SCORE_KEYS)
    if unscored and recorded and violations and all(v.get('rule') == 'scope' and v.get('reason') in recorded
                                                    for v in violations):
        return None
    return str(result.get('rules') or result.get('error'))


# -------------------------------------------------------------------------------------------- the plan
def ladder_arm(arm, f):
    """Noise-ladder arm at reliability f (run_v4 naming ``EB_f0.75`` ... ``EB_f0``); f = 1 is the arm itself."""
    return str(arm) if float(f) == 1.0 else f'{arm}_f{float(f):g}'


def plan(advanced, drop_h6, budget):
    """The confirmatory family under the substitution rule (FUTURE_WORK_V4 §6), as the lock records it.

    ``advanced`` is EB or PRIOR (PRIOR replaces EB in H1-H5 and H6 is dropped); ``budget`` is the LPP primary
    budget (L3, or L5 after the G1 fallback).
    """
    if advanced not in ADVANCEABLE:
        raise ValueError(f'The advanced arm must be one of {ADVANCEABLE}, got {advanced!r}')
    if budget not in ('L3', 'L5'):
        raise ValueError(f'The LPP primary budget is L3 or L5, got {budget!r}')
    if not isinstance(drop_h6, (bool, np.bool_)):
        raise ValueError(f'drop_h6 must be a bool, got {drop_h6!r}')
    a = advanced
    dropped = bool(drop_h6) or a == 'PRIOR'
    hypotheses = dict(
        H1=dict(test='iut', endpoint='raw', budget=budget, contrasts=[[a, 'W1'], [a, 'W0']], lmm=True,
                text=f'{a} − W1 > 0 and {a} − W0 > 0 (intersection-union)'),
        H2=dict(test='iut', endpoint='residualized', budget=budget, contrasts=[[a, 'W1'], [a, 'W0']],
                text='H1 on the residualized endpoint'),
        H3=dict(test='slope', endpoint='raw', budget=budget, f=list(LADDER_F),
                ladder=[ladder_arm(a, f) for f in LADDER_F], comparator='W1',
                text=f'mean per-subject OLS slope of ({a}[f] − W1) on f > 0'),
        H4=dict(test='positive', endpoint='raw', budget=budget, contrasts=[[a, 'EB_SELF']], null=0.0,
                text=f'{a} − EB_SELF > 0'),
        H5=dict(test='positive', endpoint='raw', budget=budget, contrasts=[[a, 'TH_SEL']], null=0.0,
                text=f'{a} − TH_SEL > 0'),
        H6=dict(test='positive', endpoint='raw', budget=H6_BUDGET, contrasts=[[a, 'W1']], null=-H6_MARGIN,
                text=f'at {H6_BUDGET}, mean({a} − W1) > −{H6_MARGIN} (sign-flip on Δ + {H6_MARGIN})'))
    order = ['H1', 'H2', 'H3', 'H4', 'H5'] + ([] if dropped else ['H6'])
    return dict(advanced=a, budget=budget, clock=CLOCK, cell_stage=STAGE_DIRS['CONFIRM'], drop_h6=dropped,
                order=order, dropped=['H6'] if dropped else [], alpha=st.ALPHA, sesoi=st.SESOI,
                hypotheses={k: hypotheses[k] for k in order})


def hypothesis_arms(spec):
    arms = [x for c in spec.get('contrasts', []) for x in c] + list(spec.get('ladder', []))
    return arms + ([spec['comparator']] if 'comparator' in spec else [])


def hypothesis_cells(spec):
    """[(budget, arm)] of the cells one hypothesis reads (at its endpoint)."""
    return [(spec['budget'], arm) for arm in dict.fromkeys(hypothesis_arms(spec))]


def required_cells(pl):
    """{(budget, arm): sorted endpoints} of every cell the confirmatory family reads."""
    need = {}
    for spec in pl['hypotheses'].values():
        for arm in hypothesis_arms(spec):
            need.setdefault((spec['budget'], arm), set()).add(spec['endpoint'])
    return {k: sorted(v) for k, v in sorted(need.items())}


def oracle_budgets(pl):
    """Budgets whose raw-endpoint contrasts report the oracle fraction (ORACLE cells, descriptive)."""
    return sorted({s['budget'] for s in pl['hypotheses'].values() if s['endpoint'] == 'raw' and 'contrasts' in s})


# ------------------------------------------------------------------------------------------------- QC
def prepared_trs(data_root):
    """{subject: {TR-count tuples in section order}} from the LPP preparation records (G4(a) run lengths).

    Reads every ``prepare_provenance*.json`` under ``data_root`` (DEV, DEV re-run and CONFIRM batch records).
    """
    records = sorted(Path(data_root).glob('prepare_provenance*.json'))
    if not records:
        raise Refused(f'no LPP preparation records under {data_root}: the TR-count QC cannot run here')
    out = {}
    for path in records:
        record = json.loads(path.read_text())
        for subject, runs in ((record.get('g4a') or {}).get('runs') or {}).items():
            trs = tuple(int(r['n_tr']) for r in sorted(runs, key=lambda r: int(r['section'])))
            out.setdefault(subject, set()).add(trs)
    return out


def default_audit(path):
    """The provenance audit the cell's writer applied: scripts/audit_v4.py (R1-R7 and the registered scope), or,
    for a Module H arm (TANGHUTH_ARMS; Modules G and H write them only when ``tanghuth.audit_errors`` is empty),
    ``src.priors.tanghuth.audit_file``: the frozen audit plus the arm's own checks, with XDEC's own checks in place
    of exactly the frozen target-decoder checks (``tanghuth.XDEC_EXEMPT``)."""
    try:
        arm = json.loads(Path(path).read_text()).get('arm')
    except (OSError, ValueError, AttributeError):
        arm = None
    if arm in TANGHUTH_ARMS:
        from src.priors import tanghuth
        return tanghuth.audit_file(path)
    import audit_v4
    return audit_v4.audit_file(path)


def inspect_cell(path, expect, endpoints, code, config, audit, reference_pool=None):
    """(problem, endpoint_problems, sha256) of one LPP cell, endpoint-blind: score blocks are checked for presence
    and finiteness only.

    ``expect`` holds the subject, clock, stage, section, budget and arm the path names; ``reference_pool`` (when
    given) is the listener set whose other members every reference arm must have used. ``problem`` is None or
    dict(category, detail) for the whole cell: 'other code', 'audit failed' and 'wrong references' refuse the run;
    'missing file', 'corrupt file', 'W0 not estimable' and 'arm not estimable' exclude. ``endpoint_problems`` maps
    an endpoint of ``endpoints`` to its problem: no score block ('corrupt file') or a non-finite score.
    """
    path = Path(path)
    if not path.is_file():
        return dict(category='missing file', detail=path.name), {}, None
    data = path.read_bytes()
    digest = sha256_bytes(data)
    try:
        cell = json.loads(data)
        if not isinstance(cell, dict):
            raise ValueError('not a JSON object')
    except ValueError as error:
        return dict(category='corrupt file', detail=f'{path.name}: unreadable ({error})'), {}, digest
    if cell.get('code_sha256') != code:
        return dict(category='other code', detail=f"{path.name}: code_sha256 {cell.get('code_sha256')}"), {}, digest
    if cell.get('config_sha256') != config:
        return dict(category='other code', detail=f"{path.name}: config_sha256 {cell.get('config_sha256')}"), {}, digest
    wrong = [k for k in ('subject', 'arm', 'budget', 'stage', 'clock') if cell.get(k) != expect[k]]
    wrong += [k for k in ('test_section', 'section', 'outer_section')
              if (k == 'test_section' or k in cell) and str(cell.get(k)) != str(expect['section'])]
    logged = logged_sections(cell)
    if logged is not None and logged != {str(expect['section'])}:
        wrong.append(f'provenance test_items {sorted(logged)}')
    if wrong:
        return dict(category='corrupt file', detail=f'{path.name}: fields {wrong} differ from its path'), {}, digest
    if audit is not None:
        why = audit_problem(path, cell, audit)
        if why:
            return dict(category='audit failed', detail=f'{path.name}: {why}'), {}, digest
    if reference_pool is not None and reads_references(expect['arm']):
        want = sorted(str(s) for s in reference_pool if str(s) != expect['subject'])
        got = sorted(str(s) for s in ((cell.get('ledger') or {}).get('references') or []))
        if got != want:
            return dict(category='wrong references', detail=f'{path.name}: {len(got)} ledger references, not the '
                                                            f'{len(want)} registered'), {}, digest
    if cell.get('estimable') is not True:
        return dict(category=NOT_ESTIMABLE[0] if expect['arm'] == 'W0' else NOT_ESTIMABLE[1],
                    detail=path.name), {}, digest
    by_endpoint = {}
    for e in endpoints:
        block = score_block(cell, e)
        if block is None or VAL not in block:
            by_endpoint[e] = dict(category='corrupt file', detail=f'{path.name}: no {e} score block')
        elif not finite_number(block[VAL]):
            by_endpoint[e] = dict(category='non-finite score', detail=f'{path.name}: {e} {VAL} = {block[VAL]!r}')
    return None, by_endpoint, digest


def prepared_ok(prepared, subject):
    """The subject-level QC: a preparation record with exactly the registered section TR counts."""
    return prepared.get(str(subject)) == {SECTION_TRS}


def registered_pool(reference_pool, prepared):
    """The registered reference pool: the listeners of ``reference_pool`` that pass the subject-level QC
    (``prepared_ok``). A listener without a valid preparation record has no prepared data, so no target can use
    it as a reference; every other listener must be used (PROTOCOL_V4 §4 records this reading of "all other
    cohort listeners")."""
    return None if reference_pool is None else [str(s) for s in reference_pool if prepared_ok(prepared, s)]


def qc(root, subjects, pl, code, prepared, config, audit=default_audit, reference_pool=None):
    """Endpoint-blind QC of every cell the confirmatory family (and its oracle fractions) reads.

    Per hypothesis H_k (``samples``): a subject is excluded from H_k, with every reason logged, for a subject-level
    problem (TR-count mismatch with the registered section lengths, no preparation record) or a problem in a cell
    H_k reads at H_k's endpoint: missing or corrupt, W0 not estimable, or (UNREGISTERED_EXCLUSIONS) another arm
    not estimable or a non-finite score. ``included``/``excluded`` are H1's sample (the primary n). A cell from
    other code or configuration, with the wrong reference set (``registered_pool``: the other listeners of
    ``reference_pool`` that pass the subject-level QC) or failing its writer's provenance audit is a refusal, not
    an exclusion. ORACLE cells (and W0 at a budget the family does not otherwise read) are descriptive: a subject
    without them stays in every sample and an oracle fraction uses the subjects of the hypothesis's sample that
    have complete ORACLE and W0 cells at its budget. No endpoint value is used.

    ``completeness`` records what ``run_primary`` checks before the start marker: the sample size of every
    hypothesis and those below 2 (``untestable``), every missing family cell of a subject that passes the
    subject-level QC (``missing``: '<budget> <arm>' -> {subject: sections}), the subjects excluded anywhere for a
    'missing file' (``missing_file_exclusions``), the systematic omissions (``systematic``: a family cell missing
    at a section for every subject that passes the subject-level QC) and the subjects that pass it but have none
    of the family's cells (``unrun``).
    """
    root = Path(root)
    need, stage = required_cells(pl), pl['cell_stage']
    files, refusals = {}, []
    oracle = {b: [] for b in oracle_budgets(pl)}
    samples = {name: dict(included=[], excluded={}) for name in pl['order']}
    pool = registered_pool(reference_pool, prepared)
    missing, passing = {}, []  # (budget, arm) -> {subject: [sections]}; subjects passing the subject-level QC.

    def check(s, t, budget, arm, endpoints):
        path = lpp_cell(root, s, pl['clock'], stage, t, budget, arm)
        expect = dict(subject=s, clock=pl['clock'], stage=stage, section=t, budget=budget, arm=arm)
        problem, by_endpoint, digest = inspect_cell(path, expect, endpoints, code, config, audit, pool)
        if digest is not None:
            files[path.relative_to(root).as_posix()] = digest
        if problem is not None and problem['category'] in REFUSALS:
            refusals.append(dict(subject=s, **problem))
        return problem, by_endpoint

    for s in subjects:
        common = []
        trs = prepared.get(s)
        if not trs:
            common.append(dict(category=MISSING, detail='no LPP preparation record'))
        elif trs != {SECTION_TRS}:
            common.append(dict(category='TR-count mismatch', detail=f'{sorted(trs)} != {list(SECTION_TRS)}'))
        if not common:
            passing.append(s)
        found = {}  # (budget, arm, endpoint) -> excluding problems of that cell's 9 folds.
        for (budget, arm), endpoints in need.items():
            for t in SECTIONS:
                problem, by_endpoint = check(s, t, budget, arm, endpoints)
                if problem is not None and problem['category'] == MISSING and not common:
                    missing.setdefault((budget, arm), {}).setdefault(s, []).append(t)
                for e in endpoints:
                    p = problem if problem is not None else by_endpoint.get(e)
                    if p is not None and p['category'] not in REFUSALS:
                        found.setdefault((budget, arm, e), []).append(p)
        for name in pl['order']:
            spec = pl['hypotheses'][name]
            reasons = common + [p for key in hypothesis_cells(spec) for p in found.get(key + (spec['endpoint'],), [])]
            if reasons:
                samples[name]['excluded'][s] = reasons
            else:
                samples[name]['included'].append(s)
        for budget in oracle:  # The oracle fraction reads ORACLE and W0 at the contrast's budget (raw).
            bad = False
            for arm in (ORACLE, 'W0'):
                if 'raw' in need.get((budget, arm), ()):
                    bad = bad or bool(found.get((budget, arm, 'raw')))
                    continue
                for t in SECTIONS:  # QC every fold.
                    problem, by_endpoint = check(s, t, budget, arm, ['raw'])
                    bad = bad or problem is not None or bool(by_endpoint)
            if not bad:
                oracle[budget].append(s)
    first = samples[pl['order'][0]]
    categories = sorted({r['category'] for v in samples.values() for rs in v['excluded'].values() for r in rs})
    return dict(subjects=list(subjects), included=list(first['included']), excluded=dict(first['excluded']),
                samples=samples, refusals=refusals, oracle_subjects=oracle, files=files,
                required=[dict(budget=b, arm=a, endpoints=e) for (b, a), e in need.items()],
                reference_pool=dict(registered=None if reference_pool is None else [str(s) for s in reference_pool],
                                    used=pool, not_prepared=None if pool is None else
                                    [str(s) for s in reference_pool if str(s) not in pool]),
                completeness=completeness(pl, samples, need, missing, passing),
                unregistered_exclusions=[c for c in categories if c in UNREGISTERED_EXCLUSIONS],
                endpoint_values_used=0)


def completeness(pl, samples, need, missing, passing):
    """The ``completeness`` block of ``qc`` (its docstring) from the per-subject missing family cells."""
    n_cells = len(need) * len(SECTIONS)
    unrun = [s for s in passing if sum(len(v.get(s, ())) for v in missing.values()) == n_cells]
    systematic = []
    for (budget, arm), by_subject in sorted(missing.items()):
        sections = [t for t in SECTIONS if passing and all(t in by_subject.get(s, ()) for s in passing)]
        if sections:
            systematic.append(dict(budget=budget, arm=arm, sections=sections))
    sizes = {name: len(samples[name]['included']) for name in pl['order']}
    excluded = sorted({s for v in samples.values() for s, rs in v['excluded'].items()
                       if any(r['category'] == MISSING for r in rs)})
    return dict(sample_sizes=sizes, untestable=[name for name in pl['order'] if sizes[name] < 2],
                passing_subject_qc=len(passing), unrun=unrun, systematic=systematic,
                missing={f'{budget} {arm}': by_subject for (budget, arm), by_subject in sorted(missing.items())},
                missing_file_exclusions=excluded)


# ------------------------------------------------------------------------------------------- analysis
class Scores:
    """Endpoint values of QC'd LPP cells, read lazily and checked against the QC digests.

    ``per_section(arm, budget, endpoint, subjects)`` is [subjects, 9 sections]; ``means`` gives R_s, the mean over
    the 9 outer folds. ``read`` lists the (arm, budget, endpoint, subjects) blocks read, in order, with the mean
    R_s over those subjects.
    """

    def __init__(self, root, clock, stage, files):
        self.root, self.clock, self.stage, self.files = Path(root), clock, stage, files
        self.cache, self.read = {}, []

    def per_section(self, arm, budget, endpoint, subjects):
        subjects = list(subjects)
        key = (arm, budget, endpoint, tuple(subjects))
        if key not in self.cache:
            out = np.empty((len(subjects), len(SECTIONS)))
            for i, s in enumerate(subjects):
                for j, t in enumerate(SECTIONS):
                    path = lpp_cell(self.root, s, self.clock, self.stage, t, budget, arm)
                    data = path.read_bytes()
                    rel = path.relative_to(self.root).as_posix()
                    if self.files.get(rel) != sha256_bytes(data):
                        raise Refused(f'{rel} changed after QC (or was not QC-checked)')
                    out[i, j] = endpoint_value(json.loads(data), endpoint)
            self.cache[key] = out
            self.read.append(dict(arm=arm, budget=budget, endpoint=endpoint, subjects=len(subjects),
                                  mean=float(out.mean(axis=1).mean())))
        return self.cache[key]

    def means(self, arm, budget, endpoint, subjects):
        return self.per_section(arm, budget, endpoint, subjects).mean(axis=1)


def contrast(scores, subjects, arm, comparator, budget, endpoint, null=0.0, oracle_subjects=None):
    """Everything §6 reports with one per-subject contrast Δ_s = R_s(arm) − R_s(comparator) over ``subjects``."""
    subjects = list(subjects)
    a, b = scores.means(arm, budget, endpoint, subjects), scores.means(comparator, budget, endpoint, subjects)
    d = a - b
    out = dict(arm=arm, comparator=comparator, budget=budget, endpoint=endpoint, mean_arm=float(a.mean()),
               mean_comparator=float(b.mean()), **st.contrast_summary(d, null=null))
    out['deltas'] = {s: float(v) for s, v in zip(subjects, d)}
    out.update(oracle_fraction=None, comparator_oracle_fraction=None, oracle_n=0, oracle_gain=None)
    if endpoint == 'raw' and oracle_subjects:
        index = [i for i, s in enumerate(subjects) if s in set(oracle_subjects)]
        if index:
            sub = [subjects[i] for i in index]
            orc, w0 = scores.means(ORACLE, budget, 'raw', sub), scores.means('W0', budget, 'raw', sub)
            out.update(oracle_fraction=st.oracle_fraction(a[index], w0, orc),
                       comparator_oracle_fraction=st.oracle_fraction(b[index], w0, orc), oracle_n=len(index),
                       oracle_gain=float(np.mean(orc - w0)))
    return out


def section_lmm(scores, subjects, arm, comparator, budget, endpoint):
    """Crossed LMM Δ_st ~ 1 + (1 | subject) + (1 | section) on the per-(subject, section) differences."""
    import pandas as pd
    subjects = list(subjects)
    a = scores.per_section(arm, budget, endpoint, subjects)
    b = scores.per_section(comparator, budget, endpoint, subjects)
    frame = pd.DataFrame([dict(subject=s, section=t, delta=float(a[i, j] - b[i, j]))
                          for i, s in enumerate(subjects) for j, t in enumerate(SECTIONS)])
    head = dict(arm=arm, comparator=comparator, budget=budget, endpoint=endpoint)
    try:
        result = st.crossed_lmm(frame)
    except RuntimeError as error:  # No optimizer and no lme4 fallback: recorded, never a silent pass.
        return dict(head, error=str(error), supports=False)
    return dict(head, **result, supports=bool(result['p_one_sided'] < st.ALPHA and result['estimate'] > 0))


def test_hypothesis(scores, spec, subjects, oracle_subjects=None):
    """One hypothesis on its QC sample ``subjects``: its statistics and a bool ``confirmed`` (the
    ``v4_stats.fixed_sequence`` input). Fewer than 2 subjects: ``testable`` False (verdict NOT TESTABLE, which
    stops the sequence like a failure; ``run_primary`` refuses this unless --deviation), nothing is read."""
    subjects = list(subjects)
    oracle = (oracle_subjects or {}).get(spec['budget'])
    if len(subjects) < 2:
        return dict(spec=spec, n=len(subjects), subjects=subjects, confirmed=False, testable=False,
                    error=f'{len(subjects)} subjects pass the endpoint-blind QC for this hypothesis (needs >= 2)')
    if spec['test'] == 'iut':
        c1, c2 = [contrast(scores, subjects, a, c, spec['budget'], spec['endpoint'], oracle_subjects=oracle)
                  for a, c in spec['contrasts']]
        out = dict(spec=spec, n=len(subjects), contrasts=[c1, c2],
                   confirmed=st.iut(c1['p_one_sided'], c2['p_one_sided'], c1['mean'], c2['mean']))
        if spec.get('lmm'):
            out['lmm'] = [section_lmm(scores, subjects, a, c, spec['budget'], spec['endpoint'])
                          for a, c in spec['contrasts']]
            out['lmm_supports'] = all(x['supports'] for x in out['lmm'])
            out['claim'] = (('generalizes over listeners; the crossed LMM also supports generalization over sections'
                             if out['lmm_supports'] else 'restricted to "across listeners of this book" (the '
                             'crossed LMM does not support generalization over sections)')
                            if out['confirmed'] else 'not confirmed')
        return out
    if spec['test'] == 'slope':
        f = np.asarray(spec['f'], dtype=np.float64)
        base = scores.means(spec['comparator'], spec['budget'], spec['endpoint'], subjects)
        y = np.stack([scores.means(arm, spec['budget'], spec['endpoint'], subjects) - base
                      for arm in spec['ladder']], axis=1)
        slopes = st.ols_slopes(f, y)
        summary = st.contrast_summary(slopes)
        summary.update(sesoi=None, sesoi_tier=None)  # SESOI is in r units; a slope has none.
        return dict(spec=spec, **summary, slopes={s: float(v) for s, v in zip(subjects, slopes)},
                    mean_by_f={f'{x:g}': float(v) for x, v in zip(f, y.mean(axis=0))},
                    confirmed=bool(summary['p_one_sided'] < st.ALPHA and summary['mean'] > 0))
    if spec['test'] == 'positive':
        (a, c), null = spec['contrasts'][0], float(spec.get('null', 0.0))
        out = contrast(scores, subjects, a, c, spec['budget'], spec['endpoint'], null=null, oracle_subjects=oracle)
        return dict(spec=spec, n=len(subjects), contrasts=[out],
                    confirmed=bool(out['p_one_sided'] < st.ALPHA and out['mean'] > null))
    raise ValueError(f"Unknown test {spec['test']!r}")


def analyze(scores, pl, samples, oracle_subjects=None):
    """H1, then the fixed sequence, each on its own QC sample (``samples[name]``); a hypothesis's cells are read
    only when it is tested."""
    first = pl['order'][0]
    details = {first: test_hypothesis(scores, pl['hypotheses'][first], samples[first], oracle_subjects)}

    def later(name):
        def run():
            details[name] = test_hypothesis(scores, pl['hypotheses'][name], samples[name], oracle_subjects)
            return details[name]['confirmed']
        return run
    results = {name: (details[first]['confirmed'] if name == first else later(name)) for name in pl['order']}
    verdicts = st.fixed_sequence(results, pl['order'])
    verdicts.update({name: 'NOT TESTABLE' for name, d in details.items() if d.get('testable') is False})
    verdicts.update({name: 'DROPPED' for name in pl['dropped']})
    means = [dict(arm=r['arm'], budget=r['budget'], endpoint=r['endpoint'], n=r['subjects'], mean=r['mean'])
             for r in scores.read]
    return dict(h1=details[first]['confirmed'], verdicts=verdicts, hypotheses=details,
                arms_read=sorted({r['arm'] for r in scores.read}), arm_means=means, blocks_read=list(scores.read))


# ----------------------------------------------------------------------------------------- the run
def check_lock(lock, fingerprints):
    """The lock registers the confirmatory run, its fingerprints equal the running code and its plan is this
    script's; returns the plan. A lock that reserves CONFIRM (development null, a G1/G-FEAS stop or the LPP G4
    fallback) is refused."""
    if lock.get('confirmatory') is False:
        raise Refused('the lock registers no confirmatory analysis: CONFIRM is reserved '
                      f"({(lock.get('confirm_status') or {}).get('reasons')})")
    missing = [k for k in ('confirmatory', 'advanced_arm', 'drop_h6', 'primary_budget', 'lpp_partition',
                           'confirmatory_plan', 'lpp_config_sha256') if k not in lock]
    if missing:
        raise Refused(f'the lock lacks {missing}')
    if lock['confirmatory'] is not True:
        raise Refused(f"the lock's confirmatory flag is {lock['confirmatory']!r}")
    if not isinstance(lock['lpp_config_sha256'], str) or not HEX64.match(lock['lpp_config_sha256']):
        raise Refused('the lock pins no LPP configuration hash (lpp_config_sha256)')
    changed = lock_mismatches(lock, fingerprints)
    if changed:
        raise Refused(f'code or configuration changed since the lock: {changed}')
    try:
        pl = plan(lock['advanced_arm'], lock['drop_h6'], (lock['primary_budget'] or {}).get('lpp'))
    except (AttributeError, ValueError) as error:
        raise Refused(f'the lock does not define a confirmatory family: {error}') from error
    if clean(lock['confirmatory_plan']) != clean(pl):
        raise Refused("the locked confirmatory plan differs from this script's plan")
    part = lock['lpp_partition']
    if not isinstance(part, dict) or not part.get('confirm') or not part.get('dev'):
        raise Refused('the lock has no LPP DEV/CONFIRM partition')
    dev, confirm = bud.lpp_partition(list(part['dev']) + list(part['confirm']))
    if list(dev) != list(part['dev']) or list(confirm) != list(part['confirm']):
        raise Refused('the locked lpp_partition is not the registered hash partition')
    return pl


def git_tree(lock, lock_path, repo=ROOT):
    """The locked files against the committed, pushed tree of the running code (``repo``, a git work tree), or
    None without git: dict(commit, lock_tracked, files, documents, ignored, problems).

    ``commit`` is the lock file's last commit when the lock is tracked (``lpp_prepare.lock_git_state``), else HEAD
    (``scripts/v4_lock.py --verify`` before the lock is committed). Problems:
    - a file of ``code_files`` or ``registration.documents`` that is not committed at ``commit`` or whose committed
      bytes differ from the lock (git-ignored files, the pinned reference checkout, are pinned by hash only);
    - a pinned code file whose running bytes differ from the lock (the registration documents are compared at the
      commit only: PROTOCOL_V4 §4 receives deviation entries after the lock);
    - ``commit`` not on origin/main.
    """
    repo = Path(repo)

    def git(*args, text=True):
        return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=text)
    try:
        if git('rev-parse', '--show-toplevel').returncode:
            return None
    except OSError:
        return None
    state = lpp_prepare.lock_git_state(lock_path)
    tracked = bool(state and state['tracked'] and state['commit'])
    commit = state['commit'] if tracked else git('rev-parse', 'HEAD').stdout.strip()
    files = dict(lock.get('code_files') or {})
    docs = dict((lock.get('registration') or {}).get('documents') or {})
    problems = [] if files else ['the lock pins no code_files']
    ignored = []
    for name, want in sorted({**files, **docs}.items()):
        if git('check-ignore', '-q', '--', name).returncode == 0:
            ignored.append(name)
        else:
            shown = git('show', f'{commit}:./{name}', text=False)
            if shown.returncode:
                problems.append(f'{name} is not committed at {commit[:12]}')
            elif sha256_bytes(shown.stdout) != want:
                problems.append(f'{name} at {commit[:12]} differs from the lock')
            if name in docs and files.get(name, want) != want:
                problems.append(f'{name}: code_files and registration.documents pin different bytes')
        if name in files and name not in docs:
            path = repo / name
            if not path.is_file() or sha256_file(path) != want:
                problems.append(f'{name}: the running file differs from the lock')
    if not commit or git('merge-base', '--is-ancestor', commit, 'origin/main').returncode:
        problems.append(f'commit {commit[:12] or "(none)"} is not on origin/main')
    return dict(commit=commit or None, lock_tracked=tracked, files=len(files), documents=len(docs), ignored=ignored,
                problems=problems)


def check_tree(lock, lock_path, tree='auto'):
    """``git_tree`` for the primary analysis (``tree`` 'auto' asks git; tests pass a state or None): refuses any
    problem, and refuses a lock written without its git check (--no-git-check) where git is unavailable."""
    state = git_tree(lock, lock_path) if tree == 'auto' else tree
    if state is None:
        if lock.get('git_check') != 'verified':
            raise Refused('git is unavailable here and the lock was written without its git check '
                          f"({lock.get('git_check')!r}): the locked code cannot be verified against the committed "
                          'tree; verify it with scripts/v4_lock.py --verify and run --confirm where git is available')
        return dict(commit=None, problems=[], note="git unavailable; the lock's own git check was verified")
    if state['problems']:
        raise Refused(f"the locked code is not the committed, pushed tree ({len(state['problems'])} problems): "
                      f"{state['problems'][:6]} (scripts/v4_lock.py --verify)")
    return state


def guard_once(root):
    for path in (PRIMARY_JSON, PRIMARY_MD, STARTED):
        if (Path(root) / path).exists():
            raise Refused(f'the primary analysis has already run or started ({path}); it runs exactly once')


def run_qc(root=ROOT, fingerprints=None, audit=default_audit, data_root=None, write=True):
    """``--qc``: endpoint-blind QC under the lock (writes results/v4/primary_qc.json); returns (lock, plan, qc)."""
    root = Path(root)
    lock_path = root / LOCK
    if not lock_path.is_file():
        raise Refused(f'{LOCK} does not exist')
    lock = json.loads(lock_path.read_text())
    pl = check_lock(lock, current_fingerprints() if fingerprints is None else fingerprints)
    prepared = prepared_trs(root / LPP_DATA if data_root is None else data_root)
    part = lock['lpp_partition']
    record = qc(root, part['confirm'], pl, lock['lpp_cell_code_sha256'], prepared, config=lock['lpp_config_sha256'],
                audit=audit, reference_pool=list(part['dev']) + list(part['confirm']))
    record.update(lock_sha256=sha256_file(lock_path), created_utc=now_utc())
    if write:
        path = root / QC_JSON
        previous = (load_json(path) or {}).get('attempts', [])
        record['attempts'] = previous + [dict(utc=record['created_utc'], included=len(record['included']),
                                              excluded=len(record['excluded']), refusals=len(record['refusals']))]
        write_json(record, path)
    return lock, pl, record


def start_marker(root, record, accept_missing=None, deviation=None):
    """Create results/v4/primary_started.json exclusively: from here on endpoint values are read."""
    path = Path(root) / STARTED
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise Refused(f'{STARTED} exists: the primary analysis already started') from error
    with os.fdopen(fd, 'w') as handle:
        handle.write(json.dumps(clean(dict(started_utc=now_utc(), lock_sha256=record['lock_sha256'],
                                           qc_sha256=sha256_bytes(json.dumps(clean(record['files']),
                                                                             sort_keys=True).encode()),
                                           included=record['included'], accept_missing=accept_missing,
                                           deviation=deviation)), indent=1) + '\n')


def completeness_gate(record, accept_missing=None, deviation=None):
    """Refuse, before the start marker, a QC record whose gaps point to a run that did not happen (``qc``'s
    ``completeness``; module docstring, step 1). Returns the conditions ``deviation`` covers."""
    gaps = record['completeness']
    covered = []
    if record['unregistered_exclusions']:
        covered.append(f"unregistered exclusions {record['unregistered_exclusions']}")
    if gaps['systematic']:
        covered.append(f"systematic omissions {[(g['budget'], g['arm'], g['sections']) for g in gaps['systematic']]}")
    if gaps['unrun']:
        covered.append(f"subjects with none of the family's cells {gaps['unrun']}")
    untestable = [h for h in gaps['untestable'] if h != record['first']]
    if untestable:
        covered.append(f"hypotheses with fewer than 2 subjects {untestable} (n = "
                       f"{ {h: gaps['sample_sizes'][h] for h in untestable} })")
    if covered and not deviation:
        raise Refused(f"{'; '.join(covered)} (see {QC_JSON}): a run that did not happen, or exclusions outside the "
                      'registered §6 list. Compute the missing cells (run_v4_lpp.py confirm) or record a PROTOCOL_V4 §4 '
                      'deviation, then rerun with --deviation naming it')
    if gaps['missing_file_exclusions'] and not accept_missing:
        detail = {k: {s: len(v) for s, v in by_subject.items()} for k, by_subject in gaps['missing'].items()}
        raise Refused(f"'missing file' exclusions of {gaps['missing_file_exclusions']} (missing cells per "
                      f"(budget, arm) and subject: {detail}; see {QC_JSON}). A registered exclusion, but it is applied "
                      'only with --accept-missing REASON (recorded in primary.json), so that a premature launch or an '
                      'unsynced results folder cannot shrink a sample')
    if len(record['included']) < 2:
        raise Refused(f"only {len(record['included'])} subjects pass the endpoint-blind QC of H1 (see {QC_JSON})")
    return covered


def run_primary(root=ROOT, fingerprints=None, audit=default_audit, data_root=None, git_state='auto', deviation=None,
                no_git_check=False, accept_missing=None, tree='auto'):
    """``--confirm``: the registered primary analysis, exactly once (module docstring).

    ``git_state`` 'auto' asks git whether the lock is committed, unmodified and pushed; where git is unavailable
    (None) the run is refused unless ``no_git_check`` (--no-git-check) opts out, which primary.json records.
    ``tree`` 'auto' compares the locked files with the committed, pushed tree (``check_tree``; tests pass a state).
    ``accept_missing`` (--accept-missing) and ``deviation`` (--deviation) are the recorded acknowledgments of
    ``completeness_gate``.
    """
    root = Path(root)
    guard_once(root)
    lock_path = root / LOCK
    if not lock_path.is_file():
        raise Refused(f'{LOCK} does not exist')
    state = lpp_prepare.lock_git_state(lock_path) if git_state == 'auto' else git_state
    if state is None and not no_git_check:
        raise Refused(f'git is unavailable for {LOCK}: it cannot be verified as committed and pushed; run where git '
                      'is available, or pass --no-git-check (recorded in primary.json)')
    if state is not None and not (state['tracked'] and state['clean'] and state['pushed']):
        raise Refused(f'{LOCK} must be committed, unmodified and pushed to origin/main: {state}')
    import statsmodels  # noqa: F401  (the crossed LMM needs it; fail before any endpoint is read)
    lock, pl, record = run_qc(root, fingerprints, audit, data_root)
    tree_state = check_tree(lock, lock_path, tree)
    if record['refusals']:
        raise Refused(f"{len(record['refusals'])} cells from other code, with other references or failing the "
                      f"audit: {record['refusals'][:5]} (see {QC_JSON})")
    covered = completeness_gate(dict(record, first=pl['order'][0]), accept_missing, deviation)
    start_marker(root, record, accept_missing, deviation)
    scores = Scores(root, pl['clock'], pl['cell_stage'], record['files'])
    samples = {name: v['included'] for name, v in record['samples'].items()}
    result = analyze(scores, pl, samples, record['oracle_subjects'])
    out = dict(protocol='ssltang-v4', stage='primary', created_utc=now_utc(), lock_sha256=record['lock_sha256'],
               lock_git=state, git_check='verified' if state is not None else 'skipped (--no-git-check)',
               code_tree=tree_state, fingerprints={k: lock.get(k) for k in FINGERPRINTS},
               plan=pl, n_registered=len(record['subjects']), n_analyzed=len(record['included']),
               subjects=record['included'], excluded=record['excluded'],
               samples={name: dict(n=len(v['included']), included=v['included'], excluded=v['excluded'])
                        for name, v in record['samples'].items()},
               unregistered_exclusions=record['unregistered_exclusions'], deviation=deviation,
               deviation_covers=covered, accept_missing=accept_missing, completeness=record['completeness'],
               reference_pool=record['reference_pool'],
               oracle_subjects=record['oracle_subjects'], qc_attempts=record['attempts'],
               cells_read=sorted(record['files'].items()), **result)
    write_json(out, root / PRIMARY_JSON)
    (root / PRIMARY_MD).write_text(primary_markdown(clean(out)))
    return out


# ---------------------------------------------------------------------------------------------- reports
CONTRAST_HEAD = ['| Contrast | Budget | Endpoint | Mean Δ | BCa 95% CI | d_z | p (one-sided) | Positive [95% CI] | '
                 'Oracle fraction | SESOI tier (90% CI) |', '|---|---|---|---:|---|---:|---:|---|---:|---|']


def contrast_row(c):
    pos, ci = c.get('positive_ci95') or [None, None], c.get('ci95_bca') or [None, None]
    null = f" (null {fmt(c['null'], 3, True)})" if c.get('null') else ''
    return (f"| {c['arm']} − {c['comparator']}{null} | {c['budget']} | {c['endpoint']} | {fmt(c['mean'], 4, True)} | "
            f"[{fmt(ci[0], 4, True)}, {fmt(ci[1], 4, True)}] | {fmt(c.get('dz'), 2)} | {fmt(c['p_one_sided'], 5)} | "
            f"{c['n_positive']}/{c['n']} [{fmt(pos[0], 2)}, {fmt(pos[1], 2)}] | {fmt(c.get('oracle_fraction'), 2)} | "
            f"{c.get('sesoi_tier') or '—'} |")


def primary_markdown(out):
    pl, h = out['plan'], out['hypotheses']
    a, h1 = pl['advanced'], h['H1']
    lines = ['# PROTOCOL_V4 primary analysis (LPP CONFIRM)', '',
             f"Run once on {out['created_utc']} by `scripts/summarize_v4.py` under `experiments/v4_lock.json` "
             f"(sha256 `{out['lock_sha256']}`). Advanced arm: **{a}**. Primary budget: **{pl['budget']}**, "
             f"{pl['clock']} clock, raw endpoint (section-macro feature r, mean over the 9 outer folds).",
             f"Subjects analyzed: **{out['n_analyzed']} of {out['n_registered']}** "
             f"({len(out['excluded'])} excluded by endpoint-blind QC, logged before scoring).", '',
             f"## H1: {h1['spec']['text']}", '',
             f"**H1 {'CONFIRMED' if h1['confirmed'] else 'NOT CONFIRMED'}** (both one-sided sign-flip p < "
             f"{pl['alpha']} and both means > 0). Claim: {h1.get('claim')}.", ''] + CONTRAST_HEAD
    lines += [contrast_row(c) for c in h1['contrasts']]
    lines += ['', '**Crossed LMM** (per-(subject, section) Δ ~ 1 + (1 | subject) + (1 | section), Satterthwaite df):',
              '', '| Contrast | Estimate | SE | df | p (one-sided) | Var subject | Var section | Engine | Supports |',
              '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for m in h1.get('lmm', []):
        if m.get('error'):
            lines.append(f"| {m['arm']} − {m['comparator']} | failed: {m['error']} | | | | | | | no |")
            continue
        lines.append(f"| {m['arm']} − {m['comparator']} | {fmt(m['estimate'], 4, True)} | {fmt(m['se'], 4)} | "
                     f"{fmt(m['df'], 1)} | {fmt(m['p_one_sided'], 5)} | {fmt(m.get('var_subject'), 7)} | "
                     f"{fmt(m.get('var_section'), 7)} | {m.get('engine')} | {'yes' if m['supports'] else 'no'} |")
    lines += ['', '## Fixed sequence', '', '| Hypothesis | Test | n (QC sample) | Verdict | Mean | p (one-sided) |',
              '|---|---|---:|---|---:|---:|']
    for name in ('H1', 'H2', 'H3', 'H4', 'H5', 'H6'):
        verdict = out['verdicts'].get(name)
        if verdict is None:
            continue
        d = h.get(name, {})
        text = (d.get('spec') or pl['hypotheses'].get(name) or {}).get('text', 'dropped (substitution rule or G3)')
        n = (out.get('samples') or {}).get(name, {}).get('n', '')
        if 'contrasts' in d:
            mean = ', '.join(fmt(c['mean'], 4, True) for c in d['contrasts'])
            p = ', '.join(fmt(c['p_one_sided'], 5) for c in d['contrasts'])
        elif d.get('error'):
            mean, p = d['error'], ''
        elif d:
            mean, p = fmt(d.get('mean'), 5, True), fmt(d.get('p_one_sided'), 5)
        else:
            mean = p = ''
        lines.append(f'| {name} | {text} | {n} | {verdict} | {mean} | {p} |')
    tested = [c for name in pl['order'][1:] if name in h for c in h[name].get('contrasts', [])]
    if tested:
        lines += ['', '**Secondary contrasts** (tested hypotheses only):', ''] + CONTRAST_HEAD
        lines += [contrast_row(c) for c in tested]
    if 'mean_by_f' in h.get('H3', {}):
        lines += ['', f"H3 mean ({a}[f] − W1) by f: " + ', '.join(f'f={k}: {fmt(v, 4, True)}'
                                                             for k, v in h['H3']['mean_by_f'].items())
                  + f"; mean slope {fmt(h['H3']['mean'], 4, True)}, BCa 95% CI "
                    f"[{fmt(h['H3']['ci95_bca'][0], 4, True)}, {fmt(h['H3']['ci95_bca'][1], 4, True)}]."]
    lines += ['', '## Arm means (R_s averaged over analyzed subjects)', '', '| Arm | Budget | Endpoint | Mean | n |',
              '|---|---|---|---:|---:|']
    lines += [f"| {m['arm']} | {m['budget']} | {m['endpoint']} | {fmt(m['mean'])} | {m['n']} |"
              for m in out['arm_means']]
    lines += ['', '## Endpoint-blind QC exclusions (H1 sample)', '']
    if out['excluded']:
        lines += ['| Subject | Reasons |', '|---|---|']
        lines += [f"| {s} | {'; '.join(r['category'] + ': ' + r['detail'] for r in rs[:4])}"
                  f"{' …' if len(rs) > 4 else ''} |" for s, rs in out['excluded'].items()]
    else:
        lines.append('None.')
    later = {k: v for k, v in (out.get('samples') or {}).items() if k != 'H1' and v['excluded']}
    if later:
        lines += ['', 'Secondary samples (each hypothesis excludes only for the cells it reads): '
                  + '; '.join(f"{k}: n = {v['n']}, excluded {sorted(v['excluded'])}" for k, v in later.items()) + '.']
    if out['unregistered_exclusions']:
        lines += ['', f"Exclusion categories beyond the registered §6 list: {out['unregistered_exclusions']} "
                      f"(deviation: {out.get('deviation')})."]
    if out.get('accept_missing'):
        lines += ['', f"'missing file' exclusions ({', '.join(out['completeness']['missing_file_exclusions'])}) were "
                      f"acknowledged before scoring: {out['accept_missing']}."]
    if out.get('deviation_covers'):
        lines += ['', f"Deviation {out.get('deviation')!r} covers: {'; '.join(out['deviation_covers'])}."]
    untestable = [k for k, v in out['verdicts'].items() if v == 'NOT TESTABLE']
    if untestable:
        lines += ['', f'NOT TESTABLE (fewer than 2 subjects in the QC sample): {untestable}; the fixed sequence stops '
                      'there.']
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------------------------- descriptive tables
def describe(deltas, subjects):
    """Descriptive contrast (the §6 summary when n >= 2; the p-values are not confirmatory)."""
    d = np.asarray(deltas, dtype=np.float64)
    if len(d) < 2:
        return dict(n=len(d), mean=float(d.mean()) if len(d) else None, subjects=list(subjects))
    return dict(st.contrast_summary(d), subjects=list(subjects))


def eb_record(cell):
    """Final-fit EB diagnostics {delay: eb_posterior info} of a cell (``final_weights.eb``), or None."""
    weights = cell.get('final_weights')
    eb = weights.get('eb') if isinstance(weights, dict) else None
    return eb if isinstance(eb, dict) and eb else None


def eb_row(head, cell, clock):
    ledger = cell.get('ledger') or {}
    pairs = ledger.get('target_pairs_dense' if clock == 'dense' else 'target_pairs_stride4')
    return dict(head, K=len(ledger.get('references') or []), pairs=pairs, eb=eb_record(cell))


def lpp_table(root, subjects, stage, code=None, clocks=CLOCKS):
    """Per (clock, budget, arm) and subject: the 9 outer-fold raw (and residualized) section r of complete cells.

    Returns (table, eb, skipped). ``table[(clock, budget, arm)][subject]`` = dict(raw=[9], residualized=[9] or
    None, r=mean raw, r_res=mean residualized or None, code). Groups with a missing fold, a non-estimable or
    unscored fold or (when ``code`` is given) a cell from other code are skipped with the reason. ``eb`` lists the
    final-fit EB diagnostics of every complete EB-type group's cells (subject, clock, budget, arm, section, K,
    pairs, eb).
    """
    root = Path(root)
    table, eb, skipped = {}, [], []
    for s in subjects:
        for clock in clocks:
            folder = root / 'results' / 'v4_lpp' / s / clock / stage / 'cells'
            groups = {}
            for path in sorted(folder.glob('t*_*.json')) if folder.is_dir() else []:
                m = LPP_NAME.match(path.name)
                if m:
                    groups.setdefault((m['budget'], m['arm']), {})[m['section']] = path
            for (budget, arm), paths in sorted(groups.items()):
                head = dict(subject=s, clock=clock, budget=budget, arm=arm)
                if set(paths) != set(SECTIONS):
                    skipped.append(dict(head, reason=f'folds {sorted(paths)} of 9'))
                    continue
                raw, res, reason, codes, rows = [], [], None, set(), []
                for t in SECTIONS:
                    cell = json.loads(paths[t].read_text())
                    codes.add(cell.get('code_sha256'))
                    if code is not None and cell.get('code_sha256') != code:
                        reason = 'other code'
                        break
                    if cell.get('estimable') is not True:
                        reason = f'section {t} not estimable'
                        break
                    try:
                        raw.append(endpoint_value(cell, 'raw'))
                    except ValueError:
                        reason = f'section {t} has no finite raw score'
                        break
                    try:
                        res.append(endpoint_value(cell, 'residualized'))
                    except ValueError:
                        res.append(None)
                    if eb_record(cell) is not None:
                        rows.append(eb_row(dict(head, section=t), cell, clock))
                if reason:
                    skipped.append(dict(head, reason=reason))
                    continue
                eb += rows
                full = all(v is not None for v in res)
                table.setdefault((clock, budget, arm), {})[s] = dict(
                    raw=raw, residualized=res if full else None, r=float(np.mean(raw)),
                    r_res=float(np.mean(res)) if full else None, code=sorted(c for c in codes if c))
    return table, eb, skipped


def ds_table(root, subjects, stage, fp=None, clocks=CLOCKS):
    """Per (clock, budget, arm) and subject: the validation (develop) or test (evaluate) r and its per-story r.

    With fingerprints ``fp``, cells from other code or configuration (``ds_code_problem``: run_v4 cells under the
    core, Module H cells under the current Module H code) are skipped. Two-reference sensitivity cells are read by
    ``ds_sensitivity`` instead. Returns (table, eb, skipped).
    """
    root, split = Path(root), 'val' if stage == 'develop' else 'test'
    table, eb, skipped = {}, [], []
    for s in subjects:
        for clock in clocks:
            folder = root / 'results' / 'v4' / s / clock / stage / 'cells'
            for path in sorted(folder.glob('*.json')) if folder.is_dir() else []:
                m = DS_NAME.match(path.name)
                if not m or m['refs']:
                    continue
                head = dict(subject=s, clock=clock, budget=m['budget'], arm=m['arm'])
                cell = json.loads(path.read_text())
                if fp is not None and ds_code_problem(cell, fp):
                    skipped.append(dict(head, reason=f'other code ({ds_code_problem(cell, fp)})'))
                    continue
                block = cell.get(split)
                value = block.get(VAL) if isinstance(block, dict) else None
                if cell.get('estimable') is not True or not finite_number(value):
                    skipped.append(dict(head, reason='not estimable or unscored'))
                    continue
                by_run = (block.get('by_run_feature_correlation') or {})
                table.setdefault((clock, m['budget'], m['arm']), {})[s] = dict(
                    r=float(value), by_run={k: float(v) for k, v in by_run.items() if finite_number(v)},
                    code=cell.get('code_sha256'))
                if eb_record(cell) is not None:
                    eb.append(eb_row(head, cell, clock))
    return table, eb, skipped


def ds_sensitivity(root, subjects, stage, fp=None, clocks=CLOCKS):
    """Module H's registered two-reference sensitivity cells (FUTURE_WORK_V4 §5 TH-SEL step 3, §8.5): every
    ``<stage>/sensitivity/refs-<R1>-<R2>/cells/*.json`` and every ``cells/<budget>_<arm>__refs-<R1>-<R2>.json``.

    A cell is read when its name parses, it is from the current code (``ds_code_problem``, with ``fp``), its
    subject, stage, clock, budget and arm are its path's, its ``references`` are the pair its path names and it
    is estimable with a finite score; otherwise it is skipped with the reason. Returns (table, skipped) with
    ``table[(clock, budget, arm, 'R1-R2')][subject]`` = dict(r, by_run, references, code).
    """
    root, split = Path(root), 'val' if stage == 'develop' else 'test'
    table, skipped = {}, []
    for s in subjects:
        for clock in clocks:
            base = root / 'results' / 'v4' / s / clock / stage
            found = []
            sens = base / 'sensitivity'
            for folder in sorted(sens.glob('refs-*')) if sens.is_dir() else []:
                found += [(p, folder.name[len('refs-'):]) for p in sorted((folder / 'cells').glob('*.json'))]
            if (base / 'cells').is_dir():
                found += [(p, None) for p in sorted((base / 'cells').glob('*__refs-*.json'))]
            for path, folder_refs in found:
                m = DS_NAME.match(path.name)
                head = dict(subject=s, clock=clock, file=path.relative_to(root).as_posix())
                if not m:
                    skipped.append(dict(head, reason='not a <budget>_<arm>.json cell'))
                    continue
                pair = m['refs'] or folder_refs
                head.update(budget=m['budget'], arm=m['arm'], refs=pair)
                if m['refs'] and folder_refs and m['refs'] != folder_refs:
                    skipped.append(dict(head, reason=f"named for {m['refs']} in the folder of {folder_refs}"))
                    continue
                cell = json.loads(path.read_text())
                why = ds_code_problem(cell, fp) if fp is not None else None
                if why:
                    skipped.append(dict(head, reason=f'other code ({why})'))
                    continue
                wrong = [k for k, v in (('subject', s), ('stage', stage), ('clock', clock), ('budget', m['budget']),
                                        ('arm', m['arm'])) if cell.get(k) != v]
                refs = [str(r) for r in cell.get('references') or []]
                if wrong or refs != pair.split('-'):
                    skipped.append(dict(head, reason=f'fields {wrong} or references {refs} differ from the path'))
                    continue
                block = cell.get(split)
                value = block.get(VAL) if isinstance(block, dict) else None
                if cell.get('estimable') is not True or not finite_number(value):
                    skipped.append(dict(head, reason='not estimable or unscored'))
                    continue
                by_run = block.get('by_run_feature_correlation') or {}
                table.setdefault((clock, m['budget'], m['arm'], pair), {})[s] = dict(
                    r=float(value), by_run={k: float(v) for k, v in by_run.items() if finite_number(v)},
                    references=refs, code=cell.get('code_sha256'))
    return table, skipped


def lpp_sensitivity(root, subjects, stage, code=None, clocks=CLOCKS, pools=None):
    """Module G's two-reference sensitivity cells (FUTURE_WORK_V4 §5 TH-SEL step 3; run_v4_lpp ``pair_folder``):
    results/v4_lpp/<sub>/<clock>/<stage>/sensitivity/refs-<R1>-<R2>/cells/t<t>_<budget>_<arm>.json, arm TH_SEL or
    XDEC, whose ``references`` are the pair the folder names (LPP ids contain '-': the folder name is compared
    whole) and, with ``pools`` ({subject: reference pool}), the target's registered pair
    (``tanghuth.sensitivity_pairs``). A (subject, clock, budget, arm) group needs all 9 outer sections from one
    current (``code``), estimable, scored pair; otherwise it is skipped with the reason. Returns (table, skipped)
    with ``table[(clock, budget, arm)][subject]`` = dict(raw=[9], r, references)."""
    from src.priors import tanghuth as th
    root = Path(root)
    table, skipped = {}, []
    for s in subjects:
        for clock in clocks:
            sens = root / 'results' / 'v4_lpp' / s / clock / stage / 'sensitivity'
            groups = {}
            for folder in sorted(sens.glob('refs-*')) if sens.is_dir() else []:
                for path in sorted((folder / 'cells').glob('*.json')):
                    m = LPP_NAME.match(path.name)
                    if not m:
                        skipped.append(dict(subject=s, clock=clock, file=path.relative_to(root).as_posix(),
                                            reason='not a t<t>_<budget>_<arm>.json cell'))
                        continue
                    groups.setdefault((m['budget'], m['arm']), {}).setdefault(folder.name, {})[m['section']] = path
            for (budget, arm), by_folder in sorted(groups.items()):
                head = dict(subject=s, clock=clock, budget=budget, arm=arm)
                if arm not in th.REFERENCE_ARMS or len(by_folder) != 1:
                    skipped.append(dict(head, reason=f'arm {arm} with pair folders {sorted(by_folder)}'))
                    continue
                (name, paths), = by_folder.items()
                if set(paths) != set(SECTIONS):
                    skipped.append(dict(head, reason=f'folds {sorted(paths)} of 9'))
                    continue
                want = None
                if pools is not None and s in pools:
                    pool = sorted(str(x) for x in pools[s] if str(x) != s)
                    want = list(th.sensitivity_pairs(pool, 'lpp', s)[0]) if len(pool) >= 2 else []
                raw, refs, reason = [], None, None
                for t in SECTIONS:
                    cell = json.loads(paths[t].read_text())
                    got = [str(r) for r in cell.get('references') or []]
                    if code is not None and cell.get('code_sha256') != code:
                        reason = 'other code'
                    elif [cell.get(k) for k in ('subject', 'clock', 'stage', 'budget', 'arm')] != [
                            s, clock, stage, budget, arm] or str(cell.get('test_section')) != t:
                        reason = f'section {t}: fields differ from the path'
                    elif 'refs-' + '-'.join(got) != name or (want is not None and got != want):
                        reason = f'section {t}: references {got} are not the pair of {name}' + (
                            f' (registered {want})' if want is not None else '')
                    elif cell.get('estimable') is not True:
                        reason = f'section {t} not estimable'
                    if reason:
                        break
                    try:
                        raw.append(endpoint_value(cell, 'raw'))
                    except ValueError:
                        reason = f'section {t} has no finite raw score'
                        break
                    refs = got
                if reason:
                    skipped.append(dict(head, reason=reason))
                    continue
                table.setdefault((clock, budget, arm), {})[s] = dict(raw=raw, r=float(np.mean(raw)), references=refs)
    return table, skipped


def lpp_sensitivity_rows(sensitivity, table):
    """Per (clock, budget, arm): the mean R_s of the targets' registered pairs and, against the whole-pool cell of
    the same arm (``whole_pool``), W0 and W1 of the budget curves, the descriptive contrast over the subjects both
    have."""
    rows = []
    for (clock, budget, arm), by_subject in sorted(sensitivity.items()):
        subjects = sorted(by_subject)
        row = dict(clock=clock, budget=budget, arm=arm, n=len(subjects),
                   mean=float(np.mean([by_subject[s]['r'] for s in subjects])),
                   subjects={s: by_subject[s]['r'] for s in subjects},
                   pairs={s: by_subject[s]['references'] for s in subjects})
        for name, comp in (('whole_pool', arm), ('W0', 'W0'), ('W1', 'W1')):
            other = table.get((clock, budget, comp)) or {}
            common = [s for s in subjects if s in other]
            if common:
                row[f'minus_{name}'] = describe([by_subject[s]['r'] - other[s]['r'] for s in common], common)
        rows.append(row)
    return rows


def lpp_controls(root, subjects, stage, code=None, clocks=CLOCKS):
    """Module G's §9 control records (run_v4_lpp ``subject_controls``): results/v4_lpp/<sub>/<clock>/<stage>/controls/
    <budget>_<arm>.json. A record is read when it is from the current code (``code``), its fields are its path's
    and its inputs are the current cells of its 9 outer sections; otherwise it is skipped with the reason. Returns
    (rows, skipped): per (clock, budget, arm) the subjects' observed R_s, within-section shift null (q95 and the
    number above it), wrong-section input, rotated-training refit and nuisance-only floors."""
    root = Path(root)
    found, skipped = {}, []
    for s in subjects:
        for clock in clocks:
            folder = root / 'results' / 'v4_lpp' / s / clock / stage / 'controls'
            for path in sorted(folder.glob('*.json')) if folder.is_dir() else []:
                budget, _, arm = path.stem.partition('_')
                head = dict(subject=s, clock=clock, budget=budget, arm=arm)
                rec = load_json(path)
                inputs = (rec or {}).get('inputs') or {}
                want = {f'{s}/{clock}/{stage}/cells/t{t}_{budget}_{arm}.json' for t in SECTIONS}
                base = root / 'results' / 'v4_lpp'
                if not isinstance(rec, dict) or (code is not None and rec.get('code_sha256') != code):
                    reason = 'other code'
                elif [rec.get(k) for k in ('kind', 'subject', 'clock', 'stage', 'budget', 'arm')] != [
                        'controls', s, clock, stage, budget, arm]:
                    reason = 'fields differ from the path'
                elif set(inputs) != want or any(not (base / rel).is_file() or sha256_file(base / rel) != digest
                                                for rel, digest in inputs.items()):
                    reason = 'computed from cells that are not the current 9 outer-section cells'
                else:
                    reason = None
                if reason:
                    skipped.append(dict(head, reason=reason))
                    continue
                found.setdefault((clock, budget, arm), {})[s] = rec
    rows = []
    for (clock, budget, arm), by_subject in sorted(found.items()):
        subjects_ = sorted(by_subject)

        def mean(get):
            values = [get(by_subject[s]) for s in subjects_]
            values = [v for v in values if finite_number(v)]
            return dict(mean=float(np.mean(values)) if values else None, n=len(values))
        null = [by_subject[s].get('shift_null') or {} for s in subjects_]
        floors = sorted({k for s in subjects_ for k in (by_subject[s].get('nuisance_floors') or {})})
        rows.append(dict(
            clock=clock, budget=budget, arm=arm, n=len(subjects_), subjects=subjects_,
            observed=mean(lambda r: r.get('observed')),
            shift_null=dict(null_q95=mean(lambda r: (r.get('shift_null') or {}).get('null_q95')),
                            n_above=sum(bool(x.get('passed')) for x in null)),
            wrong_section=mean(lambda r: (r.get('wrong_section') or {}).get('mean')),
            rotated_training=mean(lambda r: (r.get('rotated_training') or {}).get('mean')),
            nuisance_floors={k: mean(lambda r, k=k: ((r.get('nuisance_floors') or {}).get(k) or {}).get('mean'))
                             for k in floors}))
    return rows, skipped


def story_level_flips(by_subject, other, subjects):
    """Per subject: the story-level sign-flip of (arm − comparator) over the stories both cells scored."""
    flips = {}
    for s in subjects:
        runs = sorted(set(by_subject[s]['by_run']) & set(other[s]['by_run']))
        d = [by_subject[s]['by_run'][k] - other[s]['by_run'][k] for k in runs]
        if d:
            flips[s] = dict(n_stories=len(d), mean=float(np.mean(d)), p_one_sided=st.sign_flip_p(d))
    return flips


def sensitivity_rows(sensitivity, table):
    """Per (clock, budget, arm, reference pair): the mean r and, against the three-reference cell of the same arm
    (``three_references``, with per-subject story-level sign-flips), W0 and W1 of the budget curves, the
    descriptive contrast over the subjects both have."""
    rows = []
    for (clock, budget, arm, pair), by_subject in sorted(sensitivity.items()):
        subjects = sorted(by_subject)
        values = np.array([by_subject[s]['r'] for s in subjects])
        row = dict(clock=clock, budget=budget, arm=arm, references=pair.split('-'), n=len(subjects),
                   mean=float(values.mean()), subjects={s: by_subject[s]['r'] for s in subjects})
        for name, comp in (('three_references', arm), ('W0', 'W0'), ('W1', 'W1')):
            other = table.get((clock, budget, comp)) or {}
            common = [s for s in subjects if s in other]
            if not common:
                continue
            row[f'minus_{name}'] = describe([by_subject[s]['r'] - other[s]['r'] for s in common], common)
            if name == 'three_references':
                row['minus_three_references_story_flip'] = story_level_flips(by_subject, other, common)
        rows.append(row)
    return rows


def curves(table, budgets, story_flips=False):
    """Budget curves: per (clock, arm, budget) mean R, contrasts against W0 and W1 (and on the residualized
    endpoint when present), the oracle fraction and, with ``story_flips``, each subject's story-level sign-flip."""
    rows = []
    order = {b: i for i, b in enumerate(budgets)}
    for (clock, budget, arm), by_subject in sorted(table.items(), key=lambda kv: (kv[0][0], kv[0][2],
                                                                                   order.get(kv[0][1], 99))):
        subjects = sorted(by_subject)
        values = np.array([by_subject[s]['r'] for s in subjects])
        row = dict(clock=clock, budget=budget, arm=arm, n=len(subjects), mean=float(values.mean()),
                   sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                   subjects={s: by_subject[s]['r'] for s in subjects})
        res = {s: v['r_res'] for s, v in by_subject.items() if v.get('r_res') is not None}
        if res:
            row['residualized'] = dict(n=len(res), mean=float(np.mean(list(res.values()))), subjects=res)
        for comp in ('W0', 'W1'):
            other = table.get((clock, budget, comp))
            if comp == arm or not other:
                continue
            common = [s for s in subjects if s in other]
            if not common:
                continue
            row[f'minus_{comp}'] = describe([by_subject[s]['r'] - other[s]['r'] for s in common], common)
            common_res = [s for s in common if by_subject[s].get('r_res') is not None
                          and other[s].get('r_res') is not None]
            if common_res:
                row[f'minus_{comp}_residualized'] = describe(
                    [by_subject[s]['r_res'] - other[s]['r_res'] for s in common_res], common_res)
            if story_flips:
                row[f'minus_{comp}_story_flip'] = story_level_flips(by_subject, other, common)
        w0, orc = table.get((clock, budget, 'W0')), table.get((clock, budget, ORACLE))
        if w0 and orc and arm not in ('W0', ORACLE):
            common = [s for s in subjects if s in w0 and s in orc]
            if common:
                row['oracle_fraction'] = st.oracle_fraction([by_subject[s]['r'] for s in common],
                                                            [w0[s]['r'] for s in common], [orc[s]['r'] for s in common])
                row['oracle_n'] = len(common)
        rows.append(row)
    return rows


# ----------------------------------------------------------------------- development effects and R_P(K)
def effect(deltas, subjects):
    d = np.asarray(deltas, dtype=np.float64)
    return dict(n=len(d), mean=float(d.mean()), sd=float(d.std(ddof=1)) if len(d) > 1 else None,
                dz=st.dz(d) if len(d) > 1 else None, deltas=dict(zip(subjects, d.tolist())))


def development_effects(table, clock, budget, advanced, subjects, dataset):
    """Development d_z and SD(Δ) of Δ1 = A − W1 and Δ2 = A − W0 at the primary budget; refuses gaps."""
    values = {}
    for arm in (advanced, 'W1', 'W0'):
        by_subject = table.get((clock, budget, arm), {})
        missing = [s for s in subjects if s not in by_subject]
        if missing:
            raise Refused(f'{dataset} development: {arm} at {budget} ({clock}) is missing or incomplete for {missing}')
        values[arm] = np.array([by_subject[s]['r'] for s in subjects])
    return dict(dataset=dataset, clock=clock, budget=budget, arm=advanced, subjects=list(subjects),
                endpoint='validation r' if dataset == 'ds003020' else 'outer-section r, mean over 9 folds',
                minus_w1=effect(values[advanced] - values['W1'], subjects),
                minus_w0=effect(values[advanced] - values['W0'], subjects))


def spearman_brown(k, rho1):
    """R_P(K) = K r / (1 + (K − 1) r) for single-reference reliability r."""
    k = np.asarray(k, dtype=np.float64)
    return k * rho1 / (1.0 + (k - 1.0) * rho1)


def fit_spearman_brown(ks, values, weights=None):
    """Least-squares single-reference reliability r in [0, 1] of R_P(K) = K r / (1 + (K − 1) r).

    One K level is solved exactly: r = R / (K − (K − 1) R), clipped to [0, 1].
    """
    ks, values = np.asarray(ks, dtype=np.float64), np.asarray(values, dtype=np.float64)
    w = np.ones_like(values) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(ks) == 1:
        k, r = ks[0], float(np.clip(values[0], 0.0, 1.0))
        return 1.0 if r >= 1 else float(np.clip(r / (k - (k - 1.0) * r), 0.0, 1.0))
    from scipy.optimize import minimize_scalar

    def loss(rho):
        return float(np.sum(w * (values - spearman_brown(ks, rho)) ** 2))
    best = minimize_scalar(loss, bounds=(0.0, 1.0), method='bounded', options=dict(xatol=1e-12))
    return float(min((0.0, 1.0, float(best.x)), key=loss))


def reliability_fit(eb_rows, clock, budget, predict_k=RELIABILITY_K):
    """R_P(K) on DEV with a Spearman-Brown fit and its K = 48 prediction (FUTURE_WORK_V4 §8.4).

    R_P = b² Var(p) / (Var z − σ²) per (subject, outer fold, delay) from the final-fit EB diagnostics
    (``src.priors.eb.reliability``) of the EB arm (K = all DEV references) and any K-subset arm ``EB_K<k>...``.
    Averaged over folds and delays per (subject, K), then over subjects per K; the fit weights K levels by
    their number of subjects. Undefined R_P (Var z <= σ²) are dropped and counted.
    """
    from src.priors import eb as ebm
    per, undefined = {}, 0
    for row in eb_rows:
        if row['clock'] != clock or row['budget'] != budget or not (row['arm'] == 'EB'
                                                                     or row['arm'].startswith('EB_K')):
            continue
        for info in row['eb'].values():
            r_p = ebm.reliability(info)['R_P']
            if r_p is None or not math.isfinite(r_p):
                undefined += 1
                continue
            per.setdefault((row['K'], row['subject']), []).append(r_p)
    if not per:
        raise Refused(f'no EB diagnostics at {budget} ({clock}) for the R_P(K) fit')
    by_k = {}
    for (k, s), v in sorted(per.items()):
        by_k.setdefault(k, {})[s] = float(np.mean(v))
    ks = sorted(by_k)
    means = [float(np.mean(list(by_k[k].values()))) for k in ks]
    rho1 = fit_spearman_brown(ks, means, [len(by_k[k]) for k in ks])
    return dict(form='R_P(K) = K r / (1 + (K - 1) r)', clock=clock, budget=budget, rho1=rho1,
                by_k={str(k): dict(n_subjects=len(by_k[k]), mean=m, subjects=by_k[k]) for k, m in zip(ks, means)},
                fitted={str(k): float(spearman_brown(k, rho1)) for k in ks}, n_k_levels=len(ks),
                undefined_values=undefined, predict_k=predict_k, prediction=float(spearman_brown(predict_k, rho1)),
                note=None if len(ks) > 1 else 'one K level: the one-parameter form is solved exactly')


RELIABILITY_KEYS = ('lambda_own', 'R_P', 'R_own', 'R_EB', 'crossover_lambda', 'delta_n')


def kappa_schedule(eb_rows, dataset, arm='EB'):
    """Mean EB hyperparameters (κ, b, τ², σ²) and the §8.3 reliability-model quantities against labeled pairs, per
    (clock, budget), from the final-fit diagnostics of ``arm``."""
    from src.priors import eb as ebm
    groups = {}
    for row in eb_rows:
        if row['arm'] == arm:
            groups.setdefault((row['clock'], row['budget']), []).append(row)
    out = []
    for (clock, budget), rows in sorted(groups.items()):
        infos = [(i, r.get('pairs')) for r in rows for i in r['eb'].values()]
        pairs = [p for _, p in infos if finite_number(p)]
        model = [ebm.reliability(i, p if finite_number(p) else None) for i, p in infos]
        record = dict(dataset=dataset, clock=clock, budget=budget, cells=len(rows),
                      subjects=len({r['subject'] for r in rows}), pairs=float(np.mean(pairs)) if pairs else None)
        for k in ('kappa', 'b', 'tau2', 'sigma2'):
            values = [i[k] for i, _ in infos if finite_number(i.get(k))]
            record[k] = float(np.mean(values)) if values else None
        record['reliability'] = {}
        for k in RELIABILITY_KEYS:
            values = [m[k] for m in model if finite_number(m[k])]
            record['reliability'][k] = dict(mean=float(np.mean(values)) if values else None, n=len(values))
        out.append(record)
    return out


# ------------------------------------------------------------------------------------- descriptive modes
def lpp_summary(root, subjects, stage, code=None, gates=None, partition='DEV', pools=None):
    """Descriptive LPP summary (budget curves, contrasts, oracle fractions, κ schedule, the two-reference
    sensitivity and the §9 control records; DEV: d_z and R_P(K)). ``pools``: {subject: reference pool} for the
    registered-pair check of ``lpp_sensitivity``."""
    table, eb, skipped = lpp_table(root, subjects, stage, code)
    sensitivity, sensitivity_skipped = lpp_sensitivity(root, subjects, stage, code, pools=pools)
    controls, controls_skipped = lpp_controls(root, subjects, stage, code)
    out = dict(protocol='ssltang-v4', stage=f'lpp_{partition.lower()}_descriptive', created_utc=now_utc(),
               descriptive=True, subjects=list(subjects), cell_stage=stage, code_filter=code,
               curves=curves(table, LPP_BUDGETS), skipped=skipped, kappa_schedule=kappa_schedule(eb, 'lpp'),
               lpp_sensitivity=lpp_sensitivity_rows(sensitivity, table), lpp_sensitivity_skipped=sensitivity_skipped,
               controls=controls, controls_skipped=controls_skipped)
    if gates and gates.get('advanced') in ADVANCEABLE:
        budget = (gates.get('G1') or {}).get('lpp_primary')
        out['advanced'], out['primary_budget'] = gates['advanced'], budget
        try:
            out['development'] = development_effects(table, CLOCK, budget, gates['advanced'], subjects, 'lpp')
        except Refused as error:
            out['development'] = dict(error=error.reason)
        try:
            out['reliability'] = reliability_fit(eb, CLOCK, budget)
        except Refused as error:
            out['reliability'] = dict(error=error.reason)
    return out


def ds003020_summary(root, stage, subjects=bud.SUBJECTS, fp=None, gates=None):
    table, eb, skipped = ds_table(root, subjects, stage, fp)
    code_filter = None if fp is None else {k: fp[k] for k in ('code_sha256', 'tanghuth_cell_code_sha256',
                                                               'config_sha256')}
    sensitivity, sensitivity_skipped = ds_sensitivity(root, subjects, stage, fp)
    out = dict(protocol='ssltang-v4', stage=f'ds003020_{stage}_descriptive', created_utc=now_utc(), descriptive=True,
               exposed_test=stage == 'evaluate', subjects=list(subjects), code_filter=code_filter,
               curves=curves(table, DS_BUDGETS, story_flips=True), skipped=skipped,
               sensitivity=sensitivity_rows(sensitivity, table), sensitivity_skipped=sensitivity_skipped,
               kappa_schedule=kappa_schedule(eb, 'ds003020'))
    if gates and gates.get('advanced') in ADVANCEABLE and stage == 'develop':
        try:
            out['development'] = development_effects(table, CLOCK, gates['G1']['primary'], gates['advanced'],
                                                     DS_DEV_SUBJECTS, 'ds003020')
        except Refused as error:
            out['development'] = dict(error=error.reason)
    return out


def curves_markdown(title, out, budgets):
    lines = [f'# {title}', '', f"Descriptive only (generated {out['created_utc']}); p-values are not confirmatory.", '']
    for clock in CLOCKS:
        rows = [r for r in out['curves'] if r['clock'] == clock]
        if not rows:
            continue
        present = [b for b in budgets if any(r['budget'] == b for r in rows)]
        lines += [f'## {clock} clock', '', '| Arm | ' + ' | '.join(f'{b} mean (n) | {b} Δ W0 | {b} Δ W1'
                                                              for b in present) + ' |',
                  '|---|' + '---:|---:|---:|' * len(present)]
        for arm in sorted({r['arm'] for r in rows}):
            cells = []
            for b in present:
                r = next((x for x in rows if x['arm'] == arm and x['budget'] == b), None)
                if r is None:
                    cells += ['', '', '']
                    continue
                cells += [f"{fmt(r['mean'])} ({r['n']})", fmt((r.get('minus_W0') or {}).get('mean'), 4, True),
                          fmt((r.get('minus_W1') or {}).get('mean'), 4, True)]
            lines.append(f'| {arm} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    dev = out.get('development')
    if dev and 'error' not in dev:
        lines += [f"**Development effect** ({dev['dataset']}, {dev['clock']}, {dev['budget']}, {dev['arm']}): "
                  f"Δ W1 mean {fmt(dev['minus_w1']['mean'], 4, True)}, SD {fmt(dev['minus_w1']['sd'])}, "
                  f"d_z {fmt(dev['minus_w1']['dz'], 2)}; Δ W0 mean {fmt(dev['minus_w0']['mean'], 4, True)}, "
                  f"SD {fmt(dev['minus_w0']['sd'])}, d_z {fmt(dev['minus_w0']['dz'], 2)}.", '']
    rel = out.get('reliability')
    if rel and 'error' not in rel:
        lines += [f"**R_P(K)** (Spearman-Brown, {rel['n_k_levels']} K level(s)): r = {fmt(rel['rho1'])}; "
                  f"predicted R_P({rel['predict_k']}) = {fmt(rel['prediction'])}.", '']
    if out.get('kappa_schedule'):
        lines += ['**κ schedule** (final-fit EB, mean over cells and delays):', '',
                  '| Clock | Budget | Pairs | κ | b | τ² | σ² | R_P |', '|---|---|---:|---:|---:|---:|---:|---:|']
        lines += [f"| {k['clock']} | {k['budget']} | {fmt(k['pairs'], 0)} | {fmt(k['kappa'], 3)} | {fmt(k['b'], 3)} | "
                  f"{fmt(k['tau2'], 5)} | {fmt(k['sigma2'], 5)} | {fmt(k['reliability']['R_P']['mean'], 3)} |"
                  for k in out['kappa_schedule']]
        lines.append('')
    if out.get('sensitivity'):
        lines += ['## Two-reference sensitivity (Module H, FUTURE_WORK_V4 §5 TH-SEL step 3, §8.5)', '',
                  '| Clock | Budget | Arm | References | Mean (n) | Δ three references (n) | Story-level p < 0.025 | '
                  'Δ W0 | Δ W1 |', '|---|---|---|---|---:|---:|---:|---:|---:|']
        for r in out['sensitivity']:
            three = r.get('minus_three_references') or {}
            flips = r.get('minus_three_references_story_flip') or {}
            lines.append(f"| {r['clock']} | {r['budget']} | {r['arm']} | {'-'.join(r['references'])} | "
                         f"{fmt(r['mean'])} ({r['n']}) | {fmt(three.get('mean'), 4, True)} ({three.get('n', 0)}) | "
                         f"{sum(1 for v in flips.values() if (v.get('p_one_sided') or 1) < st.ALPHA)}/{len(flips)} | "
                         f"{fmt((r.get('minus_W0') or {}).get('mean'), 4, True)} | "
                         f"{fmt((r.get('minus_W1') or {}).get('mean'), 4, True)} |")
        lines.append('')
    if out.get('lpp_sensitivity'):
        lines += ["## Two-reference sensitivity (each target's registered pair; FUTURE_WORK_V4 §5 TH-SEL step 3)", '',
                  '| Clock | Budget | Arm | Mean (n) | Δ whole pool (n) | Δ W0 | Δ W1 |',
                  '|---|---|---|---:|---:|---:|---:|']
        for r in out['lpp_sensitivity']:
            pool = r.get('minus_whole_pool') or {}
            lines.append(f"| {r['clock']} | {r['budget']} | {r['arm']} | {fmt(r['mean'])} ({r['n']}) | "
                         f"{fmt(pool.get('mean'), 4, True)} ({pool.get('n', 0)}) | "
                         f"{fmt((r.get('minus_W0') or {}).get('mean'), 4, True)} | "
                         f"{fmt((r.get('minus_W1') or {}).get('mean'), 4, True)} |")
        lines.append('')
    if out.get('controls'):
        lines += ['## §9 controls (primary cell; subject-level records)', '',
                  '| Clock | Budget | Arm | n | R_s | Shift-null q95 | Above q95 | Wrong section | Rotated training | '
                  'Nuisance floor (both) |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in out['controls']:
            both = (r.get('nuisance_floors') or {}).get('both') or {}
            lines.append(f"| {r['clock']} | {r['budget']} | {r['arm']} | {r['n']} | {fmt(r['observed']['mean'])} | "
                         f"{fmt(r['shift_null']['null_q95']['mean'])} | {r['shift_null']['n_above']}/{r['n']} | "
                         f"{fmt(r['wrong_section']['mean'])} | {fmt(r['rotated_training']['mean'])} | "
                         f"{fmt(both.get('mean'))} |")
        lines.append('')
    if out.get('skipped') or out.get('sensitivity_skipped') or out.get('lpp_sensitivity_skipped') or \
            out.get('controls_skipped'):
        lines += [f"Skipped {len(out.get('skipped') or [])} incomplete, stale or non-estimable cell groups, "
                  f"{len(out.get('sensitivity_skipped') or []) + len(out.get('lpp_sensitivity_skipped') or [])} "
                  f"sensitivity cells or groups and {len(out.get('controls_skipped') or [])} control records "
                  '(see the JSON).', '']
    return '\n'.join(lines)


def lpp_partition_of(root):
    """The LPP {dev, confirm} partition from the lock (a reserved lock records it under ``lpp_reserved``), else
    from data/lpp/prepare_provenance.json."""
    lock = load_json(Path(root) / LOCK) or {}
    partition = (lock.get('lpp_partition') or (lock.get('lpp_reserved') or {}).get('partition')
                 or (load_json(Path(root) / LPP_DATA / 'prepare_provenance.json') or {}).get('partition'))
    if not partition:
        raise Refused('no LPP partition (experiments/v4_lock.json or data/lpp/prepare_provenance.json)')
    return partition


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--confirm', action='store_true', help='the registered primary analysis (exactly once)')
    mode.add_argument('--qc', action='store_true', help='endpoint-blind QC of the CONFIRM cells only')
    mode.add_argument('--dev', action='store_true', help='descriptive LPP DEV summary')
    mode.add_argument('--ds003020', action='store_true', help='descriptive ds003020 budget curves')
    mode.add_argument('--confirm-curves', action='store_true', help='descriptive LPP CONFIRM curves (after --confirm)')
    p.add_argument('--stage', choices=['develop', 'evaluate'], default='develop', help='--ds003020 cell stage')
    p.add_argument('--any-code', action='store_true', help='descriptive modes: do not filter cells by fingerprint')
    p.add_argument('--deviation', help='--confirm: the PROTOCOL_V4 §4 entry covering unregistered QC exclusions, '
                                       'systematic omissions, never-run subjects or untestable secondaries')
    p.add_argument('--accept-missing', metavar='REASON',
                   help="--confirm: acknowledge the 'missing file' exclusions the QC found (recorded in primary.json)")
    p.add_argument('--no-git-check', action='store_true',
                   help='--confirm where git is unavailable: proceed without verifying that the lock is committed '
                        'and pushed (recorded in primary.json)')
    p.add_argument('--root', default=str(ROOT), help='project root holding results/, data/ and experiments/ (code '
                                                     'fingerprints always come from the running code)')
    a = p.parse_args(argv)
    root = Path(a.root)
    if a.confirm:
        out = run_primary(root, deviation=a.deviation, no_git_check=a.no_git_check, accept_missing=a.accept_missing)
        print(json.dumps(dict(h1=out['h1'], verdicts=out['verdicts'], n=out['n_analyzed']), indent=1))
        return
    if a.qc:
        guard_once(root)
        _, _, record = run_qc(root)
        print(json.dumps(clean({k: record[k] for k in ('included', 'excluded', 'refusals', 'unregistered_exclusions',
                                                       'completeness', 'reference_pool')}
                               | {'samples': {h: len(v['included']) for h, v in record['samples'].items()}}),
                         indent=1))
        return
    gates = load_json(root / OUT / 'gates.json')
    fp = None if a.any_code else current_fingerprints()
    out_dir = root / OUT
    if a.ds003020:
        if a.stage == 'evaluate' and not (root / LOCK).is_file():
            raise Refused('ds003020 test scores are summarized only after the v4 lock')
        out = ds003020_summary(root, a.stage, fp=fp, gates=gates)
        write_json(out, out_dir / f'ds003020_{a.stage}_summary.json')
        (out_dir / f'DS003020_{a.stage.upper()}.md').write_text(
            curves_markdown(f'ds003020 budget curves ({a.stage})', clean(out), DS_BUDGETS))
        return
    partition = lpp_partition_of(root)
    code = None if fp is None else fp['lpp_cell_code_sha256']

    def pools(members):  # Each target's registered reference pool (run_v4_lpp.reference_pool), when recorded.
        try:
            pool = registered_pool(list(members), prepared_trs(root / LPP_DATA))
        except Refused:
            return None
        return {s: pool for s in members}
    if a.dev:
        out = lpp_summary(root, partition['dev'], STAGE_DIRS['DEV'], code, gates, 'DEV', pools(partition['dev']))
        write_json(out, out_dir / 'lpp_dev_summary.json')
        (out_dir / 'LPP_DEV.md').write_text(curves_markdown('LPP DEV budget curves', clean(out), LPP_BUDGETS))
        return
    if not (root / PRIMARY_JSON).is_file():
        raise Refused('the CONFIRM curves are read only after the primary analysis (results/v4/primary.json)')
    out = lpp_summary(root, partition['confirm'], STAGE_DIRS['CONFIRM'], code, None, 'CONFIRM',
                      pools(list(partition['dev']) + list(partition['confirm'])))
    write_json(out, out_dir / 'lpp_confirm_summary.json')
    (out_dir / 'LPP_CONFIRM.md').write_text(curves_markdown('LPP CONFIRM budget curves', clean(out), LPP_BUDGETS))


if __name__ == '__main__':
    main()
