#!/usr/bin/env python3
"""Write experiments/v4_lock.json (PROTOCOL_V4 §3; FUTURE_WORK_V4 §7; V4_IMPLEMENTATION Module I).

The lock is written once. It takes one of two forms, decided mechanically by the gates:
  confirmatory  G1 set a primary budget, G-FEAS passed, an arm (EB or PRIOR) advanced and LPP G4 passed: the
                lock registers the CONFIRM run and the primary analysis (``confirmatory_plan``) and carries the
                keys that authorize a CONFIRM download and run (lpp_partition, lpp_mask_sha256, lpp_code_sha256).
  reserved      a registered stop (G1 failed at S3 and S6, G-FEAS failed), the development null (nothing passed
                G2) or the LPP G4 fallback: CONFIRM stays untouched. The lock still pins every fingerprint and
                development cell so that the ds003020 test can be scored descriptively under v4, but it omits the
                CONFIRM-authorizing keys (lpp_prepare.py --confirm, run_v4_lpp.py confirm and summarize_v4.py
                --confirm refuse it) and records the partition under ``lpp_reserved``. A reserved lock is written
                only with --reserve-confirm (it forfeits CONFIRM), and --reserve-confirm is refused when the gates
                register CONFIRM. A missing results/v4_lpp/g4.json after an arm advanced is a refusal, not a
                fallback.
Refuses (exit 1, nothing written) unless:
  * results/v4/g0.json passed under the current run_v4 core fingerprint (``run_v4.code_sha256()``);
  * results/v4/gates.json was written under the same fingerprint; its G1 primary budget is the one its own G1
    verdicts register (S3 if S3 passed, else S6, else none) and every gate number on the decision path is
    recomputed from the g1.json files and development cells on disk and equals gates.json; the advancement is
    the registered table's reading of G2/G3;
  * every ds003020 development cell (results/v4/<S>/<clock>/develop/cells/*.json, UTS01-UTS10, and Module H's
    sensitivity/refs-<R1>-<R2>/cells/) is from the current code (``summarize_v4.ds_code_problem``: run_v4 cells
    under the core, Module H cells under ``tanghuth_cell_code_sha256`` with the current Module H file hashes)
    and configuration, matches its path (and reference set) and passes its writer's provenance audit
    (``summarize_v4.default_audit``: audit_v4, or tanghuth.audit_file for Module H arms; an unscored cell may fail only
    on the scope errors its writer recorded);
  * every LPP DEV cell (results/v4_lpp/<DEV>/<clock>/develop/cells, Module G's two-reference sensitivity cells
    <DEV>/<clock>/develop/sensitivity/refs-<R1>-<R2>/cells and the G4(b) cells <DEV>/dense/g4/cells) passes
    ``summarize_v4.inspect_cell`` under ``lpp_cell_code_sha256`` and one LPP configuration hash, which must be
    ``config_sha256`` (the configuration run_v4_lpp.py confirm runs under): path fields, provenance test section,
    the registered references (the other DEV listeners that pass the subject-level QC, ``summarize_v4.
    registered_pool``; a pair cell: the target's registered pair, ``tanghuth.sensitivity_pairs``) and the audit;
    not-estimable cells are hashed; every §9 control record (<DEV>/<clock>/develop/controls/<budget>_<arm>.json) is
    from the current LPP code and was computed from the current cells of its 9 outer sections;
  * confirmatory: the LPP DEV cells exercise the locked plan, i.e. the primary analysis's endpoint-blind QC
    (``summarize_v4.qc``) run on the DEV cells (stage develop) excludes nobody from any hypothesis and finds
    complete ORACLE/W0 cells at every oracle budget; LPP G4 passed under the current LPP code, and its verdict
    is recomputed from the per-listener records (``g4_records``: every DEV listener's g4b.json under the current
    LPP code with passed = observed > null q95, at least 6 passing; then every g4c.json over the 9 outer
    sections with every registered ISP fit and passed = no alpha at the grid edge) and equals g4.json's dev list
    (the registered DEV partition), b, c and verdict (a G4 failure is recomputed the same way); every DEV listener
    has its G4(b) W0 L8 cells for all 9 sections; the LPP preparation record data/lpp/prepare_provenance.json is
    a complete DEV record written by the current preparation code (``lpp_prepare.code_fingerprint()``), its
    partition is the registered hash partition of the 49 EN listeners and data/lpp/mask.npz (when present) has
    the recorded content digest; the R_P(K) fit on DEV covers the registered K levels 1, 2, 4 (the EB_K<k>_d<j>
    subset arms) and 8 (EB) for every DEV listener (FUTURE_WORK_V4 §8.4), unless --deviation names the
    PROTOCOL_V4 §4 entry that covers the gap (recorded in the lock);
  * nothing confirmatory exists yet: no lock, no ds003020 evaluate output, no LPP evaluate output, CONFIRM
    result, raw data or preparation record, no primary analysis output;
  * every fingerprinted file (``code_files()``: the v4 files, run_v4.CORE, the v3/§12 files of
    run_v3.LOCKED_CODE and run_v3_weights.CODE, lpp_prepare.CODE_FILES, the configuration and the registration
    documents) is committed and unmodified in git (git-ignored files are recorded by hash only). Where git is
    unavailable the lock is refused unless --no-git-check, which the lock records; such a lock must then pass
    ``--verify`` where git is available before it is committed (``verify_lock``: the fingerprints and every
    pinned file against the running code and against the committed tree at HEAD, or at the lock's commit once it
    is tracked, which must be on origin/main), and summarize_v4.py --confirm repeats that comparison
    (``summarize_v4.git_tree``) and refuses such a lock where git is unavailable;
  * the lock is accepted by every reader that runs under it and, when reserved, refused by every CONFIRM reader:
    lpp_prepare.check_lock, summarize_v4.check_lock and (with fingerprints computed from the code)
    run_v4.py --stage evaluate, run_v4_tanghuth.check_lock and run_v4_lpp.check_lock.
It records the gate decision (advanced_arm, primary_budget, drop_h6, confirm_status), the confirmatory plan, the
fingerprints of summarize_v4.FINGERPRINTS, the LPP configuration hash, the sha256 of every fingerprinted file,
every development cell and every gate file (g0, gates, simulate, g1 per subject, LPP g4.json, g4b.json and
g4c.json per DEV listener), lpp_partition, lpp_mask_sha256, lpp_code_sha256 and lpp_dataset_doi (confirmatory),
the development d_z and SD(Δ) of A − W1 and A − W0 at the primary budget (ds003020 dev, LPP DEV; descriptive
for every candidate arm when reserved), the R_P(K) Spearman-Brown fit on DEV with its K = 48 prediction, the κ
schedule and reliability-model quantities, and the registered hash orders, seeds and grids (PROTOCOL_V4 §2)
with their realized values. Commit and push the lock to main before any ds003020 test score and before any
CONFIRM download; ``--dry-run`` prints the lock summary without writing it.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import lpp_prepare  # noqa: E402
import summarize_v4 as sv  # noqa: E402
from src import budgets as bud  # noqa: E402
from src import v4_stats as st  # noqa: E402
from src.priors import eb as ebm  # noqa: E402
from src.priors import isp as ispm  # noqa: E402

G0 = sv.OUT / 'g0.json'
GATES = sv.OUT / 'gates.json'
SIMULATE = sv.OUT / 'simulate.json'
LPP_OUT = Path('results/v4_lpp')
G4 = LPP_OUT / 'g4.json'
PREPARE = sv.LPP_DATA / 'prepare_provenance.json'
MASK = sv.LPP_DATA / 'mask.npz'
DOCS = ('experiments/PROTOCOL_V4.md', 'FUTURE_WORK_V4.md', 'experiments/V4_IMPLEMENTATION.md')
CODE_FILES = ('scripts/run_v4_lpp.py', 'scripts/run_v4_tanghuth.py', 'scripts/summarize_v4.py', 'scripts/v4_lock.py',
              'scripts/v4_gates.py', 'scripts/v4_g0.py', 'scripts/v4_simulate.py', 'src/priors/tanghuth.py',
              'configs/v3/structured_ridge.yaml') + DOCS
TOL = 1e-12
LPP_PRIMARY = {'S3': 'L3', 'S6': 'L5'}
G4_CELLS = Path('dense') / 'g4' / 'cells'  # Module G: G4(b) W0 L8 cells, results/v4_lpp/<DEV>/dense/g4/cells.
G4_MIN_PASS = 6  # run_v4_lpp.G4_MIN_PASS: G4(b) passes in at least 6 of the 9 DEV listeners.
RP_K_LEVELS = (1, 2, 4, 8)  # R_P(K) on DEV: run_v4_lpp.SUBSET_K['DEV'] (EB_K<k>_d<j>) and K = 8 (EB itself).
CELL_KEYS = ('ds003020', 'ds003020_sensitivity', 'lpp_dev', 'lpp_dev_sensitivity', 'lpp_dev_controls', 'lpp_g4')


class Refused(SystemExit):
    """The lock is refused (exit status 1 with the reason); nothing is written."""

    def __init__(self, reason):
        super().__init__(f'v4_lock refused: {reason}')
        self.reason = reason


def need(ok, reason):
    if not ok:
        raise Refused(reason)


def load(root, rel, what):
    path = Path(root) / rel
    need(path.is_file(), f'{rel} does not exist ({what})')
    try:
        return json.loads(path.read_text())
    except ValueError as error:
        raise Refused(f'{rel} is not valid JSON: {error}') from error


def attempt(fn):
    """fn(), or dict(error=reason) when it refuses (descriptive entries of a reserved lock)."""
    try:
        return fn()
    except (sv.Refused, Refused) as error:
        return dict(error=error.reason)


# ------------------------------------------------------------------------------------------------ gates
def check_g0(g0, core):
    need(g0.get('passed') is True, 'results/v4/g0.json did not pass')
    need(g0.get('code_sha256') == core, 'results/v4/g0.json is not from the current run_v4 core')


def expected_advancement(gates):
    """(advanced, drop_h6) of the registered advancement table from gates.json's G2 and G3 verdicts."""
    g2, g3 = gates.get('G2') or {}, gates.get('G3') or {}
    if (g2.get('EB') or {}).get('passed') is True:
        return 'EB', g3.get('passed') is not True
    if (g2.get('PRIOR') or {}).get('passed') is True:
        return 'PRIOR', True
    return None, None


def gate_decision(gates, core):
    """dict(advanced, drop_h6, ds003020, lpp, stop, reasons): the registered gate path gates.json records.

    ``stop`` is None (an arm advanced), 'G1' (no primary budget), 'G_FEAS' or 'G2' (the development null); the
    G1 primary budget must be the one G1's own verdicts register.
    """
    need(gates.get('code_sha256') == core, 'results/v4/gates.json is not from the current run_v4 core')
    g1 = gates.get('G1') or {}
    flags = {n: (g1.get(n) or {}).get('passed') for n in ('S3', 'S6')}
    need(all(isinstance(v, bool) for v in flags.values()), 'G1 has not been computed at S3 and S6')
    primary = 'S3' if flags['S3'] else 'S6' if flags['S6'] else None
    need(g1.get('primary') == primary,
         f"G1 primary {g1.get('primary')!r} contradicts the G1 verdicts, which register {primary!r}")
    need(g1.get('lpp_primary') == LPP_PRIMARY.get(primary),
         f"G1 lpp_primary {g1.get('lpp_primary')!r} is not {LPP_PRIMARY.get(primary)!r} for {primary!r}")
    out = dict(advanced=None, drop_h6=None, ds003020=primary, lpp=LPP_PRIMARY.get(primary), stop=None, reasons=[])
    if primary is None:
        return dict(out, stop='G1', reasons=['G1 failed at S3 and S6: ISP is uninformative at matched scan time '
                                             '(FUTURE_WORK_V4 §7); CONFIRM is not run'])
    feas = gates.get('G_FEAS') or {}
    need(feas.get('budget') == primary and isinstance(feas.get('passed'), bool),
         'G-FEAS has not been computed at the primary budget')
    if not feas['passed']:
        return dict(out, stop='G_FEAS', reasons=['G-FEAS failed: even a perfect prior cannot help at this scan time '
                                                 '(FUTURE_WORK_V4 §7); CONFIRM is not run'])
    need(all(k in gates for k in ('G2', 'G3', 'advanced', 'drop_h6')),
         'G2/G3 have not been computed (gates.json records no advancement)')
    want = expected_advancement(gates)
    need((gates['advanced'], gates['drop_h6']) == want,
         f"gates.json advanced {gates['advanced']} (drop_h6 {gates['drop_h6']}) but G2/G3 give {want}")
    if want[0] is None:
        return dict(out, stop='G2', reasons=['development null: neither EB nor PRIOR passed G2; CONFIRM stays '
                                             'untouched and is reserved (advancement table)'])
    return dict(out, advanced=want[0], drop_h6=want[1])


def close(a, b):
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(float(a) - float(b)) <= TOL


def dev_value(root, subject, budget, arm, core):
    """Validation r of a dense ds003020 development cell (as scripts/v4_gates.py reads it)."""
    path = sv.ds_cell(root, subject, 'dense', 'develop', budget, arm)
    need(path.is_file(), f'{path.relative_to(root)} is missing: the gates cannot be verified')
    cell = json.loads(path.read_text())
    need(cell.get('code_sha256') == core, f'{path.relative_to(root)} is not from the current code')
    value = (cell.get('val') or {}).get(sv.VAL) if cell.get('estimable') is True else None
    need(sv.finite_number(value), f'{path.relative_to(root)} is not estimable')
    return float(value)


def verify_gates(root, gates, core, stop=None, subjects=sv.DS_DEV_SUBJECTS):
    """Recompute the gates on the decision path (G1; G-FEAS unless G1 stopped; G2 and G3 unless G1 or G-FEAS
    stopped) from the files on disk; every number must equal gates.json."""
    import numpy as np
    problems, verified = [], ['G1']
    for name in ('S3', 'S6'):
        rho = []
        for s in subjects:
            g1 = load(root, sv.OUT / s / 'g1.json', 'G1 input')
            need(g1.get('code_sha256') == core, f'{s} g1.json is not from the current code')
            rho.append(g1['budgets'][name]['spearman'])
        rho = np.array(rho, dtype=np.float64)
        got = (gates['G1'].get(name) or {})
        if not (close(got.get('median'), float(np.median(rho))) and got.get('above_010') == int((rho > 0.10).sum())
                and got.get('passed') == bool(np.median(rho) >= 0.20 and (rho > 0.10).sum() >= 7)):
            problems.append(f'G1 {name}')
    if stop != 'G1':
        primary = gates['G1']['primary']

        def deltas(budget, a, b):
            return np.array([dev_value(root, s, budget, a, core) - dev_value(root, s, budget, b, core)
                             for s in subjects])
        d = deltas(primary, 'ORACLE', 'W0')
        feas = gates['G_FEAS']
        if not (close(feas.get('mean'), float(d.mean())) and feas.get('passed') == bool(d.mean() >= 0.005)):
            problems.append('G_FEAS')
        verified.append('G_FEAS')
        if stop != 'G_FEAS':
            for arm in sv.ADVANCEABLE:
                d1, d2 = deltas(primary, arm, 'W1'), deltas(primary, arm, 'W0')
                got = (gates.get('G2') or {}).get(arm) or {}
                if not (close(got.get('mean_minus_w1'), float(d1.mean()))
                        and close(got.get('mean_minus_w0'), float(d2.mean()))
                        and got.get('positive_vs_w0') == int((d2 > 0).sum())
                        and got.get('passed') == bool(d1.mean() >= 0.005 and d2.mean() >= 0.003
                                                      and (d2 > 0).sum() >= 7)):
                    problems.append(f'G2 {arm}')
            d3 = deltas('ALL', 'EB', 'W1')
            g3 = gates.get('G3') or {}
            if not (close(g3.get('mean'), float(d3.mean())) and close(g3.get('worst'), float(d3.min()))
                    and g3.get('passed') == bool(d3.mean() >= -0.003 and d3.min() >= -0.010)):
                problems.append('G3')
            verified += ['G2', 'G3']
    need(not problems, f'gates.json does not match the development files on disk: {problems}')
    return dict(verified=verified, subjects=list(subjects))


def g4_fits_expected():
    """{(test section, budget, inner fold or None)} of the ISP fits G4(c) must record: every outer section, every
    LPP budget, every inner fold and the final fit (run_v4_lpp.g4_subject_c)."""
    return {(t, name, i) for t in sv.SECTIONS for name, budget in bud.lpp_section_budgets(t, sv.SECTIONS).items()
            for i in list(range(len(bud.inner_folds(name, budget)))) + [None]}


def g4_records(root, code, dev):
    """The LPP G4 verdict recomputed from the per-listener records (run_v4_lpp.g4_verdict's rule).

    (b) per DEV listener, ``<DEV>/dense/g4/g4b.json`` under ``code`` whose ``passed`` is observed > null q95 (both
    finite); b = at least G4_MIN_PASS listeners pass. (c), only when (b) passed, per DEV listener ``<DEV>/g4c.json``
    under ``code`` over the 9 outer sections, recording every ISP fit (``g4_fits_expected``), with ``passed``,
    ``n_at_edge`` and ``n_fits`` consistent with the fits; c = every listener passes. Refuses a missing,
    stale or internally inconsistent record.
    """
    b_subjects, c_subjects = {}, {}
    for s in dev:
        rel = LPP_OUT / s / 'dense' / 'g4' / 'g4b.json'
        rec = load(root, rel, 'LPP G4(b) record')
        need(rec.get('code_sha256') == code, f'{rel} is not from the current LPP code')
        obs, q95 = rec.get('observed'), rec.get('null_q95')
        flag = bool(sv.finite_number(obs) and sv.finite_number(q95) and obs > q95)
        need(rec.get('passed') is flag, f"{rel} records passed={rec.get('passed')!r}, but observed {obs} against the "
                                        f'null q95 {q95} gives {flag}')
        b_subjects[s] = flag
    n = sum(b_subjects.values())
    b = n >= G4_MIN_PASS
    if b:
        expected = g4_fits_expected()
        for s in dev:
            rel = LPP_OUT / s / 'g4c.json'
            rec = load(root, rel, 'LPP G4(c) record')
            need(rec.get('code_sha256') == code, f'{rel} is not from the current LPP code')
            need(rec.get('sections') == list(sv.SECTIONS), f"{rel} covers sections {rec.get('sections')}, not all 9")
            fits = rec.get('fits')
            need(isinstance(fits, list) and all(isinstance(f, dict) for f in fits), f'{rel} records no ISP fits')
            got = [(str(f.get('test_section')), f.get('budget'), f.get('fold')) for f in fits]
            need(len(got) == len(set(got)) and set(got) == expected,
                 f'{rel} does not record exactly the {len(expected)} registered ISP fits ({len(got)} recorded)')
            edges = sum(bool(f.get('at_edge')) for f in fits)
            flag = edges == 0
            need(rec.get('passed') is flag and rec.get('n_at_edge') == edges and rec.get('n_fits') == len(fits),
                 f"{rel} records passed={rec.get('passed')!r}, n_at_edge={rec.get('n_at_edge')!r}, "
                 f"n_fits={rec.get('n_fits')!r}, but its fits give {flag}, {edges}, {len(fits)}")
            c_subjects[s] = flag
    c = all(c_subjects.values()) if b else None
    return dict(dev=list(dev), b=b, n_passed=n, c=c, passed=bool(b and c), b_subjects=b_subjects,
                c_subjects=c_subjects if b else None)


def lpp_g4(root, fp, dev=None):
    """LPP G4 (results/v4_lpp/g4.json): dict(status='missing'|'passed'|'failed', ...); a g4.json from other code
    (or with no fingerprint) is refused. With the registered DEV partition ``dev``, the verdict is recomputed from
    the per-listener records (``g4_records``) and g4.json's dev list, b, n_passed, c and passed must equal it
    (``verified``); without it (no preparation record, which a confirmatory lock refuses) it is unverified."""
    path = Path(root) / G4
    if not path.is_file():
        return dict(status='missing')
    g4 = load(root, G4, 'LPP G4 feasibility, Module G')
    need(g4.get('code_sha256') == fp['lpp_cell_code_sha256'],
         f"{G4} is not from the current LPP code (code_sha256 {g4.get('code_sha256')})")
    need(isinstance(g4.get('passed'), bool), f'{G4} records no verdict')
    recomputed = None
    if dev is not None:
        recomputed = g4_records(root, fp['lpp_cell_code_sha256'], dev)
        b, c = g4.get('b') if isinstance(g4.get('b'), dict) else {}, g4.get('c')
        claimed = dict(dev=g4.get('dev'), b=b.get('passed'), n_passed=b.get('n_passed'),
                       c=c.get('passed') if isinstance(c, dict) else c, passed=g4['passed'])
        differ = sorted(k for k in claimed if claimed[k] != recomputed[k])
        need(not differ, f'{G4} does not match the per-listener G4 records on disk: {differ} '
                         f"(recorded {[claimed[k] for k in differ]}, recomputed {[recomputed[k] for k in differ]})")
    return dict(status='passed' if g4['passed'] else 'failed', passed=g4['passed'], sha256=sv.sha256_file(path),
                b=g4.get('b'), c=g4.get('c'), note=g4.get('note'), verified=recomputed is not None,
                recomputed=recomputed, adaptation=g4.get('adaptation', 'none recorded (no W0-only revision)'))


# -------------------------------------------------------------------------------------------- the cells
def ds_cell_problem(path, expect, fp, audit, refs=None):
    """None, or why the ds003020 development cell at ``path`` cannot be locked."""
    try:
        cell = json.loads(path.read_text())
    except ValueError as error:
        return f'unreadable ({error})'
    if not isinstance(cell, dict):
        return 'not a JSON object'
    why = sv.ds_code_problem(cell, fp)
    if why:
        return why
    wrong = [k for k in ('subject', 'stage', 'clock', 'budget', 'arm') if cell.get(k) != expect[k]]
    if wrong:
        return f'fields {wrong} differ from the path'
    if refs is not None and [str(r) for r in cell.get('references') or []] != list(refs):
        return f"references {cell.get('references')} are not the path's {list(refs)}"
    if audit is not None:
        why = sv.audit_problem(path, cell, audit)
        if why:
            return f'provenance audit failed: {why}'
    return None


def ds_folders(root, subject, clock):
    """(cells folder, reference set or None, lock key) of one ds003020 subject and clock at stage develop."""
    base = Path(root) / 'results' / 'v4' / subject / clock / 'develop'
    out = [(base / 'cells', None, 'ds003020')]
    sens = base / 'sensitivity'
    for folder in sorted(sens.glob('refs-*')) if sens.is_dir() else []:
        out.append((folder / 'cells', folder.name[len('refs-'):].split('-'), 'ds003020_sensitivity'))
    return out


def lpp_subjects_on_disk(root):
    folder = Path(root) / LPP_OUT
    return sorted(p.name for p in folder.glob('sub-*') if p.is_dir()) if folder.is_dir() else []


def lpp_pair_problem(cell, folder, subject, reference_pool):
    """None, or why a Module G two-reference sensitivity cell (run_v4_lpp ``pair_folder``: <stage>/sensitivity/
    refs-<R1>-<R2>/cells/, arm TH_SEL or XDEC) is not the registered pair cell: its ``references`` (and ledger
    references) must be the pair its folder names (LPP ids contain '-', so the folder name is compared whole) and,
    with ``reference_pool``, the target's registered pair (``tanghuth.sensitivity_pairs``)."""
    from src.priors import tanghuth as th
    if cell.get('arm') not in th.REFERENCE_ARMS:
        return f"arm {cell.get('arm')!r} is not a two-reference sensitivity arm {list(th.REFERENCE_ARMS)}"
    refs = [str(r) for r in cell.get('references') or []]
    if len(refs) != 2 or 'refs-' + '-'.join(refs) != folder.name:
        return f'references {refs} are not the pair of its folder {folder.name}'
    if [str(r) for r in (cell.get('ledger') or {}).get('references') or []] != refs:
        return f"ledger references {(cell.get('ledger') or {}).get('references')} are not the pair {refs}"
    if reference_pool is not None:
        pool = sorted(str(s) for s in reference_pool if str(s) != str(subject))
        want = list(th.sensitivity_pairs(pool, 'lpp', subject)[0]) if len(pool) >= 2 else None
        if refs != want:
            return f'references {refs} are not the registered pair {want} of {subject}'
    return None


def lpp_controls_problem(root, path, subject, clock, config, code):
    """None, or why a §9 control record (run_v4_lpp ``subject_controls``: <stage>/controls/<budget>_<arm>.json)
    cannot be locked: other code or configuration, fields that differ from its path, or inputs that are not the
    current cells of its 9 outer sections (a record computed from cells since recomputed)."""
    try:
        rec = json.loads(path.read_text())
    except ValueError as error:
        return f'unreadable ({error})'
    if not isinstance(rec, dict):
        return 'not a JSON object'
    budget, _, arm = path.stem.partition('_')
    if rec.get('code_sha256') != code or rec.get('config_sha256') != config:
        return 'not from the current LPP code and configuration'
    wrong = [k for k, v in (('kind', 'controls'), ('subject', subject), ('clock', clock), ('stage', 'develop'),
                            ('budget', budget), ('arm', arm)) if rec.get(k) != v]
    if wrong:
        return f'fields {wrong} differ from the path'
    inputs = rec.get('inputs') if isinstance(rec.get('inputs'), dict) else {}
    want = {f'{subject}/{clock}/develop/cells/t{t}_{budget}_{arm}.json' for t in sv.SECTIONS}
    if set(inputs) != want:
        return f'inputs {sorted(inputs)[:3]}... are not the 9 outer-section cells of {budget} {arm}'
    base = Path(root) / LPP_OUT
    stale = sorted(rel for rel, digest in inputs.items()
                   if not (base / rel).is_file() or sv.sha256_file(base / rel) != digest)
    if stale:
        return f'computed from cells that changed since: {stale[:3]}'
    return None


def development_cells(root, fp, dev, audit, reference_pool=None, require_lpp=True):
    """{'ds003020', 'ds003020_sensitivity', 'lpp_dev', 'lpp_dev_sensitivity', 'lpp_dev_controls', 'lpp_g4':
    {rel: sha256}, 'lpp_config': sha256 or None}; refuses any cell that cannot be locked (module docstring).

    LPP DEV: the cells/ folders, Module G's two-reference sensitivity cells (<DEV>/<clock>/develop/sensitivity/
    refs-<R1>-<R2>/cells/, ``lpp_pair_problem``) and the §9 control records (<DEV>/<clock>/develop/controls/
    <budget>_<arm>.json, ``lpp_controls_problem``; their float64 predictions are pinned by the digests in the
    cells)."""
    root = Path(root)
    out = dict(ds003020={}, ds003020_sensitivity={}, lpp_dev={}, lpp_dev_sensitivity={}, lpp_dev_controls={},
               lpp_g4={})
    problems = []
    for s in bud.SUBJECTS:
        for clock in sv.CLOCKS:
            for folder, refs, key in ds_folders(root, s, clock):
                for path in sorted(folder.glob('*.json')) if folder.is_dir() else []:
                    rel, m = path.relative_to(root).as_posix(), sv.DS_NAME.match(path.name)
                    if not m:
                        problems.append(f'{rel}: not a <budget>_<arm>.json cell')
                        continue
                    want = refs if refs is not None else (m['refs'].split('-') if m['refs'] else None)
                    expect = dict(subject=s, stage='develop', clock=clock, budget=m['budget'], arm=m['arm'])
                    problem = ds_cell_problem(path, expect, fp, audit, want)
                    if problem:
                        problems.append(f'{rel}: {problem}')
                    out['ds003020_sensitivity' if m['refs'] else key][rel] = sv.sha256_file(path)
    lpp, controls = [], []
    for s in dev:
        for clock in sv.CLOCKS:
            base = root / LPP_OUT / s / clock / 'develop'
            folder = base / 'cells'
            lpp += [('lpp_dev', s, clock, p) for p in (sorted(folder.glob('*.json')) if folder.is_dir() else [])]
            sens = base / 'sensitivity'
            for pair in sorted(sens.glob('refs-*')) if sens.is_dir() else []:
                lpp += [('lpp_dev_sensitivity', s, clock, p) for p in sorted((pair / 'cells').glob('*.json'))]
            folder = base / 'controls'
            controls += [(s, clock, p) for p in (sorted(folder.glob('*.json')) if folder.is_dir() else [])]
        folder = root / LPP_OUT / s / G4_CELLS
        lpp += [('lpp_g4', s, 'dense', p) for p in (sorted(folder.glob('*.json')) if folder.is_dir() else [])]
    configs = set()
    for *_, path in lpp:
        try:
            configs.add(json.loads(path.read_text()).get('config_sha256'))
        except (ValueError, AttributeError):
            pass
    config = next(iter(configs)) if len(configs) == 1 else None
    if lpp or require_lpp:
        need(config is not None,
             f'LPP DEV cells carry {len(configs)} configuration hashes: {sorted(map(str, configs))}')
        # run_v4_lpp confirm runs only under the lock's config_sha256, and QC compares CONFIRM cells with
        # lpp_config_sha256: the two must be one configuration.
        need(config == fp['config_sha256'], f"LPP DEV cells carry configuration {config}, not the locked "
                                            f"config_sha256 {fp['config_sha256']}")
    for key, s, clock, path in lpp:
        rel, m = path.relative_to(root).as_posix(), sv.LPP_NAME.match(path.name)
        if not m:
            problems.append(f'{rel}: not a t<t>_<budget>_<arm>.json cell')
            continue
        expect = dict(subject=s, clock=clock, stage='develop', section=m['section'], budget=m['budget'], arm=m['arm'])
        pair = key == 'lpp_dev_sensitivity'  # Its ledger lists the pair, not the pool: checked by lpp_pair_problem.
        problem, _, digest = sv.inspect_cell(path, expect, [], fp['lpp_cell_code_sha256'], config, audit,
                                             None if pair else reference_pool)
        if problem is not None and problem['category'] not in sv.NOT_ESTIMABLE:
            problems.append(f"{rel}: {problem['category']}: {problem['detail']}")
        elif pair:
            why = lpp_pair_problem(json.loads(path.read_text()), path.parent.parent, s, reference_pool)
            if why:
                problems.append(f'{rel}: {why}')
        out[key][rel] = digest
    for s, clock, path in controls:
        rel = path.relative_to(root).as_posix()
        why = lpp_controls_problem(root, path, s, clock, config, fp['lpp_cell_code_sha256'])
        if why:
            problems.append(f'{rel}: {why}')
        out['lpp_dev_controls'][rel] = sv.sha256_file(path)
    need(not problems, f'{len(problems)} development cells cannot be locked: {problems[:10]}')
    need(out['ds003020'], 'no ds003020 development cells')
    if require_lpp:
        need(out['lpp_dev'], 'no LPP DEV development cells')
        want = {f't{t}_L8_W0.json' for t in sv.SECTIONS}
        missing = [s for s in dev if not want <= {Path(rel).name for rel in out['lpp_g4']
                                                  if rel.startswith(f'{LPP_OUT.as_posix()}/{s}/')}]
        need(not missing, f'the LPP G4(b) cells (W0, L8, all 9 outer sections) are incomplete for {missing}')
    out['lpp_config'] = config
    return out


def dev_plan_check(root, pl, dev, code, config, audit):
    """The primary analysis's endpoint-blind QC on the LPP DEV cells (stage develop) must exclude nobody from any
    hypothesis and find complete ORACLE/W0 cells at every oracle budget: the locked plan is executable."""
    root = Path(root)
    dev_pl = dict(pl, cell_stage=sv.STAGE_DIRS['DEV'])
    rec = sv.qc(root, dev, dev_pl, code, sv.prepared_trs(root / sv.LPP_DATA), config, audit, reference_pool=dev)
    problems = {f"{s}: {r['category']}: {r['detail']}" for v in rec['samples'].values()
                for s, rs in v['excluded'].items() for r in rs}
    problems |= {f"{r['subject']}: {r['category']}: {r['detail']}" for r in rec['refusals']}
    problems |= {f'{b}: no complete ORACLE/W0 cells for {sorted(set(dev) - set(v))}'
                 for b, v in rec['oracle_subjects'].items() if set(v) != set(dev)}
    need(not problems, f'the LPP DEV cells do not exercise the locked confirmatory plan ({len(problems)} problems): '
                       f'{sorted(problems)[:10]}')
    return dict(stage=dev_pl['cell_stage'], subjects=list(dev), cells=len(rec['files']), required=rec['required'],
                oracle_budgets=sorted(rec['oracle_subjects']), excluded=0)


def gate_files(root, dev=(), subjects=bud.SUBJECTS):
    """sha256 of the gate inputs and outputs present on disk: g0, gates, simulate, g1 per ds003020 subject, LPP
    g4.json and, per DEV listener, its G4(b) record (dense/g4/g4b.json) and G4(c) record (g4c.json)."""
    root = Path(root)
    rels = [G0, GATES, SIMULATE, G4] + [sv.OUT / s / 'g1.json' for s in subjects]
    rels += [p for s in dev for p in (LPP_OUT / s / 'dense' / 'g4' / 'g4b.json', LPP_OUT / s / 'g4c.json')]
    return {Path(rel).as_posix(): sv.sha256_file(root / rel) for rel in rels if (root / rel).is_file()}


# --------------------------------------------------------------------------------------------------- LPP
def lpp_preparation(root, code):
    """Partition, mask digest and preparation-code fingerprint from the DEV preparation record."""
    root = Path(root)
    record = load(root, PREPARE, 'LPP DEV preparation record')
    missing = [k for k in lpp_prepare.DEV_RECORD_KEYS if k not in record]
    need(not missing, f'{PREPARE} lacks {missing}')
    need(record.get('mode') == 'DEV', f"{PREPARE} is a {record.get('mode')} record, not the DEV record")
    need((record.get('g4a') or {}).get('passed') is True, 'LPP G4(a) did not pass')
    changed = sorted(k for k in set(code) | set(record['code']) if code.get(k) != record['code'].get(k))
    need(not changed, f'the LPP DEV outputs were prepared by other code: {changed}')
    part = record.get('partition') or {}
    dev, confirm = list(part.get('dev') or []), list(part.get('confirm') or [])
    need(len(dev) + len(confirm) == lpp_prepare.N_EN, f'the partition has {len(dev) + len(confirm)} EN listeners, '
                                                         f'not {lpp_prepare.N_EN}')
    hashed = bud.lpp_partition(dev + confirm)
    need(list(hashed[0]) == dev and list(hashed[1]) == confirm, 'the recorded partition is not the registered hash '
                                                                 'partition')
    digest = str((record.get('mask') or {}).get('content_sha256'))
    need(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'no mask content digest in the record')
    verified = (root / MASK).is_file()
    if verified:
        import numpy as np
        with np.load(root / MASK) as handle:
            state = {k: handle[k] for k in handle.files}
        need(lpp_prepare.state_digest(state) == digest, f'{MASK} differs from the recorded mask content digest')
    mask = record['mask']
    return dict(partition=dict(dev=dev, confirm=confirm), mask_sha256=digest, code=dict(record['code']),
                record=dict(sha256=sv.sha256_file(root / PREPARE), created_utc=record.get('created_utc'),
                            mask_npz_verified=verified, mask_voxels=mask.get('mask_voxels'),
                            columns=mask.get('columns'), block_size=mask.get('block_size'),
                            deviations=record.get('deviations', [])))


def nothing_confirmatory(root, confirm):
    """Refuse when any confirmatory material or output already exists."""
    root = Path(root)
    found = [p.as_posix() for p in (sv.LOCK, sv.PRIMARY_JSON, sv.PRIMARY_MD, sv.STARTED, sv.QC_JSON)
             if (root / p).exists()]
    for base in (root / 'results' / 'v4', root / LPP_OUT):
        found += sorted(p.relative_to(root).as_posix() for p in base.glob('*/*/evaluate/**/*') if p.is_file())
    found += sorted(p.relative_to(root).as_posix() for p in (root / sv.LPP_DATA).glob('prepare_provenance_confirm*'))
    for s in confirm:
        found += [p.as_posix() for p in (LPP_OUT / s, sv.LPP_DATA / 'raw' / s, sv.LPP_DATA / 'semantic' / s)
                  if (root / p).exists()]
    need(not found, f'confirmatory material or outputs exist before the lock: {found[:10]}')


# ------------------------------------------------------------------------------------------- the record
def registered_constants(subjects):
    """Hash orders, seeds and grids of PROTOCOL_V4 §2 (by reference), with their realized values."""
    import numpy as np
    from src.priors import acoustic as acm
    return dict(
        reference='PROTOCOL_V4 §2 (Constants fixed now) and §4 items 5-9',
        hash_order_rule='ascending hexadecimal SHA-256 of the registered string',
        hash_orders=dict(ds003020_budget=f'{bud.BUDGET_PREFIX}<story>', lpp_subject=f'{bud.LPP_SUBJECT_PREFIX}'
                         '<participant_id>', lpp_section=f'{bud.LPP_SECTION_PREFIX}<t>:<s>',
                         th_sel_rand='ssltang-v4-thsel-rand:<subject>'),
        lpp_section_budgets={t: bud.lpp_section_budgets(t, sv.SECTIONS) for t in sv.SECTIONS},
        ladder=dict(levels=[0.75, 0.5, 0.25, 0.0], seed="int(sha256('ssltang-v4-ladder:<subject>:<f>')[:8], 16); "
                    "f = 0: int(sha256('ssltang-v4-perm:<subject>')[:8], 16)",
                    seeds={s: {f'{f:g}': ispm.ladder_seed(s, f) for f in (0.75, 0.5, 0.25, 0.0)} for s in subjects}),
        statistics=dict(sign_flip=dict(seed=st.FLIP_SEED, n=st.N_FLIPS), bca=dict(seed=st.BOOT_SEED, n=st.N_BOOT),
                        alpha=st.ALPHA, sesoi=st.SESOI, tiers=list(st.TIERS), lmm_optimizers=list(st.OPTIMIZERS),
                        lmm_fallback='lme4/lmerTest via rpy2, Satterthwaite df'),
        grids=dict(decoder_alpha='numpy.logspace(-1, 7, 17)', gamma=[0.5, 1.0, 2.0], encoding_alpha='decoder grid',
                   isp_self_converter_alpha=np.asarray(ispm.ALPHAS).tolist(), tie='larger alpha, tolerance 1e-12'),
        isp=dict(n_components=ispm.N_COMPONENTS, lags=list(ispm.LAGS), pca_seed=0, exact_svd_rows=ispm.EXACT_ROWS),
        eb=dict(shifts=list(ebm.SHIFTS), parts=ebm.SHIFT_PARTS, min_shift_tr=ebm.MIN_SHIFT_TR, clip=ebm.CLIP,
                mad_scale=ebm.MAD_SCALE, tau2_floor=ebm.TAU2_FLOOR),
        acoustic=dict(f0_method=getattr(acm, 'F0_METHOD', None)))


def code_files():
    """Every file whose bytes determine a locked number: CODE_FILES, run_v4.CORE, the v3/§12 files
    (run_v3.LOCKED_CODE, run_v3_weights.CODE) and the LPP preparation code (lpp_prepare.CODE_FILES)."""
    import run_v3
    import run_v3_weights
    import run_v4
    files = (list(CODE_FILES) + list(run_v4.CORE) + list(run_v3.LOCKED_CODE) + list(run_v3_weights.CODE)
             + list(lpp_prepare.CODE_FILES))
    return tuple(dict.fromkeys(files))


def file_hashes(files):
    return {f: sv.sha256_file(ROOT / f) for f in files if (ROOT / f).is_file()}


def git_state(files):
    """dict(head, dirty, ignored) of ``files`` in the running tree's git repository; None without git.

    ``dirty``: present files that are untracked or differ from HEAD; git-ignored files (the pinned reference/
    checkout) are listed in ``ignored`` and pinned by their hash in ``code_files`` instead.
    """
    def git(*args):
        return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)
    try:
        if git('rev-parse', '--show-toplevel').returncode:
            return None
    except OSError:
        return None
    present = [f for f in files if (ROOT / f).is_file()]
    ignored = [f for f in present if git('check-ignore', '-q', '--', f).returncode == 0]
    dirty = [f for f in present if f not in ignored and (git('ls-files', '--error-unmatch', '--', f).returncode
                                                         or git('diff', '--quiet', 'HEAD', '--', f).returncode)]
    return dict(head=git('rev-parse', 'HEAD').stdout.strip() or None, dirty=dirty, ignored=ignored)


def verify_consumers(lock, fp, dev, confirm, lpp_code, runners):
    """Run the lock through its readers; returns {reader: 'accepts' | 'refuses'}.

    Confirmatory: every reader must accept it. Reserved: the CONFIRM readers (lpp_prepare.check_lock,
    summarize_v4.check_lock, run_v4_lpp.check_lock) must refuse it and the ds003020 readers (run_v4 evaluate,
    run_v4_tanghuth.check_lock) accept it. ``runners`` adds the runner checks, which compare the lock with the
    running code (so only with fingerprints computed from it).
    """
    confirmatory = lock['confirmatory']
    result = {}

    def verdict(name, fn, errors, confirm_reader=True):
        try:
            fn()
            result[name], why = 'accepts', None
        except errors as error:
            result[name], why = 'refuses', getattr(error, 'reason', None) or str(error)
        if confirmatory or not confirm_reader:
            need(result[name] == 'accepts', f'{name} would refuse this lock: {why}')
        else:
            need(result[name] == 'refuses', f'{name} would accept a lock that reserves CONFIRM')
    verdict('lpp_prepare.check_lock', lambda: lpp_prepare.check_lock(lock, dev, confirm, lpp_code),
            (RuntimeError, KeyError, TypeError, ValueError))
    verdict('summarize_v4.check_lock', lambda: sv.check_lock(lock, fp), (sv.Refused,))
    if not runners:
        return result
    import run_v4
    import run_v4_lpp
    import run_v4_tanghuth
    from src.runtime import load_config

    def run_v4_evaluate():
        if lock.get('code_sha256') != run_v4.code_sha256():
            raise SystemExit('Code changed since the v4 lock')
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / sv.LOCK.name
        path.write_text(json.dumps(lock))
        verdict('run_v4 --stage evaluate', run_v4_evaluate, (SystemExit,), confirm_reader=False)
        verdict('run_v4_tanghuth.check_lock', lambda: run_v4_tanghuth.check_lock(path), (SystemExit,),
                confirm_reader=False)
        ctx = run_v4_lpp.Context(load_config(ROOT / sv.CONFIG, []), device='cpu', lock_path=path)
        subject = confirm[0] if confirm else 'sub-none'
        verdict('run_v4_lpp.check_lock', lambda: run_v4_lpp.check_lock(ctx, subject, git_state=None), (SystemExit,))
    return result


def development_record(ds_tab, lpp_tab, lpp_eb, decision, dev, confirmatory):
    """(effects, R_P(K) fit): required at the primary budget for a confirmatory lock; descriptive (every candidate
    arm, errors recorded) for a reserved one."""
    advanced, budgets = decision['advanced'], dict(ds003020=decision['ds003020'], lpp=decision['lpp'])
    if confirmatory:
        effects = dict(
            ds003020=sv.development_effects(ds_tab, sv.CLOCK, budgets['ds003020'], advanced, sv.DS_DEV_SUBJECTS,
                                            'ds003020'),
            lpp=sv.development_effects(lpp_tab, sv.CLOCK, budgets['lpp'], advanced, dev, 'lpp'))
        return effects, sv.reliability_fit(lpp_eb, sv.CLOCK, budgets['lpp'])
    effects = dict(descriptive=True)
    for dataset, tab, subjects in (('ds003020', ds_tab, sv.DS_DEV_SUBJECTS), ('lpp', lpp_tab, dev)):
        if budgets[dataset] is None or not subjects:
            continue
        effects[dataset] = {arm: attempt(lambda arm=arm: sv.development_effects(
            tab, sv.CLOCK, budgets[dataset], arm, subjects, dataset)) for arm in ([advanced] if advanced
                                                                                   else sv.ADVANCEABLE)}
    reliability = attempt(lambda: sv.reliability_fit(lpp_eb, sv.CLOCK, budgets['lpp'])) if budgets['lpp'] else None
    return effects, reliability


def rp_k_gaps(reliability, dev):
    """Why the DEV R_P(K) fit is not the registered one (FUTURE_WORK_V4 §8.4: K <= 8 fitted with the Spearman-Brown
    form): each of RP_K_LEVELS must be present for every DEV listener; [] when it is."""
    by_k = (reliability or {}).get('by_k') or {}
    gaps = []
    for k in RP_K_LEVELS:
        got = sorted((by_k.get(str(k)) or {}).get('subjects') or {})
        if got != sorted(dev):
            gaps.append(f'K = {k}: {len(got)} of {len(dev)} DEV listeners')
    return gaps


def build_lock(root=ROOT, fingerprints=None, lpp_code=None, audit=sv.default_audit, git='auto', no_git_check=False,
               reserve_confirm=False, runners=None, created_utc=None, deviation=None):
    """The lock record (module docstring); raises Refused on any failed precondition.

    ``fingerprints`` None computes them from the running code (``summarize_v4.current_fingerprints``) and runs
    the runner checks of ``verify_consumers`` (``runners`` overrides); ``git`` 'auto' asks git (tests pass a
    state or None); ``no_git_check`` permits the lock without git; ``reserve_confirm`` permits a lock that
    reserves CONFIRM (required exactly when the gates do not register it); ``deviation`` (--deviation) names the
    PROTOCOL_V4 §4 entry that covers a confirmatory lock whose R_P(K) fit lacks registered K levels (``rp_k_gaps``),
    recorded in the lock.
    """
    root = Path(root)
    real = fingerprints is None
    try:
        fp = sv.current_fingerprints() if real else dict(fingerprints)
        missing = [k for k in sv.FINGERPRINTS if not fp.get(k)]
        need(not missing, f'fingerprints missing: {missing}')
        core = fp['code_sha256']
        check_g0(load(root, G0, 'G0 integrity'), core)
        gates = load(root, GATES, 'development gates')
        decision = gate_decision(gates, core)
        verification = verify_gates(root, gates, core, decision['stop'])
        lpp_code = lpp_prepare.code_fingerprint() if lpp_code is None else lpp_code
        prep = attempt(lambda: lpp_preparation(root, lpp_code))
        part = prep.get('partition')
        g4 = lpp_g4(root, fp, list(part['dev']) if part else None)
        reasons = list(decision['reasons'])
        if decision['advanced'] is not None:
            need(g4['status'] != 'missing', f'{decision["advanced"]} advanced but LPP G4 has not been run ({G4}): '
                                            'the lock cannot tell whether CONFIRM is registered')
            if g4['status'] == 'failed':
                reasons.append('LPP G4 failed under the current LPP code: CONFIRM is not registered (GIN fallback '
                               'with a downgraded claim, FUTURE_WORK_V4 §7)')
        confirmatory = not reasons
        need(confirmatory or reserve_confirm, f'the gates register no confirmatory LPP run ({reasons}); a lock that '
                                              'reserves CONFIRM is written only with --reserve-confirm')
        need(not (confirmatory and reserve_confirm), 'the gates register the confirmatory LPP run: --reserve-confirm '
                                                     'does not apply')
        need(not (confirmatory and 'error' in prep), prep.get('error'))
        dev = list(part['dev']) if part else lpp_subjects_on_disk(root)
        confirm = list(part['confirm']) if part else []
        nothing_confirmatory(root, confirm)
        # The DEV reference pool as run_v4_lpp.reference_pool forms it (summarize_v4.registered_pool).
        pool = sv.registered_pool(dev, sv.prepared_trs(root / sv.LPP_DATA)) if part else None
        cells = development_cells(root, fp, dev, audit, reference_pool=pool, require_lpp=confirmatory)
        state = git_state(code_files()) if git == 'auto' else git
        need(state is not None or no_git_check, 'git is unavailable: the fingerprinted files cannot be verified as '
                                                'committed; lock where git is available, or pass --no-git-check '
                                                '(recorded in the lock; then verify the lock with --verify where '
                                                'git is available before committing it)')
        need(state is None or not state['dirty'], f"fingerprinted files are not committed: {state and state['dirty']}")
        advanced, budgets = decision['advanced'], dict(ds003020=decision['ds003020'], lpp=decision['lpp'])
        pl = sv.plan(advanced, decision['drop_h6'], budgets['lpp']) if confirmatory else None
        if confirmatory:
            need(g4.get('verified') is True, f'{G4} could not be recomputed from the per-listener records')
        plan_check = (dev_plan_check(root, pl, dev, fp['lpp_cell_code_sha256'], cells['lpp_config'], audit)
                      if confirmatory else None)
        ds_tab, ds_eb, _ = sv.ds_table(root, bud.SUBJECTS, 'develop', fp)
        lpp_tab, lpp_eb, _ = sv.lpp_table(root, dev, 'develop', fp['lpp_cell_code_sha256'])
        effects, reliability = development_record(ds_tab, lpp_tab, lpp_eb, decision, dev, confirmatory)
        if confirmatory:
            gaps = rp_k_gaps(reliability, dev)
            need(not gaps or deviation, f'the R_P(K) fit on DEV is not the registered K <= 8 fit (FUTURE_WORK_V4 '
                                        f'§8.4) but an extrapolation from {reliability.get("n_k_levels")} K level(s): '
                                        f'{gaps}; run the EB_K subset arms on DEV (run_v4_lpp.py develop --arms '
                                        'EB_K), or record a PROTOCOL_V4 §4 deviation and pass --deviation naming it')
            reliability = dict(reliability, registered_levels=list(RP_K_LEVELS), gaps=gaps,
                               deviation=deviation if gaps else None)
        kappa = dict(ds003020=sv.kappa_schedule(ds_eb, 'ds003020'), lpp=sv.kappa_schedule(lpp_eb, 'lpp'))
    except sv.Refused as error:
        raise Refused(error.reason) from error
    lock = dict(
        protocol='ssltang-v4', stage='lock', created_utc=created_utc or sv.now_utc(),
        registration=dict(documents=file_hashes(DOCS), deviations='PROTOCOL_V4 §4 as of these document hashes'),
        confirmatory=confirmatory,
        confirm_status=dict(status='registered' if confirmatory else 'reserved', reasons=reasons),
        advanced_arm=advanced, primary_budget=budgets, drop_h6=decision['drop_h6'], confirmatory_plan=pl,
        **{k: fp[k] for k in sv.FINGERPRINTS},
        lpp_config_sha256=cells['lpp_config'],
        lock_writer_sha256=sv.sha256_file(Path(__file__).resolve()),
        code_files=file_hashes(code_files()),
        gates=dict(decision=gates.get('decision'), stop=decision['stop'], verification=verification,
                   G1={k: v for k, v in gates['G1'].items() if k in ('S3', 'S6', 'primary', 'lpp_primary', 'decision')},
                   G_FEAS=gates.get('G_FEAS'), G2=gates.get('G2'), G3=gates.get('G3'), lpp_g4=g4),
        gate_files=gate_files(root, dev),
        development_cells=dict({k: cells[k] for k in CELL_KEYS}, counts={k: len(cells[k]) for k in CELL_KEYS}),
        lpp_dev_plan_check=plan_check,
        development_effects=effects, rp_k=reliability, kappa_schedule=kappa,
        hash_orders_and_seeds=registered_constants(list(bud.SUBJECTS) + dev + confirm), deviation=deviation,
        git=state, git_check='verified' if state is not None else 'skipped (--no-git-check)')
    if confirmatory:
        lock.update(lpp_partition=prep['partition'], lpp_mask_sha256=prep['mask_sha256'], lpp_code_sha256=prep['code'],
                    lpp_dataset_doi=lpp_prepare.DOI, lpp_preparation=prep['record'])
    else:
        lock['lpp_reserved'] = dict(
            note='CONFIRM is reserved: lpp_partition, lpp_mask_sha256 and lpp_code_sha256, which authorize a CONFIRM '
                 'download and run, are deliberately absent', partition=part, mask_sha256=prep.get('mask_sha256'),
            code=prep.get('code'), preparation=prep.get('record') or dict(error=prep.get('error')),
            dataset_doi=lpp_prepare.DOI)
    lock = sv.clean(lock)
    lock['readers'] = verify_consumers(lock, fp, dev, confirm, lpp_code, real if runners is None else runners)
    return lock


def verify_lock(root=ROOT, fingerprints=None, tree='auto'):
    """``--verify``: the lock against the running code and the committed, pushed git tree; needs git.

    Run where git is available, after pulling a lock written with --no-git-check and before committing it, and
    again once it is committed and pushed. Problems: a FINGERPRINTS value that differs from the running code; a
    fingerprinted file (``code_files()``) the lock does not pin; scripts/v4_lock.py differing from the lock
    writer; and every ``summarize_v4.git_tree`` problem (pinned files and registration documents against the lock's
    commit, or HEAD while the lock is untracked, which must be on origin/main; pinned code as it runs here).
    Returns the verification record; raises Refused on any problem.
    """
    root = Path(root)
    lock = load(root, sv.LOCK, 'the lock to verify')
    fp = sv.current_fingerprints() if fingerprints is None else dict(fingerprints)
    problems = [f'{k} differs from the running code' for k in sv.lock_mismatches(lock, fp)]
    pinned = set(lock.get('code_files') or {})
    problems += [f'{f} is fingerprinted but not pinned by the lock' for f in code_files()
                 if (ROOT / f).is_file() and f not in pinned]
    if lock.get('lock_writer_sha256') != sv.sha256_file(Path(__file__).resolve()):
        problems.append('scripts/v4_lock.py differs from the lock writer (lock_writer_sha256)')
    state = sv.git_tree(lock, root / sv.LOCK) if tree == 'auto' else tree
    need(state is not None, 'git is unavailable: --verify compares the lock with the committed tree and needs git')
    problems += state['problems']
    record = dict(lock_sha256=sv.sha256_file(root / sv.LOCK), lock_git_check=lock.get('git_check'),
                  commit=state['commit'], lock_tracked=state['lock_tracked'], files=state['files'],
                  documents=state['documents'], ignored=state['ignored'], problems=problems, verified=not problems)
    need(not problems, f'the lock does not match the running code and the committed, pushed tree '
                       f'({len(problems)} problems): {problems[:10]}')
    return record


def write_lock(lock, root=ROOT):
    path = Path(root) / sv.LOCK
    need(not path.exists(), f'{sv.LOCK} already exists; the lock is written once')
    sv.write_json(lock, path)
    return sv.sha256_file(path)


def summary_of(lock):
    effects = lock.get('development_effects') or {}
    dz = {k: dict(minus_w1=v['minus_w1']['dz'], minus_w0=v['minus_w0']['dz'])
          for k, v in effects.items() if isinstance(v, dict) and 'minus_w1' in v}
    return dict(confirm_status=lock['confirm_status'], advanced_arm=lock['advanced_arm'],
                primary_budget=lock['primary_budget'], drop_h6=lock['drop_h6'],
                cells=lock['development_cells']['counts'], dz=dz, readers=lock.get('readers'),
                rp_k_prediction=(lock.get('rp_k') or {}).get('prediction'),
                rp_k_levels=(lock.get('rp_k') or {}).get('n_k_levels'), deviation=lock.get('deviation'),
                lpp_g4_verified=((lock.get('gates') or {}).get('lpp_g4') or {}).get('verified'),
                git_check=lock['git_check'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default=str(ROOT), help='project root holding results/, data/ and experiments/ (code '
                                                     'fingerprints always come from the running code)')
    p.add_argument('--dry-run', action='store_true', help='print the lock summary without writing it')
    p.add_argument('--no-git-check', action='store_true',
                   help='lock where git is unavailable (the lock records that the commit check was skipped)')
    p.add_argument('--reserve-confirm', action='store_true',
                   help='write the lock that reserves CONFIRM (only when the gates register no confirmatory run)')
    p.add_argument('--deviation', help='the PROTOCOL_V4 §4 entry covering an R_P(K) fit without the registered DEV '
                                       'K levels (recorded in the lock)')
    p.add_argument('--verify', action='store_true',
                   help='verify the existing lock against the running code and the committed, pushed git tree '
                        '(needs git; run before committing a lock written with --no-git-check, and after pushing it)')
    a = p.parse_args(argv)
    if a.verify:
        print(json.dumps(verify_lock(Path(a.root)), indent=1))
        return
    lock = build_lock(Path(a.root), no_git_check=a.no_git_check, reserve_confirm=a.reserve_confirm,
                      deviation=a.deviation)
    summary = summary_of(lock)
    if a.dry_run:
        print(json.dumps(summary, indent=1))
        return
    digest = write_lock(lock, Path(a.root))
    print(json.dumps(dict(summary, lock=sv.LOCK.as_posix(), lock_sha256=digest), indent=1))
    if lock['git_check'] != 'verified':
        print('The lock was written without git: run scripts/v4_lock.py --verify where git is available before '
              'committing it (summarize_v4.py --confirm verifies the committed tree again).')
    print('Commit and push experiments/v4_lock.json to main before any ds003020 test score'
          + (f' or CONFIRM download; lpp_prepare.py --confirm needs --lock-sha256 {digest}.' if lock['confirmatory']
             else '; CONFIRM stays untouched (reserved lock).'))


if __name__ == '__main__':
    main()
