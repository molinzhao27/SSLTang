#!/usr/bin/env python3
"""Re-audit saved v4 cells: replay every embedded provenance log against R1-R7 and its registered scope (G0e).

Usage: scripts/audit_v4.py [--manifests data/semantic] <cell.json> [...]. Each file is one cell (a JSON
object) and may embed one or more logs (``Provenance.to_dict``, found anywhere in the JSON by their schema).
Neither the stored audit result nor the log's own scope is trusted:
  0. The cell names its ``subject`` and ``stage`` (and, at develop/evaluate, its ``arm``). Its score payload
     follows the stage: nothing at g1; ds003020 ``test``/``test_shift_null`` only at evaluate.
  1. ``audit`` is re-run on each rebuilt log (rules R1-R7).
  2. The scope is recomputed. The budget comes from the registered hash order: ``ds003020_budgets`` over
     the ten subject manifests, or ``lpp_section_budgets`` for the log's single outer test section. The
     validation/test stories come from the target manifest's splits (LPP: none, the outer section). The
     scored items follow the stage: g1 nothing; develop validation (LPP: the outer section); evaluate
     validation and test. The budget name comes from the log or the cell's ``budget``, the stage from the
     cell, and the log's must agree when given. The log's target must be the cell's ``subject``. A log's arm
     must be the cell's ``arm``; an ORACLE log is also accepted in a g1 cell (the full-label W1 map). Only a
     secondary arm (``SECONDARY_ARMS``) may carry the TS or SS reference rule. Regime L needs the cell to
     declare ``regime='L'``. A cell ledger's reference rule must be the log's. TS stimuli must be absent
     from the target's whole manifest (LPP has no TS material: everyone heard the book).
  3. Inner folds (develop/evaluate): the log declares exactly the registered inner decoder folds, as
     {position: held stories} (``inner_folds`` over the budget's hash order; ALL: ``all_folds``, the v3
     session rule), and every fold logs tagged target reads (``fold_reads`` of the cell's arm) over its
     training part (ISP: its shared stories) and target BOLD ``decoder_score`` on its held part. With R7
     (a tagged read never fits on the fold's held part), a leaky per-fold quantity must show up tagged.
  4. Completeness: every log has reads; develop/evaluate logs read target BOLD and transcripts for the
     final ``decoder_fit`` (untagged) over the whole budget and target BOLD for ``decoder_score``
     (untagged) on every scored item.
One JSON line is printed per file, then a summary. Exits 1 if any file has a violation, a scope error,
an unreadable log or no log at all, else 0.
"""
import argparse
from collections import namedtuple
from functools import lru_cache
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import budgets as bud  # noqa: E402
from src.provenance import SCHEMA, SECONDARY_ARMS, STAGES, Provenance  # noqa: E402

MANIFESTS = ROOT / 'data' / 'semantic'
ISP_ARMS = ('PRIOR', 'EB', 'EB_SELF', 'EB_MISALIGN')  # Plus the EB_f ladder.
Scope = namedtuple('Scope', 'budget val test scored folds isp')


def logs(obj, path='$'):
    """(JSON path, provenance log) for every embedded log, depth-first."""
    if isinstance(obj, dict):
        if obj.get('schema') == SCHEMA:
            yield path, obj
            return
        for key, value in obj.items():
            yield from logs(value, f'{path}.{key}')
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from logs(value, f'{path}[{i}]')


@lru_cache(maxsize=None)
def manifests(root):
    """{subject: manifest} for UTS01-UTS10 under ``root``."""
    return {s: json.loads((Path(root) / s / 'semantic_manifest.json').read_text()) for s in bud.SUBJECTS}


def whole(items, what):
    """Stimulus set of whole-stimulus items (registered budgets and splits are whole stories or sections)."""
    partial = sorted(i.stimulus for i in items if i.trs is not None)
    if partial:
        raise ValueError(f'{what} items {partial} are partial stimuli')
    return {i.stimulus for i in items}


def registered(p, name, stage, root):
    """Registered ``Scope`` of log ``p`` at budget ``name`` and ``stage``: the budget (list, hash order), the
    validation, test and scored stimulus sets, the inner decoder folds (list of held-story tuples; None
    if not estimable) and the ISP stories (shared by every subject; None: all)."""
    if p.dataset == 'ds003020':
        if name not in bud.DS003020_BUDGETS:
            raise ValueError(f'{name!r} is not a ds003020 budget')
        ms = manifests(str(root))
        splits, shared = bud.split_stories(ms[p.target]), bud.shared_stories(ms)
        val, test = set(splits.get('val', ())), set(splits.get('test', ()))
        budget = bud.ds003020_budgets(ms[p.target], shared)[name]
        folds = bud.all_folds(ms[p.target]) if name == 'ALL' else bud.inner_folds(name, budget)
        return Scope(budget, val, test, {'g1': set(), 'develop': val, 'evaluate': val | test}[stage], folds,
                     set(shared))
    if name not in bud.LPP_SIZES or stage == 'g1':
        raise ValueError(f'{name!r} at stage {stage} is not an LPP budget cell')
    test = whole(p.test_items, 'test')
    if len(test) != 1 or not test <= set(bud.LPP_SECTIONS):
        raise ValueError(f'LPP test items {sorted(test)} are not one outer section')
    budget = bud.lpp_section_budgets(next(iter(test)), bud.LPP_SECTIONS)[name]
    return Scope(budget, set(), test, test, bud.inner_folds(name, budget), None)


def fold_reads(arm):
    """(kind, purpose, ISP stories only) target reads every inner fold of ``arm`` logs over its training part.

    All arms standardize and fit the decoder per fold; W1 and the EB arms also refit the own encoding
    scores, the ISP and SELF priors (PRIOR, EB, the EB_f ladder, EB_SELF, EB_MISALIGN) read the target
    BOLD of the fold's shared stories, and P_AC its audio.
    """
    arm = str(arm)
    out = [('brain', 'standardize', False), ('brain', 'decoder_fit', False), ('transcript', 'decoder_fit', False)]
    if arm == 'W1' or arm.startswith('EB'):
        out.append(('transcript', 'own_encoding', False))
    if arm in ISP_ARMS or arm.startswith('EB_f'):
        out.append(('brain', 'isp_response', True))
    if arm == 'P_AC':
        out.append(('audio', 'own_encoding', False))
    return out


def cell_errors(cell):
    """Reasons why the cell's own fields (subject, stage, arm, score payload) are not a v4 cell's."""
    if not isinstance(cell, dict):
        return ['the cell is not a JSON object']
    errors, subject, stage = [], cell.get('subject'), cell.get('stage')
    scored = [k for k in ('val', 'test', 'test_shift_null') if k in cell]
    if not isinstance(subject, str):
        errors.append('the cell names no subject')
    if stage not in STAGES:
        errors.append(f'the cell stage {stage!r} is not one of {STAGES}')
    elif stage == 'g1' and scored:
        errors.append(f'stage g1 scores nothing, but the cell holds {scored}')
    elif stage != 'g1' and not isinstance(cell.get('arm'), str):
        errors.append(f'a {stage} cell names no arm')
    early = [k for k in scored if k.startswith('test')]
    if stage == 'develop' and early and isinstance(subject, str) and bud.dataset_of(subject) == 'ds003020':
        errors.append(f'{early} scored at stage {stage}: ds003020 test stories are scored only at evaluate')
    return errors


def scope_errors(cell, p, root=MANIFESTS):
    """Reasons why rebuilt log ``p`` is not the registered, complete scope of its ``cell`` (empty if it is)."""
    cell = cell if isinstance(cell, dict) else {}
    meta = {k: cell[k] for k in ('subject', 'arm', 'stage', 'budget', 'regime') if isinstance(cell.get(k), str)}
    errors = []

    def need(ok, reason):
        if not ok:
            errors.append(reason)
    stage, name = meta.get('stage') or p.stage, p.budget_name or meta.get('budget')
    need(meta.get('subject') in (None, p.target),
         f"log target {p.target} is not the cell subject {meta.get('subject')}")
    need(None in (p.stage, meta.get('stage')) or p.stage == meta['stage'],
         f"log stage {p.stage} in a {meta.get('stage')} cell")
    need(None in (p.budget_name, meta.get('budget')) or p.budget_name == meta['budget'],
         f"log budget {p.budget_name} in a {meta.get('budget')} cell")
    need(p.arm is None or p.arm == meta.get('arm') or (p.arm == 'ORACLE' and stage == 'g1'),
         f"log of arm {p.arm} in a cell of arm {meta.get('arm')}")
    need(p.reference_rule == 'none' or meta.get('arm') in SECONDARY_ARMS,
         f"reference rule {p.reference_rule} in a cell of arm {meta.get('arm')}: TS/SS belong to the secondary arms")
    need(p.regime == 'S' or meta.get('regime') == 'L', 'regime L log in a cell that does not declare regime L')
    for key in ('ledger', 'ledger_record'):
        ledger = cell.get(key)
        if isinstance(ledger, dict) and 'reference_rule' in ledger:
            need(ledger['reference_rule'] == p.reference_rule,
                 f"{key} reference rule {ledger['reference_rule']} but the log's is {p.reference_rule}")
    need(bool(p.records), 'no reads recorded')
    need(stage in STAGES, 'no stage: the scored items cannot be checked')
    need(name is not None, 'no budget name: the budget cannot be checked')
    if stage not in STAGES or name is None:
        return errors
    try:
        expected = registered(p, name, stage, root)
    except (OSError, KeyError, ValueError) as error:
        return errors + [f'registered scope unavailable: {error!r}']
    for what, items, want in zip(('budget', 'validation', 'test', 'scored'),
                                 (p.budget, p.val_items, p.test_items, p.score_items),
                                 (set(expected.budget),) + expected[1:4]):
        try:
            got = whole(items, what)
        except ValueError as error:
            errors.append(str(error))
            continue
        need(got == want, f'{what} {sorted(got)} is not the registered {name} {stage} {what} {sorted(want)}')
    if p.ts_allowed is not None:
        heard = (set(bud.LPP_SECTIONS) if p.dataset == 'lpp'
                 else {r['story'] for r in manifests(str(root))[p.target]['runs']})
        need(not heard & {i.stimulus for i in p.ts_allowed},
             f'TS stimuli {sorted(heard & {i.stimulus for i in p.ts_allowed})} were heard by the target')
    if stage == 'g1':
        return errors
    for kind, purpose, items in (('brain', 'decoder_fit', p.budget), ('transcript', 'decoder_fit', p.budget),
                                 ('brain', 'decoder_score', p.score_items)):
        missing = [i.stimulus for i in p.uncovered(kind, purpose, items, fold=None)]
        need(not missing, f'no untagged (final) target {kind} {purpose} read for {missing}')
    want = None if expected.folds is None else {str(k): set(f) for k, f in enumerate(expected.folds)}
    try:
        got = None if p.folds is None else {k: whole(v, f'fold {k}') for k, v in p.folds.items()}
    except ValueError as error:
        return errors + [str(error)]
    if want is None or got != want:
        shown = lambda folds: None if folds is None else {k: sorted(v) for k, v in folds.items()}
        return errors + [f'inner folds {shown(got)} are not the registered {name} folds {shown(want)}: '
                         'R7 cannot be checked']
    arm = p.arm or meta.get('arm')
    for k, held in want.items():
        fit = [s for s in expected.budget if s not in held]
        for kind, purpose, isp in fold_reads(arm):
            items = [s for s in fit if not isp or expected.isp is None or s in expected.isp]
            missing = [i.stimulus for i in p.uncovered(kind, purpose, items, fold=k)]
            need(not missing, f'inner fold {k}: no tagged target {kind} {purpose} read for {missing}')
        missing = [i.stimulus for i in p.uncovered('brain', 'decoder_score', sorted(held), fold=k)]
        need(not missing, f'inner fold {k}: no tagged target brain decoder_score read for {missing}')
    return errors


def audit_file(path, root=MANIFESTS):
    out = dict(file=str(path), logs=0, violations=[], flags=0)
    try:
        cell = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        return dict(out, passed=False, error=f'unreadable: {error}')
    out['violations'] += [dict(rule='scope', path='$', reason=r) for r in cell_errors(cell)]
    for where, log in logs(cell):
        out['logs'] += 1
        try:
            p = Provenance.from_dict(log)
        except (KeyError, TypeError, ValueError) as error:
            out['violations'].append(dict(rule='invalid', path=where, reason=f'unreadable log: {error!r}'))
            continue
        result = p.audit()
        out['flags'] += len(result['flags'])
        out['violations'] += [dict(v, path=where) for v in result['violations']]
        out['violations'] += [dict(rule='scope', path=where, reason=r) for r in scope_errors(cell, p, root)]
    if not out['logs']:
        out['error'] = 'no provenance log'
    out['rules'] = sorted({v['rule'] for v in out['violations']})
    out['passed'] = bool(out['logs']) and not out['violations']
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cells', nargs='+', help='cell JSON files')
    p.add_argument('--manifests', default=str(MANIFESTS),
                   help='ds003020 manifest root: <dir>/<subject>/semantic_manifest.json')
    a = p.parse_args(argv)
    failed = []
    for path in a.cells:
        out = audit_file(path, a.manifests)
        if not out['passed']:
            failed.append(path)
        print(json.dumps(out), flush=True)
    print(json.dumps(dict(files=len(a.cells), failed=len(failed), passed=not failed)), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
