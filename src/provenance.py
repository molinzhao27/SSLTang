"""v4 read provenance and the leakage rules R1-R7 (FUTURE_WORK_V4 §3; G0e).

Every loader logs one record per read: (subject, stimulus, kind, purpose, TR range, GPT-2 context,
inner fold). ``check`` replays the log against the cell's scope: the target's budget B, its validation
and test stimuli, the items this cell may score, the inner decoder folds, the reference-transcript rule
(TS list, SS or none) and the arm. Items are stimulus names, ``(subject, stimulus)`` pairs,
``(stimulus, (start, stop))`` or ``(subject, stimulus, (start, stop))``; TR ranges are half-open and None
means the whole stimulus. Budget, validation, test, scored and fold items describe the target's material,
so they match reads of any subject by stimulus and time ("budget stimuli and times"; R5's "any subject").
Duplicate reads are stored once with a count. ds003020 targets are UTS01-UTS10; any other target is an
LPP listener.

Defaults fail closed: no reference transcripts (rule 'none'; TS or SS only for a ``SECONDARY_ARMS`` arm),
no oracle reads (only ``arm='ORACLE'``), scoring only the validation stories (ds003020) or the outer test
section (LPP), and nothing in stage g1; ds003020 test stories are scored only at stage 'evaluate', whatever
``score_items`` says. LPP cells always require context resets. Inner folds, when declared, partition the
budget into at least two non-empty held-out parts.

Rules (a read may break several; ``check`` names every rule broken):
  R1  target transcript/audio only at budget stimuli and times, and never for a transcript-free purpose
      (``TRANSCRIPT_FREE``: ISP, PCA bases, converters). ``oracle`` reads are flagged in an ORACLE cell and
      violations elsewhere. Scoring reads (``decoder_score``, ``nuisance_apply``) of validation/test
      stimuli are evaluation, not pairing (R5 limits them to the scored items).
  R2  target BOLD outside B only for ``decoder_score`` on validation/test stimuli. ``oracle`` as for R1;
      regime L also allows ``isp_response`` on non-validation/test material (flagged, not scan-matched).
  R3  reference BOLD: ``isp_predictor`` and ``converter`` only at budget stimuli and times (regime L:
      ``isp_predictor`` on any non-validation/test material, flagged); ``pca_basis`` and ``reference_model``
      on anything but validation/test material; no other purpose.
  R4  reference models (``reference_model`` reads of any kind) belong to the secondary arms only
      (``SECONDARY_ARMS``; FUTURE_WORK_V4 §3), and a reference transcript/audio read feeds nothing else:
      TS needs a ``ts_allowed`` stimulus, SS excludes validation/test stimuli and is flagged, rule 'none'
      (the default, and the only rule a Tier-1 or shared log may carry) allows nothing.
  R5  validation/test material (any subject) feeds only ``decoder_score``, ``nuisance_apply`` and
      ``context_reset``: never a prior, converter, reference model, standardization, nuisance fit, decoder
      fit or the oracle, and never anything inside an inner fold (selection). Scoring reads stay on the
      cell's scored items (no test score before the lock).
  R6  GPT-2 context never crosses a stimulus: a transcript read's ``context`` (the stimuli whose words fed
      its causal context) must be its own stimulus; with ``require_context_reset`` (always on in LPP), every
      target transcript feature read needs a ``context_reset`` record for that section, which the target
      builder writes.
  R7  inner folds (§3: every target-derived quantity is recomputed from the fold's training part of B): a
      read tagged with inner decoder fold f may not fit anything (``FOLD_FIT``, or any target read except
      scoring, context resets and the oracle) on f's held-out material, and its scoring reads
      (``decoder_score``, ``nuisance_apply``) stay on f's held-out material. Tag reads with ``fold=`` or
      inside ``with log.inner_fold(f):``; ``scripts/audit_v4.py`` requires the registered folds and the
      tagged reads of every fold.
"""
from __future__ import annotations
from collections import namedtuple
from contextlib import contextmanager
from .budgets import DS003020_BUDGETS, LPP_SIZES, REFERENCE_RULES, dataset_of

SCHEMA = 'ssltang-v4-provenance'
KINDS = ('brain', 'transcript', 'audio')
PURPOSES = ('decoder_fit', 'decoder_score', 'standardize', 'own_encoding', 'isp_response', 'isp_predictor',
            'pca_basis', 'reference_model', 'converter', 'nuisance_fit', 'nuisance_apply', 'oracle',
            'context_reset')
REGIMES = ('S', 'L')
STAGES = ('g1', 'develop', 'evaluate')
BUDGET_NAMES = DS003020_BUDGETS + tuple(LPP_SIZES)
SCORING = ('decoder_score', 'nuisance_apply', 'context_reset')  # The only uses of validation/test material (R5).
TRANSCRIPT_FREE = ('isp_response', 'isp_predictor', 'pca_basis', 'converter')  # Label-free priors (R1).
FOLD_FIT = ('decoder_fit', 'standardize', 'own_encoding', 'isp_response', 'isp_predictor', 'converter',
            'nuisance_fit')  # Fitted per inner fold from its training part of B (R7).
FOLD_SCORE = ('decoder_score', 'nuisance_apply')  # Inner-CV scoring: the fold's held-out part only (R7).
SECONDARY_ARMS = ('TH_SEL', 'TH_SEL_soft', 'XDEC', 'HYB', 'SISP', 'EB_SISP', 'XENC', 'EB_XENC', 'WEN', 'POP_anat',
                  'EB_POP')  # The only arms that may use reference transcripts or reference models (R4).
ANY_FOLD = object()  # ``uncovered``: reads of any inner fold, tagged or not.
Item = namedtuple('Item', 'subject stimulus trs')
Read = namedtuple('Read', 'subject stimulus kind purpose trs context fold')


class ProvenanceError(RuntimeError):
    """A provenance log broke at least one rule; ``rules`` names them and ``violations`` lists every one."""

    def __init__(self, violations):
        self.violations = list(violations)
        self.rules = sorted({v['rule'] for v in self.violations})
        shown = '; '.join(f"{v['rule']} {v['read']}: {v['reason']}" for v in self.violations[:10])
        more = f'; and {len(self.violations) - 10} more' if len(self.violations) > 10 else ''
        super().__init__(f"Provenance violations ({', '.join(self.rules)}): {shown}{more}")


def _span(trs):
    if trs is None:
        return None
    if isinstance(trs, range):
        if trs.step != 1:
            raise ValueError('TR ranges must be contiguous')
        trs = (trs.start, trs.stop)
    start, stop = (int(t) for t in trs)
    if not 0 <= start < stop:
        raise ValueError(f'Invalid TR range {trs}')
    return start, stop


def _item(value):
    if isinstance(value, Item):
        return value
    if isinstance(value, dict):
        return Item(None if value.get('subject') is None else str(value['subject']), str(value['stimulus']),
                    _span(value.get('trs')))
    if isinstance(value, (str, int)):
        return Item(None, str(value), None)
    value = tuple(value)
    if len(value) == 2 and (value[1] is None or isinstance(value[1], (tuple, list, range))):
        return Item(None, str(value[0]), _span(value[1]))
    if len(value) == 2:
        return Item(str(value[0]), str(value[1]), None)
    if len(value) == 3:
        return Item(None if value[0] is None else str(value[0]), str(value[1]), _span(value[2]))
    raise ValueError(f'Cannot read item {value!r}')


def _item_dict(item):
    return dict(subject=item.subject, stimulus=item.stimulus, trs=None if item.trs is None else list(item.trs))


def _spans(items, stimulus, subject=None):
    """TR spans of ``items`` at ``stimulus``; with ``subject``, items naming another subject are skipped."""
    return [i.trs for i in items if i.stimulus == stimulus and (subject is None or i.subject in (None, subject))]


def _covered(spans, trs):
    if any(s is None for s in spans):
        return bool(spans)
    if trs is None:
        return False
    position = trs[0]
    for start, stop in sorted(spans):
        if start > position:
            break
        position = max(position, stop)
    return position >= trs[1]


def _overlaps(spans, trs):
    return any(s is None or trs is None or (s[0] < trs[1] and trs[0] < s[1]) for s in spans)


def _describe(read):
    where = '' if read.trs is None else f'[{read.trs[0]}:{read.trs[1]}]'
    fold = '' if read.fold is None else f' (inner fold {read.fold})'
    return f'{read.subject} {read.kind} {read.stimulus}{where} for {read.purpose}{fold}'


class Provenance:
    """Read log of one cell, checked against rules R1-R7.

    ``target`` is the target subject; ``regime`` is 'S' (scan-matched, primary) or 'L';
    ``budget``, ``val_items`` and ``test_items`` are the target's material. Optional scope:
    ``ts_allowed`` lists the true-strict stimuli for reference transcripts (rule 'TS'); otherwise
    the rule is 'none' unless ``reference_rule='SS'`` is passed. TS and SS need a secondary ``arm``.
    ``require_context_reset`` defaults to (and in LPP must be) on for LPP targets. ``score_items``
    defaults by ``stage`` ('g1' | 'develop' | 'evaluate'): nothing in g1, else the outer test
    section in LPP; in ds003020 the validation stories, plus the test stories at 'evaluate'.
    ``folds`` maps inner decoder fold ids (or list positions) to their held-out budget items; they
    must partition the budget (each item held out once, at least two non-empty folds).
    ``arm`` ties the log to one arm; only 'ORACLE' may make ``oracle`` reads. ``budget_name``
    ('S3' ... 'L8') and ``stage`` are recorded so that ``scripts/audit_v4.py`` can recompute the
    registered scope.
    """

    def __init__(self, target, regime, budget, val_items, test_items, ts_allowed=None, reference_rule=None,
                 require_context_reset=None, score_items=None, folds=None, arm=None, budget_name=None, stage=None):
        if regime not in REGIMES:
            raise ValueError(f'regime must be one of {REGIMES}')
        if stage not in (None,) + STAGES:
            raise ValueError(f'stage must be one of {STAGES}')
        if budget_name not in (None,) + BUDGET_NAMES:
            raise ValueError(f'budget_name must be one of {BUDGET_NAMES}')
        self.target, self.regime, self.stage, self.budget_name = str(target), regime, stage, budget_name
        self.arm, self.dataset = None if arm is None else str(arm), dataset_of(target)
        self.budget, self.val_items, self.test_items = (self._own(items, name) for name, items in
                                                        (('budget', budget), ('val', val_items), ('test', test_items)))
        self.held = self.val_items + self.test_items
        clash = [i for i in self.budget if _overlaps(_spans(self.held, i.stimulus), i.trs)]
        if clash:
            raise ValueError(f'Budget overlaps validation/test material: {clash}')
        if score_items is None:
            score_items = ([] if stage == 'g1' else self.test_items if self.dataset == 'lpp'
                           else self.val_items + (self.test_items if stage == 'evaluate' else []))
        self.score_items = self._own(score_items, 'score')
        if any(not _covered(_spans(self.held, i.stimulus), i.trs) for i in self.score_items):
            raise ValueError(f'Scored items must be validation/test material: {self.score_items}')
        if stage == 'g1' and self.score_items:
            raise ValueError('Stage g1 scores nothing')
        if (stage != 'evaluate' and self.dataset == 'ds003020'
                and any(_overlaps(_spans(self.test_items, i.stimulus), i.trs) for i in self.score_items)):
            raise ValueError('ds003020 test stories are scored only at stage evaluate (test needs the lock)')
        self.folds, self.current_fold = None, None
        if folds is not None:
            pairs = list(folds.items() if isinstance(folds, dict) else enumerate(folds))
            self.folds = {str(k): self._own([held] if isinstance(held, (str, int, dict, Item)) else held,
                                            f'fold {k}') for k, held in pairs}
            if len(self.folds) != len(pairs) or len(self.folds) < 2 or not all(self.folds.values()):
                raise ValueError(f'Inner CV needs at least two distinct, non-empty folds: {folds!r}')
            outside = [i for held in self.folds.values() for i in held
                       if not _covered(_spans(self.budget, i.stimulus), i.trs)]
            if outside:
                raise ValueError(f'Inner folds hold out budget material only: {outside}')
            twice = [i for k, held in self.folds.items() for i in held
                     if _overlaps(_spans([j for k2, h in self.folds.items() if k2 != k for j in h], i.stimulus), i.trs)]
            every = [i for held in self.folds.values() for i in held]
            never = [i for i in self.budget if not _covered(_spans(every, i.stimulus), i.trs)]
            if twice or never:
                raise ValueError(f'Inner folds must partition the budget: held twice {twice}, never held {never}')
        self.ts_allowed = None if ts_allowed is None else [_item(i) for i in ts_allowed]
        heard = [i for i in self.ts_allowed or () if _overlaps(_spans(self.budget + self.held, i.stimulus), i.trs)]
        if heard:
            raise ValueError(f'TS stimuli must be unheard by the target, not its budget/validation/test: {heard}')
        rule = reference_rule or ('TS' if ts_allowed is not None else 'none')
        if rule not in REFERENCE_RULES or (rule == 'TS') != (ts_allowed is not None):
            raise ValueError(f'reference_rule {rule!r}: one of {REFERENCE_RULES}, with ts_allowed exactly under TS')
        if rule != 'none' and self.arm not in SECONDARY_ARMS:
            raise ValueError(f'R4: reference_rule {rule} is for the secondary arms {SECONDARY_ARMS} only, '
                             f'not arm {self.arm}')
        if require_context_reset is None:
            require_context_reset = self.dataset == 'lpp'
        if self.dataset == 'lpp' and not require_context_reset:
            raise ValueError('R6: LPP cells always require per-section context resets')
        self.reference_rule, self.require_context_reset = rule, bool(require_context_reset)
        self.records = {}

    def _own(self, items, name):
        items = [_item(i) for i in items]
        foreign = [i for i in items if i.subject not in (None, self.target)]
        if foreign:
            raise ValueError(f'{name} items must be the target {self.target} material: {foreign}')
        return items

    def record(self, subject, stimulus, kind, purpose, trs=None, context=None, n=1, fold=None):
        """Log one read (``n`` identical reads). ``context`` names the stimuli whose words fed a GPT-2
        causal context (R6); the target builder logs ``kind='transcript', purpose='context_reset'``.
        ``fold`` tags a read made inside inner decoder fold ``fold`` (R7); inside ``inner_fold`` it defaults
        to that block's fold."""
        if kind not in KINDS:
            raise ValueError(f'kind must be one of {KINDS}')
        if purpose not in PURPOSES:
            raise ValueError(f'purpose must be one of {PURPOSES}')
        if (purpose == 'context_reset' or context is not None) and kind != 'transcript':
            raise ValueError('Context records are transcript reads')
        fold = self.current_fold if fold is None else self._fold(fold)
        read = Read(str(subject), str(stimulus), kind, purpose, _span(trs),
                    None if context is None else tuple(sorted({str(c) for c in context})), fold)
        self.records[read] = self.records.get(read, 0) + int(n)
        return self

    def _fold(self, fold):
        if self.folds is None or str(fold) not in self.folds:
            raise ValueError(f'Unknown inner fold {fold!r}')
        return str(fold)

    @contextmanager
    def inner_fold(self, fold):
        """Tag every read recorded in the block with inner decoder fold ``fold`` (R7)."""
        previous, self.current_fold = self.current_fold, self._fold(fold)
        try:
            yield self
        finally:
            self.current_fold = previous

    def audit(self):
        """{'violations': [...], 'flags': [...]}, one entry per (record, rule), without raising."""
        violations, flags = [], []
        resets = {(r.subject, r.stimulus) for r in self.records if r.purpose == 'context_reset'}
        for index, read in enumerate(self.records):
            subject, stimulus, kind, purpose, trs, context, fold = read
            note = lambda out, rule, reason: out.append(dict(rule=rule, index=index, read=_describe(read),
                                                             reason=reason))
            in_budget = _covered(_spans(self.budget, stimulus), trs)
            held = _overlaps(_spans(self.held, stimulus), trs)
            is_target = subject == self.target
            if held and fold is not None:
                note(violations, 'R5', f'validation/test material inside inner fold {fold} (selection)')
            elif held and purpose not in SCORING:
                note(violations, 'R5', f'target validation/test material feeds {purpose}')
            elif held and purpose != 'context_reset' and not _covered(_spans(self.score_items, stimulus), trs):
                note(violations, 'R5', f'validation/test material this {self.stage or "cell"} does not score')
            if purpose == 'context_reset':
                pass
            elif is_target and purpose == 'oracle':
                rule = 'R1' if kind != 'brain' else 'R2'
                if self.arm == 'ORACLE':
                    note(flags, rule, 'oracle read: over budget by design, never confirmatory')
                else:
                    note(violations, rule, f'oracle read outside an ORACLE cell (arm {self.arm})')
            elif is_target and kind != 'brain':
                if purpose in TRANSCRIPT_FREE:
                    note(violations, 'R1', f'target {kind} feeds the transcript-free {purpose}')
                elif not in_budget and not (held and purpose in SCORING):
                    note(violations, 'R1', f'target {kind} outside the budget')
            elif is_target:
                if in_budget or (held and purpose == 'decoder_score'):
                    pass
                elif self.regime == 'L' and purpose == 'isp_response' and not held:
                    note(flags, 'R2', 'regime L: unlabeled target BOLD outside the budget (not scan-matched)')
                else:
                    note(violations, 'R2', 'target BOLD outside the budget')
            elif kind == 'brain':
                if purpose in ('isp_predictor', 'converter'):
                    if in_budget:
                        pass
                    elif self.regime == 'L' and purpose == 'isp_predictor' and not held:
                        note(flags, 'R3', 'regime L: reference BOLD outside the budget (not scan-matched)')
                    else:
                        note(violations, 'R3', 'reference BOLD outside the budget stimuli and times')
                elif purpose in ('pca_basis', 'reference_model'):
                    if held:
                        note(violations, 'R3', 'reference BOLD on target validation/test material')
                    if purpose == 'reference_model' and self.arm not in SECONDARY_ARMS:
                        note(violations, 'R4', f'reference models belong to the secondary arms, not arm {self.arm}')
                else:
                    note(violations, 'R3', f'reference BOLD may not be read for {purpose}')
            elif purpose != 'reference_model':
                note(violations, 'R4', f'reference {kind} may only feed a reference model')
            elif self.arm not in SECONDARY_ARMS:
                note(violations, 'R4', f'reference transcripts belong to the secondary arms, not arm {self.arm}')
            elif self.reference_rule == 'none':
                note(violations, 'R4', 'reference transcripts are not allowed in this cell (rule none)')
            elif self.reference_rule == 'TS':
                if not _covered(_spans(self.ts_allowed, stimulus, subject), trs):
                    note(violations, 'R4', 'not a true-strict (TS) stimulus')
            elif held:
                note(violations, 'R4', 'SS excludes target validation/test material')
            else:
                note(flags, 'R4', 'simulated-strict (SS) reference transcript')
            if context is not None and set(context) != {stimulus}:
                note(violations, 'R6', f'GPT-2 context drawn from {list(context)}')
            if (self.require_context_reset and is_target and kind == 'transcript'
                    and purpose not in ('nuisance_fit', 'nuisance_apply', 'context_reset')
                    and (subject, stimulus) not in resets):
                note(violations, 'R6', 'no context reset recorded at the start of this section')
            if fold is None:
                continue
            out = _spans(self.folds[fold], stimulus)
            if purpose in FOLD_SCORE:
                if not _covered(out, trs):
                    note(violations, 'R7', f'inner fold {fold} scores outside its held-out part')
            elif ((purpose in FOLD_FIT or (is_target and purpose not in ('context_reset', 'oracle')))
                  and _overlaps(out, trs)):
                note(violations, 'R7', f'{purpose} on the held-out part of inner fold {fold}')
        return dict(violations=violations, flags=flags)

    def check(self):
        """Raise ProvenanceError naming every broken rule; otherwise return the flags."""
        result = self.audit()
        if result['violations']:
            raise ProvenanceError(result['violations'])
        return result['flags']

    def uncovered(self, kind, purpose, items, fold=ANY_FOLD):
        """Items (target material) not covered by the target's reads of ``kind`` for ``purpose``; ``fold``
        keeps only the reads tagged with that inner fold (None: untagged reads, e.g. the final fit)."""
        spans, fold = {}, fold if fold in (ANY_FOLD, None) else str(fold)
        for r in self.records:
            if (r.subject == self.target and r.kind == kind and r.purpose == purpose
                    and (fold is ANY_FOLD or r.fold == fold)):
                spans.setdefault(r.stimulus, []).append(r.trs)
        return [i for i in map(_item, items) if not _covered(spans.get(i.stimulus, []), i.trs)]

    def scope(self):
        """Constructor arguments as JSON-ready values (``Provenance(**p.scope())`` rebuilds an empty log)."""
        items = lambda values: None if values is None else [_item_dict(i) for i in values]
        return dict(target=self.target, regime=self.regime, budget=items(self.budget),
                    val_items=items(self.val_items), test_items=items(self.test_items),
                    ts_allowed=items(self.ts_allowed), reference_rule=self.reference_rule,
                    require_context_reset=self.require_context_reset, score_items=items(self.score_items),
                    folds=None if self.folds is None else {k: items(v) for k, v in self.folds.items()},
                    arm=self.arm, budget_name=self.budget_name, stage=self.stage)

    def merged(self, *others, **changes):
        """A new log with this scope (``changes`` override arguments, e.g. ``arm='ORACLE'``) holding the reads
        of this log and ``others``, which must share target, regime, material and folds."""
        args = dict(self.scope(), **changes)
        if 'ts_allowed' in changes and 'reference_rule' not in changes:
            args['reference_rule'] = None  # Inferred: TS with a list, else 'none'.
        out = Provenance(**args)
        fixed = ('target', 'regime', 'budget', 'val_items', 'test_items', 'folds')
        for log in (self,) + others:
            if any(log.scope()[k] != out.scope()[k] for k in fixed):
                raise ValueError('Merged logs must share target, regime, material and folds')
            for read, n in log.records.items():
                out.records[read] = out.records.get(read, 0) + n
        return out

    def to_dict(self):
        """JSON-ready log for the cell, with the audit result at write time."""
        result = self.audit()
        return dict(schema=SCHEMA, **self.scope(),
                    records=[dict(subject=r.subject, stimulus=r.stimulus, kind=r.kind, purpose=r.purpose,
                                  trs=None if r.trs is None else list(r.trs),
                                  context=None if r.context is None else list(r.context), fold=r.fold, n=n)
                             for r, n in self.records.items()],
                    violations=result['violations'], flags=result['flags'])

    @classmethod
    def from_dict(cls, log):
        """Rebuild a saved log (``to_dict``) so that ``check`` can be re-run on it."""
        if log.get('schema') != SCHEMA:
            raise ValueError(f'Not a {SCHEMA} log')
        out = cls(log['target'], log['regime'], log['budget'], log['val_items'], log['test_items'],
                  ts_allowed=log['ts_allowed'], reference_rule=log['reference_rule'],
                  require_context_reset=log['require_context_reset'], score_items=log.get('score_items'),
                  folds=log.get('folds'), arm=log.get('arm'), budget_name=log.get('budget_name'),
                  stage=log.get('stage'))
        for r in log['records']:
            out.record(r['subject'], r['stimulus'], r['kind'], r['purpose'], r['trs'], r['context'], r.get('n', 1),
                       r.get('fold'))
        return out
