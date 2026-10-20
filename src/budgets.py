"""v4 budgets, registered hash orders, folds and the per-cell ledger (PROTOCOL_V4 §2; FUTURE_WORK_V4 §3-§4).

Every order is ascending hexadecimal SHA-256 of a registered string. The order of a
subset is therefore the restriction of the full order: the ds003020 budgets
S3 ⊂ S6 ⊂ S15 ⊂ ALL are nested, ``isp_stories(ALL, shared)`` equals S15 exactly, and
the LPP section budgets L3 ⊂ L5 ⊂ L8 are prefixes of each outer fold's non-test
sections. Stimuli and sections enter the hashed strings through ``str``.
"""
from __future__ import annotations
import hashlib
import math
from numbers import Integral, Real
from .transcripts import consumed_words

SUBJECTS = tuple(f'UTS{i:02d}' for i in range(1, 11))
BUDGET_PREFIX = 'ssltang-v4-budget:'
LPP_SUBJECT_PREFIX = 'ssltang-v4-lpp-subject:'
LPP_SECTION_PREFIX = 'ssltang-v4-lpp-section:'
DS003020_BUDGETS = ('S3', 'S6', 'S15', 'ALL')
PREFIX_SIZES = {'S3': 3, 'S6': 6}
LPP_SIZES = {'L3': 3, 'L5': 5, 'L8': 8}
ROUND_ROBIN = {'S6': 3, 'S15': 5}  # Inner decoder folds, round-robin over the budget's hash order.
LEAVE_ONE_OUT = ('S3',) + tuple(LPP_SIZES)
LPP_DEV = 9
LPP_SECTIONS = tuple(str(k) for k in range(1, 10))  # Section ids '1'..'9' (the LPP manifest story/session).
TR_SECONDS = 2.0
REFERENCE_RULES = ('none', 'TS', 'SS')


def sha256_hex(text):
    return hashlib.sha256(str(text).encode()).hexdigest()


def dataset_of(subject):
    """'ds003020' for UTS01-UTS10; any other subject is an LPP listener (e.g. 'sub-EN057')."""
    return 'ds003020' if str(subject) in SUBJECTS else 'lpp'


def hash_order(items, prefix):
    """``items`` in ascending hexadecimal SHA-256 of ``prefix + str(item)`` (PROTOCOL_V4 §2)."""
    items = list(items)
    if len({str(i) for i in items}) != len(items):
        raise ValueError('hash_order needs distinct items')
    return sorted(items, key=lambda item: sha256_hex(f'{prefix}{item}'))


def split_stories(manifest):
    """{split: sorted stories} of one subject manifest; every story must belong to exactly one split."""
    splits = {}
    for run in manifest['runs']:
        splits.setdefault(run['story'], set()).add(run['split'])
    mixed = sorted(story for story, s in splits.items() if len(s) > 1)
    if mixed:
        raise ValueError(f'Stories in more than one split: {mixed}')
    out = {}
    for story, (split,) in sorted(splits.items()):
        out.setdefault(split, []).append(story)
    return out


def shared_stories(manifests):
    """Sorted stories present in every manifest of ``manifests`` (subject -> manifest)."""
    if not manifests:
        raise ValueError('No manifests')
    return sorted(set.intersection(*({r['story'] for r in m['runs']} for m in manifests.values())))


def ts_stories(target_manifest, reference_manifest):
    """True-strict (TS) stories: in the reference's manifest and never heard by the target."""
    heard = {r['story'] for r in target_manifest['runs']}
    return sorted({r['story'] for r in reference_manifest['runs']} - heard)


def ds003020_budgets(target_manifest, shared):
    """{'S3', 'S6', 'S15', 'ALL'}: lists of the target's training stories in budget hash order.

    S3 and S6 are prefixes of the hash order of ``train_shared`` (the target's
    training-split stories within ``shared``), S15 is all of it and ALL is every
    training-split story in the same hash order.
    """
    splits = split_stories(target_manifest)
    missing = sorted(set(shared) - {s for stories in splits.values() for s in stories})
    if missing:
        raise ValueError(f'Shared stories absent from the target manifest: {missing}')
    shared, train = set(shared), splits.get('train', [])
    train_shared = hash_order([s for s in train if s in shared], BUDGET_PREFIX)
    if len(train_shared) < max(PREFIX_SIZES.values()):
        raise ValueError(f'Only {len(train_shared)} training shared stories')
    out = {name: train_shared[:size] for name, size in PREFIX_SIZES.items()}
    out.update(S15=train_shared, ALL=hash_order(train, BUDGET_PREFIX))
    return out


def isp_stories(budget_stories, shared):
    """Budget stories heard by every subject, in budget order (ISP at ALL uses the S15 stories)."""
    shared = set(shared)
    return [s for s in budget_stories if s in shared]


def inner_folds(budget_name, stories):
    """Held-out story (or section) tuples of the inner decoder CV over B, in the given (hash) order.

    S3 and the LPP budgets L3/L5/L8 are leave-one-out; S6 (3 folds) and S15 (5 folds)
    are round-robin. ALL uses the v3 fixed-fold rule (``all_folds``).
    """
    stories = list(stories)
    if len(set(stories)) != len(stories):
        raise ValueError('Duplicate stories in a budget')
    if budget_name in LEAVE_ONE_OUT:
        if len(stories) < 2:
            raise ValueError(f'{budget_name}: leave-one-out needs at least two stories')
        return [(s,) for s in stories]
    if budget_name in ROUND_ROBIN:
        k = ROUND_ROBIN[budget_name]
        if len(stories) < k:
            raise ValueError(f'{budget_name}: {k} folds need at least {k} stories')
        return [tuple(stories[f::k]) for f in range(k)]
    if budget_name == 'ALL':
        raise ValueError('ALL uses the v3 fixed-fold rule (all_folds), not inner_folds')
    raise ValueError(f'Unknown budget {budget_name!r}')


def all_folds(target_manifest):
    """Inner decoder folds at ALL: sorted story tuples of the v3 session folds over the training runs, or None.

    ``src.structured_ridge.budget_folds`` over the fixed five-fold session partition (leave one
    session out below five sessions; None when fewer than two folds), as ``run_v4`` derives them.
    """
    from .structured_ridge import budget_folds, fixed_folds
    runs = [r for r in target_manifest['runs'] if r['split'] == 'train']
    sessions = [r.get('session') for r in runs]
    folds, _ = budget_folds(sessions, range(len(runs)), fixed_folds(sessions))
    if folds is None:
        return None
    return [tuple(sorted({r['story'] for r, s in zip(runs, sessions) if s in set(f)})) for f in folds]


def half_groups(stories_of_rows, trs_of_rows, n_tr_by_story):
    """Half-story group per row: ``f'{story}:{h}'`` with h = 1 iff ``tr >= n_tr // 2``."""
    if len(stories_of_rows) != len(trs_of_rows):
        raise ValueError('One TR per row is required')
    out = []
    for story, tr in zip(stories_of_rows, trs_of_rows):
        n, tr = int(n_tr_by_story[story]), int(tr)
        if n < 2 or not 0 <= tr < n:
            raise ValueError(f'TR {tr} outside story {story} of {n} TRs')
        out.append(f'{story}:{int(tr >= n // 2)}')
    return out


def half_folds(stories):
    """Leave-one-half-out folds over the half-story groups of ``stories`` (two per story)."""
    return [(f'{s}:{h}',) for s in stories for h in (0, 1)]


def lpp_partition(participant_ids):
    """(DEV, CONFIRM): the first 9 and the rest of the ids in ``ssltang-v4-lpp-subject:<id>`` order.

    Ids are hashed exactly as given, so all must share one form (all with or all
    without the BIDS ``sub-`` prefix).
    """
    ids = [str(i) for i in participant_ids]
    if len(ids) <= LPP_DEV:
        raise ValueError(f'Need more than {LPP_DEV} participants')
    if len({i.startswith('sub-') for i in ids}) > 1:
        raise ValueError('Participant ids mix the sub- prefixed and bare forms')
    order = hash_order(ids, LPP_SUBJECT_PREFIX)
    return order[:LPP_DEV], order[LPP_DEV:]


def lpp_section_budgets(test_section, sections):
    """{'L3', 'L5', 'L8'}: prefixes of the non-test sections in ``ssltang-v4-lpp-section:<t>:<s>`` order."""
    sections = list(sections)
    if test_section not in sections:
        raise ValueError(f'Test section {test_section!r} not among {sections}')
    others = [s for s in sections if s != test_section]
    if len(others) < max(LPP_SIZES.values()):
        raise ValueError(f'Only {len(others)} non-test sections')
    order = hash_order(others, f'{LPP_SECTION_PREFIX}{test_section}:')
    return {name: order[:size] for name, size in LPP_SIZES.items()}


def bold_minutes(n_trs, tr_seconds=TR_SECONDS):
    """Minutes of BOLD in runs of ``n_trs`` TRs."""
    return float(sum(int(n) for n in n_trs)) * float(tr_seconds) / 60.0


class Ledger(dict):
    """Per-cell label and exposure ledger (FUTURE_WORK_V4 §3), a JSON-ready dict serialized into every cell.

    Extends the v3 exposure counter: target labeled pairs on the dense and stride-4
    clocks, target BOLD minutes, unique transcript word instances (each in-bin word
    plus up to 20 preceding words of its run, as ``src.transcripts.consumed_words``)
    and their GPT-2 tokens, and the references with their BOLD minutes, labeled pairs
    and transcript rule ('none' | 'TS' | 'SS').
    """
    FIELDS = (('target_pairs_dense', 0), ('target_pairs_stride4', 0), ('target_bold_minutes', 0.0),
              ('transcript_words', 0), ('transcript_tokens', None), ('references', ()),
              ('reference_bold_minutes', 0.0), ('reference_labeled_pairs', 0), ('reference_rule', 'none'))
    COUNTS = ('target_pairs_dense', 'target_pairs_stride4', 'transcript_words', 'reference_labeled_pairs')
    MINUTES = ('target_bold_minutes', 'reference_bold_minutes')

    def __init__(self, **values):
        unknown = sorted(set(values) - {k for k, _ in self.FIELDS})
        if unknown:
            raise ValueError(f'Unknown ledger fields {unknown}')
        super().__init__((k, values.get(k, default)) for k, default in self.FIELDS)
        self.validate()

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def validate(self):
        """Coerce to plain JSON types; raise ValueError on any inconsistent field."""
        def count(name, value):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer, got {value!r}')
            return int(value)
        for k in self.COUNTS:
            self[k] = count(k, self[k])
        for k in self.MINUTES:
            v = self[k]
            if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v < 0:
                raise ValueError(f'{k} must be finite and nonnegative, got {v!r}')
            self[k] = float(v)
        if self['transcript_tokens'] is not None:
            self['transcript_tokens'] = count('transcript_tokens', self['transcript_tokens'])
        refs = self['references']
        if isinstance(refs, str) or any(not isinstance(r, str) for r in refs) or len(set(refs)) != len(refs):
            raise ValueError('references must be distinct subject ids')
        self['references'] = list(refs)
        rule = self['reference_rule']
        if rule not in REFERENCE_RULES:
            raise ValueError(f'reference_rule must be one of {REFERENCE_RULES}')
        if rule == 'none' and self['reference_labeled_pairs']:
            raise ValueError('Reference labeled pairs require the TS or SS rule')
        if not refs and (self['reference_bold_minutes'] or self['reference_labeled_pairs'] or rule != 'none'):
            raise ValueError('Reference exposure without references')
        return self

    def target(self, pairs_dense, pairs_stride4, minutes):
        """Target labeled pairs on both clocks and the target BOLD minutes read (budget B)."""
        self.update(target_pairs_dense=pairs_dense, target_pairs_stride4=pairs_stride4, target_bold_minutes=minutes)
        return self.validate()

    def transcript(self, words_by_run, targets, tokenizer=None):
        """Unique word instances (and GPT-2 tokens, given a tokenizer) consumed by (run_id, TR) targets.

        ``words_by_run`` maps run id -> ordered word rows with ``word`` and ``bin_index``
        (``src.transcripts.run_words``). Tokens follow the v3 convention: the sum over
        unique instances of ``len(tokenizer.encode(' ' + word, add_special_tokens=False))``.
        """
        consumed = consumed_words(words_by_run, targets)['all']
        self['transcript_words'] = len(consumed)
        self['transcript_tokens'] = None if tokenizer is None else int(sum(
            len(tokenizer.encode(' ' + words_by_run[run][i]['word'], add_special_tokens=False))
            for run, i in consumed))
        return self.validate()

    def reference(self, references, minutes, labeled_pairs=0, rule='none'):
        """References (sorted), their BOLD minutes, labeled pairs and transcript rule."""
        self.update(references=sorted(references), reference_bold_minutes=minutes,
                    reference_labeled_pairs=labeled_pairs, reference_rule=rule)
        return self.validate()

    def to_dict(self):
        return dict(self.validate())
