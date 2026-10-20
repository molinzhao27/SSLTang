"""v4 provenance auditor (G0e): rules R1-R7, planted leaks that must raise, clean cells that must pass,
and the CLI's recomputation of each log's registered scope."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_v4  # noqa: E402
from src import budgets as bud  # noqa: E402
from src.budgets import lpp_section_budgets  # noqa: E402
from src.provenance import Provenance, ProvenanceError  # noqa: E402

TARGET, REFS = 'UTS05', ('UTS01', 'UTS02')
BUDGET, OUTSIDE, VAL, TEST = ['a', 'b', 'c'], 'd', ['v1', 'v2'], ['t1']
INNER = [('a',), ('b',), ('c',)]
SECTIONS = list(bud.LPP_SECTIONS)
LPP_TARGET, LPP_REFS, LPP_TEST = 'sub-EN057', ('sub-EN058', 'sub-EN059'), '4'


def ds_cell(stage='evaluate', **kwargs):
    """A clean regime-S ds003020 cell: every read the v4 arms make (inner folds included), inside the rules."""
    p = Provenance(TARGET, kwargs.pop('regime', 'S'), BUDGET, VAL, TEST, stage=stage, folds=INNER, **kwargs)
    for f, held in enumerate(INNER):
        for st in BUDGET:
            if st in held:
                p.record(TARGET, st, 'brain', 'decoder_score', fold=f)  # Inner-CV scoring of the held story.
                continue
            for purpose in ('standardize', 'isp_response', 'decoder_fit'):
                p.record(TARGET, st, 'brain', purpose, fold=f)
            p.record(TARGET, st, 'transcript', 'own_encoding', fold=f)
            for ref in REFS:
                p.record(ref, st, 'brain', 'isp_predictor', fold=f)
    for st in BUDGET:
        for kind in ('brain', 'transcript'):
            for purpose in ('decoder_fit', 'standardize', 'own_encoding', 'decoder_score'):
                p.record(TARGET, st, kind, purpose)
        p.record(TARGET, st, 'brain', 'isp_response')
        p.record(TARGET, st, 'audio', 'own_encoding')
        p.record(TARGET, st, 'transcript', 'nuisance_fit')
        for ref in REFS:
            p.record(ref, st, 'brain', 'isp_predictor', trs=(0, 300))
            p.record(ref, st, 'brain', 'converter')
    for st in VAL + (TEST if stage == 'evaluate' else []):
        p.record(TARGET, st, 'brain', 'decoder_score')
        p.record(TARGET, st, 'transcript', 'decoder_score')  # Scoring targets.
        p.record(TARGET, st, 'transcript', 'nuisance_apply')  # Residualized endpoint.
    for ref in REFS:
        for st in BUDGET + [OUTSIDE, 'ref_only_story']:
            p.record(ref, st, 'brain', 'pca_basis')  # R3: any non-validation/test material.
    return p


def lpp_cell(test=LPP_TEST, budget='L3', resets=True):
    """A clean LPP cell for one outer fold (leave-one-section-out inner folds, tagged), with the builder's
    per-section context resets (R6)."""
    stories = lpp_section_budgets(test, SECTIONS)[budget]
    p = Provenance(LPP_TARGET, 'S', stories, [], [test], stage='develop', budget_name=budget,
                   folds=bud.inner_folds(budget, stories))
    for st in SECTIONS:
        if resets:
            p.record(LPP_TARGET, st, 'transcript', 'context_reset')
    for k, held in p.folds.items():
        with p.inner_fold(k):
            for st in stories:
                if st in {i.stimulus for i in held}:
                    p.record(LPP_TARGET, st, 'brain', 'decoder_score')
                    continue
                for purpose in ('standardize', 'decoder_fit', 'isp_response'):
                    p.record(LPP_TARGET, st, 'brain', purpose)
                for purpose in ('decoder_fit', 'own_encoding'):
                    p.record(LPP_TARGET, st, 'transcript', purpose, context=[st])
                for ref in LPP_REFS:
                    p.record(ref, st, 'brain', 'isp_predictor')
    for st in stories:
        p.record(LPP_TARGET, st, 'brain', 'decoder_fit')
        p.record(LPP_TARGET, st, 'transcript', 'decoder_fit', context=[st])
        for ref in LPP_REFS:
            p.record(ref, st, 'brain', 'isp_predictor')
    p.record(LPP_TARGET, test, 'brain', 'decoder_score')
    p.record(LPP_TARGET, test, 'transcript', 'decoder_score', context=[test])
    for ref in LPP_REFS:
        for st in SECTIONS:
            if st != test:
                p.record(ref, st, 'brain', 'pca_basis')
    return p


def rules_of(p):
    with pytest.raises(ProvenanceError) as error:
        p.check()
    return set(error.value.rules)


def test_clean_cells_pass_and_survive_a_json_round_trip():
    for p in (ds_cell(), ds_cell(stage='develop'), lpp_cell(), lpp_cell(test='9', budget='L8')):
        assert p.check() == []
        again = Provenance.from_dict(json.loads(json.dumps(p.to_dict())))
        assert again.records == p.records and again.check() == []
        assert again.to_dict() == p.to_dict()


def test_duplicate_reads_are_stored_once_with_a_count():
    p = ds_cell()
    p.record(TARGET, 'a', 'brain', 'decoder_fit')
    log = p.to_dict()
    entry = [r for r in log['records'] if r['stimulus'] == 'a' and r['purpose'] == 'decoder_fit'
             and r['kind'] == 'brain' and r['fold'] is None]
    assert len(entry) == 1 and entry[0]['n'] == 2


# ----------------------------------------------------------------------------------------- planted leaks (G0e)
def test_planted_test_section_reference_response_used_for_isp():
    p = lpp_cell()
    p.record(LPP_REFS[0], LPP_TEST, 'brain', 'isp_predictor')
    assert {'R3', 'R5'} <= rules_of(p)
    p = lpp_cell()
    p.record(LPP_REFS[0], LPP_TEST, 'brain', 'pca_basis')  # Not even for a PCA basis.
    assert {'R3', 'R5'} <= rules_of(p)


def test_planted_out_of_budget_target_transcript():
    p = ds_cell()
    p.record(TARGET, OUTSIDE, 'transcript', 'decoder_fit')
    assert rules_of(p) == {'R1'}
    p = ds_cell()
    p.record(TARGET, OUTSIDE, 'audio', 'own_encoding')
    assert rules_of(p) == {'R1'}


def test_planted_reference_model_containing_a_target_test_story():
    p = ds_cell(reference_rule='SS', arm='TH_SEL')
    p.record(REFS[0], TEST[0], 'transcript', 'reference_model')
    assert {'R4', 'R5'} <= rules_of(p)
    p = ds_cell()
    p.record(REFS[0], TEST[0], 'brain', 'reference_model')
    assert {'R3', 'R5'} <= rules_of(p)


def test_planted_cross_section_gpt2_context():
    p = lpp_cell()
    st = lpp_section_budgets(LPP_TEST, SECTIONS)['L3'][1]
    previous = str(int(st) - 1) if st != '1' else '2'
    p.record(LPP_TARGET, st, 'transcript', 'decoder_fit', context=[previous, st])
    assert rules_of(p) == {'R6'}
    p = lpp_cell(resets=False)  # Book-level context: no reset at section starts.
    assert rules_of(p) == {'R6'}
    p = lpp_cell(resets=False)
    p.record(LPP_TARGET, SECTIONS[0], 'transcript', 'context_reset')  # Only the book start was reset.
    with pytest.raises(ProvenanceError, match='R6'):
        p.check()


def test_r6_is_always_on_for_lpp_targets():
    p = Provenance(LPP_TARGET, 'S', ['5', '7', '2'], [], ['4'])  # No flag: LPP turns it on.
    assert p.require_context_reset
    p.record(LPP_TARGET, '5', 'transcript', 'decoder_fit')
    assert rules_of(p) == {'R6'}
    with pytest.raises(ValueError, match='R6'):
        Provenance(LPP_TARGET, 'S', ['5', '7', '2'], [], ['4'], require_context_reset=False)
    log = lpp_cell().to_dict()
    log['require_context_reset'] = False  # A saved LPP log may not switch R6 off either.
    with pytest.raises(ValueError, match='R6'):
        Provenance.from_dict(log)
    assert not Provenance(TARGET, 'S', BUDGET, VAL, TEST).require_context_reset  # ds003020: per-story context.


# ------------------------------------------------------------------------------------------------ each rule
def test_r1_r2_oracle_reads_only_in_an_oracle_cell():
    p = ds_cell(arm='ORACLE')
    p.record(TARGET, OUTSIDE, 'transcript', 'oracle')
    p.record(TARGET, OUTSIDE, 'brain', 'oracle')
    flags = p.check()
    assert {f['rule'] for f in flags} == {'R1', 'R2'} and all('oracle' in f['reason'] for f in flags)
    assert p.to_dict()['flags'] == flags
    p.record(TARGET, TEST[0], 'transcript', 'oracle')
    assert rules_of(p) == {'R5'}
    for arm in (None, 'EB', 'W1'):  # Shared or other-arm logs: every oracle read is a violation.
        p = ds_cell(arm=arm)
        p.record(TARGET, OUTSIDE, 'transcript', 'oracle')
        p.record(TARGET, 'a', 'brain', 'oracle')
        assert rules_of(p) == {'R1', 'R2'}


def test_r1_target_transcripts_never_feed_transcript_free_priors():
    for purpose in ('isp_response', 'isp_predictor', 'pca_basis', 'converter'):
        for kind in ('transcript', 'audio'):
            p = ds_cell()
            p.record(TARGET, 'a', kind, purpose)  # In budget, but ISP/PCA/converters are label-free.
            assert rules_of(p) == {'R1'}


def test_r2_target_bold_outside_the_budget():
    p = ds_cell()
    p.record(TARGET, OUTSIDE, 'brain', 'own_encoding')
    assert rules_of(p) == {'R2'}
    p = ds_cell()
    p.record(TARGET, OUTSIDE, 'brain', 'decoder_score')  # Scoring only on validation/test stimuli.
    assert rules_of(p) == {'R2'}
    p = ds_cell()
    p.record(TARGET, VAL[0], 'brain', 'isp_response')
    assert rules_of(p) == {'R2', 'R5'}
    p = ds_cell()
    p.record(TARGET, TEST[0], 'brain', 'nuisance_apply')
    assert rules_of(p) == {'R2'}


def test_regime_l_allows_unlabeled_isp_material_outside_the_budget_but_not_validation_or_test():
    p = ds_cell(regime='L')
    p.record(TARGET, OUTSIDE, 'brain', 'isp_response')
    p.record(REFS[0], OUTSIDE, 'brain', 'isp_predictor')
    assert {f['rule'] for f in p.check()} == {'R2', 'R3'}
    p.record(TARGET, OUTSIDE, 'transcript', 'own_encoding')  # Labels stay inside B.
    assert rules_of(p) == {'R1'}
    p = ds_cell(regime='L')
    p.record(TARGET, VAL[0], 'brain', 'isp_response')
    assert rules_of(p) == {'R2', 'R5'}
    p = ds_cell()
    p.record(REFS[0], OUTSIDE, 'brain', 'isp_predictor')  # Regime S: budget only.
    assert rules_of(p) == {'R3'}


def test_r3_reference_bold_at_budget_times_only():
    p = Provenance(TARGET, 'S', [('a', (0, 50)), ('a', (50, 100)), (TARGET, 'b')], VAL, TEST)
    p.record(REFS[0], 'a', 'brain', 'isp_predictor', trs=(20, 80))
    p.record(REFS[0], 'a', 'brain', 'converter', trs=range(0, 100))
    p.record(REFS[0], 'b', 'brain', 'isp_predictor')
    assert p.check() == []
    p.record(REFS[0], 'a', 'brain', 'isp_predictor', trs=(80, 120))
    assert rules_of(p) == {'R3'}
    p = Provenance(TARGET, 'S', [('a', (0, 100))], VAL, TEST)
    p.record(REFS[0], 'a', 'brain', 'isp_predictor')  # Whole story read; the budget covers only TRs 0-99.
    assert rules_of(p) == {'R3'}
    p = ds_cell()
    p.record(REFS[0], VAL[1], 'brain', 'pca_basis')
    assert rules_of(p) == {'R3', 'R5'}
    p = ds_cell()
    p.record(REFS[0], 'a', 'brain', 'decoder_fit')
    assert rules_of(p) == {'R3'}


def test_r4_reference_transcripts_are_forbidden_by_default():
    p = Provenance(TARGET, 'S', BUDGET, VAL, ['t1'])  # Built as run_v4 builds every Tier-1 cell.
    assert p.reference_rule == 'none'
    p.record('UTS01', 'zz_ref_train', 'transcript', 'reference_model')
    p.record('UTS01', 'a', 'transcript', 'reference_model')
    assert rules_of(p) == {'R4'} and p.to_dict()['reference_rule'] == 'none'


def test_r4_reference_transcripts_under_ts_ss_and_none():
    p = ds_cell(ts_allowed=['ts1', ('UTS01', 'ts2')], arm='TH_SEL')
    p.record('UTS01', 'ts1', 'transcript', 'reference_model')
    p.record('UTS01', 'ts2', 'transcript', 'reference_model')
    p.record('UTS01', 'ts2', 'brain', 'reference_model')
    assert p.check() == []
    p.record('UTS02', 'ts2', 'transcript', 'reference_model')  # ts2 is TS for UTS01 only.
    assert rules_of(p) == {'R4'}
    p = ds_cell(ts_allowed=['ts1'], arm='XDEC')
    p.record('UTS01', OUTSIDE, 'transcript', 'reference_model')  # Not on the TS list.
    assert rules_of(p) == {'R4'}
    p = ds_cell(reference_rule='SS', arm='TH_SEL')  # SS: any material except validation/test, flagged.
    p.record('UTS01', 'a', 'transcript', 'reference_model')
    p.record('UTS01', OUTSIDE, 'transcript', 'reference_model')
    flags = p.check()
    assert [f['rule'] for f in flags] == ['R4', 'R4'] and all('SS' in f['reason'] for f in flags)
    p = ds_cell(reference_rule='none', arm='TH_SEL')
    p.record('UTS01', OUTSIDE, 'transcript', 'reference_model')
    assert rules_of(p) == {'R4'}
    p = ds_cell(reference_rule='SS', arm='TH_SEL')
    p.record('UTS01', 'a', 'transcript', 'isp_predictor')
    assert rules_of(p) == {'R4'}


def test_r4_reference_transcripts_and_models_belong_to_the_secondary_arms():
    for arm in (None, 'W0', 'W1', 'PRIOR', 'EB', 'EB_SELF', 'P_AC', 'ORACLE', 'TH-SEL'):
        for kwargs in (dict(reference_rule='SS'), dict(ts_allowed=['ts1'])):
            with pytest.raises(ValueError, match='R4'):  # A Tier-1 or shared log never permits them.
                Provenance(TARGET, 'S', BUDGET, VAL, TEST, arm=arm, **kwargs)
    shared = ds_cell()
    secondary = shared.merged(arm='TH_SEL', ts_allowed=['ts1'])  # Each secondary arm gets its own log.
    assert secondary.reference_rule == 'TS' and shared.merged(arm='WEN', reference_rule='SS').check() == []
    with pytest.raises(ValueError, match='R4'):
        secondary.merged(arm='EB')  # An EB cell cannot inherit the secondary scope.
    log = secondary.to_dict()
    log['arm'] = 'EB'
    with pytest.raises(ValueError, match='R4'):
        Provenance.from_dict(log)
    for arm in (None, 'EB', 'W0'):  # Reference models (even BOLD-only) are secondary-arm material.
        p = ds_cell(arm=arm)
        p.record(REFS[0], OUTSIDE, 'brain', 'reference_model')
        assert rules_of(p) == {'R4'}
    p = ds_cell(arm='XDEC')
    p.record(REFS[0], OUTSIDE, 'brain', 'reference_model')
    assert p.check() == []


def test_ts_stimuli_must_be_unheard_by_the_target():
    for heard in ('a', 'v1', 't1', ('UTS01', 'a')):  # Budget, validation or test material is not TS.
        with pytest.raises(ValueError, match='TS'):
            Provenance(TARGET, 'S', BUDGET, VAL, TEST, ts_allowed=['x', heard])


def test_r5_scoring_is_limited_to_the_stage_scored_items():
    p = ds_cell(stage='develop')  # Development scores validation stories only.
    p.record(TARGET, TEST[0], 'brain', 'decoder_score')
    p.record(TARGET, TEST[0], 'transcript', 'decoder_score')
    assert rules_of(p) == {'R5'}
    p = Provenance(TARGET, 'S', BUDGET, VAL, TEST)  # No stage: fail closed, as development.
    assert [i.stimulus for i in p.score_items] == VAL
    p.record(TARGET, TEST[0], 'brain', 'decoder_score')
    assert rules_of(p) == {'R5'}
    p = Provenance(TARGET, 'S', BUDGET, VAL, TEST, stage='g1')
    p.record(TARGET, VAL[0], 'brain', 'decoder_score')
    assert rules_of(p) == {'R5'}
    for stage in ('develop', None):  # Explicit test scoring needs stage evaluate, also when no stage is given.
        with pytest.raises(ValueError, match='only at stage evaluate'):
            Provenance(TARGET, 'S', BUDGET, VAL, TEST, stage=stage, score_items=VAL + TEST)
    assert Provenance(LPP_TARGET, 'S', ['5', '7', '2'], [], ['4'], score_items=['4']).score_items  # LPP: outer.
    with pytest.raises(ValueError, match='g1'):
        Provenance(TARGET, 'S', BUDGET, VAL, TEST, stage='g1', score_items=VAL)
    with pytest.raises(ValueError, match='validation/test'):
        Provenance(TARGET, 'S', BUDGET, VAL, TEST, score_items=['a'])
    assert [i.stimulus for i in lpp_cell().score_items] == [LPP_TEST]  # LPP: the outer test section.


def test_r5_nuisance_models_are_fit_on_the_budget_and_only_applied_to_scored_items():
    p = ds_cell()
    p.record(TARGET, TEST[0], 'transcript', 'nuisance_fit')  # Calibration on test rows.
    assert rules_of(p) == {'R1', 'R5'}
    p = ds_cell(stage='develop')
    p.record(TARGET, TEST[0], 'transcript', 'nuisance_apply')
    assert rules_of(p) == {'R5'}
    with pytest.raises(ValueError):
        ds_cell().record(TARGET, 'a', 'transcript', 'nuisance')  # Ambiguous: fit or apply?


def test_r7_inner_fold_reads_stay_on_the_fold_training_part():
    for kind, purpose in (('brain', 'isp_response'), ('brain', 'standardize'), ('transcript', 'own_encoding'),
                          ('brain', 'decoder_fit')):
        p = ds_cell()
        p.record(TARGET, 'a', kind, purpose, fold=0)  # Fold 0 holds out story a.
        assert rules_of(p) == {'R7'}
    p = ds_cell()
    p.record(REFS[0], 'b', 'brain', 'isp_predictor', fold='1')  # Any subject at the held-out times.
    assert rules_of(p) == {'R7'}
    p = ds_cell()
    p.record(REFS[0], 'a', 'brain', 'pca_basis', fold=0)  # Reference-only PCA bases are not target-derived.
    p.record(TARGET, 'a', 'brain', 'decoder_score', fold=0)
    assert p.check() == []
    with pytest.raises(ValueError, match='fold'):
        p.record(TARGET, 'a', 'brain', 'standardize', fold=7)
    with pytest.raises(ValueError, match='fold'):
        Provenance(TARGET, 'S', BUDGET, VAL, TEST).record(TARGET, 'a', 'brain', 'standardize', fold=0)
    with pytest.raises(ValueError, match='budget material'):
        Provenance(TARGET, 'S', BUDGET, VAL, TEST, folds=[('a',), (OUTSIDE,)])
    p = Provenance(TARGET, 'S', ['a', ('b', (0, 100)), 'c'], VAL, TEST,
                   folds={'x': ['a', ('b', (50, 100))], 'y': [('b', (0, 50)), 'c']})
    p.record(TARGET, 'b', 'brain', 'isp_response', trs=(50, 100), fold='y')  # Partial folds by TR.
    assert p.check() == []
    p.record(TARGET, 'b', 'brain', 'isp_response', trs=(40, 60), fold='y')
    assert rules_of(p) == {'R7'}


def test_r7_covers_every_target_derived_read_and_inner_fold_scoring():
    for kind, purpose in (('brain', 'pca_basis'), ('transcript', 'reference_model'), ('brain', 'nuisance_fit')):
        p = ds_cell()
        p.record(TARGET, 'a', kind, purpose, fold=0)  # The SELF basis or a target model on the held story.
        assert rules_of(p) == {'R7'}, purpose
    p = ds_cell(stage='develop')
    p.record(TARGET, VAL[0], 'brain', 'decoder_score', fold=0)  # Alpha/gamma tuned on validation.
    p.record(TARGET, VAL[0], 'transcript', 'decoder_score', fold=0)
    assert rules_of(p) == {'R5', 'R7'}
    for kind, purpose in (('brain', 'decoder_score'), ('transcript', 'nuisance_apply')):
        p = ds_cell()
        p.record(TARGET, 'b', kind, purpose, fold=0)  # Fold 0 trains on b: scoring rows it was fitted on.
        assert rules_of(p) == {'R7'}, purpose
    p = ds_cell()
    p.record(TARGET, VAL[0], 'transcript', 'context_reset', fold=1)
    assert rules_of(p) == {'R5'}


def test_inner_folds_partition_the_budget_and_tag_reads_in_a_block():
    for bad in ([BUDGET], [[], [], []], [('a', 'b'), ('b', 'c')], [('a',), ('b',)], {0: 'a', '0': 'b', 1: 'c'}, []):
        with pytest.raises(ValueError, match='folds'):
            Provenance(TARGET, 'S', BUDGET, VAL, TEST, folds=bad)
    p = Provenance(TARGET, 'S', BUDGET, VAL, TEST, folds=INNER)
    with p.inner_fold(1):
        p.record(TARGET, 'a', 'brain', 'isp_response')
        p.record(TARGET, 'c', 'brain', 'isp_response', fold=2)  # An explicit tag wins.
    p.record(TARGET, 'a', 'brain', 'isp_response')
    assert {(r.stimulus, r.fold) for r in p.records} == {('a', None), ('a', '1'), ('c', '2')}
    assert p.current_fold is None and rules_of(p) == {'R7'}  # Fold 2 holds out c.
    assert p.uncovered('brain', 'isp_response', ['a', 'c'], fold=None) == [(None, 'c', None)]
    assert p.uncovered('brain', 'isp_response', ['a', 'c'], fold=1) == [(None, 'c', None)]
    assert p.uncovered('brain', 'isp_response', ['a', 'c']) == []
    with pytest.raises(ValueError, match='fold'):
        with p.inner_fold(5):
            pass


def test_error_names_every_broken_rule():
    p = ds_cell()
    p.record(TARGET, OUTSIDE, 'transcript', 'decoder_fit')
    p.record(REFS[1], TEST[0], 'brain', 'isp_predictor')
    with pytest.raises(ProvenanceError, match=r'\(R1, R3, R5\)') as error:
        p.check()
    assert len(error.value.violations) == 3
    assert p.to_dict()['violations'] == error.value.violations


def test_construction_and_record_validation():
    with pytest.raises(ValueError, match='overlaps'):
        Provenance(TARGET, 'S', BUDGET + ['t1'], VAL, TEST)
    with pytest.raises(ValueError):
        Provenance(TARGET, 'S', [('UTS01', 'a')], VAL, TEST)  # Budget items are the target's material.
    for kwargs in (dict(regime='X'), dict(ts_allowed=['x'], reference_rule='SS'), dict(reference_rule='TS'),
                   dict(stage='test'), dict(budget_name='S4')):
        with pytest.raises(ValueError):
            Provenance(TARGET, kwargs.pop('regime', 'S'), BUDGET, VAL, TEST, **kwargs)
    p = ds_cell()
    for args in ((TARGET, 'a', 'bold', 'decoder_fit'), (TARGET, 'a', 'brain', 'peeking'),
                 (TARGET, 'a', 'brain', 'context_reset'), (TARGET, 'a', 'brain', 'decoder_fit', (5, 5))):
        with pytest.raises(ValueError):
            p.record(*args)


def test_merged_logs_keep_every_read_and_take_the_new_arm():
    shared, oracle = ds_cell(), ds_cell(arm='ORACLE').merged()
    oracle.records.clear()
    oracle.record(TARGET, OUTSIDE, 'brain', 'oracle')
    with pytest.raises(ProvenanceError):
        shared.merged(oracle).check()  # The shared (arm-free) scope refuses oracle reads.
    cell = shared.merged(oracle, arm='ORACLE')
    assert cell.arm == 'ORACLE' and len(cell.check()) == 1
    assert sum(cell.records.values()) == sum(shared.records.values()) + 1
    with pytest.raises(ValueError, match='share'):
        shared.merged(lpp_cell())
    assert shared.uncovered('brain', 'decoder_fit', BUDGET + [OUTSIDE]) == [(None, OUTSIDE, None)]


# ------------------------------------------------------------------------------------------------- audit CLI
STORIES = [f'story{i:02d}' for i in range(24)]
DS_TARGET = 'UTS05'


def write_manifests(root):
    """Ten synthetic ds003020 manifests: UTS01-08 split 15/5/4 and UTS09/10 13/11/0 over 24 shared stories,
    in seven sessions (so the v3 fixed five-fold session rule applies at ALL)."""
    for s in bud.SUBJECTS:
        cut = (13, 24) if s in ('UTS09', 'UTS10') else (15, 20)
        splits = dict(train=STORIES[:cut[0]] + [f'{s}_only'], val=STORIES[cut[0]:cut[1]], test=STORIES[cut[1]:])
        runs = [dict(run_id=f'{s}-{st}', story=st, split=split, session=str(i % 7), n_tr=300)
                for split, stories in splits.items() for i, st in enumerate(stories)]
        (root / s).mkdir(parents=True)
        (root / s / 'semantic_manifest.json').write_text(json.dumps(dict(runs=runs)))
    return root


def registered(root, name='S3'):
    ms = audit_v4.manifests(str(root))
    splits = bud.split_stories(ms[DS_TARGET])
    return bud.ds003020_budgets(ms[DS_TARGET], bud.shared_stories(ms))[name], splits['val'], splits['test']


FOLD_READS = (('brain', 'standardize'), ('brain', 'decoder_fit'), ('brain', 'isp_response'),
              ('transcript', 'decoder_fit'), ('transcript', 'own_encoding'), ('audio', 'own_encoding'))


def fit_log(stories, val, test, stage='develop', reads=True, final=True, name='S3', fold_reads=FOLD_READS,
            **kwargs):
    """A run_v4-like budget log. Inside each inner fold (``name``'s registered folds unless ``folds`` is
    given; none at g1), tagged: every arm's reads on the fold's training part and the held-out scoring.
    Then, untagged: ISP over the budget, the final decoder fit and the stage's scores (``final``)."""
    if 'folds' not in kwargs and stage != 'g1':
        kwargs['folds'] = bud.inner_folds(name, stories)
    p = Provenance(DS_TARGET, kwargs.pop('regime', 'S'), stories, val, test, stage=stage, **kwargs)
    for k, held in (p.folds or {}).items() if reads else ():
        with p.inner_fold(k):
            for st in stories:
                if st in {i.stimulus for i in held}:
                    p.record(DS_TARGET, st, 'brain', 'decoder_score')
                    continue
                for kind, purpose in fold_reads:
                    p.record(DS_TARGET, st, kind, purpose)
                p.record('UTS01', st, 'brain', 'isp_predictor')
    for st in stories if reads and final else ():
        for kind in ('brain', 'transcript'):
            p.record(DS_TARGET, st, kind, 'decoder_fit')
        p.record(DS_TARGET, st, 'brain', 'isp_response')
        p.record('UTS01', st, 'brain', 'isp_predictor')
    for item in p.score_items if reads and final else ():
        p.record(DS_TARGET, item.stimulus, 'brain', 'decoder_score')
    return p


def all_folds(root):
    return bud.all_folds(audit_v4.manifests(str(root))[DS_TARGET])


def fit_cell(log, name='S3', stage='develop', arm='EB', **extra):
    """A run_v4 stage_fit-like cell around ``log``; ``extra`` adds or replaces cell fields."""
    return dict(dict(protocol='ssltang-v4', stage=stage, subject=DS_TARGET, budget=name, arm=arm,
                     ledger_record=bud.Ledger().to_dict(), provenance=log.to_dict()), **extra)


def write(path, obj):
    path.write_text(json.dumps(obj))
    return str(path)


def audit(root, *paths):
    return audit_v4.main(['--manifests', str(root), *map(str, paths)])


def test_audit_cli_passes_clean_cells_and_fails_leaks(tmp_path, capsys):
    root = write_manifests(tmp_path / 'semantic')
    stories, val, test = registered(root)
    clean = write(tmp_path / 'clean.json', fit_cell(fit_log(stories, val, test)))
    named = write(tmp_path / 'named.json', fit_cell(fit_log(stories, val, test, budget_name='S3')))
    nested = write(tmp_path / 'nested.json', dict(stage='develop', budget='L3', subject=LPP_TARGET, arm='EB', folds=[
        dict(provenance=lpp_cell().to_dict()), dict(provenance=lpp_cell(test='1').to_dict())]))
    assert audit(root, clean, named, nested) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert lines[2]['logs'] == 2 and lines[-1]['passed']
    leak = fit_log(stories, val, test, reference_rule='SS', arm='TH_SEL')
    leak.record('UTS01', test[0], 'transcript', 'reference_model')
    leaky = write(tmp_path / 'leaky.json', fit_cell(leak, arm='TH_SEL', ledger_record=None))
    assert audit(root, clean, leaky) == 1
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert lines[1]['rules'] == ['R4', 'R5'] and lines[-1]['failed'] == 1


def test_audit_cli_rechecks_rather_than_trusting_the_stored_result(tmp_path, capsys):
    root = write_manifests(tmp_path / 'semantic')
    stories, val, test = registered(root)
    outside = next(s for s in registered(root, 'ALL')[0] if s not in stories)
    log = fit_log(stories, val, test).to_dict()
    log['records'].append(dict(subject=DS_TARGET, stimulus=outside, kind='transcript', purpose='decoder_fit',
                               trs=None, context=None, n=1))  # Planted after the write-time audit.
    assert log['violations'] == []
    tampered = write(tmp_path / 'tampered.json', dict(fit_cell(Provenance(DS_TARGET, 'S', [], [], [])),
                                                      provenance=log))
    missing = write(tmp_path / 'missing.json', dict(arm='W0', val=dict(r=0.1)))
    broken = tmp_path / 'broken.json'
    broken.write_text('{not json')
    for path, reason in ((tampered, 'R1'), (missing, 'no provenance log'), (broken, 'unreadable')):
        assert audit(root, path) == 1
        assert reason in capsys.readouterr().out


def failed_reasons(root, path, capsys):
    capsys.readouterr()
    assert audit(root, path) == 1
    out = json.loads(capsys.readouterr().out.splitlines()[0])
    return ' | '.join(f"{v['rule']}: {v['reason']}" for v in out['violations'])


def test_audit_recomputes_the_registered_scope(tmp_path, capsys):
    root = write_manifests(tmp_path / 'semantic')
    stories, val, test = registered(root)
    wide, _, _ = registered(root, 'ALL')
    partial, unscored = fit_log(stories, val, test, final=False), fit_log(stories, val, test, final=False)
    for st in stories:
        partial.record(DS_TARGET, st, 'brain', 'decoder_fit')
        for kind in ('brain', 'transcript'):
            unscored.record(DS_TARGET, st, kind, 'decoder_fit')
    for st in val:
        partial.record(DS_TARGET, st, 'brain', 'decoder_score')
    cases = {
        # A cell at S3 whose log was built with the ALL budget (as stage_g1 builds its S3/S6 ISP log).
        'wide': (fit_cell(fit_log(wide, val, test)), 'is not the registered S3'),
        # No validation/test material: R5 and the held-material checks become vacuous.
        'empty_held': (fit_cell(fit_log(stories, [], [], score_items=[])), 'validation [] is not'),
        # A primary cell logged under regime L.
        'regime_l': (fit_cell(fit_log(stories, val, test, regime='L')), 'regime L'),
        # A log without reads.
        'no_reads': (fit_cell(fit_log(stories, val, test, reads=False)), 'no reads recorded'),
        # Decoder reads that do not cover the budget or the scored split.
        'partial': (fit_cell(partial), 'no untagged (final) target transcript decoder_fit read for'),
        'unscored': (fit_cell(unscored), 'no untagged (final) target brain decoder_score read for'),
        # A g1-style cell: one log, no budget name anywhere.
        'unscoped': (dict(stage='g1', subject=DS_TARGET, provenance=fit_log(wide, val, test, stage='g1').to_dict()),
                     'no budget name'),
        # Scope fields that disagree with the cell.
        'subject': (dict(fit_cell(fit_log(stories, val, test)), subject='UTS06'), 'is not the cell subject'),
        'stage': (fit_cell(fit_log(stories, val, test, stage='develop'), stage='evaluate'), 'log stage develop'),
        'budget': (fit_cell(fit_log(stories, val, test, budget_name='S3'), name='S6'), 'log budget S3'),
        'ledger': (dict(fit_cell(fit_log(stories, val, test)), ledger_record=bud.Ledger(
            references=['UTS01'], reference_rule='SS').to_dict()), 'reference rule SS'),
        # A TS list holding a training story outside the budget (heard by the target).
        'ts_heard': (fit_cell(fit_log(stories, val, test, ts_allowed=['ref_story', wide[-1]], arm='TH_SEL'),
                              arm='TH_SEL'), 'were heard'),
        # An evaluate log that scores validation only (registered: validation and test).
        'scored': (fit_cell(fit_log(stories, val, test, stage='evaluate', score_items=val), stage='evaluate'),
                   'scored'),
    }
    for name, (cell, reason) in cases.items():
        path = write(tmp_path / f'{name}.json', cell)
        assert reason in failed_reasons(root, path, capsys), name
    evaluate = write(tmp_path / 'evaluate.json', fit_cell(fit_log(stories, val, test, stage='evaluate'),
                                                          stage='evaluate', val=dict(r=0.1), test=dict(r=0.1),
                                                          test_shift_null=dict(p=0.5)))
    regime_l = write(tmp_path / 'regime_l_ok.json', fit_cell(fit_log(stories, val, test, regime='L'), regime='L'))
    ts = write(tmp_path / 'ts_ok.json', fit_cell(fit_log(stories, val, test, ts_allowed=['UTS01_only'], arm='XDEC'),
                                                 arm='XDEC', ledger_record=None))  # Only the reference heard it.
    assert audit(root, evaluate, regime_l, ts) == 0
    assert 'registered scope unavailable' in failed_reasons(tmp_path / 'nowhere', write(
        tmp_path / 'clean.json', fit_cell(fit_log(stories, val, test))), capsys)


def test_audit_ties_oracle_reads_to_the_oracle_arm(tmp_path, capsys):
    root = write_manifests(tmp_path / 'semantic')
    stories, val, test = registered(root)
    wide, _, _ = registered(root, 'ALL')
    shared = fit_log(stories, val, test, budget_name='S3')
    oracle = shared.merged(arm='ORACLE')
    oracle.records.clear()
    for st in wide:
        oracle.record(DS_TARGET, st, 'transcript', 'oracle')
        oracle.record(DS_TARGET, st, 'brain', 'oracle')
    ok = write(tmp_path / 'oracle.json', fit_cell(shared.merged(oracle, arm='ORACLE'), arm='ORACLE'))
    assert audit(root, ok) == 0
    # An EB cell carrying the oracle-enabled log (e.g. weights built from the full-label map).
    eb = write(tmp_path / 'eb.json', fit_cell(shared.merged(oracle, arm='ORACLE'), arm='EB'))
    assert 'log of arm ORACLE in a cell of arm EB' in failed_reasons(root, eb, capsys)
    # The shared log itself refuses oracle reads.
    eb = write(tmp_path / 'eb_shared.json', fit_cell(shared.merged(oracle), arm='EB'))
    assert 'oracle read outside an ORACLE cell' in failed_reasons(root, eb, capsys)
    # g1: per-budget ISP logs plus the full-label oracle log.
    logs = {name: fit_log(registered(root, name)[0], val, test, stage='g1', budget_name=name, reads=False)
            for name in ('S3', 'S6')}
    for name, log in logs.items():
        for st in log.budget:
            log.record(DS_TARGET, st.stimulus, 'brain', 'isp_response')
    full = Provenance(DS_TARGET, 'S', wide, val, test, stage='g1', budget_name='ALL', arm='ORACLE')
    for st in wide:
        full.record(DS_TARGET, st, 'transcript', 'oracle')
    g1 = dict(stage='g1', subject=DS_TARGET, budgets={k: dict(provenance=v.to_dict()) for k, v in logs.items()},
              oracle=dict(provenance=full.to_dict()))
    assert audit(root, write(tmp_path / 'g1.json', g1)) == 0
    logs['S3'].record(DS_TARGET, wide[-1], 'brain', 'isp_response')  # S3 ISP reading S15 material.
    g1['budgets']['S3']['provenance'] = logs['S3'].to_dict()
    assert 'R2' in failed_reasons(root, write(tmp_path / 'g1_leak.json', g1), capsys)


def test_audit_requires_the_registered_inner_folds_and_their_tagged_reads(tmp_path, capsys):
    root = write_manifests(tmp_path / 'semantic')
    stories, val, test = registered(root)
    folds = bud.inner_folds('S3', stories)
    # Folds 0 and 1 swapped: an ISP fit on the story fold 0 really holds, tagged fold 0, passes R7 in the log.
    swapped = fit_log(stories, val, test, folds=[folds[1], folds[0], folds[2]])
    swapped.record(DS_TARGET, folds[0][0], 'brain', 'isp_response', fold=0)
    assert swapped.check() == []
    assert 'are not the registered S3 folds' in failed_reasons(root, write(tmp_path / 'swap.json',
                                                                           fit_cell(swapped)), capsys)
    # No folds declared (untagged per-fold ISP over the whole budget is invisible to R7).
    bare = fit_log(stories, val, test, folds=None)
    for st in stories:
        bare.record(DS_TARGET, st, 'brain', 'isp_response')
    assert 'inner folds None are not the registered' in failed_reasons(root, write(tmp_path / 'bare.json',
                                                                                   fit_cell(bare)), capsys)
    # One fold holding the whole budget, or empty folds: unreadable logs.
    for bad in ({'0': stories}, {'0': [], '1': [], '2': []}):
        log = fit_log(stories, val, test).to_dict()
        log['folds'] = bad
        assert 'unreadable log' in failed_reasons(root, write(tmp_path / 'bad.json', fit_cell(bare, provenance=log)),
                                                  capsys)
    # Registered folds, but the per-fold reads are untagged.
    untagged = fit_log(stories, val, test, reads=False)
    for st in stories:
        for kind, purpose in FOLD_READS:
            untagged.record(DS_TARGET, st, kind, purpose)
    for st in val:
        untagged.record(DS_TARGET, st, 'brain', 'decoder_score')
    assert 'inner fold 0: no tagged target' in failed_reasons(root, write(tmp_path / 'untagged.json',
                                                                          fit_cell(untagged)), capsys)
    # The tagged leak itself.
    leak = fit_log(stories, val, test)
    leak.record(DS_TARGET, folds[0][0], 'brain', 'isp_response', fold=0)
    assert 'R7' in failed_reasons(root, write(tmp_path / 'leak.json', fit_cell(leak)), capsys)
    # Arm-specific per-fold reads: without per-fold ISP a W1 or W0 cell passes, the ISP arms do not.
    no_isp = fit_log(stories, val, test, fold_reads=[r for r in FOLD_READS if r[1] != 'isp_response'])
    assert audit(root, *(write(tmp_path / f'{arm}.json', fit_cell(no_isp, arm=arm)) for arm in ('W0', 'W1'))) == 0
    for arm in ('EB', 'PRIOR', 'EB_f0.5', 'EB_SELF', 'EB_MISALIGN'):
        reasons = failed_reasons(root, write(tmp_path / f'{arm}.json', fit_cell(no_isp, arm=arm)), capsys)
        assert 'no tagged target brain isp_response' in reasons, arm
    no_own = fit_log(stories, val, test, fold_reads=[r for r in FOLD_READS if r[1] != 'own_encoding'])
    for arm in ('W1', 'EB', 'P_AC'):
        reasons = failed_reasons(root, write(tmp_path / f'{arm}.json', fit_cell(no_own, arm=arm)), capsys)
        assert 'own_encoding read for' in reasons, arm


def test_audit_passes_clean_cells_at_every_budget_with_the_registered_folds(tmp_path, capsys):
    root = write_manifests(tmp_path / 'semantic')
    _, val, test = registered(root)
    paths = []
    for name in ('S3', 'S6', 'S15', 'ALL'):
        stories = registered(root, name)[0]
        extra = dict(folds=all_folds(root)) if name == 'ALL' else {}
        paths.append(write(tmp_path / f'{name}.json', fit_cell(fit_log(stories, val, test, name=name, **extra),
                                                                name=name)))
    assert audit(root, *paths) == 0
    folds = all_folds(root)
    assert len(folds) == 5 and sorted(s for f in folds for s in f) == sorted(registered(root, 'ALL')[0])
    wide = registered(root, 'ALL')[0]  # ALL with leave-one-story-out folds instead of the session folds.
    wrong = write(tmp_path / 'wrong.json', fit_cell(fit_log(wide, val, test, name='S3'), name='ALL'))
    assert 'are not the registered ALL folds' in failed_reasons(root, wrong, capsys)


def test_audit_ties_reference_rules_to_the_secondary_arms(tmp_path, capsys):
    root = write_manifests(tmp_path / 'semantic')
    stories, val, test = registered(root)
    for kwargs in (dict(ts_allowed=['UTS01_only']), dict(reference_rule='SS')):
        log = fit_log(stories, val, test, arm='TH_SEL', **kwargs)
        log.record('UTS01', 'UTS01_only', 'transcript', 'reference_model')
        ok = write(tmp_path / 'th_sel.json', fit_cell(log, arm='TH_SEL', ledger_record=None))
        assert audit(root, ok) == 0
        for arm in ('EB', 'W0'):  # A Tier-1 cell carrying the secondary log.
            reasons = failed_reasons(root, write(tmp_path / 'eb.json', fit_cell(log, arm=arm, ledger_record=None)),
                                     capsys)
            assert 'TS/SS belong to the secondary arms' in reasons and 'log of arm TH_SEL' in reasons
            tampered = log.to_dict()
            tampered['arm'] = arm  # The log relabeled as the Tier-1 arm cannot even be rebuilt.
            cell = fit_cell(log, arm=arm, ledger_record=None, provenance=tampered)
            assert 'unreadable log' in failed_reasons(root, write(tmp_path / 'eb.json', cell), capsys)


def test_audit_checks_the_cell_fields_and_score_payload(tmp_path, capsys):
    root = write_manifests(tmp_path / 'semantic')
    stories, val, test = registered(root)
    log = fit_log(stories, val, test)
    bare = fit_cell(log)
    for key in ('stage', 'subject', 'arm'):
        bare.pop(key)
    g1 = dict(stage='g1', subject=DS_TARGET, provenance=fit_log(stories, val, test, stage='g1',
                                                                 budget_name='S3').to_dict())
    cases = {
        'test_before_lock': (fit_cell(log, test=dict(r=0.1)), "['test'] scored at stage develop"),
        'null_before_lock': (fit_cell(log, test_shift_null=dict(p=0.5)), "['test_shift_null'] scored at stage"),
        'bare': (bare, 'the cell names no subject'),
        'bare_stage': (bare, "the cell stage None"),
        'no_arm': ({k: v for k, v in fit_cell(log).items() if k != 'arm'}, 'a develop cell names no arm'),
        'g1_scores': (dict(g1, val=dict(r=0.1)), 'stage g1 scores nothing'),
        'not_object': ([fit_cell(log)], 'not a JSON object'),
    }
    for name, (cell, reason) in cases.items():
        assert reason in failed_reasons(root, write(tmp_path / f'{name}.json', cell), capsys), name
    dev = write(tmp_path / 'dev.json', fit_cell(log, val=dict(r=0.1)))
    assert audit(root, write(tmp_path / 'g1.json', g1), dev) == 0
