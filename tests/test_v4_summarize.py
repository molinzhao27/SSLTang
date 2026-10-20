"""v4 Module I: primary analysis (scripts/summarize_v4.py) and lock writer (scripts/v4_lock.py).

Every fixture is a temporary project root with JSON cells in the shapes the writers produce (Module G LPP cells
with nested ``residualized.test`` blocks and ledgers, Module H ds003020 cells with their code fields, XDEC cells,
sensitivity cells, G4(b) records); provenance logs, where a test needs the real audits, are built with
src.provenance. No data files, no GPU.
"""
import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_v4  # noqa: E402
import lpp_prepare  # noqa: E402
import summarize_v4 as sv  # noqa: E402
import v4_lock as vl  # noqa: E402
from src import budgets as bud  # noqa: E402
from src import v4_stats as st  # noqa: E402
from src.provenance import Provenance  # noqa: E402

VAL = sv.VAL
FP = dict(code_sha256='c' * 64, config_sha256='f' * 64, lpp_cell_code_sha256='1' * 64, lpp_runner_sha256='2' * 64,
          tanghuth_sha256='3' * 64, tanghuth_runner_sha256='9' * 64, tanghuth_cell_code_sha256='8' * 64,
          summarize_sha256='4' * 64)
LPP_CONFIG = FP['config_sha256']  # Module G cells carry run_v3.config_sha256 of the same config.
PREP_CODE = {'scripts/lpp_prepare.py': '6' * 64, 'src/budgets.py': '7' * 64}
MASK_SHA = 'a' * 64
SECTIONS = sv.SECTIONS


def ok_audit(path):
    return dict(passed=True)


def ids(n):
    return [f'sub-EN{57 + i:03d}' for i in range(n)]


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def eb_info(var_z=2.0, sigma2=1.0, b=1.0, var_p=0.5, kappa=0.5):
    return dict(a=0.0, b=b, var_z=var_z, var_p=var_p, sigma2=sigma2, kappa=kappa, tau2=var_z - sigma2, V=1.0)


# ------------------------------------------------------------------------------- writer-shaped LPP cells
def lpp_log(subject, t, name, arm, stage, refs=(), final=True):
    """A provenance log as Module G writes it (audit_v4's registered fold reads; TH_SEL and XDEC with Module H's
    reads, which ``tanghuth.audit_file`` checks)."""
    from src.priors import tanghuth as th
    t = str(t)
    budget = bud.lpp_section_budgets(t, SECTIONS)[name]
    if arm == 'XDEC':
        p = Provenance(subject, 'S', budget, [], [t], reference_rule='SS', arm='XDEC', budget_name=name, stage=stage)
        th.log_reads(p, [th.read(r, x, k, 'reference_model') for r in refs for x in SECTIONS if x != t
                         for k in ('brain', 'transcript')])
        th.log_reads(p, th.xdec_reads(subject, refs, budget, [t]))
        p.check()
        return p.to_dict()
    folds = [tuple(f) for f in bud.inner_folds(name, budget)]
    p = Provenance(subject, 'S', budget, [], [t], folds=folds, arm=arm, budget_name=name, stage=stage,
                   reference_rule='SS' if arm == 'TH_SEL' else 'none')
    for x in list(budget) + [t]:
        p.record(subject, x, 'transcript', 'context_reset')
    unheard = th.lpp_unheard(SECTIONS, budget, t) if arm == 'TH_SEL' else []
    if arm == 'TH_SEL':  # Reference models (SS: any section but the target's test section), then selections.
        th.log_reads(p, [th.read(r, x, k, 'reference_model') for r in refs for x in SECTIONS if x != t
                         for k in ('brain', 'transcript')])
    for i, held in enumerate(folds):
        with p.inner_fold(i):
            fit = [x for x in budget if x not in held]
            for kind, purpose, _ in audit_v4.fold_reads(arm):
                for x in fit:
                    p.record(subject, x, kind, purpose)
            if arm == 'TH_SEL':  # The selection is refit inside every inner fold.
                th.log_reads(p, th.thsel_reads(subject, refs, fit, unheard))
            for x in held:
                p.record(subject, x, 'brain', 'decoder_score')
                p.record(subject, x, 'transcript', 'decoder_score')
    if arm == 'TH_SEL':
        th.log_reads(p, th.thsel_reads(subject, refs, budget, unheard))
    if final:
        for x in budget:
            p.record(subject, x, 'brain', 'decoder_fit')
            p.record(subject, x, 'transcript', 'decoder_fit')
        p.record(subject, t, 'brain', 'decoder_score')
        p.record(subject, t, 'transcript', 'decoder_score')
    p.check()
    return p.to_dict()


def lpp_cell(subject, t, budget, arm, stage, raw, res, pool, fp=FP, config=None, logs=False, eb=None):
    """A Module G cell (base_cell + add_scores; TH_SEL / XDEC with Module H's fields)."""
    config = fp['config_sha256'] if config is None else config
    refs = sorted(x for x in pool if x != subject)
    reference_rule = 'SS' if arm in ('TH_SEL', 'XDEC') else 'none'
    cell = dict(protocol='ssltang-v4', dataset='lpp', stage=stage, subject=subject, clock='dense', test_section=str(t),
                budget=budget, arm=arm, regime='S', estimable=True, code_sha256=fp['lpp_cell_code_sha256'],
                core_code_sha256=fp['code_sha256'], config_sha256=config, mask_sha256=MASK_SHA,
                ledger=dict(references=refs if sv.reads_references(arm) else [], reference_rule=reference_rule,
                            target_pairs_dense=0 if arm == 'XDEC' else 900, target_pairs_stride4=0 if arm == 'XDEC'
                            else 225, transcript_words=0 if arm == 'XDEC' else 3000),
                test={VAL: raw, 'by_run_feature_correlation': {f'lppEN_section-{t}': raw}})
    if arm != 'XDEC' and res is not None:
        cell['residualized'] = dict(test={VAL: res}, nuisance=dict(alpha=10.0), rule='targets minus the B-fit nuisance')
    if arm == 'XDEC':
        cell.update(references=refs, reference_rule='SS', target_labels=0)
    if arm == 'TH_SEL':
        cell.update(references=refs, reference_rule='SS', engine_arm='PRIOR')
    if eb is not None:
        cell['final_weights'] = dict(eb={d: eb for d in ('1', '2', '3')})
    if logs:
        cell['provenance'] = lpp_log(subject, t, budget, arm, stage, refs if arm in ('XDEC', 'TH_SEL') else ())
    return cell


def family_cells(pl):
    """{(budget, arm)} of every cell the plan reads, plus ORACLE and W0 at the oracle budgets."""
    need = set(sv.required_cells(pl))
    for budget in sv.oracle_budgets(pl):
        need |= {(budget, 'ORACLE'), (budget, 'W0')}
    return sorted(need)


# ------------------------------------------------------------------------------------- CONFIRM fixture
BASE = dict(W0=0.0, W1=0.01, EB=0.03, PRIOR=0.02, ORACLE=0.05, EB_SELF=0.015, TH_SEL=0.012)
L8 = dict(W0=0.0, W1=0.04, EB=0.041, PRIOR=0.041, ORACLE=0.06)


def arm_effect(arm, budget, endpoint, scenario):
    m = re.fullmatch(r'(EB|PRIOR)_f(.+)', arm)
    if m:
        v = 0.01 + 0.02 * float(m[2])
    else:
        v = (L8 if budget == 'L8' else BASE)[arm]
    if scenario == 'prior_null' and arm == 'PRIOR':
        v = 0.005  # Below W1: H1 fails deterministically.
    if endpoint == 'residualized':
        v = (-0.005 if arm == 'EB' else 0.0) if scenario == 'h2_fails' else v / 2
    return v


def confirm_lock(advanced, drop_h6, dev, confirm, pl, **changes):
    return dict(dict(confirmatory=True, confirm_status=dict(status='registered', reasons=[]), advanced_arm=advanced,
                     drop_h6=drop_h6, primary_budget=dict(ds003020='S3', lpp='L3'),
                     lpp_partition=dict(dev=dev, confirm=confirm), confirmatory_plan=sv.clean(pl),
                     lpp_config_sha256=LPP_CONFIG, **FP), **changes)


def prep_runs(subjects):
    return {s: [dict(section=k, run_number=k, n_tr=n) for k, n in zip(range(1, 10), sv.SECTION_TRS)] for s in subjects}


def confirm_root(tmp, advanced='EB', drop_h6=False, scenario='strong', n_ids=17, lock=True, value=None,
                 unprepared=(), mismatched=()):
    """A project root with a lock, preparation records (DEV and a CONFIRM batch) and CONFIRM cells for the whole
    confirmatory family. ``value(i, s, t, budget, arm, endpoint)`` overrides the scenario's cell values;
    ``unprepared`` CONFIRM listeners have no preparation record and no cells, ``mismatched`` ones a record with a
    section-1 TR count of 281 (and cells); no cell uses either as a reference."""
    root = Path(tmp)
    dev, confirm = bud.lpp_partition(ids(n_ids))
    pool = [s for s in list(dev) + list(confirm) if s not in unprepared and s not in mismatched]
    pl = sv.plan(advanced, drop_h6, 'L3')
    rng = np.random.default_rng(7)
    multiplier = {s: 1.0 + 0.3 * rng.standard_normal() for s in confirm}
    section = {t: 0.02 * rng.standard_normal() for t in SECTIONS}
    values = {}
    for i, s in enumerate(confirm):
        if s in unprepared:
            continue
        for budget, arm in family_cells(pl):
            for t in SECTIONS:
                v = {}
                for endpoint in ('raw', 'residualized'):
                    v[endpoint] = (value(i, s, t, budget, arm, endpoint) if value is not None else
                                   0.10 + 0.005 * i + section[t] + multiplier[s] * arm_effect(arm, budget, endpoint,
                                                                                              scenario)
                                   + 0.002 * rng.standard_normal())
                    values[(s, t, budget, arm, endpoint)] = v[endpoint]
                save(sv.lpp_cell(root, s, 'dense', 'evaluate', t, budget, arm),
                     lpp_cell(s, t, budget, arm, 'evaluate', v['raw'], v['residualized'], pool))
    save(root / 'data/lpp/prepare_provenance.json', dict(mode='DEV', g4a=dict(passed=True, runs=prep_runs(dev))))
    runs = prep_runs([s for s in confirm if s not in unprepared])
    for s in mismatched:
        runs[s][0]['n_tr'] = 281
    save(root / 'data/lpp/prepare_provenance_confirm_b1.json', dict(mode='CONFIRM', g4a=dict(passed=True, runs=runs)))
    if lock:
        save(root / sv.LOCK, confirm_lock(advanced, drop_h6, dev, confirm, pl))
    return root, confirm, values


def r_s(values, subjects, budget, arm, endpoint='raw'):
    return np.array([np.mean([values[(s, t, budget, arm, endpoint)] for t in SECTIONS]) for s in subjects])


TREE_OK = dict(commit='0' * 40, lock_tracked=True, files=60, documents=3, ignored=[], problems=[])


def run(root, **kw):
    kw = dict(dict(fingerprints=FP, audit=ok_audit, git_state=None, no_git_check=True, tree=TREE_OK), **kw)
    return sv.run_primary(root, **kw)


@pytest.fixture(scope='module')
def strong(tmp_path_factory):
    root, confirm, values = confirm_root(tmp_path_factory.mktemp('strong'))
    return root, confirm, values, run(root)


# --------------------------------------------------------------------------------- the lock key schema
def test_fingerprint_schema_is_the_runners_schema():
    """Every key run_v4_lpp.check_lock and run_v4_tanghuth.check_lock compare is pinned, with their values."""
    import run_v4
    import run_v4_lpp
    import run_v4_tanghuth
    from src.priors import tanghuth as th
    from src.runtime import load_config
    fp = sv.current_fingerprints()
    assert set(fp) == set(sv.FINGERPRINTS) and all(sv.HEX64.match(v) for v in fp.values())
    assert fp['summarize_sha256'] == sv.sha256_file(Path(sv.__file__))  # The running script, not --root's copy.
    ctx = run_v4_lpp.Context(load_config(ROOT / sv.CONFIG, []), device='cpu')
    want = run_v4_lpp.fingerprints(ctx)
    assert {k: fp.get(k) for k in want} == want
    assert fp['lpp_cell_code_sha256'] == run_v4_lpp.code_sha256() and fp['code_sha256'] == run_v4.code_sha256()
    assert fp['tanghuth_cell_code_sha256'] == run_v4_tanghuth.code_sha256()
    assert fp['tanghuth_sha256'] == run_v4.file_sha('src/priors/tanghuth.py')
    assert sv.TANGHUTH_ARMS == th.ARMS
    for arm in set(run_v4.REFERENCE_ARMS) | set(run_v4_lpp.REFERENCE_ARMS) | set(th.REFERENCE_ARMS):
        assert sv.reads_references(arm), arm
    for arm in ('W0', 'W1', 'EB_SELF', 'ORACLE', 'TH_SEL_rand', 'P_AC', 'EB_K2'):
        assert not sv.reads_references(arm), arm


def test_code_files_cover_every_fingerprinted_file():
    import run_v3
    import run_v3_weights
    import run_v4
    files = set(vl.code_files())
    assert set(run_v4.CORE) | set(run_v3.LOCKED_CODE) | set(run_v3_weights.CODE) | set(lpp_prepare.CODE_FILES) <= files
    assert {'scripts/run_v4_lpp.py', 'scripts/run_v4_tanghuth.py', 'src/priors/tanghuth.py',
            'scripts/summarize_v4.py', 'scripts/v4_lock.py'} <= files
    textgrid = 'reference/semantic-decoding/decoding/utils_ridge/textgrid.py'
    state = vl.git_state([textgrid])
    if state is not None and (ROOT / textgrid).is_file():  # Git-ignored: pinned by hash, never "dirty".
        assert state['ignored'] == [textgrid] and state['dirty'] == []


# ------------------------------------------------------------------------------------------- the plan
def test_plan_family_and_substitution():
    pl = sv.plan('EB', False, 'L3')
    assert pl['order'] == ['H1', 'H2', 'H3', 'H4', 'H5', 'H6'] and pl['dropped'] == []
    assert pl['hypotheses']['H3']['ladder'] == ['EB_f0', 'EB_f0.25', 'EB_f0.5', 'EB_f0.75', 'EB']
    assert pl['hypotheses']['H6']['budget'] == 'L8' and pl['hypotheses']['H6']['null'] == -0.005
    assert (pl['clock'], pl['cell_stage']) == ('dense', 'evaluate')
    prior = sv.plan('PRIOR', False, 'L5')
    assert prior['order'] == ['H1', 'H2', 'H3', 'H4', 'H5'] and prior['dropped'] == ['H6']
    assert prior['hypotheses']['H1']['contrasts'] == [['PRIOR', 'W1'], ['PRIOR', 'W0']]
    assert prior['hypotheses']['H3']['ladder'][0] == 'PRIOR_f0' and prior['budget'] == 'L5'
    assert sv.plan('EB', True, 'L3')['dropped'] == ['H6']
    need = sv.required_cells(pl)
    assert need[('L3', 'EB')] == ['raw', 'residualized'] and need[('L8', 'W1')] == ['raw']
    assert ('L3', 'ORACLE') not in need and sv.oracle_budgets(pl) == ['L3', 'L8']
    assert sv.hypothesis_cells(pl['hypotheses']['H5']) == [('L3', 'EB'), ('L3', 'TH_SEL')]
    for bad in (('W1', False, 'L3'), ('EB', False, 'L8'), ('EB', None, 'L3')):
        with pytest.raises(ValueError):
            sv.plan(*bad)


def test_ds_cell_names():
    m = sv.DS_NAME.match('S3_TH_SEL__refs-UTS01-UTS02.json')
    assert (m['budget'], m['arm'], m['refs']) == ('S3', 'TH_SEL', 'UTS01-UTS02')
    for name, arm in (('S3_EB_f0.75.json', 'EB_f0.75'), ('ALL_TH_SEL_rand.json', 'TH_SEL_rand'), ('S15_W0.json', 'W0')):
        m = sv.DS_NAME.match(name)
        assert m['arm'] == arm and m['refs'] is None


# ---------------------------------------------------------------------------------------- the analysis
def test_h1_statistics_match_v4_stats(strong):
    root, confirm, values, out = strong
    h1 = out['hypotheses']['H1']
    assert out['h1'] is True and h1['confirmed'] is True and out['n_analyzed'] == len(confirm)
    eb, w1, w0, orc = (r_s(values, confirm, 'L3', a) for a in ('EB', 'W1', 'W0', 'ORACLE'))
    for c, d in zip(h1['contrasts'], (eb - w1, eb - w0)):
        ref = st.contrast_summary(d)
        for k in ('mean', 'sd', 'dz', 'p_one_sided', 'n_positive', 'sesoi_tier'):
            assert c[k] == ref[k]
        assert np.allclose(c['ci95_bca'], ref['ci95_bca']) and np.allclose(c['ci90_bca'], ref['ci90_bca'])
        assert np.allclose(c['positive_ci95'], st.exact_binomial_ci(ref['n_positive'], len(d)))
        assert c['oracle_fraction'] == pytest.approx(st.oracle_fraction(eb, w0, orc))
        assert c['p_one_sided'] < 0.025 and c['oracle_n'] == len(confirm)
    assert st.iut(h1['contrasts'][0]['p_one_sided'], h1['contrasts'][1]['p_one_sided'],
                  h1['contrasts'][0]['mean'], h1['contrasts'][1]['mean'])
    for lmm in h1['lmm']:
        assert lmm['n'] == 9 * len(confirm) and lmm['n_sections'] == 9 and isinstance(lmm['supports'], bool)
    assert h1['claim'].startswith('generalizes') == h1['lmm_supports']


def test_h1_contrast_against_independent_references(strong):
    """p by exact enumeration of all 2^n sign flips; the BCa CI by scipy.stats.bootstrap; d_z by hand."""
    from scipy import stats
    root, confirm, values, out = strong
    d = r_s(values, confirm, 'L3', 'EB') - r_s(values, confirm, 'L3', 'W1')
    c = out['hypotheses']['H1']['contrasts'][0]
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
    exact = float(np.mean(signs @ d >= d.sum() - 1e-12))
    assert c['p_one_sided'] == pytest.approx(exact, abs=3e-3)
    assert c['mean'] == pytest.approx(d.mean()) and c['dz'] == pytest.approx(d.mean() / d.std(ddof=1))
    boot = stats.bootstrap((d,), np.mean, n_resamples=st.N_BOOT, method='BCa', confidence_level=0.95,
                           random_state=np.random.default_rng(st.BOOT_SEED), vectorized=True)
    assert np.allclose(c['ci95_bca'], [boot.confidence_interval.low, boot.confidence_interval.high], rtol=1e-9)
    assert c['n_positive'] == int((d > 0).sum()) and c['deltas'] == pytest.approx(dict(zip(confirm, d)))


def test_fixed_sequence_secondaries(strong):
    root, confirm, values, out = strong
    assert out['verdicts'] == {h: 'CONFIRMED' for h in ('H1', 'H2', 'H3', 'H4', 'H5', 'H6')}
    h = out['hypotheses']
    d2 = r_s(values, confirm, 'L3', 'EB', 'residualized') - r_s(values, confirm, 'L3', 'W1', 'residualized')
    assert h['H2']['contrasts'][0]['p_one_sided'] == st.sign_flip_p(d2)
    assert h['H2']['contrasts'][0]['oracle_fraction'] is None  # Raw endpoint only.
    f = np.array(sv.LADDER_F)
    y = np.stack([r_s(values, confirm, 'L3', a) for a in h['H3']['spec']['ladder']], 1)
    y -= r_s(values, confirm, 'L3', 'W1')[:, None]
    slopes = np.array([np.polyfit(f, row, 1)[0] for row in y])
    assert np.allclose(list(h['H3']['slopes'].values()), slopes)
    assert h['H3']['p_one_sided'] == st.sign_flip_p(slopes) and h['H3']['sesoi_tier'] is None
    d6 = r_s(values, confirm, 'L8', 'EB') - r_s(values, confirm, 'L8', 'W1')
    c6 = h['H6']['contrasts'][0]
    assert c6['p_one_sided'] == st.sign_flip_p(d6 + 0.005) and c6['null'] == -0.005
    assert c6['oracle_n'] == len(confirm)  # W0 and ORACLE at L8 were QC'd for the oracle fraction.
    d4 = r_s(values, confirm, 'L3', 'EB') - r_s(values, confirm, 'L3', 'EB_SELF')
    assert h['H4']['contrasts'][0]['mean'] == pytest.approx(d4.mean())
    assert {k: v['n'] for k, v in out['samples'].items()} == {k: len(confirm) for k in out['plan']['order']}


def test_outputs_written_once(strong):
    root, confirm, values, out = strong
    saved = json.loads((root / sv.PRIMARY_JSON).read_text())
    assert saved['verdicts'] == out['verdicts'] and saved['lock_sha256'] == sv.sha256_file(root / sv.LOCK)
    assert len(saved['cells_read']) > 0 and (root / sv.STARTED).is_file()
    assert saved['git_check'] == 'skipped (--no-git-check)' and set(saved['fingerprints']) == set(sv.FINGERPRINTS)
    text = (root / sv.PRIMARY_MD).read_text()
    assert 'H1 CONFIRMED' in text and 'Crossed LMM' in text and 'n (QC sample)' in text
    with pytest.raises(sv.Refused, match='exactly once'):
        run(root)
    with pytest.raises(sv.Refused, match='exactly once'):
        sv.main(['--qc', '--root', str(root)])


def test_start_marker_blocks_a_second_look(tmp_path):
    root, _, _ = confirm_root(tmp_path)
    save(root / sv.STARTED, {})
    with pytest.raises(sv.Refused, match='exactly once'):
        run(root)


def test_failed_step_stops_the_sequence_and_later_cells_are_not_read(tmp_path):
    root, confirm, values = confirm_root(tmp_path, scenario='h2_fails')
    out = run(root)
    assert out['verdicts']['H1'] == 'CONFIRMED' and out['verdicts']['H2'] == 'NOT CONFIRMED'
    assert [out['verdicts'][h] for h in ('H3', 'H4', 'H5', 'H6')] == ['NOT TESTED'] * 4
    assert set(out['arms_read']) == {'EB', 'W1', 'W0', 'ORACLE'}
    assert {r['endpoint'] for r in out['blocks_read']} == {'raw', 'residualized'}
    assert 'H3' not in out['hypotheses']


def test_prior_substitution_drops_h6(tmp_path):
    root, confirm, values = confirm_root(tmp_path, advanced='PRIOR', scenario='prior_null')
    out = run(root)
    assert out['plan']['advanced'] == 'PRIOR' and out['h1'] is False
    assert out['verdicts'] == dict(H1='NOT CONFIRMED', H2='NOT TESTED', H3='NOT TESTED', H4='NOT TESTED',
                                   H5='NOT TESTED', H6='DROPPED')
    assert out['hypotheses']['H1']['claim'] == 'not confirmed'
    assert out['hypotheses']['H1']['contrasts'][0]['mean'] < 0


# --------------------------------------------------------------------------------------------- the QC
def qc_record(root, **kw):
    return sv.run_qc(root, fingerprints=FP, audit=kw.pop('audit', ok_audit), write=False, **kw)[2]


def edit_cell(root, subject, section, budget, arm, drop=(), **changes):
    path = sv.lpp_cell(root, subject, 'dense', 'evaluate', section, budget, arm)
    cell = json.loads(path.read_text())
    cell.update(changes)
    for key in drop:
        cell.pop(key)
    path.write_text(json.dumps(cell))
    return path


def test_qc_registered_exclusions_and_oracle(tmp_path, monkeypatch):
    confirm = bud.lpp_partition(ids(17))[1]
    root, confirm, _ = confirm_root(tmp_path, mismatched=[confirm[4]])
    sv.lpp_cell(root, confirm[0], 'dense', 'evaluate', '4', 'L3', 'W1').unlink()
    edit_cell(root, confirm[1], '2', 'L3', 'W0', estimable=False)
    edit_cell(root, confirm[2], '9', 'L3', 'EB', drop=['test'])
    sv.lpp_cell(root, confirm[3], 'dense', 'evaluate', '1', 'L3', 'ORACLE').unlink()

    def forbidden(*args, **kw):
        raise AssertionError('QC used an endpoint value')
    monkeypatch.setattr(sv, 'endpoint_value', forbidden)
    rec = qc_record(root)
    cats = {s: sorted({r['category'] for r in rs}) for s, rs in rec['excluded'].items()}
    assert cats == {confirm[0]: ['missing file'], confirm[1]: ['W0 not estimable'], confirm[2]: ['corrupt file'],
                    confirm[4]: ['TR-count mismatch']}
    assert rec['included'] == [s for s in confirm if s not in cats] and not rec['refusals']
    assert confirm[3] in rec['included'] and confirm[3] not in rec['oracle_subjects']['L3']
    assert confirm[3] in rec['oracle_subjects']['L8'] and rec['unregistered_exclusions'] == []
    assert all(len(v) == 64 for v in rec['files'].values())


def test_qc_samples_are_per_hypothesis(tmp_path):
    """A secondary-only defect leaves the H1 sample intact and needs no deviation."""
    root, confirm, values = confirm_root(tmp_path)
    edit_cell(root, confirm[0], '5', 'L3', 'EB', drop=['residualized'])  # Read only by H2.
    sv.lpp_cell(root, confirm[1], 'dense', 'evaluate', '2', 'L8', 'W1').unlink()  # Read only by H6.
    sv.lpp_cell(root, confirm[2], 'dense', 'evaluate', '7', 'L3', 'EB_f0.5').unlink()  # Read only by H3.
    rec = qc_record(root)
    samples = {k: sorted(v['excluded']) for k, v in rec['samples'].items()}
    assert samples == dict(H1=[], H2=[confirm[0]], H3=[confirm[2]], H4=[], H5=[], H6=[confirm[1]])
    assert rec['included'] == list(confirm) and rec['unregistered_exclusions'] == []
    gaps = rec['completeness']
    assert gaps['missing'] == {'L8 W1': {confirm[1]: ['2']}, 'L3 EB_f0.5': {confirm[2]: ['7']}}
    assert gaps['missing_file_exclusions'] == sorted([confirm[1], confirm[2]])
    assert gaps['systematic'] == [] and gaps['unrun'] == [] and gaps['untestable'] == []
    with pytest.raises(sv.Refused, match='--accept-missing'):  # Registered, but acknowledged before it applies.
        run(root)
    assert not (root / sv.STARTED).exists()
    out = run(root, accept_missing='two files lost in transfer (ticket 12)')
    assert out['accept_missing'] == 'two files lost in transfer (ticket 12)' and out['deviation_covers'] == []
    assert json.loads((root / sv.STARTED).read_text())['accept_missing'] == out['accept_missing']
    assert out['n_analyzed'] == len(confirm) and out['deviation'] is None and out['excluded'] == {}
    assert out['hypotheses']['H1']['contrasts'][0]['n'] == len(confirm)
    assert out['hypotheses']['H2']['contrasts'][0]['n'] == len(confirm) - 1
    assert out['hypotheses']['H3']['n'] == len(confirm) - 1 and confirm[2] not in out['hypotheses']['H3']['slopes']
    c6 = out['hypotheses']['H6']['contrasts'][0]
    keep = [s for s in confirm if s != confirm[1]]
    d6 = r_s(values, keep, 'L8', 'EB') - r_s(values, keep, 'L8', 'W1')
    assert c6['n'] == len(keep) and c6['p_one_sided'] == st.sign_flip_p(d6 + 0.005)
    assert out['samples']['H6']['n'] == len(keep) and 'Secondary samples' in (root / sv.PRIMARY_MD).read_text()


def test_qc_non_finite_score_is_not_a_registered_exclusion(tmp_path):
    root, confirm, _ = confirm_root(tmp_path)
    edit_cell(root, confirm[0], '9', 'L3', 'EB', test={VAL: float('nan')})
    rec = qc_record(root)
    assert [r['category'] for r in rec['excluded'][confirm[0]]] == ['non-finite score']
    assert rec['unregistered_exclusions'] == ['non-finite score']
    with pytest.raises(sv.Refused, match='deviation'):
        run(root)


def test_qc_refusals(tmp_path):
    root, confirm, _ = confirm_root(tmp_path)
    edit_cell(root, confirm[0], '3', 'L3', 'TH_SEL', code_sha256='0' * 64)
    edit_cell(root, confirm[1], '3', 'L3', 'W1', config_sha256='0' * 64)
    bad = sv.lpp_cell(root, confirm[2], 'dense', 'evaluate', '5', 'L8', 'EB')
    edit_cell(root, confirm[3], '6', 'L3', 'EB', drop=['config_sha256'])  # The configuration hash is mandatory.
    dev = bud.lpp_partition(ids(17))[0]
    edit_cell(root, confirm[4], '1', 'L3', 'EB', ledger=dict(references=[s for s in dev], reference_rule='none'))
    rec = qc_record(root, audit=lambda p: dict(passed=p != bad, rules=['R5']))
    assert sorted((r['subject'], r['category']) for r in rec['refusals']) == sorted(
        [(confirm[0], 'other code'), (confirm[1], 'other code'), (confirm[2], 'audit failed'),
         (confirm[3], 'other code'), (confirm[4], 'wrong references')])
    with pytest.raises(sv.Refused, match='other code, with other references or failing the audit'):
        run(root, audit=lambda p: dict(passed=p != bad))
    assert not (root / sv.STARTED).exists() and not (root / sv.PRIMARY_JSON).exists()


def test_qc_section_fields_and_provenance_must_match_the_path(tmp_path):
    root, confirm, _ = confirm_root(tmp_path)
    pool = bud.lpp_partition(ids(17))[0] + confirm
    log = lpp_log(confirm[0], '5', 'L3', 'EB', 'evaluate')  # A section-5 result saved as section 3.
    edit_cell(root, confirm[0], '3', 'L3', 'EB', provenance=log)
    edit_cell(root, confirm[1], '4', 'L3', 'W1', drop=['test_section'])
    edit_cell(root, confirm[2], '4', 'L3', 'W1', drop=['clock'])
    good = lpp_log(confirm[3], '2', 'L3', 'EB', 'evaluate')
    edit_cell(root, confirm[3], '2', 'L3', 'EB', provenance=good)
    rec = qc_record(root)
    details = {s: [r['detail'] for r in rs] for s, rs in rec['excluded'].items()}
    assert set(details) == {confirm[0], confirm[1], confirm[2]}
    assert "provenance test_items ['5']" in details[confirm[0]][0]
    assert 'test_section' in details[confirm[1]][0] and 'clock' in details[confirm[2]][0]
    assert all(r['category'] == 'corrupt file' for rs in rec['excluded'].values() for r in rs)
    assert pool and confirm[3] in rec['included']


def test_qc_unscored_cell_may_fail_only_on_its_recorded_scope_errors(tmp_path):
    root, confirm, _ = confirm_root(tmp_path)
    reasons = ['no untagged (final) target brain decoder_fit read for [...]']
    unscored = edit_cell(root, confirm[0], '3', 'L3', 'W0', estimable=False, scope_errors=reasons,
                         drop=['test', 'residualized'])
    scored = edit_cell(root, confirm[1], '3', 'L3', 'W0', scope_errors=reasons)
    other = edit_cell(root, confirm[2], '3', 'L3', 'W0', estimable=False, scope_errors=['something else'],
                      drop=['test', 'residualized'])

    def audit(path):
        if path in (unscored, scored, other):
            return dict(passed=False, rules=['scope'], violations=[dict(rule='scope', path='$', reason=reasons[0])])
        return dict(passed=True)
    rec = qc_record(root, audit=audit)
    assert [r['category'] for r in rec['excluded'][confirm[0]]] == ['W0 not estimable']
    assert sorted((r['subject'], r['category']) for r in rec['refusals']) == [(confirm[1], 'audit failed'),
                                                                             (confirm[2], 'audit failed')]


def test_unregistered_exclusion_needs_a_recorded_deviation(tmp_path):
    root, confirm, _ = confirm_root(tmp_path)
    edit_cell(root, confirm[0], '6', 'L3', 'EB_SELF', estimable=False)
    with pytest.raises(sv.Refused, match='deviation'):
        run(root)
    assert not (root / sv.STARTED).exists()
    assert json.loads((root / sv.QC_JSON).read_text())['unregistered_exclusions'] == ['arm not estimable']
    out = run(root, deviation='PROTOCOL_V4 §4 item 99')
    assert out['deviation'] == 'PROTOCOL_V4 §4 item 99' and confirm[0] in out['subjects']  # EB_SELF: H4 only.
    assert confirm[0] in out['samples']['H4']['excluded'] and out['samples']['H4']['n'] == len(confirm) - 1
    assert len(out['qc_attempts']) == 2


# ------------------------------------------------------------------------- completeness before the start
def remove_cells(root, subjects, budget, arm, sections=SECTIONS):
    for subject in subjects:
        for t in sections:
            sv.lpp_cell(root, subject, 'dense', 'evaluate', t, budget, arm).unlink()


def test_systematic_omission_is_refused_before_the_start(tmp_path):
    """PRIOR advanced and the PRIOR_f* ladder never run: H3's sample is empty. Refused before the start marker; under
    a recorded deviation H3 is NOT TESTABLE (not NOT CONFIRMED) and the sequence stops there."""
    root, confirm, _ = confirm_root(tmp_path, advanced='PRIOR')
    for f in (0.0, 0.25, 0.5, 0.75):
        remove_cells(root, confirm, 'L3', sv.ladder_arm('PRIOR', f))
    rec = qc_record(root)
    gaps = rec['completeness']
    assert gaps['untestable'] == ['H3'] and gaps['sample_sizes']['H3'] == 0 and gaps['sample_sizes']['H1'] == 8
    assert [(g['budget'], g['arm'], g['sections']) for g in gaps['systematic']] == [
        ('L3', sv.ladder_arm('PRIOR', f), list(SECTIONS)) for f in (0.0, 0.25, 0.5, 0.75)]
    assert rec['unregistered_exclusions'] == [] and not rec['refusals']  # All registered 'missing file' exclusions.
    with pytest.raises(sv.Refused, match='a run that did not happen'):
        run(root, accept_missing='acknowledged')
    assert not (root / sv.STARTED).exists() and not (root / sv.PRIMARY_JSON).exists()
    out = run(root, accept_missing='acknowledged', deviation='PROTOCOL_V4 §4 item 98')
    assert out['verdicts'] == dict(H1='CONFIRMED', H2='CONFIRMED', H3='NOT TESTABLE', H4='NOT TESTED',
                                   H5='NOT TESTED', H6='DROPPED')
    assert 'systematic omissions' in out['deviation_covers'][0] and 'H3' in out['deviation_covers'][1]
    assert out['hypotheses']['H3']['testable'] is False and 'PRIOR_f0.5' not in out['arms_read']
    assert 'NOT TESTABLE' in (root / sv.PRIMARY_MD).read_text()


def test_systematic_omission_at_one_section(tmp_path):
    """One reference's model fails at section 4, so TH_SEL has no cell there for any target: not a data defect."""
    root, confirm, _ = confirm_root(tmp_path)
    remove_cells(root, confirm, 'L3', 'TH_SEL', ['4'])
    gaps = qc_record(root)['completeness']
    assert gaps['systematic'] == [dict(budget='L3', arm='TH_SEL', sections=['4'])] and gaps['untestable'] == ['H5']
    with pytest.raises(sv.Refused, match=r"systematic omissions \[\('L3', 'TH_SEL', \['4'\]\)\]"):
        run(root, accept_missing='acknowledged')
    assert not (root / sv.STARTED).exists()


def test_premature_launch_is_refused_before_the_start(tmp_path):
    """Three listeners still computing (partial cells) and one never run: refused; the never-run listener needs a
    deviation, the partial ones an explicit acknowledgment."""
    root, confirm, _ = confirm_root(tmp_path)
    remove_cells(root, confirm[5:], 'L3', 'EB', ['7', '8', '9'])
    for budget, arm in family_cells(sv.plan('EB', False, 'L3')):
        remove_cells(root, confirm[4:5], budget, arm)
    rec = qc_record(root)
    gaps = rec['completeness']
    assert gaps['unrun'] == [confirm[4]] and gaps['systematic'] == []
    assert gaps['missing']['L3 EB'] == {**{s: ['7', '8', '9'] for s in confirm[5:]}, confirm[4]: list(SECTIONS)}
    assert rec['included'] == list(confirm[:4]) and gaps['missing_file_exclusions'] == sorted(confirm[4:])
    with pytest.raises(sv.Refused, match="none of the family's cells"):
        run(root, accept_missing='acknowledged')
    remove = root / 'results/v4_lpp' / confirm[4]
    shutil.rmtree(remove)
    edit_record = root / 'data/lpp/prepare_provenance_confirm_b1.json'
    record = json.loads(edit_record.read_text())
    record['g4a']['runs'].pop(confirm[4])  # Now a subject-level exclusion: no preparation record.
    edit_record.write_text(json.dumps(record))
    with pytest.raises(sv.Refused, match='wrong references|other references'):  # Cells used it as a reference.
        run(root, accept_missing='acknowledged')
    assert not (root / sv.STARTED).exists()


def test_missing_files_need_an_acknowledgment(tmp_path):
    root, confirm, _ = confirm_root(tmp_path)
    remove_cells(root, confirm[5:], 'L3', 'EB', ['7', '8', '9'])
    with pytest.raises(sv.Refused, match="'missing file' exclusions of .*--accept-missing"):
        run(root)
    assert not (root / sv.STARTED).exists()
    out = run(root, accept_missing='listeners 6-8 still computing; analysed as registered')
    assert out['n_analyzed'] == 5 and sorted(out['excluded']) == sorted(confirm[5:])
    assert out['completeness']['missing_file_exclusions'] == sorted(confirm[5:])
    assert 'acknowledged before scoring' in (root / sv.PRIMARY_MD).read_text()


def test_reference_pool_is_the_listeners_passing_subject_qc(tmp_path):
    """A CONFIRM listener that was never prepared (e.g. its TR counts failed lpp_prepare.verify_lengths) cannot be a
    reference: the others' 47-reference cells pass, and a cell that used it is refused."""
    dev, confirm = bud.lpp_partition(ids(17))
    root, confirm, _ = confirm_root(tmp_path, unprepared=[confirm[7]])
    rec = qc_record(root)
    assert not rec['refusals'] and rec['reference_pool']['not_prepared'] == [confirm[7]]
    assert list(rec['excluded']) == [confirm[7]] and {r['category'] for r in rec['excluded'][confirm[7]]} == {
        'missing file'} and rec['excluded'][confirm[7]][0]['detail'] == 'no LPP preparation record'
    assert rec['completeness']['unrun'] == [] and rec['completeness']['systematic'] == []  # Not a never-run subject.
    out = run(root, accept_missing=f'{confirm[7]} failed lpp_prepare.verify_lengths')
    assert out['n_analyzed'] == 7 and out['reference_pool']['used'] == [s for s in list(dev) + list(confirm)
                                                                         if s != confirm[7]]
    root2, _, _ = confirm_root(tmp_path / 'b', mismatched=[confirm[7]])
    edit_cell(root2, confirm[0], '3', 'L3', 'EB', ledger=dict(references=sorted(x for x in list(dev) + list(confirm)
                                                                               if x != confirm[0])))
    rec = qc_record(root2)
    assert [(r['subject'], r['category']) for r in rec['refusals']] == [(confirm[0], 'wrong references')]


# --------------------------------------------------------------- statistics against independent references
D1_PASS = [0.012, -0.003, 0.009, 0.006, -0.004, 0.011, 0.003, 0.008, -0.002, 0.010, 0.005, -0.003]  # exact p 0.0183
D1_FAIL = [0.012, -0.004, 0.009, 0.006, -0.007, 0.011, 0.003, 0.008, -0.002, 0.010, 0.005, -0.003]  # exact p 0.0327
D6_PASS = [0.002, -0.007, -0.002, -0.001, -0.008, 0.003, 0.000, -0.006, -0.001, 0.004, -0.003, -0.007]  # p 0.0227
D6_FAIL = [0.002, -0.007, -0.002, -0.001, -0.008, 0.003, 0.000, -0.006, -0.003, 0.004, -0.003, -0.007]  # p 0.0283


def designed(d1, d6, section_sd=0.0):
    """Cell values whose per-subject means are exactly the given Δ: Δ1 = A − W1 = d1 (Δ2 = 0.01 + d1) at L3 and
    A − W1 = d6 at L8, every secondary strongly positive; per-(subject, section) wobble and an optional section
    effect on A are zero-mean over the 9 sections."""
    def value(i, s, t, budget, arm, endpoint):
        j = int(t) - 1
        common = 0.10 + 0.005 * i + 0.01 * ((7 * j) % 9 - 4) / 4
        wobble = 0.0005 * (((j + i) % 9) - 4)
        if endpoint == 'residualized':
            return common + dict(W0=0.0, W1=0.005, EB=0.02 + 0.001 * i).get(arm, 0.0)
        if budget == 'L8':
            return common + dict(W0=0.0, W1=0.04, EB=0.04 + d6[i] + wobble, ORACLE=0.06)[arm]
        m = re.fullmatch(r'EB_f(.+)', arm)
        if m:
            return common + 0.01 - 0.02 + 0.02 * float(m[1]) + 0.001 * i * float(m[1])
        if arm == 'EB':
            return common + 0.01 + d1[i] + section_sd * (j - 4) / 2.58 + wobble
        return common + dict(W0=0.0, W1=0.01, ORACLE=0.05, EB_SELF=0.0, TH_SEL=0.0)[arm]
    return value


def exact_p(d):
    """One-sided sign-flip p by exact enumeration of all 2^n flips (ties count, as in v4_stats)."""
    d = np.asarray(d, dtype=np.float64)
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
    return float(np.mean(signs @ d >= d.sum() - 1e-12))


MC_TOL = 3 * np.sqrt(0.025 * 0.975 / st.N_FLIPS)  # 3 Monte Carlo SEs at p = 0.025, about 1.5e-3.


@pytest.mark.parametrize('d1, d6, h1, h6', [(D1_PASS, D6_PASS, 'CONFIRMED', 'CONFIRMED'),
                                            (D1_FAIL, D6_PASS, 'NOT CONFIRMED', 'NOT TESTED'),
                                            (D1_PASS, D6_FAIL, 'CONFIRMED', 'NOT CONFIRMED')])
def test_mixed_sign_p_values_against_exact_enumeration(tmp_path, d1, d6, h1, h6):
    """Non-extreme p (mixed-sign Δ, n = 12): the H1 and H6 p against exact enumeration, and their verdicts."""
    root, confirm, values = confirm_root(tmp_path, n_ids=21, value=designed(d1, d6))
    out = run(root)
    r = {a: r_s(values, confirm, 'L3', a) for a in ('EB', 'W1', 'W0')}
    delta1 = r['EB'] - r['W1']
    assert np.allclose(delta1, d1, atol=1e-12) and 0.01 < exact_p(delta1) < 0.05 and delta1.mean() > 0
    c1, c2 = out['hypotheses']['H1']['contrasts']
    assert c1['p_one_sided'] == pytest.approx(exact_p(delta1), abs=MC_TOL)
    assert c2['p_one_sided'] == pytest.approx(exact_p(r['EB'] - r['W0']), abs=MC_TOL)
    assert (exact_p(delta1) < 0.025) == (h1 == 'CONFIRMED') == out['h1']  # Mean > 0 either way: p decides.
    assert out['verdicts']['H1'] == h1 and out['verdicts']['H6'] == h6
    if h6 != 'NOT TESTED':
        assert [out['verdicts'][h] for h in ('H2', 'H3', 'H4', 'H5')] == ['CONFIRMED'] * 4
        delta6 = r_s(values, confirm, 'L8', 'EB') - r_s(values, confirm, 'L8', 'W1')
        assert np.allclose(delta6, d6, atol=1e-12) and -0.005 < delta6.mean() < 0
        c6 = out['hypotheses']['H6']['contrasts'][0]
        assert c6['p_one_sided'] == pytest.approx(exact_p(delta6 + 0.005), abs=MC_TOL)
        assert c6['p_one_sided'] < 0.025 if h6 == 'CONFIRMED' else c6['p_one_sided'] > 0.025
        assert c6['mean'] == pytest.approx(delta6.mean()) and c6['null'] == -0.005


def test_h1_confirmed_but_lmm_restricts_the_claim(tmp_path):
    """Every listener gains, but the gain varies strongly over sections: H1 is CONFIRMED and the crossed LMM (section
    variance dominating) does not support generalization over sections, so the claim is restricted."""
    d1 = [0.01 + 0.0005 * (i - 4) for i in range(8)]
    root, confirm, values = confirm_root(tmp_path, value=designed(d1, D6_PASS[:8], section_sd=0.05))
    out = run(root)
    h1 = out['hypotheses']['H1']
    assert out['h1'] is True and all(c['n_positive'] == 8 for c in h1['contrasts'])
    for lmm in h1['lmm']:
        assert 'error' not in lmm and lmm['supports'] is False and lmm['var_section'] > lmm['var_subject']
        assert lmm['estimate'] == pytest.approx(np.mean(d1) + (0.01 if lmm['comparator'] == 'W0' else 0.0))
    assert h1['lmm_supports'] is False
    assert h1['claim'] == ('restricted to "across listeners of this book" (the crossed LMM does not support '
                           'generalization over sections)')
    assert 'across listeners of this book' in (root / sv.PRIMARY_MD).read_text()


# ------------------------------------------------------------------------------ the locked code in git
def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@t', *args],
                          capture_output=True, text=True, check=True)


def test_git_tree_compares_the_lock_with_the_committed_pushed_tree(tmp_path):
    if shutil.which('git') is None:
        pytest.skip('git is not installed')
    origin, repo = tmp_path / 'origin.git', tmp_path / 'repo'
    subprocess.run(['git', 'init', '-q', '--bare', str(origin)], check=True)
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    git(repo, 'symbolic-ref', 'HEAD', 'refs/heads/main')
    files = {'scripts/a.py': 'x = 1\n', 'experiments/PROTOCOL_V4.md': '# v4\n', '.gitignore': 'ref/\n',
             'ref/pinned.py': 'y = 2\n'}
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    git(repo, 'add', '-A')
    git(repo, 'commit', '-q', '-m', 'code')
    git(repo, 'remote', 'add', 'origin', str(origin))
    git(repo, 'push', '-q', 'origin', 'main')
    h = {k: sv.sha256_bytes(v.encode()) for k, v in files.items()}
    lock = dict(code_files={k: h[k] for k in ('scripts/a.py', 'experiments/PROTOCOL_V4.md', 'ref/pinned.py')},
                registration=dict(documents={'experiments/PROTOCOL_V4.md': h['experiments/PROTOCOL_V4.md']}))
    lock_path = repo / 'experiments/v4_lock.json'
    state = sv.git_tree(lock, lock_path, repo)
    assert state['problems'] == [] and state['lock_tracked'] is False and state['ignored'] == ['ref/pinned.py']
    (repo / 'experiments/PROTOCOL_V4.md').write_text('# v4\n\n## 4. item 17\n')  # A §4 entry after the lock.
    assert sv.git_tree(lock, lock_path, repo)['problems'] == []
    (repo / 'scripts/a.py').write_text('x = 2\n')  # Edited, not committed: the running code is not the lock's.
    assert sv.git_tree(lock, lock_path, repo)['problems'] == ['scripts/a.py: the running file differs from the lock']
    (repo / 'scripts/a.py').write_text(files['scripts/a.py'])
    (repo / 'ref/pinned.py').write_text('y = 3\n')
    assert sv.git_tree(lock, lock_path, repo)['problems'] == ['ref/pinned.py: the running file differs from the lock']
    (repo / 'ref/pinned.py').write_text(files['ref/pinned.py'])
    (repo / 'scripts/b.py').write_text('z = 0\n')  # Pinned by a lock written from an rsynced, uncommitted tree.
    stale = dict(lock, code_files=dict(lock['code_files'], **{'scripts/b.py': sv.sha256_file(repo / 'scripts/b.py')}),
                 registration=dict(documents={'experiments/PROTOCOL_V4.md': '0' * 64}))
    problems = sv.git_tree(stale, lock_path, repo)['problems']
    assert any(p.startswith('scripts/b.py is not committed') for p in problems)
    assert any(p.startswith('experiments/PROTOCOL_V4.md at ') and p.endswith('differs from the lock') for p in problems)
    git(repo, 'add', 'scripts/b.py')
    git(repo, 'commit', '-q', '-m', 'unpushed')
    assert sv.git_tree(lock, lock_path, repo)['problems'][-1].endswith('is not on origin/main')
    git(repo, 'push', '-q', 'origin', 'main')
    lock_path.write_text(json.dumps(lock))
    git(repo, 'add', 'experiments/v4_lock.json')
    git(repo, 'commit', '-q', '-m', 'lock')
    git(repo, 'push', '-q', 'origin', 'main')
    state = sv.git_tree(lock, lock_path, repo)
    assert state['lock_tracked'] is True and state['problems'] == []
    assert state['commit'] == git(repo, 'rev-parse', 'HEAD').stdout.strip()
    with pytest.raises(sv.Refused, match='not the committed, pushed tree'):
        sv.check_tree(stale, lock_path, sv.git_tree(stale, lock_path, repo))


def test_primary_refuses_unverified_code(tmp_path):
    root, _, _ = confirm_root(tmp_path)
    with pytest.raises(sv.Refused, match='written without its git check'):
        run(root, tree=None)  # No git here, and the lock was written with --no-git-check.
    with pytest.raises(sv.Refused, match='not the committed, pushed tree'):
        run(root, tree=dict(TREE_OK, problems=['scripts/summarize_v4.py at 0123 differs from the lock']))
    assert not (root / sv.STARTED).exists()
    lock = json.loads((root / sv.LOCK).read_text())
    (root / sv.LOCK).write_text(json.dumps(dict(lock, git_check='verified')))
    out = run(root, tree=None)
    assert out['code_tree']['note'].startswith('git unavailable')


def test_cell_changed_after_qc_is_refused(tmp_path):
    root, confirm, _ = confirm_root(tmp_path)
    lock, pl, rec = sv.run_qc(root, fingerprints=FP, audit=ok_audit, write=False)
    edit_cell(root, confirm[0], '1', 'L3', 'EB', test={VAL: 0.9})
    scores = sv.Scores(root, 'dense', 'evaluate', rec['files'])
    samples = {k: v['included'] for k, v in rec['samples'].items()}
    with pytest.raises(sv.Refused, match='changed after QC'):
        sv.analyze(scores, pl, samples, rec['oracle_subjects'])


def test_default_audit_uses_the_writers_audit(tmp_path):
    from src.priors import tanghuth as th
    dev = bud.lpp_partition(ids(lpp_prepare.N_EN))[0]
    xdec = tmp_path / 't3_L3_XDEC.json'
    save(xdec, lpp_cell(dev[0], '3', 'L3', 'XDEC', 'develop', 0.05, None, dev, logs=True))
    w0 = tmp_path / 't3_L3_W0.json'
    save(w0, lpp_cell(dev[0], '3', 'L3', 'W0', 'develop', 0.05, 0.02, dev, logs=True))
    assert not audit_v4.audit_file(xdec)['passed'] and th.audit_file(xdec)['passed']
    assert sv.default_audit(xdec)['passed']
    assert sv.default_audit(w0) == audit_v4.audit_file(w0) and sv.default_audit(w0)['passed']
    sel = tmp_path / 't3_L3_TH_SEL.json'
    save(sel, lpp_cell(dev[0], '3', 'L3', 'TH_SEL', 'develop', 0.05, 0.02, dev, logs=True))
    assert th.audit_file(sel)['passed'] and sv.default_audit(sel) == th.audit_file(sel)
    once = json.loads(sel.read_text())  # A selection computed once on B and reused in every fold.
    once['provenance']['records'] = [r for r in once['provenance']['records']
                                     if not (r['purpose'] == 'converter' and r['fold'] is not None)]
    save(sel, once)
    assert audit_v4.audit_file(sel)['passed'] and not sv.default_audit(sel)['passed']  # Module H's own checks.


# ------------------------------------------------------------------------------------------- the lock
def test_lock_preconditions(tmp_path):
    root, confirm, _ = confirm_root(tmp_path)
    lock = json.loads((root / sv.LOCK).read_text())
    assert sv.check_lock(lock, FP)['advanced'] == 'EB'
    with pytest.raises(sv.Refused, match='changed since the lock'):
        sv.check_lock(lock, dict(FP, summarize_sha256='9' * 64))
    with pytest.raises(sv.Refused, match='changed since the lock'):
        sv.check_lock(lock, dict(FP, tanghuth_runner_sha256='0' * 64))
    with pytest.raises(sv.Refused, match='changed since the lock'):
        sv.check_lock({k: v for k, v in lock.items() if k != 'lpp_cell_code_sha256'}, FP)
    with pytest.raises(sv.Refused, match='confirmatory plan differs'):
        sv.check_lock(dict(lock, drop_h6=True), FP)
    with pytest.raises(sv.Refused, match='lacks'):
        sv.check_lock({k: v for k, v in lock.items() if k != 'confirmatory_plan'}, FP)
    with pytest.raises(sv.Refused, match='no LPP configuration hash'):
        sv.check_lock(dict(lock, lpp_config_sha256=None), FP)
    with pytest.raises(sv.Refused, match='registers no confirmatory analysis'):
        sv.check_lock(dict(lock, confirmatory=False, confirm_status=dict(status='reserved', reasons=['null'])), FP)
    dev, confirm = lock['lpp_partition']['dev'], lock['lpp_partition']['confirm']
    swapped = dict(lock, lpp_partition=dict(dev=confirm[:1] + dev[1:], confirm=dev[:1] + confirm[1:]))
    with pytest.raises(sv.Refused, match='hash partition'):
        sv.check_lock(swapped, FP)
    with pytest.raises(sv.Refused, match='pushed'):
        sv.run_primary(root, fingerprints=FP, audit=ok_audit,
                       git_state=dict(tracked=True, clean=True, pushed=False, commit='x'))
    with pytest.raises(sv.Refused, match='git is unavailable'):
        sv.run_primary(root, fingerprints=FP, audit=ok_audit, git_state=None)  # No silent skip.
    (root / sv.LOCK).unlink()
    with pytest.raises(sv.Refused, match='does not exist'):
        run(root)


# ------------------------------------------------------------------------------- development fixture
DS_EFFECT = dict(S3=dict(W0=0.0, W1=0.004, EB=0.012, PRIOR=0.006, ORACLE=0.05),
                 ALL=dict(W0=0.0, W1=0.05, EB=0.051, PRIOR=0.03))
DS_NULL = dict(S3=dict(W0=0.0, W1=0.004, EB=0.005, PRIOR=0.003, ORACLE=0.05),
               ALL=dict(W0=0.0, W1=0.05, EB=0.051, PRIOR=0.03))
G1_RHO = dict(S3=[0.22, 0.47, 0.43, 0.14, 0.18, 0.09, 0.01, 0.22, 0.24],
              S6=[0.48, 0.62, 0.62, 0.18, 0.21, 0.14, 0.00, 0.32, 0.28])
LPP_DEV_EFFECT = dict(W0=0.0, W1=0.006, EB=0.01, PRIOR=0.008, ORACLE=0.03, EB_SELF=0.004, TH_SEL=0.003, XDEC=-0.01)


def ds_value(i, budget, arm, effect=DS_EFFECT):
    return 0.12 + 0.01 * i + effect[budget][arm] * (1 + 0.1 * (i - 4) / 4)


def gates_for(subjects, fp=FP, effect=DS_EFFECT):
    """results/v4/gates.json as scripts/v4_gates.py computes it."""
    def deltas(budget, a, b):
        return np.array([ds_value(i, budget, a, effect) - ds_value(i, budget, b, effect)
                         for i in range(len(subjects))])
    g1 = {}
    for name, rho in G1_RHO.items():
        rho = np.array(rho)
        g1[name] = dict(spearman=rho.tolist(), median=float(np.median(rho)), above_010=int((rho > 0.10).sum()),
                        passed=bool(np.median(rho) >= 0.20 and (rho > 0.10).sum() >= 7))
    g1.update(primary='S3', lpp_primary='L3', decision='primary budget S3')
    d = deltas('S3', 'ORACLE', 'W0')
    g2 = {}
    for arm in ('EB', 'PRIOR'):
        d1, d2 = deltas('S3', arm, 'W1'), deltas('S3', arm, 'W0')
        g2[arm] = dict(mean_minus_w1=float(d1.mean()), mean_minus_w0=float(d2.mean()),
                       positive_vs_w0=int((d2 > 0).sum()),
                       passed=bool(d1.mean() >= 0.005 and d2.mean() >= 0.003 and (d2 > 0).sum() >= 7))
    d3 = deltas('ALL', 'EB', 'W1')
    g3 = dict(mean=float(d3.mean()), worst=float(d3.min()), passed=bool(d3.mean() >= -0.003 and d3.min() >= -0.010))
    advanced = 'EB' if g2['EB']['passed'] else 'PRIOR' if g2['PRIOR']['passed'] else None
    drop = None if advanced is None else (advanced == 'PRIOR' or not g3['passed'])
    return dict(code_sha256=fp['code_sha256'], G1=g1, G_FEAS=dict(budget='S3', mean=float(d.mean()),
                                                                   passed=bool(d.mean() >= 0.005)),
                G2=g2, G3=g3, advanced=advanced, drop_h6=drop, decision=f'advance {advanced}')


def lpp_dev_value(i, t, budget, arm):
    m = re.fullmatch(r'(EB|PRIOR)_f(.+)', arm)
    effect = 0.006 + 0.004 * float(m[2]) if m else LPP_DEV_EFFECT[arm]
    return 0.08 + 0.004 * i + 0.002 * int(t) + (0.01 if budget == 'L8' else 0.0) + effect * (1 + 0.05 * i)


def ds_cell_dict(s, clock, budget, arm, v, i, fp=FP, **extra):
    cell = dict(protocol='ssltang-v4', stage='develop', subject=s, clock=clock, budget=budget, arm=arm,
                estimable=True, code_sha256=fp['code_sha256'], config_sha256=fp['config_sha256'],
                val={VAL: v, 'by_run_feature_correlation': {'storyA': v, 'storyB': v + 0.001 * (i - 4),
                                                            'storyC': v - 0.002}},
                ledger=dict(references=[x for x in bud.SUBJECTS if x != s], target_pairs_dense=1000))
    cell.update(extra)
    return cell


def tanghuth_cell(s, budget, arm, v, fp=FP, refs=('UTS01', 'UTS02', 'UTS03')):
    """A Module H ds003020 cell (run_v4_tanghuth.code_fields)."""
    return dict(protocol='ssltang-v4', block='published-method', stage='develop', subject=s, clock='dense',
                budget=budget, arm=arm, regime='S', estimable=True,
                references=list(refs) if arm != 'TH_SEL_rand' else [], val={VAL: v},
                code_sha256=fp['tanghuth_cell_code_sha256'], core_sha256=fp['code_sha256'],
                tanghuth_sha256=fp['tanghuth_sha256'], tanghuth_runner_sha256=fp['tanghuth_runner_sha256'],
                config_sha256=fp['config_sha256'])


def dev_root(tmp, fp=FP, logs=False, effect=DS_EFFECT, advanced='EB', drop_h6=False):
    """Development artifacts for the lock: G0, gates, g1, ds003020 (run_v4 and Module H) cells, LPP DEV cells for
    the whole plan (plus XDEC), G4 with its per-listener records and cells, and the LPP preparation record."""
    root = Path(tmp)
    dev, confirm = bud.lpp_partition(ids(lpp_prepare.N_EN))
    core = fp['code_sha256']
    save(root / 'results/v4/g0.json', dict(passed=True, code_sha256=core))
    save(root / 'results/v4/gates.json', gates_for(sv.DS_DEV_SUBJECTS, fp, effect))
    for i, s in enumerate(sv.DS_DEV_SUBJECTS):
        save(root / 'results/v4' / s / 'g1.json', dict(stage='g1', subject=s, code_sha256=core, budgets={
            name: dict(spearman=G1_RHO[name][i]) for name in ('S3', 'S6')}))
        for budget, arms in effect.items():
            for arm in arms:
                weights = dict(eb={d: eb_info(kappa=0.3) for d in ('1', '2', '3')})
                extra = dict(final_weights=weights) if arm == 'EB' else {}
                save(sv.ds_cell(root, s, 'dense', 'develop', budget, arm),
                     ds_cell_dict(s, 'dense', budget, arm, ds_value(i, budget, arm, effect), i, fp, **extra))
    for arm in ('TH_SEL', 'TH_SEL_rand', 'XDEC'):
        save(sv.ds_cell(root, 'UTS05', 'dense', 'develop', 'S3', arm), tanghuth_cell('UTS05', 'S3', arm, 0.1, fp))
    save(root / 'results/v4/UTS05/dense/develop/sensitivity/refs-UTS01-UTS02/cells/S6_TH_SEL.json',
         tanghuth_cell('UTS05', 'S6', 'TH_SEL', 0.1, fp, refs=('UTS01', 'UTS02')))
    pl = sv.plan(advanced, drop_h6, 'L3')
    cells = family_cells(pl) + [('L3', 'XDEC')]
    for i, s in enumerate(dev):
        for t in SECTIONS:
            for budget, arm in cells:
                v = lpp_dev_value(i, t, budget, arm)
                eb = eb_info() if arm == 'EB' else None
                save(sv.lpp_cell(root, s, 'dense', 'develop', t, budget, arm),
                     lpp_cell(s, t, budget, arm, 'develop', v, v / 2, dev, fp, logs=logs, eb=eb))
            save(root / 'results/v4_lpp' / s / 'dense/g4/cells' / f't{t}_L8_W0.json',
                 lpp_cell(s, t, 'L8', 'W0', 'develop', 0.05, None, dev, fp, logs=logs))
        for k in vl.RP_K_LEVELS[:-1]:  # The R_P(K) subset arms (one draw), R_P = K / (8 + K): rho1 = 1/9.
            arm = f'EB_K{k}_d0'
            for t in SECTIONS:
                cell = lpp_cell(s, t, 'L3', arm, 'develop', lpp_dev_value(i, t, 'L3', 'EB'), None, dev, fp,
                                logs=logs, eb=eb_info(var_p=k / (8 + k)))
                cell['ledger']['references'] = sorted(x for x in dev if x != s)[:k]
                save(sv.lpp_cell(root, s, 'dense', 'develop', t, 'L3', arm), cell)
    runs = {s: [dict(section=k, run_number=k, n_tr=n) for k, n in zip(range(1, 10), sv.SECTION_TRS)] for s in dev}
    save(root / 'data/lpp/prepare_provenance.json',
         dict(mode='DEV', partition=dict(dev=dev, confirm=confirm), code=PREP_CODE, created_utc='2026-10-04T00:00:00',
              mask=dict(content_sha256=MASK_SHA, mask_voxels=100, columns=100, block_size=1),
              g4a=dict(passed=True, runs=runs), inputs={}, outputs={}, output_content={}, versions={}))
    write_g4(root, dev, fp)
    return root, dev, confirm


def write_g4(root, dev, fp=FP, failing_b=(), edges=()):
    """Module G's G4 records: g4b per DEV listener (``failing_b`` below their null q95), g4c (only if (b) passed;
    ``edges`` with one ISP alpha at the grid edge) and g4.json from run_v4_lpp.g4_verdict."""
    import run_v4_lpp
    code = fp['lpp_cell_code_sha256']
    b = {}
    for s in dev:
        b[s] = dict(protocol='ssltang-v4', stage='g4b', subject=s, clock='dense', budget='L8', arm='W0',
                    code_sha256=code, observed=0.01 if s in failing_b else 0.05, null_q95=0.02,
                    passed=s not in failing_b)
        save(root / 'results/v4_lpp' / s / 'dense/g4/g4b.json', b[s])
    c = None
    if run_v4_lpp.g4_verdict(b)['b']['passed']:
        c = {}
        for s in dev:
            fits = [dict(test_section=t, budget=name, fold=i, alpha=1e3, at_edge=s in edges and t == '1' and i is None
                         and name == 'L3') for t, name, i in sorted(vl.g4_fits_expected(), key=str)]
            edge = sum(f['at_edge'] for f in fits)
            c[s] = dict(protocol='ssltang-v4', stage='g4c', subject=s, code_sha256=code, sections=list(SECTIONS),
                        n_fits=len(fits), n_at_edge=edge, passed=edge == 0, fits=fits)
            save(root / 'results/v4_lpp' / s / 'g4c.json', c[s])
    save(root / 'results/v4_lpp/g4.json', dict(protocol='ssltang-v4', stage='g4', code_sha256=code, dev=list(dev),
                                               **run_v4_lpp.g4_verdict(b, c)))


def build(root, **kw):
    kw = dict(dict(fingerprints=FP, lpp_code=PREP_CODE, audit=ok_audit, git=None, no_git_check=True), **kw)
    return vl.build_lock(root, **kw)


@pytest.fixture(scope='module')
def dev_template(tmp_path_factory):
    """One development root, copied per test (the fixture writes thousands of small files)."""
    root, dev, confirm = dev_root(tmp_path_factory.mktemp('dev_template'))
    return root, dev, confirm


@pytest.fixture
def dev_copy(dev_template, tmp_path):
    import shutil
    template, dev, confirm = dev_template
    root = tmp_path / 'root'
    shutil.copytree(template, root)
    return root, dev, confirm


def test_lock_written_with_registered_contents(dev_copy):
    root, dev, confirm = dev_copy
    lock = build(root)
    digest = vl.write_lock(lock, root)
    saved = json.loads((root / sv.LOCK).read_text())
    assert digest == sv.sha256_file(root / sv.LOCK) and saved == lock
    assert saved['confirmatory'] is True and saved['confirm_status'] == dict(status='registered', reasons=[])
    assert (saved['advanced_arm'], saved['drop_h6'], saved['primary_budget']) == ('EB', False, dict(ds003020='S3',
                                                                                                   lpp='L3'))
    for k, v in FP.items():
        assert saved[k] == v
    assert lpp_prepare.check_lock(saved, dev, confirm, PREP_CODE) == MASK_SHA  # What lpp_prepare --confirm checks.
    assert sv.check_lock(saved, FP) == sv.clean(sv.plan('EB', False, 'L3'))  # What summarize_v4 --confirm checks.
    assert saved['readers'] == {'lpp_prepare.check_lock': 'accepts', 'summarize_v4.check_lock': 'accepts'}
    per_section = len(family_cells(sv.plan('EB', False, 'L3'))) + 1 + 3  # The family, XDEC and EB_K1/2/4_d0.
    counts = saved['development_cells']['counts']
    assert counts == dict(ds003020=9 * 9 + 3, ds003020_sensitivity=1, lpp_dev=9 * 9 * per_section,
                          lpp_dev_sensitivity=0, lpp_dev_controls=0, lpp_g4=9 * 9)  # Runner-written pair cells and
    # control records: tests/test_v4_lpp_runner.py::test_module_i_locks_and_reads_the_pair_cells_and_control_records.
    assert saved['lpp_config_sha256'] == LPP_CONFIG and saved['git_check'] == 'skipped (--no-git-check)'
    rel = sv.lpp_cell(root, dev[0], 'dense', 'develop', '1', 'L3', 'EB').relative_to(root).as_posix()
    assert saved['development_cells']['lpp_dev'][rel] == sv.sha256_file(root / rel)
    assert f'results/v4_lpp/{dev[0]}/dense/g4/cells/t1_L8_W0.json' in saved['development_cells']['lpp_g4']
    assert 'results/v4/UTS05/dense/develop/cells/S3_XDEC.json' in saved['development_cells']['ds003020']
    assert set(saved['gate_files']) >= {'results/v4/g0.json', 'results/v4/gates.json', 'results/v4_lpp/g4.json',
                                        f'results/v4_lpp/{dev[0]}/dense/g4/g4b.json',
                                        f'results/v4_lpp/{dev[0]}/g4c.json'}
    assert saved['lpp_dev_plan_check']['subjects'] == dev and saved['lpp_dev_plan_check']['excluded'] == 0
    assert saved['code_files']['scripts/run_v4.py'] == sv.sha256_file(ROOT / 'scripts/run_v4.py')
    assert set(saved['registration']['documents']) == set(vl.DOCS)
    eff = saved['development_effects']
    d_ds = np.array([ds_value(i, 'S3', 'EB') - ds_value(i, 'S3', 'W1') for i in range(9)])
    assert eff['ds003020']['minus_w1']['dz'] == pytest.approx(st.dz(d_ds))
    assert eff['ds003020']['minus_w1']['sd'] == pytest.approx(d_ds.std(ddof=1))
    d_lpp = np.array([np.mean([lpp_dev_value(i, t, 'L3', 'EB') - lpp_dev_value(i, t, 'L3', 'W0') for t in SECTIONS])
                      for i in range(9)])
    assert eff['lpp']['minus_w0']['dz'] == pytest.approx(st.dz(d_lpp)) and eff['lpp']['subjects'] == dev
    rp = saved['rp_k']  # R_P = b^2 Var(p) / (Var z - sigma^2) = K / (8 + K): 0.5 at K = 8.
    assert rp['by_k']['8']['mean'] == pytest.approx(0.5) and rp['rho1'] == pytest.approx(1 / 9)
    assert rp['by_k']['1']['mean'] == pytest.approx(1 / 9) and rp['by_k']['4']['n_subjects'] == 9
    assert rp['n_k_levels'] == 4 and rp['gaps'] == [] and rp['deviation'] is None and saved['deviation'] is None
    assert rp['prediction'] == pytest.approx(48 / 56) and rp['predict_k'] == 48
    g4 = saved['gates']['lpp_g4']
    assert g4['verified'] is True and g4['recomputed']['n_passed'] == 9 and g4['recomputed']['c'] is True
    kap = saved['kappa_schedule']
    assert kap['ds003020'][0]['kappa'] == pytest.approx(0.3) and kap['lpp'][0]['reliability']['R_P']['mean'] == 0.5
    seeds = saved['hash_orders_and_seeds']
    assert seeds['statistics']['sign_flip'] == dict(seed=20261003, n=100000)
    assert seeds['statistics']['bca'] == dict(seed=20261004, n=10000)
    assert seeds['lpp_section_budgets']['1'] == bud.lpp_section_budgets('1', SECTIONS)
    assert set(seeds['ladder']['seeds']) == set(bud.SUBJECTS) | set(dev) | set(confirm)
    with pytest.raises(vl.Refused, match='written once'):
        vl.write_lock(lock, root)


def edit(root, rel, drop=(), **changes):
    path = Path(root) / rel
    data = json.loads(path.read_text())
    data.update(changes)
    for key in drop:
        data.pop(key)
    path.write_text(json.dumps(data))


def _gates(root, **changes):
    edit(root, 'results/v4/gates.json', **changes)


def _g2_wrong(root):
    gates = json.loads((root / 'results/v4/gates.json').read_text())
    gates['G2']['EB']['mean_minus_w1'] += 1e-6
    (root / 'results/v4/gates.json').write_text(json.dumps(gates))


def _g1_contradiction(root):
    gates = json.loads((root / 'results/v4/gates.json').read_text())
    gates['G1'].update(primary='S6', lpp_primary='L5')  # S3 passed: S6 would be a post-hoc budget choice.
    gates['G_FEAS']['budget'] = 'S6'
    (root / 'results/v4/gates.json').write_text(json.dumps(gates))


def _cell(root, **changes):  # A cell the gate verification does not read (ALL W0).
    edit(root, 'results/v4/UTS02/dense/develop/cells/ALL_W0.json', **changes)


def _dev(i=3):
    return bud.lpp_partition(ids(lpp_prepare.N_EN))[0][i]


def _lpp(root, arm='W0', t='7', budget='L3', drop=(), **changes):
    edit(root, sv.lpp_cell(root, _dev(), 'dense', 'develop', t, budget, arm).relative_to(root), drop, **changes)


def _devs():
    return list(bud.lpp_partition(ids(lpp_prepare.N_EN))[0])


def _g4_claims_pass(root):
    write_g4(root, _devs(), edges=_devs()[2:3])
    g4 = json.loads((root / 'results/v4_lpp/g4.json').read_text())
    g4['c']['passed'] = g4['passed'] = True
    (root / 'results/v4_lpp/g4.json').write_text(json.dumps(g4))


def _g4c_drop_fit(root):
    rel = f'results/v4_lpp/{_dev()}/g4c.json'
    rec = json.loads((root / rel).read_text())
    rec['fits'] = rec['fits'][1:]
    rec['n_fits'] -= 1
    (root / rel).write_text(json.dumps(rec))


def _drop_subsets(root):
    for path in (root / 'results/v4_lpp').glob('*/dense/develop/cells/t*_L3_EB_K*_d0.json'):
        path.unlink()


def _confirm_data(root):
    s = bud.lpp_partition(ids(lpp_prepare.N_EN))[1][0]
    (root / 'data/lpp/raw' / s).mkdir(parents=True)


def _drop_key(root, rel, key):
    edit(root, rel, drop=[key])


SENSITIVITY = 'results/v4/UTS05/dense/develop/sensitivity/refs-UTS01-UTS02/cells/S6_TH_SEL.json'
REFUSALS = {
    'g0 missing': (lambda r: (r / 'results/v4/g0.json').unlink(), 'g0.json does not exist'),
    'g0 failed': (lambda r: edit(r, 'results/v4/g0.json', passed=False), 'did not pass'),
    'g0 other code': (lambda r: edit(r, 'results/v4/g0.json', code_sha256='0' * 64), 'g0.json is not from'),
    'gates other code': (lambda r: _gates(r, code_sha256='0' * 64), 'gates.json is not from'),
    'nothing advanced (contradicts G2)': (lambda r: _gates(r, advanced=None, drop_h6=None), 'G2/G3 give'),
    'table mismatch': (lambda r: _gates(r, drop_h6=True), 'G2/G3 give'),
    'g1 primary contradicts its verdicts': (_g1_contradiction, 'contradicts the G1 verdicts'),
    'feas incomplete': (lambda r: _gates(r, G_FEAS=dict(budget='S3')), 'G-FEAS has not been computed'),
    'feas numbers differ': (lambda r: _gates(r, G_FEAS=dict(budget='S3', mean=0.0, passed=False)),
                            'does not match the development files'),
    'gates stale': (_g2_wrong, 'does not match the development files'),
    'ds cell other code': (lambda r: _cell(r, code_sha256='0' * 64), 'not the current run_v4 core'),
    'ds cell other config': (lambda r: _cell(r, config_sha256='0' * 64), 'not the current configuration'),
    'ds cell without config': (lambda r: _drop_key(r, 'results/v4/UTS02/dense/develop/cells/ALL_W0.json',
                                                   'config_sha256'), 'not the current configuration'),
    'ds cell wrong path': (lambda r: _cell(r, arm='W1'), 'differ from the path'),
    'gate cell other code': (lambda r: edit(r, 'results/v4/UTS02/dense/develop/cells/S3_W1.json',
                                            code_sha256='0' * 64), 'is not from the current code'),
    'module H cell under the bare core': (lambda r: edit(r, 'results/v4/UTS05/dense/develop/cells/S3_TH_SEL.json',
                                                         code_sha256=FP['code_sha256']), 'current Module H'),
    'module H cell stale tanghuth.py': (lambda r: edit(r, 'results/v4/UTS05/dense/develop/cells/S3_XDEC.json',
                                                       tanghuth_sha256='0' * 64), 'Module H fields'),
    'sensitivity references differ': (lambda r: edit(r, SENSITIVITY, references=['UTS01', 'UTS03']),
                                      "not the path's"),
    'lpp cell other code': (lambda r: _lpp(r, code_sha256='0' * 64), 'cannot be locked'),
    'lpp configs differ': (lambda r: _lpp(r, config_sha256='0' * 64), 'configuration hashes'),
    'lpp config missing': (lambda r: _lpp(r, drop=['config_sha256']), 'configuration hashes'),
    'lpp wrong references': (lambda r: _lpp(r, arm='EB', ledger=dict(references=[_dev(0)])), 'wrong references'),
    'lpp section from provenance': (lambda r: _lpp(r, provenance=lpp_log(_dev(), '2', 'L3', 'W0', 'develop')),
                                    "provenance test_items"),
    'lpp section field': (lambda r: _lpp(r, test_section='2'), 'corrupt file'),
    'g4 cell other code': (lambda r: edit(r, f'results/v4_lpp/{_dev()}/dense/g4/cells/t2_L8_W0.json',
                                          code_sha256='0' * 64), 'cannot be locked'),
    'g4b record missing': (lambda r: (r / f'results/v4_lpp/{_dev()}/dense/g4/g4b.json').unlink(),
                           'g4b.json does not exist'),
    'g4c record missing': (lambda r: (r / f'results/v4_lpp/{_dev()}/g4c.json').unlink(), 'g4c.json does not exist'),
    'g4 verdict contradicts its records': (lambda r: edit(r, 'results/v4_lpp/g4.json', passed=False),
                                           r"does not match the per-listener G4 records on disk: \['passed'\]"),
    'g4 claims a pass its records fail': (_g4_claims_pass, "does not match the per-listener G4 records"),
    'g4 dev list': (lambda r: edit(r, 'results/v4_lpp/g4.json', dev=['sub-x']), r"records on disk: \['dev'\]"),
    'g4b inconsistent': (lambda r: edit(r, f'results/v4_lpp/{_dev()}/dense/g4/g4b.json', observed=0.0),
                         'records passed=True, but observed 0.0'),
    'g4b other code': (lambda r: edit(r, f'results/v4_lpp/{_dev()}/dense/g4/g4b.json', code_sha256='0' * 64),
                       'g4b.json is not from the current LPP code'),
    'g4c inconsistent': (lambda r: edit(r, f'results/v4_lpp/{_dev()}/g4c.json', n_at_edge=1),
                         'g4c.json records passed=True, n_at_edge=1'),
    'g4c incomplete fits': (_g4c_drop_fit, 'registered ISP fits'),
    'g4c sections': (lambda r: edit(r, f'results/v4_lpp/{_dev()}/g4c.json', sections=['1']), 'not all 9'),
    'g4 cells incomplete': (lambda r: (r / f'results/v4_lpp/{_dev()}/dense/g4/cells/t5_L8_W0.json').unlink(),
                            'G4\\(b\\) cells .* are incomplete'),
    'rp_k one level': (_drop_subsets, 'not the registered K <= 8 fit'),
    'dev plan: TH_SEL missing': (lambda r: sv.lpp_cell(r, _dev(), 'dense', 'develop', '4', 'L3', 'TH_SEL').unlink(),
                                 'do not exercise the locked confirmatory plan'),
    'dev plan: residualized missing': (lambda r: _lpp(r, arm='W1', drop=['residualized']), 'no residualized score'),
    'dev plan: ladder not estimable': (lambda r: _lpp(r, arm='EB_f0.25', estimable=False), 'arm not estimable'),
    'dev plan: L8 oracle missing': (lambda r: sv.lpp_cell(r, _dev(), 'dense', 'develop', '4', 'L8',
                                                          'ORACLE').unlink(), 'no complete ORACLE/W0'),
    'prep other code': (lambda r: edit(r, 'data/lpp/prepare_provenance.json', code={'x': 'y'}), 'other code'),
    'prep incomplete': (lambda r: _drop_key(r, 'data/lpp/prepare_provenance.json', 'output_content'), 'lacks'),
    'g4 failed': (lambda r: write_g4(r, _devs(), failing_b=_devs()[:4]), 'only with --reserve-confirm'),
    'g4 failed at (c)': (lambda r: write_g4(r, _devs(), edges=_devs()[2:3]), 'only with --reserve-confirm'),
    'g4 missing': (lambda r: (r / 'results/v4_lpp/g4.json').unlink(), 'LPP G4 has not been run'),
    'g4 without fingerprint': (lambda r: _drop_key(r, 'results/v4_lpp/g4.json', 'code_sha256'),
                               'not from the current LPP code'),
    'confirm data exists': (_confirm_data, 'before the lock'),
    'test cell exists': (lambda r: save(r / 'results/v4/UTS05/dense/evaluate/cells/S3_W0.json', {}), 'before the lock'),
    'lpp evaluate output exists': (lambda r: save(r / f'results/v4_lpp/{_dev()}/dense/evaluate/cells/x.json', {}),
                                   'before the lock'),
    'lock exists': (lambda r: save(r / sv.LOCK, {}), 'before the lock'),
}


@pytest.mark.parametrize('case', sorted(REFUSALS))
def test_lock_refusals(dev_copy, case):
    root, _, _ = dev_copy
    mutate, message = REFUSALS[case]
    mutate(root)
    with pytest.raises(vl.Refused, match=message):
        build(root)
    assert not (root / sv.LOCK).exists() or case == 'lock exists'


def test_lock_refuses_audit_failures_git_and_a_changed_mask(dev_copy):
    root, _, _ = dev_copy
    with pytest.raises(vl.Refused, match='provenance audit failed'):
        build(root, audit=lambda p: dict(passed='UTS07' not in str(p), rules=['R7']))
    with pytest.raises(vl.Refused, match='audit failed'):
        build(root, audit=lambda p: dict(passed='t3_L3_EB_SELF' not in str(p), rules=['R7']))
    with pytest.raises(vl.Refused, match='not committed'):
        build(root, git=dict(head='abc', dirty=['scripts/run_v4_lpp.py'], ignored=[]))
    with pytest.raises(vl.Refused, match='git is unavailable'):
        build(root, no_git_check=False)
    assert build(root, git=dict(head='abc', dirty=[], ignored=[]), no_git_check=False)['git_check'] == 'verified'
    with pytest.raises(vl.Refused, match='fingerprints missing'):
        build(root, fingerprints={k: v for k, v in FP.items() if k != 'lpp_cell_code_sha256'})
    with pytest.raises(vl.Refused, match='does not apply'):
        build(root, reserve_confirm=True)
    rng = np.random.default_rng(0)
    state = dict(mask=rng.random((4, 4, 4)) > 0.5, membership=np.arange(5), coords=np.ones((5, 3), dtype=np.int64),
                 affine=np.eye(4), block_size=np.array(1), grid_shape=np.array([4, 4, 4]))
    np.savez(root / 'data/lpp/mask.npz', **state)
    with pytest.raises(vl.Refused, match='mask content digest'):
        build(root)
    edit(root, 'data/lpp/prepare_provenance.json',
         mask=dict(content_sha256=lpp_prepare.state_digest(state), mask_voxels=3, columns=5, block_size=1))
    lock = build(root)
    assert lock['lpp_preparation']['mask_npz_verified'] is True
    assert lock['lpp_mask_sha256'] == lpp_prepare.state_digest(state)


def test_lock_rp_k_without_its_k_levels_needs_a_deviation(dev_copy):
    root, dev, _ = dev_copy
    _drop_subsets(root)
    with pytest.raises(vl.Refused, match=r'extrapolation from 1 K level\(s\)'):
        build(root)
    lock = build(root, deviation='PROTOCOL_V4 §4 item 97: EB_K subsets dropped (Tier 2)')
    rp = lock['rp_k']
    assert rp['n_k_levels'] == 1 and rp['deviation'] == lock['deviation'] and len(rp['gaps']) == 3
    assert rp['prediction'] == pytest.approx(48 / 56)  # The one-point solve, now recorded as a deviation.
    for path in (root / 'results/v4_lpp' / dev[0] / 'dense/develop/cells').glob('t*_L3_EB.json'):
        path.unlink()  # K = 8 incomplete for one listener: a gap too.
    assert 'K = 8: 8 of 9 DEV listeners' in vl.rp_k_gaps(sv.reliability_fit(
        sv.lpp_table(root, dev, 'develop', FP['lpp_cell_code_sha256'])[1], 'dense', 'L3'), dev)


def test_g4_records_recompute_module_gs_verdict(dev_copy):
    """g4_records reproduces run_v4_lpp.g4_verdict on the same records (a pass, a (b) failure, a (c) failure)."""
    import run_v4_lpp
    root, dev, _ = dev_copy
    assert vl.G4_MIN_PASS == run_v4_lpp.G4_MIN_PASS
    assert vl.RP_K_LEVELS == tuple(run_v4_lpp.SUBSET_K['DEV']) + (len(dev) - 1,)
    for failing, edges in (((), ()), (dev[:3], ()), (dev[:4], ()), ((), dev[1:2])):
        write_g4(root, dev, failing_b=failing, edges=edges)
        g4 = json.loads((root / 'results/v4_lpp/g4.json').read_text())
        got = vl.g4_records(root, FP['lpp_cell_code_sha256'], dev)
        assert (got['b'], got['n_passed'], got['passed']) == (g4['b']['passed'], g4['b']['n_passed'], g4['passed'])
        assert got['c'] == (g4['c'] or {}).get('passed')
        assert vl.lpp_g4(root, FP, dev)['status'] == ('passed' if g4['passed'] else 'failed')


def test_verify_lock(dev_copy):
    root, dev, _ = dev_copy
    lock = build(root)
    vl.write_lock(lock, root)
    record = vl.verify_lock(root, fingerprints=FP, tree=TREE_OK)
    assert record['verified'] is True and record['problems'] == [] and record['lock_sha256'] == sv.sha256_file(
        root / sv.LOCK) and record['lock_git_check'] == 'skipped (--no-git-check)'
    with pytest.raises(vl.Refused, match='summarize_sha256 differs from the running code'):
        vl.verify_lock(root, fingerprints=dict(FP, summarize_sha256='0' * 64), tree=TREE_OK)
    with pytest.raises(vl.Refused, match='needs git'):
        vl.verify_lock(root, fingerprints=FP, tree=None)
    with pytest.raises(vl.Refused, match='scripts/run_v4_lpp.py is not committed'):
        vl.verify_lock(root, fingerprints=FP, tree=dict(TREE_OK, problems=['scripts/run_v4_lpp.py is not committed at '
                                                                           'abc']))
    edit(root, sv.LOCK, code_files={k: v for k, v in lock['code_files'].items() if k != 'scripts/v4_gates.py'})
    if (ROOT / 'scripts/v4_gates.py').is_file():
        with pytest.raises(vl.Refused, match='v4_gates.py is fingerprinted but not pinned'):
            vl.verify_lock(root, fingerprints=FP, tree=TREE_OK)


def test_lock_refuses_lpp_cells_under_another_configuration(dev_copy):
    root, _, _ = dev_copy
    with pytest.raises(vl.Refused, match='not the locked config_sha256'):
        build(root, fingerprints=dict(FP, config_sha256='e' * 64))


def test_lock_accepts_writer_shaped_cells(dev_copy):
    """__refs- named sensitivity cells, not-estimable LPP cells with recorded scope errors, Module H cells."""
    root, dev, _ = dev_copy
    save(root / 'results/v4/UTS06/dense/develop/cells/S3_TH_SEL__refs-UTS01-UTS02.json',
         tanghuth_cell('UTS06', 'S3', 'TH_SEL', 0.1, refs=('UTS01', 'UTS02')))
    unscored = lpp_cell(dev[0], '2', 'L5', 'EB_MISALIGN', 'develop', 0.0, 0.0, dev)
    for key in ('test', 'residualized'):
        unscored.pop(key)
    unscored.update(estimable=False, scope_errors=['no untagged (final) target brain decoder_fit read for [...]'])
    path = sv.lpp_cell(root, dev[0], 'dense', 'develop', '2', 'L5', 'EB_MISALIGN')
    save(path, unscored)

    def audit(p):
        if Path(p) == path:
            return dict(passed=False, rules=['scope'],
                        violations=[dict(rule='scope', reason=unscored['scope_errors'][0])])
        return dict(passed=True)
    lock = build(root, audit=audit)
    assert 'results/v4/UTS06/dense/develop/cells/S3_TH_SEL__refs-UTS01-UTS02.json' in \
        lock['development_cells']['ds003020_sensitivity']
    assert path.relative_to(root).as_posix() in lock['development_cells']['lpp_dev']
    edit(root, 'results/v4/UTS06/dense/develop/cells/S3_TH_SEL__refs-UTS01-UTS02.json', references=['UTS01'])
    with pytest.raises(vl.Refused, match="not the path's"):
        build(root, audit=audit)


def test_lock_refuses_prior_without_its_ladder(tmp_path):
    """Under the substitution rule the plan's H3 ladder is PRIOR_f*: DEV must have produced it."""
    root, dev, _ = dev_root(tmp_path, advanced='PRIOR', drop_h6=True)
    pl = sv.plan('PRIOR', True, 'L3')
    assert sv.required_cells(pl)[('L3', 'PRIOR_f0.5')] == ['raw']
    assert vl.dev_plan_check(root, pl, dev, FP['lpp_cell_code_sha256'], LPP_CONFIG, ok_audit)['excluded'] == 0
    with pytest.raises(vl.Refused, match='do not exercise the locked confirmatory plan'):
        vl.dev_plan_check(root, sv.plan('EB', False, 'L3'), dev, FP['lpp_cell_code_sha256'], LPP_CONFIG, ok_audit)
    sv.lpp_cell(root, dev[0], 'dense', 'develop', '3', 'L3', 'PRIOR_f0.5').unlink()
    with pytest.raises(vl.Refused, match='do not exercise the locked confirmatory plan'):
        vl.dev_plan_check(root, pl, dev, FP['lpp_cell_code_sha256'], LPP_CONFIG, ok_audit)


def test_gate_decision_paths():
    def gates(s3=True, s6=True, primary='S3', feas=True, eb=True, prior=False, g3=True, **changes):
        out = dict(code_sha256='c', G1=dict(S3=dict(passed=s3), S6=dict(passed=s6), primary=primary,
                                            lpp_primary=vl.LPP_PRIMARY.get(primary)),
                   G_FEAS=dict(budget=primary, passed=feas), G2=dict(EB=dict(passed=eb), PRIOR=dict(passed=prior)),
                   G3=dict(passed=g3))
        adv = vl.expected_advancement(out)
        out.update(advanced=adv[0], drop_h6=adv[1])
        return dict(out, **changes)
    assert vl.gate_decision(gates(), 'c')['advanced'] == 'EB'
    assert vl.gate_decision(gates(g3=False), 'c')['drop_h6'] is True
    assert vl.gate_decision(gates(eb=False, prior=True), 'c')['advanced'] == 'PRIOR'
    assert vl.gate_decision(gates(s3=False, primary='S6'), 'c')['lpp'] == 'L5'
    null = vl.gate_decision(gates(eb=False), 'c')
    assert null['stop'] == 'G2' and null['advanced'] is None and 'development null' in null['reasons'][0]
    assert vl.gate_decision(gates(feas=False), 'c')['stop'] == 'G_FEAS'
    stop = vl.gate_decision(gates(s3=False, s6=False, primary=None), 'c')
    assert stop['stop'] == 'G1' and stop['ds003020'] is None
    with pytest.raises(vl.Refused, match='contradicts'):
        vl.gate_decision(gates(primary='S6'), 'c')
    with pytest.raises(vl.Refused, match='contradicts'):
        vl.gate_decision(gates(s3=False, s6=False, primary='S6'), 'c')
    with pytest.raises(vl.Refused, match='records no advancement'):
        vl.gate_decision({k: v for k, v in gates().items() if k != 'advanced'}, 'c')


def test_reserved_lock_after_the_development_null(tmp_path):
    """Nothing passes G2: the lock pins everything for the descriptive ds003020 test, and CONFIRM stays closed."""
    root, dev, confirm = dev_root(tmp_path, effect=DS_NULL)
    assert json.loads((root / 'results/v4/gates.json').read_text())['advanced'] is None
    with pytest.raises(vl.Refused, match='only with --reserve-confirm'):
        build(root)
    lock = build(root, reserve_confirm=True)
    assert lock['confirmatory'] is False and lock['confirmatory_plan'] is None and lock['advanced_arm'] is None
    assert lock['confirm_status']['status'] == 'reserved' and 'development null' in lock['confirm_status']['reasons'][0]
    for key in ('lpp_partition', 'lpp_mask_sha256', 'lpp_code_sha256'):
        assert key not in lock
    assert lock['lpp_reserved']['partition'] == dict(dev=dev, confirm=confirm)
    assert lock['readers'] == {'lpp_prepare.check_lock': 'refuses', 'summarize_v4.check_lock': 'refuses'}
    with pytest.raises(RuntimeError, match='lacks'):
        lpp_prepare.check_lock(lock, dev, confirm, PREP_CODE)
    with pytest.raises(sv.Refused, match='registers no confirmatory analysis'):
        sv.check_lock(lock, FP)
    assert set(lock['development_effects']['ds003020']) == {'EB', 'PRIOR'}
    assert lock['development_cells']['counts']['lpp_dev'] > 0 and lock['code_sha256'] == FP['code_sha256']
    vl.write_lock(lock, root)
    with pytest.raises(sv.Refused, match='registers no confirmatory analysis'):
        run(root)


def test_reserved_lock_after_g_feas_failed(tmp_path):
    effect = dict(S3=dict(DS_EFFECT['S3'], ORACLE=0.003), ALL=DS_EFFECT['ALL'])
    root, dev, confirm = dev_root(tmp_path, effect=effect)
    with pytest.raises(vl.Refused, match='only with --reserve-confirm'):
        build(root)
    lock = build(root, reserve_confirm=True)
    assert lock['gates']['stop'] == 'G_FEAS' and lock['gates']['verification']['verified'] == ['G1', 'G_FEAS']
    assert lock['advanced_arm'] is None and 'G-FEAS failed' in lock['confirm_status']['reasons'][0]


def test_reserved_lock_after_g4_failed(dev_copy):
    root, dev, confirm = dev_copy
    write_g4(root, dev, failing_b=dev[:4])
    lock = build(root, reserve_confirm=True)
    assert lock['confirmatory'] is False and lock['advanced_arm'] == 'EB'
    assert 'LPP G4 failed' in lock['confirm_status']['reasons'][0] and lock['gates']['lpp_g4']['status'] == 'failed'
    assert set(lock['development_effects']['lpp']) == {'EB'} and 'lpp_partition' not in lock


def test_lock_integration_with_real_fingerprints_audits_and_readers(tmp_path):
    """The lock built from the running code's fingerprints, writer-shaped cells with real provenance logs (XDEC
    under Module H's audit) is accepted by run_v4_lpp.check_lock and run_v4_tanghuth.check_lock."""
    import run_v4_lpp
    import run_v4_tanghuth
    from src.runtime import load_config
    fp = sv.current_fingerprints()
    root, dev, confirm = dev_root(tmp_path, fp=fp, logs=True)

    def audit(path):  # Real audits for LPP cells; ds003020 scopes need the manifests (absent here).
        return sv.default_audit(path) if '/results/v4_lpp/' in Path(path).as_posix() else dict(passed=True)
    lock = vl.build_lock(root, lpp_code=PREP_CODE, audit=audit, git=None, no_git_check=True)
    assert lock['readers'] == {'lpp_prepare.check_lock': 'accepts', 'summarize_v4.check_lock': 'accepts',
                               'run_v4 --stage evaluate': 'accepts', 'run_v4_tanghuth.check_lock': 'accepts',
                               'run_v4_lpp.check_lock': 'accepts'}
    xdec = sv.lpp_cell(root, dev[0], 'dense', 'develop', '1', 'L3', 'XDEC')
    assert not audit_v4.audit_file(xdec)['passed'] and xdec.relative_to(root).as_posix() in \
        lock['development_cells']['lpp_dev']
    vl.write_lock(lock, root)
    path = root / sv.LOCK
    ctx = run_v4_lpp.Context(load_config(ROOT / sv.CONFIG, []), device='cpu', lock_path=path)
    assert run_v4_lpp.check_lock(ctx, confirm[0], git_state=None)['lpp_cell_code_sha256'] == fp['lpp_cell_code_sha256']
    run_v4_tanghuth.check_lock(path)
    with pytest.raises(SystemExit, match='not a CONFIRM listener'):
        run_v4_lpp.check_lock(ctx, dev[0], git_state=None)
    # The reserved form of the same development record: the CONFIRM readers refuse it, run_v4_tanghuth accepts.
    path.unlink()
    write_g4(root, dev, fp, failing_b=dev[:4])
    reserved = vl.build_lock(root, lpp_code=PREP_CODE, audit=audit, git=None, no_git_check=True, reserve_confirm=True)
    assert reserved['readers'] == {'lpp_prepare.check_lock': 'refuses', 'summarize_v4.check_lock': 'refuses',
                                   'run_v4 --stage evaluate': 'accepts', 'run_v4_tanghuth.check_lock': 'accepts',
                                   'run_v4_lpp.check_lock': 'refuses'}


# ------------------------------------------------------------------------------------- descriptive modes
def test_lpp_dev_summary_and_reliability(dev_copy):
    root, dev, _ = dev_copy
    gates = json.loads((root / 'results/v4/gates.json').read_text())
    out = sv.lpp_summary(root, dev, 'develop', FP['lpp_cell_code_sha256'], gates, 'DEV')
    row = next(r for r in out['curves'] if r['arm'] == 'EB' and r['budget'] == 'L3')
    d = np.array([np.mean([lpp_dev_value(i, t, 'L3', 'EB') - lpp_dev_value(i, t, 'L3', 'W1') for t in SECTIONS])
                  for i in range(9)])
    assert row['n'] == 9 and row['minus_W1']['mean'] == pytest.approx(d.mean())
    assert row['minus_W1']['p_one_sided'] == st.sign_flip_p(d)
    assert row['minus_W1_residualized']['mean'] == pytest.approx(d.mean() / 2)  # Nested residualized.test blocks.
    assert out['development']['minus_w1']['dz'] == pytest.approx(st.dz(d))
    assert out['reliability']['prediction'] == pytest.approx(48 / 56)
    stale = sv.lpp_summary(root, dev, 'develop', '0' * 64, gates, 'DEV')
    assert stale['curves'] == [] and {r['reason'] for r in stale['skipped']} == {'other code'}
    assert 'error' in stale['development']


def test_spearman_brown_fit():
    rho = 0.07
    ks = np.array([1, 2, 4, 8])
    assert sv.fit_spearman_brown(ks, sv.spearman_brown(ks, rho)) == pytest.approx(rho, abs=1e-9)
    assert sv.fit_spearman_brown([8], [sv.spearman_brown(8, rho)]) == pytest.approx(rho)
    assert sv.fit_spearman_brown([8], [1.3]) == 1.0 and sv.fit_spearman_brown([8], [-0.2]) == 0.0
    rows = [dict(clock='dense', budget='L3', arm=arm, K=k, subject=s, eb={'1': eb_info(var_p=r_p)})
            for arm, k in (('EB', 8), ('EB_K2_d0', 2)) for s, r_p in (('a', sv.spearman_brown(k, rho)),
                                                                      ('b', sv.spearman_brown(k, rho)))]
    rows.append(dict(clock='dense', budget='L3', arm='EB', K=8, subject='c', eb={'1': eb_info(var_z=0.5)}))
    fit = sv.reliability_fit(rows, 'dense', 'L3')
    assert fit['rho1'] == pytest.approx(rho, abs=1e-9) and fit['undefined_values'] == 1 and fit['n_k_levels'] == 2
    with pytest.raises(sv.Refused):
        sv.reliability_fit(rows, 'stride4', 'L3')


def test_ds003020_summary_and_cli(dev_copy):
    root, dev, _ = dev_copy
    gates = json.loads((root / 'results/v4/gates.json').read_text())
    out = sv.ds003020_summary(root, 'develop', fp=FP, gates=gates)
    row = next(r for r in out['curves'] if r['arm'] == 'EB' and r['budget'] == 'S3')
    flips = row['minus_W0_story_flip']['UTS01']
    story = [ds_value(0, 'S3', 'EB') - ds_value(0, 'S3', 'W0')] * 3  # The story offsets cancel in the difference.
    assert flips['n_stories'] == 3 and flips['p_one_sided'] == st.sign_flip_p(story)
    assert row['oracle_fraction'] == pytest.approx(
        st.oracle_fraction([ds_value(i, 'S3', 'EB') for i in range(9)], [ds_value(i, 'S3', 'W0') for i in range(9)],
                           [ds_value(i, 'S3', 'ORACLE') for i in range(9)]))
    assert out['development']['budget'] == 'S3' and out['development']['minus_w0']['n'] == 9
    assert out['kappa_schedule'][0]['pairs'] == 1000
    assert {r['arm'] for r in out['curves'] if r['budget'] == 'S3'} >= {'TH_SEL', 'TH_SEL_rand', 'XDEC'}
    edit(root, 'results/v4/UTS05/dense/develop/cells/S3_TH_SEL.json', tanghuth_sha256='0' * 64)
    stale = sv.ds003020_summary(root, 'develop', fp=FP, gates=gates)
    assert not any(r['arm'] == 'TH_SEL' for r in stale['curves'])  # A stale Module H cell is not summarized.
    sv.main(['--ds003020', '--any-code', '--root', str(root)])
    assert (root / 'results/v4/ds003020_develop_summary.json').is_file()
    assert '## dense clock' in (root / 'results/v4/DS003020_DEVELOP.md').read_text()
    sv.main(['--dev', '--any-code', '--root', str(root)])
    assert json.loads((root / 'results/v4/lpp_dev_summary.json').read_text())['subjects'] == dev
    with pytest.raises(sv.Refused, match='after the v4 lock'):
        sv.main(['--ds003020', '--stage', 'evaluate', '--any-code', '--root', str(root)])
    with pytest.raises(sv.Refused, match='after the primary analysis'):
        sv.main(['--confirm-curves', '--any-code', '--root', str(root)])


def test_ds003020_two_reference_sensitivity(dev_copy):
    """Module H's two-reference cells (folder and __refs- named forms) are summarized per pair against the
    three-reference cells, with story-level sign-flips; stale or mislabeled cells are skipped with the reason."""
    root, _, _ = dev_copy
    runs = {'storyA': 0.0, 'storyB': 0.004, 'storyC': -0.001, 'storyD': 0.002}

    def cell(s, arm, v, refs, **changes):
        out = tanghuth_cell(s, 'S3', arm, v, refs=refs)
        out['val']['by_run_feature_correlation'] = {k: v + d for k, d in runs.items()}
        out.update(changes)
        return out
    three, pair = {'UTS05': 0.10, 'UTS06': 0.12}, {'UTS05': 0.09, 'UTS06': 0.115}
    for s in three:
        save(sv.ds_cell(root, s, 'dense', 'develop', 'S3', 'XDEC'), cell(s, 'XDEC', three[s], ('UTS01', 'UTS02', 'UTS03')))
        save(root / f'results/v4/{s}/dense/develop/sensitivity/refs-UTS01-UTS02/cells/S3_XDEC.json',
             cell(s, 'XDEC', pair[s], ('UTS01', 'UTS02')))
        save(root / f'results/v4/{s}/dense/develop/cells/S3_XDEC__refs-UTS01-UTS03.json',
             cell(s, 'XDEC', pair[s] + 0.01, ('UTS01', 'UTS03')))
    save(root / 'results/v4/UTS07/dense/develop/sensitivity/refs-UTS02-UTS03/cells/S3_XDEC.json',
         cell('UTS07', 'XDEC', 0.1, ('UTS02', 'UTS03'), tanghuth_sha256='0' * 64))  # Stale Module H code.
    save(root / 'results/v4/UTS08/dense/develop/sensitivity/refs-UTS02-UTS03/cells/S3_XDEC.json',
         cell('UTS08', 'XDEC', 0.1, ('UTS01', 'UTS02')))  # References are not the folder's pair.
    out = sv.ds003020_summary(root, 'develop', fp=FP)
    rows = {(r['arm'], r['budget'], '-'.join(r['references'])): r for r in out['sensitivity']}
    row = rows[('XDEC', 'S3', 'UTS01-UTS02')]
    assert row['n'] == 2 and row['minus_three_references']['mean'] == pytest.approx(-0.0075)
    flips = row['minus_three_references_story_flip']
    assert flips['UTS05'] == dict(n_stories=4, mean=pytest.approx(-0.01), p_one_sided=st.sign_flip_p([-0.01] * 4))
    assert rows[('XDEC', 'S3', 'UTS01-UTS03')]['minus_three_references']['mean'] == pytest.approx(0.0025)
    assert 'minus_W0' in row and 'minus_W1' in row
    assert ('TH_SEL', 'S6', 'UTS01-UTS02') in rows and 'minus_three_references' not in rows[('TH_SEL', 'S6',
                                                                                              'UTS01-UTS02')]
    reasons = {r['subject']: r['reason'] for r in out['sensitivity_skipped']}
    assert reasons['UTS07'].startswith('other code') and 'differ from the path' in reasons['UTS08']
    assert not any('__refs-' in str(r.get('arm')) for r in out['curves'])  # Not in the budget curves.
    text = sv.curves_markdown('ds003020', sv.clean(out), sv.DS_BUDGETS)
    assert '## Two-reference sensitivity' in text and '| dense | S3 | XDEC | UTS01-UTS02 |' in text
