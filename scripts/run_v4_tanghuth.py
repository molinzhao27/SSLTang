#!/usr/bin/env python3
"""PROTOCOL_V4 published-method block on ds003020 (FUTURE_WORK_V4 §4.1, §5, §8 item 5; V4_IMPLEMENTATION Module H).

Targets UTS04-UTS10, references UTS01-UTS03 under the TS rule (reference transcripts only at stories the target
never heard), budgets S3/S6/S15, arms TH_SEL, TH_SEL_rand and XDEC (``src/priors/tanghuth.py``).
Stages (one target subject and clock per call):
  develop   validation scores; refused unless G0 has passed under the current run_v4 core (as run_v4)
  evaluate  validation and test scores; refused unless experiments/v4_lock.json pins the current run_v4 core
            (``code_sha256``), src/priors/tanghuth.py (``tanghuth_sha256``) and this block's cell code
            (``tanghuth_cell_code_sha256`` = ``code_sha256()`` here). TH_SEL and TH_SEL_rand reuse the matching
            develop selections and recompute the final fit, as run_v4's evaluate stage; XDEC selects nothing on
            target data and is recomputed.
TH_SEL and TH_SEL_rand use run_v4's engine unchanged: the selection mask enters through the engine's PRIOR slot
(``tanghuth.selection_priors``), so inner CV, tie rules, kernels, final fits and scoring are run_v4's, and the
TH_SEL selection is recomputed from every inner fold's fitting stories.
TS material per (target, reference) (``ts_material``):
  labeled    all the reference's TS stories (``budgets.ts_stories``: in its manifest, never heard by the target),
             in any of its splits (its own validation/test stories are not the target's material);
             ``ts_allowed`` (reference transcripts) is their union over the references
  selection  TH_SEL's selection material: the TS stories every reference heard, in any of their splits
             (``tanghuth.ds003020_selection``; BOLD only, raw H5 per-run z-scored, logged 'reference_model')
A reference model (its top-10k voxels by nested W1 encoding r and, for XDEC, a §12 W1 decoder on them) is fit once
per (reference, labeled stories, clock) on its delay rows at the cell's clock from every split (``reference_rows``:
training rows from the run_v4 caches, validation/test rows from the v3 stride-4 caches or, on the dense clock, from
dense caches this block builds under data/cache/v4/tanghuth/rows/dense/<reference>), with the §12 rule (the fixed
five-fold partition of the reference's sessions, folds without labeled rows dropped), and cached under
data/cache/v4/tanghuth/references.
Cells: results/v4/<S>/<clock>/<stage>/cells/<budget>_<arm>.json, carrying ``code_sha256()`` (run_v4's core plus
Module H). The two-reference sensitivity analysis (``--sensitivity``) is registered here as every pair of
UTS01-UTS03 (``SENSITIVITY_PAIRS`` = ``tanghuth.sensitivity_pairs``), each run and reported, for TH_SEL and XDEC
only (TH_SEL_rand reads no reference); its cells go to results/v4/<S>/<clock>/<stage>/sensitivity/refs-<R1>-<R2>/
cells/. The pairs of one call share their converter alphas (``tanghuth.thsel_scores(alpha_memo=...)``). Once
experiments/v4_lock.json exists, an existing develop cell is never overwritten. Every cell embeds its own
provenance log (arm, TS rule, ts_allowed) and is written only after R1-R7 and the registered-scope audit
(``tanghuth.audit_errors``: the frozen audit plus each arm's own checks) pass.
"""
import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v3 as v3  # noqa: E402
import run_v4 as v4  # noqa: E402
from src import budgets as bud  # noqa: E402
from src.evaluation.endpoints import score, shift_null  # noqa: E402
from src.priors import isp as ispm  # noqa: E402
from src.priors import tanghuth as th  # noqa: E402
from src.provenance import Provenance  # noqa: E402
from src.runtime import load_config, save_json  # noqa: E402
from src.structured_ridge import budget_folds, fixed_folds  # noqa: E402
from src.targets import TargetIndex  # noqa: E402
from src.voxel_kernels import DELAYS, _h5_path, cache_delay_rows, verify_cache  # noqa: E402

TARGETS = tuple(f'UTS{i:02d}' for i in range(4, 11))
REFERENCES = ('UTS01', 'UTS02', 'UTS03')
DATASET = 'ds003020'
SENSITIVITY_PAIRS = tuple(th.sensitivity_pairs(REFERENCES, DATASET))  # Registered two-reference sets: every pair.
BUDGETS = ('S3', 'S6', 'S15')
CACHE = v4.CACHE / 'tanghuth'
ROWS = CACHE / 'rows'  # Dense delay rows of the references' validation/test splits (their TS labels).
SPLITS = ('train', 'val', 'test')
OUT = v4.OUT
LOCK = Path('experiments/v4_lock.json')


# ------------------------------------------------------------------------------------------- helpers
def code_sha256():
    """The code fingerprint this block's cells carry as ``code_sha256``: run_v4's core plus Module H."""
    return v4.sha(v4.code_sha256() + th.code_sha256())


def stage_dir(subject, clock, stage, arm, refs=REFERENCES):
    """results/v4/<S>/<clock>/<stage>, or its sensitivity/refs-... folder for a non-default reference set (any
    arm: nothing run with another reference set can land in the main, locked folder)."""
    folder = OUT / subject / clock / stage
    if tuple(refs) != REFERENCES:
        folder = folder / 'sensitivity' / ('refs-' + '-'.join(refs))
    return folder


def cell_path(subject, clock, stage, name, arm, refs=REFERENCES):
    return stage_dir(subject, clock, stage, arm, refs) / 'cells' / f'{name}_{arm}.json'


def check_refs(refs, arms=()):
    """The registered reference sets: UTS01-UTS03, or a ``SENSITIVITY_PAIRS`` pair for TH_SEL / XDEC only."""
    refs = tuple(refs)
    if refs == REFERENCES:
        return refs
    if refs not in SENSITIVITY_PAIRS:
        raise SystemExit(f'Reference set {list(refs)} is neither {list(REFERENCES)} nor a registered sensitivity '
                         f'pair {[list(p) for p in SENSITIVITY_PAIRS]}')
    other = sorted(set(arms) - set(th.REFERENCE_ARMS))
    if other:
        raise SystemExit(f'The two-reference sensitivity runs {list(th.REFERENCE_ARMS)} only, not {other}')
    return refs


def ts_material(manifests, subject, refs):
    """({ref: its labeled TS stories}, ts_allowed, selection material), all sorted.

    A reference's labeled TS stories are all its TS stories (``budgets.ts_stories``: in its manifest, never heard
    by the target), in any of its splits; ``ts_allowed`` is their union. The selection material is the TS stories
    every reference heard, in any split (``tanghuth.ds003020_selection``; BOLD only).
    """
    by_ref = {ref: bud.ts_stories(manifests[subject], manifests[ref]) for ref in refs}
    union = sorted(set().union(*by_ref.values()))
    return by_ref, union, th.ds003020_selection(manifests, subject, refs)


def check_writable(path, stage, lock=None):
    """Refuse to overwrite a develop cell once the v4 lock exists (its sha256 may be pinned there)."""
    lock = LOCK if lock is None else Path(lock)
    if stage == 'develop' and lock.is_file() and Path(path).exists():
        raise SystemExit(f'{path} exists and {lock} exists: develop cells are not rewritten after the lock')


def load_raw(raw_root, run):
    """[n_tr, V] float32 BOLD in the delay-cache representation (raw H5 rows, non-finite values set to 0)."""
    with h5py.File(_h5_path(run, raw_root), 'r') as handle:
        x = np.nan_to_num(np.asarray(handle['data'], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if len(x) != int(run['n_tr']):
        raise ValueError(f"{run['source_key']}: {len(x)} TRs, manifest says {run['n_tr']}")
    return x


class Bold:
    """One subject's per-run BOLD, loaded once per story: per-run z-scored (``isp.load_bold``) or in the cache
    representation (``load_raw``), restricted to ``voxels`` when given. Loads log nothing (callers log reads)."""

    def __init__(self, raw_root, manifest, voxels=None):
        self.raw_root, self.voxels, self.memo = raw_root, voxels, {}
        self.runs = {r['story']: r for r in manifest['runs']}

    def _get(self, kind, story):
        if (kind, story) not in self.memo:
            x = (ispm.load_bold if kind == 'z' else load_raw)(self.raw_root, self.runs[story])
            self.memo[(kind, story)] = x if self.voxels is None else np.ascontiguousarray(x[:, self.voxels])
        return self.memo[(kind, story)]

    def z(self, story):
        return self._get('z', story)

    def raw(self, story):
        return self._get('raw', story)


def reference_rows(cfg, ref_tgt, stories):
    """[(split, index, cache paths, positions)]: the reference's delay rows at ``stories`` on its clock, from every
    split holding one of them (no rows are loaded here).

    Training rows come from the reference Target's cache. Validation/test rows (a reference's own validation and
    test stories are TS stories of every target that never heard them) come from the v3 stride-4 caches on the
    stride-4 clock (verified) and, on the dense clock, from dense caches built once under ROWS/dense/<reference>
    (``cache_delay_rows``: the same eligibility, delays and representation as the training caches). A cache holds
    its whole split; only the rows of ``stories`` are ever loaded (and so logged by the engine).
    """
    data = cfg['data']
    split_of = {r['story']: r['split'] for r in ref_tgt.manifest['runs']}
    unknown = sorted(set(stories) - set(split_of))
    if unknown:
        raise SystemExit(f'{ref_tgt.subject}: stories {unknown} are not in its manifest')
    out = []
    for split in SPLITS:
        want = {s for s in stories if split_of[s] == split}
        if not want:
            continue
        if split == 'train':
            index, paths = ref_tgt.index['train'], ref_tgt.cache['train']
        elif ref_tgt.clock == 'dense':
            index = TargetIndex(ref_tgt.manifest_path, split, stride=1, window=data['window'],
                                phase=data['target_phase'], cutoffs=data['cutoffs'])
            paths = cache_delay_rows(index, cfg['raw_root'], ROWS / 'dense' / ref_tgt.subject)[0]
        else:
            index = ref_tgt.index[split]
            paths = verify_cache(index, cfg['raw_root'], Path(cfg['cache_root']) / ref_tgt.subject)[0]
        pos = np.array([i for i, (ri, _) in enumerate(index.items) if index.runs[ri]['story'] in want],
                       dtype=np.int64)
        got = {index.runs[index.items[i][0]]['story'] for i in pos}
        if got != want:
            raise th.NotEstimable(f'{ref_tgt.subject}: no eligible {split} rows for {sorted(want - got)}')
        out.append((split, index, paths, pos))
    return out


def rows_budget(ref_tgt, stories, parts, device):
    """(Target-like, run_v4 Budget) over the rows ``parts`` (``reference_rows``), loaded one delay at a time."""
    n = sum(len(pos) for *_, pos in parts)
    y = np.concatenate([index.targets(pos) for _, index, _, pos in parts]).astype(np.float64)
    story = np.concatenate([[index.runs[index.items[i][0]]['story'] for i in pos] for _, index, _, pos in parts])
    tr = np.concatenate([index.target_trs[pos] for _, index, _, pos in parts])
    run = np.concatenate([index.run_ids[pos] for _, index, _, pos in parts])
    run_sessions = {}
    for _, index, _, _ in parts:
        run_sessions.update(index.sessions)

    def raw(d):
        out, start = None, 0
        for _, _, paths, pos in parts:
            block = torch.from_numpy(np.array(np.load(paths[d], mmap_mode='r')[pos], dtype=np.float32))
            if out is None:
                out = torch.empty((n, block.shape[1]), dtype=torch.float32, device=device)
            out[start:start + len(pos)] = block.to(device)
            start += len(pos)
            del block
        return out
    tl = th.target_like(ref_tgt.subject, ref_tgt.alphas, device, run_sessions, ref_tgt.n_tr, clock=ref_tgt.clock)
    return tl, th.budget_from_arrays(tl, 'ALL', stories, raw, y, story, tr, run, [run_sessions[r] for r in run])


def reference_model(cfg, ref, clock, stories, device, decoder):
    """(model, info): the cached or freshly fit TS model of ``ref`` on ``stories`` (any split) at ``clock``."""
    ref_tgt = v4.Target(cfg, ref, clock, device)
    parts = reference_rows(cfg, ref_tgt, stories)
    metas = {split: json.loads((Path(paths[DELAYS[0]]).parent / f'{split}.meta.json').read_text())
             for split, _, paths, _ in parts}
    by_split = {split: sorted({index.runs[index.items[i][0]]['story'] for i in pos}) for split, index, _, pos in parts}

    def key(dec):
        return v4.sha(json.dumps([ref, clock, sorted(stories), list(ref_tgt.alphas), th.REF_TOP, dec, metas,
                                  th.code_sha256(), v4.code_sha256(), v3.config_sha256(cfg)]))[:20]
    for dec in ((True,) if decoder else (True, False)):
        path = CACHE / 'references' / f'{ref}_{clock}_{key(dec)}.npz'
        if path.exists():
            return th.ReferenceModel.load(path), dict(cache=path.name, cached=True)
    tl, bd = rows_budget(ref_tgt, stories, parts, device)
    sessions = [r['session'] for r in ref_tgt.manifest['runs']]  # §12: fixed partition of the reference's sessions.
    folds, rule = budget_folds(list(bd.session), list(range(len(bd.rows))), fixed_folds(sessions))
    if folds is None:
        raise th.NotEstimable(f'{ref}: {rule}')
    story_folds = [tuple(sorted({str(bd.story[i]) for i in range(len(bd.rows)) if bd.session[i] in set(f)}))
                   for f in folds]
    print(json.dumps(dict(reference=ref, clock=clock, stories=len(stories), rows=len(bd.rows), decoder=decoder,
                          splits={k: len(v) for k, v in by_split.items()})), flush=True)
    progress = lambda e: print(json.dumps(dict(reference=ref, held=len(e['held']), seconds=round(e['seconds'], 1))),
                               flush=True)
    model = th.fit_reference_model(tl, bd, 'ALL', (story_folds, [tuple(f) for f in folds]), k=th.REF_TOP,
                                   decoder=decoder, log_fold=progress)
    model.info.update(clock=clock, fold_rule=rule, splits=by_split,
                      minutes=bud.bold_minutes([ref_tgt.n_tr[s] for s in stories]))
    path = CACHE / 'references' / f'{ref}_{clock}_{key(decoder)}.npz'
    model.save(path)
    del bd
    torch.cuda.empty_cache()
    return model, dict(cache=path.name, cached=False)


def reference_ledger(ledger, manifests, refs, models, stories, extra=()):
    """Add the references' BOLD minutes over the stories actually read (converter ``stories``, each model's
    labeled stories ``models[ref].stimuli`` and ``extra``, the selection material), labeled pairs and the TS
    rule to a ledger dict."""
    minutes = 0.0
    for ref in refs:
        n_tr = {r['story']: int(r['n_tr']) for r in manifests[ref]['runs']}
        read = set(stories) | set(models[ref].stimuli) | set(extra)
        minutes += bud.bold_minutes([n_tr[s] for s in sorted(read)])
    pairs = sum(int(models[ref].info['labeled_rows']) for ref in refs)
    return bud.Ledger.from_dict(ledger).reference(list(refs), minutes, pairs, th.rule_for(DATASET)).to_dict()


def code_fields(cfg):
    return dict(code_sha256=code_sha256(), core_sha256=v4.code_sha256(),
                tanghuth_sha256=v4.file_sha('src/priors/tanghuth.py'),
                tanghuth_runner_sha256=v4.file_sha('scripts/run_v4_tanghuth.py'), config_sha256=v3.config_sha256(cfg))


def finish(cell, prov, name, arm, refs, predictions=None):
    """Audit, then write the cell (and its float16 predictions); nothing is written on any violation."""
    prov.check()  # R1-R7: refuse to write any score on a violation.
    cell['provenance'] = prov.to_dict()
    errors = th.audit_errors(cell, prov)
    if errors:
        raise SystemExit(f'Registered-scope audit failed for {name} {arm}: {errors}')
    folder = stage_dir(cell['subject'], cell['clock'], cell['stage'], arm, refs)
    check_writable(folder / 'cells' / f'{name}_{arm}.json', cell['stage'])
    if predictions:
        (folder / 'predictions').mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'predictions' / f'{name}_{arm}.npz',
                            **{k: np.asarray(v, dtype=np.float16) for k, v in predictions.items()})
    save_json(cell, folder / 'cells' / f'{name}_{arm}.json')
    shown = ' '.join(f"{k}={cell[k]['session_macro_feature_correlation']:.4f}" for k in ('val', 'test') if k in cell)
    print(f"{cell['subject']} {cell['clock']} {name} {arm}: {shown}", flush=True)


# ---------------------------------------------------------------------------------------------- arms
def selection_cell(ctx, bd, name, stories, story_folds, arm):
    """TH_SEL or TH_SEL_rand through run_v4's engine (PRIOR slot carrying the selection mask)."""
    tgt, stage, refs = ctx['tgt'], ctx['stage'], ctx['refs']
    subject = tgt.subject
    prov = Provenance(subject, 'S', stories, tgt.split_stories['val'], tgt.split_stories['test'],
                      ts_allowed=ctx['ts_allowed'] if arm == 'TH_SEL' else None, folds=story_folds, arm=arm,
                      budget_name=name, stage=stage)
    n_voxels = int(bd.raw[DELAYS[0]].shape[1])
    k = th.goal_size(n_voxels, DATASET)
    if arm == 'TH_SEL':
        for ref in refs:
            th.log_reads(prov, th.model_reads(ctx['models'][ref]))
        selector = th.ThSelector(subject, ctx['target'].z, refs, lambda ref, st: ctx['ref_bold'][ref].z(st),
                                 ctx['unheard'], k, prov, device=ctx['device'], alpha_memo=ctx['alpha_memo'])
    else:
        selector = th.RandomSelector(n_voxels, k, subject)
    pri = th.selection_priors(tgt, prov, ctx['device'], selector)
    use = [th.ENGINE_ARM]
    if stage == 'evaluate':  # Reuse the develop selection (identical inner CV under the locked code).
        path = cell_path(subject, tgt.clock, 'develop', name, arm, refs)
        dev = json.loads(path.read_text())
        if dev.get('code_sha256') != code_sha256() or dev.get('provenance', {}).get('folds') is None:
            raise SystemExit(f'{path} is not from the locked code')
        selections, curves, logs = {th.ENGINE_ARM: dev['selection']}, {th.ENGINE_ARM: dev['cv_curves']}, dev['folds']
        for i, held in enumerate(story_folds):  # The reused inner CV read exactly the develop folds' material.
            with prov.inner_fold(i):
                fit = [s for s in stories if s not in held]
                v4.log_reads(bd, prov, bd.positions(fit), bd.positions(held), ['W0'], pri, ())
                if arm == 'TH_SEL':
                    th.log_reads(prov, th.thsel_reads(subject, refs, fit, ctx['unheard']))
    else:
        selections, curves, logs = v4.run_cv(
            tgt, bd, name, use, pri, prov, None, story_folds, None,
            lambda e: print(json.dumps(dict(budget=name, arm=arm, held=e['held'], seconds=round(e['seconds'], 1))),
                            flush=True))
    predictions, final = v4.final_fit(tgt, bd, name, use, selections, ctx['eval_splits'], pri, prov, None, None)
    ledger = v4.ledger_for(tgt, bd, 'W0', None, [])  # Target part: B labels and BOLD (no reference arm of run_v4).
    if arm == 'TH_SEL':
        ledger = reference_ledger(ledger, ctx['manifests'], refs, ctx['models'], stories, ctx['unheard'])
    cell = dict(protocol='ssltang-v4', block='published-method', stage=stage, subject=subject, clock=tgt.clock,
                budget=name, arm=arm, regime='S', engine_arm=th.ENGINE_ARM,
                references=list(refs) if arm == 'TH_SEL' else [], reference_rule=prov.reference_rule,
                selection_material=list(ctx['unheard']) if arm == 'TH_SEL' else None,
                estimable=th.ENGINE_ARM in predictions, selection=selections.get(th.ENGINE_ARM),
                selection_note='binary mask: the engine gamma rules tie (weights equal at every gamma)',
                cv_curves=curves.get(th.ENGINE_ARM), folds=logs, final_selection=final.get('isp'),
                final_weights=final['weights'].get(th.ENGINE_ARM), ledger=ledger,
                reference_models={r: dict(ctx['models'][r].summary(), **ctx['model_info'][r]) for r in refs}
                if arm == 'TH_SEL' else None, **code_fields(ctx['cfg']))
    preds = None
    if th.ENGINE_ARM in predictions:
        preds, diag = predictions[th.ENGINE_ARM]
        cell['final_fit'] = diag
        for sp in ctx['eval_splits']:
            cell[sp] = v4.report(tgt, bd, preds[sp], sp)
        if 'test' in ctx['eval_splits']:
            cell['test_shift_null'] = test_null(ctx['cfg'], tgt, preds['test'])
    finish(cell, prov, name, arm, refs, preds)


def xdec_cell(ctx, name, stories):
    """XDEC: references' W1 decoders applied to converted target BOLD; no target label is read."""
    tgt, stage, refs, eval_splits = ctx['tgt'], ctx['stage'], ctx['refs'], ctx['eval_splits']
    subject = tgt.subject
    prov = Provenance(subject, 'S', stories, tgt.split_stories['val'], tgt.split_stories['test'],
                      ts_allowed=ctx['ts_allowed'], arm='XDEC', budget_name=name, stage=stage)
    for ref in refs:
        th.log_reads(prov, th.model_reads(ctx['models'][ref]))
    parts = {}
    for sp in eval_splits:
        index = tgt.index[sp]
        parts[sp] = (np.array([index.runs[ri]['story'] for ri, _ in index.items]), np.asarray(index.target_trs))
    eval_story = np.concatenate([parts[sp][0] for sp in eval_splits])
    eval_tr = np.concatenate([parts[sp][1] for sp in eval_splits])
    target_runs = {st: ctx['target'].z(st) for st in stories}
    reference_runs = {ref: {st: ctx['ref_bold'][ref].raw(st) for st in stories} for ref in refs}
    eval_runs = {e: ctx['target'].z(e) for e in dict.fromkeys(eval_story.tolist())}
    th.log_reads(prov, th.xdec_reads(subject, refs, list(target_runs), list(eval_runs)))
    decoders = {ref: ctx['models'][ref].decoder for ref in refs}
    pred, info = th.xdec_predict(target_runs, reference_runs, decoders, stories, eval_runs, eval_story, eval_tr,
                                 device=ctx['device'])
    ledger = bud.Ledger().target(0, 0, bud.bold_minutes([tgt.n_tr[s] for s in stories])).to_dict()
    cell = dict(protocol='ssltang-v4', block='published-method', stage=stage, subject=subject, clock=tgt.clock,
                budget=name, arm='XDEC', regime='S', references=list(refs), reference_rule=prov.reference_rule,
                estimable=True, converters=info, target_labels=0,
                score_normalization='reference label mean/SD averaged over references (no target label)',
                ledger=reference_ledger(ledger, ctx['manifests'], refs, ctx['models'], stories),
                reference_models={r: dict(ctx['models'][r].summary(), **ctx['model_info'][r]) for r in refs},
                **code_fields(ctx['cfg']))
    mean, std = th.scoring_stats(decoders)
    predictions, start = {}, 0
    for sp in eval_splits:
        n = len(parts[sp][0])
        predictions[sp], start = pred[start:start + n], start + n
        spl = tgt.split[sp]
        cell[sp] = score(predictions[sp], spl.targets, spl.run_ids, spl.target_trs, spl.sessions, mean, std,
                         ctx['cfg']['evaluation']['exclusions'])
    if 'test' in eval_splits:
        cell['test_shift_null'] = test_null(ctx['cfg'], tgt, predictions['test'])
    finish(cell, prov, name, 'XDEC', refs, predictions)


def test_null(cfg, tgt, prediction):
    ev, spl = cfg['evaluation'], tgt.split['test']
    return shift_null(prediction, spl.targets, spl.run_ids, spl.sessions, cfg['data']['stride'], ev['shift_min_tr'],
                      ev['shift_draws'], ev['shift_seed'])


# --------------------------------------------------------------------------------------------- stages
def stage_fit(cfg, subject, clock, names, arms, stage, device, refs=REFERENCES, alpha_memo=None):
    """Fit and write the cells of one target, clock and reference set. ``alpha_memo``: a dict shared by the
    reference sets of one call (the sensitivity pairs), so each reference's converter alpha is fit once."""
    refs = check_refs(refs, arms)
    for name in names:  # Before any fit: no locked develop cell is recomputed.
        for arm in arms:
            check_writable(cell_path(subject, clock, stage, name, arm, refs), stage)
    eval_splits = ('val',) if stage == 'develop' else ('val', 'test')
    tgt = v4.Target(cfg, subject, clock, device, eval_splits)
    manifests = v4.manifests_all()
    budgets = bud.ds003020_budgets(tgt.manifest, bud.shared_stories(manifests))
    ts_by_ref, ts_allowed, unheard = ts_material(manifests, subject, refs)
    models, model_info = {}, {}
    if any(a in th.REFERENCE_ARMS for a in arms):
        for ref in refs:
            models[ref], model_info[ref] = reference_model(cfg, ref, clock, ts_by_ref[ref], device, 'XDEC' in arms)
    ctx = dict(cfg=cfg, tgt=tgt, stage=stage, refs=list(refs), eval_splits=eval_splits, device=device,
               manifests=manifests, models=models, model_info=model_info, ts_allowed=ts_allowed, unheard=unheard,
               alpha_memo=alpha_memo,
               target=Bold(cfg['raw_root'], tgt.manifest),
               ref_bold={r: Bold(cfg['raw_root'], manifests[r], models[r].voxels) for r in models})
    for name in names:
        stories = budgets[name]
        print(json.dumps(dict(subject=subject, clock=clock, budget=name, stories=len(stories), arms=list(arms),
                              references=list(refs), unheard=len(unheard))), flush=True)
        selection_arms = [a for a in arms if a != 'XDEC']
        if selection_arms:
            story_folds, _ = v4.decoder_folds(tgt, name, stories)
            bd = v4.Budget(tgt, name, stories, eval_splits)
            for arm in selection_arms:
                selection_cell(ctx, bd, name, stories, story_folds, arm)
            del bd
            torch.cuda.empty_cache()
        if 'XDEC' in arms:
            xdec_cell(ctx, name, stories)
            torch.cuda.empty_cache()


def check_lock(path=LOCK):
    if not path.exists():
        raise SystemExit('No experiments/v4_lock.json: test scores need the v4 lock')
    lock = json.loads(path.read_text())
    if lock.get('code_sha256') != v4.code_sha256():
        raise SystemExit('run_v4 core changed since the v4 lock')
    if lock.get('tanghuth_sha256') != v4.file_sha('src/priors/tanghuth.py'):
        raise SystemExit('src/priors/tanghuth.py changed since the v4 lock (or the lock has no tanghuth_sha256)')
    if lock.get('tanghuth_cell_code_sha256') != code_sha256():
        raise SystemExit('This block changed since the v4 lock (or the lock has no tanghuth_cell_code_sha256)')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', choices=['develop', 'evaluate'], required=True)
    p.add_argument('--subject', required=True, choices=TARGETS)
    p.add_argument('--clock', choices=['dense', 'stride4'], default='dense')
    p.add_argument('--budgets', nargs='+', default=list(BUDGETS), choices=BUDGETS)
    p.add_argument('--arms', nargs='+', default=None, choices=th.ARMS,
                   help='default: all of TH_SEL, TH_SEL_rand, XDEC (with --sensitivity: TH_SEL, XDEC)')
    p.add_argument('--sensitivity', action='store_true',
                   help='the registered two-reference sensitivity: TH_SEL / XDEC with every pair of UTS01-UTS03')
    p.add_argument('--config', default='configs/v3/structured_ridge.yaml')
    p.add_argument('--device', default='cuda')
    a = p.parse_args(argv)
    arms = list(a.arms or (th.REFERENCE_ARMS if a.sensitivity else th.ARMS))
    if a.sensitivity and set(arms) - set(th.REFERENCE_ARMS):
        p.error('The two-reference sensitivity runs TH_SEL and XDEC only (TH_SEL_rand reads no reference)')
    ref_sets = SENSITIVITY_PAIRS if a.sensitivity else (REFERENCES,)
    memo = {} if len(ref_sets) > 1 else None  # Converter alphas shared by the pairs (content-keyed).
    cfg = load_config(a.config, [f'data.manifest=data/semantic/{a.subject}/semantic_manifest.json'])
    torch.backends.cuda.matmul.allow_tf32 = False
    v3.verify_lock(cfg)
    if a.stage == 'develop' and not v4.g0_passed():
        raise SystemExit('G0 has not passed under this code (results/v4/g0.json)')
    if a.stage == 'evaluate':
        check_lock()
    for refs in ref_sets:
        stage_fit(cfg, a.subject, a.clock, a.budgets, arms, a.stage, a.device, refs, memo)


if __name__ == '__main__':
    main()
