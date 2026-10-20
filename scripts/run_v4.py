#!/usr/bin/env python3
"""PROTOCOL_V4 on ds003020: inter-subject (ISP) and empirical-Bayes (EB) voxel priors.

Stages (one target subject and clock per call):
  g1        training data only: Spearman between ISP at S3/S6 and the full-label W1 encoding map
  develop   validation scores of the requested arms at the requested story budgets (regime S)
  evaluate  test scores, only after experiments/v4_lock.json matches this code; the inner CV selections
            are reused from the matching develop cells and only the final fit is recomputed
Every arm shares one decoder: all-voxel kernel ridge on standardized (voxel, delay) columns scaled
by arm-specific weights (mean w^2 = 1). Every target-derived quantity is computed from the fold's
fitting part of the budget only. Provenance is logged from the row arrays actually used, and both
the R1-R7 audit and the registered-scope audit must pass before any cell is written.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_v4
import run_v3 as v3
import run_v3_weights as w3
from src import budgets as bud
from src.encoding_weights import FALLBACK_FRACTION, TIE, encoding_scores
from src.evaluation.endpoints import score, shift_null
from src.multikernel_ridge import KernelRidgePath, run_feature_r, session_macro_scores
from src.priors import eb as ebm
from src.priors import isp as ispm
from src.probe import Split
from src.provenance import Provenance
from src.runtime import load_config, save_json
from src.structured_ridge import budget_folds, fixed_folds
from src.targets import TargetIndex
from src.voxel_kernels import DELAYS, column_stats, verify_cache

ROOT = Path(__file__).resolve().parents[1]
CORE = ('src/budgets.py', 'src/provenance.py', 'src/priors/__init__.py', 'src/priors/isp.py', 'src/priors/eb.py',
        'src/priors/acoustic.py', 'src/v4_stats.py', 'src/windows.py', 'src/runtime.py', 'src/transcripts.py',
        'scripts/run_v4.py', 'scripts/audit_v4.py', 'scripts/lpp_prepare.py')
GAMMAS = (0.5, 1.0, 2.0)
LADDER = (0.75, 0.5, 0.25, 0.0)
DEV_SUBJECTS = ('UTS01', 'UTS02', 'UTS03', 'UTS05', 'UTS06', 'UTS07', 'UTS08', 'UTS09', 'UTS10')
TIER1 = ('W0', 'W1', 'PRIOR', 'EB') + tuple(f'EB_f{f:g}' for f in LADDER) + ('EB_SELF', 'EB_MISALIGN', 'P_AC',
                                                                           'ORACLE')
ALL_ARMS = ('W0', 'W1', 'PRIOR', 'EB')
REFERENCE_ARMS = ('PRIOR', 'EB', 'EB_MISALIGN') + tuple(f'EB_f{f:g}' for f in LADDER)
ISP_ALPHAS = np.logspace(0, 10, 21)
N_COMPONENTS = 100
LAGS = (-1, 0, 1)
CHUNK = 16384
GROUP = 8  # Weight sets per kernel pass.
CACHE = Path('data/cache/v4')
OUT = Path('results/v4')


def file_sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def code_sha256():
    digest = hashlib.sha256(w3.code_sha256().encode())
    for name in CORE:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_savez(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def manifests_all():
    return {s: json.loads(Path(f'data/semantic/{s}/semantic_manifest.json').read_text()) for s in bud.SUBJECTS}


# ----------------------------------------------------------------------------------------------- data
class Target:
    """One target subject on one training clock: indices, caches and story metadata.

    Targets are never loaded wholesale: budget rows load their own targets (``Budget``), and only the
    requested evaluation splits are materialized (no test target is read before stage evaluate).
    """

    def __init__(self, cfg, subject, clock, device, eval_splits=()):
        data = cfg['data']
        self.cfg, self.subject, self.clock, self.device = cfg, subject, clock, torch.device(device)
        self.manifest_path = f'data/semantic/{subject}/semantic_manifest.json'
        common = dict(window=data['window'], phase=data['target_phase'], cutoffs=data['cutoffs'])
        stride = 1 if clock == 'dense' else data['stride']
        self.index = {'train': TargetIndex(self.manifest_path, 'train', stride=stride, **common)}
        for s in ('val', 'test'):
            self.index[s] = TargetIndex(self.manifest_path, s, stride=data['stride'], **common)
        root = Path(cfg['cache_root'])
        self.train_dir = root.parent / (root.name + '_dense') / subject if clock == 'dense' else root / subject
        self.cache = {'train': verify_cache(self.index['train'], cfg['raw_root'], self.train_dir)[0]}
        for s in eval_splits:
            self.cache[s] = verify_cache(self.index[s], cfg['raw_root'], root / subject)[0]
        self.split = {s: Split(self.index[s], 'cpu') for s in eval_splits}
        train = self.index['train']
        self.row_story = np.array([train.runs[ri]['story'] for ri, _ in train.items])
        self.row_tr = train.target_trs
        self.row_run = train.run_ids
        self.row_session = [train.sessions[r] for r in train.run_ids]
        self.run_sessions = dict(train.sessions)
        self.manifest = json.loads(Path(self.manifest_path).read_text())
        self.runs_by_story = {r['story']: r for r in self.manifest['runs']}
        self.n_tr = {st: int(r['n_tr']) for st, r in self.runs_by_story.items()}
        self.split_stories = {s: sorted({r['story'] for r in self.index[s].runs}) for s in self.index}
        self.alphas = np.logspace(*cfg['alphas'])

    def rows_of(self, stories):
        return np.flatnonzero(np.isin(self.row_story, list(set(stories))))


class Budget:
    """Raw design rows of one budget on the GPU (float32), standardized per fold in float64."""

    def __init__(self, tgt, name, stories, eval_splits):
        self.tgt, self.name, self.stories = tgt, name, list(stories)
        self.rows = tgt.rows_of(stories)
        dev = tgt.device
        self.raw = {d: torch.from_numpy(np.array(np.load(tgt.cache['train'][d], mmap_mode='r')[self.rows],
                                                 dtype=np.float32)).to(dev) for d in DELAYS}
        self.eval_raw = {s: {d: torch.from_numpy(np.array(np.load(tgt.cache[s][d], mmap_mode='r'),
                                                          dtype=np.float32)).to(dev) for d in DELAYS}
                         for s in eval_splits}
        self.y = tgt.index['train'].targets(self.rows).astype(np.float64)
        self.story = tgt.row_story[self.rows]
        self.tr = tgt.row_tr[self.rows]
        self.run = tgt.row_run[self.rows]
        self.session = [tgt.row_session[i] for i in self.rows]

    def positions(self, stories):
        return np.flatnonzero(np.isin(self.story, list(set(stories))))

    def log(self, prov, pos, kind, purpose):
        """Record the target stories of the rows ``pos`` actually used (provenance from arrays, not intent)."""
        for st in sorted(set(self.story[np.asarray(pos, dtype=np.int64)])):
            prov.record(self.tgt.subject, st, kind, purpose)

    def standardized(self, d, stats_pos, prov=None):
        if prov is not None:
            self.log(prov, stats_pos, 'brain', 'standardize')
        x = self.raw[d].to(torch.float64)
        mean, std = column_stats(x, torch.as_tensor(stats_pos, device=x.device))
        return x.sub_(mean).div_(std), (mean, std)


# --------------------------------------------------------------------------------------------- priors
class References:
    """Reference listeners' PCA bases and per-story projections (label-free; rule R3)."""

    def __init__(self, cfg, target, prov, device):
        self.cfg, self.target, self.prov, self.device = cfg, target, prov, device
        self.subjects = [s for s in bud.SUBJECTS if s != target.subject]
        self.manifests = {s: json.loads(Path(f'data/semantic/{s}/semantic_manifest.json').read_text())
                          for s in self.subjects}
        self.excluded = sorted(set(target.split_stories['val']) | set(target.split_stories['test']))
        self.proj, self.basis_info = {}, {}
        self.bases = {s: self._basis(s) for s in self.subjects}

    def _basis(self, ref):
        runs = {r['story']: r for r in self.manifests[ref]['runs']}
        allowed = sorted(set(runs) - set(self.excluded))
        sources = [[runs[st]['source_key'], (Path(self.cfg['raw_root']) / runs[st]['source_key']).stat().st_size]
                   for st in allowed]
        key = sha(json.dumps([ref, allowed, sources, N_COMPONENTS, 0, file_sha('src/priors/isp.py')]))[:20]
        path = CACHE / 'bases' / f'{ref}_{key}.npz'
        for st in allowed:
            self.prov.record(ref, st, 'brain', 'pca_basis')
        self.basis_info[ref] = dict(key=key, stories=len(allowed),
                                    minutes=bud.bold_minutes([int(runs[st]['n_tr']) for st in allowed]))
        if path.exists():
            z = np.load(path)
            return z['mean'], z['comps']
        mean, comps = ispm.reference_basis([ispm.load_bold(self.cfg['raw_root'], runs[st]) for st in allowed],
                                           n_components=N_COMPONENTS, seed=0, device=self.device)
        atomic_savez(path, mean=mean, comps=comps, stories=np.array(allowed))
        return mean, comps

    def projection(self, ref, story):
        """Reference BOLD at a target budget story, projected on its basis: [n_tr, K]."""
        self.prov.record(ref, story, 'brain', 'isp_predictor')
        key = (ref, story)
        if key not in self.proj:
            run = next(r for r in self.manifests[ref]['runs'] if r['story'] == story)
            self.proj[key] = ispm.project(ispm.load_bold(self.cfg['raw_root'], run), self.bases[ref], self.device)
        return self.proj[key]

    def design(self, stories, misaligned=False):
        return ispm.design({ref: {st: self.projection(ref, st) for st in stories} for ref in self.subjects},
                           stories, LAGS, misaligned)


class Priors:
    """Fold-level score maps for every arm, computed from the fold's fitting stories only."""

    def __init__(self, tgt, refs, prov, device, shared):
        self.tgt, self.refs, self.prov, self.device, self.shared = tgt, refs, prov, device, set(shared)
        self.bold, self.coords = {}, None

    def target_bold(self, story, purpose):
        self.prov.record(self.tgt.subject, story, 'brain', purpose)
        if story not in self.bold:
            self.bold[story] = ispm.load_bold(self.tgt.cfg['raw_root'], self.tgt.runs_by_story[story])
        return self.bold[story]

    def isp(self, stories, misaligned=False):
        stories = [s for s in stories if s in self.shared]
        x = self.refs.design(stories, misaligned=misaligned)
        r, alpha, curve = ispm.isp_scores({s: self.target_bold(s, 'isp_response') for s in stories}, x, stories,
                                          ISP_ALPHAS, self.device)
        return r, dict(alpha=float(alpha), alpha_at_edge=bool(alpha in (ISP_ALPHAS[0], ISP_ALPHAS[-1])),
                       stories=stories, positive=float((r > 0).mean()), curve=np.asarray(curve).tolist())

    def self_prior(self, stories):
        if self.coords is None:
            from src.windows import resolve_path
            self.coords = np.load(resolve_path(self.tgt.manifest['normalization_path'],
                                               self.tgt.manifest_path))['coords']
        runs = {s: self.target_bold(s, 'isp_response') for s in stories}
        r, info = ispm.self_scores(runs, self.coords, list(stories), n_components=N_COMPONENTS, margin=4,
                                   alphas=ISP_ALPHAS, device=self.device)
        return r, {k: v for k, v in info.items() if isinstance(v, (int, float, str, list))} if isinstance(info, dict) else {}

    def acoustic_features(self, bd, fit_pos):
        from src.priors import acoustic as acm
        per_story, info = {}, {}
        for st in sorted(set(bd.story[fit_pos])):
            self.prov.record(self.tgt.subject, st, 'audio', 'own_encoding')
            per_story[st], info[st] = acm.story_features(st, self.tgt.runs_by_story[st], self.tgt.manifest_path,
                                                         return_info=True)
        feats = acm.acoustic_rows(per_story, list(bd.story), bd.tr, rows=fit_pos)
        return feats, {s: {k: v for k, v in i.items() if isinstance(v, (str, int, float))} for s, i in info.items()}


def soft_weights(scores, gamma, broadcast=False):
    """{d: w} from score arrays (per column or per voxel), rule max(s, 0)^gamma; mean(w^2) = 1."""
    maps = {d: scores for d in DELAYS} if broadcast else scores
    flat = np.concatenate([np.asarray(maps[d], dtype=np.float64) for d in DELAYS])
    positive = float((flat > 0).mean())
    fallback = positive < FALLBACK_FRACTION
    w = np.ones_like(flat) if fallback else np.maximum(flat, 0.0) ** float(gamma)
    w = w / np.sqrt(np.mean(np.square(w)))
    out, start = {}, 0
    for d in DELAYS:
        n = len(maps[d])
        out[d] = w[start:start + n]
        start += n
    return out, dict(fallback=bool(fallback), positive_fraction=positive, nonzero_fraction=float((w > 0).mean()))


# --------------------------------------------------------------------------------------------- engine
def nested_setup(tgt, bd, name, fit_pos, session_folds, held_session):
    """Encoding groups and folds over the fitting rows (half-story groups; §12 rule at ALL)."""
    if name == 'ALL':
        groups = list(bd.session)
        enc = [tuple(f) for f in session_folds if held_session is None or tuple(f) != tuple(held_session)]
    else:
        groups = bud.half_groups(bd.story, bd.tr, tgt.n_tr)
        enc = [(g,) for g in sorted({groups[i] for i in fit_pos})]
    return groups, enc


def fold_maps(tgt, bd, name, fit_pos, session_folds, held_session, arms, pri, oracle):
    """Score maps for all arms from the fitting rows ``fit_pos`` (one delay design in memory at a time)."""
    prov = pri.prov
    groups, enc = nested_setup(tgt, bd, name, fit_pos, session_folds, held_session)
    fit_stories = [s for s in bd.stories if s in set(bd.story[fit_pos])]
    maps, info = {}, {}
    need_own = 'W1' in arms or any(a.startswith('EB') for a in arms)
    need_null = any(a.startswith('EB') for a in arms)
    need_pac = 'P_AC' in arms
    own_ok = need_own and len(enc) >= 2
    feats = None
    if need_pac and len(enc) >= 2:
        feats, info['pac'] = pri.acoustic_features(bd, fit_pos)
    if own_ok or feats is not None:
        if own_ok:
            bd.log(prov, fit_pos, 'transcript', 'own_encoding')
            maps['own'], info['own_alpha'] = {}, {}
            if need_null:
                maps['sigma2'], info['null'] = {}, {}
        if feats is not None:
            maps['pac'] = {}
        for d in DELAYS:
            x = bd.standardized(d, fit_pos, prov)[0]
            if own_ok:
                r, alpha, _ = encoding_scores(bd.y, x, list(fit_pos), groups, enc, tgt.alphas, tgt.device)
                maps['own'][d], info['own_alpha'][d] = r, float(alpha)
                if need_null:
                    inf = {}
                    maps['sigma2'][d] = ebm.null_sigma2(bd.y, x, list(fit_pos), groups, enc, tgt.alphas, tgt.device,
                                                        list(bd.run), bd.tr, info=inf)
                    info['null'][d] = {k: inf.get(k) for k in ('sigma2_by_shift', 'alpha_by_shift', 'shift_rows',
                                                               'min_tr_offset', 'min_tr_offset_overall')}
            if feats is not None:
                from src.priors import acoustic as acm
                maps['pac'][d] = acm.acoustic_scores(feats, x, list(fit_pos), groups, enc, tgt.alphas, tgt.device)[0]
            del x
            torch.cuda.empty_cache()
    if any(a in arms for a in ('PRIOR', 'EB')) or any(a.startswith('EB_f') for a in arms):
        maps['isp'], info['isp'] = pri.isp(fit_stories)
    if 'EB_MISALIGN' in arms:
        maps['isp_mis'], info['isp_mis'] = pri.isp(fit_stories, misaligned=True)
    if 'EB_SELF' in arms:
        maps['self'], info['self'] = pri.self_prior([s for s in fit_stories if s in pri.shared])
    if 'ORACLE' in arms:
        maps['oracle'] = oracle
    return maps, info


def eb_scores(maps, p):
    """Per-delay EB posterior on the r scale (tanh of the posterior mean) and its diagnostics."""
    out, diag = {}, {}
    for d in DELAYS:
        m, inf = ebm.eb_posterior(ebm.fisher(maps['own'][d]), p, maps['sigma2'][d])
        out[d], diag[d] = np.tanh(m), {k: float(v) for k, v in inf.items()}
    return out, diag


def arm_weights(arm, maps, subject):
    """{gamma: ({d: w} or None, info)} for one arm (plus 'eb' diagnostics); None if not estimable here."""
    if arm == 'W0':
        return {1.0: (None, dict(rule='uniform'))}
    simple = {'W1': ('own', False), 'ORACLE': ('oracle', False), 'PRIOR': ('isp', True), 'P_AC': ('pac', False)}
    if arm in simple:
        src = maps.get(simple[arm][0])
        return None if src is None else {g: soft_weights(src, g, broadcast=simple[arm][1]) for g in GAMMAS}
    if not arm.startswith('EB'):
        raise ValueError(f'Unknown arm {arm}')
    if 'own' not in maps or 'sigma2' not in maps:
        return None
    if arm == 'EB':
        p = ebm.fisher(maps['isp'])
    elif arm.startswith('EB_f'):
        f = float(arm[4:])
        p = ispm.ladder(ebm.fisher(maps['isp']), f, ispm.ladder_seed(subject, f))
    elif arm == 'EB_SELF':
        p = ebm.fisher(maps['self'])
    elif arm == 'EB_MISALIGN':
        p = ebm.fisher(maps['isp_mis'])
    else:
        raise ValueError(f'Unknown arm {arm}')
    src, diag = eb_scores(maps, p)
    return {g: soft_weights(src, g) for g in GAMMAS} | {'eb': diag}


def build_kernels(bd, stats_pos, weight_sets, kernel_pos, eval_splits=(), prov=None):
    """Weighted linear kernels over ``kernel_pos`` and cross-kernels for eval splits, per weight set."""
    dev = bd.raw[1].device
    kp = torch.as_tensor(kernel_pos, device=dev)
    kernels = {k: None for k in weight_sets}
    cross = {k: {s: None for s in eval_splits} for k in weight_sets}
    for d in DELAYS:
        x, (mean, std) = bd.standardized(d, stats_pos, prov)
        xk = x.index_select(0, kp)
        del x
        evals = {s: bd.eval_raw[s][d].to(torch.float64).sub_(mean).div_(std) for s in eval_splits}
        for key, wd in weight_sets.items():
            w = None if wd is None else torch.as_tensor(wd[d], dtype=torch.float64, device=dev)
            keep = torch.arange(xk.shape[1], device=dev) if w is None else torch.nonzero(w).squeeze(1)
            for start in range(0, len(keep), CHUNK):
                cols = keep[start:start + CHUNK]
                xb = xk.index_select(1, cols)
                if w is not None:
                    xb.mul_(w.index_select(0, cols))
                term = xb @ xb.T
                kernels[key] = term if kernels[key] is None else kernels[key].add_(term)
                for s, e in evals.items():
                    eb_ = e.index_select(1, cols)
                    if w is not None:
                        eb_.mul_(w.index_select(0, cols))
                    c = eb_ @ xb.T
                    cross[key][s] = c if cross[key][s] is None else cross[key][s].add_(c)
                del xb, term
        del xk, evals
        torch.cuda.empty_cache()
    for key in kernels:
        kernels[key] = (kernels[key] + kernels[key].T) * 0.5
    return kernels, cross


def select(curves, rules, alphas):
    best = None
    for rule in rules:
        for k in range(len(alphas) - 1, -1, -1):
            v = curves[rule][k]
            if np.isfinite(v) and (best is None or v > best[0] + TIE):
                best = (v, rule, k)
    if best is None:
        return None
    return dict(gamma=float(best[1]), alpha=float(alphas[best[2]]), cv_score=float(best[0]),
                alpha_at_grid_edge=best[2] in (0, len(alphas) - 1))


def decoder_folds(tgt, name, stories):
    """(held story tuples, §12 session folds at ALL or None), in the registered order."""
    if name != 'ALL':
        return [tuple(f) for f in bud.inner_folds(name, stories)], None
    rows = tgt.rows_of(stories)
    folds, _ = budget_folds(tgt.row_session, list(rows), fixed_folds(tgt.row_session))
    story_folds = [tuple(sorted({tgt.row_story[i] for i in rows if tgt.row_session[i] in set(f)})) for f in folds]
    if story_folds != [tuple(f) for f in bud.all_folds(tgt.manifest)]:
        raise ValueError('ALL folds differ from budgets.all_folds')
    return story_folds, [tuple(f) for f in folds]


def run_cv(tgt, bd, name, arms, pri, prov, oracle, story_folds, session_folds, log_fold):
    """Inner decoder CV over the budget: per (arm, gamma) per-run curves, then selections."""
    alphas = tgt.alphas
    results, logs = {}, []
    y = torch.as_tensor(bd.y, dtype=torch.float64, device=tgt.device)
    for i, held in enumerate(story_folds):
        start = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with prov.inner_fold(i):
            fit_pos = bd.positions([s for s in bd.stories if s not in held])
            hold_pos = bd.positions(held)
            maps, info = fold_maps(tgt, bd, name, fit_pos, session_folds,
                                   session_folds[i] if session_folds else None, arms, pri, oracle)
            weight_sets, meta = {}, {}
            for arm in arms:
                ws = arm_weights(arm, maps, tgt.subject)
                if ws is None:
                    continue
                for g, val in ws.items():
                    if g == 'eb':
                        meta[f'{arm}|eb'] = val
                    else:
                        weight_sets[(arm, g)], meta[(arm, g)] = val
            kernel_pos = np.concatenate([fit_pos, hold_pos])
            nf = len(fit_pos)
            fit_t, hold_t = (torch.as_tensor(p, device=tgt.device) for p in (fit_pos, hold_pos))
            held_runs = list(dict.fromkeys(bd.run[hold_pos]))
            codes = torch.as_tensor([held_runs.index(r) for r in bd.run[hold_pos]], device=tgt.device)
            keys = list(weight_sets)
            for g0 in range(0, len(keys), GROUP):
                group = {k: weight_sets[k] for k in keys[g0:g0 + GROUP]}
                kernels, _ = build_kernels(bd, fit_pos, group, kernel_pos, prov=prov)
                bd.log(prov, fit_pos, 'brain', 'decoder_fit')
                bd.log(prov, fit_pos, 'transcript', 'decoder_fit')
                bd.log(prov, hold_pos, 'brain', 'decoder_score')
                bd.log(prov, hold_pos, 'transcript', 'decoder_score')
                for key in list(kernels):
                    k = kernels.pop(key)
                    path = KernelRidgePath(k[:nf, :nf].contiguous(), y.index_select(0, fit_t), tgt.device)
                    proj = path.project(k[nf:, :nf].contiguous())
                    r = run_feature_r(path.predict_path(proj, alphas), y.index_select(0, hold_t), codes, len(held_runs))
                    results.setdefault(key, []).append((held_runs, torch.nanmean(r, dim=-1).cpu().numpy()))
                    del k, path, proj, r
                del kernels
                torch.cuda.empty_cache()
        logs.append(dict(held=list(held), n_fit=int(nf), n_held=int(len(hold_pos)), seconds=time.monotonic() - start,
                         peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
                         estimable=sorted({k[0] for k in weight_sets}), priors=info,
                         weights={f'{k[0]}|{k[1]:g}': v for k, v in meta.items() if isinstance(k, tuple)},
                         eb={k: v for k, v in meta.items() if isinstance(k, str)}))
        log_fold(logs[-1])
    curves, selections = {}, {}
    for arm in arms:
        rules = [1.0] if arm == 'W0' else list(GAMMAS)
        per_rule = {}
        for g in rules:
            entries = results.get((arm, g), [])
            if len(entries) != len(story_folds):
                continue
            runs = [r for hr, _ in entries for r in hr]
            per_run = np.concatenate([v for _, v in entries], axis=-1)
            per_rule[g] = session_macro_scores(per_run, [tgt.run_sessions[r] for r in runs])
        curves[arm] = {f'{g:g}': [None if not np.isfinite(x) else float(x) for x in c] for g, c in per_rule.items()}
        selections[arm] = select(per_rule, rules, alphas) if per_rule else None
    return selections, curves, logs


def final_fit(tgt, bd, name, arms, selections, eval_splits, pri, prov, oracle, session_folds):
    """Weights from the whole budget for each selected (arm, gamma); predictions for the eval splits."""
    all_pos = np.arange(len(bd.rows))
    maps, info = fold_maps(tgt, bd, name, all_pos, session_folds or [], None, arms, pri, oracle)
    sets, meta = {}, {}
    for arm in arms:
        sel = selections.get(arm)
        if sel is None:
            continue
        ws = arm_weights(arm, maps, tgt.subject)
        if ws is None or sel['gamma'] not in ws:
            continue
        sets[arm], meta[arm] = ws[sel['gamma']]
        if 'eb' in ws:
            meta[arm] = dict(meta[arm], eb=ws['eb'])
    y = torch.as_tensor(bd.y, dtype=torch.float64, device=tgt.device)
    predictions, keys = {}, list(sets)
    for g0 in range(0, len(keys), GROUP):
        group = {k: sets[k] for k in keys[g0:g0 + GROUP]}
        kernels, cross = build_kernels(bd, all_pos, group, all_pos, eval_splits, prov=prov)
        bd.log(prov, all_pos, 'brain', 'decoder_fit')
        bd.log(prov, all_pos, 'transcript', 'decoder_fit')
        for arm in list(kernels):
            k = kernels.pop(arm)
            path = KernelRidgePath(k, y, tgt.device)
            predictions[arm] = ({s: path.predict(path.project(cross[arm][s]), selections[arm]['alpha']).cpu().numpy()
                                 for s in eval_splits}, path.diagnostics)
            del k, path
        del kernels, cross
        torch.cuda.empty_cache()
    for s in eval_splits:
        for st in tgt.split_stories[s]:
            prov.record(tgt.subject, st, 'brain', 'decoder_score')
            prov.record(tgt.subject, st, 'transcript', 'decoder_score')
    return predictions, dict(info, weights=meta)


def report(tgt, bd, prediction, split):
    mean, std = bd.y.mean(0), np.maximum(bd.y.std(0), 1e-6)
    sp = tgt.split[split]
    return score(prediction, sp.targets, sp.run_ids, sp.target_trs, sp.sessions, mean, std,
                 tgt.cfg['evaluation']['exclusions'])


def oracle_scores(tgt, prov):
    """W1 scores from all training labels (§12 rule); over budget by design (purpose 'oracle')."""
    meta = json.loads((tgt.train_dir / 'train.meta.json').read_text())
    key = sha(json.dumps([tgt.subject, tgt.clock, list(tgt.alphas), meta, file_sha('src/encoding_weights.py'),
                          file_sha('scripts/run_v3_weights.py')]))[:20]
    path = CACHE / 'oracle' / f'{tgt.subject}_{tgt.clock}_{key}.npz'
    for st in tgt.split_stories['train']:
        prov.record(tgt.subject, st, 'transcript', 'oracle')
        prov.record(tgt.subject, st, 'brain', 'oracle')
    if path.exists():
        z = np.load(path)
        return {d: z[f'd{d}'] for d in DELAYS}
    rows = np.arange(len(tgt.row_story))
    folds, _ = budget_folds(tgt.row_session, list(rows), fixed_folds(tgt.row_session))
    designs = w3.Designs(tgt.cache, list(rows), tgt.device)
    y = tgt.index['train'].targets()
    out = {}
    for d in DELAYS:
        x = designs.train(d)
        out[d] = encoding_scores(y, x, list(rows), tgt.row_session, [tuple(f) for f in folds], tgt.alphas,
                                 tgt.device)[0]
        del x
        torch.cuda.empty_cache()
    atomic_savez(path, **{f'd{d}': v for d, v in out.items()})
    return out


def ledger_for(tgt, bd, arm, refs, isp_stories):
    """Per-arm label/exposure ledger (FUTURE_WORK_V4 §3)."""
    from src.transcripts import run_words
    data = tgt.cfg['data']
    stories = tgt.split_stories['train'] if arm == 'ORACLE' else bd.stories
    dense_idx = tgt.index['train'] if tgt.clock == 'dense' else TargetIndex(
        tgt.manifest_path, 'train', stride=1, window=data['window'], phase=data['target_phase'], cutoffs=data['cutoffs'])
    keep = set(stories)
    dense_pairs = [(dense_idx.runs[ri]['run_id'], t) for ri, t in dense_idx.items if dense_idx.runs[ri]['story'] in keep]
    stride4 = sum(t % data['stride'] == data['target_phase'] for _, t in dense_pairs)
    words = {r['run_id']: run_words(r, tgt.manifest_path) for r in tgt.index['train'].runs if r['story'] in keep}
    own_pairs = dense_pairs if tgt.clock == 'dense' else [p for p in dense_pairs if p[1] % data['stride'] == data['target_phase']]
    ledger = bud.Ledger().target(len(dense_pairs), int(stride4), bud.bold_minutes([tgt.n_tr[s] for s in stories]))
    ledger.transcript(words, own_pairs)
    if arm in REFERENCE_ARMS:
        isp_minutes = len(refs.subjects) * bud.bold_minutes([tgt.n_tr[s] for s in isp_stories])
        basis_minutes = sum(v['minutes'] for v in refs.basis_info.values())
        ledger.reference(refs.subjects, isp_minutes + basis_minutes)
    return ledger.to_dict()


# --------------------------------------------------------------------------------------------- stages
def stage_g1(cfg, subject, device):
    """Training data only: Spearman(ISP at S3 and S6, full-label stride-4 W1 map, max over delays)."""
    from scipy.stats import spearmanr
    tgt = Target(cfg, subject, 'stride4', device)
    shared = bud.shared_stories(manifests_all())
    budgets = bud.ds003020_budgets(tgt.manifest, shared)
    prov = Provenance(subject, 'S', budgets['S6'], tgt.split_stories['val'], tgt.split_stories['test'],
                      arm='ORACLE', budget_name='S6', stage='g1')
    refs = References(cfg, tgt, prov, device)
    pri = Priors(tgt, refs, prov, device, shared)
    full = oracle_scores(tgt, prov)
    full_max = np.max(np.stack([full[d] for d in DELAYS]), axis=0)
    out = dict(protocol='ssltang-v4', stage='g1', subject=subject, code_sha256=code_sha256(),
               config_sha256=v3.config_sha256(cfg), budgets={})
    for name in ('S3', 'S6'):
        r, info = pri.isp(budgets[name])
        out['budgets'][name] = dict(stories=budgets[name], spearman=float(spearmanr(r, full_max).correlation),
                                    isp_alpha=info['alpha'], isp_alpha_at_edge=info['alpha_at_edge'],
                                    isp_positive=info['positive'])
    prov.check()
    out['provenance'] = prov.to_dict()
    errors = audit_v4.cell_errors(out) + audit_v4.scope_errors(out, prov)
    if errors:
        raise SystemExit(f'Registered-scope audit failed: {errors}')
    save_json(out, OUT / subject / 'g1.json')
    print(json.dumps(dict(subject=subject, **{k: round(v['spearman'], 4) for k, v in out['budgets'].items()})),
          flush=True)


def develop_cell(subject, clock, name, arm):
    return OUT / subject / clock / 'develop' / 'cells' / f'{name}_{arm}.json'


def stage_fit(cfg, subject, clock, budget_names, arms, stage, device):
    eval_splits = ('val',) if stage == 'develop' else ('val', 'test')
    tgt = Target(cfg, subject, clock, device, eval_splits)
    shared = bud.shared_stories(manifests_all())
    budgets = bud.ds003020_budgets(tgt.manifest, shared)
    folder = OUT / subject / clock / stage
    for name in budget_names:
        stories = budgets[name]
        use = [a for a in arms if name != 'ALL' or a in ALL_ARMS]
        story_folds, session_folds = decoder_folds(tgt, name, stories)
        common = dict(folds=story_folds, budget_name=name, stage=stage)
        prov = Provenance(subject, 'S', stories, tgt.split_stories['val'], tgt.split_stories['test'], **common)
        oracle_prov = Provenance(subject, 'S', stories, tgt.split_stories['val'], tgt.split_stories['test'],
                                 arm='ORACLE', **common)
        refs = References(cfg, tgt, prov, device)
        pri = Priors(tgt, refs, prov, device, shared)
        oracle = oracle_scores(tgt, oracle_prov) if 'ORACLE' in use else None
        print(json.dumps(dict(subject=subject, clock=clock, budget=name, stories=len(stories), arms=use)), flush=True)
        bd = Budget(tgt, name, stories, eval_splits)
        if stage == 'evaluate':  # Reuse the develop selections (identical inner CV under the locked code).
            selections, curves, logs = {}, {}, []
            for arm in use:
                dev_cell = json.loads(develop_cell(subject, clock, name, arm).read_text())
                if dev_cell.get('code_sha256') != code_sha256() or dev_cell.get('provenance', {}).get('folds') is None:
                    raise SystemExit(f'{develop_cell(subject, clock, name, arm)} is not from the locked code')
                selections[arm], curves[arm], logs = dev_cell['selection'], dev_cell['cv_curves'], dev_cell['folds']
            for i, held in enumerate(story_folds):  # The reused inner CV read exactly the develop folds' material.
                with prov.inner_fold(i):
                    fit_pos, hold_pos = bd.positions([s for s in stories if s not in held]), bd.positions(held)
                    log_reads(bd, prov, fit_pos, hold_pos, use, pri, shared)
        else:
            selections, curves, logs = run_cv(
                tgt, bd, name, use, pri, prov, oracle, story_folds, session_folds,
                lambda e: print(json.dumps(dict(budget=name, held=e['held'], seconds=round(e['seconds'], 1),
                                                peak_gpu_gib=e['peak_gpu_gib'])), flush=True))
        predictions, final = final_fit(tgt, bd, name, use, selections, eval_splits, pri, prov, oracle, session_folds)
        logs_by_arm = {arm: (prov.merged(oracle_prov, arm=arm) if arm == 'ORACLE' else prov.merged(arm=arm))
                       for arm in use}
        isp_used = bud.isp_stories(stories, shared)
        for arm in use:
            log = logs_by_arm[arm]
            log.check()  # R1-R7: refuse to write any score on a violation.
            cell = dict(protocol='ssltang-v4', stage=stage, subject=subject, clock=clock, budget=name, arm=arm,
                        regime='S', estimable=arm in predictions, selection=selections.get(arm),
                        cv_curves=curves.get(arm), folds=logs,
                        final_priors={k: v for k, v in final.items() if k in ('isp', 'isp_mis', 'own_alpha', 'null',
                                                                                'self', 'pac')},
                        final_weights=final['weights'].get(arm), ledger=ledger_for(tgt, bd, arm, refs, isp_used),
                        reference_bases=refs.basis_info if arm in REFERENCE_ARMS else None,
                        provenance=log.to_dict(), code_sha256=code_sha256(), config_sha256=v3.config_sha256(cfg))
            if arm in predictions:
                pred, diag = predictions[arm]
                cell['final_fit'] = diag
                for sp in eval_splits:
                    cell[sp] = report(tgt, bd, pred[sp], sp)
                if 'test' in eval_splits:
                    ev, spl = cfg['evaluation'], tgt.split['test']
                    cell['test_shift_null'] = shift_null(pred['test'], spl.targets, spl.run_ids, spl.sessions,
                                                         cfg['data']['stride'], ev['shift_min_tr'], ev['shift_draws'],
                                                         ev['shift_seed'])
            errors = audit_v4.cell_errors(cell) + audit_v4.scope_errors(cell, log)
            if errors:
                raise SystemExit(f'Registered-scope audit failed for {name} {arm}: {errors}')
            if arm in predictions:
                (folder / 'predictions').mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / 'predictions' / f'{name}_{arm}.npz',
                                    **{k: v.astype(np.float16) for k, v in predictions[arm][0].items()})
                shown = ' '.join(f"{k}={cell[k]['session_macro_feature_correlation']:.4f}" for k in eval_splits)
                print(f"{subject} {clock} {name} {arm}: gamma={selections[arm]['gamma']:g} "
                      f"alpha={selections[arm]['alpha']:.3g} cv={selections[arm]['cv_score']:.4f} {shown}", flush=True)
            save_json(cell, folder / 'cells' / f'{name}_{arm}.json')
        del bd
        torch.cuda.empty_cache()


def log_reads(bd, prov, fit_pos, hold_pos, arms, pri, shared):
    """Evaluate stage: record the inner-CV reads that the reused develop selections were computed from."""
    for kind, purpose in (('brain', 'standardize'), ('brain', 'decoder_fit'), ('transcript', 'decoder_fit')):
        bd.log(prov, fit_pos, kind, purpose)
    bd.log(prov, hold_pos, 'brain', 'decoder_score')
    bd.log(prov, hold_pos, 'transcript', 'decoder_score')
    if 'W1' in arms or any(a.startswith('EB') for a in arms):
        bd.log(prov, fit_pos, 'transcript', 'own_encoding')
    if 'P_AC' in arms:
        bd.log(prov, fit_pos, 'audio', 'own_encoding')
    if any(a in REFERENCE_ARMS or a == 'EB_SELF' for a in arms):
        for st in sorted(set(bd.story[fit_pos]) & set(shared)):
            prov.record(bd.tgt.subject, st, 'brain', 'isp_response')
            if any(a in REFERENCE_ARMS for a in arms):
                for ref in pri.refs.subjects:
                    prov.record(ref, st, 'brain', 'isp_predictor')


def g0_passed():
    path = OUT / 'g0.json'
    if not path.exists():
        return False
    g0 = json.loads(path.read_text())
    return bool(g0.get('passed')) and g0.get('code_sha256') == code_sha256()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['g1', 'develop', 'evaluate'], required=True)
    p.add_argument('--subject', required=True, choices=bud.SUBJECTS)
    p.add_argument('--clock', choices=['dense', 'stride4'], default='dense')
    p.add_argument('--budgets', nargs='+', default=['S3', 'S6', 'S15', 'ALL'], choices=['S3', 'S6', 'S15', 'ALL'])
    p.add_argument('--arms', nargs='+', default=list(TIER1), choices=list(TIER1))
    p.add_argument('--config', default='configs/v3/structured_ridge.yaml')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    cfg = load_config(a.config, [f'data.manifest=data/semantic/{a.subject}/semantic_manifest.json'])
    torch.backends.cuda.matmul.allow_tf32 = False
    v3.verify_lock(cfg)
    if a.stage == 'g1':
        stage_g1(cfg, a.subject, a.device)
        return
    g0_run = a.stage == 'develop' and a.clock == 'stride4' and a.budgets == ['ALL'] and set(a.arms) <= {'W0', 'W1'}
    if not g0_run and not g0_passed():
        raise SystemExit('G0 has not passed under this code (results/v4/g0.json); only the G0(a) run is allowed')
    if a.stage == 'evaluate':
        lock = json.loads(Path('experiments/v4_lock.json').read_text())
        if lock.get('code_sha256') != code_sha256():
            raise SystemExit('Code changed since the v4 lock')
    stage_fit(cfg, a.subject, a.clock, a.budgets, a.arms, a.stage, a.device)


if __name__ == '__main__':
    main()
