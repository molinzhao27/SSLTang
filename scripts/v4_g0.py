#!/usr/bin/env python3
"""PROTOCOL_V4 G0 integrity gate (run on the remote, after the runner's ALL/stride-4 W0/W1 cells for UTS01-03).

(a) W0 and W1 from scripts/run_v4.py at ALL on the stride-4 clock reproduce the §12 development cells of UTS01-03
    (identical gamma and alpha, validation r to 1e-10 relative); the cells must carry the current code fingerprint.
(b) EB identities and (e) planted provenance leaks: the v4 unit tests (tests/test_v4_*.py) must pass.
(c) The UTS01 ISP map reproduces the pilot's Spearman against the full-label encoding map (0.539) to 1e-6 when the
    module pipeline is fed the pilot's PCA (uncentered float32 SVD scores u*s); the production PCA (centered,
    float64 projection) is reported alongside (PROTOCOL_V4 §4).
(d) The synthetic EB recovery test (results/v4/simulate.json) passed.
Writes results/v4/g0.json with an overall verdict and the code fingerprint. No v4 decoding score is computed here.
"""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v4
from src.priors import isp as ispm
from src.runtime import load_config, save_json

SUBJECTS = [f'UTS{i:02d}' for i in range(1, 11)]


def check_a(subject, code):
    out = {}
    for arm, old in (('W0', 'W0_f1'), ('W1', 'W1_f1')):
        new = json.loads(Path(f'results/v4/{subject}/stride4/develop/cells/ALL_{arm}.json').read_text())
        ref = json.loads(Path(f'results/v3w/{subject}/develop/cells/{old}.json').read_text())
        rv, nv = ref['val']['session_macro_feature_correlation'], new['val']['session_macro_feature_correlation']
        gamma_ref = 1.0 if arm == 'W0' else float(ref['selection']['rule'][1])
        same_code = (new.get('code_sha256') == code and new.get('stage') == 'develop' and new.get('clock') == 'stride4'
                     and new.get('budget') == 'ALL')
        out[arm] = dict(alpha_v4=new['selection']['alpha'], alpha_s12=ref['selection']['alpha'],
                        gamma_v4=new['selection']['gamma'], gamma_s12=gamma_ref, val_v4=nv, val_s12=rv,
                        rel_diff=abs(nv - rv) / abs(rv), current_code=same_code,
                        passed=bool(same_code and new['selection']['alpha'] == ref['selection']['alpha']
                                    and new['selection']['gamma'] == gamma_ref and abs(nv - rv) / abs(rv) <= 1e-10))
    return out


def full_label_map(device):
    import torch
    import run_v3 as v3
    import run_v3_weights as w
    from src.encoding_weights import encoding_scores
    from src.structured_ridge import budget_folds
    ctx = v3.Context(load_config('configs/v3/structured_ridge.yaml',
                                 ['data.manifest=data/semantic/UTS01/semantic_manifest.json']), device)
    rows = ctx.budgets['1']
    folds, _ = budget_folds(ctx.row_sessions, rows, ctx.fixed)
    designs = w.Designs(ctx.cache, list(range(len(ctx.row_sessions))), device)
    enc = []
    for d in (1, 2, 3):
        xd = designs.train(d)
        enc.append(encoding_scores(ctx.split['train'].targets, xd, rows, ctx.row_sessions, folds, ctx.alphas, device)[0])
        del xd
        torch.cuda.empty_cache()
    return np.max(np.stack(enc), axis=0)


def check_c(cfg, enc_max, pca, device='cuda'):
    import torch
    target = 'UTS01'
    runs = {s: {r['story']: r for r in json.loads(Path(f'data/semantic/{s}/semantic_manifest.json').read_text())['runs']}
            for s in SUBJECTS}
    shared = sorted(set.intersection(*(set(r) for r in runs.values())))
    stories = [st for st in shared if runs[target][st]['split'] == 'train']
    load = lambda run: ispm.load_bold(cfg['raw_root'], run)
    target_runs = {st: load(runs[target][st]) for st in stories}
    scores = {}
    for ref in (s for s in SUBJECTS if s != target):
        bold = [load(runs[ref][st]) for st in stories]
        if pca == 'pilot':  # scripts/v4_pilot_isp.py: uncentered float32 SVD of the concatenation, scores u*s.
            zt = torch.as_tensor(np.concatenate(bold), dtype=torch.float32, device=device)
            u, sv, _ = torch.linalg.svd(zt, full_matrices=False)
            pcs = (u[:, :100] * sv[:100]).cpu().numpy()
            bounds = np.cumsum([0] + [len(b) for b in bold])
            scores[ref] = {st: pcs[bounds[i]:bounds[i + 1]] for i, st in enumerate(stories)}
            del zt, u
        else:
            basis = ispm.reference_basis(bold, n_components=100, device=device)
            scores[ref] = {st: ispm.project(b, basis, device) for st, b in zip(stories, bold)}
    r, alpha, _ = ispm.isp_scores(target_runs, ispm.design(scores, stories), stories, ispm.PILOT_ALPHAS, device,
                                  groups='pilot')
    rho = float(spearmanr(r, enc_max).correlation)
    pilot = json.loads(Path('results/v4_pilot/UTS01_isp_summary.json').read_text())['spearman_isp_vs_encoding_max']
    return dict(pca=pca, spearman=rho, pilot=pilot, alpha=float(alpha), abs_diff=abs(rho - pilot),
                passed=bool(abs(rho - pilot) <= 1e-6))


def main():
    cfg = load_config('configs/v3/structured_ridge.yaml', [])
    code = run_v4.code_sha256()
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q', *map(str, sorted(Path('tests').glob('test_v4_*.py')))],
                           capture_output=True, text=True)
    sim = json.loads(Path('results/v4/simulate.json').read_text())
    enc_max = full_label_map('cuda')
    out = dict(protocol='ssltang-v4', stage='g0', code_sha256=code,
               a={s: check_a(s, code) for s in ('UTS01', 'UTS02', 'UTS03')},
               tests=dict(returncode=tests.returncode, summary=tests.stdout.strip().splitlines()[-1:]),
               c=check_c(cfg, enc_max, 'pilot'), c_production=check_c(cfg, enc_max, 'production'),
               d=dict(passed=bool(sim.get('passed')), summary=sim.get('summary')))
    out['passed'] = bool(all(v['passed'] for s in out['a'].values() for v in s.values()) and tests.returncode == 0
                         and out['c']['passed'] and out['d']['passed'])
    save_json(out, 'results/v4/g0.json')
    print(json.dumps({k: v for k, v in out.items() if k != 'tests'}, indent=1), flush=True)


if __name__ == '__main__':
    main()
