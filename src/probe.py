"""Validation-selected ridge probes shared by every v2 readout arm.

For each label budget, every option of a feature family (layer, delay set or
component count) is fit on that budget's labeled training rows across one
prespecified alpha grid. The (option, alpha) with the best validation
session-macro feature correlation is selected; test data never enter selection.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch
from .readout import ALPHAS, RidgePath, Standardizer, run_codes, run_correlations, session_macro
from .targets import TargetIndex


class Split:
    """Targets and run/session identity of one held-out or training split."""

    def __init__(self, index: TargetIndex, device='cpu'):
        self.index = index
        self.targets = index.targets()
        self.run_ids, self.target_trs = index.run_ids, index.target_trs
        self.sessions = index.sessions
        self.names, codes = run_codes(self.run_ids)
        self.codes = torch.as_tensor(codes, device=device)
        self.device = device
        self._y = None

    @property
    def y(self):
        if self._y is None:
            self._y = torch.as_tensor(self.targets, dtype=torch.float64, device=self.device)
        return self._y

    def score_alpha_path(self, ridge, projected, alphas):
        return [session_macro(run_correlations(ridge.predict(projected, a), self.y, self.codes,
                                               len(self.names)), self.names, self.sessions)
                for a in alphas]


def select_ridge(options, budgets, train: Split, val: Split, test: Split, alphas=ALPHAS,
                 device='cuda', standardize=True):
    """Return, per budget, the validation grid and the selected predictions.

    ``options`` maps an option name to (train_all, val, test) feature arrays.
    ``budgets`` maps a budget name to labeled training row positions.
    ``standardize`` z-scores each feature using all training inputs (no
    labels); pass False for features whose relative scale is deliberate.
    """
    results = {name: {'grid': {}, 'score': -np.inf, 'option': None, 'alpha': None} for name in budgets}
    prepared = {}
    for option, (x_train, x_val, x_test) in options.items():
        if standardize:
            scale = Standardizer(x_train)
            x_train, x_val, x_test = scale(x_train), scale(x_val), scale(x_test)
        prepared[option] = (x_train, x_val, x_test)
        for budget, rows in budgets.items():
            ridge = RidgePath(x_train[rows], train.targets[rows], device)
            scores = val.score_alpha_path(ridge, ridge.project(x_val), alphas)
            entry = results[budget]
            entry['grid'][option] = scores
            finite = np.isfinite(scores)
            if not finite.any():
                continue
            best = int(np.nanargmax(np.where(finite, scores, np.nan)))
            if scores[best] > entry['score']:
                entry.update(score=float(scores[best]), option=option, alpha=alphas[best],
                             alpha_at_grid_edge=best in (0, len(alphas) - 1),
                             n_features=int(x_train.shape[1]))
            del ridge
    # Test data are projected only after selection, for the selected option.
    for budget, entry in results.items():
        if entry['option'] is None:
            continue
        x_train, x_val, x_test = prepared[entry['option']]
        rows = budgets[budget]
        ridge = RidgePath(x_train[rows], train.targets[rows], device)
        entry['prediction_val'] = ridge.predict(ridge.project(x_val), entry['alpha']).cpu().numpy()
        entry['prediction_test'] = ridge.predict(ridge.project(x_test), entry['alpha']).cpu().numpy()
        del ridge
    return results


def grouped_cv_alpha(x, y, groups, alphas=ALPHAS, device='cuda'):
    """Choose alpha using only labeled training rows: leave-one-group-out CV.

    Each held-out group's mean per-feature correlation is averaged over groups.
    No validation labels are used, so all supervision is counted in the budget.
    """
    groups = np.asarray(groups)
    names = list(dict.fromkeys(groups.tolist()))
    if len(names) < 2:
        raise ValueError('Grouped CV needs at least two labeled groups')
    scores = np.zeros((len(names), len(alphas)))
    for g, name in enumerate(names):
        held = groups == name
        ridge = RidgePath(x[~held], y[~held], device)
        projected = ridge.project(x[held])
        target = torch.as_tensor(y[held], dtype=torch.float64, device=ridge.device)
        codes = torch.zeros(int(held.sum()), dtype=torch.long, device=ridge.device)
        for k, alpha in enumerate(alphas):
            scores[g, k] = run_correlations(ridge.predict(projected, alpha), target, codes, 1)[0]
    mean = np.nanmean(scores, axis=0)
    best = int(np.nanargmax(mean))
    return alphas[best], mean.tolist(), len(names)


def code_fingerprint(root=None):
    """SHA256 over tracked-source contents, recorded because runs may precede a commit."""
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(list((root/'src').rglob('*.py')) + list((root/'scripts').glob('*.py'))):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
