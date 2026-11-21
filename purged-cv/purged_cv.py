"""Purged k-fold cross-validation for financial machine learning.

A label such as "sign of the return over the next h days" is decided by prices on days t+1..t+h,
so neighbouring labels share most of their window, and rolling-window features overlap the same
way. Shuffled k-fold puts those near-duplicates on both sides of the split, and a flexible model
scores well by looking up its neighbours' answers. López de Prado (2018, ch. 7) fixes this by:

- purging: dropping every training sample whose label window overlaps a test label window, and
- embargoing: also dropping a short span of training samples right after each test fold, since
  features built from trailing windows there still contain the test period's prices.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class PurgedKFold(BaseCrossValidator):
    """Contiguous k-fold where sample i's label depends on times [start[i], end[i]].

    Parameters
    ----------
    label_start, label_end : arrays of the first and last time index each label depends on.
    n_splits : number of contiguous test folds.
    embargo : number of samples dropped from training after each test fold.
    """

    def __init__(self, label_start, label_end, n_splits=5, embargo=0):
        self.label_start = np.asarray(label_start)
        self.label_end = np.asarray(label_end)
        if self.label_start.shape != self.label_end.shape or np.any(self.label_end < self.label_start):
            raise ValueError("label_start and label_end must align and satisfy start <= end")
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.embargo = embargo

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        n = len(self.label_start)
        indices = np.arange(n)
        for test in np.array_split(indices, self.n_splits):
            t0, t1 = self.label_start[test].min(), self.label_end[test].max()
            # Overlap of [s, e] with [t0, t1] means s <= t1 and e >= t0.
            overlaps = (self.label_start <= t1) & (self.label_end >= t0)
            after = (indices > test[-1]) & (indices <= test[-1] + self.embargo)
            train = indices[~overlaps & ~after]
            yield train, test


def walk_forward_splits(n, n_splits, horizon):
    """Expanding-window splits that train only on the past, with labels resolved before the test."""
    folds = np.array_split(np.arange(n), n_splits + 1)
    for test in folds[1:]:
        yield np.arange(max(test[0] - horizon, 0)), test


def make_dataset(n, horizon=20, signal=0.0, seed=None):
    """Daily log returns, trailing-window features, and forward-return sign labels.

    With signal = 0 the returns are i.i.d., so no feature can predict the label and the true
    accuracy of any model is 50%. With signal > 0, a feature built from past returns shifts the
    drift of the next h days, planting genuine predictability.

    Returns X, y, label_start, label_end, where sample t's label depends on returns t+1..t+h.
    """
    rng = np.random.default_rng(seed)
    lookbacks = (1, 5, 10, 20, 60)
    burn = max(lookbacks)
    total = n + burn + horizon
    eps = rng.standard_normal(total) * 0.01
    returns = eps.copy()
    if signal:
        # A slow driver observed in the features: the 20-day trailing mean of the innovations.
        driver = np.convolve(eps, np.ones(20) / 20, mode="full")[:total]
        returns[1:] += signal * 0.01 * np.sign(driver[:-1])
    cum = np.concatenate([[0.0], np.cumsum(returns)])
    t = np.arange(burn, burn + n)
    features = [cum[t + 1] - cum[t + 1 - L] for L in lookbacks]
    features.append(np.array([returns[i - 19 : i + 1].std() for i in t]))
    X = np.column_stack(features)
    forward = cum[t + 1 + horizon] - cum[t + 1]
    y = (forward > 0).astype(int)
    idx = np.arange(n)
    return X, y, idx + 1, idx + horizon
