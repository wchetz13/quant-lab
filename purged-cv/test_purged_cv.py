import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

from purged_cv import PurgedKFold, make_dataset, walk_forward_splits


def auc(X, y, cv):
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=0)
    return cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()


@pytest.mark.parametrize("seed", range(5))
def test_no_training_label_overlaps_a_test_label(seed):
    rng = np.random.default_rng(seed)
    n, horizon = int(rng.integers(200, 800)), int(rng.integers(1, 40))
    start = np.arange(n) + 1
    end = start + horizon - 1
    cv = PurgedKFold(start, end, n_splits=int(rng.integers(2, 10)), embargo=int(rng.integers(0, 30)))
    for train, test in cv.split():
        t0, t1 = start[test].min(), end[test].max()
        assert not np.any((start[train] <= t1) & (end[train] >= t0))


def test_test_folds_partition_the_samples():
    start = np.arange(500)
    cv = PurgedKFold(start, start + 10, n_splits=7)
    tests = np.concatenate([test for _, test in cv.split()])
    assert np.array_equal(np.sort(tests), np.arange(500))


def test_embargo_removes_samples_after_each_test_fold():
    start = np.arange(1000)
    plain = PurgedKFold(start, start, n_splits=5, embargo=0)
    embargoed = PurgedKFold(start, start, n_splits=5, embargo=25)
    for (train_a, test), (train_b, _) in zip(plain.split(), embargoed.split()):
        removed = np.setdiff1d(train_a, train_b)
        expected = np.arange(test[-1] + 1, min(test[-1] + 26, 1000))
        assert np.array_equal(removed, expected)


def test_walk_forward_trains_only_on_resolved_past_labels():
    horizon = 20
    for train, test in walk_forward_splits(1000, 4, horizon):
        assert train.max() + horizon < test[0]


def test_labels_are_the_sign_of_the_forward_return():
    h = 10
    X, y, start, end = make_dataset(500, horizon=h, seed=1)
    assert np.all(end - start == h - 1)
    # Column 0 is each day's own 1-day return, so sample t's label is the sign of the next h of them.
    forward = np.array([X[t + 1 : t + 1 + h, 0].sum() for t in range(500 - h)])
    assert np.array_equal(y[: 500 - h], (forward > 0).astype(int))


def test_shuffled_kfold_finds_skill_in_pure_noise_and_purged_cv_does_not():
    X, y, start, end = make_dataset(2000, horizon=20, signal=0.0, seed=0)
    leaky = auc(X, y, KFold(5, shuffle=True, random_state=0))
    honest = auc(X, y, PurgedKFold(start, end, 5, embargo=60))
    assert leaky > 0.62
    assert abs(honest - 0.5) < 0.06


def test_purged_cv_still_detects_a_real_signal():
    X, y, start, end = make_dataset(2000, horizon=20, signal=0.5, seed=0)
    assert auc(X, y, PurgedKFold(start, end, 5, embargo=60)) > 0.6


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(label_start=[0, 1], label_end=[1], n_splits=2),
        dict(label_start=[2, 3], label_end=[1, 4], n_splits=2),
        dict(label_start=[0, 1], label_end=[1, 2], n_splits=1),
    ],
)
def test_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        PurgedKFold(**kwargs)
