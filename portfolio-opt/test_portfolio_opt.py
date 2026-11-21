import numpy as np
import pytest

from portfolio_opt import (
    black_litterman,
    efficient_frontier,
    implied_returns,
    ledoit_wolf_identity,
    max_sharpe,
    min_variance,
    portfolio_stats,
    risk_contributions,
    risk_parity,
)

VOL = np.array([0.18, 0.22, 0.07, 0.12, 0.30])
CORR = np.array(
    [
        [1.0, 0.7, -0.1, 0.5, 0.3],
        [0.7, 1.0, -0.1, 0.5, 0.35],
        [-0.1, -0.1, 1.0, 0.1, 0.0],
        [0.5, 0.5, 0.1, 1.0, 0.25],
        [0.3, 0.35, 0.0, 0.25, 1.0],
    ]
)
COV = CORR * np.outer(VOL, VOL)
MU = np.array([0.07, 0.08, 0.03, 0.05, 0.10])


def test_min_variance_beats_random_portfolios():
    w = min_variance(COV)
    assert w.sum() == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        v = w + rng.normal(0, 0.05, w.size)
        v /= v.sum()
        assert v @ COV @ v >= w @ COV @ w - 1e-14


def test_long_only_min_variance_matches_unconstrained_when_inactive():
    cov = np.diag([0.04, 0.09, 0.16])
    np.testing.assert_allclose(min_variance(cov, long_only=True), min_variance(cov), atol=1e-6)


def test_long_only_constraint_binds():
    vol = np.array([0.10, 0.30, 0.15])
    corr = np.array([[1, 0.95, 0.2], [0.95, 1, 0.2], [0.2, 0.2, 1.0]])
    cov = corr * np.outer(vol, vol)
    w = min_variance(cov)
    assert w.min() < 0  # the unconstrained solution shorts the noisy twin
    lo = min_variance(cov, long_only=True)
    assert lo.min() >= 0 and lo.sum() == pytest.approx(1.0)
    assert lo @ cov @ lo >= w @ cov @ w


def test_tangency_maximises_sharpe_and_frontier_touches_it():
    rf = 0.02
    w = max_sharpe(MU, COV, rf)
    ret, vol = portfolio_stats(w, MU, COV)
    sharpe = (ret - rf) / vol
    rng = np.random.default_rng(1)
    for _ in range(200):
        v = w + rng.normal(0, 0.05, w.size)
        v /= v.sum()
        r, s = portfolio_stats(v, MU, COV)
        assert (r - rf) / s <= sharpe + 1e-12
    (fw,) = efficient_frontier(MU, COV, [ret])
    assert portfolio_stats(fw, MU, COV)[1] <= vol + 1e-12  # frontier point has no more risk than tangency


def test_frontier_hits_target_return_and_is_convex_in_variance():
    targets = np.linspace(0.04, 0.10, 7)
    ws = efficient_frontier(MU, COV, targets)
    np.testing.assert_allclose(ws @ MU, targets, atol=1e-10)
    np.testing.assert_allclose(ws.sum(axis=1), 1.0, atol=1e-10)
    var = np.einsum("ij,jk,ik->i", ws, COV, ws)
    assert np.all(np.diff(var, 2) > 0)


def test_closed_form_frontier_matches_numeric_long_only_when_inactive():
    cov = np.diag([0.04, 0.09, 0.16])
    mu = np.array([0.05, 0.06, 0.07])
    closed = efficient_frontier(mu, cov, [0.06])
    numeric = efficient_frontier(mu, cov, [0.06], long_only=True)
    np.testing.assert_allclose(closed, numeric, atol=1e-5)


def test_risk_parity_equal_contributions():
    w = risk_parity(COV)
    assert w.sum() == pytest.approx(1.0) and np.all(w > 0)
    np.testing.assert_allclose(risk_contributions(w, COV), 0.2, atol=1e-9)


def test_risk_parity_uncorrelated_is_inverse_vol():
    vol = np.array([0.1, 0.2, 0.4])
    w = risk_parity(np.diag(vol**2))
    expected = (1 / vol) / (1 / vol).sum()
    np.testing.assert_allclose(w, expected, atol=1e-10)


def test_risk_budget_respected():
    b = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
    w = risk_parity(COV, budget=b)
    np.testing.assert_allclose(risk_contributions(w, COV), b, atol=1e-9)


def test_risk_parity_rejects_bad_budget():
    with pytest.raises(ValueError):
        risk_parity(COV, budget=np.array([0.5, 0.5, 0.1, 0.0, 0.0]))


def test_reverse_optimisation_round_trip():
    w_mkt = np.array([0.3, 0.2, 0.3, 0.1, 0.1])
    pi = implied_returns(COV, w_mkt, 2.5)
    np.testing.assert_allclose(max_sharpe(pi, COV), w_mkt, atol=1e-10)


def test_bl_without_views_returns_prior():
    pi = implied_returns(COV, np.full(5, 0.2), 2.5)
    P = np.array([[1.0, -1, 0, 0, 0]])
    mean, _ = black_litterman(COV, pi, P, P @ pi)  # view agrees with the prior
    np.testing.assert_allclose(mean, pi, atol=1e-12)


def test_bl_tiny_omega_matches_view_and_big_omega_ignores_it():
    pi = implied_returns(COV, np.full(5, 0.2), 2.5)
    P = np.array([[0.0, 1, 0, 0, -1]])
    Q = np.array([0.04])
    sure, _ = black_litterman(COV, pi, P, Q, omega=np.array([[1e-14]]))
    assert (P @ sure)[0] == pytest.approx(0.04, abs=1e-8)
    vague, _ = black_litterman(COV, pi, P, Q, omega=np.array([[1e8]]))
    np.testing.assert_allclose(vague, pi, atol=1e-6)


def test_bl_matches_precision_weighted_formula():
    pi = implied_returns(COV, np.full(5, 0.2), 2.5)
    P = np.array([[1.0, 0, 0, 0, 0], [0, 0, 1.0, -1, 0]])
    Q = np.array([0.09, 0.01])
    tau, om = 0.05, np.diag([0.001, 0.002])
    mean, cov_post = black_litterman(COV, pi, P, Q, tau=tau, omega=om)
    prec = np.linalg.inv(tau * COV) + P.T @ np.linalg.inv(om) @ P
    ref = np.linalg.solve(prec, np.linalg.inv(tau * COV) @ pi + P.T @ np.linalg.inv(om) @ Q)
    np.testing.assert_allclose(mean, ref, atol=1e-10)
    np.testing.assert_allclose(cov_post, np.linalg.inv(prec), atol=1e-10)


def test_ledoit_wolf_properties():
    rng = np.random.default_rng(2)
    x = rng.multivariate_normal(np.zeros(5), COV, size=60)
    shrunk, delta = ledoit_wolf_identity(x)
    assert 0 < delta < 1
    assert np.all(np.linalg.eigvalsh(shrunk) > 0)
    assert np.trace(shrunk) == pytest.approx(np.trace(np.cov(x.T, bias=True)))
    big = rng.multivariate_normal(np.zeros(5), COV, size=200_000)
    assert ledoit_wolf_identity(big)[1] < 0.01
