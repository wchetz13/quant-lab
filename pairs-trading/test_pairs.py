import numpy as np
import pytest

from pairs import PairFit, backtest, fit_pair, half_life, positions, simulate_pair


def test_hedge_ratio_is_recovered():
    x, y = simulate_pair(2000, beta=1.5, seed=0)
    # OLS is superconsistent under cointegration (error ~ 1/n), but 2,000 points still leave ~0.02.
    assert fit_pair(x, y).beta == pytest.approx(1.5, abs=0.05)


def test_half_life_matches_the_ar_coefficient_on_a_long_sample():
    # Short samples bias the AR(1) coefficient low (Kendall), which shortens the half-life.
    phi = 0.95
    x, y = simulate_pair(20_000, beta=1.0, phi=phi, seed=1)
    assert fit_pair(x, y).half_life == pytest.approx(np.log(2) / -np.log(phi), rel=0.1)


def test_half_life_of_a_random_walk_is_infinite():
    assert half_life(np.cumsum(np.ones(100))) == float("inf")


def test_cointegration_test_has_power_and_controls_false_positives():
    true_pass = np.mean([fit_pair(*simulate_pair(500, seed=s)).pvalue < 0.05 for s in range(60)])
    false_pass = np.mean([fit_pair(*simulate_pair(500, cointegrated=False, seed=1000 + s)).pvalue < 0.05 for s in range(400)])
    assert true_pass > 0.7
    assert 0.01 < false_pass < 0.1


def test_true_pairs_make_money_out_of_sample():
    sharpes = []
    for s in range(20):
        x, y = simulate_pair(1000, seed=s)
        sharpes.append(backtest(x[500:], y[500:], fit_pair(x[:500], y[:500]))[1])
    assert np.mean(sharpes) > 1.0


def test_spurious_pairs_look_good_in_sample_and_fail_out_of_sample():
    in_sample, out_sample = [], []
    for s in range(800):
        x, y = simulate_pair(1000, cointegrated=False, seed=10_000 + s)
        fit = fit_pair(x[:500], y[:500])
        if fit.pvalue < 0.05:
            in_sample.append(backtest(x[:500], y[:500], fit)[1])
            out_sample.append(backtest(x[500:], y[500:], fit)[1])
    assert len(in_sample) > 20
    assert np.mean(in_sample) > 1.2
    assert abs(np.mean(out_sample)) < 0.4


def test_positions_never_look_ahead():
    x, y = simulate_pair(600, seed=3)
    fit = fit_pair(x[:300], y[:300])
    base = positions(x, y, fit)
    rng = np.random.default_rng(0)
    for t in (350, 450, 550):
        x2, y2 = x.copy(), y.copy()
        y2[t + 1 :] += rng.standard_normal(600 - t - 1) * 5
        assert np.array_equal(positions(x2, y2, fit)[: t + 1], base[: t + 1])


def test_entry_and_exit_rules():
    fit = PairFit(alpha=0.0, beta=0.0, pvalue=0.0, spread_mean=0.0, spread_std=1.0, half_life=1.0)
    z = np.array([0.0, 2.5, 1.0, 0.4, -2.5, -1.0, -0.2, 0.0])
    assert np.array_equal(positions(np.zeros_like(z), z, fit), [0, -1, -1, 0, 1, 1, 0, 0])
