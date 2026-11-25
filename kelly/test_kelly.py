"""Tests for kelly.py: closed forms vs numerics, Monte Carlo checks of the growth and drawdown laws."""

import numpy as np
import pytest

from kelly import (
    drawdown_probability, estimated_kelly_growth, fractional_growth_ratio, gaussian_growth,
    kelly_binary, kelly_discrete, kelly_gaussian, log_growth, max_drawdown, simulate_binary,
)


def test_binary_known_values():
    assert kelly_binary(0.6, 1.0) == pytest.approx(0.2)
    assert kelly_binary(0.5, 2.0) == pytest.approx(0.25)


def test_binary_no_edge_gives_zero():
    assert kelly_binary(0.5, 1.0) == 0.0
    assert kelly_binary(0.3, 1.0) == 0.0


def test_binary_input_validation():
    with pytest.raises(ValueError):
        kelly_binary(1.2)
    with pytest.raises(ValueError):
        kelly_binary(0.5, 0.0)


@pytest.mark.parametrize("p,b", [(0.55, 1.0), (0.4, 2.5), (0.7, 0.5)])
def test_discrete_solver_matches_closed_form(p, b):
    f = kelly_discrete(np.array([b, -1.0]), np.array([p, 1 - p]))
    # a pure win/lose bet with full loss of the stake: max fraction is 1, optimum interior
    assert f == pytest.approx(kelly_binary(p, b), abs=1e-6)


def test_kelly_is_the_maximiser_of_growth():
    x, q = np.array([0.8, 0.1, -0.5]), np.array([0.3, 0.4, 0.3])
    f_star = kelly_discrete(x, q)
    grid = np.linspace(0, 1.5, 3001)
    assert abs(grid[np.argmax(log_growth(grid, x, q))] - f_star) < 1e-3
    assert log_growth(f_star, x, q) >= log_growth(f_star * 0.9, x, q)
    assert log_growth(f_star, x, q) >= log_growth(f_star * 1.1, x, q)


def test_discrete_zero_without_edge_and_cap_on_big_losses():
    assert kelly_discrete(np.array([1.0, -1.0]), np.array([0.5, 0.5])) == 0.0
    assert kelly_discrete(np.array([5.0, -1.0]), np.array([0.5, 0.5])) > 0.0
    # losses beyond -100% cap the stake at 1 / 1.5 so wealth stays positive
    f = kelly_discrete(np.array([5.0, -1.0, -1.5]), np.array([0.5, 0.45, 0.05]))
    assert 0.0 < f < 1 / 1.5


def test_log_growth_ruin_is_minus_inf():
    assert log_growth(1.0, np.array([1.0, -1.0]), np.array([0.5, 0.5])) == -np.inf


def test_log_growth_rejects_bad_probs():
    with pytest.raises(ValueError):
        log_growth(0.1, np.array([1.0, -1.0]), np.array([0.7, 0.7]))


def test_gaussian_kelly_maximises_gaussian_growth():
    mu, sigma, rf = 0.08, 0.2, 0.02
    f_star = kelly_gaussian(mu, sigma, rf)
    assert f_star == pytest.approx(1.5)
    grid = np.linspace(0, 4, 4001)
    assert grid[np.argmax(gaussian_growth(grid, mu, sigma, rf))] == pytest.approx(f_star, abs=2e-3)
    assert gaussian_growth(f_star, mu, sigma, rf) == pytest.approx(rf + (mu - rf) ** 2 / (2 * sigma**2))


def test_small_bet_limit_matches_gaussian_formula():
    # many tiny independent edges approximate the diffusion: f* ~ mean / variance
    x = np.array([0.001 + 0.02, 0.001 - 0.02])
    q = np.array([0.5, 0.5])
    assert kelly_discrete(x, q) == pytest.approx(0.001 / 0.02**2, rel=0.02)


def test_fractional_growth_ratio():
    assert fractional_growth_ratio(0.5) == pytest.approx(0.75)
    assert fractional_growth_ratio(1.0) == pytest.approx(1.0)
    assert fractional_growth_ratio(2.0) == pytest.approx(0.0)  # double Kelly: zero growth


def test_drawdown_probability_formula():
    assert drawdown_probability(0.5, 1.0) == pytest.approx(0.5)
    assert drawdown_probability(0.5, 0.5) == pytest.approx(0.5**3)
    with pytest.raises(ValueError):
        drawdown_probability(1.5)


def test_simulation_growth_matches_expectation():
    p, b = 0.55, 1.0
    f = kelly_binary(p, b)
    paths = simulate_binary(p, b, f, n_bets=2000, n_paths=400, seed=1)
    emp = np.log(paths[:, -1]).mean() / 2000
    exact = log_growth(f, np.array([b, -1.0]), np.array([p, 1 - p]))
    assert emp == pytest.approx(exact, abs=4e-4)


def test_full_kelly_beats_neighbours_in_realised_growth():
    p, b = 0.55, 1.0
    f_star = kelly_binary(p, b)
    g = [np.log(simulate_binary(p, b, f_star * c, 3000, 300, seed=2)[:, -1]).mean() for c in (0.5, 1.0, 2.5)]
    assert g[1] > g[0] and g[1] > g[2]


def test_double_kelly_growth_is_negative_for_discrete_bets():
    p, b = 0.55, 1.0
    f_star = kelly_binary(p, b)
    exact_2x = log_growth(2 * f_star, np.array([b, -1.0]), np.array([p, 1 - p]))
    assert exact_2x < 0.0  # discrete bets: double Kelly is slightly worse than zero growth


def test_drawdown_law_matches_simulation():
    # small stakes approximate the diffusion limit, where P(hit alpha) = alpha^(2/c - 1)
    p, b = 0.51, 1.0
    f_full = kelly_binary(p, b)
    for c, alpha in [(1.0, 0.5), (0.5, 0.5)]:
        paths = simulate_binary(p, b, c * f_full, n_bets=40_000, n_paths=1500, seed=3)
        hit = (paths.min(axis=1) <= alpha).mean()
        assert hit == pytest.approx(drawdown_probability(alpha, c), abs=0.05)


def test_max_drawdown_basic():
    path = np.array([[1.0, 2.0, 1.0, 3.0, 1.5]])
    assert max_drawdown(path)[0] == pytest.approx(0.5)


def test_fractional_kelly_cuts_drawdown():
    p, b = 0.55, 1.0
    f = kelly_binary(p, b)
    dd_full = max_drawdown(simulate_binary(p, b, f, 1000, 500, seed=4)).mean()
    dd_half = max_drawdown(simulate_binary(p, b, f / 2, 1000, 500, seed=4)).mean()
    assert dd_half < dd_full


def test_estimation_error_penalises_full_kelly_more_than_half():
    g_full, neg_full = estimated_kelly_growth(0.53, 1.0, 100, 1.0, seed=5)
    g_half, neg_half = estimated_kelly_growth(0.53, 1.0, 100, 0.5, seed=5)
    assert g_half > g_full
    assert neg_half < neg_full


def test_estimation_error_vanishes_with_data():
    g_small, _ = estimated_kelly_growth(0.55, 1.0, 50, 1.0, seed=6)
    g_big, _ = estimated_kelly_growth(0.55, 1.0, 20_000, 1.0, seed=6)
    exact = log_growth(0.1, np.array([1.0, -1.0]), np.array([0.55, 0.45]))
    assert g_big == pytest.approx(exact, rel=0.05)
    assert g_small < g_big
