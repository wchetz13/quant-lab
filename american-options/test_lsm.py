import numpy as np
import pytest
from scipy.stats import norm

from lsm import binomial_price, call_payoff, lsm_price, put_payoff

K, R = 40.0, 0.06

# Longstaff & Schwartz (2001), Table 1: finite-difference values of a put exercisable 50 times a year.
PAPER_CASES = [
    (36.0, 0.2, 1, 4.478),
    (36.0, 0.4, 1, 7.101),
    (40.0, 0.2, 2, 2.885),
    (44.0, 0.2, 1, 1.110),
]


def bs_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return K * np.exp(-r * T) * norm.cdf(-d1 + sigma * np.sqrt(T)) - S * norm.cdf(-d1)


@pytest.mark.parametrize("S, sigma, T, reference", PAPER_CASES)
def test_bermudan_tree_reproduces_published_values(S, sigma, T, reference):
    price = binomial_price(S, K, T, R, 0.0, sigma, n_steps=2000 * T, exercise_dates=50 * T)
    assert price == pytest.approx(reference, abs=2e-3)


@pytest.mark.parametrize("S, sigma, T, reference", PAPER_CASES)
def test_lsm_is_a_tight_lower_bound(S, sigma, T, reference):
    price, stderr = lsm_price(S, K, T, R, 0.0, sigma, n_steps=50 * T, n_paths=100_000, seed=11)
    # Out-of-sample LSM follows a suboptimal policy, so it cannot beat the true Bermudan value
    # beyond noise, and with a cubic Laguerre basis it should land within a few cents of it.
    assert price < reference + 3 * stderr
    assert price > reference - 0.04


def test_tree_european_converges_to_black_scholes():
    tree = binomial_price(36.0, K, 1.0, R, 0.0, 0.2, n_steps=4000, exercise_dates=0)
    assert tree == pytest.approx(bs_put(36.0, K, 1.0, R, 0.2), abs=1e-3)


def test_american_put_carries_an_early_exercise_premium():
    american = binomial_price(36.0, K, 1.0, R, 0.0, 0.2)
    european = binomial_price(36.0, K, 1.0, R, 0.0, 0.2, exercise_dates=0)
    assert american > european + 0.5
    assert american >= K - 36.0


def test_never_exercise_a_call_early_without_dividends():
    # Merton (1973): with q = 0 the call is worth more alive than exercised at every node.
    args = (44.0, K, 1.0, R, 0.0, 0.3, call_payoff)
    assert binomial_price(*args) == pytest.approx(binomial_price(*args, exercise_dates=0), abs=1e-12)


def test_dividends_make_early_exercise_of_a_call_valuable():
    args = (44.0, K, 1.0, R, 0.08, 0.3, call_payoff)
    assert binomial_price(*args) > binomial_price(*args, exercise_dates=0) + 0.05


def test_deep_in_the_money_put_is_exercised_immediately():
    price, _ = lsm_price(5.0, K, 1.0, R, 0.0, 0.2, n_steps=50, n_paths=10_000, seed=0)
    assert price == pytest.approx(K - 5.0)


def test_rejects_exercise_dates_that_do_not_divide_the_tree():
    with pytest.raises(ValueError):
        binomial_price(36.0, K, 1.0, R, 0.0, 0.2, n_steps=1000, exercise_dates=30)


def test_put_payoff_is_intrinsic_value():
    assert np.array_equal(put_payoff(np.array([30.0, 40.0, 50.0]), K), [10.0, 0.0, 0.0])
