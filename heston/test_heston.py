import numpy as np
import pytest

from heston import (
    HestonParams,
    bs_call,
    call_price,
    characteristic_function,
    implied_vol,
    put_price,
    simulate_qe,
)

S, T, R, Q = 100.0, 1.0, 0.03, 0.01
SKEWED = HestonParams(v0=0.04, kappa=1.5, theta=0.06, xi=0.6, rho=-0.7)


def test_characteristic_function_is_normalised_and_martingale():
    assert characteristic_function(0.0, S, T, R, Q, SKEWED) == pytest.approx(1.0)
    # E[S_T] = phi(-i) must equal the forward, or the model admits arbitrage.
    assert characteristic_function(-1j, S, T, R, Q, SKEWED).real == pytest.approx(S * np.exp((R - Q) * T))


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_constant_variance_limit_is_black_scholes(K):
    # rho = 0: with correlation the price moves at first order in xi (the skew), not second.
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=1e-4, rho=0.0)
    assert call_price(S, K, T, R, Q, p) == pytest.approx(bs_call(S, K, T, R, Q, 0.2), abs=1e-6)


def test_deterministic_variance_uses_integrated_variance():
    v0, kappa, theta = 0.09, 1.5, 0.04
    p = HestonParams(v0=v0, kappa=kappa, theta=theta, xi=1e-4, rho=0.0)
    total = theta * T + (v0 - theta) * (1 - np.exp(-kappa * T)) / kappa
    assert call_price(S, 110.0, T, R, Q, p) == pytest.approx(
        bs_call(S, 110.0, T, R, Q, np.sqrt(total / T)), abs=1e-6
    )


def test_prices_are_free_of_static_arbitrage_across_strikes():
    strikes = np.linspace(60, 160, 21)
    calls = np.array([call_price(S, K, T, R, Q, SKEWED) for K in strikes])
    assert np.all(np.diff(calls) < 0)
    assert np.all(np.diff(calls, 2) > 0)
    lower = np.maximum(S * np.exp(-Q * T) - strikes * np.exp(-R * T), 0)
    assert np.all((calls > lower) & (calls < S * np.exp(-Q * T)))


def test_negative_correlation_produces_a_downward_skew():
    vols = [implied_vol(call_price(S, K, T, R, Q, SKEWED), S, K, T, R, Q) for K in (80, 100, 120)]
    assert vols[0] > vols[1] > vols[2]


def test_long_maturities_do_not_cross_the_branch_cut():
    # Heston's original formula jumps here; the call price must instead rise smoothly in T.
    maturities = np.arange(1.0, 31.0)
    prices = np.array([call_price(S, 100.0, t, 0.02, 0.0, SKEWED) for t in maturities])
    steps = np.diff(prices)
    assert np.all(steps > 0)
    assert np.all(np.abs(np.diff(steps)) < 2.0)


def test_put_call_parity():
    c, p = call_price(S, 95.0, T, R, Q, SKEWED), put_price(S, 95.0, T, R, Q, SKEWED)
    assert c - p == pytest.approx(S * np.exp(-Q * T) - 95.0 * np.exp(-R * T))


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_qe_monte_carlo_agrees_with_fourier_price(K):
    # Feller is violated here (2 kappa theta < xi^2), the regime where Euler schemes break down.
    assert not SKEWED.feller
    ST = simulate_qe(S, T, R, Q, SKEWED, n_steps=50, n_paths=200_000, seed=7)
    payoff = np.exp(-R * T) * np.maximum(ST - K, 0)
    stderr = payoff.std() / np.sqrt(payoff.size)
    assert abs(payoff.mean() - call_price(S, K, T, R, Q, SKEWED)) < 4 * stderr


def test_qe_variance_is_never_negative():
    p = HestonParams(v0=0.01, kappa=0.5, theta=0.02, xi=1.0, rho=-0.9)
    ST = simulate_qe(S, 2.0, R, Q, p, n_steps=40, n_paths=20_000, seed=3)
    assert np.all(np.isfinite(ST)) and np.all(ST > 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(v0=-0.01, kappa=1, theta=0.04, xi=0.5, rho=0),
        dict(v0=0.04, kappa=0, theta=0.04, xi=0.5, rho=0),
        dict(v0=0.04, kappa=1, theta=0.04, xi=0.5, rho=1),
    ],
)
def test_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        HestonParams(**kwargs)
