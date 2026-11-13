import numpy as np
import pytest
from scipy.stats import norm

from dupire import LocalVolSurface, local_variance_unchecked, simulate
from svi import SVI

VOGT = SVI(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)


def equity_slice(T):
    return SVI(a=0.02 * T + 0.002 * T**2, b=0.15 * np.sqrt(T), rho=-0.6, m=0.02, sigma=0.15)


EQUITY = LocalVolSurface([(T, equity_slice(T)) for T in (0.25, 0.5, 1.0, 2.0)])


def black_call(k, w):
    """Undiscounted call on a unit forward at log-moneyness k with total variance w."""
    sd = np.sqrt(w)
    d1 = -k / sd + sd / 2
    return norm.cdf(d1) - np.exp(k) * norm.cdf(d1 - sd)


def mc_call(x, k):
    # Control variate on S_T / F_T, whose mean is exactly 1 under the forward measure.
    s = np.exp(x)
    pay = np.maximum(s - np.exp(k), 0)
    cov = np.cov(pay, s)
    adj = pay - cov[0, 1] / cov[1, 1] * (s - 1)
    return adj.mean(), adj.std() / np.sqrt(adj.size)


def test_flat_smile_gives_constant_local_vol():
    flat = LocalVolSurface([(0.5, SVI(0.02, 0.0, 0.0, 0.0, 0.1)), (1.0, SVI(0.04, 0.0, 0.0, 0.0, 0.1))])
    for T in (0.2, 0.7, 1.0):
        assert np.allclose(flat.local_vol(np.linspace(-1, 1, 9), T), 0.2)


def test_interpolation_reproduces_the_slices():
    k = np.linspace(-0.5, 0.5, 11)
    for T in (0.25, 0.5, 1.0, 2.0):
        assert np.allclose(EQUITY.total_variance(k, T), equity_slice(T).w(k))


def test_local_skew_is_twice_implied_skew_at_the_money():
    # Gatheral's rule of thumb, exact as T -> 0 when the smile is fixed in vol across expiries.
    v = SVI(a=0.04, b=0.1, rho=-0.5, m=0.0, sigma=0.3)
    surface = LocalVolSurface([(T, SVI(v.a * T, v.b * T, v.rho, v.m, v.sigma)) for T in (0.01, 0.02, 0.05)])
    h, T = 1e-4, 0.015
    implied = (surface.implied_vol(h, T) - surface.implied_vol(-h, T)) / (2 * h)
    local = (surface.local_vol(h, T) - surface.local_vol(-h, T)) / (2 * h)
    assert local / implied == pytest.approx(2.0, abs=0.01)


def test_monte_carlo_under_local_vol_reproduces_the_surface():
    T = 1.0
    x = simulate(EQUITY, T, n_steps=400, n_paths=200_000, seed=1)
    for k in (-0.3, -0.1, 0.0, 0.1, 0.2):
        price, stderr = mc_call(x, k)
        assert abs(price - black_call(k, EQUITY.total_variance(k, T))) < 4 * stderr


def test_euler_bias_shrinks_with_the_time_step():
    target = black_call(0.0, EQUITY.total_variance(0.0, 1.0))
    coarse = mc_call(simulate(EQUITY, 1.0, n_steps=25, n_paths=100_000, seed=5), 0.0)[0] - target
    fine = mc_call(simulate(EQUITY, 1.0, n_steps=200, n_paths=100_000, seed=5), 0.0)[0] - target
    assert coarse > 15e-4
    assert abs(fine) < coarse / 3


def test_butterfly_arbitrage_means_negative_local_variance():
    k = np.linspace(-1.5, 1.5, 3001)
    negative = k[local_variance_unchecked(VOGT, k, 1.0) < 0]
    assert negative.size and 0.6 < negative.min() and negative.max() < 1.3


def test_rejects_surfaces_with_arbitrage():
    with pytest.raises(ValueError, match="butterfly"):
        LocalVolSurface([(1.0, VOGT), (2.0, equity_slice(2.0))])
    with pytest.raises(ValueError, match="calendar"):
        LocalVolSurface([(1.0, equity_slice(2.0)), (2.0, equity_slice(1.0))])


def test_rejects_bad_expiries():
    with pytest.raises(ValueError, match="expiries"):
        LocalVolSurface([(0.0, equity_slice(1.0)), (1.0, equity_slice(1.0))])
