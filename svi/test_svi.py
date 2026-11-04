import numpy as np
import pytest
from scipy.integrate import trapezoid

from svi import (
    SVI,
    butterfly_arbitrage_free,
    calendar_arbitrage_free,
    durrleman_g,
    fit,
    risk_neutral_density,
    satisfies_lee_bound,
)

EQUITY = SVI(a=0.04, b=0.4, rho=-0.4, m=0.05, sigma=0.25)
# Axel Vogt's slice (Gatheral & Jacquier, 2014): a smooth, sensible-looking smile with butterfly arbitrage.
VOGT = SVI(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
K = np.linspace(-0.6, 0.5, 23)


def test_derivatives_match_finite_differences():
    k, h = np.linspace(-1, 1, 11), 1e-5
    assert np.allclose(EQUITY.dw(k), (EQUITY.w(k + h) - EQUITY.w(k - h)) / (2 * h), atol=1e-8)
    assert np.allclose(EQUITY.d2w(k), (EQUITY.w(k + h) - 2 * EQUITY.w(k) + EQUITY.w(k - h)) / h**2, atol=1e-4)


def test_minimum_variance_is_closed_form():
    k = np.linspace(-2, 2, 200001)
    assert EQUITY.w(k).min() == pytest.approx(EQUITY.min_variance, abs=1e-10)


def test_fit_recovers_parameters_from_a_clean_slice():
    fitted = fit(K, EQUITY.w(K))
    for name in ("a", "b", "rho", "m", "sigma"):
        assert getattr(fitted, name) == pytest.approx(getattr(EQUITY, name), abs=1e-8)


def test_fit_to_noisy_vols_is_accurate_to_the_noise_level():
    T, noise = 0.5, 0.002
    rng = np.random.default_rng(0)
    quotes = EQUITY.implied_vol(K, T) + rng.normal(0, noise, K.size)
    fitted = fit(K, quotes**2 * T)
    assert np.sqrt(np.mean((fitted.implied_vol(K, T) - quotes) ** 2)) < noise
    assert np.abs(fitted.implied_vol(K, T) - EQUITY.implied_vol(K, T)).max() < 2 * noise


def test_equity_slice_is_arbitrage_free():
    assert butterfly_arbitrage_free(EQUITY)
    assert satisfies_lee_bound(EQUITY)


def test_vogt_slice_has_butterfly_arbitrage():
    k = np.linspace(-1.5, 1.5, 3001)
    g = durrleman_g(VOGT, k)
    assert not butterfly_arbitrage_free(VOGT)
    assert 0.6 < k[g.argmin()] < 1.2


def test_density_is_a_martingale_probability_when_arbitrage_free():
    k = np.linspace(-80, 80, 400001)
    density = risk_neutral_density(EQUITY, k)
    assert np.all(density >= 0)
    assert trapezoid(density, k) == pytest.approx(1.0, abs=1e-6)
    # E[S_T / F] = 1: the forward is the risk-neutral mean.
    assert trapezoid(density * np.exp(k), k) == pytest.approx(1.0, abs=1e-6)


def test_arbitrage_shows_up_as_negative_probability():
    k = np.linspace(-12, 12, 200001)
    density = risk_neutral_density(VOGT, k)
    # Total mass is still one, so the arbitrage is invisible unless you look pointwise.
    assert trapezoid(density, k) == pytest.approx(1.0, abs=1e-6)
    assert density.min() < 0


def test_calendar_spread_arbitrage_is_detected():
    short = SVI(a=0.02, b=0.3, rho=-0.4, m=0.0, sigma=0.2)
    long = SVI(a=0.04, b=0.35, rho=-0.4, m=0.0, sigma=0.25)
    steep = SVI(a=0.03, b=0.6, rho=-0.4, m=0.0, sigma=0.1)
    assert calendar_arbitrage_free([(1.0, long), (0.5, short)])
    # steep's wings rise above the longer slice: total variance would fall with maturity there.
    assert not calendar_arbitrage_free([(0.5, steep), (1.0, long)])


@pytest.mark.parametrize(
    "params",
    [
        dict(a=0.04, b=-0.1, rho=0.0, m=0.0, sigma=0.2),
        dict(a=0.04, b=0.1, rho=1.0, m=0.0, sigma=0.2),
        dict(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.0),
        dict(a=-0.5, b=0.1, rho=0.0, m=0.0, sigma=0.2),
    ],
)
def test_rejects_invalid_parameters(params):
    with pytest.raises(ValueError):
        SVI(**params)
