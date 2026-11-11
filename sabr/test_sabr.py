import numpy as np
import pytest
from scipy.optimize import brentq

from sabr import SABR, black_call, calibrate, hagan_vol, implied_density, simulate

EQUITY = SABR(alpha=0.25, beta=1.0, rho=-0.3, nu=0.4)
RATES = SABR(alpha=0.03, beta=0.5, rho=-0.3, nu=0.5)


def mc_vol(F, K, T, terminal):
    # Price the out-of-the-money side and convert by parity: far less noise than deep ITM calls.
    otm = np.maximum(K - terminal, 0) if K < F else np.maximum(terminal - K, 0)
    call = otm.mean() + max(F - K, 0)
    return brentq(lambda s: black_call(F, K, T, s) - call, 1e-6, 5)


def test_atm_is_continuous():
    vols = hagan_vol(100.0, [100 - 1e-7, 100.0, 100 + 1e-7], 1.0, EQUITY)
    # Only the smile's own slope over the 2e-7 strike step, no jump from the z / x(z) branch.
    assert np.ptp(vols) < 1e-9


def test_no_vol_of_vol_and_beta_one_is_black_scholes():
    p = SABR(alpha=0.2, beta=1.0, rho=0.0, nu=0.0)
    assert np.allclose(hagan_vol(100.0, [60.0, 100.0, 150.0], 2.0, p), 0.2, atol=1e-14)


def test_backbone_follows_beta():
    # Leading order, ATM vol ~ alpha / F^(1 - beta): halving F raises it by 2^(1 - beta).
    ratio = hagan_vol(0.015, 0.015, 0.01, RATES) / hagan_vol(0.03, 0.03, 0.01, RATES)
    assert ratio == pytest.approx(2 ** (1 - RATES.beta), rel=1e-3)


def test_negative_rho_gives_downward_skew():
    vols = hagan_vol(100.0, [80.0, 100.0, 120.0], 1.0, SABR(0.25, 1.0, -0.6, 0.3))
    assert vols[0] > vols[1] > vols[2]


def test_matches_monte_carlo_at_short_expiry():
    T = 0.25
    terminal = simulate(100.0, T, EQUITY, n_steps=100, n_paths=200_000, seed=3)
    for K in (80.0, 90.0, 100.0, 110.0, 120.0):
        assert abs(hagan_vol(100.0, K, T, EQUITY) - mc_vol(100.0, K, T, terminal)) < 0.0015


def test_expansion_error_grows_with_expiry():
    terminal = simulate(100.0, 5.0, EQUITY, n_steps=250, n_paths=200_000, seed=3)
    assert hagan_vol(100.0, 100.0, 5.0, EQUITY) - mc_vol(100.0, 100.0, 5.0, terminal) > 0.004


def test_density_turns_negative_at_low_strikes_for_long_expiries():
    strikes = np.linspace(0.001, 0.03, 300)
    assert implied_density(0.03, strikes, 1.0, RATES).min() > 0
    long = implied_density(0.03, strikes, 10.0, RATES)
    assert long.min() < 0
    assert strikes[long.argmin()] < 0.01


def test_calibration_recovers_parameters():
    strikes = np.linspace(70, 130, 13)
    fitted = calibrate(100.0, strikes, 1.0, hagan_vol(100.0, strikes, 1.0, EQUITY), beta=1.0)
    assert (fitted.alpha, fitted.rho, fitted.nu) == pytest.approx((EQUITY.alpha, EQUITY.rho, EQUITY.nu), abs=1e-8)


@pytest.mark.parametrize(
    "params",
    [
        dict(alpha=0.0, beta=0.5, rho=0.0, nu=0.3),
        dict(alpha=0.2, beta=1.5, rho=0.0, nu=0.3),
        dict(alpha=0.2, beta=0.5, rho=-1.0, nu=0.3),
        dict(alpha=0.2, beta=0.5, rho=0.0, nu=-0.1),
    ],
)
def test_rejects_invalid_parameters(params):
    with pytest.raises(ValueError):
        SABR(**params)
