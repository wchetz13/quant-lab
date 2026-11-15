import numpy as np
import pytest

from calibration import (
    LOWER,
    UPPER,
    Surface,
    calibrate,
    correlation_from_jacobian,
    kappa_profile,
    model_ivs,
    synthetic_surface,
)
from heston import HestonParams

TRUE = HestonParams(v0=0.04, kappa=1.5, theta=0.06, xi=0.6, rho=-0.7)
SMALL = dict(maturities=(0.5, 1.0, 2.0), moneyness=(0.85, 0.95, 1.0, 1.05, 1.15))


@pytest.fixture(scope="module")
def clean_fit():
    surf = synthetic_surface(TRUE, **SMALL)
    return surf, calibrate(surf, n_starts=2, seed=0)


def test_synthetic_surface_round_trips_through_model_ivs():
    surf = synthetic_surface(TRUE, **SMALL)
    x = np.array([TRUE.v0, TRUE.kappa, TRUE.theta, TRUE.xi, TRUE.rho])
    assert model_ivs(x, surf) == pytest.approx(surf.iv, abs=1e-9)


def test_recovers_parameters_from_a_noiseless_surface(clean_fit):
    _, res = clean_fit
    assert res.rmse_vol_pts < 1e-4
    for name in ("v0", "kappa", "theta", "xi", "rho"):
        assert getattr(res.params, name) == pytest.approx(getattr(TRUE, name), rel=1e-3, abs=1e-3)


def test_fit_stays_inside_bounds(clean_fit):
    _, res = clean_fit
    assert np.all(res.vector >= LOWER) and np.all(res.vector <= UPPER)


def test_noisy_fit_error_is_on_the_order_of_the_noise():
    noise = 0.002  # 0.2 vol points
    surf = synthetic_surface(TRUE, noise=noise, seed=3, **SMALL)
    res = calibrate(surf, n_starts=2, seed=0)
    assert res.rmse_vol_pts < 2 * noise * 100  # 15 quotes, 5 params: near the noise level
    assert res.params.rho < -0.5 and res.params.v0 == pytest.approx(TRUE.v0, abs=0.01)


def test_kappa_and_theta_estimates_are_strongly_anticorrelated(clean_fit):
    _, res = clean_fit
    corr = res.parameter_correlation()
    assert corr[1, 2] < -0.8
    assert np.allclose(np.diag(corr), 1.0)


def test_kappa_profile_has_its_minimum_at_the_truth_and_theta_trades_off():
    """Pinning kappa away from the truth costs fit quality and pushes theta the other way."""
    surf = synthetic_surface(TRUE, **SMALL)
    kappas = [0.75, 1.5, 3.0]
    prof = kappa_profile(surf, kappas, n_starts=2)
    rmse = [r.rmse_vol_pts for r in prof]
    assert rmse[1] < 1e-4 < min(rmse[0], rmse[2])
    thetas = [r.params.theta for r in prof]
    assert thetas[0] > thetas[1] > thetas[2]


def test_fixed_parameter_is_respected():
    surf = synthetic_surface(TRUE, **SMALL)
    res = calibrate(surf, n_starts=1, fixed={"rho": -0.7, "xi": 0.6})
    assert res.params.rho == -0.7 and res.params.xi == 0.6
    assert np.all(res.jacobian[:, [3, 4]] == 0)


def test_correlation_from_jacobian_is_symmetric_with_unit_diagonal():
    J = np.random.default_rng(0).standard_normal((30, 5))
    c = correlation_from_jacobian(J)
    assert np.allclose(c, c.T) and np.allclose(np.diag(c), 1.0)
    assert np.all(np.abs(c) <= 1 + 1e-12)


def test_rejects_unknown_fixed_parameter_and_bad_surface():
    surf = synthetic_surface(TRUE, **SMALL)
    with pytest.raises(ValueError):
        calibrate(surf, fixed={"lambda": 1.0})
    with pytest.raises(ValueError):
        Surface(100, 0.03, 0.01, np.array([1.0]), np.array([100.0, 90.0]), np.array([0.2]))
    with pytest.raises(ValueError):
        Surface(100, 0.03, 0.01, np.array([1.0]), np.array([100.0]), np.array([-0.2]))
