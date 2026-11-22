import numpy as np
import pytest

from garch import (
    diebold_mariano, ewma_forecasts, fit_garch, garch_rolling_forecasts, mincer_zarnowitz,
    qlike_loss, mse_loss, rolling_variance_forecasts, simulate_gjr, variance_filter,
)


@pytest.fixture(scope="module")
def data():
    return simulate_gjr(6000, seed=1)


def test_simulation_unconditional_variance():
    r, _ = simulate_gjr(60000, seed=3, nu=None)
    v = 0.02 / (1 - 0.03 - 0.05 - 0.88)
    assert np.var(r) == pytest.approx(v, rel=0.15)


def test_simulation_rejects_nonstationary():
    with pytest.raises(ValueError):
        simulate_gjr(100, alpha=0.1, gamma=0.1, beta=0.9)


def test_filter_matches_true_variance(data):
    r, s2 = data
    f = variance_filter(r, 0.02, 0.03, 0.10, 0.88, s2[0])
    np.testing.assert_allclose(f[:-1], s2, rtol=1e-8)


def test_garch_recovers_parameters():
    r, _ = simulate_gjr(8000, omega=0.02, alpha=0.08, gamma=0.0, beta=0.90, seed=5, nu=None)
    fit = fit_garch(r)
    assert fit.alpha == pytest.approx(0.08, abs=0.03)
    assert fit.beta == pytest.approx(0.90, abs=0.03)
    assert fit.persistence == pytest.approx(0.98, abs=0.015)


def test_gjr_recovers_leverage(data):
    r, _ = data
    fit = fit_garch(r, leverage=True, dist="t")
    assert fit.gamma == pytest.approx(0.10, abs=0.06)
    assert fit.nu == pytest.approx(7.0, abs=2.5)
    assert fit.persistence < 1


def test_gjr_beats_garch_on_leverage_data(data):
    r, _ = data
    a, b = fit_garch(r), fit_garch(r, leverage=True)
    assert b.loglik > a.loglik
    assert b.bic < a.bic


def test_student_t_beats_normal_on_fat_tails(data):
    r, _ = data
    assert fit_garch(r, dist="t").loglik > fit_garch(r, dist="normal").loglik


def test_forecast_mean_reverts_to_long_run(data):
    r, _ = data
    fit = fit_garch(r, leverage=True)
    f = fit.forecast(2000)
    assert f[0] == pytest.approx(fit.next_variance())
    assert f[-1] == pytest.approx(fit.long_run_variance, rel=1e-6)
    assert np.all(np.diff(f) * (fit.long_run_variance - f[0]) >= 0)


def test_fit_validates_input():
    with pytest.raises(ValueError):
        fit_garch(np.zeros(10))
    with pytest.raises(ValueError):
        fit_garch(np.random.default_rng(0).normal(size=300), dist="laplace")


def test_ewma_is_causal():
    r = np.random.default_rng(0).normal(size=200)
    a = ewma_forecasts(r)
    r2 = r.copy()
    r2[100:] += 5
    b = ewma_forecasts(r2)
    np.testing.assert_allclose(a[:101], b[:101])


def test_rolling_variance_window():
    r = np.arange(1.0, 11.0)
    out = rolling_variance_forecasts(r, 3)
    assert np.isnan(out[2]) and out[3] == pytest.approx((1 + 4 + 9) / 3)


def test_rolling_forecasts_are_out_of_sample(data):
    r, _ = data
    r = r[:900].copy()
    a = garch_rolling_forecasts(r, start=600, refit_every=50)
    r2 = r.copy()
    r2[750:] *= 10  # changing the future must not change earlier forecasts
    b = garch_rolling_forecasts(r2, start=600, refit_every=50)
    np.testing.assert_allclose(a[600:751], b[600:751])
    assert np.isnan(a[:600]).all()


def test_qlike_minimised_at_truth():
    rng = np.random.default_rng(0)
    s2 = 2.0
    proxy = s2 * rng.standard_normal(200000) ** 2
    grid = np.linspace(0.5, 4, 36)
    best = grid[np.argmin([qlike_loss(np.full_like(proxy, g), proxy).mean() for g in grid])]
    assert best == pytest.approx(s2, abs=0.2)
    assert qlike_loss(np.array([1.0]), np.array([1.0]))[0] == pytest.approx(0.0)
    assert mse_loss(np.array([2.0]), np.array([5.0]))[0] == 9.0


def test_diebold_mariano_detects_difference_and_null():
    rng = np.random.default_rng(1)
    a = rng.exponential(1.0, 3000)
    stat, p = diebold_mariano(a, a + 0.3 + 0.1 * rng.standard_normal(3000))
    assert stat < 0 and p < 1e-6
    stat, p = diebold_mariano(a, rng.exponential(1.0, 3000))
    assert p > 0.01


def test_mincer_zarnowitz_perfect_and_biased():
    x = np.linspace(1, 5, 100)
    a, b, r2 = mincer_zarnowitz(x, 2 * x + 1)
    assert (a, b, r2) == pytest.approx((1.0, 2.0, 1.0))
