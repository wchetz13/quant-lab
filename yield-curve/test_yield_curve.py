import numpy as np
import pytest

from yield_curve import (
    Deposit,
    DiscountCurve,
    Swap,
    bootstrap,
    fit_nelson_siegel,
    fit_nss,
    nss_zero_rate,
    par_swap_rate,
)

TRUE = (0.045, -0.02, -0.015, 0.02, 1.8, 9.0)


def true_curve(maturities):
    t = np.asarray(maturities, float)
    return DiscountCurve(t, np.exp(-nss_zero_rate(t, *TRUE) * t))


def market(swap_years=(1, 2, 3, 5, 7, 10, 15, 20, 30)):
    ref = true_curve(np.arange(1, 31, dtype=float))
    deposits = [Deposit(m, float((1 / true_curve([m]).df(m) - 1) / m)) for m in (0.25, 0.5)]
    swaps = [Swap(n, par_swap_rate(ref, n)) for n in swap_years]
    return deposits, swaps


def test_flat_curve_interpolation_and_rates():
    c = DiscountCurve([1, 5, 10], np.exp(-0.03 * np.array([1, 5, 10])))
    t = np.array([0.3, 2.0, 7.5, 10.0])
    assert np.allclose(c.zero_rate(t), 0.03)
    assert np.allclose(c.forward_rate(1.0, 9.0), 0.03)
    assert np.allclose(c.df(0.0), 1.0)


def test_extrapolation_holds_last_forward_rate():
    c = DiscountCurve([1, 2], [np.exp(-0.02), np.exp(-0.05)])
    assert c.forward_rate(2.0, 5.0) == pytest.approx(0.03)
    assert c.instantaneous_forward([0.5, 1.5, 3.0]) == pytest.approx([0.02, 0.03, 0.03])


def test_curve_validation():
    with pytest.raises(ValueError):
        DiscountCurve([2, 1], [0.9, 0.95])
    with pytest.raises(ValueError):
        DiscountCurve([1], [-0.5])
    with pytest.raises(ValueError):
        Swap(2.5, 0.03)


def test_deposit_bootstrap_is_closed_form():
    c = bootstrap([Deposit(0.5, 0.04)])
    assert c.df(0.5) == pytest.approx(1 / 1.02)


def test_flat_swap_curve_gives_known_discount_factors():
    # annual par rate s on every year 1..5: D(1) = 1/(1+s), and D(n) from the usual recursion
    s = 0.04
    c = bootstrap([Swap(n, s) for n in range(1, 6)])
    d = [1 / (1 + s)]
    for n in range(2, 6):
        d.append((1 - s * sum(d)) / (1 + s))
    assert np.allclose(c.df(np.arange(1, 6)), d, rtol=1e-12)


def test_bootstrapped_curve_reprices_every_instrument():
    deposits, swaps = market()
    curve = bootstrap(deposits + swaps)
    for dep in deposits:
        assert (1 / curve.df(dep.maturity) - 1) / dep.maturity == pytest.approx(dep.rate, abs=1e-12)
    for sw in swaps:
        assert par_swap_rate(curve, int(sw.maturity)) == pytest.approx(sw.rate, abs=1e-12)


def test_bootstrap_with_all_annual_swaps_recovers_true_curve():
    ref = true_curve(np.arange(1, 31, dtype=float))
    swaps = [Swap(n, par_swap_rate(ref, n)) for n in range(1, 31)]
    curve = bootstrap(swaps)
    assert np.allclose(curve.df(np.arange(1, 31)), ref.df(np.arange(1, 31)), rtol=1e-10)


def test_sparse_swaps_still_reasonably_close_to_true_zero_curve():
    deposits, swaps = market()
    curve = bootstrap(deposits + swaps)
    t = np.linspace(1, 30, 30)
    assert np.abs(curve.zero_rate(t) - true_curve(t).zero_rate(t)).max() < 5e-4


def test_duplicate_maturities_rejected():
    with pytest.raises(ValueError):
        bootstrap([Swap(2, 0.03), Swap(2, 0.031)])


def test_nss_limits():
    t = np.array([1e-4, 1, 5, 50, 1e4])
    z = nss_zero_rate(t, *TRUE)
    assert z[0] == pytest.approx(TRUE[0] + TRUE[1], abs=1e-4)  # short end: beta0 + beta1
    assert z[-1] == pytest.approx(TRUE[0], abs=1e-3)  # long end: beta0


def test_nss_fit_recovers_noiseless_curve():
    t = np.linspace(0.25, 30, 60)
    y = nss_zero_rate(t, *TRUE)
    params, rmse = fit_nss(t, y)
    assert rmse < 1e-7
    assert np.allclose(nss_zero_rate(t, *params), y, atol=1e-6)


def test_nelson_siegel_recovers_its_own_curve_and_nss_is_never_worse():
    ns = (0.04, -0.015, 0.02, 0.0, 2.5, 1.0)
    t = np.linspace(0.25, 30, 60)
    p, rmse = fit_nelson_siegel(t, nss_zero_rate(t, *ns))
    assert rmse < 1e-9
    assert p[4] == pytest.approx(2.5, rel=1e-4)
    y = nss_zero_rate(t, *TRUE)
    assert fit_nss(t, y)[1] <= fit_nelson_siegel(t, y)[1] + 1e-12


def test_noisy_fit_error_is_at_noise_level():
    rng = np.random.default_rng(1)
    t = np.linspace(0.25, 30, 60)
    y = nss_zero_rate(t, *TRUE) + rng.normal(0, 1e-4, t.size)
    params, rmse = fit_nss(t, y)
    assert rmse < 1.2e-4
    assert np.abs(nss_zero_rate(t, *params) - nss_zero_rate(t, *TRUE)).max() < 3e-4
