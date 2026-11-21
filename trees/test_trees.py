"""Tests for the lattice pricers: analytic limits, convergence, no-arbitrage and cross-checks."""

import numpy as np
import pytest

from trees import (bbs_price, bbsr_price, binomial_greeks, binomial_params, binomial_price,
                   black_scholes, trinomial_price)

ARGS = dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.02, sigma=0.2)


@pytest.mark.parametrize("method", ["crr", "jr", "lr"])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_binomial_converges_to_black_scholes(method, kind):
    n = 1001
    bs = black_scholes(kind=kind, **ARGS)
    assert binomial_price(n=n, kind=kind, method=method, **ARGS) == pytest.approx(bs, abs=0.02)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_trinomial_converges_to_black_scholes(kind):
    assert trinomial_price(n=500, kind=kind, **ARGS) == pytest.approx(black_scholes(kind=kind, **ARGS), abs=0.01)


def test_leisen_reimer_beats_crr_at_equal_steps():
    bs = black_scholes(kind="put", **ARGS)
    err_lr = abs(binomial_price(n=51, method="lr", **ARGS) - bs)
    err_crr = abs(binomial_price(n=51, method="crr", **ARGS) - bs)
    assert err_lr < err_crr / 5


def test_bbsr_is_more_accurate_than_crr():
    bs = black_scholes(kind="put", **ARGS)
    assert abs(bbsr_price(n=100, **ARGS) - bs) < abs(binomial_price(n=100, **ARGS) - bs) / 10


def test_leisen_reimer_requires_odd_steps():
    with pytest.raises(ValueError):
        binomial_price(n=100, method="lr", **ARGS)


@pytest.mark.parametrize("method", ["crr", "jr", "lr"])
def test_risk_neutral_mean_is_matched(method):
    p = binomial_params(method, n=101, **ARGS)
    dt = ARGS["T"] / 101
    assert p.p * p.u + (1 - p.p) * p.d == pytest.approx(np.exp((ARGS["r"] - ARGS["q"]) * dt), rel=1e-12)


def test_put_call_parity_on_european_tree():
    call = binomial_price(n=400, kind="call", **ARGS)
    put = binomial_price(n=400, kind="put", **ARGS)
    parity = ARGS["S"] * np.exp(-ARGS["q"] * ARGS["T"]) - ARGS["K"] * np.exp(-ARGS["r"] * ARGS["T"])
    assert call - put == pytest.approx(parity, abs=1e-9)


def test_american_put_dominates_european_and_intrinsic():
    eu = binomial_price(n=400, kind="put", **ARGS)
    am = binomial_price(n=400, kind="put", american=True, **ARGS)
    assert am > eu
    assert am >= max(ARGS["K"] - ARGS["S"], 0.0)


def test_american_call_without_dividends_equals_european():
    a = dict(ARGS, q=0.0)
    eu = binomial_price(n=300, kind="call", **a)
    am = binomial_price(n=300, kind="call", american=True, **a)
    assert am == pytest.approx(eu, abs=1e-10)


def test_american_put_matches_known_reference():
    # Widely quoted benchmark: S=K=100, r=5%, sigma=20%, T=1 American put is about 6.09 (no dividends).
    price = bbsr_price(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.2, n=1000, kind="put", american=True)
    assert price == pytest.approx(6.0896, abs=5e-3)


def test_american_put_binomial_and_trinomial_agree():
    a = dict(ARGS, S=90.0)
    b = binomial_price(n=1000, kind="put", american=True, **a)
    t = trinomial_price(n=600, kind="put", american=True, **a)
    assert b == pytest.approx(t, abs=0.01)


def test_deep_itm_american_put_is_intrinsic():
    price = binomial_price(S=40, K=100, T=1.0, r=0.05, q=0.0, sigma=0.2, n=200, kind="put", american=True)
    assert price == pytest.approx(60.0, abs=1e-9)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_tree_greeks_match_black_scholes(kind):
    g = binomial_greeks(n=800, kind=kind, **ARGS)
    h = 1e-2
    f = lambda s: black_scholes(S=s, K=100, T=1.0, r=0.05, q=0.02, sigma=0.2, kind=kind)
    delta = (f(100 + h) - f(100 - h)) / (2 * h)
    gamma = (f(100 + h) - 2 * f(100) + f(100 - h)) / h**2
    ft = lambda t: black_scholes(S=100, K=100, T=t, r=0.05, q=0.02, sigma=0.2, kind=kind)
    theta = -(ft(1.0 + h) - ft(1.0 - h)) / (2 * h)
    assert g["delta"] == pytest.approx(delta, abs=2e-3)
    assert g["gamma"] == pytest.approx(gamma, abs=2e-4)
    assert g["theta"] == pytest.approx(theta, abs=0.02)


def test_price_monotone_in_volatility():
    prices = [binomial_price(n=200, kind="put", american=True, **dict(ARGS, sigma=s)) for s in (0.1, 0.2, 0.3)]
    assert prices == sorted(prices)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        binomial_price(n=100, kind="straddle", **ARGS)
    with pytest.raises(ValueError):
        binomial_params("nope", n=100, **ARGS)
    with pytest.raises(ValueError):
        # Huge drift relative to volatility pushes the CRR probability outside (0, 1).
        binomial_price(S=100, K=100, T=1.0, r=2.0, q=0.0, sigma=0.05, n=10)
