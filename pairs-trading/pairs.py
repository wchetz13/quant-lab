"""Pairs trading with cointegration.

Two I(1) prices are cointegrated when some combination y - beta x is stationary. The spread then
mean-reverts and can be traded: short it when it is rich, buy it when it is cheap. The danger
is that finding such pairs means testing many candidates, and a 5% test passes 5% of unrelated
pairs by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.stattools import coint


def simulate_pair(n, beta=1.5, phi=0.95, spread_vol=0.5, cointegrated=True, seed=None):
    """Log-price-like series x (a random walk) and y.

    Cointegrated: y = beta x + s with s an AR(1) with coefficient phi (an OU process sampled
    daily), so y - beta x is stationary. Otherwise y is an independent random walk.
    """
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    if not cointegrated:
        return x, np.cumsum(rng.standard_normal(n))
    s = np.zeros(n)
    shocks = rng.standard_normal(n) * spread_vol
    for t in range(1, n):
        s[t] = phi * s[t - 1] + shocks[t]
    return x, beta * x + s


@dataclass(frozen=True)
class PairFit:
    alpha: float
    beta: float
    pvalue: float
    spread_mean: float
    spread_std: float
    half_life: float


def half_life(spread):
    """OU half-life from the AR(1) regression ds_t = a + b s_{t-1}: ln 2 / -ln(1 + b)."""
    s = np.asarray(spread, float)
    lagged = s[:-1] - s[:-1].mean()
    b = np.dot(lagged, np.diff(s)) / np.dot(lagged, lagged)
    return float(np.log(2) / -np.log1p(b)) if -1 < b < 0 else float("inf")


def fit_pair(x, y) -> PairFit:
    """Engle–Granger: OLS of y on x, then a unit-root test of the residual.

    statsmodels' coint uses MacKinnon's critical values, which account for beta being estimated.
    Ordinary ADF tables would reject far too often, because OLS picks the beta whose residual
    looks most stationary.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.column_stack([np.ones_like(x), x])
    (alpha, beta), *_ = np.linalg.lstsq(A, y, rcond=None)
    spread = y - alpha - beta * x
    pvalue = coint(y, x, trend="c", autolag="aic")[1]
    return PairFit(float(alpha), float(beta), float(pvalue), float(spread.mean()), float(spread.std(ddof=1)), half_life(spread))


def positions(x, y, fit: PairFit, entry=2.0, exit=0.5):
    """Spread position in {-1, 0, +1} at each time, from the frozen formation-window fit.

    Position at t uses only data up to t: it is decided after observing the spread at t and held
    over (t, t + 1].
    """
    z = (np.asarray(y) - fit.alpha - fit.beta * np.asarray(x) - fit.spread_mean) / fit.spread_std
    pos = np.zeros(len(z))
    current = 0.0
    for t, zt in enumerate(z):
        if current == 0:
            if zt > entry:
                current = -1.0
            elif zt < -entry:
                current = 1.0
        elif (current < 0 and zt < exit) or (current > 0 and zt > -exit):
            current = 0.0
        pos[t] = current
    return pos


def backtest(x, y, fit: PairFit, entry=2.0, exit=0.5):
    """Daily P&L of one unit of spread (long y, short beta x), in spread units, and its Sharpe ratio."""
    spread = np.asarray(y) - fit.beta * np.asarray(x)
    pos = positions(x, y, fit, entry, exit)
    pnl = pos[:-1] * np.diff(spread)
    sd = pnl.std(ddof=1)
    sharpe = float(np.sqrt(252) * pnl.mean() / sd) if sd > 0 else 0.0
    return pnl, sharpe


def formation_trading(x, y, split):
    """Fit on [0, split), trade on [split, end)."""
    fit = fit_pair(x[:split], y[:split])
    return fit, backtest(x[split:], y[split:], fit)
