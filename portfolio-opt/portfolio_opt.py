"""Portfolio construction: mean-variance, risk parity and Black-Litterman.

Conventions: ``mu`` is a vector of expected (annual) returns, ``cov`` the covariance matrix of the same
horizon, weights sum to one. Functions without a ``long_only`` flag are closed form and allow shorting.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _check(mu, cov) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = mu.size
    if mu.ndim != 1 or cov.shape != (n, n):
        raise ValueError("mu must be 1-D and cov must be (n, n)")
    if not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError("cov must be symmetric")
    return mu, cov


def portfolio_stats(w, mu, cov) -> tuple[float, float]:
    """Return (expected return, volatility) of weights ``w``."""
    mu, cov = _check(mu, cov)
    w = np.asarray(w, dtype=float)
    return float(w @ mu), float(np.sqrt(w @ cov @ w))


def min_variance(cov, long_only: bool = False) -> np.ndarray:
    """Global minimum-variance weights: w = Sigma^-1 1 / (1' Sigma^-1 1), or SLSQP if long-only."""
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if not long_only:
        x = np.linalg.solve(cov, np.ones(n))
        return x / x.sum()
    return _long_only(lambda w: w @ cov @ w, lambda w: 2 * cov @ w, n)


def max_sharpe(mu, cov, rf: float = 0.0, long_only: bool = False) -> np.ndarray:
    """Tangency portfolio: w proportional to Sigma^-1 (mu - rf), or SLSQP if long-only."""
    mu, cov = _check(mu, cov)
    ex = mu - rf
    if not long_only:
        x = np.linalg.solve(cov, ex)
        if x.sum() <= 0:
            raise ValueError("tangency portfolio undefined: no asset has a positive excess return on net")
        return x / x.sum()

    # Long-only: the Charnes-Cooper transform turns max Sharpe into the convex QP
    #   min y'Sigma y  s.t.  y'(mu - rf) = 1, y >= 0,   with w = y / sum(y).
    if ex.max() <= 0:
        raise ValueError("no asset has a positive excess return")
    y0 = np.zeros(mu.size)
    y0[np.argmax(ex)] = 1.0 / ex.max()
    res = minimize(
        lambda y: y @ cov @ y,
        y0,
        jac=lambda y: 2 * cov @ y,
        bounds=[(0.0, None)] * mu.size,
        constraints=[{"type": "eq", "fun": lambda y: y @ ex - 1.0, "jac": lambda y: ex}],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"long-only optimisation failed: {res.message}")
    y = np.clip(res.x, 0.0, None)
    return y / y.sum()


def _long_only(fun, jac, n: int, extra=()) -> np.ndarray:
    res = minimize(
        fun,
        np.full(n, 1.0 / n),
        jac=jac,
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *extra],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"long-only optimisation failed: {res.message}")
    return np.clip(res.x, 0.0, None) / np.clip(res.x, 0.0, None).sum()


def efficient_frontier(mu, cov, targets, long_only: bool = False) -> np.ndarray:
    """Minimum-variance weights for each target return; returns an array of shape (len(targets), n).

    Unconstrained, the solution is the two-fund result w = Sigma^-1 (lam 1 + gam mu) with (lam, gam)
    solving the 2x2 system built from A = 1'S1, B = 1'S mu, C = mu'S mu (S = Sigma^-1).
    """
    mu, cov = _check(mu, cov)
    targets = np.atleast_1d(np.asarray(targets, dtype=float))
    n = mu.size
    if not long_only:
        s_one = np.linalg.solve(cov, np.ones(n))
        s_mu = np.linalg.solve(cov, mu)
        a, b, c = s_one.sum(), mu @ s_one, mu @ s_mu
        out = []
        for m in targets:
            lam, gam = np.linalg.solve([[a, b], [b, c]], [1.0, m])
            out.append(lam * s_one + gam * s_mu)
        return np.array(out)
    out = []
    for m in targets:
        eq = {"type": "eq", "fun": lambda w, m=m: w @ mu - m}
        out.append(_long_only(lambda w: w @ cov @ w, lambda w: 2 * cov @ w, n, extra=(eq,)))
    return np.array(out)


def risk_contributions(w, cov) -> np.ndarray:
    """Fractional contribution of each asset to portfolio variance: w_i (Sigma w)_i / w'Sigma w."""
    w = np.asarray(w, dtype=float)
    m = np.asarray(cov, dtype=float) @ w
    return w * m / (w @ m)


def risk_parity(cov, budget=None, tol: float = 1e-10, max_iter: int = 200) -> np.ndarray:
    """Long-only risk-budgeting weights (equal risk contribution by default).

    Uses the convex formulation of Spinu (2013): minimise  x'Sigma x / 2 - sum_i b_i ln x_i  over x > 0
    by damped Newton, then normalise x to sum to one. At the optimum Sigma x = b / x, i.e. each
    contribution x_i (Sigma x)_i equals the budget b_i.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    b = np.full(n, 1.0 / n) if budget is None else np.asarray(budget, dtype=float)
    if b.shape != (n,) or np.any(b <= 0) or not np.isclose(b.sum(), 1.0):
        raise ValueError("budget must be positive and sum to one")

    def f(x):
        return 0.5 * x @ cov @ x - b @ np.log(x)

    x = 1.0 / np.sqrt(np.diag(cov))
    for _ in range(max_iter):
        grad = cov @ x - b / x
        if np.linalg.norm(grad) < tol:
            break
        step = np.linalg.solve(cov + np.diag(b / x**2), grad)
        t = 1.0
        while np.any(x - t * step <= 0) or f(x - t * step) > f(x) - 1e-4 * t * (grad @ step) + 1e-14 * abs(f(x)):
            t *= 0.5
            if t < 1e-16:
                break
        x = x - t * step
    else:
        raise RuntimeError("risk_parity did not converge")
    return x / x.sum()


def implied_returns(cov, w_mkt, risk_aversion: float) -> np.ndarray:
    """Reverse optimisation: the excess returns that make ``w_mkt`` optimal, pi = delta Sigma w."""
    return risk_aversion * np.asarray(cov, dtype=float) @ np.asarray(w_mkt, dtype=float)


def black_litterman(cov, pi, P, Q, tau: float = 0.05, omega=None) -> tuple[np.ndarray, np.ndarray]:
    """Black-Litterman posterior mean and covariance of expected returns.

    Prior: mu ~ N(pi, tau Sigma). Views: P mu = Q + e, e ~ N(0, Omega). The posterior mean is
    pi + tau Sigma P' (tau P Sigma P' + Omega)^-1 (Q - P pi). If ``omega`` is None, the He-Litterman
    choice Omega = diag(tau P Sigma P') is used, which makes view confidence proportional to prior
    confidence in the view portfolio. Returns (posterior mean, posterior covariance of mu); the
    covariance of returns is Sigma + posterior covariance of mu.
    """
    cov = np.asarray(cov, dtype=float)
    pi = np.asarray(pi, dtype=float)
    P = np.atleast_2d(np.asarray(P, dtype=float))
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    if P.shape != (Q.size, cov.shape[0]):
        raise ValueError("P must have shape (n_views, n_assets) matching Q")
    prior_var = tau * P @ cov @ P.T
    om = np.diag(np.diag(prior_var)) if omega is None else np.atleast_2d(np.asarray(omega, dtype=float))
    gain = tau * cov @ P.T @ np.linalg.inv(prior_var + om)
    post_mean = pi + gain @ (Q - P @ pi)
    post_cov = tau * cov - gain @ P @ (tau * cov)
    return post_mean, post_cov


def ledoit_wolf_identity(returns) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance towards mu_bar * I.

    ``returns`` is a (T, n) array. Returns (shrunk covariance, shrinkage intensity in [0, 1]).
    """
    x = np.asarray(returns, dtype=float)
    t, n = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / t
    mu_bar = np.trace(s) / n
    d2 = np.sum((s - mu_bar * np.eye(n)) ** 2) / n
    b2 = sum(np.sum((np.outer(r, r) - s) ** 2) for r in x) / (n * t**2)
    b2 = min(b2, d2)
    delta = 0.0 if d2 == 0 else b2 / d2
    return delta * mu_bar * np.eye(n) + (1 - delta) * s, float(delta)
