"""American option pricing: Longstaff–Schwartz regression Monte Carlo against a binomial benchmark.

An American option can be exercised at any time, so its value is an optimal stopping problem:

    V_0 = sup_tau E[e^{-r tau} h(S_tau)]

The holder exercises when the payoff beats the continuation value, the expected discounted value
of waiting. Longstaff and Schwartz (2001) estimate that conditional expectation by least squares
regression of realised future cash flows on functions of the current price, stepping backward in
time across simulated paths.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import laguerre


def gbm_paths(S, T, r, q, sigma, n_steps, n_paths, rng):
    """Price paths at n_steps exercise dates, with antithetic pairs. Shape (n_paths, n_steps)."""
    dt = T / n_steps
    z = rng.standard_normal((n_paths // 2, n_steps))
    z = np.concatenate([z, -z])
    log_inc = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    return S * np.exp(np.cumsum(log_inc, axis=1))


def put_payoff(S, K):
    return np.maximum(K - S, 0.0)


def call_payoff(S, K):
    return np.maximum(S - K, 0.0)


def _basis(x, degree):
    # Weighted Laguerre polynomials, as in the original paper, on price scaled by the strike.
    return np.column_stack([np.exp(-x / 2) * laguerre.Laguerre.basis(k)(x) for k in range(degree + 1)])


def fit_exercise_policy(paths, K, r, dt, payoff, degree=3):
    """Backward induction. Returns the regression coefficients for each exercise date but the last."""
    n_paths, n_steps = paths.shape
    cash = payoff(paths[:, -1], K)
    coefs = [None] * (n_steps - 1)
    for t in range(n_steps - 2, -1, -1):
        cash = cash * np.exp(-r * dt)
        exercise = payoff(paths[:, t], K)
        itm = exercise > 0
        # Regress only on in-the-money paths: those are the only ones where the decision matters,
        # and fitting there concentrates the basis where the exercise boundary lives.
        if itm.sum() <= degree + 1:
            continue
        X = _basis(paths[itm, t] / K, degree)
        beta, *_ = np.linalg.lstsq(X, cash[itm], rcond=None)
        coefs[t] = beta
        continuation = X @ beta
        stop = exercise[itm] > continuation
        idx = np.flatnonzero(itm)[stop]
        cash[idx] = exercise[idx]
    return coefs


def apply_exercise_policy(paths, K, r, dt, payoff, coefs, degree=3):
    """Discounted cash flows from following a fixed policy on (ideally fresh) paths."""
    n_paths, n_steps = paths.shape
    value = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    for t in range(n_steps - 1):
        if coefs[t] is None:
            continue
        exercise = payoff(paths[:, t], K)
        candidates = alive & (exercise > 0)
        continuation = _basis(paths[candidates, t] / K, degree) @ coefs[t]
        idx = np.flatnonzero(candidates)[exercise[candidates] > continuation]
        value[idx] = exercise[idx] * np.exp(-r * dt * (t + 1))
        alive[idx] = False
    last = alive
    value[last] = payoff(paths[last, -1], K) * np.exp(-r * dt * n_steps)
    return value


def lsm_price(S, K, T, r, q, sigma, payoff=put_payoff, n_steps=50, n_paths=100_000, degree=3, seed=None):
    """Longstaff–Schwartz price with its standard error.

    The policy is fitted on one set of paths and priced on an independent set. Reusing the fitting
    paths lets the regression see the future it is judged on, which biases the price upward; on
    fresh paths any estimated policy is merely suboptimal, so the estimate is a true lower bound.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    coefs = fit_exercise_policy(gbm_paths(S, T, r, q, sigma, n_steps, n_paths, rng), K, r, dt, payoff, degree)
    value = apply_exercise_policy(gbm_paths(S, T, r, q, sigma, n_steps, n_paths, rng), K, r, dt, payoff, coefs, degree)
    # Exercising immediately is also allowed.
    price = max(value.mean(), float(payoff(np.array(S), K)))
    # Antithetic halves are correlated, so the standard error is computed over pair averages.
    pairs = 0.5 * (value[: n_paths // 2] + value[n_paths // 2 :])
    return price, pairs.std(ddof=1) / np.sqrt(pairs.size)


def binomial_price(S, K, T, r, q, sigma, payoff=put_payoff, n_steps=2000, exercise_dates=None):
    """Cox–Ross–Rubinstein tree, vectorised over each time slice.

    exercise_dates=None is American (exercise at every node), 0 is European, and k > 0 is
    Bermudan with k equally spaced dates, which is the contract Longstaff–Schwartz actually prices.
    """
    if exercise_dates:
        if n_steps % exercise_dates:
            raise ValueError("n_steps must be a multiple of exercise_dates")
        every = n_steps // exercise_dates
    else:
        every = 1
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(n_steps + 1)
    values = payoff(S * u ** (n_steps - 2 * j), K)
    for n in range(n_steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if exercise_dates != 0 and n % every == 0 and n > 0:
            values = np.maximum(values, payoff(S * u ** (n - 2 * j[: n + 1]), K))
    return float(values[0])
