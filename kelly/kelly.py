"""Kelly criterion position sizing: closed forms, numerical solutions, fractional Kelly.

Kelly maximises the expected log growth rate of wealth. For a bet that returns ``X`` per unit
staked (``X > -1``) and a fraction ``f`` of wealth staked,

    g(f) = E[log(1 + f X)].

Results implemented here:

* binary bet: ``f* = p - (1 - p) / b`` (win ``b``, lose 1);
* finite outcome distributions: maximise ``g`` numerically (concave in ``f``);
* Gaussian returns (continuous-time limit): ``f* = mu / sigma^2`` with growth ``mu^2 / (2 sigma^2)``;
* fractional Kelly ``c f*``: growth ``g(c f*) = (1 - (1 - c)^2) g*`` in the Gaussian limit and the
  probability of ever drawing down to a fraction ``a`` of the starting wealth ``a^(2/c - 1)``.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize


def kelly_binary(p: float, b: float = 1.0) -> float:
    """Kelly fraction for a bet winning ``b`` per unit with probability ``p``, losing 1 otherwise.

    Returns 0 when the edge ``p (b + 1) - 1`` is not positive (never bet a negative edge).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    if b <= 0:
        raise ValueError("b must be positive")
    return max(0.0, p - (1.0 - p) / b)


def log_growth(f: float | np.ndarray, outcomes: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Expected log growth ``E[log(1 + f X)]`` for a discrete distribution of returns ``X``.

    Fractions that risk ruin (``1 + f x <= 0`` for some outcome of positive probability) give
    ``-inf``.
    """
    x = np.asarray(outcomes, dtype=float)
    q = np.asarray(probs, dtype=float)
    if x.shape != q.shape or x.ndim != 1:
        raise ValueError("outcomes and probs must be 1-D arrays of equal length")
    if np.any(q < 0) or not np.isclose(q.sum(), 1.0):
        raise ValueError("probs must be non-negative and sum to one")
    f_arr = np.atleast_1d(np.asarray(f, dtype=float))
    wealth = 1.0 + f_arr[:, None] * x[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(wealth > 0, np.log(np.where(wealth > 0, wealth, 1.0)), -np.inf)
    g = np.where(q[None, :] > 0, q[None, :] * logs, 0.0).sum(axis=1)
    return g if np.ndim(f) else g[0]


def kelly_discrete(outcomes: np.ndarray, probs: np.ndarray) -> float:
    """Growth-optimal fraction (long-only, no leverage cap) for a discrete return distribution.

    ``g`` is strictly concave on the interval ``f < 1 / max loss`` where wealth stays positive, so
    a bounded scalar search over that interval finds the global optimum. The result is 0 when the mean return is
    not positive.
    """
    x = np.asarray(outcomes, dtype=float)
    q = np.asarray(probs, dtype=float)
    if float(q @ x) <= 0:
        return 0.0
    losers = x[(x < 0) & (q > 0)]
    upper = 1.0 / -losers.min() if losers.size else 1e6
    upper *= 1 - 1e-9
    res = optimize.minimize_scalar(
        lambda f: -log_growth(f, x, q), bounds=(0.0, upper), method="bounded",
        options={"xatol": 1e-12},
    )
    return float(res.x)


def kelly_gaussian(mu: float, sigma: float, rf: float = 0.0) -> float:
    """Continuous-time Kelly leverage ``(mu - rf) / sigma^2`` for a risky asset (Merton fraction)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return (mu - rf) / sigma**2


def gaussian_growth(f: float | np.ndarray, mu: float, sigma: float, rf: float = 0.0) -> np.ndarray:
    """Continuous-time growth rate ``rf + f (mu - rf) - f^2 sigma^2 / 2`` at leverage ``f``."""
    f = np.asarray(f, dtype=float)
    return rf + f * (mu - rf) - 0.5 * f**2 * sigma**2


def fractional_growth_ratio(c: np.ndarray | float) -> np.ndarray:
    """Share of the optimal Gaussian growth rate earned by betting ``c`` times Kelly: ``1-(1-c)^2``."""
    c = np.asarray(c, dtype=float)
    return 1.0 - (1.0 - c) ** 2


def drawdown_probability(alpha: float, c: float = 1.0) -> float:
    """P(wealth ever falls to ``alpha`` times its start) betting ``c`` times Kelly (continuous time).

    Equals ``alpha^(2/c - 1)``; for full Kelly this is simply ``alpha``. Valid for ``c > 0`` in
    the diffusion limit with a constant fraction of wealth staked.
    """
    if not 0.0 < alpha < 1.0 or c <= 0:
        raise ValueError("need 0 < alpha < 1 and c > 0")
    return float(alpha ** (2.0 / c - 1.0))


def simulate_binary(
    p: float, b: float, f: float, n_bets: int, n_paths: int, seed: int = 0
) -> np.ndarray:
    """Wealth paths (``n_paths`` x ``n_bets + 1``, starting at 1) staking fraction ``f`` each bet."""
    rng = np.random.default_rng(seed)
    wins = rng.random((n_paths, n_bets)) < p
    growth = np.where(wins, 1.0 + f * b, 1.0 - f)
    return np.concatenate([np.ones((n_paths, 1)), np.cumprod(growth, axis=1)], axis=1)


def max_drawdown(paths: np.ndarray) -> np.ndarray:
    """Maximum peak-to-trough fall of each wealth path, as a fraction of the running peak."""
    peak = np.maximum.accumulate(paths, axis=1)
    return (1.0 - paths / peak).max(axis=1)


def estimated_kelly_growth(
    p_true: float, b: float, n_obs: int, c: float, n_trials: int = 20_000, seed: int = 0
) -> tuple[float, float]:
    """True log growth when sizing a binary bet with an estimated win probability.

    Each trial observes ``n_obs`` bets, estimates ``p_hat``, sizes at ``c * kelly_binary(p_hat, b)``
    and is scored by the exact growth ``g`` under ``p_true``. Returns ``(mean growth, share of
    trials with negative growth)``. Overbetting from noisy edges is what fractional Kelly insures
    against.
    """
    rng = np.random.default_rng(seed)
    p_hat = rng.binomial(n_obs, p_true, n_trials) / n_obs
    f = np.clip(c * np.maximum(0.0, p_hat - (1.0 - p_hat) / b), 0.0, 1.0 - 1e-9)
    g = p_true * np.log1p(f * b) + (1.0 - p_true) * np.log1p(-f)
    return float(g.mean()), float((g < 0).mean())
