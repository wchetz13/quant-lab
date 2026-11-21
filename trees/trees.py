"""Lattice option pricing: binomial and trinomial trees, and how fast they converge.

A recombining tree discretises the risk-neutral dynamics of the underlying and prices an option by
backward induction:

    V_i^n = e^{-r dt} E_Q[V^{n+1} | node i],    V_i^n <- max(V_i^n, h(S_i^n))   (American)

The terminal payoff is a kink in the price, so the tree price oscillates around the Black–Scholes
value as the strike moves between nodes. This module implements several ways to tame that:
Cox–Ross–Rubinstein (CRR), Jarrow–Rudd (JR), Leisen–Reimer (LR, strike-centred), a Boyle
trinomial tree, and Broadie–Detemple's Black–Scholes smoothing with Richardson extrapolation
(BBSR).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

BINOMIAL_METHODS = ("crr", "jr", "lr")


def black_scholes(S: float, K: float, T: float, r: float, q: float, sigma: float, kind: str = "put") -> float:
    """Closed-form European price with continuous dividend yield ``q``."""
    _check_kind(kind)
    if T <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sq
    d2 = d1 - sq
    if kind == "call":
        return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))


def _check_kind(kind: str) -> None:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def _payoff(S: np.ndarray, K: float, kind: str) -> np.ndarray:
    return np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)


def _peizer_pratt(z: float, n: int) -> float:
    """Peizer–Pratt method 2 inversion: a binomial probability matching the normal cdf at z."""
    inner = 0.25 - 0.25 * np.exp(-((z / (n + 1 / 3 + 0.1 / (n + 1))) ** 2) * (n + 1 / 6))
    return 0.5 + np.sign(z) * np.sqrt(inner)


@dataclass(frozen=True)
class BinomialParams:
    """Up/down factors and up-probability of a one-step binomial move."""

    u: float
    d: float
    p: float


def binomial_params(method: str, S: float, K: float, T: float, r: float, q: float, sigma: float, n: int) -> BinomialParams:
    """Parameters for ``method`` in ``{"crr", "jr", "lr"}``; all match the risk-neutral mean exactly."""
    dt = T / n
    growth = np.exp((r - q) * dt)
    if method == "crr":
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (growth - d) / (u - d)
    elif method == "jr":
        drift = (r - q - 0.5 * sigma**2) * dt
        u = np.exp(drift + sigma * np.sqrt(dt))
        d = np.exp(drift - sigma * np.sqrt(dt))
        p = (growth - d) / (u - d)
    elif method == "lr":
        if n % 2 == 0:
            raise ValueError("Leisen-Reimer needs an odd number of steps")
        sq = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sq
        d2 = d1 - sq
        p = _peizer_pratt(d2, n)
        p_star = _peizer_pratt(d1, n)
        u = growth * p_star / p
        d = (growth - p * u) / (1.0 - p)
    else:
        raise ValueError(f"unknown method {method!r}; choose from {BINOMIAL_METHODS}")
    if not 0.0 < p < 1.0:
        raise ValueError(f"risk-neutral probability {p:.4f} outside (0, 1): increase n or check inputs")
    return BinomialParams(float(u), float(d), float(p))


def _backward(values: np.ndarray, S_term_log: np.ndarray, params: BinomialParams, S: float, K: float,
              r: float, dt: float, kind: str, american: bool, n_from: int, keep: tuple[int, ...] = ()):
    """Roll ``values`` (nodes at step ``n_from``) back to step 0. Returns (values, {step: values})."""
    u, d, p = params.u, params.d, params.p
    disc = np.exp(-r * dt)
    saved: dict[int, np.ndarray] = {}
    for step in range(n_from - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            j = np.arange(step + 1)
            S_nodes = S * u**j * d ** (step - j)
            values = np.maximum(values, _payoff(S_nodes, K, kind))
        if step in keep:
            saved[step] = values.copy()
    return values, saved


def binomial_price(S: float, K: float, T: float, r: float, q: float, sigma: float, n: int = 500,
                   kind: str = "put", american: bool = False, method: str = "crr") -> float:
    """Price a vanilla option on an ``n``-step recombining binomial tree."""
    _check_kind(kind)
    params = binomial_params(method, S, K, T, r, q, sigma, n)
    j = np.arange(n + 1)
    S_T = S * params.u**j * params.d ** (n - j)
    values, _ = _backward(_payoff(S_T, K, kind), S_T, params, S, K, r, T / n, kind, american, n)
    return float(values[0])


def binomial_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float, n: int = 500,
                    kind: str = "put", american: bool = False) -> dict[str, float]:
    """Price, delta, gamma and theta (per year) read off the first two layers of a CRR tree.

    CRR has u d = 1, so the middle node at step 2 sits at the spot price and gives theta
    without bumping the tree.
    """
    _check_kind(kind)
    if n < 3:
        raise ValueError("need n >= 3 steps")
    params = binomial_params("crr", S, K, T, r, q, sigma, n)
    u, d = params.u, params.d
    j = np.arange(n + 1)
    S_T = S * u**j * d ** (n - j)
    values, saved = _backward(_payoff(S_T, K, kind), S_T, params, S, K, r, T / n, kind, american, n, keep=(1, 2))
    v1, v2 = saved[1], saved[2]
    Su, Sd = S * u, S * d
    Suu, Sud, Sdd = S * u * u, S, S * d * d
    delta = (v1[1] - v1[0]) / (Su - Sd)
    gamma = ((v2[2] - v2[1]) / (Suu - Sud) - (v2[1] - v2[0]) / (Sud - Sdd)) / (0.5 * (Suu - Sdd))
    theta = (v2[1] - values[0]) / (2.0 * T / n)
    return {"price": float(values[0]), "delta": float(delta), "gamma": float(gamma), "theta": float(theta)}


def trinomial_price(S: float, K: float, T: float, r: float, q: float, sigma: float, n: int = 300,
                    kind: str = "put", american: bool = False) -> float:
    """Boyle trinomial tree in log-price space with spacing ``dx = sigma sqrt(3 dt)``.

    The up/middle/down probabilities match the first two moments of the log-price increment.
    """
    _check_kind(kind)
    dt = T / n
    dx = sigma * np.sqrt(3.0 * dt)
    nu = r - q - 0.5 * sigma**2
    a = (sigma**2 * dt + nu**2 * dt**2) / dx**2
    b = nu * dt / dx
    pu, pd = 0.5 * (a + b), 0.5 * (a - b)
    pm = 1.0 - pu - pd
    if min(pu, pm, pd) < 0:
        raise ValueError("negative trinomial probability: increase n")
    disc = np.exp(-r * dt)
    offsets = np.arange(-n, n + 1)
    values = _payoff(S * np.exp(offsets * dx), K, kind)
    for step in range(n - 1, -1, -1):
        values = disc * (pu * values[2:] + pm * values[1:-1] + pd * values[:-2])
        if american:
            values = np.maximum(values, _payoff(S * np.exp(np.arange(-step, step + 1) * dx), K, kind))
    return float(values[0])


def bbs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, n: int = 500,
              kind: str = "put", american: bool = False) -> float:
    """Broadie–Detemple Binomial Black–Scholes: the last step uses the closed form.

    The final layer of nodes is replaced by Black–Scholes values over one step (floored at
    intrinsic value for American options), which removes the kink that causes oscillation.
    """
    _check_kind(kind)
    if n < 2:
        raise ValueError("need n >= 2 steps")
    params = binomial_params("crr", S, K, T, r, q, sigma, n)
    dt = T / n
    j = np.arange(n)
    S_pen = S * params.u**j * params.d ** (n - 1 - j)
    values = np.array([black_scholes(s, K, dt, r, q, sigma, kind) for s in S_pen])
    if american:
        values = np.maximum(values, _payoff(S_pen, K, kind))
    values, _ = _backward(values, S_pen, params, S, K, r, dt, kind, american, n - 1)
    return float(values[0])


def bbsr_price(S: float, K: float, T: float, r: float, q: float, sigma: float, n: int = 500,
               kind: str = "put", american: bool = False) -> float:
    """BBS with Richardson extrapolation: ``2 V(n) - V(n/2)`` removes the leading O(1/n) error."""
    n = n if n % 2 == 0 else n + 1
    fine = bbs_price(S, K, T, r, q, sigma, n, kind, american)
    coarse = bbs_price(S, K, T, r, q, sigma, n // 2, kind, american)
    return 2.0 * fine - coarse
