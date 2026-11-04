"""Gatheral's SVI parameterisation of the implied volatility smile, with no-arbitrage checks.

Raw SVI models total implied variance w = sigma_BS^2 T as a function of log-moneyness k = ln(K/F):

    w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))

Five parameters give a smile that is asymptotically linear in both wings, which Lee's moment
formula says it must be, and the derivatives are in closed form, so arbitrage can be checked
exactly rather than on a price grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, lsq_linear, minimize


@dataclass(frozen=True)
class SVI:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if self.b < 0 or not -1 < self.rho < 1 or self.sigma <= 0:
            raise ValueError("need b >= 0, |rho| < 1 and sigma > 0")
        if self.min_variance < 0:
            raise ValueError("parameters give negative total variance")

    @property
    def min_variance(self) -> float:
        """Minimum of w over k, attained in closed form."""
        return self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2)

    def w(self, k):
        x = np.asarray(k) - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x**2 + self.sigma**2))

    def dw(self, k):
        x = np.asarray(k) - self.m
        return self.b * (self.rho + x / np.sqrt(x**2 + self.sigma**2))

    def d2w(self, k):
        x = np.asarray(k) - self.m
        return self.b * self.sigma**2 / (x**2 + self.sigma**2) ** 1.5

    def implied_vol(self, k, T):
        return np.sqrt(self.w(k) / T)


def durrleman_g(s: SVI, k):
    """Durrleman's function. The slice is free of butterfly arbitrage iff g(k) >= 0 for all k."""
    w, dw, d2w = s.w(k), s.dw(k), s.d2w(k)
    return (1 - k * dw / (2 * w)) ** 2 - dw**2 / 4 * (1 / w + 0.25) + d2w / 2


def risk_neutral_density(s: SVI, k):
    """Density of ln(S_T / F) implied by the slice (Gatheral & Jacquier, 2014).

    It is the second strike derivative of the call price written in total-variance terms, so it
    is negative exactly where g is: butterfly arbitrage is a negative probability.
    """
    w = s.w(k)
    d_minus = -k / np.sqrt(w) - np.sqrt(w) / 2
    return durrleman_g(s, k) / np.sqrt(2 * np.pi * w) * np.exp(-(d_minus**2) / 2)


def butterfly_arbitrage_free(s: SVI, k_grid=None) -> bool:
    k = np.linspace(-3, 3, 4001) if k_grid is None else k_grid
    return bool(np.all(durrleman_g(s, k) >= -1e-12))


def satisfies_lee_bound(s: SVI) -> bool:
    """Lee (2004): wing slopes of w are at most 2, i.e. b (1 + |rho|) <= 2."""
    return s.b * (1 + abs(s.rho)) <= 2


def calendar_arbitrage_free(slices, k_grid=None) -> bool:
    """Total variance must be non-decreasing in maturity at every k. slices: [(T, SVI)]."""
    k = np.linspace(-3, 3, 601) if k_grid is None else k_grid
    ordered = [s for _, s in sorted(slices, key=lambda ts: ts[0])]
    return all(np.all(later.w(k) >= earlier.w(k) - 1e-12) for earlier, later in zip(ordered, ordered[1:]))


def _inner_fit(k, w, m, sigma):
    """For fixed (m, sigma) SVI is linear: w = a + d y + c sqrt(y^2 + 1), y = (k - m) / sigma.

    The box keeps the slice valid: c = b sigma >= 0, |d| = |rho| c <= c is enforced afterwards
    by clipping, and a >= -c keeps the minimum variance above zero for any rho.
    """
    y = (k - m) / sigma
    z = np.sqrt(y**2 + 1)
    A = np.column_stack([np.ones_like(y), y, z])
    res = lsq_linear(A, w, bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]))
    a, d, c = res.x
    d = float(np.clip(d, -0.999 * c, 0.999 * c))
    return a, d, c, float(np.sum((A @ np.array([a, d, c]) - w) ** 2))


def fit(k, w, weights=None) -> SVI:
    """Calibrate raw SVI to total variances w at log-moneyness k.

    Uses the quasi-explicit method of De Marco and Martini (Zeliade, 2009): the five-parameter
    non-convex problem collapses to a two-dimensional search over (m, sigma), with the other
    three parameters solved exactly by linear least squares at each step. A final joint
    least-squares polish on all five raw parameters removes the clipping error.
    """
    k, w = np.asarray(k, float), np.asarray(w, float)
    wt = np.ones_like(w) if weights is None else np.sqrt(np.asarray(weights, float))

    def outer(x):
        m, log_sigma = x
        return _inner_fit(k, w, m, np.exp(log_sigma))[3]

    best = None
    for m0 in np.linspace(k.min(), k.max(), 5):
        for s0 in (0.05, 0.2, 0.5):
            r = minimize(outer, [m0, np.log(s0)], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-16})
            if best is None or r.fun < best.fun:
                best = r
    m, sigma = best.x[0], float(np.exp(best.x[1]))
    a, d, c, _ = _inner_fit(k, w, m, sigma)
    start = [a, c / sigma, d / c if c > 0 else 0.0, m, sigma]

    def residuals(p):
        a_, b_, rho_, m_, s_ = p
        x = k - m_
        return wt * (a_ + b_ * (rho_ * x + np.sqrt(x**2 + s_**2)) - w)

    lo = [-np.inf, 0.0, -0.999, -np.inf, 1e-4]
    hi = [np.inf, np.inf, 0.999, np.inf, np.inf]
    start = np.clip(start, lo, hi)
    p = least_squares(residuals, start, bounds=(lo, hi), xtol=1e-15, ftol=1e-15, gtol=1e-15).x
    return SVI(*map(float, p))
