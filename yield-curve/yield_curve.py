"""Yield curve construction: bootstrapping from deposits and par swaps, and Nelson-Siegel-Svensson fits.

Conventions: all rates are continuously compounded zero rates unless stated otherwise, times are in
years, and swaps pay a fixed coupon annually. Discount factors are interpolated log-linearly, which is
the same as assuming piecewise-constant instantaneous forward rates between nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize


class DiscountCurve:
    """Discount curve defined by nodes (t_i, D(t_i)) with log-linear interpolation, D(0) = 1."""

    def __init__(self, times, discount_factors) -> None:
        t = np.asarray(times, dtype=float)
        d = np.asarray(discount_factors, dtype=float)
        if t.ndim != 1 or t.shape != d.shape or t.size == 0:
            raise ValueError("times and discount_factors must be 1-D arrays of equal, non-zero length")
        if t[0] <= 0 or np.any(np.diff(t) <= 0):
            raise ValueError("times must be positive and strictly increasing")
        if np.any(d <= 0):
            raise ValueError("discount factors must be positive")
        self.times = np.concatenate(([0.0], t))
        self._log_d = np.concatenate(([0.0], np.log(d)))

    def log_df(self, t):
        """ln D(t); linear between nodes and extrapolated with the last forward rate."""
        t = np.asarray(t, dtype=float)
        inside = np.interp(t, self.times, self._log_d)
        slope = (self._log_d[-1] - self._log_d[-2]) / (self.times[-1] - self.times[-2])
        return np.where(t > self.times[-1], self._log_d[-1] + slope * (t - self.times[-1]), inside)

    def df(self, t):
        return np.exp(self.log_df(t))

    def zero_rate(self, t):
        """Continuously compounded zero rate -ln D(t) / t."""
        t = np.asarray(t, dtype=float)
        return -self.log_df(t) / t

    def forward_rate(self, t1, t2):
        """Average continuously compounded forward rate between t1 < t2."""
        t1, t2 = np.asarray(t1, dtype=float), np.asarray(t2, dtype=float)
        return (self.log_df(t1) - self.log_df(t2)) / (t2 - t1)

    def instantaneous_forward(self, t):
        """Piecewise-constant forward -d ln D / dt (right-continuous)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        slopes = -np.diff(self._log_d) / np.diff(self.times)
        idx = np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, slopes.size - 1)
        return slopes[idx]


@dataclass(frozen=True)
class Deposit:
    """Money-market deposit paying 1 + rate * maturity at maturity (simple interest)."""

    maturity: float
    rate: float


@dataclass(frozen=True)
class Swap:
    """Par swap with annual fixed payments at 1, 2, ..., maturity (maturity must be an integer)."""

    maturity: float
    rate: float

    def __post_init__(self) -> None:
        if self.maturity < 1 or abs(self.maturity - round(self.maturity)) > 1e-12:
            raise ValueError("swap maturity must be a whole number of years >= 1")


def par_swap_rate(curve: DiscountCurve, maturity: int) -> float:
    """Par rate s with s * sum_i D(i) = 1 - D(maturity) over annual dates i = 1..maturity."""
    pay = np.arange(1, int(maturity) + 1, dtype=float)
    return float((1.0 - curve.df(maturity)) / curve.df(pay).sum())


def bootstrap(instruments: list[Deposit | Swap]) -> DiscountCurve:
    """Bootstrap a discount curve so that every instrument reprices to par.

    Instruments are processed in order of maturity. Deposits give a discount factor in closed form.
    For a swap whose intermediate annual dates fall between existing nodes, the new discount factor
    is found with a root finder, since interpolating the earlier dates depends on it.
    """
    ordered = sorted(instruments, key=lambda i: i.maturity)
    if len({i.maturity for i in ordered}) != len(ordered):
        raise ValueError("each maturity may appear only once")
    times: list[float] = []
    dfs: list[float] = []
    for inst in ordered:
        if isinstance(inst, Deposit):
            times.append(inst.maturity)
            dfs.append(1.0 / (1.0 + inst.rate * inst.maturity))
            continue
        n = int(round(inst.maturity))
        pay = np.arange(1, n + 1, dtype=float)

        def residual(df_n: float) -> float:
            curve = DiscountCurve([*times, inst.maturity], [*dfs, df_n])
            return inst.rate * curve.df(pay).sum() - (1.0 - df_n)

        df_n = brentq(residual, 1e-6, 5.0, xtol=1e-14)
        times.append(inst.maturity)
        dfs.append(df_n)
    return DiscountCurve(times, dfs)


def nss_zero_rate(t, beta0, beta1, beta2, beta3, tau1, tau2):
    """Nelson-Siegel-Svensson zero rate; beta3 = 0 gives the Nelson-Siegel curve."""
    t = np.asarray(t, dtype=float)
    x1, x2 = t / tau1, t / tau2
    f1 = -np.expm1(-x1) / x1
    f2 = -np.expm1(-x2) / x2
    return beta0 + beta1 * f1 + beta2 * (f1 - np.exp(-x1)) + beta3 * (f2 - np.exp(-x2))


def fit_nelson_siegel(t, zero_rates):
    """Fit Nelson-Siegel by variable projection: betas by least squares, tau1 by 1-D search.

    Returns (params, rmse) with params = (beta0, beta1, beta2, 0, tau1, 1).
    """
    t, y = np.asarray(t, float), np.asarray(zero_rates, float)

    def solve(tau):
        x = t / tau
        f = -np.expm1(-x) / x
        design = np.column_stack([np.ones_like(t), f, f - np.exp(-x)])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        return beta, float(np.sqrt(np.mean((design @ beta - y) ** 2)))

    grid = np.geomspace(0.1, 20.0, 80)
    tau = grid[int(np.argmin([solve(g)[1] for g in grid]))]
    res = minimize(lambda p: solve(p[0])[1], [tau], bounds=[(0.05, 30.0)], method="L-BFGS-B")
    tau = float(res.x[0])
    beta, rmse = solve(tau)
    return (*beta, 0.0, tau, 1.0), rmse


def fit_nss(t, zero_rates, tau_grid=None):
    """Fit Nelson-Siegel-Svensson with variable projection over (tau1, tau2) with tau1 < tau2.

    For fixed taus the four betas are linear. A coarse grid search over the taus avoids the poor
    local minima caused by the near-collinearity of the two hump loadings, then a local polish
    refines them. Returns (params, rmse) with params = (beta0, beta1, beta2, beta3, tau1, tau2).
    """
    t, y = np.asarray(t, float), np.asarray(zero_rates, float)

    def solve(t1, t2):
        x1, x2 = t / t1, t / t2
        f1, f2 = -np.expm1(-x1) / x1, -np.expm1(-x2) / x2
        design = np.column_stack([np.ones_like(t), f1, f1 - np.exp(-x1), f2 - np.exp(-x2)])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        return beta, float(np.sqrt(np.mean((design @ beta - y) ** 2)))

    grid = np.geomspace(0.2, 30.0, 25) if tau_grid is None else np.asarray(tau_grid, float)
    best = min(((solve(a, b)[1], a, b) for a in grid for b in grid if b > 1.5 * a))
    res = minimize(
        lambda p: solve(*p)[1] if p[1] > p[0] else 1e3,
        best[1:],
        bounds=[(0.05, 30.0), (0.05, 60.0)],
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-14},
    )
    t1, t2 = (float(v) for v in res.x)
    beta, rmse = solve(t1, t2)
    return (*beta, t1, t2), rmse
