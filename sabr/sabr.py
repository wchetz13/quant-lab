"""SABR stochastic volatility model and Hagan's implied volatility expansion.

    dF_t = alpha_t F_t^beta dW_t,    d alpha_t = nu alpha_t dZ_t,    d<W, Z>_t = rho dt

SABR is the market standard for interest rate smiles. Hagan et al. (2002) give an asymptotic
expansion for the Black implied volatility that is fast and accurate for short expiries, but it
is not exact: at long expiries and low strikes it implies negative probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm


@dataclass(frozen=True)
class SABR:
    alpha: float
    beta: float
    rho: float
    nu: float

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.nu < 0 or not 0 <= self.beta <= 1 or not -1 < self.rho < 1:
            raise ValueError("need alpha > 0, nu >= 0, 0 <= beta <= 1 and |rho| < 1")


def hagan_vol(F, K, T, p: SABR):
    """Black (lognormal) implied volatility from Hagan et al. (2002), eq. 2.17a."""
    F, K = float(F), np.asarray(K, float)
    a, b, r, n = p.alpha, p.beta, p.rho, p.nu
    omb = 1 - b
    fk = F * K
    log_fk = np.log(F / K)
    fk_pow = fk ** (omb / 2)
    z = n / a * fk_pow * log_fk
    # z / x(z) -> 1 as K -> F; use its Taylor series near zero to avoid 0/0.
    small = np.abs(z) < 1e-6
    zs = np.where(small, 1.0, z)
    x = np.log((np.sqrt(1 - 2 * r * zs + zs**2) + zs - r) / (1 - r))
    z_over_x = np.where(small, 1 - r * z / 2 + (2 - 3 * r**2) * z**2 / 12, zs / x)
    denom = fk_pow * (1 + omb**2 / 24 * log_fk**2 + omb**4 / 1920 * log_fk**4)
    correction = 1 + (omb**2 / 24 * a**2 / fk**omb + r * b * n * a / (4 * fk_pow) + (2 - 3 * r**2) / 24 * n**2) * T
    return a / denom * z_over_x * correction


def black_call(F, K, T, vol, discount=1.0):
    sd = vol * np.sqrt(T)
    d1 = np.log(F / K) / sd + sd / 2
    return discount * (F * norm.cdf(d1) - K * norm.cdf(d1 - sd))


def implied_density(F, K, T, p: SABR, h=None):
    """Risk-neutral density of F_T from Breeden–Litzenberger: d^2 C / dK^2 under Hagan vols."""
    K = np.asarray(K, float)
    h = 1e-4 * F if h is None else h
    c = lambda k: black_call(F, k, T, hagan_vol(F, k, T, p))
    return (c(K + h) - 2 * c(K) + c(K - h)) / h**2


def simulate(F, T, p: SABR, n_steps, n_paths, seed=None):
    """Terminal forwards. Volatility is stepped exactly (it is lognormal); the forward uses
    log-Euler for beta = 1 and Euler with absorption at zero otherwise."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    f = np.full(n_paths, float(F))
    a = np.full(n_paths, p.alpha)
    alive = np.ones(n_paths, bool)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = p.rho * z1 + np.sqrt(1 - p.rho**2) * rng.standard_normal(n_paths)
        if p.beta == 1:
            f *= np.exp(a * np.sqrt(dt) * z1 - 0.5 * a**2 * dt)
        else:
            f = np.where(alive, f + a * np.maximum(f, 0) ** p.beta * np.sqrt(dt) * z1, 0.0)
            alive &= f > 0
            f = np.maximum(f, 0.0)
        a *= np.exp(p.nu * np.sqrt(dt) * z2 - 0.5 * p.nu**2 * dt)
    return f


def calibrate(F, K, T, vols, beta, start=(None, 0.0, 0.5)) -> SABR:
    """Fit alpha, rho, nu to market vols with beta fixed, as is market practice.

    beta and rho both control the skew and are nearly collinear on a single smile, so beta is
    chosen from the desired backbone (how the ATM vol moves with F) rather than fitted.
    """
    K, vols = np.asarray(K, float), np.asarray(vols, float)
    atm = vols[np.argmin(np.abs(K - F))]
    a0 = start[0] if start[0] is not None else atm * F ** (1 - beta)

    def residuals(x):
        return hagan_vol(F, K, T, SABR(np.exp(x[0]), beta, np.tanh(x[1]), np.exp(x[2]))) - vols

    x0 = [np.log(a0), np.arctanh(start[1]), np.log(start[2])]
    x = least_squares(residuals, x0, xtol=1e-14, ftol=1e-14, gtol=1e-14).x
    return SABR(float(np.exp(x[0])), beta, float(np.tanh(x[1])), float(np.exp(x[2])))
