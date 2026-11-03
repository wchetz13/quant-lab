"""Heston (1993) stochastic volatility model.

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW^S_t
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW^v_t,    d<W^S, W^v>_t = rho dt

European prices come from the characteristic function by Fourier inversion. Paths come from
Andersen's (2008) quadratic-exponential scheme, which keeps the variance non-negative without
the truncation bias of Euler schemes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass(frozen=True)
class HestonParams:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        if min(self.v0, self.theta) < 0 or min(self.kappa, self.xi) <= 0:
            raise ValueError("v0, theta must be >= 0 and kappa, xi > 0")
        if not -1 < self.rho < 1:
            raise ValueError("rho must lie in (-1, 1)")

    @property
    def feller(self) -> bool:
        """True when 2 kappa theta >= xi^2, so the variance never reaches zero."""
        return 2 * self.kappa * self.theta >= self.xi**2


def characteristic_function(u, S, T, r, q, p: HestonParams):
    """E[exp(i u ln S_T)] under the risk-neutral measure.

    Uses the "little Heston trap" form (Albrecher et al., 2007). Heston's original form takes
    the complex logarithm across its branch cut at long maturities and returns prices with
    jumps in them; this form stays on the principal branch.
    """
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    beta = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(beta**2 + p.xi**2 * (iu + u**2))
    g = (beta - d) / (beta + d)
    edt = np.exp(-d * T)
    C = p.kappa * p.theta / p.xi**2 * ((beta - d) * T - 2 * np.log((1 - g * edt) / (1 - g)))
    D = (beta - d) / p.xi**2 * (1 - edt) / (1 - g * edt)
    return np.exp(iu * (np.log(S) + (r - q) * T) + C + D * p.v0)


def call_price(S, K, T, r, q, p: HestonParams) -> float:
    """European call by Gil-Pelaez inversion: C = S e^{-qT} P1 - K e^{-rT} P2."""
    k = np.log(K)
    forward = S * np.exp((r - q) * T)

    def integrand(u, j):
        if j == 1:
            # Share-measure probability: tilt by S_T / F, i.e. shift the argument by -i.
            phi = characteristic_function(u - 1j, S, T, r, q, p) / forward
        else:
            phi = characteristic_function(u, S, T, r, q, p)
        return (np.exp(-1j * u * k) * phi / (1j * u)).real

    P1, P2 = (0.5 + quad(integrand, 0, np.inf, args=(j,), limit=500)[0] / np.pi for j in (1, 2))
    return float(S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2)


def put_price(S, K, T, r, q, p: HestonParams) -> float:
    return call_price(S, K, T, r, q, p) - S * np.exp(-q * T) + K * np.exp(-r * T)


def simulate_qe(S, T, r, q, p: HestonParams, n_steps, n_paths, seed=None, psi_c=1.5):
    """Terminal prices S_T from Andersen's QE scheme with central-difference log-price weights."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    ekt = np.exp(-p.kappa * dt)
    g1 = g2 = 0.5
    K0 = -p.rho * p.kappa * p.theta * dt / p.xi
    K1 = g1 * dt * (p.kappa * p.rho / p.xi - 0.5) - p.rho / p.xi
    K2 = g2 * dt * (p.kappa * p.rho / p.xi - 0.5) + p.rho / p.xi
    K3 = g1 * dt * (1 - p.rho**2)
    K4 = g2 * dt * (1 - p.rho**2)

    x = np.full(n_paths, np.log(S))
    v = np.full(n_paths, p.v0)
    for _ in range(n_steps):
        # Match the exact conditional mean and variance of v_{t+dt} given v_t.
        m = p.theta + (v - p.theta) * ekt
        s2 = v * p.xi**2 * ekt / p.kappa * (1 - ekt) + p.theta * p.xi**2 / (2 * p.kappa) * (1 - ekt) ** 2
        psi = s2 / m**2
        z, u, zx = rng.standard_normal(n_paths), rng.uniform(size=n_paths), rng.standard_normal(n_paths)

        v_next = np.empty(n_paths)
        quad_ = psi <= psi_c
        # Low dispersion: v' = a (b + Z)^2, a scaled non-central chi-square with one degree of freedom.
        inv = 2 / psi[quad_]
        b2 = inv - 1 + np.sqrt(inv * (inv - 1))
        a = m[quad_] / (1 + b2)
        v_next[quad_] = a * (np.sqrt(b2) + z[quad_]) ** 2
        # High dispersion: a point mass at zero mixed with an exponential tail.
        exp_ = ~quad_
        pz = (psi[exp_] - 1) / (psi[exp_] + 1)
        beta = (1 - pz) / m[exp_]
        ue = u[exp_]
        v_next[exp_] = np.where(ue <= pz, 0.0, np.log((1 - pz) / np.maximum(1 - ue, 1e-300)) / beta)

        x += (r - q) * dt + K0 + K1 * v + K2 * v_next + np.sqrt(K3 * v + K4 * v_next) * zx
        v = v_next
    return np.exp(x)


def bs_call(S, K, T, r, q, sigma) -> float:
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def implied_vol(price, S, K, T, r, q) -> float:
    """Black–Scholes volatility that reproduces a call price."""
    return brentq(lambda s: bs_call(S, K, T, r, q, s) - price, 1e-6, 5.0, xtol=1e-12)
