"""GARCH(1,1) and GJR-GARCH(1,1) volatility models: estimation, forecasting, evaluation.

Model (returns r_t in percent, mean zero after demeaning):

    r_t = sigma_t z_t,   z_t iid, mean 0, variance 1 (Gaussian or standardised Student-t)
    sigma_t^2 = omega + (alpha + gamma * 1[r_{t-1} < 0]) r_{t-1}^2 + beta sigma_{t-1}^2

``gamma = 0`` is plain GARCH(1,1); ``gamma > 0`` is the GJR leverage effect. Parameters are
estimated by (quasi) maximum likelihood. Variance forecasts are scored with MSE and QLIKE
against a realised proxy, the loss functions that rank volatility forecasts consistently
(Patton, 2011).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats

_EPS = 1e-12


# --------------------------------------------------------------------------- simulation


def simulate_gjr(
    n: int,
    omega: float = 0.02,
    alpha: float = 0.03,
    gamma: float = 0.10,
    beta: float = 0.88,
    nu: float | None = 7.0,
    seed: int = 0,
    burn: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate GJR-GARCH(1,1) returns (percent) and the true conditional variances.

    ``nu=None`` draws Gaussian innovations, otherwise unit-variance Student-t with ``nu`` dof.
    """
    if alpha < 0 or gamma < -alpha or beta < 0 or omega <= 0:
        raise ValueError("invalid parameters")
    if alpha + gamma / 2 + beta >= 1:
        raise ValueError("persistence alpha + gamma/2 + beta must be below 1")
    if nu is not None and nu <= 2:
        raise ValueError("nu must exceed 2")
    rng = np.random.default_rng(seed)
    m = n + burn
    z = rng.standard_normal(m) if nu is None else rng.standard_t(nu, m) * np.sqrt((nu - 2) / nu)
    r = np.empty(m)
    s2 = np.empty(m)
    v = omega / (1 - alpha - gamma / 2 - beta)
    for t in range(m):
        s2[t] = v
        r[t] = np.sqrt(v) * z[t]
        v = omega + (alpha + gamma * (r[t] < 0)) * r[t] ** 2 + beta * v
    return r[burn:], s2[burn:]


# --------------------------------------------------------------------------- filter + likelihood


def variance_filter(
    r: np.ndarray, omega: float, alpha: float, gamma: float, beta: float, s2_0: float
) -> np.ndarray:
    """Conditional variances sigma_t^2 for t = 0..len(r); the last entry is the one-step forecast."""
    n = len(r)
    s2 = np.empty(n + 1)
    s2[0] = s2_0
    for t in range(n):
        s2[t + 1] = omega + (alpha + (gamma if r[t] < 0 else 0.0)) * r[t] ** 2 + beta * s2[t]
    return s2


def _neg_loglik(theta: np.ndarray, r: np.ndarray, dist: str, leverage: bool) -> float:
    omega, alpha, beta = theta[:3]
    gamma = theta[3] if leverage else 0.0
    nu = theta[-1] if dist == "t" else None
    persistence = alpha + gamma / 2 + beta
    if omega <= 0 or alpha < 0 or beta < 0 or gamma < -alpha or persistence >= 1:
        return 1e10
    s2 = variance_filter(r, omega, alpha, gamma, beta, float(np.var(r)))[:-1]
    s2 = np.maximum(s2, _EPS)
    if dist == "normal":
        ll = -0.5 * (np.log(2 * np.pi) + np.log(s2) + r**2 / s2)
    else:
        if nu is None or nu <= 2.05:
            return 1e10
        c = (
            special.gammaln((nu + 1) / 2)
            - special.gammaln(nu / 2)
            - 0.5 * np.log(np.pi * (nu - 2))
        )
        ll = c - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log1p(r**2 / (s2 * (nu - 2)))
    return float(-np.sum(ll))


@dataclass(frozen=True)
class GarchFit:
    """Fitted (GJR-)GARCH(1,1). Returns are in the units used for estimation."""

    omega: float
    alpha: float
    beta: float
    gamma: float
    nu: float | None
    loglik: float
    n_obs: int
    last_return: float
    last_variance: float
    leverage: bool = False

    @property
    def persistence(self) -> float:
        return self.alpha + self.gamma / 2 + self.beta

    @property
    def long_run_variance(self) -> float:
        return self.omega / (1 - self.persistence)

    @property
    def n_params(self) -> int:
        return 3 + int(self.leverage) + int(self.nu is not None)

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.loglik

    @property
    def bic(self) -> float:
        return self.n_params * np.log(self.n_obs) - 2 * self.loglik

    def next_variance(self) -> float:
        """One-step-ahead variance forecast sigma_{T+1}^2."""
        g = self.gamma if self.last_return < 0 else 0.0
        return self.omega + (self.alpha + g) * self.last_return**2 + self.beta * self.last_variance

    def forecast(self, horizon: int) -> np.ndarray:
        """Variance forecasts for h = 1..horizon: V + persistence^(h-1) (sigma_{T+1}^2 - V)."""
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        v = self.long_run_variance
        h = np.arange(horizon)
        return v + self.persistence**h * (self.next_variance() - v)


def fit_garch(
    returns: np.ndarray, leverage: bool = False, dist: str = "normal"
) -> GarchFit:
    """Maximum likelihood fit of GARCH(1,1) (``leverage=False``) or GJR-GARCH(1,1).

    ``dist='normal'`` gives Gaussian QMLE (consistent for the variance parameters even if the
    innovations are fat-tailed); ``dist='t'`` also estimates the Student-t degrees of freedom.
    Returns should be demeaned and in percent (or similar) so the optimiser is well scaled.
    """
    if dist not in ("normal", "t"):
        raise ValueError("dist must be 'normal' or 't'")
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or len(r) < 250:
        raise ValueError("need a one-dimensional series of at least 250 observations")
    v = float(np.var(r))
    x0 = [0.05 * v, 0.05, 0.90]
    bounds = [(1e-8 * v, 10 * v), (0.0, 0.5), (0.0, 0.999)]
    if leverage:
        x0.append(0.05)
        bounds.append((-0.05, 0.5))
    if dist == "t":
        x0.append(8.0)
        bounds.append((2.1, 100.0))
    res = optimize.minimize(
        _neg_loglik, x0, args=(r, dist, leverage), bounds=bounds, method="L-BFGS-B"
    )
    # Nelder-Mead polish: the persistence constraint is not a simple box, so refine locally.
    res2 = optimize.minimize(
        _neg_loglik, res.x, args=(r, dist, leverage), method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000},
    )
    theta = res2.x if res2.fun <= res.fun else res.x
    omega, alpha, beta = theta[:3]
    gamma = theta[3] if leverage else 0.0
    nu = float(theta[-1]) if dist == "t" else None
    s2 = variance_filter(r, omega, alpha, gamma, beta, v)
    return GarchFit(
        omega=float(omega), alpha=float(alpha), beta=float(beta), gamma=float(gamma), nu=nu,
        loglik=-_neg_loglik(theta, r, dist, leverage), n_obs=len(r),
        last_return=float(r[-1]), last_variance=float(s2[-2]), leverage=leverage,
    )


# --------------------------------------------------------------------------- baselines


def ewma_forecasts(r: np.ndarray, lam: float = 0.94, s2_0: float | None = None) -> np.ndarray:
    """RiskMetrics EWMA one-step variance forecasts; element t forecasts r_t from r_{<t}."""
    r = np.asarray(r, dtype=float)
    out = np.empty(len(r))
    s2 = float(np.var(r[:50])) if s2_0 is None else s2_0
    for t in range(len(r)):
        out[t] = s2
        s2 = lam * s2 + (1 - lam) * r[t] ** 2
    return out


def rolling_variance_forecasts(r: np.ndarray, window: int) -> np.ndarray:
    """Equal-weight rolling-window variance forecast of r_t from the previous ``window`` returns."""
    r = np.asarray(r, dtype=float)
    out = np.full(len(r), np.nan)
    sq = np.cumsum(np.insert(r**2, 0, 0.0))
    for t in range(window, len(r)):
        out[t] = (sq[t] - sq[t - window]) / window
    return out


def garch_rolling_forecasts(
    r: np.ndarray, start: int, refit_every: int = 21, leverage: bool = False,
    dist: str = "normal", window: int | None = None,
) -> np.ndarray:
    """Out-of-sample one-step variance forecasts for t >= start (earlier entries are NaN).

    Parameters are re-estimated every ``refit_every`` days on an expanding window (or the last
    ``window`` days); between refits the variance filter is simply updated with new returns,
    so no information from day t is used to forecast day t.
    """
    r = np.asarray(r, dtype=float)
    out = np.full(len(r), np.nan)
    fit: GarchFit | None = None
    for t in range(start, len(r)):
        if fit is None or (t - start) % refit_every == 0:
            lo = 0 if window is None else max(0, t - window)
            fit = fit_garch(r[lo:t], leverage=leverage, dist=dist)
            s2_next = fit.next_variance()
        else:
            g = fit.gamma if r[t - 1] < 0 else 0.0
            s2_next = fit.omega + (fit.alpha + g) * r[t - 1] ** 2 + fit.beta * s2_next
        out[t] = s2_next
    return out


# --------------------------------------------------------------------------- evaluation


def mse_loss(forecast: np.ndarray, proxy: np.ndarray) -> np.ndarray:
    """Squared error of the variance forecast against the proxy."""
    return (np.asarray(proxy) - np.asarray(forecast)) ** 2


def qlike_loss(forecast: np.ndarray, proxy: np.ndarray) -> np.ndarray:
    """QLIKE = proxy/forecast - log(proxy/forecast) - 1 (zero at a perfect forecast)."""
    f = np.maximum(np.asarray(forecast, dtype=float), _EPS)
    p = np.maximum(np.asarray(proxy, dtype=float), _EPS)
    return p / f - np.log(p / f) - 1.0


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lags: int = 5) -> tuple[float, float]:
    """Diebold-Mariano test of equal expected loss (HAC variance, Bartlett kernel).

    Returns (statistic, two-sided p-value). A negative statistic means model A has lower loss.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    dc = d - d.mean()
    var = float(dc @ dc) / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * float(dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(var / n)
    return float(stat), float(2 * stats.norm.sf(abs(stat)))


def mincer_zarnowitz(forecast: np.ndarray, proxy: np.ndarray) -> tuple[float, float, float]:
    """Regress proxy on forecast: proxy = a + b forecast. Returns (a, b, R^2).

    Unbiased forecasts have a = 0 and b = 1. With a noisy proxy (squared returns) R^2 is low
    even for the true variance, so compare R^2 across models rather than to 1.
    """
    x = np.asarray(forecast, dtype=float)
    y = np.asarray(proxy, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    return float(coef[0]), float(coef[1]), r2
