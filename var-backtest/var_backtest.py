"""Value-at-Risk and Expected Shortfall forecasters with formal backtests.

Four one-day forecasters are compared on the same return series: historical simulation, a
Gaussian model with rolling volatility, RiskMetrics EWMA, and filtered historical simulation
(FHS). Their violation sequences are scored with Kupiec's proportion-of-failures test,
Christoffersen's independence test, and the Basel traffic-light zones.

Conventions: returns are log returns, VaR and ES are reported as positive loss numbers, and
``alpha`` is the tail probability (0.01 means 99% VaR). A violation occurs when ``r_t < -VaR_t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- data


def simulate_garch(
    n: int,
    omega: float = 2e-6,
    alpha: float = 0.09,
    beta: float = 0.90,
    nu: float = 6.0,
    seed: int = 0,
    burn: int = 500,
) -> np.ndarray:
    """Simulate GARCH(1,1) returns with unit-variance Student-t innovations.

    sigma_t^2 = omega + alpha r_{t-1}^2 + beta sigma_{t-1}^2. Volatility clusters and tails are
    fat, which is what makes the Gaussian and unconditional models fail in a backtest.
    """
    if alpha + beta >= 1 or nu <= 2:
        raise ValueError("need alpha + beta < 1 and nu > 2")
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, size=n + burn) * np.sqrt((nu - 2) / nu)
    r = np.empty(n + burn)
    var = omega / (1 - alpha - beta)
    for t in range(n + burn):
        r[t] = np.sqrt(var) * z[t]
        var = omega + alpha * r[t] ** 2 + beta * var
    return r[burn:]


# --------------------------------------------------------------------------- forecasters


def _check(returns: np.ndarray, window: int, alpha: float) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5)")
    if window < 20 or window >= len(r):
        raise ValueError("window must be at least 20 and shorter than the series")
    return r


def historical(returns, window: int = 500, alpha: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Historical simulation: empirical quantile and tail mean of the last ``window`` returns.

    Returns (VaR, ES) arrays aligned with ``returns[window:]``.
    """
    r = _check(returns, window, alpha)
    var, es = np.empty(len(r) - window), np.empty(len(r) - window)
    for i in range(len(var)):
        w = r[i : i + window]
        q = np.quantile(w, alpha)
        var[i] = -q
        tail = w[w <= q]
        es[i] = -tail.mean()
    return var, es


def gaussian(returns, window: int = 500, alpha: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean normal VaR/ES with the sample standard deviation of the last ``window`` returns."""
    r = _check(returns, window, alpha)
    sigma = np.array([np.sqrt(np.mean(r[i : i + window] ** 2)) for i in range(len(r) - window)])
    z = stats.norm.ppf(alpha)
    return -z * sigma, sigma * stats.norm.pdf(z) / alpha


def ewma_volatility(returns, lam: float = 0.94, init_window: int = 250) -> np.ndarray:
    """RiskMetrics one-step volatility forecasts.

    ``out[t]`` uses returns up to t-1 only: sigma_t^2 = lam sigma_{t-1}^2 + (1 - lam) r_{t-1}^2,
    seeded with the sample variance of the first ``init_window`` returns.
    """
    r = np.asarray(returns, dtype=float)
    if not 0 < lam < 1:
        raise ValueError("lam must lie in (0, 1)")
    var = np.empty(len(r))
    var[0] = np.mean(r[:init_window] ** 2)
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return np.sqrt(var)


def ewma(
    returns, window: int = 500, alpha: float = 0.01, lam: float = 0.94
) -> tuple[np.ndarray, np.ndarray]:
    """RiskMetrics: Gaussian VaR/ES scaled by the EWMA volatility forecast."""
    r = _check(returns, window, alpha)
    sigma = ewma_volatility(r, lam)[window:]
    z = stats.norm.ppf(alpha)
    return -z * sigma, sigma * stats.norm.pdf(z) / alpha


def filtered_historical(
    returns, window: int = 500, alpha: float = 0.01, lam: float = 0.94
) -> tuple[np.ndarray, np.ndarray]:
    """Filtered historical simulation (Barone-Adesi et al., 1999).

    Past returns are standardised by the EWMA volatility that was known at the time, the
    empirical quantile and tail mean of those residuals are taken, and they are rescaled by
    today's volatility forecast. This keeps the fat tails of the data but reacts to clustering.
    """
    r = _check(returns, window, alpha)
    sigma = ewma_volatility(r, lam)
    z = r / sigma
    var, es = np.empty(len(r) - window), np.empty(len(r) - window)
    for i in range(len(var)):
        w = z[i : i + window]
        q = np.quantile(w, alpha)
        s = sigma[i + window]
        var[i] = -q * s
        es[i] = -w[w <= q].mean() * s
    return var, es


FORECASTERS = {
    "Historical": historical,
    "Gaussian": gaussian,
    "EWMA (0.94)": ewma,
    "Filtered HS": filtered_historical,
}


# --------------------------------------------------------------------------- backtests


def violations(realised: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Boolean hit sequence: True where the loss exceeds the VaR forecast."""
    realised, var = np.asarray(realised), np.asarray(var)
    if realised.shape != var.shape:
        raise ValueError("realised and var must have the same shape")
    return realised < -var


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float

    def reject(self, level: float = 0.05) -> bool:
        return self.p_value < level


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(hits: np.ndarray, alpha: float) -> TestResult:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test of unconditional coverage.

    H0: the violation probability equals ``alpha``. LR_uc ~ chi-square(1).
    """
    hits = np.asarray(hits, dtype=bool)
    n, x = len(hits), int(hits.sum())
    pi = x / n
    ll0 = _xlogy(n - x, 1 - alpha) + _xlogy(x, alpha)
    ll1 = _xlogy(n - x, 1 - pi) + _xlogy(x, pi)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return TestResult(lr, float(stats.chi2.sf(lr, 1)))


def christoffersen_independence(hits: np.ndarray) -> TestResult:
    """Christoffersen (1998) test that violations do not cluster (first-order Markov chain).

    H0: P(hit today | hit yesterday) = P(hit today | no hit yesterday). LR_ind ~ chi-square(1).
    """
    h = np.asarray(hits, dtype=int)
    prev, cur = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll1 = (
        _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    )
    lr = max(-2 * (ll0 - ll1), 0.0)
    return TestResult(lr, float(stats.chi2.sf(lr, 1)))


def conditional_coverage(hits: np.ndarray, alpha: float) -> TestResult:
    """Christoffersen's joint test: LR_cc = LR_uc + LR_ind ~ chi-square(2)."""
    lr = kupiec_pof(hits, alpha).statistic + christoffersen_independence(hits).statistic
    return TestResult(lr, float(stats.chi2.sf(lr, 2)))


def basel_zone(n_violations: int, alpha: float = 0.01, n_obs: int = 250) -> str:
    """Basel traffic-light zone, from binomial cumulative probabilities.

    With ``cdf = P(X <= n_violations)`` under a correct model: green below 95%, yellow below
    99.99%, red beyond. For 250 days at 99% this gives green 0-4, yellow 5-9, red 10+, matching
    the Basel Committee table.
    """
    cdf = stats.binom.cdf(n_violations, n_obs, alpha)
    if cdf < 0.95:
        return "green"
    return "yellow" if cdf < 0.9999 else "red"


def acerbi_szekely_z2(
    realised: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float
) -> float:
    """Acerbi-Szekely Z2 statistic for Expected Shortfall.

    Z2 = sum_t [ L_t 1{L_t > VaR_t} / (T alpha ES_t) ] - 1 with L_t = -r_t. It is zero in
    expectation for a correct ES, positive when ES is understated.
    """
    loss = -np.asarray(realised)
    hit = loss > np.asarray(var)
    return float(np.sum(loss * hit / np.asarray(es)) / (len(loss) * alpha) - 1.0)


# --------------------------------------------------------------------------- driver


def backtest_all(
    returns: np.ndarray, window: int = 500, alpha: float = 0.01
) -> dict[str, dict[str, float]]:
    """Run every forecaster and collect the backtest statistics."""
    r = np.asarray(returns, dtype=float)
    out: dict[str, dict[str, float]] = {}
    for name, fn in FORECASTERS.items():
        var, es = fn(r, window, alpha)
        realised = r[window:]
        hits = violations(realised, var)
        out[name] = {
            "violations": int(hits.sum()),
            "rate": float(hits.mean()),
            "kupiec_p": kupiec_pof(hits, alpha).p_value,
            "independence_p": christoffersen_independence(hits).p_value,
            "cc_p": conditional_coverage(hits, alpha).p_value,
            "es_z2": acerbi_szekely_z2(realised, var, es, alpha),
        }
    return out
