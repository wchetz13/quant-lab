"""Reproduce the README tables and figure. All data are synthetic, generated from a known mu and Sigma."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from portfolio_opt import (
    black_litterman,
    efficient_frontier,
    implied_returns,
    ledoit_wolf_identity,
    max_sharpe,
    min_variance,
    portfolio_stats,
    risk_contributions,
    risk_parity,
)

NAMES = ["US eq", "Intl eq", "Gov bonds", "Corp bonds", "EM eq"]
VOL = np.array([0.18, 0.22, 0.07, 0.12, 0.30])
CORR = np.array(
    [
        [1.0, 0.7, -0.1, 0.5, 0.3],
        [0.7, 1.0, -0.1, 0.5, 0.35],
        [-0.1, -0.1, 1.0, 0.1, 0.0],
        [0.5, 0.5, 0.1, 1.0, 0.25],
        [0.3, 0.35, 0.0, 0.25, 1.0],
    ]
)
COV = CORR * np.outer(VOL, VOL)
MU = np.array([0.07, 0.055, 0.03, 0.045, 0.10])
RF = 0.02
W_MKT = np.array([0.35, 0.25, 0.20, 0.10, 0.10])


def fmt(w) -> str:
    return " | ".join(f"{x * 100:6.1f}" for x in w)


def weights_table() -> dict[str, np.ndarray]:
    ports = {
        "Equal weight": np.full(5, 0.2),
        "Min variance (L/S)": min_variance(COV),
        "Min variance (long-only)": min_variance(COV, long_only=True),
        "Max Sharpe (L/S)": max_sharpe(MU, COV, RF),
        "Max Sharpe (long-only)": max_sharpe(MU, COV, RF, long_only=True),
        "Risk parity": risk_parity(COV),
    }
    print("Weights (%) and risk, true mu/Sigma")
    print("portfolio | " + " | ".join(NAMES) + " | ret % | vol % | Sharpe")
    for k, w in ports.items():
        r, v = portfolio_stats(w, MU, COV)
        print(f"{k} | {fmt(w)} | {r * 100:5.2f} | {v * 100:5.2f} | {(r - RF) / v:5.3f}")
    print("\nRisk parity contributions (%):", np.round(risk_contributions(ports['Risk parity'], COV) * 100, 2))
    return ports


FALLBACKS = {False: 0, True: 0}


def tangency_or_equal(mu_hat, cov_hat, long_only: bool = False) -> np.ndarray:
    """Unconstrained tangency weights; equal weight when the estimate has no valid tangency portfolio."""
    try:
        return max_sharpe(mu_hat, cov_hat, RF, long_only=long_only)
    except ValueError:
        FALLBACKS[long_only] += 1
        return np.full(5, 0.2)


def estimation_study(n_trials: int = 1000, months: int = 120, seed: int = 11) -> None:
    """Estimate mu and Sigma from `months` of simulated returns, then score weights with the TRUE mu, Sigma."""
    rng = np.random.default_rng(seed)
    strategies = ["Max Sharpe (L/S)", "Max Sharpe (long-only)", "Min var (sample)", "Min var (Ledoit-Wolf)", "Risk parity", "Equal weight"]
    sharpe = {s: [] for s in strategies}
    turnover_gross = {s: [] for s in strategies}
    deltas = []
    for _ in range(n_trials):
        x = rng.multivariate_normal(MU / 12, COV / 12, size=months)
        mu_hat, cov_hat = x.mean(axis=0) * 12, np.cov(x.T) * 12
        lw, delta = ledoit_wolf_identity(x)
        deltas.append(delta)
        w = {
            "Max Sharpe (L/S)": tangency_or_equal(mu_hat, cov_hat),
            "Max Sharpe (long-only)": tangency_or_equal(mu_hat, cov_hat, long_only=True),
            "Min var (sample)": min_variance(cov_hat),
            "Min var (Ledoit-Wolf)": min_variance(lw * 12),
            "Risk parity": risk_parity(cov_hat),
            "Equal weight": np.full(5, 0.2),
        }
        for s, ws in w.items():
            r, v = portfolio_stats(ws, MU, COV)
            sharpe[s].append((r - RF) / v)
            turnover_gross[s].append(np.abs(ws).sum())
    oracle = max_sharpe(MU, COV, RF)
    r, v = portfolio_stats(oracle, MU, COV)
    print(f"\nEstimation error: {n_trials} trials, {months} months of data each. Out-of-sample = true mu, Sigma.")
    print(f"Oracle (true-parameter) Sharpe = {(r - RF) / v:.3f}; mean Ledoit-Wolf shrinkage = {np.mean(deltas):.3f}")
    print(
        f"No valid tangency portfolio (replaced by equal weight): L/S {FALLBACKS[False]} trials, "
        f"long-only {FALLBACKS[True]} trials"
    )
    print("strategy | mean true Sharpe | 5th pct | 95th pct | mean gross leverage")
    for s in strategies:
        a = np.array(sharpe[s])
        print(f"{s} | {a.mean():.3f} | {np.percentile(a, 5):.3f} | {np.percentile(a, 95):.3f} | {np.mean(turnover_gross[s]):.2f}")


def bl_demo() -> None:
    delta = 2.5
    pi = implied_returns(COV, W_MKT, delta)
    P = np.array([[0, 1.0, 0, 0, 0], [0, 0, 0, 1.0, -1.0]])
    Q = np.array([0.10, 0.00])  # Intl equities return 10%; corporates and EM expected to match each other
    mean, cov_mu = black_litterman(COV, pi, P, Q, tau=0.05)
    w_prior = max_sharpe(pi, COV)
    w_post = max_sharpe(mean, COV + cov_mu)
    print("\nBlack-Litterman (delta=2.5, tau=0.05, He-Litterman Omega)")
    print("asset | market w % | implied excess mu % | posterior excess mu % | posterior-optimal w %")
    for i, n in enumerate(NAMES):
        print(f"{n} | {W_MKT[i] * 100:5.1f} | {pi[i] * 100:5.2f} | {mean[i] * 100:5.2f} | {w_post[i] * 100:6.1f}")
    print("weights recover market portfolio with no views:", np.round(w_prior * 100, 1))


def figure(ports: dict[str, np.ndarray]) -> None:
    targets = np.linspace(0.03, 0.10, 60)
    ls = efficient_frontier(MU, COV, targets)
    lo = efficient_frontier(MU, COV, np.linspace(0.031, 0.099, 40), long_only=True)
    vol = lambda ws: np.sqrt(np.einsum("ij,jk,ik->i", ws, COV, ws))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(vol(ls) * 100, ls @ MU * 100, label="Frontier (long/short)")
    ax.plot(vol(lo) * 100, lo @ MU * 100, label="Frontier (long-only)")
    ax.scatter(VOL * 100, MU * 100, c="gray", marker="x", label="Assets")
    for i, n in enumerate(NAMES):
        ax.annotate(n, (VOL[i] * 100, MU[i] * 100), xytext=(4, 4), textcoords="offset points", fontsize=8)
    for k in ("Equal weight", "Risk parity", "Max Sharpe (long-only)", "Min variance (long-only)"):
        r, v = portfolio_stats(ports[k], MU, COV)
        ax.scatter(v * 100, r * 100, s=45, zorder=3, label=k)
    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Expected return (%)")
    ax.set_title("Mean-variance frontier, synthetic 5-asset universe")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("portfolio_opt.png", dpi=150)


def main() -> None:
    ports = weights_table()
    estimation_study()
    bl_demo()
    figure(ports)


if __name__ == "__main__":
    main()
