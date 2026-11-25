"""Kelly sizing study on simulated bets. Reproduces every table and the figure in the README.

``python run_kelly.py``. All data are synthetic with a known edge, so the study shows how sizing
rules behave rather than claiming anything about a real market.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kelly import (
    drawdown_probability, estimated_kelly_growth, kelly_binary, log_growth, max_drawdown,
    simulate_binary,
)

P, B = 0.55, 1.0
OUT = Path(__file__).with_name("kelly_study.png")


def sizing_table() -> None:
    f_star = kelly_binary(P, B)
    x, q = np.array([B, -1.0]), np.array([P, 1 - P])
    print(f"Bet: p={P}, b={B}, Kelly f*={f_star:.3f}, optimal growth/bet={log_growth(f_star, x, q):.5f}")
    print("\nTable 1: 1,000 bets, 5,000 paths")
    print("| c (x Kelly) | growth/bet | median wealth | P(wealth<1) | mean max DD | P(ever <= 0.5) |")
    print("|---|---|---|---|---|---|")
    for c in (0.25, 0.5, 1.0, 1.5, 2.0):
        paths = simulate_binary(P, B, c * f_star, 1000, 5000, seed=7)
        g = log_growth(c * f_star, x, q)
        print(
            f"| {c:.2f} | {g:.5f} | {np.median(paths[:, -1]):.1f} | {(paths[:, -1] < 1).mean():.3f} "
            f"| {max_drawdown(paths).mean():.3f} | {(paths.min(axis=1) <= 0.5).mean():.3f} |"
        )


def drawdown_table() -> None:
    p, b = 0.51, 1.0
    f_full = kelly_binary(p, b)
    print(f"\nTable 2: P(wealth ever falls to alpha), p={p}; 1,500 paths of 40,000 bets")
    print("| c | alpha | simulated | alpha^(2/c-1) |")
    print("|---|---|---|---|")
    for c in (0.5, 1.0):
        paths = simulate_binary(p, b, c * f_full, 40_000, 1500, seed=3)
        for alpha in (0.5, 0.25):
            print(f"| {c} | {alpha} | {(paths.min(axis=1) <= alpha).mean():.3f} | {drawdown_probability(alpha, c):.3f} |")


def estimation_table() -> dict[int, dict[float, float]]:
    p_true = 0.53
    cs = np.round(np.arange(0.1, 1.51, 0.1), 2)
    grid: dict[int, dict[float, float]] = {}
    print(f"\nTable 3: sizing from an estimated edge (true p={p_true}, b=1, true f*={kelly_binary(p_true, B):.2f})")
    print("| bets observed | growth full Kelly (x1e-4) | growth half Kelly (x1e-4) | best c | P(g<0) full | P(g<0) half |")
    print("|---|---|---|---|---|---|")
    for n in (50, 100, 500, 2000):
        grid[n] = {float(c): estimated_kelly_growth(p_true, B, n, float(c), seed=11)[0] for c in cs}
        best = max(grid[n], key=grid[n].get)
        gf, nf = estimated_kelly_growth(p_true, B, n, 1.0, seed=11)
        gh, nh = estimated_kelly_growth(p_true, B, n, 0.5, seed=11)
        print(f"| {n} | {gf * 1e4:.2f} | {gh * 1e4:.2f} | {best:.1f} | {nf:.3f} | {nh:.3f} |")
    return grid


def figure(grid: dict[int, dict[float, float]]) -> None:
    f_star = kelly_binary(P, B)
    x, q = np.array([B, -1.0]), np.array([P, 1 - P])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cs = np.linspace(0, 2.2, 400)
    ax[0].plot(cs, log_growth(cs * f_star, x, q) / log_growth(f_star, x, q))
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].axvline(1, color="gray", ls="--", lw=0.8)
    ax[0].set(xlabel="fraction of Kelly c", ylabel="growth / optimal growth", title="Growth is flat near Kelly, then falls")
    for c in (0.5, 1.0, 2.0):
        paths = simulate_binary(P, B, c * f_star, 1000, 5000, seed=7)
        lo, med, hi = np.percentile(paths, [10, 50, 90], axis=0)
        line, = ax[1].plot(med, label=f"{c:g}x Kelly")
        ax[1].fill_between(range(len(med)), lo, hi, color=line.get_color(), alpha=0.12)
    ax[1].set(yscale="log", ylim=(1e-2, 1e5), xlabel="bet", ylabel="wealth (median, 10-90%)", title="Wealth paths")
    ax[1].legend()
    for n, curve in grid.items():
        ks = np.array(list(curve))
        ax[2].plot(ks, np.array(list(curve.values())) * 1e4, label=f"{n} bets")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set(xlabel="fraction of estimated Kelly c", ylabel="true growth per bet (x1e-4)", title="Estimated edge: the best c is below 1")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)


if __name__ == "__main__":
    sizing_table()
    drawdown_table()
    figure(estimation_table())
