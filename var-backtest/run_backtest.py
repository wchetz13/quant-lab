"""Reproduce the README tables and figure: python run_backtest.py"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import var_backtest as vb

N, WINDOW, ALPHA, SEED = 4000, 500, 0.01, 3


def main() -> None:
    r = vb.simulate_garch(N, seed=SEED)
    realised = r[WINDOW:]
    res = vb.backtest_all(r, WINDOW, ALPHA)
    print(f"{len(realised)} out-of-sample days, expected violations {ALPHA * len(realised):.0f}\n")
    print("| Model | Violations | Rate | Kupiec p | Independence p | Cond. coverage p | ES Z2 |")
    print("|---|---|---|---|---|---|---|")
    for name, m in res.items():
        print(
            f"| {name} | {m['violations']} | {m['rate']:.2%} | {m['kupiec_p']:.3f} | "
            f"{m['independence_p']:.3f} | {m['cc_p']:.3f} | {m['es_z2']:+.2f} |"
        )

    print("\nBasel zones, worst rolling 250-day window:")
    for name, fn in vb.FORECASTERS.items():
        var, _ = fn(r, WINDOW, ALPHA)
        h = vb.violations(realised, var).astype(int)
        worst = max(np.convolve(h, np.ones(250, dtype=int), "valid"))
        print(f"  {name}: {worst} violations -> {vb.basel_zone(int(worst), ALPHA, 250)}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    t = np.arange(len(realised))
    for ax, name in zip(axes, ["Gaussian", "Filtered HS"]):
        var, _ = vb.FORECASTERS[name](r, WINDOW, ALPHA)
        hit = vb.violations(realised, var)
        ax.plot(t, realised, lw=0.5, color="0.55", label="return")
        ax.plot(t, -var, lw=1.2, color="C3", label="-VaR (99%)")
        ax.scatter(t[hit], realised[hit], s=14, color="k", zorder=3, label=f"violation ({hit.sum()})")
        ax.set_title(name)
        ax.legend(loc="lower left", fontsize=8)
    axes[1].set_xlabel("out-of-sample day")
    fig.tight_layout()
    fig.savefig("var_backtest.png", dpi=130)


if __name__ == "__main__":
    main()
