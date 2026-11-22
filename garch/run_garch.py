"""Out-of-sample volatility forecasting study on simulated GJR-GARCH data.

Reproduces every table and the figure in the README: ``python run_garch.py``.
Data are synthetic (true model known), so the study shows what each forecaster can and cannot
recover rather than claiming anything about a real market.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from garch import (
    diebold_mariano, ewma_forecasts, fit_garch, garch_rolling_forecasts, mincer_zarnowitz,
    mse_loss, qlike_loss, rolling_variance_forecasts, simulate_gjr,
)

TRUE = dict(omega=0.02, alpha=0.03, gamma=0.10, beta=0.88, nu=7.0)
N, START = 4500, 1500


def main() -> None:
    t0 = time.time()
    r, s2_true = simulate_gjr(N, seed=11, **TRUE)

    print("## In-sample fits (first 3000 obs)")
    print("| Model | omega | alpha | gamma | beta | nu | persistence | loglik | BIC |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, lev, dist in [("GARCH-N", False, "normal"), ("GARCH-t", False, "t"),
                            ("GJR-N", True, "normal"), ("GJR-t", True, "t")]:
        f = fit_garch(r[:3000], leverage=lev, dist=dist)
        nu = f"{f.nu:.2f}" if f.nu else "-"
        print(f"| {name} | {f.omega:.4f} | {f.alpha:.3f} | {f.gamma:.3f} | {f.beta:.3f} | {nu} "
              f"| {f.persistence:.3f} | {f.loglik:.1f} | {f.bic:.1f} |")
    print(f"| truth | {TRUE['omega']} | {TRUE['alpha']} | {TRUE['gamma']} | {TRUE['beta']} | "
          f"{TRUE['nu']} | {TRUE['alpha'] + TRUE['gamma'] / 2 + TRUE['beta']:.3f} | | |")

    fc = {
        "Rolling 250d": rolling_variance_forecasts(r, 250),
        "Rolling 60d": rolling_variance_forecasts(r, 60),
        "EWMA 0.94": ewma_forecasts(r),
        "GARCH-N": garch_rolling_forecasts(r, START, leverage=False),
        "GJR-t": garch_rolling_forecasts(r, START, leverage=True, dist="t"),
        "True variance": s2_true,
    }
    sl = slice(START, N)
    proxy = r[sl] ** 2
    q = {k: qlike_loss(v[sl], proxy) for k, v in fc.items()}
    m = {k: mse_loss(v[sl], proxy) for k, v in fc.items()}

    print(f"\n## Out-of-sample one-day variance forecasts ({N - START} days, refit every 21d)")
    print("| Forecaster | QLIKE | MSE | MZ slope | MZ R^2 | DM vs GJR-t (QLIKE) p |")
    print("|---|---|---|---|---|---|")
    for k, v in fc.items():
        _, b, r2 = mincer_zarnowitz(v[sl], proxy)
        p = "-" if k in ("GJR-t",) else f"{diebold_mariano(q[k], q['GJR-t'])[1]:.3f}"
        print(f"| {k} | {q[k].mean():.4f} | {m[k].mean():.2f} | {b:.2f} | {r2:.3f} | {p} |")

    final = fit_garch(r[:START], leverage=True, dist="t")
    h = np.arange(1, 251)
    print("\n## Term structure from the last in-sample day (annualised vol, %)")
    for hh in (1, 10, 60, 250):
        print(f"h={hh}: {np.sqrt(252 * final.forecast(hh)[-1]):.2f}")
    print(f"long run: {np.sqrt(252 * final.long_run_variance):.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    days = np.arange(START, N)
    ax[0].plot(days, np.sqrt(252 * s2_true[sl]), color="0.2", lw=1.2, label="True")
    ax[0].plot(days, np.sqrt(252 * fc["GJR-t"][sl]), color="C3", lw=1, label="GJR-t forecast")
    ax[0].plot(days, np.sqrt(252 * fc["EWMA 0.94"][sl]), color="C0", lw=0.8, alpha=0.7, label="EWMA")
    ax[0].set(title="Out-of-sample volatility (annualised %)", xlabel="day")
    ax[0].legend(frameon=False)
    ax[1].plot(h, np.sqrt(252 * final.forecast(250)), color="C3")
    ax[1].axhline(np.sqrt(252 * final.long_run_variance), color="0.4", ls="--", lw=1,
                  label="long-run")
    ax[1].set(title="GJR-t variance forecast term structure", xlabel="horizon (days)",
              ylabel="annualised vol %")
    ax[1].legend(frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("garch_forecasts.png"), dpi=150)
    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
