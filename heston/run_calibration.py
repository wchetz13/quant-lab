"""Reproduce the numbers and figure in the README: python run_calibration.py"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibration import PARAM_NAMES, calibrate, kappa_profile, model_ivs, synthetic_surface
from heston import HestonParams

TRUE = HestonParams(v0=0.04, kappa=1.5, theta=0.06, xi=0.6, rho=-0.7)
OUT = Path(__file__).parent / "calibration_profile.png"


def main() -> None:
    clean = synthetic_surface(TRUE)
    noisy = synthetic_surface(TRUE, noise=0.002, seed=7)  # 0.2 vol points of quote noise

    print("| | " + " | ".join(PARAM_NAMES) + " | RMSE (vol pts) |")
    print("|---|" + "---|" * 6)
    t = [getattr(TRUE, n) for n in PARAM_NAMES]
    print("| true | " + " | ".join(f"{v:.4f}" for v in t) + " | |")
    fits = {}
    for label, surf in (("clean", clean), ("noisy", noisy)):
        res = calibrate(surf, n_starts=6, seed=0)
        fits[label] = res
        print(f"| {label} fit | " + " | ".join(f"{v:.4f}" for v in res.vector) + f" | {res.rmse_vol_pts:.4f} |")

    res = fits["noisy"]
    print(f"\nnoisy fit: max |error| {res.max_abs_vol_pts:.3f} vol pts, cond(J) {res.condition_number():.0f}")
    corr = res.parameter_correlation()
    print("correlation(kappa, theta) =", f"{corr[1, 2]:.2f}", " correlation(kappa, xi) =", f"{corr[1, 3]:.2f}")
    print("kappa*theta: true", TRUE.kappa * TRUE.theta, "fit", res.params.kappa * res.params.theta)

    kappas = [0.5, 1.0, 1.5, 2.5, 4.0, 6.0]
    prof = kappa_profile(noisy, kappas)
    print("\n| kappa | theta | kappa*theta | xi | RMSE (vol pts) |\n|---|---|---|---|---|")
    for k, r in zip(kappas, prof):
        print(f"| {k:.1f} | {r.params.theta:.4f} | {k * r.params.theta:.4f} | {r.params.xi:.3f} | {r.rmse_vol_pts:.4f} |")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for t_ in sorted(set(noisy.T)):
        m = noisy.T == t_
        line, = ax[0].plot(np.log(noisy.K[m] / (noisy.S * np.exp((noisy.r - noisy.q) * t_))), noisy.iv[m] * 100, "o", ms=4)
        fit = model_ivs(res.vector, noisy)[m] * 100
        ax[0].plot(np.log(noisy.K[m] / (noisy.S * np.exp((noisy.r - noisy.q) * t_))), fit, "-", color=line.get_color(), label=f"T={t_:g}")
    ax[0].set(xlabel="log-moneyness", ylabel="implied vol (%)", title="Noisy quotes (dots) vs fit (lines)")
    ax[0].legend()
    ax[1].plot(kappas, [r.rmse_vol_pts for r in prof], "o-")
    ax[1].set(xlabel="kappa (pinned)", ylabel="RMSE (vol pts)", title="Profile RMSE: minimum near the true kappa")
    ax[2].plot(kappas, [r.params.theta for r in prof], "o-", label="theta")
    ax[2].plot(kappas, [k * r.params.theta for k, r in zip(kappas, prof)], "s--", label="kappa*theta")
    ax[2].set(xlabel="kappa (pinned)", title="Refit theta falls as kappa rises")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)


if __name__ == "__main__":
    main()
