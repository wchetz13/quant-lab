"""Reproduce the README tables and figure. All market data are synthetic, generated from a known NSS curve."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from yield_curve import Deposit, DiscountCurve, Swap, bootstrap, fit_nelson_siegel, fit_nss, nss_zero_rate, par_swap_rate

TRUE = (0.045, -0.02, -0.015, 0.02, 1.8, 9.0)
SWAP_YEARS = (1, 2, 3, 5, 7, 10, 15, 20, 30)
NOISE_BP = 1.0


def true_curve() -> DiscountCurve:
    t = np.arange(1, 31, dtype=float)
    return DiscountCurve(t, np.exp(-nss_zero_rate(t, *TRUE) * t))


def main() -> None:
    ref = true_curve()
    t_dep = np.array([0.25, 0.5])
    z_dep = nss_zero_rate(t_dep, *TRUE)
    deposits = [Deposit(float(m), float((np.exp(z * m) - 1) / m)) for m, z in zip(t_dep, z_dep)]
    swaps = [Swap(n, par_swap_rate(ref, n)) for n in SWAP_YEARS]
    curve = bootstrap(deposits + swaps)

    grid = np.linspace(0.5, 30, 60)
    truth = nss_zero_rate(grid, *TRUE)
    boot_err = (curve.zero_rate(grid) - truth) * 1e4
    print("Bootstrap (2 deposits + 9 par swaps, noiseless)")
    print(f"  max |zero error| = {np.abs(boot_err).max():.2f} bp, rms = {np.sqrt(np.mean(boot_err**2)):.2f} bp")

    rng = np.random.default_rng(7)
    obs_t = np.linspace(0.5, 30, 40)
    obs = nss_zero_rate(obs_t, *TRUE) + rng.normal(0, NOISE_BP * 1e-4, obs_t.size)
    ns, ns_rmse = fit_nelson_siegel(obs_t, obs)
    nss, nss_rmse = fit_nss(obs_t, obs)
    print(f"\nFits to {obs_t.size} noisy zero rates ({NOISE_BP:.0f} bp noise)")
    print("model | in-sample rmse (bp) | error vs truth on 0.5-30y: rms (bp), max (bp) | 40y error (bp)")
    for name, p, rmse in (("NS", ns, ns_rmse), ("NSS", nss, nss_rmse)):
        e = (nss_zero_rate(grid, *p) - truth) * 1e4
        e40 = (nss_zero_rate(40.0, *p) - nss_zero_rate(40.0, *TRUE)) * 1e4
        print(f"{name:4} | {rmse * 1e4:5.2f} | {np.sqrt(np.mean(e**2)):5.2f}, {np.abs(e).max():5.2f} | {e40:6.2f}")
    print(f"NSS params: beta={np.round(nss[:4], 4)}, tau1={nss[4]:.2f}, tau2={nss[5]:.2f}")
    print(f"NS  params: beta={np.round(ns[:3], 4)}, tau1={ns[4]:.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(grid, truth * 100, "k-", label="true NSS curve")
    ax[0].plot(grid, curve.zero_rate(grid) * 100, "--", label="bootstrapped (11 instruments)")
    ax[0].plot(obs_t, obs * 100, ".", ms=4, alpha=0.6, label="noisy zero rates")
    ax[0].plot(grid, nss_zero_rate(grid, *ns) * 100, label="Nelson-Siegel fit")
    ax[0].plot(grid, nss_zero_rate(grid, *nss) * 100, label="NSS fit")
    ax[0].set(xlabel="maturity (years)", ylabel="zero rate (%)", title="Zero curve")
    ax[0].legend(fontsize=8)
    ax[1].step(grid, curve.instantaneous_forward(grid) * 100, where="post", label="bootstrapped forward")
    f_nss = nss_zero_rate(grid, *nss) + grid * (nss_zero_rate(grid + 1e-4, *nss) - nss_zero_rate(grid - 1e-4, *nss)) / 2e-4
    ax[1].plot(grid, f_nss * 100, label="NSS forward")
    ax[1].set(xlabel="maturity (years)", ylabel="instantaneous forward (%)", title="Forward curve")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("yield_curve.png", dpi=130)


if __name__ == "__main__":
    main()
