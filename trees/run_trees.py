"""Reproduce the README tables and figure: convergence, oscillation and American put benchmarks."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trees import (bbs_price, bbsr_price, binomial_greeks, binomial_price, black_scholes, trinomial_price)

OUT = Path(__file__).parent
EU = dict(S=100.0, K=105.0, T=1.0, r=0.05, q=0.02, sigma=0.25)
AM = dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2)


def pricers(kind="put", american=False):
    """Name -> function(params, n) with an odd step count where Leisen-Reimer requires it."""
    odd = lambda n: n if n % 2 else n + 1
    return {
        "CRR": lambda a, n: binomial_price(n=n, kind=kind, american=american, method="crr", **a),
        "Jarrow-Rudd": lambda a, n: binomial_price(n=n, kind=kind, american=american, method="jr", **a),
        "Leisen-Reimer": lambda a, n: binomial_price(n=odd(n), kind=kind, american=american, method="lr", **a),
        "Trinomial": lambda a, n: trinomial_price(n=n, kind=kind, american=american, **a),
        "BBS": lambda a, n: bbs_price(n=n, kind=kind, american=american, **a),
        "BBSR": lambda a, n: bbsr_price(n=n, kind=kind, american=american, **a),
    }


def convergence_table(steps=(10, 25, 50, 100, 200, 400)):
    bs = black_scholes(kind="put", **EU)
    print(f"European put, Black-Scholes = {bs:.6f}\n")
    print("| Method | " + " | ".join(f"n={n}" for n in steps) + " |")
    print("|---|" + "---:|" * len(steps))
    errs = {}
    for name, f in pricers().items():
        errs[name] = [abs(f(EU, n) - bs) for n in steps]
        print(f"| {name} | " + " | ".join(f"{e:.1e}" for e in errs[name]) + " |")
    return errs, steps


def sawtooth(n_range=range(20, 121)):
    """Error against n for the ATM-ish strike shows the odd/even oscillation of CRR."""
    bs = black_scholes(kind="put", **EU)
    crr = np.array([binomial_price(n=n, kind="put", method="crr", **EU) - bs for n in n_range])
    bbs = np.array([bbs_price(n=n, kind="put", **EU) - bs for n in n_range])
    return np.array(list(n_range)), crr, bbs


def american_table():
    ref = bbsr_price(n=4000, kind="put", american=True, **AM)
    print(f"\nAmerican put S=K=100, r=5%, sigma=20%, T=1; reference (BBSR n=4000) = {ref:.5f}\n")
    print("| Method | n=50 | n=200 | n=800 | time n=800 (ms) |")
    print("|---|---:|---:|---:|---:|")
    for name, f in pricers(american=True).items():
        vals = [f(AM, n) for n in (50, 200, 800)]
        t0 = time.perf_counter()
        f(AM, 800)
        ms = 1e3 * (time.perf_counter() - t0)
        print(f"| {name} | " + " | ".join(f"{v:.5f}" for v in vals) + f" | {ms:.0f} |")
    return ref


def premium_and_greeks():
    print("\nEarly-exercise premium and tree Greeks (CRR, n=1000), put, K=100, r=5%, sigma=20%, T=1\n")
    print("| Spot | European | American | Premium | Delta | Gamma | Theta |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for s in (80, 90, 100, 110, 120):
        a = dict(AM, S=float(s))
        eu = binomial_price(n=1000, kind="put", **a)
        g = binomial_greeks(n=1000, kind="put", american=True, **a)
        print(f"| {s} | {eu:.4f} | {g['price']:.4f} | {g['price'] - eu:.4f} | {g['delta']:.4f} | {g['gamma']:.4f} | {g['theta']:.4f} |")


def figure(errs, steps, sw):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, e in errs.items():
        ax[0].loglog(steps, e, marker="o", label=name)
    ax[0].set(xlabel="steps n", ylabel="|price - Black-Scholes|", title="European put: absolute error")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    n, crr, bbs = sw
    ax[1].plot(n, crr, label="CRR")
    ax[1].plot(n, bbs, label="BBS")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(xlabel="steps n", ylabel="price - Black-Scholes", title="CRR oscillates; BBS is smooth")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=150)


if __name__ == "__main__":
    errs, steps = convergence_table()
    american_table()
    premium_and_greeks()
    figure(errs, steps, sawtooth())
