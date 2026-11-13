"""Dupire local volatility extracted from an SVI implied volatility surface.

Dupire (1994): there is exactly one diffusion dS = (r - q) S dt + sigma_loc(S, t) S dW whose
European prices match a given arbitrage-free surface. In total implied variance w(k, T), with
k = ln(K / F_T) (Gatheral, The Volatility Surface, eq. 1.10):

    sigma_loc^2(k, T) = (dw/dT) / g(k, T)

where g is Durrleman's butterfly function from svi/. Local variance therefore exists exactly when
the surface is free of calendar (dw/dT >= 0) and butterfly (g > 0) arbitrage.
"""

from __future__ import annotations

import numpy as np

from svi import SVI, butterfly_arbitrage_free, calendar_arbitrage_free


class LocalVolSurface:
    """SVI slices interpolated linearly in total variance at fixed log-moneyness.

    Linear interpolation of w in T preserves the no-calendar-arbitrage ordering of the slices,
    and before the first expiry w scales with T, so implied vol is flat in time there.
    """

    def __init__(self, slices, check_arbitrage=True):
        self.slices = sorted(slices, key=lambda ts: ts[0])
        self.T = np.array([t for t, _ in self.slices])
        if self.T[0] <= 0 or np.any(np.diff(self.T) <= 0):
            raise ValueError("expiries must be positive and distinct")
        if check_arbitrage:
            bad = [t for t, s in self.slices if not butterfly_arbitrage_free(s)]
            if bad:
                raise ValueError(f"butterfly arbitrage in slices at T = {bad}")
            if not calendar_arbitrage_free(self.slices):
                raise ValueError("calendar arbitrage: total variance decreases with maturity")

    def _bracket(self, T):
        i = int(np.clip(np.searchsorted(self.T, T), 1, len(self.T) - 1))
        return i - 1, i

    def _interp(self, attr, k, T):
        """Interpolate w (or a k-derivative of w) and return it with dw/dT."""
        k = np.asarray(k, float)
        if T <= self.T[0]:
            s = self.slices[0][1]
            scale = T / self.T[0]
            return getattr(s, attr)(k) * scale, s.w(k) / self.T[0]
        i, j = self._bracket(T)
        (t0, s0), (t1, s1) = self.slices[i], self.slices[j]
        u = (min(T, t1) - t0) / (t1 - t0)
        value = (1 - u) * getattr(s0, attr)(k) + u * getattr(s1, attr)(k)
        return value, (s1.w(k) - s0.w(k)) / (t1 - t0)

    def total_variance(self, k, T):
        return self._interp("w", k, T)[0]

    def implied_vol(self, k, T):
        return np.sqrt(self.total_variance(k, T) / T)

    def local_variance(self, k, T):
        w, dw_dT = self._interp("w", k, T)
        dw = self._interp("dw", k, T)[0]
        d2w = self._interp("d2w", k, T)[0]
        k = np.asarray(k, float)
        g = (1 - k * dw / (2 * w)) ** 2 - dw**2 / 4 * (1 / w + 0.25) + d2w / 2
        return dw_dT / g

    def local_vol(self, k, T):
        return np.sqrt(self.local_variance(k, T))


def local_variance_unchecked(slice_: SVI, k, T):
    """Local variance for a slice whose smile is held fixed in vol as T varies (w = v(k) T).

    No arbitrage check: used to show what happens on a slice that has butterfly arbitrage.
    """
    k = np.asarray(k, float)
    w, dw, d2w = slice_.w(k) * T, slice_.dw(k) * T, slice_.d2w(k) * T
    g = (1 - k * dw / (2 * w)) ** 2 - dw**2 / 4 * (1 / w + 0.25) + d2w / 2
    return slice_.w(k) / g


def simulate(surface: LocalVolSurface, T, n_steps, n_paths, seed=None):
    """Terminal log-moneyness x_T = ln(S_T / F_T) under local vol, by log-Euler.

    Working relative to the forward removes rates and dividends: dx = -sigma^2/2 dt + sigma dW.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    x = np.zeros(n_paths)
    for n in range(n_steps):
        t = max(n * dt, 1e-4)
        var = np.maximum(surface.local_variance(x, t), 0.0)
        x += -0.5 * var * dt + np.sqrt(var * dt) * rng.standard_normal(n_paths)
    return x
