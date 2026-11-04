# SVI volatility smiles and static arbitrage

Market quotes give implied volatility at a handful of strikes. Pricing anything else, such as
an off-grid strike, a local volatility model or a variance swap, needs a smooth smile. That
smile also has to be **arbitrage-free**, or it implies negative probabilities.

Gatheral's raw SVI ("stochastic volatility inspired") models total implied variance
$w = \sigma_{BS}^2 T$ as a function of log-moneyness $k = \ln(K/F)$:

$$
w(k) = a + b\left(\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right)
$$

Each of the five parameters has a shape meaning:

- $a$ sets the level.
- $b$ sets the wing slopes.
- $\rho$ sets the skew.
- $m$ shifts the smile left or right.
- $\sigma$ sets how curved the at-the-money region is.

The wings are linear in $k$, which Lee's moment formula requires, and every derivative is in
closed form.

## Calibration: quasi-explicit SVI

Fitting all five parameters at once is a non-convex problem with poor local minima.
`fit` follows De Marco and Martini (Zeliade, 2009) instead. With $y = (k - m)/\sigma$, the
model for fixed $(m, \sigma)$ is

$$
w = a + d\,y + c\sqrt{y^2 + 1}, \qquad c = b\sigma,\; d = \rho b \sigma
$$

That is **linear** in $(a, c, d)$. So the optimiser searches only over $(m, \sigma)$, using
Nelder–Mead from a small grid of starting points, and solves the other three parameters exactly
by bounded linear least squares at each step. A final joint least-squares polish then refines
all five raw parameters.

## No-arbitrage checks

**Butterfly arbitrage.** Call prices must be convex in strike. In total-variance terms this is
Durrleman's condition:

$$
g(k) = \left(1 - \frac{k w'}{2w}\right)^2 - \frac{w'^2}{4}\left(\frac1w + \frac14\right) + \frac{w''}{2} \;\ge\; 0
$$

$g$ is not just a test. The implied risk-neutral density of $\ln(S_T/F)$ is

$$
p(k) = \frac{g(k)}{\sqrt{2\pi w(k)}}\, e^{-d_-(k)^2/2}, \qquad d_-(k) = -\frac{k}{\sqrt{w}} - \frac{\sqrt{w}}{2}
$$

So butterfly arbitrage is literally a negative probability.

**Calendar arbitrage.** Total variance must not decrease with maturity at any $k$.

**Lee's bound.** The wing slopes must satisfy $b(1 + |\rho|) \le 2$.

## Results

**Calibration.** The test smile is $a = 0.04$, $b = 0.4$, $\rho = -0.4$, $m = 0.05$,
$\sigma = 0.25$, at $T = 0.5$. It is sampled at 23 strikes, with Gaussian noise of 20 basis
points added to each implied vol.

| | $a$ | $b$ | $\rho$ | $m$ | $\sigma$ |
|---|---|---|---|---|---|
| True | 0.0400 | 0.4000 | −0.4000 | 0.0500 | 0.2500 |
| Fitted to noisy quotes | 0.0512 | 0.3882 | −0.4201 | 0.0467 | 0.2267 |

- **Fit quality:** the RMSE to the quotes is 13 bp, below the 20 bp noise, and the fitted smile
  is within 26 bp of the true one everywhere.
- **Clean data:** on noiseless quotes, all five parameters come back to $10^{-8}$.
- **Speed:** a fit takes about 70 ms.

The individual parameters move more than the smile does, because $a$ and $\sigma$ partly trade
off against each other. Market data hits the same identifiability issue.

**Arbitrage.** Vogt's slice ($a = -0.041$, $b = 0.1331$, $\rho = 0.306$, $m = 0.3586$,
$\sigma = 0.4153$) is the standard counterexample. Its smile looks perfectly reasonable, yet:

| Slice | $\min_k g(k)$ | Where | Density integrates to | Butterfly-free |
|---|---|---|---|---|
| Test smile above | 0.197 | $k = -1.26$ | 1 | yes |
| Vogt (Gatheral & Jacquier, 2014) | −0.033 | $k = 0.88$ | 1 | **no** |

Vogt's implied density still integrates to one. It goes negative for $k$ between about 0.64 and
1.26, which a check on total mass alone would never catch.

## Tests

`test_svi.py` checks:

- **Closed forms:** the derivatives match finite differences, and the minimum variance is exact.
- **Calibration:** exact recovery from clean data, and noise-level accuracy from noisy quotes.
- **Arbitrage-free density:** the equity slice's density is non-negative, integrates to 1, and
  has mean $E[S_T/F] = 1$.
- **Detection:** the Vogt arbitrage is found in the right place, and a calendar-spread crossing
  is caught.
- **Validation:** parameters that give negative variance, or have $b < 0$, $|\rho| \ge 1$ or
  $\sigma \le 0$, are rejected.

## References

- Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives. *Global Derivatives & Risk Management*.
- Gatheral, J., Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance* 14(1).
- De Marco, S., Martini, C. (2009). Quasi-explicit calibration of Gatheral's SVI model. *Zeliade Systems white paper*.
- Lee, R. (2004). The moment formula for implied volatility at extreme strikes. *Mathematical Finance* 14(3).
