# Dupire local volatility from an SVI surface

`svi/` fits arbitrage-free implied vol smiles, but a smile is a quote, not a model. To price
path-dependent products consistently with those quotes, you need dynamics. Dupire (1994) showed
there is exactly one diffusion

$$
dS_t = (r - q) S_t\,dt + \sigma_{loc}(S_t, t)\,S_t\,dW_t
$$

that reproduces every European price on an arbitrage-free surface. In total implied variance
$w(k, T)$, with $k = \ln(K/F_T)$ (Gatheral, *The Volatility Surface*, eq. 1.10):

$$
\sigma_{loc}^2(k, T) = \frac{\partial w / \partial T}{g(k, T)}
$$

The denominator $g$ is **Durrleman's function from `svi/`**, and the numerator is the
**calendar condition**. So local variance exists exactly when the surface is free of butterfly
and calendar arbitrage. The arbitrage checks from the previous project are what make this one
well-posed.

## What's here

`dupire.py`:

- **`LocalVolSurface`:** SVI slices with $w$ interpolated linearly in $T$ at fixed $k$. That
  preserves the slices' calendar ordering. Before the first expiry, implied vol is held flat in
  time.
  - The constructor **refuses** any surface with butterfly or calendar arbitrage.
  - Derivatives in $k$ come from SVI's closed forms, so the only numerical differentiation is
    in $T$.
- **`simulate`:** Monte Carlo under local vol, working in log-moneyness relative to the
  forward ($dx = -\tfrac12\sigma_{loc}^2\,dt + \sigma_{loc}\,dW$). That removes rates and
  dividends from the problem.

## Results

### Round trip

The surface has SVI slices at 3, 6, 12 and 24 months, with an equity-style skew. Monte Carlo
under the extracted local vol used 400 steps, 200,000 paths, seed 1, and a control variate on
$S_T/F_T$, whose mean is exactly 1. The 1-year results, with prices as a fraction of the
forward:

| $k$ | Implied vol | Local vol | Black price from surface | Local-vol MC (± 1 s.e.) |
|---|---|---|---|---|
| −0.3 | 32.2% | 44.0% | 0.28527 | 0.28522 ± 0.00015 |
| −0.1 | 24.8% | 23.7% | 0.14914 | 0.14924 ± 0.00019 |
| 0.0 | 21.6% | 16.2% | 0.08586 | 0.08607 ± 0.00020 |
| +0.1 | 20.1% | 15.6% | 0.04176 | 0.04202 ± 0.00018 |
| +0.2 | 20.2% | 18.4% | 0.01899 | 0.01923 ± 0.00015 |

The local-vol model reprices the input surface. Local vol is far more skewed than implied vol:
it is 44% at $k = -0.3$, where implied vol is 32%. Implied vol is roughly an average of local
vol along the paths to a strike, so local vol has to swing harder to produce the same smile.

### Gatheral's rule of thumb

For a short-dated smile held fixed in vol across expiries, the at-the-money local skew is
**1.998×** the implied skew. Theory says 2 as $T \to 0$.

### Discretisation bias

The log-Euler scheme has first-order weak error. The bias is the average error in the ATM
1-year call, in basis points of the forward, over 4 seeds × 200,000 paths:

| Steps | 25 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|
| Bias (bp of F) | 25.7 | 12.1 | 7.2 | 3.6 | 1.9 |

The bias halves every time the step count doubles, as first-order theory predicts.

### Arbitrage means no local vol

On Vogt's arbitrageable slice (see `svi/`), local variance is **negative** for $k$ between about
0.64 and 1.26. That is exactly where Durrleman's $g < 0$. No real volatility produces those
prices, which is why `LocalVolSurface` rejects such inputs.

## Tests

`test_dupire.py` checks:

- **Flat smile:** a flat smile gives constant local vol, and interpolation reproduces each slice
  exactly.
- **Skew rule:** the ATM local skew is twice the implied skew, to within 0.01.
- **Round trip:** local-vol Monte Carlo reprices the surface within 4 standard errors across
  strikes.
- **Euler bias:** the bias shrinks with the time step.
- **Arbitrage:** Vogt's slice gives negative local variance in the right place, and surfaces
  with butterfly or calendar arbitrage are rejected.

## References

- Dupire, B. (1994). Pricing with a smile. *Risk* 7(1).
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley, ch. 1–3.
- Gatheral, J., Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance* 14(1).
