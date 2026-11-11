# SABR and Hagan's implied volatility formula

SABR is the market-standard smile model for interest rate options (swaptions, caps and floors):

$$
dF_t = \alpha_t F_t^{\beta}\,dW_t, \qquad d\alpha_t = \nu\,\alpha_t\,dZ_t, \qquad d\langle W, Z\rangle_t = \rho\,dt
$$

Each parameter has a distinct job:

- $\alpha$ sets the vol level.
- $\beta$ sets the **backbone**, meaning how ATM vol moves when the forward moves.
- $\rho$ sets the skew.
- $\nu$ (vol of vol) sets the smile's curvature.

Its popularity comes from Hagan et al. (2002), who derived a closed-form expansion for the Black
implied volatility. That makes calibration instant. This project implements the formula, checks
it against Monte Carlo, and shows where it breaks.

## What's here

`sabr.py`:

- **`hagan_vol`:** Hagan's eq. 2.17a. The $z/x(z)$ term is 0/0 at the money, so it switches to
  its Taylor series there to avoid cancellation.
- **`simulate`:** Monte Carlo of the full model.
  - Volatility is stepped exactly, since it is lognormal.
  - The forward uses log-Euler for $\beta = 1$, and Euler with absorption at zero otherwise.
- **`implied_density`:** the risk-neutral density of $F_T$ implied by Hagan vols, via
  Breeden–Litzenberger ($\partial^2 C/\partial K^2$).
- **`calibrate`:** fits $\alpha, \rho, \nu$ with $\beta$ fixed, as desks do. On a single smile
  $\beta$ and $\rho$ both move the skew and are nearly collinear, so $\beta$ is chosen from the
  backbone you want rather than fitted. The fit is unconstrained: it optimises over
  $\log\alpha$, $\operatorname{artanh}\rho$ and $\log\nu$.

## Results

### Accuracy vs. Monte Carlo

Errors are Hagan vol minus Monte Carlo implied vol, in vol points, for
$\alpha = 0.25$, $\beta = 1$, $\rho = -0.3$, $\nu = 0.4$, $F = 100$. Each Monte Carlo run used
400,000 paths and priced the out-of-the-money side.

| Expiry | K = 80 | K = 90 | K = 100 | K = 110 | K = 120 |
|---|---|---|---|---|---|
| 3 months | +0.03 | +0.03 | 0.00 | −0.01 | −0.03 |
| 1 year | +0.11 | +0.08 | +0.03 | +0.01 | 0.00 |
| 5 years | +1.04 | +0.88 | +0.81 | +0.72 | +0.67 |

The expansion is an asymptotic series in $\nu^2 T$. At short expiries it is within a few
hundredths of a vol point. By 5 years it is off by about a full vol point, worst on the
low-strike wing.

### Arbitrage at long expiries

The rates parameters are $\alpha = 0.03$, $\beta = 0.5$, $\rho = -0.3$, $\nu = 0.5$, with the
forward at 3%. The density implied by Hagan vols is:

| Expiry | Minimum density | Negative for strikes |
|---|---|---|
| 1 year | > 0 | none |
| 5 years | −27.5 | below 0.40% |
| 10 years | −171.2 | below 1.14% |

A negative density means a butterfly spread with a negative price, which is free money. This is
the well-known failure of Hagan's formula at low rates and long expiries. It motivated the
arbitrage-free SABR of Hagan et al. (2014) and the shifted-SABR conventions desks adopted when
rates went near zero. The `svi/` project in this repo detects the same failure in implied-vol
smiles, using Durrleman's condition.

## Tests

`test_sabr.py` checks:

- **Limits:** the formula is continuous through the money, and with $\beta = 1$, $\nu = 0$ it
  reduces exactly to Black–Scholes.
- **Backbone:** ATM vol scales as $F^{\beta - 1}$.
- **Skew:** negative $\rho$ produces a downward skew.
- **Accuracy:** within 0.15 vol points of Monte Carlo at 3 months, and measurably biased at
  5 years.
- **Arbitrage:** the density is positive at 1 year and negative at low strikes at 10 years.
- **Calibration:** exact parameter recovery, and invalid parameters are rejected.

## References

- Hagan, P., Kumar, D., Lesniewski, A., Woodward, D. (2002). Managing smile risk. *Wilmott Magazine*.
- Hagan, P., Kumar, D., Lesniewski, A., Woodward, D. (2014). Arbitrage-free SABR. *Wilmott Magazine*.
- Breeden, D., Litzenberger, R. (1978). Prices of state-contingent claims implicit in option prices. *Journal of Business* 51(4).
