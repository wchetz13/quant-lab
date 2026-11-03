# Heston stochastic volatility

Black–Scholes holds volatility constant, so it can't produce the volatility smile seen in every
equity options market. Heston (1993) makes variance a mean-reverting square-root process that is
correlated with the stock:

$$
dS_t = (r - q) S_t\,dt + \sqrt{v_t}\,S_t\,dW^S_t, \qquad
dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW^v_t, \qquad
d\langle W^S, W^v\rangle_t = \rho\,dt
$$

This folder prices European options two independent ways and checks that they agree.

## Fourier pricing

The characteristic function of $\ln S_T$ is known in closed form. Prices follow from Gil-Pelaez inversion:

$$
C = S e^{-qT} P_1 - K e^{-rT} P_2, \qquad
P_j = \frac12 + \frac1\pi \int_0^\infty \operatorname{Re}\!\left[\frac{e^{-iu\ln K}\,\varphi_j(u)}{iu}\right] du
$$

`characteristic_function` uses the "little Heston trap" form of Albrecher et al. (2007). Heston's
original form puts a complex logarithm across its branch cut once maturities get long, and the
prices it returns jump. A test prices maturities from 1 to 30 years and checks that the curve
stays smooth.

## Monte Carlo: Andersen's QE scheme

`simulate_qe` implements Andersen's (2008) quadratic-exponential scheme. Each step matches the
exact conditional mean and variance of $v_{t+\Delta}$:

- **Low dispersion** ($\psi \le 1.5$): the scheme draws a scaled non-central chi-square, $a(b+Z)^2$.
- **High dispersion**: it draws from a point mass at zero mixed with an exponential tail.

The variance can't go negative, so the scheme needs no truncation fix. That matters when the
Feller condition $2\kappa\theta \ge \xi^2$ fails, which is the usual case for calibrated equity
parameters. Euler schemes are badly biased in that regime.

## Results

The parameters are $v_0 = 0.04$, $\kappa = 1.5$, $\theta = 0.06$, $\xi = 0.6$, $\rho = -0.7$ (Feller
violated), with $S = 100$, $T = 1$, $r = 3\%$, $q = 1\%$. The QE run used 400,000 paths, 100 steps and seed 1:

| Strike | Fourier | QE Monte Carlo (± 1 s.e.) | Black–Scholes implied vol |
|---|---|---|---|
| 70 | 32.1074 | 32.0953 ± 0.0278 | 28.7% |
| 85 | 19.3569 | 19.3473 ± 0.0234 | 24.2% |
| 100 | 8.8192 | 8.8206 ± 0.0168 | 20.0% |
| 115 | 2.3863 | 2.3890 ± 0.0092 | 16.6% |
| 130 | 0.4009 | 0.4002 ± 0.0041 | 15.2% |

Both methods agree to within one standard error at every strike. Negative spot–vol correlation
produces the familiar equity skew: implied vol falls from 28.7% to 15.2% across strikes.

## Tests

`test_heston.py` checks:

- **Martingale property:** $\varphi(0) = 1$ and $\varphi(-i)$ equals the forward.
- **Black–Scholes limit:** with $\xi \to 0$, the model matches Black–Scholes at constant variance. When $v_0 \ne \theta$, it matches Black–Scholes at the integrated variance.
- **No static arbitrage:** call prices are decreasing and convex in strike and stay inside the model-free bounds.
- **Skew direction, put–call parity and long-maturity smoothness.**
- **Fourier vs. QE:** the two prices agree within 4 standard errors when the Feller condition fails, and simulated paths stay finite and positive under extreme vol-of-vol.

## References

- Heston, S. (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies* 6(2).
- Albrecher, H., Mayer, P., Schoutens, W., Tistaert, J. (2007). The little Heston trap. *Wilmott Magazine*.
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. *Journal of Computational Finance* 11(3).
