# American options: Longstaff–Schwartz Monte Carlo

The holder of an American option can exercise at any time, so pricing one is an optimal stopping problem:

$$
V_0 = \sup_{\tau} \; \mathbb{E}\left[e^{-r\tau}\,h(S_\tau)\right]
$$

At each date the holder should exercise when the payoff $h(S_t)$ exceeds the **continuation
value**, which is the expected discounted value of waiting. Trees and PDE grids compute that
expectation directly, but they scale badly with the number of underlyings. Longstaff and
Schwartz (2001) estimate it from simulation instead.

## Method

`lsm.py`:

1. Simulates GBM paths at the exercise dates, using antithetic pairs.
2. Steps backward in time. At each date it regresses realised discounted cash flows on weighted
   Laguerre polynomials of $S_t/K$.
   - It fits only in-the-money paths. Those are the only paths where the exercise decision
     matters, and fitting there puts the basis where the exercise boundary is.
3. Prices on a **fresh** set of paths, following the fitted exercise rule.

Step 3 matters because the two choices bias the estimate in opposite directions:

- **Reusing the fitting paths** lets the regression see the future it is graded on, which
  biases the price upward.
- **Fresh paths** make the estimate a genuine **lower bound**, because any estimated policy is
  at best optimal and usually slightly worse.

`binomial_price` is the benchmark: a vectorised Cox–Ross–Rubinstein tree. It prices American
(exercise at every node), European, or Bermudan contracts. Bermudan means exercise only on $k$
fixed dates, which is the contract LSM actually prices.

## Results

The setup is a put with $K = 40$ and $r = 6\%$, exercisable 50 times a year. The trees use
2,000 steps per year. LSM uses 100,000 paths, a cubic Laguerre basis (four terms, as in the paper) and seed 1.

| $S_0$ | $\sigma$ | $T$ | European | Bermudan tree | Published (L&S Table 1) | LSM (± 1 s.e.) | American tree |
|---|---|---|---|---|---|---|---|
| 36 | 20% | 1 | 3.844 | 4.478 | 4.478 | 4.479 ± 0.006 | 4.487 |
| 36 | 40% | 1 | 6.711 | 7.101 | 7.101 | 7.107 ± 0.008 | 7.109 |
| 40 | 20% | 2 | 2.356 | 2.885 | 2.885 | 2.861 ± 0.007 | 2.890 |
| 44 | 20% | 1 | 1.017 | 1.110 | 1.110 | 1.108 ± 0.006 | 1.113 |

- **Benchmark check:** the Bermudan tree reproduces the paper's finite-difference values to
  three decimals.
- **Exercise premium:** the right to exercise early is worth up to 0.63, or about 16% of the
  European price.
- **Bermudan vs. American:** restricting exercise to 50 dates a year costs less than a cent.

### Where the lower bound bites

The two-year case comes in low because a cubic basis can't trace the exercise boundary well
over a long horizon. The estimate gets tighter as the basis grows and with more fitting paths.
Over 100 exercise dates against the Bermudan tree value of 2.885, the results are:

| Laguerre terms | 100k paths | 400k paths |
|---|---|---|
| 4 (cubic, default) | 2.861 (−3.6 s.e.) | 2.878 (−2.1 s.e.) |
| 5 | 2.864 (−3.1 s.e.) | 2.881 (−1.1 s.e.) |
| 6 | 2.867 (−2.8 s.e.) | 2.882 (−0.6 s.e.) |

## Tests

`test_lsm.py` checks:

- **Published benchmarks:** the Bermudan tree matches all four of the paper's values.
- **Lower bound:** LSM never beats the Bermudan value beyond noise, and lands within four cents
  of it.
- **Black–Scholes convergence:** the European tree converges to Black–Scholes.
- **No-arbitrage results:**
  - An American put is worth more than a European put and at least its intrinsic value.
  - Merton's theorem: an American call on a non-dividend stock is never exercised early, so
    it equals the European call exactly. With a dividend yield, early exercise becomes
    valuable.
  - A deep in-the-money put is exercised immediately.

## References

- Longstaff, F., Schwartz, E. (2001). Valuing American options by simulation: a simple least-squares approach. *Review of Financial Studies* 14(1).
- Cox, J., Ross, S., Rubinstein, M. (1979). Option pricing: a simplified approach. *Journal of Financial Economics* 7(3).
- Merton, R. (1973). Theory of rational option pricing. *Bell Journal of Economics and Management Science* 4(1).
