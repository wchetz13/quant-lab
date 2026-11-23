# Pairs trading with cointegration, and the pair-search trap

Two prices are **cointegrated** when, although each wanders like a random walk, some combination
$s_t = y_t - \beta x_t$ is stationary. The spread then mean-reverts, and you can trade it:
sell it when it is rich, buy it when it is cheap. This is the core of classic statistical
arbitrage.

The catch is how pairs are found. Desks screen hundreds of candidates, and a 5% test passes 5%
of completely unrelated pairs by construction. This project builds the strategy and then
measures what that screening does.

## What's here

`pairs.py`:

- **`fit_pair`:** Engle–Granger. OLS of $y$ on $x$ gives the hedge ratio, then a unit-root
  test runs on the residual spread.
  - It uses `statsmodels.coint`, which applies **MacKinnon critical values**. Plain ADF tables
    would reject far too often, because OLS picks the $\beta$ whose residual looks most
    stationary.
- **`half_life`:** the Ornstein–Uhlenbeck half-life $\ln 2 / -\ln(1 + b)$, from the regression
  $\Delta s_t = a + b\, s_{t-1}$.
- **`positions` / `backtest`:** a z-score rule. Enter at $|z| > 2$ and exit at $|z| < 0.5$.
  - The hedge ratio, mean and volatility are fitted on a formation window, then **frozen** for
    the trading window.
  - The position at $t$ uses only data up to $t$, and a test checks this by perturbing the
    future.
- **`simulate_pair`:** either a truly cointegrated pair, where the spread is an AR(1) with
  $\phi = 0.95$ (half-life of 13.5 days), or two independent random walks.

## Results

Each pair has 1,000 days: a 500-day formation window, then a 500-day trading window.
Out-of-sample means the trading window. Sharpe ratios are annualised (± 1 s.e.).

| | 200 truly cointegrated pairs | Spurious pairs (from 3,000 independent random walks) |
|---|---|---|
| Pass the cointegration test (p < 0.05) | 84% | **6.3%** (189 pairs) |
| In-sample Sharpe of those that pass | **1.83 ± 0.03** | **1.83 ± 0.03** |
| Out-of-sample Sharpe | 1.57 ± 0.03 | 0.10 ± 0.06 |
| Share profitable out of sample | 100% | 58% |

**In-sample backtests can't tell them apart.** Spurious pairs that pass the screen look exactly
as good as real ones, with an in-sample Sharpe of 1.83 in both cases. That is the trap:

- Out of sample, real pairs keep most of their edge.
- Spurious pairs collapse to a Sharpe that is statistically zero, and they are profitable only
  58% of the time.

Zero is the correct answer for spurious pairs. Their spread increments are independent of the
past, so no position rule can have positive expected P&L.

Two smaller findings:

- **False positives:** the test passes 6.3% of independent random walks, slightly above its
  nominal 5% size. That is the usual distortion from choosing the ADF lag length by AIC.
- **Half-life bias:** on a 500-day formation window, the median estimated half-life is 10.7
  days against a true 13.5. AR(1) coefficients are biased low in short samples (Kendall's
  bias), which overstates how fast a spread mean-reverts. The tests confirm the half-life
  converges on a long sample.

**Practical takeaways:**

- Treat a pair screen as a multiple-testing problem: correct the threshold, or demand an
  economic link between the two assets.
- Always judge a pair on data the screen didn't see.

## Tests

`test_pairs.py` checks:

- **Estimation:** the hedge ratio is recovered, the half-life matches the AR coefficient on a
  long sample, and it is infinite for a random walk.
- **The cointegration test:** it has power on true pairs and a false-positive rate near its
  size.
- **Profitability:** true pairs earn a Sharpe above 1 out of sample. Spurious pairs look good in
  sample and earn nothing out of sample.
- **No look-ahead:** positions don't change when future data is altered.
- **Trading rules:** the entry and exit rules are checked on a hand-built z-score path.

## References

- Engle, R., Granger, C. (1987). Co-integration and error correction. *Econometrica* 55(2).
- MacKinnon, J. (2010). Critical values for cointegration tests. Queen's Economics Department Working Paper 1227.
- Gatev, E., Goetzmann, W., Rouwenhorst, K. (2006). Pairs trading: performance of a relative-value arbitrage rule. *Review of Financial Studies* 19(3).
