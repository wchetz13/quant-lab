# Roadmap

Running backlog for quant-lab. Issue numbers refer to this repository. Pending PRs are noted so work is not duplicated.

## Done (on main)

- [x] **heston**: Fourier pricing, QE Monte Carlo, skew
- [x] **american-options**: Longstaff–Schwartz, Bermudan tree (#1)
- [x] **svi**: smile calibration, static arbitrage checks (#4)
- [x] **sabr**: Hagan expansion, Monte Carlo, calibration (#5, PR #8)
- [x] **dupire**: local vol from an SVI surface (#7, PR #9)
- [x] **heston calibration** to an implied vol surface (#3, PR #10): multi-start bounded least squares, kappa/theta identifiability
- [x] **var-backtest**: VaR/ES forecasters with Kupiec, Christoffersen and Basel backtests (PR #11)
- [x] **yield-curve**: bootstrapping deposits and par swaps, Nelson–Siegel–Svensson (PR #12)
- [x] **portfolio-opt**: mean-variance frontier, risk parity, Black–Litterman, estimation-error study (in review)
- [x] **purged-cv**: purged k-fold with embargo and a leakage study (#14)

## Next

1. Heston calibration to a real (downloaded) option chain, once network access allows; compare with the synthetic study
2. yield-curve: monotone-convex interpolation, multi-curve (OIS discounting + projection), swaption/bond pricing on the curve, real Treasury data
3. Binomial/trinomial trees with Richardson extrapolation
4. portfolio-opt follow-ups: walk-forward backtest on real ETF returns, turnover/transaction-cost constraints, resampled efficiency, hierarchical risk parity
5. GARCH volatility forecasting; pairs trading with cointegration; Kelly sizing; Fama–French regressions
6. Showcase finished projects on willhetzinger.github.io (the profile README is live; keep its featured table current)
