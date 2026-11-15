# Roadmap

Running backlog for quant-lab. Issue numbers refer to this repository. Pending PRs are noted so work is not duplicated.

## Done (on main)

- [x] **heston**: Fourier pricing, QE Monte Carlo, skew
- [x] **american-options**: Longstaff–Schwartz, Bermudan tree (#1)
- [x] **svi**: smile calibration, static arbitrage checks (#4)

## In review

- [ ] **heston calibration** to an implied vol surface (#3): `heston/calibration.py`, multi-start bounded least squares, kappa/theta identifiability
- [ ] **sabr**: Hagan expansion, Monte Carlo, calibration (#5, PR #8)
- [ ] **dupire**: local vol from an SVI surface (#7, PR #9)

## Next

1. Heston calibration to a real (downloaded) option chain, once network access allows; compare with the synthetic study
2. Binomial/trinomial trees with Richardson extrapolation
3. VaR/CVaR with Kupiec and Christoffersen backtests
4. Yield curve bootstrapping and Nelson–Siegel–Svensson
5. Portfolio optimisation: mean-variance frontier, risk parity, Black–Litterman
6. GARCH volatility forecasting; pairs trading with cointegration; Kelly sizing; Fama–French regressions
7. Showcase finished projects on willhetzinger.github.io and the profile README
