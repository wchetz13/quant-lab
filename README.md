# quant-lab

Self-contained quantitative finance projects. Each folder has a module, a README that explains the
math and shows results, and a test suite that checks the implementation against analytic limits,
independent methods, and no-arbitrage conditions.

| Project | What it covers |
|---|---|
| [heston](heston/) | Heston stochastic volatility: Fourier pricing with the "little trap" characteristic function, Andersen QE Monte Carlo, the implied volatility skew, and calibration to an implied vol surface with identifiability diagnostics |
| [american-options](american-options/) | Longstaff–Schwartz regression Monte Carlo for American and Bermudan options, benchmarked against a Bermudan binomial tree and the published values from the original paper |
| [svi](svi/) | Gatheral's SVI smile with quasi-explicit calibration, Durrleman's butterfly condition, calendar-spread checks, and the implied risk-neutral density |
| [sabr](sabr/) | SABR with Hagan's implied vol expansion, checked against Monte Carlo, with calibration and the negative-density failure at long expiries |
| [dupire](dupire/) | Dupire local volatility extracted from an arbitrage-free SVI surface, with a local-vol Monte Carlo that reprices the input smile |
| [var-backtest](var-backtest/) | VaR and Expected Shortfall from historical, Gaussian, EWMA and filtered historical simulation, scored with Kupiec, Christoffersen, Basel traffic-light and Acerbi–Székely tests on GARCH data |
| [yield-curve](yield-curve/) | Discount curve bootstrapping from deposits and par swaps, Nelson–Siegel–Svensson fitting by variable projection, and forward curves |
| [purged-cv](purged-cv/) | Machine learning for finance: purged k-fold cross-validation with embargo, and a leakage study where shuffled k-fold finds AUC 0.70 in pure noise |
| [portfolio-opt](portfolio-opt/) | Mean-variance frontier (closed form and long-only), risk parity via Spinu's convex formulation, Black–Litterman, and a Monte Carlo study of estimation error with Ledoit–Wolf shrinkage |

## Running the tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

CI runs the suite on Python 3.11, 3.12 and 3.13.
