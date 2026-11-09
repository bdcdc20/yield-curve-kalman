# yield-curve-kalman

Forecasting government yield curves as a **state estimation problem**. The
model treats the curve the way a navigation filter treats an aircraft: you
never observe the true state (the curve's level, slope and curvature), only
noisy measurements of it (yields at each maturity). A Kalman filter fuses
those measurements with macro data into an estimate of the hidden factors,
and updates it as each new observation arrives.

**Research question:** does macro information (real activity, inflation, the
policy rate) improve out-of-sample yield forecasts beyond what the curve
already contains, in USD, EUR and GBP?

## Approach

1. **Dynamic Nelson–Siegel** (Diebold & Li, 2006). The yield at maturity τ is
   a fixed function of three latent factors:
   `y(τ) = L + S·(1−e^{−λτ})/(λτ) + C·[(1−e^{−λτ})/(λτ) − e^{−λτ}] + ε`
2. **State space with macro** (Diebold, Rudebusch & Aruoba, 2006). The factors and
   the three macro variables follow a joint VAR(1). The filter estimates it
   by maximum likelihood, and it handles missing maturities natively.
3. **Out-of-sample evaluation.** Forecasts are made on a rolling basis at 1-, 6- and
   12-month horizons, using real-time macro vintages (ALFRED) so the backtest
   never uses data that hadn't been published yet. They are compared with a
   random walk, using the Clark & West (2007) test because the benchmark is
   nested in the model.

## Status

- [x] Data pipeline: curves and macro for USD / EUR / GBP from free public sources
- [ ] Yields-only DNS model, two-step and one-step (Kalman) estimation
- [ ] Yields + macro model (DRA)
- [ ] Rolling forecasts, random-walk benchmark, Clark–West tests
- [ ] Cross-currency comparison and write-up

## Data

All sources are public. See [`config/data_sources.yaml`](config/data_sources.yaml).

| | Curve | Real activity | Inflation | Policy rate |
|---|---|---|---|---|
| USD | Fed, Gürkaynak–Sack–Wright Svensson parameters (1961–) | Capacity utilisation (FRED `TCU`, with ALFRED vintages) | PCE price index, y/y (FRED `PCEPI`, with vintages) | Fed funds (FRED `FEDFUNDS`) |
| EUR | ECB AAA government Svensson parameters (2004–) | Industrial production, y/y (ECB) | HICP annual rate (ECB) | Deposit facility rate (ECB) |
| GBP | BoE nominal government liability curve (1979–) | Index of production, y/y (ONS) | CPI annual rate (ONS) | Bank Rate (BoE) |

Curves are sampled at month-end on the 17 maturities used by DRA (3m–10y).
Svensson curves are evaluated from their published parameters. BoE yields are
taken only at maturities the BoE actually publishes. **No interpolation:** a
missing point stays missing, and the Kalman filter deals with it instead of
being handed made-up observations.

EUR and GBP macro data have no free vintage history, so they use the latest
vintage with a conservative publication lag (`release_lag_months` in the config).

## Usage

```bash
pip install -e ".[dev]"
echo "FRED_API_KEY=your_key" > .env   # free key: https://fredaccount.stlouisfed.org/apikeys
python -m yck fetch                   # all currencies
python -m yck fetch --only USD        # one currency
pytest                                # offline parser tests
```

Outputs go to `data/processed/` (`curves_{CCY}.parquet`, `macro_{CCY}.parquet`,
`vintages_USD.parquet`) along with `coverage.md`, which lists date ranges, gaps
and cross-checks (e.g. ECB curves rebuilt from parameters vs the ECB's own
published yields). The **fetch data** GitHub Action runs the same pipeline
monthly and attaches the outputs as an artifact.

## References

- Diebold & Li (2006), *Forecasting the term structure of government bond yields*, J. Econometrics
- Diebold, Rudebusch & Aruoba (2006), *The macroeconomy and the yield curve*, J. Econometrics
- Diebold & Rudebusch (2013), *Yield Curve Modeling and Forecasting*, Princeton
- Gürkaynak, Sack & Wright (2007), *The U.S. Treasury yield curve: 1961 to the present*, J. Monetary Economics
- Clark & West (2007), *Approximately normal tests for equal predictive accuracy in nested models*, J. Econometrics
- Durbin & Koopman (2012), *Time Series Analysis by State Space Methods*, OUP
