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

Contents:

1. [Quick start](#1-quick-start)
2. [Project status](#2-project-status)
3. [Repository layout](#3-repository-layout)
4. [Data pipeline](#4-data-pipeline)
5. [Data findings and quirks](#5-data-findings-and-quirks-read-before-modelling)
6. [Model specification](#6-model-specification)
7. [Forecast evaluation protocol](#7-forecast-evaluation-protocol)
8. [Roadmap](#8-roadmap-next-steps-in-order)
9. [Conventions and decisions](#9-conventions-and-decisions-already-made)
10. [GitHub Actions and tooling](#10-github-actions-and-tooling)
11. [Troubleshooting](#11-troubleshooting)
12. [References](#12-references)

---

## 1. Quick start

Requirements: Python ≥ 3.10 and git. Run all commands from the repository
root (the folder containing this README).

```bash
# 1. Install the package in editable mode, with test dependencies
python3 -m pip install -e ".[dev]"            # add ",notebook" for matplotlib + jupyter

# 2. Create the local secrets file (git ignores it, so it is never committed).
#    Replace YOUR_KEY with the FRED API key from
#    https://fredaccount.stlouisfed.org/apikeys
echo "FRED_API_KEY=YOUR_KEY" > .env

# 3. Run the offline tests (no network needed)
python3 -m pytest -q

# 4. Download all public data and build the monthly panels (~2-5 minutes)
python3 -m yck fetch

# 5. Read the coverage report
cat data/processed/coverage.md
```

Useful variants:

```bash
python3 -m yck fetch --only USD          # one currency (USD, EUR, GBP; several allowed)
python3 -m yck fetch --refresh           # ignore the cache in data/raw and re-download
python3 -m yck probe-ecb ICP             # list ECB dataflows whose id/name matches "ICP"
python3 -m yck probe-ecb HICP "?"        # show the key dimensions of the HICP dataflow
python3 -m yck probe-ecb HICP "M...000000..ANR"   # list matching series and their last date
python3 -m yck backtest --ccy USD        # Kalman DNS out-of-sample forecasts -> data/results/ (cached, parallel)
```

Notes:

- `.env` is read automatically by `python -m yck` (see `src/yck/__main__.py`).
  Only `FRED_API_KEY` is needed. Without it, everything still runs except the
  ALFRED real-time vintages for USD, which are reported as FAILED in
  `coverage.md`.
- `data/raw/` (downloads) and `data/processed/` (outputs) are git-ignored and
  rebuilt by `python3 -m yck fetch`.
- If the repository lives inside OneDrive, OneDrive syncing the hidden `.git`
  folder can occasionally cause git "lock"/"file exists" errors. Pausing
  OneDrive sync while running git commands avoids this.

## 2. Project status

As of 11 Nov 2026. All Kalman results use the long-run mean μ fixed at the
real-time sample mean (§6.3).

| Stage | Status |
|---|---|
| Data pipeline (curves + macro, USD/EUR/GBP) | **Done.** Verified end-to-end on GitHub Actions; every source fetches and is current |
| Offline tests | **Done.** 140 tests (parsers, real-time panels, DNS, backtest, Kalman filter vs statsmodels and brute force, steady-state shortcuts vs the full recursion, EM, exact score, stationarity repair, shrinkage, DM/CW, model confidence set, impulse responses vs statsmodels, shifting endpoints and AR(1) errors without look-ahead), all passing (`pytest -q`) |
| CI (tests on every push) and monthly data-fetch workflow | **Done** |
| Two-step Diebold–Li baseline (yields only) | **Done** (`src/yck/model/dns.py`). USD 1985–2000: factor/proxy correlations 0.97–0.999, fit RMSE 2–8 bp (3m: 8 bp) |
| USD two-step notebook (factor plots, first RW comparison) | **Done** (`notebooks/01_two_step_usd.ipynb`); first results below |
| Kalman filter + RTS smoother with missing observations | **Done** (`src/yck/model/kalman.py`). Matches statsmodels to ~1e-9 (log-likelihood, filtered/smoothed states and covariances), with gaps; 0.16–0.36 ms per filter + smoother pass on USD 1985– (§8, step 4d) |
| One-step maximum-likelihood DNS (yields only) | **Done** (`statespace_dns.py`, `notebooks/02_kalman_usd.ipynb`). EM + exact score, 1 bp noise floor, multistart, μ fixed at the sample mean, stationary estimates; every backtest in the project (`backtest --all`) ~1.5 min from scratch, seconds from cache |
| Forward-rate benchmark, Diebold–Mariano and Clark–West tests | **Done** (`eval/backtest.py`, `eval/stats.py`); USD results in §7.4. Optional SPF survey benchmark not done |
| Yields + macro model (Diebold–Rudebusch–Aruoba 2006), USD | **Done** (`statespace_dns.py`, `data/realtime.py`, `model/irf.py`, `notebooks/03_macro_usd.ipynb`): real-time macro, full / policy-rate-only / shrunk models, impulse responses with bands, variance decompositions, out-of-sample tests. Results in §7.4 |
| EUR and GBP, cross-currency comparison | **Done** (`notebooks/04_eur_gbp.ipynb`); results in §7.4 and below |
| Accuracy improvements from the literature | **Items 1–2 done** (`model/adjust.py`, `notebooks/05_accuracy.ipynb`, results in §7.7): AR(1) measurement errors fix the 1-month forecasts; shifting endpoints do not help on balance. Items 3–8 not started (§8 step 7b; next: forecast combination, forward anchoring) |
| Trading evaluation | Not started (§7.5; after §8 step 7b) |
| Final write-up | Not started (§8, step 9) |

Latest verified coverage (month-end, output of `python -m yck fetch`):

| | Curve | Real activity | Inflation | Policy rate |
|---|---|---|---|---|
| **USD** | 1961-06 → 2026-10, no gaps | 1967-01 → 2026-09 | 1960-01 → 2026-09 | 1954-07 → 2026-10 |
| **EUR** | 2004-09 → 2026-10, no gaps | 1992-01 → 2026-08 | 1997-01 → 2026-10 | 1999-01 → 2026-10 |
| **GBP** | 1979-01 → 2026-10 (short-end gaps, see §5) | 1949-01 → 2026-08 | 1989-01 → 2026-09 | 1975-01 → 2026-10 |

USD ALFRED vintages: capacity utilisation 386 vintages (first 1996-11-15);
PCE price index 317 vintages (first 2000-08-01).

**First results (two-step DNS, USD, `notebooks/01_two_step_usd.ipynb`).**
Estimation from 1985, out-of-sample origins 2000-01 → 2026-10, re-estimated
monthly. RMSE relative to the random walk (< 1 = better than RW):

| | 3m | 12m | 24m | 60m | 120m |
|---|---|---|---|---|---|
| AR(1), h = 6 | 1.09 | 1.14 | 1.16 | 1.20 | 1.08 |
| VAR(1), h = 6 | **0.84** | **0.95** | 1.01 | 1.08 | 1.03 |
| AR(1), h = 12 | 1.06 | 1.10 | 1.15 | 1.26 | 1.17 |
| VAR(1), h = 12 | **0.88** | **0.96** | 1.02 | 1.12 | 1.07 |

The AR(1) baseline loses to the random walk after 2000. The VAR(1) wins
only at the short end. The main cause is bias: forecasts revert to a high
historical mean, so they ran too high in 2000–2015. For example, the mean
12-month-ahead error at 10y was +58 bp in 2000–07 and +80 bp in 2008–15
for the AR(1), against +20 / +31 bp for the RW. A rolling 10y window does
not fix this. Can macro information correct it? That is the question for
§6.4.

**One-step Kalman / ML DNS (`notebooks/02_kalman_usd.ipynb`).** In-sample
log-likelihood 13,284 (λ fixed) and 15,394 (λ estimated, 0.039), with a
1 bp noise floor, against 11,168 at the two-step estimates. Out of sample
(refit yearly, filtered states), h = 12:

| | 3m | 12m | 24m | 60m | 120m |
|---|---|---|---|---|---|
| Two-step VAR(1) | **0.88** | **0.96** | 1.02 | 1.12 | 1.07 |
| Kalman, λ fixed | 0.89 | 0.98 | 1.05 | 1.15 | 1.10 |
| Kalman, λ estimated | 0.92 | 0.99 | 1.04 | 1.12 | 1.12 |

Much better in-sample fit, no better forecasts: the Kalman model matches the
two-step VAR(1) but does not beat it. The reason is the same mean-reversion
bias. μ is fixed at the 1985-to-date mean of the level factor (5.4–7.9%
across refits), while the 10y yield was mostly 1–5% after 2000 (10y, h = 12,
2000–07: +51 bp, against +47 bp for the two-step VAR and +20 bp for the
random walk). With μ *estimated*, as in an earlier version, the bias was
larger (+61 bp) and every ratio 0.02–0.04 worse; estimated μ is barely
identified (§6.3).

**Yields + macro and the cross-currency comparison (`notebooks/03_macro_usd.ipynb`,
`notebooks/04_eur_gbp.ipynb`, details in §7.4).**

- **USD:** only the policy rate adds information. The 4-state model beats
  yields-only by Clark–West at every maturity at h = 12 (RMSE −1% to −3%)
  and is level with forwards (RMSE ratio 0.95–1.01, not significant). The
  full model with capacity utilisation and inflation overfits.
- **EUR:** every model loses to the random walk (Kalman models by 9–63%)
  and significantly to forwards, because forecasts revert to a 2004-onward
  mean the zero/negative-rate decade never returned to (10y, h = 12,
  2010–13: +92 bp too high).
- **GBP:** close to the random walk and forwards (ratio to forwards
  0.93–1.16), never significantly better; macro does not help.
- **In sample, the curve leads policy in all three currencies** (a
  curvature shock raises the policy rate by 20–30 bp at a 15–20 month peak),
  while macro shocks explain only 4–10% of the factors' 12-month forecast
  variance.
- **No model beats the forward curve significantly in any currency at 6 or
  12 months.**

**Accuracy improvements, step 7b items 1–2 (`notebooks/05_accuracy.ipynb`,
details in §7.7).** The grid of 54 variants was fixed before it was run, and
a model confidence set covers every model in the project.

- **AR(1) measurement errors** cut 1-month RMSE significantly at both ends
  of the curve in every currency. Examples: USD 3m −11 to −18%, 10y
  −11 to −13%; EUR 3m −20 to −33%; GBP 10y −16%. At h = 1 the 90% model
  confidence set keeps every AR(1) variant and drops the plain λ-fixed
  Kalman models. USD 10y against forwards goes from 1.13*** to 1.00. No gain at
  h = 12: the residuals' half-lives are 1–6 months.
- **Shifting endpoints** remove most of the rally bias (10y, h = 12,
  2014–21: USD +45 → +2 bp, EUR +78 → +38 bp), but the trailing endpoint
  overshoots when rates turn (2022–26: USD −56 → −117 bp). Net: no
  significant RMSE change. SPF survey endpoints make USD worse, and the
  ECB long-term HICP expectation leaves the EUR 10y bias unchanged.
- **Still nothing beats forwards or the random walk at h = 6–12**, and the
  confidence set cannot separate the models there.

## 3. Repository layout

```
config/data_sources.yaml    every series ID, URL and setting; fix broken IDs here, not in code
src/yck/
  __main__.py               CLI: `python -m yck fetch | probe-ecb | backtest`; loads .env
  data/http.py              requests session, retries, on-disk cache in data/raw
  data/sources.py           one parse_* (pure, unit-tested) + fetch_* per source; ECB probes; ALFRED as_of()
  data/curves.py            Svensson/Nelson-Siegel evaluation, month-end sampling
  data/build.py             orchestrates everything and writes data/processed + coverage.md
  data/realtime.py          real-time macro panels: ALFRED vintages (USD) or release lags, as known at each origin
  model/dns.py              Nelson-Siegel loadings, two-step Diebold-Li estimation, factor dynamics, forecasts
  model/kalman.py           Kalman filter (missing rows), log-likelihood, RTS smoother, smoother sufficient statistics (compiled)
  model/statespace_dns.py   ML DNS (yields only or yields + macro): transforms, EM, exact score, multistart L-BFGS-B, 1 bp floor, shrinkage prior
  model/irf.py              impulse responses (Cholesky, generalised), variance decompositions, bands from the observed information
  model/adjust.py           forecast adjustments outside the likelihood: shifting endpoints (EWMA, surveys), AR(1) measurement errors
  eval/backtest.py          recursive out-of-sample forecasts (no look-ahead), parallel cached refits, random walk, forwards, RMSE tables
  eval/stats.py             Diebold-Mariano (HLN) and Clark-West tests, model confidence set, ratio/significance tables
  eval/grid.py              the standard and step 7b grids of backtests as one job (CLI `backtest --all | --accuracy`, notebooks' `ensure_results`)
notebooks/                  analysis notebooks, committed with outputs
tests/                      offline tests: parsers (small samples in each source's real format), real-time panels, DNS, Kalman, backtest, stats, IRFs
.github/workflows/
  ci.yml                    pytest on every push / PR
  fetch-data.yml            full fetch on a GitHub runner (manual + monthly cron); coverage report in the log
  probe-ecb.yml             runs `yck probe-ecb` on a runner (for finding ECB keys)
data/                       raw/ and processed/ (git-ignored)
```

Further model and evaluation modules are planned in §8.

## 4. Data pipeline

### 4.1 Outputs (`data/processed/`)

| File | Content |
|---|---|
| `curves_{USD,EUR,GBP}.parquet` | Month-end zero-coupon yields in **percent, continuously compounded** (BoE: see 4.2). Index `date` = calendar month-end. Columns `3m, 6m, 9m, 12m, 15m, 18m, 21m, 24m, 30m, 36m, 48m, 60m, 72m, 84m, 96m, 108m, 120m` |
| `curves_ext_{USD,EUR,GBP}.parquet` | The same month-end curves at every maturity the forward-rate benchmark needs (§7.4): the grid, each horizon h in `forecast_horizons_months`, and each grid τ + h (1m–132m). GBP: only maturities the BoE publishes (61/73/85/97/109/121m are not, so 1-month forwards for τ ≥ 60m are missing) |
| `macro_{USD,EUR,GBP}.parquet` | Monthly `real_activity`, `inflation`, `policy_rate`, **latest vintage**, index = month-end. Transforms already applied (see table below) |
| `surveys_{USD,EUR}.parquet` | Quarterly survey expectations indexed by the first day of the survey quarter: USD `bond10`, `cpi10`; EUR `hicp_lt`. **Not yet aligned to publication dates**: use `yck.model.adjust.survey_endpoint`, which applies `survey_known_month` |
| `vintages_USD.parquet` | ALFRED real-time history, long format: `series_id, date, realtime_start, realtime_end, value` (`realtime_end` of a still-current vintage is set to 2262-04-11) |
| `coverage.md` | First/last date and in-range gaps for every series, plus cross-checks |

Load example:

```python
import pandas as pd
y = pd.read_parquet("data/processed/curves_USD.parquet")      # T x 17
m = pd.read_parquet("data/processed/macro_USD.parquet")       # T x 3
v = pd.read_parquet("data/processed/vintages_USD.parquet")
from yck.data.sources import as_of
tcu_as_seen_on = as_of(v[v.series_id == "TCU"], "2008-10-31")  # series as published that day
```

### 4.2 Sources

All free and public; exact IDs live in `config/data_sources.yaml`.

**Curves.** All sampled at the last available trading day of each month. The
unfinished current month is dropped.

| | Source | How the grid is built |
|---|---|---|
| USD | Fed, Gürkaynak–Sack–Wright (2007), `feds200628.csv`, daily from 1961 | Svensson curve evaluated from the published parameters `BETA0–3, TAU1, TAU2` at the grid maturities. Before 1980 GSW is plain Nelson–Siegel (BETA3/TAU2 missing), so the 4th term is dropped |
| EUR | ECB `YC` dataflow, AAA-rated euro-area central government, Svensson parameters `B.U2.EUR.4F.G_N_A.SV_C_YM.{BETA0..TAU2}`, daily from 2004-09-06 | Same Svensson evaluation. Cross-checked against the ECB's own published spot rates SR_1Y/5Y/10Y: max abs difference 0.003 / 0.008 / 0.006 pp |
| GBP | BoE nominal government liability curve (Anderson–Sleath spline), `glcnominalddata.zip`, daily from 1979 | Read directly from sheets "4. spot curve" (years) and "3. spot, short end" (months). Pre-2005 workbooks name them "4. nominal spot curve" / "3. nominal spot, short end"; both names are configured. **Only maturities the BoE publishes are used; no interpolation** |

**Macro.**

| | Real activity | Inflation | Policy rate |
|---|---|---|---|
| USD | FRED `TCU` capacity utilisation, level, **with ALFRED vintages** | FRED `PCEPI`, 12-month % change, **with ALFRED vintages** | FRED `FEDFUNDS`, monthly average, level (never revised) |
| EUR | ECB `STBS.M.I10.Y.PROD.NS0020.4D0.N.IX`: euro area (21) industrial production ex construction, SA + working-day adjusted, 12-month % change | ECB `HICP.M.U2.N.000000.4D0.ANR`: HICP all items, annual rate, changing composition | ECB `FM.D.U2.EUR.4F.KR.DFR.LEV`: deposit facility rate, month-end value |
| GBP | ONS `K222` (diop) index of production, 12-month % change | ONS `D7G7` (mm23) CPI annual rate | BoE IADB `IUDBEDR` Bank Rate, month-end value |

These follow DRA (2006), which uses capacity utilisation, annual PCE
inflation and the fed funds rate. EUR and GBP have no free vintage
history, so they use the latest vintage plus a conservative publication lag
(`release_lag_months` in the config: real activity 2, inflation 1, policy
rate 0). **The lag is not applied in the processed files.** It must be
applied in the forecasting code (§7.3).

**Surveys** (shifting endpoints, §7.7). Quarterly. A survey counts as known
from the end of the second month of its quarter (`survey_known_month`).

| | Series | Coverage |
|---|---|---|
| USD | Philadelphia Fed SPF `Median_BOND10_Level.xlsx`: median expected average 10y Treasury yield over the next 10 years | First-quarter surveys only, 1992– |
| USD | Philadelphia Fed SPF `Median_CPI10_Level.xlsx`: median expected average CPI inflation over the next 10 years | Every quarter, 1991Q4– |
| EUR | ECB `SPF.Q.U2.HICP.POINT.LT.Q.AVG`: longer-term (about 5 years ahead) HICP expectation, average of point forecasts | 1999– (some early years annual) |

## 5. Data findings and quirks (read before modelling)

1. **Common sample.** EUR curves start Sep 2004, which limits any
   three-currency comparison to 2004-09 onwards. USD (1961–) and GBP (1979–)
   can additionally be studied on long samples.
2. **GBP short-end gaps are genuine.** The BoE does not publish the 3m and 6m
   points on many dates (3m: 1980-08 → 2024-10 with 331 missing months in
   range; 6m: 116 missing; 9m: ~10; 12m: 1). They are left as NaN on purpose.
   The Kalman filter handles them by dropping missing rows from the
   measurement equation (§6.3). Do **not** interpolate or fill them; an
   invented observation would make the fit look better than it is. For the
   two-step baseline, fit each month's cross-section on the maturities
   available that month.
3. **USD short end vs T-bills.** GSW 3m is 0.26 pp above `DTB3` on average
   (6m: 0.17 pp above `DTB6`). Expected: bills are quoted on a discount basis
   and GSW excludes bills, so its fit below ~1y is extrapolated. Keep GSW for
   consistency, and mention this in the write-up as a robustness point.
4. **ECB dataflows changed in 2026.** The old `ICP` (HICP) and `STS`
   (industrial production) dataflows froze at Dec 2025 and Sep 2025. The
   replacements are `HICP` and `STBS`, with different key structures
   (already configured). If an ECB series stops updating again, use
   `python -m yck probe-ecb` (§1) or the "probe ECB key" workflow to find the
   new key.
5. **Month-end alignment.** Curves are the last trading day of each month.
   Macro observations are indexed to the end of their reference month, but
   are *released* later (§7.3).
6. **Units.** All yields and rates are in percent (4.25 = 4.25%).
   Maturities are in months in column names; the DNS loadings in §6 use
   months.

## 6. Model specification

Notation: $y_t(\tau)$ is the yield at maturity $\tau$ (months) at month $t$.
The grid is the 17 maturities $\tau \in \{3,\dots,120\}$.

### 6.1 Dynamic Nelson–Siegel measurement

$$
y_t(\tau) = L_t + S_t\,\frac{1-e^{-\lambda\tau}}{\lambda\tau}
+ C_t\left(\frac{1-e^{-\lambda\tau}}{\lambda\tau} - e^{-\lambda\tau}\right) + \varepsilon_t(\tau)
$$

In vector form, $y_t = \Lambda(\lambda) f_t + \varepsilon_t$ with
$f_t = (L_t, S_t, C_t)'$ and $\Lambda$ of size 17 × 3.

### 6.2 Two-step Diebold–Li (2006) baseline (do this first)

1. Fix $\lambda = 0.0609$. This is Diebold–Li's value for maturities in
   months; it puts the peak of the curvature loading at about 30 months
   (29.4 exactly).
2. For each month, regress the cross-section of yields on $\Lambda$ by OLS,
   using only the maturities observed that month, to get
   $\hat f_t = (\hat L_t, \hat S_t, \hat C_t)$.
3. Fit a VAR(1) to $\hat f_t$, or three separate AR(1)s, which is DL's
   preferred forecasting specification:
   $\hat f_{t+h} = \hat\mu + \hat A^{h}(\hat f_t - \hat\mu)$.
4. Forecast yields as $\hat y_{t+h|t} = \Lambda \hat f_{t+h|t}$.

Sanity checks: $\hat L_t$ should track the 10y yield, $-\hat S_t$ the
10y–3m spread, and $\hat C_t$ roughly $2y_{24m} - y_{3m} - y_{120m}$.
Report the fit RMSE by maturity (DL get ~5–10 bp for USD).

### 6.3 One-step state-space (Kalman filter, maximum likelihood)

State equation (VAR(1) in deviations from the mean):

$$
f_t - \mu = A (f_{t-1} - \mu) + \eta_t, \qquad \eta_t \sim N(0, Q)
$$

Measurement equation:

$$
y_t = \Lambda(\lambda) f_t + \varepsilon_t, \qquad \varepsilon_t \sim N(0, H),\ H \text{ diagonal}
$$

Implementation notes:

- **Parameters:** $A$ (9), $Q$ via its Cholesky factor (6),
  $\log \operatorname{diag}(H)$ (17), and $\lambda$ (1, estimate it as
  $\exp$ of a free parameter, or fix it at 0.0609 first).
- **Long-run mean $\mu$: fixed at the sample mean** (default since 11 Nov
  2026, `fit_dns_mle(mu="sample")`; `mu="mle"` estimates it). At each refit,
  $\mu$ is the mean of the month-by-month OLS factors and of the observed
  macro series *up to the origin*, so there is no look-ahead. Why: near a unit
  root $c = (I - A)\mu$ barely depends on $\mu$, so the ML estimate is not
  identified. On GBP 1993–2012 it was −152% for Bank Rate and −93% for the
  level, the optimiser stopped at the stationarity cap, and repeated runs
  scattered by ~260 log-likelihood points. With $\mu$ fixed every refit is
  interior (largest spectral radius in the whole grid 0.998), runs from
  different starts agree exactly, and the grid fits ~3× faster. The
  estimated μ was also behind the USD mean-reversion bias (§2: μ of 6–8%).
- **Initial state:** the unconditional mean $\mu$, with the unconditional
  covariance from the discrete Lyapunov equation
  ($P_0 = A P_0 A' + Q$, `scipy.linalg.solve_discrete_lyapunov`).
  If $A$ is not stable, the filter falls back to a diffuse prior (large
  $P_0$) so that any trial value can be evaluated. That fallback makes the
  likelihood jump *up* at the unit circle, and on short or trending samples
  the optimiser used to stop at explosive $A$ (USD shrinkage models, EUR,
  GBP). Estimates are therefore required to be stationary: the search runs
  unrestricted, and a run that ends explosive is redone with explosive values
  rejected, and on the boundary surface if the cap binds
  (`statespace_dns._fit_from`). With μ fixed (below) neither happens in the
  current grid.
- **Missing data:** at each $t$, keep only the rows of $y_t$, $\Lambda$ and
  $H$ for observed maturities; skip the update entirely if none are
  observed. This is how the GBP short-end gaps are handled.
- **Likelihood:** the prediction-error decomposition, summing
  $-\tfrac12[\log|F_t| + v_t' F_t^{-1} v_t + n_t \log 2\pi]$. Maximise with
  EM (Shumway–Stoffer), then L-BFGS-B with the exact score (Fisher's
  identity: the gradient of the expected complete-data log-likelihood, from
  one smoother pass), from the two-step estimates plus random perturbations
  (6 starts). Measurement-noise sd is bounded below by **1 bp**: without the
  floor, some GSW maturities are fitted to ~0.01 bp, and the likelihood is
  then too ill-conditioned to maximise reliably (§8, step 4b).
- Write a small, own Kalman filter (numpy) and test it against
  `statsmodels.tsa.statespace` on simulated data. The custom code is part
  of the project's value and needs to handle missing rows.
- **Smoother:** a Rauch–Tung–Striebel smoother, for plotting factors only.
  Forecasts must use **filtered** states.

### 6.4 Yields + macro (Diebold–Rudebusch–Aruoba 2006)

The state becomes $x_t = (L_t, S_t, C_t, \mathrm{CU}_t, \mathrm{INFL}_t, \mathrm{RATE}_t)'$:
the three latent factors plus real activity, inflation and the policy rate.

- **State:** $x_t - \mu = A (x_{t-1} - \mu) + \eta_t$, with $A$ 6×6 and
  $Q$ 6×6 (full, or block-restricted as in DRA).
- **Measurement:** the 17 yields load on the first three states via
  $\Lambda$. The 3 macro variables are observed **without error**: they
  load 1:1 on their own state, with zero measurement variance. In
  practice, use a tiny variance such as 1e-8 to keep $F_t$ invertible.
- **Compare:** yields-only (§6.3) against yields + macro. Report the
  estimated $A$ and impulse responses (do macro shocks move the factors,
  and the factors the macro variables?) and the variance decompositions,
  as DRA do.
- **Nesting:** the yields-only model is the yields + macro model with the
  off-diagonal blocks of $A$ (macro → factors) set to zero. That makes the
  out-of-sample comparison a **nested** one, which is why Clark–West is
  used (§7.2).

## 7. Forecast evaluation protocol

### 7.1 Design

- **Horizons** h = 1, 6, 12 months. 6 and 12 are the headline results.
- **Recursive (expanding) estimation,** re-estimated every month, or every
  12 months to save time. A rolling 10-year window is the robustness check.
- **Out-of-sample periods:** USD 2000-01 onwards (limited by ALFRED vintage
  coverage, §2). EUR and GBP from 2010-01, so there are ≥ 5 years of
  estimation data for EUR. Also report the common window from 2010-01 for
  all three.
- **Error metric:** RMSE in basis points, by maturity and horizon.

### 7.2 Benchmarks and tests

| Comparison | Test | Why |
|---|---|---|
| Any model vs **random walk** ($\hat y_{t+h} = y_t$) or **forwards** | Diebold–Mariano (1995) with the Harvey–Leybourne–Newbold (1997) small-sample correction, two-sided | RW and forwards are the standard hard-to-beat benchmarks; not nested in DNS |
| **Yields + macro vs yields-only** | **Clark–West (2007)**, one-sided | Nested models: under the null, the larger model's extra parameter noise biases plain MSE comparisons against it |

Clark–West adjusted loss differential, with model 1 the small (nested)
model and model 2 the large one:

$$
\hat f_{t+h} = e_{1,t+h}^2 - \left[e_{2,t+h}^2 - (\hat y_{1,t+h} - \hat y_{2,t+h})^2\right]
$$

Regress $\hat f$ on a constant with HAC standard errors (lag h−1, see below)
and reject the null of equal accuracy if t > 1.282 (10%) or t > 1.645 (5%).

**Variance estimator (`src/yck/eval/stats.py`).** h-step forecasts made
monthly overlap, so loss differentials are MA(h−1). The variance of their
mean uses autocovariances up to lag h−1 with **equal (rectangular) weights**,
as in Diebold & Mariano (1995), which is exact for that structure; Bartlett
is the fallback if the estimate is not positive. Newey–West's Bartlett
weights shrink those autocovariances, and in simulation (300 observations,
h = 12) a nominal 5% test with them rejected 8–11% of the time. The
rectangular estimator with the HLN correction rejects 5.5–6.5%
(`tests/test_stats.py` checks this).

### 7.3 Real-time macro (no look-ahead)

- **USD:** at forecast origin $t$ (a month-end), build each vintaged macro
  series with `as_of(vintages, t)`, which returns the series exactly as
  published on that date. The last available reference month is then
  usually t−1 or t−2. Because the state equation needs a value at every
  $t$, treat the not-yet-released months as **missing observations** in
  the measurement equation (the filter handles this). Do not forward-fill.
- **EUR / GBP:** use the latest vintage, but set the last
  `release_lag_months[var]` months before each origin to missing. This
  approximates what was known at the time; state it clearly as a
  limitation.
- `FEDFUNDS`, the ECB DFR and Bank Rate are known at month-end (lag 0).
- **Withdrawn history.** Around BEA comprehensive revisions ALFRED records
  most of the PCE price index as missing until it is republished on the new
  basis (23 Dec 2003 – 4 Jan 2004, and 4 Aug 2009). Taken literally, the
  2003-12 origin would have no inflation data at all. The panel uses the last
  complete vintage instead (`realtime.as_of_complete`), which is what a
  forecaster then actually had.
- **Size of revisions (USD, 2000–2026, latest known value at each origin vs
  today's vintage):** capacity utilisation mean |revision| 0.98 pp (max
  2.8 pp, about a quarter of its standard deviation); core PCE inflation
  0.18 pp.

### 7.4 Market and survey benchmarks

The random walk is a statistical yardstick. A user of the model would
instead ask whether it beats **what the market already prices**.

- **Forward rates (all currencies).** The curve at origin $t$ implies a
  forecast of every future yield. With continuously compounded zero yields
  in months,
  $$
  F_t(h, \tau) = \frac{(\tau + h)\, y_t(\tau + h) - h\, y_t(h)}{\tau}
  $$
  is the market-implied forecast of $y_{t+h}(\tau)$. Short-rate futures
  (SOFR, fed funds, €STR, SONIA) price essentially the same quantity, up to
  a small convexity adjustment, but their history is licensed. Forwards
  computed from the public curves keep the public-data rule. GSW and ECB
  curves are evaluated from their parameters at the extra maturities needed
  (e.g. 132m for the 10y yield 12 months ahead). For the BoE, use the
  published maturities (to 40y). The pipeline stores the needed maturities
  in `curves_ext_{CCY}.parquet`; `eval.backtest.forward_forecasts` computes
  the forwards. All three sources are continuously compounded. For the BoE
  this was verified: forwards implied from its spot curve match its
  published forward curve to < 0.1 bp, whereas semi-annual compounding
  would be off by 6–9 bp.
- **Expected result.** Forwards usually forecast *worse* than the random
  walk, because they include a term premium (Fama–Bliss 1987,
  Campbell–Shiller 1991). Diebold–Li (2006) find DNS beats forwards.
- **USD result (2000–2026).** That holds only from 5y out (RMSE ratio to
  RW 1.06–1.13 at h = 6, 12). At the short end forwards *beat* the random
  walk (3m: 0.80 / 0.83 / 0.89 at h = 1 / 6 / 12), so the market's priced
  path for policy rates was informative. The best yields-only model (two-step
  VAR(1)) is within about ±1% of forwards at most points. Its clearest
  edge is the 10y at h = 12 (RMSE 5% below forwards).
- **Significance (HLN-corrected DM, USD 2000–2026).** At h = 6 and 12,
  *nothing* significantly beats the random walk. The short-end gains
  (VAR(1) 3m: 0.84 / 0.88) are not significant. Only at h = 1 do forwards
  (3m: 0.80, 1%) and the VAR(1) (3m: 0.86, 1%) beat it. **No yields-only
  model beats forwards at any maturity or horizon.** At h = 1 they are
  significantly worse at 3m and 5y; at h = 6 and 12 the VAR(1) is
  statistically level (ratios 0.95–1.01). The DL AR(1) is significantly
  worse than both benchmarks almost everywhere. In other words, the curve
  alone contains nothing the market has not already priced, which makes
  the macro model (§6.4) the real test.
- **Yields + macro, first results (USD, real-time macro).** The full DRA
  model (6 states, 74 parameters with μ fixed; Kalman, λ fixed) is *worse*
  than yields-only: RMSE ratio to RW at h = 12 is 0.97 / 1.06 / 1.13 / 1.22 /
  1.23 (3m / 12m / 24m / 5y / 10y), significantly worse than RW at 2y+.
  Clark–West finds no gain from macro at h = 12, only faint short-end
  signal at h = 6 (p 0.05–0.09). The likely cause is overfitting:
  36 VAR coefficients, most weakly identified. It is also numerically
  fragile: two fits within 0.7 log-likelihood points of each other can give
  12-month forecasts up to ~150 bp apart (§8, step 4d). A *parsimonious* version
  supports this. A two-step VAR on (level, slope, curvature, fed funds)
  beats the yields-only VAR by Clark–West at the short end (3m and 12m at
  h = 6, 12, p ≤ 0.05). Its RMSE ratios to RW are the best so far
  (3m 0.81 / 0.86, 12m 0.93 / 0.95 at h = 6 / 12), and for the first time
  it is below forwards (0.95–0.99), though not significantly. So macro
  information helps the short end when the model is small enough to
  estimate it.
- **Restricted and shrunk macro models (USD, Kalman, real-time macro, μ fixed).**
  RMSE relative to yields-only Kalman, Clark–West p in brackets where < 0.10:

  | | 3m | 12m | 24m | 5y | 10y |
  |---|---|---|---|---|---|
  | Policy rate only, h = 6 | 0.96 (0.02) | 0.98 (0.03) | 0.99 (0.07) | 0.99 | 1.01 |
  | Policy rate only, h = 12 | 0.98 (0.03) | 0.99 (0.05) | 0.99 (0.05) | 0.98 (0.06) | 0.97 (0.07) |
  | All 3, shrunk τ = 0.01, h = 12 | 1.01 | 1.00 | 1.00 | 1.02 | 1.03 |
  | All 3, shrunk τ = 0.03, h = 12 | 1.12 | 1.10 | 1.09 | 1.08 | 1.13 |
  | All 3, shrunk τ = 0.1, h = 12 | 1.14 | 1.12 | 1.10 | 1.07 | 1.12 |
  | All 3, shrunk τ = 0.3, h = 12 | 1.11 | 1.10 | 1.09 | 1.07 | 1.12 |
  | All 3, unrestricted, h = 12 | 1.10 | 1.09 | 1.08 | 1.06 | 1.12 |

  The **policy rate** adds significant information, at every maturity at
  h = 12. Capacity utilisation and inflation do not help out of sample:
  shrinking their links towards zero (a Minnesota-style prior on both the
  lagged cross-terms of A and the factor–macro shock covariance, so that
  τ → 0 is exactly yields-only) only gets back to yields-only accuracy, and
  any weight on them makes forecasts worse. The full τ grid is reported, and
  no τ beats the policy-rate model, so the choice of τ involves no data
  snooping. **No model beats forwards significantly.** The best (policy
  rate only) is level with them: RMSE ratio 0.98–1.01 at h = 6 and
  0.95–1.00 at h = 12. (An earlier version of this table came from fits
  where 17 of 27 τ = 0.01 refits had explosive A; see §6.3.)
- **Why so little? Impulse responses and variance decompositions** (USD
  1985–2026, latest vintage, `notebooks/03_macro_usd.ipynb`; Cholesky order
  capacity utilisation, inflation, fed funds, level, slope, curvature; 90%
  bands from the MLE's asymptotic distribution). As in DRA (2006), the
  curve leads the macro economy more than the reverse. Curve shocks explain
  about three quarters of the fed funds rate's 12-month forecast variance
  (curvature alone half). A slope shock (short rates up relative to long)
  lowers inflation by about 10 bp over 6–24 months. In the other direction,
  macro shocks explain only 1–10% of the factors' forecast variance at
  1–12 months (12–32% at 5 years). An inflation shock lifts the level by
  about 5 bp, and activity and policy shocks move mostly the short end. So
  at forecasting horizons, what the curve does not already contain is mostly
  the policy rate, which matches the out-of-sample result. Generalised
  (order-free) responses agree on these links. A generalised fed funds shock
  raises activity and inflation, because it includes the macro news the Fed
  reacts to, which is why a recursive identification is needed.
- **EUR and GBP (2010–2026, `notebooks/04_eur_gbp.ipynb`).** EUR is
  estimated from 2004-09 (start of the ECB curve), GBP from 1993-01
  (inflation targeting). EUR/GBP macro data have no vintages, so the panels
  use release lags (§7.3). The GBP short end is evaluated at 6m: the BoE
  published its 3m point in only 39 of the 202 months since 2010, almost all
  in 2010–12 when Bank Rate sat at 0.5%.
  - *EUR:* the Kalman models' RMSE is 9–63% above the random walk at
    h = 6, 12, and forwards beat them significantly at every maturity up to
    5y. At h = 12 the 10y forecasts ran +92 bp too high in 2010–13 and +78 bp
    in 2014–21 (random walk: +39, +28): they revert to a level mean of
    2.3–4.3%, which the zero/negative-rate decade never returned to. The
    two-step VAR(1) reverts less and is the best EUR model (1–26% above the
    random walk). The policy rate helps only one month ahead.
  - *GBP:* the yields-only Kalman model beats the random walk at 6m
    (0.86–0.88), is level at 1–2y and significantly worse at 10y
    (1.13–1.18). Against forwards its ratio is 0.93–1.16, never
    significantly below 1. Macro variables do not help.
  - *In sample* the pattern of USD repeats: a curvature shock raises the
    policy rate by 20–30 bp at a 15–20 month peak in all three currencies,
    curve shocks explain 45% (EUR) to 73–76% (USD, GBP) of the policy rate's
    12-month forecast variance, and macro shocks only 4–10% of the factors'.
- **Cross-currency summary (common window, origins 2010-01 onwards; mean
  RMSE ratio over 6m, 12m, 24m, 5y, 10y).**

  | | USD h=6 | USD h=12 | EUR h=6 | EUR h=12 | GBP h=6 | GBP h=12 |
  |---|---|---|---|---|---|---|
  | Yields-only Kalman / RW | 0.95 | 0.96 | 1.26 | 1.30 | 1.02 | 1.03 |
  | Yields-only Kalman / forwards | 1.00 | 0.99 | 1.31 | 1.28 | 1.04 | 0.99 |
  | Two-step VAR(1) / forwards | 0.99 | 0.99 | 1.10 | 1.12 | 1.04 | 0.98 |
  | Policy rate only / forwards | 0.98 | 0.98 | 1.33 | 1.35 | 1.09 | 1.05 |

  **No model beats the forward curve significantly in any currency at 6 or
  12 months.** The forward curve already embeds the expected policy path; a
  statistical model that reverts to a historical mean adds error whenever
  rates trend (EUR), and macro variables beyond the policy rate add little
  that the curve has not already priced.
- **Surveys (USD, optional).** Philadelphia Fed Survey of Professional
  Forecasters: quarterly 3m T-bill and 10y Treasury forecasts (free).
- Test every model against each benchmark with Diebold–Mariano (§7.2).

### 7.5 Trading evaluation (economic value)

Lower RMSE does not by itself mean money can be made. A rates position
already pays or earns the forward, so its P&L depends on the realised rate
**relative to the forward**, not to today's rate. The trading signal is
therefore

$$
s_t(h, \tau) = F_t(h, \tau) - \hat y_{t+h|t}(\tau)
$$

$s > 0$: the market prices higher rates than the model expects, so go long
duration (receive fixed or buy futures); $s < 0$: the opposite. The random
walk itself implies a trade, **carry / roll-down** (receive the term
premium), and that is the bar a model must clear.

- **Returns without licensed data.** h-month holding-period excess returns
  of zero-coupon bonds come straight from the curves:
  $rx_{t+h}(n) = n\,y_t(n) - (n-h)\,y_{t+h}(n-h) - h\,y_t(h)$ (in % over
  the period, maturities in months). These approximate futures and swap
  P&L. Real contracts add basis, roll and margin effects; state this as a
  limitation.
- **Strategies, mapped to DNS factors:**

  | Strategy | Factor | Instruments it stands in for |
  |---|---|---|
  | Short-rate path vs forwards | short end (3m–12m) | SOFR / fed funds, €STR / Euribor, SONIA futures |
  | Duration timing | level | Treasury TU/FV/TY, Schatz/Bobl/Bund, Gilt futures; swaps |
  | Steepener / flattener, DV01-neutral | slope | futures pairs, swap curve trades |
  | Butterfly (e.g. 2s5s10s) | curvature | three-leg futures or swaps |
  | Cross-currency relative value | level/slope spreads | USD vs EUR vs GBP |

  Factor-neutral portfolios are built from the Nelson–Siegel loadings, so
  that each portfolio is exposed to one factor only.
- **Sizing.** Position size is proportional to signal divided by forecast
  variance. The Kalman model gives the forecast covariance
  $Z P_{t+h|t} Z' + H$. Trade only when $|s|$ exceeds costs plus a
  multiple of the forecast's standard error. Measure positions in DV01 and
  target a fixed portfolio volatility.
- **Benchmarks:** buy-and-hold duration, carry / roll-down (the
  random-walk view), yields-only vs yields + macro.
- **Metrics:** Sharpe ratio, maximum drawdown, hit rate, turnover,
  certainty-equivalent return, all after costs (assume ~0.25–0.5 bp per
  trade for short-rate futures and <0.5 bp for bond futures; report
  sensitivity to the assumption).
- **Literature:** Thornton & Valente (2012) find that statistical
  predictability of bond returns gave no economic value over forwards.
  Gargano, Pettenuzzo & Timmermann (2019) find value when macro information
  is used. The research question (does macro help?) is answered both in
  RMSE and in P&L.
- **Caveats to report:** the mean-reversion bias of §2 would have kept a
  model short duration through the 2000–2015 bond rally; regime changes
  (zero lower bound, 2022 hikes); GSW curves are smoothed and exclude
  on-the-run bonds. A positive result means "worth testing on contract
  data", not "ready to trade".

### 7.6 Why the models lose, and what to change (literature review, 11 Nov 2026)

The losses in §7.4 have four separate causes. Each shows up in a different
part of the error tables (numbers from notebooks 02–04).

| # | Failure | Where it shows | Evidence |
|---|---|---|---|
| 1 | Persistent fitting errors at the ends of the curve, treated as white noise | Short maturities, h = 1 | Measurement-error sd at 3m: 19 bp (USD), 20 bp (EUR), 29 bp (GBP), against a 1-month random-walk RMSE of 24 bp at USD 3m. Kalman / forwards at h = 1: 1.22 (USD 3m), 2.40 (EUR 3m), 3.29 (GBP 6m), all significant |
| 2 | Mean reversion to a fixed historical mean | Long maturities, h = 6–12, trending periods | 10y mean error at h = 12: USD +51 bp in 2000–07 (RW +20); EUR +92 bp in 2010–13 and +78 bp in 2014–21 (RW +39, +28). Largest eigenvalue of A 0.993 (half-life ~8 years) |
| 3 | Too many weakly identified VAR coefficients | Every macro model except policy rate only | Full DRA 6–23% worse than yields-only at h = 12 (USD); 210 of 1,000 USD impulse-response draws explosive |
| 4 | The model has less information than the market | Short end, h = 6–12 | Forwards beat the RW at USD 3m (0.80 / 0.83 / 0.89 at h = 1 / 6 / 12); no model beats forwards significantly anywhere |

- **Failure 1.** Three Nelson–Siegel factors with λ = 0.0609 cannot match
  the GSW, ECB and BoE curves at 3m and 10y (all three are generated by
  Svensson-type or spline curves). The misfit changes slowly, so today's
  gap predicts next month's, but the forecast Λf̂ throws it away. That is why
  the model loses badly to forwards at h = 1 but is level with them at
  h = 12. Hamilton & Wu (2014) find serially correlated errors in every
  affine model they test. Raviv (2013) adds an intercept and an AR(1) on the
  Diebold–Li errors and cuts US 10y RMSPE at h = 12 by ~31% and 3m at h = 1
  by ~15% (1994–2009, Clark–West significant at all horizons).
- **Failure 2.** μ is the average since 1985 (USD), 2004 (EUR) or 1993
  (GBP), and rates fell through every window. van Dijk, Koopman, van der Wel
  & Wright (2014) let the factors revert to a slowly moving mean (exponential
  smoothing with α = 0.1 a month, long-range surveys, or smoothed real-time
  inflation and growth). Their best version is 0.77 × RW at 10y, h = 12, and
  removes a ~65 bp Diebold–Li overprediction; the gain comes from the level.
  Bauer & Rudebusch (2020) tie the trend to i* = r* + π* and cut long-horizon
  RMSE ~40% against a fixed endpoint. Duffee (2011) finds the most useful
  restriction is a random walk for the level, and Carriero, Kapetanios &
  Marcellino (2012) shrink yield VARs towards random-walk-like univariate
  models.
- **Failure 3.** The current shrinkage pulls macro links towards zero, so it
  can at best recover yields-only. Factors from large macro panels (Moench
  2008; Coroneo, Giannone & Modugno 2016) are the literature's answer, but the
  large gains are on pre-2008 US samples. de Pooter, Ravazzolo & van Dijk
  (2010) find macro models win around recessions and lose in calm periods,
  and that forecast combination gives gains that hold over time.
- **Failure 4.** Altavilla, Giacomini & Ragusa (2017) tilt a model's forecast
  density to match survey forecasts of short yields and let the model carry
  that to the rest of the curve (gains up to 52% in 2000–2012). Here the
  public counterpart is the forward curve itself.
- **Regimes.** The smooth shadow-rate DNS of Opschoor & van der Wel (2025)
  beats DNS at all horizons, most in the ZLB years, and shifting endpoints
  add further gains. Relevant to EUR (negative rates 2014–22) and GBP.
- **Lower priority for accuracy.** No-arbitrage restrictions (AFNS,
  Christensen, Diebold & Rudebusch 2011) help less than dynamic restrictions
  (Duffee 2011). Machine learning on excess returns (Bianchi, Büchner &
  Tamoni 2021) loses much of its edge with real-time data; it belongs to the
  trading step, not RMSE.

Free data that keeps the public-data rule: Philadelphia Fed SPF (3m bill and
10y forecasts; `BOND10`, the 10-year average 10y yield, first-quarter
surveys since 1992; long-run inflation) and ECB SPF longer-term HICP
expectations (quarterly). There is no long free UK survey, so GBP uses the
smoothing variant. The expected effects in §8 step 7b are a reading of the
literature, not results from this project.

### 7.7 Results: persistent errors and shifting endpoints (step 7b items 1–2)

`notebooks/05_accuracy.ipynb`. The grid (§8 step 7b) was fixed and committed
before any of it was run. It covers yields only and yields + policy rate,
each with every endpoint and error model: 24 USD, 18 EUR and 12 GBP
backtests. Origins from 2000 (USD) and 2010 (EUR, GBP). Each change is tested
against its own model without it: Clark–West for the error models, which
nest it; Diebold–Mariano for the endpoints, which do not. The 90% model
confidence set (MCS; Hansen, Lunde & Nason 2011, squared-error loss, block
bootstrap) covers every model in notebooks 02–05: 34 in USD, 28 in EUR and
22 in GBP.

**AR(1) measurement errors: one-month RMSE ratio to the same model without
them** (fixed μ; the range covers yields only and policy rate, `ar1` and
`ar1c`). All significant at 1% (Clark–West), except GBP 5y (5–10%):

| | Short end | 5y | 10y |
|---|---|---|---|
| USD | 3m: 0.82–0.89 | 0.96 | 0.87–0.89 |
| EUR | 3m: 0.67–0.80 | 0.87–0.89 | 0.99–1.03 |
| GBP | 6m: 0.74–0.90 | 0.99 | 0.84 |

- **Confidence set.** At h = 1, the set on the loss pooled over maturities
  keeps every AR(1) variant, the random walk, the λ-estimated Kalman model
  and, in USD and EUR, forwards and the two-step VAR. It drops every other
  plain Kalman model: the λ-fixed models (MCS p-values 0.00–0.06 in all
  three currencies), the endpoint models without AR(1) errors, and the full
  macro and shrinkage models.
- **Against forwards at h = 1:**
  - USD 10y goes from 1.13*** to 1.00 and USD 3m from 1.22–1.27*** to
    1.03–1.09.
  - EUR 3m (1.61–1.74***) and GBP 6m (1.59–1.97***) still lose clearly.
- **No gain at h = 12** (0.96–1.07). The residuals' half-lives are 1–6
  months in USD and EUR; GBP 10y is the exception at 16 months.
- **The intercept (`ar1c`) adds little over `ar1`.** Raviv's (2013) ~31%
  gain at 10y, h = 12, is not reproduced.

**Shifting endpoints: mean 10y forecast error at h = 12, yields only (bp,
forecast − realised):**

| | Period | Fixed μ | EWMA | Survey | Random walk | Forwards |
|---|---|---|---|---|---|---|
| USD | 2000–07 | +51 | +12 | +43 (`bond10`) / +74 (`cpi10`) | +20 | +47 |
| USD | 2014–21 | +45 | +2 | +63 / +80 | +12 | +33 |
| USD | 2022–26 | −56 | −117 | −122 / −54 | −62 | −56 |
| EUR | 2010–13 | +92 | +68 | +93 (`hicp_lt`) | +39 | +76 |
| EUR | 2014–21 | +78 | +38 | +77 | +28 | +49 |
| EUR | 2022–26 | −57 | −110 | −56 | −69 | −63 |
| GBP | 2014–21 | +42 | +14 | – | +20 | +43 |
| GBP | 2022–26 | −93 | −151 | – | −84 | −74 |

- **The EWMA endpoint removes most of the bias in trending years, then
  overshoots when rates turn.** The endpoint lags, and μ_level (the average
  gap between the level and its trailing mean) is negative over a
  falling-rate sample, so the model extrapolates the trend.
- **Net 10y RMSE at h = 12 against fixed μ:** 0.97 (USD), 1.00 (EUR), 1.06
  (GBP). None is significant.
- **US surveys make things worse.** Survey forecasters expected
  normalisation just as the model did: 5y at h = 12 is 7–17% worse than
  fixed μ, significantly for `cpi10`.
- **The ECB long-term HICP expectation helps the 3m–2y slightly** (2–4% at
  h = 12, DM 5–10%) but leaves the 10y bias where it was. It moved only
  between 1.7% and 2.0% while real rates fell. An anchor that tracks
  falling rates needs r* as well (Bauer & Rudebusch 2020).
- van Dijk et al.'s 0.77 × random walk at 10y, h = 12, is not reproduced;
  the best here is 1.07 (USD, EWMA).

**What changes.**
- AR(1) errors (`ar1`) become part of the model for one-month forecasts and
  for the trading signal.
- Endpoints are not adopted as the default; EWMA stays as a variant for
  trending regimes.
- At h = 6–12 no variant beats forwards or the random walk, and the MCS
  keeps 20–34 models at every maturity. The remaining gap is failure 4, the
  information in the forward curve, which items 3–4 of step 7b target.

## 8. Roadmap (next steps, in order)

1. ✅ **`src/yck/model/dns.py`:** Nelson–Siegel loadings, two-step DL
   estimation (cross-section OLS with missing maturities, then AR(1)/VAR(1)),
   forecasts. Add tests: recover known factors from a simulated curve;
   loadings at the τ → 0 and τ → ∞ limits.
2. ✅ **`notebooks/01_two_step_usd.ipynb`:** plot USD factors against the
   empirical proxies (§6.2), fit RMSE by maturity, and a first out-of-sample
   comparison with the random walk at h = 1, 6, 12.
3. ✅ **`src/yck/model/kalman.py`:** a generic linear-Gaussian filter
   (predict, update with missing rows, log-likelihood) and an RTS smoother.
   Test it against `statsmodels` on simulated data, including missing rows.
4. ✅ **`src/yck/model/statespace_dns.py`:** maximum-likelihood DNS (§6.3),
   recursive forecasts (`eval/backtest.py`), `notebooks/02_kalman_usd.ipynb`.
4b. ✅ **Robust and fast estimation.** Compiled filter + smoother returning
   sufficient statistics; EM; exact score via Fisher's identity (tested
   against numerical gradients and brute-force Gaussian conditioning);
   L-BFGS-B with restarts; 6 starts per fit; 1 bp noise floor. What was
   found: numerical-gradient L-BFGS-B stopped with large gradients, and
   warm starts chained across refits drifted between local optima. The root
   cause was near-zero noise variances (~0.01 bp) on some maturities, which
   make the likelihood badly conditioned. With the floor, every fit
   converges and starts agree on one or two optima. Refits are independent,
   parallel and cached (`python -m yck backtest`): a full USD backtest
   takes ~1 min on 8 cores from scratch (was 45–60 min per λ variant)
4c. ✅ **Performance pass** (profiled first; each change verified to give
   the same optima). Per-call and per-fit costs on the 6-state USD macro
   model, and full-backtest wall times on 8 cores:

   | | Before | After | How |
   |---|---|---|---|
   | Log-likelihood | 5.8 ms | 1.0 ms | Information-form filter (Woodbury + determinant lemma): k×k algebra instead of an n×n factorisation each month; Z′H⁻¹Z rebuilt only when the missing-data pattern changes |
   | Score | 28 ms | 0.3 ms | Closed-form gradient of the expected complete-data log-likelihood (adjoint Lyapunov solve for the stationary prior) |
   | Multistart | 6 full optimisations | 6 short scouts + best 2 finished | Pruning: starts are within ~1 point of their optimum after 0.5% of the work |
   | Stopping | 5,800 passes/start, 43% gaining < 0.001 | gradient < 0.005, gain < 1e-7 | Optima match the old strict settings to ≤ 0.01 |
   | Two-step refits per origin | O(t) each | O(1) each | Prefix sums of the regression cross-products (sliding window for rolling) |
   | USD backtest, yields-only λ fixed / λ est. / macro | 291 s / 622 s / 1,300 s | 21 s / 37 s / 104 s | All of the above; yields-only log-likelihoods within 0.001 of before, macro within 0.08 |
   and seconds from the cache.
4d. ✅ **Second performance pass** (11 Nov 2026; profiled first, every change
   checked against the previous outputs). The whole standard grid (USD, EUR,
   GBP x yields only with both λ, full macro, policy rate only, four
   shrinkage τ: 24 backtests, 488 refits) took **1,238 s** of fitting run as 24
   separate commands; it now takes **~250 s** end to end with
   `python -m yck backtest --ccy USD EUR GBP --all`, with bit-identical
   outputs, and **95 s** after μ was fixed at the sample mean (§6.3; a
   better-conditioned likelihood, not a speed trick). Profiling showed 87–89% of each fit inside the filter/smoother
   pass, and the predicted covariance converging within ~4 months.

   | | Before | After | How |
   |---|---|---|---|
   | Filter + smoother + statistics, yields only / macro | 2.31 / 3.58 ms | 0.16 / 0.36 ms | **Steady state:** P_{t\|t-1} does not depend on the data, so once it stops changing (same observation pattern) the filter reuses P, its inverse and log-determinants, and the smoother reuses its gain and, once converged, P_{t\|T}; any change in the missing pattern recomputes. **No temporaries:** k ≤ 6 algebra written as loops into preallocated buffers (allocation cost more than the arithmetic). **Statistics:** each series' Σ E[xx'] is the total minus its missing months, O(N k²) instead of O(N n k²) |
   | Lyapunov solve (stationary prior, adjoint in the score) | scipy, ~100 µs | compiled, same direct method | 8,000 calls per macro fit |
   | Model rebuilt in the score | twice per evaluation | once | passed from the objective |
   | Scheduling | one pool and one barrier per backtest | one pool for all refits, longest first (LPT) | `prefit`, `--all` |
   | Real-time macro panels | rebuilt per call | memoised by origin | shared by refits, forecasts and variants |
   | Full grid, fitting / planning + forecasts | 1,238 s / ~50 s | 244 s / 10 s | all of the above |
   | Full grid after fixing μ (§6.3) | | 86 s / 9 s | fewer, better-conditioned iterations |

   **Notebooks and data plumbing** (same rules: profile, then change, then
   check outputs are identical). All four notebooks run in ~22 s together
   from the cache (01: 3 s, 02: 4 s, 03: 6 s, 04: 8 s); what is left is
   full-sample fits, matplotlib and imports.

   | | Before | After | How |
   |---|---|---|---|
   | 320 USD real-time macro panels | 2.95 s | 0.32 s | `to_monthly` fast path for series already one-per-month (numpy month arithmetic, y/y as a 12-month array shift; the generic path rebuilt a calendar in pure Python each call), and `VintageTable`: each ALFRED table preprocessed once into sorted arrays, so a vintage lookup is two vectorised comparisons. Identical on all 1,004 USD (series, origin) pairs |
   | 1,000 impulse-response draws (6 states, 60 months) | ~0.5 s | 0.012 s | all draws as stacks of matrices (batched eigenvalues, Cholesky, matrix powers) instead of a Python loop; same random stream, identical draws |
   | Notebooks on a fresh clone | each launched its own backtests | one shared grid run | `eval.grid.ensure_results` (same code as `backtest --all`) |

   Tested and rejected: finishing only one scout per likelihood basin (14%
   faster, but lost 0.2–0.6 log-likelihood points on the flat macro ridges);
   single-threaded BLAS in the workers (no effect). The remaining limit is
   hardware: on an M1 four of the eight workers run on efficiency cores.
5. **Benchmarks and tests (`src/yck/eval/`):** forward-rate panel in the
   data pipeline and forward benchmark (§7.4), optional SPF survey,
   Diebold–Mariano and Clark–West tests (§7.2), results tables (CSV +
   markdown).
6. ✅ **Yields + macro (§6.4).** Done: real-time macro panels
   (`data/realtime.py`), the DRA state-space model (any number of macro
   states, analytic score), real-time backtest (`python -m yck backtest
   --macro`), policy-rate-only variant (`--macro-vars`), shrinkage prior on
   all factor–macro links (`--shrink`). Results (§7.4): only the policy rate
   helps (short end); the full model overfits; shrinkage does not rescue
   capacity utilisation or inflation. Impulse responses, variance
   decompositions and parameter-uncertainty bands (`model/irf.py`) and
   `notebooks/03_macro_usd.ipynb` complete the step. Original plan: real-time macro panels first (§7.3: ALFRED
   vintages for USD, release lags for EUR/GBP, unreleased months as missing
   observations). Then the 6-state DRA model with macro observed without
   error, with EM and the exact score extended to it, impulse responses and
   variance decompositions. Rerun step 5 for it: macro vs yields-only with
   Clark–West, and against forwards with Diebold–Mariano.
7. ✅ **EUR and GBP.** Steps 5 and 6 for both, and the cross-currency
   comparison (`notebooks/04_eur_gbp.ipynb`, §7.4). GBP is estimated from
   1993 (inflation targeting) and evaluated at 6m at the short end (the BoE
   3m exists for only 39 months since 2010). Along the way: estimates are
   required to be stationary, and μ is fixed at the real-time sample mean
   (§6.3), which made every fit well-posed and changed all Kalman results
   (USD RMSE ratios improved by 0.01–0.07).
7b. **Accuracy improvements (§7.6), in this order.** Items 1–2 ✅ (results in
   §7.7: AR(1) errors adopted for h = 1, endpoints not adopted); next: item 3,
   then 4. Fix the full grid of
   variants before running it, report every variant, and add a model
   confidence set across all models to the DM/CW tables, to avoid data
   snooping. Each step keeps real-time data and filtered states (§7.3).

   | # | Change | Fixes (§7.6) | How it fits the code | Expected effect (literature) | Cost |
   |---|---|---|---|---|---|
   | 1 | AR(1) measurement errors | 1 | Cheap version: forecast Λf̂ + ρʰeₜ per maturity, ρ fitted on filtered residuals up to the origin. Full version: errors as extra states, or a few error principal components to keep k small | Removes most of the h = 1 loss to forwards; smaller gains at h = 6–12 | Low / medium |
   | 2 | Shifting endpoint for the level | 2 | μ is already set outside the likelihood at each refit; make it μₜ and forecast μₜ₊ₕ = μₜ. Variants: exponential smoothing (α = 0.1, all currencies); SPF `BOND10` / long-run inflation (USD); ECB SPF long-term HICP (EUR). Series IDs go in `config/data_sources.yaml` | Largest at 5y–10y, h = 12; should remove most of the EUR level bias | Low |
   | 3 | Forecast combination | 2, 3 | Equal-weight average of Kalman, two-step VAR, forwards and RW, then performance weights; post-processing of existing outputs | Steady gains, largest at long maturities | Very low |
   | 4 | Anchor the short end to forwards | 4 | Exponential tilting so the 3m–12m forecast mean matches the forward-implied path; the model sets the long end. Using forwards instead of surveys is untested in the literature | Matches forwards at the short end, keeps the model's 10y edge | Medium |
   | 5 | Prior centred on a random walk | 2, 3 | Reuse the shrinkage code with A's prior centred on the identity for the level (Minnesota); macro block limited to the policy rate | Fewer explosive draws; overlaps with #2, test together | Low |
   | 6 | Shadow-rate DNS (EUR, GBP) | 2, regimes | Non-linear measurement: extended or unscented filter; exact score and steady-state shortcuts need rework | Gains in the zero/negative-rate years; complementary to #2 | High |
   | 7 | Macro factors instead of single series | 3 | One or two factors from a large public macro panel per currency, entering like the policy rate | Uncertain after 2008 | Medium–high |
   | 8 | Better cross-sectional fit | 1 | Second slope/curvature factor (dynamic Svensson) or time-varying λ | Lower 3m and 10y noise; overlaps with #1 | Medium |

   **Grid for items 1–2, fixed on 11 Nov 2026 before any of it was run**
   (`eval/grid.py: accuracy_grid`; `python -m yck backtest --ccy USD EUR GBP --accuracy`):

   - Models: yields only and yields + policy rate (λ fixed, no shrinkage),
     the two best of step 7.
   - Endpoint for the level (`model/adjust.py`): none; `ewma` (one-sided
     exponential smoothing of the two-step level factor, α = 0.1, all
     currencies); `bond10` (SPF median expected 10-year average 10y yield,
     USD); `cpi10` (SPF median 10-year CPI inflation, USD); `hicp_lt` (ECB
     SPF longer-term HICP, EUR). Implemented by detrending: the endpoint is
     subtracted from every yield and from the policy rate, the model is
     estimated and filtered as before with μ at the real-time sample mean
     (for an inflation endpoint μ_level is then the average real rate plus
     term premium), and the origin's endpoint is added back, held flat. A
     survey counts as known from the end of the second month of its quarter.
     Survey variants start their estimation sample at the first survey
     (USD: Nov 1991 / Feb 1992); forecast origins are unchanged.
   - Measurement errors: none; `ar1` (ρʰeₜ); `ar1c` (m + ρʰ(eₜ − m), Raviv
     2013), fitted per maturity at each origin on the filtered residuals up
     to the origin, ρ clipped to ±0.99.
   - 24 USD, 18 EUR and 12 GBP backtests; each is reported, against the
     random walk and forwards (DM), against its plain version (CW for the
     error models, which nest it; DM for endpoints, which do not), and in a
     90% model confidence set (Hansen, Lunde & Nason 2011, `eval/stats.py:
     mcs`, squared-error loss, block bootstrap with blocks of max(h, 3)
     months) over every model in the step 7 and 7b grids plus the random
     walk, forwards and the two-step model.

   **Grid for items 3–4, fixed on 11 Nov 2026 before any of it was run**
   (`eval/grid.py: anchor_grid, COMBINATION_SETS`; `eval/combine.py`;
   `python -m yck backtest --ccy USD EUR GBP --anchor`):

   - **Item 4, anchoring to forwards (`anc-fwd`).** The model's Gaussian
     h-step forecast of the curve, N(μ, Σ) with Σ = Λ P_{t+h|t} Λ' + H, is
     exponentially tilted (minimum Kullback–Leibler change; Altavilla,
     Giacomini & Ragusa 2017). The means at 3m, 6m, 9m and 12m then equal
     the forward rates at the origin: μ* = μ + Σ·s Σss⁻¹(F − μs). Forwards
     whose inputs are not published are not used, so GBP often anchors
     fewer maturities. Applied to yields only and to yields + policy rate,
     each plain and with `ar1` errors: 4 backtests per currency, no
     re-estimation.
   - **Item 3, forecast combination.** Two sets of models:
     - set A: yields + `ar1`, two-step VAR(1), forwards, random walk;
     - set B: set A plus yields + policy rate + `ar1`.

     Each set has two weighting schemes:
     - equal weights;
     - inverse-MSE weights (Bates & Granger 1969), per maturity and
       horizon. They use only errors realised by the origin (targets
       ≤ T − h) over an expanding window, with equal weights until every
       available model has 24 realised errors.

     That gives 4 combinations per currency. A missing component (e.g. GBP
     forwards at h = 1) is dropped and the remaining weights renormalised.
   - **Reporting.** Every variant is reported against forwards and the
     random walk (DM). Each anchored model is tested against the same model
     unanchored (DM). The 90% model confidence set is extended to every
     model from steps 7 and 7b.

   Fixing failures 1 and 2 also matters for step 8: the trading signal is
   F − ŷ, and the level bias would keep the strategy short duration through
   the 2000–2015 and 2010–2021 rallies.
8. **Trading evaluation (`src/yck/eval/trading.py`, §7.5), last, once the
   models are final (after step 7b).** Excess returns from the curves, carry / roll-down
   benchmark, then strategies: short-rate vs forwards first, then duration,
   curve and fly, then cross-currency relative value. Variance-based
   sizing, costs, Sharpe and certainty-equivalent return. The headline
   result is macro vs no macro in both RMSE and P&L.
9. Write-up in `README.md` (results section): RMSE ratios against RW and
   forwards, DM/CW statistics, strategy performance after costs, plus factor
   and impulse-response charts.

Dependencies: `scipy` is core; `statsmodels` is in `dev` (test reference
only); `matplotlib` and `jupyter` are in the `notebook` extra.

## 9. Conventions and decisions already made

- **Public data only.** No Bloomberg or other licensed data; that was a
  deliberate compliance decision. Do not add proprietary sources.
- **No interpolation** of missing curve points; missing stays NaN and the
  Kalman filter deals with it.
- Series IDs, URLs and settings live in `config/data_sources.yaml`, never
  hard-coded.
- Every new data source gets a pure `parse_*` function with an offline test
  using a small sample in the source's real format, plus a thin `fetch_*`
  wrapper.
- Yields in percent, maturities in months, dates = calendar month-end.
- Filtered (not smoothed) states for anything out-of-sample.
- Keep the code readable and explained: this repository is a public
  portfolio piece, so docstrings should say *why*, not just *what*.
- Commit small, descriptive commits. CI must stay green.

## 10. GitHub Actions and tooling

- **tests** (`ci.yml`): runs `pytest -q` on every push and pull request.
- **fetch data** (`fetch-data.yml`): runs `python -m yck fetch` on a GitHub
  runner, manually (Actions tab → "fetch data" → Run workflow, optional
  input such as `USD EUR`) and monthly via cron on the 3rd at 06:17 UTC. It
  needs the repository secret `FRED_API_KEY` (already set). The full
  `coverage.md` is printed in the log and the job summary, and the
  processed parquet files are attached as the `processed-data` artifact.
  The job fails if any source fails.
- **probe ECB key** (`probe-ecb.yml`): Actions tab → "probe ECB key" →
  inputs `flow` and `key` (blank key = list dataflows matching `flow`;
  `?` = show dimensions).
- With the GitHub CLI: `gh workflow run fetch-data.yml` and
  `gh run watch` / `gh run view --log`.

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FRED_API_KEY is not set` | Create `.env` in the repo root (§1, step 2) |
| An ECB series ends early (stale) | The ECB moved it to a new dataflow; find the new key with `probe-ecb` (§1, §5.4) and edit `config/data_sources.yaml` |
| ECB `504 Gateway Time-out` | Transient. The code already requests one series at a time with a 180 s timeout; rerun |
| BoE "No GLC sheets found" | The BoE renamed sheets; check the sheet names logged by the fetch and add them to `curves.GBP.sheets` in the config |
| Stale data after a fix | Cached downloads in `data/raw/`; run with `--refresh` or delete the file |
| git "lock" / "file exists" errors in OneDrive | Pause OneDrive sync, delete the stale `.git/*.lock` file, retry |

## 12. References

- Diebold & Li (2006), *Forecasting the term structure of government bond yields*, J. Econometrics
- Diebold, Rudebusch & Aruoba (2006), *The macroeconomy and the yield curve: a dynamic latent factor approach*, J. Econometrics
- Diebold & Rudebusch (2013), *Yield Curve Modeling and Forecasting*, Princeton (available via the University of Bath library)
- Durbin & Koopman (2012), *Time Series Analysis by State Space Methods*, OUP (missing observations)
- Hamilton (1994), *Time Series Analysis*, ch. 13 (Kalman filter and MLE)
- Gürkaynak, Sack & Wright (2007), *The U.S. Treasury yield curve: 1961 to the present*, J. Monetary Economics
- Christensen, Diebold & Rudebusch (2011), *The affine arbitrage-free class of Nelson–Siegel term structure models*, J. Econometrics
- Koopman, Mallee & van der Wel (2010), time-varying λ and volatility in DNS, JBES
- Joslin, Priebsch & Singleton (2014), *Risk premiums in dynamic term structure models with unspanned macro risks*, J. Finance
- Cochrane & Piazzesi (2005), *Bond risk premia*, AER
- Ludvigson & Ng (2009), *Macro factors in bond risk premia*, RFS
- Adrian, Crump & Moench (2013), *Pricing the term structure with linear regressions*, JFE
- Bauer & Hamilton (2018), *Robust bond risk premia*, RFS
- Diebold & Mariano (1995), *Comparing predictive accuracy*, JBES
- Clark & West (2007), *Approximately normal tests for equal predictive accuracy in nested models*, J. Econometrics

Accuracy improvements (§7.6, §8 step 7b):

- Altavilla, Giacomini & Ragusa (2017), *Anchoring the yield curve using survey expectations*, J. Applied Econometrics
- Bauer & Rudebusch (2020), *Interest rates under falling stars*, AER
- Bianchi, Büchner & Tamoni (2021), *Bond risk premiums with machine learning*, RFS
- Carriero, Kapetanios & Marcellino (2012), *Forecasting government bond yields with large Bayesian vector autoregressions*, J. Banking & Finance
- Christensen, Diebold & Rudebusch (2009), *An arbitrage-free generalized Nelson–Siegel term structure model*, Econometrics Journal
- Coroneo, Giannone & Modugno (2016), *Unspanned macroeconomic factors in the yield curve*, JBES
- de Pooter, Ravazzolo & van Dijk (2010), *Term structure forecasting using macro factors and forecast combination*, Fed IFDP 993
- Diebold, Li & Yue (2008), *Global yield curve dynamics and interactions: a dynamic Nelson–Siegel approach*, J. Econometrics
- Duffee (2011), *Forecasting with the term structure: the role of no-arbitrage restrictions*, JHU working paper
- Hamilton & Wu (2014), *Testable implications of affine term structure models*, J. Econometrics
- Hevia, Gonzalez-Rozada, Sola & Spagnolo (2015), *Estimating and forecasting the yield curve using a Markov switching dynamic Nelson and Siegel model*, J. Applied Econometrics
- Moench (2008), *Forecasting the yield curve in a data-rich environment: a no-arbitrage factor-augmented VAR approach*, J. Econometrics
- Opschoor & van der Wel (2025), *A smooth shadow-rate dynamic Nelson–Siegel model for yields at the zero lower bound*, JBES
- Raviv (2013), *Prediction bias correction for dynamic term structure models*, Tinbergen Institute DP 13-041
- van Dijk, Koopman, van der Wel & Wright (2014), *Forecasting interest rates with shifting endpoints*, J. Applied Econometrics
