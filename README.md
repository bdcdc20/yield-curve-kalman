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

As of 10 Nov 2026:

| Stage | Status |
|---|---|
| Data pipeline (curves + macro, USD/EUR/GBP) | **Done.** Verified end-to-end on GitHub Actions; every source fetches and is current |
| Offline tests | **Done.** 55 tests (parsers, DNS, backtest, Kalman filter vs statsmodels and brute force, EM, exact score), all passing (`pytest -q`) |
| CI (tests on every push) and monthly data-fetch workflow | **Done** |
| Two-step Diebold–Li baseline (yields only) | **Done** (`src/yck/model/dns.py`). USD 1985–2000: factor/proxy correlations 0.97–0.999, fit RMSE 2–8 bp (3m: 8 bp) |
| USD two-step notebook (factor plots, first RW comparison) | **Done** (`notebooks/01_two_step_usd.ipynb`); first results below |
| Kalman filter + RTS smoother with missing observations | **Done** (`src/yck/model/kalman.py`). Matches statsmodels to ~1e-9 (log-likelihood, filtered/smoothed states and covariances), with gaps; ~20 ms per likelihood on USD 1985– |
| One-step maximum-likelihood DNS (yields only) | **Done** (`statespace_dns.py`, `notebooks/02_kalman_usd.ipynb`). EM + exact score, 1 bp noise floor, multistart; full USD backtest ~1 min from scratch (both λ variants), seconds from cache |
| Forward-rate and survey benchmarks, trading evaluation | Planned (§7.4, §7.5) |
| Yields + macro model (Diebold–Rudebusch–Aruoba 2006) | Not started |
| Rolling out-of-sample forecasts, benchmarks, tests | Not started |
| Cross-currency comparison and write-up | Not started |

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
log-likelihood 13,285 (λ fixed) and 15,396 (λ estimated, 0.039), with a
1 bp noise floor, against 11,168 at the two-step estimates. Out of sample (refit yearly, filtered
states), h = 12:

| | 3m | 12m | 24m | 60m | 120m |
|---|---|---|---|---|---|
| Two-step VAR(1) | **0.88** | **0.96** | 1.02 | 1.12 | 1.07 |
| Kalman, λ fixed | 0.91 | 1.00 | 1.07 | 1.19 | 1.14 |
| Kalman, λ estimated | 0.94 | 1.01 | 1.07 | 1.17 | 1.19 |

Better in-sample fit, worse forecasts. The estimated long-run level μ was
6.1–8.3% in every refit while the 10y yield was mostly 1–5%, so the
mean-reversion bias is stronger, not weaker (10y, h = 12, 2000–07: +61 bp
vs +47 bp for the two-step VAR). An earlier numerical-gradient estimator
often stopped short of the maximum, but its forecasts differed from these by
only a few bp (RMSE ratios ≤ 0.03), so the conclusion is robust.

## 3. Repository layout

```
config/data_sources.yaml    every series ID, URL and setting; fix broken IDs here, not in code
src/yck/
  __main__.py               CLI: `python -m yck fetch | probe-ecb`; loads .env
  data/http.py              requests session, retries, on-disk cache in data/raw
  data/sources.py           one parse_* (pure, unit-tested) + fetch_* per source; ECB probes; ALFRED as_of()
  data/curves.py            Svensson/Nelson-Siegel evaluation, month-end sampling
  data/build.py             orchestrates everything and writes data/processed + coverage.md
  model/dns.py              Nelson-Siegel loadings, two-step Diebold-Li estimation, factor dynamics, forecasts
  model/kalman.py           Kalman filter (missing rows), log-likelihood, RTS smoother, smoother sufficient statistics (compiled)
  model/statespace_dns.py   ML DNS: parameter transforms, EM, exact score, multistart L-BFGS-B, 1 bp noise floor
  eval/backtest.py          recursive out-of-sample forecasts (no look-ahead), parallel cached refits, random walk, RMSE tables
notebooks/                  analysis notebooks, committed with outputs
tests/                      offline tests: parsers (small samples in each source's real format), DNS, backtest
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

- **Parameters:** $\mu$ (3), $A$ (9), $Q$ via its Cholesky factor (6),
  $\log \operatorname{diag}(H)$ (17), and $\lambda$ (1, estimate it as
  $\exp$ of a free parameter, or fix it at 0.0609 first).
- **Initial state:** the unconditional mean $\mu$, with the unconditional
  covariance from the discrete Lyapunov equation
  ($P_0 = A P_0 A' + Q$, `scipy.linalg.solve_discrete_lyapunov`).
  If $A$ is not stable, fall back to a diffuse prior (large $P_0$).
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
  model (6 states, 80 parameters; Kalman, λ fixed) is *worse* than
  yields-only: RMSE ratio to RW at h = 12 is 1.00 / 1.10 / 1.18 / 1.29 /
  1.25 (3m / 12m / 24m / 5y / 10y), significantly worse than RW at 2y+.
  Clark–West finds no gain from macro at h = 12, only faint short-end
  signal at h = 1 and 6 (p 0.02–0.10). The likely cause is overfitting:
  36 VAR coefficients, most weakly identified. A *parsimonious* version
  supports this. A two-step VAR on (level, slope, curvature, fed funds)
  beats the yields-only VAR by Clark–West at the short end (3m and 12m at
  h = 6, 12, p ≤ 0.05). Its RMSE ratios to RW are the best so far
  (3m 0.81 / 0.86, 12m 0.93 / 0.95 at h = 6 / 12), and for the first time
  it is below forwards (0.95–0.99), though not significantly. So macro
  information helps the short end when the model is small enough to
  estimate it. Next: restricted / shrunk macro models (§8, step 6).
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
5. **Benchmarks and tests (`src/yck/eval/`):** forward-rate panel in the
   data pipeline and forward benchmark (§7.4), optional SPF survey,
   Diebold–Mariano and Clark–West tests (§7.2), results tables (CSV +
   markdown).
6. 🔶 **Yields + macro (§6.4).** Done: real-time macro panels
   (`data/realtime.py`), the DRA state-space model (any number of macro
   states, analytic score), real-time backtest (`python -m yck backtest
   --macro`). First result: the full 6-state model overfits (§7.4).
   Remaining: restricted or shrunk versions (policy rate only, a
   block-restricted A, shrinkage towards yields-only), impulse responses and
   a notebook. Original plan: real-time macro panels first (§7.3: ALFRED
   vintages for USD, release lags for EUR/GBP, unreleased months as missing
   observations). Then the 6-state DRA model with macro observed without
   error, with EM and the exact score extended to it, impulse responses and
   variance decompositions. Rerun step 5 for it: macro vs yields-only with
   Clark–West, and against forwards with Diebold–Mariano.
7. **EUR and GBP.** Run steps 5 and 6 for both. GBP exercises the
   missing-short-end handling. Cross-currency comparison.
8. **Trading evaluation (`src/yck/eval/trading.py`, §7.5), last, once the
   models are final.** Excess returns from the curves, carry / roll-down
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
