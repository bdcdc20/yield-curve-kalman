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
python3 -m pip install -e ".[dev]"

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
| Offline tests for all parsers | **Done.** 14 tests, all passing (`pytest -q`) |
| CI (tests on every push) and monthly data-fetch workflow | **Done** |
| Two-step Diebold–Li baseline (yields only) | **Next.** Not started |
| One-step Kalman / maximum-likelihood DNS (yields only) | Not started |
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

## 3. Repository layout

```
config/data_sources.yaml    every series ID, URL and setting; fix broken IDs here, not in code
src/yck/
  __main__.py               CLI: `python -m yck fetch | probe-ecb`; loads .env
  data/http.py              requests session, retries, on-disk cache in data/raw
  data/sources.py           one parse_* (pure, unit-tested) + fetch_* per source; ECB probes; ALFRED as_of()
  data/curves.py            Svensson/Nelson-Siegel evaluation, month-end sampling
  data/build.py             orchestrates everything and writes data/processed + coverage.md
tests/test_sources.py       offline tests with small samples in each source's real format
.github/workflows/
  ci.yml                    pytest on every push / PR
  fetch-data.yml            full fetch on a GitHub runner (manual + monthly cron); coverage report in the log
  probe-ecb.yml             runs `yck probe-ecb` on a runner (for finding ECB keys)
data/                       raw/ and processed/ (git-ignored)
```

Model code does not exist yet. Planned location: `src/yck/model/` (see §8).

## 4. Data pipeline

### 4.1 Outputs (`data/processed/`)

| File | Content |
|---|---|
| `curves_{USD,EUR,GBP}.parquet` | Month-end zero-coupon yields in **percent, continuously compounded** (BoE: see 4.2). Index `date` = calendar month-end. Columns `3m, 6m, 9m, 12m, 15m, 18m, 21m, 24m, 30m, 36m, 48m, 60m, 72m, 84m, 96m, 108m, 120m` |
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
   months; it maximises the curvature loading at 30 months.
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
  `scipy.optimize.minimize` (L-BFGS-B, then Nelder–Mead or BFGS polish).
  Start from the two-step estimates.
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
| Any model vs **random walk** ($\hat y_{t+h} = y_t$) | Diebold–Mariano (1995), HAC variance with Newey–West lag h−1 | RW is the standard hard-to-beat benchmark; not nested in DNS |
| **Yields + macro vs yields-only** | **Clark–West (2007)**, one-sided | Nested models: under the null, the larger model's extra parameter noise biases plain MSE comparisons against it |

Clark–West adjusted loss differential, with model 1 the small (nested)
model and model 2 the large one:

$$
\hat f_{t+h} = e_{1,t+h}^2 - \left[e_{2,t+h}^2 - (\hat y_{1,t+h} - \hat y_{2,t+h})^2\right]
$$

Regress $\hat f$ on a constant with Newey–West standard errors (lag h−1)
and reject the null of equal accuracy if t > 1.282 (10%) or t > 1.645 (5%).

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

## 8. Roadmap (next steps, in order)

1. **`src/yck/model/dns.py`:** Nelson–Siegel loadings, two-step DL
   estimation (cross-section OLS with missing maturities, then AR(1)/VAR(1)),
   forecasts. Add tests: recover known factors from a simulated curve;
   loadings at the τ → 0 and τ → ∞ limits.
2. **`notebooks/01_two_step_usd.ipynb`:** plot USD factors against the
   empirical proxies (§6.2), fit RMSE by maturity, and a first out-of-sample
   comparison with the random walk at h = 1, 6, 12.
3. **`src/yck/model/kalman.py`:** a generic linear-Gaussian filter
   (predict, update with missing rows, log-likelihood) and an RTS smoother.
   Test it against `statsmodels` on simulated data, including missing rows.
4. **`src/yck/model/statespace_dns.py`:** parameter packing/unpacking,
   maximum-likelihood estimation of §6.3, then the macro extension of §6.4.
5. **`src/yck/eval/`:** forecasting loop (§7.1), real-time macro panels
   (§7.3), RW benchmark, Diebold–Mariano and Clark–West tests, results
   tables (CSV + markdown).
6. Run for USD, then EUR and GBP. Cross-currency comparison.
7. Write-up in `README.md` (results section): headline table of RMSE
   ratios against RW and the CW statistics, plus factor and impulse-response
   charts.

Dependencies to add when needed: `scipy`, `statsmodels` (test reference
only), `matplotlib`, and `jupyter` under `[project.optional-dependencies]`.

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
