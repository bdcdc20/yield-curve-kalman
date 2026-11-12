"""Generates notebooks/05_accuracy.ipynb (run from the repo root, then execute with nbconvert)."""
import sys
from pathlib import Path

import nbformat as nbf

# The takeaways are written after reading the executed notebook's outputs, and kept beside this script.
TAKEAWAYS_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "takeaways" / "nb05.md"
TAKEAWAYS = TAKEAWAYS_FILE.read_text()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 05 · Accuracy improvements: shifting endpoints and persistent fitting errors

Notebooks 02–04 found that no model beats the forward curve, and README §7.6
traced the losses to four failures. This notebook tests the fixes for the
first two (README §8, step 7b, items 1–2):

1. **Persistent fitting errors.** Three Nelson–Siegel factors cannot match
   the published curves at 3m and 10y, and the misfit changes slowly. The
   plain forecast Λf̂ throws it away. Fix: an AR(1) for each maturity's
   filtered residual, added to the forecast: ρʰeₜ (`ar1`), or
   m + ρʰ(eₜ − m) with an intercept (`ar1c`, Raviv 2013).
2. **Mean reversion to a fixed mean.** μ is the average since 1985, 1993 or
   2004, and rates fell through every window, so the level kept forecasting a
   rebound. Fix: a shifting endpoint for the level, which the factors revert
   to instead (van Dijk, Koopman, van der Wel & Wright 2014):
   `ewma`, exponential smoothing of the level factor (α = 0.1 a month);
   `bond10` and `cpi10`, the Philadelphia Fed SPF's expected 10-year yield
   and 10-year CPI inflation (USD); `hicp_lt`, the ECB SPF's longer-term
   HICP expectation (EUR).

**The grid was fixed before anything was run** (README §8 step 7b; commit
`65d0165`, before any results existed): yields only and yields + policy rate,
each with every endpoint available for the currency and every error model,
24 USD, 18 EUR and 12 GBP backtests. All are reported below. Because many
variants are compared, §6 adds a model confidence set (Hansen, Lunde & Nason
2011) over every model from notebooks 02–05, which accounts for the search.

How it is implemented (`yck.model.adjust`):

- **Endpoint by detrending.** The level's loading is 1 at every maturity, so
  reverting to μ + eₜ is the ordinary DNS model for yₜ − eₜ. The endpoint is
  subtracted from every yield and from the policy rate, the model is estimated
  and filtered as before (μ at the real-time sample mean), and the origin's
  eₜ is added back to the forecast, held flat. With an inflation endpoint,
  μ_level becomes the average real rate plus term premium.
- **Real time.** The EWMA at month s uses curves up to s only. A survey
  counts as known from the end of the second month of its quarter.
- **Errors.** At each origin, ρ (and m) are fitted by OLS on the filtered
  residuals up to the origin. No re-estimation is needed, so the `ar1`
  variants reuse the plain models' fits.
""")

code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

from yck.model.adjust import fit_error_ar
from yck.model.dns import fit_factors
from yck.model.statespace_dns import fit_dns_mle
from yck.eval.backtest import dns_two_step_forecasts, random_walk_forecasts, forward_forecasts
from yck.eval.grid import (STANDARD_GRID, ENDPOINTS, ERROR_MODELS, accuracy_grid, ensure_results, load_endpoint,
                           run_tag, _grid_for)
from yck.eval.stats import dm_table, cw_table, mcs_table

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RESULTS, CONFIG, DATA = ROOT / "data/results", ROOT / "config/data_sources.yaml", ROOT / "data"
CCYS = ["USD", "EUR", "GBP"]
EST_START = {"USD": "1985-01-31", "EUR": "2004-09-30", "GBP": "1993-01-31"}
FIRST_ORIGIN = {"USD": "2000-01-31", "EUR": "2010-01-31", "GBP": "2010-01-31"}
H = (1, 6, 12)
SHOW = {"USD": ["3m", "12m", "24m", "60m", "120m"], "EUR": ["3m", "12m", "24m", "60m", "120m"],
        "GBP": ["6m", "12m", "24m", "60m", "120m"]}  # GBP 3m: 39 published months since 2010 (notebook 04)
EP_LABEL = {None: "fixed μ", "ewma": "EWMA", "bond10": "SPF 10y yield", "cpi10": "SPF 10y CPI",
            "hicp_lt": "ECB SPF HICP"}
ERR_LABEL = {None: "", "ar1": " + AR(1)", "ar1c": " + AR(1) with mean"}
MODEL_LABEL = {None: "yields", ("policy_rate",): "policy rate"}

ensure_results(CCYS, DATA, CONFIG)                      # step 7 grid (for the model confidence set)
ensure_results(CCYS, DATA, CONFIG, grid=accuracy_grid)  # step 7b grid
y = {c: pd.read_parquet(ROOT / f"data/processed/curves_{c}.parquet").loc[EST_START[c]:] for c in CCYS}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

def label(m, ep, err):
    return f"{MODEL_LABEL[m]}: {EP_LABEL[ep]}{ERR_LABEL[err]}"

fc, variants = {}, {}
for c in CCYS:
    fc[c] = {}
    for m, lam, tau, ep, err, *_ in _grid_for(accuracy_grid, c):
        tag = run_tag(m, lam, tau, 1.0, ep, err)
        fc[c][label(m, ep, err)] = {h: pd.read_parquet(RESULTS / f"fc_{c}_kalman_{tag}_h{h}.parquet") for h in H}
    variants[c] = list(fc[c])
    for m, lam, tau, *_ in _grid_for(STANDARD_GRID, c):  # the other step 7 models, for the confidence set
        tag = run_tag(m, lam, tau)
        name = "step 7: " + tag.replace("_lamfixed", "").replace("lamfixed", "yields").replace("lamest", "yields, λ est.")
        if m in (None, ("policy_rate",)) and lam == "fixed" and tau is None:
            continue  # same as the plain step 7b variants
        fc[c][name] = {h: pd.read_parquet(RESULTS / f"fc_{c}_kalman_{tag}_h{h}.parquet") for h in H}
    fc[c]["two-step AR(1)"] = dns_two_step_forecasts(y[c], H, FIRST_ORIGIN[c])
    fc[c]["two-step VAR(1)"] = dns_two_step_forecasts(y[c], H, FIRST_ORIGIN[c], kind="var1")
    fc[c]["random walk"] = random_walk_forecasts(y[c], H, FIRST_ORIGIN[c])
    fc[c]["forwards"] = forward_forecasts(pd.read_parquet(ROOT / f"data/processed/curves_ext_{c}.parquet"),
                                          y[c].columns, H, FIRST_ORIGIN[c])
pd.DataFrame({c: {"step 7b variants": len(variants[c]), "models in the confidence set": len(fc[c])} for c in CCYS})
""")

md(r"""
## 1. The endpoints

Each panel shows the month-by-month two-step level factor (what the endpoint
replaces μ for), the EWMA endpoint and the survey endpoints, as known in real
time. The dashed line is the fixed μ of the last refit (the real-time sample
mean of the level).
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
for ax, c in zip(axes, CCYS):
    full = pd.read_parquet(ROOT / f"data/processed/curves_{c}.parquet")
    level = fit_factors(y[c])["level"]
    ax.plot(level, lw=0.8, color="0.6", label="level factor")
    for ep in ENDPOINTS[c][1:]:
        e = load_endpoint(ep, c, full, DATA, CONFIG).loc[EST_START[c]:]
        ax.plot(e, lw=1.3, label=EP_LABEL[ep], drawstyle="steps-post" if ep != "ewma" else "default")
    ax.axhline(level.mean(), ls="--", color="k", lw=0.8, label="fixed μ (full sample)")
    ax.axvline(pd.Timestamp(FIRST_ORIGIN[c]), color="k", lw=0.5, alpha=0.5)
    ax.set_title(c, loc="left", fontsize=10)
axes[0].set_ylabel("%"); axes[0].legend(fontsize=7, frameon=False)
axes[1].legend(fontsize=7, frameon=False); fig.tight_layout()
""")

md(r"""
## 2. How persistent are the fitting errors?

The AR(1) fitted to each maturity's filtered residual by the plain
yields-only model, on the full sample (estimated once here for
illustration; the backtests refit at every origin on data up to that
origin). A ρ near 1 with a large standard deviation is a misfit that today's
residual predicts well; the plain forecast ignores it.
""")

code(r"""
rows = {}
for c in CCYS:
    f = fit_dns_mle(y[c], n_starts=6)
    resid = y[c].to_numpy() - f.filtered_states(y[c]) @ f.statespace().Z.T
    m, rho = fit_error_ar(resid, "ar1c")
    sd = np.nanstd(resid, axis=0)
    for col in SHOW[c]:
        j = list(y[c].columns).index(col)
        rows[(c, col)] = {"residual sd (bp)": 100 * sd[j], "mean m (bp)": 100 * m[j], "ρ": rho[j],
                          "half-life (months)": np.log(0.5) / np.log(rho[j]) if 0 < rho[j] < 1 else np.nan}
pd.DataFrame(rows).T.round(2)
""")

md(r"""
## 3. Every variant against forwards

RMSE relative to the forward-rate forecast (< 1: the model wins), with
Diebold–Mariano significance in either direction (* 10%, ** 5%, *** 1%).
Rows are model: endpoint + error model; the first row of each block is the
plain step 7 model. Forecast origins from 2000 (USD) and 2010 (EUR, GBP).
GBP has no forwards at h = 1 beyond 2 years (the 1m yield and the 61m+
points are not published), and its h = 1 forwards cover only 36 months,
because the BoE rarely publishes a 1m yield.
""")

code(r"""
def after(c, h):
    return y[c].loc[pd.Timestamp(FIRST_ORIGIN[c]) + pd.offsets.MonthEnd(h):]

def vs(c, bench, h, names, field="cell"):
    t = dm_table({n: fc[c][n][h] for n in [bench] + names}, bench, after(c, h), h, SHOW[c])
    return t.xs(field, axis=1, level=1).T

for c in CCYS:
    t = pd.concat({f"h = {h}": vs(c, "forwards", h, variants[c]) for h in H}, axis=1)
    display(t.style.set_caption(f"{c}: RMSE / forwards").set_table_styles(
        [{"selector": "caption", "props": "font-weight: bold; text-align: left"}]))
""")

md(r"""
The same variants against the random walk (ratio only):
""")

code(r"""
for c in CCYS:
    t = pd.concat({f"h = {h}": vs(c, "random walk", h, variants[c], "ratio").astype(float) for h in H}, axis=1)
    display(t.round(2).style.set_caption(f"{c}: RMSE / random walk").set_table_styles(
        [{"selector": "caption", "props": "font-weight: bold; text-align: left"}]))
""")

md(r"""
## 4. Does each fix help its own model?

Each change is tested against the model without it, on the same sample.

- **Error models** nest the plain model (ρ = 0, m = 0), so the test is
  Clark–West, one-sided in favour of the AR(1).
- **Endpoints** do not nest the fixed mean, so the test is Diebold–Mariano,
  two-sided. Each endpoint is compared with fixed μ, with no error model on
  either side.

Cells show the RMSE ratio, new / old.
""")

code(r"""
def nested(c, h):
    rows = {}
    for m in MODEL_LABEL:
        for ep in ENDPOINTS[c]:
            base = label(m, ep, None)
            for err in ERROR_MODELS[1:]:
                t = cw_table({base: fc[c][base][h], "x": fc[c][label(m, ep, err)][h]}, base, after(c, h), h, SHOW[c])
                rows[label(m, ep, err)] = t[("x", "cell")]
            if ep is not None:
                plain = label(m, None, None)
                t = dm_table({plain: fc[c][plain][h], "x": fc[c][base][h]}, plain, after(c, h), h, SHOW[c])
                rows[base + "  (vs fixed μ, DM)"] = t[("x", "cell")]
    return pd.DataFrame(rows).T

for c in CCYS:
    t = pd.concat({f"h = {h}": nested(c, h) for h in (1, 12)}, axis=1)
    display(t.style.set_caption(f"{c}: RMSE ratio to the same model without the change").set_table_styles(
        [{"selector": "caption", "props": "font-weight: bold; text-align: left"}]))
""")

md(r"""
## 5. Where the endpoint helps and where it hurts

Mean 10-year forecast error at h = 12 (forecast − realised, bp) by
period, for the yields-only model with each endpoint, against the random walk
and forwards. Positive means rates were forecast too high.
""")

code(r"""
PERIODS = {"USD": [("2000", "2007"), ("2008", "2013"), ("2014", "2021"), ("2022", "2026")],
           "EUR": [("2010", "2013"), ("2014", "2021"), ("2022", "2026")],
           "GBP": [("2010", "2013"), ("2014", "2021"), ("2022", "2026")]}
bias = {}
for c in CCYS:
    names = [label(None, ep, None) for ep in ENDPOINTS[c]] + ["random walk", "forwards"]
    for a, b in PERIODS[c]:
        bias[(c, f"{a}–{b}")] = {n.replace("yields: ", ""): 100 * (fc[c][n][12]["120m"] - y[c]["120m"]).loc[a:b].mean()
                                 for n in names}
pd.DataFrame(bias).T.round(0)
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.3), sharey=True)
for ax, c in zip(axes, CCYS):
    ax.plot(y[c]["120m"].loc[FIRST_ORIGIN[c]:], color="k", lw=1.2, label="realised 10y")
    for ep in ENDPOINTS[c][:2]:
        n = label(None, ep, None)
        ax.plot(fc[c][n][12]["120m"].loc[FIRST_ORIGIN[c]:], lw=0.9, label=f"{EP_LABEL[ep]}, 12 months earlier")
    ax.plot(fc[c]["forwards"][12]["120m"].loc[FIRST_ORIGIN[c]:], lw=0.8, color="0.6", label="forward, 12 months earlier")
    ax.set_title(c, loc="left", fontsize=10)
axes[0].set_ylabel("%"); axes[0].legend(fontsize=7, frameon=False); fig.tight_layout()
""")

md(r"""
## 6. Model confidence set

The 90% model confidence set over **every** model in notebooks 02–05: the
step 7b variants, the other step 7 models (λ estimated, full macro,
shrinkage), the two-step models, the random walk and forwards. It contains
the best model with 90% probability, so a model outside it is beaten by the
others jointly, not just in one pairwise test chosen after the fact.

Loss is the squared error; the 'all' column averages it over the maturities
shown. GBP forwards have no h = 1 forecasts beyond 2 years, so they are
left out of those columns. The second table gives the MCS p-values (pooled
loss) of the plain models, their AR(1) versions and the two benchmarks: a
model is in the set when p >= 0.10.
""")

code(r"""
SHORT = "3m / 6m (GBP)"
mcs = {(c, h): mcs_table({n: f[h] for n, f in fc[c].items()}, after(c, h), h, SHOW[c], n_boot=2000)
               .rename(columns={SHOW[c][0]: SHORT})
       for c in CCYS for h in H}
summary = {(c, f"h = {h}"): {col: f"{int((t[col] >= 0.10).sum())} / {int(t[col].notna().sum())}" for col in t.columns}
           for (c, h), t in mcs.items()}
display(pd.DataFrame(summary).T.style.set_caption("Models in the 90% MCS / models compared").set_table_styles(
    [{"selector": "caption", "props": "font-weight: bold; text-align: left"}]))
""")

code(r"""
# h = 1, pooled loss: which models survive, and the MCS p-values of the plain models and benchmarks
keep = {c: list(mcs[(c, 1)].index[mcs[(c, 1)]["all"] >= 0.10]) for c in CCYS}
for c in CCYS:
    print(f"{c}, h = 1, all maturities: {len(keep[c])} models in the 90% set:\n   " + "\n   ".join(keep[c]))
show = [label(None, None, None), label(None, None, "ar1"), label(("policy_rate",), None, None),
        label(("policy_rate",), None, "ar1"), "random walk", "forwards"]
pd.DataFrame({(c, f"h = {h}"): mcs[(c, h)].loc[show, "all"] for c in CCYS for h in H}).round(2)
""")

md(r"""
## 7. Takeaways

""" + TAKEAWAYS)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/05_accuracy.ipynb")
