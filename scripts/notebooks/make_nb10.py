"""Generates notebooks/10_macro_factors.ipynb (run from the repo root, then execute with nbconvert)."""
import sys
from pathlib import Path

import nbformat as nbf

# The takeaways are written after reading the executed notebook's outputs, and kept beside this script.
TAKEAWAYS_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "takeaways" / "nb10.md"
TAKEAWAYS = TAKEAWAYS_FILE.read_text()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 10 · Macro factors from large public panels

The three-series DRA model overfits (notebook 03), and only the policy rate
carries information about future yields, in USD only (notebooks 03–04). The
factor-model answer is to summarise a large macro panel in one or two
factors (Moench 2008; Coroneo, Giannone & Modugno 2016) and let those enter
the state like the policy rate (README §8 step 7b, item 7).

**Fixed before the run** (commit `4f0ec02`; `yck.model.macro_factors`):

- **Panels.**
  - USD: FRED-MD, 126 series, with every monthly vintage since 1999-08, so
    the USD factors are fully real-time.
  - EUR: EA-MD-QD, the 47 monthly series.
  - GBP: UK-MD, 111 series.

  Interest rates and spreads are excluded, so the factors carry macro
  information rather than the curve itself.
- **Factors** (McCracken & Ng 2016):
  - transformation codes; outliers set to missing;
  - EM principal components on the estimation window;
  - unit variance;
  - sign fixed by industrial production (factor 1) and consumer prices
    (factor 2).
- **Real time.** Factors are re-extracted at every origin from the panel
  known then: the FRED-MD vintage of that month (USD), or the latest file
  cut at each series' publication delay (EUR, GBP; revisions ignored).
- **Models.** Yields + factor 1; yields + factors 1–2; yields + policy
  rate + factors 1–2. Each is run plain and with AR(1) errors.
""")




code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

from yck.eval.backtest import dns_kalman_forecasts, dns_two_step_forecasts, random_walk_forecasts, forward_forecasts
from yck.eval.grid import (STANDARD_GRID, BACKTEST_DEFAULTS, accuracy_grid, anchor_grid, shadow_grid, rwprior_grid,
                           svensson_grid, SVENSSON_LAM2, factor_grid, plan_runs,
                           combination_forecasts, ensure_results, tag_of, _grid_for)
from yck.eval.stats import dm_table, mcs_table

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RESULTS, CONFIG, DATA = ROOT / "data/results", ROOT / "config/data_sources.yaml", ROOT / "data"
CCYS = ["USD", "EUR", "GBP"]
H = (1, 6, 12)
SHOW = {"USD": ["3m", "12m", "24m", "60m", "120m"], "EUR": ["3m", "12m", "24m", "60m", "120m"],
        "GBP": ["6m", "12m", "24m", "60m", "120m"]}
FIRST_ORIGIN = {c: BACKTEST_DEFAULTS[c][1] for c in CCYS}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
CAP = [{"selector": "caption", "props": "font-weight: bold; text-align: left"}]

GRIDS = (STANDARD_GRID, accuracy_grid, anchor_grid, shadow_grid, rwprior_grid, svensson_grid, factor_grid)
for g in GRIDS:  # one grid run if anything is missing
    ensure_results(CCYS, DATA, CONFIG, grid=g)

def name_of(row):
    m, lam, tau, ep, err, anc, sr, rw, l2 = tuple(row) + (None,) * (9 - len(row))
    LBL = {("policy_rate",): "policy rate", ("factor1",): "1 factor", ("factor1", "factor2"): "2 factors",
           ("policy_rate", "factor1", "factor2"): "policy rate + 2 factors"}
    base = "shadow" if sr else (LBL.get(m, "full macro") if m else "yields")
    base += "" if lam == "fixed" else ", λ est."
    base += f", shrink {tau:g}" if tau is not None else ""
    base += f", RW prior {rw:g}" if rw is not None else ""
    base += f", Svensson {'short' if l2 > 0.1 else 'long'}" if l2 is not None else ""
    base += f", endpoint {ep}" if ep else ""
    base += " + AR(1)" if err == "ar1" else " + AR(1) with mean" if err == "ar1c" else ""
    return base + (", anchored" if anc else "") + (f" ({sr})" if sr else "")

y, fc = {}, {}
for c in CCYS:
    est = BACKTEST_DEFAULTS[c][0]
    full = pd.read_parquet(ROOT / f"data/processed/curves_{c}.parquet")
    y[c] = full.loc[est:] if est else full
    fc[c] = {}
    for g in GRIDS:
        for r in _grid_for(g, c):
            fc[c][name_of(r)] = {h: pd.read_parquet(RESULTS / f"fc_{c}_kalman_{tag_of(r)}_h{h}.parquet") for h in H}
    fc[c]["two-step AR(1)"] = dns_two_step_forecasts(y[c], H, FIRST_ORIGIN[c])
    fc[c]["two-step VAR(1)"] = dns_two_step_forecasts(y[c], H, FIRST_ORIGIN[c], kind="var1")
    fc[c]["random walk"] = random_walk_forecasts(y[c], H, FIRST_ORIGIN[c])
    fc[c]["forwards"] = forward_forecasts(pd.read_parquet(ROOT / f"data/processed/curves_ext_{c}.parquet"),
                                          y[c].columns, H, FIRST_ORIGIN[c])
    fc[c].update(combination_forecasts(c, DATA, CONFIG))
FAC = [name_of(r) for r in _grid_for(factor_grid, "USD")]
YIELDS_OF = {p: "yields" + (" + AR(1)" if "AR(1)" in p else "") for p in FAC}
pd.DataFrame({c: {"models compared": len(fc[c])} for c in CCYS})
""")


md(r"""
## 1. The factors

The factors as extracted at the last origin (the full sample known in
October 2026), next to their anchor series (transformed, standardised).
Factor 1 is a real-activity factor in all three panels; factor 2 follows
prices in USD and GBP but much less in EUR.
""")

code(r"""
import yaml
from yck.model.macro_factors import RealTimeFactors, transform
spec = yaml.safe_load(CONFIG.read_text())["macro_panels"]
fig, axes = plt.subplots(2, 3, figsize=(13, 5), sharex="col")
corr = {}
for j, c in enumerate(CCYS):
    rt = RealTimeFactors.load(c, DATA, CONFIG, 2, start=BACKTEST_DEFAULTS[c][0])
    origin = y[c].index[-1]
    f = rt.at(origin)
    raw, tc = rt.panel_at(origin)
    anc = spec[c]["anchors"]
    x = transform(raw[anc], tc).loc[f.index]
    for i in range(2):
        ax = axes[i, j]
        z = (x[anc[i]] - x[anc[i]].mean()) / x[anc[i]].std()
        ax.plot(z.clip(-6, 6), lw=0.6, color="0.6", label=f"{anc[i]} (std.)")
        ax.plot(f[f"factor{i + 1}"].clip(-6, 6), lw=0.9, label=f"factor {i + 1}")
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        corr[(c, f"factor {i + 1}")] = round(float(f[f"factor{i + 1}"].corr(x[anc[i]])), 2)
    axes[0, j].set_title(f"{c} ({raw.shape[1] - len(spec[c]['exclude'])} series)", loc="left", fontsize=10)
fig.tight_layout()
pd.Series(corr, name="correlation with anchor").unstack()
""")

md(r"""
## 2. Do the factors add information?

- **Against yields only:** Clark–West, nested; one-sided stars in favour
  of the factor model. Cells are RMSE ratios (< 1 means the factors help).
- **Policy rate + 2 factors against the policy-rate model:** DM,
  two-sided.
""")

code(r"""
def after(c, h):
    return y[c].loc[pd.Timestamp(FIRST_ORIGIN[c]) + pd.offsets.MonthEnd(h):]

def vs(c, bench, h, names, field="cell"):
    t = dm_table({n: fc[c][n][h] for n in [bench] + names}, bench, after(c, h), h, SHOW[c])
    return t.xs(field, axis=1, level=1).T

from yck.eval.stats import cw_table
for c in CCYS:
    rows = {}
    for h in H:
        for p in FAC:
            t = cw_table({YIELDS_OF[p]: fc[c][YIELDS_OF[p]][h], p: fc[c][p][h]}, YIELDS_OF[p], after(c, h), h, SHOW[c])
            rows[(f"h = {h}", p + " ÷ yields (CW)")] = t[(p, "cell")]
        for e in ("", " + AR(1)"):
            p, b = "policy rate + 2 factors" + e, "policy rate" + e
            rows[(f"h = {h}", p + " ÷ policy rate (DM)")] = vs(c, b, h, [p]).loc[p]
    display(pd.DataFrame(rows).T.style.set_caption(f"{c}").set_table_styles(CAP))
""")

md(r"""
## 3. Against forwards and the random walk
""")

code(r"""
SHOWN = ["yields", "policy rate", "full macro"] + [p for p in FAC if "AR(1)" not in p] + ["combination A, inverse MSE"]
for c in CCYS:
    t = pd.concat({**{f"÷ forwards, h = {h}": vs(c, "forwards", h, SHOWN) for h in H},
                   **{f"÷ RW, h = {h}": vs(c, "random walk", h, SHOWN, "ratio").astype(float).round(2) for h in H}},
                  axis=1)
    display(t.style.set_caption(f"{c}").set_table_styles(CAP))
""")

md(r"""
## 4. Model confidence set over everything

The 90% MCS over every model in notebooks 02–10 (pooled squared-error
loss).
""")

code(r"""
SHORT = "3m / 6m (GBP)"
mcs = {(c, h): mcs_table({n: f[h] for n, f in fc[c].items()}, after(c, h), h, SHOW[c], n_boot=2000)
               .rename(columns={SHOW[c][0]: SHORT})
       for c in CCYS for h in H}
summary = {(c, f"h = {h}"): {col: f"{int((t[col] >= 0.10).sum())} / {int(t[col].notna().sum())}" for col in t.columns}
           for (c, h), t in mcs.items()}
display(pd.DataFrame(summary).T.style.set_caption("Models in the 90% MCS / models compared").set_table_styles(CAP))
KEY = ["yields", "policy rate", "full macro"] + FAC + ["combination A, inverse MSE", "random walk", "forwards"]
pd.DataFrame({(c, f"h = {h}"): mcs[(c, h)].loc[KEY, "all"] for c in CCYS for h in H}).round(2)
""")

md(r"""
## 5. Takeaways

""" + TAKEAWAYS)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/10_macro_factors.ipynb")
