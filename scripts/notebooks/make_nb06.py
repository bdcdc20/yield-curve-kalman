"""Generates notebooks/06_combination_anchor.ipynb (run from the repo root, then execute with nbconvert)."""
import sys
from pathlib import Path

import nbformat as nbf

# The takeaways are written after reading the executed notebook's outputs, and kept beside this script.
TAKEAWAYS_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "takeaways" / "nb06.md"
TAKEAWAYS = TAKEAWAYS_FILE.read_text()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 06 · Forecast combination and anchoring to the forward curve

Notebook 05 left two facts. AR(1) measurement errors fix the one-month
forecasts. At 6–12 months nothing beats forwards or the random walk, and
README §7.6 (failure 4) puts that down to the forward curve knowing the
expected policy path. This notebook tests the next two items of README §8
step 7b, both aimed at that horizon:

3. **Forecast combination.** Average the model with the forecasts it
   competes against. Models fail in different periods, and averages hold up
   over time in the term-structure literature (de Pooter, Ravazzolo & van
   Dijk 2010).
   - **Set A:** yields + AR(1) errors, two-step VAR(1), forwards and the
     random walk.
   - **Set B:** set A plus yields + policy rate + AR(1).
   - **Weights:** equal, or inverse past MSE (Bates & Granger 1969).
     Inverse-MSE weights are set per maturity and horizon, from errors
     already realised at the origin only, with equal weights until each
     model has 24 of them (`yck.eval.combine`).
4. **Anchoring the short end to forwards.** Tilt the model's Gaussian
   forecast of the curve so the 3m–12m means equal the forward rates
   (Altavilla, Giacomini & Ragusa 2017). The rest of the curve moves by its
   forecast covariance with the short end: μ* = μ + Σ·s Σss⁻¹(F − μs), with
   Σ = Λ P_{t+h|t} Λ' + H from the model (`yck.model.adjust.anchor_mean`).
   It is applied to yields only and policy rate only, each plain and with
   AR(1) errors. The literature tilts to surveys; tilting to forwards is
   untested.

**The grid was fixed and committed before anything was run** (README §8 step
7b, commit `328364e`). Everything in it is reported below.
""")

code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

from yck.model.dns import maturities_from_columns
from yck.eval.backtest import dns_two_step_forecasts, random_walk_forecasts, forward_forecasts
from yck.eval.combine import combine
from yck.eval.grid import (STANDARD_GRID, BACKTEST_DEFAULTS, COMBINATION_SETS, accuracy_grid, anchor_grid,
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

for g in (STANDARD_GRID, accuracy_grid, anchor_grid):  # one grid run if anything is missing
    ensure_results(CCYS, DATA, CONFIG, grid=g)

def name_of(row):
    m, lam, tau, ep, err, anc, *_ = tuple(row) + (None,) * (9 - len(row))
    base = ("policy rate" if m == ("policy_rate",) else "full macro" if m else "yields")
    base += "" if lam == "fixed" else ", λ est."
    base += f", shrink {tau:g}" if tau is not None else ""
    base += f", endpoint {ep}" if ep else ""
    base += " + AR(1)" if err == "ar1" else " + AR(1) with mean" if err == "ar1c" else ""
    return base + (", anchored" if anc else "")

y, fc = {}, {}
for c in CCYS:
    est = BACKTEST_DEFAULTS[c][0]
    full = pd.read_parquet(ROOT / f"data/processed/curves_{c}.parquet")
    y[c] = full.loc[est:] if est else full
    fc[c] = {}
    for g in (STANDARD_GRID, accuracy_grid, anchor_grid):
        for r in _grid_for(g, c):
            fc[c][name_of(r)] = {h: pd.read_parquet(RESULTS / f"fc_{c}_kalman_{tag_of(r)}_h{h}.parquet") for h in H}
    fc[c]["two-step AR(1)"] = dns_two_step_forecasts(y[c], H, FIRST_ORIGIN[c])
    fc[c]["two-step VAR(1)"] = dns_two_step_forecasts(y[c], H, FIRST_ORIGIN[c], kind="var1")
    fc[c]["random walk"] = random_walk_forecasts(y[c], H, FIRST_ORIGIN[c])
    fc[c]["forwards"] = forward_forecasts(pd.read_parquet(ROOT / f"data/processed/curves_ext_{c}.parquet"),
                                          y[c].columns, H, FIRST_ORIGIN[c])
    fc[c].update(combination_forecasts(c, DATA, CONFIG, sets=("A", "B")))
ANCHORED = [name_of(r) for r in _grid_for(anchor_grid, "USD")]
UNANCHORED = [name_of(r[:5]) for r in _grid_for(anchor_grid, "USD")]
COMBOS = [n for n in fc["USD"] if n.startswith("combination")]
pd.DataFrame({c: {"models compared": len(fc[c])} for c in CCYS})
""")

md(r"""
## 1. What anchoring does to one forecast

The EUR 12-month forecast made in January 2015, when the ECB was about to
start buying bonds. The plain model forecast a rebound towards its mean.
The forwards priced rates near zero for a long time. The anchored forecast
takes the forwards at 3m–12m and lets the model carry them out along the
curve.
""")

code(r"""
c, origin, h = "EUR", pd.Timestamp("2015-01-31"), 12
target = origin + pd.offsets.MonthEnd(h)
tau = maturities_from_columns(y[c].columns)
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(tau, y[c].loc[origin], "k--", lw=1, label=f"curve at origin ({origin:%b %Y})")
ax.plot(tau, y[c].loc[target], "k", lw=1.5, label=f"realised ({target:%b %Y})")
ax.plot(tau, fc[c]["yields + AR(1)"][h].loc[target], lw=1.2, label="model (yields + AR(1))")
ax.plot(tau, fc[c]["yields + AR(1), anchored"][h].loc[target], lw=1.2, label="anchored to forwards")
ax.plot(tau, fc[c]["forwards"][h].loc[target], lw=1, color="0.5", label="forwards")
ax.set_xlabel("maturity (months)"); ax.set_ylabel("%"); ax.legend(fontsize=8, frameon=False)
ax.set_title(f"EUR, 12-month forecasts made in {origin:%b %Y}", loc="left", fontsize=10);
""")

md(r"""
## 2. Anchoring: every variant against forwards, the random walk and itself

Cells are RMSE ratios. Against forwards and against the unanchored model
they carry Diebold–Mariano stars in either direction (* 10%, ** 5%, *** 1%).
Below 1 means the anchored model wins. At the anchored maturities the
forecast equals the forward rate, so the forward ratio there is 1 by
construction (up to GBP months where a forward is missing).
""")

code(r"""
def after(c, h):
    return y[c].loc[pd.Timestamp(FIRST_ORIGIN[c]) + pd.offsets.MonthEnd(h):]

def vs(c, bench, h, names, field="cell"):
    t = dm_table({n: fc[c][n][h] for n in [bench] + names}, bench, after(c, h), h, SHOW[c])
    return t.xs(field, axis=1, level=1).T

for c in CCYS:
    rows = {}
    for a, u in zip(ANCHORED, UNANCHORED):
        for h in H:
            rows[(a, f"h = {h}", "÷ forwards")] = vs(c, "forwards", h, [a]).loc[a]
            rows[(a, f"h = {h}", "÷ unanchored")] = vs(c, u, h, [a]).loc[a]
            rows[(a, f"h = {h}", "÷ random walk")] = vs(c, "random walk", h, [a], "ratio").loc[a].astype(float).round(2)
    display(pd.DataFrame(rows).T.style.set_caption(f"{c}: anchored models").set_table_styles(CAP))
""")

md(r"""
## 3. Combinations against forwards and the random walk

RMSE / forwards with Diebold–Mariano stars, and RMSE / random walk (ratio
only).
""")

code(r"""
for c in CCYS:
    t = pd.concat({**{f"÷ forwards, h = {h}": vs(c, "forwards", h, COMBOS) for h in H},
                   **{f"÷ RW, h = {h}": vs(c, "random walk", h, COMBOS, "ratio").astype(float).round(2) for h in H}},
                  axis=1)
    display(t.style.set_caption(f"{c}: combinations").set_table_styles(CAP))
""")

md(r"""
### Where the inverse-MSE weights go

Set A's real-time weights for the 10-year yield at h = 12. They move slowly,
because the window expands. The dotted line marks the switch from equal
weights, once every forecast has 24 realised errors.
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
for ax, c in zip(axes, CCYS):
    comp = {"yields + AR(1)": fc[c]["yields + AR(1)"][12], "two-step VAR(1)": fc[c]["two-step VAR(1)"][12],
            "forwards": fc[c]["forwards"][12], "random walk": fc[c]["random walk"][12]}
    _, w = combine(comp, y[c], 12, "inv_mse", return_weights=True)
    for n, wn in w.items():
        ax.plot(wn["120m"].loc[FIRST_ORIGIN[c]:], lw=1.2, label=n)
    ax.axvline(pd.Timestamp(FIRST_ORIGIN[c]) + pd.offsets.MonthEnd(12 + 24 - 1), ls=":", color="k", lw=0.8)
    ax.set_title(c, loc="left", fontsize=10)
axes[0].set_ylabel("weight, 10y, h = 12"); axes[0].legend(fontsize=7, frameon=False); fig.tight_layout()
""")

md(r"""
## 4. Model confidence set over everything

The 90% model confidence set (Hansen, Lunde & Nason 2011) over every model
in notebooks 02–06: the step 7 grid, the step 7b grid (endpoints, error
models, anchoring), the combinations, the two-step models, the random walk
and forwards. The first table counts the models in the set out of those
compared. The second gives the MCS p-values for the pooled loss over
maturities: a model is in the set when p ≥ 0.10.
""")

code(r"""
SHORT = "3m / 6m (GBP)"
mcs = {(c, h): mcs_table({n: f[h] for n, f in fc[c].items()}, after(c, h), h, SHOW[c], n_boot=2000)
               .rename(columns={SHOW[c][0]: SHORT})
       for c in CCYS for h in H}
summary = {(c, f"h = {h}"): {col: f"{int((t[col] >= 0.10).sum())} / {int(t[col].notna().sum())}" for col in t.columns}
           for (c, h), t in mcs.items()}
display(pd.DataFrame(summary).T.style.set_caption("Models in the 90% MCS / models compared").set_table_styles(CAP))
KEY = ["yields", "yields + AR(1)", "policy rate + AR(1)", "yields + AR(1), anchored",
       "policy rate + AR(1), anchored"] + COMBOS + ["two-step VAR(1)", "random walk", "forwards"]
pd.DataFrame({(c, f"h = {h}"): mcs[(c, h)].loc[KEY, "all"] for c in CCYS for h in H}).round(2)
""")

code(r"""
# Lowest pooled MSE at each horizon on the MCS's common sample, and ranks among all models compared
rows = {}
for (c, h), t in mcs.items():
    a = after(c, h)[SHOW[c]]
    pooled = pd.DataFrame({n: ((f[h].reindex(a.index)[SHOW[c]] - a) ** 2).mean(axis=1, skipna=False)
                           for n, f in fc[c].items()})
    L = pooled.dropna(axis=1, how="all").dropna().mean()
    best = L.idxmin()
    rows[(c, f"h = {h}")] = {"lowest pooled MSE": best,
                             "rank of forwards": int(L.rank()["forwards"]) if "forwards" in L else "–",
                             "rank of random walk": int(L.rank()["random walk"]),
                             "rank of best combination": int(L.rank()[COMBOS].min()),
                             "models": len(L)}
pd.DataFrame(rows).T
""")

md(r"""
## 5. Takeaways

""" + TAKEAWAYS)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/06_combination_anchor.ipynb")
