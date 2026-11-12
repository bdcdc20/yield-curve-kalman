"""Generates notebooks/09_svensson.ipynb (run from the repo root, then execute with nbconvert)."""
import sys
from pathlib import Path

import nbformat as nbf

# The takeaways are written after reading the executed notebook's outputs, and kept beside this script.
TAKEAWAYS_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "takeaways" / "nb09.md"
TAKEAWAYS = TAKEAWAYS_FILE.read_text()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 09 · A second curvature factor: the dynamic Svensson model

Three Nelson–Siegel factors with λ = 0.0609 cannot match the published
curves at the short end and at 10 years. Notebook 05 found the fitting
errors there persistent (README §7.6, failure 1), and fixed them after the
fact with AR(1) errors. This notebook fixes the fit instead (README §8,
step 7b, item 8). The three source curves are themselves Svensson or spline
curves, and Svensson (1994) adds a second curvature term with its own decay
λ₂. The dynamic version (De Pooter 2007) makes it a fourth factor in the
state.

**Fixed before the run** (commit `0f0bb3a`):

- **Model.** A 4-factor model estimated exactly like the 3-factor one: EM +
  exact score, 1 bp floor, μ at the real-time sample mean, 6 starts.
- **λ₂ is set by where the second hump peaks** (λτ = 1.7933 at the peak):
  - short: 9 months, λ₂ = 0.1993;
  - long: 90 months, λ₂ = 0.0199.
- **Variants.** Yields only, each plain and with AR(1) errors.

The 3-factor fits are unchanged: a refit reproduces the cached
log-likelihood exactly.
""")



code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

from yck.eval.backtest import dns_kalman_forecasts, dns_two_step_forecasts, random_walk_forecasts, forward_forecasts
from yck.eval.grid import (STANDARD_GRID, BACKTEST_DEFAULTS, accuracy_grid, anchor_grid, shadow_grid, rwprior_grid,
                           svensson_grid, SVENSSON_LAM2, plan_runs,
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

GRIDS = (STANDARD_GRID, accuracy_grid, anchor_grid, shadow_grid, rwprior_grid, svensson_grid)
for g in GRIDS:  # one grid run if anything is missing
    ensure_results(CCYS, DATA, CONFIG, grid=g)

def name_of(row):
    m, lam, tau, ep, err, anc, sr, rw, l2 = tuple(row) + (None,) * (9 - len(row))
    base = "shadow" if sr else ("policy rate" if m == ("policy_rate",) else "full macro" if m else "yields")
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
SV = [name_of(r) for r in _grid_for(svensson_grid, "USD")]
BASE_OF = {p: "yields" + (" + AR(1)" if "AR(1)" in p else "") for p in SV}
pd.DataFrame({c: {"models compared": len(fc[c])} for c in CCYS})
""")


md(r"""
## 1. The fit

Measurement-noise sd by maturity at the last refit, and the log-likelihood
gain over the 3-factor model at the first and last refit. The short hump
takes out the short-end misfit and the long hump the 10-year one.
""")

code(r"""
from yck.eval.backtest import dns_kalman_forecasts
rows = {}
for c in CCYS:
    def fits_of(tag, grid):
        kw = [k for cc, t, k in plan_runs([c], DATA, CONFIG, grid=grid) if t == tag][0]
        f = []
        dns_kalman_forecasts(**kw, horizons=(1,), cache_dir=RESULTS / "cache", fits=f)
        return f
    f3 = fits_of("lamfixed", STANDARD_GRID)
    for name, l2 in SVENSSON_LAM2.items():
        f4 = fits_of(f"sv{l2:.4f}_lamfixed", svensson_grid)
        sd3, sd4 = 100 * np.sqrt(f3[-1][1].params.h), 100 * np.sqrt(f4[-1][1].params.h)
        cols = list(f3[-1][1].yield_columns)
        lab = {SHOW[c][0]: "short end (3m; GBP 6m)", "60m": "5y", "120m": "10y"}
        rows[(c, name)] = {**{f"noise sd {lab[m]}, 3f → 4f (bp)": f"{sd3[cols.index(m)]:.1f} → {sd4[cols.index(m)]:.1f}"
                              for m in lab},
                           "log-lik gain, first / last refit": f"{f4[0][1].loglik - f3[0][1].loglik:.0f} / "
                                                               f"{f4[-1][1].loglik - f3[-1][1].loglik:.0f}"}
pd.DataFrame(rows).T
""")

md(r"""
## 2. Against the 3-factor model

RMSE ratio, Svensson ÷ 3-factor model with the same error model
(Diebold–Mariano, two-sided stars; < 1 means the fourth factor helps).
""")

code(r"""
def after(c, h):
    return y[c].loc[pd.Timestamp(FIRST_ORIGIN[c]) + pd.offsets.MonthEnd(h):]

def vs(c, bench, h, names, field="cell"):
    t = dm_table({n: fc[c][n][h] for n in [bench] + names}, bench, after(c, h), h, SHOW[c])
    return t.xs(field, axis=1, level=1).T

for c in CCYS:
    t = pd.concat({f"h = {h}": pd.DataFrame({p: vs(c, BASE_OF[p], h, [p]).loc[p] for p in SV}).T for h in H}, axis=1)
    display(t.style.set_caption(f"{c}: Svensson ÷ 3-factor").set_table_styles(CAP))
""")

md(r"""
## 3. Against forwards and the random walk
""")

code(r"""
SHOWN = ["yields", "yields + AR(1)"] + SV + ["combination A, inverse MSE"]
for c in CCYS:
    t = pd.concat({**{f"÷ forwards, h = {h}": vs(c, "forwards", h, SHOWN) for h in H},
                   **{f"÷ RW, h = {h}": vs(c, "random walk", h, SHOWN, "ratio").astype(float).round(2) for h in H}},
                  axis=1)
    display(t.style.set_caption(f"{c}").set_table_styles(CAP))
""")

md(r"""
## 4. Model confidence set over everything

The 90% MCS over every model in notebooks 02–09 (pooled squared-error
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
KEY = ["yields", "yields + AR(1)"] + SV + ["combination A, inverse MSE", "random walk", "forwards"]
pd.DataFrame({(c, f"h = {h}"): mcs[(c, h)].loc[KEY, "all"] for c in CCYS for h in H}).round(2)
""")

md(r"""
## 5. Takeaways

""" + TAKEAWAYS)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/09_svensson.ipynb")
