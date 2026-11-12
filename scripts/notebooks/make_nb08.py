"""Generates notebooks/08_rw_prior.ipynb (run from the repo root, then execute with nbconvert)."""
import sys
from pathlib import Path

import nbformat as nbf

# The takeaways are written after reading the executed notebook's outputs, and kept beside this script.
TAKEAWAYS_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "takeaways" / "nb08.md"
TAKEAWAYS = TAKEAWAYS_FILE.read_text()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 08 · A random-walk prior on the factor dynamics

Notebooks 05–07 kept finding the same thing: the model reverts too fast
towards a historical mean, while the random walk, which does not revert at
all, is hard to beat at 6–12 months. Duffee (2011) finds that the most
useful restriction for forecasting is a random walk for the level, and
Carriero, Kapetanios & Marcellino (2012) shrink yield VARs towards
random-walk-like models. This notebook tests that idea (README §8 step 7b,
item 5).

**The prior** (`yck.model.statespace_dns.rw_prior`, fixed before the run,
commit `04a8463`). A Minnesota prior on the whole state VAR matrix A,
centred on the identity: every state a random walk.

- **Own lags:** N(1, τ²).
- **Cross lags:** N(0, (0.5 τ s_i/s_j)²), with s the two-step shock sds.
- **No prior on Q.**
- **Estimation:** maximised together with the likelihood (MAP).
  τ ∈ {0.03, 0.1, 0.3}: smaller τ gives forecasts closer to a random walk,
  larger τ the plain MLE. μ stays at the real-time sample mean; with A = I
  it drops out of the forecast.
- **Models:** yields only and yields + policy rate (λ fixed), each plain
  and with AR(1) errors.
""")


code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

from yck.eval.backtest import dns_kalman_forecasts, dns_two_step_forecasts, random_walk_forecasts, forward_forecasts
from yck.eval.grid import (STANDARD_GRID, BACKTEST_DEFAULTS, accuracy_grid, anchor_grid, shadow_grid, rwprior_grid,
                           plan_runs,
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

GRIDS = (STANDARD_GRID, accuracy_grid, anchor_grid, shadow_grid, rwprior_grid)
for g in GRIDS:  # one grid run if anything is missing
    ensure_results(CCYS, DATA, CONFIG, grid=g)

def name_of(row):
    m, lam, tau, ep, err, anc, sr, rw, *_ = tuple(row) + (None,) * (9 - len(row))
    base = "shadow" if sr else ("policy rate" if m == ("policy_rate",) else "full macro" if m else "yields")
    base += "" if lam == "fixed" else ", λ est."
    base += f", shrink {tau:g}" if tau is not None else ""
    base += f", RW prior {rw:g}" if rw is not None else ""
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
PRIOR = [name_of(r) for r in _grid_for(rwprior_grid, "USD")]
BASE_OF = {p: p.split(", RW prior")[0] + (" + AR(1)" if "AR(1)" in p else "") for p in PRIOR}
pd.DataFrame({c: {"models compared": len(fc[c])} for c in CCYS})
""")


md(r"""
## 1. What the prior does to A

The largest eigenvalue of A and the largest distance |A − I| across the
refits of each model. With τ = 0.03 every state is held close to a random
walk. With τ = 0.3 the estimates are close to the unrestricted ones.
""")

code(r"""
rows = {}
for c in CCYS:
    for m, mtag, k in [("yields", "", 3), ("policy rate", "macro-policy_rate_", 4)]:
        for t in (None,) + (0.03, 0.1, 0.3):
            d = pd.read_csv(RESULTS / f"params_{c}_kalman_{mtag}{'' if t is None else f'rw{t:g}_'}lamfixed.csv")
            A = d[[f"A{i}{j}" for i in range(k) for j in range(k)]].to_numpy().reshape(-1, k, k)
            rad = np.abs(np.linalg.eigvals(A)).max(axis=1)
            rows[(c, m, "none" if t is None else f"τ = {t}")] = {
                "max |eig A|, range over refits": f"{rad.min():.3f}–{rad.max():.4f}",
                "max |A − I|, mean over refits": round(float(np.abs(A - np.eye(k)).max(axis=(1, 2)).mean()), 2)}
pd.DataFrame(rows).T
""")

md(r"""
## 2. Against the same model without the prior

RMSE ratio, prior ÷ no prior, same error model (Diebold–Mariano, two-sided
stars; < 1 means the prior helps).
""")

code(r"""
def after(c, h):
    return y[c].loc[pd.Timestamp(FIRST_ORIGIN[c]) + pd.offsets.MonthEnd(h):]

def vs(c, bench, h, names, field="cell"):
    t = dm_table({n: fc[c][n][h] for n in [bench] + names}, bench, after(c, h), h, SHOW[c])
    return t.xs(field, axis=1, level=1).T

for c in CCYS:
    t = pd.concat({f"h = {h}": pd.DataFrame({p: vs(c, BASE_OF[p], h, [p]).loc[p] for p in PRIOR}).T for h in H},
                  axis=1)
    display(t.style.set_caption(f"{c}: RW prior ÷ no prior").set_table_styles(CAP))
""")

md(r"""
## 3. Against forwards and the random walk
""")

code(r"""
SHOWN = ["yields", "policy rate"] + PRIOR + ["combination A, inverse MSE"]
for c in CCYS:
    t = pd.concat({**{f"÷ forwards, h = {h}": vs(c, "forwards", h, SHOWN) for h in (6, 12)},
                   **{f"÷ RW, h = {h}": vs(c, "random walk", h, SHOWN) for h in (6, 12)}}, axis=1)
    display(t.style.set_caption(f"{c}").set_table_styles(CAP))
""")

md(r"""
## 4. Model confidence set over everything

The 90% MCS over every model in notebooks 02–08 (pooled squared-error
loss), with the p-values of the prior variants, the plain models and the
benchmarks.
""")

code(r"""
SHORT = "3m / 6m (GBP)"
mcs = {(c, h): mcs_table({n: f[h] for n, f in fc[c].items()}, after(c, h), h, SHOW[c], n_boot=2000)
               .rename(columns={SHOW[c][0]: SHORT})
       for c in CCYS for h in H}
summary = {(c, f"h = {h}"): {col: f"{int((t[col] >= 0.10).sum())} / {int(t[col].notna().sum())}" for col in t.columns}
           for (c, h), t in mcs.items()}
display(pd.DataFrame(summary).T.style.set_caption("Models in the 90% MCS / models compared").set_table_styles(CAP))
KEY = ["yields", "policy rate"] + PRIOR + ["combination A, inverse MSE", "random walk", "forwards"]
pd.DataFrame({(c, f"h = {h}"): mcs[(c, h)].loc[KEY, "all"] for c in CCYS for h in H}).round(2)
""")

md(r"""
## 5. Takeaways

""" + TAKEAWAYS)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/08_rw_prior.ipynb")
