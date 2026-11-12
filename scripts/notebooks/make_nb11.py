"""Generates notebooks/11_trading.ipynb (run from the repo root, then execute with nbconvert)."""
import sys
from pathlib import Path

import nbformat as nbf

# The takeaways are written after reading the executed notebook's outputs, and kept beside this script.
TAKEAWAYS_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "takeaways" / "nb11.md"
TAKEAWAYS = TAKEAWAYS_FILE.read_text()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 11 · Trading evaluation: is any of it worth money against the forward curve?

Notebooks 02–10 measured forecasts by RMSE. A rates position already pays or
earns the forward rate, so its P&L depends on the realised rate **relative
to the forward**. This notebook turns every forecast into trades and asks
whether any of them beats the trade that needs no model at all: carry and
roll-down (README §7.5, §8 step 8).

**The P&L identity.** Receive the forward F_t(h, τ) on a τ-month zero
starting in h months (an FRA, a rate future or a forward-starting swap
leg). Per unit of DV01 it earns 100 (F_t(h, τ) − y_{t+h}(τ)) bp at t + h.
This is exact for zero-coupon bonds: the h-month excess return of an
(h + τ)-month zero is τ/12 (F − y_{t+h}(τ)). So the curves alone give the
P&L, and no licensed contract data is needed.

**Signal.** A forecast ŷ implies s = 100 (F − ŷ): receive when s > 0. The
random walk (ŷ = y_t) gives s = F − y_t, carry plus roll-down, which earns
the term premium.

**Grid** (`yck.eval.trading.run_trading`, fixed and committed before any
P&L was looked at, commit `42de439`):

- **Signals:**
  - carry (random walk);
  - yields only;
  - the best pure model of step 7b (USD/GBP: short-hump Svensson + AR(1);
    EUR: shadow-rate + AR(1));
  - policy rate + AR(1);
  - full macro (DRA);
  - combination A.

  Outright trades also get always-receive (buy-and-hold duration).
- **Strategies:**
  - outright short end (3m; GBP 6m), 2y, 5y, 10y;
  - a DV01-neutral 2s10s steepener and 2s5s10s butterfly;
  - cross-currency 10y spreads.
- **Holding periods.** h = 6 and 12 months (all currencies), and 1 month
  (USD, EUR; GBP has no 1-month forwards). A new tranche opens every month
  and is held h months, each with 1/h of the DV01 budget.
- **Sizing,** decided at the origin:
  - `sign`: one unit of DV01;
  - `scaled`: s / σ clipped to ±2, with σ the sd of the instrument's P&L
    realised by then.
- **Costs.** 0.5 bp per unit of DV01 per leg per round trip; sensitivity
  in §5.
- **Statistics.** Sharpe ratios are of h-month returns annualised by
  √(12/h). An earlier draft annualised by √12, overstating them √h-fold. It
  was caught and fixed before this notebook was written (commit after
  `42de439`). t-statistics use autocovariances up to lag h − 1.

**Limitations.** These are curve-implied returns, not contract returns:
there is no futures basis, roll, margin or financing spread. The GSW, ECB
and BoE curves are smoothed fits. A positive result would mean "worth
testing on contract data", not "ready to trade".
""")

code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display
pd.set_option("display.max_rows", 300)

from yck.eval.grid import ensure_results, STANDARD_GRID, accuracy_grid, shadow_grid, svensson_grid
from yck.eval.trading import (run_trading, difference_test, ALWAYS_RECEIVE, COST_GRID_BP, BASE_COST_BP)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RESULTS, CONFIG, DATA = ROOT / "data/results", ROOT / "config/data_sources.yaml", ROOT / "data"
CCYS = ["USD", "EUR", "GBP"]
for g in (STANDARD_GRID, accuracy_grid, shadow_grid, svensson_grid):
    ensure_results(CCYS, DATA, CONFIG, grid=g)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
CAP = [{"selector": "caption", "props": "font-weight: bold; text-align: left"}]
SHORT = {"carry (random walk)": "carry", "yields": "yields", "best pure model": "best model",
         "policy rate + AR(1)": "policy rate", "full macro": "full macro",
         "combination A (inverse MSE)": "combination A", ALWAYS_RECEIVE: "always receive"}
ORDER = ["always receive", "carry", "yields", "best model", "policy rate", "full macro", "combination A"]

tab, series = run_trading(CCYS, DATA, CONFIG)
tab["sig"] = tab["signal"].map(SHORT)
tab.to_csv(RESULTS / "trading_summary.csv", index=False)
pd.DataFrame({"strategy × signal × sizing rows": [len(tab)],
              "months per test, min–max": [f"{tab.months.min():.0f}–{tab.months.max():.0f}"]})
""")

md(r"""
## 1. What carry and the models earned

Cumulative net P&L (bp per unit of DV01 run) of the 10-year outright trade
at h = 12, scaled sizing, for carry, buy-and-hold, the best model and
combination A.
""")

code(r"""
fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex="col")
for j, c in enumerate(CCYS):
    for i, strat in enumerate(["10y", "2s10s steepener"]):
        ax, runs = axes[i, j], series[(c, 12, strat)]
        for name in ["carry (random walk)", "best pure model", "combination A (inverse MSE)", ALWAYS_RECEIVE]:
            key = (name, "sign") if name == ALWAYS_RECEIVE else (name, "scaled")
            if key in runs:
                ax.plot(runs[key].cumsum(), lw=1.2 if "carry" in name else 0.9, label=SHORT[name])
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{c}: {strat}, h = 12", loc="left", fontsize=10)
axes[0, 0].set_ylabel("cumulative bp per unit DV01"); axes[1, 0].set_ylabel("cumulative bp per unit DV01")
axes[0, 0].legend(fontsize=7, frameon=False); fig.tight_layout()
""")

md(r"""
## 2. Sharpe ratios, every strategy and signal

Net of 0.5 bp costs, scaled sizing. Buy-and-hold is shown for outright
trades only. The `sign` sizing tables follow below.
""")

code(r"""
for sz in ("scaled", "sign"):
    t = tab[(tab.sizing == sz) | (tab.signal == ALWAYS_RECEIVE)]
    out = pd.concat({f"h = {h}": t[t.h == h].pivot_table(index=["ccy", "strategy"], columns="sig", values="sharpe")
                     .reindex(columns=ORDER) for h in (1, 6, 12)}, axis=1)
    display(out.round(2).style.set_caption(f"Sharpe ratio, {sz} sizing").set_table_styles(CAP)
            .background_gradient(cmap="RdYlGn", vmin=-1, vmax=1, axis=None).format("{:.2f}", na_rep=""))
""")

md(r"""
## 3. Does any model beat carry?

Each model signal's mean net P&L minus carry's (bp per unit DV01 a year),
with its HAC t-statistic, scaled sizing. With this many comparisons some
|t| > 2 are expected by chance. The count below compares them with the
roughly 2.5% per tail that pure chance would give.
""")

code(r"""
m = tab[(tab.sizing == "scaled") & tab.t_vs_carry.notna()]
cell = m.assign(cell=m.apply(lambda r: f"{r.vs_carry:+.1f} ({r.t_vs_carry:+.1f})", axis=1))
display(cell.pivot_table(index=["ccy", "h", "strategy"], columns="sig", values="cell", aggfunc="first")
        .reindex(columns=ORDER[2:]).style.set_caption("Model − carry: annual bp (t)").set_table_styles(CAP))
for sz in ("scaled", "sign"):
    mm = tab[(tab.sizing == sz) & tab.t_vs_carry.notna()]
    print(f"{sz}: {len(mm)} comparisons; t > 2: {int((mm.t_vs_carry > 2).sum())}, t < -2: "
          f"{int((mm.t_vs_carry < -2).sum())} (about {0.025 * len(mm):.0f} each expected by chance)")
""")

md(r"""
## 4. The macro question: do macro signals add P&L?

The README asks whether macro information pays in P&L as well as in RMSE.
The table gives the policy-rate model and the full macro model, each minus
yields only: annual bp (t), scaled sizing.
""")

code(r"""
rows = {}
for (c, h, strat), runs in series.items():
    if ("yields", "scaled") not in runs:
        continue
    for name in ("policy rate + AR(1)", "full macro"):
        d = difference_test(runs[(name, "scaled")], runs[("yields", "scaled")], h)
        rows[(c, h, strat, SHORT[name])] = f"{d['diff']:+.1f} ({d['t']:+.1f})"
pd.Series(rows).unstack().rename_axis(["ccy", "h", "strategy"])
""")

md(r"""
## 5. Cost sensitivity

The Sharpe ratio of each signal's best strategy at h = 12 (by net Sharpe at
the base cost), and of carry on the same strategy, at costs from 0 to 2 bp
per unit of DV01 per leg.
""")

code(r"""
by_cost = {c: run_trading(CCYS, DATA, CONFIG, cost_bp=c)[0] for c in COST_GRID_BP}
base = tab[(tab.h == 12) & (tab.sizing == "scaled") & (tab.signal != ALWAYS_RECEIVE)]
rows = {}
for (ccy, name), g in base.groupby(["ccy", "signal"]):
    strat = g.loc[g.sharpe.idxmax(), "strategy"]
    for nm in (name, "carry (random walk)"):
        key = (ccy, SHORT[name], strat, SHORT[nm])
        rows[key] = {f"{c:g} bp": by_cost[c].query("ccy == @ccy and h == 12 and sizing == 'scaled' and "
                                                   "strategy == @strat and signal == @nm").sharpe.iloc[0]
                     for c in COST_GRID_BP}
pd.DataFrame(rows).T.round(2).rename_axis(["ccy", "signal's best", "strategy", "shown"])
""")

md(r"""
## 6. Takeaways

""" + TAKEAWAYS)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/11_trading.ipynb")
