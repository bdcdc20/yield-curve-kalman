"""Generates notebooks/07_shadow_rate.ipynb (run from the repo root, then execute with nbconvert)."""
import sys
from pathlib import Path

import nbformat as nbf

# The takeaways are written after reading the executed notebook's outputs, and kept beside this script.
TAKEAWAYS_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "takeaways" / "nb07.md"
TAKEAWAYS = TAKEAWAYS_FILE.read_text()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 07 · A smooth shadow-rate model for the zero-lower-bound years

In 2009–2022 short rates sat at or below zero in all three currencies, for
13 years in GBP. The linear DNS model has no notion of a floor. It forecasts
a short rate that drifts below the bound, and it treats a curve pinned at the
bound like any other curve. README §7.6 flagged this regime problem for EUR
and GBP (roadmap step 7b item 6).

The smooth shadow-rate DNS of Opschoor & van der Wel (2025) keeps the
Nelson–Siegel factors and their VAR, but reads each observed yield through
a smooth maximum:

y(τ) = r_LB + κ f((s(τ) − r_LB)/κ),  f(x) = xΦ(x) + φ(x),  s(τ) = Λ(τ) x_t.

- **Far above the bound** y = s, the linear model.
- **Near the bound** yields flatten towards r_LB while the shadow curve s
  can keep falling.
- **Identity behind the formula:** κ f(x/κ) = E[max(0, x + κZ)]. So κ acts
  like the volatility in a bond-option price.

**Choices, fixed before the run** (README §8, commit `a35a586`):

- κ = 1 (the paper's forecasting setting).
- r_LB = min(0, lowest yield in the estimation sample), learnt in real time.
- μ is fixed at the real-time sample mean, as for the linear model.
- Estimation is by maximum likelihood with the extended Kalman filter, from
  the linear fit of the same sample.
- Two forecasts: **plug-in** (the forecast shadow curve through the
  measurement function) and **mean** (the expectation over the shadow
  curve's forecast variance: the same formula with κ' = √(κ² + v)). Each is
  run plain and with AR(1) errors.

**One change after the grid was fixed, before any forecast was looked at.**
The first run imposed stationarity by rejecting explosive A, as the linear
code does. Diagnostics on the fits showed that this left L-BFGS-B stopping
up to 60 log-likelihood points short of the optimum: the shadow model's
level wants to sit near a unit root, and line searches kept hitting the
rejection wall. Restarts from perturbed points found the better optima. The
estimator now uses the Ansley–Kohn (1986) reparameterisation, as the paper
does (`yck.model.shadow.stationary_from_free`). Every parameter vector is
then a stationary VAR, and a single start reaches the restart optima to
within about 1 point. The first run's fits were discarded through a new
version in the cache key.
""")

code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

from yck.eval.backtest import dns_kalman_forecasts, dns_two_step_forecasts, random_walk_forecasts, forward_forecasts
from yck.eval.grid import (STANDARD_GRID, BACKTEST_DEFAULTS, accuracy_grid, anchor_grid, shadow_grid, plan_runs,
                           combination_forecasts, ensure_results, tag_of, _grid_for)
from yck.eval.stats import dm_table, mcs_table
from yck.model.statespace_dns import fit_dns_mle
from yck.model.shadow import fit_shadow_dns

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RESULTS, CONFIG, DATA = ROOT / "data/results", ROOT / "config/data_sources.yaml", ROOT / "data"
CCYS = ["USD", "EUR", "GBP"]
H = (1, 6, 12)
SHOW = {"USD": ["3m", "12m", "24m", "60m", "120m"], "EUR": ["3m", "12m", "24m", "60m", "120m"],
        "GBP": ["6m", "12m", "24m", "60m", "120m"]}
FIRST_ORIGIN = {c: BACKTEST_DEFAULTS[c][1] for c in CCYS}
ZLB = {"USD": ("2009", "2015"), "EUR": ("2014", "2022"), "GBP": ("2009", "2021")}  # policy rate at or near zero
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
CAP = [{"selector": "caption", "props": "font-weight: bold; text-align: left"}]

for g in (STANDARD_GRID, accuracy_grid, anchor_grid, shadow_grid):  # one grid run if anything is missing
    ensure_results(CCYS, DATA, CONFIG, grid=g)

def name_of(row):
    m, lam, tau, ep, err, anc, sr, *_ = tuple(row) + (None,) * (9 - len(row))
    base = "shadow" if sr else ("policy rate" if m == ("policy_rate",) else "full macro" if m else "yields")
    base += "" if lam == "fixed" else ", λ est."
    base += f", shrink {tau:g}" if tau is not None else ""
    base += f", endpoint {ep}" if ep else ""
    base += " + AR(1)" if err == "ar1" else " + AR(1) with mean" if err == "ar1c" else ""
    return base + (", anchored" if anc else "") + (f" ({sr})" if sr else "")

y, fc = {}, {}
for c in CCYS:
    est = BACKTEST_DEFAULTS[c][0]
    full = pd.read_parquet(ROOT / f"data/processed/curves_{c}.parquet")
    y[c] = full.loc[est:] if est else full
    fc[c] = {}
    for g in (STANDARD_GRID, accuracy_grid, anchor_grid, shadow_grid):
        for r in _grid_for(g, c):
            fc[c][name_of(r)] = {h: pd.read_parquet(RESULTS / f"fc_{c}_kalman_{tag_of(r)}_h{h}.parquet") for h in H}
    fc[c]["two-step AR(1)"] = dns_two_step_forecasts(y[c], H, FIRST_ORIGIN[c])
    fc[c]["two-step VAR(1)"] = dns_two_step_forecasts(y[c], H, FIRST_ORIGIN[c], kind="var1")
    fc[c]["random walk"] = random_walk_forecasts(y[c], H, FIRST_ORIGIN[c])
    fc[c]["forwards"] = forward_forecasts(pd.read_parquet(ROOT / f"data/processed/curves_ext_{c}.parquet"),
                                          y[c].columns, H, FIRST_ORIGIN[c])
    fc[c].update(combination_forecasts(c, DATA, CONFIG))
SHADOW = [name_of(r) for r in _grid_for(shadow_grid, "USD")]
LINEAR_OF = {s: ("yields + AR(1)" if "AR(1)" in s else "yields") for s in SHADOW}
pd.DataFrame({c: {"models compared": len(fc[c])} for c in CCYS})
""")

md(r"""
## 1. The fits

Each refit's lower bound and log-likelihood gain over the linear model on
the same sample. The gain is close to zero while the sample holds only
"normal" rates, and grows once the zero-rate years enter it. EUR's first
refit (2010) is the one case where the shadow model fits worse: κ and r_LB
are fixed, so it does not nest the linear model exactly.
""")

code(r"""
rows, gains = {}, {}
for c in CCYS:
    run = [kw for cc, tag, kw in plan_runs([c], DATA, CONFIG, grid=shadow_grid) if tag == "lamfixed_sr-plugin"][0]
    fits = []
    dns_kalman_forecasts(**run, horizons=(1,), cache_dir=RESULTS / "cache", fits=fits)
    g = pd.Series({t: f.loglik - f.linear_loglik for t, f in fits})
    gains[c] = g
    rows[c] = {"refits": len(fits), "r_LB first / last": f"{fits[0][1].lb:.2f} / {fits[-1][1].lb:.2f}",
               "log-lik gain, first refit": round(g.iloc[0], 1), "log-lik gain, last refit": round(g.iloc[-1], 1),
               "optimiser stopped on tolerance": int(sum(not f.converged for _, f in fits))}
display(pd.DataFrame(rows))
fig, ax = plt.subplots(figsize=(8, 3))
for c in CCYS:
    ax.plot(gains[c], marker="o", ms=3, lw=1, label=c)
ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("log-lik, shadow − linear"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Shadow-rate log-likelihood gain at each refit (sample up to the refit date)", loc="left", fontsize=10);
""")

md(r"""
## 2. The shadow short rate

The full-sample fits' filtered shadow short rate s(0) = level + slope,
against the observed short end. While the curve sits at the bound, the
shadow rate keeps moving and measures how far below zero policy is, in the
model's terms.
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.3), sharey=True)
for ax, c in zip(axes, CCYS):
    lin = fit_dns_mle(y[c], n_starts=6)
    sh = fit_shadow_dns(y[c], lin.params, lin.loglik)
    a = sh.filtered_states(y[c])
    ax.plot(y[c].index, a[:, 0] + a[:, 1], lw=1.2, label="shadow short rate")
    ax.plot(y[c][SHOW[c][0]], lw=0.9, color="k", alpha=0.7, label=f"observed {SHOW[c][0]}")
    ax.axhline(sh.lb, ls=":", color="C3", lw=1, label=f"r_LB = {sh.lb:.2f}%")
    ax.set_title(f"{c} (log-lik +{sh.loglik - lin.loglik:.0f} over linear)", loc="left", fontsize=10)
axes[0].set_ylabel("%"); axes[0].legend(fontsize=7, frameon=False); fig.tight_layout()
""")

md(r"""
## 3. Shadow against linear

RMSE ratio of each shadow variant to the linear model with the same error
model (Diebold–Mariano, two-sided stars; < 1 means the shadow model wins).
""")

code(r"""
def after(c, h):
    return y[c].loc[pd.Timestamp(FIRST_ORIGIN[c]) + pd.offsets.MonthEnd(h):]

def vs(c, bench, h, names, field="cell"):
    t = dm_table({n: fc[c][n][h] for n in [bench] + names}, bench, after(c, h), h, SHOW[c])
    return t.xs(field, axis=1, level=1).T

for c in CCYS:
    t = pd.concat({f"h = {h}": pd.DataFrame({s: vs(c, LINEAR_OF[s], h, [s]).loc[s] for s in SHADOW}).T for h in H},
                  axis=1)
    display(t.style.set_caption(f"{c}: shadow ÷ linear").set_table_styles(CAP))
""")

md(r"""
## 4. Against forwards and the random walk

The shadow variants and the shadow combination (set A with the shadow model
in place of the linear one), next to the linear model and combination A
from notebook 06.
""")

code(r"""
SHOWN = ["yields", "yields + AR(1)"] + SHADOW + ["combination A, inverse MSE", "combination A-shadow, equal",
                                                 "combination A-shadow, inverse MSE"]
for c in CCYS:
    t = pd.concat({**{f"÷ forwards, h = {h}": vs(c, "forwards", h, SHOWN) for h in H},
                   **{f"÷ RW, h = {h}": vs(c, "random walk", h, SHOWN, "ratio").astype(float).round(2) for h in H}},
                  axis=1)
    display(t.style.set_caption(f"{c}").set_table_styles(CAP))
""")

md(r"""
## 5. Bias in the zero-rate years

Mean 12-month-ahead forecast error (forecast − realised, bp) for targets in
the years the policy rate sat at or near the bound.
""")

code(r"""
bias = {}
for c in CCYS:
    z0, z1 = ZLB[c]
    for n in ["yields", "shadow (plugin)", "shadow (mean)", "random walk", "forwards"]:
        bias[(c, f"{z0}–{z1}", n)] = {("short end (3m; GBP 6m)" if m == SHOW[c][0] else m):
                                       100 * (fc[c][n][12][m] - y[c][m]).loc[z0:z1].mean() for m in SHOW[c]}
pd.DataFrame(bias).T.round(0)
""")

md(r"""
## 6. Model confidence set over everything

The 90% MCS over every model in notebooks 02–07 (squared-error loss pooled
over the maturities shown), with the p-values of the shadow models next to
the linear ones and the benchmarks.
""")

code(r"""
SHORT = "3m / 6m (GBP)"
mcs = {(c, h): mcs_table({n: f[h] for n, f in fc[c].items()}, after(c, h), h, SHOW[c], n_boot=2000)
               .rename(columns={SHOW[c][0]: SHORT})
       for c in CCYS for h in H}
summary = {(c, f"h = {h}"): {col: f"{int((t[col] >= 0.10).sum())} / {int(t[col].notna().sum())}" for col in t.columns}
           for (c, h), t in mcs.items()}
display(pd.DataFrame(summary).T.style.set_caption("Models in the 90% MCS / models compared").set_table_styles(CAP))
KEY = ["yields", "yields + AR(1)"] + SHADOW + ["combination A, inverse MSE", "combination A-shadow, inverse MSE",
                                               "random walk", "forwards"]
pd.DataFrame({(c, f"h = {h}"): mcs[(c, h)].loc[KEY, "all"] for c in CCYS for h in H}).round(2)
""")

md(r"""
## 7. Takeaways

""" + TAKEAWAYS)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/07_shadow_rate.ipynb")
