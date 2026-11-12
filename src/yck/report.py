"""Figures for the README write-up (`python -m yck figures`), rebuilt from the pipeline's outputs.

Data and backtests are not in git, so the figures are generated from them and
committed as PNGs in docs/figures/. Every figure uses only out-of-sample
forecasts that the backtests already produced (or, for impulse responses, the
full-sample yields + macro fit, as in notebook 04).

Style: one light surface, recessive grid, thin marks, categorical colours in a
fixed order (the dataviz reference palette), benchmarks in muted grey, one
y-axis per panel.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CCYS = ("USD", "EUR", "GBP")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")  # fixed categorical order, never cycled
BENCH = "#898781"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "font.size": 9, "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 1.6, "figure.dpi": 150})


def _save(fig, out: Path, name: str, title: str):
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)


def fig_curves(data_dir: Path, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    for ax, c in zip(axes, CCYS):
        y = pd.read_parquet(data_dir / "processed" / f"curves_{c}.parquet").loc["1985":]
        m = pd.read_parquet(data_dir / "processed" / f"macro_{c}.parquet")["policy_rate"].loc[y.index[0]:]
        short = "6m" if c == "GBP" else "3m"
        ax.plot(m, color=BENCH, lw=1.0, label="policy rate")
        ax.plot(y[short], color=SERIES[0], lw=1.2, label="short end (3m; GBP 6m)")
        ax.plot(y["120m"], color=SERIES[1], lw=1.2, label="10y")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(c)
    axes[0].set_ylabel("% a year")
    axes[0].legend(loc="upper right")
    _save(fig, out, "curves.png", "Zero-coupon yields and the policy rate, month-end")


def fig_rmse(data_dir: Path, config: Path, out: Path):
    from yck.eval.backtest import forward_forecasts, random_walk_forecasts
    from yck.eval.grid import BACKTEST_DEFAULTS, combination_forecasts
    from yck.eval.stats import dm_table
    from yck.eval.trading import BEST_MODEL

    mats = {"USD": ["3m", "12m", "24m", "60m", "120m"], "EUR": ["3m", "12m", "24m", "60m", "120m"],
            "GBP": ["6m", "12m", "24m", "60m", "120m"]}
    hs = (1, 6, 12)
    fig, axes = plt.subplots(3, 3, figsize=(12, 7.5), sharex="col", sharey="row")
    for j, c in enumerate(CCYS):
        est, fo = BACKTEST_DEFAULTS[c]
        y = pd.read_parquet(data_dir / "processed" / f"curves_{c}.parquet")
        ext = pd.read_parquet(data_dir / "processed" / f"curves_ext_{c}.parquet")
        res = data_dir / "results"
        load = lambda t: {h: pd.read_parquet(res / f"fc_{c}_kalman_{t}_h{h}.parquet") for h in hs}
        fc = {"random walk": random_walk_forecasts(y.loc[est:] if est else y, hs, fo),
              "yields only": load("lamfixed"), "best model of step 7b": load(BEST_MODEL[c]),
              "policy rate + AR(1)": load("macro-policy_rate_lamfixed_err-ar1"),
              "combination A": combination_forecasts(c, data_dir, config, hs, sets=("A",))["combination A, inverse MSE"]}
        fw = forward_forecasts(ext, y.columns, hs, fo)
        for i, h in enumerate(hs):
            ax = axes[i, j]
            a = y.loc[pd.Timestamp(fo) + pd.offsets.MonthEnd(h):]
            t = dm_table({"forwards": fw[h], **{k: v[h] for k, v in fc.items()}}, "forwards", a, h, mats[c])
            ratio = t.xs("ratio", axis=1, level=1).astype(float)
            x = np.arange(len(mats[c]))
            ax.axhline(1.0, color=INK2, lw=0.9)
            for k, name in enumerate(fc):
                col = BENCH if name == "random walk" else SERIES[k - 1]
                ax.plot(x, ratio[name].to_numpy(), marker="o", ms=4, lw=1.2, color=col, label=name,
                        ls="--" if name == "random walk" else "-")
            ax.set_xticks(x, [m.replace("120m", "10y").replace("60m", "5y").replace("24m", "2y")
                              .replace("12m", "1y") for m in mats[c]])
            ax.set_ylim(0.7, 1.9 if h == 1 else 1.6)
            if j == 0:
                ax.set_ylabel(f"h = {h} month{'s' if h > 1 else ''}\nRMSE ÷ forwards")
            if i == 0:
                ax.set_title(f"{c} (origins from {fo[:4]})")
    axes[0, 2].legend(loc="upper right")
    fig.text(0.01, -0.01, "Below 1: more accurate than the forward curve. Values above the axis are clipped "
             "(EUR and GBP short end at h = 1).", color=INK2, fontsize=8)
    _save(fig, out, "rmse_vs_forwards.png", "Out-of-sample forecast accuracy relative to the forward curve")


def fig_irf(data_dir: Path, out: Path):
    from yck.model import irf
    from yck.model.statespace_dns import fit_dns_mle

    est = {"USD": "1985-01-31", "EUR": None, "GBP": "1993-01-31"}
    pairs = [("policy_rate", "curvature", "policy rate ← curvature shock"),
             ("level", "policy_rate", "level ← policy-rate shock"),
             ("slope", "policy_rate", "slope ← policy-rate shock")]
    fig, axes = plt.subplots(3, 3, figsize=(12, 7), sharex=True)
    for j, c in enumerate(CCYS):
        y = pd.read_parquet(data_dir / "processed" / f"curves_{c}.parquet")
        y = y.loc[est[c]:] if est[c] else y
        m = pd.read_parquet(data_dir / "processed" / f"macro_{c}.parquet")
        data = y.join(m[["real_activity", "inflation", "policy_rate"]], how="left")
        fit = fit_dns_mle(data, n_starts=6)
        d = irf.irf_draws(fit, data, horizon=60, n_draws=1000)
        lo, hi = d.band(0.9)
        names = fit.state_names
        hh = np.arange(d.point.shape[0])
        for i, (resp, shock, label) in enumerate(pairs):
            ax = axes[i, j]
            r, s = names.index(resp), names.index(shock)
            ax.fill_between(hh, 100 * lo[:, r, s], 100 * hi[:, r, s], color=SERIES[0], alpha=0.18, lw=0)
            ax.plot(hh, 100 * d.point[:, r, s], color=SERIES[0], lw=1.6)
            ax.axhline(0, color=INK2, lw=0.7)
            if i == 0:
                ax.set_title(c)
            if j == 0:
                ax.set_ylabel(f"{label}\n(bp)")
    for ax in axes[-1]:
        ax.set_xlabel("months after the shock")
    _save(fig, out, "impulse_responses.png",
          "Impulse responses of the yields + macro model (one-sd shocks, Cholesky, 90% bands)")


def fig_bias(data_dir: Path, out: Path):
    from yck.eval.backtest import forward_forecasts, random_walk_forecasts

    y = pd.read_parquet(data_dir / "processed" / "curves_EUR.parquet")
    ext = pd.read_parquet(data_dir / "processed" / "curves_ext_EUR.parquet")
    res = data_dir / "results"
    load = lambda t: pd.read_parquet(res / f"fc_EUR_kalman_{t}_h12.parquet")
    fc = {"fixed mean (step 7)": load("lamfixed"), "EWMA endpoint": load("ep-ewma_lamfixed"),
          "random-walk prior": load("rw0.03_lamfixed"), "shadow rate": load("lamfixed_sr-plugin"),
          "random walk": random_walk_forecasts(y, (12,), "2010-01-31")[12],
          "forwards": forward_forecasts(ext, y.columns, (12,), "2010-01-31")[12]}
    periods = [("2010", "2013"), ("2014", "2021"), ("2022", "2026")]
    vals = {n: [100 * (f["120m"] - y["120m"]).loc[a:b].mean() for a, b in periods] for n, f in fc.items()}
    fig, ax = plt.subplots(figsize=(10, 3.6))
    w = 0.13
    x = np.arange(len(periods))
    colours = list(SERIES[:4]) + [BENCH, INK2]
    for k, (n, v) in enumerate(vals.items()):
        ax.bar(x + (k - 2.5) * w, v, w * 0.92, color=colours[k], label=n,
               hatch="//" if n in ("random walk", "forwards") else None, edgecolor=SURFACE, linewidth=0.5)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, [f"{a}–{b}" for a, b in periods])
    ax.set_ylabel("mean error, bp\n(forecast − realised)")
    ax.legend(ncol=3, loc="lower left")
    _save(fig, out, "eur_bias.png", "EUR 10-year yield, 12-month-ahead mean forecast error by period")


def fig_pnl(data_dir: Path, config: Path, out: Path):
    from yck.eval.trading import ALWAYS_RECEIVE, run_trading

    _, series = run_trading(CCYS, data_dir, config)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=False)
    show = [("carry (random walk)", "scaled", "carry / roll-down", SERIES[0]),
            ("best pure model", "scaled", "best model", SERIES[1]),
            ("combination A (inverse MSE)", "scaled", "combination A", SERIES[3]),
            (ALWAYS_RECEIVE, "sign", "always receive", BENCH)]
    for ax, c in zip(axes, CCYS):
        runs = series[(c, 12, "10y")]
        for key, sz, label, col in show:
            p = runs[(key, sz)]
            ax.plot(p.cumsum(), color=col, lw=1.4 if key.startswith("carry") else 1.1, label=label,
                    ls="--" if key == ALWAYS_RECEIVE else "-")
        ax.axhline(0, color=INK2, lw=0.7)
        ax.set_title(c)
    axes[0].set_ylabel("cumulative net P&L\n(bp per unit of DV01)")
    axes[0].legend(loc="upper left")
    _save(fig, out, "pnl_10y.png", "Receiving the 10-year forward on each signal, 12-month tranches, after costs")


def make_figures(data_dir: Path, config: Path, out_dir: Path) -> list[Path]:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_curves(data_dir, out_dir)
    fig_rmse(data_dir, config, out_dir)
    fig_irf(data_dir, out_dir)
    fig_bias(data_dir, out_dir)
    fig_pnl(data_dir, config, out_dir)
    return sorted(out_dir.glob("*.png"))
