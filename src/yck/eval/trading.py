"""Trading evaluation: do the forecasts make money against the forward curve? (README sections 7.5, 8 step 8)

**Instruments.** A position that receives the forward rate F_t(h, tau) for a
tau-month zero starting in h months (an FRA, a futures contract or a
forward-starting swap leg) earns, per unit of DV01, at t + h

    delta_{t+h}(tau) = 100 (F_t(h, tau) - y_{t+h}(tau))   basis points.

This is exact for zero-coupon bonds: the h-month excess return of an
(h + tau)-month zero is tau/12 (F_t(h, tau) - y_{t+h}(tau)). So realised P&L
needs only the curves and the forward benchmark, both indexed by **target
date** t + h, like every forecast in `yck.eval`. Spreads combine maturities
with DV01 weights (`LEGS`): a 2s10s steepener, a 2s5s10s butterfly, and
cross-currency spreads of the same maturity.

**Signal.** A forecast y_hat_{t+h|t} implies the expected P&L
s_t = 100 (F_t(h, tau) - y_hat): receive when positive. The random walk
(y_hat = y_t) gives s = F - y_t, carry plus roll-down, the benchmark that
earns the term premium without any model.

**Sizing** (`positions`), decided at the origin t with nothing dated after it:

- `sign`: one unit of DV01 in the direction of the signal.
- `scaled`: s / sigma, clipped to [-2, 2], with sigma the standard deviation
  of the instrument's realised P&L over targets known at t (at least
  `MIN_HISTORY` of them; no position before that).

**Holding.** Each month a new tranche is opened and held h months, so with
h > 1 there are h overlapping tranches, each with 1/h of the DV01 budget
(Jegadeesh & Titman 1993). A tranche's P&L is booked in the month it closes,
so a month's P&L is 1/h of the closing tranche's; the totals equal those of
the running portfolio, and figures are per unit of DV01 run in total. Tests
use autocovariances up to lag h - 1, since neighbouring tranches overlap.

**Costs.** `cost_bp` per unit of DV01 per leg for a round trip, charged on
each tranche (|x| times the sum of absolute leg weights).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from yck.eval.stats import long_run_var_of_mean

MIN_HISTORY = 24  # realised P&L observations needed before `scaled` sizing starts
CLIP = 2.0
LEGS = {  # DV01 weights per maturity: + receives (long duration)
    "short": None,  # filled per currency with its short-end maturity (3m; GBP 6m)
    "2y": {"24m": 1.0},
    "5y": {"60m": 1.0},
    "10y": {"120m": 1.0},
    "2s10s steepener": {"24m": 1.0, "120m": -1.0},
    "2s5s10s fly": {"60m": 2.0, "24m": -1.0, "120m": -1.0},
}


def legs_for(strategy: str, short_column: str) -> dict:
    return {short_column: 1.0} if strategy == "short" else LEGS[strategy]


def realised_pnl(forward: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """delta = 100 (F - y) in bp per unit DV01 received, by target date and maturity."""
    idx = forward.index.intersection(actual.index)
    return 100.0 * (forward.loc[idx] - actual.loc[idx, forward.columns])


def signal(forward: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """s = 100 (F - y_hat) in bp, by target date: the P&L the forecast expects from receiving the forward."""
    idx = forward.index.intersection(forecast.index)
    return 100.0 * (forward.loc[idx] - forecast.loc[idx, forward.columns])


def combine_legs(frame: pd.DataFrame, legs: dict) -> pd.Series:
    """DV01-weighted sum over legs (NaN if any leg is missing)."""
    return sum(w * frame[m] for m, w in legs.items())


def positions(sig: pd.Series, pnl: pd.Series, h: int, sizing: str = "scaled",
              min_history: int = MIN_HISTORY) -> pd.Series:
    """Position (DV01 units) for each target date T, decided at origin T - h.

    `scaled` uses the sd of realised P&L for targets <= T - h: a calendar shift
    by h months of the expanding sd.
    """
    if sizing == "sign":
        return np.sign(sig)
    if sizing != "scaled":
        raise ValueError(f"sizing must be 'sign' or 'scaled', not {sizing!r}")
    cal = pd.date_range(min(sig.index.min(), pnl.index.min()), max(sig.index.max(), pnl.index.max()), freq="ME")
    p = pnl.reindex(cal)
    sd = p.expanding(min_periods=min_history).std()
    known = sd.shift(h)  # at origin T - h, realised P&L is known for targets up to T - h
    return (sig / known.reindex(sig.index)).clip(-CLIP, CLIP)


def strategy_pnl(sig: pd.Series, pnl: pd.Series, h: int, sizing: str = "scaled", cost_bp: float = 0.5,
                 leg_weight: float = 1.0) -> pd.DataFrame:
    """Per target date (the month a tranche closes): position x of that tranche, and its gross P&L, cost and
    net P&L as a share of a portfolio running one unit of DV01 in total (so divided by h)."""
    x = positions(sig, pnl, h, sizing)
    idx = x.dropna().index.intersection(pnl.dropna().index)
    x, d = x.loc[idx], pnl.loc[idx]
    gross = x * d / h
    cost = cost_bp * x.abs() * leg_weight / h
    return pd.DataFrame({"x": x, "gross": gross, "cost": cost, "net": gross - cost})


def summary(pnl: pd.Series, h: int) -> dict:
    """Annualised mean (bp per unit DV01), volatility, Sharpe ratio, t-statistic, hit rate, max drawdown.

    Each monthly entry is 1/h of an h-month tranche's P&L (`strategy_pnl`), so:

    - mean: 12 x the monthly mean (the totals match the running portfolio);
    - volatility and Sharpe: those of an h-month return, annualised by
      sqrt(12 / h): vol = sqrt(12 h) sd(monthly entries), Sharpe = sqrt(12 / h)
      mean / sd. Annualising by sqrt(12) would treat each h-month return as a
      one-month return and overstate the Sharpe ratio by sqrt(h);
    - t-statistic of the mean: autocovariances up to lag h - 1, because
      neighbouring tranches overlap (as in the DM test).
    """
    p = pnl.dropna()
    if len(p) < 12:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "t", "hit", "max_dd", "months")}
    se = np.sqrt(long_run_var_of_mean(p.to_numpy(), h))
    cum = p.cumsum()
    return {"mean": 12 * p.mean(), "vol": np.sqrt(12 * h) * p.std(), "sharpe": np.sqrt(12 / h) * p.mean() / p.std(),
            "t": p.mean() / se if se > 0 else np.nan, "hit": float((p > 0).mean()),
            "max_dd": float((cum.cummax() - cum).max()), "months": len(p)}


def difference_test(a: pd.Series, b: pd.Series, h: int) -> dict:
    """Mean of a - b on common months, annualised, with its HAC t-statistic (lags up to h - 1)."""
    d = (a - b).dropna()
    se = np.sqrt(long_run_var_of_mean(d.to_numpy(), h))
    return {"diff": 12 * d.mean(), "t": d.mean() / se if se > 0 else np.nan, "months": len(d)}


# --------------------------------------------------------------------------- the step 8 grid (fixed before it was run)
# Forecast files behind each model signal (fc_{CCY}_kalman_{tag}_h{h}.parquet); `None` = built elsewhere.
BEST_MODEL = {"USD": "sv0.1993_lamfixed_err-ar1", "EUR": "lamfixed_err-ar1_sr-plugin", "GBP": "sv0.1993_lamfixed_err-ar1"}
SIGNALS = ("carry (random walk)", "yields", "best pure model", "policy rate + AR(1)", "full macro",
           "combination A (inverse MSE)")
TAGS = {"yields": "lamfixed", "policy rate + AR(1)": "macro-policy_rate_lamfixed_err-ar1", "full macro": "macro_lamfixed"}
STRATEGIES = ("short", "2y", "5y", "10y", "2s10s steepener", "2s5s10s fly")
CROSS = (("USD", "EUR"), ("USD", "GBP"), ("EUR", "GBP"))  # receive the first currency's 10y, pay the second's
TRADE_HORIZONS = {"USD": (1, 6, 12), "EUR": (1, 6, 12), "GBP": (6, 12)}  # GBP has no 1-month forwards (no 1m yield)
SIZINGS = ("sign", "scaled")
BASE_COST_BP = 0.5
COST_GRID_BP = (0.0, 0.25, 0.5, 1.0, 2.0)
ALWAYS_RECEIVE = "always receive (long duration)"


def trading_inputs(ccy: str, data_dir, config) -> dict:
    """Realised P&L and every signal, by horizon: {"pnl": {h: frame}, "signals": {name: {h: frame}}, "short": col}."""
    from pathlib import Path

    from yck.eval.backtest import forward_forecasts, random_walk_forecasts
    from yck.eval.grid import BACKTEST_DEFAULTS, combination_forecasts

    est, fo = BACKTEST_DEFAULTS[ccy]
    proc, res = Path(data_dir) / "processed", Path(data_dir) / "results"
    y = pd.read_parquet(proc / f"curves_{ccy}.parquet")
    ext = pd.read_parquet(proc / f"curves_ext_{ccy}.parquet")
    hs = TRADE_HORIZONS[ccy]
    fwd = forward_forecasts(ext, y.columns, hs, fo)
    load = lambda tag: {h: pd.read_parquet(res / f"fc_{ccy}_kalman_{tag}_h{h}.parquet") for h in hs}
    fc = {"carry (random walk)": random_walk_forecasts(y, hs, fo),
          "best pure model": load(BEST_MODEL[ccy]), **{k: load(t) for k, t in TAGS.items()},
          "combination A (inverse MSE)": combination_forecasts(ccy, data_dir, config, hs, sets=("A",))[
              "combination A, inverse MSE"]}
    return {"pnl": {h: realised_pnl(fwd[h], y) for h in hs},
            "signals": {n: {h: signal(fwd[h], f[h]) for h in hs} for n, f in fc.items()},
            "short": "6m" if ccy == "GBP" else "3m"}


def _evaluate(pnl: pd.Series, sigs: dict, h: int, leg_weight: float, cost_bp: float, with_receive: bool) -> dict:
    """Net P&L per signal and sizing on the common sample where every one is defined."""
    runs = {}
    for name, s in sigs.items():
        for sz in SIZINGS:
            runs[(name, sz)] = strategy_pnl(s, pnl, h, sz, cost_bp, leg_weight)["net"]
    if with_receive:  # buy-and-hold duration: one unit received every month
        ones = pd.Series(1.0, index=pnl.index)
        runs[(ALWAYS_RECEIVE, "sign")] = strategy_pnl(ones, pnl, h, "sign", cost_bp, leg_weight)["net"]
    common = None
    for r in runs.values():
        common = r.dropna().index if common is None else common.intersection(r.dropna().index)
    return {k: v.loc[common] for k, v in runs.items()}


def run_trading(ccys, data_dir, config, cost_bp: float = BASE_COST_BP) -> tuple[pd.DataFrame, dict]:
    """Every strategy in the grid: a summary table and the monthly net P&L series {(ccy, h, strategy): {...}}."""
    inputs = {c: trading_inputs(c, data_dir, config) for c in ccys}
    rows, series = [], {}

    def record(ccy, h, strat, runs):
        series[(ccy, h, strat)] = runs
        carry = {sz: runs.get(("carry (random walk)", sz)) for sz in SIZINGS}
        for (name, sz), p in runs.items():
            row = {"ccy": ccy, "h": h, "strategy": strat, "signal": name, "sizing": sz, **summary(p, h)}
            base = carry.get(sz)
            if base is not None and name != "carry (random walk)" and name != ALWAYS_RECEIVE:
                d = difference_test(p, base, h)
                row.update({"vs_carry": d["diff"], "t_vs_carry": d["t"]})
            rows.append(row)

    for c in ccys:
        inp = inputs[c]
        for h in TRADE_HORIZONS[c]:
            for strat in STRATEGIES:
                legs = legs_for(strat, inp["short"])
                pnl = combine_legs(inp["pnl"][h], legs)
                sigs = {n: combine_legs(s[h], legs) for n, s in inp["signals"].items()}
                outright = len(legs) == 1
                record(c, h, strat, _evaluate(pnl, sigs, h, sum(abs(w) for w in legs.values()), cost_bp, outright))
    for a, b in CROSS:
        if a not in inputs or b not in inputs:
            continue
        for h in sorted(set(TRADE_HORIZONS[a]) & set(TRADE_HORIZONS[b])):
            pnl = inputs[a]["pnl"][h]["120m"] - inputs[b]["pnl"][h]["120m"]
            sigs = {n: inputs[a]["signals"][n][h]["120m"] - inputs[b]["signals"][n][h]["120m"] for n in SIGNALS}
            record(f"{a}-{b}", h, "10y spread", _evaluate(pnl, sigs, h, 2.0, cost_bp, False))
    return pd.DataFrame(rows), series
