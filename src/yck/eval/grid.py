"""The standard grid of Kalman backtests, run as one job (CLI `backtest --all` and the notebooks).

Each currency x model variant is one backtest; together they are 24
backtests and ~490 re-estimations. `run_grid` estimates all of them in one
process pool, longest first (`backtest.prefit`), then writes each backtest's
forecasts and parameters from the fit cache. `ensure_results` does that only
for currencies whose result files are missing, so a notebook on a fresh clone
triggers one ~1.5 min run instead of a pool per backtest.

`accuracy_grid` is the roadmap step 7b grid (shifting endpoints x AR(1)
measurement errors on the two best models), fixed before it was run.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import yaml

from yck.data.realtime import RealTimeMacro
from yck.eval.backtest import (dns_kalman_forecasts, dns_two_step_forecasts, forward_forecasts, prefit,
                               random_walk_forecasts)
from yck.eval.combine import combine
from yck.model.adjust import ewma_endpoint, survey_endpoint
from yck.model.macro_factors import CombinedMacro, RealTimeFactors

# USD forecasts from 2000 (ALFRED vintages); EUR and GBP from 2010, so EUR has at
# least 5 years of estimation data. GBP estimation starts in 1993 with inflation
# targeting (Oct 1992); UK CPI only starts in 1989 anyway.
BACKTEST_DEFAULTS = {"USD": ("1985-01-31", "2000-01-31"), "EUR": (None, "2010-01-31"),
                     "GBP": ("1993-01-31", "2010-01-31")}
ALL_VARS = ("real_activity", "inflation", "policy_rate")
STANDARD_GRID = (  # (macro variables or None, lambda mode, shrinkage tau): what README section 7.4 reports
    [(None, "fixed", None), (None, "est", None), (ALL_VARS, "fixed", None), (("policy_rate",), "fixed", None)]
    + [(ALL_VARS, "fixed", tau) for tau in (0.01, 0.03, 0.1, 0.3)])
HORIZONS = (1, 6, 12)

# Roadmap step 7b (README sections 7.6 and 8), fixed before any of it was run: the two best models so far,
# each with every endpoint available for the currency and every measurement-error model. All are reported.
ACCURACY_MODELS = (None, ("policy_rate",))  # yields only; yields + policy rate (lambda fixed, no shrinkage)
ENDPOINTS = {"USD": (None, "ewma", "bond10", "cpi10"), "EUR": (None, "ewma", "hicp_lt"), "GBP": (None, "ewma")}
ERROR_MODELS = (None, "ar1", "ar1c")


def accuracy_grid(ccy: str) -> list[tuple]:
    """Step 7b items 1-2 for one currency: (macro vars, lambda mode, tau, endpoint, error model) tuples."""
    return [(m, "fixed", None, e, err) for m in ACCURACY_MODELS for e in ENDPOINTS[ccy] for err in ERROR_MODELS]


# Step 7b items 3-4, fixed before they were run (README section 8).
def anchor_grid(ccy: str) -> list[tuple]:
    """Item 4: the two models, plain and with AR(1) errors, tilted to the forwards at 3m-12m."""
    return [(m, "fixed", None, None, err, "fwd") for m in ACCURACY_MODELS for err in (None, "ar1")]


COMBINATION_SETS = {  # item 3: which forecasts are averaged
    "A": ("yields + AR(1)", "two-step VAR(1)", "forwards", "random walk"),
    "B": ("yields + AR(1)", "policy rate + AR(1)", "two-step VAR(1)", "forwards", "random walk"),
    "A-shadow": ("shadow + AR(1)", "two-step VAR(1)", "forwards", "random walk"),  # item 6
}


# Step 7b item 6, fixed before it was run (README section 8).
def shadow_grid(ccy: str) -> list[tuple]:
    """Smooth shadow-rate DNS, yields only: plug-in and exact-mean forecasts, each plain and with AR(1) errors."""
    return [(None, "fixed", None, None, err, None, mode) for mode in ("plugin", "mean") for err in (None, "ar1")]
COMBINATION_WEIGHTS = ("equal", "inv_mse")


# Step 7b item 5, fixed before it was run (README section 8).
RW_TAUS = (0.03, 0.1, 0.3)


def rwprior_grid(ccy: str) -> list[tuple]:
    """Minnesota random-walk prior on A (tau in RW_TAUS) for yields only and policy rate, plain and with AR(1)."""
    return [(m, "fixed", None, None, err, None, None, t) for m in ACCURACY_MODELS for t in RW_TAUS
            for err in (None, "ar1")]


# Step 7b item 8, fixed before it was run (README section 8): a second curvature factor (dynamic Svensson)
# whose loading peaks at 9 months (short end) or 90 months (long end); lambda * tau = 1.7933 at the peak.
SVENSSON_LAM2 = {"short": 1.7933 / 9, "long": 1.7933 / 90}


def svensson_grid(ccy: str) -> list[tuple]:
    """4-factor dynamic Svensson, yields only, for each second-curvature decay, plain and with AR(1) errors."""
    return [(None, "fixed", None, None, err, None, None, None, l2) for l2 in SVENSSON_LAM2.values()
            for err in (None, "ar1")]


# Step 7b item 7, fixed before it was run (README section 8): macro factors from large public panels.
FACTOR_SETS = (("factor1",), ("factor1", "factor2"), ("policy_rate", "factor1", "factor2"))


def factor_grid(ccy: str) -> list[tuple]:
    """Yields + 1 or 2 panel factors, and yields + policy rate + 2 factors, plain and with AR(1) errors."""
    return [(m, "fixed", None, None, err) for m in FACTOR_SETS for err in (None, "ar1")]


def load_macro(ccy: str, mvars, data_dir: Path, config: Path, start=None):
    """Real-time macro source for a variable set: named series (`RealTimeMacro`), panel factors
    ('factor1', 'factor2', ...: `RealTimeFactors`), or both side by side."""
    named = tuple(v for v in mvars if not v.startswith("factor"))
    r = sum(v.startswith("factor") for v in mvars)
    parts = []
    if named:
        parts.append(RealTimeMacro.load(ccy, data_dir, config, named))
    if r:
        parts.append(RealTimeFactors.load(ccy, data_dir, config, r, start=start))
    return parts[0] if len(parts) == 1 else CombinedMacro(*parts)


def run_tag(mvars, lam_mode: str, tau, floor=1.0, endpoint=None, error_ar=None, anchor=None, shadow=None,
            rw_tau=None, lam2=None) -> str:
    """File tag of one backtest, e.g. 'macro-policy_rate_ep-ewma_lamfixed_err-ar1c'
    (fc_{CCY}_kalman_{tag}_h{h}.parquet). Without an endpoint, error model or anchor the tags are those of step 7."""
    mtag = "" if mvars is None else ("macro_" if tuple(mvars) == ALL_VARS else "macro-" + "+".join(mvars) + "_")
    return (mtag + (f"shrink{tau:g}_" if tau is not None else "") + (f"rw{rw_tau:g}_" if rw_tau is not None else "")
            + (f"sv{lam2:.4f}_" if lam2 is not None else "")
            + (f"ep-{endpoint}_" if endpoint else "")
            + f"lam{'est' if lam_mode == 'est' else 'fixed'}" + ("" if floor else "_nofloor")
            + (f"_err-{error_ar}" if error_ar else "") + (f"_anc-{anchor}" if anchor else "")
            + (f"_sr-{shadow}" if shadow else ""))


def load_endpoint(name: str, ccy: str, yields: pd.DataFrame, data_dir: Path, config: Path) -> pd.Series:
    """Real-time endpoint series on the curve's months: 'ewma', or a survey column of surveys_{CCY}.parquet."""
    if name == "ewma":
        return ewma_endpoint(yields)
    survey = pd.read_parquet(Path(data_dir) / "processed" / f"surveys_{ccy}.parquet")[name]
    known = yaml.safe_load(Path(config).read_text()).get("survey_known_month", 2)
    return survey_endpoint(survey, yields.index, known)


def _grid_for(grid, ccy: str) -> list[tuple]:
    """Grid entries as 9-tuples (macro vars, lambda mode, tau, endpoint, error model, anchor, shadow, rw_tau,
    lam2); `grid` is a list of 3- to 9-tuples, or a function of the currency."""
    rows = grid(ccy) if callable(grid) else grid
    return [tuple(r) + (None,) * (9 - len(r)) for r in rows]


def plan_runs(ccys, data_dir: Path, config: Path, grid=STANDARD_GRID, est_start=None, first_origin=None,
              every: int = 12, floor=1.0, starts: int = 6) -> list[tuple[str, str, dict]]:
    """(currency, tag, keyword arguments for `dns_kalman_forecasts`) for every backtest in the grid."""
    runs = []
    for ccy in ccys:
        start_default, origin_default = BACKTEST_DEFAULTS[ccy]
        es, fo = est_start or start_default, first_origin or origin_default
        y = pd.read_parquet(Path(data_dir) / "processed" / f"curves_{ccy}.parquet")
        macros, endpoints = {}, {}  # built once per currency, shared by the variants (macro panels memoised)
        ext = None
        for mvars, lam_mode, tau, ep, err, anc, sr, rw, l2 in _grid_for(grid, ccy):
            if mvars is not None and mvars not in macros:
                macros[mvars] = load_macro(ccy, mvars, data_dir, config, start=es)
            if ep is not None and ep not in endpoints:
                endpoints[ep] = load_endpoint(ep, ccy, y, data_dir, config)
            if anc is not None and ext is None:
                ext = pd.read_parquet(Path(data_dir) / "processed" / f"curves_ext_{ccy}.parquet")
            runs.append((ccy, run_tag(mvars, lam_mode, tau, floor, ep, err, anc, sr, rw, l2),
                         dict(yields=y, est_start=es, first_origin=fo, reestimate_every=every,
                              estimate_lambda=lam_mode == "est", h_floor_bp=floor, n_starts=starts,
                              macro=macros.get(mvars), shrink_tau=tau, endpoint=endpoints.get(ep), error_ar=err,
                              anchor=ext if anc == "fwd" else None, shadow=sr, rw_tau=rw, lam2=l2)))
    return runs


def run_grid(ccys, data_dir: Path, config: Path, jobs: int | None = None, use_cache: bool = True,
             verbose: bool = True, **plan_kw) -> None:
    """Estimate every refit of the grid in one longest-first pool, then write forecasts and parameters."""
    jobs = jobs or os.cpu_count() or 1
    out_dir = Path(data_dir) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache" if use_cache else None
    runs = plan_runs(ccys, data_dir, config, **plan_kw)
    t_all = time.perf_counter()
    if cache_dir is not None:
        if verbose:
            print(f"== {len(runs)} backtests ({', '.join(ccys)}): estimating all refits", flush=True)
        prefit([kw for _, _, kw in runs], jobs, cache_dir, verbose=verbose)
    for ccy, tag, kw in runs:
        fits, t0 = [], time.perf_counter()
        # Everything is cached after prefit, so reading fits needs no process pool.
        fc = dns_kalman_forecasts(**kw, horizons=HORIZONS, jobs=1 if cache_dir is not None else jobs,
                                  cache_dir=cache_dir, fits=fits)
        for h, df in fc.items():
            df.to_parquet(out_dir / f"fc_{ccy}_kalman_{tag}_h{h}.parquet")
        pd.DataFrame([{"origin": t, "loglik": f.loglik, "lam": f.params.lam, "converged": f.converged,
                       **{f"A{i}{j}": f.params.A[i, j] for i in range(f.params.k) for j in range(f.params.k)},
                       **{f"mu{i}": f.params.mu[i] for i in range(f.params.k)}}
                      for t, f in fits]).to_csv(out_dir / f"params_{ccy}_kalman_{tag}.csv", index=False)
        if verbose:
            print(f"   {ccy} {tag}: {len(fits)} fits, forecasts in {time.perf_counter() - t0:.1f}s", flush=True)
    if verbose:
        print(f"== all done in {time.perf_counter() - t_all:.0f}s -> {out_dir}", flush=True)


def ensure_results(ccys, data_dir: Path, config: Path, jobs: int | None = None, grid=STANDARD_GRID) -> None:
    """Run `grid` (default: the standard grid) for every currency in `ccys` whose result files are not all present."""
    out_dir = Path(data_dir) / "results"
    missing = [c for c in ccys
               if not all((out_dir / f"fc_{c}_kalman_{tag_of(r)}_h{h}.parquet").exists()
                          for r in _grid_for(grid, c) for h in HORIZONS)]
    if missing:
        run_grid(missing, data_dir, config, jobs=jobs, grid=grid)


def tag_of(row) -> str:
    """run_tag of one grid entry (3- to 6-tuple) with the default noise floor."""
    r = tuple(row) + (None,) * (9 - len(row))
    return run_tag(*r[:3], 1.0, *r[3:])


def combination_forecasts(ccy: str, data_dir: Path, config: Path, horizons=HORIZONS, sets=None) -> dict[str, dict]:
    """Items 3 and 6: every combination in COMBINATION_SETS (or `sets`) x COMBINATION_WEIGHTS, {name: {h: DataFrame}}.

    Components come from the step 7b results (yields / policy rate with AR(1)
    errors), the two-step VAR(1), forwards and the random walk, all from the
    currency's first origin. Runs the needed backtests first if missing.
    """
    ensure_results([ccy], data_dir, config, grid=accuracy_grid)
    ensure_results([ccy], data_dir, config, grid=shadow_grid)
    est, fo = BACKTEST_DEFAULTS[ccy]
    y = pd.read_parquet(Path(data_dir) / "processed" / f"curves_{ccy}.parquet")
    y_est = y.loc[est:] if est else y
    ext = pd.read_parquet(Path(data_dir) / "processed" / f"curves_ext_{ccy}.parquet")
    res = Path(data_dir) / "results"
    load = lambda tag: {h: pd.read_parquet(res / f"fc_{ccy}_kalman_{tag}_h{h}.parquet") for h in horizons}
    comp = {"yields + AR(1)": load(tag_of((None, "fixed", None, None, "ar1"))),
            "policy rate + AR(1)": load(tag_of((("policy_rate",), "fixed", None, None, "ar1"))),
            "shadow + AR(1)": load(tag_of((None, "fixed", None, None, "ar1", None, "plugin"))),
            "two-step VAR(1)": dns_two_step_forecasts(y_est, horizons, fo, kind="var1"),
            "forwards": forward_forecasts(ext, y.columns, horizons, fo),
            "random walk": random_walk_forecasts(y_est, horizons, fo)}
    return {f"combination {k}, {w.replace('inv_mse', 'inverse MSE')}":
            {h: combine({n: comp[n][h] for n in members}, y, h, w) for h in horizons}
            for k, members in COMBINATION_SETS.items() if sets is None or k in sets for w in COMBINATION_WEIGHTS}
