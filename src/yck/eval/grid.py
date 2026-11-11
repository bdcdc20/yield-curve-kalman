"""The standard grid of Kalman backtests, run as one job (CLI `backtest --all` and the notebooks).

Each currency x model variant is one backtest; together they are 24
backtests and ~490 re-estimations. `run_grid` estimates all of them in one
process pool, longest first (`backtest.prefit`), then writes each backtest's
forecasts and parameters from the fit cache. `ensure_results` does that only
for currencies whose result files are missing, so a notebook on a fresh clone
triggers one ~1.5 min run instead of a pool per backtest.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from yck.data.realtime import RealTimeMacro
from yck.eval.backtest import dns_kalman_forecasts, prefit

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


def run_tag(mvars, lam_mode: str, tau, floor=1.0) -> str:
    """File tag of one backtest, e.g. 'macro-policy_rate_lamfixed' (fc_{CCY}_kalman_{tag}_h{h}.parquet)."""
    mtag = "" if mvars is None else ("macro_" if len(mvars) == 3 else "macro-" + "+".join(mvars) + "_")
    return (mtag + (f"shrink{tau:g}_" if tau is not None else "")
            + f"lam{'est' if lam_mode == 'est' else 'fixed'}" + ("" if floor else "_nofloor"))


def plan_runs(ccys, data_dir: Path, config: Path, grid=STANDARD_GRID, est_start=None, first_origin=None,
              every: int = 12, floor=1.0, starts: int = 6) -> list[tuple[str, str, dict]]:
    """(currency, tag, keyword arguments for `dns_kalman_forecasts`) for every backtest in the grid."""
    runs = []
    for ccy in ccys:
        start_default, origin_default = BACKTEST_DEFAULTS[ccy]
        es, fo = est_start or start_default, first_origin or origin_default
        y = pd.read_parquet(Path(data_dir) / "processed" / f"curves_{ccy}.parquet")
        macros = {}  # one panel builder per variable set, shared by its variants (memoised by origin)
        for mvars, lam_mode, tau in grid:
            if mvars is not None and mvars not in macros:
                macros[mvars] = RealTimeMacro.load(ccy, data_dir, config, mvars)
            runs.append((ccy, run_tag(mvars, lam_mode, tau, floor),
                         dict(yields=y, est_start=es, first_origin=fo, reestimate_every=every,
                              estimate_lambda=lam_mode == "est", h_floor_bp=floor, n_starts=starts,
                              macro=macros.get(mvars), shrink_tau=tau)))
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


def ensure_results(ccys, data_dir: Path, config: Path, jobs: int | None = None) -> None:
    """Run the standard grid for every currency in `ccys` whose result files are not all present."""
    out_dir = Path(data_dir) / "results"
    missing = [c for c in ccys
               if not all((out_dir / f"fc_{c}_kalman_{run_tag(m, l, t)}_h{h}.parquet").exists()
                          for m, l, t in STANDARD_GRID for h in HORIZONS)]
    if missing:
        run_grid(missing, data_dir, config, jobs=jobs)
