"""Command line entry point: `python -m yck fetch | probe-ecb | backtest`."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=value lines) so the FRED key never lives in code."""
    import os
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def main(argv: list[str] | None = None) -> int:
    _load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(prog="yck")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download public data and build monthly panels")
    f.add_argument("--only", nargs="*", choices=["USD", "EUR", "GBP"], help="limit to these currencies")
    f.add_argument("--refresh", action="store_true", help="re-download instead of using data/raw cache")
    f.add_argument("--config", type=Path, default=ROOT / "config" / "data_sources.yaml")
    f.add_argument("--data-dir", type=Path, default=ROOT / "data")
    pr = sub.add_parser("probe-ecb", help="list ECB series matching a wildcard key, with last date")
    pr.add_argument("flow")
    pr.add_argument("key", nargs="?", default="", help="omit to list dataflows whose id/name matches FLOW")
    b = sub.add_parser("backtest", help="recursive out-of-sample Kalman DNS forecasts (cached, parallel)")
    b.add_argument("--ccy", nargs="*", default=["USD"], choices=["USD", "EUR", "GBP"])
    b.add_argument("--lam", nargs="*", default=["fixed", "est"], choices=["fixed", "est"],
                   help="lambda fixed at 0.0609 and/or estimated")
    b.add_argument("--floor", type=float, default=1.0, help="measurement-noise sd floor in bp (0 = none)")
    b.add_argument("--starts", type=int, default=6, help="starting points per re-estimation")
    b.add_argument("--every", type=int, default=12, help="re-estimate every N months")
    b.add_argument("--jobs", type=int, default=0, help="parallel processes (0 = all cores)")
    b.add_argument("--est-start", help="first month of the estimation sample (default: per currency)")
    b.add_argument("--first-origin", help="first forecast origin (default: per currency)")
    b.add_argument("--no-cache", action="store_true", help="refit everything instead of reusing data/results/cache")
    b.add_argument("--macro", action="store_true", help="yields + macro model with real-time macro panels (README 6.4)")
    b.add_argument("--macro-vars", nargs="*", default=["real_activity", "inflation", "policy_rate"],
                   choices=["real_activity", "inflation", "policy_rate"], help="macro variables to include")
    b.add_argument("--shrink", type=float, nargs="*", default=[None],
                   help="shrinkage tau(s) for the factor<->macro cross terms (yields + macro only)")
    b.add_argument("--all", action="store_true",
                   help="the standard grid for every --ccy: yields only (both lambda), full macro, policy rate "
                        "only, shrinkage 0.01-0.3; all refits share one pool, longest first")
    b.add_argument("--config", type=Path, default=ROOT / "config" / "data_sources.yaml")
    b.add_argument("--data-dir", type=Path, default=ROOT / "data")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if a.cmd == "fetch":
        from .data.build import run
        ok = run(a.config, a.data_dir, a.only, a.refresh)
        print(f"\nReport: {a.data_dir / 'processed' / 'coverage.md'}")
        return 0 if ok else 1
    if a.cmd == "probe-ecb":
        import pandas as pd
        from .data.sources import ecb_dimensions, list_ecb_dataflows, probe_ecb
        with pd.option_context("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 200):
            if a.key == "?":
                out = ecb_dimensions(a.flow)
            else:
                out = probe_ecb(a.flow, a.key) if a.key else list_ecb_dataflows(a.flow)
            print(out.to_string(index=False))
        return 0
    if a.cmd == "backtest":
        return _backtest(a)
    return 2


# Defaults from README section 7.1: USD forecasts from 2000 (ALFRED vintages);
# EUR and GBP from 2010, so EUR has at least 5 years of estimation data.
# GBP estimation starts in 1993, with inflation targeting (Oct 1992): from 1985
# the Bank Rate disinflation pushes A to a unit root, where the stationary prior
# breaks down (README section 7.4), and UK CPI only starts in 1989 anyway.
_BACKTEST_DEFAULTS = {"USD": ("1985-01-31", "2000-01-31"), "EUR": (None, "2010-01-31"), "GBP": ("1993-01-31", "2010-01-31")}


ALL_VARS = ("real_activity", "inflation", "policy_rate")
STANDARD_GRID = (  # (macro variables or None, lambda mode, shrinkage tau): what README section 7.4 reports
    [(None, "fixed", None), (None, "est", None), (ALL_VARS, "fixed", None), (("policy_rate",), "fixed", None)]
    + [(ALL_VARS, "fixed", tau) for tau in (0.01, 0.03, 0.1, 0.3)])


def _backtest(a) -> int:
    import os
    import time

    import pandas as pd
    from .data.realtime import RealTimeMacro
    from .eval.backtest import dns_kalman_forecasts, prefit

    jobs = a.jobs or os.cpu_count() or 1
    out_dir = a.data_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = None if a.no_cache else out_dir / "cache"
    floor = a.floor or None
    if a.all:
        grid = STANDARD_GRID
    else:
        grid = [(tuple(a.macro_vars) if a.macro else None, l, t) for l in a.lam for t in a.shrink]

    # 1. Plan every backtest (currency x model variant).
    runs = []
    for ccy in a.ccy:
        est_start, first_origin = _BACKTEST_DEFAULTS[ccy]
        est_start, first_origin = a.est_start or est_start, a.first_origin or first_origin
        y = pd.read_parquet(a.data_dir / "processed" / f"curves_{ccy}.parquet")
        macros = {}
        for mvars, lam_mode, tau in grid:
            if mvars is not None and mvars not in macros:
                macros[mvars] = RealTimeMacro.load(ccy, a.data_dir, a.config, mvars)
            mtag = "" if mvars is None else ("macro_" if len(mvars) == 3 else "macro-" + "+".join(mvars) + "_")
            tag = (mtag + (f"shrink{tau:g}_" if tau is not None else "")
                   + f"lam{'est' if lam_mode == 'est' else 'fixed'}" + ("" if floor else "_nofloor"))
            runs.append((ccy, tag, dict(yields=y, est_start=est_start, first_origin=first_origin,
                                        reestimate_every=a.every, estimate_lambda=lam_mode == "est",
                                        h_floor_bp=floor, n_starts=a.starts, macro=macros.get(mvars),
                                        shrink_tau=tau)))

    # 2. All their re-estimations in one pool, longest first (skipped without a cache).
    t_all = time.perf_counter()
    if cache_dir is not None:
        print(f"== {len(runs)} backtests ({', '.join(a.ccy)}): estimating all refits", flush=True)
        prefit([kw for _, _, kw in runs], jobs, cache_dir, verbose=True)

    # 3. Forecasts, from the cached fits.
    for ccy, tag, kw in runs:
        fits, t0 = [], time.perf_counter()
        fc = dns_kalman_forecasts(**kw, jobs=1 if cache_dir is not None else jobs, cache_dir=cache_dir, fits=fits)
        for h, df in fc.items():
            df.to_parquet(out_dir / f"fc_{ccy}_kalman_{tag}_h{h}.parquet")
        pd.DataFrame([{"origin": t, "loglik": f.loglik, "lam": f.params.lam, "converged": f.converged,
                       **{f"A{i}{j}": f.params.A[i, j] for i in range(f.params.k) for j in range(f.params.k)},
                       **{f"mu{i}": f.params.mu[i] for i in range(f.params.k)}}
                      for t, f in fits]).to_csv(out_dir / f"params_{ccy}_kalman_{tag}.csv", index=False)
        print(f"   {ccy} {tag}: {len(fits)} fits, forecasts in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"== all done in {time.perf_counter() - t_all:.0f}s -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
