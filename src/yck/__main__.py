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
    b.add_argument("--accuracy", action="store_true",
                   help="the step 7b grid for every --ccy: yields only and policy rate only, each with every "
                        "shifting endpoint and AR(1) measurement-error model (README 7.6)")
    b.add_argument("--anchor", action="store_true",
                   help="step 7b item 4: yields only and policy rate only (plain and AR(1) errors), tilted to "
                        "the forward curve at 3m-12m")
    b.add_argument("--shadow", action="store_true",
                   help="step 7b item 6: smooth shadow-rate DNS (yields only), plug-in and exact-mean forecasts, "
                        "plain and with AR(1) errors")
    b.add_argument("--rwprior", action="store_true",
                   help="step 7b item 5: Minnesota random-walk prior on A (tau 0.03, 0.1, 0.3), yields only and "
                        "policy rate, plain and with AR(1) errors")
    b.add_argument("--svensson", action="store_true",
                   help="step 7b item 8: 4-factor dynamic Svensson (second curvature peaking at 9 or 90 months), "
                        "yields only, plain and with AR(1) errors")
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


def _backtest(a) -> int:
    from .eval.grid import (STANDARD_GRID, accuracy_grid, anchor_grid, run_grid, rwprior_grid, shadow_grid,
                            svensson_grid)

    if a.svensson:
        grid = svensson_grid
    elif a.rwprior:
        grid = rwprior_grid
    elif a.shadow:
        grid = shadow_grid
    elif a.anchor:
        grid = anchor_grid
    elif a.accuracy:
        grid = accuracy_grid
    elif a.all:
        grid = STANDARD_GRID
    else:
        grid = [(tuple(a.macro_vars) if a.macro else None, l, t) for l in a.lam for t in a.shrink]
    run_grid(a.ccy, a.data_dir, a.config, jobs=a.jobs or None, use_cache=not a.no_cache, grid=grid,
             est_start=a.est_start, first_origin=a.first_origin, every=a.every, floor=a.floor or None,
             starts=a.starts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
