"""Download every source in config/data_sources.yaml and write tidy monthly panels.

Outputs (data/processed/):
    curves_{CCY}.parquet    month-end zero yields (%), columns '3m'..'120m'
    curves_ext_{CCY}.parquet  same curves at every maturity the forward benchmark needs
    macro_{CCY}.parquet     monthly real_activity, inflation, policy_rate (latest vintage)
    vintages_USD.parquet    ALFRED real-time history for series with vintages: true
    surveys_{CCY}.parquet   quarterly survey expectations (SPF), indexed by survey quarter
    coverage.md             date ranges, gaps and cross-checks for every series
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import sources as src
from .curves import label_months, month_end, svensson_yield

log = logging.getLogger("yck.data")


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


# --------------------------------------------------------------------------- curves
def forward_maturities(grid: list[int], horizons: list[int]) -> list[int]:
    """Maturities needed to compute forwards F_t(h, tau) for every grid tau and horizon h.

    F_t(h, tau) needs y_t(h) and y_t(tau + h), so this is the grid plus each
    h plus each tau + h.
    """
    return sorted(set(grid) | set(horizons) | {m + h for m in grid for h in horizons})


def build_curve(ccy: str, spec: dict, grid: list[int], raw: Path, refresh: bool,
                horizons: list[int] = ()) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Returns the month-end curve on the grid, the extended curve for forwards, and report notes."""
    notes: list[str] = []
    grid_full = forward_maturities(grid, list(horizons))
    years = [m / 12 for m in grid_full]

    if spec["source"] == "fed_gsw":
        daily = svensson_yield(src.fetch_gsw(spec["url"], raw, refresh), years)
        for bill, col in zip(spec.get("short_end_check", []), (3, 6)):
            try:
                b = month_end(src.fetch_fred(bill, raw, refresh).to_frame())[bill]
                c = month_end(daily)[col / 12].reindex(b.index)
                d = (c - b).dropna()
                notes.append(f"GSW {col}m minus {bill} (discount basis, so not like-for-like): "
                             f"mean {d.mean():.2f}pp, mean abs {d.abs().mean():.2f}pp, n={len(d)}")
            except Exception as e:  # check only; never block the build
                notes.append(f"short-end check {bill} failed: {e}")

    elif spec["source"] == "ecb_yc":
        keys = "+".join(spec["params"])
        params = src.fetch_ecb(spec.get("flow", "YC"), f"{spec['key_prefix']}.{keys}", raw,
                               f"ecb_yc_params_{ccy}.csv", refresh)
        daily = svensson_yield(params, years)
        check_keys = "+".join(spec.get("check_rates", []))
        if check_keys:
            pub = src.fetch_ecb(spec.get("flow", "YC"), f"{spec['key_prefix']}.{check_keys}", raw,
                                f"ecb_yc_check_{ccy}.csv", refresh)
            for col in pub.columns:
                yrs = int(col.removeprefix("SR_").removesuffix("Y"))
                mine = svensson_yield(params.reindex(pub.index), [yrs])[yrs]
                diff = (mine - pub[col]).abs().dropna()
                notes.append(f"ECB reconstruction vs published {col}: max abs diff {diff.max():.4f}pp (n={len(diff)})")

    elif spec["source"] == "boe_glc":
        wide = src.fetch_boe_glc(spec["url"], spec["sheets"], raw, refresh)
        # Exact maturities only: points the BoE doesn't publish stay NaN.
        daily = wide.reindex(columns=[float(m) for m in grid_full])
        daily.columns = years
        missing = [m for m in grid if float(m) not in set(wide.columns)]
        if missing:
            notes.append(f"BoE publishes no exact maturity for {missing} months (left as NaN)")
        missing_fwd = [m for m in grid_full if m not in grid and float(m) not in set(wide.columns)]
        if missing_fwd:
            notes.append(f"Forward benchmark: BoE publishes no exact maturity for {missing_fwd} months, "
                         "so forwards needing them are missing")
    else:
        raise ValueError(f"unknown curve source {spec['source']}")

    # The grid panel is built on its own columns, so it is identical to what it
    # would be without the extra maturities (month_end drops all-NaN rows).
    monthly = month_end(daily[[m / 12 for m in grid]])
    monthly.columns = label_months(grid)
    ext = month_end(daily)
    ext.columns = label_months(grid_full)
    return monthly, ext, notes


# --------------------------------------------------------------------------- macro
def _fetch_macro_series(spec: dict, name: str, raw: Path, refresh: bool) -> pd.Series:
    s = spec["source"]
    if s == "fred":
        return src.fetch_fred(spec["id"], raw, refresh)
    if s == "ecb":
        df = src.fetch_ecb(spec["flow"], spec["key"], raw, f"ecb_{spec['flow']}_{name}.csv", refresh)
        return df.iloc[:, 0]
    if s == "ons":
        return src.fetch_ons(spec["uri"], raw, name, refresh)
    if s == "boe_iadb":
        return src.fetch_iadb(spec["code"], raw, refresh)
    raise ValueError(f"unknown macro source {s}")


def _fetch_survey(spec: dict, name: str, raw: Path, refresh: bool) -> pd.Series:
    if spec["source"] == "spf":
        return src.fetch_spf(spec["url"], spec["column"], raw, refresh)
    if spec["source"] == "ecb":
        return src.fetch_ecb(spec["flow"], spec["key"], raw, f"ecb_{spec['flow']}_{name}.csv", refresh).iloc[:, 0]
    raise ValueError(f"unknown survey source {spec['source']}")


def to_monthly(s: pd.Series, transform: str, today: pd.Timestamp | None = None) -> pd.Series:
    """Month-end series (last observation of each month), optionally as % change on 12 months earlier.

    Fast path for data that are already one observation per month (FRED,
    ALFRED, ECB, ONS monthly series): month-ends come from numpy month
    arithmetic and the y/y change from a 12-month array shift. The generic path
    (`month_end` + `asfreq`) groups by month and rebuilds a calendar in pure
    Python, which made real-time panels (hundreds of vintages) slow. Both give
    identical output (tested). As in `month_end`, the unfinished current
    calendar month is dropped.
    """
    s = s.dropna()
    months = s.index.values.astype("datetime64[M]")
    if len(months) > 1 and bool(np.all(np.diff(months).astype(np.int64) > 0)):
        current = np.datetime64((today or pd.Timestamp.today()).to_period("M").start_time, "M")
        keep = months < current
        months, vals = months[keep], s.to_numpy(float)[keep]
        if transform == "yoy" and len(months):
            full = np.arange(months[0], months[-1] + np.timedelta64(1, "M"))
            arr = np.full(full.size, np.nan)
            arr[(months - months[0]).astype(np.int64)] = vals
            prev = np.full(full.size, np.nan)
            prev[12:] = arr[:-12]
            months, vals = full, 100 * (arr / prev - 1)
        unit = getattr(s.index, "unit", "ns")  # keep the input's resolution, as the generic path does
        ends = ((months + np.timedelta64(1, "M")).astype("datetime64[D]") - np.timedelta64(1, "D")).astype(f"datetime64[{unit}]")
        return pd.Series(vals, index=pd.DatetimeIndex(ends, name="date"), name=s.name if s.name is not None else 0)
    m = month_end(s.to_frame(), today).iloc[:, 0]
    if transform == "yoy":
        m = m.asfreq("ME")
        m = 100 * (m / m.shift(12) - 1)
    return m


def build_macro(ccy: str, spec: dict, raw: Path, refresh: bool) -> tuple[pd.DataFrame, dict]:
    cols, vint = {}, {}
    for var, s in spec.items():
        name = f"{ccy}_{var}"
        cols[var] = to_monthly(_fetch_macro_series(s, name, raw, refresh), s.get("transform", "level"))
        if s.get("vintages"):
            vint[var] = src.fetch_alfred(s["id"], raw, refresh)
    return pd.DataFrame(cols), vint


# --------------------------------------------------------------------------- report
def coverage(name: str, df: pd.DataFrame) -> list[str]:
    out = [f"### {name}", "", "| series | first | last | months | missing in range |", "|---|---|---|---|---|"]
    for c in df.columns:
        s = df[c]
        v = s.dropna()
        if v.empty:
            out.append(f"| {c} | – | – | 0 | – |")
            continue
        rng = s.loc[v.index[0]:v.index[-1]]
        out.append(f"| {c} | {v.index[0]:%Y-%m} | {v.index[-1]:%Y-%m} | {len(v)} | {rng.isna().sum()} |")
    return out + [""]


def run(config_path: Path, data_dir: Path, only: list[str] | None = None, refresh: bool = False) -> bool:
    cfg = load_config(config_path)
    raw, out = data_dir / "raw", data_dir / "processed"
    out.mkdir(parents=True, exist_ok=True)
    grid = cfg["maturities_months"]
    report = ["# Data coverage", ""]
    ok = True

    for ccy, spec in cfg["curves"].items():
        if only and ccy not in only:
            continue
        try:
            curve, ext, notes = build_curve(ccy, spec, grid, raw, refresh, cfg.get("forecast_horizons_months", []))
            curve.to_parquet(out / f"curves_{ccy}.parquet")
            ext.to_parquet(out / f"curves_ext_{ccy}.parquet")
            report += coverage(f"{ccy} curve ({spec['source']})", curve) + [f"- {n}" for n in notes] + [""]
            log.info("%s curve: %s to %s", ccy, curve.index[0].date(), curve.index[-1].date())
        except Exception as e:
            ok = False
            log.error("%s curve failed: %s", ccy, e)
            report += [f"### {ccy} curve FAILED", "", "```", traceback.format_exc(limit=3), "```", ""]

    for ccy, spec in cfg["macro"].items():
        if only and ccy not in only:
            continue
        for var, s in spec.items():
            try:
                macro, vint = build_macro(ccy, {var: s}, raw, refresh)
            except Exception as e:
                ok = False
                log.error("%s %s failed: %s", ccy, var, e)
                report += [f"### {ccy} {var} FAILED ({s})", "", "```", traceback.format_exc(limit=3), "```", ""]
                continue
            path = out / f"macro_{ccy}.parquet"
            prev = pd.read_parquet(path) if path.exists() else pd.DataFrame()
            prev = prev.drop(columns=[var], errors="ignore")
            pd.concat([prev, macro], axis=1).sort_index().to_parquet(path)
            for v, df in vint.items():
                vp = out / f"vintages_{ccy}.parquet"
                old = pd.read_parquet(vp) if vp.exists() else pd.DataFrame()
                if not old.empty:
                    old = old[old["series_id"] != df["series_id"].iloc[0]]
                pd.concat([old, df], ignore_index=True).to_parquet(vp)
                report.append(f"- {ccy} {v}: {df['realtime_start'].nunique()} ALFRED vintages, "
                              f"first {df['realtime_start'].min():%Y-%m-%d}")

    for ccy, spec in cfg.get("surveys", {}).items():
        if only and ccy not in only:
            continue
        cols = {}
        for var, s in spec.items():
            try:
                cols[var] = _fetch_survey(s, f"{ccy}_{var}", raw, refresh)
            except Exception as e:
                ok = False
                log.error("%s survey %s failed: %s", ccy, var, e)
                report += [f"### {ccy} survey {var} FAILED ({s})", "", "```", traceback.format_exc(limit=3), "```", ""]
        if cols:
            df = pd.DataFrame(cols).sort_index()
            df.to_parquet(out / f"surveys_{ccy}.parquet")
            report += coverage(f"{ccy} surveys (quarterly, by survey quarter)", df)

    for ccy in cfg["macro"]:
        p = out / f"macro_{ccy}.parquet"
        if p.exists() and (not only or ccy in only):
            report += coverage(f"{ccy} macro (latest vintage)", pd.read_parquet(p))

    (out / "coverage.md").write_text("\n".join(report) + "\n")
    return ok
