"""Parsers for each public data source.

Every source has a pure `parse_*` function (bytes/text in, DataFrame out) that
is unit-tested offline, and a thin `fetch_*` wrapper that downloads and parses.
"""
from __future__ import annotations

import io
import logging
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .http import fetch_to_cache, get

log = logging.getLogger("yck.data")

SVENSSON_PARAMS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]


# --------------------------------------------------------------------------- Fed GSW
def parse_gsw(text: str) -> pd.DataFrame:
    """feds200628.csv: a block of notes, then a header row starting with 'Date'."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Date,"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), na_values=["NA", "", "-999.99"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df.index.name = "date"
    return df[SVENSSON_PARAMS].astype(float)


def fetch_gsw(url: str, raw_dir: Path, refresh: bool = False) -> pd.DataFrame:
    path = fetch_to_cache(url, raw_dir, "fed_gsw_feds200628.csv", refresh=refresh)
    return parse_gsw(path.read_text(errors="replace"))


# --------------------------------------------------------------------------- ECB SDMX
ECB_API = "https://data-api.ecb.europa.eu/service/data"


def parse_ecb_csv(text: str) -> pd.DataFrame:
    """ECB 'csvdata' output -> wide frame, one column per series key (last key segment)."""
    df = pd.read_csv(io.StringIO(text))
    df["series"] = df["KEY"].str.split(".").str[-1]
    df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
    wide = df.pivot_table(index="TIME_PERIOD", columns="series", values="OBS_VALUE", aggfunc="last")
    wide.index.name = "date"
    wide.columns.name = None
    return wide.sort_index()


def fetch_ecb(flow: str, key: str, raw_dir: Path, name: str, refresh: bool = False) -> pd.DataFrame:
    """Fetch one ECB series key. A '+' in the last segment is split into one request per series,
    because the ECB API times out on large multi-series daily requests."""
    head, _, last = key.rpartition(".")
    frames = []
    for part in last.split("+"):
        k = f"{head}.{part}" if head else part
        path = fetch_to_cache(f"{ECB_API}/{flow}/{k}", raw_dir, f"{name.removesuffix('.csv')}_{part}.csv",
                              params={"format": "csvdata"}, refresh=refresh, timeout=180)
        frames.append(parse_ecb_csv(path.read_text()))
    return pd.concat(frames, axis=1).sort_index()


def probe_ecb(flow: str, key: str) -> pd.DataFrame:
    """List every series matching a (possibly wildcarded) ECB key with its last observation.
    E.g. probe_ecb('ICP', 'M..N.000000.4.ANR') shows which euro-area code is still updated."""
    r = get(f"{ECB_API}/{flow}/{key}", params={"format": "csvdata", "lastNObservations": 1}, timeout=180)
    df = pd.read_csv(io.StringIO(r.text))
    cols = [c for c in ("KEY", "TIME_PERIOD", "OBS_VALUE", "TITLE") if c in df.columns]
    return df[cols].sort_values("TIME_PERIOD", ascending=False)


# --------------------------------------------------------------------------- BoE GLC
def parse_boe_sheet(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    """One sheet of a BoE GLC workbook (read with header=None).

    The layout is: a few title rows, a row whose first cell is 'years:' or
    'months:' followed by the maturities, then rows of date + yields.
    Returns yields with columns = maturity in months.
    """
    first = df.iloc[:, 0].astype(str).str.strip().str.lower()
    hdr = int(np.flatnonzero(first.str.startswith(unit))[0])
    mats = pd.to_numeric(df.iloc[hdr, 1:], errors="coerce")
    keep = mats.notna().to_numpy()
    body = df.iloc[hdr + 1:]
    dates = pd.to_datetime(body.iloc[:, 0], errors="coerce", format="mixed")
    body = body[dates.notna()]
    vals = body.iloc[:, 1:].loc[:, keep].apply(pd.to_numeric, errors="coerce")
    months = mats[keep].to_numpy(float) * (12.0 if unit == "years" else 1.0)
    vals.columns = np.round(months, 6)
    vals.index = pd.DatetimeIndex(dates[dates.notna()], name="date")
    return vals


def _pick_sheet(book: dict, names) -> str | None:
    names = [names] if isinstance(names, str) else list(names)
    return next((n for n in names if n in book), None)


def parse_boe_zip(blob: bytes, main_sheet, short_sheet) -> pd.DataFrame:
    """All workbooks in the zip, main + short-end sheets, merged on (date, maturity in months).

    main_sheet / short_sheet may be a name or a list of names (older workbooks
    call them '4. nominal spot curve' etc.).
    """
    frames = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith((".xlsx", ".xls")):
                log.info("BoE zip: skipping %s", name)
                continue
            book = pd.read_excel(io.BytesIO(zf.read(name)), sheet_name=None, header=None)
            log.info("BoE zip: %s sheets=%s", name, list(book))
            parts = []
            short_name, main_name = _pick_sheet(book, short_sheet), _pick_sheet(book, main_sheet)
            if short_name:
                parts.append(parse_boe_sheet(book[short_name], "months"))
            if main_name:
                parts.append(parse_boe_sheet(book[main_name], "years"))
            if not parts:
                log.warning("BoE zip: %s has neither %r nor %r", name, main_sheet, short_sheet)
            if parts:
                log.info("BoE zip: %s -> %s to %s", name, parts[-1].index.min().date(), parts[-1].index.max().date())
                # Short-end sheet wins where both have the same maturity.
                merged = parts[0]
                for p in parts[1:]:
                    merged = merged.combine_first(p)
                frames.append(merged)
    if not frames:
        raise ValueError("No GLC sheets found in BoE zip; check sheet names in config")
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.reindex(sorted(out.columns), axis=1)


def fetch_boe_glc(url: str, sheets: dict, raw_dir: Path, refresh: bool = False) -> pd.DataFrame:
    path = fetch_to_cache(url, raw_dir, "boe_glcnominalddata.zip", refresh=refresh)
    return parse_boe_zip(path.read_bytes(), sheets["main"], sheets["short_end"])


# --------------------------------------------------------------------------- BoE IADB
IADB_URL = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"


def parse_iadb(text: str, code: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().upper() for c in df.columns]
    s = pd.Series(pd.to_numeric(df[code], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df["DATE"], format="%d %b %Y"), name=code)
    s.index.name = "date"
    return s.sort_index()


def fetch_iadb(code: str, raw_dir: Path, refresh: bool = False) -> pd.Series:
    params = {"csv.x": "yes", "Datefrom": "01/Jan/1970", "Dateto": "now", "SeriesCodes": code,
              "CSVF": "TN", "UsingCodes": "Y", "VPD": "Y", "VFD": "N"}
    path = fetch_to_cache(IADB_URL, raw_dir, f"boe_iadb_{code}.csv", params=params, refresh=refresh)
    return parse_iadb(path.read_text(), code)


# --------------------------------------------------------------------------- ONS
ONS_GENERATOR = "https://www.ons.gov.uk/generator"
_ONS_MONTH = re.compile(r"^\d{4} [A-Z]{3}$")


def parse_ons(text: str, name: str) -> pd.Series:
    """ONS time-series CSV: metadata rows, then annual/quarterly/monthly blocks. Keep monthly."""
    df = pd.read_csv(io.StringIO(text), header=None, names=["period", "value"], dtype=str)
    df["period"] = df["period"].str.strip().str.strip('"')
    df = df[df["period"].str.match(_ONS_MONTH, na=False)]
    idx = pd.to_datetime(df["period"], format="%Y %b") + pd.offsets.MonthEnd(0)
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").to_numpy(), index=idx, name=name)
    s.index.name = "date"
    return s.sort_index()


def fetch_ons(uri: str, raw_dir: Path, name: str, refresh: bool = False) -> pd.Series:
    path = fetch_to_cache(ONS_GENERATOR, raw_dir, f"ons_{name}.csv",
                          params={"format": "csv", "uri": uri}, refresh=refresh)
    return parse_ons(path.read_text(), name)


# --------------------------------------------------------------------------- FRED / ALFRED
FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API = "https://api.stlouisfed.org/fred/series/observations"


def parse_fredgraph(text: str, series_id: str) -> pd.Series:
    """Latest-vintage CSV. Date column is 'observation_date' (current) or 'DATE' (older)."""
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    date_col = df.columns[0]
    s = pd.Series(pd.to_numeric(df[series_id], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df[date_col]), name=series_id)
    s.index.name = "date"
    return s.sort_index()


def fetch_fred(series_id: str, raw_dir: Path, refresh: bool = False) -> pd.Series:
    path = fetch_to_cache(FREDGRAPH, raw_dir, f"fred_{series_id}.csv", params={"id": series_id},
                          refresh=refresh)
    return parse_fredgraph(path.read_text(), series_id)


def parse_alfred(observations: list[dict], series_id: str) -> pd.DataFrame:
    """ALFRED observations (output_type=1) -> long frame: date, realtime_start, realtime_end, value."""
    df = pd.DataFrame(observations)[["date", "realtime_start", "realtime_end", "value"]]
    df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    # '9999-12-31' means "still current"; keep it as NaT-free by capping.
    df["realtime_end"] = pd.to_datetime(df["realtime_end"].replace("9999-12-31", "2262-04-11"))
    df.insert(0, "series_id", series_id)
    return df


def fetch_alfred(series_id: str, raw_dir: Path, refresh: bool = False,
                 observation_start: str = "1960-01-01") -> pd.DataFrame:
    cache = raw_dir / f"alfred_{series_id}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is not set; needed for ALFRED vintages")
    obs, offset = [], 0
    while True:
        r = get(FRED_API, params={
            "series_id": series_id, "api_key": key, "file_type": "json",
            "realtime_start": "1776-07-04", "realtime_end": "9999-12-31",
            "observation_start": observation_start, "output_type": 1,
            "limit": 100000, "offset": offset,
        }).json()
        batch = r.get("observations", [])
        obs.extend(batch)
        offset += len(batch)
        if not batch or offset >= int(r.get("count", 0)):
            break
    df = parse_alfred(obs, series_id)
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def as_of(vintages: pd.DataFrame, asof) -> pd.Series:
    """The series exactly as it was published on `asof` (what a forecaster saw that day)."""
    asof = pd.Timestamp(asof)
    v = vintages[(vintages["realtime_start"] <= asof) & (vintages["realtime_end"] >= asof)]
    return v.set_index("date")["value"].sort_index()
