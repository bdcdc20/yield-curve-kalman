"""Real-time macro panels: the macro data a forecaster actually had at each forecast origin.

Macro data are published with a lag and then revised, so the latest vintage
(macro_{CCY}.parquet) contains information that nobody had at the time. Using
it in a backtest is look-ahead bias. README section 7.3 sets the rules:

- **Series with ALFRED vintages (USD capacity utilisation and PCE prices):**
  at origin t, use the series exactly as published on t (`as_of`), and apply
  the transform (e.g. 12-month % change) within that vintage. Months not yet
  published are simply absent, so they become missing observations, which the
  Kalman filter handles. Nothing is forward-filled.
- **Series without vintages (EUR, GBP, and never-revised policy rates):** use
  the latest vintage, but blank the last `release_lag_months[var]` months
  before each origin. This approximates what was known at the time and is
  stated as a limitation.
- **Origins before a series' first vintage** (PCEPI vintages start 2000-08):
  use the earliest vintage, truncated with the release lag.
- **Withdrawn history.** During comprehensive NIPA revisions ALFRED records
  most of the PCE price index as missing ('.') until BEA republishes it on
  the new basis (PCEPI: 2003-12-23 to 2004-01-04, and 2009-08-04). On those
  days the series cannot be used, so the last complete vintage is used
  instead: the data a forecaster already had (`as_of_complete`).

Monthly data are indexed by the end of their reference month throughout.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .build import to_monthly
from .sources import as_of

VARIABLES = ("real_activity", "inflation", "policy_rate")


def as_of_complete(vintages: pd.DataFrame, asof) -> pd.Series:
    """The latest vintage published on or before `asof` that has no missing values.

    Normally this is `as_of(vintages, asof)`. If that vintage records values as
    missing (history withdrawn during a revision), step back to the day before
    the missing records were published, repeatedly if needed.
    """
    d = pd.Timestamp(asof)
    while True:
        live = vintages[(vintages["realtime_start"] <= d) & (vintages["realtime_end"] >= d)]
        gaps = live[live["value"].isna()]
        if gaps.empty:
            return live.set_index("date")["value"].sort_index()
        d = gaps["realtime_start"].min() - pd.Timedelta(days=1)


class RealTimeMacro:
    """Macro panel for one currency as known at any origin: `rt.at(origin)`."""

    def __init__(self, latest: pd.DataFrame, specs: dict, release_lag: dict,
                 vintages: pd.DataFrame | None = None, variables=VARIABLES):
        self.variables = tuple(variables)
        unknown = set(self.variables) - set(VARIABLES)
        if unknown:
            raise ValueError(f"unknown macro variables {sorted(unknown)}; choose from {VARIABLES}")
        self.latest = latest[list(self.variables)]
        self.specs = specs
        self.release_lag = release_lag
        self.vintages = {}
        if vintages is not None and not vintages.empty:
            for var, spec in specs.items():
                if spec.get("vintages") and var in self.variables:
                    v = vintages[vintages["series_id"] == spec["id"]]
                    if not v.empty:
                        self.vintages[var] = v

    @classmethod
    def load(cls, ccy: str, data_dir: Path, config_path: Path, variables=VARIABLES) -> "RealTimeMacro":
        cfg = yaml.safe_load(Path(config_path).read_text())
        processed = Path(data_dir) / "processed"
        latest = pd.read_parquet(processed / f"macro_{ccy}.parquet")
        vp = processed / f"vintages_{ccy}.parquet"
        vint = pd.read_parquet(vp) if vp.exists() else None
        return cls(latest, cfg["macro"][ccy], cfg["release_lag_months"], vint, variables)

    def _lagged_latest(self, var: str, origin: pd.Timestamp) -> pd.Series:
        last_known = origin - pd.offsets.MonthEnd(self.release_lag.get(var, 0))
        return self.latest[var].loc[:last_known]

    def _from_vintage(self, var: str, origin: pd.Timestamp) -> pd.Series:
        v = self.vintages[var]
        first = v["realtime_start"].min()
        transform = self.specs[var].get("transform", "level")
        if origin < first:
            raw = as_of_complete(v, first)
            s = to_monthly(raw, transform)
            return s.loc[: origin - pd.offsets.MonthEnd(self.release_lag.get(var, 0))]
        return to_monthly(as_of_complete(v, origin), transform).loc[:origin]

    def at(self, origin, index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
        """Macro panel as known at month-end `origin`, reindexed to `index` (NaN = not yet known)."""
        origin = pd.Timestamp(origin)
        cols = {}
        for var in self.variables:
            cols[var] = self._from_vintage(var, origin) if var in self.vintages else self._lagged_latest(var, origin)
        out = pd.DataFrame(cols)
        out = out.loc[:origin]
        if index is not None:
            out = out.reindex(index[index <= origin])
        return out
