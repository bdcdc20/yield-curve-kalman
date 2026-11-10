"""Recursive out-of-sample forecasts and the random-walk benchmark.

The rule throughout is that a forecast made at origin t uses nothing dated
after t. For the two-step model that holds as follows:

- Step 1 factors are fitted month by month from that month's cross-section
  alone, so the factor for month s never depends on yields after s.
- Step 2 dynamics are re-estimated at every origin, using only factors up to t
  (an expanding window, or a rolling one if `window` is set).

Forecasts are stored by **target** date (origin + h months), so that they line
up directly with the realised curve and with other models' forecasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from yck.model.dns import DL_LAMBDA, fit_dynamics, fit_factors, maturities_from_columns, ns_loadings


def _target(origin: pd.Timestamp, h: int) -> pd.Timestamp:
    return origin + pd.offsets.MonthEnd(h)


def dns_two_step_forecasts(
    yields: pd.DataFrame,
    horizons=(1, 6, 12),
    first_origin: str = "2000-01-31",
    est_start: str | None = None,
    window: int | None = None,
    kind: str = "ar1",
    lam: float = DL_LAMBDA,
) -> dict[int, pd.DataFrame]:
    """Two-step Diebold-Li forecasts, re-estimated at every origin from `first_origin`.

    Parameters
    ----------
    yields : month-end curve panel (columns like '3m'), NaN where unobserved.
    est_start : first month of the estimation sample (default: start of data).
    window : if set, a rolling window of this many months instead of expanding.
    kind : 'ar1' (Diebold-Li's preferred specification) or 'var1'.

    Returns {h: DataFrame indexed by target date}. Origins whose target falls
    after the end of the data are still forecast, so the last rows are genuine
    forecasts beyond the end of the data.
    """
    y = yields.loc[est_start:] if est_start else yields
    factors = fit_factors(y, lam)
    X = ns_loadings(maturities_from_columns(y.columns), lam)
    origins = factors.loc[first_origin:].dropna().index

    out: dict[int, list] = {h: [] for h in horizons}
    for t in origins:
        hist = factors.loc[:t]
        if window:
            hist = hist.iloc[-window:]
        dyn = fit_dynamics(hist, kind)
        f_t = factors.loc[t].to_numpy()
        for h in horizons:
            out[h].append((_target(t, h), dyn.forecast(f_t, h) @ X.T))

    return {
        h: pd.DataFrame(np.array([v for _, v in rows]), index=pd.DatetimeIndex([d for d, _ in rows], name="date"),
                        columns=y.columns)
        for h, rows in out.items()
    }


def random_walk_forecasts(yields: pd.DataFrame, horizons=(1, 6, 12), first_origin: str = "2000-01-31"):
    """No-change benchmark: y_{t+h|t} = y_t, by maturity, indexed by target date.

    Where y_t is missing (e.g. a GBP short-end gap) the forecast is missing too,
    and that point drops out of every model's RMSE (see `rmse_table`).
    """
    base = yields.loc[first_origin:]
    out = {}
    for h in horizons:
        fc = base.copy()
        fc.index = pd.DatetimeIndex([_target(t, h) for t in base.index], name="date")
        out[h] = fc
    return out


def rmse_table(forecasts: dict[str, pd.DataFrame], actual: pd.DataFrame) -> pd.DataFrame:
    """RMSE in basis points by maturity, one column per model, on a common sample.

    A point counts only if the realised yield and every model's forecast
    exist, so every model is scored on exactly the same targets.
    """
    names = list(forecasts)
    idx = actual.index
    for f in forecasts.values():
        idx = idx.intersection(f.index)
    a = actual.loc[idx]
    mask = a.notna()
    for f in forecasts.values():
        mask &= f.loc[idx].notna()
    cols = {n: 100 * np.sqrt(((forecasts[n].loc[idx] - a)[mask] ** 2).mean()) for n in names}
    table = pd.DataFrame(cols)
    table["n_obs"] = mask.sum()
    return table
