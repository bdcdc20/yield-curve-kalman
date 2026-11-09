"""Curve utilities: Svensson/Nelson-Siegel evaluation and month-end sampling."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _loading(n: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ((1-e^{-x})/x, (1-e^{-x})/x - e^{-x}) with x = n/tau, stable as x -> 0."""
    x = n / tau
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(x < 1e-8, 1.0 - x / 2.0, -np.expm1(-x) / x)
    return slope, slope - np.exp(-x)


def svensson_yield(params: pd.DataFrame, maturities_years) -> pd.DataFrame:
    """Evaluate Svensson zero-coupon yields (continuously compounded, in %).

    params needs columns BETA0..BETA3, TAU1, TAU2 (one row per date). Where
    BETA3/TAU2 are missing (e.g. GSW before 1980 is plain Nelson-Siegel) the
    fourth term is dropped.
    """
    m = np.asarray(maturities_years, dtype=float)[None, :]
    b0, b1, b2 = (params[c].to_numpy(float)[:, None] for c in ("BETA0", "BETA1", "BETA2"))
    b3 = params["BETA3"].to_numpy(float)[:, None]
    t1 = params["TAU1"].to_numpy(float)[:, None]
    t2 = params["TAU2"].to_numpy(float)[:, None]

    s1, c1 = _loading(m, t1)
    y = b0 + b1 * s1 + b2 * c1
    nss = np.isfinite(b3) & np.isfinite(t2) & (t2 > 0)
    _, c2 = _loading(m, np.where(nss, t2, 1.0))
    y = y + np.where(nss, b3, 0.0) * c2
    return pd.DataFrame(y, index=params.index, columns=list(maturities_years))


def month_end(df: pd.DataFrame) -> pd.DataFrame:
    """Last available day in each calendar month, indexed by month-end date.

    Takes the whole row from that day (not the last non-missing value per
    column), so every maturity in a month comes from the same curve and
    genuine gaps stay as NaN.
    """
    df = df.sort_index().dropna(how="all")
    out = df.groupby(df.index.to_period("M")).tail(1).copy()
    out.index = out.index.to_period("M").to_timestamp(how="end").normalize()
    out.index.name = "date"
    return out


def label_months(cols) -> list[str]:
    return [f"{int(c)}m" for c in cols]
