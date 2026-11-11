"""Forecast adjustments that live outside the likelihood (README section 7.6, roadmap step 7b).

Two of the four failures diagnosed in section 7.6 are fixed here without
touching the estimator.

**Shifting endpoint (failure 2).** With mu fixed at the sample mean, the level
reverts to the average since the 1980s/1990s, so forecasts kept predicting a
rebound through every rally. Following van Dijk, Koopman, van der Wel &
Wright (2014), the level reverts instead to a slowly moving endpoint e_t:

    x_t - mu - e_t [1, 0, 0]' = A (x_{t-1} - mu - e_{t-1} [1, 0, 0]') + eta_t.

The level's loading is 1 at every maturity, so this is the ordinary DNS model
for the *detrended* curve y_t - e_t: subtract e_t from every yield (and from
the policy rate, which shares the trend), estimate and filter as before, and
add e_t back to the forecast, with the endpoint itself forecast as a random
walk (e_{t+h|t} = e_t). mu, still fixed at the real-time sample mean, now
holds the average gap between the level and the endpoint (for an inflation
survey: the real rate plus the term premium).

Endpoints are real-time series: the value for month s uses only what was
known at the end of s.

- `ewma_endpoint`: one-sided exponential smoothing of the two-step level
  factor, which is itself fitted month by month from that month's curve.
- `survey_endpoint`: the latest long-run survey expectation published by s.

**AR(1) measurement errors (failure 1).** The model's fitting errors at the
ends of the curve are persistent, but Lambda f_hat throws them away. At each
origin t the filtered residuals e_s = y_s - Lambda a_{s|s}, s <= t, are fitted
maturity by maturity with an AR(1), and the forecast becomes

    y_{t+h|t} = Lambda f_{t+h|t} + m + rho^h (e_t - m)

(`ar1c`: Raviv 2013, with an intercept m), or with m = 0 (`ar1`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from yck.model.dns import DL_LAMBDA, fit_factors

ENDPOINT_ALPHA = 0.1  # monthly smoothing weight of the EWMA endpoint (van Dijk et al. 2014)
RHO_MAX = 0.99  # AR(1) coefficients are clipped to [-RHO_MAX, RHO_MAX], so error forecasts always decay
TRENDED_MACRO = ("policy_rate",)  # macro series in yield units that share the level's endpoint


# --------------------------------------------------------------------------- shifting endpoints
def ewma_endpoint(yields: pd.DataFrame, alpha: float = ENDPOINT_ALPHA, lam: float = DL_LAMBDA) -> pd.Series:
    """e_s = alpha L_s + (1 - alpha) e_{s-1}, with L the month-by-month OLS level factor.

    One-sided, so e_s uses yields up to s only. Months without a level (too
    few maturities observed) keep the previous value: nothing new was learnt.
    """
    level = fit_factors(yields, lam)["level"]
    return level.ewm(alpha=alpha, adjust=False, ignore_na=True).mean().rename("endpoint")


def survey_endpoint(survey: pd.Series, index: pd.DatetimeIndex, known_month: int = 2) -> pd.Series:
    """The latest survey value published by each month-end in `index`.

    `survey` is indexed by the first day of the survey quarter; it counts as
    known from the end of month `known_month` of that quarter. Before the first
    survey the endpoint is missing. Holding the latest published value is what
    a forecaster had, not an interpolation: nothing between surveys is
    invented.
    """
    s = survey.dropna()
    known = pd.DatetimeIndex(s.index) + pd.offsets.MonthEnd(known_month)
    pub = pd.Series(s.to_numpy(float), index=known).groupby(level=0).last().sort_index()
    pos = np.searchsorted(pub.index.values, pd.DatetimeIndex(index).values, side="right") - 1
    vals = np.where(pos >= 0, pub.to_numpy()[np.maximum(pos, 0)], np.nan)
    return pd.Series(vals, index=index, name="endpoint")


def detrend(panel: pd.DataFrame, endpoint: pd.Series, yield_columns) -> pd.DataFrame:
    """Subtract the endpoint from every yield and from macro series in yield units (the policy rate)."""
    out = panel.copy()
    e = endpoint.reindex(panel.index).to_numpy(float)
    cols = list(yield_columns) + [c for c in TRENDED_MACRO if c in panel.columns]
    out[cols] = out[cols].to_numpy(float) - e[:, None]
    return out


# --------------------------------------------------------------------------- AR(1) measurement errors
def fit_error_ar(resid: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-column AR(1) of a residual path (rows = months): returns (m, rho).

    'ar1': e_s = rho e_{s-1} + u_s (m = 0). 'ar1c': e_s - m = rho (e_{s-1} - m) + u_s,
    by OLS of e_s on (1, e_{s-1}). Only pairs of consecutive observed months
    are used. Columns with fewer than 3 pairs get rho = 0 (and m = their mean).
    """
    if mode not in ("ar1", "ar1c"):
        raise ValueError(f"mode must be 'ar1' or 'ar1c', not {mode!r}")
    x, y = resid[:-1], resid[1:]
    ok = np.isfinite(x) & np.isfinite(y)
    n = ok.sum(axis=0)
    x0, y0 = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
    m = np.zeros(resid.shape[1])
    few = n < 3
    with np.errstate(invalid="ignore", divide="ignore"):
        if mode == "ar1":
            rho = np.clip(np.nan_to_num((x0 * y0).sum(0) / (x0 * x0).sum(0)), -RHO_MAX, RHO_MAX)
        else:
            xm, ym = x0.sum(0) / n, y0.sum(0) / n
            sxy = np.where(ok, (x - xm) * (y - ym), 0.0).sum(0)
            sxx = np.where(ok, (x - xm) ** 2, 0.0).sum(0)
            rho = np.clip(np.nan_to_num(sxy / sxx), -RHO_MAX, RHO_MAX)
            m = np.nan_to_num((ym - rho * xm) / (1 - rho))  # mean of the AR(1); |rho| < 1 after clipping
            obs = np.isfinite(resid)
            col_mean = np.where(obs, resid, 0.0).sum(0) / np.maximum(obs.sum(0), 1)
            m[few] = col_mean[few]
    rho[few] = 0.0
    return m, rho


def error_forecast(m: np.ndarray, rho: np.ndarray, e_t: np.ndarray, h: int) -> np.ndarray:
    """E[e_{t+h} | e_t] = m + rho^h (e_t - m); where e_t is missing (yield not observed at t), m."""
    dev = np.where(np.isfinite(e_t), e_t - m, 0.0)
    return m + rho ** h * dev
