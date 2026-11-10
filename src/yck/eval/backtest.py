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
from yck.model.statespace_dns import DNSFit, fit_dns_mle


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


def dns_kalman_forecasts(
    yields: pd.DataFrame,
    horizons=(1, 6, 12),
    first_origin: str = "2000-01-31",
    est_start: str | None = None,
    reestimate_every: int = 12,
    estimate_lambda: bool = False,
    lam: float = DL_LAMBDA,
    fits: list | None = None,
    verbose: bool = False,
) -> dict[int, pd.DataFrame]:
    """One-step (Kalman, maximum-likelihood) DNS forecasts from every origin.

    Parameters are re-estimated every `reestimate_every` origins (12 = yearly)
    on data up to that origin. Each re-estimation runs twice, from the two-step
    estimates and from the previous estimate, and keeps the higher likelihood.
    Warm starts alone are faster, but chained over many years they can drift
    onto a worse local optimum: for USD the 2014 refit lost 175 log-likelihood
    points that way. The cold start puts a floor under every refit.
    Between re-estimations the parameters are held fixed and the forecast
    starts from the *filtered* state a_{t|t}, which uses data up to t only.
    So no forecast ever sees data dated after its origin, even though one
    filter pass covers the whole sample.

    If `fits` is a list, each (origin, DNSFit) is appended to it, so parameter
    stability can be inspected afterwards.
    """
    y = yields.loc[est_start:] if est_start else yields
    origins = y.loc[first_origin:].index
    pos = {d: i for i, d in enumerate(y.index)}
    out: dict[int, list] = {h: [] for h in horizons}
    fit: DNSFit | None = None
    a_filt = None
    for i, t in enumerate(origins):
        if i % reestimate_every == 0:
            sample = y.loc[:t]
            cold = fit_dns_mle(sample, estimate_lambda=estimate_lambda, lam=lam)
            warm = fit_dns_mle(sample, start=fit.params, estimate_lambda=estimate_lambda,
                               lam=fit.params.lam) if fit else None
            fit = warm if warm is not None and warm.loglik > cold.loglik else cold
            a_filt = fit.filter(y).a_filt  # row s uses y up to s only
            if fits is not None:
                fits.append((t, fit))
            if verbose:
                gap = f", warm-cold {warm.loglik - cold.loglik:+.1f}" if warm else ""
                print(f"{t:%Y-%m}: loglik {fit.loglik:.1f}{gap}, lambda {fit.params.lam:.4f}, "
                      f"{cold.seconds + (warm.seconds if warm else 0):.0f}s", flush=True)
        a_t = a_filt[pos[t]]
        for h in horizons:
            out[h].append((_target(t, h), fit.forecast(a_t, h)))
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
