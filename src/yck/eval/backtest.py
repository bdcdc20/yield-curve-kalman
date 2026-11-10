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

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from yck.model.dns import DL_LAMBDA, fit_factors, maturities_from_columns, ns_loadings, recursive_dynamics
from yck.model.statespace_dns import DNSFit, DNSParams, fit_dns_mle


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
    # O(1) per origin via prefix sums; identical to fit_dynamics(factors.loc[:t]).
    for t, dyn in zip(origins, recursive_dynamics(factors, origins, kind, window)):
        f_t = factors.loc[t].to_numpy()
        for h in horizons:
            out[h].append((_target(t, h), dyn.forecast(f_t, h) @ X.T))

    return {
        h: pd.DataFrame(np.array([v for _, v in rows]), index=pd.DatetimeIndex([d for d, _ in rows], name="date"),
                        columns=y.columns)
        for h, rows in out.items()
    }


FIT_CACHE_VERSION = 3  # bump when the estimator changes, so cached fits are not reused


def _cache_key(sample: pd.DataFrame, settings: dict) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(sample.to_numpy(float)).tobytes())
    h.update(np.asarray(sample.index.asi8).tobytes())
    h.update(json.dumps({"cols": list(sample.columns), "v": FIT_CACHE_VERSION, **settings}, sort_keys=True).encode())
    return h.hexdigest()[:24]


def _fit_to_json(fit: DNSFit) -> dict:
    p = fit.params
    return {"mu": p.mu.tolist(), "A": p.A.tolist(), "Q": p.Q.tolist(), "h": p.h.tolist(), "lam": p.lam,
            "loglik": fit.loglik, "n_params": fit.n_params, "nobs": fit.nobs, "converged": fit.converged,
            "message": fit.message, "seconds": fit.seconds, "columns": fit.columns,
            "estimate_lambda": fit.estimate_lambda}


def _fit_from_json(d: dict) -> DNSFit:
    params = DNSParams(mu=np.array(d["mu"]), A=np.array(d["A"]), Q=np.array(d["Q"]), h=np.array(d["h"]), lam=d["lam"])
    return DNSFit(params=params, **{k: d[k] for k in ("loglik", "n_params", "nobs", "converged", "message",
                                                     "seconds", "columns", "estimate_lambda")})


def _fit_origin(sample: pd.DataFrame, settings: dict, cache_dir: str | None) -> DNSFit:
    """Fit one re-estimation origin, reading or writing the on-disk cache. Top-level so workers can pickle it."""
    path = Path(cache_dir) / f"{_cache_key(sample, settings)}.json" if cache_dir else None
    if path is not None and path.exists():
        return _fit_from_json(json.loads(path.read_text()))
    fit = fit_dns_mle(sample, **settings)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_fit_to_json(fit)))
    return fit


def dns_kalman_forecasts(
    yields: pd.DataFrame,
    horizons=(1, 6, 12),
    first_origin: str = "2000-01-31",
    est_start: str | None = None,
    reestimate_every: int = 12,
    estimate_lambda: bool = False,
    lam: float = DL_LAMBDA,
    h_floor_bp: float | None = 1.0,
    n_starts: int = 6,
    jobs: int = 1,
    cache_dir: str | Path | None = None,
    fits: list | None = None,
    verbose: bool = False,
    macro=None,
) -> dict[int, pd.DataFrame]:
    """One-step (Kalman, maximum-likelihood) DNS forecasts from every origin.

    With `macro` (a `yck.data.realtime.RealTimeMacro`, or anything with an
    `at(origin, index)` method), this is the yields + macro model (README
    section 6.4). Every re-estimation and every forecast at origin t uses the
    macro panel *as known at t*: unreleased months are missing observations,
    and earlier months carry the values published by t, not later
    revisions. Because the vintage changes each month, the filter is re-run
    on each origin's own panel.

    Parameters are re-estimated every `reestimate_every` origins (12 = yearly)
    on data up to that origin. Each re-estimation is independent: it uses
    `n_starts` starting points (the two-step estimates plus random
    perturbations) and keeps the best fit. So the re-estimations can run in
    parallel (`jobs` processes), and the result does not depend on the order
    they run in. Earlier versions warm-started each refit from the previous
    one; chained over many years, that drifted between local optima.

    Between re-estimations the parameters are held fixed and the forecast
    starts from the *filtered* state a_{t|t}, which uses data up to t only.
    So no forecast ever sees data dated after its origin, even though one
    filter pass covers the whole sample.

    Fits are cached in `cache_dir` (one JSON file per origin, keyed by a hash
    of the exact data and settings). Re-running with unchanged data, or after
    a new month arrives, only fits what is new.

    If `fits` is a list, each (origin, DNSFit) is appended to it.
    """
    y = yields.loc[est_start:] if est_start else yields
    origins = y.loc[first_origin:].index
    refit_at = list(origins[::reestimate_every])
    settings = {"estimate_lambda": estimate_lambda, "lam": lam, "h_floor_bp": h_floor_bp, "n_starts": n_starts}
    cache = str(cache_dir) if cache_dir else None

    def panel(t):
        """Everything a forecaster had at origin t."""
        if macro is None:
            return y.loc[:t]
        return pd.concat([y.loc[:t], macro.at(t, index=y.index)], axis=1)

    samples = [panel(t) for t in refit_at]

    t0 = time.perf_counter()
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            fitted = list(ex.map(_fit_origin, samples, [settings] * len(samples), [cache] * len(samples)))
    else:
        fitted = [_fit_origin(s, settings, cache) for s in samples]
    if verbose:
        for t, f in zip(refit_at, fitted):
            print(f"{t:%Y-%m}: loglik {f.loglik:.1f}, lambda {f.params.lam:.4f}, {f.seconds:.0f}s ({f.message[:30]})")
        print(f"{len(fitted)} fits in {time.perf_counter() - t0:.0f}s wall time", flush=True)
    if fits is not None:
        fits.extend(zip(refit_at, fitted))

    pos = {d: i for i, d in enumerate(y.index)}
    out: dict[int, list] = {h: [] for h in horizons}
    for fit, t_fit, t_next in zip(fitted, refit_at, refit_at[1:] + [None]):
        block = origins[(origins >= t_fit) & ((origins < t_next) if t_next is not None else True)]
        if macro is None:
            a_filt = fit.filtered_states(y)  # row s uses y up to s only
            states = {t: a_filt[pos[t]] for t in block}
        else:
            states = {t: fit.filtered_states(panel(t))[-1] for t in block}
        for t in block:
            for h in horizons:
                out[h].append((_target(t, h), fit.forecast(states[t], h)))
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


def forward_forecasts(curves_ext: pd.DataFrame, columns, horizons=(1, 6, 12),
                      first_origin: str = "2000-01-31") -> dict[int, pd.DataFrame]:
    """Market-implied forecasts: the forward rate F_t(h, tau) as the forecast of y_{t+h}(tau).

    With continuously compounded zero yields and maturities in months,

        F_t(h, tau) = [(tau + h) y_t(tau + h) - h y_t(h)] / tau,

    the yield that locks in today, for a tau-month loan starting in h months.
    It is what the market prices, so a model has to beat it to be worth
    trading on (README section 7.4). Short-rate futures price essentially
    the same quantity, up to a convexity adjustment.

    `curves_ext` must hold the needed maturities (the pipeline writes them
    to curves_ext_{CCY}.parquet). A forward whose inputs are missing (e.g. a
    maturity the BoE does not publish) is missing, never interpolated.
    Indexed by target date, like the model forecasts.
    """
    base = curves_ext.loc[first_origin:]
    out = {}
    for h in horizons:
        fc = pd.DataFrame(index=base.index, columns=list(columns), dtype=float)
        for col in columns:
            tau = maturities_from_columns([col])[0]
            far, near = f"{int(tau + h)}m", f"{int(h)}m"
            if far in base and near in base:
                fc[col] = ((tau + h) * base[far] - h * base[near]) / tau
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
