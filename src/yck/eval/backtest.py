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

from yck.model.adjust import anchor_mean, detrend, error_forecast, fit_error_ar, yield_forecast_cov
from yck.model.dns import DL_LAMBDA, fit_factors, maturities_from_columns, ns_loadings, recursive_dynamics
from yck.model.shadow import KAPPA, SHADOW_FIT_VERSION, ShadowFit, fit_shadow_dns
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


FIT_CACHE_VERSION = 11  # bump when the estimator changes, so cached fits are not reused


def _cache_key(sample: pd.DataFrame, settings: dict) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(sample.to_numpy(float)).tobytes())
    h.update(np.asarray(sample.index.asi8).tobytes())
    h.update(json.dumps({"cols": list(sample.columns), "v": FIT_CACHE_VERSION, **settings}, sort_keys=True).encode())
    return h.hexdigest()[:24]


def _fit_to_json(fit) -> dict:
    p = fit.params
    if isinstance(fit, ShadowFit):
        return {"kind": "shadow", "mu": p.mu.tolist(), "A": p.A.tolist(), "Q": p.Q.tolist(), "h": p.h.tolist(),
                "lam": p.lam, "lb": fit.lb, "kappa": fit.kappa, "linear_loglik": fit.linear_loglik,
                **{k: getattr(fit, k) for k in ("loglik", "n_params", "nobs", "converged", "message", "seconds",
                                                 "columns")}}
    return {"mu": p.mu.tolist(), "A": p.A.tolist(), "Q": p.Q.tolist(), "h": p.h.tolist(), "lam": p.lam,
            "loglik": fit.loglik, "n_params": fit.n_params, "nobs": fit.nobs, "converged": fit.converged,
            "message": fit.message, "seconds": fit.seconds, "columns": fit.columns,
            "estimate_lambda": fit.estimate_lambda, "penalty": fit.penalty, "mu_fixed": fit.mu_fixed}


def _fit_from_json(d: dict):
    params = DNSParams(mu=np.array(d["mu"]), A=np.array(d["A"]), Q=np.array(d["Q"]), h=np.array(d["h"]), lam=d["lam"])
    if d.get("kind") == "shadow":
        return ShadowFit(params=params, lb=d["lb"], kappa=d["kappa"], linear_loglik=d["linear_loglik"],
                         **{k: d[k] for k in ("loglik", "n_params", "nobs", "converged", "message", "seconds",
                                              "columns")})
    return DNSFit(params=params, penalty=d.get("penalty", 0.0), mu_fixed=d.get("mu_fixed", False),
                  **{k: d[k] for k in ("loglik", "n_params", "nobs", "converged", "message",
                                       "seconds", "columns", "estimate_lambda")})


def _fit_origin(sample: pd.DataFrame, settings: dict, cache_dir: str | None) -> DNSFit:
    """Fit one re-estimation origin, reading or writing the on-disk cache. Top-level so workers can pickle it."""
    path = Path(cache_dir) / f"{_cache_key(sample, settings)}.json" if cache_dir else None
    if path is not None and path.exists():
        return _fit_from_json(json.loads(path.read_text()))
    if settings.get("model") == "shadow":  # EKF ML from the (cached) linear fit of the same sample
        linear = _fit_origin(sample, {k: v for k, v in settings.items()
                                      if k not in ("model", "kappa", "shadow_version")}, cache_dir)
        fit = fit_shadow_dns(sample, linear.params, linear.loglik, kappa=settings["kappa"],
                             h_floor_bp=settings["h_floor_bp"])
    else:
        fit = fit_dns_mle(sample, **settings)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_fit_to_json(fit)))
    return fit


def _refit_plan(yields, first_origin, est_start, reestimate_every, estimate_lambda, lam, h_floor_bp, n_starts,
                macro, shrink_tau, endpoint=None, shadow=None):
    """What a Kalman backtest re-estimates: (y, origins, refit dates, fit settings, panel function).

    With an `endpoint`, the estimation sample starts at its first available
    month and `panel` returns the detrended data (`yck.model.adjust.detrend`).
    """
    y = yields.loc[est_start:] if est_start else yields
    if endpoint is not None:
        y = y.loc[endpoint.reindex(y.index).first_valid_index():]
    origins = y.loc[first_origin:].index
    refit_at = list(origins[::reestimate_every])
    settings = {"estimate_lambda": estimate_lambda, "lam": lam, "h_floor_bp": h_floor_bp, "n_starts": n_starts}
    if shrink_tau is not None:
        settings["shrink_tau"] = shrink_tau  # only added when used, so existing cache keys stay valid
    if shadow is not None:
        if macro is not None or endpoint is not None or estimate_lambda:
            raise ValueError("the shadow-rate model is yields only, lambda fixed, without an endpoint")
        settings.update(model="shadow", kappa=KAPPA, shadow_version=SHADOW_FIT_VERSION)

    def panel(t):
        """Everything a forecaster had at origin t."""
        d = y.loc[:t] if macro is None else pd.concat([y.loc[:t], macro.at(t, index=y.index)], axis=1)
        return d if endpoint is None else detrend(d, endpoint, y.columns)

    return y, origins, refit_at, settings, panel


def _fit_cost(sample: pd.DataFrame, settings: dict) -> float:
    """Rough relative cost of one re-estimation: months x states^2, more with lambda or shrinkage."""
    k = 3 + sum(1 for c in sample.columns if not str(c).endswith("m"))
    return len(sample) * k * k * (1.5 if settings.get("estimate_lambda") else 1.0) * (
        1.5 if "shrink_tau" in settings else 1.0) * (4.0 if settings.get("model") == "shadow" else 1.0)


def prefit(specs: list[dict], jobs: int, cache_dir: str | Path, verbose: bool = False) -> int:
    """Estimate every re-estimation of several backtests in one process pool, longest first.

    `specs` are keyword-argument dicts for `dns_kalman_forecasts`. Running the
    backtests one by one starts a pool per backtest and waits for each one's
    slowest refit, so cores sit idle at every barrier. Here all refits of all
    specs (currencies, model variants) form one queue, already-cached fits are
    skipped, and the rest are dispatched longest-processing-time first (the
    LPT greedy rule): big USD macro fits start first and short ones fill the
    gaps at the end. Fits land in the cache, so the backtests themselves then
    run from it. Returns the number of fits computed.
    """
    tasks, seen = [], set()
    for spec in specs:
        kw = {k: v for k, v in spec.items() if k in ("first_origin", "est_start", "reestimate_every",
                                                      "estimate_lambda", "lam", "h_floor_bp", "n_starts",
                                                      "macro", "shrink_tau", "endpoint", "shadow")}
        full = {"first_origin": "2000-01-31", "est_start": None, "reestimate_every": 12, "estimate_lambda": False,
                "lam": DL_LAMBDA, "h_floor_bp": 1.0, "n_starts": 6, "macro": None, "shrink_tau": None,
                "endpoint": None, "shadow": None, **kw}
        _, _, refit_at, settings, panel = _refit_plan(spec["yields"], **full)
        for t in refit_at:
            sample = panel(t)
            key = _cache_key(sample, settings)
            if key in seen or (Path(cache_dir) / f"{key}.json").exists():
                continue
            seen.add(key)
            tasks.append((_fit_cost(sample, settings), sample, settings))
    tasks.sort(key=lambda x: -x[0])
    t0 = time.perf_counter()
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
            list(ex.map(_fit_origin, [s for _, s, _ in tasks], [st for _, _, st in tasks],
                        [str(cache_dir)] * len(tasks)))
    if verbose:
        print(f"prefit: {len(tasks)} fits in {time.perf_counter() - t0:.0f}s wall time ({jobs} jobs, longest first)",
              flush=True)
    return len(tasks)


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
    shrink_tau: float | None = None,
    endpoint: pd.Series | None = None,
    error_ar: str | None = None,
    anchor: pd.DataFrame | None = None,
    anchor_columns=("3m", "6m", "9m", "12m"),
    shadow: str | None = None,
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

    Step 7b adjustments (README section 7.6, `yck.model.adjust`):

    - `endpoint` (a monthly real-time series): the level reverts to it instead
      of to a fixed mean. The model is estimated and filtered on the
      detrended panel and the origin's endpoint is added back to the forecast.
      The estimation sample starts where the endpoint does. Refits differ
      from the plain model's, so they get their own cache entries.
    - `error_ar` ('ar1' or 'ar1c'): adds the forecast of a per-maturity AR(1)
      fitted at each origin to the filtered residuals up to that origin. It
      reuses the plain model's fits (no re-estimation).
    - `anchor` (a `curves_ext` panel): tilts each forecast so that the
      `anchor_columns` equal their forward rates at the origin
      (`yck.model.adjust.anchor_mean`), using the model's forecast covariance
      from P_{t|t}. Forwards whose inputs are missing are not used. Also no
      re-estimation.
    - `shadow` ('plugin' or 'mean'): the smooth shadow-rate model
      (`yck.model.shadow`), estimated by EKF maximum likelihood from the
      linear fit, with forecasts through the measurement function. Yields
      only.
    """
    y, origins, refit_at, settings, panel = _refit_plan(yields, first_origin, est_start, reestimate_every,
                                                        estimate_lambda, lam, h_floor_bp, n_starts, macro, shrink_tau,
                                                        endpoint, shadow)
    cache = str(cache_dir) if cache_dir else None
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
    shift = np.zeros(len(y)) if endpoint is None else endpoint.reindex(y.index).to_numpy(float)
    out: dict[int, list] = {h: [] for h in horizons}
    for fit, t_fit, t_next in zip(fitted, refit_at, refit_at[1:] + [None]):
        block = origins[(origins >= t_fit) & ((origins < t_next) if t_next is not None else True)]
        ss = fit.statespace()
        n = len(y.columns)
        Z = ss.Z[:n]

        def states(d):  # filtered states, plus their covariances when anchoring or the shadow forecast needs them
            if anchor is None and shadow is None:
                return fit.filtered_states(d), None
            r = fit.filter(d)
            return r.a_filt, r.P_filt

        if macro is None:  # one pass for the block: row s of the filter uses data up to s only
            whole = panel(y.index[-1])
            whole_states = states(whole)
        for t in block:
            if macro is None:
                data, (a, P) = whole, whole_states
            else:  # the macro vintage changes with the origin: one filter pass per origin
                data = panel(t)
                a, P = states(data)
            i = pos[t]
            if error_ar is not None:  # residuals up to the origin only
                Yt = data[y.columns].to_numpy(float)[:i + 1]
                resid = Yt - fit.fitted(a[:i + 1])
                m, rho = fit_error_ar(resid, error_ar)
            for h in horizons:
                f = (fit.forecast(a[i], h, P[i], shadow) if shadow else fit.forecast(a[i], h)) + shift[i]
                if error_ar is not None:
                    f = f + error_forecast(m, rho, resid[-1], h)
                if anchor is not None:
                    fwd = np.full(n, np.nan)
                    for col in anchor_columns:
                        if col in y.columns and t in anchor.index:
                            fwd[y.columns.get_loc(col)] = _forward_rate(anchor.loc[t], maturities_from_columns([col])[0], h)
                    cov = yield_forecast_cov(ss.T, ss.Q, Z, ss.H[:n, :n], P[i], h)
                    f = anchor_mean(f, cov, fwd)
                out[h].append((_target(t, h), f))
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


def _forward_rate(row: pd.Series, tau: float, h: int) -> float:
    """F_t(h, tau) = [(tau + h) y_t(tau + h) - h y_t(h)] / tau from one curves_ext row; NaN if an input is missing."""
    far, near = f"{int(tau + h)}m", f"{int(h)}m"
    if far not in row.index or near not in row.index:
        return np.nan
    return float(((tau + h) * row[far] - h * row[near]) / tau)


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
