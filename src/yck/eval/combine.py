"""Forecast combination (README section 7.6, roadmap step 7b item 3).

Averaging forecasts from different models is one of the few gains in the
term-structure literature that holds up over time (de Pooter, Ravazzolo &
van Dijk 2010): models fail in different periods, and an average is never
the worst of them. Two weighting schemes, both per maturity and horizon:

- `equal`: the plain average of the available forecasts.
- `inv_mse` (Bates & Granger 1969): weights proportional to 1 / MSE of each
  model's *past* forecasts. The forecast for target T was made at origin
  T - h, so only errors whose targets are <= T - h were known then. Until
  every available model has `min_obs` such errors, the weights are equal.

Where a model has no forecast (e.g. GBP forwards at h = 1 beyond 2 years),
the combination averages the others, with their weights renormalised.
Forecasts are indexed by target date, as everywhere in `yck.eval`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_OBS = 24  # realised errors each model needs before performance weights replace equal weights


def combine(forecasts: dict[str, pd.DataFrame], actual: pd.DataFrame, h: int, weights: str = "equal",
            min_obs: int = MIN_OBS) -> pd.DataFrame:
    """Combined h-step forecast, indexed by target date, columns as in `actual`."""
    if weights not in ("equal", "inv_mse"):
        raise ValueError(f"weights must be 'equal' or 'inv_mse', not {weights!r}")
    cols = list(actual.columns)
    idx = None
    for f in forecasts.values():
        idx = f.index if idx is None else idx.union(f.index)
    # A regular month-end calendar, so "targets <= T - h" is a shift by h rows.
    cal = pd.date_range(idx.min(), idx.max(), freq="ME", name="date")
    F = np.stack([f.reindex(index=cal, columns=cols).to_numpy(float) for f in forecasts.values()])  # (m, T, n)
    avail = np.isfinite(F)
    if weights == "equal":
        W = avail.astype(float)
    else:
        a = actual.reindex(index=cal, columns=cols).to_numpy(float)
        e2 = (F - a[None]) ** 2
        known = np.isfinite(e2)
        cs = np.cumsum(np.where(known, e2, 0.0), axis=1)
        cnt = np.cumsum(known, axis=1)
        # Errors known at the origin T - h: cumulative sums up to row T - h.
        cs_k, cnt_k = np.zeros_like(cs), np.zeros_like(cnt)
        cs_k[:, h:], cnt_k[:, h:] = cs[:, :-h], cnt[:, :-h]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(cnt_k > 0, cnt_k / cs_k, 0.0)
        ready = np.all(~avail | (cnt_k >= min_obs), axis=0)  # every available model has enough history
        W = np.where(ready[None], inv, 1.0) * avail
    W = np.where(avail, W, 0.0)
    tot = W.sum(axis=0)
    with np.errstate(invalid="ignore"):
        out = np.where(tot > 0, (W * np.nan_to_num(F)).sum(axis=0) / tot, np.nan)
    return pd.DataFrame(out, index=cal, columns=cols).loc[idx.min():idx.max()].dropna(how="all")
