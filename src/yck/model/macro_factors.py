"""Real-time macro factors from large monthly panels (README section 8 step 7b item 7).

The DRA model uses three hand-picked macro series and overfits (README
section 7.4). Factor models summarise a large panel in a few factors instead
(Stock & Watson 2002; for yields: Moench 2008; Coroneo, Giannone & Modugno
2016). The steps follow McCracken & Ng (2016):

1. Transform each series to stationarity by its code (`transform`).
2. Mark outliers (more than 10 interquartile ranges from the median) as missing.
3. Standardise, then extract r principal components by the EM algorithm of
   Stock & Watson (2002), which handles missing values and the ragged edge
   (`em_pca`).
4. Scale each factor to unit variance and fix its sign: factor 1 loads
   positively on industrial production, factor 2 on consumer prices
   (`anchors` in the config), so the factors keep a meaning across origins.

Interest rates and spreads are excluded from the panels (config), so the
factors carry macro information, not the yield curve itself.

**Real time** (`RealTimeFactors.at`): at each forecast origin the factors are
re-extracted from the panel as it was known then.

- USD: the FRED-MD vintage published in the origin's month, from 1999-08.
- EUR, GBP: only the latest vintage is public. Each series is cut where it
  would have ended, given its publication delay in the latest file (its
  ragged edge): a series published with a two-month delay is known up to
  t - 2 at origin t. Revisions are ignored, as for the other EUR/GBP macro
  data (README section 7.3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MIN_SHARE = 0.3  # a month gets factor values only if at least this share of the panel is observed
MAX_MISSING = 0.2  # series missing more than this share of the window are dropped
OUTLIER_IQR = 10.0


def transform(panel: pd.DataFrame, tcode: dict) -> pd.DataFrame:
    """McCracken-Ng transformations: 1 x, 2 dx, 3 d2x, 4 log x, 5 dlog x, 6 d2log x, 7 d(x_t/x_{t-1} - 1)."""
    out = {}
    for c in panel.columns:
        x, k = panel[c].astype(float), tcode.get(c, 1)
        if k in (4, 5, 6):
            x = np.log(x.where(x > 0))
        if k in (2, 5):
            x = x.diff()
        elif k in (3, 6):
            x = x.diff().diff()
        elif k == 7:
            x = (x / x.shift(1) - 1).diff()
        out[c] = x
    return pd.DataFrame(out, index=panel.index)


def remove_outliers(X: pd.DataFrame, c: float = OUTLIER_IQR) -> pd.DataFrame:
    med = X.median()
    iqr = X.quantile(0.75) - X.quantile(0.25)
    return X.where((X - med).abs() <= c * iqr.where(iqr > 0))


def em_pca(X: np.ndarray, r: int, tol: float = 1e-6, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Principal components of a standardised T x N panel with missing values (Stock & Watson 2002 EM).

    Missing entries start at 0 (the mean) and are replaced by the r-factor fit
    until the fit stops changing. Returns factors F (T x r, F'F/T = I) and
    loadings L (N x r), with X ~ F L'.
    """
    miss = ~np.isfinite(X)
    Z = np.where(miss, 0.0, X)
    T = X.shape[0]
    prev = None
    for _ in range(max_iter):
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        F = np.sqrt(T) * U[:, :r]
        L = Z.T @ F / T
        fit = F @ L.T
        Z = np.where(miss, fit, X)
        if prev is not None and np.sum((fit - prev) ** 2) <= tol * np.sum(prev ** 2):
            break
        prev = fit
    return F, L


def extract_factors(raw: pd.DataFrame, tcode: dict, r: int, start=None, anchors=(), exclude=()) -> pd.DataFrame:
    """Factors from one panel as known at one origin: columns factor1..factor{r} on the panel's months."""
    X = transform(raw.drop(columns=[c for c in exclude if c in raw.columns]), tcode)
    if start is not None:
        X = X.loc[pd.Timestamp(start):]
    X = remove_outliers(X)
    X = X.loc[:, X.isna().mean() <= MAX_MISSING]
    X = X.loc[:, X.std() > 0]
    Xs = (X - X.mean()) / X.std()
    F, L = em_pca(Xs.to_numpy(float), r)
    names = list(X.columns)
    for j in range(r):
        anchor = anchors[j] if j < len(anchors) else None
        ref = L[names.index(anchor), j] if anchor in names else L[:, j].sum()
        if ref < 0:
            F[:, j], L[:, j] = -F[:, j], -L[:, j]
    out = pd.DataFrame(F, index=X.index, columns=[f"factor{j + 1}" for j in range(r)])
    out = out.where(Xs.notna().mean(axis=1) >= MIN_SHARE)  # months with too little data stay unobserved
    return out / out.std()  # unit variance over the observed months


class RealTimeFactors:
    """Macro factors as known at each forecast origin; `at(origin, index)` like `yck.data.realtime.RealTimeMacro`."""

    def __init__(self, r: int, tcode: dict, start=None, anchors=(), exclude=(), vintages: dict | None = None,
                 latest: pd.DataFrame | None = None):
        if (vintages is None) == (latest is None):
            raise ValueError("pass FRED-MD style vintages or a single latest panel")
        self.r, self.tcode, self.start, self.anchors, self.exclude = r, tcode, start, tuple(anchors), tuple(exclude)
        self.vintages = vintages
        self.latest = latest
        if latest is not None:  # publication delay of each series in the latest file, in months
            ref = pd.Period(latest.attrs.get("vintage", f"{latest.index[-1] + pd.offsets.MonthEnd(1):%Y-%m}"), "M")
            last = latest.apply(lambda s: s.last_valid_index())
            delay = pd.Series({c: (ref - pd.Period(d, "M")).n for c, d in last.items() if d is not None})
            typical = int(delay[delay <= 6].median())
            self.delay = delay.where(delay <= 6, typical).clip(lower=1).astype(int)  # > 6: discontinued, not late
        self._memo: dict = {}

    @classmethod
    def load(cls, ccy: str, data_dir: Path, config: Path, r: int, start=None) -> "RealTimeFactors":
        spec = yaml.safe_load(Path(config).read_text())["macro_panels"][ccy]
        proc = Path(data_dir) / "processed"
        kw = dict(start=start, anchors=spec.get("anchors", ()), exclude=spec.get("exclude", ()))
        if spec["source"] == "fredmd":
            vint = {p.stem: pd.read_parquet(p) for p in sorted((proc / "fredmd_vintages").glob("*.parquet"))}
            tcode = {}
            for v in vint.values():
                tcode.update(v.attrs.get("tcode", {}))
            return cls(r, tcode, vintages=vint, **kw)
        latest = pd.read_parquet(proc / f"panel_{ccy}.parquet")
        return cls(r, latest.attrs["tcode"], latest=latest, **kw)

    def panel_at(self, origin) -> tuple[pd.DataFrame, dict]:
        """The raw panel known at month-end `origin`, and its codes."""
        t = pd.Timestamp(origin)
        if self.vintages is not None:
            label = max((v for v in self.vintages if v <= f"{t:%Y-%m}"), default=None)
            if label is None:
                raise ValueError(f"no FRED-MD vintage published by {t:%Y-%m}")
            v = self.vintages[label]
            return v.loc[:t], v.attrs.get("tcode", self.tcode)
        known = self.latest.loc[:t].copy()
        for c, d in self.delay.items():
            known.loc[known.index > t - pd.offsets.MonthEnd(d), c] = np.nan
        return known, self.tcode

    def at(self, origin, index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
        t = pd.Timestamp(origin)
        if t not in self._memo:
            raw, tcode = self.panel_at(t)
            self._memo[t] = extract_factors(raw, tcode, self.r, self.start, self.anchors, self.exclude)
        f = self._memo[t].copy()
        return f if index is None else f.reindex(index[index <= t])


class CombinedMacro:
    """Several real-time macro sources side by side (e.g. the policy rate and panel factors)."""

    def __init__(self, *parts):
        self.parts = parts

    def at(self, origin, index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
        return pd.concat([p.at(origin, index=index) for p in self.parts], axis=1)
