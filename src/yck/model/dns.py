"""Dynamic Nelson-Siegel: loadings and the two-step Diebold-Li (2006) baseline.

The two-step estimator is the simplest version of the model and the yardstick
for everything that follows:

1. For each month, regress the cross-section of yields on the Nelson-Siegel
   loadings (lambda fixed), which gives level, slope and curvature factors.
2. Fit AR(1)s or a VAR(1) to those factor time series and iterate them
   forward to forecast the factors, and hence the whole curve.

The one-step Kalman version (README section 6.3) estimates the same model
jointly by maximum likelihood. The two-step estimates computed here are its
starting values.

Conventions: yields in percent, maturities in months, lambda per month.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Diebold-Li's lambda for maturities in months: it puts the peak of the
# curvature loading at about 30 months (29.4 exactly), between the 2y and 3y
# points that identify curvature.
DL_LAMBDA = 0.0609
FACTORS = ("level", "slope", "curvature")


def maturities_from_columns(columns) -> np.ndarray:
    """Turn column labels such as '3m', '120m' into maturities in months."""
    out = []
    for c in columns:
        m = re.fullmatch(r"(\d+(?:\.\d+)?)m", str(c))
        if not m:
            raise ValueError(f"column {c!r} is not a maturity label like '24m'")
        out.append(float(m.group(1)))
    return np.asarray(out)


def ns_loadings(tau, lam: float = DL_LAMBDA) -> np.ndarray:
    """Nelson-Siegel factor loadings, shape (len(tau), 3).

    Columns are level (1), slope ((1-e^{-x})/x) and curvature
    ((1-e^{-x})/x - e^{-x}), with x = lam*tau. As tau -> 0 they tend to
    (1, 1, 0), so the short rate is level + slope. As tau -> inf they tend to
    (1, 0, 0), so the long rate is the level. expm1 keeps the slope loading
    accurate for small x, where 1 - e^{-x} would lose precision to cancellation.
    """
    x = lam * np.asarray(tau, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(x < 1e-8, 1.0 - x / 2.0, -np.expm1(-x) / x)
    curv = slope - np.exp(-x)
    return np.column_stack([np.ones_like(x), slope, curv])


def fit_factors(yields: pd.DataFrame, lam: float = DL_LAMBDA, min_obs: int = 3) -> pd.DataFrame:
    """Step 1: OLS of each month's yields on the loadings -> (level, slope, curvature).

    Each month uses only the maturities observed that month. Missing points
    are never filled, because an invented yield would make the fit look
    better than the data allow (README section 5.2). Months with fewer than
    `min_obs` observed maturities get NaN factors: with three unknowns, fewer
    than three points cannot pin them down.

    Months that share a missing-data pattern share a design matrix, so the
    regression is solved once per pattern, not once per month.
    """
    if min_obs < 3:
        raise ValueError("min_obs must be at least 3 (three factors)")
    tau = maturities_from_columns(yields.columns)
    X = ns_loadings(tau, lam)
    Y = yields.to_numpy(float)
    obs = np.isfinite(Y)
    F = np.full((len(Y), 3), np.nan)

    patterns, inverse = np.unique(obs, axis=0, return_inverse=True)
    for k, mask in enumerate(patterns):
        if mask.sum() < min_obs:
            continue
        rows = np.flatnonzero(inverse.ravel() == k)
        beta, *_ = np.linalg.lstsq(X[mask], Y[np.ix_(rows, mask)].T, rcond=None)
        F[rows] = beta.T
    return pd.DataFrame(F, index=yields.index, columns=list(FACTORS))


def fitted_yields(factors: pd.DataFrame, columns, lam: float = DL_LAMBDA) -> pd.DataFrame:
    """Yields implied by the factors at the maturities in `columns` (labels like '24m')."""
    X = ns_loadings(maturities_from_columns(columns), lam)
    return pd.DataFrame(factors.to_numpy(float) @ X.T, index=factors.index, columns=list(columns))


def fit_rmse_bp(yields: pd.DataFrame, factors: pd.DataFrame, lam: float = DL_LAMBDA) -> pd.Series:
    """In-sample cross-sectional fit error by maturity, RMSE in basis points.

    Computed over observed points only. Diebold-Li report roughly 5-10 bp
    for USD in the belly, and more at the very short end.
    """
    resid = yields - fitted_yields(factors, yields.columns, lam)
    return 100.0 * np.sqrt((resid**2).mean())


@dataclass
class FactorDynamics:
    """Step 2: f_t - mu = A (f_{t-1} - mu) + eta_t, eta_t ~ (0, Q).

    Writing the model in deviations from the mean means the h-step forecast
    mu + A^h (f_t - mu) decays towards the unconditional mean mu, which
    is what makes DNS forecasts differ from a random walk.
    """

    mu: np.ndarray  # (3,) unconditional mean
    A: np.ndarray  # (3, 3) autoregressive matrix (diagonal for AR(1)s)
    Q: np.ndarray  # (3, 3) residual covariance; seeds the Kalman MLE
    nobs: int

    def forecast(self, f_t, h: int) -> np.ndarray:
        """h-step-ahead factor forecast from origin f_t (works for one row or many)."""
        if h < 0:
            raise ValueError("h must be non-negative")
        f_t = np.asarray(f_t, dtype=float)
        Ah = np.linalg.matrix_power(self.A, h)
        return self.mu + (f_t - self.mu) @ Ah.T

    def is_stationary(self) -> bool:
        return bool(np.max(np.abs(np.linalg.eigvals(self.A))) < 1.0)


def fit_dynamics(factors: pd.DataFrame, kind: str = "ar1") -> FactorDynamics:
    """Fit AR(1)s ('ar1', Diebold-Li's preferred forecasting model) or a VAR(1) ('var1') by OLS.

    The regression is f_t = c + A f_{t-1} + e_t, using only pairs of
    consecutive months where both factor vectors exist, so a gap in the factors
    never links two months that are not adjacent. The mean is recovered as
    mu = (I - A)^{-1} c.
    """
    if kind not in ("ar1", "var1"):
        raise ValueError("kind must be 'ar1' or 'var1'")
    F = factors.to_numpy(float)
    ok = np.isfinite(F).all(axis=1)
    pairs = ok[1:] & ok[:-1]
    if pairs.sum() < 10:
        raise ValueError(f"only {pairs.sum()} usable consecutive pairs; need at least 10")
    y, x = F[1:][pairs], F[:-1][pairs]
    n, k = y.shape

    if kind == "var1":
        Z = np.column_stack([np.ones(n), x])
        B, *_ = np.linalg.lstsq(Z, y, rcond=None)
        c, A = B[0], B[1:].T
    else:
        c, a = np.empty(k), np.empty(k)
        for i in range(k):
            Z = np.column_stack([np.ones(n), x[:, i]])
            (c[i], a[i]), *_ = np.linalg.lstsq(Z, y[:, i], rcond=None)
        A = np.diag(a)

    resid = y - c - x @ A.T
    n_par = A.shape[1] + 1 if kind == "var1" else 2
    Q = resid.T @ resid / (n - n_par)
    mu = np.linalg.solve(np.eye(k) - A, c)
    return FactorDynamics(mu=mu, A=A, Q=Q, nobs=n)


def forecast_yields(
    factors: pd.DataFrame,
    dynamics: FactorDynamics,
    h: int,
    columns,
    lam: float = DL_LAMBDA,
) -> pd.DataFrame:
    """Yield forecasts y_{t+h|t} = Lambda f_{t+h|t}, made at every origin t in `factors`.

    The result is indexed by origin date. The target date is h months later.
    For a genuine out-of-sample exercise, `dynamics` must be estimated only on
    data up to each origin; the evaluation loop (README section 7) does that.
    """
    f_h = dynamics.forecast(factors.to_numpy(float), h)
    X = ns_loadings(maturities_from_columns(columns), lam)
    return pd.DataFrame(f_h @ X.T, index=factors.index, columns=list(columns))
