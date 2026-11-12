"""Smooth shadow-rate DNS (Opschoor & van der Wel 2025), README section 7.6, roadmap step 7b item 6.

Near the lower bound the linear DNS model misbehaves: it lets short yields
fall freely below zero in forecasts, and it treats a curve pinned at the
bound like any other curve. Black's (1995) shadow-rate idea writes the
observed rate as max(r_LB, shadow rate). Opschoor & van der Wel apply a
*smooth* version of the max to every maturity of a Nelson-Siegel shadow
curve s_t(tau) = Lambda(tau) x_t:

    y_t(tau) = r_LB + kappa f((s_t(tau) - r_LB) / kappa),   f(x) = x Phi(x) + phi(x),

with Phi and phi the standard normal cdf and pdf. Far above the bound
y = s (the ordinary DNS model); near it, yields bend smoothly towards r_LB.
The factors follow the same VAR(1) as in `statespace_dns`, so the model is
a non-linear state-space model, estimated by maximum likelihood with the
extended Kalman filter (EKF). The EKF linearises the measurement around the
predicted state, and the slope dy/ds is simply Phi(d).

Choices (fixed before running, README section 8 step 7b):

- kappa = 1 (percentage points), as in the paper's forecasting exercise,
  where estimating kappa and r_LB on samples with high rates overfits.
- r_LB = min(0, lowest yield in the estimation sample): 0 % as in the
  paper unless yields have already gone below zero (EUR, GBP 2020), learnt
  in real time at each refit.
- mu fixed at the real-time sample mean of the two-step factors, as for DNS.
- Start: the linear DNS maximum-likelihood fit of the same sample.

Forecasts. The identity kappa f(x / kappa) = E[max(0, x + kappa Z)], Z ~ N(0, 1)
gives two versions: `plugin` puts the forecast shadow curve into the
measurement function, and `mean` takes the expectation over the shadow
curve's forecast uncertainty v, which is the same formula with
kappa' = sqrt(kappa^2 + v).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from scipy.stats import norm

from yck.model.dns import maturities_from_columns
from yck.model.kalman import StateSpace
from yck.model.statespace_dns import MAX_RADIUS, DNSParams, sample_mean, spectral_radius, split_columns

KAPPA = 1.0  # smoothness of the lower bound, percentage points


def zlb_yield(s, lb: float, kappa: float = KAPPA):
    """Observed yield from the shadow yield: lb + kappa f((s - lb) / kappa), f(x) = x Phi(x) + phi(x)."""
    d = (np.asarray(s, float) - lb) / kappa
    return lb + kappa * (d * norm.cdf(d) + norm.pdf(d))


def lower_bound(yields: pd.DataFrame) -> float:
    """r_LB = min(0, lowest yield observed in the sample): known at the origin."""
    return float(min(0.0, np.nanmin(yields.to_numpy(float))))


@njit(cache=True)
def _ekf(Y, Z, hvar, T, Q, c, a1, P1, lb, kappa, store):
    """Extended Kalman filter for y = lb + kappa f((Z a - lb) / kappa) + eps. Returns (loglik, a_filt, P_filt)."""
    N, n = Y.shape
    k = T.shape[0]
    a = a1.copy()
    P = P1.copy()
    ll = 0.0
    a_f = np.empty((N if store else 0, k))
    P_f = np.empty((N if store else 0, k, k))
    rt2 = math.sqrt(2.0)
    c0 = 1.0 / math.sqrt(2.0 * math.pi)
    for t in range(N):
        m = 0
        idx = np.empty(n, np.int64)
        for j in range(n):
            if np.isfinite(Y[t, j]):
                idx[m] = j
                m += 1
        if m > 0:
            idx = idx[:m]
            J = np.empty((m, k))
            v = np.empty(m)
            F = np.zeros((m, m))
            for r in range(m):
                j = idx[r]
                s = 0.0
                for q in range(k):
                    s += Z[j, q] * a[q]
                d = (s - lb) / kappa
                Phi = 0.5 * (1.0 + math.erf(d / rt2))
                phi = c0 * math.exp(-0.5 * d * d)
                v[r] = Y[t, j] - (lb + (s - lb) * Phi + kappa * phi)
                for q in range(k):
                    J[r, q] = Phi * Z[j, q]
                F[r, r] = hvar[j]
            PJt = P @ J.T
            F += J @ PJt
            L = np.linalg.cholesky(F)
            Finv = np.linalg.inv(F)
            logdet = 0.0
            for r in range(m):
                logdet += 2.0 * math.log(L[r, r])
            ll += -0.5 * (m * math.log(2.0 * math.pi) + logdet + v @ (Finv @ v))
            K = PJt @ Finv
            a = a + K @ v
            P = P - K @ PJt.T
            P = 0.5 * (P + P.T)
        if store:
            a_f[t] = a
            P_f[t] = P
        a = c + T @ a
        P = T @ P @ T.T + Q
    return ll, a_f, P_f


@dataclass
class ShadowFilter:
    a_filt: np.ndarray
    P_filt: np.ndarray
    loglik: float


@dataclass
class ShadowFit:
    params: DNSParams
    lb: float
    kappa: float
    loglik: float
    n_params: int
    nobs: int
    converged: bool
    message: str
    seconds: float
    columns: list
    linear_loglik: float  # the starting linear DNS fit, evaluated by its own (exact) Kalman filter
    estimate_lambda: bool = False
    mu_fixed: bool = True
    penalty: float = 0.0
    kind: str = field(default="shadow")

    @property
    def yield_columns(self) -> list:
        return split_columns(self.columns)[0]

    def statespace(self) -> StateSpace:
        """The linear part (shadow curve and state dynamics); the measurement is non-linear on top of it."""
        return self.params.statespace(maturities_from_columns(self.yield_columns))

    def filter(self, data: pd.DataFrame) -> ShadowFilter:
        ss = self.statespace()
        ll, a, P = _ekf(data[self.columns].to_numpy(float), ss.Z, np.diag(ss.H).copy(), ss.T, ss.Q, ss.c, ss.a1,
                        ss.P1, self.lb, self.kappa, True)
        return ShadowFilter(a, P, ll)

    def filtered_states(self, data: pd.DataFrame) -> np.ndarray:
        return self.filter(data).a_filt

    def fitted(self, a: np.ndarray) -> np.ndarray:
        """Model yields at the filtered states (rows of a)."""
        return zlb_yield(np.atleast_2d(a) @ self.statespace().Z.T, self.lb, self.kappa)

    def forecast(self, a_filt_t, h: int, P_filt_t=None, mode: str = "plugin") -> np.ndarray:
        """y_{t+h|t}: `plugin` (forecast shadow curve through the measurement) or `mean` (needs P_filt_t)."""
        ss = self.statespace()
        a, P = np.asarray(a_filt_t, float), None if P_filt_t is None else np.asarray(P_filt_t, float)
        for _ in range(h):
            a = ss.c + ss.T @ a
            if P is not None:
                P = ss.T @ P @ ss.T.T + ss.Q
        s = ss.Z @ a
        if mode == "plugin":
            return zlb_yield(s, self.lb, self.kappa)
        if mode != "mean" or P is None:
            raise ValueError("mode must be 'plugin', or 'mean' with P_filt_t")
        kap = np.sqrt(self.kappa ** 2 + np.einsum("ij,jk,ik->i", ss.Z, P, ss.Z))
        d = (s - self.lb) / kap
        return self.lb + kap * (d * norm.cdf(d) + norm.pdf(d))


def fit_shadow_dns(data: pd.DataFrame, start: DNSParams, linear_loglik: float = np.nan, kappa: float = KAPPA,
                   lb: float | None = None, h_floor_bp: float | None = 1.0, maxiter: int = 3000,
                   max_restarts: int = 5) -> ShadowFit:
    """EKF maximum likelihood from a linear DNS start (yields only, lambda fixed at start.lam, mu fixed)."""
    t0 = time.perf_counter()
    ycols, mcols = split_columns(data.columns)
    if mcols:
        raise ValueError("the shadow-rate model is implemented for yields only")
    Y = data[ycols].to_numpy(float)
    tau = maturities_from_columns(ycols)
    n, k = len(ycols), 3
    lb = lower_bound(data[ycols]) if lb is None else lb
    mu = sample_mean(data, start.lam)
    h_min = (h_floor_bp / 100.0) ** 2 if h_floor_bp else 1e-10
    p0 = DNSParams(mu, start.A, start.Q, np.maximum(start.h, h_min), start.lam)
    theta0 = p0.to_vector()[k:]  # mu is not estimated
    nobs = int(np.isfinite(Y).sum())

    def unpack(th):
        return DNSParams.from_vector(np.concatenate([mu, th]), n, False, start.lam, k)

    def fun(th):
        p = unpack(th)
        if not np.all(np.isfinite(th)) or spectral_radius(p.A) >= MAX_RADIUS:
            return 1e6
        ss = p.statespace(tau)
        ll, _, _ = _ekf(Y, ss.Z, np.diag(ss.H).copy(), ss.T, ss.Q, ss.c, ss.a1, ss.P1, lb, kappa, False)
        return -ll / nobs if np.isfinite(ll) else 1e6

    o = k * k + k * (k + 1) // 2
    bounds = [(None, None)] * o + [(np.log(h_min), None)] * n
    # L-BFGS-B with finite-difference gradients can stop early on "relative reduction"; restart from the
    # result until a run gains less than 0.01 log-likelihood points.
    th, best = theta0, fun(theta0)
    for _ in range(max_restarts):
        res = minimize(fun, th, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "maxfun": 50 * maxiter, "ftol": 1e-12, "gtol": 1e-6})
        gain = (best - res.fun) * nobs
        if res.fun < best:
            th, best = res.x, res.fun
        if gain < 0.01:
            break
    p = unpack(th)
    return ShadowFit(params=p, lb=lb, kappa=kappa, loglik=-fun(th) * nobs, n_params=th.size, nobs=nobs,
                     converged=bool(res.success), message=str(res.message), seconds=time.perf_counter() - t0,
                     columns=list(ycols), linear_loglik=linear_loglik)
