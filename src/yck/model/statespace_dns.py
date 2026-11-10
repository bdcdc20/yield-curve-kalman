"""One-step dynamic Nelson-Siegel: the state-space model estimated by maximum likelihood.

Same model as the two-step baseline (README sections 6.1 and 6.3), but every
parameter is estimated jointly from the Kalman-filter likelihood:

    y_t       = Lambda(lam) f_t + eps_t,           eps_t ~ N(0, H), H diagonal
    f_t - mu  = A (f_{t-1} - mu) + eta_t,          eta_t ~ N(0, Q)

Why bother, if two-step works? The two-step method treats the estimated factors
as if they were data, ignoring their estimation error. It also cannot weight
maturities by how noisy they are, and it handles missing maturities only month
by month. The one-step model fixes all three, and it gives the
likelihood-based framework that the macro extension (section 6.4) needs.

Parameterisation: the optimiser works on an unconstrained vector theta. The
transforms below guarantee a valid model for any theta:

- Q = L L' with L lower-triangular and exp() on its diagonal, so Q is always
  positive definite;
- H = diag(exp(log_h)), so measurement variances are always positive;
- lam = exp(log_lam) when lambda is estimated.

A is left free; if a trial A has an eigenvalue >= 1 the stationary prior
does not exist and a diffuse prior is used for that evaluation instead
(README section 6.3).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from yck.model.dns import DL_LAMBDA, FACTORS, fit_dynamics, fit_factors, fitted_yields, maturities_from_columns, ns_loadings
from yck.model.kalman import (FilterResult, StateSpace, forecast_state, kalman_filter, kalman_loglik,
                              rts_smoother, stationary_init)

DIFFUSE_VAR = 100.0  # prior variance (pp^2) when A is not stable: wide relative to yield levels
_TRIL = np.tril_indices(3)
_DIAG_IN_TRIL = np.flatnonzero(_TRIL[0] == _TRIL[1])  # positions of L's diagonal within the 6 entries


@dataclass
class DNSParams:
    mu: np.ndarray  # (3,) unconditional factor means
    A: np.ndarray  # (3, 3) factor VAR(1) matrix
    Q: np.ndarray  # (3, 3) factor shock covariance
    h: np.ndarray  # (n,) measurement error variances, one per maturity
    lam: float = DL_LAMBDA

    # ----------------------------------------------------------------- packing
    def to_vector(self, estimate_lambda: bool = False) -> np.ndarray:
        L = np.linalg.cholesky(self.Q)
        l = L[_TRIL].copy()
        l[_DIAG_IN_TRIL] = np.log(l[_DIAG_IN_TRIL])
        parts = [self.mu, self.A.ravel(), l, np.log(self.h)]
        if estimate_lambda:
            parts.append([np.log(self.lam)])
        return np.concatenate(parts)

    @classmethod
    def from_vector(cls, theta, n: int, estimate_lambda: bool = False, lam: float = DL_LAMBDA) -> "DNSParams":
        theta = np.asarray(theta, float)
        mu, A = theta[:3], theta[3:12].reshape(3, 3)
        l = theta[12:18].copy()
        l[_DIAG_IN_TRIL] = np.exp(l[_DIAG_IN_TRIL])
        L = np.zeros((3, 3))
        L[_TRIL] = l
        h = np.exp(theta[18:18 + n])
        if estimate_lambda:
            lam = float(np.exp(theta[18 + n]))
        return cls(mu=mu, A=A, Q=L @ L.T, h=h, lam=lam)

    # ----------------------------------------------------------------- model
    def statespace(self, tau) -> StateSpace:
        """The Kalman-filter representation at maturities tau (months)."""
        c = (np.eye(3) - self.A) @ self.mu
        try:
            a1, P1 = stationary_init(self.A, self.Q, c)
        except ValueError:
            a1, P1 = self.mu.copy(), DIFFUSE_VAR * np.eye(3)
        return StateSpace(Z=ns_loadings(tau, self.lam), H=np.diag(self.h), T=self.A, Q=self.Q,
                          a1=a1, P1=P1, c=c)


def two_step_start(yields: pd.DataFrame, lam: float = DL_LAMBDA) -> DNSParams:
    """Starting values from the two-step estimates: VAR(1) on OLS factors, residual variances for H."""
    f = fit_factors(yields, lam)
    dyn = fit_dynamics(f, "var1")
    resid = yields - fitted_yields(f, yields.columns, lam)
    h = np.maximum((resid**2).mean().to_numpy(), 1e-6)  # floor: log(0) would break the transform
    return DNSParams(mu=dyn.mu, A=dyn.A, Q=dyn.Q, h=h, lam=lam)


@dataclass
class DNSFit:
    params: DNSParams
    loglik: float
    n_params: int
    nobs: int  # number of observed yields used
    converged: bool
    message: str
    seconds: float
    columns: list
    estimate_lambda: bool

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * self.n_params

    @property
    def bic(self) -> float:
        return -2 * self.loglik + np.log(self.nobs) * self.n_params

    def statespace(self) -> StateSpace:
        return self.params.statespace(maturities_from_columns(self.columns))

    def filter(self, yields: pd.DataFrame) -> FilterResult:
        return kalman_filter(self.statespace(), yields[self.columns].to_numpy(float))

    def filtered_factors(self, yields: pd.DataFrame) -> pd.DataFrame:
        """E[f_t | y_1..y_t]: uses only data up to t, so safe for forecasting."""
        return pd.DataFrame(self.filter(yields).a_filt, index=yields.index, columns=list(FACTORS))

    def smoothed_factors(self, yields: pd.DataFrame) -> pd.DataFrame:
        """E[f_t | all data]: for in-sample plots only, never for forecasts."""
        ss = self.statespace()
        a_s, _ = rts_smoother(ss, kalman_filter(ss, yields[self.columns].to_numpy(float)))
        return pd.DataFrame(a_s, index=yields.index, columns=list(FACTORS))

    def forecast(self, a_filt_t, h: int) -> np.ndarray:
        """Yield forecast y_{t+h|t} at all maturities, from the filtered state at t."""
        ss = self.statespace()
        return ss.Z @ forecast_state(ss, a_filt_t, h)


def fit_dns_mle(
    yields: pd.DataFrame,
    start: DNSParams | None = None,
    estimate_lambda: bool = False,
    lam: float = DL_LAMBDA,
    maxiter: int = 2000,
    polish: bool = True,
) -> DNSFit:
    """Maximum-likelihood estimation of the DNS state-space model.

    Minimises minus the average log-likelihood per observed yield (average
    rather than sum, so the scale of the objective, and hence the optimiser's
    tolerances, does not depend on the sample length). Uses L-BFGS-B and then,
    if `polish`, BFGS from that point, which tightens convergence on the
    flat, highly persistent directions typical of yield models.
    `start` defaults to the two-step estimates; pass the previous estimate
    when re-estimating on a slightly longer sample, which is much faster.
    """
    t0 = time.perf_counter()
    cols = list(yields.columns)
    tau = maturities_from_columns(cols)
    Y = yields.to_numpy(float)
    n_obs = int(np.isfinite(Y).sum())
    start = start or two_step_start(yields, lam)
    if estimate_lambda is False:
        start = DNSParams(start.mu, start.A, start.Q, start.h, lam)
    theta0 = start.to_vector(estimate_lambda)

    def objective(theta):
        p = DNSParams.from_vector(theta, len(cols), estimate_lambda, lam)
        ll = kalman_loglik(p.statespace(tau), Y)
        return -ll / n_obs if np.isfinite(ll) else 1e6

    res = minimize(objective, theta0, method="L-BFGS-B", options={"maxiter": maxiter, "maxfun": 50 * maxiter})
    if polish:
        res2 = minimize(objective, res.x, method="BFGS", options={"maxiter": maxiter, "gtol": 1e-6})
        if res2.fun <= res.fun:
            res = res2

    params = DNSParams.from_vector(res.x, len(cols), estimate_lambda, lam)
    return DNSFit(params=params, loglik=-res.fun * n_obs, n_params=theta0.size, nobs=n_obs,
                  converged=bool(res.success), message=str(res.message),
                  seconds=time.perf_counter() - t0, columns=cols, estimate_lambda=estimate_lambda)
