"""One-step dynamic Nelson-Siegel: the state-space model estimated by maximum likelihood.

Same model as the two-step baseline (README sections 6.1 and 6.3), but every
parameter is estimated jointly from the Kalman-filter likelihood:

    y_t       = Lambda(lam) f_t + eps_t,           eps_t ~ N(0, H), H diagonal
    x_t - mu  = A (x_{t-1} - mu) + eta_t,          eta_t ~ N(0, Q)

**Yields only:** x_t = f_t = (level, slope, curvature).

**Yields + macro (Diebold, Rudebusch & Aruoba 2006, README section 6.4):**
x_t = (level, slope, curvature, m_1..m_q). The macro variables are observed
without error (each loads 1:1 on its own state, with a tiny fixed variance
`MACRO_H` to keep F_t invertible), and the full VAR(1) lets macro shocks move
the factors and vice versa. Pass the macro series as extra columns of the
data frame (any column not named like '24m'). With no macro columns this is
exactly the yields-only model.

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

Estimation is done in two phases, both driven by one compiled filter +
smoother pass that returns sufficient statistics (`kalman.smoothed_stats`):

1. EM (Shumway & Stoffer 1982). Closed-form updates of A, mu, Q and H from
   the smoothed moments. EM is robust from a poor start and never needs a
   step size, but it slows down near the optimum.
2. L-BFGS-B with the exact score. By Fisher's identity, the gradient of the
   log-likelihood equals the gradient of the expected complete-data
   log-likelihood, with expectations held at the current parameters. Given
   the statistics, that function is a cheap closed form, so the gradient
   costs one smoother pass instead of one filter pass per parameter (the
   numerical-gradient approach, which also stalled on the flat ridges of
   this likelihood).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from yck.model.dns import DL_LAMBDA, FACTORS, fit_dynamics, fit_factors, fitted_yields, maturities_from_columns, ns_loadings
from yck.model.kalman import (FilterResult, SmootherStats, StateSpace, forecast_state, kalman_filter,
                              rts_smoother, smoothed_stats, stationary_init)

DIFFUSE_VAR = 100.0  # prior variance (pp^2) when A is not stable: wide relative to yield levels
MACRO_H = 1e-8  # measurement variance of macro series: "observed without error", kept invertible


def split_columns(columns) -> tuple[list, list]:
    """(yield columns, macro columns): yields are labelled like '24m', everything else is macro."""
    ycols = [c for c in columns if re.fullmatch(r"\d+(?:\.\d+)?m", str(c))]
    return ycols, [c for c in columns if c not in ycols]


def _tril(k: int):
    idx = np.tril_indices(k)
    return idx, np.flatnonzero(idx[0] == idx[1])


@dataclass
class DNSParams:
    mu: np.ndarray  # (k,) unconditional state means; k = 3 factors + number of macro variables
    A: np.ndarray  # (k, k) state VAR(1) matrix
    Q: np.ndarray  # (k, k) state shock covariance
    h: np.ndarray  # (n,) measurement error variances, one per yield maturity
    lam: float = DL_LAMBDA

    @property
    def k(self) -> int:
        return self.mu.size

    # ----------------------------------------------------------------- packing
    def to_vector(self, estimate_lambda: bool = False) -> np.ndarray:
        (rows, cols), diag = _tril(self.k)
        l = np.linalg.cholesky(self.Q)[rows, cols]
        l[diag] = np.log(l[diag])
        parts = [self.mu, self.A.ravel(), l, np.log(self.h)]
        if estimate_lambda:
            parts.append([np.log(self.lam)])
        return np.concatenate(parts)

    @classmethod
    def from_vector(cls, theta, n: int, estimate_lambda: bool = False, lam: float = DL_LAMBDA,
                    k: int = 3) -> "DNSParams":
        theta = np.asarray(theta, float)
        (rows, cols), diag = _tril(k)
        nl = rows.size
        mu, A = theta[:k], theta[k:k + k * k].reshape(k, k)
        o = k + k * k
        l = theta[o:o + nl].copy()
        l[diag] = np.exp(l[diag])
        L = np.zeros((k, k))
        L[rows, cols] = l
        h = np.exp(theta[o + nl:o + nl + n])
        if estimate_lambda:
            lam = float(np.exp(theta[o + nl + n]))
        return cls(mu=mu, A=A, Q=L @ L.T, h=h, lam=lam)

    # ----------------------------------------------------------------- model
    def statespace(self, tau) -> StateSpace:
        """The Kalman-filter representation: yields at maturities tau (months), then the macro series."""
        k, q = self.k, self.k - 3
        n = len(tau)
        Z = np.zeros((n + q, k))
        Z[:n, :3] = ns_loadings(tau, self.lam)
        Z[n:, 3:] = np.eye(q)
        H = np.diag(np.concatenate([self.h, np.full(q, MACRO_H)]))
        c = (np.eye(k) - self.A) @ self.mu
        try:
            a1, P1 = stationary_init(self.A, self.Q, c)
        except ValueError:
            a1, P1 = self.mu.copy(), DIFFUSE_VAR * np.eye(k)
        return StateSpace(Z=Z, H=H, T=self.A, Q=self.Q, a1=a1, P1=P1, c=c)


def two_step_start(data: pd.DataFrame, lam: float = DL_LAMBDA) -> DNSParams:
    """Starting values: OLS factors (step 1) and a VAR(1) on factors plus any macro columns.

    Measurement variances come from the cross-sectional residuals.
    """
    ycols, mcols = split_columns(data.columns)
    yields = data[ycols]
    f = fit_factors(yields, lam)
    states = pd.concat([f, data[mcols]], axis=1) if mcols else f
    dyn = fit_dynamics(states, "var1")
    resid = yields - fitted_yields(f, ycols, lam)
    h = np.maximum((resid**2).mean().to_numpy(), 1e-6)  # floor: log(0) would break the transform
    return DNSParams(mu=dyn.mu, A=dyn.A, Q=dyn.Q, h=h, lam=lam)


@dataclass
class DNSFit:
    params: DNSParams
    loglik: float
    n_params: int
    nobs: int  # number of observed values used (yields and macro)
    converged: bool
    message: str
    seconds: float
    columns: list  # yield columns, then macro columns
    estimate_lambda: bool

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * self.n_params

    @property
    def bic(self) -> float:
        return -2 * self.loglik + np.log(self.nobs) * self.n_params

    @property
    def yield_columns(self) -> list:
        return split_columns(self.columns)[0]

    @property
    def state_names(self) -> list:
        return list(FACTORS) + split_columns(self.columns)[1]

    def statespace(self) -> StateSpace:
        return self.params.statespace(maturities_from_columns(self.yield_columns))

    def filter(self, data: pd.DataFrame) -> FilterResult:
        return kalman_filter(self.statespace(), data[self.columns].to_numpy(float))

    def filtered_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """E[x_t | data up to t]: safe for forecasting."""
        return pd.DataFrame(self.filter(data).a_filt, index=data.index, columns=self.state_names)

    def smoothed_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """E[x_t | all data]: for in-sample plots only, never for forecasts."""
        ss = self.statespace()
        a_s, _ = rts_smoother(ss, kalman_filter(ss, data[self.columns].to_numpy(float)))
        return pd.DataFrame(a_s, index=data.index, columns=self.state_names)

    def forecast(self, a_filt_t, h: int) -> np.ndarray:
        """Yield forecast y_{t+h|t} at the yield maturities, from the filtered state at t."""
        ss = self.statespace()
        n = len(self.yield_columns)
        return ss.Z[:n] @ forecast_state(ss, a_filt_t, h)


def _logdet_and_inv(M):
    L = np.linalg.cholesky(M)
    Linv = np.linalg.inv(L)
    return 2.0 * np.log(np.diag(L)).sum(), Linv.T @ Linv


def expected_complete_loglik(theta, st: SmootherStats, tau, estimate_lambda: bool = False,
                             lam: float = DL_LAMBDA) -> float:
    """E[log p(states, observed data; theta)], expectations from the smoother run in `st`.

    Constants are dropped. Three terms, each a closed form in the statistics:

    - measurement, per yield maturity j: -1/2 [n_j log h_j + S_j / h_j], where
      S_j = sum (y - Z_j x)^2 in expectation = syy - 2 Z_j sya + Z_j sxx Z_j'.
      Macro rows have fixed loadings and variance, so their term does not
      depend on theta and is omitted;
    - transition: -1/2 [(N-1) log|Q| + tr(Q^{-1} W)], where
      W = S11 - B S10' - S10 B' + B S00 B' and B = [c, A];
    - first state: -1/2 [log|P1| + tr(P1^{-1} E[(x_1 - a1)(x_1 - a1)'])].
    """
    n = len(tau)
    k = st.S11.shape[0]
    p = DNSParams.from_vector(theta, n, estimate_lambda, lam, k)
    ss = p.statespace(tau)
    Z = ss.Z[:n]
    S = st.syy[:n] - 2.0 * np.einsum("jk,jk->j", Z, st.sya[:n]) + np.einsum("jk,jkl,jl->j", Z, st.sxx[:n], Z)
    ll = -0.5 * np.sum(st.n_i[:n] * np.log(p.h) + S / p.h)

    B = np.column_stack([ss.c, ss.T])
    W = st.S11 - B @ st.S10.T - st.S10 @ B.T + B @ st.S00 @ B.T
    logdet_Q, Qinv = _logdet_and_inv(p.Q)
    ll += -0.5 * (st.n_trans * logdet_Q + np.sum(Qinv * W))

    x1 = st.a_s[0]
    M = np.outer(x1 - ss.a1, x1 - ss.a1) + st.P_s[0]
    logdet_P1, P1inv = _logdet_and_inv(ss.P1)
    ll += -0.5 * (logdet_P1 + np.sum(P1inv * M))
    return float(ll)


def score(theta, st: SmootherStats, tau, estimate_lambda: bool = False, lam: float = DL_LAMBDA,
          eps: float = 1e-5) -> np.ndarray:
    """Exact gradient of the log-likelihood at theta (Fisher's identity).

    Differentiates the closed-form expected complete-data log-likelihood with
    the smoother statistics held fixed. Central differences are accurate here
    (errors ~1e-9) because that function is smooth and costs microseconds;
    no filter is re-run.
    """
    theta = np.asarray(theta, float)
    g = np.empty_like(theta)
    for i in range(theta.size):
        e = np.zeros_like(theta)
        e[i] = eps
        g[i] = (expected_complete_loglik(theta + e, st, tau, estimate_lambda, lam)
                - expected_complete_loglik(theta - e, st, tau, estimate_lambda, lam)) / (2 * eps)
    return g


def em_update(p: DNSParams, st: SmootherStats, tau, h_min: float = 1e-10) -> DNSParams:
    """One EM M-step: closed-form A, mu, Q, H given the smoother statistics (lambda unchanged).

    The transition is a regression of x_t on (1, x_{t-1}) in expectation:
    B = S10 S00^{-1}, Q = (S11 - B S10') / (N-1). Each maturity's variance is
    its expected squared residual, averaged over the months it is observed.
    The first-state term is left out of the M-step (standard practice); the
    gradient phase that follows maximises the exact likelihood.
    """
    n, k = len(tau), p.k
    B = np.linalg.solve(st.S00.T, st.S10.T).T
    c, A = B[:, 0], B[:, 1:]
    Q = (st.S11 - B @ st.S10.T) / st.n_trans
    Q = (Q + Q.T) / 2
    Z = ns_loadings(tau, p.lam)
    S = st.syy[:n] - 2.0 * np.einsum("jk,jk->j", Z, st.sya[:n, :3]) \
        + np.einsum("jk,jkl,jl->j", Z, st.sxx[:n, :3, :3], Z)
    h = np.maximum(S / np.maximum(st.n_i[:n], 1.0), h_min)
    try:
        mu = np.linalg.solve(np.eye(k) - A, c)
    except np.linalg.LinAlgError:
        mu = p.mu
    return DNSParams(mu=mu, A=A, Q=Q, h=h, lam=p.lam)


def fit_dns_mle(
    data: pd.DataFrame,
    start: DNSParams | None = None,
    estimate_lambda: bool = False,
    lam: float = DL_LAMBDA,
    h_floor_bp: float | None = 1.0,
    n_starts: int = 1,
    seed: int = 0,
    **kwargs,
) -> DNSFit:
    """Maximum-likelihood estimation of the DNS state-space model (yields only, or yields + macro).

    `data` holds yield columns ('3m', ..., '120m') and optionally macro
    columns, which turn the model into the DRA yields + macro model.

    Each start runs EM and then L-BFGS-B with the exact score (`_fit_from`).
    With `n_starts > 1`, the extra starts are random perturbations of the
    first (fixed `seed`, so results are reproducible), and the
    highest-likelihood fit is kept.

    **Measurement-noise floor.** By default each maturity's noise sd is
    bounded below by `h_floor_bp` = 1 bp. Without it, the USD likelihood
    pushes a few maturities to ~0.01 bp: GSW yields are points on a smooth
    fitted Svensson curve, so three factors can match some of them almost
    exactly. That makes the likelihood so badly conditioned that quasi-Newton
    methods stop with a large gradient, and leaves scattered local maxima.
    No real yield is measured to better than about 1 bp. With the floor,
    every start converges (gradient < 0.02) and starts land on one or two
    reproducible optima. Pass `h_floor_bp=None` for the unrestricted model.
    """
    ycols, mcols = split_columns(data.columns)
    data = data[ycols + mcols]
    p0 = start or two_step_start(data, lam)
    rng = np.random.default_rng(seed)
    best = None
    for i in range(n_starts):
        p = p0 if i == 0 else _perturb(p0, rng)
        fit = _fit_from(data, len(ycols), p, estimate_lambda, lam, h_floor_bp, **kwargs)
        if best is None or fit.loglik > best.loglik:
            best = fit
    return best


def _perturb(p: DNSParams, rng) -> DNSParams:
    """A random start near p: shifted means, jittered (still stable) A, rescaled Q, H and lambda."""
    k = p.k
    A = p.A + rng.normal(scale=0.03, size=(k, k))
    if np.max(np.abs(np.linalg.eigvals(A))) >= 0.999:
        A = p.A
    return DNSParams(mu=p.mu + rng.normal(scale=1.0, size=k), A=A, Q=p.Q * np.exp(rng.normal(scale=0.5)),
                     h=p.h * np.exp(rng.normal(scale=1.0, size=p.h.size)), lam=p.lam * np.exp(rng.normal(scale=0.2)))


def _fit_from(
    data: pd.DataFrame,
    n: int,
    p: DNSParams,
    estimate_lambda: bool,
    lam: float,
    h_floor_bp: float | None,
    em_iters: int = 300,
    em_tol: float = 1e-2,
    maxiter: int = 5000,
    max_restarts: int = 10,
) -> DNSFit:
    """One start: EM, then L-BFGS-B with the exact score, restarted until it stops improving.

    `data` has the n yield columns first, then any macro columns. The
    objective is minus the average log-likelihood per observed value, so its
    scale and hence the optimiser's tolerances do not depend on sample
    length. EM stops when an iteration gains less than `em_tol`
    log-likelihood points.
    """
    t0 = time.perf_counter()
    cols = list(data.columns)
    tau = maturities_from_columns(cols[:n])
    Y = data.to_numpy(float)
    n_obs = int(np.isfinite(Y).sum())
    h_min = (h_floor_bp / 100.0) ** 2 if h_floor_bp else 1e-10
    p = DNSParams(p.mu, p.A, p.Q, np.maximum(p.h, h_min), p.lam if estimate_lambda else lam)
    k = p.k

    # Phase 1: EM.
    prev = -np.inf
    em_done = 0
    for em_done in range(1, em_iters + 1):
        st = smoothed_stats(p.statespace(tau), Y)
        if not np.isfinite(st.loglik) or st.loglik - prev < em_tol:
            break
        prev = st.loglik
        p_new = em_update(p, st, tau, h_min=h_min)
        if not np.all(np.isfinite(p_new.A)) or np.min(np.linalg.eigvalsh(p_new.Q)) <= 0:
            break
        p = p_new

    # Phase 2: quasi-Newton on the exact likelihood and score.
    def fun(theta):
        q = DNSParams.from_vector(theta, n, estimate_lambda, lam, k)
        st = smoothed_stats(q.statespace(tau), Y)
        if not np.isfinite(st.loglik):
            return 1e6, np.zeros_like(theta)
        return -st.loglik / n_obs, -score(theta, st, tau, estimate_lambda, lam) / n_obs

    theta0 = p.to_vector(estimate_lambda)
    n_dyn = k + k * k + k * (k + 1) // 2
    bounds = [(None, None)] * n_dyn + [(np.log(h_min) if h_floor_bp else None, None)] * n
    bounds += [(None, None)] * (theta0.size - len(bounds))
    # L-BFGS-B can declare convergence when its steps shrink on an ill-conditioned
    # surface; restarting resets its curvature memory. Stop when a restart gains
    # less than 1e-3 log-likelihood points.
    res = None
    restarts = 0
    for restarts in range(1, max_restarts + 1):
        r = minimize(fun, theta0 if res is None else res.x, jac=True, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": maxiter, "maxfun": 2 * maxiter, "ftol": 1e-13, "gtol": 1e-8})
        gain = np.inf if res is None else (res.fun - r.fun) * n_obs
        res = r if res is None or r.fun < res.fun else res
        if gain < 1e-3:
            break
    params = DNSParams.from_vector(res.x, n, estimate_lambda, lam, k)
    return DNSFit(params=params, loglik=-res.fun * n_obs, n_params=theta0.size, nobs=n_obs,
                  converged=bool(res.success),
                  message=f"EM {em_done} it; L-BFGS-B x{restarts}: {res.message}",
                  seconds=time.perf_counter() - t0, columns=cols, estimate_lambda=estimate_lambda)
