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
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize

from yck.model.dns import (DL_LAMBDA, FACTORS, fit_dynamics, fit_factors, fitted_yields, maturities_from_columns,
                           ns_loadings, ns_loadings_dlam)
from yck.model.kalman import (FilterResult, SmootherStats, StateSpace, filtered_states, forecast_state,
                              kalman_filter, rts_smoother, smoothed_stats, stationary_init)

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
    penalty: float = 0.0  # log prior of the shrinkage (<= 0); loglik excludes it

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

    def filtered_states(self, data: pd.DataFrame) -> np.ndarray:
        """a_{t|t} for every row of `data`, from the compiled filter (fast; for backtests)."""
        return filtered_states(self.statespace(), data[self.columns].to_numpy(float))

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


def score_fd(theta, st: SmootherStats, tau, estimate_lambda: bool = False, lam: float = DL_LAMBDA,
             eps: float = 1e-5) -> np.ndarray:
    """Reference score: central differences of `expected_complete_loglik` (slow; used to test `score`)."""
    theta = np.asarray(theta, float)
    g = np.empty_like(theta)
    for i in range(theta.size):
        e = np.zeros_like(theta)
        e[i] = eps
        g[i] = (expected_complete_loglik(theta + e, st, tau, estimate_lambda, lam)
                - expected_complete_loglik(theta - e, st, tau, estimate_lambda, lam)) / (2 * eps)
    return g


def score(theta, st: SmootherStats, tau, estimate_lambda: bool = False, lam: float = DL_LAMBDA) -> np.ndarray:
    """Exact gradient of the log-likelihood at theta, in closed form (Fisher's identity).

    The gradient of the log-likelihood equals the gradient of the expected
    complete-data log-likelihood with the smoother statistics held fixed.
    Each of its terms has a matrix-calculus derivative:

    - measurement: d/dlog h_j = -1/2 (n_j - S_j / h_j); for lambda, S_j depends on
      the loadings Z_j(lambda) and d/dlambda uses `ns_loadings_dlam`;
    - transition: d/dB = Q^{-1} (S10 - B S00) with B = [c, A], c = (I - A) mu, and
      d/dQ = -1/2 [(N-1) Q^{-1} - Q^{-1} W Q^{-1}];
    - first state x_1 ~ N(mu, P1), where P1 solves P1 = A P1 A' + Q: d/dmu =
      P1^{-1} (E x_1 - mu); with G = d/dP1, one adjoint Lyapunov solve
      Lam = A' Lam A + G gives d/dQ += Lam and d/dA += 2 Lam A P1.

    Q = L L' with log-diagonal L maps through d/dL = 2 G_Q L. Tested against
    `score_fd` and against numerical gradients of the Kalman log-likelihood.
    """
    n = len(tau)
    k = st.S11.shape[0]
    p = DNSParams.from_vector(theta, n, estimate_lambda, lam, k)
    ss = p.statespace(tau)
    A, mu = p.A, p.mu
    Ik = np.eye(k)

    # Measurement (yield rows only).
    Z = ss.Z[:n, :3]
    sya, sxx = st.sya[:n, :3], st.sxx[:n, :3, :3]
    S = st.syy[:n] - 2.0 * np.einsum("jk,jk->j", Z, sya) + np.einsum("jk,jkl,jl->j", Z, sxx, Z)
    g_logh = -0.5 * (st.n_i[:n] - S / p.h)

    # Transition.
    _, Qinv = _logdet_and_inv(p.Q)
    B = np.column_stack([ss.c, A])
    W = st.S11 - B @ st.S10.T - st.S10 @ B.T + B @ st.S00 @ B.T
    dB = Qinv @ (st.S10 - B @ st.S00)
    dc, dA = dB[:, 0], dB[:, 1:].copy()
    dmu = (Ik - A).T @ dc
    dA -= np.outer(dc, mu)
    dQ = -0.5 * (st.n_trans * Qinv - Qinv @ W @ Qinv)

    # First state.
    x1 = st.a_s[0]
    _, P1inv = _logdet_and_inv(ss.P1)
    dmu += P1inv @ (x1 - ss.a1)
    if np.max(np.abs(np.linalg.eigvals(A))) < 1.0:  # stationary prior: P1 depends on A and Q
        M = np.outer(x1 - ss.a1, x1 - ss.a1) + st.P_s[0]
        G1 = -0.5 * (P1inv - P1inv @ M @ P1inv)
        Lam = solve_discrete_lyapunov(A.T, G1)
        Lam = (Lam + Lam.T) / 2
        dQ += Lam
        dA += 2.0 * Lam @ A @ ss.P1

    # Q = L L' with log diagonal.
    (rows, cols), diag = _tril(k)
    Lq = np.linalg.cholesky(p.Q)
    dQ = (dQ + dQ.T) / 2
    dL = (2.0 * dQ @ Lq)[rows, cols]
    dL[diag] *= Lq[rows, cols][diag]

    parts = [dmu, dA.ravel(), dL, g_logh]
    if estimate_lambda:
        dZ = ns_loadings_dlam(tau, p.lam)
        dS = -2.0 * np.einsum("jk,jk->j", dZ, sya) + 2.0 * np.einsum("jk,jkl,jl->j", dZ, sxx, Z)
        parts.append([p.lam * np.sum(-0.5 * dS / p.h)])
    return np.concatenate(parts)


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
    keep: int = 2,
    prune_iters: int = 100,
    shrink_tau: float | None = None,
    **kwargs,
) -> DNSFit:
    """Maximum-likelihood estimation of the DNS state-space model (yields only, or yields + macro).

    `data` holds yield columns ('3m', ..., '120m') and optionally macro
    columns, which turn the model into the DRA yields + macro model.

    Each start runs EM and then L-BFGS-B with the exact score (`_fit_from`).
    With `n_starts > 1`, the extra starts are random perturbations of the
    first (fixed `seed`, so results are reproducible). All starts are scouted
    briefly (EM + `prune_iters` quasi-Newton iterations) and only the best
    `keep` are run to convergence: branch-and-bound style pruning, which costs
    a fraction of optimising every start fully.

    **Measurement-noise floor.** By default each maturity's noise sd is
    bounded below by `h_floor_bp` = 1 bp. Without it, the USD likelihood
    pushes a few maturities to ~0.01 bp: GSW yields are points on a smooth
    fitted Svensson curve, so three factors can match some of them almost
    exactly. That makes the likelihood so badly conditioned that quasi-Newton
    methods stop with a large gradient, and leaves scattered local maxima.
    No real yield is measured to better than about 1 bp. With the floor,
    every start converges (gradient < 0.02) and starts land on one or two
    reproducible optima. Pass `h_floor_bp=None` for the unrestricted model.

    **Shrinkage** (`shrink_tau`, yields + macro only). The full VAR between
    factors and macro has k^2 coefficients, and unrestricted it overfits out of
    sample. A Minnesota-style Gaussian prior pulls the *cross* coefficients
    (macro -> factors and factors -> macro) towards zero, with prior sd
    tau * s_i / s_j, where s are the shock sds of the two-step start (so the
    prior does not depend on units). tau -> 0 separates the curve and macro
    dynamics (yields-only forecasts); tau -> inf is the full DRA model. The fit
    maximises log-likelihood + log-prior (MAP); `loglik` reports the
    likelihood alone and `penalty` the log-prior.
    """
    ycols, mcols = split_columns(data.columns)
    data = data[ycols + mcols]
    n = len(ycols)
    p0 = start or two_step_start(data, lam)
    if shrink_tau is not None and mcols:
        kwargs["shrink"] = (float(shrink_tau), np.sqrt(np.diag(two_step_start(data, lam).Q)))
    if n_starts <= 1:
        return _fit_from(data, n, p0, estimate_lambda, lam, h_floor_bp, **kwargs)

    # Multistart with pruning. Every start gets EM plus a short quasi-Newton
    # burst, which takes it to within ~1 log-likelihood point of its own optimum,
    # while distinct optima are tens to hundreds of points apart. Only the best
    # `keep` starts are then run to full convergence; the rest are abandoned.
    rng = np.random.default_rng(seed)
    starts = [p0] + [_perturb(p0, rng) for _ in range(n_starts - 1)]
    t0 = time.perf_counter()
    scouted = [_fit_from(data, n, p, estimate_lambda, lam, h_floor_bp, maxiter=prune_iters, max_restarts=1,
                         **{k: v for k, v in kwargs.items() if k not in ("maxiter", "max_restarts")})
               for p in starts]
    objective = lambda f: f.loglik + f.penalty if np.isfinite(f.loglik) else -np.inf
    scouted.sort(key=lambda f: -objective(f))
    best = None
    for f in scouted[:keep]:
        fit = _fit_from(data, n, f.params, estimate_lambda, lam, h_floor_bp, em_iters=0,
                        **{k: v for k, v in kwargs.items() if k != "em_iters"})
        if best is None or objective(fit) > objective(best):
            best = fit
    best.seconds = time.perf_counter() - t0
    best.message = f"{n_starts} starts, best {keep} finished; " + best.message
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
    grad_tol: float = 0.005,
    shrink: tuple | None = None,
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
    a_slice = slice(k, k + k * k)  # A's position in theta (row-major)
    l_slice = slice(k + k * k, k + k * k + k * (k + 1) // 2)  # Cholesky factor of Q

    def fun(theta):
        q = DNSParams.from_vector(theta, n, estimate_lambda, lam, k)
        st = smoothed_stats(q.statespace(tau), Y)
        if not np.isfinite(st.loglik):
            return 1e6, np.zeros_like(theta)
        g = score(theta, st, tau, estimate_lambda, lam)
        pen = 0.0
        if shrink is not None:
            pen, dA, dl = shrinkage_prior(q, *shrink)
            g[a_slice] += dA.ravel()
            g[l_slice] += dl
        return -(st.loglik + pen) / n_obs, -g / n_obs

    theta0 = p.to_vector(estimate_lambda)
    n_dyn = k + k * k + k * (k + 1) // 2
    bounds = [(None, None)] * n_dyn + [(np.log(h_min) if h_floor_bp else None, None)] * n
    bounds += [(None, None)] * (theta0.size - len(bounds))
    # Stopping rules in log-likelihood units: converged when no parameter's
    # gradient exceeds `grad_tol`, or an iteration gains < 1e-7. On USD refits
    # this reaches the optimum of the old, much stricter settings to within
    # 0.01 log-likelihood points at a fraction of the cost. A restart (which
    # resets L-BFGS-B's curvature memory) is only spent when the run stopped with
    # a large gradient, i.e. it stalled rather than converged.
    opts = {"maxiter": maxiter, "maxfun": 2 * maxiter, "ftol": 1e-7 / n_obs, "gtol": grad_tol / n_obs}
    res = None
    restarts = 0
    for restarts in range(1, max_restarts + 1):
        r = minimize(fun, theta0 if res is None else res.x, jac=True, method="L-BFGS-B", bounds=bounds,
                     options=opts)
        res = r if res is None or r.fun < res.fun else res
        free = np.array([not (lo is not None and x <= lo + 1e-9) for x, (lo, _) in zip(res.x, bounds)])
        # Converged, or a restart that made no progress: the gradient is at its
        # numerical floor (~0.01 log-lik units on USD), so further restarts are waste.
        if np.max(np.abs(res.jac[free]), initial=0.0) * n_obs < grad_tol or (restarts > 1 and r.nit <= 1):
            break
    params = DNSParams.from_vector(res.x, n, estimate_lambda, lam, k)
    pen = shrinkage_prior(params, *shrink)[0] if shrink is not None else 0.0
    return DNSFit(params=params, loglik=-res.fun * n_obs - pen, n_params=theta0.size, nobs=n_obs,
                  converged=bool(res.success),
                  message=f"EM {em_done} it; L-BFGS-B x{restarts}: {res.message}",
                  seconds=time.perf_counter() - t0, columns=cols, estimate_lambda=estimate_lambda,
                  penalty=pen)


def cross_penalty(A: np.ndarray, tau: float, scales: np.ndarray) -> tuple[float, np.ndarray]:
    """Log of the Gaussian shrinkage prior on the factor<->macro cross terms of A, and its gradient.

    Entries A[i, j] with exactly one of i, j among the 3 factors get prior
    N(0, (tau * s_i / s_j)^2); all other entries are unpenalised.
    """
    k = A.shape[0]
    idx = np.arange(k)
    cross = (idx[:, None] < 3) != (idx[None, :] < 3)
    w = np.where(cross, 1.0 / (tau * scales[:, None] / scales[None, :]) ** 2, 0.0)
    return float(-0.5 * np.sum(w * A**2)), -w * A


def shrinkage_prior(p: DNSParams, tau: float, scales: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Log prior shrinking every factor<->macro link to zero, with gradients w.r.t. A and the Q-Cholesky vector.

    Two blocks are shrunk:

    - the lagged cross terms of A (`cross_penalty`);
    - the factor-macro block of the shock covariance. With factors ordered
      first, Q = L L' has a zero factor-macro block exactly when
      L[macro, factor] = 0, so those entries get prior N(0, (tau * s_i)^2),
      roughly a prior of tau on the shock correlations.

    With both blocks at zero the likelihood factorises into the yields-only
    model times an independent macro VAR. So tau -> 0 reproduces the yields-only
    forecasts, and tau -> inf is the full DRA model. Shrinking A alone does
    not: correlated shocks still let macro data re-estimate the curve dynamics.
    """
    k = p.k
    pen_A, dA = cross_penalty(p.A, tau, scales)
    (rows, cols), diag = _tril(k)
    L = np.linalg.cholesky(p.Q)
    l = L[rows, cols]
    cross = (rows >= 3) & (cols < 3)
    w = np.where(cross, 1.0 / (tau * scales[rows]) ** 2, 0.0)
    pen_L = -0.5 * np.sum(w * l**2)
    dl = -w * l  # no diagonal entries are penalised, so the log-diagonal transform does not enter
    return pen_A + float(pen_L), dA, dl
