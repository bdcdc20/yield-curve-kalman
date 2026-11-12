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

A is left free in theta, and the model is specified as stationary: the
first state has the stationary prior, whose variance grows without bound
as an eigenvalue of A approaches 1. For unstable trial values the filter
falls back to a fixed diffuse prior, so it can evaluate any parameters, but
that makes the likelihood jump *up* at the unit circle. On short or trending
samples (EUR; GBP; the shrunk USD macro model) the optimiser crossed and
stopped at explosive A. The estimate is therefore required to be stationary
(spectral radius < `MAX_RADIUS`): the search runs unrestricted, because some
starts reach a stationary optimum only through the explosive region, and a
run that ends explosive is redone with explosive values rejected, starting
from A scaled back inside (`_fit_from`).

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

from yck.model.dns import (DL_LAMBDA, FACTORS, factor_names, fit_dynamics, fit_factors, fitted_yields, maturities_from_columns,
                           ns_loadings, ns_loadings_dlam)
from yck.model.kalman import (FilterResult, SmootherStats, StateSpace, filtered_states, forecast_state,
                              kalman_filter, lyapunov, rts_smoother, smoothed_stats, stationary_init)

DIFFUSE_VAR = 100.0  # prior variance (pp^2) when A is not stable: wide relative to yield levels
MACRO_H = 1e-8  # measurement variance of macro series: "observed without error", kept invertible
MAX_RADIUS = 0.9999  # estimation is restricted to A with spectral radius below this (stationary model)


def spectral_radius(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(A))))


BOUNDARY_RADIUS = MAX_RADIUS * (1.0 - 1e-9)  # "on the cap", just inside the rejection test


def radius_gradient(B: np.ndarray) -> np.ndarray:
    """d rho(B) / dB for the spectral radius, assuming a simple dominant eigenvalue (or complex pair).

    With B = V diag(w) V^{-1}, the rows y of V^{-1} are left eigenvectors
    normalised so y'v = 1, and first-order perturbation gives
    d lambda = y' dB v. For rho = |lambda|, d rho = Re(conj(lambda) d lambda) / |lambda|.
    A complex pair gives the same real result from either member.
    """
    w, V = np.linalg.eig(B)
    i = int(np.argmax(np.abs(w)))
    y = np.linalg.inv(V)[i]
    return np.real(np.conj(w[i]) / np.abs(w[i]) * np.outer(y, V[:, i]))


def boundary_chain(gA: np.ndarray, B: np.ndarray, radius: float = BOUNDARY_RADIUS) -> np.ndarray:
    """Gradient w.r.t. B of a function of A = radius * B / rho(B), given its gradient gA w.r.t. A.

    dA = (c / rho) dB - (c / rho^2) B d rho, so
    df/dB = (c / rho) gA - (c / rho^2) <gA, B> d rho / dB. The result is orthogonal
    to B (scaling B does not move A), by Euler's identity <d rho / dB, B> = rho.
    """
    r = spectral_radius(B)
    return (radius / r) * gA - (radius / r**2) * np.sum(gA * B) * radius_gradient(B)


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
    lam2: float | None = None  # second curvature decay: the 4-factor dynamic Svensson model (step 7b item 8)

    @property
    def k(self) -> int:
        return self.mu.size

    @property
    def nf(self) -> int:
        """Number of yield factors: 3 (Nelson-Siegel) or 4 (Svensson)."""
        return 3 if self.lam2 is None else 4

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
                    k: int = 3, lam2: float | None = None) -> "DNSParams":
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
        return cls(mu=mu, A=A, Q=L @ L.T, h=h, lam=lam, lam2=lam2)

    # ----------------------------------------------------------------- model
    def statespace(self, tau) -> StateSpace:
        """The Kalman-filter representation: yields at maturities tau (months), then the macro series."""
        k, nf = self.k, self.nf
        q = k - nf
        n = len(tau)
        Z = np.zeros((n + q, k))
        Z[:n, :nf] = ns_loadings(tau, self.lam, self.lam2)
        Z[n:, nf:] = np.eye(q)
        H = np.diag(np.concatenate([self.h, np.full(q, MACRO_H)]))
        c = (np.eye(k) - self.A) @ self.mu
        try:
            a1, P1 = stationary_init(self.A, self.Q, c)
        except ValueError:
            a1, P1 = self.mu.copy(), DIFFUSE_VAR * np.eye(k)
        return StateSpace(Z=Z, H=H, T=self.A, Q=self.Q, a1=a1, P1=P1, c=c)


def two_step_start(data: pd.DataFrame, lam: float = DL_LAMBDA, lam2: float | None = None) -> DNSParams:
    """Starting values: OLS factors (step 1) and a VAR(1) on factors plus any macro columns.

    Measurement variances come from the cross-sectional residuals.
    """
    ycols, mcols = split_columns(data.columns)
    yields = data[ycols]
    f = fit_factors(yields, lam, lam2=lam2)
    states = pd.concat([f, data[mcols]], axis=1) if mcols else f
    dyn = fit_dynamics(states, "var1")
    resid = yields - fitted_yields(f, ycols, lam, lam2)
    h = np.maximum((resid**2).mean().to_numpy(), 1e-6)  # floor: log(0) would break the transform
    return DNSParams(mu=dyn.mu, A=dyn.A, Q=dyn.Q, h=h, lam=lam, lam2=lam2)


def sample_mean(data: pd.DataFrame, lam: float = DL_LAMBDA, lam2: float | None = None) -> np.ndarray:
    """Sample mean of the states: the month-by-month OLS factors (step 1 of Diebold-Li) and the observed macro series.

    Used as the fixed long-run mean mu (`fit_dns_mle(mu="sample")`). It uses only
    the rows of `data`, so in a backtest it is the mean of what was known at
    the origin.
    """
    ycols, mcols = split_columns(data.columns)
    f = fit_factors(data[ycols], lam, lam2=lam2)
    means = [f[c].mean() for c in f.columns] + [data[c].mean() for c in mcols]
    return np.asarray(means, float)


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
    mu_fixed: bool = False  # mu set to the sample mean rather than estimated

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
        return factor_names(self.params.lam2) + split_columns(self.columns)[1]

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

    def fitted(self, a: np.ndarray) -> np.ndarray:
        """Model yields at the states in the rows of a (yield maturities only)."""
        return np.atleast_2d(a) @ self.statespace().Z[:len(self.yield_columns)].T

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


def score(theta, st: SmootherStats, tau, estimate_lambda: bool = False, lam: float = DL_LAMBDA,
          params: DNSParams | None = None, model: StateSpace | None = None) -> np.ndarray:
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
    # The optimiser has already built these for the likelihood; pass them to skip a rebuild
    # (which includes a Lyapunov solve for the stationary prior).
    p = params if params is not None else DNSParams.from_vector(theta, n, estimate_lambda, lam, k)
    ss = model if model is not None else p.statespace(tau)
    A, mu = p.A, p.mu
    Ik = np.eye(k)

    # Measurement (yield rows only).
    nf = p.nf
    Z = ss.Z[:n, :nf]
    sya, sxx = st.sya[:n, :nf], st.sxx[:n, :nf, :nf]
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
        Lam = lyapunov(A.T, G1)
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


def em_update(p: DNSParams, st: SmootherStats, tau, h_min: float = 1e-10,
              mu_fixed: np.ndarray | None = None) -> DNSParams:
    """One EM M-step: closed-form A, mu, Q, H given the smoother statistics (lambda unchanged).

    The transition is a regression of x_t on (1, x_{t-1}) in expectation:
    B = S10 S00^{-1}, Q = (S11 - B S10') / (N-1). Each maturity's variance is
    its expected squared residual, averaged over the months it is observed.
    The first-state term is left out of the M-step (standard practice); the
    gradient phase that follows maximises the exact likelihood.
    """
    n, k = len(tau), p.k
    if mu_fixed is None:
        B = np.linalg.solve(st.S00.T, st.S10.T).T
        c, A = B[:, 0], B[:, 1:]
        Q = (st.S11 - B @ st.S10.T) / st.n_trans
    else:
        # mu known: regress z_t = x_t - mu on z_{t-1} without intercept. The sums of
        # E[z z'] follow from the same statistics: with m1 = sum E x_t, m0 = sum E x_{t-1}
        # and M transitions, sum E[z_t z_{t-1}'] = S10[:, 1:] - m1 mu' - mu m0' + M mu mu', etc.
        mu, M = mu_fixed, st.n_trans
        m1, m0 = st.S10[:, 0], st.S00[1:, 0]
        Z10 = st.S10[:, 1:] - np.outer(m1, mu) - np.outer(mu, m0) + M * np.outer(mu, mu)
        Z00 = st.S00[1:, 1:] - np.outer(m0, mu) - np.outer(mu, m0) + M * np.outer(mu, mu)
        Z11 = st.S11 - np.outer(m1, mu) - np.outer(mu, m1) + M * np.outer(mu, mu)
        A = np.linalg.solve(Z00.T, Z10.T).T
        Q = (Z11 - A @ Z10.T) / M
    Q = (Q + Q.T) / 2
    nf = p.nf
    Z = ns_loadings(tau, p.lam, p.lam2)
    S = st.syy[:n] - 2.0 * np.einsum("jk,jk->j", Z, st.sya[:n, :nf]) \
        + np.einsum("jk,jkl,jl->j", Z, st.sxx[:n, :nf, :nf], Z)
    h = np.maximum(S / np.maximum(st.n_i[:n], 1.0), h_min)
    if mu_fixed is not None:
        return DNSParams(mu=mu_fixed.copy(), A=A, Q=Q, h=h, lam=p.lam, lam2=p.lam2)
    try:
        mu = np.linalg.solve(np.eye(k) - A, c)
    except np.linalg.LinAlgError:
        mu = p.mu
    return DNSParams(mu=mu, A=A, Q=Q, h=h, lam=p.lam, lam2=p.lam2)


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
    mu: str = "sample",
    rw_tau: float | None = None,
    lam2: float | None = None,
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

    **Random-walk prior** (`rw_tau`, README section 8 step 7b item 5). A
    Minnesota prior centred on A = I, i.e. on every state following a random
    walk (Carriero, Kapetanios & Marcellino 2012; Duffee 2011 finds a random
    walk for the level the most useful restriction): own lags ~ N(1, tau^2),
    cross lags ~ N(0, (RW_THETA tau s_i / s_j)^2), no prior on Q. Small tau
    gives near-random-walk forecasts, large tau the MLE.
    """
    ycols, mcols = split_columns(data.columns)
    data = data[ycols + mcols]
    n = len(ycols)
    if lam2 is not None and estimate_lambda:
        raise ValueError("lambda is fixed in the Svensson (lam2) model")
    p0 = start or two_step_start(data, lam, lam2)
    if mu == "sample":
        kwargs["mu_fixed"] = sample_mean(data, lam, p0.lam2)
        p0 = DNSParams(kwargs["mu_fixed"].copy(), p0.A, p0.Q, p0.h, p0.lam, p0.lam2)
    elif mu != "mle":
        raise ValueError(f"mu must be 'sample' or 'mle', not {mu!r}")
    if shrink_tau is not None and rw_tau is not None:
        raise ValueError("choose one prior: shrink_tau (cross terms to zero) or rw_tau (random walk)")
    if shrink_tau is not None and mcols:
        kwargs["shrink"] = (float(shrink_tau), np.sqrt(np.diag(two_step_start(data, lam, lam2).Q)), "cross")
    if rw_tau is not None:
        kwargs["shrink"] = (float(rw_tau), np.sqrt(np.diag(two_step_start(data, lam, lam2).Q)), "rw")
    if n_starts <= 1:
        return _fit_from(data, n, p0, estimate_lambda, lam, h_floor_bp, **kwargs)

    # Multistart with pruning. Every start gets EM plus a short quasi-Newton
    # burst, which takes it to within ~1 log-likelihood point of its own optimum,
    # while distinct optima are tens to hundreds of points apart. Only the best
    # `keep` starts are then run to full convergence; the rest are abandoned.
    rng = np.random.default_rng(seed)
    starts = [p0] + [_perturb(p0, rng) for _ in range(n_starts - 1)]
    if "mu_fixed" in kwargs:  # perturb everything except the fixed mean
        starts = [DNSParams(p0.mu.copy(), q.A, q.Q, q.h, q.lam, q.lam2) for q in starts]
    t0 = time.perf_counter()
    scouted = [_fit_from(data, n, p, estimate_lambda, lam, h_floor_bp, maxiter=prune_iters, max_restarts=1,
                         **{k: v for k, v in kwargs.items() if k not in ("maxiter", "max_restarts")})
               for p in starts]
    objective = lambda f: f.loglik + f.penalty if np.isfinite(f.loglik) else -np.inf
    scouted.sort(key=lambda f: -objective(f))
    # Note: finishing two scouts from the same basin is not wasted on the flat
    # ridges of the macro likelihood; the second often gets further (tested:
    # finishing one per basin was 14% faster but lost 0.2-0.6 log-lik points).
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
                     h=p.h * np.exp(rng.normal(scale=1.0, size=p.h.size)), lam=p.lam * np.exp(rng.normal(scale=0.2)),
                     lam2=p.lam2)


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
    mu_fixed: np.ndarray | None = None,
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
    p = DNSParams(p.mu, p.A, p.Q, np.maximum(p.h, h_min), p.lam if estimate_lambda else lam, p.lam2)
    k, lam2 = p.k, p.lam2

    # Phase 1: EM.
    prev = -np.inf
    em_done = 0
    for em_done in range(1, em_iters + 1):
        st = smoothed_stats(p.statespace(tau), Y)
        if not np.isfinite(st.loglik) or st.loglik - prev < em_tol:
            break
        prev = st.loglik
        p_new = em_update(p, st, tau, h_min=h_min, mu_fixed=mu_fixed)
        if not np.all(np.isfinite(p_new.A)) or np.min(np.linalg.eigvalsh(p_new.Q)) <= 0:
            break
        p = p_new

    # Phase 2: quasi-Newton on the exact likelihood and score.
    a_slice = slice(k, k + k * k)  # A's position in theta (row-major)
    l_slice = slice(k + k * k, k + k * k + k * (k + 1) // 2)  # Cholesky factor of Q

    restrict = False  # set for the repair run below

    def fun(theta):
        q = DNSParams.from_vector(theta, n, estimate_lambda, lam, k, lam2)
        if restrict and (not np.all(np.isfinite(q.A)) or spectral_radius(q.A) >= MAX_RADIUS):
            return 1e6, np.zeros_like(theta)  # outside the stationary region: rejected, the line search backs off
        ss = q.statespace(tau)
        st = smoothed_stats(ss, Y)
        if not np.isfinite(st.loglik):
            return 1e6, np.zeros_like(theta)
        g = score(theta, st, tau, estimate_lambda, lam, params=q, model=ss)
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
    if mu_fixed is not None:  # equal bounds: L-BFGS-B holds these coordinates fixed
        theta0[:k] = mu_fixed
        bounds[:k] = [(m, m) for m in mu_fixed]
    # Stopping rules in log-likelihood units: converged when no parameter's
    # gradient exceeds `grad_tol`, or an iteration gains < 1e-7. On USD refits
    # this reaches the optimum of the old, much stricter settings to within
    # 0.01 log-likelihood points at a fraction of the cost. A restart (which
    # resets L-BFGS-B's curvature memory) is only spent when the run stopped with
    # a large gradient, i.e. it stalled rather than converged.
    opts = {"maxiter": maxiter, "maxfun": 2 * maxiter, "ftol": 1e-7 / n_obs, "gtol": grad_tol / n_obs}

    def optimise(theta_start, objective=None):
        objective = objective or fun
        res, restarts = None, 0
        for restarts in range(1, max_restarts + 1):
            r = minimize(objective, theta_start if res is None else res.x, jac=True, method="L-BFGS-B",
                         bounds=bounds, options=opts)
            res = r if res is None or r.fun < res.fun else res
            free = np.array([not (lo is not None and x <= lo + 1e-9) for x, (lo, _) in zip(res.x, bounds)])
            # Converged, or a restart that made no progress: the gradient is at its
            # numerical floor (~0.01 log-lik units on USD), so further restarts are waste.
            if np.max(np.abs(res.jac[free]), initial=0.0) * n_obs < grad_tol or (restarts > 1 and r.nit <= 1):
                break
        return res, restarts

    # The search itself is unrestricted: some starts reach a stationary optimum
    # only through the explosive region, and blocking that path loses it. A run
    # that *ends* explosive (an artefact of the prior switch at the unit circle)
    # is repaired: A is scaled back inside and the run is redone restricted to
    # stationary A, where the stationary prior keeps it inside.
    res, restarts = optimise(theta0)
    note = ""
    A_end = DNSParams.from_vector(res.x, n, estimate_lambda, lam, k, lam2).A
    if not np.all(np.isfinite(A_end)) or spectral_radius(A_end) >= MAX_RADIUS:
        q = DNSParams.from_vector(res.x, n, estimate_lambda, lam, k, lam2)
        q.A = q.A * 0.99 * MAX_RADIUS / spectral_radius(q.A) if np.all(np.isfinite(q.A)) else p.A * 0.95
        restrict = True
        res, restarts = optimise(q.to_vector(estimate_lambda))
        note = " [ended explosive: re-run restricted to stationary A]"

        # If the restricted optimum presses against the cap, the constraint binds:
        # the best stationary fit lies on the boundary rho(A) = cap (the data want a
        # unit root). A wall that rejects trial points is a poor way to optimise along
        # it (line searches stop wherever they hit it), so optimise on the boundary
        # surface itself, with A = cap * B / rho(B) and B free, and keep the better fit.
        A_res = DNSParams.from_vector(res.x, n, estimate_lambda, lam, k, lam2).A
        if spectral_radius(A_res) > MAX_RADIUS - 1e-4:
            cap = MAX_RADIUS * (1.0 - 1e-9)  # on the cap, just inside the rejection test

            def on_boundary(thB):
                th = thB.copy()
                th[a_slice] = (cap * thB[a_slice].reshape(k, k)
                               / spectral_radius(thB[a_slice].reshape(k, k))).ravel()
                return th

            def fun_boundary(thB):
                B = thB[a_slice].reshape(k, k)
                if not np.all(np.isfinite(B)) or spectral_radius(B) == 0.0:
                    return 1e6, np.zeros_like(thB)
                f, g = fun(on_boundary(thB))
                if f >= 1e6:
                    return f, g
                g = g.copy()
                g[a_slice] = boundary_chain(g[a_slice].reshape(k, k), B, cap).ravel()
                return f, g

            res_b, restarts_b = optimise(res.x, fun_boundary)
            if res_b.fun < res.fun:
                res_b.x = on_boundary(res_b.x)
                res, restarts = res_b, restarts_b
                note = " [ended explosive: optimised on the stationarity boundary]"
    params = DNSParams.from_vector(res.x, n, estimate_lambda, lam, k, lam2)
    pen = shrinkage_prior(params, *shrink)[0] if shrink is not None else 0.0
    return DNSFit(params=params, loglik=-res.fun * n_obs - pen,
                  n_params=theta0.size - (k if mu_fixed is not None else 0), nobs=n_obs,
                  converged=bool(res.success),
                  message=f"EM {em_done} it; L-BFGS-B x{restarts}: {res.message}{note}",
                  seconds=time.perf_counter() - t0, columns=cols, estimate_lambda=estimate_lambda,
                  penalty=pen, mu_fixed=mu_fixed is not None)


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


RW_THETA = 0.5  # Minnesota cross-lag tightness relative to own lags


def rw_prior(A: np.ndarray, tau: float, scales: np.ndarray) -> tuple[float, np.ndarray]:
    """Log of the Minnesota random-walk prior on A and its gradient: A_ii ~ N(1, tau^2),
    A_ij ~ N(0, (RW_THETA tau s_i / s_j)^2)."""
    k = A.shape[0]
    sd = RW_THETA * tau * scales[:, None] / scales[None, :]
    np.fill_diagonal(sd, tau)
    dev = A - np.eye(k)
    w = 1.0 / sd ** 2
    return float(-0.5 * np.sum(w * dev ** 2)), -w * dev


def shrinkage_prior(p: DNSParams, tau: float, scales: np.ndarray, kind: str = "cross") -> tuple[float, np.ndarray, np.ndarray]:
    """Log prior and its gradients w.r.t. A and the Q-Cholesky vector: `cross` (below) or `rw` (`rw_prior`).

    `cross` shrinks every factor<->macro link to zero.

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
    if kind == "rw":
        pen, dA = rw_prior(p.A, tau, scales)
        return pen, dA, np.zeros(k * (k + 1) // 2)
    pen_A, dA = cross_penalty(p.A, tau, scales)
    (rows, cols), diag = _tril(k)
    L = np.linalg.cholesky(p.Q)
    l = L[rows, cols]
    cross = (rows >= 3) & (cols < 3)
    w = np.where(cross, 1.0 / (tau * scales[rows]) ** 2, 0.0)
    pen_L = -0.5 * np.sum(w * l**2)
    dl = -w * l  # no diagonal entries are penalised, so the log-diagonal transform does not enter
    return pen_A + float(pen_L), dA, dl
