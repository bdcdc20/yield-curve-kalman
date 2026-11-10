"""A small linear-Gaussian Kalman filter and RTS smoother that handle missing observations.

State-space form (Durbin & Koopman 2012 timing, as in statsmodels):

    y_t     = d + Z x_t + eps_t,        eps_t ~ N(0, H)     (measurement)
    x_{t+1} = c + T x_t + eta_t,        eta_t ~ N(0, Q)     (transition)
    x_1     ~ N(a1, P1)                                      (prior on the first state)

For the DNS model, x_t is (level, slope, curvature), Z is the matrix of
Nelson-Siegel loadings, and c = (I - A) mu, T = A.

Missing data: at each t, only the observed rows of y_t, Z, d and H enter the
update. If nothing is observed, the update is skipped and the filtered state
equals the predicted one. The likelihood then counts only what was actually
observed, which is the right treatment for the GBP short-end gaps and for
macro data not yet released (README sections 5.2 and 7.3). Nothing is
interpolated.

The filter is written out explicitly rather than taken from a library,
because it is the core of the project. tests/test_kalman.py checks it
against statsmodels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_discrete_lyapunov

LOG_2PI = np.log(2.0 * np.pi)


@dataclass
class StateSpace:
    """System matrices. n = number of observed series, k = number of states."""

    Z: np.ndarray  # (n, k)
    H: np.ndarray  # (n, n)
    T: np.ndarray  # (k, k)
    Q: np.ndarray  # (k, k)
    a1: np.ndarray  # (k,)
    P1: np.ndarray  # (k, k)
    d: np.ndarray = field(default=None)  # (n,), default 0
    c: np.ndarray = field(default=None)  # (k,), default 0

    def __post_init__(self):
        n, k = np.shape(self.Z)
        self.d = np.zeros(n) if self.d is None else np.asarray(self.d, float)
        self.c = np.zeros(k) if self.c is None else np.asarray(self.c, float)


@dataclass
class FilterResult:
    loglik: float
    loglik_t: np.ndarray  # (N,) contribution of each period (0 if nothing observed)
    a_pred: np.ndarray  # (N+1, k) a_{t|t-1}; the last row is the one-step forecast past the sample
    P_pred: np.ndarray  # (N+1, k, k)
    a_filt: np.ndarray  # (N, k) a_{t|t}
    P_filt: np.ndarray  # (N, k, k)
    n_obs: np.ndarray  # (N,) number of observed series each period


def stationary_init(T: np.ndarray, Q: np.ndarray, c: np.ndarray | None = None):
    """Unconditional mean and covariance of x_t = c + T x_{t-1} + eta, as (a1, P1).

    Solves P = T P T' + Q (discrete Lyapunov equation). This is the natural
    prior for a stationary model: before seeing any data, the state is a draw
    from its long-run distribution. Raises ValueError if T has an eigenvalue
    on or outside the unit circle, because then that distribution does not
    exist and a diffuse prior is needed instead.
    """
    k = T.shape[0]
    if np.max(np.abs(np.linalg.eigvals(T))) >= 1.0:
        raise ValueError("transition matrix is not stable; no stationary distribution")
    c = np.zeros(k) if c is None else np.asarray(c, float)
    a1 = np.linalg.solve(np.eye(k) - T, c)
    P1 = solve_discrete_lyapunov(T, Q)
    return a1, (P1 + P1.T) / 2


def kalman_filter(model: StateSpace, y) -> FilterResult:
    """Run the filter over y (N x n, NaN = missing) and return states and log-likelihood.

    The log-likelihood is the prediction-error decomposition:
    sum_t -0.5 * (n_t log 2pi + log|F_t| + v_t' F_t^{-1} v_t),
    where v_t is the one-step prediction error of the observed y_t and F_t
    its covariance. F_t is factorised by Cholesky, which is faster and more
    stable than inverting it, and fails loudly (LinAlgError) if a parameter
    guess makes F_t not positive definite.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    N = y.shape[0]
    k = model.T.shape[0]
    Z, H, T, Q, d, c = model.Z, model.H, model.T, model.Q, model.d, model.c

    a_pred = np.empty((N + 1, k))
    P_pred = np.empty((N + 1, k, k))
    a_filt = np.empty((N, k))
    P_filt = np.empty((N, k, k))
    ll_t = np.zeros(N)
    n_obs = np.zeros(N, dtype=int)

    a, P = np.asarray(model.a1, float), np.asarray(model.P1, float)
    for t in range(N):
        a_pred[t], P_pred[t] = a, P
        obs = np.isfinite(y[t])
        n_t = int(obs.sum())
        n_obs[t] = n_t
        if n_t:
            Zt = Z[obs]
            v = y[t, obs] - d[obs] - Zt @ a
            PZ = P @ Zt.T  # (k, n_t)
            F = Zt @ PZ + H[np.ix_(obs, obs)]
            cf = cho_factor(F, lower=True, check_finite=False)
            Finv_v = cho_solve(cf, v, check_finite=False)
            K = cho_solve(cf, PZ.T, check_finite=False).T  # P Z' F^{-1}
            a = a + K @ v
            P = P - K @ PZ.T
            P = (P + P.T) / 2  # keep it symmetric against rounding drift
            logdet = 2.0 * np.log(np.diag(cf[0])).sum()
            ll_t[t] = -0.5 * (n_t * LOG_2PI + logdet + v @ Finv_v)
        a_filt[t], P_filt[t] = a, P
        a = c + T @ a
        P = T @ P @ T.T + Q
    a_pred[N], P_pred[N] = a, P

    return FilterResult(loglik=float(ll_t.sum()), loglik_t=ll_t, a_pred=a_pred, P_pred=P_pred,
                        a_filt=a_filt, P_filt=P_filt, n_obs=n_obs)


def _loglik_loop(y, Z, H, d, T, Q, c, a1, P1):
    """Log-likelihood only: the same recursion as `kalman_filter`, written for numba.

    Maximum likelihood evaluates the filter thousands of times, and in pure Python
    the per-period overhead (not the 17x17 algebra) dominates. Compiled, this is
    the same arithmetic without that overhead. It returns -inf when F_t is not
    positive definite, so the optimiser simply rejects that parameter guess.
    """
    N, n = y.shape
    a = a1.copy()
    P = P1.copy()
    ll = 0.0
    for t in range(N):
        idx = np.empty(n, np.int64)
        m = 0
        for j in range(n):
            if np.isfinite(y[t, j]):
                idx[m] = j
                m += 1
        if m > 0:
            idx = idx[:m]
            Zt = np.ascontiguousarray(Z[idx])
            Ht = np.ascontiguousarray(H[idx][:, idx])
            v = y[t][idx] - d[idx] - Zt @ a
            PZ = P @ Zt.T
            F = Zt @ PZ + Ht
            ok = True
            for i in range(m):
                if not F[i, i] > 0.0:
                    ok = False
            if not ok:
                return -np.inf
            # Cholesky by hand: numba raises (and cannot be caught cheaply) on non-PD input.
            L = np.zeros((m, m))
            for i in range(m):
                for j in range(i + 1):
                    s = F[i, j]
                    for q in range(j):
                        s -= L[i, q] * L[j, q]
                    if i == j:
                        if s <= 0.0:
                            return -np.inf
                        L[i, i] = np.sqrt(s)
                    else:
                        L[i, j] = s / L[j, j]
            Finv_v = np.linalg.solve(F, v)
            K = np.linalg.solve(F, np.ascontiguousarray(PZ.T)).T
            a = a + K @ v
            P = P - K @ PZ.T
            P = 0.5 * (P + P.T)
            ll += -0.5 * (m * LOG_2PI + 2.0 * np.log(np.diag(L)).sum() + v @ Finv_v)
        a = c + T @ a
        P = T @ P @ T.T + Q
    return ll


try:
    from numba import njit

    _loglik_fast = njit(cache=True)(_loglik_loop)
except ImportError:  # numba is optional: fall back to the readable filter
    _loglik_fast = None


def kalman_loglik(model: StateSpace, y) -> float:
    """Log-likelihood of y under `model`; identical to kalman_filter(...).loglik but much faster.

    Returns -inf (instead of raising) when the parameters make F_t singular.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if _loglik_fast is None:
        try:
            return kalman_filter(model, y).loglik
        except np.linalg.LinAlgError:
            return -np.inf
    f = lambda x: np.ascontiguousarray(x, dtype=float)
    return float(_loglik_fast(f(y), f(model.Z), f(model.H), f(model.d), f(model.T), f(model.Q),
                              f(model.c), f(model.a1), f(model.P1)))


def rts_smoother(model: StateSpace, res: FilterResult):
    """Rauch-Tung-Striebel smoother: E[x_t | all data] and its covariance, as (a_smooth, P_smooth).

    Smoothed states use future observations, so they are for plots and for
    reading the factors in-sample only. Out-of-sample forecasts must start from
    filtered states (README section 9).
    """
    N, k = res.a_filt.shape
    a_s = res.a_filt.copy()
    P_s = res.P_filt.copy()
    for t in range(N - 2, -1, -1):
        # J_t = P_{t|t} T' P_{t+1|t}^{-1}
        J = np.linalg.solve(res.P_pred[t + 1], model.T @ res.P_filt[t]).T
        a_s[t] = res.a_filt[t] + J @ (a_s[t + 1] - res.a_pred[t + 1])
        P_s[t] = res.P_filt[t] + J @ (P_s[t + 1] - res.P_pred[t + 1]) @ J.T
        P_s[t] = (P_s[t] + P_s[t].T) / 2
    return a_s, P_s


def forecast_state(model: StateSpace, a_filt_t: np.ndarray, h: int) -> np.ndarray:
    """h-step-ahead state forecast from the filtered state at t: c + T c + ... + T^h a_{t|t}."""
    a = np.asarray(a_filt_t, float)
    for _ in range(h):
        a = model.c + model.T @ a
    return a
