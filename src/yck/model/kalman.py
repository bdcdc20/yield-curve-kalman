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
    P1 = lyapunov(T, Q)
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


# Steady-state shortcut (README section 8, step 4d). P_{t|t-1} does not depend on
# the data, only on the parameters and on which series are observed, and for
# these models it converges within a few months. Once successive predicted
# covariances agree to STEADY_TOL (relative) and the observation pattern is
# unchanged, the filter stops recomputing P, its inverse and log-determinants,
# and the smoother reuses its gain. Any change in the pattern recomputes from
# that month on. A negative tolerance switches the shortcut off (used in tests).
STEADY_TOL = 1e-13

try:
    from numba import njit

    _jit = njit(cache=True)
    HAVE_NUMBA = True
except ImportError:  # numba is optional: the loops below then run as plain (slow) Python
    def _jit(f):
        return f

    HAVE_NUMBA = False


@_jit
def _lyapunov_direct(A, Q):
    k = A.shape[0]
    lhs = -np.kron(A, A)
    for i in range(k * k):
        lhs[i, i] += 1.0
    return np.linalg.solve(lhs, Q.ravel().copy()).reshape(k, k)


def lyapunov(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Solve P = A P A' + Q for small A: (I - A kron A) vec(P) = vec(Q).

    The same direct method as `scipy.linalg.solve_discrete_lyapunov` (which
    is what this replaces), compiled: for k <= 6 the 36 x 36 solve itself takes
    microseconds, and scipy's Python wrappers cost ~10x more. It is called
    twice per likelihood evaluation (stationary prior, and the adjoint in the
    score).
    """
    A = np.ascontiguousarray(A, dtype=float)
    Q = np.ascontiguousarray(Q, dtype=float)
    if not HAVE_NUMBA:
        return solve_discrete_lyapunov(A, Q)
    return _lyapunov_direct(A, Q)


@_jit
def _chol_logdet(F):
    """log|F| by Cholesky, or nan if F is not positive definite.

    Written by hand because numba's own Cholesky raises on non-PD input, and
    that cannot be caught cheaply inside compiled code.
    """
    m = F.shape[0]
    L = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1):
            s = F[i, j]
            for q in range(j):
                s -= L[i, q] * L[j, q]
            if i == j:
                if not s > 0.0:
                    return np.nan
                L[i, i] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return 2.0 * np.log(np.diag(L)).sum()


@_jit
def _forward_dense(y, Z, H, d, T, Q, c, a1, P1, a_pred, P_pred, a_filt, P_filt, frz):
    """Covariance-form filter (as in `kalman_filter`), used when H is not diagonal.

    Fills a_pred/P_pred/a_filt/P_filt in place (pass arrays with a zero
    leading dimension to skip storing) and returns the log-likelihood, or -inf
    when F_t is not positive definite, so an optimiser simply rejects that
    parameter guess.
    """
    N, n = y.shape
    store = a_pred.shape[0] > 0
    if frz.shape[0] > 0:
        frz[:] = False  # no steady-state shortcut in the dense form
    a = a1.copy()
    P = P1.copy()
    ll = 0.0
    for t in range(N):
        if store:
            a_pred[t] = a
            P_pred[t] = P
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
            logdet = _chol_logdet(F)
            if np.isnan(logdet):
                return -np.inf
            Finv_v = np.linalg.solve(F, v)
            K = np.linalg.solve(F, np.ascontiguousarray(PZ.T)).T
            a = a + K @ v
            P = P - K @ PZ.T
            P = 0.5 * (P + P.T)
            ll += -0.5 * (m * LOG_2PI + logdet + v @ Finv_v)
        if store:
            a_filt[t] = a
            P_filt[t] = P
        a = c + T @ a
        P = T @ P @ T.T + Q
    return ll


@_jit
def _chol_inv_logdet(M):
    """(inverse, log-determinant) of a small SPD matrix by Cholesky; (empty, nan) if not SPD."""
    m = M.shape[0]
    L = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1):
            s = M[i, j]
            for q in range(j):
                s -= L[i, q] * L[j, q]
            if i == j:
                if not s > 0.0:
                    return np.empty((0, 0)), np.nan
                L[i, i] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    Linv = np.zeros((m, m))  # forward substitution: L Linv = I
    for i in range(m):
        Linv[i, i] = 1.0 / L[i, i]
        for j in range(i):
            s = 0.0
            for q in range(j, i):
                s -= L[i, q] * Linv[q, j]
            Linv[i, j] = s / L[i, i]
    return Linv.T @ Linv, 2.0 * np.log(np.diag(L)).sum()


@_jit
def _max_abs_diff(A, B):
    m = 0.0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            x = abs(A[i, j] - B[i, j])
            if x > m:
                m = x
    return m


@_jit
def _max_abs(A):
    m = 0.0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            x = abs(A[i, j])
            if x > m:
                m = x
    return m


@_jit
def _forward_info(y, Z, h, d, T, Q, c, a1, P1, a_pred, P_pred, a_filt, P_filt, tol, frz):
    """The filter in information form, for diagonal H (all models in this project).

    The covariance form factorises the n x n innovation covariance F_t each
    month (O(n^3), n = 17 yields + macro). With H = diag(h), the Woodbury
    identity and the matrix determinant lemma give everything from k x k
    algebra (k = number of states):

        M_t = P_{t|t-1}^{-1} + Z' H^{-1} Z          (k x k)
        P_{t|t} = M_t^{-1},   a_{t|t} = a_{t|t-1} + M_t^{-1} Z' H^{-1} v_t
        log|F_t| = sum log h + log|P_{t|t-1}| + log|M_t|
        v' F^{-1} v = sum_j v_j (v_j - z_j (a_{t|t} - a_{t|t-1})) / h_j

    The last form uses the post-update residual, so there is no cancellation
    between two large numbers when some h_j are tiny (macro rows). Z' H^{-1} Z
    depends only on which series are observed, so it is recomputed only when
    that pattern changes from the previous month: in practice almost never.

    **Steady state.** The covariance recursion does not involve the data. When
    P_{t+1|t} agrees with P_{t|t-1} to `tol` (relative) under an unchanged
    observation pattern, P is frozen: the inverses, log-determinants and
    P_{t|t} of the last computed month are reused until the pattern changes.
    Only the data-dependent part (innovations, state update, quadratic form)
    is computed each month. frz[t] (if stored) is True when month t reused
    the previous month's covariances, which are then bitwise identical; the
    smoother uses this to reuse its gain. tol < 0 disables freezing.
    """
    N, n = y.shape
    k = T.shape[0]
    store = a_pred.shape[0] > 0
    keep_frz = frz.shape[0] > 0
    hinv = 1.0 / h
    logh = np.log(h)
    a = a1.copy()
    P = P1.copy()
    Pf = P1.copy()
    Minv = np.eye(k)
    logdet_P = 0.0
    logdet_M = 0.0
    ll = 0.0
    ZHZ = np.zeros((k, k))
    sum_logh = 0.0
    mask = np.zeros(n, np.bool_)
    prev = np.zeros(n, np.bool_)
    have = False
    frozen = False
    u = np.empty(k)
    v = np.empty(n)
    da = np.empty(k)
    tmp = np.empty(k)
    for t in range(N):
        if store:
            a_pred[t] = a
            P_pred[t] = P
        m = 0
        same = have
        for j in range(n):
            ob = np.isfinite(y[t, j])
            mask[j] = ob
            if ob:
                m += 1
            if ob != prev[j]:
                same = False
        if not same or m == 0:
            frozen = False
        skipped = frozen
        if m > 0:
            if not same:  # observation pattern changed: rebuild Z' H^-1 Z for it
                ZHZ[:, :] = 0.0
                sum_logh = 0.0
                for j in range(n):
                    if mask[j]:
                        sum_logh += logh[j]
                        for r in range(k):
                            zr = Z[j, r] * hinv[j]
                            if zr != 0.0:
                                for s_ in range(k):
                                    ZHZ[r, s_] += zr * Z[j, s_]
                prev[:] = mask
                have = True
            u[:] = 0.0
            for j in range(n):
                if mask[j]:
                    vj = y[t, j] - d[j]
                    for r in range(k):
                        vj -= Z[j, r] * a[r]
                    v[j] = vj
                    w = hinv[j] * vj
                    for r in range(k):
                        u[r] += Z[j, r] * w
            if not frozen:
                Pinv, logdet_P = _chol_inv_logdet(P)
                if np.isnan(logdet_P):
                    return -np.inf
                Minv, logdet_M = _chol_inv_logdet(Pinv + ZHZ)
                if np.isnan(logdet_M):
                    return -np.inf
                Pf = 0.5 * (Minv + Minv.T)
            for r in range(k):
                acc = 0.0
                for s_ in range(k):
                    acc += Minv[r, s_] * u[s_]
                da[r] = acc
            quad = 0.0
            for j in range(n):
                if mask[j]:
                    e = v[j]
                    for r in range(k):
                        e -= Z[j, r] * da[r]
                    quad += hinv[j] * v[j] * e
            for r in range(k):
                a[r] += da[r]
            ll += -0.5 * (m * LOG_2PI + sum_logh + logdet_P + logdet_M + quad)
        else:
            Pf = P.copy()
        if store:
            a_filt[t] = a
            P_filt[t] = Pf
        if keep_frz:
            frz[t] = skipped
        for r in range(k):  # a <- c + T a, in place
            acc = c[r]
            for s_ in range(k):
                acc += T[r, s_] * a[s_]
            tmp[r] = acc
        a[:] = tmp
        if not frozen:
            Pn = T @ Pf @ T.T + Q
            if m > 0 and same and tol >= 0.0 and _max_abs_diff(Pn, P) <= tol * (1.0 + _max_abs(P)):
                frozen = True  # converged: keep P (bitwise) from here while the pattern holds
            else:
                P = Pn
    return ll


@_jit
def _forward(y, Z, H, d, T, Q, c, a1, P1, a_pred, P_pred, a_filt, P_filt, tol, frz):
    """Filter recursion: information form when H is diagonal, covariance form otherwise.

    Fills a_pred/P_pred/a_filt/P_filt in place (pass arrays with a zero
    leading dimension to skip storing) and returns the log-likelihood, or -inf
    when the parameters make a covariance matrix not positive definite.
    """
    n = H.shape[0]
    diag = True
    for i in range(n):
        for j in range(n):
            if i != j and H[i, j] != 0.0:
                diag = False
    if diag:
        hd = np.empty(n)
        for i in range(n):
            hd[i] = H[i, i]
        return _forward_info(y, Z, hd, d, T, Q, c, a1, P1, a_pred, P_pred, a_filt, P_filt, tol, frz)
    return _forward_dense(y, Z, H, d, T, Q, c, a1, P1, a_pred, P_pred, a_filt, P_filt, frz)


@_jit
def _loglik_loop(y, Z, H, d, T, Q, c, a1, P1, tol):
    k = T.shape[0]
    e1, e2 = np.empty((0, k)), np.empty((0, k, k))
    return _forward(y, Z, H, d, T, Q, c, a1, P1, e1, e2, e1, e2, tol, np.empty(0, np.bool_))


@_jit
def _stats_loop(y, Z, H, d, T, Q, c, a1, P1, tol):
    """Filter, RTS smoother, and the sufficient statistics of the expected complete-data log-likelihood.

    Steady-state shortcuts (exact up to `tol`, see `_forward_info`):

    - the smoother gain J_t = P_{t|t} T' P_{t+1|t}^{-1} depends only on the
      covariances, so it is reused (no linear solve) across months the filter
      marked as frozen;
    - with J fixed, the smoothed covariance recursion is a contraction; once
      P_{t|T} stops changing it is copied instead of recomputed;
    - each series' sum of E[x x'] over the months it is observed is the sum
      over all months minus the months it is missing (rare), instead of one
      k x k accumulation per series per month: O(N k^2) instead of O(N n k^2).
    """
    N, n = y.shape
    k = T.shape[0]
    a_pred = np.empty((N, k))
    P_pred = np.empty((N, k, k))
    a_s = np.empty((N, k))
    P_s = np.empty((N, k, k))
    frz = np.zeros(N, np.bool_)
    ll = _forward(y, Z, H, d, T, Q, c, a1, P1, a_pred, P_pred, a_s, P_s, tol, frz)

    n_i = np.zeros(n)
    syy = np.zeros(n)
    sya = np.zeros((n, k))
    sxx = np.zeros((n, k, k))
    S11 = np.zeros((k, k))
    S10 = np.zeros((k, k + 1))
    S00 = np.zeros((k + 1, k + 1))
    if not np.isfinite(ll):
        return ll, n_i, syy, sya, sxx, S11, S10, S00, a_s, P_s

    # Backward pass (a_s, P_s hold the filtered values on entry). C[t] = Cov(x_{t+1}, x_t | all data).
    # Small fixed-size algebra is written as loops into preallocated buffers: on
    # k <= 6 matrices, allocating a temporary costs more than the arithmetic.
    C = np.zeros((N, k, k))
    J = np.zeros((k, k))
    D = np.empty((k, k))
    DJ = np.empty((k, k))
    da = np.empty(k)
    ps_frozen = False
    for t in range(N - 2, -1, -1):
        # J_t equals J_{t+1} when P_{t|t} = P_{t+1|t+1} and P_{t+1|t} = P_{t+2|t+1}, i.e. months t+1 and t+2 frozen.
        reuse = tol >= 0.0 and t <= N - 3 and frz[t + 1] and frz[t + 2]
        if not reuse:
            J = np.linalg.solve(P_pred[t + 1], T @ P_s[t]).T  # P_{t|t} T' P_{t+1|t}^{-1}
            J = np.ascontiguousarray(J)
            ps_frozen = False
        for r in range(k):  # a_s[t] += J (a_s[t+1] - a_pred[t+1])
            da[r] = a_s[t + 1, r] - a_pred[t + 1, r]
        for r in range(k):
            acc = 0.0
            for q in range(k):
                acc += J[r, q] * da[q]
            a_s[t, r] += acc
        if ps_frozen:
            P_s[t] = P_s[t + 1]
        else:
            for r in range(k):  # P_s[t] += J (P_s[t+1] - P_pred[t+1]) J'
                for q in range(k):
                    D[r, q] = P_s[t + 1, r, q] - P_pred[t + 1, r, q]
            for r in range(k):
                for q in range(k):
                    acc = 0.0
                    for w in range(k):
                        acc += D[r, w] * J[q, w]
                    DJ[r, q] = acc
            for r in range(k):
                for q in range(r + 1):
                    acc = 0.0
                    for w in range(k):
                        acc += J[r, w] * DJ[w, q]
                    # symmetrise: average the (r, q) and (q, r) entries of P_filt + J D J'
                    val = 0.5 * (P_s[t, r, q] + P_s[t, q, r]) + acc
                    P_s[t, r, q] = val
                    P_s[t, q, r] = val
            if reuse and _max_abs_diff(P_s[t], P_s[t + 1]) <= tol * (1.0 + _max_abs(P_s[t + 1])):
                ps_frozen = True
        for r in range(k):  # C[t] = P_s[t+1] J'
            for q in range(k):
                acc = 0.0
                for w in range(k):
                    acc += P_s[t + 1, r, w] * J[q, w]
                C[t, r, q] = acc

    Sxx = np.zeros((k, k))  # sum over all months of E[x_t x_t']
    miss = np.zeros((n, k, k))  # per series: the same sum over months it is missing
    for t in range(N):
        for r in range(k):
            ar = a_s[t, r]
            for q in range(k):
                Sxx[r, q] += ar * a_s[t, q] + P_s[t, r, q]
        for j in range(n):
            if np.isfinite(y[t, j]):
                yj = y[t, j] - d[j]
                n_i[j] += 1.0
                syy[j] += yj * yj
                for r in range(k):
                    sya[j, r] += yj * a_s[t, r]
            else:
                for r in range(k):
                    for q in range(k):
                        miss[j, r, q] += a_s[t, r] * a_s[t, q] + P_s[t, r, q]
        if t >= 1:  # transition pairs (x_t, x_{t-1}); regressor z = (1, x_{t-1})
            for r in range(k):
                S10[r, 0] += a_s[t, r]
                for q in range(k):
                    S10[r, 1 + q] += a_s[t, r] * a_s[t - 1, q] + C[t - 1, r, q]
        if t <= N - 2:
            for r in range(k):
                S00[0, 1 + r] += a_s[t, r]
    for j in range(n):
        sxx[j] = Sxx - miss[j]
    Exx0 = np.outer(a_s[0], a_s[0]) + P_s[0]
    ExxN = np.outer(a_s[N - 1], a_s[N - 1]) + P_s[N - 1]
    S11[:, :] = Sxx - Exx0
    S00[0, 0] = N - 1.0
    S00[1:, 0] = S00[0, 1:]
    S00[1:, 1:] = Sxx - ExxN
    return ll, n_i, syy, sya, sxx, S11, S10, S00, a_s, P_s


def kalman_loglik(model: StateSpace, y, tol: float = STEADY_TOL) -> float:
    """Log-likelihood of y under `model`; identical to kalman_filter(...).loglik but much faster.

    Returns -inf (instead of raising) when the parameters make F_t singular.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if not HAVE_NUMBA:
        try:
            return kalman_filter(model, y).loglik
        except np.linalg.LinAlgError:
            return -np.inf
    f = lambda x: np.ascontiguousarray(x, dtype=float)
    return float(_loglik_loop(f(y), f(model.Z), f(model.H), f(model.d), f(model.T), f(model.Q),
                              f(model.c), f(model.a1), f(model.P1), tol))


@dataclass
class SmootherStats:
    """Everything the expected complete-data log-likelihood needs, from one filter + smoother pass.

    With E[.] = expectation given all data (the smoother), x_t the state and
    z_t = (1, x_t')':

    - per observed series j (sums over months where y_tj is observed):
      n_i[j] = count, syy[j] = sum (y_tj - d_j)^2,
      sya[j] = sum (y_tj - d_j) E[x_t], sxx[j] = sum E[x_t x_t'];
    - transition, over t = 2..N: S11 = sum E[x_t x_t'],
      S10 = sum E[x_t z_{t-1}'], S00 = sum E[z_{t-1} z_{t-1}'];
    - first state: a_s[0], P_s[0].

    These sums are what make both EM (closed-form updates) and the exact
    score (Fisher's identity) cheap: after one pass, neither needs the data
    again.
    """

    loglik: float
    n_i: np.ndarray
    syy: np.ndarray
    sya: np.ndarray
    sxx: np.ndarray
    S11: np.ndarray
    S10: np.ndarray
    S00: np.ndarray
    a_s: np.ndarray  # (N, k) smoothed states
    P_s: np.ndarray  # (N, k, k)

    @property
    def n_trans(self) -> int:
        return int(round(self.S00[0, 0]))


def smoothed_stats(model: StateSpace, y, tol: float = STEADY_TOL) -> SmootherStats:
    """One compiled filter + RTS smoother pass returning the log-likelihood and sufficient statistics.

    Assumes H is diagonal for the measurement statistics (as in DNS), so each
    series' noise can be handled on its own. If the parameters are invalid
    (F_t not positive definite) the log-likelihood is -inf and the
    statistics are zeros.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    f = lambda x: np.ascontiguousarray(x, dtype=float)
    out = _stats_loop(f(y), f(model.Z), f(model.H), f(model.d), f(model.T), f(model.Q),
                      f(model.c), f(model.a1), f(model.P1), tol)
    return SmootherStats(float(out[0]), *out[1:])


def filtered_states(model: StateSpace, y, tol: float = STEADY_TOL) -> np.ndarray:
    """a_{t|t} for every t, from the compiled filter (same numbers as `kalman_filter(...).a_filt`).

    Used by the backtests, which need many filter passes but only the filtered
    states.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    N, k = y.shape[0], model.T.shape[0]
    f = lambda x: np.ascontiguousarray(x, dtype=float)
    a_pred, P_pred = np.empty((N, k)), np.empty((N, k, k))
    a_filt, P_filt = np.empty((N, k)), np.empty((N, k, k))
    ll = _forward(f(y), f(model.Z), f(model.H), f(model.d), f(model.T), f(model.Q), f(model.c),
                  f(model.a1), f(model.P1), a_pred, P_pred, a_filt, P_filt, tol, np.empty(0, np.bool_))
    if not np.isfinite(ll):
        raise np.linalg.LinAlgError("filter failed: covariance not positive definite")
    return a_filt


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
