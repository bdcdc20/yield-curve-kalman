"""Impulse responses and variance decompositions for the state VAR(1) (README section 6.4).

The DNS state follows x_t - mu = A (x_{t-1} - mu) + eta_t, eta_t ~ N(0, Q). Its
moving-average form is x_{t+h} - E x = sum_i A^i eta_{t+h-i}, so a shock eta_t
moves the state h months later by A^h eta_t. The questions DRA (2006) ask of
the yields + macro model are answered from A and Q alone:

- **Impulse responses:** how do the factors (and hence every yield) respond to
  a one-standard-deviation macro shock, and the macro variables to a factor
  shock? Shocks in Q are correlated, so a "shock to inflation" needs an
  identifying assumption:

  * *Cholesky (recursive).* Orthogonalise with Q = L L' for a chosen ordering;
    a variable can react within the month to shocks of variables ordered
    before it, but not after. The default puts the slow-moving macro
    variables first (activity, inflation, then the policy rate) and the yield
    factors last: markets price news the same day, while activity and prices
    respond to rates with a lag. Results can depend on this choice.
  * *Generalised* (Pesaran & Shin 1998). The response to shock j is
    A^h Q e_j / sqrt(Q_jj): the expected path given that variable j's shock
    is one sd, with the others moving as their correlation implies. It does
    not depend on any ordering (it equals the Cholesky response with j
    ordered first) but shocks are not orthogonal, so it has no variance
    decomposition. Used here as the robustness check on the ordering.

- **Variance decomposition:** the share of each variable's h-step forecast
  error variance due to each orthogonalised shock.

- **Bands:** parameter uncertainty from the observed information matrix (the
  numerical Jacobian of the exact analytic score), draws of the parameters
  from the asymptotic normal distribution of the MLE, and pointwise
  quantiles of the resulting responses. This ignores the 1 bp noise floor's
  boundary (those parameters are held fixed) and is meant for the unshrunk
  MLE; a shrinkage fit would need the prior's curvature too.

Array conventions: responses have shape (horizon + 1, k, k), indexed
[h, responding variable, shock], in the model's state order (level, slope,
curvature, then macro columns). Shocks are one standard deviation and the
units are those of the states (percentage points).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from yck.model.dns import maturities_from_columns
from yck.model.kalman import smoothed_stats
from yck.model.statespace_dns import DNSFit, DNSParams, score

DEFAULT_ORDER = ("real_activity", "inflation", "policy_rate", "level", "slope", "curvature")


def ma_coefficients(A: np.ndarray, horizon: int) -> np.ndarray:
    """Psi_h = A^h for h = 0..horizon, shape (horizon + 1, k, k)."""
    k = A.shape[0]
    psi = np.empty((horizon + 1, k, k))
    psi[0] = np.eye(k)
    for h in range(1, horizon + 1):
        psi[h] = A @ psi[h - 1]
    return psi


def resolve_order(names, order=None) -> list[int]:
    """State indices in identification order.

    `order` may hold state names or indices; names not in the model are
    skipped, so `DEFAULT_ORDER` works for any subset of macro variables. States
    left out are appended in model order.
    """
    names = list(names)
    seq = DEFAULT_ORDER if order is None else order
    idx = [names.index(o) if isinstance(o, str) else int(o) for o in seq if not isinstance(o, str) or o in names]
    if len(set(idx)) != len(idx):
        raise ValueError(f"repeated state in ordering {order}")
    return idx + [i for i in range(len(names)) if i not in idx]


def cholesky_impact(Q: np.ndarray, order) -> np.ndarray:
    """Impact matrix B (state x shock, both in model order) with B B' = Q, recursive in `order`.

    Column j is the impact of a one-sd orthogonal shock to state j. A state
    responds on impact only to its own shock and to shocks of states ordered
    before it.
    """
    order = np.asarray(order)
    L = np.linalg.cholesky(Q[np.ix_(order, order)])
    B = np.empty_like(Q)
    B[np.ix_(order, order)] = L
    return B


def generalized_impact(Q: np.ndarray) -> np.ndarray:
    """Pesaran-Shin impact matrix: column j is Q e_j / sqrt(Q_jj)."""
    return Q / np.sqrt(np.diag(Q))[None, :]


def impulse_responses(A: np.ndarray, Q: np.ndarray, horizon: int, order=None,
                      kind: str = "cholesky") -> np.ndarray:
    """Responses [h, variable, shock] of the states to one-sd shocks; `order` is a list of state indices."""
    if kind == "cholesky":
        B = cholesky_impact(Q, list(range(A.shape[0])) if order is None else order)
    elif kind == "generalized":
        B = generalized_impact(Q)
    else:
        raise ValueError(f"kind must be 'cholesky' or 'generalized', not {kind!r}")
    return ma_coefficients(A, horizon) @ B


def variance_decomposition(A: np.ndarray, Q: np.ndarray, horizon: int, order=None) -> np.ndarray:
    """Forecast error variance shares [h - 1, variable, shock] for h = 1..horizon (rows sum to 1).

    The h-step forecast error is sum_{i<h} A^i B u_{t+h-i} with orthogonal unit
    shocks u, so its variance due to shock j is sum_{i<h} (A^i B)[:, j]^2.
    """
    theta = impulse_responses(A, Q, horizon - 1, order, "cholesky")
    contrib = np.cumsum(theta**2, axis=0)
    return contrib / contrib.sum(axis=2, keepdims=True)


def yield_responses(responses: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """Responses of yields [h, maturity, shock]: the Nelson-Siegel loadings applied to the factor responses."""
    return np.einsum("mf,hfs->hms", loadings, responses[:, :3, :])


# --------------------------------------------------------------------------- parameter uncertainty
@dataclass
class Information:
    theta: np.ndarray  # MLE in the optimiser's parameterisation
    free: np.ndarray  # bool: parameters not at the noise-floor bound
    hessian: np.ndarray  # d^2 loglik / d theta^2 over the free parameters
    cov: np.ndarray  # asymptotic covariance of the free parameters, (-hessian)^-1
    n_clipped: int  # eigenvalues of -hessian raised to keep cov positive definite


def observed_information(fit: DNSFit, data: pd.DataFrame, h_floor_bp: float | None = 1.0,
                         eps: float = 1e-4) -> Information:
    """Observed information at the MLE: central differences of the exact score.

    Each score costs one filter + smoother pass, so this takes 2 x (number of
    parameters) passes, a fraction of a second on USD. Measurement variances
    sitting on the noise floor are boundary estimates with no normal
    asymptotics; they are held fixed. If -hessian is not positive definite
    (flat ridges), its small or negative eigenvalues are raised to 1e-8 of the
    largest and the count is reported.
    """
    p, cols = fit.params, fit.columns
    tau = maturities_from_columns(fit.yield_columns)
    n, k = len(tau), p.k
    Y = data[cols].to_numpy(float)
    theta = p.to_vector(fit.estimate_lambda)
    free = np.ones(theta.size, bool)
    if fit.mu_fixed:  # mu set to the sample mean, not estimated
        free[:k] = False
    if h_floor_bp:
        o = k + k * k + k * (k + 1) // 2
        free[o:o + n] = p.h > (h_floor_bp / 100.0) ** 2 * (1 + 1e-6)

    def g(th):
        q = DNSParams.from_vector(th, n, fit.estimate_lambda, p.lam, k)
        return score(th, smoothed_stats(q.statespace(tau), Y), tau, fit.estimate_lambda, p.lam)

    idx = np.flatnonzero(free)
    H = np.empty((idx.size, idx.size))
    for c, i in enumerate(idx):
        e = np.zeros_like(theta)
        e[i] = eps
        H[:, c] = (g(theta + e) - g(theta - e))[idx] / (2 * eps)
    H = (H + H.T) / 2
    w, V = np.linalg.eigh(-H)
    floor = 1e-8 * w.max()
    n_clipped = int(np.sum(w < floor))
    cov = (V / np.maximum(w, floor)) @ V.T
    return Information(theta=theta, free=free, hessian=H, cov=cov, n_clipped=n_clipped)


@dataclass
class IRFDraws:
    point: np.ndarray  # (horizon + 1, k, k) at the MLE
    draws: np.ndarray  # (n_kept, horizon + 1, k, k)
    n_rejected: int  # draws with an explosive A (max |eig| >= 1), discarded
    info: Information

    def band(self, level: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        a = (1 - level) / 2
        return np.quantile(self.draws, a, axis=0), np.quantile(self.draws, 1 - a, axis=0)


def irf_draws(fit: DNSFit, data: pd.DataFrame, horizon: int = 60, order=None, kind: str = "cholesky",
              n_draws: int = 1000, seed: int = 0, h_floor_bp: float | None = 1.0,
              info: Information | None = None) -> IRFDraws:
    """Impulse responses at the MLE plus parameter draws from N(theta_hat, cov) for confidence bands.

    `order` takes state names or indices (default `DEFAULT_ORDER`). Draws
    whose A is explosive are discarded, which conditions on stationarity.
    """
    p = fit.params
    k = p.k
    idx = resolve_order(fit.state_names, order)
    info = info or observed_information(fit, data, h_floor_bp)
    point = impulse_responses(p.A, p.Q, horizon, idx, kind)
    rng = np.random.default_rng(seed)
    C = np.linalg.cholesky(info.cov)
    # All draws at once (stacks of k x k matrices) instead of a Python loop per draw.
    # Row i of the (n_draws, m) normal matrix is exactly what draw i got in the loop version.
    th = np.tile(info.theta, (n_draws, 1))
    th[:, info.free] += rng.standard_normal((n_draws, C.shape[0])) @ C.T
    A, Q = _draw_A_Q(th, k)
    stable = np.max(np.abs(np.linalg.eigvals(A)), axis=1) < 1.0
    return IRFDraws(point=point, draws=_batched_responses(A[stable], Q[stable], horizon, idx, kind),
                    n_rejected=int((~stable).sum()), info=info)


def _draw_A_Q(th: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """A and Q for a stack of parameter vectors (same transforms as `DNSParams.from_vector`)."""
    A = th[:, k:k + k * k].reshape(-1, k, k)
    rows, cols = np.tril_indices(k)
    o = k + k * k
    l = th[:, o:o + rows.size].copy()
    diag = rows == cols
    l[:, diag] = np.exp(l[:, diag])
    L = np.zeros((th.shape[0], k, k))
    L[:, rows, cols] = l
    return A, L @ L.transpose(0, 2, 1)


def _batched_responses(A: np.ndarray, Q: np.ndarray, horizon: int, order, kind: str) -> np.ndarray:
    """`impulse_responses` for a stack of (A, Q): shape (draws, horizon + 1, k, k)."""
    nd, k = A.shape[0], A.shape[1]
    if kind == "cholesky":
        o = np.asarray(order)
        B = np.zeros_like(Q)
        B[np.ix_(np.arange(nd), o, o)] = np.linalg.cholesky(Q[:, o][:, :, o])
    elif kind == "generalized":
        B = Q / np.sqrt(np.diagonal(Q, axis1=1, axis2=2))[:, None, :]
    else:
        raise ValueError(f"kind must be 'cholesky' or 'generalized', not {kind!r}")
    out = np.empty((nd, horizon + 1, k, k))
    out[:, 0] = B
    for h in range(1, horizon + 1):
        out[:, h] = A @ out[:, h - 1]  # A^h B
    return out


def irf_frame(responses: np.ndarray, names, shock: str) -> pd.DataFrame:
    """One shock's responses as a horizon x variable frame."""
    names = list(names)
    return pd.DataFrame(responses[:, :, names.index(shock)], columns=names).rename_axis("horizon")
