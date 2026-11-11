"""Offline tests for impulse responses, variance decompositions and parameter-uncertainty bands."""
import numpy as np
import pandas as pd
import pytest

from yck.model.irf import (cholesky_impact, impulse_responses, irf_draws, ma_coefficients, observed_information,
                           resolve_order, variance_decomposition, yield_responses)
from yck.model.statespace_dns import DNSParams, fit_dns_mle

RNG = np.random.default_rng(5)
K = 4
A = np.array([[0.9, 0.1, 0.0, 0.05], [0.0, 0.8, 0.1, 0.0], [0.05, 0.0, 0.7, 0.1], [0.0, 0.1, 0.0, 0.6]])
M = RNG.normal(size=(K, K))
Q = M @ M.T + 0.5 * np.eye(K)


def test_ma_coefficients_are_powers_of_A():
    psi = ma_coefficients(A, 5)
    np.testing.assert_array_equal(psi[0], np.eye(K))
    np.testing.assert_allclose(psi[5], np.linalg.matrix_power(A, 5), atol=1e-14)


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [3, 1, 0, 2], [2, 3, 1, 0]])
def test_cholesky_impact_factors_Q_and_is_recursive(order):
    B = cholesky_impact(Q, order)
    np.testing.assert_allclose(B @ B.T, Q, atol=1e-12)
    for pos, j in enumerate(order):  # shock j does not move states ordered before it on impact
        for i in order[:pos]:
            assert B[i, j] == 0.0


def test_matches_statsmodels_var_irf_and_fevd():
    from statsmodels.tsa.api import VAR

    N = 400
    x = np.zeros((N, K))
    L = np.linalg.cholesky(Q)
    for t in range(1, N):
        x[t] = A @ x[t - 1] + L @ RNG.normal(size=K)
    res = VAR(x).fit(1, trend="c")
    A_hat, Q_hat = res.coefs[0], res.sigma_u
    ours = impulse_responses(A_hat, Q_hat, 10)
    np.testing.assert_allclose(ours, res.orth_ma_rep(10), atol=1e-10)
    np.testing.assert_allclose(variance_decomposition(A_hat, Q_hat, 10), res.fevd(10).decomp.transpose(1, 0, 2),
                               atol=1e-10)


def test_generalized_response_equals_cholesky_with_that_shock_first():
    gen = impulse_responses(A, Q, 8, kind="generalized")
    for j in range(K):
        order = [j] + [i for i in range(K) if i != j]
        np.testing.assert_allclose(gen[:, :, j], impulse_responses(A, Q, 8, order)[:, :, j], atol=1e-12)


def test_variance_decomposition_rows_sum_to_one_and_total_is_order_free():
    for order in ([0, 1, 2, 3], [3, 2, 1, 0]):
        vd = variance_decomposition(A, Q, 12, order)
        np.testing.assert_allclose(vd.sum(axis=2), 1.0, atol=1e-12)
        assert np.all(vd >= 0)
    # the forecast error variance itself, sum_i A^i Q A^i', does not depend on the ordering
    tot = [np.cumsum(impulse_responses(A, Q, 11, o)**2, axis=0).sum(axis=2) for o in ([0, 1, 2, 3], [3, 2, 1, 0])]
    np.testing.assert_allclose(tot[0], tot[1], atol=1e-10)


def test_yield_responses_apply_loadings():
    from yck.model.dns import ns_loadings

    irf = impulse_responses(A, Q, 4)
    Z = ns_loadings([3, 24, 120], 0.0609)
    np.testing.assert_allclose(yield_responses(irf, Z)[2], Z @ irf[2, :3, :], atol=1e-14)


def test_resolve_order_skips_absent_names_and_appends_the_rest():
    names = ["level", "slope", "curvature", "policy_rate"]
    assert resolve_order(names) == [3, 0, 1, 2]
    assert resolve_order(names, ["slope"]) == [1, 0, 2, 3]
    with pytest.raises(ValueError):
        resolve_order(names, ["level", 0])


# --------------------------------------------------------------------------- bands on a simulated DRA model
@pytest.fixture(scope="module")
def dra():
    from test_statespace_dns import _simulate_dra  # pytest puts tests/ on sys.path

    data, p, _ = _simulate_dra(N=400, seed=3)
    return data, p, fit_dns_mle(data, mu="mle")  # mu estimated, so the information covers every parameter


def test_observed_information_matches_second_differences_of_loglik(dra):
    from yck.model.kalman import kalman_filter
    from yck.model.dns import maturities_from_columns

    data, _, fit = dra
    info = observed_information(fit, data, h_floor_bp=None)
    assert info.free.all() and info.n_clipped == 0
    tau = maturities_from_columns(fit.yield_columns)
    Y = data[fit.columns].to_numpy()

    def ll(th):
        return kalman_filter(DNSParams.from_vector(th, len(tau), k=fit.params.k).statespace(tau), Y).loglik

    k = fit.params.k
    for i in [0, k + 4, k + k * k]:  # a mean, an A entry, a Q-Cholesky entry
        e = np.zeros_like(info.theta)
        e[i] = 1e-3
        d2 = (ll(info.theta + e) - 2 * ll(info.theta) + ll(info.theta - e)) / 1e-6
        assert info.hessian[i, i] == pytest.approx(d2, rel=0.02)


def test_bands_cover_the_true_response_and_contain_the_point(dra):
    data, p, fit = dra
    names = fit.state_names
    order = ["real_activity", "inflation", "level", "slope", "curvature"]
    d = irf_draws(fit, data, horizon=24, order=order, n_draws=300, h_floor_bp=None)
    lo, hi = d.band(0.9)
    assert d.draws.shape[1:] == (25, 5, 5) and d.n_rejected < 30
    assert np.all(lo <= d.point + 1e-9) and np.all(d.point <= hi + 1e-9)
    true = impulse_responses(p.A, p.Q, 24, resolve_order(names, order))
    lvl, infl = names.index("level"), names.index("inflation")
    path = true[1:, lvl, infl]  # level response to inflation, the simulated macro -> level link
    covered = np.mean((lo[1:, lvl, infl] <= path) & (path <= hi[1:, lvl, infl]))
    assert covered > 0.8
    assert np.all(lo[6:13, lvl, infl] > 0)  # the effect is significant at 6-12 months


def test_fixed_mu_is_excluded_from_the_information(dra):
    data, _, _ = dra
    fit = fit_dns_mle(data)  # default: mu fixed at the sample mean
    info = observed_information(fit, data, h_floor_bp=None)
    k = fit.params.k
    assert fit.mu_fixed and not info.free[:k].any() and info.free[k:].all()


@pytest.mark.parametrize("kind", ["cholesky", "generalized"])
def test_batched_draws_match_the_per_draw_loop(dra, kind):
    from yck.model.statespace_dns import DNSParams

    data, _, fit = dra
    order = ["real_activity", "inflation", "level", "slope", "curvature"]
    d = irf_draws(fit, data, horizon=24, order=order, kind=kind, n_draws=200, seed=3, h_floor_bp=None)
    # reference: the original loop, draw by draw from the same random stream
    rng = np.random.default_rng(3)
    C = np.linalg.cholesky(d.info.cov)
    idx = resolve_order(fit.state_names, order)
    ref, rejected = [], 0
    for _ in range(200):
        th = d.info.theta.copy()
        th[d.info.free] += C @ rng.standard_normal(C.shape[0])
        q = DNSParams.from_vector(th, len(fit.yield_columns), k=fit.params.k)
        if np.max(np.abs(np.linalg.eigvals(q.A))) >= 1.0:
            rejected += 1
            continue
        ref.append(impulse_responses(q.A, q.Q, 24, idx, kind))
    assert d.n_rejected == rejected
    np.testing.assert_allclose(d.draws, np.array(ref), rtol=1e-10, atol=1e-12)
