"""Offline tests for the Kalman filter and smoother, using statsmodels as the reference."""
import numpy as np
import pytest

from yck.model.dns import ns_loadings
from yck.model.kalman import StateSpace, forecast_state, kalman_filter, kalman_loglik, rts_smoother, stationary_init

sm_ks = pytest.importorskip("statsmodels.tsa.statespace.kalman_smoother")


def _dns_like_system(seed=0, N=150):
    """3 persistent factors observed through NS loadings at 6 maturities, with noise."""
    rng = np.random.default_rng(seed)
    Z = ns_loadings([3, 12, 24, 60, 84, 120])
    A = np.array([[0.97, 0.02, 0.0], [0.0, 0.92, 0.05], [0.03, 0.0, 0.80]])
    mu = np.array([5.0, -1.5, 0.5])
    L = np.array([[0.30, 0, 0], [0.10, 0.40, 0], [-0.05, 0.10, 0.60]])
    Q = L @ L.T
    H = np.diag(rng.uniform(0.01, 0.05, size=6) ** 2)
    c = (np.eye(3) - A) @ mu
    a1, P1 = stationary_init(A, Q, c)
    model = StateSpace(Z=Z, H=H, T=A, Q=Q, a1=a1, P1=P1, c=c)

    x = np.empty((N, 3))
    x[0] = rng.multivariate_normal(a1, P1)
    for t in range(1, N):
        x[t] = c + A @ x[t - 1] + L @ rng.normal(size=3)
    y = x @ Z.T + rng.normal(size=(N, 6)) * np.sqrt(np.diag(H))
    return model, y


def _with_gaps(y, seed=1):
    y = y.copy()
    rng = np.random.default_rng(seed)
    y[rng.random(y.shape) < 0.15] = np.nan  # scattered missing points
    y[20:45, :2] = np.nan                   # a GBP-style short-end gap
    y[60] = np.nan                          # a fully missing month
    y[61:63] = np.nan
    y[90, 1:] = np.nan                      # a single observed series
    return y


def _statsmodels(model: StateSpace, y):
    N, n = y.shape
    k = model.T.shape[0]
    ks = sm_ks.KalmanSmoother(k_endog=n, k_states=k)
    ks.bind(y.copy())  # NaN = missing, handled natively
    ks["design"], ks["obs_intercept"], ks["obs_cov"] = model.Z, model.d, model.H
    ks["transition"], ks["state_intercept"], ks["state_cov"] = model.T, model.c, model.Q
    ks["selection"] = np.eye(k)
    ks.initialize_known(model.a1, model.P1)
    return ks.smooth()


@pytest.mark.parametrize("gaps", [False, True], ids=["complete", "missing"])
def test_matches_statsmodels(gaps):
    model, y = _dns_like_system()
    if gaps:
        y = _with_gaps(y)
    ours = kalman_filter(model, y)
    ref = _statsmodels(model, y)

    np.testing.assert_allclose(ours.loglik_t, ref.llf_obs, rtol=1e-9, atol=1e-9)
    assert ours.loglik == pytest.approx(ref.llf_obs.sum(), rel=1e-10)
    np.testing.assert_allclose(ours.a_filt, ref.filtered_state.T, atol=1e-9)
    np.testing.assert_allclose(ours.P_filt, ref.filtered_state_cov.transpose(2, 0, 1), atol=1e-10)
    np.testing.assert_allclose(ours.a_pred, ref.predicted_state.T, atol=1e-9)
    np.testing.assert_allclose(ours.P_pred, ref.predicted_state_cov.transpose(2, 0, 1), atol=1e-10)

    a_s, P_s = rts_smoother(model, ours)
    np.testing.assert_allclose(a_s, ref.smoothed_state.T, atol=1e-8)
    np.testing.assert_allclose(P_s, ref.smoothed_state_cov.transpose(2, 0, 1), atol=1e-9)


def test_fully_missing_period_skips_update():
    model, y = _dns_like_system(N=30)
    y[10] = np.nan
    res = kalman_filter(model, y)
    assert res.n_obs[10] == 0 and res.loglik_t[10] == 0.0
    np.testing.assert_array_equal(res.a_filt[10], res.a_pred[10])
    np.testing.assert_array_equal(res.P_filt[10], res.P_pred[10])


def test_scalar_one_step_by_hand():
    # Local level: y = x + e, H=1; x' = 0.5 x + eta, Q=1; prior N(2, 4); observe y=4.
    m = StateSpace(Z=np.eye(1), H=np.eye(1), T=np.array([[0.5]]), Q=np.eye(1),
                   a1=np.array([2.0]), P1=np.array([[4.0]]))
    res = kalman_filter(m, np.array([[4.0]]))
    F, v = 5.0, 2.0
    assert res.a_filt[0, 0] == pytest.approx(2.0 + 4.0 / F * v)       # 3.6
    assert res.P_filt[0, 0, 0] == pytest.approx(4.0 - 16.0 / F)        # 0.8
    assert res.a_pred[1, 0] == pytest.approx(0.5 * 3.6)
    assert res.P_pred[1, 0, 0] == pytest.approx(0.25 * 0.8 + 1.0)
    assert res.loglik == pytest.approx(-0.5 * (np.log(2 * np.pi) + np.log(F) + v**2 / F))


def test_stationary_init_solves_lyapunov_and_rejects_unit_root():
    T = np.array([[0.9, 0.1], [0.0, 0.5]])
    Q = np.array([[1.0, 0.2], [0.2, 0.5]])
    a1, P1 = stationary_init(T, Q, c=np.array([1.0, 1.0]))
    np.testing.assert_allclose(P1, T @ P1 @ T.T + Q)
    np.testing.assert_allclose(a1, T @ a1 + [1.0, 1.0])
    with pytest.raises(ValueError):
        stationary_init(np.eye(2), Q)


def test_forecast_state_reverts_to_mean():
    model, _ = _dns_like_system()
    a = np.array([8.0, 1.0, -2.0])
    assert forecast_state(model, a, 0) == pytest.approx(a)
    assert forecast_state(model, a, 1) == pytest.approx(model.c + model.T @ a)
    assert forecast_state(model, a, 2000) == pytest.approx(model.a1, abs=1e-8)


@pytest.mark.parametrize("gaps", [False, True], ids=["complete", "missing"])
def test_fast_loglik_equals_reference_filter(gaps):
    model, y = _dns_like_system()
    if gaps:
        y = _with_gaps(y)
    assert kalman_loglik(model, y) == pytest.approx(kalman_filter(model, y).loglik, rel=1e-11)


def test_fast_loglik_rejects_invalid_parameters():
    model, y = _dns_like_system(N=20)
    model.H = -np.eye(6)  # makes F_t indefinite
    assert kalman_loglik(model, y) == -np.inf


# --------------------------------------------------------------------------- smoother statistics
def _brute_force_posterior(model: StateSpace, y):
    """E and Cov of all stacked states given the observed y, by direct Gaussian conditioning."""
    N, n = y.shape
    k = model.T.shape[0]
    # Prior mean and covariance of the stacked states x_1..x_N.
    m = np.zeros(N * k)
    V = np.zeros((N * k, N * k))
    m[:k], V[:k, :k] = model.a1, model.P1
    for t in range(1, N):
        s, p = slice(t * k, (t + 1) * k), slice((t - 1) * k, t * k)
        m[s] = model.c + model.T @ m[p]
        V[s, : t * k] = model.T @ V[p, : t * k]       # Cov(x_t, x_<t)
        V[: t * k, s] = V[s, : t * k].T
        V[s, s] = model.T @ V[p, p] @ model.T.T + model.Q
    Zbig = np.kron(np.eye(N), model.Z)
    obs = np.isfinite(y).ravel()
    Zo = Zbig[obs]
    my = np.tile(model.d, N)[obs] + Zo @ m
    Vyy = Zo @ V @ Zo.T + np.kron(np.eye(N), model.H)[np.ix_(obs, obs)]
    Vxy = V @ Zo.T
    G = np.linalg.solve(Vyy, Vxy.T).T
    mean = m + G @ (y.ravel()[obs] - my)
    cov = V - G @ Vxy.T
    return mean.reshape(N, k), cov


def test_smoothed_stats_match_brute_force_conditioning():
    from yck.model.kalman import smoothed_stats

    rng = np.random.default_rng(7)
    k, n, N = 2, 3, 6
    T = np.array([[0.8, 0.1], [-0.2, 0.6]])
    Q = np.array([[0.5, 0.1], [0.1, 0.3]])
    c = np.array([0.3, -0.1])
    a1, P1 = stationary_init(T, Q, c)
    model = StateSpace(Z=rng.normal(size=(n, k)), H=np.diag([0.2, 0.4, 0.3]), T=T, Q=Q,
                       a1=a1, P1=P1, c=c, d=np.array([0.1, 0.0, -0.2]))
    y = rng.normal(size=(N, n))
    y[1, 0] = y[3] = np.nan
    y[4, 1:] = np.nan
    st = smoothed_stats(model, y)
    mean, cov = _brute_force_posterior(model, y)
    E = lambda t: mean[t]
    Exx = lambda t, s: np.outer(mean[t], mean[s]) + cov[t * k:(t + 1) * k, s * k:(s + 1) * k]

    assert st.loglik == pytest.approx(kalman_filter(model, y).loglik, rel=1e-12)
    np.testing.assert_allclose(st.a_s, mean, atol=1e-10)
    np.testing.assert_allclose(st.S11, sum(Exx(t, t) for t in range(1, N)), atol=1e-10)
    np.testing.assert_allclose(st.S10[:, 0], sum(E(t) for t in range(1, N)), atol=1e-10)
    np.testing.assert_allclose(st.S10[:, 1:], sum(Exx(t, t - 1) for t in range(1, N)), atol=1e-10)  # lag-one terms
    np.testing.assert_allclose(st.S00[1:, 1:], sum(Exx(t, t) for t in range(N - 1)), atol=1e-10)
    for j in range(n):
        ts = [t for t in range(N) if np.isfinite(y[t, j])]
        assert st.n_i[j] == len(ts)
        assert st.syy[j] == pytest.approx(sum((y[t, j] - model.d[j]) ** 2 for t in ts))
        np.testing.assert_allclose(st.sya[j], sum((y[t, j] - model.d[j]) * E(t) for t in ts), atol=1e-10)
        np.testing.assert_allclose(st.sxx[j], sum(Exx(t, t) for t in ts), atol=1e-10)


def test_information_form_matches_covariance_form_with_non_diagonal_fallback():
    from yck.model.kalman import smoothed_stats

    model, y = _dns_like_system()
    y = _with_gaps(y)
    ref = kalman_filter(model, y)
    st = smoothed_stats(model, y)  # information form (H diagonal)
    assert st.loglik == pytest.approx(ref.loglik, rel=1e-11)
    # A non-diagonal H takes the covariance-form path and still matches the reference.
    model.H = model.H + 0.0001 * (np.ones((6, 6)) - np.eye(6))
    assert kalman_loglik(model, y) == pytest.approx(kalman_filter(model, y).loglik, rel=1e-11)


def test_compiled_filtered_states_match_reference():
    from yck.model.kalman import filtered_states

    model, y = _dns_like_system()
    y = _with_gaps(y)
    np.testing.assert_allclose(filtered_states(model, y), kalman_filter(model, y).a_filt, atol=1e-10)


@pytest.mark.parametrize("gaps", ["none", "scattered short end", "ragged edge"])
def test_steady_state_shortcuts_match_the_full_recursion(gaps):
    """Freezing the converged covariances (filter and smoother) must not change any output."""
    from yck.model.kalman import kalman_loglik, smoothed_stats, filtered_states
    from yck.model.statespace_dns import DNSParams

    tau = np.array([3, 6, 12, 24, 36, 60, 84, 120], float)
    rng = np.random.default_rng(11)
    k = 4  # 3 factors + one macro series, observed (almost) without error
    A = np.diag([0.97, 0.9, 0.8, 0.95]); A[0, 3] = 0.05
    p = DNSParams(mu=np.array([5.0, -1.5, 0.0, 3.0]), A=A, Q=np.diag([0.04, 0.09, 0.25, 0.1]),
                  h=np.full(len(tau), 0.03**2), lam=0.0609)
    ss = p.statespace(tau)
    N = 400
    x = np.empty((N, k)); x[0] = p.mu
    for t in range(1, N):
        x[t] = ss.c + A @ x[t - 1] + rng.normal(size=k) * np.sqrt(np.diag(p.Q))
    Y = x @ ss.Z.T + rng.normal(size=(N, len(tau) + 1)) * np.sqrt(np.diag(ss.H))
    if gaps == "scattered short end":
        Y[rng.random(N) < 0.4, 0] = np.nan
        Y[rng.random(N) < 0.15, 1] = np.nan
    elif gaps == "ragged edge":
        Y[-2:, -1] = np.nan
        Y[100:103, :] = np.nan  # months with nothing observed

    assert kalman_loglik(ss, Y) == pytest.approx(kalman_loglik(ss, Y, tol=-1.0), rel=1e-12, abs=1e-8)
    np.testing.assert_allclose(filtered_states(ss, Y), filtered_states(ss, Y, tol=-1.0), rtol=1e-10, atol=1e-10)
    fast, ref = smoothed_stats(ss, Y), smoothed_stats(ss, Y, tol=-1.0)
    assert fast.loglik == pytest.approx(ref.loglik, rel=1e-12, abs=1e-8)
    for name in ("n_i", "syy", "sya", "sxx", "S11", "S10", "S00", "a_s", "P_s"):
        np.testing.assert_allclose(getattr(fast, name), getattr(ref, name), rtol=1e-9, atol=1e-9, err_msg=name)


def test_compiled_lyapunov_matches_scipy():
    from scipy.linalg import solve_discrete_lyapunov
    from yck.model.kalman import lyapunov

    rng = np.random.default_rng(2)
    for k in (3, 4, 6):
        A = rng.normal(size=(k, k))
        A *= 0.97 / np.max(np.abs(np.linalg.eigvals(A)))
        M = rng.normal(size=(k, k))
        Q = M @ M.T
        np.testing.assert_allclose(lyapunov(A, Q), solve_discrete_lyapunov(A, Q), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(lyapunov(A.T, Q), solve_discrete_lyapunov(A.T, Q), rtol=1e-10, atol=1e-12)
