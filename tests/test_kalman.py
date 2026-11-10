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
