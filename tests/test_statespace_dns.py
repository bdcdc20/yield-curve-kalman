"""Offline tests for the maximum-likelihood DNS model."""
import numpy as np
import pandas as pd
import pytest

from yck.model.dns import ns_loadings
from yck.model.kalman import kalman_filter
from yck.model.statespace_dns import DNSParams, fit_dns_mle, two_step_start

GRID = [3, 6, 12, 24, 36, 60, 84, 120]
COLS = [f"{m}m" for m in GRID]

TRUE = DNSParams(
    mu=np.array([5.0, -1.5, 0.0]),
    A=np.array([[0.97, 0.02, 0.0], [0.0, 0.92, 0.04], [0.03, 0.0, 0.85]]),
    Q=np.array([[0.09, -0.02, 0.0], [-0.02, 0.16, 0.03], [0.0, 0.03, 0.36]]),
    h=np.full(len(GRID), 0.05**2),
    lam=0.0609,
)


def _simulate(p: DNSParams, N=360, seed=0, lam=None):
    rng = np.random.default_rng(seed)
    ss = p.statespace(GRID)
    f = np.empty((N, 3))
    f[0] = rng.multivariate_normal(ss.a1, ss.P1)
    Lq = np.linalg.cholesky(p.Q)
    for t in range(1, N):
        f[t] = ss.c + p.A @ f[t - 1] + Lq @ rng.normal(size=3)
    y = f @ ns_loadings(GRID, lam or p.lam).T + rng.normal(size=(N, len(GRID))) * np.sqrt(p.h)
    idx = pd.date_range("1990-01-31", periods=N, freq="ME", name="date")
    return pd.DataFrame(y, index=idx, columns=COLS), f


def test_pack_unpack_round_trip():
    for est_lam in (False, True):
        theta = TRUE.to_vector(est_lam)
        assert theta.size == 3 + 9 + 6 + len(GRID) + est_lam
        back = DNSParams.from_vector(theta, len(GRID), est_lam, TRUE.lam)
        for a, b in [(back.mu, TRUE.mu), (back.A, TRUE.A), (back.Q, TRUE.Q), (back.h, TRUE.h)]:
            np.testing.assert_allclose(a, b, atol=1e-12)
        assert back.lam == pytest.approx(TRUE.lam)


def test_any_theta_gives_a_valid_model():
    rng = np.random.default_rng(3)
    for _ in range(20):
        p = DNSParams.from_vector(rng.normal(scale=2, size=3 + 9 + 6 + len(GRID)), len(GRID))
        assert np.all(np.linalg.eigvalsh(p.Q) > 0) and np.all(p.h > 0)
        ss = p.statespace(GRID)  # unstable A falls back to the diffuse prior instead of failing
        assert np.all(np.linalg.eigvalsh(ss.P1) > 0)


@pytest.fixture(scope="module")
def sim_fit():
    y, f = _simulate(TRUE)
    y.iloc[::7, :2] = np.nan  # some missing short-end points
    return y, f, fit_dns_mle(y)


def test_mle_improves_on_start_and_truth(sim_fit):
    y, _, fit = sim_fit
    Y = y.to_numpy()
    ll_start = kalman_filter(two_step_start(y).statespace(GRID), Y).loglik
    ll_true = kalman_filter(TRUE.statespace(GRID), Y).loglik
    assert fit.loglik >= ll_start - 1e-6
    assert fit.loglik >= ll_true - 1e-6  # the MLE must fit the sample at least as well as the truth
    assert fit.loglik == pytest.approx(kalman_filter(fit.statespace(), Y).loglik, rel=1e-9)


def test_mle_recovers_parameters(sim_fit):
    _, _, fit = sim_fit
    p = fit.params
    np.testing.assert_allclose(np.diag(p.A), np.diag(TRUE.A), atol=0.05)
    np.testing.assert_allclose(np.sqrt(p.h), np.sqrt(TRUE.h), rtol=0.2)
    np.testing.assert_allclose(np.sqrt(np.diag(p.Q)), np.sqrt(np.diag(TRUE.Q)), rtol=0.2)


def test_filtered_factors_track_truth(sim_fit):
    y, f, fit = sim_fit
    est = fit.filtered_factors(y).to_numpy()
    ideal = kalman_filter(TRUE.statespace(GRID), y.to_numpy()).a_filt  # best possible: true parameters
    for i in range(3):
        corr = np.corrcoef(est[:, i], f[:, i])[0, 1]
        ceiling = np.corrcoef(ideal[:, i], f[:, i])[0, 1]
        assert corr > 0.95 and corr > ceiling - 0.01  # curvature is the hardest to pin down


def test_estimates_lambda():
    true = DNSParams(TRUE.mu, TRUE.A, TRUE.Q, TRUE.h, lam=0.04)
    y, _ = _simulate(true, N=300, seed=5)
    fit = fit_dns_mle(y, estimate_lambda=True)  # starts from DL's 0.0609
    # Sampling noise in lambda is several percent at this sample size; the point is
    # that it moves from 0.0609 to near the truth, and fits better than the truth.
    assert fit.params.lam == pytest.approx(0.04, rel=0.10)
    assert fit.loglik >= kalman_filter(true.statespace(GRID), y.to_numpy()).loglik


# --------------------------------------------------------------------------- EM and exact score
@pytest.mark.parametrize("est_lam", [False, True], ids=["lam-fixed", "lam-estimated"])
@pytest.mark.parametrize("gaps", [False, True], ids=["complete", "missing"])
def test_score_matches_numerical_gradient_of_loglik(est_lam, gaps):
    from yck.model.kalman import kalman_loglik, smoothed_stats
    from yck.model.statespace_dns import score

    y, _ = _simulate(TRUE, N=150, seed=11)
    if gaps:
        y.iloc[::4, :3] = np.nan
        y.iloc[30] = np.nan
    Y = y.to_numpy()
    tau = np.array(GRID, float)
    # A deliberately non-optimal point, so the gradient is far from zero.
    p = DNSParams(TRUE.mu + [0.3, -0.2, 0.1], TRUE.A * 0.98, TRUE.Q * 1.3, TRUE.h * 0.7, lam=0.07)
    theta = p.to_vector(est_lam)
    st = smoothed_stats(p.statespace(tau), Y)
    g = score(theta, st, tau, est_lam, p.lam)

    def ll(th):
        return kalman_loglik(DNSParams.from_vector(th, len(GRID), est_lam, p.lam).statespace(tau), Y)

    num = np.array([(ll(theta + e) - ll(theta - e)) / 2e-6 for e in np.eye(theta.size) * 1e-6])
    np.testing.assert_allclose(g, num, rtol=1e-4, atol=1e-3)


def test_em_increases_likelihood():
    from yck.model.kalman import smoothed_stats
    from yck.model.statespace_dns import em_update

    y, _ = _simulate(TRUE, N=200, seed=2)
    y.iloc[::5, :2] = np.nan
    tau, Y = np.array(GRID, float), y.to_numpy()
    p = two_step_start(y)
    lls = []
    for _ in range(25):
        st = smoothed_stats(p.statespace(tau), Y)
        lls.append(st.loglik)
        p = em_update(p, st, tau)
    assert lls[-1] > lls[0] + 1.0
    # Monotone up to the first-state term, which the M-step leaves out.
    assert np.all(np.diff(lls) > -1e-3)


def test_noise_floor_is_respected_and_multistart_never_worse():
    y, _ = _simulate(DNSParams(TRUE.mu, TRUE.A, TRUE.Q, np.full(len(GRID), 0.004**2), TRUE.lam), N=200, seed=4)
    one = fit_dns_mle(y, h_floor_bp=1.0)
    assert np.all(np.sqrt(one.params.h) * 100 >= 1.0 - 1e-9)  # true noise is 0.4 bp, so the floor binds
    free = fit_dns_mle(y, h_floor_bp=None)
    assert np.sqrt(free.params.h).min() * 100 < 1.0
    multi = fit_dns_mle(y, h_floor_bp=1.0, n_starts=3)
    assert multi.loglik >= one.loglik - 1e-6


# --------------------------------------------------------------------------- yields + macro (DRA)
def _simulate_dra(N=300, seed=21, release_lag=2):
    """3 factors + 2 macro states; inflation (state 4) feeds into the level factor."""
    rng = np.random.default_rng(seed)
    k = 5
    A = np.diag([0.95, 0.9, 0.8, 0.9, 0.85])
    A[0, 4] = 0.08          # macro -> level: the effect the macro model should find
    A[3, 1] = -0.05         # slope -> real activity
    mu = np.array([5.0, -1.5, 0.0, 0.0, 2.0])
    Q = np.diag([0.04, 0.09, 0.25, 0.3, 0.1])
    p = DNSParams(mu=mu, A=A, Q=Q, h=np.full(len(GRID), 0.05**2), lam=0.0609)
    ss = p.statespace(np.array(GRID, float))
    x = np.empty((N, k))
    x[0] = ss.a1
    for t in range(1, N):
        x[t] = ss.c + A @ x[t - 1] + rng.normal(size=k) * np.sqrt(np.diag(Q))
    y = x[:, :3] @ ns_loadings(GRID).T + rng.normal(size=(N, len(GRID))) * 0.05
    idx = pd.date_range("1990-01-31", periods=N, freq="ME", name="date")
    data = pd.DataFrame(y, index=idx, columns=COLS)
    data["real_activity"], data["inflation"] = x[:, 3], x[:, 4]
    data.iloc[-release_lag:, -2:] = np.nan  # not yet published at the end of the sample
    return data, p, x


def test_macro_pack_unpack_and_layout():
    data, p, _ = _simulate_dra(N=50)
    theta = p.to_vector()
    assert theta.size == 5 + 25 + 15 + len(GRID)
    back = DNSParams.from_vector(theta, len(GRID), k=5)
    np.testing.assert_allclose(back.A, p.A, atol=1e-12)
    np.testing.assert_allclose(back.Q, p.Q, atol=1e-12)
    ss = p.statespace(np.array(GRID, float))
    assert ss.Z.shape == (len(GRID) + 2, 5)
    np.testing.assert_array_equal(ss.Z[len(GRID):, 3:], np.eye(2))       # macro observed 1:1
    np.testing.assert_array_equal(ss.Z[:len(GRID), 3:], 0.0)             # yields load on factors only


def test_macro_score_matches_numerical_gradient():
    from yck.model.kalman import kalman_loglik, smoothed_stats
    from yck.model.statespace_dns import score

    data, p, _ = _simulate_dra(N=120)
    Y, tau = data.to_numpy(), np.array(GRID, float)
    q = DNSParams(p.mu + 0.2, p.A * 0.97, p.Q * 1.4, p.h * 0.6, p.lam)
    theta = q.to_vector()
    g = score(theta, smoothed_stats(q.statespace(tau), Y), tau)

    def ll(th):
        return kalman_loglik(DNSParams.from_vector(th, len(GRID), k=5).statespace(tau), Y)

    num = np.array([(ll(theta + e) - ll(theta - e)) / 2e-6 for e in np.eye(theta.size) * 1e-6])
    np.testing.assert_allclose(g, num, rtol=1e-4, atol=2e-3)


@pytest.fixture(scope="module")
def dra_fit():
    data, p, x = _simulate_dra()
    return data, p, x, fit_dns_mle(data)


def test_macro_mle_recovers_macro_to_level_effect(dra_fit):
    data, p, _, fit = dra_fit
    assert fit.params.k == 5 and fit.state_names[3:] == ["real_activity", "inflation"]
    assert fit.params.A[0, 4] == pytest.approx(0.08, abs=0.04)
    assert fit.loglik >= kalman_filter(p.statespace(np.array(GRID, float)), data.to_numpy()).loglik - 1e-6


def test_macro_states_reproduce_observed_macro_and_forecast_yields_only(dra_fit):
    data, _, _, fit = dra_fit
    ff = fit.filtered_factors(data)
    obs = data[["real_activity", "inflation"]].dropna()
    np.testing.assert_allclose(ff.loc[obs.index, ["real_activity", "inflation"]], obs, atol=1e-3)
    assert fit.forecast(ff.iloc[-1].to_numpy(), 6).shape == (len(GRID),)


def test_yields_only_is_the_special_case_without_macro_columns(dra_fit):
    data, _, _, _ = dra_fit
    fit = fit_dns_mle(data[COLS])
    assert fit.params.k == 3 and fit.columns == COLS


# --------------------------------------------------------------------------- analytic score
@pytest.mark.parametrize("case", ["yields", "yields-lambda", "macro", "unstable-A"])
def test_analytic_score_equals_finite_difference_score(case):
    from yck.model.kalman import smoothed_stats
    from yck.model.statespace_dns import score, score_fd

    tau = np.array(GRID, float)
    if case.startswith("yields"):
        y, _ = _simulate(TRUE, N=120, seed=13)
        y.iloc[::5, :2] = np.nan
        Y = y.to_numpy()
        q = DNSParams(TRUE.mu + [0.3, -0.2, 0.1], TRUE.A * 0.98, TRUE.Q * 1.3, TRUE.h * 0.7, lam=0.07)
    else:
        data, p, _ = _simulate_dra(N=120)
        Y = data.to_numpy()
        q = DNSParams(p.mu + 0.2, p.A * 0.97, p.Q * 1.4, p.h * 0.6, p.lam)
    if case == "unstable-A":  # diffuse prior branch: P1 no longer depends on A, Q
        q = DNSParams(q.mu, q.A + np.eye(q.k) * 0.1, q.Q, q.h, q.lam)
        assert np.max(np.abs(np.linalg.eigvals(q.A))) >= 1
    est = case == "yields-lambda"
    theta = q.to_vector(est)
    st = smoothed_stats(q.statespace(tau), Y)
    np.testing.assert_allclose(score(theta, st, tau, est, q.lam), score_fd(theta, st, tau, est, q.lam),
                               rtol=1e-5, atol=1e-4)


def test_lambda_derivative_of_loadings():
    from yck.model.dns import ns_loadings_dlam
    tau = np.array([0.5, 3, 30, 120, 400.0])
    num = (ns_loadings(tau, 0.06 + 1e-7) - ns_loadings(tau, 0.06 - 1e-7)) / 2e-7
    np.testing.assert_allclose(ns_loadings_dlam(tau, 0.06), num, rtol=1e-6, atol=1e-8)
