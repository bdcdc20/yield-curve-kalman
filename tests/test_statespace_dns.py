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
