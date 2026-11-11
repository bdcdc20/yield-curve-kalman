"""Offline tests for the step 7b forecast adjustments: shifting endpoints and AR(1) measurement errors."""
import numpy as np
import pandas as pd
import pytest

from yck.model.adjust import RHO_MAX, detrend, error_forecast, ewma_endpoint, fit_error_ar, survey_endpoint
from yck.model.dns import fit_factors, ns_loadings

GRID = [3, 12, 24, 60, 120]
COLS = [f"{m}m" for m in GRID]


def _curves(T=60, seed=0):
    rng = np.random.default_rng(seed)
    f = np.column_stack([5 + np.cumsum(rng.normal(scale=0.2, size=T)), rng.normal(-1, 0.3, T), rng.normal(0, 0.3, T)])
    idx = pd.date_range("2000-01-31", periods=T, freq="ME", name="date")
    return pd.DataFrame(f @ ns_loadings(GRID).T, index=idx, columns=COLS)


def test_ewma_endpoint_is_the_one_sided_recursion_and_ignores_the_future():
    y = _curves()
    e = ewma_endpoint(y, alpha=0.1)
    L = fit_factors(y)["level"].to_numpy()
    ref = [L[0]]
    for v in L[1:]:
        ref.append(0.1 * v + 0.9 * ref[-1])
    np.testing.assert_allclose(e.to_numpy(), ref, rtol=1e-12)
    y2 = y.copy()
    y2.iloc[40:] += 3.0
    pd.testing.assert_series_equal(e.iloc[:40], ewma_endpoint(y2, alpha=0.1).iloc[:40])


def test_ewma_endpoint_carries_through_months_without_a_level():
    y = _curves()
    y.iloc[10, :3] = np.nan  # two maturities left: no level that month
    e = ewma_endpoint(y)
    assert e.iloc[10] == e.iloc[9]


def test_survey_endpoint_respects_publication_dates():
    survey = pd.Series([7.5, 7.0], index=pd.to_datetime(["1992-01-01", "1993-01-01"]))
    idx = pd.date_range("1991-12-31", "1993-03-31", freq="ME")
    e = survey_endpoint(survey, idx, known_month=2)
    assert np.isnan(e.loc["1991-12-31"]) and np.isnan(e.loc["1992-01-31"])  # Q1 1992 survey not out yet
    assert e.loc["1992-02-29"] == 7.5
    assert e.loc["1993-01-31"] == 7.5  # held until the next survey is published
    assert e.loc["1993-02-28"] == 7.0


def test_detrend_shifts_yields_and_policy_rate_only():
    y = _curves(T=5)
    panel = y.assign(policy_rate=1.0, inflation=2.0)
    e = pd.Series(np.arange(5.0), index=y.index)
    d = detrend(panel, e, COLS)
    np.testing.assert_allclose(d[COLS].to_numpy(), y.to_numpy() - np.arange(5.0)[:, None])
    np.testing.assert_allclose(d["policy_rate"], 1.0 - np.arange(5.0))
    np.testing.assert_allclose(d["inflation"], 2.0)


def _ar1(T, m, rho, rng):
    e = np.empty(T)
    e[0] = m
    for t in range(1, T):
        e[t] = m + rho * (e[t - 1] - m) + rng.normal(scale=0.1)
    return e


def test_fit_error_ar_recovers_mean_and_persistence():
    rng = np.random.default_rng(1)
    resid = np.column_stack([_ar1(4000, 0.2, 0.8, rng), _ar1(4000, 0.0, 0.3, rng)])
    resid[::7, 0] = np.nan  # gaps only drop the pairs they touch
    m, rho = fit_error_ar(resid, "ar1c")
    np.testing.assert_allclose(rho, [0.8, 0.3], atol=0.03)
    np.testing.assert_allclose(m, [0.2, 0.0], atol=0.03)
    m0, rho0 = fit_error_ar(resid[:, 1:], "ar1")
    assert m0[0] == 0.0 and rho0[0] == pytest.approx(0.3, abs=0.03)


def test_fit_error_ar_clips_and_handles_short_columns():
    resid = np.column_stack([np.arange(50.0), np.r_[1.0, 2.0, np.full(48, np.nan)]])
    m, rho = fit_error_ar(resid, "ar1c")
    assert rho[0] <= RHO_MAX and np.isfinite(m[0])  # a trend would give rho >= 1
    assert rho[1] == 0.0 and m[1] == pytest.approx(1.5)  # one pair: no AR, just the mean


def test_error_forecast_decays_to_the_mean_and_handles_missing():
    m, rho = np.array([0.1, 0.0]), np.array([0.5, 0.9])
    e_t = np.array([0.5, np.nan])
    np.testing.assert_allclose(error_forecast(m, rho, e_t, 1), [0.1 + 0.5 * 0.4, 0.0])
    np.testing.assert_allclose(error_forecast(m, rho, e_t, 200), [0.1, 0.0], atol=1e-12)


# --------------------------------------------------------------------------- anchoring to forwards
def test_anchor_mean_hits_the_targets_and_is_the_gaussian_conditional_mean():
    from yck.model.adjust import anchor_mean
    rng = np.random.default_rng(0)
    B = rng.normal(size=(5, 5))
    cov = B @ B.T + 0.1 * np.eye(5)
    mean = rng.normal(size=5)
    target = np.array([1.0, np.nan, 2.0, np.nan, np.nan])
    out = anchor_mean(mean, cov, target)
    np.testing.assert_allclose(out[[0, 2]], [1.0, 2.0])
    s, r = [0, 2], [1, 3, 4]
    ref = mean[r] + cov[np.ix_(r, s)] @ np.linalg.solve(cov[np.ix_(s, s)], target[s] - mean[s])
    np.testing.assert_allclose(out[r], ref)
    np.testing.assert_allclose(anchor_mean(mean, cov, np.full(5, np.nan)), mean)
    # A maturity uncorrelated with the anchored ones does not move.
    cov2 = np.diag([1.0, 2.0, 3.0])
    np.testing.assert_allclose(anchor_mean(np.zeros(3), cov2, np.array([1.0, np.nan, np.nan])), [1.0, 0, 0])


def test_yield_forecast_cov_recursion():
    from yck.model.adjust import yield_forecast_cov
    T = np.array([[0.9, 0.1], [0.0, 0.5]])
    Q, Z, H = np.diag([0.2, 0.1]), np.array([[1.0, 0.5], [1.0, 0.0], [0.0, 1.0]]), 0.01 * np.eye(3)
    P0 = np.diag([0.3, 0.4])
    P2 = T @ (T @ P0 @ T.T + Q) @ T.T + Q
    np.testing.assert_allclose(yield_forecast_cov(T, Q, Z, H, P0, 2), Z @ P2 @ Z.T + H)
    np.testing.assert_allclose(yield_forecast_cov(T, Q, Z, H, P0, 0), Z @ P0 @ Z.T + H)
