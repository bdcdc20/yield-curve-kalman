"""Offline tests for the Nelson-Siegel loadings and the two-step Diebold-Li baseline."""
import numpy as np
import pandas as pd
import pytest

from yck.model.dns import (
    DL_LAMBDA,
    FactorDynamics,
    fit_dynamics,
    fit_factors,
    fit_rmse_bp,
    fitted_yields,
    forecast_yields,
    maturities_from_columns,
    ns_loadings,
)

GRID = [3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 48, 60, 72, 84, 96, 108, 120]
COLS = [f"{m}m" for m in GRID]


def _curves(F: np.ndarray, lam: float = DL_LAMBDA) -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=len(F), freq="ME", name="date")
    return pd.DataFrame(F @ ns_loadings(GRID, lam).T, index=idx, columns=COLS)


# --------------------------------------------------------------------------- loadings
def test_loadings_limits():
    short, long_ = ns_loadings([1e-12, 1e9])
    assert short == pytest.approx([1.0, 1.0, 0.0], abs=1e-9)   # short rate = level + slope
    assert long_ == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)   # long rate = level


def test_curvature_loading_peaks_near_30_months():
    # The peak is at lam*tau ~= 1.793, i.e. ~29.4 months for DL's lambda.
    tau = np.linspace(1, 120, 11901)
    assert 29 < tau[np.argmax(ns_loadings(tau)[:, 2])] < 30


def test_maturities_from_columns():
    assert maturities_from_columns(["3m", "120m"]).tolist() == [3.0, 120.0]
    with pytest.raises(ValueError):
        maturities_from_columns(["10y"])


# --------------------------------------------------------------------------- step 1
def test_fit_factors_recovers_known_factors():
    F = np.array([[5.0, -2.0, 1.0], [4.0, 0.5, -1.5], [6.5, -3.0, 2.0]])
    est = fit_factors(_curves(F))
    np.testing.assert_allclose(est.to_numpy(), F, atol=1e-10)
    assert list(est.columns) == ["level", "slope", "curvature"]


def test_fit_factors_uses_only_observed_maturities_and_never_fills():
    F = np.array([[5.0, -2.0, 1.0], [4.0, 0.5, -1.5], [6.5, -3.0, 2.0]])
    y = _curves(F)
    y.iloc[0, :2] = np.nan          # GBP-style short-end gap
    y.iloc[1, [0, 5, 10]] = np.nan  # a different pattern
    y.iloc[2, 3:] = np.nan          # only 3 points left: still identified
    est = fit_factors(y)
    np.testing.assert_allclose(est.to_numpy(), F, atol=1e-9)
    assert y.isna().sum().sum() == 2 + 3 + 14   # input untouched


def test_fit_factors_too_few_points_gives_nan():
    y = _curves(np.array([[5.0, -2.0, 1.0], [4.0, 0.5, -1.5]]))
    y.iloc[0, 2:] = np.nan
    est = fit_factors(y)
    assert est.iloc[0].isna().all()
    assert est.iloc[1].notna().all()


def test_fit_rmse_is_zero_on_exact_curve_and_in_bp():
    F = np.array([[5.0, -2.0, 1.0], [4.0, 0.5, -1.5]])
    y = _curves(F)
    assert fit_rmse_bp(y, fit_factors(y)).max() < 1e-8
    noisy = y.copy()
    noisy.iloc[:, 0] += 0.10  # 10 bp error at 3m only
    # Fitted factors absorb part of it, so RMSE at 3m is positive but below 10 bp.
    assert 0 < fit_rmse_bp(noisy, fit_factors(noisy))["3m"] < 10


def test_fitted_yields_round_trip():
    F = np.array([[5.0, -2.0, 1.0]])
    y = _curves(F)
    np.testing.assert_allclose(fitted_yields(fit_factors(y), COLS).to_numpy(), y.to_numpy(), atol=1e-10)


# --------------------------------------------------------------------------- step 2
def _simulate_var(A, mu, T=4000, seed=0):
    rng = np.random.default_rng(seed)
    f = np.empty((T, 3))
    f[0] = mu
    for t in range(1, T):
        f[t] = mu + A @ (f[t - 1] - mu) + rng.normal(scale=0.2, size=3)
    idx = pd.date_range("1700-01-31", periods=T, freq="ME")
    return pd.DataFrame(f, index=idx, columns=["level", "slope", "curvature"])


def test_var1_recovers_parameters():
    A = np.array([[0.98, 0.02, 0.0], [0.0, 0.95, 0.03], [0.01, 0.0, 0.85]])
    mu = np.array([5.0, -1.5, 0.3])
    dyn = fit_dynamics(_simulate_var(A, mu), kind="var1")
    np.testing.assert_allclose(dyn.A, A, atol=0.03)
    np.testing.assert_allclose(np.diag(dyn.Q), 0.04, rtol=0.1)
    assert dyn.is_stationary()


def test_ar1_is_diagonal_and_recovers_mean():
    A = np.diag([0.95, 0.9, 0.8])
    mu = np.array([5.0, -1.5, 0.3])
    dyn = fit_dynamics(_simulate_var(A, mu), kind="ar1")
    assert np.count_nonzero(dyn.A - np.diag(np.diag(dyn.A))) == 0
    np.testing.assert_allclose(np.diag(dyn.A), np.diag(A), atol=0.02)
    np.testing.assert_allclose(dyn.mu, mu, atol=0.3)


def test_fit_dynamics_skips_pairs_across_gaps():
    f = _simulate_var(np.diag([0.9, 0.9, 0.9]), np.zeros(3), T=200)
    f.iloc[50] = np.nan
    assert fit_dynamics(f).nobs == 199 - 2


def test_forecast_formula():
    dyn = FactorDynamics(mu=np.array([5.0, -1.0, 0.0]), A=np.diag([0.9, 0.8, 0.5]), Q=np.eye(3), nobs=1)
    f = np.array([6.0, -2.0, 1.0])
    assert dyn.forecast(f, 0) == pytest.approx(f)
    assert dyn.forecast(f, 2) == pytest.approx([5.0 + 0.81, -1.0 - 0.64, 0.25])
    assert dyn.forecast(f, 500) == pytest.approx(dyn.mu)


def test_forecast_yields_matches_loadings_times_factor_forecast():
    dyn = FactorDynamics(mu=np.array([5.0, -1.0, 0.0]), A=np.diag([0.9, 0.8, 0.5]), Q=np.eye(3), nobs=1)
    F = np.array([[6.0, -2.0, 1.0], [4.0, 0.0, 0.5]])
    fac = pd.DataFrame(F, columns=["level", "slope", "curvature"])
    out = forecast_yields(fac, dyn, 6, COLS)
    expected = dyn.forecast(F, 6) @ ns_loadings(GRID).T
    np.testing.assert_allclose(out.to_numpy(), expected)
