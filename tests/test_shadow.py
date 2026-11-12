"""Offline tests for the smooth shadow-rate DNS model (step 7b item 6)."""
import numpy as np
import pandas as pd
import pytest

from yck.model.dns import ns_loadings
from yck.model.kalman import kalman_loglik
from yck.model.shadow import ShadowFit, _ekf, fit_shadow_dns, lower_bound, zlb_yield
from yck.model.statespace_dns import fit_dns_mle

GRID = [3, 12, 24, 60, 120]
COLS = [f"{m}m" for m in GRID]


def test_zlb_yield_is_a_smooth_max():
    s = np.array([-3.0, 0.0, 5.0])
    y = zlb_yield(s, lb=0.0, kappa=1.0)
    assert y[2] == pytest.approx(5.0, abs=1e-5)          # far above the bound: the shadow yield
    assert y[1] == pytest.approx(1 / np.sqrt(2 * np.pi))  # at the bound: kappa phi(0)
    assert 0 < y[0] < 0.01                                 # far below: close to the bound, never under it
    assert np.all(np.diff(zlb_yield(np.linspace(-3, 3, 50), -0.5, 0.7)) > 0)


def test_zlb_yield_is_expected_max_of_shadow_plus_noise():
    rng = np.random.default_rng(0)
    z = rng.standard_normal(400_000)
    for s in (-0.5, 0.3, 1.2):
        mc = np.mean(np.maximum(0.0, s - (-0.2) + 0.8 * z)) + (-0.2)
        assert zlb_yield(s, -0.2, 0.8) == pytest.approx(mc, abs=3e-3)


def _panel(T=150, seed=1, low=False):
    rng = np.random.default_rng(seed)
    mu = np.array([1.0 if low else 5.0, -1.0, 0.0])
    f = np.empty((T, 3))
    f[0] = mu
    for t in range(1, T):
        f[t] = mu + 0.95 * (f[t - 1] - mu) + rng.normal(scale=0.2, size=3)
    s = f @ ns_loadings(GRID).T
    y = zlb_yield(s, 0.0, 1.0) if low else s
    y = y + rng.normal(scale=0.03, size=y.shape)
    return pd.DataFrame(y, index=pd.date_range("1990-01-31", periods=T, freq="ME", name="date"), columns=COLS)


def test_ekf_with_a_distant_bound_is_the_kalman_filter():
    y = _panel()
    y.iloc[::7, 1] = np.nan
    f = fit_dns_mle(y)
    ss = f.statespace()
    ll, a, _ = _ekf(y.to_numpy(), ss.Z, np.diag(ss.H).copy(), ss.T, ss.Q, ss.c, ss.a1, ss.P1, -1e3, 1.0, True)
    assert ll == pytest.approx(kalman_loglik(ss, y.to_numpy()), rel=1e-10)
    np.testing.assert_allclose(a, f.filtered_states(y), atol=1e-8)


def test_shadow_fit_improves_on_linear_near_the_bound_and_forecasts_respect_it():
    y = _panel(low=True)
    assert lower_bound(y) <= 0.0
    lin = fit_dns_mle(y)
    sh = fit_shadow_dns(y, lin.params, lin.loglik, lb=0.0)
    assert sh.loglik > lin.loglik + 5
    r = sh.filter(y)
    for h in (1, 12):
        plug = sh.forecast(r.a_filt[-1], h)
        mean = sh.forecast(r.a_filt[-1], h, r.P_filt[-1], "mean")
        assert np.all(plug > 0) and np.all(mean >= plug - 1e-12)  # Jensen: the measurement is convex
    # No state uncertainty at all (h = 0, P = 0): the exact mean is the plug-in forecast.
    np.testing.assert_allclose(sh.forecast(r.a_filt[-1], 0, np.zeros((3, 3)), "mean"), sh.forecast(r.a_filt[-1], 0))


def test_shadow_backtest_no_look_ahead_and_cache_round_trip(tmp_path):
    from yck.eval.backtest import dns_kalman_forecasts
    y = _panel(T=130, low=True)
    origin = y.index[115]
    kw = dict(horizons=(1, 6), first_origin=origin, reestimate_every=12, n_starts=1, shadow="mean", error_ar="ar1")
    fits = []
    base = dns_kalman_forecasts(y, cache_dir=tmp_path, fits=fits, **kw)
    assert isinstance(fits[0][1], ShadowFit)
    again = dns_kalman_forecasts(y, cache_dir=tmp_path, **kw)  # from the JSON cache
    for h in (1, 6):
        pd.testing.assert_frame_equal(base[h], again[h])
    y2 = y.copy()
    y2.iloc[116:] += 2.0
    after = dns_kalman_forecasts(y2, **kw)
    for h in (1, 6):
        t = origin + pd.offsets.MonthEnd(h)
        pd.testing.assert_series_equal(base[h].loc[t], after[h].loc[t], atol=1e-8)


def test_ansley_kohn_map_is_stationary_and_inverts():
    from yck.model.kalman import lyapunov
    from yck.model.shadow import free_from_stationary, stationary_from_free
    rng = np.random.default_rng(3)
    for _ in range(20):
        B = rng.normal(scale=3.0, size=(3, 3))
        L = np.tril(rng.normal(size=(3, 3)))
        np.fill_diagonal(L, np.abs(np.diag(L)) + 0.1)
        A, V = stationary_from_free(B, L)
        Q = L @ L.T
        assert np.max(np.abs(np.linalg.eigvals(A))) < 1
        np.testing.assert_allclose(V, A @ V @ A.T + Q, atol=1e-9)
        A2, _ = stationary_from_free(free_from_stationary(A, Q), L)
        np.testing.assert_allclose(A2, A, atol=1e-8)
