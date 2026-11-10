"""Offline tests for the recursive forecasting loop and benchmarks."""
import numpy as np
import pandas as pd
import pytest

from yck.eval.backtest import dns_kalman_forecasts, dns_two_step_forecasts, random_walk_forecasts, rmse_table
from yck.model.dns import ns_loadings

GRID = [3, 12, 24, 60, 120]
COLS = [f"{m}m" for m in GRID]


def _panel(T=120, seed=1):
    rng = np.random.default_rng(seed)
    f = np.empty((T, 3))
    f[0] = [5, -1, 0]
    for t in range(1, T):
        f[t] = [5, -1, 0] + 0.9 * (f[t - 1] - [5, -1, 0]) + rng.normal(scale=0.2, size=3)
    y = f @ ns_loadings(GRID).T + rng.normal(scale=0.05, size=(T, len(GRID)))
    idx = pd.date_range("1990-01-31", periods=T, freq="ME", name="date")
    return pd.DataFrame(y, index=idx, columns=COLS)


def test_no_look_ahead():
    y = _panel()
    origin = y.index[80]
    base = dns_two_step_forecasts(y, horizons=(1, 6), first_origin=origin)
    shocked = y.copy()
    shocked.iloc[81:] += 3.0  # rewrite the future
    after = dns_two_step_forecasts(shocked, horizons=(1, 6), first_origin=origin)
    for h in (1, 6):
        target = origin + pd.offsets.MonthEnd(h)
        pd.testing.assert_series_equal(base[h].loc[target], after[h].loc[target])


def test_forecasts_are_indexed_by_target_date():
    y = _panel()
    fc = dns_two_step_forecasts(y, horizons=(6,), first_origin=y.index[100])
    assert fc[6].index[0] == y.index[106]
    assert len(fc[6]) == len(y) - 100  # last origins forecast past the end of the data


def test_random_walk_is_shifted_curve():
    y = _panel()
    rw = random_walk_forecasts(y, horizons=(12,), first_origin=y.index[50])
    pd.testing.assert_series_equal(rw[12].loc[y.index[62]], y.iloc[50], check_names=False)


def test_rmse_table_uses_common_sample_and_bp():
    idx = pd.date_range("2000-01-31", periods=4, freq="ME")
    actual = pd.DataFrame({"3m": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    a = pd.DataFrame({"3m": [1.1, 1.1, 1.1, 5.0]}, index=idx)
    b = pd.DataFrame({"3m": [1.0, 1.0, 1.0, np.nan]}, index=idx)  # last point missing
    t = rmse_table({"a": a, "b": b}, actual)
    assert t.loc["3m", "a"] == pytest.approx(10.0)  # 0.1 pp = 10 bp; the 5.0 miss is excluded
    assert t.loc["3m", "b"] == pytest.approx(0.0)
    assert t.loc["3m", "n_obs"] == 3


def test_kalman_forecasts_no_look_ahead_and_reestimation_schedule():
    y = _panel(T=140)
    y.iloc[::5, 0] = np.nan  # missing points are fine for the filter
    origin = y.index[100]
    fits = []
    base = dns_kalman_forecasts(y, horizons=(1, 6), first_origin=origin, reestimate_every=4, fits=fits)
    assert [t for t, _ in fits] == list(y.index[100::4])  # re-estimated every 4 origins
    shocked = y.copy()
    shocked.iloc[101:] += 3.0  # rewrite everything after the first origin
    after = dns_kalman_forecasts(shocked, horizons=(1, 6), first_origin=origin, reestimate_every=4)
    for h in (1, 6):
        target = origin + pd.offsets.MonthEnd(h)
        pd.testing.assert_series_equal(base[h].loc[target], after[h].loc[target])
