"""Offline tests for the recursive forecasting loop and benchmarks."""
import numpy as np
import pandas as pd
import pytest

from yck.eval.backtest import (dns_kalman_forecasts, dns_two_step_forecasts, forward_forecasts,
                               random_walk_forecasts, rmse_table)
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
    base = dns_kalman_forecasts(y, horizons=(1, 6), first_origin=origin, reestimate_every=4, n_starts=1, fits=fits)
    assert [t for t, _ in fits] == list(y.index[100::4])  # re-estimated every 4 origins
    shocked = y.copy()
    shocked.iloc[101:] += 3.0  # rewrite everything after the first origin
    after = dns_kalman_forecasts(shocked, horizons=(1, 6), first_origin=origin, reestimate_every=4, n_starts=1)
    for h in (1, 6):
        target = origin + pd.offsets.MonthEnd(h)
        pd.testing.assert_series_equal(base[h].loc[target], after[h].loc[target])


def test_kalman_forecasts_parallel_and_cache_reproduce_serial(tmp_path):
    y = _panel(T=110)
    kw = dict(horizons=(1, 6), first_origin=y.index[96], reestimate_every=6, n_starts=1)
    serial = dns_kalman_forecasts(y, **kw)
    parallel = dns_kalman_forecasts(y, jobs=2, cache_dir=tmp_path, **kw)
    assert len(list(tmp_path.glob("*.json"))) == 3  # one cached fit per re-estimation origin
    cached_fits = []
    cached = dns_kalman_forecasts(y, cache_dir=tmp_path, fits=cached_fits, **kw)
    for h in (1, 6):
        pd.testing.assert_frame_equal(serial[h], parallel[h])
        pd.testing.assert_frame_equal(serial[h], cached[h])
    # Changing the data changes the cache key, so nothing stale is reused.
    y2 = y.copy()
    y2.iloc[0, 0] += 0.01
    dns_kalman_forecasts(y2, cache_dir=tmp_path, **kw)
    assert len(list(tmp_path.glob("*.json"))) == 6


# --------------------------------------------------------------------------- forward-rate benchmark
def test_forward_of_flat_curve_is_flat_and_matches_hand_calculation():
    idx = pd.date_range("2000-01-31", periods=3, freq="ME", name="date")
    flat = pd.DataFrame(4.0, index=idx, columns=["1m", "6m", "12m", "13m", "18m", "24m"])
    fw = forward_forecasts(flat, ["12m"], horizons=(1, 6, 12), first_origin=idx[0])
    for h in (1, 6, 12):
        np.testing.assert_allclose(fw[h]["12m"], 4.0)
    # Upward-sloping: y(1m)=1, y(13m)=3 -> 12m rate starting in 1 month = (13*3 - 1*1)/12.
    up = flat.assign(**{"1m": 1.0, "13m": 3.0})
    assert forward_forecasts(up, ["12m"], horizons=(1,), first_origin=idx[0])[1]["12m"].iloc[0] == pytest.approx(38 / 12)


def test_forward_missing_inputs_stay_missing_and_index_is_target_date():
    idx = pd.date_range("2000-01-31", periods=3, freq="ME", name="date")
    c = pd.DataFrame({"6m": [2.0, np.nan, 2.0], "12m": 2.5, "18m": 3.0}, index=idx)
    fw = forward_forecasts(c, ["12m", "24m"], horizons=(6,), first_origin=idx[0])[6]
    assert fw.index[0] == pd.Timestamp("2000-07-31")
    assert np.isnan(fw["12m"].iloc[1])           # y(6m) missing that month
    assert fw["24m"].isna().all()                # y(30m) not available at all
    assert fw["12m"].iloc[0] == pytest.approx((18 * 3.0 - 6 * 2.0) / 12)


class _LaggedMacro:
    """Minimal real-time macro source: latest values, last `lag` months before each origin unknown."""

    def __init__(self, latest, lag=1):
        self.latest, self.lag = latest, lag

    def at(self, origin, index=None):
        known = self.latest.loc[: origin - pd.offsets.MonthEnd(self.lag)]
        return known.reindex(index[index <= origin]) if index is not None else known


def test_macro_forecasts_have_no_look_ahead_in_yields_or_macro():
    y = _panel(T=130)
    rng = np.random.default_rng(5)
    m = pd.DataFrame({"inflation": 2 + np.cumsum(rng.normal(scale=0.1, size=len(y)))}, index=y.index)
    origin = y.index[110]
    kw = dict(horizons=(1, 6), first_origin=origin, reestimate_every=12, n_starts=1)
    base = dns_kalman_forecasts(y, macro=_LaggedMacro(m), **kw)
    y2, m2 = y.copy(), m.copy()
    y2.iloc[111:] += 2.0
    m2.iloc[110:] += 5.0  # month 110 itself is not yet released at origin 110 (lag 1)
    after = dns_kalman_forecasts(y2, macro=_LaggedMacro(m2), **kw)
    for h in (1, 6):
        target = origin + pd.offsets.MonthEnd(h)
        pd.testing.assert_series_equal(base[h].loc[target], after[h].loc[target])
    # ...and the macro data do matter: changing the released history changes the forecast.
    m3 = m.copy()
    m3.iloc[:100] += 1.0
    other = dns_kalman_forecasts(y, macro=_LaggedMacro(m3), **kw)
    assert not np.allclose(base[1].iloc[0], other[1].iloc[0])
