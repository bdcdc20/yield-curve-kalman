"""Offline tests for the Diebold-Mariano and Clark-West tests."""
import numpy as np
import pandas as pd
import pytest

from yck.eval.stats import (_block_bootstrap_means, clark_west, cw_table, diebold_mariano, dm_table, mcs,
                             mcs_table, newey_west_var_of_mean)


def test_newey_west_by_hand():
    d = np.array([1.0, 3.0, 2.0, 6.0])
    u = d - d.mean()  # [-2, 0, -1, 3]
    g0 = (u @ u) / 4                       # 14/4
    g1 = (u[1:] @ u[:-1]) / 4              # (0*-2 + -1*0 + 3*-1)/4 = -3/4
    assert newey_west_var_of_mean(d, 0) == pytest.approx(g0 / 4)
    assert newey_west_var_of_mean(d, 1) == pytest.approx((g0 + 2 * 0.5 * g1) / 4)
    assert newey_west_var_of_mean(d, 1, "rectangular") == pytest.approx((g0 + 2 * g1) / 4)


def _overlapping_errors(rng, T, h, scale=1.0):
    """Errors of h-step forecasts made monthly: MA(h-1) sums of monthly shocks."""
    shocks = rng.normal(scale=scale, size=T + h - 1)
    return np.convolve(shocks, np.ones(h), mode="valid")


@pytest.mark.parametrize("h", [6, 12])
def test_dm_size_with_overlapping_errors_is_close_to_nominal(h):
    # Bartlett weights gave 8-11% here; the rectangular estimator with HLN gives ~5-6.5%.
    rng = np.random.default_rng(0)
    T, reps = 300, 1000
    rejections = 0
    for _ in range(reps):
        common = _overlapping_errors(rng, T, h)
        e1 = common + _overlapping_errors(rng, T, h, 0.5)
        e2 = common + _overlapping_errors(rng, T, h, 0.5)  # equally accurate
        rejections += diebold_mariano(e1, e2, h)["pvalue"] < 0.05
    assert 0.03 < rejections / reps < 0.08


def test_dm_detects_a_better_forecast_and_sign_convention():
    rng = np.random.default_rng(1)
    common = _overlapping_errors(rng, 300, 6)
    bench = common + _overlapping_errors(rng, 300, 6, 1.0)
    model = common + _overlapping_errors(rng, 300, 6, 0.3)
    r = diebold_mariano(bench, model, 6)
    assert r["stat"] > 0 and r["pvalue"] < 0.01  # positive = model more accurate
    assert diebold_mariano(model, bench, 6)["stat"] < 0


def test_dm_drops_missing_pairs():
    e1 = pd.Series([1.0, np.nan, 2.0] * 10)
    e2 = pd.Series([0.5, 1.0, np.nan] * 10)
    assert diebold_mariano(e1, e2, 1)["n"] == 10


def test_clark_west_formula_and_direction():
    rng = np.random.default_rng(2)
    T = 400
    y = rng.normal(size=T)
    f_small = np.zeros(T)
    f_large = 0.5 * y + rng.normal(scale=0.2, size=T)  # informative large model
    r = clark_west(y - f_small, y - f_large, f_small, f_large, 1)
    assert r["stat"] > 1.645 and r["pvalue"] < 0.05
    # Hand check of the adjusted loss on a tiny input (h = 1, no autocorrelation terms).
    es, el, fs, fl = np.array([1.0, 2.0]), np.array([0.5, 1.0]), np.array([0.0, 0.0]), np.array([0.5, 1.0])
    adj = es**2 - (el**2 - (fs - fl) ** 2)  # [1, 4]: the large model's deviation is added back
    assert adj.tolist() == [1.0, 4.0]


def test_dm_table_layout_and_ratio():
    idx = pd.date_range("2000-01-31", periods=40, freq="ME")
    rng = np.random.default_rng(3)
    actual = pd.DataFrame({"3m": rng.normal(size=40), "120m": rng.normal(size=40)}, index=idx)
    rw = actual + 1.0
    model = actual + 0.5
    t = dm_table({"RW": rw, "M": model}, "RW", actual, h=1)
    assert list(t.index) == ["3m", "120m"]
    assert t.loc["3m", ("M", "ratio")] == pytest.approx(0.5)
    assert t.loc["3m", ("M", "cell")].startswith("0.50")


def test_cw_table_matches_clark_west_on_the_common_sample():
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(4)
    actual = pd.DataFrame({"3m": rng.normal(size=60), "120m": rng.normal(size=60)}, index=idx)
    small = actual + rng.normal(scale=1.0, size=(60, 2))
    large = actual + rng.normal(scale=0.5, size=(60, 2))
    large.iloc[:5, 0] = np.nan  # dropped for every model at 3m
    t = cw_table({"S": small, "L": large}, "S", actual, h=1)
    ok = large["3m"].notna()
    ref = clark_west((small - actual)["3m"][ok], (large - actual)["3m"][ok], small["3m"][ok], large["3m"][ok], 1)
    assert t.loc["3m", ("L", "cw")] == pytest.approx(ref["stat"])
    assert t.loc["3m", ("L", "ratio")] < 1 and t.loc["3m", ("L", "cell")].endswith("***")
    assert ("S", "ratio") not in t.columns


# --------------------------------------------------------------------------- model confidence set
def test_block_bootstrap_means_match_explicit_resampling():
    rng = np.random.default_rng(3)
    L = rng.normal(size=(23, 2))
    got = _block_bootstrap_means(L, 5, 4, np.random.default_rng(9))
    starts = np.random.default_rng(9).integers(0, 23 - 5 + 1, size=(4, 5))
    for b in range(4):
        rows = np.concatenate([np.arange(s0, s0 + 5) for s0 in starts[b]])[:23]
        np.testing.assert_allclose(got[b], L[rows].mean(axis=0))


def test_mcs_drops_a_clearly_worse_model_and_keeps_equal_ones():
    rng = np.random.default_rng(0)
    T = 300
    common = rng.normal(size=T) ** 2
    L = pd.DataFrame({"a": common + 0.1 * rng.normal(size=T) ** 2, "b": common + 0.1 * rng.normal(size=T) ** 2,
                      "bad": common + 1.0 + 0.1 * rng.normal(size=T) ** 2})
    res = mcs(L, block=3, n_boot=1000)
    assert res.loc["bad", "p_mcs"] < 0.01 and res.loc["bad", "rank"] == 1
    assert res.loc[["a", "b"], "p_mcs"].min() > 0.10
    assert res["p_mcs"].max() == 1.0
    order = res.sort_values("rank")["p_mcs"].to_numpy()
    assert np.all(np.diff(order) >= 0)  # p-values never fall along the elimination order


def test_mcs_table_layout_and_pooled_column():
    idx = pd.date_range("2000-01-31", periods=200, freq="ME")
    rng = np.random.default_rng(2)
    actual = pd.DataFrame({"3m": rng.normal(size=200), "10y": rng.normal(size=200)}, index=idx)
    good = actual + 0.1 * rng.normal(size=(200, 2))
    bad = actual + 1.0 * rng.normal(size=(200, 2))
    bad.iloc[0, 0] = np.nan  # dropped from that maturity's sample only
    t = mcs_table({"good": good, "bad": bad}, actual, h=1, n_boot=500)
    assert list(t.columns) == ["3m", "10y", "all"]
    assert (t.loc["good"] == 1.0).all() and (t.loc["bad"] < 0.05).all()
