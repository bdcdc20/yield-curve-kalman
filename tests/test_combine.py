"""Offline tests for forecast combination (step 7b item 3)."""
import numpy as np
import pandas as pd
import pytest

from yck.eval.combine import combine

IDX = pd.date_range("2000-01-31", periods=80, freq="ME", name="date")


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    actual = pd.DataFrame({"3m": rng.normal(size=80), "10y": rng.normal(size=80)}, index=IDX)
    good = actual + 0.1 * rng.normal(size=(80, 2))
    bad = actual + 1.0 * rng.normal(size=(80, 2))
    return actual, good, bad


def test_equal_weights_is_the_average_and_skips_missing_models():
    actual, good, bad = _setup()
    bad.iloc[5, 0] = np.nan
    c = combine({"g": good, "b": bad}, actual, h=1)
    np.testing.assert_allclose(c.iloc[10], (good.iloc[10] + bad.iloc[10]) / 2)
    assert c.iloc[5, 0] == good.iloc[5, 0]  # only the available forecast


def test_inverse_mse_uses_only_errors_known_at_the_origin():
    actual, good, bad = _setup()
    h, min_obs = 6, 24
    c = combine({"g": good, "b": bad}, actual, h, "inv_mse", min_obs)
    # Equal weights until each model has min_obs errors with targets <= T - h (rows 0..T-h).
    last_equal = min_obs + h - 2
    np.testing.assert_allclose(c.iloc[last_equal], (good.iloc[last_equal] + bad.iloc[last_equal]) / 2)
    assert not np.allclose(c.iloc[last_equal + 1], (good.iloc[last_equal + 1] + bad.iloc[last_equal + 1]) / 2)
    T = 50
    e_g = ((good - actual).iloc[:T - h + 1] ** 2).mean()
    e_b = ((bad - actual).iloc[:T - h + 1] ** 2).mean()
    w = (1 / e_g) / (1 / e_g + 1 / e_b)
    np.testing.assert_allclose(c.iloc[T], w * good.iloc[T] + (1 - w) * bad.iloc[T])
    assert (w > 0.9).all()
    # Rewriting realised yields after T - h does not change the combination for target T.
    a2 = actual.copy()
    a2.iloc[T - h + 1:] += 5.0
    c2 = combine({"g": good, "b": bad}, a2, h, "inv_mse", min_obs)
    pd.testing.assert_series_equal(c.iloc[T], c2.iloc[T])


def test_combine_rejects_unknown_weights():
    actual, good, bad = _setup()
    with pytest.raises(ValueError):
        combine({"g": good}, actual, 1, "best")


def test_returned_weights_sum_to_one_and_reproduce_the_combination():
    actual, good, bad = _setup()
    c, w = combine({"g": good, "b": bad}, actual, 1, "inv_mse", return_weights=True)
    np.testing.assert_allclose((w["g"] + w["b"]).to_numpy(), 1.0)
    np.testing.assert_allclose(c.to_numpy(), (w["g"] * good.loc[c.index] + w["b"] * bad.loc[c.index]).to_numpy())
