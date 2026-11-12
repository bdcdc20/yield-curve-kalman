"""Offline tests for the trading evaluation (step 8)."""
import numpy as np
import pandas as pd
import pytest

from yck.eval.backtest import forward_forecasts, random_walk_forecasts
from yck.eval.trading import (difference_test, positions, realised_pnl, signal, strategy_pnl, summary)

IDX = pd.date_range("2000-01-31", periods=120, freq="ME")


def test_forward_pnl_is_the_zero_coupon_excess_return():
    # Holding an (h + tau)-month zero for h months, funded at the h-month rate, earns tau/12 (F - y_{t+h}(tau)).
    rng = np.random.default_rng(0)
    h, tau = 6, 24
    mats = [h, tau, tau + h]
    ext = pd.DataFrame(3 + rng.normal(scale=0.5, size=(40, 3)).cumsum(0) * 0.1, index=IDX[:40],
                       columns=[f"{m}m" for m in mats])
    F = forward_forecasts(ext, [f"{tau}m"], (h,), IDX[0])[h]
    d = realised_pnl(F, ext[[f"{tau}m"]])
    t = IDX[10]
    T = t + pd.offsets.MonthEnd(h)
    n = tau + h
    rx = (n * ext.loc[t, f"{n}m"] - tau * ext.loc[T, f"{tau}m"] - h * ext.loc[t, f"{h}m"]) / 12  # percent
    assert d.loc[T, f"{tau}m"] * tau / 12 / 100 == pytest.approx(rx)


def test_random_walk_signal_is_carry_and_rolldown():
    ext = pd.DataFrame({"12m": np.linspace(2, 3, 30), "24m": np.linspace(3, 4, 30), "36m": np.linspace(3.5, 4.5, 30)},
                       index=IDX[:30])
    F = forward_forecasts(ext, ["24m"], (12,), IDX[0])[12]
    rw = random_walk_forecasts(ext[["24m"]], (12,), IDX[0])[12]
    s = signal(F, rw)
    t = IDX[5]
    T = t + pd.offsets.MonthEnd(12)
    assert s.loc[T, "24m"] == pytest.approx(100 * (F.loc[T, "24m"] - ext.loc[t, "24m"]))


def test_scaled_positions_use_only_pnl_known_at_the_origin():
    rng = np.random.default_rng(1)
    pnl = pd.Series(rng.normal(size=120), index=IDX)
    sig = pd.Series(rng.normal(size=120), index=IDX)
    h = 6
    x = positions(sig, pnl, h, "scaled", min_history=24)
    T = IDX[60]
    sd_known = pnl.loc[:IDX[54]].std()  # targets up to the origin T - h
    assert x.loc[T] == pytest.approx(np.clip(sig.loc[T] / sd_known, -2, 2))
    assert x.loc[:IDX[24 + h - 2]].isna().all()  # not enough history yet
    pnl2 = pnl.copy()
    pnl2.loc[IDX[55]:] *= 100  # the future changes; positions up to T do not
    pd.testing.assert_series_equal(positions(sig, pnl2, h, "scaled").loc[:T], x.loc[:T])


def test_strategy_pnl_costs_and_tranche_scaling():
    pnl = pd.Series(10.0, index=IDX)
    sig = pd.Series(np.where(np.arange(120) % 2, 1.0, -1.0), index=IDX)
    out = strategy_pnl(sig, pnl, h=6, sizing="sign", cost_bp=0.5, leg_weight=2.0)
    np.testing.assert_allclose(out["gross"], np.sign(sig) * 10 / 6)
    np.testing.assert_allclose(out["cost"], 0.5 * 2.0 / 6)
    np.testing.assert_allclose(out["net"], out["gross"] - out["cost"])


def test_summary_and_difference_test():
    rng = np.random.default_rng(2)
    p = pd.Series(1.0 + rng.normal(size=600), index=pd.date_range("1950-01-31", periods=600, freq="ME"))
    s = summary(p, 1)
    assert s["sharpe"] == pytest.approx(np.sqrt(12) * p.mean() / p.std())
    assert s["t"] > 10 and 0.5 < s["hit"] < 1 and s["max_dd"] >= 0
    q = p - 0.5 + 0.1 * rng.normal(size=600)
    d = difference_test(p, q, 1)
    assert d["diff"] == pytest.approx(12 * (p - q).mean()) and d["t"] > 5


def test_sharpe_of_overlapping_tranches_is_annualised_by_sqrt_12_over_h():
    # Non-overlapping 12-month returns with mean 1 and sd 2: annual Sharpe 0.5, whatever the bookkeeping.
    rng = np.random.default_rng(3)
    annual = 1.0 + 2.0 * rng.standard_normal(4000)
    booked = pd.Series(annual / 12, index=pd.date_range("1000-01-31", periods=4000, freq="ME", unit="s"))
    s = summary(booked, 12)
    assert s["sharpe"] == pytest.approx(np.mean(annual) / np.std(annual, ddof=1), rel=1e-12)
    assert s["vol"] == pytest.approx(np.std(annual, ddof=1), rel=1e-12)
    assert s["mean"] == pytest.approx(np.mean(annual), rel=1e-12)
