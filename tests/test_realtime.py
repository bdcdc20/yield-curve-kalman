"""Offline tests for real-time macro panels (no look-ahead in the macro data)."""
import numpy as np
import pandas as pd
import pytest

from yck.data.realtime import RealTimeMacro

FAR = pd.Timestamp("2262-04-11")


def _vintages():
    """TCU-like level series and a PCE-like index, each with a first release and a later revision."""
    rows = []
    months = pd.date_range("2000-01-01", "2001-06-01", freq="MS")
    for i, d in enumerate(months):
        release = d + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)  # published mid next month
        revision = pd.Timestamp("2001-09-15")
        # capacity utilisation: first print 80 + i, revised up by 1 point
        rows.append(("TCU", d, release, revision - pd.Timedelta(days=1), 80.0 + i))
        rows.append(("TCU", d, revision, FAR, 81.0 + i))
        # price index: grows 0.5%/month in the first print, 1%/month after revision
        rows.append(("PCEPI", d, release, revision - pd.Timedelta(days=1), 100 * 1.005**i))
        rows.append(("PCEPI", d, revision, FAR, 100 * 1.01**i))
    v = pd.DataFrame(rows, columns=["series_id", "date", "realtime_start", "realtime_end", "value"])
    # Only data released by each date exist in that vintage (first prints have realtime_start = release).
    return v


SPECS = {
    "real_activity": {"source": "fred", "id": "TCU", "transform": "level", "vintages": True},
    "inflation": {"source": "fred", "id": "PCEPI", "transform": "yoy", "vintages": True},
    "policy_rate": {"source": "fred", "id": "FEDFUNDS", "transform": "level", "vintages": False},
}
LAGS = {"real_activity": 2, "inflation": 1, "policy_rate": 0}


def _latest():
    idx = pd.date_range("2000-01-31", "2001-06-30", freq="ME")
    return pd.DataFrame({"real_activity": 81.0 + np.arange(len(idx)), "inflation": 12.68,
                         "policy_rate": np.arange(len(idx), dtype=float)}, index=idx)


def test_vintaged_series_use_what_was_published_then():
    rt = RealTimeMacro(_latest(), SPECS, LAGS, _vintages())
    p = rt.at("2001-03-31")
    # Feb 2001 was released mid-March: known at end-March. March 2001 was not.
    assert p.loc["2001-02-28", "real_activity"] == 80.0 + 13  # first print, not the later revision
    assert "2001-03-31" not in p.index or np.isnan(p.loc["2001-03-31", "real_activity"])
    # After the September 2001 revision the revised value is used.
    assert rt.at("2001-09-30").loc["2001-02-28", "real_activity"] == 81.0 + 13


def test_yoy_is_computed_within_the_vintage():
    rt = RealTimeMacro(_latest(), SPECS, LAGS, _vintages())
    before = rt.at("2001-03-31").loc["2001-01-31", "inflation"]
    after = rt.at("2001-09-30").loc["2001-01-31", "inflation"]
    assert before == pytest.approx(100 * (1.005**12 - 1))
    assert after == pytest.approx(100 * (1.01**12 - 1))


def test_series_without_vintages_use_release_lag():
    rt = RealTimeMacro(_latest(), SPECS, {**LAGS, "policy_rate": 1}, _vintages())
    p = rt.at("2001-03-31")
    assert p["policy_rate"].last_valid_index() == pd.Timestamp("2001-02-28")


def test_origin_before_first_vintage_uses_earliest_vintage_with_lag():
    rt = RealTimeMacro(_latest(), SPECS, LAGS, _vintages())
    p = rt.at("2000-01-31")  # first release is mid-Feb 2000
    assert p["real_activity"].dropna().empty or p["real_activity"].last_valid_index() <= pd.Timestamp("1999-11-30")


def test_reindex_marks_unknown_months_missing():
    rt = RealTimeMacro(_latest(), SPECS, LAGS, _vintages())
    idx = pd.date_range("2000-01-31", "2001-06-30", freq="ME")
    p = rt.at("2001-03-31", index=idx)
    assert p.index[-1] == pd.Timestamp("2001-03-31")
    assert np.isnan(p.loc["2001-03-31", "real_activity"]) and p.loc["2001-03-31", "policy_rate"] == 14.0


def test_variable_subset():
    rt = RealTimeMacro(_latest(), SPECS, LAGS, _vintages(), variables=("policy_rate",))
    assert list(rt.at("2001-03-31").columns) == ["policy_rate"]
    with pytest.raises(ValueError):
        RealTimeMacro(_latest(), SPECS, LAGS, _vintages(), variables=("gdp",))


def test_withdrawn_history_falls_back_to_last_complete_vintage():
    """ALFRED marks PCEPI history missing for a few days around comprehensive revisions."""
    v = _vintages()
    gap_start, gap_end = pd.Timestamp("2001-04-20"), pd.Timestamp("2001-05-05")
    live = (v["realtime_start"] <= gap_start) & (v["realtime_end"] >= gap_start)
    withdrawn = v[(v["series_id"] == "PCEPI") & live].copy()
    v.loc[withdrawn.index, "realtime_end"] = gap_start - pd.Timedelta(days=1)  # old vintage ends
    blank = withdrawn.assign(realtime_start=gap_start, realtime_end=gap_end, value=np.nan)
    blank.loc[blank["date"] >= pd.Timestamp("2001-03-01"), "value"] = 200.0  # only recent months on the new basis
    reissued = withdrawn.assign(realtime_start=gap_end + pd.Timedelta(days=1))
    v = pd.concat([v, blank, reissued], ignore_index=True)

    rt = RealTimeMacro(_latest(), SPECS, LAGS, v)
    during = rt.at("2001-04-30")["inflation"].dropna()
    assert during.loc["2001-02-28"] == pytest.approx(100 * (1.005**12 - 1))  # the vintage of 2001-04-19
    assert during.index[-1] == pd.Timestamp("2001-03-31")  # March was printed on 2001-04-15, before the gap
    assert rt.at("2001-05-31")["inflation"].dropna().index[-1] == pd.Timestamp("2001-04-30")
