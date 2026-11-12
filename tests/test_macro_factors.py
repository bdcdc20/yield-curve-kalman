"""Offline tests for large-panel macro factors (step 7b item 7): parsers, transforms, EM-PCA, real-time rules."""
import io

import numpy as np
import pandas as pd
import pytest

from yck.data import sources as src
from yck.model.macro_factors import RealTimeFactors, em_pca, extract_factors, transform


# --------------------------------------------------------------------------- parsers
def test_parse_fredmd_csv():
    text = ("sasdate,INDPRO,UNRATE\nTransform:,5,2\n1/1/2000,100,4.0\n2/1/2000,101,4.1\n3/1/2000,,4.2\n")
    df = src.parse_fredmd_csv(text)
    assert df.attrs["tcode"] == {"INDPRO": 5, "UNRATE": 2}
    assert list(df.index) == list(pd.to_datetime(["2000-01-31", "2000-02-29", "2000-03-31"]))
    assert np.isnan(df.loc["2000-03-31", "INDPRO"]) and df.loc["2000-02-29", "UNRATE"] == 4.1


def test_parse_ukmd_csv():
    text = (",Date,IOP_PROD,CPI_ALL\n1,Transform,5,5\n2,2000-01-01,99.0,80.0\n3,2000-02-01,99.5,80.2\n")
    df = src.parse_ukmd_csv(text)
    assert df.attrs["tcode"] == {"IOP_PROD": 5, "CPI_ALL": 5}
    assert df.index[-1] == pd.Timestamp("2000-02-29") and df.loc["2000-01-31", "IOP_PROD"] == 99.0


def test_parse_eamdqd_xlsx_keeps_monthly_series_and_maps_codes():
    buf = io.BytesIO()
    data = pd.DataFrame({"Time": pd.to_datetime(["2000-01-01", "2000-02-01", "2000-03-01"]),
                         "GDP_EA": [np.nan, np.nan, 1.0], "IPMN_EA": [100.0, 101.0, 102.0], "UNETOT_EA": [9.0, 9.1, 9.2]})
    info = pd.DataFrame({"Name": ["GDP_EA", "IPMN_EA", "UNETOT_EA"], "Frequency": ["Q", "M", "M"], "TR1": [2, 2, 5]})
    with pd.ExcelWriter(buf) as w:
        data.to_excel(w, sheet_name="data", index=False)
        info.to_excel(w, sheet_name="info", index=False)
    df = src.parse_eamdqd_xlsx(buf.getvalue())
    assert list(df.columns) == ["IPMN_EA", "UNETOT_EA"]
    assert df.attrs["tcode"] == {"IPMN_EA": 5, "UNETOT_EA": 2}  # EA codes 2 (dlog), 5 (d) -> McCracken-Ng 5, 2
    assert df.index[0] == pd.Timestamp("2000-01-31")


# --------------------------------------------------------------------------- transforms and PCA
def test_transform_codes():
    x = pd.DataFrame({c: [1.0, 2.0, 4.0, 8.0] for c in "abcdefg"})
    t = transform(x, {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7})
    np.testing.assert_allclose(t["b"].iloc[1:], [1, 2, 4])
    np.testing.assert_allclose(t["c"].iloc[2:], [1, 2])
    np.testing.assert_allclose(t["e"].iloc[1:], np.log(2))
    np.testing.assert_allclose(t["f"].iloc[2:], 0, atol=1e-12)
    np.testing.assert_allclose(t["g"].iloc[2:], 0, atol=1e-12)  # growth rate constant at 100%
    np.testing.assert_allclose(t["a"], x["a"])


def _factor_panel(T=240, N=40, seed=0):
    rng = np.random.default_rng(seed)
    F = np.cumsum(rng.normal(size=(T, 2)) * 0.3, axis=0) * 0 + rng.normal(size=(T, 2))
    L = rng.normal(size=(N, 2))
    X = F @ L.T + 0.3 * rng.normal(size=(T, N))
    return F, L, X


def test_em_pca_recovers_factors_with_missing_values():
    F, _, X = _factor_panel()
    rng = np.random.default_rng(1)
    X[rng.random(X.shape) < 0.15] = np.nan
    X[-3:, :25] = np.nan  # a ragged edge
    Xs = (X - np.nanmean(X, 0)) / np.nanstd(X, 0)
    Fh, _ = em_pca(Xs, 2)
    # Same space: regress the true factors on the estimates.
    B, *_ = np.linalg.lstsq(Fh, F - F.mean(0), rcond=None)
    resid = (F - F.mean(0)) - Fh @ B
    assert 1 - resid.var() / (F - F.mean(0)).var() > 0.95


def _panel_frame(seed=0, T=240):
    F, _, X = _factor_panel(T=T, seed=seed)
    idx = pd.date_range("2000-01-31", periods=T, freq="ME")
    cols = [f"s{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(np.cumsum(X, axis=0), index=idx, columns=cols)  # levels; code 2 differences them back
    df.attrs["tcode"] = {c: 2 for c in cols}
    return df


def test_factor_sign_follows_the_anchor_and_scale_is_unit():
    df = _panel_frame()
    f = extract_factors(df, df.attrs["tcode"], 2, anchors=("s0", "s1"))
    x = df.diff()
    assert f["factor1"].corr(x["s0"]) > 0
    np.testing.assert_allclose(f.std(), 1.0, rtol=1e-6)
    g = extract_factors(-df.rename(columns=lambda c: c), df.attrs["tcode"], 2, anchors=("s0", "s1"))
    assert g["factor1"].corr(-x["s0"]) > 0  # sign fixed by the anchor, whatever the data's sign


# --------------------------------------------------------------------------- real time
def test_vintages_are_used_only_once_published():
    a, b = _panel_frame(0).loc[:"2010-12-31"], _panel_frame(1).loc[:"2011-01-31"]
    rt = RealTimeFactors(1, a.attrs["tcode"], vintages={"2011-01": a, "2011-02": b})
    raw, _ = rt.panel_at("2011-01-31")
    pd.testing.assert_frame_equal(raw, a)
    raw, _ = rt.panel_at("2011-02-28")
    pd.testing.assert_frame_equal(raw, b.loc[:"2011-02-28"])
    with pytest.raises(ValueError):
        rt.panel_at("2010-12-31")


def test_ragged_edge_rule_and_no_look_ahead():
    df = _panel_frame().loc[:"2019-08-31"]
    df.loc["2019-08-31", ["s1", "s2"]] = np.nan  # two series published with a 2-month delay
    df.attrs["vintage"] = "2019-09"
    rt = RealTimeFactors(2, df.attrs["tcode"], latest=df)
    assert rt.delay["s0"] == 1 and rt.delay["s1"] == 2
    raw, _ = rt.panel_at("2015-06-30")
    assert raw.index[-1] == pd.Timestamp("2015-06-30")
    assert np.isnan(raw.loc["2015-06-30", "s0"]) and not np.isnan(raw.loc["2015-05-31", "s0"])
    assert np.isnan(raw.loc["2015-05-31", "s1"]) and not np.isnan(raw.loc["2015-04-30", "s1"])
    f = rt.at("2015-06-30")
    df2 = df.copy()
    df2.loc["2015-06-30":] += 50.0
    rt2 = RealTimeFactors(2, df.attrs["tcode"], latest=df2)
    pd.testing.assert_frame_equal(f, rt2.at("2015-06-30"))
