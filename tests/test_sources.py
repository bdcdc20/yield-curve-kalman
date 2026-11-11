"""Offline tests: each parser against a small sample in the source's real layout."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from yck.data import sources as src
from yck.data.curves import month_end, svensson_yield


# --------------------------------------------------------------------------- curve maths
def test_svensson_short_end_limit_is_beta0_plus_beta1():
    p = pd.DataFrame({"BETA0": [5.0], "BETA1": [-2.0], "BETA2": [1.0], "BETA3": [0.5],
                      "TAU1": [1.5], "TAU2": [8.0]})
    y = svensson_yield(p, [1e-10, 1000.0])
    assert y.iloc[0, 0] == pytest.approx(3.0, abs=1e-6)   # short rate = b0 + b1
    assert y.iloc[0, 1] == pytest.approx(5.0, abs=1e-2)   # long rate -> b0


def test_svensson_matches_hand_calculation():
    p = pd.DataFrame({"BETA0": [4.0], "BETA1": [-1.0], "BETA2": [2.0], "BETA3": [-1.5],
                      "TAU1": [2.0], "TAU2": [10.0]})
    n = 5.0
    x1, x2 = n / 2.0, n / 10.0
    l1 = (1 - np.exp(-x1)) / x1
    l2 = (1 - np.exp(-x2)) / x2
    expected = 4.0 - 1.0 * l1 + 2.0 * (l1 - np.exp(-x1)) - 1.5 * (l2 - np.exp(-x2))
    assert svensson_yield(p, [n]).iloc[0, 0] == pytest.approx(expected, rel=1e-12)


def test_svensson_drops_fourth_term_when_missing():
    ns = pd.DataFrame({"BETA0": [4.0], "BETA1": [-1.0], "BETA2": [2.0], "BETA3": [np.nan],
                       "TAU1": [2.0], "TAU2": [np.nan]})
    zero = ns.assign(BETA3=0.0, TAU2=5.0)
    assert svensson_yield(ns, [3.0]).iloc[0, 0] == pytest.approx(svensson_yield(zero, [3.0]).iloc[0, 0])


def test_month_end_takes_whole_last_row_and_keeps_gaps():
    idx = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-27"])
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, np.nan, 30.0]}, index=idx)
    m = month_end(df, today=pd.Timestamp("2026-11-09"))
    assert list(m.index) == list(pd.to_datetime(["2020-01-31", "2020-02-29"]))
    assert m.loc["2020-01-31", "a"] == 2.0
    assert np.isnan(m.loc["2020-01-31", "b"])  # not back-filled from the 30th


# --------------------------------------------------------------------------- Fed GSW
GSW_SAMPLE = """The Federal Reserve Board
Note: ...
Series,Compounding Convention,...
,,
Date,BETA0,BETA1,BETA2,BETA3,SVEN1F01,SVENY01,TAU1,TAU2
1961-06-14,3.917606,-1.277955,-1.949397,NA,3.5,3.0,0.339218,NA
2024-01-02,4.04,0.82,-1.66,-2.1,4.2,4.5,1.9,12.5
"""


def test_parse_gsw():
    df = src.parse_gsw(GSW_SAMPLE)
    assert list(df.columns) == src.SVENSSON_PARAMS
    assert df.index[0] == pd.Timestamp("1961-06-14")
    assert np.isnan(df.loc["1961-06-14", "BETA3"])
    assert df.loc["2024-01-02", "TAU2"] == 12.5


# --------------------------------------------------------------------------- ECB
ECB_SAMPLE = """KEY,FREQ,REF_AREA,CURRENCY,PROVIDER_FM,INSTRUMENT_FM,PROVIDER_FM_ID,DATA_TYPE_FM,TIME_PERIOD,OBS_VALUE,TITLE
YC.B.U2.EUR.4F.G_N_A.SV_C_YM.BETA0,B,U2,EUR,4F,G_N_A,SV_C_YM,BETA0,2024-01-02,2.1,x
YC.B.U2.EUR.4F.G_N_A.SV_C_YM.TAU1,B,U2,EUR,4F,G_N_A,SV_C_YM,TAU1,2024-01-02,1.7,x
YC.B.U2.EUR.4F.G_N_A.SV_C_YM.BETA0,B,U2,EUR,4F,G_N_A,SV_C_YM,BETA0,2024-01-03,2.2,x
"""


def test_parse_ecb_csv():
    df = src.parse_ecb_csv(ECB_SAMPLE)
    assert set(df.columns) == {"BETA0", "TAU1"}
    assert df.loc["2024-01-03", "BETA0"] == 2.2
    assert np.isnan(df.loc["2024-01-03", "TAU1"])


# --------------------------------------------------------------------------- BoE GLC
def _boe_sheet(unit, mats, rows):
    sheet = [["UK nominal spot curve"], [None], [unit + ":"] + mats, [None]]
    return pd.DataFrame(sheet + rows)


def _boe_zip():
    main = _boe_sheet("years", [0.5, 1.0, 1.5, 2.0],
                      [[pd.Timestamp("2024-01-02"), 4.9, 4.6, 4.3, 4.1]])
    short = _boe_sheet("months", [1, 2, 3, 6],
                       [[pd.Timestamp("2024-01-02"), 5.1, 5.05, 5.0, 4.95]])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        short.to_excel(xw, sheet_name="3. spot, short end", header=False, index=False)
        main.to_excel(xw, sheet_name="4. spot curve", header=False, index=False)
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("GLC Nominal daily data_2016 to present.xlsx", buf.getvalue())
    return z.getvalue()


def test_month_end_drops_unfinished_current_month():
    idx = pd.to_datetime(["2026-10-30", "2026-11-06"])
    m = month_end(pd.DataFrame({"a": [1.0, 2.0]}, index=idx), today=pd.Timestamp("2026-11-09"))
    assert list(m.index) == [pd.Timestamp("2026-10-31")]


def test_parse_boe_zip_accepts_alternative_sheet_names():
    df = src.parse_boe_zip(_boe_zip(), ["4. nominal spot curve", "4. spot curve"],
                           ["3. nominal spot, short end", "3. spot, short end"])
    assert 6.0 in df.columns


def test_parse_boe_zip_merges_sheets_in_months():
    df = src.parse_boe_zip(_boe_zip(), "4. spot curve", "3. spot, short end")
    assert list(df.columns) == [1.0, 2.0, 3.0, 6.0, 12.0, 18.0, 24.0]
    assert df.loc["2024-01-02", 6.0] == 4.95   # short-end sheet wins on overlap
    assert df.loc["2024-01-02", 18.0] == 4.3


# --------------------------------------------------------------------------- BoE IADB / ONS
def test_parse_iadb():
    s = src.parse_iadb('DATE,IUDBEDR\n02 Jan 1975,11.5\n03 Jan 1975,11.25\n', "IUDBEDR")
    assert s.loc["1975-01-03"] == 11.25


ONS_SAMPLE = '''"Title","CPI ANNUAL RATE 00: ALL ITEMS 2015=100"
"CDID","D7G7"
"Release date","15-10-2025"
"1989","5.2"
"1989 Q1","4.9"
"1989 JAN","4.9"
"1989 FEB","5.0"
'''


def test_parse_ons_keeps_only_monthly():
    s = src.parse_ons(ONS_SAMPLE, "cpi")
    assert list(s.index) == list(pd.to_datetime(["1989-01-31", "1989-02-28"]))
    assert s.iloc[-1] == 5.0


# --------------------------------------------------------------------------- FRED / ALFRED
@pytest.mark.parametrize("date_col", ["observation_date", "DATE"])
def test_parse_fredgraph(date_col):
    s = src.parse_fredgraph(f"{date_col},TCU\n2020-01-01,77.1\n2020-02-01,.\n", "TCU")
    assert s.iloc[0] == 77.1 and np.isnan(s.iloc[1])


def test_alfred_as_of_returns_what_was_published_then():
    obs = [
        {"date": "2020-01-01", "realtime_start": "2020-02-14", "realtime_end": "2020-03-16", "value": "76.8"},
        {"date": "2020-01-01", "realtime_start": "2020-03-17", "realtime_end": "9999-12-31", "value": "77.0"},
        {"date": "2020-02-01", "realtime_start": "2020-03-17", "realtime_end": "9999-12-31", "value": "."},
    ]
    v = src.parse_alfred(obs, "TCU")
    assert src.as_of(v, "2020-03-01").to_dict() == {pd.Timestamp("2020-01-01"): 76.8}
    later = src.as_of(v, "2021-01-01")
    assert later.loc["2020-01-01"] == 77.0 and np.isnan(later.loc["2020-02-01"])


# --------------------------------------------------------------------------- surveys
def test_parse_spf_xlsx_drops_blank_quarters():
    buf = io.BytesIO()
    pd.DataFrame({"YEAR": [1991, 1992, 1992, 1993], "QUARTER": [4, 1, 2, 1],
                  "BOND10": [np.nan, 7.5, np.nan, 7.0]}).to_excel(buf, index=False, sheet_name="Median_Level")
    s = src.parse_spf_xlsx(buf.getvalue(), "BOND10")
    assert list(s.index) == list(pd.to_datetime(["1992-01-01", "1993-01-01"]))
    assert list(s) == [7.5, 7.0]


def test_parse_ecb_csv_quarterly_periods():
    text = ("KEY,FREQ,TIME_PERIOD,OBS_VALUE\n"
            "SPF.Q.U2.HICP.POINT.LT.Q.AVG,Q,1999-Q1,1.86\n"
            "SPF.Q.U2.HICP.POINT.LT.Q.AVG,Q,2026-Q4,2.02\n")
    df = src.parse_ecb_csv(text)
    assert list(df.index) == list(pd.to_datetime(["1999-01-01", "2026-10-01"]))
    assert df["AVG"].iloc[-1] == 2.02
