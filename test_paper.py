import pandas as pd
import pytest

from paper import grade_position, is_month_end_data
from predictor.backtest import COST_BPS
from predictor.provenance import payload_sha256


def test_month_end_guard_accepts_final_weekday_and_rejects_midmonth():
    assert is_month_end_data(pd.Timestamp("2026-09-30"),
                             pd.Timestamp("2026-09-30"))
    assert is_month_end_data(pd.Timestamp("2026-10-30"),
                             pd.Timestamp("2026-10-30"))  # Saturday month-end
    assert not is_month_end_data(pd.Timestamp("2026-09-11"),
                                 pd.Timestamp("2026-09-14"))


def test_grade_uses_next_open_and_actual_close_on_gap_stop():
    dates = pd.bdate_range("2026-02-02", periods=20)
    stock = pd.DataFrame({
        "Open": [100.0] + [95.0] * 19,
        "Close": [95.0, 80.0] + [90.0] * 18,
    }, index=dates)
    market = pd.DataFrame({
        "Open": [50.0] + [51.0] * 19,
        "Close": [50.0] * 19 + [55.0],
    }, index=dates)
    result = grade_position(stock, market, pd.Timestamp("2026-01-30"))
    assert result is not None
    assert result["entry"] == 100.0
    assert result["exit"] == 80.0
    assert result["exit_date"] == "2026-02-03"
    assert result["ret"] == pytest.approx(-0.20 - 2 * COST_BPS / 1e4)
    assert result["market_ret"] == pytest.approx(0.10)


def test_manifest_hash_is_canonical():
    assert payload_sha256({"a": 1, "b": 2}) == payload_sha256({"b": 2, "a": 1})
