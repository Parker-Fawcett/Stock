import numpy as np
import pandas as pd
import pytest

from predictor.backtest import equal_weight_turnover, leg_simple
from predictor.evaluate import max_drawdown


def test_equal_weight_turnover_handles_replacements_and_resizing():
    assert equal_weight_turnover(["A", "B"], ["C", "D"]) == pytest.approx(1.0)
    assert equal_weight_turnover(["A", "B"], ["A", "B", "C"]) == pytest.approx(1 / 3)


def test_gap_stop_uses_observed_close_and_rejects_invalid_fraction():
    dates = pd.bdate_range("2026-01-02", periods=3)
    prices = pd.Series([100.0, 80.0, 90.0], index=dates)
    assert leg_simple(prices, dates[0], 2, False, 0.85) == pytest.approx(-0.20)
    with pytest.raises(ValueError):
        leg_simple(prices, dates[0], 2, False, 2.0)


def test_drawdown_includes_starting_equity():
    assert max_drawdown(pd.Series([0.8, 0.9])) == pytest.approx(-0.20)


def test_leg_simple_raises_on_non_finite_price():
    """A ticker missing a row for one date in a fold's panel (normal --
    not every name has a row on every date) must not silently become a NaN
    return that corrupts the whole month's weighted average. It must raise,
    so _run_book's per-leg try/except drops only that name."""
    dates = pd.bdate_range("2026-01-02", periods=5)
    entry_missing = pd.Series([np.nan, 101.0, 102.0, 103.0, 104.0], index=dates)
    with pytest.raises(ValueError):
        leg_simple(entry_missing, dates[0], 3, False, None)

    exit_missing = pd.Series([100.0, 101.0, 102.0, np.nan, np.nan], index=dates)
    with pytest.raises(ValueError):
        leg_simple(exit_missing, dates[0], 3, False, None)
