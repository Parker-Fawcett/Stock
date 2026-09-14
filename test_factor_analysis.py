import numpy as np
import pandas as pd
import pytest

from factor_analysis import (block_bootstrap_indices, realized_month,
                             stat_cagr, stat_sharpe)


def test_block_bootstrap_indices_cover_valid_range_at_requested_length():
    rng = np.random.default_rng(0)
    idx = block_bootstrap_indices(n=10, block_len=3, rng=rng)
    assert len(idx) == 10
    assert idx.min() >= 0 and idx.max() < 10


def test_cagr_and_sharpe_match_hand_calculation_on_flat_returns():
    x = np.full(12, 0.01)
    assert stat_cagr(x) == pytest.approx(1.01 ** 12 - 1)
    assert np.isnan(stat_sharpe(np.zeros(5)))


def test_realized_month_is_one_calendar_month_after_decision_date():
    assert realized_month(pd.Timestamp("2020-02-28")) == pd.Period("2020-03", freq="M")
