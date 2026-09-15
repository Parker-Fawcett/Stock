import numpy as np
import pandas as pd

import improve


def test_rank_book_flat_prices_cannot_create_returns():
    dates = pd.bdate_range("2020-01-01", "2021-04-30")
    tickers = ["A", "B", "C", "D"]
    px = pd.DataFrame(100.0, index=dates, columns=tickers)
    decisions = [pd.Timestamp("2021-01-29"), pd.Timestamp("2021-02-26")]
    test = pd.DataFrame(
        [{"Date": date, "ticker": ticker}
         for date in decisions for ticker in tickers]
    )
    proba = np.tile([0.1, 0.2, 0.3, 0.4], len(decisions))

    nets = improve._rank_book(test, proba, px, ml_weight=1.0)

    assert list(nets.index) == decisions
    assert np.isclose(nets.iloc[0], -0.5 * improve.COST / 1e4)
    assert np.isclose(nets.iloc[1], 0.0)
