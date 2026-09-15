import numpy as np

from pathlib import Path

from qc_reconcile import qc_daily_equity, qc_pick_table


def test_qc_daily_equity_uses_last_close_per_new_york_day():
    payload = {
        "charts": {"Strategy Equity": {"series": {"Equity": {"values": [
            [1609477200, 100, 101, 99, 100],
            [1609506000, 100, 103, 99, 102],
            [1609765200, 102, 104, 101, 103],
        ]}}}}
    }

    equity = qc_daily_equity(payload)

    assert len(equity) == 2
    assert np.allclose(equity.values, [102, 103])


def test_qc_pick_table_normalizes_historical_tickers(tmp_path: Path):
    log = tmp_path / "run.txt"
    log.write_text("2021-01-01 10:30:00 picks: BSIG,GET,UCBI\n")
    payload = {"orders": {
        "1": {"symbol": {"value": "RHP", "id": "GET R735", "permtick": "RHP"}},
        "2": {"symbol": {"value": "UCB", "id": "UCBI SD3K", "permtick": "UCB"}},
    }}

    picks = qc_pick_table(log, payload)

    assert picks.iloc[0]["qc_picks_current"] == "AAMI,RHP,UCB"
