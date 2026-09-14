"""Value run: fund CSV -> Graham IV -> backtest, tune/holdout, no selection.

Primary discount 0.5 (his). Sensitivity 0.3/0.7 reported, never selected.
Split: first half of months = tune (context), second half = holdout.
Fair benchmark (calculable names) + SPY reference always shown.

Usage:
  python3 value_run.py --fund data/value_fund_25.csv --market SPY
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from predictor.data import download_universe
from value.backtest import run_value_backtest
from value.fundamentals import load_csv
from value.graham import compute_iv_panel


def max_dd(equity: pd.Series) -> float:
    curve = pd.concat([pd.Series([1.0]), equity.reset_index(drop=True)],
                      ignore_index=True)
    return float((curve / curve.cummax() - 1).min())


def report(eq: pd.DataFrame, bench: pd.DataFrame, tag: str) -> dict:
    e = eq["equity"]
    yrs = max(1, (eq["date"].max() - eq["date"].min()).days / 365.25)
    m = eq.set_index("date")["equity"].resample("ME").last().pct_change().dropna()
    return {"tag": tag, "months": len(eq),
            "CAGR": round(float(e.iloc[-1] ** (1 / yrs) - 1), 4),
            "maxDD": round(max_dd(e), 3),
            "sharpe_m": round(float(m.mean() / (m.std() + 1e-12) * np.sqrt(12)), 3)
            if len(m) > 3 else 0.0,
            "bench_CAGR": round(float(bench["bench"].iloc[-1] ** (1 / yrs) - 1), 4)
            if len(bench) else 0.0,
            "avg_n": round(float(eq["n"].mean()), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fund", required=True)
    ap.add_argument("--market", default="SPY")
    ap.add_argument("--discounts", default="0.5,0.3,0.7")
    ap.add_argument("--cost", type=float, default=5.0)
    args = ap.parse_args()
    fund = load_csv(args.fund)
    print(f"filings: {len(fund)} tickers: {fund.ticker.nunique()} "
          f"{fund.filed_date.min().date()} -> {fund.filed_date.max().date()}")
    iv = compute_iv_panel(fund)
    iv = iv.dropna(subset=["iv"])
    print(f"IV rows: {len(iv)} tickers: {iv.ticker.nunique()}")
    tickers = sorted(iv.ticker.unique().tolist())
    paths = download_universe(tickers, market=args.market)
    prices = {}
    for tk, p in paths.items():
        try:
            d = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date")
            prices[tk] = d[["Date", "Close"]]
        except Exception as e:  # noqa: BLE001
            print(f"price skip {tk}: {e}")
    rows = []
    for disc in [float(x) for x in args.discounts.split(",")]:
        eq, bench = run_value_backtest(iv, prices, discount=disc,
                                       cost_bps=args.cost)
        mid = len(eq) // 2
        for tag, sl in (("tune", slice(None, mid)), ("hold", slice(mid, None))):
            r = report(eq.iloc[sl].reset_index(drop=True),
                       bench.iloc[sl].reset_index(drop=True),
                       f"d{disc}/{tag}")
            rows.append(r)
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    primary = out[(out.tag == "d0.5/hold")].iloc[0].to_dict()
    print("PRIMARY d0.5/hold:", {k: primary[k] for k in
          ("CAGR", "maxDD", "sharpe_m", "bench_CAGR", "avg_n")})


if __name__ == "__main__":
    main()
