"""Multi-asset trend (Faber-style): hold asset if above 10-month SMA.

Frozen rule, reported once, no selection. Assets: SPY, IWM, TLT, GLD.
Equal weight among held, monthly rebalance, 10bps (ETFs are liquid).
Evaluated 2011+, split at midpoint, vs SPY buy-hold.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from predictor import evaluate as ev
from predictor.data import download_universe

ASSETS = ["SPY", "IWM", "TLT", "GLD"]
LOOKBACK = 210  # ~10 trading months


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=10.0)
    ap.add_argument("--assets", default="SPY,IWM,TLT,GLD",
                    help="comma-separated ETFs (pre-committed intl test: EFA,EEM,VNQ,GLD)")
    ap.add_argument("--stride", type=int, default=1,
                    help="rebalance every Nth month (pre-committed test: 3)")
    ap.add_argument("--hold", type=int, default=21,
                    help="trading days to hold (pre-committed test: 63)")
    args = ap.parse_args()
    assets = [a.strip().upper() for a in args.assets.split(",") if a.strip()]
    paths = download_universe(assets, market="SPY")
    px = {}
    for tk, p in paths.items():
        try:
            d = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date")
            px[tk.upper()] = d.set_index("Date")["Close"]
        except Exception as e:  # noqa: BLE001
            print(f"skip {tk}: {e}")
    px = pd.DataFrame(px).sort_index().dropna(how="all")
    months = pd.to_datetime(px.index.to_series()).dt.to_period("M").drop_duplicates().sort_values()
    rows, prev = [], []
    trade_months = set(months[i] for i in range(0, len(months), args.stride))
    for m in months:
        d0 = px.index[pd.to_datetime(px.index.to_series()).dt.to_period("M") == m].max()
        hist = px.loc[px.index <= d0]
        if len(hist) < LOOKBACK + 5:
            continue
        sma = hist.tail(LOOKBACK).mean()
        held = [a for a in assets if a in hist.columns
                and hist[a].loc[d0] > sma[a]]
        if m not in trade_months:
            held = prev  # hold between rebalances, no turnover
        turnover = len(set(held) ^ set(prev)) / max(1, max(len(held), len(prev)))
        cost = turnover * args.cost / 1e4
        fut = px.loc[px.index > d0]
        if held:
            rs = []
            for a in held:
                try:
                    # monthly marked-to-market always; stride only gates turnover
                    r = fut[a].iloc[:21]
                    rs.append(float(np.log(r.iloc[-1] / px.loc[d0, a])))
                except Exception:  # noqa: BLE001
                    continue
            gross = float(np.mean(rs)) if rs else 0.0
        else:
            gross = 0.0
        rows.append({"date": d0, "n": len(held), "gross": gross,
                     "cost": cost, "net": gross - cost, "turnover": turnover})
        prev = held
    b = pd.DataFrame(rows)
    b["equity"] = np.exp(b["net"].cumsum())
    # SPY buy-hold over same window
    spy = px["SPY"].loc[px.index >= b["date"].iloc[0]]
    bh = float(spy.iloc[-1] / spy.iloc[0])
    yrs = (b["date"].max() - b["date"].min()).days / 365.25
    mid = len(b) // 2
    print(f"months: {len(b)} mean_n: {b.n.mean():.1f}")
    for tag, sl in (("tune", slice(None, mid)), ("hold", slice(mid, None))):
        part = b.iloc[sl].reset_index(drop=True).copy()
        part["equity"] = np.exp(part["net"].cumsum())  # reset: slices share a running curve
        s = ev.summarize(part, label=f"maat/{tag}")
        print(tag, {k: s[k] for k in ("CAGR", "maxDD", "Sharpe_m", "exposure")})
    print(f"SPY buy-hold CAGR same window: {bh**(1/yrs)-1:.4f}")
    print(f"strategy CAGR full: {float(b.equity.iloc[-1])**(1/yrs)-1:.4f} "
          f"maxDD: {ev.max_drawdown(b.equity):.3f}")


if __name__ == "__main__":
    main()
