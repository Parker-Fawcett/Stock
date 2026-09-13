"""Classic cross-sectional momentum: does the ML add anything?

Rule, frozen before first run: each month-end, rank by 12m-minus-1m
total return, hold top decile (min 5 names) for 1 month, equal weight.
Costs on turnover. No fits, no parameters to tune — the point is to
grade the ML against the textbook.

Universe/market passed in; tune/holdout split reported, nothing selected.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from predictor import backtest as bt
from predictor import evaluate as ev
from predictor.data import download_universe, assert_vintage


def momentum_panel(tickers: list[str], market: str) -> pd.DataFrame:
    paths = download_universe(tickers, market=market)
    assert_vintage({k: str(v) for k, v in paths.items()})
    frames = []
    for tk, p in paths.items():
        try:
            d = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date")
            d["ticker"] = tk.upper()
            frames.append(d[["Date", "ticker", "Close"]])
        except Exception as e:  # noqa: BLE001
            print(f"skip {tk}: {e}")
    return pd.concat(frames, ignore_index=True)


def run_momentum(panel: pd.DataFrame, cost_bps: float = 25.0,
                 lookback: int = 252, skip: int = 21,
                 decile: float = 0.10) -> pd.DataFrame:
    px = panel.set_index(["ticker", "Date"])["Close"].unstack("ticker")
    universe = [c for c in px.columns if c != "MKT"]
    months = pd.to_datetime(px.index.to_series()).dt.to_period("M").drop_duplicates()
    months = months.sort_values()
    rows, prev = [], []
    for m in months[13:]:  # need 12m formation
        d0 = px.index[pd.to_datetime(px.index.to_series()).dt.to_period("M") == m].max()
        hist = px.loc[px.index <= d0]
        if len(hist) < lookback + 5:
            continue
        base = hist.iloc[-lookback]
        ref = hist.iloc[-skip]
        mom = (ref / base - 1).dropna()
        mom = mom[[c for c in mom.index if c in universe]]
        if len(mom) < 10:
            continue
        cut = mom.quantile(1 - decile)
        picks = mom[mom >= cut].index.tolist()[: max(5, int(len(mom) * decile) + 1)]
        turnover = len(set(picks) ^ set(prev)) / max(1, max(len(picks), len(prev)))
        cost = turnover * cost_bps / 1e4
        fut = px.loc[px.index > d0]
        rets = []
        for tk in picks:
            try:
                r = fut[tk].iloc[:21]
                rets.append(float(np.log(r.iloc[-1] / px.loc[d0, tk])))
            except Exception:  # noqa: BLE001
                continue
        gross = float(np.mean(rets)) if rets else 0.0
        rows.append({"date": d0, "n": len(picks), "gross": gross,
                     "cost": cost, "net": gross - cost, "turnover": turnover})
        prev = picks
    out = pd.DataFrame(rows)
    out["equity"] = np.exp(out["net"].cumsum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="smallcap")
    ap.add_argument("--market", default="IWM")
    ap.add_argument("--cost", type=float, default=25.0)
    args = ap.parse_args()
    from predictor import universe as U

    tickers = [t.strip().upper() for t in getattr(U, args.universe.upper())]
    panel = momentum_panel(tickers, args.market)
    # normalize market col name
    panel.loc[panel.ticker == args.market.upper(), "ticker"] = "MKT"
    b = run_momentum(panel, cost_bps=args.cost)
    mid = len(b) // 2
    for tag, sl in (("tune", slice(None, mid)), ("hold", slice(mid, None))):
        part = b.iloc[sl].reset_index(drop=True).copy()
        part["equity"] = np.exp(part["net"].cumsum())  # reset: slices share a running curve
        s = ev.summarize(part, label=f"mom/{tag}")
        print(tag, {k: s[k] for k in ("CAGR", "maxDD", "Sharpe_m", "exposure")})
    print("months:", len(b), "mean_n:", round(b.n.mean(), 1))


if __name__ == "__main__":
    main()
