"""Ensemble: 50% ML-wide + 50% textbook momentum. Zero new parameters.

Pre-committed bar on tune-half (sc_full folds 0-11):
  pass = Sharpe_mean > 0.41 AND maxDD > -0.31 (momentum-alone numbers).
Fail means the ML adds nothing to momentum. Holdout once only on pass.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from predictor import backtest as bt
from predictor import evaluate as ev


def load_full_px(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for tk in tickers:
        p = Path("data/raw") / f"{tk}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date")
        d["ticker"] = tk.upper()
        frames.append(d[["Date", "ticker", "Close"]])
    return (pd.concat(frames, ignore_index=True)
            .set_index(["ticker", "Date"])["Close"].unstack("ticker")
            .sort_index())


def mom_monthly_net(px: pd.DataFrame, months, cost_bps: float = 25.0) -> pd.Series:
    nets, prev = {}, []
    for m in months:
        d0 = px.index[pd.to_datetime(px.index.to_series()).dt.to_period("M") == m].max()
        hist = px.loc[px.index <= d0]
        if len(hist) < 278:
            continue
        mom = (hist.iloc[-22] / hist.iloc[-273] - 1).dropna().sort_values(
            ascending=False)
        if len(mom) < 10:
            continue
        cut = mom.quantile(0.90)
        picks = mom[mom >= cut].index.tolist()[: max(5, int(len(mom) * 0.1) + 1)]
        turnover = len(set(picks) ^ set(prev)) / max(1, max(len(picks), len(prev)))
        cost = turnover * cost_bps / 1e4
        fut = px.loc[px.index > d0]
        rets = []
        for tk in picks:
            try:
                r = fut[tk].iloc[:21]
                # simple return, to match run_backtest_risk's "ml" leg below
                # (the ledger rewrite moved that to simple returns; mixing a
                # log-return momentum leg with a simple-return ML leg and
                # then averaging the two would silently mismatch units).
                rets.append(float(r.iloc[-1] / px.loc[d0, tk] - 1))
            except Exception:  # noqa: BLE001
                continue
        nets[d0] = float(np.mean(rets)) - cost if rets else 0.0
        prev = picks
    return pd.Series(nets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/sc_full")
    ap.add_argument("--split", default="tune")
    ap.add_argument("--cost", type=float, default=25.0)
    args = ap.parse_args()
    files = sorted(glob.glob(str(Path(args.cache) / "fold*.csv")),
                   key=lambda f: int("".join(c for c in Path(f).stem if c.isdigit())))
    if args.split == "tune":
        files = files[:len(files) // 2]
    elif args.split == "holdout":
        files = files[len(files) // 2:]
    print(f"folds: {len(files)} [{args.split}]")
    from predictor.universe import SMALLCAP

    full_px = load_full_px([t.upper() for t in SMALLCAP] + ["IWM"])
    full_px = full_px[[c for c in full_px.columns if c != "IWM"]]
    all_nets = {"ml": [], "mom": [], "ens": []}
    for fp in files:
        t = pd.read_csv(fp, parse_dates=["Date"])
        p = t["proba"].values
        b = bt.run_backtest_risk(t, p, cost_bps=args.cost, top_n=20, min_names=1)
        ml = b.set_index("date")["net"]
        ml.index = pd.to_datetime(ml.index)
        months = b["date"]
        mom = mom_monthly_net(full_px, pd.to_datetime(months).dt.to_period("M"),
                              cost_bps=args.cost)
        mom.index = pd.to_datetime(mom.index)
        common = ml.index.intersection(mom.index)
        ml, mom = ml.loc[common], mom.loc[common]
        ens = (ml + mom) / 2
        all_nets["ml"].append(ml)
        all_nets["mom"].append(mom)
        all_nets["ens"].append(ens)
    for name, series in all_nets.items():
        m = pd.concat(series).sort_index()
        eq = (1 + m).cumprod()
        sh = float(m.mean() / (m.std() + 1e-12) * np.sqrt(12))
        yrs = len(eq) / 12
        print(f"{name}: CAGR {float(eq.iloc[-1]**(1/yrs)-1):+.4f} "
              f"maxDD {ev.max_drawdown(eq):+.3f} Sharpe {sh:+.3f}")


if __name__ == "__main__":
    main()
