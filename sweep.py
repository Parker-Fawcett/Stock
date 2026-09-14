"""Cost sweep on cached fold probas — no refits.

Usage:
  python3 sweep.py --cache data/cache/sc_price --costs 0,5,10,25,50
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--costs", default="0,5,10,25,50")
    ap.add_argument("--monkeys", type=int, default=200)
    ap.add_argument("--risk", action="store_true",
                    help="use run_backtest_risk instead of baseline")
    ap.add_argument("--config", default="",
                    help="pre-committed config: base | wide | v2 | ls (overrides --risk)")
    ap.add_argument("--split", default="all",
                    help="tune = first half of folds, holdout = second half")
    args = ap.parse_args()
    costs = [float(c) for c in args.costs.split(",") if c.strip() != ""]
    # Pre-committed configs, chosen BEFORE seeing tune results.
    # base: original 10-name, no stops. wide: 20-name + stops, no breadth
    # gate. v2: 20-name + stops + breadth gate + vol target + DD brake.
    # ls: dollar-neutral long top-10 / short bottom-10 with borrow costs.
    if args.config == "wide":
        def btfn(t, p, cost_bps):
            return bt.run_backtest_risk(t, p, cost_bps=cost_bps,
                                        top_n=20, min_names=1)
    elif args.config == "v2":
        btfn = bt.run_backtest_risk
    elif args.config == "ls":
        btfn = bt.run_backtest_ls
    elif args.config == "base":
        btfn = bt.run_backtest
    else:
        btfn = bt.run_backtest_risk if args.risk else bt.run_backtest
    files = sorted(glob.glob(str(Path(args.cache) / "fold*.csv")),
                   key=lambda f: int("".join(c for c in Path(f).stem
                                             if c.isdigit())))
    if args.split == "tune":
        files = files[:len(files) // 2]
    elif args.split == "holdout":
        files = files[len(files) // 2:]
    print(f"folds cached: {len(files)} [{args.split}::{args.config or 'risk' if args.risk else args.config or 'base'}]")
    tests, probas = [], []
    for fp in files:
        t = pd.read_csv(fp, parse_dates=["Date"])
        probas.append(t["proba"].values)
        tests.append(t.drop(columns=["proba"]))
    rows = []
    for c in costs:
        nets, stats = [], []
        for t, p in zip(tests, probas):
            b = btfn(t, p, cost_bps=c)
            nets.append(b.set_index("date")["net"])
            s = ev.summarize(b)
            stats.append(s)
        equity = (1 + pd.concat(nets).sort_index()).cumprod()
        years = len(equity) / 12
        mk = [float(bt.run_monkeys(t, cost_bps=c)["monkey_mean"].iloc[0])
              for t in tests]
        rows.append({
            "cost_bps": c,
            "equity_end": round(float(equity.iloc[-1]), 3),
            "CAGR": round(float(equity.iloc[-1] ** (1 / years) - 1), 4),
            "maxDD": round(ev.max_drawdown(equity), 3),
            "sharpe_mean": round(float(np.mean([s["Sharpe_m"] for s in stats])), 3),
            "avg_monthly_net": round(float(pd.concat(nets).mean()), 5),
            "fold_CAGR_mean": round(float(np.mean([s["CAGR"] for s in stats])), 4),
            "monkey_mean": round(float(np.mean(mk)), 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
