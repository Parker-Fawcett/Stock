"""End-to-end demo: download -> panel -> walk-forward -> backtest vs monkeys.

Run:
  python3 run.py --tickers AAPL,MSFT,JNJ,XOM,SPY --start 20150101
Free Stooq data; survivorship-biased by construction. Numbers below are
a plumbing check, not a claim.
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
from predictor.features import build_panel, model_cols
from predictor.model_lgbm import fit_fold, walk_forward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="AAPL,MSFT,JNJ,PG,XOM,CVX,KO,MRK,WMT,IBM")
    ap.add_argument("--market", default="SPY")
    ap.add_argument("--start", default="20150101")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--universe", default="",
                    help="named universe from predictor.universe, e.g. sp100")
    ap.add_argument("--recent", type=int, default=0,
                    help="only evaluate the last K folds (0 = all)")
    ap.add_argument("--no-insider", action="store_true",
                    help="skip Form 4 merge (ablation)")
    ap.add_argument("--save-proba", default="",
                    help="dir to cache per-fold test+proba for cost sweeps")
    args = ap.parse_args()

    if args.universe:
        from predictor import universe as U
        names = getattr(U, args.universe.upper())
        tickers = [t.strip().upper() for t in names]
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    print(f"downloading {len(tickers) + 1} symbols from Yahoo...")
    paths = download_universe(tickers, market=args.market)
    assert_vintage({k: str(v) for k, v in paths.items()})
    price_paths = {k: str(v) for k, v in paths.items()}
    print("building panel...")
    panel = build_panel(price_paths, market=args.market)
    if args.no_insider:
        print("insider merge skipped (--no-insider)")
    else:
        try:
            from predictor.insider import merge_onto_panel

            panel = merge_onto_panel(panel)
            print("insider merge:",
                  [c for c in model_cols(panel) if c.startswith("ins_")])
        except Exception as e:  # noqa: BLE001 - insider optional
            print(f"insider merge skipped: {e}")
    print(f"panel rows={len(panel)} dates={panel['Date'].nunique()} "
          f"tickers={panel['ticker'].nunique()}")
    folds = walk_forward(panel)
    if args.recent and len(folds) > args.recent:
        folds = folds[-args.recent:]
    print(f"folds={len(folds)}")
    if not folds:
        print("not enough history for walk-forward; add tickers/years")
        return
    aucs, summaries = [], []
    cols = model_cols(panel)
    cache = None
    if args.save_proba:
        cache = Path(args.save_proba)
        cache.mkdir(parents=True, exist_ok=True)
    for fi, f in enumerate(folds):
        for seed in range(args.seeds):
            model = fit_fold(f.train, f.val, seed=seed)
            p = model.predict_proba(f.test[cols].values)[:, 1]
            aucs.append(ev.auc_score(f.test["label"].values, p))
            if seed == 0:  # backtest first seed only per fold (speed)
                if cache is not None:
                    t = f.test[["Date", "ticker", "Close", "label"]].copy()
                    t["proba"] = p
                    t.to_csv(cache / f"fold{fi}.csv", index=False)
                b = bt.run_backtest(f.test, p)
                s = ev.summarize(b, label=f"fold{fi}")
                s["AUC"] = aucs[-1]
                summaries.append(s)
                mk = bt.run_monkeys(f.test)
                s["monkey_mean"] = float(mk["monkey_mean"].iloc[0])
    print(f"mean AUC over {len(aucs)} fits: {np.nanmean(aucs):.3f}")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
