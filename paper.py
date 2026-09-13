"""Forward paper trading. FROZEN pipeline — no tuning here, ever.

Every run trains once on the trailing window (3y train / 6mo val,
labels need 20d forward so train ends 20 trading days ago — purge is
automatic), predicts today, and appends picks to data/paper/log.csv.

The market grades it from there. Rules:
- Run after month-end close. Trade next open (fills worse than the
  backtest's month-end close assumption — noted, conservative).
- Wide config, seed 0 only: top-20, proba >= 0.30, stops at 0.85x ref.
- Grade command scores picks once 20 trading days have passed, vs IWM.

Usage:
  python3 paper.py predict --universe smallcap --market IWM
  python3 paper.py grade
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from predictor.backtest import COST_BPS
from predictor.data import download_universe, assert_vintage
from predictor.features import HORIZON, build_latest, build_panel, model_cols
from predictor.model_lgbm import fit_fold

MODEL_VERSION = "wide-seed0-v1"  # frozen. Bump only with a dated reason.
PAPER_DIR = Path(__file__).resolve().parent / "data" / "paper"
PAPER_DIR.mkdir(parents=True, exist_ok=True)
LOG = PAPER_DIR / "log.csv"


def trailing_split(panel: pd.DataFrame):
    """Train/val split on the labeled panel only. Live inference uses
    build_latest() separately -- panel's own last date is always HORIZON
    trading days stale (its rows need a known forward return)."""
    dates = np.sort(panel["Date"].unique())
    v1 = dates[-1] - pd.to_timedelta(6 * 30, unit="D")
    r1 = v1 - pd.to_timedelta(3 * 365, unit="D")
    embargo = pd.to_timedelta(HORIZON, unit="D")
    train = panel[(panel["Date"] >= r1) & (panel["Date"] < v1)]
    # PURGE by the actual trading-day label window, not a calendar guess.
    train = train[train["label_end"] < v1 - embargo]
    val = panel[panel["Date"] >= v1]
    return train, val


def cmd_predict(args) -> None:
    if args.universe:
        from predictor import universe as U

        tickers = [t.strip().upper() for t in getattr(U, args.universe.upper())]
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    paths = download_universe(tickers, market=args.market,
                                refresh=getattr(args, "refresh", False))
    assert_vintage({k: str(v) for k, v in paths.items()})
    str_paths = {k: str(v) for k, v in paths.items()}
    panel = build_panel(str_paths, market=args.market)
    latest, asof = build_latest(str_paths, market=args.market)
    cols = model_cols(panel)
    train, val = trailing_split(panel)
    print(f"asof={asof.date()} train={len(train)} val={len(val)} "
          f"universe={latest['ticker'].nunique()}")
    if len(train) < 500 or len(val) < 100:
        print("not enough trailing history")
        return
    model = fit_fold(train, val, seed=0)
    latest = latest.copy()
    latest["proba"] = model.predict_proba(latest[cols].values)[:, 1]
    cands = latest.sort_values("proba", ascending=False)
    quals = cands[cands["proba"] >= 0.30]
    n = min(20, len(quals))
    picks = quals.head(n) if n >= 1 else quals.head(0)
    rows = []
    for _, r in picks.iterrows():
        rows.append({"date": asof.date().isoformat(), "ticker": r["ticker"],
                     "proba": round(float(r["proba"]), 4),
                     "ref_close": round(float(r["Close"]), 2),
                     "stop": round(float(r["Close"]) * 0.85, 2),
                     "model": MODEL_VERSION, "universe": args.universe or "custom",
                     "market": args.market})
    out = pd.DataFrame(rows)
    if LOG.exists():
        old = pd.read_csv(LOG)
        out = pd.concat([old, out], ignore_index=True).drop_duplicates(
            subset=["date", "ticker", "model"])
    out.to_csv(LOG, index=False)
    print(f"picks ({len(out[out.date == asof.date().isoformat()])} new):")
    print(out[out.date == asof.date().isoformat()].to_string(index=False)
          if len(rows) else "CASH — no qualifiers")
    # Textbook momentum sleeve, same date, separate model tag.
    # Frozen rule: top decile 12m-1m, min 5 names, equal weight.
    mom_rows = []
    try:
        pxm = {}
        for tk in tickers:
            if tk == args.market.upper():
                continue
            d = pd.read_csv(
                paths[tk], parse_dates=["Date"]).sort_values("Date")
            pxm[tk] = d.set_index("Date")["Close"]
        pxm = pd.DataFrame(pxm).sort_index()
        # Same frozen 12m-1m spec as mom_run.py (lookback=252, skip=21) --
        # this used to be -273/-22, a different, undocumented window.
        if len(pxm) >= 252:
            mom = pxm.iloc[-21] / pxm.iloc[-252] - 1
            mom = mom.dropna().sort_values(ascending=False)
            cut = mom.quantile(0.90)
            picks_m = mom[mom >= cut].index.tolist()
            if len(picks_m) >= 5:
                for tk in picks_m:
                    px = float(pxm[tk].iloc[-1])
                    mom_rows.append(
                        {"date": asof.date().isoformat(), "ticker": tk,
                         "proba": round(float(mom[tk]), 4), "ref_close": px,
                         "stop": round(px * 0.85, 2), "model": "mom12-1-v1",
                         "universe": args.universe or "custom",
                         "market": args.market})
                mout = pd.concat([pd.read_csv(LOG), pd.DataFrame(mom_rows)],
                                 ignore_index=True).drop_duplicates(
                    subset=["date", "ticker", "model"])
                mout.to_csv(LOG, index=False)
                print(f"mom picks ({len(mom_rows)}): "
                      + ",".join(mom_rows and [r["ticker"] for r in mom_rows]))
            else:
                print("mom sleeve: CASH (<5 qualifiers)")
    except Exception as e:  # noqa: BLE001
        print(f"mom sleeve skipped: {e}")


def cmd_grade(args) -> None:
    if not LOG.exists():
        print("no picks logged yet")
        return
    log = pd.read_csv(LOG, parse_dates=["date"])
    paths = download_universe(
        sorted(log["ticker"].unique().tolist()) + [args.market])
    px_all = {}
    for tk, p in paths.items():
        d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
        px_all[tk.upper()] = d
    recs = []
    for _, r in log.iterrows():
        tk, d0 = r["ticker"], r["date"]
        try:
            s = px_all[tk].loc[px_all[tk].index > d0].iloc[:HORIZON]
            m = px_all[args.market].loc[px_all[args.market].index > d0].iloc[:HORIZON]
            if len(s) < HORIZON or len(m) < HORIZON:
                continue  # not matured
            entry = float(r["ref_close"])
            # gap-aware stop, entry consistently at the logged decision-day
            # close (matches predictor/backtest.py's ledger): the old code
            # credited exactly the stop threshold even on a worse gap, and
            # measured the no-stop case from a *different* entry (next
            # day's close) than the stop case (ref_close) -- two prices for
            # one trade. Simple returns throughout, one round-trip cost.
            hit = s[s <= r["stop"]]
            ret = float(hit.iloc[0] / entry - 1) if len(hit) else float(s.iloc[-1] / entry - 1)
            ret -= 2 * COST_BPS / 1e4
            mret = float(m.iloc[-1] / m.iloc[0] - 1)
            recs.append({"date": d0.date().isoformat(), "ticker": tk,
                         "model": r["model"],
                         "ret": round(ret, 4), "iwm": round(mret, 4),
                         "excess": round(ret - mret, 4)})
        except Exception:  # noqa: BLE001
            continue
    if not recs:
        print(f"{len(log)} picks logged, none matured (need {HORIZON} trading days)")
        return
    g = pd.DataFrame(recs)
    for model, gm in g.groupby("model"):
        print(f"[{model}] n={len(gm)} hit={((gm.ret > 0).mean()):.2f} "
              f"mean={gm.ret.mean():.4f} excess={gm.excess.mean():.4f}")
    print(g.tail(10).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--tickers", default="")
    p.add_argument("--universe", default="smallcap")
    p.add_argument("--market", default="IWM")
    p.add_argument("--refresh", action="store_true",
                   help="force fresh Yahoo downloads (do this monthly)")
    g = sub.add_parser("grade")
    g.add_argument("--market", default="IWM")
    args = ap.parse_args()
    if args.cmd == "predict":
        cmd_predict(args)
    else:
        cmd_grade(args)


if __name__ == "__main__":
    main()
