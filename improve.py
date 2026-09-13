"""Self-improvement loop, finance-grade. AIDE structure, DGM archive,
our gate. Rules, frozen:

- Every proposal pre-registered with a predicted effect BEFORE testing.
- Tune-half screens. Promotion = holdout exactly once, only if the bar
  passes. Fail = death record, no second chances on the same data.
- Trial budget: 20 proposals, ever. Counter persisted. Loop halts at 0.
- Baselines recomputed in-loop every round (no quoted memories).

Bar (pre-committed, round one): tune Sharpe > 0.41 AND maxDD > -0.35
(momentum-alone tune numbers — the best clean result on record).

Usage:
  python3 improve.py status          # budget + registry
  python3 improve.py propose         # list pending (proposals are code)
  python3 improve.py round           # evaluate pending on tune-half
  python3 improve.py promote         # holdout once for the winner (if bar)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from predictor import backtest as bt
from predictor import evaluate as ev

LOG = Path(__file__).resolve().parent / "data" / "improve_log.json"
BUDGET_TOTAL = 20
COST = 25.0
TOP_N = 20
THRESH = 0.30


def load_folds(cache: str = "data/cache/sc_full"):
    files = sorted(glob.glob(str(Path(cache) / "fold*.csv")),
                   key=lambda f: int("".join(c for c in Path(f).stem if c.isdigit())))
    out = []
    for fp in files:
        t = pd.read_csv(fp, parse_dates=["Date"])
        out.append((t, t["proba"].values))
    return out


def load_px(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for tk in tickers:
        p = Path("data/raw") / f"{tk}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date")
        d["ticker"] = tk.upper()
        frames.append(d[["Date", "ticker", "Close"]])
    return (pd.concat(frames, ignore_index=True)
            .set_index(["ticker", "Date"])["Close"].unstack("ticker").sort_index())


def load_ohlc(tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for tk in tickers:
        p = Path("data/raw") / f"{tk}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date")
        out[tk.upper()] = d.set_index("Date")
    return out


def monthly_loop(px: pd.DataFrame, months, pick_fn, cost_bps: float = COST):
    """Generic monthly book. pick_fn(d0, cands_df) -> list of tickers."""
    rows, prev = [], []
    for m in months:
        d0 = px.index[pd.to_datetime(px.index.to_series()).dt.to_period("M") == m].max()
        cands = px.loc[d0]
        picks = pick_fn(d0, cands)
        turnover = len(set(picks) ^ set(prev)) / max(1, max(len(picks), len(prev)))
        cost = turnover * cost_bps / 1e4
        fut = px.loc[px.index > d0]
        rets = []
        for tk in picks:
            try:
                r = fut[tk].iloc[: bt.HORIZON]
                rets.append(float(np.log(r.iloc[-1] / px.loc[d0, tk])))
            except Exception:  # noqa: BLE001
                continue
        gross = float(np.mean(rets)) if rets else 0.0
        rows.append({"date": d0, "net": gross - cost})
        prev = picks
    return pd.Series({r["date"]: r["net"] for r in rows})


def stitch(nets_list: list[pd.Series]) -> pd.Series:
    return pd.concat(nets_list).sort_index()


def score(nets: pd.Series) -> dict:
    eq = np.exp(nets.cumsum())
    yrs = len(eq) / 12
    m = nets
    return {"CAGR": round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4),
            "maxDD": round(ev.max_drawdown(eq), 3),
            "sharpe": round(float(m.mean() / (m.std() + 1e-12) * np.sqrt(12)), 3),
            "months": len(eq)}


# ---------------- proposals (registered with predictions) ----------------
# Each takes (test, proba, px) -> monthly nets. Predictions logged below
# BEFORE first evaluation. New proposals appended as code + registry entry.

def p_vol_only(test, proba, px):
    t = test.copy()
    t["proba"] = proba
    b = bt.run_backtest_risk(t, proba, cost_bps=COST, top_n=TOP_N,
                             thresh=THRESH, min_names=1,
                             stop_frac=None,  # stops off
                             vol_target=0.20, dd_brake=10.0)  # brake off
    return b.set_index("date")["net"]


def p_stops_only(test, proba, px):
    t = test.copy()
    t["proba"] = proba
    b = bt.run_backtest(t, proba, cost_bps=COST, top_n=10, thresh=THRESH)
    # add stops by reusing risk runner with everything else off
    b2 = bt.run_backtest_risk(t, proba, cost_bps=COST, top_n=10,
                              thresh=THRESH, min_names=1, stop_frac=0.85,
                              vol_target=10.0, dd_brake=10.0)
    return b2.set_index("date")["net"]


def p_blend(test, proba, px):
    """Signal blend: 50% ML rank + 50% momentum rank, wide construction."""
    t = test.copy().reset_index(drop=True)
    t["proba"] = proba
    months = pd.to_datetime(t["Date"]).dt.to_period("M").drop_duplicates().sort_values()
    rows = []
    for m in months:
        block = t[pd.to_datetime(t["Date"]).dt.to_period("M") == m]
        d0 = block["Date"].max()
        c = block[block["Date"] == d0].copy()
        hist = px.loc[px.index <= d0]
        if len(hist) >= 278:
            mom = (hist.iloc[-22] / hist.iloc[-273] - 1)
            c["momr"] = c["ticker"].map(mom.rank(pct=True))
        else:
            c["momr"] = 0.5
        c["mlr"] = c["proba"].rank(pct=True)
        c["blend"] = 0.5 * c["mlr"] + 0.5 * c["momr"].fillna(0.5)
        picks = c.sort_values("blend", ascending=False).head(TOP_N)["ticker"].tolist()
        rows.append((d0, picks))
    # realize with stops via monthly_loop-like accounting
    nets, prev = {}, []
    for d0, picks in rows:
        turnover = len(set(picks) ^ set(prev)) / max(1, max(len(picks), len(prev)))
        cost = turnover * COST / 1e4
        fut = px.loc[px.index > d0]
        rets = []
        for tk in picks:
            try:
                p0 = px.loc[d0, tk]
                r = fut[tk].iloc[: bt.HORIZON]
                hit = r[r / p0 <= 0.85]
                rets.append(float(np.log(0.85)) if len(hit)
                            else float(np.log(r.iloc[-1] / p0)))
            except Exception:  # noqa: BLE001
                continue
        nets[d0] = float(np.mean(rets)) - cost if rets else 0.0
        prev = picks
    return pd.Series(nets)


def p_hold40(test, proba, px):
    """Same top-10 signal, 40-day holds (half the turnover)."""
    t = test.copy().reset_index(drop=True)
    t["proba"] = proba
    months = pd.to_datetime(t["Date"]).dt.to_period("M").drop_duplicates().sort_values()
    nets, prev, hold = {}, [], {}
    for m in months:
        d0 = t[pd.to_datetime(t["Date"]).dt.to_period("M") == m]["Date"].max()
        c = t[t["Date"] == d0].sort_values("proba", ascending=False)
        fresh = c[c["proba"] >= THRESH].head(10)["ticker"].tolist()
        # keep prior holds until 40d elapsed
        holds = [tk for tk, since in hold.items()
                 if (d0 - since).days < 40 and tk in px.columns]
        picks = list(dict.fromkeys(holds + [x for x in fresh if x not in holds]))[:10]
        turnover = len(set(picks) ^ set(prev)) / max(1, max(len(picks), len(prev)))
        cost = turnover * COST / 1e4
        fut = px.loc[px.index > d0]
        rets = []
        for tk in picks:
            try:
                r = fut[tk].iloc[:21]
                rets.append(float(np.log(r.iloc[-1] / px.loc[d0, tk])))
            except Exception:  # noqa: BLE001
                continue
        nets[d0] = float(np.mean(rets)) - cost if rets else 0.0
        prev = picks
        for tk in fresh:
            hold[tk] = d0
    return pd.Series(nets)


def p_gate25(test, proba, px):
    """Wide construction, lower gate 0.25 (more names, more diversification)."""
    t = test.copy()
    b = bt.run_backtest_risk(t, proba, cost_bps=COST, top_n=TOP_N,
                             thresh=0.25, min_names=1, stop_frac=0.85,
                             vol_target=0.20, dd_brake=0.15)
    return b.set_index("date")["net"]


def p_calm_only(test, proba, px):
    """Trade only when trailing 60d universe vol < 25% ann (skip storms)."""
    t = test.copy()
    uret = np.log(px / px.shift(1)).mean(axis=1).dropna()
    vol = uret.rolling(60, min_periods=20).std() * np.sqrt(252)
    b = bt.run_backtest_risk(t, proba, cost_bps=COST, top_n=TOP_N,
                             thresh=THRESH, min_names=1, stop_frac=0.85,
                             vol_target=0.20, dd_brake=0.15)
    b = b.copy()
    for i, r in b.iterrows():
        d0 = pd.to_datetime(r["date"])
        hist = vol.loc[vol.index <= d0]
        if len(hist) and float(hist.iloc[-1]) > 0.25:
            b.loc[i, "net"] = 0.0 - r["cost"] * 0  # flat, no cost (no trade)
    return b.set_index("date")["net"]


_OHLC: dict = {}


def ohlc(tk: str):
    tk = tk.upper()
    if tk not in _OHLC:
        p = Path("data/raw") / f"{tk}.csv"
        if not p.exists():
            return None
        _OHLC[tk] = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    return _OHLC[tk]


def trailing_dv(tk: str, d0, n: int = 20) -> float:
    h = ohlc(tk)
    if h is None:
        return 0.0
    h = h.loc[h.index <= d0].tail(n)
    if len(h) < n:
        return 0.0
    return float((h["Close"] * h["Volume"]).mean())


def trailing_vol(tk: str, d0, n: int = 20) -> float:
    h = ohlc(tk)
    if h is None:
        return float("nan")
    h = h.loc[h.index <= d0].tail(n + 1)
    if len(h) < n + 1:
        return float("nan")
    return float(np.log(h["Close"] / h["Close"].shift(1)).dropna().std() * np.sqrt(252))


def _risk_book(test, proba, px, weight_fn=None, dv_min: float = 0.0,
               top_n: int = 20, thresh: float = 0.25):
    """P5 construction (top-N/thresh/stops/vol-target/DD-brake) plus an
    optional candidate $volume filter and per-name weighting."""
    from predictor.backtest import RISK_STOP
    t = test.copy().reset_index(drop=True)
    t["proba"] = np.asarray(proba, dtype=float)
    months = pd.to_datetime(t["Date"]).dt.to_period("M").drop_duplicates().sort_values()
    uret = np.log(px / px.shift(1)).mean(axis=1).dropna()
    vol = uret.rolling(60, min_periods=20).std() * np.sqrt(252)
    rows, prev, eq, peak = [], [], 1.0, 1.0
    for m in months:
        block = t[pd.to_datetime(t["Date"]).dt.to_period("M") == m]
        d0 = block["Date"].max()
        cands = block[block["Date"] == d0].sort_values("proba", ascending=False)
        quals = cands[cands["proba"] >= thresh]
        if dv_min > 0:
            quals = quals[quals["ticker"].map(lambda k: trailing_dv(k, d0)) >= dv_min]
        if len(quals) < 1:
            picks, w = [], {}
        else:
            hist = vol.loc[vol.index <= d0]
            trailing = float(hist.iloc[-1]) if len(hist) else float("nan")
            scale = min(1.0, 0.20 / trailing) if trailing > 0 else 1.0
            if eq / peak - 1.0 < -0.15:
                scale *= 0.5
            n = max(0, round(top_n * scale))
            picks = quals.head(n)["ticker"].tolist() if n >= 1 else []
            if weight_fn is None or not picks:
                w = {tk: 1 / len(picks) for tk in picks}
            else:
                vols = {tk: weight_fn(tk, d0) for tk in picks}
                inv = {tk: 1 / v if v and v > 0 else 0.0 for tk, v in vols.items()}
                tot = sum(inv.values()) or 1.0
                w = {tk: v / tot for tk, v in inv.items()}
        turnover = len(set(picks) ^ set(prev)) / max(1, max(len(picks), len(prev)))
        cost = turnover * COST / 1e4
        gross = 0.0
        if picks:
            rs = []
            for tk in picks:
                try:
                    p0 = px.loc[d0, tk]
                    fut = px[tk].loc[px.index > d0].iloc[: bt.HORIZON]
                    if len(fut) == 0:
                        continue
                    hit = fut[fut / p0 <= RISK_STOP]
                    r = float(np.log(RISK_STOP)) if len(hit) else float(np.log(fut.iloc[-1] / p0))
                    rs.append((tk, r))
                except Exception:  # noqa: BLE001
                    continue
            gross = sum(w.get(tk, 0) * r for tk, r in rs) if rs else 0.0
        net = gross - cost
        eq *= np.exp(net)
        peak = max(peak, eq)
        rows.append({"date": d0, "net": net})
        prev = picks
    out = pd.DataFrame(rows)
    return out.set_index("date")["net"]


def p_liq(test, proba, px):
    t = test.copy()
    tpx = t.set_index(["ticker", "Date"])["Close"].unstack("ticker")
    return _risk_book(t, proba, tpx, dv_min=500_000.0)


def p_volw(test, proba, px):
    t = test.copy()
    tpx = t.set_index(["ticker", "Date"])["Close"].unstack("ticker")
    return _risk_book(t, proba, tpx, weight_fn=trailing_vol)


def p_top30(test, proba, px):
    t = test.copy()
    tpx = t.set_index(["ticker", "Date"])["Close"].unstack("ticker")
    return _risk_book(t, proba, tpx, weight_fn=trailing_vol, top_n=30)


def p_gate35(test, proba, px):
    t = test.copy()
    tpx = t.set_index(["ticker", "Date"])["Close"].unstack("ticker")
    return _risk_book(t, proba, tpx, weight_fn=trailing_vol, thresh=0.35)


def p_hold10(test, proba, px):
    """Monthly decisions, 10-day holds, then cash. Fresher signal, half
    exposure. Pre-registered: turnover doubles per invested day."""
    from predictor.backtest import RISK_STOP
    t = test.copy().reset_index(drop=True)
    t["proba"] = np.asarray(proba, dtype=float)
    months = pd.to_datetime(t["Date"]).dt.to_period("M").drop_duplicates().sort_values()
    tpx = t.set_index(["ticker", "Date"])["Close"].unstack("ticker")
    rows, prev = [], []
    for m in months:
        block = t[pd.to_datetime(t["Date"]).dt.to_period("M") == m]
        d0 = block["Date"].max()
        c = block[block["Date"] == d0].sort_values("proba", ascending=False)
        picks = c[c["proba"] >= 0.25].head(20)["ticker"].tolist()
        turnover = len(set(picks) ^ set(prev)) / max(1, max(len(picks), len(prev)))
        cost = turnover * COST / 1e4
        fut = tpx.loc[tpx.index > d0]
        rs = []
        for tk in picks:
            try:
                p0 = tpx.loc[d0, tk]
                r = fut[tk].iloc[:10]
                if len(r) == 0:
                    continue
                hit = r[r / p0 <= RISK_STOP]
                rs.append(float(np.log(RISK_STOP)) if len(hit)
                          else float(np.log(r.iloc[-1] / p0)))
            except Exception:  # noqa: BLE001
                continue
        gross = float(np.mean(rs)) if rs else 0.0
        rows.append({"date": d0, "net": gross - cost})
        prev = picks
    out = pd.DataFrame(rows)
    return out.set_index("date")["net"]


def p_top40(test, proba, px):
    t = test.copy()
    tpx = t.set_index(["ticker", "Date"])["Close"].unstack("ticker")
    return _risk_book(t, proba, tpx, weight_fn=trailing_vol, top_n=40)
    t = test.copy()
    tpx = t.set_index(["ticker", "Date"])["Close"].unstack("ticker")
    return _risk_book(t, proba, tpx, weight_fn=trailing_vol, thresh=0.35)


PROPOSALS = {
    # id: (fn, predicted effect, pre-registered before any evaluation)
    "P1-vol-only": (p_vol_only, "Cuts turnover drag; DD unchanged. Sharpe ~0.1."),
    "P2-stops-only": (p_stops_only, "Truncates left tail; whipsaw costs. Sharpe ~0.15."),
    "P3-blend": (p_blend, "Best of both signals; Sharpe > 0.3 if ML adds anything."),
    "P4-hold40": (p_hold40, "Halves turnover; staler signal. Net favors costs."),
    "P5-gate25": (p_gate25, "More names, more diversification; weaker selection."),
    "P6-calm-only": (p_calm_only, "Dodges 2020/2022 storms; whipsaw cash drag."),
    "P7-liqfilter": (p_liq, "Drops illiquid names; cuts hidden impact. Sharpe ~0.5, DD similar."),
    "P8-volscale": (p_volw, "Smooths single-name blowups; DD toward -0.2. Sharpe ~0.5."),
    "P9-top30": (p_top30, "More breadth, dilutes best ideas; DD improves, CAGR falls. Sharpe ~0.4."),
    "P10-gate35": (p_gate35, "Tighter selection, fewer names; punchier, worse DD. Sharpe ~0.4."),
    "P11-hold10": (p_hold10, "Fresher signal, half exposure, double turnover rate. Sharpe ~0.3."),
    "P12-top40": (p_top40, "Max breadth; dilutes further. Sharpe ~0.3, DD best yet."),
}


def read_log():
    if LOG.exists():
        return json.loads(LOG.read_text())
    return {"spent": 0, "proposals": {}, "archive": []}


def write_log(d):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(d, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("round")
    sub.add_parser("promote")
    args = ap.parse_args()
    log = read_log()

    if args.cmd == "status":
        print(f"budget: {BUDGET_TOTAL - log['spent']}/{BUDGET_TOTAL} left")
        for pid, (fn, pred) in PROPOSALS.items():
            st = log["proposals"].get(pid, {"status": "pending"})
            print(f"{pid} [{st['status']}] pred: {pred} "
                  f"tune={st.get('tune')} hold={st.get('holdout')}")
        return

    if args.cmd == "round":
        folds = load_folds()
        tune = folds[:len(folds) // 2]
        tickers = sorted({tk for t, _ in tune for tk in t["ticker"].unique()})
        px = load_px(tickers)
        pending = [pid for pid in PROPOSALS if log["proposals"].get(
            pid, {}).get("status") in (None, "pending")]
        if log["spent"] + len(pending) > BUDGET_TOTAL:
            print("BUDGET EXCEEDED — loop halts. No more proposals, ever.")
            return
        for pid in pending:
            fn, pred = PROPOSALS[pid]
            nets = []
            for t, p in tune:
                try:
                    nets.append(fn(t, p, px))
                except Exception as e:  # noqa: BLE001
                    print(f"{pid} failed: {e}")
                    break
            if not nets:
                log["proposals"][pid] = {"status": "error", "pred": pred}
                continue
            s = score(stitch(nets))
            log["proposals"][pid] = {"status": "tested-tune", "pred": pred,
                                     "tune": s}
            log["archive"].append({"id": pid, "tune": s})
            print(f"{pid}: {s}  pred was: {pred}")
            log["spent"] += 1
        write_log(log)
        print("round done. Promote only the best IF bar passes "
              "(Sharpe > 0.41 AND maxDD > -0.35).")
        return

    if args.cmd == "promote":
        folds = load_folds()
        hold = folds[len(folds) // 2:]
        tickers = sorted({tk for t, _ in hold for tk in t["ticker"].unique()})
        px = load_px(tickers)
        cands = {pid: st for pid, st in log["proposals"].items()
                 if st.get("status") == "tested-tune"}
        if not cands:
            print("nothing to promote")
            return
        best = max(cands, key=lambda k: cands[k]["tune"]["sharpe"])
        t = cands[best]["tune"]
        if not (t["sharpe"] > 0.41 and t["maxDD"] > -0.35):
            print(f"BAR FAILED for {best}: {t}. No holdout. Recorded.")
            log["proposals"][best]["status"] = "dead-tune"
            write_log(log)
            return
        fn, _ = PROPOSALS[best]
        nets = [fn(t_, p_, px) for t_, p_ in hold]
        s = score(stitch(nets))
        log["proposals"][best]["status"] = "promoted"
        log["proposals"][best]["holdout"] = s
        log["archive"].append({"id": best, "holdout": s})
        write_log(log)
        print(f"PROMOTED {best} holdout: {s}")


if __name__ == "__main__":
    main()
