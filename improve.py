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
  python3 improve.py round --cache data/cache/sc_full_v2 --mode exploratory
                                      # evaluate pending on tune-half
  python3 improve.py promote --cache data/cache/sc_full_v2
                                      # holdout once for that cache's winner
  python3 improve.py validate --id P15-mc-price-v1
                                      # execute one registered external check
  python3 improve.py replay-promoted --cache data/cache/sc_full
  python3 improve.py replay-promoted --cache data/cache/sc_full_v2
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
        turnover = bt.equal_weight_turnover(prev, picks)
        cost = turnover * cost_bps / 1e4
        fut = px.loc[px.index > d0]
        rets = []
        for tk in picks:
            try:
                r = fut[tk].iloc[: bt.HORIZON]
                rets.append(float(r.iloc[-1] / px.loc[d0, tk] - 1))
            except Exception:  # noqa: BLE001
                continue
        gross = float(np.mean(rets)) if rets else 0.0
        rows.append({"date": d0, "net": gross - cost})
        prev = picks
    return pd.Series({r["date"]: r["net"] for r in rows})


def stitch(nets_list: list[pd.Series]) -> pd.Series:
    return pd.concat(nets_list).sort_index()


def score(nets: pd.Series) -> dict:
    eq = (1 + nets).cumprod()
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
        if len(hist) >= 252:
            mom = (hist.iloc[-21] / hist.iloc[-252] - 1)
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
        turnover = bt.equal_weight_turnover(prev, picks)
        cost = turnover * COST / 1e4
        fut = px.loc[px.index > d0]
        rets = []
        for tk in picks:
            try:
                rets.append(bt.leg_simple(px[tk], d0, bt.HORIZON,
                                          short=False, stop_frac=0.85))
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
        turnover = bt.equal_weight_turnover(prev, picks)
        cost = turnover * COST / 1e4
        fut = px.loc[px.index > d0]
        rets = []
        for tk in picks:
            try:
                r = fut[tk].iloc[:21]
                rets.append(float(r.iloc[-1] / px.loc[d0, tk] - 1))
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
    rows, prev_w, eq, peak = [], {}, 1.0, 1.0
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
            picks = quals.head(top_n)["ticker"].tolist()
            if weight_fn is None or not picks:
                w = {tk: scale / len(picks) for tk in picks}
            else:
                vols = {tk: weight_fn(tk, d0) for tk in picks}
                inv = {tk: 1 / v if v and v > 0 else 0.0 for tk, v in vols.items()}
                tot = sum(inv.values()) or 1.0
                w = {tk: scale * v / tot for tk, v in inv.items()}
        turnover = bt.weight_turnover(prev_w, w)
        cost = turnover * COST / 1e4
        gross = 0.0
        if picks:
            rs = []
            for tk in picks:
                try:
                    r = bt.leg_simple(px[tk], d0, bt.HORIZON,
                                      short=False, stop_frac=RISK_STOP)
                    rs.append((tk, r))
                except Exception:  # noqa: BLE001
                    continue
            gross = sum(w.get(tk, 0) * r for tk, r in rs) if rs else 0.0
        net = gross - cost
        eq *= 1 + net
        peak = max(peak, eq)
        rows.append({"date": d0, "net": net})
        prev_w = dict(w)
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
        turnover = bt.equal_weight_turnover(prev, picks)
        cost = turnover * COST / 1e4
        fut = tpx.loc[tpx.index > d0]
        rs = []
        for tk in picks:
            try:
                rs.append(bt.leg_simple(tpx[tk], d0, 10,
                                        short=False, stop_frac=RISK_STOP))
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


def _rank_book(test, proba, px, ml_weight: float,
               consensus: bool = False, inverse_vol: bool = False):
    """Monthly top-20 book using cross-sectional ranks, not proba levels.

    This removes sensitivity to probability calibration while retaining a
    fixed, auditable selection boundary. ``ml_weight=0`` is the unspent
    momentum control recomputed alongside every new round.
    """
    from predictor.backtest import RISK_STOP

    t = test.copy().reset_index(drop=True)
    t["proba"] = np.asarray(proba, dtype=float)
    months = pd.to_datetime(t["Date"]).dt.to_period("M").drop_duplicates().sort_values()
    rows, prev_w = [], {}
    for m in months:
        block = t[pd.to_datetime(t["Date"]).dt.to_period("M") == m]
        d0 = block["Date"].max()
        c = block[block["Date"] == d0].copy()
        hist = px.loc[px.index <= d0]
        if len(hist) < 252:
            continue
        mom = hist.iloc[-21] / hist.iloc[-252] - 1
        c["mom_rank"] = c["ticker"].map(mom.rank(pct=True))
        c["ml_rank"] = c["proba"].rank(pct=True)
        c = c.dropna(subset=["mom_rank", "ml_rank"])
        c["score"] = ml_weight * c["ml_rank"] + (1 - ml_weight) * c["mom_rank"]
        if consensus:
            c = c[(c["ml_rank"] >= 0.60) & (c["mom_rank"] >= 0.60)]
        picks = c.sort_values("score", ascending=False).head(TOP_N)["ticker"].tolist()

        if inverse_vol and picks:
            inv = {}
            for tk in picks:
                v = trailing_vol(tk, d0)
                inv[tk] = 1 / v if np.isfinite(v) and v > 0 else 0.0
            total = sum(inv.values())
            weights = ({tk: value / total for tk, value in inv.items()}
                       if total else {tk: 1 / len(picks) for tk in picks})
        else:
            weights = {tk: 1 / len(picks) for tk in picks} if picks else {}

        turnover = bt.weight_turnover(prev_w, weights)
        gross = 0.0
        realized = []
        for tk in picks:
            try:
                realized.append((tk, bt.leg_simple(
                    px[tk], d0, bt.HORIZON, short=False, stop_frac=RISK_STOP)))
            except Exception:  # noqa: BLE001
                continue
        if realized:
            gross = sum(weights.get(tk, 0.0) * ret for tk, ret in realized)
        rows.append({"date": d0, "net": gross - turnover * COST / 1e4})
        prev_w = weights
    return pd.DataFrame(rows).set_index("date")["net"]


def p_ml_rank(test, proba, px):
    return _rank_book(test, proba, px, ml_weight=1.0)


def p_mom75_ml25(test, proba, px):
    return _rank_book(test, proba, px, ml_weight=0.25)


def p_consensus40(test, proba, px):
    return _rank_book(test, proba, px, ml_weight=0.50, consensus=True)


def p_mom75_ml25_volw(test, proba, px):
    return _rank_book(test, proba, px, ml_weight=0.25, inverse_vol=True)


def momentum_control(test, proba, px):
    return _rank_book(test, proba, px, ml_weight=0.0)


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
    "P13-ml-rank": (p_ml_rank,
        "Removing the absolute probability gate improves stability, but weak ML ranking keeps Sharpe below momentum."),
    "P14-mom75-ml25": (p_mom75_ml25,
        "A momentum-dominant blend preserves most baseline strength; ML adds little, expected Sharpe near but below momentum."),
    "P15-consensus40": (p_consensus40,
        "Requiring both ranks in the top 40% raises signal agreement but concentrates the book; Sharpe may improve while DD worsens."),
    "P16-mom75-ml25-volw": (p_mom75_ml25_volw,
        "Momentum-heavy ranks plus inverse-volatility weights should reduce DD; expected best risk-adjusted candidate of this round."),
}


def read_log():
    if LOG.exists():
        return json.loads(LOG.read_text())
    return {"spent": 0, "proposals": {}, "archive": []}


def write_log(d):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(d, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    round_cmd = sub.add_parser("round")
    round_cmd.add_argument("--cache", required=True,
                           help="fold cache to test; required to lock data vintage")
    round_cmd.add_argument("--mode", required=True,
                           choices=("exploratory", "preregistered"),
                           help="evidence status; explored history cannot be promoted")
    promote_cmd = sub.add_parser("promote")
    promote_cmd.add_argument("--cache", required=True,
                             help="same fold cache used for the tune round")
    validate_cmd = sub.add_parser("validate")
    validate_cmd.add_argument("--id", required=True,
                              help="pending id in external_validations registry")
    replay = sub.add_parser("replay-promoted")
    replay.add_argument("--cache", required=True,
                        help="fold cache to replay; required so cache vintage is explicit")
    replay.add_argument("--start", help="optional inclusive result start date")
    replay.add_argument("--end", help="optional inclusive result end date")
    replay.add_argument("--half", choices=("all", "tune", "holdout"),
                        default="all", help="which original cache half to replay")
    args = ap.parse_args()
    log = read_log()

    if args.cmd == "status":
        print(f"budget: {BUDGET_TOTAL - log['spent']}/{BUDGET_TOTAL} left")
        print("registry metrics are historical pre-fix outputs; "
              "run replay-promoted with an explicit --cache for corrected metrics")
        for pid, (fn, pred) in PROPOSALS.items():
            st = log["proposals"].get(pid, {"status": "pending"})
            print(f"{pid} [{st['status']}] pred: {pred} "
                  f"tune={st.get('tune')} hold={st.get('holdout')}")
        return

    if args.cmd == "replay-promoted":
        # Correctness replay only: do not spend budget, change statuses, or
        # overwrite the immutable historical registry.
        folds = load_folds(args.cache)
        mid = len(folds) // 2
        promoted = ("P5-gate25", "P8-volscale", "P9-top30", "P12-top40")
        print(f"cache: {args.cache} folds: {len(folds)}")
        halves = (("tune", folds[:mid]), ("holdout", folds[mid:]))
        for half, selected in halves:
            if args.half != "all" and args.half != half:
                continue
            tickers = sorted({tk for t, _ in selected
                              for tk in t["ticker"].unique()})
            px = load_px(tickers)
            for pid in promoted:
                fn, _ = PROPOSALS[pid]
                nets = stitch([fn(t, p, px) for t, p in selected])
                if args.start:
                    nets = nets.loc[nets.index >= pd.Timestamp(args.start)]
                if args.end:
                    nets = nets.loc[nets.index <= pd.Timestamp(args.end)]
                if nets.empty:
                    raise ValueError("date filters removed every replay result")
                result = score(nets)
                window = (f"{pd.Timestamp(nets.index.min()).date()}.."
                          f"{pd.Timestamp(nets.index.max()).date()}")
                print(f"{pid} {half} [{window}]: {result}")
        return

    if args.cmd == "validate":
        plans = log.get("external_validations", {})
        if args.id not in plans:
            raise ValueError(f"unregistered external validation: {args.id}")
        plan = plans[args.id]
        if plan.get("status") != "pending":
            raise ValueError(f"external validation {args.id} already executed")
        pid = plan["proposal"]
        if pid not in PROPOSALS:
            raise ValueError(f"unknown proposal in validation plan: {pid}")
        cache = plan["cache"]
        folds = load_folds(cache)
        if not folds:
            raise ValueError(f"no folds found in {cache}")
        tickers = sorted({tk for t, _ in folds for tk in t["ticker"].unique()})
        px = load_px(tickers)
        candidate = stitch([PROPOSALS[pid][0](t, p, px) for t, p in folds])
        control = stitch([momentum_control(t, p, px) for t, p in folds])
        candidate, control = candidate.align(control, join="inner")
        if candidate.empty:
            raise ValueError("candidate and control have no common results")
        candidate_score = score(candidate)
        control_score = score(control)
        gate = plan["gate"]
        passed = (
            candidate_score["CAGR"] > control_score["CAGR"]
            and candidate_score["sharpe"] > control_score["sharpe"]
            and candidate_score["maxDD"]
            >= control_score["maxDD"] - float(gate["max_dd_tolerance"])
        )
        plan.update({
            "status": "passed-external" if passed else "failed-external",
            "window": (f"{pd.Timestamp(candidate.index.min()).date()}.."
                       f"{pd.Timestamp(candidate.index.max()).date()}"),
            "candidate": candidate_score,
            "control": control_score,
            "passed": passed,
        })
        write_log(log)
        print(f"{args.id} candidate: {candidate_score}")
        print(f"{args.id} momentum-control: {control_score}")
        print(f"{args.id}: {'PASS' if passed else 'FAIL'} under registered gate")
        return

    if args.cmd == "round":
        folds = load_folds(args.cache)
        if not folds:
            raise ValueError(f"no folds found in {args.cache}")
        tune = folds[:len(folds) // 2]
        tickers = sorted({tk for t, _ in tune for tk in t["ticker"].unique()})
        px = load_px(tickers)
        pending = [pid for pid in PROPOSALS if log["proposals"].get(
            pid, {}).get("status") in (None, "pending")]
        if not pending:
            print("no pending proposals")
            return
        if log["spent"] + len(pending) > BUDGET_TOTAL:
            print("BUDGET EXCEEDED — loop halts. No more proposals, ever.")
            return
        control_nets = stitch([momentum_control(t, p, px) for t, p in tune])
        control = score(control_nets)
        print(f"momentum-control: {control} (does not spend budget)")
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
            status = ("exploratory-tune" if args.mode == "exploratory"
                      else "tested-tune")
            log["proposals"][pid] = {"status": status, "pred": pred,
                                     "cache": args.cache, "mode": args.mode,
                                     "tune": s}
            log["archive"].append({"id": pid, "cache": args.cache,
                                   "mode": args.mode, "tune": s})
            print(f"{pid}: {s}  pred was: {pred}")
            log["spent"] += 1
        log.setdefault("rounds", []).append({
            "cache": args.cache,
            "mode": args.mode,
            "proposals": pending,
            "window": (f"{pd.Timestamp(control_nets.index.min()).date()}.."
                       f"{pd.Timestamp(control_nets.index.max()).date()}"),
            "control": control,
        })
        write_log(log)
        if args.mode == "exploratory":
            print("round done. Exploratory history is ineligible for promotion.")
        else:
            print("round done. Promote only the best IF bar passes "
                  "(Sharpe > 0.41 AND maxDD > -0.35).")
        return

    if args.cmd == "promote":
        folds = load_folds(args.cache)
        if not folds:
            raise ValueError(f"no folds found in {args.cache}")
        hold = folds[len(folds) // 2:]
        tickers = sorted({tk for t, _ in hold for tk in t["ticker"].unique()})
        px = load_px(tickers)
        cands = {pid: st for pid, st in log["proposals"].items()
                 if st.get("status") == "tested-tune"
                 and st.get("cache") == args.cache}
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
