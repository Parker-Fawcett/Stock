"""Prospective paper trading for the frozen ML and momentum sleeves.

The v2 series is provenance-locked and uses next-session open entries. Legacy
rows remain in data/paper/log.csv and are never mixed into the clean series.

Usage:
  python3 paper.py predict --universe smallcap --market IWM --refresh
  python3 paper.py grade --refresh
  python3 paper.py grade --legacy
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import sklearn

from predictor.backtest import COST_BPS
from predictor.data import assert_vintage, download_universe
from predictor.features import HORIZON, build_latest, build_panel, model_cols
from predictor.model_lgbm import fit_fold
from predictor.provenance import file_sha256, git_state, payload_sha256

ML_MODEL = "wide-seed0-v1"
MOM_MODEL = "mom12-1-v1"
SERIES = "prospective-v2"
PAPER_DIR = Path(__file__).resolve().parent / "data" / "paper"
RUN_DIR = PAPER_DIR / "runs"
LEGACY_LOG = PAPER_DIR / "log.csv"
LOG = PAPER_DIR / "log_v2.csv"
GRADES = PAPER_DIR / "grades_v2.csv"
STOP_FRAC = 0.85
THRESHOLD = 0.30
TOP_N = 20

LOG_COLUMNS = [
    "series", "run_id", "created_at_utc", "decision_date", "ticker",
    "signal_value", "signal_close", "stop_frac", "entry_rule", "model",
    "universe", "market", "code_commit", "manifest_sha256",
]
GRADE_COLUMNS = [
    "series", "run_id", "decision_date", "entry_date", "exit_date", "ticker",
    "model", "entry", "exit", "ret", "market_ret", "excess",
    "graded_at_utc", "manifest_sha256",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def trailing_split(panel: pd.DataFrame):
    dates = np.sort(panel["Date"].unique())
    val_start = dates[-1] - pd.to_timedelta(6 * 30, unit="D")
    train_start = val_start - pd.to_timedelta(3 * 365, unit="D")
    embargo = pd.to_timedelta(HORIZON, unit="D")
    train = panel[(panel["Date"] >= train_start) &
                  (panel["Date"] < val_start)]
    train = train[train["label_end"] < val_start - embargo]
    val = panel[panel["Date"] >= val_start]
    return train, val


def is_month_end_data(asof: pd.Timestamp, today: pd.Timestamp) -> bool:
    """Allow a completed prior month or its final weekday observation."""
    asof, today = pd.Timestamp(asof).normalize(), pd.Timestamp(today).normalize()
    if asof.to_period("M") < today.to_period("M"):
        return True
    return (asof + pd.offsets.BDay(1)).to_period("M") != asof.to_period("M")


def _read_table(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path)


def _append_table(path: Path, rows: pd.DataFrame, columns: list[str]) -> None:
    if rows.empty:
        return
    rows.reindex(columns=columns).to_csv(
        path, mode="a", header=not path.exists() or path.stat().st_size == 0,
        index=False)


def _universe(args) -> list[str]:
    if args.universe:
        from predictor import universe as universe_module

        return [ticker.strip().upper()
                for ticker in getattr(universe_module, args.universe.upper())]
    return [ticker.strip().upper() for ticker in args.tickers.split(",")
            if ticker.strip()]


def _run_exists(decision_date: str, universe: str, market: str) -> bool:
    if not RUN_DIR.exists():
        return False
    for path in RUN_DIR.glob("*.json"):
        payload = json.loads(path.read_text())
        arguments = payload.get("arguments", {})
        if (payload.get("decision_date") == decision_date and
                arguments.get("universe") == universe and
                arguments.get("market") == market):
            return True
    return False


def _momentum_rows(paths, tickers: list[str], market: str,
                   decision_date: str) -> list[dict]:
    prices = {}
    for ticker in tickers:
        if ticker == market.upper() or ticker not in paths:
            continue
        frame = pd.read_csv(paths[ticker], parse_dates=["Date"]).sort_values("Date")
        prices[ticker] = frame.set_index("Date")["Close"]
    px = pd.DataFrame(prices).sort_index()
    if len(px) < 252:
        return []
    momentum = (px.iloc[-21] / px.iloc[-252] - 1).dropna().sort_values(
        ascending=False)
    cutoff = momentum.quantile(0.90)
    picks = momentum[momentum >= cutoff]
    if len(picks) < 5:
        return []
    return [{
        "decision_date": decision_date,
        "ticker": ticker,
        "signal_value": round(float(value), 6),
        "signal_close": round(float(px[ticker].iloc[-1]), 4),
        "stop_frac": STOP_FRAC,
        "entry_rule": "next_session_open",
        "model": MOM_MODEL,
    } for ticker, value in picks.items()]


def cmd_predict(args) -> None:
    state = git_state()
    if state.get("pipeline_dirty") and not args.allow_dirty:
        raise RuntimeError("refusing prospective prediction from dirty pipeline code")
    tickers = _universe(args)
    paths = download_universe(tickers, market=args.market, refresh=args.refresh)
    assert_vintage({key: str(value) for key, value in paths.items()})
    str_paths = {key: str(value) for key, value in paths.items()}
    panel = build_panel(str_paths, market=args.market)
    latest, asof = build_latest(str_paths, market=args.market)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None)
    if not is_month_end_data(asof, today) and not args.allow_offcycle:
        raise RuntimeError(
            f"latest data {asof.date()} is not a completed month-end; "
            "use --allow-offcycle only for an explicitly labeled test")

    decision_date = asof.date().isoformat()
    universe_name = args.universe or "custom"
    market_name = args.market.upper()
    if _run_exists(decision_date, universe_name, market_name):
        raise RuntimeError(f"prospective run already exists for {decision_date}")
    old = _read_table(LOG, LOG_COLUMNS)
    duplicate = old[(old["decision_date"].astype(str) == decision_date) &
                    (old["universe"] == universe_name) &
                    (old["market"] == market_name)]
    if not duplicate.empty:
        raise RuntimeError(f"prospective run already exists for {decision_date}")

    cols = model_cols(panel)
    train, val = trailing_split(panel)
    if len(train) < 500 or len(val) < 100:
        raise RuntimeError("not enough trailing history")
    model = fit_fold(train, val, seed=0)
    latest = latest.copy()
    latest["proba"] = model.predict_proba(latest[cols].values)[:, 1]
    ml = latest[latest["proba"] >= THRESHOLD].nlargest(TOP_N, "proba")
    signal_rows = [{
        "decision_date": decision_date,
        "ticker": row["ticker"],
        "signal_value": round(float(row["proba"]), 6),
        "signal_close": round(float(row["Close"]), 4),
        "stop_frac": STOP_FRAC,
        "entry_rule": "next_session_open",
        "model": ML_MODEL,
    } for _, row in ml.iterrows()]
    signal_rows.extend(_momentum_rows(paths, tickers, args.market, decision_date))

    created = utc_now()
    run_id = created.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    manifest = {
        "schema_version": 2,
        "series": SERIES,
        "run_id": run_id,
        "created_at_utc": created.isoformat(),
        "decision_date": decision_date,
        "code": state,
        "arguments": {
            "universe": universe_name,
            "custom_tickers": args.tickers,
            "market": market_name,
            "refresh": bool(args.refresh),
            "allow_offcycle": bool(args.allow_offcycle),
        },
        "data": {
            "asof": decision_date,
            "panel_start": str(pd.Timestamp(panel.Date.min()).date()),
            "panel_end": str(pd.Timestamp(panel.Date.max()).date()),
            "panel_rows": len(panel),
            "latest_universe_size": int(latest["ticker"].nunique()),
            "input_sha256": {ticker: file_sha256(path)
                             for ticker, path in sorted(str_paths.items())},
        },
        "model": {
            "columns": cols,
            "backend": f"{type(model).__module__}.{type(model).__name__}",
            "seed": 0,
            "train_start": str(pd.Timestamp(train.Date.min()).date()),
            "train_end": str(pd.Timestamp(train.Date.max()).date()),
            "train_rows": len(train),
            "val_start": str(pd.Timestamp(val.Date.min()).date()),
            "val_end": str(pd.Timestamp(val.Date.max()).date()),
            "val_rows": len(val),
            "ml_version": ML_MODEL,
            "momentum_version": MOM_MODEL,
            "threshold": THRESHOLD,
            "top_n": TOP_N,
            "stop_frac": STOP_FRAC,
            "entry_rule": "next_session_open",
        },
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": sys.version,
            "scikit_learn": sklearn.__version__,
        },
        "signals": signal_rows,
    }
    manifest_hash = payload_sha256(manifest)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RUN_DIR / f"{run_id}.json"
    with manifest_path.open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    rows = [{
        "series": SERIES,
        "run_id": run_id,
        "created_at_utc": created.isoformat(),
        **signal,
        "universe": universe_name,
        "market": market_name,
        "code_commit": state.get("commit", "unknown"),
        "manifest_sha256": manifest_hash,
    } for signal in signal_rows]
    _append_table(LOG, pd.DataFrame(rows), LOG_COLUMNS)
    counts = pd.Series([row["model"] for row in rows]).value_counts().to_dict()
    print(f"recorded run {run_id} at {decision_date}: {counts or {'CASH': 0}}")
    print(f"manifest {manifest_hash}")


def grade_position(stock: pd.DataFrame, market: pd.DataFrame,
                   decision_date, stop_frac: float = STOP_FRAC) -> dict | None:
    stock = stock.loc[stock.index > pd.Timestamp(decision_date)].iloc[:HORIZON]
    market = market.loc[market.index > pd.Timestamp(decision_date)].iloc[:HORIZON]
    if len(stock) < HORIZON or len(market) < HORIZON:
        return None
    entry = float(stock.iloc[0]["Open"])
    stop = entry * stop_frac
    hit = stock[stock["Close"] <= stop]
    if len(hit):
        exit_price = float(hit.iloc[0]["Close"])
        exit_date = hit.index[0]
    else:
        exit_price = float(stock.iloc[-1]["Close"])
        exit_date = stock.index[-1]
    ret = exit_price / entry - 1 - 2 * COST_BPS / 1e4
    market_ret = float(market.iloc[-1]["Close"] / market.iloc[0]["Open"] - 1)
    return {
        "entry_date": stock.index[0].date().isoformat(),
        "exit_date": pd.Timestamp(exit_date).date().isoformat(),
        "entry": entry,
        "exit": exit_price,
        "ret": ret,
        "market_ret": market_ret,
        "excess": ret - market_ret,
    }


def _verify_manifest(row) -> None:
    path = RUN_DIR / f"{row['run_id']}.json"
    payload = json.loads(path.read_text())
    if payload_sha256(payload) != row["manifest_sha256"]:
        raise RuntimeError(f"manifest hash mismatch for {row['run_id']}")
    if (payload.get("run_id") != row["run_id"] or
            payload.get("series") != row["series"] or
            payload.get("decision_date") != str(row["decision_date"])):
        raise RuntimeError(f"log metadata mismatch for {row['run_id']}")
    matches = [signal for signal in payload.get("signals", [])
               if signal.get("ticker") == row["ticker"] and
               signal.get("model") == row["model"]]
    if len(matches) != 1:
        raise RuntimeError(f"log signal missing from manifest for {row['run_id']}")
    signal = matches[0]
    for field in ("signal_value", "signal_close", "stop_frac"):
        if not np.isclose(float(signal[field]), float(row[field])):
            raise RuntimeError(f"log {field} mismatch for {row['run_id']}")


def cmd_grade(args) -> None:
    if args.legacy:
        return cmd_grade_legacy(args)
    log = _read_table(LOG, LOG_COLUMNS)
    if log.empty:
        print("clean prospective series has no picks yet")
        return
    grades = _read_table(GRADES, GRADE_COLUMNS)
    done = set(zip(grades.get("run_id", []), grades.get("ticker", []),
                   grades.get("model", [])))
    pending = log[[((row.run_id, row.ticker, row.model) not in done)
                   for row in log.itertuples()]]
    if pending.empty:
        print("all prospective picks are already graded")
        return
    paths = download_universe(
        sorted(set(pending["ticker"])) + sorted(set(pending["market"])),
        market=args.market, refresh=args.refresh)
    frames = {ticker.upper(): pd.read_csv(path, parse_dates=["Date"])
              .sort_values("Date").set_index("Date")
              for ticker, path in paths.items()}
    rows = []
    for row in pending.to_dict("records"):
        _verify_manifest(row)
        ticker, market = row["ticker"], row["market"]
        if ticker not in frames or market not in frames:
            continue
        result = grade_position(frames[ticker], frames[market],
                                row["decision_date"], float(row["stop_frac"]))
        if result is None:
            continue
        rows.append({
            "series": SERIES,
            "run_id": row["run_id"],
            "decision_date": row["decision_date"],
            "ticker": ticker,
            "model": row["model"],
            **{key: round(value, 6) if isinstance(value, float) else value
               for key, value in result.items()},
            "graded_at_utc": utc_now().isoformat(),
            "manifest_sha256": row["manifest_sha256"],
        })
    _append_table(GRADES, pd.DataFrame(rows), GRADE_COLUMNS)
    all_grades = _read_table(GRADES, GRADE_COLUMNS)
    if not rows:
        print(f"{len(pending)} clean picks pending; none has {HORIZON} sessions yet")
        return
    for model, group in all_grades.groupby("model"):
        print(f"[{model}] n={len(group)} hit={(group.ret > 0).mean():.2f} "
              f"mean={group.ret.mean():.4f} excess={group.excess.mean():.4f}")


def cmd_grade_legacy(args) -> None:
    if not LEGACY_LOG.exists():
        print("no legacy picks logged")
        return
    log = pd.read_csv(LEGACY_LOG, parse_dates=["date"])
    paths = download_universe(sorted(set(log["ticker"])) + [args.market],
                              market=args.market, refresh=args.refresh)
    closes = {ticker.upper(): pd.read_csv(path, parse_dates=["Date"])
              .sort_values("Date").set_index("Date")["Close"]
              for ticker, path in paths.items()}
    rows = []
    for row in log.itertuples():
        try:
            stock = closes[row.ticker].loc[closes[row.ticker].index > row.date].iloc[:HORIZON]
            market = closes[args.market].loc[closes[args.market].index > row.date].iloc[:HORIZON]
            if len(stock) < HORIZON or len(market) < HORIZON:
                continue
            hit = stock[stock <= row.stop]
            ret = ((float(hit.iloc[0]) if len(hit) else float(stock.iloc[-1])) /
                   float(row.ref_close) - 1 - 2 * COST_BPS / 1e4)
            market_ret = float(market.iloc[-1] / market.iloc[0] - 1)
            rows.append({"model": row.model, "ret": ret,
                         "excess": ret - market_ret})
        except Exception:  # noqa: BLE001
            continue
    if not rows:
        print("legacy picks are not mature")
        return
    for model, group in pd.DataFrame(rows).groupby("model"):
        print(f"[legacy:{model}] n={len(group)} hit={(group.ret > 0).mean():.2f} "
              f"mean={group.ret.mean():.4f} excess={group.excess.mean():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("--tickers", default="")
    predict.add_argument("--universe", default="smallcap")
    predict.add_argument("--market", default="IWM")
    predict.add_argument("--refresh", action="store_true")
    predict.add_argument("--allow-dirty", action="store_true",
                         help="development only; manifest records dirty state")
    predict.add_argument("--allow-offcycle", action="store_true",
                         help="development only; permit a non-month-end decision")
    grade = sub.add_parser("grade")
    grade.add_argument("--market", default="IWM")
    grade.add_argument("--refresh", action="store_true")
    grade.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    if args.cmd == "predict":
        cmd_predict(args)
    else:
        cmd_grade(args)


if __name__ == "__main__":
    main()
