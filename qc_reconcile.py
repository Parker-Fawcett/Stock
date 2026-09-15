"""Reconcile QuantConnect daily equity with the local momentum ledger.

Usage:
  python3 qc_reconcile.py --qc-json "/path/to/Determined Black Cow.json"

The raw cloud download is not committed. The output records its SHA-256 hash,
an extracted daily equity path, aligned monthly returns, and summary metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mom_run import run_momentum
from predictor import universe as U
from predictor.evaluate import max_drawdown


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_sha256(paths: list[Path]) -> str:
    """Hash filenames and contents so the local input set is reproducible."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def qc_daily_equity(payload: dict) -> pd.Series:
    values = payload["charts"]["Strategy Equity"]["series"]["Equity"]["values"]
    frame = pd.DataFrame(values, columns=["timestamp", "open", "high", "low", "close"])
    frame["date"] = (pd.to_datetime(frame["timestamp"], unit="s", utc=True)
                     .dt.tz_convert("America/New_York").dt.tz_localize(None))
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    equity = frame.set_index("date")["close"].astype(float)
    return equity.groupby(equity.index.normalize()).last().rename("qc_equity")


def local_panel() -> pd.DataFrame:
    frames = []
    for ticker in [t for t in U.SMALLCAP if t != "LEG"]:
        path = Path("data/raw") / f"{ticker}.csv"
        data = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
        data["ticker"] = ticker
        frames.append(data[["Date", "ticker", "Close"]])
    return pd.concat(frames, ignore_index=True)


def local_momentum(panel: pd.DataFrame) -> pd.DataFrame:
    result = run_momentum(panel, cost_bps=25.0)
    return result[(result["date"] >= pd.Timestamp("2011-02-28"))
                  & (result["date"] <= pd.Timestamp("2026-05-29"))].copy()


def local_pick_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the frozen local ranker and retain its selected names."""
    px = panel.set_index(["ticker", "Date"])["Close"].unstack("ticker")
    months = pd.to_datetime(px.index.to_series()).dt.to_period("M").drop_duplicates()
    rows = []
    for month in months.sort_values()[13:]:
        decision = px.index[pd.to_datetime(px.index.to_series()).dt.to_period("M") == month].max()
        hist = px.loc[px.index <= decision]
        if len(hist) < 257:
            continue
        momentum = (hist.iloc[-21] / hist.iloc[-252] - 1).dropna()
        if len(momentum) < 10:
            continue
        cutoff = momentum.quantile(0.9)
        picks = momentum[momentum >= cutoff].index.tolist()[:max(5, int(len(momentum) * 0.10) + 1)]
        rows.append({
            "realization_month": month + 1,
            "local_picks": ",".join(picks),
            "local_n": len(picks),
        })
    return pd.DataFrame(rows).set_index("realization_month")


def qc_symbol_map(payload: dict) -> dict[str, str]:
    """Map point-in-time/IPO tickers in logs to current requested tickers."""
    mapping: dict[str, str] = {}
    for order in payload.get("orders", {}).values():
        symbol = order.get("symbol", {})
        current = symbol.get("value")
        sid = symbol.get("id", "")
        if current:
            mapping[current] = current
            if sid:
                mapping[sid.split()[0]] = current
    # Intermediate tickers aren't necessarily the SID's first or current value.
    mapping.update({"BSIG": "AAMI", "NYMT": "ADAM"})
    return mapping


def qc_pick_table(log_path: Path, payload: dict) -> pd.DataFrame:
    pattern = re.compile(r"^(\d{4}-\d{2})-\d{2} .*? picks: ([A-Z0-9,.-]+)$", re.MULTILINE)
    mapping = qc_symbol_map(payload)
    rows = []
    for month, raw in pattern.findall(log_path.read_text()):
        raw_picks = raw.split(",")
        picks = sorted(mapping.get(ticker, ticker) for ticker in raw_picks)
        rows.append({
            "realization_month": pd.Period(month, "M"),
            "qc_picks_raw": ",".join(raw_picks),
            "qc_picks_current": ",".join(picks),
            "qc_n": len(picks),
        })
    if not rows:
        raise ValueError(f"No 'picks:' records found in {log_path}")
    return pd.DataFrame(rows).set_index("realization_month")


def series_metrics(returns: pd.Series) -> dict[str, float | int]:
    returns = returns.dropna().astype(float)
    equity = (1 + returns).cumprod()
    return {
        "months": int(len(returns)),
        "CAGR": float(equity.iloc[-1] ** (12 / len(returns)) - 1),
        "Sharpe_monthly": float(returns.mean() / returns.std() * np.sqrt(12)),
        "maxDD_monthly_endpoints": float(max_drawdown(equity)),
    }


def reconcile(qc_json: Path, output: Path, qc_log: Path | None = None) -> dict:
    payload = json.loads(qc_json.read_text())
    daily = qc_daily_equity(payload)
    qc_monthly_equity = daily.groupby(daily.index.to_period("M")).last()
    qc_monthly_returns = qc_monthly_equity.pct_change().rename("qc_return")

    panel = local_panel()
    local = local_momentum(panel)
    local_returns = local.set_index(local["date"].dt.to_period("M"))["net"]
    # A local row is labeled by its month-end decision date but realizes over
    # the following 21 sessions. Shift it into the following calendar month.
    local_returns.index = local_returns.index + 1
    local_returns = local_returns.rename("local_return")

    aligned = pd.concat([local_returns, qc_monthly_returns], axis=1).dropna()
    # QC ends June 17, 2026. Exclude that partial calendar month from the core
    # path comparison; retain it in the exported table with a completeness flag.
    aligned["complete_qc_month"] = aligned.index <= pd.Period("2026-05", "M")
    aligned["difference"] = aligned["qc_return"] - aligned["local_return"]
    complete = aligned[aligned["complete_qc_month"]]

    qc_daily_return = daily.pct_change().dropna()
    qc_daily_curve = daily / daily.iloc[0]
    monthly_qc_curve = qc_monthly_equity / qc_monthly_equity.iloc[0]
    local_curve = (1 + local["net"]).cumprod()
    diff = complete["difference"]
    correlation = float(complete[["local_return", "qc_return"]].corr().iloc[0, 1])

    daily_dd = qc_daily_curve / qc_daily_curve.cummax() - 1
    daily_trough = daily_dd.idxmin()
    daily_peak = qc_daily_curve.loc[:daily_trough].idxmax()
    monthly_dd = monthly_qc_curve / monthly_qc_curve.cummax() - 1
    local_dd = local_curve / local_curve.cummax() - 1
    local_trough = int(local_dd.idxmin())
    local_peak = int(local_curve.loc[:local_trough].idxmax())

    summary = {
        "source": {
            "file": qc_json.name,
            "sha256": file_sha256(qc_json),
            "backtest": payload["state"]["Name"],
            "algorithm_id": payload["state"].get("AlgorithmId"),
            "local_raw_files": len([t for t in U.SMALLCAP if t != "LEG"]),
            "local_raw_aggregate_sha256": files_sha256([
                Path("data/raw") / f"{ticker}.csv"
                for ticker in U.SMALLCAP if ticker != "LEG"
            ]),
            "mom_run_sha256": file_sha256(Path("mom_run.py")),
            "universe_sha256": file_sha256(Path("predictor/universe.py")),
            "reconciler_sha256": file_sha256(Path(__file__)),
        },
        "reported_quantconnect": payload["statistics"],
        "matched_complete_months": {
            "start": str(complete.index.min()),
            "end": str(complete.index.max()),
            "months": int(len(complete)),
            "return_correlation": correlation,
            "return_r_squared": correlation ** 2,
            "mean_qc_minus_local": float(diff.mean()),
            "mean_absolute_difference": float(diff.abs().mean()),
            "root_mean_squared_difference": float(np.sqrt(np.mean(diff ** 2))),
            "quantconnect_metrics": series_metrics(complete["qc_return"]),
            "local_metrics": series_metrics(complete["local_return"]),
        },
        "frequency_reconciliation": {
            "qc_reported_sharpe": float(payload["totalPerformance"]["portfolioStatistics"]["sharpeRatio"]),
            "qc_naive_daily_sharpe": float(qc_daily_return.mean() / qc_daily_return.std() * np.sqrt(252)),
            "qc_monthly_sharpe": series_metrics(qc_monthly_returns)["Sharpe_monthly"],
            "local_monthly_sharpe": series_metrics(local_returns)["Sharpe_monthly"],
            "qc_reported_daily_maxDD": float(daily_dd.min()),
            "qc_monthly_endpoint_maxDD": float(monthly_dd.min()),
            "local_monthly_endpoint_maxDD": float(local_dd.min()),
            "qc_daily_peak": str(daily_peak.date()),
            "qc_daily_trough": str(daily_trough.date()),
            "local_monthly_peak_decision": str(local.loc[local_peak, "date"].date()),
            "local_monthly_trough_decision": str(local.loc[local_trough, "date"].date()),
        },
        "monthly_metrics": {
            "quantconnect_all_calendar_months": series_metrics(qc_monthly_returns),
            "local_all_21_session_legs": series_metrics(local_returns),
        },
    }

    if qc_log is not None:
        log_text = qc_log.read_text()
        algorithm_match = re.search(r"Launching analysis for ([a-f0-9]+)", log_text)
        if algorithm_match:
            summary["source"]["algorithm_id"] = algorithm_match.group(1)
        picks = qc_pick_table(qc_log, payload).join(local_pick_table(panel), how="inner")
        def score(row: pd.Series) -> pd.Series:
            qc_set = set(row["qc_picks_current"].split(","))
            local_set = set(row["local_picks"].split(","))
            intersection = qc_set & local_set
            return pd.Series({
                "pick_intersection_n": len(intersection),
                "pick_union_n": len(qc_set | local_set),
                "pick_jaccard": len(intersection) / len(qc_set | local_set),
                "exact_pick_match": qc_set == local_set,
            })
        picks = picks.join(picks.apply(score, axis=1))
        summary["source"]["log_file"] = qc_log.name
        summary["source"]["log_sha256"] = file_sha256(qc_log)
        summary["pick_reconciliation"] = {
            "start": str(picks.index.min()),
            "end": str(picks.index.max()),
            "months": int(len(picks)),
            "log_limit_note": "QuantConnect free-tier export truncated at 10 KB.",
            "mean_jaccard": float(picks["pick_jaccard"].mean()),
            "median_jaccard": float(picks["pick_jaccard"].median()),
            "mean_intersection_names": float(picks["pick_intersection_n"].mean()),
            "exact_match_months": int(picks["exact_pick_match"].sum()),
        }
        joined = complete.join(picks, how="inner")
        exact = joined[joined["exact_pick_match"]]
        nonexact = joined[~joined["exact_pick_match"]]
        summary["pick_reconciliation"].update({
            "return_MAE_exact_pick_months": float(exact["difference"].abs().mean()),
            "return_MAE_nonexact_pick_months": float(nonexact["difference"].abs().mean()),
            "abs_return_difference_vs_pick_mismatch_correlation": float(
                joined["difference"].abs().corr(1 - joined["pick_jaccard"])),
        })
        export_picks = picks.copy()
        export_picks.index = export_picks.index.astype(str)
        export_picks.index.name = "realization_month"
        export_picks.to_csv(output / "monthly_pick_comparison.csv")

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": daily.index, "equity": daily.values}).to_csv(
        output / "determined_black_cow_daily_equity.csv", index=False)
    export = aligned.copy()
    export.index = export.index.astype(str)
    export.index.name = "realization_month"
    export.to_csv(output / "monthly_return_comparison.csv")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qc-json", type=Path, required=True)
    parser.add_argument("--qc-log", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("data/qc_reconciliation"))
    args = parser.parse_args()
    summary = reconcile(args.qc_json, args.output, args.qc_log)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
