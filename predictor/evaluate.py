"""Evaluation: rank quality + money quality + risk. Both, always.

His lesson: AUC without P&L is trivia, P&L without monkeys is luck
(Lh1vrIcpJN4, noK0IwZAnyE). Report mean over folds/seeds, never best.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, p))
    except Exception:  # noqa: BLE001 - single class in slice
        return float("nan")


def max_drawdown(equity: pd.Series) -> float:
    # Include starting equity so a loss in the first measured period is visible.
    curve = pd.concat([pd.Series([1.0]), equity.reset_index(drop=True)],
                      ignore_index=True)
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(dd.min())


def summarize(bt: pd.DataFrame, label: str = "ai") -> dict:
    eq = bt["equity"]
    n_months = max(1, len(bt))
    years = n_months / 12
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if len(eq) else 0.0
    monthly = bt["net"]
    sharpe = float(monthly.mean() / (monthly.std() + 1e-12) * np.sqrt(12))
    return {
        "label": label,
        "months": n_months,
        "equity_end": float(eq.iloc[-1]) if len(eq) else 1.0,
        "CAGR": cagr,
        "Sharpe_m": sharpe,
        "maxDD": max_drawdown(eq) if len(eq) else 0.0,
        "avg_turnover": float(bt["turnover"].mean()) if len(bt) else 0.0,
        "exposure": float((bt["n"] > 0).mean()) if len(bt) else 0.0,
    }


def cost_sweep_report(test, proba_fn, costs_bps=(0, 5, 10, 25)) -> pd.DataFrame:
    from .backtest import run_backtest

    rows = []
    for c in costs_bps:
        bt = run_backtest(test, proba_fn, cost_bps=c)
        s = summarize(bt)
        s["cost_bps"] = c
        rows.append(s)
    return pd.DataFrame(rows)
