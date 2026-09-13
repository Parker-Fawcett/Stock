"""Monthly Graham backtest. His rules, our harness.

Rules (SgQ-RwC97us): buy when close <= (1-discount)*IV, sell at IV or
after 3 years, equal weight, no drift rebalancing, costs on trades.
Fixes vs his versions:
- Fair benchmark: equal-weight buy-hold of the CALCULABLE universe only
  (his own Q6vgadS1HiE correction — dropping unvaluable stocks from the
  strategy but not the benchmark is cheating).
- filed_date availability everywhere (no lookahead into restatements).
- Tune/holdout split from minute one. Primary discount 0.5 (his);
  0.3/0.7 reported as sensitivity, never selected on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOLD_MAX_YEARS = 3


def run_value_backtest(
    iv: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    discount: float = 0.5,
    cost_bps: float = 5.0,
    start_cash: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    iv = iv.sort_values(["ticker", "filed_date"]).reset_index(drop=True)
    # month-ends spanning all data
    all_dates = sorted({d for df in prices.values() for d in df["Date"]})
    months = pd.to_datetime(pd.Series(all_dates)).dt.to_period("M").drop_duplicates()
    months = months.sort_values()
    px = {tk: df.set_index("Date")["Close"] for tk, df in prices.items()}

    def asof_iv(tk: str, d) -> float | None:
        rows = iv[(iv.ticker == tk) & (iv.filed_date <= d)]
        if len(rows) == 0 or pd.isna(rows.iloc[-1]["iv"]):
            return None
        return float(rows.iloc[-1]["iv"])

    def px_at(tk: str, d):
        try:
            return float(px[tk].loc[px[tk].index <= d].iloc[-1])
        except Exception:  # noqa: BLE001
            return None

    cash = start_cash
    holds: dict[str, dict] = {}
    bench_px: dict[str, float] = {}
    rows, brows = [], []
    for m in months:
        d = pd.Timestamp(m.to_timestamp("M"))
        # settle sells first
        for tk in list(holds):
            p = px_at(tk, d)
            if p is None:
                continue
            h = holds[tk]
            aged = (d - h["buy_date"]).days > HOLD_MAX_YEARS * 365
            if p >= h["iv"] or aged:
                cash += h["shares"] * p * (1 - cost_bps / 1e4)
                del holds[tk]
        # buys: find every qualifier first, then size each one against the
        # book that results from ALL of them buying, not just itself. The
        # old per-ticker loop gave the first qualifier every dollar of
        # cash and left nothing for the rest -- diversification depended
        # on ticker iteration order instead of the equal-weight rule.
        qualifiers = []
        for tk in px:
            if tk in holds:
                continue
            v = asof_iv(tk, d)
            p = px_at(tk, d)
            if v is None or p is None or v <= 0:
                continue
            if p <= (1 - discount) * v:
                qualifiers.append((tk, v, p))
        if qualifiers:
            n_after = len(holds) + len(qualifiers)
            held_value = sum(px_at(k, d) * h["shares"] for k, h in holds.items())
            alloc = (cash + held_value) / n_after
            for tk, v, p in qualifiers:
                spend = min(cash, alloc)
                if spend > 0:
                    sh = spend * (1 - cost_bps / 1e4) / p
                    holds[tk] = {"shares": sh, "iv": v, "buy_date": d}
                    cash -= spend
        # mark
        book = cash + sum((px_at(k, d) or 0) * h["shares"] for k, h in holds.items())
        # fair benchmark: equal-weight buy-hold of calculable names only
        calc = [tk for tk in px if asof_iv(tk, d) is not None and px_at(tk, d)]
        if calc and not bench_px:
            bench_px = {tk: 1.0 / len(calc) / px_at(tk, d) for tk in calc}
        bval = sum(bench_px.get(tk, 0) * (px_at(tk, d) or 0) for tk in bench_px)
        rows.append({"date": d, "equity": book / start_cash, "n": len(holds)})
        brows.append({"date": d, "bench": bval if bench_px else 1.0})
    eq = pd.DataFrame(rows)
    bench = pd.DataFrame(brows)
    return eq, bench
