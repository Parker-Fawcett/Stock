"""Backtest: monthly rebalance, cash by default, costs from day one.

Ledger design (rewritten Sep 2026 after external review):
- SIMPLE returns everywhere. Equal weight = mean of simple returns;
  equity compounds as product of (1+R). Averaging log returns then
  exponentiating understated dispersion (+100%/-50% across two names
  is +25%, not 0%).
- Stops are gap-aware: exit at the FIRST close beyond the stop, at that
  actual close — never credited at exactly the threshold. Short stops
  are losses (sign was wrong before; verified).
- stop_frac is validated to (0, 1). Values above 1 used to credit
  log(2) monthly on flat prices. Structurally impossible now.
- Exposure scaling (vol target, drawdown brake) scales WEIGHTS with the
  remainder in cash — it used to cut names while remaining names kept
  full capital, which concentrated instead of de-risking.
- Turnover = half the sum of absolute weight changes, so sizing shifts
  and stop exits cost. Monkeys use identical accounting.
- Holds are 20 trading days on a ~21-day monthly grid, so overlap is
  ~1 day by construction. Fold-end windows truncate; documented.

Monkey baseline every run (his best habit): N_MONKEYS random
portfolios, same dates/sizing/costs, mean/std bands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import HORIZON

TOP_N = 10
# Labels are cross-sectional top-20% (base rate 0.20), so calibrated
# proba centers near 0.2. Gate at 0.30 = only buy when meaningfully
# above baseline confidence. His V5 used ~30% the same way.
THRESH = 0.30
COST_BPS = 5.0  # one-way, on traded notional
N_MONKEYS = 200
REBAL_FREQ = "M"

# Risk defaults: tame drawdowns, not chase return.
RISK_TOP_N = 20
RISK_VOL_TARGET = 0.20   # annualized vol target on universe regime
RISK_MIN_NAMES = 10      # breadth requirement: fewer qualifiers -> cash
RISK_STOP = 0.85         # per-name stop: exit at first close <= 85% of entry
RISK_DD_BRAKE = 0.15     # halve exposure while trailing DD worse than this


def _month_keys(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.to_period(REBAL_FREQ)


def _check_stop(stop_frac: float | None) -> None:
    if stop_frac is not None and not (0.0 < stop_frac < 1.0):
        raise ValueError(f"stop_frac must be in (0,1) or None, got {stop_frac}")


def weight_turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    """Portfolio turnover under the repository's half-L1 convention."""
    keys = set(previous) | set(current)
    return 0.5 * sum(abs(current.get(k, 0.0) - previous.get(k, 0.0))
                     for k in keys)


def equal_weight_turnover(previous, current) -> float:
    """Turnover between two equally weighted security lists."""
    old = list(dict.fromkeys(previous))
    new = list(dict.fromkeys(current))
    old_w = {tk: 1.0 / len(old) for tk in old} if old else {}
    new_w = {tk: 1.0 / len(new) for tk in new} if new else {}
    return weight_turnover(old_w, new_w)


def leg_simple(px_col, d0, horizon: int, short: bool,
               stop_frac: float | None) -> float:
    """Simple return of one position held up to `horizon` trading days.
    Gap-aware stops: exit at the first close beyond the stop, realized
    at that actual close."""
    _check_stop(stop_frac)
    p0 = float(px_col.loc[d0])
    fut = px_col.loc[px_col.index > d0].iloc[:horizon]
    if len(fut) == 0:
        return 0.0
    if stop_frac is not None:
        if short:
            hit = fut[fut / p0 >= 1 / stop_frac]
            if len(hit):
                return float(p0 / hit.iloc[0] - 1)  # stopped: a loss
            return float(p0 / fut.iloc[-1] - 1)
        hit = fut[fut / p0 <= stop_frac]
        if len(hit):
            return float(hit.iloc[0] / p0 - 1)  # actual close, gaps included
        return float(fut.iloc[-1] / p0 - 1)
    if short:
        return float(p0 / fut.iloc[-1] - 1)
    return float(fut.iloc[-1] / p0 - 1)


def _run_book(test: pd.DataFrame, schedule, cost_bps: float = COST_BPS,
              borrow_apr: float = 0.0, horizon: int = HORIZON,
              stop_long: float | None = None,
              stop_short: float | None = None) -> pd.DataFrame:
    """Core ledger. schedule(d0, cands, state) -> (longs {tk: w}, shorts {tk: w})
    with weights as fractions of equity (rest = cash). Stops passed
    explicitly so no runner can silently disable them."""
    _check_stop(stop_long)
    _check_stop(stop_short)
    t = test.copy().reset_index(drop=True)
    t["month"] = _month_keys(t["Date"])
    months = t["month"].drop_duplicates().sort_values()
    px = t.set_index(["ticker", "Date"])["Close"].unstack("ticker")
    rows = []
    prev_w: dict = {}
    state = {"equity": 1.0, "peak": 1.0}
    for m in months:
        block = t[t["month"] == m]
        d0 = block["Date"].max()
        cands = block[block["Date"] == d0]
        longs, shorts = schedule(d0, cands, state)
        w = {tk: wt for tk, wt in longs.items()}
        w.update({tk: -wt for tk, wt in shorts.items()})
        turnover = weight_turnover(prev_w, w)
        cost = turnover * cost_bps / 1e4
        borrow = sum(shorts.values()) * borrow_apr / 12 if shorts else 0.0
        gross = 0.0
        for tk, wt in longs.items():
            try:
                gross += wt * leg_simple(px[tk], d0, horizon, False, stop_long)
            except Exception:  # noqa: BLE001
                continue
        for tk, wt in shorts.items():
            try:
                gross += wt * leg_simple(px[tk], d0, horizon, True, stop_short)
            except Exception:  # noqa: BLE001
                continue
        net = gross - cost - borrow
        rows.append({"month": str(m), "date": d0,
                     "n": len(longs) + len(shorts),
                     "gross": gross, "cost": cost + borrow, "net": net,
                     "turnover": turnover})
        state["equity"] *= 1 + net
        state["peak"] = max(state["peak"], state["equity"])
        prev_w = dict(w)
    out = pd.DataFrame(rows)
    out["equity"] = (1 + out["net"]).cumprod()
    return out


def _prepared(test: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    t = test.copy().reset_index(drop=True)
    if len(t) != len(proba):
        raise ValueError("proba must have one value per test row")
    t["proba"] = np.asarray(proba, dtype=float)
    return t


def run_backtest(
    test: pd.DataFrame, proba: np.ndarray, top_n: int = TOP_N,
    thresh: float = THRESH, cost_bps: float = COST_BPS,
) -> pd.DataFrame:
    """Long-only, equal-weight, monthly model portfolio."""
    t = _prepared(test, proba)

    def schedule(_d0, cands, _state):
        picks = (cands[cands["proba"] >= thresh]
                 .sort_values("proba", ascending=False)
                 .head(top_n)["ticker"].tolist())
        w = 1.0 / len(picks) if picks else 0.0
        return ({tk: w for tk in picks}, {})

    return _run_book(t, schedule, cost_bps=cost_bps)


def run_backtest_risk(
    test: pd.DataFrame, proba: np.ndarray, top_n: int = RISK_TOP_N,
    thresh: float = THRESH, cost_bps: float = COST_BPS,
    vol_target: float = RISK_VOL_TARGET, min_names: int = RISK_MIN_NAMES,
    stop_frac: float | None = RISK_STOP, dd_brake: float = RISK_DD_BRAKE,
) -> pd.DataFrame:
    """Long-only ledger with breadth, volatility, and drawdown controls."""
    _check_stop(stop_frac)
    t = _prepared(test, proba)
    px = t.set_index(["ticker", "Date"])["Close"].unstack("ticker")
    uret = px.pct_change(fill_method=None).mean(axis=1).dropna()
    vol = uret.rolling(60, min_periods=20).std() * np.sqrt(252)

    def schedule(d0, cands, state):
        quals = (cands[cands["proba"] >= thresh]
                 .sort_values("proba", ascending=False))
        if len(quals) < min_names:
            return {}, {}
        hist = vol.loc[vol.index <= d0]
        trailing = float(hist.iloc[-1]) if len(hist) else np.nan
        exposure = min(1.0, vol_target / trailing) if trailing > 0 else 1.0
        if state["equity"] / state["peak"] - 1 < -dd_brake:
            exposure *= 0.5
        picks = quals.head(top_n)["ticker"].tolist()
        return ({tk: exposure / len(picks) for tk in picks}, {})

    return _run_book(t, schedule, cost_bps=cost_bps, stop_long=stop_frac)


LS_TOP_N = 10
LS_BOT_N = 10
LS_BORROW_APR = 0.05


def run_backtest_ls(
    test: pd.DataFrame, proba: np.ndarray, top_n: int = LS_TOP_N,
    bot_n: int = LS_BOT_N, thresh: float = THRESH,
    cost_bps: float = COST_BPS, borrow_apr: float = LS_BORROW_APR,
    stop_frac: float | None = RISK_STOP,
) -> pd.DataFrame:
    """Dollar-neutral long/short book with equal 50% long and short sleeves."""
    _check_stop(stop_frac)
    t = _prepared(test, proba)

    def schedule(_d0, cands, _state):
        ranked = cands.sort_values("proba", ascending=False)
        longs = ranked[ranked["proba"] >= thresh].head(top_n)["ticker"].tolist()
        shorts = (ranked[ranked["proba"] <= 1 - thresh]
                  .tail(bot_n)["ticker"].tolist())
        longs = [tk for tk in longs if tk not in shorts]
        if not longs or not shorts:
            return {}, {}
        return ({tk: 0.5 / len(longs) for tk in longs},
                {tk: 0.5 / len(shorts) for tk in shorts})

    return _run_book(t, schedule, cost_bps=cost_bps, borrow_apr=borrow_apr,
                     stop_long=stop_frac, stop_short=stop_frac)


def run_monkeys(
    test: pd.DataFrame, top_n: int = TOP_N, cost_bps: float = COST_BPS,
    n: int = N_MONKEYS, seed: int = 0,
) -> pd.DataFrame:
    """Random long-only portfolios using the same ledger and turnover costs."""
    rng = np.random.default_rng(seed)
    endpoints = test.copy()
    endpoints["month"] = _month_keys(endpoints["Date"])
    eq = []
    for _ in range(n):
        def schedule(_d0, cands, _state):
            avail = cands["ticker"].drop_duplicates().to_numpy()
            picks = rng.choice(avail, size=min(top_n, len(avail)), replace=False)
            w = 1.0 / len(picks) if len(picks) else 0.0
            return ({tk: w for tk in picks}, {})
        eq.append(float(_run_book(endpoints, schedule, cost_bps=cost_bps)
                        ["equity"].iloc[-1]))
    s = pd.Series(eq)
    return pd.DataFrame([{"monkey_mean": s.mean(), "monkey_std": s.std(),
                          "monkey_p5": s.quantile(.05),
                          "monkey_p95": s.quantile(.95)}])
