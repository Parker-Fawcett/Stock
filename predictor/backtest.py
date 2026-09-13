"""Backtest: monthly rebalance, cash by default, costs from day one.

Fixes vs his series:
- His early bot HAD to buy daily (1F0gYkk7YYw). Here cash is default;
  we hold at most TOP_N names with proba above THRESH, else cash.
- His live test assumed open-to-close fills he couldn't make with a day
  job (Lh1vrIcpJN4). Here rebalance is monthly on closes, with COST_BPS
  + slippage charged on every change.
- Monkey baseline every run (his best habit): N_MONKEYS random
  portfolios, same dates/sizing, mean/std bands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_COLS, HORIZON

TOP_N = 10
# Labels are cross-sectional top-20% (base rate 0.20), so calibrated
# proba centers near 0.2. Gate at 0.30 = only buy when meaningfully
# above baseline confidence. His V5 used ~30% the same way.
THRESH = 0.30
COST_BPS = 5.0  # one-way
N_MONKEYS = 200
REBAL_FREQ = "M"

# Risk defaults: tame the -67% maxDD, not chase return.
RISK_TOP_N = 20
RISK_VOL_TARGET = 0.20   # annualized vol target on universe regime
RISK_MIN_NAMES = 10      # breadth requirement: fewer qualifiers -> cash
RISK_STOP = 0.85         # per-name stop: exit if close falls 15% below entry
RISK_DD_BRAKE = 0.15     # halve exposure while trailing DD worse than this


def _month_keys(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.to_period(REBAL_FREQ)


def run_backtest(
    test: pd.DataFrame,
    proba: np.ndarray,
    top_n: int = TOP_N,
    thresh: float = THRESH,
    cost_bps: float = COST_BPS,
) -> pd.DataFrame:
    t = test.copy().reset_index(drop=True)
    t["proba"] = np.asarray(proba, dtype=float)
    t["month"] = _month_keys(t["Date"])
    months = t["month"].drop_duplicates().sort_values()
    # forward 20d total return per row (close t -> t+20)
    px = t.set_index(["ticker", "Date"])["Close"].unstack("ticker")
    fwd = {}
    for m in months:
        block = t[t["month"] == m]
        d0 = block["Date"].max()  # decide on last observed date of month
        cands = block[block["Date"] == d0].sort_values("proba", ascending=False)
        picks = cands[cands["proba"] >= thresh].head(top_n)["ticker"].tolist()
        fwd[m] = (d0, picks)
    rows = []
    prev: list[str] = []
    for m in months:
        d0, picks = fwd[m]
        turnover = len(set(picks) ^ set(prev)) / max(1, max(len(picks), len(prev)))
        cost = turnover * cost_bps / 1e4
        # realized: equal-weight picks held HORIZON days from d0
        if picks:
            rets = []
            for tk in picks:
                try:
                    p0 = px.loc[d0, tk]
                    future = px[tk].loc[px.index > d0]
                    p1 = future.iloc[min(HORIZON - 1, len(future) - 1)]
                    rets.append(float(np.log(p1 / p0)))
                except Exception:  # noqa: BLE001
                    continue
            gross = float(np.mean(rets)) if rets else 0.0
        else:
            gross = 0.0  # cash
        rows.append({"month": str(m), "date": d0, "n": len(picks),
                     "gross": gross, "cost": cost, "net": gross - cost,
                     "turnover": turnover})
        prev = picks
    out = pd.DataFrame(rows)
    out["equity"] = np.exp(out["net"].cumsum())
    return out


def run_monkeys(
    test: pd.DataFrame,
    top_n: int = TOP_N,
    cost_bps: float = COST_BPS,
    n: int = N_MONKEYS,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = test.copy().reset_index(drop=True)
    t["month"] = _month_keys(t["Date"])
    months = t["month"].drop_duplicates().sort_values()
    px = t.set_index(["ticker", "Date"])["Close"].unstack("ticker")
    eq = {}
    for i in range(n):
        rets = []
        for m in months:
            block = t[t["month"] == m]
            d0 = block["Date"].max()
            avail = block[block["Date"] == d0]["ticker"].unique().tolist()
            k = min(top_n, len(avail))
            picks = rng.choice(avail, size=k, replace=False).tolist() if k else []
            if picks:
                rs = []
                for tk in picks:
                    try:
                        p0 = px.loc[d0, tk]
                        future = px[tk].loc[px.index > d0]
                        p1 = future.iloc[min(HORIZON - 1, len(future) - 1)]
                        rs.append(float(np.log(p1 / p0)))
                    except Exception:  # noqa: BLE001
                        continue
                gross = float(np.mean(rs)) if rs else 0.0
            else:
                gross = 0.0
            cost = 1.0 * cost_bps / 1e4  # full turnover assumption, conservative
            rets.append(gross - cost)
        eq[i] = float(np.exp(np.cumsum(rets)[-1])) if rets else 1.0
    s = pd.Series(eq)
    return pd.DataFrame([{"monkey_mean": s.mean(), "monkey_std": s.std(),
                           "monkey_p5": s.quantile(0.05),
                           "monkey_p95": s.quantile(0.95)}])


def run_backtest_risk(
    test: pd.DataFrame,
    proba: np.ndarray,
    top_n: int = RISK_TOP_N,
    thresh: float = THRESH,
    cost_bps: float = COST_BPS,
    vol_target: float = RISK_VOL_TARGET,
    min_names: int = RISK_MIN_NAMES,
    stop_frac: float = RISK_STOP,
    dd_brake: float = RISK_DD_BRAKE,
) -> pd.DataFrame:
    """Risk-controlled version. Same signal, breadth + stops, all trailing:

    1. Breadth: fewer than min_names qualifiers -> cash. Concentrated
       1-2 stock books caused the worst blowups, not the market.
    2. Per-name stop: close-based stop at stop_frac of entry (stop order).
    3. Vol target: scale names by vol_target / trailing-60d universe vol.
    4. Drawdown brake: halve names while running DD < -dd_brake.
    """
    t = test.copy().reset_index(drop=True)
    t["proba"] = np.asarray(proba, dtype=float)
    t["month"] = _month_keys(t["Date"])
    months = t["month"].drop_duplicates().sort_values()
    px = t.set_index(["ticker", "Date"])["Close"].unstack("ticker")
    # trailing regime vol from universe equal-weight daily returns
    uret = np.log(px / px.shift(1)).mean(axis=1).dropna()
    vol = uret.rolling(60, min_periods=20).std() * np.sqrt(252)

    def stopped_ret(tk: str, d0) -> float:
        p0 = px.loc[d0, tk]
        future = px[tk].loc[px.index > d0].iloc[:HORIZON]
        if len(future) == 0:
            return 0.0
        hit = future[future / p0 <= stop_frac]
        if len(hit):
            return float(np.log(stop_frac))  # stopped out, close-based
        return float(np.log(future.iloc[-1] / p0))

    rows = []
    prev: list[str] = []
    eq = 1.0
    peak = 1.0
    for m in months:
        block = t[t["month"] == m]
        d0 = block["Date"].max()
        cands = block[block["Date"] == d0].sort_values("proba", ascending=False)
        quals = cands[cands["proba"] >= thresh]
        if len(quals) < min_names:
            picks = []  # no breadth -> cash
        else:
            hist = vol.loc[vol.index <= d0]
            trailing = float(hist.iloc[-1]) if len(hist) else float("nan")
            scale = min(1.0, vol_target / trailing) if trailing > 0 else 1.0
            dd = eq / peak - 1.0
            if dd < -dd_brake:
                scale *= 0.5
            n = max(0, round(top_n * scale))
            picks = (quals.head(n)["ticker"].tolist() if n >= min_names else [])
        turnover = len(set(picks) ^ set(prev)) / max(1, max(len(picks), len(prev)))
        cost = turnover * cost_bps / 1e4
        if picks:
            rets = []
            for tk in picks:
                try:
                    rets.append(stopped_ret(tk, d0))
                except Exception:  # noqa: BLE001
                    continue
            gross = float(np.mean(rets)) if rets else 0.0
        else:
            gross = 0.0
        net = gross - cost
        eq *= np.exp(net)
        peak = max(peak, eq)
        rows.append({"month": str(m), "date": d0, "n": len(picks),
                     "gross": gross, "cost": cost, "net": net,
                     "turnover": turnover})
        prev = picks
    out = pd.DataFrame(rows)
    out["equity"] = np.exp(out["net"].cumsum())
    return out


# Long-short defaults. Small-cap borrow is expensive and sometimes
# unavailable; 5%/yr + full turnover costs is deliberately harsh.
LS_TOP_N = 10
LS_BOT_N = 10
LS_BORROW_APR = 0.05


def run_backtest_ls(
    test: pd.DataFrame,
    proba: np.ndarray,
    top_n: int = LS_TOP_N,
    bot_n: int = LS_BOT_N,
    thresh: float = THRESH,
    cost_bps: float = COST_BPS,
    borrow_apr: float = LS_BORROW_APR,
) -> pd.DataFrame:
    """Dollar-neutral: long top-N above thresh, short bottom-N below
    (1 - thresh). Same 20d hold, stops on both legs, borrow accrued
    monthly on the short book. Cuts market beta; borrow + turnover
    decide if anything is left."""
    t = test.copy().reset_index(drop=True)
    t["proba"] = np.asarray(proba, dtype=float)
    t["month"] = _month_keys(t["Date"])
    months = t["month"].drop_duplicates().sort_values()
    px = t.set_index(["ticker", "Date"])["Close"].unstack("ticker")

    def leg_ret(tk: str, d0, short: bool) -> float:
        p0 = px.loc[d0, tk]
        future = px[tk].loc[px.index > d0].iloc[:HORIZON]
        if len(future) == 0:
            return 0.0
        if short:
            hit = future[future / p0 >= 1 / RISK_STOP]
            r = float(np.log(1 / RISK_STOP)) if len(hit) else float(
                np.log(p0 / future.iloc[-1]))
        else:
            hit = future[future / p0 <= RISK_STOP]
            r = float(np.log(RISK_STOP)) if len(hit) else float(
                np.log(future.iloc[-1] / p0))
        return r

    rows = []
    prev_l: list[str] = []
    prev_s: list[str] = []
    for m in months:
        block = t[t["month"] == m]
        d0 = block["Date"].max()
        cands = block[block["Date"] == d0].sort_values("proba", ascending=False)
        longs = cands[cands["proba"] >= thresh].head(top_n)["ticker"].tolist()
        shorts = cands[cands["proba"] <= 1 - thresh].tail(bot_n)["ticker"].tolist()
        longs = [x for x in longs if x not in shorts]
        if not longs or not shorts:
            longs, shorts = [], []  # no pair trade -> cash
        tl = len(set(longs) ^ set(prev_l)) / max(1, max(len(longs), len(prev_l)))
        ts = len(set(shorts) ^ set(prev_s)) / max(1, max(len(shorts), len(prev_s)))
        cost = (tl + ts) * 0.5 * cost_bps / 1e4
        borrow = (len(shorts) > 0) * borrow_apr / 12 * 0.5
        if longs:
            rl, rs = [], []
            for tk in longs:
                try:
                    rl.append(leg_ret(tk, d0, False))
                except Exception:  # noqa: BLE001
                    continue
            for tk in shorts:
                try:
                    rs.append(leg_ret(tk, d0, True))
                except Exception:  # noqa: BLE001
                    continue
            gross = 0.5 * ((float(np.mean(rl)) if rl else 0.0)
                           + (float(np.mean(rs)) if rs else 0.0))
        else:
            gross = 0.0
        net = gross - cost - borrow
        rows.append({"month": str(m), "date": d0,
                     "n": len(longs) + len(shorts),
                     "gross": gross, "cost": cost + borrow, "net": net,
                     "turnover": (tl + ts) / 2})
        prev_l, prev_s = longs, shorts
    out = pd.DataFrame(rows)
    out["equity"] = np.exp(out["net"].cumsum())
    return out
