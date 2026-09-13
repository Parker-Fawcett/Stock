"""Trailing-only features + 20d forward excess labels. No lookahead.

Lessons baked in (from the source project's public build log):
- V1 leaked via overlapping windows + stock-split (aiFpAl3mgGk). All
  features here use only data <= date t (shifted), labels use t+20.
- MSE on 1-day log returns learns the mean (kRa3PUxNBTM). We use
  cross-sectional rank labels on 20-day excess vs SPY instead.
- Famous macro (VIX/rates/QE) got priced in (t2f0vyfABdM). Features
  below are price/volume only; strange data plugs in via add_alpha().
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 20  # trading days forward
TOP_Q = 0.20  # top 20% per date = 1


def load_ohlc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().capitalize() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"].astype(float)
    logp = np.log(c)
    d["ret_1"] = logp.diff(1)
    d["ret_5"] = logp.diff(5)
    d["ret_20"] = logp.diff(20)
    d["mom_20"] = c / c.shift(20) - 1.0
    d["vol_20"] = d["ret_1"].rolling(20).std()
    d["range"] = (d["High"] - d["Low"]) / c.shift(1)
    d["range_20"] = d["range"].rolling(20).mean()
    d["vol_chg"] = d["Volume"] / d["Volume"].rolling(20).mean() - 1.0
    delta = c.diff(1)
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    d["ma10_gap"] = c / c.rolling(10).mean() - 1.0
    d["ma50_gap"] = c / c.rolling(50).mean() - 1.0
    return d


def add_alpha(df: pd.DataFrame, alpha: pd.Series | None) -> pd.DataFrame:
    """Plug strange data here (e.g. Glassdoor flow, insider cluster).
    Must be indexed by date, trailing only. None = skip."""
    d = df.copy()
    if alpha is not None:
        a = alpha.reindex(d["Date"].values if "Date" in d else None)
        d["alpha_1"] = np.asarray(a, dtype=float)
    return d


FEATURE_COLS = [
    "ret_1", "ret_5", "ret_20", "mom_20", "vol_20",
    "range", "range_20", "vol_chg", "rsi_14", "ma10_gap", "ma50_gap",
]

# Strange-data columns ( predictor/insider.py ). Present only when the
# monthly Form 4 merge has run; model uses whichever exist.
# Single source of truth lives in insider.py (imported, not copied —
# a stale copy here once silently dropped the conviction features).
try:
    from predictor.insider import INS_COLS
except Exception:  # noqa: BLE001 - insider optional at import time
    INS_COLS = ["ins_missing"]


def model_cols(df) -> list:
    return [c for c in FEATURE_COLS + INS_COLS if c in df.columns]


def build_panel(price_paths: dict[str, str], market: str = "SPY") -> pd.DataFrame:
    frames = []
    for ticker, path in price_paths.items():
        try:
            df = add_features(load_ohlc(path))
        except Exception as e:  # noqa: BLE001
            print(f"skip {ticker}: {e}")
            continue
        df["ticker"] = ticker.upper()
        frames.append(df[["Date", "ticker", "Close"] + FEATURE_COLS])
    panel = pd.concat(frames, ignore_index=True)
    mkt_path = price_paths.get(market.upper())
    if mkt_path is None:
        raise ValueError(f"market {market} missing from price_paths")
    mkt = load_ohlc(mkt_path)[["Date", "Close"]].rename(columns={"Close": "mkt"})
    panel = panel.merge(mkt, on="Date", how="left")
    panel = panel.sort_values(["ticker", "Date"]).reset_index(drop=True)
    g = panel.groupby("ticker", group_keys=False)
    panel["fwd_ret"] = g["Close"].apply(lambda s: np.log(s.shift(-HORIZON) / s))
    # market forward return aligned by date (same shift for every ticker)
    mkt_log = np.log(mkt.set_index("Date")["mkt"])
    panel["mkt_fwd"] = panel["Date"].map(mkt_log.shift(-HORIZON) - mkt_log).astype(float)
    panel["excess"] = panel["fwd_ret"] - panel["mkt_fwd"]
    panel["label_end"] = panel["Date"] + pd.to_timedelta(HORIZON, unit="D")
    # cross-sectional label: top 20% excess per date = 1
    panel["label"] = (
        panel.groupby("Date")["excess"]
        .transform(lambda s: (s.rank(pct=True) >= 1 - TOP_Q).astype(float))
    )
    panel = panel.dropna(subset=FEATURE_COLS + ["excess"]).reset_index(drop=True)
    return panel
