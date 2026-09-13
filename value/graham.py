"""Graham intrinsic value, his formula, validated by hand.

His method (oJQqiogr6S0, SgQ-RwC97us):
  g    = slope of log quarterly EPS over ~30 quarters (7y)
  IV   = EPS_TTM * (8.5 + 2*g*100) * 4.4 / AAA_yield
  buy  if price <= (1 - discount) * IV   (discount default 0.5)
  sell at IV or after 3 years.

AAA default 4.4 = Graham's original constant (his code downloaded the
live AAA table; without a FRED key we use the constant and SAY so).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AAA_DEFAULT = 4.4
MIN_QUARTERS = 12


def growth_rate(eps_hist: pd.Series, ppy: int = 4) -> float:
    """Slope of log(EPS) on time, annualized. ppy = periods per year.
    Positive EPS required; else 0."""
    e = eps_hist.dropna().astype(float).tail(30 if ppy == 4 else 10)
    need = 3 * ppy + (2 if ppy == 1 else 0)  # 12 quarters or 5 years
    if len(e) < need or (e <= 0).any():
        return 0.0
    x = np.arange(len(e))
    slope = np.polyfit(x, np.log(e.values), 1)[0]
    return float(np.clip(slope * ppy, -0.5, 0.5))


def graham_iv(eps_ttm: float, g: float, aaa: float = AAA_DEFAULT) -> float:
    if eps_ttm is None or eps_ttm <= 0:
        return float("nan")
    mult = 8.5 + 2 * g * 100
    if mult <= 0:
        return float("nan")
    return float(eps_ttm * mult * 4.4 / aaa)


def graham_number(eps_ttm: float, bvps: float) -> float:
    if eps_ttm is None or bvps is None or eps_ttm <= 0 or bvps <= 0:
        return float("nan")
    return float(np.sqrt(22.5 * eps_ttm * bvps))


def compute_iv_panel(fund: pd.DataFrame, aaa: float = AAA_DEFAULT) -> pd.DataFrame:
    """One IV row per filing, using only that filing's trailing history.
    Annual vs quarterly inferred from median filing gap."""
    out = []
    for tk, f in fund.sort_values("filed_date").groupby("ticker", sort=False):
        f = f.reset_index(drop=True)
        gaps = f["filed_date"].diff().dt.days.median()
        ppy = 4 if (pd.isna(gaps) or gaps < 200) else 1
        for i, r in f.iterrows():
            g = growth_rate(f.loc[:i, "qeps"], ppy) if "qeps" in f else 0.0
            out.append({"ticker": tk, "filed_date": r["filed_date"],
                        "period_end": r["period_end"], "eps_ttm": r["eps_ttm"],
                        "bvps": r["bvps"], "g": g,
                        "iv": graham_iv(r["eps_ttm"], g, aaa),
                        "graham_num": graham_number(r["eps_ttm"], r["bvps"])})
    return pd.DataFrame(out)
