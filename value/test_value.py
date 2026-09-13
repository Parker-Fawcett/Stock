"""Hand-validation: one company, spreadsheet-checkable numbers.

Mirrors his ABBV video (oJQqiogr6S0): compute by hand, then assert the
code agrees. Quarterly EPS grows exactly 2%/qtr from $1.00:
  g = ln(1.02)*4 = 0.0793/yr, TTM EPS = 4.9257, mult = 24.36, IV = 120.0
  Graham number with BVPS 20 = 47.08
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .graham import compute_iv_panel, graham_iv, graham_number, growth_rate

QEPS = [round(1.00 * 1.02 ** i, 4) for i in range(13)]
G_EXPECT = round(np.log(1.02) * 4, 4)          # 0.0793
TTM_EXPECT = round(sum(QEPS[-4:]), 4)          # 4.9257
IV_EXPECT = round(TTM_EXPECT * (8.5 + 2 * G_EXPECT * 100), 2)  # 120.0
GN_EXPECT = round(float(np.sqrt(22.5 * TTM_EXPECT * 20.0)), 2)  # 47.08


def fixture() -> pd.DataFrame:
    rows = []
    d = pd.Timestamp("2020-01-01")
    for i, e in enumerate(QEPS):
        fd = d + pd.to_timedelta(45 + i * 91, unit="D")
        pe = d + pd.to_timedelta(i * 91, unit="D")
        rows.append({"ticker": "HAND", "filed_date": fd, "period_end": pe,
                     "qeps": e,
                     "eps_ttm": round(sum(QEPS[max(0, i - 3):i + 1]), 4),
                     "bvps": 20.0})
    return pd.DataFrame(rows)


def test_hand() -> None:
    f = fixture()
    assert abs(growth_rate(f["qeps"]) - G_EXPECT) < 1e-3, growth_rate(f["qeps"])
    assert abs(f.iloc[-1]["eps_ttm"] - TTM_EXPECT) < 1e-3
    iv = graham_iv(TTM_EXPECT, G_EXPECT)
    assert abs(iv - IV_EXPECT) < 0.05, (iv, IV_EXPECT)
    assert abs(graham_number(TTM_EXPECT, 20.0) - GN_EXPECT) < 0.05
    panel = compute_iv_panel(f)
    assert abs(panel.iloc[-1]["iv"] - IV_EXPECT) < 0.05
    # early rows lack 12 quarters -> g=0, still defined via TTM
    assert panel["iv"].notna().sum() > 0
    print(f"hand-check OK: g={G_EXPECT} ttm={TTM_EXPECT} IV={IV_EXPECT} GN={GN_EXPECT}")


if __name__ == "__main__":
    test_hand()
