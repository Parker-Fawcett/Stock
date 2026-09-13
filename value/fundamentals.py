"""Point-in-time fundamentals. Interface + fixture, no lookahead.

The free-data situation, verified Sep 2026:
- SEC Archives / EDGAR JSON / bulk statement ZIPs: WAF-blocked (403).
- French factor library: reachable, used for regime check only.
- Real quarterly history needs a key (FMP free tier works):
  set FMP_KEY and use load_fmp(). One-line plug, validated below.

His lesson (oJQqiogr6S0): validate the math by hand on ONE company
before scaling. test_value.py does exactly that with the fixture.
"""
from __future__ import annotations

import os
import urllib.parse
import urllib.request
import json
from pathlib import Path

import pandas as pd

COLS = ["ticker", "filed_date", "period_end", "qeps", "eps_ttm", "bvps"]


def load_csv(path: str | Path) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["filed_date", "period_end"])
    assert set(COLS) <= set(d.columns), f"need {COLS}"
    return d.sort_values(["ticker", "filed_date"]).reset_index(drop=True)


def load_fmp(tickers: list[str], years: int = 12) -> pd.DataFrame:
    """Annual EPS + book value per share, filed-date stamped.
    Free plan covers annual statements only (quarterly is paywalled, 402).
    2 calls per ticker — ~50 names/day on the 250/day free plan."""
    key = os.environ.get("FMP_KEY", "")
    if not key:
        raise RuntimeError("set FMP_KEY (free at site financialmodelingprep.com)")
    rows = []
    for t in tickers:
        tk = t.upper()
        try:
            iq = urllib.parse.urlencode({"symbol": tk, "period": "annual",
                                         "limit": years, "apikey": key})
            url = f"https://financialmodelingprep.com/stable/income-statement?{iq}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            inc = json.loads(urllib.request.urlopen(req, timeout=30).read())
            bq = urllib.parse.urlencode({"symbol": tk, "period": "annual",
                                         "limit": years, "apikey": key})
            url = f"https://financialmodelingprep.com/stable/balance-sheet-statement?{bq}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            bal = json.loads(urllib.request.urlopen(req, timeout=30).read())
        except Exception as e:  # noqa: BLE001
            print(f"FMP failed {tk}: {e}")
            continue
        import time
        time.sleep(1.2)  # free-tier rate limit is tight; unhurried wins
        bmap = {r.get("date"): r for r in bal if isinstance(r, dict)}
        for r in inc:
            if not isinstance(r, dict):
                continue
            b = bmap.get(r.get("date"), {})
            eq = b.get("totalStockholdersEquity")
            sh = b.get("weightedAverageShsOutDil") or r.get("weightedAverageShsOutDil")
            try:
                bvps = float(eq) / float(sh) if eq and sh else None
            except Exception:  # noqa: BLE001
                bvps = None
            qeps = r.get("epsDiluted")
            rows.append({"ticker": tk, "filed_date": r.get("filingDate"),
                         "period_end": r.get("date"), "qeps": qeps,
                         "eps_ttm": qeps, "bvps": bvps})
        import time
        time.sleep(0.5)
    f = pd.DataFrame(rows)
    if f.empty:
        print("no fundamentals fetched (key quota?); returning empty")
        return f
    f["filed_date"] = pd.to_datetime(f["filed_date"])
    f["period_end"] = pd.to_datetime(f["period_end"])
    for c in ("qeps", "eps_ttm", "bvps"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f = f.sort_values(["ticker", "period_end"])
    return f[COLS].dropna(subset=["eps_ttm", "bvps"]).reset_index(drop=True)
