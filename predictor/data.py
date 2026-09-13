"""Free price downloader (Yahoo chart API -> local CSV cache).

Survivorship warning, from LosingLoonies finale (noK0IwZAnyE):
 tickers that exist TODAY hide bankrupt losers. Anything run on this
 free data is BIASED upward. Use a point-in-time membership list +
 delisted prices (e.g. Massive/CRSP) before trusting a number.
 This module is for scaffolding, not for claiming edge.
"""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _to_ts(s: str) -> int:
    d = dt.datetime.strptime(s, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp())


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")


def download_yahoo(ticker: str, start: str = "20100101",
                   end: str | None = None, refresh: bool = False) -> Path:
    if end is None:
        end = _today()
    out = RAW_DIR / f"{ticker.upper()}.csv"
    if out.exists() and not refresh and out.stat().st_size > 5000:
        # validate cache is really csv, not an error page
        head = out.read_bytes()[:15]
        if head.startswith(b"Date,"):
            return out
        out.unlink()
    p1, p2 = _to_ts(start), _to_ts(end)
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker.upper()}"
           f"?period1={p1}&period2={p2}&interval=1d&events=div%7Csplit")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.loads(r.read().decode())
    res = payload["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    adj = None
    try:
        adj = res["indicators"]["adjclose"][0]["adjclose"]
    except Exception:  # noqa: BLE001
        adj = None
    import csv

    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
        for i, t in enumerate(ts):
            o, h, l, c, v = q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i]
            if c is None:
                continue
            if adj is not None and adj[i] and q["close"][i]:
                factor = adj[i] / q["close"][i]
                o, h, l, c = (x * factor if x else x for x in (o, h, l, c))
            d = dt.datetime.fromtimestamp(t, tz=dt.timezone.utc).strftime("%Y-%m-%d")
            w.writerow([d, o, h, l, c, v or 0])
    time.sleep(0.4)
    return out


def download_universe(tickers: list[str], market: str = "SPY",
                      start: str = "20100101", end: str | None = None,
                      refresh: bool = False) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for t in [market, *[x for x in tickers if x.upper() != market.upper()]]:
        try:
            paths[t.upper()] = download_yahoo(t, start=start, end=end,
                                              refresh=refresh)
        except Exception as e:  # noqa: BLE001 - free data is flaky, keep going
            print(f"download failed {t}: {e}")
    return paths
