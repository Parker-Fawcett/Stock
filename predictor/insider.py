"""SEC Form 4 insider features via OpenInsider (no API key, reachable).

Why this source: SEC Archives/EDGAR JSON are WAF-blocked from here and
Quiver needs $30/mo + token. OpenInsider republishes Form 4s with
filing date, trade date, P/S direction, price, qty.

Lookahead discipline (his Congress lesson LvUcTBmVCXI):
 Form 4s legally lag trades by ~2 business days. Everything is keyed on
 FILING date, and monthly buckets become available month_end + 3 days.
 Never trade date.
"""
from __future__ import annotations

import html
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
RAW = Path(__file__).resolve().parent.parent / "data" / "insider" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

WINDOWS = [
    {"fd": "-1", "fdr": "01/01/2019 - 01/01/2022"},  # custom range, pre-4y
    {"fd": "1461"},  # last 4 years
]

INS_COLS = [
    "ins_conv_3m", "ins_convval_3m", "ins_missing",
]

CSUITE = ("CEO", "CFO", "COO", "PRESIDENT")


def _is_csuite(title: str) -> bool:
    t = str(title).upper()
    if "DIR" in t and not any(k in t for k in CSUITE):
        return False  # pure directors excluded
    return any(k in t for k in CSUITE)


def monthly_conviction(df: pd.DataFrame) -> pd.DataFrame:
    """Conviction buys only: C-suite open-market buy, >$50k, first in 12mo.
    Literature: presence of such buys predicts; raw counts do not."""
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    buys = d[(d["side"] == "P") & d["title"].map(_is_csuite)
             & (d["sval"] > 50000)].sort_values("filing_dt")
    if buys.empty:
        return pd.DataFrame(columns=["ticker", "avail", "ins_conv_3m",
                                     "ins_convval_3m"])  # caller flags missing
    # first buy in trailing 12mo per (ticker, insider)
    buys["prev"] = buys.groupby(["ticker", "insider"])["filing_dt"].shift(1)
    conv = buys[buys["prev"].isna()
                | (buys["filing_dt"] - buys["prev"] > pd.to_timedelta(365, unit="D"))]
    conv = conv.copy()
    conv["ym"] = conv["filing_dt"].dt.to_period("M")
    g = conv.groupby(["ticker", "ym"])
    m = g.agg(n=("sval", "size"), val=("sval", "sum")).reset_index()
    m["month_end"] = m["ym"].dt.to_timestamp("M")
    m["avail"] = m["month_end"] + pd.to_timedelta(3, unit="D")
    # expand to all filing-months so rolling means carry correctly
    allm = (d.assign(ym=d["filing_dt"].dt.to_period("M"))
            [["ticker", "ym"]].drop_duplicates())
    allm["month_end"] = allm["ym"].dt.to_timestamp("M")
    allm["avail"] = allm["month_end"] + pd.to_timedelta(3, unit="D")
    m = allm.merge(m[["ticker", "avail", "n", "val"]], on=["ticker", "avail"],
                   how="left")
    m[["n", "val"]] = m[["n", "val"]].fillna(0.0)
    m = m.sort_values(["ticker", "month_end"])
    m["ins_conv_3m"] = m.groupby("ticker")["n"].transform(
        lambda s: (s > 0).astype(float).rolling(3, min_periods=1).mean())
    m["ins_convval_3m"] = m.groupby("ticker")["val"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    return m[["ticker", "avail", "ins_conv_3m", "ins_convval_3m"]]


def _get(url: str, tries: int = 3) -> str:
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=30).read().decode(
                "utf-8", errors="replace")
        except Exception:
            time.sleep(2 + a * 2)
    raise RuntimeError(f"fetch failed: {url[:120]}")


def _parse_table(h: str) -> pd.DataFrame:
    m = re.search(r'<table[^>]*tinytable[^>]*>(.*?)</table>', h, re.S)
    if not m:
        return pd.DataFrame()
    t = m.group(1)
    rows = []
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", t, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)
        if len(cells) < 13:
            continue
        txt = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in cells]
        # ticker cell: <a href="/AAPL" ...>AAPL</a> plus hover junk
        sym = re.search(r'">([A-Z][A-Z\.\-]{0,7})</a>', cells[3])
        ticker = sym.group(1) if sym else re.sub(r"[^A-Z\.\-]", "", txt[3])[:8]
        rows.append({
            "filing_dt": txt[1][:10], "trade_dt": txt[2][:10],
            "ticker": ticker, "insider": txt[4], "title": txt[5],
            "ttype": txt[6], "price": txt[7], "qty": txt[8],
            "value": txt[11],
        })
    return pd.DataFrame(rows)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["filing_dt"] = pd.to_datetime(d["filing_dt"], errors="coerce")
    d["trade_dt"] = pd.to_datetime(d["trade_dt"], errors="coerce")
    d = d.dropna(subset=["filing_dt"])
    d["side"] = d["ttype"].str[:1]  # P / S
    d = d[d["side"].isin(["P", "S"])]  # open-market only
    for c in ("price", "qty", "value"):
        d[c] = (d[c].astype(str).str.replace(r"[$,%]", "", regex=True)
                .str.replace(",", "", regex=False))
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["qty"] = d["qty"].abs()
    # signed value: buys +, sells -
    d["sval"] = d["value"].abs() * d["side"].map({"P": 1.0, "S": -1.0})
    key = ["filing_dt", "trade_dt", "ticker", "insider", "side", "qty", "price"]
    return d.drop_duplicates(key).reset_index(drop=True)


def fetch_ticker(ticker: str, refresh: bool = False) -> pd.DataFrame:
    out = RAW / f"{ticker.upper()}.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out, parse_dates=["filing_dt", "trade_dt"])
    frames = []
    for w in WINDOWS:
        for page in range(1, 6):  # cnt=100/page, cap 500 per window
            q = {"s": ticker.upper(), "cnt": "100", "page": str(page)}
            q.update(w)
            url = "http://openinsider.com/screener?" + urllib.parse.urlencode(q)
            got = _clean(_parse_table(_get(url)))
            frames.append(got)
            time.sleep(0.7)
            if len(got) < 100:
                break
    df = (pd.concat(frames, ignore_index=True).drop_duplicates()
          if frames and any(not f.empty for f in frames)
          else pd.DataFrame())
    if not df.empty:
        df.to_csv(out, index=False)
    return df


def monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker x filing-month. Available month_end + 3 days."""
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["ym"] = d["filing_dt"].dt.to_period("M")
    g = d.groupby(["ticker", "ym"])
    m = g.apply(lambda x: pd.Series({
        "n_buy": int((x["side"] == "P").sum()),
        "n_sell": int((x["side"] == "S").sum()),
        "buyers": x.loc[x["side"] == "P", "insider"].nunique(),
        "netval": float(x["sval"].sum()),
    }), include_groups=False).reset_index()
    m["month_end"] = m["ym"].dt.to_timestamp("M")
    m["avail"] = m["month_end"] + pd.to_timedelta(3, unit="D")
    m["buy_ratio"] = m["n_buy"] / (m["n_buy"] + m["n_sell"]).replace(0, 1)
    m["cluster_buy"] = ((m["buyers"] >= 3)).astype(int)
    m = m.sort_values(["ticker", "month_end"])
    for c in ("n_buy", "netval", "buy_ratio"):
        m[c + "_3m"] = m.groupby("ticker")[c].transform(
            lambda s: s.rolling(3, min_periods=1).mean())
    m["buy_ratio_3m"] = m.groupby("ticker")["buy_ratio"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    return m[["ticker", "avail", "n_buy_3m" if "n_buy_3m" in m else "n_buy",
              "netval_3m" if "netval_3m" in m else "netval",
              "buy_ratio_3m", "cluster_buy"]].rename(columns={
        "n_buy_3m": "ins_nbuy_3m", "n_buy": "ins_nbuy_3m",
        "netval_3m": "ins_netval_3m", "netval": "ins_netval_3m",
        "buy_ratio_3m": "ins_buy_ratio_3m", "cluster_buy": "ins_cluster_buy",
    })


def build_all(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for i, t in enumerate(tickers):
        try:
            df = fetch_ticker(t)
            print(f"[{i + 1}/{len(tickers)}] {t}: {len(df)} txns")
            if not df.empty:
                frames.append(monthly_conviction(df))
        except Exception as e:  # noqa: BLE001
            print(f"[{i + 1}/{len(tickers)}] {t} FAILED: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    mp = RAW.parent / "monthly.csv"
    if mp.exists():  # keep other universes; dedupe on rebuild
        old = pd.read_csv(mp)
        out = pd.concat([old, out], ignore_index=True).drop_duplicates()
    out.to_csv(mp, index=False)
    return out


def merge_onto_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """As-of merge: for each (ticker, date) take latest monthly bucket
    with avail <= date. Missing history -> 0 + ins_missing=1."""
    p = panel.copy()
    mp = RAW.parent / "monthly.csv"
    if not mp.exists():
        for c in INS_COLS:
            p[c] = 0.0 if c != "ins_missing" else 1.0
        return p
    m = pd.read_csv(mp, parse_dates=["avail"])
    m = m.sort_values(["ticker", "avail"])
    p = p.sort_values(["Date"])
    parts = []
    for tk, block in p.groupby("ticker", sort=False):
        block = block.sort_values("Date")
        mb = (m[m["ticker"] == tk.upper()].sort_values("avail")
              .drop(columns=["ticker"]))
        if mb.empty:
            block = block.copy()
            for c in INS_COLS:
                block[c] = 0.0 if c != "ins_missing" else 1.0
        else:
            block = pd.merge_asof(block, mb, left_on="Date", right_on="avail",
                                  direction="backward")
            block["ins_missing"] = block["avail"].isna().astype(int)
            for c in INS_COLS:
                if c != "ins_missing":
                    block[c] = block[c].fillna(0.0)
            block = block.drop(columns=["avail"])
        parts.append(block)
    return (pd.concat(parts, ignore_index=True)
            .sort_values(["ticker", "Date"]).reset_index(drop=True))
