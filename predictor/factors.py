"""Fama-French 5 factors + momentum, monthly, from Ken French's data library.

Free, public, standard academic source. Cached locally like Yahoo prices;
values are converted from percent to decimal (French publishes e.g. 1.23
meaning 1.23%).
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

FACTOR_DIR = Path(__file__).resolve().parent.parent / "data" / "factors"
FACTOR_DIR.mkdir(parents=True, exist_ok=True)

FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_5_Factors_2x3_CSV.zip")
MOM_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Momentum_Factor_CSV.zip")


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        dest.write_bytes(resp.read())
    return dest


def _parse_monthly_csv(raw_text: str, cols: list[str]) -> pd.DataFrame:
    """French's CSVs have a text header, a monthly block keyed YYYYMM, a
    blank line, then an annual block. Keep only the monthly block."""
    lines = raw_text.splitlines()
    start = next(i for i, ln in enumerate(lines)
                if ln.strip() and ln.strip().split(",")[0].isdigit()
                and len(ln.strip().split(",")[0]) == 6)
    rows = []
    for ln in lines[start:]:
        parts = [p.strip() for p in ln.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 6:
            break  # end of monthly block (blank line / annual header)
        rows.append(parts)
    df = pd.DataFrame(rows, columns=["yyyymm"] + cols)
    df["Date"] = pd.to_datetime(df["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)
    for c in cols:
        df[c] = df[c].astype(float) / 100.0
    return df.set_index("Date")[cols]


def load_ff5_momentum(refresh: bool = False) -> pd.DataFrame:
    """Monthly Mkt-RF, SMB, HML, RMW, CMA, RF, Mom -- decimals, indexed by
    month-end Date. Mom is dropped for months it doesn't cover (rare gap
    at the very start of either series; harmless for a 2011+ study)."""
    ff5_zip = FACTOR_DIR / "ff5.zip"
    mom_zip = FACTOR_DIR / "momentum.zip"
    if refresh:
        for p in (ff5_zip, mom_zip):
            p.unlink(missing_ok=True)
    _download(FF5_URL, ff5_zip)
    _download(MOM_URL, mom_zip)

    with zipfile.ZipFile(ff5_zip) as z:
        ff5_text = z.read(z.namelist()[0]).decode("latin1")
    with zipfile.ZipFile(mom_zip) as z:
        mom_text = z.read(z.namelist()[0]).decode("latin1")

    ff5 = _parse_monthly_csv(ff5_text, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    mom = _parse_monthly_csv(mom_text, ["Mom"])
    mom = mom[mom["Mom"] > -0.98]  # French's -99 missing-data sentinel, post /100
    return ff5.join(mom, how="left")
