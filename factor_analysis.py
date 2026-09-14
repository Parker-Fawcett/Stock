"""Prespecified stats package (PAPER.md Section 9, items 1-2): moving-block
bootstrap confidence intervals and CAPM / Fama-French-5-plus-momentum
regressions on the frozen momentum sleeves. Nothing here retunes a rule --
momentum's 252/21 spec, universes, and costs are unchanged; this only adds
uncertainty quantification around already-reported numbers.

Usage:
  python3 factor_analysis.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from predictor import evaluate as ev
from predictor import universe as U
from predictor.data import assert_vintage, download_universe
from predictor.factors import load_ff5_momentum
from mom_run import momentum_panel, run_momentum

RNG_SEED = 0
N_BOOT = 5000
BLOCK_LEN = 6  # months; ~ half a year, short enough to have many blocks in
               # a 94-month holdout, long enough to keep monthly autocorrelation


def block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap: concatenate random-start blocks of
    length block_len (wrapping past the end) until length n, then trim."""
    n_blocks = -(-n // block_len)  # ceil
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(s + np.arange(block_len)) % n for s in starts])[:n]
    return idx


def bootstrap_ci(series: np.ndarray, stat_fn, n_boot: int = N_BOOT,
                 block_len: int = BLOCK_LEN, seed: int = RNG_SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = len(series)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = block_bootstrap_indices(n, block_len, rng)
        draws[b] = stat_fn(series[idx])
    point = stat_fn(series)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "n": n, "n_boot": n_boot,
            "block_len": block_len}


def stat_mean_monthly(x: np.ndarray) -> float:
    return float(np.mean(x))


def stat_cagr(x: np.ndarray) -> float:
    equity = float(np.prod(1 + x))
    years = len(x) / 12
    return float(equity ** (1 / years) - 1)


def stat_sharpe(x: np.ndarray) -> float:
    sd = float(np.std(x, ddof=1))
    return float(np.mean(x) / sd * np.sqrt(12)) if sd > 0 else float("nan")


def momentum_monthly_series(tickers: list[str], market: str) -> pd.DataFrame:
    panel = momentum_panel(tickers, market)
    panel.loc[panel.ticker == market.upper(), "ticker"] = "MKT"
    b = run_momentum(panel, cost_bps=25.0)
    b["date"] = pd.to_datetime(b["date"])
    return b[["date", "net"]].rename(columns={"net": "ret"})


def spy_monthly_series(decision_dates: pd.Series) -> pd.Series:
    """SPY simple return over the same 21-trading-day window used by
    run_momentum, decided on the same dates, for a like-for-like
    momentum-minus-SPY comparison (not just a CAGR-level SPY buy-hold)."""
    paths = download_universe(["SPY"], market="SPY")
    assert_vintage({k: str(v) for k, v in paths.items()})
    px = pd.read_csv(paths["SPY"], parse_dates=["Date"]).sort_values("Date").set_index("Date")["Close"]
    rets = {}
    for d0 in decision_dates:
        d0 = pd.Timestamp(d0)
        fut = px.loc[px.index > d0].iloc[:21]
        if len(fut) < 21:
            continue
        rets[d0] = float(fut.iloc[-1] / px.loc[px.index <= d0].iloc[-1] - 1)
    return pd.Series(rets)


def realized_month(decision_date: pd.Timestamp) -> pd.Period:
    """A position decided at d0's month-end is held over the following
    ~21 trading days, i.e. realized mostly in the next calendar month."""
    return (pd.Timestamp(decision_date).to_period("M") + 1)


def run_factor_regression(label: str, monthly_ret: pd.Series, factors: pd.DataFrame) -> None:
    aligned = monthly_ret.to_frame("ret").join(factors, how="inner")
    if len(aligned) < 24:
        print(f"[{label}] too few overlapping months ({len(aligned)}); skipped")
        return
    excess = aligned["ret"] - aligned["RF"]

    print(f"\n[{label}] n={len(aligned)} months "
          f"({aligned.index.min()}..{aligned.index.max()})")
    for name, cols in (("CAPM", ["Mkt-RF"]),
                       ("FF5+Mom", ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"])):
        X = sm.add_constant(aligned[cols])
        # Newey-West HAC, 3 lags: standard choice for monthly data with mild
        # autocorrelation from the 21-trading-day overlap in decision windows.
        model = sm.OLS(excess.values, X.values).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        alpha, alpha_se, alpha_t = model.params[0], model.bse[0], model.tvalues[0]
        annualized_alpha = (1 + alpha) ** 12 - 1
        print(f"  {name}: alpha={alpha:+.4f}/mo ({annualized_alpha:+.2%}/yr) "
              f"t={alpha_t:+.2f} HAC-SE={alpha_se:.4f} R2={model.rsquared:.3f}")
        for i, col in enumerate(cols, start=1):
            print(f"      {col:<8} beta={model.params[i]:+.3f} "
                  f"t={model.tvalues[i]:+.2f}")


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    factors = load_ff5_momentum()

    universes = {
        "small-cap": [t.upper() for t in U.SMALLCAP if t.upper() != "LEG"],
        "mid-cap": [t.upper() for t in U.MIDCAP],
    }

    print("=" * 72)
    print("PART 1: CAPM / Fama-French-5+Momentum regressions, full sample")
    print("=" * 72)
    for label, tickers in universes.items():
        series = momentum_monthly_series(tickers, "IWM")
        series["month"] = series["date"].apply(realized_month)
        monthly = series.groupby("month")["ret"].apply(
            lambda x: float(np.prod(1 + x) - 1))  # collapse same-month rows if any
        monthly.index = monthly.index.to_timestamp("M")
        run_factor_regression(f"{label} momentum", monthly, factors)

    print("\n" + "=" * 72)
    print("PART 2: Moving-block bootstrap 95% CIs, holdout half only")
    print(f"(block length={BLOCK_LEN} months, {N_BOOT} resamples, seed={RNG_SEED})")
    print("=" * 72)
    for label, tickers in universes.items():
        series = momentum_monthly_series(tickers, "IWM")
        mid = len(series) // 2
        holdout = series.iloc[mid:].reset_index(drop=True)
        ret = holdout["ret"].to_numpy()

        spy = spy_monthly_series(holdout["date"])
        common = holdout.set_index("date").join(spy.rename("spy"), how="inner")
        excess_vs_spy = (common["ret"] - common["spy"]).to_numpy()

        print(f"\n[{label}] holdout n={len(ret)} months "
              f"({holdout['date'].min().date()}..{holdout['date'].max().date()})")
        for name, fn, x in (
            ("mean monthly return", stat_mean_monthly, ret),
            ("CAGR", stat_cagr, ret),
            ("Sharpe (annualized)", stat_sharpe, ret),
            ("mean monthly excess vs SPY", stat_mean_monthly, excess_vs_spy),
        ):
            r = bootstrap_ci(x, fn)
            print(f"  {name:<28} point={r['point']:+.4f} "
                  f"95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] (n={r['n']})")


if __name__ == "__main__":
    main()
