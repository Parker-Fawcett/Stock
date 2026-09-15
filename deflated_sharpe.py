"""Deflated Sharpe Ratio (PAPER.md Section 9 item 3): Bailey & Lopez de Prado
(2014). Corrects the observed Sharpe of a strategy SELECTED as the best of N
trials for the fact that the max of N noisy estimates is expected to be large
even under a null of zero true skill.

Applies to the self-improvement loop specifically -- that is the one process
in this project that actually maximizes over a trial pool. Momentum was
"frozen before first run... not fitted" (mom_run.py's own docstring) and is
not a selection-among-trials result, so DSR does not apply to it the same
way; PAPER.md Section 6.2 does not claim a searched Sharpe for momentum.

All 12 formal loop proposals are rerun on the tune half of the *legacy*
fold cache (sc_full) -- the same pre-registered dates the real promotion
decisions were made on -- under the CURRENT, corrected accounting (ledger,
turnover, risk-scaling: see FINDINGS.md's improve.py fixes). This is a
deliberate choice: building rigorous statistics on top of numbers already
known to contain a return-accounting bug would just produce a differently
wrong number.

Usage:
  python3 deflated_sharpe.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

from improve import PROPOSALS, load_folds, load_px, stitch

EULER_MASCHERONI = 0.5772156649015329

# The historical DSR target is the original promotion loop, before the later
# exploratory P13-P16 round. Do not inherit newly appended registry entries.
FORMAL_PROPOSAL_IDS = tuple(f"P{number}" for number in range(1, 13))

# Broad trial count: the 12 formal loop proposals plus every other distinct
# configuration reported in FINDINGS.md's "Experiment log" as of Sep 2026,
# counting sub-variants an item names explicitly (e.g. item 7's risk-overlay
# v1/v2, item 8's "3 configs", item 12-value's two discount thresholds).
# Excludes item 19 (a data-hygiene incident, not a strategy) and items
# 13/14/17/21 (those narrate the same 12 formal loop trials, not additional
# ones). Hand count -- large-cap LSTM/ML/+Form4 (3), small-cap price/+raw
# Form4/+conviction Form4 (3), risk overlay v1+v2 (2), tune/holdout 3
# configs (3), long-short (1), mid-cap ML (1), momentum small+mid as one
# frozen rule (1), ensemble (1), vol-scaled monkeys (1), quality sleeve (1),
# multi-asset monthly+quarterly (2), intl sleeve (1), value d0.5+d0.0 (2) =
# 22 earlier informal + 12 formal + 4 later exploratory (P13-P16) = 38.
# Documented here so it can be checked and disputed rather than asserted.
BROAD_TRIAL_COUNT = 38


def expected_max_sharpe(variance_of_trial_sharpes: float, n_trials: int) -> float:
    """E[max of N Sharpe ratios] under a null of zero true skill and
    independent trials (Bailey & Lopez de Prado 2014, eq. 8)."""
    if n_trials <= 1:
        return 0.0
    z1 = norm.ppf(1 - 1 / n_trials)
    z2 = norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(variance_of_trial_sharpes) *
                ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def probabilistic_sharpe_ratio(sr_hat: float, sr_star: float, n: int,
                               skewness: float, kurt: float) -> float:
    """PSR(SR*): probability the true (per-period) Sharpe exceeds SR*,
    given n observations with the given skewness/(Pearson, normal=3)
    kurtosis of returns. All Sharpe ratios here are per-period (monthly),
    not annualized -- annualizing and de-annualizing must be consistent
    or the skew/kurtosis correction terms are wrong."""
    denom = np.sqrt(max(1e-12, 1 - skewness * sr_hat + (kurt - 1) / 4 * sr_hat ** 2))
    z = (sr_hat - sr_star) * np.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def monthly_sharpe(returns: np.ndarray) -> float:
    sd = float(np.std(returns, ddof=1))
    return float(np.mean(returns) / sd) if sd > 0 else 0.0


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    folds = load_folds("data/cache/sc_full")
    mid = len(folds) // 2
    tune_folds = folds[:mid]
    tickers = sorted({tk for t, _ in tune_folds for tk in t["ticker"].unique()})
    px = load_px(tickers)

    trials = {}
    formal = {
        pid: value for pid, value in PROPOSALS.items()
        if pid.split("-", 1)[0] in FORMAL_PROPOSAL_IDS
    }
    if len(formal) != 12:
        raise ValueError(f"expected 12 original proposals, found {sorted(formal)}")
    for pid, (fn, _pred) in formal.items():
        nets = stitch([fn(t, p, px) for t, p in tune_folds])
        trials[pid] = nets.to_numpy()
        print(f"{pid:<14} n={len(nets):>3} monthly_sharpe={monthly_sharpe(trials[pid]):+.3f} "
              f"annualized={monthly_sharpe(trials[pid]) * np.sqrt(12):+.3f}")

    monthly_sharpes = {pid: monthly_sharpe(r) for pid, r in trials.items()}
    best_pid = max(monthly_sharpes, key=monthly_sharpes.get)
    print(f"\nbest of {len(trials)} formal trials (current, corrected code, "
          f"pre-registered tune dates): {best_pid} "
          f"(monthly SR={monthly_sharpes[best_pid]:+.3f}, "
          f"annualized={monthly_sharpes[best_pid] * np.sqrt(12):+.3f})")

    best_returns = trials[best_pid]
    n = len(best_returns)
    g3 = float(skew(best_returns))
    g4 = float(kurtosis(best_returns, fisher=False))  # Pearson convention, normal=3
    sr_hat = monthly_sharpes[best_pid]
    print(f"{best_pid} tune-half return series: n={n} skew={g3:+.3f} "
          f"kurtosis(Pearson)={g4:.3f}")

    print("\n--- Sensitivity range: formal vs. informal trial count ---")
    print("(sigma_SR estimated from the formal 12; P1-vol-only excluded from "
          "that estimate as a documented implementation artifact -- credited "
          "return on flat prices under a disabled-stop bug, not a genuine "
          "strategy-space outcome -- but still counted in N since a real "
          "budget slot and researcher decision was spent on it.)")

    all_sr = np.array(list(monthly_sharpes.values()))
    sr_excl_p1 = np.array([v for k, v in monthly_sharpes.items() if k != "P1-vol-only"])
    variance_excl_p1 = float(np.var(sr_excl_p1, ddof=1))

    scenarios = [
        ("N=12 (formal loop, as run)", 12, float(np.var(all_sr, ddof=1))),
        ("N=12 (formal loop, sigma excl. P1 artifact)", 12, variance_excl_p1),
        (f"N={BROAD_TRIAL_COUNT} (broad: all reported configs, sigma held at "
         "formal-ex-P1 level)", BROAD_TRIAL_COUNT, variance_excl_p1),
    ]
    for label, n_trials, var_sr in scenarios:
        sr_star = expected_max_sharpe(var_sr, n_trials)
        dsr = probabilistic_sharpe_ratio(sr_hat, sr_star, n, g3, g4)
        psr0 = probabilistic_sharpe_ratio(sr_hat, 0.0, n, g3, g4)
        print(f"{label}:")
        print(f"  E[max SR | N trials, sigma_SR={np.sqrt(var_sr):.3f}] "
              f"= {sr_star:+.3f}/mo ({sr_star * np.sqrt(12):+.3f} annualized)")
        print(f"  PSR(0) [naive, ignores selection] = {psr0:.3f}")
        print(f"  DSR = PSR(SR*) [corrected for {n_trials}-trial selection] = {dsr:.3f}")


if __name__ == "__main__":
    main()
