"""Refit legacy and corrected purge logic on one frozen data snapshot.

Unlike comparing old cache directories, this holds the panel, universe, fold
schedule, model code, seed, and test rows constant. Only split purging differs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from predictor.features import HORIZON, build_panel, model_cols
from predictor.model_lgbm import fit_fold, walk_forward
from predictor.universe import SMALLCAP


@dataclass
class LegacyFold:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def legacy_walk_forward(panel: pd.DataFrame) -> list[LegacyFold]:
    """Exact pre-fix split logic from commit 091b80f^ for ablation only."""
    dates = np.sort(panel["Date"].unique())
    start = dates[0] + pd.to_timedelta(int(3.0 * 365), unit="D")
    step = pd.to_timedelta(6 * 30, unit="D")
    val_len = pd.to_timedelta(6 * 30, unit="D")
    test_len = pd.to_timedelta(6 * 30, unit="D")
    train_len = pd.to_timedelta(int(3.0 * 365), unit="D")
    folds: list[LegacyFold] = []
    t = start
    while t + val_len + test_len <= dates[-1]:
        te0, te1 = t + val_len, t + val_len + test_len
        va0 = t
        tr0 = t - train_len
        test = panel[(panel["Date"] >= te0) & (panel["Date"] < te1)]
        val = panel[(panel["Date"] >= va0) & (panel["Date"] < te0)]
        train = panel[(panel["Date"] >= tr0) & (panel["Date"] < va0)]
        purge_cut = va0 - pd.to_timedelta(HORIZON, unit="D")
        train = train[train["Date"] < purge_cut]
        if len(train) > 500 and len(val) > 100 and len(test) > 100:
            folds.append(LegacyFold(train, val, test))
        t += step
    return folds


def save_predictions(folds, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for number, fold in enumerate(folds):
        model = fit_fold(fold.train, fold.val, seed=0)
        probability = model.predict_proba(
            fold.test[model_cols(fold.test)].values
        )[:, 1]
        saved = fold.test[["Date", "ticker", "Close", "label"]].copy()
        saved["proba"] = probability
        saved.to_csv(output / f"fold{number}.csv", index=False)
        print(f"{output.name} fold {number + 1}/{len(folds)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-out", required=True)
    parser.add_argument("--fixed-out", required=True)
    args = parser.parse_args()

    tickers = [ticker.upper() for ticker in SMALLCAP] + ["IWM"]
    paths = {ticker: str(Path("data/raw") / f"{ticker}.csv")
             for ticker in tickers
             if (Path("data/raw") / f"{ticker}.csv").exists()}
    panel = build_panel(paths, market="IWM")
    legacy = legacy_walk_forward(panel)
    fixed = walk_forward(panel)
    print(f"panel rows={len(panel):,} dates={panel.Date.min().date()}.."
          f"{panel.Date.max().date()} tickers={panel.ticker.nunique()}")
    print(f"folds legacy={len(legacy)} fixed={len(fixed)}")
    if len(legacy) != len(fixed):
        raise ValueError("fold counts differ; ablation is not controlled")
    for old, new in zip(legacy, fixed):
        if not old.test[["Date", "ticker"]].reset_index(drop=True).equals(
                new.test[["Date", "ticker"]].reset_index(drop=True)):
            raise ValueError("test rows differ; ablation is not controlled")
    save_predictions(legacy, Path(args.legacy_out))
    save_predictions(fixed, Path(args.fixed_out))


if __name__ == "__main__":
    main()
