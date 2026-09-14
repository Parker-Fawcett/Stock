"""Compare two saved walk-forward probability caches on identical rows.

This diagnoses whether a portfolio reversal is mostly a calendar-window effect
or whether the model rankings themselves changed. It does not refit a model.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from predictor.evaluate import auc_score


def load_cache(path: str, prefix: str) -> tuple[pd.DataFrame, list[dict]]:
    files = sorted(
        glob.glob(str(Path(path) / "fold*.csv")),
        key=lambda f: int("".join(c for c in Path(f).stem if c.isdigit())),
    )
    frames, folds = [], []
    for number, filename in enumerate(files):
        frame = pd.read_csv(filename, parse_dates=["Date"])
        frame["fold"] = number
        frames.append(frame)
        folds.append({
            "fold": number,
            "start": frame["Date"].min().date(),
            "end": frame["Date"].max().date(),
            "rows": len(frame),
        })
    if not frames:
        raise ValueError(f"no fold CSVs found in {path}")
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["Date", "ticker"]).any():
        raise ValueError(f"duplicate Date/ticker rows in {path}")
    keep = ["Date", "ticker", "Close", "label", "proba", "fold"]
    return data[keep].rename(columns={c: f"{c}_{prefix}" for c in keep[2:]}), folds


def selected(group: pd.DataFrame, suffix: str, threshold: float) -> set[str]:
    p = f"proba_{suffix}"
    return set(group[group[p] >= threshold].nlargest(20, p)["ticker"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    left, left_folds = load_cache(args.left, "left")
    right, right_folds = load_cache(args.right, "right")
    common = left.merge(right, on=["Date", "ticker"], how="inner")
    if args.start:
        common = common[common["Date"] >= pd.Timestamp(args.start)]
    if args.end:
        common = common[common["Date"] <= pd.Timestamp(args.end)]
    if common.empty:
        raise ValueError("the caches have no rows in common")

    print(f"left:  {args.left} folds={len(left_folds)} rows={len(left):,} "
          f"dates={left.Date.min().date()}..{left.Date.max().date()} "
          f"tickers={left.ticker.nunique()}")
    print(f"right: {args.right} folds={len(right_folds)} rows={len(right):,} "
          f"dates={right.Date.min().date()}..{right.Date.max().date()} "
          f"tickers={right.ticker.nunique()}")
    print(f"common: rows={len(common):,} dates={common.Date.min().date()}.."
          f"{common.Date.max().date()} tickers={common.ticker.nunique()}")

    label_match = float((common["label_left"] == common["label_right"]).mean())
    close_rel = ((common["Close_left"] / common["Close_right"]) - 1).abs()
    pearson = float(common["proba_left"].corr(common["proba_right"]))
    spearman = float(common["proba_left"].corr(common["proba_right"], method="spearman"))
    mad = float((common["proba_left"] - common["proba_right"]).abs().mean())
    print(f"labels equal={label_match:.3%} close median abs diff={close_rel.median():.3%} "
          f"close max abs diff={close_rel.max():.3%}")
    print(f"probability Pearson={pearson:.3f} Spearman={spearman:.3f} "
          f"mean abs diff={mad:.3f}")
    print(f"common-row AUC left={auc_score(common.label_left.values, common.proba_left.values):.3f} "
          f"right={auc_score(common.label_right.values, common.proba_right.values):.3f}")

    rank_corrs, overlaps, left_sizes, right_sizes = [], [], [], []
    for _, group in common.groupby("Date"):
        rank_corrs.append(group["proba_left"].corr(group["proba_right"], method="spearman"))
        lp = selected(group, "left", args.threshold)
        rp = selected(group, "right", args.threshold)
        union = lp | rp
        overlaps.append(len(lp & rp) / len(union) if union else 1.0)
        left_sizes.append(len(lp))
        right_sizes.append(len(rp))
    print(f"per-date rank Spearman median={np.nanmedian(rank_corrs):.3f} "
          f"mean={np.nanmean(rank_corrs):.3f}")
    print(f"top-20 threshold-{args.threshold:.2f} Jaccard median={np.median(overlaps):.3f} "
          f"mean={np.mean(overlaps):.3f}; mean sizes left={np.mean(left_sizes):.1f} "
          f"right={np.mean(right_sizes):.1f}")

    print("fold endpoints left:", ", ".join(str(f["end"]) for f in left_folds))
    print("fold endpoints right:", ", ".join(str(f["end"]) for f in right_folds))


if __name__ == "__main__":
    main()
