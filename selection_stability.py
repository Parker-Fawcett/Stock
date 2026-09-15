"""Measure portfolio-selection stability for the controlled purge ablation.

The portfolio rebalances monthly, so this analysis uses the last cached
observation in each calendar month. It exports the month-level path, a
distance-to-boundary diagnostic, a machine-readable summary, and a paper-ready
two-panel figure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cache_compare import load_cache


DISTANCE_BINS = [-np.inf, 0.01, 0.02, 0.05, 0.10, np.inf]
DISTANCE_LABELS = ["0-1 pp", "1-2 pp", "2-5 pp", "5-10 pp", ">10 pp"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def cache_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path.glob("fold*.csv"),
                   key=lambda item: int("".join(c for c in item.stem if c.isdigit())))
    for file in files:
        digest.update(file.name.encode())
        digest.update(b"\0")
        digest.update(file.read_bytes())
    return digest.hexdigest()


def selection(group: pd.DataFrame, suffix: str, threshold: float,
              top_n: int) -> tuple[set[str], float]:
    probability = f"proba_{suffix}"
    eligible = group[group[probability] >= threshold].nlargest(top_n, probability)
    picks = set(eligible["ticker"])
    boundary = threshold
    if len(eligible) == top_n:
        boundary = max(threshold, float(eligible[probability].min()))
    return picks, boundary


def analyze(left_path: Path, right_path: Path, threshold: float = 0.25,
            top_n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    left, _ = load_cache(str(left_path), "legacy")
    right, _ = load_cache(str(right_path), "corrected")
    common = left.merge(right, on=["Date", "ticker"], how="inner")
    common["month"] = common["Date"].dt.to_period("M")
    month_ends = common.groupby("month")["Date"].transform("max")
    decisions = common[common["Date"] == month_ends].copy()

    monthly_rows: list[dict] = []
    security_rows: list[dict] = []
    for date, group in decisions.groupby("Date", sort=True):
        legacy, legacy_boundary = selection(group, "legacy", threshold, top_n)
        corrected, corrected_boundary = selection(group, "corrected", threshold, top_n)
        union = legacy | corrected
        intersection = legacy & corrected
        rank_correlation = group["proba_legacy"].corr(
            group["proba_corrected"], method="spearman")
        monthly_rows.append({
            "date": date,
            "legacy_n": len(legacy),
            "corrected_n": len(corrected),
            "intersection_n": len(intersection),
            "union_n": len(union),
            # Two empty books contain no investment decision to compare.
            "jaccard": len(intersection) / len(union) if union else np.nan,
            "exact_match": legacy == corrected,
            "rank_spearman": rank_correlation,
            "legacy_boundary": legacy_boundary,
            "corrected_boundary": corrected_boundary,
            "mean_absolute_probability_change": float(
                (group["proba_legacy"] - group["proba_corrected"]).abs().mean()),
        })
        for row in group.itertuples(index=False):
            in_legacy = row.ticker in legacy
            in_corrected = row.ticker in corrected
            distance = 0.5 * (
                abs(row.proba_legacy - legacy_boundary)
                + abs(row.proba_corrected - corrected_boundary)
            )
            security_rows.append({
                "date": date,
                "ticker": row.ticker,
                "proba_legacy": row.proba_legacy,
                "proba_corrected": row.proba_corrected,
                "selected_legacy": in_legacy,
                "selected_corrected": in_corrected,
                "membership_changed": in_legacy != in_corrected,
                "mean_distance_to_arm_boundary": distance,
            })

    monthly = pd.DataFrame(monthly_rows)
    securities = pd.DataFrame(security_rows)
    securities["distance_bin"] = pd.cut(
        securities["mean_distance_to_arm_boundary"], bins=DISTANCE_BINS,
        labels=DISTANCE_LABELS, right=True)
    by_distance = (securities.groupby("distance_bin", observed=False)
                   .agg(names=("ticker", "size"),
                        changed=("membership_changed", "sum"),
                        disagreement_rate=("membership_changed", "mean"))
                   .reset_index())

    summary = {
        "source": {
            "legacy_cache": str(left_path),
            "legacy_cache_sha256": cache_sha256(left_path),
            "corrected_cache": str(right_path),
            "corrected_cache_sha256": cache_sha256(right_path),
            "script_sha256": file_sha256(Path(__file__)),
        },
        "specification": {
            "decision_frequency": "last cached observation in each calendar month",
            "probability_threshold": threshold,
            "maximum_names": top_n,
            "distance_definition": (
                "mean absolute probability distance from each arm's effective "
                "threshold/top-count boundary"
            ),
        },
        "monthly_stability": {
            "start": str(monthly["date"].min().date()),
            "end": str(monthly["date"].max().date()),
            "decision_months": int(len(monthly)),
            "months_with_nonempty_union": int(monthly["jaccard"].notna().sum()),
            "mean_jaccard": float(monthly["jaccard"].mean()),
            "median_jaccard": float(monthly["jaccard"].median()),
            "jaccard_p10": float(monthly["jaccard"].quantile(0.10)),
            "jaccard_p90": float(monthly["jaccard"].quantile(0.90)),
            "exact_match_months_with_nonempty_union": int(
                (monthly["exact_match"] & monthly["jaccard"].notna()).sum()),
            "mean_rank_spearman": float(monthly["rank_spearman"].mean()),
            "median_rank_spearman": float(monthly["rank_spearman"].median()),
            "mean_legacy_names": float(monthly["legacy_n"].mean()),
            "mean_corrected_names": float(monthly["corrected_n"].mean()),
        },
        "distance_to_boundary": [
            {
                "bin": str(row.distance_bin),
                "names": int(row.names),
                "changed": int(row.changed),
                "disagreement_rate": float(row.disagreement_rate),
            }
            for row in by_distance.itertuples(index=False)
        ],
    }
    return monthly, by_distance, summary


def plot(monthly: pd.DataFrame, by_distance: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 160,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.scatter(monthly["date"], monthly["jaccard"], s=12, alpha=0.38,
               color="#496A81", edgecolors="none", label="Monthly")
    rolling = monthly.set_index("date")["jaccard"].rolling(12, min_periods=3).mean()
    ax.plot(rolling.index, rolling, color="#C44900", linewidth=2,
            label="12-month mean")
    ax.axhline(monthly["jaccard"].mean(), color="#222222", linewidth=1,
               linestyle="--", label=f"Full mean {monthly['jaccard'].mean():.3f}")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Selected-name Jaccard overlap")
    ax.set_title("A. Portfolio overlap at monthly decisions")
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    rates = by_distance["disagreement_rate"].to_numpy() * 100
    bars = ax.bar(by_distance["distance_bin"].astype(str), rates,
                  color="#496A81", width=0.72)
    ax.set_ylabel("Names with changed membership (%)")
    ax.set_xlabel("Mean distance from arm-specific portfolio boundary")
    ax.set_title("B. Instability concentrates near the cutoff")
    ax.set_ylim(0, max(45, float(rates.max()) * 1.18))
    ax.grid(axis="y", alpha=0.2)
    for bar, rate, count in zip(bars, rates, by_distance["names"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.1f}%\nn={int(count):,}", ha="center", va="bottom",
                fontsize=8)

    fig.suptitle("Selection stability after repairing the label purge", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", metadata={"Software": "matplotlib"})
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight",
                metadata={"Creator": "selection_stability.py"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path,
                        default=Path("data/cache/purge_legacy_ablation"))
    parser.add_argument("--corrected", type=Path,
                        default=Path("data/cache/purge_fixed_ablation"))
    parser.add_argument("--output-data", type=Path,
                        default=Path("data/selection_stability"))
    parser.add_argument("--figure", type=Path,
                        default=Path("figures/selection_stability.png"))
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    monthly, by_distance, summary = analyze(
        args.legacy, args.corrected, args.threshold, args.top_n)
    args.output_data.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(args.output_data / "monthly_stability.csv", index=False)
    by_distance.to_csv(args.output_data / "distance_to_boundary.csv", index=False)
    (args.output_data / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(monthly, by_distance, args.figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
