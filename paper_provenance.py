"""Build and verify immutable provenance for every paper table and figure.

Usage:
  python3 paper_provenance.py build
  python3 paper_provenance.py verify

The verifier does not rerun expensive model fits. It proves that the displayed
table/figure, its generating code, its input artifacts, parameters, commands,
and the recorded software environment are unchanged from the locked manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from predictor import universe as U


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "PAPER.md"
DEFAULT_MANIFEST = ROOT / "data" / "paper_provenance" / "manifest.json"
PACKAGES = [
    "numpy", "pandas", "scikit-learn", "scipy", "statsmodels", "matplotlib",
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def existing(paths: list[Path]) -> list[Path]:
    return sorted({path for path in paths if path.is_file()}, key=relative)


def files_in(path: str, pattern: str = "*") -> list[Path]:
    return existing(list((ROOT / path).glob(pattern)))


def raw_files(tickers: list[str]) -> list[Path]:
    return existing([ROOT / "data" / "raw" / f"{ticker}.csv" for ticker in tickers])


def bundle(paths: list[Path]) -> dict:
    files = {relative(path): file_sha256(path) for path in existing(paths)}
    digest = hashlib.sha256()
    for name, value in files.items():
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(value))
    return {
        "file_count": len(files),
        "aggregate_sha256": digest.hexdigest(),
        "files": files,
    }


def source_bundles() -> dict[str, dict]:
    small = [ticker for ticker in U.SMALLCAP if ticker != "LEG"] + ["IWM", "SPY"]
    mid = list(U.MIDCAP) + ["IWM", "SPY"]
    return {
        "controlled_ablation_caches": bundle(
            files_in("data/cache/purge_legacy_ablation", "fold*.csv")
            + files_in("data/cache/purge_fixed_ablation", "fold*.csv")
        ),
        "legacy_loop_cache": bundle(files_in("data/cache/sc_full", "fold*.csv")),
        "corrected_model_cache": bundle(files_in("data/cache/sc_full_v2", "fold*.csv")),
        "smallcap_prices": bundle(raw_files(small)),
        "midcap_prices": bundle(raw_files(mid)),
        "factor_inputs": bundle(files_in("data/factors")),
        "quantconnect_evidence": bundle(files_in("data/qc_reconciliation")),
        "selection_evidence": bundle(
            files_in("data/selection_stability")
            + existing([
                ROOT / "figures" / "selection_stability.png",
                ROOT / "figures" / "selection_stability.pdf",
            ])
        ),
        "research_registry": bundle(existing([ROOT / "data" / "improve_log.json"])),
        "ablation_code": bundle(existing([
            ROOT / "purge_ablation.py", ROOT / "cache_compare.py",
            ROOT / "selection_stability.py", ROOT / "predictor" / "features.py",
            ROOT / "predictor" / "model_lgbm.py", ROOT / "predictor" / "evaluate.py",
        ])),
        "portfolio_code": bundle(existing([
            ROOT / "mom_run.py", ROOT / "ens_test.py", ROOT / "improve.py",
            ROOT / "predictor" / "backtest.py", ROOT / "predictor" / "evaluate.py",
            ROOT / "predictor" / "universe.py",
        ])),
        "factor_code": bundle(existing([
            ROOT / "factor_analysis.py", ROOT / "mom_run.py",
            ROOT / "predictor" / "factors.py", ROOT / "predictor" / "backtest.py",
            ROOT / "predictor" / "evaluate.py", ROOT / "predictor" / "universe.py",
        ])),
        "dsr_code": bundle(existing([
            ROOT / "deflated_sharpe.py", ROOT / "improve.py",
            ROOT / "predictor" / "backtest.py", ROOT / "predictor" / "evaluate.py",
        ])),
        "quantconnect_code": bundle(existing([
            ROOT / "qc_momentum_matched.py", ROOT / "qc_reconcile.py",
            ROOT / "mom_run.py", ROOT / "predictor" / "universe.py",
            ROOT / "predictor" / "evaluate.py",
        ])),
    }


def numbered_blocks(text: str, kind: str) -> dict[str, str]:
    """Extract a numbered Markdown caption and its table/image payload."""
    lines = text.splitlines()
    blocks: dict[str, str] = {}
    for index, line in enumerate(lines):
        marker = f"**{kind} "
        if not line.startswith(marker):
            continue
        number = line[len(marker):].split(".", 1)[0]
        if kind == "Figure":
            collected = []
            if index >= 2 and lines[index - 2].startswith("!["):
                collected.append(lines[index - 2])
                collected.append("")
            for candidate in lines[index:]:
                if collected and not candidate.strip():
                    break
                collected.append(candidate)
            blocks[number] = "\n".join(collected).strip() + "\n"
            continue
        start = index
        collected = []
        saw_payload = False
        for candidate in lines[start:]:
            if candidate.startswith("|") or candidate.startswith("!["):
                saw_payload = True
                collected.append(candidate)
                continue
            if collected and saw_payload and not candidate.strip():
                break
            collected.append(candidate)
        blocks[number] = "\n".join(collected).strip() + "\n"
    return blocks


def environment() -> dict:
    versions = {}
    for package in PACKAGES + ["lightgbm"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            if package != "lightgbm":
                versions[package] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
        "requirements_sha256": file_sha256(ROOT / "requirements.txt"),
    }


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def artifact_specs() -> dict[str, dict]:
    return {
        "table_1": {
            "label": "Controlled purge ablation",
            "bundles": ["controlled_ablation_caches", "ablation_code", "selection_evidence"],
            "commands": [
                "python3 cache_compare.py data/cache/purge_legacy_ablation data/cache/purge_fixed_ablation",
                "MPLBACKEND=Agg python3 selection_stability.py",
            ],
            "parameters": {"seed": 0, "probability_threshold": 0.25, "top_n": 20},
        },
        "figure_1": {
            "label": "Selection stability after repairing the label purge",
            "bundles": ["controlled_ablation_caches", "ablation_code", "selection_evidence"],
            "commands": ["MPLBACKEND=Agg python3 selection_stability.py"],
            "parameters": {"decision_frequency": "monthly", "threshold": 0.25, "top_n": 20},
        },
        "table_2": {
            "label": "Previously selected overlays under controlled purge ablation",
            "bundles": ["controlled_ablation_caches", "smallcap_prices", "portfolio_code", "research_registry"],
            "commands": [
                "python3 improve.py replay-promoted --cache data/cache/purge_legacy_ablation",
                "python3 improve.py replay-promoted --cache data/cache/purge_fixed_ablation",
            ],
            "parameters": {"cost_bps": 25, "split": "recorded tune/holdout"},
        },
        "table_3": {
            "label": "Frozen 12-minus-1-month momentum",
            "bundles": ["smallcap_prices", "midcap_prices", "portfolio_code"],
            "commands": [
                "python3 mom_run.py --universe smallcap --market IWM --cost 25",
                "python3 mom_run.py --universe midcap --market IWM --cost 25",
            ],
            "parameters": {"lookback_sessions": 252, "skip_sessions": 21, "cost_bps": 25},
        },
        "table_4": {
            "label": "Machine learning, momentum, and fixed 50/50 ensemble",
            "bundles": ["corrected_model_cache", "smallcap_prices", "portfolio_code"],
            "commands": [
                "python3 ens_test.py --cache data/cache/sc_full_v2 --split tune --cost 25",
                "python3 ens_test.py --cache data/cache/sc_full_v2 --split holdout --cost 25",
            ],
            "parameters": {"ensemble_weights": [0.5, 0.5], "cost_bps": 25},
        },
        "table_5": {
            "label": "Fixed-survivor implementation comparison",
            "bundles": ["quantconnect_evidence", "quantconnect_code", "smallcap_prices"],
            "commands": [
                "python3 qc_reconcile.py --qc-json \"$QC_RESULT_JSON\" --qc-log \"$QC_LOG\"",
            ],
            "parameters": {"complete_months": 183, "exclude_partial_june_2026": True},
        },
        "table_6": {
            "label": "CAPM and FF5-plus-momentum regressions",
            "bundles": ["smallcap_prices", "midcap_prices", "factor_inputs", "factor_code"],
            "commands": ["python3 factor_analysis.py"],
            "parameters": {"hac_lags": 3, "sample_months": 185},
        },
        "table_7": {
            "label": "Moving-block bootstrap confidence intervals",
            "bundles": ["smallcap_prices", "midcap_prices", "factor_inputs", "factor_code"],
            "commands": ["python3 factor_analysis.py"],
            "parameters": {"seed": 0, "resamples": 5000, "block_months": 6, "holdout_months": 94},
        },
        "table_8": {
            "label": "Deflated Sharpe Ratio",
            "bundles": ["legacy_loop_cache", "smallcap_prices", "dsr_code", "research_registry"],
            "commands": ["python3 deflated_sharpe.py"],
            "parameters": {"historical_trials": 12, "broad_trials": 38, "exclude_p1_from_sigma_only": True},
        },
    }


def build_manifest() -> dict:
    text = PAPER.read_text()
    tables = numbered_blocks(text, "Table")
    figures = numbered_blocks(text, "Figure")
    specs = artifact_specs()
    if set(tables) != {str(number) for number in range(1, 9)}:
        raise ValueError(f"expected Tables 1-8, found {sorted(tables)}")
    if set(figures) != {"1"}:
        raise ValueError(f"expected Figure 1, found {sorted(figures)}")
    artifacts = {}
    for key, spec in specs.items():
        kind, number = key.split("_")
        block = tables[number] if kind == "table" else figures[number]
        artifacts[key] = {
            **spec,
            "paper_block_sha256": sha256_bytes(block.encode()),
        }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": git_commit(),
        "provenance_tool_sha256": file_sha256(Path(__file__)),
        "environment": environment(),
        "bundles": source_bundles(),
        "artifacts": artifacts,
    }


def verify(manifest: dict) -> list[str]:
    errors: list[str] = []
    if file_sha256(Path(__file__)) != manifest["provenance_tool_sha256"]:
        errors.append("provenance verifier changed")
    current_environment = environment()
    if current_environment != manifest["environment"]:
        errors.append("environment lock differs")
    current_bundles = source_bundles()
    for name, expected in manifest["bundles"].items():
        if name not in current_bundles:
            errors.append(f"bundle missing: {name}")
        elif current_bundles[name] != expected:
            errors.append(f"bundle changed: {name}")
    text = PAPER.read_text()
    blocks = {
        "table": numbered_blocks(text, "Table"),
        "figure": numbered_blocks(text, "Figure"),
    }
    for key, artifact in manifest["artifacts"].items():
        kind, number = key.split("_")
        block = blocks.get(kind, {}).get(number)
        if block is None:
            errors.append(f"paper block missing: {key}")
            continue
        actual = sha256_bytes(block.encode())
        if actual != artifact["paper_block_sha256"]:
            errors.append(f"paper block changed: {key}")
        for bundle_name in artifact["bundles"]:
            if bundle_name not in manifest["bundles"]:
                errors.append(f"unknown bundle {bundle_name} in {key}")
        if not artifact["commands"]:
            errors.append(f"no reproduction command: {key}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["build", "verify"])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    if args.mode == "build":
        manifest = build_manifest()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"locked {len(manifest['artifacts'])} paper artifacts in {relative(manifest_path)}")
        return
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    errors = verify(json.loads(manifest_path.read_text()))
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        raise SystemExit(1)
    print(f"PASS: {len(json.loads(manifest_path.read_text())['artifacts'])} paper artifacts verified")


if __name__ == "__main__":
    main()
