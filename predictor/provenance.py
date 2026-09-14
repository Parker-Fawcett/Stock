"""Small provenance helpers shared by cached and prospective research runs."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def git_state(paths: tuple[str, ...] = ("predictor", "run.py", "paper.py")) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip())
        pipeline_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", *paths],
            text=True).strip())
        working = subprocess.check_output(
            ["git", "diff", "HEAD", "--", *paths], text=True)
        return {
            "commit": commit,
            "dirty": dirty,
            "pipeline_dirty": pipeline_dirty,
            "pipeline_diff_sha256": hashlib.sha256(
                working.encode("utf-8")).hexdigest(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
