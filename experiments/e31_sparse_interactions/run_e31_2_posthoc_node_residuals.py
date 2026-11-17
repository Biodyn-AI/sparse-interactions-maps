# ruff: noqa: EXE002
"""Run the pure post-hoc E31.2 node-versus-edge transfer diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from experiments.e31_sparse_interactions.e31_2_posthoc_node_residuals import (
    analyze_node_residual_transfer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "e31_2_mixed_resolution_v1"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "runs" / "e31_2_posthoc_node_residuals_v1" / "report.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report(run_dir: Path) -> dict[str, Any]:
    discovery_path = run_dir / "discovery_labels.pt"
    confirmation_path = run_dir / "confirmation_labels.pt"
    discovery_sha = _sha256(discovery_path)
    confirmation_sha = _sha256(confirmation_path)
    discovery = torch.load(discovery_path, map_location="cpu", weights_only=False)
    confirmation = torch.load(confirmation_path, map_location="cpu", weights_only=False)
    report = analyze_node_residual_transfer(discovery, confirmation)
    report["sealed_inputs"] = {
        "discovery_labels": {
            "path": str(discovery_path.relative_to(PROJECT_ROOT)),
            "sha256": discovery_sha,
        },
        "confirmation_labels": {
            "path": str(confirmation_path.relative_to(PROJECT_ROOT)),
            "sha256": confirmation_sha,
        },
    }
    # Re-hash after analysis so an accidental input mutation cannot go unnoticed.
    if (
        _sha256(discovery_path) != discovery_sha
        or _sha256(confirmation_path) != confirmation_sha
    ):
        raise RuntimeError("a sealed E31.2 input changed during post-hoc analysis")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {args.output}")
    report = build_report(args.run_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "status": report["status"]}))


if __name__ == "__main__":
    main()
