# ruff: noqa: EXE002
"""Dry-run or acquire the exploratory E31 old-panel interaction atlas."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from experiments.e30_redundant_causal_interfaces.custody import (
    atomic_json_save,
    atomic_torch_save,
    ensure_write_once_available,
    runtime_environment,
    sha256,
    verify_checkpoint_closure,
    verify_source_closure,
)
from experiments.e31_sparse_interactions.development_atlas import (
    CLAIM_BOUNDARY,
    PARENT_SHA256,
    acquire_missing_masks,
    build_parent_reuse_plan,
    load_sealed_e29_15_confirmation,
    merge_exploratory_atlas,
)
from experiments.e31_sparse_interactions.protocol import (
    build_exhaustive_pair_atlas,
    canonical_sha256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
E31_SOURCE_DIRECTORY = Path(__file__).resolve().parent
E29_SOURCE_DIRECTORY = REPOSITORY_ROOT / "experiments/e29_gpt2_stable_head_interactions"
E30_SOURCE_DIRECTORY = REPOSITORY_ROOT / "experiments/e30_redundant_causal_interfaces"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/e31_1_exploratory_old_panel_atlas_v1"

# Explicit files only: directory scans on the research volume would include
# macOS AppleDouble entries and would not describe the Python import closure.
SOURCE_PATHS = {
    "e31/__init__.py": E31_SOURCE_DIRECTORY / "__init__.py",
    "e31/protocol.py": E31_SOURCE_DIRECTORY / "protocol.py",
    "e31/development_atlas.py": E31_SOURCE_DIRECTORY / "development_atlas.py",
    "e31/run_development_atlas.py": E31_SOURCE_DIRECTORY / "run_development_atlas.py",
    "e29/__init__.py": E29_SOURCE_DIRECTORY / "__init__.py",
    "e29/e29_15_mechanism_discovery_protocol.py": E29_SOURCE_DIRECTORY
    / "e29_15_mechanism_discovery_protocol.py",
    "e29/e29_15_ioi_acquisition.py": E29_SOURCE_DIRECTORY / "e29_15_ioi_acquisition.py",
    "e29/e29_15_ioi_patch_runner.py": E29_SOURCE_DIRECTORY
    / "e29_15_ioi_patch_runner.py",
    "e29/e29_10_float64_protocol.py": E29_SOURCE_DIRECTORY
    / "e29_10_float64_protocol.py",
    "e29/e29_10_float64_runner.py": E29_SOURCE_DIRECTORY / "e29_10_float64_runner.py",
    "e29/e29_9_seven_gate_protocol.py": E29_SOURCE_DIRECTORY
    / "e29_9_seven_gate_protocol.py",
    "e29/generalized_sublayer_patch.py": E29_SOURCE_DIRECTORY
    / "generalized_sublayer_patch.py",
    "e30/__init__.py": E30_SOURCE_DIRECTORY / "__init__.py",
    "e30/custody.py": E30_SOURCE_DIRECTORY / "custody.py",
}


def build_dry_run_report() -> dict[str, Any]:
    """Verify the sealed parent and report exact future work without writes."""
    parent = load_sealed_e29_15_confirmation()
    atlas = build_exhaustive_pair_atlas()
    plan = build_parent_reuse_plan(parent, atlas)
    source_hashes = verify_source_closure(SOURCE_PATHS)
    return {
        "schema_version": "e31_1_exploratory_development_dry_run_v1",
        "status": "dry_run_no_model_call_no_write",
        "atlas": {
            "manifest_sha256": canonical_sha256(atlas),
            "backgrounds": atlas["background_count"],
            "quartets": atlas["quartet_count"],
            "logical_cell_references": atlas["logical_cell_reference_count"],
            "unique_masks": atlas["unique_condition_count"],
        },
        "reuse": {
            "sealed_parent_masks": plan["reused_mask_count"],
            "missing_masks": plan["missing_mask_count"],
            "reused_context_mask_cells": plan["reused_context_mask_cells"],
            "missing_context_mask_cells": plan["missing_context_mask_cells"],
        },
        "parent_sha256": dict(PARENT_SHA256),
        "source_code_sha256": source_hashes,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _default_runner(parent: Mapping[str, Any]) -> Any:
    """Load the exact local GPT-2-small parent only after attempt publication."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from experiments.e29_gpt2_stable_head_interactions.e29_15_ioi_patch_runner import (
        E2915IOIPatchRunner,
    )

    registration = parent["registration"]
    snapshot = Path(registration["model_snapshot"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        attn_implementation="eager",
        torch_dtype=torch.float64,
    ).to(device="cpu", dtype=torch.float64)
    model.eval()
    return E2915IOIPatchRunner(
        model,
        tokenizer,
        batch_size=int(registration["execution"]["batch_size"]),
    )


def run_development_atlas(
    *,
    output_directory: Path = DEFAULT_OUTPUT,
    runner_factory: Callable[[Mapping[str, Any]], Any] = _default_runner,
) -> dict[str, Any]:
    """Acquire only absent masks and publish one new write-once merged atlas."""
    attempt_path = output_directory / "development_attempt.json"
    atlas_path = output_directory / "exhaustive_pair_atlas.pt"
    custody_path = output_directory / "development_custody.json"
    for path in (attempt_path, atlas_path, custody_path):
        ensure_write_once_available(path)

    parent = load_sealed_e29_15_confirmation()
    atlas = build_exhaustive_pair_atlas()
    plan = build_parent_reuse_plan(parent, atlas)
    source_hashes = verify_source_closure(SOURCE_PATHS)
    registration = parent["registration"]
    checkpoint_hashes = verify_checkpoint_closure(
        Path(registration["model_snapshot"]), registration["checkpoint_sha256"]
    )
    environment = runtime_environment()
    if environment != registration["environment"]:
        raise RuntimeError("runtime environment differs from the sealed parent")
    attempt = {
        "schema_version": "e31_1_exploratory_development_attempt_v1",
        "status": "attempt_consumed_before_tokenizer_or_model_load",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "atlas_manifest_sha256": canonical_sha256(atlas),
        "parent_paths": dict(parent["paths"]),
        "parent_sha256": dict(parent["sha256"]),
        "source_code_sha256_verified_before_attempt": source_hashes,
        "checkpoint_sha256_verified_before_attempt": checkpoint_hashes,
        "environment_verified_before_attempt": environment,
        "execution": dict(registration["execution"]),
        "reused_mask_count": plan["reused_mask_count"],
        "missing_mask_count": plan["missing_mask_count"],
        "missing_context_mask_cells": plan["missing_context_mask_cells"],
        "output_atlas": str(atlas_path.resolve()),
        "output_custody": str(custody_path.resolve()),
    }
    atomic_json_save(attempt, attempt_path)
    attempt_sha256 = sha256(attempt_path)

    started = time.monotonic()
    runner = runner_factory(parent)
    missing = acquire_missing_masks(runner, parent, atlas, plan)
    merged = merge_exploratory_atlas(parent, atlas, plan, missing)
    merged["execution_provenance"] = {
        "attempt": str(attempt_path.resolve()),
        "attempt_sha256": attempt_sha256,
        "source_code_sha256": source_hashes,
        "checkpoint_sha256": checkpoint_hashes,
        "environment": environment,
    }
    atomic_torch_save(merged, atlas_path)
    atlas_sha256 = sha256(atlas_path)
    custody = {
        "schema_version": "e31_1_exploratory_development_custody_v1",
        "status": "complete_write_once_exploratory_atlas",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "attempt": str(attempt_path.resolve()),
        "attempt_sha256": attempt_sha256,
        "atlas": str(atlas_path.resolve()),
        "atlas_sha256": atlas_sha256,
        "atlas_manifest_sha256": canonical_sha256(atlas),
        "parent_sha256": dict(parent["sha256"]),
        "source_code_sha256": source_hashes,
        "checkpoint_sha256": checkpoint_hashes,
        "reused_mask_count": plan["reused_mask_count"],
        "newly_acquired_mask_count": plan["missing_mask_count"],
        "new_context_mask_cells": plan["missing_context_mask_cells"],
        "all_integrity_gates_passed": True,
        "runtime_seconds": time.monotonic() - started,
    }
    atomic_json_save(custody, custody_path)
    return custody


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if arguments.dry_run:
        output = build_dry_run_report()
    else:
        output = run_development_atlas(output_directory=arguments.output)
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT",
    "SOURCE_PATHS",
    "build_dry_run_report",
    "run_development_atlas",
]
