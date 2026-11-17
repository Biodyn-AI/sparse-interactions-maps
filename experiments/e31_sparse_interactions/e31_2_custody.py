# ruff: noqa: EXE002
"""Fixed dependencies and write-once primitives for prospective E31.2.

This module owns no model calls.  It verifies the exact GPT-2-small snapshot,
the complete local source closure, and the disclosed E31.1 V2 parent evidence
before registration, acquisition, or scoring may proceed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from experiments.e30_redundant_causal_interfaces.custody import (
    atomic_json_save,
    atomic_torch_save,
    ensure_write_once_available,
    require_exact_keys,
    runtime_environment,
    sha256,
    valid_sha256,
    verify_checkpoint_closure,
    verify_file_hash,
    verify_source_closure,
    write_once_reservation_path,
)
from experiments.e31_sparse_interactions.prospective_custody import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_MODEL_REVISION,
    EXPECTED_MODEL_SNAPSHOT,
    EXPECTED_RUNTIME_ENVIRONMENT,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
E31_SOURCE_DIRECTORY = Path(__file__).resolve().parent
E29_SOURCE_DIRECTORY = REPOSITORY_ROOT / "experiments/e29_gpt2_stable_head_interactions"
E30_SOURCE_DIRECTORY = REPOSITORY_ROOT / "experiments/e30_redundant_causal_interfaces"

E31_1_PARENT_REGISTRATION_SHA256 = (
    "247c78888c7abf37761d89e78f279a1fcec971e00ea2a54c32950c8e796052ee"
)
E31_1_PARENT_SCORE_SHA256 = (
    "ddde7dcdef70525f8fdc3ccc75fe4c57852f87be805b4406a8c716624be57ba5"
)
E31_1_PARENT_SCORE_CUSTODY_SHA256 = (
    "5d254ab1109c4c5b7aa3da5e4d5820f639a5f3e0a0938ed688f81bc310bdfdc0"
)
E31_1_PARENT_PATHS = {
    "registration": REPOSITORY_ROOT
    / "notes/E31.1-PROSPECTIVE-SPARSE-ATLAS-V2-ORDER-REPAIR-REGISTRATION.json",
    "score": REPOSITORY_ROOT
    / "runs/e31_1_prospective_sparse_atlas_v2_order_repair/prospective_score.pt",
    "score_custody": REPOSITORY_ROOT
    / "runs/e31_1_prospective_sparse_atlas_v2_order_repair/score_custody.json",
}
REGISTERED_PARENT_PROVENANCE = {
    "selection_timing": (
        "post_hoc_after_full_E31.1_V2_discovery_and_confirmation_score_inspection"
    ),
    "e31_1_v2_registration_sha256": E31_1_PARENT_REGISTRATION_SHA256,
    "e31_1_v2_score_sha256": E31_1_PARENT_SCORE_SHA256,
    "e31_1_v2_score_artifact_sha256": E31_1_PARENT_SCORE_SHA256,
    "e31_1_v2_score_custody_sha256": E31_1_PARENT_SCORE_CUSTODY_SHA256,
    "e31_1_v2_atlas_verdict": "fail",
    "selection_description": (
        "the three parents are the E31.1 primary k10 hub-off raw-discovery top "
        "three coarse pairs in their exact registered order"
    ),
    "discovery_only_selection_claimed": False,
    "e31_1_confirmation_had_already_been_inspected": True,
    "post_discovery_order_repair_disclosed": True,
    "e31_2_outcomes_used": False,
    "prospective_scope": (
        "fresh E31.2 prompts, head-level outcomes, rectangle selection transfer, "
        "and direct group reconstruction only"
    ),
}

# Complete import closure for registration, acquisition, geometry, and scoring.
# Tests are not scientific sources and are intentionally excluded.
SOURCE_PATHS = {
    "e31/__init__.py": E31_SOURCE_DIRECTORY / "__init__.py",
    "e31/mixed_resolution_panel.py": E31_SOURCE_DIRECTORY / "mixed_resolution_panel.py",
    "e31/mixed_resolution_protocol.py": E31_SOURCE_DIRECTORY
    / "mixed_resolution_protocol.py",
    "e31/mixed_resolution_scoring.py": E31_SOURCE_DIRECTORY
    / "mixed_resolution_scoring.py",
    "e31/e31_2_head_patch_runner.py": E31_SOURCE_DIRECTORY
    / "e31_2_head_patch_runner.py",
    "e31/e31_2_head_acquisition.py": E31_SOURCE_DIRECTORY / "e31_2_head_acquisition.py",
    "e31/e31_2_custody.py": E31_SOURCE_DIRECTORY / "e31_2_custody.py",
    "e31/e31_2_registration.py": E31_SOURCE_DIRECTORY / "e31_2_registration.py",
    "e31/e31_2_orchestration.py": E31_SOURCE_DIRECTORY / "e31_2_orchestration.py",
    "e31/run_e31_2_mixed_resolution.py": E31_SOURCE_DIRECTORY
    / "run_e31_2_mixed_resolution.py",
    "e31/prospective_acquisition.py": E31_SOURCE_DIRECTORY
    / "prospective_acquisition.py",
    "e31/prospective_custody.py": E31_SOURCE_DIRECTORY / "prospective_custody.py",
    "e31/prospective_protocol.py": E31_SOURCE_DIRECTORY / "prospective_protocol.py",
    "e31/prospective_registration.py": E31_SOURCE_DIRECTORY
    / "prospective_registration.py",
    "e31/panel.py": E31_SOURCE_DIRECTORY / "panel.py",
    "e29/__init__.py": E29_SOURCE_DIRECTORY / "__init__.py",
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def verify_parent_provenance() -> dict[str, Any]:
    """Verify the exact inspected E31.1 V2 score and its custody links."""
    verify_file_hash(
        E31_1_PARENT_PATHS["registration"], E31_1_PARENT_REGISTRATION_SHA256
    )
    verify_file_hash(E31_1_PARENT_PATHS["score"], E31_1_PARENT_SCORE_SHA256)
    verify_file_hash(
        E31_1_PARENT_PATHS["score_custody"], E31_1_PARENT_SCORE_CUSTODY_SHA256
    )
    custody = _read_json(E31_1_PARENT_PATHS["score_custody"])
    if (
        custody.get("registration_sha256") != E31_1_PARENT_REGISTRATION_SHA256
        or custody.get("score_sha256") != E31_1_PARENT_SCORE_SHA256
        or custody.get("atlas_verdict") != "fail"
        or custody.get("post_discovery_repair_disclosed") is not True
    ):
        raise RuntimeError("E31.1 V2 score custody or disclosed verdict changed")
    score = torch.load(
        E31_1_PARENT_PATHS["score"], map_location="cpu", weights_only=True
    )
    if (
        not isinstance(score, Mapping)
        or score.get("registration_sha256") != E31_1_PARENT_REGISTRATION_SHA256
        or score.get("score", {}).get("atlas_verdict") != "fail"
    ):
        raise RuntimeError("E31.1 V2 parent score contents changed")
    selected_pairs = (
        score.get("score", {})
        .get("regimes", {})
        .get("nonhub_k10_hub_off", {})
        .get("raw_confirmation", {})
        .get("discovery_order_pairs", [])[:3]
    )
    if selected_pairs != [[10, 16], [16, 18], [10, 20]]:
        raise RuntimeError("E31.1 raw-discovery top-three parent ancestry changed")
    return dict(REGISTERED_PARENT_PROVENANCE)


def verify_fixed_dependencies() -> dict[str, Any]:
    """Verify source, checkpoint, runtime, and E31.1 parent evidence."""
    source_hashes = verify_source_closure(SOURCE_PATHS)
    checkpoint_hashes = verify_checkpoint_closure(
        EXPECTED_MODEL_SNAPSHOT, EXPECTED_CHECKPOINT_SHA256
    )
    environment = runtime_environment()
    if environment != EXPECTED_RUNTIME_ENVIRONMENT:
        raise RuntimeError("runtime environment differs from prospective E31.2")
    return {
        "source_code_sha256": source_hashes,
        "checkpoint_sha256": checkpoint_hashes,
        "environment": environment,
        "parent_provenance": verify_parent_provenance(),
    }


def verify_registered_dependencies(registration: Mapping[str, Any]) -> None:
    """Re-verify every dependency and compare it to the registration."""
    observed = verify_fixed_dependencies()
    for field in (
        "source_code_sha256",
        "checkpoint_sha256",
        "environment",
        "parent_provenance",
    ):
        if registration.get(field) != observed[field]:
            raise RuntimeError(f"registered {field} differs from live dependency")


__all__ = [
    "E31_1_PARENT_PATHS",
    "E31_1_PARENT_REGISTRATION_SHA256",
    "E31_1_PARENT_SCORE_CUSTODY_SHA256",
    "E31_1_PARENT_SCORE_SHA256",
    "EXPECTED_CHECKPOINT_SHA256",
    "EXPECTED_MODEL_REVISION",
    "EXPECTED_MODEL_SNAPSHOT",
    "EXPECTED_RUNTIME_ENVIRONMENT",
    "REGISTERED_PARENT_PROVENANCE",
    "REPOSITORY_ROOT",
    "SOURCE_PATHS",
    "atomic_json_save",
    "atomic_torch_save",
    "ensure_write_once_available",
    "require_exact_keys",
    "sha256",
    "valid_sha256",
    "verify_file_hash",
    "verify_fixed_dependencies",
    "verify_parent_provenance",
    "verify_registered_dependencies",
    "write_once_reservation_path",
]
