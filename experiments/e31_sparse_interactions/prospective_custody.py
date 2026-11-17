# ruff: noqa: EXE002
"""Fixed dependencies and write-once custody for prospective E31.1."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

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

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
E31_SOURCE_DIRECTORY = Path(__file__).resolve().parent
E29_SOURCE_DIRECTORY = REPOSITORY_ROOT / "experiments/e29_gpt2_stable_head_interactions"
E30_SOURCE_DIRECTORY = REPOSITORY_ROOT / "experiments/e30_redundant_causal_interfaces"

EXPECTED_MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
EXPECTED_MODEL_SNAPSHOT = Path(
    "/Users/ihorkendiukhov/.cache/huggingface/hub/"
    "models--gpt2/snapshots/" + EXPECTED_MODEL_REVISION
)
EXPECTED_CHECKPOINT_SHA256 = {
    "config.json": "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
    "generation_config.json": (
        "ed0b32ac72c0f5f44a719abb2d7786ea5146c871f83717b7f2018065954de02b"
    ),
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "model.safetensors": (
        "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
    ),
    "tokenizer.json": (
        "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"
    ),
    "tokenizer_config.json": (
        "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f"
    ),
    "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
}
EXPECTED_RUNTIME_ENVIRONMENT = {
    "python": "3.12.9",
    "torch": "2.13.0",
    "transformers": "5.14.1",
    "safetensors": "0.8.0",
    "platform": "macOS-26.3-arm64-arm-64bit",
}
EXPECTED_COMPONENT_MAP = [
    [layer, kind] for layer in range(12) for kind in ("attention", "mlp")
]

# This is the complete local import closure that can affect registration,
# acquisition, or scoring. Tests and exploratory E31 modules are deliberately
# absent because no prospective result imports them.
SOURCE_PATHS = {
    "e31/__init__.py": E31_SOURCE_DIRECTORY / "__init__.py",
    "e31/panel.py": E31_SOURCE_DIRECTORY / "panel.py",
    "e31/prospective_protocol.py": E31_SOURCE_DIRECTORY / "prospective_protocol.py",
    "e31/prospective_scoring.py": E31_SOURCE_DIRECTORY / "prospective_scoring.py",
    "e31/prospective_custody.py": E31_SOURCE_DIRECTORY / "prospective_custody.py",
    "e31/prospective_registration.py": E31_SOURCE_DIRECTORY
    / "prospective_registration.py",
    "e31/prospective_acquisition.py": E31_SOURCE_DIRECTORY
    / "prospective_acquisition.py",
    "e31/prospective_orchestration.py": E31_SOURCE_DIRECTORY
    / "prospective_orchestration.py",
    "e31/run_prospective.py": E31_SOURCE_DIRECTORY / "run_prospective.py",
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


def verify_fixed_dependencies() -> dict[str, dict[str, str]]:
    """Verify the complete prospective source, checkpoint, and environment."""
    source_hashes = verify_source_closure(SOURCE_PATHS)
    checkpoint_hashes = verify_checkpoint_closure(
        EXPECTED_MODEL_SNAPSHOT, EXPECTED_CHECKPOINT_SHA256
    )
    environment = runtime_environment()
    if environment != EXPECTED_RUNTIME_ENVIRONMENT:
        raise RuntimeError("runtime environment differs from prospective E31.1")
    return {
        "source_code_sha256": source_hashes,
        "checkpoint_sha256": checkpoint_hashes,
        "environment": environment,
    }


def verify_registered_dependencies(registration: Mapping[str, Any]) -> None:
    """Re-verify every registered dependency immediately before a stage."""
    observed = verify_fixed_dependencies()
    for field in ("source_code_sha256", "checkpoint_sha256", "environment"):
        if registration.get(field) != observed[field]:
            raise RuntimeError(f"registered {field} differs from live dependency")


__all__ = [
    "EXPECTED_CHECKPOINT_SHA256",
    "EXPECTED_COMPONENT_MAP",
    "EXPECTED_MODEL_REVISION",
    "EXPECTED_MODEL_SNAPSHOT",
    "EXPECTED_RUNTIME_ENVIRONMENT",
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
    "verify_registered_dependencies",
    "write_once_reservation_path",
]
