"""Write-once custody primitives for E30.1 registration and acquisition."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
E30_SOURCE_DIRECTORY = Path(__file__).resolve().parent
E29_SOURCE_DIRECTORY = REPOSITORY_ROOT / "experiments/e29_gpt2_stable_head_interactions"

EXPECTED_MODEL_REVISION = "6dcaa7a952f72f9298047fd5137cd6e4f05f41da"
EXPECTED_MODEL_SNAPSHOT = Path(
    "/Users/ihorkendiukhov/.cache/huggingface/hub/"
    "models--openai-community--gpt2-medium/snapshots/" + EXPECTED_MODEL_REVISION
)
EXPECTED_CHECKPOINT_SHA256 = {
    "config.json": "ef1a44d889ad1a0acc7731c78134f1b87d2d222f110e97dd10fd4117331caf22",
    "generation_config.json": (
        "b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805"
    ),
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "model.safetensors": (
        "fc5a354a19255ad494f3d71549390baca1ccf61d1d822b9408971705c687c9cd"
    ),
    "tokenizer_config.json": (
        "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f"
    ),
    "tokenizer.json": (
        "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"
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

PARENT_ARTIFACTS = {
    "e29_14d_registration": {
        "path": str(
            (
                REPOSITORY_ROOT
                / "notes/E29.14d-DIRECT-CAUSAL-CARRIER-REGISTRATION.json"
            ).resolve()
        ),
        "sha256": "3eceee73e1cc3633c610eb1be5a1d3ed4e2c2d4724b8680840d61ba89f9a7467",
    },
    "e29_14d_protocol": {
        "path": str(
            (
                REPOSITORY_ROOT / "runs/e29_14d_direct_backup_v1/protocol_conditions.pt"
            ).resolve()
        ),
        "sha256": "a60c1bf4a2d83d8735dac2dd878fc415f7b3fd75af65b08b3574b530ddf4923b",
    },
    "e29_14d_attempt": {
        "path": str(
            (
                REPOSITORY_ROOT / "runs/e29_14d_direct_backup_v1/"
                "direct_backup_acquisition_attempt.json"
            ).resolve()
        ),
        "sha256": "f44a8c9b74b966e3647f663b51bf7d53182943ab1ed385df5d6f7c93976faf28",
    },
    "e29_14d_labels": {
        "path": str(
            (
                REPOSITORY_ROOT
                / "runs/e29_14d_direct_backup_v1/direct_backup_labels.pt"
            ).resolve()
        ),
        "sha256": "53f5013a40b9ed32ac6d6229ad6844cea33c2231be9a2f8b1f642401ffd465a7",
    },
    "e29_14d_custody": {
        "path": str(
            (
                REPOSITORY_ROOT / "runs/e29_14d_direct_backup_v1/"
                "direct_backup_acquisition_custody.json"
            ).resolve()
        ),
        "sha256": "7f1fd09b97fb4602ddb78a12a020ad6fd8004beb1bb420cdd74481499797e110",
    },
    "e29_14d_summary": {
        "path": str(
            (
                REPOSITORY_ROOT
                / "runs/e29_14d_direct_backup_v1/direct_backup_summary.json"
            ).resolve()
        ),
        "sha256": "08069c5b6ff4d4c100e0e259b3debec455951e5e673c853404b19457edd17a49",
    },
    "e29_14a_position_summary": {
        "path": str(
            (
                REPOSITORY_ROOT / "runs/e29_14a_position_lexical_interchange_v1/"
                "position_semantics_summary.json"
            ).resolve()
        ),
        "sha256": "e5a307c981c0c72e9997ad08378f60e44e61afca91c670c09308006cfb624f9a",
    },
    "e29_14c_scale_summary": {
        "path": str(
            (
                REPOSITORY_ROOT
                / "runs/e29_14c_hub_off_scale_ladder_v2/scale_scope_summary.json"
            ).resolve()
        ),
        "sha256": "0da2fff7fb113c8cfc9ee50cac564b9eb85c01f20846019769fcdc556ff4d33a",
    },
    "e29_14c_posthoc_nominations": {
        "path": str(
            (
                REPOSITORY_ROOT / "runs/e29_14c_hub_off_scale_ladder_v2/"
                "sparse_backup_nominations_posthoc.json"
            ).resolve()
        ),
        "sha256": "74ff2e2680abf19197e35a8db620bedbf3d4945cecae410ecaf7aef4cb14359e",
    },
}

# E30 files plus the complete local import closure used to construct its
# inherited E29.14d prompt panel.  Scoring is named prospectively: no seal may
# be produced until that file exists and its final hash enters the closure.
SOURCE_PATHS = {
    "e30/__init__.py": E30_SOURCE_DIRECTORY / "__init__.py",
    "e30/acquisition.py": E30_SOURCE_DIRECTORY / "acquisition.py",
    "e30/custody.py": E30_SOURCE_DIRECTORY / "custody.py",
    "e30/panel.py": E30_SOURCE_DIRECTORY / "panel.py",
    "e30/protocol.py": E30_SOURCE_DIRECTORY / "protocol.py",
    "e30/runner.py": E30_SOURCE_DIRECTORY / "runner.py",
    "e30/scoring.py": E30_SOURCE_DIRECTORY / "scoring.py",
    "e30/registration.py": E30_SOURCE_DIRECTORY / "registration.py",
    "e30/run_acquisition.py": E30_SOURCE_DIRECTORY / "run_acquisition.py",
    "e30/run_scoring.py": E30_SOURCE_DIRECTORY / "run_scoring.py",
    "e29/__init__.py": E29_SOURCE_DIRECTORY / "__init__.py",
    "e29/e29_14d_prompt_panel.py": E29_SOURCE_DIRECTORY / "e29_14d_prompt_panel.py",
    "e29/e29_14b_prompt_panel.py": E29_SOURCE_DIRECTORY / "e29_14b_prompt_panel.py",
    "e29/e29_14a_prompt_panel.py": E29_SOURCE_DIRECTORY / "e29_14a_prompt_panel.py",
    "e29/e29_13_prompt_panel.py": E29_SOURCE_DIRECTORY / "e29_13_prompt_panel.py",
    "e29/e29_12_prompt_panel.py": E29_SOURCE_DIRECTORY / "e29_12_prompt_panel.py",
    "e29/e29_11_multiscale_protocol.py": (
        E29_SOURCE_DIRECTORY / "e29_11_multiscale_protocol.py"
    ),
    "e29/e29_9_seven_gate_protocol.py": (
        E29_SOURCE_DIRECTORY / "e29_9_seven_gate_protocol.py"
    ),
    "e29/e29_8_prompt_panel.py": E29_SOURCE_DIRECTORY / "e29_8_prompt_panel.py",
}

ATTEMPT_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "registration",
        "registration_sha256",
        "expected_registration_sha256_supplied",
        "parent_artifacts_verified_before_attempt",
        "source_code_sha256_verified_before_attempt",
        "checkpoint_sha256_verified_before_attempt",
        "protocol_sha256",
        "condition_manifest_sha256",
        "query_manifest_sha256",
        "environment",
        "execution",
        "raw_acquisition",
        "acquisition_custody",
    }
)


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def require_exact_keys(
    record: Mapping[str, Any],
    expected: Collection[str],
    *,
    label: str,
) -> None:
    observed = set(record)
    required = set(expected)
    if observed != required:
        raise RuntimeError(
            f"{label} keys changed; missing={sorted(required - observed)}, "
            f"extra={sorted(observed - required)}"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_once_reservation_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.write-once-reservation")


def ensure_write_once_available(path: Path) -> None:
    if path.exists() or write_once_reservation_path(path).exists():
        raise FileExistsError(f"refusing to overwrite {path}")


def _publish_write_once(temporary_path: Path, path: Path) -> None:
    """Reserve once with an atomic directory, then atomically publish.

    The research volume is exFAT, so hard-link-based no-replace publication is
    unavailable.  A permanent adjacent reservation directory makes every
    writer using this custody API race-safe and keeps the path consumed even
    if the published artifact is later removed.
    """
    reservation = write_once_reservation_path(path)
    try:
        reservation.mkdir()
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite {path}") from None
    _fsync_directory(path.parent)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    os.replace(temporary_path, path)
    _fsync_directory(path.parent)


def atomic_json_save(record: Any, path: Path) -> None:
    """Atomically create strict JSON and refuse every overwrite."""
    ensure_write_once_available(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix="tmp-",
        suffix=".json",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _publish_write_once(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_torch_save(record: Any, path: Path) -> None:
    """Atomically create a Torch artifact and refuse every overwrite."""
    ensure_write_once_available(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix="tmp-",
        suffix=".pt",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(record, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        _publish_write_once(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def verify_file_hash(path: Path, expected_sha256: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(f"hash mismatch for {path}")
    return observed


def verify_checkpoint_closure(
    snapshot: Path,
    expected_hashes: Mapping[str, str] = EXPECTED_CHECKPOINT_SHA256,
) -> dict[str, str]:
    """Verify every and only registered local checkpoint/tokenizer file."""
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    expected_names = set(expected_hashes)
    observed_names = {path.name for path in snapshot.iterdir()}
    if observed_names != expected_names:
        raise RuntimeError("checkpoint file set changed")
    observed = {}
    for filename, expected in expected_hashes.items():
        if not valid_sha256(expected):
            raise ValueError("checkpoint contains an invalid expected hash")
        observed[filename] = verify_file_hash(snapshot / filename, expected)
    return observed


def verify_source_closure(
    source_paths: Mapping[str, Path] = SOURCE_PATHS,
) -> dict[str, str]:
    """Hash the explicit E30 plus transitive prompt-construction closure."""
    observed = {}
    for label, path in source_paths.items():
        if not isinstance(label, str) or not label or Path(label).is_absolute():
            raise ValueError("source labels must be nonempty relative names")
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[label] = sha256(path)
    return observed


def runtime_environment() -> dict[str, str]:
    """Return exact runtime versions without loading tokenizer or model weights."""
    import safetensors
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "platform": platform.platform(),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def verify_parent_artifacts(
    artifact_specs: Mapping[str, Mapping[str, str]] = PARENT_ARTIFACTS,
) -> dict[str, dict[str, str]]:
    """Verify exact E29.14d parents and their internal custody links."""
    if set(artifact_specs) != set(PARENT_ARTIFACTS):
        raise RuntimeError("parent artifact set changed")
    verified = {}
    for label, spec in artifact_specs.items():
        require_exact_keys(spec, {"path", "sha256"}, label=f"{label} parent spec")
        path = Path(spec["path"])
        digest = str(spec["sha256"])
        if not valid_sha256(digest):
            raise ValueError("parent artifact contains an invalid hash")
        verify_file_hash(path, digest)
        verified[label] = {"path": str(path.resolve()), "sha256": digest}

    registration = _load_json(Path(verified["e29_14d_registration"]["path"]))
    custody = _load_json(Path(verified["e29_14d_custody"]["path"]))
    summary = _load_json(Path(verified["e29_14d_summary"]["path"]))
    if (
        registration.get("model_revision") != EXPECTED_MODEL_REVISION
        or Path(str(registration.get("model_snapshot"))) != EXPECTED_MODEL_SNAPSHOT
        or registration.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError("E29.14d parent model snapshot changed")
    expected_links = {
        "registration_sha256": verified["e29_14d_registration"]["sha256"],
        "protocol_artifact_sha256_verified_before_acquisition": verified[
            "e29_14d_protocol"
        ]["sha256"],
        "acquisition_attempt_sha256": verified["e29_14d_attempt"]["sha256"],
        "direct_backup_labels_sha256": verified["e29_14d_labels"]["sha256"],
    }
    if any(custody.get(key) != value for key, value in expected_links.items()):
        raise RuntimeError("E29.14d parent custody links changed")
    summary_custody = summary.get("custody")
    if (
        summary.get("integrity_passed") is not True
        or summary.get("all_registered_integrity_gates_passed") is not True
        or not isinstance(summary_custody, Mapping)
        or summary_custody.get("registration_sha256")
        != verified["e29_14d_registration"]["sha256"]
        or summary_custody.get("acquisition_custody_sha256")
        != verified["e29_14d_custody"]["sha256"]
        or summary_custody.get("direct_backup_labels_sha256")
        != verified["e29_14d_labels"]["sha256"]
    ):
        raise RuntimeError("E29.14d parent summary integrity changed")
    return verified


def build_attempt_record(
    *,
    registration: Mapping[str, Any],
    registration_path: Path,
    registration_sha256: str,
    expected_registration_sha256: str,
    raw_path: Path,
    custody_path: Path,
) -> dict[str, Any]:
    """Build the marker that must become durable before any model load."""
    if registration_sha256 != expected_registration_sha256 or not valid_sha256(
        registration_sha256
    ):
        raise RuntimeError("registration hash differs before attempt")
    record = {
        "schema_version": "e30_1_causal_topology_acquisition_attempt_v1",
        "status": "attempt_consumed_before_tokenizer_or_model_load",
        "registration": str(registration_path.resolve()),
        "registration_sha256": registration_sha256,
        "expected_registration_sha256_supplied": expected_registration_sha256,
        "parent_artifacts_verified_before_attempt": dict(
            registration["parent_artifacts"]
        ),
        "source_code_sha256_verified_before_attempt": dict(
            registration["source_code_sha256"]
        ),
        "checkpoint_sha256_verified_before_attempt": dict(
            registration["checkpoint_sha256"]
        ),
        "protocol_sha256": registration["protocol_sha256"],
        "condition_manifest_sha256": registration["condition_manifest_sha256"],
        "query_manifest_sha256": registration["query_manifest_sha256"],
        "environment": dict(registration["environment"]),
        "execution": dict(registration["execution"]),
        "raw_acquisition": str(raw_path.resolve()),
        "acquisition_custody": str(custody_path.resolve()),
    }
    require_exact_keys(record, ATTEMPT_REQUIRED_KEYS, label="E30.1 attempt")
    return record


def create_attempt_marker(record: Mapping[str, Any], path: Path) -> str:
    require_exact_keys(record, ATTEMPT_REQUIRED_KEYS, label="E30.1 attempt")
    if record["status"] != "attempt_consumed_before_tokenizer_or_model_load":
        raise RuntimeError("E30.1 attempt status changed")
    atomic_json_save(dict(record), path)
    return sha256(path)


__all__ = [
    "ATTEMPT_REQUIRED_KEYS",
    "EXPECTED_CHECKPOINT_SHA256",
    "EXPECTED_MODEL_REVISION",
    "EXPECTED_MODEL_SNAPSHOT",
    "EXPECTED_RUNTIME_ENVIRONMENT",
    "PARENT_ARTIFACTS",
    "REPOSITORY_ROOT",
    "SOURCE_PATHS",
    "atomic_json_save",
    "atomic_torch_save",
    "build_attempt_record",
    "create_attempt_marker",
    "ensure_write_once_available",
    "require_exact_keys",
    "runtime_environment",
    "sha256",
    "valid_sha256",
    "verify_checkpoint_closure",
    "verify_file_hash",
    "verify_parent_artifacts",
    "verify_source_closure",
    "write_once_reservation_path",
]
