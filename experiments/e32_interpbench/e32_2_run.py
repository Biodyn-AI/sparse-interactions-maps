# ruff: noqa: EXE002
"""Canonical write-once production driver for E32.2.

The driver has one registration path, one run directory, and no injectable
scientific callbacks.  Discovery is durably committed before confirmation is
authorized for each case.  All four acquisitions are then hashed and a score
attempt is durably reserved before the scorer loads their response tensors or
the driver opens official ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from experiments.e32_interpbench.adapter import (
    DEFAULT_ENV_DIR,
    DEFAULT_IIT_DIR,
    DEFAULT_MODEL_ROOT,
    DEFAULT_OFFICIAL_CODE_DIR,
    DEFAULT_TRACR_DIR,
    IIT_REVISION,
    INTERPBENCH_CODE_REVISION,
    TRACR_REVISION,
    internal_ground_truth_edges,
    load_case_ground_truth,
    load_official_case,
    load_pinned_model,
    promoted_ground_truth_internal_nodes,
    sha256_file,
    verify_case_artifacts,
    verify_git_revision,
)
from experiments.e32_interpbench.e32_2_acquisition import (
    acquire_split_bundle,
    prior_candidate_token_rows,
)
from experiments.e32_interpbench.e32_2_orchestration import (
    MANDATORY_PANEL_KEYS,
    aggregate_actual_query_accounting,
    assert_complete_fresh_panel,
    validate_case_bundle_pair,
)
from experiments.e32_interpbench.e32_2_protocol import (
    CASE_ORDER,
    EXPECTED_UNIVERSE_COUNTS,
    expected_case_provenance,
    protocol_dry_run_manifest,
    split_request,
)
from experiments.e32_interpbench.eap_baseline import (
    AUTO_CIRCUIT_REVISION,
    DEFAULT_AUTO_CIRCUIT_DIR,
    PINNED_EAP_ENVIRONMENT,
    verify_pinned_eap_runtime,
)

CANONICAL_REGISTRATION_PATH = (
    REPOSITORY_ROOT / "notes" / "E32.2-NODE-FIRST-DIRECTED-REGISTRATION.json"
)
CANONICAL_RUN_ATTEMPT_PATH = (
    REPOSITORY_ROOT / "notes" / "E32.2-NODE-FIRST-DIRECTED-RUN-ATTEMPT.json"
)
CANONICAL_RUN_DIRECTORY = REPOSITORY_ROOT / "runs" / "e32_2_node_first_directed_v1"

SOURCE_RELATIVE_PATHS = (
    "experiments/e32_interpbench/__init__.py",
    "experiments/e32_interpbench/adapter.py",
    "experiments/e32_interpbench/component_patching.py",
    "experiments/e32_interpbench/acquisition.py",
    "experiments/e32_interpbench/protocol.py",
    "experiments/e32_interpbench/methods.py",
    "experiments/e32_interpbench/scoring.py",
    "experiments/e32_interpbench/analysis.py",
    "experiments/e32_interpbench/eap_baseline.py",
    "experiments/e32_interpbench/e32_2_protocol.py",
    "experiments/e32_interpbench/e32_2_directed_replacement.py",
    "experiments/e32_interpbench/e32_2_acquisition.py",
    "experiments/e32_interpbench/e32_2_scoring.py",
    "experiments/e32_interpbench/e32_2_orchestration.py",
    "experiments/e32_interpbench/e32_2_run.py",
)

RUNTIME_SOURCE_PACKAGES = ("tracr", "iit", "transformer_lens", "auto_circuit")
EXPECTED_RUNTIME_PACKAGE_TREES = {
    "tracr": {
        "file_count": 57,
        "tree_sha256": "37f97634e3787fefe0e4ed3ec8389a06536c421e6bf48fcf3de9361cc660a5ed",
    },
    "iit": {
        "file_count": 45,
        "tree_sha256": "ec3992cfe9e3e2d1fbf397a30a3a508db8a9de8fb69334f392837bd5893fd262",
    },
    "transformer_lens": {
        "file_count": 21,
        "tree_sha256": "e35bb9d7c29f8d3bc510a826d9def1961efa2fdd05f5c716c355ac1cce8e859d",
    },
    "auto_circuit": {
        "file_count": 53,
        "tree_sha256": "9341912da7f8695fa2fc43f8de67e49d68e763fdeb612aff5521636eea8b6d0f",
    },
}

ACQUISITION_FILENAMES = tuple(
    f"case{case_id}_{split}.pt" for case_id, split in MANDATORY_PANEL_KEYS
)
CUSTODY_FILENAMES = tuple(
    filename
    for case_id in CASE_ORDER
    for filename in (
        f"case{case_id}_discovery_custody.json",
        f"case{case_id}_confirmation_authorization.json",
        f"case{case_id}_confirmation_custody.json",
    )
) + ("acquisition_custody.json",)

# Inspected official identities are explicit rather than represented only by
# counts.  The raw artifact digest remains the upstream identity anchor.
EXPECTED_GROUND_TRUTH_IDENTITIES = {
    2: {
        "nodes": (
            "blocks.0.attn.hook_result[0]",
            "blocks.3.attn.hook_result[0]",
            "blocks.1.hook_mlp_out",
            "blocks.0.hook_mlp_out",
            "blocks.2.hook_mlp_out",
        ),
        "edges": (
            ("blocks.0.attn.hook_result[0]", "blocks.0.hook_mlp_out"),
            ("blocks.0.hook_mlp_out", "blocks.1.hook_mlp_out"),
            ("blocks.1.hook_mlp_out", "blocks.2.hook_mlp_out"),
            ("blocks.2.hook_mlp_out", "blocks.3.attn.hook_result[0]"),
        ),
    },
    21: {
        "nodes": (
            "blocks.3.attn.hook_result[0]",
            "blocks.2.hook_mlp_out",
            "blocks.1.hook_mlp_out",
            "blocks.0.hook_mlp_out",
        ),
        "edges": (
            ("blocks.2.hook_mlp_out", "blocks.3.attn.hook_result[0]"),
            ("blocks.1.hook_mlp_out", "blocks.2.hook_mlp_out"),
        ),
    },
}


@dataclass(frozen=True)
class ScoreStageReservation:
    """Durable authorization to open the four acquisition outcomes for scoring."""

    path: Path
    registration_sha256: str
    run_attempt_sha256: str
    acquisition_artifact_sha256: dict[str, str]
    custody_artifact_sha256: dict[str, str]
    payload_sha256: str


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def exclusive_json_write(path: Path, value: Any) -> None:
    """Write JSON exactly once using the filesystem exclusive-create flag."""
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_parent(path)


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_torch_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_parent(path)


def _source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_RELATIVE_PATHS:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing registered source: {path}")
        hashes[relative] = sha256_file(path)
    return hashes


def _clean_git_revision(
    checkout: Path, expected_revision: str, *, include_untracked: bool = False
) -> str:
    revision = verify_git_revision(checkout, expected_revision)
    status = subprocess.run(
        [
            "git",
            "-C",
            str(checkout.resolve()),
            "status",
            "--porcelain",
            f"--untracked-files={'all' if include_untracked else 'no'}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(f"pinned checkout has source drift: {checkout}")
    return revision


def _package_tree(root: Path) -> dict[str, Any]:
    file_hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    if not file_hashes:
        raise RuntimeError(f"runtime package has no auditable sources: {root}")
    return {
        "root": str(root.resolve()),
        "file_count": len(file_hashes),
        "tree_sha256": _json_sha256(file_hashes),
        "file_sha256": file_hashes,
    }


def _runtime_package_source_record(package_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.submodule_search_locations is None:
        return {"status": "not_installed"}
    locations = tuple(spec.submodule_search_locations)
    if len(locations) != 1:
        raise RuntimeError(
            f"expected one source root for {package_name}, got {locations}"
        )
    return {"status": "present", **_package_tree(Path(locations[0]).resolve())}


def _installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _environment_path_spec_present(module_name: str) -> bool:
    """Check the environment path without consulting runtime module stubs.

    The sealed benchmark adapter may install a bare ``pygraphviz`` module in
    ``sys.modules`` when that optional native dependency is unavailable.
    ``importlib.util.find_spec`` raises for such a module because its
    ``__spec__`` is ``None``.  PathFinder records the underlying environment
    branch directly, so the custody record is stable before and after the
    adapter creates its compatibility stub.
    """
    return importlib.machinery.PathFinder.find_spec(module_name) is not None


def _all_installed_distributions() -> dict[str, tuple[dict[str, str], ...]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for distribution in importlib.metadata.distributions():
        metadata_path = getattr(distribution, "_path", None)
        # macOS AppleDouble sidecars can resemble ``.dist-info`` directories
        # to importlib.metadata but are not installed Python distributions.
        if metadata_path is not None and Path(metadata_path).name.startswith("._"):
            continue
        try:
            raw_name = distribution.metadata.get("Name")
            version = distribution.version
        except UnicodeError as error:
            raise RuntimeError(
                f"installed distribution metadata is unreadable: {metadata_path}"
            ) from error
        if not isinstance(raw_name, str) or not raw_name:
            raise RuntimeError("installed distribution lacks a canonical name")
        name = raw_name.lower().replace("_", "-").replace(".", "-")
        grouped.setdefault(name, []).append(
            {
                "version": version,
                "metadata_path": (
                    "unavailable"
                    if metadata_path is None
                    else str(Path(metadata_path).resolve())
                ),
            }
        )
    return {
        name: tuple(
            sorted(
                records, key=lambda record: (record["version"], record["metadata_path"])
            )
        )
        for name, records in sorted(grouped.items())
    }


def _environment_record() -> dict[str, Any]:
    distributions = (
        "torch",
        "transformer-lens",
        "numpy",
        "jax",
        "jaxlib",
        "auto-circuit",
        "einops",
        "huggingface-hub",
        "blobfile",
        "kaleido",
        "ordered-set",
        "plotly",
        "vscode-tqdm",
        "word2word",
    )
    return {
        "python": platform.python_version(),
        "executable": str(Path(sys.executable).absolute()),
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "platform": platform.platform(),
        "packages": {name: _installed_version(name) for name in distributions},
        "all_installed_distributions": _all_installed_distributions(),
        "pygraphviz_branch": {
            "availability_semantics": (
                "environment_path_spec_ignoring_runtime_sys_modules_stubs"
            ),
            "environment_path_spec_present": _environment_path_spec_present(
                "pygraphviz"
            ),
            "distribution_version": _installed_version("pygraphviz"),
        },
        "runtime_package_sources": {
            name: _runtime_package_source_record(name)
            for name in RUNTIME_SOURCE_PACKAGES
        },
    }


def _dependency_record() -> dict[str, Any]:
    source_revisions = {
        "interpbench": _clean_git_revision(
            DEFAULT_OFFICIAL_CODE_DIR,
            INTERPBENCH_CODE_REVISION,
            include_untracked=True,
        ),
        "tracr": _clean_git_revision(
            DEFAULT_TRACR_DIR, TRACR_REVISION, include_untracked=True
        ),
        "iit": _clean_git_revision(
            DEFAULT_IIT_DIR, IIT_REVISION, include_untracked=True
        ),
        "auto_circuit": _clean_git_revision(
            DEFAULT_AUTO_CIRCUIT_DIR,
            AUTO_CIRCUIT_REVISION,
            include_untracked=True,
        ),
    }
    checkout_package = _package_tree(DEFAULT_AUTO_CIRCUIT_DIR / "auto_circuit")
    interpbench_package = _package_tree(
        DEFAULT_OFFICIAL_CODE_DIR / "circuits_benchmark"
    )
    expected_auto = EXPECTED_RUNTIME_PACKAGE_TREES["auto_circuit"]
    if {
        key: checkout_package[key] for key in ("file_count", "tree_sha256")
    } != expected_auto:
        raise RuntimeError("pinned AutoCircuit checkout package tree changed")
    return {
        "source_directories": {
            "interpbench": str(DEFAULT_OFFICIAL_CODE_DIR.resolve()),
            "tracr": str(DEFAULT_TRACR_DIR.resolve()),
            "iit": str(DEFAULT_IIT_DIR.resolve()),
            "auto_circuit": str(DEFAULT_AUTO_CIRCUIT_DIR.resolve()),
            "model_root": str(DEFAULT_MODEL_ROOT.resolve()),
        },
        "source_revisions": source_revisions,
        "tracked_worktrees_clean": {name: True for name in source_revisions},
        "auto_circuit_checkout_package": checkout_package,
        "interpbench_checkout_package": interpbench_package,
        "case_artifact_sha256": {
            str(case_id): verify_case_artifacts(DEFAULT_MODEL_ROOT, case_id)
            for case_id in CASE_ORDER
        },
    }


def _require_production_environment() -> dict[str, Any]:
    observed = _environment_record()
    required_versions = {
        key: value for key, value in PINNED_EAP_ENVIRONMENT.items() if key != "python"
    }
    wrong = {
        "python": {
            "expected": PINNED_EAP_ENVIRONMENT["python"],
            "observed": observed["python"],
        }
        if observed["python"] != PINNED_EAP_ENVIRONMENT["python"]
        else None,
        "executable": {
            "expected": str((DEFAULT_ENV_DIR / "bin" / "python").absolute()),
            "observed": observed["executable"],
        }
        if observed["executable"]
        != str((DEFAULT_ENV_DIR / "bin" / "python").absolute())
        else None,
        "sys_prefix": {
            "expected": str(DEFAULT_ENV_DIR.resolve()),
            "observed": observed["sys_prefix"],
        }
        if observed["sys_prefix"] != str(DEFAULT_ENV_DIR.resolve())
        else None,
    }
    wrong = {key: value for key, value in wrong.items() if value is not None}
    for distribution, expected in required_versions.items():
        actual = observed["packages"].get(distribution)
        if actual != expected:
            wrong[f"package:{distribution}"] = {
                "expected": expected,
                "observed": actual,
            }
    if wrong:
        raise RuntimeError(f"E32.2 requires the pinned production environment: {wrong}")

    site_packages = DEFAULT_ENV_DIR / "lib" / "python3.11" / "site-packages"
    for package_name, expected in EXPECTED_RUNTIME_PACKAGE_TREES.items():
        source = observed["runtime_package_sources"][package_name]
        if source.get("status") != "present":
            raise RuntimeError(f"missing runtime source package {package_name}")
        if source["root"] != str((site_packages / package_name).resolve()):
            raise RuntimeError(f"runtime package root changed: {package_name}")
        if {key: source[key] for key in expected} != expected:
            raise RuntimeError(f"runtime package source tree changed: {package_name}")
    runtime = verify_pinned_eap_runtime()
    installed_auto = observed["runtime_package_sources"]["auto_circuit"]
    checkout_auto = _package_tree(DEFAULT_AUTO_CIRCUIT_DIR / "auto_circuit")
    if installed_auto["file_sha256"] != checkout_auto["file_sha256"]:
        raise RuntimeError("installed AutoCircuit package differs from clean checkout")
    return {"environment": observed, "pinned_eap_runtime": runtime}


def build_registration_record() -> dict[str, Any]:
    """Build the exact record re-derived before every acquisition stage."""
    protocol = protocol_dry_run_manifest()
    return {
        "schema_version": "e32_2_node_first_directed_registration_v1",
        "canonical_registration_path": str(CANONICAL_REGISTRATION_PATH.resolve()),
        "canonical_run_attempt_path": str(CANONICAL_RUN_ATTEMPT_PATH.resolve()),
        "canonical_run_directory": str(CANONICAL_RUN_DIRECTORY.resolve()),
        "protocol": protocol,
        "inspected_ground_truth_identities": EXPECTED_GROUND_TRUTH_IDENTITIES,
        "source_sha256": _source_hashes(),
        "dependencies": _dependency_record(),
        "environment": _environment_record(),
        "execution": {
            "device": "cpu",
            "all_cases_mandatory": True,
            "case97_is_development_only": True,
            "single_canonical_registration_and_run": True,
            "write_once_artifacts_and_stage_reservations": True,
            "full_prior_candidate_pools_excluded_per_case": True,
            "cross_case_token_ids_are_not_shared_row_identities": True,
            "discovery_artifact_precedes_confirmation_authorization": True,
            "confirmation_authorization_precedes_confirmation_model_call": True,
            "score_attempt_precedes_score_stage_opening_acquisition_outcomes": True,
            "ground_truth_opened_only_in_score_stage": True,
            "actual_and_operational_accounting_separate": True,
            "production_scientific_callbacks": False,
        },
    }


def verify_registration_record(record: Mapping[str, Any]) -> None:
    if _canonical_json_bytes(record) != _canonical_json_bytes(
        build_registration_record()
    ):
        raise ValueError(
            "registration differs from exact current protocol/dependencies"
        )


def _load_registration() -> tuple[dict[str, Any], str]:
    if not CANONICAL_REGISTRATION_PATH.is_file():
        raise FileNotFoundError("E32.2 registration is not sealed")
    record = json.loads(CANONICAL_REGISTRATION_PATH.read_text())
    verify_registration_record(record)
    return record, sha256_file(CANONICAL_REGISTRATION_PATH)


def seal_registration() -> dict[str, Any]:
    _require_production_environment()
    if CANONICAL_REGISTRATION_PATH.exists():
        raise FileExistsError("canonical E32.2 registration is already reserved")
    if CANONICAL_RUN_ATTEMPT_PATH.exists():
        raise FileExistsError("canonical E32.2 run attempt is already consumed")
    if CANONICAL_RUN_DIRECTORY.exists():
        raise FileExistsError("canonical E32.2 run directory is not pristine")
    record = build_registration_record()
    if CANONICAL_RUN_DIRECTORY.exists():
        raise FileExistsError("canonical E32.2 run directory is not pristine")
    exclusive_json_write(CANONICAL_REGISTRATION_PATH, record)
    return record


def _run_attempt_payload(
    *, registration_sha256: str, run_directory: Path
) -> dict[str, str]:
    return {
        "schema_version": "e32_2_run_attempt_reservation_v1",
        "registration_sha256": _validated_sha256(
            registration_sha256, label="registration_sha256"
        ),
        "canonical_run_directory": str(Path(run_directory).resolve()),
        "status": "irreversible_attempt_reserved_before_run_directory_or_model",
    }


def reserve_run_attempt(
    *, path: Path, registration_sha256: str, run_directory: Path
) -> str:
    """Irreversibly consume one run version outside its deletable directory."""
    payload = _run_attempt_payload(
        registration_sha256=registration_sha256, run_directory=run_directory
    )
    exclusive_json_write(Path(path), payload)
    return sha256_file(path)


def verify_run_attempt(
    *,
    path: Path,
    registration_sha256: str,
    run_directory: Path,
    expected_sha256: str,
) -> None:
    expected = _run_attempt_payload(
        registration_sha256=registration_sha256, run_directory=run_directory
    )
    try:
        observed = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(
            "external run-attempt reservation is missing or malformed"
        ) from error
    if _canonical_json_bytes(observed) != _canonical_json_bytes(
        expected
    ) or sha256_file(path) != _validated_sha256(
        expected_sha256, label="expected run-attempt hash"
    ):
        raise ValueError("external run-attempt reservation changed")


def _validated_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _score_reservation_payload(
    *,
    registration_sha256: str,
    run_attempt_sha256: str,
    acquisition_artifact_sha256: Mapping[str, str],
    custody_artifact_sha256: Mapping[str, str],
) -> dict[str, Any]:
    registration_hash = _validated_sha256(
        registration_sha256, label="registration_sha256"
    )
    run_attempt_hash = _validated_sha256(run_attempt_sha256, label="run_attempt_sha256")
    hashes = dict(acquisition_artifact_sha256)
    if tuple(hashes) != ACQUISITION_FILENAMES:
        raise ValueError("score stage requires all four ordered acquisition artifacts")
    for filename, digest in hashes.items():
        _validated_sha256(digest, label=f"artifact hash for {filename}")
    custody_hashes = dict(custody_artifact_sha256)
    if tuple(custody_hashes) != CUSTODY_FILENAMES:
        raise ValueError("score stage requires the complete ordered custody chain")
    for filename, digest in custody_hashes.items():
        _validated_sha256(digest, label=f"custody hash for {filename}")
    return {
        "schema_version": "e32_2_score_attempt_v1",
        "registration_sha256": registration_hash,
        "run_attempt_sha256": run_attempt_hash,
        "acquisition_artifact_sha256": hashes,
        "custody_artifact_sha256": custody_hashes,
        "status": "durable_before_score_stage_opens_acquisition_outcomes_or_gt",
    }


def reserve_score_stage(
    *,
    run_directory: Path,
    registration_sha256: str,
    run_attempt_sha256: str,
    acquisition_artifact_sha256: Mapping[str, str],
    custody_artifact_sha256: Mapping[str, str],
) -> ScoreStageReservation:
    """Verify all four artifact hashes, then exclusively reserve scoring."""
    directory = Path(run_directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError("score-stage run directory does not exist")
    payload = _score_reservation_payload(
        registration_sha256=registration_sha256,
        run_attempt_sha256=run_attempt_sha256,
        acquisition_artifact_sha256=acquisition_artifact_sha256,
        custody_artifact_sha256=custody_artifact_sha256,
    )
    for filename, digest in payload["acquisition_artifact_sha256"].items():
        path = directory / filename
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError("acquisition artifact missing or differs before scoring")
    for filename, digest in payload["custody_artifact_sha256"].items():
        path = directory / filename
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError("custody artifact missing or differs before scoring")
    path = directory / "score_attempt.json"
    exclusive_json_write(path, payload)
    return ScoreStageReservation(
        path=path,
        registration_sha256=payload["registration_sha256"],
        run_attempt_sha256=payload["run_attempt_sha256"],
        acquisition_artifact_sha256=dict(payload["acquisition_artifact_sha256"]),
        custody_artifact_sha256=dict(payload["custody_artifact_sha256"]),
        payload_sha256=_json_sha256(payload),
    )


def verify_score_stage_reservation(reservation: ScoreStageReservation) -> None:
    """Recheck the durable reservation and every bound acquisition artifact."""
    if not isinstance(reservation, ScoreStageReservation):
        raise TypeError("a score-stage reservation is required before outcome access")
    expected = _score_reservation_payload(
        registration_sha256=reservation.registration_sha256,
        run_attempt_sha256=reservation.run_attempt_sha256,
        acquisition_artifact_sha256=reservation.acquisition_artifact_sha256,
        custody_artifact_sha256=reservation.custody_artifact_sha256,
    )
    try:
        observed = json.loads(reservation.path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("score-stage reservation is missing or malformed") from error
    if (
        _canonical_json_bytes(observed) != _canonical_json_bytes(expected)
        or _json_sha256(expected) != reservation.payload_sha256
    ):
        raise ValueError("score-stage reservation differs from its frozen payload")
    directory = reservation.path.parent
    for filename, digest in reservation.acquisition_artifact_sha256.items():
        path = directory / filename
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError("score-stage acquisition artifact changed")
    for filename, digest in reservation.custody_artifact_sha256.items():
        path = directory / filename
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError("score-stage custody artifact changed")


def load_score_stage_artifacts(
    reservation: ScoreStageReservation,
) -> dict[tuple[int, str], dict[str, Any]]:
    """Open acquisition responses only after their score attempt is durable."""
    verify_score_stage_reservation(reservation)
    panel: dict[tuple[int, str], dict[str, Any]] = {}
    for (case_id, split), filename in zip(
        MANDATORY_PANEL_KEYS, ACQUISITION_FILENAMES, strict=True
    ):
        envelope = torch.load(
            reservation.path.parent / filename,
            map_location="cpu",
            weights_only=False,
        )
        if (
            not isinstance(envelope, Mapping)
            or envelope.get("schema_version") != "e32_2_split_artifact_v1"
            or envelope.get("registration_sha256") != reservation.registration_sha256
            or envelope.get("run_attempt_sha256") != reservation.run_attempt_sha256
            or envelope.get("case_id") != case_id
            or envelope.get("split") != split
        ):
            raise ValueError("score-stage split artifact envelope changed")
        bundle = envelope.get("acquisition")
        if not isinstance(bundle, dict):
            raise TypeError("score-stage acquisition payload must be a dictionary")
        if split == "confirmation":
            authorization_hash = envelope.get("confirmation_authorization_sha256")
            if bundle.get("confirmation_authorization_sha256") != authorization_hash:
                raise ValueError("confirmation bundle authorization hash changed")
            authorization_path = (
                reservation.path.parent
                / f"case{case_id}_confirmation_authorization.json"
            )
            if (
                not authorization_path.is_file()
                or sha256_file(authorization_path) != authorization_hash
            ):
                raise ValueError("confirmation authorization artifact changed")
            authorization = json.loads(authorization_path.read_text())
            discovery_filename = f"case{case_id}_discovery.pt"
            discovery_custody_path = (
                reservation.path.parent / f"case{case_id}_discovery_custody.json"
            )
            discovery_bundle = panel.get((case_id, "discovery"))
            if (
                authorization.get("schema_version")
                != "e32_2_confirmation_authorization_v1"
                or authorization.get("case_id") != case_id
                or authorization.get("registration_sha256")
                != reservation.registration_sha256
                or authorization.get("run_attempt_sha256")
                != reservation.run_attempt_sha256
                or authorization.get("discovery_artifact_sha256")
                != reservation.acquisition_artifact_sha256[discovery_filename]
                or not discovery_custody_path.is_file()
                or authorization.get("discovery_custody_sha256")
                != sha256_file(discovery_custody_path)
                or discovery_bundle is None
                or tuple(authorization.get("frozen_discovery_nodes", ()))
                != tuple(discovery_bundle.get("discovery_frozen_nodes", ()))
                or tuple(
                    tuple(edge)
                    for edge in authorization.get("frozen_induced_edges", ())
                )
                != tuple(discovery_bundle.get("discovery_induced_edges", ()))
            ):
                raise ValueError("confirmation authorization chain changed")
        panel[(case_id, split)] = bundle
    verify_score_stage_reservation(reservation)
    return panel


def _authorization_record(
    *,
    case_id: int,
    registration_sha256: str,
    run_attempt_sha256: str,
    discovery_artifact_sha256: str,
    discovery_custody_sha256: str,
    discovery_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "e32_2_confirmation_authorization_v1",
        "case_id": case_id,
        "registration_sha256": registration_sha256,
        "run_attempt_sha256": run_attempt_sha256,
        "discovery_artifact_sha256": discovery_artifact_sha256,
        "discovery_custody_sha256": discovery_custody_sha256,
        "frozen_discovery_nodes": discovery_bundle["discovery_frozen_nodes"],
        "frozen_induced_edges": discovery_bundle["discovery_induced_edges"],
        "status": "durable_before_confirmation_eligibility_or_model_call",
    }


def _load_exact_ground_truth(case_id: int) -> dict[str, Any]:
    """Open the hash-bound official GT only inside the reserved score stage."""
    provenance = expected_case_provenance(case_id)
    raw = load_case_ground_truth(DEFAULT_MODEL_ROOT, case_id, verify=True)
    nodes = promoted_ground_truth_internal_nodes(raw)
    edges = internal_ground_truth_edges(raw)
    counts = EXPECTED_UNIVERSE_COUNTS[case_id]
    expected = EXPECTED_GROUND_TRUTH_IDENTITIES[case_id]
    if (
        nodes != expected["nodes"]
        or edges != expected["edges"]
        or len(nodes) != counts["ground_truth_nodes"]
        or len(edges) != counts["ground_truth_edges"]
    ):
        raise RuntimeError("exact promoted ground-truth identities changed")
    return {
        "case_id": case_id,
        "node_ids": nodes,
        "directed_edges": edges,
        "raw_edges_sha256": provenance["artifact_sha256"]["edges.pkl"],
        "identity_source": "hash_bound_official_edges_pkl_promoted_exactly",
    }


def _verify_live_registration(registration_sha256: str) -> None:
    _record, observed_hash = _load_registration()
    if observed_hash != registration_sha256:
        raise ValueError("registration hash changed during E32.2 execution")


def _verify_live_custody(*, registration_sha256: str, run_attempt_sha256: str) -> None:
    _verify_live_registration(registration_sha256)
    verify_run_attempt(
        path=CANONICAL_RUN_ATTEMPT_PATH,
        registration_sha256=registration_sha256,
        run_directory=CANONICAL_RUN_DIRECTORY,
        expected_sha256=run_attempt_sha256,
    )


def _acquire_case(
    *,
    case_id: int,
    registration_sha256: str,
    run_attempt_sha256: str,
    artifact_hashes: dict[str, str],
    custody_hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], tuple[tuple[int, ...], ...]]:
    """Acquire one D→authorization→C chain with fresh models in every lane."""
    _verify_live_custody(
        registration_sha256=registration_sha256,
        run_attempt_sha256=run_attempt_sha256,
    )
    case = load_official_case(case_id)
    prior_rows = prior_candidate_token_rows(case)
    bundles: dict[str, dict[str, Any]] = {}
    authorization_hash: str | None = None

    for split in ("discovery", "confirmation"):
        _verify_live_custody(
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
        )
        request = split_request(case_id, split)
        if split == "confirmation" and authorization_hash is None:
            raise RuntimeError("confirmation lacks a durable D-derived authorization")
        excluded_rows = prior_rows
        discovery_nodes = None
        if split == "confirmation":
            excluded_rows = (*prior_rows, *bundles["discovery"]["selected_token_rows"])
            discovery_nodes = bundles["discovery"]["discovery_frozen_nodes"]

        bundle = acquire_split_bundle(
            request=request,
            case=case,
            selection_model=load_pinned_model(
                DEFAULT_MODEL_ROOT, case_id, device="cpu"
            ),
            directed_model=load_pinned_model(DEFAULT_MODEL_ROOT, case_id, device="cpu"),
            eap_model=load_pinned_model(DEFAULT_MODEL_ROOT, case_id, device="cpu"),
            eap_ig_model=load_pinned_model(DEFAULT_MODEL_ROOT, case_id, device="cpu"),
            excluded_rows=excluded_rows,
            selected_node_count=EXPECTED_UNIVERSE_COUNTS[case_id]["shortlist_nodes"],
            discovery_selected_nodes=discovery_nodes,
        )
        if split == "confirmation":
            bundle["confirmation_authorization_sha256"] = authorization_hash
        filename = f"case{case_id}_{split}.pt"
        envelope = {
            "schema_version": "e32_2_split_artifact_v1",
            "registration_sha256": registration_sha256,
            "run_attempt_sha256": run_attempt_sha256,
            "case_id": case_id,
            "split": split,
            "exact_provenance": expected_case_provenance(case_id),
            "confirmation_authorization_sha256": authorization_hash,
            "acquisition": bundle,
        }
        path = CANONICAL_RUN_DIRECTORY / filename
        _exclusive_torch_write(path, envelope)
        artifact_hashes[filename] = sha256_file(path)
        bundles[split] = bundle

        custody_path = CANONICAL_RUN_DIRECTORY / f"case{case_id}_{split}_custody.json"
        custody_record = {
            "schema_version": "e32_2_split_custody_v1",
            "case_id": case_id,
            "split": split,
            "registration_sha256": registration_sha256,
            "run_attempt_sha256": run_attempt_sha256,
            "artifact_filename": filename,
            "artifact_sha256": artifact_hashes[filename],
            "row_custody": bundle["row_custody"],
            "confirmation_authorization_sha256": authorization_hash,
        }
        exclusive_json_write(custody_path, custody_record)
        custody_hashes[custody_path.name] = sha256_file(custody_path)

        if split == "discovery":
            authorization = _authorization_record(
                case_id=case_id,
                registration_sha256=registration_sha256,
                run_attempt_sha256=run_attempt_sha256,
                discovery_artifact_sha256=artifact_hashes[filename],
                discovery_custody_sha256=custody_hashes[custody_path.name],
                discovery_bundle=bundle,
            )
            authorization_path = (
                CANONICAL_RUN_DIRECTORY
                / f"case{case_id}_confirmation_authorization.json"
            )
            exclusive_json_write(authorization_path, authorization)
            authorization_hash = sha256_file(authorization_path)
            custody_hashes[authorization_path.name] = authorization_hash

    return bundles["discovery"], bundles["confirmation"], prior_rows


def run_registered_experiment(*, confirm: bool = False) -> dict[str, Any]:
    """Run the sole registered four-panel experiment with no injection points."""
    if not confirm:
        raise ValueError("run requires confirm=True")
    _require_production_environment()
    _registration, registration_sha256 = _load_registration()
    if CANONICAL_RUN_ATTEMPT_PATH.exists():
        raise FileExistsError("canonical E32.2 run attempt is already consumed")
    if CANONICAL_RUN_DIRECTORY.exists():
        raise FileExistsError("canonical E32.2 run directory is already consumed")
    run_attempt_sha256 = reserve_run_attempt(
        path=CANONICAL_RUN_ATTEMPT_PATH,
        registration_sha256=registration_sha256,
        run_directory=CANONICAL_RUN_DIRECTORY,
    )
    CANONICAL_RUN_DIRECTORY.mkdir(parents=True, exist_ok=False)
    acquisition_attempt_path = CANONICAL_RUN_DIRECTORY / "acquisition_attempt.json"
    exclusive_json_write(
        acquisition_attempt_path,
        {
            "schema_version": "e32_2_acquisition_attempt_v1",
            "registration_sha256": registration_sha256,
            "run_attempt_sha256": run_attempt_sha256,
            "mandatory_panel": [list(key) for key in MANDATORY_PANEL_KEYS],
            "status": "durable_before_first_model_call",
        },
    )

    artifact_hashes: dict[str, str] = {}
    custody_hashes: dict[str, str] = {}
    acquisition_panel: dict[tuple[int, str], dict[str, Any]] = {}
    prior_rows_by_case: dict[int, tuple[tuple[int, ...], ...]] = {}
    score_reservation: ScoreStageReservation | None = None
    try:
        for case_id in CASE_ORDER:
            discovery, confirmation, prior_rows = _acquire_case(
                case_id=case_id,
                registration_sha256=registration_sha256,
                run_attempt_sha256=run_attempt_sha256,
                artifact_hashes=artifact_hashes,
                custody_hashes=custody_hashes,
            )
            acquisition_panel[(case_id, "discovery")] = discovery
            acquisition_panel[(case_id, "confirmation")] = confirmation
            prior_rows_by_case[case_id] = prior_rows

        _verify_live_custody(
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
        )
        panel_custody = assert_complete_fresh_panel(
            acquisition_panel, prior_rows_by_case=prior_rows_by_case
        )
        actual_accounting = aggregate_actual_query_accounting(acquisition_panel)
        final_custody_path = CANONICAL_RUN_DIRECTORY / "acquisition_custody.json"
        exclusive_json_write(
            final_custody_path,
            {
                "schema_version": "e32_2_acquisition_custody_v1",
                "registration_sha256": registration_sha256,
                "run_attempt_sha256": run_attempt_sha256,
                "acquisition_artifact_sha256": artifact_hashes,
                "stage_custody_sha256": custody_hashes,
                "panel_custody": panel_custody,
                "case_authorization_sha256": {
                    str(case_id): custody_hashes[
                        f"case{case_id}_confirmation_authorization.json"
                    ]
                    for case_id in CASE_ORDER
                },
                "actual_query_accounting": actual_accounting,
                "status": "complete_before_score_attempt",
            },
        )
        custody_hashes[final_custody_path.name] = sha256_file(final_custody_path)

        score_reservation = reserve_score_stage(
            run_directory=CANONICAL_RUN_DIRECTORY,
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
            acquisition_artifact_sha256=artifact_hashes,
            custody_artifact_sha256=custody_hashes,
        )
        # Drop the in-memory scientific responses.  The only score-stage path
        # reloads their hash-bound artifacts after verifying the reservation.
        acquisition_panel.clear()
        score_panel = load_score_stage_artifacts(score_reservation)
        for case_id in CASE_ORDER:
            validate_case_bundle_pair(
                score_panel[(case_id, "discovery")],
                score_panel[(case_id, "confirmation")],
                open_scientific_outcomes=True,
            )

        ground_truth = {
            case_id: _load_exact_ground_truth(case_id) for case_id in CASE_ORDER
        }
        from experiments.e32_interpbench.e32_2_orchestration import (
            score_complete_panel,
        )

        scientific = score_complete_panel(
            score_panel, ground_truth_by_case=ground_truth
        )
        verify_score_stage_reservation(score_reservation)
        _verify_live_custody(
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
        )
        summary = {
            **scientific,
            "registration_sha256": registration_sha256,
            "run_attempt_sha256": run_attempt_sha256,
            "acquisition_attempt_sha256": sha256_file(acquisition_attempt_path),
            "score_attempt_sha256": sha256_file(score_reservation.path),
            "acquisition_artifact_sha256": artifact_hashes,
            "custody_artifact_sha256": custody_hashes,
            "actual_query_accounting": actual_accounting,
            "claim_boundary": {
                "cases_and_ground_truth_were_inspected_during_design": True,
                "case2_and_case21_model_outcomes_unseen_at_registration": True,
                "case97_is_development_only": True,
                "target_is_promoted_internal_direct_edge_support": True,
                "pair_interaction_is_not_a_direct_edge_label": True,
                "faithfulness_not_established_by_localization": True,
                "no_external_generalization_claim": True,
            },
        }
        exclusive_json_write(CANONICAL_RUN_DIRECTORY / "summary.json", summary)
        return summary
    except BaseException as error:
        failure_path = CANONICAL_RUN_DIRECTORY / "failure.json"
        if not failure_path.exists():
            exclusive_json_write(
                failure_path,
                {
                    "schema_version": "e32_2_failure_v1",
                    "registration_sha256": registration_sha256,
                    "run_attempt_sha256": run_attempt_sha256,
                    "acquisition_attempt_sha256": sha256_file(acquisition_attempt_path),
                    "score_attempt_sha256": (
                        None
                        if score_reservation is None
                        else sha256_file(score_reservation.path)
                    ),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "completed_acquisition_artifact_sha256": artifact_hashes,
                    "completed_custody_artifact_sha256": custody_hashes,
                },
            )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("dry-run")
    subparsers.add_parser("seal")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--confirm", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.command == "dry-run":
        record = build_registration_record()
        print(
            json.dumps(
                {
                    "status": "dry_run_only_no_model_calls_no_writes",
                    "registration_sha256": _json_sha256(record),
                    "protocol": record["protocol"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "seal":
        seal_registration()
        print(
            json.dumps(
                {"registration_sha256": sha256_file(CANONICAL_REGISTRATION_PATH)}
            )
        )
        return 0
    summary = run_registered_experiment(confirm=arguments.confirm)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACQUISITION_FILENAMES",
    "CANONICAL_REGISTRATION_PATH",
    "CANONICAL_RUN_ATTEMPT_PATH",
    "CANONICAL_RUN_DIRECTORY",
    "CUSTODY_FILENAMES",
    "EXPECTED_GROUND_TRUTH_IDENTITIES",
    "SOURCE_RELATIVE_PATHS",
    "ScoreStageReservation",
    "build_registration_record",
    "exclusive_json_write",
    "load_score_stage_artifacts",
    "main",
    "reserve_run_attempt",
    "reserve_score_stage",
    "run_registered_experiment",
    "seal_registration",
    "verify_registration_record",
    "verify_run_attempt",
    "verify_score_stage_reservation",
]
