# ruff: noqa: EXE002
"""Canonical write-once production driver for the lean E32.1 benchmark.

The driver intentionally has one output path and no injectable scientific
callables.  A dry run verifies exact local/external dependencies without
loading a model or writing an artifact.  ``seal`` must precede ``run``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from experiments.e32_interpbench.acquisition import (
    acquire_official_pair_atlas,
    build_case_patch_runner,
)
from experiments.e32_interpbench.adapter import (
    DEFAULT_ENV_DIR,
    DEFAULT_IIT_DIR,
    DEFAULT_MODEL_ROOT,
    DEFAULT_OFFICIAL_CODE_DIR,
    DEFAULT_TRACR_DIR,
    IIT_REVISION,
    INTERPBENCH_CODE_REVISION,
    TRACR_REVISION,
    behaviorally_eligible_pair,
    internal_ground_truth_edges,
    load_case_ground_truth,
    load_official_case,
    load_pinned_model,
    promoted_ground_truth_internal_nodes,
    sha256_file,
    verify_case_artifacts,
    verify_git_revision,
)
from experiments.e32_interpbench.e32_1_orchestration import (
    AcquiredSplit,
    ExactGroundTruth,
    run_hierarchical_panel,
)
from experiments.e32_interpbench.e32_1_protocol import (
    CASE_ORDER,
    SplitRequest,
    expected_case_provenance,
    protocol_dry_run_manifest,
)

CANONICAL_REGISTRATION_PATH = (
    REPOSITORY_ROOT / "notes" / "E32.1-EXACT-GT-REGISTRATION.json"
)
RUNTIME_SOURCE_PACKAGES = ("tracr", "iit", "transformer_lens")
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
}
CANONICAL_RUN_DIRECTORY = REPOSITORY_ROOT / "runs" / "e32_1_exact_gt_v1"
SOURCE_RELATIVE_PATHS = (
    "experiments/e32_interpbench/__init__.py",
    "experiments/e32_interpbench/adapter.py",
    "experiments/e32_interpbench/component_patching.py",
    "experiments/e32_interpbench/acquisition.py",
    "experiments/e32_interpbench/protocol.py",
    "experiments/e32_interpbench/methods.py",
    "experiments/e32_interpbench/scoring.py",
    "experiments/e32_interpbench/analysis.py",
    "experiments/e32_interpbench/e32_1_protocol.py",
    "experiments/e32_interpbench/e32_1_orchestration.py",
    "experiments/e32_interpbench/e32_1_run.py",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def exclusive_json_write(path: Path, value: Any) -> None:
    """Write JSON once using the filesystem's exclusive-create primitive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _exclusive_torch_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())


def _source_hashes() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative in SOURCE_RELATIVE_PATHS:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing registered source: {path}")
        observed[relative] = sha256_file(path)
    return observed


def _clean_git_revision(checkout: Path, expected_revision: str) -> str:
    revision = verify_git_revision(checkout, expected_revision)
    tracked_status = subprocess.run(
        [
            "git",
            "-C",
            str(checkout.resolve()),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked_status:
        raise RuntimeError(f"pinned checkout has tracked changes: {checkout}")
    return revision


def _dependency_record() -> dict[str, Any]:
    source_revisions = {
        "interpbench": _clean_git_revision(
            DEFAULT_OFFICIAL_CODE_DIR, INTERPBENCH_CODE_REVISION
        ),
        "tracr": _clean_git_revision(DEFAULT_TRACR_DIR, TRACR_REVISION),
        "iit": _clean_git_revision(DEFAULT_IIT_DIR, IIT_REVISION),
    }
    artifacts = {
        str(case_id): verify_case_artifacts(DEFAULT_MODEL_ROOT, case_id)
        for case_id in CASE_ORDER
    }
    return {
        "source_directories": {
            "interpbench": str(DEFAULT_OFFICIAL_CODE_DIR.resolve()),
            "tracr": str(DEFAULT_TRACR_DIR.resolve()),
            "iit": str(DEFAULT_IIT_DIR.resolve()),
            "model_root": str(DEFAULT_MODEL_ROOT.resolve()),
        },
        "source_revisions": source_revisions,
        "tracked_worktrees_clean": {name: True for name in source_revisions},
        "case_artifact_sha256": artifacts,
    }


def _installed_version(distribution: str) -> str:
    """Return an auditable value even in the deliberately lean test environment."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _runtime_package_source_record(package_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.submodule_search_locations is None:
        return {"status": "not_installed"}
    locations = tuple(spec.submodule_search_locations)
    if len(locations) != 1:
        raise RuntimeError(
            f"expected one source root for {package_name}, got {locations}"
        )
    root = Path(locations[0]).resolve()
    source_hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    if not source_hashes:
        raise RuntimeError(f"runtime package has no auditable sources: {package_name}")
    return {
        "status": "present",
        "root": str(root),
        "file_count": len(source_hashes),
        "tree_sha256": _json_sha256(source_hashes),
        "file_sha256": source_hashes,
    }


def _environment_record() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformer_lens": _installed_version("transformer-lens"),
        "numpy": _installed_version("numpy"),
        "jax": _installed_version("jax"),
        "jaxlib": _installed_version("jaxlib"),
        "executable": str(Path(os.sys.executable).absolute()),
        "sys_prefix": str(Path(os.sys.prefix).resolve()),
        "platform": platform.platform(),
        "runtime_package_sources": {
            package_name: _runtime_package_source_record(package_name)
            for package_name in RUNTIME_SOURCE_PACKAGES
        },
    }


def _require_production_environment() -> None:
    observed = _environment_record()
    required = {
        "python": "3.11.9",
        "torch": "2.3.1",
        "transformer_lens": "1.19.0",
        "numpy": "1.26.4",
        "jax": "0.4.30",
        "jaxlib": "0.4.30",
        "executable": str((DEFAULT_ENV_DIR / "bin" / "python").absolute()),
        "sys_prefix": str(DEFAULT_ENV_DIR.resolve()),
    }
    wrong = {
        key: {"expected": value, "observed": observed[key]}
        for key, value in required.items()
        if observed[key] != value
    }
    if wrong:
        raise RuntimeError(f"E32.1 requires the pinned production environment: {wrong}")
    site_packages = DEFAULT_ENV_DIR / "lib" / "python3.11" / "site-packages"
    wrong_roots = {
        package_name: source_record
        for package_name, source_record in observed["runtime_package_sources"].items()
        if source_record.get("status") != "present"
        or source_record.get("root") != str((site_packages / package_name).resolve())
    }
    if wrong_roots:
        raise RuntimeError(
            "E32.1 runtime package sources are not the pinned environment: "
            f"{wrong_roots}"
        )
    wrong_trees = {
        package_name: {
            "expected": EXPECTED_RUNTIME_PACKAGE_TREES[package_name],
            "observed": {
                "file_count": source_record["file_count"],
                "tree_sha256": source_record["tree_sha256"],
            },
        }
        for package_name, source_record in observed["runtime_package_sources"].items()
        if source_record["file_count"]
        != EXPECTED_RUNTIME_PACKAGE_TREES[package_name]["file_count"]
        or source_record["tree_sha256"]
        != EXPECTED_RUNTIME_PACKAGE_TREES[package_name]["tree_sha256"]
    }
    if wrong_trees:
        raise RuntimeError(f"E32.1 runtime package source drift: {wrong_trees}")


def build_registration_record() -> dict[str, Any]:
    """Build the deterministic record that is checked again before every call."""
    return {
        "schema_version": "e32_1_exact_gt_registration_v1",
        "canonical_registration_path": str(CANONICAL_REGISTRATION_PATH.resolve()),
        "canonical_run_directory": str(CANONICAL_RUN_DIRECTORY.resolve()),
        "case_order": CASE_ORDER,
        "protocol": protocol_dry_run_manifest(),
        "source_sha256": _source_hashes(),
        "dependencies": _dependency_record(),
        "environment": _environment_record(),
        "execution": {
            "device": "cpu",
            "model_dtype": "checkpoint_float32",
            "single_canonical_run": True,
            "registration_precedes_every_model_call": True,
            "behavior_selection_precedes_patched_outcomes": True,
            "all_selected_clean_source_rows_unique": True,
            "confirmation_rows_exclude_discovery_rows_response_blindly": True,
            "atlas_artifacts_and_summary_are_write_once": True,
        },
    }


def verify_registration_record(record: dict[str, Any]) -> None:
    expected = build_registration_record()
    if _canonical_json_bytes(record) != _canonical_json_bytes(expected):
        raise ValueError(
            "registration differs from exact current protocol/dependencies"
        )


def _load_registration() -> tuple[dict[str, Any], str]:
    if not CANONICAL_REGISTRATION_PATH.is_file():
        raise FileNotFoundError("E32.1 registration is not sealed")
    record = json.loads(CANONICAL_REGISTRATION_PATH.read_text())
    verify_registration_record(record)
    return record, sha256_file(CANONICAL_REGISTRATION_PATH)


def seal_registration() -> dict[str, Any]:
    _require_production_environment()
    if CANONICAL_RUN_DIRECTORY.exists():
        raise FileExistsError("canonical E32.1 run directory is not pristine")
    record = build_registration_record()
    # Recheck immediately before the exclusive write so dependency work cannot
    # hide a concurrently created outcome directory.
    if CANONICAL_RUN_DIRECTORY.exists():
        raise FileExistsError("canonical E32.1 run directory is not pristine")
    exclusive_json_write(CANONICAL_REGISTRATION_PATH, record)
    return record


def _selected_token_rows(atlas: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(token) for token in row)
        for key in ("clean_inputs", "corrupt_inputs")
        for row in torch.as_tensor(atlas[key], device="cpu").tolist()
    )


def _ground_truth(case_id: int) -> ExactGroundTruth:
    raw = load_case_ground_truth(DEFAULT_MODEL_ROOT, case_id, verify=True)
    return ExactGroundTruth(
        case_id=case_id,
        node_ids=promoted_ground_truth_internal_nodes(raw),
        directed_edges=internal_ground_truth_edges(raw),
        raw_edges_sha256=expected_case_provenance(case_id)["artifact_sha256"][
            "edges.pkl"
        ],
    )


def run_registered_experiment(*, confirm: bool = False) -> dict[str, Any]:
    """Execute the sole registered hierarchy; no scientific injection points."""
    if not confirm:
        raise ValueError("run requires confirm=True")
    _require_production_environment()
    _registration, registration_sha256 = _load_registration()
    if CANONICAL_RUN_DIRECTORY.exists():
        raise FileExistsError("canonical E32.1 run directory is already consumed")
    CANONICAL_RUN_DIRECTORY.mkdir(parents=True, exist_ok=False)
    attempt_path = CANONICAL_RUN_DIRECTORY / "attempt.json"
    exclusive_json_write(
        attempt_path,
        {
            "schema_version": "e32_1_attempt_v1",
            "registration_sha256": registration_sha256,
            "maximum_model_examples_if_primary_fails": 26_592,
            "maximum_model_examples_if_all_cases_run": 112_032,
            "status": "durable_before_first_model_or_gt_outcome_open",
        },
    )

    case_runtime: dict[int, dict[str, Any]] = {}
    acquired: dict[tuple[int, str], AcquiredSplit] = {}
    artifact_paths: dict[str, str] = {}

    def acquire(request: SplitRequest) -> AcquiredSplit:
        # Fail closed on any source, checkpoint, environment, or registration
        # drift immediately before each split's first model call.
        current_registration, current_sha = _load_registration()
        del current_registration
        if current_sha != registration_sha256:
            raise ValueError("registration hash changed during execution")
        runtime = case_runtime.get(request.case_id)
        if runtime is None:
            runtime = {
                "case": load_official_case(request.case_id),
                "model": load_pinned_model(
                    DEFAULT_MODEL_ROOT, request.case_id, device="cpu"
                ),
            }
            case_runtime[request.case_id] = runtime
        excluded_rows: tuple[tuple[int, ...], ...] = ()
        if request.split == "confirmation":
            discovery_key = (request.case_id, "discovery")
            if discovery_key not in acquired:
                raise RuntimeError("confirmation requested before discovery")
            excluded_rows = _selected_token_rows(acquired[discovery_key].atlas)
        eligible = behaviorally_eligible_pair(
            runtime["model"],
            runtime["case"],
            required_count=request.required_count,
            candidate_count=request.candidate_count,
            clean_seed=request.clean_seed,
            corrupt_seed=request.corrupt_seed,
            device="cpu",
            excluded_token_rows=excluded_rows,
            require_unique_selected_token_rows=True,
        )
        runner = build_case_patch_runner(
            case_id=request.case_id,
            model=runtime["model"],
            condition_batch_size=request.atlas_condition_batch_size,
        )
        atlas = acquire_official_pair_atlas(
            case_id=request.case_id,
            runner=runner,
            official_pair=eligible,
            smoke_condition_count=request.smoke_condition_count,
            smoke_condition_batch_size=request.smoke_condition_batch_size,
        )
        split = AcquiredSplit(
            request=request,
            atlas=atlas,
            exact_provenance=expected_case_provenance(request.case_id),
        )
        filename = f"case{request.case_id}_{request.split}_atlas.pt"
        path = CANONICAL_RUN_DIRECTORY / filename
        _exclusive_torch_write(
            path,
            {
                "schema_version": "e32_1_split_envelope_v1",
                "registration_sha256": registration_sha256,
                "request": asdict(request),
                "exact_provenance": split.exact_provenance,
                "atlas": atlas,
            },
        )
        artifact_paths[filename] = sha256_file(path)
        acquired[(request.case_id, request.split)] = split
        return split

    try:
        score = run_hierarchical_panel(acquire, _ground_truth)
        _load_registration()
        actual_examples = sum(
            int(case["query_accounting"]["combined_model_examples"])
            for case in score["case_results"].values()
        )
        expected_examples = (
            26_592
            if score["secondary_status"] == "not_run_primary_gate_failed"
            else 112_032
        )
        if actual_examples != expected_examples:
            raise RuntimeError("final query accounting differs from registered total")
        summary = {
            **score,
            "registration_sha256": registration_sha256,
            "attempt_sha256": sha256_file(attempt_path),
            "atlas_artifact_sha256": artifact_paths,
            "actual_model_examples": actual_examples,
            "claim_boundary": {
                "benchmark_was_not_hidden_during_method_design": True,
                "internal_promoted_gt_only": True,
                "no_eap_superiority_claim_in_e32_1": True,
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
                    "schema_version": "e32_1_failure_v1",
                    "registration_sha256": registration_sha256,
                    "attempt_sha256": sha256_file(attempt_path),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "completed_atlas_artifact_sha256": artifact_paths,
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
    "CANONICAL_REGISTRATION_PATH",
    "CANONICAL_RUN_DIRECTORY",
    "SOURCE_RELATIVE_PATHS",
    "build_registration_record",
    "exclusive_json_write",
    "main",
    "run_registered_experiment",
    "seal_registration",
    "verify_registration_record",
]
