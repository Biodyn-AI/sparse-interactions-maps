# ruff: noqa: EXE002
"""Canonical write-once driver for prospective E32.3 sparse-core scoring.

The public production entrypoints have no scientific callback injection.  A
private executor seam exists solely so tests can exercise crash custody without
running InterpBench models.  The phase order is global: all input panels are
behavior-screened, all discoveries are persisted, all three authorizations are
persisted, and only then may any confirmation intervention begin.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
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
    DEFAULT_MODEL_ROOT,
    sha256_file,
)
from experiments.e32_interpbench.e32_2_run import (
    _dependency_record as _e32_2_dependency_record,
)
from experiments.e32_interpbench.e32_2_run import (
    _environment_record as _e32_2_environment_record,
)
from experiments.e32_interpbench.e32_2_run import (
    _exclusive_torch_write,
    _require_production_environment,
    exclusive_json_write,
)
from experiments.e32_interpbench.e32_3_cohort import (
    CASE_MANIFESTS,
    CASE_ORDER,
    acquire_case_panel,
    case_universe,
    load_verified_model,
    load_verified_official_case,
    verify_case_artifacts,
)
from experiments.e32_interpbench.e32_3_orchestration import (
    MANDATORY_PANEL_KEYS,
    aggregate_actual_query_accounting,
    freeze_discovery_supports,
    score_complete_panel,
    validate_exact_panel_alignment,
    validate_global_behavior_screen,
    validate_split_request_alignment,
)
from experiments.e32_interpbench.e32_3_protocol import protocol_manifest


@dataclass(frozen=True)
class RunPaths:
    registration_path: Path
    run_attempt_path: Path
    run_directory: Path

    def resolved(self) -> RunPaths:
        return RunPaths(
            self.registration_path.expanduser().resolve(),
            self.run_attempt_path.expanduser().resolve(),
            self.run_directory.expanduser().resolve(),
        )


CANONICAL_PATHS = RunPaths(
    REPOSITORY_ROOT / "notes" / "E32.3-SPARSE-CORE-REGISTRATION.json",
    REPOSITORY_ROOT / "notes" / "E32.3-SPARSE-CORE-RUN-ATTEMPT.json",
    REPOSITORY_ROOT / "runs" / "e32_3_sparse_core_v1",
).resolved()
CANONICAL_REGISTRATION_PATH = CANONICAL_PATHS.registration_path
CANONICAL_RUN_ATTEMPT_PATH = CANONICAL_PATHS.run_attempt_path
CANONICAL_RUN_DIRECTORY = CANONICAL_PATHS.run_directory

_SOURCE_ROOT_MODULES = (
    "experiments.e32_interpbench.e32_3_run",
    "experiments.e32_interpbench.e32_3_orchestration",
    "experiments.e32_interpbench.e32_3_protocol",
    "experiments.e32_interpbench.e32_3_cohort",
    "experiments.e32_interpbench.e32_3_acquisition",
    "experiments.e32_interpbench.e32_3_baselines",
    "experiments.e32_interpbench.e32_3_scoring",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validated_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _module_source_path(module: str) -> Path | None:
    if not module.startswith("experiments.e32_interpbench"):
        return None
    path = REPOSITORY_ROOT / f"{module.replace('.', '/')}.py"
    if path.is_file():
        return path
    package = REPOSITORY_ROOT / module.replace(".", "/") / "__init__.py"
    return package if package.is_file() else None


def _source_relative_paths() -> tuple[str, ...]:
    """Discover the exact local transitive import closure from registered roots."""
    pending = list(_SOURCE_ROOT_MODULES)
    visited: set[str] = set()
    paths: set[str] = {"experiments/e32_interpbench/__init__.py"}
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        path = _module_source_path(module)
        if path is None:
            raise FileNotFoundError(f"registered local module has no source: {module}")
        paths.add(str(path.relative_to(REPOSITORY_ROOT)))
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports = (node.module,)
            else:
                continue
            for imported in imports:
                if (
                    imported.startswith("experiments.e32_interpbench")
                    and imported not in visited
                    and _module_source_path(imported) is not None
                ):
                    pending.append(imported)
    return tuple(sorted(paths))


SOURCE_RELATIVE_PATHS = _source_relative_paths()


def _source_hashes() -> dict[str, str]:
    observed_paths = _source_relative_paths()
    if observed_paths != SOURCE_RELATIVE_PATHS:
        raise RuntimeError("registered local source closure changed")
    return {
        relative: sha256_file(REPOSITORY_ROOT / relative)
        for relative in SOURCE_RELATIVE_PATHS
    }


def _panel_identity_sha256(
    *, case_id: int, split: str, selected_token_rows: Sequence[Sequence[int]]
) -> str:
    return _json_sha256(
        {
            "case_id": case_id,
            "split": split,
            "selected_token_rows": tuple(tuple(row) for row in selected_token_rows),
        }
    )


class _ProductionExecutor:
    """Fixed production bridge; construction is private to public entrypoints."""

    def registration_dependencies(self) -> dict[str, Any]:
        production = _require_production_environment()
        dependency = _e32_2_dependency_record()
        return {
            "pinned_checkouts_and_runtime_trees": dependency,
            "pinned_environment": production,
            "environment_replay": _e32_2_environment_record(),
            "e32_3_case_artifact_sha256": {
                str(case_id): verify_case_artifacts(DEFAULT_MODEL_ROOT, case_id)
                for case_id in CASE_ORDER
            },
        }

    def screen_all_cases(self) -> tuple[dict[str, Any], dict[int, Any]]:
        records: dict[str, Any] = {}
        panels: dict[int, Any] = {}
        for case_id in CASE_ORDER:
            case = load_verified_official_case(case_id)
            model = load_verified_model(case_id, device="cpu")
            panel = acquire_case_panel(case_id, model=model, case=case, device="cpu")
            panels[case_id] = panel
            records[str(case_id)] = _behavior_screen_record(panel)
        return records, panels

    def acquire_discovery(self, case_id: int, panel: Any) -> dict[str, Any]:
        return _acquire_production_split(case_id, "discovery", panel, None)

    def acquire_confirmation(
        self, case_id: int, panel: Any, authorization: Mapping[str, Any]
    ) -> dict[str, Any]:
        return _acquire_production_split(case_id, "confirmation", panel, authorization)

    def score_complete_panel(
        self, panel: Mapping[tuple[int, str], Mapping[str, Any]]
    ) -> dict[str, Any]:
        return score_complete_panel(panel)


def _behavior_screen_record(panel: Any) -> dict[str, Any]:
    from experiments.e32_interpbench.e32_2_acquisition import token_rows

    case_id = int(panel.case_id)
    discovery = panel.discovery.pair
    confirmation = panel.confirmation.pair
    return {
        "case_id": case_id,
        "behavior_only": True,
        "smoke_rows": tuple(panel.smoke_rows),
        "discovery_clean_rows": token_rows(
            discovery.clean_inputs, label="discovery clean inputs"
        ),
        "discovery_corrupt_rows": token_rows(
            discovery.corrupt_inputs, label="discovery corrupt inputs"
        ),
        "confirmation_clean_rows": token_rows(
            confirmation.clean_inputs, label="confirmation clean inputs"
        ),
        "confirmation_corrupt_rows": token_rows(
            confirmation.corrupt_inputs, label="confirmation corrupt inputs"
        ),
    }


def _acquire_production_split(
    case_id: int,
    split: str,
    panel: Any,
    authorization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Production implementation is completed below the lifecycle helpers."""
    # Kept as one narrow bridge so registration and custody code cannot adapt
    # scientific choices.  It is deliberately imported lazily: dry-run and
    # registration verification perform no model acquisition.
    return _live_split_bundle(
        case_id=case_id,
        split=split,
        eligible=getattr(panel, split),
        authorization=authorization,
    )


def _build_registration_record(*, executor: Any, paths: RunPaths) -> dict[str, Any]:
    resolved = paths.resolved()
    return {
        "schema_version": "e32_3_sparse_core_registration_v1",
        "canonical_registration_path": str(resolved.registration_path),
        "canonical_run_attempt_path": str(resolved.run_attempt_path),
        "canonical_run_directory": str(resolved.run_directory),
        "protocol": protocol_manifest(),
        "split_request_alignment": validate_split_request_alignment(),
        "source_sha256": _source_hashes(),
        "dependencies": executor.registration_dependencies(),
        "execution": {
            "device": "cpu",
            "all_cases_mandatory": True,
            "global_behavior_screen_before_any_intervention": True,
            "all_discoveries_before_any_authorization": True,
            "all_authorizations_before_any_confirmation": True,
            "discovery_reload_before_authorization_required": True,
            "score_attempt_before_score_stage_confirmation_outcome_reload_or_scoring": True,
            "exact_score_replay_required": True,
            "production_executor_injectable": False,
            "ground_truth_known_pre_run": True,
            "ground_truth_used_for_cohort_selection": True,
            "ground_truth_used_for_within_case_selection_or_primary_scoring": False,
        },
    }


def build_registration_record() -> dict[str, Any]:
    """Build the canonical record without intervention/model acquisition."""
    return _build_registration_record(
        executor=_ProductionExecutor(), paths=CANONICAL_PATHS
    )


def _seal_registration(*, executor: Any, paths: RunPaths) -> dict[str, Any]:
    resolved = paths.resolved()
    for path, label in (
        (resolved.registration_path, "registration"),
        (resolved.run_attempt_path, "run attempt"),
        (resolved.run_directory, "run directory"),
    ):
        if path.exists():
            raise FileExistsError(f"canonical E32.3 {label} is already consumed")
    record = _build_registration_record(executor=executor, paths=resolved)
    # Dependency verification can be slow.  Recheck irreversible state after
    # it completes so a concurrent attempt cannot race the registration write.
    if resolved.run_attempt_path.exists() or resolved.run_directory.exists():
        raise FileExistsError("canonical E32.3 run state changed during registration")
    if resolved.registration_path.exists():
        raise FileExistsError("canonical E32.3 registration was concurrently written")
    exclusive_json_write(resolved.registration_path, record)
    return record


def seal_registration() -> dict[str, Any]:
    """Write the sole canonical production registration exactly once."""
    return _seal_registration(executor=_ProductionExecutor(), paths=CANONICAL_PATHS)


def _load_registration(*, executor: Any, paths: RunPaths) -> tuple[dict[str, Any], str]:
    resolved = paths.resolved()
    try:
        record = json.loads(resolved.registration_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("E32.3 registration is missing or malformed") from error
    expected = _build_registration_record(executor=executor, paths=resolved)
    if _canonical_json_bytes(record) != _canonical_json_bytes(expected):
        raise ValueError("registration differs from exact current source/dependencies")
    return record, sha256_file(resolved.registration_path)


def _reserve_run_attempt(*, paths: RunPaths, registration_sha256: str) -> str:
    resolved = paths.resolved()
    if (
        resolved.run_attempt_path == resolved.run_directory
        or resolved.run_directory
        in (
            resolved.run_attempt_path,
            *resolved.run_attempt_path.parents,
        )
    ):
        raise ValueError("run-attempt sentinel must be external to the run directory")
    payload = {
        "schema_version": "e32_3_run_attempt_v1",
        "registration_sha256": _validated_sha256(
            registration_sha256, label="registration_sha256"
        ),
        "canonical_run_directory": str(resolved.run_directory),
        "status": "irreversible_attempt_reserved_before_run_directory_or_model",
    }
    exclusive_json_write(resolved.run_attempt_path, payload)
    return sha256_file(resolved.run_attempt_path)


def _verify_live_custody(
    *,
    executor: Any,
    paths: RunPaths,
    registration_sha256: str,
    run_attempt_sha256: str,
) -> None:
    _record, observed_registration = _load_registration(executor=executor, paths=paths)
    if observed_registration != registration_sha256:
        raise ValueError("registration hash changed during E32.3 execution")
    expected_attempt = {
        "schema_version": "e32_3_run_attempt_v1",
        "registration_sha256": registration_sha256,
        "canonical_run_directory": str(paths.run_directory.resolve()),
        "status": "irreversible_attempt_reserved_before_run_directory_or_model",
    }
    try:
        observed_attempt = json.loads(paths.run_attempt_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("run-attempt sentinel is missing or malformed") from error
    if (
        _canonical_json_bytes(observed_attempt)
        != _canonical_json_bytes(expected_attempt)
        or sha256_file(paths.run_attempt_path) != run_attempt_sha256
    ):
        raise ValueError("run-attempt sentinel changed during E32.3 execution")


def _write_split_artifact(
    *,
    paths: RunPaths,
    registration_sha256: str,
    run_attempt_sha256: str,
    case_id: int,
    split: str,
    bundle: Mapping[str, Any],
    expected_rows: Sequence[Sequence[int]],
    authorization_sha256: str | None,
) -> tuple[str, str]:
    _validate_split_bundle(
        bundle,
        case_id=case_id,
        split=split,
        expected_rows=expected_rows,
        authorization_sha256=authorization_sha256,
    )
    filename = f"case{case_id}_{split}.pt"
    envelope = {
        "schema_version": "e32_3_split_artifact_v1",
        "registration_sha256": registration_sha256,
        "run_attempt_sha256": run_attempt_sha256,
        "case_id": case_id,
        "split": split,
        "confirmation_authorization_sha256": authorization_sha256,
        "acquisition": dict(bundle),
    }
    artifact_path = paths.run_directory / filename
    _exclusive_torch_write(artifact_path, envelope)
    artifact_sha256 = sha256_file(artifact_path)
    custody_path = paths.run_directory / f"case{case_id}_{split}_custody.json"
    custody = {
        "schema_version": "e32_3_split_custody_v1",
        "case_id": case_id,
        "split": split,
        "registration_sha256": registration_sha256,
        "run_attempt_sha256": run_attempt_sha256,
        "artifact_filename": filename,
        "artifact_sha256": artifact_sha256,
        "panel_identity_sha256": bundle["panel_identity_sha256"],
        "selected_role_row_count": len(tuple(bundle["selected_token_rows"])),
        "behavioral_eligibility": bundle["behavioral_eligibility"],
        "query_accounting": bundle["query_accounting"],
        "confirmation_authorization_sha256": authorization_sha256,
    }
    exclusive_json_write(custody_path, custody)
    return artifact_sha256, sha256_file(custody_path)


def _validate_split_bundle(
    bundle: Mapping[str, Any],
    *,
    case_id: int,
    split: str,
    expected_rows: Sequence[Sequence[int]],
    authorization_sha256: str | None,
) -> None:
    expected_schema = f"e32_3_{split}_acquisition_v1"
    if (
        not isinstance(bundle, Mapping)
        or bundle.get("schema_version") != expected_schema
        or bundle.get("case_id") != case_id
        or bundle.get("split") != split
    ):
        raise ValueError("split acquisition identity changed")
    rows = tuple(tuple(row) for row in bundle.get("selected_token_rows", ()))
    expected = tuple(tuple(row) for row in expected_rows)
    required = 64 if split == "discovery" else 128
    if rows != expected or len(rows) != 2 * required or len(set(rows)) != len(rows):
        raise RuntimeError("split acquisition lost its behavior-screened rows")
    identity = _panel_identity_sha256(
        case_id=case_id, split=split, selected_token_rows=rows
    )
    if bundle.get("panel_identity_sha256") != identity:
        raise ValueError("split panel identity hash changed")
    eligibility = bundle.get("behavioral_eligibility")
    if (
        not isinstance(eligibility, Mapping)
        or eligibility.get("required_count") != required
        or len(tuple(eligibility.get("selected_candidate_indices", ()))) != required
        or eligibility.get("all_selected_pairs_correct") is not True
        or eligibility.get("selection_policy")
        != (
            "first_jointly_correct_with_globally_unique_clean_source_rows_and_"
            "response_blind_exclusions"
        )
    ):
        raise ValueError("BehaviorallyEligiblePair proof changed or was discarded")
    family_identity = bundle.get("family_panel_identity_sha256")
    if (
        not isinstance(family_identity, Mapping)
        or tuple(family_identity) != ("whole_component", "exact_fine", "eap_ig5")
        or any(value != identity for value in family_identity.values())
    ):
        raise ValueError("whole/exact/EAP acquisitions used different input panels")
    validate_exact_panel_alignment(bundle)
    if split == "confirmation":
        if (
            authorization_sha256 is None
            or bundle.get("confirmation_authorization_sha256") != authorization_sha256
        ):
            raise ValueError("confirmation artifact lacks its durable authorization")
    elif authorization_sha256 is not None:
        raise ValueError("discovery cannot carry confirmation authorization")


def _load_artifact(
    *,
    path: Path,
    expected_sha256: str,
    registration_sha256: str,
    run_attempt_sha256: str,
    case_id: int,
    split: str,
) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("split artifact changed after custody")
    envelope = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(envelope, Mapping)
        or envelope.get("schema_version") != "e32_3_split_artifact_v1"
        or envelope.get("registration_sha256") != registration_sha256
        or envelope.get("run_attempt_sha256") != run_attempt_sha256
        or envelope.get("case_id") != case_id
        or envelope.get("split") != split
        or not isinstance(envelope.get("acquisition"), dict)
    ):
        raise ValueError("split artifact envelope changed")
    return envelope["acquisition"]


def _authorization_record(
    *,
    case_id: int,
    registration_sha256: str,
    run_attempt_sha256: str,
    discovery_artifact_sha256: str,
    discovery_custody_sha256: str,
    frozen_supports: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "e32_3_confirmation_authorization_v1",
        "case_id": case_id,
        "registration_sha256": registration_sha256,
        "run_attempt_sha256": run_attempt_sha256,
        "discovery_artifact_sha256": discovery_artifact_sha256,
        "discovery_custody_sha256": discovery_custody_sha256,
        "frozen_supports": dict(frozen_supports),
        "status": "durable_before_confirmation_intervention",
    }


@dataclass(frozen=True)
class _ScoreReservation:
    path: Path
    payload_sha256: str
    registration_sha256: str
    run_attempt_sha256: str
    artifact_sha256: Mapping[str, str]
    custody_sha256: Mapping[str, str]


def _reserve_score(
    *,
    paths: RunPaths,
    registration_sha256: str,
    run_attempt_sha256: str,
    artifact_hashes: Mapping[str, str],
    custody_hashes: Mapping[str, str],
) -> _ScoreReservation:
    expected_artifacts = tuple(
        f"case{case_id}_{split}.pt" for case_id, split in MANDATORY_PANEL_KEYS
    )
    if tuple(artifact_hashes) != expected_artifacts:
        raise ValueError("score attempt requires the complete ordered panel")
    for filename, digest in {**artifact_hashes, **custody_hashes}.items():
        _validated_sha256(digest, label=f"hash for {filename}")
        path = paths.run_directory / filename
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError("score-bound artifact differs before reservation")
    payload = {
        "schema_version": "e32_3_score_attempt_v1",
        "registration_sha256": registration_sha256,
        "run_attempt_sha256": run_attempt_sha256,
        "acquisition_artifact_sha256": dict(artifact_hashes),
        "custody_artifact_sha256": dict(custody_hashes),
        "status": ("durable_before_score_stage_confirmation_outcome_reload_or_scoring"),
    }
    path = paths.run_directory / "score_attempt.json"
    exclusive_json_write(path, payload)
    return _ScoreReservation(
        path,
        _json_sha256(payload),
        registration_sha256,
        run_attempt_sha256,
        dict(artifact_hashes),
        dict(custody_hashes),
    )


def _verify_score(reservation: _ScoreReservation) -> None:
    try:
        payload = json.loads(reservation.path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("score attempt is missing or malformed") from error
    if _json_sha256(payload) != reservation.payload_sha256:
        raise ValueError("score attempt changed after reservation")
    for filename, digest in {
        **reservation.artifact_sha256,
        **reservation.custody_sha256,
    }.items():
        path = reservation.path.parent / filename
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError("score-bound artifact changed after reservation")


def _load_score_panel(reservation: _ScoreReservation) -> dict[tuple[int, str], Any]:
    _verify_score(reservation)
    panel: dict[tuple[int, str], Any] = {}
    for case_id, split in MANDATORY_PANEL_KEYS:
        filename = f"case{case_id}_{split}.pt"
        panel[(case_id, split)] = _load_artifact(
            path=reservation.path.parent / filename,
            expected_sha256=reservation.artifact_sha256[filename],
            registration_sha256=reservation.registration_sha256,
            run_attempt_sha256=reservation.run_attempt_sha256,
            case_id=case_id,
            split=split,
        )
    _verify_score(reservation)
    return panel


def _screen_rows(record: Mapping[str, Any], split: str) -> tuple[tuple[int, ...], ...]:
    return (
        *tuple(tuple(row) for row in record[f"{split}_clean_rows"]),
        *tuple(tuple(row) for row in record[f"{split}_corrupt_rows"]),
    )


def _run_registered_experiment(
    *, executor: Any, paths: RunPaths, confirm: bool = False
) -> dict[str, Any]:
    if not confirm:
        raise ValueError("run requires confirm=True")
    resolved = paths.resolved()
    _registration, registration_sha256 = _load_registration(
        executor=executor, paths=resolved
    )
    if resolved.run_attempt_path.exists():
        raise FileExistsError("canonical E32.3 run attempt is already consumed")
    if resolved.run_directory.exists():
        raise FileExistsError("canonical E32.3 run directory is already consumed")
    run_attempt_sha256 = _reserve_run_attempt(
        paths=resolved, registration_sha256=registration_sha256
    )
    resolved.run_directory.mkdir(parents=True, exist_ok=False)
    acquisition_attempt = resolved.run_directory / "acquisition_attempt.json"
    exclusive_json_write(
        acquisition_attempt,
        {
            "schema_version": "e32_3_acquisition_attempt_v1",
            "registration_sha256": registration_sha256,
            "run_attempt_sha256": run_attempt_sha256,
            "phase_order": (
                "global_behavior_screen",
                "all_discoveries",
                "all_authorizations",
                "all_confirmations",
                "score_attempt",
                "score_and_exact_replay",
            ),
            "status": "durable_before_first_behavior_or_intervention_model_call",
        },
    )

    artifact_hashes: dict[str, str] = {}
    custody_hashes: dict[str, str] = {}
    accounting_panel: dict[tuple[int, str], dict[str, Any]] = {}
    authorizations: dict[int, dict[str, Any]] = {}
    score_reservation: _ScoreReservation | None = None
    try:
        screen_cases, case_panels = executor.screen_all_cases()
        if tuple(case_panels) != CASE_ORDER:
            raise ValueError("executor did not return every screened case in order")
        screen_audit = validate_global_behavior_screen(screen_cases)
        screen_path = resolved.run_directory / "global_behavior_screen.json"
        exclusive_json_write(
            screen_path,
            {
                "schema_version": "e32_3_global_behavior_screen_v1",
                "registration_sha256": registration_sha256,
                "run_attempt_sha256": run_attempt_sha256,
                "cases": screen_cases,
                "audit": screen_audit,
                "status": "complete_before_any_patched_intervention",
            },
        )
        custody_hashes[screen_path.name] = sha256_file(screen_path)

        # Global D barrier: every discovery is durable before support freezing.
        _verify_live_custody(
            executor=executor,
            paths=resolved,
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
        )
        for case_id in CASE_ORDER:
            discovery = executor.acquire_discovery(case_id, case_panels[case_id])
            artifact, custody = _write_split_artifact(
                paths=resolved,
                registration_sha256=registration_sha256,
                run_attempt_sha256=run_attempt_sha256,
                case_id=case_id,
                split="discovery",
                bundle=discovery,
                expected_rows=_screen_rows(screen_cases[str(case_id)], "discovery"),
                authorization_sha256=None,
            )
            artifact_hashes[f"case{case_id}_discovery.pt"] = artifact
            custody_hashes[f"case{case_id}_discovery_custody.json"] = custody
            accounting_panel[(case_id, "discovery")] = {
                "query_accounting": discovery["query_accounting"]
            }

        # Global authorization barrier.  Reopen each sealed D artifact and
        # recompute every support from its exact selector vectors.
        for case_id in CASE_ORDER:
            filename = f"case{case_id}_discovery.pt"
            discovery = _load_artifact(
                path=resolved.run_directory / filename,
                expected_sha256=artifact_hashes[filename],
                registration_sha256=registration_sha256,
                run_attempt_sha256=run_attempt_sha256,
                case_id=case_id,
                split="discovery",
            )
            frozen = freeze_discovery_supports(case_id=case_id, discovery=discovery)
            custody_name = f"case{case_id}_discovery_custody.json"
            authorization = _authorization_record(
                case_id=case_id,
                registration_sha256=registration_sha256,
                run_attempt_sha256=run_attempt_sha256,
                discovery_artifact_sha256=artifact_hashes[filename],
                discovery_custody_sha256=custody_hashes[custody_name],
                frozen_supports=frozen,
            )
            path = (
                resolved.run_directory
                / f"case{case_id}_confirmation_authorization.json"
            )
            exclusive_json_write(path, authorization)
            digest = sha256_file(path)
            custody_hashes[path.name] = digest
            authorizations[case_id] = {
                **authorization,
                "authorization_sha256": digest,
            }

        # No C intervention call is reachable until every authorization exists.
        _verify_live_custody(
            executor=executor,
            paths=resolved,
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
        )
        for case_id in CASE_ORDER:
            authorization_path = (
                resolved.run_directory
                / f"case{case_id}_confirmation_authorization.json"
            )
            expected_authorization = authorizations[case_id]["authorization_sha256"]
            if (
                not authorization_path.is_file()
                or sha256_file(authorization_path) != expected_authorization
            ):
                raise ValueError(
                    "global confirmation authorization barrier changed before C"
                )
        for case_id in CASE_ORDER:
            authorization = authorizations[case_id]
            confirmation = executor.acquire_confirmation(
                case_id, case_panels[case_id], authorization
            )
            digest = authorization["authorization_sha256"]
            artifact, custody = _write_split_artifact(
                paths=resolved,
                registration_sha256=registration_sha256,
                run_attempt_sha256=run_attempt_sha256,
                case_id=case_id,
                split="confirmation",
                bundle=confirmation,
                expected_rows=_screen_rows(screen_cases[str(case_id)], "confirmation"),
                authorization_sha256=digest,
            )
            artifact_hashes[f"case{case_id}_confirmation.pt"] = artifact
            custody_hashes[f"case{case_id}_confirmation_custody.json"] = custody
            accounting_panel[(case_id, "confirmation")] = {
                "query_accounting": confirmation["query_accounting"]
            }

        ordered_artifacts = {
            f"case{case_id}_{split}.pt": artifact_hashes[f"case{case_id}_{split}.pt"]
            for case_id, split in MANDATORY_PANEL_KEYS
        }
        actual_accounting = aggregate_actual_query_accounting(
            {key: accounting_panel[key] for key in MANDATORY_PANEL_KEYS}
        )
        accounting_panel.clear()
        aggregate_path = resolved.run_directory / "acquisition_custody.json"
        exclusive_json_write(
            aggregate_path,
            {
                "schema_version": "e32_3_acquisition_custody_v1",
                "registration_sha256": registration_sha256,
                "run_attempt_sha256": run_attempt_sha256,
                "acquisition_artifact_sha256": ordered_artifacts,
                "stage_custody_sha256": dict(custody_hashes),
                "authorization_sha256": {
                    str(case_id): authorizations[case_id]["authorization_sha256"]
                    for case_id in CASE_ORDER
                },
                "global_behavior_screen_audit": screen_audit,
                "actual_query_accounting": actual_accounting,
                "status": "complete_before_score_attempt",
            },
        )
        custody_hashes[aggregate_path.name] = sha256_file(aggregate_path)

        _verify_live_custody(
            executor=executor,
            paths=resolved,
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
        )
        score_reservation = _reserve_score(
            paths=resolved,
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
            artifact_hashes=ordered_artifacts,
            custody_hashes=custody_hashes,
        )
        score_panel = _load_score_panel(score_reservation)
        scientific = executor.score_complete_panel(score_panel)
        replay = executor.score_complete_panel(score_panel)
        if _canonical_json_bytes(scientific) != _canonical_json_bytes(replay):
            raise RuntimeError("exact score replay differs from the first score")
        _verify_score(score_reservation)
        summary = {
            **scientific,
            "registration_sha256": registration_sha256,
            "run_attempt_sha256": run_attempt_sha256,
            "acquisition_attempt_sha256": sha256_file(acquisition_attempt),
            "score_attempt_sha256": sha256_file(score_reservation.path),
            "acquisition_artifact_sha256": ordered_artifacts,
            "custody_artifact_sha256": custody_hashes,
            "actual_query_accounting": actual_accounting,
            "exact_score_replay": True,
        }
        exclusive_json_write(resolved.run_directory / "summary.json", summary)
        return summary
    except BaseException as error:
        failure_path = resolved.run_directory / "failure.json"
        if not failure_path.exists():
            exclusive_json_write(
                failure_path,
                {
                    "schema_version": "e32_3_failure_v1",
                    "registration_sha256": registration_sha256,
                    "run_attempt_sha256": run_attempt_sha256,
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


def run_registered_experiment(*, confirm: bool = False) -> dict[str, Any]:
    """Run the sole production experiment; no executor can be supplied."""
    return _run_registered_experiment(
        executor=_ProductionExecutor(), paths=CANONICAL_PATHS, confirm=confirm
    )


def _live_split_bundle(
    *,
    case_id: int,
    split: str,
    eligible: Any,
    authorization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Acquire one exact scientific split from fixed lower-level primitives."""
    import math

    from experiments.e32_interpbench.e32_2_acquisition import token_rows
    from experiments.e32_interpbench.e32_3_acquisition import (
        acquire_confirmation_set_interventions,
        acquire_whole_component_atlas,
        build_cohort_component_runner,
        validate_confirmation_supports,
        whole_component_acquisition_to_dict,
    )
    from experiments.e32_interpbench.e32_3_baselines import (
        generic_eap_run_to_dict,
        generic_exact_run_to_dict,
        run_generic_eap_ig5,
        run_generic_exact_fine_replacement,
    )
    from experiments.e32_interpbench.e32_3_orchestration import (
        acquisition_supports_from_frozen,
    )
    from experiments.e32_interpbench.e32_3_scoring import (
        build_selector_score_vectors,
        exact_fine_interaction_mass,
        raw_pair_interaction_mass,
    )

    if case_id not in CASE_ORDER or split not in {"discovery", "confirmation"}:
        raise ValueError("live split identity is outside the registered panel")
    if not hasattr(eligible, "pair"):
        raise TypeError("live split requires a BehaviorallyEligiblePair proof")
    if (split == "discovery") != (authorization is None):
        raise ValueError("only confirmation accepts a frozen authorization")
    manifest = CASE_MANIFESTS[case_id]
    pair = eligible.pair
    batch_size = 64 if split == "discovery" else 128
    if len(pair.clean_inputs) != batch_size or len(pair.corrupt_inputs) != batch_size:
        raise ValueError("eligible pair differs from frozen split size")
    nodes, whole_edges = case_universe(case_id)
    confirmation_supports: tuple[Any, ...] = ()
    frozen: Mapping[str, Any] | None = None
    if split == "confirmation":
        assert authorization is not None
        if (
            authorization.get("case_id") != case_id
            or authorization.get("status") != "durable_before_confirmation_intervention"
        ):
            raise ValueError("confirmation authorization identity changed")
        candidate_frozen = authorization.get("frozen_supports")
        if not isinstance(candidate_frozen, Mapping):
            raise TypeError("confirmation authorization lacks frozen supports")
        confirmation_supports = acquisition_supports_from_frozen(candidate_frozen)
        # This case-aware check recomputes every random SHA support and runs
        # before the first confirmation model is loaded.
        validate_confirmation_supports(
            case_id=case_id,
            component_ids=nodes,
            supports=confirmation_supports,
        )
        frozen = candidate_frozen

    clean_rows = token_rows(pair.clean_inputs, label=f"{split} clean inputs")
    corrupt_rows = token_rows(pair.corrupt_inputs, label=f"{split} corrupt inputs")
    selected_rows = (*clean_rows, *corrupt_rows)
    panel_identity = _panel_identity_sha256(
        case_id=case_id, split=split, selected_token_rows=selected_rows
    )
    eligibility = {
        "required_count": batch_size,
        "candidate_count": int(eligible.candidate_count),
        "eligible_candidate_count": int(eligible.eligible_candidate_count),
        "row_disjoint_eligible_candidate_count": int(
            eligible.row_disjoint_eligible_candidate_count
        ),
        "selected_candidate_indices": tuple(eligible.selected_candidate_indices),
        "selection_policy": eligible.selection_policy,
        "all_selected_pairs_correct": True,
        "clean_incorrect_contexts": int(eligible.clean_incorrect_contexts),
        "source_incorrect_contexts": int(eligible.source_incorrect_contexts),
        "either_incorrect_contexts": int(eligible.either_incorrect_contexts),
    }
    if (
        eligibility["selection_policy"]
        != (
            "first_jointly_correct_with_globally_unique_clean_source_rows_and_"
            "response_blind_exclusions"
        )
        or len(eligibility["selected_candidate_indices"]) != batch_size
        or int(eligible.eligibility_physical_forwards) != 2
    ):
        raise ValueError("BehaviorallyEligiblePair proof differs from protocol")

    def require_counts(
        label: str,
        observed: Mapping[str, int],
        *,
        physical_forwards: int,
        model_examples: int,
    ) -> dict[str, int]:
        expected = {
            "physical_forwards": physical_forwards,
            "model_examples": model_examples,
        }
        if dict(observed) != expected:
            raise RuntimeError(
                f"{label} observed work differs from frozen accounting: "
                f"expected {expected!r}, observed {dict(observed)!r}"
            )
        return expected

    runner = build_cohort_component_runner(
        case_id=case_id,
        model=load_verified_model(case_id, device="cpu"),
        condition_batch_size=32,
    )
    whole = acquire_whole_component_atlas(
        case_id=case_id,
        split=split,
        runner=runner,
        pair=pair,
        discovery_smoke_condition_count=8,
        discovery_smoke_batch_size=8,
    )
    if whole.component_ids != nodes or whole.whole_edges != whole_edges:
        raise RuntimeError("whole-component helper changed the registered universe")
    whole_record = whole_component_acquisition_to_dict(whole)
    preparation = whole.query_accounting["bidirectional_preparation"]
    preparation_accounting = require_counts(
        "bidirectional preparation",
        {
            "physical_forwards": int(preparation["physical_forwards"]),
            "model_examples": int(preparation["model_examples"]),
        },
        physical_forwards=4,
        model_examples=4 * batch_size,
    )
    if split == "discovery":
        smoke = whole.batch_invariance_smoke
        if not isinstance(smoke, Mapping) or smoke.get("passed") is not True:
            raise RuntimeError("whole-component helper lost the discovery smoke")
        smoke_accounting = require_counts(
            "discovery batch-invariance smoke",
            {
                "physical_forwards": int(smoke["physical_forwards"]),
                "model_examples": int(smoke["model_examples"]),
            },
            physical_forwards=9,
            model_examples=16 * batch_size,
        )
    else:
        if whole.batch_invariance_smoke is not None:
            raise RuntimeError("confirmation unexpectedly repeated the D-only smoke")
        smoke = {"passed": True, "status": "D_only_smoke_not_repeated_in_C"}
        smoke_accounting = {"physical_forwards": 0, "model_examples": 0}
    atlas_observed = whole.query_accounting["whole_atlas"]
    atlas_accounting = require_counts(
        "whole-component atlas",
        {
            "physical_forwards": int(atlas_observed["patched_physical_forwards"]),
            "model_examples": int(atlas_observed["patched_model_examples"]),
        },
        physical_forwards=math.ceil((len(nodes) + len(whole_edges)) / 32),
        model_examples=(len(nodes) + len(whole_edges)) * batch_size,
    )
    raw_mass = raw_pair_interaction_mass(whole.pair_interactions)

    def activation_stack(captures: Mapping[str, torch.Tensor]) -> torch.Tensor:
        values = []
        for unit in runner.units:
            activation = captures[unit.hook_name]
            if unit.kind == "attention_head":
                activation = activation[:, :, int(unit.head_index), :]
            values.append(activation.detach().cpu())
        stacked = torch.stack(values)
        if stacked.shape[:3] != (len(nodes), batch_size, manifest.n_ctx):
            raise RuntimeError("whole-component activation capture axes changed")
        return stacked

    clean_activations = activation_stack(whole.clean_activations)
    corrupt_activations = activation_stack(whole.corrupt_activations)

    case = load_verified_official_case(case_id)
    exact = run_generic_exact_fine_replacement(
        case_id=case_id,
        n_layers=manifest.n_layers,
        n_heads=manifest.n_heads,
        n_ctx=manifest.n_ctx,
        d_vocab_out=manifest.d_vocab_out,
        model=load_verified_model(case_id, device="cpu"),
        case=case,
        eligible_pair=eligible,
        device="cpu",
    )
    exact_fine_count = len(exact.fine_edge_names)
    if exact.grouping.edge_order != whole_edges or exact_fine_count != len(
        exact.fine_to_whole_indices
    ):
        raise RuntimeError("exact fine grouping differs from whole-component DAG")
    exact_accounting = require_counts(
        "exact fine replacement",
        {
            "physical_forwards": int(exact.query_accounting.observed_physical_forwards),
            "model_examples": int(exact.query_accounting.observed_model_examples),
        },
        physical_forwards=4 + 2 * exact_fine_count,
        model_examples=(4 + 2 * exact_fine_count) * batch_size,
    )
    exact_accounting.update(
        {
            "no_grad_forwards": int(exact.query_accounting.observed_no_grad_forwards),
            "grad_enabled_forwards": int(
                exact.query_accounting.observed_grad_enabled_forwards
            ),
            "backward_passes": int(exact.query_accounting.backward_passes),
        }
    )
    exact_mass = exact_fine_interaction_mass(
        clean_output=exact.clean_baseline_output,
        corrupt_output=exact.corrupt_baseline_output,
        corrupt_into_clean_outputs=exact.corrupt_into_clean_outputs,
        clean_into_corrupt_outputs=exact.clean_into_corrupt_outputs,
    )

    eap = run_generic_eap_ig5(
        case_id=case_id,
        n_layers=manifest.n_layers,
        n_heads=manifest.n_heads,
        n_ctx=manifest.n_ctx,
        d_vocab_out=manifest.d_vocab_out,
        model=load_verified_model(case_id, device="cpu"),
        case=case,
        eligible_pair=eligible,
        device="cpu",
    )
    if (
        eap.grouping.edge_order != whole_edges
        or eap.internal_fine_edge_names != exact.fine_edge_names
        or eap.fine_to_whole_indices != exact.fine_to_whole_indices
    ):
        raise RuntimeError("same-panel exact and EAP fine identities differ")
    eap_observed = eap.observed_query_accounting
    eap_accounting = require_counts(
        "EAP-IG5",
        {
            "physical_forwards": int(eap_observed["physical_forwards"]),
            "model_examples": int(eap_observed["model_examples"]),
        },
        physical_forwards=7,
        model_examples=7 * batch_size,
    )
    eap_accounting.update(
        {
            "no_grad_forwards": int(eap_observed["no_grad_forwards"]),
            "grad_enabled_forwards": int(eap_observed["grad_enabled_forwards"]),
            "backward_passes_inferred_from_pinned_control_flow": int(
                eap_observed["backward_passes_inferred_from_pinned_control_flow"]
            ),
            "gradient_examples": int(eap.query_accounting.gradient_examples),
        }
    )
    if (
        eap_accounting["backward_passes_inferred_from_pinned_control_flow"] != 6
        or eap_accounting["gradient_examples"] != 6 * batch_size
    ):
        raise RuntimeError("EAP inferred backward/gradient example custody changed")
    eap_scores = torch.tensor(eap.internal_fine_scores, dtype=torch.float64)
    selector_scores = build_selector_score_vectors(
        node_universe=nodes,
        whole_edges=whole_edges,
        clean_output=whole.clean_output,
        singleton_outputs=whole.singleton_outputs,
        clean_activations=clean_activations,
        corrupt_activations=corrupt_activations,
        raw_whole_mass=raw_mass,
        eap_ig5_fine_scores=eap_scores,
        fine_to_whole_indices=exact.fine_to_whole_indices,
    )

    set_accounting = {"physical_forwards": 0, "model_examples": 0}
    deterministic_set_outputs: dict[str, Any] = {}
    random_set_outputs: tuple[dict[str, Any], ...] = ()
    singleton_curve_set_outputs: dict[int, Any] = {}
    all_corrupt_output: torch.Tensor | None = None
    set_record: dict[str, Any] | None = None
    if split == "confirmation":
        assert authorization is not None and frozen is not None
        set_record = acquire_confirmation_set_interventions(
            case_id=case_id,
            runner=runner,
            whole=whole,
            supports=confirmation_supports,
        )
        set_query = set_record["query_accounting"]
        set_accounting = require_counts(
            "confirmation set interventions",
            {
                "physical_forwards": int(set_query["physical_forwards"]),
                "model_examples": int(set_query["model_examples"]),
            },
            physical_forwards=5,
            model_examples=(1 + 2 * len(confirmation_supports)) * batch_size,
        )
        all_corrupt_output = set_record["all_corrupt_into_clean_output"]
        deterministic_set_outputs = dict(set_record["deterministic_set_outputs"])
        random_set_outputs = tuple(set_record["random_set_outputs"])
        singleton_curve_set_outputs = {
            len(record["selected_nodes"]): record
            for record in set_record["singleton_curve_set_outputs"]
        }

    query_accounting = {
        "eligibility": require_counts(
            "behavioral eligibility",
            {
                "physical_forwards": int(eligible.eligibility_physical_forwards),
                "model_examples": int(eligible.eligibility_model_examples),
            },
            physical_forwards=2,
            model_examples=2 * int(eligible.candidate_count),
        ),
        "preparation": preparation_accounting,
        "atlas_smoke": smoke_accounting,
        "atlas": atlas_accounting,
        "exact_fine": exact_accounting,
        "eap_ig5": eap_accounting,
    }
    if split == "confirmation":
        query_accounting["set_interventions"] = set_accounting

    bundle: dict[str, Any] = {
        "schema_version": f"e32_3_{split}_acquisition_v1",
        "case_id": case_id,
        "split": split,
        "component_ids": nodes,
        "whole_edges": whole_edges,
        "selected_token_rows": selected_rows,
        "panel_identity_sha256": panel_identity,
        "family_panel_identity_sha256": {
            "whole_component": panel_identity,
            "exact_fine": panel_identity,
            "eap_ig5": panel_identity,
        },
        "behavioral_eligibility": eligibility,
        "batch_invariance_smoke": smoke,
        "whole_component": whole_record,
        "clean_output": whole.clean_output,
        "corrupt_output": whole.corrupt_output,
        "singleton_outputs": whole.singleton_outputs,
        "pair_interactions": whole.pair_interactions,
        "selector_scores": selector_scores,
        "raw_mass": raw_mass,
        "fine_edge_names": exact.fine_edge_names,
        "eap_fine_edge_names": eap.internal_fine_edge_names,
        "fine_to_whole_indices": exact.fine_to_whole_indices,
        "exact_fine": generic_exact_run_to_dict(exact),
        "eap_ig5": generic_eap_run_to_dict(eap),
        "exact_clean_output": exact.clean_baseline_output,
        "exact_corrupt_output": exact.corrupt_baseline_output,
        "exact_corrupt_into_clean_outputs": exact.corrupt_into_clean_outputs,
        "exact_clean_into_corrupt_outputs": exact.clean_into_corrupt_outputs,
        "exact_mass": exact_mass,
        "eap_ig5_fine_scores": eap_scores,
        "clean_target_indices": pair.clean_targets.argmax(dim=-1)[:, 1:].cpu(),
        "query_accounting": query_accounting,
    }
    if split == "confirmation":
        assert authorization is not None and all_corrupt_output is not None
        bundle.update(
            {
                "confirmation_authorization_sha256": authorization[
                    "authorization_sha256"
                ],
                "all_corrupt_into_clean_output": all_corrupt_output,
                "deterministic_set_outputs": deterministic_set_outputs,
                "random_set_outputs": random_set_outputs,
                "singleton_curve_set_outputs": singleton_curve_set_outputs,
                "confirmation_set_interventions": set_record,
            }
        )
    return bundle


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
                    "status": "dry_run_only_no_intervention_calls_no_writes",
                    "registration_sha256": _json_sha256(record),
                    "protocol": record["protocol"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "seal":
        record = seal_registration()
        print(json.dumps({"registration_sha256": _json_sha256(record)}))
        return 0
    summary = run_registered_experiment(confirm=arguments.confirm)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_REGISTRATION_PATH",
    "CANONICAL_RUN_ATTEMPT_PATH",
    "CANONICAL_RUN_DIRECTORY",
    "SOURCE_RELATIVE_PATHS",
    "RunPaths",
    "build_registration_record",
    "run_registered_experiment",
    "seal_registration",
]
