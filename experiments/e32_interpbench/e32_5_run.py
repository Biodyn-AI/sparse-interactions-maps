# ruff: noqa: EXE002
"""Write-once orchestration for the E32.5 signed group-edge experiment.

Case 58 development is an explicit nonclaim pilot.  It completes before the
final algorithm and source are registered.  The final run binds that pilot,
acquires every prospective D artifact, freezes and rechecks every support,
then crosses one global authorization barrier before any C acquisition.
Before C, the runner reopens only hash-bound prospective D artifacts and the
already-exposed pilot, solely to recompute support authorizations from durable
bytes.  No repair/C artifact or complete score panel is reopened until
``score_attempt.json`` is durable.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from experiments.e32_interpbench.adapter import sha256_file
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
from experiments.e32_interpbench.e32_5_cohort import (
    CASE_ORDER as COHORT_CASE_ORDER,
)
from experiments.e32_interpbench.e32_5_cohort import (
    DEFAULT_E32_5_MODEL_ROOT,
    DESIGN_BEHAVIOR_EXPOSURE,
    SPLIT_ORDER_BY_CASE,
    behaviorally_select_preflight,
    case58_historical_raw_rows,
    load_verified_model,
    load_verified_official_case,
    preflight_cohort_inputs,
    preflight_split_inputs,
    raw_role_rows,
    split_request,
    verify_case_artifacts,
)
from experiments.e32_interpbench.e32_5_cohort import (
    GENERALIZATION_CASE_ORDER as COHORT_GENERALIZATION_CASE_ORDER,
)
from experiments.e32_interpbench.e32_5_protocol import (
    CASE_ORDER,
    DEVELOPMENT_ATTEMPT_PATH,
    DEVELOPMENT_ATTEMPT_STATUS,
    DEVELOPMENT_RUN_DIRECTORY,
    DEVELOPMENT_SUMMARY_STATUS,
    DEVELOPMENT_TERMINAL_FILENAMES,
    EXPERIMENT_ID,
    GENERALIZATION_CASE_ORDER,
    protocol_manifest,
)

if CASE_ORDER != COHORT_CASE_ORDER or (
    GENERALIZATION_CASE_ORDER != COHORT_GENERALIZATION_CASE_ORDER
):
    raise RuntimeError("E32.5 cohort and protocol case orders differ")

# This literal must be replaced after the nonclaim pilot is inspected.  The
# protocol rejects an empty or pending disclosure, so final registration is
# impossible until the exact postpilot changes (or literal "none") are stated.
DEVELOPMENT_TUNING_DISCLOSURE = "PENDING_AFTER_CASE58_DEVELOPMENT_PILOT"


@dataclass(frozen=True)
class DevelopmentPaths:
    attempt_path: Path
    run_directory: Path

    def resolved(self) -> DevelopmentPaths:
        return DevelopmentPaths(
            self.attempt_path.expanduser().resolve(),
            self.run_directory.expanduser().resolve(),
        )


@dataclass(frozen=True)
class RunPaths:
    registration_path: Path
    run_attempt_path: Path
    run_directory: Path
    development: DevelopmentPaths

    def resolved(self) -> RunPaths:
        return RunPaths(
            self.registration_path.expanduser().resolve(),
            self.run_attempt_path.expanduser().resolve(),
            self.run_directory.expanduser().resolve(),
            self.development.resolved(),
        )


CANONICAL_DEVELOPMENT_PATHS = DevelopmentPaths(
    REPOSITORY_ROOT / DEVELOPMENT_ATTEMPT_PATH,
    REPOSITORY_ROOT / DEVELOPMENT_RUN_DIRECTORY,
).resolved()
CANONICAL_PATHS = RunPaths(
    REPOSITORY_ROOT / "notes" / "E32.5-SIGNED-GROUP-EDGE-SELECTOR-REGISTRATION.json",
    REPOSITORY_ROOT / "notes" / "E32.5-SIGNED-GROUP-EDGE-SELECTOR-RUN-ATTEMPT.json",
    REPOSITORY_ROOT / "runs" / EXPERIMENT_ID,
    CANONICAL_DEVELOPMENT_PATHS,
).resolved()
CANONICAL_DEVELOPMENT_ATTEMPT_PATH = CANONICAL_DEVELOPMENT_PATHS.attempt_path
CANONICAL_DEVELOPMENT_RUN_DIRECTORY = CANONICAL_DEVELOPMENT_PATHS.run_directory
CANONICAL_REGISTRATION_PATH = CANONICAL_PATHS.registration_path
CANONICAL_RUN_ATTEMPT_PATH = CANONICAL_PATHS.run_attempt_path
CANONICAL_RUN_DIRECTORY = CANONICAL_PATHS.run_directory

_SOURCE_ROOT_MODULES = (
    "experiments.e32_interpbench.e32_5_run",
    "experiments.e32_interpbench.e32_5_cohort",
    "experiments.e32_interpbench.e32_5_protocol",
    "experiments.e32_interpbench.e32_5_selector",
    "experiments.e32_interpbench.e32_5_scoring",
    "experiments.e32_interpbench.e32_5_acquisition",
    "experiments.e32_interpbench.e32_5_pipeline",
)
_ACCOUNTING_KEYS = (
    "physical_forwards",
    "model_examples",
    "no_grad_forwards",
    "grad_enabled_forwards",
    "gradient_examples",
    "backward_passes_inferred_from_pinned_control_flow",
    "mask_context_count",
    "active_mask_coordinate_checks",
)
_PILOT_QUERY_CEILING = {
    "physical_forwards": 639,
    "model_examples": 40_672,
    "grad_enabled_forwards": 7,
    "gradient_examples": 224,
    "backward_passes_inferred_from_pinned_control_flow": 7,
}
_FINAL_QUERY_CEILING = {
    "physical_forwards": 4_159,
    "model_examples": 401_184,
}
_CUMULATIVE_QUERY_CEILING = {
    "physical_forwards": 4_798,
    "model_examples": 441_856,
}
_FINAL_ARTIFACT_KEYS = (
    *((case_id, "discovery") for case_id in GENERALIZATION_CASE_ORDER),
    (58, "repair_confirmation"),
    *((case_id, "confirmation") for case_id in GENERALIZATION_CASE_ORDER),
)
_FINAL_CUSTODY_FILENAMES = (
    "acquisition_attempt.json",
    "input_preflight.json",
    "behavior_screen.json",
    *(f"case{case_id}_discovery_custody.json" for case_id in GENERALIZATION_CASE_ORDER),
    "case58_repair_authorization.json",
    *(
        f"case{case_id}_confirmation_authorization.json"
        for case_id in GENERALIZATION_CASE_ORDER
    ),
    "global_support_authorization.json",
    "case58_repair_confirmation_custody.json",
    *(
        f"case{case_id}_confirmation_custody.json"
        for case_id in GENERALIZATION_CASE_ORDER
    ),
    "acquisition_custody.json",
)


def _canonical_json_bytes(value: Any) -> bytes:
    normalized = json.loads(json.dumps(value, allow_nan=False))
    return json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
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
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None or spec.origin is None or spec.origin in {"built-in", "frozen"}:
        return None
    path = Path(spec.origin).resolve()
    return path if path.suffix == ".py" and path.is_file() else None


def _source_relative_paths() -> tuple[str, ...]:
    """Return the complete repository-local Python import closure."""
    pending = list(_SOURCE_ROOT_MODULES)
    visited_modules: set[str] = set()
    paths: set[str] = set()
    while pending:
        module = pending.pop()
        if module in visited_modules:
            continue
        visited_modules.add(module)
        path = _module_source_path(module)
        if path is None:
            # The pipeline is allowed to land after this runner during tests,
            # but production registration requires it through the executor.
            if module != "experiments.e32_interpbench.e32_5_pipeline":
                raise FileNotFoundError(f"missing E32.5 source module {module}")
            continue
        if REPOSITORY_ROOT not in path.parents:
            continue
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        paths.add(relative)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = (node.module,)
            for name in names:
                if name == "experiments" or name.startswith("experiments."):
                    pending.append(name)
    return tuple(sorted(paths))


def _source_hashes() -> dict[str, str]:
    return {
        relative: sha256_file(REPOSITORY_ROOT / relative)
        for relative in _source_relative_paths()
    }


def _assert_external_sentinel(path: Path, run_directory: Path, *, label: str) -> None:
    resolved_path = path.resolve()
    resolved_directory = run_directory.resolve()
    if (
        resolved_path == resolved_directory
        or resolved_directory in resolved_path.parents
        or resolved_path in resolved_directory.parents
    ):
        raise ValueError(f"{label} must be external to its run directory")


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _accounting_record(value: Any, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(_ACCOUNTING_KEYS):
        raise ValueError(f"{label} must exactly follow {_ACCOUNTING_KEYS!r}")
    record = {
        key: _nonnegative_integer(value[key], label=f"{label}.{key}")
        for key in _ACCOUNTING_KEYS
    }
    if (
        record["no_grad_forwards"] + record["grad_enabled_forwards"]
        != record["physical_forwards"]
    ):
        raise ValueError(f"{label} forward-mode accounting does not sum")
    if (
        record["backward_passes_inferred_from_pinned_control_flow"]
        != record["grad_enabled_forwards"]
        or record["gradient_examples"] > record["model_examples"]
        or record["model_examples"] < record["physical_forwards"]
        or (record["physical_forwards"] == 0 and record["model_examples"] != 0)
    ):
        raise ValueError(f"{label} gradient/example accounting is inconsistent")
    return record


def _sum_records(records: Sequence[Mapping[str, int]]) -> dict[str, int]:
    return {
        key: sum(int(record[key]) for record in records) for key in _ACCOUNTING_KEYS
    }


def _validate_accounting(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or tuple(value) != ("families", "total"):
        raise ValueError(f"{label} must contain ordered families and total")
    raw_families = value["families"]
    if not isinstance(raw_families, Mapping) or not raw_families:
        raise ValueError(f"{label}.families must be a nonempty ordered mapping")
    families: dict[str, dict[str, int]] = {}
    for family, raw in raw_families.items():
        if not isinstance(family, str) or not family or family in families:
            raise ValueError(f"{label} contains an invalid family name")
        families[family] = _accounting_record(raw, label=f"{label}.{family}")
    total = _accounting_record(value["total"], label=f"{label}.total")
    if total != _sum_records(tuple(families.values())):
        raise ValueError(f"{label}.total differs from exact family sum")
    return {"families": families, "total": total}


def _aggregate_accounting(
    named: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    families: dict[str, dict[str, int]] = {}
    for stage_name, raw in named:
        stage = _validate_accounting(raw, label=f"query_accounting[{stage_name}]")
        for family, record in stage["families"].items():
            key = f"{stage_name}:{family}"
            if key in families:
                raise ValueError("query-accounting family collision")
            families[key] = record
    return {"families": families, "total": _sum_records(tuple(families.values()))}


def _within_ceiling(
    accounting: Mapping[str, Any], ceiling: Mapping[str, int], *, label: str
) -> None:
    total = accounting["total"]
    exceeded = {
        key: (int(total[key]), maximum)
        for key, maximum in ceiling.items()
        if int(total[key]) > maximum
    }
    if exceeded:
        raise RuntimeError(f"{label} query ceiling exceeded: {exceeded}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _scientific_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and bool(torch.equal(left, right))
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and tuple(left) == tuple(right)
            and all(_scientific_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            isinstance(left, (tuple, list))
            and isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(_scientific_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    if isinstance(left, float) and isinstance(right, float):
        return (math.isnan(left) and math.isnan(right)) or left == right
    return left == right


def _torch_load(path: Path) -> Any:
    return torch.load(path, map_location="cpu")


def _expected_schema(case_id: int, split: str) -> tuple[int, str]:
    allowed = {
        58: ("development", "repair_confirmation"),
        **{case: ("discovery", "confirmation") for case in GENERALIZATION_CASE_ORDER},
    }
    if case_id not in allowed or split not in allowed[case_id]:
        raise ValueError("artifact identity is outside the E32.5 lifecycle")
    return case_id, split


def _validate_bundle(bundle: Any, *, case_id: int, split: str) -> dict[str, Any]:
    _expected_schema(case_id, split)
    if not isinstance(bundle, Mapping):
        raise TypeError("raw acquisition bundle must be a mapping")
    required = (
        "schema_version",
        "case_id",
        "split",
        "selected_token_rows",
        "query_accounting",
        "scientific_payload",
    )
    if any(key not in bundle for key in required):
        raise ValueError("raw acquisition bundle is incomplete")
    if (
        bundle["schema_version"] != "e32_5_raw_acquisition_v1"
        or bundle["case_id"] != case_id
        or bundle["split"] != split
    ):
        raise ValueError("raw acquisition bundle identity changed")
    if any(key in bundle for key in ("score", "summary", "verdict", "case_score")):
        raise ValueError("raw acquisition artifact contains trusted derived scoring")
    rows = tuple(tuple(row) for row in bundle["selected_token_rows"])
    expected_row_count = 128 if split in {"development", "discovery"} else 256
    if (
        len(rows) != expected_row_count
        or len(rows) != len(set(rows))
        or len({len(row) for row in rows}) != 1
        or any(
            not row
            or any(
                isinstance(token, bool) or not isinstance(token, int) for token in row
            )
            for row in rows
        )
    ):
        raise ValueError("raw acquisition selected rows changed")
    accounting = _validate_accounting(
        bundle["query_accounting"], label=f"case{case_id}_{split}"
    )
    if not isinstance(bundle["scientific_payload"], Mapping):
        raise TypeError("raw scientific payload must be a mapping")
    return {
        **dict(bundle),
        "selected_token_rows": rows,
        "query_accounting": accounting,
    }


def _normalize_live_bundle(value: Any, *, split: str) -> dict[str, Any]:
    """Normalize the pipeline's typed raw serializer to the runner envelope."""
    if not isinstance(value, Mapping):
        raise TypeError("live pipeline serializer must return a mapping")
    raw = dict(value)
    if raw.get("schema_version") == "e32_5_raw_acquisition_v1":
        return raw
    accounting = raw.get("query_accounting")
    if not isinstance(accounting, Mapping) or "total" not in accounting:
        raise ValueError("live pipeline accounting is incomplete")
    families = {key: item for key, item in accounting.items() if key != "total"}
    if not families:
        raise ValueError("live pipeline accounting has no stage families")
    return {
        "schema_version": "e32_5_raw_acquisition_v1",
        "case_id": raw.get("case_id"),
        "split": split,
        "selected_token_rows": raw.get("selected_token_rows"),
        "query_accounting": {
            "families": families,
            "total": accounting["total"],
        },
        "scientific_payload": raw.get("scientific_payload"),
    }


def _artifact_filename(case_id: int, split: str) -> str:
    return f"case{case_id}_{split}.pt"


def _custody_filename(case_id: int, split: str) -> str:
    return f"case{case_id}_{split}_custody.json"


def _write_bundle(
    *,
    run_directory: Path,
    bundle: Mapping[str, Any],
    case_id: int,
    split: str,
    attempt_sha256: str,
    registration_sha256: str | None,
    authorization_sha256: str | None,
) -> tuple[str, str]:
    validated = _validate_bundle(bundle, case_id=case_id, split=split)
    envelope = {
        **validated,
        "attempt_sha256": attempt_sha256,
        "registration_sha256": registration_sha256,
        "authorization_sha256": authorization_sha256,
    }
    artifact_path = run_directory / _artifact_filename(case_id, split)
    _exclusive_torch_write(artifact_path, envelope)
    artifact_sha256 = sha256_file(artifact_path)
    custody_path = run_directory / _custody_filename(case_id, split)
    custody = {
        "schema_version": f"{EXPERIMENT_ID}_{split}_custody_v1",
        "case_id": case_id,
        "split": split,
        "attempt_sha256": attempt_sha256,
        "registration_sha256": registration_sha256,
        "authorization_sha256": authorization_sha256,
        "artifact_filename": artifact_path.name,
        "artifact_sha256": artifact_sha256,
        "selected_token_rows_sha256": _json_sha256(validated["selected_token_rows"]),
        "query_accounting": validated["query_accounting"],
        "status": "raw_outcomes_durable_without_trusted_derived_score",
    }
    exclusive_json_write(custody_path, custody)
    return artifact_sha256, sha256_file(custody_path)


def _load_bundle(
    *,
    path: Path,
    expected_sha256: str,
    case_id: int,
    split: str,
    attempt_sha256: str,
    registration_sha256: str | None,
    authorization_sha256: str | None,
) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("raw acquisition artifact hash changed")
    value = _torch_load(path)
    validated = _validate_bundle(value, case_id=case_id, split=split)
    if (
        value.get("attempt_sha256") != attempt_sha256
        or value.get("registration_sha256") != registration_sha256
        or value.get("authorization_sha256") != authorization_sha256
    ):
        raise ValueError("raw acquisition artifact custody identity changed")
    return validated


def _eligibility_record(value: Any) -> dict[str, Any]:
    return {
        "candidate_count": int(value.candidate_count),
        "eligible_candidate_count": int(value.eligible_candidate_count),
        "row_disjoint_eligible_candidate_count": int(
            value.row_disjoint_eligible_candidate_count
        ),
        "selected_candidate_indices": tuple(value.selected_candidate_indices),
        "eligibility_physical_forwards": int(value.eligibility_physical_forwards),
        "eligibility_model_examples": int(value.eligibility_model_examples),
        "selection_policy": str(value.selection_policy),
    }


def _selected_rows(value: Any) -> tuple[tuple[int, ...], ...]:
    return raw_role_rows(value.pair)


def _preflight_record(value: Any) -> dict[str, Any]:
    request = value.request
    return {
        "case_id": request.case_id,
        "split": request.split,
        "request": asdict(request),
        "clean_unique_rows": value.clean_unique_rows,
        "corrupt_unique_rows": value.corrupt_unique_rows,
        "role_union_unique_rows": value.role_union_unique_rows,
        "excluded_unique_rows": value.excluded_unique_rows,
        "overlap_with_exclusions": value.overlap_with_exclusions,
        "response_blind_selectable_indices": value.response_blind_selectable_indices,
        "observed_raw_identity_sha256": value.observed_raw_identity_sha256,
        "observed_target_identity_sha256": value.observed_target_identity_sha256,
    }


class _ProductionExecutor:
    """Narrow production boundary; public runner entrypoints never accept it."""

    def registration_dependencies(self) -> dict[str, Any]:
        _require_production_environment()
        artifacts = {
            str(case_id): verify_case_artifacts(DEFAULT_E32_5_MODEL_ROOT, case_id)
            for case_id in CASE_ORDER
        }
        runtime = _e32_2_dependency_record()
        source_directories = {
            **runtime["source_directories"],
            "model_root": str(DEFAULT_E32_5_MODEL_ROOT.resolve()),
        }
        runtime = {
            **runtime,
            "source_directories": source_directories,
        }
        runtime.pop("case_artifact_sha256", None)
        pipeline_path = _module_source_path(
            "experiments.e32_interpbench.e32_5_pipeline"
        )
        if pipeline_path is None:
            raise FileNotFoundError("E32.5 live pipeline source is missing")
        return {
            "environment": _e32_2_environment_record(),
            "runtime_source_trees": runtime,
            "model_root": str(DEFAULT_E32_5_MODEL_ROOT.resolve()),
            "model_artifact_sha256": artifacts,
            "design_behavior_exposure": DESIGN_BEHAVIOR_EXPOSURE,
        }

    def pilot_input_preflight(self):
        case = load_verified_official_case(58)
        preflight = preflight_split_inputs(
            case,
            split_request(58, "development"),
            excluded_token_rows=case58_historical_raw_rows(case),
        )
        return (
            {
                "schema_version": "e32_5_case58_development_input_preflight_v1",
                "case_id": 58,
                "split": "development",
                "verified": True,
                "model_calls": 0,
                "preflight": _preflight_record(preflight),
            },
            (case, preflight),
        )

    def pilot_behavior_screen(self, state):
        case, preflight = state
        eligible = behaviorally_select_preflight(
            load_verified_model(58, device="cpu"), preflight, device="cpu"
        )
        return (
            {
                "schema_version": "e32_5_case58_development_behavior_screen_v1",
                "case_id": 58,
                "split": "development",
                "behavior_only": True,
                "eligibility": _eligibility_record(eligible),
                "selected_token_rows": _selected_rows(eligible),
            },
            {"case": case, "eligible": eligible},
        )

    def acquire_development(self, panel):
        from experiments.e32_interpbench.e32_5_pipeline import (
            run_e32_5_development_pilot,
            serialize_e32_5_discovery_bundle,
        )

        acquired = run_e32_5_development_pilot(
            case_id=58, case=panel["case"], eligible_pair=panel["eligible"]
        )
        serialized = serialize_e32_5_discovery_bundle(acquired)
        return _normalize_live_bundle(serialized, split="development")

    def observe_development(self, bundle):
        from experiments.e32_interpbench.e32_5_pipeline import (
            observe_e32_5_development,
        )

        return observe_e32_5_development(bundle)

    def final_input_preflight(self):
        cases = {
            case_id: load_verified_official_case(case_id) for case_id in CASE_ORDER
        }
        panels = preflight_cohort_inputs(cases)
        return (
            {
                "schema_version": "e32_5_final_input_preflight_v1",
                "case_order": list(CASE_ORDER),
                "verified": True,
                "model_calls": 0,
                "cases": {
                    str(case_id): {
                        split: _preflight_record(preflight)
                        for split, preflight in panels[case_id].items()
                    }
                    for case_id in CASE_ORDER
                },
            },
            {"cases": cases, "preflights": panels},
        )

    def final_behavior_screen(self, state):
        cases = state["cases"]
        preflights = state["preflights"]
        panels: dict[int, dict[str, Any]] = {}
        records: dict[str, Any] = {}
        for case_id in CASE_ORDER:
            splits = (
                ("repair_confirmation",)
                if case_id == 58
                else ("discovery", "confirmation")
            )
            model = load_verified_model(case_id, device="cpu")
            panels[case_id] = {"case": cases[case_id]}
            records[str(case_id)] = {"case_id": case_id, "splits": {}}
            for split in splits:
                eligible = behaviorally_select_preflight(
                    model, preflights[case_id][split], device="cpu"
                )
                panels[case_id][split] = eligible
                records[str(case_id)]["splits"][split] = {
                    "eligibility": _eligibility_record(eligible),
                    "selected_token_rows": _selected_rows(eligible),
                }
        return (
            {
                "schema_version": "e32_5_final_behavior_screen_v1",
                "case_order": list(CASE_ORDER),
                "behavior_only": True,
                "cases": records,
            },
            panels,
        )

    def acquire_discovery(self, case_id: int, panel: Mapping[str, Any]):
        from experiments.e32_interpbench.e32_5_pipeline import (
            run_e32_5_discovery,
            serialize_e32_5_discovery_bundle,
        )

        acquired = run_e32_5_discovery(
            case_id=case_id,
            case=panel["case"],
            eligible_pair=panel["discovery"],
        )
        return _normalize_live_bundle(
            serialize_e32_5_discovery_bundle(acquired), split="discovery"
        )

    def freeze_support(self, case_id: int, development: Mapping[str, Any]):
        from experiments.e32_interpbench.e32_5_pipeline import (
            freeze_e32_5_support,
        )

        return freeze_e32_5_support(case_id=case_id, raw_bundle=development)

    def acquire_repair_confirmation(self, panel, authorization):
        from experiments.e32_interpbench.e32_5_pipeline import (
            run_e32_5_confirmation,
            serialize_e32_5_confirmation_bundle,
        )

        acquired = run_e32_5_confirmation(
            case_id=58,
            case=panel["case"],
            eligible_pair=panel["repair_confirmation"],
            authorization=authorization,
        )
        return serialize_e32_5_confirmation_bundle(
            acquired, split="repair_confirmation"
        )

    def acquire_confirmation(self, case_id, panel, authorization):
        from experiments.e32_interpbench.e32_5_pipeline import (
            run_e32_5_confirmation,
            serialize_e32_5_confirmation_bundle,
        )

        acquired = run_e32_5_confirmation(
            case_id=case_id,
            case=panel["case"],
            eligible_pair=panel["confirmation"],
            authorization=authorization,
        )
        return serialize_e32_5_confirmation_bundle(acquired, split="confirmation")

    def score_raw_panel(self, panel):
        from experiments.e32_interpbench.e32_5_pipeline import (
            score_e32_5_raw_panel,
        )

        return score_e32_5_raw_panel(panel)


def _expected_split_preflight(case_id: int, split: str) -> dict[str, Any]:
    request = split_request(case_id, split)
    return {
        "case_id": case_id,
        "split": split,
        "request": asdict(request),
        "clean_unique_rows": request.candidate_count,
        "corrupt_unique_rows": request.candidate_count,
        "role_union_unique_rows": 2 * request.candidate_count,
        "excluded_unique_rows": request.expected_excluded_unique_rows,
        "overlap_with_exclusions": 0,
        "response_blind_selectable_indices": tuple(
            range(request.response_blind_capacity)
        ),
        "observed_raw_identity_sha256": request.raw_identity_sha256,
        "observed_target_identity_sha256": request.target_identity_sha256,
    }


def _validate_split_preflight(
    record: Any, *, case_id: int, split: str, label: str
) -> None:
    expected = _expected_split_preflight(case_id, split)
    if not isinstance(record, Mapping) or _canonical_json_bytes(
        record
    ) != _canonical_json_bytes(expected):
        raise ValueError(f"{label} split request, identity, or capacity changed")


def _validate_pilot_input(record: Any) -> None:
    if (
        not isinstance(record, Mapping)
        or record.get("schema_version") != "e32_5_case58_development_input_preflight_v1"
        or record.get("case_id") != 58
        or record.get("split") != "development"
        or record.get("verified") is not True
        or record.get("model_calls") != 0
    ):
        raise ValueError("case58 development input preflight changed")
    _validate_split_preflight(
        record.get("preflight"),
        case_id=58,
        split="development",
        label="case58 development preflight",
    )


def _validate_pilot_behavior(record: Any) -> None:
    if (
        not isinstance(record, Mapping)
        or record.get("schema_version") != "e32_5_case58_development_behavior_screen_v1"
        or record.get("case_id") != 58
        or record.get("split") != "development"
        or record.get("behavior_only") is not True
        or "query_accounting" in record
    ):
        raise ValueError("case58 development behavior screen changed")
    _validated_selected_rows(
        record.get("selected_token_rows"), split="development", label="pilot behavior"
    )


def _validated_selected_rows(
    value: Any, *, split: str, label: str
) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} selected rows must be a sequence")
    rows = tuple(tuple(row) for row in value)
    expected = 128 if split in {"development", "discovery"} else 256
    if (
        len(rows) != expected
        or len(rows) != len(set(rows))
        or len({len(row) for row in rows}) != 1
        or any(
            not row
            or any(
                isinstance(token, bool) or not isinstance(token, int) for token in row
            )
            for row in rows
        )
    ):
        raise ValueError(f"{label} selected rows changed")
    return rows


def _final_behavior_rows(
    record: Mapping[str, Any], *, case_id: int, split: str
) -> tuple[tuple[int, ...], ...]:
    cases = record.get("cases")
    if not isinstance(cases, Mapping):
        raise TypeError("final behavior cases must be a mapping")
    case = cases.get(str(case_id))
    if (
        not isinstance(case, Mapping)
        or case.get("case_id") != case_id
        or not isinstance(case.get("splits"), Mapping)
    ):
        raise ValueError("final behavior case identity changed")
    split_record = case["splits"].get(split)
    if not isinstance(split_record, Mapping):
        raise TypeError("final behavior split must be a mapping")
    return _validated_selected_rows(
        split_record.get("selected_token_rows"),
        split=split,
        label=f"case{case_id}_{split} behavior",
    )


def _require_selected_rows(
    bundle: Mapping[str, Any], expected: Sequence[Sequence[int]], *, label: str
) -> None:
    if tuple(bundle["selected_token_rows"]) != tuple(tuple(row) for row in expected):
        raise ValueError(f"{label} acquisition rows differ from behavior custody")


def _reserve_development_attempt(
    *, paths: DevelopmentPaths, dependencies: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    resolved = paths.resolved()
    _assert_external_sentinel(
        resolved.attempt_path, resolved.run_directory, label="development attempt"
    )
    payload = {
        "schema_version": f"{EXPERIMENT_ID}_development_attempt_v1",
        "experiment_id": EXPERIMENT_ID,
        "canonical_development_run_directory": str(resolved.run_directory),
        "source_sha256": _source_hashes(),
        "dependencies": dependencies,
        "status": DEVELOPMENT_ATTEMPT_STATUS,
    }
    exclusive_json_write(resolved.attempt_path, payload)
    return sha256_file(resolved.attempt_path), payload


def _verify_development_live_custody(
    *, executor: Any, paths: DevelopmentPaths, attempt_sha256: str, attempt: Any
) -> None:
    if sha256_file(paths.attempt_path) != attempt_sha256 or _canonical_json_bytes(
        json.loads(paths.attempt_path.read_text())
    ) != _canonical_json_bytes(attempt):
        raise ValueError("development attempt changed during pilot")
    if _source_hashes() != attempt["source_sha256"]:
        raise ValueError("source closure changed during development pilot")
    if executor.registration_dependencies() != attempt["dependencies"]:
        raise ValueError("runtime or artifact closure changed during development pilot")


def _develop_case58(
    *, executor: Any, paths: DevelopmentPaths, confirm: bool = False
) -> dict[str, Any]:
    if not confirm:
        raise ValueError("case58 nonclaim development requires confirm=True")
    resolved = paths.resolved()
    if resolved.attempt_path.exists():
        raise FileExistsError("E32.5 development attempt is already consumed")
    if resolved.run_directory.exists():
        raise FileExistsError("E32.5 development run directory is already consumed")

    dependencies = executor.registration_dependencies()
    attempt_sha256, attempt = _reserve_development_attempt(
        paths=resolved, dependencies=dependencies
    )
    resolved.run_directory.mkdir(parents=True, exist_ok=False)
    acquisition_attempt_path = (
        resolved.run_directory / "development_acquisition_attempt.json"
    )
    exclusive_json_write(
        acquisition_attempt_path,
        {
            "schema_version": f"{EXPERIMENT_ID}_development_acquisition_attempt_v1",
            "development_attempt_sha256": attempt_sha256,
            "phase_order": (
                "case58_input_preflight",
                "case58_behavior_screen",
                "case58_development_D_only",
                "development_observation",
                "terminal_nonclaim_summary",
            ),
            "status": "durable_before_case58_raw_input_generation_or_model_call",
        },
    )
    artifact_hashes: dict[str, str] = {}
    custody_hashes: dict[str, str] = {}
    try:
        preflight_record, private_state = executor.pilot_input_preflight()
        _validate_pilot_input(preflight_record)
        preflight_path = resolved.run_directory / "case58_input_preflight.json"
        exclusive_json_write(
            preflight_path,
            {
                **dict(preflight_record),
                "development_attempt_sha256": attempt_sha256,
                "status": "durable_before_case58_development_behavior_model",
            },
        )
        custody_hashes[preflight_path.name] = sha256_file(preflight_path)
        _verify_development_live_custody(
            executor=executor,
            paths=resolved,
            attempt_sha256=attempt_sha256,
            attempt=attempt,
        )

        behavior_record, panel = executor.pilot_behavior_screen(private_state)
        _validate_pilot_behavior(behavior_record)
        behavior_path = resolved.run_directory / "case58_behavior_screen.json"
        exclusive_json_write(
            behavior_path,
            {
                **dict(behavior_record),
                "development_attempt_sha256": attempt_sha256,
                "status": "durable_before_case58_development_intervention",
            },
        )
        custody_hashes[behavior_path.name] = sha256_file(behavior_path)
        _verify_development_live_custody(
            executor=executor,
            paths=resolved,
            attempt_sha256=attempt_sha256,
            attempt=attempt,
        )

        raw = executor.acquire_development(panel)
        bundle = _validate_bundle(raw, case_id=58, split="development")
        _require_selected_rows(
            bundle,
            behavior_record["selected_token_rows"],
            label="case58 development",
        )
        artifact_sha256, custody_sha256 = _write_bundle(
            run_directory=resolved.run_directory,
            bundle=bundle,
            case_id=58,
            split="development",
            attempt_sha256=attempt_sha256,
            registration_sha256=None,
            authorization_sha256=None,
        )
        artifact_hashes["case58_development.pt"] = artifact_sha256
        custody_hashes["case58_development_custody.json"] = custody_sha256

        observation = executor.observe_development(bundle)
        if not isinstance(observation, Mapping) or (
            observation.get("case_id") != 58
            or observation.get("evidence_role") != "nonclaim_method_development"
        ):
            raise ValueError("development observation identity changed")
        observation_path = resolved.run_directory / "development_observation.json"
        exclusive_json_write(
            observation_path,
            {
                **_json_safe(observation),
                "development_attempt_sha256": attempt_sha256,
                "case58_development_artifact_sha256": artifact_sha256,
                "claim_status": "none_nonclaim_method_development_only",
            },
        )
        custody_hashes[observation_path.name] = sha256_file(observation_path)
        pilot_accounting = bundle["query_accounting"]
        _within_ceiling(
            pilot_accounting, _PILOT_QUERY_CEILING, label="case58 development pilot"
        )
        _verify_development_live_custody(
            executor=executor,
            paths=resolved,
            attempt_sha256=attempt_sha256,
            attempt=attempt,
        )
        summary = {
            "schema_version": f"{EXPERIMENT_ID}_development_summary_v1",
            "experiment_id": EXPERIMENT_ID,
            "status": DEVELOPMENT_SUMMARY_STATUS,
            "claim_status": "none_nonclaim_method_development_only",
            "development_attempt_sha256": attempt_sha256,
            "development_acquisition_attempt_sha256": sha256_file(
                acquisition_attempt_path
            ),
            "pilot_source_sha256": attempt["source_sha256"],
            "pilot_dependencies_sha256": _json_sha256(attempt["dependencies"]),
            "artifact_sha256": artifact_hashes,
            "custody_sha256": custody_hashes,
            "query_accounting": pilot_accounting,
            "query_ceiling": _PILOT_QUERY_CEILING,
            "query_ceiling_pass": True,
            "repair_or_prospective_model_calls": 0,
            "algorithm_frozen": False,
            "final_registration_required_after_postpilot_tuning_disclosure": True,
        }
        exclusive_json_write(
            resolved.run_directory / "development_summary.json", summary
        )
        return summary
    except BaseException as error:
        failure_path = resolved.run_directory / "development_failure.json"
        if not failure_path.exists():
            exclusive_json_write(
                failure_path,
                {
                    "schema_version": f"{EXPERIMENT_ID}_development_failure_v1",
                    "development_attempt_sha256": attempt_sha256,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "completed_artifact_sha256": artifact_hashes,
                    "completed_custody_sha256": custody_hashes,
                },
            )
        raise


def develop_case58(*, confirm: bool = False) -> dict[str, Any]:
    """Run the one nonclaim development pilot; no repair or claim case is run."""
    return _develop_case58(
        executor=_ProductionExecutor(),
        paths=CANONICAL_DEVELOPMENT_PATHS,
        confirm=confirm,
    )


def _load_development_ancestry(paths: DevelopmentPaths) -> dict[str, Any]:
    resolved = paths.resolved()
    summary_path = resolved.run_directory / "development_summary.json"
    try:
        attempt = json.loads(resolved.attempt_path.read_text())
        summary = json.loads(summary_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("E32.5 development summary is missing or malformed") from error
    observed_entries = {path.name for path in resolved.run_directory.iterdir()}
    if observed_entries != set(DEVELOPMENT_TERMINAL_FILENAMES) or any(
        not (resolved.run_directory / filename).is_file()
        for filename in DEVELOPMENT_TERMINAL_FILENAMES
    ):
        raise ValueError("development terminal file set differs from protocol")
    if (
        attempt.get("status") != DEVELOPMENT_ATTEMPT_STATUS
        or attempt.get("canonical_development_run_directory")
        != str(resolved.run_directory)
        or summary.get("status") != DEVELOPMENT_SUMMARY_STATUS
        or summary.get("development_attempt_sha256")
        != sha256_file(resolved.attempt_path)
        or summary.get("pilot_source_sha256") != attempt.get("source_sha256")
        or summary.get("repair_or_prospective_model_calls") != 0
        or summary.get("algorithm_frozen") is not False
    ):
        raise ValueError("development ancestry identity changed")
    expected_hashes = {
        "development_acquisition_attempt.json": summary.get(
            "development_acquisition_attempt_sha256"
        ),
        **dict(summary.get("artifact_sha256", {})),
        **dict(summary.get("custody_sha256", {})),
    }
    for filename in DEVELOPMENT_TERMINAL_FILENAMES[:-1]:
        expected = _validated_sha256(
            expected_hashes.get(filename), label=f"pilot hash for {filename}"
        )
        if sha256_file(resolved.run_directory / filename) != expected:
            raise ValueError(f"development terminal file changed: {filename}")
    accounting = _validate_accounting(
        summary.get("query_accounting"), label="pilot summary accounting"
    )
    _within_ceiling(accounting, _PILOT_QUERY_CEILING, label="pilot ancestry")
    return {
        "status": DEVELOPMENT_SUMMARY_STATUS,
        "attempt_path": str(resolved.attempt_path),
        "run_directory": str(resolved.run_directory),
        "development_attempt_sha256": sha256_file(resolved.attempt_path),
        "development_summary_sha256": sha256_file(summary_path),
        "terminal_file_sha256": {
            filename: sha256_file(resolved.run_directory / filename)
            for filename in DEVELOPMENT_TERMINAL_FILENAMES
        },
        "pilot_source_sha256": summary["pilot_source_sha256"],
        "pilot_query_accounting": accounting,
        "claim_status": summary["claim_status"],
    }


def _build_registration_record(*, executor: Any, paths: RunPaths) -> dict[str, Any]:
    resolved = paths.resolved()
    ancestry = _load_development_ancestry(resolved.development)
    protocol = protocol_manifest(
        development_tuning_disclosure=DEVELOPMENT_TUNING_DISCLOSURE
    )
    return {
        "schema_version": f"{EXPERIMENT_ID}_registration_v1",
        "experiment_id": EXPERIMENT_ID,
        "canonical_registration_path": str(resolved.registration_path),
        "canonical_run_attempt_path": str(resolved.run_attempt_path),
        "canonical_run_directory": str(resolved.run_directory),
        "development_ancestry": ancestry,
        "protocol": protocol,
        "source_sha256": _source_hashes(),
        "dependencies": executor.registration_dependencies(),
        "execution": {
            "device": "cpu",
            "production_executor_injectable": False,
            "final_run_reruns_case58_development": False,
            "pilot_artifact_reopened_only_as_disclosed_development_ancestry": True,
            "all_prospective_discoveries_durable_before_any_support_authorization": True,
            "all_four_supports_recomputed_before_global_authorization": True,
            "no_confirmation_outcome_exists_before_global_authorization": True,
            "hash_bound_D_reload_is_required_for_independent_authorization": True,
            "score_attempt_precedes_every_C_artifact_reload_and_scoring_reload": True,
            "score_attempt_binds_and_reloads_raw_case58_pilot_ancestry": True,
            "raw_score_recomputation_and_exact_replay_required": True,
            "behavior_screen_accounting_lives_only_in_split_bundles": True,
        },
    }


def build_registration_record() -> dict[str, Any]:
    """Build the final registration; fails while tuning disclosure is pending."""
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
            raise FileExistsError(f"canonical E32.5 {label} is already consumed")
    record = _build_registration_record(executor=executor, paths=resolved)
    if resolved.run_attempt_path.exists() or resolved.run_directory.exists():
        raise FileExistsError("canonical E32.5 final run state changed during seal")
    exclusive_json_write(resolved.registration_path, record)
    return record


def seal_registration() -> dict[str, Any]:
    """Seal the sole final registration exactly once; executor is noninjectable."""
    return _seal_registration(executor=_ProductionExecutor(), paths=CANONICAL_PATHS)


def _load_registration(*, executor: Any, paths: RunPaths) -> tuple[dict[str, Any], str]:
    try:
        observed = json.loads(paths.registration_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("E32.5 registration is missing or malformed") from error
    expected = _build_registration_record(executor=executor, paths=paths)
    if _canonical_json_bytes(observed) != _canonical_json_bytes(expected):
        raise ValueError("registration differs from current source/runtime/ancestry")
    return observed, sha256_file(paths.registration_path)


def _reserve_run_attempt(*, paths: RunPaths, registration_sha256: str) -> str:
    _assert_external_sentinel(
        paths.run_attempt_path, paths.run_directory, label="final run attempt"
    )
    payload = {
        "schema_version": f"{EXPERIMENT_ID}_run_attempt_v1",
        "registration_sha256": _validated_sha256(
            registration_sha256, label="registration_sha256"
        ),
        "canonical_run_directory": str(paths.run_directory),
        "status": "irreversible_final_attempt_reserved_before_run_directory_or_model",
    }
    exclusive_json_write(paths.run_attempt_path, payload)
    return sha256_file(paths.run_attempt_path)


def _verify_live_custody(
    *,
    executor: Any,
    paths: RunPaths,
    registration_sha256: str,
    run_attempt_sha256: str,
) -> None:
    _record, observed_registration = _load_registration(executor=executor, paths=paths)
    if observed_registration != registration_sha256:
        raise ValueError("registration hash changed during final run")
    expected = {
        "schema_version": f"{EXPERIMENT_ID}_run_attempt_v1",
        "registration_sha256": registration_sha256,
        "canonical_run_directory": str(paths.run_directory),
        "status": "irreversible_final_attempt_reserved_before_run_directory_or_model",
    }
    try:
        observed = json.loads(paths.run_attempt_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("final run attempt is missing or malformed") from error
    if (
        _canonical_json_bytes(observed) != _canonical_json_bytes(expected)
        or sha256_file(paths.run_attempt_path) != run_attempt_sha256
    ):
        raise ValueError("final run-attempt sentinel changed")


def _validate_final_preflight(record: Any) -> None:
    if (
        not isinstance(record, Mapping)
        or record.get("schema_version") != "e32_5_final_input_preflight_v1"
        or tuple(record.get("case_order", ())) != CASE_ORDER
        or record.get("verified") is not True
        or record.get("model_calls") != 0
    ):
        raise ValueError("final input preflight identity changed")
    cases = record.get("cases")
    expected_case_keys = tuple(str(case_id) for case_id in CASE_ORDER)
    if not isinstance(cases, Mapping) or tuple(cases) != expected_case_keys:
        raise ValueError("final input preflight case order changed")
    for case_id in CASE_ORDER:
        splits = cases[str(case_id)]
        expected_splits = SPLIT_ORDER_BY_CASE[case_id]
        if not isinstance(splits, Mapping) or tuple(splits) != expected_splits:
            raise ValueError(f"case{case_id} input preflight split order changed")
        for split in expected_splits:
            _validate_split_preflight(
                splits[split],
                case_id=case_id,
                split=split,
                label=f"case{case_id} {split} preflight",
            )


def _validate_final_behavior(record: Any) -> None:
    if (
        not isinstance(record, Mapping)
        or record.get("schema_version") != "e32_5_final_behavior_screen_v1"
        or tuple(record.get("case_order", ())) != CASE_ORDER
        or record.get("behavior_only") is not True
        or "query_accounting" in record
    ):
        raise ValueError("final behavior screen identity changed")
    expected_cases = tuple(str(case_id) for case_id in CASE_ORDER)
    cases = record.get("cases")
    if not isinstance(cases, Mapping) or tuple(cases) != expected_cases:
        raise ValueError("final behavior case order changed")
    _final_behavior_rows(record, case_id=58, split="repair_confirmation")
    for case_id in GENERALIZATION_CASE_ORDER:
        _final_behavior_rows(record, case_id=case_id, split="discovery")
        _final_behavior_rows(record, case_id=case_id, split="confirmation")


def _validate_frozen_support(value: Any, *, case_id: int) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != "e32_5_frozen_support_v1"
        or value.get("case_id") != case_id
    ):
        raise ValueError("frozen support identity changed")
    support = tuple(value.get("support", ()))
    if (
        not support
        or len(support) != len(set(support))
        or value.get("support_size") != len(support)
        or any(not isinstance(edge, str) or not edge for edge in support)
    ):
        raise ValueError("frozen support is malformed")
    json_safe = _json_safe(value)
    _canonical_json_bytes(json_safe)
    return json_safe


def _authorization_record(
    *,
    case_id: int,
    source_split: str,
    support: Mapping[str, Any],
    source_artifact_sha256: str,
    source_custody_sha256: str,
    registration_sha256: str,
    run_attempt_sha256: str,
) -> dict[str, Any]:
    destination = "repair_confirmation" if case_id == 58 else "confirmation"
    return {
        "schema_version": f"{EXPERIMENT_ID}_support_authorization_v1",
        "case_id": case_id,
        "source_split": source_split,
        "destination_split": destination,
        "registration_sha256": registration_sha256,
        "run_attempt_sha256": run_attempt_sha256,
        "source_artifact_sha256": source_artifact_sha256,
        "source_custody_sha256": source_custody_sha256,
        "frozen_support": support,
        "frozen_support_sha256": _json_sha256(support),
        "status": "durable_before_any_confirmation_outcome",
    }


def _authorization_filename(case_id: int) -> str:
    suffix = "repair" if case_id == 58 else "confirmation"
    return f"case{case_id}_{suffix}_authorization.json"


@dataclass(frozen=True)
class _ScoreReservation:
    path: Path
    payload_sha256: str
    registration_sha256: str
    run_attempt_sha256: str
    final_artifact_sha256: Mapping[str, str]
    final_custody_sha256: Mapping[str, str]
    development_ancestry: Mapping[str, Any]


def _reserve_score(
    *,
    paths: RunPaths,
    registration: Mapping[str, Any],
    registration_sha256: str,
    run_attempt_sha256: str,
    artifact_hashes: Mapping[str, str],
    custody_hashes: Mapping[str, str],
) -> _ScoreReservation:
    expected_artifacts = tuple(
        _artifact_filename(case_id, split) for case_id, split in _FINAL_ARTIFACT_KEYS
    )
    if tuple(artifact_hashes) != expected_artifacts:
        raise ValueError("score attempt requires the exact ordered final artifact set")
    if tuple(custody_hashes) != _FINAL_CUSTODY_FILENAMES:
        raise ValueError("score attempt requires the exact ordered custody closure")
    expected_entries = set(artifact_hashes) | set(custody_hashes)
    observed_entries = {path.name for path in paths.run_directory.iterdir()}
    if observed_entries != expected_entries or any(
        not (paths.run_directory / filename).is_file() for filename in expected_entries
    ):
        raise ValueError("pre-score run-directory file set differs from custody")
    for filename, digest in {**artifact_hashes, **custody_hashes}.items():
        _validated_sha256(digest, label=f"hash for {filename}")
        if sha256_file(paths.run_directory / filename) != digest:
            raise ValueError("score-bound final artifact changed before reservation")
    ancestry = registration["development_ancestry"]
    observed_ancestry = _load_development_ancestry(paths.development)
    if _canonical_json_bytes(ancestry) != _canonical_json_bytes(observed_ancestry):
        raise ValueError("development ancestry changed before score reservation")
    payload = {
        "schema_version": f"{EXPERIMENT_ID}_score_attempt_v1",
        "registration_sha256": registration_sha256,
        "run_attempt_sha256": run_attempt_sha256,
        "final_acquisition_artifact_sha256": dict(artifact_hashes),
        "final_custody_sha256": dict(custody_hashes),
        "development_ancestry": ancestry,
        "status": "durable_before_any_final_artifact_reload_or_raw_score_recomputation",
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
        dict(ancestry),
    )


def _verify_score_reservation(reservation: _ScoreReservation, paths: RunPaths) -> None:
    try:
        payload = json.loads(reservation.path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("score attempt is missing or malformed") from error
    if _json_sha256(payload) != reservation.payload_sha256:
        raise ValueError("score attempt changed after reservation")
    for filename, digest in {
        **reservation.final_artifact_sha256,
        **reservation.final_custody_sha256,
    }.items():
        if sha256_file(paths.run_directory / filename) != digest:
            raise ValueError("score-bound final file changed after reservation")
    if _canonical_json_bytes(_load_development_ancestry(paths.development)) != (
        _canonical_json_bytes(reservation.development_ancestry)
    ):
        raise ValueError("score-bound development ancestry changed")


def _load_pilot_bundle(
    *, paths: RunPaths, ancestry: Mapping[str, Any]
) -> dict[str, Any]:
    return _load_bundle(
        path=paths.development.run_directory / "case58_development.pt",
        expected_sha256=ancestry["terminal_file_sha256"]["case58_development.pt"],
        case_id=58,
        split="development",
        attempt_sha256=ancestry["development_attempt_sha256"],
        registration_sha256=None,
        authorization_sha256=None,
    )


def _load_score_panel(
    *, reservation: _ScoreReservation, paths: RunPaths
) -> dict[tuple[int, str], dict[str, Any]]:
    _verify_score_reservation(reservation, paths)
    panel: dict[tuple[int, str], dict[str, Any]] = {
        (58, "development"): _load_pilot_bundle(
            paths=paths, ancestry=reservation.development_ancestry
        )
    }
    for case_id, split in _FINAL_ARTIFACT_KEYS:
        filename = _artifact_filename(case_id, split)
        authorization_sha256 = None
        if "confirmation" in split:
            authorization_path = paths.run_directory / _authorization_filename(case_id)
            authorization_sha256 = sha256_file(authorization_path)
        panel[(case_id, split)] = _load_bundle(
            path=paths.run_directory / filename,
            expected_sha256=reservation.final_artifact_sha256[filename],
            case_id=case_id,
            split=split,
            attempt_sha256=reservation.run_attempt_sha256,
            registration_sha256=reservation.registration_sha256,
            authorization_sha256=authorization_sha256,
        )
    _verify_score_reservation(reservation, paths)
    return panel


def _run_registered_experiment(
    *, executor: Any, paths: RunPaths, confirm: bool = False
) -> dict[str, Any]:
    if not confirm:
        raise ValueError("final E32.5 run requires confirm=True")
    resolved = paths.resolved()
    registration, registration_sha256 = _load_registration(
        executor=executor, paths=resolved
    )
    if resolved.run_attempt_path.exists():
        raise FileExistsError("canonical E32.5 run attempt is already consumed")
    if resolved.run_directory.exists():
        raise FileExistsError("canonical E32.5 run directory is already consumed")
    run_attempt_sha256 = _reserve_run_attempt(
        paths=resolved, registration_sha256=registration_sha256
    )
    resolved.run_directory.mkdir(parents=True, exist_ok=False)
    acquisition_attempt_path = resolved.run_directory / "acquisition_attempt.json"
    exclusive_json_write(
        acquisition_attempt_path,
        {
            "schema_version": f"{EXPERIMENT_ID}_acquisition_attempt_v1",
            "registration_sha256": registration_sha256,
            "run_attempt_sha256": run_attempt_sha256,
            "phase_order": (
                "final_input_preflight",
                "final_behavior_screen",
                "all_prospective_discoveries",
                "all_four_support_authorizations_and_recomputations",
                "global_authorization_barrier",
                "case58_repair_and_all_prospective_confirmations",
                "exact_accounting_and_custody",
                "score_attempt",
                "pilot_and_final_raw_artifact_reload_and_exact_score_replay",
            ),
            "status": "durable_before_final_raw_input_generation_or_model_call",
        },
    )

    bundles: dict[tuple[int, str], Mapping[str, Any]] = {}
    artifact_hashes: dict[str, str] = {}
    custody_hashes: dict[str, str] = {
        acquisition_attempt_path.name: sha256_file(acquisition_attempt_path)
    }
    authorizations: dict[int, dict[str, Any]] = {}
    score_reservation: _ScoreReservation | None = None
    try:
        preflight_record, private_state = executor.final_input_preflight()
        _validate_final_preflight(preflight_record)
        preflight_path = resolved.run_directory / "input_preflight.json"
        exclusive_json_write(
            preflight_path,
            {
                **dict(preflight_record),
                "registration_sha256": registration_sha256,
                "run_attempt_sha256": run_attempt_sha256,
                "status": "durable_before_any_final_behavior_model",
            },
        )
        custody_hashes[preflight_path.name] = sha256_file(preflight_path)
        _verify_live_custody(
            executor=executor,
            paths=resolved,
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
        )

        behavior_record, panels = executor.final_behavior_screen(private_state)
        _validate_final_behavior(behavior_record)
        behavior_path = resolved.run_directory / "behavior_screen.json"
        exclusive_json_write(
            behavior_path,
            {
                **dict(behavior_record),
                "registration_sha256": registration_sha256,
                "run_attempt_sha256": run_attempt_sha256,
                "accounting_location": "split_bundles_only_no_duplicate_sum",
                "status": "durable_before_any_final_intervention",
            },
        )
        custody_hashes[behavior_path.name] = sha256_file(behavior_path)
        _verify_live_custody(
            executor=executor,
            paths=resolved,
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
        )

        # The exposed pilot may be reopened before final scoring only to derive
        # the frozen repair support.  No final D/C artifact is reopened here.
        pilot = _load_pilot_bundle(
            paths=resolved, ancestry=registration["development_ancestry"]
        )
        bundles[(58, "development")] = pilot
        for case_id in GENERALIZATION_CASE_ORDER:
            discovery = _validate_bundle(
                executor.acquire_discovery(case_id, panels[case_id]),
                case_id=case_id,
                split="discovery",
            )
            _require_selected_rows(
                discovery,
                _final_behavior_rows(
                    behavior_record, case_id=case_id, split="discovery"
                ),
                label=f"case{case_id} discovery",
            )
            artifact, custody = _write_bundle(
                run_directory=resolved.run_directory,
                bundle=discovery,
                case_id=case_id,
                split="discovery",
                attempt_sha256=run_attempt_sha256,
                registration_sha256=registration_sha256,
                authorization_sha256=None,
            )
            artifact_hashes[_artifact_filename(case_id, "discovery")] = artifact
            custody_hashes[_custody_filename(case_id, "discovery")] = custody
            bundles[(case_id, "discovery")] = discovery

        source_bundles = {
            58: pilot,
            **{
                case_id: bundles[(case_id, "discovery")]
                for case_id in GENERALIZATION_CASE_ORDER
            },
        }
        first_freeze: dict[int, dict[str, Any]] = {}
        for case_id in CASE_ORDER:
            frozen = _validate_frozen_support(
                executor.freeze_support(case_id, source_bundles[case_id]),
                case_id=case_id,
            )
            first_freeze[case_id] = frozen
            if case_id == 58:
                source_artifact = registration["development_ancestry"][
                    "terminal_file_sha256"
                ]["case58_development.pt"]
                source_custody = registration["development_ancestry"][
                    "terminal_file_sha256"
                ]["case58_development_custody.json"]
                source_split = "development"
            else:
                source_artifact = artifact_hashes[
                    _artifact_filename(case_id, "discovery")
                ]
                source_custody = custody_hashes[_custody_filename(case_id, "discovery")]
                source_split = "discovery"
            authorization = _authorization_record(
                case_id=case_id,
                source_split=source_split,
                support=frozen,
                source_artifact_sha256=source_artifact,
                source_custody_sha256=source_custody,
                registration_sha256=registration_sha256,
                run_attempt_sha256=run_attempt_sha256,
            )
            path = resolved.run_directory / _authorization_filename(case_id)
            exclusive_json_write(path, authorization)
            digest = sha256_file(path)
            custody_hashes[path.name] = digest
            authorizations[case_id] = {
                **authorization,
                "authorization_sha256": digest,
            }

        reloaded_sources: dict[int, Mapping[str, Any]] = {
            58: _load_pilot_bundle(
                paths=resolved, ancestry=registration["development_ancestry"]
            )
        }
        for case_id in GENERALIZATION_CASE_ORDER:
            filename = _artifact_filename(case_id, "discovery")
            reloaded_sources[case_id] = _load_bundle(
                path=resolved.run_directory / filename,
                expected_sha256=artifact_hashes[filename],
                case_id=case_id,
                split="discovery",
                attempt_sha256=run_attempt_sha256,
                registration_sha256=registration_sha256,
                authorization_sha256=None,
            )
        for case_id in CASE_ORDER:
            recomputed = _validate_frozen_support(
                executor.freeze_support(case_id, reloaded_sources[case_id]),
                case_id=case_id,
            )
            if _canonical_json_bytes(recomputed) != _canonical_json_bytes(
                first_freeze[case_id]
            ):
                raise ValueError(f"case-{case_id} authorization recomputation differs")
        _verify_live_custody(
            executor=executor,
            paths=resolved,
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
        )
        for case_id in CASE_ORDER:
            path = resolved.run_directory / _authorization_filename(case_id)
            authorization = authorizations[case_id]
            if sha256_file(path) != authorization["authorization_sha256"]:
                raise ValueError("support authorization changed before global barrier")
            if (
                case_id != 58
                and sha256_file(
                    resolved.run_directory / _artifact_filename(case_id, "discovery")
                )
                != authorization["source_artifact_sha256"]
            ):
                raise ValueError("discovery artifact changed before global barrier")
        confirmation_paths = tuple(
            resolved.run_directory / _artifact_filename(case_id, split)
            for case_id, split in _FINAL_ARTIFACT_KEYS
            if "confirmation" in split
        )
        if any(path.exists() for path in confirmation_paths):
            raise FileExistsError(
                "confirmation artifact exists before global authorization"
            )
        global_path = resolved.run_directory / "global_support_authorization.json"
        exclusive_json_write(
            global_path,
            {
                "schema_version": f"{EXPERIMENT_ID}_global_support_authorization_v1",
                "registration_sha256": registration_sha256,
                "run_attempt_sha256": run_attempt_sha256,
                "authorization_sha256": {
                    str(case_id): authorizations[case_id]["authorization_sha256"]
                    for case_id in CASE_ORDER
                },
                "all_discoveries_durable": True,
                "all_supports_recomputed_exactly": True,
                "confirmation_artifacts_present": False,
                "status": "global_barrier_durable_before_first_confirmation_call",
            },
        )
        custody_hashes[global_path.name] = sha256_file(global_path)
        _verify_live_custody(
            executor=executor,
            paths=resolved,
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
        )
        for case_id in CASE_ORDER:
            path = resolved.run_directory / _authorization_filename(case_id)
            if sha256_file(path) != authorizations[case_id]["authorization_sha256"]:
                raise ValueError("authorization changed after global barrier")
        if sha256_file(global_path) != custody_hashes[global_path.name]:
            raise ValueError("global authorization changed before confirmation")

        repair = _validate_bundle(
            executor.acquire_repair_confirmation(panels[58], authorizations[58]),
            case_id=58,
            split="repair_confirmation",
        )
        _require_selected_rows(
            repair,
            _final_behavior_rows(
                behavior_record, case_id=58, split="repair_confirmation"
            ),
            label="case58 repair confirmation",
        )
        artifact, custody = _write_bundle(
            run_directory=resolved.run_directory,
            bundle=repair,
            case_id=58,
            split="repair_confirmation",
            attempt_sha256=run_attempt_sha256,
            registration_sha256=registration_sha256,
            authorization_sha256=authorizations[58]["authorization_sha256"],
        )
        artifact_hashes[_artifact_filename(58, "repair_confirmation")] = artifact
        custody_hashes[_custody_filename(58, "repair_confirmation")] = custody
        bundles[(58, "repair_confirmation")] = repair
        for case_id in GENERALIZATION_CASE_ORDER:
            confirmation = _validate_bundle(
                executor.acquire_confirmation(
                    case_id, panels[case_id], authorizations[case_id]
                ),
                case_id=case_id,
                split="confirmation",
            )
            _require_selected_rows(
                confirmation,
                _final_behavior_rows(
                    behavior_record, case_id=case_id, split="confirmation"
                ),
                label=f"case{case_id} confirmation",
            )
            artifact, custody = _write_bundle(
                run_directory=resolved.run_directory,
                bundle=confirmation,
                case_id=case_id,
                split="confirmation",
                attempt_sha256=run_attempt_sha256,
                registration_sha256=registration_sha256,
                authorization_sha256=authorizations[case_id]["authorization_sha256"],
            )
            artifact_hashes[_artifact_filename(case_id, "confirmation")] = artifact
            custody_hashes[_custody_filename(case_id, "confirmation")] = custody
            bundles[(case_id, "confirmation")] = confirmation

        ordered_artifacts = {
            _artifact_filename(case_id, split): artifact_hashes[
                _artifact_filename(case_id, split)
            ]
            for case_id, split in _FINAL_ARTIFACT_KEYS
        }
        final_accounting = _aggregate_accounting(
            tuple(
                (
                    f"case{case_id}_{split}",
                    bundles[(case_id, split)]["query_accounting"],
                )
                for case_id, split in _FINAL_ARTIFACT_KEYS
            )
        )
        _within_ceiling(final_accounting, _FINAL_QUERY_CEILING, label="final run")
        pilot_accounting = registration["development_ancestry"][
            "pilot_query_accounting"
        ]
        cumulative_accounting = _aggregate_accounting(
            (
                ("nonclaim_case58_development", pilot_accounting),
                ("final_run", final_accounting),
            )
        )
        _within_ceiling(
            cumulative_accounting,
            _CUMULATIVE_QUERY_CEILING,
            label="pilot plus final cumulative work",
        )
        acquisition_custody_path = resolved.run_directory / "acquisition_custody.json"
        exclusive_json_write(
            acquisition_custody_path,
            {
                "schema_version": f"{EXPERIMENT_ID}_acquisition_custody_v1",
                "registration_sha256": registration_sha256,
                "run_attempt_sha256": run_attempt_sha256,
                "final_acquisition_artifact_sha256": ordered_artifacts,
                "stage_custody_sha256": dict(custody_hashes),
                "final_query_accounting": final_accounting,
                "pilot_query_accounting_reported_separately": pilot_accounting,
                "cumulative_query_accounting": cumulative_accounting,
                "final_query_ceiling": _FINAL_QUERY_CEILING,
                "final_query_ceiling_pass": True,
                "behavior_accounting_counted_once_in_split_bundles": True,
                "status": "complete_before_score_attempt",
            },
        )
        custody_hashes[acquisition_custody_path.name] = sha256_file(
            acquisition_custody_path
        )
        _verify_live_custody(
            executor=executor,
            paths=resolved,
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
        )
        score_reservation = _reserve_score(
            paths=resolved,
            registration=registration,
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
            artifact_hashes=ordered_artifacts,
            custody_hashes=custody_hashes,
        )
        score_panel = _load_score_panel(reservation=score_reservation, paths=resolved)
        scientific = executor.score_raw_panel(score_panel)
        replay_panel = _load_score_panel(reservation=score_reservation, paths=resolved)
        replay = executor.score_raw_panel(replay_panel)
        if not _scientific_equal(scientific, replay):
            raise RuntimeError("exact raw score replay differs")
        _verify_score_reservation(score_reservation, resolved)
        _verify_live_custody(
            executor=executor,
            paths=resolved,
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
        )
        score_path = resolved.run_directory / "scientific_score.pt"
        _exclusive_torch_write(score_path, scientific)
        score_sha256 = sha256_file(score_path)
        _verify_score_reservation(score_reservation, resolved)
        _verify_live_custody(
            executor=executor,
            paths=resolved,
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
        )
        summary = {
            **_json_safe(scientific),
            "schema_version": f"{EXPERIMENT_ID}_summary_v1",
            "registration_sha256": registration_sha256,
            "run_attempt_sha256": run_attempt_sha256,
            "acquisition_attempt_sha256": sha256_file(acquisition_attempt_path),
            "score_attempt_sha256": sha256_file(score_reservation.path),
            "scientific_score_sha256": score_sha256,
            "raw_score_recomputation_sha256": _json_sha256(_json_safe(scientific)),
            "final_acquisition_artifact_sha256": ordered_artifacts,
            "custody_sha256": custody_hashes,
            "development_ancestry": registration["development_ancestry"],
            "final_query_accounting": final_accounting,
            "cumulative_query_accounting": cumulative_accounting,
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
                    "schema_version": f"{EXPERIMENT_ID}_failure_v1",
                    "registration_sha256": registration_sha256,
                    "run_attempt_sha256": run_attempt_sha256,
                    "score_attempt_sha256": (
                        None
                        if score_reservation is None
                        else sha256_file(score_reservation.path)
                    ),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "completed_artifact_sha256": artifact_hashes,
                    "completed_custody_sha256": custody_hashes,
                },
            )
        raise


def run_registered_experiment(*, confirm: bool = False) -> dict[str, Any]:
    """Run the sole final experiment; no executor can be supplied."""
    return _run_registered_experiment(
        executor=_ProductionExecutor(), paths=CANONICAL_PATHS, confirm=confirm
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--develop-case58", action="store_true")
    action.add_argument("--build-registration", action="store_true")
    action.add_argument("--seal-registration", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args(argv)
    if args.develop_case58:
        output = develop_case58(confirm=args.confirm)
    elif args.build_registration:
        output = build_registration_record()
    elif args.seal_registration:
        output = seal_registration()
    else:
        output = run_registered_experiment(confirm=args.confirm)
    print(json.dumps(_json_safe(output), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_DEVELOPMENT_ATTEMPT_PATH",
    "CANONICAL_DEVELOPMENT_RUN_DIRECTORY",
    "CANONICAL_REGISTRATION_PATH",
    "CANONICAL_RUN_ATTEMPT_PATH",
    "CANONICAL_RUN_DIRECTORY",
    "DEVELOPMENT_TUNING_DISCLOSURE",
    "DevelopmentPaths",
    "RunPaths",
    "build_registration_record",
    "develop_case58",
    "run_registered_experiment",
    "seal_registration",
]
