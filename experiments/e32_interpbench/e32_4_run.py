# ruff: noqa: EXE002
"""Canonical write-once runner for E32.4 internal edge-set faithfulness.

The public entrypoints have no executor injection.  A private executor seam is
kept only for lifecycle tests.  Production order is fixed: every raw input pool
is inspected without a model, every fresh panel is behavior-screened, all D
artifacts and their support authorizations become durable, every authorization
is reverified, all C artifacts become durable, and only then is a score attempt
reserved and the tensor artifacts reopened.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from experiments.e32_interpbench.adapter import DEFAULT_MODEL_ROOT, sha256_file
from experiments.e32_interpbench.e32_2_acquisition import token_rows
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
    SMOKE_REQUESTS,
    case_universe,
    load_verified_model,
    load_verified_official_case,
    verify_case_artifacts,
)
from experiments.e32_interpbench.e32_3_cohort import (
    split_request as e32_3_split_request,
)
from experiments.e32_interpbench.e32_3_v2_input_repair import (
    EXPECTED_RAW_POOL_SHA256,
    _select_behaviorally_eligible_from_pair,
    generate_official_pair_allow_raw_duplicates,
    raw_candidate_pool_audit,
)
from experiments.e32_interpbench.e32_4_protocol import (
    CASE_GEOMETRY,
    CASE_ORDER,
    CORE_ANCESTRY,
    CORE_CASE_ANCESTRY,
    CORE_NODE_SUPPORTS,
    DETERMINISTIC_METHOD_ORDER,
    EXPECTED_DISCOVERY_QUERY_ACCOUNTING,
    EXPECTED_QUERY_CEILINGS,
    INPUT_PREFLIGHT,
    MANDATORY_SPLITS,
    PRIMARY_METHOD,
    RANDOM_CONTROL_REPLICATES,
    descriptive_support_sizes,
    deterministic_random_supports,
    primary_support_size,
    protocol_manifest,
    split_request,
)
from experiments.e32_interpbench.eap_baseline import DEFAULT_AUTO_CIRCUIT_DIR

EXPERIMENT_ID = "e32_4_internal_edge_set_faithfulness_v1"
CLAIM_BOUNDARY = "boundary-conditioned internal inter-component fine-edge support"


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
    REPOSITORY_ROOT
    / "notes"
    / "E32.4-INTERNAL-EDGE-SET-FAITHFULNESS-REGISTRATION.json",
    REPOSITORY_ROOT / "notes" / "E32.4-INTERNAL-EDGE-SET-FAITHFULNESS-RUN-ATTEMPT.json",
    REPOSITORY_ROOT / "runs" / EXPERIMENT_ID,
).resolved()
CANONICAL_REGISTRATION_PATH = CANONICAL_PATHS.registration_path
CANONICAL_RUN_ATTEMPT_PATH = CANONICAL_PATHS.run_attempt_path
CANONICAL_RUN_DIRECTORY = CANONICAL_PATHS.run_directory

E32_3_REGISTRATION_PATH = (
    REPOSITORY_ROOT / "notes" / "E32.3-SPARSE-CORE-V2-INPUT-REPAIR-REGISTRATION.json"
)
E32_3_RUN_DIRECTORY = REPOSITORY_ROOT / "runs" / "e32_3_sparse_core_v2_input_repair"
E32_3_SUMMARY_PATH = E32_3_RUN_DIRECTORY / "summary.json"
E32_3_V2_SOURCE_PATH = (
    REPOSITORY_ROOT / "experiments/e32_interpbench/e32_3_v2_input_repair.py"
)
E32_3_V2_SOURCE_SHA256 = (
    "5c1efe43ab15400f24d4c6c977ad4d246e2486fb5714b8e69df00cc465b5078c"
)
AUTO_CIRCUIT_ABLATION_RELATIVE_PATH = "auto_circuit/utils/ablation_activations.py"
AUTO_CIRCUIT_ABLATION_SHA256 = (
    "2fd77a905140c0e3c63e8dc2494b316639e0216dac6f5e4410328bc27193c588"
)

_SOURCE_ROOT_MODULES = (
    "experiments.e32_interpbench.e32_4_run",
    "experiments.e32_interpbench.e32_4_protocol",
    "experiments.e32_interpbench.e32_4_eap",
    "experiments.e32_interpbench.e32_4_edge_sets",
    "experiments.e32_interpbench.e32_4_scoring",
)

_DISCOVERY_SCORE_INPUT_KEYS = (
    "induced_fine_edge_ids",
    "discovery_selector_scores",
)
_CONFIRMATION_SCORE_INPUT_KEYS = (
    "clean_output",
    "actual_corrupt_output",
    "clean_target_indices",
    "resample_all_internal_output",
    "resample_induced_core_output",
    "resample_outside_core_output",
    "resample_deterministic_outputs",
    "resample_exact_mass_curve_outputs",
    "resample_random_outputs",
    "resample_within_core_primary_complement_output",
    "mean_all_internal_output",
    "mean_induced_core_output",
    "mean_outside_core_output",
    "mean_primary_output",
    "mean_primary_complement_output",
)
_TOTAL_ACCOUNTING_KEYS = (
    "physical_forwards",
    "model_examples",
    "gradient_forwards",
    "gradient_examples",
    "backward_passes_inferred_from_pinned_control_flow",
)
_EXPECTED_PREFLIGHT_CAPACITY = {
    58: {"discovery": 96, "confirmation": 192},
    129: {"discovery": 96, "confirmation": 192},
    25: {"discovery": 86, "confirmation": 172},
}


def _canonical_json_bytes(value: Any) -> bytes:
    # The round trip normalizes integer mapping keys to their on-disk JSON
    # strings before sorting (otherwise numeric and lexical sort orders differ).
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
    if not module.startswith("experiments.e32_interpbench"):
        return None
    path = REPOSITORY_ROOT / f"{module.replace('.', '/')}.py"
    if path.is_file():
        return path
    package = REPOSITORY_ROOT / module.replace(".", "/") / "__init__.py"
    return package if package.is_file() else None


def _source_relative_paths() -> tuple[str, ...]:
    """Return the transitive local import closure of every E32.4 root."""
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
                imports = tuple(alias.name for alias in node.names)
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
    observed = _source_relative_paths()
    if observed != SOURCE_RELATIVE_PATHS:
        raise RuntimeError("registered E32.4 local source closure changed")
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


def _verify_e32_3_ancestry() -> dict[str, Any]:
    """Verify every sealed E32.3 anchor and its registered source closure."""
    expected_artifacts = {
        "registration": (E32_3_REGISTRATION_PATH, CORE_ANCESTRY["registration_sha256"]),
        "summary": (E32_3_SUMMARY_PATH, CORE_ANCESTRY["summary_sha256"]),
        **{
            f"case{case_id}_authorization": (
                E32_3_RUN_DIRECTORY / f"case{case_id}_confirmation_authorization.json",
                CORE_CASE_ANCESTRY[case_id]["authorization_sha256"],
            )
            for case_id in CASE_ORDER
        },
        "v2_source": (E32_3_V2_SOURCE_PATH, E32_3_V2_SOURCE_SHA256),
    }
    observed_artifacts: dict[str, str] = {}
    for label, (path, expected) in expected_artifacts.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing sealed E32.3 {label}: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"sealed E32.3 {label} hash changed")
        observed_artifacts[label] = observed

    try:
        registration = json.loads(E32_3_REGISTRATION_PATH.read_text())
    except json.JSONDecodeError as error:
        raise ValueError("sealed E32.3 registration is malformed") from error
    registered_sources = registration.get("source_sha256")
    if not isinstance(registered_sources, Mapping) or not registered_sources:
        raise ValueError("sealed E32.3 registration lacks its source closure")
    for relative, expected in registered_sources.items():
        _validated_sha256(expected, label=f"sealed E32.3 source {relative}")
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"sealed E32.3 source closure changed: {relative}")
    registered_v2 = (
        registration.get("dependencies", {})
        .get("e32_3_v2_input_repair", {})
        .get("v2_source_sha256")
    )
    if registered_v2 != E32_3_V2_SOURCE_SHA256:
        raise ValueError("sealed E32.3 V2 source anchor changed")
    try:
        summary = json.loads(E32_3_SUMMARY_PATH.read_text())
    except json.JSONDecodeError as error:
        raise ValueError("sealed E32.3 summary is malformed") from error
    for case_id in CASE_ORDER:
        selected = tuple(summary["cases"][str(case_id)]["selected_nodes"])
        if selected != CORE_NODE_SUPPORTS[case_id]:
            raise ValueError(f"sealed E32.3 case-{case_id} node core changed")
        compact_digest = hashlib.sha256(
            json.dumps(selected, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            compact_digest
            != CORE_CASE_ANCESTRY[case_id]["selected_node_support_sha256"]
        ):
            raise ValueError(f"sealed E32.3 case-{case_id} node-core hash changed")
        authorization_path = (
            E32_3_RUN_DIRECTORY / f"case{case_id}_confirmation_authorization.json"
        )
        try:
            sealed_authorization = json.loads(authorization_path.read_text())
            authorized_nodes = tuple(
                sealed_authorization["frozen_supports"]["deterministic_supports"][
                    "singleton_output_effect"
                ]["selected_nodes"]
            )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"sealed E32.3 case-{case_id} authorization core is malformed"
            ) from error
        if authorized_nodes != selected:
            raise ValueError(
                f"sealed E32.3 case-{case_id} summary/auth node cores differ"
            )
    return {
        "schema_version": "e32_4_e32_3_ancestry_verification_v1",
        "verified": True,
        "registration_sha256": observed_artifacts["registration"],
        "summary_sha256": observed_artifacts["summary"],
        "authorization_sha256": {
            str(case_id): observed_artifacts[f"case{case_id}_authorization"]
            for case_id in CASE_ORDER
        },
        "v2_source_sha256": observed_artifacts["v2_source"],
        "sealed_source_sha256": dict(registered_sources),
        "sealed_source_closure_unchanged": True,
    }


def _verify_auto_circuit_ablation_source() -> dict[str, str]:
    """Bind the directly executed ablation helper in checkout and environment."""
    checkout = DEFAULT_AUTO_CIRCUIT_DIR / AUTO_CIRCUIT_ABLATION_RELATIVE_PATH
    installed = (
        Path(sys.prefix)
        / "lib/python3.11/site-packages"
        / AUTO_CIRCUIT_ABLATION_RELATIVE_PATH
    )
    for label, path in (("checkout", checkout), ("installed", installed)):
        if not path.is_file() or sha256_file(path) != AUTO_CIRCUIT_ABLATION_SHA256:
            raise ValueError(f"AutoCircuit ablation source changed in {label}")
    return {
        "relative_path": AUTO_CIRCUIT_ABLATION_RELATIVE_PATH,
        "sha256": AUTO_CIRCUIT_ABLATION_SHA256,
        "checkout_path": str(checkout.resolve()),
        "installed_path": str(installed.resolve()),
    }


@dataclass(frozen=True)
class _RawCaseState:
    case_id: int
    case: Any
    historical_rows: tuple[tuple[int, ...], ...]
    discovery_candidates: Any
    confirmation_candidates: Any
    discovery_raw_rows: tuple[tuple[int, ...], ...]
    confirmation_excluded_rows: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _FreshCasePanel:
    case_id: int
    discovery: Any
    confirmation: Any


def _unique_rows(rows: Collection[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    normalized: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for value in rows:
        row = tuple(value)
        if not row or any(
            isinstance(token, bool) or not isinstance(token, int) for token in row
        ):
            raise ValueError("raw token rows must be nonempty integer sequences")
        if row not in seen:
            seen.add(row)
            normalized.append(row)
    return tuple(normalized)


def _pair_rows(pair: Any, *, label: str) -> tuple[tuple[int, ...], ...]:
    return (
        *token_rows(pair.clean_inputs, label=f"{label} clean inputs"),
        *token_rows(pair.corrupt_inputs, label=f"{label} corrupt inputs"),
    )


def _response_blind_selectable_count(
    pair: Any, *, excluded_rows: Collection[Sequence[int]]
) -> int:
    used = set(_unique_rows(excluded_rows))
    clean = token_rows(pair.clean_inputs, label="candidate clean inputs")
    corrupt = token_rows(pair.corrupt_inputs, label="candidate corrupt inputs")
    if len(clean) != len(corrupt):
        raise ValueError("candidate role counts differ")
    accepted = 0
    for clean_row, corrupt_row in zip(clean, corrupt, strict=True):
        if clean_row == corrupt_row or clean_row in used or corrupt_row in used:
            continue
        accepted += 1
        used.update((clean_row, corrupt_row))
    return accepted


def _eligibility_record(value: Any) -> dict[str, Any]:
    return {
        "required_count": len(value.pair.clean_inputs),
        "candidate_count": int(value.candidate_count),
        "eligible_candidate_count": int(value.eligible_candidate_count),
        "row_disjoint_eligible_candidate_count": int(
            value.row_disjoint_eligible_candidate_count
        ),
        "selected_candidate_indices": tuple(value.selected_candidate_indices),
        "selection_policy": value.selection_policy,
        "all_selected_pairs_correct": True,
        "clean_incorrect_contexts": int(value.clean_incorrect_contexts),
        "source_incorrect_contexts": int(value.source_incorrect_contexts),
        "either_incorrect_contexts": int(value.either_incorrect_contexts),
        "eligibility_physical_forwards": int(value.eligibility_physical_forwards),
        "eligibility_model_examples": int(value.eligibility_model_examples),
        "excluded_token_row_count": int(value.excluded_token_row_count),
    }


class _ProductionExecutor:
    """Fixed production composition; constructed only by public entrypoints."""

    def registration_dependencies(self) -> dict[str, Any]:
        # These routines verify the clean source checkouts, complete runtime
        # package trees, and the installed-vs-checkout AutoCircuit identity.
        production = _require_production_environment()
        dependencies = _e32_2_dependency_record()
        return {
            "pinned_checkouts_and_runtime_trees": dependencies,
            "pinned_environment": production,
            "environment_replay": _e32_2_environment_record(),
            "direct_auto_circuit_ablation_source": (
                _verify_auto_circuit_ablation_source()
            ),
            "e32_3_sealed_ancestry": _verify_e32_3_ancestry(),
            "case_artifact_sha256": {
                str(case_id): verify_case_artifacts(DEFAULT_MODEL_ROOT, case_id)
                for case_id in CASE_ORDER
            },
            "learned_model_loaded": False,
        }

    def preflight_all_input_pools(self) -> tuple[dict[str, Any], object]:
        case_records: dict[str, Any] = {}
        states: dict[int, _RawCaseState] = {}
        for case_id in CASE_ORDER:
            case = load_verified_official_case(case_id)
            historical_audits: dict[str, Any] = {}
            historical_rows: list[tuple[int, ...]] = []
            smoke = SMOKE_REQUESTS[case_id]
            historical_requests = {
                "smoke": (
                    smoke.sample_count,
                    smoke.clean_seed,
                    smoke.corrupt_seed,
                ),
                **{
                    split: (
                        e32_3_split_request(case_id, split).candidate_count,
                        e32_3_split_request(case_id, split).clean_seed,
                        e32_3_split_request(case_id, split).corrupt_seed,
                    )
                    for split in MANDATORY_SPLITS
                },
            }
            for split, (count, clean_seed, corrupt_seed) in historical_requests.items():
                pair = generate_official_pair_allow_raw_duplicates(
                    case,
                    sample_count=count,
                    clean_seed=clean_seed,
                    corrupt_seed=corrupt_seed,
                )
                audit = raw_candidate_pool_audit(
                    case_id=case_id,
                    split=split,
                    candidate_count=count,
                    clean_seed=clean_seed,
                    corrupt_seed=corrupt_seed,
                    pair=pair,
                )
                if audit["identity_sha256"] != EXPECTED_RAW_POOL_SHA256[case_id][split]:
                    raise ValueError(
                        f"historical E32.3 case-{case_id} {split} raw identity changed"
                    )
                historical_audits[split] = audit
                historical_rows.extend(_pair_rows(pair, label=f"historical {split}"))
            old_rows = _unique_rows(historical_rows)
            expected_old_count = INPUT_PREFLIGHT["old_raw_role_exclusion_unique_rows"][
                case_id
            ]
            if len(old_rows) != expected_old_count:
                raise RuntimeError("historical E32.3 raw-row exclusion count changed")

            fresh_pairs: dict[str, Any] = {}
            fresh_audits: dict[str, Any] = {}
            for split in MANDATORY_SPLITS:
                request = split_request(case_id, split)
                pair = generate_official_pair_allow_raw_duplicates(
                    case,
                    sample_count=request.candidate_count,
                    clean_seed=request.clean_seed,
                    corrupt_seed=request.corrupt_seed,
                )
                audit = raw_candidate_pool_audit(
                    case_id=case_id,
                    split=split,
                    candidate_count=request.candidate_count,
                    clean_seed=request.clean_seed,
                    corrupt_seed=request.corrupt_seed,
                    pair=pair,
                )
                if (
                    audit["identity_sha256"] != request.raw_identity_sha256
                    or audit["target_identity_sha256"] != request.target_identity_sha256
                ):
                    raise ValueError(
                        f"fresh case-{case_id} {split} raw/target identity changed"
                    )
                fresh_pairs[split] = pair
                fresh_audits[split] = audit

            discovery_rows = _unique_rows(
                _pair_rows(fresh_pairs["discovery"], label="fresh discovery")
            )
            confirmation_rows = _unique_rows(
                _pair_rows(fresh_pairs["confirmation"], label="fresh confirmation")
            )
            old_new_overlap = len(
                set(old_rows).intersection((*discovery_rows, *confirmation_rows))
            )
            if (
                old_new_overlap
                != INPUT_PREFLIGHT["old_new_role_overlap_count"][case_id]
            ):
                raise RuntimeError("historical/fresh raw-row overlap count changed")
            confirmation_exclusions = _unique_rows((*old_rows, *discovery_rows))
            discovery_capacity = _response_blind_selectable_count(
                fresh_pairs["discovery"], excluded_rows=old_rows
            )
            confirmation_capacity = _response_blind_selectable_count(
                fresh_pairs["confirmation"], excluded_rows=confirmation_exclusions
            )
            if discovery_capacity != _EXPECTED_PREFLIGHT_CAPACITY[case_id]["discovery"]:
                raise RuntimeError("fresh discovery response-blind capacity regressed")
            if (
                confirmation_capacity
                != _EXPECTED_PREFLIGHT_CAPACITY[case_id]["confirmation"]
            ):
                raise RuntimeError(
                    "fresh confirmation response-blind capacity regressed"
                )

            case_records[str(case_id)] = {
                "case_id": case_id,
                "historical_e32_3_raw_pool_audits": historical_audits,
                "fresh_raw_pool_audits": fresh_audits,
                "registered_split_requests": {
                    split: asdict(split_request(case_id, split))
                    for split in MANDATORY_SPLITS
                },
                "historical_unique_raw_role_rows": len(old_rows),
                "fresh_discovery_unique_raw_role_rows": len(discovery_rows),
                "historical_fresh_role_overlap_count": old_new_overlap,
                "discovery_response_blind_capacity": discovery_capacity,
                "confirmation_response_blind_capacity_after_all_discovery_raw_rows": (
                    confirmation_capacity
                ),
                "all_raw_pool_identities_verified": True,
                "historical_e32_3_raw_rows_excluded": True,
                "confirmation_excludes_all_new_discovery_raw_rows": True,
            }
            states[case_id] = _RawCaseState(
                case_id=case_id,
                case=case,
                historical_rows=old_rows,
                discovery_candidates=fresh_pairs["discovery"],
                confirmation_candidates=fresh_pairs["confirmation"],
                discovery_raw_rows=discovery_rows,
                confirmation_excluded_rows=confirmation_exclusions,
            )
        return (
            {
                "schema_version": "e32_4_all_pool_input_preflight_v1",
                "verified": True,
                "model_calls": 0,
                "case_order": CASE_ORDER,
                "cases": case_records,
            },
            states,
        )

    def screen_all_fresh_panels(
        self, private_state: object
    ) -> tuple[dict[str, Any], dict[int, object]]:
        if not isinstance(private_state, Mapping) or tuple(private_state) != CASE_ORDER:
            raise TypeError(
                "preflight private state differs from the complete case panel"
            )
        records: dict[str, Any] = {}
        panels: dict[int, object] = {}
        for case_id in CASE_ORDER:
            state = private_state[case_id]
            if not isinstance(state, _RawCaseState) or state.case_id != case_id:
                raise TypeError(
                    "preflight case state changed before behavior screening"
                )
            model = load_verified_model(case_id, device="cpu")
            discovery_request = split_request(case_id, "discovery")
            confirmation_request = split_request(case_id, "confirmation")
            discovery = _select_behaviorally_eligible_from_pair(
                model,
                state.case,
                candidates=state.discovery_candidates,
                required_count=discovery_request.required_count,
                candidate_count=discovery_request.candidate_count,
                device="cpu",
                excluded_token_rows=state.historical_rows,
                require_unique_selected_token_rows=True,
            )
            confirmation = _select_behaviorally_eligible_from_pair(
                model,
                state.case,
                candidates=state.confirmation_candidates,
                required_count=confirmation_request.required_count,
                candidate_count=confirmation_request.candidate_count,
                device="cpu",
                excluded_token_rows=state.confirmation_excluded_rows,
                require_unique_selected_token_rows=True,
            )
            discovery_rows = _pair_rows(discovery.pair, label="selected discovery")
            confirmation_rows = _pair_rows(
                confirmation.pair, label="selected confirmation"
            )
            if (
                len(set(discovery_rows)) != len(discovery_rows)
                or len(set(confirmation_rows)) != len(confirmation_rows)
                or set(discovery_rows).intersection(confirmation_rows)
                or set(discovery_rows).intersection(state.historical_rows)
                or set(confirmation_rows).intersection(state.confirmation_excluded_rows)
            ):
                raise RuntimeError(
                    "fresh behavior-selected rows violate frozen exclusions"
                )
            panel = _FreshCasePanel(case_id, discovery, confirmation)
            panels[case_id] = panel
            records[str(case_id)] = {
                "case_id": case_id,
                "behavior_only": True,
                "discovery_clean_rows": token_rows(
                    discovery.pair.clean_inputs, label="discovery clean inputs"
                ),
                "discovery_corrupt_rows": token_rows(
                    discovery.pair.corrupt_inputs, label="discovery corrupt inputs"
                ),
                "confirmation_clean_rows": token_rows(
                    confirmation.pair.clean_inputs, label="confirmation clean inputs"
                ),
                "confirmation_corrupt_rows": token_rows(
                    confirmation.pair.corrupt_inputs,
                    label="confirmation corrupt inputs",
                ),
                "discovery_eligibility": _eligibility_record(discovery),
                "confirmation_eligibility": _eligibility_record(confirmation),
                "selected_rows_globally_unique_and_response_blind_fresh": True,
            }
        return records, panels

    def acquire_discovery(self, case_id: int, panel: object) -> dict[str, Any]:
        if not isinstance(panel, _FreshCasePanel) or panel.case_id != case_id:
            raise TypeError("discovery requires the screened fresh case panel")
        return _live_discovery_bundle(case_id=case_id, eligible=panel.discovery)

    def freeze_discovery_supports(
        self, case_id: int, discovery: Mapping[str, Any]
    ) -> dict[str, Any]:
        return _freeze_production_discovery_supports(
            case_id=case_id, discovery=discovery
        )

    def acquire_confirmation(
        self,
        case_id: int,
        panel: object,
        authorization: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(panel, _FreshCasePanel) or panel.case_id != case_id:
            raise TypeError("confirmation requires the screened fresh case panel")
        return _live_confirmation_bundle(
            case_id=case_id,
            eligible=panel.confirmation,
            authorization=authorization,
        )

    def score_complete_panel(
        self, panel: Mapping[tuple[int, str], Mapping[str, Any]]
    ) -> dict[str, Any]:
        return _score_production_panel(panel)


def _build_registration_record(*, executor: Any, paths: RunPaths) -> dict[str, Any]:
    resolved = paths.resolved()
    return {
        "schema_version": f"{EXPERIMENT_ID}_registration",
        "experiment_id": EXPERIMENT_ID,
        "canonical_registration_path": str(resolved.registration_path),
        "canonical_run_attempt_path": str(resolved.run_attempt_path),
        "canonical_run_directory": str(resolved.run_directory),
        "protocol": protocol_manifest(),
        "source_sha256": _source_hashes(),
        "dependencies": executor.registration_dependencies(),
        "execution": {
            "device": "cpu",
            "all_cases_mandatory": True,
            "all_raw_pool_input_identities_preflight_before_any_model_load": True,
            "all_fresh_panels_screened_before_any_intervention": True,
            "all_discoveries_durable_before_any_support_authorization": True,
            "all_authorizations_durable_and_globally_reverified_before_any_confirmation": True,
            "discovery_supports_recomputed_twice": True,
            "authorization_binds_discovery_artifact_and_custody": True,
            "score_attempt_precedes_every_e32_4_acquisition_artifact_reload": True,
            "exact_pure_score_replay_required": True,
            "production_executor_injectable": False,
            "ground_truth_opened_only_after_score_attempt": True,
            "claim_boundary": CLAIM_BOUNDARY,
        },
    }


def build_registration_record() -> dict[str, Any]:
    """Build the canonical record without loading a learned model or writing."""
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
            raise FileExistsError(f"canonical E32.4 {label} is already consumed")
    record = _build_registration_record(executor=executor, paths=resolved)
    if resolved.run_attempt_path.exists() or resolved.run_directory.exists():
        raise FileExistsError("canonical E32.4 run state changed during registration")
    if resolved.registration_path.exists():
        raise FileExistsError("canonical E32.4 registration was concurrently written")
    exclusive_json_write(resolved.registration_path, record)
    return record


def seal_registration() -> dict[str, Any]:
    """Write the sole production registration exactly once."""
    return _seal_registration(executor=_ProductionExecutor(), paths=CANONICAL_PATHS)


def _load_registration(*, executor: Any, paths: RunPaths) -> tuple[dict[str, Any], str]:
    resolved = paths.resolved()
    try:
        observed = json.loads(resolved.registration_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("E32.4 registration is missing or malformed") from error
    expected = _build_registration_record(executor=executor, paths=resolved)
    if _canonical_json_bytes(observed) != _canonical_json_bytes(expected):
        raise ValueError("registration differs from exact current source/dependencies")
    return observed, sha256_file(resolved.registration_path)


def _reserve_run_attempt(*, paths: RunPaths, registration_sha256: str) -> str:
    resolved = paths.resolved()
    if (
        resolved.run_attempt_path == resolved.run_directory
        or resolved.run_directory
        in (resolved.run_attempt_path, *resolved.run_attempt_path.parents)
    ):
        raise ValueError("run-attempt sentinel must be external to the run directory")
    payload = {
        "schema_version": f"{EXPERIMENT_ID}_run_attempt",
        "registration_sha256": _validated_sha256(
            registration_sha256, label="registration_sha256"
        ),
        "canonical_run_directory": str(resolved.run_directory),
        "status": "irreversible_attempt_reserved_before_run_directory_or_any_model",
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
    # Rebuilding the production registration repeats clean checkout, complete
    # runtime-tree, installed-source, ancestry, and artifact verification.
    _record, observed_registration = _load_registration(executor=executor, paths=paths)
    if observed_registration != registration_sha256:
        raise ValueError("registration hash changed during E32.4 execution")
    expected_attempt = {
        "schema_version": f"{EXPERIMENT_ID}_run_attempt",
        "registration_sha256": registration_sha256,
        "canonical_run_directory": str(paths.run_directory.resolve()),
        "status": "irreversible_attempt_reserved_before_run_directory_or_any_model",
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
        raise ValueError("run-attempt sentinel changed during E32.4 execution")


def _validate_input_preflight(record: Mapping[str, Any]) -> None:
    if (
        not isinstance(record, Mapping)
        or record.get("schema_version") != "e32_4_all_pool_input_preflight_v1"
        or record.get("verified") is not True
        or record.get("model_calls") != 0
        or tuple(record.get("case_order", ())) != CASE_ORDER
    ):
        raise ValueError("input preflight identity or no-model proof changed")
    cases = record.get("cases")
    if not isinstance(cases, Mapping) or tuple(cases) != tuple(map(str, CASE_ORDER)):
        raise ValueError("input preflight must contain every case in order")
    for case_id in CASE_ORDER:
        case = cases[str(case_id)]
        if not isinstance(case, Mapping) or any(
            case.get(flag) is not True
            for flag in (
                "all_raw_pool_identities_verified",
                "historical_e32_3_raw_rows_excluded",
                "confirmation_excludes_all_new_discovery_raw_rows",
            )
        ):
            raise ValueError(f"case-{case_id} input preflight exclusion proof changed")
        # Production records expose the complete frozen request.  Tests may use
        # the smaller private seam, but any present production field is strict.
        fresh = case.get("fresh_raw_pool_audits")
        requests = case.get("registered_split_requests")
        if (fresh is None) != (requests is None):
            raise ValueError("preflight request and raw identity custody must coexist")
        if fresh is not None:
            if (
                not isinstance(fresh, Mapping)
                or not isinstance(requests, Mapping)
                or tuple(fresh) != MANDATORY_SPLITS
                or tuple(requests) != MANDATORY_SPLITS
            ):
                raise ValueError("preflight fresh split order changed")
            for split in MANDATORY_SPLITS:
                request = asdict(split_request(case_id, split))
                if dict(requests[split]) != request:
                    raise ValueError("preflight registered split request changed")
                audit = fresh[split]
                if (
                    audit.get("case_id") != case_id
                    or audit.get("split") != split
                    or audit.get("candidate_count") != request["candidate_count"]
                    or audit.get("clean_seed") != request["clean_seed"]
                    or audit.get("corrupt_seed") != request["corrupt_seed"]
                    or audit.get("identity_sha256") != request["raw_identity_sha256"]
                    or audit.get("target_identity_sha256")
                    != request["target_identity_sha256"]
                ):
                    raise ValueError("preflight fresh raw/target identity changed")
            if (
                case.get("historical_unique_raw_role_rows")
                != INPUT_PREFLIGHT["old_raw_role_exclusion_unique_rows"][case_id]
            ):
                raise ValueError("historical exclusion count changed")
            if (
                case.get("historical_fresh_role_overlap_count")
                != INPUT_PREFLIGHT["old_new_role_overlap_count"][case_id]
            ):
                raise ValueError("historical/fresh overlap count changed")
            for split, field in (
                ("discovery", "discovery_response_blind_capacity"),
                (
                    "confirmation",
                    "confirmation_response_blind_capacity_after_all_discovery_raw_rows",
                ),
            ):
                capacity = case.get(field)
                expected = _EXPECTED_PREFLIGHT_CAPACITY[case_id][split]
                if (
                    isinstance(capacity, bool)
                    or not isinstance(capacity, int)
                    or capacity != expected
                ):
                    raise ValueError("preflight exact response-blind capacity changed")


def _screen_rows(record: Mapping[str, Any], split: str) -> tuple[tuple[int, ...], ...]:
    return (
        *tuple(tuple(row) for row in record[f"{split}_clean_rows"]),
        *tuple(tuple(row) for row in record[f"{split}_corrupt_rows"]),
    )


def _validate_global_behavior_screen(
    records: Mapping[str, Any], panels: Mapping[int, object]
) -> dict[str, Any]:
    if tuple(records) != tuple(map(str, CASE_ORDER)) or tuple(panels) != CASE_ORDER:
        raise ValueError("behavior screen must contain every case in order")
    audit: dict[str, Any] = {}
    for case_id in CASE_ORDER:
        record = records[str(case_id)]
        if not isinstance(record, Mapping) or record.get("case_id") != case_id:
            raise ValueError("behavior-screen case identity changed")
        discovery = _screen_rows(record, "discovery")
        confirmation = _screen_rows(record, "confirmation")
        expected_d = 2 * split_request(case_id, "discovery").required_count
        expected_c = 2 * split_request(case_id, "confirmation").required_count
        if (
            record.get("behavior_only") is not True
            or len(discovery) != expected_d
            or len(confirmation) != expected_c
            or len(set(discovery)) != len(discovery)
            or len(set(confirmation)) != len(confirmation)
            or set(discovery).intersection(confirmation)
            or record.get("selected_rows_globally_unique_and_response_blind_fresh")
            is not True
        ):
            raise ValueError("behavior screen lost exact fresh unique D/C rows")
        audit[str(case_id)] = {
            "discovery_selected_role_rows": len(discovery),
            "confirmation_selected_role_rows": len(confirmation),
            "cross_split_overlap_count": 0,
            "verified": True,
        }
    return audit


def _score_input_keys(split: str) -> tuple[str, ...]:
    if split == "discovery":
        return _DISCOVERY_SCORE_INPUT_KEYS
    if split == "confirmation":
        return _CONFIRMATION_SCORE_INPUT_KEYS
    raise ValueError("split must be discovery or confirmation")


def _validate_split_bundle(
    bundle: Mapping[str, Any],
    *,
    case_id: int,
    split: str,
    expected_rows: Sequence[Sequence[int]],
    authorization_sha256: str | None,
) -> None:
    if (
        not isinstance(bundle, Mapping)
        or bundle.get("schema_version") != f"e32_4_{split}_acquisition_v1"
        or bundle.get("case_id") != case_id
        or bundle.get("split") != split
    ):
        raise ValueError("split acquisition identity changed")
    rows = tuple(tuple(row) for row in bundle.get("selected_token_rows", ()))
    expected = tuple(tuple(row) for row in expected_rows)
    required = split_request(case_id, split).required_count
    if rows != expected or len(rows) != 2 * required or len(set(rows)) != len(rows):
        raise ValueError("split acquisition lost its behavior-screened rows")
    if bundle.get("panel_identity_sha256") != _panel_identity_sha256(
        case_id=case_id, split=split, selected_token_rows=rows
    ):
        raise ValueError("split panel identity hash changed")
    score_inputs = bundle.get("score_inputs")
    if not isinstance(score_inputs, Mapping) or tuple(
        score_inputs
    ) != _score_input_keys(split):
        raise ValueError("split score_inputs have missing, extra, or reordered fields")
    if not isinstance(bundle.get("query_accounting"), Mapping):
        raise TypeError("split acquisition lacks query accounting")
    observed_authorization = bundle.get("confirmation_authorization_sha256")
    if split == "confirmation":
        if (
            authorization_sha256 is None
            or observed_authorization != authorization_sha256
        ):
            raise ValueError("confirmation artifact lacks its exact authorization")
    elif authorization_sha256 is not None or observed_authorization is not None:
        raise ValueError("discovery cannot carry confirmation authorization")


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
        "schema_version": f"{EXPERIMENT_ID}_split_artifact",
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
        "schema_version": f"{EXPERIMENT_ID}_split_custody",
        "case_id": case_id,
        "split": split,
        "registration_sha256": registration_sha256,
        "run_attempt_sha256": run_attempt_sha256,
        "artifact_filename": filename,
        "artifact_sha256": artifact_sha256,
        "panel_identity_sha256": bundle["panel_identity_sha256"],
        "selected_role_row_count": len(tuple(bundle["selected_token_rows"])),
        "query_accounting": bundle["query_accounting"],
        "confirmation_authorization_sha256": authorization_sha256,
    }
    exclusive_json_write(custody_path, custody)
    return artifact_sha256, sha256_file(custody_path)


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
        or envelope.get("schema_version") != f"{EXPERIMENT_ID}_split_artifact"
        or envelope.get("registration_sha256") != registration_sha256
        or envelope.get("run_attempt_sha256") != run_attempt_sha256
        or envelope.get("case_id") != case_id
        or envelope.get("split") != split
        or not isinstance(envelope.get("acquisition"), dict)
    ):
        raise ValueError("split artifact envelope changed")
    return envelope["acquisition"]


def _validate_frozen_supports(
    *, case_id: int, frozen: Mapping[str, Any], discovery: Mapping[str, Any]
) -> None:
    if (
        not isinstance(frozen, Mapping)
        or frozen.get("schema_version") != "e32_4_frozen_discovery_supports_v1"
        or frozen.get("case_id") != case_id
    ):
        raise ValueError("frozen discovery support identity changed")
    score_inputs = discovery.get("score_inputs")
    if not isinstance(score_inputs, Mapping) or tuple(score_inputs) != (
        _DISCOVERY_SCORE_INPUT_KEYS
    ):
        raise ValueError("discovery score inputs changed before support freezing")
    fine_ids = tuple(score_inputs["induced_fine_edge_ids"])
    if (
        tuple(frozen.get("induced_fine_edge_ids", ())) != fine_ids
        or len(fine_ids) != CASE_GEOMETRY[case_id]["induced_fine_edges"]
        or len(set(fine_ids)) != len(fine_ids)
    ):
        raise ValueError("frozen support universe changed")
    support_size = primary_support_size(case_id)
    if frozen.get("support_size") != support_size:
        raise ValueError("frozen primary support size changed")
    from experiments.e32_interpbench.e32_4_scoring import deterministic_top_support

    selectors = score_inputs["discovery_selector_scores"]
    if (
        not isinstance(selectors, Mapping)
        or tuple(selectors) != DETERMINISTIC_METHOD_ORDER
    ):
        raise ValueError("discovery selector order changed")
    deterministic = tuple(frozen.get("deterministic_supports", ()))
    if len(deterministic) != len(DETERMINISTIC_METHOD_ORDER):
        raise ValueError("frozen deterministic support count changed")
    for record, method in zip(deterministic, DETERMINISTIC_METHOD_ORDER, strict=True):
        expected = deterministic_top_support(
            fine_edge_ids=fine_ids,
            scores=selectors[method],
            support_size=support_size,
        )
        if (
            record.get("method") != method
            or tuple(record.get("fine_edge_ids", ())) != expected
        ):
            raise ValueError("frozen deterministic support differs from D scores")
    curve_sizes = descriptive_support_sizes(case_id)
    curves = tuple(frozen.get("exact_mass_curve_supports", ()))
    if len(curves) != len(curve_sizes):
        raise ValueError("frozen exact-mass curve support count changed")
    for record, (curve_id, size) in zip(curves, curve_sizes.items(), strict=True):
        expected = deterministic_top_support(
            fine_edge_ids=fine_ids,
            scores=selectors[PRIMARY_METHOD],
            support_size=size,
        )
        if (
            record.get("curve_id") != curve_id
            or record.get("method") != PRIMARY_METHOD
            or tuple(record.get("fine_edge_ids", ())) != expected
        ):
            raise ValueError("frozen exact-mass curve differs from D scores")
    random = tuple(frozen.get("random_supports", ()))
    expected_random = deterministic_random_supports(
        case_id=case_id,
        fine_edge_ids=fine_ids,
        support_size=support_size,
    )
    if len(random) != RANDOM_CONTROL_REPLICATES:
        raise ValueError("frozen random support count changed")
    for replicate, (record, expected) in enumerate(
        zip(random, expected_random, strict=True)
    ):
        if (
            record.get("replicate") != replicate
            or tuple(record.get("fine_edge_ids", ())) != expected
        ):
            raise ValueError("frozen random support differs from deterministic replay")
    edge_supports = frozen.get("edge_supports")
    universe_record = frozen.get("edge_universe")
    if (edge_supports is None) != (universe_record is None):
        raise ValueError("frozen edge supports and universe must coexist")
    if edge_supports is not None:
        universe = _universe_from_record(universe_record)
        parallel = (*deterministic, *curves, *random)
        if len(tuple(edge_supports)) != len(parallel):
            raise ValueError("frozen edge-support axis differs from labeled supports")
        for edge_record, labeled in zip(edge_supports, parallel, strict=True):
            indices = tuple(edge_record.get("selected_indices", ()))
            try:
                selected_names = tuple(
                    universe.fine_edge_names[index] for index in indices
                )
            except (IndexError, TypeError) as error:
                raise ValueError(
                    "frozen edge support has invalid full indices"
                ) from error
            if (
                edge_record.get("support_id") != labeled.get("support_id")
                or edge_record.get("method") != labeled.get("method")
                or selected_names != tuple(labeled.get("fine_edge_ids", ()))
            ):
                raise ValueError(
                    "frozen edge selected_indices differ from labeled fine_edge_ids"
                )
    mask_plan = frozen.get("confirmation_mask_plan")
    if not isinstance(mask_plan, Mapping) or tuple(mask_plan) != (
        "schema_version",
        "resample",
        "mean_corrupt_primary_only",
    ):
        raise ValueError("frozen confirmation mask-plan identity changed")
    for name, maximum in (
        (
            "resample",
            EXPECTED_QUERY_CEILINGS["confirmation_formula"][
                "resample_maximum_nonempty_distinct_masks"
            ],
        ),
        (
            "mean_corrupt_primary_only",
            EXPECTED_QUERY_CEILINGS["confirmation_formula"]["mean_corrupt_masks"],
        ),
    ):
        record = mask_plan[name]
        count = record.get("deduped_condition_count")
        condition_ids = tuple(record.get("condition_ids", ()))
        _validated_sha256(
            record.get("fine_masks_sha256"), label=f"{name} fine_masks_sha256"
        )
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= maximum
            or len(condition_ids) != count
            or len(set(condition_ids)) != count
        ):
            raise ValueError("frozen deduplicated confirmation mask plan changed")
        if "fine_masks" in record:
            masks = tuple(
                tuple(bool(value) for value in row) for row in record["fine_masks"]
            )
            if (
                len(masks) != count
                or _json_sha256(masks) != record["fine_masks_sha256"]
            ):
                raise ValueError(
                    "frozen confirmation mask bytes differ from their hash"
                )
    if (
        mask_plan["mean_corrupt_primary_only"]["deduped_condition_count"]
        != EXPECTED_QUERY_CEILINGS["confirmation_formula"]["mean_corrupt_masks"]
    ):
        raise ValueError(
            "mean-corrupt primary-only plan must contain exactly five masks"
        )


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
        "schema_version": f"{EXPERIMENT_ID}_confirmation_authorization",
        "case_id": case_id,
        "registration_sha256": registration_sha256,
        "run_attempt_sha256": run_attempt_sha256,
        "discovery_artifact_sha256": discovery_artifact_sha256,
        "discovery_custody_sha256": discovery_custody_sha256,
        "frozen_supports": dict(frozen_supports),
        "status": "durable_before_any_confirmation_call",
    }


def _nonnegative_accounting_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _accounting_numbers(record: Mapping[str, Any], *, label: str) -> dict[str, int]:
    return {
        name: _nonnegative_accounting_integer(
            record.get(name, 0), label=f"{label}.{name}"
        )
        for name in _TOTAL_ACCOUNTING_KEYS
    }


def _validate_split_query_accounting(
    *,
    case_id: int,
    split: str,
    accounting: Mapping[str, Any],
    authorization: Mapping[str, Any] | None,
) -> dict[str, int]:
    if not isinstance(accounting, Mapping):
        raise TypeError("query accounting must be a mapping")
    required_sections = (
        (
            "behavioral_eligibility",
            "whole_component_atlas",
            "core_singleton_replacement",
            "regular_eap",
            "eap_ig5",
            "total",
        )
        if split == "discovery"
        else (
            "behavioral_eligibility",
            "resample_set_panel",
            "mean_corrupt_primary_sensitivity",
            "total",
        )
    )
    if tuple(accounting) != required_sections:
        raise ValueError("query accounting sections are missing, extra, or reordered")
    stages = {
        name: _accounting_numbers(accounting[name], label=name)
        for name in required_sections[:-1]
    }
    total = _accounting_numbers(accounting["total"], label="total")
    summed = {
        name: sum(stage[name] for stage in stages.values())
        for name in _TOTAL_ACCOUNTING_KEYS
    }
    if summed != total:
        raise ValueError("query accounting total differs from exact section sum")
    request = split_request(case_id, split)
    eligibility = stages["behavioral_eligibility"]
    if eligibility != {
        "physical_forwards": 2,
        "model_examples": 2 * request.candidate_count,
        "gradient_forwards": 0,
        "gradient_examples": 0,
        "backward_passes_inferred_from_pinned_control_flow": 0,
    }:
        raise ValueError("behavioral eligibility accounting changed")
    if split == "discovery":
        expected = dict(EXPECTED_QUERY_CEILINGS[case_id]["discovery"])
        if total != expected:
            raise ValueError("discovery query accounting differs from exact protocol")
        if stages != EXPECTED_DISCOVERY_QUERY_ACCOUNTING[case_id]:
            raise ValueError("discovery per-family exact query accounting changed")
    else:
        if authorization is None:
            raise ValueError("confirmation accounting requires authorization")
        plan = authorization["frozen_supports"]["confirmation_mask_plan"]
        for section, plan_name in (
            ("resample_set_panel", "resample"),
            (
                "mean_corrupt_primary_sensitivity",
                "mean_corrupt_primary_only",
            ),
        ):
            count = plan[plan_name]["deduped_condition_count"]
            stage = accounting[section]
            expected_forwards = 3 + count
            if stage.get("condition_count") != count or stages[section] != {
                "physical_forwards": expected_forwards,
                "model_examples": expected_forwards * request.required_count,
                "gradient_forwards": 0,
                "gradient_examples": 0,
                "backward_passes_inferred_from_pinned_control_flow": 0,
            }:
                raise ValueError(
                    "confirmation query work differs from authorized mask plan"
                )
        ceiling = EXPECTED_QUERY_CEILINGS[case_id]["confirmation"]
        if (
            total["physical_forwards"] > ceiling["maximum_physical_forwards"]
            or total["model_examples"] > ceiling["maximum_model_examples"]
            or any(total[name] for name in _TOTAL_ACCOUNTING_KEYS[2:])
        ):
            raise ValueError("confirmation query ceiling exceeded")
    return total


def _aggregate_query_accounting(
    panel: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    authorizations: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_keys = tuple(
        (case_id, split) for case_id in CASE_ORDER for split in MANDATORY_SPLITS
    )
    if tuple(panel) != expected_keys:
        raise ValueError("query accounting requires the complete ordered panel")
    splits: dict[str, dict[str, dict[str, int]]] = {}
    totals: list[dict[str, int]] = []
    for case_id in CASE_ORDER:
        splits[str(case_id)] = {}
        for split in MANDATORY_SPLITS:
            bundle = panel[(case_id, split)]
            total = _validate_split_query_accounting(
                case_id=case_id,
                split=split,
                accounting=bundle["query_accounting"],
                authorization=(
                    None if split == "discovery" else authorizations[case_id]
                ),
            )
            splits[str(case_id)][split] = total
            totals.append(total)
    overall = {
        name: sum(total[name] for total in totals) for name in _TOTAL_ACCOUNTING_KEYS
    }
    ceiling = EXPECTED_QUERY_CEILINGS["overall"]
    if (
        overall["physical_forwards"] > ceiling["maximum_physical_forwards"]
        or overall["model_examples"] > ceiling["maximum_model_examples"]
        or overall["gradient_forwards"] != ceiling["gradient_forwards"]
        or overall["gradient_examples"] != ceiling["gradient_examples"]
        or overall["backward_passes_inferred_from_pinned_control_flow"]
        != ceiling["backward_passes_inferred_from_pinned_control_flow"]
    ):
        raise ValueError("overall exact query sum exceeds or differs from protocol")
    return {
        "splits": splits,
        "overall": overall,
        "registered_ceilings": _json_safe(EXPECTED_QUERY_CEILINGS),
        "exact_split_sum_verified": True,
        "query_ceiling_pass": True,
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
        f"case{case_id}_{split}.pt"
        for case_id in CASE_ORDER
        for split in MANDATORY_SPLITS
    )
    if tuple(artifact_hashes) != expected_artifacts:
        raise ValueError("score attempt requires the complete ordered panel")
    expected_custody = (
        "input_preflight.json",
        "global_behavior_screen.json",
        *(f"case{case_id}_discovery_custody.json" for case_id in CASE_ORDER),
        *(f"case{case_id}_confirmation_authorization.json" for case_id in CASE_ORDER),
        *(f"case{case_id}_confirmation_custody.json" for case_id in CASE_ORDER),
        "acquisition_custody.json",
    )
    if tuple(custody_hashes) != expected_custody:
        raise ValueError("score attempt requires the exact ordered custody closure")
    for filename, digest in {**artifact_hashes, **custody_hashes}.items():
        _validated_sha256(digest, label=f"hash for {filename}")
        path = paths.run_directory / filename
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError("score-bound artifact differs before reservation")
    payload = {
        "schema_version": f"{EXPERIMENT_ID}_score_attempt",
        "registration_sha256": registration_sha256,
        "run_attempt_sha256": run_attempt_sha256,
        "acquisition_artifact_sha256": dict(artifact_hashes),
        "custody_artifact_sha256": dict(custody_hashes),
        "status": (
            "durable_before_any_e32_4_acquisition_artifact_reload_or_scientific_scoring"
        ),
    }
    path = paths.run_directory / "score_attempt.json"
    exclusive_json_write(path, payload)
    return _ScoreReservation(
        path=path,
        payload_sha256=_json_sha256(payload),
        registration_sha256=registration_sha256,
        run_attempt_sha256=run_attempt_sha256,
        artifact_sha256=dict(artifact_hashes),
        custody_sha256=dict(custody_hashes),
    )


def _verify_score_reservation(reservation: _ScoreReservation) -> None:
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


def _load_score_panel(
    reservation: _ScoreReservation,
) -> dict[tuple[int, str], dict[str, Any]]:
    _verify_score_reservation(reservation)
    panel: dict[tuple[int, str], dict[str, Any]] = {}
    for case_id in CASE_ORDER:
        for split in MANDATORY_SPLITS:
            filename = f"case{case_id}_{split}.pt"
            panel[(case_id, split)] = _load_artifact(
                path=reservation.path.parent / filename,
                expected_sha256=reservation.artifact_sha256[filename],
                registration_sha256=reservation.registration_sha256,
                run_attempt_sha256=reservation.run_attempt_sha256,
                case_id=case_id,
                split=split,
            )
    _verify_score_reservation(reservation)
    return panel


def _scientific_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        if left.dtype != right.dtype or left.shape != right.shape:
            return False
        if left.is_floating_point() or left.is_complex():
            return bool(torch.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True))
        return bool(torch.equal(left, right))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return tuple(left) == tuple(right) and all(
            _scientific_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if not isinstance(left, (tuple, list)) or not isinstance(right, (tuple, list)):
            return False
        return len(left) == len(right) and all(
            _scientific_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, float) and isinstance(right, float):
        return (math.isnan(left) and math.isnan(right)) or left == right
    return left == right


def _json_safe(value: Any) -> Any:
    """Preserve structure while mapping tensors and nonfinite descriptives to JSON."""
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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
        raise FileExistsError("canonical E32.4 run attempt is already consumed")
    if resolved.run_directory.exists():
        raise FileExistsError("canonical E32.4 run directory is already consumed")
    run_attempt_sha256 = _reserve_run_attempt(
        paths=resolved, registration_sha256=registration_sha256
    )
    resolved.run_directory.mkdir(parents=True, exist_ok=False)
    acquisition_attempt = resolved.run_directory / "acquisition_attempt.json"
    exclusive_json_write(
        acquisition_attempt,
        {
            "schema_version": f"{EXPERIMENT_ID}_acquisition_attempt",
            "registration_sha256": registration_sha256,
            "run_attempt_sha256": run_attempt_sha256,
            "phase_order": (
                "all_raw_pool_input_preflight",
                "global_behavior_screen",
                "all_discoveries",
                "all_support_authorizations",
                "global_authorization_reverification",
                "all_confirmations",
                "query_accounting_and_acquisition_custody",
                "score_attempt",
                "torch_artifact_reload_and_exact_score_replay",
            ),
            "status": "durable_before_first_raw_input_generation_or_model_call",
        },
    )

    artifact_hashes: dict[str, str] = {}
    custody_hashes: dict[str, str] = {}
    bundles: dict[tuple[int, str], Mapping[str, Any]] = {}
    authorizations: dict[int, dict[str, Any]] = {}
    score_reservation: _ScoreReservation | None = None
    try:
        preflight_record, private_state = executor.preflight_all_input_pools()
        _validate_input_preflight(preflight_record)
        preflight_path = resolved.run_directory / "input_preflight.json"
        exclusive_json_write(
            preflight_path,
            {
                **dict(preflight_record),
                "registration_sha256": registration_sha256,
                "run_attempt_sha256": run_attempt_sha256,
                "status": "durable_before_any_learned_model_load",
            },
        )
        custody_hashes[preflight_path.name] = sha256_file(preflight_path)

        # A full dependency/runtime replay immediately precedes behavior models.
        _verify_live_custody(
            executor=executor,
            paths=resolved,
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
        )
        screen_records, case_panels = executor.screen_all_fresh_panels(private_state)
        screen_audit = _validate_global_behavior_screen(screen_records, case_panels)
        screen_path = resolved.run_directory / "global_behavior_screen.json"
        exclusive_json_write(
            screen_path,
            {
                "schema_version": f"{EXPERIMENT_ID}_global_behavior_screen",
                "registration_sha256": registration_sha256,
                "run_attempt_sha256": run_attempt_sha256,
                "cases": screen_records,
                "audit": screen_audit,
                "status": "complete_before_any_intervention_model_call",
            },
        )
        custody_hashes[screen_path.name] = sha256_file(screen_path)

        # A second replay immediately precedes the discovery intervention phase.
        _verify_live_custody(
            executor=executor,
            paths=resolved,
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
        )
        for case_id in CASE_ORDER:
            discovery = executor.acquire_discovery(case_id, case_panels[case_id])
            _validate_split_query_accounting(
                case_id=case_id,
                split="discovery",
                accounting=discovery["query_accounting"],
                authorization=None,
            )
            artifact, custody = _write_split_artifact(
                paths=resolved,
                registration_sha256=registration_sha256,
                run_attempt_sha256=run_attempt_sha256,
                case_id=case_id,
                split="discovery",
                bundle=discovery,
                expected_rows=_screen_rows(screen_records[str(case_id)], "discovery"),
                authorization_sha256=None,
            )
            artifact_name = f"case{case_id}_discovery.pt"
            custody_name = f"case{case_id}_discovery_custody.json"
            artifact_hashes[artifact_name] = artifact
            custody_hashes[custody_name] = custody
            bundles[(case_id, "discovery")] = discovery

        # Freeze from the in-memory D objects that were just written.  No .pt
        # artifact is reopened before the score-attempt sentinel.
        for case_id in CASE_ORDER:
            discovery = bundles[(case_id, "discovery")]
            frozen = executor.freeze_discovery_supports(case_id, discovery)
            _validate_frozen_supports(
                case_id=case_id, frozen=frozen, discovery=discovery
            )
            artifact_name = f"case{case_id}_discovery.pt"
            custody_name = f"case{case_id}_discovery_custody.json"
            authorization = _authorization_record(
                case_id=case_id,
                registration_sha256=registration_sha256,
                run_attempt_sha256=run_attempt_sha256,
                discovery_artifact_sha256=artifact_hashes[artifact_name],
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

        # Recompute all 6 deterministic, 2 curve, and 256 random supports.
        recomputed_supports: dict[int, Mapping[str, Any]] = {}
        for case_id in CASE_ORDER:
            discovery = bundles[(case_id, "discovery")]
            recomputed = executor.freeze_discovery_supports(case_id, discovery)
            _validate_frozen_supports(
                case_id=case_id, frozen=recomputed, discovery=discovery
            )
            recomputed_supports[case_id] = recomputed

        # Replay the complete runtime closure first.  Then re-open every durable
        # authorization and its bound D files as the final barrier immediately
        # before the first confirmation intervention.
        _verify_live_custody(
            executor=executor,
            paths=resolved,
            registration_sha256=registration_sha256,
            run_attempt_sha256=run_attempt_sha256,
        )
        for case_id in CASE_ORDER:
            recomputed = recomputed_supports[case_id]
            authorization = authorizations[case_id]
            path = (
                resolved.run_directory
                / f"case{case_id}_confirmation_authorization.json"
            )
            artifact_name = f"case{case_id}_discovery.pt"
            custody_name = f"case{case_id}_discovery_custody.json"
            if (
                _canonical_json_bytes(recomputed)
                != _canonical_json_bytes(authorization["frozen_supports"])
                or not path.is_file()
                or sha256_file(path) != authorization["authorization_sha256"]
                or sha256_file(resolved.run_directory / artifact_name)
                != authorization["discovery_artifact_sha256"]
                or sha256_file(resolved.run_directory / custody_name)
                != authorization["discovery_custody_sha256"]
            ):
                raise ValueError("global confirmation authorization barrier changed")
        for case_id in CASE_ORDER:
            authorization = authorizations[case_id]
            confirmation = executor.acquire_confirmation(
                case_id, case_panels[case_id], authorization
            )
            _validate_split_query_accounting(
                case_id=case_id,
                split="confirmation",
                accounting=confirmation["query_accounting"],
                authorization=authorization,
            )
            artifact, custody = _write_split_artifact(
                paths=resolved,
                registration_sha256=registration_sha256,
                run_attempt_sha256=run_attempt_sha256,
                case_id=case_id,
                split="confirmation",
                bundle=confirmation,
                expected_rows=_screen_rows(
                    screen_records[str(case_id)], "confirmation"
                ),
                authorization_sha256=authorization["authorization_sha256"],
            )
            artifact_name = f"case{case_id}_confirmation.pt"
            custody_name = f"case{case_id}_confirmation_custody.json"
            artifact_hashes[artifact_name] = artifact
            custody_hashes[custody_name] = custody
            bundles[(case_id, "confirmation")] = confirmation

        ordered_bundles = {
            (case_id, split): bundles[(case_id, split)]
            for case_id in CASE_ORDER
            for split in MANDATORY_SPLITS
        }
        actual_accounting = _aggregate_query_accounting(
            ordered_bundles, authorizations=authorizations
        )
        ordered_artifacts = {
            f"case{case_id}_{split}.pt": artifact_hashes[f"case{case_id}_{split}.pt"]
            for case_id in CASE_ORDER
            for split in MANDATORY_SPLITS
        }
        aggregate_path = resolved.run_directory / "acquisition_custody.json"
        exclusive_json_write(
            aggregate_path,
            {
                "schema_version": f"{EXPERIMENT_ID}_acquisition_custody",
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
        # This is the first reload of a sealed E32.4 acquisition artifact.
        score_panel = _load_score_panel(score_reservation)
        scientific = executor.score_complete_panel(score_panel)
        replay = executor.score_complete_panel(score_panel)
        if not _scientific_equal(scientific, replay):
            raise RuntimeError("exact pure score replay differs from the first score")
        _verify_score_reservation(score_reservation)
        score_artifact_path = resolved.run_directory / "scientific_score.pt"
        _exclusive_torch_write(score_artifact_path, scientific)
        score_artifact_sha256 = sha256_file(score_artifact_path)
        _verify_score_reservation(score_reservation)
        summary = {
            **_json_safe(scientific),
            "schema_version": f"{EXPERIMENT_ID}_summary",
            "registration_sha256": registration_sha256,
            "run_attempt_sha256": run_attempt_sha256,
            "acquisition_attempt_sha256": sha256_file(acquisition_attempt),
            "score_attempt_sha256": sha256_file(score_reservation.path),
            "scientific_score_artifact_sha256": score_artifact_sha256,
            "acquisition_artifact_sha256": ordered_artifacts,
            "custody_artifact_sha256": custody_hashes,
            "actual_query_accounting": actual_accounting,
            "query_ceiling_pass": actual_accounting["query_ceiling_pass"],
            "nonfinite_descriptives_encoded_as_json_null": True,
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
                    "schema_version": f"{EXPERIMENT_ID}_failure",
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


def _zero_gradient_stage(
    *, physical_forwards: int, model_examples: int
) -> dict[str, int]:
    return {
        "physical_forwards": int(physical_forwards),
        "model_examples": int(model_examples),
        "gradient_forwards": 0,
        "gradient_examples": 0,
        "backward_passes_inferred_from_pinned_control_flow": 0,
    }


def _eligibility_accounting(eligible: Any) -> dict[str, int]:
    return _zero_gradient_stage(
        physical_forwards=int(eligible.eligibility_physical_forwards),
        model_examples=int(eligible.eligibility_model_examples),
    )


def _eap_accounting(run: Any) -> dict[str, int]:
    observed = run.observed_query_accounting
    gradients = int(observed["grad_enabled_forwards"])
    return {
        "physical_forwards": int(observed["physical_forwards"]),
        "model_examples": int(observed["model_examples"]),
        "gradient_forwards": gradients,
        "gradient_examples": int(run.query_accounting.gradient_examples),
        "backward_passes_inferred_from_pinned_control_flow": int(
            observed["backward_passes_inferred_from_pinned_control_flow"]
        ),
    }


def _sum_accounting_stages(stages: Sequence[Mapping[str, int]]) -> dict[str, int]:
    return {
        name: sum(int(stage[name]) for stage in stages)
        for name in _TOTAL_ACCOUNTING_KEYS
    }


def _live_discovery_bundle(*, case_id: int, eligible: Any) -> dict[str, Any]:
    """Acquire the registered D-only atlas, exact core, EAP, and EAP-IG5."""
    from experiments.e32_interpbench.e32_3_acquisition import (
        acquire_whole_component_atlas,
        build_cohort_component_runner,
        whole_component_acquisition_to_dict,
    )
    from experiments.e32_interpbench.e32_3_scoring import raw_pair_interaction_mass
    from experiments.e32_interpbench.e32_4_eap import (
        generic_eap_run_to_dict,
        run_generic_eap,
    )
    from experiments.e32_interpbench.e32_4_edge_sets import (
        build_induced_edge_universe,
        core_singleton_acquisition_to_dict,
        run_core_singleton_replacement,
    )
    from experiments.e32_interpbench.e32_4_scoring import (
        build_selector_score_vectors,
    )

    if case_id not in CASE_ORDER or not hasattr(eligible, "pair"):
        raise ValueError("live discovery is outside the registered panel")
    manifest = CASE_MANIFESTS[case_id]
    pair = eligible.pair
    request = split_request(case_id, "discovery")
    if len(pair.clean_inputs) != request.required_count:
        raise ValueError("live discovery pair differs from registered batch size")
    case = load_verified_official_case(case_id)
    component_ids, whole_edges = case_universe(case_id)

    whole_runner = build_cohort_component_runner(
        case_id=case_id,
        model=load_verified_model(case_id, device="cpu"),
        condition_batch_size=32,
    )
    whole = acquire_whole_component_atlas(
        case_id=case_id,
        split="discovery",
        runner=whole_runner,
        pair=pair,
        discovery_smoke_condition_count=8,
        discovery_smoke_batch_size=8,
    )
    if whole.component_ids != component_ids or whole.whole_edges != whole_edges:
        raise RuntimeError("whole atlas differs from the registered internal DAG")

    regular_eap = run_generic_eap(
        case_id=case_id,
        n_layers=manifest.n_layers,
        n_heads=manifest.n_heads,
        n_ctx=manifest.n_ctx,
        d_vocab_out=manifest.d_vocab_out,
        model=load_verified_model(case_id, device="cpu"),
        case=case,
        eligible_pair=eligible,
        method="eap",
        device="cpu",
    )
    universe = build_induced_edge_universe(
        regular_eap.grouping,
        selected_nodes=CORE_NODE_SUPPORTS[case_id],
    )
    if (
        universe.whole_edges != whole_edges
        or universe.fine_edge_count != CASE_GEOMETRY[case_id]["fine_edges"]
        or universe.induced_fine_edge_count
        != CASE_GEOMETRY[case_id]["induced_fine_edges"]
    ):
        raise RuntimeError("EAP-derived edge universe differs from registered geometry")

    singleton = run_core_singleton_replacement(
        case_id=case_id,
        n_layers=manifest.n_layers,
        n_heads=manifest.n_heads,
        n_ctx=manifest.n_ctx,
        d_vocab_out=manifest.d_vocab_out,
        model=load_verified_model(case_id, device="cpu"),
        case=case,
        eligible_pair=eligible,
        universe=universe,
        device="cpu",
    )
    integrated_eap = run_generic_eap(
        case_id=case_id,
        n_layers=manifest.n_layers,
        n_heads=manifest.n_heads,
        n_ctx=manifest.n_ctx,
        d_vocab_out=manifest.d_vocab_out,
        model=load_verified_model(case_id, device="cpu"),
        case=case,
        eligible_pair=eligible,
        method="eap_ig",
        device="cpu",
    )
    if (
        regular_eap.internal_fine_edge_names != universe.fine_edge_names
        or integrated_eap.internal_fine_edge_names != universe.fine_edge_names
        or regular_eap.fine_to_whole_indices != universe.fine_to_whole_indices
        or integrated_eap.fine_to_whole_indices != universe.fine_to_whole_indices
    ):
        raise RuntimeError("regular EAP, EAP-IG5, and exact fine axes differ")
    if not torch.equal(
        singleton.clean_baseline_output, whole.clean_output
    ) or not torch.equal(singleton.corrupt_baseline_output, whole.corrupt_output):
        raise RuntimeError("D exact and whole-component endpoint baselines differ")

    raw_whole_mass = raw_pair_interaction_mass(whole.pair_interactions)
    selector_scores = build_selector_score_vectors(
        component_ids=component_ids,
        whole_edges=whole_edges,
        fine_to_whole_indices=universe.fine_to_whole_indices,
        induced_fine_indices=universe.induced_fine_indices,
        clean_output=singleton.clean_baseline_output,
        corrupt_output=singleton.corrupt_baseline_output,
        corrupt_into_clean_outputs=singleton.corrupt_into_clean_outputs,
        clean_into_corrupt_outputs=singleton.clean_into_corrupt_outputs,
        eap_fine_scores=regular_eap.internal_fine_scores,
        eap_ig5_fine_scores=integrated_eap.internal_fine_scores,
        raw_whole_mass=raw_whole_mass,
    )
    induced_ids = tuple(
        universe.fine_edge_names[index] for index in universe.induced_fine_indices
    )

    whole_stage = _zero_gradient_stage(**whole.query_accounting["total"])
    singleton_stage = _zero_gradient_stage(
        physical_forwards=singleton.query_accounting.observed_physical_forwards,
        model_examples=singleton.query_accounting.observed_model_examples,
    )
    stages = {
        "behavioral_eligibility": _eligibility_accounting(eligible),
        "whole_component_atlas": whole_stage,
        "core_singleton_replacement": singleton_stage,
        "regular_eap": _eap_accounting(regular_eap),
        "eap_ig5": _eap_accounting(integrated_eap),
    }
    query_accounting = {
        **stages,
        "total": _sum_accounting_stages(tuple(stages.values())),
    }
    selected_rows = _pair_rows(pair, label="discovery acquisition")
    return {
        "schema_version": "e32_4_discovery_acquisition_v1",
        "case_id": case_id,
        "split": "discovery",
        "selected_token_rows": selected_rows,
        "panel_identity_sha256": _panel_identity_sha256(
            case_id=case_id,
            split="discovery",
            selected_token_rows=selected_rows,
        ),
        "confirmation_authorization_sha256": None,
        "behavioral_eligibility": _eligibility_record(eligible),
        "score_inputs": {
            "induced_fine_edge_ids": induced_ids,
            "discovery_selector_scores": selector_scores,
        },
        "query_accounting": query_accounting,
        "scientific_payload": {
            "component_ids": component_ids,
            "whole_edges": whole_edges,
            "edge_universe": core_singleton_acquisition_to_dict(singleton)["universe"],
            "whole_component": whole_component_acquisition_to_dict(whole),
            "raw_whole_mass": raw_whole_mass,
            "core_singleton": core_singleton_acquisition_to_dict(singleton),
            "regular_eap": generic_eap_run_to_dict(regular_eap),
            "eap_ig5": generic_eap_run_to_dict(integrated_eap),
        },
    }


def _universe_from_record(record: Mapping[str, Any]) -> Any:
    from experiments.e32_interpbench.e32_4_edge_sets import InducedEdgeUniverse

    return InducedEdgeUniverse(
        case_id=int(record["case_id"]),
        selected_nodes=tuple(record["selected_nodes"]),
        whole_edges=tuple(tuple(edge) for edge in record["whole_edges"]),
        fine_edge_names=tuple(record["fine_edge_names"]),
        fine_to_whole_indices=tuple(record["fine_to_whole_indices"]),
        induced_whole_indices=tuple(record["induced_whole_indices"]),
        induced_fine_indices=tuple(record["induced_fine_indices"]),
    )


def _frozen_plan_record(plan: Any) -> dict[str, Any]:
    masks = tuple(
        tuple(bool(value) for value in row) for row in plan.fine_masks.tolist()
    )
    return {
        "deduped_condition_count": int(plan.condition_count),
        "condition_ids": tuple(plan.condition_ids),
        "fine_masks": masks,
        "fine_masks_sha256": _json_sha256(masks),
        "all_internal_condition_index": plan.all_internal_condition_index,
        "induced_condition_index": plan.induced_condition_index,
        "outside_induced_condition_index": plan.outside_induced_condition_index,
        "primary_support_id": plan.primary_support_id,
        "primary_within_induced_complement_condition_index": (
            plan.primary_within_induced_complement_condition_index
        ),
        "support_references": tuple(
            asdict(reference) for reference in plan.support_references
        ),
    }


def _freeze_production_discovery_supports(
    *, case_id: int, discovery: Mapping[str, Any]
) -> dict[str, Any]:
    from experiments.e32_interpbench.e32_4_edge_sets import (
        FrozenEdgeSupport,
        build_edge_condition_plan,
    )
    from experiments.e32_interpbench.e32_4_scoring import deterministic_top_support

    if discovery.get("case_id") != case_id or discovery.get("split") != "discovery":
        raise ValueError("support freezing requires the matching D bundle")
    score_inputs = discovery["score_inputs"]
    fine_ids = tuple(score_inputs["induced_fine_edge_ids"])
    selectors = score_inputs["discovery_selector_scores"]
    universe = _universe_from_record(discovery["scientific_payload"]["edge_universe"])
    if fine_ids != tuple(
        universe.fine_edge_names[index] for index in universe.induced_fine_indices
    ):
        raise ValueError("D score axis differs from its frozen induced universe")

    deterministic_records: list[dict[str, Any]] = []
    curve_records: list[dict[str, Any]] = []
    random_records: list[dict[str, Any]] = []
    edge_supports: list[Any] = []
    primary_size = primary_support_size(case_id)

    def freeze(
        *,
        support_id: str,
        method: str,
        selected_names: Sequence[str],
        kind: str,
        replicate: int | None = None,
    ) -> Any:
        selected_set = set(selected_names)
        indices = tuple(
            index
            for index, name in enumerate(universe.fine_edge_names)
            if name in selected_set
        )
        if len(indices) != len(selected_names):
            raise ValueError("selected D support escaped the full fine universe")
        support = FrozenEdgeSupport(
            support_id=support_id,
            resolution="fine",
            method=method,
            support_size=len(indices),
            selected_indices=indices,
            kind=kind,
            replicate_index=replicate,
        )
        edge_supports.append(support)
        return support

    for method in DETERMINISTIC_METHOD_ORDER:
        names = deterministic_top_support(
            fine_edge_ids=fine_ids,
            scores=selectors[method],
            support_size=primary_size,
        )
        support = freeze(
            support_id=method,
            method=method,
            selected_names=names,
            kind="deterministic",
        )
        deterministic_records.append({**asdict(support), "fine_edge_ids": tuple(names)})
    for curve_id, size in descriptive_support_sizes(case_id).items():
        names = deterministic_top_support(
            fine_edge_ids=fine_ids,
            scores=selectors[PRIMARY_METHOD],
            support_size=size,
        )
        support = freeze(
            support_id=f"{PRIMARY_METHOD}:{curve_id}",
            method=PRIMARY_METHOD,
            selected_names=names,
            kind="deterministic",
        )
        curve_records.append(
            {**asdict(support), "curve_id": curve_id, "fine_edge_ids": tuple(names)}
        )
    for replicate, names in enumerate(
        deterministic_random_supports(
            case_id=case_id,
            fine_edge_ids=fine_ids,
            support_size=primary_size,
        )
    ):
        support = freeze(
            support_id=f"random:{replicate:03d}",
            method="random_control",
            selected_names=names,
            kind="random",
            replicate=replicate,
        )
        random_records.append(
            {**asdict(support), "replicate": replicate, "fine_edge_ids": tuple(names)}
        )
    full_plan = build_edge_condition_plan(
        universe,
        tuple(edge_supports),
        primary_support_id=PRIMARY_METHOD,
    )
    primary_support = edge_supports[0]
    mean_plan = build_edge_condition_plan(universe, (primary_support,))
    return {
        "schema_version": "e32_4_frozen_discovery_supports_v1",
        "case_id": case_id,
        "support_size": primary_size,
        "induced_fine_edge_ids": fine_ids,
        "discovery_selector_scores_sha256": _json_sha256(
            {
                method: torch.as_tensor(selectors[method]).tolist()
                for method in DETERMINISTIC_METHOD_ORDER
            }
        ),
        "edge_universe": dict(discovery["scientific_payload"]["edge_universe"]),
        "edge_supports": tuple(asdict(support) for support in edge_supports),
        "deterministic_supports": tuple(deterministic_records),
        "exact_mass_curve_supports": tuple(curve_records),
        "random_supports": tuple(random_records),
        "confirmation_mask_plan": {
            "schema_version": "e32_4_deduplicated_confirmation_mask_plan_v1",
            "resample": _frozen_plan_record(full_plan),
            "mean_corrupt_primary_only": _frozen_plan_record(mean_plan),
        },
    }


def _supports_and_plans_from_authorization(
    *, case_id: int, authorization: Mapping[str, Any]
) -> tuple[Any, tuple[Any, ...], Any, Any]:
    from experiments.e32_interpbench.e32_4_edge_sets import (
        FrozenEdgeSupport,
        build_edge_condition_plan,
    )

    if (
        authorization.get("case_id") != case_id
        or authorization.get("status") != "durable_before_any_confirmation_call"
    ):
        raise ValueError("confirmation authorization identity changed")
    frozen = authorization.get("frozen_supports")
    if not isinstance(frozen, Mapping):
        raise TypeError("confirmation authorization lacks frozen D supports")
    universe = _universe_from_record(frozen["edge_universe"])
    supports = tuple(
        FrozenEdgeSupport(**dict(record)) for record in frozen["edge_supports"]
    )
    labeled = (
        *tuple(frozen["deterministic_supports"]),
        *tuple(frozen["exact_mass_curve_supports"]),
        *tuple(frozen["random_supports"]),
    )
    if len(supports) != len(labeled):
        raise ValueError("authorized edge-support and label axes differ")
    for support, record in zip(supports, labeled, strict=True):
        selected_names = tuple(
            universe.fine_edge_names[index] for index in support.selected_indices
        )
        if (
            support.support_id != record["support_id"]
            or support.method != record["method"]
            or selected_names != tuple(record["fine_edge_ids"])
        ):
            raise ValueError(
                "authorized selected_indices differ from labeled fine_edge_ids"
            )
    full_plan = build_edge_condition_plan(
        universe, supports, primary_support_id=PRIMARY_METHOD
    )
    primary = next(
        support for support in supports if support.support_id == PRIMARY_METHOD
    )
    mean_plan = build_edge_condition_plan(universe, (primary,))
    registered = frozen["confirmation_mask_plan"]
    if _canonical_json_bytes(_frozen_plan_record(full_plan)) != _canonical_json_bytes(
        registered["resample"]
    ) or _canonical_json_bytes(_frozen_plan_record(mean_plan)) != _canonical_json_bytes(
        registered["mean_corrupt_primary_only"]
    ):
        raise ValueError("recomputed C mask plans differ from durable authorization")
    return universe, supports, full_plan, mean_plan


def _set_output_record(
    *,
    support_record: Mapping[str, Any],
    selected_output: torch.Tensor,
    complement_output: torch.Tensor,
) -> dict[str, Any]:
    return {
        "selected_fine_edge_ids": tuple(support_record["fine_edge_ids"]),
        "selected_output": selected_output,
        "complement_output": complement_output,
    }


def _edge_set_accounting(value: Any) -> dict[str, int]:
    return _zero_gradient_stage(
        physical_forwards=value.query_accounting.observed_physical_forwards,
        model_examples=value.query_accounting.observed_model_examples,
    )


def _live_confirmation_bundle(
    *,
    case_id: int,
    eligible: Any,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Acquire resample full-panel and mean-corrupt primary-only C outcomes."""
    from experiments.e32_interpbench.e32_4_edge_sets import (
        EdgeAblation,
        edge_set_acquisition_to_dict,
        run_edge_set_replacement,
    )

    if case_id not in CASE_ORDER or not hasattr(eligible, "pair"):
        raise ValueError("live confirmation is outside the registered panel")
    universe, supports, resample_plan, mean_plan = (
        _supports_and_plans_from_authorization(
            case_id=case_id, authorization=authorization
        )
    )
    manifest = CASE_MANIFESTS[case_id]
    case = load_verified_official_case(case_id)
    resample = run_edge_set_replacement(
        case_id=case_id,
        n_layers=manifest.n_layers,
        n_heads=manifest.n_heads,
        n_ctx=manifest.n_ctx,
        d_vocab_out=manifest.d_vocab_out,
        model=load_verified_model(case_id, device="cpu"),
        case=case,
        eligible_pair=eligible,
        universe=universe,
        plan=resample_plan,
        ablation=EdgeAblation.RESAMPLE,
        device="cpu",
    )
    mean = run_edge_set_replacement(
        case_id=case_id,
        n_layers=manifest.n_layers,
        n_heads=manifest.n_heads,
        n_ctx=manifest.n_ctx,
        d_vocab_out=manifest.d_vocab_out,
        model=load_verified_model(case_id, device="cpu"),
        case=case,
        eligible_pair=eligible,
        universe=universe,
        plan=mean_plan,
        ablation=EdgeAblation.BATCH_TOKENWISE_MEAN_CORRUPT,
        device="cpu",
    )
    if not torch.equal(
        resample.clean_baseline_output, mean.clean_baseline_output
    ) or not torch.equal(
        resample.corrupt_baseline_output, mean.corrupt_baseline_output
    ):
        raise RuntimeError("resample and mean-corrupt C endpoint baselines differ")
    if resample.primary_within_induced_complement_output is None:
        raise RuntimeError("resample panel lacks authorized I-minus-primary output")

    frozen = authorization["frozen_supports"]
    deterministic = tuple(frozen["deterministic_supports"])
    curves = tuple(frozen["exact_mass_curve_supports"])
    random = tuple(frozen["random_supports"])
    if len(supports) != len(deterministic) + len(curves) + len(random):
        raise RuntimeError("resample support output axis differs from authorization")
    output_by_id = {
        support.support_id: (
            resample.support_selected_outputs[index],
            resample.support_complement_outputs[index],
        )
        for index, support in enumerate(supports)
    }

    def output(record: Mapping[str, Any]) -> dict[str, Any]:
        selected, complement = output_by_id[record["support_id"]]
        return _set_output_record(
            support_record=record,
            selected_output=selected,
            complement_output=complement,
        )

    deterministic_outputs = {
        method: output(record)
        for method, record in zip(
            DETERMINISTIC_METHOD_ORDER, deterministic, strict=True
        )
    }
    curve_outputs = {record["curve_id"]: output(record) for record in curves}
    random_outputs = tuple(
        {"replicate": replicate, **output(record)}
        for replicate, record in enumerate(random)
    )
    mean_primary_record = deterministic[0]
    mean_primary_output = _set_output_record(
        support_record=mean_primary_record,
        selected_output=mean.support_selected_outputs[0],
        complement_output=mean.support_complement_outputs[0],
    )
    pair = eligible.pair
    score_inputs = {
        "clean_output": resample.clean_baseline_output,
        "actual_corrupt_output": resample.corrupt_baseline_output,
        "clean_target_indices": pair.clean_targets.argmax(dim=-1)[:, 1:].cpu(),
        "resample_all_internal_output": resample.all_internal_output,
        "resample_induced_core_output": resample.induced_output,
        "resample_outside_core_output": resample.outside_induced_output,
        "resample_deterministic_outputs": deterministic_outputs,
        "resample_exact_mass_curve_outputs": curve_outputs,
        "resample_random_outputs": random_outputs,
        "resample_within_core_primary_complement_output": (
            resample.primary_within_induced_complement_output
        ),
        "mean_all_internal_output": mean.all_internal_output,
        "mean_induced_core_output": mean.induced_output,
        "mean_outside_core_output": mean.outside_induced_output,
        "mean_primary_output": mean_primary_output["selected_output"],
        "mean_primary_complement_output": mean_primary_output["complement_output"],
    }
    resample_stage = {
        "condition_count": int(resample.query_accounting.condition_count),
        **_edge_set_accounting(resample),
    }
    mean_stage = {
        "condition_count": int(mean.query_accounting.condition_count),
        **_edge_set_accounting(mean),
    }
    stages = {
        "behavioral_eligibility": _eligibility_accounting(eligible),
        "resample_set_panel": resample_stage,
        "mean_corrupt_primary_sensitivity": mean_stage,
    }
    query_accounting = {
        **stages,
        "total": _sum_accounting_stages(tuple(stages.values())),
    }
    selected_rows = _pair_rows(pair, label="confirmation acquisition")
    return {
        "schema_version": "e32_4_confirmation_acquisition_v1",
        "case_id": case_id,
        "split": "confirmation",
        "selected_token_rows": selected_rows,
        "panel_identity_sha256": _panel_identity_sha256(
            case_id=case_id,
            split="confirmation",
            selected_token_rows=selected_rows,
        ),
        "confirmation_authorization_sha256": authorization["authorization_sha256"],
        "behavioral_eligibility": _eligibility_record(eligible),
        "score_inputs": score_inputs,
        "query_accounting": query_accounting,
        "scientific_payload": {
            "panel": {
                "clean_inputs": pair.clean_inputs.detach().cpu().clone(),
                "corrupt_inputs": pair.corrupt_inputs.detach().cpu().clone(),
                "clean_targets": pair.clean_targets.detach().cpu().clone(),
                "corrupt_targets": pair.corrupt_targets.detach().cpu().clone(),
            },
            "resample_set_panel": edge_set_acquisition_to_dict(resample),
            "mean_corrupt_primary_sensitivity": edge_set_acquisition_to_dict(mean),
            "resample_mean_selected_rows_identical": True,
            "resample_mean_endpoint_baselines_identical": True,
        },
    }


def _transitive_non_edges(
    *,
    nodes: Sequence[str],
    ground_truth_edges: Sequence[tuple[str, str]],
    candidate_edges: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    adjacency = {node: set() for node in nodes}
    for source, target in ground_truth_edges:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set())
    reachable: set[tuple[str, str]] = set()
    for source, immediate_targets in adjacency.items():
        frontier = list(immediate_targets)
        seen: set[str] = set()
        while frontier:
            target = frontier.pop()
            if target in seen:
                continue
            seen.add(target)
            reachable.add((source, target))
            frontier.extend(adjacency.get(target, ()))
    direct = {tuple(edge) for edge in ground_truth_edges}
    return tuple(
        tuple(edge)
        for edge in candidate_edges
        if tuple(edge) in reachable and tuple(edge) not in direct
    )


def _recompute_internal_eap_scores(
    record: Mapping[str, Any],
    *,
    fine_edge_names: Sequence[str],
    label: str,
) -> tuple[float, ...]:
    """Rebuild the internal score axis from raw official fine-edge records."""
    names = tuple(fine_edge_names)
    if (
        not names
        or len(names) != len(set(names))
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError(f"{label} canonical fine-edge names are malformed")
    raw_by_name: dict[str, float] = {}
    for raw in record.get("all_fine_scores", ()):
        if not isinstance(raw, Mapping):
            raise TypeError(f"{label} raw fine scores must be mappings")
        name = raw.get("fine_edge_name")
        score = raw.get("signed_score")
        if (
            not isinstance(name, str)
            or not name
            or name in raw_by_name
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError(f"{label} raw fine scores are malformed or duplicate")
        raw_by_name[name] = float(score)
    if any(name not in raw_by_name for name in names):
        raise ValueError(f"{label} raw fine scores do not cover canonical internal U")
    recomputed = tuple(raw_by_name[name] for name in names)
    stored_names = tuple(record.get("internal_fine_edge_names", ()))
    stored_scores = tuple(record.get("internal_fine_scores", ()))
    if stored_names != names or stored_scores != recomputed:
        raise ValueError(f"{label} derived internal scores differ from raw records")
    return recomputed


def _recompute_discovery_score_inputs(
    discovery: Mapping[str, Any],
) -> dict[str, Any]:
    from experiments.e32_interpbench.e32_3_scoring import raw_pair_interaction_mass
    from experiments.e32_interpbench.e32_4_scoring import (
        build_selector_score_vectors,
    )

    payload = discovery["scientific_payload"]
    whole = payload["whole_component"]
    singleton = payload["core_singleton"]
    universe = payload["edge_universe"]
    regular_eap = payload["regular_eap"]
    integrated_eap = payload["eap_ig5"]
    rows = (
        *token_rows(whole["clean_inputs"], label="score-stage D clean inputs"),
        *token_rows(whole["corrupt_inputs"], label="score-stage D corrupt inputs"),
    )
    case_id = int(discovery["case_id"])
    if rows != tuple(
        tuple(row) for row in discovery["selected_token_rows"]
    ) or discovery["panel_identity_sha256"] != _panel_identity_sha256(
        case_id=case_id, split="discovery", selected_token_rows=rows
    ):
        raise ValueError("raw D panel tensors differ from selected-row custody")
    raw_mass = raw_pair_interaction_mass(whole["pair_interactions"])
    if not torch.equal(torch.as_tensor(payload["raw_whole_mass"]), raw_mass):
        raise ValueError("stored D raw whole mass differs from raw pair interactions")
    fine_names = tuple(universe["fine_edge_names"])
    regular_eap_scores = _recompute_internal_eap_scores(
        regular_eap,
        fine_edge_names=fine_names,
        label="regular EAP",
    )
    integrated_eap_scores = _recompute_internal_eap_scores(
        integrated_eap,
        fine_edge_names=fine_names,
        label="EAP-IG5",
    )
    selectors = build_selector_score_vectors(
        component_ids=tuple(payload["component_ids"]),
        whole_edges=tuple(tuple(edge) for edge in payload["whole_edges"]),
        fine_to_whole_indices=tuple(universe["fine_to_whole_indices"]),
        induced_fine_indices=tuple(universe["induced_fine_indices"]),
        clean_output=singleton["clean_baseline_output"],
        corrupt_output=singleton["corrupt_baseline_output"],
        corrupt_into_clean_outputs=singleton["corrupt_into_clean_outputs"],
        clean_into_corrupt_outputs=singleton["clean_into_corrupt_outputs"],
        eap_fine_scores=regular_eap_scores,
        eap_ig5_fine_scores=integrated_eap_scores,
        raw_whole_mass=raw_mass,
    )
    induced_ids = tuple(fine_names[index] for index in universe["induced_fine_indices"])
    return {
        "induced_fine_edge_ids": induced_ids,
        "discovery_selector_scores": selectors,
    }


def _raw_edge_set_outputs(record: Mapping[str, Any]) -> dict[str, Any]:
    """Rejoin serialized plan references to raw condition outputs."""
    plan = record["plan"]
    universe = record["universe"]
    condition_outputs = torch.as_tensor(record["condition_outputs"])
    clean = torch.as_tensor(record["clean_baseline_output"])

    def condition(index: Any) -> torch.Tensor:
        if index is None:
            return clean
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("raw edge-set condition reference is not an integer")
        return condition_outputs[index]

    for stored_name, index_name in (
        ("all_internal_output", "all_internal_condition_index"),
        ("induced_output", "induced_condition_index"),
        ("outside_induced_output", "outside_induced_condition_index"),
    ):
        if not torch.equal(
            torch.as_tensor(record[stored_name]), condition(plan[index_name])
        ):
            raise ValueError("raw edge-set anchor output differs from plan reference")
    within_index = plan["primary_within_induced_complement_condition_index"]
    within = None if within_index is None else condition(within_index)
    stored_within = record["primary_within_induced_complement_output"]
    if (within is None) != (stored_within is None) or (
        within is not None and not torch.equal(torch.as_tensor(stored_within), within)
    ):
        raise ValueError("raw I-minus-primary output differs from plan reference")

    supports = tuple(plan["supports"])
    references = tuple(plan["support_references"])
    selected_stack = torch.as_tensor(record["support_selected_outputs"])
    complement_stack = torch.as_tensor(record["support_complement_outputs"])
    if (
        len(supports) != len(references)
        or len(supports) != len(selected_stack)
        or len(supports) != len(complement_stack)
    ):
        raise ValueError("raw edge-set support/reference/output axes differ")
    fine_names = tuple(universe["fine_edge_names"])
    outputs: list[dict[str, Any]] = []
    for index, (support, reference) in enumerate(
        zip(supports, references, strict=True)
    ):
        if support["support_id"] != reference["support_id"]:
            raise ValueError("raw edge-set support and reference IDs differ")
        selected = condition(reference["selected_condition_index"])
        complement = condition(reference["complement_condition_index"])
        if not torch.equal(selected_stack[index], selected) or not torch.equal(
            complement_stack[index], complement
        ):
            raise ValueError("raw edge-set support stack differs from plan reference")
        names = tuple(fine_names[position] for position in support["selected_indices"])
        outputs.append(
            {
                "support": support,
                "selected_fine_edge_ids": names,
                "selected_output": selected,
                "complement_output": complement,
            }
        )
    return {
        "supports": tuple(outputs),
        "clean_output": clean,
        "corrupt_output": torch.as_tensor(record["corrupt_baseline_output"]),
        "all_internal_output": condition(plan["all_internal_condition_index"]),
        "induced_output": condition(plan["induced_condition_index"]),
        "outside_induced_output": condition(plan["outside_induced_condition_index"]),
        "within_primary_output": within,
    }


def _serialized_plan_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a tensor-artifact plan to the JSON authorization form."""
    plan = record["plan"]
    masks = tuple(
        tuple(bool(value) for value in row)
        for row in torch.as_tensor(plan["fine_masks"], dtype=torch.bool).tolist()
    )
    return {
        "deduped_condition_count": len(masks),
        "condition_ids": tuple(plan["condition_ids"]),
        "fine_masks": masks,
        "fine_masks_sha256": _json_sha256(masks),
        "all_internal_condition_index": plan["all_internal_condition_index"],
        "induced_condition_index": plan["induced_condition_index"],
        "outside_induced_condition_index": plan["outside_induced_condition_index"],
        "primary_support_id": plan["primary_support_id"],
        "primary_within_induced_complement_condition_index": plan[
            "primary_within_induced_complement_condition_index"
        ],
        "support_references": tuple(plan["support_references"]),
    }


def _recompute_confirmation_score_inputs(
    confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    payload = confirmation["scientific_payload"]
    panel = payload["panel"]
    case_id = int(confirmation["case_id"])
    rows = (
        *token_rows(panel["clean_inputs"], label="score-stage C clean inputs"),
        *token_rows(panel["corrupt_inputs"], label="score-stage C corrupt inputs"),
    )
    if rows != tuple(
        tuple(row) for row in confirmation["selected_token_rows"]
    ) or confirmation["panel_identity_sha256"] != _panel_identity_sha256(
        case_id=case_id, split="confirmation", selected_token_rows=rows
    ):
        raise ValueError("raw C panel tensors differ from selected-row custody")
    resample_record = payload["resample_set_panel"]
    mean_record = payload["mean_corrupt_primary_sensitivity"]
    if (
        resample_record.get("ablation") != "resample"
        or mean_record.get("ablation") != "batch_tokenwise_mean_corrupt"
        or _canonical_json_bytes(resample_record["universe"])
        != _canonical_json_bytes(mean_record["universe"])
    ):
        raise ValueError("raw C ablation identities or universes differ")
    resample = _raw_edge_set_outputs(resample_record)
    mean = _raw_edge_set_outputs(mean_record)
    if not torch.equal(
        resample["clean_output"], mean["clean_output"]
    ) or not torch.equal(resample["corrupt_output"], mean["corrupt_output"]):
        raise ValueError("raw resample and mean C baselines differ")
    all_resample = resample["supports"]
    deterministic = all_resample[: len(DETERMINISTIC_METHOD_ORDER)]
    curve_count = len(descriptive_support_sizes(case_id))
    curves = all_resample[
        len(DETERMINISTIC_METHOD_ORDER) : len(DETERMINISTIC_METHOD_ORDER) + curve_count
    ]
    random = all_resample[len(DETERMINISTIC_METHOD_ORDER) + curve_count :]
    if (
        tuple(record["support"]["support_id"] for record in deterministic)
        != DETERMINISTIC_METHOD_ORDER
        or len(random) != RANDOM_CONTROL_REPLICATES
        or len(mean["supports"]) != 1
        or mean["supports"][0]["support"]["support_id"] != PRIMARY_METHOD
        or resample["within_primary_output"] is None
    ):
        raise ValueError("raw C support order differs from frozen protocol")

    def scored(record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "selected_fine_edge_ids": record["selected_fine_edge_ids"],
            "selected_output": record["selected_output"],
            "complement_output": record["complement_output"],
        }

    curve_labels = tuple(descriptive_support_sizes(case_id))
    return {
        "clean_output": resample["clean_output"],
        "actual_corrupt_output": resample["corrupt_output"],
        "clean_target_indices": torch.as_tensor(panel["clean_targets"])
        .argmax(dim=-1)[:, 1:]
        .cpu(),
        "resample_all_internal_output": resample["all_internal_output"],
        "resample_induced_core_output": resample["induced_output"],
        "resample_outside_core_output": resample["outside_induced_output"],
        "resample_deterministic_outputs": {
            method: scored(record)
            for method, record in zip(
                DETERMINISTIC_METHOD_ORDER, deterministic, strict=True
            )
        },
        "resample_exact_mass_curve_outputs": {
            label: scored(record)
            for label, record in zip(curve_labels, curves, strict=True)
        },
        "resample_random_outputs": tuple(
            {"replicate": replicate, **scored(record)}
            for replicate, record in enumerate(random)
        ),
        "resample_within_core_primary_complement_output": resample[
            "within_primary_output"
        ],
        "mean_all_internal_output": mean["all_internal_output"],
        "mean_induced_core_output": mean["induced_output"],
        "mean_outside_core_output": mean["outside_induced_output"],
        "mean_primary_output": mean["supports"][0]["selected_output"],
        "mean_primary_complement_output": mean["supports"][0]["complement_output"],
    }


def _score_production_panel(
    panel: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Pure score replay; official GT is opened only inside this score stage."""
    from experiments.e32_interpbench.e32_3_cohort import load_exact_ground_truth
    from experiments.e32_interpbench.e32_4_scoring import (
        descriptive_ground_truth_metrics,
        evaluate_case,
        evaluate_panel_verdict,
        exact_fine_ground_truth_ids,
    )

    expected = tuple(
        (case_id, split) for case_id in CASE_ORDER for split in MANDATORY_SPLITS
    )
    if tuple(panel) != expected:
        raise ValueError("score stage requires the complete ordered D/C panel")
    case_scores: dict[str, Any] = {}
    for case_id in CASE_ORDER:
        discovery = panel[(case_id, "discovery")]
        confirmation = panel[(case_id, "confirmation")]
        discovery_inputs = discovery.get("score_inputs")
        confirmation_inputs = confirmation.get("score_inputs")
        if (
            not isinstance(discovery_inputs, Mapping)
            or tuple(discovery_inputs) != _DISCOVERY_SCORE_INPUT_KEYS
            or not isinstance(confirmation_inputs, Mapping)
            or tuple(confirmation_inputs) != _CONFIRMATION_SCORE_INPUT_KEYS
        ):
            raise ValueError("score inputs changed after acquisition custody")
        recomputed_discovery = _recompute_discovery_score_inputs(discovery)
        recomputed_confirmation = _recompute_confirmation_score_inputs(confirmation)
        discovery_for_freeze = {
            **discovery,
            "score_inputs": recomputed_discovery,
        }
        expected_frozen = _freeze_production_discovery_supports(
            case_id=case_id, discovery=discovery_for_freeze
        )
        raw_resample = confirmation["scientific_payload"]["resample_set_panel"]
        raw_mean = confirmation["scientific_payload"][
            "mean_corrupt_primary_sensitivity"
        ]
        expected_plans = expected_frozen["confirmation_mask_plan"]
        if (
            _canonical_json_bytes(raw_resample["plan"]["supports"])
            != _canonical_json_bytes(expected_frozen["edge_supports"])
            or _canonical_json_bytes(raw_mean["plan"]["supports"])
            != _canonical_json_bytes((expected_frozen["edge_supports"][0],))
            or _canonical_json_bytes(_serialized_plan_record(raw_resample))
            != _canonical_json_bytes(expected_plans["resample"])
            or _canonical_json_bytes(_serialized_plan_record(raw_mean))
            != _canonical_json_bytes(expected_plans["mean_corrupt_primary_only"])
        ):
            raise ValueError("raw C masks differ from D-recomputed support plans")
        if _canonical_json_bytes(
            discovery["scientific_payload"]["edge_universe"]
        ) != _canonical_json_bytes(
            confirmation["scientific_payload"]["resample_set_panel"]["universe"]
        ):
            raise ValueError("D and C raw fine-edge universes differ")
        if not _scientific_equal(discovery_inputs, recomputed_discovery):
            raise ValueError("serialized D score_inputs differ from raw acquisitions")
        if not _scientific_equal(confirmation_inputs, recomputed_confirmation):
            raise ValueError("serialized C score_inputs differ from raw acquisitions")
        score = evaluate_case(
            case_id=case_id,
            **recomputed_discovery,
            **recomputed_confirmation,
        )

        universe = discovery["scientific_payload"]["edge_universe"]
        full_fine_ids = tuple(universe["fine_edge_names"])
        whole_edges = tuple(tuple(edge) for edge in universe["whole_edges"])
        mapping = tuple(universe["fine_to_whole_indices"])
        full_fine_whole_edges = tuple(whole_edges[index] for index in mapping)
        ground_truth = load_exact_ground_truth(DEFAULT_MODEL_ROOT, case_id)
        ground_truth_whole = tuple(ground_truth.internal_edges)
        fine_truth = exact_fine_ground_truth_ids(
            case_id=case_id,
            full_fine_edge_ids=full_fine_ids,
            induced_fine_edge_ids=recomputed_discovery["induced_fine_edge_ids"],
            regular_eap_all_fine_records=discovery["scientific_payload"]["regular_eap"][
                "all_fine_scores"
            ],
            official_raw_edges=ground_truth.raw_edges,
        )
        transitive = _transitive_non_edges(
            nodes=tuple(discovery["scientific_payload"]["component_ids"]),
            ground_truth_edges=ground_truth_whole,
            candidate_edges=whole_edges,
        )
        score["descriptive_ground_truth"] = descriptive_ground_truth_metrics(
            full_fine_edge_ids=full_fine_ids,
            full_fine_whole_edges=full_fine_whole_edges,
            induced_fine_edge_ids=recomputed_discovery["induced_fine_edge_ids"],
            selector_scores=recomputed_discovery["discovery_selector_scores"],
            selected_supports=score["supports"],
            fine_ground_truth_ids=fine_truth,
            whole_ground_truth_edges=ground_truth_whole,
            transitive_non_edges=transitive,
        )
        case_scores[str(case_id)] = score
    return {
        "schema_version": "e32_4_internal_edge_set_scientific_score_v1",
        "case_order": CASE_ORDER,
        "cases": case_scores,
        "panel_verdict": evaluate_panel_verdict(case_scores),
        "ground_truth_opened_only_after_score_attempt": True,
        "ground_truth_used_for_selection_or_gates": False,
    }


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
        record = seal_registration()
        print(json.dumps({"registration_sha256": _json_sha256(record)}))
        return 0
    summary = run_registered_experiment(confirm=arguments.confirm)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_PATHS",
    "CANONICAL_REGISTRATION_PATH",
    "CANONICAL_RUN_ATTEMPT_PATH",
    "CANONICAL_RUN_DIRECTORY",
    "REPOSITORY_ROOT",
    "SOURCE_RELATIVE_PATHS",
    "RunPaths",
    "build_registration_record",
    "run_registered_experiment",
    "seal_registration",
]
