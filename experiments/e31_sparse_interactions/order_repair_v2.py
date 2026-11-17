# ruff: noqa: EXE002
"""Transparent V2 continuation after E31.1's outer-dictionary order failure.

V1 acquired and sealed discovery responses, then stopped before fitting any
geometry or opening patched confirmation responses.  This module preserves
every V1 byte and applies the unique response-independent permutation from the
assembler's per-scale key order to the scorer's already-registered key order.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_15_ioi_acquisition import (
    clean_panel_viability_report,
)
from experiments.e30_redundant_causal_interfaces.custody import (
    atomic_json_save,
    atomic_torch_save,
    ensure_write_once_available,
    runtime_environment,
    sha256,
    valid_sha256,
    verify_checkpoint_closure,
    verify_file_hash,
    verify_source_closure,
    write_once_reservation_path,
)
from experiments.e31_sparse_interactions.panel import (
    build_prompt_manifest,
    canonical_sha256,
)
from experiments.e31_sparse_interactions.prospective_acquisition import (
    acquire_prepared_split,
    validate_acquired_split,
)
from experiments.e31_sparse_interactions.prospective_custody import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_MODEL_SNAPSHOT,
    EXPECTED_RUNTIME_ENVIRONMENT,
    REPOSITORY_ROOT,
)
from experiments.e31_sparse_interactions.prospective_custody import (
    SOURCE_PATHS as V1_SOURCE_PATHS,
)
from experiments.e31_sparse_interactions.prospective_orchestration import (
    _default_runner_factory,
    _validate_clean_screen,
)
from experiments.e31_sparse_interactions.prospective_protocol import (
    PROTOCOL_SHA256,
    SPLIT_MANIFEST_SHA256,
    build_split_mask_manifest,
)
from experiments.e31_sparse_interactions.prospective_registration import (
    validate_registration_payload,
)
from experiments.e31_sparse_interactions.prospective_scoring import (
    assemble_regime_panels,
    fit_discovery_geometry,
    score_with_frozen_geometry,
)

ASSEMBLED_REGIME_ORDER = (
    "nonhub_k02_hub_off",
    "nonhub_k02_hub_on",
    "spokes_k02",
    "nonhub_k10_hub_off",
    "nonhub_k10_hub_on",
    "spokes_k10",
    "nonhub_k18_hub_off",
    "nonhub_k18_hub_on",
    "spokes_k18",
)
REGISTERED_REGIME_ORDER = (
    "nonhub_k02_hub_off",
    "nonhub_k02_hub_on",
    "nonhub_k10_hub_off",
    "nonhub_k10_hub_on",
    "nonhub_k18_hub_off",
    "nonhub_k18_hub_on",
    "spokes_k02",
    "spokes_k10",
    "spokes_k18",
)
PRIMARY_REGIME_ID = "nonhub_k10_hub_off"

V1_REGISTRATION_PATH = (
    REPOSITORY_ROOT / "notes/E31.1-PROSPECTIVE-SPARSE-ATLAS-REGISTRATION.json"
)
V1_RUN_DIRECTORY = REPOSITORY_ROOT / "runs/e31_1_prospective_sparse_atlas_v1"
V2_REGISTRATION_PATH = (
    REPOSITORY_ROOT
    / "notes/E31.1-PROSPECTIVE-SPARSE-ATLAS-V2-ORDER-REPAIR-REGISTRATION.json"
)
V2_RUN_DIRECTORY = (
    REPOSITORY_ROOT / "runs/e31_1_prospective_sparse_atlas_v2_order_repair"
)

V1_ARTIFACTS = {
    "registration": (
        V1_REGISTRATION_PATH,
        "a9600918bbeef14bf8db83c249106d888fd8bafbe8715cfc8787a4c877d1eb73",
    ),
    "global_attempt": (
        V1_RUN_DIRECTORY / "global_attempt.json",
        "865e649ed879344b61dd277e5f1f09c73f53ee62d06f2bb55f5c7bf1c4c1e9ac",
    ),
    "clean_screen": (
        V1_RUN_DIRECTORY / "all_registered_clean_screen.pt",
        "f6f649698d2f5d6d0434346187e6a0f11e895b5332df6e709847bc3f8ceb8033",
    ),
    "clean_custody": (
        V1_RUN_DIRECTORY / "all_registered_clean_screen_custody.json",
        "ad813dd6597ca5719e73616b7417990e93a78808e7327f1eccffc506623efa11",
    ),
    "discovery_labels": (
        V1_RUN_DIRECTORY / "discovery_labels.pt",
        "e293f1c2895bf3ed3687a0f1aa7021cec453007d413ab6e0d1dde5082bb93e61",
    ),
    "discovery_custody": (
        V1_RUN_DIRECTORY / "discovery_custody.json",
        "a1741ee02664ffe9e12c379dc728aa94085f4fde41994f7ff6402640474bdb9b",
    ),
}
V1_FORBIDDEN_AFTER_FAILURE = (
    "discovery_geometry.pt",
    "discovery_geometry_custody.json",
    "confirmation_authorization.json",
    "confirmation_labels.pt",
    "confirmation_custody.json",
    "score_attempt.json",
    "prospective_score.pt",
    "score_custody.json",
    "clean_screen_failure.pt",
    "clean_screen_failure_custody.json",
    "run_failure_custody.json",
)

V2_SOURCE_PATHS = {
    **V1_SOURCE_PATHS,
    "e31/order_repair_v2.py": Path(__file__).resolve(),
}
V2_SCHEMA_VERSION = "e31_1_order_repair_registration_v2"
V2_UNSEALED_STATUS = "order_repair_unsealed_before_patched_confirmation"
V2_SEALED_STATUS = "order_repair_sealed_before_patched_confirmation"
V1_CONSUMED_FORWARD_EXAMPLES = 124_736
V2_RECOVERY_FORWARD_EXAMPLES = 248_896
LIFETIME_FORWARD_EXAMPLES = 373_632
UNIQUE_REGISTERED_WORKLOAD = 373_568


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _load_torch(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"expected a mapping Torch artifact in {path}")
    return value


def canonicalize_panel_order(
    panels: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Apply the unique key-only V2 repair without copying any panel value."""
    if tuple(panels) != ASSEMBLED_REGIME_ORDER:
        raise RuntimeError("assembled regime order differs from the V2 repair premise")
    if set(panels) != set(REGISTERED_REGIME_ORDER):
        raise RuntimeError("assembled regime key set differs from the registered set")
    repaired = {key: panels[key] for key in REGISTERED_REGIME_ORDER}
    if any(repaired[key] is not panels[key] for key in REGISTERED_REGIME_ORDER):
        raise RuntimeError("order repair copied or changed a panel value")
    return repaired


def _verify_v1_failure_state() -> dict[str, Any]:
    for path, digest in V1_ARTIFACTS.values():
        verify_file_hash(path, digest)
        if not write_once_reservation_path(path).is_dir():
            raise RuntimeError(f"V1 write-once reservation is absent: {path.name}")
    expected_run_entries = {
        path.name
        for name, (path, _digest) in V1_ARTIFACTS.items()
        if name != "registration"
    }
    expected_run_entries |= {
        write_once_reservation_path(path).name
        for name, (path, _digest) in V1_ARTIFACTS.items()
        if name != "registration"
    }
    observed_run_entries = {
        path.name
        for path in V1_RUN_DIRECTORY.iterdir()
        if not path.name.startswith("._")
    }
    if observed_run_entries != expected_run_entries:
        raise RuntimeError("V1 failure-state file/reservation set changed")
    for filename in V1_FORBIDDEN_AFTER_FAILURE:
        path = V1_RUN_DIRECTORY / filename
        if path.exists() or write_once_reservation_path(path).exists():
            raise RuntimeError(
                f"V1 post-discovery artifact unexpectedly exists: {filename}"
            )

    registration = validate_registration_payload(
        _read_json(V1_REGISTRATION_PATH), verify_external_files=True
    )
    if (
        registration["protocol_sha256"] != PROTOCOL_SHA256
        or registration["split_manifest_sha256"] != SPLIT_MANIFEST_SHA256
        or registration["scoring_source_sha256"]
        != "182aefd99140ea09e6dcbf0e1300bea8d674301522d3086f5b5170a4c7c1486a"
    ):
        raise RuntimeError("V1 scientific protocol or scorer binding changed")

    attempt = _read_json(V1_ARTIFACTS["global_attempt"][0])
    clean_custody = _read_json(V1_ARTIFACTS["clean_custody"][0])
    discovery_custody = _read_json(V1_ARTIFACTS["discovery_custody"][0])
    if (
        attempt.get("registration_sha256") != V1_ARTIFACTS["registration"][1]
        or attempt.get("protocol_sha256") != PROTOCOL_SHA256
        or clean_custody.get("registration_sha256") != V1_ARTIFACTS["registration"][1]
        or clean_custody.get("global_attempt_sha256")
        != V1_ARTIFACTS["global_attempt"][1]
        or clean_custody.get("clean_screen_sha256") != V1_ARTIFACTS["clean_screen"][1]
        or discovery_custody.get("registration_sha256")
        != V1_ARTIFACTS["registration"][1]
        or discovery_custody.get("global_attempt_sha256")
        != V1_ARTIFACTS["global_attempt"][1]
        or discovery_custody.get("clean_screen_sha256")
        != V1_ARTIFACTS["clean_screen"][1]
        or discovery_custody.get("labels_sha256") != V1_ARTIFACTS["discovery_labels"][1]
        or discovery_custody.get("discovery_geometry_sha256") is not None
        or discovery_custody.get("confirmation_authorization_sha256") is not None
    ):
        raise RuntimeError("V1 failure-state custody chain changed")
    return registration


def _verify_dependencies() -> dict[str, Any]:
    parent_registration = _verify_v1_failure_state()
    source_hashes = verify_source_closure(V2_SOURCE_PATHS)
    checkpoint_hashes = verify_checkpoint_closure(
        EXPECTED_MODEL_SNAPSHOT, EXPECTED_CHECKPOINT_SHA256
    )
    environment = runtime_environment()
    if environment != EXPECTED_RUNTIME_ENVIRONMENT:
        raise RuntimeError("runtime environment differs from the V2 repair seal")
    if checkpoint_hashes != parent_registration["checkpoint_sha256"]:
        raise RuntimeError("V2 checkpoint differs from V1")
    return {
        "source_code_sha256": source_hashes,
        "checkpoint_sha256": checkpoint_hashes,
        "environment": environment,
    }


def _build_registration(
    dependencies: Mapping[str, Any], *, sealed: bool
) -> dict[str, Any]:
    if set(dependencies["source_code_sha256"]) != set(V2_SOURCE_PATHS):
        raise RuntimeError("V2 source closure changed")
    return {
        "schema_version": V2_SCHEMA_VERSION,
        "status": V2_SEALED_STATUS if sealed else V2_UNSEALED_STATUS,
        "scientific_protocol": {
            "v1_registration_sha256": V1_ARTIFACTS["registration"][1],
            "protocol_sha256": PROTOCOL_SHA256,
            "split_manifest_sha256": dict(SPLIT_MANIFEST_SHA256),
            "scoring_source_sha256": (
                "182aefd99140ea09e6dcbf0e1300bea8d674301522d3086f5b5170a4c7c1486a"
            ),
            "estimands_gates_bootstrap_and_controls_changed": False,
        },
        "v1_failure_state": {
            "artifacts": {
                name: {"path": str(path.resolve()), "sha256": digest}
                for name, (path, digest) in V1_ARTIFACTS.items()
            },
            "forbidden_absent_files_and_reservations": list(V1_FORBIDDEN_AFTER_FAILURE),
            "exception": ("ValueError: the registered discovery regime order changed"),
            "failure_stage": "before_any_lambda_rank_control_or_geometry_fit",
            "discovery_artifact_opened_for_key_order_diagnosis": True,
            "discovery_response_values_or_statistics_inspected_for_repair": False,
            "discovery_geometry_computed_in_preseal_implementation_test": True,
            "preseal_geometry_fields_reported": {
                "primary_regime_id": PRIMARY_REGIME_ID,
                "matched_control_support_count": 10_000,
            },
            "preseal_selected_edge_identities_or_ranks_inspected": False,
            "preseal_matched_control_support_memberships_or_identities_inspected": (
                False
            ),
            "preseal_lambda_curvature_or_effect_energy_values_inspected": False,
            "patched_confirmation_responses_or_labels_opened": False,
            "confirmation_clean_endpoints_already_screened_in_v1": True,
        },
        "repair": {
            "assembled_regime_order": list(ASSEMBLED_REGIME_ORDER),
            "registered_regime_order": list(REGISTERED_REGIME_ORDER),
            "operation": "outer_mapping_key_permutation_only",
            "same_nine_key_set_required": True,
            "identical_panel_object_per_key_required": True,
            "response_values_used_to_choose_permutation": False,
            "discovery_rerun_allowed": False,
        },
        "accounting": {
            "v1_consumed_forward_examples": V1_CONSUMED_FORWARD_EXAMPLES,
            "v2_confirmation_clean_reprepare_examples": 64,
            "v2_confirmation_patched_examples": 248_832,
            "v2_recovery_forward_examples": V2_RECOVERY_FORWARD_EXAMPLES,
            "planned_lifetime_forward_examples_on_success": (LIFETIME_FORWARD_EXAMPLES),
            "unique_original_registered_workload": UNIQUE_REGISTERED_WORKLOAD,
            "discovery_rerun_examples": 0,
            "second_batch_smoke_examples": 0,
        },
        "custody_order": [
            "v2_registration",
            "v2_attempt",
            "v1_failure_reproduction",
            "discovery_geometry_and_controls",
            "confirmation_authorization",
            "confirmation_clean_revalidation",
            "confirmation_patched_labels",
            "pure_score_attempt_and_score",
        ],
        "claim_boundary": {
            "prospective_relative_to_patched_confirmation_outcomes": True,
            "prospective_relative_to_discovery": False,
            "ordinary_v1_completion": False,
            "post_discovery_operational_repair_disclosed": True,
            "unsealed_real_discovery_geometry_implementation_test": True,
            "preseal_geometry_fields_exposed": (
                "fixed_primary_regime_id_and_successful_10000_control_count_only"
            ),
            "preseal_selected_edge_identities_or_ranks_inspected": False,
            "preseal_matched_control_support_memberships_or_identities_inspected": (
                False
            ),
            "preseal_lambda_curvature_or_effect_energy_values_inspected": False,
            "prediction_topology_direct_communication_global_sparsity": (
                "not_authorized"
            ),
        },
        "registered_paths": {
            "registration": str(V2_REGISTRATION_PATH.resolve()),
            "output_directory": str(V2_RUN_DIRECTORY.resolve()),
            "separate_from_v1_output": True,
        },
        "v2_preseal_state": {
            "output_non_appledouble_entries": [],
            "checked_immediately_before_registration_write": True,
        },
        "source_code_sha256": dict(dependencies["source_code_sha256"]),
        "checkpoint_sha256": dict(dependencies["checkpoint_sha256"]),
        "environment": dict(dependencies["environment"]),
    }


def dry_run() -> dict[str, Any]:
    dependencies = _verify_dependencies()
    registration = _build_registration(dependencies, sealed=False)
    return {
        "status": "v2_order_repair_dry_run_no_model_no_write",
        "registration_canonical_sha256": canonical_sha256(registration),
        "parent_discovery_sha256": V1_ARTIFACTS["discovery_labels"][1],
        "planned_lifetime_forward_examples_on_success": (LIFETIME_FORWARD_EXAMPLES),
        "repair": registration["repair"],
    }


def _verify_v2_output_pristine() -> None:
    if not V2_RUN_DIRECTORY.exists():
        return
    entries = sorted(
        path.name
        for path in V2_RUN_DIRECTORY.iterdir()
        if not path.name.startswith("._")
    )
    if entries:
        raise RuntimeError(
            "V2 output directory was not pristine at registration seal: "
            + ", ".join(entries)
        )


def seal_registration(
    path: Path = V2_REGISTRATION_PATH,
) -> dict[str, Any]:
    if path.resolve() != V2_REGISTRATION_PATH.resolve():
        raise RuntimeError("V2 registration path differs from the frozen path")
    _verify_v2_output_pristine()
    ensure_write_once_available(path)
    registration = _build_registration(_verify_dependencies(), sealed=True)
    _verify_v2_output_pristine()
    atomic_json_save(registration, path)
    return {
        "status": "v2_order_repair_registration_sealed",
        "registration": str(path.resolve()),
        "registration_sha256": sha256(path),
    }


def _load_registration(path: Path, expected_sha256: str) -> dict[str, Any]:
    if path.resolve() != V2_REGISTRATION_PATH.resolve():
        raise RuntimeError("V2 registration path differs from the frozen path")
    if not valid_sha256(expected_sha256):
        raise ValueError("expected V2 registration SHA-256 is invalid")
    if not write_once_reservation_path(path).is_dir():
        raise RuntimeError("V2 registration write-once reservation is absent")
    verify_file_hash(path, expected_sha256)
    observed = _read_json(path)
    expected = _build_registration(_verify_dependencies(), sealed=True)
    if observed != expected:
        raise RuntimeError("V2 order-repair registration changed")
    return observed


def _nested_equal(left: Any, right: Any, *, path: str = "root") -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        ):
            raise RuntimeError(f"frozen geometry changed at {path}")
        return
    if type(left) is not type(right):
        raise RuntimeError(f"frozen geometry type changed at {path}")
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not (isinstance(left, Mapping) and tuple(left) == tuple(right)):
            raise RuntimeError(f"frozen geometry mapping changed at {path}")
        for key in left:
            _nested_equal(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not (not isinstance(right, (str, bytes)) and len(left) == len(right)):
            raise RuntimeError(f"frozen geometry sequence changed at {path}")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _nested_equal(a, b, path=f"{path}[{index}]")
        return
    if left != right:
        raise RuntimeError(f"frozen geometry scalar changed at {path}")


def _load_validated_v1_inputs() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any]
]:
    registration = _verify_v1_failure_state()
    clean = _load_torch(V1_ARTIFACTS["clean_screen"][0])
    _validate_clean_screen(clean, registration)
    discovery = validate_acquired_split(
        _load_torch(V1_ARTIFACTS["discovery_labels"][0]),
        registration=registration,
        registration_sha256=V1_ARTIFACTS["registration"][1],
        split="discovery",
        clean_screen_sha256=V1_ARTIFACTS["clean_screen"][1],
    )
    return registration, clean, discovery


def _discovery_panels(discovery: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    assembled = assemble_regime_panels(
        responses=discovery["normalized_responses"],
        split_manifest=build_split_mask_manifest("discovery"),
        prompt_records=discovery["prompt_records"],
    )
    try:
        fit_discovery_geometry(
            discovery_panels=assembled,
            primary_regime_id=PRIMARY_REGIME_ID,
        )
    except ValueError as error:
        if str(error) != "the registered discovery regime order changed":
            raise RuntimeError("V1 failure did not reproduce exactly") from error
    else:
        raise RuntimeError("V1 order failure unexpectedly did not reproduce")
    return canonicalize_panel_order(assembled)


def _confirmation_clean_revalidation(
    runner: Any,
    parent_registration: Mapping[str, Any],
    parent_clean: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    contexts = tuple(
        runner.prepare_context(record)
        for record in build_prompt_manifest("confirmation")
    )
    report = clean_panel_viability_report(contexts)
    metadata = tuple(dict(context.clean_metadata) for context in contexts)
    base = torch.tensor(
        [context.clean_base_donor_minus_base_margin for context in contexts],
        dtype=torch.float64,
    )
    donor = torch.tensor(
        [context.clean_donor_donor_minus_base_margin for context in contexts],
        dtype=torch.float64,
    )
    endpoint = torch.tensor(
        [context.clean_endpoint for context in contexts], dtype=torch.float64
    )
    expected_metadata = tuple(parent_clean["clean_metadata"]["confirmation"])
    expected_contract = tuple(
        parent_registration["clean_metadata_contract"]["confirmation"]
    )
    if (
        report != parent_clean["reports"]["confirmation"]
        or report.get("passed") is not True
        or metadata != expected_metadata
        or any(
            {key: observed.get(key) for key in contract} != contract
            for observed, contract in zip(metadata, expected_contract, strict=True)
        )
        or not torch.equal(
            base, parent_clean["clean_base_donor_minus_base_margins"]["confirmation"]
        )
        or not torch.equal(
            donor,
            parent_clean["clean_donor_donor_minus_base_margins"]["confirmation"],
        )
        or not torch.equal(endpoint, parent_clean["clean_endpoints"]["confirmation"])
    ):
        raise RuntimeError("reprepared confirmation clean rows differ from V1")
    return contexts, {
        "schema_version": "e31_1_confirmation_clean_revalidation_v2",
        "status": "all_32_reprepared_clean_rows_bitwise_match_v1",
        "context_count": 32,
        "clean_forward_examples": 64,
        "report": deepcopy(report),
        "clean_metadata": metadata,
        "clean_base_donor_minus_base_margins": base,
        "clean_donor_donor_minus_base_margins": donor,
        "clean_endpoints": endpoint,
        "parent_clean_screen_sha256": V1_ARTIFACTS["clean_screen"][1],
    }


def _validate_v2_recovery_chain(
    *,
    output_directory: Path,
    registration: Mapping[str, Any],
    registration_sha256: str,
) -> dict[str, str]:
    """Validate every V1 and V2 backlink after the score attempt exists."""
    paths = {
        "attempt": output_directory / "recovery_attempt.json",
        "geometry": output_directory / "discovery_geometry.pt",
        "geometry_custody": output_directory / "discovery_geometry_custody.json",
        "authorization": output_directory / "confirmation_authorization.json",
        "revalidation": output_directory / "confirmation_clean_revalidation.pt",
        "revalidation_custody": output_directory
        / "confirmation_clean_revalidation_custody.json",
        "confirmation": output_directory / "confirmation_labels.pt",
        "confirmation_custody": output_directory / "confirmation_custody.json",
    }
    for name, path in paths.items():
        if not write_once_reservation_path(path).is_dir():
            raise RuntimeError(f"V2 recovery artifact reservation is absent: {name}")
    digests = {name: sha256(path) for name, path in paths.items()}
    expected_attempt = {
        "schema_version": "e31_1_order_repair_attempt_v2",
        "status": "attempt_consumed_before_discovery_fit_or_model_load",
        "registration_sha256": registration_sha256,
        "source_code_sha256": registration["source_code_sha256"],
        "parent_artifacts": registration["v1_failure_state"]["artifacts"],
        "actual_lifetime_forward_examples_at_attempt": (V1_CONSUMED_FORWARD_EXAMPLES),
        "planned_lifetime_forward_examples_on_success": (LIFETIME_FORWARD_EXAMPLES),
    }
    expected_geometry_custody = {
        "schema_version": "e31_1_order_repair_geometry_custody_v2",
        "status": "geometry_and_controls_frozen_before_patched_confirmation",
        "registration_sha256": registration_sha256,
        "recovery_attempt_sha256": digests["attempt"],
        "v1_discovery_labels_sha256": V1_ARTIFACTS["discovery_labels"][1],
        "discovery_geometry_sha256": digests["geometry"],
        "matched_control_support_count": 10_000,
        "patched_confirmation_outcomes_used": False,
    }
    expected_authorization = {
        "schema_version": "e31_1_order_repair_confirmation_authorization_v2",
        "status": "patched_confirmation_authorized_after_v2_geometry_seal",
        "registration_sha256": registration_sha256,
        "v1_registration_sha256": V1_ARTIFACTS["registration"][1],
        "v1_discovery_labels_sha256": V1_ARTIFACTS["discovery_labels"][1],
        "discovery_geometry_sha256": digests["geometry"],
        "geometry_custody_sha256": digests["geometry_custody"],
        "adaptation_after_authorization_allowed": False,
    }
    expected_revalidation_custody = {
        "schema_version": "e31_1_confirmation_clean_revalidation_custody_v2",
        "status": "bitwise_match_to_v1_before_any_patched_confirmation",
        "registration_sha256": registration_sha256,
        "authorization_sha256": digests["authorization"],
        "revalidation_sha256": digests["revalidation"],
        "clean_forward_examples": 64,
        "patched_confirmation_examples_before_revalidation": 0,
        "actual_lifetime_forward_examples_after_revalidation": 124_800,
    }
    expected_confirmation_custody = {
        "schema_version": "e31_1_order_repair_confirmation_custody_v2",
        "status": "unchanged_v1_confirmation_masks_acquired_write_once",
        "registration_sha256": registration_sha256,
        "v1_registration_sha256": V1_ARTIFACTS["registration"][1],
        "recovery_attempt_sha256": digests["attempt"],
        "discovery_geometry_sha256": digests["geometry"],
        "authorization_sha256": digests["authorization"],
        "revalidation_sha256": digests["revalidation"],
        "revalidation_custody_sha256": digests["revalidation_custody"],
        "confirmation_labels_sha256": digests["confirmation"],
        "confirmation_clean_reprepare_examples": 64,
        "confirmation_patched_examples": 248_832,
        "discovery_rerun_examples": 0,
        "actual_lifetime_forward_examples": LIFETIME_FORWARD_EXAMPLES,
    }
    observed = (
        _read_json(paths["attempt"]),
        _read_json(paths["geometry_custody"]),
        _read_json(paths["authorization"]),
        _read_json(paths["revalidation_custody"]),
        _read_json(paths["confirmation_custody"]),
    )
    expected = (
        expected_attempt,
        expected_geometry_custody,
        expected_authorization,
        expected_revalidation_custody,
        expected_confirmation_custody,
    )
    if observed != expected:
        raise RuntimeError("V2 recovery custody chain changed")
    return {
        **{
            f"v1_{name}_sha256": digest
            for name, (_path, digest) in V1_ARTIFACTS.items()
        },
        **{f"v2_{name}_sha256": digest for name, digest in digests.items()},
    }


def _score_existing(
    *,
    registration_path: Path,
    registration_sha256: str,
    output_directory: Path,
) -> dict[str, Any]:
    if output_directory.resolve() != V2_RUN_DIRECTORY.resolve():
        raise RuntimeError("V2 output directory differs from the frozen path")
    registration = _load_registration(registration_path, registration_sha256)
    _verify_dependencies()
    paths = {
        "attempt": output_directory / "recovery_attempt.json",
        "geometry": output_directory / "discovery_geometry.pt",
        "geometry_custody": output_directory / "discovery_geometry_custody.json",
        "authorization": output_directory / "confirmation_authorization.json",
        "revalidation": output_directory / "confirmation_clean_revalidation.pt",
        "revalidation_custody": output_directory
        / "confirmation_clean_revalidation_custody.json",
        "confirmation": output_directory / "confirmation_labels.pt",
        "confirmation_custody": output_directory / "confirmation_custody.json",
        "score_attempt": output_directory / "score_attempt.json",
        "score": output_directory / "prospective_score.pt",
        "score_custody": output_directory / "score_custody.json",
    }
    for name in ("score_attempt", "score", "score_custody"):
        ensure_write_once_available(paths[name])

    confirmation_custody = _read_json(paths["confirmation_custody"])
    expected_geometry_sha256 = confirmation_custody.get("discovery_geometry_sha256")
    expected_confirmation_sha256 = confirmation_custody.get(
        "confirmation_labels_sha256"
    )
    if not (
        valid_sha256(expected_geometry_sha256)
        and valid_sha256(expected_confirmation_sha256)
    ):
        raise RuntimeError("V2 confirmation custody omits expected score hashes")

    atomic_json_save(
        {
            "schema_version": "e31_1_order_repair_score_attempt_v2",
            "status": "attempt_consumed_before_hashing_or_opening_outcome_artifacts",
            "registration_sha256": registration_sha256,
            "expected_geometry_sha256": expected_geometry_sha256,
            "expected_confirmation_sha256": expected_confirmation_sha256,
            "expected_confirmation_custody_sha256": sha256(
                paths["confirmation_custody"]
            ),
            "source_code_sha256": registration["source_code_sha256"],
        },
        paths["score_attempt"],
    )
    acquisition_custody_chain = _validate_v2_recovery_chain(
        output_directory=output_directory,
        registration=registration,
        registration_sha256=registration_sha256,
    )
    geometry_sha256 = acquisition_custody_chain["v2_geometry_sha256"]
    confirmation_sha256 = acquisition_custody_chain["v2_confirmation_sha256"]

    parent_registration, parent_clean, discovery = _load_validated_v1_inputs()
    discovery_panels = _discovery_panels(discovery)
    geometry_record = _load_torch(paths["geometry"])
    if (
        geometry_record.get("schema_version") != "e31_1_order_repair_geometry_v2"
        or geometry_record.get("status")
        != "outer_order_repaired_geometry_frozen_before_patched_confirmation"
        or geometry_record.get("registration_sha256") != registration_sha256
        or geometry_record.get("v1_discovery_labels_sha256")
        != V1_ARTIFACTS["discovery_labels"][1]
        or geometry_record.get("repair") != registration["repair"]
    ):
        raise RuntimeError("V2 frozen geometry header changed")
    revalidation = _load_torch(paths["revalidation"])
    if (
        revalidation.get("schema_version") != "e31_1_confirmation_clean_revalidation_v2"
        or revalidation.get("status") != "all_32_reprepared_clean_rows_bitwise_match_v1"
        or revalidation.get("context_count") != 32
        or revalidation.get("clean_forward_examples") != 64
        or revalidation.get("parent_clean_screen_sha256")
        != V1_ARTIFACTS["clean_screen"][1]
        or revalidation.get("report") != parent_clean["reports"]["confirmation"]
        or tuple(revalidation.get("clean_metadata", ()))
        != tuple(parent_clean["clean_metadata"]["confirmation"])
        or not torch.equal(
            revalidation["clean_base_donor_minus_base_margins"],
            parent_clean["clean_base_donor_minus_base_margins"]["confirmation"],
        )
        or not torch.equal(
            revalidation["clean_donor_donor_minus_base_margins"],
            parent_clean["clean_donor_donor_minus_base_margins"]["confirmation"],
        )
        or not torch.equal(
            revalidation["clean_endpoints"],
            parent_clean["clean_endpoints"]["confirmation"],
        )
    ):
        raise RuntimeError("V2 confirmation clean revalidation changed")
    recomputed_geometry = fit_discovery_geometry(
        discovery_panels=discovery_panels,
        primary_regime_id=PRIMARY_REGIME_ID,
    )
    _nested_equal(recomputed_geometry, geometry_record["frozen_geometry"])
    confirmation = validate_acquired_split(
        _load_torch(paths["confirmation"]),
        registration=parent_registration,
        registration_sha256=V1_ARTIFACTS["registration"][1],
        split="confirmation",
        clean_screen_sha256=V1_ARTIFACTS["clean_screen"][1],
    )
    confirmation_panels = canonicalize_panel_order(
        assemble_regime_panels(
            responses=confirmation["normalized_responses"],
            split_manifest=build_split_mask_manifest("confirmation"),
            prompt_records=confirmation["prompt_records"],
        )
    )
    score = score_with_frozen_geometry(
        discovery_panels=discovery_panels,
        confirmation_panels=confirmation_panels,
        frozen_geometry=geometry_record["frozen_geometry"],
    )
    score_record = {
        "schema_version": "e31_1_order_repair_prospective_score_v2",
        "status": "patched_confirmation_scored_under_unchanged_v1_protocol",
        "registration_sha256": registration_sha256,
        "v1_registration_sha256": V1_ARTIFACTS["registration"][1],
        "discovery_labels_sha256": V1_ARTIFACTS["discovery_labels"][1],
        "discovery_geometry_sha256": geometry_sha256,
        "confirmation_labels_sha256": confirmation_sha256,
        "acquisition_custody_chain": acquisition_custody_chain,
        "actual_lifetime_forward_examples": LIFETIME_FORWARD_EXAMPLES,
        "score": score,
        "claim_boundary": registration["claim_boundary"],
    }
    atomic_torch_save(score_record, paths["score"])
    custody = {
        "schema_version": "e31_1_order_repair_score_custody_v2",
        "status": "complete_write_once_v2_score",
        "registration_sha256": registration_sha256,
        "score_attempt_sha256": sha256(paths["score_attempt"]),
        "score_sha256": sha256(paths["score"]),
        "acquisition_custody_chain": acquisition_custody_chain,
        "atlas_verdict": score["atlas_verdict"],
        "actual_lifetime_forward_examples": LIFETIME_FORWARD_EXAMPLES,
        "post_discovery_repair_disclosed": True,
        "preseal_discovery_geometry_implementation_test_disclosed": True,
        "claim_boundary": registration["claim_boundary"],
    }
    atomic_json_save(custody, paths["score_custody"])
    return custody


def run_repair(
    *,
    registration_path: Path = V2_REGISTRATION_PATH,
    registration_sha256: str,
    output_directory: Path = V2_RUN_DIRECTORY,
) -> dict[str, Any]:
    if output_directory.resolve() != V2_RUN_DIRECTORY.resolve():
        raise RuntimeError("V2 output directory differs from the frozen path")
    registration = _load_registration(registration_path, registration_sha256)
    paths = {
        "attempt": output_directory / "recovery_attempt.json",
        "geometry": output_directory / "discovery_geometry.pt",
        "geometry_custody": output_directory / "discovery_geometry_custody.json",
        "authorization": output_directory / "confirmation_authorization.json",
        "revalidation": output_directory / "confirmation_clean_revalidation.pt",
        "revalidation_custody": output_directory
        / "confirmation_clean_revalidation_custody.json",
        "revalidation_failure": output_directory
        / "confirmation_clean_revalidation_failure.pt",
        "revalidation_failure_custody": output_directory
        / "confirmation_clean_revalidation_failure_custody.json",
        "confirmation": output_directory / "confirmation_labels.pt",
        "confirmation_custody": output_directory / "confirmation_custody.json",
        "score_attempt": output_directory / "score_attempt.json",
        "score": output_directory / "prospective_score.pt",
        "score_custody": output_directory / "score_custody.json",
    }
    for path in paths.values():
        ensure_write_once_available(path)
    atomic_json_save(
        {
            "schema_version": "e31_1_order_repair_attempt_v2",
            "status": "attempt_consumed_before_discovery_fit_or_model_load",
            "registration_sha256": registration_sha256,
            "source_code_sha256": registration["source_code_sha256"],
            "parent_artifacts": registration["v1_failure_state"]["artifacts"],
            "actual_lifetime_forward_examples_at_attempt": (
                V1_CONSUMED_FORWARD_EXAMPLES
            ),
            "planned_lifetime_forward_examples_on_success": (LIFETIME_FORWARD_EXAMPLES),
        },
        paths["attempt"],
    )
    attempt_sha256 = sha256(paths["attempt"])

    parent_registration, parent_clean, discovery = _load_validated_v1_inputs()
    discovery_panels = _discovery_panels(discovery)
    frozen_geometry = fit_discovery_geometry(
        discovery_panels=discovery_panels,
        primary_regime_id=PRIMARY_REGIME_ID,
    )
    if (
        frozen_geometry["regimes"][PRIMARY_REGIME_ID]["matched_controls"].get(
            "support_count"
        )
        != 10_000
    ):
        raise RuntimeError("V2 geometry omitted the 10,000 frozen controls")
    geometry_record = {
        "schema_version": "e31_1_order_repair_geometry_v2",
        "status": "outer_order_repaired_geometry_frozen_before_patched_confirmation",
        "registration_sha256": registration_sha256,
        "v1_discovery_labels_sha256": V1_ARTIFACTS["discovery_labels"][1],
        "repair": registration["repair"],
        "frozen_geometry": frozen_geometry,
    }
    atomic_torch_save(geometry_record, paths["geometry"])
    geometry_sha256 = sha256(paths["geometry"])
    atomic_json_save(
        {
            "schema_version": "e31_1_order_repair_geometry_custody_v2",
            "status": "geometry_and_controls_frozen_before_patched_confirmation",
            "registration_sha256": registration_sha256,
            "recovery_attempt_sha256": attempt_sha256,
            "v1_discovery_labels_sha256": V1_ARTIFACTS["discovery_labels"][1],
            "discovery_geometry_sha256": geometry_sha256,
            "matched_control_support_count": 10_000,
            "patched_confirmation_outcomes_used": False,
        },
        paths["geometry_custody"],
    )
    atomic_json_save(
        {
            "schema_version": "e31_1_order_repair_confirmation_authorization_v2",
            "status": "patched_confirmation_authorized_after_v2_geometry_seal",
            "registration_sha256": registration_sha256,
            "v1_registration_sha256": V1_ARTIFACTS["registration"][1],
            "v1_discovery_labels_sha256": V1_ARTIFACTS["discovery_labels"][1],
            "discovery_geometry_sha256": geometry_sha256,
            "geometry_custody_sha256": sha256(paths["geometry_custody"]),
            "adaptation_after_authorization_allowed": False,
        },
        paths["authorization"],
    )
    authorization_sha256 = sha256(paths["authorization"])

    _verify_dependencies()
    runner = _default_runner_factory(parent_registration)
    try:
        contexts, revalidation = _confirmation_clean_revalidation(
            runner, parent_registration, parent_clean
        )
    except Exception as error:
        failure = {
            "schema_version": "e31_1_confirmation_clean_revalidation_failure_v2",
            "status": "failed_before_any_patched_confirmation_response",
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "patched_confirmation_examples": 0,
        }
        atomic_torch_save(failure, paths["revalidation_failure"])
        atomic_json_save(
            {
                "schema_version": (
                    "e31_1_confirmation_clean_revalidation_failure_custody_v2"
                ),
                "status": "failed_before_any_patched_confirmation_response",
                "registration_sha256": registration_sha256,
                "authorization_sha256": authorization_sha256,
                "failure_sha256": sha256(paths["revalidation_failure"]),
                "patched_confirmation_examples": 0,
            },
            paths["revalidation_failure_custody"],
        )
        raise
    atomic_torch_save(revalidation, paths["revalidation"])
    revalidation_sha256 = sha256(paths["revalidation"])
    atomic_json_save(
        {
            "schema_version": "e31_1_confirmation_clean_revalidation_custody_v2",
            "status": "bitwise_match_to_v1_before_any_patched_confirmation",
            "registration_sha256": registration_sha256,
            "authorization_sha256": authorization_sha256,
            "revalidation_sha256": revalidation_sha256,
            "clean_forward_examples": 64,
            "patched_confirmation_examples_before_revalidation": 0,
            "actual_lifetime_forward_examples_after_revalidation": 124_800,
        },
        paths["revalidation_custody"],
    )
    revalidation_custody_sha256 = sha256(paths["revalidation_custody"])

    confirmation = acquire_prepared_split(
        runner,
        {"confirmation": contexts},
        split="confirmation",
        registration_sha256=V1_ARTIFACTS["registration"][1],
        clean_screen_sha256=V1_ARTIFACTS["clean_screen"][1],
    )
    validate_acquired_split(
        confirmation,
        registration=parent_registration,
        registration_sha256=V1_ARTIFACTS["registration"][1],
        split="confirmation",
        clean_screen_sha256=V1_ARTIFACTS["clean_screen"][1],
    )
    atomic_torch_save(confirmation, paths["confirmation"])
    confirmation_sha256 = sha256(paths["confirmation"])
    atomic_json_save(
        {
            "schema_version": "e31_1_order_repair_confirmation_custody_v2",
            "status": "unchanged_v1_confirmation_masks_acquired_write_once",
            "registration_sha256": registration_sha256,
            "v1_registration_sha256": V1_ARTIFACTS["registration"][1],
            "recovery_attempt_sha256": attempt_sha256,
            "discovery_geometry_sha256": geometry_sha256,
            "authorization_sha256": authorization_sha256,
            "revalidation_sha256": revalidation_sha256,
            "revalidation_custody_sha256": revalidation_custody_sha256,
            "confirmation_labels_sha256": confirmation_sha256,
            "confirmation_clean_reprepare_examples": 64,
            "confirmation_patched_examples": 248_832,
            "discovery_rerun_examples": 0,
            "actual_lifetime_forward_examples": LIFETIME_FORWARD_EXAMPLES,
        },
        paths["confirmation_custody"],
    )
    score_custody = _score_existing(
        registration_path=registration_path,
        registration_sha256=registration_sha256,
        output_directory=output_directory,
    )
    return {
        "status": "v2_order_repair_confirmation_and_score_complete",
        "registration_sha256": registration_sha256,
        "discovery_geometry_sha256": geometry_sha256,
        "confirmation_labels_sha256": confirmation_sha256,
        "score_sha256": score_custody["score_sha256"],
        "atlas_verdict": score_custody["atlas_verdict"],
        "actual_lifetime_forward_examples": LIFETIME_FORWARD_EXAMPLES,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--seal-registration", action="store_true")
    modes.add_argument("--confirm-run", action="store_true")
    modes.add_argument("--score-existing", action="store_true")
    parser.add_argument("--registration", type=Path, default=V2_REGISTRATION_PATH)
    parser.add_argument("--output", type=Path, default=V2_RUN_DIRECTORY)
    parser.add_argument("--registration-sha256")
    arguments = parser.parse_args()
    started = time.monotonic()
    if arguments.dry_run:
        report = dry_run()
    elif arguments.seal_registration:
        report = seal_registration(arguments.registration)
    else:
        if arguments.registration_sha256 is None:
            raise SystemExit("--registration-sha256 is required")
        if arguments.score_existing:
            report = _score_existing(
                registration_path=arguments.registration,
                registration_sha256=arguments.registration_sha256,
                output_directory=arguments.output,
            )
        else:
            report = run_repair(
                registration_path=arguments.registration,
                registration_sha256=arguments.registration_sha256,
                output_directory=arguments.output,
            )
    report["command_runtime_seconds"] = time.monotonic() - started
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
