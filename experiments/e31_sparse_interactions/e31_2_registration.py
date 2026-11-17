# ruff: noqa: EXE002
"""Strict prospective registration for the E31.2 mixed-resolution atlas."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from experiments.e31_sparse_interactions.e31_2_custody import (
    E31_1_PARENT_SCORE_SHA256,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_MODEL_REVISION,
    EXPECTED_MODEL_SNAPSHOT,
    EXPECTED_RUNTIME_ENVIRONMENT,
    REGISTERED_PARENT_PROVENANCE,
    SOURCE_PATHS,
    valid_sha256,
    verify_file_hash,
    verify_registered_dependencies,
    write_once_reservation_path,
)
from experiments.e31_sparse_interactions.mixed_resolution_panel import (
    PROMPT_COUNTS,
    PROMPT_MANIFEST_SHA256,
    build_prompt_manifest,
    canonical_sha256,
    validate_prompt_manifest,
)
from experiments.e31_sparse_interactions.mixed_resolution_protocol import (
    ATLAS_MANIFEST_SHA256,
    BACKGROUND_MANIFEST_SHA256,
    PROTOCOL_SHA256,
    build_atlas_manifest,
    build_protocol,
    validate_protocol,
)

REGISTRATION_SCHEMA_VERSION = "e31_2_mixed_resolution_registration_v1"
UNSEALED_STATUS = "prospective_unsealed_before_any_e31_2_model_response"
SEALED_STATUS = "prospective_sealed_before_any_e31_2_model_response"
TOKEN_CONTRACT_SCHEMA_VERSION = "e31_2_gpt2_token_contract_v1"
REGISTERED_REGISTRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "notes/E31.2-MIXED-RESOLUTION-REGISTRATION.json"
).resolve()
REGISTERED_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "runs/e31_2_mixed_resolution_v1"
).resolve()
REGISTERED_OUTPUT_FILENAMES = (
    "global_attempt.json",
    "all_registered_clean_screen.pt",
    "all_registered_clean_screen_custody.json",
    "discovery_labels.pt",
    "discovery_custody.json",
    "discovery_geometry.pt",
    "discovery_geometry_custody.json",
    "confirmation_reconstruction_manifest.json",
    "confirmation_authorization.json",
    "confirmation_labels.pt",
    "confirmation_custody.json",
    "score_attempt.json",
    "prospective_score.pt",
    "score_custody.json",
    "run_failure_custody.json",
)

EXECUTION = {
    "device": "cpu",
    "dtype": "float64",
    "attention_implementation": "eager",
    "batch_size": 32,
    "local_files_only": True,
    "model_eval": True,
}
PATCH_SEMANTICS = {
    "direction": "clean_donor_updates_into_base_recipient",
    "scope": "all_positions",
    "coarse_component_order": "layer_major_attention_then_mlp",
    "coarse_component_count": 24,
    "fine_head_order": "layer_major_head_0_through_11",
    "fine_head_count": 144,
    "fine_patch_site": "merged_attention_head_vector_immediately_before_c_proj",
    "fine_projection_semantics": (
        "bias_free_per_head_W_O_contributions_are_summed_and_c_proj_bias_added_once"
    ),
    "later_components_recomputed": True,
    "all_head_to_whole_attention_equivalence_atol_rtol": 1e-9,
    "response_coordinate": (
        "(patched_donor_minus_base_margin-clean_base_donor_minus_base_margin)"
        "/clean_endpoint"
    ),
    "clipping": False,
}
CLEAN_PANEL_GATE = {
    "scope": "all_48_registered_rows_prepared_and_screened_before_any_patch",
    "per_row": {
        "clean_logits_finite": True,
        "clean_logit_absolute_max": 10_000.0,
        "clean_endpoint_finite": True,
        "minimum_clean_endpoint_inclusive": 0.25,
        "head_vector_norm_diagnostics_pass": True,
    },
    "per_split_after_all_rows_prepared": {
        "minimum_base_correct_fraction": 0.75,
        "minimum_donor_correct_fraction": 0.75,
        "mean_base_correct_minus_donor_margin_strictly_positive": True,
        "mean_donor_correct_minus_base_margin_strictly_positive": True,
    },
    "row_policy": "retain_and_score_every_registered_context",
    "outcome_based_row_filtering_allowed": False,
}
RESPONSE_IDENTITIES = {
    "clean_endpoint": (
        "clean_donor_donor_minus_base_margin-clean_base_donor_minus_base_margin"
    ),
    "raw_margin_shift_from_base": (
        "raw_donor_minus_base_margin-clean_base_donor_minus_base_margin"
    ),
    "normalized_response": "raw_margin_shift_from_base/clean_endpoint",
    "primary_interaction": "R11-R10-R01+R00",
    "absolute_tolerance": 1e-12,
    "relative_tolerance": 1e-12,
    "clipping": False,
}

REGISTRATION_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "question",
        "model_id",
        "model_snapshot",
        "model_revision",
        "checkpoint_sha256",
        "environment",
        "execution",
        "patch_semantics",
        "clean_panel_gate",
        "response_identities",
        "protocol_schema_version",
        "protocol_sha256",
        "prompt_manifest_sha256",
        "background_manifest_sha256",
        "atlas_manifest_sha256",
        "query_accounting",
        "batch_invariance_smoke",
        "prompt_token_contract",
        "prompt_token_contract_sha256",
        "source_code_sha256",
        "scoring_source_sha256",
        "runner_source_sha256",
        "acquisition_source_sha256",
        "parent_provenance",
        "outcome_custody",
        "output_contract",
        "claim_boundary",
        "registered_paths",
        "protocol_deviations",
    }
)


def build_discovery_smoke_contract() -> dict[str, Any]:
    """Freeze 16 response-blind D masks for the batch 1 versus 32 smoke."""
    atlas = build_atlas_manifest("discovery")
    ranked = sorted(
        atlas["conditions"],
        key=lambda row: (
            hashlib.sha256(
                (
                    "e31.2|batch-smoke|discovery|context-0|" + str(row["condition_id"])
                ).encode("ascii")
            ).hexdigest(),
            int(row["condition_index"]),
        ),
    )[:16]
    indices = [int(row["condition_index"]) for row in ranked]
    ids = [str(row["condition_id"]) for row in ranked]
    return {
        "selection_rule": (
            "take the first 16 discovery condition IDs ranked by lowercase SHA256 "
            "of e31.2|batch-smoke|discovery|context-0|condition_id, tie condition_index"
        ),
        "context_index": 0,
        "pair_id": build_prompt_manifest("discovery")[0]["pair_id"],
        "condition_indices": indices,
        "condition_ids": ids,
        "selected_rows_sha256": canonical_sha256(
            [
                {"condition_index": index, "condition_id": row_id}
                for index, row_id in zip(indices, ids, strict=True)
            ]
        ),
        "reference_batch_size": 1,
        "candidate_batch_size": 32,
        "absolute_tolerance": 1e-9,
        "relative_tolerance": 1e-9,
        "physical_context_mask_cells": 32,
        "must_finish_before_full_discovery_atlas": True,
    }


def build_registered_paths() -> dict[str, Any]:
    """Bind the sole registration and run paths, including reservations."""
    artifact_paths = {
        filename: str((REGISTERED_OUTPUT_DIRECTORY / filename).resolve())
        for filename in REGISTERED_OUTPUT_FILENAMES
    }
    return {
        "registration": str(REGISTERED_REGISTRATION_PATH),
        "registration_write_once_reservation": str(
            write_once_reservation_path(REGISTERED_REGISTRATION_PATH).resolve()
        ),
        "output_directory": str(REGISTERED_OUTPUT_DIRECTORY),
        "artifact_paths": artifact_paths,
        "artifact_write_once_reservations": {
            filename: str(write_once_reservation_path(Path(path)).resolve())
            for filename, path in artifact_paths.items()
        },
        "alternate_registration_or_output_paths_allowed": False,
        "output_directory_must_be_pristine_at_seal": True,
    }


def derive_prompt_token_contract(tokenizer: Any) -> dict[str, Any]:
    """Bind exact prompts, token rows, differing positions, and targets."""
    splits = {}
    for split in ("discovery", "confirmation"):
        records = build_prompt_manifest(split)
        checks = validate_prompt_manifest(tokenizer, records, split=split)
        rows = []
        for record, check in zip(records, checks, strict=True):
            rows.append(
                {
                    "pair_id": record["pair_id"],
                    "base_input_token_ids": list(
                        tokenizer.encode(
                            record["base_prompt"], add_special_tokens=False
                        )
                    ),
                    "donor_input_token_ids": list(
                        tokenizer.encode(
                            record["donor_prompt"], add_special_tokens=False
                        )
                    ),
                    "token_length": int(check["token_count"]),
                    "differing_prompt_position": int(
                        check["differing_prompt_position"]
                    ),
                    "base_target_token_id": int(check["base_target_token_id"]),
                    "donor_target_token_id": int(check["donor_target_token_id"]),
                }
            )
        splits[split] = {
            "prompt_manifest_sha256": PROMPT_MANIFEST_SHA256[split],
            "context_count": PROMPT_COUNTS[split],
            "rows": rows,
        }
    return {
        "schema_version": TOKEN_CONTRACT_SCHEMA_VERSION,
        "tokenizer_json_sha256": EXPECTED_CHECKPOINT_SHA256["tokenizer.json"],
        "splits": splits,
    }


def validate_prompt_token_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise TypeError("prompt token contract must be a mapping")
    copied = deepcopy(dict(contract))
    if (
        copied.get("schema_version") != TOKEN_CONTRACT_SCHEMA_VERSION
        or copied.get("tokenizer_json_sha256")
        != EXPECTED_CHECKPOINT_SHA256["tokenizer.json"]
        or set(copied.get("splits", {})) != {"discovery", "confirmation"}
    ):
        raise RuntimeError("prompt token contract header changed")
    for split in ("discovery", "confirmation"):
        split_contract = copied["splits"][split]
        records = build_prompt_manifest(split)
        rows = split_contract.get("rows")
        if (
            split_contract.get("prompt_manifest_sha256")
            != PROMPT_MANIFEST_SHA256[split]
            or split_contract.get("context_count") != PROMPT_COUNTS[split]
            or not isinstance(rows, Sequence)
            or len(rows) != len(records)
        ):
            raise RuntimeError(f"{split} token contract header changed")
        for record, row in zip(records, rows, strict=True):
            required = {
                "pair_id",
                "base_input_token_ids",
                "donor_input_token_ids",
                "token_length",
                "differing_prompt_position",
                "base_target_token_id",
                "donor_target_token_id",
            }
            if not isinstance(row, Mapping) or set(row) != required:
                raise RuntimeError(f"{split} token row keys changed")
            base = row["base_input_token_ids"]
            donor = row["donor_input_token_ids"]
            if not isinstance(base, Sequence) or not isinstance(donor, Sequence):
                raise TypeError(f"{split} token arrays changed")
            differing = [
                index
                for index, values in enumerate(zip(base, donor, strict=True))
                if values[0] != values[1]
            ]
            if (
                row["pair_id"] != record["pair_id"]
                or len(base) != len(donor)
                or len(base) != row["token_length"]
                or differing != [row["differing_prompt_position"]]
                or row["base_target_token_id"] == row["donor_target_token_id"]
            ):
                raise RuntimeError(f"{split} token row alignment changed")
    return copied


def build_registration_record(
    *,
    source_code_sha256: Mapping[str, str],
    environment: Mapping[str, str],
    prompt_token_contract: Mapping[str, Any],
    sealed: bool = False,
) -> dict[str, Any]:
    """Build a fully bound unsealed dry-run or seal-ready record."""
    token_contract = validate_prompt_token_contract(prompt_token_contract)
    if set(source_code_sha256) != set(SOURCE_PATHS) or any(
        not valid_sha256(value) for value in source_code_sha256.values()
    ):
        raise RuntimeError("prospective source hash closure changed")
    if dict(environment) != EXPECTED_RUNTIME_ENVIRONMENT:
        raise RuntimeError("prospective runtime environment changed")
    protocol = validate_protocol(build_protocol())
    if (
        protocol["parent_selection_ancestry"]["e31_1_v2_score_artifact_sha256"]
        != E31_1_PARENT_SCORE_SHA256
    ):
        raise RuntimeError("protocol parent score ancestry changed")
    record = {
        "schema_version": REGISTRATION_SCHEMA_VERSION,
        "status": SEALED_STATUS if sealed else UNSEALED_STATUS,
        "question": protocol["question"],
        "model_id": "gpt2",
        "model_snapshot": str(EXPECTED_MODEL_SNAPSHOT),
        "model_revision": EXPECTED_MODEL_REVISION,
        "checkpoint_sha256": dict(EXPECTED_CHECKPOINT_SHA256),
        "environment": dict(environment),
        "execution": deepcopy(EXECUTION),
        "patch_semantics": deepcopy(PATCH_SEMANTICS),
        "clean_panel_gate": deepcopy(CLEAN_PANEL_GATE),
        "response_identities": deepcopy(RESPONSE_IDENTITIES),
        "protocol_schema_version": protocol["schema_version"],
        "protocol_sha256": PROTOCOL_SHA256,
        "prompt_manifest_sha256": dict(PROMPT_MANIFEST_SHA256),
        "background_manifest_sha256": dict(BACKGROUND_MANIFEST_SHA256),
        "atlas_manifest_sha256": dict(ATLAS_MANIFEST_SHA256),
        "query_accounting": deepcopy(protocol["query_accounting"]),
        "batch_invariance_smoke": build_discovery_smoke_contract(),
        "prompt_token_contract": token_contract,
        "prompt_token_contract_sha256": canonical_sha256(token_contract),
        "source_code_sha256": dict(source_code_sha256),
        "scoring_source_sha256": source_code_sha256["e31/mixed_resolution_scoring.py"],
        "runner_source_sha256": source_code_sha256["e31/e31_2_head_patch_runner.py"],
        "acquisition_source_sha256": source_code_sha256[
            "e31/e31_2_head_acquisition.py"
        ],
        "parent_provenance": deepcopy(REGISTERED_PARENT_PROVENANCE),
        "outcome_custody": {
            "registration_before_any_e31_2_outcome": True,
            "global_attempt_before_tokenizer_or_model_load": True,
            "all_48_clean_rows_precede_patches": True,
            "duplicate_clean_forwards_allowed": False,
            "discovery_smoke_precedes_full_discovery_atlas": True,
            "discovery_labels_write_once": True,
            "discovery_geometry_and_10000_controls_frozen_before_confirmation": True,
            "confirmation_authorization_precedes_any_confirmation_patch": True,
            "score_attempt_precedes_score_stage_hash_verification_or_opening": True,
            "geometry_replayed_exactly_before_score": True,
            "source_checkpoint_environment_reverified_before_every_stage": True,
        },
        "output_contract": {
            "response_dtype": "float64",
            "mask_dtype": "uint8",
            "fine_quartet_shape": {
                "discovery": [3, 144, 16, 4, 4],
                "confirmation": [3, 144, 32, 4, 4],
            },
            "direct_reconstruction_quartet_shape": [3, 2, 32, 4, 4],
            "all12_reference_quartet_shape": [3, 32, 4, 4],
            "all_raw_and_normalized_coordinates_required": True,
            "all_registered_rows_retained": True,
        },
        "claim_boundary": {
            "parent_selection_was_posthoc_e31_1": True,
            "e31_1_confirmation_was_inspected": True,
            "no_e31_2_outcomes_used_for_registration": True,
            "prospective_e31_2_head_localization_and_reconstruction_only": True,
            "task_model_architecture_universality_not_authorized": True,
        },
        "registered_paths": build_registered_paths(),
        "protocol_deviations": [],
    }
    if set(record) != REGISTRATION_REQUIRED_KEYS:
        raise RuntimeError("prospective registration keys changed")
    return record


def validate_registration_payload(
    registration: Mapping[str, Any], *, verify_external_files: bool = True
) -> dict[str, Any]:
    if not isinstance(registration, Mapping):
        raise TypeError("registration must be a mapping")
    copied = deepcopy(dict(registration))
    if set(copied) != REGISTRATION_REQUIRED_KEYS:
        raise RuntimeError("prospective registration keys changed")
    if copied["status"] not in {UNSEALED_STATUS, SEALED_STATUS}:
        raise RuntimeError("prospective registration status changed")
    expected = build_registration_record(
        source_code_sha256=copied["source_code_sha256"],
        environment=copied["environment"],
        prompt_token_contract=copied["prompt_token_contract"],
        sealed=copied["status"] == SEALED_STATUS,
    )
    if copied != expected:
        changed = [key for key in expected if copied.get(key) != expected[key]]
        label = changed[0].replace("_", " ") if changed else "payload"
        raise RuntimeError(f"prospective {label} changed")
    if verify_external_files:
        verify_registered_dependencies(copied)
        for label, digest in copied["source_code_sha256"].items():
            verify_file_hash(SOURCE_PATHS[label], digest)
    return copied


def load_verified_registration(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not valid_sha256(expected_sha256):
        raise ValueError("expected registration SHA-256 is invalid")
    verify_file_hash(path, expected_sha256)
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("registration artifact must be a JSON object")
    validated = validate_registration_payload(value)
    if validated["status"] != SEALED_STATUS:
        raise RuntimeError("acquisition requires a sealed registration")
    return validated


__all__ = [
    "CLEAN_PANEL_GATE",
    "EXECUTION",
    "PATCH_SEMANTICS",
    "REGISTERED_OUTPUT_DIRECTORY",
    "REGISTERED_OUTPUT_FILENAMES",
    "REGISTERED_REGISTRATION_PATH",
    "REGISTRATION_REQUIRED_KEYS",
    "REGISTRATION_SCHEMA_VERSION",
    "RESPONSE_IDENTITIES",
    "SEALED_STATUS",
    "TOKEN_CONTRACT_SCHEMA_VERSION",
    "UNSEALED_STATUS",
    "build_discovery_smoke_contract",
    "build_registered_paths",
    "build_registration_record",
    "derive_prompt_token_contract",
    "load_verified_registration",
    "validate_prompt_token_contract",
    "validate_registration_payload",
]
