# ruff: noqa: EXE002
"""Strict prospective registration payload for the E31.1 atlas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from experiments.e31_sparse_interactions.panel import (
    PROMPT_COUNTS,
    PROMPT_MANIFEST_SHA256,
    build_prompt_manifest,
    canonical_sha256,
    validate_prompt_manifest,
)
from experiments.e31_sparse_interactions.prospective_custody import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_COMPONENT_MAP,
    EXPECTED_MODEL_REVISION,
    EXPECTED_MODEL_SNAPSHOT,
    EXPECTED_RUNTIME_ENVIRONMENT,
    SOURCE_PATHS,
    valid_sha256,
    verify_file_hash,
    verify_registered_dependencies,
)
from experiments.e31_sparse_interactions.prospective_protocol import (
    PROTOCOL_SHA256,
    SPLIT_MANIFEST_SHA256,
    build_prospective_protocol,
    build_split_mask_manifest,
    validate_prospective_protocol,
)

REGISTRATION_SCHEMA_VERSION = "e31_1_prospective_sparse_atlas_registration_v1"
UNSEALED_STATUS = "prospective_unsealed_before_any_e31_1_model_response"
SEALED_STATUS = "prospective_sealed_before_any_e31_1_model_response"
TOKEN_CONTRACT_SCHEMA_VERSION = "e31_1_gpt2_token_contract_v1"
RESPONSE_COORDINATE = (
    "(patched_donor_minus_base_margin-clean_base_donor_minus_base_margin)"
    "/clean_endpoint"
)
PATCH_SEMANTICS = {
    "direction": "clean_donor_updates_into_base_recipient",
    "scope": "all_positions",
    "component_order": "layer_major_attention_then_mlp",
    "component_semantics": "attention_and_mlp_residual_updates",
    "selected_update_source": "aligned_clean_donor_output_projection_update",
    "later_components_recomputed": True,
    "response_coordinate": RESPONSE_COORDINATE,
    "unclipped": True,
    "directed_role_adapter": (
        "internally set name_a=base_target and name_b=donor_target for the sealed "
        "E29.15 runner, then restore every original registered panel field"
    ),
}
CLEAN_PANEL_GATE = {
    "per_row_before_any_patch": {
        "clean_logits_finite": True,
        "clean_logit_absolute_max": 10_000.0,
        "clean_endpoint_finite": True,
        "minimum_clean_endpoint_inclusive": 0.25,
    },
    "aggregate_before_any_patch": {
        "scope": "each_split_separately_after_all_48_contexts_are_prepared",
        "minimum_base_correct_fraction": 0.75,
        "minimum_donor_correct_fraction": 0.75,
        "mean_base_correct_minus_donor_margin_strictly_positive": True,
        "mean_donor_correct_minus_base_margin_strictly_positive": True,
    },
    "required_correct_count": "ceil(0.75 * registered_context_count)",
    "row_policy": "retain_and_score_every_registered_context",
    "outcome_based_row_filtering_allowed": False,
    "patched_call_before_full_clean_screen_allowed": False,
}
RESPONSE_IDENTITIES = {
    "clean_endpoint": (
        "clean_donor_donor_minus_base_margin-clean_base_donor_minus_base_margin"
    ),
    "raw_margin_shift_from_base": (
        "raw_donor_minus_base_margin-clean_base_donor_minus_base_margin"
    ),
    "normalized_response": "raw_margin_shift_from_base/clean_endpoint",
    "absolute_tolerance": 1e-12,
    "relative_tolerance": 1e-12,
    "clipping": False,
}
EXECUTION = {
    "device": "cpu",
    "dtype": "float64",
    "attention_implementation": "eager",
    "batch_size": 32,
    "local_files_only": True,
    "model_eval": True,
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
        "component_map",
        "patch_semantics",
        "clean_panel_gate",
        "response_identities",
        "protocol_schema_version",
        "protocol_sha256",
        "prompt_manifest_sha256",
        "split_manifest_sha256",
        "split_accounting",
        "prompt_token_contract",
        "prompt_token_contract_sha256",
        "clean_metadata_contract",
        "clean_metadata_contract_sha256",
        "source_code_sha256",
        "scoring_source_sha256",
        "output_contract",
        "outcome_claims",
        "outcome_custody",
        "protocol_deviations",
    }
)


def derive_prompt_token_contract(tokenizer: Any) -> dict[str, Any]:
    """Bind exact fresh prompts, token rows, differing positions, and targets."""
    splits: dict[str, Any] = {}
    for split in ("discovery", "confirmation"):
        records = build_prompt_manifest(split)
        checks = validate_prompt_manifest(tokenizer, records, split=split)
        rows = []
        for record, check in zip(records, checks, strict=True):
            base_ids = tokenizer.encode(record["base_prompt"], add_special_tokens=False)
            donor_ids = tokenizer.encode(
                record["donor_prompt"], add_special_tokens=False
            )
            rows.append(
                {
                    "pair_id": record["pair_id"],
                    "base_input_token_ids": list(base_ids),
                    "donor_input_token_ids": list(donor_ids),
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
            "context_count": len(records),
            "rows": rows,
        }
    return {
        "schema_version": TOKEN_CONTRACT_SCHEMA_VERSION,
        "tokenizer_json_sha256": EXPECTED_CHECKPOINT_SHA256["tokenizer.json"],
        "splits": splits,
    }


def validate_prompt_token_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the token contract without loading a tokenizer or model."""
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


def build_clean_metadata_contract(
    token_contract: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Build exact response-blind metadata projections for acquired clean rows."""
    validated = validate_prompt_token_contract(token_contract)
    output: dict[str, list[dict[str, Any]]] = {}
    fixed = {
        "answer_position": -1,
        "minimum_clean_endpoint": 0.25,
        "model_name_or_path": str(EXPECTED_MODEL_SNAPSHOT),
        "model_type": "gpt2",
        "model_layer_count": 12,
        "component_count": 24,
        "component_semantics": "attention_and_mlp_residual_updates",
        "patch_direction": "clean_donor_updates_into_base_recipient",
        "patch_scope": "all_positions",
        "response_coordinate": RESPONSE_COORDINATE,
        "attention_implementation": "eager",
        "device": "cpu",
        "model_dtype": "float64",
    }
    for split in ("discovery", "confirmation"):
        rows = []
        for prompt, tokens in zip(
            build_prompt_manifest(split),
            validated["splits"][split]["rows"],
            strict=True,
        ):
            rows.append(
                {
                    **deepcopy(prompt),
                    "base_target_token_id": tokens["base_target_token_id"],
                    "donor_target_token_id": tokens["donor_target_token_id"],
                    "token_length": tokens["token_length"],
                    "differing_token_position": tokens["differing_prompt_position"],
                    **fixed,
                }
            )
        output[split] = rows
    return output


def _protocol_and_hashes() -> tuple[dict[str, Any], str, dict[str, str]]:
    protocol = build_prospective_protocol()
    observed_protocol_sha = canonical_sha256(protocol)
    observed_splits = {
        split: canonical_sha256(build_split_mask_manifest(split))
        for split in ("discovery", "confirmation")
    }
    # During implementation the design may deliberately carry placeholders.
    # Such a payload can be dry-run and tested but cannot be sealed.
    if PROTOCOL_SHA256 != "TO_BE_PINNED":
        validate_prospective_protocol(protocol)
        if observed_protocol_sha != PROTOCOL_SHA256:
            raise RuntimeError("prospective protocol hash differs from its pin")
    for split, digest in observed_splits.items():
        pinned = SPLIT_MANIFEST_SHA256[split]
        if pinned != "TO_BE_PINNED" and digest != pinned:
            raise RuntimeError(f"{split} manifest hash differs from its pin")
    return protocol, observed_protocol_sha, observed_splits


def build_registration_record(
    *,
    source_code_sha256: Mapping[str, str],
    environment: Mapping[str, str],
    prompt_token_contract: Mapping[str, Any],
    sealed: bool = False,
) -> dict[str, Any]:
    """Build an exact unsealed dry-run or seal-ready registration payload."""
    token_contract = validate_prompt_token_contract(prompt_token_contract)
    if set(source_code_sha256) != set(SOURCE_PATHS) or any(
        not valid_sha256(value) for value in source_code_sha256.values()
    ):
        raise RuntimeError("prospective source hash closure changed")
    if dict(environment) != EXPECTED_RUNTIME_ENVIRONMENT:
        raise RuntimeError("prospective runtime environment changed")
    protocol, protocol_sha, split_hashes = _protocol_and_hashes()
    if sealed and (
        PROTOCOL_SHA256 == "TO_BE_PINNED"
        or any(value == "TO_BE_PINNED" for value in SPLIT_MANIFEST_SHA256.values())
    ):
        raise RuntimeError("cannot seal while protocol hashes are unpinned")
    clean_contract = build_clean_metadata_contract(token_contract)
    split_accounting = {
        split: {
            "context_count": PROMPT_COUNTS[split],
            "contrast_count": protocol["split_manifests"][split]["contrast_count"],
            "logical_cell_reference_count": protocol["split_manifests"][split][
                "logical_cell_reference_count"
            ],
            "unique_condition_count": protocol["split_manifests"][split][
                "unique_condition_count"
            ],
        }
        for split in ("discovery", "confirmation")
    }
    record = {
        "schema_version": REGISTRATION_SCHEMA_VERSION,
        "status": SEALED_STATUS if sealed else UNSEALED_STATUS,
        "question": (
            "Are exact task-, scale-, and hub-conditioned GPT-2-small sublayer "
            "interaction atlases sparse and stable on fresh directed IOI prompts?"
        ),
        "model_id": "gpt2",
        "model_snapshot": str(EXPECTED_MODEL_SNAPSHOT),
        "model_revision": EXPECTED_MODEL_REVISION,
        "checkpoint_sha256": dict(EXPECTED_CHECKPOINT_SHA256),
        "environment": dict(environment),
        "execution": deepcopy(EXECUTION),
        "component_map": deepcopy(EXPECTED_COMPONENT_MAP),
        "patch_semantics": deepcopy(PATCH_SEMANTICS),
        "clean_panel_gate": deepcopy(CLEAN_PANEL_GATE),
        "response_identities": deepcopy(RESPONSE_IDENTITIES),
        "protocol_schema_version": protocol["schema_version"],
        "protocol_sha256": protocol_sha,
        "prompt_manifest_sha256": dict(PROMPT_MANIFEST_SHA256),
        "split_manifest_sha256": split_hashes,
        "split_accounting": split_accounting,
        "prompt_token_contract": token_contract,
        "prompt_token_contract_sha256": canonical_sha256(token_contract),
        "clean_metadata_contract": clean_contract,
        "clean_metadata_contract_sha256": {
            split: canonical_sha256(clean_contract[split])
            for split in ("discovery", "confirmation")
        },
        "source_code_sha256": dict(source_code_sha256),
        "scoring_source_sha256": source_code_sha256["e31/prospective_scoring.py"],
        "output_contract": {
            "split_label_schema": "e31_1_prospective_response_panel_v1",
            "matrix_orientation": "condition_by_registered_context",
            "coordinates": [
                "normalized_responses",
                "raw_donor_minus_base_margins",
                "raw_margin_shifts_from_base",
            ],
            "all_three_coordinates_required": True,
            "unaggregated_condition_rows_required": True,
            "all_registered_context_rows_required": True,
        },
        "outcome_claims": {
            "exhaustive_atlas_and_registered_scoring": "authorized",
            "prediction": "deferred_to_e31_2",
            "topology": "deferred_to_e31_2",
            "direct_communication_from_pair_effect_alone": "not_authorized",
            "global_model_sparsity": "not_authorized",
        },
        "outcome_custody": {
            "registration_before_model_response": True,
            "attempt_before_tokenizer_or_model_load": True,
            "source_checkpoint_environment_reverified_before_each_acquisition": True,
            "source_checkpoint_environment_reverified_before_scoring": True,
            "discovery_and_confirmation_artifacts_write_once": True,
            "scoring_opens_only_hash_verified_registered_labels": True,
        },
        "protocol_deviations": [],
    }
    if set(record) != REGISTRATION_REQUIRED_KEYS:
        raise RuntimeError("prospective registration keys changed")
    return record


def validate_registration_payload(
    registration: Mapping[str, Any],
    *,
    verify_external_files: bool = True,
) -> dict[str, Any]:
    """Strictly validate one dry-run or sealed prospective registration."""
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
    for field in REGISTRATION_REQUIRED_KEYS - {"status"}:
        if copied[field] != expected[field]:
            label = field.replace("_", " ")
            raise RuntimeError(f"prospective {label} changed")
    if copied["prompt_token_contract_sha256"] != canonical_sha256(
        copied["prompt_token_contract"]
    ):
        raise RuntimeError("prospective token contract hash changed")
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
    "REGISTRATION_REQUIRED_KEYS",
    "REGISTRATION_SCHEMA_VERSION",
    "RESPONSE_COORDINATE",
    "RESPONSE_IDENTITIES",
    "SEALED_STATUS",
    "TOKEN_CONTRACT_SCHEMA_VERSION",
    "UNSEALED_STATUS",
    "build_clean_metadata_contract",
    "build_registration_record",
    "derive_prompt_token_contract",
    "load_verified_registration",
    "validate_prompt_token_contract",
    "validate_registration_payload",
]
