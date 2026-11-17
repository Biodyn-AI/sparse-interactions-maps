# ruff: noqa: EXE002
"""Complete an exploratory pair atlas by reusing sealed E29.15 rows.

This module never edits an E29 artifact.  It verifies the exact sealed parent,
maps its masks onto the E31 atlas, asks the unchanged E29.15 acquisition API
for only absent masks, and merges both sources into a new development-only
artifact.  The old prompt population has already been observed, so the output
is explicitly unsuitable for a confirmatory scientific claim.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_15_ioi_acquisition import (
    acquire_ioi_response_panel,
)
from experiments.e30_redundant_causal_interfaces.custody import sha256
from experiments.e31_sparse_interactions.protocol import (
    COMPONENT_COUNT,
    canonical_sha256,
    validate_exhaustive_pair_atlas,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARENT_REGISTRATION = (
    REPOSITORY_ROOT / "notes/E29.15-EQUAL-BUDGET-MECHANISM-DISCOVERY-REGISTRATION.json"
)
DEFAULT_PARENT_LABELS = (
    REPOSITORY_ROOT
    / "runs/e29_15_equal_budget_mechanism_discovery_v1/confirmation_labels.pt"
)
DEFAULT_PARENT_CUSTODY = (
    REPOSITORY_ROOT
    / "runs/e29_15_equal_budget_mechanism_discovery_v1/confirmation_custody.json"
)
PARENT_SHA256 = {
    "registration": "c3049fe9e3996850dc33564ea73deee4f890c89712860e09a236f05e487a5c5c",
    "labels": "762a39a43475083a154f0e700c6a7466ae7bc6208b022b32d8435aeafcf31079",
    "custody": "9119ea84d9ecf97f513cf8095961c6bee296a9016ada972a8e66a3693b793d1c",
}
CLAIM_BOUNDARY = {
    "scope": "exploratory_development_only",
    "reason": (
        "the E29.15 confirmation prompt population and some intervention "
        "responses were observed before this exhaustive atlas was specified"
    ),
    "confirmatory_claim_allowed": False,
    "novelty_or_superiority_claim_allowed": False,
    "required_next_step": (
        "freeze hypotheses and scoring before acquisition on a wholly fresh "
        "prompt population"
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _verify_exact_file(path: Path, expected_sha256: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(f"sealed parent hash mismatch for {path}")
    return observed


def _validate_parent_labels(
    labels: Mapping[str, Any],
    registration: Mapping[str, Any],
    custody: Mapping[str, Any],
) -> None:
    required = {
        "schema_version",
        "stage",
        "response_coordinate",
        "pair_ids",
        "prompt_records",
        "mask_ids",
        "masks",
        "component_map",
        "normalized_responses",
        "raw_donor_minus_base_margins",
        "raw_margin_shifts_from_base",
        "clean_base_donor_minus_base_margins",
        "clean_donor_donor_minus_base_margins",
        "clean_endpoints",
        "clean_metadata",
        "protocol_commitments",
        "all_integrity_gates_passed",
    }
    if not required <= set(labels):
        raise RuntimeError("sealed E29.15 labels omit required fields")
    if (
        labels["schema_version"] != "e29_15_ioi_response_panel_v1"
        or labels["stage"] != "confirmation"
        or labels["all_integrity_gates_passed"] is not True
    ):
        raise RuntimeError("sealed E29.15 label schema, stage, or integrity changed")
    masks = labels["masks"]
    responses = labels["normalized_responses"]
    context_count = len(registration["confirmation_prompt_records"])
    if (
        not isinstance(masks, torch.Tensor)
        or tuple(masks.shape) != (2_006, COMPONENT_COUNT)
        or not bool(torch.logical_or(masks == 0, masks == 1).all())
        or len(torch.unique(masks.to(torch.uint8), dim=0)) != len(masks)
    ):
        raise RuntimeError("sealed E29.15 confirmation masks changed")
    if (
        not isinstance(responses, torch.Tensor)
        or tuple(responses.shape) != (len(masks), context_count)
        or not bool(responses.isfinite().all())
    ):
        raise RuntimeError("sealed E29.15 confirmation responses changed")
    for field in (
        "raw_donor_minus_base_margins",
        "raw_margin_shifts_from_base",
    ):
        table = labels[field]
        if (
            not isinstance(table, torch.Tensor)
            or table.shape != responses.shape
            or not bool(table.isfinite().all())
        ):
            raise RuntimeError(f"sealed E29.15 {field} changed")
    if tuple(labels["prompt_records"]) != tuple(
        registration["confirmation_prompt_records"]
    ):
        raise RuntimeError("sealed E29.15 confirmation prompts changed")
    expected_pair_ids = tuple(
        record["pair_id"] for record in registration["confirmation_prompt_records"]
    )
    if tuple(labels["pair_ids"]) != expected_pair_ids:
        raise RuntimeError("sealed E29.15 pair order changed")
    expected_component_map = tuple(tuple(row) for row in registration["component_map"])
    if tuple(tuple(row) for row in labels["component_map"]) != expected_component_map:
        raise RuntimeError("sealed E29.15 component map changed")
    commitments = labels["protocol_commitments"]
    if (
        not isinstance(commitments, Mapping)
        or commitments.get("registration_sha256") != PARENT_SHA256["registration"]
        or commitments.get("prompt_panel_sha256")
        != registration["confirmation_prompt_records_sha256"]
        or commitments.get("mask_manifest_sha256")
        != custody["confirmation_manifest_sha256"]
        or commitments.get("nomination_seal_sha256")
        != custody["nomination_seal_sha256"]
    ):
        raise RuntimeError("sealed E29.15 protocol commitments changed")
    if (
        custody.get("registration_sha256") != PARENT_SHA256["registration"]
        or custody.get("labels_sha256") != PARENT_SHA256["labels"]
        or custody.get("prompt_records_sha256")
        != registration["confirmation_prompt_records_sha256"]
        or custody.get("all_integrity_gates_passed") is not True
    ):
        raise RuntimeError("sealed E29.15 custody links changed")


def load_sealed_e29_15_confirmation(
    *,
    registration_path: Path = DEFAULT_PARENT_REGISTRATION,
    labels_path: Path = DEFAULT_PARENT_LABELS,
    custody_path: Path = DEFAULT_PARENT_CUSTODY,
) -> dict[str, Any]:
    """Verify exact parent hashes and load the immutable confirmation panel."""
    paths = {
        "registration": registration_path,
        "labels": labels_path,
        "custody": custody_path,
    }
    for label, path in paths.items():
        _verify_exact_file(path, PARENT_SHA256[label])
    registration = _load_json(registration_path)
    custody = _load_json(custody_path)
    labels = torch.load(labels_path, map_location="cpu", weights_only=False)
    if not isinstance(labels, Mapping):
        raise TypeError("sealed E29.15 labels must contain a mapping")
    _validate_parent_labels(labels, registration, custody)
    return {
        "registration": registration,
        "labels": dict(labels),
        "custody": custody,
        "paths": {label: str(path.resolve()) for label, path in paths.items()},
        "sha256": dict(PARENT_SHA256),
    }


def build_parent_reuse_plan(
    parent: Mapping[str, Any], atlas: Mapping[str, Any]
) -> dict[str, Any]:
    """Map exact old mask rows to atlas rows; mark all other rows missing."""
    validated_atlas = validate_exhaustive_pair_atlas(atlas)
    labels = parent.get("labels")
    if not isinstance(labels, Mapping):
        raise TypeError("parent must contain validated labels")
    parent_masks = labels["masks"].detach().cpu().to(torch.uint8)
    atlas_masks = torch.tensor(
        [row["mask"] for row in validated_atlas["conditions"]], dtype=torch.uint8
    )
    parent_index_by_mask = {
        tuple(int(value) for value in row.tolist()): index
        for index, row in enumerate(parent_masks)
    }
    if len(parent_index_by_mask) != len(parent_masks):
        raise RuntimeError("parent mask rows are not unique")
    parent_row_by_atlas_index = torch.full((len(atlas_masks),), -1, dtype=torch.int64)
    for atlas_index, row in enumerate(atlas_masks):
        parent_row = parent_index_by_mask.get(
            tuple(int(value) for value in row.tolist())
        )
        if parent_row is not None:
            parent_row_by_atlas_index[atlas_index] = parent_row
    reused = torch.where(parent_row_by_atlas_index >= 0)[0]
    missing = torch.where(parent_row_by_atlas_index < 0)[0]
    condition_ids = tuple(row["condition_id"] for row in validated_atlas["conditions"])
    context_count = len(labels["pair_ids"])
    return {
        "atlas_manifest_sha256": canonical_sha256(validated_atlas),
        "atlas_mask_count": len(atlas_masks),
        "atlas_masks": atlas_masks,
        "condition_ids": condition_ids,
        "parent_row_by_atlas_index": parent_row_by_atlas_index,
        "reused_atlas_indices": reused,
        "reused_mask_count": len(reused),
        "reused_context_mask_cells": len(reused) * context_count,
        "missing_atlas_indices": missing,
        "missing_masks": atlas_masks[missing].clone(),
        "missing_condition_ids": tuple(condition_ids[index] for index in missing),
        "missing_mask_count": len(missing),
        "missing_context_mask_cells": len(missing) * context_count,
    }


def acquire_missing_masks(
    runner: Any,
    parent: Mapping[str, Any],
    atlas: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    acquire_fn: Callable[..., dict[str, Any]] = acquire_ioi_response_panel,
) -> dict[str, Any]:
    """Acquire precisely the atlas rows not present in the sealed parent."""
    validated_atlas = validate_exhaustive_pair_atlas(atlas)
    if plan.get("atlas_manifest_sha256") != canonical_sha256(validated_atlas):
        raise RuntimeError("reuse plan belongs to a different atlas")
    labels = parent["labels"]
    missing_masks = plan["missing_masks"]
    missing_ids = tuple(plan["missing_condition_ids"])
    if len(missing_masks) == 0:
        raise RuntimeError("the exploratory atlas unexpectedly has no missing rows")
    return acquire_fn(
        runner,
        labels["prompt_records"],
        missing_masks,
        stage="confirmation",
        mask_ids=missing_ids,
        nomination_provenance=(
            {
                "source": "e31_exhaustive_pair_atlas",
                "scope": "missing_masks_only",
                "claim": "exploratory_development_only",
            },
        ),
        protocol_commitments={
            "registration_sha256": PARENT_SHA256["registration"],
            "prompt_panel_sha256": parent["registration"][
                "confirmation_prompt_records_sha256"
            ],
            "parent_labels_sha256": PARENT_SHA256["labels"],
            "parent_custody_sha256": PARENT_SHA256["custody"],
            "atlas_manifest_sha256": canonical_sha256(validated_atlas),
        },
        smoke_masks=missing_masks[: min(64, len(missing_masks))],
    )


def _validate_missing_acquisition(
    missing: Mapping[str, Any],
    parent: Mapping[str, Any],
    atlas: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    labels = parent["labels"]
    required = {
        "schema_version",
        "stage",
        "response_coordinate",
        "pair_ids",
        "prompt_records",
        "mask_ids",
        "masks",
        "component_map",
        "normalized_responses",
        "raw_donor_minus_base_margins",
        "raw_margin_shifts_from_base",
        "clean_base_donor_minus_base_margins",
        "clean_donor_donor_minus_base_margins",
        "clean_endpoints",
        "clean_metadata",
        "protocol_commitments",
        "query_accounting",
        "all_integrity_gates_passed",
    }
    if not required <= set(missing):
        raise RuntimeError("missing-mask acquisition omits required fields")
    if (
        missing["schema_version"] != "e29_15_ioi_response_panel_v1"
        or missing["stage"] != "confirmation"
        or missing["all_integrity_gates_passed"] is not True
    ):
        raise RuntimeError("missing-mask acquisition integrity failed")
    if (
        tuple(missing["pair_ids"]) != tuple(labels["pair_ids"])
        or tuple(missing["prompt_records"]) != tuple(labels["prompt_records"])
        or tuple(tuple(row) for row in missing["component_map"])
        != tuple(tuple(row) for row in labels["component_map"])
        or missing["response_coordinate"] != labels["response_coordinate"]
    ):
        raise RuntimeError("missing-mask acquisition changed the parent population")
    if tuple(missing["mask_ids"]) != tuple(plan["missing_condition_ids"]):
        raise RuntimeError("missing-mask acquisition order changed")
    if not torch.equal(
        missing["masks"].detach().cpu().to(torch.uint8), plan["missing_masks"]
    ):
        raise RuntimeError("missing-mask acquisition evaluated the wrong masks")
    context_count = len(labels["pair_ids"])
    expected_shape = (plan["missing_mask_count"], context_count)
    for field in (
        "normalized_responses",
        "raw_donor_minus_base_margins",
        "raw_margin_shifts_from_base",
    ):
        table = missing[field]
        if (
            not isinstance(table, torch.Tensor)
            or tuple(table.shape) != expected_shape
            or not bool(table.isfinite().all())
        ):
            raise RuntimeError(f"missing-mask {field} shape or finiteness changed")
    for field in (
        "clean_base_donor_minus_base_margins",
        "clean_donor_donor_minus_base_margins",
        "clean_endpoints",
    ):
        if not torch.allclose(
            missing[field].double(), labels[field].double(), atol=1e-9, rtol=1e-9
        ):
            raise RuntimeError(f"missing-mask clean endpoint {field} changed")
    if missing["protocol_commitments"].get("atlas_manifest_sha256") != canonical_sha256(
        atlas
    ):
        raise RuntimeError("missing acquisition has the wrong atlas commitment")


def merge_exploratory_atlas(
    parent: Mapping[str, Any],
    atlas: Mapping[str, Any],
    plan: Mapping[str, Any],
    missing: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge old and new exact rows without aggregating any quartet or context."""
    validated_atlas = validate_exhaustive_pair_atlas(atlas)
    _validate_missing_acquisition(missing, parent, validated_atlas, plan)
    labels = parent["labels"]
    context_count = len(labels["pair_ids"])
    atlas_count = plan["atlas_mask_count"]
    reused_atlas = plan["reused_atlas_indices"]
    reused_parent = plan["parent_row_by_atlas_index"][reused_atlas]
    missing_atlas = plan["missing_atlas_indices"]
    merged_tables: dict[str, torch.Tensor] = {}
    for field in (
        "normalized_responses",
        "raw_donor_minus_base_margins",
        "raw_margin_shifts_from_base",
    ):
        table = torch.empty((atlas_count, context_count), dtype=torch.float64)
        table[reused_atlas] = labels[field][reused_parent].double()
        table[missing_atlas] = missing[field].double()
        if not bool(table.isfinite().all()):
            raise RuntimeError(f"merged {field} contains a non-finite value")
        merged_tables[field] = table
    source_kind = [""] * atlas_count
    source_row_index = torch.empty(atlas_count, dtype=torch.int64)
    for atlas_index, parent_index in zip(
        reused_atlas.tolist(), reused_parent.tolist(), strict=True
    ):
        source_kind[atlas_index] = "sealed_e29_15_confirmation"
        source_row_index[atlas_index] = parent_index
    for new_index, atlas_index in enumerate(missing_atlas.tolist()):
        source_kind[atlas_index] = "new_e31_missing_mask_acquisition"
        source_row_index[atlas_index] = new_index
    if any(not value for value in source_kind):
        raise RuntimeError("an atlas row lacks provenance")
    return {
        "schema_version": "e31_1_exploratory_development_pair_atlas_v1",
        "status": "complete_exploratory_development_atlas",
        "claim_boundary": deepcopy(CLAIM_BOUNDARY),
        "atlas_manifest_sha256": canonical_sha256(validated_atlas),
        "backgrounds": deepcopy(validated_atlas["backgrounds"]),
        "conditions": deepcopy(validated_atlas["conditions"]),
        "quartets": deepcopy(validated_atlas["quartets"]),
        "pair_ids": tuple(labels["pair_ids"]),
        "prompt_records": tuple(deepcopy(labels["prompt_records"])),
        "component_map": tuple(tuple(row) for row in labels["component_map"]),
        "response_coordinate": labels["response_coordinate"],
        "mask_ids": tuple(plan["condition_ids"]),
        "masks": plan["atlas_masks"].clone(),
        **merged_tables,
        "clean_base_donor_minus_base_margins": labels[
            "clean_base_donor_minus_base_margins"
        ].clone(),
        "clean_donor_donor_minus_base_margins": labels[
            "clean_donor_donor_minus_base_margins"
        ].clone(),
        "clean_endpoints": labels["clean_endpoints"].clone(),
        "clean_metadata": tuple(deepcopy(labels["clean_metadata"])),
        "row_provenance": {
            "source_kind": tuple(source_kind),
            "source_row_index": source_row_index,
        },
        "provenance": {
            "parent_paths": deepcopy(parent["paths"]),
            "parent_sha256": deepcopy(parent["sha256"]),
            "parent_mask_count": len(labels["masks"]),
            "reused_mask_count": plan["reused_mask_count"],
            "newly_acquired_mask_count": plan["missing_mask_count"],
            "reused_context_mask_cells": plan["reused_context_mask_cells"],
            "new_context_mask_cells": plan["missing_context_mask_cells"],
            "new_acquisition_query_accounting": deepcopy(missing["query_accounting"]),
            "new_acquisition_protocol_commitments": deepcopy(
                missing["protocol_commitments"]
            ),
            "new_acquisition_integrity": {
                "clean_panel_viability": deepcopy(missing.get("clean_panel_viability")),
                "batch_invariance_smoke": deepcopy(
                    missing.get("batch_invariance_smoke")
                ),
                "numerical_summary": deepcopy(missing.get("numerical_summary")),
                "all_integrity_gates_passed": missing["all_integrity_gates_passed"],
            },
        },
        "all_rows_have_exact_source_provenance": True,
        "all_integrity_gates_passed": True,
    }


__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_PARENT_CUSTODY",
    "DEFAULT_PARENT_LABELS",
    "DEFAULT_PARENT_REGISTRATION",
    "PARENT_SHA256",
    "acquire_missing_masks",
    "build_parent_reuse_plan",
    "load_sealed_e29_15_confirmation",
    "merge_exploratory_atlas",
]
