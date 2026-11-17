# ruff: noqa: EXE002
"""Directed-role adapter and exact prospective E31.1 acquisition primitives."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Literal

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_15_ioi_acquisition import (
    CleanPanelViabilityError,
    clean_panel_viability_report,
)
from experiments.e29_gpt2_stable_head_interactions.e29_15_ioi_patch_runner import (
    E2915IOIPatchRunner,
    PreparedIOIPatchContext,
)
from experiments.e31_sparse_interactions.panel import (
    PROMPT_RECORD_KEYS,
    build_prompt_manifest,
)
from experiments.e31_sparse_interactions.prospective_protocol import (
    DESIGN_DOMAIN,
    build_prospective_protocol,
    build_split_mask_manifest,
    canonical_sha256,
)
from experiments.e31_sparse_interactions.prospective_registration import (
    CLEAN_PANEL_GATE,
    RESPONSE_COORDINATE,
    validate_registration_payload,
)

Split = Literal["discovery", "confirmation"]
SPLITS = ("discovery", "confirmation")
BATCH_SMOKE_MASK_COUNT = 16
BATCH_SMOKE_REFERENCE_BATCH_SIZE = 1
BATCH_SMOKE_CANDIDATE_BATCH_SIZE = 32
BATCH_SMOKE_ATOL = 1e-9
BATCH_SMOKE_RTOL = 1e-9


def adapt_directed_ioi_record(record: Mapping[str, Any]) -> dict[str, str]:
    """Return an internal copy satisfying the sealed runner's role invariants.

    Original prompts, targets, and registered name fields are never mutated.
    The sealed E29.15 runner only needs its internal ``name_a`` to denote the
    base target and ``name_b`` to denote the donor target.
    """
    if not isinstance(record, Mapping) or tuple(record) != PROMPT_RECORD_KEYS:
        raise ValueError("directed IOI record keys or order changed")
    original = {key: str(record[key]) for key in PROMPT_RECORD_KEYS}
    if any(not value for value in original.values()):
        raise ValueError("directed IOI records require nonempty strings")
    if {original["base_target"], original["donor_target"]} != {
        original["name_a"],
        original["name_b"],
    }:
        raise ValueError("directed targets must be exactly the registered name pair")
    if (
        original["base_repeated_subject"] != original["donor_target"]
        or original["donor_repeated_subject"] != original["base_target"]
    ):
        raise ValueError("directed repeated-subject roles changed")
    adapted = dict(original)
    adapted["name_a"] = original["base_target"]
    adapted["name_b"] = original["donor_target"]
    return adapted


class E311DirectedIOIPatchRunner(E2915IOIPatchRunner):
    """Use the unchanged E29.15 patch engine for both registered directions."""

    def prepare_context(self, record: Mapping[str, Any]) -> PreparedIOIPatchContext:
        original = {key: str(record[key]) for key in PROMPT_RECORD_KEYS}
        adapted = adapt_directed_ioi_record(record)
        prepared = super().prepare_context(adapted)
        metadata = dict(prepared.clean_metadata)
        for field in ("base_prompt", "donor_prompt", "base_target", "donor_target"):
            if metadata[field] != original[field]:
                raise RuntimeError("directed role adapter changed prompt or target")
        if (
            prepared.base_target_token_id != metadata["base_target_token_id"]
            or prepared.donor_target_token_id != metadata["donor_target_token_id"]
        ):
            raise RuntimeError("directed role adapter changed target token IDs")
        metadata.update(original)
        return replace(
            prepared,
            clean_metadata=MappingProxyType(metadata),
        )


def masks_from_split_manifest(
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], torch.Tensor]:
    """Return exact ordered, unique binary masks from one split manifest."""
    if not isinstance(manifest, Mapping):
        raise TypeError("split manifest must be a mapping")
    split = manifest.get("split")
    if split not in SPLITS or dict(manifest) != build_split_mask_manifest(split):
        raise ValueError("split mask manifest differs from the prospective protocol")
    identifiers = []
    rows = []
    for expected_index, condition in enumerate(manifest["conditions"]):
        mask = tuple(int(value) for value in condition["mask"])
        if (
            condition["condition_index"] != expected_index
            or condition["mask_sha256"] != canonical_sha256(list(mask))
            or len(mask) != 24
            or any(value not in (0, 1) for value in mask)
            or tuple(index for index, value in enumerate(mask) if value)
            != tuple(condition["selected_components"])
        ):
            raise RuntimeError("a registered condition row changed")
        identifiers.append(str(condition["condition_id"]))
        rows.append(mask)
    masks = torch.tensor(rows, dtype=torch.uint8)
    if len(set(identifiers)) != len(identifiers) or len(
        torch.unique(masks, dim=0)
    ) != len(masks):
        raise RuntimeError("registered condition identifiers or masks are not unique")
    return tuple(identifiers), masks


def build_batch_smoke_manifest(
    discovery_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Freeze 16 response-blind discovery rows for one batch-size smoke."""
    identifiers, _masks = masks_from_split_manifest(discovery_manifest)
    conditions = discovery_manifest["conditions"]

    def rank(index: int) -> tuple[str, int]:
        condition = conditions[index]
        payload = (
            DESIGN_DOMAIN
            + "|batch-smoke|"
            + identifiers[index]
            + "|"
            + str(condition["mask_sha256"])
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), index

    selected = sorted(range(len(conditions)), key=rank)[:BATCH_SMOKE_MASK_COUNT]
    return tuple(
        {
            "smoke_index": smoke_index,
            "condition_index": condition_index,
            "condition_id": identifiers[condition_index],
            "mask_sha256": conditions[condition_index]["mask_sha256"],
            "selection_sha256": rank(condition_index)[0],
        }
        for smoke_index, condition_index in enumerate(selected)
    )


def _clean_screen_record(
    prepared: Mapping[str, Sequence[PreparedIOIPatchContext]],
) -> dict[str, Any]:
    reports = {split: clean_panel_viability_report(prepared[split]) for split in SPLITS}
    passed = all(report["passed"] for report in reports.values())
    return {
        "schema_version": "e31_1_all_registered_clean_screen_v1",
        "status": (
            "all_48_clean_rows_passed_before_any_patch"
            if passed
            else "all_48_clean_rows_prepared_but_clean_gate_failed"
        ),
        "split_order": list(SPLITS),
        "total_context_count": sum(len(prepared[split]) for split in SPLITS),
        "total_clean_forward_examples": 2
        * sum(len(prepared[split]) for split in SPLITS),
        "reports": reports,
        "clean_metadata": {
            split: [dict(context.clean_metadata) for context in prepared[split]]
            for split in SPLITS
        },
        "clean_base_donor_minus_base_margins": {
            split: torch.tensor(
                [
                    context.clean_base_donor_minus_base_margin
                    for context in prepared[split]
                ],
                dtype=torch.float64,
            )
            for split in SPLITS
        },
        "clean_donor_donor_minus_base_margins": {
            split: torch.tensor(
                [
                    context.clean_donor_donor_minus_base_margin
                    for context in prepared[split]
                ],
                dtype=torch.float64,
            )
            for split in SPLITS
        },
        "clean_endpoints": {
            split: torch.tensor(
                [context.clean_endpoint for context in prepared[split]],
                dtype=torch.float64,
            )
            for split in SPLITS
        },
        "gate": deepcopy(CLEAN_PANEL_GATE),
        "row_policy": "retain_and_score_every_registered_context",
        "all_splits_passed": passed,
    }


def screen_all_registered_clean_rows(
    runner: E311DirectedIOIPatchRunner,
) -> tuple[dict[str, tuple[PreparedIOIPatchContext, ...]], dict[str, Any]]:
    """Prepare and screen all 48 rows before exposing a patched response."""
    if not isinstance(runner, E311DirectedIOIPatchRunner):
        raise TypeError("runner must be an E311DirectedIOIPatchRunner")
    prepared = {
        split: tuple(
            runner.prepare_context(record) for record in build_prompt_manifest(split)
        )
        for split in SPLITS
    }
    record = _clean_screen_record(prepared)
    if record["total_context_count"] != 48:
        raise RuntimeError("prospective clean screen did not prepare all 48 rows")
    if not record["all_splits_passed"]:
        raise CleanPanelViabilityError(record)
    return prepared, record


def validate_response_identities(labels: Mapping[str, Any]) -> None:
    """Verify exact clean, raw-margin, raw-shift, and normalized identities."""
    required = {
        "normalized_responses",
        "raw_donor_minus_base_margins",
        "raw_margin_shifts_from_base",
        "clean_base_donor_minus_base_margins",
        "clean_donor_donor_minus_base_margins",
        "clean_endpoints",
    }
    if not isinstance(labels, Mapping) or not required <= set(labels):
        raise RuntimeError("response artifact omits exact coordinate tables")
    for name in required:
        value = labels[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
            raise RuntimeError(f"{name} must remain an exact float64 tensor")
    normalized = torch.as_tensor(labels["normalized_responses"]).double()
    raw_margin = torch.as_tensor(labels["raw_donor_minus_base_margins"]).double()
    raw_shift = torch.as_tensor(labels["raw_margin_shifts_from_base"]).double()
    base = torch.as_tensor(labels["clean_base_donor_minus_base_margins"]).double()
    donor = torch.as_tensor(labels["clean_donor_donor_minus_base_margins"]).double()
    endpoint = torch.as_tensor(labels["clean_endpoints"]).double()
    if (
        normalized.ndim != 2
        or raw_margin.shape != normalized.shape
        or raw_shift.shape != normalized.shape
        or base.shape != (normalized.shape[1],)
        or donor.shape != base.shape
        or endpoint.shape != base.shape
        or not all(
            bool(value.isfinite().all())
            for value in (normalized, raw_margin, raw_shift, base, donor, endpoint)
        )
    ):
        raise RuntimeError("response coordinate shape or finiteness changed")
    if not torch.allclose(endpoint, donor - base, atol=1e-12, rtol=1e-12):
        raise RuntimeError("clean endpoint identity changed")
    if not bool((endpoint >= 0.25).all()):
        raise RuntimeError("clean endpoint floor changed")
    if not torch.allclose(raw_shift, raw_margin - base[None], atol=1e-12, rtol=1e-12):
        raise RuntimeError("raw margin-shift identity changed")
    if not torch.allclose(
        normalized, raw_shift / endpoint[None], atol=1e-12, rtol=1e-12
    ):
        raise RuntimeError("normalized response identity changed")


def acquire_prepared_split(
    runner: E311DirectedIOIPatchRunner,
    prepared: Mapping[str, Sequence[PreparedIOIPatchContext]],
    *,
    split: Split,
    registration_sha256: str,
    clean_screen_sha256: str,
) -> dict[str, Any]:
    """Evaluate one registered split only after the global clean-screen seal."""
    if split not in SPLITS:
        raise ValueError("split must be discovery or confirmation")
    contexts = tuple(prepared[split])
    if len(contexts) != len(build_prompt_manifest(split)):
        raise RuntimeError("prepared split context count changed")
    manifest = build_split_mask_manifest(split)
    mask_ids, masks = masks_from_split_manifest(manifest)
    smoke_manifest: tuple[dict[str, Any], ...] = ()
    smoke = None
    smoke_masks = None
    if split == "discovery":
        # This numerical gate deliberately precedes the first full-atlas patch.
        smoke_manifest = build_batch_smoke_manifest(manifest)
        smoke_indices = [row["condition_index"] for row in smoke_manifest]
        smoke_masks = masks[smoke_indices]
        smoke = runner.batch_invariance_smoke(
            contexts[0],
            smoke_masks,
            reference_batch_size=BATCH_SMOKE_REFERENCE_BATCH_SIZE,
            candidate_batch_size=BATCH_SMOKE_CANDIDATE_BATCH_SIZE,
        )
        smoke["context_index"] = 0
        smoke["pair_id"] = str(contexts[0].clean_metadata["pair_id"])
        if not smoke["passed"]:
            raise RuntimeError("registered discovery batch-invariance smoke failed")

    normalized_columns = []
    raw_margin_columns = []
    raw_shift_columns = []
    for context in contexts:
        response = runner.responses_from_prepared(context, masks)
        normalized_columns.append(response.normalized_transfer)
        raw_margin_columns.append(response.raw_donor_minus_base_margin)
        raw_shift_columns.append(response.raw_margin_shift_from_base)
    normalized = torch.stack(normalized_columns, dim=1).double()
    raw_margin = torch.stack(raw_margin_columns, dim=1).double()
    raw_shift = torch.stack(raw_shift_columns, dim=1).double()

    labels = {
        "schema_version": "e31_1_prospective_response_panel_v1",
        "stage": split,
        "response_coordinate": RESPONSE_COORDINATE,
        "pair_ids": tuple(record["pair_id"] for record in build_prompt_manifest(split)),
        "prompt_records": tuple(deepcopy(build_prompt_manifest(split))),
        "mask_ids": mask_ids,
        "masks": masks,
        "component_map": tuple(runner.components),
        "normalized_responses": normalized,
        "raw_donor_minus_base_margins": raw_margin,
        "raw_margin_shifts_from_base": raw_shift,
        "clean_base_donor_minus_base_margins": torch.tensor(
            [context.clean_base_donor_minus_base_margin for context in contexts],
            dtype=torch.float64,
        ),
        "clean_donor_donor_minus_base_margins": torch.tensor(
            [context.clean_donor_donor_minus_base_margin for context in contexts],
            dtype=torch.float64,
        ),
        "clean_endpoints": torch.tensor(
            [context.clean_endpoint for context in contexts], dtype=torch.float64
        ),
        "clean_metadata": tuple(dict(context.clean_metadata) for context in contexts),
        "protocol_commitments": {
            "registration_sha256": registration_sha256,
            "protocol_sha256": canonical_sha256(build_prospective_protocol()),
            "prompt_manifest_sha256": canonical_sha256(build_prompt_manifest(split)),
            "split_manifest_sha256": canonical_sha256(manifest),
            "clean_screen_sha256": clean_screen_sha256,
        },
        "batch_invariance_smoke_manifest": smoke_manifest,
        "batch_invariance_smoke_manifest_sha256": canonical_sha256(smoke_manifest),
        "batch_invariance_smoke": smoke,
        "batch_smoke_masks": smoke_masks,
        "query_accounting": {
            "context_count": len(contexts),
            "mask_rows": len(masks),
            "patched_context_mask_cells": len(contexts) * len(masks),
            "clean_forward_examples_are_in_global_screen": True,
            "batch_smoke_mask_rows": len(smoke_manifest),
            "batch_smoke_physical_context_mask_cells": 2 * len(smoke_manifest),
        },
        "all_integrity_gates_passed": True,
        "unclipped": True,
    }
    validate_response_identities(labels)
    return labels


def validate_acquired_split(
    labels: Mapping[str, Any],
    *,
    registration: Mapping[str, Any],
    registration_sha256: str,
    split: Split,
    clean_screen_sha256: str,
) -> dict[str, Any]:
    """Validate exact split order, masks, metadata, gates, and identities."""
    validated_registration = validate_registration_payload(
        registration, verify_external_files=False
    )
    if split not in SPLITS or labels.get("stage") != split:
        raise RuntimeError("acquired split stage changed")
    if (
        labels.get("schema_version") != "e31_1_prospective_response_panel_v1"
        or labels.get("all_integrity_gates_passed") is not True
        or labels.get("unclipped") is not True
        or labels.get("response_coordinate") != RESPONSE_COORDINATE
    ):
        raise RuntimeError("acquired split schema or integrity changed")
    manifest = build_split_mask_manifest(split)
    expected_ids, expected_masks = masks_from_split_manifest(manifest)
    acquired_masks = labels.get("masks")
    if (
        tuple(labels.get("mask_ids", ())) != expected_ids
        or not isinstance(acquired_masks, torch.Tensor)
        or acquired_masks.dtype != torch.uint8
        or not torch.equal(acquired_masks, expected_masks)
    ):
        raise RuntimeError("acquired split mask rows changed")
    expected_prompts = build_prompt_manifest(split)
    if tuple(labels.get("prompt_records", ())) != expected_prompts or tuple(
        labels.get("pair_ids", ())
    ) != tuple(record["pair_id"] for record in expected_prompts):
        raise RuntimeError("acquired prompt rows changed")
    if [list(row) for row in labels.get("component_map", ())] != validated_registration[
        "component_map"
    ]:
        raise RuntimeError("acquired component map changed")
    commitments = labels.get("protocol_commitments")
    expected_commitments = {
        "registration_sha256": registration_sha256,
        "protocol_sha256": validated_registration["protocol_sha256"],
        "prompt_manifest_sha256": validated_registration["prompt_manifest_sha256"][
            split
        ],
        "split_manifest_sha256": validated_registration["split_manifest_sha256"][split],
        "clean_screen_sha256": clean_screen_sha256,
    }
    if commitments != expected_commitments:
        raise RuntimeError("acquired protocol commitments changed")
    validate_response_identities(labels)
    metadata = labels.get("clean_metadata")
    contracts = validated_registration["clean_metadata_contract"][split]
    if not isinstance(metadata, Sequence) or len(metadata) != len(contracts):
        raise RuntimeError("acquired clean metadata row count changed")
    for index, (observed, contract) in enumerate(zip(metadata, contracts, strict=True)):
        if not isinstance(observed, Mapping):
            raise TypeError("clean metadata rows must be mappings")
        if {key: observed.get(key) for key in contract} != contract:
            raise RuntimeError("acquired clean metadata projection changed")
        expected_outcomes = {
            "clean_base_donor_minus_base_margin": float(
                labels["clean_base_donor_minus_base_margins"][index]
            ),
            "clean_donor_donor_minus_base_margin": float(
                labels["clean_donor_donor_minus_base_margins"][index]
            ),
            "clean_endpoint": float(labels["clean_endpoints"][index]),
        }
        if any(
            float(observed.get(key, math.nan)) != value
            for key, value in expected_outcomes.items()
        ):
            raise RuntimeError("acquired clean outcome metadata changed")
    expected_shape = (
        manifest["unique_condition_count"],
        len(expected_prompts),
    )
    if tuple(labels["normalized_responses"].shape) != expected_shape:
        raise RuntimeError("acquired response matrix shape changed")
    if split == "discovery":
        expected_smoke = build_batch_smoke_manifest(manifest)
        smoke = labels.get("batch_invariance_smoke")
        smoke_masks = labels.get("batch_smoke_masks")
        expected_indices = [row["condition_index"] for row in expected_smoke]
        expected_smoke_masks = expected_masks[expected_indices]
        expected_coordinates = {"normalized_transfer", "raw_donor_minus_base_margin"}
        if (
            tuple(labels.get("batch_invariance_smoke_manifest", ())) != expected_smoke
            or labels.get("batch_invariance_smoke_manifest_sha256")
            != canonical_sha256(expected_smoke)
            or not isinstance(smoke, Mapping)
            or smoke.get("passed") is not True
            or smoke.get("context_index") != 0
            or smoke.get("pair_id") != expected_prompts[0]["pair_id"]
            or smoke.get("mask_rows") != BATCH_SMOKE_MASK_COUNT
            or smoke.get("reference_batch_size") != BATCH_SMOKE_REFERENCE_BATCH_SIZE
            or smoke.get("candidate_batch_size") != BATCH_SMOKE_CANDIDATE_BATCH_SIZE
            or smoke.get("atol") != BATCH_SMOKE_ATOL
            or smoke.get("rtol") != BATCH_SMOKE_RTOL
            or set(smoke.get("coordinates", {})) != expected_coordinates
            or any(
                record.get("passed") is not True
                for record in smoke.get("coordinates", {}).values()
            )
            or not isinstance(smoke_masks, torch.Tensor)
            or smoke_masks.dtype != torch.uint8
            or not torch.equal(smoke_masks, expected_smoke_masks)
        ):
            raise RuntimeError("discovery batch smoke changed or failed")
    elif labels.get("batch_invariance_smoke") is not None:
        raise RuntimeError("confirmation must not consume a second batch smoke")
    expected_accounting = {
        "context_count": len(expected_prompts),
        "mask_rows": len(expected_masks),
        "patched_context_mask_cells": len(expected_prompts) * len(expected_masks),
        "clean_forward_examples_are_in_global_screen": True,
        "batch_smoke_mask_rows": BATCH_SMOKE_MASK_COUNT if split == "discovery" else 0,
        "batch_smoke_physical_context_mask_cells": (
            2 * BATCH_SMOKE_MASK_COUNT if split == "discovery" else 0
        ),
    }
    if labels.get("query_accounting") != expected_accounting:
        raise RuntimeError("acquired query accounting changed")
    return deepcopy(dict(labels))


__all__ = [
    "BATCH_SMOKE_ATOL",
    "BATCH_SMOKE_CANDIDATE_BATCH_SIZE",
    "BATCH_SMOKE_MASK_COUNT",
    "BATCH_SMOKE_REFERENCE_BATCH_SIZE",
    "BATCH_SMOKE_RTOL",
    "E311DirectedIOIPatchRunner",
    "acquire_prepared_split",
    "adapt_directed_ioi_record",
    "build_batch_smoke_manifest",
    "masks_from_split_manifest",
    "screen_all_registered_clean_rows",
    "validate_acquired_split",
    "validate_response_identities",
]
