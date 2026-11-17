# ruff: noqa: EXE002
"""Pure in-memory acquisition helpers for the frozen E31.2 manifests.

The response-blind protocol owns all prompts, backgrounds, condition IDs, and
quartet references.  This module only converts a validated manifest to masks,
prepares the 48 clean contexts once, evaluates those masks, and assembles exact
``00, 10, 01, 11`` tensor views.  It performs no model loading, filesystem
access, sealing, head nomination, or statistical scoring.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_15_ioi_acquisition import (
    CleanPanelViabilityError,
    clean_panel_viability_report,
)
from experiments.e31_sparse_interactions.e31_2_head_patch_runner import (
    ATTENTION_HEAD_SHAPE,
    COARSE_COMPONENT_COUNT,
    WHOLE_ATTENTION_EQUIVALENCE_ATOL,
    E312HeadPatchRunner,
    PreparedHeadPatchContext,
)
from experiments.e31_sparse_interactions.mixed_resolution_panel import (
    PROMPT_COUNTS,
    PROMPT_MANIFEST_SHA256,
    PROMPT_RECORD_KEYS,
    build_prompt_manifest,
    canonical_sha256,
)
from experiments.e31_sparse_interactions.mixed_resolution_protocol import (
    CELL_BITS,
    COVER_COUNT,
    HEAD_COUNT,
    PARENT_PAIRS,
    build_atlas_manifest,
    build_reconstruction_manifest,
    validate_atlas_manifest,
)

Split = Literal["discovery", "confirmation"]
SPLITS = ("discovery", "confirmation")
DISCOVERY_SMOKE_CONDITION_COUNT = 16
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _validated_commitments(value: Mapping[str, str] | None) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("protocol_commitments must be a mapping")
    commitments = {}
    for key, digest in value.items():
        if not isinstance(key, str) or not key.endswith("_sha256"):
            raise ValueError("protocol commitment keys must end in _sha256")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError("protocol commitments require lowercase SHA-256 values")
        commitments[key] = digest
    return commitments


def _validated_records(
    prompt_records: Sequence[Mapping[str, Any]], *, split: Split
) -> tuple[dict[str, Any], ...]:
    if isinstance(prompt_records, (str, bytes)) or not isinstance(
        prompt_records, Sequence
    ):
        raise TypeError("prompt records must be a sequence")
    records = tuple(dict(record) for record in prompt_records)
    if len(records) != PROMPT_COUNTS[split]:
        raise ValueError(f"{split} requires exactly {PROMPT_COUNTS[split]} prompts")
    pair_ids = tuple(record.get("pair_id") for record in records)
    if any(not isinstance(pair_id, str) or not pair_id for pair_id in pair_ids):
        raise ValueError("every prompt requires a nonempty pair_id")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("prompt pair_id values must be unique within a split")
    if any(record.get("split") != split for record in records):
        raise ValueError("prompt-record split changed")
    if any(tuple(record) != PROMPT_RECORD_KEYS for record in records):
        raise ValueError("prompt record keys or order changed")
    expected = tuple(build_prompt_manifest(split))
    if (
        records != expected
        or canonical_sha256(records) != PROMPT_MANIFEST_SHA256[split]
    ):
        raise ValueError(f"{split} prompt records differ from the frozen panel")
    return records


def _original_prompt_record(context: PreparedHeadPatchContext) -> dict[str, str]:
    metadata = context.clean_metadata
    record = {key: str(metadata[key]) for key in PROMPT_RECORD_KEYS}
    if tuple(record) != PROMPT_RECORD_KEYS:
        raise RuntimeError("prepared context lost original prompt custody")
    return record


def _validated_prepared(
    runner: E312HeadPatchRunner,
    prepared: Sequence[PreparedHeadPatchContext],
    *,
    split: Split,
) -> tuple[PreparedHeadPatchContext, ...]:
    if isinstance(prepared, (str, bytes)) or not isinstance(prepared, Sequence):
        raise TypeError("prepared contexts must be a sequence")
    contexts = tuple(prepared)
    if len(contexts) != PROMPT_COUNTS[split]:
        raise ValueError(f"{split} requires {PROMPT_COUNTS[split]} prepared contexts")
    if any(
        not isinstance(context, PreparedHeadPatchContext)
        or context._runner is not runner
        for context in contexts
    ):
        raise ValueError("a prepared context belongs to a different runner")
    records = tuple(_original_prompt_record(context) for context in contexts)
    _validated_records(records, split=split)
    return contexts


def prepare_and_screen_all_registered_contexts(
    runner: E312HeadPatchRunner,
    prompt_records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, tuple[PreparedHeadPatchContext, ...]], dict[str, Any]]:
    """Prepare all 48 clean rows once before any patched response is allowed."""
    if not isinstance(runner, E312HeadPatchRunner):
        raise TypeError("runner must be an E312HeadPatchRunner")
    if (
        not isinstance(prompt_records_by_split, Mapping)
        or tuple(prompt_records_by_split) != SPLITS
    ):
        raise ValueError(
            "prompt_records_by_split must be ordered discovery, confirmation"
        )
    records = {
        split: _validated_records(prompt_records_by_split[split], split=split)
        for split in SPLITS
    }
    prepared = {
        split: tuple(runner.prepare_context(record) for record in records[split])
        for split in SPLITS
    }
    reports = {split: clean_panel_viability_report(prepared[split]) for split in SPLITS}
    passed = all(report["passed"] for report in reports.values())
    clean_screen = {
        "schema_version": "e31_2_all_registered_clean_screen_v1",
        "status": (
            "all_48_clean_rows_passed_before_any_patch"
            if passed
            else "all_48_clean_rows_prepared_but_gate_failed"
        ),
        "split_order": SPLITS,
        "context_count_by_split": dict(PROMPT_COUNTS),
        "total_context_count": sum(PROMPT_COUNTS.values()),
        "total_clean_forward_examples": 2 * sum(PROMPT_COUNTS.values()),
        "reports": reports,
        "prompt_records": records,
        "clean_metadata": {
            split: tuple(dict(context.clean_metadata) for context in prepared[split])
            for split in SPLITS
        },
        "clean_head_norm_diagnostics": {
            split: tuple(
                dict(context.clean_head_norm_diagnostics) for context in prepared[split]
            )
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
        "all_head_norm_diagnostics_passed": all(
            context.clean_head_norm_diagnostics["passed"]
            for split in SPLITS
            for context in prepared[split]
        ),
        "all_splits_passed": passed,
        "row_policy": "retain_and_score_every_registered_context",
    }
    if not passed:
        raise CleanPanelViabilityError(clean_screen)
    return prepared, clean_screen


def masks_from_atlas_manifest(
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
    """Return exact condition IDs and coarse/head masks from a frozen atlas."""
    frozen = validate_atlas_manifest(manifest)
    condition_ids = []
    coarse_rows = []
    head_rows = []
    for expected_index, condition in enumerate(frozen["conditions"]):
        if condition["condition_index"] != expected_index:
            raise RuntimeError("atlas condition order changed")
        condition_ids.append(str(condition["condition_id"]))
        coarse_rows.append(condition["coarse_component_mask"])
        head_rows.append(condition["attention_head_mask"])
    coarse = torch.tensor(coarse_rows, dtype=torch.uint8)
    heads = torch.tensor(head_rows, dtype=torch.uint8)
    if (
        tuple(coarse.shape)
        != (frozen["unique_condition_count"], COARSE_COMPONENT_COUNT)
        or tuple(heads.shape)
        != (frozen["unique_condition_count"], *ATTENTION_HEAD_SHAPE)
        or len(condition_ids) != len(set(condition_ids))
    ):
        raise RuntimeError("atlas condition tensor shape or IDs changed")
    if bool(torch.logical_and(coarse[:, 0::2].bool(), heads.any(dim=2)).any()):
        raise RuntimeError("atlas contains a coarse/fine attention overlap")
    return tuple(condition_ids), coarse, heads


def _validated_smoke_indices(
    values: Sequence[int], *, split: Split, row_count: int
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("smoke_condition_indices must be a sequence")
    indices = tuple(values)
    expected = DISCOVERY_SMOKE_CONDITION_COUNT if split == "discovery" else 0
    if len(indices) != expected:
        raise ValueError(f"{split} requires exactly {expected} batch-smoke rows")
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= row_count
        for index in indices
    ) or len(indices) != len(set(indices)):
        raise ValueError("batch-smoke indices must be unique valid condition rows")
    return indices


def _acquire_response_tables(
    runner: E312HeadPatchRunner,
    contexts: Sequence[PreparedHeadPatchContext],
    coarse_masks: torch.Tensor,
    head_masks: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    normalized = []
    raw_margin = []
    raw_shift = []
    errors = []
    for context in contexts:
        responses = runner.responses_from_prepared(context, coarse_masks, head_masks)
        normalized.append(responses.normalized_transfer)
        raw_margin.append(responses.raw_donor_minus_base_margin)
        raw_shift.append(responses.raw_margin_shift_from_base)
        errors.append(
            float(responses.composite_reconstruction["maximum_absolute_error"])
        )
    tables = {
        "normalized": torch.stack(normalized, dim=1).double(),
        "raw_margin": torch.stack(raw_margin, dim=1).double(),
        "raw_shift": torch.stack(raw_shift, dim=1).double(),
    }
    maximum_error = max(errors) if errors else 0.0
    return tables, {
        "semantics": "live_exact_c_proj_W_O_head_composite_reconstruction",
        "maximum_absolute_error": maximum_error,
        "finite": math.isfinite(maximum_error),
        "passed": math.isfinite(maximum_error)
        and all(context.clean_head_norm_diagnostics["passed"] for context in contexts),
    }


def _condition_index(manifest: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(condition["condition_id"]): int(condition["condition_index"])
        for condition in manifest["conditions"]
    }


def _assemble_atlas_quartets(
    table: torch.Tensor, manifest: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    context_count = table.shape[1]
    parent_count = len(PARENT_PAIRS)
    fine = torch.full(
        (parent_count, HEAD_COUNT * HEAD_COUNT, context_count, COVER_COUNT, 4),
        float("nan"),
        dtype=torch.float64,
    )
    all_heads = torch.full(
        (parent_count, context_count, COVER_COUNT, 4),
        float("nan"),
        dtype=torch.float64,
    )
    index = _condition_index(manifest)
    for contrast in manifest["fine_contrasts"]:
        rows = [index[condition_id] for condition_id in contrast["condition_ids"]]
        fine[
            contrast["parent_index"],
            contrast["pair_index"],
            :,
            contrast["cover_rank"],
            :,
        ] = table[rows].T
    for contrast in manifest["coarse_contrasts"]:
        rows = [index[condition_id] for condition_id in contrast["condition_ids"]]
        all_heads[contrast["parent_index"], :, contrast["cover_rank"], :] = table[
            rows
        ].T
    if not bool(fine.isfinite().all() and all_heads.isfinite().all()):
        raise RuntimeError("atlas quartet assembly left an unfilled cell")
    return fine, all_heads


def _clean_tensors(
    contexts: Sequence[PreparedHeadPatchContext],
) -> dict[str, torch.Tensor]:
    return {
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
    }


def _numerical_summary(table: torch.Tensor) -> dict[str, float | bool]:
    finite = bool(table.isfinite().all())
    return {
        "finite": finite,
        "absolute_maximum": float(table.abs().max()) if finite else float("nan"),
        "minimum": float(table.min()) if finite else float("nan"),
        "maximum": float(table.max()) if finite else float("nan"),
    }


def acquire_head_atlas_from_prepared(
    runner: E312HeadPatchRunner,
    prepared: Sequence[PreparedHeadPatchContext],
    manifest: Mapping[str, Any],
    *,
    split: Split,
    smoke_condition_indices: Sequence[int],
    protocol_commitments: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Acquire all 2,064 frozen atlas conditions with zero clean reruns."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS!r}")
    contexts = _validated_prepared(runner, prepared, split=split)
    frozen_manifest = validate_atlas_manifest(manifest)
    if frozen_manifest["split"] != split:
        raise ValueError("atlas manifest split changed")
    condition_ids, coarse_masks, head_masks = masks_from_atlas_manifest(frozen_manifest)
    smoke_indices = _validated_smoke_indices(
        smoke_condition_indices, split=split, row_count=len(condition_ids)
    )
    commitments = _validated_commitments(protocol_commitments)
    smoke = None
    if smoke_indices:
        selected = torch.tensor(smoke_indices, dtype=torch.long)
        smoke = runner.batch_invariance_smoke(
            contexts[0], coarse_masks[selected], head_masks[selected]
        )
        smoke.update(
            {
                "context_index": 0,
                "pair_id": contexts[0].clean_metadata["pair_id"],
                "condition_indices": smoke_indices,
                "condition_ids": tuple(condition_ids[index] for index in smoke_indices),
            }
        )
        if not smoke["passed"]:
            raise RuntimeError("E31.2 discovery batch-invariance smoke failed")
    # The smoke is deliberately complete before the first full-atlas response.
    tables, reconstruction = _acquire_response_tables(
        runner, contexts, coarse_masks, head_masks
    )
    normalized_fine, normalized_all = _assemble_atlas_quartets(
        tables["normalized"], frozen_manifest
    )
    raw_margin_fine, raw_margin_all = _assemble_atlas_quartets(
        tables["raw_margin"], frozen_manifest
    )
    raw_shift_fine, raw_shift_all = _assemble_atlas_quartets(
        tables["raw_shift"], frozen_manifest
    )
    clean = _clean_tensors(contexts)
    artifact = {
        "schema_version": "e31_2_mixed_resolution_atlas_labels_v1",
        "stage": split,
        "manifest_sha256": canonical_sha256(frozen_manifest),
        "prompt_manifest_sha256": PROMPT_MANIFEST_SHA256[split],
        "response_coordinate": contexts[0].clean_metadata["response_coordinate"],
        "pair_ids": tuple(context.clean_metadata["pair_id"] for context in contexts),
        "prompt_records": tuple(
            _original_prompt_record(context) for context in contexts
        ),
        "condition_ids": condition_ids,
        "coarse_component_masks": coarse_masks,
        "attention_head_masks": head_masks,
        "coarse_component_map": tuple(runner.components),
        "head_component_map": tuple(runner.head_components),
        "parent_pair_order": tuple(frozen_manifest["parent_pair_order"]),
        "head_order": tuple(frozen_manifest["head_order"]),
        "ordered_cell_bits": tuple(CELL_BITS),
        "normalized_responses": tables["normalized"],
        "raw_donor_minus_base_margins": tables["raw_margin"],
        "raw_margin_shifts_from_base": tables["raw_shift"],
        "normalized_fine_quartets": normalized_fine,
        "raw_margin_fine_quartets": raw_margin_fine,
        "raw_shift_fine_quartets": raw_shift_fine,
        "normalized_all_head_quartets": normalized_all,
        "raw_margin_all_head_quartets": raw_margin_all,
        "raw_shift_all_head_quartets": raw_shift_all,
        **clean,
        "clean_metadata": tuple(dict(context.clean_metadata) for context in contexts),
        "clean_head_norm_diagnostics": tuple(
            dict(context.clean_head_norm_diagnostics) for context in contexts
        ),
        "protocol_commitments": commitments,
        "batch_invariance_smoke": smoke,
        "composite_reconstruction": reconstruction,
        "whole_attention_response_equivalence": {
            "evaluated": False,
            "reason": "no_extra_model_calls_and_no_fresh_coarse_reference",
            "runner_regression_tolerance": WHOLE_ATTENTION_EQUIVALENCE_ATOL,
        },
        "numerical_summary": {
            name: _numerical_summary(table)
            for name, table in {
                "normalized_responses": tables["normalized"],
                "raw_donor_minus_base_margins": tables["raw_margin"],
                "raw_margin_shifts_from_base": tables["raw_shift"],
                "normalized_fine_quartets": normalized_fine,
            }.items()
        },
        "query_accounting": {
            "context_count": len(contexts),
            "condition_rows": len(condition_ids),
            "patched_condition_context_cells": len(contexts) * len(condition_ids),
            "clean_forward_examples": 0,
            "clean_forward_examples_are_in_global_screen": True,
            "whole_attention_equivalence_additional_context_cells": 0,
            "batch_smoke_condition_rows": len(smoke_indices),
            "batch_smoke_physical_context_cells": 2 * len(smoke_indices),
        },
    }
    artifact["all_integrity_gates_passed"] = (
        all(summary["finite"] for summary in artifact["numerical_summary"].values())
        and reconstruction["passed"]
        and (smoke is None or smoke["passed"])
    )
    validate_atlas_labels(artifact, manifest=frozen_manifest)
    return artifact


def validate_atlas_labels(
    labels: Mapping[str, Any], *, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed on masks, response identities, and exact quartet gathers."""
    if not isinstance(labels, Mapping):
        raise TypeError("atlas labels must be a mapping")
    frozen_manifest = validate_atlas_manifest(manifest)
    split = frozen_manifest["split"]
    condition_ids, coarse_masks, head_masks = masks_from_atlas_manifest(frozen_manifest)
    context_count = PROMPT_COUNTS[split]
    if (
        labels.get("schema_version") != "e31_2_mixed_resolution_atlas_labels_v1"
        or labels.get("stage") != split
        or labels.get("manifest_sha256") != canonical_sha256(frozen_manifest)
        or labels.get("prompt_manifest_sha256") != PROMPT_MANIFEST_SHA256[split]
        or tuple(labels.get("condition_ids", ())) != condition_ids
        or not torch.equal(labels.get("coarse_component_masks"), coarse_masks)
        or not torch.equal(labels.get("attention_head_masks"), head_masks)
        or labels.get("all_integrity_gates_passed") is not True
    ):
        raise RuntimeError("atlas label header, masks, or integrity status changed")
    expected_records = tuple(build_prompt_manifest(split))
    if tuple(labels.get("prompt_records", ())) != expected_records or tuple(
        labels.get("pair_ids", ())
    ) != tuple(record["pair_id"] for record in expected_records):
        raise RuntimeError("atlas labels changed frozen prompt records or order")
    metadata = labels.get("clean_metadata")
    if not isinstance(metadata, Sequence) or len(metadata) != context_count:
        raise RuntimeError("atlas labels changed clean metadata rows")
    for observed, expected in zip(metadata, expected_records, strict=True):
        if any(
            str(observed.get(key)) != str(expected[key]) for key in PROMPT_RECORD_KEYS
        ):
            raise RuntimeError("clean metadata lost original prompt custody")
    names = (
        "normalized_responses",
        "raw_donor_minus_base_margins",
        "raw_margin_shifts_from_base",
    )
    for name in names:
        value = labels.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float64
            or tuple(value.shape) != (len(condition_ids), context_count)
            or not bool(value.isfinite().all())
        ):
            raise RuntimeError(f"{name} changed shape, dtype, or finiteness")
    base = labels.get("clean_base_donor_minus_base_margins")
    donor = labels.get("clean_donor_donor_minus_base_margins")
    endpoint = labels.get("clean_endpoints")
    if any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float64
        or tuple(value.shape) != (context_count,)
        for value in (base, donor, endpoint)
    ):
        raise RuntimeError("clean response coordinates changed")
    raw_margin = labels["raw_donor_minus_base_margins"]
    raw_shift = labels["raw_margin_shifts_from_base"]
    normalized = labels["normalized_responses"]
    if (
        not torch.allclose(endpoint, donor - base, atol=1e-12, rtol=1e-12)
        or not torch.allclose(
            raw_shift, raw_margin - base[None], atol=1e-12, rtol=1e-12
        )
        or not torch.allclose(
            normalized, raw_shift / endpoint[None], atol=1e-12, rtol=1e-12
        )
    ):
        raise RuntimeError("atlas response identity changed")
    for prefix, table in (
        ("normalized", normalized),
        ("raw_margin", raw_margin),
        ("raw_shift", raw_shift),
    ):
        fine, all_heads = _assemble_atlas_quartets(table, frozen_manifest)
        if not torch.equal(
            labels.get(f"{prefix}_fine_quartets"), fine
        ) or not torch.equal(labels.get(f"{prefix}_all_head_quartets"), all_heads):
            raise RuntimeError("atlas quartet view changed")
    return dict(labels)


def masks_from_reconstruction_manifest(
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
    """Return only C reconstruction conditions absent from the atlas."""
    if not isinstance(manifest, Mapping) or "selections" not in manifest:
        raise TypeError("reconstruction manifest must be a mapping")
    expected = build_reconstruction_manifest(manifest["selections"])
    if dict(manifest) != expected:
        raise ValueError("reconstruction manifest differs from canonical construction")
    rows = expected["new_conditions"]
    ids = tuple(str(row["condition_id"]) for row in rows)
    coarse = torch.tensor(
        [row["coarse_component_mask"] for row in rows], dtype=torch.uint8
    )
    heads = torch.tensor(
        [row["attention_head_mask"] for row in rows], dtype=torch.uint8
    )
    if rows:
        if tuple(coarse.shape) != (len(rows), COARSE_COMPONENT_COUNT) or tuple(
            heads.shape
        ) != (len(rows), *ATTENTION_HEAD_SHAPE):
            raise RuntimeError("reconstruction mask shape changed")
        if bool(torch.logical_and(coarse[:, 0::2].bool(), heads.any(dim=2)).any()):
            raise RuntimeError("reconstruction contains coarse/fine overlap")
    else:
        coarse = torch.empty(0, COARSE_COMPONENT_COUNT, dtype=torch.uint8)
        heads = torch.empty(0, *ATTENTION_HEAD_SHAPE, dtype=torch.uint8)
    return ids, coarse, heads


def _assemble_reconstruction_quartets(
    *,
    atlas_table: torch.Tensor,
    atlas_manifest: Mapping[str, Any],
    new_table: torch.Tensor,
    reconstruction_manifest: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    context_count = atlas_table.shape[1]
    direct = torch.full(
        (len(PARENT_PAIRS), 2, context_count, COVER_COUNT, 4),
        float("nan"),
        dtype=torch.float64,
    )
    all12 = torch.full(
        (len(PARENT_PAIRS), context_count, COVER_COUNT, 4),
        float("nan"),
        dtype=torch.float64,
    )
    atlas_index = _condition_index(atlas_manifest)
    new_index = {
        str(row["condition_id"]): int(row["condition_index"])
        for row in reconstruction_manifest["new_conditions"]
    }

    def rows_for(condition_ids: Sequence[str]) -> torch.Tensor:
        values = []
        for condition_id in condition_ids:
            if condition_id in atlas_index:
                values.append(atlas_table[atlas_index[condition_id]])
            elif condition_id in new_index:
                values.append(new_table[new_index[condition_id]])
            else:
                raise RuntimeError("reconstruction references an unknown condition")
        return torch.stack(values)

    support_index = {"pair_selected": 0, "singleton_selected": 1}
    for contrast in reconstruction_manifest["contrasts"]:
        direct[
            contrast["parent_index"],
            support_index[contrast["support_name"]],
            :,
            contrast["cover_rank"],
            :,
        ] = rows_for(contrast["condition_ids"]).T
        reference = rows_for(contrast["direct_all_12_condition_ids"]).T
        target = all12[contrast["parent_index"], :, contrast["cover_rank"], :]
        if bool(torch.isfinite(target).all()) and not torch.equal(target, reference):
            raise RuntimeError("all-12 reconstruction references disagree")
        all12[contrast["parent_index"], :, contrast["cover_rank"], :] = reference
    if not bool(direct.isfinite().all() and all12.isfinite().all()):
        raise RuntimeError("reconstruction quartet assembly left an unfilled cell")
    return direct, all12


def acquire_reconstruction_from_prepared(
    runner: E312HeadPatchRunner,
    prepared_confirmation: Sequence[PreparedHeadPatchContext],
    reconstruction_manifest: Mapping[str, Any],
    confirmation_atlas_labels: Mapping[str, Any],
    *,
    protocol_commitments: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Acquire only new C group conditions and reuse inherited atlas cells."""
    contexts = _validated_prepared(runner, prepared_confirmation, split="confirmation")
    atlas_manifest = build_atlas_manifest("confirmation")
    atlas = validate_atlas_labels(confirmation_atlas_labels, manifest=atlas_manifest)
    clean = _clean_tensors(contexts)
    for name in (
        "clean_base_donor_minus_base_margins",
        "clean_donor_donor_minus_base_margins",
        "clean_endpoints",
    ):
        if not torch.equal(clean[name], atlas[name]):
            raise RuntimeError(
                "prepared confirmation contexts disagree with the inherited atlas"
            )
    new_ids, coarse_masks, head_masks = masks_from_reconstruction_manifest(
        reconstruction_manifest
    )
    frozen_reconstruction = build_reconstruction_manifest(
        reconstruction_manifest["selections"]
    )
    commitments = _validated_commitments(protocol_commitments)
    if new_ids:
        tables, reconstruction = _acquire_response_tables(
            runner, contexts, coarse_masks, head_masks
        )
    else:
        empty = torch.empty(0, len(contexts), dtype=torch.float64)
        tables = {"normalized": empty, "raw_margin": empty, "raw_shift": empty}
        reconstruction = {
            "semantics": "no_new_conditions",
            "maximum_absolute_error": 0.0,
            "finite": True,
            "passed": True,
        }
    assembled = {}
    for prefix, atlas_key, new_key in (
        ("normalized", "normalized_responses", "normalized"),
        ("raw_margin", "raw_donor_minus_base_margins", "raw_margin"),
        ("raw_shift", "raw_margin_shifts_from_base", "raw_shift"),
    ):
        direct, all12 = _assemble_reconstruction_quartets(
            atlas_table=atlas[atlas_key],
            atlas_manifest=atlas_manifest,
            new_table=tables[new_key],
            reconstruction_manifest=frozen_reconstruction,
        )
        assembled[f"{prefix}_direct_group_quartets"] = direct
        assembled[f"{prefix}_all12_reference_quartets"] = all12
    artifact = {
        "schema_version": "e31_2_confirmation_reconstruction_labels_v1",
        "stage": "confirmation",
        "manifest_sha256": canonical_sha256(frozen_reconstruction),
        "prompt_manifest_sha256": PROMPT_MANIFEST_SHA256["confirmation"],
        "selection_sha256": frozen_reconstruction["selection_sha256"],
        "support_order": tuple(frozen_reconstruction["support_order"]),
        "pair_ids": tuple(context.clean_metadata["pair_id"] for context in contexts),
        "prompt_records": tuple(
            _original_prompt_record(context) for context in contexts
        ),
        "new_condition_ids": new_ids,
        "new_coarse_component_masks": coarse_masks,
        "new_attention_head_masks": head_masks,
        "normalized_new_responses": tables["normalized"],
        "raw_margin_new_responses": tables["raw_margin"],
        "raw_shift_new_responses": tables["raw_shift"],
        **assembled,
        "inherited_confirmation_atlas_manifest_sha256": atlas["manifest_sha256"],
        "protocol_commitments": commitments,
        "composite_reconstruction": reconstruction,
        "query_accounting": {
            "context_count": len(contexts),
            "new_condition_rows": len(new_ids),
            "maximum_new_condition_rows": frozen_reconstruction[
                "maximum_new_condition_count"
            ],
            "patched_new_condition_context_cells": len(contexts) * len(new_ids),
            "inherited_atlas_condition_context_cells_reused": len(contexts)
            * len(atlas["condition_ids"]),
            "duplicate_inherited_condition_calls": 0,
            "clean_forward_examples": 0,
            "clean_forward_examples_are_in_global_screen": True,
        },
        "all_integrity_gates_passed": reconstruction["passed"]
        and all(bool(table.isfinite().all()) for table in assembled.values()),
    }
    validate_reconstruction_labels(
        artifact,
        reconstruction_manifest=frozen_reconstruction,
        confirmation_atlas_labels=atlas,
    )
    return artifact


def validate_reconstruction_labels(
    labels: Mapping[str, Any],
    *,
    reconstruction_manifest: Mapping[str, Any],
    confirmation_atlas_labels: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate new masks and exact inherited/new quartet assembly."""
    if not isinstance(labels, Mapping):
        raise TypeError("reconstruction labels must be a mapping")
    frozen = build_reconstruction_manifest(reconstruction_manifest["selections"])
    atlas_manifest = build_atlas_manifest("confirmation")
    atlas = validate_atlas_labels(confirmation_atlas_labels, manifest=atlas_manifest)
    new_ids, coarse, heads = masks_from_reconstruction_manifest(frozen)
    context_count = PROMPT_COUNTS["confirmation"]
    if (
        labels.get("schema_version") != "e31_2_confirmation_reconstruction_labels_v1"
        or labels.get("stage") != "confirmation"
        or labels.get("manifest_sha256") != canonical_sha256(frozen)
        or labels.get("prompt_manifest_sha256")
        != PROMPT_MANIFEST_SHA256["confirmation"]
        or tuple(labels.get("new_condition_ids", ())) != new_ids
        or not torch.equal(labels.get("new_coarse_component_masks"), coarse)
        or not torch.equal(labels.get("new_attention_head_masks"), heads)
        or labels.get("all_integrity_gates_passed") is not True
    ):
        raise RuntimeError("reconstruction label header or masks changed")
    expected_records = tuple(build_prompt_manifest("confirmation"))
    if tuple(labels.get("prompt_records", ())) != expected_records or tuple(
        labels.get("pair_ids", ())
    ) != tuple(record["pair_id"] for record in expected_records):
        raise RuntimeError(
            "reconstruction labels changed frozen prompt records or order"
        )
    new_tables = {
        "normalized": labels.get("normalized_new_responses"),
        "raw_margin": labels.get("raw_margin_new_responses"),
        "raw_shift": labels.get("raw_shift_new_responses"),
    }
    for name, table in new_tables.items():
        if (
            not isinstance(table, torch.Tensor)
            or table.dtype != torch.float64
            or tuple(table.shape) != (len(new_ids), context_count)
            or not bool(table.isfinite().all())
        ):
            raise RuntimeError(f"{name} reconstruction table changed")
    if len(new_ids) and (
        not torch.allclose(
            new_tables["raw_shift"],
            new_tables["raw_margin"]
            - atlas["clean_base_donor_minus_base_margins"][None],
            atol=1e-12,
            rtol=1e-12,
        )
        or not torch.allclose(
            new_tables["normalized"],
            new_tables["raw_shift"] / atlas["clean_endpoints"][None],
            atol=1e-12,
            rtol=1e-12,
        )
    ):
        raise RuntimeError("new reconstruction response identity changed")
    for prefix, atlas_key in (
        ("normalized", "normalized_responses"),
        ("raw_margin", "raw_donor_minus_base_margins"),
        ("raw_shift", "raw_margin_shifts_from_base"),
    ):
        direct, all12 = _assemble_reconstruction_quartets(
            atlas_table=atlas[atlas_key],
            atlas_manifest=atlas_manifest,
            new_table=new_tables[prefix],
            reconstruction_manifest=frozen,
        )
        if not torch.equal(
            labels.get(f"{prefix}_direct_group_quartets"), direct
        ) or not torch.equal(labels.get(f"{prefix}_all12_reference_quartets"), all12):
            raise RuntimeError("reconstruction quartet view changed")
    return dict(labels)


__all__ = [
    "DISCOVERY_SMOKE_CONDITION_COUNT",
    "SPLITS",
    "acquire_head_atlas_from_prepared",
    "acquire_reconstruction_from_prepared",
    "masks_from_atlas_manifest",
    "masks_from_reconstruction_manifest",
    "prepare_and_screen_all_registered_contexts",
    "validate_atlas_labels",
    "validate_reconstruction_labels",
]
