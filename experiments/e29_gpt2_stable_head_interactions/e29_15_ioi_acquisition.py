# ruff: noqa: EXE002
"""Pure, in-memory acquisition primitives for E29.15.

This module deliberately performs no model loading and no filesystem writes.
The eventual sealed drivers can verify their custody chain, construct an
``E2915IOIPatchRunner``, call this helper, and atomically save the returned
payload without changing the scientific acquisition semantics.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_15_ioi_patch_runner import (
    COMPONENT_COUNT,
    RESPONSE_COORDINATE,
    E2915IOIPatchRunner,
    PreparedIOIPatchContext,
)

AcquisitionStage = Literal["discovery", "confirmation"]
ACQUISITION_STAGES = ("discovery", "confirmation")
MINIMUM_ENDPOINT_CORRECT_FRACTION = 0.75
_SHA256 = re.compile(r"[0-9a-f]{64}")


class CleanPanelViabilityError(RuntimeError):
    """Raised before patched outcomes when an aggregate clean gate fails."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        super().__init__("aggregate clean IOI panel viability gate failed")
        self.report = deepcopy(dict(report))


def clean_panel_viability_report(
    contexts: Sequence[PreparedIOIPatchContext],
) -> dict[str, Any]:
    """Score the preregistered aggregate gate without filtering prompt rows."""
    if not contexts:
        raise ValueError("at least one prepared IOI context is required")
    context_count = len(contexts)
    required_correct = math.ceil(MINIMUM_ENDPOINT_CORRECT_FRACTION * context_count)
    base_correct_count = sum(context.clean_base_correct for context in contexts)
    donor_correct_count = sum(context.clean_donor_correct for context in contexts)
    mean_base_correct_margin = (
        sum(-context.clean_base_donor_minus_base_margin for context in contexts)
        / context_count
    )
    mean_donor_correct_margin = (
        sum(context.clean_donor_donor_minus_base_margin for context in contexts)
        / context_count
    )
    gates = {
        "base_correct_count": base_correct_count >= required_correct,
        "donor_correct_count": donor_correct_count >= required_correct,
        "mean_base_correct_margin_positive": mean_base_correct_margin > 0.0,
        "mean_donor_correct_margin_positive": mean_donor_correct_margin > 0.0,
    }
    per_context = tuple(
        {
            "pair_id": str(context.clean_metadata["pair_id"]),
            "clean_base_donor_minus_base_margin": (
                context.clean_base_donor_minus_base_margin
            ),
            "clean_donor_donor_minus_base_margin": (
                context.clean_donor_donor_minus_base_margin
            ),
            "clean_endpoint": context.clean_endpoint,
            "clean_base_correct": context.clean_base_correct,
            "clean_donor_correct": context.clean_donor_correct,
        }
        for context in contexts
    )
    return {
        "context_count": context_count,
        "minimum_endpoint_correct_fraction": MINIMUM_ENDPOINT_CORRECT_FRACTION,
        "required_correct_contexts": required_correct,
        "base_correct_contexts": base_correct_count,
        "donor_correct_contexts": donor_correct_count,
        "base_correct_fraction": base_correct_count / context_count,
        "donor_correct_fraction": donor_correct_count / context_count,
        "mean_base_correct_minus_donor_margin": mean_base_correct_margin,
        "mean_donor_correct_minus_base_margin": mean_donor_correct_margin,
        "per_context": per_context,
        "gates": gates,
        "passed": all(gates.values()),
        "row_policy": "retain_and_score_every_preregistered_context",
    }


def _validated_masks(masks: torch.Tensor) -> torch.Tensor:
    if not isinstance(masks, torch.Tensor):
        raise TypeError("masks must be a tensor")
    if masks.ndim != 2 or masks.shape[1] != COMPONENT_COUNT or len(masks) == 0:
        raise ValueError("masks must have shape (rows, 24) with at least one row")
    if not bool(torch.logical_or(masks == 0, masks == 1).all()):
        raise ValueError("masks must be binary")
    canonical = masks.detach().to(device="cpu", dtype=torch.uint8).clone()
    if len(torch.unique(canonical, dim=0)) != len(canonical):
        raise ValueError("acquisition masks must be unique")
    return canonical


def _validated_mask_ids(mask_ids: Sequence[str], *, rows: int) -> tuple[str, ...]:
    if isinstance(mask_ids, (str, bytes)) or not isinstance(mask_ids, Sequence):
        raise TypeError("mask_ids must be a sequence of strings")
    identifiers = tuple(mask_ids)
    if len(identifiers) != rows or any(
        not isinstance(identifier, str) or not identifier for identifier in identifiers
    ):
        raise ValueError("one nonempty string mask_id is required per mask row")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("mask_ids must be unique")
    return identifiers


def _validated_records(
    records: Sequence[Mapping[str, Any]], *, stage: AcquisitionStage
) -> tuple[dict[str, Any], ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("prompt records must be a sequence")
    copied = tuple(dict(record) for record in records)
    if not copied:
        raise ValueError("at least one IOI prompt record is required")
    pair_ids = tuple(record.get("pair_id") for record in copied)
    if any(not isinstance(pair_id, str) or not pair_id for pair_id in pair_ids):
        raise ValueError("every IOI prompt record needs a nonempty pair_id")
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("IOI pair_id values must be unique")
    if any(record.get("split") != stage for record in copied):
        raise ValueError("every IOI prompt-record split must match acquisition stage")
    return copied


def _validated_commitments(
    commitments: Mapping[str, str] | None,
) -> dict[str, str]:
    if commitments is None:
        return {}
    if not isinstance(commitments, Mapping):
        raise TypeError("protocol_commitments must be a mapping")
    validated: dict[str, str] = {}
    for key, digest in commitments.items():
        if not isinstance(key, str) or not key.endswith("_sha256"):
            raise ValueError("protocol commitment keys must end in _sha256")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(
                "protocol commitments require a 64-character lowercase SHA-256"
            )
        validated[key] = digest
    return validated


def _numerical_summary(table: torch.Tensor) -> dict[str, float | bool]:
    finite = bool(table.isfinite().all())
    return {
        "finite": finite,
        "absolute_maximum": float(table.abs().max()) if finite else float("nan"),
        "minimum": float(table.min()) if finite else float("nan"),
        "maximum": float(table.max()) if finite else float("nan"),
    }


def acquire_ioi_response_panel(
    runner: E2915IOIPatchRunner,
    prompt_records: Sequence[Mapping[str, Any]],
    masks: torch.Tensor,
    *,
    stage: AcquisitionStage,
    mask_ids: Sequence[str],
    nomination_provenance: Sequence[Any] = (),
    protocol_commitments: Mapping[str, str] | None = None,
    smoke_masks: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Acquire one discovery or confirmation panel without writing it.

    Every clean pair is prepared and viability-checked before the first patched
    response. The returned tensor tables use ``(mask, context)`` orientation.
    Confirmation masks may be a de-duplicated union of nominations from many
    methods; ``nomination_provenance`` is carried through verbatim for custody.
    """
    if not isinstance(runner, E2915IOIPatchRunner):
        raise TypeError("runner must be an E2915IOIPatchRunner")
    if stage not in ACQUISITION_STAGES:
        raise ValueError(f"stage must be one of {ACQUISITION_STAGES!r}")
    records = _validated_records(prompt_records, stage=stage)
    canonical_masks = _validated_masks(masks)
    identifiers = _validated_mask_ids(mask_ids, rows=len(canonical_masks))
    commitments = _validated_commitments(protocol_commitments)
    if isinstance(nomination_provenance, (str, bytes)) or not isinstance(
        nomination_provenance, Sequence
    ):
        raise TypeError("nomination_provenance must be a sequence")
    frozen_provenance = tuple(deepcopy(tuple(nomination_provenance)))

    # Fail closed on every clean prompt before exposing any patched outcome.
    prepared = tuple(runner.prepare_context(record) for record in records)
    viability = clean_panel_viability_report(prepared)
    if not viability["passed"]:
        raise CleanPanelViabilityError(viability)
    normalized_columns = []
    raw_margin_columns = []
    raw_shift_columns = []
    for context in prepared:
        responses = runner.responses_from_prepared(context, canonical_masks)
        normalized_columns.append(responses.normalized_transfer)
        raw_margin_columns.append(responses.raw_donor_minus_base_margin)
        raw_shift_columns.append(responses.raw_margin_shift_from_base)
    normalized = torch.stack(normalized_columns, dim=1).double()
    raw_margins = torch.stack(raw_margin_columns, dim=1).double()
    raw_shifts = torch.stack(raw_shift_columns, dim=1).double()

    smoke = None
    canonical_smoke = None
    smoke_rows = 0
    if smoke_masks is not None:
        canonical_smoke = _validated_masks(smoke_masks)
        smoke_rows = len(canonical_smoke)
        # One fixed context is sufficient to detect batch-dependent execution;
        # using all contexts would consume outcome calls without a new gate.
        smoke = runner.batch_invariance_smoke(prepared[0], canonical_smoke)
        smoke["context_index"] = 0
        smoke["pair_id"] = records[0]["pair_id"]

    summaries = {
        "normalized_responses": _numerical_summary(normalized),
        "raw_donor_minus_base_margins": _numerical_summary(raw_margins),
        "raw_margin_shifts_from_base": _numerical_summary(raw_shifts),
    }
    numerical_passed = all(summary["finite"] for summary in summaries.values())
    batch_passed = smoke is None or bool(smoke["passed"])
    clean_endpoints = torch.tensor(
        [context.clean_endpoint for context in prepared], dtype=torch.float64
    )
    return {
        "schema_version": "e29_15_ioi_response_panel_v1",
        "stage": stage,
        "response_coordinate": RESPONSE_COORDINATE,
        "pair_ids": tuple(record["pair_id"] for record in records),
        "prompt_records": records,
        "mask_ids": identifiers,
        "masks": canonical_masks,
        "component_map": tuple(runner.components),
        "normalized_responses": normalized,
        "raw_donor_minus_base_margins": raw_margins,
        "raw_margin_shifts_from_base": raw_shifts,
        "clean_base_donor_minus_base_margins": torch.tensor(
            [context.clean_base_donor_minus_base_margin for context in prepared],
            dtype=torch.float64,
        ),
        "clean_donor_donor_minus_base_margins": torch.tensor(
            [context.clean_donor_donor_minus_base_margin for context in prepared],
            dtype=torch.float64,
        ),
        "clean_endpoints": clean_endpoints,
        "clean_metadata": tuple(dict(context.clean_metadata) for context in prepared),
        "clean_panel_viability": viability,
        "nomination_provenance": frozen_provenance,
        "protocol_commitments": commitments,
        "batch_invariance_smoke": smoke,
        "batch_smoke_masks": canonical_smoke,
        "numerical_summary": summaries,
        "query_accounting": {
            "context_count": len(prepared),
            "mask_rows": len(canonical_masks),
            "patched_context_mask_cells": len(prepared) * len(canonical_masks),
            "clean_forward_examples": 2 * len(prepared),
            "batch_smoke_mask_rows": smoke_rows,
            "batch_smoke_physical_context_mask_cells": 2 * smoke_rows,
        },
        "all_integrity_gates_passed": numerical_passed and batch_passed,
    }


__all__ = [
    "ACQUISITION_STAGES",
    "MINIMUM_ENDPOINT_CORRECT_FRACTION",
    "AcquisitionStage",
    "CleanPanelViabilityError",
    "acquire_ioi_response_panel",
    "clean_panel_viability_report",
]
