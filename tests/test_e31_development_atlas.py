# ruff: noqa: EXE002
from __future__ import annotations

from copy import deepcopy

import torch

from experiments.e31_sparse_interactions.development_atlas import (
    CLAIM_BOUNDARY,
    acquire_missing_masks,
    build_parent_reuse_plan,
    load_sealed_e29_15_confirmation,
    merge_exploratory_atlas,
)
from experiments.e31_sparse_interactions.protocol import (
    build_exhaustive_pair_atlas,
    canonical_sha256,
)
from experiments.e31_sparse_interactions.run_development_atlas import (
    build_dry_run_report,
)


def test_real_sealed_parent_validates_and_reuses_only_exact_mask_rows() -> None:
    parent = load_sealed_e29_15_confirmation()
    atlas = build_exhaustive_pair_atlas()
    plan = build_parent_reuse_plan(parent, atlas)

    assert parent["labels"]["masks"].shape == (2_006, 24)
    assert parent["labels"]["normalized_responses"].shape == (2_006, 32)
    assert plan["atlas_mask_count"] == 2_408
    assert plan["reused_mask_count"] == 843
    assert plan["missing_mask_count"] == 1_565
    assert plan["reused_context_mask_cells"] == 26_976
    assert plan["missing_context_mask_cells"] == 50_080
    assert len(set(plan["missing_condition_ids"])) == 1_565
    assert torch.equal(
        plan["missing_masks"],
        plan["atlas_masks"][plan["missing_atlas_indices"]],
    )


def test_dry_run_report_is_model_free_and_exposes_exact_future_work() -> None:
    report = build_dry_run_report()

    assert report["status"] == "dry_run_no_model_call_no_write"
    assert report["atlas"]["quartets"] == 2_208
    assert report["atlas"]["logical_cell_references"] == 8_832
    assert report["atlas"]["unique_masks"] == 2_408
    assert report["reuse"]["sealed_parent_masks"] == 843
    assert report["reuse"]["missing_masks"] == 1_565
    assert report["reuse"]["missing_context_mask_cells"] == 50_080
    assert report["claim_boundary"] == CLAIM_BOUNDARY


def test_missing_acquisition_receives_only_uncovered_masks_and_old_prompts() -> None:
    parent = load_sealed_e29_15_confirmation()
    atlas = build_exhaustive_pair_atlas()
    plan = build_parent_reuse_plan(parent, atlas)
    sentinel_runner = object()
    captured: dict[str, object] = {}

    def fake_acquire(runner: object, records: object, masks: torch.Tensor, **kwargs):
        captured.update(
            runner=runner,
            records=records,
            masks=masks.clone(),
            kwargs=deepcopy(kwargs),
        )
        return {"sentinel": True}

    output = acquire_missing_masks(
        sentinel_runner,
        parent,
        atlas,
        plan,
        acquire_fn=fake_acquire,
    )

    assert output == {"sentinel": True}
    assert captured["runner"] is sentinel_runner
    assert captured["records"] == parent["labels"]["prompt_records"]
    assert torch.equal(captured["masks"], plan["missing_masks"])
    kwargs = captured["kwargs"]
    assert kwargs["stage"] == "confirmation"
    assert tuple(kwargs["mask_ids"]) == tuple(plan["missing_condition_ids"])
    assert torch.equal(kwargs["smoke_masks"], plan["missing_masks"][:64])
    assert kwargs["protocol_commitments"]["atlas_manifest_sha256"] == (
        canonical_sha256(atlas)
    )


def test_merge_preserves_parent_and_new_rows_with_explicit_exploratory_provenance() -> (
    None
):
    parent = load_sealed_e29_15_confirmation()
    atlas = build_exhaustive_pair_atlas()
    plan = build_parent_reuse_plan(parent, atlas)
    labels = parent["labels"]
    missing_count = plan["missing_mask_count"]
    context_count = len(labels["pair_ids"])
    new_normalized = torch.arange(
        missing_count * context_count, dtype=torch.float64
    ).reshape(missing_count, context_count)
    new_raw_margin = new_normalized + 10_000.0
    new_raw_shift = new_normalized - 10_000.0
    missing = {
        "schema_version": "e29_15_ioi_response_panel_v1",
        "stage": "confirmation",
        "response_coordinate": labels["response_coordinate"],
        "pair_ids": labels["pair_ids"],
        "prompt_records": labels["prompt_records"],
        "mask_ids": tuple(plan["missing_condition_ids"]),
        "masks": plan["missing_masks"].clone(),
        "component_map": labels["component_map"],
        "normalized_responses": new_normalized,
        "raw_donor_minus_base_margins": new_raw_margin,
        "raw_margin_shifts_from_base": new_raw_shift,
        "clean_base_donor_minus_base_margins": labels[
            "clean_base_donor_minus_base_margins"
        ].clone(),
        "clean_donor_donor_minus_base_margins": labels[
            "clean_donor_donor_minus_base_margins"
        ].clone(),
        "clean_endpoints": labels["clean_endpoints"].clone(),
        "clean_metadata": labels["clean_metadata"],
        "clean_panel_viability": labels["clean_panel_viability"],
        "nomination_provenance": (),
        "protocol_commitments": {"atlas_manifest_sha256": canonical_sha256(atlas)},
        "batch_invariance_smoke": {"passed": True},
        "batch_smoke_masks": plan["missing_masks"][:64],
        "numerical_summary": {},
        "query_accounting": {
            "context_count": context_count,
            "mask_rows": missing_count,
            "patched_context_mask_cells": missing_count * context_count,
        },
        "all_integrity_gates_passed": True,
    }

    merged = merge_exploratory_atlas(parent, atlas, plan, missing)

    assert merged["status"] == "complete_exploratory_development_atlas"
    assert merged["claim_boundary"] == CLAIM_BOUNDARY
    assert merged["masks"].shape == (2_408, 24)
    assert merged["normalized_responses"].shape == (2_408, 32)
    assert merged["provenance"]["reused_mask_count"] == 843
    assert merged["provenance"]["newly_acquired_mask_count"] == 1_565
    reused_atlas = plan["reused_atlas_indices"]
    reused_parent = plan["parent_row_by_atlas_index"][reused_atlas]
    torch.testing.assert_close(
        merged["normalized_responses"][reused_atlas],
        labels["normalized_responses"][reused_parent],
    )
    torch.testing.assert_close(
        merged["normalized_responses"][plan["missing_atlas_indices"]],
        new_normalized,
    )
    assert set(merged["row_provenance"]) == {
        "source_kind",
        "source_row_index",
    }
    assert merged["quartets"] == atlas["quartets"]
