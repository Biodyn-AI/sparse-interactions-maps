# ruff: noqa: EXE002
from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.e31_sparse_interactions.mixed_resolution_panel import PROMPT_COUNTS
from experiments.e31_sparse_interactions.mixed_resolution_protocol import (
    ATLAS_MANIFEST_SHA256,
    CELL_BITS,
    HEAD_COUNT,
    PARENT_PAIRS,
    PRIMARY_GATES,
    PROTOCOL_SHA256,
    build_atlas_manifest,
    build_background_manifest,
    build_joint_matched_rectangle_controls,
    build_protocol,
    build_reconstruction_manifest,
    canonical_sha256,
    validate_atlas_manifest,
    validate_reconstruction_selections,
)


def _selections(*, identical_methods: bool = False) -> dict:
    records = []
    for parent_index, parent in enumerate(PARENT_PAIRS):
        pair_a = tuple(range(parent_index, parent_index + 4))
        pair_b = tuple(range(4 + parent_index, 8 + parent_index))
        singleton_a = pair_a if identical_methods else (0, 2, 4, 6)
        singleton_b = pair_b if identical_methods else (1, 3, 5, 7)
        records.append(
            {
                "parent_pair_id": parent["parent_pair_id"],
                "pair_selected": {
                    "heads_a": list(pair_a),
                    "heads_b": list(pair_b),
                },
                "singleton_selected": {
                    "heads_a": list(singleton_a),
                    "heads_b": list(singleton_b),
                },
            }
        )
    return {
        "schema_version": "e31_2_discovery_rectangle_selections_v1",
        "parents": records,
    }


def test_backgrounds_are_exact_response_blind_k10_hub_off() -> None:
    expected_parent_ids = tuple(parent["parent_pair_id"] for parent in PARENT_PAIRS)
    assert expected_parent_ids == ("l05a_l08a", "l08a_l09a", "l05a_l10a")
    split_masks = {}
    for split in ("discovery", "confirmation"):
        backgrounds = build_background_manifest(split)
        assert len(backgrounds) == 3 * 4
        assert tuple(dict.fromkeys(row["parent_pair_id"] for row in backgrounds)) == (
            expected_parent_ids
        )
        assert len({row["background_id"] for row in backgrounds}) == len(backgrounds)
        assert len(
            {tuple(row["coarse_background_mask"]) for row in backgrounds}
        ) == len(backgrounds)
        for row in backgrounds:
            parent = next(
                item
                for item in PARENT_PAIRS
                if item["parent_pair_id"] == row["parent_pair_id"]
            )
            mask = row["coarse_background_mask"]
            assert len(mask) == 24
            assert sum(mask) == 10
            assert mask[1] == 0
            assert mask[parent["component_a"]] == 0
            assert mask[parent["component_b"]] == 0
            assert row["k_nonhub_on"] == 10
            assert row["hub_state"] == "off"
            assert row["background_sha256"] == canonical_sha256(mask)
        split_masks[split] = {
            tuple(row["coarse_background_mask"]) for row in backgrounds
        }
    assert split_masks["discovery"].isdisjoint(split_masks["confirmation"])


def test_atlas_manifest_is_exact_172_condition_factorial_per_parent_cover() -> None:
    for split in ("discovery", "confirmation"):
        manifest = build_atlas_manifest(split)
        assert validate_atlas_manifest(manifest) == manifest
        assert canonical_sha256(manifest) == ATLAS_MANIFEST_SHA256[split]
        assert manifest["unique_condition_count"] == 2_064
        assert len(manifest["conditions"]) == 2_064
        assert len(manifest["fine_contrasts"]) == 3 * 4 * 144
        assert len(manifest["coarse_contrasts"]) == 3 * 4
        assert manifest["logical_fine_cell_reference_count"] == 6_912
        assert manifest["logical_coarse_cell_reference_count"] == 48
        assert len({row["condition_id"] for row in manifest["conditions"]}) == 2_064

        by_id = {row["condition_id"]: row for row in manifest["conditions"]}
        for row in manifest["conditions"]:
            assert row["condition_index"] >= 0
            assert len(row["coarse_component_mask"]) == 24
            assert len(row["attention_head_mask"]) == 12
            assert all(len(heads) == HEAD_COUNT for heads in row["attention_head_mask"])
            for layer, head_bits in enumerate(row["attention_head_mask"]):
                if row["coarse_component_mask"][2 * layer]:
                    assert not any(head_bits)
        for contrast in manifest["fine_contrasts"]:
            assert contrast["ordered_cell_bits"] == list(CELL_BITS)
            assert contrast["pair_index"] == (
                HEAD_COUNT * contrast["head_a"] + contrast["head_b"]
            )
            assert all(identifier in by_id for identifier in contrast["condition_ids"])

        for parent in PARENT_PAIRS:
            for cover_rank in range(4):
                refs = {
                    condition_id
                    for row in manifest["fine_contrasts"]
                    if row["parent_pair_id"] == parent["parent_pair_id"]
                    and row["cover_rank"] == cover_rank
                    for condition_id in row["condition_ids"]
                }
                coarse = next(
                    row
                    for row in manifest["coarse_contrasts"]
                    if row["parent_pair_id"] == parent["parent_pair_id"]
                    and row["cover_rank"] == cover_rank
                )
                refs.update(coarse["condition_ids"])
                assert len(refs) == 172


def test_reconstruction_manifest_is_d_sealed_and_bounded() -> None:
    selections = validate_reconstruction_selections(_selections())
    manifest = build_reconstruction_manifest(selections)
    assert manifest["split"] == "confirmation"
    assert manifest["support_order"] == ["pair_selected", "singleton_selected"]
    assert len(manifest["contrasts"]) == 3 * 4 * 2
    assert manifest["maximum_new_condition_count"] == 72
    assert len(manifest["new_conditions"]) <= 72
    assert all(
        row["ordered_cell_bits"] == list(CELL_BITS) for row in manifest["contrasts"]
    )
    assert all(
        row["condition_ids"][0].startswith("e31_2_confirmation_")
        for row in manifest["contrasts"]
    )

    deduplicated = build_reconstruction_manifest(
        validate_reconstruction_selections(_selections(identical_methods=True))
    )
    assert len(deduplicated["new_conditions"]) == 3 * 4 * 3

    changed = deepcopy(_selections())
    changed["parents"][0]["pair_selected"]["heads_a"] = [0, 1, 1, 2]
    with pytest.raises(ValueError, match="four sorted unique heads"):
        validate_reconstruction_selections(changed)


def test_matched_controls_are_unique_joint_response_blind_rectangles() -> None:
    selections = validate_reconstruction_selections(_selections())
    controls = build_joint_matched_rectangle_controls(selections, count=100)
    assert controls["support_count"] == 100
    assert controls["uses_model_outcomes"] is False
    assert len(controls["supports"]) == 100
    assert len({canonical_sha256(row) for row in controls["supports"]}) == 100
    excluded = {
        canonical_sha256(
            [
                {
                    "parent_pair_id": row["parent_pair_id"],
                    **row[support_name],
                }
                for row in selections["parents"]
            ]
        )
        for support_name in ("pair_selected", "singleton_selected")
    }
    for support in controls["supports"]:
        assert canonical_sha256(support) not in excluded
        assert [row["parent_pair_id"] for row in support] == [
            parent["parent_pair_id"] for parent in PARENT_PAIRS
        ]
        assert all(len(row["heads_a"]) == len(row["heads_b"]) == 4 for row in support)


def test_protocol_freezes_raw_primary_gates_and_query_ceiling() -> None:
    protocol = build_protocol()
    assert canonical_sha256(protocol) == PROTOCOL_SHA256
    ancestry = protocol["parent_selection_ancestry"]
    assert ancestry["e31_1_v2_score_artifact_sha256"] == (
        "ddde7dcdef70525f8fdc3ccc75fe4c57852f87be805b4406a8c716624be57ba5"
    )
    assert ancestry["discovery_only_selection_claimed"] is False
    assert ancestry["e31_1_confirmation_had_already_been_inspected"] is True
    assert ancestry["e31_2_outcomes_used"] is False
    assert protocol["response"]["primary_interaction"] == "R11-R10-R01+R00"
    assert protocol["response"]["saturation_residualization"] is False
    assert protocol["primary_gates"] == PRIMARY_GATES
    assert PRIMARY_GATES["per_parent_sign_replication_count_minimum"] == 12
    assert PRIMARY_GATES["per_parent_energy_spearman_minimum"] == 0.5
    assert protocol["query_accounting"] == {
        "discovery_atlas_patch_cells": 2_064 * PROMPT_COUNTS["discovery"],
        "confirmation_atlas_patch_cells": 2_064 * PROMPT_COUNTS["confirmation"],
        "maximum_confirmation_reconstruction_patch_cells": 72
        * PROMPT_COUNTS["confirmation"],
        "clean_forward_examples": 2 * sum(PROMPT_COUNTS.values()),
        "batch_smoke_physical_context_mask_cells": 32,
        "maximum_registered_model_forwards": 101_504,
    }
