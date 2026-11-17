# ruff: noqa: EXE002
from __future__ import annotations

from itertools import combinations

import torch

from experiments.e31_sparse_interactions.protocol import (
    BACKGROUND_COUNT,
    COMPONENT_COUNT,
    EXPECTED_BACKGROUND_MANIFEST_SHA256,
    PAIR_COUNT,
    build_exhaustive_pair_atlas,
    canonical_sha256,
    validate_exhaustive_pair_atlas,
)


def test_exact_exhaustive_pair_atlas_is_deduplicated_and_deterministic() -> None:
    atlas = validate_exhaustive_pair_atlas(build_exhaustive_pair_atlas())

    assert atlas["component_count"] == COMPONENT_COUNT == 24
    assert atlas["background_count"] == BACKGROUND_COUNT == 8
    assert atlas["pair_count_per_background"] == PAIR_COUNT == 276
    assert atlas["quartet_count"] == BACKGROUND_COUNT * PAIR_COUNT == 2_208
    assert atlas["logical_cell_reference_count"] == 8_832
    assert atlas["unique_condition_count"] == 2_408
    assert atlas["background_manifest_sha256"] == (EXPECTED_BACKGROUND_MANIFEST_SHA256)
    assert canonical_sha256(atlas) == canonical_sha256(build_exhaustive_pair_atlas())

    condition_ids = tuple(row["condition_id"] for row in atlas["conditions"])
    masks = torch.tensor(
        [row["mask"] for row in atlas["conditions"]], dtype=torch.uint8
    )
    assert len(condition_ids) == len(set(condition_ids)) == 2_408
    assert len(torch.unique(masks, dim=0)) == 2_408
    assert masks.shape == (2_408, 24)


def test_every_background_pair_has_an_exact_ordered_four_cell_lattice() -> None:
    atlas = build_exhaustive_pair_atlas()
    conditions = {
        row["condition_id"]: set(row["selected_components"])
        for row in atlas["conditions"]
    }
    expected_pairs = tuple(combinations(range(COMPONENT_COUNT), 2))

    for background_index in range(BACKGROUND_COUNT):
        rows = [
            row
            for row in atlas["quartets"]
            if row["background_index"] == background_index
        ]
        assert tuple(tuple(row["pair"]) for row in rows) == expected_pairs
        for row in rows:
            first, second = row["pair"]
            assert row["ordered_cell_bits"] == ["00", "10", "01", "11"]
            assert row["edge_class"] == (
                "hub_spoke" if 1 in (first, second) else "nonhub_pair"
            )
            cells = [conditions[value] for value in row["condition_ids"]]
            assert first not in cells[0] and second not in cells[0]
            assert cells[1] == cells[0] | {first}
            assert cells[2] == cells[0] | {second}
            assert cells[3] == cells[0] | {first, second}


def test_reverse_condition_references_preserve_every_logical_cell() -> None:
    atlas = build_exhaustive_pair_atlas()
    expected = {
        (quartet["quartet_id"], bits, condition_id)
        for quartet in atlas["quartets"]
        for bits, condition_id in zip(
            quartet["ordered_cell_bits"], quartet["condition_ids"], strict=True
        )
    }
    observed = {
        (reference["quartet_id"], reference["cell_bits"], condition["condition_id"])
        for condition in atlas["conditions"]
        for reference in condition["quartet_references"]
    }

    assert observed == expected
    assert len(observed) == 8_832
