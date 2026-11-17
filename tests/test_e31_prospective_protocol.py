# ruff: noqa: EXE002
from __future__ import annotations

from collections import Counter
from itertools import combinations

from experiments.e31_sparse_interactions.prospective_protocol import (
    ASINH_LAMBDA_GRID,
    COMPONENT_COUNT,
    COVER_TARGET,
    HIERARCHICAL_SCALE_GATE,
    HUB_COMPONENT,
    NONHUB_COMPONENTS,
    NONHUB_EDGE_COUNT,
    PRIMARY_GATES,
    PRIMARY_STRATUM,
    PROTOCOL_SHA256,
    SCALE_CARDINALITIES,
    SMOKE_MASK_COUNT,
    SPLIT_MANIFEST_SHA256,
    SPOKE_COUNT,
    build_covering_background_manifest,
    build_prospective_protocol,
    build_split_mask_manifest,
    canonical_sha256,
    validate_prospective_protocol,
)


def test_hash_greedy_covers_every_edge_four_times_at_exact_scales() -> None:
    nonhub_edges = set(combinations(NONHUB_COMPONENTS, 2))
    assert len(nonhub_edges) == NONHUB_EDGE_COUNT == 253
    assert COMPONENT_COUNT == 24 and HUB_COMPONENT == 1 and SPOKE_COUNT == 23

    for split in ("discovery", "confirmation"):
        rows = build_covering_background_manifest(split)
        for k in SCALE_CARDINALITIES:
            scale_rows = [row for row in rows if row["k_nonhub_on"] == k]
            coverage = Counter()
            spoke_coverage = Counter()
            for row in scale_rows:
                background = set(row["selected_components"])
                assert len(background) == k
                assert HUB_COMPONENT not in background
                absent = sorted(set(NONHUB_COMPONENTS) - background)
                coverage.update(combinations(absent, 2))
                spoke_coverage.update(absent)
            assert min(coverage[edge] for edge in nonhub_edges) >= COVER_TARGET
            assert (
                min(spoke_coverage[node] for node in NONHUB_COMPONENTS) >= COVER_TARGET
            )


def test_split_manifests_have_exact_quartets_and_global_mask_deduplication() -> None:
    for split in ("discovery", "confirmation"):
        manifest = build_split_mask_manifest(split)
        masks = [tuple(row["mask"]) for row in manifest["conditions"]]
        assert len(masks) == len(set(masks)) == manifest["unique_condition_count"]
        assert all(len(mask) == COMPONENT_COUNT for mask in masks)
        assert all(value in (0, 1) for mask in masks for value in mask)
        condition = {
            row["condition_id"]: set(row["selected_components"])
            for row in manifest["conditions"]
        }
        assert manifest["contrast_count"] == (
            NONHUB_EDGE_COUNT * len(SCALE_CARDINALITIES) * 2 * COVER_TARGET
            + SPOKE_COUNT * len(SCALE_CARDINALITIES) * COVER_TARGET
        )
        for contrast in manifest["contrasts"]:
            first, second = contrast["factor_components"]
            cells = [condition[value] for value in contrast["condition_ids"]]
            assert contrast["ordered_cell_bits"] == ["00", "10", "01", "11"]
            assert first not in cells[0] and second not in cells[0]
            assert cells[1] == cells[0] | {first}
            assert cells[2] == cells[0] | {second}
            assert cells[3] == cells[0] | {first, second}
            assert len(cells[0] - {HUB_COMPONENT}) == contrast["k_nonhub_on"]
            if contrast["edge_class"] == "nonhub_pair":
                assert contrast["hub_state"] in ("off", "on")
                assert (HUB_COMPONENT in cells[0]) == (contrast["hub_state"] == "on")
            else:
                assert contrast["hub_state"] == "crossing"
                assert first == HUB_COMPONENT

        assert "prediction_masks" not in manifest
        assert "prediction_mask_count" not in manifest
        assert canonical_sha256(manifest) == SPLIT_MANIFEST_SHA256[split]


def test_protocol_freezes_scoring_gates_and_query_accounting() -> None:
    protocol = validate_prospective_protocol(build_prospective_protocol())

    assert ASINH_LAMBDA_GRID == (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    assert PRIMARY_STRATUM == {"hub_state": "off", "k_nonhub_on": 10}
    assert PRIMARY_GATES["top24_confirmation_energy_coverage_minimum"] == 0.60
    assert PRIMARY_GATES["top48_confirmation_energy_coverage_minimum"] == 0.80
    assert PRIMARY_GATES["replicated_top24_sign_count_minimum"] == 18
    assert (
        PRIMARY_GATES[
            "signed_interaction_pearson_one_sided_95_lower_bound_strict_minimum"
        ]
        == 0.0
    )
    assert "prediction" not in protocol
    assert "topology" not in protocol
    assert HIERARCHICAL_SCALE_GATE["regime_id"] == "nonhub_k02_hub_off"
    assert protocol["decision"]["hierarchical_scale_gate"] == HIERARCHICAL_SCALE_GATE
    assert protocol["query_accounting"]["total_clean_forwards"] == 96
    assert SMOKE_MASK_COUNT == 16
    assert protocol["query_accounting"]["batch_smoke_physical_context_mask_cells"] == 32
    assert protocol["query_accounting"]["maximum_registered_model_forwards"] == 373_568
    assert protocol["clean_screen"]["row_policy"] == "retain_every_registered_context"
    assert (
        protocol["clean_screen"]["all_splits_screened_before_any_patched_call"] is True
    )
    assert protocol["development_ancestry"]["development_outcomes_informed"]
    assert protocol["custody_order"][3].startswith("fit and write-once seal discovery")
    assert protocol["query_accounting"]["maximum_registered_model_forwards"] < 700_000
    assert canonical_sha256(protocol) == PROTOCOL_SHA256
