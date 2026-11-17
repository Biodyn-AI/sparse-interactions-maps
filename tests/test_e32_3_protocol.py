# ruff: noqa: EXE002
"""Tests for the response-prospective E32.3 protocol."""

from __future__ import annotations

import math

import pytest

from experiments.e32_interpbench.e32_3_protocol import (
    CASE_ORDER,
    CONFIRMATORY_SUPPORT_FRACTION,
    DETERMINISTIC_METHOD_ORDER,
    EXPECTED_QUERY_ACCOUNTING,
    EXPECTED_UNIVERSE_COUNTS,
    RANDOM_CONTROL_REPLICATES,
    RANDOM_CONTROL_SEED,
    SCREEN_GATES,
    ZERO_MASS_POLICY,
    deterministic_random_support,
    protocol_manifest,
    split_request,
    support_sizes,
)


def test_frozen_cohort_splits_and_methods() -> None:
    assert CASE_ORDER == (58, 129, 25)
    assert CONFIRMATORY_SUPPORT_FRACTION == 0.5
    assert DETERMINISTIC_METHOD_ORDER == (
        "singleton_output_effect",
        "activation_shift_rms",
        "raw_pair_incident_mean",
        "rank_one_incident_mean",
        "eap_ig5_incident_mean",
    )
    assert RANDOM_CONTROL_REPLICATES == 64
    assert RANDOM_CONTROL_SEED == 32_300_064

    discovery = split_request(58, "discovery")
    confirmation = split_request(58, "confirmation")
    assert (discovery.required_count, discovery.candidate_count) == (64, 96)
    assert (discovery.clean_seed, discovery.corrupt_seed) == (758100, 758101)
    assert (confirmation.required_count, confirmation.candidate_count) == (128, 192)
    assert (confirmation.clean_seed, confirmation.corrupt_seed) == (758200, 758201)
    assert discovery.excluded_smoke_clean_seed == 758000
    assert discovery.excluded_smoke_corrupt_seed == 758001


@pytest.mark.parametrize(
    ("node_count", "expected"),
    ((15, (0, 4, 8, 12, 15)), (10, (0, 3, 5, 8, 10))),
)
def test_support_sizes_use_frozen_ceiling_rule(
    node_count: int, expected: tuple[int, ...]
) -> None:
    assert support_sizes(node_count) == expected
    assert expected[2] == math.ceil(CONFIRMATORY_SUPPORT_FRACTION * node_count)


def test_sha_random_controls_are_exact_and_reproducible() -> None:
    nodes = tuple(f"n{index}" for index in range(15))
    supports = tuple(
        deterministic_random_support(
            case_id=58,
            component_ids=nodes,
            support_size=8,
            replicate=replicate,
        )
        for replicate in range(RANDOM_CONTROL_REPLICATES)
    )
    assert all(len(support) == 8 and len(set(support)) == 8 for support in supports)
    assert supports == tuple(
        deterministic_random_support(
            case_id=58,
            component_ids=nodes,
            support_size=8,
            replicate=replicate,
        )
        for replicate in range(RANDOM_CONTROL_REPLICATES)
    )
    assert len(set(supports)) > 60
    assert supports[0] == ("n2", "n4", "n6", "n8", "n9", "n10", "n11", "n12")

    with pytest.raises(ValueError, match="replicate"):
        deterministic_random_support(
            case_id=58,
            component_ids=nodes,
            support_size=8,
            replicate=RANDOM_CONTROL_REPLICATES,
        )


def test_protocol_gates_and_disclosure_are_frozen() -> None:
    assert SCREEN_GATES == {
        "maximum_induced_whole_edge_fraction": 0.30,
        "maximum_induced_fine_edge_fraction": 0.30,
        "minimum_confirmation_raw_mass_retention": 0.75,
        "minimum_confirmation_exact_mass_retention": 0.75,
        "maximum_discovery_confirmation_retention_gap": 0.10,
        "minimum_singleton_spearman": 0.80,
        "minimum_singleton_top_half_jaccard": 0.60,
        "minimum_raw_mass_spearman": 0.80,
        "minimum_exact_mass_spearman": 0.80,
    }
    request = split_request(129, "discovery")
    assert request.ground_truth_disclosure == (
        "public_ground_truth_known_pre_run;used_only_for_cohort_nontriviality_organism_"
        "screen;forbidden_for_within_case_selector_support_mask_threshold_primary_"
        "metric_or_verdict"
    )
    assert EXPECTED_UNIVERSE_COUNTS[25] == {
        "layers": 2,
        "heads": 4,
        "nodes": 10,
        "whole_edges": 33,
        "fine_edges": 73,
    }
    manifest = protocol_manifest()
    assert manifest["mandatory_all_cases"] is True
    assert manifest["zero_mass_policy"] == ZERO_MASS_POLICY
    assert manifest["ground_truth_used_for_cohort_selection"] is True
    assert (
        manifest["ground_truth_used_for_within_case_selection_or_primary_scoring"]
        is False
    )
    assert "hidden" not in str(manifest).lower()


def test_exact_query_accounting_is_frozen_in_registration() -> None:
    assert EXPECTED_QUERY_ACCOUNTING == {
        58: {
            "discovery": {
                "physical_forwards": 444,
                "model_examples": 35_200,
                "backward_passes_inferred_from_pinned_control_flow": 6,
                "gradient_examples": 384,
            },
            "confirmation": {
                "physical_forwards": 440,
                "model_examples": 86_656,
                "backward_passes_inferred_from_pinned_control_flow": 6,
                "gradient_examples": 768,
            },
        },
        129: {
            "discovery": {
                "physical_forwards": 444,
                "model_examples": 35_200,
                "backward_passes_inferred_from_pinned_control_flow": 6,
                "gradient_examples": 384,
            },
            "confirmation": {
                "physical_forwards": 440,
                "model_examples": 86_656,
                "backward_passes_inferred_from_pinned_control_flow": 6,
                "gradient_examples": 768,
            },
        },
        25: {
            "discovery": {
                "physical_forwards": 174,
                "model_examples": 14_272,
                "backward_passes_inferred_from_pinned_control_flow": 6,
                "gradient_examples": 384,
            },
            "confirmation": {
                "physical_forwards": 170,
                "model_examples": 44_800,
                "backward_passes_inferred_from_pinned_control_flow": 6,
                "gradient_examples": 768,
            },
        },
        "overall": {
            "physical_forwards": 2_112,
            "model_examples": 302_784,
            "backward_passes_inferred_from_pinned_control_flow": 36,
            "gradient_examples": 3_456,
        },
    }
    manifest = protocol_manifest()
    assert manifest["expected_query_accounting"] == {
        str(key): value for key, value in EXPECTED_QUERY_ACCOUNTING.items()
    }


def test_every_case_has_exact_frozen_scientific_and_smoke_seeds() -> None:
    expected = {
        58: ((758000, 758001), (758100, 758101), (758200, 758201)),
        129: ((829000, 829001), (829100, 829101), (829200, 829201)),
        25: ((725000, 725001), (725100, 725101), (725200, 725201)),
    }
    for case_id, (smoke, discovery, confirmation) in expected.items():
        d_request = split_request(case_id, "discovery")
        c_request = split_request(case_id, "confirmation")
        assert (
            d_request.excluded_smoke_clean_seed,
            d_request.excluded_smoke_corrupt_seed,
        ) == smoke
        assert (d_request.clean_seed, d_request.corrupt_seed) == discovery
        assert (c_request.clean_seed, c_request.corrupt_seed) == confirmation
        assert d_request.freshness_policy == c_request.freshness_policy
