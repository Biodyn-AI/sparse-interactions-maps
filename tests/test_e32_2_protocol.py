# ruff: noqa: EXE002
"""Pure registration and cost tests for the prospective E32.2 protocol."""

from __future__ import annotations

import pytest

from experiments.e32_interpbench.e32_2_protocol import (
    CASE_ORDER,
    COMPARATOR_METHODS,
    DEVELOPMENT_CASE_ID,
    EXPECTED_UNIVERSE_COUNTS,
    FULL_UNIVERSE_FLOOR,
    METHOD_ORDER,
    PRIMARY_CASE_ID,
    PROSPECTIVE_CASE_IDS,
    REPLICATION_CASE_ID,
    SCREEN_GATES,
    SUPERIORITY_GATES,
    USEFUL_GATES,
    operational_query_accounting,
    protocol_dry_run_manifest,
    split_request,
    validation_query_accounting,
)


def test_cases_panels_and_corrected_case21_seed_are_frozen() -> None:
    assert DEVELOPMENT_CASE_ID == 97
    assert PRIMARY_CASE_ID == 2
    assert REPLICATION_CASE_ID == 21
    assert PROSPECTIVE_CASE_IDS == (2, 21)
    assert CASE_ORDER == (2, 21)

    case2_d = split_request(2, "discovery")
    case2_c = split_request(2, "confirmation")
    case21_d = split_request(21, "discovery")
    case21_c = split_request(21, "confirmation")

    assert (case2_d.required_count, case2_d.candidate_count) == (64, 80)
    assert (case2_c.required_count, case2_c.candidate_count) == (128, 160)
    assert (case2_d.clean_seed, case2_d.corrupt_seed) == (52_032, 52_033)
    assert (case2_c.clean_seed, case2_c.corrupt_seed) == (52_034, 52_035)
    assert (case21_d.clean_seed, case21_d.corrupt_seed) == (62_032, 62_033)
    # 62034/62035 yield only 159 official unique rows and are forbidden.
    assert (case21_c.clean_seed, case21_c.corrupt_seed) == (62_040, 62_041)

    with pytest.raises(ValueError, match="case_id"):
        split_request(113, "discovery")
    with pytest.raises(ValueError, match="split"):
        split_request(2, "test")


def test_universes_shortlist_and_gate_constants_are_exact() -> None:
    assert EXPECTED_UNIVERSE_COUNTS[2] == {
        "nodes": 20,
        "whole_edges": 166,
        "fine_edges": 406,
        "ground_truth_nodes": 5,
        "ground_truth_edges": 4,
        "shortlist_nodes": 10,
        "minimum_induced_whole_edges": 32,
        "maximum_induced_whole_edges": 43,
        "minimum_induced_fine_edges": 58,
        "maximum_induced_fine_edges": 113,
    }
    assert EXPECTED_UNIVERSE_COUNTS[21]["nodes"] == 20
    assert EXPECTED_UNIVERSE_COUNTS[21]["ground_truth_edges"] == 2
    assert EXPECTED_UNIVERSE_COUNTS[21]["shortlist_nodes"] == 10
    assert EXPECTED_UNIVERSE_COUNTS[97]["shortlist_nodes"] == 8

    assert FULL_UNIVERSE_FLOOR == -1.0
    assert METHOD_ORDER[0] == "bidirectional_fine_replacement"
    assert set(COMPARATOR_METHODS).issubset(METHOD_ORDER)
    assert SCREEN_GATES == {
        "minimum_direct_edge_retention": 1.0,
        "maximum_induced_whole_edge_fraction": 0.30,
        "maximum_induced_fine_edge_fraction": 0.30,
        "minimum_operational_model_example_savings": 0.60,
    }
    assert USEFUL_GATES["minimum_confirmation_auprc"] == 0.90
    assert USEFUL_GATES["require_perfect_true_positives_at_gt_k"] is True
    assert SUPERIORITY_GATES["minimum_auprc_margin_over_every_comparator"] == 0.05
    assert SUPERIORITY_GATES["true_positive_improvement_capped_at_perfect"] is True


@pytest.mark.parametrize(
    ("fine_count", "expected_examples", "expected_savings"),
    ((58, 27_744, 1.0 - 27_744 / 157_152), (113, 48_864, 1.0 - 48_864 / 157_152)),
)
def test_operational_accounting_is_counterfactual_induced_cascade(
    fine_count: int, expected_examples: int, expected_savings: float
) -> None:
    accounting = operational_query_accounting(2, fine_count)

    assert accounting["panel_model_examples_including_eligibility"] == expected_examples
    assert (
        accounting["exhaustive_exact_model_examples_including_eligibility"] == 157_152
    )
    assert accounting["model_example_savings_fraction"] == pytest.approx(
        expected_savings
    )
    assert accounting["claim_scope"] == (
        "counterfactual_operational_cascade_vs_exhaustive_fine_replacement"
    )
    assert accounting["includes_full_raw_pair_atlas"] is False
    assert accounting["includes_eap_comparators"] is False


@pytest.mark.parametrize(
    ("fine_count", "expected_examples", "expected_forwards"),
    ((58, 156_576, 1_010), (113, 163_616, 1_120)),
)
def test_actual_validation_accounting_includes_frozen_lanes(
    fine_count: int, expected_examples: int, expected_forwards: int
) -> None:
    accounting = validation_query_accounting(2, fine_count)

    assert (
        accounting["actual_model_examples_including_eligibility"] == expected_examples
    )
    assert (
        accounting["actual_physical_forwards_including_eligibility"]
        == expected_forwards
    )
    assert accounting["discovery_exact_scope"] == "induced_fine_edges"
    assert accounting["confirmation_exact_scope"] == "all_406_internal_fine_edges"
    assert accounting["includes_full_raw_pair_atlas_both_splits"] is True
    assert accounting["includes_eap_and_eap_ig_both_splits"] is True
    assert accounting["includes_batch_invariance_smoke"] is True
    assert accounting["gradient_examples"] == 1_344
    assert accounting["backward_passes"] == 14


def test_fine_count_outside_top_half_structural_bounds_fails_closed() -> None:
    with pytest.raises(ValueError, match="induced_fine_edge_count"):
        operational_query_accounting(2, 57)
    with pytest.raises(ValueError, match="induced_fine_edge_count"):
        validation_query_accounting(21, 114)


def test_dry_run_manifest_discloses_hierarchy_and_claim_boundary() -> None:
    manifest = protocol_dry_run_manifest()

    assert manifest["schema_version"] == "e32_2_node_first_directed_validation_v1"
    assert tuple(manifest["prospective_case_ids"]) == (2, 21)
    assert manifest["hierarchy"] == {
        "primary_case": 2,
        "mandatory_replication_case": 21,
        "development_case": 97,
        "replication_runs_regardless_of_primary_result": True,
        "overall_pass_requires_both_prospective_cases": True,
    }
    assert manifest["claims"]["cases_and_gt_were_inspected"] is True
    assert manifest["claims"]["hidden_or_untouched_benchmark"] is False
    assert manifest["claims"]["cheaper_than_exhaustive_eap"] is False
    assert manifest["claims"]["pair_j_target"] == "causal_reachability_not_direct_edges"
    assert manifest["claims"]["faithfulness_claim_in_e32_2"] is False
