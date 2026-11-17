# ruff: noqa: EXE002
"""Frozen, model-free protocol tests for E32.4 edge-set faithfulness."""

from __future__ import annotations

import math

import pytest

from experiments.e32_interpbench.e32_4_protocol import (
    ABSOLUTE_PRIMARY_GATES,
    BRIDGE_GATES,
    CASE_GEOMETRY,
    CASE_ORDER,
    CORE_ANCESTRY,
    CORE_CASE_ANCESTRY,
    CORE_NODE_SUPPORTS,
    DETERMINISTIC_METHOD_ORDER,
    END_TO_END_ALIGNMENT_GATES,
    EXPECTED_DISCOVERY_QUERY_ACCOUNTING,
    GROUND_TRUTH_FINE_LABEL_COUNTS,
    GROUND_TRUTH_OFF_CORE_SCORE_FLOOR,
    PRIMARY_METHOD,
    RANDOM_CONTROL_95TH_INDEX,
    RANDOM_CONTROL_REPLICATES,
    ROBUSTNESS_GATES,
    SELECTOR_GATES,
    descriptive_support_sizes,
    deterministic_random_supports,
    primary_support_size,
    protocol_manifest,
    split_request,
)


def test_exact_discovery_family_accounting_is_registered() -> None:
    expected_non_gradient = {
        "gradient_forwards": 0,
        "gradient_examples": 0,
        "backward_passes_inferred_from_pinned_control_flow": 0,
    }
    for case_id, (whole, singleton) in {
        58: ((17, 7_808), (94, 6_016)),
        129: ((17, 7_808), (100, 6_400)),
        25: ((15, 4_032), (38, 2_432)),
    }.items():
        stages = EXPECTED_DISCOVERY_QUERY_ACCOUNTING[case_id]
        assert tuple(stages) == (
            "behavioral_eligibility",
            "whole_component_atlas",
            "core_singleton_replacement",
            "regular_eap",
            "eap_ig5",
        )
        assert stages["behavioral_eligibility"] == {
            "physical_forwards": 2,
            "model_examples": 192,
            **expected_non_gradient,
        }
        assert stages["whole_component_atlas"] == {
            "physical_forwards": whole[0],
            "model_examples": whole[1],
            **expected_non_gradient,
        }
        assert stages["core_singleton_replacement"] == {
            "physical_forwards": singleton[0],
            "model_examples": singleton[1],
            **expected_non_gradient,
        }
        assert stages["regular_eap"] == {
            "physical_forwards": 2,
            "model_examples": 128,
            "gradient_forwards": 1,
            "gradient_examples": 64,
            "backward_passes_inferred_from_pinned_control_flow": 1,
        }
        assert stages["eap_ig5"] == {
            "physical_forwards": 7,
            "model_examples": 448,
            "gradient_forwards": 6,
            "gradient_examples": 384,
            "backward_passes_inferred_from_pinned_control_flow": 6,
        }

        totals = {
            key: sum(stage[key] for stage in stages.values())
            for key in next(iter(stages.values()))
        }
        manifest = protocol_manifest()
        assert manifest["expected_discovery_query_accounting"][str(case_id)] == stages
        assert manifest["expected_query_ceilings"][str(case_id)]["discovery"] == totals


def test_coordinate_exact_fine_ground_truth_rule_is_literal() -> None:
    assert GROUND_TRUTH_FINE_LABEL_COUNTS == {
        58: {"full_internal_U": 3, "induced_core_I": 3},
        129: {"full_internal_U": 4, "induced_core_I": 4},
        25: {"full_internal_U": 3, "induced_core_I": 2},
    }
    diagnostic = protocol_manifest()["ground_truth_diagnostic"]
    assert diagnostic["fine_label_rule"] == (
        "positive_iff_regular_eap_raw_source_target_is_in_official_raw_edges_"
        "restricted_to_full_internal_U"
    )
    assert diagnostic["fine_label_whole_child_expansion"] is False
    assert diagnostic["whole_promotion_used_only_for_separate_whole_diagnostic"]
    assert diagnostic["expected_fine_label_counts"] == {
        str(case_id): counts
        for case_id, counts in GROUND_TRUTH_FINE_LABEL_COUNTS.items()
    }
    assert diagnostic["case25_induced_core_fine_truth_coverage"] == {
        "numerator": 2,
        "denominator": 3,
    }


def test_frozen_core_ancestry_geometry_and_methods() -> None:
    assert CASE_ORDER == (58, 129, 25)
    assert CORE_ANCESTRY["registration_sha256"] == (
        "1c42b4761573286af06604094efc5a85b0f873d773a99afe97d9767e25ab86a4"
    )
    assert CORE_ANCESTRY["summary_sha256"] == (
        "2a04b0659fb4b3615d642bfb36e0ad787cb2a98540ab5dacc43f910a199f1647"
    )
    assert CORE_ANCESTRY["source"] == "sealed_e32_3_v2_summary_selected_nodes"
    assert CORE_CASE_ANCESTRY[58] == {
        "authorization_sha256": (
            "ebd0b6e01998951968fd284463dbbbd2d8f5b2ebf3619801feb544166451fd21"
        ),
        "selected_node_support_sha256": (
            "8864b22af413302ff38bff2198408f2c670213eb9817062b3c70a75093e47123"
        ),
        "selected_node_support_sha256_encoding": "compact_utf8_json_list",
    }
    assert tuple(
        map(
            len,
            (CORE_NODE_SUPPORTS[58], CORE_NODE_SUPPORTS[129], CORE_NODE_SUPPORTS[25]),
        )
    ) == (8, 8, 5)
    assert CASE_GEOMETRY == {
        58: {
            "nodes": 15,
            "whole_edges": 87,
            "fine_edges": 207,
            "core_nodes": 8,
            "induced_whole_edges": 25,
            "induced_fine_edges": 45,
        },
        129: {
            "nodes": 15,
            "whole_edges": 87,
            "fine_edges": 207,
            "core_nodes": 8,
            "induced_whole_edges": 22,
            "induced_fine_edges": 48,
        },
        25: {
            "nodes": 10,
            "whole_edges": 33,
            "fine_edges": 73,
            "core_nodes": 5,
            "induced_whole_edges": 9,
            "induced_fine_edges": 17,
        },
    }
    assert PRIMARY_METHOD == "bidirectional_exact_mass"
    assert DETERMINISTIC_METHOD_ORDER == (
        "bidirectional_exact_mass",
        "bidirectional_directional_projection",
        "eap_absolute",
        "eap_ig5_absolute",
        "raw_whole_mass_equal_fine_split",
        "masked_rank_one_equal_fine_split",
    )


def test_fresh_splits_and_response_blind_capacity_are_exact() -> None:
    expected = {
        58: {
            "discovery": (
                64,
                96,
                758300,
                758301,
                "bc8879564658999d87ffcd7e02ba257fd30f5c705575c179ed40fa806ea909e2",
                "fa2b4270c357382c3e315b511229eb2ee8aa14d2c535f959ffe2ee58f4ef17e2",
                96,
            ),
            "confirmation": (
                128,
                192,
                758400,
                758401,
                "b40acc67a23373d5a733bbbc05fb059d447ef6e238f157df3ade7b68d1c6eff2",
                "d7248d18b2daf46a336cfcc9f5ce9fdfaac632bcb2b773faffa17f0e76c4d6c3",
                192,
            ),
        },
        129: {
            "discovery": (
                64,
                96,
                829300,
                829301,
                "cca526b44bd06f9191135b1e764323aecd7c647dcac3d9978605898fdcf38a35",
                "12776f7293cad6efe98979877baf2211e68bc75a0844d08672093216b28e9bc2",
                96,
            ),
            "confirmation": (
                128,
                192,
                829400,
                829401,
                "42627b1c5d57412b0bdf6bba9b8051213d36929f4791a25ef0967f89228f22a2",
                "c71665081e17bad3f687bec8d5849d96f77b723b75898ff9f298be7aa8516e43",
                192,
            ),
        },
        25: {
            "discovery": (
                64,
                96,
                725300,
                725301,
                "1e55a22b22df3a7b40a870498493596b9034ebcc1e192f9b10d21b870eab3fad",
                "63af3ad08c7639822dedb4dbe88672e5bdb860389fd55a73569359c297b51382",
                86,
            ),
            "confirmation": (
                128,
                192,
                725400,
                725401,
                "c4c8a0cbb9ff48bd9973e06a626644c6708180a9e2b6debe4d9d2e18eaa47719",
                "00b41e9e01aab9e870b3b79117af6273652c5d7cbebdbb89649bfce386f80848",
                172,
            ),
        },
    }
    for case_id, splits in expected.items():
        for split, values in splits.items():
            request = split_request(case_id, split)
            assert (
                request.required_count,
                request.candidate_count,
                request.clean_seed,
                request.corrupt_seed,
                request.raw_identity_sha256,
                request.target_identity_sha256,
                request.conservative_capacity,
            ) == values
            assert request.conservative_capacity >= request.required_count
            assert request.exclude_all_e32_3_raw_candidate_role_rows is True
            assert request.raw_duplicates_permitted_but_selected_rows_unique is True


def test_primary_and_descriptive_edge_budgets_are_frozen() -> None:
    assert tuple(primary_support_size(case_id) for case_id in CASE_ORDER) == (20, 20, 7)
    assert descriptive_support_sizes(58) == {"k05": 10, "k15": 31}
    assert descriptive_support_sizes(129) == {"k05": 10, "k15": 31}
    assert descriptive_support_sizes(25) == {"k05": 3, "k15": 10}
    for case_id in CASE_ORDER:
        size = primary_support_size(case_id)
        assert size == math.floor(0.10 * CASE_GEOMETRY[case_id]["fine_edges"])
        assert size / CASE_GEOMETRY[case_id]["fine_edges"] < 0.10
        assert size <= CASE_GEOMETRY[case_id]["induced_fine_edges"]


def test_256_random_supports_are_uniform_family_unique_and_reproducible() -> None:
    fine_edges = tuple(f"edge-{index}" for index in range(17))
    first = deterministic_random_supports(
        case_id=25,
        fine_edge_ids=fine_edges,
        support_size=7,
    )
    second = deterministic_random_supports(
        case_id=25,
        fine_edge_ids=fine_edges,
        support_size=7,
    )
    assert RANDOM_CONTROL_REPLICATES == 256
    assert RANDOM_CONTROL_95TH_INDEX == 243
    assert first == second
    assert len(first) == RANDOM_CONTROL_REPLICATES
    assert len(set(first)) == RANDOM_CONTROL_REPLICATES
    assert all(
        len(support) == 7
        and len(set(support)) == 7
        and tuple(edge for edge in fine_edges if edge in support) == support
        for support in first
    )

    with pytest.raises(ValueError, match="distinct"):
        deterministic_random_supports(
            case_id=25,
            fine_edge_ids=("a", "b", "c"),
            support_size=1,
        )


def test_all_gate_tiers_and_claim_boundary_are_explicit() -> None:
    assert BRIDGE_GATES == {
        "minimum_necessity_projection": 0.75,
        "minimum_sufficiency_projection": 0.75,
        "maximum_necessity_residual_error": 0.25,
        "maximum_sufficiency_residual_error": 0.25,
        "minimum_necessity_loss_recovery": 0.75,
        "minimum_sufficiency_loss_recovery": 0.75,
    }
    assert ABSOLUTE_PRIMARY_GATES["minimum_necessity_projection"] == 0.90
    assert ABSOLUTE_PRIMARY_GATES["maximum_sufficiency_residual_error"] == 0.10
    assert ROBUSTNESS_GATES == BRIDGE_GATES
    assert END_TO_END_ALIGNMENT_GATES == {
        "minimum_projection_capture": 0.75,
        "maximum_residual_error": 0.25,
        "minimum_loss_capture": 0.75,
    }
    assert SELECTOR_GATES == {
        "minimum_margin_over_random_95th_percentile": 0.05,
        "maximum_deficit_to_best_deterministic_baseline": 0.05,
    }
    assert GROUND_TRUTH_OFF_CORE_SCORE_FLOOR == -1.0

    manifest = protocol_manifest()
    assert manifest["mandatory_all_cases"] is True
    assert manifest["ground_truth_used_for_selection_or_gates"] is False
    assert manifest["ground_truth_diagnostic"]["ranking_universe"] == (
        "U_all_internal_fine_edges"
    )
    assert manifest["ground_truth_diagnostic"]["outside_I_score"] == -1.0
    assert manifest["resample_primary_can_be_rescued_by_robustness"] is False
    assert (
        manifest["development_exposure"][
            "primary_support_sizes_20_20_7_and_point90_gates_chosen_after_e32_3_individual_edge_mass_diagnostics"
        ]
        is True
    )
    assert manifest["random_control"]["replicates"] == 256
    assert manifest["random_control"]["empirical_95th_sorted_index"] == 243
    assert manifest["interventions"]["primary_complement"] == "U_minus_S"
    assert manifest["interventions"]["bridge_complement"] == "U_minus_I"
    assert manifest["verdict_hierarchy"][
        "all_three_cases_required_with_no_majority_rule"
    ]
    assert manifest["verdict_hierarchy"]["selector_requires_absolute_sparse_set"]
    assert "not_included" in manifest["baseline_feasibility"]["acdc"]
    assert "not_included" in manifest["baseline_feasibility"]["edge_pruning"]
    assert "boundary_conditioned" in manifest["claim_boundary"]
    for forbidden in ("direct_wiring", "minimal", "sota", "model_generalization"):
        assert forbidden in manifest["claim_boundary"]
