# ruff: noqa: EXE002
"""Pure-scoring tests for E32.4 simultaneous sparse edge sets."""

from __future__ import annotations

import math

import pytest
import torch

from experiments.e32_interpbench.e32_4_protocol import (
    DETERMINISTIC_METHOD_ORDER,
    PRIMARY_METHOD,
    RANDOM_CONTROL_REPLICATES,
    descriptive_support_sizes,
    deterministic_random_supports,
)
from experiments.e32_interpbench.e32_4_scoring import (
    build_selector_score_vectors,
    descriptive_ground_truth_metrics,
    deterministic_top_support,
    endpoint_alignment_metrics,
    equal_split_whole_scores,
    evaluate_case,
    evaluate_panel_verdict,
    exact_fine_ground_truth_ids,
    faithfulness_gate_checks,
    joint_faithfulness_score,
)


def test_selector_builder_has_all_six_methods_and_separate_qkv() -> None:
    nodes = ("a", "b", "c")
    whole_edges = (("a", "b"), ("a", "c"), ("b", "c"))
    # The first whole edge has three Q/K/V children; the other two are MLP edges.
    fine_to_whole = (0, 0, 0, 1, 2)
    induced = (0, 1, 3, 4)
    clean = torch.tensor([[[2.0, -2.0]], [[1.0, -1.0]]])
    corrupt = -clean
    corrupt_into_clean = torch.stack(
        (
            clean - 0.2 * (clean - corrupt),
            clean - 0.4 * (clean - corrupt),
            clean - 0.6 * (clean - corrupt),
            clean - 0.8 * (clean - corrupt),
        )
    )
    clean_into_corrupt = torch.stack(
        (
            corrupt + 0.2 * (clean - corrupt),
            corrupt + 0.4 * (clean - corrupt),
            corrupt + 0.6 * (clean - corrupt),
            corrupt + 0.8 * (clean - corrupt),
        )
    )
    vectors = build_selector_score_vectors(
        component_ids=nodes,
        whole_edges=whole_edges,
        fine_to_whole_indices=fine_to_whole,
        induced_fine_indices=induced,
        clean_output=clean,
        corrupt_output=corrupt,
        corrupt_into_clean_outputs=corrupt_into_clean,
        clean_into_corrupt_outputs=clean_into_corrupt,
        eap_fine_scores=(-1.0, 2.0, -3.0, 4.0, -5.0),
        eap_ig5_fine_scores=(-2.0, 3.0, -4.0, 5.0, -6.0),
        raw_whole_mass=(12.0, 8.0, 2.0),
    )
    assert tuple(vectors) == DETERMINISTIC_METHOD_ORDER
    assert all(vector.shape == (4,) for vector in vectors.values())
    assert torch.equal(vectors["eap_absolute"], torch.tensor([1.0, 2.0, 4.0, 5.0]))
    assert torch.equal(
        vectors["raw_whole_mass_equal_fine_split"],
        torch.tensor([4.0, 4.0, 8.0, 2.0], dtype=torch.float64),
    )
    # Exact directional progress is sign-aware and ordered by the declared effect.
    assert torch.allclose(
        vectors["bidirectional_directional_projection"],
        torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64),
    )


def test_equal_split_and_top_support_are_canonical_and_strict() -> None:
    assert torch.equal(
        equal_split_whole_scores(
            whole_scores=(12.0, 8.0, 2.0),
            fine_to_whole_indices=(0, 0, 0, 1, 2),
        ),
        torch.tensor([4.0, 4.0, 4.0, 8.0, 2.0], dtype=torch.float64),
    )
    assert deterministic_top_support(
        fine_edge_ids=("q", "k", "v", "m"),
        scores=(1.0, 1.0, 1.0, 0.0),
        support_size=2,
    ) == ("q", "k")


def _ideal_outputs(clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return -clean, clean


def test_faithfulness_and_endpoint_alignment_have_separate_gates() -> None:
    clean = torch.tensor([[[3.0, -3.0]], [[2.0, -2.0]]])
    all_internal, complement = _ideal_outputs(clean)
    targets = torch.zeros(2, 1, dtype=torch.long)
    from experiments.e32_interpbench.e32_3_scoring import set_faithfulness_metrics

    faith = set_faithfulness_metrics(
        clean_output=clean,
        all_corrupt_into_clean_output=all_internal,
        selected_corrupt_into_clean_output=all_internal,
        complement_corrupt_into_clean_output=complement,
        clean_target_indices=targets,
    )
    checks = faithfulness_gate_checks(
        faith,
        minimum_projection=0.90,
        maximum_residual_error=0.10,
        minimum_loss_recovery=0.90,
    )
    assert checks["pass"] is True
    assert joint_faithfulness_score(faith) == 1.0

    alignment = endpoint_alignment_metrics(
        clean_output=clean,
        internal_anchor_output=all_internal,
        actual_corrupt_output=all_internal,
        clean_target_indices=targets,
    )
    assert alignment["projection_capture"] == 1.0
    assert alignment["residual_error"] == 0.0
    assert alignment["loss_capture"] == 1.0
    assert alignment["alignment_pass"] is True

    misaligned = endpoint_alignment_metrics(
        clean_output=clean,
        internal_anchor_output=torch.zeros_like(clean),
        actual_corrupt_output=all_internal,
        clean_target_indices=targets,
    )
    assert misaligned["alignment_pass"] is False

    uninformative_faith = set_faithfulness_metrics(
        clean_output=clean,
        all_corrupt_into_clean_output=clean,
        selected_corrupt_into_clean_output=clean,
        complement_corrupt_into_clean_output=clean,
        clean_target_indices=targets,
    )
    assert (
        faithfulness_gate_checks(
            uninformative_faith,
            minimum_projection=0.90,
            maximum_residual_error=0.10,
            minimum_loss_recovery=0.90,
        )["pass"]
        is False
    )
    assert math.isnan(joint_faithfulness_score(uninformative_faith))

    uninformative_alignment = endpoint_alignment_metrics(
        clean_output=clean,
        internal_anchor_output=clean,
        actual_corrupt_output=clean,
        clean_target_indices=targets,
    )
    assert uninformative_alignment["endpoint_informative"] is False
    assert uninformative_alignment["alignment_pass"] is False
    assert math.isnan(uninformative_alignment["projection_capture"])


def _record(
    support: tuple[str, ...], selected: torch.Tensor, complement: torch.Tensor
) -> dict[str, object]:
    return {
        "selected_fine_edge_ids": support,
        "selected_output": selected,
        "complement_output": complement,
    }


def test_case_evaluator_separates_absolute_selector_robustness_and_end_to_end() -> None:
    case_id = 25
    induced_edges = tuple(f"e{index}" for index in range(17))
    k = 7
    base_scores = torch.arange(17.0, 0.0, -1.0)
    selector_scores = {
        method: base_scores + method_index * 1e-6
        for method_index, method in enumerate(DETERMINISTIC_METHOD_ORDER)
    }
    supports = {
        method: deterministic_top_support(
            fine_edge_ids=induced_edges,
            scores=scores,
            support_size=k,
        )
        for method, scores in selector_scores.items()
    }
    clean = torch.tensor([[[3.0, -3.0]], [[2.0, -2.0]]])
    all_internal = -clean
    deterministic_outputs = {
        method: _record(support, all_internal, clean)
        for method, support in supports.items()
    }
    curve_outputs = {
        label: _record(
            deterministic_top_support(
                fine_edge_ids=induced_edges,
                scores=selector_scores[PRIMARY_METHOD],
                support_size=size,
            ),
            all_internal,
            clean,
        )
        for label, size in descriptive_support_sizes(case_id).items()
    }
    halfway = torch.zeros_like(clean)
    random_supports = deterministic_random_supports(
        case_id=case_id,
        fine_edge_ids=induced_edges,
        support_size=k,
    )
    random_outputs = tuple(
        {
            "replicate": replicate,
            **_record(support, halfway, halfway),
        }
        for replicate, support in enumerate(random_supports)
    )

    score = evaluate_case(
        case_id=case_id,
        induced_fine_edge_ids=induced_edges,
        discovery_selector_scores=selector_scores,
        clean_output=clean,
        actual_corrupt_output=all_internal,
        clean_target_indices=torch.zeros(2, 1, dtype=torch.long),
        resample_all_internal_output=all_internal,
        resample_induced_core_output=all_internal,
        resample_outside_core_output=clean,
        resample_deterministic_outputs=deterministic_outputs,
        resample_exact_mass_curve_outputs=curve_outputs,
        resample_random_outputs=random_outputs,
        resample_within_core_primary_complement_output=clean,
        mean_all_internal_output=all_internal,
        mean_induced_core_output=all_internal,
        mean_outside_core_output=clean,
        mean_primary_output=all_internal,
        mean_primary_complement_output=clean,
    )
    assert score["supports"][PRIMARY_METHOD] == supports[PRIMARY_METHOD]
    assert score["gates"]["absolute_sparse_set_pass"] is True
    assert score["gates"]["selector_pass"] is True
    assert score["gates"]["mean_corrupt_robustness_pass"] is True
    assert score["gates"]["resample_end_to_end_pass"] is True
    assert score["gates"]["mean_corrupt_can_rescue_resample"] is False
    assert score["selector_comparison"]["random_95th_sorted_index"] == 243
    assert len(score["random_joint_faithfulness"]) == RANDOM_CONTROL_REPLICATES
    assert tuple(score["exact_mass_support_curve"]) == ("k05", "k15")
    assert all(
        not record["used_for_primary_or_selector_gates"]
        for record in score["exact_mass_support_curve"].values()
    )

    # A bad resample primary remains a primary failure even when mean corruption passes.
    bad_outputs = {
        method: _record(supports[method], halfway, halfway)
        for method in DETERMINISTIC_METHOD_ORDER
    }
    low_random_outputs = tuple(
        {
            "replicate": replicate,
            **_record(support, clean, all_internal),
        }
        for replicate, support in enumerate(random_supports)
    )
    failed = evaluate_case(
        case_id=case_id,
        induced_fine_edge_ids=induced_edges,
        discovery_selector_scores=selector_scores,
        clean_output=clean,
        actual_corrupt_output=all_internal,
        clean_target_indices=torch.zeros(2, 1, dtype=torch.long),
        resample_all_internal_output=all_internal,
        resample_induced_core_output=all_internal,
        resample_outside_core_output=clean,
        resample_deterministic_outputs=bad_outputs,
        # Ideal descriptive curves cannot replace the registered q10 result.
        resample_exact_mass_curve_outputs=curve_outputs,
        resample_random_outputs=low_random_outputs,
        resample_within_core_primary_complement_output=clean,
        mean_all_internal_output=all_internal,
        mean_induced_core_output=all_internal,
        mean_outside_core_output=clean,
        mean_primary_output=all_internal,
        mean_primary_complement_output=clean,
    )
    assert failed["gates"]["absolute_sparse_set_pass"] is False
    assert failed["selector_comparison"]["deterministic_noninferiority_pass"] is True
    assert failed["selector_comparison"]["random_superiority_pass"] is True
    assert failed["gates"]["selector_pass"] is False
    assert failed["gates"]["mean_corrupt_robustness_pass"] is True
    assert failed["gates"]["case_pass"] is False
    assert all(
        record["gate_checks"]["pass"]
        for record in failed["exact_mass_support_curve"].values()
    )


def test_ground_truth_metrics_are_descriptive_and_never_return_gates() -> None:
    fine_ids = ("ab.q", "ab.k", "ac", "bc", "off_core")
    fine_whole_edges = (
        ("a", "b"),
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
        ("c", "d"),
    )
    induced_ids = fine_ids[:-1]
    support = ("ab.q", "bc")
    metrics = descriptive_ground_truth_metrics(
        full_fine_edge_ids=fine_ids,
        full_fine_whole_edges=fine_whole_edges,
        induced_fine_edge_ids=induced_ids,
        selector_scores={PRIMARY_METHOD: (4.0, 1.0, 2.0, 3.0)},
        selected_supports={PRIMARY_METHOD: support},
        fine_ground_truth_ids=("ab.q", "bc", "off_core"),
        whole_ground_truth_edges=(("a", "b"), ("b", "c"), ("c", "d")),
        transitive_non_edges=(("a", "c"),),
    )
    assert "gates" not in metrics
    assert metrics["ground_truth_used_for_selection_or_gates"] is False
    assert metrics["ranking_universe"] == "U_all_internal_fine_edges"
    assert metrics["outside_induced_core_score"] == -1.0
    assert metrics["fine_ground_truth_coverage_by_induced_core"] == 2 / 3
    assert metrics["methods"][PRIMARY_METHOD]["fine"]["top_k_recall"] == 2 / 3
    assert metrics["methods"][PRIMARY_METHOD]["direct_vs_transitive"]["auroc"] == 2 / 3


def test_exact_fine_ground_truth_uses_raw_coordinates_without_qkv_expansion() -> None:
    fixtures = {
        58: {
            "fine": ("q", "k", "v", "mlp-a", "mlp-b"),
            "induced": ("q", "k", "v", "mlp-a", "mlp-b"),
            "truth": ("q", "mlp-a", "mlp-b"),
        },
        129: {
            "fine": ("q", "k", "v", "mlp-a", "mlp-b"),
            "induced": ("q", "k", "v", "mlp-a", "mlp-b"),
            "truth": ("q", "k", "mlp-a", "mlp-b"),
        },
        25: {
            "fine": ("q", "k", "mlp", "off-core"),
            "induced": ("q", "k", "mlp"),
            "truth": ("q", "mlp", "off-core"),
        },
    }
    for case_id, fixture in fixtures.items():
        records = tuple(
            {
                "fine_edge_name": name,
                "source": f"source:{name}",
                "target": f"target:{name}",
            }
            for name in reversed(fixture["fine"])
        )
        raw_truth = tuple(
            (f"source:{name}", f"target:{name}") for name in fixture["truth"]
        )
        observed = exact_fine_ground_truth_ids(
            case_id=case_id,
            full_fine_edge_ids=fixture["fine"],
            induced_fine_edge_ids=fixture["induced"],
            regular_eap_all_fine_records=records,
            official_raw_edges=raw_truth,
        )
        assert observed == fixture["truth"]

    # A true Q coordinate does not label its K/V siblings by whole-head promotion.
    assert fixtures[58]["truth"][0] == "q"
    assert not {"k", "v"}.intersection(fixtures[58]["truth"])

    duplicate_coordinate = (
        {"fine_edge_name": "q", "source": "s", "target": "t"},
        {"fine_edge_name": "k", "source": "s", "target": "t"},
        {"fine_edge_name": "v", "source": "v", "target": "v-target"},
        {"fine_edge_name": "mlp-a", "source": "a", "target": "a-target"},
        {"fine_edge_name": "mlp-b", "source": "b", "target": "b-target"},
    )
    with pytest.raises(ValueError, match="coordinate"):
        exact_fine_ground_truth_ids(
            case_id=58,
            full_fine_edge_ids=fixtures[58]["fine"],
            induced_fine_edge_ids=fixtures[58]["induced"],
            regular_eap_all_fine_records=duplicate_coordinate,
            official_raw_edges=(("s", "t"), ("a", "a-target"), ("b", "b-target")),
        )


def test_panel_verdict_is_conjunctive_by_tier() -> None:
    passing = {
        "gates": {
            "absolute_sparse_set_pass": True,
            "selector_pass": True,
            "mean_corrupt_robustness_pass": True,
            "resample_end_to_end_pass": True,
            "case_pass": True,
        }
    }
    panel = {str(case_id): {"case_id": case_id, **passing} for case_id in (58, 129, 25)}
    verdict = evaluate_panel_verdict(panel)
    assert verdict["absolute_sparse_set_generalization_pass"] is True
    assert verdict["selector_generalization_pass"] is True
    assert verdict["mean_corrupt_robustness_generalization_pass"] is True
    assert verdict["majority_rule_allowed"] is False

    panel["58"] = {
        "case_id": 58,
        "gates": {
            **passing["gates"],
            "mean_corrupt_robustness_pass": False,
        },
    }
    verdict = evaluate_panel_verdict(panel)
    assert verdict["absolute_sparse_set_generalization_pass"] is True
    assert verdict["mean_corrupt_robustness_generalization_pass"] is False
