# ruff: noqa: EXE002
"""Pure scorer tests for E32.3 sparse-core faithfulness."""

from __future__ import annotations

import torch

from experiments.e32_interpbench.adapter import patchable_internal_universe
from experiments.e32_interpbench.component_patching import build_patch_units
from experiments.e32_interpbench.e32_3_protocol import (
    DETERMINISTIC_METHOD_ORDER,
    RANDOM_CONTROL_REPLICATES,
    deterministic_random_support,
)
from experiments.e32_interpbench.e32_3_scoring import (
    activation_shift_rms_scores,
    build_selector_score_vectors,
    deterministic_top_support,
    evaluate_panel_verdict,
    evaluate_singleton_support_curve,
    evaluate_sparse_core_case,
    exact_fine_interaction_mass,
    incident_mean_node_scores,
    interaction_mass_retention,
    promote_absolute_fine_max,
    set_faithfulness_metrics,
    singleton_output_effect_scores,
    spearman_correlation,
)


def test_primary_and_activation_shift_scores_preserve_node_axis() -> None:
    clean = torch.zeros(2, 1, 2)
    singletons = torch.stack((clean + 1.0, clean + 2.0, clean + 3.0))
    assert torch.equal(
        singleton_output_effect_scores(clean, singletons),
        torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64),
    )

    clean_activation = torch.zeros(3, 2, 1, 4)
    corrupt_activation = clean_activation.clone()
    corrupt_activation[0] += 1.0
    corrupt_activation[1] += 2.0
    corrupt_activation[2] += 3.0
    assert torch.equal(
        activation_shift_rms_scores(clean_activation, corrupt_activation),
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
    )


def test_exact_mass_is_sign_symmetric_bidirectional_energy() -> None:
    clean = torch.zeros(2, 1, 2)
    corrupt = torch.zeros_like(clean)
    corrupt_into_clean = torch.stack((clean + 2.0, clean - 4.0))
    clean_into_corrupt = torch.stack((corrupt - 2.0, corrupt + 2.0))
    mass = exact_fine_interaction_mass(
        clean_output=clean,
        corrupt_output=corrupt,
        corrupt_into_clean_outputs=corrupt_into_clean,
        clean_into_corrupt_outputs=clean_into_corrupt,
    )
    assert torch.equal(mass, torch.tensor([4.0, 10.0], dtype=torch.float64))


def test_mass_retention_uses_both_promoted_endpoints() -> None:
    edges = (("a", "b"), ("a", "c"), ("b", "c"))
    retention = interaction_mass_retention(
        item_mass=torch.tensor([8.0, 1.0, 1.0]),
        selected_nodes=("a", "b"),
        whole_edges=edges,
    )
    assert retention == 0.8

    fine_retention = interaction_mass_retention(
        item_mass=torch.tensor([3.0, 5.0, 1.0, 1.0]),
        selected_nodes=("a", "b"),
        whole_edges=edges,
        fine_to_whole_indices=(0, 0, 1, 2),
    )
    assert fine_retention == 0.8


def test_set_faithfulness_exact_endpoints_are_ideal() -> None:
    clean = torch.tensor([[[2.0, -2.0]], [[3.0, -3.0]]])
    all_corrupt = -clean
    metrics = set_faithfulness_metrics(
        clean_output=clean,
        all_corrupt_into_clean_output=all_corrupt,
        selected_corrupt_into_clean_output=all_corrupt,
        complement_corrupt_into_clean_output=clean,
        clean_target_indices=torch.zeros(2, 1, dtype=torch.long),
    )
    assert metrics["necessity_projection"] == 1.0
    assert metrics["sufficiency_projection"] == 1.0
    assert metrics["necessity_residual_error"] == 0.0
    assert metrics["sufficiency_residual_error"] == 0.0
    assert metrics["necessity_loss_recovery"] == 1.0
    assert metrics["sufficiency_loss_recovery"] == 1.0
    assert metrics["endpoint_informative"] is True
    assert metrics["per_example"].keys() == {
        "vector_gap_energy",
        "necessity_projection_numerator",
        "sufficiency_projection_numerator",
        "necessity_residual_energy",
        "sufficiency_residual_energy",
        "clean_target_loss",
        "all_corrupt_target_loss",
        "selected_target_loss",
        "complement_target_loss",
    }


def test_incident_eap_and_tie_stable_support_helpers() -> None:
    nodes = ("a", "b", "c")
    edges = (("a", "b"), ("a", "c"), ("b", "c"))
    incident = incident_mean_node_scores(
        node_universe=nodes,
        whole_edges=edges,
        whole_edge_scores=torch.tensor([2.0, 4.0, 8.0]),
    )
    assert torch.equal(incident, torch.tensor([3.0, 5.0, 6.0], dtype=torch.float64))
    promoted = promote_absolute_fine_max(
        fine_scores=torch.tensor([-2.0, 1.0, -4.0, 8.0]),
        fine_to_whole_indices=(0, 0, 1, 2),
        whole_edge_count=3,
    )
    assert torch.equal(promoted, torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64))
    assert deterministic_top_support(
        node_universe=nodes, node_scores=(1.0, 1.0, 0.0), support_size=1
    ) == ("a",)
    assert spearman_correlation((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)) == 1.0


def _fine_mapping_for_transformer_edges(
    nodes: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> tuple[int, ...]:
    del nodes
    mapping = []
    for index, (_source, target) in enumerate(edges):
        mapping.extend([index] * (3 if ".attn.hook_result[" in target else 1))
    return tuple(mapping)


def test_public_case_scorer_returns_full_vectors_and_conjunctive_gates() -> None:
    units = build_patch_units(2, 4)
    nodes = tuple(unit.component_id for unit in units)
    universe_nodes, edges = patchable_internal_universe(2, 4)
    assert nodes == universe_nodes and len(edges) == 33
    fine_mapping = _fine_mapping_for_transformer_edges(nodes, edges)
    assert len(fine_mapping) == 73
    support_size = 5

    # Every deterministic method freezes the same known top-half support in D.
    base_scores = torch.tensor(
        [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        dtype=torch.float64,
    )
    discovery_scores = {
        method: base_scores + method_index * 1e-3
        for method_index, method in enumerate(DETERMINISTIC_METHOD_ORDER)
    }
    confirmation_scores = {
        method: values * 2.0 for method, values in discovery_scores.items()
    }
    selected = deterministic_top_support(
        node_universe=nodes,
        node_scores=base_scores,
        support_size=support_size,
    )

    whole_inside = torch.tensor(
        [source in selected and target in selected for source, target in edges]
    )
    raw_mass = torch.where(whole_inside, 20.0, 0.1).to(torch.float64)
    fine_inside = torch.tensor(
        [whole_inside[index] for index in fine_mapping], dtype=torch.bool
    )
    exact_mass = torch.where(fine_inside, 20.0, 0.1).to(torch.float64)

    clean = torch.tensor([[[3.0, -3.0]], [[2.0, -2.0]]])
    all_corrupt = -clean
    ideal_record = {
        "selected_nodes": selected,
        "selected_corrupt_into_clean_output": all_corrupt,
        "complement_corrupt_into_clean_output": clean,
    }
    deterministic_outputs = {
        method: dict(ideal_record) for method in DETERMINISTIC_METHOD_ORDER
    }
    random_outputs = []
    halfway = torch.zeros_like(clean)
    for replicate in range(RANDOM_CONTROL_REPLICATES):
        random_outputs.append(
            {
                "replicate": replicate,
                "selected_nodes": deterministic_random_support(
                    case_id=25,
                    component_ids=nodes,
                    support_size=support_size,
                    replicate=replicate,
                ),
                "selected_corrupt_into_clean_output": halfway,
                "complement_corrupt_into_clean_output": halfway,
            }
        )

    score = evaluate_sparse_core_case(
        case_id=25,
        component_ids=nodes,
        whole_edges=edges,
        fine_to_whole_indices=fine_mapping,
        discovery_selector_scores=discovery_scores,
        confirmation_selector_scores=confirmation_scores,
        discovery_raw_mass=raw_mass,
        confirmation_raw_mass=raw_mass * 2.0,
        discovery_exact_mass=exact_mass,
        confirmation_exact_mass=exact_mass * 2.0,
        confirmation_clean_output=clean,
        confirmation_all_corrupt_into_clean_output=all_corrupt,
        confirmation_clean_target_indices=torch.zeros(2, 1, dtype=torch.long),
        deterministic_set_outputs=deterministic_outputs,
        random_set_outputs=random_outputs,
    )
    assert score["schema_version"] == "e32_3_sparse_core_case_score_v1"
    assert score["selected_nodes"] == selected
    assert score["full_vectors"]["confirmation_exact_mass"] == tuple(
        float(value) for value in exact_mass * 2.0
    )
    assert score["gates"]["screen_pass"] is True
    assert score["gates"]["faithfulness_pass"] is True
    assert score["gates"]["selector_pass"] is True
    assert score["gates"]["case_pass"] is True
    assert score["ground_truth_used_for_cohort_selection"] is True
    assert (
        score["ground_truth_used_for_within_case_selection_or_primary_scoring"] is False
    )


def test_selector_vector_builder_freezes_all_five_methods() -> None:
    nodes = ("a", "b", "c")
    edges = (("a", "b"), ("a", "c"), ("b", "c"))
    clean = torch.zeros(2, 1, 2)
    singleton = torch.stack((clean + 1, clean + 2, clean + 3))
    activation_clean = torch.zeros(3, 2, 1, 2)
    activation_corrupt = torch.stack(
        (activation_clean[0] + 1, activation_clean[1] + 2, activation_clean[2] + 3)
    )
    pair_mass = torch.tensor([2.0, 4.0, 8.0])
    vectors = build_selector_score_vectors(
        node_universe=nodes,
        whole_edges=edges,
        clean_output=clean,
        singleton_outputs=singleton,
        clean_activations=activation_clean,
        corrupt_activations=activation_corrupt,
        raw_whole_mass=pair_mass,
        eap_ig5_fine_scores=torch.tensor([-2.0, 1.0, -4.0, 8.0]),
        fine_to_whole_indices=(0, 0, 1, 2),
    )
    assert tuple(vectors) == DETERMINISTIC_METHOD_ORDER
    assert all(vector.shape == (3,) for vector in vectors.values())


def test_zero_total_mass_is_a_noninformative_screen_failure() -> None:
    nodes, edges = patchable_internal_universe(2, 4)
    fine_mapping = _fine_mapping_for_transformer_edges(nodes, edges)
    scores = torch.arange(len(nodes), 0, -1, dtype=torch.float64)
    selector_scores = {method: scores.clone() for method in DETERMINISTIC_METHOD_ORDER}
    selected = deterministic_top_support(
        node_universe=nodes, node_scores=scores, support_size=5
    )
    clean = torch.tensor([[[2.0, -2.0]], [[3.0, -3.0]]])
    all_corrupt = -clean
    ideal = {
        "selected_nodes": selected,
        "selected_corrupt_into_clean_output": all_corrupt,
        "complement_corrupt_into_clean_output": clean,
    }
    random_outputs = tuple(
        {
            "replicate": replicate,
            "selected_nodes": deterministic_random_support(
                case_id=25,
                component_ids=nodes,
                support_size=5,
                replicate=replicate,
            ),
            "selected_corrupt_into_clean_output": torch.zeros_like(clean),
            "complement_corrupt_into_clean_output": torch.zeros_like(clean),
        }
        for replicate in range(RANDOM_CONTROL_REPLICATES)
    )

    score = evaluate_sparse_core_case(
        case_id=25,
        component_ids=nodes,
        whole_edges=edges,
        fine_to_whole_indices=fine_mapping,
        discovery_selector_scores=selector_scores,
        confirmation_selector_scores=selector_scores,
        discovery_raw_mass=torch.zeros(len(edges)),
        confirmation_raw_mass=torch.zeros(len(edges)),
        discovery_exact_mass=torch.zeros(len(fine_mapping)),
        confirmation_exact_mass=torch.zeros(len(fine_mapping)),
        confirmation_clean_output=clean,
        confirmation_all_corrupt_into_clean_output=all_corrupt,
        confirmation_clean_target_indices=torch.zeros(2, 1, dtype=torch.long),
        deterministic_set_outputs={
            method: dict(ideal) for method in DETERMINISTIC_METHOD_ORDER
        },
        random_set_outputs=random_outputs,
    )

    assert score["mass_informative"] is False
    assert set(score["mass_informative_by_vector"].values()) == {False}
    assert score["retention"]["discovery_raw"] is None
    assert score["retention"]["confirmation_exact"] is None
    assert score["stability"]["raw_mass_spearman"] is None
    assert score["stability"]["exact_mass_spearman"] is None
    assert score["gates"]["screen_checks"]["mass_informative"] is False
    assert score["gates"]["screen_pass"] is False


def test_singleton_curve_has_only_frozen_quartile_prefixes() -> None:
    nodes = tuple(f"n{index}" for index in range(10))
    scores = torch.arange(10.0, 0.0, -1.0)
    clean = torch.tensor([[[2.0, -2.0]], [[3.0, -3.0]]])
    all_corrupt = -clean
    records = {}
    for size in (3, 5, 8):
        records[size] = {
            "selected_nodes": deterministic_top_support(
                node_universe=nodes,
                node_scores=scores,
                support_size=size,
            ),
            "selected_corrupt_into_clean_output": all_corrupt,
            "complement_corrupt_into_clean_output": clean,
        }
    curve = evaluate_singleton_support_curve(
        component_ids=nodes,
        discovery_singleton_scores=scores,
        clean_output=clean,
        all_corrupt_into_clean_output=all_corrupt,
        clean_target_indices=torch.zeros(2, 1, dtype=torch.long),
        set_outputs_by_support_size=records,
    )
    assert curve["support_sizes"] == (3, 5, 8)
    assert curve["support_fractions"] == (0.3, 0.5, 0.8)
    assert curve["joint_loss_recovery"] == (1.0, 1.0, 1.0)


def test_panel_verdict_requires_every_mandatory_case() -> None:
    passing = {
        "gates": {
            "screen_pass": True,
            "faithfulness_pass": True,
            "selector_pass": True,
            "case_pass": True,
        }
    }
    panel = {str(case_id): {"case_id": case_id, **passing} for case_id in (58, 129, 25)}
    verdict = evaluate_panel_verdict(panel)
    assert verdict["sparse_core_generalization_pass"] is True
    assert verdict["singleton_selector_generalization_pass"] is True

    panel["129"] = {
        "case_id": 129,
        "gates": {
            "screen_pass": True,
            "faithfulness_pass": False,
            "selector_pass": True,
            "case_pass": False,
        },
    }
    verdict = evaluate_panel_verdict(panel)
    assert verdict["sparse_core_generalization_pass"] is False
    assert verdict["singleton_selector_generalization_pass"] is False
    assert verdict["majority_rule_allowed"] is False
    assert verdict["ground_truth_used_for_cohort_selection"] is True
    assert (
        verdict["ground_truth_used_for_within_case_selection_or_primary_scoring"]
        is False
    )
