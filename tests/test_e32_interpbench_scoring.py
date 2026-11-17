# ruff: noqa: EXE002
"""Tests for exact E32 localization and held-out faithfulness scoring."""

from __future__ import annotations

import pytest
import torch

from experiments.e32_interpbench.protocol import (
    CLAIM_BOUNDARIES,
    AdaptedScores,
    ScoreSnapshot,
)
from experiments.e32_interpbench.scoring import (
    binary_ranking_metrics,
    evaluate_localization,
    evaluate_method_panel,
    evaluate_query_prefixes,
    normalized_faithfulness_curves,
    paired_bootstrap_difference,
    path_recovery_metrics,
)

NODES = ("a", "b", "c", "d")
EDGES = (("a", "b"), ("b", "c"), ("a", "d"), ("d", "c"))
GROUND_TRUTH = (("a", "b"), ("b", "c"))


def test_exact_binary_metrics_handle_ties_without_order_dependent_ap() -> None:
    metrics = binary_ranking_metrics(
        scores=(0.9, 0.8, 0.8, 0.1),
        labels=(True, False, True, False),
    )
    assert metrics["auroc"] == pytest.approx(0.875)
    assert metrics["auprc"] == pytest.approx(5.0 / 6.0)
    assert metrics["top_k"] == 2
    assert metrics["top_k_precision"] == 0.5
    assert metrics["top_k_recall"] == 0.5
    assert metrics["top_k_f1"] == 0.5
    # The universe order is the frozen tie-break for finite top-k selection only.
    assert metrics["selected_indices"] == (0, 1)


def test_node_edge_localization_and_selected_support_precision_are_exact() -> None:
    score = evaluate_localization(
        node_universe=NODES,
        edge_universe=EDGES,
        ground_truth_edges=GROUND_TRUTH,
        node_scores=(0.9, 0.8, 0.7, 0.1),
        edge_scores=(0.9, 0.8, 0.2, 0.1),
    )
    assert score["ground_truth_node_count"] == 3
    assert score["ground_truth_edge_count"] == 2
    assert score["node"]["auroc"] == 1.0
    assert score["edge"]["auprc"] == 1.0
    assert score["node"]["support_precision"] == 1.0
    assert score["edge"]["support_precision"] == 1.0


def test_node_ground_truth_can_include_components_lost_when_edges_collapse() -> None:
    score = evaluate_localization(
        node_universe=("a", "b", "c", "d", "e"),
        edge_universe=EDGES,
        ground_truth_edges=GROUND_TRUTH,
        ground_truth_nodes=("a", "b", "c", "d"),
        node_scores=(0.9, 0.8, 0.7, 0.6, 0.1),
        edge_scores=(0.9, 0.8, 0.2, 0.1),
    )
    assert score["ground_truth_node_count"] == 4
    assert score["node"]["auroc"] == 1.0
    assert score["node"]["selected_support"] == ("a", "b", "c", "d")


def test_method_panel_compares_identical_frozen_universes_and_complete_paths() -> None:
    panel = evaluate_method_panel(
        method_scores={
            "truth_ranked": AdaptedScores(
                node_scores=(0.9, 0.8, 0.7, 0.1),
                edge_scores=(0.9, 0.8, 0.2, 0.1),
            ),
            "shortcut_ranked": AdaptedScores(
                node_scores=(0.9, 0.1, 0.8, 0.7),
                edge_scores=(0.2, 0.1, 0.9, 0.8),
            ),
        },
        node_universe=NODES,
        edge_universe=EDGES,
        ground_truth_edges=GROUND_TRUTH,
    )
    assert tuple(panel) == ("truth_ranked", "shortcut_ranked")
    assert panel["truth_ranked"]["edge"]["top_k_f1"] == 1.0
    assert panel["truth_ranked"]["path"]["path_recall"] == 1.0
    assert panel["shortcut_ranked"]["path"]["path_precision"] == 0.0


def test_path_metric_counts_complete_source_to_sink_paths_not_edge_membership() -> None:
    score = path_recovery_metrics(
        selected_edges=EDGES,
        ground_truth_edges=GROUND_TRUTH,
    )
    assert score["ground_truth_paths"] == (("a", "b", "c"),)
    assert score["selected_boundary_paths"] == (
        ("a", "b", "c"),
        ("a", "d", "c"),
    )
    assert score["path_precision"] == 0.5
    assert score["path_recall"] == 1.0
    assert score["path_f1"] == pytest.approx(2.0 / 3.0)
    assert score["metric_status"] == "e32_defined_not_official_interpbench"


def test_held_out_necessity_and_sufficiency_are_normalized_per_example() -> None:
    score = normalized_faithfulness_curves(
        support_sizes=(0, 1, 2),
        support_total=2,
        clean=(1.0, 1.0),
        empty=(0.0, 0.0),
        keep_selected=((0.0, 0.0), (0.5, 0.7), (1.0, 1.0)),
        remove_selected=((1.0, 1.0), (0.4, 0.6), (0.0, 0.0)),
    )
    assert score["support_fraction"] == (0.0, 0.5, 1.0)
    assert score["sufficiency_mean"] == pytest.approx((0.0, 0.6, 1.0))
    assert score["necessity_mean"] == pytest.approx((0.0, 0.5, 1.0))
    assert score["sufficiency_auc"] == pytest.approx(0.55)
    assert score["necessity_auc"] == pytest.approx(0.5)
    assert score["per_example_sufficiency"].shape == (2, 3)


def test_query_prefix_analysis_uses_nested_snapshots_and_frozen_universes() -> None:
    prefixes = evaluate_query_prefixes(
        snapshots=(
            ScoreSnapshot(
                queries=4,
                node_scores=(0.4, 0.3, 0.2, 0.1),
                edge_scores=(0.4, 0.3, 0.2, 0.1),
            ),
            ScoreSnapshot(
                queries=8,
                node_scores=(0.9, 0.8, 0.7, 0.1),
                edge_scores=(0.9, 0.8, 0.2, 0.1),
            ),
        ),
        node_universe=NODES,
        edge_universe=EDGES,
        ground_truth_edges=GROUND_TRUTH,
    )
    assert tuple(row["queries"] for row in prefixes) == (4, 8)
    assert prefixes[-1]["edge"]["auprc"] == 1.0
    assert prefixes[-1]["edge"]["top_k_f1"] == 1.0

    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_query_prefixes(
            snapshots=(
                ScoreSnapshot(
                    queries=4, node_scores=(0.0,) * 4, edge_scores=(0.0,) * 4
                ),
                ScoreSnapshot(
                    queries=4, node_scores=(0.0,) * 4, edge_scores=(0.0,) * 4
                ),
            ),
            node_universe=NODES,
            edge_universe=EDGES,
            ground_truth_edges=GROUND_TRUTH,
        )


def test_paired_bootstrap_is_deterministic_and_preserves_pairing() -> None:
    left = torch.tensor([[2.0, 4.0], [3.0, 5.0], [4.0, 6.0]], dtype=torch.float64)
    right = left - 1.0
    first = paired_bootstrap_difference(left, right, replicates=128, seed=32001)
    second = paired_bootstrap_difference(left, right, replicates=128, seed=32001)
    assert first == second
    assert first["mean_difference"] == (1.0, 1.0)
    assert first["ci_lower"] == (1.0, 1.0)
    assert first["ci_upper"] == (1.0, 1.0)
    assert first["probability_greater_than_zero"] == (1.0, 1.0)

    with pytest.raises(ValueError, match="same shape"):
        paired_bootstrap_difference(left, right[:, :1], replicates=32, seed=1)


def test_claim_boundaries_distinguish_benchmark_gt_from_e32_metrics() -> None:
    assert CLAIM_BOUNDARIES["official_ground_truth_is_directed_edge_support_only"]
    assert CLAIM_BOUNDARIES["node_labels_are_e32_projection_of_official_edge_endpoints"]
    assert CLAIM_BOUNDARIES["path_metric_is_e32_defined_not_official"]
    assert CLAIM_BOUNDARIES["pair_second_difference_is_not_a_communication_path"]
    assert CLAIM_BOUNDARIES["bootstrap_does_not_establish_external_generalization"]
    assert all(isinstance(value, bool) and value for value in CLAIM_BOUNDARIES.values())
