# ruff: noqa: EXE002
"""Tests for the lean E32 atlas-to-method analysis layer."""

from __future__ import annotations

import torch

from experiments.e32_interpbench.analysis import (
    build_atlas_method_scores,
    edge_sparsity_summary,
    transitive_candidate_non_edges,
)

NODES = ("a", "b", "c", "d")
EDGES = (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"))


def _two_class_effect(amplitudes: torch.Tensor) -> torch.Tensor:
    return torch.stack((-amplitudes, amplitudes), dim=-1)


def test_atlas_methods_reduce_declared_item_first_axes_and_frozen_universes() -> None:
    batch = 4
    positions = 2
    clean = torch.zeros(batch, positions, 2)
    singleton_amplitudes = torch.tensor((4.0, 3.0, 2.0, 1.0))
    singleton = _two_class_effect(
        singleton_amplitudes[:, None, None].expand(-1, batch, positions)
    )
    pair_amplitudes = torch.tensor(
        (
            (5.0, 5.0),
            (1.0, 1.0),
            (4.0, 4.0),
            (0.5, 0.5),
        )
    )
    interactions = _two_class_effect(pair_amplitudes[:, None, :].expand(-1, batch, -1))

    methods = build_atlas_method_scores(
        node_universe=NODES,
        edge_universe=EDGES,
        clean_output=clean,
        singleton_outputs=singleton,
        pair_interactions=interactions,
    )

    assert tuple(methods) == (
        "singleton_endpoint_product",
        "raw_pair_interaction",
        "rank_one_node_factor",
        "positive_pair_residual",
        "absolute_pair_residual",
    )
    assert methods["raw_pair_interaction"].edge_scores == (25.0, 1.0, 16.0, 0.25)
    assert methods["singleton_endpoint_product"].node_scores == (
        16.0,
        9.0,
        4.0,
        1.0,
    )
    assert all(len(method.node_scores) == 4 for method in methods.values())
    assert all(len(method.edge_scores) == 4 for method in methods.values())


def test_edge_sparsity_reports_k90_and_effective_support() -> None:
    summary = edge_sparsity_summary((9.0, 1.0, 0.0, 0.0))
    assert summary["edge_count"] == 4
    assert summary["k90"] == 1
    assert summary["top_1_energy_fraction"] == 0.9
    assert summary["effective_support"] == 100.0 / 82.0


def test_transitive_candidate_non_edges_separate_reachability_from_direct_gt() -> None:
    candidates = (*EDGES, ("a", "d"))
    assert transitive_candidate_non_edges(
        node_universe=NODES,
        edge_universe=candidates,
        ground_truth_edges=(("a", "b"), ("b", "d")),
    ) == (("a", "d"),)
