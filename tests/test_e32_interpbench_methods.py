# ruff: noqa: EXE002
"""Tests for model-agnostic E32 sparse-interaction score construction."""

from __future__ import annotations

import pytest
import torch

from experiments.e32_interpbench.methods import (
    adapt_ground_truth_edges,
    adapt_method_scores,
    endpoint_singleton_factor,
    masked_endpoint_singleton_factor,
    masked_rank_one_endpoint_factor,
    rank_one_endpoint_factor,
    raw_pair_interaction_scores,
    singleton_node_scores,
    sparse_pair_residual,
)


def test_generic_edge_list_adapter_promotes_then_deduplicates() -> None:
    raw_edges = [
        ("embed", "head.q[0]"),
        ("embed", "head.k[0]"),
        ("head.result[0]", "mlp"),
    ]

    def promote(node: str) -> str:
        return "head[0]" if node.startswith("head.") else node

    adapted = adapt_ground_truth_edges(raw_edges, promote_node=promote)
    assert adapted == (("embed", "head[0]"), ("head[0]", "mlp"))


def test_singleton_and_raw_pair_scores_are_mean_squared_effects() -> None:
    singleton_effects = torch.tensor(
        [[1.0, 2.0, -1.0], [3.0, 0.0, 1.0]], dtype=torch.float64
    )
    assert torch.equal(
        singleton_node_scores(singleton_effects),
        torch.tensor([5.0, 2.0, 1.0], dtype=torch.float64),
    )

    # Ordered cells are 00, 10, 01, 11. The two signed interactions are 2 and -2.
    quartets = torch.tensor(
        [
            [[1.0, 3.0, 4.0, 8.0], [0.0, 2.0, 3.0, 3.0]],
            [[0.0, 0.0, 0.0, 2.0], [2.0, 4.0, 5.0, 5.0]],
        ],
        dtype=torch.float64,
    )
    assert torch.equal(
        raw_pair_interaction_scores(quartets),
        torch.tensor([4.0, 4.0], dtype=torch.float64),
    )


def test_endpoint_factors_and_sparse_residual_separate_a_rank_one_backbone() -> None:
    pair_scores = torch.tensor([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]], dtype=torch.float64)
    factor = rank_one_endpoint_factor(pair_scores)
    assert torch.allclose(factor.reconstruction, pair_scores, atol=1e-12)
    assert torch.allclose(
        factor.reconstruction,
        torch.outer(factor.left_scores, factor.right_scores),
        atol=1e-12,
    )

    singleton_factor = endpoint_singleton_factor(
        torch.tensor([2.0, 1.0]), torch.tensor([1.0, 2.0, 3.0])
    )
    assert torch.equal(singleton_factor, pair_scores)

    perturbed = pair_scores.clone()
    perturbed[1, 2] += 5.0
    positive = sparse_pair_residual(perturbed, pair_scores, mode="positive")
    absolute = sparse_pair_residual(pair_scores - 6.0, pair_scores, mode="absolute")
    assert positive.nonzero().tolist() == [[1, 2]]
    assert positive[1, 2] == 5.0
    assert torch.equal(absolute, torch.full_like(pair_scores, 6.0))


def test_masked_rank_one_factor_ignores_invalid_dag_cells_instead_of_zero_filling() -> (
    None
):
    # These are the only valid directed candidate pairs in the ordered DAG.
    # Their scores are exactly source_propensity * target_propensity:
    # a->b = 2*3, a->c = 2*4, b->c = 1*4.
    factor = masked_rank_one_endpoint_factor(
        node_universe=("a", "b", "c"),
        edge_universe=(("a", "b"), ("a", "c"), ("b", "c")),
        edge_scores=(6.0, 8.0, 4.0),
        max_iterations=10_000,
        tolerance=1e-14,
    )
    assert factor.converged is True
    assert torch.allclose(
        factor.edge_reconstruction,
        torch.tensor([6.0, 8.0, 4.0], dtype=torch.float64),
        atol=1e-9,
    )
    assert factor.explained_energy_fraction == pytest.approx(1.0, abs=1e-12)
    assert factor.valid_edge_count == 3
    # A zero-filled 3x3 SVD would treat six invalid/diagonal cells as evidence;
    # the masked fit has no reconstruction entries for any of them.
    assert tuple(factor.edge_reconstruction.shape) == (3,)


def test_masked_rank_one_factor_rejects_diagonal_or_out_of_universe_edges() -> None:
    with pytest.raises(ValueError, match="self-edges"):
        masked_rank_one_endpoint_factor(
            node_universe=("a", "b"),
            edge_universe=(("a", "a"),),
            edge_scores=(1.0,),
        )
    with pytest.raises(ValueError, match="node_universe"):
        masked_rank_one_endpoint_factor(
            node_universe=("a", "b"),
            edge_universe=(("a", "c"),),
            edge_scores=(1.0,),
        )


def test_masked_endpoint_singleton_factor_scores_only_declared_edges() -> None:
    observed = masked_endpoint_singleton_factor(
        node_universe=("a", "b", "c"),
        edge_universe=(("a", "b"), ("a", "c"), ("b", "c")),
        singleton_scores=(2.0, 3.0, 4.0),
    )
    assert torch.equal(observed, torch.tensor([6.0, 8.0, 12.0]).double())


def test_method_adapter_promotes_collisions_with_an_explicit_reducer() -> None:
    node_universe = ("embed", "head[0]", "mlp")
    edge_universe = (("embed", "head[0]"), ("head[0]", "mlp"))

    def promote(node: str) -> str:
        return "head[0]" if node.startswith("head.") else node

    scores = adapt_method_scores(
        node_universe=node_universe,
        edge_universe=edge_universe,
        raw_node_scores={"head.q[0]": 0.4, "head.k[0]": 0.8},
        raw_edge_scores={
            ("embed", "head.q[0]"): 0.3,
            ("embed", "head.k[0]"): 0.7,
        },
        promote_node=promote,
        collision_reducer="max",
        missing_score=0.0,
    )
    assert scores.node_scores == (0.0, 0.8, 0.0)
    assert scores.edge_scores == (0.7, 0.0)

    with pytest.raises(ValueError, match="outside the declared universe"):
        adapt_method_scores(
            node_universe=node_universe,
            edge_universe=edge_universe,
            raw_node_scores={"unknown": 1.0},
            raw_edge_scores={},
        )
