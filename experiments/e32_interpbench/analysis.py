# ruff: noqa: EXE002
"""Bridge E32 item-first atlas tensors to exact localization score vectors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from experiments.e32_interpbench.methods import (
    masked_endpoint_singleton_factor,
    masked_rank_one_endpoint_factor,
    sparse_pair_residual,
)
from experiments.e32_interpbench.protocol import AdaptedScores, Edge


def _finite_double(value: Any, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must be finite")
    return tensor


def _validated_universes(
    node_universe: Sequence[str], edge_universe: Sequence[Edge]
) -> tuple[tuple[str, ...], tuple[Edge, ...]]:
    nodes = tuple(node_universe)
    edges = tuple(tuple(edge) for edge in edge_universe)
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("node_universe must be nonempty and unique")
    if not edges or len(edges) != len(set(edges)):
        raise ValueError("edge_universe must be nonempty and unique")
    if any(
        len(edge) != 2 or edge[0] not in nodes or edge[1] not in nodes for edge in edges
    ):
        raise ValueError("edge_universe endpoints must lie in node_universe")
    return nodes, edges


def _item_first_energy(
    values: Any,
    *,
    item_count: int,
    label: str,
) -> torch.Tensor:
    tensor = _finite_double(values, label=label)
    if tensor.ndim != 4 or tensor.shape[0] != item_count:
        raise ValueError(
            f"{label} must have shape [item, example, non_bos_position, class]"
        )
    if min(tensor.shape[1:]) < 1:
        raise ValueError(f"{label} sample axes must be nonempty")
    return tensor.square().mean(dim=(1, 2, 3))


def _incident_mean(
    *,
    nodes: tuple[str, ...],
    edges: tuple[Edge, ...],
    edge_scores: torch.Tensor,
) -> torch.Tensor:
    if edge_scores.ndim != 1 or edge_scores.numel() != len(edges):
        raise ValueError("edge scores must align exactly with edge_universe")
    scores = torch.zeros(len(nodes), dtype=torch.float64)
    counts = torch.zeros(len(nodes), dtype=torch.float64)
    index = {node: position for position, node in enumerate(nodes)}
    for edge_index, (source, target) in enumerate(edges):
        for node in (source, target):
            position = index[node]
            scores[position] += edge_scores[edge_index]
            counts[position] += 1.0
    if bool((counts == 0.0).any()):
        raise ValueError("every node must occur in at least one candidate edge")
    return scores / counts


def _adapted(node_scores: torch.Tensor, edge_scores: torch.Tensor) -> AdaptedScores:
    return AdaptedScores(
        node_scores=tuple(float(value) for value in node_scores),
        edge_scores=tuple(float(value) for value in edge_scores),
    )


def build_atlas_method_scores(
    *,
    node_universe: Sequence[str],
    edge_universe: Sequence[Edge],
    clean_output: Any,
    singleton_outputs: Any,
    pair_interactions: Any,
) -> dict[str, AdaptedScores]:
    """Construct the frozen E32 non-gradient method panel.

    Inputs must already be the acquisition's scientific coordinate:
    class-centered logits at non-BOS positions.  Axes are explicit and
    item-first, preventing the generic method helpers from silently reducing
    over nodes or edges as if they were sample axes.
    """
    nodes, edges = _validated_universes(node_universe, edge_universe)
    empty = _finite_double(clean_output, label="clean_output")
    singletons = _finite_double(singleton_outputs, label="singleton_outputs")
    if empty.ndim != 3 or tuple(singletons.shape) != (
        len(nodes),
        *empty.shape,
    ):
        raise ValueError(
            "clean/singleton outputs must have [example, non_bos_position, class] "
            "and [node, example, non_bos_position, class] shapes"
        )
    singleton_effects = singletons - empty.unsqueeze(0)
    singleton_energy = _item_first_energy(
        singleton_effects,
        item_count=len(nodes),
        label="singleton effects",
    )
    pair_energy = _item_first_energy(
        pair_interactions,
        item_count=len(edges),
        label="pair interactions",
    )

    endpoint_product = masked_endpoint_singleton_factor(
        node_universe=nodes,
        edge_universe=edges,
        singleton_scores=singleton_energy,
    )
    raw_pair_nodes = _incident_mean(
        nodes=nodes,
        edges=edges,
        edge_scores=pair_energy,
    )
    rank_one = masked_rank_one_endpoint_factor(
        node_universe=nodes,
        edge_universe=edges,
        edge_scores=pair_energy,
    )
    rank_one_nodes = _incident_mean(
        nodes=nodes,
        edges=edges,
        edge_scores=rank_one.edge_reconstruction,
    )
    positive_residual = sparse_pair_residual(
        pair_energy, rank_one.edge_reconstruction, mode="positive"
    )
    absolute_residual = sparse_pair_residual(
        pair_energy, rank_one.edge_reconstruction, mode="absolute"
    )
    return {
        "singleton_endpoint_product": _adapted(singleton_energy, endpoint_product),
        "raw_pair_interaction": _adapted(raw_pair_nodes, pair_energy),
        "rank_one_node_factor": _adapted(rank_one_nodes, rank_one.edge_reconstruction),
        "positive_pair_residual": _adapted(
            _incident_mean(
                nodes=nodes,
                edges=edges,
                edge_scores=positive_residual,
            ),
            positive_residual,
        ),
        "absolute_pair_residual": _adapted(
            _incident_mean(
                nodes=nodes,
                edges=edges,
                edge_scores=absolute_residual,
            ),
            absolute_residual,
        ),
    }


def edge_sparsity_summary(edge_scores: Sequence[float]) -> dict[str, int | float]:
    """Return selection-independent concentration statistics for edge energy."""
    values = _finite_double(edge_scores, label="edge_scores")
    if values.ndim != 1 or values.numel() < 1 or bool((values < 0.0).any()):
        raise ValueError("edge_scores must be a nonempty nonnegative vector")
    total = float(values.sum())
    if not total > 0.0:
        raise ValueError("edge_scores must have positive total energy")
    ordered = values.sort(descending=True).values
    cumulative = ordered.cumsum(dim=0) / total
    k90 = int(torch.searchsorted(cumulative, torch.tensor(0.9)).item()) + 1
    squared_sum = float(values.square().sum())
    effective_support = total * total / squared_sum
    return {
        "edge_count": values.numel(),
        "k90": k90,
        "top_1_energy_fraction": float(ordered[0]) / total,
        "effective_support": effective_support,
        "normalized_effective_support": effective_support / values.numel(),
    }


def transitive_candidate_non_edges(
    *,
    node_universe: Sequence[str],
    edge_universe: Sequence[Edge],
    ground_truth_edges: Sequence[Edge],
) -> tuple[Edge, ...]:
    """Return candidate negatives connected by a length-two-or-more GT path."""
    nodes, candidates = _validated_universes(node_universe, edge_universe)
    direct = frozenset(tuple(edge) for edge in ground_truth_edges)
    if not direct or not direct.issubset(candidates):
        raise ValueError("ground_truth_edges must be a nonempty candidate subset")
    adjacency: dict[str, tuple[str, ...]] = {
        node: tuple(target for source, target in direct if source == node)
        for node in nodes
    }

    def reachable(source: str, target: str) -> bool:
        frontier = list(adjacency[source])
        visited = {source}
        while frontier:
            node = frontier.pop()
            if node == target:
                return True
            if node not in visited:
                visited.add(node)
                frontier.extend(adjacency[node])
        return False

    return tuple(
        edge
        for edge in candidates
        if edge not in direct and reachable(edge[0], edge[1])
    )


__all__ = [
    "build_atlas_method_scores",
    "edge_sparsity_summary",
    "transitive_candidate_non_edges",
]
