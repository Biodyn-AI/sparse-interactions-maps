# ruff: noqa: EXE002
"""Model-agnostic sparse-interaction score constructors and adapters."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from experiments.e32_interpbench.protocol import AdaptedScores, Edge


def _finite_double(value: Any, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must be finite")
    return tensor


def _nonnegative_double(value: Any, *, label: str) -> torch.Tensor:
    tensor = _finite_double(value, label=label)
    if bool((tensor < 0.0).any()):
        raise ValueError(f"{label} must be nonnegative")
    return tensor


def _identity_node(node: str) -> str:
    return node


def _promoted_node(node: Any, promote_node: Callable[[str], str]) -> str:
    if not isinstance(node, str) or not node:
        raise TypeError("node identifiers must be nonempty strings")
    promoted = promote_node(node)
    if not isinstance(promoted, str) or not promoted:
        raise TypeError("promote_node must return a nonempty string")
    return promoted


def adapt_ground_truth_edges(
    raw_edges: Sequence[tuple[str, str]],
    *,
    promote_node: Callable[[str], str] = _identity_node,
) -> tuple[Edge, ...]:
    """Apply a generic endpoint map to a ``list[tuple[str, str]]`` edge list.

    Promotion happens before stable deduplication.  This is important because
    several fine endpoints can collapse to one component.  Exact InterpBench
    Q/K/V normalization also reroutes and drops edge classes, so callers must
    use the benchmark adapter for that official comparison rather than a plain
    endpoint map here.
    """
    if isinstance(raw_edges, (str, bytes)):
        raise TypeError("raw_edges must be a sequence of directed pairs")
    result: list[Edge] = []
    seen: set[Edge] = set()
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, (tuple, list)) or len(raw_edge) != 2:
            raise TypeError("each ground-truth edge must be a directed pair")
        edge = (
            _promoted_node(raw_edge[0], promote_node),
            _promoted_node(raw_edge[1], promote_node),
        )
        if edge not in seen:
            seen.add(edge)
            result.append(edge)
    if not result:
        raise ValueError("ground-truth edge support must not be empty")
    return tuple(result)


def _mean_square_over_sample_dims(
    effects: torch.Tensor, *, item_ndim: int
) -> torch.Tensor:
    if isinstance(item_ndim, bool) or not isinstance(item_ndim, int) or item_ndim < 1:
        raise ValueError("item_ndim must be a positive integer")
    if effects.ndim <= item_ndim:
        raise ValueError("effects must include at least one sample dimension")
    sample_dims = tuple(range(effects.ndim - item_ndim))
    return effects.square().mean(dim=sample_dims)


def singleton_node_scores(effects: Any, *, item_ndim: int = 1) -> torch.Tensor:
    """Return mean-squared singleton effects, preserving trailing item axes."""
    values = _finite_double(effects, label="singleton effects")
    return _mean_square_over_sample_dims(values, item_ndim=item_ndim)


def raw_pair_interaction_scores(quartets: Any, *, item_ndim: int = 1) -> torch.Tensor:
    """Score raw ``R11-R10-R01+R00`` interactions by held-out mean square.

    Cells must be ordered ``00,10,01,11``.  ``item_ndim=1`` scores an edge
    list; ``item_ndim=2`` preserves a dense left-by-right pair matrix.
    """
    values = _finite_double(quartets, label="pair quartets")
    if values.ndim < item_ndim + 2 or values.shape[-1] != 4:
        raise ValueError("pair quartets need sample, item, and four cell axes")
    interaction = values[..., 3] - values[..., 1] - values[..., 2] + values[..., 0]
    return _mean_square_over_sample_dims(interaction, item_ndim=item_ndim)


@dataclass(frozen=True)
class RankOneEndpointFactor:
    left_scores: torch.Tensor
    right_scores: torch.Tensor
    reconstruction: torch.Tensor
    explained_energy_fraction: float


def rank_one_endpoint_factor(pair_scores: Any) -> RankOneEndpointFactor:
    """Fit a nonnegative leading-SVD endpoint factor to a dense score matrix."""
    matrix = _nonnegative_double(pair_scores, label="pair scores")
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("pair scores must be a nonempty matrix")
    left_vectors, singular_values, right_vectors_t = torch.linalg.svd(
        matrix, full_matrices=False
    )
    scale = torch.sqrt(singular_values[0])
    # A nonnegative matrix has a nonnegative leading singular-vector choice.
    # abs removes the otherwise arbitrary simultaneous SVD sign.
    left = scale * left_vectors[:, 0].abs()
    right = scale * right_vectors_t[0].abs()
    reconstruction = torch.outer(left, right)
    total_energy = float(matrix.square().sum())
    explained = (
        1.0
        if total_energy == 0.0
        else float(singular_values[0].square()) / total_energy
    )
    return RankOneEndpointFactor(left, right, reconstruction, explained)


@dataclass(frozen=True)
class MaskedRankOneEndpointFactor:
    """Source/target factors fitted only on declared directed candidate edges."""

    source_scores: torch.Tensor
    target_scores: torch.Tensor
    edge_reconstruction: torch.Tensor
    explained_energy_fraction: float
    squared_error: float
    valid_edge_count: int
    converged: bool
    iterations: int


def masked_rank_one_endpoint_factor(
    *,
    node_universe: Sequence[str],
    edge_universe: Sequence[Edge],
    edge_scores: Any,
    max_iterations: int = 2_000,
    tolerance: float = 1e-12,
) -> MaskedRankOneEndpointFactor:
    """Fit ``source_propensity * target_propensity`` on valid edges only.

    Alternating nonnegative least squares never materializes a zero-filled
    square matrix.  Missing, reverse, and otherwise invalid DAG cells therefore
    contribute no loss.  Source and target roles remain separate even when they
    refer to the same model component.
    """
    nodes, edges = _validate_universes(node_universe, edge_universe)
    if any(source == target for source, target in edges):
        raise ValueError("masked endpoint fitting does not allow self-edges")
    values = _nonnegative_double(edge_scores, label="edge scores")
    if values.ndim != 1 or values.numel() != len(edges):
        raise ValueError("edge_scores must align exactly with edge_universe")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")

    node_index = {node: index for index, node in enumerate(nodes)}
    source_index = torch.tensor(
        [node_index[source] for source, _ in edges], dtype=torch.long
    )
    target_index = torch.tensor(
        [node_index[target] for _, target in edges], dtype=torch.long
    )
    source_active = tuple(sorted(set(source_index.tolist())))
    target_active = tuple(sorted(set(target_index.tolist())))
    total_energy = float(values.square().sum())
    if total_energy == 0.0:
        zeros = torch.zeros(len(nodes), dtype=torch.float64)
        return MaskedRankOneEndpointFactor(
            source_scores=zeros,
            target_scores=zeros.clone(),
            edge_reconstruction=torch.zeros_like(values),
            explained_energy_fraction=1.0,
            squared_error=0.0,
            valid_edge_count=len(edges),
            converged=True,
            iterations=0,
        )

    initial_scale = math.sqrt(float(values.mean()))
    source_factor = torch.full((len(nodes),), initial_scale, dtype=torch.float64)
    target_factor = torch.full((len(nodes),), initial_scale, dtype=torch.float64)
    previous_reconstruction = source_factor[source_index] * target_factor[target_index]
    converged = False
    iterations = max_iterations
    comparison_scale = max(1.0, float(values.max()))

    for iteration in range(1, max_iterations + 1):
        updated_source = torch.zeros_like(source_factor)
        for index in source_active:
            edge_mask = source_index == index
            targets = target_index[edge_mask]
            denominator = target_factor[targets].square().sum()
            if float(denominator) > 0.0:
                updated_source[index] = (
                    values[edge_mask] * target_factor[targets]
                ).sum() / denominator

        updated_target = torch.zeros_like(target_factor)
        for index in target_active:
            edge_mask = target_index == index
            sources = source_index[edge_mask]
            denominator = updated_source[sources].square().sum()
            if float(denominator) > 0.0:
                updated_target[index] = (
                    values[edge_mask] * updated_source[sources]
                ).sum() / denominator

        # Remove the arbitrary global scale without changing predictions.
        active_source_values = updated_source[list(source_active)]
        source_rms = torch.sqrt(active_source_values.square().mean())
        if float(source_rms) > 0.0:
            updated_source /= source_rms
            updated_target *= source_rms

        reconstruction = updated_source[source_index] * updated_target[target_index]
        maximum_change = float((reconstruction - previous_reconstruction).abs().max())
        source_factor = updated_source
        target_factor = updated_target
        previous_reconstruction = reconstruction
        if maximum_change <= tolerance * comparison_scale:
            converged = True
            iterations = iteration
            break

    squared_error = float((values - previous_reconstruction).square().sum())
    explained = 1.0 - squared_error / total_energy
    return MaskedRankOneEndpointFactor(
        source_scores=source_factor,
        target_scores=target_factor,
        edge_reconstruction=previous_reconstruction,
        explained_energy_fraction=explained,
        squared_error=squared_error,
        valid_edge_count=len(edges),
        converged=converged,
        iterations=iterations,
    )


def endpoint_singleton_factor(left_scores: Any, right_scores: Any) -> torch.Tensor:
    """Return the product baseline from nonnegative endpoint singleton scores."""
    left = _nonnegative_double(left_scores, label="left singleton scores")
    right = _nonnegative_double(right_scores, label="right singleton scores")
    if left.ndim != 1 or right.ndim != 1 or not left.numel() or not right.numel():
        raise ValueError("endpoint singleton scores must be nonempty vectors")
    return torch.outer(left, right)


def masked_endpoint_singleton_factor(
    *,
    node_universe: Sequence[str],
    edge_universe: Sequence[Edge],
    singleton_scores: Any,
) -> torch.Tensor:
    """Multiply endpoint singleton scores on declared candidate edges only."""
    nodes, edges = _validate_universes(node_universe, edge_universe)
    values = _nonnegative_double(singleton_scores, label="singleton scores")
    if values.ndim != 1 or values.numel() != len(nodes):
        raise ValueError("singleton_scores must align exactly with node_universe")
    node_index = {node: index for index, node in enumerate(nodes)}
    return torch.tensor(
        [
            float(values[node_index[source]] * values[node_index[target]])
            for source, target in edges
        ],
        dtype=torch.float64,
    )


def sparse_pair_residual(
    pair_scores: Any,
    endpoint_baseline: Any,
    *,
    mode: Literal["positive", "absolute"] = "positive",
) -> torch.Tensor:
    """Rank pair evidence left after an endpoint-factor baseline.

    Positive mode asks which pairs are stronger than the endpoint baseline;
    absolute mode asks where that separable baseline is inaccurate in either
    direction.  Neither mode identifies an independent causal pathway.
    """
    pair = _finite_double(pair_scores, label="pair scores")
    baseline = _finite_double(endpoint_baseline, label="endpoint baseline")
    if pair.shape != baseline.shape:
        raise ValueError("pair scores and endpoint baseline must have the same shape")
    residual = pair - baseline
    if mode == "positive":
        return residual.clamp_min(0.0)
    if mode == "absolute":
        return residual.abs()
    raise ValueError("mode must be 'positive' or 'absolute'")


def _validate_universes(
    node_universe: Sequence[str], edge_universe: Sequence[Edge]
) -> tuple[tuple[str, ...], tuple[Edge, ...]]:
    nodes = tuple(node_universe)
    edges = tuple(tuple(edge) for edge in edge_universe)
    if not nodes or any(not isinstance(node, str) or not node for node in nodes):
        raise ValueError("node_universe must contain nonempty strings")
    if len(set(nodes)) != len(nodes):
        raise ValueError("node_universe must be unique")
    if not edges or any(
        len(edge) != 2 or not all(isinstance(node, str) and node for node in edge)
        for edge in edges
    ):
        raise ValueError("edge_universe must contain directed string pairs")
    if len(set(edges)) != len(edges):
        raise ValueError("edge_universe must be unique")
    if any(source not in nodes or target not in nodes for source, target in edges):
        raise ValueError("every edge endpoint must occur in node_universe")
    return nodes, edges


def _aggregate_promoted_scores(
    records: Sequence[tuple[Any, float]],
    *,
    reducer: Literal["max", "mean", "sum"],
) -> dict[Any, float]:
    grouped: dict[Any, list[float]] = {}
    for identifier, score in records:
        numeric = float(score)
        if not math.isfinite(numeric):
            raise ValueError("method scores must be finite")
        grouped.setdefault(identifier, []).append(numeric)
    if reducer == "max":
        return {key: max(values) for key, values in grouped.items()}
    if reducer == "mean":
        return {key: sum(values) / len(values) for key, values in grouped.items()}
    if reducer == "sum":
        return {key: sum(values) for key, values in grouped.items()}
    raise ValueError("collision_reducer must be 'max', 'mean', or 'sum'")


def adapt_method_scores(
    *,
    node_universe: Sequence[str],
    edge_universe: Sequence[Edge],
    raw_node_scores: Mapping[str, float],
    raw_edge_scores: Mapping[Edge, float],
    promote_node: Callable[[str], str] = _identity_node,
    collision_reducer: Literal["max", "mean", "sum"] = "max",
    missing_score: float = 0.0,
) -> AdaptedScores:
    """Promote, collision-reduce, and align one method to frozen universes."""
    nodes, edges = _validate_universes(node_universe, edge_universe)
    fill = float(missing_score)
    if not math.isfinite(fill):
        raise ValueError("missing_score must be finite")
    promoted_nodes = [
        (_promoted_node(node, promote_node), score)
        for node, score in raw_node_scores.items()
    ]
    promoted_edges = [
        (
            (
                _promoted_node(edge[0], promote_node),
                _promoted_node(edge[1], promote_node),
            ),
            score,
        )
        for edge, score in raw_edge_scores.items()
        if isinstance(edge, (tuple, list)) and len(edge) == 2
    ]
    if len(promoted_edges) != len(raw_edge_scores):
        raise TypeError("raw edge-score keys must be directed pairs")
    node_map = _aggregate_promoted_scores(promoted_nodes, reducer=collision_reducer)
    edge_map = _aggregate_promoted_scores(promoted_edges, reducer=collision_reducer)
    unknown_nodes = set(node_map).difference(nodes)
    unknown_edges = set(edge_map).difference(edges)
    if unknown_nodes or unknown_edges:
        raise ValueError("promoted method output lies outside the declared universe")
    return AdaptedScores(
        node_scores=tuple(node_map.get(node, fill) for node in nodes),
        edge_scores=tuple(edge_map.get(edge, fill) for edge in edges),
    )
