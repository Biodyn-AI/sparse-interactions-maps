# ruff: noqa: EXE002
"""Exact, pure scoring for E32 sparse-interaction benchmark comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from experiments.e32_interpbench.protocol import AdaptedScores, Edge, ScoreSnapshot


def _finite_vector(value: Any, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(f"{label} must be a nonempty vector")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must be finite")
    return tensor


def binary_ranking_metrics(
    *,
    scores: Sequence[float],
    labels: Sequence[bool],
    top_k: int | None = None,
) -> dict[str, Any]:
    """Return exact tie-aware AUROC/AP and deterministic top-k F1.

    AUROC uses half credit for positive-negative ties.  AUPRC is threshold-
    grouped average precision, so it is invariant to ordering within a tie.
    Frozen universe order is used only to make finite top-k selection exact.
    """
    values = _finite_vector(scores, label="scores")
    truth = torch.as_tensor(labels, dtype=torch.bool, device="cpu")
    if truth.ndim != 1 or truth.shape != values.shape:
        raise ValueError("labels must be a boolean vector matching scores")
    positive_count = int(truth.sum())
    negative_count = truth.numel() - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ranking metrics require positive and negative examples")

    positives = values[truth]
    negatives = values[~truth]
    comparisons = positives[:, None] - negatives[None, :]
    auroc = float(
        ((comparisons > 0.0).double() + 0.5 * (comparisons == 0.0).double()).mean()
    )

    grouped: dict[float, tuple[int, int]] = {}
    for score, is_positive in zip(values.tolist(), truth.tolist(), strict=True):
        group_positive, group_total = grouped.get(score, (0, 0))
        grouped[score] = (group_positive + int(is_positive), group_total + 1)
    cumulative_positive = 0
    cumulative_total = 0
    auprc = 0.0
    for threshold in sorted(grouped, reverse=True):
        group_positive, group_total = grouped[threshold]
        cumulative_positive += group_positive
        cumulative_total += group_total
        if group_positive:
            auprc += (group_positive / positive_count) * (
                cumulative_positive / cumulative_total
            )

    selected_count = positive_count if top_k is None else top_k
    if (
        isinstance(selected_count, bool)
        or not isinstance(selected_count, int)
        or not 1 <= selected_count <= values.numel()
    ):
        raise ValueError("top_k must be an integer within the universe")
    ranked = sorted(
        range(values.numel()), key=lambda index: (-float(values[index]), index)
    )
    selected = tuple(ranked[:selected_count])
    true_positive = sum(bool(truth[index]) for index in selected)
    precision = true_positive / selected_count
    recall = true_positive / positive_count
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {
        "auroc": auroc,
        "auprc": auprc,
        "top_k": selected_count,
        "top_k_precision": precision,
        "top_k_recall": recall,
        "top_k_f1": f1,
        "selected_indices": selected,
    }


def _validated_graph(
    node_universe: Sequence[str], edge_universe: Sequence[Edge]
) -> tuple[tuple[str, ...], tuple[Edge, ...]]:
    nodes = tuple(node_universe)
    edges = tuple(tuple(edge) for edge in edge_universe)
    if not nodes or len(set(nodes)) != len(nodes):
        raise ValueError("node_universe must be nonempty and unique")
    if not edges or len(set(edges)) != len(edges):
        raise ValueError("edge_universe must be nonempty and unique")
    if any(
        len(edge) != 2 or edge[0] not in nodes or edge[1] not in nodes for edge in edges
    ):
        raise ValueError("edge_universe endpoints must lie in node_universe")
    return nodes, edges


def evaluate_localization(
    *,
    node_universe: Sequence[str],
    edge_universe: Sequence[Edge],
    ground_truth_edges: Sequence[Edge],
    ground_truth_nodes: Sequence[str] | None = None,
    node_scores: Sequence[float],
    edge_scores: Sequence[float],
    top_k_nodes: int | None = None,
    top_k_edges: int | None = None,
) -> dict[str, Any]:
    """Score node and directed-edge support against promoted InterpBench GT."""
    nodes, edges = _validated_graph(node_universe, edge_universe)
    truth_edges = frozenset(tuple(edge) for edge in ground_truth_edges)
    if not truth_edges or not truth_edges.issubset(edges):
        raise ValueError(
            "ground_truth_edges must be a nonempty subset of edge_universe"
        )
    edge_truth_nodes = frozenset(node for edge in truth_edges for node in edge)
    if ground_truth_nodes is None:
        truth_nodes = edge_truth_nodes
    else:
        truth_node_sequence = tuple(ground_truth_nodes)
        truth_nodes = frozenset(truth_node_sequence)
        if (
            not truth_nodes
            or len(truth_nodes) != len(truth_node_sequence)
            or not truth_nodes.issubset(nodes)
            or not edge_truth_nodes.issubset(truth_nodes)
        ):
            raise ValueError(
                "ground_truth_nodes must be a unique node-universe subset "
                "containing every ground-truth edge endpoint"
            )
    node_values = _finite_vector(node_scores, label="node_scores")
    edge_values = _finite_vector(edge_scores, label="edge_scores")
    if node_values.numel() != len(nodes) or edge_values.numel() != len(edges):
        raise ValueError("score lengths must match their frozen universes")
    node_metrics = binary_ranking_metrics(
        scores=node_values.tolist(),
        labels=tuple(node in truth_nodes for node in nodes),
        top_k=top_k_nodes,
    )
    edge_metrics = binary_ranking_metrics(
        scores=edge_values.tolist(),
        labels=tuple(edge in truth_edges for edge in edges),
        top_k=top_k_edges,
    )
    node_metrics["support_precision"] = node_metrics["top_k_precision"]
    edge_metrics["support_precision"] = edge_metrics["top_k_precision"]
    node_metrics["selected_support"] = tuple(
        nodes[index] for index in node_metrics["selected_indices"]
    )
    edge_metrics["selected_support"] = tuple(
        edges[index] for index in edge_metrics["selected_indices"]
    )
    return {
        "ground_truth_node_count": len(truth_nodes),
        "ground_truth_edge_count": len(truth_edges),
        "node": node_metrics,
        "edge": edge_metrics,
    }


def evaluate_method_panel(
    *,
    method_scores: Mapping[str, AdaptedScores],
    node_universe: Sequence[str],
    edge_universe: Sequence[Edge],
    ground_truth_edges: Sequence[Edge],
    ground_truth_nodes: Sequence[str] | None = None,
    top_k_nodes: int | None = None,
    top_k_edges: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare methods on one frozen universe, GT support, and top-k policy."""
    if not method_scores:
        raise ValueError("method_scores must not be empty")
    panel: dict[str, dict[str, Any]] = {}
    for method_name, scores in method_scores.items():
        if not isinstance(method_name, str) or not method_name:
            raise ValueError("method names must be nonempty strings")
        if not isinstance(scores, AdaptedScores):
            raise TypeError("every method value must be AdaptedScores")
        localization = evaluate_localization(
            node_universe=node_universe,
            edge_universe=edge_universe,
            ground_truth_edges=ground_truth_edges,
            ground_truth_nodes=ground_truth_nodes,
            node_scores=scores.node_scores,
            edge_scores=scores.edge_scores,
            top_k_nodes=top_k_nodes,
            top_k_edges=top_k_edges,
        )
        localization["path"] = path_recovery_metrics(
            selected_edges=localization["edge"]["selected_support"],
            ground_truth_edges=ground_truth_edges,
        )
        panel[method_name] = localization
    return panel


def _boundary_nodes(edges: frozenset[Edge]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    sources_seen = {source for source, _ in edges}
    targets_seen = {target for _, target in edges}
    return tuple(sorted(sources_seen - targets_seen)), tuple(
        sorted(targets_seen - sources_seen)
    )


def _simple_boundary_paths(
    edges: frozenset[Edge],
    sources: Sequence[str],
    sinks: Sequence[str],
    *,
    max_paths: int,
) -> tuple[tuple[str, ...], ...]:
    adjacency: dict[str, list[str]] = {}
    nodes = set()
    for source, target in edges:
        adjacency.setdefault(source, []).append(target)
        nodes.update((source, target))
    for targets in adjacency.values():
        targets.sort()
    sink_set = set(sinks)
    paths: list[tuple[str, ...]] = []

    def visit(node: str, path: tuple[str, ...]) -> None:
        if node in sink_set:
            paths.append(path)
            if len(paths) > max_paths:
                raise ValueError("path count exceeds max_paths")
            return
        for target in adjacency.get(node, ()):
            if target not in path:
                visit(target, (*path, target))

    for source in sorted(set(sources)):
        if source in nodes:
            visit(source, (source,))
    return tuple(sorted(set(paths)))


def path_recovery_metrics(
    *,
    selected_edges: Sequence[Edge],
    ground_truth_edges: Sequence[Edge],
    source_nodes: Sequence[str] | None = None,
    sink_nodes: Sequence[str] | None = None,
    max_paths: int = 100_000,
) -> dict[str, Any]:
    """Compare complete selected source-to-sink paths with GT paths.

    This exact path-completion metric is defined by E32, not by InterpBench.
    Boundaries default to source/sink nodes inferred from the GT graph.
    """
    if isinstance(max_paths, bool) or not isinstance(max_paths, int) or max_paths < 1:
        raise ValueError("max_paths must be a positive integer")
    selected = frozenset(tuple(edge) for edge in selected_edges)
    truth = frozenset(tuple(edge) for edge in ground_truth_edges)
    if not truth:
        raise ValueError("ground_truth_edges must not be empty")
    inferred_sources, inferred_sinks = _boundary_nodes(truth)
    sources = inferred_sources if source_nodes is None else tuple(source_nodes)
    sinks = inferred_sinks if sink_nodes is None else tuple(sink_nodes)
    if not sources or not sinks:
        raise ValueError("path metric requires nonempty source and sink boundaries")
    truth_paths = _simple_boundary_paths(truth, sources, sinks, max_paths=max_paths)
    selected_paths = _simple_boundary_paths(
        selected, sources, sinks, max_paths=max_paths
    )
    if not truth_paths:
        raise ValueError("ground-truth graph has no complete boundary path")
    truth_set = set(truth_paths)
    true_selected = len(set(selected_paths).intersection(truth_set))
    precision = 0.0 if not selected_paths else true_selected / len(selected_paths)
    recall = true_selected / len(truth_paths)
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {
        "ground_truth_paths": truth_paths,
        "selected_boundary_paths": selected_paths,
        "path_precision": precision,
        "path_recall": recall,
        "path_f1": f1,
        "metric_status": "e32_defined_not_official_interpbench",
    }


def _finite_curve_input(value: Any, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must be finite")
    return tensor


def _trapezoid(y: torch.Tensor, x: torch.Tensor) -> float:
    return float(((y[1:] + y[:-1]) * 0.5 * (x[1:] - x[:-1])).sum())


def normalized_faithfulness_curves(
    *,
    support_sizes: Sequence[int],
    support_total: int,
    clean: Any,
    empty: Any,
    keep_selected: Any,
    remove_selected: Any,
) -> dict[str, Any]:
    """Return held-out per-example and mean necessity/sufficiency curves.

    Inputs are behavior utilities where larger is better.  Scores are
    normalized per example by ``clean-empty`` and intentionally not clipped.
    """
    if (
        isinstance(support_total, bool)
        or not isinstance(support_total, int)
        or support_total < 1
    ):
        raise ValueError("support_total must be a positive integer")
    sizes = tuple(support_sizes)
    if (
        not sizes
        or any(isinstance(size, bool) or not isinstance(size, int) for size in sizes)
        or tuple(sorted(set(sizes))) != sizes
        or sizes[0] != 0
        or sizes[-1] != support_total
    ):
        raise ValueError("support_sizes must increase uniquely from 0 to support_total")
    clean_values = _finite_curve_input(clean, label="clean")
    empty_values = _finite_curve_input(empty, label="empty")
    keep = _finite_curve_input(keep_selected, label="keep_selected")
    remove = _finite_curve_input(remove_selected, label="remove_selected")
    if clean_values.ndim != 1 or empty_values.shape != clean_values.shape:
        raise ValueError("clean and empty must be matching example vectors")
    expected = (len(sizes), clean_values.numel())
    if tuple(keep.shape) != expected or tuple(remove.shape) != expected:
        raise ValueError("intervention arrays must have shape (prefixes, examples)")
    denominator = clean_values - empty_values
    if bool(torch.isclose(denominator, torch.zeros_like(denominator)).any()):
        raise ValueError("every example needs a nonzero clean-empty utility gap")
    sufficiency = (keep - empty_values[None, :]) / denominator[None, :]
    necessity = (clean_values[None, :] - remove) / denominator[None, :]
    sufficiency_mean = sufficiency.mean(dim=1)
    necessity_mean = necessity.mean(dim=1)
    fractions = torch.tensor(sizes, dtype=torch.float64) / support_total
    return {
        "support_sizes": sizes,
        "support_fraction": tuple(float(value) for value in fractions),
        "sufficiency_mean": tuple(float(value) for value in sufficiency_mean),
        "necessity_mean": tuple(float(value) for value in necessity_mean),
        "sufficiency_auc": _trapezoid(sufficiency_mean, fractions),
        "necessity_auc": _trapezoid(necessity_mean, fractions),
        "per_example_sufficiency": sufficiency.transpose(0, 1).contiguous(),
        "per_example_necessity": necessity.transpose(0, 1).contiguous(),
        "utility_orientation": "larger_is_better",
        "metric_status": "e32_intervention_output_not_official_ground_truth",
    }


def evaluate_query_prefixes(
    *,
    snapshots: Sequence[ScoreSnapshot],
    node_universe: Sequence[str],
    edge_universe: Sequence[Edge],
    ground_truth_edges: Sequence[Edge],
    ground_truth_nodes: Sequence[str] | None = None,
    top_k_nodes: int | None = None,
    top_k_edges: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Evaluate cumulative discovery snapshots at every registered query budget."""
    snapshots = tuple(snapshots)
    if not snapshots:
        raise ValueError("snapshots must not be empty")
    queries = tuple(snapshot.queries for snapshot in snapshots)
    if tuple(sorted(set(queries))) != queries:
        raise ValueError("snapshot queries must be strictly increasing")
    return tuple(
        {
            "queries": snapshot.queries,
            **evaluate_localization(
                node_universe=node_universe,
                edge_universe=edge_universe,
                ground_truth_edges=ground_truth_edges,
                ground_truth_nodes=ground_truth_nodes,
                node_scores=snapshot.node_scores,
                edge_scores=snapshot.edge_scores,
                top_k_nodes=top_k_nodes,
                top_k_edges=top_k_edges,
            ),
        }
        for snapshot in snapshots
    )


def _python_number_or_tuple(value: torch.Tensor) -> float | tuple[Any, ...]:
    if value.ndim == 0:
        return float(value)
    return tuple(_python_number_or_tuple(item) for item in value)


def paired_bootstrap_difference(
    left: Any,
    right: Any,
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired row bootstrap for scalar or prefix-vector benchmark-unit scores."""
    left_values = torch.as_tensor(left, dtype=torch.float64, device="cpu")
    right_values = torch.as_tensor(right, dtype=torch.float64, device="cpu")
    if left_values.shape != right_values.shape:
        raise ValueError("paired inputs must have the same shape")
    if left_values.ndim < 1 or left_values.shape[0] < 2:
        raise ValueError("paired inputs need at least two benchmark units")
    if not bool(
        torch.isfinite(left_values).all() and torch.isfinite(right_values).all()
    ):
        raise ValueError("paired inputs must be finite")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 2
    ):
        raise ValueError("replicates must be an integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    differences = left_values - right_values
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        differences.shape[0],
        (replicates, differences.shape[0]),
        generator=generator,
    )
    bootstrap = differences[indices].mean(dim=1)
    alpha = (1.0 - confidence) / 2.0
    lower = torch.quantile(bootstrap, alpha, dim=0)
    upper = torch.quantile(bootstrap, 1.0 - alpha, dim=0)
    return {
        "unit_count": differences.shape[0],
        "replicates": replicates,
        "seed": seed,
        "confidence": confidence,
        "mean_difference": _python_number_or_tuple(differences.mean(dim=0)),
        "ci_lower": _python_number_or_tuple(lower),
        "ci_upper": _python_number_or_tuple(upper),
        "probability_greater_than_zero": _python_number_or_tuple(
            (bootstrap > 0.0).double().mean(dim=0)
        ),
    }
