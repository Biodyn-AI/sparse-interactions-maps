# ruff: noqa: EXE002
"""Pure scoring for E32.3 sparse-core mass and set faithfulness."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from experiments.e32_interpbench.e32_3_protocol import (
    CASE_ORDER,
    DETERMINISTIC_METHOD_ORDER,
    EXPECTED_UNIVERSE_COUNTS,
    FAITHFULNESS_GATES,
    PRIMARY_METHOD,
    RANDOM_CONTROL_REPLICATES,
    SCREEN_GATES,
    SELECTOR_GATES,
    confirmatory_support_size,
    deterministic_random_support,
    support_sizes,
)
from experiments.e32_interpbench.methods import masked_rank_one_endpoint_factor


def _finite_double(value: Any, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must be finite")
    return tensor


def _item_energy(value: Any, *, label: str) -> torch.Tensor:
    tensor = _finite_double(value, label=label)
    if tensor.ndim != 4 or min(tensor.shape) < 1:
        raise ValueError(
            f"{label} must have shape [item, example, non_bos_position, coordinate]"
        )
    return tensor.square().mean(dim=(1, 2, 3))


def singleton_output_effect_scores(
    clean_output: Any, singleton_outputs: Any
) -> torch.Tensor:
    """Mean-squared class-centered output displacement for every node."""
    clean = _finite_double(clean_output, label="clean_output")
    singletons = _finite_double(singleton_outputs, label="singleton_outputs")
    if (
        clean.ndim != 3
        or singletons.ndim != 4
        or tuple(singletons.shape[1:]) != tuple(clean.shape)
    ):
        raise ValueError(
            "outputs must have [example, position, class] and "
            "[node, example, position, class] shapes"
        )
    return _item_energy(singletons - clean.unsqueeze(0), label="singleton effects")


def activation_shift_rms_scores(
    clean_activations: Any, corrupt_activations: Any
) -> torch.Tensor:
    """Task-conditioned donor-minus-clean activation RMS per whole component.

    Inputs use one common padded output coordinate per whole attention head or
    MLP: ``[node, example, position, d_model]``.  This avoids comparing raw
    weight norms across differently shaped head and MLP parameterizations.
    """
    clean = _finite_double(clean_activations, label="clean_activations")
    corrupt = _finite_double(corrupt_activations, label="corrupt_activations")
    if clean.ndim != 4 or corrupt.shape != clean.shape or min(clean.shape) < 1:
        raise ValueError(
            "activation tensors must share nonempty "
            "[node, example, position, d_model] shapes"
        )
    return torch.sqrt(
        _item_energy(corrupt - clean, label="donor-minus-clean activations")
    )


def raw_pair_interaction_mass(pair_interactions: Any) -> torch.Tensor:
    """Return whole-edge raw conditional second-difference energy."""
    return _item_energy(pair_interactions, label="pair_interactions")


def exact_fine_interaction_mass(
    *,
    clean_output: Any,
    corrupt_output: Any,
    corrupt_into_clean_outputs: Any,
    clean_into_corrupt_outputs: Any,
) -> torch.Tensor:
    """Return sign-symmetric exact fine-edge displacement energy.

    Q/K/V remain distinct fine coordinates.  No directional projection,
    ReLU, clipping, or whole-edge max promotion is applied.
    """
    clean = _finite_double(clean_output, label="clean_output")
    corrupt = _finite_double(corrupt_output, label="corrupt_output")
    necessity = _finite_double(
        corrupt_into_clean_outputs, label="corrupt_into_clean_outputs"
    )
    sufficiency = _finite_double(
        clean_into_corrupt_outputs, label="clean_into_corrupt_outputs"
    )
    if clean.ndim != 3 or corrupt.shape != clean.shape:
        raise ValueError(
            "clean and corrupt outputs must share [example, position, class]"
        )
    if (
        necessity.ndim != 4
        or sufficiency.shape != necessity.shape
        or tuple(necessity.shape[1:]) != tuple(clean.shape)
        or necessity.shape[0] < 1
    ):
        raise ValueError(
            "replacement outputs must share [fine, example, position, class]"
        )
    necessity_energy = _item_energy(
        clean.unsqueeze(0) - necessity, label="necessity fine displacement"
    )
    sufficiency_energy = _item_energy(
        sufficiency - corrupt.unsqueeze(0), label="sufficiency fine displacement"
    )
    return 0.5 * (necessity_energy + sufficiency_energy)


def _validated_graph(
    whole_edges: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    edges = tuple(tuple(edge) for edge in whole_edges)
    if (
        not edges
        or len(edges) != len(set(edges))
        or any(
            len(edge) != 2
            or not all(isinstance(node, str) and node for node in edge)
            or edge[0] == edge[1]
            for edge in edges
        )
    ):
        raise ValueError("whole_edges must be unique directed pairs")
    return edges


def interaction_mass_retention(
    *,
    item_mass: Any,
    selected_nodes: Sequence[str],
    whole_edges: Sequence[tuple[str, str]],
    fine_to_whole_indices: Sequence[int] | None = None,
) -> float:
    """Fraction of nonnegative mass whose promoted endpoints both lie in S."""
    mass = _finite_double(item_mass, label="item_mass")
    edges = _validated_graph(whole_edges)
    selected = frozenset(selected_nodes)
    universe_nodes = frozenset(node for edge in edges for node in edge)
    if (
        mass.ndim != 1
        or not selected
        or not selected.issubset(universe_nodes)
        or bool((mass < 0.0).any())
        or not float(mass.sum()) > 0.0
    ):
        raise ValueError("mass and selected support must be valid and nonempty")
    if fine_to_whole_indices is None:
        if mass.numel() != len(edges):
            raise ValueError("whole mass must align with whole_edges")
        promoted_indices = tuple(range(len(edges)))
    else:
        promoted_indices = tuple(fine_to_whole_indices)
        if len(promoted_indices) != mass.numel() or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(edges)
            for index in promoted_indices
        ):
            raise ValueError("fine_to_whole_indices must align with fine mass")
    retained = torch.tensor(
        [
            edges[whole_index][0] in selected and edges[whole_index][1] in selected
            for whole_index in promoted_indices
        ],
        dtype=torch.bool,
    )
    return float(mass[retained].sum() / mass.sum())


def incident_mean_node_scores(
    *,
    node_universe: Sequence[str],
    whole_edges: Sequence[tuple[str, str]],
    whole_edge_scores: Any,
) -> torch.Tensor:
    """Degree-normalized mean score over incoming and outgoing whole edges."""
    nodes = tuple(node_universe)
    edges = _validated_graph(whole_edges)
    values = _finite_double(whole_edge_scores, label="whole_edge_scores")
    if (
        not nodes
        or len(nodes) != len(set(nodes))
        or any(not isinstance(node, str) or not node for node in nodes)
        or values.ndim != 1
        or values.numel() != len(edges)
        or bool((values < 0.0).any())
    ):
        raise ValueError("nodes and nonnegative whole-edge scores must align")
    index = {node: position for position, node in enumerate(nodes)}
    if any(source not in index or target not in index for source, target in edges):
        raise ValueError("whole-edge endpoints must lie in node_universe")
    totals = torch.zeros(len(nodes), dtype=torch.float64)
    counts = torch.zeros(len(nodes), dtype=torch.float64)
    for edge_index, (source, target) in enumerate(edges):
        for node in (source, target):
            node_index = index[node]
            totals[node_index] += values[edge_index]
            counts[node_index] += 1.0
    if bool((counts == 0.0).any()):
        raise ValueError("every node must occur in a whole edge")
    return totals / counts


def promote_absolute_fine_max(
    *,
    fine_scores: Any,
    fine_to_whole_indices: Sequence[int],
    whole_edge_count: int,
) -> torch.Tensor:
    """Promote absolute EAP-IG5 Q/K/V scores by a frozen whole-edge max."""
    values = _finite_double(fine_scores, label="fine_scores")
    mapping = tuple(fine_to_whole_indices)
    if (
        isinstance(whole_edge_count, bool)
        or not isinstance(whole_edge_count, int)
        or whole_edge_count < 1
        or values.ndim != 1
        or values.numel() != len(mapping)
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < whole_edge_count
            for index in mapping
        )
        or set(mapping) != set(range(whole_edge_count))
    ):
        raise ValueError("fine scores and complete fine-to-whole mapping must align")
    promoted = torch.full((whole_edge_count,), -torch.inf, dtype=torch.float64)
    absolute = values.abs()
    for fine_index, whole_index in enumerate(mapping):
        promoted[whole_index] = torch.maximum(
            promoted[whole_index], absolute[fine_index]
        )
    if not bool(torch.isfinite(promoted).all()):
        raise RuntimeError("fine promotion left an empty whole-edge group")
    return promoted


def build_selector_score_vectors(
    *,
    node_universe: Sequence[str],
    whole_edges: Sequence[tuple[str, str]],
    clean_output: Any,
    singleton_outputs: Any,
    clean_activations: Any,
    corrupt_activations: Any,
    raw_whole_mass: Any,
    eap_ig5_fine_scores: Any,
    fine_to_whole_indices: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Build the five frozen D or C node-selector vectors."""
    nodes = tuple(node_universe)
    edges = _validated_graph(whole_edges)
    singleton = singleton_output_effect_scores(clean_output, singleton_outputs)
    activation = activation_shift_rms_scores(clean_activations, corrupt_activations)
    raw_mass = _finite_double(raw_whole_mass, label="raw_whole_mass")
    if singleton.numel() != len(nodes) or activation.numel() != len(nodes):
        raise ValueError("singleton and activation vectors must align with nodes")
    raw_nodes = incident_mean_node_scores(
        node_universe=nodes,
        whole_edges=edges,
        whole_edge_scores=raw_mass,
    )
    rank_one = masked_rank_one_endpoint_factor(
        node_universe=nodes,
        edge_universe=edges,
        edge_scores=raw_mass,
    )
    rank_one_nodes = incident_mean_node_scores(
        node_universe=nodes,
        whole_edges=edges,
        whole_edge_scores=rank_one.edge_reconstruction,
    )
    eap_whole = promote_absolute_fine_max(
        fine_scores=eap_ig5_fine_scores,
        fine_to_whole_indices=fine_to_whole_indices,
        whole_edge_count=len(edges),
    )
    eap_nodes = incident_mean_node_scores(
        node_universe=nodes,
        whole_edges=edges,
        whole_edge_scores=eap_whole,
    )
    return {
        "singleton_output_effect": singleton,
        "activation_shift_rms": activation,
        "raw_pair_incident_mean": raw_nodes,
        "rank_one_incident_mean": rank_one_nodes,
        "eap_ig5_incident_mean": eap_nodes,
    }


def deterministic_top_support(
    *, node_universe: Sequence[str], node_scores: Any, support_size: int
) -> tuple[str, ...]:
    """Select a score-descending prefix with canonical-index tie breaking."""
    nodes = tuple(node_universe)
    values = _finite_double(node_scores, label="node_scores")
    if (
        not nodes
        or len(nodes) != len(set(nodes))
        or values.ndim != 1
        or values.numel() != len(nodes)
        or isinstance(support_size, bool)
        or not isinstance(support_size, int)
        or not 0 <= support_size <= len(nodes)
    ):
        raise ValueError("node universe, score vector, and support size must align")
    ranked = sorted(range(len(nodes)), key=lambda index: (-float(values[index]), index))
    chosen = frozenset(ranked[:support_size])
    return tuple(node for index, node in enumerate(nodes) if index in chosen)


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = sorted(range(values.numel()), key=lambda index: float(values[index]))
    ranks = torch.empty(values.numel(), dtype=torch.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and float(values[order[end]]) == float(
            values[order[start]]
        ):
            end += 1
        average = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average
        start = end
    return ranks


def spearman_correlation(left: Any, right: Any) -> float:
    """Tie-aware Spearman correlation; return NaN for a constant vector."""
    left_values = _finite_double(left, label="left")
    right_values = _finite_double(right, label="right")
    if (
        left_values.ndim != 1
        or right_values.shape != left_values.shape
        or left_values.numel() < 2
    ):
        raise ValueError("Spearman inputs must be matching vectors of length >= 2")
    left_rank = _average_ranks(left_values)
    right_rank = _average_ranks(right_values)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = torch.sqrt(
        left_centered.square().sum() * right_centered.square().sum()
    )
    if float(denominator) == 0.0:
        return float("nan")
    return float((left_centered * right_centered).sum() / denominator)


def support_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    """Jaccard similarity for two nonempty unique supports."""
    left_set = frozenset(left)
    right_set = frozenset(right)
    if (
        not left_set
        or not right_set
        or len(left_set) != len(tuple(left))
        or len(right_set) != len(tuple(right))
    ):
        raise ValueError("supports must be nonempty and unique")
    return len(left_set & right_set) / len(left_set | right_set)


def _target_indices(value: Any, *, shape: tuple[int, int]) -> torch.Tensor:
    targets = torch.as_tensor(value, device="cpu")
    if (
        targets.shape != shape
        or targets.dtype == torch.bool
        or targets.is_floating_point()
    ):
        raise ValueError("clean_target_indices must be integer [example, position]")
    return targets.to(torch.long)


def set_faithfulness_metrics(
    *,
    clean_output: Any,
    all_corrupt_into_clean_output: Any,
    selected_corrupt_into_clean_output: Any,
    complement_corrupt_into_clean_output: Any,
    clean_target_indices: Any,
) -> dict[str, Any]:
    """Score selected-set necessity and keep-core sufficiency.

    All output tensors are canonical class-centered non-BOS logits.  The all-U
    corrupt-into-clean intervention is the exact internal-universe endpoint,
    so residual embeddings outside U cannot distort the normalization.
    """
    clean = _finite_double(clean_output, label="clean_output")
    all_corrupt = _finite_double(
        all_corrupt_into_clean_output, label="all_corrupt_into_clean_output"
    )
    selected = _finite_double(
        selected_corrupt_into_clean_output,
        label="selected_corrupt_into_clean_output",
    )
    complement = _finite_double(
        complement_corrupt_into_clean_output,
        label="complement_corrupt_into_clean_output",
    )
    if (
        clean.ndim != 3
        or clean.shape[-1] < 2
        or any(
            value.shape != clean.shape for value in (all_corrupt, selected, complement)
        )
    ):
        raise ValueError("all set outputs must share [example, position, class]")
    targets = _target_indices(clean_target_indices, shape=tuple(clean.shape[:2]))
    if bool((targets < 0).any()) or bool((targets >= clean.shape[-1]).any()):
        raise ValueError("clean target index lies outside the class axis")

    gap = clean - all_corrupt
    reduction_axes = (1, 2)
    per_example_gap = gap.square().sum(dim=reduction_axes)
    necessity_projection_numerator = ((clean - selected) * gap).sum(dim=reduction_axes)
    sufficiency_projection_numerator = ((complement - all_corrupt) * gap).sum(
        dim=reduction_axes
    )
    necessity_residual = (selected - all_corrupt).square().sum(dim=reduction_axes)
    sufficiency_residual = (complement - clean).square().sum(dim=reduction_axes)
    total_gap = float(per_example_gap.sum())

    def per_example_loss(logits: torch.Tensor) -> torch.Tensor:
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)
        return token_losses.mean(dim=1)

    clean_loss = per_example_loss(clean)
    all_corrupt_loss = per_example_loss(all_corrupt)
    selected_loss = per_example_loss(selected)
    complement_loss = per_example_loss(complement)
    loss_gap = float((all_corrupt_loss - clean_loss).sum())
    endpoint_informative = total_gap > 0.0 and loss_gap > 0.0

    def safe_ratio(numerator: torch.Tensor, denominator: float) -> float:
        return (
            float(numerator.sum()) / denominator if denominator > 0.0 else float("nan")
        )

    metrics: dict[str, Any] = {
        "endpoint_informative": endpoint_informative,
        "vector_gap_energy": total_gap,
        "clean_target_loss_gap": loss_gap / clean.shape[0],
        "necessity_projection": safe_ratio(necessity_projection_numerator, total_gap),
        "sufficiency_projection": safe_ratio(
            sufficiency_projection_numerator, total_gap
        ),
        "necessity_residual_error": safe_ratio(necessity_residual, total_gap),
        "sufficiency_residual_error": safe_ratio(sufficiency_residual, total_gap),
        "necessity_loss_recovery": (
            float((selected_loss - clean_loss).sum()) / loss_gap
            if loss_gap > 0.0
            else float("nan")
        ),
        "sufficiency_loss_recovery": (
            float((all_corrupt_loss - complement_loss).sum()) / loss_gap
            if loss_gap > 0.0
            else float("nan")
        ),
        "normalization": "clean_vs_all_U_corrupt_into_clean",
        "clipping": False,
    }
    metrics["per_example"] = {
        "vector_gap_energy": per_example_gap,
        "necessity_projection_numerator": necessity_projection_numerator,
        "sufficiency_projection_numerator": sufficiency_projection_numerator,
        "necessity_residual_energy": necessity_residual,
        "sufficiency_residual_energy": sufficiency_residual,
        "clean_target_loss": clean_loss,
        "all_corrupt_target_loss": all_corrupt_loss,
        "selected_target_loss": selected_loss,
        "complement_target_loss": complement_loss,
    }
    return metrics


def _validated_selector_panel(
    value: Mapping[str, Any], *, node_count: int, label: str
) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or tuple(value) != DETERMINISTIC_METHOD_ORDER:
        raise ValueError(
            f"{label} methods must exactly follow {DETERMINISTIC_METHOD_ORDER!r}"
        )
    vectors: dict[str, torch.Tensor] = {}
    for method in DETERMINISTIC_METHOD_ORDER:
        vector = _finite_double(value[method], label=f"{label} {method}")
        if vector.ndim != 1 or vector.numel() != node_count:
            raise ValueError(f"{label} {method} must align with component_ids")
        vectors[method] = vector
    return vectors


def _set_record_metrics(
    *,
    record: Mapping[str, Any],
    expected_support: tuple[str, ...],
    clean_output: torch.Tensor,
    all_corrupt_output: torch.Tensor,
    targets: torch.Tensor,
    label: str,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"{label} set output must be a mapping")
    try:
        selected_nodes = tuple(record["selected_nodes"])
        selected_output = record["selected_corrupt_into_clean_output"]
        complement_output = record["complement_corrupt_into_clean_output"]
    except KeyError as error:
        raise ValueError(f"{label} set output is incomplete") from error
    if selected_nodes != expected_support:
        raise ValueError(f"{label} selected support differs from the frozen selector")
    return set_faithfulness_metrics(
        clean_output=clean_output,
        all_corrupt_into_clean_output=all_corrupt_output,
        selected_corrupt_into_clean_output=selected_output,
        complement_corrupt_into_clean_output=complement_output,
        clean_target_indices=targets,
    )


def _finite_gate(
    value: float, *, minimum: float | None = None, maximum: float | None = None
) -> bool:
    if not math.isfinite(value):
        return False
    return (minimum is None or value >= minimum) and (
        maximum is None or value <= maximum
    )


def evaluate_sparse_core_case(
    *,
    case_id: int,
    component_ids: Sequence[str],
    whole_edges: Sequence[tuple[str, str]],
    fine_to_whole_indices: Sequence[int],
    discovery_selector_scores: Mapping[str, Any],
    confirmation_selector_scores: Mapping[str, Any],
    discovery_raw_mass: Any,
    confirmation_raw_mass: Any,
    discovery_exact_mass: Any,
    confirmation_exact_mass: Any,
    confirmation_clean_output: Any,
    confirmation_all_corrupt_into_clean_output: Any,
    confirmation_clean_target_indices: Any,
    deterministic_set_outputs: Mapping[str, Mapping[str, Any]],
    random_set_outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate one mandatory case from canonical, already-acquired tensors.

    Discovery scores freeze every support.  Confirmation scores are used only
    for stability.  The public InterpBench ground truth is deliberately absent
    from this API and cannot influence any primary E32.3 result.
    """
    if case_id not in CASE_ORDER:
        raise ValueError(f"case_id must be one of {CASE_ORDER!r}")
    nodes = tuple(component_ids)
    edges = _validated_graph(whole_edges)
    expected = EXPECTED_UNIVERSE_COUNTS[case_id]
    if (
        not nodes
        or len(nodes) != len(set(nodes))
        or any(not isinstance(node, str) or not node for node in nodes)
        or len(nodes) != expected["nodes"]
        or len(edges) != expected["whole_edges"]
    ):
        raise ValueError("component/whole-edge universe differs from frozen counts")
    node_set = frozenset(nodes)
    if any(
        source not in node_set or target not in node_set for source, target in edges
    ):
        raise ValueError("whole-edge endpoints escape component_ids")
    fine_mapping = tuple(fine_to_whole_indices)
    if (
        len(fine_mapping) != expected["fine_edges"]
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(edges)
            for index in fine_mapping
        )
        or set(fine_mapping) != set(range(len(edges)))
    ):
        raise ValueError("fine-to-whole mapping differs from frozen exact universe")

    discovery_scores = _validated_selector_panel(
        discovery_selector_scores, node_count=len(nodes), label="discovery"
    )
    confirmation_scores = _validated_selector_panel(
        confirmation_selector_scores, node_count=len(nodes), label="confirmation"
    )
    support_size = confirmatory_support_size(len(nodes))
    selected_by_method = {
        method: deterministic_top_support(
            node_universe=nodes,
            node_scores=discovery_scores[method],
            support_size=support_size,
        )
        for method in DETERMINISTIC_METHOD_ORDER
    }
    selected = selected_by_method[PRIMARY_METHOD]
    confirmation_selected = deterministic_top_support(
        node_universe=nodes,
        node_scores=confirmation_scores[PRIMARY_METHOD],
        support_size=support_size,
    )

    d_raw = _finite_double(discovery_raw_mass, label="discovery_raw_mass")
    c_raw = _finite_double(confirmation_raw_mass, label="confirmation_raw_mass")
    d_exact = _finite_double(discovery_exact_mass, label="discovery_exact_mass")
    c_exact = _finite_double(confirmation_exact_mass, label="confirmation_exact_mass")
    if (
        d_raw.ndim != 1
        or c_raw.shape != d_raw.shape
        or d_raw.numel() != len(edges)
        or d_exact.ndim != 1
        or c_exact.shape != d_exact.shape
        or d_exact.numel() != len(fine_mapping)
        or any(bool((value < 0.0).any()) for value in (d_raw, c_raw, d_exact, c_exact))
    ):
        raise ValueError("raw/exact mass vectors must align and be nonnegative")

    mass_informative_by_vector = {
        "discovery_raw": float(d_raw.sum()) > 0.0,
        "confirmation_raw": float(c_raw.sum()) > 0.0,
        "discovery_exact": float(d_exact.sum()) > 0.0,
        "confirmation_exact": float(c_exact.sum()) > 0.0,
    }
    mass_informative = all(mass_informative_by_vector.values())

    def retention_or_nan(mass: torch.Tensor, *, fine: bool = False) -> float:
        if not float(mass.sum()) > 0.0:
            return float("nan")
        return interaction_mass_retention(
            item_mass=mass,
            selected_nodes=selected,
            whole_edges=edges,
            fine_to_whole_indices=fine_mapping if fine else None,
        )

    raw_retention_d = retention_or_nan(d_raw)
    raw_retention_c = retention_or_nan(c_raw)
    exact_retention_d = retention_or_nan(d_exact, fine=True)
    exact_retention_c = retention_or_nan(c_exact, fine=True)
    selected_set = frozenset(selected)
    induced_whole = tuple(
        index
        for index, (source, target) in enumerate(edges)
        if source in selected_set and target in selected_set
    )
    induced_fine = tuple(
        fine_index
        for fine_index, whole_index in enumerate(fine_mapping)
        if whole_index in frozenset(induced_whole)
    )

    clean = _finite_double(confirmation_clean_output, label="confirmation clean")
    all_corrupt = _finite_double(
        confirmation_all_corrupt_into_clean_output,
        label="confirmation all-corrupt",
    )
    if clean.ndim != 3 or all_corrupt.shape != clean.shape:
        raise ValueError("confirmation endpoints must share [example, position, class]")
    targets = _target_indices(
        confirmation_clean_target_indices, shape=tuple(clean.shape[:2])
    )
    if (
        not isinstance(deterministic_set_outputs, Mapping)
        or tuple(deterministic_set_outputs) != DETERMINISTIC_METHOD_ORDER
    ):
        raise ValueError(
            "deterministic set outputs must follow the frozen method order"
        )
    deterministic_metrics = {
        method: _set_record_metrics(
            record=deterministic_set_outputs[method],
            expected_support=selected_by_method[method],
            clean_output=clean,
            all_corrupt_output=all_corrupt,
            targets=targets,
            label=method,
        )
        for method in DETERMINISTIC_METHOD_ORDER
    }

    random_records = tuple(random_set_outputs)
    if len(random_records) != RANDOM_CONTROL_REPLICATES:
        raise ValueError(
            f"random_set_outputs must contain {RANDOM_CONTROL_REPLICATES} records"
        )
    random_metrics = []
    for replicate, record in enumerate(random_records):
        if record.get("replicate") != replicate:
            raise ValueError("random control records must follow replicate order")
        expected_random = deterministic_random_support(
            case_id=case_id,
            component_ids=nodes,
            support_size=support_size,
            replicate=replicate,
        )
        random_metrics.append(
            _set_record_metrics(
                record=record,
                expected_support=expected_random,
                clean_output=clean,
                all_corrupt_output=all_corrupt,
                targets=targets,
                label=f"random replicate {replicate}",
            )
        )

    def joint_loss(metrics: Mapping[str, Any]) -> float:
        return min(
            float(metrics["necessity_loss_recovery"]),
            float(metrics["sufficiency_loss_recovery"]),
        )

    primary_metrics = deterministic_metrics[PRIMARY_METHOD]
    primary_joint = joint_loss(primary_metrics)
    baseline_joint = {
        method: joint_loss(deterministic_metrics[method])
        for method in DETERMINISTIC_METHOD_ORDER
        if method != PRIMARY_METHOD
    }
    best_baseline_method = max(baseline_joint, key=baseline_joint.get)  # type: ignore[arg-type]
    best_baseline = baseline_joint[best_baseline_method]
    random_joint = torch.tensor(
        [joint_loss(metrics) for metrics in random_metrics], dtype=torch.float64
    )
    random_95 = float(torch.quantile(random_joint, 0.95))

    stability_values = {
        "singleton_spearman": spearman_correlation(
            discovery_scores[PRIMARY_METHOD], confirmation_scores[PRIMARY_METHOD]
        ),
        "singleton_top_half_jaccard": support_jaccard(selected, confirmation_selected),
        "raw_mass_spearman": (
            spearman_correlation(d_raw, c_raw)
            if mass_informative_by_vector["discovery_raw"]
            and mass_informative_by_vector["confirmation_raw"]
            else float("nan")
        ),
        "exact_mass_spearman": (
            spearman_correlation(d_exact, c_exact)
            if mass_informative_by_vector["discovery_exact"]
            and mass_informative_by_vector["confirmation_exact"]
            else float("nan")
        ),
    }
    structure = {
        "node_count": len(nodes),
        "support_size": support_size,
        "support_fraction": support_size / len(nodes),
        "induced_whole_edge_count": len(induced_whole),
        "induced_whole_edge_fraction": len(induced_whole) / len(edges),
        "induced_fine_edge_count": len(induced_fine),
        "induced_fine_edge_fraction": len(induced_fine) / len(fine_mapping),
    }
    retention_values = {
        "discovery_raw": raw_retention_d,
        "confirmation_raw": raw_retention_c,
        "raw_discovery_confirmation_gap": abs(raw_retention_d - raw_retention_c),
        "discovery_exact": exact_retention_d,
        "confirmation_exact": exact_retention_c,
        "exact_discovery_confirmation_gap": abs(exact_retention_d - exact_retention_c),
    }

    def finite_or_none(value: float) -> float | None:
        return value if math.isfinite(value) else None

    stability = {key: finite_or_none(value) for key, value in stability_values.items()}
    retention = {key: finite_or_none(value) for key, value in retention_values.items()}

    screen_checks = {
        "mass_informative": mass_informative,
        "induced_whole_edge_fraction": _finite_gate(
            structure["induced_whole_edge_fraction"],
            maximum=SCREEN_GATES["maximum_induced_whole_edge_fraction"],
        ),
        "induced_fine_edge_fraction": _finite_gate(
            structure["induced_fine_edge_fraction"],
            maximum=SCREEN_GATES["maximum_induced_fine_edge_fraction"],
        ),
        "confirmation_raw_mass_retention": _finite_gate(
            raw_retention_c,
            minimum=SCREEN_GATES["minimum_confirmation_raw_mass_retention"],
        ),
        "confirmation_exact_mass_retention": _finite_gate(
            exact_retention_c,
            minimum=SCREEN_GATES["minimum_confirmation_exact_mass_retention"],
        ),
        "raw_retention_gap": _finite_gate(
            retention_values["raw_discovery_confirmation_gap"],
            maximum=SCREEN_GATES["maximum_discovery_confirmation_retention_gap"],
        ),
        "exact_retention_gap": _finite_gate(
            retention_values["exact_discovery_confirmation_gap"],
            maximum=SCREEN_GATES["maximum_discovery_confirmation_retention_gap"],
        ),
        "singleton_spearman": _finite_gate(
            stability_values["singleton_spearman"],
            minimum=SCREEN_GATES["minimum_singleton_spearman"],
        ),
        "singleton_top_half_jaccard": _finite_gate(
            stability_values["singleton_top_half_jaccard"],
            minimum=SCREEN_GATES["minimum_singleton_top_half_jaccard"],
        ),
        "raw_mass_spearman": _finite_gate(
            stability_values["raw_mass_spearman"],
            minimum=SCREEN_GATES["minimum_raw_mass_spearman"],
        ),
        "exact_mass_spearman": _finite_gate(
            stability_values["exact_mass_spearman"],
            minimum=SCREEN_GATES["minimum_exact_mass_spearman"],
        ),
    }
    faithfulness_checks = {
        "endpoint_informative": bool(primary_metrics["endpoint_informative"]),
        "necessity_projection": _finite_gate(
            float(primary_metrics["necessity_projection"]),
            minimum=FAITHFULNESS_GATES["minimum_necessity_projection"],
        ),
        "sufficiency_projection": _finite_gate(
            float(primary_metrics["sufficiency_projection"]),
            minimum=FAITHFULNESS_GATES["minimum_sufficiency_projection"],
        ),
        "necessity_residual_error": _finite_gate(
            float(primary_metrics["necessity_residual_error"]),
            maximum=FAITHFULNESS_GATES["maximum_necessity_residual_error"],
        ),
        "sufficiency_residual_error": _finite_gate(
            float(primary_metrics["sufficiency_residual_error"]),
            maximum=FAITHFULNESS_GATES["maximum_sufficiency_residual_error"],
        ),
        "necessity_loss_recovery": _finite_gate(
            float(primary_metrics["necessity_loss_recovery"]),
            minimum=FAITHFULNESS_GATES["minimum_necessity_loss_recovery"],
        ),
        "sufficiency_loss_recovery": _finite_gate(
            float(primary_metrics["sufficiency_loss_recovery"]),
            minimum=FAITHFULNESS_GATES["minimum_sufficiency_loss_recovery"],
        ),
    }
    selector_checks = {
        "margin_over_random_95th_percentile": _finite_gate(
            primary_joint - random_95,
            minimum=SELECTOR_GATES["minimum_margin_over_random_95th_percentile"],
        ),
        "noninferior_to_best_deterministic_baseline": _finite_gate(
            primary_joint - best_baseline,
            minimum=-SELECTOR_GATES["maximum_deficit_to_best_deterministic_baseline"],
        ),
    }
    gates = {
        "screen_checks": screen_checks,
        "screen_pass": all(screen_checks.values()),
        "faithfulness_checks": faithfulness_checks,
        "faithfulness_pass": all(faithfulness_checks.values()),
        "selector_checks": selector_checks,
        "selector_pass": all(selector_checks.values()),
    }
    gates["case_pass"] = bool(
        gates["screen_pass"] and gates["faithfulness_pass"] and gates["selector_pass"]
    )

    def vector_tuple(value: torch.Tensor) -> tuple[float, ...]:
        return tuple(float(item) for item in value)

    return {
        "schema_version": "e32_3_sparse_core_case_score_v1",
        "case_id": case_id,
        "confirmatory_support_fraction": 0.5,
        "selected_nodes": selected,
        "selected_nodes_by_method": selected_by_method,
        "mass_informative": mass_informative,
        "mass_informative_by_vector": mass_informative_by_vector,
        "structure": structure,
        "retention": retention,
        "stability": stability,
        "set_faithfulness": deterministic_metrics,
        "random_controls": {
            "replicate_count": RANDOM_CONTROL_REPLICATES,
            "joint_loss_recovery": vector_tuple(random_joint),
            "joint_loss_recovery_95th_percentile": random_95,
        },
        "selector_comparison": {
            "primary_method": PRIMARY_METHOD,
            "primary_joint_loss_recovery": primary_joint,
            "deterministic_baseline_joint_loss_recovery": baseline_joint,
            "best_deterministic_baseline_method": best_baseline_method,
            "best_deterministic_baseline_joint_loss_recovery": best_baseline,
            "margin_over_random_95th_percentile": primary_joint - random_95,
            "margin_over_best_deterministic_baseline": (primary_joint - best_baseline),
        },
        "full_vectors": {
            "discovery_selector_scores": {
                method: vector_tuple(discovery_scores[method])
                for method in DETERMINISTIC_METHOD_ORDER
            },
            "confirmation_selector_scores": {
                method: vector_tuple(confirmation_scores[method])
                for method in DETERMINISTIC_METHOD_ORDER
            },
            "discovery_raw_mass": vector_tuple(d_raw),
            "confirmation_raw_mass": vector_tuple(c_raw),
            "discovery_exact_mass": vector_tuple(d_exact),
            "confirmation_exact_mass": vector_tuple(c_exact),
        },
        "gates": gates,
        "ground_truth_used_for_cohort_selection": True,
        "ground_truth_used_for_within_case_selection_or_primary_scoring": False,
    }


def evaluate_singleton_support_curve(
    *,
    component_ids: Sequence[str],
    discovery_singleton_scores: Any,
    clean_output: Any,
    all_corrupt_into_clean_output: Any,
    clean_target_indices: Any,
    set_outputs_by_support_size: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Score the descriptive singleton-only 25/50/75-percent budget curve."""
    nodes = tuple(component_ids)
    scores = _finite_double(
        discovery_singleton_scores, label="discovery_singleton_scores"
    )
    if (
        not nodes
        or len(nodes) != len(set(nodes))
        or scores.ndim != 1
        or scores.numel() != len(nodes)
    ):
        raise ValueError("component IDs and singleton scores must align")
    frozen_sizes = support_sizes(len(nodes))[1:4]
    if (
        not isinstance(set_outputs_by_support_size, Mapping)
        or tuple(set_outputs_by_support_size) != frozen_sizes
    ):
        raise ValueError("singleton curve must contain ordered 25/50/75% supports")
    clean = _finite_double(clean_output, label="clean_output")
    all_corrupt = _finite_double(
        all_corrupt_into_clean_output, label="all_corrupt_into_clean_output"
    )
    if clean.ndim != 3 or all_corrupt.shape != clean.shape:
        raise ValueError("curve endpoints must share [example, position, class]")
    targets = _target_indices(clean_target_indices, shape=tuple(clean.shape[:2]))
    metrics = []
    selected_supports = []
    for size in frozen_sizes:
        support = deterministic_top_support(
            node_universe=nodes, node_scores=scores, support_size=size
        )
        selected_supports.append(support)
        metrics.append(
            _set_record_metrics(
                record=set_outputs_by_support_size[size],
                expected_support=support,
                clean_output=clean,
                all_corrupt_output=all_corrupt,
                targets=targets,
                label=f"singleton support {size}",
            )
        )
    return {
        "schema_version": "e32_3_singleton_support_curve_v1",
        "support_sizes": frozen_sizes,
        "support_fractions": tuple(size / len(nodes) for size in frozen_sizes),
        "selected_supports": tuple(selected_supports),
        "necessity_loss_recovery": tuple(
            float(metric["necessity_loss_recovery"]) for metric in metrics
        ),
        "sufficiency_loss_recovery": tuple(
            float(metric["sufficiency_loss_recovery"]) for metric in metrics
        ),
        "joint_loss_recovery": tuple(
            min(
                float(metric["necessity_loss_recovery"]),
                float(metric["sufficiency_loss_recovery"]),
            )
            for metric in metrics
        ),
        "metrics": tuple(metrics),
        "confirmatory_index": 1,
        "curve_status": "descriptive_except_predeclared_50_percent_point",
    }


def evaluate_panel_verdict(
    case_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen all-three-cases conjunction without majority rescue."""
    expected_keys = tuple(str(case_id) for case_id in CASE_ORDER)
    if not isinstance(case_results, Mapping) or tuple(case_results) != expected_keys:
        raise ValueError(f"case_results must follow mandatory order {expected_keys!r}")
    sparse_case_passes = {}
    selector_case_passes = {}
    for case_id in CASE_ORDER:
        result = case_results[str(case_id)]
        if result.get("case_id") != case_id or not isinstance(
            result.get("gates"), Mapping
        ):
            raise ValueError("case result identity or gates changed")
        gates = result["gates"]
        required = {
            "screen_pass",
            "faithfulness_pass",
            "selector_pass",
            "case_pass",
        }
        if not required.issubset(gates) or any(
            not isinstance(gates[key], bool) for key in required
        ):
            raise ValueError("case gate booleans are incomplete")
        sparse_case_passes[str(case_id)] = bool(
            gates["screen_pass"] and gates["faithfulness_pass"]
        )
        selector_case_passes[str(case_id)] = bool(gates["case_pass"])
    return {
        "schema_version": "e32_3_panel_verdict_v1",
        "case_order": CASE_ORDER,
        "sparse_core_case_passes": sparse_case_passes,
        "singleton_selector_case_passes": selector_case_passes,
        "sparse_core_generalization_pass": all(sparse_case_passes.values()),
        "singleton_selector_generalization_pass": all(selector_case_passes.values()),
        "majority_rule_allowed": False,
        "ground_truth_used_for_cohort_selection": True,
        "ground_truth_used_for_within_case_selection_or_primary_scoring": False,
        "ground_truth_used_for_verdict": False,
    }


__all__ = [
    "activation_shift_rms_scores",
    "build_selector_score_vectors",
    "deterministic_top_support",
    "evaluate_panel_verdict",
    "evaluate_singleton_support_curve",
    "evaluate_sparse_core_case",
    "exact_fine_interaction_mass",
    "incident_mean_node_scores",
    "interaction_mass_retention",
    "promote_absolute_fine_max",
    "raw_pair_interaction_mass",
    "set_faithfulness_metrics",
    "singleton_output_effect_scores",
    "spearman_correlation",
    "support_jaccard",
]
