# ruff: noqa: EXE002
"""Pure scoring and frozen gates for E32.2 directed edge validation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from experiments.e32_interpbench.analysis import build_atlas_method_scores
from experiments.e32_interpbench.e32_2_protocol import (
    EAP_VARIANTS,
    FULL_UNIVERSE_FLOOR,
    SCREEN_GATES,
    SUPERIORITY_GATES,
    USEFUL_GATES,
)
from experiments.e32_interpbench.protocol import AdaptedScores, Edge
from experiments.e32_interpbench.scoring import binary_ranking_metrics


def _finite_double(value: Any, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must be finite")
    return tensor


@dataclass(frozen=True)
class BidirectionalFineScores:
    """Per-context directional progress and the final fine-edge score."""

    sufficiency_by_fine_context: torch.Tensor
    necessity_by_fine_context: torch.Tensor
    bidirectional_by_fine_context: torch.Tensor
    fine_scores: torch.Tensor
    informative_context_mask: torch.Tensor


def bidirectional_fine_replacement_scores(
    *,
    clean_output: Any,
    corrupt_output: Any,
    clean_into_corrupt_outputs: Any,
    corrupt_into_clean_outputs: Any,
) -> BidirectionalFineScores:
    """Score exact directed replacement separately for every fine edge.

    Inputs are class-centered non-BOS logits.  The unambiguous intervention
    names describe source activation then base run:

    - ``clean_into_corrupt`` is sufficiency: insert the clean source edge into
      the corrupt-input computation;
    - ``corrupt_into_clean`` is necessity: insert the corrupt source edge into
      the clean-input computation.

    For context ``c`` and fine edge ``f``, with
    ``D_c = Y_clean,c - Y_corrupt,c``::

        S_cf = <Y_clean_into_corrupt,cf - Y_corrupt,c, D_c> / ||D_c||^2
        N_cf = <Y_clean,c - Y_corrupt_into_clean,cf, D_c> / ||D_c||^2
        b_cf = sqrt(relu(S_cf) * relu(N_cf))
        B_f  = mean_c b_cf

    An exactly zero clean/corrupt gap carries no directional information.  It
    contributes registered zeros to ``S``, ``N``, and the all-context mean;
    such contexts are reported by ``informative_context_mask`` rather than
    silently dropped.
    """
    clean = _finite_double(clean_output, label="clean_output")
    corrupt = _finite_double(corrupt_output, label="corrupt_output")
    sufficiency_outputs = _finite_double(
        clean_into_corrupt_outputs, label="clean_into_corrupt_outputs"
    )
    necessity_outputs = _finite_double(
        corrupt_into_clean_outputs, label="corrupt_into_clean_outputs"
    )
    if clean.ndim != 3 or corrupt.shape != clean.shape:
        raise ValueError(
            "clean_output and corrupt_output must have the same shape "
            "[context, non_bos_position, class]"
        )
    expected_replacement_shape = (sufficiency_outputs.shape[0], *clean.shape)
    if (
        sufficiency_outputs.ndim != 4
        or necessity_outputs.shape != sufficiency_outputs.shape
        or tuple(sufficiency_outputs.shape) != expected_replacement_shape
        or sufficiency_outputs.shape[0] < 1
    ):
        raise ValueError(
            "replacement outputs must share shape "
            "[fine_edge, context, non_bos_position, class]"
        )

    gap = clean - corrupt
    denominator = gap.square().sum(dim=(1, 2))
    informative = denominator > 0.0
    safe_denominator = torch.where(
        informative, denominator, torch.ones_like(denominator)
    )
    sufficiency_numerator = (
        (sufficiency_outputs - corrupt.unsqueeze(0)) * gap.unsqueeze(0)
    ).sum(dim=(2, 3))
    necessity_numerator = (
        (clean.unsqueeze(0) - necessity_outputs) * gap.unsqueeze(0)
    ).sum(dim=(2, 3))
    sufficiency = sufficiency_numerator / safe_denominator.unsqueeze(0)
    necessity = necessity_numerator / safe_denominator.unsqueeze(0)
    zero_gap = ~informative.unsqueeze(0)
    sufficiency = torch.where(zero_gap, torch.zeros_like(sufficiency), sufficiency)
    necessity = torch.where(zero_gap, torch.zeros_like(necessity), necessity)
    bidirectional = torch.sqrt(torch.relu(sufficiency) * torch.relu(necessity))
    return BidirectionalFineScores(
        sufficiency_by_fine_context=sufficiency,
        necessity_by_fine_context=necessity,
        bidirectional_by_fine_context=bidirectional,
        fine_scores=bidirectional.mean(dim=1),
        informative_context_mask=informative,
    )


def promote_fine_scores_max(
    *,
    fine_scores: Any,
    fine_to_whole_indices: Sequence[int],
    whole_edge_count: int,
) -> torch.Tensor:
    """Max-promote separate Q/K/V or MLP fine scores to whole edges."""
    values = _finite_double(fine_scores, label="fine_scores")
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("fine_scores must be a nonempty vector")
    if isinstance(whole_edge_count, bool) or not isinstance(whole_edge_count, int):
        raise TypeError("whole_edge_count must be an integer")
    if whole_edge_count < 1:
        raise ValueError("whole_edge_count must be positive")
    indices = tuple(fine_to_whole_indices)
    if len(indices) != values.numel() or any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < whole_edge_count
        for index in indices
    ):
        raise ValueError(
            "fine_to_whole_indices must align with scores and lie in the whole universe"
        )
    if set(indices) != set(range(whole_edge_count)):
        raise ValueError(
            "fine-to-whole promotion must have complete whole-edge coverage"
        )

    promoted = torch.full((whole_edge_count,), -torch.inf, dtype=torch.float64)
    for fine_index, whole_index in enumerate(indices):
        promoted[whole_index] = torch.maximum(promoted[whole_index], values[fine_index])
    if not bool(torch.isfinite(promoted).all()):
        raise RuntimeError("fine-to-whole promotion left an uncovered edge")
    return promoted


def _validated_edges(value: Sequence[Edge], *, label: str) -> tuple[Edge, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of directed edges")
    edges = tuple(tuple(edge) for edge in value)
    if (
        not edges
        or len(edges) != len(set(edges))
        or any(len(edge) != 2 for edge in edges)
    ):
        raise ValueError(f"{label} must be nonempty unique directed edges")
    return edges


def embed_shortlist_scores(
    *,
    full_edge_universe: Sequence[Edge],
    shortlisted_edges: Sequence[Edge],
    shortlisted_scores: Any,
    floor: float = FULL_UNIVERSE_FLOOR,
) -> torch.Tensor:
    """Embed shortlist scores into the full universe at a registered floor."""
    full_edges = _validated_edges(full_edge_universe, label="full_edge_universe")
    shortlist = _validated_edges(shortlisted_edges, label="shortlisted_edges")
    if not set(shortlist).issubset(full_edges):
        raise ValueError("shortlisted_edges must be a subset of full_edge_universe")
    values = _finite_double(shortlisted_scores, label="shortlisted_scores")
    if values.ndim != 1 or values.numel() != len(shortlist):
        raise ValueError("shortlisted_scores must align exactly with shortlisted_edges")
    if not math.isfinite(floor):
        raise ValueError("floor must be finite")
    if bool((values <= floor).any()):
        raise ValueError("shortlisted scores must lie above the full-universe floor")

    result = torch.full((len(full_edges),), float(floor), dtype=torch.float64)
    full_index = {edge: index for index, edge in enumerate(full_edges)}
    for edge, score in zip(shortlist, values, strict=True):
        result[full_index[edge]] = score
    return result


def best_eap_variant(
    variant_metrics: Mapping[str, Mapping[str, float | int]],
) -> tuple[str, dict[str, float | int]]:
    """Return the strongest frozen C EAP variant as a conservative comparator.

    This post-outcome maximum is intentionally favorable to the comparator and
    does not license a model-selection or EAP superiority claim.
    """
    if set(variant_metrics) != set(EAP_VARIANTS):
        raise ValueError(f"variant_metrics must contain exactly {EAP_VARIANTS!r}")
    validated: dict[str, dict[str, float | int]] = {}
    for name in EAP_VARIANTS:
        metrics = variant_metrics[name]
        try:
            auprc = float(metrics["auprc"])
            true_positives = int(metrics["true_positives_at_gt_k"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("each EAP variant needs AUPRC and TP@GT-K") from error
        if not math.isfinite(auprc) or not 0.0 <= auprc <= 1.0:
            raise ValueError("EAP AUPRC must lie in [0, 1]")
        if true_positives < 0:
            raise ValueError("EAP true positives must be nonnegative")
        validated[name] = {
            "auprc": auprc,
            "true_positives_at_gt_k": true_positives,
        }
    best_name = max(EAP_VARIANTS, key=lambda name: validated[name]["auprc"])
    return best_name, validated[best_name]


def direct_vs_transitive_auroc(
    *,
    scores: Any,
    edge_universe: Sequence[Edge],
    direct_edges: Sequence[Edge],
    transitive_non_edges: Sequence[Edge],
) -> float:
    """Measure direct-edge separation from the hard transitive negatives only."""
    edges = _validated_edges(edge_universe, label="edge_universe")
    direct = _validated_edges(direct_edges, label="direct_edges")
    transitive = _validated_edges(transitive_non_edges, label="transitive_non_edges")
    if set(direct) & set(transitive):
        raise ValueError("direct_edges and transitive_non_edges must be disjoint")
    if not (set(direct) | set(transitive)).issubset(edges):
        raise ValueError("direct and transitive edges must lie in edge_universe")
    values = _finite_double(scores, label="scores")
    if values.ndim != 1 or values.numel() != len(edges):
        raise ValueError("scores must align exactly with edge_universe")
    edge_index = {edge: index for index, edge in enumerate(edges)}
    selected = (*direct, *transitive)
    selected_scores = [float(values[edge_index[edge]]) for edge in selected]
    labels = [True] * len(direct) + [False] * len(transitive)
    return float(binary_ranking_metrics(scores=selected_scores, labels=labels)["auroc"])


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = sorted(
        range(values.numel()), key=lambda index: (float(values[index]), index)
    )
    ranks = torch.empty(values.numel(), dtype=torch.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def spearman_correlation(left: Any, right: Any) -> float:
    """Tie-aware Spearman with an explicit constant-vector convention."""
    x = _finite_double(left, label="left")
    y = _finite_double(right, label="right")
    if x.ndim != 1 or y.shape != x.shape or x.numel() < 2:
        raise ValueError("left and right must be matching vectors of length at least 2")
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    x_norm = float(torch.linalg.vector_norm(x_centered))
    y_norm = float(torch.linalg.vector_norm(y_centered))
    if x_norm == 0.0 and y_norm == 0.0:
        return 1.0
    if x_norm == 0.0 or y_norm == 0.0:
        return 0.0
    return float(torch.dot(x_centered, y_centered) / (x_norm * y_norm))


def build_exhaustive_baseline_scores(
    *,
    node_universe: Sequence[str],
    edge_universe: Sequence[Edge],
    clean_output: Any,
    singleton_outputs: Any,
    pair_interactions: Any,
) -> dict[str, AdaptedScores]:
    """Return the frozen full raw-J, rank-one, and singleton baselines."""
    panel = build_atlas_method_scores(
        node_universe=node_universe,
        edge_universe=edge_universe,
        clean_output=clean_output,
        singleton_outputs=singleton_outputs,
        pair_interactions=pair_interactions,
    )
    return {
        name: panel[name]
        for name in (
            "raw_pair_interaction",
            "rank_one_node_factor",
            "singleton_endpoint_product",
        )
    }


def _unit_interval(value: float, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return number


def _true_positive_count(
    value: int, *, ground_truth_edge_count: int, label: str
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if not 0 <= value <= ground_truth_edge_count:
        raise ValueError(f"{label} must lie between zero and GT edge count")
    return value


def evaluate_case_gates(
    *,
    ground_truth_edge_count: int,
    direct_edge_retention: float,
    induced_whole_edge_fraction: float,
    induced_fine_edge_fraction: float,
    operational_model_example_savings: float,
    confirmation_auprc: float,
    confirmation_true_positives_at_gt_k: int,
    exhaustive_exact_auprc: float,
    exhaustive_exact_true_positives_at_gt_k: int,
    best_eap_auprc: float,
    best_eap_true_positives_at_gt_k: int,
    raw_pair_auprc: float,
    raw_pair_true_positives_at_gt_k: int,
    other_exhaustive_comparator_auprcs: Sequence[float],
    other_exhaustive_comparator_true_positives: Sequence[int],
    direct_vs_transitive_auroc: float,
    node_gated_raw_direct_vs_transitive_auroc: float,
    discovery_confirmation_spearman: float,
) -> dict[str, Any]:
    """Apply useful and stronger-superiority gates without adaptive choices."""
    if (
        isinstance(ground_truth_edge_count, bool)
        or not isinstance(ground_truth_edge_count, int)
        or ground_truth_edge_count < 1
    ):
        raise ValueError("ground_truth_edge_count must be a positive integer")
    retention = _unit_interval(direct_edge_retention, label="direct_edge_retention")
    whole_fraction = _unit_interval(
        induced_whole_edge_fraction, label="induced_whole_edge_fraction"
    )
    fine_fraction = _unit_interval(
        induced_fine_edge_fraction, label="induced_fine_edge_fraction"
    )
    savings = _unit_interval(
        operational_model_example_savings,
        label="operational_model_example_savings",
    )
    primary_ap = _unit_interval(confirmation_auprc, label="confirmation_auprc")
    exhaustive_ap = _unit_interval(
        exhaustive_exact_auprc, label="exhaustive_exact_auprc"
    )
    eap_ap = _unit_interval(best_eap_auprc, label="best_eap_auprc")
    raw_ap = _unit_interval(raw_pair_auprc, label="raw_pair_auprc")
    direct_transitive = _unit_interval(
        direct_vs_transitive_auroc, label="direct_vs_transitive_auroc"
    )
    raw_direct_transitive = _unit_interval(
        node_gated_raw_direct_vs_transitive_auroc,
        label="node_gated_raw_direct_vs_transitive_auroc",
    )
    if not math.isfinite(discovery_confirmation_spearman) or not (
        -1.0 <= discovery_confirmation_spearman <= 1.0
    ):
        raise ValueError("discovery_confirmation_spearman must lie in [-1, 1]")

    primary_tp = _true_positive_count(
        confirmation_true_positives_at_gt_k,
        ground_truth_edge_count=ground_truth_edge_count,
        label="confirmation_true_positives_at_gt_k",
    )
    comparator_aps = [exhaustive_ap, eap_ap, raw_ap]
    comparator_tps = [
        _true_positive_count(
            exhaustive_exact_true_positives_at_gt_k,
            ground_truth_edge_count=ground_truth_edge_count,
            label="exhaustive_exact_true_positives_at_gt_k",
        ),
        _true_positive_count(
            best_eap_true_positives_at_gt_k,
            ground_truth_edge_count=ground_truth_edge_count,
            label="best_eap_true_positives_at_gt_k",
        ),
        _true_positive_count(
            raw_pair_true_positives_at_gt_k,
            ground_truth_edge_count=ground_truth_edge_count,
            label="raw_pair_true_positives_at_gt_k",
        ),
    ]
    if len(other_exhaustive_comparator_auprcs) != len(
        other_exhaustive_comparator_true_positives
    ):
        raise ValueError("other comparator AUPRC and TP sequences must align")
    for index, (auprc, true_positives) in enumerate(
        zip(
            other_exhaustive_comparator_auprcs,
            other_exhaustive_comparator_true_positives,
            strict=True,
        )
    ):
        comparator_aps.append(
            _unit_interval(auprc, label=f"other comparator {index} AUPRC")
        )
        comparator_tps.append(
            _true_positive_count(
                true_positives,
                ground_truth_edge_count=ground_truth_edge_count,
                label=f"other comparator {index} TP",
            )
        )

    screen_checks = {
        "direct_edge_retention": retention
        >= SCREEN_GATES["minimum_direct_edge_retention"],
        "induced_whole_edge_fraction": whole_fraction
        <= SCREEN_GATES["maximum_induced_whole_edge_fraction"],
        "induced_fine_edge_fraction": fine_fraction
        <= SCREEN_GATES["maximum_induced_fine_edge_fraction"],
        "operational_model_example_savings": savings
        >= SCREEN_GATES["minimum_operational_model_example_savings"],
    }
    useful_checks = {
        "confirmation_auprc": primary_ap >= USEFUL_GATES["minimum_confirmation_auprc"],
        "perfect_true_positives_at_gt_k": primary_tp == ground_truth_edge_count,
        "noninferior_to_exhaustive_exact": primary_ap
        >= exhaustive_ap - USEFUL_GATES["maximum_auprc_deficit_to_exhaustive_exact"],
        "noninferior_to_best_eap_variant": primary_ap
        >= eap_ap - USEFUL_GATES["maximum_auprc_deficit_to_best_eap_variant"],
        "auprc_margin_over_raw_pair": primary_ap
        >= raw_ap + USEFUL_GATES["minimum_auprc_margin_over_raw_pair"],
        "direct_vs_transitive_margin": direct_transitive
        >= raw_direct_transitive
        + USEFUL_GATES["minimum_direct_vs_transitive_auroc_margin_over_node_gated_raw"],
        "discovery_confirmation_spearman": discovery_confirmation_spearman
        >= USEFUL_GATES["minimum_induced_discovery_confirmation_spearman"],
    }
    required_superiority_tp = min(
        ground_truth_edge_count,
        max(comparator_tps) + SUPERIORITY_GATES["minimum_true_positive_improvement"],
    )
    superiority_checks = {
        "auprc_margin_over_every_comparator": all(
            primary_ap
            >= comparator_ap
            + SUPERIORITY_GATES["minimum_auprc_margin_over_every_comparator"]
            for comparator_ap in comparator_aps
        ),
        "tp_improvement_capped_at_perfect": primary_tp >= required_superiority_tp,
    }
    screen_pass = all(screen_checks.values())
    useful_pass = screen_pass and all(useful_checks.values())
    return {
        "screen_checks": screen_checks,
        "screen_pass": screen_pass,
        "useful_checks": useful_checks,
        "useful_pass": useful_pass,
        "superiority_checks": superiority_checks,
        "superiority_required_true_positives_at_gt_k": required_superiority_tp,
        "superiority_pass": useful_pass and all(superiority_checks.values()),
    }


__all__ = [
    "BidirectionalFineScores",
    "best_eap_variant",
    "bidirectional_fine_replacement_scores",
    "build_exhaustive_baseline_scores",
    "direct_vs_transitive_auroc",
    "embed_shortlist_scores",
    "evaluate_case_gates",
    "promote_fine_scores_max",
    "spearman_correlation",
]
