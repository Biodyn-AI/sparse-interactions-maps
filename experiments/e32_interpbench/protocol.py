# ruff: noqa: EXE002
"""Small, model-free protocol types and claim boundaries for E32.

Official InterpBench ground truth is a directed edge list.  Everything else in
this package is an E32 evaluation convention layered on top of that edge list.
Keeping that distinction in executable metadata prevents a localization score
or intervention curve from being mistaken for additional benchmark truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Edge = tuple[str, str]

CLAIM_BOUNDARIES = {
    "official_ground_truth_is_directed_edge_support_only": True,
    "node_labels_are_e32_projection_of_official_edge_endpoints": True,
    "qkv_discoveries_require_whole_head_promotion_before_gt_scoring": True,
    "support_precision_is_e32_defined_not_official": True,
    "path_metric_is_e32_defined_not_official": True,
    "necessity_and_sufficiency_curves_are_e32_interventions_not_ground_truth": True,
    "pair_second_difference_is_not_a_communication_path": True,
    "rank_one_factor_is_descriptive_endpoint_separability": True,
    "pair_residual_does_not_isolate_higher_order_causality": True,
    "localization_does_not_by_itself_establish_faithfulness": True,
    "necessity_can_be_low_when_redundant_routes_exist": True,
    "query_prefixes_are_nested_not_independent_replicates": True,
    "bootstrap_does_not_establish_external_generalization": True,
}


def _finite_tuple(values: tuple[float, ...], *, label: str) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized or not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{label} must be a nonempty finite sequence")
    return normalized


@dataclass(frozen=True)
class ScoreSnapshot:
    """Complete node/edge scores after one cumulative discovery-query prefix."""

    queries: int
    node_scores: tuple[float, ...]
    edge_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if isinstance(self.queries, bool) or not isinstance(self.queries, int):
            raise TypeError("queries must be an integer")
        if self.queries <= 0:
            raise ValueError("queries must be positive")
        object.__setattr__(
            self,
            "node_scores",
            _finite_tuple(self.node_scores, label="node_scores"),
        )
        object.__setattr__(
            self,
            "edge_scores",
            _finite_tuple(self.edge_scores, label="edge_scores"),
        )


@dataclass(frozen=True)
class AdaptedScores:
    """Scores aligned exactly to frozen, ordered node and edge universes."""

    node_scores: tuple[float, ...]
    edge_scores: tuple[float, ...]
