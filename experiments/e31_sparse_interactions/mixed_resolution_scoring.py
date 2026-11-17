# ruff: noqa: EXE002
"""Pure two-stage scorer for the E31.2 mixed-resolution head atlas.

The scorer deliberately has no model, filesystem, registration, or acquisition
dependency.  Discovery freezes one exact 4-by-4 head rectangle inside each of
three coarse attention-pair parents.  Confirmation then evaluates those frozen
supports on the raw four-cell second difference and scores separately acquired
group-composite interventions against the corresponding whole-block response.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from itertools import combinations
from typing import Any

import torch

from experiments.e31_sparse_interactions.mixed_resolution_protocol import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CELL_BITS,
    COVER_COUNT,
    HEAD_COUNT,
    MATCHED_CONTROL_COUNT,
    MATCHED_CONTROL_SEED,
    PARENT_PAIRS,
    PRIMARY_GATES,
    RECTANGLE_HEAD_COUNT,
    build_joint_matched_rectangle_controls,
)

SCHEMA_VERSION = "e31_2_mixed_resolution_head_atlas_v1"
GEOMETRY_SCHEMA_VERSION = "e31_2_frozen_head_geometry_v1"
SCORE_SCHEMA_VERSION = "e31_2_mixed_resolution_head_score_v1"
PARENT_IDS = tuple(parent["parent_pair_id"] for parent in PARENT_PAIRS)
PARENT_COUNT = len(PARENT_IDS)
EDGE_COUNT_PER_PARENT = HEAD_COUNT * HEAD_COUNT
RECTANGLE_SIDE = RECTANGLE_HEAD_COUNT
RECTANGLE_EDGE_COUNT = RECTANGLE_SIDE * RECTANGLE_SIDE
DISCOVERY_CONTEXT_COUNT = 16
CONFIRMATION_CONTEXT_COUNT = 32
ORDERED_CELL_BITS = CELL_BITS

DEFAULT_MATCHED_CONTROL_COUNT = MATCHED_CONTROL_COUNT
DEFAULT_MATCHED_CONTROL_SEED = MATCHED_CONTROL_SEED
DEFAULT_BOOTSTRAP_REPLICATES = BOOTSTRAP_REPLICATES
DEFAULT_BOOTSTRAP_SEED = BOOTSTRAP_SEED
DEFAULT_PRIMARY_GATES = PRIMARY_GATES

CLAIM_BOUNDARIES = {
    "one_checkpoint_and_one_behavior": True,
    "three_discovery_frozen_coarse_attention_pairs_only": True,
    "raw_conditional_second_differences_not_independent_causal_paths": True,
    "higher_order_effects_can_contribute_to_pair_differences": True,
    "summed_edge_energy_is_not_an_orthogonal_output_variance_decomposition": True,
    "head_rectangle_is_a_support_not_a_direct_communication_graph": True,
    "direct_group_reconstruction_has_no_sum_or_affine_calibration": True,
    "residual_diagnostics_are_descriptive_only": True,
}

_HEAD_SUBSETS = tuple(combinations(range(HEAD_COUNT), RECTANGLE_SIDE))


def _as_finite_double(value: Any, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must be finite")
    return tensor


def _validated_fine_quartets(
    value: Any,
    *,
    label: str,
    expected_context_count: int | None = None,
) -> torch.Tensor:
    tensor = _as_finite_double(value, label=label)
    if (
        tensor.ndim != 5
        or tensor.shape[0] != PARENT_COUNT
        or tensor.shape[1] != EDGE_COUNT_PER_PARENT
        or tensor.shape[3] != COVER_COUNT
        or tensor.shape[4] != len(ORDERED_CELL_BITS)
    ):
        raise ValueError(
            f"{label} must have shape (three parents, 144 edges, contexts, "
            "4 covers, 4 ordered cells)"
        )
    if expected_context_count is not None and tensor.shape[2] != expected_context_count:
        raise ValueError(
            f"{label} must contain exactly {expected_context_count} contexts"
        )
    return tensor


def _validated_direct_quartets(
    value: Any,
    *,
    label: str,
    context_count: int,
) -> torch.Tensor:
    tensor = _as_finite_double(value, label=label)
    expected = (PARENT_COUNT, context_count, COVER_COUNT, len(ORDERED_CELL_BITS))
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{label} must have shape {expected}")
    return tensor


def _validated_parent_ids(parent_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(parent_ids, (str, bytes)):
        raise TypeError("parent_ids must be a sequence")
    normalized = tuple(parent_ids)
    if len(normalized) != PARENT_COUNT:
        raise ValueError("parent_ids must contain exactly three parents")
    if any(not isinstance(value, str) or not value for value in normalized):
        raise TypeError("parent identifiers must be nonempty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError("parent identifiers must be unique")
    return normalized


def _validated_gates(gates: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(gates, Mapping) or set(gates) != set(DEFAULT_PRIMARY_GATES):
        raise ValueError("primary_gates must provide the complete frozen schema")
    normalized = dict(gates)
    probability_fields = (
        "aggregate_confirmation_energy_coverage_minimum",
        "aggregate_fraction_of_oracle_rectangle_coverage_minimum",
        "aggregate_coverage_one_sided_95_lower_bound_minimum",
        "matched_control_quantile_strictly_exceeded",
    )
    for field in probability_fields:
        value = normalized[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field} must be numeric")
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{field} must lie in (0, 1]")
    for field in (
        "per_parent_energy_spearman_minimum",
        "pooled_direct_composite_r2_minimum",
        "pooled_direct_composite_pearson_minimum",
    ):
        value = normalized[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{field} must be finite")
    integer_bounds = {
        "matched_control_count": 1,
        "per_parent_sign_replication_count_minimum": 1,
        "per_parent_sign_replication_denominator": 1,
        "pair_selected_mse_beats_singleton_parent_count_minimum": 1,
        "parent_count": 1,
    }
    for field, minimum in integer_bounds.items():
        value = normalized[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
    if normalized["per_parent_sign_replication_count_minimum"] > 16:
        raise ValueError("the sign-replication threshold cannot exceed 16")
    if normalized["per_parent_sign_replication_denominator"] != RECTANGLE_EDGE_COUNT:
        raise ValueError("the sign-replication denominator must equal 16")
    if (
        normalized["pair_selected_mse_beats_singleton_parent_count_minimum"]
        > PARENT_COUNT
    ):
        raise ValueError("the reconstruction-win threshold cannot exceed 3")
    if normalized["parent_count"] != PARENT_COUNT:
        raise ValueError("the primary family must contain exactly three parents")
    if normalized["all_gates_required"] is not True:
        raise ValueError("all registered primary gates must be required")
    return normalized


def raw_second_difference(quartets: Any) -> torch.Tensor:
    """Return signed ``R11 - R10 - R01 + R00`` without clipping or squaring."""
    values = _as_finite_double(quartets, label="quartets")
    if values.ndim < 1 or values.shape[-1] != len(ORDERED_CELL_BITS):
        raise ValueError("quartets must end in four ordered 00,10,01,11 cells")
    return values[..., 3] - values[..., 1] - values[..., 2] + values[..., 0]


def _edge_indices(
    left_heads: Sequence[int], right_heads: Sequence[int]
) -> tuple[int, ...]:
    return tuple(
        left * HEAD_COUNT + right for left in left_heads for right in right_heads
    )


def _validated_head_set(value: Sequence[int], *, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    normalized = tuple(value)
    if (
        len(normalized) != RECTANGLE_SIDE
        or tuple(sorted(set(normalized))) != normalized
        or any(
            isinstance(head, bool)
            or not isinstance(head, int)
            or head < 0
            or head >= HEAD_COUNT
            for head in normalized
        )
    ):
        raise ValueError(f"{label} must be four unique ascending head indices")
    return normalized


def _rectangle_record(
    left_heads: Sequence[int],
    right_heads: Sequence[int],
    *,
    energy_sum: float,
) -> dict[str, Any]:
    left = _validated_head_set(left_heads, label="left_heads")
    right = _validated_head_set(right_heads, label="right_heads")
    return {
        "left_heads": list(left),
        "right_heads": list(right),
        "edge_indices": list(_edge_indices(left, right)),
        "energy_sum": float(energy_sum),
    }


def select_best_rectangle(energy: Any) -> dict[str, Any]:
    """Find the exact maximum-energy 4x4 Cartesian head rectangle.

    For each of the 495 left-head sets, the best four right columns are exact.
    Enumerating those row sets therefore examines the global optimum while
    avoiding a materialized 245,025-element candidate table.  Equal scores use
    the lexicographically smallest pair of ascending head sets.
    """
    values = _as_finite_double(energy, label="head-pair energy")
    if tuple(values.shape) != (HEAD_COUNT, HEAD_COUNT):
        raise ValueError("head-pair energy must have shape (12, 12)")
    if bool((values < 0.0).any()):
        raise ValueError("head-pair energy must be nonnegative")
    best_score = -math.inf
    best_support: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    for left_heads in _HEAD_SUBSETS:
        column_scores = values[list(left_heads)].sum(dim=0)
        ranked_columns = sorted(
            range(HEAD_COUNT),
            key=lambda column: (-float(column_scores[column]), column),
        )
        right_heads = tuple(sorted(ranked_columns[:RECTANGLE_SIDE]))
        score = float(column_scores[list(right_heads)].sum())
        candidate = (left_heads, right_heads)
        if score > best_score or (
            score == best_score and (best_support is None or candidate < best_support)
        ):
            best_score = score
            best_support = candidate
    if best_support is None:  # pragma: no cover - all validated grids have candidates
        raise RuntimeError("rectangle search produced no candidate")
    return _rectangle_record(*best_support, energy_sum=best_score)


def _top_singleton_heads(energy: torch.Tensor) -> tuple[int, ...]:
    if tuple(energy.shape) != (HEAD_COUNT,):
        raise ValueError("singleton energy must have shape (12,)")
    ranked = sorted(range(HEAD_COUNT), key=lambda head: (-float(energy[head]), head))
    return tuple(sorted(ranked[:RECTANGLE_SIDE]))


def _singleton_rectangle(quartets: torch.Tensor) -> dict[str, Any]:
    reshaped = quartets.reshape(
        HEAD_COUNT,
        HEAD_COUNT,
        quartets.shape[1],
        quartets.shape[2],
        len(ORDERED_CELL_BITS),
    )
    left_marginals = reshaped[..., 1] - reshaped[..., 0]
    right_marginals = reshaped[..., 2] - reshaped[..., 0]
    # A head's singleton estimate is pooled over its otherwise redundant partner
    # axis, contexts, and covers.  Squaring occurs only after the signed marginal.
    left_energy = left_marginals.square().mean(dim=(1, 2, 3))
    right_energy = right_marginals.square().mean(dim=(0, 2, 3))
    left_heads = _top_singleton_heads(left_energy)
    right_heads = _top_singleton_heads(right_energy)
    return {
        **_rectangle_record(left_heads, right_heads, energy_sum=0.0),
        "left_singleton_energy": [float(value) for value in left_energy],
        "right_singleton_energy": [float(value) for value in right_energy],
        "selection_rule": "top4_mean_squared_signed_singleton_marginal_per_side",
    }


def _canonical_support(
    records: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    if len(records) != PARENT_COUNT:
        raise ValueError("a joint support must contain three parent rectangles")
    return tuple(
        (
            _validated_head_set(left, label="control left heads"),
            _validated_head_set(right, label="control right heads"),
        )
        for left, right in records
    )


def fit_discovery_geometry(
    *,
    discovery_quartets: Any,
    parent_ids: Sequence[str] = PARENT_IDS,
    primary_gates: Mapping[str, Any] = DEFAULT_PRIMARY_GATES,
    matched_control_count: int = DEFAULT_MATCHED_CONTROL_COUNT,
    matched_control_seed: int = DEFAULT_MATCHED_CONTROL_SEED,
    enforce_registered_design: bool = True,
) -> dict[str, Any]:
    """Freeze every outcome-dependent quantity using discovery only."""
    normalized_parents = _validated_parent_ids(parent_ids)
    gates = _validated_gates(primary_gates)
    expected_contexts = DISCOVERY_CONTEXT_COUNT if enforce_registered_design else None
    quartets = _validated_fine_quartets(
        discovery_quartets,
        label="discovery_quartets",
        expected_context_count=expected_contexts,
    )
    if matched_control_seed != DEFAULT_MATCHED_CONTROL_SEED:
        raise ValueError("matched controls must use the protocol-owned seed")
    if enforce_registered_design and (
        normalized_parents != PARENT_IDS
        or gates != DEFAULT_PRIMARY_GATES
        or matched_control_count != DEFAULT_MATCHED_CONTROL_COUNT
        or matched_control_seed != DEFAULT_MATCHED_CONTROL_SEED
    ):
        raise ValueError("the registered E31.2 discovery design changed")

    interactions = raw_second_difference(quartets)
    energy = interactions.square().mean(dim=(2, 3))
    signed_mean = interactions.mean(dim=(2, 3))
    parents: dict[str, Any] = {}
    for parent_index, parent_id in enumerate(normalized_parents):
        selected = select_best_rectangle(
            energy[parent_index].reshape(HEAD_COUNT, HEAD_COUNT)
        )
        singleton = _singleton_rectangle(quartets[parent_index])
        total = float(energy[parent_index].sum())
        parents[parent_id] = {
            "parent_index": parent_index,
            "edge_order": "left_head_major_then_right_head_0_to_11",
            "discovery_raw_energy": [float(value) for value in energy[parent_index]],
            "discovery_raw_signed_mean": [
                float(value) for value in signed_mean[parent_index]
            ],
            "selected_rectangle": {
                **selected,
                "discovery_energy_coverage": (
                    selected["energy_sum"] / total if total > 0.0 else 0.0
                ),
                "selection_rule": "exact_maximum_sum_of_16_raw_pair_energies",
                "tie_break": "lexicographic_left_heads_then_right_heads",
            },
            "singleton_rectangle": singleton,
        }
    reconstruction_selections = {
        "schema_version": "e31_2_discovery_rectangle_selections_v1",
        "parents": [
            {
                "parent_pair_id": parent_id,
                "pair_selected": {
                    "heads_a": deepcopy(
                        parents[parent_id]["selected_rectangle"]["left_heads"]
                    ),
                    "heads_b": deepcopy(
                        parents[parent_id]["selected_rectangle"]["right_heads"]
                    ),
                },
                "singleton_selected": {
                    "heads_a": deepcopy(
                        parents[parent_id]["singleton_rectangle"]["left_heads"]
                    ),
                    "heads_b": deepcopy(
                        parents[parent_id]["singleton_rectangle"]["right_heads"]
                    ),
                },
            }
            for parent_id in normalized_parents
        ],
    }
    controls = build_joint_matched_rectangle_controls(
        reconstruction_selections, count=matched_control_count
    )
    return {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "status": "discovery_geometry_frozen_before_confirmation",
        "confirmation_outcomes_used": False,
        "parent_ids": list(normalized_parents),
        "fine_quartets_shape": list(quartets.shape),
        "raw_interaction_formula": "R11-R10-R01+R00",
        "square_after_signed_second_difference": True,
        "rectangle_side": RECTANGLE_SIDE,
        "primary_gates": deepcopy(gates),
        "parents": parents,
        "reconstruction_selections": reconstruction_selections,
        "matched_controls": controls,
        "claim_boundaries": deepcopy(CLAIM_BOUNDARIES),
        "registered_design_enforced": enforce_registered_design,
    }


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    vector = values.flatten().double()
    order = torch.argsort(vector, stable=True)
    ranks = torch.empty_like(vector)
    start = 0
    while start < len(vector):
        end = start + 1
        current = vector[order[start]]
        while end < len(vector) and bool(vector[order[end]] == current):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    first = left.flatten().double()
    second = right.flatten().double()
    if first.shape != second.shape or len(first) < 2:
        raise ValueError("correlation inputs must be aligned and nontrivial")
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.sqrt(first.square().sum() * second.square().sum())
    if float(denominator) == 0.0:
        return None
    return float((first * second).sum() / denominator)


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float | None:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _support_from_record(
    record: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(record, Mapping):
        raise TypeError("rectangle record must be a mapping")
    left = _validated_head_set(record.get("left_heads", ()), label="left_heads")
    right = _validated_head_set(record.get("right_heads", ()), label="right_heads")
    expected_indices = list(_edge_indices(left, right))
    if "edge_indices" in record and record["edge_indices"] != expected_indices:
        raise ValueError("rectangle edge indices differ from its head sets")
    return left, right


def _support_energy(
    energy: torch.Tensor, support: tuple[tuple[int, ...], tuple[int, ...]]
) -> float:
    left, right = support
    grid = energy.reshape(HEAD_COUNT, HEAD_COUNT)
    return float(grid[list(left)][:, list(right)].sum())


def _control_supports(
    controls: Mapping[str, Any], *, parent_count: int
) -> tuple[tuple[tuple[tuple[int, ...], tuple[int, ...]], ...], ...]:
    if not isinstance(controls, Mapping):
        raise TypeError("matched controls must be a mapping")
    if (
        controls.get("schema_version") != "e31_2_joint_matched_rectangle_controls_v1"
        or controls.get("generator")
        != "sha256_modulo_lexicographic_12_choose_4_per_side"
        or controls.get("seed") != DEFAULT_MATCHED_CONTROL_SEED
        or controls.get("selected_and_singleton_supports_excluded") is not True
        or controls.get("uses_model_outcomes") is not False
    ):
        raise ValueError("matched controls differ from the protocol-owned generator")
    rows = controls.get("supports")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("matched controls omit support rows")
    normalized = []
    for row in rows:
        if not isinstance(row, Sequence) or len(row) != parent_count:
            raise ValueError("every matched control must contain three rectangles")
        if any(not isinstance(record, Mapping) for record in row):
            raise TypeError("every matched-control rectangle must be a mapping")
        if tuple(record.get("parent_pair_id") for record in row) != PARENT_IDS:
            raise ValueError("matched controls changed registered parent order")
        normalized.append(
            _canonical_support(
                tuple(
                    (
                        tuple(record.get("heads_a", ())),
                        tuple(record.get("heads_b", ())),
                    )
                    for record in row
                )
            )
        )
    result = tuple(normalized)
    if len(result) != controls.get("support_count") or len(result) != len(set(result)):
        raise ValueError("matched-control custody fields are inconsistent")
    return result


def _empirical_quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile values must be nonempty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _score_energy_atlas(
    *,
    discovery_interactions: torch.Tensor,
    confirmation_interactions: torch.Tensor,
    geometry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor]:
    discovery_energy = discovery_interactions.square().mean(dim=(2, 3))
    confirmation_energy = confirmation_interactions.square().mean(dim=(2, 3))
    discovery_means = discovery_interactions.mean(dim=(2, 3))
    confirmation_means = confirmation_interactions.mean(dim=(2, 3))
    parent_ids = tuple(geometry["parent_ids"])
    selected_supports = []
    parent_records: dict[str, Any] = {}
    selected_energy = 0.0
    oracle_energy = 0.0
    for parent_index, parent_id in enumerate(parent_ids):
        geometry_parent = geometry["parents"][parent_id]
        support = _support_from_record(geometry_parent["selected_rectangle"])
        selected_supports.append(support)
        indices = list(_edge_indices(*support))
        selected_value = float(confirmation_energy[parent_index, indices].sum())
        total_value = float(confirmation_energy[parent_index].sum())
        oracle = select_best_rectangle(
            confirmation_energy[parent_index].reshape(HEAD_COUNT, HEAD_COUNT)
        )
        oracle_value = float(oracle["energy_sum"])
        sign_count = int(
            (
                discovery_means[parent_index, indices]
                * confirmation_means[parent_index, indices]
                > 0.0
            ).sum()
        )
        selected_energy += selected_value
        oracle_energy += oracle_value
        parent_records[parent_id] = {
            "parent_index": parent_index,
            "selected_rectangle": deepcopy(geometry_parent["selected_rectangle"]),
            "confirmation_oracle_rectangle": oracle,
            "confirmation_total_energy": total_value,
            "selected_confirmation_energy": selected_value,
            "selected_confirmation_energy_coverage": (
                selected_value / total_value if total_value > 0.0 else 0.0
            ),
            "selected_fraction_of_oracle_rectangle_coverage": (
                selected_value / oracle_value if oracle_value > 0.0 else 0.0
            ),
            "full_map_energy_spearman": _spearman(
                discovery_energy[parent_index], confirmation_energy[parent_index]
            ),
            "full_map_energy_pearson_descriptive": _pearson(
                discovery_energy[parent_index], confirmation_energy[parent_index]
            ),
            "selected_sign_replication_count": sign_count,
            "selected_sign_replication_denominator": RECTANGLE_EDGE_COUNT,
        }
    total_energy = float(confirmation_energy.sum())
    aggregate_coverage = selected_energy / total_energy if total_energy > 0.0 else 0.0
    aggregate_oracle_efficiency = (
        selected_energy / oracle_energy if oracle_energy > 0.0 else 0.0
    )

    controls_record = geometry["matched_controls"]
    controls = _control_supports(controls_record, parent_count=PARENT_COUNT)
    control_coverages = []
    for control in controls:
        value = sum(
            _support_energy(confirmation_energy[parent], support)
            for parent, support in enumerate(control)
        )
        control_coverages.append(value / total_energy if total_energy > 0.0 else 0.0)
    probability = float(
        geometry["primary_gates"]["matched_control_quantile_strictly_exceeded"]
    )
    tested_quantile = _empirical_quantile(control_coverages, probability)
    primary = {
        "selected_edge_count": PARENT_COUNT * RECTANGLE_EDGE_COUNT,
        "total_edge_count": PARENT_COUNT * EDGE_COUNT_PER_PARENT,
        "aggregate_confirmation_energy_coverage": aggregate_coverage,
        "aggregate_fraction_of_oracle_rectangle_coverage": aggregate_oracle_efficiency,
        "pooled_full_map_energy_spearman_descriptive": _spearman(
            discovery_energy, confirmation_energy
        ),
        "matched_controls": {
            "support_count": len(controls),
            "tested_quantile_probability": probability,
            "coverage_tested_quantile": tested_quantile,
            "selected_minus_tested_quantile": aggregate_coverage - tested_quantile,
            "passed": aggregate_coverage > tested_quantile,
        },
    }
    return primary, parent_records, discovery_energy, confirmation_energy


def _agreement(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    predicted = prediction.flatten().double()
    observed = target.flatten().double()
    if predicted.shape != observed.shape or len(predicted) < 2:
        raise ValueError("reconstruction vectors must be aligned and nontrivial")
    residual_sum = float((observed - predicted).square().sum())
    total_sum = float((observed - observed.mean()).square().sum())
    return {
        "sample_count": len(observed),
        "mse": float((observed - predicted).square().mean()),
        "r2": (1.0 - residual_sum / total_sum) if total_sum > 0.0 else None,
        "pearson": _pearson(predicted, observed),
    }


def _score_reconstruction(
    *,
    pair_quartets: torch.Tensor,
    singleton_quartets: torch.Tensor,
    coarse_quartets: torch.Tensor,
    parent_ids: tuple[str, ...],
) -> dict[str, Any]:
    pair = raw_second_difference(pair_quartets)
    singleton = raw_second_difference(singleton_quartets)
    coarse = raw_second_difference(coarse_quartets)
    pair_parents = {}
    singleton_parents = {}
    wins = 0
    for parent_index, parent_id in enumerate(parent_ids):
        pair_metric = _agreement(pair[parent_index], coarse[parent_index])
        singleton_metric = _agreement(singleton[parent_index], coarse[parent_index])
        pair_parents[parent_id] = pair_metric
        singleton_parents[parent_id] = singleton_metric
        wins += pair_metric["mse"] < singleton_metric["mse"]
    pair_pooled = _agreement(pair, coarse)
    singleton_pooled = _agreement(singleton, coarse)
    return {
        "target": "direct_all12_by_all12_parent_composite_second_difference",
        "prediction_semantics": "direct_acquired_4head_by_4head_group_composite",
        "sum_of_pair_coefficients_used": False,
        "affine_calibration_used": False,
        "pair_selected_composite": {
            "pooled_r2": pair_pooled["r2"],
            "pooled_pearson": pair_pooled["pearson"],
            "pooled_mse": pair_pooled["mse"],
            "parents": pair_parents,
        },
        "singleton_selected_composite": {
            "pooled_r2": singleton_pooled["r2"],
            "pooled_pearson": singleton_pooled["pearson"],
            "pooled_mse": singleton_pooled["mse"],
            "parents": singleton_parents,
        },
        "pair_selected_mse_beats_singleton_parent_count": int(wins),
        "parent_count": PARENT_COUNT,
        "comparison_metric": "strictly_lower_confirmation_mse",
    }


def _metric_at_least(value: Any, threshold: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and bool(math.isfinite(float(value)) and float(value) >= float(threshold))
    )


def evaluate_primary_gate(
    *,
    primary: Mapping[str, Any],
    parents: Mapping[str, Mapping[str, Any]],
    reconstruction: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    gates: Mapping[str, Any] = DEFAULT_PRIMARY_GATES,
) -> dict[str, Any]:
    """Evaluate the single aggregate family and its all-parent conjunctions."""
    thresholds = _validated_gates(gates)
    if tuple(parents) != PARENT_IDS:
        raise ValueError("parents must preserve the three registered parent IDs")
    spearman_pass = all(
        _metric_at_least(
            record.get("full_map_energy_spearman"),
            thresholds["per_parent_energy_spearman_minimum"],
        )
        for record in parents.values()
    )
    sign_pass = all(
        _metric_at_least(
            record.get("selected_sign_replication_count"),
            thresholds["per_parent_sign_replication_count_minimum"],
        )
        and record.get("selected_sign_replication_denominator")
        == thresholds["per_parent_sign_replication_denominator"]
        for record in parents.values()
    )
    controls = primary.get("matched_controls", {})
    pair_reconstruction = reconstruction.get("pair_selected_composite", {})
    checks = {
        "aggregate_confirmation_energy_coverage": _metric_at_least(
            primary.get("aggregate_confirmation_energy_coverage"),
            thresholds["aggregate_confirmation_energy_coverage_minimum"],
        ),
        "aggregate_fraction_of_oracle_rectangle_coverage": _metric_at_least(
            primary.get("aggregate_fraction_of_oracle_rectangle_coverage"),
            thresholds["aggregate_fraction_of_oracle_rectangle_coverage_minimum"],
        ),
        "aggregate_coverage_bootstrap_lower_bound": _metric_at_least(
            bootstrap.get("aggregate_coverage_one_sided_95_lower"),
            thresholds["aggregate_coverage_one_sided_95_lower_bound_minimum"],
        ),
        "matched_random_controls": (
            controls.get("passed") is True
            and _metric_at_least(
                controls.get("support_count"),
                thresholds["matched_control_count"],
            )
        ),
        "all_parent_full_map_energy_spearman": spearman_pass,
        "all_parent_selected_sign_replication": sign_pass,
        "reconstruction_r2": _metric_at_least(
            pair_reconstruction.get("pooled_r2"),
            thresholds["pooled_direct_composite_r2_minimum"],
        ),
        "reconstruction_pearson": _metric_at_least(
            pair_reconstruction.get("pooled_pearson"),
            thresholds["pooled_direct_composite_pearson_minimum"],
        ),
        "pair_reconstruction_beats_singleton": _metric_at_least(
            reconstruction.get("pair_selected_mse_beats_singleton_parent_count"),
            thresholds["pair_selected_mse_beats_singleton_parent_count_minimum"],
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": deepcopy(thresholds),
        "all_checks_required": True,
        "all_three_parent_conjunctions_required": True,
    }


def _validated_context_structure(
    *,
    context_strata: Sequence[Hashable],
    context_cluster_ids: Sequence[Hashable],
    context_count: int,
) -> tuple[tuple[Hashable, ...], tuple[Hashable, ...]]:
    if isinstance(context_strata, (str, bytes)) or isinstance(
        context_cluster_ids, (str, bytes)
    ):
        raise TypeError("context metadata must be sequences")
    strata = tuple(context_strata)
    clusters = tuple(context_cluster_ids)
    if len(strata) != context_count or len(clusters) != context_count:
        raise ValueError("context strata and clusters must align with contexts")
    try:
        set(strata)
        set(clusters)
    except TypeError as error:
        raise TypeError("context strata and clusters must be hashable") from error
    if not strata or not clusters:
        raise ValueError("context metadata must be nonempty")
    cluster_strata: dict[Hashable, Hashable] = {}
    for stratum, cluster in zip(strata, clusters, strict=True):
        if cluster in cluster_strata and cluster_strata[cluster] != stratum:
            raise ValueError("each semantic-frame cluster must belong to one template")
        cluster_strata[cluster] = stratum
    cluster_sizes: dict[Hashable, int] = {}
    for cluster in clusters:
        cluster_sizes[cluster] = cluster_sizes.get(cluster, 0) + 1
    if len(set(cluster_sizes.values())) != 1:
        raise ValueError("semantic-frame clusters must have equal size")
    return strata, clusters


def stratified_cluster_bootstrap_indices(
    *,
    context_strata: Sequence[Hashable],
    context_cluster_ids: Sequence[Hashable],
    replicates: int,
    seed: int,
) -> torch.Tensor:
    """Resample semantic frames within template strata, retaining both directions."""
    context_count = len(context_strata)
    strata, clusters = _validated_context_structure(
        context_strata=context_strata,
        context_cluster_ids=context_cluster_ids,
        context_count=context_count,
    )
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 1
    ):
        raise ValueError("bootstrap replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    stratum_order = tuple(dict.fromkeys(strata))
    cluster_order = tuple(dict.fromkeys(clusters))
    context_indices = {
        cluster: tuple(
            index for index, value in enumerate(clusters) if value == cluster
        )
        for cluster in cluster_order
    }
    clusters_by_stratum = {
        stratum: tuple(
            cluster
            for cluster in cluster_order
            if strata[context_indices[cluster][0]] == stratum
        )
        for stratum in stratum_order
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    rows = []
    for _ in range(replicates):
        row = []
        for stratum in stratum_order:
            candidates = clusters_by_stratum[stratum]
            draws = torch.randint(
                len(candidates),
                (len(candidates),),
                generator=generator,
            )
            for draw in draws.tolist():
                row.extend(context_indices[candidates[draw]])
        if len(row) != context_count:
            raise RuntimeError("a stratified cluster bootstrap changed context count")
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long)


def _bootstrap_reconstruction_metrics(
    prediction: torch.Tensor, target: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Inputs are parent x context x cover. Advanced indexing inserts the
    # replicate axis between parent and resampled context axes.
    predicted = prediction[:, indices, :].permute(1, 0, 2, 3).flatten(start_dim=1)
    observed = target[:, indices, :].permute(1, 0, 2, 3).flatten(start_dim=1)
    residual = (observed - predicted).square()
    mse = residual.mean(dim=1)
    total = (observed - observed.mean(dim=1, keepdim=True)).square().sum(dim=1)
    r2 = 1.0 - residual.sum(dim=1) / total
    centered_prediction = predicted - predicted.mean(dim=1, keepdim=True)
    centered_observed = observed - observed.mean(dim=1, keepdim=True)
    denominator = torch.sqrt(
        centered_prediction.square().sum(dim=1) * centered_observed.square().sum(dim=1)
    )
    pearson = (centered_prediction * centered_observed).sum(dim=1) / denominator
    return r2, pearson, mse


def _batched_oracle_energy(energy: torch.Tensor) -> torch.Tensor:
    """Return exact best-rectangle energy for replicate x parent grids."""
    if energy.ndim != 4 or tuple(energy.shape[1:]) != (
        PARENT_COUNT,
        HEAD_COUNT,
        HEAD_COUNT,
    ):
        raise ValueError("batched energy must have shape (replicates, 3, 12, 12)")
    best = torch.full((energy.shape[0], PARENT_COUNT), -torch.inf, dtype=torch.float64)
    subset_tensor = torch.tensor(_HEAD_SUBSETS, dtype=torch.long)
    # Chunking avoids materializing replicate x parent x 495 x 4 x 12.
    for start in range(0, len(_HEAD_SUBSETS), 16):
        rows = subset_tensor[start : start + 16]
        row_scores = energy[:, :, rows, :].sum(dim=3)
        candidates = torch.topk(
            row_scores, k=RECTANGLE_SIDE, dim=-1, largest=True, sorted=False
        ).values.sum(dim=-1)
        best = torch.maximum(best, candidates.max(dim=-1).values)
    return best


def _lower_quantile(values: torch.Tensor, probability: float = 0.05) -> float | None:
    finite = values[torch.isfinite(values)].flatten()
    if len(finite) == 0:
        return None
    return _empirical_quantile([float(value) for value in finite], probability)


def _bootstrap_summary(
    *,
    discovery_interactions: torch.Tensor,
    confirmation_interactions: torch.Tensor,
    selected_supports: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    pair_reconstruction: torch.Tensor,
    singleton_reconstruction: torch.Tensor,
    coarse_reconstruction: torch.Tensor,
    context_strata: Sequence[Hashable],
    context_cluster_ids: Sequence[Hashable],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    indices = stratified_cluster_bootstrap_indices(
        context_strata=context_strata,
        context_cluster_ids=context_cluster_ids,
        replicates=replicates,
        seed=seed,
    )
    discovery_energy = discovery_interactions.square().mean(dim=(2, 3))
    discovery_means = discovery_interactions.mean(dim=(2, 3))
    context_energy = confirmation_interactions.square().mean(dim=3)
    context_means = confirmation_interactions.mean(dim=3)
    # Advanced indexing gives parent x edge x replicate x resampled-context.
    boot_energy = context_energy[:, :, indices].mean(dim=-1).permute(2, 0, 1)
    boot_means = context_means[:, :, indices].mean(dim=-1).permute(2, 0, 1)
    support_mask = torch.zeros((PARENT_COUNT, EDGE_COUNT_PER_PARENT), dtype=torch.bool)
    for parent, support in enumerate(selected_supports):
        support_mask[parent, list(_edge_indices(*support))] = True
    selected_energy = (boot_energy * support_mask[None]).sum(dim=(1, 2))
    total_energy = boot_energy.sum(dim=(1, 2))
    coverage = torch.where(total_energy > 0.0, selected_energy / total_energy, 0.0)
    oracle_energy = _batched_oracle_energy(
        boot_energy.reshape(-1, PARENT_COUNT, HEAD_COUNT, HEAD_COUNT)
    ).sum(dim=1)
    oracle_efficiency = torch.where(
        oracle_energy > 0.0, selected_energy / oracle_energy, 0.0
    )

    minimum_spearman = []
    for replicate in range(replicates):
        correlations = [
            _spearman(discovery_energy[parent], boot_energy[replicate, parent])
            for parent in range(PARENT_COUNT)
        ]
        minimum_spearman.append(
            min(value for value in correlations if value is not None)
            if all(value is not None for value in correlations)
            else math.nan
        )
    sign_counts = ((discovery_means[None] * boot_means > 0.0) & support_mask[None]).sum(
        dim=2
    )
    minimum_sign_count = sign_counts.min(dim=1).values.double()

    pair_r2, pair_pearson, _ = _bootstrap_reconstruction_metrics(
        pair_reconstruction, coarse_reconstruction, indices
    )
    # Parent-wise wins require parent-specific MSE, not the pooled MSE above.
    parent_wins = torch.zeros(replicates, dtype=torch.float64)
    for parent in range(PARENT_COUNT):
        _, _, pair_parent_mse = _bootstrap_reconstruction_metrics(
            pair_reconstruction[parent : parent + 1],
            coarse_reconstruction[parent : parent + 1],
            indices,
        )
        _, _, singleton_parent_mse = _bootstrap_reconstruction_metrics(
            singleton_reconstruction[parent : parent + 1],
            coarse_reconstruction[parent : parent + 1],
            indices,
        )
        parent_wins += (pair_parent_mse < singleton_parent_mse).double()

    return {
        "replicates": replicates,
        "seed": seed,
        "resample_unit": "semantic_frame_with_both_directions",
        "cluster_unit": "provided_semantic_frame_cluster",
        "stratified_by": "provided_template_context_strata",
        "common_resamples_for_all_parents_and_metrics": True,
        "multiplicity_summary": "minimum_over_three_parent_conjunctions",
        "aggregate_coverage_one_sided_95_lower": _lower_quantile(coverage),
        "aggregate_fraction_of_oracle_rectangle_coverage_one_sided_95_lower": (
            _lower_quantile(oracle_efficiency)
        ),
        "minimum_parent_energy_spearman_one_sided_95_lower": _lower_quantile(
            torch.tensor(minimum_spearman, dtype=torch.float64)
        ),
        "minimum_parent_sign_count_one_sided_95_lower": _lower_quantile(
            minimum_sign_count
        ),
        "reconstruction_r2_one_sided_95_lower": _lower_quantile(pair_r2),
        "reconstruction_pearson_one_sided_95_lower": _lower_quantile(pair_pearson),
        "pair_selected_mse_beats_singleton_parent_count_one_sided_95_lower": (
            _lower_quantile(parent_wins)
        ),
        "primary_gate_fields": ["aggregate_coverage_one_sided_95_lower"],
        "remaining_bootstrap_intervals_are_descriptive": True,
    }


def _residual_diagnostics(
    *,
    discovery_residual_interactions: Any | None,
    confirmation_residual_interactions: Any | None,
    geometry: Mapping[str, Any],
    discovery_shape: tuple[int, ...],
    confirmation_shape: tuple[int, ...],
) -> dict[str, Any]:
    if (
        discovery_residual_interactions is None
        and confirmation_residual_interactions is None
    ):
        return {"computed": False, "reason": "residual_interactions_not_supplied"}
    if (
        discovery_residual_interactions is None
        or confirmation_residual_interactions is None
    ):
        raise ValueError(
            "both discovery and confirmation residual interactions are required"
        )
    discovery = _as_finite_double(
        discovery_residual_interactions, label="discovery residual interactions"
    )
    confirmation = _as_finite_double(
        confirmation_residual_interactions, label="confirmation residual interactions"
    )
    expected_discovery = discovery_shape[:-1]
    expected_confirmation = confirmation_shape[:-1]
    if (
        tuple(discovery.shape) != expected_discovery
        or tuple(confirmation.shape) != expected_confirmation
    ):
        raise ValueError(
            "residual interactions must align with raw interaction tensors"
        )
    primary, parents, _, _ = _score_energy_atlas(
        discovery_interactions=discovery,
        confirmation_interactions=confirmation,
        geometry=geometry,
    )
    return {
        "computed": True,
        "inferential_status": "descriptive_only",
        "aggregate_confirmation_energy_coverage": primary[
            "aggregate_confirmation_energy_coverage"
        ],
        "aggregate_fraction_of_oracle_rectangle_coverage": primary[
            "aggregate_fraction_of_oracle_rectangle_coverage"
        ],
        "parents": {
            parent_id: {
                "full_map_energy_spearman": record["full_map_energy_spearman"],
                "selected_sign_replication_count": record[
                    "selected_sign_replication_count"
                ],
            }
            for parent_id, record in parents.items()
        },
        "changes_primary_verdict": False,
    }


def score_with_frozen_geometry(
    *,
    discovery_quartets: Any,
    confirmation_quartets: Any,
    pair_selected_composite_quartets: Any,
    singleton_selected_composite_quartets: Any,
    coarse_composite_quartets: Any,
    discovery_context_strata: Sequence[Hashable],
    discovery_context_cluster_ids: Sequence[Hashable],
    confirmation_context_strata: Sequence[Hashable],
    confirmation_context_cluster_ids: Sequence[Hashable],
    frozen_geometry: Mapping[str, Any],
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    discovery_residual_interactions: Any | None = None,
    confirmation_residual_interactions: Any | None = None,
    enforce_registered_design: bool = True,
) -> dict[str, Any]:
    """Score untouched confirmation outcomes under discovery-frozen geometry."""
    if not isinstance(frozen_geometry, Mapping):
        raise TypeError("frozen_geometry must be a mapping")
    geometry = deepcopy(dict(frozen_geometry))
    if (
        geometry.get("schema_version") != GEOMETRY_SCHEMA_VERSION
        or geometry.get("status") != "discovery_geometry_frozen_before_confirmation"
        or geometry.get("confirmation_outcomes_used") is not False
    ):
        raise ValueError("frozen geometry header changed")
    parent_ids = _validated_parent_ids(geometry.get("parent_ids", ()))
    gates = _validated_gates(geometry.get("primary_gates", {}))
    expected_discovery = DISCOVERY_CONTEXT_COUNT if enforce_registered_design else None
    expected_confirmation = (
        CONFIRMATION_CONTEXT_COUNT if enforce_registered_design else None
    )
    discovery = _validated_fine_quartets(
        discovery_quartets,
        label="discovery_quartets",
        expected_context_count=expected_discovery,
    )
    confirmation = _validated_fine_quartets(
        confirmation_quartets,
        label="confirmation_quartets",
        expected_context_count=expected_confirmation,
    )
    if enforce_registered_design and (
        parent_ids != PARENT_IDS
        or gates != DEFAULT_PRIMARY_GATES
        or bootstrap_replicates != DEFAULT_BOOTSTRAP_REPLICATES
        or bootstrap_seed != DEFAULT_BOOTSTRAP_SEED
    ):
        raise ValueError("the registered E31.2 scoring design changed")

    recomputed = fit_discovery_geometry(
        discovery_quartets=discovery,
        parent_ids=parent_ids,
        primary_gates=gates,
        matched_control_count=int(geometry["matched_controls"]["support_count"]),
        matched_control_seed=int(geometry["matched_controls"]["seed"]),
        enforce_registered_design=enforce_registered_design,
    )
    if recomputed != geometry:
        raise RuntimeError("frozen geometry differs from recomputation")

    discovery_strata, discovery_clusters = _validated_context_structure(
        context_strata=discovery_context_strata,
        context_cluster_ids=discovery_context_cluster_ids,
        context_count=discovery.shape[2],
    )
    confirmation_strata, confirmation_clusters = _validated_context_structure(
        context_strata=confirmation_context_strata,
        context_cluster_ids=confirmation_context_cluster_ids,
        context_count=confirmation.shape[2],
    )
    pair_composite = _validated_direct_quartets(
        pair_selected_composite_quartets,
        label="pair_selected_composite_quartets",
        context_count=confirmation.shape[2],
    )
    singleton_composite = _validated_direct_quartets(
        singleton_selected_composite_quartets,
        label="singleton_selected_composite_quartets",
        context_count=confirmation.shape[2],
    )
    coarse_composite = _validated_direct_quartets(
        coarse_composite_quartets,
        label="coarse_composite_quartets",
        context_count=confirmation.shape[2],
    )

    discovery_interactions = raw_second_difference(discovery)
    confirmation_interactions = raw_second_difference(confirmation)
    primary, parents, _, _ = _score_energy_atlas(
        discovery_interactions=discovery_interactions,
        confirmation_interactions=confirmation_interactions,
        geometry=geometry,
    )
    reconstruction = _score_reconstruction(
        pair_quartets=pair_composite,
        singleton_quartets=singleton_composite,
        coarse_quartets=coarse_composite,
        parent_ids=parent_ids,
    )
    selected_supports = tuple(
        _support_from_record(geometry["parents"][parent_id]["selected_rectangle"])
        for parent_id in parent_ids
    )
    bootstrap = _bootstrap_summary(
        discovery_interactions=discovery_interactions,
        confirmation_interactions=confirmation_interactions,
        selected_supports=selected_supports,
        pair_reconstruction=raw_second_difference(pair_composite),
        singleton_reconstruction=raw_second_difference(singleton_composite),
        coarse_reconstruction=raw_second_difference(coarse_composite),
        context_strata=confirmation_strata,
        context_cluster_ids=confirmation_clusters,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    primary_gate = evaluate_primary_gate(
        primary=primary,
        parents=parents,
        reconstruction=reconstruction,
        bootstrap=bootstrap,
        gates=gates,
    )
    residual = _residual_diagnostics(
        discovery_residual_interactions=discovery_residual_interactions,
        confirmation_residual_interactions=confirmation_residual_interactions,
        geometry=geometry,
        discovery_shape=tuple(discovery.shape),
        confirmation_shape=tuple(confirmation.shape),
    )
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "status": "confirmation_scored_without_model_calls_or_adaptation",
        "frozen_geometry_recomputed_exactly": True,
        "confirmation_adaptation_allowed": False,
        "parent_ids": list(parent_ids),
        "raw_interaction_formula": "R11-R10-R01+R00",
        "primary_verdict": "pass" if primary_gate["passed"] else "fail",
        "primary": primary,
        "parents": parents,
        "reconstruction": reconstruction,
        "primary_gate": primary_gate,
        "bootstrap": bootstrap,
        "residual_diagnostics": residual,
        "multiplicity": {
            "aggregate_primary_family_count": 1,
            "aggregate_coverage_oracle_and_control_are_single_joint_tests": True,
            "parent_metrics_are_conjunctive_diagnostics_not_separate_claims": True,
            "all_three_parent_conjunctions_required": True,
            "posthoc_e31_2_parent_adaptation_allowed": False,
            "parent_selection_was_posthoc_e31_1": True,
        },
        "context_accounting": {
            "discovery_context_count": discovery.shape[2],
            "confirmation_context_count": confirmation.shape[2],
            "discovery_semantic_frame_count": len(set(discovery_clusters)),
            "confirmation_semantic_frame_count": len(set(confirmation_clusters)),
            "discovery_template_stratum_count": len(set(discovery_strata)),
            "confirmation_template_stratum_count": len(set(confirmation_strata)),
        },
        "claim_boundaries": deepcopy(CLAIM_BOUNDARIES),
    }


__all__ = [
    "CLAIM_BOUNDARIES",
    "CONFIRMATION_CONTEXT_COUNT",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_MATCHED_CONTROL_COUNT",
    "DEFAULT_MATCHED_CONTROL_SEED",
    "DEFAULT_PRIMARY_GATES",
    "DISCOVERY_CONTEXT_COUNT",
    "EDGE_COUNT_PER_PARENT",
    "HEAD_COUNT",
    "PARENT_IDS",
    "RECTANGLE_EDGE_COUNT",
    "RECTANGLE_SIDE",
    "evaluate_primary_gate",
    "fit_discovery_geometry",
    "raw_second_difference",
    "score_with_frozen_geometry",
    "select_best_rectangle",
    "stratified_cluster_bootstrap_indices",
]
