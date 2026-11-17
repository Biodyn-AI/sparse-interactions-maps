# ruff: noqa: EXE002
"""Pure prospective scoring for the E31.1 conditional interaction atlas.

The scorer consumes already acquired exact ``00, 10, 01, 11`` response
quartets.  It never loads a model, reads an acquisition artifact, chooses a
prompt, or writes a result.  Discovery data select only the saturation curve
and edge order; every reported confirmation endpoint is then evaluated on the
frozen order using the disjoint confirmation panel.

The finite differences measured here are conditional causal effects.  They
can contain contributions from higher-order mechanisms, so their summed
square is an interaction-atlas concentration measure rather than an
orthogonal decomposition of model output variance.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from itertools import combinations
from typing import Any

import torch

from experiments.e31_sparse_interactions.prospective_protocol import (
    ASINH_LAMBDA_GRID,
    BOOTSTRAP,
    COVER_TARGET,
    DESIGN_DOMAIN,
    HIERARCHICAL_SCALE_GATE,
    HUB_COMPONENT,
    NONHUB_COMPONENTS,
    NONHUB_EDGE_COUNT,
    PRIMARY_GATES,
    SATURATION_EDGE_FOLD_COUNT,
    SATURATION_HUBER_DELTA,
    SCALE_CARDINALITIES,
    SPOKE_COUNT,
)

DEFAULT_CURVATURE_GRID = ASINH_LAMBDA_GRID
DEFAULT_BOOTSTRAP_REPLICATES = int(BOOTSTRAP["replicates"])
DEFAULT_BOOTSTRAP_SEED = int(BOOTSTRAP["seed"])
DEFAULT_PRIMARY_GATES = PRIMARY_GATES
HUBER_DELTA = SATURATION_HUBER_DELTA
EDGE_FOLD_COUNT = SATURATION_EDGE_FOLD_COUNT
MATCHED_CONTROL_SEED = 3_101_002
DEFAULT_CLAIM_BOUNDARIES = {
    "one_checkpoint_and_one_behavior": True,
    "architecture_native_sublayers_not_learned_features": True,
    "conditional_second_differences_not_independent_causal_paths": True,
    "higher_order_effects_can_contribute_to_pair_differences": True,
    "summed_edge_energy_is_not_an_orthogonal_output_variance_decomposition": True,
    "saturation_null_selected_on_discovery_only": True,
    "exhaustive_pair_evaluation_does_not_establish_query_efficiency": True,
    "predictive_and_topology_claims_deferred": True,
}


def _as_double_tensor(value: Any, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu().double()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must contain only finite values")
    return tensor


def _validated_quartets(value: Any, *, label: str) -> torch.Tensor:
    quartets = _as_double_tensor(value, label=label)
    if quartets.ndim not in (3, 4) or quartets.shape[0] < 1 or quartets.shape[1] < 2:
        raise ValueError(f"{label} must have shape (edges, contexts, covers, 4)")
    if quartets.shape[-1] != 4:
        raise ValueError(f"{label} must use ordered cells 00, 10, 01, 11")
    if quartets.ndim == 3:
        quartets = quartets.unsqueeze(2)
    if quartets.shape[2] < 1:
        raise ValueError(f"{label} must contain at least one background cover")
    return quartets


def _validated_curvature_grid(values: Sequence[float]) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("curvature_grid must be a sequence")
    grid = tuple(float(value) for value in values)
    if (
        not grid
        or any(not math.isfinite(value) or value < 0.0 for value in grid)
        or tuple(sorted(set(grid))) != grid
    ):
        raise ValueError(
            "curvature_grid must be nonempty, finite, unique, nonnegative, and sorted"
        )
    if 0.0 not in grid:
        raise ValueError("curvature_grid must include the identity value 0")
    return grid


def _validated_pairs(
    pairs: Sequence[Sequence[int]], *, edge_count: int, label: str
) -> tuple[tuple[int, int], ...]:
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    normalized = []
    for pair in pairs:
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(f"{label} must contain two-component pairs")
        left, right = pair
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not isinstance(left, int)
            or not isinstance(right, int)
            or left < 0
            or left >= right
        ):
            raise ValueError(f"{label} pairs must be ordered nonnegative integers")
        normalized.append((left, right))
    result = tuple(normalized)
    if len(result) != edge_count:
        raise ValueError(f"{label} length must equal the quartet edge count")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must contain unique pairs")
    return result


def _validated_strata(
    strata: Sequence[Hashable], *, context_count: int, label: str
) -> tuple[Hashable, ...]:
    if not isinstance(strata, Sequence) or isinstance(strata, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(strata)
    if len(result) != context_count:
        raise ValueError(f"{label} length must equal the context count")
    try:
        unique_count = len(set(result))
    except TypeError as error:
        raise TypeError(f"{label} entries must be hashable") from error
    if unique_count < 1:
        raise ValueError(f"{label} must be nonempty")
    return result


def raw_second_difference(quartets: Any) -> torch.Tensor:
    """Return exact ``R11 - R10 - R01 + R00`` effects.

    The input can have arbitrary leading dimensions, but its final dimension
    must be the registered cell order ``00, 10, 01, 11``.
    """
    values = _as_double_tensor(quartets, label="quartets")
    if values.ndim < 1 or values.shape[-1] != 4:
        raise ValueError("quartets must end in ordered cells 00, 10, 01, 11")
    return values[..., 3] - values[..., 1] - values[..., 2] + values[..., 0]


def arcsinh_forward(latent: Any, curvature: float) -> torch.Tensor:
    """Apply the endpoint-normalized, monotone, unbounded arcsinh link.

    ``g_lambda(z) = asinh(lambda*z) / asinh(lambda)`` fixes zero and one.
    The registered zero-curvature member is the identity map.
    """
    values = _as_double_tensor(latent, label="latent values")
    parameter = float(curvature)
    if not math.isfinite(parameter) or parameter < 0.0:
        raise ValueError("curvature must be finite and nonnegative")
    if parameter == 0.0:
        return values.clone()
    output = torch.asinh(parameter * values) / math.asinh(parameter)
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError("arcsinh link produced a nonfinite value")
    return output


def _arcsinh_inverse(response: torch.Tensor, curvature: float) -> torch.Tensor:
    if curvature == 0.0:
        return response.clone()
    output = torch.sinh(response * math.asinh(curvature)) / curvature
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError("inverse arcsinh link produced a nonfinite value")
    return output


def arcsinh_joint_prediction(quartets: Any, curvature: float) -> torch.Tensor:
    """Predict ``R11`` from the other three cells under link-scale additivity."""
    values = _as_double_tensor(quartets, label="quartets")
    if values.ndim < 1 or values.shape[-1] != 4:
        raise ValueError("quartets must end in ordered cells 00, 10, 01, 11")
    parameter = float(curvature)
    if not math.isfinite(parameter) or parameter < 0.0:
        raise ValueError("curvature must be finite and nonnegative")
    latent_00 = _arcsinh_inverse(values[..., 0], parameter)
    latent_10 = _arcsinh_inverse(values[..., 1], parameter)
    latent_01 = _arcsinh_inverse(values[..., 2], parameter)
    latent_11 = latent_10 + latent_01 - latent_00
    return arcsinh_forward(latent_11, parameter)


def _arcsinh_residual(quartets: torch.Tensor, curvature: float) -> torch.Tensor:
    return quartets[..., 3] - arcsinh_joint_prediction(quartets, curvature)


def _huber_loss(values: torch.Tensor, *, delta: float = HUBER_DELTA) -> torch.Tensor:
    absolute = values.abs()
    return torch.where(
        absolute <= delta,
        0.5 * values.square(),
        delta * (absolute - 0.5 * delta),
    )


def _select_curvature(
    quartets: torch.Tensor, curvature_grid: tuple[float, ...]
) -> tuple[float, dict[str, float]]:
    losses = {
        str(curvature): float(
            _huber_loss(_arcsinh_residual(quartets, curvature)).mean()
        )
        for curvature in curvature_grid
    }
    # The grid is sorted, so this deterministic key chooses the simpler,
    # lower-curvature null on exact loss ties.
    selected = min(curvature_grid, key=lambda value: (losses[str(value)], value))
    return selected, losses


def _edge_fold(pair: tuple[int, int]) -> int:
    edge_class = "hub_spoke" if HUB_COMPONENT in pair else "nonhub_pair"
    payload = (f"{DESIGN_DOMAIN}|lambda-fold|{edge_class}|{pair[0]}|{pair[1]}").encode(
        "ascii"
    )
    return int(hashlib.sha256(payload).hexdigest(), 16) % EDGE_FOLD_COUNT


def crossfit_arcsinh_residuals(
    quartets: Any,
    *,
    context_strata: Sequence[Hashable],
    pairs: Sequence[Sequence[int]] | None = None,
    curvature_grid: Sequence[float] = DEFAULT_CURVATURE_GRID,
) -> dict[str, Any]:
    """Cross-fit the null by both held template and response-blind edge fold.

    For every held edge-fold/template combination, lambda is selected using
    only rows with a different template and edge fold.  Component identities
    only assign response-blind folds; the null itself sees response geometry
    but no identity feature.  One final lambda is then fitted on every
    discovery row and frozen for confirmation.
    """
    values = _validated_quartets(quartets, label="quartets")
    strata = _validated_strata(
        context_strata,
        context_count=values.shape[1],
        label="context_strata",
    )
    grid = _validated_curvature_grid(curvature_grid)
    if pairs is None:
        edge_folds = tuple(
            index % min(EDGE_FOLD_COUNT, values.shape[0])
            for index in range(values.shape[0])
        )
        normalized_pairs = None
    else:
        normalized_pairs = _validated_pairs(
            pairs, edge_count=values.shape[0], label="pairs"
        )
        edge_folds = tuple(_edge_fold(pair) for pair in normalized_pairs)
    unique_templates = tuple(dict.fromkeys(strata))
    unique_edge_folds = tuple(dict.fromkeys(edge_folds))
    if len(unique_templates) < 2 or len(unique_edge_folds) < 2:
        raise ValueError(
            "cross-fitting requires at least two templates and two edge folds"
        )

    cross_fitted = torch.empty(values.shape[:3], dtype=torch.float64)
    fold_curvatures: list[float] = []
    fold_records: list[dict[str, Any]] = []
    for template in unique_templates:
        held_contexts = [
            index for index, label in enumerate(strata) if label == template
        ]
        train_contexts = [
            index for index, label in enumerate(strata) if label != template
        ]
        for edge_fold in unique_edge_folds:
            held_edges = [
                index for index, fold in enumerate(edge_folds) if fold == edge_fold
            ]
            train_edges = [
                index for index, fold in enumerate(edge_folds) if fold != edge_fold
            ]
            if (
                not held_edges
                or not train_edges
                or not held_contexts
                or not train_contexts
            ):
                raise ValueError("cross-fitting requires nonempty train and held rows")
            training_rows = values[train_edges][:, train_contexts]
            selected, losses = _select_curvature(training_rows, grid)
            held_rows = values[held_edges][:, held_contexts]
            held_residuals = _arcsinh_residual(held_rows, selected)
            for held_edge_offset, edge_index in enumerate(held_edges):
                cross_fitted[edge_index, held_contexts] = held_residuals[
                    held_edge_offset
                ]
            fold_curvatures.append(selected)
            fold_records.append(
                {
                    "template": str(template),
                    "edge_fold": edge_fold,
                    "training_edge_count": len(train_edges),
                    "held_edge_count": len(held_edges),
                    "training_context_count": len(train_contexts),
                    "held_context_count": len(held_contexts),
                    "selected_curvature": selected,
                    "training_huber_loss_by_curvature": losses,
                }
            )

    refit_curvature, refit_losses = _select_curvature(values, grid)
    return {
        "cross_fitted_residuals": cross_fitted,
        "fold_curvatures": tuple(fold_curvatures),
        "fold_records": tuple(fold_records),
        "refit_curvature": refit_curvature,
        "refit_huber_loss_by_curvature": refit_losses,
        "curvature_grid": grid,
        "huber_delta": HUBER_DELTA,
        "edge_fold_count": EDGE_FOLD_COUNT,
        "edge_fold_ids": edge_folds,
        "edge_pairs_supplied": normalized_pairs is not None,
    }


def _edge_energy(effects: torch.Tensor) -> torch.Tensor:
    if effects.ndim not in (2, 3) or any(size < 1 for size in effects.shape):
        raise ValueError("effects must have shape (edges, contexts[, covers])")
    return effects.square().mean(dim=tuple(range(1, effects.ndim)))


def _ordered_indices(
    energy: torch.Tensor, pairs: tuple[tuple[int, int], ...]
) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(len(pairs)),
            key=lambda index: (-float(energy[index]), pairs[index]),
        )
    )


def _safe_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    centered_left = left - left.mean()
    centered_right = right - right.mean()
    denominator = float(
        torch.sqrt(centered_left.square().sum() * centered_right.square().sum())
    )
    if denominator == 0.0:
        return float(torch.allclose(left, right, atol=1e-15, rtol=0.0))
    return float((centered_left * centered_right).sum() / denominator)


def _rank_positions(order: Sequence[int], count: int) -> torch.Tensor:
    positions = torch.empty(count, dtype=torch.float64)
    for rank, index in enumerate(order):
        positions[index] = float(rank)
    return positions


def summarize_energy(
    discovery_energy: Any,
    confirmation_energy: Any,
    *,
    pairs: Sequence[Sequence[int]],
    top_counts: Sequence[int] = (24, 48),
) -> dict[str, Any]:
    """Score a discovery-frozen edge order on all confirmation edges."""
    discovery = _as_double_tensor(discovery_energy, label="discovery_energy")
    confirmation = _as_double_tensor(confirmation_energy, label="confirmation_energy")
    if (
        discovery.ndim != 1
        or confirmation.ndim != 1
        or discovery.shape != confirmation.shape
        or len(discovery) < 1
    ):
        raise ValueError("energy vectors must be aligned nonempty vectors")
    if bool((discovery < 0).any()) or bool((confirmation < 0).any()):
        raise ValueError("energy vectors must be nonnegative")
    normalized_pairs = _validated_pairs(pairs, edge_count=len(discovery), label="pairs")
    counts = tuple(int(value) for value in top_counts)
    if not counts or any(value < 1 for value in counts):
        raise ValueError("top_counts must contain positive integers")

    discovery_order = _ordered_indices(discovery, normalized_pairs)
    confirmation_order = _ordered_indices(confirmation, normalized_pairs)
    total = float(confirmation.sum())
    squared_total = float(confirmation.square().sum())
    coverage: dict[str, float] = {}
    oracle_efficiency: dict[str, float] = {}
    for requested_count in counts:
        count = min(requested_count, len(discovery))
        frozen_value = float(confirmation[list(discovery_order[:count])].sum())
        oracle_value = float(confirmation[list(confirmation_order[:count])].sum())
        coverage[str(requested_count)] = frozen_value / total if total > 0.0 else 0.0
        oracle_efficiency[str(requested_count)] = (
            frozen_value / oracle_value if oracle_value > 0.0 else 0.0
        )

    ordered_confirmation = confirmation[list(discovery_order)]
    if total > 0.0:
        cumulative = torch.cumsum(ordered_confirmation, dim=0) / total
        recovered_k90 = int(torch.nonzero(cumulative >= 0.90)[0]) + 1
        effective_support = total * total / squared_total
    else:
        recovered_k90 = None
        effective_support = 0.0

    discovery_positions = _rank_positions(discovery_order, len(discovery))
    confirmation_positions = _rank_positions(confirmation_order, len(discovery))
    return {
        "edge_count": len(discovery),
        "confirmation_total_energy": total,
        "discovery_order_indices": list(discovery_order),
        "discovery_order_pairs": [list(normalized_pairs[i]) for i in discovery_order],
        "confirmation_oracle_order_indices": list(confirmation_order),
        "confirmation_oracle_order_pairs": [
            list(normalized_pairs[i]) for i in confirmation_order
        ],
        "coverage_by_top_k": coverage,
        "oracle_efficiency_by_top_k": oracle_efficiency,
        "recovered_k90": recovered_k90,
        "effective_support": effective_support,
        "energy_correlation": _safe_correlation(discovery, confirmation),
        "rank_spearman_correlation": _safe_correlation(
            discovery_positions, confirmation_positions
        ),
    }


def _bootstrap_indices(
    strata: tuple[Hashable, ...],
    cluster_ids: tuple[Hashable, ...],
    *,
    replicates: int,
    seed: int,
) -> torch.Tensor:
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 1
    ):
        raise ValueError("bootstrap_replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    cluster_members: dict[Hashable, list[int]] = {}
    cluster_strata: dict[Hashable, Hashable] = {}
    for index, (stratum, cluster_id) in enumerate(
        zip(strata, cluster_ids, strict=True)
    ):
        if cluster_id in cluster_strata and cluster_strata[cluster_id] != stratum:
            raise ValueError("one bootstrap cluster cannot span template strata")
        cluster_strata[cluster_id] = stratum
        cluster_members.setdefault(cluster_id, []).append(index)
    cluster_sizes = {len(members) for members in cluster_members.values()}
    if len(cluster_sizes) != 1:
        raise ValueError("bootstrap clusters must all have the same row count")
    groups: dict[Hashable, list[Hashable]] = {}
    for cluster_id, stratum in cluster_strata.items():
        groups.setdefault(stratum, []).append(cluster_id)
    sampled_groups = []
    for clusters in groups.values():
        draws = torch.randint(
            len(clusters), (replicates, len(clusters)), generator=generator
        )
        rows = torch.tensor(
            [cluster_members[cluster] for cluster in clusters], dtype=torch.long
        )
        sampled = rows[draws].reshape(replicates, -1)
        sampled_groups.append(sampled)
    return torch.cat(sampled_groups, dim=1)


def _quantile(values: torch.Tensor, probability: float) -> float:
    return float(torch.quantile(values.double(), probability))


def _bootstrap_energy_metrics(
    confirmation_residuals: torch.Tensor,
    discovery_energy: torch.Tensor,
    discovery_signed_mean: torch.Tensor,
    discovery_order: tuple[int, ...],
    *,
    context_strata: tuple[Hashable, ...],
    context_cluster_ids: tuple[Hashable, ...],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    indices = _bootstrap_indices(
        context_strata,
        context_cluster_ids,
        replicates=replicates,
        seed=seed,
    )
    chunk_size = 128
    top24 = list(discovery_order[: min(24, len(discovery_order))])
    top48 = list(discovery_order[: min(48, len(discovery_order))])
    coverage24 = torch.empty(replicates, dtype=torch.float64)
    coverage48 = torch.empty(replicates, dtype=torch.float64)
    k90 = torch.empty(replicates, dtype=torch.float64)
    effective_support = torch.empty(replicates, dtype=torch.float64)
    correlations = torch.empty(replicates, dtype=torch.float64)
    centered_discovery = discovery_signed_mean - discovery_signed_mean.mean()
    discovery_sum_squares = centered_discovery.square().sum()
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        # This is the only large temporary: E x chunk x C x covers.  Chunking
        # prevents the registered 10k bootstrap from materializing ~2.6 GB.
        boot_rows = confirmation_residuals[:, indices[start:stop]]
        boot_energy = boot_rows.square().mean(dim=(2, 3)).T
        boot_signed_mean = boot_rows.mean(dim=(2, 3)).T
        total = boot_energy.sum(dim=1)
        safe_total = torch.where(total > 0.0, total, torch.ones_like(total))
        coverage24[start:stop] = boot_energy[:, top24].sum(dim=1) / safe_total
        coverage48[start:stop] = boot_energy[:, top48].sum(dim=1) / safe_total
        ordered = boot_energy[:, list(discovery_order)]
        cumulative = torch.cumsum(ordered, dim=1) / safe_total[:, None]
        crosses = cumulative >= 0.90
        chunk_k90 = torch.argmax(crosses.to(torch.int64), dim=1).double() + 1.0
        k90[start:stop] = torch.where(
            total > 0.0,
            chunk_k90,
            torch.full_like(chunk_k90, float("nan")),
        )
        denominator = boot_energy.square().sum(dim=1)
        effective_support[start:stop] = torch.where(
            denominator > 0.0,
            total.square() / denominator,
            torch.zeros_like(total),
        )
        centered_boot = boot_signed_mean - boot_signed_mean.mean(dim=1, keepdim=True)
        numerator = (centered_boot * centered_discovery[None, :]).sum(dim=1)
        correlation_denominator = torch.sqrt(
            centered_boot.square().sum(dim=1) * discovery_sum_squares
        )
        correlations[start:stop] = torch.where(
            correlation_denominator > 0.0,
            numerator / correlation_denominator,
            torch.zeros_like(numerator),
        )
    finite_k90 = k90[torch.isfinite(k90)]
    return {
        "replicates": replicates,
        "seed": seed,
        "resample_unit": "semantic_frame_with_both_directions",
        "cluster_unit": "provided_semantic_frame_cluster",
        "stratified_by": "provided_template_context_strata",
        "common_resamples_for_all_edges": True,
        "deterministic_chunk_size": chunk_size,
        "saturation_curvature_refit_inside_bootstrap": False,
        "coverage_top24_one_sided_95_lower": _quantile(coverage24, 0.05),
        "coverage_top48_one_sided_95_lower": _quantile(coverage48, 0.05),
        "signed_interaction_pearson_one_sided_95_lower": _quantile(correlations, 0.05),
        "recovered_k90_one_sided_95_upper": (
            _quantile(finite_k90, 0.95) if len(finite_k90) else None
        ),
        "effective_support_one_sided_95_upper": _quantile(effective_support, 0.95),
    }


def _score_controls(
    control_supports: Mapping[str, Sequence[Sequence[int]]] | None,
    confirmation_energy: torch.Tensor,
    *,
    primary_coverage: float,
    quantile: float,
) -> dict[str, Any]:
    if control_supports is None:
        return {
            "available": False,
            "families": {},
            "minimum_family_support_count": 0,
            "beats_every_control_top24_coverage": False,
        }
    if not isinstance(control_supports, Mapping):
        raise TypeError("control_supports must be a mapping")
    total = float(confirmation_energy.sum())
    families: dict[str, Any] = {}
    passes = []
    for family, supports in control_supports.items():
        if not isinstance(family, str) or not family:
            raise ValueError("control family identifiers must be nonempty strings")
        if not isinstance(supports, Sequence) or not supports:
            raise ValueError("every control family must contain at least one support")
        values = []
        normalized_supports = []
        for support in supports:
            indices = tuple(int(value) for value in support)
            if (
                not indices
                or len(indices) != len(set(indices))
                or any(
                    index < 0 or index >= len(confirmation_energy) for index in indices
                )
            ):
                raise ValueError(
                    "control supports must contain unique valid edge indices"
                )
            normalized_supports.append(list(indices))
            value = float(confirmation_energy[list(indices)].sum())
            values.append(value / total if total > 0.0 else 0.0)
        tensor = torch.tensor(values, dtype=torch.float64)
        null_quantile = _quantile(tensor, quantile)
        passed = primary_coverage > null_quantile
        passes.append(passed)
        families[family] = {
            "support_count": len(values),
            "supports": normalized_supports,
            "coverage_values": values,
            "tested_quantile": quantile,
            "coverage_tested_quantile": null_quantile,
            "primary_minus_tested_quantile": primary_coverage - null_quantile,
            "passed": passed,
        }
    return {
        "available": bool(families),
        "families": families,
        "minimum_family_support_count": min(
            record["support_count"] for record in families.values()
        ),
        "beats_every_control_top24_coverage": bool(passes) and all(passes),
    }


def _component_strength_quartiles(
    quartets: torch.Tensor,
    pairs: tuple[tuple[int, int], ...],
) -> tuple[dict[int, float], dict[int, int]]:
    """Pool squared endpoint marginals under both incident-partner states."""
    squared_marginals: dict[int, list[torch.Tensor]] = {}
    for edge_index, (first, second) in enumerate(pairs):
        cells = quartets[edge_index]
        first_off = cells[..., 1] - cells[..., 0]
        first_on = cells[..., 3] - cells[..., 2]
        second_off = cells[..., 2] - cells[..., 0]
        second_on = cells[..., 3] - cells[..., 1]
        squared_marginals.setdefault(first, []).extend(
            (first_off.square().reshape(-1), first_on.square().reshape(-1))
        )
        squared_marginals.setdefault(second, []).extend(
            (second_off.square().reshape(-1), second_on.square().reshape(-1))
        )
    strengths = {
        component: float(torch.cat(rows).mean())
        for component, rows in squared_marginals.items()
    }
    ordered_components = sorted(
        strengths, key=lambda component: (strengths[component], component)
    )
    count = len(ordered_components)
    quartiles = {
        component: min(3, rank * 4 // count)
        for rank, component in enumerate(ordered_components)
    }
    return strengths, quartiles


def _edge_matching_bin(
    pair: tuple[int, int], component_quartiles: Mapping[int, int]
) -> tuple[str, str, int, int]:
    kinds = tuple("A" if component % 2 == 0 else "M" for component in pair)
    pair_kind = "".join(sorted(kinds))
    layer_distance = abs(pair[0] // 2 - pair[1] // 2)
    if layer_distance <= 2:
        distance_bin = "0_2"
    elif layer_distance <= 5:
        distance_bin = "3_5"
    else:
        distance_bin = "6_plus"
    endpoint_quartiles = tuple(
        sorted(component_quartiles[component] for component in pair)
    )
    return pair_kind, distance_bin, endpoint_quartiles[0], endpoint_quartiles[1]


def build_matched_control_supports(
    *,
    pairs: Sequence[Sequence[int]],
    discovery_quartets: Any,
    selected_indices: Sequence[int],
    support_count: int | None = None,
) -> dict[str, Any]:
    """Build deterministic discovery-only matched controls for one support.

    Matching is joint over attention/MLP pair kind, layer-distance bin, and the
    unordered endpoint strength-quartile pair.  Strength uses both endpoint
    marginal effects for every incident partner state.  Confirmation outcomes
    are deliberately absent from this API.
    """
    quartets = _validated_quartets(discovery_quartets, label="discovery_quartets")
    normalized_pairs = _validated_pairs(
        pairs, edge_count=quartets.shape[0], label="pairs"
    )
    selected = tuple(int(index) for index in selected_indices)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(index < 0 or index >= len(normalized_pairs) for index in selected)
    ):
        raise ValueError("selected_indices must be unique valid edge indices")
    count = (
        int(DEFAULT_PRIMARY_GATES["matched_random_support_count"])
        if support_count is None
        else support_count
    )
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("support_count must be a positive integer")

    strengths, quartiles = _component_strength_quartiles(quartets, normalized_pairs)
    bins_by_edge = tuple(
        _edge_matching_bin(pair, quartiles) for pair in normalized_pairs
    )
    candidates_by_bin: dict[tuple[str, str, int, int], list[int]] = {}
    for edge_index, matching_bin in enumerate(bins_by_edge):
        candidates_by_bin.setdefault(matching_bin, []).append(edge_index)
    required_by_bin: dict[tuple[str, str, int, int], int] = {}
    for edge_index in selected:
        matching_bin = bins_by_edge[edge_index]
        required_by_bin[matching_bin] = required_by_bin.get(matching_bin, 0) + 1
    for matching_bin, required in required_by_bin.items():
        if len(candidates_by_bin[matching_bin]) < required:
            raise RuntimeError("a matched-control bin has too few candidate edges")

    supports = []
    seen_supports: set[tuple[int, ...]] = set()
    selected_support = tuple(sorted(selected))
    salt = 0
    maximum_attempts = max(100_000, 100 * count)
    while len(supports) < count and salt < maximum_attempts:
        support = []
        for matching_bin in sorted(required_by_bin, key=str):
            ranked = sorted(
                candidates_by_bin[matching_bin],
                key=lambda edge_index: hashlib.sha256(
                    (
                        f"{MATCHED_CONTROL_SEED}|{salt}|{matching_bin}|"
                        f"{normalized_pairs[edge_index][0]}|"
                        f"{normalized_pairs[edge_index][1]}"
                    ).encode("ascii")
                ).hexdigest(),
            )
            support.extend(ranked[: required_by_bin[matching_bin]])
        normalized_support = tuple(sorted(support))
        if len(normalized_support) != len(set(normalized_support)) or len(
            normalized_support
        ) != len(selected):
            raise RuntimeError("a generated matched support has invalid cardinality")
        if (
            normalized_support != selected_support
            and normalized_support not in seen_supports
        ):
            seen_supports.add(normalized_support)
            supports.append(normalized_support)
        salt += 1
    if len(supports) != count:
        raise RuntimeError(
            "the selected joint-bin profile cannot produce the registered number "
            "of unique matched supports"
        )

    selected_bin_counts = {str(key): value for key, value in required_by_bin.items()}
    all_match = all(
        {
            str(key): sum(bins_by_edge[index] == key for index in support)
            for key in required_by_bin
        }
        == selected_bin_counts
        for support in supports
    )
    if not all_match:
        raise RuntimeError("a generated support violated the frozen matching bins")
    return {
        "schema_version": "e31_1_discovery_matched_controls_v1",
        "generator": "sha256_rank_within_joint_matching_bins",
        "seed": MATCHED_CONTROL_SEED,
        "candidate_salts_examined": salt,
        "uses_confirmation_outcomes": False,
        "support_count": count,
        "support_size": len(selected),
        "selected_indices": list(selected),
        "selected_support_excluded": all(
            support != selected_support for support in supports
        ),
        "component_strength_definition": (
            "mean_squared_endpoint_marginal_over_both_partner_states_incident_edges_"
            "contexts_and_covers"
        ),
        "component_strengths": {
            str(component): value for component, value in strengths.items()
        },
        "component_strength_quartiles": {
            str(component): value for component, value in quartiles.items()
        },
        "selected_joint_bin_counts": selected_bin_counts,
        "supports": supports,
        "supports_are_unique": len(supports) == len(set(supports)),
        "all_supports_match_joint_bin_counts": all_match,
    }


def _evaluate_primary_gate(
    energy_summary: Mapping[str, Any],
    stability: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    raw_residual_overlap: Mapping[str, Any],
    controls: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "top24_confirmation_coverage": (
            energy_summary["coverage_by_top_k"]["24"]
            >= gates["top24_confirmation_energy_coverage_minimum"]
        ),
        "top24_oracle_efficiency": (
            energy_summary["oracle_efficiency_by_top_k"]["24"]
            >= gates["top24_fraction_of_confirmation_oracle_coverage_minimum"]
        ),
        "top48_confirmation_coverage": (
            energy_summary["coverage_by_top_k"]["48"]
            >= gates["top48_confirmation_energy_coverage_minimum"]
        ),
        "top24_sign_replication": (
            stability["sign_replication_count_top24"]
            >= gates["replicated_top24_sign_count_minimum"]
        ),
        "discovery_confirmation_signed_interaction_pearson": (
            stability["signed_interaction_pearson"]
            >= gates["discovery_confirmation_signed_interaction_pearson_minimum"]
        ),
        "bootstrap_top24_coverage_lower": (
            bootstrap["coverage_top24_one_sided_95_lower"]
            >= gates["top24_coverage_one_sided_95_lower_bound_minimum"]
        ),
        "bootstrap_signed_interaction_correlation_lower": (
            bootstrap["signed_interaction_pearson_one_sided_95_lower"]
            > gates[
                "signed_interaction_pearson_one_sided_95_lower_bound_strict_minimum"
            ]
        ),
        "raw_residual_overlap": (
            raw_residual_overlap["raw_top48_contains_residual_top24"]
            >= gates["residual_top24_in_raw_top48_count_minimum"]
        ),
        "matched_random_controls": (
            controls["beats_every_control_top24_coverage"]
            and controls["minimum_family_support_count"]
            >= gates["matched_random_support_count"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": deepcopy(dict(gates)),
        "all_checks_required": True,
    }


def _edge_records(
    pairs: tuple[tuple[int, int], ...],
    discovery_raw: torch.Tensor,
    confirmation_raw: torch.Tensor,
    discovery_residual: torch.Tensor,
    confirmation_residual: torch.Tensor,
    discovery_order: Sequence[int],
    confirmation_order: Sequence[int],
) -> list[dict[str, Any]]:
    discovery_rank = {index: rank + 1 for rank, index in enumerate(discovery_order)}
    confirmation_rank = {
        index: rank + 1 for rank, index in enumerate(confirmation_order)
    }
    records = []
    for index, pair in enumerate(pairs):
        records.append(
            {
                "edge_index": index,
                "pair": list(pair),
                "discovery_raw_signed_mean": float(discovery_raw[index].mean()),
                "confirmation_raw_signed_mean": float(confirmation_raw[index].mean()),
                "discovery_raw_energy": float(discovery_raw[index].square().mean()),
                "confirmation_raw_energy": float(
                    confirmation_raw[index].square().mean()
                ),
                "discovery_residual_signed_mean": float(
                    discovery_residual[index].mean()
                ),
                "confirmation_residual_signed_mean": float(
                    confirmation_residual[index].mean()
                ),
                "discovery_residual_energy": float(
                    discovery_residual[index].square().mean()
                ),
                "confirmation_residual_energy": float(
                    confirmation_residual[index].square().mean()
                ),
                "discovery_residual_rank": discovery_rank[index],
                "confirmation_residual_oracle_rank": confirmation_rank[index],
            }
        )
    return records


def score_regime(
    *,
    regime_id: str,
    discovery_quartets: Any,
    confirmation_quartets: Any,
    pairs: Sequence[Sequence[int]],
    discovery_context_strata: Sequence[Hashable],
    confirmation_context_strata: Sequence[Hashable],
    discovery_context_cluster_ids: Sequence[Hashable] | None = None,
    confirmation_context_cluster_ids: Sequence[Hashable] | None = None,
    confirmation_pairs: Sequence[Sequence[int]] | None = None,
    curvature_grid: Sequence[float] = DEFAULT_CURVATURE_GRID,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    control_supports: Mapping[str, Sequence[Sequence[int]]] | None = None,
    primary_gates: Mapping[str, Any] = DEFAULT_PRIMARY_GATES,
    compute_inference: bool = True,
) -> dict[str, Any]:
    """Score one fixed edge regime without reading, writing, or model calls."""
    if not isinstance(regime_id, str) or not regime_id:
        raise ValueError("regime_id must be a nonempty string")
    discovery = _validated_quartets(discovery_quartets, label="discovery_quartets")
    confirmation = _validated_quartets(
        confirmation_quartets, label="confirmation_quartets"
    )
    if discovery.shape[0] != confirmation.shape[0]:
        raise ValueError("discovery and confirmation must have the same edge count")
    normalized_pairs = _validated_pairs(
        pairs, edge_count=discovery.shape[0], label="pairs"
    )
    normalized_confirmation_pairs = _validated_pairs(
        confirmation_pairs if confirmation_pairs is not None else pairs,
        edge_count=confirmation.shape[0],
        label="confirmation_pairs",
    )
    if normalized_pairs != normalized_confirmation_pairs:
        raise ValueError("discovery and confirmation must have the same ordered pairs")
    discovery_strata = _validated_strata(
        discovery_context_strata,
        context_count=discovery.shape[1],
        label="discovery_context_strata",
    )
    confirmation_strata = _validated_strata(
        confirmation_context_strata,
        context_count=confirmation.shape[1],
        label="confirmation_context_strata",
    )
    discovery_clusters = _validated_strata(
        (
            discovery_context_cluster_ids
            if discovery_context_cluster_ids is not None
            else tuple(range(discovery.shape[1]))
        ),
        context_count=discovery.shape[1],
        label="discovery_context_cluster_ids",
    )
    confirmation_clusters = _validated_strata(
        (
            confirmation_context_cluster_ids
            if confirmation_context_cluster_ids is not None
            else tuple(range(confirmation.shape[1]))
        ),
        context_count=confirmation.shape[1],
        label="confirmation_context_cluster_ids",
    )
    grid = _validated_curvature_grid(curvature_grid)
    if set(primary_gates) != set(DEFAULT_PRIMARY_GATES):
        raise ValueError("primary_gates must provide the complete frozen gate schema")

    raw_discovery = raw_second_difference(discovery)
    raw_confirmation = raw_second_difference(confirmation)
    saturation_fit = crossfit_arcsinh_residuals(
        discovery,
        context_strata=discovery_strata,
        pairs=normalized_pairs,
        curvature_grid=grid,
    )
    residual_discovery = saturation_fit["cross_fitted_residuals"]
    refit_curvature = saturation_fit["refit_curvature"]
    residual_confirmation = _arcsinh_residual(confirmation, refit_curvature)

    discovery_energy = _edge_energy(residual_discovery)
    confirmation_energy = _edge_energy(residual_confirmation)
    energy_summary = summarize_energy(
        discovery_energy,
        confirmation_energy,
        pairs=normalized_pairs,
        top_counts=(24, 48),
    )
    discovery_order = tuple(energy_summary["discovery_order_indices"])
    confirmation_order = tuple(energy_summary["confirmation_oracle_order_indices"])
    top24_count = min(24, len(normalized_pairs))
    discovery_top24 = set(discovery_order[:top24_count])
    confirmation_top24 = set(confirmation_order[:top24_count])
    intersection = discovery_top24 & confirmation_top24
    union = discovery_top24 | confirmation_top24
    discovery_means = residual_discovery.mean(dim=(1, 2))
    confirmation_means = residual_confirmation.mean(dim=(1, 2))
    sign_count = sum(
        float(discovery_means[index] * confirmation_means[index]) > 0.0
        for index in discovery_order[:top24_count]
    )
    stability = {
        "top24_intersection_count": len(intersection),
        "top24_jaccard": len(intersection) / len(union) if union else 1.0,
        "sign_replication_count_top24": sign_count,
        "sign_replication_denominator": top24_count,
        "energy_correlation": energy_summary["energy_correlation"],
        "signed_interaction_pearson": _safe_correlation(
            discovery_means, confirmation_means
        ),
        "rank_spearman_correlation": energy_summary["rank_spearman_correlation"],
    }

    raw_discovery_energy = _edge_energy(raw_discovery)
    raw_confirmation_energy = _edge_energy(raw_confirmation)
    raw_summary = summarize_energy(
        raw_discovery_energy,
        raw_confirmation_energy,
        pairs=normalized_pairs,
        top_counts=(24, 48),
    )
    raw_top48 = set(raw_summary["discovery_order_indices"][: min(48, len(pairs))])
    raw_residual_intersection = raw_top48 & discovery_top24
    raw_residual_overlap = {
        "raw_top48_contains_residual_top24": len(raw_residual_intersection),
        "residual_top24_denominator": top24_count,
        "overlap_edge_indices": sorted(raw_residual_intersection),
        "overlap_pairs": [
            list(normalized_pairs[index]) for index in sorted(raw_residual_intersection)
        ],
    }

    if compute_inference:
        bootstrap = _bootstrap_energy_metrics(
            residual_confirmation,
            discovery_energy,
            discovery_means,
            discovery_order,
            context_strata=confirmation_strata,
            context_cluster_ids=confirmation_clusters,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
    else:
        bootstrap = {
            "computed": False,
            "reason": "regime_is_registered_descriptive_only",
        }
    primary_coverage = energy_summary["coverage_by_top_k"]["24"]
    controls = _score_controls(
        control_supports,
        confirmation_energy,
        primary_coverage=primary_coverage,
        quantile=float(
            primary_gates["matched_random_support_quantile_strictly_exceeded"]
        ),
    )
    primary_gate = (
        _evaluate_primary_gate(
            energy_summary,
            stability,
            bootstrap,
            raw_residual_overlap,
            controls,
            primary_gates,
        )
        if compute_inference
        else {
            "applicable": False,
            "passed": None,
            "checks": {},
            "thresholds": {},
        }
    )
    saturation_record = {
        "family": "endpoint_normalized_arcsinh_link",
        "identity_nested_at_curvature_zero": True,
        "uses_component_identities": False,
        "uses_confirmation_R11_to_fit": False,
        "curvature_grid": list(saturation_fit["curvature_grid"]),
        "fold_curvatures": list(saturation_fit["fold_curvatures"]),
        "fold_records": list(saturation_fit["fold_records"]),
        "refit_curvature": refit_curvature,
        "refit_huber_loss_by_curvature": saturation_fit[
            "refit_huber_loss_by_curvature"
        ],
        "huber_delta": saturation_fit["huber_delta"],
        "edge_fold_count": saturation_fit["edge_fold_count"],
        "edge_fold_ids": list(saturation_fit["edge_fold_ids"]),
    }
    return {
        "schema_version": "e31_1_prospective_regime_score_v1",
        "regime_id": regime_id,
        "edge_count": len(normalized_pairs),
        "discovery_context_count": discovery.shape[1],
        "confirmation_context_count": confirmation.shape[1],
        "background_cover_count": discovery.shape[2],
        "discovery_semantic_frame_count": len(set(discovery_clusters)),
        "confirmation_semantic_frame_count": len(set(confirmation_clusters)),
        "raw_interaction_formula": "R11-R10-R01+R00",
        "raw_interactions_shape": list(raw_confirmation.shape),
        "residual_interaction_formula": "R11-arcsinh_link_additive_prediction",
        "saturation": saturation_record,
        "discovery": {
            "ranking_is_cross_fitted": True,
            "residual_energy": [float(value) for value in discovery_energy],
            "raw_energy": [float(value) for value in raw_discovery_energy],
        },
        "confirmation": energy_summary,
        "raw_confirmation": raw_summary,
        "stability": stability,
        "bootstrap": bootstrap,
        "raw_residual_overlap": raw_residual_overlap,
        "controls": controls,
        "primary_gate": primary_gate,
        "inferential_status": (
            "inferential_metrics_computed" if compute_inference else "descriptive_only"
        ),
        "edges": _edge_records(
            normalized_pairs,
            raw_discovery,
            raw_confirmation,
            residual_discovery,
            residual_confirmation,
            discovery_order,
            confirmation_order,
        ),
        "claim_boundaries": deepcopy(DEFAULT_CLAIM_BOUNDARIES),
    }


def _panel_value(panel: Mapping[str, Any], key: str, regime_id: str) -> Any:
    if key not in panel:
        raise ValueError(f"regime {regime_id!r} omits required field {key!r}")
    return panel[key]


def _semantic_frame_id(pair_id: str) -> str:
    for suffix in ("_a_to_b", "_b_to_a"):
        if pair_id.endswith(suffix):
            return pair_id[: -len(suffix)]
    raise ValueError("prompt pair_id must end in a registered IOI direction")


def assemble_regime_panels(
    *,
    responses: Any,
    split_manifest: Mapping[str, Any],
    prompt_records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Assemble canonical regime tensors from condition-indexed responses.

    The acquisition contract stores one response row per globally deduplicated
    condition.  This adapter follows each contrast's four condition IDs and
    produces the scorer's ``edge x context x cover x cell`` tensors without
    averaging or reordering an outcome.
    """
    if not isinstance(split_manifest, Mapping):
        raise TypeError("split_manifest must be a mapping")
    conditions = split_manifest.get("conditions")
    contrasts = split_manifest.get("contrasts")
    if not isinstance(conditions, Sequence) or not isinstance(contrasts, Sequence):
        raise TypeError("split_manifest must contain condition and contrast sequences")
    if not isinstance(prompt_records, Sequence) or not prompt_records:
        raise ValueError("prompt_records must be a nonempty sequence")
    context_strata = []
    context_cluster_ids = []
    for record in prompt_records:
        if not isinstance(record, Mapping):
            raise TypeError("every prompt record must be a mapping")
        stratum = record.get("stratum_id")
        pair_id = record.get("pair_id")
        if not isinstance(stratum, str) or not isinstance(pair_id, str):
            raise TypeError("prompt records must contain string stratum_id and pair_id")
        context_strata.append(stratum)
        context_cluster_ids.append(_semantic_frame_id(pair_id))

    values = _as_double_tensor(responses, label="responses")
    if values.ndim != 2 or tuple(values.shape) != (
        len(conditions),
        len(prompt_records),
    ):
        raise ValueError("responses must have shape (conditions, prompt_contexts)")
    condition_index_by_id: dict[str, int] = {}
    for expected_index, record in enumerate(conditions):
        if not isinstance(record, Mapping):
            raise TypeError("every condition record must be a mapping")
        condition_id = record.get("condition_id")
        if (
            record.get("condition_index") != expected_index
            or not isinstance(condition_id, str)
            or condition_id in condition_index_by_id
        ):
            raise ValueError(
                "condition indices and identifiers must be unique and ordered"
            )
        condition_index_by_id[condition_id] = expected_index

    grouped: dict[
        tuple[str, int, str], dict[tuple[int, int], list[Mapping[str, Any]]]
    ] = {}
    seen_contrast_ids = set()
    for expected_index, contrast in enumerate(contrasts):
        if not isinstance(contrast, Mapping):
            raise TypeError("every contrast record must be a mapping")
        contrast_id = contrast.get("contrast_id")
        if (
            contrast.get("contrast_index") != expected_index
            or not isinstance(contrast_id, str)
            or contrast_id in seen_contrast_ids
        ):
            raise ValueError(
                "contrast indices and identifiers must be unique and ordered"
            )
        seen_contrast_ids.add(contrast_id)
        if tuple(contrast.get("ordered_cell_bits", ())) != ("00", "10", "01", "11"):
            raise ValueError("every contrast must preserve the registered cell order")
        condition_ids = tuple(contrast.get("condition_ids", ()))
        if len(condition_ids) != 4 or any(
            condition_id not in condition_index_by_id for condition_id in condition_ids
        ):
            raise ValueError("a contrast references unknown or incomplete conditions")
        edge_class = contrast.get("edge_class")
        hub_state = contrast.get("hub_state")
        k = contrast.get("k_nonhub_on")
        # `factor_components` preserves cell-axis semantics and is hub-first
        # for spokes.  Pair identity uses the separately registered sorted
        # `edge`; condition_ids above retain the factor/cell order unchanged.
        pair_value = contrast.get("edge", contrast.get("factor_components"))
        pair = _validated_pairs((pair_value,), edge_count=1, label="contrast pair")[0]
        if edge_class not in ("nonhub_pair", "hub_spoke") or not isinstance(k, int):
            raise ValueError("a contrast has an invalid edge class or scale")
        if edge_class == "nonhub_pair" and hub_state not in ("off", "on"):
            raise ValueError("a nonhub contrast must have an off/on hub state")
        if edge_class == "hub_spoke" and hub_state != "crossing":
            raise ValueError("a hub-spoke contrast must cross the hub state")
        grouped.setdefault((edge_class, k, str(hub_state)), {}).setdefault(
            pair, []
        ).append(contrast)

    panels: dict[str, dict[str, Any]] = {}
    for (edge_class, k, hub_state), by_pair in grouped.items():
        if edge_class == "nonhub_pair":
            regime_id = f"nonhub_k{k:02d}_hub_{hub_state}"
        else:
            regime_id = f"spokes_k{k:02d}"
        pairs = tuple(sorted(by_pair))
        edge_tensors = []
        cover_ids = []
        for pair in pairs:
            covers = sorted(by_pair[pair], key=lambda row: int(row["cover_rank"]))
            cover_ranks = tuple(int(row["cover_rank"]) for row in covers)
            if cover_ranks != tuple(range(len(covers))) or len(covers) != 4:
                raise ValueError("every edge must have exactly four ordered covers")
            cover_cells = []
            edge_cover_ids = []
            for contrast in covers:
                indices = [
                    condition_index_by_id[condition_id]
                    for condition_id in contrast["condition_ids"]
                ]
                cover_cells.append(values[indices].T)
                edge_cover_ids.append(str(contrast["background_id"]))
            edge_tensors.append(torch.stack(cover_cells, dim=1))
            cover_ids.append(edge_cover_ids)
        panels[regime_id] = {
            "regime_id": regime_id,
            "edge_class": edge_class,
            "k_nonhub_on": k,
            "hub_state": hub_state,
            "pairs": pairs,
            "quartets": torch.stack(edge_tensors, dim=0),
            "context_strata": tuple(context_strata),
            "context_cluster_ids": tuple(context_cluster_ids),
            "background_ids_by_edge_and_cover": cover_ids,
        }
    return panels


def _registered_regime_ids() -> tuple[str, ...]:
    return tuple(
        f"nonhub_k{k:02d}_hub_{hub_state}"
        for k in SCALE_CARDINALITIES
        for hub_state in ("off", "on")
    ) + tuple(f"spokes_k{k:02d}" for k in SCALE_CARDINALITIES)


def _validate_registered_panels(
    panels: Mapping[str, Mapping[str, Any]], *, context_count: int, split: str
) -> None:
    if tuple(panels) != _registered_regime_ids():
        raise ValueError(f"the registered {split} regime order changed")
    expected_nonhub_pairs = tuple(combinations(NONHUB_COMPONENTS, 2))
    expected_spokes = tuple(
        tuple(sorted((HUB_COMPONENT, component))) for component in NONHUB_COMPONENTS
    )
    expected_spokes = tuple(sorted(expected_spokes))
    for regime_id, panel in panels.items():
        quartets = _validated_quartets(
            _panel_value(panel, "quartets", regime_id),
            label=f"{regime_id} quartets",
        )
        expected_pairs = (
            expected_spokes
            if regime_id.startswith("spokes_")
            else expected_nonhub_pairs
        )
        pairs = _validated_pairs(
            _panel_value(panel, "pairs", regime_id),
            edge_count=quartets.shape[0],
            label=f"{regime_id} pairs",
        )
        expected_edge_count = (
            SPOKE_COUNT if regime_id.startswith("spokes_") else NONHUB_EDGE_COUNT
        )
        if (
            pairs != expected_pairs
            or quartets.shape[0] != expected_edge_count
            or quartets.shape[1] != context_count
            or quartets.shape[2] != COVER_TARGET
        ):
            raise ValueError(f"the registered {split} {regime_id} tensor changed")
        clusters = _validated_strata(
            _panel_value(panel, "context_cluster_ids", regime_id),
            context_count=context_count,
            label=f"{regime_id} context_cluster_ids",
        )
        if len(set(clusters)) * 2 != context_count:
            raise ValueError("registered semantic frames must contain two directions")


def fit_discovery_geometry(
    *,
    discovery_panels: Mapping[str, Mapping[str, Any]],
    primary_regime_id: str,
    curvature_grid: Sequence[float] = DEFAULT_CURVATURE_GRID,
    primary_gates: Mapping[str, Any] = DEFAULT_PRIMARY_GATES,
    enforce_registered_design: bool = True,
) -> dict[str, Any]:
    """Fit and serialize every discovery-only quantity frozen before C opens."""
    if not isinstance(discovery_panels, Mapping) or not discovery_panels:
        raise TypeError("discovery_panels must be a nonempty mapping")
    if primary_regime_id not in discovery_panels:
        raise ValueError("primary_regime_id is absent from discovery_panels")
    grid = _validated_curvature_grid(curvature_grid)
    if set(primary_gates) != set(DEFAULT_PRIMARY_GATES):
        raise ValueError("primary_gates must provide the complete frozen gate schema")
    if enforce_registered_design:
        if (
            primary_regime_id != "nonhub_k10_hub_off"
            or grid != DEFAULT_CURVATURE_GRID
            or dict(primary_gates) != dict(DEFAULT_PRIMARY_GATES)
        ):
            raise ValueError("the registered discovery scoring constants changed")
        _validate_registered_panels(
            discovery_panels, context_count=16, split="discovery"
        )

    regimes: dict[str, Any] = {}
    for regime_id, panel in discovery_panels.items():
        if not isinstance(panel, Mapping):
            raise TypeError("every discovery regime must be a mapping")
        quartets = _validated_quartets(
            _panel_value(panel, "quartets", regime_id),
            label=f"{regime_id} discovery quartets",
        )
        pairs = _validated_pairs(
            _panel_value(panel, "pairs", regime_id),
            edge_count=quartets.shape[0],
            label=f"{regime_id} pairs",
        )
        strata = _validated_strata(
            _panel_value(panel, "context_strata", regime_id),
            context_count=quartets.shape[1],
            label=f"{regime_id} context_strata",
        )
        fit = crossfit_arcsinh_residuals(
            quartets,
            context_strata=strata,
            pairs=pairs,
            curvature_grid=grid,
        )
        residuals = fit["cross_fitted_residuals"]
        residual_energy = _edge_energy(residuals)
        raw_effects = raw_second_difference(quartets)
        raw_energy = _edge_energy(raw_effects)
        residual_order = _ordered_indices(residual_energy, pairs)
        raw_order = _ordered_indices(raw_energy, pairs)
        regime_record: dict[str, Any] = {
            "regime_id": regime_id,
            "pairs": [list(pair) for pair in pairs],
            "edge_count": len(pairs),
            "context_count": quartets.shape[1],
            "cover_count": quartets.shape[2],
            "context_strata": [str(value) for value in strata],
            "cross_fitted_residual_energy": [float(value) for value in residual_energy],
            "cross_fitted_residual_signed_mean": [
                float(value) for value in residuals.mean(dim=(1, 2))
            ],
            "cross_fitted_raw_energy": [float(value) for value in raw_energy],
            "residual_order_indices": list(residual_order),
            "residual_order_pairs": [list(pairs[index]) for index in residual_order],
            "raw_order_indices": list(raw_order),
            "refit_curvature": fit["refit_curvature"],
            "curvature_grid": list(fit["curvature_grid"]),
            "huber_delta": fit["huber_delta"],
            "edge_fold_count": fit["edge_fold_count"],
            "edge_fold_ids": list(fit["edge_fold_ids"]),
            "crossfit_records": list(fit["fold_records"]),
            "refit_huber_loss_by_curvature": fit["refit_huber_loss_by_curvature"],
        }
        if regime_id == primary_regime_id:
            controls = build_matched_control_supports(
                pairs=pairs,
                discovery_quartets=quartets,
                selected_indices=residual_order[: min(24, len(residual_order))],
                support_count=int(primary_gates["matched_random_support_count"]),
            )
            regime_record["matched_controls"] = controls
        else:
            regime_record["matched_controls"] = None
        regimes[regime_id] = regime_record
    return {
        "schema_version": "e31_1_frozen_discovery_geometry_v1",
        "status": "discovery_only_frozen_before_confirmation",
        "confirmation_outcomes_used": False,
        "primary_regime_id": primary_regime_id,
        "curvature_grid": list(grid),
        "primary_gates": deepcopy(dict(primary_gates)),
        "regimes": regimes,
        "claim_boundaries": deepcopy(DEFAULT_CLAIM_BOUNDARIES),
        "registered_design_enforced": enforce_registered_design,
    }


def score_with_frozen_geometry(
    *,
    discovery_panels: Mapping[str, Mapping[str, Any]],
    confirmation_panels: Mapping[str, Mapping[str, Any]],
    frozen_geometry: Mapping[str, Any],
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    enforce_registered_design: bool = True,
) -> dict[str, Any]:
    """Validate a discovery seal, then score confirmation without adapting it."""
    if not isinstance(frozen_geometry, Mapping):
        raise TypeError("frozen_geometry must be a mapping")
    geometry = deepcopy(dict(frozen_geometry))
    primary_regime_id = geometry.get("primary_regime_id")
    if not isinstance(primary_regime_id, str):
        raise TypeError("frozen_geometry omits a primary regime")
    grid = geometry.get("curvature_grid")
    gates = geometry.get("primary_gates")
    if enforce_registered_design:
        if (
            bootstrap_replicates != DEFAULT_BOOTSTRAP_REPLICATES
            or bootstrap_seed != DEFAULT_BOOTSTRAP_SEED
        ):
            raise ValueError("the registered bootstrap constants changed")
        _validate_registered_panels(
            discovery_panels, context_count=16, split="discovery"
        )
        _validate_registered_panels(
            confirmation_panels, context_count=32, split="confirmation"
        )
    recomputed = fit_discovery_geometry(
        discovery_panels=discovery_panels,
        primary_regime_id=primary_regime_id,
        curvature_grid=grid,
        primary_gates=gates,
        enforce_registered_design=enforce_registered_design,
    )
    if recomputed != geometry:
        raise RuntimeError("frozen discovery geometry differs from recomputation")
    if tuple(discovery_panels) != tuple(confirmation_panels):
        raise ValueError("discovery and confirmation regime order must match exactly")

    hierarchical_id = str(HIERARCHICAL_SCALE_GATE["regime_id"])
    scored_by_regime: dict[str, dict[str, Any]] = {}
    evaluation_order = (primary_regime_id,) + tuple(
        regime_id for regime_id in discovery_panels if regime_id != primary_regime_id
    )
    for regime_id in evaluation_order:
        discovery = discovery_panels[regime_id]
        confirmation = confirmation_panels[regime_id]
        geometry_regime = geometry["regimes"][regime_id]
        controls_record = geometry_regime["matched_controls"]
        control_supports = (
            {"joint_matched_sha256": controls_record["supports"]}
            if controls_record is not None
            else None
        )
        discovery_pairs = _panel_value(discovery, "pairs", regime_id)
        primary_has_passed = (
            bool(scored_by_regime[primary_regime_id]["primary_gate"]["passed"])
            if primary_regime_id in scored_by_regime
            else False
        )
        compute_inference = regime_id == primary_regime_id or (
            regime_id == hierarchical_id and primary_has_passed
        )
        scored_by_regime[regime_id] = score_regime(
            regime_id=regime_id,
            discovery_quartets=_panel_value(discovery, "quartets", regime_id),
            confirmation_quartets=_panel_value(confirmation, "quartets", regime_id),
            pairs=discovery_pairs,
            confirmation_pairs=confirmation.get("pairs", discovery_pairs),
            discovery_context_strata=_panel_value(
                discovery, "context_strata", regime_id
            ),
            confirmation_context_strata=_panel_value(
                confirmation, "context_strata", regime_id
            ),
            discovery_context_cluster_ids=discovery.get("context_cluster_ids"),
            confirmation_context_cluster_ids=confirmation.get("context_cluster_ids"),
            curvature_grid=grid,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            control_supports=control_supports,
            primary_gates=gates,
            compute_inference=compute_inference,
        )
        observed_geometry = scored_by_regime[regime_id]
        if (
            observed_geometry["saturation"]["refit_curvature"]
            != geometry_regime["refit_curvature"]
            or observed_geometry["confirmation"]["discovery_order_indices"]
            != geometry_regime["residual_order_indices"]
        ):
            raise RuntimeError("confirmation scoring adapted frozen geometry")
        if regime_id != primary_regime_id:
            # Secondary regimes remain complete descriptive outputs.  They do
            # not inherit the primary conjunction or expose extra verdicts.
            observed_geometry.pop("primary_gate")
        if regime_id == hierarchical_id and not primary_has_passed:
            observed_geometry["bootstrap"] = {
                "computed": False,
                "reason": "not_tested_due_to_primary_failure",
            }
            observed_geometry["inferential_status"] = (
                "descriptive_only_due_to_primary_failure"
            )

    regimes = {regime_id: scored_by_regime[regime_id] for regime_id in discovery_panels}

    primary = regimes[primary_regime_id]
    passed = bool(primary["primary_gate"]["passed"])
    if hierarchical_id not in regimes:
        if enforce_registered_design:
            raise RuntimeError("the registered hierarchical scale regime is absent")
        hierarchical_metrics = None
    else:
        hierarchical_metrics = regimes[hierarchical_id]
    if passed and hierarchical_metrics is not None:
        hierarchical_checks = {
            "top24_confirmation_energy_coverage": (
                hierarchical_metrics["confirmation"]["coverage_by_top_k"]["24"]
                >= HIERARCHICAL_SCALE_GATE["top24_confirmation_energy_coverage_minimum"]
            ),
            "top24_coverage_one_sided_95_lower_bound": (
                hierarchical_metrics["bootstrap"]["coverage_top24_one_sided_95_lower"]
                >= HIERARCHICAL_SCALE_GATE[
                    "top24_coverage_one_sided_95_lower_bound_minimum"
                ]
            ),
        }
        hierarchical_passed: bool | None = all(hierarchical_checks.values())
    else:
        hierarchical_checks = {}
        hierarchical_passed = None
    hierarchy_available = hierarchical_metrics is not None
    if not hierarchy_available:
        hierarchical_status = "not_available_in_generic_panel"
    elif hierarchical_passed is True:
        hierarchical_status = "pass"
    elif hierarchical_passed is False:
        hierarchical_status = "fail"
    else:
        hierarchical_status = "not_tested_due_to_primary_failure"
    hierarchical_gate = {
        "regime_id": hierarchical_id,
        "tested_only_if_primary_passes": True,
        "evaluated": passed and hierarchy_available,
        "passed": hierarchical_passed,
        "status": hierarchical_status,
        "checks": hierarchical_checks,
        "thresholds": deepcopy(dict(HIERARCHICAL_SCALE_GATE)),
    }
    if hierarchical_metrics is not None:
        hierarchical_metrics["hierarchical_gate"] = hierarchical_gate
    atlas_passed = passed and hierarchical_passed is True
    if atlas_passed:
        promotion_decision = "eligible_only_for_separately_registered_e31_2"
    elif not passed:
        promotion_decision = "primary_failed_no_hierarchical_test"
    elif not hierarchy_available:
        promotion_decision = "hierarchical_scale_not_available_in_generic_panel"
    else:
        promotion_decision = "primary_passed_but_hierarchical_scale_failed"
    return {
        "schema_version": "e31_1_prospective_atlas_score_v1",
        "status": "scored_without_model_calls",
        "frozen_geometry_recomputed_exactly": True,
        "confirmation_adaptation_allowed": False,
        "primary_regime_id": primary_regime_id,
        "primary_verdict": "pass" if passed else "fail",
        "hierarchical_scale_gate": hierarchical_gate,
        "atlas_verdict": "pass" if atlas_passed else "fail",
        "primary_gates": deepcopy(dict(gates)),
        "regimes": regimes,
        "claim_boundaries": deepcopy(DEFAULT_CLAIM_BOUNDARIES),
        "predictive_or_topology_claim_authorized": False,
        "task_or_model_scale_promotion_authorized": False,
        "promotion_decision": promotion_decision,
    }


def score_prospective_atlas(
    *,
    discovery_panels: Mapping[str, Mapping[str, Any]],
    confirmation_panels: Mapping[str, Mapping[str, Any]],
    primary_regime_id: str,
    curvature_grid: Sequence[float] = DEFAULT_CURVATURE_GRID,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    primary_gates: Mapping[str, Any] = DEFAULT_PRIMARY_GATES,
) -> dict[str, Any]:
    """Generic convenience wrapper; official custody uses the two-stage APIs."""
    if not isinstance(discovery_panels, Mapping) or not isinstance(
        confirmation_panels, Mapping
    ):
        raise TypeError("discovery_panels and confirmation_panels must be mappings")
    if tuple(discovery_panels) != tuple(confirmation_panels):
        raise ValueError("discovery and confirmation regime order must match exactly")
    if primary_regime_id not in discovery_panels:
        raise ValueError("primary_regime_id is absent from the panel mappings")
    geometry = fit_discovery_geometry(
        discovery_panels=discovery_panels,
        primary_regime_id=primary_regime_id,
        curvature_grid=curvature_grid,
        primary_gates=primary_gates,
        enforce_registered_design=False,
    )
    return score_with_frozen_geometry(
        discovery_panels=discovery_panels,
        confirmation_panels=confirmation_panels,
        frozen_geometry=geometry,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        enforce_registered_design=False,
    )


__all__ = [
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CLAIM_BOUNDARIES",
    "DEFAULT_CURVATURE_GRID",
    "DEFAULT_PRIMARY_GATES",
    "arcsinh_forward",
    "arcsinh_joint_prediction",
    "assemble_regime_panels",
    "build_matched_control_supports",
    "crossfit_arcsinh_residuals",
    "fit_discovery_geometry",
    "raw_second_difference",
    "score_prospective_atlas",
    "score_regime",
    "score_with_frozen_geometry",
    "summarize_energy",
]
