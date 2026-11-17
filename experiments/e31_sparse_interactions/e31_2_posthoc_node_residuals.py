# ruff: noqa: EXE002
"""Post-hoc node-versus-edge transfer diagnostic for the sealed E31.2 atlas.

This module is deliberately pure: it consumes already-acquired discovery and
confirmation label dictionaries and performs no model or filesystem calls.
Nothing here changes the preregistered E31.2 verdict.  Its narrow purpose is to
ask whether discovery head-pair energy contains confirmation-stable edge detail
that cannot be explained by separable importance of the two endpoint heads.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from experiments.e31_sparse_interactions.mixed_resolution_protocol import (
    COVER_COUNT,
    HEAD_COUNT,
    PARENT_PAIRS,
)

SCHEMA_VERSION = "e31_2_posthoc_node_residual_transfer_v1"
PARENT_IDS = tuple(parent["parent_pair_id"] for parent in PARENT_PAIRS)
DEFAULT_TOP_KS = (1, 2, 4, 8, 16)

CLAIM_BOUNDARIES = {
    "posthoc_after_e31_2_confirmation_was_seen": True,
    "development_diagnostic_only": True,
    "changes_e31_2_verdict": False,
    "uses_no_new_model_calls": True,
    "single_checkpoint_single_task": True,
    "energy_is_mean_squared_conditional_second_difference": True,
    "energy_is_not_an_orthogonal_causal_variance_decomposition": True,
    "residual_supports_are_not_direct_communication_edges": True,
}


def _finite_double(value: Any, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must be finite")
    return tensor


def _fine_quartets(labels: Mapping[str, Any]) -> torch.Tensor:
    if not isinstance(labels, Mapping):
        raise TypeError("labels must be a mapping")
    atlas = labels.get("atlas", labels)
    if not isinstance(atlas, Mapping) or "normalized_fine_quartets" not in atlas:
        raise KeyError("labels must contain normalized_fine_quartets")
    return _finite_double(
        atlas["normalized_fine_quartets"], label="normalized_fine_quartets"
    )


def raw_second_difference(quartets: Any) -> torch.Tensor:
    """Compute ``R11 - R10 - R01 + R00`` from ordered 00/10/01/11 cells."""
    values = _finite_double(quartets, label="quartets")
    if values.ndim < 1 or values.shape[-1] != 4:
        raise ValueError("quartets must end in four ordered cells")
    return values[..., 3] - values[..., 1] - values[..., 2] + values[..., 0]


def compute_split_maps(fine_quartets: Any) -> dict[str, torch.Tensor]:
    """Reduce a split's full quartets to interaction and singleton energy maps."""
    quartets = _finite_double(fine_quartets, label="fine_quartets")
    expected_prefix = (len(PARENT_IDS), HEAD_COUNT * HEAD_COUNT)
    if (
        quartets.ndim != 5
        or tuple(quartets.shape[:2]) != expected_prefix
        or quartets.shape[2] < 1
        or quartets.shape[3] != COVER_COUNT
        or quartets.shape[4] != 4
    ):
        raise ValueError(
            "fine_quartets must have shape (3, 144, contexts, 4 covers, "
            "4 ordered cells)"
        )

    cells = quartets.reshape(
        len(PARENT_IDS),
        HEAD_COUNT,
        HEAD_COUNT,
        quartets.shape[2],
        COVER_COUNT,
        4,
    )
    interaction = raw_second_difference(cells)
    left_effect = cells[..., 1] - cells[..., 0]
    right_effect = cells[..., 2] - cells[..., 0]
    return {
        "interaction_energy": interaction.square().mean(dim=(3, 4)),
        "signed_mean": interaction.mean(dim=(3, 4)),
        # Each singleton condition is repeated over the opposite head in the
        # atlas.  Pooling that redundant axis also tolerates tiny numerics.
        "left_singleton_energy": left_effect.square().mean(dim=(2, 3, 4)),
        "right_singleton_energy": right_effect.square().mean(dim=(1, 3, 4)),
        "left_singleton_signed_mean": left_effect.mean(dim=(2, 3, 4)),
        "right_singleton_signed_mean": right_effect.mean(dim=(1, 3, 4)),
        "context_count": torch.tensor(quartets.shape[2], dtype=torch.int64),
    }


def fit_origin_scale(basis: Any, target: Any) -> float:
    """Least-squares scalar for ``target ~= alpha * basis``, without intercept."""
    x = _finite_double(basis, label="basis").reshape(-1)
    y = _finite_double(target, label="target").reshape(-1)
    if x.shape != y.shape:
        raise ValueError("basis and target must have the same number of entries")
    denominator = float(torch.dot(x, x))
    if denominator == 0.0:
        return 0.0
    return float(torch.dot(x, y)) / denominator


def row_column_outer(energy: Any) -> torch.Tensor:
    """Maximum-entropy-style separable map preserving row/column energy totals."""
    values = _finite_double(energy, label="energy")
    if tuple(values.shape) != (HEAD_COUNT, HEAD_COUNT):
        raise ValueError("energy must have shape (12, 12)")
    if bool((values < 0.0).any()):
        raise ValueError("energy must be nonnegative")
    total = values.sum()
    if float(total) == 0.0:
        return torch.zeros_like(values)
    return torch.outer(values.sum(dim=1), values.sum(dim=0)) / total


def rank1_svd_map(energy: Any) -> torch.Tensor:
    """Return the nonnegative orientation of the leading rank-one SVD term."""
    values = _finite_double(energy, label="energy")
    if tuple(values.shape) != (HEAD_COUNT, HEAD_COUNT):
        raise ValueError("energy must have shape (12, 12)")
    if bool((values < 0.0).any()):
        raise ValueError("energy must be nonnegative")
    if float(values.square().sum()) == 0.0:
        return torch.zeros_like(values)
    left, singular_values, right_h = torch.linalg.svd(values, full_matrices=False)
    # A leading singular pair for a nonnegative matrix admits a nonnegative
    # orientation.  abs() also removes arbitrary sign choices across platforms.
    return singular_values[0] * torch.outer(left[:, 0].abs(), right_h[0].abs())


def rank1_signed_svd_map(signed_map: Any) -> torch.Tensor:
    """Return the leading rank-one SVD term without discarding its sign pattern."""
    values = _finite_double(signed_map, label="signed_map")
    if tuple(values.shape) != (HEAD_COUNT, HEAD_COUNT):
        raise ValueError("signed_map must have shape (12, 12)")
    if float(values.square().sum()) == 0.0:
        return torch.zeros_like(values)
    left, singular_values, right_h = torch.linalg.svd(values, full_matrices=False)
    return singular_values[0] * torch.outer(left[:, 0], right_h[0])


def _average_tied_ranks(values: torch.Tensor) -> torch.Tensor:
    flat = values.reshape(-1)
    order = sorted(range(flat.numel()), key=lambda index: (float(flat[index]), index))
    ranks = torch.empty_like(flat)
    start = 0
    while start < len(order):
        end = start + 1
        value = float(flat[order[start]])
        while end < len(order) and float(flat[order[end]]) == value:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def _pearson(first: torch.Tensor, second: torch.Tensor) -> float | None:
    x = first.reshape(-1) - first.mean()
    y = second.reshape(-1) - second.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) == 0.0:
        return 1.0 if torch.equal(first, second) else None
    value = float(torch.dot(x, y) / denominator)
    # Suppress harmless floating-point excursions such as 1.0000000000000002.
    value = max(-1.0, min(1.0, value))
    if abs(abs(value) - 1.0) <= 4.0 * torch.finfo(torch.float64).eps:
        return math.copysign(1.0, value)
    return value


def map_metrics(prediction: Any, target: Any) -> dict[str, float | None]:
    """No-calibration map agreement: R2, Pearson, Spearman, MSE, and cosine."""
    predicted = _finite_double(prediction, label="prediction")
    observed = _finite_double(target, label="target")
    if predicted.shape != observed.shape:
        raise ValueError("prediction and target must have identical shapes")
    residual_sum = float((observed - predicted).square().sum())
    total_sum = float((observed - observed.mean()).square().sum())
    r2 = None if total_sum == 0.0 else 1.0 - residual_sum / total_sum
    norm_product = torch.linalg.vector_norm(predicted) * torch.linalg.vector_norm(
        observed
    )
    cosine = (
        None
        if float(norm_product) == 0.0
        else float(
            torch.dot(predicted.reshape(-1), observed.reshape(-1)) / norm_product
        )
    )
    return {
        "r2": r2,
        "pearson": _pearson(predicted, observed),
        "spearman": _pearson(
            _average_tied_ranks(predicted), _average_tied_ranks(observed)
        ),
        "mse": residual_sum / observed.numel(),
        "cosine": cosine,
    }


def _ranked_indices(score: torch.Tensor, *, absolute: bool) -> list[int]:
    flat = score.reshape(-1)
    return sorted(
        range(flat.numel()),
        key=lambda index: (
            -float(flat[index].abs() if absolute else flat[index]),
            index,
        ),
    )


def support_metrics(
    discovery_score: Any,
    confirmation_energy: Any,
    *,
    k: int,
    absolute_score: bool = False,
) -> dict[str, Any]:
    """Evaluate a discovery-frozen top-k support on confirmation energy."""
    score = _finite_double(discovery_score, label="discovery_score")
    energy = _finite_double(confirmation_energy, label="confirmation_energy")
    if score.shape != energy.shape or score.ndim != 2:
        raise ValueError("score and confirmation_energy must be same-shaped matrices")
    if bool((energy < 0.0).any()):
        raise ValueError("confirmation_energy must be nonnegative")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= energy.numel():
        raise ValueError("k must be an integer in the map's support range")

    selected = _ranked_indices(score, absolute=absolute_score)[:k]
    oracle = _ranked_indices(energy, absolute=False)[:k]
    flat_energy = energy.reshape(-1)
    selected_energy = float(flat_energy[selected].sum())
    oracle_energy = float(flat_energy[oracle].sum())
    total_energy = float(flat_energy.sum())
    overlap = len(set(selected).intersection(oracle))
    return {
        "k": k,
        "selected_indices": selected,
        "oracle_indices": oracle,
        "support_overlap_count": overlap,
        "support_overlap_fraction": overlap / k,
        "selected_confirmation_energy": selected_energy,
        "confirmation_total_energy": total_energy,
        "confirmation_oracle_energy": oracle_energy,
        "confirmation_energy_coverage": (
            0.0 if total_energy == 0.0 else selected_energy / total_energy
        ),
        "fraction_of_confirmation_oracle_coverage": (
            0.0 if oracle_energy == 0.0 else selected_energy / oracle_energy
        ),
    }


def _predictors(split_maps: Mapping[str, torch.Tensor], parent: int) -> dict[str, Any]:
    target = split_maps["interaction_energy"][parent]
    singleton_basis = torch.outer(
        split_maps["left_singleton_energy"][parent],
        split_maps["right_singleton_energy"][parent],
    )
    alpha = fit_origin_scale(singleton_basis, target)
    return {
        "singleton_effect_outer": {
            "map": alpha * singleton_basis,
            "alpha": alpha,
            "endpoint_only": True,
            "support_ranking_requires_full_pair_map": False,
            "reported_alpha_uses_pair_map": True,
            "construction": (
                "outer product of mean-squared singleton effects; one no-intercept "
                "least-squares scalar fitted within this split"
            ),
        },
        "row_column_outer": {
            "map": row_column_outer(target),
            "endpoint_only": False,
            "support_ranking_requires_full_pair_map": True,
            "reported_alpha_uses_pair_map": False,
            "construction": "outer product of row/column energy totals divided by total",
        },
        "rank1_svd": {
            "map": rank1_svd_map(target),
            "endpoint_only": False,
            "support_ranking_requires_full_pair_map": True,
            "reported_alpha_uses_pair_map": False,
            "construction": "leading rank-one SVD term of the energy map",
        },
        "raw_pair_map": {
            "map": target.clone(),
            "endpoint_only": False,
            "support_ranking_requires_full_pair_map": True,
            "reported_alpha_uses_pair_map": False,
            "construction": "unmodified discovery head-pair energy map",
        },
    }


def _validated_top_ks(top_ks: Sequence[int]) -> tuple[int, ...]:
    if isinstance(top_ks, (str, bytes)):
        raise TypeError("top_ks must be a sequence")
    values = tuple(top_ks)
    if (
        not values
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= HEAD_COUNT * HEAD_COUNT
            for value in values
        )
        or tuple(sorted(set(values))) != values
    ):
        raise ValueError("top_ks must be unique ascending valid support sizes")
    return values


def _residual_record(
    discovery_residual: torch.Tensor,
    confirmation_residual: torch.Tensor,
    discovery_raw: torch.Tensor,
    confirmation_raw: torch.Tensor,
    top_ks: tuple[int, ...],
) -> dict[str, Any]:
    discovery_fraction = float(discovery_residual.square().sum()) / max(
        float(discovery_raw.square().sum()), torch.finfo(torch.float64).tiny
    )
    confirmation_fraction = float(confirmation_residual.square().sum()) / max(
        float(confirmation_raw.square().sum()), torch.finfo(torch.float64).tiny
    )
    return {
        "discovery_residual_squared_norm_fraction": discovery_fraction,
        "confirmation_residual_squared_norm_fraction": confirmation_fraction,
        "discovery_to_confirmation": map_metrics(
            discovery_residual, confirmation_residual
        ),
        "supports": {
            str(k): support_metrics(
                discovery_residual,
                confirmation_residual.square(),
                k=k,
                absolute_score=True,
            )
            for k in top_ks
        },
    }


def analyze_node_residual_transfer(
    discovery_labels: Mapping[str, Any],
    confirmation_labels: Mapping[str, Any],
    *,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
) -> dict[str, Any]:
    """Compare discovery-frozen node/separable maps on confirmation.

    Predictor parameters and supports are frozen from discovery.  For the
    explicitly descriptive residual analysis only, each split is residualized
    against its own node model before discovery residual supports are evaluated
    on confirmation residual energy.  This prevents a change in overall node
    scale from masquerading as edge-specific transfer, but it is not a held-out
    fitted predictor and is labelled accordingly in the returned artifact.
    """
    ks = _validated_top_ks(top_ks)
    discovery = compute_split_maps(_fine_quartets(discovery_labels))
    confirmation = compute_split_maps(_fine_quartets(confirmation_labels))

    parents: dict[str, Any] = {}
    pooled_predictions: dict[str, list[torch.Tensor]] = {}
    pooled_targets: list[torch.Tensor] = []
    for parent_index, parent_id in enumerate(PARENT_IDS):
        d_target = discovery["interaction_energy"][parent_index]
        c_target = confirmation["interaction_energy"][parent_index]
        d_predictors = _predictors(discovery, parent_index)
        c_descriptive_predictors = _predictors(confirmation, parent_index)
        pooled_targets.append(c_target)

        predictor_records = {}
        for name, predictor in d_predictors.items():
            frozen_map = predictor["map"]
            pooled_predictions.setdefault(name, []).append(frozen_map)
            record = {
                "construction": predictor["construction"],
                "endpoint_only": predictor["endpoint_only"],
                "support_ranking_requires_full_pair_map": predictor[
                    "support_ranking_requires_full_pair_map"
                ],
                "reported_alpha_uses_pair_map": predictor[
                    "reported_alpha_uses_pair_map"
                ],
                "discovery_fit": map_metrics(frozen_map, d_target),
                "confirmation": map_metrics(frozen_map, c_target),
                "supports": {
                    str(k): support_metrics(frozen_map, c_target, k=k) for k in ks
                },
            }
            if "alpha" in predictor:
                record["discovery_alpha"] = predictor["alpha"]
                record["confirmation_alpha_descriptive_only"] = (
                    c_descriptive_predictors[name]["alpha"]
                )
            predictor_records[name] = record

        residual_records = {}
        for name in ("singleton_effect_outer", "row_column_outer", "rank1_svd"):
            residual_records[name] = _residual_record(
                d_target - d_predictors[name]["map"],
                c_target - c_descriptive_predictors[name]["map"],
                d_target,
                c_target,
                ks,
            )

        d_signed = discovery["signed_mean"][parent_index]
        c_signed = confirmation["signed_mean"][parent_index]
        d_signed_basis = torch.outer(
            discovery["left_singleton_signed_mean"][parent_index],
            discovery["right_singleton_signed_mean"][parent_index],
        )
        c_signed_basis = torch.outer(
            confirmation["left_singleton_signed_mean"][parent_index],
            confirmation["right_singleton_signed_mean"][parent_index],
        )
        d_signed_alpha = fit_origin_scale(d_signed_basis, d_signed)
        c_signed_alpha = fit_origin_scale(c_signed_basis, c_signed)
        d_signed_prediction = d_signed_alpha * d_signed_basis
        c_signed_prediction = c_signed_alpha * c_signed_basis
        d_signed_rank1 = rank1_signed_svd_map(d_signed)
        c_signed_rank1 = rank1_signed_svd_map(c_signed)
        signed_diagnostic = {
            "endpoint_singleton_outer": {
                "construction": (
                    "outer product of signed mean singleton effects with one "
                    "no-intercept scalar fitted on discovery"
                ),
                "endpoint_only": True,
                "support_ranking_requires_full_pair_map": False,
                "reported_alpha_uses_pair_map": True,
                "discovery_alpha": d_signed_alpha,
                "confirmation_alpha_descriptive_only": c_signed_alpha,
                "discovery_fit": map_metrics(d_signed_prediction, d_signed),
                "confirmation": map_metrics(d_signed_prediction, c_signed),
                "supports": {
                    str(k): support_metrics(d_signed_prediction.square(), c_target, k=k)
                    for k in ks
                },
            },
            "raw_pair_signed_map": {
                "construction": "unmodified discovery signed-mean pair map",
                "endpoint_only": False,
                "support_ranking_requires_full_pair_map": True,
                "reported_alpha_uses_pair_map": False,
                "confirmation": map_metrics(d_signed, c_signed),
                "supports": {
                    str(k): support_metrics(d_signed.square(), c_target, k=k)
                    for k in ks
                },
            },
            "rank1_svd_signed_map": {
                "construction": "leading rank-one SVD term of discovery signed map",
                "endpoint_only": False,
                "support_ranking_requires_full_pair_map": True,
                "reported_alpha_uses_pair_map": False,
                "discovery_fit": map_metrics(d_signed_rank1, d_signed),
                "confirmation": map_metrics(d_signed_rank1, c_signed),
                "supports": {
                    str(k): support_metrics(d_signed_rank1.square(), c_target, k=k)
                    for k in ks
                },
            },
            "residual_after_endpoint_outer": _residual_record(
                d_signed - d_signed_prediction,
                c_signed - c_signed_prediction,
                d_signed,
                c_signed,
                ks,
            ),
            "residual_after_split_local_rank1_svd": _residual_record(
                d_signed - d_signed_rank1,
                c_signed - c_signed_rank1,
                d_signed,
                c_signed,
                ks,
            ),
        }
        parents[parent_id] = {
            "discovery_total_energy": float(d_target.sum()),
            "confirmation_total_energy": float(c_target.sum()),
            "predictors": predictor_records,
            "split_local_residual_transfer_descriptive_only": residual_records,
            "signed_map_diagnostic": signed_diagnostic,
        }

    pooled_target = torch.stack(pooled_targets)
    pooled = {}
    for name, values in pooled_predictions.items():
        prediction = torch.stack(values)
        pooled[name] = {
            "confirmation": map_metrics(prediction, pooled_target),
            "supports_global": {
                str(len(PARENT_IDS) * k): support_metrics(
                    prediction.reshape(len(PARENT_IDS) * HEAD_COUNT, HEAD_COUNT),
                    pooled_target.reshape(len(PARENT_IDS) * HEAD_COUNT, HEAD_COUNT),
                    k=len(PARENT_IDS) * k,
                )
                for k in ks
                if len(PARENT_IDS) * k <= pooled_target.numel()
            },
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "posthoc_development_diagnostic_only",
        "claim_boundaries": dict(CLAIM_BOUNDARIES),
        "parent_ids": list(PARENT_IDS),
        "top_ks_per_parent": list(ks),
        "discovery_context_count": int(discovery["context_count"]),
        "confirmation_context_count": int(confirmation["context_count"]),
        "definitions": {
            "interaction": "R11-R10-R01+R00",
            "map_target": "mean interaction squared over contexts and four covers",
            "predictor_freeze": (
                "all confirmation predictor maps and ranked supports use discovery "
                "values with no confirmation calibration"
            ),
            "residual_analysis": (
                "descriptive split-local nuisance removal; discovery absolute-residual "
                "support is evaluated on squared confirmation residual"
            ),
            "query_efficiency": (
                "singleton outer support rankings need only endpoint effects; their "
                "reported map R2 scale uses one alpha fitted on the full discovery map. "
                "Row/column, SVD, and raw support rankings require the full pair atlas"
            ),
        },
        "parents": parents,
        "pooled_predictors": pooled,
    }
