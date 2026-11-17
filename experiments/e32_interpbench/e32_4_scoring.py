# ruff: noqa: EXE002
"""Pure scoring for prospective E32.4 simultaneous internal edge sets."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from experiments.e32_interpbench.e32_2_scoring import (
    bidirectional_fine_replacement_scores,
)
from experiments.e32_interpbench.e32_3_scoring import (
    exact_fine_interaction_mass,
    set_faithfulness_metrics,
)
from experiments.e32_interpbench.e32_4_protocol import (
    ABSOLUTE_PRIMARY_GATES,
    BRIDGE_GATES,
    CASE_GEOMETRY,
    CASE_ORDER,
    DETERMINISTIC_METHOD_ORDER,
    END_TO_END_ALIGNMENT_GATES,
    GROUND_TRUTH_FINE_LABEL_COUNTS,
    GROUND_TRUTH_OFF_CORE_SCORE_FLOOR,
    PRIMARY_METHOD,
    RANDOM_CONTROL_95TH_INDEX,
    RANDOM_CONTROL_REPLICATES,
    ROBUSTNESS_GATES,
    SELECTOR_GATES,
    descriptive_support_sizes,
    deterministic_random_supports,
    primary_support_size,
)
from experiments.e32_interpbench.methods import masked_rank_one_endpoint_factor
from experiments.e32_interpbench.scoring import binary_ranking_metrics


def _finite_double(value: Any, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must be finite")
    return tensor


def _validated_ids(value: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    identifiers = tuple(value)
    if (
        not identifiers
        or len(identifiers) != len(set(identifiers))
        or any(
            not isinstance(identifier, str) or not identifier
            for identifier in identifiers
        )
    ):
        raise ValueError(f"{label} must be unique nonempty strings")
    return identifiers


def _validated_whole_edges(
    component_ids: Sequence[str], whole_edges: Sequence[tuple[str, str]]
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    components = _validated_ids(component_ids, label="component_ids")
    edges = tuple(tuple(edge) for edge in whole_edges)
    if (
        not edges
        or len(edges) != len(set(edges))
        or any(
            len(edge) != 2
            or edge[0] not in components
            or edge[1] not in components
            or edge[0] == edge[1]
            for edge in edges
        )
    ):
        raise ValueError("whole_edges must be unique directed component pairs")
    return components, edges


def equal_split_whole_scores(
    *, whole_scores: Any, fine_to_whole_indices: Sequence[int]
) -> torch.Tensor:
    """Split each whole-edge value equally among its registered fine children."""
    values = _finite_double(whole_scores, label="whole_scores")
    mapping = tuple(fine_to_whole_indices)
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("whole_scores must be a nonempty vector")
    if bool((values < 0.0).any()):
        raise ValueError("whole_scores must be nonnegative")
    if (
        not mapping
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < values.numel()
            for index in mapping
        )
        or set(mapping) != set(range(values.numel()))
    ):
        raise ValueError("fine_to_whole_indices must cover every whole edge")
    indices = torch.tensor(mapping, dtype=torch.long)
    child_counts = torch.bincount(indices, minlength=values.numel()).to(torch.float64)
    return values[indices] / child_counts[indices]


def build_selector_score_vectors(
    *,
    component_ids: Sequence[str],
    whole_edges: Sequence[tuple[str, str]],
    fine_to_whole_indices: Sequence[int],
    induced_fine_indices: Sequence[int],
    clean_output: Any,
    corrupt_output: Any,
    corrupt_into_clean_outputs: Any,
    clean_into_corrupt_outputs: Any,
    eap_fine_scores: Any,
    eap_ig5_fine_scores: Any,
    raw_whole_mass: Any,
) -> dict[str, torch.Tensor]:
    """Build the six frozen D selectors, retaining separate Q/K/V children."""
    components, edges = _validated_whole_edges(component_ids, whole_edges)
    mapping = tuple(fine_to_whole_indices)
    # This validates complete fine-to-whole coverage before any fitting.
    raw_split = equal_split_whole_scores(
        whole_scores=raw_whole_mass,
        fine_to_whole_indices=mapping,
    )
    if raw_split.numel() != len(mapping) or max(mapping) >= len(edges):
        raise ValueError("fine-to-whole mapping must align with whole_edges")
    induced = tuple(induced_fine_indices)
    if (
        not induced
        or tuple(sorted(induced)) != induced
        or len(set(induced)) != len(induced)
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(mapping)
            for index in induced
        )
    ):
        raise ValueError("induced_fine_indices must be unique and strictly increasing")

    exact = exact_fine_interaction_mass(
        clean_output=clean_output,
        corrupt_output=corrupt_output,
        corrupt_into_clean_outputs=corrupt_into_clean_outputs,
        clean_into_corrupt_outputs=clean_into_corrupt_outputs,
    )
    directional = bidirectional_fine_replacement_scores(
        clean_output=clean_output,
        corrupt_output=corrupt_output,
        corrupt_into_clean_outputs=corrupt_into_clean_outputs,
        clean_into_corrupt_outputs=clean_into_corrupt_outputs,
    ).fine_scores
    if exact.numel() != len(induced) or directional.numel() != len(induced):
        raise ValueError("exact replacement outputs must align with induced fine edges")

    eap = _finite_double(eap_fine_scores, label="eap_fine_scores")
    eap_ig5 = _finite_double(eap_ig5_fine_scores, label="eap_ig5_fine_scores")
    if eap.ndim != 1 or eap.numel() != len(mapping) or eap_ig5.shape != eap.shape:
        raise ValueError("EAP and EAP-IG5 must align with the full fine universe")

    raw = _finite_double(raw_whole_mass, label="raw_whole_mass")
    if raw.ndim != 1 or raw.numel() != len(edges) or bool((raw < 0.0).any()):
        raise ValueError(
            "raw_whole_mass must align with whole_edges and be nonnegative"
        )
    rank_one = masked_rank_one_endpoint_factor(
        node_universe=components,
        edge_universe=edges,
        edge_scores=raw,
    )
    rank_one_split = equal_split_whole_scores(
        whole_scores=rank_one.edge_reconstruction,
        fine_to_whole_indices=mapping,
    )
    index = torch.tensor(induced, dtype=torch.long)
    return {
        PRIMARY_METHOD: exact,
        "bidirectional_directional_projection": directional,
        "eap_absolute": eap.abs()[index],
        "eap_ig5_absolute": eap_ig5.abs()[index],
        "raw_whole_mass_equal_fine_split": raw_split[index],
        "masked_rank_one_equal_fine_split": rank_one_split[index],
    }


def deterministic_top_support(
    *, fine_edge_ids: Sequence[str], scores: Any, support_size: int
) -> tuple[str, ...]:
    """Select descending scores with frozen-index ties; emit canonical order."""
    fine = _validated_ids(fine_edge_ids, label="fine_edge_ids")
    values = _finite_double(scores, label="scores")
    if (
        values.ndim != 1
        or values.numel() != len(fine)
        or isinstance(support_size, bool)
        or not isinstance(support_size, int)
        or not 1 <= support_size <= len(fine)
    ):
        raise ValueError("scores and support_size must align with fine_edge_ids")
    ranked = sorted(range(len(fine)), key=lambda index: (-float(values[index]), index))
    chosen = frozenset(ranked[:support_size])
    return tuple(edge for index, edge in enumerate(fine) if index in chosen)


def _unit_threshold(value: float, *, label: str) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return threshold


def faithfulness_gate_checks(
    metrics: Mapping[str, Any],
    *,
    minimum_projection: float,
    maximum_residual_error: float,
    minimum_loss_recovery: float,
) -> dict[str, bool]:
    """Apply symmetric necessity/sufficiency thresholds without clipping."""
    min_projection = _unit_threshold(minimum_projection, label="minimum_projection")
    max_residual = _unit_threshold(
        maximum_residual_error, label="maximum_residual_error"
    )
    min_loss = _unit_threshold(minimum_loss_recovery, label="minimum_loss_recovery")

    def finite_compare(name: str, predicate: Any) -> bool:
        try:
            number = float(metrics[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"faithfulness metrics missing numeric {name}") from error
        return math.isfinite(number) and bool(predicate(number))

    checks = {
        "endpoint_informative": metrics.get("endpoint_informative") is True,
        "necessity_projection_pass": finite_compare(
            "necessity_projection", lambda value: value >= min_projection
        ),
        "sufficiency_projection_pass": finite_compare(
            "sufficiency_projection", lambda value: value >= min_projection
        ),
        "necessity_residual_error_pass": finite_compare(
            "necessity_residual_error", lambda value: value <= max_residual
        ),
        "sufficiency_residual_error_pass": finite_compare(
            "sufficiency_residual_error", lambda value: value <= max_residual
        ),
        "necessity_loss_recovery_pass": finite_compare(
            "necessity_loss_recovery", lambda value: value >= min_loss
        ),
        "sufficiency_loss_recovery_pass": finite_compare(
            "sufficiency_loss_recovery", lambda value: value >= min_loss
        ),
    }
    checks["pass"] = all(checks.values())
    return checks


def joint_faithfulness_score(metrics: Mapping[str, Any]) -> float:
    """Conservative unclipped minimum over all six faithfulness coordinates."""
    if metrics.get("endpoint_informative") is not True:
        return float("nan")
    names = (
        "necessity_projection",
        "sufficiency_projection",
        "necessity_residual_error",
        "sufficiency_residual_error",
        "necessity_loss_recovery",
        "sufficiency_loss_recovery",
    )
    try:
        values = [float(metrics[name]) for name in names]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("faithfulness metrics are incomplete") from error
    if not all(math.isfinite(value) for value in values):
        return float("nan")
    return min(values[0], values[1], 1.0 - values[2], 1.0 - values[3], *values[4:])


def _target_indices(
    value: Any, *, shape: tuple[int, int], classes: int
) -> torch.Tensor:
    targets = torch.as_tensor(value, device="cpu")
    if (
        targets.shape != shape
        or targets.dtype == torch.bool
        or targets.is_floating_point()
    ):
        raise ValueError("clean_target_indices must be integer [example, position]")
    targets = targets.to(torch.long)
    if bool((targets < 0).any()) or bool((targets >= classes).any()):
        raise ValueError("clean target index lies outside the class axis")
    return targets


def endpoint_alignment_metrics(
    *,
    clean_output: Any,
    internal_anchor_output: Any,
    actual_corrupt_output: Any,
    clean_target_indices: Any,
) -> dict[str, Any]:
    """Describe how well an internal anchor reaches the actual corrupt endpoint."""
    clean = _finite_double(clean_output, label="clean_output")
    anchor = _finite_double(internal_anchor_output, label="internal_anchor_output")
    corrupt = _finite_double(actual_corrupt_output, label="actual_corrupt_output")
    if (
        clean.ndim != 3
        or clean.shape[-1] < 2
        or anchor.shape != clean.shape
        or corrupt.shape != clean.shape
    ):
        raise ValueError("endpoint outputs must share [example, position, class]")
    targets = _target_indices(
        clean_target_indices,
        shape=tuple(clean.shape[:2]),
        classes=clean.shape[-1],
    )
    gap = clean - corrupt
    displacement = clean - anchor
    axes = (1, 2)
    gap_energy = gap.square().sum(dim=axes)
    projection_numerator = (displacement * gap).sum(dim=axes)
    residual_energy = (anchor - corrupt).square().sum(dim=axes)

    def per_example_loss(logits: torch.Tensor) -> torch.Tensor:
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)
        return token_loss.mean(dim=1)

    clean_loss = per_example_loss(clean)
    anchor_loss = per_example_loss(anchor)
    corrupt_loss = per_example_loss(corrupt)
    total_gap = float(gap_energy.sum())
    loss_gap = float((corrupt_loss - clean_loss).sum())
    informative = total_gap > 0.0 and loss_gap > 0.0
    projection = (
        float(projection_numerator.sum()) / total_gap
        if total_gap > 0.0
        else float("nan")
    )
    residual = (
        float(residual_energy.sum()) / total_gap if total_gap > 0.0 else float("nan")
    )
    loss_capture = (
        float((anchor_loss - clean_loss).sum()) / loss_gap
        if loss_gap > 0.0
        else float("nan")
    )
    alignment_pass = (
        informative
        and projection >= END_TO_END_ALIGNMENT_GATES["minimum_projection_capture"]
        and residual <= END_TO_END_ALIGNMENT_GATES["maximum_residual_error"]
        and loss_capture >= END_TO_END_ALIGNMENT_GATES["minimum_loss_capture"]
    )
    return {
        "endpoint_informative": informative,
        "projection_capture": projection,
        "residual_error": residual,
        "loss_capture": loss_capture,
        "alignment_pass": alignment_pass,
        "used_for_boundary_conditioned_primary_claim": False,
        "normalization": "clean_vs_actual_corrupt_input_endpoint",
        "per_example": {
            "vector_gap_energy": gap_energy,
            "projection_numerator": projection_numerator,
            "residual_energy": residual_energy,
            "clean_target_loss": clean_loss,
            "anchor_target_loss": anchor_loss,
            "actual_corrupt_target_loss": corrupt_loss,
        },
    }


def _selector_panel(
    value: Mapping[str, Any], *, fine_count: int
) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or tuple(value) != DETERMINISTIC_METHOD_ORDER:
        raise ValueError(
            "discovery_selector_scores must exactly follow "
            f"{DETERMINISTIC_METHOD_ORDER!r}"
        )
    panel: dict[str, torch.Tensor] = {}
    for method in DETERMINISTIC_METHOD_ORDER:
        scores = _finite_double(value[method], label=f"{method} scores")
        if scores.ndim != 1 or scores.numel() != fine_count:
            raise ValueError(f"{method} scores must align with the induced fine core")
        panel[method] = scores
    return panel


def _set_record(
    *,
    record: Mapping[str, Any],
    expected_support: tuple[str, ...],
    clean_output: Any,
    all_internal_output: Any,
    clean_target_indices: Any,
    minimum_projection: float,
    maximum_residual_error: float,
    minimum_loss_recovery: float,
    label: str,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"{label} must be a mapping")
    try:
        observed_support = tuple(record["selected_fine_edge_ids"])
        selected_output = record["selected_output"]
        complement_output = record["complement_output"]
    except KeyError as error:
        raise ValueError(f"{label} is missing a registered field") from error
    if observed_support != expected_support:
        raise ValueError(f"{label} support differs from the frozen D support")
    faithfulness = set_faithfulness_metrics(
        clean_output=clean_output,
        all_corrupt_into_clean_output=all_internal_output,
        selected_corrupt_into_clean_output=selected_output,
        complement_corrupt_into_clean_output=complement_output,
        clean_target_indices=clean_target_indices,
    )
    return {
        "selected_fine_edge_ids": expected_support,
        "faithfulness": faithfulness,
        "joint_faithfulness": joint_faithfulness_score(faithfulness),
        "gate_checks": faithfulness_gate_checks(
            faithfulness,
            minimum_projection=minimum_projection,
            maximum_residual_error=maximum_residual_error,
            minimum_loss_recovery=minimum_loss_recovery,
        ),
    }


def _direct_set_metrics(
    *,
    clean_output: Any,
    all_internal_output: Any,
    selected_output: Any,
    complement_output: Any,
    clean_target_indices: Any,
    minimum_projection: float,
    maximum_residual_error: float,
    minimum_loss_recovery: float,
) -> dict[str, Any]:
    faithfulness = set_faithfulness_metrics(
        clean_output=clean_output,
        all_corrupt_into_clean_output=all_internal_output,
        selected_corrupt_into_clean_output=selected_output,
        complement_corrupt_into_clean_output=complement_output,
        clean_target_indices=clean_target_indices,
    )
    return {
        "faithfulness": faithfulness,
        "joint_faithfulness": joint_faithfulness_score(faithfulness),
        "gate_checks": faithfulness_gate_checks(
            faithfulness,
            minimum_projection=minimum_projection,
            maximum_residual_error=maximum_residual_error,
            minimum_loss_recovery=minimum_loss_recovery,
        ),
    }


def evaluate_case(
    *,
    case_id: int,
    induced_fine_edge_ids: Sequence[str],
    discovery_selector_scores: Mapping[str, Any],
    clean_output: Any,
    actual_corrupt_output: Any,
    clean_target_indices: Any,
    resample_all_internal_output: Any,
    resample_induced_core_output: Any,
    resample_outside_core_output: Any,
    resample_deterministic_outputs: Mapping[str, Mapping[str, Any]],
    resample_exact_mass_curve_outputs: Mapping[str, Mapping[str, Any]],
    resample_random_outputs: Sequence[Mapping[str, Any]],
    resample_within_core_primary_complement_output: Any,
    mean_all_internal_output: Any,
    mean_induced_core_output: Any,
    mean_outside_core_output: Any,
    mean_primary_output: Any,
    mean_primary_complement_output: Any,
) -> dict[str, Any]:
    """Score one frozen C case without adapting supports or thresholds."""
    if case_id not in CASE_ORDER:
        raise ValueError(f"case_id must be one of {CASE_ORDER!r}")
    fine = _validated_ids(induced_fine_edge_ids, label="induced_fine_edge_ids")
    if len(fine) != CASE_GEOMETRY[case_id]["induced_fine_edges"]:
        raise ValueError("induced fine-edge count differs from frozen core geometry")
    selectors = _selector_panel(discovery_selector_scores, fine_count=len(fine))
    support_size = primary_support_size(case_id)
    supports = {
        method: deterministic_top_support(
            fine_edge_ids=fine,
            scores=selectors[method],
            support_size=support_size,
        )
        for method in DETERMINISTIC_METHOD_ORDER
    }
    if (
        not isinstance(resample_deterministic_outputs, Mapping)
        or tuple(resample_deterministic_outputs) != DETERMINISTIC_METHOD_ORDER
    ):
        raise ValueError(
            "resample deterministic outputs must follow frozen method order"
        )

    deterministic_records = {
        method: _set_record(
            record=resample_deterministic_outputs[method],
            expected_support=supports[method],
            clean_output=clean_output,
            all_internal_output=resample_all_internal_output,
            clean_target_indices=clean_target_indices,
            minimum_projection=ABSOLUTE_PRIMARY_GATES["minimum_necessity_projection"],
            maximum_residual_error=ABSOLUTE_PRIMARY_GATES[
                "maximum_necessity_residual_error"
            ],
            minimum_loss_recovery=ABSOLUTE_PRIMARY_GATES[
                "minimum_necessity_loss_recovery"
            ],
            label=f"resample {method}",
        )
        for method in DETERMINISTIC_METHOD_ORDER
    }

    curve_sizes = descriptive_support_sizes(case_id)
    if not isinstance(resample_exact_mass_curve_outputs, Mapping) or tuple(
        resample_exact_mass_curve_outputs
    ) != tuple(curve_sizes):
        raise ValueError("exact-mass curve outputs must contain k05 then k15")
    curve_records: dict[str, dict[str, Any]] = {}
    for curve_id, curve_size in curve_sizes.items():
        expected = deterministic_top_support(
            fine_edge_ids=fine,
            scores=selectors[PRIMARY_METHOD],
            support_size=curve_size,
        )
        curve_record = _set_record(
            record=resample_exact_mass_curve_outputs[curve_id],
            expected_support=expected,
            clean_output=clean_output,
            all_internal_output=resample_all_internal_output,
            clean_target_indices=clean_target_indices,
            minimum_projection=ABSOLUTE_PRIMARY_GATES["minimum_necessity_projection"],
            maximum_residual_error=ABSOLUTE_PRIMARY_GATES[
                "maximum_necessity_residual_error"
            ],
            minimum_loss_recovery=ABSOLUTE_PRIMARY_GATES[
                "minimum_necessity_loss_recovery"
            ],
            label=f"resample exact-mass {curve_id}",
        )
        curve_record["support_size"] = curve_size
        curve_record["used_for_primary_or_selector_gates"] = False
        curve_records[curve_id] = curve_record

    expected_random = deterministic_random_supports(
        case_id=case_id,
        fine_edge_ids=fine,
        support_size=support_size,
    )
    random_inputs = tuple(resample_random_outputs)
    if len(random_inputs) != RANDOM_CONTROL_REPLICATES:
        raise ValueError("resample random outputs must contain exactly 256 replicates")
    random_records: list[dict[str, Any]] = []
    for replicate, (record, expected) in enumerate(
        zip(random_inputs, expected_random, strict=True)
    ):
        if record.get("replicate") != replicate:
            raise ValueError("random replicate indices must be complete and ordered")
        scored = _set_record(
            record=record,
            expected_support=expected,
            clean_output=clean_output,
            all_internal_output=resample_all_internal_output,
            clean_target_indices=clean_target_indices,
            minimum_projection=ABSOLUTE_PRIMARY_GATES["minimum_necessity_projection"],
            maximum_residual_error=ABSOLUTE_PRIMARY_GATES[
                "maximum_necessity_residual_error"
            ],
            minimum_loss_recovery=ABSOLUTE_PRIMARY_GATES[
                "minimum_necessity_loss_recovery"
            ],
            label=f"resample random replicate {replicate}",
        )
        scored["replicate"] = replicate
        random_records.append(scored)

    bridge = _direct_set_metrics(
        clean_output=clean_output,
        all_internal_output=resample_all_internal_output,
        selected_output=resample_induced_core_output,
        complement_output=resample_outside_core_output,
        clean_target_indices=clean_target_indices,
        minimum_projection=BRIDGE_GATES["minimum_necessity_projection"],
        maximum_residual_error=BRIDGE_GATES["maximum_necessity_residual_error"],
        minimum_loss_recovery=BRIDGE_GATES["minimum_necessity_loss_recovery"],
    )
    primary = deterministic_records[PRIMARY_METHOD]
    within_core = _direct_set_metrics(
        clean_output=clean_output,
        all_internal_output=resample_induced_core_output,
        selected_output=resample_deterministic_outputs[PRIMARY_METHOD][
            "selected_output"
        ],
        complement_output=resample_within_core_primary_complement_output,
        clean_target_indices=clean_target_indices,
        minimum_projection=ABSOLUTE_PRIMARY_GATES["minimum_necessity_projection"],
        maximum_residual_error=ABSOLUTE_PRIMARY_GATES[
            "maximum_necessity_residual_error"
        ],
        minimum_loss_recovery=ABSOLUTE_PRIMARY_GATES["minimum_necessity_loss_recovery"],
    )
    within_core["used_for_primary_gates"] = False

    mean_bridge = _direct_set_metrics(
        clean_output=clean_output,
        all_internal_output=mean_all_internal_output,
        selected_output=mean_induced_core_output,
        complement_output=mean_outside_core_output,
        clean_target_indices=clean_target_indices,
        minimum_projection=ROBUSTNESS_GATES["minimum_necessity_projection"],
        maximum_residual_error=ROBUSTNESS_GATES["maximum_necessity_residual_error"],
        minimum_loss_recovery=ROBUSTNESS_GATES["minimum_necessity_loss_recovery"],
    )
    mean_primary = _direct_set_metrics(
        clean_output=clean_output,
        all_internal_output=mean_all_internal_output,
        selected_output=mean_primary_output,
        complement_output=mean_primary_complement_output,
        clean_target_indices=clean_target_indices,
        minimum_projection=ROBUSTNESS_GATES["minimum_necessity_projection"],
        maximum_residual_error=ROBUSTNESS_GATES["maximum_necessity_residual_error"],
        minimum_loss_recovery=ROBUSTNESS_GATES["minimum_necessity_loss_recovery"],
    )
    resample_alignment = endpoint_alignment_metrics(
        clean_output=clean_output,
        internal_anchor_output=resample_all_internal_output,
        actual_corrupt_output=actual_corrupt_output,
        clean_target_indices=clean_target_indices,
    )
    mean_alignment = endpoint_alignment_metrics(
        clean_output=clean_output,
        internal_anchor_output=mean_all_internal_output,
        actual_corrupt_output=actual_corrupt_output,
        clean_target_indices=clean_target_indices,
    )

    primary_joint = float(primary["joint_faithfulness"])
    comparator_joint = {
        method: float(deterministic_records[method]["joint_faithfulness"])
        for method in DETERMINISTIC_METHOD_ORDER
        if method != PRIMARY_METHOD
    }
    random_joint = tuple(
        float(record["joint_faithfulness"]) for record in random_records
    )
    selector_values_finite = math.isfinite(primary_joint) and all(
        math.isfinite(value) for value in (*comparator_joint.values(), *random_joint)
    )
    best_comparator = max(comparator_joint.values())
    random_95th = sorted(random_joint)[RANDOM_CONTROL_95TH_INDEX]
    deterministic_deficit = best_comparator - primary_joint
    random_margin = primary_joint - random_95th
    deterministic_noninferiority = (
        selector_values_finite
        and deterministic_deficit
        <= SELECTOR_GATES["maximum_deficit_to_best_deterministic_baseline"]
    )
    random_superiority = (
        selector_values_finite
        and random_margin
        >= SELECTOR_GATES["minimum_margin_over_random_95th_percentile"]
    )

    sparsity_fraction = support_size / CASE_GEOMETRY[case_id]["fine_edges"]
    structural_sparse = sparsity_fraction <= 0.10
    absolute_pass = bool(
        structural_sparse
        and bridge["gate_checks"]["pass"]
        and primary["gate_checks"]["pass"]
    )
    selector_pass = bool(
        absolute_pass and deterministic_noninferiority and random_superiority
    )
    robustness_pass = bool(
        mean_bridge["gate_checks"]["pass"] and mean_primary["gate_checks"]["pass"]
    )
    return {
        "schema_version": "e32_4_case_score_v1",
        "case_id": case_id,
        "supports": supports,
        "primary_support_size": support_size,
        "sparsity_fraction_of_all_internal_fine_edges": sparsity_fraction,
        "resample_bridge": bridge,
        "resample_deterministic": deterministic_records,
        "exact_mass_support_curve": curve_records,
        "resample_within_core_primary_fallback": within_core,
        "resample_random": tuple(random_records),
        "random_joint_faithfulness": random_joint,
        "mean_corrupt_bridge": mean_bridge,
        "mean_corrupt_primary": mean_primary,
        "input_endpoint_alignment": {
            "resample": resample_alignment,
            "batch_tokenwise_mean_corrupt": mean_alignment,
        },
        "selector_comparison": {
            "primary_joint_faithfulness": primary_joint,
            "deterministic_comparator_joint_faithfulness": comparator_joint,
            "best_deterministic_comparator": best_comparator,
            "deficit_to_best_deterministic": deterministic_deficit,
            "deterministic_noninferiority_pass": deterministic_noninferiority,
            "random_95th_sorted_index": RANDOM_CONTROL_95TH_INDEX,
            "random_95th_joint_faithfulness": random_95th,
            "margin_over_random_95th": random_margin,
            "random_superiority_pass": random_superiority,
        },
        "gates": {
            "absolute_sparse_set_pass": absolute_pass,
            "selector_pass": selector_pass,
            "mean_corrupt_robustness_pass": robustness_pass,
            "resample_end_to_end_pass": bool(
                absolute_pass and resample_alignment["alignment_pass"]
            ),
            "mean_corrupt_can_rescue_resample": False,
            "case_pass": absolute_pass,
        },
        "claim_boundary": (
            "boundary_conditioned_internal_inter_component_fine_edge_support"
        ),
    }


def _stable_unique_edges(
    value: Sequence[tuple[str, str]], *, label: str
) -> tuple[tuple[str, str], ...]:
    edges = tuple(tuple(edge) for edge in value)
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
        raise ValueError(f"{label} must be unique directed edges")
    return edges


def exact_fine_ground_truth_ids(
    *,
    case_id: int,
    full_fine_edge_ids: Sequence[str],
    induced_fine_edge_ids: Sequence[str],
    regular_eap_all_fine_records: Sequence[Mapping[str, Any]],
    official_raw_edges: Sequence[tuple[str, str]],
) -> tuple[str, ...]:
    """Label fine U by exact raw endpoints; never expand promoted whole truth."""
    if case_id not in CASE_ORDER:
        raise ValueError(f"case_id must be one of {CASE_ORDER!r}")
    fine = _validated_ids(full_fine_edge_ids, label="full_fine_edge_ids")
    induced = _validated_ids(induced_fine_edge_ids, label="induced_fine_edge_ids")
    induced_set = set(induced)
    if tuple(edge for edge in fine if edge in induced_set) != induced:
        raise ValueError("induced_fine_edge_ids must be a canonical subset of full U")

    coordinates_by_name: dict[str, tuple[str, str]] = {}
    names_by_coordinate: dict[tuple[str, str], str] = {}
    for record in regular_eap_all_fine_records:
        if not isinstance(record, Mapping):
            raise TypeError("regular-EAP fine records must be mappings")
        try:
            name = record["fine_edge_name"]
            coordinate = (record["source"], record["target"])
        except KeyError as error:
            raise ValueError("regular-EAP fine record is incomplete") from error
        if (
            not isinstance(name, str)
            or not name
            or name in coordinates_by_name
            or not all(isinstance(node, str) and node for node in coordinate)
        ):
            raise ValueError("regular-EAP fine records are malformed or duplicate")
        if coordinate in names_by_coordinate:
            raise ValueError("distinct regular-EAP fine names share one raw coordinate")
        coordinates_by_name[name] = coordinate
        names_by_coordinate[coordinate] = name
    if any(name not in coordinates_by_name for name in fine):
        raise ValueError("regular-EAP fine records do not cover full internal U")

    raw_truth = set(
        _stable_unique_edges(official_raw_edges, label="official_raw_edges")
    )
    truth = tuple(name for name in fine if coordinates_by_name[name] in raw_truth)
    expected = GROUND_TRUTH_FINE_LABEL_COUNTS[case_id]
    if (
        len(truth) != expected["full_internal_U"]
        or len(set(truth) & induced_set) != expected["induced_core_I"]
    ):
        raise ValueError("coordinate-exact fine ground-truth counts changed")
    return truth


def _support_counts(selected: set[Any], truth: set[Any]) -> dict[str, float | int]:
    true_positive = len(selected & truth)
    precision = true_positive / len(selected)
    recall = true_positive / len(truth)
    return {
        "selected_count": len(selected),
        "ground_truth_count": len(truth),
        "true_positives": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": (
            0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        ),
    }


def descriptive_ground_truth_metrics(
    *,
    full_fine_edge_ids: Sequence[str],
    full_fine_whole_edges: Sequence[tuple[str, str]],
    induced_fine_edge_ids: Sequence[str],
    selector_scores: Mapping[str, Any],
    selected_supports: Mapping[str, Sequence[str]],
    fine_ground_truth_ids: Sequence[str],
    whole_ground_truth_edges: Sequence[tuple[str, str]],
    transitive_non_edges: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Rank descriptively over U, assigning every unqueried U\\I edge -1."""
    fine = _validated_ids(full_fine_edge_ids, label="full_fine_edge_ids")
    induced = _validated_ids(induced_fine_edge_ids, label="induced_fine_edge_ids")
    promoted = tuple(tuple(edge) for edge in full_fine_whole_edges)
    if len(promoted) != len(fine):
        raise ValueError("full_fine_whole_edges must align with full_fine_edge_ids")
    induced_set = set(induced)
    if tuple(edge for edge in fine if edge in induced_set) != induced:
        raise ValueError("induced_fine_edge_ids must be a canonical subset of full U")
    whole = tuple(dict.fromkeys(promoted))
    _stable_unique_edges(whole, label="promoted whole-edge universe")
    fine_truth = set(
        _validated_ids(fine_ground_truth_ids, label="fine_ground_truth_ids")
    )
    whole_truth = set(
        _stable_unique_edges(whole_ground_truth_edges, label="whole_ground_truth_edges")
    )
    transitive = set(
        _stable_unique_edges(transitive_non_edges, label="transitive_non_edges")
    )
    if whole_truth & transitive:
        raise ValueError(
            "direct ground truth and transitive non-edges must be disjoint"
        )
    if not isinstance(selector_scores, Mapping) or tuple(selector_scores) != tuple(
        selected_supports
    ):
        raise ValueError(
            "selector scores and supports must have identical method order"
        )

    fine_index = {edge: index for index, edge in enumerate(fine)}
    induced_index = {edge: index for index, edge in enumerate(induced)}
    whole_index = {edge: index for index, edge in enumerate(whole)}
    fine_truth_in_universe = fine_truth & set(fine)
    whole_truth_in_universe = whole_truth & set(whole)
    transitive_in_universe = transitive & set(whole)
    if (
        not fine_truth_in_universe
        or len(fine_truth_in_universe) == len(fine)
        or not whole_truth_in_universe
        or len(whole_truth_in_universe) == len(whole)
        or not transitive_in_universe
    ):
        raise ValueError(
            "descriptive ranking needs in-universe positives and negatives"
        )

    methods: dict[str, Any] = {}
    for method, raw_scores in selector_scores.items():
        queried_scores = _finite_double(raw_scores, label=f"{method} selector scores")
        if (
            queried_scores.ndim != 1
            or queried_scores.numel() != len(induced)
            or bool((queried_scores < 0.0).any())
        ):
            raise ValueError(
                f"{method} selector scores must be nonnegative and align with I"
            )
        support = tuple(selected_supports[method])
        if (
            not support
            or len(support) != len(set(support))
            or not set(support).issubset(induced)
            or support
            != deterministic_top_support(
                fine_edge_ids=induced,
                scores=queried_scores,
                support_size=len(support),
            )
        ):
            raise ValueError(f"{method} support must be its exact deterministic top-k")

        scores = torch.full(
            (len(fine),),
            GROUND_TRUTH_OFF_CORE_SCORE_FLOOR,
            dtype=torch.float64,
        )
        for edge in induced:
            scores[fine_index[edge]] = queried_scores[induced_index[edge]]
        fine_ranking = binary_ranking_metrics(
            scores=scores.tolist(),
            labels=[edge in fine_truth_in_universe for edge in fine],
            top_k=len(support),
        )
        whole_scores = torch.full((len(whole),), -torch.inf, dtype=torch.float64)
        for fine_position, edge in enumerate(promoted):
            position = whole_index[edge]
            whole_scores[position] = torch.maximum(
                whole_scores[position], scores[fine_position]
            )
        if not bool(torch.isfinite(whole_scores).all()):
            raise RuntimeError("whole-edge max promotion left an empty group")
        selected_whole = tuple(
            dict.fromkeys(promoted[fine_index[edge]] for edge in support)
        )
        whole_ranking = binary_ranking_metrics(
            scores=whole_scores.tolist(),
            labels=[edge in whole_truth_in_universe for edge in whole],
            top_k=len(selected_whole),
        )
        direct_transitive_order = (
            *[edge for edge in whole if edge in whole_truth_in_universe],
            *[edge for edge in whole if edge in transitive_in_universe],
        )
        direct_transitive = binary_ranking_metrics(
            scores=[
                float(whole_scores[whole_index[edge]])
                for edge in direct_transitive_order
            ],
            labels=[
                edge in whole_truth_in_universe for edge in direct_transitive_order
            ],
            top_k=len(whole_truth_in_universe),
        )
        methods[method] = {
            "fine": fine_ranking,
            "whole": whole_ranking,
            "direct_vs_transitive": direct_transitive,
            "fine_selected_support": _support_counts(set(support), fine_truth),
            "whole_selected_support": _support_counts(set(selected_whole), whole_truth),
        }
    return {
        "ground_truth_used_for_selection_or_gates": False,
        "ranking_universe": "U_all_internal_fine_edges",
        "outside_induced_core_score": GROUND_TRUTH_OFF_CORE_SCORE_FLOOR,
        "fine_ground_truth_coverage_by_induced_core": len(
            fine_truth_in_universe & induced_set
        )
        / len(fine_truth),
        "whole_ground_truth_coverage_by_induced_core": len(
            whole_truth_in_universe & {promoted[fine_index[edge]] for edge in induced}
        )
        / len(whole_truth),
        "fine_ground_truth_coverage_by_full_universe": len(fine_truth_in_universe)
        / len(fine_truth),
        "whole_ground_truth_coverage_by_full_universe": len(whole_truth_in_universe)
        / len(whole_truth),
        "methods": methods,
    }


def evaluate_panel_verdict(
    case_scores: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require all three registered cases independently for every claim tier."""
    if not isinstance(case_scores, Mapping) or set(case_scores) != {
        str(case_id) for case_id in CASE_ORDER
    }:
        raise ValueError("case_scores must contain exactly the three string case IDs")
    ordered = []
    for case_id in CASE_ORDER:
        record = case_scores[str(case_id)]
        if record.get("case_id") != case_id or not isinstance(
            record.get("gates"), Mapping
        ):
            raise ValueError(
                "each case score must carry its matching case_id and gates"
            )
        ordered.append(record["gates"])

    def conjunction(name: str) -> bool:
        values = [gates.get(name) for gates in ordered]
        if any(type(value) is not bool for value in values):
            raise ValueError(f"every case must provide boolean {name}")
        return all(values)

    return {
        "absolute_sparse_set_generalization_pass": conjunction(
            "absolute_sparse_set_pass"
        ),
        "selector_generalization_pass": conjunction("selector_pass"),
        "mean_corrupt_robustness_generalization_pass": conjunction(
            "mean_corrupt_robustness_pass"
        ),
        "resample_end_to_end_generalization_pass": conjunction(
            "resample_end_to_end_pass"
        ),
        "all_cases_required": True,
        "majority_rule_allowed": False,
        "claim_boundary": (
            "boundary_conditioned_internal_inter_component_fine_edge_support;"
            "not_direct_wiring;not_minimal;not_sota;not_model_generalization"
        ),
    }


__all__ = [
    "build_selector_score_vectors",
    "descriptive_ground_truth_metrics",
    "deterministic_top_support",
    "endpoint_alignment_metrics",
    "equal_split_whole_scores",
    "evaluate_case",
    "evaluate_panel_verdict",
    "exact_fine_ground_truth_ids",
    "faithfulness_gate_checks",
    "joint_faithfulness_score",
]
