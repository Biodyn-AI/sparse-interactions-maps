# ruff: noqa: EXE002
"""Pure exact-set and prospective panel scoring for E32.5."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from experiments.e32_interpbench.e32_3_scoring import set_faithfulness_metrics
from experiments.e32_interpbench.e32_4_scoring import (
    endpoint_alignment_metrics,
    faithfulness_gate_checks,
    joint_faithfulness_score,
)
from experiments.e32_interpbench.e32_5_protocol import (
    CLAIM_BOUNDARY,
    CONTEXTUAL_SIGNED_STATIC_METHOD,
    CONTEXTUAL_UNSIGNED_MASS_METHOD,
    DETERMINISTIC_BASELINE_ORDER,
    END_TO_END_ALIGNMENT_GATES,
    GENERALIZATION_CASE_ORDER,
    GROUP_INTERACTION_ADVANTAGE_MARGIN,
    PRIMARY_GATES,
    RANDOM_CONTROL_95TH_INDEX,
    RANDOM_CONTROL_REPLICATES,
    REPAIR_CLAIM_BOUNDARY,
    SELECTOR_GATES,
    SIGN_AWARE_ADVANTAGE_MARGIN,
)
from experiments.e32_interpbench.e32_5_selector import ExactGroupEvaluation


def _edge_ids(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("fine_edge_ids must be a sequence")
    edge_ids = tuple(value)
    if (
        not edge_ids
        or len(edge_ids) != len(set(edge_ids))
        or any(not isinstance(edge_id, str) or not edge_id for edge_id in edge_ids)
    ):
        raise ValueError("fine_edge_ids must be unique nonempty strings")
    return edge_ids


def _canonical_support(
    edge_ids: tuple[str, ...], support: Sequence[str]
) -> tuple[str, ...]:
    if isinstance(support, (str, bytes)):
        raise TypeError("support must be a sequence")
    supplied = tuple(support)
    selected = frozenset(supplied)
    if (
        not selected
        or len(selected) != len(supplied)
        or not selected.issubset(edge_ids)
        or len(selected) == len(edge_ids)
    ):
        raise ValueError("support must be a nonempty proper unique subset of full U")
    return tuple(edge_id for edge_id in edge_ids if edge_id in selected)


def score_exact_group(
    *,
    fine_edge_ids: Sequence[str],
    support: Sequence[str],
    clean_output: Any,
    all_internal_output: Any,
    selected_output: Any,
    complement_output: Any,
    clean_target_indices: Any,
    gates: Mapping[str, float] = PRIMARY_GATES,
) -> ExactGroupEvaluation:
    r"""Score simultaneous S and the literal full-U complement U\S."""
    edge_ids = _edge_ids(fine_edge_ids)
    canonical = _canonical_support(edge_ids, support)
    selected = frozenset(canonical)
    complement = tuple(edge_id for edge_id in edge_ids if edge_id not in selected)
    metrics = set_faithfulness_metrics(
        clean_output=clean_output,
        all_corrupt_into_clean_output=all_internal_output,
        selected_corrupt_into_clean_output=selected_output,
        complement_corrupt_into_clean_output=complement_output,
        clean_target_indices=clean_target_indices,
    )
    expected_gate_keys = (
        "minimum_projection",
        "maximum_residual_error",
        "minimum_loss_recovery",
    )
    if tuple(gates) != expected_gate_keys:
        raise ValueError(f"gates must exactly follow {expected_gate_keys!r}")
    checks = faithfulness_gate_checks(
        metrics,
        minimum_projection=float(gates["minimum_projection"]),
        maximum_residual_error=float(gates["maximum_residual_error"]),
        minimum_loss_recovery=float(gates["minimum_loss_recovery"]),
    )
    return ExactGroupEvaluation(
        support=canonical,
        complement=complement,
        joint_faithfulness=joint_faithfulness_score(metrics),
        gate_pass=checks["pass"],
        selected_output=torch.as_tensor(selected_output).detach().cpu(),
        complement_output=torch.as_tensor(complement_output).detach().cpu(),
        metrics={**metrics, "gate_checks": checks},
    )


def internal_endpoint_status(
    *,
    clean_output: Any,
    all_internal_output: Any,
    clean_target_indices: Any,
) -> dict[str, Any]:
    """Authorize search only when the clean-to-F(U) endpoint is informative."""
    metrics = set_faithfulness_metrics(
        clean_output=clean_output,
        all_corrupt_into_clean_output=all_internal_output,
        selected_corrupt_into_clean_output=all_internal_output,
        complement_corrupt_into_clean_output=clean_output,
        clean_target_indices=clean_target_indices,
    )
    informative = bool(metrics["endpoint_informative"])
    return {
        "endpoint_informative": informative,
        "vector_gap_energy": float(metrics["vector_gap_energy"]),
        "clean_target_loss_gap": float(metrics["clean_target_loss_gap"]),
        "search_authorized": informative,
        "terminal_method_verdict": (
            "continue_to_signed_group_selection"
            if informative
            else "scientific_fail_uninformative_internal_endpoint"
        ),
    }


def _alignment_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("end_to_end_alignment must be a mapping")
    informative = value.get("endpoint_informative")
    observed_pass = value.get("alignment_pass")
    if not isinstance(informative, bool) or not isinstance(observed_pass, bool):
        raise TypeError(
            "endpoint informative and alignment pass flags must be booleans"
        )
    scalar_keys = ("projection_capture", "residual_error", "loss_capture")
    try:
        scalars = {
            key: None if value[key] is None else float(value[key])
            for key in scalar_keys
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("end-to-end alignment metrics are incomplete") from error
    finite = all(
        number is not None and math.isfinite(number) for number in scalars.values()
    )
    if informative and not finite:
        raise ValueError("informative end-to-end alignment metrics must be finite")
    expected_pass = (
        informative
        and finite
        and scalars["projection_capture"] is not None
        and scalars["projection_capture"]
        >= END_TO_END_ALIGNMENT_GATES["minimum_projection_capture"]
        and scalars["residual_error"] is not None
        and scalars["residual_error"]
        <= END_TO_END_ALIGNMENT_GATES["maximum_residual_error"]
        and scalars["loss_capture"] is not None
        and scalars["loss_capture"]
        >= END_TO_END_ALIGNMENT_GATES["minimum_loss_capture"]
    )
    if observed_pass is not expected_pass:
        raise ValueError("end-to-end alignment pass is inconsistent with frozen gates")
    return {
        "endpoint_informative": informative,
        **{
            key: number if number is not None and math.isfinite(number) else None
            for key, number in scalars.items()
        },
        "alignment_pass": expected_pass,
        "used_for_boundary_conditioned_primary_claim": False,
        "normalization": "clean_vs_actual_corrupt_input_endpoint",
    }


def score_end_to_end_alignment(
    *,
    clean_output: Any,
    all_internal_output: Any,
    actual_corrupt_output: Any,
    clean_target_indices: Any,
) -> dict[str, Any]:
    """Score F(U) against the actual corrupt-input endpoint as a separate tier."""
    metrics = endpoint_alignment_metrics(
        clean_output=clean_output,
        internal_anchor_output=all_internal_output,
        actual_corrupt_output=actual_corrupt_output,
        clean_target_indices=clean_target_indices,
    )
    return {**metrics, **_alignment_summary(metrics)}


def _finite_number(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _evaluate_case_record(
    *,
    case_id: int,
    evidence_role: str,
    claim_boundary: str,
    validation_joint: float,
    confirmation_joint: float,
    confirmation_absolute_pass: bool,
    end_to_end_alignment: Mapping[str, Any],
    random_joint_faithfulness: Sequence[float],
    random_95th_sorted_index: int,
    deterministic_comparator_joint: Mapping[str, float],
) -> dict[str, Any]:
    if isinstance(case_id, bool) or not isinstance(case_id, int):
        raise TypeError("case_id must be an integer")
    if not isinstance(confirmation_absolute_pass, bool):
        raise TypeError("confirmation_absolute_pass must be a boolean")
    validation = _finite_number(validation_joint, label="validation_joint")
    confirmation = _finite_number(confirmation_joint, label="confirmation_joint")
    random_values = tuple(
        _finite_number(value, label="random_joint_faithfulness")
        for value in random_joint_faithfulness
    )
    if (
        len(random_values) != RANDOM_CONTROL_REPLICATES
        or isinstance(random_95th_sorted_index, bool)
        or random_95th_sorted_index != RANDOM_CONTROL_95TH_INDEX
    ):
        raise ValueError("random controls must follow the frozen 256-replicate panel")
    comparators = {
        str(method): _finite_number(value, label=f"comparator {method}")
        for method, value in deterministic_comparator_joint.items()
    }
    if tuple(comparators) != DETERMINISTIC_BASELINE_ORDER:
        raise ValueError("deterministic comparators changed from the frozen order")
    alignment = _alignment_summary(end_to_end_alignment)
    random_95th = sorted(random_values)[random_95th_sorted_index]
    best_value = max(comparators.values())
    best_method = next(
        method
        for method in DETERMINISTIC_BASELINE_ORDER
        if comparators[method] == best_value
    )
    random_margin = confirmation - random_95th
    comparator_deficit = best_value - confirmation
    validation_pass = validation >= PRIMARY_GATES["minimum_projection"]
    absolute_pass = confirmation_absolute_pass
    random_pass = random_margin >= SELECTOR_GATES["minimum_margin_over_random_95th"]
    noninferiority_pass = (
        comparator_deficit <= SELECTOR_GATES["maximum_deficit_to_best_deterministic"]
    )
    selector_pass = absolute_pass and random_pass and noninferiority_pass
    sign_margin = confirmation - comparators[CONTEXTUAL_UNSIGNED_MASS_METHOD]
    interaction_margin = confirmation - comparators[CONTEXTUAL_SIGNED_STATIC_METHOD]
    return {
        "schema_version": "e32_5_case_score_v1",
        "case_id": case_id,
        "evidence_role": evidence_role,
        "claim_boundary": claim_boundary,
        "validation_joint_faithfulness": validation,
        "confirmation_joint_faithfulness": confirmation,
        "random_95th_joint_faithfulness": random_95th,
        "random_95th_sorted_index": random_95th_sorted_index,
        "margin_over_random_95th": random_margin,
        "best_deterministic_method": best_method,
        "best_deterministic_joint_faithfulness": best_value,
        "deficit_to_best_deterministic": comparator_deficit,
        "sign_aware_margin_over_unsigned_exact": sign_margin,
        "group_interaction_margin_over_static_directional": interaction_margin,
        "deterministic_comparator_joint_faithfulness": comparators,
        "end_to_end_alignment": alignment,
        "gates": {
            "validation_reached_primary_threshold": validation_pass,
            "confirmation_absolute_pass": confirmation_absolute_pass,
            "absolute_pass": absolute_pass,
            "end_to_end_alignment_pass": alignment["alignment_pass"],
            "random_superiority_pass": random_pass,
            "deterministic_noninferiority_pass": noninferiority_pass,
            "selector_pass": selector_pass,
            "sign_aware_advantage_pass": (
                absolute_pass and sign_margin >= SIGN_AWARE_ADVANTAGE_MARGIN
            ),
            "group_interaction_advantage_pass": (
                absolute_pass
                and interaction_margin >= GROUP_INTERACTION_ADVANTAGE_MARGIN
            ),
            "case_pass": selector_pass,
        },
    }


def evaluate_case(
    *,
    case_id: int,
    validation_joint: float,
    confirmation_joint: float,
    confirmation_absolute_pass: bool,
    end_to_end_alignment: Mapping[str, Any],
    random_joint_faithfulness: Sequence[float],
    random_95th_sorted_index: int,
    deterministic_comparator_joint: Mapping[str, float],
) -> dict[str, Any]:
    """Score one member of the untouched prospective scaling panel."""
    if isinstance(case_id, bool) or case_id not in GENERALIZATION_CASE_ORDER:
        raise ValueError("case_id is not in the prospective generalization cohort")
    return _evaluate_case_record(
        case_id=case_id,
        evidence_role="prospective_generalization",
        claim_boundary=CLAIM_BOUNDARY,
        validation_joint=validation_joint,
        confirmation_joint=confirmation_joint,
        confirmation_absolute_pass=confirmation_absolute_pass,
        end_to_end_alignment=end_to_end_alignment,
        random_joint_faithfulness=random_joint_faithfulness,
        random_95th_sorted_index=random_95th_sorted_index,
        deterministic_comparator_joint=deterministic_comparator_joint,
    )


def evaluate_repair_case(
    *,
    case_id: int,
    validation_joint: float,
    confirmation_joint: float,
    confirmation_absolute_pass: bool,
    end_to_end_alignment: Mapping[str, Any],
    random_joint_faithfulness: Sequence[float],
    random_95th_sorted_index: int,
    deterministic_comparator_joint: Mapping[str, float],
) -> dict[str, Any]:
    """Score the fresh case-58 repair without promoting it to panel evidence."""
    if case_id != 58:
        raise ValueError("repair evaluation is reserved for development case 58")
    return _evaluate_case_record(
        case_id=case_id,
        evidence_role="within_model_repair_only",
        claim_boundary=REPAIR_CLAIM_BOUNDARY,
        validation_joint=validation_joint,
        confirmation_joint=confirmation_joint,
        confirmation_absolute_pass=confirmation_absolute_pass,
        end_to_end_alignment=end_to_end_alignment,
        random_joint_faithfulness=random_joint_faithfulness,
        random_95th_sorted_index=random_95th_sorted_index,
        deterministic_comparator_joint=deterministic_comparator_joint,
    )


def evaluate_panel(case_scores: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Conjoin all claim cases and expose a separate limited scale tier."""
    expected = tuple(str(case_id) for case_id in GENERALIZATION_CASE_ORDER)
    if tuple(case_scores) != expected:
        raise ValueError("case_scores must exactly follow the prospective case order")
    absolute_passes: dict[int, bool] = {}
    selector_passes: dict[int, bool] = {}
    endpoint_alignment_passes: dict[int, bool] = {}
    sign_passes: list[bool] = []
    interaction_passes: list[bool] = []
    for case_id in GENERALIZATION_CASE_ORDER:
        record = case_scores[str(case_id)]
        if record.get("case_id") != case_id or not isinstance(
            record.get("gates"), Mapping
        ):
            raise ValueError("case score identity or gates are malformed")
        gates = record["gates"]
        required_boolean_gates = (
            "absolute_pass",
            "selector_pass",
            "end_to_end_alignment_pass",
        )
        if any(not isinstance(gates.get(key), bool) for key in required_boolean_gates):
            raise TypeError(
                "case score must expose boolean absolute, selector, and endpoint gates"
            )
        absolute_passes[case_id] = gates["absolute_pass"]
        selector_passes[case_id] = gates["selector_pass"]
        endpoint_alignment_passes[case_id] = gates["end_to_end_alignment_pass"]
        sign_passes.append(gates.get("sign_aware_advantage_pass") is True)
        interaction_passes.append(gates.get("group_interaction_advantage_pass") is True)
    absolute_count = sum(absolute_passes.values())
    selector_count = sum(selector_passes.values())
    absolute_generalization = all(absolute_passes.values())
    selector_generalization = all(selector_passes.values())
    absolute_scaled_replication = absolute_passes[82] and absolute_count >= 2
    selector_scaled_replication = selector_passes[82] and selector_count >= 2
    return {
        "schema_version": "e32_5_panel_verdict_v1",
        "absolute_faithfulness_generalization_pass": absolute_generalization,
        "selector_superiority_generalization_pass": selector_generalization,
        "generalization_pass": selector_generalization,
        "generalization_pass_definition": (
            "alias_of_selector_superiority_generalization_pass"
        ),
        "end_to_end_alignment_generalization_pass": all(
            endpoint_alignment_passes.values()
        ),
        "end_to_end_alignment_is_separate_from_boundary_conditioned_primary": True,
        "absolute_faithfulness_scaled_replication_pass": (absolute_scaled_replication),
        "selector_superiority_scaled_replication_pass": (selector_scaled_replication),
        "scaled_replication_pass": selector_scaled_replication,
        "scaled_replication_pass_definition": (
            "alias_of_selector_superiority_scaled_replication_pass"
        ),
        "absolute_scaled_replication_definition": (
            "case82_absolute_pass_and_at_least_two_of_three_absolute_pass;"
            "not_generalization"
        ),
        "selector_scaled_replication_definition": (
            "case82_selector_pass_and_at_least_two_of_three_selector_pass;"
            "not_generalization"
        ),
        "sign_aware_advantage_generalization_pass": all(sign_passes),
        "group_interaction_advantage_generalization_pass": all(interaction_passes),
        "absolute_case_passes": {
            str(case_id): value for case_id, value in absolute_passes.items()
        },
        "selector_case_passes": {
            str(case_id): value for case_id, value in selector_passes.items()
        },
        "all_cases_required": True,
        "majority_rule_allowed": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


__all__ = [
    "evaluate_case",
    "evaluate_panel",
    "evaluate_repair_case",
    "internal_endpoint_status",
    "score_end_to_end_alignment",
    "score_exact_group",
]
