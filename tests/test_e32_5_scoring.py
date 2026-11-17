# ruff: noqa: EXE002
from __future__ import annotations

import pytest
import torch

from experiments.e32_interpbench.e32_5_protocol import (
    DETERMINISTIC_BASELINE_ORDER,
    RANDOM_CONTROL_95TH_INDEX,
    RANDOM_CONTROL_REPLICATES,
)
from experiments.e32_interpbench.e32_5_scoring import (
    evaluate_case,
    evaluate_panel,
    evaluate_repair_case,
    internal_endpoint_status,
    score_end_to_end_alignment,
    score_exact_group,
)


def _logits(value: float) -> torch.Tensor:
    return torch.tensor([[[value, -value]]], dtype=torch.float64)


def _alignment_record(*, passes: bool = True) -> dict[str, object]:
    return {
        "endpoint_informative": True,
        "projection_capture": 0.90 if passes else 0.20,
        "residual_error": 0.01 if passes else 0.80,
        "loss_capture": 0.90 if passes else 0.20,
        "alignment_pass": passes,
    }


def test_exact_group_score_uses_full_u_partition_and_six_gates() -> None:
    score = score_exact_group(
        fine_edge_ids=("a", "b"),
        support=("a",),
        clean_output=_logits(1.0),
        all_internal_output=_logits(0.0),
        selected_output=_logits(0.05),
        complement_output=_logits(0.95),
        clean_target_indices=torch.tensor([[0]]),
    )
    assert score.support == ("a",)
    assert score.complement == ("b",)
    assert score.metrics["necessity_projection"] == pytest.approx(0.95)
    assert score.metrics["sufficiency_projection"] == pytest.approx(0.95)
    assert score.metrics["necessity_residual_error"] == pytest.approx(0.0025)
    assert score.metrics["sufficiency_residual_error"] == pytest.approx(0.0025)
    assert score.gate_pass is True


def test_end_to_end_alignment_is_a_separate_actual_corrupt_endpoint_tier() -> None:
    alignment = score_end_to_end_alignment(
        clean_output=_logits(1.0),
        all_internal_output=_logits(0.05),
        actual_corrupt_output=_logits(0.0),
        clean_target_indices=torch.tensor([[0]]),
    )
    assert alignment["endpoint_informative"] is True
    assert alignment["alignment_pass"] is True
    assert alignment["used_for_boundary_conditioned_primary_claim"] is False


def test_uninformative_internal_endpoint_is_a_scientific_terminal_verdict() -> None:
    status = internal_endpoint_status(
        clean_output=_logits(1.0),
        all_internal_output=_logits(1.0),
        clean_target_indices=torch.tensor([[0]]),
    )
    assert status == {
        "endpoint_informative": False,
        "vector_gap_energy": 0.0,
        "clean_target_loss_gap": 0.0,
        "search_authorized": False,
        "terminal_method_verdict": "scientific_fail_uninformative_internal_endpoint",
    }


def test_uninformative_actual_endpoint_round_trips_as_a_false_secondary_tier() -> None:
    alignment = score_end_to_end_alignment(
        clean_output=_logits(1.0),
        all_internal_output=_logits(1.0),
        actual_corrupt_output=_logits(1.0),
        clean_target_indices=torch.tensor([[0]]),
    )
    assert alignment["endpoint_informative"] is False
    assert alignment["projection_capture"] is None
    assert alignment["alignment_pass"] is False
    result = evaluate_case(
        case_id=63,
        validation_joint=0.91,
        confirmation_joint=0.92,
        confirmation_absolute_pass=True,
        end_to_end_alignment=alignment,
        random_joint_faithfulness=(0.10,) * RANDOM_CONTROL_REPLICATES,
        random_95th_sorted_index=RANDOM_CONTROL_95TH_INDEX,
        deterministic_comparator_joint=dict.fromkeys(
            DETERMINISTIC_BASELINE_ORDER, 0.90
        ),
    )
    assert result["gates"]["end_to_end_alignment_pass"] is False
    assert result["gates"]["case_pass"] is True


def test_case_gates_separate_absolute_selector_sign_and_interaction_claims() -> None:
    result = evaluate_case(
        case_id=63,
        validation_joint=0.20,
        confirmation_joint=0.92,
        confirmation_absolute_pass=True,
        end_to_end_alignment=_alignment_record(passes=False),
        random_joint_faithfulness=(0.10,) * RANDOM_CONTROL_REPLICATES,
        random_95th_sorted_index=RANDOM_CONTROL_95TH_INDEX,
        deterministic_comparator_joint=dict(
            zip(
                DETERMINISTIC_BASELINE_ORDER,
                (0.88, 0.80, 0.70, 0.60),
                strict=True,
            )
        ),
    )
    assert result["gates"]["validation_reached_primary_threshold"] is False
    assert result["gates"]["absolute_pass"] is True
    assert result["gates"]["end_to_end_alignment_pass"] is False
    assert result["gates"]["selector_pass"] is True
    assert result["gates"]["sign_aware_advantage_pass"] is True
    assert result["gates"]["group_interaction_advantage_pass"] is False
    assert result["gates"]["case_pass"] is True


def test_panel_is_all_case_conjunctive_and_excludes_development_case() -> None:
    passing = {
        "case_id": 63,
        "gates": {
            "absolute_pass": True,
            "selector_pass": True,
            "end_to_end_alignment_pass": True,
        },
    }
    failing = {
        "case_id": 103,
        "gates": {
            "absolute_pass": True,
            "selector_pass": False,
            "end_to_end_alignment_pass": False,
        },
    }
    other = {
        "case_id": 82,
        "gates": {
            "absolute_pass": True,
            "selector_pass": True,
            "end_to_end_alignment_pass": True,
        },
    }
    verdict = evaluate_panel({"63": passing, "103": failing, "82": other})
    assert verdict["all_cases_required"] is True
    assert verdict["absolute_faithfulness_generalization_pass"] is True
    assert verdict["selector_superiority_generalization_pass"] is False
    assert verdict["generalization_pass"] is False
    assert verdict["end_to_end_alignment_generalization_pass"] is False
    assert verdict["generalization_pass_definition"] == (
        "alias_of_selector_superiority_generalization_pass"
    )
    assert verdict["absolute_faithfulness_scaled_replication_pass"] is True
    assert verdict["selector_superiority_scaled_replication_pass"] is True
    assert verdict["scaled_replication_pass"] is True
    assert verdict["scaled_replication_pass_definition"] == (
        "alias_of_selector_superiority_scaled_replication_pass"
    )
    with pytest.raises(ValueError, match="prospective case order"):
        evaluate_panel({"58": passing, "63": passing, "103": failing, "82": other})


def test_case58_repair_has_a_separate_non_generalization_verdict() -> None:
    comparators = dict(
        zip(
            DETERMINISTIC_BASELINE_ORDER,
            (0.80, 0.70, 0.60, 0.50),
            strict=True,
        )
    )
    repair = evaluate_repair_case(
        case_id=58,
        validation_joint=0.91,
        confirmation_joint=0.92,
        confirmation_absolute_pass=True,
        end_to_end_alignment=_alignment_record(),
        random_joint_faithfulness=(0.10,) * RANDOM_CONTROL_REPLICATES,
        random_95th_sorted_index=RANDOM_CONTROL_95TH_INDEX,
        deterministic_comparator_joint=comparators,
    )
    assert repair["evidence_role"] == "within_model_repair_only"
    assert repair["gates"]["case_pass"] is True
    assert "not_prospective_panel_evidence" in repair["claim_boundary"]


def test_tied_deterministic_comparator_uses_frozen_order() -> None:
    result = evaluate_case(
        case_id=63,
        validation_joint=0.91,
        confirmation_joint=0.92,
        confirmation_absolute_pass=True,
        end_to_end_alignment=_alignment_record(),
        random_joint_faithfulness=(0.10,) * RANDOM_CONTROL_REPLICATES,
        random_95th_sorted_index=RANDOM_CONTROL_95TH_INDEX,
        deterministic_comparator_joint=dict.fromkeys(
            DETERMINISTIC_BASELINE_ORDER, 0.90
        ),
    )
    assert result["best_deterministic_method"] == DETERMINISTIC_BASELINE_ORDER[0]


@pytest.mark.parametrize("tampered_flag", [1, "false"])
def test_absolute_confirmation_flag_rejects_non_booleans(tampered_flag: object) -> None:
    with pytest.raises(TypeError, match="confirmation_absolute_pass"):
        evaluate_case(
            case_id=63,
            validation_joint=0.91,
            confirmation_joint=0.92,
            confirmation_absolute_pass=tampered_flag,  # type: ignore[arg-type]
            end_to_end_alignment=_alignment_record(),
            random_joint_faithfulness=(0.10,) * RANDOM_CONTROL_REPLICATES,
            random_95th_sorted_index=RANDOM_CONTROL_95TH_INDEX,
            deterministic_comparator_joint=dict.fromkeys(
                DETERMINISTIC_BASELINE_ORDER, 0.90
            ),
        )
