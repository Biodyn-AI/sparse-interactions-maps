# ruff: noqa: EXE002
"""Pure lifecycle/custody tests for the prospective E32.3 cohort."""

from __future__ import annotations

import pytest
import torch

from experiments.e32_interpbench.e32_3_cohort import (
    CASE_ORDER,
    EXPECTED_RESPONSE_BLIND_CAPACITY,
    case_universe,
)
from experiments.e32_interpbench.e32_3_orchestration import (
    acquisition_supports_from_frozen,
    aggregate_actual_query_accounting,
    freeze_discovery_supports,
    validate_exact_panel_alignment,
    validate_global_behavior_screen,
    validate_split_request_alignment,
)
from experiments.e32_interpbench.e32_3_protocol import (
    DETERMINISTIC_METHOD_ORDER,
    RANDOM_CONTROL_REPLICATES,
    confirmatory_support_size,
)


def _discovery(case_id: int) -> dict[str, object]:
    nodes, _edges = case_universe(case_id)
    scores = torch.arange(len(nodes), 0, -1, dtype=torch.float64)
    return {
        "schema_version": "e32_3_discovery_acquisition_v1",
        "case_id": case_id,
        "split": "discovery",
        "component_ids": nodes,
        "selector_scores": {
            method: scores + index / 100
            for index, method in enumerate(DETERMINISTIC_METHOD_ORDER)
        },
    }


def test_discovery_freezes_all_deterministic_curve_and_random_supports() -> None:
    frozen = freeze_discovery_supports(case_id=25, discovery=_discovery(25))
    nodes, _edges = case_universe(25)
    half = confirmatory_support_size(len(nodes))

    assert tuple(frozen["deterministic_supports"]) == DETERMINISTIC_METHOD_ORDER
    assert tuple(frozen["singleton_curve_supports"]) == (3, 5, 8)
    assert len(frozen["random_supports"]) == RANDOM_CONTROL_REPLICATES
    assert len(frozen["all_supports"]) == (
        len(DETERMINISTIC_METHOD_ORDER) + 2 + RANDOM_CONTROL_REPLICATES
    )
    assert len({record["support_id"] for record in frozen["all_supports"]}) == len(
        frozen["all_supports"]
    )
    assert all(
        len(record["selected_nodes"]) == half for record in frozen["random_supports"]
    )
    assert (
        frozen["singleton_curve_supports"][half]["support_id"]
        == frozen["deterministic_supports"]["singleton_output_effect"]["support_id"]
    )

    acquisition = acquisition_supports_from_frozen(frozen)
    assert tuple(support.method for support in acquisition[:7]) == (
        "singleton_output_effect",
        "singleton_output_effect",
        "singleton_output_effect",
        *DETERMINISTIC_METHOD_ORDER[1:],
    )
    assert all(support.method == "random_control" for support in acquisition[7:])
    assert tuple(support.replicate_index for support in acquisition[7:]) == tuple(
        range(RANDOM_CONTROL_REPLICATES)
    )


def _screen_case(case_id: int) -> dict[str, object]:
    # Token IDs are task-local, but every row identity is globally unique
    # within a case across smoke, D clean/source, and C clean/source.
    cursor = case_id * 10_000

    def rows(count: int) -> tuple[tuple[int, int], ...]:
        nonlocal cursor
        values = tuple((cursor + index, case_id) for index in range(count))
        cursor += count
        return values

    return {
        "case_id": case_id,
        "behavior_only": True,
        "smoke_rows": rows(
            EXPECTED_RESPONSE_BLIND_CAPACITY[case_id]["smoke_unique_rows"]
        ),
        "discovery_clean_rows": rows(64),
        "discovery_corrupt_rows": rows(64),
        "confirmation_clean_rows": rows(128),
        "confirmation_corrupt_rows": rows(128),
    }


def test_global_behavior_screen_requires_all_cases_and_no_within_case_leakage() -> None:
    screen = {str(case_id): _screen_case(case_id) for case_id in CASE_ORDER}
    audit = validate_global_behavior_screen(screen)
    assert audit["case_order"] == CASE_ORDER
    assert audit["total_scientific_role_rows"] == 3 * (2 * 64 + 2 * 128)

    bad = {key: dict(value) for key, value in screen.items()}
    bad["58"]["confirmation_clean_rows"] = (
        bad["58"]["discovery_clean_rows"]
        + tuple(bad["58"]["confirmation_clean_rows"])[64:]
    )
    try:
        validate_global_behavior_screen(bad)
    except RuntimeError as error:
        assert "leakage" in str(error)
    else:  # pragma: no cover - fail with a useful message
        raise AssertionError("cross-split leakage was accepted")


def test_actual_query_accounting_sums_only_observed_integer_counters() -> None:
    def split_accounting(case_id: int, split: str) -> dict[str, dict[str, int]]:
        nodes, edges = case_universe(case_id)
        batch = 64 if split == "discovery" else 128
        fine = 207 if len(nodes) == 15 else 73
        accounting = {
            "eligibility": {
                "physical_forwards": 2,
                "model_examples": 192 if split == "discovery" else 384,
            },
            "preparation": {
                "physical_forwards": 4,
                "model_examples": 4 * batch,
            },
            "atlas_smoke": {
                "physical_forwards": 9 if split == "discovery" else 0,
                "model_examples": 16 * batch if split == "discovery" else 0,
            },
            "atlas": {
                "physical_forwards": (len(nodes) + len(edges) + 31) // 32,
                "model_examples": (len(nodes) + len(edges)) * batch,
            },
            "exact_fine": {
                "physical_forwards": 4 + 2 * fine,
                "model_examples": (4 + 2 * fine) * batch,
            },
            "eap_ig5": {
                "physical_forwards": 7,
                "model_examples": 7 * batch,
                "backward_passes_inferred_from_pinned_control_flow": 6,
                "gradient_examples": 6 * batch,
            },
        }
        if split == "confirmation":
            accounting["set_interventions"] = {
                "physical_forwards": 5,
                "model_examples": 143 * batch,
            }
        return accounting

    panel = {
        (case_id, split): {"query_accounting": split_accounting(case_id, split)}
        for case_id in CASE_ORDER
        for split in ("discovery", "confirmation")
    }
    accounting = aggregate_actual_query_accounting(panel)
    assert accounting["totals"]["physical_forwards"] == 2_112
    assert accounting["totals"]["model_examples"] == 302_784
    assert (
        accounting["totals"]["backward_passes_inferred_from_pinned_control_flow"] == 36
    )
    assert accounting["totals"]["gradient_examples"] == 3_456
    assert accounting["panel_count"] == 6


def test_cohort_and_protocol_split_requests_are_exactly_aligned() -> None:
    audit = validate_split_request_alignment()
    assert audit["case_order"] == CASE_ORDER
    assert audit["all_requests_equal"] is True


def test_exact_and_eap_same_panel_identity_tampering_is_rejected() -> None:
    endpoint = torch.zeros(2, 1, 2)
    bundle = {
        "fine_edge_names": ("q", "k", "v"),
        "eap_fine_edge_names": ("q", "k", "v"),
        "fine_to_whole_indices": (0, 0, 0),
        "clean_output": endpoint,
        "corrupt_output": endpoint + 1.0,
        "exact_clean_output": endpoint.clone(),
        "exact_corrupt_output": endpoint + 1.0,
    }
    validate_exact_panel_alignment(bundle)

    changed_names = {**bundle, "eap_fine_edge_names": ("q", "v", "k")}
    with pytest.raises(ValueError, match="fine identities"):
        validate_exact_panel_alignment(changed_names)
    changed_endpoint = {
        **bundle,
        "exact_clean_output": endpoint + 2e-6,
    }
    with pytest.raises(ValueError, match="canonical endpoints"):
        validate_exact_panel_alignment(changed_endpoint)
