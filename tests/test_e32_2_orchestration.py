# ruff: noqa: EXE002
"""Model-free custody and accounting tests for E32.2 orchestration."""

from __future__ import annotations

import pytest

from experiments.e32_interpbench.e32_2_orchestration import (
    aggregate_actual_query_accounting,
    assert_complete_fresh_panel,
)


def _bundle(case_id: int, split: str, offset: int) -> dict[str, object]:
    count = 64 if split == "discovery" else 128
    rows = tuple((offset + index, offset + index + 1) for index in range(2 * count))
    return {
        "schema_version": "e32_2_split_acquisition_v1",
        "case_id": case_id,
        "split": split,
        "selected_token_rows": rows,
        "raw_pair_atlas": {
            "clean_inputs": rows[:count],
            "corrupt_inputs": rows[count:],
        },
        "actual_query_accounting": {
            "behavior_and_raw_atlas": {
                "physical_forwards": 10,
                "model_examples": 1_000,
            },
            "directed_replacement": {
                "observed_physical_forwards": 20,
                "observed_model_examples": 2_000,
                "observed_no_grad_forwards": 20,
                "observed_grad_enabled_forwards": 0,
                "backward_passes": 0,
            },
            "official_eap": {
                "physical_forwards": 2,
                "model_examples": 200,
                "no_grad_forwards": 1,
                "grad_enabled_forwards": 1,
                "backward_passes": 1,
                "matches_official_control_flow": True,
            },
            "official_eap_ig": {
                "physical_forwards": 7,
                "model_examples": 700,
                "no_grad_forwards": 1,
                "grad_enabled_forwards": 6,
                "backward_passes": 6,
                "matches_official_control_flow": True,
            },
            "operational_budget_status": "reported_separately_by_protocol",
        },
    }


def _panel() -> dict[tuple[int, str], dict[str, object]]:
    return {
        (2, "discovery"): _bundle(2, "discovery", 1_000),
        (2, "confirmation"): _bundle(2, "confirmation", 10_000),
        (21, "discovery"): _bundle(21, "discovery", 100_000),
        (21, "confirmation"): _bundle(21, "confirmation", 1_000_000),
    }


def _prior_rows(case_offset: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (case_offset + 2 * index, case_offset + 2 * index + 1) for index in range(480)
    )


def test_complete_panel_requires_both_cases_and_both_splits_with_fresh_roles() -> None:
    panel = _panel()
    prior = {
        2: _prior_rows(10_000_000),
        21: _prior_rows(20_000_000),
    }

    audit = assert_complete_fresh_panel(panel, prior_rows_by_case=prior)

    assert tuple(audit["case_order"]) == (2, 21)
    assert audit["all_cases_mandatory"] is True
    assert audit["total_selected_role_rows"] == 2 * (128 + 256)
    assert all(
        case["cross_split_overlap_count"] == 0 for case in audit["cases"].values()
    )

    missing = dict(panel)
    missing.pop((21, "confirmation"))
    with pytest.raises(ValueError, match="complete mandatory panel"):
        assert_complete_fresh_panel(missing, prior_rows_by_case=prior)


def test_panel_rejects_prior_or_cross_split_leakage() -> None:
    panel = _panel()
    prior_overlap = {
        2: ((1_000, 1_001), *_prior_rows(10_000_000)[:479]),
        21: _prior_rows(20_000_000),
    }
    with pytest.raises(RuntimeError, match="token-row leakage"):
        assert_complete_fresh_panel(panel, prior_rows_by_case=prior_overlap)

    confirmation = dict(panel[(2, "confirmation")])
    confirmation_rows = list(confirmation["selected_token_rows"])
    confirmation_rows[0] = panel[(2, "discovery")]["selected_token_rows"][0]
    confirmation["selected_token_rows"] = tuple(confirmation_rows)
    overlapping = {**panel, (2, "confirmation"): confirmation}
    with pytest.raises(RuntimeError, match="token-row leakage"):
        assert_complete_fresh_panel(
            overlapping,
            prior_rows_by_case={
                2: _prior_rows(10_000_000),
                21: _prior_rows(20_000_000),
            },
        )

    mismatched = _panel()
    discovery = dict(mismatched[(2, "discovery")])
    raw = dict(discovery["raw_pair_atlas"])
    clean_rows = list(raw["clean_inputs"])
    clean_rows[0] = (99_999_999, 99_999_998)
    raw["clean_inputs"] = tuple(clean_rows)
    discovery["raw_pair_atlas"] = raw
    mismatched[(2, "discovery")] = discovery
    with pytest.raises(RuntimeError, match="custody rows differ from scored"):
        assert_complete_fresh_panel(
            mismatched,
            prior_rows_by_case={
                2: _prior_rows(10_000_000),
                21: _prior_rows(20_000_000),
            },
        )


def test_actual_accounting_is_observed_and_separate_from_operational_budget() -> None:
    accounting = aggregate_actual_query_accounting(_panel())

    assert accounting["schema_version"] == "e32_2_actual_query_accounting_v1"
    assert accounting["physical_forwards"] == 4 * (10 + 20 + 2 + 7)
    assert accounting["model_examples"] == 4 * (1_000 + 2_000 + 200 + 700)
    assert accounting["backward_passes"] == 4 * 7
    assert accounting["grad_enabled_forwards"] == 4 * 7
    assert accounting["no_grad_forwards"] == 4 * (10 + 20 + 1 + 1)
    assert (
        accounting["no_grad_forwards"] + accounting["grad_enabled_forwards"]
        == accounting["physical_forwards"]
    )
    assert accounting["operational_budget_included"] is False

    tampered = _panel()
    broken = dict(tampered[(2, "discovery")])
    broken_accounting = dict(broken["actual_query_accounting"])
    broken_eap = dict(broken_accounting["official_eap"])
    broken_eap["matches_official_control_flow"] = False
    broken_accounting["official_eap"] = broken_eap
    broken["actual_query_accounting"] = broken_accounting
    tampered[(2, "discovery")] = broken
    with pytest.raises(ValueError, match="official control flow"):
        aggregate_actual_query_accounting(tampered)
