# ruff: noqa: EXE002
from __future__ import annotations

import hashlib
import json

import pytest

from experiments.e32_interpbench.e32_5_protocol import (
    ADDITION_DIRECTIONAL_QUOTA,
    ADDITION_RESIDUAL_QUOTA,
    CASE_GEOMETRY,
    DETERMINISTIC_BASELINE_ORDER,
    DEVELOPMENT_CASE_ORDER,
    GENERALIZATION_CASE_ORDER,
    LOCAL_SEARCH_ROUNDS,
    PRIMARY_GATES,
    RANDOM_CONTROL_NAMESPACE,
    RANDOM_CONTROL_REPLICATES,
    RANDOM_CONTROL_SEED_SHA256,
    REMOVAL_DIRECTIONAL_QUOTA,
    REMOVAL_RESIDUAL_QUOTA,
    SEARCH_ROW_BOUNDS,
    VALIDATION_ROW_BOUNDS,
    confirmation_support_size,
    deterministic_random_supports,
    protocol_manifest,
    selector_query_ceiling,
)


def _manifest() -> dict[str, object]:
    return protocol_manifest(
        development_tuning_disclosure="none; model-free protocol unit-test fixture"
    )


def test_full_universe_support_sizes_and_search_budget_are_frozen() -> None:
    assert DEVELOPMENT_CASE_ORDER == (58,)
    assert GENERALIZATION_CASE_ORDER == (63, 103, 82)
    assert {
        case_id: geometry.full_fine_edge_count
        for case_id, geometry in CASE_GEOMETRY.items()
    } == {
        58: 207,
        63: 73,
        103: 207,
        82: 406,
    }
    assert {
        case_id: confirmation_support_size(case_id) for case_id in CASE_GEOMETRY
    } == {
        58: 31,
        63: 10,
        103: 31,
        82: 60,
    }
    assert LOCAL_SEARCH_ROUNDS == 3
    assert (REMOVAL_RESIDUAL_QUOTA, REMOVAL_DIRECTIONAL_QUOTA) == (2, 2)
    assert (ADDITION_RESIDUAL_QUOTA, ADDITION_DIRECTIONAL_QUOTA) == (4, 4)
    largest = selector_query_ceiling(82)
    assert largest["singleton_partition_physical_forwards"] == 816
    assert largest["maximum_adaptive_exact_acquisitions"] == 4
    assert largest["maximum_selector_physical_forwards"] == 1026
    assert largest["validation_additional_model_masks"] == 0
    assert largest["dense_complement_active_mask_coordinates"] == 164430


def test_manifest_separates_exposed_development_from_claim_cases() -> None:
    manifest = _manifest()
    assert (
        manifest["development_ancestry"]["case58_is_nonclaim_method_development"]
        is True
    )
    assert (
        manifest["development_ancestry"]["e32_4_outputs_are_development_only"] is True
    )
    assert manifest["prospective_claim"]["case_order"] == [63, 103, 82]
    assert manifest["prospective_claim"]["all_cases_required"] is True
    assert manifest["selector"]["selectable_universe"] == "U_all_internal_fine_edges"
    assert manifest["selector"]["residual_proxy_may_choose_or_stop"] is False
    assert manifest["selector"]["confirmation_support_fraction"] == 0.15
    assert SEARCH_ROW_BOUNDS == (0, 32)
    assert VALIDATION_ROW_BOUNDS == (32, 64)
    assert manifest["splits"]["support_selection_rows"] == [0, 32]
    assert manifest["splits"]["validation_path_selection_rows"] == [32, 64]
    assert manifest["baselines"]["deterministic_support_selection_rows"] == [0, 32]
    assert manifest["baselines"]["random_support_selection_is_response_blind"] is True
    assert DETERMINISTIC_BASELINE_ORDER[0] == "contextual_signed_partition_static"
    assert DETERMINISTIC_BASELINE_ORDER[1] == "contextual_unsigned_partition_mass"
    assert PRIMARY_GATES == {
        "minimum_projection": 0.90,
        "maximum_residual_error": 0.10,
        "minimum_loss_recovery": 0.90,
    }
    lifecycle = manifest["development_pilot_lifecycle"]
    assert lifecycle["attempt_path"] == (
        "notes/E32.5-SIGNED-GROUP-EDGE-SELECTOR-DEVELOPMENT-ATTEMPT.json"
    )
    assert lifecycle["final_algorithm_and_source_freeze_occurs_after_pilot"] is True
    assert lifecycle["case58_pilot_is_excluded_from_every_claim"] is True
    assert lifecycle["final_run_reruns_case58_pilot"] is False
    assert lifecycle["development_tuning_disclosure"] == (
        "none; model-free protocol unit-test fixture"
    )


def test_final_manifest_requires_a_literal_postpilot_tuning_disclosure() -> None:
    with pytest.raises(ValueError, match="development_tuning_disclosure"):
        protocol_manifest(development_tuning_disclosure="")
    with pytest.raises(ValueError, match="development_tuning_disclosure"):
        protocol_manifest(development_tuning_disclosure="pending pilot")


def test_random_controls_use_a_frozen_e32_5_namespace_and_unique_uniform_subsets() -> (
    None
):
    assert RANDOM_CONTROL_NAMESPACE == (
        "e32.5/random-controls/v1/sha256-rejection-unrank"
    )
    assert RANDOM_CONTROL_SEED_SHA256 == (
        "8dee1d79fff2f82a0f3e5ea0561596f07b32c4cbcac2e5977b24af313a020bf9"
    )
    edge_ids = tuple(f"fine:{index:03d}" for index in range(73))
    supports = deterministic_random_supports(63, edge_ids, 10)
    assert supports == deterministic_random_supports(63, edge_ids, 10)
    assert len(supports) == RANDOM_CONTROL_REPLICATES
    assert len(set(supports)) == RANDOM_CONTROL_REPLICATES
    supports_sha256 = hashlib.sha256(
        json.dumps(supports, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert supports_sha256 == (
        "8ad819f36869e2cbc19780a389fcf042b27a42e6bebe4bf02016b044d660bdbc"
    )
    canonical_index = {edge_id: index for index, edge_id in enumerate(edge_ids)}
    for support in supports:
        assert len(support) == 10
        assert tuple(sorted(support, key=canonical_index.__getitem__)) == support
        assert set(support).issubset(edge_ids)
    sampler = _manifest()["baselines"]["random_control_sampler"]
    assert sampler["namespace"] == RANDOM_CONTROL_NAMESPACE
    assert sampler["seed_sha256"] == RANDOM_CONTROL_SEED_SHA256
    assert sampler["sampling"] == ("exact_rejection_sampled_uniform_combination_rank")


def test_random_controls_fail_closed_on_noncanonical_inputs() -> None:
    edge_ids = tuple(f"fine:{index:03d}" for index in range(73))
    with pytest.raises(ValueError, match="registered support size"):
        deterministic_random_supports(63, edge_ids, 9)
    with pytest.raises(ValueError, match="full fine universe size"):
        deterministic_random_supports(63, edge_ids[:-1], 10)
    with pytest.raises(ValueError, match="unique nonempty strings"):
        deterministic_random_supports(63, (*edge_ids[:-1], edge_ids[0]), 10)
