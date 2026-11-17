# ruff: noqa: EXE002
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from experiments.e32_interpbench.adapter import OfficialDataPair
from experiments.e32_interpbench.e32_3_cohort import (
    CASE_MANIFESTS as E32_3_CASE_MANIFESTS,
)
from experiments.e32_interpbench.e32_5_cohort import (
    CASE58_HISTORICAL_RAW_REQUESTS,
    CASE_GEOMETRY,
    CASE_MANIFESTS,
    CASE_ORDER,
    DESIGN_BEHAVIOR_EXPOSURE,
    GENERALIZATION_CASE_ORDER,
    SPLIT_ORDER_BY_CASE,
    InputPreflight,
    _json_sha256,
    behaviorally_select_preflight,
    raw_identity_payload,
    response_blind_selectable_indices,
    split_request,
    target_identity_payload,
)


def _pair(
    clean_rows: list[list[int]], corrupt_rows: list[list[int]]
) -> OfficialDataPair:
    clean = torch.tensor(clean_rows, dtype=torch.long)
    corrupt = torch.tensor(corrupt_rows, dtype=torch.long)
    classes = 5
    clean_targets = torch.nn.functional.one_hot(
        clean.remainder(classes), num_classes=classes
    ).to(torch.float32)
    corrupt_targets = torch.nn.functional.one_hot(
        corrupt.remainder(classes), num_classes=classes
    ).to(torch.float32)
    return OfficialDataPair(clean, clean_targets, corrupt, corrupt_targets)


def test_frozen_cohort_geometry_and_claim_roles() -> None:
    assert CASE_ORDER == (58, 63, 103, 82)
    assert GENERALIZATION_CASE_ORDER == (63, 103, 82)
    assert SPLIT_ORDER_BY_CASE == {
        58: ("development", "repair_confirmation"),
        63: ("discovery", "confirmation"),
        103: ("discovery", "confirmation"),
        82: ("discovery", "confirmation"),
    }
    assert {
        case_id: (
            geometry["nodes"],
            geometry["whole_edges"],
            geometry["full_fine_edge_count"],
        )
        for case_id, geometry in CASE_GEOMETRY.items()
    } == {
        58: (15, 87, 207),
        63: (10, 33, 73),
        103: (15, 87, 207),
        82: (20, 166, 406),
    }
    assert CASE_MANIFESTS[58].claim_role == "within_model_repair_only"
    assert all(
        CASE_MANIFESTS[case_id].claim_role == "prospective_model_generalization"
        for case_id in GENERALIZATION_CASE_ORDER
    )


def test_case58_reuses_exact_verified_artifact_manifest() -> None:
    assert CASE_MANIFESTS[58].sha256 == E32_3_CASE_MANIFESTS[58].sha256
    assert len(CASE58_HISTORICAL_RAW_REQUESTS) == 5
    assert (
        sum(request.candidate_count * 2 for request in CASE58_HISTORICAL_RAW_REQUESTS)
        == 1_344
    )
    assert DESIGN_BEHAVIOR_EXPOSURE["total"] == {
        "physical_forwards": 40,
        "model_examples": 5_760,
        "gradient_enabled_forwards": 0,
        "gradient_examples": 0,
        "backward_passes": 0,
        "patched_intervention_model_examples": 0,
    }


def test_frozen_split_requests_and_preflight_capacities() -> None:
    expected = {
        (58, "development"): (64, 96, 5_005_810, 5_005_811, 96, 96),
        (58, "repair_confirmation"): (
            128,
            192,
            5_005_820,
            5_005_821,
            192,
            192,
        ),
        (63, "discovery"): (64, 96, 5_006_310, 5_006_311, 96, 96),
        (63, "confirmation"): (
            128,
            192,
            5_006_320,
            5_006_321,
            192,
            190,
        ),
        (103, "discovery"): (64, 96, 5_010_310, 5_010_311, 96, 96),
        (103, "confirmation"): (
            128,
            192,
            5_010_320,
            5_010_321,
            192,
            192,
        ),
        (82, "discovery"): (64, 96, 5_008_210, 5_008_211, 96, 96),
        (82, "confirmation"): (
            128,
            192,
            5_008_220,
            5_008_221,
            192,
            192,
        ),
    }
    for identity, values in expected.items():
        request = split_request(*identity)
        assert (
            request.required_count,
            request.candidate_count,
            request.clean_seed,
            request.corrupt_seed,
            request.response_blind_capacity,
            request.behaviorally_selectable_capacity,
        ) == values
        assert len(request.raw_identity_sha256) == 64
        assert len(request.target_identity_sha256) == 64

    with pytest.raises(ValueError, match="case_id"):
        split_request(999, "discovery")
    with pytest.raises(ValueError, match="split"):
        split_request(63, "development")


def test_response_blind_selection_rejects_exclusions_and_role_reuse() -> None:
    pair = _pair(
        [[1, 1], [2, 2], [2, 2], [3, 3], [4, 4]],
        [[5, 5], [6, 6], [7, 7], [3, 3], [8, 8]],
    )
    indices = response_blind_selectable_indices(
        pair, excluded_token_rows=((1, 1), (9, 9))
    )
    assert indices == (1, 4)


def test_identity_payloads_bind_rows_targets_and_schema() -> None:
    request = split_request(63, "discovery")
    pair = _pair([[1, 2], [3, 4]], [[4, 3], [2, 1]])
    raw = raw_identity_payload(request, pair)
    targets = target_identity_payload(request, pair)
    assert raw["schema_version"] == "e32_5_raw_candidate_pool_v1"
    assert targets["schema_version"] == "e32_5_target_identity_v1"
    assert raw["clean_input_rows"] == ((1, 2), (3, 4))
    assert targets["clean_target_argmax_rows"] == ((1, 2), (3, 4))
    assert _json_sha256(raw) != _json_sha256(targets)


class _PerfectModel:
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(inputs.remainder(5), 5).to(torch.float32)

    def to(self, _device: str) -> _PerfectModel:
        return self

    def eval(self) -> _PerfectModel:
        return self


def test_behavior_selection_uses_frozen_preflight_without_regeneration() -> None:
    pair = _pair(
        [[1, 1], [2, 2], [3, 3]],
        [[4, 4], [0, 0], [1, 2]],
    )
    request = replace(
        split_request(63, "discovery"),
        required_count=2,
        candidate_count=3,
        response_blind_capacity=3,
        behaviorally_selectable_capacity=3,
    )
    preflight = InputPreflight(
        request=request,
        pair=pair,
        clean_unique_rows=3,
        corrupt_unique_rows=3,
        role_union_unique_rows=6,
        excluded_unique_rows=0,
        overlap_with_exclusions=0,
        response_blind_selectable_indices=(0, 1, 2),
        observed_raw_identity_sha256="0" * 64,
        observed_target_identity_sha256="1" * 64,
    )
    eligible = behaviorally_select_preflight(_PerfectModel(), preflight)
    assert eligible.selected_candidate_indices == (0, 1)
    assert eligible.eligibility_physical_forwards == 2
    assert eligible.eligibility_model_examples == 6
    assert eligible.row_disjoint_eligible_candidate_count == 3
