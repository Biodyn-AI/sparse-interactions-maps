# ruff: noqa: EXE002
"""Tests for the prospective E32.3 organism cohort and row custody."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from experiments.e32_interpbench.adapter import DEFAULT_MODEL_ROOT
from experiments.e32_interpbench.e32_3_cohort import (
    CASE_MANIFESTS,
    CASE_ORDER,
    EXPECTED_RESPONSE_BLIND_CAPACITY,
    SMOKE_REQUESTS,
    CasePanel,
    _acquire_case_panel,
    case_universe,
    load_exact_ground_truth,
    project_categorical_output,
    smoke_excluded_rows,
    split_request,
    validate_case_panel,
    verify_case_artifacts,
)

EXPECTED_HASHES = {
    25: {
        "edges.pkl": "bd358562dc56469dbf03912dac0c5d3d5489b0f2dadcb85a7cd7c24f9582a74a",
        "ll_model_cfg.pkl": "c5df91fe7e78fbfc9fe45fe234027fb7c15ecd4c9a0b7a4776da0bf067e09db5",
        "ll_model.pth": "0e72ecd925276352d1240da0b40019b38ef254cbfdad61b234d96672fc4ae2e2",
        "meta.json": "9760198867e197a1aeba2e60440ad37e236f57c14f21c6e7d73a187b67b34c0c",
    },
    58: {
        "edges.pkl": "4edacbb2f328f7d3526a6658a58484634a73124ea79cace68231ae0bb1ab20ea",
        "ll_model_cfg.pkl": "8cc71fd34570fd0bef7a3f511b8675d7cbaec6db017ce3cfbee7a74f8a23e501",
        "ll_model.pth": "95278b3bf4897a27606f35ae5ed251fb022d9e4158f7575d32ab42d6fa142e20",
        "meta.json": "1cfffe661e1435f142d96ab22909dc78623087ec98adbc0a8bda025fda348881",
    },
    129: {
        "edges.pkl": "b01d4b9b1e5740fd2934079065cb57785bb111918826a8e8ad7b5b4af00c7bcc",
        "ll_model_cfg.pkl": "a4c611ecd0f8c57d44f555697540a82578e20f4dbe88bb1bd25132bd06244fa2",
        "ll_model.pth": "f497c70d4f781598a2d86469fc884a7aee35013d2ca2851e03a40077224fc6ab",
        "meta.json": "d9329a1cbcd6819eaab7573093a4eb96ed417c9efd2206270517d6897eb93c5b",
    },
}


def test_case_order_manifests_and_artifact_hashes_are_exact() -> None:
    assert CASE_ORDER == (58, 129, 25)
    assert set(CASE_MANIFESTS) == {25, 58, 129}
    for case_id, expected_hashes in EXPECTED_HASHES.items():
        manifest = CASE_MANIFESTS[case_id]
        assert manifest.output_kind == "categorical"
        assert dict(manifest.sha256) == expected_hashes


def test_case_geometries_and_ground_truth_counts_are_frozen() -> None:
    assert (
        CASE_MANIFESTS[58].n_layers,
        CASE_MANIFESTS[58].n_heads,
        CASE_MANIFESTS[58].d_vocab_out,
    ) == (3, 4, 11)
    assert (
        CASE_MANIFESTS[129].n_layers,
        CASE_MANIFESTS[129].n_heads,
        CASE_MANIFESTS[129].d_vocab_out,
    ) == (3, 4, 2)
    assert (
        CASE_MANIFESTS[25].n_layers,
        CASE_MANIFESTS[25].n_heads,
        CASE_MANIFESTS[25].d_vocab_out,
    ) == (2, 4, 56)

    for case_id in CASE_ORDER:
        manifest = CASE_MANIFESTS[case_id]
        assert len(manifest.internal_ground_truth_nodes) == 4
        assert len(manifest.internal_ground_truth_edges) == 3
        nodes, edges = case_universe(case_id)
        expected_counts = (15, 87) if manifest.n_layers == 3 else (10, 33)
        assert (len(nodes), len(edges)) == expected_counts


def test_scientific_and_smoke_requests_are_exact() -> None:
    expected = {
        58: ((758_100, 758_101), (758_200, 758_201), (758_000, 758_001)),
        129: ((829_100, 829_101), (829_200, 829_201), (829_000, 829_001)),
        25: ((725_100, 725_101), (725_200, 725_201), (725_000, 725_001)),
    }
    for case_id, (discovery_seeds, confirmation_seeds, smoke_seeds) in expected.items():
        discovery = split_request(case_id, "discovery")
        confirmation = split_request(case_id, "confirmation")
        smoke = SMOKE_REQUESTS[case_id]
        assert (discovery.required_count, discovery.candidate_count) == (64, 96)
        assert (confirmation.required_count, confirmation.candidate_count) == (
            128,
            192,
        )
        assert (discovery.clean_seed, discovery.corrupt_seed) == discovery_seeds
        assert (confirmation.clean_seed, confirmation.corrupt_seed) == (
            confirmation_seeds
        )
        assert (smoke.sample_count, smoke.clean_seed, smoke.corrupt_seed) == (
            96,
            *smoke_seeds,
        )

    with pytest.raises(ValueError, match="case_id"):
        split_request(124, "discovery")
    with pytest.raises(ValueError, match="split"):
        split_request(58, "faithfulness")


def test_response_blind_capacity_is_above_required_panel_size() -> None:
    assert EXPECTED_RESPONSE_BLIND_CAPACITY == {
        58: {"smoke_unique_rows": 192, "discovery": 96, "confirmation": 192},
        129: {"smoke_unique_rows": 192, "discovery": 96, "confirmation": 192},
        25: {"smoke_unique_rows": 190, "discovery": 91, "confirmation": 183},
    }
    for case_id in CASE_ORDER:
        assert EXPECTED_RESPONSE_BLIND_CAPACITY[case_id]["discovery"] >= 64
        assert EXPECTED_RESPONSE_BLIND_CAPACITY[case_id]["confirmation"] >= 128


def test_persisted_artifacts_and_exact_ground_truth_verify() -> None:
    for case_id in CASE_ORDER:
        assert (
            verify_case_artifacts(DEFAULT_MODEL_ROOT, case_id)
            == EXPECTED_HASHES[case_id]
        )
        truth = load_exact_ground_truth(DEFAULT_MODEL_ROOT, case_id)
        manifest = CASE_MANIFESTS[case_id]
        assert truth.raw_edge_count == manifest.raw_ground_truth_edges
        assert truth.internal_nodes == manifest.internal_ground_truth_nodes
        assert truth.internal_edges == manifest.internal_ground_truth_edges


def test_categorical_projection_checks_exact_geometry_and_centers_classes() -> None:
    logits = torch.arange(2 * 10 * 11, dtype=torch.float32).reshape(2, 10, 11)
    projected = project_categorical_output(58, logits)
    assert projected.shape == (2, 9, 11)
    assert torch.allclose(projected.mean(dim=-1), torch.zeros(2, 9))

    with pytest.raises(ValueError, match="geometry"):
        project_categorical_output(58, torch.zeros(2, 9, 11))
    with pytest.raises(ValueError, match="geometry"):
        project_categorical_output(58, torch.zeros(2, 10, 2))


@dataclass(frozen=True)
class _RawPair:
    clean_inputs: torch.Tensor
    clean_targets: torch.Tensor
    corrupt_inputs: torch.Tensor
    corrupt_targets: torch.Tensor


@dataclass(frozen=True)
class _Eligible:
    pair: _RawPair


def _raw_pair(start: int, count: int, classes: int) -> _RawPair:
    clean = torch.arange(start, start + count * 10).reshape(count, 10)
    corrupt = torch.arange(start + 100_000, start + 100_000 + count * 10).reshape(
        count, 10
    )
    targets = torch.zeros(count, 10, classes)
    return _RawPair(clean, targets, corrupt, targets.clone())


def test_smoke_exclusions_regenerate_both_fixed_roles() -> None:
    calls: list[tuple[int, int, int]] = []

    def generate(
        _case: object, *, sample_count: int, clean_seed: int, corrupt_seed: int
    ) -> _RawPair:
        calls.append((sample_count, clean_seed, corrupt_seed))
        return _raw_pair(clean_seed, sample_count, classes=11)

    rows = smoke_excluded_rows(58, object(), generate_pair=generate)

    assert calls == [(96, 758_000, 758_001)]
    assert len(rows) == 192
    assert len(set(rows)) == 192


def test_validate_case_panel_rejects_smoke_and_cross_split_leakage() -> None:
    smoke = tuple((index, index + 1) for index in range(192))
    discovery = _Eligible(_raw_pair(10_000, 64, classes=11))
    confirmation = _Eligible(_raw_pair(30_000, 128, classes=11))
    panel = CasePanel(58, discovery, confirmation, smoke)

    summary = validate_case_panel(panel)
    assert summary["discovery_unique_role_rows"] == 128
    assert summary["confirmation_unique_role_rows"] == 256
    assert summary["smoke_unique_rows"] == 192

    leaked = _Eligible(
        _RawPair(
            clean_inputs=confirmation.pair.clean_inputs.clone(),
            clean_targets=confirmation.pair.clean_targets,
            corrupt_inputs=confirmation.pair.corrupt_inputs.clone(),
            corrupt_targets=confirmation.pair.corrupt_targets,
        )
    )
    leaked.pair.clean_inputs[0] = discovery.pair.clean_inputs[0]
    with pytest.raises(RuntimeError, match="token-row leakage"):
        validate_case_panel(CasePanel(58, discovery, leaked, smoke))


def test_acquire_case_panel_passes_smoke_then_discovery_rows_as_exclusions() -> None:
    calls: list[tuple[int, int, int, int]] = []

    def generate(
        _case: object, *, sample_count: int, clean_seed: int, corrupt_seed: int
    ) -> _RawPair:
        return _raw_pair(clean_seed, sample_count, classes=2)

    def select(
        _model: object,
        _case: object,
        *,
        required_count: int,
        candidate_count: int,
        clean_seed: int,
        corrupt_seed: int,
        device: str,
        excluded_token_rows: tuple[tuple[int, ...], ...],
        require_unique_selected_token_rows: bool,
    ) -> _Eligible:
        del corrupt_seed, device
        assert require_unique_selected_token_rows is True
        calls.append(
            (
                required_count,
                candidate_count,
                clean_seed,
                len(excluded_token_rows),
            )
        )
        start = 1_000_000 if required_count == 64 else 2_000_000
        return _Eligible(_raw_pair(start, required_count, classes=2))

    panel = _acquire_case_panel(
        129,
        model=object(),
        case=object(),
        device="cpu",
        generate_pair=generate,
        select_eligible=select,
    )

    assert calls == [
        (64, 96, 829_100, 192),
        (128, 192, 829_200, 320),
    ]
    assert validate_case_panel(panel)["confirmation_unique_role_rows"] == 256
