# ruff: noqa: EXE002
"""Focused tests for the post-hoc E31.2 node/residual diagnostic."""

from __future__ import annotations

import torch

from experiments.e31_sparse_interactions.e31_2_posthoc_node_residuals import (
    HEAD_COUNT,
    analyze_node_residual_transfer,
    compute_split_maps,
    fit_origin_scale,
    rank1_signed_svd_map,
    rank1_svd_map,
    row_column_outer,
    support_metrics,
)


def _quartets(
    left_effect: torch.Tensor,
    right_effect: torch.Tensor,
    interaction: torch.Tensor,
    *,
    context_count: int = 2,
    cover_count: int = 4,
) -> torch.Tensor:
    """Build exact 00/10/01/11 cells for three identical parents."""
    cells = torch.zeros(
        3,
        HEAD_COUNT,
        HEAD_COUNT,
        context_count,
        cover_count,
        4,
        dtype=torch.float64,
    )
    cells[..., 1] = left_effect[:, None, None, None]
    cells[..., 2] = right_effect[None, :, None, None]
    cells[..., 3] = (
        left_effect[:, None, None, None]
        + right_effect[None, :, None, None]
        + interaction[:, :, None, None]
    )
    return cells.reshape(3, HEAD_COUNT * HEAD_COUNT, context_count, cover_count, 4)


def test_compute_split_maps_recovers_energy_and_singleton_effect_energy() -> None:
    left = torch.arange(1, HEAD_COUNT + 1, dtype=torch.float64)
    right = torch.arange(2, HEAD_COUNT + 2, dtype=torch.float64)
    interaction = left[:, None] * right[None, :]

    maps = compute_split_maps(_quartets(left, right, interaction))

    assert torch.equal(maps["signed_mean"][0], interaction)
    assert torch.equal(maps["interaction_energy"][0], interaction.square())
    assert torch.equal(maps["left_singleton_energy"][0], left.square())
    assert torch.equal(maps["right_singleton_energy"][0], right.square())
    assert torch.equal(maps["left_singleton_signed_mean"][0], left)
    assert torch.equal(maps["right_singleton_signed_mean"][0], right)


def test_three_separable_predictors_exactly_recover_a_rank_one_energy_map() -> None:
    left = torch.arange(1, HEAD_COUNT + 1, dtype=torch.float64).square()
    right = torch.arange(2, HEAD_COUNT + 2, dtype=torch.float64).square()
    target = 2.5 * torch.outer(left, right)

    alpha = fit_origin_scale(torch.outer(left, right), target)

    assert alpha == 2.5
    assert torch.allclose(alpha * torch.outer(left, right), target)
    assert torch.allclose(row_column_outer(target), target)
    assert torch.allclose(rank1_svd_map(target), target, atol=1e-9, rtol=1e-9)


def test_signed_svd_recovers_a_mixed_sign_rank_one_map() -> None:
    left = torch.linspace(-2.0, 3.0, HEAD_COUNT, dtype=torch.float64)
    right = torch.linspace(4.0, -1.0, HEAD_COUNT, dtype=torch.float64)
    target = torch.outer(left, right)

    assert torch.allclose(rank1_signed_svd_map(target), target, atol=1e-9, rtol=1e-9)


def test_support_metrics_use_frozen_ranking_and_confirmation_energy() -> None:
    score = torch.zeros(HEAD_COUNT, HEAD_COUNT, dtype=torch.float64)
    target = torch.ones_like(score)
    score[0, 0] = 3.0
    score[1, 1] = 2.0
    target[0, 0] = 10.0
    target[1, 1] = 5.0

    metrics = support_metrics(score, target, k=2)

    assert metrics["selected_indices"] == [0, HEAD_COUNT + 1]
    assert metrics["support_overlap_count"] == 2
    assert metrics["confirmation_energy_coverage"] == 15.0 / 157.0
    assert metrics["fraction_of_confirmation_oracle_coverage"] == 1.0


def test_end_to_end_marks_posthoc_and_recovers_identical_split_supports() -> None:
    left = torch.arange(1, HEAD_COUNT + 1, dtype=torch.float64)
    right = torch.arange(2, HEAD_COUNT + 2, dtype=torch.float64)
    interaction = left[:, None] * right[None, :]
    labels = {"normalized_fine_quartets": _quartets(left, right, interaction)}

    report = analyze_node_residual_transfer(labels, labels, top_ks=(1, 4))

    assert report["status"] == "posthoc_development_diagnostic_only"
    assert report["claim_boundaries"]["changes_e31_2_verdict"] is False
    for parent in report["parents"].values():
        assert parent["predictors"]["raw_pair_map"]["confirmation"]["pearson"] == 1.0
        assert (
            parent["predictors"]["raw_pair_map"]["supports"]["4"][
                "support_overlap_count"
            ]
            == 4
        )
        signed = parent["signed_map_diagnostic"]
        assert signed["endpoint_singleton_outer"]["confirmation"]["pearson"] == 1.0
        assert signed["raw_pair_signed_map"]["confirmation"]["pearson"] == 1.0
        assert signed["rank1_svd_signed_map"]["confirmation"]["pearson"] == 1.0
        assert (
            signed["residual_after_endpoint_outer"]["supports"]["4"][
                "support_overlap_count"
            ]
            == 4
        )
        assert (
            signed["residual_after_split_local_rank1_svd"]["supports"]["4"][
                "support_overlap_count"
            ]
            == 4
        )


def test_full_map_predictors_are_distinguished_from_endpoint_only_predictor() -> None:
    left = torch.arange(1, HEAD_COUNT + 1, dtype=torch.float64)
    right = torch.arange(2, HEAD_COUNT + 2, dtype=torch.float64)
    labels = {
        "normalized_fine_quartets": _quartets(
            left, right, left[:, None] * right[None, :]
        )
    }

    report = analyze_node_residual_transfer(labels, labels, top_ks=(1,))
    predictors = next(iter(report["parents"].values()))["predictors"]

    assert predictors["singleton_effect_outer"]["endpoint_only"] is True
    assert (
        predictors["singleton_effect_outer"]["support_ranking_requires_full_pair_map"]
        is False
    )
    assert predictors["singleton_effect_outer"]["reported_alpha_uses_pair_map"] is True
    for name in ("row_column_outer", "rank1_svd", "raw_pair_map"):
        assert predictors[name]["endpoint_only"] is False
        assert predictors[name]["support_ranking_requires_full_pair_map"] is True


def test_invalid_quartet_shape_is_rejected() -> None:
    invalid = torch.zeros(3, 144, 2, 4, 3, dtype=torch.float64)
    try:
        compute_split_maps(invalid)
    except ValueError as error:
        assert "ordered cells" in str(error)
    else:  # pragma: no cover - documents the required rejection
        raise AssertionError("invalid quartet shape was accepted")
