# ruff: noqa: EXE002
"""Tests for the pure E31.2 mixed-resolution head scorer."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from experiments.e31_sparse_interactions.mixed_resolution_scoring import (
    DEFAULT_PRIMARY_GATES,
    PARENT_IDS,
    evaluate_primary_gate,
    fit_discovery_geometry,
    raw_second_difference,
    score_with_frozen_geometry,
    select_best_rectangle,
    stratified_cluster_bootstrap_indices,
)


def _fine_quartets(
    effects: torch.Tensor,
    *,
    context_count: int,
    cover_count: int = 4,
    left_marginals: torch.Tensor | None = None,
    right_marginals: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build fine quartets with exactly controlled marginal and pair effects."""
    if tuple(effects.shape) != (3, 12, 12):
        raise ValueError("effects must have shape (3, 12, 12)")
    left = (
        torch.zeros((3, 12), dtype=torch.float64)
        if left_marginals is None
        else left_marginals.double()
    )
    right = (
        torch.zeros((3, 12), dtype=torch.float64)
        if right_marginals is None
        else right_marginals.double()
    )
    context_scale = 1.0 + 0.01 * torch.arange(context_count, dtype=torch.float64)
    cover_scale = 1.0 + 0.005 * torch.arange(cover_count, dtype=torch.float64)
    scale = context_scale[:, None] * cover_scale[None, :]
    quartets = torch.zeros(
        (3, 12, 12, context_count, cover_count, 4), dtype=torch.float64
    )
    quartets[..., 1] = left[:, :, None, None, None] * scale[None, None]
    quartets[..., 2] = right[:, None, :, None, None] * scale[None, None]
    quartets[..., 3] = quartets[..., 1] + quartets[..., 2]
    quartets[..., 3] += effects[:, :, :, None, None] * scale[None, None]
    return quartets.reshape(3, 144, context_count, cover_count, 4)


def _direct_quartets(raw_effects: torch.Tensor) -> torch.Tensor:
    """Encode parent x context x cover raw effects as four-cell quartets."""
    if raw_effects.ndim != 3 or raw_effects.shape[0] != 3:
        raise ValueError("raw_effects must have shape (3, contexts, covers)")
    quartets = torch.zeros((*raw_effects.shape, 4), dtype=torch.float64)
    quartets[..., 1] = 0.2
    quartets[..., 2] = 0.3
    quartets[..., 3] = 0.5 + raw_effects
    return quartets


def _panel_metadata(context_count: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if context_count % 8:
        raise ValueError("synthetic context count must be divisible by eight")
    cluster_count = context_count // 2
    strata = tuple(
        f"template_{cluster // (cluster_count // 4)}"
        for cluster in range(cluster_count)
        for _ in range(2)
    )
    clusters = tuple(
        f"frame_{cluster}" for cluster in range(cluster_count) for _ in range(2)
    )
    return strata, clusters


def _synthetic_inputs() -> dict[str, object]:
    discovery_effects = torch.empty((3, 12, 12), dtype=torch.float64)
    selected = (
        ((0, 1, 2, 3), (0, 1, 2, 3)),
        ((2, 3, 4, 5), (4, 5, 6, 7)),
        ((6, 7, 8, 9), (8, 9, 10, 11)),
    )
    for parent in range(3):
        discovery_effects[parent] = 0.01 + 1e-5 * torch.arange(
            144, dtype=torch.float64
        ).reshape(12, 12)
        rows, columns = selected[parent]
        discovery_effects[parent][list(rows)][:, list(columns)] = 1.0
        # Chained advanced indexing above returns a copy; assign every cell directly.
        for row in rows:
            for column in columns:
                discovery_effects[parent, row, column] = 1.0 + 1e-4 * (
                    row * 12 + column
                )
    confirmation_effects = 0.9 * discovery_effects
    singleton = torch.linspace(0.1, 1.2, 12, dtype=torch.float64).repeat(3, 1)
    discovery = _fine_quartets(
        discovery_effects,
        context_count=16,
        left_marginals=singleton,
        right_marginals=singleton,
    )
    confirmation = _fine_quartets(
        confirmation_effects,
        context_count=32,
        left_marginals=singleton,
        right_marginals=singleton,
    )
    context = torch.arange(32, dtype=torch.float64)[None, :, None]
    cover = torch.arange(4, dtype=torch.float64)[None, None, :]
    parent = torch.arange(3, dtype=torch.float64)[:, None, None]
    coarse_effect = 0.25 + 0.04 * parent + 0.01 * context + 0.005 * cover
    discovery_strata, discovery_clusters = _panel_metadata(16)
    confirmation_strata, confirmation_clusters = _panel_metadata(32)
    return {
        "discovery": discovery,
        "confirmation": confirmation,
        "pair_composite": _direct_quartets(coarse_effect.clone()),
        "singleton_composite": _direct_quartets(0.1 * coarse_effect),
        "coarse_composite": _direct_quartets(coarse_effect),
        "discovery_strata": discovery_strata,
        "discovery_clusters": discovery_clusters,
        "confirmation_strata": confirmation_strata,
        "confirmation_clusters": confirmation_clusters,
        "expected_selected": selected,
    }


def _small_gates() -> dict[str, float | int]:
    return {**DEFAULT_PRIMARY_GATES, "matched_control_count": 128}


def test_raw_second_difference_preserves_signed_cells_before_squaring() -> None:
    quartets = torch.tensor(
        [[[[[1.0, 3.0, 4.0, 8.0]]], [[[1.0, 3.0, 4.0, 6.0]]]]],
        dtype=torch.float64,
    )
    observed = raw_second_difference(quartets)
    assert torch.equal(observed, torch.tensor([[[[2.0]], [[0.0]]]]).double())


def test_rectangle_search_is_global_and_lexicographically_tied() -> None:
    energy = torch.zeros((12, 12), dtype=torch.float64)
    rows = (1, 3, 5, 7)
    columns = (2, 4, 6, 8)
    for row in rows:
        for column in columns:
            energy[row, column] = 2.0
    selected = select_best_rectangle(energy)
    assert selected["left_heads"] == list(rows)
    assert selected["right_heads"] == list(columns)
    assert selected["edge_indices"] == [
        row * 12 + col for row in rows for col in columns
    ]

    tied = select_best_rectangle(torch.zeros((12, 12), dtype=torch.float64))
    assert tied["left_heads"] == [0, 1, 2, 3]
    assert tied["right_heads"] == [0, 1, 2, 3]


def test_discovery_geometry_freezes_pair_and_singleton_rectangles_and_controls() -> (
    None
):
    inputs = _synthetic_inputs()
    first = fit_discovery_geometry(
        discovery_quartets=inputs["discovery"],
        parent_ids=PARENT_IDS,
        primary_gates=_small_gates(),
        matched_control_count=128,
        enforce_registered_design=False,
    )
    second = fit_discovery_geometry(
        discovery_quartets=inputs["discovery"],
        parent_ids=PARENT_IDS,
        primary_gates=_small_gates(),
        matched_control_count=128,
        enforce_registered_design=False,
    )
    assert first == second
    assert first["confirmation_outcomes_used"] is False
    for parent_id, expected in zip(
        PARENT_IDS, inputs["expected_selected"], strict=True
    ):
        record = first["parents"][parent_id]
        assert record["selected_rectangle"]["left_heads"] == list(expected[0])
        assert record["selected_rectangle"]["right_heads"] == list(expected[1])
        assert record["singleton_rectangle"]["left_heads"] == [8, 9, 10, 11]
        assert record["singleton_rectangle"]["right_heads"] == [8, 9, 10, 11]
    controls = first["matched_controls"]
    assert controls["support_count"] == 128
    assert controls["selected_and_singleton_supports_excluded"] is True
    assert controls["uses_model_outcomes"] is False
    assert len(controls["supports"]) == 128
    selections = first["reconstruction_selections"]
    assert selections["schema_version"] == "e31_2_discovery_rectangle_selections_v1"
    assert [row["parent_pair_id"] for row in selections["parents"]] == list(PARENT_IDS)


def test_cluster_bootstrap_keeps_directions_together_within_templates() -> None:
    strata, clusters = _panel_metadata(32)
    indices = stratified_cluster_bootstrap_indices(
        context_strata=strata,
        context_cluster_ids=clusters,
        replicates=16,
        seed=31_200_002,
    )
    assert tuple(indices.shape) == (16, 32)
    for replicate in indices:
        counts = {cluster: 0 for cluster in set(clusters)}
        for index in replicate.tolist():
            counts[clusters[index]] += 1
        assert all(count % 2 == 0 for count in counts.values())
        for stratum in set(strata):
            observed = sum(
                counts[cluster]
                for cluster in counts
                if strata[clusters.index(cluster)] == stratum
            )
            assert observed == 8


def test_complete_score_passes_a_sparse_portable_head_atlas() -> None:
    inputs = _synthetic_inputs()
    geometry = fit_discovery_geometry(
        discovery_quartets=inputs["discovery"],
        parent_ids=PARENT_IDS,
        primary_gates=_small_gates(),
        matched_control_count=128,
        enforce_registered_design=False,
    )
    score = score_with_frozen_geometry(
        discovery_quartets=inputs["discovery"],
        confirmation_quartets=inputs["confirmation"],
        pair_selected_composite_quartets=inputs["pair_composite"],
        singleton_selected_composite_quartets=inputs["singleton_composite"],
        coarse_composite_quartets=inputs["coarse_composite"],
        discovery_context_strata=inputs["discovery_strata"],
        discovery_context_cluster_ids=inputs["discovery_clusters"],
        confirmation_context_strata=inputs["confirmation_strata"],
        confirmation_context_cluster_ids=inputs["confirmation_clusters"],
        frozen_geometry=geometry,
        bootstrap_replicates=64,
        bootstrap_seed=31_200_004,
        enforce_registered_design=False,
    )
    assert score["primary_verdict"] == "pass"
    assert score["primary"]["aggregate_confirmation_energy_coverage"] > 0.99
    assert score["primary"]["aggregate_fraction_of_oracle_rectangle_coverage"] > 0.99
    assert score["primary"]["matched_controls"]["passed"] is True
    assert all(
        record["full_map_energy_spearman"] > 0.999
        and record["selected_sign_replication_count"] == 16
        for record in score["parents"].values()
    )
    reconstruction = score["reconstruction"]
    assert reconstruction["pair_selected_composite"]["pooled_r2"] == pytest.approx(1.0)
    assert reconstruction["pair_selected_composite"]["pooled_pearson"] == pytest.approx(
        1.0
    )
    assert reconstruction["pair_selected_mse_beats_singleton_parent_count"] == 3
    assert score["multiplicity"]["aggregate_primary_family_count"] == 1
    assert score["multiplicity"]["all_three_parent_conjunctions_required"] is True
    assert score["multiplicity"]["posthoc_e31_2_parent_adaptation_allowed"] is False
    assert score["multiplicity"]["parent_selection_was_posthoc_e31_1"] is True
    assert "posthoc_parent_selection_allowed" not in score["multiplicity"]
    assert score["bootstrap"]["resample_unit"] == "semantic_frame_with_both_directions"


def test_support_switch_fails_without_changing_discovery_geometry() -> None:
    inputs = _synthetic_inputs()
    geometry = fit_discovery_geometry(
        discovery_quartets=inputs["discovery"],
        parent_ids=PARENT_IDS,
        primary_gates=_small_gates(),
        matched_control_count=128,
        enforce_registered_design=False,
    )
    confirmation_effects = torch.full((3, 12, 12), 0.001, dtype=torch.float64)
    for parent in range(3):
        for left in range(8, 12):
            for right in range(8, 12):
                confirmation_effects[parent, left, right] = 1.0
    unstable_confirmation = _fine_quartets(confirmation_effects, context_count=32)
    score = score_with_frozen_geometry(
        discovery_quartets=inputs["discovery"],
        confirmation_quartets=unstable_confirmation,
        pair_selected_composite_quartets=inputs["pair_composite"],
        singleton_selected_composite_quartets=inputs["singleton_composite"],
        coarse_composite_quartets=inputs["coarse_composite"],
        discovery_context_strata=inputs["discovery_strata"],
        discovery_context_cluster_ids=inputs["discovery_clusters"],
        confirmation_context_strata=inputs["confirmation_strata"],
        confirmation_context_cluster_ids=inputs["confirmation_clusters"],
        frozen_geometry=geometry,
        bootstrap_replicates=16,
        bootstrap_seed=31_200_006,
        enforce_registered_design=False,
    )
    assert score["primary_verdict"] == "fail"
    assert score["primary"]["aggregate_confirmation_energy_coverage"] < 0.20
    assert (
        score["primary_gate"]["checks"]["aggregate_confirmation_energy_coverage"]
        is False
    )


def test_reconstruction_uses_direct_values_without_affine_calibration() -> None:
    inputs = _synthetic_inputs()
    geometry = fit_discovery_geometry(
        discovery_quartets=inputs["discovery"],
        parent_ids=PARENT_IDS,
        primary_gates=_small_gates(),
        matched_control_count=128,
        enforce_registered_design=False,
    )
    doubled = inputs["pair_composite"].clone()
    doubled_effect = 2.0 * raw_second_difference(doubled)
    doubled = _direct_quartets(doubled_effect)
    score = score_with_frozen_geometry(
        discovery_quartets=inputs["discovery"],
        confirmation_quartets=inputs["confirmation"],
        pair_selected_composite_quartets=doubled,
        singleton_selected_composite_quartets=inputs["singleton_composite"],
        coarse_composite_quartets=inputs["coarse_composite"],
        discovery_context_strata=inputs["discovery_strata"],
        discovery_context_cluster_ids=inputs["discovery_clusters"],
        confirmation_context_strata=inputs["confirmation_strata"],
        confirmation_context_cluster_ids=inputs["confirmation_clusters"],
        frozen_geometry=geometry,
        bootstrap_replicates=16,
        bootstrap_seed=31_200_008,
        enforce_registered_design=False,
    )
    reconstruction = score["reconstruction"]["pair_selected_composite"]
    assert reconstruction["pooled_pearson"] == pytest.approx(1.0)
    assert reconstruction["pooled_r2"] < 0.5
    assert score["primary_gate"]["checks"]["reconstruction_r2"] is False


def test_each_parent_sign_and_spearman_check_is_conjunctive() -> None:
    parents = {
        parent_id: {
            "full_map_energy_spearman": 0.9,
            "selected_sign_replication_count": 16,
            "selected_sign_replication_denominator": 16,
        }
        for parent_id in PARENT_IDS
    }
    primary = {
        "aggregate_confirmation_energy_coverage": 0.8,
        "aggregate_fraction_of_oracle_rectangle_coverage": 0.9,
        "matched_controls": {"passed": True, "support_count": 10_000},
    }
    reconstruction = {
        "pair_selected_composite": {"pooled_r2": 0.8, "pooled_pearson": 0.9},
        "pair_selected_mse_beats_singleton_parent_count": 3,
    }
    passing = evaluate_primary_gate(
        primary=primary,
        parents=parents,
        reconstruction=reconstruction,
        bootstrap={"aggregate_coverage_one_sided_95_lower": 0.8},
        gates=DEFAULT_PRIMARY_GATES,
    )
    assert passing["passed"] is True
    broken = deepcopy(parents)
    broken[PARENT_IDS[-1]]["selected_sign_replication_count"] = 11
    assert (
        evaluate_primary_gate(
            primary=primary,
            parents=broken,
            reconstruction=reconstruction,
            bootstrap={"aggregate_coverage_one_sided_95_lower": 0.8},
            gates=DEFAULT_PRIMARY_GATES,
        )["passed"]
        is False
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("nonfinite", "finite"),
        ("wrong_edges", "144"),
        ("wrong_parents", "three"),
        ("tampered_geometry", "differs from recomputation"),
    ],
)
def test_scorer_rejects_malformed_or_adapted_inputs(
    mutation: str, message: str
) -> None:
    inputs = _synthetic_inputs()
    discovery = inputs["discovery"].clone()
    parent_ids = PARENT_IDS
    if mutation == "nonfinite":
        discovery[0, 0, 0, 0, 0] = torch.nan
    elif mutation == "wrong_edges":
        discovery = discovery[:, :-1]
    elif mutation == "wrong_parents":
        parent_ids = PARENT_IDS[:2]
    if mutation != "tampered_geometry":
        with pytest.raises((TypeError, ValueError), match=message):
            fit_discovery_geometry(
                discovery_quartets=discovery,
                parent_ids=parent_ids,
                primary_gates=_small_gates(),
                matched_control_count=8,
                enforce_registered_design=False,
            )
        return
    geometry = fit_discovery_geometry(
        discovery_quartets=discovery,
        parent_ids=PARENT_IDS,
        primary_gates=_small_gates(),
        matched_control_count=128,
        enforce_registered_design=False,
    )
    geometry["parents"][PARENT_IDS[0]]["selected_rectangle"]["left_heads"] = [
        0,
        1,
        2,
        4,
    ]
    with pytest.raises(RuntimeError, match=message):
        score_with_frozen_geometry(
            discovery_quartets=discovery,
            confirmation_quartets=inputs["confirmation"],
            pair_selected_composite_quartets=inputs["pair_composite"],
            singleton_selected_composite_quartets=inputs["singleton_composite"],
            coarse_composite_quartets=inputs["coarse_composite"],
            discovery_context_strata=inputs["discovery_strata"],
            discovery_context_cluster_ids=inputs["discovery_clusters"],
            confirmation_context_strata=inputs["confirmation_strata"],
            confirmation_context_cluster_ids=inputs["confirmation_clusters"],
            frozen_geometry=geometry,
            bootstrap_replicates=8,
            enforce_registered_design=False,
        )
