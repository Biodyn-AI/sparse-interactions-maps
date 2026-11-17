# ruff: noqa: EXE002
"""Tests for the pure prospective E31.1 interaction scorer."""

from __future__ import annotations

import math

import pytest
import torch

from experiments.e31_sparse_interactions.prospective_scoring import (
    DEFAULT_CLAIM_BOUNDARIES,
    DEFAULT_PRIMARY_GATES,
    arcsinh_forward,
    arcsinh_joint_prediction,
    assemble_regime_panels,
    build_matched_control_supports,
    crossfit_arcsinh_residuals,
    fit_discovery_geometry,
    raw_second_difference,
    score_prospective_atlas,
    score_regime,
    score_with_frozen_geometry,
    summarize_energy,
)


def _pairs(count: int) -> tuple[tuple[int, int], ...]:
    """Return a deterministic list of unique, lexicographically ordered pairs."""
    pairs = []
    for right in range(1, 24):
        for left in range(right):
            pairs.append((left, right))
            if len(pairs) == count:
                return tuple(pairs)
    raise ValueError("requested too many pairs")


def _effect_quartets(
    effects: torch.Tensor,
    context_count: int,
    *,
    context_scale: float = 0.01,
    cover_count: int | None = None,
) -> torch.Tensor:
    """Build exact quartets whose raw and identity-residual effects are known."""
    edge_count = len(effects)
    quartets = torch.zeros((edge_count, context_count, 4), dtype=torch.float64)
    context_offsets = context_scale * torch.arange(context_count, dtype=torch.float64)
    quartets[:, :, 1] = 0.2 + context_offsets
    quartets[:, :, 2] = 0.3 + context_offsets
    quartets[:, :, 3] = 0.5 + 2.0 * context_offsets + effects[:, None]
    if cover_count is None:
        return quartets
    return quartets[:, :, None, :].expand(-1, -1, cover_count, -1).clone()


def test_raw_second_difference_uses_registered_cell_order() -> None:
    quartets = torch.tensor(
        [
            [[1.0, 3.0, 4.0, 6.0]],
            [[1.0, 3.0, 4.0, 7.0]],
        ],
        dtype=torch.float64,
    )
    observed = raw_second_difference(quartets)
    assert torch.equal(observed, torch.tensor([[0.0], [1.0]]).double())


def test_registered_gate_names_signed_interaction_not_signed_energy() -> None:
    assert (
        "signed_interaction_pearson_one_sided_95_lower_bound_strict_minimum"
        in DEFAULT_PRIMARY_GATES
    )
    assert (
        "signed_energy_correlation_one_sided_95_lower_bound_strict_minimum"
        not in DEFAULT_PRIMARY_GATES
    )


def test_arcsinh_null_exactly_removes_link_curvature() -> None:
    curvature = 2.0
    latent = torch.tensor(
        [
            [[0.10, 0.40, 0.50, 0.80], [0.20, 0.70, 0.35, 0.85]],
            [[-0.10, 0.25, 0.45, 0.80], [0.30, 0.55, 0.60, 0.85]],
        ],
        dtype=torch.float64,
    )
    observed = arcsinh_forward(latent, curvature)
    prediction = arcsinh_joint_prediction(observed, curvature)
    assert torch.allclose(prediction, observed[..., 3], atol=1e-12, rtol=0.0)
    assert raw_second_difference(observed).abs().max() > 1e-3


def test_crossfit_selects_the_true_arcsinh_curvature_and_refits() -> None:
    latent_base = torch.linspace(-0.2, 0.4, 12, dtype=torch.float64).reshape(3, 4)
    latent_left = latent_base + 0.35
    latent_right = latent_base + 0.55
    latent_joint = latent_left + latent_right - latent_base
    latent = torch.stack((latent_base, latent_left, latent_right, latent_joint), dim=-1)
    quartets = arcsinh_forward(latent, 2.0)
    fit = crossfit_arcsinh_residuals(
        quartets,
        context_strata=("a", "b", "c", "d"),
        curvature_grid=(0.0, 0.5, 1.0, 2.0, 4.0),
    )
    assert fit["refit_curvature"] == 2.0
    assert set(fit["fold_curvatures"]) == {2.0}
    assert fit["cross_fitted_residuals"].abs().max() < 1e-12


def test_energy_summary_uses_frozen_discovery_order() -> None:
    discovery = torch.tensor([3.0, 2.0, 1.0, 0.0]).double()
    confirmation = torch.tensor([9.0, 4.0, 1.0, 0.0]).double()
    summary = summarize_energy(
        discovery,
        confirmation,
        pairs=((0, 1), (0, 2), (0, 3), (1, 2)),
        top_counts=(1, 2, 4),
    )
    assert summary["discovery_order_indices"] == [0, 1, 2, 3]
    assert summary["coverage_by_top_k"]["1"] == pytest.approx(9.0 / 14.0)
    assert summary["coverage_by_top_k"]["2"] == pytest.approx(13.0 / 14.0)
    assert summary["recovered_k90"] == 2
    assert summary["effective_support"] == pytest.approx(2.0)
    assert summary["oracle_efficiency_by_top_k"]["1"] == pytest.approx(1.0)


def test_assemble_regime_panels_follows_condition_references_without_averaging() -> (
    None
):
    pairs = ((0, 2), (0, 3))
    conditions = []
    contrasts = []
    condition_count = 0
    for pair_index, pair in enumerate(pairs):
        for cover in range(4):
            condition_ids = []
            for _ in range(4):
                condition_ids.append(f"condition_{condition_count}")
                conditions.append(
                    {
                        "condition_index": condition_count,
                        "condition_id": condition_ids[-1],
                    }
                )
                condition_count += 1
            contrasts.append(
                {
                    "contrast_index": len(contrasts),
                    "contrast_id": f"contrast_{pair_index}_{cover}",
                    "edge_class": "nonhub_pair",
                    "factor_components": list(pair),
                    "k_nonhub_on": 10,
                    "hub_state": "off",
                    "cover_rank": cover,
                    "background_id": f"background_{pair_index}_{cover}",
                    "ordered_cell_bits": ["00", "10", "01", "11"],
                    "condition_ids": condition_ids,
                }
            )
    responses = torch.arange(condition_count * 4, dtype=torch.float64).reshape(
        condition_count, 4
    )
    prompt_records = (
        {"stratum_id": "t0", "pair_id": "frame_0_a_to_b"},
        {"stratum_id": "t0", "pair_id": "frame_0_b_to_a"},
        {"stratum_id": "t1", "pair_id": "frame_1_a_to_b"},
        {"stratum_id": "t1", "pair_id": "frame_1_b_to_a"},
    )
    panels = assemble_regime_panels(
        responses=responses,
        split_manifest={"conditions": conditions, "contrasts": contrasts},
        prompt_records=prompt_records,
    )
    panel = panels["nonhub_k10_hub_off"]
    assert panel["pairs"] == pairs
    assert panel["quartets"].shape == (2, 4, 4, 4)
    assert torch.equal(panel["quartets"][0, :, 0, :], responses[:4].T)
    assert panel["context_strata"] == ("t0", "t0", "t1", "t1")
    assert panel["context_cluster_ids"] == (
        "frame_0",
        "frame_0",
        "frame_1",
        "frame_1",
    )


def test_assemble_spoke_uses_sorted_edge_without_reordering_factor_cells() -> None:
    conditions = [
        {"condition_index": index, "condition_id": f"c{index}"} for index in range(16)
    ]
    contrasts = [
        {
            "contrast_index": cover,
            "contrast_id": f"spoke_{cover}",
            "edge_class": "hub_spoke",
            "edge": [0, 1],
            "factor_components": [1, 0],
            "k_nonhub_on": 2,
            "hub_state": "crossing",
            "cover_rank": cover,
            "background_id": f"b{cover}",
            "ordered_cell_bits": ["00", "10", "01", "11"],
            "condition_ids": [f"c{4 * cover + cell}" for cell in range(4)],
        }
        for cover in range(4)
    ]
    responses = torch.arange(16 * 4, dtype=torch.float64).reshape(16, 4)
    prompts = tuple(
        {
            "stratum_id": f"t{index // 2}",
            "pair_id": f"frame_{index // 2}_{'a_to_b' if index % 2 == 0 else 'b_to_a'}",
        }
        for index in range(4)
    )
    panel = assemble_regime_panels(
        responses=responses,
        split_manifest={"conditions": conditions, "contrasts": contrasts},
        prompt_records=prompts,
    )["spokes_k02"]
    assert panel["pairs"] == ((0, 1),)
    assert torch.equal(panel["quartets"][0, :, 0], responses[:4].T)


def test_score_regime_reports_all_primary_statistics_reproducibly() -> None:
    edge_count = 60
    effects = torch.cat((torch.linspace(1.0, 0.5, 24), torch.zeros(36))).double()
    discovery = _effect_quartets(effects, 8, cover_count=4)
    confirmation = _effect_quartets(0.95 * effects, 16, cover_count=4)
    pairs = _pairs(edge_count)
    control_supports = {
        "random_hash": (tuple(range(36, 60)),),
        "type_depth_matched": (tuple(range(24, 48)),),
    }
    arguments = {
        "regime_id": "nonhub_k10_hub_off",
        "discovery_quartets": discovery,
        "confirmation_quartets": confirmation,
        "pairs": pairs,
        "discovery_context_strata": tuple(f"s{i % 4}" for i in range(8)),
        "confirmation_context_strata": tuple(f"s{i % 4}" for i in range(16)),
        "curvature_grid": (0.0, 1.0),
        "bootstrap_replicates": 128,
        "bootstrap_seed": 31_001,
        "control_supports": control_supports,
        "primary_gates": {
            **DEFAULT_PRIMARY_GATES,
            "matched_random_support_count": 1,
        },
    }
    first = score_regime(**arguments)
    second = score_regime(**arguments)
    assert first == second
    assert first["raw_interactions_shape"] == [60, 16, 4]
    assert first["saturation"]["refit_curvature"] == 0.0
    assert first["confirmation"]["coverage_by_top_k"]["24"] > 0.99
    assert first["confirmation"]["coverage_by_top_k"]["48"] > 0.99
    assert first["stability"]["sign_replication_count_top24"] == 24
    assert first["stability"]["signed_interaction_pearson"] > 0.999
    assert first["raw_residual_overlap"]["raw_top48_contains_residual_top24"] == 24
    assert first["controls"]["beats_every_control_top24_coverage"] is True
    assert first["bootstrap"]["replicates"] == 128
    assert first["bootstrap"]["coverage_top24_one_sided_95_lower"] > 0.99
    assert first["bootstrap"]["signed_interaction_pearson_one_sided_95_lower"] > 0.99
    assert first["primary_gate"]["passed"] is True


def test_matched_controls_are_deterministic_unique_and_exact_size() -> None:
    nonhub = tuple(component for component in range(24) if component != 1)
    pairs = tuple(
        (left, right)
        for right_index, right in enumerate(nonhub)
        for left in nonhub[:right_index]
    )
    quartets = _effect_quartets(torch.linspace(0.1, 1.0, len(pairs)), 8, cover_count=4)
    first = build_matched_control_supports(
        pairs=pairs,
        discovery_quartets=quartets,
        selected_indices=tuple(range(24)),
        support_count=64,
    )
    second = build_matched_control_supports(
        pairs=pairs,
        discovery_quartets=quartets,
        selected_indices=tuple(range(24)),
        support_count=64,
    )
    assert first == second
    assert len(first["supports"]) == 64
    assert all(len(support) == len(set(support)) == 24 for support in first["supports"])
    assert len(set(first["supports"])) == 64
    assert tuple(range(24)) not in first["supports"]
    assert first["selected_support_excluded"] is True
    assert first["all_supports_match_joint_bin_counts"] is True


def test_score_regime_fails_an_unstable_discovery_ranking() -> None:
    edge_count = 60
    discovery_effects = torch.cat((torch.ones(24), torch.zeros(36))).double()
    confirmation_effects = torch.cat(
        (torch.zeros(24), torch.zeros(12), torch.ones(24))
    ).double()
    scored = score_regime(
        regime_id="unstable",
        discovery_quartets=_effect_quartets(discovery_effects, 8),
        confirmation_quartets=_effect_quartets(confirmation_effects, 16),
        pairs=_pairs(edge_count),
        discovery_context_strata=tuple(f"s{i % 4}" for i in range(8)),
        confirmation_context_strata=tuple(f"s{i % 4}" for i in range(16)),
        curvature_grid=(0.0,),
        bootstrap_replicates=64,
        bootstrap_seed=31_002,
    )
    assert scored["confirmation"]["coverage_by_top_k"]["24"] < 1e-20
    assert scored["stability"]["sign_replication_count_top24"] == 0
    assert scored["primary_gate"]["passed"] is False


def test_score_prospective_atlas_keeps_complete_regimes_and_claim_boundaries() -> None:
    effects = torch.cat((torch.ones(24), torch.zeros(6))).double()
    panel = {
        "pairs": _pairs(30),
        "quartets": _effect_quartets(effects, 8),
        "context_strata": tuple(f"s{(i // 2) % 4}" for i in range(8)),
    }
    confirmation = {
        **panel,
        "quartets": _effect_quartets(0.9 * effects, 16),
        "context_strata": tuple(f"s{(i // 2) % 4}" for i in range(16)),
    }
    generic_gates = {
        **DEFAULT_PRIMARY_GATES,
        "matched_random_support_count": 1,
    }
    result = score_prospective_atlas(
        discovery_panels={"primary": panel, "spokes_k10": panel},
        confirmation_panels={"primary": confirmation, "spokes_k10": confirmation},
        primary_regime_id="primary",
        curvature_grid=(0.0,),
        bootstrap_replicates=64,
        bootstrap_seed=31_003,
        primary_gates=generic_gates,
    )
    assert list(result["regimes"]) == ["primary", "spokes_k10"]
    assert result["primary_regime_id"] == "primary"
    assert result["primary_verdict"] in {"pass", "fail"}
    assert result["claim_boundaries"] == DEFAULT_CLAIM_BOUNDARIES
    assert result["primary_gates"] == generic_gates
    assert result["predictive_or_topology_claim_authorized"] is False
    assert result["hierarchical_scale_gate"]["evaluated"] is False
    assert (
        result["hierarchical_scale_gate"]["status"] == "not_available_in_generic_panel"
    )


def test_two_stage_geometry_is_frozen_and_uses_common_registered_seed() -> None:
    effects = torch.cat((torch.ones(24), torch.zeros(6))).double()
    discovery_panel = {
        "pairs": _pairs(30),
        "quartets": _effect_quartets(effects, 8, cover_count=4),
        "context_strata": tuple(f"s{(i // 2) % 4}" for i in range(8)),
        "context_cluster_ids": tuple(i // 2 for i in range(8)),
    }
    confirmation_panel = {
        **discovery_panel,
        "quartets": _effect_quartets(0.9 * effects, 16, cover_count=4),
        "context_strata": tuple(f"s{(i // 2) % 4}" for i in range(16)),
        "context_cluster_ids": tuple(i // 2 for i in range(16)),
    }
    discovery = {
        "primary": discovery_panel,
        "nonhub_k02_hub_off": discovery_panel,
    }
    confirmation = {
        "primary": confirmation_panel,
        "nonhub_k02_hub_off": confirmation_panel,
    }
    gates = {**DEFAULT_PRIMARY_GATES, "matched_random_support_count": 1}
    geometry = fit_discovery_geometry(
        discovery_panels=discovery,
        primary_regime_id="primary",
        curvature_grid=(0.0,),
        primary_gates=gates,
        enforce_registered_design=False,
    )
    assert geometry["confirmation_outcomes_used"] is False
    assert (
        geometry["regimes"]["primary"]["matched_controls"]["selected_support_excluded"]
        is True
    )
    scored = score_with_frozen_geometry(
        discovery_panels=discovery,
        confirmation_panels=confirmation,
        frozen_geometry=geometry,
        bootstrap_replicates=64,
        bootstrap_seed=31_005,
        enforce_registered_design=False,
    )
    assert scored["regimes"]["primary"]["bootstrap"]["seed"] == 31_005
    assert (
        scored["regimes"]["primary"]["bootstrap"]["resample_unit"]
        == "semantic_frame_with_both_directions"
    )
    assert scored["regimes"]["nonhub_k02_hub_off"]["bootstrap"]["seed"] == 31_005
    corrupted = {**geometry, "curvature_grid": [0.0, 1.0]}
    with pytest.raises(RuntimeError, match="differs from recomputation"):
        score_with_frozen_geometry(
            discovery_panels=discovery,
            confirmation_panels=confirmation,
            frozen_geometry=corrupted,
            bootstrap_replicates=8,
            bootstrap_seed=31_005,
            enforce_registered_design=False,
        )


def test_hierarchical_bootstrap_is_skipped_when_primary_fails() -> None:
    discovery_effects = torch.cat((torch.ones(24), torch.zeros(6))).double()
    failing_effects = torch.cat((torch.zeros(24), torch.ones(6))).double()
    discovery_panel = {
        "pairs": _pairs(30),
        "quartets": _effect_quartets(discovery_effects, 8, cover_count=4),
        "context_strata": tuple(f"s{(i // 2) % 4}" for i in range(8)),
        "context_cluster_ids": tuple(i // 2 for i in range(8)),
    }
    failing_confirmation = {
        **discovery_panel,
        "quartets": _effect_quartets(failing_effects, 16, cover_count=4),
        "context_strata": tuple(f"s{(i // 2) % 4}" for i in range(16)),
        "context_cluster_ids": tuple(i // 2 for i in range(16)),
    }
    discovery = {
        "primary": discovery_panel,
        "nonhub_k02_hub_off": discovery_panel,
    }
    confirmation = {
        "primary": failing_confirmation,
        "nonhub_k02_hub_off": failing_confirmation,
    }
    gates = {**DEFAULT_PRIMARY_GATES, "matched_random_support_count": 1}
    geometry = fit_discovery_geometry(
        discovery_panels=discovery,
        primary_regime_id="primary",
        curvature_grid=(0.0,),
        primary_gates=gates,
        enforce_registered_design=False,
    )
    scored = score_with_frozen_geometry(
        discovery_panels=discovery,
        confirmation_panels=confirmation,
        frozen_geometry=geometry,
        bootstrap_replicates=16,
        bootstrap_seed=31_006,
        enforce_registered_design=False,
    )
    hierarchy = scored["regimes"]["nonhub_k02_hub_off"]
    assert scored["primary_verdict"] == "fail"
    assert scored["hierarchical_scale_gate"]["evaluated"] is False
    assert hierarchy["bootstrap"] == {
        "computed": False,
        "reason": "not_tested_due_to_primary_failure",
    }
    assert hierarchy["inferential_status"] == (
        "descriptive_only_due_to_primary_failure"
    )


def test_official_geometry_entrypoint_rejects_nonregistered_panel() -> None:
    panel = {
        "pairs": _pairs(4),
        "quartets": _effect_quartets(torch.ones(4), 4, cover_count=4),
        "context_strata": ("a", "a", "b", "b"),
        "context_cluster_ids": (0, 0, 1, 1),
    }
    with pytest.raises(ValueError, match="regime order"):
        fit_discovery_geometry(
            discovery_panels={"nonhub_k10_hub_off": panel},
            primary_regime_id="nonhub_k10_hub_off",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("nonfinite", "finite"),
        ("duplicate_pair", "unique"),
        ("pair_mismatch", "same ordered pairs"),
        ("strata_mismatch", "context_strata"),
    ],
)
def test_score_regime_rejects_malformed_inputs(mutation: str, message: str) -> None:
    pairs = list(_pairs(4))
    discovery = _effect_quartets(torch.ones(4), 4)
    confirmation = _effect_quartets(torch.ones(4), 8)
    discovery_strata = ("a", "b", "a", "b")
    confirmation_strata = ("a", "b", "a", "b", "a", "b", "a", "b")
    confirmation_pairs = list(pairs)
    if mutation == "nonfinite":
        confirmation[0, 0, 0] = math.nan
    elif mutation == "duplicate_pair":
        pairs[1] = pairs[0]
        confirmation_pairs[1] = confirmation_pairs[0]
    elif mutation == "pair_mismatch":
        confirmation_pairs[0] = (20, 23)
    else:
        discovery_strata = ("a",)
    with pytest.raises((TypeError, ValueError), match=message):
        score_regime(
            regime_id="malformed",
            discovery_quartets=discovery,
            confirmation_quartets=confirmation,
            pairs=tuple(pairs),
            confirmation_pairs=tuple(confirmation_pairs),
            discovery_context_strata=discovery_strata,
            confirmation_context_strata=confirmation_strata,
            curvature_grid=(0.0,),
            bootstrap_replicates=8,
            bootstrap_seed=31_004,
        )
