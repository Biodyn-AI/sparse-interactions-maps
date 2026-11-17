# ruff: noqa: EXE002
"""Pure score and gate tests for E32.2 directed validation."""

from __future__ import annotations

import pytest
import torch

from experiments.e32_interpbench.e32_2_scoring import (
    best_eap_variant,
    bidirectional_fine_replacement_scores,
    direct_vs_transitive_auroc,
    embed_shortlist_scores,
    evaluate_case_gates,
    promote_fine_scores_max,
    spearman_correlation,
)


def test_per_context_bidirectional_formula_and_zero_gap_policy() -> None:
    # Context 0 has gap [1, 0].  Fine 0 makes half-progress in each direction;
    # fine 1 moves backward in sufficiency and is ReLU-clipped.  Context 1 has
    # exactly zero clean/corrupt gap and contributes a registered zero.
    clean = torch.tensor([[[1.0, 0.0]], [[3.0, 3.0]]])
    corrupt = torch.tensor([[[0.0, 0.0]], [[3.0, 3.0]]])
    clean_into_corrupt = torch.tensor(
        [
            [[[0.5, 0.0]], [[9.0, 9.0]]],
            [[[-1.0, 0.0]], [[8.0, 8.0]]],
        ]
    )
    corrupt_into_clean = torch.tensor(
        [
            [[[0.5, 0.0]], [[1.0, 1.0]]],
            [[[0.0, 0.0]], [[2.0, 2.0]]],
        ]
    )

    scored = bidirectional_fine_replacement_scores(
        clean_output=clean,
        corrupt_output=corrupt,
        clean_into_corrupt_outputs=clean_into_corrupt,
        corrupt_into_clean_outputs=corrupt_into_clean,
    )

    assert torch.equal(scored.informative_context_mask, torch.tensor([True, False]))
    assert scored.sufficiency_by_fine_context.tolist() == [[0.5, 0.0], [-1.0, 0.0]]
    assert scored.necessity_by_fine_context.tolist() == [[0.5, 0.0], [1.0, 0.0]]
    assert scored.bidirectional_by_fine_context.tolist() == [[0.5, 0.0], [0.0, 0.0]]
    # B_f is mean over all registered contexts, including zero-gap contexts.
    assert scored.fine_scores.tolist() == [0.25, 0.0]


def test_bidirectional_scoring_validates_axes_and_finiteness() -> None:
    clean = torch.zeros(2, 3, 4)
    corrupt = torch.ones(2, 3, 4)
    replacements = torch.zeros(5, 2, 3, 4)

    with pytest.raises(ValueError, match="same shape"):
        bidirectional_fine_replacement_scores(
            clean_output=clean,
            corrupt_output=corrupt[:, :, :3],
            clean_into_corrupt_outputs=replacements,
            corrupt_into_clean_outputs=replacements,
        )
    bad = replacements.clone()
    bad[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        bidirectional_fine_replacement_scores(
            clean_output=clean,
            corrupt_output=corrupt,
            clean_into_corrupt_outputs=bad,
            corrupt_into_clean_outputs=replacements,
        )


def test_fine_qkv_scores_promote_by_max_to_whole_edges() -> None:
    promoted = promote_fine_scores_max(
        fine_scores=[0.2, 0.8, 0.4, 0.3],
        fine_to_whole_indices=[0, 0, 0, 1],
        whole_edge_count=2,
    )
    assert promoted.tolist() == [0.8, 0.3]

    with pytest.raises(ValueError, match="coverage"):
        promote_fine_scores_max(
            fine_scores=[0.2, 0.8],
            fine_to_whole_indices=[0, 0],
            whole_edge_count=2,
        )


def test_shortlist_scores_are_embedded_in_full_universe_at_fixed_floor() -> None:
    full = (("a", "b"), ("a", "c"), ("b", "c"), ("c", "d"))
    embedded = embed_shortlist_scores(
        full_edge_universe=full,
        shortlisted_edges=(full[2], full[0]),
        shortlisted_scores=[0.7, 0.2],
    )
    assert embedded.tolist() == [0.2, -1.0, 0.7, -1.0]

    with pytest.raises(ValueError, match="above the full-universe floor"):
        embed_shortlist_scores(
            full_edge_universe=full,
            shortlisted_edges=(full[0],),
            shortlisted_scores=[-1.0],
        )


def test_best_eap_variant_is_conservative_post_outcome_comparator() -> None:
    name, metrics = best_eap_variant(
        {
            "eap_signed_max": {"auprc": 0.5, "true_positives_at_gt_k": 2},
            "eap_absolute_max": {"auprc": 0.6, "true_positives_at_gt_k": 1},
            "eap_ig_signed_max": {"auprc": 0.55, "true_positives_at_gt_k": 3},
            "eap_ig_absolute_max": {"auprc": 0.6, "true_positives_at_gt_k": 2},
        }
    )
    # Ties use the caller's frozen insertion order, never a favorable TP tie-break.
    assert name == "eap_absolute_max"
    assert metrics["auprc"] == 0.6


def test_direct_vs_transitive_auroc_ignores_unrelated_edges() -> None:
    edges = (("a", "b"), ("a", "c"), ("a", "d"), ("b", "d"))
    assert (
        direct_vs_transitive_auroc(
            scores=[0.9, 0.2, 100.0, 0.8],
            edge_universe=edges,
            direct_edges=(edges[0], edges[3]),
            transitive_non_edges=(edges[1],),
        )
        == 1.0
    )


def test_tie_aware_spearman_and_constant_policy() -> None:
    assert spearman_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman_correlation([1, 1, 2], [4, 4, 5]) == pytest.approx(1.0)
    assert spearman_correlation([1, 1], [2, 2]) == 1.0
    assert spearman_correlation([1, 1], [2, 3]) == 0.0


def test_useful_and_superiority_gates_apply_capped_tp_rule() -> None:
    verdict = evaluate_case_gates(
        ground_truth_edge_count=4,
        direct_edge_retention=1.0,
        induced_whole_edge_fraction=0.25,
        induced_fine_edge_fraction=0.28,
        operational_model_example_savings=0.70,
        confirmation_auprc=0.96,
        confirmation_true_positives_at_gt_k=4,
        exhaustive_exact_auprc=0.97,
        exhaustive_exact_true_positives_at_gt_k=4,
        best_eap_auprc=0.93,
        best_eap_true_positives_at_gt_k=4,
        raw_pair_auprc=0.80,
        raw_pair_true_positives_at_gt_k=2,
        other_exhaustive_comparator_auprcs=(0.89, 0.88),
        other_exhaustive_comparator_true_positives=(3, 2),
        direct_vs_transitive_auroc=0.85,
        node_gated_raw_direct_vs_transitive_auroc=0.70,
        discovery_confirmation_spearman=0.8,
    )

    assert verdict["screen_pass"]
    assert verdict["useful_pass"]
    # AP is not +.05 above exhaustive exact, but perfect baselines do not make
    # the TP improvement condition impossible.
    assert verdict["superiority_checks"]["tp_improvement_capped_at_perfect"]
    assert not verdict["superiority_pass"]
