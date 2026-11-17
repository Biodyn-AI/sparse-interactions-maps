# ruff: noqa: EXE002
from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from experiments.e32_interpbench.e32_5_selector import (
    ExactGroupEvaluation,
    accept_exact_swap_batch,
    exact_group_local_search,
    initial_static_support,
    propose_exact_swap_batch,
    residual_swap_shortlists,
    select_validation_path_support,
    signed_singleton_partition_scores,
)


def _outputs(value: float) -> torch.Tensor:
    return torch.tensor([[[value, -value]]], dtype=torch.float64)


def test_partition_singleton_score_preserves_direction_and_complement_context() -> None:
    clean = _outputs(1.0)
    all_internal = _outputs(0.0)
    singleton = torch.stack((_outputs(0.5), _outputs(1.5), _outputs(-1.0)))
    leave_one_clean = torch.stack((_outputs(0.5), _outputs(-0.5), _outputs(2.0)))
    score = signed_singleton_partition_scores(
        clean_output=clean,
        all_internal_output=all_internal,
        singleton_outputs=singleton,
        leave_one_clean_outputs=leave_one_clean,
    )

    assert score.necessity_projection.tolist() == pytest.approx([0.5, -0.5, 2.0])
    assert score.sufficiency_projection.tolist() == pytest.approx([0.5, -0.5, 2.0])
    assert score.signed_score.tolist() == pytest.approx([0.5, 0.0, 2.0])
    assert score.unsigned_exact_mass.tolist() == pytest.approx([0.25, 0.25, 4.0])


def test_shortlist_is_union_of_residual_and_static_group_diverse_channels() -> None:
    edge_ids = tuple(f"e{i}" for i in range(8))
    groups = (0, 0, 1, 1, 2, 2, 3, 4)
    support = ("e0", "e2", "e4", "e6")
    directional = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    removal_gain = torch.tensor([9.0, 0.0, 8.0, 0.0, 1.0, 0.0, 2.0, 0.0])
    addition_gain = torch.tensor([0.0, 9.0, 0.0, 8.0, 0.0, 7.0, 0.0, 6.0])

    shortlist = residual_swap_shortlists(
        fine_edge_ids=edge_ids,
        fine_to_group_indices=groups,
        current_support=support,
        directional_scores=directional,
        predicted_removal_gain=removal_gain,
        predicted_addition_gain=addition_gain,
        removal_residual_quota=2,
        removal_directional_quota=2,
        addition_residual_quota=2,
        addition_directional_quota=2,
    )

    assert shortlist.removal_residual == ("e0", "e2")
    assert shortlist.removal_directional == ("e6", "e4")
    assert shortlist.removals == ("e0", "e2", "e4", "e6")
    assert shortlist.addition_residual == ("e1", "e3")
    # Directional backfill skips the residual picks and spans new whole groups.
    assert shortlist.addition_directional == ("e5", "e7")
    assert shortlist.additions == ("e1", "e3", "e5", "e7")


def test_exact_group_search_not_residual_proxy_chooses_and_stops() -> None:
    edge_ids = ("a", "b", "c", "d")
    groups = (0, 1, 2, 3)
    directional = torch.tensor([4.0, 3.0, 2.0, 1.0])
    singleton = torch.stack(tuple(_outputs(0.5) for _ in edge_ids))
    leave_one = torch.stack(tuple(_outputs(0.5) for _ in edge_ids))
    exact_joint = {
        ("a", "b"): 0.40,
        ("a", "c"): 0.30,
        ("a", "d"): 0.20,
        ("b", "c"): 0.95,
        ("b", "d"): 0.10,
        ("c", "d"): 0.05,
    }
    calls: list[tuple[str, ...]] = []

    def evaluate_one(support: Sequence[str]) -> ExactGroupEvaluation:
        canonical = tuple(edge for edge in edge_ids if edge in frozenset(support))
        calls.append(canonical)
        joint = exact_joint[canonical]
        return ExactGroupEvaluation(
            support=canonical,
            complement=tuple(edge for edge in edge_ids if edge not in canonical),
            joint_faithfulness=joint,
            gate_pass=joint >= 0.9,
            selected_output=_outputs(joint),
            complement_output=_outputs(1.0 - joint),
            metrics={"joint": joint},
        )

    batch_sizes: list[int] = []

    def batch_evaluator(
        supports: Sequence[Sequence[str]],
    ) -> dict[tuple[str, ...], ExactGroupEvaluation]:
        batch_sizes.append(len(supports))
        return {tuple(support): evaluate_one(support) for support in supports}

    search = exact_group_local_search(
        fine_edge_ids=edge_ids,
        fine_to_group_indices=groups,
        directional_scores=directional,
        clean_output=_outputs(1.0),
        all_internal_output=_outputs(0.0),
        singleton_outputs=singleton,
        leave_one_clean_outputs=leave_one,
        support_size=2,
        batch_evaluator=batch_evaluator,
        max_rounds=3,
        removal_residual_quota=1,
        removal_directional_quota=1,
        addition_residual_quota=1,
        addition_directional_quota=1,
    )

    assert search.initial_support == ("a", "b")
    assert search.final_support == ("b", "c")
    assert search.accepted_path == (("a", "b"), ("b", "c"))
    assert search.stop_reason == "no_strict_exact_improvement"
    assert ("b", "c") in calls
    assert all(round_record.exact_candidate_scores for round_record in search.rounds)
    assert batch_sizes[0] == 1
    assert all(size <= 4 for size in batch_sizes[1:])


def test_stepwise_primitives_match_monolithic_candidate_order_and_acceptance() -> None:
    edge_ids = ("a", "b", "c", "d")
    groups = (0, 1, 2, 3)
    directional = torch.tensor([4.0, 3.0, 2.0, 1.0])
    singleton = torch.stack(tuple(_outputs(0.5) for _ in edge_ids))
    leave_one = torch.stack(tuple(_outputs(0.5) for _ in edge_ids))
    exact_joint = {
        ("a", "b"): 0.40,
        ("a", "c"): 0.30,
        ("a", "d"): 0.20,
        ("b", "c"): 0.95,
        ("b", "d"): 0.10,
        ("c", "d"): 0.05,
    }

    def evaluation(support: Sequence[str]) -> ExactGroupEvaluation:
        canonical = tuple(edge for edge in edge_ids if edge in frozenset(support))
        joint = exact_joint[canonical]
        return ExactGroupEvaluation(
            support=canonical,
            complement=tuple(edge for edge in edge_ids if edge not in canonical),
            joint_faithfulness=joint,
            gate_pass=joint >= 0.9,
            selected_output=_outputs(joint),
            complement_output=_outputs(1.0 - joint),
            metrics={"joint": joint},
        )

    initial = initial_static_support(
        fine_edge_ids=edge_ids,
        directional_scores=directional,
        support_size=2,
    )
    assert initial == ("a", "b")
    incumbent = evaluation(initial)
    proposal = propose_exact_swap_batch(
        fine_edge_ids=edge_ids,
        fine_to_group_indices=groups,
        directional_scores=directional,
        clean_output=_outputs(1.0),
        all_internal_output=_outputs(0.0),
        singleton_outputs=singleton,
        leave_one_clean_outputs=leave_one,
        current_evaluation=incumbent,
        round_index=0,
        removal_residual_quota=1,
        removal_directional_quota=1,
        addition_residual_quota=1,
        addition_directional_quota=1,
    )
    observed = {support: evaluation(support) for support in proposal.supports}
    with pytest.raises(ValueError, match="exact requested support order"):
        accept_exact_swap_batch(
            fine_edge_ids=edge_ids,
            proposal=proposal,
            candidate_evaluations=dict(reversed(tuple(observed.items()))),
        )
    outcome = accept_exact_swap_batch(
        fine_edge_ids=edge_ids,
        proposal=proposal,
        candidate_evaluations=observed,
    )
    assert outcome.stop_reason is None
    assert outcome.accepted_evaluation is not None
    assert outcome.accepted_evaluation.support == ("b", "c")
    tied = {
        support: ExactGroupEvaluation(
            support=support,
            complement=tuple(edge for edge in edge_ids if edge not in support),
            joint_faithfulness=0.50,
            gate_pass=False,
            selected_output=_outputs(0.50),
            complement_output=_outputs(0.50),
            metrics={},
        )
        for support in proposal.supports
    }
    tied_outcome = accept_exact_swap_batch(
        fine_edge_ids=edge_ids,
        proposal=proposal,
        candidate_evaluations=tied,
    )
    assert tied_outcome.accepted_evaluation is not None
    assert tied_outcome.accepted_evaluation.support == min(
        proposal.supports,
        key=lambda support: tuple(edge_ids.index(edge) for edge in support),
    )

    monolithic_batches: list[tuple[tuple[str, ...], ...]] = []

    def batch_evaluator(
        supports: Sequence[Sequence[str]],
    ) -> dict[tuple[str, ...], ExactGroupEvaluation]:
        canonical = tuple(tuple(support) for support in supports)
        monolithic_batches.append(canonical)
        return {support: evaluation(support) for support in canonical}

    search = exact_group_local_search(
        fine_edge_ids=edge_ids,
        fine_to_group_indices=groups,
        directional_scores=directional,
        clean_output=_outputs(1.0),
        all_internal_output=_outputs(0.0),
        singleton_outputs=singleton,
        leave_one_clean_outputs=leave_one,
        support_size=2,
        batch_evaluator=batch_evaluator,
        max_rounds=1,
        removal_residual_quota=1,
        removal_directional_quota=1,
        addition_residual_quota=1,
        addition_directional_quota=1,
    )
    assert monolithic_batches == [((initial),), proposal.supports]
    assert search.final_support == outcome.accepted_evaluation.support
    assert search.rounds == (outcome.round_record,)

    no_candidates = propose_exact_swap_batch(
        fine_edge_ids=edge_ids,
        fine_to_group_indices=groups,
        directional_scores=directional,
        clean_output=_outputs(1.0),
        all_internal_output=_outputs(0.0),
        singleton_outputs=singleton,
        leave_one_clean_outputs=leave_one,
        current_evaluation=incumbent,
        round_index=0,
        removal_residual_quota=0,
        removal_directional_quota=0,
        addition_residual_quota=0,
        addition_directional_quota=0,
    )
    stopped = accept_exact_swap_batch(
        fine_edge_ids=edge_ids,
        proposal=no_candidates,
        candidate_evaluations={},
    )
    assert stopped.round_record is None
    assert stopped.accepted_evaluation is None
    assert stopped.stop_reason == "no_swap_candidates"


def test_validation_selects_earliest_exact_best_without_changing_path() -> None:
    path = (("a", "b"), ("b", "c"), ("c", "d"))
    values = {path[0]: 0.7, path[1]: 0.9, path[2]: 0.9}

    def evaluator(support: Sequence[str]) -> ExactGroupEvaluation:
        canonical = tuple(support)
        return ExactGroupEvaluation(
            support=canonical,
            complement=(),
            joint_faithfulness=values[canonical],
            gate_pass=values[canonical] >= 0.9,
            selected_output=_outputs(values[canonical]),
            complement_output=_outputs(0.0),
            metrics={},
        )

    def batch_evaluator(
        supports: Sequence[Sequence[str]],
    ) -> dict[tuple[str, ...], ExactGroupEvaluation]:
        return {tuple(support): evaluator(support) for support in supports}

    selected = select_validation_path_support(
        accepted_path=path, batch_evaluator=batch_evaluator
    )
    assert selected.selected_path_index == 1
    assert selected.selected_support == ("b", "c")
    assert tuple(record.support for record in selected.path_scores) == path
