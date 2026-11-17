# ruff: noqa: EXE002
"""Pure, scalable signed singleton and exact group-local edge selection."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from experiments.e32_interpbench.e32_5_protocol import (
    ADDITION_DIRECTIONAL_QUOTA,
    ADDITION_RESIDUAL_QUOTA,
    LOCAL_SEARCH_ROUNDS,
    REMOVAL_DIRECTIONAL_QUOTA,
    REMOVAL_RESIDUAL_QUOTA,
)


def _finite_double(value: Any, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must be finite")
    return tensor


def _edge_ids(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("fine_edge_ids must be a sequence")
    edge_ids = tuple(value)
    if (
        not edge_ids
        or len(edge_ids) != len(set(edge_ids))
        or any(not isinstance(edge_id, str) or not edge_id for edge_id in edge_ids)
    ):
        raise ValueError("fine_edge_ids must be unique nonempty strings")
    return edge_ids


def _canonical_support(
    edge_ids: tuple[str, ...], support: Sequence[str], *, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(support, (str, bytes)):
        raise TypeError("support must be a sequence")
    supplied = tuple(support)
    selected = frozenset(supplied)
    if len(selected) != len(supplied) or not selected.issubset(edge_ids):
        raise ValueError("support must contain unique members of fine_edge_ids")
    if not allow_empty and not selected:
        raise ValueError("support must be nonempty")
    return tuple(edge_id for edge_id in edge_ids if edge_id in selected)


def _group_indices(value: Sequence[int], *, count: int) -> tuple[int, ...]:
    groups = tuple(value)
    if len(groups) != count or any(
        isinstance(group, bool) or not isinstance(group, int) or group < 0
        for group in groups
    ):
        raise ValueError("fine_to_group_indices must be nonnegative integers per edge")
    return groups


@dataclass(frozen=True)
class SingletonPartitionScores:
    necessity_projection: torch.Tensor
    sufficiency_projection: torch.Tensor
    signed_score: torch.Tensor
    unsigned_exact_mass: torch.Tensor
    gap_energy: float


@dataclass(frozen=True)
class SwapShortlists:
    removal_residual: tuple[str, ...]
    removal_directional: tuple[str, ...]
    addition_residual: tuple[str, ...]
    addition_directional: tuple[str, ...]
    removals: tuple[str, ...]
    additions: tuple[str, ...]


@dataclass(frozen=True)
class ExactGroupEvaluation:
    support: tuple[str, ...]
    complement: tuple[str, ...]
    joint_faithfulness: float
    gate_pass: bool
    selected_output: torch.Tensor
    complement_output: torch.Tensor
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class CandidateScore:
    removed_edge: str
    added_edge: str
    support: tuple[str, ...]
    joint_faithfulness: float
    gate_pass: bool


@dataclass(frozen=True)
class SearchRound:
    round_index: int
    starting_support: tuple[str, ...]
    starting_joint_faithfulness: float
    shortlists: SwapShortlists
    exact_candidate_scores: tuple[CandidateScore, ...]
    accepted_support: tuple[str, ...] | None
    accepted_joint_faithfulness: float | None


@dataclass(frozen=True)
class CandidateMove:
    removed_edge: str
    added_edge: str
    support: tuple[str, ...]


@dataclass(frozen=True)
class ExactSwapProposal:
    round_index: int
    starting_support: tuple[str, ...]
    starting_joint_faithfulness: float
    output_shape: tuple[int, ...]
    shortlists: SwapShortlists
    moves: tuple[CandidateMove, ...]

    @property
    def supports(self) -> tuple[tuple[str, ...], ...]:
        return tuple(move.support for move in self.moves)


@dataclass(frozen=True)
class ExactSwapOutcome:
    round_record: SearchRound | None
    accepted_evaluation: ExactGroupEvaluation | None
    stop_reason: str | None


@dataclass(frozen=True)
class LocalSearchResult:
    initial_support: tuple[str, ...]
    final_support: tuple[str, ...]
    accepted_path: tuple[tuple[str, ...], ...]
    accepted_path_joint_faithfulness: tuple[float, ...]
    rounds: tuple[SearchRound, ...]
    exact_support_evaluations: int
    stop_reason: str


@dataclass(frozen=True)
class ValidationPathScore:
    path_index: int
    support: tuple[str, ...]
    joint_faithfulness: float
    gate_pass: bool


@dataclass(frozen=True)
class ValidationPathSelection:
    selected_path_index: int
    selected_support: tuple[str, ...]
    selected_evaluation: ExactGroupEvaluation
    path_scores: tuple[ValidationPathScore, ...]


def signed_singleton_partition_scores(
    *,
    clean_output: Any,
    all_internal_output: Any,
    singleton_outputs: Any,
    leave_one_clean_outputs: Any,
) -> SingletonPartitionScores:
    r"""Score F({e}) necessity and F(U\{e}) sufficiency against one full-U gap."""
    clean = _finite_double(clean_output, label="clean_output")
    all_internal = _finite_double(all_internal_output, label="all_internal_output")
    singleton = _finite_double(singleton_outputs, label="singleton_outputs")
    leave_one = _finite_double(leave_one_clean_outputs, label="leave_one_clean_outputs")
    if (
        clean.ndim != 3
        or all_internal.shape != clean.shape
        or singleton.ndim != 4
        or leave_one.shape != singleton.shape
        or tuple(singleton.shape[1:]) != tuple(clean.shape)
        or singleton.shape[0] < 1
    ):
        raise ValueError(
            "partition outputs must align as [example,position,class] and "
            "[fine_edge,example,position,class]"
        )
    gap = clean - all_internal
    gap_energy = float(gap.square().sum())
    if gap_energy <= 0.0:
        raise ValueError("clean versus F(U) endpoint must be informative")
    necessity_effect = clean.unsqueeze(0) - singleton
    sufficiency_effect = leave_one - all_internal.unsqueeze(0)
    axes = tuple(range(1, singleton.ndim))
    necessity = (necessity_effect * gap.unsqueeze(0)).sum(dim=axes) / gap_energy
    sufficiency = (sufficiency_effect * gap.unsqueeze(0)).sum(dim=axes) / gap_energy
    signed = torch.sqrt(torch.relu(necessity) * torch.relu(sufficiency))
    unsigned = (
        0.5
        * (
            necessity_effect.square().sum(dim=axes)
            + sufficiency_effect.square().sum(dim=axes)
        )
        / gap_energy
    )
    return SingletonPartitionScores(
        necessity, sufficiency, signed, unsigned, gap_energy
    )


def residual_swap_gains(
    *,
    clean_output: Any,
    all_internal_output: Any,
    singleton_outputs: Any,
    leave_one_clean_outputs: Any,
    selected_output: Any,
    complement_output: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return additive residual gains used only to shortlist exact swaps."""
    partition = signed_singleton_partition_scores(
        clean_output=clean_output,
        all_internal_output=all_internal_output,
        singleton_outputs=singleton_outputs,
        leave_one_clean_outputs=leave_one_clean_outputs,
    )
    clean = _finite_double(clean_output, label="clean_output")
    all_internal = _finite_double(all_internal_output, label="all_internal_output")
    singleton = _finite_double(singleton_outputs, label="singleton_outputs")
    leave_one = _finite_double(leave_one_clean_outputs, label="leave_one_clean_outputs")
    selected = _finite_double(selected_output, label="selected_output")
    complement = _finite_double(complement_output, label="complement_output")
    if selected.shape != clean.shape or complement.shape != clean.shape:
        raise ValueError("current group outputs must align with endpoint outputs")
    necessity_marginal = clean.unsqueeze(0) - singleton
    sufficiency_marginal = leave_one - all_internal.unsqueeze(0)
    necessity_residual = selected - all_internal
    sufficiency_residual = clean - complement
    axes = tuple(range(1, singleton.ndim))

    def gain(sign: float) -> torch.Tensor:
        next_necessity = necessity_residual.unsqueeze(0) + sign * necessity_marginal
        next_sufficiency = (
            sufficiency_residual.unsqueeze(0) + sign * sufficiency_marginal
        )
        before = necessity_residual.square().sum() + sufficiency_residual.square().sum()
        after = next_necessity.square().sum(dim=axes) + next_sufficiency.square().sum(
            dim=axes
        )
        return (before - after) / (2.0 * partition.gap_energy)

    # Adding an edge subtracts its two singleton marginals from the residual;
    # removing an edge adds them back. Exact group evaluation makes the decision.
    addition_gain = gain(-1.0)
    removal_gain = gain(1.0)
    return removal_gain, addition_gain


def _rank(
    indices: Sequence[int], scores: torch.Tensor, *, descending: bool
) -> tuple[int, ...]:
    direction = -1.0 if descending else 1.0
    return tuple(
        sorted(indices, key=lambda index: (direction * float(scores[index]), index))
    )


def initial_static_support(
    *,
    fine_edge_ids: Sequence[str],
    directional_scores: Any,
    support_size: int,
) -> tuple[str, ...]:
    """Return the canonical fixed-cardinality static signed initialization."""
    edge_ids = _edge_ids(fine_edge_ids)
    directional = _finite_double(directional_scores, label="directional_scores")
    if (
        directional.ndim != 1
        or directional.numel() != len(edge_ids)
        or isinstance(support_size, bool)
        or not isinstance(support_size, int)
        or not 1 <= support_size < len(edge_ids)
    ):
        raise ValueError("support size and directional scores must align with full U")
    ranked = _rank(range(len(edge_ids)), directional, descending=True)
    selected = frozenset(ranked[:support_size])
    return tuple(edge_id for index, edge_id in enumerate(edge_ids) if index in selected)


def _group_diverse(
    ranked_indices: Sequence[int], groups: tuple[int, ...]
) -> tuple[int, ...]:
    first: list[int] = []
    later: list[int] = []
    seen: set[int] = set()
    for index in ranked_indices:
        if groups[index] in seen:
            later.append(index)
        else:
            seen.add(groups[index])
            first.append(index)
    return (*first, *later)


def _take_new(ranked: Sequence[int], *, quota: int, used: set[int]) -> tuple[int, ...]:
    if isinstance(quota, bool) or not isinstance(quota, int) or quota < 0:
        raise ValueError("shortlist quotas must be nonnegative integers")
    if quota == 0:
        return ()
    chosen: list[int] = []
    for index in ranked:
        if index not in used:
            chosen.append(index)
            used.add(index)
            if len(chosen) == quota:
                break
    return tuple(chosen)


def residual_swap_shortlists(
    *,
    fine_edge_ids: Sequence[str],
    fine_to_group_indices: Sequence[int],
    current_support: Sequence[str],
    directional_scores: Any,
    predicted_removal_gain: Any,
    predicted_addition_gain: Any,
    removal_residual_quota: int = REMOVAL_RESIDUAL_QUOTA,
    removal_directional_quota: int = REMOVAL_DIRECTIONAL_QUOTA,
    addition_residual_quota: int = ADDITION_RESIDUAL_QUOTA,
    addition_directional_quota: int = ADDITION_DIRECTIONAL_QUOTA,
) -> SwapShortlists:
    """Union frozen residual and static channels; neither channel selects a set."""
    edge_ids = _edge_ids(fine_edge_ids)
    groups = _group_indices(fine_to_group_indices, count=len(edge_ids))
    support = _canonical_support(edge_ids, current_support)
    selected = frozenset(support)
    directional = _finite_double(directional_scores, label="directional_scores")
    removal = _finite_double(predicted_removal_gain, label="predicted_removal_gain")
    addition = _finite_double(predicted_addition_gain, label="predicted_addition_gain")
    if any(
        vector.ndim != 1 or vector.numel() != len(edge_ids)
        for vector in (directional, removal, addition)
    ):
        raise ValueError("all shortlist score vectors must align with fine_edge_ids")
    selected_indices = tuple(
        i for i, edge_id in enumerate(edge_ids) if edge_id in selected
    )
    outside_indices = tuple(
        i for i, edge_id in enumerate(edge_ids) if edge_id not in selected
    )

    removal_used: set[int] = set()
    removal_residual = _take_new(
        _rank(selected_indices, removal, descending=True),
        quota=removal_residual_quota,
        used=removal_used,
    )
    removal_directional = _take_new(
        _rank(selected_indices, directional, descending=False),
        quota=removal_directional_quota,
        used=removal_used,
    )
    addition_used: set[int] = set()
    addition_residual = _take_new(
        _rank(outside_indices, addition, descending=True),
        quota=addition_residual_quota,
        used=addition_used,
    )
    directional_rank = _group_diverse(
        _rank(outside_indices, directional, descending=True), groups
    )
    addition_directional = _take_new(
        directional_rank,
        quota=addition_directional_quota,
        used=addition_used,
    )

    def names(indices: Sequence[int]) -> tuple[str, ...]:
        return tuple(edge_ids[index] for index in indices)

    return SwapShortlists(
        removal_residual=names(removal_residual),
        removal_directional=names(removal_directional),
        addition_residual=names(addition_residual),
        addition_directional=names(addition_directional),
        removals=names(sorted(removal_used)),
        additions=names(sorted(addition_used)),
    )


def _validate_exact_evaluation(
    evaluation: ExactGroupEvaluation,
    *,
    edge_ids: tuple[str, ...],
    expected_support: tuple[str, ...],
    output_shape: torch.Size,
) -> None:
    expected_complement = tuple(
        edge_id for edge_id in edge_ids if edge_id not in frozenset(expected_support)
    )
    if (
        evaluation.support != expected_support
        or evaluation.complement != expected_complement
    ):
        raise ValueError("exact evaluator changed the requested full-U partition")
    if not math.isfinite(float(evaluation.joint_faithfulness)):
        raise ValueError("exact evaluator returned nonfinite joint faithfulness")
    if (
        torch.as_tensor(evaluation.selected_output).shape != output_shape
        or torch.as_tensor(evaluation.complement_output).shape != output_shape
    ):
        raise ValueError("exact evaluator outputs changed the registered output shape")


def propose_exact_swap_batch(
    *,
    fine_edge_ids: Sequence[str],
    fine_to_group_indices: Sequence[int],
    directional_scores: Any,
    clean_output: Any,
    all_internal_output: Any,
    singleton_outputs: Any,
    leave_one_clean_outputs: Any,
    current_evaluation: ExactGroupEvaluation,
    round_index: int,
    removal_residual_quota: int = REMOVAL_RESIDUAL_QUOTA,
    removal_directional_quota: int = REMOVAL_DIRECTIONAL_QUOTA,
    addition_residual_quota: int = ADDITION_RESIDUAL_QUOTA,
    addition_directional_quota: int = ADDITION_DIRECTIONAL_QUOTA,
) -> ExactSwapProposal:
    """Propose one canonical exact-swap batch without choosing an outcome."""
    edge_ids = _edge_ids(fine_edge_ids)
    groups = _group_indices(fine_to_group_indices, count=len(edge_ids))
    clean = _finite_double(clean_output, label="clean_output")
    if (
        isinstance(round_index, bool)
        or not isinstance(round_index, int)
        or round_index < 0
    ):
        raise ValueError("round_index must be a nonnegative integer")
    if not isinstance(current_evaluation, ExactGroupEvaluation):
        raise TypeError("current_evaluation must be ExactGroupEvaluation")
    current_support = _canonical_support(edge_ids, current_evaluation.support)
    if current_support != current_evaluation.support:
        raise ValueError("current evaluation support is not in canonical full-U order")
    _validate_exact_evaluation(
        current_evaluation,
        edge_ids=edge_ids,
        expected_support=current_support,
        output_shape=clean.shape,
    )
    removal_gain, addition_gain = residual_swap_gains(
        clean_output=clean,
        all_internal_output=all_internal_output,
        singleton_outputs=singleton_outputs,
        leave_one_clean_outputs=leave_one_clean_outputs,
        selected_output=current_evaluation.selected_output,
        complement_output=current_evaluation.complement_output,
    )
    shortlists = residual_swap_shortlists(
        fine_edge_ids=edge_ids,
        fine_to_group_indices=groups,
        current_support=current_support,
        directional_scores=directional_scores,
        predicted_removal_gain=removal_gain,
        predicted_addition_gain=addition_gain,
        removal_residual_quota=removal_residual_quota,
        removal_directional_quota=removal_directional_quota,
        addition_residual_quota=addition_residual_quota,
        addition_directional_quota=addition_directional_quota,
    )
    moves: list[CandidateMove] = []
    current_members = frozenset(current_support)
    for removed in shortlists.removals:
        for added in shortlists.additions:
            members = (current_members - {removed}) | {added}
            support = tuple(edge_id for edge_id in edge_ids if edge_id in members)
            moves.append(CandidateMove(removed, added, support))
    supports = tuple(move.support for move in moves)
    if len(supports) != len(set(supports)):
        raise RuntimeError("canonical swap proposal unexpectedly contains duplicates")
    return ExactSwapProposal(
        round_index=round_index,
        starting_support=current_support,
        starting_joint_faithfulness=float(current_evaluation.joint_faithfulness),
        output_shape=tuple(clean.shape),
        shortlists=shortlists,
        moves=tuple(moves),
    )


def accept_exact_swap_batch(
    *,
    fine_edge_ids: Sequence[str],
    proposal: ExactSwapProposal,
    candidate_evaluations: Mapping[tuple[str, ...], ExactGroupEvaluation],
) -> ExactSwapOutcome:
    """Apply canonical exact ties and strict acceptance to one proposed batch."""
    edge_ids = _edge_ids(fine_edge_ids)
    if not isinstance(proposal, ExactSwapProposal):
        raise TypeError("proposal must be ExactSwapProposal")
    if not isinstance(candidate_evaluations, Mapping):
        raise TypeError("candidate_evaluations must be a mapping")
    starting = _canonical_support(edge_ids, proposal.starting_support)
    if (
        starting != proposal.starting_support
        or not math.isfinite(proposal.starting_joint_faithfulness)
        or isinstance(proposal.round_index, bool)
        or not isinstance(proposal.round_index, int)
        or proposal.round_index < 0
    ):
        raise ValueError("swap proposal starting state is malformed")
    expected_supports: list[tuple[str, ...]] = []
    starting_members = frozenset(starting)
    for move in proposal.moves:
        if (
            move.removed_edge not in starting_members
            or move.added_edge in starting_members
        ):
            raise ValueError("swap proposal move is not one-for-one")
        expected_members = (starting_members - {move.removed_edge}) | {move.added_edge}
        expected_support = tuple(
            edge_id for edge_id in edge_ids if edge_id in expected_members
        )
        if move.support != expected_support:
            raise ValueError("swap proposal support differs from its declared move")
        expected_supports.append(expected_support)
    supports = tuple(expected_supports)
    if len(supports) != len(set(supports)):
        raise ValueError("swap proposal supports must be unique")
    if tuple(candidate_evaluations) != supports:
        raise ValueError(
            "candidate evaluations must preserve the exact requested support order"
        )
    if not supports:
        return ExactSwapOutcome(None, None, "no_swap_candidates")

    candidates: list[CandidateScore] = []
    best: ExactGroupEvaluation | None = None
    best_indices: tuple[int, ...] | None = None
    index_of = {edge_id: index for index, edge_id in enumerate(edge_ids)}
    for move in proposal.moves:
        evaluation = candidate_evaluations[move.support]
        _validate_exact_evaluation(
            evaluation,
            edge_ids=edge_ids,
            expected_support=move.support,
            output_shape=torch.Size(proposal.output_shape),
        )
        candidates.append(
            CandidateScore(
                move.removed_edge,
                move.added_edge,
                move.support,
                float(evaluation.joint_faithfulness),
                bool(evaluation.gate_pass),
            )
        )
        candidate_indices = tuple(index_of[edge] for edge in move.support)
        if (
            best is None
            or evaluation.joint_faithfulness > best.joint_faithfulness
            or (
                evaluation.joint_faithfulness == best.joint_faithfulness
                and best_indices is not None
                and candidate_indices < best_indices
            )
        ):
            best = evaluation
            best_indices = candidate_indices
    assert best is not None
    accepted = best.joint_faithfulness > proposal.starting_joint_faithfulness
    round_record = SearchRound(
        proposal.round_index,
        proposal.starting_support,
        proposal.starting_joint_faithfulness,
        proposal.shortlists,
        tuple(candidates),
        best.support if accepted else None,
        float(best.joint_faithfulness) if accepted else None,
    )
    return ExactSwapOutcome(
        round_record,
        best if accepted else None,
        None if accepted else "no_strict_exact_improvement",
    )


def _evaluate_exact_batch(
    batch_evaluator: Callable[
        [Sequence[Sequence[str]]], Mapping[tuple[str, ...], ExactGroupEvaluation]
    ],
    supports: tuple[tuple[str, ...], ...],
) -> tuple[ExactGroupEvaluation, ...]:
    if not supports or len(supports) != len(set(supports)):
        raise ValueError("exact support batch must be nonempty and unique")
    observed = batch_evaluator(supports)
    if not isinstance(observed, Mapping) or tuple(observed) != supports:
        raise ValueError(
            "batch evaluator must preserve the exact requested support order"
        )
    return tuple(observed[support] for support in supports)


def exact_group_local_search(
    *,
    fine_edge_ids: Sequence[str],
    fine_to_group_indices: Sequence[int],
    directional_scores: Any,
    clean_output: Any,
    all_internal_output: Any,
    singleton_outputs: Any,
    leave_one_clean_outputs: Any,
    support_size: int,
    batch_evaluator: Callable[
        [Sequence[Sequence[str]]], Mapping[tuple[str, ...], ExactGroupEvaluation]
    ],
    max_rounds: int = LOCAL_SEARCH_ROUNDS,
    removal_residual_quota: int = REMOVAL_RESIDUAL_QUOTA,
    removal_directional_quota: int = REMOVAL_DIRECTIONAL_QUOTA,
    addition_residual_quota: int = ADDITION_RESIDUAL_QUOTA,
    addition_directional_quota: int = ADDITION_DIRECTIONAL_QUOTA,
) -> LocalSearchResult:
    r"""Run fixed-cardinality swaps; only exact S/U\S evaluations choose or stop."""
    edge_ids = _edge_ids(fine_edge_ids)
    groups = _group_indices(fine_to_group_indices, count=len(edge_ids))
    directional = _finite_double(directional_scores, label="directional_scores")
    clean = _finite_double(clean_output, label="clean_output")
    all_internal = _finite_double(all_internal_output, label="all_internal_output")
    singleton = _finite_double(singleton_outputs, label="singleton_outputs")
    leave_one = _finite_double(leave_one_clean_outputs, label="leave_one_clean_outputs")
    if (
        directional.ndim != 1
        or directional.numel() != len(edge_ids)
        or isinstance(support_size, bool)
        or not isinstance(support_size, int)
        or not 1 <= support_size < len(edge_ids)
        or isinstance(max_rounds, bool)
        or not isinstance(max_rounds, int)
        or max_rounds < 0
    ):
        raise ValueError("search sizes and directional scores must align with full U")
    # This also validates the singleton partition shapes and informative endpoint.
    signed_singleton_partition_scores(
        clean_output=clean,
        all_internal_output=all_internal,
        singleton_outputs=singleton,
        leave_one_clean_outputs=leave_one,
    )
    initial = initial_static_support(
        fine_edge_ids=edge_ids,
        directional_scores=directional,
        support_size=support_size,
    )
    current_support = initial
    current = _evaluate_exact_batch(batch_evaluator, (current_support,))[0]
    _validate_exact_evaluation(
        current,
        edge_ids=edge_ids,
        expected_support=current_support,
        output_shape=clean.shape,
    )
    exact_count = 1
    accepted_path = [current_support]
    accepted_joints = [float(current.joint_faithfulness)]
    round_records: list[SearchRound] = []
    stop_reason = "maximum_rounds_reached"

    for round_index in range(max_rounds):
        proposal = propose_exact_swap_batch(
            fine_edge_ids=edge_ids,
            fine_to_group_indices=groups,
            directional_scores=directional,
            clean_output=clean,
            all_internal_output=all_internal,
            singleton_outputs=singleton,
            leave_one_clean_outputs=leave_one,
            current_evaluation=current,
            round_index=round_index,
            removal_residual_quota=removal_residual_quota,
            removal_directional_quota=removal_directional_quota,
            addition_residual_quota=addition_residual_quota,
            addition_directional_quota=addition_directional_quota,
        )
        candidate_evaluations = (
            {} if not proposal.moves else batch_evaluator(proposal.supports)
        )
        outcome = accept_exact_swap_batch(
            fine_edge_ids=edge_ids,
            proposal=proposal,
            candidate_evaluations=candidate_evaluations,
        )
        exact_count += len(proposal.moves)
        if outcome.round_record is not None:
            round_records.append(outcome.round_record)
        if outcome.stop_reason is not None:
            stop_reason = outcome.stop_reason
            break
        if outcome.accepted_evaluation is None:  # pragma: no cover - invariant
            raise RuntimeError("accepted exact swap lacks its evaluation")
        current = outcome.accepted_evaluation
        current_support = current.support
        accepted_path.append(current_support)
        accepted_joints.append(float(current.joint_faithfulness))

    return LocalSearchResult(
        initial,
        current_support,
        tuple(accepted_path),
        tuple(accepted_joints),
        tuple(round_records),
        exact_count,
        stop_reason,
    )


def select_validation_path_support(
    *,
    accepted_path: Sequence[Sequence[str]],
    batch_evaluator: Callable[
        [Sequence[Sequence[str]]], Mapping[tuple[str, ...], ExactGroupEvaluation]
    ],
) -> ValidationPathSelection:
    """Evaluate only the frozen search path and choose the earliest exact maximum."""
    path = tuple(tuple(support) for support in accepted_path)
    if not path or len(path) != len(set(path)):
        raise ValueError("accepted_path must contain unique supports")
    evaluations = list(_evaluate_exact_batch(batch_evaluator, path))
    scores: list[ValidationPathScore] = []
    for path_index, (support, evaluation) in enumerate(
        zip(path, evaluations, strict=True)
    ):
        if evaluation.support != support or not math.isfinite(
            float(evaluation.joint_faithfulness)
        ):
            raise ValueError("validation evaluator changed a frozen path support")
        scores.append(
            ValidationPathScore(
                path_index,
                support,
                float(evaluation.joint_faithfulness),
                bool(evaluation.gate_pass),
            )
        )
    best_value = max(score.joint_faithfulness for score in scores)
    selected_index = next(
        score.path_index for score in scores if score.joint_faithfulness == best_value
    )
    return ValidationPathSelection(
        selected_index,
        path[selected_index],
        evaluations[selected_index],
        tuple(scores),
    )


__all__ = [
    "CandidateMove",
    "CandidateScore",
    "ExactGroupEvaluation",
    "ExactSwapOutcome",
    "ExactSwapProposal",
    "LocalSearchResult",
    "SearchRound",
    "SingletonPartitionScores",
    "SwapShortlists",
    "ValidationPathScore",
    "ValidationPathSelection",
    "accept_exact_swap_batch",
    "exact_group_local_search",
    "initial_static_support",
    "propose_exact_swap_batch",
    "residual_swap_gains",
    "residual_swap_shortlists",
    "select_validation_path_support",
    "signed_singleton_partition_scores",
]
