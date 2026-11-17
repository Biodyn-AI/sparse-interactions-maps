# ruff: noqa: EXE002
"""Split-aware live composition and raw custody for prospective E32.5.

Discovery and confirmation are deliberately separate entry points.  This file
first composes the D-only model calls needed by the case-58 development pilot:
one adaptive patching model on D64 and two distinct official EAP models on the
fixed search rows 0:32.  Validation rows 32:64 remain unavailable to all
comparator rankings.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import torch

from experiments.e32_interpbench.adapter import (
    BehaviorallyEligiblePair,
    OfficialDataPair,
)
from experiments.e32_interpbench.e32_4_eap import (
    GenericEAPRun,
    generic_eap_run_to_dict,
    run_generic_eap,
)
from experiments.e32_interpbench.e32_4_edge_sets import (
    FrozenEdgeSupport,
    InducedEdgeUniverse,
    build_edge_condition_plan,
    build_induced_edge_universe,
    edge_set_acquisition_to_dict,
)
from experiments.e32_interpbench.e32_5_acquisition import (
    E32_5_D_BATCH_SIZE,
    E32_5_TRAIN_ROW_COUNT,
    E32_5AdaptiveAcquisition,
    run_e32_5_adaptive_replacement,
)
from experiments.e32_interpbench.e32_5_cohort import (
    CASE_MANIFESTS,
    SPLIT_ORDER_BY_CASE,
    case_universe,
    load_verified_model,
    split_request,
)
from experiments.e32_interpbench.e32_5_protocol import (
    CASE_GEOMETRY,
    LOCAL_SEARCH_ROUNDS,
    PRIMARY_METHOD,
    VALIDATION_EXAMPLE_COUNT,
    confirmation_support_size,
)
from experiments.e32_interpbench.e32_5_scoring import (
    internal_endpoint_status,
    score_exact_group,
)
from experiments.e32_interpbench.e32_5_selector import (
    ExactGroupEvaluation,
    LocalSearchResult,
    SearchRound,
    SingletonPartitionScores,
    ValidationPathSelection,
    accept_exact_swap_batch,
    initial_static_support,
    propose_exact_swap_batch,
    select_validation_path_support,
    signed_singleton_partition_scores,
)
from experiments.e32_interpbench.eap_baseline import eap_query_accounting

DiscoverySplit = Literal["development", "discovery"]
ModelRole = Literal["regular_eap", "adaptive_patch", "eap_ig5"]


@dataclass(frozen=True)
class PipelineStageAccounting:
    """One stage normalized across patching and gradient acquisitions."""

    physical_forwards: int
    model_examples: int
    no_grad_forwards: int
    grad_enabled_forwards: int
    gradient_examples: int
    backward_passes_inferred_from_pinned_control_flow: int
    mask_context_count: int = 0
    active_mask_coordinate_checks: int = 0


@dataclass(frozen=True)
class DiscoveryQueryAccounting:
    behavioral_eligibility: PipelineStageAccounting
    adaptive_patch: PipelineStageAccounting
    regular_eap: PipelineStageAccounting
    eap_ig5: PipelineStageAccounting
    total: PipelineStageAccounting


@dataclass(frozen=True)
class E32_5DiscoveryBundle:
    """Raw D acquisition before any durable support authorization."""

    case_id: int
    split: DiscoverySplit
    eligible_panel: BehaviorallyEligiblePair
    search_eligible_panel: BehaviorallyEligiblePair
    universe: InducedEdgeUniverse
    adaptive: E32_5AdaptiveAcquisition
    regular_eap: GenericEAPRun
    eap_ig5: GenericEAPRun | None
    query_accounting: DiscoveryQueryAccounting


@dataclass(frozen=True)
class DiscoveryScoringSlices:
    """Shape-checked D train/validation inputs reconstructed from raw custody."""

    train_clean_output: torch.Tensor
    train_corrupt_output: torch.Tensor
    train_all_internal_output: torch.Tensor
    train_singleton_outputs: torch.Tensor
    train_leave_one_outputs: torch.Tensor
    train_clean_target_indices: torch.Tensor
    validation_clean_output: torch.Tensor
    validation_corrupt_output: torch.Tensor
    validation_all_internal_output: torch.Tensor
    validation_singleton_outputs: torch.Tensor
    validation_leave_one_outputs: torch.Tensor
    validation_clean_target_indices: torch.Tensor
    fine_edge_ids: tuple[str, ...]
    fine_to_whole_indices: tuple[int, ...]
    regular_eap_signed_scores: torch.Tensor
    eap_ig5_signed_scores: torch.Tensor | None


@dataclass(frozen=True)
class DiscoverySearchReplay:
    singleton_scores: SingletonPartitionScores
    search: LocalSearchResult
    validation: ValidationPathSelection


@dataclass(frozen=True)
class _ExactAcquisitionView:
    supports: tuple[tuple[str, ...], ...]
    selected_outputs: torch.Tensor
    complement_outputs: torch.Tensor


@dataclass(frozen=True)
class _DerivedSearchState:
    singleton_scores: SingletonPartitionScores
    initial_support: tuple[str, ...]
    current_evaluation: ExactGroupEvaluation
    accepted_path: tuple[tuple[str, ...], ...]
    accepted_joints: tuple[float, ...]
    round_records: tuple[SearchRound, ...]
    stop_reason: str | None


def _discovery_split(case_id: int, split: str) -> DiscoverySplit:
    try:
        expected = SPLIT_ORDER_BY_CASE[case_id][0]
    except KeyError as error:
        raise ValueError("case_id is outside the frozen E32.5 cohort") from error
    if split != expected or split not in {"development", "discovery"}:
        raise ValueError("D split differs from the frozen case lifecycle")
    return split


def _validate_discovery_eligible(
    eligible: BehaviorallyEligiblePair,
) -> None:
    if not isinstance(eligible, BehaviorallyEligiblePair):
        raise TypeError("eligible_panel must be BehaviorallyEligiblePair")
    pair = eligible.pair
    tensors = (
        pair.clean_inputs,
        pair.clean_targets,
        pair.corrupt_inputs,
        pair.corrupt_targets,
    )
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("eligible panel must contain tensor inputs and targets")
    if (
        len(pair.clean_inputs) != E32_5_D_BATCH_SIZE
        or len(pair.corrupt_inputs) != E32_5_D_BATCH_SIZE
        or len(pair.clean_targets) != E32_5_D_BATCH_SIZE
        or len(pair.corrupt_targets) != E32_5_D_BATCH_SIZE
        or len(eligible.selected_candidate_indices) != E32_5_D_BATCH_SIZE
    ):
        raise ValueError("E32.5 discovery requires one exact D64 eligible panel")
    if (
        pair.clean_inputs.ndim != 2
        or pair.corrupt_inputs.shape != pair.clean_inputs.shape
        or pair.clean_targets.ndim != 3
        or pair.corrupt_targets.shape != pair.clean_targets.shape
        or tuple(pair.clean_targets.shape[:2]) != tuple(pair.clean_inputs.shape)
    ):
        raise ValueError("eligible discovery inputs and categorical targets misalign")


def _search_eligible_slice(
    eligible: BehaviorallyEligiblePair,
) -> BehaviorallyEligiblePair:
    pair = eligible.pair
    stop = E32_5_TRAIN_ROW_COUNT
    search_pair = OfficialDataPair(
        clean_inputs=pair.clean_inputs[:stop].detach().clone(),
        clean_targets=pair.clean_targets[:stop].detach().clone(),
        corrupt_inputs=pair.corrupt_inputs[:stop].detach().clone(),
        corrupt_targets=pair.corrupt_targets[:stop].detach().clone(),
    )
    return replace(
        eligible,
        pair=search_pair,
        candidate_count=stop,
        eligible_candidate_count=stop,
        selected_candidate_indices=eligible.selected_candidate_indices[:stop],
        eligibility_physical_forwards=0,
        eligibility_model_examples=0,
        row_disjoint_eligible_candidate_count=min(
            eligible.row_disjoint_eligible_candidate_count, stop
        ),
    )


def _adaptive_stage(value: E32_5AdaptiveAcquisition) -> PipelineStageAccounting:
    accounting = value.query_accounting
    if not accounting.exact_expected_observed_match:
        raise RuntimeError("adaptive D accounting is not exact")
    return PipelineStageAccounting(
        physical_forwards=accounting.observed_physical_forwards,
        model_examples=accounting.observed_model_examples,
        no_grad_forwards=accounting.observed_no_grad_forwards,
        grad_enabled_forwards=accounting.observed_grad_enabled_forwards,
        gradient_examples=0,
        backward_passes_inferred_from_pinned_control_flow=(accounting.backward_passes),
        mask_context_count=accounting.mask_context_count,
        active_mask_coordinate_checks=accounting.active_mask_coordinate_checks,
    )


def _eap_stage(run: GenericEAPRun, *, method: str) -> PipelineStageAccounting:
    integrated = 5 if method == "eap_ig" else None
    expected = eap_query_accounting(
        example_count=E32_5_TRAIN_ROW_COUNT,
        method=method,  # type: ignore[arg-type]
        integrated_grad_samples=integrated,
    )
    observed = dict(run.observed_query_accounting)
    expected_observed = {
        "physical_forwards": expected.physical_model_forwards,
        "model_examples": expected.model_examples,
        "no_grad_forwards": expected.source_ablation_forwards,
        "grad_enabled_forwards": expected.gradient_forwards,
        "backward_passes_inferred_from_pinned_control_flow": expected.backward_passes,
    }
    if run.query_accounting != expected or observed != expected_observed:
        raise RuntimeError(
            f"{method} D accounting differs from the frozen search32 work"
        )
    return PipelineStageAccounting(
        physical_forwards=expected.physical_model_forwards,
        model_examples=expected.model_examples,
        no_grad_forwards=expected.source_ablation_forwards,
        grad_enabled_forwards=expected.gradient_forwards,
        gradient_examples=expected.gradient_examples,
        backward_passes_inferred_from_pinned_control_flow=(expected.backward_passes),
    )


def _sum_stages(*stages: PipelineStageAccounting) -> PipelineStageAccounting:
    fields = tuple(PipelineStageAccounting.__dataclass_fields__)
    return PipelineStageAccounting(
        **{
            field: sum(int(getattr(stage, field)) for stage in stages)
            for field in fields
        }
    )


def _validate_discovery_outputs(
    *,
    case_id: int,
    eligible: BehaviorallyEligiblePair,
    universe: InducedEdgeUniverse,
    adaptive: E32_5AdaptiveAcquisition,
    regular_eap: GenericEAPRun,
    eap_ig5: GenericEAPRun | None,
) -> DiscoveryQueryAccounting:
    if (
        universe.case_id != case_id
        or adaptive.universe != universe
        or adaptive.base.query_accounting.example_count != E32_5_D_BATCH_SIZE
    ):
        raise ValueError(
            "adaptive D output differs from the frozen case and D64 universe"
        )
    if (
        regular_eap.case_id != case_id
        or regular_eap.settings.method != "eap"
        or regular_eap.settings.integrated_grad_samples is not None
    ):
        raise ValueError("regular EAP identity changed")
    search_authorized = adaptive.internal_endpoint_status.get("search_authorized")
    if search_authorized is True:
        if (
            eap_ig5 is None
            or eap_ig5.case_id != case_id
            or eap_ig5.settings.method != "eap_ig"
            or eap_ig5.settings.integrated_grad_samples != 5
        ):
            raise ValueError("authorized D search requires exact EAP-IG5 identity")
    elif search_authorized is False:
        if eap_ig5 is not None:
            raise ValueError("scientific terminal must stop before EAP-IG5")
    else:
        raise ValueError("adaptive D lacks a boolean endpoint authorization")
    for run in (regular_eap, *((eap_ig5,) if eap_ig5 is not None else ())):
        if (
            run.internal_fine_edge_names != universe.fine_edge_names
            or run.fine_to_whole_indices != universe.fine_to_whole_indices
            or len(run.internal_fine_scores) != universe.fine_edge_count
        ):
            raise ValueError("EAP fine axis differs from the adaptive full-U axis")

    clean = adaptive.base.clean_baseline_output
    targets = eligible.pair.clean_targets
    if clean.ndim != 3 or tuple(targets.shape) != (
        E32_5_D_BATCH_SIZE,
        clean.shape[1] + 1,
        clean.shape[2],
    ):
        raise ValueError("adaptive outputs are not centered non-BOS [D64,n_ctx-1,C]")
    expected_endpoint_status = internal_endpoint_status(
        clean_output=clean[:E32_5_TRAIN_ROW_COUNT],
        all_internal_output=adaptive.base.all_internal_output[:E32_5_TRAIN_ROW_COUNT],
        clean_target_indices=targets.argmax(dim=-1)[:E32_5_TRAIN_ROW_COUNT, 1:],
    )
    if dict(adaptive.internal_endpoint_status) != expected_endpoint_status:
        raise ValueError(
            "adaptive endpoint assessment did not use only D search rows0:32"
        )
    eligibility = PipelineStageAccounting(
        physical_forwards=int(eligible.eligibility_physical_forwards),
        model_examples=int(eligible.eligibility_model_examples),
        no_grad_forwards=int(eligible.eligibility_physical_forwards),
        grad_enabled_forwards=0,
        gradient_examples=0,
        backward_passes_inferred_from_pinned_control_flow=0,
    )
    adaptive_stage = _adaptive_stage(adaptive)
    regular_stage = _eap_stage(regular_eap, method="eap")
    ig_stage = (
        _eap_stage(eap_ig5, method="eap_ig")
        if eap_ig5 is not None
        else PipelineStageAccounting(*(0,) * 6)
    )
    return DiscoveryQueryAccounting(
        behavioral_eligibility=eligibility,
        adaptive_patch=adaptive_stage,
        regular_eap=regular_stage,
        eap_ig5=ig_stage,
        total=_sum_stages(eligibility, adaptive_stage, regular_stage, ig_stage),
    )


def _full_u_from_regular_eap(run: GenericEAPRun) -> InducedEdgeUniverse:
    nodes, whole_edges = case_universe(run.case_id)
    if run.grouping.edge_order != whole_edges:
        raise ValueError("regular EAP whole-edge axis differs from canonical case U")
    universe = build_induced_edge_universe(
        run.grouping,
        selected_nodes=nodes,
    )
    try:
        expected_fine_count = CASE_GEOMETRY[run.case_id].full_fine_edge_count
    except KeyError as error:
        raise ValueError(
            "regular EAP case is outside the frozen E32.5 cohort"
        ) from error
    if (
        universe.induced_fine_edge_count != universe.fine_edge_count
        or universe.induced_whole_edge_count != universe.whole_edge_count
        or universe.fine_edge_count != expected_fine_count
    ):
        raise ValueError("regular EAP did not derive the exact registered full-U graph")
    return universe


def _support_names(
    universe: InducedEdgeUniverse, support: FrozenEdgeSupport
) -> tuple[str, ...]:
    if support.resolution != "fine":
        raise ValueError("E32.5 search supports must remain on the fine axis")
    return tuple(universe.fine_edge_names[index] for index in support.selected_indices)


def _live_acquisition_view(
    acquisition: Any,
    *,
    universe: InducedEdgeUniverse,
) -> _ExactAcquisitionView:
    if acquisition is None or acquisition.universe != universe:
        raise ValueError("live search acquisition differs from the full-U universe")
    return _ExactAcquisitionView(
        supports=tuple(
            _support_names(universe, support) for support in acquisition.plan.supports
        ),
        selected_outputs=acquisition.support_selected_outputs[
            :, :E32_5_TRAIN_ROW_COUNT
        ],
        complement_outputs=acquisition.support_complement_outputs[
            :, :E32_5_TRAIN_ROW_COUNT
        ],
    )


def _score_view_support(
    *,
    universe: InducedEdgeUniverse,
    clean_output: torch.Tensor,
    all_internal_output: torch.Tensor,
    clean_target_indices: torch.Tensor,
    view: _ExactAcquisitionView,
    support_index: int,
) -> ExactGroupEvaluation:
    return score_exact_group(
        fine_edge_ids=universe.fine_edge_names,
        support=view.supports[support_index],
        clean_output=clean_output,
        all_internal_output=all_internal_output,
        selected_output=view.selected_outputs[support_index],
        complement_output=view.complement_outputs[support_index],
        clean_target_indices=clean_target_indices,
    )


def _derive_search_state(
    *,
    universe: InducedEdgeUniverse,
    clean_output: torch.Tensor,
    all_internal_output: torch.Tensor,
    singleton_outputs: torch.Tensor,
    leave_one_outputs: torch.Tensor,
    clean_target_indices: torch.Tensor,
    initial_view: _ExactAcquisitionView,
    round_views: Sequence[_ExactAcquisitionView],
) -> _DerivedSearchState:
    scores = signed_singleton_partition_scores(
        clean_output=clean_output,
        all_internal_output=all_internal_output,
        singleton_outputs=singleton_outputs,
        leave_one_clean_outputs=leave_one_outputs,
    )
    support_size = confirmation_support_size(universe.case_id)
    initial_support = initial_static_support(
        fine_edge_ids=universe.fine_edge_names,
        directional_scores=scores.signed_score,
        support_size=support_size,
    )
    if initial_view.supports != (initial_support,):
        raise ValueError("initial live/raw support differs from shared selector output")
    current = _score_view_support(
        universe=universe,
        clean_output=clean_output,
        all_internal_output=all_internal_output,
        clean_target_indices=clean_target_indices,
        view=initial_view,
        support_index=0,
    )
    accepted_path = [initial_support]
    accepted_joints = [float(current.joint_faithfulness)]
    round_records: list[SearchRound] = []
    stop_reason: str | None = None
    for round_index, view in enumerate(round_views):
        if stop_reason is not None:
            raise ValueError("raw/live search acquired a round after selector stop")
        proposal = propose_exact_swap_batch(
            fine_edge_ids=universe.fine_edge_names,
            fine_to_group_indices=universe.fine_to_whole_indices,
            directional_scores=scores.signed_score,
            clean_output=clean_output,
            all_internal_output=all_internal_output,
            singleton_outputs=singleton_outputs,
            leave_one_clean_outputs=leave_one_outputs,
            current_evaluation=current,
            round_index=round_index,
        )
        if view.supports != proposal.supports:
            raise ValueError(
                "live/raw candidate sequence differs from shared selector proposal"
            )
        evaluations = {
            support: _score_view_support(
                universe=universe,
                clean_output=clean_output,
                all_internal_output=all_internal_output,
                clean_target_indices=clean_target_indices,
                view=view,
                support_index=index,
            )
            for index, support in enumerate(view.supports)
        }
        outcome = accept_exact_swap_batch(
            fine_edge_ids=universe.fine_edge_names,
            proposal=proposal,
            candidate_evaluations=evaluations,
        )
        if outcome.round_record is not None:
            round_records.append(outcome.round_record)
        if outcome.accepted_evaluation is None:
            stop_reason = outcome.stop_reason
        else:
            current = outcome.accepted_evaluation
            accepted_path.append(current.support)
            accepted_joints.append(float(current.joint_faithfulness))
    return _DerivedSearchState(
        singleton_scores=scores,
        initial_support=initial_support,
        current_evaluation=current,
        accepted_path=tuple(accepted_path),
        accepted_joints=tuple(accepted_joints),
        round_records=tuple(round_records),
        stop_reason=stop_reason,
    )


def _next_search_proposal(
    *,
    universe: InducedEdgeUniverse,
    clean_output: torch.Tensor,
    all_internal_output: torch.Tensor,
    singleton_outputs: torch.Tensor,
    leave_one_outputs: torch.Tensor,
    state: _DerivedSearchState,
    round_index: int,
):
    return propose_exact_swap_batch(
        fine_edge_ids=universe.fine_edge_names,
        fine_to_group_indices=universe.fine_to_whole_indices,
        directional_scores=state.singleton_scores.signed_score,
        clean_output=clean_output,
        all_internal_output=all_internal_output,
        singleton_outputs=singleton_outputs,
        leave_one_clean_outputs=leave_one_outputs,
        current_evaluation=state.current_evaluation,
        round_index=round_index,
    )


class E32_5SearchPlanner:
    """Deferred acquisition adapter using only shared pure selector primitives."""

    def __init__(
        self,
        *,
        universe: InducedEdgeUniverse,
        clean_target_indices: Any,
    ) -> None:
        if not isinstance(universe, InducedEdgeUniverse):
            raise TypeError("universe must be InducedEdgeUniverse")
        targets = torch.as_tensor(clean_target_indices).detach().cpu().clone()
        if (
            targets.ndim != 2
            or targets.shape[0] != E32_5_D_BATCH_SIZE
            or targets.dtype == torch.bool
            or targets.dtype.is_floating_point
        ):
            raise ValueError("planner targets must be integer non-BOS [D64,position]")
        self.universe = universe
        self.clean_target_indices = targets.to(torch.long)

    def _train_base(self, base: Any) -> tuple[torch.Tensor, ...]:
        if base.universe != self.universe:
            raise ValueError("planner base differs from the frozen full-U universe")
        rows = slice(0, E32_5_TRAIN_ROW_COUNT)
        if tuple(base.clean_baseline_output.shape[:2]) != tuple(
            self.clean_target_indices.shape
        ):
            raise ValueError("planner targets differ from adaptive non-BOS outputs")
        return (
            base.clean_baseline_output[rows],
            base.all_internal_output[rows],
            base.support_selected_outputs[:, rows],
            base.support_complement_outputs[:, rows],
            self.clean_target_indices[rows],
        )

    def assess_internal_endpoint(self, base: Any) -> Mapping[str, Any]:
        clean, all_internal, _singleton, _leave_one, targets = self._train_base(base)
        return internal_endpoint_status(
            clean_output=clean,
            all_internal_output=all_internal,
            clean_target_indices=targets,
        )

    def plan_initial_support(self, base: Any):
        clean, all_internal, singleton, leave_one, _targets = self._train_base(base)
        scores = signed_singleton_partition_scores(
            clean_output=clean,
            all_internal_output=all_internal,
            singleton_outputs=singleton,
            leave_one_clean_outputs=leave_one,
        )
        names = initial_static_support(
            fine_edge_ids=self.universe.fine_edge_names,
            directional_scores=scores.signed_score,
            support_size=confirmation_support_size(self.universe.case_id),
        )
        return build_edge_condition_plan(
            self.universe,
            (self._freeze_support("search:initial", names),),
        )

    def _freeze_support(
        self, support_id: str, names: Sequence[str]
    ) -> FrozenEdgeSupport:
        selected = frozenset(names)
        indices = tuple(
            index
            for index, name in enumerate(self.universe.fine_edge_names)
            if name in selected
        )
        if len(indices) != len(tuple(names)):
            raise ValueError("selector support escaped the canonical full-U axis")
        return FrozenEdgeSupport(
            support_id=support_id,
            resolution="fine",
            method=PRIMARY_METHOD,
            support_size=len(indices),
            selected_indices=indices,
            kind="deterministic",
        )

    def plan_next_round(self, base: Any, initial: Any, rounds: tuple[Any, ...]):
        clean, all_internal, singleton, leave_one, targets = self._train_base(base)
        state = _derive_search_state(
            universe=self.universe,
            clean_output=clean,
            all_internal_output=all_internal,
            singleton_outputs=singleton,
            leave_one_outputs=leave_one,
            clean_target_indices=targets,
            initial_view=_live_acquisition_view(initial, universe=self.universe),
            round_views=tuple(
                _live_acquisition_view(round_acquisition, universe=self.universe)
                for round_acquisition in rounds
            ),
        )
        if state.stop_reason is not None or len(rounds) >= LOCAL_SEARCH_ROUNDS:
            return None
        proposal = _next_search_proposal(
            universe=self.universe,
            clean_output=clean,
            all_internal_output=all_internal,
            singleton_outputs=singleton,
            leave_one_outputs=leave_one,
            state=state,
            round_index=len(rounds),
        )
        if not proposal.supports:
            outcome = accept_exact_swap_batch(
                fine_edge_ids=self.universe.fine_edge_names,
                proposal=proposal,
                candidate_evaluations={},
            )
            if outcome.stop_reason != "no_swap_candidates":
                raise RuntimeError("empty shared proposal did not stop canonically")
            return None
        supports = tuple(
            self._freeze_support(
                f"search:round:{len(rounds)}:candidate:{index:02d}", names
            )
            for index, names in enumerate(proposal.supports)
        )
        return build_edge_condition_plan(self.universe, supports)


def run_e32_5_live_pipeline(
    *,
    case_id: int,
    split: str,
    eligible_panel: BehaviorallyEligiblePair,
    model_factory: Callable[[ModelRole], Any],
    adaptive_acquirer: Callable[
        [
            Any,
            BehaviorallyEligiblePair,
            InducedEdgeUniverse,
            E32_5SearchPlanner,
        ],
        E32_5AdaptiveAcquisition,
    ],
    eap_acquirer: Callable[[Any, BehaviorallyEligiblePair, str], GenericEAPRun],
) -> E32_5DiscoveryBundle:
    """Run the D-only composition; this function never opens confirmation."""
    frozen_split = _discovery_split(case_id, split)
    _validate_discovery_eligible(eligible_panel)
    if not all(
        callable(value) for value in (model_factory, adaptive_acquirer, eap_acquirer)
    ):
        raise TypeError("pipeline acquisition dependencies must be callable")

    roles: tuple[ModelRole, ...] = ("regular_eap", "adaptive_patch", "eap_ig5")
    search_eligible = _search_eligible_slice(eligible_panel)
    live_models: list[Any] = []

    def fresh(role: ModelRole) -> Any:
        model = model_factory(role)
        if model is None or any(model is prior for prior in live_models):
            raise ValueError("each D method requires a fresh distinct model")
        live_models.append(model)
        return model

    regular_eap = eap_acquirer(fresh(roles[0]), search_eligible, "eap")
    if regular_eap.case_id != case_id or regular_eap.settings.method != "eap":
        raise ValueError("regular EAP identity changed before full-U derivation")
    universe = _full_u_from_regular_eap(regular_eap)
    planner = E32_5SearchPlanner(
        universe=universe,
        clean_target_indices=eligible_panel.pair.clean_targets.argmax(dim=-1)[:, 1:],
    )
    adaptive = adaptive_acquirer(fresh(roles[1]), eligible_panel, universe, planner)
    eap_ig5 = (
        eap_acquirer(fresh(roles[2]), search_eligible, "eap_ig")
        if adaptive.internal_endpoint_status.get("search_authorized") is True
        else None
    )
    accounting = _validate_discovery_outputs(
        case_id=case_id,
        eligible=eligible_panel,
        universe=universe,
        adaptive=adaptive,
        regular_eap=regular_eap,
        eap_ig5=eap_ig5,
    )
    return E32_5DiscoveryBundle(
        case_id=case_id,
        split=frozen_split,
        eligible_panel=eligible_panel,
        search_eligible_panel=search_eligible,
        universe=universe,
        adaptive=adaptive,
        regular_eap=regular_eap,
        eap_ig5=eap_ig5,
        query_accounting=accounting,
    )


def _adaptive_to_dict(value: E32_5AdaptiveAcquisition) -> dict[str, Any]:
    return {
        "schema_version": "e32_5_adaptive_acquisition_v1",
        "base": edge_set_acquisition_to_dict(value.base),
        "initial": (
            None
            if value.initial is None
            else edge_set_acquisition_to_dict(value.initial)
        ),
        "rounds": tuple(
            edge_set_acquisition_to_dict(acquisition) for acquisition in value.rounds
        ),
        "stop_reason": value.stop_reason,
        "internal_endpoint_status": dict(value.internal_endpoint_status),
        "query_accounting": asdict(value.query_accounting),
    }


def _cpu_clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def _selected_token_rows(pair: OfficialDataPair) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(token) for token in row)
        for tensor in (pair.clean_inputs, pair.corrupt_inputs)
        for row in tensor.detach().cpu().tolist()
    )


def serialize_e32_5_discovery_bundle(
    bundle: E32_5DiscoveryBundle,
    *,
    split: str | None = None,
) -> dict[str, Any]:
    """Return a torch-save-ready raw D envelope with no trusted final scores."""
    if not isinstance(bundle, E32_5DiscoveryBundle):
        raise TypeError("bundle must be E32_5DiscoveryBundle")
    if split is not None and split != bundle.split:
        raise ValueError("requested serialization split differs from D bundle")
    pair = bundle.eligible_panel.pair
    target_indices = pair.clean_targets.argmax(dim=-1)[:, 1:]
    return {
        "schema_version": "e32_5_discovery_raw_bundle_v1",
        "case_id": bundle.case_id,
        "split": bundle.split,
        "selected_token_rows": _selected_token_rows(pair),
        "query_accounting": asdict(bundle.query_accounting),
        "scientific_payload": {
            "panel": {
                "clean_inputs": _cpu_clone(pair.clean_inputs),
                "clean_targets": _cpu_clone(pair.clean_targets),
                "corrupt_inputs": _cpu_clone(pair.corrupt_inputs),
                "corrupt_targets": _cpu_clone(pair.corrupt_targets),
                "selected_candidate_indices": (
                    bundle.eligible_panel.selected_candidate_indices
                ),
                "eligibility": {
                    "candidate_count": bundle.eligible_panel.candidate_count,
                    "eligible_candidate_count": (
                        bundle.eligible_panel.eligible_candidate_count
                    ),
                    "clean_incorrect_contexts": (
                        bundle.eligible_panel.clean_incorrect_contexts
                    ),
                    "source_incorrect_contexts": (
                        bundle.eligible_panel.source_incorrect_contexts
                    ),
                    "either_incorrect_contexts": (
                        bundle.eligible_panel.either_incorrect_contexts
                    ),
                    "eligibility_physical_forwards": (
                        bundle.eligible_panel.eligibility_physical_forwards
                    ),
                    "eligibility_model_examples": (
                        bundle.eligible_panel.eligibility_model_examples
                    ),
                    "output_kind": bundle.eligible_panel.output_kind,
                    "row_disjoint_eligible_candidate_count": (
                        bundle.eligible_panel.row_disjoint_eligible_candidate_count
                    ),
                    "row_overlap_excluded_contexts": (
                        bundle.eligible_panel.row_overlap_excluded_contexts
                    ),
                    "excluded_token_row_count": (
                        bundle.eligible_panel.excluded_token_row_count
                    ),
                    "selection_policy": bundle.eligible_panel.selection_policy,
                },
                "search_selected_candidate_indices": (
                    bundle.search_eligible_panel.selected_candidate_indices
                ),
                "search_row_slice": (0, E32_5_TRAIN_ROW_COUNT),
                "validation_row_slice": (
                    E32_5_TRAIN_ROW_COUNT,
                    E32_5_TRAIN_ROW_COUNT + VALIDATION_EXAMPLE_COUNT,
                ),
            },
            "clean_target_indices": _cpu_clone(target_indices),
            "adaptive": _adaptive_to_dict(bundle.adaptive),
            "regular_eap": generic_eap_run_to_dict(bundle.regular_eap),
            "eap_ig5": (
                None
                if bundle.eap_ig5 is None
                else generic_eap_run_to_dict(bundle.eap_ig5)
            ),
        },
    }


def _raw_tensor(record: Mapping[str, Any], name: str) -> torch.Tensor:
    value = record.get(name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"raw discovery {name} must be a tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"raw discovery {name} contains nonfinite values")
    return value.detach().cpu()


def _universe_from_record(value: Any) -> InducedEdgeUniverse:
    if not isinstance(value, Mapping):
        raise TypeError("raw acquisition universe must be a mapping")
    try:
        return InducedEdgeUniverse(
            case_id=value["case_id"],
            selected_nodes=tuple(value["selected_nodes"]),
            whole_edges=tuple(tuple(edge) for edge in value["whole_edges"]),
            fine_edge_names=tuple(value["fine_edge_names"]),
            fine_to_whole_indices=tuple(value["fine_to_whole_indices"]),
            induced_whole_indices=tuple(value["induced_whole_indices"]),
            induced_fine_indices=tuple(value["induced_fine_indices"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("raw acquisition universe is malformed") from error


def _validated_raw_edge_acquisition(
    record: Mapping[str, Any],
    *,
    universe: InducedEdgeUniverse,
    expected_batch: int,
    label: str,
) -> dict[str, Any]:
    if (
        record.get("schema_version") != "e32_4_edge_set_acquisition_v1"
        or record.get("case_id") != universe.case_id
        or record.get("ablation") != "resample"
        or _universe_from_record(record.get("universe")) != universe
    ):
        raise ValueError(f"raw {label} identity or full-U universe changed")
    raw_plan = record.get("plan")
    if not isinstance(raw_plan, Mapping):
        raise TypeError(f"raw {label} plan must be a mapping")
    try:
        supports = tuple(
            FrozenEdgeSupport(**dict(support)) for support in raw_plan["supports"]
        )
        rebuilt = build_edge_condition_plan(
            universe,
            supports,
            primary_support_id=raw_plan["primary_support_id"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"raw {label} support plan is malformed") from error
    raw_masks = _raw_tensor(raw_plan, "fine_masks")
    expected_references = tuple(asdict(value) for value in rebuilt.support_references)
    if (
        raw_masks.dtype != torch.bool
        or not torch.equal(raw_masks, rebuilt.fine_masks)
        or tuple(raw_plan.get("condition_ids", ())) != rebuilt.condition_ids
        or raw_plan.get("all_internal_condition_index")
        != rebuilt.all_internal_condition_index
        or raw_plan.get("induced_condition_index") != rebuilt.induced_condition_index
        or raw_plan.get("outside_induced_condition_index")
        != rebuilt.outside_induced_condition_index
        or raw_plan.get("primary_within_induced_complement_condition_index")
        != rebuilt.primary_within_induced_complement_condition_index
        or tuple(raw_plan.get("support_references", ())) != expected_references
        or raw_plan.get("empty_mask_reuses_clean_baseline") is not True
    ):
        raise ValueError(f"raw {label} differs from its canonical full-U plan")

    clean = _raw_tensor(record, "clean_baseline_output")
    corrupt = _raw_tensor(record, "corrupt_baseline_output")
    conditions = _raw_tensor(record, "condition_outputs")
    selected = _raw_tensor(record, "support_selected_outputs")
    complements = _raw_tensor(record, "support_complement_outputs")
    if clean.ndim != 3 or tuple(clean.shape[:1]) != (expected_batch,):
        raise ValueError(f"raw {label} baseline output shape changed")
    output_shape = tuple(clean.shape)
    if (
        tuple(corrupt.shape) != output_shape
        or tuple(conditions.shape) != (rebuilt.condition_count, *output_shape)
        or tuple(selected.shape) != (len(supports), *output_shape)
        or tuple(complements.shape) != tuple(selected.shape)
    ):
        raise ValueError(f"raw {label} condition/support output axes changed")

    def referenced(index: int | None) -> torch.Tensor:
        return clean if index is None else conditions[index]

    anchors = {
        "all_internal_output": referenced(rebuilt.all_internal_condition_index),
        "induced_output": referenced(rebuilt.induced_condition_index),
        "outside_induced_output": referenced(rebuilt.outside_induced_condition_index),
    }
    if any(
        not torch.equal(_raw_tensor(record, name), output)
        for name, output in anchors.items()
    ):
        raise ValueError(
            f"raw {label} anchor aliases differ from canonical plan references"
        )
    for support_index, reference in enumerate(rebuilt.support_references):
        if not torch.equal(
            selected[support_index], referenced(reference.selected_condition_index)
        ) or not torch.equal(
            complements[support_index],
            referenced(reference.complement_condition_index),
        ):
            raise ValueError(
                f"raw {label} support aliases differ from canonical plan references"
            )
    primary_output = record.get("primary_within_induced_complement_output")
    primary_index = rebuilt.primary_within_induced_complement_condition_index
    expected_primary = None if primary_index is None else referenced(primary_index)
    if (primary_output is None) != (expected_primary is None) or (
        expected_primary is not None
        and not torch.equal(torch.as_tensor(primary_output), expected_primary)
    ):
        raise ValueError(f"raw {label} primary complement alias changed")

    for output in (clean, corrupt, conditions):
        if not torch.allclose(
            output.sum(dim=-1),
            torch.zeros_like(output[..., 0]),
            rtol=0.0,
            atol=1e-5,
        ):
            raise ValueError(f"raw {label} outputs are not class-centered")

    accounting = record.get("query_accounting")
    forwards = 3 + rebuilt.condition_count
    expected_accounting = {
        "example_count": expected_batch,
        "condition_count": rebuilt.condition_count,
        "source_cache_forwards": 1,
        "baseline_forwards": 2,
        "patched_forwards": rebuilt.condition_count,
        "mask_context_count": rebuilt.condition_count,
        "active_mask_coordinate_checks": int(rebuilt.fine_masks.sum()),
        "expected_physical_forwards": forwards,
        "expected_model_examples": expected_batch * forwards,
        "observed_physical_forwards": forwards,
        "observed_model_examples": expected_batch * forwards,
        "observed_no_grad_forwards": forwards,
        "observed_grad_enabled_forwards": 0,
        "backward_passes": 0,
    }
    if not isinstance(accounting, Mapping) or dict(accounting) != expected_accounting:
        raise ValueError(f"raw {label} query accounting changed")
    return {
        "plan": rebuilt,
        "supports": supports,
        "clean": clean,
        "corrupt": corrupt,
        "conditions": conditions,
        "selected": selected,
        "complements": complements,
        "query_accounting": expected_accounting,
        **anchors,
    }


def _rebuild_internal_eap_scores(
    record: Any,
    *,
    universe: InducedEdgeUniverse,
    method: str,
) -> torch.Tensor:
    if not isinstance(record, Mapping) or record.get("case_id") != universe.case_id:
        raise ValueError(f"raw {method} identity changed")
    settings = record.get("settings")
    expected_samples = 5 if method == "eap_ig" else None
    if (
        not isinstance(settings, Mapping)
        or settings.get("method") != method
        or settings.get("integrated_grad_samples") != expected_samples
        or tuple(record.get("edge_order", ())) != universe.whole_edges
        or tuple(record.get("internal_fine_edge_names", ())) != universe.fine_edge_names
        or tuple(record.get("fine_to_whole_indices", ()))
        != universe.fine_to_whole_indices
    ):
        raise ValueError(f"raw {method} settings or full-U axes changed")
    raw_all = tuple(record.get("all_fine_scores", ()))
    try:
        score_by_name = {
            str(score["fine_edge_name"]): float(score["signed_score"])
            for score in raw_all
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"raw {method} all-fine scores are malformed") from error
    if len(score_by_name) != len(raw_all) or not set(universe.fine_edge_names).issubset(
        score_by_name
    ):
        raise ValueError(f"raw {method} all-fine score names changed")
    rebuilt = tuple(score_by_name[name] for name in universe.fine_edge_names)
    stored = tuple(float(value) for value in record.get("internal_fine_scores", ()))
    if rebuilt != stored or not bool(torch.isfinite(torch.tensor(rebuilt)).all()):
        raise ValueError(
            f"raw {method} internal vector differs from raw all-fine scores"
        )
    expected_accounting = eap_query_accounting(
        example_count=E32_5_TRAIN_ROW_COUNT,
        method=method,  # type: ignore[arg-type]
        integrated_grad_samples=expected_samples,
    )
    expected_observed = {
        "physical_forwards": expected_accounting.physical_model_forwards,
        "model_examples": expected_accounting.model_examples,
        "no_grad_forwards": expected_accounting.source_ablation_forwards,
        "grad_enabled_forwards": expected_accounting.gradient_forwards,
        "backward_passes_inferred_from_pinned_control_flow": (
            expected_accounting.backward_passes
        ),
    }
    if (
        record.get("query_accounting") != asdict(expected_accounting)
        or record.get("observed_query_accounting") != expected_observed
    ):
        raise ValueError(f"raw {method} query accounting changed")
    return torch.tensor(rebuilt, dtype=torch.float64)


def discovery_scoring_slices(
    record: Mapping[str, Any],
) -> DiscoveryScoringSlices:
    """Rebuild scorer inputs and reject target/output slice drift or tampering."""
    if not isinstance(record, Mapping):
        raise TypeError("raw discovery must be a mapping")
    if record.get("schema_version") == "e32_5_raw_acquisition_v1":
        accounting = record.get("query_accounting")
        if not isinstance(accounting, Mapping) or tuple(accounting) != (
            "families",
            "total",
        ):
            raise ValueError("normalized raw discovery accounting changed")
        families = accounting.get("families")
        if not isinstance(families, Mapping):
            raise TypeError("normalized raw discovery families must be a mapping")
        record = {
            **dict(record),
            "schema_version": "e32_5_discovery_raw_bundle_v1",
            "query_accounting": {**dict(families), "total": accounting["total"]},
        }
    if record.get("schema_version") != "e32_5_discovery_raw_bundle_v1":
        raise ValueError("raw discovery schema changed")
    payload = record.get("scientific_payload")
    if not isinstance(payload, Mapping):
        raise TypeError("raw discovery lacks a scientific payload")
    panel = payload.get("panel")
    adaptive = payload.get("adaptive")
    if not isinstance(panel, Mapping) or not isinstance(adaptive, Mapping):
        raise TypeError("raw discovery panel or adaptive payload is malformed")
    base = adaptive.get("base")
    if not isinstance(base, Mapping):
        raise TypeError("raw discovery lacks its base partition atlas")

    case_id = record.get("case_id")
    split = record.get("split")
    if isinstance(case_id, bool) or not isinstance(case_id, int):
        raise TypeError("raw discovery case_id must be an integer")
    frozen_split = _discovery_split(case_id, str(split))
    universe = _universe_from_record(base.get("universe"))
    canonical_nodes, canonical_whole = case_universe(case_id)
    if (
        universe.case_id != case_id
        or universe.selected_nodes != canonical_nodes
        or universe.whole_edges != canonical_whole
        or universe.induced_fine_edge_count != universe.fine_edge_count
        or universe.induced_whole_edge_count != universe.whole_edge_count
        or universe.fine_edge_count != CASE_GEOMETRY[case_id].full_fine_edge_count
    ):
        raise ValueError("raw discovery base universe differs from canonical full U")

    clean_inputs = _raw_tensor(panel, "clean_inputs")
    corrupt_inputs = _raw_tensor(panel, "corrupt_inputs")
    clean_targets = _raw_tensor(panel, "clean_targets")
    corrupt_targets = _raw_tensor(panel, "corrupt_targets")
    clean_target_indices = _raw_tensor(payload, "clean_target_indices")
    expected_targets = clean_targets.argmax(dim=-1)[:, 1:]
    expected_rows = tuple(
        tuple(int(token) for token in row)
        for tensor in (clean_inputs, corrupt_inputs)
        for row in tensor.tolist()
    )
    selected_indices = tuple(panel.get("selected_candidate_indices", ()))
    eligibility = panel.get("eligibility")
    if not isinstance(eligibility, Mapping):
        raise TypeError("raw discovery lacks behavioral eligibility custody")
    eligibility_integer_fields = (
        "candidate_count",
        "eligible_candidate_count",
        "clean_incorrect_contexts",
        "source_incorrect_contexts",
        "either_incorrect_contexts",
        "eligibility_physical_forwards",
        "eligibility_model_examples",
        "row_disjoint_eligible_candidate_count",
        "row_overlap_excluded_contexts",
        "excluded_token_row_count",
    )
    if any(
        isinstance(eligibility.get(name), bool)
        or not isinstance(eligibility.get(name), int)
        or int(eligibility[name]) < 0
        for name in eligibility_integer_fields
    ):
        raise ValueError("raw discovery eligibility counts are malformed")
    candidate_count = int(eligibility["candidate_count"])
    eligible_count = int(eligibility["eligible_candidate_count"])
    row_disjoint_count = int(eligibility["row_disjoint_eligible_candidate_count"])
    if (
        clean_inputs.ndim != 2
        or tuple(corrupt_inputs.shape) != tuple(clean_inputs.shape)
        or tuple(clean_inputs.shape[:1]) != (E32_5_D_BATCH_SIZE,)
        or tuple(clean_targets.shape[:2]) != tuple(clean_inputs.shape)
        or tuple(corrupt_targets.shape) != tuple(clean_targets.shape)
        or tuple(record.get("selected_token_rows", ())) != expected_rows
        or len(selected_indices) != E32_5_D_BATCH_SIZE
        or len(set(selected_indices)) != E32_5_D_BATCH_SIZE
        or any(index < 0 or index >= candidate_count for index in selected_indices)
        or eligible_count > candidate_count
        or int(eligibility["either_incorrect_contexts"])
        != candidate_count - eligible_count
        or int(eligibility["clean_incorrect_contexts"])
        > int(eligibility["either_incorrect_contexts"])
        or int(eligibility["source_incorrect_contexts"])
        > int(eligibility["either_incorrect_contexts"])
        or row_disjoint_count < E32_5_D_BATCH_SIZE
        or int(eligibility["eligibility_physical_forwards"]) != 2
        or int(eligibility["eligibility_model_examples"]) != 2 * candidate_count
        or eligibility.get("output_kind") != "categorical"
        or eligibility.get("selection_policy")
        != split_request(case_id, frozen_split).selection_policy
        or tuple(panel.get("search_selected_candidate_indices", ()))
        != selected_indices[:E32_5_TRAIN_ROW_COUNT]
    ):
        raise ValueError(
            "raw discovery selected token rows or eligibility identity changed"
        )
    if panel.get("search_row_slice") != (0, E32_5_TRAIN_ROW_COUNT) or panel.get(
        "validation_row_slice"
    ) != (E32_5_TRAIN_ROW_COUNT, E32_5_D_BATCH_SIZE):
        raise ValueError("raw discovery train/validation slice metadata changed")

    base_outputs = _validated_raw_edge_acquisition(
        base,
        universe=universe,
        expected_batch=E32_5_D_BATCH_SIZE,
        label="base atlas",
    )
    expected_singletons = tuple((index,) for index in range(universe.fine_edge_count))
    if (
        tuple(support.selected_indices for support in base_outputs["supports"])
        != expected_singletons
        or base_outputs["plan"].condition_count != 2 * universe.fine_edge_count + 1
    ):
        raise ValueError("raw discovery singleton axis is not the complete full U")
    clean = base_outputs["clean"]
    corrupt = base_outputs["corrupt"]
    all_internal = base_outputs["all_internal_output"]
    singleton = base_outputs["selected"]
    leave_one = base_outputs["complements"]
    output_shape = (E32_5_D_BATCH_SIZE, *tuple(clean.shape[1:]))
    if (
        clean.ndim != 3
        or tuple(clean.shape) != output_shape
        or tuple(corrupt.shape) != output_shape
        or tuple(all_internal.shape) != output_shape
        or singleton.ndim != 4
        or tuple(singleton.shape[1:]) != output_shape
        or tuple(leave_one.shape) != tuple(singleton.shape)
        or tuple(expected_targets.shape) != output_shape[:2]
        or tuple(clean_target_indices.shape) != output_shape[:2]
        or not torch.equal(clean_target_indices, expected_targets)
    ):
        raise ValueError(
            "raw discovery scorer outputs and clean non-BOS target indices misalign"
        )
    expected_endpoint_status = internal_endpoint_status(
        clean_output=clean[:E32_5_TRAIN_ROW_COUNT],
        all_internal_output=all_internal[:E32_5_TRAIN_ROW_COUNT],
        clean_target_indices=clean_target_indices[:E32_5_TRAIN_ROW_COUNT],
    )
    if dict(adaptive.get("internal_endpoint_status", {})) != expected_endpoint_status:
        raise ValueError("raw adaptive endpoint status differs from D search rows0:32")

    initial_record = adaptive.get("initial")
    terminal = adaptive.get("stop_reason") == (
        "scientific_fail_uninformative_internal_endpoint"
    )
    if terminal and (initial_record is not None or tuple(adaptive.get("rounds", ()))):
        raise ValueError("scientific terminal raw D unexpectedly contains search calls")
    if not terminal and not isinstance(initial_record, Mapping):
        raise TypeError("authorized raw D lacks its initial exact support")
    acquisition_names: list[str | int] = []
    validated_adaptive_acquisitions = [base_outputs]
    if initial_record is not None:
        acquisition_names.append("initial")
    acquisition_names.extend(range(len(tuple(adaptive.get("rounds", ())))))
    for acquisition_name in acquisition_names:
        if acquisition_name == "initial":
            acquisition = adaptive.get("initial")
        else:
            acquisition = tuple(adaptive["rounds"])[acquisition_name]
        if not isinstance(acquisition, Mapping):
            raise TypeError("raw adaptive acquisition is malformed")
        candidate = _validated_raw_edge_acquisition(
            acquisition,
            universe=universe,
            expected_batch=E32_5_D_BATCH_SIZE,
            label=f"adaptive {acquisition_name}",
        )
        validated_adaptive_acquisitions.append(candidate)
        if (
            not torch.equal(candidate["clean"], clean)
            or not torch.equal(candidate["corrupt"], corrupt)
            or not torch.equal(candidate["all_internal_output"], all_internal)
        ):
            raise ValueError(
                "raw adaptive acquisitions changed clean/corrupt/F(U) baselines"
            )

    regular_eap = _rebuild_internal_eap_scores(
        payload.get("regular_eap"), universe=universe, method="eap"
    )
    if terminal:
        if payload.get("eap_ig5") is not None:
            raise ValueError("scientific terminal raw D unexpectedly contains EAP-IG5")
        eap_ig5 = None
    else:
        eap_ig5 = _rebuild_internal_eap_scores(
            payload.get("eap_ig5"), universe=universe, method="eap_ig"
        )

    raw_accounting = record.get("query_accounting")
    stage_names = (
        "behavioral_eligibility",
        "adaptive_patch",
        "regular_eap",
        "eap_ig5",
    )
    accounting_fields = tuple(PipelineStageAccounting.__dataclass_fields__)
    if not isinstance(raw_accounting, Mapping) or tuple(raw_accounting) != (
        *stage_names,
        "total",
    ):
        raise ValueError("raw discovery accounting sections changed")

    acquisition_records = tuple(
        value["query_accounting"] for value in validated_adaptive_acquisitions
    )

    def acquired_total(field: str) -> int:
        return sum(int(value[field]) for value in acquisition_records)

    expected_adaptive_accounting = {
        "acquisition_count": len(acquisition_records),
        "example_count": E32_5_D_BATCH_SIZE,
        **{
            field: acquired_total(field)
            for field in (
                "condition_count",
                "source_cache_forwards",
                "baseline_forwards",
                "patched_forwards",
                "mask_context_count",
                "active_mask_coordinate_checks",
                "expected_physical_forwards",
                "expected_model_examples",
                "observed_physical_forwards",
                "observed_model_examples",
                "observed_no_grad_forwards",
                "observed_grad_enabled_forwards",
                "backward_passes",
            )
        },
    }
    if dict(adaptive.get("query_accounting", {})) != expected_adaptive_accounting:
        raise ValueError("raw adaptive accounting differs from acquired plans")

    eligibility_stage = PipelineStageAccounting(
        physical_forwards=int(eligibility["eligibility_physical_forwards"]),
        model_examples=int(eligibility["eligibility_model_examples"]),
        no_grad_forwards=int(eligibility["eligibility_physical_forwards"]),
        grad_enabled_forwards=0,
        gradient_examples=0,
        backward_passes_inferred_from_pinned_control_flow=0,
    )
    adaptive_stage = PipelineStageAccounting(
        physical_forwards=expected_adaptive_accounting["observed_physical_forwards"],
        model_examples=expected_adaptive_accounting["observed_model_examples"],
        no_grad_forwards=expected_adaptive_accounting["observed_no_grad_forwards"],
        grad_enabled_forwards=expected_adaptive_accounting[
            "observed_grad_enabled_forwards"
        ],
        gradient_examples=0,
        backward_passes_inferred_from_pinned_control_flow=(
            expected_adaptive_accounting["backward_passes"]
        ),
        mask_context_count=expected_adaptive_accounting["mask_context_count"],
        active_mask_coordinate_checks=expected_adaptive_accounting[
            "active_mask_coordinate_checks"
        ],
    )

    def expected_eap_stage(method: str) -> PipelineStageAccounting:
        integrated = 5 if method == "eap_ig" else None
        value = eap_query_accounting(
            example_count=E32_5_TRAIN_ROW_COUNT,
            method=method,  # type: ignore[arg-type]
            integrated_grad_samples=integrated,
        )
        return PipelineStageAccounting(
            physical_forwards=value.physical_model_forwards,
            model_examples=value.model_examples,
            no_grad_forwards=value.source_ablation_forwards,
            grad_enabled_forwards=value.gradient_forwards,
            gradient_examples=value.gradient_examples,
            backward_passes_inferred_from_pinned_control_flow=value.backward_passes,
        )

    regular_stage = expected_eap_stage("eap")
    ig_stage = (
        PipelineStageAccounting(*(0,) * 6) if terminal else expected_eap_stage("eap_ig")
    )
    expected_stages = {
        "behavioral_eligibility": eligibility_stage,
        "adaptive_patch": adaptive_stage,
        "regular_eap": regular_stage,
        "eap_ig5": ig_stage,
    }
    for name, expected in expected_stages.items():
        if dict(raw_accounting.get(name, {})) != asdict(expected):
            raise ValueError(f"raw discovery {name} accounting differs from raw work")
    expected_total = _sum_stages(*expected_stages.values())
    if dict(raw_accounting.get("total", {})) != asdict(expected_total):
        raise ValueError("raw discovery accounting total differs from raw work")
    if tuple(raw_accounting["total"]) != accounting_fields:
        raise ValueError("raw discovery accounting field order changed")

    train = slice(0, E32_5_TRAIN_ROW_COUNT)
    validation = slice(E32_5_TRAIN_ROW_COUNT, E32_5_D_BATCH_SIZE)
    return DiscoveryScoringSlices(
        train_clean_output=clean[train],
        train_corrupt_output=corrupt[train],
        train_all_internal_output=all_internal[train],
        train_singleton_outputs=singleton[:, train],
        train_leave_one_outputs=leave_one[:, train],
        train_clean_target_indices=clean_target_indices[train],
        validation_clean_output=clean[validation],
        validation_corrupt_output=corrupt[validation],
        validation_all_internal_output=all_internal[validation],
        validation_singleton_outputs=singleton[:, validation],
        validation_leave_one_outputs=leave_one[:, validation],
        validation_clean_target_indices=clean_target_indices[validation],
        fine_edge_ids=universe.fine_edge_names,
        fine_to_whole_indices=universe.fine_to_whole_indices,
        regular_eap_signed_scores=regular_eap,
        eap_ig5_signed_scores=eap_ig5,
    )


def _raw_acquisition_view(
    record: Mapping[str, Any],
    *,
    universe: InducedEdgeUniverse,
    label: str,
) -> _ExactAcquisitionView:
    validated = _validated_raw_edge_acquisition(
        record,
        universe=universe,
        expected_batch=E32_5_D_BATCH_SIZE,
        label=label,
    )
    return _ExactAcquisitionView(
        supports=tuple(
            _support_names(universe, support) for support in validated["supports"]
        ),
        selected_outputs=validated["selected"],
        complement_outputs=validated["complements"],
    )


def replay_e32_5_discovery_search(
    record: Mapping[str, Any],
) -> DiscoverySearchReplay | None:
    """Replay shared selector state and validation from raw D conditions only."""
    slices = discovery_scoring_slices(record)
    payload = record["scientific_payload"]
    adaptive = payload["adaptive"]
    if adaptive["stop_reason"] == "scientific_fail_uninformative_internal_endpoint":
        return None
    base_record = adaptive["base"]
    universe = _universe_from_record(base_record["universe"])
    initial_record = adaptive["initial"]
    if not isinstance(initial_record, Mapping):
        raise TypeError("informative raw D lacks its initial support acquisition")
    initial_view = _raw_acquisition_view(
        initial_record,
        universe=universe,
        label="search initial",
    )
    round_records = tuple(adaptive.get("rounds", ()))
    round_views = tuple(
        _raw_acquisition_view(
            round_record,
            universe=universe,
            label=f"search round {round_index}",
        )
        for round_index, round_record in enumerate(round_records)
    )
    state = _derive_search_state(
        universe=universe,
        clean_output=slices.train_clean_output,
        all_internal_output=slices.train_all_internal_output,
        singleton_outputs=slices.train_singleton_outputs,
        leave_one_outputs=slices.train_leave_one_outputs,
        clean_target_indices=slices.train_clean_target_indices,
        initial_view=_ExactAcquisitionView(
            initial_view.supports,
            initial_view.selected_outputs[:, :E32_5_TRAIN_ROW_COUNT],
            initial_view.complement_outputs[:, :E32_5_TRAIN_ROW_COUNT],
        ),
        round_views=tuple(
            _ExactAcquisitionView(
                view.supports,
                view.selected_outputs[:, :E32_5_TRAIN_ROW_COUNT],
                view.complement_outputs[:, :E32_5_TRAIN_ROW_COUNT],
            )
            for view in round_views
        ),
    )
    stop_reason = state.stop_reason
    if stop_reason is None and len(round_views) < LOCAL_SEARCH_ROUNDS:
        proposal = _next_search_proposal(
            universe=universe,
            clean_output=slices.train_clean_output,
            all_internal_output=slices.train_all_internal_output,
            singleton_outputs=slices.train_singleton_outputs,
            leave_one_outputs=slices.train_leave_one_outputs,
            state=state,
            round_index=len(round_views),
        )
        if proposal.supports:
            raise ValueError(
                "raw D stopped before the shared selector reached a terminal"
            )
        outcome = accept_exact_swap_batch(
            fine_edge_ids=universe.fine_edge_names,
            proposal=proposal,
            candidate_evaluations={},
        )
        stop_reason = outcome.stop_reason
    if stop_reason is None:
        stop_reason = "maximum_rounds_reached"
    expected_acquisition_stop = (
        "maximum_rounds_reached"
        if len(round_views) == LOCAL_SEARCH_ROUNDS
        else "planner_stopped"
    )
    if adaptive.get("stop_reason") != expected_acquisition_stop:
        raise ValueError("raw adaptive acquisition stop differs from shared planner")

    validation_outputs: dict[tuple[str, ...], tuple[torch.Tensor, torch.Tensor]] = {}
    for view in (initial_view, *round_views):
        for index, support in enumerate(view.supports):
            validation_outputs[support] = (
                view.selected_outputs[index, E32_5_TRAIN_ROW_COUNT:],
                view.complement_outputs[index, E32_5_TRAIN_ROW_COUNT:],
            )

    def validation_evaluator(
        supports: Sequence[Sequence[str]],
    ) -> Mapping[tuple[str, ...], ExactGroupEvaluation]:
        requested = tuple(tuple(support) for support in supports)
        evaluations: dict[tuple[str, ...], ExactGroupEvaluation] = {}
        for support in requested:
            try:
                selected, complement = validation_outputs[support]
            except KeyError as error:
                raise ValueError(
                    "accepted path support lacks raw validation outputs"
                ) from error
            evaluations[support] = score_exact_group(
                fine_edge_ids=universe.fine_edge_names,
                support=support,
                clean_output=slices.validation_clean_output,
                all_internal_output=slices.validation_all_internal_output,
                selected_output=selected,
                complement_output=complement,
                clean_target_indices=slices.validation_clean_target_indices,
            )
        return evaluations

    validation = select_validation_path_support(
        accepted_path=state.accepted_path,
        batch_evaluator=validation_evaluator,
    )
    search = LocalSearchResult(
        initial_support=state.initial_support,
        final_support=state.current_evaluation.support,
        accepted_path=state.accepted_path,
        accepted_path_joint_faithfulness=state.accepted_joints,
        rounds=state.round_records,
        exact_support_evaluations=1 + sum(len(view.supports) for view in round_views),
        stop_reason=stop_reason,
    )
    return DiscoverySearchReplay(state.singleton_scores, search, validation)


def run_e32_5_development_pilot(
    *,
    case_id: int,
    case: Any,
    eligible_pair: BehaviorallyEligiblePair,
    device: str = "cpu",
) -> E32_5DiscoveryBundle:
    """Run the real case-58 D-only pilot with one fresh model per role."""
    if case_id != 58:
        raise ValueError("the development pilot is reserved for case 58")
    manifest = CASE_MANIFESTS[case_id]

    def model_factory(_role: ModelRole) -> Any:
        return load_verified_model(case_id, device=device)

    def acquire_eap(
        model: Any, eligible: BehaviorallyEligiblePair, method: str
    ) -> GenericEAPRun:
        return run_generic_eap(
            case_id=case_id,
            n_layers=manifest.n_layers,
            n_heads=manifest.n_heads,
            n_ctx=manifest.n_ctx,
            d_vocab_out=manifest.d_vocab_out,
            model=model,
            case=case,
            eligible_pair=eligible,
            method=method,
            device=device,
        )

    def acquire_adaptive(
        model: Any,
        eligible: BehaviorallyEligiblePair,
        universe: InducedEdgeUniverse,
        planner: E32_5SearchPlanner,
    ) -> E32_5AdaptiveAcquisition:
        return run_e32_5_adaptive_replacement(
            case_id=case_id,
            n_layers=manifest.n_layers,
            n_heads=manifest.n_heads,
            n_ctx=manifest.n_ctx,
            d_vocab_out=manifest.d_vocab_out,
            model=model,
            case=case,
            eligible_pair=eligible,
            universe=universe,
            assess_internal_endpoint=planner.assess_internal_endpoint,
            plan_initial_support=planner.plan_initial_support,
            plan_next_round=planner.plan_next_round,
            device=device,
        )

    return run_e32_5_live_pipeline(
        case_id=case_id,
        split="development",
        eligible_panel=eligible_pair,
        model_factory=model_factory,
        adaptive_acquirer=acquire_adaptive,
        eap_acquirer=acquire_eap,
    )


def _observation_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _observation_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_observation_safe(item) for item in value]
    return value


def observe_e32_5_development(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a nonclaim, JSON-safe case-58 observation from raw D bytes."""
    if record.get("case_id") != 58 or record.get("split") != "development":
        raise ValueError("development observation requires the case-58 D artifact")
    replay = replay_e32_5_discovery_search(record)
    endpoint = record["scientific_payload"]["adaptive"]["internal_endpoint_status"]
    common = {
        "schema_version": "e32_5_case58_development_observation_v1",
        "case_id": 58,
        "evidence_role": "nonclaim_method_development",
        "claim_status": "none_nonclaim_method_development_only",
        "internal_endpoint": _observation_safe(endpoint),
    }
    if replay is None:
        return {
            **common,
            "scientific_terminal": True,
            "candidate_support": None,
            "candidate_support_size": 0,
            "search": None,
            "validation": None,
        }
    evaluation = replay.validation.selected_evaluation
    signed = replay.singleton_scores.signed_score
    unsigned = replay.singleton_scores.unsigned_exact_mass
    return {
        **common,
        "scientific_terminal": False,
        "candidate_support": list(replay.validation.selected_support),
        "candidate_support_size": len(replay.validation.selected_support),
        "singleton_partition_summary": {
            "fine_edge_count": int(signed.numel()),
            "positive_signed_edge_count": int((signed > 0).sum()),
            "maximum_signed_score": float(signed.max()),
            "maximum_unsigned_exact_mass": float(unsigned.max()),
        },
        "search": {
            "initial_support": list(replay.search.initial_support),
            "final_search_support": list(replay.search.final_support),
            "accepted_path": [list(value) for value in replay.search.accepted_path],
            "accepted_path_joint_faithfulness": list(
                replay.search.accepted_path_joint_faithfulness
            ),
            "round_count": len(replay.search.rounds),
            "exact_support_evaluations": replay.search.exact_support_evaluations,
            "stop_reason": replay.search.stop_reason,
        },
        "validation": {
            "selected_path_index": replay.validation.selected_path_index,
            "joint_faithfulness": float(evaluation.joint_faithfulness),
            "gate_pass": bool(evaluation.gate_pass),
            "metrics": _observation_safe(evaluation.metrics),
            "path_scores": _observation_safe(
                tuple(asdict(value) for value in replay.validation.path_scores)
            ),
        },
    }


def run_e32_5_case58_development(
    *,
    eligible_panel: BehaviorallyEligiblePair,
    model_factory: Callable[[ModelRole], Any],
    adaptive_acquirer: Callable[
        [
            Any,
            BehaviorallyEligiblePair,
            InducedEdgeUniverse,
            E32_5SearchPlanner,
        ],
        E32_5AdaptiveAcquisition,
    ],
    eap_acquirer: Callable[[Any, BehaviorallyEligiblePair, str], GenericEAPRun],
    artifact_writer: Callable[[dict[str, Any]], None] | None = None,
) -> E32_5DiscoveryBundle:
    """Execute only the case-58 D pilot and optionally persist its raw envelope."""
    bundle = run_e32_5_live_pipeline(
        case_id=58,
        split="development",
        eligible_panel=eligible_panel,
        model_factory=model_factory,
        adaptive_acquirer=adaptive_acquirer,
        eap_acquirer=eap_acquirer,
    )
    if artifact_writer is not None:
        if not callable(artifact_writer):
            raise TypeError("artifact_writer must be callable")
        artifact_writer(serialize_e32_5_discovery_bundle(bundle))
    return bundle


__all__ = [
    "DiscoveryQueryAccounting",
    "DiscoveryScoringSlices",
    "E32_5DiscoveryBundle",
    "PipelineStageAccounting",
    "discovery_scoring_slices",
    "observe_e32_5_development",
    "replay_e32_5_discovery_search",
    "run_e32_5_case58_development",
    "run_e32_5_development_pilot",
    "run_e32_5_live_pipeline",
    "serialize_e32_5_discovery_bundle",
]
