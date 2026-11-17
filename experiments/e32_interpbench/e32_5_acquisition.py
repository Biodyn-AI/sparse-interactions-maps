# ruff: noqa: EXE002
"""Fail-closed adaptive acquisition for prospective E32.5 edge search.

The selector proposes one complete batch of supports at a time.  This module
acquires every exact ``S`` and ``U\\S`` intervention in that batch together,
so each adaptive round pays the fixed acquisition overhead only once.  It does
not rank supports or interpret model outputs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from experiments.e32_interpbench.adapter import (
    DEFAULT_ENV_DIR,
    BehaviorallyEligiblePair,
)
from experiments.e32_interpbench.e32_4_edge_sets import (
    EdgeAblation,
    EdgeConditionPlan,
    EdgeSetAcquisition,
    EdgeSetQueryAccounting,
    EdgeSetRuntime,
    FrozenEdgeSupport,
    InducedEdgeUniverse,
    _acquire_edge_sets_from_patchable,
    _assert_exact_masks,
    _prepare_live_model,
    build_edge_condition_plan,
)
from experiments.e32_interpbench.eap_baseline import DEFAULT_AUTO_CIRCUIT_DIR

E32_5_D_BATCH_SIZE = 64
E32_5_TRAIN_ROW_COUNT = 32
E32_5_MAX_ADAPTIVE_ROUNDS = 3
# At most 4 x 8 one-for-one swaps.  The incumbent is acquired separately.
E32_5_MAX_SUPPORTS_PER_ROUND = 32

AdaptiveStopReason = Literal[
    "planner_stopped",
    "maximum_rounds_reached",
    "scientific_fail_uninformative_internal_endpoint",
]
AcquirePlan = Callable[[EdgeConditionPlan], EdgeSetAcquisition]
AssessInternalEndpoint = Callable[[EdgeSetAcquisition], Mapping[str, Any]]
PlanInitialSupport = Callable[[EdgeSetAcquisition], EdgeConditionPlan]
PlanNextRound = Callable[
    [
        EdgeSetAcquisition,
        EdgeSetAcquisition,
        tuple[EdgeSetAcquisition, ...],
    ],
    EdgeConditionPlan | None,
]


def _require_full_u(universe: InducedEdgeUniverse) -> None:
    if not isinstance(universe, InducedEdgeUniverse):
        raise TypeError("universe must be InducedEdgeUniverse")
    if universe.induced_fine_indices != tuple(range(universe.fine_edge_count)):
        raise ValueError("E32.5 requires the complete internal fine universe")
    if universe.induced_whole_indices != tuple(range(universe.whole_edge_count)):
        raise ValueError("E32.5 requires the complete internal whole universe")


def build_full_u_singleton_partition_plan(
    universe: InducedEdgeUniverse,
) -> EdgeConditionPlan:
    """Build the response-blind atlas ``F(U), F({e}), F(U\\{e})`` for all e."""
    _require_full_u(universe)
    supports = tuple(
        FrozenEdgeSupport(
            support_id=f"full_u_singleton:{edge_index:04d}",
            resolution="fine",
            method="full_u_singleton_partition",
            support_size=1,
            selected_indices=(edge_index,),
            kind="deterministic",
        )
        for edge_index in range(universe.fine_edge_count)
    )
    return build_edge_condition_plan(universe, supports)


def _same_plan(left: EdgeConditionPlan, right: EdgeConditionPlan) -> bool:
    return (
        left.universe == right.universe
        and left.supports == right.supports
        and left.condition_ids == right.condition_ids
        and torch.equal(left.fine_masks, right.fine_masks)
        and left.all_internal_condition_index == right.all_internal_condition_index
        and left.induced_condition_index == right.induced_condition_index
        and left.outside_induced_condition_index
        == right.outside_induced_condition_index
        and left.primary_support_id == right.primary_support_id
        and left.primary_within_induced_complement_condition_index
        == right.primary_within_induced_complement_condition_index
        and left.support_references == right.support_references
        and left.empty_mask_reuses_clean_baseline
        == right.empty_mask_reuses_clean_baseline
    )


def _validate_support_plan(
    plan: EdgeConditionPlan,
    *,
    universe: InducedEdgeUniverse,
    maximum_supports: int,
    label: str,
) -> None:
    if not isinstance(plan, EdgeConditionPlan) or plan.universe != universe:
        raise ValueError(f"{label} plan differs from the frozen full-U universe")
    if len(plan.supports) > maximum_supports:
        raise ValueError(
            f"{label} plan may contain at most {maximum_supports} supports"
        )
    if any(
        support.resolution != "fine"
        or not support.selected_indices
        or len(support.selected_indices) == universe.fine_edge_count
        for support in plan.supports
    ):
        raise ValueError("adaptive supports must be nonempty proper fine subsets of U")
    selected_sets = tuple(support.selected_indices for support in plan.supports)
    if len(selected_sets) != len(set(selected_sets)):
        raise ValueError("adaptive round plan contains duplicate support sets")
    rebuilt = build_edge_condition_plan(
        universe,
        plan.supports,
        primary_support_id=plan.primary_support_id,
    )
    if not _same_plan(plan, rebuilt):
        raise ValueError(
            "adaptive round plan is not the canonical exact partition plan"
        )


def _finite_output(value: Any, *, label: str, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{label} differs from the registered output shape")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains nonfinite values")
    return value


def _validate_query_accounting(
    accounting: EdgeSetQueryAccounting,
    *,
    plan: EdgeConditionPlan,
) -> None:
    if not isinstance(accounting, EdgeSetQueryAccounting):
        raise TypeError("acquisition accounting must be EdgeSetQueryAccounting")
    batch = accounting.example_count
    expected_forwards = 3 + plan.condition_count
    expected_examples = batch * expected_forwards
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("acquisition example count must be positive")
    if not (
        accounting.condition_count == plan.condition_count
        and accounting.source_cache_forwards == 1
        and accounting.baseline_forwards == 2
        and accounting.patched_forwards == plan.condition_count
        and accounting.mask_context_count == plan.condition_count
        and accounting.active_mask_coordinate_checks == int(plan.fine_masks.sum())
        and accounting.expected_physical_forwards == expected_forwards
        and accounting.expected_model_examples == expected_examples
        and accounting.observed_physical_forwards == expected_forwards
        and accounting.observed_model_examples == expected_examples
        and accounting.observed_no_grad_forwards == expected_forwards
        and accounting.observed_grad_enabled_forwards == 0
        and accounting.backward_passes == 0
    ):
        raise RuntimeError("adaptive acquisition differs from exact query accounting")


def _validate_acquisition(
    acquisition: EdgeSetAcquisition,
    *,
    requested_plan: EdgeConditionPlan,
) -> None:
    if not isinstance(acquisition, EdgeSetAcquisition):
        raise TypeError("acquire_plan must return EdgeSetAcquisition")
    if (
        acquisition.case_id != requested_plan.universe.case_id
        or acquisition.universe != requested_plan.universe
        or not _same_plan(acquisition.plan, requested_plan)
        or acquisition.ablation is not EdgeAblation.RESAMPLE
    ):
        raise ValueError(
            "acquisition identity differs from the requested resample plan"
        )
    _validate_query_accounting(acquisition.query_accounting, plan=requested_plan)

    batch = acquisition.query_accounting.example_count
    clean = acquisition.clean_baseline_output
    if not isinstance(clean, torch.Tensor) or clean.ndim != 3:
        raise ValueError("clean baseline must be [example,position,class]")
    output_shape = tuple(clean.shape)
    if output_shape[0] != batch or output_shape[1] < 1 or output_shape[2] < 2:
        raise ValueError("clean baseline differs from registered output geometry")
    _finite_output(clean, label="clean_baseline_output", shape=output_shape)
    for label, output in (
        ("corrupt_baseline_output", acquisition.corrupt_baseline_output),
        ("all_internal_output", acquisition.all_internal_output),
        ("induced_output", acquisition.induced_output),
        ("outside_induced_output", acquisition.outside_induced_output),
    ):
        _finite_output(output, label=label, shape=output_shape)

    condition_shape = (requested_plan.condition_count, *output_shape)
    selected_shape = (len(requested_plan.supports), *output_shape)
    conditions = _finite_output(
        acquisition.condition_outputs,
        label="condition_outputs",
        shape=condition_shape,
    )
    selected = _finite_output(
        acquisition.support_selected_outputs,
        label="support_selected_outputs",
        shape=selected_shape,
    )
    complements = _finite_output(
        acquisition.support_complement_outputs,
        label="support_complement_outputs",
        shape=selected_shape,
    )

    def referenced(index: int | None) -> torch.Tensor:
        return clean if index is None else conditions[index]

    if not torch.equal(
        acquisition.all_internal_output,
        referenced(requested_plan.all_internal_condition_index),
    ) or not torch.equal(
        acquisition.induced_output,
        referenced(requested_plan.induced_condition_index),
    ):
        raise RuntimeError("full-U acquisition aliases do not reference exact outputs")
    if not torch.equal(
        acquisition.outside_induced_output,
        referenced(requested_plan.outside_induced_condition_index),
    ):
        raise RuntimeError("empty outside-U acquisition does not reuse clean baseline")
    for support_index, reference in enumerate(requested_plan.support_references):
        if not torch.equal(
            selected[support_index], referenced(reference.selected_condition_index)
        ) or not torch.equal(
            complements[support_index],
            referenced(reference.complement_condition_index),
        ):
            raise RuntimeError(
                "support outputs differ from their exact plan references"
            )

    primary_output = acquisition.primary_within_induced_complement_output
    primary_index = requested_plan.primary_within_induced_complement_condition_index
    if requested_plan.primary_support_id is None:
        if primary_output is not None:
            raise RuntimeError("unrequested primary complement output was returned")
    elif primary_output is None or not torch.equal(
        primary_output, referenced(primary_index)
    ):
        raise RuntimeError("primary complement output differs from its exact reference")


@dataclass(frozen=True)
class AdaptiveQueryAccounting:
    """Exact aggregate work over the base atlas and batched adaptive rounds."""

    acquisition_count: int
    example_count: int
    condition_count: int
    source_cache_forwards: int
    baseline_forwards: int
    patched_forwards: int
    mask_context_count: int
    active_mask_coordinate_checks: int
    expected_physical_forwards: int
    expected_model_examples: int
    observed_physical_forwards: int
    observed_model_examples: int
    observed_no_grad_forwards: int
    observed_grad_enabled_forwards: int
    backward_passes: int

    @property
    def exact_expected_observed_match(self) -> bool:
        return (
            self.observed_physical_forwards == self.expected_physical_forwards
            and self.observed_model_examples == self.expected_model_examples
            and self.observed_no_grad_forwards == self.expected_physical_forwards
            and self.observed_grad_enabled_forwards == 0
            and self.backward_passes == 0
        )


def _aggregate_accounting(
    acquisitions: Sequence[EdgeSetAcquisition],
) -> AdaptiveQueryAccounting:
    if not acquisitions:
        raise ValueError("at least one acquisition is required")
    records = tuple(value.query_accounting for value in acquisitions)
    example_count = records[0].example_count
    if any(record.example_count != example_count for record in records):
        raise ValueError("adaptive rounds changed the acquired example rows")

    def total(field: str) -> int:
        return sum(int(getattr(record, field)) for record in records)

    return AdaptiveQueryAccounting(
        acquisition_count=len(records),
        example_count=example_count,
        condition_count=total("condition_count"),
        source_cache_forwards=total("source_cache_forwards"),
        baseline_forwards=total("baseline_forwards"),
        patched_forwards=total("patched_forwards"),
        mask_context_count=total("mask_context_count"),
        active_mask_coordinate_checks=total("active_mask_coordinate_checks"),
        expected_physical_forwards=total("expected_physical_forwards"),
        expected_model_examples=total("expected_model_examples"),
        observed_physical_forwards=total("observed_physical_forwards"),
        observed_model_examples=total("observed_model_examples"),
        observed_no_grad_forwards=total("observed_no_grad_forwards"),
        observed_grad_enabled_forwards=total("observed_grad_enabled_forwards"),
        backward_passes=total("backward_passes"),
    )


@dataclass(frozen=True)
class E32_5AdaptiveAcquisition:
    """Raw base and round acquisitions, without selector-derived summaries."""

    universe: InducedEdgeUniverse
    base: EdgeSetAcquisition
    initial: EdgeSetAcquisition | None
    rounds: tuple[EdgeSetAcquisition, ...]
    stop_reason: AdaptiveStopReason
    internal_endpoint_status: Mapping[str, Any]
    query_accounting: AdaptiveQueryAccounting


def _validated_endpoint_status(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or tuple(value) != (
        "endpoint_informative",
        "vector_gap_energy",
        "clean_target_loss_gap",
        "search_authorized",
        "terminal_method_verdict",
    ):
        raise ValueError("internal endpoint status changed from the frozen schema")
    informative = value["endpoint_informative"]
    authorized = value["search_authorized"]
    verdict = value["terminal_method_verdict"]
    if not isinstance(informative, bool) or authorized is not informative:
        raise ValueError("internal endpoint authorization is inconsistent")
    expected_verdict = (
        "continue_to_signed_group_selection"
        if informative
        else "scientific_fail_uninformative_internal_endpoint"
    )
    if verdict != expected_verdict:
        raise ValueError("internal endpoint terminal verdict is inconsistent")
    for label in ("vector_gap_energy", "clean_target_loss_gap"):
        scalar = torch.as_tensor(value[label], dtype=torch.float64)
        if scalar.ndim or not bool(torch.isfinite(scalar)):
            raise ValueError(f"internal endpoint {label} must be a finite scalar")
    return dict(value)


def execute_adaptive_acquisition(
    *,
    universe: InducedEdgeUniverse,
    acquire_plan: AcquirePlan,
    assess_internal_endpoint: AssessInternalEndpoint,
    plan_initial_support: PlanInitialSupport,
    plan_next_round: PlanNextRound,
    cleanup: Callable[[], None],
) -> E32_5AdaptiveAcquisition:
    """Acquire the atlas, one incumbent, then at most three swap batches."""
    _require_full_u(universe)
    if (
        not callable(acquire_plan)
        or not callable(assess_internal_endpoint)
        or not callable(plan_initial_support)
        or not callable(plan_next_round)
    ):
        raise TypeError("acquisition and planning callbacks must be callable")
    if not callable(cleanup):
        raise TypeError("cleanup must be callable")

    try:
        base_plan = build_full_u_singleton_partition_plan(universe)
        base = acquire_plan(base_plan)
        _validate_acquisition(base, requested_plan=base_plan)
        endpoint_status = _validated_endpoint_status(assess_internal_endpoint(base))
        if not endpoint_status["search_authorized"]:
            accounting = _aggregate_accounting((base,))
            return E32_5AdaptiveAcquisition(
                universe=universe,
                base=base,
                initial=None,
                rounds=(),
                stop_reason="scientific_fail_uninformative_internal_endpoint",
                internal_endpoint_status=endpoint_status,
                query_accounting=accounting,
            )

        initial_plan = plan_initial_support(base)
        if (
            not isinstance(initial_plan, EdgeConditionPlan)
            or len(initial_plan.supports) != 1
        ):
            raise ValueError("E32.5 requires exactly one initial support")
        _validate_support_plan(
            initial_plan,
            universe=universe,
            maximum_supports=1,
            label="initial support",
        )
        initial = acquire_plan(initial_plan)
        _validate_acquisition(initial, requested_plan=initial_plan)
        if (
            initial.query_accounting.example_count
            != base.query_accounting.example_count
        ):
            raise ValueError("initial support changed the acquired example rows")

        rounds: list[EdgeSetAcquisition] = []
        stop_reason: AdaptiveStopReason = "maximum_rounds_reached"
        for _round_index in range(E32_5_MAX_ADAPTIVE_ROUNDS):
            next_plan = plan_next_round(base, initial, tuple(rounds))
            if next_plan is None:
                stop_reason = "planner_stopped"
                break
            _validate_support_plan(
                next_plan,
                universe=universe,
                maximum_supports=E32_5_MAX_SUPPORTS_PER_ROUND,
                label="adaptive round",
            )
            acquired = acquire_plan(next_plan)
            _validate_acquisition(acquired, requested_plan=next_plan)
            if (
                acquired.query_accounting.example_count
                != base.query_accounting.example_count
            ):
                raise ValueError("adaptive round changed the acquired example rows")
            rounds.append(acquired)
        all_acquisitions = (base, initial, *rounds)
        accounting = _aggregate_accounting(all_acquisitions)
        if not accounting.exact_expected_observed_match:
            raise RuntimeError("aggregate adaptive work differs from exact accounting")
        return E32_5AdaptiveAcquisition(
            universe,
            base,
            initial,
            tuple(rounds),
            stop_reason,
            endpoint_status,
            accounting,
        )
    finally:
        cleanup()


def acquire_e32_5_from_patchable(
    *,
    case_id: int,
    auto_model: Any,
    forward_boundary: Any,
    grouping: Any,
    universe: InducedEdgeUniverse,
    clean_inputs: Any,
    corrupt_inputs: Any,
    expected_sequence_length: int,
    expected_classes: int,
    runtime: EdgeSetRuntime,
    assess_internal_endpoint: AssessInternalEndpoint,
    plan_initial_support: PlanInitialSupport,
    plan_next_round: PlanNextRound,
) -> E32_5AdaptiveAcquisition:
    """Reuse one prepared patchable model for the base atlas and all rounds."""
    _require_full_u(universe)

    def acquire(plan: EdgeConditionPlan) -> EdgeSetAcquisition:
        acquired = _acquire_edge_sets_from_patchable(
            case_id=case_id,
            auto_model=auto_model,
            forward_boundary=forward_boundary,
            grouping=grouping,
            universe=universe,
            plan=plan,
            clean_inputs=clean_inputs,
            corrupt_inputs=corrupt_inputs,
            expected_sequence_length=expected_sequence_length,
            expected_classes=expected_classes,
            runtime=runtime,
            ablation=EdgeAblation.RESAMPLE,
        )
        if acquired.query_accounting.example_count != E32_5_D_BATCH_SIZE:
            raise ValueError(
                f"E32.5 D acquisition requires exactly {E32_5_D_BATCH_SIZE} rows"
            )
        return acquired

    def cleanup() -> None:
        runtime.set_all_masks(auto_model, 0.0)
        if _assert_exact_masks(auto_model, ()) != 0:
            raise RuntimeError("adaptive acquisition cleanup left an active coordinate")

    return execute_adaptive_acquisition(
        universe=universe,
        acquire_plan=acquire,
        assess_internal_endpoint=assess_internal_endpoint,
        plan_initial_support=plan_initial_support,
        plan_next_round=plan_next_round,
        cleanup=cleanup,
    )


def run_e32_5_adaptive_replacement(
    *,
    case_id: int,
    n_layers: int,
    n_heads: int,
    n_ctx: int,
    d_vocab_out: int,
    model: Any,
    case: Any,
    eligible_pair: BehaviorallyEligiblePair,
    universe: InducedEdgeUniverse,
    assess_internal_endpoint: AssessInternalEndpoint,
    plan_initial_support: PlanInitialSupport,
    plan_next_round: PlanNextRound,
    device: str = "cpu",
    environment_dir: Path | str = DEFAULT_ENV_DIR,
    auto_circuit_dir: Path | str = DEFAULT_AUTO_CIRCUIT_DIR,
) -> E32_5AdaptiveAcquisition:
    """Prepare one fresh patchable model, then run the complete D64 search."""
    pair, auto_model, boundary, grouping, runtime = _prepare_live_model(
        case_id=case_id,
        n_layers=n_layers,
        n_heads=n_heads,
        n_ctx=n_ctx,
        d_vocab_out=d_vocab_out,
        model=model,
        case=case,
        eligible_pair=eligible_pair,
        device=device,
        environment_dir=environment_dir,
        auto_circuit_dir=auto_circuit_dir,
    )
    return acquire_e32_5_from_patchable(
        case_id=case_id,
        auto_model=auto_model,
        forward_boundary=boundary,
        grouping=grouping,
        universe=universe,
        clean_inputs=pair.clean_inputs.to(device),
        corrupt_inputs=pair.corrupt_inputs.to(device),
        expected_sequence_length=n_ctx,
        expected_classes=d_vocab_out,
        runtime=runtime,
        assess_internal_endpoint=assess_internal_endpoint,
        plan_initial_support=plan_initial_support,
        plan_next_round=plan_next_round,
    )


__all__ = [
    "E32_5_D_BATCH_SIZE",
    "E32_5_MAX_ADAPTIVE_ROUNDS",
    "E32_5_MAX_SUPPORTS_PER_ROUND",
    "E32_5_TRAIN_ROW_COUNT",
    "AdaptiveQueryAccounting",
    "E32_5AdaptiveAcquisition",
    "acquire_e32_5_from_patchable",
    "build_full_u_singleton_partition_plan",
    "execute_adaptive_acquisition",
    "run_e32_5_adaptive_replacement",
]
