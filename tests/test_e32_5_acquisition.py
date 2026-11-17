# ruff: noqa: EXE002
"""Tests for prospective E32.5 adaptive full-universe acquisition."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

import experiments.e32_interpbench.e32_5_acquisition as acquisition_module
from experiments.e32_interpbench.e32_4_edge_sets import (
    EdgeAblation,
    EdgeSetAcquisition,
    EdgeSetQueryAccounting,
    EdgeSetRuntime,
    FrozenEdgeSupport,
    InducedEdgeUniverse,
    build_edge_condition_plan,
)
from experiments.e32_interpbench.e32_5_acquisition import (
    E32_5_D_BATCH_SIZE,
    E32_5_MAX_ADAPTIVE_ROUNDS,
    E32_5_MAX_SUPPORTS_PER_ROUND,
    acquire_e32_5_from_patchable,
    build_full_u_singleton_partition_plan,
    execute_adaptive_acquisition,
    run_e32_5_adaptive_replacement,
)


def _full_universe() -> InducedEdgeUniverse:
    return InducedEdgeUniverse(
        case_id=58,
        selected_nodes=("a", "b", "c"),
        whole_edges=(("a", "b"), ("b", "c")),
        fine_edge_names=("ab.Q", "ab.K", "ab.V", "bc"),
        fine_to_whole_indices=(0, 0, 0, 1),
        induced_whole_indices=(0, 1),
        induced_fine_indices=(0, 1, 2, 3),
    )


def _partial_universe() -> InducedEdgeUniverse:
    return InducedEdgeUniverse(
        case_id=58,
        selected_nodes=("a", "b"),
        whole_edges=(("a", "b"), ("b", "c")),
        fine_edge_names=("ab.Q", "ab.K", "ab.V", "bc"),
        fine_to_whole_indices=(0, 0, 0, 1),
        induced_whole_indices=(0,),
        induced_fine_indices=(0, 1, 2),
    )


def _candidate_plan(universe: InducedEdgeUniverse, *, round_index: int):
    supports = tuple(
        FrozenEdgeSupport(
            support_id=f"round:{round_index}:candidate:{index}",
            resolution="fine",
            method="toy_swap",
            support_size=1,
            selected_indices=(index,),
            kind="deterministic",
        )
        for index in (0, 1)
    )
    return build_edge_condition_plan(universe, supports)


def _initial_plan(universe: InducedEdgeUniverse):
    return build_edge_condition_plan(
        universe,
        (
            FrozenEdgeSupport(
                support_id="initial",
                resolution="fine",
                method="toy_initial",
                support_size=2,
                selected_indices=(0, 1),
                kind="deterministic",
            ),
        ),
    )


def _informative_endpoint(_base) -> dict[str, object]:
    return {
        "endpoint_informative": True,
        "vector_gap_energy": 1.0,
        "clean_target_loss_gap": 1.0,
        "search_authorized": True,
        "terminal_method_verdict": "continue_to_signed_group_selection",
    }


def _fake_acquisition(plan, *, batch: int = 2) -> EdgeSetAcquisition:
    positions, classes = 2, 3
    clean = torch.zeros(batch, positions, classes)
    corrupt = torch.full_like(clean, -1.0)
    conditions = torch.stack(
        [
            torch.full_like(clean, float(index + 1))
            for index in range(len(plan.fine_masks))
        ]
    )

    def referenced(index: int | None) -> torch.Tensor:
        return clean if index is None else conditions[index]

    selected = torch.stack(
        [
            referenced(record.selected_condition_index)
            for record in plan.support_references
        ]
    )
    complements = torch.stack(
        [
            referenced(record.complement_condition_index)
            for record in plan.support_references
        ]
    )
    forwards = 3 + plan.condition_count
    accounting = EdgeSetQueryAccounting(
        example_count=batch,
        condition_count=plan.condition_count,
        source_cache_forwards=1,
        baseline_forwards=2,
        patched_forwards=plan.condition_count,
        mask_context_count=plan.condition_count,
        active_mask_coordinate_checks=int(plan.fine_masks.sum()),
        expected_physical_forwards=forwards,
        expected_model_examples=batch * forwards,
        observed_physical_forwards=forwards,
        observed_model_examples=batch * forwards,
        observed_no_grad_forwards=forwards,
        observed_grad_enabled_forwards=0,
        backward_passes=0,
    )
    return EdgeSetAcquisition(
        case_id=plan.universe.case_id,
        ablation=EdgeAblation.RESAMPLE,
        universe=plan.universe,
        plan=plan,
        clean_baseline_output=clean,
        corrupt_baseline_output=corrupt,
        all_internal_output=referenced(plan.all_internal_condition_index),
        induced_output=referenced(plan.induced_condition_index),
        outside_induced_output=referenced(plan.outside_induced_condition_index),
        primary_within_induced_complement_output=None,
        condition_outputs=conditions,
        support_selected_outputs=selected,
        support_complement_outputs=complements,
        query_accounting=accounting,
    )


def test_full_u_singleton_plan_contains_exact_e_and_u_minus_e_masks() -> None:
    universe = _full_universe()
    plan = build_full_u_singleton_partition_plan(universe)

    assert len(plan.supports) == universe.fine_edge_count
    assert plan.condition_count == 2 * universe.fine_edge_count + 1
    assert plan.induced_condition_index == plan.all_internal_condition_index
    assert plan.outside_induced_condition_index is None
    for edge_index, (support, reference) in enumerate(
        zip(plan.supports, plan.support_references, strict=True)
    ):
        assert support.selected_indices == (edge_index,)
        selected = plan.mask_for_condition(reference.selected_condition_index)
        complement = plan.mask_for_condition(reference.complement_condition_index)
        assert selected.sum().item() == 1
        assert selected[edge_index].item() is True
        assert torch.equal(~selected, complement)


def test_full_u_singleton_plan_rejects_historical_induced_subcore() -> None:
    with pytest.raises(ValueError, match="complete internal fine universe"):
        build_full_u_singleton_partition_plan(_partial_universe())


def test_adaptive_executor_caps_rounds_and_aggregates_exact_accounting() -> None:
    universe = _full_universe()
    planned_rounds: list[int] = []
    cleanup_calls = 0

    def planner(_base, _initial, rounds):
        planned_rounds.append(len(rounds))
        return _candidate_plan(universe, round_index=len(rounds))

    def cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    acquired = execute_adaptive_acquisition(
        universe=universe,
        acquire_plan=_fake_acquisition,
        assess_internal_endpoint=_informative_endpoint,
        plan_initial_support=lambda _base: _initial_plan(universe),
        plan_next_round=planner,
        cleanup=cleanup,
    )

    assert planned_rounds == list(range(E32_5_MAX_ADAPTIVE_ROUNDS))
    assert acquired.initial.plan.supports[0].support_id == "initial"
    assert len(acquired.rounds) == E32_5_MAX_ADAPTIVE_ROUNDS
    assert acquired.stop_reason == "maximum_rounds_reached"
    assert cleanup_calls == 1
    expected_forwards = sum(
        acquisition.query_accounting.expected_physical_forwards
        for acquisition in (acquired.base, acquired.initial, *acquired.rounds)
    )
    assert acquired.query_accounting.expected_physical_forwards == expected_forwards
    assert acquired.query_accounting.observed_physical_forwards == expected_forwards
    assert acquired.query_accounting.exact_expected_observed_match is True


def test_adaptive_executor_stops_on_none_and_preserves_raw_outputs() -> None:
    universe = _full_universe()

    def planner(_base, _initial, rounds):
        if rounds:
            return None
        return _candidate_plan(universe, round_index=0)

    acquired = execute_adaptive_acquisition(
        universe=universe,
        acquire_plan=_fake_acquisition,
        assess_internal_endpoint=_informative_endpoint,
        plan_initial_support=lambda _base: _initial_plan(universe),
        plan_next_round=planner,
        cleanup=lambda: None,
    )

    assert acquired.stop_reason == "planner_stopped"
    assert len(acquired.rounds) == 1
    assert acquired.base.support_selected_outputs.shape == (4, 2, 2, 3)
    assert acquired.base.support_complement_outputs.shape == (4, 2, 2, 3)
    assert acquired.initial.support_selected_outputs.shape == (1, 2, 2, 3)
    assert acquired.rounds[0].condition_outputs.shape[0] == 5


@pytest.mark.parametrize("failure_site", ["planner", "acquirer"])
def test_adaptive_executor_fails_closed(failure_site: str) -> None:
    universe = _full_universe()
    cleanup_calls = 0
    acquisition_calls = 0

    def acquire(plan):
        nonlocal acquisition_calls
        acquisition_calls += 1
        if failure_site == "acquirer" and acquisition_calls == 2:
            raise RuntimeError("injected acquisition failure")
        return _fake_acquisition(plan)

    def planner(_base, _initial, _rounds):
        if failure_site == "planner":
            raise RuntimeError("injected planner failure")
        return _candidate_plan(universe, round_index=0)

    def cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    with pytest.raises(RuntimeError, match="injected"):
        execute_adaptive_acquisition(
            universe=universe,
            acquire_plan=acquire,
            assess_internal_endpoint=_informative_endpoint,
            plan_initial_support=lambda _base: _initial_plan(universe),
            plan_next_round=planner,
            cleanup=cleanup,
        )

    assert cleanup_calls == 1


def test_adaptive_executor_rejects_oversized_or_cross_universe_rounds() -> None:
    universe = _full_universe()

    def run_with(planner: Callable):
        return execute_adaptive_acquisition(
            universe=universe,
            acquire_plan=_fake_acquisition,
            assess_internal_endpoint=_informative_endpoint,
            plan_initial_support=lambda _base: _initial_plan(universe),
            plan_next_round=planner,
            cleanup=lambda: None,
        )

    too_many = tuple(
        FrozenEdgeSupport(
            support_id=f"candidate:{index}",
            resolution="fine",
            method="toy",
            support_size=1,
            selected_indices=(index % universe.fine_edge_count,),
            kind="deterministic",
        )
        for index in range(E32_5_MAX_SUPPORTS_PER_ROUND + 1)
    )
    with pytest.raises(ValueError, match="at most 32"):
        run_with(
            lambda _base, _initial, _rounds: build_edge_condition_plan(
                universe, too_many
            )
        )

    partial = _partial_universe()
    with pytest.raises(ValueError, match="frozen full-U universe"):
        run_with(
            lambda _base, _initial, _rounds: _candidate_plan(partial, round_index=0)
        )


def test_adaptive_executor_requires_exactly_one_initial_support() -> None:
    universe = _full_universe()
    with pytest.raises(ValueError, match="exactly one initial support"):
        execute_adaptive_acquisition(
            universe=universe,
            acquire_plan=_fake_acquisition,
            assess_internal_endpoint=_informative_endpoint,
            plan_initial_support=lambda _base: _candidate_plan(universe, round_index=0),
            plan_next_round=lambda _base, _initial, _rounds: None,
            cleanup=lambda: None,
        )


def test_uninformative_base_is_preserved_as_scientific_terminal() -> None:
    universe = _full_universe()
    initial_calls = 0

    def initial(_base):
        nonlocal initial_calls
        initial_calls += 1
        return _initial_plan(universe)

    acquired = execute_adaptive_acquisition(
        universe=universe,
        acquire_plan=_fake_acquisition,
        assess_internal_endpoint=lambda _base: {
            "endpoint_informative": False,
            "vector_gap_energy": 0.0,
            "clean_target_loss_gap": 0.0,
            "search_authorized": False,
            "terminal_method_verdict": (
                "scientific_fail_uninformative_internal_endpoint"
            ),
        },
        plan_initial_support=initial,
        plan_next_round=lambda _base, _initial, _rounds: pytest.fail(
            "round planner must not run"
        ),
        cleanup=lambda: None,
    )

    assert initial_calls == 0
    assert acquired.initial is None
    assert acquired.rounds == ()
    assert acquired.stop_reason == "scientific_fail_uninformative_internal_endpoint"
    assert acquired.query_accounting.acquisition_count == 1
    assert acquired.internal_endpoint_status["search_authorized"] is False


def test_patchable_bridge_reuses_one_model_and_requires_one_d64_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe = _full_universe()
    auto_model = object()
    acquired_model_ids: list[int] = []
    cleanup_calls = 0

    def fake_live_acquisition(**kwargs) -> EdgeSetAcquisition:
        acquired_model_ids.append(id(kwargs["auto_model"]))
        return _fake_acquisition(kwargs["plan"], batch=E32_5_D_BATCH_SIZE)

    def set_all_masks(model, value: float) -> None:
        nonlocal cleanup_calls
        assert model is auto_model
        assert value == 0.0
        cleanup_calls += 1

    monkeypatch.setattr(
        acquisition_module,
        "_acquire_edge_sets_from_patchable",
        fake_live_acquisition,
    )
    monkeypatch.setattr(acquisition_module, "_assert_exact_masks", lambda *_args: 0)
    runtime = EdgeSetRuntime(
        src_ablations=lambda *_args: None,
        patch_mode=lambda *_args, **_kwargs: None,
        set_all_masks=set_all_masks,
        resample_ablation=object(),
        batch_tokenwise_mean_ablation=object(),
    )

    acquired = acquire_e32_5_from_patchable(
        case_id=universe.case_id,
        auto_model=auto_model,
        forward_boundary=object(),
        grouping=object(),
        universe=universe,
        clean_inputs=object(),
        corrupt_inputs=object(),
        expected_sequence_length=2,
        expected_classes=3,
        runtime=runtime,
        assess_internal_endpoint=_informative_endpoint,
        plan_initial_support=lambda _base: _initial_plan(universe),
        plan_next_round=lambda _base, _initial, _rounds: None,
    )

    assert acquired.query_accounting.example_count == E32_5_D_BATCH_SIZE
    assert acquired.query_accounting.acquisition_count == 2
    assert acquired_model_ids == [id(auto_model), id(auto_model)]
    assert cleanup_calls == 1


def test_patchable_bridge_rejects_wrong_batch_after_base_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe = _full_universe()
    acquisition_calls = 0
    cleanup_calls = 0

    def fake_live_acquisition(**kwargs) -> EdgeSetAcquisition:
        nonlocal acquisition_calls
        acquisition_calls += 1
        return _fake_acquisition(kwargs["plan"], batch=2)

    def set_all_masks(_model, _value: float) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    monkeypatch.setattr(
        acquisition_module,
        "_acquire_edge_sets_from_patchable",
        fake_live_acquisition,
    )
    monkeypatch.setattr(acquisition_module, "_assert_exact_masks", lambda *_args: 0)
    runtime = EdgeSetRuntime(
        src_ablations=lambda *_args: None,
        patch_mode=lambda *_args, **_kwargs: None,
        set_all_masks=set_all_masks,
        resample_ablation=object(),
        batch_tokenwise_mean_ablation=object(),
    )

    with pytest.raises(ValueError, match="exactly 64 rows"):
        acquire_e32_5_from_patchable(
            case_id=universe.case_id,
            auto_model=object(),
            forward_boundary=object(),
            grouping=object(),
            universe=universe,
            clean_inputs=object(),
            corrupt_inputs=object(),
            expected_sequence_length=2,
            expected_classes=3,
            runtime=runtime,
            assess_internal_endpoint=_informative_endpoint,
            plan_initial_support=lambda _base: _initial_plan(universe),
            plan_next_round=lambda _base, _initial, _rounds: None,
        )

    assert acquisition_calls == 1
    assert cleanup_calls == 1


def test_public_bridge_prepares_exactly_one_fresh_patchable_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe = _full_universe()
    prepared_model = object()
    expected = object()
    preparation_calls = 0
    acquisition_calls: list[dict] = []
    pair = SimpleNamespace(
        clean_inputs=torch.zeros(E32_5_D_BATCH_SIZE, 2, dtype=torch.long),
        corrupt_inputs=torch.ones(E32_5_D_BATCH_SIZE, 2, dtype=torch.long),
    )

    def prepare(**_kwargs):
        nonlocal preparation_calls
        preparation_calls += 1
        return pair, prepared_model, "boundary", "grouping", "runtime"

    def acquire(**kwargs):
        acquisition_calls.append(kwargs)
        return expected

    monkeypatch.setattr(acquisition_module, "_prepare_live_model", prepare)
    monkeypatch.setattr(acquisition_module, "acquire_e32_5_from_patchable", acquire)

    actual = run_e32_5_adaptive_replacement(
        case_id=universe.case_id,
        n_layers=2,
        n_heads=2,
        n_ctx=2,
        d_vocab_out=3,
        model=object(),
        case=object(),
        eligible_pair=object(),
        universe=universe,
        assess_internal_endpoint=_informative_endpoint,
        plan_initial_support=lambda _base: _initial_plan(universe),
        plan_next_round=lambda _base, _initial, _rounds: None,
        device="cpu",
    )

    assert actual is expected
    assert preparation_calls == 1
    assert len(acquisition_calls) == 1
    assert acquisition_calls[0]["auto_model"] is prepared_model
    assert acquisition_calls[0]["clean_inputs"] is pair.clean_inputs
    assert acquisition_calls[0]["corrupt_inputs"] is pair.corrupt_inputs
