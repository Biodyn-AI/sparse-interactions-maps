# ruff: noqa: EXE002
"""Tests for E32.5 split-aware live composition and raw replay custody."""

from __future__ import annotations

import pytest
import torch

import experiments.e32_interpbench.e32_5_pipeline as pipeline_module
from experiments.e32_interpbench.adapter import (
    BehaviorallyEligiblePair,
    OfficialDataPair,
)
from experiments.e32_interpbench.e32_2_directed_replacement import (
    PromotedFineEdgeGroup,
    PromotedFineEdgeGrouping,
)
from experiments.e32_interpbench.e32_4_eap import GenericEAPRun
from experiments.e32_interpbench.e32_4_edge_sets import (
    EdgeAblation,
    EdgeSetAcquisition,
    EdgeSetQueryAccounting,
    FrozenEdgeSupport,
    InducedEdgeUniverse,
    build_edge_condition_plan,
)
from experiments.e32_interpbench.e32_5_acquisition import (
    AdaptiveQueryAccounting,
    E32_5AdaptiveAcquisition,
    build_full_u_singleton_partition_plan,
    execute_adaptive_acquisition,
)
from experiments.e32_interpbench.e32_5_cohort import case_universe, split_request
from experiments.e32_interpbench.e32_5_pipeline import (
    E32_5SearchPlanner,
    discovery_scoring_slices,
    observe_e32_5_development,
    replay_e32_5_discovery_search,
    run_e32_5_case58_development,
    run_e32_5_development_pilot,
    run_e32_5_live_pipeline,
    serialize_e32_5_discovery_bundle,
)
from experiments.e32_interpbench.e32_5_scoring import internal_endpoint_status
from experiments.e32_interpbench.eap_baseline import (
    FineEdgeScore,
    eap_query_accounting,
    official_eap_settings,
)


def _universe(*, case_id: int = 58, fine_count: int = 207) -> InducedEdgeUniverse:
    nodes, whole_edges = case_universe(case_id)
    remaining = fine_count - len(whole_edges)
    extra_per_whole, extra_remainder = divmod(remaining, len(whole_edges))
    fine_to_whole = tuple(
        whole_index
        for whole_index in range(len(whole_edges))
        for _ in range(
            1 + extra_per_whole + (1 if whole_index < extra_remainder else 0)
        )
    )
    return InducedEdgeUniverse(
        case_id=case_id,
        selected_nodes=nodes,
        whole_edges=whole_edges,
        fine_edge_names=tuple(f"edge:{index:03d}" for index in range(fine_count)),
        fine_to_whole_indices=fine_to_whole,
        induced_whole_indices=tuple(range(len(whole_edges))),
        induced_fine_indices=tuple(range(fine_count)),
    )


def _eligible(*, case_id: int = 58, batch: int = 64) -> BehaviorallyEligiblePair:
    n_ctx, classes = 3, 5
    clean_inputs = torch.arange(batch * n_ctx).reshape(batch, n_ctx).remainder(11)
    corrupt_inputs = (clean_inputs + 2).remainder(11)
    clean_targets = torch.nn.functional.one_hot(
        clean_inputs.remainder(classes), num_classes=classes
    ).to(torch.float32)
    corrupt_targets = torch.nn.functional.one_hot(
        corrupt_inputs.remainder(classes), num_classes=classes
    ).to(torch.float32)
    return BehaviorallyEligiblePair(
        pair=OfficialDataPair(
            clean_inputs,
            clean_targets,
            corrupt_inputs,
            corrupt_targets,
        ),
        candidate_count=96,
        eligible_candidate_count=96,
        clean_incorrect_contexts=0,
        source_incorrect_contexts=0,
        either_incorrect_contexts=0,
        selected_candidate_indices=tuple(range(batch)),
        eligibility_physical_forwards=2,
        eligibility_model_examples=2 * 96,
        output_kind="categorical",
        row_disjoint_eligible_candidate_count=96,
        selection_policy=split_request(
            case_id, "development" if case_id == 58 else "discovery"
        ).selection_policy,
    )


def _edge_acquisition(
    plan,
    *,
    batch: int,
    positions: int = 2,
    classes: int = 5,
    uninformative_all_internal: bool = False,
) -> EdgeSetAcquisition:
    clean = torch.zeros(batch, positions, classes)
    class_direction = torch.arange(classes, dtype=torch.float32)
    class_direction -= class_direction.mean()
    corrupt = -class_direction.expand_as(clean).clone()
    conditions = torch.stack(
        [
            (float(index + 1) * class_direction).expand_as(clean).clone()
            for index in range(plan.condition_count)
        ]
    )
    if uninformative_all_internal:
        conditions[plan.all_internal_condition_index].zero_()

    def output(index: int | None) -> torch.Tensor:
        return clean if index is None else conditions[index]

    selected = torch.stack(
        [
            output(reference.selected_condition_index)
            for reference in plan.support_references
        ]
    )
    complements = torch.stack(
        [
            output(reference.complement_condition_index)
            for reference in plan.support_references
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
        all_internal_output=output(plan.all_internal_condition_index),
        induced_output=output(plan.induced_condition_index),
        outside_induced_output=output(plan.outside_induced_condition_index),
        primary_within_induced_complement_output=None,
        condition_outputs=conditions,
        support_selected_outputs=selected,
        support_complement_outputs=complements,
        query_accounting=accounting,
    )


def _adaptive(
    universe: InducedEdgeUniverse,
    eligible: BehaviorallyEligiblePair | None = None,
) -> E32_5AdaptiveAcquisition:
    if eligible is None:
        eligible = _eligible()
    base = _edge_acquisition(build_full_u_singleton_partition_plan(universe), batch=64)
    initial_support = FrozenEdgeSupport(
        support_id="initial",
        resolution="fine",
        method="signed_group_local_search",
        support_size=31,
        selected_indices=tuple(range(31)),
        kind="deterministic",
    )
    initial = _edge_acquisition(
        build_edge_condition_plan(universe, (initial_support,)), batch=64
    )
    records = (base.query_accounting, initial.query_accounting)

    def total(name: str) -> int:
        return sum(int(getattr(record, name)) for record in records)

    accounting = AdaptiveQueryAccounting(
        acquisition_count=2,
        example_count=64,
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
        observed_grad_enabled_forwards=0,
        backward_passes=0,
    )
    return E32_5AdaptiveAcquisition(
        universe=universe,
        base=base,
        initial=initial,
        rounds=(),
        stop_reason="planner_stopped",
        internal_endpoint_status=internal_endpoint_status(
            clean_output=base.clean_baseline_output[:32],
            all_internal_output=base.all_internal_output[:32],
            clean_target_indices=eligible.pair.clean_targets.argmax(dim=-1)[:32, 1:],
        ),
        query_accounting=accounting,
    )


def _terminal_adaptive(
    universe: InducedEdgeUniverse, eligible: BehaviorallyEligiblePair
) -> E32_5AdaptiveAcquisition:
    base = _edge_acquisition(
        build_full_u_singleton_partition_plan(universe),
        batch=64,
        uninformative_all_internal=True,
    )
    record = base.query_accounting
    accounting = AdaptiveQueryAccounting(
        acquisition_count=1,
        example_count=64,
        condition_count=record.condition_count,
        source_cache_forwards=record.source_cache_forwards,
        baseline_forwards=record.baseline_forwards,
        patched_forwards=record.patched_forwards,
        mask_context_count=record.mask_context_count,
        active_mask_coordinate_checks=record.active_mask_coordinate_checks,
        expected_physical_forwards=record.expected_physical_forwards,
        expected_model_examples=record.expected_model_examples,
        observed_physical_forwards=record.observed_physical_forwards,
        observed_model_examples=record.observed_model_examples,
        observed_no_grad_forwards=record.observed_no_grad_forwards,
        observed_grad_enabled_forwards=0,
        backward_passes=0,
    )
    return E32_5AdaptiveAcquisition(
        universe=universe,
        base=base,
        initial=None,
        rounds=(),
        stop_reason="scientific_fail_uninformative_internal_endpoint",
        internal_endpoint_status=internal_endpoint_status(
            clean_output=base.clean_baseline_output[:32],
            all_internal_output=base.all_internal_output[:32],
            clean_target_indices=eligible.pair.clean_targets.argmax(dim=-1)[:32, 1:],
        ),
        query_accounting=accounting,
    )


def _planned_adaptive(
    universe: InducedEdgeUniverse, planner: E32_5SearchPlanner
) -> E32_5AdaptiveAcquisition:
    return execute_adaptive_acquisition(
        universe=universe,
        acquire_plan=lambda plan: _edge_acquisition(plan, batch=64),
        assess_internal_endpoint=planner.assess_internal_endpoint,
        plan_initial_support=planner.plan_initial_support,
        plan_next_round=planner.plan_next_round,
        cleanup=lambda: None,
    )


def _eap(universe: InducedEdgeUniverse, *, method: str) -> GenericEAPRun:
    names = universe.fine_edge_names
    groups = tuple(
        PromotedFineEdgeGroup(
            promoted_edge=edge,
            fine_edge_names=tuple(
                name
                for name, mapped in zip(
                    names, universe.fine_to_whole_indices, strict=True
                )
                if mapped == whole_index
            ),
            fine_edges=tuple(
                name
                for name, mapped in zip(
                    names, universe.fine_to_whole_indices, strict=True
                )
                if mapped == whole_index
            ),
            collision_count=universe.fine_to_whole_indices.count(whole_index),
        )
        for whole_index, edge in enumerate(universe.whole_edges)
    )
    grouping = PromotedFineEdgeGrouping(
        case_id=universe.case_id,
        edge_order=universe.whole_edges,
        groups=groups,
        fine_edge_count=len(names),
        internal_fine_edge_count=len(names),
        excluded_boundary_fine_edge_count=0,
    )
    integrated = 5 if method == "eap_ig" else None
    accounting = eap_query_accounting(
        example_count=32,
        method=method,
        integrated_grad_samples=integrated,
    )
    settings = official_eap_settings(method, integrated_grad_samples=integrated)
    scores = tuple(
        FineEdgeScore("a", "b", name, float(index)) for index, name in enumerate(names)
    )
    return GenericEAPRun(
        case_id=universe.case_id,
        settings=settings,
        grouping=grouping,
        all_fine_scores=scores,
        internal_fine_edge_names=names,
        internal_fine_scores=tuple(score.signed_score for score in scores),
        fine_to_whole_indices=universe.fine_to_whole_indices,
        query_accounting=accounting,
        observed_query_accounting={
            "physical_forwards": accounting.physical_model_forwards,
            "model_examples": accounting.model_examples,
            "no_grad_forwards": accounting.source_ablation_forwards,
            "grad_enabled_forwards": accounting.gradient_forwards,
            "backward_passes_inferred_from_pinned_control_flow": (
                accounting.backward_passes
            ),
        },
    )


def _run_development():
    universe = _universe()
    eligible = _eligible()
    roles: list[str] = []
    models: dict[str, object] = {}
    adaptive_eligible: list[BehaviorallyEligiblePair] = []
    eap_eligible: list[BehaviorallyEligiblePair] = []

    def model_factory(role: str) -> object:
        roles.append(role)
        model = object()
        models[role] = model
        return model

    def adaptive_acquirer(model, observed_eligible, observed_universe, _planner):
        assert model is models["adaptive_patch"]
        assert observed_universe == universe
        adaptive_eligible.append(observed_eligible)
        return _adaptive(universe)

    def eap_acquirer(model, observed_eligible, method: str):
        expected_role = "regular_eap" if method == "eap" else "eap_ig5"
        assert model is models[expected_role]
        eap_eligible.append(observed_eligible)
        return _eap(universe, method=method)

    bundle = run_e32_5_live_pipeline(
        case_id=58,
        split="development",
        eligible_panel=eligible,
        model_factory=model_factory,
        adaptive_acquirer=adaptive_acquirer,
        eap_acquirer=eap_acquirer,
    )
    return bundle, eligible, roles, adaptive_eligible, eap_eligible


def test_development_composition_uses_fresh_models_and_fixed_d_slices() -> None:
    bundle, eligible, roles, adaptive_eligible, eap_eligible = _run_development()

    assert roles == ["regular_eap", "adaptive_patch", "eap_ig5"]
    assert adaptive_eligible == [eligible]
    assert len(eap_eligible) == 2
    assert eap_eligible[0] is eap_eligible[1]
    search = eap_eligible[0]
    assert search is bundle.search_eligible_panel
    assert torch.equal(search.pair.clean_inputs, eligible.pair.clean_inputs[:32])
    assert search.selected_candidate_indices == eligible.selected_candidate_indices[:32]
    assert bundle.query_accounting.total.grad_enabled_forwards == 7
    assert bundle.query_accounting.total.gradient_examples == 224
    assert (
        bundle.query_accounting.total.backward_passes_inferred_from_pinned_control_flow
        == 7
    )


def test_raw_discovery_serialization_replays_non_bos_targets_and_slices() -> None:
    bundle, _eligible_panel, _roles, _adaptive, _eap = _run_development()
    record = serialize_e32_5_discovery_bundle(bundle)
    replay = discovery_scoring_slices(record)

    assert record["split"] == "development"
    assert record["scientific_payload"]["adaptive"]["base"]["condition_outputs"].shape[
        1:
    ] == (64, 2, 5)
    assert replay.train_clean_output.shape == (32, 2, 5)
    assert replay.validation_clean_output.shape == (32, 2, 5)
    assert replay.train_clean_target_indices.shape == (32, 2)
    assert replay.validation_clean_target_indices.shape == (32, 2)
    expected = bundle.eligible_panel.pair.clean_targets.argmax(dim=-1)[:, 1:]
    assert torch.equal(
        torch.cat(
            (
                replay.train_clean_target_indices,
                replay.validation_clean_target_indices,
            )
        ),
        expected,
    )

    tampered = {
        **record,
        "scientific_payload": {
            **record["scientific_payload"],
            "clean_target_indices": bundle.eligible_panel.pair.clean_targets.argmax(
                dim=-1
            ),
        },
    }
    with pytest.raises(ValueError, match="non-BOS target"):
        discovery_scoring_slices(tampered)


@pytest.mark.parametrize(
    ("tamper_kind", "message"),
    (
        ("selected_alias", "plan references"),
        ("eap_alias", "raw all-fine scores"),
        ("row_order", "selected token rows"),
    ),
)
def test_raw_replay_rejects_derived_alias_and_row_tampering(
    tamper_kind: str, message: str
) -> None:
    bundle, _eligible_panel, _roles, _adaptive, _eap = _run_development()
    record = serialize_e32_5_discovery_bundle(bundle)
    if tamper_kind == "selected_alias":
        record["scientific_payload"]["adaptive"]["base"][
            "support_selected_outputs"
        ].add_(1.0)
    elif tamper_kind == "eap_alias":
        raw_eap = record["scientific_payload"]["regular_eap"]
        raw_eap["internal_fine_scores"] = (
            -999.0,
            *raw_eap["internal_fine_scores"][1:],
        )
    else:
        record["selected_token_rows"] = tuple(reversed(record["selected_token_rows"]))
    with pytest.raises(ValueError, match=message):
        discovery_scoring_slices(record)


def test_raw_replay_rejects_coordinated_stage_and_total_accounting_tamper() -> None:
    bundle, _eligible_panel, _roles, _adaptive, _eap = _run_development()
    record = serialize_e32_5_discovery_bundle(bundle)
    record["query_accounting"]["adaptive_patch"]["physical_forwards"] += 1
    record["query_accounting"]["adaptive_patch"]["no_grad_forwards"] += 1
    record["query_accounting"]["total"]["physical_forwards"] += 1
    record["query_accounting"]["total"]["no_grad_forwards"] += 1

    with pytest.raises(ValueError, match="adaptive_patch accounting"):
        discovery_scoring_slices(record)


def test_case58_pilot_entry_writes_complete_raw_bundle_once() -> None:
    universe = _universe()
    eligible = _eligible()
    written: list[dict] = []

    def model_factory(_role: str) -> object:
        return object()

    bundle = run_e32_5_case58_development(
        eligible_panel=eligible,
        model_factory=model_factory,
        adaptive_acquirer=lambda _model, _eligible, _universe, _planner: _adaptive(
            universe
        ),
        eap_acquirer=lambda _model, _eligible, method: _eap(universe, method=method),
        artifact_writer=written.append,
    )

    assert bundle.case_id == 58
    assert len(written) == 1
    expected = serialize_e32_5_discovery_bundle(bundle)
    assert tuple(written[0]) == tuple(expected)
    assert written[0]["case_id"] == expected["case_id"]
    assert torch.equal(
        written[0]["scientific_payload"]["adaptive"]["base"]["condition_outputs"],
        expected["scientific_payload"]["adaptive"]["base"]["condition_outputs"],
    )


def test_live_composition_rejects_reused_models() -> None:
    universe = _universe()
    eligible = _eligible()
    reused = object()

    with pytest.raises(ValueError, match="fresh distinct model"):
        run_e32_5_live_pipeline(
            case_id=58,
            split="development",
            eligible_panel=eligible,
            model_factory=lambda _role: reused,
            adaptive_acquirer=lambda _model, _eligible, _universe, _planner: _adaptive(
                universe
            ),
            eap_acquirer=lambda _model, _eligible, method: _eap(
                universe, method=method
            ),
        )


def test_uninformative_endpoint_stops_before_ig5_and_serializes_base() -> None:
    universe = _universe()
    eligible = _eligible()
    roles: list[str] = []

    bundle = run_e32_5_live_pipeline(
        case_id=58,
        split="development",
        eligible_panel=eligible,
        model_factory=lambda role: roles.append(role) or object(),
        adaptive_acquirer=lambda _model, observed, observed_universe, _planner: (
            _terminal_adaptive(observed_universe, observed)
        ),
        eap_acquirer=lambda _model, _eligible, method: (
            _eap(universe, method=method)
            if method == "eap"
            else pytest.fail("IG5 must not run after scientific terminal")
        ),
    )

    assert roles == ["regular_eap", "adaptive_patch"]
    assert bundle.eap_ig5 is None
    assert bundle.query_accounting.eap_ig5.physical_forwards == 0
    record = serialize_e32_5_discovery_bundle(bundle)
    assert record["scientific_payload"]["adaptive"]["initial"] is None
    assert record["scientific_payload"]["eap_ig5"] is None
    replay = discovery_scoring_slices(record)
    assert replay.eap_ig5_signed_scores is None


def test_live_planner_and_raw_replay_have_identical_support_sequence() -> None:
    universe = _universe()
    eligible = _eligible()
    bundle = run_e32_5_live_pipeline(
        case_id=58,
        split="development",
        eligible_panel=eligible,
        model_factory=lambda _role: object(),
        adaptive_acquirer=lambda _model, _eligible, observed_universe, planner: (
            _planned_adaptive(observed_universe, planner)
        ),
        eap_acquirer=lambda _model, _eligible, method: _eap(universe, method=method),
    )
    raw = serialize_e32_5_discovery_bundle(bundle)
    replay = replay_e32_5_discovery_search(raw)
    assert replay is not None

    live_support_batches = (
        tuple(
            _support.selected_indices
            for _support in bundle.adaptive.initial.plan.supports
        ),
        *tuple(
            tuple(support.selected_indices for support in acquisition.plan.supports)
            for acquisition in bundle.adaptive.rounds
        ),
    )
    edge_index = {name: index for index, name in enumerate(universe.fine_edge_names)}
    replay_support_batches = (
        (tuple(edge_index[name] for name in replay.search.initial_support),),
        *tuple(
            tuple(
                tuple(edge_index[name] for name in candidate.support)
                for candidate in round_record.exact_candidate_scores
            )
            for round_record in replay.search.rounds
        ),
    )
    assert live_support_batches == replay_support_batches


def test_validation_rows_cannot_change_live_search_plans() -> None:
    universe = _universe()
    eligible = _eligible()
    targets = eligible.pair.clean_targets.argmax(dim=-1)[:, 1:]
    changed_validation = targets.clone()
    changed_validation[32:] = (changed_validation[32:] + 2).remainder(5)

    first = _planned_adaptive(
        universe,
        E32_5SearchPlanner(universe=universe, clean_target_indices=targets),
    )
    second = _planned_adaptive(
        universe,
        E32_5SearchPlanner(universe=universe, clean_target_indices=changed_validation),
    )

    def support_sequence(acquired: E32_5AdaptiveAcquisition):
        return (
            tuple(
                support.selected_indices for support in acquired.initial.plan.supports
            ),
            *tuple(
                tuple(support.selected_indices for support in round_.plan.supports)
                for round_ in acquired.rounds
            ),
        )

    assert support_sequence(first) == support_sequence(second)


def test_production_development_bridge_uses_three_fresh_verified_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe = _universe()
    eligible = _eligible()
    models = [object(), object(), object()]
    loaded: list[tuple[int, str]] = []
    eap_calls: list[tuple[object, int, str]] = []
    adaptive_calls: list[tuple[object, int]] = []

    def load_model(case_id: int, *, device: str):
        loaded.append((case_id, device))
        return models[len(loaded) - 1]

    def acquire_eap(**kwargs):
        eap_calls.append(
            (
                kwargs["model"],
                len(kwargs["eligible_pair"].pair.clean_inputs),
                kwargs["method"],
            )
        )
        return _eap(universe, method=kwargs["method"])

    def acquire_adaptive(**kwargs):
        adaptive_calls.append(
            (kwargs["model"], len(kwargs["eligible_pair"].pair.clean_inputs))
        )
        assert callable(kwargs["assess_internal_endpoint"])
        assert callable(kwargs["plan_initial_support"])
        assert callable(kwargs["plan_next_round"])
        return _adaptive(universe)

    monkeypatch.setattr(pipeline_module, "load_verified_model", load_model)
    monkeypatch.setattr(pipeline_module, "run_generic_eap", acquire_eap)
    monkeypatch.setattr(
        pipeline_module, "run_e32_5_adaptive_replacement", acquire_adaptive
    )

    bundle = run_e32_5_development_pilot(
        case_id=58,
        case=object(),
        eligible_pair=eligible,
        device="cpu",
    )

    assert loaded == [(58, "cpu")] * 3
    assert eap_calls == [(models[0], 32, "eap"), (models[2], 32, "eap_ig")]
    assert adaptive_calls == [(models[1], 64)]
    assert bundle.case_id == 58
    assert bundle.split == "development"


def test_development_observation_is_pure_json_safe_replay() -> None:
    universe = _universe()
    eligible = _eligible()
    bundle = run_e32_5_live_pipeline(
        case_id=58,
        split="development",
        eligible_panel=eligible,
        model_factory=lambda _role: object(),
        adaptive_acquirer=lambda _model, _eligible, observed_universe, planner: (
            _planned_adaptive(observed_universe, planner)
        ),
        eap_acquirer=lambda _model, _eligible, method: _eap(universe, method=method),
    )
    raw = serialize_e32_5_discovery_bundle(bundle)

    observation = observe_e32_5_development(raw)
    replay = replay_e32_5_discovery_search(raw)
    assert replay is not None
    assert observation["schema_version"] == ("e32_5_case58_development_observation_v1")
    assert observation["evidence_role"] == "nonclaim_method_development"
    assert tuple(observation["candidate_support"]) == replay.validation.selected_support
    assert observation["candidate_support_size"] == len(
        replay.validation.selected_support
    )
    assert observation["validation"]["joint_faithfulness"] == pytest.approx(
        replay.validation.selected_evaluation.joint_faithfulness
    )
    assert "selected_output" not in observation["validation"]
    assert "complement_output" not in observation["validation"]

    normalized = {
        **raw,
        "schema_version": "e32_5_raw_acquisition_v1",
        "query_accounting": {
            "families": {
                key: value
                for key, value in raw["query_accounting"].items()
                if key != "total"
            },
            "total": raw["query_accounting"]["total"],
        },
    }
    assert observe_e32_5_development(normalized) == observation
