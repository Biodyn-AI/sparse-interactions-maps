# ruff: noqa: EXE002
"""Write-once canonical driver tests for E32.3."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from experiments.e32_interpbench.adapter import (
    BehaviorallyEligiblePair,
    OfficialDataPair,
)
from experiments.e32_interpbench.component_patching import (
    build_component_atlas,
    build_patch_units,
    class_center_logits,
    vector_pair_interactions,
)
from experiments.e32_interpbench.e32_3_cohort import (
    CASE_ORDER,
    EXPECTED_RESPONSE_BLIND_CAPACITY,
    case_universe,
)
from experiments.e32_interpbench.e32_3_orchestration import (
    freeze_discovery_supports,
)
from experiments.e32_interpbench.e32_3_protocol import DETERMINISTIC_METHOD_ORDER
from experiments.e32_interpbench.e32_3_run import (
    RunPaths,
    _live_split_bundle,
    _panel_identity_sha256,
    _run_registered_experiment,
    _seal_registration,
    run_registered_experiment,
    seal_registration,
)


def _screen_case(case_id: int) -> dict[str, object]:
    cursor = case_id * 10_000

    def rows(count: int) -> tuple[tuple[int, int], ...]:
        nonlocal cursor
        values = tuple((cursor + index, case_id) for index in range(count))
        cursor += count
        return values

    return {
        "case_id": case_id,
        "behavior_only": True,
        "smoke_rows": rows(
            EXPECTED_RESPONSE_BLIND_CAPACITY[case_id]["smoke_unique_rows"]
        ),
        "discovery_clean_rows": rows(64),
        "discovery_corrupt_rows": rows(64),
        "confirmation_clean_rows": rows(128),
        "confirmation_corrupt_rows": rows(128),
    }


class _FakeExecutor:
    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths
        self.events: list[str] = []
        self.score_calls = 0

    def registration_dependencies(self) -> dict[str, object]:
        return {
            "test_runtime": "pinned",
            "case_artifact_sha256": {
                str(case_id): {"artifact": f"{'a' * 62}{case_id % 10:02d}"}
                for case_id in CASE_ORDER
            },
        }

    def screen_all_cases(self) -> tuple[dict[str, object], dict[int, object]]:
        self.events.append("screen")
        return (
            {str(case_id): _screen_case(case_id) for case_id in CASE_ORDER},
            {case_id: object() for case_id in CASE_ORDER},
        )

    def acquire_discovery(self, case_id: int, _panel: object) -> dict[str, Any]:
        expected_prior = ["screen", *(f"D:{value}" for value in CASE_ORDER)]
        assert self.events == expected_prior[: CASE_ORDER.index(case_id) + 1]
        self.events.append(f"D:{case_id}")
        nodes, edges = case_universe(case_id)
        scores = torch.arange(len(nodes), 0, -1, dtype=torch.float64)
        return {
            "schema_version": "e32_3_discovery_acquisition_v1",
            "case_id": case_id,
            "split": "discovery",
            "component_ids": nodes,
            "whole_edges": edges,
            "selector_scores": {
                method: scores + index / 100
                for index, method in enumerate(DETERMINISTIC_METHOD_ORDER)
            },
            **self._custody_fields(case_id, "discovery"),
        }

    def acquire_confirmation(
        self, case_id: int, _panel: object, authorization: dict[str, Any]
    ) -> dict[str, Any]:
        assert all(
            (
                self.paths.run_directory
                / f"case{other}_confirmation_authorization.json"
            ).is_file()
            for other in CASE_ORDER
        )
        assert authorization["status"] == "durable_before_confirmation_intervention"
        assert len(authorization["frozen_supports"]["random_supports"]) == 64
        self.events.append(f"C:{case_id}")
        nodes, edges = case_universe(case_id)
        return {
            "schema_version": "e32_3_confirmation_acquisition_v1",
            "case_id": case_id,
            "split": "confirmation",
            "component_ids": nodes,
            "whole_edges": edges,
            "confirmation_authorization_sha256": authorization["authorization_sha256"],
            **self._custody_fields(case_id, "confirmation"),
        }

    @staticmethod
    def _custody_fields(case_id: int, split: str) -> dict[str, Any]:
        required = 64 if split == "discovery" else 128
        screen = _screen_case(case_id)
        rows = (
            *screen[f"{split}_clean_rows"],
            *screen[f"{split}_corrupt_rows"],
        )
        panel_identity = _panel_identity_sha256(
            case_id=case_id, split=split, selected_token_rows=rows
        )
        families = (
            "eligibility",
            "preparation",
            "atlas_smoke",
            "atlas",
            "exact_fine",
            "eap_ig5",
            *(("set_interventions",) if split == "confirmation" else ()),
        )
        nodes, edges = case_universe(case_id)
        fine = 207 if len(nodes) == 15 else 73
        query_accounting = {
            "eligibility": {
                "physical_forwards": 2,
                "model_examples": 192 if split == "discovery" else 384,
            },
            "preparation": {
                "physical_forwards": 4,
                "model_examples": 4 * required,
            },
            "atlas_smoke": {
                "physical_forwards": 9 if split == "discovery" else 0,
                "model_examples": 16 * required if split == "discovery" else 0,
            },
            "atlas": {
                "physical_forwards": (len(nodes) + len(edges) + 31) // 32,
                "model_examples": (len(nodes) + len(edges)) * required,
            },
            "exact_fine": {
                "physical_forwards": 4 + 2 * fine,
                "model_examples": (4 + 2 * fine) * required,
            },
            "eap_ig5": {
                "physical_forwards": 7,
                "model_examples": 7 * required,
                "backward_passes_inferred_from_pinned_control_flow": 6,
                "gradient_examples": 6 * required,
            },
        }
        if split == "confirmation":
            query_accounting["set_interventions"] = {
                "physical_forwards": 5,
                "model_examples": 143 * required,
            }
        assert tuple(query_accounting) == families
        return {
            "selected_token_rows": rows,
            "behavioral_eligibility": {
                "required_count": required,
                "selected_candidate_indices": tuple(range(required)),
                "selection_policy": (
                    "first_jointly_correct_with_globally_unique_clean_source_rows_"
                    "and_response_blind_exclusions"
                ),
                "all_selected_pairs_correct": True,
            },
            "panel_identity_sha256": panel_identity,
            "family_panel_identity_sha256": {
                family: panel_identity
                for family in ("whole_component", "exact_fine", "eap_ig5")
            },
            "fine_edge_names": ("edge:0",),
            "eap_fine_edge_names": ("edge:0",),
            "fine_to_whole_indices": (0,),
            "clean_output": torch.zeros(required, 1, 2),
            "corrupt_output": torch.ones(required, 1, 2),
            "exact_clean_output": torch.zeros(required, 1, 2),
            "exact_corrupt_output": torch.ones(required, 1, 2),
            "query_accounting": query_accounting,
        }

    def score_complete_panel(self, panel: dict[tuple[int, str], Any]) -> dict[str, Any]:
        assert (self.paths.run_directory / "score_attempt.json").is_file()
        assert tuple(panel) == tuple(
            (case_id, split)
            for case_id in CASE_ORDER
            for split in ("discovery", "confirmation")
        )
        self.score_calls += 1
        self.events.append(f"score:{self.score_calls}")
        return {
            "schema_version": "fake_exact_score_v1",
            "case_order": CASE_ORDER,
            "verdict": True,
        }


def _paths(tmp_path: Path) -> RunPaths:
    return RunPaths(
        registration_path=tmp_path / "registration.json",
        run_attempt_path=tmp_path / "attempt.json",
        run_directory=tmp_path / "deletable" / "run",
    )


def test_private_fake_executor_exercises_full_d_authorize_c_score_lifecycle(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    executor = _FakeExecutor(paths)
    _seal_registration(executor=executor, paths=paths)
    summary = _run_registered_experiment(executor=executor, paths=paths, confirm=True)

    assert executor.events == [
        "screen",
        "D:58",
        "D:129",
        "D:25",
        "C:58",
        "C:129",
        "C:25",
        "score:1",
        "score:2",
    ]
    assert executor.score_calls == 2
    assert summary["exact_score_replay"] is True
    assert paths.run_attempt_path.is_file()
    assert paths.run_attempt_path.parent != paths.run_directory
    assert (paths.run_directory / "acquisition_custody.json").is_file()
    assert (paths.run_directory / "summary.json").is_file()


def test_registration_and_external_attempt_are_write_once(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    executor = _FakeExecutor(paths)
    _seal_registration(executor=executor, paths=paths)
    try:
        _seal_registration(executor=executor, paths=paths)
    except FileExistsError:
        pass
    else:  # pragma: no cover
        raise AssertionError("registration overwrite was accepted")

    _run_registered_experiment(executor=executor, paths=paths, confirm=True)
    try:
        _run_registered_experiment(executor=executor, paths=paths, confirm=True)
    except FileExistsError as error:
        assert "attempt" in str(error)
    else:  # pragma: no cover
        raise AssertionError("a second canonical attempt was accepted")


def test_public_production_entrypoints_have_no_executor_injection() -> None:
    assert tuple(inspect.signature(seal_registration).parameters) == ()
    assert tuple(inspect.signature(run_registered_experiment).parameters) == (
        "confirm",
    )


def test_live_discovery_bridge_composes_one_pair_across_fresh_model_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Toy integration of the real bridge, with only live kernels replaced."""
    case_id = 25
    batch, sequence, classes = 64, 10, 56
    clean_inputs = torch.arange(batch * sequence).reshape(batch, sequence) % 5
    corrupt_inputs = (clean_inputs + 1) % 5
    targets = torch.nn.functional.one_hot(
        torch.zeros(batch, sequence, dtype=torch.long), num_classes=classes
    ).float()
    pair = OfficialDataPair(
        clean_inputs=clean_inputs,
        clean_targets=targets,
        corrupt_inputs=corrupt_inputs,
        corrupt_targets=targets.clone(),
    )
    eligible = BehaviorallyEligiblePair(
        pair=pair,
        candidate_count=96,
        eligible_candidate_count=96,
        clean_incorrect_contexts=0,
        source_incorrect_contexts=0,
        either_incorrect_contexts=0,
        selected_candidate_indices=tuple(range(batch)),
        eligibility_physical_forwards=2,
        eligibility_model_examples=192,
        output_kind="categorical",
        row_disjoint_eligible_candidate_count=96,
        row_overlap_excluded_contexts=0,
        excluded_token_row_count=190,
        selection_policy=(
            "first_jointly_correct_with_globally_unique_clean_source_rows_and_"
            "response_blind_exclusions"
        ),
    )
    models: list[object] = []
    family_pairs: list[tuple[str, object, object]] = []

    def load_model(_case_id: int, *, device: str) -> object:
        assert _case_id == case_id and device == "cpu"
        model = SimpleNamespace(
            cfg=SimpleNamespace(
                n_layers=2,
                n_heads=4,
                n_ctx=sequence,
                d_vocab_out=classes,
            )
        )
        models.append(model)
        return model

    def branch_output(inputs: torch.Tensor) -> torch.Tensor:
        coordinates = torch.linspace(-1.0, 1.0, classes)
        return inputs.float().unsqueeze(-1) * coordinates

    class ToyRunner:
        def __init__(
            self,
            model: object,
            units: tuple[Any, ...],
            *,
            condition_batch_size: int,
            valid_directed_edges: tuple[tuple[str, str], ...],
        ) -> None:
            assert model is models[0] and condition_batch_size == 32
            self.units = units
            self.valid_directed_edges = valid_directed_edges
            self.physical = 0
            self.examples = 0

        def forward_counts(self) -> dict[str, int]:
            return {
                "physical_forwards": self.physical,
                "model_examples": self.examples,
            }

        def prepare_pair(self, value: OfficialDataPair) -> SimpleNamespace:
            self.physical += 2
            self.examples += 2 * len(value.clean_inputs)
            donor: dict[str, torch.Tensor] = {}
            for unit in self.units:
                base = value.corrupt_inputs.float().unsqueeze(-1).expand(-1, -1, 62)
                if unit.kind == "attention_head":
                    donor.setdefault(
                        unit.hook_name,
                        base.unsqueeze(2).expand(-1, -1, 4, -1).clone(),
                    )
                else:
                    donor[unit.hook_name] = base.clone()
            return SimpleNamespace(
                clean_inputs=value.clean_inputs,
                corrupt_inputs=value.corrupt_inputs,
                clean_output=branch_output(value.clean_inputs),
                corrupt_output=branch_output(value.corrupt_inputs),
                donor_activations=donor,
            )

        def batch_invariance_smoke(self, prepared: Any, masks: Any, **kwargs: Any):
            del prepared, kwargs
            assert len(masks) == 8
            self.physical += 9
            self.examples += 16 * batch
            return {
                "passed": True,
                "physical_forwards": 9,
                "model_examples": 16 * batch,
            }

        def acquire_atlas(self, prepared: Any) -> SimpleNamespace:
            atlas = build_component_atlas(
                self.units, valid_directed_edges=self.valid_directed_edges
            )
            nonempty = len(atlas.condition_masks) - 1
            self.physical += 2
            self.examples += nonempty * batch
            base = (
                prepared.clean_output.unsqueeze(0)
                .expand(len(atlas.condition_masks), -1, -1, -1)
                .clone()
            )
            count = atlas.condition_masks.sum(dim=1).float().square()
            direction = torch.linspace(-1.0, 1.0, classes)
            outputs = base + count[:, None, None, None] * direction
            singleton = outputs[list(atlas.singleton_condition_indices)]
            pairs = outputs[list(atlas.pair_condition_indices)]
            interactions = vector_pair_interactions(
                empty_output=outputs[0],
                singleton_outputs=singleton,
                pair_outputs=pairs,
                pair_component_indices=atlas.pair_component_indices,
            )
            return SimpleNamespace(
                atlas=atlas,
                condition_outputs=outputs,
                pair_interactions=interactions,
                query_accounting={
                    "patched_physical_forwards": 2,
                    "patched_model_examples": nonempty * batch,
                },
            )

    _nodes, edges = case_universe(case_id)
    mapping = tuple(
        whole_index
        for whole_index, (_source, target) in enumerate(edges)
        for _ in range(3 if ".attn.hook_result[" in target else 1)
    )
    fine_names = tuple(f"fine:{index}" for index in range(len(mapping)))
    clean_scientific = class_center_logits(branch_output(clean_inputs))[..., 1:, :]
    corrupt_scientific = class_center_logits(branch_output(corrupt_inputs))[..., 1:, :]
    fine_direction = torch.linspace(-1.0, 1.0, classes)

    def exact_run(**kwargs: Any) -> SimpleNamespace:
        family_pairs.append(("exact", kwargs["model"], kwargs["eligible_pair"]))
        assert kwargs["eligible_pair"] is eligible
        offset = fine_direction.expand(len(mapping), batch, sequence - 1, -1)
        return SimpleNamespace(
            grouping=SimpleNamespace(edge_order=edges),
            fine_edge_names=fine_names,
            fine_to_whole_indices=mapping,
            clean_baseline_output=clean_scientific,
            corrupt_baseline_output=corrupt_scientific,
            corrupt_into_clean_outputs=clean_scientific.unsqueeze(0) + offset,
            clean_into_corrupt_outputs=corrupt_scientific.unsqueeze(0) - offset,
            query_accounting=SimpleNamespace(
                observed_physical_forwards=4 + 2 * len(mapping),
                observed_model_examples=(4 + 2 * len(mapping)) * batch,
                observed_no_grad_forwards=4 + 2 * len(mapping),
                observed_grad_enabled_forwards=0,
                backward_passes=0,
            ),
        )

    def eap_run(**kwargs: Any) -> SimpleNamespace:
        family_pairs.append(("eap", kwargs["model"], kwargs["eligible_pair"]))
        assert kwargs["eligible_pair"] is eligible
        return SimpleNamespace(
            grouping=SimpleNamespace(edge_order=edges),
            internal_fine_edge_names=fine_names,
            fine_to_whole_indices=mapping,
            internal_fine_scores=tuple(
                float(index + 1) for index in range(len(mapping))
            ),
            observed_query_accounting={
                "physical_forwards": 7,
                "model_examples": 7 * batch,
                "no_grad_forwards": 1,
                "grad_enabled_forwards": 6,
                "backward_passes_inferred_from_pinned_control_flow": 6,
            },
            query_accounting=SimpleNamespace(gradient_examples=6 * batch),
        )

    import experiments.e32_interpbench.e32_3_acquisition as acquisition_module
    import experiments.e32_interpbench.e32_3_baselines as baseline_module
    import experiments.e32_interpbench.e32_3_run as run_module

    monkeypatch.setattr(run_module, "load_verified_model", load_model)
    monkeypatch.setattr(
        run_module, "load_verified_official_case", lambda _case: object()
    )
    monkeypatch.setattr(acquisition_module, "ComponentPatchRunner", ToyRunner)
    monkeypatch.setattr(
        baseline_module, "run_generic_exact_fine_replacement", exact_run
    )
    monkeypatch.setattr(baseline_module, "run_generic_eap_ig5", eap_run)
    monkeypatch.setattr(
        baseline_module,
        "generic_exact_run_to_dict",
        lambda _run: {"settings": "exact", "fine_edge_names": fine_names},
    )
    monkeypatch.setattr(
        baseline_module,
        "generic_eap_run_to_dict",
        lambda _run: {"settings": "eap_ig5", "fine_edge_names": fine_names},
    )

    bundle = _live_split_bundle(
        case_id=case_id,
        split="discovery",
        eligible=eligible,
        authorization=None,
    )

    assert len(models) == 3 and len({id(model) for model in models}) == 3
    assert [
        (family, model is models[index + 1], proof is eligible)
        for index, (family, model, proof) in enumerate(family_pairs)
    ] == [
        ("exact", True, True),
        ("eap", True, True),
    ]
    assert set(bundle["family_panel_identity_sha256"].values()) == {
        bundle["panel_identity_sha256"]
    }
    assert bundle["fine_edge_names"] == bundle["eap_fine_edge_names"]
    assert bundle["query_accounting"]["preparation"]["physical_forwards"] == 4
    assert bundle["query_accounting"]["atlas_smoke"]["physical_forwards"] == 9
    assert tuple(bundle["selector_scores"]) == DETERMINISTIC_METHOD_ORDER
    assert {
        "condition_ids",
        "condition_masks",
        "clean_inputs",
        "corrupt_inputs",
    }.issubset(bundle["whole_component"])
    assert bundle["exact_fine"]["settings"] == "exact"
    assert bundle["eap_ig5"]["settings"] == "eap_ig5"
    assert (
        bundle["query_accounting"]["eap_ig5"][
            "backward_passes_inferred_from_pinned_control_flow"
        ]
        == 6
    )
    assert bundle["query_accounting"]["eap_ig5"]["gradient_examples"] == 384
    assert (
        sum(
            family["physical_forwards"]
            for family in bundle["query_accounting"].values()
        )
        == 174
    )
    assert (
        sum(family["model_examples"] for family in bundle["query_accounting"].values())
        == 14_272
    )


def _frozen_supports(case_id: int) -> dict[str, Any]:
    nodes, _edges = case_universe(case_id)
    scores = torch.arange(len(nodes), 0, -1, dtype=torch.float64)
    return freeze_discovery_supports(
        case_id=case_id,
        discovery={
            "schema_version": "e32_3_discovery_acquisition_v1",
            "case_id": case_id,
            "split": "discovery",
            "component_ids": nodes,
            "selector_scores": {
                method: scores + index / 100
                for index, method in enumerate(DETERMINISTIC_METHOD_ORDER)
            },
        },
    )


def test_cross_case_confirmation_authorization_fails_before_any_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = SimpleNamespace(
        clean_inputs=torch.zeros(128, 10, dtype=torch.long),
        corrupt_inputs=torch.ones(128, 10, dtype=torch.long),
    )
    model_calls = 0

    def forbidden_model(*_args: Any, **_kwargs: Any) -> object:
        nonlocal model_calls
        model_calls += 1
        return object()

    import experiments.e32_interpbench.e32_3_run as run_module

    monkeypatch.setattr(run_module, "load_verified_model", forbidden_model)
    with pytest.raises(ValueError, match="random support"):
        _live_split_bundle(
            case_id=58,
            split="confirmation",
            eligible=SimpleNamespace(pair=pair),
            authorization={
                "case_id": 58,
                "status": "durable_before_confirmation_intervention",
                # Case 129 has the same component IDs as 58; only the
                # case-specific random-control SHA identity exposes the swap.
                "frozen_supports": _frozen_supports(129),
                "authorization_sha256": "a" * 64,
            },
        )
    assert model_calls == 0


def test_live_confirmation_bridge_reuses_whole_runner_for_all_71_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = 25
    batch, sequence, classes = 128, 10, 56
    nodes, edges = case_universe(case_id)
    units = build_patch_units(2, 4)
    clean_inputs = torch.arange(batch * sequence).reshape(batch, sequence)
    corrupt_inputs = clean_inputs + 10_000
    targets = torch.nn.functional.one_hot(
        torch.zeros(batch, sequence, dtype=torch.long), num_classes=classes
    ).float()
    pair = OfficialDataPair(clean_inputs, targets, corrupt_inputs, targets.clone())
    eligible = BehaviorallyEligiblePair(
        pair=pair,
        candidate_count=192,
        eligible_candidate_count=192,
        clean_incorrect_contexts=0,
        source_incorrect_contexts=0,
        either_incorrect_contexts=0,
        selected_candidate_indices=tuple(range(batch)),
        eligibility_physical_forwards=2,
        eligibility_model_examples=384,
        output_kind="categorical",
        row_disjoint_eligible_candidate_count=192,
        selection_policy=(
            "first_jointly_correct_with_globally_unique_clean_source_rows_and_"
            "response_blind_exclusions"
        ),
    )
    authorization = {
        "case_id": case_id,
        "status": "durable_before_confirmation_intervention",
        "frozen_supports": _frozen_supports(case_id),
        "authorization_sha256": "b" * 64,
    }
    runner = SimpleNamespace(units=units)
    clean_output = torch.zeros(batch, sequence - 1, classes)
    corrupt_output = torch.ones_like(clean_output)
    attention = torch.zeros(batch, sequence, 4, 62)
    mlp = torch.zeros(batch, sequence, 62)
    captures = {
        **{f"blocks.{layer}.attn.hook_result": attention for layer in range(2)},
        **{f"blocks.{layer}.hook_mlp_out": mlp for layer in range(2)},
    }
    whole = SimpleNamespace(
        component_ids=nodes,
        whole_edges=edges,
        query_accounting={
            "bidirectional_preparation": {
                "physical_forwards": 4,
                "model_examples": 4 * batch,
            },
            "batch_invariance_smoke": None,
            "whole_atlas": {
                "patched_physical_forwards": 2,
                "patched_model_examples": 43 * batch,
            },
        },
        batch_invariance_smoke=None,
        pair_interactions=torch.zeros(1),
        clean_activations=captures,
        corrupt_activations=captures,
        clean_output=clean_output,
        corrupt_output=corrupt_output,
        singleton_outputs=torch.zeros(1),
        clean_targets=targets,
    )
    model_objects: list[object] = []
    set_call: dict[str, Any] = {}

    def load_model(*_args: Any, **_kwargs: Any) -> object:
        model = object()
        model_objects.append(model)
        return model

    def set_interventions(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["runner"] is runner and kwargs["whole"] is whole
        supports = tuple(kwargs["supports"])
        assert len(supports) == 71
        set_call["supports"] = supports
        common = torch.zeros_like(clean_output)

        def record(support: Any) -> dict[str, Any]:
            return {
                "support_id": support.support_id,
                "kind": support.kind,
                "method": support.method,
                "quantile": support.quantile,
                "replicate": support.replicate_index,
                "selected_nodes": support.selected_nodes,
                "selected_corrupt_into_clean_output": common,
                "complement_corrupt_into_clean_output": common,
            }

        records = tuple(record(support) for support in supports)
        deterministic = {
            method: next(
                item
                for item in records
                if item["method"] == method and item["quantile"] == 0.5
            )
            for method in DETERMINISTIC_METHOD_ORDER
        }
        return {
            "query_accounting": {
                "physical_forwards": 5,
                "model_examples": 143 * batch,
            },
            "all_corrupt_into_clean_output": corrupt_output,
            "deterministic_set_outputs": deterministic,
            "singleton_curve_set_outputs": records[:3],
            "random_set_outputs": records[7:],
            "all_support_records": records,
        }

    mapping = tuple(
        whole_index
        for whole_index, (_source, target) in enumerate(edges)
        for _ in range(3 if ".attn.hook_result[" in target else 1)
    )
    names = tuple(f"fine:{index}" for index in range(len(mapping)))
    exact = SimpleNamespace(
        grouping=SimpleNamespace(edge_order=edges),
        fine_edge_names=names,
        fine_to_whole_indices=mapping,
        clean_baseline_output=clean_output,
        corrupt_baseline_output=corrupt_output,
        corrupt_into_clean_outputs=torch.zeros(1),
        clean_into_corrupt_outputs=torch.zeros(1),
        query_accounting=SimpleNamespace(
            observed_physical_forwards=150,
            observed_model_examples=150 * batch,
            observed_no_grad_forwards=150,
            observed_grad_enabled_forwards=0,
            backward_passes=0,
        ),
    )
    eap = SimpleNamespace(
        grouping=SimpleNamespace(edge_order=edges),
        internal_fine_edge_names=names,
        fine_to_whole_indices=mapping,
        internal_fine_scores=tuple(float(index + 1) for index in range(len(mapping))),
        observed_query_accounting={
            "physical_forwards": 7,
            "model_examples": 7 * batch,
            "no_grad_forwards": 1,
            "grad_enabled_forwards": 6,
            "backward_passes_inferred_from_pinned_control_flow": 6,
        },
        query_accounting=SimpleNamespace(gradient_examples=6 * batch),
    )

    import experiments.e32_interpbench.e32_3_acquisition as acquisition_module
    import experiments.e32_interpbench.e32_3_baselines as baseline_module
    import experiments.e32_interpbench.e32_3_run as run_module
    import experiments.e32_interpbench.e32_3_scoring as scoring_module

    monkeypatch.setattr(run_module, "load_verified_model", load_model)
    monkeypatch.setattr(
        run_module, "load_verified_official_case", lambda _case: object()
    )
    monkeypatch.setattr(
        acquisition_module, "build_cohort_component_runner", lambda **_kwargs: runner
    )
    monkeypatch.setattr(
        acquisition_module, "acquire_whole_component_atlas", lambda **_kwargs: whole
    )
    monkeypatch.setattr(
        acquisition_module,
        "whole_component_acquisition_to_dict",
        lambda _whole: {
            "condition_ids": tuple(f"condition:{index}" for index in range(44)),
            "condition_masks": torch.zeros(44, 10, dtype=torch.bool),
            "clean_inputs": clean_inputs,
            "corrupt_inputs": corrupt_inputs,
        },
    )
    monkeypatch.setattr(
        acquisition_module, "acquire_confirmation_set_interventions", set_interventions
    )
    monkeypatch.setattr(
        baseline_module,
        "run_generic_exact_fine_replacement",
        lambda **_kwargs: exact,
    )
    monkeypatch.setattr(baseline_module, "run_generic_eap_ig5", lambda **_kwargs: eap)
    monkeypatch.setattr(
        baseline_module,
        "generic_exact_run_to_dict",
        lambda _run: {"settings": "exact", "fine_edge_names": names},
    )
    monkeypatch.setattr(
        baseline_module,
        "generic_eap_run_to_dict",
        lambda _run: {"settings": "eap_ig5", "fine_edge_names": names},
    )
    monkeypatch.setattr(
        scoring_module, "raw_pair_interaction_mass", lambda _value: torch.ones(33)
    )
    monkeypatch.setattr(
        scoring_module, "exact_fine_interaction_mass", lambda **_kwargs: torch.ones(73)
    )
    monkeypatch.setattr(
        scoring_module,
        "build_selector_score_vectors",
        lambda **_kwargs: {
            method: torch.arange(10, dtype=torch.float64)
            for method in DETERMINISTIC_METHOD_ORDER
        },
    )

    bundle = _live_split_bundle(
        case_id=case_id,
        split="confirmation",
        eligible=eligible,
        authorization=authorization,
    )

    assert len(model_objects) == 3 and len(set_call["supports"]) == 71
    assert len(bundle["confirmation_set_interventions"]["all_support_records"]) == 71
    assert bundle["confirmation_authorization_sha256"] == "b" * 64
    assert bundle["query_accounting"]["set_interventions"] == {
        "physical_forwards": 5,
        "model_examples": 18_304,
    }
    assert (
        sum(
            family["physical_forwards"]
            for family in bundle["query_accounting"].values()
        )
        == 170
    )
    assert (
        sum(family["model_examples"] for family in bundle["query_accounting"].values())
        == 44_800
    )
