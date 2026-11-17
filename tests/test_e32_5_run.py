# ruff: noqa: EXE002
"""Lifecycle tests for the write-once E32.5 pilot and final run."""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

import experiments.e32_interpbench.e32_5_run as run_module
from experiments.e32_interpbench.e32_5_run import (
    CASE_ORDER,
    GENERALIZATION_CASE_ORDER,
    DevelopmentPaths,
    RunPaths,
    _build_registration_record,
    _develop_case58,
    _run_registered_experiment,
    _seal_registration,
    build_registration_record,
    develop_case58,
    run_registered_experiment,
    seal_registration,
)


def _accounting(family: str, *, forwards: int = 2, examples: int = 128):
    record = {
        "physical_forwards": forwards,
        "model_examples": examples,
        "no_grad_forwards": forwards,
        "grad_enabled_forwards": 0,
        "gradient_examples": 0,
        "backward_passes_inferred_from_pinned_control_flow": 0,
        "mask_context_count": 0,
        "active_mask_coordinate_checks": 0,
    }
    return {"families": {family: record}, "total": dict(record)}


def _selected_rows(case_id: int, split: str) -> tuple[tuple[int, int], ...]:
    count = 128 if split in {"development", "discovery"} else 256
    return tuple((case_id, index) for index in range(count))


def _input_preflight(case_id: int, split: str) -> dict[str, Any]:
    request = run_module.split_request(case_id, split)
    return {
        "case_id": case_id,
        "split": split,
        "request": asdict(request),
        "clean_unique_rows": request.candidate_count,
        "corrupt_unique_rows": request.candidate_count,
        "role_union_unique_rows": 2 * request.candidate_count,
        "excluded_unique_rows": request.expected_excluded_unique_rows,
        "overlap_with_exclusions": 0,
        "response_blind_selectable_indices": tuple(
            range(request.response_blind_capacity)
        ),
        "observed_raw_identity_sha256": request.raw_identity_sha256,
        "observed_target_identity_sha256": request.target_identity_sha256,
    }


def _bundle(case_id: int, split: str, marker: float) -> dict[str, Any]:
    return {
        "schema_version": "e32_5_raw_acquisition_v1",
        "case_id": case_id,
        "split": split,
        "selected_token_rows": _selected_rows(case_id, split),
        "query_accounting": _accounting(f"case{case_id}_{split}"),
        "scientific_payload": {
            "clean_output": torch.tensor([marker], dtype=torch.float64),
            "raw_condition_outputs": torch.tensor(
                [marker, marker + 1.0], dtype=torch.float64
            ),
        },
    }


class _FakeExecutor:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.freeze_calls: dict[int, int] = {case_id: 0 for case_id in CASE_ORDER}
        self.nondeterministic_case: int | None = None
        self.mutate_source_case: int | None = None
        self.nondeterministic_score = False
        self.score_calls = 0

    def registration_dependencies(self) -> dict[str, Any]:
        self.events.append("registration_dependencies")
        return {"runtime": "fake-pinned", "artifact_tree_sha256": "a" * 64}

    def pilot_input_preflight(self):
        self.events.append("pilot_input_preflight")
        return (
            {
                "schema_version": "e32_5_case58_development_input_preflight_v1",
                "case_id": 58,
                "split": "development",
                "verified": True,
                "model_calls": 0,
                "preflight": _input_preflight(58, "development"),
            },
            {"pilot": "private"},
        )

    def pilot_behavior_screen(self, state):
        assert state == {"pilot": "private"}
        self.events.append("pilot_behavior_screen")
        return (
            {
                "schema_version": "e32_5_case58_development_behavior_screen_v1",
                "case_id": 58,
                "split": "development",
                "behavior_only": True,
                "selected_token_rows": _selected_rows(58, "development"),
            },
            {"case_id": 58, "split": "development"},
        )

    def acquire_development(self, panel):
        assert panel == {"case_id": 58, "split": "development"}
        self.events.append("acquire_development")
        return _bundle(58, "development", 58.0)

    def observe_development(self, bundle):
        assert bundle["split"] == "development"
        self.events.append("observe_development")
        return {
            "schema_version": "e32_5_case58_development_observation_v1",
            "case_id": 58,
            "evidence_role": "nonclaim_method_development",
            "candidate_support": ["fine:000"],
        }

    def final_input_preflight(self):
        self.events.append("final_input_preflight")
        return (
            {
                "schema_version": "e32_5_final_input_preflight_v1",
                "case_order": list(CASE_ORDER),
                "verified": True,
                "model_calls": 0,
                "cases": {
                    str(case_id): {
                        split: _input_preflight(case_id, split)
                        for split in run_module.SPLIT_ORDER_BY_CASE[case_id]
                    }
                    for case_id in CASE_ORDER
                },
            },
            {case_id: {"case_id": case_id} for case_id in CASE_ORDER},
        )

    def final_behavior_screen(self, state):
        assert tuple(state) == CASE_ORDER
        self.events.append("final_behavior_screen")
        return (
            {
                "schema_version": "e32_5_final_behavior_screen_v1",
                "case_order": list(CASE_ORDER),
                "behavior_only": True,
                "cases": {
                    "58": {
                        "case_id": 58,
                        "splits": {
                            "repair_confirmation": {
                                "selected_token_rows": _selected_rows(
                                    58, "repair_confirmation"
                                )
                            }
                        },
                    },
                    **{
                        str(case_id): {
                            "case_id": case_id,
                            "splits": {
                                "discovery": {
                                    "selected_token_rows": _selected_rows(
                                        case_id, "discovery"
                                    )
                                },
                                "confirmation": {
                                    "selected_token_rows": _selected_rows(
                                        case_id, "confirmation"
                                    )
                                },
                            },
                        }
                        for case_id in GENERALIZATION_CASE_ORDER
                    },
                },
            },
            {case_id: {"case_id": case_id} for case_id in CASE_ORDER},
        )

    def freeze_support(self, case_id, development):
        assert development["case_id"] == case_id
        self.freeze_calls[case_id] += 1
        self.events.append(f"freeze_support:{case_id}:{self.freeze_calls[case_id]}")
        clean = development["scientific_payload"]["clean_output"]
        suffix = int(float(clean.reshape(-1)[0]))
        if self.nondeterministic_case == case_id:
            suffix += self.freeze_calls[case_id]
        if self.mutate_source_case == case_id and self.freeze_calls[case_id] == 1:
            clean.add_(500.0)
        return {
            "schema_version": "e32_5_frozen_support_v1",
            "case_id": case_id,
            "support": [f"case{case_id}:fine:{suffix:03d}"],
            "support_size": 1,
            "selector_query_count": 1,
        }

    def acquire_repair_confirmation(self, panel, authorization):
        assert panel["case_id"] == 58 and authorization["case_id"] == 58
        self.events.append("acquire_repair_confirmation")
        return _bundle(58, "repair_confirmation", 158.0)

    def acquire_discovery(self, case_id, panel):
        assert panel["case_id"] == case_id
        self.events.append(f"acquire_discovery:{case_id}")
        return _bundle(case_id, "discovery", float(case_id))

    def acquire_confirmation(self, case_id, panel, authorization):
        assert panel["case_id"] == case_id and authorization["case_id"] == case_id
        self.events.append(f"acquire_confirmation:{case_id}")
        return _bundle(case_id, "confirmation", float(case_id + 1000))

    def score_raw_panel(self, panel):
        self.score_calls += 1
        self.events.append(f"score_raw_panel:{self.score_calls}")
        assert tuple(panel) == (
            (58, "development"),
            (63, "discovery"),
            (103, "discovery"),
            (82, "discovery"),
            (58, "repair_confirmation"),
            (63, "confirmation"),
            (103, "confirmation"),
            (82, "confirmation"),
        )
        value = sum(
            float(bundle["scientific_payload"]["clean_output"].sum())
            for bundle in panel.values()
        )
        if self.nondeterministic_score and self.score_calls == 2:
            value += 1.0
        return {
            "schema_version": "e32_5_raw_score_recomputation_v1",
            "repair": {"case_id": 58, "value": value},
            "cases": {
                str(case_id): {"case_id": case_id, "value": value}
                for case_id in GENERALIZATION_CASE_ORDER
            },
            "panel": {"generalization_pass": True},
        }


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Other agents may land the separately owned live pipeline while these
    # lifecycle-only tests run. Freeze this fake executor's source view so a
    # concurrent legitimate file landing cannot masquerade as in-run TOCTOU.
    frozen_source = run_module._source_hashes()
    monkeypatch.setattr(run_module, "_source_hashes", lambda: frozen_source)
    monkeypatch.setattr(
        run_module,
        "DEVELOPMENT_TUNING_DISCLOSURE",
        "none; lifecycle fake kept the frozen algorithm unchanged after pilot",
    )
    development = DevelopmentPaths(
        tmp_path / "development-attempt.json",
        tmp_path / "development-run",
    )
    final = RunPaths(
        tmp_path / "registration.json",
        tmp_path / "run-attempt.json",
        tmp_path / "final-run",
        development,
    )
    return development, final


def _complete_pilot(executor: _FakeExecutor, development: DevelopmentPaths) -> None:
    _develop_case58(executor=executor, paths=development, confirm=True)


def _seal(executor: _FakeExecutor, paths: RunPaths) -> None:
    _seal_registration(executor=executor, paths=paths)


def test_public_entrypoints_are_not_executor_injectable() -> None:
    assert tuple(inspect.signature(develop_case58).parameters) == ("confirm",)
    assert tuple(inspect.signature(build_registration_record).parameters) == ()
    assert tuple(inspect.signature(seal_registration).parameters) == ()
    assert tuple(inspect.signature(run_registered_experiment).parameters) == (
        "confirm",
    )


def test_registration_is_response_blind_and_requires_terminal_pilot(
    paths,
) -> None:
    development, final = paths
    executor = _FakeExecutor()
    with pytest.raises(ValueError, match="development summary"):
        _build_registration_record(executor=executor, paths=final)
    assert executor.events == []

    _complete_pilot(executor, development)
    executor.events.clear()
    record = _build_registration_record(executor=executor, paths=final)
    assert executor.events == ["registration_dependencies"]
    assert record["development_ancestry"]["status"] == (
        "nonclaim_method_development_complete_algorithm_not_frozen_no_claim"
    )
    assert record["execution"]["final_run_reruns_case58_development"] is False
    assert record["execution"]["production_executor_injectable"] is False
    assert (
        record["execution"][
            "hash_bound_D_reload_is_required_for_independent_authorization"
        ]
        is True
    )
    assert (
        record["execution"][
            "score_attempt_precedes_every_C_artifact_reload_and_scoring_reload"
        ]
        is True
    )


def test_nonclaim_pilot_stops_before_repair_or_prospective_work(paths) -> None:
    development, _final = paths
    executor = _FakeExecutor()
    summary = _develop_case58(executor=executor, paths=development, confirm=True)

    assert executor.events == [
        "registration_dependencies",
        "pilot_input_preflight",
        "registration_dependencies",
        "pilot_behavior_screen",
        "registration_dependencies",
        "acquire_development",
        "observe_development",
        "registration_dependencies",
    ]
    assert summary["claim_status"] == "none_nonclaim_method_development_only"
    assert summary["repair_or_prospective_model_calls"] == 0
    assert (development.run_directory / "case58_development.pt").is_file()
    assert not (development.run_directory / "score_attempt.json").exists()
    assert {path.name for path in development.run_directory.iterdir()} == {
        "development_acquisition_attempt.json",
        "case58_input_preflight.json",
        "case58_behavior_screen.json",
        "case58_development.pt",
        "case58_development_custody.json",
        "development_observation.json",
        "development_summary.json",
    }


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    (
        ("request.clean_seed", -1),
        ("observed_raw_identity_sha256", "0" * 64),
        ("clean_unique_rows", 0),
        ("response_blind_selectable_indices", ()),
    ),
)
def test_pilot_preflight_tamper_blocks_before_behavior_model(
    paths, field: str, tampered_value: Any
) -> None:
    development, _final = paths
    executor = _FakeExecutor()
    original_preflight = executor.pilot_input_preflight

    def tampered_preflight():
        record, state = original_preflight()
        preflight = record["preflight"]
        if field.startswith("request."):
            preflight["request"][field.removeprefix("request.")] = tampered_value
        else:
            preflight[field] = tampered_value
        return record, state

    executor.pilot_input_preflight = tampered_preflight  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="split request, identity, or capacity"):
        _develop_case58(executor=executor, paths=development, confirm=True)
    assert "pilot_behavior_screen" not in executor.events


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    (
        ("request.corrupt_seed", -1),
        ("observed_target_identity_sha256", "0" * 64),
        ("excluded_unique_rows", 0),
        ("role_union_unique_rows", 0),
    ),
)
def test_final_preflight_tamper_blocks_before_any_final_behavior_model(
    paths, field: str, tampered_value: Any
) -> None:
    development, final = paths
    executor = _FakeExecutor()
    _complete_pilot(executor, development)
    _seal(executor, final)
    executor.events.clear()
    original_preflight = executor.final_input_preflight

    def tampered_preflight():
        record, state = original_preflight()
        preflight = record["cases"]["103"]["confirmation"]
        if field.startswith("request."):
            preflight["request"][field.removeprefix("request.")] = tampered_value
        else:
            preflight[field] = tampered_value
        return record, state

    executor.final_input_preflight = tampered_preflight  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="split request, identity, or capacity"):
        _run_registered_experiment(executor=executor, paths=final, confirm=True)
    assert "final_behavior_screen" not in executor.events


def test_nonclaim_pilot_attempt_is_write_once_before_executor_reentry(paths) -> None:
    development, _final = paths
    executor = _FakeExecutor()
    _develop_case58(executor=executor, paths=development, confirm=True)
    events = list(executor.events)
    with pytest.raises(FileExistsError, match="development attempt"):
        _develop_case58(executor=executor, paths=development, confirm=True)
    assert executor.events == events


def test_final_registration_refuses_pending_postpilot_disclosure(
    paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    development, final = paths
    executor = _FakeExecutor()
    _complete_pilot(executor, development)
    monkeypatch.setattr(
        run_module,
        "DEVELOPMENT_TUNING_DISCLOSURE",
        "PENDING_AFTER_CASE58_DEVELOPMENT_PILOT",
    )
    with pytest.raises(ValueError, match="literally state the postpilot changes"):
        _build_registration_record(executor=executor, paths=final)


def test_production_shaped_accounting_is_normalized_and_counted_once() -> None:
    def stage(forwards: int, examples: int) -> dict[str, int]:
        return {
            "physical_forwards": forwards,
            "model_examples": examples,
            "no_grad_forwards": forwards,
            "grad_enabled_forwards": 0,
            "gradient_examples": 0,
            "backward_passes_inferred_from_pinned_control_flow": 0,
            "mask_context_count": forwards - 1,
            "active_mask_coordinate_checks": 3 * forwards,
        }

    families = {
        "behavioral_eligibility": stage(2, 192),
        "adaptive_patch": stage(8, 512),
        "regular_eap": stage(2, 64),
        "eap_ig5": stage(5, 160),
    }
    total = {
        key: sum(record[key] for record in families.values())
        for key in next(iter(families.values()))
    }
    normalized = run_module._normalize_live_bundle(
        {
            "schema_version": "e32_5_discovery_raw_bundle_v1",
            "case_id": 63,
            "split": "discovery",
            "selected_token_rows": tuple((1, index) for index in range(128)),
            "query_accounting": {**families, "total": total},
            "scientific_payload": {"raw": torch.tensor([1.0])},
        },
        split="discovery",
    )
    validated = run_module._validate_bundle(normalized, case_id=63, split="discovery")
    assert tuple(validated["query_accounting"]["families"]) == tuple(families)
    assert validated["query_accounting"]["total"]["physical_forwards"] == 17
    # Behavior screening is an identity audit in the lifecycle; its two
    # forwards occur exactly once here, inside the split bundle.
    assert (
        validated["query_accounting"]["families"]["behavioral_eligibility"][
            "physical_forwards"
        ]
        == 2
    )


def test_production_pilot_bridge_uses_public_pipeline_and_normalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.e32_interpbench.e32_5_pipeline as pipeline

    acquired = object()
    calls: list[tuple[Any, ...]] = []

    def run_pilot(**kwargs):
        calls.append(
            ("run", kwargs["case_id"], kwargs["case"], kwargs["eligible_pair"])
        )
        return acquired

    def serialize(bundle):
        assert bundle is acquired
        calls.append(("serialize",))
        generic = _bundle(58, "development", 1.0)
        accounting = generic["query_accounting"]
        return {
            **generic,
            "schema_version": "e32_5_discovery_raw_bundle_v1",
            "query_accounting": {
                **accounting["families"],
                "total": accounting["total"],
            },
        }

    monkeypatch.setattr(pipeline, "run_e32_5_development_pilot", run_pilot)
    monkeypatch.setattr(pipeline, "serialize_e32_5_discovery_bundle", serialize)
    actual = run_module._ProductionExecutor().acquire_development(
        {"case": "case", "eligible": "eligible"}
    )

    assert calls == [
        ("run", 58, "case", "eligible"),
        ("serialize",),
    ]
    assert actual["schema_version"] == "e32_5_raw_acquisition_v1"
    assert tuple(actual["query_accounting"]) == ("families", "total")


def test_final_lifecycle_has_global_barriers_and_score_reload_embargo(
    paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    development, final = paths
    executor = _FakeExecutor()
    _complete_pilot(executor, development)
    _seal(executor, final)
    executor.events.clear()

    real_load = run_module._torch_load
    loaded: list[tuple[str, bool]] = []

    def guarded_load(path: Path):
        score_reserved = (final.run_directory / "score_attempt.json").is_file()
        if path.parent == final.run_directory and not score_reserved:
            assert path.name in {
                "case63_discovery.pt",
                "case103_discovery.pt",
                "case82_discovery.pt",
            }
        loaded.append((path.name, score_reserved))
        return real_load(path)

    monkeypatch.setattr(run_module, "_torch_load", guarded_load)
    summary = _run_registered_experiment(executor=executor, paths=final, confirm=True)

    expected_events = [
        "registration_dependencies",
        "final_input_preflight",
        "registration_dependencies",
        "final_behavior_screen",
        "registration_dependencies",
        "acquire_discovery:63",
        "acquire_discovery:103",
        "acquire_discovery:82",
        "freeze_support:58:1",
        "freeze_support:63:1",
        "freeze_support:103:1",
        "freeze_support:82:1",
        "freeze_support:58:2",
        "freeze_support:63:2",
        "freeze_support:103:2",
        "freeze_support:82:2",
        "registration_dependencies",
        "registration_dependencies",
        "acquire_repair_confirmation",
        "acquire_confirmation:63",
        "acquire_confirmation:103",
        "acquire_confirmation:82",
        "registration_dependencies",
        "score_raw_panel:1",
        "score_raw_panel:2",
        "registration_dependencies",
        "registration_dependencies",
    ]
    assert executor.events == expected_events
    assert loaded == [
        ("case58_development.pt", False),
        ("case58_development.pt", False),
        ("case63_discovery.pt", False),
        ("case103_discovery.pt", False),
        ("case82_discovery.pt", False),
        ("case58_development.pt", True),
        ("case63_discovery.pt", True),
        ("case103_discovery.pt", True),
        ("case82_discovery.pt", True),
        ("case58_repair_confirmation.pt", True),
        ("case63_confirmation.pt", True),
        ("case103_confirmation.pt", True),
        ("case82_confirmation.pt", True),
        ("case58_development.pt", True),
        ("case63_discovery.pt", True),
        ("case103_discovery.pt", True),
        ("case82_discovery.pt", True),
        ("case58_repair_confirmation.pt", True),
        ("case63_confirmation.pt", True),
        ("case103_confirmation.pt", True),
        ("case82_confirmation.pt", True),
    ]
    assert summary["exact_score_replay"] is True
    assert summary["final_query_accounting"]["total"]["physical_forwards"] == 14
    assert summary["cumulative_query_accounting"]["total"]["physical_forwards"] == 16
    assert (final.run_directory / "score_attempt.json").is_file()
    assert (final.run_directory / "scientific_score.pt").is_file()


def test_final_run_is_write_once_externally(paths) -> None:
    development, final = paths
    executor = _FakeExecutor()
    _complete_pilot(executor, development)
    _seal(executor, final)
    _run_registered_experiment(executor=executor, paths=final, confirm=True)
    events = list(executor.events)

    with pytest.raises(FileExistsError, match="run attempt"):
        _run_registered_experiment(executor=executor, paths=final, confirm=True)
    assert executor.events == [*events, "registration_dependencies"]


def test_nondeterministic_support_recompute_blocks_before_corresponding_c(
    paths,
) -> None:
    development, final = paths
    executor = _FakeExecutor()
    _complete_pilot(executor, development)
    _seal(executor, final)
    executor.events.clear()
    executor.nondeterministic_case = 103

    with pytest.raises(ValueError, match="authorization recomputation"):
        _run_registered_experiment(executor=executor, paths=final, confirm=True)
    assert not any(
        event.startswith("acquire_confirmation") for event in executor.events
    )
    failure = json.loads((final.run_directory / "failure.json").read_text())
    assert failure["score_attempt_sha256"] is None


def test_authorization_recompute_uses_durable_reload_not_mutated_memory(paths) -> None:
    development, final = paths
    executor = _FakeExecutor()
    _complete_pilot(executor, development)
    _seal(executor, final)
    executor.events.clear()
    executor.mutate_source_case = 103

    summary = _run_registered_experiment(executor=executor, paths=final, confirm=True)
    assert summary["exact_score_replay"] is True
    assert "acquire_confirmation:103" in executor.events


def test_score_replay_difference_fails_after_durable_score_attempt(paths) -> None:
    development, final = paths
    executor = _FakeExecutor()
    _complete_pilot(executor, development)
    _seal(executor, final)
    executor.nondeterministic_score = True

    with pytest.raises(RuntimeError, match="score replay"):
        _run_registered_experiment(executor=executor, paths=final, confirm=True)
    assert (final.run_directory / "score_attempt.json").is_file()
    assert not (final.run_directory / "summary.json").exists()


def test_registration_and_attempt_bytes_are_rechecked_before_every_model_phase(
    paths,
) -> None:
    development, final = paths
    executor = _FakeExecutor()
    _complete_pilot(executor, development)
    _seal(executor, final)
    original = executor.final_behavior_screen

    def tampering_screen(state):
        result = original(state)
        final.registration_path.write_text("{}")
        return result

    executor.final_behavior_screen = tampering_screen  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="registration"):
        _run_registered_experiment(executor=executor, paths=final, confirm=True)
    assert "acquire_repair_confirmation" not in executor.events
