# ruff: noqa: EXE002
"""Write-once lifecycle tests for E32.4 internal edge-set faithfulness."""

from __future__ import annotations

import inspect
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

import experiments.e32_interpbench.e32_4_run as run_module
from experiments.e32_interpbench.adapter import DEFAULT_ENV_DIR
from experiments.e32_interpbench.e32_4_protocol import (
    CASE_GEOMETRY,
    CASE_ORDER,
    DETERMINISTIC_METHOD_ORDER,
    EXPECTED_QUERY_CEILINGS,
    PRIMARY_METHOD,
    RANDOM_CONTROL_REPLICATES,
    descriptive_support_sizes,
    deterministic_random_supports,
    primary_support_size,
    split_request,
)
from experiments.e32_interpbench.e32_4_run import (
    CANONICAL_PATHS,
    REPOSITORY_ROOT,
    SOURCE_RELATIVE_PATHS,
    RunPaths,
    _json_safe,
    _panel_identity_sha256,
    _raw_edge_set_outputs,
    _recompute_internal_eap_scores,
    _run_registered_experiment,
    _scientific_equal,
    _seal_registration,
    _source_hashes,
    _source_relative_paths,
    _verify_e32_3_ancestry,
    run_registered_experiment,
    seal_registration,
)


def _paths(tmp_path: Path) -> RunPaths:
    return RunPaths(
        registration_path=tmp_path / "registration.json",
        run_attempt_path=tmp_path / "attempt.json",
        run_directory=tmp_path / "deletable" / "run",
    )


def _rows(case_id: int, split: str) -> tuple[tuple[int, int], ...]:
    count = 64 if split == "discovery" else 128
    offset = case_id * 100_000 + (0 if split == "discovery" else 10_000)
    clean = tuple((offset + index, case_id) for index in range(count))
    corrupt = tuple((offset + count + index, case_id) for index in range(count))
    return (*clean, *corrupt)


def _preflight_record() -> dict[str, Any]:
    return {
        "schema_version": "e32_4_all_pool_input_preflight_v1",
        "verified": True,
        "model_calls": 0,
        "case_order": CASE_ORDER,
        "cases": {
            str(case_id): {
                "case_id": case_id,
                "fresh_raw_pool_audits": {
                    split: {
                        "case_id": case_id,
                        "split": split,
                        "candidate_count": split_request(
                            case_id, split
                        ).candidate_count,
                        "clean_seed": split_request(case_id, split).clean_seed,
                        "corrupt_seed": split_request(case_id, split).corrupt_seed,
                        "identity_sha256": split_request(
                            case_id, split
                        ).raw_identity_sha256,
                        "target_identity_sha256": split_request(
                            case_id, split
                        ).target_identity_sha256,
                    }
                    for split in ("discovery", "confirmation")
                },
                "registered_split_requests": {
                    split: asdict(split_request(case_id, split))
                    for split in ("discovery", "confirmation")
                },
                "historical_unique_raw_role_rows": {
                    58: 768,
                    129: 768,
                    25: 751,
                }[case_id],
                "historical_fresh_role_overlap_count": {
                    58: 0,
                    129: 0,
                    25: 21,
                }[case_id],
                "discovery_response_blind_capacity": {
                    58: 96,
                    129: 96,
                    25: 86,
                }[case_id],
                "confirmation_response_blind_capacity_after_all_discovery_raw_rows": {
                    58: 192,
                    129: 192,
                    25: 172,
                }[case_id],
                "all_raw_pool_identities_verified": True,
                "historical_e32_3_raw_rows_excluded": True,
                "confirmation_excludes_all_new_discovery_raw_rows": True,
            }
            for case_id in CASE_ORDER
        },
    }


class _FakeExecutor:
    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths
        self.events: list[str] = []
        self.freeze_calls: dict[int, int] = {case_id: 0 for case_id in CASE_ORDER}
        self.score_calls = 0

    def registration_dependencies(self) -> dict[str, Any]:
        return {"test_runtime": "pinned", "learned_model_loaded": False}

    def preflight_all_input_pools(self) -> tuple[dict[str, Any], object]:
        self.events.append("preflight")
        return _preflight_record(), object()

    def screen_all_fresh_panels(
        self, private_state: object
    ) -> tuple[dict[str, Any], dict[int, object]]:
        assert private_state is not None
        assert self.events == ["preflight"]
        assert (self.paths.run_directory / "input_preflight.json").is_file()
        self.events.append("screen")
        screens = {}
        for case_id in CASE_ORDER:
            discovery = _rows(case_id, "discovery")
            confirmation = _rows(case_id, "confirmation")
            screens[str(case_id)] = {
                "case_id": case_id,
                "behavior_only": True,
                "discovery_clean_rows": discovery[:64],
                "discovery_corrupt_rows": discovery[64:],
                "confirmation_clean_rows": confirmation[:128],
                "confirmation_corrupt_rows": confirmation[128:],
                "selected_rows_globally_unique_and_response_blind_fresh": True,
            }
        return screens, {case_id: object() for case_id in CASE_ORDER}

    def acquire_discovery(self, case_id: int, _panel: object) -> dict[str, Any]:
        assert all(
            event.startswith(("preflight", "screen", "D:")) for event in self.events
        )
        self.events.append(f"D:{case_id}")
        return self._bundle(case_id, "discovery", authorization=None)

    def freeze_discovery_supports(
        self, case_id: int, discovery: dict[str, Any]
    ) -> dict[str, Any]:
        assert discovery["case_id"] == case_id
        self.freeze_calls[case_id] += 1
        self.events.append(f"freeze{self.freeze_calls[case_id]}:{case_id}")
        count = CASE_GEOMETRY[case_id]["induced_fine_edges"]
        fine_edges = tuple(f"case{case_id}:fine:{index}" for index in range(count))
        size = primary_support_size(case_id)
        deterministic = tuple(
            {
                "support_id": method,
                "method": method,
                "fine_edge_ids": fine_edges[:size],
            }
            for method in DETERMINISTIC_METHOD_ORDER
        )
        curves = tuple(
            {
                "support_id": f"{PRIMARY_METHOD}:{label}",
                "method": PRIMARY_METHOD,
                "curve_id": label,
                "fine_edge_ids": fine_edges[:curve_size],
            }
            for label, curve_size in descriptive_support_sizes(case_id).items()
        )
        random = deterministic_random_supports(
            case_id=case_id,
            fine_edge_ids=fine_edges,
            support_size=size,
        )
        return {
            "schema_version": "e32_4_frozen_discovery_supports_v1",
            "case_id": case_id,
            "support_size": size,
            "induced_fine_edge_ids": fine_edges,
            "deterministic_supports": deterministic,
            "exact_mass_curve_supports": curves,
            "random_supports": tuple(
                {
                    "support_id": f"random:{index:03d}",
                    "replicate": index,
                    "fine_edge_ids": support,
                }
                for index, support in enumerate(random)
            ),
            "confirmation_mask_plan": {
                "schema_version": "fake_deduped_mask_plan_v1",
                "resample": {
                    "deduped_condition_count": 17,
                    "condition_ids": tuple(f"resample:{index}" for index in range(17)),
                    "fine_masks_sha256": "1" * 64,
                },
                "mean_corrupt_primary_only": {
                    "deduped_condition_count": 5,
                    "condition_ids": tuple(f"mean:{index}" for index in range(5)),
                    "fine_masks_sha256": "2" * 64,
                },
            },
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
        assert authorization["status"] == "durable_before_any_confirmation_call"
        assert len(authorization["discovery_artifact_sha256"]) == 64
        assert len(authorization["discovery_custody_sha256"]) == 64
        frozen = authorization["frozen_supports"]
        assert len(frozen["deterministic_supports"]) == len(DETERMINISTIC_METHOD_ORDER)
        assert len(frozen["random_supports"]) == RANDOM_CONTROL_REPLICATES
        assert (
            frozen["confirmation_mask_plan"]["resample"]["deduped_condition_count"]
            == 17
        )
        self.events.append(f"C:{case_id}")
        return self._bundle(
            case_id,
            "confirmation",
            authorization=authorization,
        )

    def score_complete_panel(
        self, panel: dict[tuple[int, str], dict[str, Any]]
    ) -> dict[str, Any]:
        assert (self.paths.run_directory / "score_attempt.json").is_file()
        assert tuple(panel) == tuple(
            (case_id, split)
            for case_id in CASE_ORDER
            for split in ("discovery", "confirmation")
        )
        self.score_calls += 1
        self.events.append(f"score:{self.score_calls}")
        return {
            "schema_version": "fake_e32_4_score_v1",
            "case_order": CASE_ORDER,
            "verdict": True,
        }

    @staticmethod
    def _bundle(
        case_id: int, split: str, authorization: dict[str, Any] | None
    ) -> dict[str, Any]:
        rows = _rows(case_id, split)
        identity = _panel_identity_sha256(
            case_id=case_id,
            split=split,
            selected_token_rows=rows,
        )
        if split == "discovery":
            total = dict(EXPECTED_QUERY_CEILINGS[case_id]["discovery"])
            whole_forwards = 13 + math.ceil(
                (
                    CASE_GEOMETRY[case_id]["nodes"]
                    + CASE_GEOMETRY[case_id]["whole_edges"]
                )
                / 32
            )
            whole_examples = (
                4 * 64
                + 16 * 64
                + (
                    CASE_GEOMETRY[case_id]["nodes"]
                    + CASE_GEOMETRY[case_id]["whole_edges"]
                )
                * 64
            )
            singleton_forwards = 4 + 2 * CASE_GEOMETRY[case_id]["induced_fine_edges"]
            accounting = {
                "behavioral_eligibility": {
                    "physical_forwards": 2,
                    "model_examples": 192,
                    "gradient_forwards": 0,
                    "gradient_examples": 0,
                    "backward_passes_inferred_from_pinned_control_flow": 0,
                },
                "whole_component_atlas": {
                    "physical_forwards": whole_forwards,
                    "model_examples": whole_examples,
                    "gradient_forwards": 0,
                    "gradient_examples": 0,
                    "backward_passes_inferred_from_pinned_control_flow": 0,
                },
                "core_singleton_replacement": {
                    "physical_forwards": singleton_forwards,
                    "model_examples": singleton_forwards * 64,
                    "gradient_forwards": 0,
                    "gradient_examples": 0,
                    "backward_passes_inferred_from_pinned_control_flow": 0,
                },
                "regular_eap": {
                    "physical_forwards": 2,
                    "model_examples": 2 * 64,
                    "gradient_forwards": 1,
                    "backward_passes_inferred_from_pinned_control_flow": 1,
                    "gradient_examples": 64,
                },
                "eap_ig5": {
                    "physical_forwards": 7,
                    "model_examples": 7 * 64,
                    "gradient_forwards": 6,
                    "backward_passes_inferred_from_pinned_control_flow": 6,
                    "gradient_examples": 6 * 64,
                },
                "total": total,
            }
            induced_count = CASE_GEOMETRY[case_id]["induced_fine_edges"]
            score_inputs = {
                "induced_fine_edge_ids": tuple(
                    f"case{case_id}:fine:{index}" for index in range(induced_count)
                ),
                "discovery_selector_scores": {
                    method: tuple(
                        float(induced_count - index) for index in range(induced_count)
                    )
                    for method in DETERMINISTIC_METHOD_ORDER
                },
            }
        else:
            assert authorization is not None
            mask_plan = authorization["frozen_supports"]["confirmation_mask_plan"]
            resample_count = mask_plan["resample"]["deduped_condition_count"]
            mean_count = mask_plan["mean_corrupt_primary_only"][
                "deduped_condition_count"
            ]
            total_forwards = 2 + 3 + resample_count + 3 + mean_count
            total = {
                "physical_forwards": total_forwards,
                "model_examples": (
                    384 + (3 + resample_count) * 128 + (3 + mean_count) * 128
                ),
                "gradient_forwards": 0,
                "gradient_examples": 0,
                "backward_passes_inferred_from_pinned_control_flow": 0,
            }
            accounting = {
                "behavioral_eligibility": {
                    "physical_forwards": 2,
                    "model_examples": 384,
                    "gradient_forwards": 0,
                    "gradient_examples": 0,
                    "backward_passes_inferred_from_pinned_control_flow": 0,
                },
                "resample_set_panel": {
                    "condition_count": resample_count,
                    "physical_forwards": 3 + resample_count,
                    "model_examples": (3 + resample_count) * 128,
                    "gradient_forwards": 0,
                    "gradient_examples": 0,
                    "backward_passes_inferred_from_pinned_control_flow": 0,
                },
                "mean_corrupt_primary_sensitivity": {
                    "condition_count": mean_count,
                    "physical_forwards": 3 + mean_count,
                    "model_examples": (3 + mean_count) * 128,
                    "gradient_forwards": 0,
                    "gradient_examples": 0,
                    "backward_passes_inferred_from_pinned_control_flow": 0,
                },
                "total": total,
            }
            score_inputs = {
                "clean_output": "clean",
                "actual_corrupt_output": "corrupt",
                "clean_target_indices": "targets",
                "resample_all_internal_output": "resample_U",
                "resample_induced_core_output": "resample_I",
                "resample_outside_core_output": "resample_U_minus_I",
                "resample_deterministic_outputs": {},
                "resample_exact_mass_curve_outputs": {},
                "resample_random_outputs": (),
                "resample_within_core_primary_complement_output": "resample_I_minus_S",
                "mean_all_internal_output": "mean_U",
                "mean_induced_core_output": "mean_I",
                "mean_outside_core_output": "mean_U_minus_I",
                "mean_primary_output": "mean_S",
                "mean_primary_complement_output": "mean_U_minus_S",
            }
        return {
            "schema_version": f"e32_4_{split}_acquisition_v1",
            "case_id": case_id,
            "split": split,
            "selected_token_rows": rows,
            "panel_identity_sha256": identity,
            "confirmation_authorization_sha256": (
                None if authorization is None else authorization["authorization_sha256"]
            ),
            "query_accounting": accounting,
            "score_inputs": score_inputs,
            "scientific_payload": {"opaque_until_score_reservation": True},
        }


def test_full_preflight_d_authorize_c_score_lifecycle(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    executor = _FakeExecutor(paths)
    _seal_registration(executor=executor, paths=paths)
    summary = _run_registered_experiment(executor=executor, paths=paths, confirm=True)

    assert executor.events == [
        "preflight",
        "screen",
        "D:58",
        "D:129",
        "D:25",
        "freeze1:58",
        "freeze1:129",
        "freeze1:25",
        "freeze2:58",
        "freeze2:129",
        "freeze2:25",
        "C:58",
        "C:129",
        "C:25",
        "score:1",
        "score:2",
    ]
    assert executor.freeze_calls == {case_id: 2 for case_id in CASE_ORDER}
    assert executor.score_calls == 2
    assert summary["exact_score_replay"] is True
    assert summary["query_ceiling_pass"] is True
    assert (paths.run_directory / "input_preflight.json").is_file()
    assert (paths.run_directory / "global_behavior_screen.json").is_file()
    assert (paths.run_directory / "acquisition_custody.json").is_file()
    assert (paths.run_directory / "summary.json").is_file()


def test_registration_attempt_and_artifacts_are_write_once(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    executor = _FakeExecutor(paths)
    _seal_registration(executor=executor, paths=paths)
    with pytest.raises(FileExistsError):
        _seal_registration(executor=executor, paths=paths)

    _run_registered_experiment(executor=executor, paths=paths, confirm=True)
    with pytest.raises(FileExistsError, match="attempt"):
        _run_registered_experiment(executor=executor, paths=paths, confirm=True)


def test_external_attempt_remains_consumed_after_run_directory_deletion(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    executor = _FakeExecutor(paths)
    _seal_registration(executor=executor, paths=paths)
    _run_registered_experiment(executor=executor, paths=paths, confirm=True)
    shutil.rmtree(paths.run_directory)

    assert paths.run_attempt_path.is_file()
    with pytest.raises(FileExistsError, match="attempt"):
        _run_registered_experiment(
            executor=_FakeExecutor(paths), paths=paths, confirm=True
        )


def test_preflight_tamper_fails_before_screen_or_model(tmp_path: Path) -> None:
    class TamperedPreflight(_FakeExecutor):
        def preflight_all_input_pools(self) -> tuple[dict[str, Any], object]:
            record, state = super().preflight_all_input_pools()
            record["cases"]["25"][
                "confirmation_excludes_all_new_discovery_raw_rows"
            ] = False
            return record, state

    paths = _paths(tmp_path)
    executor = TamperedPreflight(paths)
    _seal_registration(executor=executor, paths=paths)
    with pytest.raises(ValueError, match="preflight"):
        _run_registered_experiment(executor=executor, paths=paths, confirm=True)
    assert executor.events == ["preflight"]


def test_preflight_registered_raw_or_target_identity_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    class IdentityTamper(_FakeExecutor):
        def preflight_all_input_pools(self) -> tuple[dict[str, Any], object]:
            record, state = super().preflight_all_input_pools()
            record["cases"]["58"]["fresh_raw_pool_audits"]["confirmation"][
                "target_identity_sha256"
            ] = "0" * 64
            return record, state

    paths = _paths(tmp_path)
    executor = IdentityTamper(paths)
    _seal_registration(executor=executor, paths=paths)
    with pytest.raises(ValueError, match="raw/target identity"):
        _run_registered_experiment(executor=executor, paths=paths, confirm=True)
    assert executor.events == ["preflight"]


def test_behavior_screen_duplicate_or_cross_split_row_fails_before_d(
    tmp_path: Path,
) -> None:
    class DuplicateScreen(_FakeExecutor):
        def screen_all_fresh_panels(
            self, private_state: object
        ) -> tuple[dict[str, Any], dict[int, object]]:
            records, panels = super().screen_all_fresh_panels(private_state)
            records["58"]["confirmation_clean_rows"] = (
                records["58"]["discovery_clean_rows"][0],
                *records["58"]["confirmation_clean_rows"][1:],
            )
            return records, panels

    paths = _paths(tmp_path)
    executor = DuplicateScreen(paths)
    _seal_registration(executor=executor, paths=paths)
    with pytest.raises(ValueError, match="fresh unique"):
        _run_registered_experiment(executor=executor, paths=paths, confirm=True)
    assert not any(event.startswith("D:") for event in executor.events)


def test_missing_regular_eap_accounting_stops_before_authorization_or_c(
    tmp_path: Path,
) -> None:
    class MissingEAP(_FakeExecutor):
        def acquire_discovery(self, case_id: int, panel: object) -> dict[str, Any]:
            bundle = super().acquire_discovery(case_id, panel)
            if case_id == CASE_ORDER[0]:
                del bundle["query_accounting"]["regular_eap"]
            return bundle

    paths = _paths(tmp_path)
    executor = MissingEAP(paths)
    _seal_registration(executor=executor, paths=paths)
    with pytest.raises(ValueError, match="accounting sections"):
        _run_registered_experiment(executor=executor, paths=paths, confirm=True)
    assert not any(event.startswith(("freeze", "C:")) for event in executor.events)


def test_authorization_tamper_fails_at_global_c_barrier(tmp_path: Path) -> None:
    class TamperingExecutor(_FakeExecutor):
        def freeze_discovery_supports(
            self, case_id: int, discovery: dict[str, Any]
        ) -> dict[str, Any]:
            frozen = super().freeze_discovery_supports(case_id, discovery)
            if case_id == CASE_ORDER[-1] and self.freeze_calls[case_id] == 2:
                path = (
                    self.paths.run_directory
                    / f"case{CASE_ORDER[0]}_confirmation_authorization.json"
                )
                payload = json.loads(path.read_text())
                payload["status"] = "tampered"
                path.write_text(json.dumps(payload))
            return frozen

    paths = _paths(tmp_path)
    executor = TamperingExecutor(paths)
    _seal_registration(executor=executor, paths=paths)
    with pytest.raises(ValueError, match="authorization"):
        _run_registered_experiment(executor=executor, paths=paths, confirm=True)
    assert not any(event.startswith("C:") for event in executor.events)


def test_authorization_is_reverified_after_runtime_replay_before_c(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    executor = _FakeExecutor(paths)
    original = run_module._verify_live_custody
    replay_count = 0

    def tampering_runtime_replay(**kwargs: Any) -> None:
        nonlocal replay_count
        replay_count += 1
        original(**kwargs)
        if replay_count == 3:
            path = paths.run_directory / "case58_confirmation_authorization.json"
            payload = json.loads(path.read_text())
            payload["status"] = "tampered_after_runtime_replay"
            path.write_text(json.dumps(payload))

    monkeypatch.setattr(run_module, "_verify_live_custody", tampering_runtime_replay)
    _seal_registration(executor=executor, paths=paths)
    with pytest.raises(ValueError, match="authorization"):
        _run_registered_experiment(executor=executor, paths=paths, confirm=True)

    assert replay_count == 3
    assert not any(event.startswith("C:") for event in executor.events)


def test_confirmation_bundle_authorization_hash_mismatch_is_fatal(
    tmp_path: Path,
) -> None:
    class WrongAuthorizationHash(_FakeExecutor):
        def acquire_confirmation(
            self, case_id: int, panel: object, authorization: dict[str, Any]
        ) -> dict[str, Any]:
            bundle = super().acquire_confirmation(case_id, panel, authorization)
            bundle["confirmation_authorization_sha256"] = "0" * 64
            return bundle

    paths = _paths(tmp_path)
    executor = WrongAuthorizationHash(paths)
    _seal_registration(executor=executor, paths=paths)
    with pytest.raises(ValueError, match="authorization"):
        _run_registered_experiment(executor=executor, paths=paths, confirm=True)


def test_confirmation_score_input_missing_i_minus_primary_is_fatal(
    tmp_path: Path,
) -> None:
    class MissingWithinCore(_FakeExecutor):
        def acquire_confirmation(
            self, case_id: int, panel: object, authorization: dict[str, Any]
        ) -> dict[str, Any]:
            bundle = super().acquire_confirmation(case_id, panel, authorization)
            del bundle["score_inputs"]["resample_within_core_primary_complement_output"]
            return bundle

    paths = _paths(tmp_path)
    executor = MissingWithinCore(paths)
    _seal_registration(executor=executor, paths=paths)
    with pytest.raises(ValueError, match="score_inputs"):
        _run_registered_experiment(executor=executor, paths=paths, confirm=True)


def test_score_replay_difference_is_fatal(tmp_path: Path) -> None:
    class NondeterministicScorer(_FakeExecutor):
        def score_complete_panel(
            self, panel: dict[tuple[int, str], dict[str, Any]]
        ) -> dict[str, Any]:
            value = super().score_complete_panel(panel)
            value["verdict"] = self.score_calls == 1
            return value

    paths = _paths(tmp_path)
    executor = NondeterministicScorer(paths)
    _seal_registration(executor=executor, paths=paths)
    with pytest.raises(RuntimeError, match="replay"):
        _run_registered_experiment(executor=executor, paths=paths, confirm=True)
    assert (paths.run_directory / "score_attempt.json").is_file()
    assert (paths.run_directory / "failure.json").is_file()


def test_uninformative_nan_score_replays_and_serializes_as_null(
    tmp_path: Path,
) -> None:
    class UninformativeScorer(_FakeExecutor):
        def score_complete_panel(
            self, panel: dict[tuple[int, str], dict[str, Any]]
        ) -> dict[str, Any]:
            value = super().score_complete_panel(panel)
            value["endpoint_informative"] = False
            value["descriptive_projection"] = float("nan")
            value["per_example"] = torch.tensor([float("nan"), 0.0])
            return value

    paths = _paths(tmp_path)
    executor = UninformativeScorer(paths)
    _seal_registration(executor=executor, paths=paths)
    summary = _run_registered_experiment(executor=executor, paths=paths, confirm=True)

    assert summary["endpoint_informative"] is False
    assert summary["descriptive_projection"] is None
    assert summary["per_example"] == [None, 0.0]
    assert summary["exact_score_replay"] is True


def test_tensor_nan_replay_equality_and_json_sanitization() -> None:
    left = {"value": torch.tensor([1.0, float("nan")]), "score": float("nan")}
    right = {"value": torch.tensor([1.0, float("nan")]), "score": float("nan")}
    assert _scientific_equal(left, right) is True
    assert _json_safe(left) == {"value": [1.0, None], "score": None}


def test_raw_edge_set_support_mapping_tamper_is_rejected() -> None:
    condition_outputs = torch.tensor([[[[1.0, -1.0]]], [[[2.0, -2.0]]]])
    record = {
        "universe": {"fine_edge_names": ("e0", "e1")},
        "plan": {
            "supports": (
                {
                    "support_id": "primary",
                    "selected_indices": (0,),
                },
            ),
            "support_references": (
                {
                    "support_id": "primary",
                    "selected_condition_index": 0,
                    "complement_condition_index": 1,
                },
            ),
            "all_internal_condition_index": 0,
            "induced_condition_index": 0,
            "outside_induced_condition_index": 1,
            "primary_within_induced_complement_condition_index": None,
        },
        "condition_outputs": condition_outputs,
        "clean_baseline_output": torch.zeros(1, 1, 2),
        "corrupt_baseline_output": torch.ones(1, 1, 2),
        "all_internal_output": condition_outputs[0],
        "induced_output": condition_outputs[0],
        "outside_induced_output": condition_outputs[1],
        "primary_within_induced_complement_output": None,
        # Deliberately swapped relative to the support references.
        "support_selected_outputs": condition_outputs[1:].clone(),
        "support_complement_outputs": condition_outputs[:1].clone(),
    }
    with pytest.raises(ValueError, match="support stack"):
        _raw_edge_set_outputs(record)


def test_raw_eap_records_are_authoritative_over_derived_internal_scores() -> None:
    record = {
        "all_fine_scores": (
            {"fine_edge_name": "boundary", "signed_score": 99.0},
            {"fine_edge_name": "fine-b", "signed_score": -2.0},
            {"fine_edge_name": "fine-a", "signed_score": 1.5},
        ),
        "internal_fine_edge_names": ("fine-a", "fine-b"),
        "internal_fine_scores": (1.5, -2.0),
    }
    assert _recompute_internal_eap_scores(
        record, fine_edge_names=("fine-a", "fine-b"), label="regular EAP"
    ) == (1.5, -2.0)

    record["internal_fine_scores"] = (1.5, 7.0)
    with pytest.raises(ValueError, match="derived internal scores"):
        _recompute_internal_eap_scores(
            record, fine_edge_names=("fine-a", "fine-b"), label="regular EAP"
        )


def test_score_attempt_precedes_every_torch_artifact_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    executor = _FakeExecutor(paths)
    observed_loads: list[Path] = []
    original = run_module._load_artifact

    def guarded_load_artifact(**kwargs: Any) -> dict[str, Any]:
        assert (paths.run_directory / "score_attempt.json").is_file()
        observed_loads.append(Path(kwargs["path"]))
        return original(**kwargs)

    monkeypatch.setattr(run_module, "_load_artifact", guarded_load_artifact)
    _seal_registration(executor=executor, paths=paths)
    _run_registered_experiment(executor=executor, paths=paths, confirm=True)

    assert len(observed_loads) == len(CASE_ORDER) * 2
    assert all(path.suffix == ".pt" for path in observed_loads)


def test_exact_split_totals_are_aggregated_and_bounded(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    executor = _FakeExecutor(paths)
    _seal_registration(executor=executor, paths=paths)
    summary = _run_registered_experiment(executor=executor, paths=paths, confirm=True)

    actual = summary["actual_query_accounting"]
    for case_id in CASE_ORDER:
        assert actual["splits"][str(case_id)]["discovery"] == dict(
            EXPECTED_QUERY_CEILINGS[case_id]["discovery"]
        )
    summed = {
        name: sum(
            actual["splits"][str(case_id)][split][name]
            for case_id in CASE_ORDER
            for split in ("discovery", "confirmation")
        )
        for name in actual["overall"]
    }
    assert actual["overall"] == summed


def test_query_total_or_ceiling_failure_is_fatal(tmp_path: Path) -> None:
    class OverBudgetExecutor(_FakeExecutor):
        def acquire_confirmation(
            self, case_id: int, panel: object, authorization: dict[str, Any]
        ) -> dict[str, Any]:
            bundle = super().acquire_confirmation(case_id, panel, authorization)
            if case_id == CASE_ORDER[0]:
                stage = bundle["query_accounting"]["resample_set_panel"]
                excess = (
                    EXPECTED_QUERY_CEILINGS[case_id]["confirmation"][
                        "maximum_physical_forwards"
                    ]
                    + 1
                    - bundle["query_accounting"]["total"]["physical_forwards"]
                )
                stage["physical_forwards"] += excess
                bundle["query_accounting"]["total"]["physical_forwards"] += excess
            return bundle

    paths = _paths(tmp_path)
    executor = OverBudgetExecutor(paths)
    _seal_registration(executor=executor, paths=paths)
    with pytest.raises(ValueError, match="query|ceiling"):
        _run_registered_experiment(executor=executor, paths=paths, confirm=True)


def test_source_closure_is_recursive_complete_and_current() -> None:
    required = {
        "experiments/e32_interpbench/e32_4_run.py",
        "experiments/e32_interpbench/e32_4_protocol.py",
        "experiments/e32_interpbench/e32_4_eap.py",
        "experiments/e32_interpbench/e32_4_edge_sets.py",
        "experiments/e32_interpbench/e32_4_scoring.py",
    }
    assert required.issubset(SOURCE_RELATIVE_PATHS)
    assert SOURCE_RELATIVE_PATHS == _source_relative_paths()
    assert set(_source_hashes()) == set(SOURCE_RELATIVE_PATHS)


def test_sealed_e32_3_ancestry_and_source_closure_are_unchanged() -> None:
    record = _verify_e32_3_ancestry()
    assert record["verified"] is True
    assert record["sealed_source_closure_unchanged"] is True
    assert tuple(map(int, record["authorization_sha256"])) == CASE_ORDER


def test_production_input_only_preflight_replays_all_fresh_pools_without_model() -> (
    None
):
    code = """
import json
import experiments.e32_interpbench.e32_4_run as run

def forbidden(*args, **kwargs):
    raise AssertionError('input-only preflight loaded a learned model')

run.load_verified_model = forbidden
record, state = run._ProductionExecutor().preflight_all_input_pools()
run._validate_input_preflight(record)
print(json.dumps({
    'model_calls': record['model_calls'],
    'case_order': list(state),
    'confirmation_capacity': {
        str(case_id): record['cases'][str(case_id)][
            'confirmation_response_blind_capacity_after_all_discovery_raw_rows'
        ]
        for case_id in run.CASE_ORDER
    },
    'all_exclusions_bound': all(
        set(state[case_id].historical_rows).issubset(
            state[case_id].confirmation_excluded_rows
        )
        and set(state[case_id].discovery_raw_rows).issubset(
            state[case_id].confirmation_excluded_rows
        )
        for case_id in run.CASE_ORDER
    ),
}))
"""
    completed = subprocess.run(
        [str(DEFAULT_ENV_DIR / "bin/python"), "-c", code],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)},
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    observed = json.loads(completed.stdout)
    assert observed == {
        "model_calls": 0,
        "case_order": list(CASE_ORDER),
        "confirmation_capacity": {"58": 192, "129": 192, "25": 172},
        "all_exclusions_bound": True,
    }


def test_coordinate_exact_fine_gt_keeps_partial_qkv_and_case25_omission() -> None:
    from experiments.e32_interpbench.adapter import DEFAULT_MODEL_ROOT
    from experiments.e32_interpbench.e32_3_cohort import load_exact_ground_truth
    from experiments.e32_interpbench.e32_4_scoring import (
        exact_fine_ground_truth_ids,
    )

    expected_counts = {58: 3, 129: 4, 25: 3}
    truths: dict[int, tuple[str, ...]] = {}
    bundles: dict[int, dict[str, Any]] = {}
    for case_id in CASE_ORDER:
        path = (
            REPOSITORY_ROOT
            / "runs/e32_3_sparse_core_v2_input_repair"
            / f"case{case_id}_discovery.pt"
        )
        bundle = torch.load(path, map_location="cpu", weights_only=False)["acquisition"]
        bundles[case_id] = bundle
        ground_truth = load_exact_ground_truth(DEFAULT_MODEL_ROOT, case_id)
        induced = tuple(
            fine_name
            for fine_name, whole_index in zip(
                bundle["fine_edge_names"],
                bundle["fine_to_whole_indices"],
                strict=True,
            )
            if set(bundle["whole_edges"][whole_index]).issubset(
                set(run_module.CORE_NODE_SUPPORTS[case_id])
            )
        )
        truths[case_id] = exact_fine_ground_truth_ids(
            case_id=case_id,
            full_fine_edge_ids=bundle["fine_edge_names"],
            induced_fine_edge_ids=induced,
            regular_eap_all_fine_records=bundle["eap_ig5"]["all_fine_scores"],
            official_raw_edges=ground_truth.raw_edges,
        )
        assert len(truths[case_id]) == expected_counts[case_id]

    assert "MLP 1->A2.0.Q" in truths[58]
    assert "MLP 1->A2.0.K" not in truths[58]
    assert "MLP 1->A2.0.V" not in truths[58]
    assert "MLP 0->A1.0.Q" in truths[129]
    assert "MLP 0->A1.0.K" in truths[129]
    assert "MLP 0->A1.0.V" not in truths[129]

    case25 = bundles[25]
    core = set(run_module.CORE_NODE_SUPPORTS[25])
    induced = {
        fine_name
        for fine_name, whole_index in zip(
            case25["fine_edge_names"],
            case25["fine_to_whole_indices"],
            strict=True,
        )
        if set(case25["whole_edges"][whole_index]).issubset(core)
    }
    assert len(set(truths[25]) - induced) == 1


def _canonical_state() -> tuple[bool, bool, tuple[str, ...]]:
    run_files = (
        tuple(
            sorted(
                str(path.relative_to(CANONICAL_PATHS.run_directory))
                for path in CANONICAL_PATHS.run_directory.rglob("*")
            )
        )
        if CANONICAL_PATHS.run_directory.exists()
        else ()
    )
    return (
        CANONICAL_PATHS.registration_path.exists(),
        CANONICAL_PATHS.run_attempt_path.exists(),
        run_files,
    )


def test_direct_and_module_dry_runs_are_no_write_and_cwd_independent(
    tmp_path: Path,
) -> None:
    before = _canonical_state()
    script = REPOSITORY_ROOT / "experiments/e32_interpbench/e32_4_run.py"
    pinned_python = DEFAULT_ENV_DIR / "bin/python"
    direct_environment = dict(os.environ)
    direct_environment.pop("PYTHONPATH", None)
    direct = subprocess.run(
        [str(pinned_python), str(script), "dry-run"],
        cwd=tmp_path,
        env=direct_environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    module_environment = {**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)}
    module = subprocess.run(
        [
            str(pinned_python),
            "-m",
            "experiments.e32_interpbench.e32_4_run",
            "dry-run",
        ],
        cwd=tmp_path,
        env=module_environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert json.loads(direct.stdout)["status"].startswith("dry_run_only")
    assert json.loads(module.stdout)["status"].startswith("dry_run_only")
    assert _canonical_state() == before


def test_public_production_entrypoints_are_noninjectable() -> None:
    assert tuple(inspect.signature(seal_registration).parameters) == ()
    assert tuple(inspect.signature(run_registered_experiment).parameters) == (
        "confirm",
    )
