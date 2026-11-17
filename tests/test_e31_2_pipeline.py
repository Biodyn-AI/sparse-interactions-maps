# ruff: noqa: EXE002
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.e31_sparse_interactions.e31_2_custody import (
    E31_1_PARENT_SCORE_SHA256,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_RUNTIME_ENVIRONMENT,
    SOURCE_PATHS,
    atomic_json_save,
    sha256,
)
from experiments.e31_sparse_interactions.e31_2_orchestration import (
    _context_structure,
    _load_sealed_registration,
    _ModelForwardCounter,
    _require_pristine_output_directory,
    _run_test_pipeline,
    prospective_dry_run,
    run_registered_pipeline,
)
from experiments.e31_sparse_interactions.e31_2_registration import (
    SEALED_STATUS,
    build_registration_record,
    derive_prompt_token_contract,
    validate_registration_payload,
)
from experiments.e31_sparse_interactions.mixed_resolution_panel import (
    build_prompt_manifest,
)
from experiments.e31_sparse_interactions.mixed_resolution_protocol import (
    PROTOCOL_SHA256,
    build_protocol,
)


class _PanelTokenizer:
    def __init__(self) -> None:
        self.prompts: dict[str, list[int]] = {}
        self.targets: dict[str, int] = {}
        next_target = 30_000
        for split_index, split in enumerate(("discovery", "confirmation")):
            records = build_prompt_manifest(split)
            for index in range(0, len(records), 2):
                forward, reverse = records[index : index + 2]
                base = [split_index, index, 101, 202]
                donor = [split_index, index, 303, 202]
                self.prompts[forward["base_prompt"]] = base
                self.prompts[forward["donor_prompt"]] = donor
                assert reverse["base_prompt"] == forward["donor_prompt"]
            for record in records:
                for target in (record["base_target"], record["donor_target"]):
                    if target not in self.targets:
                        self.targets[target] = next_target
                        next_target += 1

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text.startswith(" ") and text.strip() in self.targets:
            return [self.targets[text.strip()]]
        return list(self.prompts[text])


def _source_hashes() -> dict[str, str]:
    return {label: "a" * 64 for label in SOURCE_PATHS}


def _registration(*, sealed: bool = False) -> dict:
    return build_registration_record(
        source_code_sha256=_source_hashes(),
        environment=EXPECTED_RUNTIME_ENVIRONMENT,
        prompt_token_contract=derive_prompt_token_contract(_PanelTokenizer()),
        sealed=sealed,
    )


def _selections() -> dict:
    return {
        "schema_version": "e31_2_discovery_rectangle_selections_v1",
        "parents": [
            {
                "parent_pair_id": parent_id,
                "pair_selected": {
                    "heads_a": [0, 1, 2, 3],
                    "heads_b": [4, 5, 6, 7],
                },
                "singleton_selected": {
                    "heads_a": [0, 2, 4, 6],
                    "heads_b": [1, 3, 5, 7],
                },
            }
            for parent_id in ("l05a_l08a", "l08a_l09a", "l05a_l10a")
        ],
    }


def test_registration_binds_protocol_parent_score_and_exact_budget() -> None:
    registration = _registration()
    assert registration["protocol_sha256"] == PROTOCOL_SHA256
    assert registration["parent_provenance"]["e31_1_v2_score_sha256"] == (
        E31_1_PARENT_SCORE_SHA256
    )
    assert registration["query_accounting"] == build_protocol()["query_accounting"]
    assert registration["query_accounting"]["maximum_registered_model_forwards"] == (
        101_504
    )
    assert registration["outcome_custody"]["all_48_clean_rows_precede_patches"] is True
    assert registration["outcome_custody"]["duplicate_clean_forwards_allowed"] is False
    assert registration["claim_boundary"]["parent_selection_was_posthoc_e31_1"] is True
    assert (
        registration["registered_paths"][
            "alternate_registration_or_output_paths_allowed"
        ]
        is False
    )
    assert registration["output_contract"]["direct_reconstruction_quartet_shape"] == [
        3,
        2,
        32,
        4,
        4,
    ]
    assert (
        validate_registration_payload(registration, verify_external_files=False)
        == registration
    )

    changed = dict(registration)
    changed["query_accounting"] = {
        **registration["query_accounting"],
        "maximum_registered_model_forwards": 101_505,
    }
    with pytest.raises(RuntimeError, match="query accounting"):
        validate_registration_payload(changed, verify_external_files=False)


def test_dry_run_loads_no_model_and_writes_nothing(tmp_path: Path) -> None:
    report = prospective_dry_run(
        tokenizer_factory=lambda _snapshot: _PanelTokenizer(),
        verify_dependencies_fn=lambda: {
            "source_code_sha256": _source_hashes(),
            "checkpoint_sha256": dict(EXPECTED_CHECKPOINT_SHA256),
            "environment": dict(EXPECTED_RUNTIME_ENVIRONMENT),
            "parent_provenance": {"e31_1_v2_score_sha256": E31_1_PARENT_SCORE_SHA256},
        },
    )
    assert report["status"] == "dry_run_no_model_call_no_write"
    assert report["protocol_sha256"] == PROTOCOL_SHA256
    assert report["maximum_registered_model_forwards"] == 101_504
    assert not tuple(tmp_path.iterdir())


def test_context_structure_keeps_inverse_directions_in_one_cluster() -> None:
    records = build_prompt_manifest("discovery")
    strata, clusters = _context_structure(records)
    assert len(set(clusters)) == 8
    for cluster in set(clusters):
        indices = [index for index, value in enumerate(clusters) if value == cluster]
        assert len(indices) == 2
        assert len({strata[index] for index in indices}) == 1

    damaged = [dict(record) for record in records]
    damaged[0]["pair_id"] = damaged[0]["pair_id"].removesuffix("_a_to_b")
    with pytest.raises(RuntimeError, match="direction suffix"):
        _context_structure(damaged)


def test_production_entry_point_rejects_injection_and_alternate_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        run_registered_pipeline(
            registration_path=tmp_path / "registration.json",
            expected_registration_sha256="a" * 64,
            output_directory=tmp_path / "run",
            confirm_model_run=True,
            runner_factory=lambda _registration: object(),  # type: ignore[call-arg]
        )
    with pytest.raises(RuntimeError, match="registration path"):
        run_registered_pipeline(
            registration_path=tmp_path / "registration.json",
            expected_registration_sha256="a" * 64,
            output_directory=tmp_path / "run",
            confirm_model_run=True,
        )


def test_sealed_registration_requires_reservation_and_pristine_output(
    tmp_path: Path,
) -> None:
    registration = _registration(sealed=True)
    copied = tmp_path / "copied_registration.json"
    copied.write_text(json.dumps(registration), encoding="utf-8")
    with pytest.raises(RuntimeError, match="write-once reservation"):
        _load_sealed_registration(copied, sha256(copied))

    output = tmp_path / "output"
    _require_pristine_output_directory(output)
    output.mkdir()
    (output / "._metadata").write_text("ignored", encoding="utf-8")
    _require_pristine_output_directory(output)
    (output / ".global_attempt.json.write-once-reservation").mkdir()
    with pytest.raises(RuntimeError, match="not pristine"):
        _require_pristine_output_directory(output)


def test_outer_model_forward_counter_counts_attempted_batch_examples() -> None:
    class _Model(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return input_ids

    model = _Model()
    counter = _ModelForwardCounter(SimpleNamespace(model=model))
    model(input_ids=torch.ones(3, 4, dtype=torch.long))
    model(input_ids=torch.ones(2, 4, dtype=torch.long))
    assert counter.attempted_examples == 5
    assert counter.attempted_invocations == 2
    with pytest.raises(AttributeError):
        counter.attempted_examples = 0  # type: ignore[misc]
    counter.close()


def test_fake_pipeline_enforces_one_way_write_once_custody(tmp_path: Path) -> None:
    registration = _registration(sealed=True)
    assert registration["status"] == SEALED_STATUS
    registration_path = tmp_path / "registration.json"
    atomic_json_save(registration, registration_path)
    registration_sha256 = sha256(registration_path)
    output = tmp_path / "run"
    events: list[str] = []

    def verify(_registration: dict) -> None:
        events.append("verified")

    def runner_factory(_registration: dict) -> object:
        assert (output / "global_attempt.json").is_file()
        events.append("runner")
        return object()

    def prepare(_runner: object, _registration: dict):
        events.append("clean48")
        return {
            "discovery": "prepared-D",
            "confirmation": "prepared-C",
        }, {
            "schema_version": "fake_clean_screen_v1",
            "all_splits_passed": True,
            "context_count": 48,
            "clean_forward_examples": 96,
        }

    def acquire_discovery(
        _runner: object,
        prepared: str,
        _registration: dict,
        _clean_sha256: str,
    ) -> dict:
        assert prepared == "prepared-D"
        events.append("D")
        return {
            "schema_version": "fake_discovery_v1",
            "fine_quartets": torch.zeros(3, 144, 16, 4, 4, dtype=torch.float64),
            "query_accounting": {
                "atlas_patch_cells": 33_024,
                "batch_smoke_physical_context_mask_cells": 32,
                "clean_forward_examples": 0,
            },
        }

    def fit_geometry(discovery: dict) -> dict:
        assert discovery["schema_version"] == "fake_discovery_v1"
        events.append("geometry")
        return {
            "schema_version": "fake_geometry_v1",
            "status": "discovery_geometry_frozen_before_confirmation",
            "reconstruction_selections": _selections(),
            "matched_controls": {"support_count": 10_000},
        }

    def acquire_confirmation(
        _runner: object,
        prepared: str,
        _registration: dict,
        reconstruction_manifest: dict,
        _clean_sha256: str,
    ) -> dict:
        assert prepared == "prepared-C"
        assert (output / "confirmation_authorization.json").is_file()
        assert reconstruction_manifest["maximum_new_condition_count"] == 72
        events.append("C")
        return {
            "schema_version": "fake_confirmation_v1",
            "fine_quartets": torch.zeros(3, 144, 32, 4, 4, dtype=torch.float64),
            "pair_composites": torch.zeros(3, 32, 4, 4, dtype=torch.float64),
            "singleton_composites": torch.zeros(3, 32, 4, 4, dtype=torch.float64),
            "coarse_composites": torch.zeros(3, 32, 4, 4, dtype=torch.float64),
            "query_accounting": {
                "atlas_patch_cells": 66_048,
                "reconstruction_patch_cells": 2_304,
                "clean_forward_examples": 0,
            },
        }

    def score(discovery: dict, confirmation: dict, geometry: dict) -> dict:
        assert (output / "score_attempt.json").is_file()
        assert discovery["schema_version"] == "fake_discovery_v1"
        assert confirmation["schema_version"] == "fake_confirmation_v1"
        assert geometry["schema_version"] == "fake_geometry_v1"
        events.append("score")
        return {"schema_version": "fake_score_v1", "primary_verdict": "pass"}

    custody = _run_test_pipeline(
        registration_path=registration_path,
        expected_registration_sha256=registration_sha256,
        output_directory=output,
        confirm_model_run=True,
        runner_factory=runner_factory,
        prepare_all_fn=prepare,
        acquire_discovery_fn=acquire_discovery,
        fit_geometry_fn=fit_geometry,
        acquire_confirmation_fn=acquire_confirmation,
        score_fn=score,
        verify_dependencies_fn=verify,
    )
    assert custody["status"] == "test_only_e31_2_complete_write_once_score"
    assert custody["maximum_registered_model_forwards"] == 101_504
    assert events == [
        "verified",
        "runner",
        "clean48",
        "verified",
        "D",
        "verified",
        "geometry",
        "verified",
        "C",
        "verified",
        "geometry",
        "score",
    ]
    for filename in (
        "global_attempt.json",
        "all_registered_clean_screen.pt",
        "discovery_labels.pt",
        "discovery_geometry.pt",
        "confirmation_reconstruction_manifest.json",
        "confirmation_authorization.json",
        "confirmation_labels.pt",
        "score_attempt.json",
        "prospective_score.pt",
        "score_custody.json",
    ):
        assert (output / filename).is_file()
    with pytest.raises(FileExistsError):
        _run_test_pipeline(
            registration_path=registration_path,
            expected_registration_sha256=registration_sha256,
            output_directory=output,
            confirm_model_run=True,
            runner_factory=runner_factory,
            prepare_all_fn=prepare,
            acquire_discovery_fn=acquire_discovery,
            fit_geometry_fn=fit_geometry,
            acquire_confirmation_fn=acquire_confirmation,
            score_fn=score,
            verify_dependencies_fn=verify,
        )


def test_pipeline_rejects_geometry_replay_tamper(tmp_path: Path) -> None:
    # The replay equality helper is exercised through a deliberately changing fit.
    registration = _registration(sealed=True)
    registration_path = tmp_path / "registration.json"
    atomic_json_save(registration, registration_path)
    calls = 0

    def fit(_discovery: dict) -> dict:
        nonlocal calls
        calls += 1
        return {
            "schema_version": "fake_geometry_v1",
            "status": "discovery_geometry_frozen_before_confirmation",
            "reconstruction_selections": _selections(),
            "matched_controls": {"support_count": 10_000},
            "tamper": calls,
        }

    with pytest.raises(RuntimeError, match="geometry replay"):
        _run_test_pipeline(
            registration_path=registration_path,
            expected_registration_sha256=sha256(registration_path),
            output_directory=tmp_path / "run",
            confirm_model_run=True,
            runner_factory=lambda _registration: object(),
            prepare_all_fn=lambda _runner, _registration: (
                {"discovery": None, "confirmation": None},
                {
                    "all_splits_passed": True,
                    "context_count": 48,
                    "clean_forward_examples": 96,
                },
            ),
            acquire_discovery_fn=lambda *_args: {
                "query_accounting": {
                    "atlas_patch_cells": 33_024,
                    "batch_smoke_physical_context_mask_cells": 32,
                    "clean_forward_examples": 0,
                }
            },
            fit_geometry_fn=fit,
            acquire_confirmation_fn=lambda *_args: {
                "query_accounting": {
                    "atlas_patch_cells": 66_048,
                    "reconstruction_patch_cells": 2_304,
                    "clean_forward_examples": 0,
                }
            },
            score_fn=lambda *_args: {},
            verify_dependencies_fn=lambda _registration: None,
        )
