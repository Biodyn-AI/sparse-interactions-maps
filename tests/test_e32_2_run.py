# ruff: noqa: EXE002
"""Custody tests for the canonical E32.2 production driver."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from experiments.e32_interpbench.adapter import DEFAULT_ENV_DIR
from experiments.e32_interpbench.e32_2_run import (
    ACQUISITION_FILENAMES,
    CANONICAL_REGISTRATION_PATH,
    CANONICAL_RUN_ATTEMPT_PATH,
    CANONICAL_RUN_DIRECTORY,
    CUSTODY_FILENAMES,
    SOURCE_RELATIVE_PATHS,
    _all_installed_distributions,
    _environment_record,
    build_registration_record,
    exclusive_json_write,
    load_score_stage_artifacts,
    reserve_run_attempt,
    reserve_score_stage,
    run_registered_experiment,
    verify_registration_record,
    verify_run_attempt,
    verify_score_stage_reservation,
)


def test_registration_binds_exact_canonical_paths_and_complete_source_closure() -> None:
    record = build_registration_record()

    assert record["schema_version"] == "e32_2_node_first_directed_registration_v1"
    assert CANONICAL_REGISTRATION_PATH.name == (
        "E32.2-NODE-FIRST-DIRECTED-REGISTRATION.json"
    )
    assert CANONICAL_RUN_DIRECTORY.name == "e32_2_node_first_directed_v1"
    assert CANONICAL_RUN_ATTEMPT_PATH.name == (
        "E32.2-NODE-FIRST-DIRECTED-RUN-ATTEMPT.json"
    )
    assert record["canonical_registration_path"] == str(
        CANONICAL_REGISTRATION_PATH.resolve()
    )
    assert record["canonical_run_directory"] == str(CANONICAL_RUN_DIRECTORY.resolve())
    assert tuple(record["source_sha256"]) == SOURCE_RELATIVE_PATHS
    assert {
        "experiments/e32_interpbench/e32_2_protocol.py",
        "experiments/e32_interpbench/e32_2_directed_replacement.py",
        "experiments/e32_interpbench/e32_2_acquisition.py",
        "experiments/e32_interpbench/e32_2_scoring.py",
        "experiments/e32_interpbench/e32_2_orchestration.py",
        "experiments/e32_interpbench/e32_2_run.py",
    }.issubset(record["source_sha256"])
    assert tuple(record["protocol"]["case_order"]) == (2, 21)
    assert record["protocol"]["claims"]["cases_and_gt_were_inspected"] is True
    assert record["execution"]["all_cases_mandatory"] is True
    assert (
        record["execution"][
            "score_attempt_precedes_score_stage_opening_acquisition_outcomes"
        ]
        is True
    )
    verify_registration_record(record)


def test_registration_verifier_rejects_source_path_or_protocol_tamper() -> None:
    record = build_registration_record()
    source = SOURCE_RELATIVE_PATHS[0]
    for mutate in (
        lambda value: value["source_sha256"].__setitem__(source, "0" * 64),
        lambda value: value.__setitem__("canonical_run_directory", "/tmp/e32"),
        lambda value: value["protocol"].__setitem__("case_order", [21, 2]),
    ):
        wrong = json.loads(json.dumps(record))
        mutate(wrong)
        with pytest.raises(ValueError, match="registration differs"):
            verify_registration_record(wrong)


def test_distribution_closure_preserves_conflicting_duplicate_metadata(
    tmp_path: Path,
) -> None:
    class Distribution:
        def __init__(self, name: str, version: str, path: Path) -> None:
            self.metadata = {"Name": name}
            self.version = version
            self._path = path

    class AppleDoubleDistribution:
        _path = tmp_path / "._broken.dist-info"

        @property
        def metadata(self) -> object:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "sidecar")

    duplicate = (
        Distribution("Qt.Console", "5.0", tmp_path / "new.dist-info"),
        Distribution("qt_console", "4.0", tmp_path / "old.dist-info"),
        AppleDoubleDistribution(),
    )
    with patch(
        "experiments.e32_interpbench.e32_2_run.importlib.metadata.distributions",
        return_value=duplicate,
    ):
        observed = _all_installed_distributions()

    assert tuple(record["version"] for record in observed["qt-console"]) == (
        "4.0",
        "5.0",
    )


def test_environment_record_ignores_official_loader_pygraphviz_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = build_registration_record()
    before = registration["environment"]

    stub = types.ModuleType("pygraphviz")
    assert stub.__spec__ is None
    monkeypatch.setitem(sys.modules, "pygraphviz", stub)

    after = _environment_record()
    assert after == before
    verify_registration_record(registration)
    assert after["pygraphviz_branch"]["availability_semantics"] == (
        "environment_path_spec_ignoring_runtime_sys_modules_stubs"
    )


def test_exclusive_json_write_refuses_any_existing_reservation(tmp_path: Path) -> None:
    target = tmp_path / "attempt.json"
    exclusive_json_write(target, {"stage": "acquisition"})

    assert json.loads(target.read_text()) == {"stage": "acquisition"}
    with pytest.raises(FileExistsError):
        exclusive_json_write(target, {"stage": "replacement"})


def test_score_stage_must_be_reserved_once_after_all_artifact_hashes_exist(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    hashes = {}
    for index, filename in enumerate(ACQUISITION_FILENAMES):
        payload = f"artifact-{index}".encode()
        (run_directory / filename).write_bytes(payload)
        hashes[filename] = hashlib.sha256(payload).hexdigest()
    custody_hashes = {}
    for index, filename in enumerate(CUSTODY_FILENAMES):
        payload = f"custody-{index}".encode()
        (run_directory / filename).write_bytes(payload)
        custody_hashes[filename] = hashlib.sha256(payload).hexdigest()

    reservation = reserve_score_stage(
        run_directory=run_directory,
        registration_sha256="a" * 64,
        run_attempt_sha256="b" * 64,
        acquisition_artifact_sha256=hashes,
        custody_artifact_sha256=custody_hashes,
    )

    assert reservation.path.name == "score_attempt.json"
    assert reservation.acquisition_artifact_sha256 == hashes
    verify_score_stage_reservation(reservation)
    with pytest.raises(FileExistsError):
        reserve_score_stage(
            run_directory=run_directory,
            registration_sha256="a" * 64,
            run_attempt_sha256="b" * 64,
            acquisition_artifact_sha256=hashes,
            custody_artifact_sha256=custody_hashes,
        )
    reservation.path.write_text("{}")
    with pytest.raises(ValueError, match="score-stage reservation"):
        verify_score_stage_reservation(reservation)


def test_external_run_attempt_survives_deleted_run_directory(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt.json"
    run_directory = tmp_path / "run"

    first_hash = reserve_run_attempt(
        path=attempt,
        registration_sha256="a" * 64,
        run_directory=run_directory,
    )

    assert len(first_hash) == 64
    assert not run_directory.exists()
    verify_run_attempt(
        path=attempt,
        registration_sha256="a" * 64,
        run_directory=run_directory,
        expected_sha256=first_hash,
    )
    with pytest.raises(FileExistsError):
        reserve_run_attempt(
            path=attempt,
            registration_sha256="a" * 64,
            run_directory=run_directory,
        )
    attempt.write_text("{}")
    with pytest.raises(ValueError, match="run-attempt reservation changed"):
        verify_run_attempt(
            path=attempt,
            registration_sha256="a" * 64,
            run_directory=run_directory,
            expected_sha256=first_hash,
        )
    attempt.unlink()
    with pytest.raises(ValueError, match="missing or malformed"):
        verify_run_attempt(
            path=attempt,
            registration_sha256="a" * 64,
            run_directory=run_directory,
            expected_sha256=first_hash,
        )


def test_score_attempt_precedes_outcome_load_and_binds_authorization_chain(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    registration_hash = "a" * 64
    run_attempt_hash = "b" * 64
    artifact_hashes: dict[str, str] = {}
    custody_hashes: dict[str, str] = {}

    for case_id in (2, 21):
        nodes = (f"case{case_id}_node",)
        edges = ((nodes[0], nodes[0]),)
        discovery_name = f"case{case_id}_discovery.pt"
        discovery = {
            "schema_version": "e32_2_split_artifact_v1",
            "registration_sha256": registration_hash,
            "run_attempt_sha256": run_attempt_hash,
            "case_id": case_id,
            "split": "discovery",
            "confirmation_authorization_sha256": None,
            "acquisition": {
                "discovery_frozen_nodes": nodes,
                "discovery_induced_edges": edges,
            },
        }
        torch.save(discovery, run_directory / discovery_name)
        artifact_hashes[discovery_name] = hashlib.sha256(
            (run_directory / discovery_name).read_bytes()
        ).hexdigest()

        discovery_custody_name = f"case{case_id}_discovery_custody.json"
        exclusive_json_write(
            run_directory / discovery_custody_name,
            {"case_id": case_id, "split": "discovery"},
        )
        custody_hashes[discovery_custody_name] = hashlib.sha256(
            (run_directory / discovery_custody_name).read_bytes()
        ).hexdigest()
        authorization_name = f"case{case_id}_confirmation_authorization.json"
        exclusive_json_write(
            run_directory / authorization_name,
            {
                "schema_version": "e32_2_confirmation_authorization_v1",
                "case_id": case_id,
                "registration_sha256": registration_hash,
                "run_attempt_sha256": run_attempt_hash,
                "discovery_artifact_sha256": artifact_hashes[discovery_name],
                "discovery_custody_sha256": custody_hashes[discovery_custody_name],
                "frozen_discovery_nodes": nodes,
                "frozen_induced_edges": edges,
            },
        )
        authorization_hash = hashlib.sha256(
            (run_directory / authorization_name).read_bytes()
        ).hexdigest()
        custody_hashes[authorization_name] = authorization_hash

        confirmation_name = f"case{case_id}_confirmation.pt"
        confirmation = {
            "schema_version": "e32_2_split_artifact_v1",
            "registration_sha256": registration_hash,
            "run_attempt_sha256": run_attempt_hash,
            "case_id": case_id,
            "split": "confirmation",
            "confirmation_authorization_sha256": authorization_hash,
            "acquisition": {
                "discovery_frozen_nodes": nodes,
                "discovery_induced_edges": edges,
                "confirmation_authorization_sha256": authorization_hash,
            },
        }
        torch.save(confirmation, run_directory / confirmation_name)
        artifact_hashes[confirmation_name] = hashlib.sha256(
            (run_directory / confirmation_name).read_bytes()
        ).hexdigest()
        confirmation_custody_name = f"case{case_id}_confirmation_custody.json"
        exclusive_json_write(
            run_directory / confirmation_custody_name,
            {"case_id": case_id, "split": "confirmation"},
        )
        custody_hashes[confirmation_custody_name] = hashlib.sha256(
            (run_directory / confirmation_custody_name).read_bytes()
        ).hexdigest()

    exclusive_json_write(run_directory / "acquisition_custody.json", {"complete": True})
    custody_hashes["acquisition_custody.json"] = hashlib.sha256(
        (run_directory / "acquisition_custody.json").read_bytes()
    ).hexdigest()
    assert tuple(artifact_hashes) == ACQUISITION_FILENAMES
    assert tuple(custody_hashes) == CUSTODY_FILENAMES

    with patch("torch.load") as outcome_loader:
        with pytest.raises(TypeError, match="reservation is required"):
            load_score_stage_artifacts(None)
        outcome_loader.assert_not_called()

    reservation = reserve_score_stage(
        run_directory=run_directory,
        registration_sha256=registration_hash,
        run_attempt_sha256=run_attempt_hash,
        acquisition_artifact_sha256=artifact_hashes,
        custody_artifact_sha256=custody_hashes,
    )
    loaded = load_score_stage_artifacts(reservation)
    assert tuple(loaded) == (
        (2, "discovery"),
        (2, "confirmation"),
        (21, "discovery"),
        (21, "confirmation"),
    )

    (run_directory / "case21_confirmation_authorization.json").write_text("{}")
    with pytest.raises(ValueError, match="custody artifact changed"):
        load_score_stage_artifacts(reservation)


def test_production_run_has_no_scientific_callback_injection_points() -> None:
    assert tuple(inspect.signature(run_registered_experiment).parameters) == (
        "confirm",
    )


def test_direct_script_dry_run_resolves_repository_imports(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script = repository_root / "experiments" / "e32_interpbench" / "e32_2_run.py"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    pinned_python = DEFAULT_ENV_DIR / "bin" / "python"
    canonical_before = {
        path: path.exists()
        for path in (
            CANONICAL_REGISTRATION_PATH,
            CANONICAL_RUN_ATTEMPT_PATH,
            CANONICAL_RUN_DIRECTORY,
        )
    }
    temporary_before = tuple(tmp_path.iterdir())

    completed = subprocess.run(
        [str(pinned_python), str(script), "dry-run"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "dry_run_only_no_model_calls_no_writes"
    assert tuple(payload["protocol"]["case_order"]) == (2, 21)

    module_environment = dict(environment)
    module_environment["PYTHONPATH"] = str(repository_root)
    module_completed = subprocess.run(
        [
            str(pinned_python),
            "-m",
            "experiments.e32_interpbench.e32_2_run",
            "dry-run",
        ],
        cwd=tmp_path,
        env=module_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert module_completed.returncode == 0, module_completed.stderr
    assert json.loads(module_completed.stdout) == payload
    assert tuple(tmp_path.iterdir()) == temporary_before
    assert {path: path.exists() for path in canonical_before} == canonical_before
