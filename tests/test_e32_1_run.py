# ruff: noqa: EXE002
"""Custody tests for the lean, canonical E32.1 production driver."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.e32_interpbench.e32_1_protocol import CASE_ORDER
from experiments.e32_interpbench.e32_1_run import (
    CANONICAL_REGISTRATION_PATH,
    CANONICAL_RUN_DIRECTORY,
    SOURCE_RELATIVE_PATHS,
    build_registration_record,
    exclusive_json_write,
    verify_registration_record,
)


def test_registration_binds_canonical_paths_sources_dependencies_and_disclosure() -> (
    None
):
    record = build_registration_record()

    assert record["schema_version"] == "e32_1_exact_gt_registration_v1"
    assert record["canonical_registration_path"] == str(
        CANONICAL_REGISTRATION_PATH.resolve()
    )
    assert record["canonical_run_directory"] == str(CANONICAL_RUN_DIRECTORY.resolve())
    assert tuple(record["case_order"]) == CASE_ORDER
    assert tuple(record["source_sha256"]) == SOURCE_RELATIVE_PATHS
    assert record["protocol"]["claims"][
        "official_gt_and_case_identities_were_inspected_during_design"
    ]
    assert record["protocol"]["claims"]["superiority_to_eap_or_eap_ig_claim"] is False
    assert record["execution"]["device"] == "cpu"
    assert all(record["dependencies"]["tracked_worktrees_clean"].values())
    assert set(record["environment"]["runtime_package_sources"]) == {
        "tracr",
        "iit",
        "transformer_lens",
    }
    verify_registration_record(record)


def test_registration_verifier_rejects_source_or_path_tamper() -> None:
    record = build_registration_record()
    wrong_source = json.loads(json.dumps(record))
    source = SOURCE_RELATIVE_PATHS[0]
    wrong_source["source_sha256"][source] = "0" * 64
    with pytest.raises(ValueError, match="registration differs"):
        verify_registration_record(wrong_source)

    wrong_path = json.loads(json.dumps(record))
    wrong_path["canonical_run_directory"] = "/tmp/cherry-picked-e32"
    with pytest.raises(ValueError, match="registration differs"):
        verify_registration_record(wrong_path)


def test_exclusive_json_write_is_durable_and_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "record.json"

    exclusive_json_write(target, {"answer": 42})

    assert json.loads(target.read_text()) == {"answer": 42}
    with pytest.raises(FileExistsError):
        exclusive_json_write(target, {"answer": 43})


def test_direct_script_dry_run_resolves_repository_imports(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script = repository_root / "experiments" / "e32_interpbench" / "e32_1_run.py"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(script), "dry-run"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == (
        "dry_run_only_no_model_calls_no_writes"
    )
