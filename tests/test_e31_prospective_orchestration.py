# ruff: noqa: EXE002
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.e31_sparse_interactions.prospective_custody import (
    atomic_json_save,
    sha256,
)
from experiments.e31_sparse_interactions.prospective_orchestration import (
    _validate_score_custody_chain,
    _validate_total_query_accounting,
)
from experiments.e31_sparse_interactions.prospective_protocol import PROTOCOL_SHA256

REGISTRATION_SHA256 = "1" * 64
DISCOVERY_LABELS_SHA256 = "2" * 64
GEOMETRY_SHA256 = "3" * 64
CONFIRMATION_LABELS_SHA256 = "4" * 64
CLEAN_SCREEN_SHA256 = "5" * 64


def _write_chain(output: Path) -> dict[str, str]:
    attempt = {
        "schema_version": "e31_1_global_acquisition_attempt_v1",
        "status": "global_attempt_consumed_before_tokenizer_or_model_load",
        "registration_sha256": REGISTRATION_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "maximum_registered_model_forwards": 373_568,
    }
    atomic_json_save(attempt, output / "global_attempt.json")
    attempt_sha256 = sha256(output / "global_attempt.json")
    clean_custody = {
        "schema_version": "e31_1_global_clean_screen_custody_v1",
        "status": "all_48_clean_rows_passed_before_any_patched_response",
        "registration_sha256": REGISTRATION_SHA256,
        "global_attempt_sha256": attempt_sha256,
        "clean_screen_sha256": CLEAN_SCREEN_SHA256,
        "context_count": 48,
        "clean_forward_examples": 96,
    }
    atomic_json_save(clean_custody, output / "all_registered_clean_screen_custody.json")
    discovery_custody = {
        "schema_version": "e31_1_split_acquisition_custody_v1",
        "status": "discovery_labels_acquired_write_once",
        "split": "discovery",
        "registration_sha256": REGISTRATION_SHA256,
        "global_attempt_sha256": attempt_sha256,
        "clean_screen_sha256": CLEAN_SCREEN_SHA256,
        "labels_sha256": DISCOVERY_LABELS_SHA256,
        "discovery_geometry_sha256": None,
        "confirmation_authorization_sha256": None,
        "all_registered_rows_retained": True,
        "patched_response_clipping": False,
    }
    atomic_json_save(discovery_custody, output / "discovery_custody.json")
    geometry_custody = {
        "schema_version": "e31_1_discovery_geometry_custody_v1",
        "status": "geometry_and_controls_frozen_before_confirmation",
        "registration_sha256": REGISTRATION_SHA256,
        "discovery_labels_sha256": DISCOVERY_LABELS_SHA256,
        "discovery_geometry_sha256": GEOMETRY_SHA256,
        "matched_control_support_count": 10_000,
        "confirmation_outcomes_used": False,
    }
    atomic_json_save(geometry_custody, output / "discovery_geometry_custody.json")
    geometry_custody_sha256 = sha256(output / "discovery_geometry_custody.json")
    authorization = {
        "schema_version": "e31_1_confirmation_authorization_v1",
        "status": "confirmation_authorized_after_discovery_geometry_seal",
        "registration_sha256": REGISTRATION_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "discovery_labels_sha256": DISCOVERY_LABELS_SHA256,
        "discovery_geometry_sha256": GEOMETRY_SHA256,
        "discovery_geometry_custody_sha256": geometry_custody_sha256,
        "confirmation_adaptation_allowed": False,
    }
    atomic_json_save(authorization, output / "confirmation_authorization.json")
    authorization_sha256 = sha256(output / "confirmation_authorization.json")
    confirmation_custody = {
        "schema_version": "e31_1_split_acquisition_custody_v1",
        "status": "confirmation_labels_acquired_write_once",
        "split": "confirmation",
        "registration_sha256": REGISTRATION_SHA256,
        "global_attempt_sha256": attempt_sha256,
        "clean_screen_sha256": CLEAN_SCREEN_SHA256,
        "labels_sha256": CONFIRMATION_LABELS_SHA256,
        "discovery_geometry_sha256": GEOMETRY_SHA256,
        "confirmation_authorization_sha256": authorization_sha256,
        "all_registered_rows_retained": True,
        "patched_response_clipping": False,
    }
    atomic_json_save(confirmation_custody, output / "confirmation_custody.json")
    return {
        "global_attempt_sha256": attempt_sha256,
        "clean_screen_sha256": CLEAN_SCREEN_SHA256,
        "discovery_custody_sha256": sha256(output / "discovery_custody.json"),
        "discovery_geometry_custody_sha256": geometry_custody_sha256,
        "confirmation_authorization_sha256": authorization_sha256,
        "confirmation_custody_sha256": sha256(output / "confirmation_custody.json"),
    }


def _validate(output: Path) -> dict[str, str]:
    return _validate_score_custody_chain(
        output_directory=output,
        registration_sha256=REGISTRATION_SHA256,
        discovery_labels_sha256=DISCOVERY_LABELS_SHA256,
        discovery_geometry_sha256=GEOMETRY_SHA256,
        confirmation_labels_sha256=CONFIRMATION_LABELS_SHA256,
    )


def test_score_custody_chain_accepts_exact_discovery_then_confirmation_order(
    tmp_path: Path,
) -> None:
    expected = _write_chain(tmp_path)

    assert _validate(tmp_path) == expected


@pytest.mark.parametrize(
    ("filename", "field", "changed"),
    (
        ("discovery_custody.json", "labels_sha256", "9" * 64),
        ("discovery_geometry_custody.json", "confirmation_outcomes_used", True),
        ("confirmation_authorization.json", "discovery_geometry_sha256", "9" * 64),
        ("confirmation_custody.json", "confirmation_authorization_sha256", "9" * 64),
    ),
)
def test_score_custody_chain_rejects_every_broken_link(
    tmp_path: Path, filename: str, field: str, changed: object
) -> None:
    _write_chain(tmp_path)
    path = tmp_path / filename
    record = json.loads(path.read_text(encoding="utf-8"))
    corrupted = deepcopy(record)
    corrupted[field] = changed
    path.write_text(json.dumps(corrupted), encoding="utf-8")

    with pytest.raises(RuntimeError, match="custody chain"):
        _validate(tmp_path)


def test_total_query_accounting_is_recomputed_from_opened_artifacts() -> None:
    clean = {"total_clean_forward_examples": 96}
    discovery = {
        "query_accounting": {
            "patched_context_mask_cells": 124_608,
            "batch_smoke_physical_context_mask_cells": 32,
        }
    }
    confirmation = {
        "query_accounting": {
            "patched_context_mask_cells": 248_832,
            "batch_smoke_physical_context_mask_cells": 0,
        }
    }

    assert _validate_total_query_accounting(clean, discovery, confirmation) == 373_568
    discovery["query_accounting"]["patched_context_mask_cells"] -= 1
    with pytest.raises(RuntimeError, match="query accounting"):
        _validate_total_query_accounting(clean, discovery, confirmation)
