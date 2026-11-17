# ruff: noqa: EXE002
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import experiments.e31_sparse_interactions.order_repair_v2 as repair
from experiments.e30_redundant_causal_interfaces.custody import (
    sha256,
    write_once_reservation_path,
)
from experiments.e31_sparse_interactions.order_repair_v2 import (
    ASSEMBLED_REGIME_ORDER,
    LIFETIME_FORWARD_EXAMPLES,
    REGISTERED_REGIME_ORDER,
    UNIQUE_REGISTERED_WORKLOAD,
    V1_ARTIFACTS,
    V1_CONSUMED_FORWARD_EXAMPLES,
    V2_SOURCE_PATHS,
    _build_registration,
    _nested_equal,
    _score_existing,
    _validate_v2_recovery_chain,
    canonicalize_panel_order,
)


def test_order_repair_changes_only_outer_mapping_order() -> None:
    panels = {key: {"sentinel": object()} for key in ASSEMBLED_REGIME_ORDER}

    repaired = canonicalize_panel_order(panels)

    assert tuple(repaired) == REGISTERED_REGIME_ORDER
    assert set(repaired) == set(panels)
    assert all(repaired[key] is panels[key] for key in REGISTERED_REGIME_ORDER)


def test_order_repair_rejects_any_noncanonical_input_key_order() -> None:
    panels = {key: {} for key in reversed(ASSEMBLED_REGIME_ORDER)}

    with pytest.raises(RuntimeError, match="assembled regime order"):
        canonicalize_panel_order(panels)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_once_reservation_path(path).mkdir()


def _write_torch(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    write_once_reservation_path(path).mkdir()


def _registration(registration_sha256: str) -> dict:
    return {
        "source_code_sha256": {"test/source.py": "a" * 64},
        "v1_failure_state": {
            "artifacts": {
                name: {"path": str(path), "sha256": digest}
                for name, (path, digest) in V1_ARTIFACTS.items()
            }
        },
        "repair": {"operation": "outer_mapping_key_permutation_only"},
        "registration_sha256_for_test": registration_sha256,
    }


def _write_valid_recovery_chain(
    output: Path, *, registration: dict, registration_sha256: str
) -> dict[str, str]:
    paths = {
        "attempt": output / "recovery_attempt.json",
        "geometry": output / "discovery_geometry.pt",
        "geometry_custody": output / "discovery_geometry_custody.json",
        "authorization": output / "confirmation_authorization.json",
        "revalidation": output / "confirmation_clean_revalidation.pt",
        "revalidation_custody": output / "confirmation_clean_revalidation_custody.json",
        "confirmation": output / "confirmation_labels.pt",
        "confirmation_custody": output / "confirmation_custody.json",
    }
    _write_json(
        paths["attempt"],
        {
            "schema_version": "e31_1_order_repair_attempt_v2",
            "status": "attempt_consumed_before_discovery_fit_or_model_load",
            "registration_sha256": registration_sha256,
            "source_code_sha256": registration["source_code_sha256"],
            "parent_artifacts": registration["v1_failure_state"]["artifacts"],
            "actual_lifetime_forward_examples_at_attempt": (
                V1_CONSUMED_FORWARD_EXAMPLES
            ),
            "planned_lifetime_forward_examples_on_success": (LIFETIME_FORWARD_EXAMPLES),
        },
    )
    _write_torch(paths["geometry"], {"sentinel": "geometry"})
    _write_json(
        paths["geometry_custody"],
        {
            "schema_version": "e31_1_order_repair_geometry_custody_v2",
            "status": "geometry_and_controls_frozen_before_patched_confirmation",
            "registration_sha256": registration_sha256,
            "recovery_attempt_sha256": sha256(paths["attempt"]),
            "v1_discovery_labels_sha256": V1_ARTIFACTS["discovery_labels"][1],
            "discovery_geometry_sha256": sha256(paths["geometry"]),
            "matched_control_support_count": 10_000,
            "patched_confirmation_outcomes_used": False,
        },
    )
    _write_json(
        paths["authorization"],
        {
            "schema_version": ("e31_1_order_repair_confirmation_authorization_v2"),
            "status": "patched_confirmation_authorized_after_v2_geometry_seal",
            "registration_sha256": registration_sha256,
            "v1_registration_sha256": V1_ARTIFACTS["registration"][1],
            "v1_discovery_labels_sha256": V1_ARTIFACTS["discovery_labels"][1],
            "discovery_geometry_sha256": sha256(paths["geometry"]),
            "geometry_custody_sha256": sha256(paths["geometry_custody"]),
            "adaptation_after_authorization_allowed": False,
        },
    )
    _write_torch(paths["revalidation"], {"sentinel": "clean-revalidation"})
    _write_json(
        paths["revalidation_custody"],
        {
            "schema_version": ("e31_1_confirmation_clean_revalidation_custody_v2"),
            "status": "bitwise_match_to_v1_before_any_patched_confirmation",
            "registration_sha256": registration_sha256,
            "authorization_sha256": sha256(paths["authorization"]),
            "revalidation_sha256": sha256(paths["revalidation"]),
            "clean_forward_examples": 64,
            "patched_confirmation_examples_before_revalidation": 0,
            "actual_lifetime_forward_examples_after_revalidation": 124_800,
        },
    )
    _write_torch(paths["confirmation"], {"sentinel": "confirmation"})
    _write_json(
        paths["confirmation_custody"],
        {
            "schema_version": "e31_1_order_repair_confirmation_custody_v2",
            "status": "unchanged_v1_confirmation_masks_acquired_write_once",
            "registration_sha256": registration_sha256,
            "v1_registration_sha256": V1_ARTIFACTS["registration"][1],
            "recovery_attempt_sha256": sha256(paths["attempt"]),
            "discovery_geometry_sha256": sha256(paths["geometry"]),
            "authorization_sha256": sha256(paths["authorization"]),
            "revalidation_sha256": sha256(paths["revalidation"]),
            "revalidation_custody_sha256": sha256(paths["revalidation_custody"]),
            "confirmation_labels_sha256": sha256(paths["confirmation"]),
            "confirmation_clean_reprepare_examples": 64,
            "confirmation_patched_examples": 248_832,
            "discovery_rerun_examples": 0,
            "actual_lifetime_forward_examples": LIFETIME_FORWARD_EXAMPLES,
        },
    )
    return {name: sha256(path) for name, path in paths.items()}


def test_v2_recovery_chain_accepts_exact_chain_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    registration_sha256 = "b" * 64
    registration = _registration(registration_sha256)
    expected = _write_valid_recovery_chain(
        tmp_path,
        registration=registration,
        registration_sha256=registration_sha256,
    )

    observed = _validate_v2_recovery_chain(
        output_directory=tmp_path,
        registration=registration,
        registration_sha256=registration_sha256,
    )

    assert observed["v2_geometry_sha256"] == expected["geometry"]
    assert (
        observed["v2_revalidation_custody_sha256"] == expected["revalidation_custody"]
    )
    assert (
        observed["v2_confirmation_custody_sha256"] == expected["confirmation_custody"]
    )

    authorization_path = tmp_path / "confirmation_authorization.json"
    authorization_path.write_text(
        authorization_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="V2 recovery custody chain changed"):
        _validate_v2_recovery_chain(
            output_directory=tmp_path,
            registration=registration,
            registration_sha256=registration_sha256,
        )


def test_v2_recovery_chain_rejects_missing_reservation_and_deleted_artifact(
    tmp_path: Path,
) -> None:
    registration_sha256 = "1" * 64
    registration = _registration(registration_sha256)
    _write_valid_recovery_chain(
        tmp_path,
        registration=registration,
        registration_sha256=registration_sha256,
    )
    geometry_reservation = write_once_reservation_path(
        tmp_path / "discovery_geometry.pt"
    )
    geometry_reservation.rmdir()

    with pytest.raises(RuntimeError, match="reservation is absent"):
        _validate_v2_recovery_chain(
            output_directory=tmp_path,
            registration=registration,
            registration_sha256=registration_sha256,
        )

    geometry_reservation.mkdir()
    (tmp_path / "discovery_geometry.pt").unlink()
    with pytest.raises((FileNotFoundError, RuntimeError)):
        _validate_v2_recovery_chain(
            output_directory=tmp_path,
            registration=registration,
            registration_sha256=registration_sha256,
        )


def test_v2_registration_requires_artifact_and_permanent_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration_path = tmp_path / "registration.json"
    output = tmp_path / "output"
    dependencies = {
        "source_code_sha256": {label: "2" * 64 for label in V2_SOURCE_PATHS},
        "checkpoint_sha256": {"checkpoint": "3" * 64},
        "environment": {"runtime": "test"},
    }
    monkeypatch.setattr(repair, "V2_REGISTRATION_PATH", registration_path)
    monkeypatch.setattr(repair, "V2_RUN_DIRECTORY", output)
    monkeypatch.setattr(repair, "_verify_dependencies", lambda: dependencies)
    registration = _build_registration(dependencies, sealed=True)
    _write_json(registration_path, registration)
    registration_sha256 = sha256(registration_path)

    assert (
        repair._load_registration(registration_path, registration_sha256)
        == registration
    )

    reservation = write_once_reservation_path(registration_path)
    reservation.rmdir()
    with pytest.raises(RuntimeError, match="write-once reservation is absent"):
        repair._load_registration(registration_path, registration_sha256)

    reservation.mkdir()
    registration_path.unlink()
    with pytest.raises((FileNotFoundError, RuntimeError)):
        repair._load_registration(registration_path, registration_sha256)


@pytest.mark.parametrize("dirty_kind", ("artifact", "reservation"))
def test_registration_seal_rejects_nonpristine_v2_output_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dirty_kind: str,
) -> None:
    output = tmp_path / "dirty-output"
    output.mkdir()
    if dirty_kind == "artifact":
        (output / "unexpected.pt").write_bytes(b"unexpected")
    else:
        write_once_reservation_path(output / "unexpected.pt").mkdir()
    registration_path = tmp_path / "registration.json"
    monkeypatch.setattr(repair, "V2_RUN_DIRECTORY", output)
    monkeypatch.setattr(repair, "V2_REGISTRATION_PATH", registration_path)

    def dependencies_must_not_open() -> dict:
        raise AssertionError("dirty-output rejection must precede dependency reads")

    monkeypatch.setattr(repair, "_verify_dependencies", dependencies_must_not_open)

    with pytest.raises(RuntimeError, match="output directory was not pristine"):
        repair.seal_registration(registration_path)
    assert not registration_path.exists()
    assert not write_once_reservation_path(registration_path).exists()


@pytest.mark.parametrize("precreate_empty_output", (False, True))
def test_registration_seal_allows_nonexistent_or_empty_v2_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    precreate_empty_output: bool,
) -> None:
    output = tmp_path / "clean-output"
    if precreate_empty_output:
        output.mkdir()
    registration_path = tmp_path / "registration.json"
    dependencies = {
        "source_code_sha256": {label: "4" * 64 for label in V2_SOURCE_PATHS},
        "checkpoint_sha256": {"checkpoint": "5" * 64},
        "environment": {"runtime": "test"},
    }
    monkeypatch.setattr(repair, "V2_RUN_DIRECTORY", output)
    monkeypatch.setattr(repair, "V2_REGISTRATION_PATH", registration_path)
    monkeypatch.setattr(repair, "_verify_dependencies", lambda: dependencies)

    receipt = repair.seal_registration(registration_path)

    assert receipt["status"] == "v2_order_repair_registration_sealed"
    assert registration_path.is_file()
    assert write_once_reservation_path(registration_path).is_dir()


def test_nested_equal_is_type_shape_dtype_order_and_value_exact() -> None:
    value = {
        "tensor": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "sequence": (True, {"count": 1}),
    }
    _nested_equal(
        value,
        {"tensor": value["tensor"].clone(), "sequence": (True, {"count": 1})},
    )

    mismatches = (
        ({"flag": True}, {"flag": 1}),
        ({"sequence": (1, 2)}, {"sequence": [1, 2]}),
        (
            {"tensor": torch.ones(2, dtype=torch.float64)},
            {"tensor": torch.ones(2, dtype=torch.float32)},
        ),
        ({"first": 1, "second": 2}, {"second": 2, "first": 1}),
    )
    for left, right in mismatches:
        with pytest.raises(RuntimeError, match="frozen geometry"):
            _nested_equal(left, right)


def test_run_repair_custody_avoids_self_hash_and_binds_revalidation_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "v2"
    registration_path = tmp_path / "registration.json"
    registration_sha256 = "c" * 64
    registration = _registration(registration_sha256)
    registration["repair"] = {"operation": "outer_mapping_key_permutation_only"}
    frozen_geometry = {
        "regimes": {
            repair.PRIMARY_REGIME_ID: {"matched_controls": {"support_count": 10_000}}
        }
    }
    revalidation = {"schema_version": "test_revalidation"}

    monkeypatch.setattr(repair, "V2_RUN_DIRECTORY", output)
    monkeypatch.setattr(repair, "V2_REGISTRATION_PATH", registration_path)
    monkeypatch.setattr(repair, "_load_registration", lambda *_args: registration)
    monkeypatch.setattr(repair, "_verify_dependencies", dict)
    monkeypatch.setattr(
        repair,
        "_load_validated_v1_inputs",
        lambda: ({"parent": True}, {"clean": True}, {"discovery": True}),
    )
    monkeypatch.setattr(repair, "_discovery_panels", lambda _discovery: {})
    monkeypatch.setattr(
        repair, "fit_discovery_geometry", lambda **_kwargs: frozen_geometry
    )
    monkeypatch.setattr(
        repair, "_default_runner_factory", lambda _registration: object()
    )
    monkeypatch.setattr(
        repair,
        "_confirmation_clean_revalidation",
        lambda *_args: ((object(),), revalidation),
    )
    monkeypatch.setattr(
        repair,
        "acquire_prepared_split",
        lambda *_args, **_kwargs: {"stage": "confirmation"},
    )
    monkeypatch.setattr(
        repair, "validate_acquired_split", lambda labels, **_kwargs: labels
    )
    monkeypatch.setattr(
        repair,
        "_score_existing",
        lambda **_kwargs: {
            "score_sha256": "d" * 64,
            "atlas_verdict": "pass",
        },
    )

    repair.run_repair(
        registration_path=registration_path,
        registration_sha256=registration_sha256,
        output_directory=output,
    )

    revalidation_custody_path = output / "confirmation_clean_revalidation_custody.json"
    revalidation_custody = json.loads(
        revalidation_custody_path.read_text(encoding="utf-8")
    )
    confirmation_custody = json.loads(
        (output / "confirmation_custody.json").read_text(encoding="utf-8")
    )
    assert "revalidation_custody_sha256" not in revalidation_custody
    assert confirmation_custody["revalidation_custody_sha256"] == sha256(
        revalidation_custody_path
    )


def test_score_attempt_is_durable_before_recovery_chain_opens_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "v2-score"
    registration_path = tmp_path / "registration.json"
    registration_sha256 = "e" * 64
    registration = _registration(registration_sha256)
    _write_valid_recovery_chain(
        output,
        registration=registration,
        registration_sha256=registration_sha256,
    )

    class ChainOpened(RuntimeError):
        pass

    def validate_chain(**_kwargs) -> dict:
        assert (output / "score_attempt.json").is_file()
        raise ChainOpened("outcome chain opened after durable score attempt")

    monkeypatch.setattr(repair, "V2_RUN_DIRECTORY", output)
    monkeypatch.setattr(repair, "V2_REGISTRATION_PATH", registration_path)
    monkeypatch.setattr(repair, "_load_registration", lambda *_args: registration)
    monkeypatch.setattr(repair, "_verify_dependencies", dict)
    monkeypatch.setattr(repair, "_validate_v2_recovery_chain", validate_chain)

    with pytest.raises(ChainOpened, match="after durable score attempt"):
        _score_existing(
            registration_path=registration_path,
            registration_sha256=registration_sha256,
            output_directory=output,
        )


def test_v2_registration_discloses_lifetime_accounting_and_repair_boundary() -> None:
    dependencies = {
        "source_code_sha256": {label: "f" * 64 for label in V2_SOURCE_PATHS},
        "checkpoint_sha256": {"checkpoint": "g" * 64},
        "environment": {"runtime": "test"},
    }

    registration = _build_registration(dependencies, sealed=True)

    assert registration["accounting"] == {
        "v1_consumed_forward_examples": 124_736,
        "v2_confirmation_clean_reprepare_examples": 64,
        "v2_confirmation_patched_examples": 248_832,
        "v2_recovery_forward_examples": 248_896,
        "planned_lifetime_forward_examples_on_success": 373_632,
        "unique_original_registered_workload": UNIQUE_REGISTERED_WORKLOAD,
        "discovery_rerun_examples": 0,
        "second_batch_smoke_examples": 0,
    }
    assert registration["claim_boundary"] == {
        "prospective_relative_to_patched_confirmation_outcomes": True,
        "prospective_relative_to_discovery": False,
        "ordinary_v1_completion": False,
        "post_discovery_operational_repair_disclosed": True,
        "unsealed_real_discovery_geometry_implementation_test": True,
        "preseal_geometry_fields_exposed": (
            "fixed_primary_regime_id_and_successful_10000_control_count_only"
        ),
        "preseal_selected_edge_identities_or_ranks_inspected": False,
        "preseal_matched_control_support_memberships_or_identities_inspected": False,
        "preseal_lambda_curvature_or_effect_energy_values_inspected": False,
        "prediction_topology_direct_communication_global_sparsity": ("not_authorized"),
    }
    assert registration["v2_preseal_state"] == {
        "output_non_appledouble_entries": [],
        "checked_immediately_before_registration_write": True,
    }


def test_v2_dry_run_reports_planned_not_already_consumed_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = {
        "source_code_sha256": {label: "6" * 64 for label in V2_SOURCE_PATHS},
        "checkpoint_sha256": {"checkpoint": "7" * 64},
        "environment": {"runtime": "test"},
    }
    monkeypatch.setattr(repair, "_verify_dependencies", lambda: dependencies)

    report = repair.dry_run()

    assert report["planned_lifetime_forward_examples_on_success"] == 373_632
    assert "actual_lifetime_forward_examples" not in report
