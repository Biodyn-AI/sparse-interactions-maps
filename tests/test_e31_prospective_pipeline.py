# ruff: noqa: EXE002
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from experiments.e29_gpt2_stable_head_interactions.e29_15_ioi_acquisition import (
    CleanPanelViabilityError,
    acquire_ioi_response_panel,
)
from experiments.e31_sparse_interactions.prospective_acquisition import (
    E311DirectedIOIPatchRunner,
    adapt_directed_ioi_record,
    masks_from_split_manifest,
    validate_response_identities,
)
from experiments.e31_sparse_interactions.prospective_custody import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_COMPONENT_MAP,
    EXPECTED_MODEL_REVISION,
    EXPECTED_MODEL_SNAPSHOT,
    EXPECTED_RUNTIME_ENVIRONMENT,
    SOURCE_PATHS,
    atomic_json_save,
    sha256,
)
from experiments.e31_sparse_interactions.prospective_orchestration import (
    PRIMARY_REGIME_ID,
    acquire_registered_atlas,
    acquire_registered_stage,
    prospective_dry_run,
    run_registered_prospective,
    score_registered_atlas,
)
from experiments.e31_sparse_interactions.prospective_protocol import (
    PROTOCOL_SHA256,
    SPLIT_MANIFEST_SHA256,
    build_split_mask_manifest,
)
from experiments.e31_sparse_interactions.prospective_registration import (
    CLEAN_PANEL_GATE,
    PATCH_SEMANTICS,
    REGISTRATION_REQUIRED_KEYS,
    build_registration_record,
    derive_prompt_token_contract,
    validate_registration_payload,
)
from tests.test_e29_15_ioi_patch_runner import (
    _IOIToyModel,
    _record,
    _Tokenizer,
)


class _PanelTokenizer:
    """Tokenizer fixture satisfying every frozen directed prompt contract."""

    def __init__(self) -> None:
        from experiments.e31_sparse_interactions.panel import build_prompt_manifest

        self.prompt_ids: dict[str, list[int]] = {}
        self.target_ids: dict[str, int] = {}
        next_target = 10_000
        for split_index, split in enumerate(("discovery", "confirmation")):
            records = build_prompt_manifest(split)
            for frame_index in range(0, len(records), 2):
                forward, reverse = records[frame_index : frame_index + 2]
                base = [split_index, frame_index, 101, 202]
                donor = [split_index, frame_index, 303, 202]
                self.prompt_ids[forward["base_prompt"]] = base
                self.prompt_ids[forward["donor_prompt"]] = donor
                assert reverse["base_prompt"] == forward["donor_prompt"]
                assert reverse["donor_prompt"] == forward["base_prompt"]
            for record in records:
                for target in (record["base_target"], record["donor_target"]):
                    if target not in self.target_ids:
                        self.target_ids[target] = next_target
                        next_target += 1

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        stripped = text.strip()
        if stripped in self.target_ids and text.startswith(" "):
            return [self.target_ids[stripped]]
        return list(self.prompt_ids[text])


def _source_hashes() -> dict[str, str]:
    return {label: "a" * 64 for label in SOURCE_PATHS}


def _registration() -> dict:
    token_contract = derive_prompt_token_contract(_PanelTokenizer())
    return build_registration_record(
        source_code_sha256=_source_hashes(),
        environment=EXPECTED_RUNTIME_ENVIRONMENT,
        prompt_token_contract=token_contract,
    )


def _reverse_record() -> dict[str, str]:
    record = _record(pair_id="reverse", split="confirmation")
    record.update(
        {
            "name_a": "base",
            "name_b": "donor",
            "base_prompt": "donor prompt",
            "donor_prompt": "base prompt",
            "base_target": "donor",
            "donor_target": "base",
            "base_repeated_subject": "base",
            "donor_repeated_subject": "donor",
        }
    )
    return record


def test_registration_binds_every_frozen_dependency_and_clean_gate() -> None:
    registration = _registration()

    assert set(registration) == set(REGISTRATION_REQUIRED_KEYS)
    assert registration["status"] == (
        "prospective_unsealed_before_any_e31_1_model_response"
    )
    assert registration["model_revision"] == EXPECTED_MODEL_REVISION
    assert registration["model_snapshot"] == str(EXPECTED_MODEL_SNAPSHOT)
    assert registration["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert registration["component_map"] == EXPECTED_COMPONENT_MAP
    assert registration["patch_semantics"] == PATCH_SEMANTICS
    assert registration["clean_panel_gate"] == CLEAN_PANEL_GATE
    assert registration["protocol_sha256"] == PROTOCOL_SHA256
    assert registration["split_manifest_sha256"] == SPLIT_MANIFEST_SHA256
    assert registration["source_code_sha256"] == _source_hashes()
    assert registration["outcome_claims"]["prediction"] == "deferred_to_e31_2"
    assert registration["outcome_claims"]["topology"] == "deferred_to_e31_2"
    assert (
        validate_registration_payload(registration, verify_external_files=False)
        == registration
    )

    changed = deepcopy(registration)
    changed["patch_semantics"] = {**PATCH_SEMANTICS, "unclipped": False}
    with pytest.raises(RuntimeError, match="patch semantics"):
        validate_registration_payload(changed, verify_external_files=False)


def test_role_adapter_and_runner_preserve_both_directed_prompt_semantics() -> None:
    direct = _record(pair_id="direct", split="confirmation")
    reverse = _reverse_record()
    adapted_direct = adapt_directed_ioi_record(direct)
    adapted_reverse = adapt_directed_ioi_record(reverse)

    assert adapted_direct == direct
    assert adapted_reverse["base_prompt"] == reverse["base_prompt"]
    assert adapted_reverse["donor_prompt"] == reverse["donor_prompt"]
    assert adapted_reverse["base_target"] == reverse["base_target"]
    assert adapted_reverse["donor_target"] == reverse["donor_target"]
    assert adapted_reverse["name_a"] == reverse["name_b"]
    assert adapted_reverse["name_b"] == reverse["name_a"]

    runner = E311DirectedIOIPatchRunner(
        _IOIToyModel().eval(), _Tokenizer(), batch_size=2
    )
    direct_prepared = runner.prepare_context(direct)
    reverse_prepared = runner.prepare_context(reverse)
    assert direct_prepared.clean_metadata["base_prompt"] == direct["base_prompt"]
    assert reverse_prepared.clean_metadata["base_prompt"] == reverse["base_prompt"]
    assert reverse_prepared.clean_metadata["base_target"] == reverse["base_target"]
    assert reverse_prepared.base_target_token_id == 2
    assert reverse_prepared.donor_target_token_id == 1


def test_split_masks_are_exact_deduplicated_registered_rows() -> None:
    for split in ("discovery", "confirmation"):
        ids, masks = masks_from_split_manifest(build_split_mask_manifest(split))
        expected = build_split_mask_manifest(split)
        assert len(ids) == expected["unique_condition_count"]
        assert masks.shape == (expected["unique_condition_count"], 24)
        assert len(torch.unique(masks, dim=0)) == len(masks)
        assert tuple(ids) == tuple(
            row["condition_id"] for row in expected["conditions"]
        )


def test_clean_gate_fails_before_any_patched_call_and_never_filters_rows() -> None:
    model = _IOIToyModel(donor_logit_bias=1.0).eval()
    runner = E311DirectedIOIPatchRunner(model, _Tokenizer(), batch_size=2)
    records = (
        _record(pair_id="failed-0", split="confirmation"),
        _record(pair_id="failed-1", split="confirmation"),
    )
    masks = torch.zeros(2, 24, dtype=torch.uint8)
    masks[1] = 1
    forwards = 0

    def count_forward(_module, _inputs):
        nonlocal forwards
        forwards += 1

    handle = model.register_forward_pre_hook(count_forward)
    try:
        with pytest.raises(CleanPanelViabilityError):
            acquire_ioi_response_panel(
                runner,
                records,
                masks,
                stage="confirmation",
                mask_ids=("empty", "full"),
            )
    finally:
        handle.remove()
    assert forwards == 2 * len(records)


def test_raw_and_normalized_identities_are_checked_exactly() -> None:
    runner = E311DirectedIOIPatchRunner(
        _IOIToyModel().eval(), _Tokenizer(), batch_size=2
    )
    masks = torch.zeros(2, 24, dtype=torch.uint8)
    masks[1] = 1
    labels = acquire_ioi_response_panel(
        runner,
        (_record(pair_id="identity", split="confirmation"),),
        masks,
        stage="confirmation",
        mask_ids=("empty", "full"),
    )
    validate_response_identities(labels)

    corrupted = dict(labels)
    corrupted["normalized_responses"] = labels["normalized_responses"].clone()
    corrupted["normalized_responses"][0, 0] += 0.01
    with pytest.raises(RuntimeError, match="normalized response identity"):
        validate_response_identities(corrupted)


def test_acquisition_attempt_is_durable_before_runner_factory(tmp_path: Path) -> None:
    registration = _registration()
    registration["status"] = "prospective_sealed_before_any_e31_1_model_response"
    registration_path = tmp_path / "registration.json"
    atomic_json_save(registration, registration_path)
    registration_sha256 = sha256(registration_path)
    output = tmp_path / "run"
    observed: list[str] = []

    def verify_dependencies(_registration: dict) -> None:
        observed.append("verified")

    def runner_factory(_registration: dict):
        assert (output / "discovery_attempt.json").is_file()
        observed.append("runner")
        return object()

    def acquire_fn(_runner: object, _registration: dict, _stage: str, _digest: str):
        observed.append("acquired")
        return {"all_integrity_gates_passed": True}

    def validate_fn(labels: dict, **_kwargs) -> dict:
        return labels

    custody = acquire_registered_stage(
        stage="discovery",
        registration_path=registration_path,
        expected_registration_sha256=registration_sha256,
        output_directory=output,
        runner_factory=runner_factory,
        acquire_fn=acquire_fn,
        validate_labels_fn=validate_fn,
        verify_dependencies_fn=verify_dependencies,
    )

    assert observed == ["verified", "runner", "acquired"]
    assert custody["status"] == "discovery_labels_acquired_write_once"
    assert (output / "discovery_labels.pt").is_file()
    with pytest.raises(FileExistsError):
        acquire_registered_stage(
            stage="discovery",
            registration_path=registration_path,
            expected_registration_sha256=registration_sha256,
            output_directory=output,
            runner_factory=runner_factory,
            acquire_fn=acquire_fn,
            validate_labels_fn=validate_fn,
            verify_dependencies_fn=verify_dependencies,
        )


def test_dry_run_loads_no_model_and_writes_nothing(tmp_path: Path) -> None:
    report = prospective_dry_run(
        tokenizer_factory=lambda _snapshot: _PanelTokenizer(),
        verify_dependencies_fn=lambda: {
            "source_code_sha256": _source_hashes(),
            "checkpoint_sha256": dict(EXPECTED_CHECKPOINT_SHA256),
            "environment": dict(EXPECTED_RUNTIME_ENVIRONMENT),
        },
    )

    assert report["status"] == "dry_run_no_model_call_no_write"
    assert report["protocol_sha256"] == PROTOCOL_SHA256
    assert report["registration_sha256"]
    assert not tuple(tmp_path.iterdir())


def _sealed_registration(tmp_path: Path) -> tuple[Path, str, dict]:
    registration = _registration()
    registration["status"] = "prospective_sealed_before_any_e31_1_model_response"
    registration_path = tmp_path / "registration.json"
    atomic_json_save(registration, registration_path)
    return registration_path, sha256(registration_path), registration


def _fake_clean_screen(registration: dict) -> tuple[dict, dict]:
    prepared = {
        "discovery": tuple(object() for _ in range(16)),
        "confirmation": tuple(object() for _ in range(32)),
    }
    screen = {
        "schema_version": "e31_1_all_registered_clean_screen_v1",
        "status": "all_48_clean_rows_passed_before_any_patch",
        "split_order": ["discovery", "confirmation"],
        "total_context_count": 48,
        "total_clean_forward_examples": 96,
        "reports": {
            "discovery": {"passed": True, "context_count": 16},
            "confirmation": {"passed": True, "context_count": 32},
        },
        "clean_metadata": {
            split: deepcopy(registration["clean_metadata_contract"][split])
            for split in ("discovery", "confirmation")
        },
        "clean_base_donor_minus_base_margins": {
            "discovery": torch.zeros(16, dtype=torch.float64),
            "confirmation": torch.zeros(32, dtype=torch.float64),
        },
        "clean_donor_donor_minus_base_margins": {
            "discovery": torch.ones(16, dtype=torch.float64),
            "confirmation": torch.ones(32, dtype=torch.float64),
        },
        "clean_endpoints": {
            "discovery": torch.ones(16, dtype=torch.float64),
            "confirmation": torch.ones(32, dtype=torch.float64),
        },
        "gate": deepcopy(registration["clean_panel_gate"]),
        "row_policy": "retain_and_score_every_registered_context",
        "all_splits_passed": True,
    }
    return prepared, screen


def _fake_split_labels(split: str) -> dict:
    manifest = build_split_mask_manifest(split)
    context_count = 16 if split == "discovery" else 32
    smoke_cells = 32 if split == "discovery" else 0
    return {
        "stage": split,
        "normalized_responses": torch.zeros((1, 1), dtype=torch.float64),
        "prompt_records": (),
        "query_accounting": {
            "context_count": context_count,
            "mask_rows": manifest["unique_condition_count"],
            "patched_context_mask_cells": (
                context_count * manifest["unique_condition_count"]
            ),
            "clean_forward_examples_are_in_global_screen": True,
            "batch_smoke_mask_rows": 16 if split == "discovery" else 0,
            "batch_smoke_physical_context_mask_cells": smoke_cells,
        },
    }


def _patch_fast_official_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
    *,
    registration: dict,
    output: Path,
    events: list[str],
) -> None:
    import experiments.e31_sparse_interactions.prospective_orchestration as module

    def verify(_registration: dict) -> None:
        events.append("verify")

    def runner_factory(_registration: dict) -> object:
        assert (output / "global_attempt.json").is_file()
        events.append("runner")
        return object()

    def screen(_runner: object) -> tuple[dict, dict]:
        events.append("clean_all_48_once")
        return _fake_clean_screen(registration)

    def acquire(
        _runner: object,
        _prepared: dict,
        *,
        split: str,
        registration_sha256: str,
        clean_screen_sha256: str,
    ) -> dict:
        assert registration_sha256
        assert clean_screen_sha256
        if split == "discovery":
            assert (output / "all_registered_clean_screen.pt").is_file()
            events.append("discovery_smoke_then_full")
        else:
            assert (output / "discovery_geometry.pt").is_file()
            assert (output / "confirmation_authorization.json").is_file()
            events.append("confirmation")
        return _fake_split_labels(split)

    def validate(labels: dict, **_kwargs) -> dict:
        return labels

    def assemble(**kwargs) -> dict:
        events.append(f"assemble_{kwargs['prompt_records']!r}")
        return {PRIMARY_REGIME_ID: {}}

    def fit(**_kwargs) -> dict:
        events.append("fit_geometry")
        return {
            "primary_regime_id": PRIMARY_REGIME_ID,
            "regimes": {
                PRIMARY_REGIME_ID: {"matched_controls": {"support_count": 10_000}}
            },
        }

    def score(**_kwargs) -> dict:
        events.append("score")
        return {
            "atlas_verdict": "pass",
            "primary_verdict": "pass",
            "promotion_decision": "eligible_only_for_separately_registered_e31_2",
        }

    monkeypatch.setattr(module, "screen_all_registered_clean_rows", screen)
    monkeypatch.setattr(module, "acquire_prepared_split", acquire)
    monkeypatch.setattr(module, "validate_acquired_split", validate)
    monkeypatch.setattr(module, "assemble_regime_panels", assemble)
    monkeypatch.setattr(module, "fit_discovery_geometry", fit)
    monkeypatch.setattr(module, "score_with_frozen_geometry", score)
    monkeypatch.setattr(module, "_default_runner_factory", runner_factory)
    monkeypatch.setattr(module, "verify_registered_dependencies", verify)


def test_official_single_run_orders_and_custodies_the_full_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration_path, registration_sha256, registration = _sealed_registration(
        tmp_path
    )
    output = tmp_path / "official"
    events: list[str] = []
    _patch_fast_official_roundtrip(
        monkeypatch, registration=registration, output=output, events=events
    )

    receipt = run_registered_prospective(
        registration_path=registration_path,
        expected_registration_sha256=registration_sha256,
        output_directory=output,
        confirm_model_run=True,
    )

    assert receipt["status"] == "prospective_atlas_acquired_and_scored_write_once"
    assert receipt["maximum_registered_model_forwards"] == 373_568
    assert events == [
        "verify",
        "runner",
        "clean_all_48_once",
        "discovery_smoke_then_full",
        "verify",
        "assemble_()",
        "fit_geometry",
        "verify",
        "confirmation",
        "verify",
        "assemble_()",
        "assemble_()",
        "score",
    ]
    assert (output / "prospective_score.pt").is_file()
    assert (output / "score_custody.json").is_file()
    assert not (output / "run_failure_custody.json").exists()


def test_scoring_rejects_a_tampered_confirmation_authorization_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration_path, registration_sha256, registration = _sealed_registration(
        tmp_path
    )
    output = tmp_path / "tampered"
    events: list[str] = []
    _patch_fast_official_roundtrip(
        monkeypatch, registration=registration, output=output, events=events
    )
    acquisition = acquire_registered_atlas(
        registration_path=registration_path,
        expected_registration_sha256=registration_sha256,
        output_directory=output,
        confirm_model_run=True,
        runner_factory=lambda _registration: object(),
        verify_dependencies_fn=lambda _registration: None,
    )
    authorization_path = output / "confirmation_authorization.json"
    authorization_path.write_text(
        authorization_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="confirmation authorization hash"):
        score_registered_atlas(
            registration_path=registration_path,
            expected_registration_sha256=registration_sha256,
            output_directory=output,
            expected_discovery_labels_sha256=acquisition["discovery_labels_sha256"],
            expected_geometry_sha256=acquisition["discovery_geometry_sha256"],
            expected_confirmation_labels_sha256=acquisition[
                "confirmation_labels_sha256"
            ],
            confirm_score=True,
            verify_dependencies_fn=lambda _registration: None,
        )


def test_official_clean_failure_is_custodied_before_any_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.e31_sparse_interactions.prospective_orchestration as module

    registration_path, registration_sha256, _registration = _sealed_registration(
        tmp_path
    )
    output = tmp_path / "clean-failure"

    def fail_clean(_runner: object) -> None:
        raise CleanPanelViabilityError({"passed": False, "reason": "synthetic"})

    monkeypatch.setattr(module, "screen_all_registered_clean_rows", fail_clean)
    with pytest.raises(CleanPanelViabilityError):
        acquire_registered_atlas(
            registration_path=registration_path,
            expected_registration_sha256=registration_sha256,
            output_directory=output,
            confirm_model_run=True,
            runner_factory=lambda _registration: object(),
            verify_dependencies_fn=lambda _registration: None,
        )

    failure = torch.load(
        output / "clean_screen_failure.pt", map_location="cpu", weights_only=True
    )
    custody = json.loads(
        (output / "clean_screen_failure_custody.json").read_text(encoding="utf-8")
    )
    assert failure["exception_type"] == "CleanPanelViabilityError"
    assert custody["patched_response_count"] == 0
    assert custody["failure_sha256"] == sha256(output / "clean_screen_failure.pt")
    assert not (output / "all_registered_clean_screen.pt").exists()
    assert not (output / "discovery_labels.pt").exists()
