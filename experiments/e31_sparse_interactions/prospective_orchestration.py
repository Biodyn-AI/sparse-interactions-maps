# ruff: noqa: EXE002
"""Write-once orchestration for the prospective E31.1 sparse atlas.

The official acquisition is intentionally one process.  It prepares the 48
clean prompt contexts once, then reuses those exact contexts for discovery and
confirmation.  This preserves the registered 96-clean-forward accounting and
prevents confirmation from opening before discovery geometry is frozen.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_15_ioi_acquisition import (
    CleanPanelViabilityError,
)
from experiments.e31_sparse_interactions.panel import canonical_sha256
from experiments.e31_sparse_interactions.prospective_acquisition import (
    E311DirectedIOIPatchRunner,
    acquire_prepared_split,
    screen_all_registered_clean_rows,
    validate_acquired_split,
)
from experiments.e31_sparse_interactions.prospective_custody import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_MODEL_SNAPSHOT,
    REPOSITORY_ROOT,
    atomic_json_save,
    atomic_torch_save,
    ensure_write_once_available,
    sha256,
    valid_sha256,
    verify_file_hash,
    verify_fixed_dependencies,
    verify_registered_dependencies,
)
from experiments.e31_sparse_interactions.prospective_protocol import (
    PROTOCOL_SHA256,
    build_prospective_protocol,
    build_split_mask_manifest,
)
from experiments.e31_sparse_interactions.prospective_registration import (
    SEALED_STATUS,
    build_registration_record,
    derive_prompt_token_contract,
    validate_registration_payload,
)
from experiments.e31_sparse_interactions.prospective_scoring import (
    assemble_regime_panels,
    fit_discovery_geometry,
    score_with_frozen_geometry,
)

Split = Literal["discovery", "confirmation"]
DEFAULT_REGISTRATION_PATH = (
    REPOSITORY_ROOT / "notes/E31.1-PROSPECTIVE-SPARSE-ATLAS-REGISTRATION.json"
)
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "runs/e31_1_prospective_sparse_atlas_v1"
PRIMARY_REGIME_ID = "nonhub_k10_hub_off"


def _default_tokenizer_factory(snapshot: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)


def _default_runner_factory(
    registration: Mapping[str, Any],
) -> E311DirectedIOIPatchRunner:
    """Load GPT-2 only after the durable global attempt has been written."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = Path(registration["model_snapshot"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        attn_implementation="eager",
        torch_dtype=torch.float64,
    ).to(device="cpu", dtype=torch.float64)
    model.eval()
    return E311DirectedIOIPatchRunner(
        model,
        tokenizer,
        batch_size=int(registration["execution"]["batch_size"]),
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _load_torch_object(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"expected a mapping Torch artifact in {path}")
    return value


def _load_sealed_registration(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not valid_sha256(expected_sha256):
        raise ValueError("expected registration SHA-256 is invalid")
    verify_file_hash(path, expected_sha256)
    registration = validate_registration_payload(
        _read_json_object(path), verify_external_files=False
    )
    if registration["status"] != SEALED_STATUS:
        raise RuntimeError("acquisition and scoring require a sealed registration")
    return registration


def _verify_dependency_report(report: Mapping[str, Any]) -> None:
    if report.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("verified checkpoint closure differs from GPT-2 small")
    if not isinstance(report.get("source_code_sha256"), Mapping):
        raise TypeError("verified source closure is absent")


def prospective_dry_run(
    *,
    tokenizer_factory: Callable[[Path], Any] = _default_tokenizer_factory,
    verify_dependencies_fn: Callable[[], Mapping[str, Any]] = verify_fixed_dependencies,
) -> dict[str, Any]:
    """Validate every fixed input without writing or loading model weights."""
    dependencies = dict(verify_dependencies_fn())
    _verify_dependency_report(dependencies)
    tokenizer = tokenizer_factory(EXPECTED_MODEL_SNAPSHOT)
    token_contract = derive_prompt_token_contract(tokenizer)
    registration = build_registration_record(
        source_code_sha256=dependencies["source_code_sha256"],
        environment=dependencies["environment"],
        prompt_token_contract=token_contract,
        sealed=False,
    )
    protocol = build_prospective_protocol()
    return {
        "status": "dry_run_no_model_call_no_write",
        "registration_sha256": canonical_sha256(registration),
        "protocol_sha256": registration["protocol_sha256"],
        "split_manifest_sha256": registration["split_manifest_sha256"],
        "split_accounting": registration["split_accounting"],
        "query_accounting": protocol["query_accounting"],
        "model_snapshot": registration["model_snapshot"],
        "source_code_sha256": registration["source_code_sha256"],
        "checkpoint_sha256": registration["checkpoint_sha256"],
        "prediction_and_topology_claims": "deferred_to_e31_2",
    }


def seal_prospective_registration(
    *,
    registration_path: Path = DEFAULT_REGISTRATION_PATH,
    confirm_seal: bool,
    tokenizer_factory: Callable[[Path], Any] = _default_tokenizer_factory,
    verify_dependencies_fn: Callable[[], Mapping[str, Any]] = verify_fixed_dependencies,
) -> dict[str, Any]:
    """Write the prospective registration once; never load model weights."""
    if confirm_seal is not True:
        raise RuntimeError("sealing requires confirm_seal=True")
    ensure_write_once_available(registration_path)
    dependencies = dict(verify_dependencies_fn())
    _verify_dependency_report(dependencies)
    registration = build_registration_record(
        source_code_sha256=dependencies["source_code_sha256"],
        environment=dependencies["environment"],
        prompt_token_contract=derive_prompt_token_contract(
            tokenizer_factory(EXPECTED_MODEL_SNAPSHOT)
        ),
        sealed=True,
    )
    atomic_json_save(registration, registration_path)
    return {
        "status": "prospective_registration_sealed_without_model_call",
        "registration": str(registration_path.resolve()),
        "registration_sha256": sha256(registration_path),
        "protocol_sha256": registration["protocol_sha256"],
    }


def _validate_clean_screen(
    screen: Mapping[str, Any], registration: Mapping[str, Any]
) -> None:
    if (
        screen.get("schema_version") != "e31_1_all_registered_clean_screen_v1"
        or screen.get("status") != "all_48_clean_rows_passed_before_any_patch"
        or screen.get("split_order") != ["discovery", "confirmation"]
        or screen.get("total_context_count") != 48
        or screen.get("total_clean_forward_examples") != 96
        or screen.get("all_splits_passed") is not True
        or screen.get("gate") != registration["clean_panel_gate"]
    ):
        raise RuntimeError("global clean-screen header or accounting changed")
    for split, context_count in (("discovery", 16), ("confirmation", 32)):
        report = screen.get("reports", {}).get(split, {})
        metadata = screen.get("clean_metadata", {}).get(split)
        if (
            report.get("passed") is not True
            or report.get("context_count") != context_count
            or not isinstance(metadata, list)
            or len(metadata) != context_count
        ):
            raise RuntimeError(f"{split} clean-screen gate changed")
        for field in (
            "clean_base_donor_minus_base_margins",
            "clean_donor_donor_minus_base_margins",
            "clean_endpoints",
        ):
            tensor = screen.get(field, {}).get(split)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float64
                or tuple(tensor.shape) != (context_count,)
                or not bool(tensor.isfinite().all())
            ):
                raise RuntimeError(f"{split} clean-screen {field} changed")
        contracts = registration["clean_metadata_contract"][split]
        for observed, contract in zip(metadata, contracts, strict=True):
            if {key: observed.get(key) for key in contract} != contract:
                raise RuntimeError(f"{split} clean metadata contract changed")


def _attempt_record(
    registration: Mapping[str, Any], registration_path: Path, digest: str
) -> dict[str, Any]:
    return {
        "schema_version": "e31_1_global_acquisition_attempt_v1",
        "status": "global_attempt_consumed_before_tokenizer_or_model_load",
        "registration": str(registration_path.resolve()),
        "registration_sha256": digest,
        "protocol_sha256": registration["protocol_sha256"],
        "split_manifest_sha256": registration["split_manifest_sha256"],
        "source_code_sha256_verified_before_attempt": registration[
            "source_code_sha256"
        ],
        "checkpoint_sha256_verified_before_attempt": registration["checkpoint_sha256"],
        "environment_verified_before_attempt": registration["environment"],
        "maximum_registered_model_forwards": build_prospective_protocol()[
            "query_accounting"
        ]["maximum_registered_model_forwards"],
        "output_order": [
            "all_registered_clean_screen.pt",
            "discovery_labels.pt",
            "discovery_geometry.pt",
            "confirmation_authorization.json",
            "confirmation_labels.pt",
        ],
    }


def _split_custody(
    *,
    split: Split,
    registration_sha256: str,
    attempt_sha256: str,
    clean_screen_sha256: str,
    labels_sha256: str,
    geometry_sha256: str | None = None,
    authorization_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "e31_1_split_acquisition_custody_v1",
        "status": f"{split}_labels_acquired_write_once",
        "split": split,
        "registration_sha256": registration_sha256,
        "global_attempt_sha256": attempt_sha256,
        "clean_screen_sha256": clean_screen_sha256,
        "labels_sha256": labels_sha256,
        "discovery_geometry_sha256": geometry_sha256,
        "confirmation_authorization_sha256": authorization_sha256,
        "all_registered_rows_retained": True,
        "patched_response_clipping": False,
    }


def _validate_score_custody_chain(
    *,
    output_directory: Path,
    registration_sha256: str,
    discovery_labels_sha256: str,
    discovery_geometry_sha256: str,
    confirmation_labels_sha256: str,
) -> dict[str, str]:
    """Prove that confirmation opened only after discovery geometry was sealed."""
    expected_hashes = (
        registration_sha256,
        discovery_labels_sha256,
        discovery_geometry_sha256,
        confirmation_labels_sha256,
    )
    if any(not valid_sha256(value) for value in expected_hashes):
        raise ValueError("custody chain requires exact SHA-256 inputs")

    attempt_path = output_directory / "global_attempt.json"
    attempt = _read_json_object(attempt_path)
    attempt_sha256 = sha256(attempt_path)
    if (
        attempt.get("schema_version") != "e31_1_global_acquisition_attempt_v1"
        or attempt.get("status")
        != "global_attempt_consumed_before_tokenizer_or_model_load"
        or attempt.get("registration_sha256") != registration_sha256
        or attempt.get("protocol_sha256") != PROTOCOL_SHA256
        or attempt.get("maximum_registered_model_forwards") != 373_568
    ):
        raise RuntimeError("global attempt breaks the acquisition custody chain")

    clean_path = output_directory / "all_registered_clean_screen_custody.json"
    clean = _read_json_object(clean_path)
    clean_sha256 = clean.get("clean_screen_sha256")
    expected_clean = {
        "schema_version": "e31_1_global_clean_screen_custody_v1",
        "status": "all_48_clean_rows_passed_before_any_patched_response",
        "registration_sha256": registration_sha256,
        "global_attempt_sha256": attempt_sha256,
        "clean_screen_sha256": clean_sha256,
        "context_count": 48,
        "clean_forward_examples": 96,
    }
    if not valid_sha256(clean_sha256) or clean != expected_clean:
        raise RuntimeError("clean screen breaks the acquisition custody chain")

    discovery_path = output_directory / "discovery_custody.json"
    discovery = _read_json_object(discovery_path)
    expected_discovery = _split_custody(
        split="discovery",
        registration_sha256=registration_sha256,
        attempt_sha256=attempt_sha256,
        clean_screen_sha256=clean_sha256,
        labels_sha256=discovery_labels_sha256,
    )
    if discovery != expected_discovery:
        raise RuntimeError("discovery labels break the acquisition custody chain")

    geometry_custody_path = output_directory / "discovery_geometry_custody.json"
    geometry_custody = _read_json_object(geometry_custody_path)
    expected_geometry_custody = {
        "schema_version": "e31_1_discovery_geometry_custody_v1",
        "status": "geometry_and_controls_frozen_before_confirmation",
        "registration_sha256": registration_sha256,
        "discovery_labels_sha256": discovery_labels_sha256,
        "discovery_geometry_sha256": discovery_geometry_sha256,
        "matched_control_support_count": 10_000,
        "confirmation_outcomes_used": False,
    }
    if geometry_custody != expected_geometry_custody:
        raise RuntimeError("frozen geometry breaks the acquisition custody chain")
    geometry_custody_sha256 = sha256(geometry_custody_path)

    authorization_path = output_directory / "confirmation_authorization.json"
    authorization = _read_json_object(authorization_path)
    expected_authorization = {
        "schema_version": "e31_1_confirmation_authorization_v1",
        "status": "confirmation_authorized_after_discovery_geometry_seal",
        "registration_sha256": registration_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
        "discovery_labels_sha256": discovery_labels_sha256,
        "discovery_geometry_sha256": discovery_geometry_sha256,
        "discovery_geometry_custody_sha256": geometry_custody_sha256,
        "confirmation_adaptation_allowed": False,
    }
    if authorization != expected_authorization:
        raise RuntimeError("authorization breaks the acquisition custody chain")
    authorization_sha256 = sha256(authorization_path)

    confirmation_path = output_directory / "confirmation_custody.json"
    confirmation = _read_json_object(confirmation_path)
    if confirmation.get("confirmation_authorization_sha256") != authorization_sha256:
        raise RuntimeError(
            "confirmation authorization hash breaks the acquisition custody chain"
        )
    expected_confirmation = _split_custody(
        split="confirmation",
        registration_sha256=registration_sha256,
        attempt_sha256=attempt_sha256,
        clean_screen_sha256=clean_sha256,
        labels_sha256=confirmation_labels_sha256,
        geometry_sha256=discovery_geometry_sha256,
        authorization_sha256=authorization_sha256,
    )
    if confirmation != expected_confirmation:
        raise RuntimeError("confirmation labels break the acquisition custody chain")

    return {
        "global_attempt_sha256": attempt_sha256,
        "clean_screen_sha256": clean_sha256,
        "discovery_custody_sha256": sha256(discovery_path),
        "discovery_geometry_custody_sha256": geometry_custody_sha256,
        "confirmation_authorization_sha256": authorization_sha256,
        "confirmation_custody_sha256": sha256(confirmation_path),
    }


def _validate_total_query_accounting(
    clean_screen: Mapping[str, Any],
    discovery_labels: Mapping[str, Any],
    confirmation_labels: Mapping[str, Any],
) -> int:
    """Recompute the physical-forward total from the artifacts being scored."""
    try:
        observed = int(clean_screen["total_clean_forward_examples"])
        for labels in (discovery_labels, confirmation_labels):
            accounting = labels["query_accounting"]
            observed += int(accounting["patched_context_mask_cells"])
            observed += int(accounting["batch_smoke_physical_context_mask_cells"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("prospective query accounting is incomplete") from error
    expected = int(
        build_prospective_protocol()["query_accounting"][
            "maximum_registered_model_forwards"
        ]
    )
    if observed != expected:
        raise RuntimeError(
            f"prospective query accounting changed: observed {observed}, expected {expected}"
        )
    return observed


def acquire_registered_stage(
    *,
    stage: Split,
    registration_path: Path,
    expected_registration_sha256: str,
    output_directory: Path,
    runner_factory: Callable[[Mapping[str, Any]], Any] = _default_runner_factory,
    acquire_fn: Callable[..., dict[str, Any]] | None = None,
    validate_labels_fn: Callable[..., dict[str, Any]] = validate_acquired_split,
    verify_dependencies_fn: Callable[[Mapping[str, Any]], None] = (
        verify_registered_dependencies
    ),
) -> dict[str, Any]:
    """Small stage seam retained for custody tests; official runs use one process."""
    if stage not in ("discovery", "confirmation"):
        raise ValueError("stage must be discovery or confirmation")
    attempt_path = output_directory / f"{stage}_attempt.json"
    labels_path = output_directory / f"{stage}_labels.pt"
    custody_path = output_directory / f"{stage}_custody.json"
    for path in (attempt_path, labels_path, custody_path):
        ensure_write_once_available(path)
    registration = _load_sealed_registration(
        registration_path, expected_registration_sha256
    )
    verify_dependencies_fn(registration)
    atomic_json_save(
        {
            **_attempt_record(
                registration, registration_path, expected_registration_sha256
            ),
            "schema_version": "e31_1_stage_test_attempt_v1",
            "status": f"{stage}_attempt_consumed_before_runner_factory",
            "stage": stage,
        },
        attempt_path,
    )
    runner = runner_factory(registration)
    clean_screen_sha256 = "0" * 64
    if acquire_fn is None:
        prepared, clean_screen = screen_all_registered_clean_rows(runner)
        _validate_clean_screen(clean_screen, registration)
        labels = acquire_prepared_split(
            runner,
            prepared,
            split=stage,
            registration_sha256=expected_registration_sha256,
            clean_screen_sha256=clean_screen_sha256,
        )
    else:
        labels = acquire_fn(
            runner,
            registration,
            stage,
            clean_screen_sha256,
        )
    validated = validate_labels_fn(
        labels,
        registration=registration,
        registration_sha256=expected_registration_sha256,
        split=stage,
        clean_screen_sha256=clean_screen_sha256,
    )
    atomic_torch_save(validated, labels_path)
    custody = _split_custody(
        split=stage,
        registration_sha256=expected_registration_sha256,
        attempt_sha256=sha256(attempt_path),
        clean_screen_sha256=clean_screen_sha256,
        labels_sha256=sha256(labels_path),
    )
    atomic_json_save(custody, custody_path)
    return custody


def acquire_registered_atlas(
    *,
    registration_path: Path = DEFAULT_REGISTRATION_PATH,
    expected_registration_sha256: str,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    confirm_model_run: bool,
    runner_factory: Callable[[Mapping[str, Any]], E311DirectedIOIPatchRunner] = (
        _default_runner_factory
    ),
    verify_dependencies_fn: Callable[[Mapping[str, Any]], None] = (
        verify_registered_dependencies
    ),
) -> dict[str, Any]:
    """Acquire D and C in one model load with geometry sealed between them."""
    if confirm_model_run is not True:
        raise RuntimeError("model acquisition requires confirm_model_run=True")
    paths = {
        "attempt": output_directory / "global_attempt.json",
        "clean": output_directory / "all_registered_clean_screen.pt",
        "clean_custody": output_directory / "all_registered_clean_screen_custody.json",
        "clean_failure": output_directory / "clean_screen_failure.pt",
        "clean_failure_custody": output_directory / "clean_screen_failure_custody.json",
        "d_labels": output_directory / "discovery_labels.pt",
        "d_custody": output_directory / "discovery_custody.json",
        "geometry": output_directory / "discovery_geometry.pt",
        "geometry_custody": output_directory / "discovery_geometry_custody.json",
        "authorization": output_directory / "confirmation_authorization.json",
        "c_labels": output_directory / "confirmation_labels.pt",
        "c_custody": output_directory / "confirmation_custody.json",
    }
    for path in paths.values():
        ensure_write_once_available(path)
    registration = _load_sealed_registration(
        registration_path, expected_registration_sha256
    )
    verify_dependencies_fn(registration)
    atomic_json_save(
        _attempt_record(registration, registration_path, expected_registration_sha256),
        paths["attempt"],
    )
    attempt_sha256 = sha256(paths["attempt"])
    runner = runner_factory(registration)
    try:
        prepared, clean_screen = screen_all_registered_clean_rows(runner)
        _validate_clean_screen(clean_screen, registration)
    except Exception as error:
        failure_record = {
            "schema_version": "e31_1_clean_screen_failure_v1",
            "status": "clean_screen_failed_before_any_patched_response",
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "aggregate_report_available": isinstance(error, CleanPanelViabilityError),
            "aggregate_report": (
                deepcopy(error.report)
                if isinstance(error, CleanPanelViabilityError)
                else None
            ),
        }
        atomic_torch_save(failure_record, paths["clean_failure"])
        atomic_json_save(
            {
                "schema_version": "e31_1_clean_screen_failure_custody_v1",
                "status": "failed_before_any_patched_response",
                "registration_sha256": expected_registration_sha256,
                "global_attempt_sha256": attempt_sha256,
                "failure_sha256": sha256(paths["clean_failure"]),
                "patched_response_count": 0,
            },
            paths["clean_failure_custody"],
        )
        raise

    atomic_torch_save(clean_screen, paths["clean"])
    clean_sha256 = sha256(paths["clean"])
    atomic_json_save(
        {
            "schema_version": "e31_1_global_clean_screen_custody_v1",
            "status": "all_48_clean_rows_passed_before_any_patched_response",
            "registration_sha256": expected_registration_sha256,
            "global_attempt_sha256": attempt_sha256,
            "clean_screen_sha256": clean_sha256,
            "context_count": 48,
            "clean_forward_examples": 96,
        },
        paths["clean_custody"],
    )

    discovery = acquire_prepared_split(
        runner,
        prepared,
        split="discovery",
        registration_sha256=expected_registration_sha256,
        clean_screen_sha256=clean_sha256,
    )
    validate_acquired_split(
        discovery,
        registration=registration,
        registration_sha256=expected_registration_sha256,
        split="discovery",
        clean_screen_sha256=clean_sha256,
    )
    atomic_torch_save(discovery, paths["d_labels"])
    discovery_sha256 = sha256(paths["d_labels"])
    d_custody = _split_custody(
        split="discovery",
        registration_sha256=expected_registration_sha256,
        attempt_sha256=attempt_sha256,
        clean_screen_sha256=clean_sha256,
        labels_sha256=discovery_sha256,
    )
    atomic_json_save(d_custody, paths["d_custody"])

    # The scorer source/checkpoint closure is checked again before any D fit.
    verify_dependencies_fn(registration)
    discovery_panels = assemble_regime_panels(
        responses=discovery["normalized_responses"],
        split_manifest=build_split_mask_manifest("discovery"),
        prompt_records=discovery["prompt_records"],
    )
    frozen_geometry = fit_discovery_geometry(
        discovery_panels=discovery_panels,
        primary_regime_id=PRIMARY_REGIME_ID,
    )
    geometry_record = {
        "schema_version": "e31_1_discovery_geometry_seal_v1",
        "status": "discovery_geometry_frozen_before_confirmation_authorization",
        "registration_sha256": expected_registration_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
        "discovery_labels_sha256": discovery_sha256,
        "scoring_source_sha256": registration["scoring_source_sha256"],
        "frozen_geometry": frozen_geometry,
    }
    controls = frozen_geometry["regimes"][PRIMARY_REGIME_ID]["matched_controls"]
    if controls is None or controls.get("support_count") != 10_000:
        raise RuntimeError("discovery geometry omitted 10,000 frozen controls")
    atomic_torch_save(geometry_record, paths["geometry"])
    geometry_sha256 = sha256(paths["geometry"])
    atomic_json_save(
        {
            "schema_version": "e31_1_discovery_geometry_custody_v1",
            "status": "geometry_and_controls_frozen_before_confirmation",
            "registration_sha256": expected_registration_sha256,
            "discovery_labels_sha256": discovery_sha256,
            "discovery_geometry_sha256": geometry_sha256,
            "matched_control_support_count": 10_000,
            "confirmation_outcomes_used": False,
        },
        paths["geometry_custody"],
    )

    # This verification and authorization are the only gate opening C outcomes.
    verify_dependencies_fn(registration)
    atomic_json_save(
        {
            "schema_version": "e31_1_confirmation_authorization_v1",
            "status": "confirmation_authorized_after_discovery_geometry_seal",
            "registration_sha256": expected_registration_sha256,
            "protocol_sha256": PROTOCOL_SHA256,
            "discovery_labels_sha256": discovery_sha256,
            "discovery_geometry_sha256": geometry_sha256,
            "discovery_geometry_custody_sha256": sha256(paths["geometry_custody"]),
            "confirmation_adaptation_allowed": False,
        },
        paths["authorization"],
    )
    authorization_sha256 = sha256(paths["authorization"])
    confirmation = acquire_prepared_split(
        runner,
        prepared,
        split="confirmation",
        registration_sha256=expected_registration_sha256,
        clean_screen_sha256=clean_sha256,
    )
    validate_acquired_split(
        confirmation,
        registration=registration,
        registration_sha256=expected_registration_sha256,
        split="confirmation",
        clean_screen_sha256=clean_sha256,
    )
    total_model_forwards = _validate_total_query_accounting(
        clean_screen, discovery, confirmation
    )
    atomic_torch_save(confirmation, paths["c_labels"])
    confirmation_sha256 = sha256(paths["c_labels"])
    c_custody = _split_custody(
        split="confirmation",
        registration_sha256=expected_registration_sha256,
        attempt_sha256=attempt_sha256,
        clean_screen_sha256=clean_sha256,
        labels_sha256=confirmation_sha256,
        geometry_sha256=geometry_sha256,
        authorization_sha256=authorization_sha256,
    )
    atomic_json_save(c_custody, paths["c_custody"])
    return {
        "status": "prospective_atlas_acquired_geometry_frozen_before_confirmation",
        "registration_sha256": expected_registration_sha256,
        "global_attempt_sha256": attempt_sha256,
        "clean_screen_sha256": clean_sha256,
        "discovery_labels_sha256": discovery_sha256,
        "discovery_geometry_sha256": geometry_sha256,
        "confirmation_authorization_sha256": authorization_sha256,
        "confirmation_labels_sha256": confirmation_sha256,
        "maximum_registered_model_forwards": total_model_forwards,
        "prediction_and_topology_claims": "deferred_to_e31_2",
    }


def score_registered_atlas(
    *,
    registration_path: Path = DEFAULT_REGISTRATION_PATH,
    expected_registration_sha256: str,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    expected_discovery_labels_sha256: str,
    expected_geometry_sha256: str,
    expected_confirmation_labels_sha256: str,
    confirm_score: bool,
    verify_dependencies_fn: Callable[[Mapping[str, Any]], None] = (
        verify_registered_dependencies
    ),
) -> dict[str, Any]:
    """Purely score exact hash-supplied D/C labels using frozen D geometry."""
    if confirm_score is not True:
        raise RuntimeError("scoring requires confirm_score=True")
    score_attempt_path = output_directory / "score_attempt.json"
    score_path = output_directory / "prospective_score.pt"
    score_custody_path = output_directory / "score_custody.json"
    for path in (score_attempt_path, score_path, score_custody_path):
        ensure_write_once_available(path)
    registration = _load_sealed_registration(
        registration_path, expected_registration_sha256
    )
    # This is deliberately immediately before the score attempt and label opens.
    verify_dependencies_fn(registration)
    supplied = {
        "discovery_labels": expected_discovery_labels_sha256,
        "discovery_geometry": expected_geometry_sha256,
        "confirmation_labels": expected_confirmation_labels_sha256,
    }
    if any(not valid_sha256(value) for value in supplied.values()):
        raise ValueError("every scoring input requires an exact SHA-256")
    atomic_json_save(
        {
            "schema_version": "e31_1_pure_score_attempt_v1",
            "status": "score_attempt_consumed_before_opening_outcome_artifacts",
            "registration_sha256": expected_registration_sha256,
            "protocol_sha256": PROTOCOL_SHA256,
            "expected_artifact_sha256": supplied,
            "scoring_source_sha256": registration["scoring_source_sha256"],
        },
        score_attempt_path,
    )
    acquisition_chain = _validate_score_custody_chain(
        output_directory=output_directory,
        registration_sha256=expected_registration_sha256,
        discovery_labels_sha256=expected_discovery_labels_sha256,
        discovery_geometry_sha256=expected_geometry_sha256,
        confirmation_labels_sha256=expected_confirmation_labels_sha256,
    )
    discovery_path = output_directory / "discovery_labels.pt"
    geometry_path = output_directory / "discovery_geometry.pt"
    confirmation_path = output_directory / "confirmation_labels.pt"
    verify_file_hash(discovery_path, expected_discovery_labels_sha256)
    verify_file_hash(geometry_path, expected_geometry_sha256)
    verify_file_hash(confirmation_path, expected_confirmation_labels_sha256)
    clean_sha256 = acquisition_chain["clean_screen_sha256"]
    clean_screen_path = output_directory / "all_registered_clean_screen.pt"
    verify_file_hash(clean_screen_path, clean_sha256)
    clean_screen = _load_torch_object(clean_screen_path)
    _validate_clean_screen(clean_screen, registration)
    discovery = validate_acquired_split(
        _load_torch_object(discovery_path),
        registration=registration,
        registration_sha256=expected_registration_sha256,
        split="discovery",
        clean_screen_sha256=clean_sha256,
    )
    confirmation = validate_acquired_split(
        _load_torch_object(confirmation_path),
        registration=registration,
        registration_sha256=expected_registration_sha256,
        split="confirmation",
        clean_screen_sha256=clean_sha256,
    )
    total_model_forwards = _validate_total_query_accounting(
        clean_screen, discovery, confirmation
    )
    geometry_record = _load_torch_object(geometry_path)
    if (
        geometry_record.get("schema_version") != "e31_1_discovery_geometry_seal_v1"
        or geometry_record.get("status")
        != "discovery_geometry_frozen_before_confirmation_authorization"
        or geometry_record.get("registration_sha256") != expected_registration_sha256
        or geometry_record.get("protocol_sha256") != PROTOCOL_SHA256
        or geometry_record.get("discovery_labels_sha256")
        != expected_discovery_labels_sha256
        or geometry_record.get("scoring_source_sha256")
        != registration["scoring_source_sha256"]
    ):
        raise RuntimeError("frozen discovery geometry custody changed")
    discovery_panels = assemble_regime_panels(
        responses=discovery["normalized_responses"],
        split_manifest=build_split_mask_manifest("discovery"),
        prompt_records=discovery["prompt_records"],
    )
    confirmation_panels = assemble_regime_panels(
        responses=confirmation["normalized_responses"],
        split_manifest=build_split_mask_manifest("confirmation"),
        prompt_records=confirmation["prompt_records"],
    )
    score = score_with_frozen_geometry(
        discovery_panels=discovery_panels,
        confirmation_panels=confirmation_panels,
        frozen_geometry=geometry_record["frozen_geometry"],
    )
    score_record = {
        "schema_version": "e31_1_custodied_prospective_atlas_score_v1",
        "status": "exact_registered_score_without_model_calls",
        "registration_sha256": expected_registration_sha256,
        "protocol_sha256": PROTOCOL_SHA256,
        "discovery_labels_sha256": expected_discovery_labels_sha256,
        "discovery_geometry_sha256": expected_geometry_sha256,
        "confirmation_labels_sha256": expected_confirmation_labels_sha256,
        "acquisition_custody_chain": acquisition_chain,
        "verified_total_model_forwards": total_model_forwards,
        "score": score,
        "prediction_and_topology_claims": "deferred_to_e31_2",
    }
    atomic_torch_save(score_record, score_path)
    custody = {
        "schema_version": "e31_1_score_custody_v1",
        "status": "prospective_atlas_score_sealed_write_once",
        "registration_sha256": expected_registration_sha256,
        "score_attempt_sha256": sha256(score_attempt_path),
        "score_sha256": sha256(score_path),
        "acquisition_custody_chain": acquisition_chain,
        "verified_total_model_forwards": total_model_forwards,
        "atlas_verdict": score["atlas_verdict"],
        "predictive_or_topology_claim_authorized": False,
    }
    atomic_json_save(custody, score_custody_path)
    return custody


def run_registered_prospective(
    *,
    registration_path: Path = DEFAULT_REGISTRATION_PATH,
    expected_registration_sha256: str,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    confirm_model_run: bool,
) -> dict[str, Any]:
    """Acquire and score the official atlas without an adaptation seam."""
    acquisition = acquire_registered_atlas(
        registration_path=registration_path,
        expected_registration_sha256=expected_registration_sha256,
        output_directory=output_directory,
        confirm_model_run=confirm_model_run,
        runner_factory=_default_runner_factory,
        verify_dependencies_fn=verify_registered_dependencies,
    )
    score_custody = score_registered_atlas(
        registration_path=registration_path,
        expected_registration_sha256=expected_registration_sha256,
        output_directory=output_directory,
        expected_discovery_labels_sha256=acquisition["discovery_labels_sha256"],
        expected_geometry_sha256=acquisition["discovery_geometry_sha256"],
        expected_confirmation_labels_sha256=acquisition["confirmation_labels_sha256"],
        confirm_score=True,
        verify_dependencies_fn=verify_registered_dependencies,
    )
    return {
        **acquisition,
        "status": "prospective_atlas_acquired_and_scored_write_once",
        "score_custody_sha256": sha256(output_directory / "score_custody.json"),
        "score_sha256": score_custody["score_sha256"],
        "atlas_verdict": score_custody["atlas_verdict"],
    }


__all__ = [
    "DEFAULT_OUTPUT_DIRECTORY",
    "DEFAULT_REGISTRATION_PATH",
    "PRIMARY_REGIME_ID",
    "_validate_score_custody_chain",
    "_validate_total_query_accounting",
    "acquire_registered_atlas",
    "acquire_registered_stage",
    "prospective_dry_run",
    "run_registered_prospective",
    "score_registered_atlas",
    "seal_prospective_registration",
]
