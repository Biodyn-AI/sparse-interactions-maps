# ruff: noqa: EXE002
"""One-process, write-once orchestration for prospective E31.2.

The only permitted order is registration, durable attempt, one global clean
screen, discovery smoke and atlas, frozen discovery geometry, confirmation
authorization, confirmation atlas and new reconstruction rows, then pure
scoring with an exact discovery-geometry replay.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from experiments.e31_sparse_interactions.e31_2_custody import (
    E31_1_PARENT_SCORE_SHA256,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_MODEL_SNAPSHOT,
    EXPECTED_RUNTIME_ENVIRONMENT,
    SOURCE_PATHS,
    atomic_json_save,
    atomic_torch_save,
    ensure_write_once_available,
    sha256,
    valid_sha256,
    verify_file_hash,
    verify_fixed_dependencies,
    verify_registered_dependencies,
    write_once_reservation_path,
)
from experiments.e31_sparse_interactions.e31_2_head_acquisition import (
    acquire_head_atlas_from_prepared,
    acquire_reconstruction_from_prepared,
    prepare_and_screen_all_registered_contexts,
    validate_atlas_labels,
    validate_reconstruction_labels,
)
from experiments.e31_sparse_interactions.e31_2_head_patch_runner import (
    E312HeadPatchRunner,
)
from experiments.e31_sparse_interactions.e31_2_registration import (
    REGISTERED_OUTPUT_DIRECTORY,
    REGISTERED_OUTPUT_FILENAMES,
    REGISTERED_REGISTRATION_PATH,
    SEALED_STATUS,
    build_registration_record,
    derive_prompt_token_contract,
    validate_registration_payload,
)
from experiments.e31_sparse_interactions.mixed_resolution_panel import (
    build_prompt_manifest,
    canonical_sha256,
)
from experiments.e31_sparse_interactions.mixed_resolution_protocol import (
    PROTOCOL_SHA256,
    build_atlas_manifest,
    build_reconstruction_manifest,
)
from experiments.e31_sparse_interactions.mixed_resolution_scoring import (
    fit_discovery_geometry,
    score_with_frozen_geometry,
)

DEFAULT_REGISTRATION_PATH = REGISTERED_REGISTRATION_PATH
DEFAULT_OUTPUT_DIRECTORY = REGISTERED_OUTPUT_DIRECTORY
OUTPUT_FILENAMES = REGISTERED_OUTPUT_FILENAMES


class _ModelForwardCounter:
    """Count attempted outer-model batch examples before every forward call."""

    def __init__(self, runner: Any) -> None:
        model = getattr(runner, "model", None)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("production E31.2 runner must expose a torch model")
        self._examples = 0
        self._invocations = 0

        def count_forward(
            _module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 1:
                raise RuntimeError("outer model forward lacks a batch tensor")
            self._examples += int(input_ids.shape[0])
            self._invocations += 1

        self._handle = model.register_forward_pre_hook(count_forward, with_kwargs=True)

    @property
    def attempted_examples(self) -> int:
        return self._examples

    @property
    def attempted_invocations(self) -> int:
        return self._invocations

    def close(self) -> None:
        self._handle.remove()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _load_torch(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"expected a mapping Torch artifact in {path}")
    return value


def _default_tokenizer_factory(snapshot: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)


def _default_runner_factory(registration: Mapping[str, Any]) -> E312HeadPatchRunner:
    """Verify the tokenizer contract before loading model weights."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = Path(registration["model_snapshot"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    observed_contract = derive_prompt_token_contract(tokenizer)
    if observed_contract != registration["prompt_token_contract"]:
        raise RuntimeError("live tokenizer differs from the registered token contract")
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        attn_implementation="eager",
        torch_dtype=torch.float64,
    ).to(device="cpu", dtype=torch.float64)
    model.eval()
    return E312HeadPatchRunner(
        model,
        tokenizer,
        batch_size=int(registration["execution"]["batch_size"]),
    )


def _verify_dependency_report(report: Mapping[str, Any]) -> None:
    if report.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("verified checkpoint closure differs from GPT-2 small")
    if report.get("environment") != EXPECTED_RUNTIME_ENVIRONMENT:
        raise RuntimeError("verified E31.2 runtime environment changed")
    if set(report.get("source_code_sha256", {})) != set(SOURCE_PATHS):
        raise RuntimeError("verified E31.2 source closure changed")
    parent = report.get("parent_provenance", {})
    if parent.get("e31_1_v2_score_sha256") != E31_1_PARENT_SCORE_SHA256:
        raise RuntimeError("verified E31.1 parent score changed")


def prospective_dry_run(
    *,
    tokenizer_factory: Callable[[Path], Any] = _default_tokenizer_factory,
    verify_dependencies_fn: Callable[[], Mapping[str, Any]] = verify_fixed_dependencies,
) -> dict[str, Any]:
    """Validate fixed inputs without loading model weights or writing files."""
    dependencies = dict(verify_dependencies_fn())
    _verify_dependency_report(dependencies)
    token_contract = derive_prompt_token_contract(
        tokenizer_factory(EXPECTED_MODEL_SNAPSHOT)
    )
    registration = build_registration_record(
        source_code_sha256=dependencies["source_code_sha256"],
        environment=dependencies["environment"],
        prompt_token_contract=token_contract,
        sealed=False,
    )
    return {
        "status": "dry_run_no_model_call_no_write",
        "registration_sha256": canonical_sha256(registration),
        "protocol_sha256": registration["protocol_sha256"],
        "prompt_manifest_sha256": registration["prompt_manifest_sha256"],
        "atlas_manifest_sha256": registration["atlas_manifest_sha256"],
        "maximum_registered_model_forwards": registration["query_accounting"][
            "maximum_registered_model_forwards"
        ],
        "model_snapshot": registration["model_snapshot"],
        "parent_score_sha256": E31_1_PARENT_SCORE_SHA256,
        "source_code_sha256": registration["source_code_sha256"],
        "checkpoint_sha256": registration["checkpoint_sha256"],
    }


def seal_prospective_registration(
    *,
    registration_path: Path = DEFAULT_REGISTRATION_PATH,
    confirm_seal: bool,
    tokenizer_factory: Callable[[Path], Any] = _default_tokenizer_factory,
    verify_dependencies_fn: Callable[[], Mapping[str, Any]] = verify_fixed_dependencies,
) -> dict[str, Any]:
    """Write the registration once without loading model weights."""
    if confirm_seal is not True:
        raise RuntimeError("sealing requires confirm_seal=True")
    if registration_path.resolve() != REGISTERED_REGISTRATION_PATH:
        raise RuntimeError("E31.2 seal requires the sole registered path")
    ensure_write_once_available(registration_path)
    _require_pristine_output_directory(DEFAULT_OUTPUT_DIRECTORY)
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
    _require_pristine_output_directory(DEFAULT_OUTPUT_DIRECTORY)
    atomic_json_save(registration, registration_path)
    return {
        "status": "prospective_registration_sealed_without_model_call",
        "registration": str(registration_path.resolve()),
        "registration_sha256": sha256(registration_path),
        "protocol_sha256": PROTOCOL_SHA256,
    }


def _load_sealed_registration(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not valid_sha256(expected_sha256):
        raise ValueError("expected registration SHA-256 is invalid")
    if not write_once_reservation_path(path).is_dir():
        raise RuntimeError("sealed registration lacks its write-once reservation")
    verify_file_hash(path, expected_sha256)
    registration = validate_registration_payload(
        _read_json(path), verify_external_files=False
    )
    if registration["status"] != SEALED_STATUS:
        raise RuntimeError("pipeline requires a sealed E31.2 registration")
    return registration


def _validate_registered_run_paths(
    registration_path: Path, output_directory: Path
) -> None:
    if registration_path.resolve() != REGISTERED_REGISTRATION_PATH:
        raise RuntimeError(
            "E31.2 registration path differs from the sole registered path"
        )
    if output_directory.resolve() != REGISTERED_OUTPUT_DIRECTORY:
        raise RuntimeError("E31.2 output path differs from the sole registered run")


def _require_pristine_output_directory(output_directory: Path) -> None:
    """Reject any prior E31.2 output or write-once reservation before seal."""
    if not output_directory.exists():
        return
    if not output_directory.is_dir():
        raise RuntimeError("registered E31.2 output path is not a directory")
    entries = [
        path.name
        for path in output_directory.iterdir()
        if not path.name.startswith("._")
    ]
    if entries:
        raise RuntimeError(
            f"registered E31.2 output directory is not pristine: {sorted(entries)}"
        )


def _validate_clean_screen(
    screen: Mapping[str, Any], registration: Mapping[str, Any]
) -> None:
    context_count = screen.get("total_context_count", screen.get("context_count"))
    clean_examples = screen.get(
        "total_clean_forward_examples", screen.get("clean_forward_examples")
    )
    if (
        screen.get("all_splits_passed") is not True
        or context_count != 48
        or clean_examples != 96
    ):
        raise RuntimeError("global clean screen did not pass all 48 rows exactly once")
    if screen.get("schema_version") != "e31_2_all_registered_clean_screen_v1":
        # Focused tests use a deliberately tiny fake screen.
        return
    if (
        tuple(screen.get("split_order", ())) != ("discovery", "confirmation")
        or screen.get("context_count_by_split") != {"discovery": 16, "confirmation": 32}
        or screen.get("all_head_norm_diagnostics_passed") is not True
        or screen.get("row_policy") != "retain_and_score_every_registered_context"
    ):
        raise RuntimeError("global clean-screen header changed")
    for split, expected_count in (("discovery", 16), ("confirmation", 32)):
        if screen.get("reports", {}).get(split, {}).get("passed") is not True:
            raise RuntimeError(f"{split} aggregate clean gate failed")
        metadata = screen.get("clean_metadata", {}).get(split)
        token_rows = registration["prompt_token_contract"]["splits"][split]["rows"]
        prompt_rows = build_prompt_manifest(split)
        if not isinstance(metadata, Sequence) or len(metadata) != expected_count:
            raise RuntimeError(f"{split} clean metadata row count changed")
        for observed, tokens, prompt in zip(
            metadata, token_rows, prompt_rows, strict=True
        ):
            if any(observed.get(key) != prompt[key] for key in prompt):
                raise RuntimeError(f"{split} clean prompt metadata changed")
            expected_tokens = {
                "base_target_token_id": tokens["base_target_token_id"],
                "donor_target_token_id": tokens["donor_target_token_id"],
                "token_length": tokens["token_length"],
                "differing_token_position": tokens["differing_prompt_position"],
            }
            if any(
                observed.get(key) != value for key, value in expected_tokens.items()
            ):
                raise RuntimeError(f"{split} clean token metadata changed")
        for field in (
            "clean_base_donor_minus_base_margins",
            "clean_donor_donor_minus_base_margins",
            "clean_endpoints",
        ):
            tensor = screen.get(field, {}).get(split)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float64
                or tuple(tensor.shape) != (expected_count,)
                or not bool(tensor.isfinite().all())
            ):
                raise RuntimeError(f"{split} {field} changed")


def _validate_discovery_accounting(labels: Mapping[str, Any]) -> None:
    accounting = labels.get("query_accounting", {})
    atlas = accounting.get(
        "patched_condition_context_cells", accounting.get("atlas_patch_cells")
    )
    smoke = accounting.get(
        "batch_smoke_physical_context_cells",
        accounting.get("batch_smoke_physical_context_mask_cells"),
    )
    if atlas != 33_024 or smoke != 32 or accounting.get("clean_forward_examples") != 0:
        raise RuntimeError("discovery query accounting changed")
    if labels.get("schema_version") == "e31_2_mixed_resolution_atlas_labels_v1":
        for key in ("coarse_component_masks", "attention_head_masks"):
            value = labels.get(key)
            if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
                raise RuntimeError(f"discovery {key} dtype changed")
        fine = labels.get("normalized_fine_quartets")
        if (
            not isinstance(fine, torch.Tensor)
            or fine.dtype != torch.float64
            or tuple(fine.shape) != (3, 144, 16, 4, 4)
        ):
            raise RuntimeError("discovery fine quartet tensor changed")


def _validate_confirmation_accounting(labels: Mapping[str, Any]) -> tuple[int, int]:
    accounting = labels.get("query_accounting", {})
    atlas = accounting.get("atlas_patch_cells")
    reconstruction = accounting.get("reconstruction_patch_cells")
    if atlas is None and "atlas" in labels:
        atlas = labels["atlas"]["query_accounting"]["patched_condition_context_cells"]
    if reconstruction is None and "reconstruction" in labels:
        reconstruction = labels["reconstruction"]["query_accounting"][
            "patched_new_condition_context_cells"
        ]
    if (
        atlas != 66_048
        or not isinstance(reconstruction, int)
        or reconstruction < 0
        or reconstruction > 2_304
        or accounting.get("clean_forward_examples") != 0
    ):
        raise RuntimeError("confirmation query accounting changed")
    if labels.get("schema_version") == "e31_2_combined_confirmation_labels_v1":
        atlas_labels = labels.get("atlas", {})
        reconstruction_labels = labels.get("reconstruction", {})
        for key in ("coarse_component_masks", "attention_head_masks"):
            value = atlas_labels.get(key)
            if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
                raise RuntimeError(f"confirmation {key} dtype changed")
        fine = atlas_labels.get("normalized_fine_quartets")
        if (
            not isinstance(fine, torch.Tensor)
            or fine.dtype != torch.float64
            or tuple(fine.shape) != (3, 144, 32, 4, 4)
        ):
            raise RuntimeError("confirmation fine quartet tensor changed")
        direct = reconstruction_labels.get("normalized_direct_group_quartets")
        all12 = reconstruction_labels.get("normalized_all12_reference_quartets")
        if (
            not isinstance(direct, torch.Tensor)
            or direct.dtype != torch.float64
            or tuple(direct.shape) != (3, 2, 32, 4, 4)
            or not isinstance(all12, torch.Tensor)
            or all12.dtype != torch.float64
            or tuple(all12.shape) != (3, 32, 4, 4)
        ):
            raise RuntimeError("confirmation reconstruction quartet tensor changed")
        for key in (
            "new_coarse_component_masks",
            "new_attention_head_masks",
        ):
            value = reconstruction_labels.get(key)
            if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
                raise RuntimeError(f"confirmation reconstruction {key} dtype changed")
    return atlas, reconstruction


def _validate_discovery_smoke(
    labels: Mapping[str, Any], registration: Mapping[str, Any]
) -> None:
    smoke = labels.get("batch_invariance_smoke")
    if smoke is None:
        # Test doubles prove stage order through their accounting instead.
        return
    contract = registration["batch_invariance_smoke"]
    checks = {
        "passed": True,
        "context_index": contract["context_index"],
        "pair_id": contract["pair_id"],
        "condition_indices": tuple(contract["condition_indices"]),
        "condition_ids": tuple(contract["condition_ids"]),
        "reference_batch_size": contract["reference_batch_size"],
        "candidate_batch_size": contract["candidate_batch_size"],
        "mask_rows": 16,
        "atol": contract["absolute_tolerance"],
        "rtol": contract["relative_tolerance"],
    }
    if any(smoke.get(key) != value for key, value in checks.items()):
        raise RuntimeError("discovery batch-invariance smoke changed")
    required_coordinates = {
        "normalized_transfer",
        "raw_donor_minus_base_margin",
        "raw_margin_shift_from_base",
    }
    coordinates = smoke.get("coordinates", {})
    if set(coordinates) != required_coordinates or not all(
        row.get("passed") is True for row in coordinates.values()
    ):
        raise RuntimeError("discovery batch-invariance coordinates changed")


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return tuple(left) == tuple(right) and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _validate_geometry(geometry: Mapping[str, Any]) -> None:
    if (
        geometry.get("status") != "discovery_geometry_frozen_before_confirmation"
        or geometry.get("matched_controls", {}).get("support_count") != 10_000
        or "reconstruction_selections" not in geometry
    ):
        raise RuntimeError("discovery geometry, selections, or controls changed")
    build_reconstruction_manifest(geometry["reconstruction_selections"])
    if geometry.get("schema_version") == "e31_2_frozen_head_geometry_v1":
        controls = geometry["matched_controls"]
        if (
            controls.get("seed") != 3_120_001
            or controls.get("uses_model_outcomes") is not False
            or controls.get("selected_and_singleton_supports_excluded") is not True
        ):
            raise RuntimeError("discovery matched-control generator changed")


def _default_prepare_all(
    runner: E312HeadPatchRunner, _registration: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    return prepare_and_screen_all_registered_contexts(
        runner,
        {
            "discovery": build_prompt_manifest("discovery"),
            "confirmation": build_prompt_manifest("confirmation"),
        },
    )


def _protocol_commitments(
    registration: Mapping[str, Any], clean_screen_sha256: str
) -> dict[str, str]:
    return {
        "protocol_sha256": registration["protocol_sha256"],
        "prompt_manifest_sha256": canonical_sha256(
            registration["prompt_manifest_sha256"]
        ),
        "atlas_manifest_sha256": canonical_sha256(
            registration["atlas_manifest_sha256"]
        ),
        "clean_screen_sha256": clean_screen_sha256,
    }


def _default_acquire_discovery(
    runner: E312HeadPatchRunner,
    prepared: Any,
    registration: Mapping[str, Any],
    clean_screen_sha256: str,
) -> dict[str, Any]:
    labels = acquire_head_atlas_from_prepared(
        runner,
        prepared,
        build_atlas_manifest("discovery"),
        split="discovery",
        smoke_condition_indices=registration["batch_invariance_smoke"][
            "condition_indices"
        ],
        protocol_commitments=_protocol_commitments(registration, clean_screen_sha256),
    )
    return validate_atlas_labels(labels, manifest=build_atlas_manifest("discovery"))


def _default_fit_geometry(discovery: Mapping[str, Any]) -> dict[str, Any]:
    return fit_discovery_geometry(
        discovery_quartets=discovery["normalized_fine_quartets"]
    )


def _default_acquire_confirmation(
    runner: E312HeadPatchRunner,
    prepared: Any,
    registration: Mapping[str, Any],
    reconstruction_manifest: Mapping[str, Any],
    clean_screen_sha256: str,
) -> dict[str, Any]:
    commitments = _protocol_commitments(registration, clean_screen_sha256)
    atlas_manifest = build_atlas_manifest("confirmation")
    atlas = acquire_head_atlas_from_prepared(
        runner,
        prepared,
        atlas_manifest,
        split="confirmation",
        smoke_condition_indices=(),
        protocol_commitments=commitments,
    )
    atlas = validate_atlas_labels(atlas, manifest=atlas_manifest)
    reconstruction = acquire_reconstruction_from_prepared(
        runner,
        prepared,
        reconstruction_manifest,
        atlas,
        protocol_commitments=commitments,
    )
    reconstruction = validate_reconstruction_labels(
        reconstruction,
        reconstruction_manifest=reconstruction_manifest,
        confirmation_atlas_labels=atlas,
    )
    return {
        "schema_version": "e31_2_combined_confirmation_labels_v1",
        "atlas": atlas,
        "reconstruction": reconstruction,
        "query_accounting": {
            "atlas_patch_cells": atlas["query_accounting"][
                "patched_condition_context_cells"
            ],
            "reconstruction_patch_cells": reconstruction["query_accounting"][
                "patched_new_condition_context_cells"
            ],
            "clean_forward_examples": 0,
        },
    }


def _context_structure(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    strata = [str(record["stratum_id"]) for record in records]
    clusters = []
    directions = []
    for record in records:
        pair_id = str(record["pair_id"])
        for suffix in ("_a_to_b", "_b_to_a"):
            if pair_id.endswith(suffix):
                clusters.append(pair_id[: -len(suffix)])
                directions.append(suffix[1:])
                break
        else:
            raise RuntimeError("a context pair ID lost its registered direction suffix")
    by_cluster: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters):
        by_cluster.setdefault(cluster, []).append(index)
    for indices in by_cluster.values():
        if (
            len(indices) != 2
            or {directions[index] for index in indices} != {"a_to_b", "b_to_a"}
            or len({strata[index] for index in indices}) != 1
        ):
            raise RuntimeError(
                "each semantic frame must retain both inverse directions"
            )
    return strata, clusters


def _default_score(
    discovery: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    atlas = confirmation["atlas"]
    reconstruction = confirmation["reconstruction"]
    discovery_strata, discovery_clusters = _context_structure(
        discovery["prompt_records"]
    )
    confirmation_strata, confirmation_clusters = _context_structure(
        atlas["prompt_records"]
    )
    direct = reconstruction["normalized_direct_group_quartets"]
    return score_with_frozen_geometry(
        discovery_quartets=discovery["normalized_fine_quartets"],
        confirmation_quartets=atlas["normalized_fine_quartets"],
        pair_selected_composite_quartets=direct[:, 0],
        singleton_selected_composite_quartets=direct[:, 1],
        coarse_composite_quartets=reconstruction["normalized_all12_reference_quartets"],
        discovery_context_strata=discovery_strata,
        discovery_context_cluster_ids=discovery_clusters,
        confirmation_context_strata=confirmation_strata,
        confirmation_context_cluster_ids=confirmation_clusters,
        frozen_geometry=geometry,
    )


def _preflight_output_directory(output_directory: Path) -> None:
    for filename in OUTPUT_FILENAMES:
        ensure_write_once_available(output_directory / filename)


def _artifact_schema(value: str, *, test_only: bool) -> str:
    return value.replace("e31_2_", "e31_2_test_only_", 1) if test_only else value


def _artifact_status(value: str, *, test_only: bool) -> str:
    return f"test_only_{value}" if test_only else value


def _write_failure_custody(
    output_directory: Path,
    *,
    registration_sha256: str,
    attempt_sha256: str,
    stage: str,
    error: BaseException,
    test_only: bool,
    confirmation_patch_started: bool,
) -> None:
    path = output_directory / "run_failure_custody.json"
    if path.exists():
        return
    record = {
        "schema_version": _artifact_schema(
            "e31_2_run_failure_custody_v1", test_only=test_only
        ),
        "status": _artifact_status(
            "failed_closed_without_result_claim", test_only=test_only
        ),
        "failure_stage": stage,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "registration_sha256": registration_sha256,
        "global_attempt_sha256": attempt_sha256,
        "patched_confirmation_started": confirmation_patch_started,
        "scientific_score_authorized": False,
    }
    atomic_json_save(record, path)


def _execute_registered_pipeline(
    *,
    registration_path: Path,
    expected_registration_sha256: str,
    output_directory: Path,
    confirm_model_run: bool,
    runner_factory: Callable[[Mapping[str, Any]], Any] = _default_runner_factory,
    prepare_all_fn: Callable[
        [Any, Mapping[str, Any]], tuple[dict[str, Any], dict[str, Any]]
    ] = _default_prepare_all,
    acquire_discovery_fn: Callable[
        [Any, Any, Mapping[str, Any], str], dict[str, Any]
    ] = _default_acquire_discovery,
    fit_geometry_fn: Callable[
        [Mapping[str, Any]], dict[str, Any]
    ] = _default_fit_geometry,
    acquire_confirmation_fn: Callable[
        [Any, Any, Mapping[str, Any], Mapping[str, Any], str], dict[str, Any]
    ] = _default_acquire_confirmation,
    score_fn: Callable[
        [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], dict[str, Any]
    ] = _default_score,
    verify_dependencies_fn: Callable[
        [Mapping[str, Any]], Any
    ] = verify_registered_dependencies,
    test_only: bool,
) -> dict[str, Any]:
    """Internal executor; public production entry point fixes every callback."""
    if confirm_model_run is not True:
        raise RuntimeError("model acquisition requires confirm_model_run=True")
    registration_path = Path(registration_path)
    output_directory = Path(output_directory)
    if test_only:
        if (
            registration_path.resolve() == REGISTERED_REGISTRATION_PATH
            or output_directory.resolve() == REGISTERED_OUTPUT_DIRECTORY
        ):
            raise RuntimeError("test executor cannot write the canonical E31.2 chain")
    else:
        _validate_registered_run_paths(registration_path, output_directory)
    registration = _load_sealed_registration(
        registration_path, expected_registration_sha256
    )
    _preflight_output_directory(output_directory)

    # Source/checkpoint verification may read files but cannot expose a model
    # response.  The attempt is then durable before runner/tokenizer/model load.
    verify_dependencies_fn(registration)
    attempt_path = output_directory / "global_attempt.json"
    atomic_json_save(
        {
            "schema_version": _artifact_schema(
                "e31_2_global_attempt_v1", test_only=test_only
            ),
            "status": _artifact_status(
                "consumed_before_tokenizer_or_model_load", test_only=test_only
            ),
            "registration": str(registration_path.resolve()),
            "registration_sha256": expected_registration_sha256,
            "protocol_sha256": PROTOCOL_SHA256,
            "parent_score_sha256": E31_1_PARENT_SCORE_SHA256,
            "source_code_sha256_verified_before_attempt": registration[
                "source_code_sha256"
            ],
            "checkpoint_sha256_verified_before_attempt": registration[
                "checkpoint_sha256"
            ],
            "environment_verified_before_attempt": registration["environment"],
            "maximum_registered_model_forwards": 101_504,
            "one_process_one_global_clean_screen": True,
        },
        attempt_path,
    )
    attempt_sha256 = sha256(attempt_path)

    stage = "runner_load"
    forward_counter: _ModelForwardCounter | None = None
    confirmation_patch_started = False
    try:
        runner = runner_factory(registration)
        if not test_only:
            forward_counter = _ModelForwardCounter(runner)
        stage = "global_clean_screen"
        prepared, clean_screen = prepare_all_fn(runner, registration)
        _validate_clean_screen(clean_screen, registration)
        if forward_counter is not None and forward_counter.attempted_examples != 96:
            raise RuntimeError("observed clean forward count differs from exactly 96")
        clean_path = output_directory / "all_registered_clean_screen.pt"
        atomic_torch_save(clean_screen, clean_path)
        clean_sha256 = sha256(clean_path)
        clean_custody_path = (
            output_directory / "all_registered_clean_screen_custody.json"
        )
        atomic_json_save(
            {
                "schema_version": _artifact_schema(
                    "e31_2_global_clean_screen_custody_v1", test_only=test_only
                ),
                "status": _artifact_status(
                    "all_48_clean_rows_passed_before_any_patch", test_only=test_only
                ),
                "registration_sha256": expected_registration_sha256,
                "global_attempt_sha256": attempt_sha256,
                "clean_screen_sha256": clean_sha256,
                "context_count": 48,
                "clean_forward_examples": 96,
                "duplicate_clean_forwards": 0,
            },
            clean_custody_path,
        )
        clean_custody_sha256 = sha256(clean_custody_path)

        stage = "discovery"
        verify_dependencies_fn(registration)
        discovery = acquire_discovery_fn(
            runner,
            prepared["discovery"],
            registration,
            clean_sha256,
        )
        _validate_discovery_accounting(discovery)
        _validate_discovery_smoke(discovery, registration)
        if (
            forward_counter is not None
            and forward_counter.attempted_examples != 96 + 33_024 + 32
        ):
            raise RuntimeError("observed discovery forward delta changed")
        discovery_path = output_directory / "discovery_labels.pt"
        atomic_torch_save(discovery, discovery_path)
        discovery_sha256 = sha256(discovery_path)
        discovery_custody_path = output_directory / "discovery_custody.json"
        atomic_json_save(
            {
                "schema_version": _artifact_schema(
                    "e31_2_discovery_custody_v1", test_only=test_only
                ),
                "status": _artifact_status(
                    "discovery_labels_acquired_write_once", test_only=test_only
                ),
                "registration_sha256": expected_registration_sha256,
                "global_attempt_sha256": attempt_sha256,
                "clean_screen_sha256": clean_sha256,
                "discovery_labels_sha256": discovery_sha256,
                "discovery_atlas_patch_cells": 33_024,
                "batch_smoke_physical_context_mask_cells": 32,
                "all_registered_rows_retained": True,
                "response_clipping": False,
            },
            discovery_custody_path,
        )
        discovery_custody_sha256 = sha256(discovery_custody_path)

        stage = "discovery_geometry"
        verify_dependencies_fn(registration)
        discovery_for_geometry = _load_torch(discovery_path)
        verify_file_hash(discovery_path, discovery_sha256)
        geometry = fit_geometry_fn(discovery_for_geometry)
        _validate_geometry(geometry)
        geometry_path = output_directory / "discovery_geometry.pt"
        atomic_torch_save(geometry, geometry_path)
        geometry_sha256 = sha256(geometry_path)
        selection_sha256 = canonical_sha256(geometry["reconstruction_selections"])
        geometry_custody_path = output_directory / "discovery_geometry_custody.json"
        atomic_json_save(
            {
                "schema_version": _artifact_schema(
                    "e31_2_discovery_geometry_custody_v1", test_only=test_only
                ),
                "status": _artifact_status(
                    "geometry_selections_and_controls_frozen_before_confirmation",
                    test_only=test_only,
                ),
                "registration_sha256": expected_registration_sha256,
                "discovery_labels_sha256": discovery_sha256,
                "discovery_geometry_sha256": geometry_sha256,
                "reconstruction_selection_sha256": selection_sha256,
                "matched_control_support_count": 10_000,
                "confirmation_outcomes_used": False,
            },
            geometry_custody_path,
        )
        geometry_custody_sha256 = sha256(geometry_custody_path)

        reconstruction_manifest = build_reconstruction_manifest(
            geometry["reconstruction_selections"]
        )
        if test_only:
            reconstruction_manifest["schema_version"] = _artifact_schema(
                reconstruction_manifest["schema_version"], test_only=True
            )
            reconstruction_manifest["test_only_execution"] = True
        if reconstruction_manifest["actual_new_condition_count"] > 72:
            raise RuntimeError("dynamic reconstruction manifest exceeds its hard cap")
        reconstruction_manifest_path = (
            output_directory / "confirmation_reconstruction_manifest.json"
        )
        atomic_json_save(reconstruction_manifest, reconstruction_manifest_path)
        reconstruction_manifest_sha256 = sha256(reconstruction_manifest_path)

        stage = "confirmation"
        verify_dependencies_fn(registration)
        authorization_path = output_directory / "confirmation_authorization.json"
        confirmation_custody_path = output_directory / "confirmation_custody.json"
        atomic_json_save(
            {
                "schema_version": _artifact_schema(
                    "e31_2_confirmation_authorization_v1", test_only=test_only
                ),
                "status": _artifact_status(
                    "confirmation_authorized_after_geometry_and_manifest_seal",
                    test_only=test_only,
                ),
                "registration_sha256": expected_registration_sha256,
                "protocol_sha256": PROTOCOL_SHA256,
                "discovery_labels_sha256": discovery_sha256,
                "discovery_geometry_sha256": geometry_sha256,
                "discovery_geometry_custody_sha256": geometry_custody_sha256,
                "reconstruction_selection_sha256": selection_sha256,
                "reconstruction_manifest_sha256": reconstruction_manifest_sha256,
                "maximum_new_condition_count": 72,
                "confirmation_adaptation_allowed": False,
            },
            authorization_path,
        )
        authorization_sha256 = sha256(authorization_path)
        confirmation_patch_started = True
        confirmation = acquire_confirmation_fn(
            runner,
            prepared["confirmation"],
            registration,
            reconstruction_manifest,
            clean_sha256,
        )
        _, reconstruction_cells = _validate_confirmation_accounting(confirmation)
        if (
            forward_counter is not None
            and forward_counter.attempted_examples
            != 96 + 33_024 + 32 + 66_048 + reconstruction_cells
        ):
            raise RuntimeError("observed confirmation forward delta changed")
        confirmation_path = output_directory / "confirmation_labels.pt"
        atomic_torch_save(confirmation, confirmation_path)
        confirmation_sha256 = sha256(confirmation_path)
        atomic_json_save(
            {
                "schema_version": _artifact_schema(
                    "e31_2_confirmation_custody_v1", test_only=test_only
                ),
                "status": _artifact_status(
                    "confirmation_labels_acquired_write_once_after_authorization",
                    test_only=test_only,
                ),
                "registration_sha256": expected_registration_sha256,
                "global_attempt_sha256": attempt_sha256,
                "clean_screen_sha256": clean_sha256,
                "discovery_labels_sha256": discovery_sha256,
                "discovery_geometry_sha256": geometry_sha256,
                "reconstruction_manifest_sha256": reconstruction_manifest_sha256,
                "confirmation_authorization_sha256": authorization_sha256,
                "confirmation_labels_sha256": confirmation_sha256,
                "confirmation_atlas_patch_cells": 66_048,
                "confirmation_reconstruction_patch_cells": reconstruction_cells,
                "duplicate_clean_forwards": 0,
                "duplicate_inherited_atlas_patch_cells": 0,
            },
            confirmation_custody_path,
        )
        confirmation_custody_sha256 = sha256(confirmation_custody_path)

        stage = "score"
        verify_dependencies_fn(registration)
        # Consume the scoring attempt before score-specific hashing or opening
        # of confirmation labels.  Acquisition custody already sealed its hash.
        score_attempt_path = output_directory / "score_attempt.json"
        atomic_json_save(
            {
                "schema_version": _artifact_schema(
                    "e31_2_score_attempt_v1", test_only=test_only
                ),
                "status": _artifact_status(
                    "consumed_before_score_stage_hash_verification_or_label_open",
                    test_only=test_only,
                ),
                "registration_sha256": expected_registration_sha256,
                "protocol_sha256": PROTOCOL_SHA256,
                "confirmation_authorization_sha256": authorization_sha256,
                "confirmation_custody_sha256": confirmation_custody_sha256,
                "geometry_replay_required": True,
                "model_calls_authorized": False,
            },
            score_attempt_path,
        )
        score_attempt_sha256 = sha256(score_attempt_path)

        # Revalidate the complete one-way chain before opening any label tensor.
        for path, digest in (
            (registration_path, expected_registration_sha256),
            (attempt_path, attempt_sha256),
            (clean_path, clean_sha256),
            (clean_custody_path, clean_custody_sha256),
            (discovery_path, discovery_sha256),
            (discovery_custody_path, discovery_custody_sha256),
            (geometry_path, geometry_sha256),
            (geometry_custody_path, geometry_custody_sha256),
            (reconstruction_manifest_path, reconstruction_manifest_sha256),
            (authorization_path, authorization_sha256),
            (confirmation_path, confirmation_sha256),
            (confirmation_custody_path, confirmation_custody_sha256),
            (score_attempt_path, score_attempt_sha256),
        ):
            verify_file_hash(path, digest)
        discovery_reloaded = _load_torch(discovery_path)
        geometry_reloaded = _load_torch(geometry_path)
        confirmation_reloaded = _load_torch(confirmation_path)
        replayed = fit_geometry_fn(discovery_reloaded)
        if not _nested_equal(replayed, geometry_reloaded):
            raise RuntimeError("discovery geometry replay differs from frozen geometry")
        pure_score = score_fn(
            discovery_reloaded, confirmation_reloaded, geometry_reloaded
        )

        nominal_total = 96 + 33_024 + 32 + 66_048 + reconstruction_cells
        actual_total = (
            nominal_total
            if forward_counter is None
            else forward_counter.attempted_examples
        )
        if forward_counter is not None and actual_total != nominal_total:
            raise RuntimeError(
                "observed model forwards differ from artifact accounting"
            )
        if actual_total > 101_504:
            raise RuntimeError("actual E31.2 model forwards exceed the registered cap")
        score_artifact = {
            "schema_version": _artifact_schema(
                "e31_2_prospective_score_artifact_v1", test_only=test_only
            ),
            "status": _artifact_status(
                "scored_with_exact_frozen_discovery_geometry_replay",
                test_only=test_only,
            ),
            "registration_sha256": expected_registration_sha256,
            "discovery_labels_sha256": discovery_sha256,
            "discovery_geometry_sha256": geometry_sha256,
            "confirmation_reconstruction_manifest_sha256": (
                reconstruction_manifest_sha256
            ),
            "confirmation_labels_sha256": confirmation_sha256,
            "actual_model_forward_examples": actual_total,
            "maximum_registered_model_forwards": 101_504,
            "pure_score": pure_score,
            "claim_boundary": deepcopy(registration["claim_boundary"]),
        }
        score_path = output_directory / "prospective_score.pt"
        atomic_torch_save(score_artifact, score_path)
        score_sha256 = sha256(score_path)
        score_custody = {
            "schema_version": _artifact_schema(
                "e31_2_score_custody_v1", test_only=test_only
            ),
            "status": _artifact_status(
                "e31_2_complete_write_once_score", test_only=test_only
            ),
            "registration_sha256": expected_registration_sha256,
            "global_attempt_sha256": attempt_sha256,
            "clean_screen_sha256": clean_sha256,
            "clean_screen_custody_sha256": clean_custody_sha256,
            "discovery_labels_sha256": discovery_sha256,
            "discovery_custody_sha256": discovery_custody_sha256,
            "discovery_geometry_sha256": geometry_sha256,
            "discovery_geometry_custody_sha256": geometry_custody_sha256,
            "confirmation_reconstruction_manifest_sha256": (
                reconstruction_manifest_sha256
            ),
            "confirmation_authorization_sha256": authorization_sha256,
            "confirmation_labels_sha256": confirmation_sha256,
            "confirmation_custody_sha256": confirmation_custody_sha256,
            "score_attempt_sha256": score_attempt_sha256,
            "score_sha256": score_sha256,
            "actual_model_forward_examples": actual_total,
            "maximum_registered_model_forwards": 101_504,
            "duplicate_clean_forwards": 0,
            "confirmation_opened_only_after_geometry_seal": True,
            "geometry_replayed_exactly": True,
            "parent_selection_was_posthoc_e31_1": True,
        }
        atomic_json_save(score_custody, output_directory / "score_custody.json")
        if forward_counter is not None:
            forward_counter.close()
        return score_custody
    except BaseException as error:
        if forward_counter is not None:
            forward_counter.close()
        _write_failure_custody(
            output_directory,
            registration_sha256=expected_registration_sha256,
            attempt_sha256=attempt_sha256,
            stage=stage,
            error=error,
            test_only=test_only,
            confirmation_patch_started=confirmation_patch_started,
        )
        raise


def run_registered_pipeline(
    *,
    registration_path: Path = DEFAULT_REGISTRATION_PATH,
    expected_registration_sha256: str,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    confirm_model_run: bool,
) -> dict[str, Any]:
    """Run the sole registered production chain with no injectable stages."""
    return _execute_registered_pipeline(
        registration_path=registration_path,
        expected_registration_sha256=expected_registration_sha256,
        output_directory=output_directory,
        confirm_model_run=confirm_model_run,
        runner_factory=_default_runner_factory,
        prepare_all_fn=_default_prepare_all,
        acquire_discovery_fn=_default_acquire_discovery,
        fit_geometry_fn=_default_fit_geometry,
        acquire_confirmation_fn=_default_acquire_confirmation,
        score_fn=_default_score,
        verify_dependencies_fn=verify_registered_dependencies,
        test_only=False,
    )


def _run_test_pipeline(
    *,
    registration_path: Path,
    expected_registration_sha256: str,
    output_directory: Path,
    confirm_model_run: bool,
    runner_factory: Callable[[Mapping[str, Any]], Any],
    prepare_all_fn: Callable[
        [Any, Mapping[str, Any]], tuple[dict[str, Any], dict[str, Any]]
    ],
    acquire_discovery_fn: Callable[[Any, Any, Mapping[str, Any], str], dict[str, Any]],
    fit_geometry_fn: Callable[[Mapping[str, Any]], dict[str, Any]],
    acquire_confirmation_fn: Callable[
        [Any, Any, Mapping[str, Any], Mapping[str, Any], str], dict[str, Any]
    ],
    score_fn: Callable[
        [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], dict[str, Any]
    ],
    verify_dependencies_fn: Callable[[Mapping[str, Any]], Any],
) -> dict[str, Any]:
    """Exercise sequencing with tagged artifacts outside canonical paths only."""
    return _execute_registered_pipeline(
        registration_path=registration_path,
        expected_registration_sha256=expected_registration_sha256,
        output_directory=output_directory,
        confirm_model_run=confirm_model_run,
        runner_factory=runner_factory,
        prepare_all_fn=prepare_all_fn,
        acquire_discovery_fn=acquire_discovery_fn,
        fit_geometry_fn=fit_geometry_fn,
        acquire_confirmation_fn=acquire_confirmation_fn,
        score_fn=score_fn,
        verify_dependencies_fn=verify_dependencies_fn,
        test_only=True,
    )


__all__ = [
    "DEFAULT_OUTPUT_DIRECTORY",
    "DEFAULT_REGISTRATION_PATH",
    "OUTPUT_FILENAMES",
    "prospective_dry_run",
    "run_registered_pipeline",
    "seal_prospective_registration",
]
