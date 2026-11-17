# ruff: noqa: EXE002
"""Smoke-gated in-memory acquisition for the E32 InterpBench case panel."""

from __future__ import annotations

from typing import Any

import torch

from experiments.e32_interpbench.adapter import (
    CASE_MANIFESTS,
    patchable_internal_universe,
)
from experiments.e32_interpbench.component_patching import (
    BATCH_INVARIANCE_ATOL,
    ComponentPatchRunner,
    build_component_atlas,
    build_patch_units,
    class_center_logits,
    vector_pair_interactions,
)

PRIMARY_CASE_ID = 97
SECONDARY_CASE_IDS = (2, 21)
CASE_PANEL = (PRIMARY_CASE_ID, *SECONDARY_CASE_IDS)
_ELIGIBILITY_INTEGER_FIELDS = (
    "candidate_count",
    "eligible_candidate_count",
    "clean_incorrect_contexts",
    "source_incorrect_contexts",
    "either_incorrect_contexts",
    "eligibility_physical_forwards",
    "eligibility_model_examples",
    "row_disjoint_eligible_candidate_count",
    "row_overlap_excluded_contexts",
    "excluded_token_row_count",
)
_ELIGIBILITY_FIELDS = (
    *_ELIGIBILITY_INTEGER_FIELDS,
    "selected_candidate_indices",
    "output_kind",
    "selection_policy",
)


def _official_tensor(pair: Any, attribute: str) -> torch.Tensor:
    try:
        value = getattr(pair, attribute)
    except AttributeError as error:
        raise TypeError(f"official pair lacks {attribute}") from error
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{attribute} must be a tensor")
    if value.ndim < 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{attribute} must have batch and position axes")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{attribute} must be finite")
    return value.detach().cpu().clone()


def categorical_scientific_output(full_output: torch.Tensor) -> torch.Tensor:
    """Class-center categorical outputs and remove unspecified BOS position 0."""
    if not isinstance(full_output, torch.Tensor) or full_output.ndim < 3:
        raise ValueError("categorical output needs batch, position, and class axes")
    if full_output.shape[-2] < 2:
        raise ValueError("categorical output needs a non-BOS position")
    return class_center_logits(full_output)[..., 1:, :]


def _case_universe(
    case_id: int,
) -> tuple[tuple[Any, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    try:
        manifest = CASE_MANIFESTS[case_id]
    except KeyError as error:
        raise ValueError(f"case_id must be one of {CASE_PANEL!r}") from error
    nodes, edges = patchable_internal_universe(manifest.n_layers, manifest.n_heads)
    units = build_patch_units(manifest.n_layers, manifest.n_heads)
    if tuple(unit.component_id for unit in units) != nodes:
        raise RuntimeError("adapter node universe and patch-unit order disagree")
    return units, nodes, edges


def build_case_patch_runner(
    *,
    case_id: int,
    model: Any,
    condition_batch_size: int = 32,
) -> ComponentPatchRunner:
    """Construct a runner from the adapter's exact case-specific DAG universe."""
    if case_id not in CASE_PANEL:
        raise ValueError(f"case_id must be one of {CASE_PANEL!r}")
    manifest = CASE_MANIFESTS[case_id]
    cfg = getattr(model, "cfg", None)
    if (
        int(getattr(cfg, "n_layers", -1)) != manifest.n_layers
        or int(getattr(cfg, "n_heads", -1)) != manifest.n_heads
    ):
        raise ValueError("model layer/head shape differs from pinned case manifest")
    units, _nodes, edges = _case_universe(case_id)
    return ComponentPatchRunner(
        model,
        units,
        condition_batch_size=condition_batch_size,
        valid_directed_edges=edges,
    )


def _validate_case_runner(case_id: int, runner: ComponentPatchRunner) -> None:
    units, nodes, edges = _case_universe(case_id)
    del units
    if tuple(unit.component_id for unit in runner.units) != nodes:
        raise ValueError("runner units differ from adapter patchable universe")
    if runner.valid_directed_edges != edges:
        raise ValueError("runner edges differ from adapter patchable universe")


def _validated_eligible_pair(value: Any, *, case_id: int) -> tuple[Any, dict[str, Any]]:
    """Validate the narrow public surface of adapter.BehaviorallyEligiblePair."""
    if not hasattr(value, "pair") or any(
        not hasattr(value, field) for field in _ELIGIBILITY_FIELDS
    ):
        raise TypeError(
            "official_pair must be an adapter BehaviorallyEligiblePair; "
            "unfiltered OfficialDataPair is forbidden"
        )
    pair = value.pair
    clean_inputs = _official_tensor(pair, "clean_inputs")
    batch_size = len(clean_inputs)
    metadata: dict[str, Any] = {}
    for field in _ELIGIBILITY_INTEGER_FIELDS:
        field_value = getattr(value, field)
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, int)
            or field_value < 0
        ):
            raise ValueError(f"eligibility {field} must be a nonnegative integer")
        metadata[field] = field_value
    selected = tuple(value.selected_candidate_indices)
    if (
        len(selected) != batch_size
        or len(selected) != len(set(selected))
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= metadata["candidate_count"]
            for index in selected
        )
    ):
        raise ValueError(
            "eligibility selected indices must uniquely align with pair rows"
        )
    if not (
        metadata["candidate_count"]
        >= metadata["eligible_candidate_count"]
        >= metadata["row_disjoint_eligible_candidate_count"]
        >= batch_size
    ):
        raise ValueError("eligibility candidate counts are inconsistent")
    if metadata["row_overlap_excluded_contexts"] != (
        metadata["eligible_candidate_count"]
        - metadata["row_disjoint_eligible_candidate_count"]
    ):
        raise ValueError("eligibility row-overlap exclusion counts disagree")
    for field in (
        "clean_incorrect_contexts",
        "source_incorrect_contexts",
        "either_incorrect_contexts",
    ):
        if metadata[field] > metadata["candidate_count"]:
            raise ValueError(f"eligibility {field} exceeds candidate_count")
    if metadata["either_incorrect_contexts"] != (
        metadata["candidate_count"] - metadata["eligible_candidate_count"]
    ):
        raise ValueError("eligibility rejection and eligible counts disagree")
    if not (
        max(
            metadata["clean_incorrect_contexts"],
            metadata["source_incorrect_contexts"],
        )
        <= metadata["either_incorrect_contexts"]
        <= metadata["clean_incorrect_contexts"] + metadata["source_incorrect_contexts"]
    ):
        raise ValueError("eligibility branch rejection counts are inconsistent")
    if (
        metadata["eligibility_physical_forwards"] != 2
        or metadata["eligibility_model_examples"] != 2 * metadata["candidate_count"]
    ):
        raise ValueError("eligibility query counts differ from adapter protocol")
    expected_output_kind = CASE_MANIFESTS[case_id].output_kind
    if value.output_kind != expected_output_kind:
        raise ValueError("eligibility output kind differs from pinned case manifest")
    if expected_output_kind != "categorical":
        raise NotImplementedError(
            "E32 acquisition currently defines only categorical output projection"
        )
    allowed_policies = {
        "first_jointly_correct",
        (
            "first_jointly_correct_with_globally_unique_clean_source_rows_and_"
            "response_blind_exclusions"
        ),
    }
    if value.selection_policy not in allowed_policies:
        raise ValueError("eligibility selection policy is not registered")
    if value.selection_policy != "first_jointly_correct":
        corrupt_inputs = _official_tensor(pair, "corrupt_inputs")
        rows = tuple(
            tuple(int(token) for token in row) for row in clean_inputs
        ) + tuple(tuple(int(token) for token in row) for row in corrupt_inputs)
        if len(rows) != len(set(rows)):
            raise ValueError("eligibility selected token rows are not globally unique")
    metadata["selected_candidate_indices"] = selected
    metadata["output_kind"] = value.output_kind
    metadata["selection_policy"] = value.selection_policy
    metadata["selected_count"] = batch_size
    metadata["status"] = "adapter_prefiltered_and_runner_rechecked_before_patch"
    return pair, metadata


def behavioral_pair_screen(
    *,
    clean_output: torch.Tensor,
    corrupt_output: torch.Tensor,
    clean_targets: torch.Tensor,
    corrupt_targets: torch.Tensor,
    exclude_bos: bool = True,
) -> dict[str, Any]:
    """Record exact branch correctness for every official paired context."""
    tensors = {
        "clean_output": clean_output,
        "corrupt_output": corrupt_output,
        "clean_targets": clean_targets,
        "corrupt_targets": corrupt_targets,
    }
    if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError("behavior-screen inputs must be tensors")
    if clean_output.shape != clean_targets.shape or corrupt_output.shape != (
        corrupt_targets.shape
    ):
        raise ValueError("each output must have the same shape as its targets")
    if clean_output.ndim != 3 or corrupt_output.ndim != 3:
        raise ValueError("behavior-screen tensors must be [batch, position, class]")
    if clean_output.shape[:2] != corrupt_output.shape[:2]:
        raise ValueError("clean and corrupt batches must align in shape")
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError("behavior-screen tensors must be finite")

    clean_matches = clean_output.argmax(dim=-1) == clean_targets.argmax(dim=-1)
    corrupt_matches = corrupt_output.argmax(dim=-1) == corrupt_targets.argmax(dim=-1)
    if exclude_bos:
        if clean_matches.shape[1] < 2:
            raise ValueError("cannot exclude BOS from a one-position sequence")
        clean_matches = clean_matches[:, 1:]
        corrupt_matches = corrupt_matches[:, 1:]
    clean_context = clean_matches.all(dim=1).cpu()
    corrupt_context = corrupt_matches.all(dim=1).cpu()
    both = torch.logical_and(clean_context, corrupt_context)
    return {
        "exclude_bos": exclude_bos,
        "context_count": len(both),
        "evaluated_token_count_per_context": clean_matches.shape[1],
        "clean_context_correct": clean_context,
        "corrupt_context_correct": corrupt_context,
        "both_context_correct": both,
        "clean_context_correct_count": int(clean_context.sum()),
        "corrupt_context_correct_count": int(corrupt_context.sum()),
        "both_context_correct_count": int(both.sum()),
        "all_pairs_correct": bool(both.all()),
        "row_policy": "retain_all_and_publish_per_context_correctness",
    }


def acquire_official_pair_atlas(
    *,
    case_id: int,
    runner: ComponentPatchRunner,
    official_pair: Any,
    smoke_condition_count: int = 16,
    smoke_condition_batch_size: int = 8,
    smoke_absolute_tolerance: float = BATCH_INVARIANCE_ATOL,
) -> dict[str, Any]:
    """Smoke first, then acquire an exhaustive vector-output component atlas.

    This function performs no filesystem writes and no model loading.  The E32
    adapter owns official artifacts and pinned model construction; this layer
    owns only intervention execution and exact response custody.
    """
    if case_id not in CASE_PANEL:
        raise ValueError(f"case_id must be one of {CASE_PANEL!r}")
    if not isinstance(runner, ComponentPatchRunner):
        raise TypeError("runner must be a ComponentPatchRunner")
    _validate_case_runner(case_id, runner)
    selected_pair, eligibility = _validated_eligible_pair(
        official_pair, case_id=case_id
    )
    clean_inputs = _official_tensor(selected_pair, "clean_inputs")
    corrupt_inputs = _official_tensor(selected_pair, "corrupt_inputs")
    clean_targets = _official_tensor(selected_pair, "clean_targets")
    corrupt_targets = _official_tensor(selected_pair, "corrupt_targets")
    if clean_inputs.shape != corrupt_inputs.shape:
        raise ValueError("official clean/corrupt input shapes differ")
    if tuple(clean_targets.shape[:2]) != tuple(clean_inputs.shape) or tuple(
        corrupt_targets.shape[:2]
    ) != tuple(corrupt_inputs.shape):
        raise ValueError("official targets do not align with inputs")
    if (
        isinstance(smoke_condition_count, bool)
        or not isinstance(smoke_condition_count, int)
        or smoke_condition_count < 1
    ):
        raise ValueError("smoke_condition_count must be a positive integer")

    before = runner.forward_counts()
    prepared = runner.prepare_pair(selected_pair)
    behavior_screen = behavioral_pair_screen(
        clean_output=prepared.clean_output,
        corrupt_output=prepared.corrupt_output,
        clean_targets=clean_targets,
        corrupt_targets=corrupt_targets,
    )
    behavior_screen["row_policy"] = (
        "require_every_selected_clean_and_source_context_correct"
    )
    if not behavior_screen["all_pairs_correct"]:
        raise RuntimeError(
            "behavioral eligibility recheck failed before any patched response"
        )
    manifest = build_component_atlas(
        runner.units, valid_directed_edges=runner.valid_directed_edges
    )
    if smoke_condition_count > manifest.condition_count:
        raise ValueError("smoke_condition_count exceeds the exhaustive atlas")
    smoke = runner.batch_invariance_smoke(
        prepared,
        manifest.condition_masks[:smoke_condition_count],
        candidate_condition_batch_size=smoke_condition_batch_size,
        absolute_tolerance=smoke_absolute_tolerance,
    )
    if not smoke["passed"]:
        raise RuntimeError("batch-invariance smoke failed; full atlas blocked")
    atlas = runner.acquire_atlas(prepared)
    scientific_condition_outputs = categorical_scientific_output(
        atlas.condition_outputs
    )
    scientific_singleton_outputs = scientific_condition_outputs[
        list(atlas.atlas.singleton_condition_indices)
    ]
    scientific_pair_outputs = scientific_condition_outputs[
        list(atlas.atlas.pair_condition_indices)
    ]
    scientific_pair_interactions = vector_pair_interactions(
        empty_output=scientific_condition_outputs[0],
        singleton_outputs=scientific_singleton_outputs,
        pair_outputs=scientific_pair_outputs,
        pair_component_indices=atlas.atlas.pair_component_indices,
        center=False,
    )
    if not torch.equal(
        scientific_pair_interactions, atlas.pair_interactions[..., 1:, :]
    ):
        raise RuntimeError("scientific interaction projection changed exact J")
    after = runner.forward_counts()
    return {
        "schema_version": "e32_interpbench_component_atlas_v2",
        "case_id": case_id,
        "case_role": "primary" if case_id == PRIMARY_CASE_ID else "secondary",
        "output_kind": eligibility["output_kind"],
        "patch_scope": "whole_head_hook_result_and_mlp_hook_mlp_out_all_positions",
        "output_scope": "full_per_example_per_position_output_tensor",
        "component_ids": tuple(unit.component_id for unit in runner.units),
        "condition_ids": atlas.atlas.condition_ids,
        "condition_masks": atlas.atlas.condition_masks,
        "pair_component_indices": atlas.atlas.pair_component_indices,
        "directed_candidate_edges": atlas.atlas.directed_edges,
        "candidate_edge_universe_source": ("adapter.patchable_internal_universe"),
        "pair_patch_semantics": (
            "symmetric_two_component_mask; edge_arrow_is_topology_imposed"
        ),
        "clean_inputs": clean_inputs,
        "corrupt_inputs": corrupt_inputs,
        "clean_targets": clean_targets,
        "corrupt_targets": corrupt_targets,
        "clean_output": prepared.clean_output,
        "corrupt_output": prepared.corrupt_output,
        "behavioral_pair_screen": behavior_screen,
        "behavioral_eligibility": eligibility,
        "eligibility_query_accounting": {
            "physical_forwards_before_runner": eligibility[
                "eligibility_physical_forwards"
            ],
            "model_examples_before_runner": eligibility["eligibility_model_examples"],
            "source": "adapter_behaviorally_eligible_pair",
        },
        "condition_outputs": atlas.condition_outputs,
        "singleton_outputs": atlas.singleton_outputs,
        "pair_outputs": atlas.pair_outputs,
        "scientific_condition_outputs": scientific_condition_outputs,
        "scientific_singleton_outputs": scientific_singleton_outputs,
        "scientific_pair_outputs": scientific_pair_outputs,
        "pair_interactions": scientific_pair_interactions,
        "raw_pair_interactions": atlas.raw_pair_interactions,
        "scientific_output_coordinate": ("class_centered_logits_non_bos_positions"),
        "interaction_definition": (
            "J=center(Yij)-center(Yi)-center(Yj)+center(Y0), positions 1:"
        ),
        "raw_interaction_status": "descriptive_only_logit_gauge_dependent",
        "batch_invariance_smoke": smoke,
        "atlas_query_accounting": dict(atlas.query_accounting),
        "total_live_query_accounting": {
            "physical_forwards": after["physical_forwards"]
            - before["physical_forwards"],
            "model_examples": after["model_examples"] - before["model_examples"],
            "includes_clean_corrupt_preparation": True,
            "includes_batch_invariance_smoke": True,
            "includes_exhaustive_atlas": True,
            "excludes_pre_runner_eligibility": True,
        },
        "total_query_accounting_including_eligibility": {
            "physical_forwards": eligibility["eligibility_physical_forwards"]
            + after["physical_forwards"]
            - before["physical_forwards"],
            "model_examples": eligibility["eligibility_model_examples"]
            + after["model_examples"]
            - before["model_examples"],
        },
    }


__all__ = [
    "CASE_PANEL",
    "PRIMARY_CASE_ID",
    "SECONDARY_CASE_IDS",
    "acquire_official_pair_atlas",
    "behavioral_pair_screen",
    "build_case_patch_runner",
    "categorical_scientific_output",
]
