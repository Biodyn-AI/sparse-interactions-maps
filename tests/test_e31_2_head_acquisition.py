# ruff: noqa: EXE002
from __future__ import annotations

from copy import deepcopy

import pytest
import torch

import experiments.e31_sparse_interactions.e31_2_head_acquisition as acquisition
from experiments.e31_sparse_interactions.e31_2_head_acquisition import (
    acquire_head_atlas_from_prepared,
    acquire_reconstruction_from_prepared,
    masks_from_atlas_manifest,
    masks_from_reconstruction_manifest,
    prepare_and_screen_all_registered_contexts,
    validate_atlas_labels,
    validate_reconstruction_labels,
)
from experiments.e31_sparse_interactions.e31_2_head_patch_runner import (
    E312HeadPatchRunner,
)
from experiments.e31_sparse_interactions.mixed_resolution_protocol import (
    build_atlas_manifest,
    build_reconstruction_manifest,
)
from tests.test_e31_2_head_patch_runner import (
    _HeadToyModel,
    _record,
    _Tokenizer,
)


def _records(split: str, count: int) -> tuple[dict[str, str], ...]:
    return tuple(
        _record(
            pair_id=f"e31-2-{split}-{index:03d}",
            split=split,
            reverse=bool(index % 2),
        )
        for index in range(count)
    )


def _patch_prompt_panel(
    monkeypatch: pytest.MonkeyPatch,
    records_by_split: dict[str, tuple[dict[str, str], ...]],
) -> None:
    monkeypatch.setattr(
        acquisition,
        "build_prompt_manifest",
        lambda split: deepcopy(records_by_split[split]),
    )
    for split, records in records_by_split.items():
        monkeypatch.setitem(
            acquisition.PROMPT_MANIFEST_SHA256,
            split,
            acquisition.canonical_sha256(records),
        )


def _selections() -> dict[str, object]:
    return {
        "schema_version": "e31_2_discovery_rectangle_selections_v1",
        "parents": [
            {
                "parent_pair_id": parent,
                "pair_selected": {
                    "heads_a": [0, 1, 2, 3],
                    "heads_b": [0, 1, 2, 3],
                },
                "singleton_selected": {
                    "heads_a": [4, 5, 6, 7],
                    "heads_b": [4, 5, 6, 7],
                },
            }
            for parent in ("l05a_l08a", "l08a_l09a", "l05a_l10a")
        ],
    }


def test_frozen_atlas_masks_have_exact_global_shapes_and_no_overlap() -> None:
    manifest = build_atlas_manifest("discovery")

    condition_ids, coarse, heads = masks_from_atlas_manifest(manifest)

    assert len(condition_ids) == 2064
    assert len(set(condition_ids)) == 2064
    assert coarse.shape == (2064, 24)
    assert heads.shape == (2064, 12, 12)
    assert coarse.dtype == heads.dtype == torch.uint8
    assert not bool(torch.logical_and(coarse[:, 0::2].bool(), heads.any(2)).any())


def test_global_clean_screen_prepares_all_48_exactly_once_and_keeps_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = {
        "discovery": _records("discovery", 16),
        "confirmation": _records("confirmation", 32),
    }
    _patch_prompt_panel(monkeypatch, records)
    model = _HeadToyModel().eval()
    runner = E312HeadPatchRunner(model, _Tokenizer(), batch_size=256)
    forward_count = 0

    def count_forward(
        _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...]
    ) -> None:
        nonlocal forward_count
        forward_count += 1

    handle = model.register_forward_pre_hook(count_forward)
    try:
        prepared, screen = prepare_and_screen_all_registered_contexts(runner, records)
    finally:
        handle.remove()

    assert forward_count == 96
    assert len(prepared["discovery"]) == 16
    assert len(prepared["confirmation"]) == 32
    assert screen["total_context_count"] == 48
    assert screen["total_clean_forward_examples"] == 96
    assert screen["all_splits_passed"]
    assert screen["all_head_norm_diagnostics_passed"]
    for split in acquisition.SPLITS:
        for context, record in zip(prepared[split], records[split], strict=True):
            for key in acquisition.PROMPT_RECORD_KEYS:
                assert context.clean_metadata[key] == record[key]


def test_atlas_acquisition_smokes_first_and_assembles_exact_quartets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One toy context keeps this unit test cheap; production counts remain 16/32.
    records = {
        "discovery": _records("discovery", 1),
        "confirmation": _records("confirmation", 1),
    }
    _patch_prompt_panel(monkeypatch, records)
    monkeypatch.setitem(acquisition.PROMPT_COUNTS, "discovery", 1)
    runner = E312HeadPatchRunner(_HeadToyModel().eval(), _Tokenizer(), batch_size=512)
    prepared = (runner.prepare_context(records["discovery"][0]),)
    manifest = build_atlas_manifest("discovery")
    smoke_indices = tuple(range(16))
    events: list[str] = []
    original_smoke = runner.batch_invariance_smoke
    original_responses = runner.responses_from_prepared

    def smoke(*args: object, **kwargs: object) -> dict[str, object]:
        events.append("smoke_start")
        value = original_smoke(*args, **kwargs)
        events.append("smoke_end")
        return value

    def responses(*args: object, **kwargs: object):
        coarse = args[1]
        events.append(f"responses_{len(coarse)}")
        return original_responses(*args, **kwargs)

    monkeypatch.setattr(runner, "batch_invariance_smoke", smoke)
    monkeypatch.setattr(runner, "responses_from_prepared", responses)
    artifact = acquire_head_atlas_from_prepared(
        runner,
        prepared,
        manifest,
        split="discovery",
        smoke_condition_indices=smoke_indices,
        protocol_commitments={"registration_sha256": "1" * 64},
    )

    assert events.index("smoke_end") < events.index("responses_2064")
    assert artifact["normalized_responses"].shape == (2064, 1)
    assert artifact["normalized_fine_quartets"].shape == (3, 144, 1, 4, 4)
    assert artifact["normalized_all_head_quartets"].shape == (3, 1, 4, 4)
    assert artifact["raw_margin_fine_quartets"].shape == (3, 144, 1, 4, 4)
    assert artifact["batch_invariance_smoke"]["passed"]
    assert artifact["composite_reconstruction"]["passed"]
    assert artifact["whole_attention_response_equivalence"] == {
        "evaluated": False,
        "reason": "no_extra_model_calls_and_no_fresh_coarse_reference",
        "runner_regression_tolerance": 1e-9,
    }
    assert artifact["query_accounting"] == {
        "context_count": 1,
        "condition_rows": 2064,
        "patched_condition_context_cells": 2064,
        "clean_forward_examples": 0,
        "clean_forward_examples_are_in_global_screen": True,
        "whole_attention_equivalence_additional_context_cells": 0,
        "batch_smoke_condition_rows": 16,
        "batch_smoke_physical_context_cells": 32,
    }
    validate_atlas_labels(artifact, manifest=manifest)
    tampered = dict(artifact)
    tampered["prompt_records"] = (
        {**artifact["prompt_records"][0], "pair_id": "mutated"},
    )
    with pytest.raises(RuntimeError, match="prompt records"):
        validate_atlas_labels(tampered, manifest=manifest)


def test_failed_discovery_smoke_prevents_every_full_atlas_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = {
        "discovery": _records("discovery", 1),
        "confirmation": _records("confirmation", 1),
    }
    _patch_prompt_panel(monkeypatch, records)
    monkeypatch.setitem(acquisition.PROMPT_COUNTS, "discovery", 1)
    runner = E312HeadPatchRunner(_HeadToyModel().eval(), _Tokenizer(), batch_size=512)
    prepared = (runner.prepare_context(records["discovery"][0]),)
    atlas_call_count = 0

    def failed_smoke(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"passed": False}

    def count_atlas_call(*_args: object, **_kwargs: object) -> None:
        nonlocal atlas_call_count
        atlas_call_count += 1
        raise AssertionError("full atlas acquisition ran after failed smoke")

    monkeypatch.setattr(runner, "batch_invariance_smoke", failed_smoke)
    monkeypatch.setattr(runner, "responses_from_prepared", count_atlas_call)
    with pytest.raises(RuntimeError, match="batch-invariance smoke failed"):
        acquire_head_atlas_from_prepared(
            runner,
            prepared,
            build_atlas_manifest("discovery"),
            split="discovery",
            smoke_condition_indices=tuple(range(16)),
        )
    assert atlas_call_count == 0


def test_dynamic_reconstruction_acquires_only_new_rows_and_checks_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = {
        "discovery": _records("discovery", 1),
        "confirmation": _records("confirmation", 1),
    }
    _patch_prompt_panel(monkeypatch, records)
    monkeypatch.setitem(acquisition.PROMPT_COUNTS, "confirmation", 1)
    runner = E312HeadPatchRunner(_HeadToyModel().eval(), _Tokenizer(), batch_size=512)
    prepared = (runner.prepare_context(records["confirmation"][0]),)
    atlas_manifest = build_atlas_manifest("confirmation")
    atlas = acquire_head_atlas_from_prepared(
        runner,
        prepared,
        atlas_manifest,
        split="confirmation",
        smoke_condition_indices=(),
    )
    reconstruction_manifest = build_reconstruction_manifest(_selections())
    new_ids, coarse, heads = masks_from_reconstruction_manifest(reconstruction_manifest)

    labels = acquire_reconstruction_from_prepared(
        runner,
        prepared,
        reconstruction_manifest,
        atlas,
        protocol_commitments={"authorization_sha256": "2" * 64},
    )

    assert len(new_ids) == reconstruction_manifest["actual_new_condition_count"]
    assert len(new_ids) <= 72
    assert coarse.shape == (len(new_ids), 24)
    assert heads.shape == (len(new_ids), 12, 12)
    assert labels["normalized_new_responses"].shape == (len(new_ids), 1)
    assert labels["normalized_direct_group_quartets"].shape == (3, 2, 1, 4, 4)
    assert labels["normalized_all12_reference_quartets"].shape == (3, 1, 4, 4)
    assert labels["query_accounting"]["patched_new_condition_context_cells"] == len(
        new_ids
    )
    assert labels["query_accounting"]["duplicate_inherited_condition_calls"] == 0
    validate_reconstruction_labels(
        labels,
        reconstruction_manifest=reconstruction_manifest,
        confirmation_atlas_labels=atlas,
    )

    tampered = dict(labels)
    tampered["raw_shift_new_responses"] = labels["raw_shift_new_responses"].clone()
    tampered["raw_shift_new_responses"][0, 0] += 0.01
    # Keep derived quartet views untouched: the raw-coordinate identity must
    # reject the mutation before any scientific score can consume it.
    with pytest.raises(RuntimeError, match="response identity"):
        validate_reconstruction_labels(
            tampered,
            reconstruction_manifest=reconstruction_manifest,
            confirmation_atlas_labels=atlas,
        )

    prompt_tampered = dict(labels)
    prompt_tampered["prompt_records"] = (
        {**labels["prompt_records"][0], "pair_id": "mutated"},
    )
    with pytest.raises(RuntimeError, match="frozen prompt records"):
        validate_reconstruction_labels(
            prompt_tampered,
            reconstruction_manifest=reconstruction_manifest,
            confirmation_atlas_labels=atlas,
        )


def test_prepared_prompt_mutation_fails_before_any_atlas_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = {
        "discovery": _records("discovery", 1),
        "confirmation": _records("confirmation", 1),
    }
    _patch_prompt_panel(monkeypatch, records)
    monkeypatch.setitem(acquisition.PROMPT_COUNTS, "discovery", 1)
    runner = E312HeadPatchRunner(_HeadToyModel().eval(), _Tokenizer(), batch_size=512)
    mutated = deepcopy(records["discovery"][0])
    mutated["pair_id"] = "unregistered-pair"
    prepared = (runner.prepare_context(mutated),)

    with pytest.raises(ValueError, match="frozen panel"):
        acquire_head_atlas_from_prepared(
            runner,
            prepared,
            build_atlas_manifest("discovery"),
            split="discovery",
            smoke_condition_indices=tuple(range(16)),
        )
