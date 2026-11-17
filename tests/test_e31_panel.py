# ruff: noqa: EXE002
from __future__ import annotations

from transformers import AutoTokenizer

from experiments.e29_gpt2_stable_head_interactions.e29_15_mechanism_discovery_protocol import (
    build_prompt_manifest as build_e29_15_prompt_manifest,
)
from experiments.e31_sparse_interactions.panel import (
    MODEL_ID,
    MODEL_REVISION,
    PROMPT_COUNTS,
    PROMPT_MANIFEST_SHA256,
    build_prompt_manifest,
    canonical_sha256,
    validate_prompt_manifest,
)


def test_fresh_panel_has_exact_balanced_directed_frames() -> None:
    discovery = build_prompt_manifest("discovery")
    confirmation = build_prompt_manifest("confirmation")

    assert len(discovery) == PROMPT_COUNTS["discovery"] == 16
    assert len(confirmation) == PROMPT_COUNTS["confirmation"] == 32
    assert len({row["pair_id"] for row in (*discovery, *confirmation)}) == 48
    assert len({row["template_id"] for row in discovery}) == 4
    assert len({row["template_id"] for row in confirmation}) == 4
    assert {row["template_id"] for row in discovery}.isdisjoint(
        {row["template_id"] for row in confirmation}
    )

    for records in (discovery, confirmation):
        by_frame: dict[str, list[dict[str, str]]] = {}
        for record in records:
            frame_id = record["pair_id"].rsplit("_", 3)[0]
            by_frame.setdefault(frame_id, []).append(record)
        assert all(len(rows) == 2 for rows in by_frame.values())
        for rows in by_frame.values():
            forward, reverse = sorted(rows, key=lambda row: row["pair_id"])
            assert forward["base_prompt"] == reverse["donor_prompt"]
            assert forward["donor_prompt"] == reverse["base_prompt"]
            assert forward["base_target"] == reverse["donor_target"]
            assert forward["donor_target"] == reverse["base_target"]

    assert canonical_sha256(discovery) == PROMPT_MANIFEST_SHA256["discovery"]
    assert canonical_sha256(confirmation) == PROMPT_MANIFEST_SHA256["confirmation"]


def test_panel_is_disjoint_from_e29_15_and_tokenizer_valid() -> None:
    discovery = build_prompt_manifest("discovery")
    confirmation = build_prompt_manifest("confirmation")
    old_records = (
        *build_e29_15_prompt_manifest("discovery"),
        *build_e29_15_prompt_manifest("confirmation"),
    )
    old_names = {name for row in old_records for name in (row["name_a"], row["name_b"])}
    old_templates = {row["template"] for row in old_records}
    assert {
        name
        for row in (*discovery, *confirmation)
        for name in (row["name_a"], row["name_b"])
    }.isdisjoint(old_names)
    assert {row["template"] for row in (*discovery, *confirmation)}.isdisjoint(
        old_templates
    )
    assert (
        {row["name_a"] for row in discovery} | {row["name_b"] for row in discovery}
    ).isdisjoint(
        {row["name_a"] for row in confirmation}
        | {row["name_b"] for row in confirmation}
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    checked_discovery = validate_prompt_manifest(
        tokenizer, discovery, split="discovery"
    )
    checked_confirmation = validate_prompt_manifest(
        tokenizer, confirmation, split="confirmation"
    )
    assert len(checked_discovery) == 16
    assert len(checked_confirmation) == 32
    assert all(
        row["token_count"] > 0 for row in (*checked_discovery, *checked_confirmation)
    )
