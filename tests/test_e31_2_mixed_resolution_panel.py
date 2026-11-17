# ruff: noqa: EXE002
from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.e31_sparse_interactions.mixed_resolution_panel import (
    PROMPT_COUNTS,
    PROMPT_MANIFEST_SHA256,
    PROMPT_RECORD_KEYS,
    build_prompt_manifest,
    canonical_sha256,
    validate_prompt_manifest,
)
from experiments.e31_sparse_interactions.panel import (
    build_prompt_manifest as build_e31_1_prompts,
)


class _ExactPanelTokenizer:
    def __init__(self) -> None:
        self.prompt_ids: dict[str, list[int]] = {}
        self.target_ids: dict[str, int] = {}
        next_target = 20_000
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
        if text.startswith(" ") and text.strip() in self.target_ids:
            return [self.target_ids[text.strip()]]
        return list(self.prompt_ids[text])


def test_fresh_prompt_manifests_are_exact_balanced_and_disjoint() -> None:
    manifests = {
        split: build_prompt_manifest(split) for split in ("discovery", "confirmation")
    }
    old = tuple(build_e31_1_prompts(split) for split in ("discovery", "confirmation"))
    old_rows = old[0] + old[1]
    old_vocab = {
        field: {row[field] for row in old_rows}
        for field in ("name_a", "name_b", "template", "place", "object")
    }

    assert {split: len(rows) for split, rows in manifests.items()} == PROMPT_COUNTS
    assert set(manifests) == {
        "discovery",
        "confirmation",
    }
    for split, records in manifests.items():
        assert canonical_sha256(records) == PROMPT_MANIFEST_SHA256[split]
        assert len({row["pair_id"] for row in records}) == len(records)
        assert len({row["stratum_id"] for row in records}) == 4
        assert {row["split"] for row in records} == {split}
        for record in records:
            assert tuple(record) == PROMPT_RECORD_KEYS
            assert record["base_repeated_subject"] == record["donor_target"]
            assert record["donor_repeated_subject"] == record["base_target"]
            for field, prior_values in old_vocab.items():
                assert record[field] not in prior_values
        for index in range(0, len(records), 2):
            forward, reverse = records[index : index + 2]
            assert forward["base_prompt"] == reverse["donor_prompt"]
            assert forward["donor_prompt"] == reverse["base_prompt"]
            assert forward["base_target"] == reverse["donor_target"]
            assert forward["donor_target"] == reverse["base_target"]

    discovery_vocab = {
        field: {row[field] for row in manifests["discovery"]}
        for field in ("name_a", "name_b", "template", "place", "object")
    }
    for field, values in discovery_vocab.items():
        assert values.isdisjoint({row[field] for row in manifests["confirmation"]})


def test_prompt_token_validation_rejects_any_changed_row() -> None:
    tokenizer = _ExactPanelTokenizer()
    for split in ("discovery", "confirmation"):
        records = build_prompt_manifest(split)
        checks = validate_prompt_manifest(tokenizer, records, split=split)
        assert len(checks) == PROMPT_COUNTS[split]
        assert all(row["differing_prompt_position"] == 2 for row in checks)

        changed = [deepcopy(record) for record in records]
        changed[0]["object"] += "x"
        with pytest.raises(ValueError, match="manifest changed"):
            validate_prompt_manifest(tokenizer, changed, split=split)
