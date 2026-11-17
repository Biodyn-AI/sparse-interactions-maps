# ruff: noqa: EXE002
"""Fresh directed IOI prompts for the response-blind E31.2 head atlas."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

MODEL_ID = "gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SCHEMA_VERSION = "e31_2_fresh_directed_ioi_panel_v1"
PROMPT_COUNTS = {"discovery": 16, "confirmation": 32}
PROMPT_MANIFEST_SHA256 = {
    "discovery": "39dbb936732c6f7876cd7dedf6df861e7b35b3fe368ddf38587da57266438716",
    "confirmation": "1af0b95606361720dd31021c918664f90949bc8c26dcd89f42172030a8f4ca84",
}

PROMPT_RECORD_KEYS = (
    "pair_id",
    "split",
    "stratum_id",
    "template_id",
    "template",
    "name_a",
    "name_b",
    "place",
    "object",
    "base_prompt",
    "donor_prompt",
    "base_target",
    "donor_target",
    "base_repeated_subject",
    "donor_repeated_subject",
)

_TEMPLATES = {
    "discovery": (
        (
            "e31_2_discovery_dusk",
            "At dusk beside the {place}, {a} and {b} compared notes. Much later, {repeated} placed the {object} beside",
        ),
        (
            "e31_2_discovery_pause",
            "During a pause near the {place}, {a} and {b} reviewed a schedule. Before sunset, {repeated} rolled the {object} toward",
        ),
        (
            "e31_2_discovery_rain",
            "As rain fell outside the {place}, {a} and {b} exchanged stories. Afterward, {repeated} set the {object} next to",
        ),
        (
            "e31_2_discovery_corner",
            "In a quiet corner of the {place}, {a} and {b} studied a map. Eventually, {repeated} pushed the {object} over to",
        ),
    ),
    "confirmation": (
        (
            "e31_2_confirmation_tour",
            "After a short tour of the {place}, {a} and {b} rested on a bench. Near closing time, {repeated} slid the {object} across to",
        ),
        (
            "e31_2_confirmation_clouds",
            "While clouds gathered above the {place}, {a} and {b} checked the time. A moment later, {repeated} moved the {object} closer to",
        ),
        (
            "e31_2_confirmation_weekend",
            "On a quiet weekend by the {place}, {a} and {b} discussed their plans. Before departing, {repeated} left the {object} for",
        ),
        (
            "e31_2_confirmation_evening",
            "One evening behind the {place}, {a} and {b} compared their lists. At the end, {repeated} passed the {object} along to",
        ),
    ),
}

_NAMES = {
    "discovery": (
        "Alexander",
        "Angela",
        "Antonio",
        "Bella",
        "Beverly",
        "Brittany",
        "Cassidy",
        "Cole",
        "Daisy",
        "Dominic",
        "Elena",
        "Elijah",
        "Eva",
        "Faith",
        "Gabriel",
        "Isabel",
    ),
    "confirmation": (
        "Jordan",
        "Kimberly",
        "Lily",
        "Mason",
        "Mia",
        "Molly",
        "Morgan",
        "Peyton",
        "Riley",
        "Savannah",
        "Taylor",
        "Xavier",
        "Aurora",
        "Autumn",
        "Carmen",
        "Clara",
        "Diana",
        "Eden",
        "Esther",
        "Hope",
        "Iris",
        "Jenna",
        "Luna",
        "Lydia",
        "Marina",
        "Miranda",
        "Nina",
        "Piper",
        "Regina",
        "Ruby",
        "Summer",
        "Violet",
    ),
}

_PLACES = {
    "discovery": (
        "cinema",
        "gymnasium",
        "meadow",
        "orchard",
        "ranch",
        "university",
        "zoo",
        "beach",
    ),
    "confirmation": (
        "cabin",
        "cathedral",
        "classroom",
        "dock",
        "farm",
        "garage",
        "house",
        "mall",
        "mill",
        "monastery",
        "pub",
        "subway",
        "supermarket",
        "vineyard",
        "waterfall",
        "plaza",
    ),
}

_OBJECTS = {
    "discovery": (
        "ball",
        "basket",
        "clock",
        "coat",
        "doll",
        "drum",
        "fork",
        "guitar",
    ),
    "confirmation": (
        "hat",
        "helmet",
        "jacket",
        "laptop",
        "necklace",
        "phone",
        "pillow",
        "radio",
        "ring",
        "rope",
        "shirt",
        "spoon",
        "tablet",
        "violin",
        "wallet",
        "watch",
    ),
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_split(split: str) -> None:
    if split not in PROMPT_COUNTS:
        raise ValueError("split must be 'discovery' or 'confirmation'")


def _assert_fresh_vocabularies() -> None:
    """Fail closed if E31.2 accidentally reuses any E31.1 panel stratum."""
    from experiments.e31_sparse_interactions.panel import build_prompt_manifest as old

    fields = ("name_a", "name_b", "template", "place", "object")
    prior = old("discovery") + old("confirmation")
    current = _unvalidated_prompt_manifest("discovery") + _unvalidated_prompt_manifest(
        "confirmation"
    )
    for field in fields:
        if {row[field] for row in prior} & {row[field] for row in current}:
            raise RuntimeError(f"E31.2 reused an E31.1 {field}")
    discovery = _unvalidated_prompt_manifest("discovery")
    confirmation = _unvalidated_prompt_manifest("confirmation")
    for field in fields:
        if {row[field] for row in discovery} & {row[field] for row in confirmation}:
            raise RuntimeError(f"E31.2 split vocabularies overlap for {field}")


def _unvalidated_prompt_manifest(split: str) -> tuple[dict[str, str], ...]:
    names = _NAMES[split]
    places = _PLACES[split]
    objects = _OBJECTS[split]
    frame_count = PROMPT_COUNTS[split] // 2
    if not (
        len(names) == 2 * frame_count
        and len(places) == frame_count
        and len(objects) == frame_count
        and frame_count % len(_TEMPLATES[split]) == 0
    ):
        raise RuntimeError("the frozen E31.2 prompt vocabulary changed")
    frames_per_template = frame_count // len(_TEMPLATES[split])
    records: list[dict[str, str]] = []
    for frame_index in range(frame_count):
        template_index = frame_index // frames_per_template
        template_id, template = _TEMPLATES[split][template_index]
        name_a = names[2 * frame_index]
        name_b = names[2 * frame_index + 1]
        values = {
            "a": name_a,
            "b": name_b,
            "place": places[frame_index],
            "object": objects[frame_index],
        }
        prompt_a_repeated = template.format(**values, repeated=name_a)
        prompt_b_repeated = template.format(**values, repeated=name_b)
        for direction, base, donor, base_target, donor_target in (
            ("a_to_b", prompt_b_repeated, prompt_a_repeated, name_a, name_b),
            ("b_to_a", prompt_a_repeated, prompt_b_repeated, name_b, name_a),
        ):
            records.append(
                {
                    "pair_id": (f"e31_2_{split}_frame_{frame_index:02d}_{direction}"),
                    "split": split,
                    "stratum_id": f"e31_2_{split}_stratum_{template_index}",
                    "template_id": template_id,
                    "template": template,
                    "name_a": name_a,
                    "name_b": name_b,
                    "place": places[frame_index],
                    "object": objects[frame_index],
                    "base_prompt": base,
                    "donor_prompt": donor,
                    "base_target": base_target,
                    "donor_target": donor_target,
                    "base_repeated_subject": donor_target,
                    "donor_repeated_subject": base_target,
                }
            )
    return tuple(records)


def build_prompt_manifest(split: str) -> tuple[dict[str, str], ...]:
    """Build an exact fresh directed split without reading model outcomes."""
    _validate_split(split)
    _assert_fresh_vocabularies()
    records = _unvalidated_prompt_manifest(split)
    if (
        PROMPT_MANIFEST_SHA256[split] != "TO_BE_PINNED"
        and canonical_sha256(records) != PROMPT_MANIFEST_SHA256[split]
    ):
        raise RuntimeError(f"the frozen E31.2 {split} prompt hash changed")
    return records


def validate_prompt_manifest(
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> tuple[dict[str, Any], ...]:
    """Verify exact records, one-token targets, and one-position alignment."""
    _validate_split(split)
    copied = tuple(dict(record) for record in records)
    if copied != build_prompt_manifest(split):
        raise ValueError("the frozen E31.2 prompt manifest changed")
    if canonical_sha256(copied) != PROMPT_MANIFEST_SHA256[split]:
        raise RuntimeError("the frozen E31.2 prompt hash changed")
    checks = []
    for record in copied:
        if tuple(record) != PROMPT_RECORD_KEYS:
            raise ValueError("an E31.2 prompt record changed keys or key order")
        base_ids = tokenizer.encode(record["base_prompt"], add_special_tokens=False)
        donor_ids = tokenizer.encode(record["donor_prompt"], add_special_tokens=False)
        base_target = tokenizer.encode(
            " " + record["base_target"], add_special_tokens=False
        )
        donor_target = tokenizer.encode(
            " " + record["donor_target"], add_special_tokens=False
        )
        if len(base_ids) != len(donor_ids):
            raise ValueError(f"{record['pair_id']} has unequal prompt token lengths")
        differing = [
            index
            for index, values in enumerate(zip(base_ids, donor_ids, strict=True))
            if values[0] != values[1]
        ]
        if len(differing) != 1:
            raise ValueError(f"{record['pair_id']} must differ at exactly one token")
        if len(base_target) != 1 or len(donor_target) != 1:
            raise ValueError(f"{record['pair_id']} has a multi-token target")
        if base_target == donor_target:
            raise ValueError(f"{record['pair_id']} target token IDs coincide")
        checks.append(
            {
                **deepcopy(record),
                "token_count": len(base_ids),
                "differing_prompt_position": differing[0],
                "base_target_token_id": int(base_target[0]),
                "donor_target_token_id": int(donor_target[0]),
            }
        )
    return tuple(checks)


__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "PROMPT_COUNTS",
    "PROMPT_MANIFEST_SHA256",
    "PROMPT_RECORD_KEYS",
    "SCHEMA_VERSION",
    "build_prompt_manifest",
    "canonical_sha256",
    "validate_prompt_manifest",
]
