# ruff: noqa: EXE002
"""Fresh response-blind IOI panels for the prospective E31.1 atlas.

The panel is deliberately separate from the observed E29.15 population.  Each
semantic frame is used in both intervention directions, while names and
templates are disjoint between discovery and confirmation.  This module makes
no model call and performs no outcome-dependent prompt selection.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

MODEL_ID = "gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SCHEMA_VERSION = "e31_1_fresh_directed_ioi_panel_v1"
PROMPT_COUNTS = {"discovery": 16, "confirmation": 32}
PROMPT_MANIFEST_SHA256 = {
    "discovery": "11a7ea875031b6200e6dc95c4f1668f8ea83e44dab4304577cef077c82c41ecd",
    "confirmation": "db2dd2d98f990d8aecc02779463342634ba476a44a0a99b567de6ed0e63d9b09",
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
            "e31_discovery_lunch",
            "At the {place}, {a} and {b} met for lunch. Later, {repeated} mailed the {object} to",
        ),
        (
            "e31_discovery_break",
            "Near the {place}, {a} and {b} took a break. Afterwards, {repeated} lent the {object} to",
        ),
        (
            "e31_discovery_quietly",
            "Inside the {place}, {a} and {b} spoke quietly. Eventually, {repeated} tossed the {object} to",
        ),
        (
            "e31_discovery_walking",
            "Outside the {place}, {a} and {b} stopped walking. Soon, {repeated} forwarded the {object} to",
        ),
    ),
    "confirmation": (
        (
            "e31_confirmation_briefly",
            "Once {a} and {b} entered the {place}, they talked briefly. Later, {repeated} shipped the {object} to",
        ),
        (
            "e31_confirmation_together",
            "Although the {place} was busy, {a} and {b} waited together. Finally, {repeated} loaned the {object} to",
        ),
        (
            "e31_confirmation_morning",
            "This morning at the {place}, {a} and {b} shared a table. Afterwards, {repeated} carried the {object} to",
        ),
        (
            "e31_confirmation_appointment",
            "Following an appointment at the {place}, {a} and {b} stayed behind. Then, {repeated} returned the {object} to",
        ),
    ),
}

_NAMES = {
    "discovery": (
        "Amelia",
        "Caleb",
        "Charlotte",
        "Connor",
        "Ellie",
        "Harrison",
        "Jade",
        "Julian",
        "Katherine",
        "Leo",
        "Maya",
        "Miles",
        "Samantha",
        "Sean",
        "Valerie",
        "Walter",
    ),
    "confirmation": (
        "Alexandra",
        "Anthony",
        "Allison",
        "Austin",
        "Amber",
        "Bradley",
        "Andrea",
        "Brandon",
        "Ashley",
        "Bryan",
        "Bailey",
        "Chad",
        "Caroline",
        "Christian",
        "Catherine",
        "Christopher",
        "Chelsea",
        "Craig",
        "Cheryl",
        "Dalton",
        "Christine",
        "Darren",
        "Courtney",
        "Dawson",
        "Cynthia",
        "Dean",
        "Dakota",
        "Dennis",
        "Eleanor",
        "Devin",
        "Elizabeth",
        "Douglas",
    ),
}

_PLACES = {
    "discovery": (
        "observatory",
        "aquarium",
        "greenhouse",
        "planetarium",
        "archive",
        "lighthouse",
        "pavilion",
        "terminal",
    ),
    "confirmation": (
        "laboratory",
        "auditorium",
        "cottage",
        "tower",
        "marina",
        "conservatory",
        "depot",
        "chapel",
        "nursery",
        "mansion",
        "cafe",
        "arena",
        "fortress",
        "pier",
        "balcony",
        "courtyard",
    ),
}

_OBJECTS = {
    "discovery": (
        "notebook",
        "lantern",
        "magazine",
        "blanket",
        "camera",
        "compass",
        "pencil",
        "envelope",
    ),
    "confirmation": (
        "umbrella",
        "newspaper",
        "backpack",
        "bracelet",
        "dictionary",
        "microphone",
        "sweater",
        "suitcase",
        "calendar",
        "telescope",
        "headphones",
        "sandwich",
        "flashlight",
        "clipboard",
        "photograph",
        "calculator",
    ),
}


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible protocol data using one strict encoding."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_split(split: str) -> None:
    if split not in PROMPT_COUNTS:
        raise ValueError("split must be 'discovery' or 'confirmation'")


def build_prompt_manifest(split: str) -> tuple[dict[str, str], ...]:
    """Return the immutable directed IOI records for one split."""
    _validate_split(split)
    names = _NAMES[split]
    places = _PLACES[split]
    objects = _OBJECTS[split]
    frame_count = PROMPT_COUNTS[split] // 2
    if len(names) != 2 * frame_count or len(places) != frame_count:
        raise RuntimeError("the frozen E31 prompt vocabulary changed")
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
        directions = (
            ("a_to_b", prompt_b_repeated, prompt_a_repeated, name_a, name_b),
            ("b_to_a", prompt_a_repeated, prompt_b_repeated, name_b, name_a),
        )
        for (
            direction_id,
            base_prompt,
            donor_prompt,
            base_target,
            donor_target,
        ) in directions:
            records.append(
                {
                    "pair_id": f"e31_{split}_frame_{frame_index:02d}_{direction_id}",
                    "split": split,
                    "stratum_id": f"{split}_stratum_{template_index}",
                    "template_id": template_id,
                    "template": template,
                    "name_a": name_a,
                    "name_b": name_b,
                    "place": places[frame_index],
                    "object": objects[frame_index],
                    "base_prompt": base_prompt,
                    "donor_prompt": donor_prompt,
                    "base_target": base_target,
                    "donor_target": donor_target,
                    "base_repeated_subject": donor_target,
                    "donor_repeated_subject": base_target,
                }
            )
    return tuple(records)


def validate_prompt_manifest(
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> tuple[dict[str, Any], ...]:
    """Verify the frozen panel and exact GPT-2 token alignment."""
    _validate_split(split)
    copied = tuple(dict(record) for record in records)
    if copied != build_prompt_manifest(split):
        raise ValueError("the frozen E31 prompt manifest changed")
    if canonical_sha256(copied) != PROMPT_MANIFEST_SHA256[split]:
        raise RuntimeError("the frozen E31 prompt hash changed")
    checks: list[dict[str, Any]] = []
    for record in copied:
        if tuple(record) != PROMPT_RECORD_KEYS:
            raise ValueError("an E31 prompt record has changed keys or key order")
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
