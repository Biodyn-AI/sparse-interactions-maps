# ruff: noqa: EXE002
"""Pure exhaustive-pair manifest for the exploratory E31 development atlas.

The eight backgrounds are inherited unchanged from E29.15.  For each of the
276 unordered component pairs, this module records the complete ordered
``00, 10, 01, 11`` lattice after forcing the tested pair out of the
background.  Conditions are globally de-duplicated before any model call.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from itertools import combinations
from typing import Any

from experiments.e29_gpt2_stable_head_interactions.e29_15_mechanism_discovery_protocol import (
    build_effect_background_manifest,
)

SCHEMA_VERSION = "e31_1_exploratory_exhaustive_pair_atlas_v1"
COMPONENT_COUNT = 24
HUB_COMPONENT = 1
BACKGROUND_COUNT = 8
PAIR_COUNT = 276
ORDERED_CELL_BITS = ("00", "10", "01", "11")
EXPECTED_BACKGROUND_MANIFEST_SHA256 = (
    "af578c2262a912f5e5e8e3db3778e84a8ad9b69f7baadee24fa79270a23ac7ae"
)


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible protocol data with a stable strict encoding."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _inherited_backgrounds() -> tuple[dict[str, Any], ...]:
    backgrounds = tuple(deepcopy(build_effect_background_manifest()))
    if len(backgrounds) != BACKGROUND_COUNT:
        raise RuntimeError("the inherited E29.15 background count changed")
    if canonical_sha256(backgrounds) != EXPECTED_BACKGROUND_MANIFEST_SHA256:
        raise RuntimeError("the inherited E29.15 background manifest changed")
    return backgrounds


def build_exhaustive_pair_atlas() -> dict[str, Any]:
    """Build all exact pair quartets and one de-duplicated condition table."""
    backgrounds = _inherited_backgrounds()
    condition_index_by_components: dict[tuple[int, ...], int] = {}
    conditions: list[dict[str, Any]] = []
    quartets: list[dict[str, Any]] = []

    def condition_id(
        components: tuple[int, ...],
        *,
        quartet_id: str,
        quartet_index: int,
        cell_index: int,
        cell_bits: str,
    ) -> str:
        selected = tuple(sorted(set(components)))
        if selected not in condition_index_by_components:
            index = len(conditions)
            condition_index_by_components[selected] = index
            mask = [int(component in selected) for component in range(COMPONENT_COUNT)]
            conditions.append(
                {
                    "condition_index": index,
                    "condition_id": f"e31_condition_{index:04d}",
                    "selected_components": list(selected),
                    "cardinality": len(selected),
                    "mask": mask,
                    "mask_sha256": canonical_sha256(mask),
                    "quartet_references": [],
                }
            )
        index = condition_index_by_components[selected]
        identifier = conditions[index]["condition_id"]
        conditions[index]["quartet_references"].append(
            {
                "quartet_index": quartet_index,
                "quartet_id": quartet_id,
                "cell_index": cell_index,
                "cell_bits": cell_bits,
            }
        )
        return identifier

    pair_order = tuple(combinations(range(COMPONENT_COUNT), 2))
    if len(pair_order) != PAIR_COUNT:
        raise RuntimeError("the 24-component pair count changed")
    for background in backgrounds:
        background_index = int(background["background_index"])
        background_id = str(background["background_id"])
        background_components = tuple(int(v) for v in background["selected_components"])
        background_hash = canonical_sha256(list(background_components))
        for pair_index, (first, second) in enumerate(pair_order):
            quartet_index = len(quartets)
            quartet_id = (
                f"background_{background_index:02d}::pair_{first:02d}_{second:02d}"
            )
            pair = (first, second)
            base = tuple(
                component
                for component in background_components
                if component not in pair
            )
            cell_components = (
                base,
                tuple(sorted((*base, first))),
                tuple(sorted((*base, second))),
                tuple(sorted((*base, first, second))),
            )
            identifiers = [
                condition_id(
                    components,
                    quartet_id=quartet_id,
                    quartet_index=quartet_index,
                    cell_index=cell_index,
                    cell_bits=ORDERED_CELL_BITS[cell_index],
                )
                for cell_index, components in enumerate(cell_components)
            ]
            edge_class = "hub_spoke" if HUB_COMPONENT in pair else "nonhub_pair"
            hub_state = (
                "crossing_not_applicable"
                if edge_class == "hub_spoke"
                else ("on" if HUB_COMPONENT in base else "off")
            )
            quartets.append(
                {
                    "quartet_index": quartet_index,
                    "quartet_id": quartet_id,
                    "background_index": background_index,
                    "background_id": background_id,
                    "background_mask_sha256": background_hash,
                    "background_cardinality": len(background_components),
                    "complement_background_index": int(
                        background["complement_background_index"]
                    ),
                    "pair_index_within_background": pair_index,
                    "pair": [first, second],
                    "edge_class": edge_class,
                    "hub_component": HUB_COMPONENT,
                    "hub_state": hub_state,
                    "forced_off_background_components": list(base),
                    "forced_off_background_cardinality": len(base),
                    "k_nonhub_on": sum(
                        component != HUB_COMPONENT for component in base
                    ),
                    "ordered_cell_bits": list(ORDERED_CELL_BITS),
                    "condition_ids": identifiers,
                    "interaction_formula": "R11-R10-R01+R00",
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "response_blind_exploratory_manifest",
        "claim_scope": "development_only_not_confirmatory",
        "component_count": COMPONENT_COUNT,
        "hub_component": HUB_COMPONENT,
        "background_count": len(backgrounds),
        "pair_count_per_background": len(pair_order),
        "quartet_count": len(quartets),
        "logical_cell_reference_count": 4 * len(quartets),
        "unique_condition_count": len(conditions),
        "ordered_cell_bits": list(ORDERED_CELL_BITS),
        "background_manifest_sha256": EXPECTED_BACKGROUND_MANIFEST_SHA256,
        "backgrounds": list(backgrounds),
        "conditions": conditions,
        "quartets": quartets,
        "deduplication_rule": (
            "one condition per exact sorted 24-bit selected-component mask"
        ),
    }


def validate_exhaustive_pair_atlas(atlas: Mapping[str, Any]) -> dict[str, Any]:
    """Reject any mutation, reordering, omission, or duplicate condition."""
    if not isinstance(atlas, Mapping):
        raise TypeError("atlas must be a mapping")
    copied = deepcopy(dict(atlas))
    expected = build_exhaustive_pair_atlas()
    if copied != expected or canonical_sha256(copied) != canonical_sha256(expected):
        raise ValueError("the E31 exhaustive pair atlas changed")
    if (
        copied["quartet_count"] != BACKGROUND_COUNT * PAIR_COUNT
        or copied["logical_cell_reference_count"] != 8_832
        or copied["unique_condition_count"] != 2_408
    ):
        raise RuntimeError("the exact E31 atlas accounting changed")
    return copied


__all__ = [
    "BACKGROUND_COUNT",
    "COMPONENT_COUNT",
    "EXPECTED_BACKGROUND_MANIFEST_SHA256",
    "HUB_COMPONENT",
    "ORDERED_CELL_BITS",
    "PAIR_COUNT",
    "SCHEMA_VERSION",
    "build_exhaustive_pair_atlas",
    "canonical_sha256",
    "validate_exhaustive_pair_atlas",
]
