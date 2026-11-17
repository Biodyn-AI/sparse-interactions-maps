# ruff: noqa: EXE002
"""Pure response-blind protocol for the E31.2 mixed-resolution head atlas."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import cache
from itertools import combinations
from typing import Any

from experiments.e31_sparse_interactions.mixed_resolution_panel import (
    PROMPT_COUNTS,
    PROMPT_MANIFEST_SHA256,
    canonical_sha256,
)

SCHEMA_VERSION = "e31_2_mixed_resolution_sparse_head_atlas_v1"
DESIGN_DOMAIN = "e31.2|mixed-resolution|raw-head-atlas|3120001"
COMPONENT_COUNT = 24
LAYER_COUNT = 12
HEAD_COUNT = 12
HUB_COMPONENT = 1
K_NONHUB_ON = 10
COVER_COUNT = 4
CELL_BITS = ("00", "10", "01", "11")
RECTANGLE_HEAD_COUNT = 4
MATCHED_CONTROL_COUNT = 10_000
MATCHED_CONTROL_SEED = 3_120_001
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 3_120_002

PARENT_PAIRS = (
    {
        "parent_index": 0,
        "parent_pair_id": "l05a_l08a",
        "layer_a": 5,
        "layer_b": 8,
        "component_a": 10,
        "component_b": 16,
    },
    {
        "parent_index": 1,
        "parent_pair_id": "l08a_l09a",
        "layer_a": 8,
        "layer_b": 9,
        "component_a": 16,
        "component_b": 18,
    },
    {
        "parent_index": 2,
        "parent_pair_id": "l05a_l10a",
        "layer_a": 5,
        "layer_b": 10,
        "component_a": 10,
        "component_b": 20,
    },
)

PRIMARY_GATES = {
    "aggregate_confirmation_energy_coverage_minimum": 0.50,
    "aggregate_fraction_of_oracle_rectangle_coverage_minimum": 0.75,
    "aggregate_coverage_one_sided_95_lower_bound_minimum": 0.40,
    "per_parent_sign_replication_count_minimum": 12,
    "per_parent_sign_replication_denominator": 16,
    "per_parent_energy_spearman_minimum": 0.50,
    "matched_control_quantile_strictly_exceeded": 0.95,
    "matched_control_count": MATCHED_CONTROL_COUNT,
    "pooled_direct_composite_r2_minimum": 0.50,
    "pooled_direct_composite_pearson_minimum": 0.70,
    "pair_selected_mse_beats_singleton_parent_count_minimum": 2,
    "parent_count": 3,
    "all_gates_required": True,
}

ATLAS_MANIFEST_SHA256 = {
    "discovery": "a51d03933a8416d84743085607c89a9c03fefe74f8a7e797d8af79dfd48f8e9e",
    "confirmation": "90bb22faf19a908f780ecd5b4473e37cdfa000c15b77f87538a903ece53b215d",
}
BACKGROUND_MANIFEST_SHA256 = {
    "discovery": "098775b5bc66f1e6001e668504e0b37314cf994ccd056f658373aec53931e1da",
    "confirmation": "16daa281249e342fc9a357a91ee6a4b459335660c07526f9a262a0b431fa83bf",
}
PROTOCOL_SHA256 = "fe7b400f6ace57d1a265b73a66e413cf8e9f7a8dfd4a187d6864e8dce600c216"


def _validate_split(split: str) -> None:
    if split not in PROMPT_COUNTS:
        raise ValueError("split must be 'discovery' or 'confirmation'")


def _digest(*parts: Any) -> str:
    value = "|".join(str(part) for part in (DESIGN_DOMAIN, *parts))
    return hashlib.sha256(value.encode("ascii")).hexdigest()


@cache
def _background_manifest_cached(split: str) -> tuple[dict[str, Any], ...]:
    _validate_split(split)
    records = []
    used_masks: set[tuple[int, ...]] = set()
    for parent in PARENT_PAIRS:
        eligible = tuple(
            component
            for component in range(COMPONENT_COUNT)
            if component
            not in {
                HUB_COMPONENT,
                parent["component_a"],
                parent["component_b"],
            }
        )
        if len(eligible) != 21:
            raise RuntimeError("the mixed-resolution background universe changed")
        for cover_rank in range(COVER_COUNT):
            salt = 0
            while True:
                ranked = sorted(
                    eligible,
                    key=lambda component: (
                        _digest(
                            "background",
                            split,
                            parent["parent_pair_id"],
                            cover_rank,
                            salt,
                            component,
                        ),
                        component,
                    ),
                )
                selected = tuple(sorted(ranked[:K_NONHUB_ON]))
                mask = tuple(
                    int(component in selected) for component in range(COMPONENT_COUNT)
                )
                if mask not in used_masks:
                    used_masks.add(mask)
                    break
                salt += 1
            records.append(
                {
                    "background_index": len(records),
                    "background_id": (
                        f"e31_2_{split}_{parent['parent_pair_id']}_cover_{cover_rank}"
                    ),
                    "split": split,
                    "parent_pair_id": parent["parent_pair_id"],
                    "parent_index": parent["parent_index"],
                    "cover_rank": cover_rank,
                    "k_nonhub_on": K_NONHUB_ON,
                    "hub_state": "off",
                    "selected_coarse_components": list(selected),
                    "coarse_background_mask": list(mask),
                    "background_sha256": canonical_sha256(list(mask)),
                    "selection_salt": salt,
                    "selection_rule": (
                        "rank eligible coarse components by lowercase SHA256 of "
                        "DESIGN_DOMAIN|background|split|parent_pair_id|cover_rank|"
                        "salt|component, tie component, and take the first 10; "
                        "increment salt only to avoid a duplicate physical background"
                    ),
                }
            )
    return tuple(records)


def build_background_manifest(split: str) -> tuple[dict[str, Any], ...]:
    """Return twelve response-blind coarse backgrounds for one split."""
    value = deepcopy(_background_manifest_cached(split))
    pinned = BACKGROUND_MANIFEST_SHA256[split]
    if pinned != "TO_BE_PINNED" and canonical_sha256(value) != pinned:
        raise RuntimeError(f"the frozen E31.2 {split} backgrounds changed")
    return value


def _empty_head_mask() -> list[list[int]]:
    return [[0] * HEAD_COUNT for _ in range(LAYER_COUNT)]


def _state(
    background: Mapping[str, Any],
    parent: Mapping[str, Any],
    heads_a: Sequence[int] = (),
    heads_b: Sequence[int] = (),
) -> dict[str, Any]:
    head_mask = _empty_head_mask()
    for head in heads_a:
        head_mask[parent["layer_a"]][head] = 1
    for head in heads_b:
        head_mask[parent["layer_b"]][head] = 1
    return {
        "coarse_component_mask": list(background["coarse_background_mask"]),
        "attention_head_mask": head_mask,
    }


def _state_key(state: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "coarse_component_mask": state["coarse_component_mask"],
            "attention_head_mask": state["attention_head_mask"],
        }
    )


def _selected_heads(head_mask: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
    return [
        {
            "layer": layer,
            "heads": [head for head, value in enumerate(row) if value],
        }
        for layer, row in enumerate(head_mask)
        if any(row)
    ]


@cache
def _atlas_manifest_cached(split: str) -> dict[str, Any]:
    _validate_split(split)
    conditions: list[dict[str, Any]] = []
    condition_index: dict[str, int] = {}

    def add_condition(state: Mapping[str, Any], reference: str) -> str:
        key = _state_key(state)
        index = condition_index.get(key)
        if index is None:
            index = len(conditions)
            condition_index[key] = index
            condition_id = f"e31_2_{split}_condition_{index:04d}_{key[:12]}"
            coarse = list(state["coarse_component_mask"])
            heads = deepcopy(state["attention_head_mask"])
            conditions.append(
                {
                    "condition_index": index,
                    "condition_id": condition_id,
                    "condition_sha256": key,
                    "coarse_component_mask": coarse,
                    "selected_coarse_components": [
                        component for component, value in enumerate(coarse) if value
                    ],
                    "attention_head_mask": heads,
                    "selected_attention_heads": _selected_heads(heads),
                    "references": [reference],
                }
            )
        else:
            conditions[index]["references"].append(reference)
        return str(conditions[index]["condition_id"])

    fine_contrasts = []
    coarse_contrasts = []
    for background in build_background_manifest(split):
        parent = PARENT_PAIRS[background["parent_index"]]
        base_state = _state(background, parent)
        singleton_a_states = [
            _state(background, parent, (head,), ()) for head in range(HEAD_COUNT)
        ]
        singleton_b_states = [
            _state(background, parent, (), (head,)) for head in range(HEAD_COUNT)
        ]
        for head_a in range(HEAD_COUNT):
            for head_b in range(HEAD_COUNT):
                contrast_index = len(fine_contrasts)
                contrast_id = (
                    f"e31_2_{split}_{parent['parent_pair_id']}_cover_"
                    f"{background['cover_rank']}_h{head_a:02d}_h{head_b:02d}"
                )
                states = (
                    base_state,
                    singleton_a_states[head_a],
                    singleton_b_states[head_b],
                    _state(background, parent, (head_a,), (head_b,)),
                )
                condition_ids = [
                    add_condition(state, f"fine:{contrast_id}:{cell}")
                    for state, cell in zip(states, CELL_BITS, strict=True)
                ]
                fine_contrasts.append(
                    {
                        "contrast_index": contrast_index,
                        "contrast_id": contrast_id,
                        "parent_pair_id": parent["parent_pair_id"],
                        "parent_index": parent["parent_index"],
                        "layer_a": parent["layer_a"],
                        "layer_b": parent["layer_b"],
                        "component_a": parent["component_a"],
                        "component_b": parent["component_b"],
                        "head_a": head_a,
                        "head_b": head_b,
                        "pair_index": HEAD_COUNT * head_a + head_b,
                        "cover_rank": background["cover_rank"],
                        "background_id": background["background_id"],
                        "background_sha256": background["background_sha256"],
                        "k_nonhub_on": K_NONHUB_ON,
                        "hub_state": "off",
                        "ordered_cell_bits": list(CELL_BITS),
                        "condition_ids": condition_ids,
                    }
                )
        coarse_id = (
            f"e31_2_{split}_{parent['parent_pair_id']}_cover_"
            f"{background['cover_rank']}_all_heads"
        )
        coarse_states = (
            base_state,
            _state(background, parent, range(HEAD_COUNT), ()),
            _state(background, parent, (), range(HEAD_COUNT)),
            _state(
                background,
                parent,
                range(HEAD_COUNT),
                range(HEAD_COUNT),
            ),
        )
        coarse_contrasts.append(
            {
                "contrast_index": len(coarse_contrasts),
                "contrast_id": coarse_id,
                "parent_pair_id": parent["parent_pair_id"],
                "parent_index": parent["parent_index"],
                "cover_rank": background["cover_rank"],
                "background_id": background["background_id"],
                "background_sha256": background["background_sha256"],
                "factor_definition": "direct_all_12_heads_per_parent_layer",
                "ordered_cell_bits": list(CELL_BITS),
                "condition_ids": [
                    add_condition(state, f"coarse:{coarse_id}:{cell}")
                    for state, cell in zip(coarse_states, CELL_BITS, strict=True)
                ],
            }
        )
    manifest = {
        "schema_version": "e31_2_mixed_resolution_atlas_manifest_v1",
        "split": split,
        "parent_pair_order": [parent["parent_pair_id"] for parent in PARENT_PAIRS],
        "head_order": list(range(HEAD_COUNT)),
        "cover_count": COVER_COUNT,
        "ordered_cell_bits": list(CELL_BITS),
        "background_manifest_sha256": canonical_sha256(
            build_background_manifest(split)
        ),
        "conditions": conditions,
        "fine_contrasts": fine_contrasts,
        "coarse_contrasts": coarse_contrasts,
        "unique_condition_count": len(conditions),
        "fine_contrast_count": len(fine_contrasts),
        "coarse_contrast_count": len(coarse_contrasts),
        "logical_fine_cell_reference_count": 4 * len(fine_contrasts),
        "logical_coarse_cell_reference_count": 4 * len(coarse_contrasts),
        "deduplication": (
            "exact global coarse_component_mask plus attention_head_mask; every "
            "physical condition retains all logical references"
        ),
    }
    if len(conditions) != 2_064:
        raise RuntimeError("the E31.2 atlas no longer has 2,064 physical masks")
    return manifest


def build_atlas_manifest(split: str) -> dict[str, Any]:
    """Build the exact 3-parent, 4-cover fine and all-head atlas."""
    value = deepcopy(_atlas_manifest_cached(split))
    pinned = ATLAS_MANIFEST_SHA256[split]
    if pinned != "TO_BE_PINNED" and canonical_sha256(value) != pinned:
        raise RuntimeError(f"the frozen E31.2 {split} atlas changed")
    return value


def validate_atlas_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise TypeError("atlas manifest must be a mapping")
    copied = deepcopy(dict(manifest))
    split = copied.get("split")
    _validate_split(split)
    if copied != build_atlas_manifest(split):
        raise ValueError("the frozen E31.2 atlas manifest changed")
    return copied


def _validated_head_support(value: Any, *, label: str) -> dict[str, list[int]]:
    if not isinstance(value, Mapping) or set(value) != {"heads_a", "heads_b"}:
        raise ValueError(f"{label} must contain heads_a and heads_b")
    copied = {side: list(value[side]) for side in ("heads_a", "heads_b")}
    for side, heads in copied.items():
        if (
            len(heads) != RECTANGLE_HEAD_COUNT
            or heads != sorted(set(heads))
            or any(
                isinstance(head, bool)
                or not isinstance(head, int)
                or head < 0
                or head >= HEAD_COUNT
                for head in heads
            )
        ):
            raise ValueError(f"{label} {side} requires four sorted unique heads")
    return copied


def validate_reconstruction_selections(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact D-frozen pair and singleton 4x4 rectangles."""
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "parents"}:
        raise ValueError("reconstruction selections changed keys")
    if value["schema_version"] != "e31_2_discovery_rectangle_selections_v1":
        raise ValueError("reconstruction selection schema changed")
    rows = value["parents"]
    if not isinstance(rows, Sequence) or len(rows) != len(PARENT_PAIRS):
        raise ValueError("one selection row is required per parent")
    copied_rows = []
    for expected, row in zip(PARENT_PAIRS, rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != {
            "parent_pair_id",
            "pair_selected",
            "singleton_selected",
        }:
            raise ValueError("a reconstruction selection row changed keys")
        if row["parent_pair_id"] != expected["parent_pair_id"]:
            raise ValueError("reconstruction parent order changed")
        copied_rows.append(
            {
                "parent_pair_id": expected["parent_pair_id"],
                "pair_selected": _validated_head_support(
                    row["pair_selected"], label="pair_selected"
                ),
                "singleton_selected": _validated_head_support(
                    row["singleton_selected"], label="singleton_selected"
                ),
            }
        )
    return {
        "schema_version": "e31_2_discovery_rectangle_selections_v1",
        "parents": copied_rows,
    }


def build_reconstruction_manifest(selections: Mapping[str, Any]) -> dict[str, Any]:
    """Build C-only direct composite quartets after D selections are sealed."""
    frozen = validate_reconstruction_selections(selections)
    atlas = build_atlas_manifest("confirmation")
    atlas_by_key = {row["condition_sha256"]: row for row in atlas["conditions"]}
    new_conditions: list[dict[str, Any]] = []
    new_by_key: dict[str, int] = {}

    def condition_id(state: Mapping[str, Any], reference: str) -> str:
        key = _state_key(state)
        if key in atlas_by_key:
            atlas_by_key[key]["references"].append(reference)
            return str(atlas_by_key[key]["condition_id"])
        index = new_by_key.get(key)
        if index is None:
            index = len(new_conditions)
            new_by_key[key] = index
            coarse = list(state["coarse_component_mask"])
            heads = deepcopy(state["attention_head_mask"])
            new_conditions.append(
                {
                    "condition_index": index,
                    "condition_id": (
                        "e31_2_confirmation_reconstruction_condition_"
                        f"{index:03d}_{key[:12]}"
                    ),
                    "condition_sha256": key,
                    "coarse_component_mask": coarse,
                    "selected_coarse_components": [
                        component for component, bit in enumerate(coarse) if bit
                    ],
                    "attention_head_mask": heads,
                    "selected_attention_heads": _selected_heads(heads),
                    "references": [reference],
                }
            )
        else:
            new_conditions[index]["references"].append(reference)
        return str(new_conditions[index]["condition_id"])

    selections_by_parent = {row["parent_pair_id"]: row for row in frozen["parents"]}
    coarse_by_background = {
        row["background_id"]: row for row in atlas["coarse_contrasts"]
    }
    contrasts = []
    for background in build_background_manifest("confirmation"):
        parent = PARENT_PAIRS[background["parent_index"]]
        row = selections_by_parent[parent["parent_pair_id"]]
        for support_name in ("pair_selected", "singleton_selected"):
            support = row[support_name]
            contrast_id = (
                f"e31_2_confirmation_{parent['parent_pair_id']}_cover_"
                f"{background['cover_rank']}_{support_name}_composite"
            )
            states = (
                _state(background, parent),
                _state(background, parent, support["heads_a"], ()),
                _state(background, parent, (), support["heads_b"]),
                _state(
                    background,
                    parent,
                    support["heads_a"],
                    support["heads_b"],
                ),
            )
            contrasts.append(
                {
                    "contrast_index": len(contrasts),
                    "contrast_id": contrast_id,
                    "parent_pair_id": parent["parent_pair_id"],
                    "parent_index": parent["parent_index"],
                    "cover_rank": background["cover_rank"],
                    "background_id": background["background_id"],
                    "support_name": support_name,
                    "heads_a": list(support["heads_a"]),
                    "heads_b": list(support["heads_b"]),
                    "ordered_cell_bits": list(CELL_BITS),
                    "condition_ids": [
                        condition_id(state, f"reconstruction:{contrast_id}:{cell}")
                        for state, cell in zip(states, CELL_BITS, strict=True)
                    ],
                    "direct_all_12_condition_ids": list(
                        coarse_by_background[background["background_id"]][
                            "condition_ids"
                        ]
                    ),
                    "comparison_rule": (
                        "compare direct group-composite second difference to direct "
                        "all-12-plus-all-12 second difference without summing fine "
                        "coefficients or affine calibration"
                    ),
                }
            )
    return {
        "schema_version": "e31_2_confirmation_reconstruction_manifest_v1",
        "split": "confirmation",
        "selection_sha256": canonical_sha256(frozen),
        "support_order": ["pair_selected", "singleton_selected"],
        "selections": frozen,
        "new_conditions": new_conditions,
        "contrasts": contrasts,
        "maximum_new_condition_count": 72,
        "actual_new_condition_count": len(new_conditions),
        "all_12_reference_source": "frozen_confirmation_atlas_coarse_contrasts",
    }


_HEAD_RECTANGLES = tuple(combinations(range(HEAD_COUNT), RECTANGLE_HEAD_COUNT))


def build_joint_matched_rectangle_controls(
    selections: Mapping[str, Any],
    *,
    count: int = MATCHED_CONTROL_COUNT,
) -> dict[str, Any]:
    """Build deterministic joint cardinality-matched controls without outcomes."""
    frozen = validate_reconstruction_selections(selections)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("control count must be a positive integer")
    excluded = {
        tuple(
            (
                tuple(row[support]["heads_a"]),
                tuple(row[support]["heads_b"]),
            )
            for row in frozen["parents"]
        )
        for support in ("pair_selected", "singleton_selected")
    }
    supports = []
    seen = set()
    salt = 0
    while len(supports) < count:
        joint = []
        for parent in PARENT_PAIRS:
            chosen = []
            for side in ("a", "b"):
                index = int(
                    _digest(
                        "matched-control",
                        MATCHED_CONTROL_SEED,
                        salt,
                        parent["parent_pair_id"],
                        side,
                    ),
                    16,
                ) % len(_HEAD_RECTANGLES)
                chosen.append(_HEAD_RECTANGLES[index])
            joint.append((chosen[0], chosen[1]))
        key = tuple(joint)
        if key not in seen and key not in excluded:
            seen.add(key)
            supports.append(
                [
                    {
                        "parent_pair_id": parent["parent_pair_id"],
                        "heads_a": list(rectangle[0]),
                        "heads_b": list(rectangle[1]),
                    }
                    for parent, rectangle in zip(PARENT_PAIRS, joint, strict=True)
                ]
            )
        salt += 1
    return {
        "schema_version": "e31_2_joint_matched_rectangle_controls_v1",
        "generator": "sha256_modulo_lexicographic_12_choose_4_per_side",
        "seed": MATCHED_CONTROL_SEED,
        "support_count": count,
        "candidate_salts_examined": salt,
        "selected_and_singleton_supports_excluded": True,
        "uses_model_outcomes": False,
        "supports": supports,
    }


def build_protocol() -> dict[str, Any]:
    """Build the complete response-blind E31.2 design and decision contract."""
    query_accounting = {
        "discovery_atlas_patch_cells": 2_064 * PROMPT_COUNTS["discovery"],
        "confirmation_atlas_patch_cells": 2_064 * PROMPT_COUNTS["confirmation"],
        "maximum_confirmation_reconstruction_patch_cells": 72
        * PROMPT_COUNTS["confirmation"],
        "clean_forward_examples": 2 * sum(PROMPT_COUNTS.values()),
        "batch_smoke_physical_context_mask_cells": 32,
        "maximum_registered_model_forwards": 101_504,
    }
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "question": (
            "Do three stable coarse GPT-2 attention interactions resolve into "
            "small, transferable, causally reconstructive 4x4 head rectangles?"
        ),
        "model": {
            "model_id": "gpt2",
            "component_order": "12 layers, heads 0 through 11",
            "coarse_background_components": (
                "24 layer-major attention/MLP residual updates"
            ),
            "fine_factor": (
                "donor head vector immediately before attention c_proj, all "
                "positions, with c_proj and all downstream computation recomputed"
            ),
            "whole_attention_equivalence_atol_rtol": 1e-9,
        },
        "prompt_counts": dict(PROMPT_COUNTS),
        "prompt_manifest_sha256": dict(PROMPT_MANIFEST_SHA256),
        "parent_pairs": deepcopy(PARENT_PAIRS),
        "parent_selection_ancestry": {
            "selection_timing": (
                "post_hoc_after_full_E31.1_V2_discovery_and_confirmation_score_inspection"
            ),
            "e31_1_v2_score_artifact_sha256": (
                "ddde7dcdef70525f8fdc3ccc75fe4c57852f87be805b4406a8c716624be57ba5"
            ),
            "selection_description": (
                "the three parents are the E31.1 primary k10 hub-off raw-"
                "discovery top three coarse pairs in their exact registered order"
            ),
            "discovery_only_selection_claimed": False,
            "e31_1_confirmation_had_already_been_inspected": True,
            "e31_2_outcomes_used": False,
            "prospective_scope": (
                "fresh E31.2 prompts, head-level outcomes, rectangle selection "
                "transfer, and direct group reconstruction only"
            ),
        },
        "background": {
            "k_nonhub_on": K_NONHUB_ON,
            "hub_state": "off",
            "covers_per_parent": COVER_COUNT,
            "manifest_sha256": dict(BACKGROUND_MANIFEST_SHA256),
            "factor_parent_coarse_components_forced_off": True,
        },
        "atlas": {
            "manifest_sha256": dict(ATLAS_MANIFEST_SHA256),
            "unique_conditions_per_split": 2_064,
            "fine_edges_per_parent": 144,
            "fine_edge_count": 432,
            "fine_contrast_count_per_split": 1_728,
            "direct_all_12_parent_contrast_count_per_split": 12,
            "ordered_cell_bits": list(CELL_BITS),
        },
        "response": {
            "coordinate": "unclipped endpoint-normalized donor-minus-base margin shift",
            "primary_interaction": "R11-R10-R01+R00",
            "primary_energy": (
                "mean square of each unaggregated signed raw interaction over "
                "registered contexts and covers"
            ),
            "saturation_residualization": False,
            "signed_effects_averaged_before_squaring": False,
        },
        "discovery_selection": {
            "pair_rectangle": (
                "for each parent enumerate all C(12,4)^2 rectangles; maximize "
                "the sum of its 16 discovery edge energies, tie lexicographic"
            ),
            "singleton_rectangle": (
                "select four heads per side by descending discovery singleton "
                "marginal energy (A=R10-R00, B=R01-R00), tie head index"
            ),
            "geometry_sealed_before_confirmation_reconstruction": True,
        },
        "matched_controls": {
            "count": MATCHED_CONTROL_COUNT,
            "seed": MATCHED_CONTROL_SEED,
            "unit": "joint tuple of one 4x4 rectangle for each of three parents",
            "model_queries": 0,
            "generator": (
                "lowercase SHA256 modulo lexicographic 12-choose-4 support list; "
                "deduplicate joint tuples and exclude pair/singleton selections"
            ),
        },
        "reconstruction": {
            "support_order": ["pair_selected", "singleton_selected"],
            "group_size_per_parent_side": RECTANGLE_HEAD_COUNT,
            "direct_group_quartet_required": True,
            "direct_all_12_quartet_required": True,
            "sum_of_pair_coefficients_allowed": False,
            "affine_calibration_allowed": False,
            "r2_and_pearson_pooling": (
                "pool all three parents, confirmation contexts, and covers"
            ),
            "baseline_comparison": (
                "pair-selected direct-composite MSE must be strictly smaller "
                "than singleton-selected direct-composite MSE in at least two "
                "of the three parents"
            ),
            "maximum_new_conditions_per_confirmation_context": 72,
        },
        "primary_gates": deepcopy(PRIMARY_GATES),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "resample_unit": "semantic_frame_with_both_directions",
            "stratified_by": "template_stratum",
            "common_resamples": True,
        },
        "multiplicity": {
            "primary_claim_count": 1,
            "aggregate_across_three_parents": True,
            "per_parent_sign_and_spearman_checks_are_one_all_parent_conjunction": True,
            "per_parent_outputs_otherwise_descriptive": True,
        },
        "query_accounting": query_accounting,
        "claim_boundary": (
            "E31.2 tests prospective raw head-pair localization and direct "
            "reconstruction for three post-E31.1-selected GPT-2-small IOI parent "
            "pairs; it does not establish task, model, or architecture universality."
        ),
    }
    return protocol


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(protocol, Mapping):
        raise TypeError("protocol must be a mapping")
    copied = deepcopy(dict(protocol))
    if copied != build_protocol():
        raise ValueError("the frozen E31.2 protocol changed")
    if (
        PROTOCOL_SHA256 != "TO_BE_PINNED"
        and canonical_sha256(copied) != PROTOCOL_SHA256
    ):
        raise RuntimeError("the frozen E31.2 protocol hash changed")
    return copied


__all__ = [
    "ATLAS_MANIFEST_SHA256",
    "BACKGROUND_MANIFEST_SHA256",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CELL_BITS",
    "COMPONENT_COUNT",
    "COVER_COUNT",
    "DESIGN_DOMAIN",
    "HEAD_COUNT",
    "K_NONHUB_ON",
    "LAYER_COUNT",
    "MATCHED_CONTROL_COUNT",
    "MATCHED_CONTROL_SEED",
    "PARENT_PAIRS",
    "PRIMARY_GATES",
    "PROTOCOL_SHA256",
    "RECTANGLE_HEAD_COUNT",
    "SCHEMA_VERSION",
    "build_atlas_manifest",
    "build_background_manifest",
    "build_joint_matched_rectangle_controls",
    "build_protocol",
    "build_reconstruction_manifest",
    "canonical_sha256",
    "validate_atlas_manifest",
    "validate_protocol",
    "validate_reconstruction_selections",
]
