# ruff: noqa: EXE002
"""Pure response-blind protocol for E29.15 mechanism discovery.

This module fixes prompts, discovery masks, method/nomination slots, and the
two-stage confirmation design.  It performs no file I/O, model call, fitting,
registration, or outcome-dependent prompt replacement.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch

SCHEMA_VERSION = "e29_15_ioi_mechanism_discovery_response_blind_v1"
MODEL_ID = "gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
COMPONENT_COUNT = 24
DISCOVERY_BUDGETS = (64, 128, 256, 512)
PROMPTS_PER_SPLIT = 32
MINIMUM_CLEAN_ENDPOINT = 0.25
MINIMUM_ENDPOINT_CORRECT_FRACTION = 0.75
DESIGN_DOMAIN = "e29.15|ioi|mechanism-discovery|2915001"
MAXIMUM_SUFFICIENT_SET_SIZE = 12
SUFFICIENCY_TARGET = 1.0
SUFFICIENCY_ABSOLUTE_TOLERANCE = 0.10
EFFECT_BACKGROUND_COUNT = 8

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

COMPONENT_MANIFEST = tuple(
    {
        "component_index": 2 * layer + kind_index,
        "layer": layer,
        "kind": kind,
        "component_id": f"L{layer:02d}.{kind}",
    }
    for layer in range(12)
    for kind_index, kind in enumerate(("attention", "mlp"))
)

METHOD_IDS = (
    "tt_rank2",
    "tt_rank8",
    "cp_rank4",
    "mlp_parameter_matched",
    "mlp_hidden32",
    "additive",
    "pairwise",
    "tt2_shapley_adapter",
    "mlp_conditional_adapter",
    "singleton_rank",
    "random_hash_control",
)

METHOD_RECIPES = {
    "tt_rank2": {
        "architecture": "tensor_train",
        "rank": 2,
        "selection_adapter": "direct_response_surface",
    },
    "tt_rank8": {
        "architecture": "tensor_train",
        "rank": 8,
        "selection_adapter": "direct_response_surface",
    },
    "cp_rank4": {
        "architecture": "canonical_polyadic",
        "rank": 4,
        "selection_adapter": "direct_response_surface",
    },
    "mlp_parameter_matched": {
        "architecture": "one_hidden_layer_mlp",
        "hidden_width": "nearest_parameter_count_to_tt_rank2_for_32_outputs",
        "selection_adapter": "direct_response_surface",
    },
    "mlp_hidden32": {
        "architecture": "one_hidden_layer_mlp",
        "hidden_width": 32,
        "selection_adapter": "direct_response_surface",
    },
    "additive": {
        "architecture": "ridge_main_effects",
        "selection_adapter": "direct_response_surface",
    },
    "pairwise": {
        "architecture": "ridge_main_plus_all_pairs",
        "selection_adapter": "direct_response_surface",
    },
    "tt2_shapley_adapter": {
        "architecture": "shared_tt_rank2_surrogate",
        "selection_adapter": (
            "deterministic_256_permutation_shapley_for_components;_"
            "other_nomination_families_inherit_tt_rank2"
        ),
        "independent_architecture": False,
        "official_tn_shap_claim_allowed": False,
    },
    "mlp_conditional_adapter": {
        "architecture": "shared_mlp_parameter_matched_surrogate",
        "selection_adapter": "deterministic_conditional_greedy",
        "independent_architecture": False,
        "official_coax_claim_allowed": False,
    },
    "singleton_rank": {
        "architecture": "empirical_singleton_anchor_ranking",
        "selection_adapter": (
            "observed_singleton_minus_zero_for_components;_pairs_and_group_"
            "are_composed_from_singleton_scores"
        ),
    },
    "random_hash_control": {
        "architecture": "none",
        "selection_adapter": "response_blind_sha256_order",
    },
}

NOMINATION_RULES = {
    "interaction_group6": {
        "count": 1,
        "component_count": 6,
        "estimand": "mean_squared_mobius_coefficients_of_orders_2_and_3",
        "search_space": "all_134596_six_component_groups",
        "tie_break": "lexicographically_smallest_component_tuple",
    },
    "component": {
        "count": 6,
        "component_count": 1,
        "estimand": (
            "square_across_contexts_after_mean_signed_on_minus_off_"
            "over_8_fixed_backgrounds"
        ),
        "tie_break": "score_descending_then_component_index_ascending",
    },
    "pair": {
        "count": 6,
        "component_count": 2,
        "estimand": (
            "square_across_contexts_after_mean_signed_second_difference_"
            "over_8_fixed_backgrounds"
        ),
        "tie_break": "score_descending_then_lexicographic_pair",
    },
    "sufficient_set": {
        "count": 1,
        "component_count": f"1_to_{MAXIMUM_SUFFICIENT_SET_SIZE}",
        "estimand": ("greedy_minimum_mean_absolute_error_to_clean_donor_endpoint_1"),
        "algorithm": (
            "start_empty; add the unused component minimizing mean absolute "
            "predicted response minus 1 across discovery contexts; break ties by "
            "component index; stop at first mean absolute error at most 0.10 or "
            "at size 12"
        ),
        "tie_break": "component_index_ascending",
    },
}
CANDIDATE_SLOTS_PER_METHOD_BUDGET = sum(
    int(rule["count"]) for rule in NOMINATION_RULES.values()
)

CLEAN_PANEL_GATE = {
    "per_context": {
        "all_values_finite": True,
        "minimum_clean_endpoint_inclusive": MINIMUM_CLEAN_ENDPOINT,
        "endpoint_correctness_is_not_a_row_filter": True,
    },
    "aggregate_per_split": {
        "minimum_base_correct_fraction_inclusive": MINIMUM_ENDPOINT_CORRECT_FRACTION,
        "minimum_donor_correct_fraction_inclusive": MINIMUM_ENDPOINT_CORRECT_FRACTION,
        "mean_base_correct_minus_donor_margin_strict_min": 0.0,
        "mean_donor_correct_minus_base_margin_strict_min": 0.0,
        "retain_all_32_rows": True,
    },
}

RESPONSE_COORDINATE = {
    "raw_contrast": "logit(donor_target)-logit(base_target)",
    "normalized": (
        "(patched_contrast-clean_base_contrast)/"
        "(clean_donor_contrast-clean_base_contrast)"
    ),
    "clean_base_endpoint": 0.0,
    "clean_donor_endpoint": SUFFICIENCY_TARGET,
    "full_patch_is_separate_anchor_not_sufficiency_target": True,
}

SCORING_PLAN = {
    "primary_interaction_group_claim": {
        "tensor_method": "tt_rank2",
        "tensor_budget": 256,
        "baseline_method": "mlp_parameter_matched",
        "baseline_budget": 256,
        "endpoint": (
            "heldout_per_context_mean_squared_mobius_coefficients_of_orders_2_and_3_"
            "on_each_methods_nominated_six_component_lattice"
        ),
        "minimum_point_ratio": 1.25,
        "paired_mean_difference_one_sided_lower_bound_strict_minimum": 0.0,
        "pass_rule": "both_minimum_point_ratio_and_positive_lower_bound",
        "bootstrap": {
            "replicates": 10_000,
            "seed": 2_915_001,
            "resample_unit": "confirmation_prompt_context",
            "stratified_by": "confirmation_stratum_id",
            "common_resamples": True,
            "one_sided_confidence_level": 0.95,
        },
    },
    "query_efficiency_claim": {
        "tensor_method": "tt_rank2",
        "tensor_budget": 128,
        "baseline_method": "mlp_parameter_matched",
        "baseline_budget": 256,
        "endpoint": "same_as_primary_interaction_group_claim",
        "minimum_point_ratio": 1.25,
        "paired_mean_difference_one_sided_lower_bound_strict_minimum": 0.0,
        "evaluated_only_if_primary_passes": True,
    },
    "complete_secondary_outputs": (
        "all_method_by_budget_interaction_group_component_pair_sufficiency_"
        "minimality_and_complement_necessity_scores"
    ),
    "secondary_inference_multiplicity": (
        "Holm_adjust_within_each_endpoint_family;_otherwise_label_descriptive"
    ),
    "broader_mechanism_endpoints": "descriptive_only_no_supported_boolean",
    "claim_boundary": (
        "The confirmatory claim is parameter-efficient TT2 interaction-group "
        "discovery versus the parameter-matched MLP, not superiority to MLPs "
        "generally, complete mechanism discovery, or official superiority to "
        "TN-SHAP or CoAx."
    ),
}

DISCOVERY_FIT_CONFIG = {
    "budgets": list(DISCOVERY_BUDGETS),
    "primary_budget": 256,
    "top_component_count": 6,
    "top_pair_count": 6,
    "group_size": 6,
    "methods": list(METHOD_IDS),
    "split_seed": 29_150,
    "training_seeds": [29_151, 29_152, 29_153],
    "validation_fraction": 0.20,
    "minimum_validation_rows": 8,
    "ridge_values": [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0],
    "learning_rate": 0.01,
    "weight_decay": 1e-4,
    "maximum_epochs": 1_200,
    "evaluation_interval": 20,
    "patience_epochs": 240,
    "gradient_clip_norm": 5.0,
    "scoring_batch_size": 8_192,
    "maximum_group_candidates": 200_000,
    "shapley_permutations": 256,
    "greedy_sufficiency_absolute_tolerance": SUFFICIENCY_ABSOLUTE_TOLERANCE,
    "greedy_set_cap": MAXIMUM_SUFFICIENT_SET_SIZE,
    "conditional_primary_count": 2,
}

_TEMPLATES = {
    "discovery": (
        (
            "discovery_when",
            "When {a} and {b} went to the {place}, {repeated} gave a {object} to",
        ),
        (
            "discovery_after",
            "After {a} and {b} arrived at the {place}, {repeated} handed a {object} to",
        ),
        (
            "discovery_while",
            "While {a} and {b} were in the {place}, {repeated} passed a {object} to",
        ),
        (
            "discovery_then",
            "Then {a} and {b} visited the {place}; {repeated} sent a {object} to",
        ),
    ),
    "confirmation": (
        (
            "confirmation_before",
            "Before leaving the {place}, {a} and {b} talked; {repeated} offered a {object} to",
        ),
        (
            "confirmation_because",
            "Because {a} and {b} met at the {place}, {repeated} brought a {object} to",
        ),
        (
            "confirmation_during",
            "During a visit to the {place}, {a} and {b} waited; {repeated} showed a {object} to",
        ),
        (
            "confirmation_yesterday",
            "Yesterday {a} and {b} reached the {place}; afterward, {repeated} delivered a {object} to",
        ),
    ),
}


def _frozen_words(value: str) -> tuple[str, ...]:
    """Keep long response-blind vocabularies compact and exactly ordered."""
    return tuple(value.split())


_NAMES = _frozen_words(
    "Alice Bob Carol David Emma Frank Grace Henry Jack Karen Louis Mary Nancy Oscar "
    "Peter Rachel Samuel Thomas Victoria William Andrew Barbara Charles Daniel Edward "
    "Fiona George Hannah Isaac Julia Kevin Laura Michael Natalie Oliver Patricia Robert "
    "Sarah Steven Teresa Arthur Betty Colin Diane Eric Florence Gary Helen Ian Jane Keith "
    "Linda Martin Nora Paul Rebecca Simon Susan Trevor Wendy Adam Brenda Carl Denise Ethan "
    "Frances Glenn Heather Jason Kelly Mark Nicole Patrick Rose Scott Tina Albert Bonnie "
    "Curtis Donna Eugene Georgia Harold Janet Kenneth Lisa Matthew Olivia Philip Rita "
    "Stephen Virginia Aaron Beth Calvin Edwin Gloria Howard Joyce Leonard Monica Nathan "
    "Pamela Raymond Sylvia Timothy April Bruce Cathy Derek Ellen Fred Harry Joan Kyle Lucy "
    "Megan Neil Paula Roger Sally Tony Vera Abel Amy Brian Claire Dylan Elaine Felix Gina "
    "Hugo Jennifer Luke Melissa Noah Ryan Sophia Tyler Zoe Adrian Amanda Blake Brooke "
    "Cameron Damien Deborah Elliot Emily Evan Gavin Holly Jacob Leah Logan Marcus Naomi Owen "
    "Paige Quentin Robin Sebastian Stella Travis Vanessa Warren"
)

_PLACES = _frozen_words(
    "market station garden office school museum theater library hospital hotel airport "
    "restaurant stadium courthouse workshop harbor village palace factory gallery bakery "
    "pharmacy campus warehouse temple clinic bookstore park kitchen studio hall bridge"
)
_OBJECTS = _frozen_words(
    "book drink gift letter package flower ticket photo bottle snack note map card key toy "
    "tool scarf cup pen box apple parcel candle medal poster folder painting badge coin "
    "ribbon watch basket"
)


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible protocol data with one canonical encoding."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_component_manifest() -> tuple[dict[str, Any], ...]:
    return deepcopy(COMPONENT_MANIFEST)


def build_prompt_manifest(split: str) -> tuple[dict[str, str], ...]:
    """Return a frozen 32-pair IOI panel; no outcome-based replacement exists."""
    if split not in _TEMPLATES:
        raise ValueError("split must be 'discovery' or 'confirmation'")
    if len(_NAMES) < 128:
        raise RuntimeError("the frozen name pool is too small")
    name_offset = 0 if split == "discovery" else 64
    records = []
    for stratum_index, (template_id, template) in enumerate(_TEMPLATES[split]):
        for within_stratum in range(8):
            pair_index = 8 * stratum_index + within_stratum
            name_a = _NAMES[name_offset + 2 * pair_index]
            name_b = _NAMES[name_offset + 2 * pair_index + 1]
            place = _PLACES[pair_index]
            object_name = _OBJECTS[pair_index]
            values = {
                "a": name_a,
                "b": name_b,
                "place": place,
                "object": object_name,
            }
            record = {
                "pair_id": f"{split}_pair_{pair_index:02d}",
                "split": split,
                "stratum_id": f"{split}_stratum_{stratum_index}",
                "template_id": template_id,
                "template": template,
                "name_a": name_a,
                "name_b": name_b,
                "place": place,
                "object": object_name,
                "base_prompt": template.format(**values, repeated=name_b),
                "donor_prompt": template.format(**values, repeated=name_a),
                "base_target": name_a,
                "donor_target": name_b,
                "base_repeated_subject": name_b,
                "donor_repeated_subject": name_a,
            }
            records.append(record)
    return tuple(records)


def validate_prompt_manifest(
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> tuple[dict[str, Any], ...]:
    """Validate the exact panel and tokenizer alignment without model responses."""
    copied = tuple(dict(record) for record in records)
    if copied != build_prompt_manifest(split):
        raise ValueError("prompt manifest changed; prompt replacement is forbidden")
    checks = []
    for record in copied:
        base_ids = tokenizer.encode(record["base_prompt"], add_special_tokens=False)
        donor_ids = tokenizer.encode(record["donor_prompt"], add_special_tokens=False)
        base_target_ids = tokenizer.encode(
            " " + record["base_target"], add_special_tokens=False
        )
        donor_target_ids = tokenizer.encode(
            " " + record["donor_target"], add_special_tokens=False
        )
        if len(base_ids) != len(donor_ids):
            raise ValueError(f"{record['pair_id']} prompt lengths differ")
        differing = [
            index
            for index, (base_id, donor_id) in enumerate(
                zip(base_ids, donor_ids, strict=True)
            )
            if base_id != donor_id
        ]
        if len(differing) != 1:
            raise ValueError(f"{record['pair_id']} must differ at exactly one token")
        if len(base_target_ids) != 1 or len(donor_target_ids) != 1:
            raise ValueError(
                f"{record['pair_id']} target is not one leading-space token"
            )
        if base_target_ids == donor_target_ids:
            raise ValueError(f"{record['pair_id']} target IDs coincide")
        checks.append(
            {
                **deepcopy(record),
                "token_count": len(base_ids),
                "differing_prompt_position": differing[0],
                "base_target_token_id": int(base_target_ids[0]),
                "donor_target_token_id": int(donor_target_ids[0]),
            }
        )
    return tuple(checks)


def _hash_selected_components(
    label: str, cardinality: int, salt: int = 0
) -> tuple[int, ...]:
    ranked = sorted(
        range(COMPONENT_COUNT),
        key=lambda component: hashlib.sha256(
            f"{DESIGN_DOMAIN}|{label}|{salt}|{component}".encode("ascii")
        ).digest(),
    )
    return tuple(sorted(ranked[:cardinality]))


def _hash_selected_from_candidates(
    label: str, candidates: Sequence[int], cardinality: int
) -> tuple[int, ...]:
    if cardinality > len(candidates):
        raise ValueError("cardinality exceeds the frozen candidate set")
    ranked = sorted(
        (int(component) for component in candidates),
        key=lambda component: hashlib.sha256(
            f"{DESIGN_DOMAIN}|{label}|{component}".encode("ascii")
        ).digest(),
    )
    return tuple(sorted(ranked[:cardinality]))


def _build_discovery_rows() -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = [(), tuple(range(COMPONENT_COUNT))]
    for component in range(COMPONENT_COUNT):
        rows.append((component,))
        rows.append(
            tuple(index for index in range(COMPONENT_COUNT) if index != component)
        )
    seen = set(rows)
    pair_index = 0
    while len(rows) < 512:
        cardinality = 2 + pair_index % 11
        salt = 0
        while True:
            row = _hash_selected_components(
                f"discovery_pair_{pair_index}", cardinality, salt
            )
            complement = tuple(
                index for index in range(COMPONENT_COUNT) if index not in row
            )
            if row not in seen and complement not in seen and row != complement:
                break
            salt += 1
        rows.extend((row, complement))
        seen.update((row, complement))
        pair_index += 1
    return tuple(rows)


def build_discovery_masks() -> torch.Tensor:
    rows = _build_discovery_rows()
    masks = torch.zeros((len(rows), COMPONENT_COUNT), dtype=torch.float64)
    for row_index, selected in enumerate(rows):
        if selected:
            masks[row_index, list(selected)] = 1.0
    return masks


def build_discovery_mask_manifest() -> tuple[dict[str, Any], ...]:
    rows = _build_discovery_rows()
    return tuple(
        {
            "mask_index": index,
            "mask_id": f"discovery_mask_{index:03d}",
            "selected_components": list(selected),
            "cardinality": len(selected),
            "complement_mask_index": index ^ 1,
            "design_family": (
                "endpoint_anchor"
                if index < 2
                else "singleton_leave_one_out_anchor"
                if index < 50
                else "response_blind_hash_complement_pair"
            ),
        }
        for index, selected in enumerate(rows)
    )


def validate_discovery_masks(masks: torch.Tensor) -> torch.Tensor:
    expected = build_discovery_masks()
    if not isinstance(masks, torch.Tensor) or not torch.equal(
        masks.cpu().double(), expected
    ):
        raise ValueError(
            "discovery masks changed from the frozen response-blind design"
        )
    return expected


def build_effect_background_masks() -> torch.Tensor:
    """Return four deterministic multiscale masks and their complements."""
    rows: list[tuple[int, ...]] = []
    for pair_index, cardinality in enumerate((3, 6, 9, 12)):
        selected = _hash_selected_components(
            f"effect_background_{pair_index}", cardinality
        )
        complement = tuple(
            index for index in range(COMPONENT_COUNT) if index not in selected
        )
        rows.extend((selected, complement))
    masks = torch.zeros((EFFECT_BACKGROUND_COUNT, COMPONENT_COUNT), dtype=torch.float64)
    for row_index, selected in enumerate(rows):
        masks[row_index, list(selected)] = 1.0
    return masks


def build_effect_background_manifest() -> tuple[dict[str, Any], ...]:
    masks = build_effect_background_masks()
    return tuple(
        {
            "background_index": index,
            "background_id": f"effect_background_{index}",
            "selected_components": torch.where(row == 1.0)[0].tolist(),
            "cardinality": int(row.sum()),
            "complement_background_index": index ^ 1,
            "response_blind": True,
        }
        for index, row in enumerate(masks)
    )


def _candidate_selection_rule(method_id: str, kind: str) -> dict[str, Any]:
    """Return the exact frozen selector used for one method/endpoint family."""
    direct = {
        "interaction_group6": {
            "implementation": "exhaustive_direct_surrogate_group_search",
            "selection_estimand": (
                "predicted_mean_squared_mobius_coefficients_of_orders_2_and_3"
            ),
            "search_space": "all_134596_six_component_groups",
            "tie_break": "lexicographically_smallest_component_tuple",
        },
        "component": {
            "implementation": "direct_surrogate_fixed_background_component_search",
            "selection_estimand": (
                "square_across_contexts_after_mean_signed_first_difference_"
                "over_8_fixed_backgrounds"
            ),
            "search_space": "all_24_components",
            "tie_break": "score_descending_then_component_index_ascending",
        },
        "pair": {
            "implementation": "direct_surrogate_fixed_background_pair_search",
            "selection_estimand": (
                "square_across_contexts_after_mean_signed_second_difference_"
                "over_8_fixed_backgrounds"
            ),
            "search_space": "all_276_component_pairs",
            "tie_break": "score_descending_then_lexicographic_pair",
        },
        "sufficient_set": {
            "implementation": "standard_endpoint_one_mae_greedy",
            "selection_estimand": (
                "predicted_mean_absolute_error_to_clean_donor_endpoint_1"
            ),
            "search_space": "greedy_add_one_up_to_12_components",
            "tie_break": "component_index_ascending",
            "absolute_tolerance": SUFFICIENCY_ABSOLUTE_TOLERANCE,
        },
    }[kind]
    if method_id == "tt2_shapley_adapter" and kind == "component":
        return {
            "implementation": "deterministic_256_permutation_shapley_over_shared_tt2",
            "selection_estimand": "mean_squared_shapley_value_across_contexts",
            "search_space": "all_24_components",
            "tie_break": "score_descending_then_component_index_ascending",
            "permutation_count": 256,
            "official_tn_shap_claim_allowed": False,
        }
    if method_id == "mlp_conditional_adapter":
        conditional = {
            "interaction_group6": {
                "implementation": "two_primary_plus_four_conditional_compensators",
                "selection_estimand": (
                    "top_2_fixed_background_component_scores_plus_top_4_"
                    "conditional_growth_scores"
                ),
                "search_space": "all_24_components_under_fixed_seed_rule",
                "tie_break": "score_descending_then_component_index_ascending",
            },
            "component": {
                "implementation": "two_primary_then_four_conditional_compensators",
                "selection_estimand": (
                    "primary_fixed_background_score_or_conditional_growth_score"
                ),
                "search_space": "all_24_components_under_fixed_seed_rule",
                "tie_break": "primaries_first_then_score_then_component_index",
            },
            "pair": {
                "implementation": "primary_to_conditional_compensator_pairs",
                "selection_estimand": "conditional_compensator_growth_score",
                "search_space": "two_primaries_cross_four_compensators",
                "tie_break": "score_then_primary_then_compensator",
            },
            "sufficient_set": {
                **direct,
                "implementation": "standard_endpoint_one_mae_greedy_over_shared_mlp",
            },
        }
        return {
            **conditional[kind],
            "official_coax_claim_allowed": False,
        }
    if method_id == "singleton_rank":
        singleton = {
            "interaction_group6": {
                "implementation": "six_highest_observed_singleton_mean_square_scores",
                "selection_estimand": "sum_of_component_singleton_mean_square_scores",
                "search_space": "all_24_components_choose_top_6",
                "tie_break": "score_descending_then_component_index_ascending",
            },
            "component": {
                "implementation": "observed_singleton_minus_zero_mean_square_score",
                "selection_estimand": (
                    "observed_singleton_minus_zero_squared_then_mean_across_contexts"
                ),
                "search_space": "all_24_observed_singletons",
                "tie_break": "score_descending_then_component_index_ascending",
            },
            "pair": {
                "implementation": "sum_of_two_observed_singleton_mean_square_scores",
                "selection_estimand": "sum_of_the_two_component_singleton_scores",
                "search_space": "all_276_component_pairs",
                "tie_break": "score_descending_then_lexicographic_pair",
            },
            "sufficient_set": {
                **direct,
                "implementation": "endpoint_one_mae_greedy_over_refit_additive_surrogate",
            },
        }
        return singleton[kind]
    if method_id == "random_hash_control":
        implementations = {
            "interaction_group6": "maximum_response_blind_sha256_group_score",
            "component": "response_blind_sha256_component_order",
            "pair": "response_blind_sha256_pair_order",
            "sufficient_set": "first_twelve_components_in_response_blind_hash_order",
        }
        return {
            "implementation": implementations[kind],
            "selection_estimand": "response_blind_sha256_score",
            "search_space": {
                "interaction_group6": "all_134596_six_component_groups",
                "component": "all_24_components",
                "pair": "all_276_component_pairs",
                "sufficient_set": "all_24_components",
            }[kind],
            "tie_break": "lexicographic_index_order",
        }
    return direct


def _confirmation_estimand(kind: str) -> str:
    return {
        "interaction_group6": (
            "heldout_mean_squared_mobius_coefficients_of_orders_2_and_3"
        ),
        "component": (
            "heldout_square_across_contexts_after_mean_signed_first_difference_"
            "over_the_same_8_fixed_backgrounds"
        ),
        "pair": (
            "heldout_square_across_contexts_after_mean_signed_second_difference_"
            "over_the_same_8_fixed_backgrounds"
        ),
        "sufficient_set": (
            "heldout_mean_absolute_error_to_endpoint_1_with_descriptive_"
            "leave_one_out_and_complement_checks"
        ),
    }[kind]


def build_candidate_manifest() -> tuple[dict[str, Any], ...]:
    rows = []
    for method_id in METHOD_IDS:
        for budget in DISCOVERY_BUDGETS:
            for kind, rule in NOMINATION_RULES.items():
                for rank in range(1, int(rule["count"]) + 1):
                    rows.append(
                        {
                            "slot_index": len(rows),
                            "slot_id": f"{method_id}::b{budget}::{kind}::{rank}",
                            "method_id": method_id,
                            "budget": budget,
                            "discovery_mask_count": budget,
                            "discovery_mask_indices": list(range(budget)),
                            "nomination_kind": kind,
                            "rank": rank,
                            "confirmation_estimand": _confirmation_estimand(kind),
                            "selection_rule": _candidate_selection_rule(
                                method_id, kind
                            ),
                        }
                    )
    return tuple(rows)


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    return value


def _validate_nominations(
    nominations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    slots = build_candidate_manifest()
    if len(nominations) != len(slots):
        raise ValueError("nominations must fill the complete candidate manifest")
    copied = [deepcopy(dict(record)) for record in nominations]
    for slot, record in zip(slots, copied, strict=True):
        for field in ("slot_id", "method_id", "budget", "nomination_kind", "rank"):
            if record.get(field) != slot[field]:
                raise ValueError(
                    f"nomination field {field} does not match slot {slot['slot_id']}"
                )
        components = record.get("components")
        if not isinstance(components, list) or components != sorted(set(components)):
            raise ValueError("nomination components must be a sorted unique list")
        if any(
            not isinstance(component, int) or not 0 <= component < COMPONENT_COUNT
            for component in components
        ):
            raise ValueError("nomination contains an invalid component")
        kind = str(record["nomination_kind"])
        expected_size = {"interaction_group6": 6, "component": 1, "pair": 2}.get(kind)
        if expected_size is not None and len(components) != expected_size:
            raise ValueError(f"{kind} nomination has the wrong component count")
        if (
            kind == "sufficient_set"
            and not 1 <= len(components) <= MAXIMUM_SUFFICIENT_SET_SIZE
        ):
            raise ValueError(
                "sufficient-set nomination must contain 1 to 12 components"
            )
        score = record.get("predicted_score")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("predicted_score must be finite")
        record["predicted_score"] = float(score)
    return copied


def build_nomination_envelope(
    nominations: Sequence[Mapping[str, Any]],
    *,
    discovery_artifact_sha256: str,
) -> dict[str, Any]:
    """Build the immutable handoff required before confirmation is constructed."""
    artifact_hash = _validate_sha256(
        discovery_artifact_sha256, "discovery_artifact_sha256"
    )
    copied = _validate_nominations(nominations)
    payload = {
        "schema_version": "e29_15_nomination_envelope_v1",
        "discovery_artifact_sha256": artifact_hash,
        "candidate_manifest_sha256": canonical_sha256(build_candidate_manifest()),
        "nominations": copied,
    }
    return {**payload, "nomination_sha256": canonical_sha256(payload)}


def _validate_nomination_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    copied = deepcopy(dict(envelope))
    seal = copied.pop("nomination_sha256", None)
    _validate_sha256(seal, "nomination_sha256")
    if copied.get("schema_version") != "e29_15_nomination_envelope_v1":
        raise ValueError("nomination envelope schema changed")
    _validate_sha256(
        copied.get("discovery_artifact_sha256"), "discovery_artifact_sha256"
    )
    if copied.get("candidate_manifest_sha256") != canonical_sha256(
        build_candidate_manifest()
    ):
        raise ValueError("candidate manifest commitment changed")
    copied["nominations"] = _validate_nominations(copied.get("nominations", []))
    if seal != canonical_sha256(copied):
        raise ValueError("nomination envelope seal does not verify")
    return {**copied, "nomination_sha256": seal}


def build_fixed_controls() -> tuple[dict[str, Any], ...]:
    early_attention = _hash_selected_from_candidates(
        "canonical_attention_control", range(0, 18, 2), 3
    )
    attention_reference = tuple(sorted((*early_attention, 18, 20, 22)))
    return (
        {
            "control_id": "late_l9_l11_attn_mlp",
            "components": [18, 19, 20, 21, 22, 23],
            "provenance": "literature_derived_sublayer_reference",
            "gating": False,
        },
        {
            "control_id": "late_attention_plus_hash_attention",
            "components": list(attention_reference),
            "provenance": "literature_l9_l11_attention_plus_response_blind_hash",
            "gating": False,
        },
        {
            "control_id": "hash_cardinality6",
            "components": list(_hash_selected_components("fixed_control_group", 6)),
            "provenance": "response_blind_hash",
            "gating": False,
        },
    )


def build_confirmation_manifest(
    nomination_envelope: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construct exact held-out queries only after a valid nomination seal."""
    envelope = _validate_nomination_envelope(nomination_envelope)
    fixed_controls = tuple(
        dict(record)
        for record in (build_fixed_controls() if controls is None else controls)
    )
    if fixed_controls != build_fixed_controls():
        raise ValueError("confirmation controls differ from the frozen controls")

    condition_index_by_row: dict[tuple[int, ...], int] = {}
    conditions: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []

    def condition_id(components: Sequence[int]) -> str:
        selected = tuple(sorted({int(component) for component in components}))
        if selected not in condition_index_by_row:
            index = len(conditions)
            condition_index_by_row[selected] = index
            conditions.append(
                {
                    "condition_index": index,
                    "condition_id": f"confirmation_condition_{index:05d}",
                    "selected_components": list(selected),
                    "cardinality": len(selected),
                }
            )
        return conditions[condition_index_by_row[selected]]["condition_id"]

    def add_contrast(
        *,
        source_type: str,
        source_id: str,
        method_id: str,
        budget: int,
        role: str,
        cell_index: int,
        components: Sequence[int],
        background_index: int = -1,
        nominated_components: Sequence[int] = (),
    ) -> None:
        selected = tuple(sorted({int(component) for component in components}))
        nominated = tuple(
            sorted({int(component) for component in nominated_components})
        )
        contrasts.append(
            {
                "contrast_index": len(contrasts),
                "contrast_id": f"confirmation_contrast_{len(contrasts):05d}",
                "source_type": source_type,
                "source_id": source_id,
                "method_id": method_id,
                "budget": budget,
                "scientific_role": role,
                "cell_index": cell_index,
                "background_index": background_index,
                "selected_components": list(selected),
                "nominated_components": list(nominated),
                "condition_id": condition_id(selected),
            }
        )

    add_contrast(
        source_type="protocol",
        source_id="empty",
        method_id="",
        budget=0,
        role="empty_anchor",
        cell_index=0,
        components=(),
    )
    add_contrast(
        source_type="protocol",
        source_id="full",
        method_id="",
        budget=0,
        role="full_patch_anchor",
        cell_index=0,
        components=range(COMPONENT_COUNT),
    )
    background_rows = [
        tuple(record["selected_components"])
        for record in build_effect_background_manifest()
    ]

    for nomination in envelope["nominations"]:
        method_id = nomination["method_id"]
        budget = nomination["budget"]
        source_id = nomination["slot_id"]
        selected = tuple(nomination["components"])
        kind = nomination["nomination_kind"]
        if kind == "interaction_group6":
            for bits in range(64):
                cell = tuple(
                    component
                    for index, component in enumerate(selected)
                    if bits & (1 << index)
                )
                add_contrast(
                    source_type="nomination",
                    source_id=source_id,
                    method_id=method_id,
                    budget=budget,
                    role="group_lattice_cell",
                    cell_index=bits,
                    components=cell,
                    nominated_components=selected,
                )
        elif kind == "component":
            component = selected[0]
            for background_index, background in enumerate(background_rows):
                off = tuple(value for value in background if value != component)
                add_contrast(
                    source_type="nomination",
                    source_id=source_id,
                    method_id=method_id,
                    budget=budget,
                    role="component_background_off",
                    cell_index=0,
                    components=off,
                    background_index=background_index,
                    nominated_components=selected,
                )
                add_contrast(
                    source_type="nomination",
                    source_id=source_id,
                    method_id=method_id,
                    budget=budget,
                    role="component_background_on",
                    cell_index=1,
                    components=(*off, component),
                    background_index=background_index,
                    nominated_components=selected,
                )
        elif kind == "pair":
            first, second = selected
            for background_index, background in enumerate(background_rows):
                base = tuple(value for value in background if value not in selected)
                for bits in range(4):
                    cell = (
                        *base,
                        *((first,) if bits & 1 else ()),
                        *((second,) if bits & 2 else ()),
                    )
                    add_contrast(
                        source_type="nomination",
                        source_id=source_id,
                        method_id=method_id,
                        budget=budget,
                        role="pair_background_cell",
                        cell_index=bits,
                        components=cell,
                        background_index=background_index,
                        nominated_components=selected,
                    )
        else:
            add_contrast(
                source_type="nomination",
                source_id=source_id,
                method_id=method_id,
                budget=budget,
                role="sufficient_set",
                cell_index=0,
                components=selected,
                nominated_components=selected,
            )
            for index, component in enumerate(selected):
                add_contrast(
                    source_type="nomination",
                    source_id=source_id,
                    method_id=method_id,
                    budget=budget,
                    role="minimality_leave_one_out",
                    cell_index=index,
                    components=tuple(value for value in selected if value != component),
                    nominated_components=selected,
                )
            complement = tuple(
                value for value in range(COMPONENT_COUNT) if value not in selected
            )
            add_contrast(
                source_type="nomination",
                source_id=source_id,
                method_id=method_id,
                budget=budget,
                role="group_necessity_complement",
                cell_index=0,
                components=complement,
                nominated_components=selected,
            )

    for control in fixed_controls:
        selected = tuple(control["components"])
        for bits in range(64):
            cell = tuple(
                component
                for index, component in enumerate(selected)
                if bits & (1 << index)
            )
            add_contrast(
                source_type="control",
                source_id=control["control_id"],
                method_id="",
                budget=0,
                role="group_lattice_cell",
                cell_index=bits,
                components=cell,
                nominated_components=selected,
            )

    confirmation_patched = len(conditions) * PROMPTS_PER_SPLIT
    accounting = {
        "discovery_prompt_count": PROMPTS_PER_SPLIT,
        "discovery_mask_count": 512,
        "discovery_patched_examples": 512 * PROMPTS_PER_SPLIT,
        "discovery_clean_examples": 2 * PROMPTS_PER_SPLIT,
        "confirmation_prompt_count": PROMPTS_PER_SPLIT,
        "confirmation_unique_condition_count": len(conditions),
        "confirmation_contrast_reference_count": len(contrasts),
        "confirmation_patched_examples": confirmation_patched,
        "confirmation_clean_examples": 2 * PROMPTS_PER_SPLIT,
        "total_logical_examples": 512 * PROMPTS_PER_SPLIT
        + 2 * PROMPTS_PER_SPLIT
        + confirmation_patched
        + 2 * PROMPTS_PER_SPLIT,
    }
    return {
        "schema_version": "e29_15_confirmation_manifest_v1",
        "nomination_sha256": envelope["nomination_sha256"],
        "discovery_artifact_sha256": envelope["discovery_artifact_sha256"],
        "controls": [deepcopy(record) for record in fixed_controls],
        "effect_backgrounds": list(build_effect_background_manifest()),
        "estimands": {
            "component": NOMINATION_RULES["component"]["estimand"],
            "pair": NOMINATION_RULES["pair"]["estimand"],
            "sufficiency_target": SUFFICIENCY_TARGET,
            "sufficiency_absolute_tolerance": SUFFICIENCY_ABSOLUTE_TOLERANCE,
            "full_patch_is_anchor_not_target": True,
        },
        "conditions": conditions,
        "contrasts": contrasts,
        "query_accounting": accounting,
    }


MAXIMUM_CONFIRMATION_CONTRAST_REFERENCES = (
    len(METHOD_IDS)
    * len(DISCOVERY_BUDGETS)
    * (
        64
        + 6 * EFFECT_BACKGROUND_COUNT * 2
        + 6 * EFFECT_BACKGROUND_COUNT * 4
        + 1
        + MAXIMUM_SUFFICIENT_SET_SIZE
        + 1
    )
    + len(build_fixed_controls()) * 64
    + 2
)
QUERY_ACCOUNTING = {
    "discovery_prompt_count": PROMPTS_PER_SPLIT,
    "discovery_mask_count": 512,
    "nested_report_budgets": list(DISCOVERY_BUDGETS),
    "discovery_patched_examples": 512 * PROMPTS_PER_SPLIT,
    "discovery_clean_examples": 2 * PROMPTS_PER_SPLIT,
    "candidate_method_count": len(METHOD_IDS),
    "candidate_slot_count": len(METHOD_IDS)
    * len(DISCOVERY_BUDGETS)
    * CANDIDATE_SLOTS_PER_METHOD_BUDGET,
    "maximum_confirmation_contrast_references": MAXIMUM_CONFIRMATION_CONTRAST_REFERENCES,
    "maximum_confirmation_unique_conditions": MAXIMUM_CONFIRMATION_CONTRAST_REFERENCES,
    "maximum_confirmation_patched_examples": MAXIMUM_CONFIRMATION_CONTRAST_REFERENCES
    * PROMPTS_PER_SPLIT,
}


def build_e29_15_protocol_artifact() -> dict[str, Any]:
    """Return the complete discovery-stage commitment without response labels."""
    return {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "component_count": COMPONENT_COUNT,
        },
        "components": list(build_component_manifest()),
        "prompts": {
            "discovery": list(build_prompt_manifest("discovery")),
            "confirmation": list(build_prompt_manifest("confirmation")),
            "outcome_filtered_replacement_allowed": False,
        },
        "discovery_budgets": list(DISCOVERY_BUDGETS),
        "discovery_mask_manifest": list(build_discovery_mask_manifest()),
        "effect_background_manifest": list(build_effect_background_manifest()),
        "method_ids": list(METHOD_IDS),
        "method_recipes": deepcopy(METHOD_RECIPES),
        "discovery_fit_config": deepcopy(DISCOVERY_FIT_CONFIG),
        "nomination_rules": deepcopy(NOMINATION_RULES),
        "candidate_manifest": list(build_candidate_manifest()),
        "fixed_controls": list(build_fixed_controls()),
        "clean_panel_gate": deepcopy(CLEAN_PANEL_GATE),
        "response_coordinate": deepcopy(RESPONSE_COORDINATE),
        "scoring_plan": deepcopy(SCORING_PLAN),
        "query_accounting": deepcopy(QUERY_ACCOUNTING),
        "primary_claim_scope": (
            "TT rank 2 at budget 256 versus the parameter-matched MLP at budget "
            "256 for held-out interaction-group discovery; every other method and "
            "budget is descriptive, and official head-level CoAx and TN-SHAP "
            "superiority are outside this primary protocol"
        ),
        "two_stage_custody": {
            "discovery_labels_before_protocol_commitment_allowed": False,
            "confirmation_before_nomination_seal_allowed": False,
            "confirmation_manifest_embeds_nomination_sha256": True,
        },
    }


def validate_e29_15_protocol_artifact(protocol: Mapping[str, Any]) -> dict[str, Any]:
    copied = deepcopy(dict(protocol))
    expected = build_e29_15_protocol_artifact()
    if canonical_sha256(copied) != canonical_sha256(expected) or copied != expected:
        raise ValueError("E29.15 protocol artifact changed from the frozen pure design")
    return copied


__all__ = [
    "CANDIDATE_SLOTS_PER_METHOD_BUDGET",
    "CLEAN_PANEL_GATE",
    "COMPONENT_COUNT",
    "DISCOVERY_BUDGETS",
    "DISCOVERY_FIT_CONFIG",
    "EFFECT_BACKGROUND_COUNT",
    "METHOD_IDS",
    "METHOD_RECIPES",
    "MINIMUM_CLEAN_ENDPOINT",
    "MINIMUM_ENDPOINT_CORRECT_FRACTION",
    "NOMINATION_RULES",
    "PROMPT_RECORD_KEYS",
    "QUERY_ACCOUNTING",
    "RESPONSE_COORDINATE",
    "SCORING_PLAN",
    "SUFFICIENCY_TARGET",
    "build_candidate_manifest",
    "build_component_manifest",
    "build_confirmation_manifest",
    "build_discovery_mask_manifest",
    "build_discovery_masks",
    "build_e29_15_protocol_artifact",
    "build_effect_background_manifest",
    "build_effect_background_masks",
    "build_fixed_controls",
    "build_nomination_envelope",
    "build_prompt_manifest",
    "canonical_sha256",
    "validate_discovery_masks",
    "validate_e29_15_protocol_artifact",
    "validate_prompt_manifest",
]
