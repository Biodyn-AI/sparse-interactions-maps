# ruff: noqa: EXE002
"""Frozen, model-free protocol for E32.4 sparse internal edge sets.

E32.3 outcomes, including both fine-edge score panels, were open during this
design.  They are development ancestry only.  E32.4 is prospective solely for
the fresh panels and simultaneous edge-set outcomes declared here.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

CASE_ORDER = (58, 129, 25)
MANDATORY_SPLITS = ("discovery", "confirmation")

CORE_ANCESTRY = {
    "registration_sha256": (
        "1c42b4761573286af06604094efc5a85b0f873d773a99afe97d9767e25ab86a4"
    ),
    "summary_sha256": (
        "2a04b0659fb4b3615d642bfb36e0ad787cb2a98540ab5dacc43f910a199f1647"
    ),
    "source": "sealed_e32_3_v2_summary_selected_nodes",
}

CORE_CASE_ANCESTRY = {
    58: {
        "authorization_sha256": (
            "ebd0b6e01998951968fd284463dbbbd2d8f5b2ebf3619801feb544166451fd21"
        ),
        "selected_node_support_sha256": (
            "8864b22af413302ff38bff2198408f2c670213eb9817062b3c70a75093e47123"
        ),
        "selected_node_support_sha256_encoding": "compact_utf8_json_list",
    },
    129: {
        "authorization_sha256": (
            "6f76bbc0f2414290c4406a57e575144a7cd1e14a2732101afa8700cac31a6d25"
        ),
        "selected_node_support_sha256": (
            "44335790325331546874763c77527518f9d437405a025dd42b929b7cca333413"
        ),
        "selected_node_support_sha256_encoding": "compact_utf8_json_list",
    },
    25: {
        "authorization_sha256": (
            "61170db4123ee1f36ee93c02c96698edeeeb6a473c17e241039241039168abe1"
        ),
        "selected_node_support_sha256": (
            "6c0f645b3a6426c887d56fafbf78a70314a4cb0842af49ff75aee412ad32494c"
        ),
        "selected_node_support_sha256_encoding": "compact_utf8_json_list",
    },
}

# Canonical component order, not discovery-score order.
CORE_NODE_SUPPORTS = {
    58: (
        "blocks.0.attn.hook_result[0]",
        "blocks.0.attn.hook_result[1]",
        "blocks.0.attn.hook_result[3]",
        "blocks.0.hook_mlp_out",
        "blocks.1.attn.hook_result[1]",
        "blocks.1.hook_mlp_out",
        "blocks.2.attn.hook_result[0]",
        "blocks.2.hook_mlp_out",
    ),
    129: (
        "blocks.0.hook_mlp_out",
        "blocks.1.attn.hook_result[0]",
        "blocks.1.hook_mlp_out",
        "blocks.2.attn.hook_result[0]",
        "blocks.2.attn.hook_result[1]",
        "blocks.2.attn.hook_result[2]",
        "blocks.2.attn.hook_result[3]",
        "blocks.2.hook_mlp_out",
    ),
    25: (
        "blocks.0.attn.hook_result[0]",
        "blocks.0.hook_mlp_out",
        "blocks.1.attn.hook_result[1]",
        "blocks.1.attn.hook_result[3]",
        "blocks.1.hook_mlp_out",
    ),
}

CASE_GEOMETRY = {
    58: {
        "nodes": 15,
        "whole_edges": 87,
        "fine_edges": 207,
        "core_nodes": 8,
        "induced_whole_edges": 25,
        "induced_fine_edges": 45,
    },
    129: {
        "nodes": 15,
        "whole_edges": 87,
        "fine_edges": 207,
        "core_nodes": 8,
        "induced_whole_edges": 22,
        "induced_fine_edges": 48,
    },
    25: {
        "nodes": 10,
        "whole_edges": 33,
        "fine_edges": 73,
        "core_nodes": 5,
        "induced_whole_edges": 9,
        "induced_fine_edges": 17,
    },
}

PRIMARY_METHOD = "bidirectional_exact_mass"
DETERMINISTIC_METHOD_ORDER = (
    PRIMARY_METHOD,
    "bidirectional_directional_projection",
    "eap_absolute",
    "eap_ig5_absolute",
    "raw_whole_mass_equal_fine_split",
    "masked_rank_one_equal_fine_split",
)

RANDOM_CONTROL_REPLICATES = 256
RANDOM_CONTROL_QUANTILE = 0.95
RANDOM_CONTROL_95TH_INDEX = (
    math.ceil(RANDOM_CONTROL_QUANTILE * RANDOM_CONTROL_REPLICATES) - 1
)
RANDOM_CONTROL_NAMESPACE = "e32.4-induced-fine-core-random-support-v1"
RANDOM_CONTROL_SEED = 32_400_256

BRIDGE_GATES = {
    "minimum_necessity_projection": 0.75,
    "minimum_sufficiency_projection": 0.75,
    "maximum_necessity_residual_error": 0.25,
    "maximum_sufficiency_residual_error": 0.25,
    "minimum_necessity_loss_recovery": 0.75,
    "minimum_sufficiency_loss_recovery": 0.75,
}
ABSOLUTE_PRIMARY_GATES = {
    "minimum_necessity_projection": 0.90,
    "minimum_sufficiency_projection": 0.90,
    "maximum_necessity_residual_error": 0.10,
    "maximum_sufficiency_residual_error": 0.10,
    "minimum_necessity_loss_recovery": 0.90,
    "minimum_sufficiency_loss_recovery": 0.90,
}
ROBUSTNESS_GATES = dict(BRIDGE_GATES)
END_TO_END_ALIGNMENT_GATES = {
    "minimum_projection_capture": 0.75,
    "maximum_residual_error": 0.25,
    "minimum_loss_capture": 0.75,
}
SELECTOR_GATES = {
    "minimum_margin_over_random_95th_percentile": 0.05,
    "maximum_deficit_to_best_deterministic_baseline": 0.05,
}
GROUND_TRUTH_OFF_CORE_SCORE_FLOOR = -1.0
GROUND_TRUTH_FINE_LABEL_COUNTS = {
    58: {"full_internal_U": 3, "induced_core_I": 3},
    129: {"full_internal_U": 4, "induced_core_I": 4},
    25: {"full_internal_U": 3, "induced_core_I": 2},
}


@dataclass(frozen=True)
class SplitRequest:
    """One exact fresh raw-pool request and its input-only preflight custody."""

    case_id: int
    split: Literal["discovery", "confirmation"]
    required_count: int
    candidate_count: int
    clean_seed: int
    corrupt_seed: int
    raw_identity_sha256: str
    target_identity_sha256: str
    conservative_capacity: int
    exclude_all_e32_3_raw_candidate_role_rows: bool = True
    raw_duplicates_permitted_but_selected_rows_unique: bool = True
    selection_policy: str = (
        "ascending_first_jointly_behavior_correct_pair_with_globally_unique_"
        "clean_corrupt_rows_after_frozen_response_blind_exclusions"
    )


_SPLIT_VALUES = {
    58: {
        "discovery": (
            64,
            96,
            758300,
            758301,
            "bc8879564658999d87ffcd7e02ba257fd30f5c705575c179ed40fa806ea909e2",
            "fa2b4270c357382c3e315b511229eb2ee8aa14d2c535f959ffe2ee58f4ef17e2",
            96,
        ),
        "confirmation": (
            128,
            192,
            758400,
            758401,
            "b40acc67a23373d5a733bbbc05fb059d447ef6e238f157df3ade7b68d1c6eff2",
            "d7248d18b2daf46a336cfcc9f5ce9fdfaac632bcb2b773faffa17f0e76c4d6c3",
            192,
        ),
    },
    129: {
        "discovery": (
            64,
            96,
            829300,
            829301,
            "cca526b44bd06f9191135b1e764323aecd7c647dcac3d9978605898fdcf38a35",
            "12776f7293cad6efe98979877baf2211e68bc75a0844d08672093216b28e9bc2",
            96,
        ),
        "confirmation": (
            128,
            192,
            829400,
            829401,
            "42627b1c5d57412b0bdf6bba9b8051213d36929f4791a25ef0967f89228f22a2",
            "c71665081e17bad3f687bec8d5849d96f77b723b75898ff9f298be7aa8516e43",
            192,
        ),
    },
    25: {
        "discovery": (
            64,
            96,
            725300,
            725301,
            "1e55a22b22df3a7b40a870498493596b9034ebcc1e192f9b10d21b870eab3fad",
            "63af3ad08c7639822dedb4dbe88672e5bdb860389fd55a73569359c297b51382",
            86,
        ),
        "confirmation": (
            128,
            192,
            725400,
            725401,
            "c4c8a0cbb9ff48bd9973e06a626644c6708180a9e2b6debe4d9d2e18eaa47719",
            "00b41e9e01aab9e870b3b79117af6273652c5d7cbebdbb89649bfce386f80848",
            172,
        ),
    },
}

INPUT_PREFLIGHT = {
    "model_calls": 0,
    "old_raw_role_exclusion_unique_rows": {58: 768, 129: 768, 25: 751},
    "old_new_role_overlap_count": {58: 0, 129: 0, 25: 21},
    "case25_discovery_capacity_after_old_exclusions": 86,
    "case25_confirmation_capacity_after_any_discovery": 172,
}


def split_request(case_id: int, split: str) -> SplitRequest:
    """Return an immutable request; adaptive growth, reseeding, and retry are barred."""
    if case_id not in CASE_ORDER:
        raise ValueError(f"case_id must be one of {CASE_ORDER!r}")
    if split not in MANDATORY_SPLITS:
        raise ValueError(f"split must be one of {MANDATORY_SPLITS!r}")
    return SplitRequest(case_id, split, *_SPLIT_VALUES[case_id][split])  # type: ignore[arg-type]


def primary_support_size(case_id: int) -> int:
    """Use floor(10% of all internal fine edges), never 10% of the core."""
    try:
        geometry = CASE_GEOMETRY[case_id]
    except KeyError as error:
        raise ValueError(f"case_id must be one of {CASE_ORDER!r}") from error
    size = math.floor(0.10 * geometry["fine_edges"])
    if not (0 < size <= geometry["induced_fine_edges"]):
        raise RuntimeError("frozen support budget escapes the induced fine core")
    return size


def descriptive_support_sizes(case_id: int) -> dict[str, int]:
    """Return D-only exact-mass curve budgets; neither can replace primary k10."""
    if case_id not in CASE_ORDER:
        raise ValueError(f"case_id must be one of {CASE_ORDER!r}")
    total = CASE_GEOMETRY[case_id]["fine_edges"]
    sizes = {"k05": math.floor(0.05 * total), "k15": math.floor(0.15 * total)}
    if not 0 < sizes["k05"] < primary_support_size(case_id) < sizes["k15"]:
        raise RuntimeError("descriptive support budgets lost their frozen order")
    return sizes


def _validated_fine_ids(fine_edge_ids: Sequence[str]) -> tuple[str, ...]:
    fine = tuple(fine_edge_ids)
    if (
        not fine
        or len(fine) != len(set(fine))
        or any(not isinstance(edge, str) or not edge for edge in fine)
    ):
        raise ValueError("fine_edge_ids must be unique nonempty strings")
    return fine


def deterministic_random_supports(
    *, case_id: int, fine_edge_ids: Sequence[str], support_size: int
) -> tuple[tuple[str, ...], ...]:
    """Return 256 distinct pseudo-uniform k-subsets in canonical edge order.

    For each replicate and rejection nonce, independent SHA-256 keys rank all
    edges.  Taking the k smallest keys is uniform over k-subsets under the hash
    idealization.  Rejection only removes duplicate supports and is entirely
    response-blind.
    """
    if case_id not in CASE_ORDER:
        raise ValueError(f"case_id must be one of {CASE_ORDER!r}")
    fine = _validated_fine_ids(fine_edge_ids)
    if (
        isinstance(support_size, bool)
        or not isinstance(support_size, int)
        or not 0 < support_size <= len(fine)
    ):
        raise ValueError("support_size must lie in [1, len(fine_edge_ids)]")
    if math.comb(len(fine), support_size) < RANDOM_CONTROL_REPLICATES:
        raise ValueError("fine universe has fewer than 256 distinct k-subsets")

    accepted: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for replicate in range(RANDOM_CONTROL_REPLICATES):
        nonce = 0
        while True:
            keyed = []
            for index, edge in enumerate(fine):
                payload = (
                    f"{RANDOM_CONTROL_NAMESPACE}|seed={RANDOM_CONTROL_SEED}|"
                    f"case={case_id}|k={support_size}|replicate={replicate}|"
                    f"nonce={nonce}|index={index}|edge={edge}"
                ).encode()
                keyed.append((hashlib.sha256(payload).digest(), index))
            chosen = {index for _digest, index in sorted(keyed)[:support_size]}
            support = tuple(edge for index, edge in enumerate(fine) if index in chosen)
            if support not in seen:
                seen.add(support)
                accepted.append(support)
                break
            nonce += 1
            if nonce > 1_000_000:
                raise RuntimeError(
                    "random-control duplicate rejection did not terminate"
                )
    return tuple(accepted)


def _query_accounting_stage(
    physical_forwards: int,
    model_examples: int,
    gradient_forwards: int = 0,
    gradient_examples: int = 0,
) -> dict[str, int]:
    return {
        "physical_forwards": physical_forwards,
        "model_examples": model_examples,
        "gradient_forwards": gradient_forwards,
        "gradient_examples": gradient_examples,
        "backward_passes_inferred_from_pinned_control_flow": gradient_forwards,
    }


# These exact D-family counts are part of the protocol, not merely a ceiling.
EXPECTED_DISCOVERY_QUERY_ACCOUNTING = {
    58: {
        "behavioral_eligibility": _query_accounting_stage(2, 192),
        "whole_component_atlas": _query_accounting_stage(17, 7_808),
        "core_singleton_replacement": _query_accounting_stage(94, 6_016),
        "regular_eap": _query_accounting_stage(2, 128, 1, 64),
        "eap_ig5": _query_accounting_stage(7, 448, 6, 384),
    },
    129: {
        "behavioral_eligibility": _query_accounting_stage(2, 192),
        "whole_component_atlas": _query_accounting_stage(17, 7_808),
        "core_singleton_replacement": _query_accounting_stage(100, 6_400),
        "regular_eap": _query_accounting_stage(2, 128, 1, 64),
        "eap_ig5": _query_accounting_stage(7, 448, 6, 384),
    },
    25: {
        "behavioral_eligibility": _query_accounting_stage(2, 192),
        "whole_component_atlas": _query_accounting_stage(15, 4_032),
        "core_singleton_replacement": _query_accounting_stage(38, 2_432),
        "regular_eap": _query_accounting_stage(2, 128, 1, 64),
        "eap_ig5": _query_accounting_stage(7, 448, 6, 384),
    },
}


# Discovery totals sum the exact families above. Confirmation ceilings include
# behavior eligibility plus separate fresh resample and mean-corrupt models.
EXPECTED_QUERY_CEILINGS = {
    58: {
        "discovery": {
            "physical_forwards": 122,
            "model_examples": 14_592,
            "gradient_forwards": 7,
            "gradient_examples": 448,
            "backward_passes_inferred_from_pinned_control_flow": 7,
        },
        "confirmation": {
            "maximum_physical_forwards": 545,
            "maximum_model_examples": 69_888,
        },
    },
    129: {
        "discovery": {
            "physical_forwards": 128,
            "model_examples": 14_976,
            "gradient_forwards": 7,
            "gradient_examples": 448,
            "backward_passes_inferred_from_pinned_control_flow": 7,
        },
        "confirmation": {
            "maximum_physical_forwards": 545,
            "maximum_model_examples": 69_888,
        },
    },
    25: {
        "discovery": {
            "physical_forwards": 64,
            "model_examples": 7_232,
            "gradient_forwards": 7,
            "gradient_examples": 448,
            "backward_passes_inferred_from_pinned_control_flow": 7,
        },
        "confirmation": {
            "maximum_physical_forwards": 545,
            "maximum_model_examples": 69_888,
        },
    },
    "overall": {
        "maximum_physical_forwards": 1_949,
        "maximum_model_examples": 246_464,
        "gradient_forwards": 21,
        "gradient_examples": 1_344,
        "backward_passes_inferred_from_pinned_control_flow": 21,
    },
    "confirmation_formula": {
        "eligibility_forwards": 2,
        "resample_maximum_nonempty_distinct_masks": 532,
        "resample_maximum_forwards_including_cache_and_two_baselines": 535,
        "mean_corrupt_masks": 5,
        "mean_corrupt_forwards_including_cache_and_two_baselines": 8,
        "actual_counts_must_equal_the_deduplicated_authorized_mask_plan": True,
    },
}


def protocol_manifest() -> dict[str, object]:
    """Return the complete JSON-compatible registration core."""
    return {
        "schema_version": "e32_4_sparse_internal_edge_set_protocol_v1",
        "case_order": CASE_ORDER,
        "mandatory_all_cases": True,
        "core_ancestry": {
            "registration_sha256": CORE_ANCESTRY["registration_sha256"],
            "summary_sha256": CORE_ANCESTRY["summary_sha256"],
            "source": CORE_ANCESTRY["source"],
            "cases": {
                str(case_id): dict(CORE_CASE_ANCESTRY[case_id])
                for case_id in CASE_ORDER
            },
        },
        "core_node_supports": {
            str(case_id): support for case_id, support in CORE_NODE_SUPPORTS.items()
        },
        "case_geometry": {
            str(case_id): dict(values) for case_id, values in CASE_GEOMETRY.items()
        },
        "split_requests": {
            str(case_id): {
                split: asdict(split_request(case_id, split))
                for split in MANDATORY_SPLITS
            }
            for case_id in CASE_ORDER
        },
        "input_preflight": dict(INPUT_PREFLIGHT),
        "primary_method": PRIMARY_METHOD,
        "deterministic_method_order": DETERMINISTIC_METHOD_ORDER,
        "support_budgets": {
            str(case_id): {
                "primary_k10": primary_support_size(case_id),
                **descriptive_support_sizes(case_id),
            }
            for case_id in CASE_ORDER
        },
        "random_control": {
            "replicates": RANDOM_CONTROL_REPLICATES,
            "namespace": RANDOM_CONTROL_NAMESPACE,
            "seed": RANDOM_CONTROL_SEED,
            "family": "uniform_unique_k_subsets_of_same_induced_fine_core",
            "without_replacement_within_support": True,
            "duplicate_support_rejection": True,
            "empirical_95th_sorted_index": RANDOM_CONTROL_95TH_INDEX,
        },
        "granularity": {
            "atomic_edge": "auto_circuit_internal_all_position_fine_coordinate",
            "separate_qkv": True,
            "candidate_support": "I_induced_by_frozen_e32_3_node_core",
            "sparsity_denominator": "U_all_internal_fine_edges",
            "excluded_boundary_edges_always_conditioned": True,
        },
        "interventions": {
            "recipient": "clean_input_computation",
            "primary_donor": "paired_corrupt_source_activations_resample",
            "robustness_donor": "batch_tokenwise_mean_corrupt_source_activations",
            "anchor": "U",
            "primary_support": "S_subset_I",
            "primary_complement": "U_minus_S",
            "bridge_support": "I",
            "bridge_complement": "U_minus_I",
            "within_core_fallback_complement": "I_minus_S_primary",
            "simultaneous_all_position_masks": True,
        },
        "gates": {
            "bridge": dict(BRIDGE_GATES),
            "absolute_primary": dict(ABSOLUTE_PRIMARY_GATES),
            "mean_corrupt_robustness": dict(ROBUSTNESS_GATES),
            "end_to_end_alignment": dict(END_TO_END_ALIGNMENT_GATES),
            "selector": dict(SELECTOR_GATES),
        },
        "verdict_hierarchy": {
            "primary": "absolute_boundary_conditioned_resample_sparse_set",
            "separate_selector_usefulness": (
                "absolute_primary_and_random95_and_best_deterministic"
            ),
            "selector_requires_absolute_sparse_set": True,
            "separate_robustness": "batch_tokenwise_mean_corrupt",
            "optional_end_to_end": "primary_plus_actual_input_endpoint_alignment",
            "all_three_cases_required_with_no_majority_rule": True,
        },
        "baseline_feasibility": {
            "included": DETERMINISTIC_METHOD_ORDER,
            "acdc": (
                "not_included_no_pinned_fine_qkv_implementation_in_repository_and_"
                "adaptive_iterative_search_would_require_separate_registration"
            ),
            "edge_pruning": (
                "not_included_no_pinned_implementation_in_repository_and_learned_"
                "mask_optimization_would_require_separate_training_and_accounting"
            ),
        },
        "expected_query_ceilings": {
            str(key): value for key, value in EXPECTED_QUERY_CEILINGS.items()
        },
        "expected_discovery_query_accounting": {
            str(case_id): stages
            for case_id, stages in EXPECTED_DISCOVERY_QUERY_ACCOUNTING.items()
        },
        "ground_truth_used_for_selection_or_gates": False,
        "ground_truth_metrics": "descriptive_score_stage_only",
        "ground_truth_diagnostic": {
            "stage": "score_only_after_durable_score_attempt",
            "ranking_universe": "U_all_internal_fine_edges",
            "queried_scores": "I_induced_by_frozen_e32_3_node_core",
            "outside_I_score": GROUND_TRUTH_OFF_CORE_SCORE_FLOOR,
            "outside_I_convention": "fixed_below_all_nonnegative_queried_scores",
            "fine_label_rule": (
                "positive_iff_regular_eap_raw_source_target_is_in_official_raw_"
                "edges_restricted_to_full_internal_U"
            ),
            "fine_label_whole_child_expansion": False,
            "whole_promotion_used_only_for_separate_whole_diagnostic": True,
            "expected_fine_label_counts": {
                str(case_id): counts
                for case_id, counts in GROUND_TRUTH_FINE_LABEL_COUNTS.items()
            },
            "case25_induced_core_fine_truth_coverage": {
                "numerator": 2,
                "denominator": 3,
            },
            "used_for_selection_or_gates": False,
        },
        "resample_primary_can_be_rescued_by_robustness": False,
        "development_exposure": {
            "e32_3_discovery_and_confirmation_fine_vectors_opened": True,
            "e32_3_top5_top10_mass_diagnostics_opened": True,
            "primary_support_sizes_20_20_7_and_point90_gates_chosen_after_e32_3_individual_edge_mass_diagnostics": True,
            "fresh_simultaneous_set_outcomes_opened": False,
        },
        "claim_boundary": (
            "boundary_conditioned_internal_inter_component_fine_edge_support;"
            "not_direct_wiring;not_minimal;not_sota;not_model_generalization;"
            "not_natural_llm_or_other_ablation_generalization"
        ),
    }


__all__ = [
    "ABSOLUTE_PRIMARY_GATES",
    "BRIDGE_GATES",
    "CASE_GEOMETRY",
    "CASE_ORDER",
    "CORE_ANCESTRY",
    "CORE_CASE_ANCESTRY",
    "CORE_NODE_SUPPORTS",
    "DETERMINISTIC_METHOD_ORDER",
    "END_TO_END_ALIGNMENT_GATES",
    "EXPECTED_DISCOVERY_QUERY_ACCOUNTING",
    "EXPECTED_QUERY_CEILINGS",
    "GROUND_TRUTH_FINE_LABEL_COUNTS",
    "GROUND_TRUTH_OFF_CORE_SCORE_FLOOR",
    "INPUT_PREFLIGHT",
    "MANDATORY_SPLITS",
    "PRIMARY_METHOD",
    "RANDOM_CONTROL_95TH_INDEX",
    "RANDOM_CONTROL_NAMESPACE",
    "RANDOM_CONTROL_REPLICATES",
    "RANDOM_CONTROL_SEED",
    "ROBUSTNESS_GATES",
    "SELECTOR_GATES",
    "SplitRequest",
    "descriptive_support_sizes",
    "deterministic_random_supports",
    "primary_support_size",
    "protocol_manifest",
    "split_request",
]
