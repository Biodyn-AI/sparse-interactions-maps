# ruff: noqa: EXE002
"""Frozen, model-free protocol constants for the E32.5 edge selector."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

EXPERIMENT_ID = "e32_5_signed_group_edge_selector_v1"

DEVELOPMENT_ATTEMPT_PATH = (
    "notes/E32.5-SIGNED-GROUP-EDGE-SELECTOR-DEVELOPMENT-ATTEMPT.json"
)
DEVELOPMENT_RUN_DIRECTORY = "runs/e32_5_signed_group_edge_selector_v1_development"
DEVELOPMENT_TERMINAL_FILENAMES = (
    "development_acquisition_attempt.json",
    "case58_input_preflight.json",
    "case58_behavior_screen.json",
    "case58_development.pt",
    "case58_development_custody.json",
    "development_observation.json",
    "development_summary.json",
)
DEVELOPMENT_ATTEMPT_STATUS = (
    "nonclaim_case58_development_attempt_reserved_before_run_directory_or_model"
)
DEVELOPMENT_SUMMARY_STATUS = (
    "nonclaim_method_development_complete_algorithm_not_frozen_no_claim"
)

DEVELOPMENT_CASE_ORDER = (58,)
GENERALIZATION_CASE_ORDER = (63, 103, 82)
CASE_ORDER = (*DEVELOPMENT_CASE_ORDER, *GENERALIZATION_CASE_ORDER)

SEARCH_EXAMPLE_COUNT = 32
VALIDATION_EXAMPLE_COUNT = 32
DISCOVERY_ACQUISITION_EXAMPLE_COUNT = SEARCH_EXAMPLE_COUNT + VALIDATION_EXAMPLE_COUNT
CONFIRMATION_EXAMPLE_COUNT = 128
SEARCH_ROW_BOUNDS = (0, SEARCH_EXAMPLE_COUNT)
VALIDATION_ROW_BOUNDS = (
    SEARCH_EXAMPLE_COUNT,
    DISCOVERY_ACQUISITION_EXAMPLE_COUNT,
)

CONFIRMATION_SUPPORT_FRACTION = 0.15
LOCAL_SEARCH_ROUNDS = 3
REMOVAL_RESIDUAL_QUOTA = 2
REMOVAL_DIRECTIONAL_QUOTA = 2
ADDITION_RESIDUAL_QUOTA = 4
ADDITION_DIRECTIONAL_QUOTA = 4

RANDOM_CONTROL_REPLICATES = 256
RANDOM_CONTROL_95TH_INDEX = math.ceil(0.95 * RANDOM_CONTROL_REPLICATES) - 1
RANDOM_CONTROL_NAMESPACE = "e32.5/random-controls/v1/sha256-rejection-unrank"
RANDOM_CONTROL_SEED_SHA256 = (
    "8dee1d79fff2f82a0f3e5ea0561596f07b32c4cbcac2e5977b24af313a020bf9"
)

PRIMARY_METHOD = "signed_group_local_search"
CONTEXTUAL_SIGNED_STATIC_METHOD = "contextual_signed_partition_static"
CONTEXTUAL_UNSIGNED_MASS_METHOD = "contextual_unsigned_partition_mass"
DETERMINISTIC_BASELINE_ORDER = (
    CONTEXTUAL_SIGNED_STATIC_METHOD,
    CONTEXTUAL_UNSIGNED_MASS_METHOD,
    "eap_absolute",
    "eap_ig5_absolute",
)

PRIMARY_GATES = {
    "minimum_projection": 0.90,
    "maximum_residual_error": 0.10,
    "minimum_loss_recovery": 0.90,
}
SELECTOR_GATES = {
    "minimum_margin_over_random_95th": 0.05,
    "maximum_deficit_to_best_deterministic": 0.05,
}
SIGN_AWARE_ADVANTAGE_MARGIN = 0.05
GROUP_INTERACTION_ADVANTAGE_MARGIN = 0.05
END_TO_END_ALIGNMENT_GATES = {
    "minimum_projection_capture": 0.75,
    "minimum_loss_capture": 0.75,
    "maximum_residual_error": 0.25,
}

CLAIM_BOUNDARY = (
    "prospective_boundary_conditioned_full_internal_fine_edge_support;"
    "not_direct_wiring;not_minimal;not_global_optimum;not_sota;"
    "not_ground_truth_recovery;not_model_family_generalization"
)
REPAIR_CLAIM_BOUNDARY = (
    "within_model_case58_repair_only;not_prospective_panel_evidence;"
    "not_model_scaling;not_direct_wiring;not_minimal;not_global_optimum;not_sota"
)


@dataclass(frozen=True)
class CaseGeometry:
    case_id: int
    full_fine_edge_count: int
    role: str


CASE_GEOMETRY = {
    58: CaseGeometry(58, 207, "nonclaim_method_development"),
    63: CaseGeometry(63, 73, "prospective_generalization"),
    103: CaseGeometry(103, 207, "prospective_generalization"),
    82: CaseGeometry(82, 406, "prospective_generalization"),
}


def confirmation_support_size(case_id: int) -> int:
    """Return the literal at-most-15%-of-full-U support size."""
    if isinstance(case_id, bool) or case_id not in CASE_GEOMETRY:
        raise ValueError(f"unknown E32.5 case_id {case_id!r}")
    size = math.floor(
        CONFIRMATION_SUPPORT_FRACTION * CASE_GEOMETRY[case_id].full_fine_edge_count
    )
    if size < 1:
        raise RuntimeError("registered full fine universe is too small")
    return size


def _lexicographic_combination_indices(
    *, population_size: int, support_size: int, rank: int
) -> tuple[int, ...]:
    """Unrank one lexicographically ordered fixed-cardinality subset."""
    indices: list[int] = []
    next_index = 0
    remaining_rank = rank
    for remaining_slots in range(support_size, 0, -1):
        for candidate in range(next_index, population_size):
            suffix_count = math.comb(
                population_size - candidate - 1, remaining_slots - 1
            )
            if remaining_rank < suffix_count:
                indices.append(candidate)
                next_index = candidate + 1
                break
            remaining_rank -= suffix_count
        else:  # pragma: no cover - protected by the public rank bounds
            raise RuntimeError("combination rank could not be decoded")
    return tuple(indices)


def deterministic_random_supports(
    case_id: int, fine_edge_ids: Sequence[str], k: int
) -> tuple[tuple[str, ...], ...]:
    """Return the 256 frozen, unique uniform random-control supports.

    SHA-256 rejection sampling draws an unbiased integer rank in the space of
    all k-subsets. Duplicate ranks are rejected, and each support is emitted in
    the caller-provided canonical full-U order.
    """
    if isinstance(case_id, bool) or case_id not in CASE_GEOMETRY:
        raise ValueError(f"unknown E32.5 case_id {case_id!r}")
    if isinstance(fine_edge_ids, (str, bytes)):
        raise TypeError("fine_edge_ids must be a sequence")
    edge_ids = tuple(fine_edge_ids)
    if (
        not edge_ids
        or len(edge_ids) != len(set(edge_ids))
        or any(not isinstance(edge_id, str) or not edge_id for edge_id in edge_ids)
    ):
        raise ValueError("fine_edge_ids must be unique nonempty strings")
    expected_edge_count = CASE_GEOMETRY[case_id].full_fine_edge_count
    if len(edge_ids) != expected_edge_count:
        raise ValueError(
            f"case {case_id} full fine universe size must be {expected_edge_count}"
        )
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    expected_k = confirmation_support_size(case_id)
    if k != expected_k:
        raise ValueError(f"case {case_id} registered support size must be {expected_k}")

    subset_count = math.comb(len(edge_ids), k)
    if subset_count < RANDOM_CONTROL_REPLICATES:
        raise RuntimeError("registered universe cannot supply unique random controls")
    universe_sha256 = hashlib.sha256(
        json.dumps(edge_ids, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    sample_limit = (1 << 256) - ((1 << 256) % subset_count)
    seed = bytes.fromhex(RANDOM_CONTROL_SEED_SHA256)
    seen_ranks: set[int] = set()
    supports: list[tuple[str, ...]] = []
    counter = 0
    while len(supports) < RANDOM_CONTROL_REPLICATES:
        payload = (
            f"{RANDOM_CONTROL_NAMESPACE}\0{case_id}\0{len(edge_ids)}\0{k}\0"
            f"{universe_sha256}\0{counter}"
        ).encode("ascii")
        draw = int.from_bytes(hashlib.sha256(seed + b"\0" + payload).digest(), "big")
        counter += 1
        if draw >= sample_limit:
            continue
        rank = draw % subset_count
        if rank in seen_ranks:
            continue
        seen_ranks.add(rank)
        indices = _lexicographic_combination_indices(
            population_size=len(edge_ids), support_size=k, rank=rank
        )
        supports.append(tuple(edge_ids[index] for index in indices))
    return tuple(supports)


def maximum_exact_search_supports() -> int:
    """Initial support plus every possible swap in all frozen rounds."""
    return 1 + LOCAL_SEARCH_ROUNDS * (
        REMOVAL_RESIDUAL_QUOTA + REMOVAL_DIRECTIONAL_QUOTA
    ) * (ADDITION_RESIDUAL_QUOTA + ADDITION_DIRECTIONAL_QUOTA)


def maximum_exact_validation_supports() -> int:
    """Initial support and at most one accepted support per search round."""
    return LOCAL_SEARCH_ROUNDS + 1


def selector_query_ceiling(case_id: int) -> dict[str, int]:
    """Response-independent output-mask ceiling before implementation overhead."""
    fine_count = CASE_GEOMETRY[case_id].full_fine_edge_count
    search_supports = maximum_exact_search_supports()
    validation_supports = maximum_exact_validation_supports()
    return {
        "singleton_partition_physical_forwards": 2 * fine_count + 4,
        "full_u_singleton_partition_masks": 2 * fine_count,
        "maximum_adaptive_exact_acquisitions": LOCAL_SEARCH_ROUNDS + 1,
        "initial_exact_acquisition_physical_forwards": 6,
        "maximum_physical_forwards_per_full_swap_round": 4
        + 2
        * (REMOVAL_RESIDUAL_QUOTA + REMOVAL_DIRECTIONAL_QUOTA)
        * (ADDITION_RESIDUAL_QUOTA + ADDITION_DIRECTIONAL_QUOTA),
        "maximum_selector_physical_forwards": 2 * fine_count
        + 4
        + 6
        + LOCAL_SEARCH_ROUNDS
        * (
            4
            + 2
            * (REMOVAL_RESIDUAL_QUOTA + REMOVAL_DIRECTIONAL_QUOTA)
            * (ADDITION_RESIDUAL_QUOTA + ADDITION_DIRECTIONAL_QUOTA)
        ),
        "search_exact_supports": search_supports,
        "search_exact_masks": 2 * search_supports,
        "validation_in_memory_path_scores": validation_supports,
        "validation_additional_model_masks": 0,
        "dense_complement_active_mask_coordinates": fine_count * (fine_count - 1),
    }


def protocol_manifest(*, development_tuning_disclosure: str) -> dict[str, Any]:
    """Return the final JSON-safe registration fragment after the pilot."""
    if (
        not isinstance(development_tuning_disclosure, str)
        or not development_tuning_disclosure.strip()
        or "pending" in development_tuning_disclosure.casefold()
    ):
        raise ValueError(
            "development_tuning_disclosure must literally state the postpilot changes"
        )
    return {
        "schema_version": f"{EXPERIMENT_ID}_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "case_geometry": {
            str(case_id): asdict(geometry)
            for case_id, geometry in CASE_GEOMETRY.items()
        },
        "development_ancestry": {
            "case_order": list(DEVELOPMENT_CASE_ORDER),
            "case58_is_nonclaim_method_development": True,
            "e32_4_outputs_are_development_only": True,
            "case58_fresh_development_rows_and_outcomes_must_be_excluded_from_claim": True,
            "algorithm_freezes_only_after_the_case58_development_pilot": True,
            "optional_case58_repair_confirmation_is_not_in_the_generalization_panel": True,
            "case58_repair_requires_a_separate_positive_within_model_verdict": True,
            "support_fraction_and_local_search_hyperparameters_are_development_exposed": True,
        },
        "development_pilot_lifecycle": {
            "attempt_path": DEVELOPMENT_ATTEMPT_PATH,
            "run_directory": DEVELOPMENT_RUN_DIRECTORY,
            "terminal_filenames_in_order": list(DEVELOPMENT_TERMINAL_FILENAMES),
            "attempt_status": DEVELOPMENT_ATTEMPT_STATUS,
            "summary_status": DEVELOPMENT_SUMMARY_STATUS,
            "summary_binds_attempt_source_artifact_custody_observation_and_accounting": True,
            "summary_asserts_no_repair_or_prospective_confirmation_calls": True,
            "final_registration_revalidates_entire_pilot_file_and_source_closure": True,
            "final_algorithm_and_source_freeze_occurs_after_pilot": True,
            "case58_pilot_is_excluded_from_every_claim": True,
            "final_run_reruns_case58_pilot": False,
            "development_tuning_disclosure": development_tuning_disclosure.strip(),
        },
        "prospective_claim": {
            "case_order": list(GENERALIZATION_CASE_ORDER),
            "all_cases_required": True,
            "majority_rule_allowed": False,
            "claim_boundary": CLAIM_BOUNDARY,
        },
        "splits": {
            "discovery_acquisition_examples": DISCOVERY_ACQUISITION_EXAMPLE_COUNT,
            "search_examples": SEARCH_EXAMPLE_COUNT,
            "validation_examples": VALIDATION_EXAMPLE_COUNT,
            "confirmation_examples": CONFIRMATION_EXAMPLE_COUNT,
            "search_selects_an_accepted_path": True,
            "validation_selects_earliest_best_only_along_the_frozen_path": True,
            "search_and_validation_are_slices_of_one_batch64_acquisition": True,
            "support_selection_rows": list(SEARCH_ROW_BOUNDS),
            "validation_path_selection_rows": list(VALIDATION_ROW_BOUNDS),
            "all_primary_static_shortlist_and_acceptance_use_search_rows": True,
            "validation_only_chooses_earliest_support_on_the_accepted_search_path": True,
            "validation_adds_no_model_masks": True,
            "confirmation_is_untouched_until_support_authorization": True,
        },
        "selector": {
            "selectable_universe": "U_all_internal_fine_edges",
            "boundary_edges_are_always_conditioned_and_excluded": True,
            "confirmation_support_fraction": CONFIRMATION_SUPPORT_FRACTION,
            "support_size_rounding": "floor",
            "singleton_plan": "F({e})_and_F(U\\{e})_for_every_fine_edge",
            "singleton_signed_score": "sqrt(relu(necessity_projection)*relu(sufficiency_projection))",
            "residual_proxy_may_choose_or_stop": False,
            "exact_group_score_chooses_and_stops": True,
            "uninformative_internal_endpoint": (
                "scientific_fail_uninformative_internal_endpoint_before_search_or_C"
            ),
            "exact_group_orientation": "corrupt_into_clean",
            "exact_candidate_masks": ["S", "U\\S"],
            "local_search_rounds": LOCAL_SEARCH_ROUNDS,
            "adaptive_acquisitions_after_singleton_atlas": {
                "initial_support_plan": 1,
                "maximum_swap_round_plans": LOCAL_SEARCH_ROUNDS,
                "maximum_total": LOCAL_SEARCH_ROUNDS + 1,
            },
            "removal_channels": {
                "antagonistic_residual": REMOVAL_RESIDUAL_QUOTA,
                "lowest_static_directional": REMOVAL_DIRECTIONAL_QUOTA,
            },
            "addition_channels": {
                "helpful_residual": ADDITION_RESIDUAL_QUOTA,
                "static_directional_group_diverse": ADDITION_DIRECTIONAL_QUOTA,
            },
            "swap_cardinality": "one_for_one",
            "acceptance": "strict_exact_joint_improvement",
            "ties": "canonical_full_U_index_then_earliest_path",
            "complexity": {
                "model_forward_queries": "linear_in_F_plus_fixed_RAB",
                "score_memory": "O(FP);no_F_squared_Gram",
                "dense_complement_mask_coordinates_and_host_writes": "O(F_squared)",
                "largest_registered_F406_active_complement_coordinates": 406 * 405,
            },
        },
        "baselines": {
            "deterministic_order": list(DETERMINISTIC_BASELINE_ORDER),
            "contextual_signed_and_unsigned_exact_are_free_from_partition_atlas": True,
            "regular_eap_and_eap_ig5_are_field_baselines": True,
            "raw_whole_and_masked_rank_one_dropped_to_avoid_unrelated_component_atlas": True,
            "same_frozen_cardinality_as_primary": True,
            "deterministic_support_selection_rows": list(SEARCH_ROW_BOUNDS),
            "deterministic_supports_never_use_validation_rows": True,
            "random_support_selection_is_response_blind": True,
            "random_replicates": RANDOM_CONTROL_REPLICATES,
            "random_95th_sorted_index": RANDOM_CONTROL_95TH_INDEX,
            "random_control_sampler": {
                "hash": "sha256",
                "namespace": RANDOM_CONTROL_NAMESPACE,
                "seed_sha256": RANDOM_CONTROL_SEED_SHA256,
                "sampling": "exact_rejection_sampled_uniform_combination_rank",
                "replicates_are_unique": True,
                "within_support_order": "canonical_full_U_index",
                "universe_identity_is_bound": True,
            },
            "acdc_and_edge_pruning_are_not_claimed_comparators": True,
        },
        "gates": {
            "primary": dict(PRIMARY_GATES),
            "selector": dict(SELECTOR_GATES),
            "sign_aware_advantage_margin": SIGN_AWARE_ADVANTAGE_MARGIN,
            "group_interaction_advantage_margin": GROUP_INTERACTION_ADVANTAGE_MARGIN,
            "end_to_end_alignment": dict(END_TO_END_ALIGNMENT_GATES),
            "end_to_end_alignment_is_a_separate_actual_corrupt_endpoint_tier": True,
            "end_to_end_alignment_cannot_rescue_or_define_the_boundary_conditioned_primary": True,
            "sign_and_interaction_advantage_are_separate_claim_tiers": True,
            "D_validation_is_support_selection_not_independent_primary_evidence": True,
            "C_absolute_gate_is_not_vetoed_by_D_validation_score": True,
            "absolute_scaled_replication": (
                "case82_absolute_pass_and_at_least_two_of_three_absolute_pass;"
                "not_generalization"
            ),
            "selector_scaled_replication": (
                "case82_selector_pass_and_at_least_two_of_three_selector_pass;"
                "not_generalization"
            ),
            "unqualified_panel_verdict_aliases_the_stronger_selector_tier": True,
        },
        "query_ceilings": {
            str(case_id): selector_query_ceiling(case_id) for case_id in CASE_ORDER
        },
    }


__all__ = [
    "ADDITION_DIRECTIONAL_QUOTA",
    "ADDITION_RESIDUAL_QUOTA",
    "CASE_GEOMETRY",
    "CASE_ORDER",
    "CLAIM_BOUNDARY",
    "CONFIRMATION_EXAMPLE_COUNT",
    "CONFIRMATION_SUPPORT_FRACTION",
    "CONTEXTUAL_SIGNED_STATIC_METHOD",
    "CONTEXTUAL_UNSIGNED_MASS_METHOD",
    "DETERMINISTIC_BASELINE_ORDER",
    "DEVELOPMENT_ATTEMPT_PATH",
    "DEVELOPMENT_ATTEMPT_STATUS",
    "DEVELOPMENT_CASE_ORDER",
    "DEVELOPMENT_RUN_DIRECTORY",
    "DEVELOPMENT_SUMMARY_STATUS",
    "DEVELOPMENT_TERMINAL_FILENAMES",
    "DISCOVERY_ACQUISITION_EXAMPLE_COUNT",
    "END_TO_END_ALIGNMENT_GATES",
    "EXPERIMENT_ID",
    "GENERALIZATION_CASE_ORDER",
    "GROUP_INTERACTION_ADVANTAGE_MARGIN",
    "LOCAL_SEARCH_ROUNDS",
    "PRIMARY_GATES",
    "PRIMARY_METHOD",
    "RANDOM_CONTROL_95TH_INDEX",
    "RANDOM_CONTROL_NAMESPACE",
    "RANDOM_CONTROL_REPLICATES",
    "RANDOM_CONTROL_SEED_SHA256",
    "REMOVAL_DIRECTIONAL_QUOTA",
    "REMOVAL_RESIDUAL_QUOTA",
    "REPAIR_CLAIM_BOUNDARY",
    "SEARCH_EXAMPLE_COUNT",
    "SEARCH_ROW_BOUNDS",
    "SELECTOR_GATES",
    "SIGN_AWARE_ADVANTAGE_MARGIN",
    "VALIDATION_EXAMPLE_COUNT",
    "VALIDATION_ROW_BOUNDS",
    "CaseGeometry",
    "confirmation_support_size",
    "deterministic_random_supports",
    "maximum_exact_search_supports",
    "maximum_exact_validation_supports",
    "protocol_manifest",
    "selector_query_ceiling",
]
