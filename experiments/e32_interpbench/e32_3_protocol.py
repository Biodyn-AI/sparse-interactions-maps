# ruff: noqa: EXE002
"""Frozen, model-free protocol for E32.3 sparse-core faithfulness.

Ground-truth labels and counts in InterpBench are public and were known before
this protocol.  They informed only the cohort nontriviality/organism screen;
they are forbidden inputs to every within-case selector, support, mask,
threshold, primary metric, and verdict.  The prospective object is the
intervention response on the frozen discovery and confirmation panels.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

CASE_ORDER = (58, 129, 25)
MANDATORY_SPLITS = ("discovery", "confirmation")

SUPPORT_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)
CONFIRMATORY_SUPPORT_FRACTION = 0.50
SINGLETON_CURVE_FRACTIONS = (0.25, 0.50, 0.75)

PRIMARY_METHOD = "singleton_output_effect"
DETERMINISTIC_METHOD_ORDER = (
    PRIMARY_METHOD,
    "activation_shift_rms",
    "raw_pair_incident_mean",
    "rank_one_incident_mean",
    "eap_ig5_incident_mean",
)

RANDOM_CONTROL_REPLICATES = 64
RANDOM_CONTROL_NAMESPACE = "e32.3-sparse-core-random-support-v1"
RANDOM_CONTROL_SEED = 32_300_064

GROUND_TRUTH_DISCLOSURE = (
    "public_ground_truth_known_pre_run;used_only_for_cohort_nontriviality_organism_"
    "screen;forbidden_for_within_case_selector_support_mask_threshold_primary_"
    "metric_or_verdict"
)

SCREEN_GATES = {
    "maximum_induced_whole_edge_fraction": 0.30,
    "maximum_induced_fine_edge_fraction": 0.30,
    "minimum_confirmation_raw_mass_retention": 0.75,
    "minimum_confirmation_exact_mass_retention": 0.75,
    "maximum_discovery_confirmation_retention_gap": 0.10,
    "minimum_singleton_spearman": 0.80,
    "minimum_singleton_top_half_jaccard": 0.60,
    "minimum_raw_mass_spearman": 0.80,
    "minimum_exact_mass_spearman": 0.80,
}

FAITHFULNESS_GATES = {
    "minimum_necessity_projection": 0.75,
    "minimum_sufficiency_projection": 0.75,
    "maximum_necessity_residual_error": 0.25,
    "maximum_sufficiency_residual_error": 0.25,
    "minimum_necessity_loss_recovery": 0.75,
    "minimum_sufficiency_loss_recovery": 0.75,
}

SELECTOR_GATES = {
    "minimum_margin_over_random_95th_percentile": 0.05,
    "maximum_deficit_to_best_deterministic_baseline": 0.05,
}

EXACT_FINE_SETTINGS = {
    "scope": "all_internal_auto_circuit_fine_edges_both_splits",
    "necessity_orientation": "corrupt_source_edge_into_clean_recipient",
    "sufficiency_orientation": "clean_source_edge_into_corrupt_recipient",
    "mass_definition": (
        "half_sum_of_mean_squared_class_centered_non_bos_displacements"
    ),
    "qkv_policy": "retain_separate_fine_coordinates_without_max_promotion",
    "sign_policy": "symmetric_squared_magnitude_without_projection_relu_or_clipping",
}

SET_INTERVENTION_SETTINGS = {
    "recipient": "clean_input_computation",
    "donor": "corrupt_input_component_outputs",
    "necessity_mask": "selected_support_S",
    "keep_core_sufficiency_mask": "component_complement_U_minus_S",
    "anchor_masks": ("empty", "all_components_U"),
    "clean_into_corrupt_set_mask_status": "not_acquired_not_required",
}

ZERO_MASS_POLICY = (
    "any_zero_total_discovery_or_confirmation_raw_or_exact_mass_is_"
    "scientifically_noninformative_and_fails_the_screen_without_crashing"
)

EXPECTED_UNIVERSE_COUNTS = {
    58: {"layers": 3, "heads": 4, "nodes": 15, "whole_edges": 87, "fine_edges": 207},
    129: {"layers": 3, "heads": 4, "nodes": 15, "whole_edges": 87, "fine_edges": 207},
    25: {"layers": 2, "heads": 4, "nodes": 10, "whole_edges": 33, "fine_edges": 73},
}

EXPECTED_QUERY_ACCOUNTING = {
    58: {
        "discovery": {
            "physical_forwards": 444,
            "model_examples": 35_200,
            "backward_passes_inferred_from_pinned_control_flow": 6,
            "gradient_examples": 384,
        },
        "confirmation": {
            "physical_forwards": 440,
            "model_examples": 86_656,
            "backward_passes_inferred_from_pinned_control_flow": 6,
            "gradient_examples": 768,
        },
    },
    129: {
        "discovery": {
            "physical_forwards": 444,
            "model_examples": 35_200,
            "backward_passes_inferred_from_pinned_control_flow": 6,
            "gradient_examples": 384,
        },
        "confirmation": {
            "physical_forwards": 440,
            "model_examples": 86_656,
            "backward_passes_inferred_from_pinned_control_flow": 6,
            "gradient_examples": 768,
        },
    },
    25: {
        "discovery": {
            "physical_forwards": 174,
            "model_examples": 14_272,
            "backward_passes_inferred_from_pinned_control_flow": 6,
            "gradient_examples": 384,
        },
        "confirmation": {
            "physical_forwards": 170,
            "model_examples": 44_800,
            "backward_passes_inferred_from_pinned_control_flow": 6,
            "gradient_examples": 768,
        },
    },
    "overall": {
        "physical_forwards": 2_112,
        "model_examples": 302_784,
        "backward_passes_inferred_from_pinned_control_flow": 36,
        "gradient_examples": 3_456,
    },
}


@dataclass(frozen=True)
class SplitRequest:
    """One immutable, response-prospective input request."""

    case_id: int
    split: Literal["discovery", "confirmation"]
    required_count: int
    candidate_count: int
    clean_seed: int
    corrupt_seed: int
    excluded_smoke_clean_seed: int
    excluded_smoke_corrupt_seed: int
    excluded_smoke_candidate_count_per_role: int = 96
    condition_batch_size: int = 32
    ground_truth_disclosure: str = GROUND_TRUTH_DISCLOSURE
    freshness_policy: str = (
        "exclude_all_smoke_clean_and_corrupt_rows_and_require_every_selected_"
        "clean_and_corrupt_row_globally_unique_across_discovery_and_confirmation_"
        "within_case"
    )


_CASE_SEEDS = {
    58: {
        "smoke": (758_000, 758_001),
        "discovery": (758_100, 758_101),
        "confirmation": (758_200, 758_201),
    },
    129: {
        "smoke": (829_000, 829_001),
        "discovery": (829_100, 829_101),
        "confirmation": (829_200, 829_201),
    },
    25: {
        "smoke": (725_000, 725_001),
        "discovery": (725_100, 725_101),
        "confirmation": (725_200, 725_201),
    },
}


def split_request(case_id: int, split: str) -> SplitRequest:
    """Return the exact frozen request; adaptive panel growth is forbidden."""
    if case_id not in CASE_ORDER:
        raise ValueError(f"case_id must be one of {CASE_ORDER!r}")
    if split not in MANDATORY_SPLITS:
        raise ValueError(f"split must be one of {MANDATORY_SPLITS!r}")
    required_count, candidate_count = (64, 96) if split == "discovery" else (128, 192)
    clean_seed, corrupt_seed = _CASE_SEEDS[case_id][split]
    smoke_clean, smoke_corrupt = _CASE_SEEDS[case_id]["smoke"]
    return SplitRequest(
        case_id=case_id,
        split=split,  # type: ignore[arg-type]
        required_count=required_count,
        candidate_count=candidate_count,
        clean_seed=clean_seed,
        corrupt_seed=corrupt_seed,
        excluded_smoke_clean_seed=smoke_clean,
        excluded_smoke_corrupt_seed=smoke_corrupt,
    )


def support_sizes(node_count: int) -> tuple[int, ...]:
    """Convert frozen fractional budgets to unique ceiling-rounded prefixes."""
    if (
        isinstance(node_count, bool)
        or not isinstance(node_count, int)
        or node_count < 1
    ):
        raise ValueError("node_count must be a positive integer")
    sizes = tuple(math.ceil(fraction * node_count) for fraction in SUPPORT_FRACTIONS)
    if len(sizes) != len(set(sizes)):
        raise ValueError("node_count is too small for the frozen support fractions")
    return sizes


def confirmatory_support_size(node_count: int) -> int:
    """Return the sole confirmatory support budget."""
    return math.ceil(CONFIRMATORY_SUPPORT_FRACTION * node_count)


def deterministic_random_support(
    *,
    case_id: int,
    component_ids: Sequence[str],
    support_size: int,
    replicate: int,
) -> tuple[str, ...]:
    """Return one response-blind SHA-256 random-control support.

    Nodes are sorted by ``SHA256(namespace|case|k|replicate|canonical-index|id)``;
    the first ``k`` are returned in canonical component order.  This is a
    deterministic pseudo-random control, not a source of inferential p-values.
    """
    if case_id not in CASE_ORDER:
        raise ValueError(f"case_id must be one of {CASE_ORDER!r}")
    nodes = tuple(component_ids)
    if (
        not nodes
        or len(nodes) != len(set(nodes))
        or any(not isinstance(node, str) or not node for node in nodes)
    ):
        raise ValueError("component_ids must be unique nonempty strings")
    if (
        isinstance(support_size, bool)
        or not isinstance(support_size, int)
        or not 0 <= support_size <= len(nodes)
    ):
        raise ValueError("support_size must lie within the component universe")
    if (
        isinstance(replicate, bool)
        or not isinstance(replicate, int)
        or not 0 <= replicate < RANDOM_CONTROL_REPLICATES
    ):
        raise ValueError(f"replicate must lie in [0, {RANDOM_CONTROL_REPLICATES})")
    keyed = []
    for index, node in enumerate(nodes):
        payload = (
            f"{RANDOM_CONTROL_NAMESPACE}|seed={RANDOM_CONTROL_SEED}|"
            f"case={case_id}|k={support_size}|"
            f"replicate={replicate}|index={index}|node={node}"
        ).encode()
        keyed.append((hashlib.sha256(payload).hexdigest(), index))
    chosen = {index for _digest, index in sorted(keyed)[:support_size]}
    return tuple(node for index, node in enumerate(nodes) if index in chosen)


def protocol_manifest() -> dict[str, object]:
    """Return a JSON-compatible registration core without reading outcomes."""
    return {
        "schema_version": "e32_3_sparse_core_protocol_v1",
        "case_order": CASE_ORDER,
        "mandatory_all_cases": True,
        "ground_truth_disclosure": GROUND_TRUTH_DISCLOSURE,
        "ground_truth_used_for_cohort_selection": True,
        "ground_truth_used_for_within_case_selection_or_primary_scoring": False,
        "support_fractions": SUPPORT_FRACTIONS,
        "confirmatory_support_fraction": CONFIRMATORY_SUPPORT_FRACTION,
        "singleton_curve_fractions": SINGLETON_CURVE_FRACTIONS,
        "deterministic_method_order": DETERMINISTIC_METHOD_ORDER,
        "random_control": {
            "replicates": RANDOM_CONTROL_REPLICATES,
            "namespace": RANDOM_CONTROL_NAMESPACE,
            "seed": RANDOM_CONTROL_SEED,
        },
        "screen_gates": dict(SCREEN_GATES),
        "faithfulness_gates": dict(FAITHFULNESS_GATES),
        "selector_gates": dict(SELECTOR_GATES),
        "exact_fine_settings": dict(EXACT_FINE_SETTINGS),
        "set_intervention_settings": dict(SET_INTERVENTION_SETTINGS),
        "zero_mass_policy": ZERO_MASS_POLICY,
        "expected_universe_counts": {
            str(case_id): dict(counts)
            for case_id, counts in EXPECTED_UNIVERSE_COUNTS.items()
        },
        "expected_query_accounting": {
            str(key): value for key, value in EXPECTED_QUERY_ACCOUNTING.items()
        },
        "split_requests": {
            str(case_id): {
                split: asdict(split_request(case_id, split))
                for split in MANDATORY_SPLITS
            }
            for case_id in CASE_ORDER
        },
        "multiplicity_policy": (
            "one_confirmatory_budget_all_endpoints_and_all_cases_conjunctive;"
            "best_deterministic_comparator_collapsed_before_one_noninferiority_check"
        ),
        "claim_boundary": (
            "stable_faithful_sparse_component_induced_causal_core_or_reachability_"
            "region_not_direct_wiring_unique_or_minimal_circuit_real_llm_or_tensor_"
            "network_generalization"
        ),
    }


__all__ = [
    "CASE_ORDER",
    "CONFIRMATORY_SUPPORT_FRACTION",
    "DETERMINISTIC_METHOD_ORDER",
    "EXACT_FINE_SETTINGS",
    "EXPECTED_QUERY_ACCOUNTING",
    "EXPECTED_UNIVERSE_COUNTS",
    "FAITHFULNESS_GATES",
    "GROUND_TRUTH_DISCLOSURE",
    "PRIMARY_METHOD",
    "RANDOM_CONTROL_NAMESPACE",
    "RANDOM_CONTROL_REPLICATES",
    "RANDOM_CONTROL_SEED",
    "SCREEN_GATES",
    "SELECTOR_GATES",
    "SET_INTERVENTION_SETTINGS",
    "SINGLETON_CURVE_FRACTIONS",
    "SUPPORT_FRACTIONS",
    "ZERO_MASS_POLICY",
    "SplitRequest",
    "confirmatory_support_size",
    "deterministic_random_support",
    "protocol_manifest",
    "split_request",
    "support_sizes",
]
