# ruff: noqa: EXE002
"""Frozen, model-free protocol for E32.2 node-first directed validation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

from experiments.e32_interpbench.adapter import (
    CASE_MANIFESTS,
    IIT_REVISION,
    INTERPBENCH_CODE_REVISION,
    INTERPBENCH_MODEL_REVISION,
    TRACR_REVISION,
)

DEVELOPMENT_CASE_ID = 97
PRIMARY_CASE_ID = 2
REPLICATION_CASE_ID = 21
PROSPECTIVE_CASE_IDS = (PRIMARY_CASE_ID, REPLICATION_CASE_ID)
# The canonical run contains only the two mandatory prospective organisms.
# Case 97 is documented as development ancestry and is never acquired by E32.2.
CASE_ORDER = PROSPECTIVE_CASE_IDS

FULL_UNIVERSE_FLOOR = -1.0

METHOD_ORDER = (
    "bidirectional_fine_replacement",
    "exhaustive_bidirectional_fine_replacement",
    "eap_signed_max",
    "eap_absolute_max",
    "eap_ig_signed_max",
    "eap_ig_absolute_max",
    "raw_pair_interaction",
    "rank_one_node_factor",
    "singleton_endpoint_product",
)
COMPARATOR_METHODS = METHOD_ORDER[1:]
EAP_VARIANTS = METHOD_ORDER[2:6]

SCREEN_GATES = {
    "minimum_direct_edge_retention": 1.0,
    "maximum_induced_whole_edge_fraction": 0.30,
    "maximum_induced_fine_edge_fraction": 0.30,
    "minimum_operational_model_example_savings": 0.60,
}

USEFUL_GATES = {
    "minimum_confirmation_auprc": 0.90,
    "require_perfect_true_positives_at_gt_k": True,
    "maximum_auprc_deficit_to_exhaustive_exact": 0.02,
    "maximum_auprc_deficit_to_best_eap_variant": 0.02,
    "minimum_auprc_margin_over_raw_pair": 0.10,
    "minimum_direct_vs_transitive_auroc_margin_over_node_gated_raw": 0.10,
    "minimum_induced_discovery_confirmation_spearman": 0.50,
}

SUPERIORITY_GATES = {
    "minimum_auprc_margin_over_every_comparator": 0.05,
    "minimum_true_positive_improvement": 1,
    "true_positive_improvement_capped_at_perfect": True,
}

# Fine-edge counts use AutoCircuit's separate-Q/K/V graph.  The shortlist
# ranges are exact over every top-half subset of the frozen whole-node DAG, not
# estimates from model outcomes.
EXPECTED_UNIVERSE_COUNTS: dict[int, dict[str, int]] = {
    2: {
        "nodes": 20,
        "whole_edges": 166,
        "fine_edges": 406,
        "ground_truth_nodes": 5,
        "ground_truth_edges": 4,
        "shortlist_nodes": 10,
        "minimum_induced_whole_edges": 32,
        "maximum_induced_whole_edges": 43,
        "minimum_induced_fine_edges": 58,
        "maximum_induced_fine_edges": 113,
    },
    21: {
        "nodes": 20,
        "whole_edges": 166,
        "fine_edges": 406,
        "ground_truth_nodes": 4,
        "ground_truth_edges": 2,
        "shortlist_nodes": 10,
        "minimum_induced_whole_edges": 32,
        "maximum_induced_whole_edges": 43,
        "minimum_induced_fine_edges": 58,
        "maximum_induced_fine_edges": 113,
    },
    97: {
        "nodes": 15,
        "whole_edges": 87,
        "fine_edges": 207,
        "ground_truth_nodes": 6,
        "ground_truth_edges": 6,
        "shortlist_nodes": 8,
        "minimum_induced_whole_edges": 16,
        "maximum_induced_whole_edges": 26,
        "minimum_induced_fine_edges": 32,
        "maximum_induced_fine_edges": 65,
    },
}


@dataclass(frozen=True)
class SplitRequest:
    """Response-blind input request for one frozen case and split."""

    case_id: int
    split: Literal["discovery", "confirmation"]
    required_count: int
    candidate_count: int
    clean_seed: int
    corrupt_seed: int
    smoke_condition_count: int = 16
    smoke_condition_batch_size: int = 8
    atlas_condition_batch_size: int = 32


_SPLITS: dict[int, dict[str, dict[str, int]]] = {
    2: {
        "discovery": {
            "required_count": 64,
            "candidate_count": 80,
            "clean_seed": 52_032,
            "corrupt_seed": 52_033,
        },
        "confirmation": {
            "required_count": 128,
            "candidate_count": 160,
            "clean_seed": 52_034,
            "corrupt_seed": 52_035,
        },
    },
    21: {
        "discovery": {
            "required_count": 64,
            "candidate_count": 80,
            "clean_seed": 62_032,
            "corrupt_seed": 62_033,
        },
        "confirmation": {
            "required_count": 128,
            "candidate_count": 160,
            # 62034/62035 produce only 159 official unique rows.  This
            # response-blind replacement produces all 160 and was frozen
            # before any case-21 model outcome was opened.
            "clean_seed": 62_040,
            "corrupt_seed": 62_041,
        },
    },
}


def split_request(case_id: int, split: str) -> SplitRequest:
    """Return an immutable request; no adaptive panel growth is allowed."""
    if case_id not in CASE_ORDER:
        raise ValueError(f"case_id must be one of {CASE_ORDER!r}")
    try:
        values = _SPLITS[case_id][split]
    except KeyError as error:
        raise ValueError("split must be 'discovery' or 'confirmation'") from error
    return SplitRequest(case_id=case_id, split=split, **values)


def expected_case_provenance(case_id: int) -> dict[str, object]:
    """Return pinned upstream revisions and official artifact digests."""
    if case_id not in CASE_ORDER:
        raise ValueError(f"case_id must be one of {CASE_ORDER!r}")
    return {
        "source_revisions": {
            "interpbench": INTERPBENCH_CODE_REVISION,
            "tracr": TRACR_REVISION,
            "iit": IIT_REVISION,
        },
        "model_revision": INTERPBENCH_MODEL_REVISION,
        "artifact_sha256": dict(CASE_MANIFESTS[case_id].sha256),
    }


def _validated_prospective_fine_count(
    case_id: int, induced_fine_edge_count: int
) -> tuple[dict[str, int], int]:
    if case_id not in PROSPECTIVE_CASE_IDS:
        raise ValueError(f"case_id must be one of {PROSPECTIVE_CASE_IDS!r}")
    if isinstance(induced_fine_edge_count, bool) or not isinstance(
        induced_fine_edge_count, int
    ):
        raise TypeError("induced_fine_edge_count must be an integer")
    counts = EXPECTED_UNIVERSE_COUNTS[case_id]
    lower = counts["minimum_induced_fine_edges"]
    upper = counts["maximum_induced_fine_edges"]
    if not lower <= induced_fine_edge_count <= upper:
        raise ValueError(
            f"induced_fine_edge_count must lie in the exact [{lower}, {upper}] range"
        )
    return counts, induced_fine_edge_count


def operational_query_accounting(
    case_id: int, induced_fine_edge_count: int
) -> dict[str, int | float | bool | str]:
    """Counterfactual cost of running only the node-first directed cascade.

    The executable uses fresh component-patch and directed models.  Each split
    therefore pays two component-model base forwards, one forward per
    singleton, two directed-model source-cache forwards, two directed-model
    baseline forwards, and two replacements per induced fine edge.  The
    exhaustive reference omits singleton discovery and replaces every internal
    fine edge in both directions.  Behavioral eligibility is included equally.
    """
    counts, fine_count = _validated_prospective_fine_count(
        case_id, induced_fine_edge_count
    )
    discovery = split_request(case_id, "discovery")
    confirmation = split_request(case_id, "confirmation")
    panel_examples = discovery.required_count + confirmation.required_count
    eligibility_examples = 2 * (
        discovery.candidate_count + confirmation.candidate_count
    )

    operational_examples = (
        counts["nodes"] + 6 + 2 * fine_count
    ) * panel_examples + eligibility_examples
    exhaustive_examples = (
        4 + 2 * counts["fine_edges"]
    ) * panel_examples + eligibility_examples
    operational_forwards = 2 * (
        2  # eligibility clean/source candidate-pool forwards
        + 2  # clean/corrupt base outputs
        + math.ceil(counts["nodes"] / discovery.atlas_condition_batch_size)
        + 2  # clean/corrupt AutoCircuit source caches
        + 2  # fresh directed-model clean/corrupt baselines
        + 2 * fine_count
    )
    exhaustive_forwards = 2 * (
        2  # eligibility
        + 2  # clean/corrupt base outputs
        + 2  # clean/corrupt source caches
        + 2 * counts["fine_edges"]
    )
    return {
        "case_id": case_id,
        "induced_fine_edge_count": fine_count,
        "panel_model_examples_including_eligibility": operational_examples,
        "panel_physical_forwards_including_eligibility": operational_forwards,
        "exhaustive_exact_model_examples_including_eligibility": (exhaustive_examples),
        "exhaustive_exact_physical_forwards_including_eligibility": (
            exhaustive_forwards
        ),
        "model_example_savings_fraction": 1.0
        - operational_examples / exhaustive_examples,
        "claim_scope": (
            "counterfactual_operational_cascade_vs_exhaustive_fine_replacement"
        ),
        "includes_full_raw_pair_atlas": False,
        "includes_eap_comparators": False,
    }


def validation_query_accounting(
    case_id: int, induced_fine_edge_count: int
) -> dict[str, int | bool | str]:
    """Actual frozen benchmark work, kept separate from operational cost.

    D evaluates exact replacement only on induced fine edges.  C evaluates all
    fine edges, from which the frozen cascade subset is extracted.  Both splits
    also acquire a full raw-pair atlas and run official EAP plus EAP-IG.
    """
    counts, fine_count = _validated_prospective_fine_count(
        case_id, induced_fine_edge_count
    )
    discovery = split_request(case_id, "discovery")
    confirmation = split_request(case_id, "confirmation")
    node_count = counts["nodes"]
    whole_edge_count = counts["whole_edges"]
    full_fine_count = counts["fine_edges"]
    condition_batch_size = discovery.atlas_condition_batch_size

    # Full raw atlas: two base forwards, a mandatory 16-condition batch smoke
    # evaluated once at batch size 1 and once at batch size 8, then N+E patched
    # scientific conditions.  This matches the executable E32.1 runner exactly.
    raw_model_coefficient = (
        node_count + whole_edge_count + 2 + 2 * discovery.smoke_condition_count
    )
    raw_physical_forwards = (
        2
        + discovery.smoke_condition_count
        + math.ceil(
            discovery.smoke_condition_count / discovery.smoke_condition_batch_size
        )
        + math.ceil((node_count + whole_edge_count) / condition_batch_size)
    )
    # EAP is two model forwards; five-step EAP-IG is seven.
    eap_model_coefficient = 9
    eap_physical_forwards = 9

    discovery_model_coefficient = (
        raw_model_coefficient + (4 + 2 * fine_count) + eap_model_coefficient
    )
    confirmation_model_coefficient = (
        raw_model_coefficient + (4 + 2 * full_fine_count) + eap_model_coefficient
    )
    eligibility_examples = 2 * (
        discovery.candidate_count + confirmation.candidate_count
    )
    actual_examples = (
        discovery.required_count * discovery_model_coefficient
        + confirmation.required_count * confirmation_model_coefficient
        + eligibility_examples
    )

    discovery_forwards = (
        2  # eligibility
        + raw_physical_forwards
        + 4
        + 2 * fine_count
        + eap_physical_forwards
    )
    confirmation_forwards = (
        2  # eligibility
        + raw_physical_forwards
        + 4
        + 2 * full_fine_count
        + eap_physical_forwards
    )
    return {
        "case_id": case_id,
        "induced_fine_edge_count": fine_count,
        "actual_model_examples_including_eligibility": actual_examples,
        "actual_physical_forwards_including_eligibility": (
            discovery_forwards + confirmation_forwards
        ),
        "discovery_model_examples_including_eligibility": (
            discovery.required_count * discovery_model_coefficient
            + 2 * discovery.candidate_count
        ),
        "confirmation_model_examples_including_eligibility": (
            confirmation.required_count * confirmation_model_coefficient
            + 2 * confirmation.candidate_count
        ),
        "gradient_examples": 7
        * (discovery.required_count + confirmation.required_count),
        "backward_passes": 14,
        "discovery_exact_scope": "induced_fine_edges",
        "confirmation_exact_scope": f"all_{full_fine_count}_internal_fine_edges",
        "includes_full_raw_pair_atlas_both_splits": True,
        "includes_eap_and_eap_ig_both_splits": True,
        "includes_batch_invariance_smoke": True,
    }


def _case_manifest(case_id: int) -> dict[str, object]:
    manifest = CASE_MANIFESTS[case_id]
    return {
        "task": manifest.task,
        "role": (
            "development_seen"
            if case_id == DEVELOPMENT_CASE_ID
            else "prospective_primary"
            if case_id == PRIMARY_CASE_ID
            else "prospective_mandatory_replication"
        ),
        "output_kind": manifest.output_kind,
        "n_ctx": manifest.n_ctx,
        "d_vocab_out": manifest.d_vocab_out,
        "universe_counts": dict(EXPECTED_UNIVERSE_COUNTS[case_id]),
        "splits": {
            split: asdict(split_request(case_id, split))
            for split in ("discovery", "confirmation")
        },
        "exact_provenance": expected_case_provenance(case_id),
    }


def protocol_dry_run_manifest() -> dict[str, object]:
    """Return the complete model-free registration payload."""
    return {
        "schema_version": "e32_2_node_first_directed_validation_v1",
        "status": "dry_run_only_no_model_calls_no_seal",
        "case_order": CASE_ORDER,
        "prospective_case_ids": PROSPECTIVE_CASE_IDS,
        "cases": {str(case_id): _case_manifest(case_id) for case_id in CASE_ORDER},
        "development_ancestry": {
            "case_id": DEVELOPMENT_CASE_ID,
            "executed_in_e32_2": False,
            "universe_counts": dict(EXPECTED_UNIVERSE_COUNTS[DEVELOPMENT_CASE_ID]),
            "source": "sealed_e32_1_results_and_eap_feasibility_smoke",
        },
        "hierarchy": {
            "primary_case": PRIMARY_CASE_ID,
            "mandatory_replication_case": REPLICATION_CASE_ID,
            "development_case": DEVELOPMENT_CASE_ID,
            "replication_runs_regardless_of_primary_result": True,
            "overall_pass_requires_both_prospective_cases": True,
        },
        "node_shortlist": {
            "score": "mean_square_singleton_effect",
            "selection_split": "discovery_only",
            "confirmation_may_not_change_shortlist": True,
            "size_rule": "top_half_of_exact_whole_node_universe",
            "tie_break": "frozen_node_universe_order",
        },
        "primary_score": {
            "granularity": "separate_auto_circuit_fine_qkv_or_mlp_edge",
            "directions": (
                "clean_source_into_corrupt_base_sufficiency",
                "corrupt_source_into_clean_base_necessity",
            ),
            "context_normalization": "squared_clean_corrupt_logit_gap",
            "fine_combination": "sqrt(relu(S_cf)*relu(N_cf))_then_context_mean",
            "whole_edge_promotion": "maximum_over_fine_coordinates",
            "outside_shortlist_score": FULL_UNIVERSE_FLOOR,
        },
        "method_order": METHOD_ORDER,
        "eap_comparator_policy": (
            "best_confirmation_auprc_across_signed_and_absolute_max_variants; "
            "no_selection_advantage_claim"
        ),
        "screen_gates": dict(SCREEN_GATES),
        "useful_gates": dict(USEFUL_GATES),
        "superiority_gates": dict(SUPERIORITY_GATES),
        "actual_validation_lane": {
            "discovery_exact": "induced_fine_edges",
            "confirmation_exact": "exhaustive_internal_fine_edges",
            "raw_pair_atlas": "exhaustive_both_splits",
            "eap_and_eap_ig": "exhaustive_both_splits",
            "operational_cost_reported_separately": True,
        },
        "claims": {
            "cases_and_gt_were_inspected": True,
            "cases_2_and_21_outcomes_unseen_at_registration": True,
            "hidden_or_untouched_benchmark": False,
            "target": "promoted_internal_whole_component_direct_edges",
            "pair_j_target": "causal_reachability_not_direct_edges",
            "cheaper_than_exhaustive_eap": False,
            "cost_comparator": "exhaustive_bidirectional_fine_edge_replacement",
            "faithfulness_claim_in_e32_2": False,
            "external_generalization_claim": False,
            "case97_confirmatory_evidence": False,
        },
    }


__all__ = [
    "CASE_ORDER",
    "COMPARATOR_METHODS",
    "DEVELOPMENT_CASE_ID",
    "EAP_VARIANTS",
    "EXPECTED_UNIVERSE_COUNTS",
    "FULL_UNIVERSE_FLOOR",
    "METHOD_ORDER",
    "PRIMARY_CASE_ID",
    "PROSPECTIVE_CASE_IDS",
    "REPLICATION_CASE_ID",
    "SCREEN_GATES",
    "SUPERIORITY_GATES",
    "USEFUL_GATES",
    "SplitRequest",
    "expected_case_provenance",
    "operational_query_accounting",
    "protocol_dry_run_manifest",
    "split_request",
    "validation_query_accounting",
]
