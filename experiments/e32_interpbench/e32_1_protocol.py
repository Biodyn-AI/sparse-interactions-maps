# ruff: noqa: EXE002
"""Frozen, dependency-light protocol for the unsealed E32.1 experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from experiments.e32_interpbench.adapter import (
    CASE_MANIFESTS,
    IIT_REVISION,
    INTERPBENCH_CODE_REVISION,
    INTERPBENCH_MODEL_REVISION,
    TRACR_REVISION,
    patchable_internal_universe,
)

PRIMARY_CASE_ID = 97
SECONDARY_CASE_IDS = (2, 21)
CASE_ORDER = (PRIMARY_CASE_ID, *SECONDARY_CASE_IDS)
METHOD_ORDER = (
    "raw_pair_interaction",
    "singleton_endpoint_product",
    "rank_one_node_factor",
    "positive_pair_residual",
    "absolute_pair_residual",
)
COMPARATOR_METHODS = METHOD_ORDER[1:]

EXPECTED_INTERNAL_GROUND_TRUTH_COUNTS = {
    97: {"nodes": 6, "edges": 6},
    2: {"nodes": 5, "edges": 4},
    21: {"nodes": 4, "edges": 2},
}

# These are the exact official whole-head/MLP projections, not labels inferred
# from E32 outcomes.  Freezing the identities prevents a same-size fabricated
# support from passing the pure scorer with a copied artifact hash.
EXPECTED_INTERNAL_GROUND_TRUTH = {
    97: {
        "nodes": (
            "blocks.0.attn.hook_result[0]",
            "blocks.0.hook_mlp_out",
            "blocks.2.attn.hook_result[0]",
            "blocks.1.attn.hook_result[0]",
            "blocks.1.hook_mlp_out",
            "blocks.2.hook_mlp_out",
        ),
        "edges": (
            ("blocks.0.attn.hook_result[0]", "blocks.0.hook_mlp_out"),
            ("blocks.0.hook_mlp_out", "blocks.1.attn.hook_result[0]"),
            ("blocks.0.hook_mlp_out", "blocks.2.attn.hook_result[0]"),
            ("blocks.1.attn.hook_result[0]", "blocks.1.hook_mlp_out"),
            ("blocks.1.hook_mlp_out", "blocks.2.attn.hook_result[0]"),
            ("blocks.2.attn.hook_result[0]", "blocks.2.hook_mlp_out"),
        ),
    },
    2: {
        "nodes": (
            "blocks.0.attn.hook_result[0]",
            "blocks.3.attn.hook_result[0]",
            "blocks.1.hook_mlp_out",
            "blocks.0.hook_mlp_out",
            "blocks.2.hook_mlp_out",
        ),
        "edges": (
            ("blocks.0.attn.hook_result[0]", "blocks.0.hook_mlp_out"),
            ("blocks.0.hook_mlp_out", "blocks.1.hook_mlp_out"),
            ("blocks.1.hook_mlp_out", "blocks.2.hook_mlp_out"),
            ("blocks.2.hook_mlp_out", "blocks.3.attn.hook_result[0]"),
        ),
    },
    21: {
        "nodes": (
            "blocks.3.attn.hook_result[0]",
            "blocks.2.hook_mlp_out",
            "blocks.1.hook_mlp_out",
            "blocks.0.hook_mlp_out",
        ),
        "edges": (
            ("blocks.2.hook_mlp_out", "blocks.3.attn.hook_result[0]"),
            ("blocks.1.hook_mlp_out", "blocks.2.hook_mlp_out"),
        ),
    },
}

PRIMARY_GATE = {
    "minimum_confirmation_auprc_margin_over_each_comparator": 0.05,
    "minimum_additional_true_positives_at_6": 1,
    "minimum_discovery_confirmation_raw_energy_spearman": 0.50,
}

FULL_PANEL_GATE = {
    "minimum_case_wins_out_of_three": 2,
    "minimum_macro_auprc_margin_over_best_comparator": 0.05,
}


@dataclass(frozen=True)
class SplitRequest:
    """Everything an acquisition callback may use for one frozen split."""

    case_id: int
    split: Literal["discovery", "confirmation"]
    required_count: int
    candidate_count: int
    clean_seed: int
    corrupt_seed: int
    smoke_condition_count: int = 16
    smoke_condition_batch_size: int = 8
    atlas_condition_batch_size: int = 32


_SPLITS = {
    "discovery": {
        "required_count": 64,
        "candidate_count": 80,
        "clean_seed": 42_032,
        "corrupt_seed": 42_033,
    },
    "confirmation": {
        "required_count": 128,
        "candidate_count": 160,
        "clean_seed": 42_034,
        "corrupt_seed": 42_035,
    },
}


def split_request(case_id: int, split: str) -> SplitRequest:
    """Return one immutable request; no adaptive sampling is permitted."""
    if case_id not in CASE_ORDER:
        raise ValueError(f"case_id must be one of {CASE_ORDER!r}")
    try:
        values = _SPLITS[split]
    except KeyError as error:
        raise ValueError("split must be 'discovery' or 'confirmation'") from error
    return SplitRequest(case_id=case_id, split=split, **values)


def expected_case_provenance(case_id: int) -> dict[str, object]:
    """Return exact upstream source revisions and official artifact digests."""
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


def _case_plan(case_id: int) -> dict[str, object]:
    manifest = CASE_MANIFESTS[case_id]
    nodes, edges = patchable_internal_universe(manifest.n_layers, manifest.n_heads)
    condition_count = 1 + len(nodes) + len(edges)
    split_plans: dict[str, object] = {}
    for split in ("discovery", "confirmation"):
        request = split_request(case_id, split)
        # Preparation costs 2B, the smoke costs 2SB, and the exhaustive atlas
        # costs (N+E)B.  Eligibility costs two candidate-pool forwards.
        model_examples = 2 * request.candidate_count + request.required_count * (
            condition_count + 2 * request.smoke_condition_count + 1
        )
        split_plans[split] = {
            **asdict(request),
            "expected_model_examples_including_eligibility": model_examples,
        }
    return {
        "task": manifest.task,
        "role": "primary" if case_id == PRIMARY_CASE_ID else "secondary",
        "output_kind": manifest.output_kind,
        "n_layers": manifest.n_layers,
        "n_heads": manifest.n_heads,
        "n_ctx": manifest.n_ctx,
        "d_vocab_out": manifest.d_vocab_out,
        "internal_node_count": len(nodes),
        "directed_candidate_edge_count": len(edges),
        "condition_count": condition_count,
        "internal_ground_truth_counts": EXPECTED_INTERNAL_GROUND_TRUTH_COUNTS[case_id],
        "splits": split_plans,
        "exact_provenance": expected_case_provenance(case_id),
    }


def protocol_dry_run_manifest() -> dict[str, object]:
    """Describe the complete hierarchy without filesystem or model access."""
    cases = {str(case_id): _case_plan(case_id) for case_id in CASE_ORDER}
    primary_examples = sum(
        int(split["expected_model_examples_including_eligibility"])
        for split in cases[str(PRIMARY_CASE_ID)]["splits"].values()
    )
    all_examples = sum(
        int(split["expected_model_examples_including_eligibility"])
        for case in cases.values()
        for split in case["splits"].values()
    )
    return {
        "schema_version": "e32_1_exact_gt_protocol_v1",
        "status": "dry_run_only_no_model_calls_no_seal",
        "case_order": CASE_ORDER,
        "cases": cases,
        "hierarchy": {
            "primary_case": PRIMARY_CASE_ID,
            "secondary_cases": SECONDARY_CASE_IDS,
            "secondary_cases_run_only_if_primary_passes": True,
        },
        "method_order": METHOD_ORDER,
        "primary_gate": dict(PRIMARY_GATE),
        "full_panel_gate": dict(FULL_PANEL_GATE),
        "maximum_model_examples_if_primary_fails": primary_examples,
        "maximum_model_examples_if_all_cases_run": all_examples,
        "claims": {
            "target": "internal_component_node_and_directed_edge_localization",
            "official_truth_scope": "promoted_internal_projection_only",
            "pair_mask_is_symmetric": True,
            "edge_direction_is_imposed_by_transformer_topology": True,
            "pair_interaction_is_not_a_direct_communication_edge_estimator": True,
            "transitive_candidate_non_edges_reported_separately": True,
            "path_or_faithfulness_claims_in_e32_1": False,
            "superiority_to_eap_or_eap_ig_claim": False,
            "external_generalization_claim": False,
            "official_gt_and_case_identities_were_inspected_during_design": True,
            "hidden_or_untouched_benchmark_claim": False,
        },
    }


__all__ = [
    "CASE_ORDER",
    "COMPARATOR_METHODS",
    "EXPECTED_INTERNAL_GROUND_TRUTH",
    "EXPECTED_INTERNAL_GROUND_TRUTH_COUNTS",
    "FULL_PANEL_GATE",
    "METHOD_ORDER",
    "PRIMARY_CASE_ID",
    "PRIMARY_GATE",
    "SECONDARY_CASE_IDS",
    "SplitRequest",
    "expected_case_provenance",
    "protocol_dry_run_manifest",
    "split_request",
]
