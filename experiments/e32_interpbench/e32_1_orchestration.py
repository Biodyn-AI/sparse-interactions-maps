# ruff: noqa: EXE002
"""Pure evaluation and hierarchical control flow for E32.1.

Model loading, behavioral screening, and filesystem custody deliberately live
outside this module.  The functions here accept already acquired, immutable
split artifacts and return JSON-compatible scientific summaries.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from experiments.e32_interpbench.adapter import (
    CASE_MANIFESTS,
    patchable_internal_universe,
)
from experiments.e32_interpbench.analysis import (
    build_atlas_method_scores,
    edge_sparsity_summary,
    transitive_candidate_non_edges,
)
from experiments.e32_interpbench.component_patching import (
    build_component_atlas,
    build_patch_units,
    class_center_logits,
    vector_pair_interactions,
)
from experiments.e32_interpbench.e32_1_protocol import (
    CASE_ORDER,
    COMPARATOR_METHODS,
    EXPECTED_INTERNAL_GROUND_TRUTH,
    EXPECTED_INTERNAL_GROUND_TRUTH_COUNTS,
    FULL_PANEL_GATE,
    METHOD_ORDER,
    PRIMARY_CASE_ID,
    PRIMARY_GATE,
    SECONDARY_CASE_IDS,
    SplitRequest,
    expected_case_provenance,
    split_request,
)
from experiments.e32_interpbench.protocol import AdaptedScores
from experiments.e32_interpbench.scoring import evaluate_localization


@dataclass(frozen=True)
class AcquiredSplit:
    """One frozen split plus the exact upstream identity used to acquire it."""

    request: SplitRequest
    atlas: dict[str, Any]
    exact_provenance: dict[str, object]


@dataclass(frozen=True)
class ExactGroundTruth:
    """The official promoted internal support for one organism."""

    case_id: int
    node_ids: tuple[str, ...]
    directed_edges: tuple[tuple[str, str], ...]
    raw_edges_sha256: str


def _token_rows(value: Any, *, label: str) -> tuple[tuple[int, ...], ...]:
    tensor = torch.as_tensor(value, device="cpu")
    if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] < 1:
        raise ValueError(f"{label} must be a nonempty rank-two token tensor")
    if tensor.dtype == torch.bool or tensor.is_floating_point():
        raise ValueError(f"{label} must contain integer token IDs")
    return tuple(tuple(int(token) for token in row) for row in tensor.tolist())


def assert_discovery_confirmation_disjoint(
    discovery: AcquiredSplit, confirmation: AcquiredSplit
) -> dict[str, int | str]:
    """Require all selected clean/source token rows to be unique and disjoint."""
    if discovery.request.case_id != confirmation.request.case_id:
        raise ValueError("discovery and confirmation case IDs differ")
    if discovery.request.split != "discovery" or confirmation.request.split != (
        "confirmation"
    ):
        raise ValueError("split roles must be discovery then confirmation")

    discovery_rows = _token_rows(
        discovery.atlas.get("clean_inputs"), label="discovery clean inputs"
    ) + _token_rows(
        discovery.atlas.get("corrupt_inputs"), label="discovery source inputs"
    )
    confirmation_rows = _token_rows(
        confirmation.atlas.get("clean_inputs"), label="confirmation clean inputs"
    ) + _token_rows(
        confirmation.atlas.get("corrupt_inputs"), label="confirmation source inputs"
    )
    discovery_set = set(discovery_rows)
    confirmation_set = set(confirmation_rows)
    overlap = discovery_set.intersection(confirmation_set)
    if (
        len(discovery_set) != len(discovery_rows)
        or len(confirmation_set) != len(confirmation_rows)
        or overlap
    ):
        raise RuntimeError(
            "token-row leakage: each role and split must contain unique rows"
        )
    return {
        "policy": "all_clean_and_source_token_rows_unique_within_and_across_splits",
        "discovery_unique_token_rows": len(discovery_set),
        "confirmation_unique_token_rows": len(confirmation_set),
        "overlap_count": 0,
    }


def _sha256_tensor_rows(value: Any) -> str:
    tensor = torch.as_tensor(value, device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _validate_split(split: AcquiredSplit) -> None:
    request = split.request
    if request != split_request(request.case_id, request.split):
        raise ValueError("split request differs from the frozen protocol")
    if split.exact_provenance != expected_case_provenance(request.case_id):
        raise ValueError("split provenance differs from the frozen protocol")
    atlas = split.atlas
    if atlas.get("schema_version") != "e32_interpbench_component_atlas_v2":
        raise ValueError("atlas schema changed")
    if atlas.get("case_id") != request.case_id:
        raise ValueError("atlas case ID changed")
    if atlas.get("scientific_output_coordinate") != (
        "class_centered_logits_non_bos_positions"
    ):
        raise ValueError("atlas scientific coordinate changed")
    if atlas.get("pair_patch_semantics") != (
        "symmetric_two_component_mask; edge_arrow_is_topology_imposed"
    ):
        raise ValueError("atlas pair semantics changed")
    if not bool(atlas.get("behavioral_pair_screen", {}).get("all_pairs_correct")):
        raise ValueError("atlas contains a behaviorally ineligible selected row")
    if not bool(atlas.get("batch_invariance_smoke", {}).get("passed")):
        raise ValueError("atlas batch smoke did not pass")
    for key in ("clean_inputs", "corrupt_inputs"):
        tensor = torch.as_tensor(atlas.get(key), device="cpu")
        rows = _token_rows(tensor, label=key)
        if len(rows) != request.required_count:
            raise ValueError(f"{key} row count changed")
        if tuple(tensor.shape) != (
            request.required_count,
            CASE_MANIFESTS[request.case_id].n_ctx,
        ):
            raise ValueError(f"{key} position count changed")

    eligibility_metadata = atlas.get("behavioral_eligibility", {})
    expected_excluded_rows = (
        0
        if request.split == "discovery"
        else 2 * split_request(request.case_id, "discovery").required_count
    )
    if (
        eligibility_metadata.get("output_kind") != "categorical"
        or eligibility_metadata.get("selection_policy")
        != (
            "first_jointly_correct_with_globally_unique_clean_source_rows_and_"
            "response_blind_exclusions"
        )
        or eligibility_metadata.get("selected_count") != request.required_count
        or eligibility_metadata.get("excluded_token_row_count")
        != expected_excluded_rows
    ):
        raise ValueError("behavioral eligibility metadata changed")

    nodes = tuple(atlas.get("component_ids", ()))
    edges = tuple(tuple(edge) for edge in atlas.get("directed_candidate_edges", ()))
    case_manifest = CASE_MANIFESTS[request.case_id]
    expected_nodes, expected_edges = patchable_internal_universe(
        case_manifest.n_layers, case_manifest.n_heads
    )
    if nodes != expected_nodes or edges != expected_edges:
        raise ValueError("atlas node/edge universe or order changed")
    expected_atlas = build_component_atlas(
        build_patch_units(case_manifest.n_layers, case_manifest.n_heads),
        valid_directed_edges=expected_edges,
    )
    if (
        tuple(atlas.get("condition_ids", ())) != expected_atlas.condition_ids
        or not torch.equal(
            torch.as_tensor(atlas.get("condition_masks")),
            expected_atlas.condition_masks,
        )
        or tuple(atlas.get("pair_component_indices", ()))
        != expected_atlas.pair_component_indices
    ):
        raise ValueError("atlas condition manifest changed")
    scientific = torch.as_tensor(atlas.get("scientific_condition_outputs"))
    raw_conditions = torch.as_tensor(atlas.get("condition_outputs"))
    singletons = torch.as_tensor(atlas.get("scientific_singleton_outputs"))
    pairs = torch.as_tensor(atlas.get("scientific_pair_outputs"))
    interactions = torch.as_tensor(atlas.get("pair_interactions"))
    condition_count = 1 + len(nodes) + len(edges)
    expected_response_shape = (
        request.required_count,
        case_manifest.n_ctx - 1,
        case_manifest.d_vocab_out,
    )
    if scientific.ndim != 4 or tuple(scientific.shape) != (
        condition_count,
        *expected_response_shape,
    ):
        raise ValueError("scientific condition-output shape changed")
    if tuple(raw_conditions.shape) != (
        condition_count,
        request.required_count,
        case_manifest.n_ctx,
        case_manifest.d_vocab_out,
    ):
        raise ValueError("raw condition-output shape changed")
    if not torch.equal(
        scientific,
        class_center_logits(raw_conditions)[..., 1:, :],
    ):
        raise ValueError("scientific class-centering/non-BOS projection changed")
    if tuple(singletons.shape) != (len(nodes), *expected_response_shape):
        raise ValueError("singleton-output shape changed")
    if tuple(pairs.shape) != (len(edges), *expected_response_shape):
        raise ValueError("pair-output shape changed")
    if tuple(interactions.shape) != (len(edges), *expected_response_shape):
        raise ValueError("pair-interaction shape changed")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (
            scientific,
            singletons,
            pairs,
            interactions,
        )
    ):
        raise ValueError("atlas scientific tensors must be finite")
    if not torch.equal(
        singletons,
        scientific[list(expected_atlas.singleton_condition_indices)],
    ) or not torch.equal(
        pairs,
        scientific[list(expected_atlas.pair_condition_indices)],
    ):
        raise ValueError("atlas scientific singleton/pair slices changed")
    recomputed_interactions = vector_pair_interactions(
        empty_output=scientific[0],
        singleton_outputs=singletons,
        pair_outputs=pairs,
        pair_component_indices=expected_atlas.pair_component_indices,
        center=False,
    )
    if not torch.equal(interactions, recomputed_interactions):
        raise ValueError("atlas pair-interaction identity changed")

    eligibility = atlas.get("eligibility_query_accounting", {})
    smoke = atlas.get("batch_invariance_smoke", {})
    atlas_accounting = atlas.get("atlas_query_accounting", {})
    live = atlas.get("total_live_query_accounting", {})
    total = atlas.get("total_query_accounting_including_eligibility", {})
    patched_conditions = condition_count - 1
    expected_smoke_examples = 2 * request.smoke_condition_count * request.required_count
    expected_atlas_examples = patched_conditions * request.required_count
    expected_smoke_forwards = (
        request.smoke_condition_count
        + (request.smoke_condition_count + request.smoke_condition_batch_size - 1)
        // request.smoke_condition_batch_size
    )
    expected_atlas_forwards = (
        patched_conditions + request.atlas_condition_batch_size - 1
    ) // request.atlas_condition_batch_size
    expected_live_examples = (
        2 * request.required_count + expected_smoke_examples + expected_atlas_examples
    )
    expected_live_forwards = 2 + expected_smoke_forwards + expected_atlas_forwards
    expected_examples = int(total.get("model_examples", -1))
    protocol_expected = 2 * request.candidate_count + request.required_count * (
        condition_count + 2 * request.smoke_condition_count + 1
    )
    exact_accounting_checks = (
        eligibility.get("physical_forwards_before_runner") == 2,
        eligibility.get("model_examples_before_runner") == 2 * request.candidate_count,
        smoke.get("condition_count") == request.smoke_condition_count,
        smoke.get("example_count_per_condition") == request.required_count,
        smoke.get("model_examples") == expected_smoke_examples,
        smoke.get("physical_forwards") == expected_smoke_forwards,
        atlas_accounting.get("patched_condition_count") == patched_conditions,
        atlas_accounting.get("example_count_per_condition") == request.required_count,
        atlas_accounting.get("patched_model_examples") == expected_atlas_examples,
        atlas_accounting.get("patched_physical_forwards") == expected_atlas_forwards,
        live.get("model_examples") == expected_live_examples,
        live.get("physical_forwards") == expected_live_forwards,
        total.get("physical_forwards") == expected_live_forwards + 2,
        expected_examples == protocol_expected,
    )
    if not all(exact_accounting_checks):
        raise ValueError("atlas query accounting differs from the frozen protocol")


def _rank_average(values: torch.Tensor) -> torch.Tensor:
    """Average ranks for an exact tie-aware Spearman correlation."""
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values, dtype=torch.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x = _rank_average(torch.tensor(left, dtype=torch.float64))
    y = _rank_average(torch.tensor(right, dtype=torch.float64))
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt(x.square().sum() * y.square().sum())
    if not float(denominator) > 0.0:
        return 0.0
    return float((x * y).sum() / denominator)


def _method_scores(split: AcquiredSplit) -> dict[str, AdaptedScores]:
    atlas = split.atlas
    scores = build_atlas_method_scores(
        node_universe=atlas["component_ids"],
        edge_universe=atlas["directed_candidate_edges"],
        clean_output=atlas["scientific_condition_outputs"][0],
        singleton_outputs=atlas["scientific_singleton_outputs"],
        pair_interactions=atlas["pair_interactions"],
    )
    if set(scores) != set(METHOD_ORDER):
        raise RuntimeError("method set changed")
    return {method: scores[method] for method in METHOD_ORDER}


def _localization(
    scores: AdaptedScores,
    *,
    nodes: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    ground_truth: ExactGroundTruth,
) -> dict[str, Any]:
    result = evaluate_localization(
        node_universe=nodes,
        edge_universe=edges,
        ground_truth_edges=ground_truth.directed_edges,
        ground_truth_nodes=ground_truth.node_ids,
        node_scores=scores.node_scores,
        edge_scores=scores.edge_scores,
        top_k_nodes=len(ground_truth.node_ids),
        top_k_edges=len(ground_truth.directed_edges),
    )
    prevalence = len(ground_truth.directed_edges) / len(edges)
    result["edge"]["chance_auprc"] = prevalence
    result["edge"]["auprc_lift_over_chance"] = result["edge"]["auprc"] - prevalence
    result["edge"]["auprc_ratio_to_chance"] = result["edge"]["auprc"] / prevalence
    result["edge"]["true_positives_at_gt_k"] = round(
        result["edge"]["top_k_recall"] * len(ground_truth.directed_edges)
    )
    return result


def _transitive_diagnostics(
    *,
    edges: tuple[tuple[str, str], ...],
    transitive: tuple[tuple[str, str], ...],
    scores: tuple[float, ...],
    ground_truth_edge_count: int,
) -> dict[str, Any]:
    ranked = sorted(range(len(edges)), key=lambda index: (-scores[index], index))
    ranks = {edges[index]: rank + 1 for rank, index in enumerate(ranked)}
    transitive_ranks = tuple(ranks[edge] for edge in transitive)
    return {
        "ranks_one_based": transitive_ranks,
        "median_rank": (
            None
            if not transitive_ranks
            else float(torch.tensor(transitive_ranks, dtype=torch.float64).median())
        ),
        "inside_gt_k_count": sum(
            rank <= ground_truth_edge_count for rank in transitive_ranks
        ),
    }


def evaluate_case_pair(
    discovery: AcquiredSplit,
    confirmation: AcquiredSplit,
    ground_truth: ExactGroundTruth,
) -> dict[str, Any]:
    """Score fixed-seed D/C splits with response-blind cross-split exclusions."""
    _validate_split(discovery)
    _validate_split(confirmation)
    if not (
        discovery.request.case_id
        == confirmation.request.case_id
        == ground_truth.case_id
    ):
        raise ValueError("split and ground-truth case IDs differ")
    expected_counts = EXPECTED_INTERNAL_GROUND_TRUTH_COUNTS[ground_truth.case_id]
    if (
        len(ground_truth.node_ids) != expected_counts["nodes"]
        or len(ground_truth.directed_edges) != expected_counts["edges"]
    ):
        raise ValueError("internal ground-truth support count changed")
    expected_support = EXPECTED_INTERNAL_GROUND_TRUTH[ground_truth.case_id]
    if ground_truth.node_ids != expected_support["nodes"] or (
        ground_truth.directed_edges != expected_support["edges"]
    ):
        raise ValueError("internal ground-truth support identities changed")
    expected_gt_sha = expected_case_provenance(ground_truth.case_id)["artifact_sha256"][
        "edges.pkl"
    ]
    if ground_truth.raw_edges_sha256 != expected_gt_sha:
        raise ValueError("raw ground-truth artifact hash changed")

    disjointness = assert_discovery_confirmation_disjoint(discovery, confirmation)
    nodes = tuple(discovery.atlas["component_ids"])
    edges = tuple(tuple(edge) for edge in discovery.atlas["directed_candidate_edges"])
    if nodes != tuple(confirmation.atlas["component_ids"]) or edges != tuple(
        tuple(edge) for edge in confirmation.atlas["directed_candidate_edges"]
    ):
        raise ValueError("discovery and confirmation universes differ")
    discovery_scores = _method_scores(discovery)
    confirmation_scores = _method_scores(confirmation)
    methods: dict[str, Any] = {}
    correlations: dict[str, float] = {}
    for method in METHOD_ORDER:
        discovery_localization = _localization(
            discovery_scores[method],
            nodes=nodes,
            edges=edges,
            ground_truth=ground_truth,
        )
        confirmation_localization = _localization(
            confirmation_scores[method],
            nodes=nodes,
            edges=edges,
            ground_truth=ground_truth,
        )
        correlations[method] = _spearman(
            discovery_scores[method].edge_scores,
            confirmation_scores[method].edge_scores,
        )
        methods[method] = {
            "discovery": discovery_localization,
            "confirmation": confirmation_localization,
            "discovery_confirmation_edge_spearman": correlations[method],
            "confirmation_true_positives_at_gt_k": confirmation_localization["edge"][
                "true_positives_at_gt_k"
            ],
        }

    transitive = transitive_candidate_non_edges(
        node_universe=nodes,
        edge_universe=edges,
        ground_truth_edges=ground_truth.directed_edges,
    )
    query_examples = sum(
        int(
            split.atlas["total_query_accounting_including_eligibility"][
                "model_examples"
            ]
        )
        for split in (discovery, confirmation)
    )
    return {
        "schema_version": "e32_1_case_score_v1",
        "case_id": ground_truth.case_id,
        "ground_truth_node_count": len(ground_truth.node_ids),
        "ground_truth_edge_count": len(ground_truth.directed_edges),
        "methods": methods,
        "discovery_confirmation_spearman": correlations,
        "raw_pair_confirmation_sparsity": edge_sparsity_summary(
            confirmation_scores["raw_pair_interaction"].edge_scores
        ),
        "transitive_candidate_non_edges": {
            "count": len(transitive),
            "edges": transitive,
            "discovery": _transitive_diagnostics(
                edges=edges,
                transitive=transitive,
                scores=discovery_scores["raw_pair_interaction"].edge_scores,
                ground_truth_edge_count=len(ground_truth.directed_edges),
            ),
            "confirmation": _transitive_diagnostics(
                edges=edges,
                transitive=transitive,
                scores=confirmation_scores["raw_pair_interaction"].edge_scores,
                ground_truth_edge_count=len(ground_truth.directed_edges),
            ),
        },
        "discovery_confirmation_disjointness": disjointness,
        "split_token_sha256": {
            "discovery_clean": _sha256_tensor_rows(discovery.atlas["clean_inputs"]),
            "discovery_source": _sha256_tensor_rows(discovery.atlas["corrupt_inputs"]),
            "confirmation_clean": _sha256_tensor_rows(
                confirmation.atlas["clean_inputs"]
            ),
            "confirmation_source": _sha256_tensor_rows(
                confirmation.atlas["corrupt_inputs"]
            ),
        },
        "query_accounting": {"combined_model_examples": query_examples},
        "claim_boundary": {
            "direct_edge_gt_but_pair_score_can_measure_indirect_mediation": True,
            "deferred_graph_and_intervention_metrics_omitted": True,
        },
    }


def evaluate_primary_gate(primary: dict[str, Any]) -> dict[str, Any]:
    """Apply the exact conjunctive, case-97-only preregistered gate."""
    if primary.get("case_id") != PRIMARY_CASE_ID:
        raise ValueError("primary gate requires case 97")
    if primary.get("ground_truth_edge_count") != 6:
        raise ValueError("case-97 internal GT edge count changed")
    methods = primary["methods"]
    raw = methods["raw_pair_interaction"]
    raw_ap = float(raw["confirmation"]["edge"]["auprc"])
    comparator_aps = tuple(
        float(methods[method]["confirmation"]["edge"]["auprc"])
        for method in COMPARATOR_METHODS
    )
    raw_tp = int(raw["confirmation_true_positives_at_gt_k"])
    comparator_tp = max(
        int(methods[method]["confirmation_true_positives_at_gt_k"])
        for method in COMPARATOR_METHODS
    )
    stability = float(
        primary["discovery_confirmation_spearman"]["raw_pair_interaction"]
    )
    checks = {
        "confirmation_auprc_margin_over_every_comparator": all(
            raw_ap - comparator_ap
            >= PRIMARY_GATE["minimum_confirmation_auprc_margin_over_each_comparator"]
            for comparator_ap in comparator_aps
        ),
        "confirmation_true_positive_advantage_at_6": (
            raw_tp - comparator_tp
            >= PRIMARY_GATE["minimum_additional_true_positives_at_6"]
        ),
        "discovery_confirmation_raw_energy_spearman": (
            stability
            >= PRIMARY_GATE["minimum_discovery_confirmation_raw_energy_spearman"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "raw_confirmation_auprc": raw_ap,
        "best_comparator_confirmation_auprc": max(comparator_aps),
        "raw_true_positives_at_6": raw_tp,
        "best_comparator_true_positives_at_6": comparator_tp,
        "raw_discovery_confirmation_spearman": stability,
    }


def _full_panel_criterion(case_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if any("methods" not in case_results[str(case_id)] for case_id in CASE_ORDER):
        return {"status": "not_available_from_injected_test_evaluator"}
    raw_wins = 0
    margins = []
    for case_id in CASE_ORDER:
        methods = case_results[str(case_id)]["methods"]
        raw_ap = float(methods["raw_pair_interaction"]["confirmation"]["edge"]["auprc"])
        best_comparator = max(
            float(methods[method]["confirmation"]["edge"]["auprc"])
            for method in COMPARATOR_METHODS
        )
        margin = raw_ap - best_comparator
        margins.append(margin)
        raw_wins += int(margin > 0.0)
    macro_margin = sum(margins) / len(margins)
    checks = {
        "raw_auprc_wins_at_least_two_of_three_cases": raw_wins
        >= FULL_PANEL_GATE["minimum_case_wins_out_of_three"],
        "macro_raw_minus_best_comparator_auprc_at_least_point05": (
            macro_margin
            >= FULL_PANEL_GATE["minimum_macro_auprc_margin_over_best_comparator"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "raw_case_win_count": raw_wins,
        "per_case_auprc_margins": tuple(margins),
        "macro_auprc_margin": macro_margin,
    }


def run_hierarchical_panel(
    acquire: Callable[[SplitRequest], AcquiredSplit],
    ground_truth: Callable[[int], ExactGroundTruth],
) -> dict[str, Any]:
    """Run primary D/C, then authorize secondaries only after a primary pass."""
    if not callable(acquire) or not callable(ground_truth):
        raise TypeError("acquire and ground_truth must be callable")
    primary_discovery = acquire(split_request(PRIMARY_CASE_ID, "discovery"))
    primary_confirmation = acquire(split_request(PRIMARY_CASE_ID, "confirmation"))
    primary = evaluate_case_pair(
        primary_discovery,
        primary_confirmation,
        ground_truth(PRIMARY_CASE_ID),
    )
    primary_gate = evaluate_primary_gate(primary)
    case_results = {str(PRIMARY_CASE_ID): primary}
    if not primary_gate["passed"]:
        return {
            "schema_version": "e32_1_hierarchical_score_v1",
            "case_results": case_results,
            "primary_gate": primary_gate,
            "secondary_status": "not_run_primary_gate_failed",
            "full_panel_criterion": {"status": "not_tested_primary_failed"},
        }

    for case_id in SECONDARY_CASE_IDS:
        discovery = acquire(split_request(case_id, "discovery"))
        confirmation = acquire(split_request(case_id, "confirmation"))
        case_results[str(case_id)] = evaluate_case_pair(
            discovery, confirmation, ground_truth(case_id)
        )
    return {
        "schema_version": "e32_1_hierarchical_score_v1",
        "case_results": case_results,
        "primary_gate": primary_gate,
        "secondary_status": "completed_after_primary_gate_passed",
        "full_panel_criterion": _full_panel_criterion(case_results),
    }


__all__ = [
    "AcquiredSplit",
    "ExactGroundTruth",
    "assert_discovery_confirmation_disjoint",
    "evaluate_case_pair",
    "evaluate_primary_gate",
    "run_hierarchical_panel",
]
