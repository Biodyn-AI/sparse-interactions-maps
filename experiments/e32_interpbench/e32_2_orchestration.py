# ruff: noqa: EXE002
"""Pure custody and actual-work aggregation for the E32.2 panel.

The live driver persists each split before this module sees it.  These checks
therefore repeat the row-custody invariants over the serialized artifacts and
require the complete two-case, two-split panel.  Operational cascade cost is
intentionally absent here: this module totals only work that was observed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import torch

from experiments.e32_interpbench.adapter import (
    CASE_MANIFESTS,
    patchable_internal_universe,
    promote_official_node,
)
from experiments.e32_interpbench.analysis import transitive_candidate_non_edges
from experiments.e32_interpbench.component_patching import (
    build_component_atlas,
    build_patch_units,
    class_center_logits,
    vector_pair_interactions,
)
from experiments.e32_interpbench.e32_2_acquisition import (
    induced_edge_subsequence,
    select_top_singleton_nodes,
    token_rows,
)
from experiments.e32_interpbench.e32_2_directed_replacement import (
    directed_replacement_query_accounting,
)
from experiments.e32_interpbench.e32_2_protocol import (
    EXPECTED_UNIVERSE_COUNTS,
    FULL_UNIVERSE_FLOOR,
    METHOD_ORDER,
    PROSPECTIVE_CASE_IDS,
    expected_case_provenance,
    operational_query_accounting,
    split_request,
    validation_query_accounting,
)
from experiments.e32_interpbench.e32_2_scoring import (
    best_eap_variant,
    bidirectional_fine_replacement_scores,
    build_exhaustive_baseline_scores,
    direct_vs_transitive_auroc,
    embed_shortlist_scores,
    evaluate_case_gates,
    promote_fine_scores_max,
    spearman_correlation,
)
from experiments.e32_interpbench.eap_baseline import (
    FineEdgeScore,
    eap_query_accounting,
    expected_auto_circuit_counts,
    official_eap_settings,
    promote_and_aggregate_fine_scores,
)
from experiments.e32_interpbench.scoring import binary_ranking_metrics

MANDATORY_SPLITS = ("discovery", "confirmation")
MANDATORY_PANEL_KEYS = tuple(
    (case_id, split) for case_id in PROSPECTIVE_CASE_IDS for split in MANDATORY_SPLITS
)
_REQUIRED_ROLE_ROW_COUNTS = {"discovery": 128, "confirmation": 256}


def _edge_tuple(value: Any, *, label: str) -> tuple[tuple[str, str], ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of edges")
    edges = tuple(tuple(edge) for edge in value)
    if (
        not edges
        or len(edges) != len(set(edges))
        or any(
            len(edge) != 2 or not all(isinstance(node, str) and node for node in edge)
            for edge in edges
        )
    ):
        raise ValueError(f"{label} must contain nonempty unique directed edges")
    return edges


def _node_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of nodes")
    nodes = tuple(value)
    if (
        not nodes
        or len(nodes) != len(set(nodes))
        or any(not isinstance(node, str) or not node for node in nodes)
    ):
        raise ValueError(f"{label} must contain nonempty unique node IDs")
    return nodes


def _normalized_serialized_rows(
    value: Any, *, case_id: int, split: str
) -> tuple[tuple[int, ...], ...]:
    """Validate the immutable clean/source rows stored in a split artifact."""
    rows = token_rows(value, label=f"case{case_id} {split} selected token rows")
    expected = _REQUIRED_ROLE_ROW_COUNTS[split]
    if len(rows) != expected:
        raise ValueError(
            f"case{case_id} {split} must contain exactly {expected} selected role rows"
        )
    if len(set(rows)) != len(rows):
        raise RuntimeError("token-row leakage within a serialized split")
    return rows


def assert_complete_fresh_panel(
    panel: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    prior_rows_by_case: Mapping[int, Sequence[Sequence[int]]],
) -> dict[str, Any]:
    """Re-audit mandatory coverage and row disjointness after acquisition.

    Each prospective case is compared with its own full E32.1 candidate pools.
    Discovery and confirmation are also required to be mutually disjoint.  The
    two benchmark cases use separate generators, so cross-case token equality
    is not treated as leakage: token IDs do not have shared task semantics.
    """
    if set(panel) != set(MANDATORY_PANEL_KEYS):
        raise ValueError("E32.2 requires the complete mandatory panel")
    if set(prior_rows_by_case) != set(PROSPECTIVE_CASE_IDS):
        raise ValueError("prior-row custody must cover both prospective cases")

    case_audits: dict[int, dict[str, Any]] = {}
    total_rows = 0
    for case_id in PROSPECTIVE_CASE_IDS:
        if len(prior_rows_by_case[case_id]) != 480:
            raise ValueError("prior-row custody must contain the full 480-row pool")
        prior_rows = {
            tuple(int(token) for token in row) for row in prior_rows_by_case[case_id]
        }
        rows_by_split: dict[str, tuple[tuple[int, ...], ...]] = {}
        for split in MANDATORY_SPLITS:
            bundle = panel[(case_id, split)]
            if bundle.get("schema_version") != "e32_2_split_acquisition_v1":
                raise ValueError("split acquisition schema changed")
            if bundle.get("case_id") != case_id or bundle.get("split") != split:
                raise ValueError("split artifact identity differs from panel key")
            rows = _normalized_serialized_rows(
                bundle.get("selected_token_rows"), case_id=case_id, split=split
            )
            raw = bundle.get("raw_pair_atlas")
            if not isinstance(raw, Mapping):
                raise TypeError("split raw pair atlas must be a mapping")
            scored_rows = (
                *token_rows(raw.get("clean_inputs"), label="raw clean inputs"),
                *token_rows(raw.get("corrupt_inputs"), label="raw source inputs"),
            )
            if rows != scored_rows:
                raise RuntimeError(
                    "token-row leakage: custody rows differ from scored atlas rows"
                )
            if set(rows).intersection(prior_rows):
                raise RuntimeError("token-row leakage from an E32.1 candidate pool")
            rows_by_split[split] = rows
            total_rows += len(rows)

        discovery_rows = set(rows_by_split["discovery"])
        confirmation_rows = set(rows_by_split["confirmation"])
        overlap = discovery_rows.intersection(confirmation_rows)
        if overlap:
            raise RuntimeError("token-row leakage between discovery and confirmation")
        case_audits[case_id] = {
            "prior_candidate_row_count": len(prior_rows),
            "discovery_selected_role_rows": len(discovery_rows),
            "confirmation_selected_role_rows": len(confirmation_rows),
            "prior_overlap_count": 0,
            "cross_split_overlap_count": 0,
        }

    return {
        "schema_version": "e32_2_complete_panel_custody_v1",
        "case_order": PROSPECTIVE_CASE_IDS,
        "all_cases_mandatory": True,
        "cross_case_token_ids_compared": False,
        "cross_case_policy": "token_ids_are_task_local_not_shared_row_identities",
        "total_selected_role_rows": total_rows,
        "cases": case_audits,
    }


def _expected_fine_multiplicity(edge: tuple[str, str]) -> int:
    """AutoCircuit exposes Q/K/V separately for attention-head recipients."""
    return 3 if ".attn.hook_result[" in edge[1] else 1


def _validate_directed_metadata(
    directed: Any,
    *,
    case_id: int,
    split: str,
    expected_edges: tuple[tuple[str, str], ...],
    requested_edges: tuple[tuple[str, str], ...],
    example_count: int,
) -> int:
    if not isinstance(directed, Mapping):
        raise TypeError("directed replacement envelope must be a mapping")
    if (
        directed.get("schema_version") != "e32_2_directed_replacement_envelope_v1"
        or directed.get("case_id") != case_id
    ):
        raise ValueError("directed replacement identity changed")
    expected_settings = {
        "factorized": True,
        "separate_qkv": True,
        "slice_output": "not_first_seq",
        "seq_len": None,
        "ablation_type": "resample",
        "primary": "one_fine_edge_per_replacement_then_qkv_max",
        "grouped_promoted_sensitivity": False,
        "clean_orientation": "clean_recipient_corrupt_donor",
        "corrupt_orientation": "corrupt_recipient_clean_donor",
        "ce_delta": "patched_recipient_target_ce_minus_recipient_baseline_ce",
    }
    settings = directed.get("settings")
    if not isinstance(settings, Mapping) or dict(settings) != expected_settings:
        raise ValueError("directed replacement settings changed")

    grouping = directed.get("grouping")
    if not isinstance(grouping, Mapping):
        raise TypeError("directed grouping must be a mapping")
    group_edges = _edge_tuple(grouping.get("edge_order"), label="group edge order")
    if group_edges != expected_edges:
        raise ValueError("directed full promoted-edge universe or order changed")
    groups = tuple(grouping.get("groups", ()))
    if len(groups) != len(expected_edges):
        raise ValueError("directed promoted grouping length changed")
    collision_counts: list[int] = []
    group_fine_names: dict[tuple[str, str], tuple[str, ...]] = {}
    for edge, group in zip(expected_edges, groups, strict=True):
        if not isinstance(group, Mapping):
            raise TypeError("directed promoted group must be a mapping")
        fine_names = tuple(group.get("fine_edge_names", ()))
        multiplicity = _expected_fine_multiplicity(edge)
        if (
            tuple(group.get("promoted_edge", ())) != edge
            or group.get("collision_count") != multiplicity
            or len(fine_names) != multiplicity
            or len(set(fine_names)) != multiplicity
            or any(not isinstance(name, str) or not name for name in fine_names)
        ):
            raise ValueError("directed fine/group metadata changed")
        collision_counts.append(multiplicity)
        group_fine_names[edge] = fine_names
    expected_internal_fine_count = sum(collision_counts)
    graph_counts = expected_auto_circuit_counts(
        CASE_MANIFESTS[case_id].n_layers, CASE_MANIFESTS[case_id].n_heads
    )
    if (
        grouping.get("internal_fine_edge_count") != expected_internal_fine_count
        or expected_internal_fine_count
        != EXPECTED_UNIVERSE_COUNTS[case_id]["fine_edges"]
        or grouping.get("fine_edge_count") != graph_counts.fine_edges
        or grouping.get("excluded_boundary_fine_edge_count")
        != graph_counts.boundary_fine_edges
    ):
        raise ValueError("directed internal fine-edge count changed")

    observed_requested = _edge_tuple(
        directed.get("requested_promoted_edges"), label="requested promoted edges"
    )
    if observed_requested != requested_edges:
        raise ValueError(f"{split} directed promoted-edge request changed")
    full_index = {edge: index for index, edge in enumerate(expected_edges)}
    requested_fine_count = sum(
        collision_counts[full_index[edge]] for edge in requested_edges
    )
    fine_names = tuple(directed.get("fine_edge_names", ()))
    expected_fine_names = tuple(
        name for edge in requested_edges for name in group_fine_names[edge]
    )
    expected_mapping = tuple(
        index
        for index, edge in enumerate(requested_edges)
        for _ in range(collision_counts[full_index[edge]])
    )
    if (
        fine_names != expected_fine_names
        or len(set(fine_names)) != requested_fine_count
        or tuple(directed.get("fine_to_promoted_indices", ())) != expected_mapping
    ):
        raise ValueError("directed fine/group metadata changed")

    accounting = directed.get("query_accounting")
    if not isinstance(accounting, Mapping):
        raise TypeError("directed query accounting must be a mapping")
    expected_accounting = asdict(
        directed_replacement_query_accounting(
            example_count=example_count,
            promoted_edge_count=len(requested_edges),
            fine_edge_count=requested_fine_count,
            include_grouped_promoted_sensitivity=False,
        )
    )
    if dict(accounting) != expected_accounting:
        raise ValueError("directed query-accounting mismatch")
    return requested_fine_count


def _validate_eap_metadata(
    envelope: Any,
    *,
    case_id: int,
    method: str,
    expected_edges: tuple[tuple[str, str], ...],
    example_count: int,
    open_scientific_outcomes: bool,
) -> tuple[FineEdgeScore, ...] | None:
    if not isinstance(envelope, Mapping):
        raise TypeError("official EAP envelope must be a mapping")
    run = envelope.get("run")
    observed = envelope.get("observed_query_accounting")
    if not isinstance(run, Mapping) or not isinstance(observed, Mapping):
        raise TypeError("official EAP run/accounting must be mappings")
    if run.get("case_id") != case_id or dict(run.get("settings", {})) != asdict(
        official_eap_settings(method)
    ):
        raise ValueError("official EAP identity or method changed")
    promoted = run.get("promoted_scores")
    if not isinstance(promoted, Mapping):
        raise TypeError("official EAP promoted scores must be a mapping")
    collision_counts = tuple(promoted.get("collision_counts", ()))
    expected_collision_counts = tuple(
        _expected_fine_multiplicity(edge) for edge in expected_edges
    )
    expected_aggregation = {
        "signed_primary": "max_raw_score_per_promoted_edge",
        "absolute_sensitivity": "max_absolute_raw_score_per_promoted_edge",
        "support_parity": "strict_fine_threshold_then_official_binary_rerouting",
    }
    if (
        _edge_tuple(promoted.get("edge_order"), label="EAP edge order")
        != expected_edges
        or promoted.get("coverage_complete") is not True
        or promoted.get("internal_fine_edge_count")
        != EXPECTED_UNIVERSE_COUNTS[case_id]["fine_edges"]
        or collision_counts != expected_collision_counts
        or dict(promoted.get("aggregation", {})) != expected_aggregation
    ):
        raise ValueError("official EAP promoted coverage changed")
    for name in ("signed_max_scores", "absolute_max_scores"):
        if len(tuple(promoted.get(name, ()))) != len(expected_edges):
            raise ValueError("official EAP promoted score length changed")
    fine_records = tuple(run.get("fine_scores", ()))
    if promoted.get("fine_edge_count") != len(fine_records) or promoted.get(
        "excluded_boundary_fine_edge_count"
    ) != len(fine_records) - sum(expected_collision_counts):
        raise ValueError("official EAP fine/boundary counts changed")
    expected = asdict(eap_query_accounting(example_count=example_count, method=method))
    if dict(run.get("query_accounting", {})) != expected:
        raise ValueError("official EAP query-accounting mismatch")
    expected_observed = {
        "physical_forwards": expected["physical_model_forwards"],
        "model_examples": expected["model_examples"],
        "no_grad_forwards": expected["source_ablation_forwards"],
        "grad_enabled_forwards": expected["gradient_forwards"],
        "backward_passes": expected["backward_passes"],
        "matches_official_control_flow": True,
    }
    if dict(observed) != expected_observed:
        raise ValueError("official EAP observed accounting changed")
    if open_scientific_outcomes:
        try:
            raw_fine = tuple(FineEdgeScore(**dict(record)) for record in fine_records)
        except (TypeError, ValueError) as error:
            raise ValueError("official EAP raw fine-score records changed") from error
        recomputed = promote_and_aggregate_fine_scores(case_id, raw_fine)
        exact_checks = (
            recomputed.edge_order == expected_edges,
            tuple(recomputed.collision_counts) == collision_counts,
            recomputed.fine_edge_count == promoted.get("fine_edge_count"),
            recomputed.internal_fine_edge_count
            == promoted.get("internal_fine_edge_count"),
            recomputed.excluded_boundary_fine_edge_count
            == promoted.get("excluded_boundary_fine_edge_count"),
            recomputed.coverage_complete is True,
            torch.equal(
                torch.tensor(recomputed.signed_max_scores, dtype=torch.float64),
                torch.as_tensor(promoted.get("signed_max_scores"), dtype=torch.float64),
            ),
            torch.equal(
                torch.tensor(recomputed.absolute_max_scores, dtype=torch.float64),
                torch.as_tensor(
                    promoted.get("absolute_max_scores"), dtype=torch.float64
                ),
            ),
        )
        if not all(exact_checks):
            raise ValueError("official EAP promotion differs from raw fine scores")
        return raw_fine
    return None


def _validate_directed_names_against_eap(
    directed: Mapping[str, Any],
    *,
    expected_edges: tuple[tuple[str, str], ...],
    eap_fine_scores: tuple[FineEdgeScore, ...],
) -> None:
    edge_set = set(expected_edges)
    grouped: dict[tuple[str, str], list[FineEdgeScore]] = {
        edge: [] for edge in expected_edges
    }
    for fine in eap_fine_scores:
        promoted = (
            promote_official_node(fine.source),
            promote_official_node(fine.target),
        )
        if promoted in edge_set:
            grouped[promoted].append(fine)

    def order_key(fine: FineEdgeScore) -> tuple[int, str]:
        slots = tuple(
            index
            for index, letter in enumerate("qkv")
            if f"hook_{letter}_input" in fine.target
        )
        slot = slots[0] if slots else 0
        return slot, fine.fine_edge_name

    expected_names = {
        edge: tuple(
            fine.fine_edge_name for fine in sorted(grouped[edge], key=order_key)
        )
        for edge in expected_edges
    }
    observed_groups = tuple(directed["grouping"]["groups"])
    if any(
        tuple(group["fine_edge_names"]) != expected_names[edge]
        for edge, group in zip(expected_edges, observed_groups, strict=True)
    ):
        raise ValueError("directed fine names differ from the official EAP graph")


def _validate_raw_accounting(
    raw: Mapping[str, Any], *, request: Any, patched_condition_count: int
) -> None:
    smoke_forwards = (
        request.smoke_condition_count
        + (request.smoke_condition_count + request.smoke_condition_batch_size - 1)
        // request.smoke_condition_batch_size
    )
    atlas_forwards = (
        patched_condition_count + request.atlas_condition_batch_size - 1
    ) // request.atlas_condition_batch_size
    smoke_examples = 2 * request.smoke_condition_count * request.required_count
    atlas_examples = patched_condition_count * request.required_count
    live_forwards = 2 + smoke_forwards + atlas_forwards
    live_examples = 2 * request.required_count + smoke_examples + atlas_examples
    eligibility = raw.get("eligibility_query_accounting", {})
    smoke = raw.get("batch_invariance_smoke", {})
    atlas = raw.get("atlas_query_accounting", {})
    live = raw.get("total_live_query_accounting", {})
    total = raw.get("total_query_accounting_including_eligibility", {})
    if not all(
        isinstance(value, Mapping) for value in (eligibility, smoke, atlas, live, total)
    ):
        raise TypeError("raw atlas query-accounting records must be mappings")
    checks = (
        eligibility.get("physical_forwards_before_runner") == 2,
        eligibility.get("model_examples_before_runner") == 2 * request.candidate_count,
        smoke.get("condition_count") == request.smoke_condition_count,
        smoke.get("example_count_per_condition") == request.required_count,
        smoke.get("reference_condition_batch_size") == 1,
        smoke.get("candidate_condition_batch_size")
        == request.smoke_condition_batch_size,
        smoke.get("model_examples") == smoke_examples,
        smoke.get("physical_forwards") == smoke_forwards,
        atlas.get("patched_condition_count") == patched_condition_count,
        atlas.get("example_count_per_condition") == request.required_count,
        atlas.get("patched_model_examples") == atlas_examples,
        atlas.get("patched_physical_forwards") == atlas_forwards,
        live.get("model_examples") == live_examples,
        live.get("physical_forwards") == live_forwards,
        total.get("model_examples") == live_examples + 2 * request.candidate_count,
        total.get("physical_forwards") == live_forwards + 2,
    )
    if not all(checks):
        raise ValueError("raw-atlas query-accounting mismatch")


def _validate_raw_scientific_outcomes(
    raw: Mapping[str, Any], *, case_id: int, example_count: int
) -> None:
    manifest = CASE_MANIFESTS[case_id]
    nodes, edges = patchable_internal_universe(manifest.n_layers, manifest.n_heads)
    expected_atlas = build_component_atlas(
        build_patch_units(manifest.n_layers, manifest.n_heads),
        valid_directed_edges=edges,
    )
    if (
        tuple(raw.get("condition_ids", ())) != expected_atlas.condition_ids
        or not torch.equal(
            torch.as_tensor(raw.get("condition_masks"), device="cpu"),
            expected_atlas.condition_masks,
        )
        or tuple(raw.get("pair_component_indices", ()))
        != expected_atlas.pair_component_indices
    ):
        raise ValueError("raw-atlas condition manifest changed")
    raw_conditions = torch.as_tensor(raw.get("condition_outputs"), device="cpu")
    scientific = torch.as_tensor(raw.get("scientific_condition_outputs"), device="cpu")
    singletons = torch.as_tensor(raw.get("scientific_singleton_outputs"), device="cpu")
    pairs = torch.as_tensor(raw.get("scientific_pair_outputs"), device="cpu")
    interactions = torch.as_tensor(raw.get("pair_interactions"), device="cpu")
    response_shape = (
        example_count,
        manifest.n_ctx - 1,
        manifest.d_vocab_out,
    )
    condition_count = 1 + len(nodes) + len(edges)
    if (
        tuple(raw_conditions.shape)
        != (condition_count, example_count, manifest.n_ctx, manifest.d_vocab_out)
        or tuple(scientific.shape) != (condition_count, *response_shape)
        or tuple(singletons.shape) != (len(nodes), *response_shape)
        or tuple(pairs.shape) != (len(edges), *response_shape)
        or tuple(interactions.shape) != (len(edges), *response_shape)
    ):
        raise ValueError("raw-atlas scientific response axes changed")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (raw_conditions, scientific, singletons, pairs, interactions)
    ):
        raise ValueError("raw-atlas scientific outcomes must be finite")
    if (
        not torch.equal(scientific, class_center_logits(raw_conditions)[..., 1:, :])
        or not torch.equal(
            singletons,
            scientific[list(expected_atlas.singleton_condition_indices)],
        )
        or not torch.equal(
            pairs, scientific[list(expected_atlas.pair_condition_indices)]
        )
        or not torch.equal(
            interactions,
            vector_pair_interactions(
                empty_output=scientific[0],
                singleton_outputs=singletons,
                pair_outputs=pairs,
                pair_component_indices=expected_atlas.pair_component_indices,
                center=False,
            ),
        )
    ):
        raise ValueError("raw-atlas scientific projection or pair identity changed")


def validate_case_bundle_pair(
    discovery: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    *,
    open_scientific_outcomes: bool = False,
) -> dict[str, Any]:
    """Validate response-blind metadata before opening GT or score tensors."""
    case_id = discovery.get("case_id")
    if case_id not in PROSPECTIVE_CASE_IDS or confirmation.get("case_id") != case_id:
        raise ValueError("case bundle pair identity changed")
    expected_nodes, expected_edges = patchable_internal_universe(
        CASE_MANIFESTS[case_id].n_layers, CASE_MANIFESTS[case_id].n_heads
    )
    counts = EXPECTED_UNIVERSE_COUNTS[case_id]
    if (
        len(expected_nodes) != counts["nodes"]
        or len(expected_edges) != counts["whole_edges"]
    ):
        raise RuntimeError("adapter universe differs from the frozen protocol")

    for split, bundle in (
        ("discovery", discovery),
        ("confirmation", confirmation),
    ):
        request = split_request(case_id, split)
        if bundle.get("schema_version") != "e32_2_split_acquisition_v1":
            raise ValueError("split acquisition schema changed")
        if bundle.get("case_id") != case_id or bundle.get("split") != split:
            raise ValueError("split acquisition identity changed")
        if dict(bundle.get("request", {})) != asdict(request):
            raise ValueError("split request differs from the frozen protocol")
        raw = bundle.get("raw_pair_atlas")
        if not isinstance(raw, Mapping):
            raise TypeError("raw pair atlas must be a mapping")
        if (
            raw.get("schema_version") != "e32_interpbench_component_atlas_v2"
            or raw.get("case_id") != case_id
            or _node_tuple(raw.get("component_ids"), label="component IDs")
            != expected_nodes
            or _edge_tuple(raw.get("directed_candidate_edges"), label="raw edges")
            != expected_edges
            or raw.get("scientific_output_coordinate")
            != "class_centered_logits_non_bos_positions"
            or raw.get("batch_invariance_smoke", {}).get("passed") is not True
            or raw.get("behavioral_pair_screen", {}).get("all_pairs_correct")
            is not True
        ):
            raise ValueError("raw atlas identity, universe, or safeguards changed")
        _validate_raw_accounting(
            raw,
            request=request,
            patched_condition_count=len(expected_nodes) + len(expected_edges),
        )
        if open_scientific_outcomes:
            _validate_raw_scientific_outcomes(
                raw, case_id=case_id, example_count=request.required_count
            )
            selected_nodes, energies = select_top_singleton_nodes(
                node_ids=expected_nodes,
                clean_output=raw["scientific_condition_outputs"][0],
                singleton_outputs=raw["scientific_singleton_outputs"],
                selected_count=counts["shortlist_nodes"],
            )
            if (
                tuple(bundle.get("split_top_half_nodes", ())) != selected_nodes
                or tuple(bundle.get("split_singleton_node_energies", ())) != energies
            ):
                raise ValueError("serialized singleton shortlist differs from outcomes")

    discovery_nodes = _node_tuple(
        discovery.get("discovery_frozen_nodes"), label="discovery frozen nodes"
    )
    if (
        discovery_nodes
        != _node_tuple(discovery.get("split_top_half_nodes"), label="D top nodes")
        or len(discovery_nodes) != counts["shortlist_nodes"]
        or confirmation.get("discovery_frozen_nodes") != discovery_nodes
    ):
        raise ValueError("wrong discovery shortlist used in confirmation")
    induced_edges = induced_edge_subsequence(expected_edges, discovery_nodes)
    if (
        tuple(discovery.get("discovery_induced_edges", ())) != induced_edges
        or tuple(confirmation.get("discovery_induced_edges", ())) != induced_edges
    ):
        raise ValueError("discovery induced-edge support changed")

    discovery_fine_count = _validate_directed_metadata(
        discovery.get("directed_replacement"),
        case_id=case_id,
        split="discovery",
        expected_edges=expected_edges,
        requested_edges=induced_edges,
        example_count=split_request(case_id, "discovery").required_count,
    )
    confirmation_fine_count = _validate_directed_metadata(
        confirmation.get("directed_replacement"),
        case_id=case_id,
        split="confirmation",
        expected_edges=expected_edges,
        requested_edges=expected_edges,
        example_count=split_request(case_id, "confirmation").required_count,
    )
    for split, bundle in (
        ("discovery", discovery),
        ("confirmation", confirmation),
    ):
        examples = split_request(case_id, split).required_count
        eap_fine = _validate_eap_metadata(
            bundle.get("official_eap"),
            case_id=case_id,
            method="eap",
            expected_edges=expected_edges,
            example_count=examples,
            open_scientific_outcomes=open_scientific_outcomes,
        )
        _validate_eap_metadata(
            bundle.get("official_eap_ig"),
            case_id=case_id,
            method="eap_ig",
            expected_edges=expected_edges,
            example_count=examples,
            open_scientific_outcomes=open_scientific_outcomes,
        )
        if open_scientific_outcomes:
            assert eap_fine is not None
            _validate_directed_names_against_eap(
                bundle["directed_replacement"],
                expected_edges=expected_edges,
                eap_fine_scores=eap_fine,
            )
        actual = bundle.get("actual_query_accounting")
        if not isinstance(actual, Mapping):
            raise TypeError("split actual query accounting must be a mapping")
        expected_actual = {
            "behavior_and_raw_atlas": dict(
                bundle["raw_pair_atlas"]["total_query_accounting_including_eligibility"]
            ),
            "directed_replacement": dict(
                bundle["directed_replacement"]["query_accounting"]
            ),
            "official_eap": dict(bundle["official_eap"]["observed_query_accounting"]),
            "official_eap_ig": dict(
                bundle["official_eap_ig"]["observed_query_accounting"]
            ),
            "operational_budget_status": "reported_separately_by_protocol",
        }
        if dict(actual) != expected_actual:
            raise ValueError("split query-accounting mismatch")
    if confirmation_fine_count != counts["fine_edges"]:
        raise ValueError("confirmation directed replacement is not exhaustive")
    return {
        "case_id": case_id,
        "node_count": len(expected_nodes),
        "whole_edge_count": len(expected_edges),
        "induced_whole_edge_count": len(induced_edges),
        "induced_fine_edge_count": discovery_fine_count,
        "confirmation_fine_edge_count": confirmation_fine_count,
        "metadata_validated_without_ground_truth": True,
    }


def _directed_whole_scores(
    directed: Mapping[str, Any], *, example_count: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Construct the registered class-centered bidirectional exact score."""
    clean = class_center_logits(
        torch.as_tensor(directed.get("clean_baseline_logits"), device="cpu")
    )
    corrupt = class_center_logits(
        torch.as_tensor(directed.get("corrupt_baseline_logits"), device="cpu")
    )
    clean_recipient = class_center_logits(
        torch.as_tensor(
            directed.get("fine_clean_recipient_corrupt_donor_logits"),
            device="cpu",
        )
    )
    corrupt_recipient = class_center_logits(
        torch.as_tensor(
            directed.get("fine_corrupt_recipient_clean_donor_logits"),
            device="cpu",
        )
    )
    expected_base = (
        example_count,
        CASE_MANIFESTS[int(directed["case_id"])].n_ctx - 1,
        CASE_MANIFESTS[int(directed["case_id"])].d_vocab_out,
    )
    fine_count = len(tuple(directed.get("fine_edge_names", ())))
    if (
        tuple(clean.shape) != expected_base
        or tuple(corrupt.shape) != expected_base
        or tuple(clean_recipient.shape) != (fine_count, *expected_base)
        or tuple(corrupt_recipient.shape) != (fine_count, *expected_base)
    ):
        raise ValueError("directed scientific response axes changed")
    clean_ce = torch.as_tensor(
        directed.get("clean_recipient_target_ce_deltas"), device="cpu"
    )
    corrupt_ce = torch.as_tensor(
        directed.get("corrupt_recipient_target_ce_deltas"), device="cpu"
    )
    grouped_fields = (
        "grouped_clean_recipient_corrupt_donor_logits",
        "grouped_corrupt_recipient_clean_donor_logits",
        "grouped_clean_recipient_target_ce_deltas",
        "grouped_corrupt_recipient_target_ce_deltas",
    )
    if (
        tuple(clean_ce.shape) != (fine_count, example_count)
        or tuple(corrupt_ce.shape) != (fine_count, example_count)
        or not bool(torch.isfinite(clean_ce).all())
        or not bool(torch.isfinite(corrupt_ce).all())
        or any(directed.get(name) is not None for name in grouped_fields)
    ):
        raise ValueError("directed CE/grouped scientific response axes changed")
    scored = bidirectional_fine_replacement_scores(
        clean_output=clean,
        corrupt_output=corrupt,
        # Field names describe the recipient first.  The scorer arguments
        # describe the donor first, so these orientations intentionally cross.
        clean_into_corrupt_outputs=corrupt_recipient,
        corrupt_into_clean_outputs=clean_recipient,
    )
    requested = _edge_tuple(
        directed.get("requested_promoted_edges"), label="directed requested edges"
    )
    whole = promote_fine_scores_max(
        fine_scores=scored.fine_scores,
        fine_to_whole_indices=tuple(directed.get("fine_to_promoted_indices", ())),
        whole_edge_count=len(requested),
    )
    return whole, {
        "fine_scores": tuple(float(value) for value in scored.fine_scores),
        "sufficiency_mean_by_fine": tuple(
            float(value) for value in scored.sufficiency_by_fine_context.mean(dim=1)
        ),
        "necessity_mean_by_fine": tuple(
            float(value) for value in scored.necessity_by_fine_context.mean(dim=1)
        ),
        "informative_context_count": int(scored.informative_context_mask.sum()),
        "registered_zero_gap_context_count": int(
            (~scored.informative_context_mask).sum()
        ),
    }


def _ranking_summary(
    *,
    scores: Any,
    edge_universe: tuple[tuple[str, str], ...],
    ground_truth_edges: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    values = torch.as_tensor(scores, dtype=torch.float64, device="cpu")
    labels = tuple(edge in set(ground_truth_edges) for edge in edge_universe)
    metrics = binary_ranking_metrics(
        scores=values.tolist(), labels=labels, top_k=len(ground_truth_edges)
    )
    selected_indices = tuple(metrics["selected_indices"])
    selected_edges = tuple(edge_universe[index] for index in selected_indices)
    return {
        **metrics,
        "true_positives_at_gt_k": sum(
            edge in ground_truth_edges for edge in selected_edges
        ),
        "selected_edges_at_gt_k": selected_edges,
    }


def _eap_score_vectors(bundle: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    eap = bundle["official_eap"]["run"]["promoted_scores"]
    eap_ig = bundle["official_eap_ig"]["run"]["promoted_scores"]
    return {
        "eap_signed_max": torch.as_tensor(
            eap["signed_max_scores"], dtype=torch.float64, device="cpu"
        ),
        "eap_absolute_max": torch.as_tensor(
            eap["absolute_max_scores"], dtype=torch.float64, device="cpu"
        ),
        "eap_ig_signed_max": torch.as_tensor(
            eap_ig["signed_max_scores"], dtype=torch.float64, device="cpu"
        ),
        "eap_ig_absolute_max": torch.as_tensor(
            eap_ig["absolute_max_scores"], dtype=torch.float64, device="cpu"
        ),
    }


def _raw_score_vectors(bundle: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    atlas = bundle["raw_pair_atlas"]
    panel = build_exhaustive_baseline_scores(
        node_universe=atlas["component_ids"],
        edge_universe=atlas["directed_candidate_edges"],
        clean_output=atlas["scientific_condition_outputs"][0],
        singleton_outputs=atlas["scientific_singleton_outputs"],
        pair_interactions=atlas["pair_interactions"],
    )
    return {
        name: torch.tensor(scores.edge_scores, dtype=torch.float64)
        for name, scores in panel.items()
    }


def score_case_bundles(
    discovery: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    *,
    ground_truth: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one frozen D/C pair after the score-stage reservation exists."""
    structure = validate_case_bundle_pair(
        discovery, confirmation, open_scientific_outcomes=True
    )
    case_id = structure["case_id"]
    provenance = expected_case_provenance(case_id)
    ground_truth_nodes = _node_tuple(
        ground_truth.get("node_ids"), label="ground-truth nodes"
    )
    ground_truth_edges = _edge_tuple(
        ground_truth.get("directed_edges"), label="ground-truth edges"
    )
    if (
        ground_truth.get("case_id") != case_id
        or ground_truth.get("raw_edges_sha256")
        != provenance["artifact_sha256"]["edges.pkl"]
        or len(ground_truth_nodes)
        != EXPECTED_UNIVERSE_COUNTS[case_id]["ground_truth_nodes"]
        or len(ground_truth_edges)
        != EXPECTED_UNIVERSE_COUNTS[case_id]["ground_truth_edges"]
    ):
        raise ValueError("ground truth differs from the hash-bound exact identities")

    full_edges = _edge_tuple(
        discovery["raw_pair_atlas"]["directed_candidate_edges"],
        label="full edge universe",
    )
    full_nodes = _node_tuple(
        discovery["raw_pair_atlas"]["component_ids"], label="full node universe"
    )
    if not set(ground_truth_edges).issubset(full_edges) or not set(
        ground_truth_nodes
    ).issubset(full_nodes):
        raise ValueError("ground-truth identities escape the frozen universe")
    induced_edges = _edge_tuple(
        discovery["discovery_induced_edges"], label="induced edges"
    )

    discovery_exact, discovery_diagnostics = _directed_whole_scores(
        discovery["directed_replacement"],
        example_count=split_request(case_id, "discovery").required_count,
    )
    confirmation_exhaustive, confirmation_diagnostics = _directed_whole_scores(
        confirmation["directed_replacement"],
        example_count=split_request(case_id, "confirmation").required_count,
    )
    edge_index = {edge: index for index, edge in enumerate(full_edges)}
    confirmation_induced = confirmation_exhaustive[
        torch.tensor([edge_index[edge] for edge in induced_edges], dtype=torch.long)
    ]
    discovery_primary = embed_shortlist_scores(
        full_edge_universe=full_edges,
        shortlisted_edges=induced_edges,
        shortlisted_scores=discovery_exact,
        floor=FULL_UNIVERSE_FLOOR,
    )
    confirmation_primary = embed_shortlist_scores(
        full_edge_universe=full_edges,
        shortlisted_edges=induced_edges,
        shortlisted_scores=confirmation_induced,
        floor=FULL_UNIVERSE_FLOOR,
    )

    discovery_raw = _raw_score_vectors(discovery)
    confirmation_raw = _raw_score_vectors(confirmation)
    discovery_eap = _eap_score_vectors(discovery)
    confirmation_eap = _eap_score_vectors(confirmation)
    confirmation_scores: dict[str, torch.Tensor] = {
        "bidirectional_fine_replacement": confirmation_primary,
        "exhaustive_bidirectional_fine_replacement": confirmation_exhaustive,
        **confirmation_eap,
        **confirmation_raw,
    }
    if tuple(confirmation_scores) != METHOD_ORDER:
        raise RuntimeError("confirmation method order changed")
    confirmation_metrics = {
        name: _ranking_summary(
            scores=scores,
            edge_universe=full_edges,
            ground_truth_edges=ground_truth_edges,
        )
        for name, scores in confirmation_scores.items()
    }

    discovery_scores = {
        "bidirectional_fine_replacement": discovery_primary,
        **discovery_eap,
        **discovery_raw,
    }
    discovery_metrics = {
        name: _ranking_summary(
            scores=scores,
            edge_universe=full_edges,
            ground_truth_edges=ground_truth_edges,
        )
        for name, scores in discovery_scores.items()
    }
    best_eap_name, best_eap_metrics = best_eap_variant(
        {name: confirmation_metrics[name] for name in confirmation_eap}
    )

    gated_raw = embed_shortlist_scores(
        full_edge_universe=full_edges,
        shortlisted_edges=induced_edges,
        shortlisted_scores=confirmation_raw["raw_pair_interaction"][
            torch.tensor([edge_index[edge] for edge in induced_edges], dtype=torch.long)
        ],
        floor=FULL_UNIVERSE_FLOOR,
    )
    transitive = transitive_candidate_non_edges(
        node_universe=full_nodes,
        edge_universe=full_edges,
        ground_truth_edges=ground_truth_edges,
    )
    primary_direct_transitive = direct_vs_transitive_auroc(
        scores=confirmation_primary,
        edge_universe=full_edges,
        direct_edges=ground_truth_edges,
        transitive_non_edges=transitive,
    )
    raw_direct_transitive = direct_vs_transitive_auroc(
        scores=gated_raw,
        edge_universe=full_edges,
        direct_edges=ground_truth_edges,
        transitive_non_edges=transitive,
    )
    stability = spearman_correlation(discovery_exact, confirmation_induced)
    direct_retention = sum(edge in induced_edges for edge in ground_truth_edges) / len(
        ground_truth_edges
    )
    operational = operational_query_accounting(
        case_id, structure["induced_fine_edge_count"]
    )
    validation = validation_query_accounting(
        case_id, structure["induced_fine_edge_count"]
    )
    primary = confirmation_metrics["bidirectional_fine_replacement"]
    exhaustive = confirmation_metrics["exhaustive_bidirectional_fine_replacement"]
    raw = confirmation_metrics["raw_pair_interaction"]
    remaining_names = tuple(
        name
        for name in METHOD_ORDER[1:]
        if name
        not in {
            "exhaustive_bidirectional_fine_replacement",
            best_eap_name,
            "raw_pair_interaction",
        }
    )
    gates = evaluate_case_gates(
        ground_truth_edge_count=len(ground_truth_edges),
        direct_edge_retention=direct_retention,
        induced_whole_edge_fraction=len(induced_edges) / len(full_edges),
        induced_fine_edge_fraction=structure["induced_fine_edge_count"]
        / EXPECTED_UNIVERSE_COUNTS[case_id]["fine_edges"],
        operational_model_example_savings=float(
            operational["model_example_savings_fraction"]
        ),
        confirmation_auprc=float(primary["auprc"]),
        confirmation_true_positives_at_gt_k=int(primary["true_positives_at_gt_k"]),
        exhaustive_exact_auprc=float(exhaustive["auprc"]),
        exhaustive_exact_true_positives_at_gt_k=int(
            exhaustive["true_positives_at_gt_k"]
        ),
        best_eap_auprc=float(best_eap_metrics["auprc"]),
        best_eap_true_positives_at_gt_k=int(best_eap_metrics["true_positives_at_gt_k"]),
        raw_pair_auprc=float(raw["auprc"]),
        raw_pair_true_positives_at_gt_k=int(raw["true_positives_at_gt_k"]),
        other_exhaustive_comparator_auprcs=tuple(
            float(confirmation_metrics[name]["auprc"]) for name in remaining_names
        ),
        other_exhaustive_comparator_true_positives=tuple(
            int(confirmation_metrics[name]["true_positives_at_gt_k"])
            for name in remaining_names
        ),
        direct_vs_transitive_auroc=primary_direct_transitive,
        node_gated_raw_direct_vs_transitive_auroc=raw_direct_transitive,
        discovery_confirmation_spearman=stability,
    )
    return {
        "schema_version": "e32_2_case_score_v1",
        "case_id": case_id,
        "ground_truth": {
            "node_ids": ground_truth_nodes,
            "directed_edges": ground_truth_edges,
            "raw_edges_sha256": ground_truth["raw_edges_sha256"],
        },
        "structure": {
            **structure,
            "direct_edge_retention": direct_retention,
            "induced_whole_edge_fraction": len(induced_edges) / len(full_edges),
            "induced_fine_edge_fraction": structure["induced_fine_edge_count"]
            / EXPECTED_UNIVERSE_COUNTS[case_id]["fine_edges"],
        },
        "discovery_metrics": discovery_metrics,
        "confirmation_metrics": confirmation_metrics,
        "full_universe_score_vectors": {
            "edge_order": full_edges,
            "discovery": {
                name: tuple(float(value) for value in scores)
                for name, scores in discovery_scores.items()
            },
            "confirmation": {
                name: tuple(float(value) for value in scores)
                for name, scores in confirmation_scores.items()
            },
            "discovery_unavailable_methods": (
                "exhaustive_bidirectional_fine_replacement",
            ),
        },
        "best_eap_variant": best_eap_name,
        "direct_vs_transitive": {
            "transitive_negative_edges": transitive,
            "primary_auroc": primary_direct_transitive,
            "node_gated_raw_auroc": raw_direct_transitive,
        },
        "discovery_confirmation_induced_spearman": stability,
        "directed_score_diagnostics": {
            "discovery": discovery_diagnostics,
            "confirmation": confirmation_diagnostics,
        },
        "operational_query_accounting": operational,
        "validation_query_accounting": validation,
        "gates": gates,
    }


def score_complete_panel(
    panel: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    ground_truth_by_case: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Score both mandatory organisms; neither result controls the other."""
    if set(panel) != set(MANDATORY_PANEL_KEYS) or set(ground_truth_by_case) != set(
        PROSPECTIVE_CASE_IDS
    ):
        raise ValueError("scoring requires both mandatory cases and both splits")
    cases = {
        str(case_id): score_case_bundles(
            panel[(case_id, "discovery")],
            panel[(case_id, "confirmation")],
            ground_truth=ground_truth_by_case[case_id],
        )
        for case_id in PROSPECTIVE_CASE_IDS
    }
    return {
        "schema_version": "e32_2_node_first_directed_result_v1",
        "case_order": PROSPECTIVE_CASE_IDS,
        "case_results": cases,
        "overall_useful_pass": all(
            case["gates"]["useful_pass"] for case in cases.values()
        ),
        "overall_superiority_pass": all(
            case["gates"]["superiority_pass"] for case in cases.values()
        ),
        "multiplicity_policy": {
            "replication_rule": "conjunction_over_case2_and_case21",
            "best_eap_selection": "maximum_confirmation_auprc_favors_comparator",
            "pooled_or_casewise_p_values": False,
            "external_population_inference": False,
        },
    }


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def aggregate_actual_query_accounting(
    panel: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Sum observed acquisition work without mixing in operational estimates."""
    if set(panel) != set(MANDATORY_PANEL_KEYS):
        raise ValueError("actual accounting requires the complete mandatory panel")

    totals = {
        "physical_forwards": 0,
        "model_examples": 0,
        "no_grad_forwards": 0,
        "grad_enabled_forwards": 0,
        "backward_passes": 0,
    }
    by_split: dict[str, dict[str, int]] = {}
    for case_id, split in MANDATORY_PANEL_KEYS:
        accounting = panel[(case_id, split)].get("actual_query_accounting")
        if not isinstance(accounting, Mapping):
            raise TypeError("split lacks actual query accounting")
        if accounting.get("operational_budget_status") != (
            "reported_separately_by_protocol"
        ):
            raise ValueError("operational and actual accounting were mixed")

        raw = accounting.get("behavior_and_raw_atlas")
        directed = accounting.get("directed_replacement")
        eap = accounting.get("official_eap")
        eap_ig = accounting.get("official_eap_ig")
        if not all(isinstance(part, Mapping) for part in (raw, directed, eap, eap_ig)):
            raise ValueError("split accounting lane is missing")
        if not eap.get("matches_official_control_flow") or not eap_ig.get(
            "matches_official_control_flow"
        ):
            raise ValueError("official control flow was not observed exactly")

        split_totals = {
            "physical_forwards": _nonnegative_int(
                raw.get("physical_forwards"), label="raw physical_forwards"
            )
            + _nonnegative_int(
                directed.get("observed_physical_forwards"),
                label="directed observed_physical_forwards",
            )
            + _nonnegative_int(eap.get("physical_forwards"), label="EAP forwards")
            + _nonnegative_int(
                eap_ig.get("physical_forwards"), label="EAP-IG forwards"
            ),
            "model_examples": _nonnegative_int(
                raw.get("model_examples"), label="raw model_examples"
            )
            + _nonnegative_int(
                directed.get("observed_model_examples"),
                label="directed observed_model_examples",
            )
            + _nonnegative_int(eap.get("model_examples"), label="EAP examples")
            + _nonnegative_int(eap_ig.get("model_examples"), label="EAP-IG examples"),
            "no_grad_forwards": _nonnegative_int(
                raw.get("physical_forwards"), label="raw no-grad forwards"
            )
            + _nonnegative_int(
                directed.get("observed_no_grad_forwards"),
                label="directed observed_no_grad_forwards",
            )
            + _nonnegative_int(
                eap.get("no_grad_forwards"), label="EAP no-grad forwards"
            )
            + _nonnegative_int(
                eap_ig.get("no_grad_forwards"), label="EAP-IG no-grad forwards"
            ),
            "grad_enabled_forwards": _nonnegative_int(
                directed.get("observed_grad_enabled_forwards"),
                label="directed observed_grad_enabled_forwards",
            )
            + _nonnegative_int(
                eap.get("grad_enabled_forwards"), label="EAP grad forwards"
            )
            + _nonnegative_int(
                eap_ig.get("grad_enabled_forwards"), label="EAP-IG grad forwards"
            ),
            "backward_passes": _nonnegative_int(
                directed.get("backward_passes"), label="directed backward passes"
            )
            + _nonnegative_int(eap.get("backward_passes"), label="EAP backwards")
            + _nonnegative_int(eap_ig.get("backward_passes"), label="EAP-IG backwards"),
        }
        if (
            split_totals["no_grad_forwards"] + split_totals["grad_enabled_forwards"]
            != split_totals["physical_forwards"]
        ):
            raise ValueError("observed grad-mode counters do not partition forwards")
        by_split[f"case{case_id}_{split}"] = split_totals
        for name, value in split_totals.items():
            totals[name] += value

    return {
        "schema_version": "e32_2_actual_query_accounting_v1",
        **totals,
        "by_split": by_split,
        "operational_budget_included": False,
        "source": (
            "observed_forward_counters_plus_pinned_official_backward_control_flow"
        ),
        "backward_passes_observed_directly": False,
        "backward_passes_status": "inferred_from_verified_official_eap_control_flow",
    }


__all__ = [
    "MANDATORY_PANEL_KEYS",
    "MANDATORY_SPLITS",
    "aggregate_actual_query_accounting",
    "assert_complete_fresh_panel",
    "score_case_bundles",
    "score_complete_panel",
    "validate_case_bundle_pair",
]
