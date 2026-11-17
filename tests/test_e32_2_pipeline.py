# ruff: noqa: EXE002
"""Synthetic full-graph tests for E32.2 bundle validation and scoring."""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from experiments.e32_interpbench.adapter import (
    CASE_MANIFESTS,
    patchable_internal_universe,
    promote_official_node,
)
from experiments.e32_interpbench.component_patching import (
    build_component_atlas,
    build_patch_units,
    vector_pair_interactions,
)
from experiments.e32_interpbench.e32_2_directed_replacement import (
    build_promoted_fine_edge_grouping,
    directed_replacement_query_accounting,
)
from experiments.e32_interpbench.e32_2_orchestration import (
    score_case_bundles,
    score_complete_panel,
    validate_case_bundle_pair,
)
from experiments.e32_interpbench.e32_2_protocol import (
    expected_case_provenance,
    split_request,
)
from experiments.e32_interpbench.e32_2_run import (
    EXPECTED_GROUND_TRUTH_IDENTITIES,
)
from experiments.e32_interpbench.eap_baseline import (
    FineEdgeScore,
    OfficialEAPRun,
    eap_query_accounting,
    official_eap_run_to_dict,
    official_eap_settings,
    promote_and_aggregate_fine_scores,
)


def _raw_node(module_name: str, head_index: int | None) -> str:
    return module_name if head_index is None else f"{module_name}[{head_index}]"


def _complete_fine_scores(case_id: int) -> tuple[FineEdgeScore, ...]:
    manifest = CASE_MANIFESTS[case_id]
    truth = set(EXPECTED_GROUND_TRUTH_IDENTITIES[case_id]["edges"])
    sources: list[tuple[str, int | None, int]] = [("blocks.0.hook_resid_pre", None, 0)]
    for layer in range(manifest.n_layers):
        sources.extend(
            (f"blocks.{layer}.attn.hook_result", head, 2 * layer + 1)
            for head in range(manifest.n_heads)
        )
        sources.append((f"blocks.{layer}.hook_mlp_out", None, 2 * layer + 2))
    destinations: list[tuple[str, int | None, int]] = []
    for layer in range(manifest.n_layers):
        destinations.extend(
            (f"blocks.{layer}.hook_{letter}_input", head, 2 * layer + 1)
            for head in range(manifest.n_heads)
            for letter in "qkv"
        )
        destinations.append((f"blocks.{layer}.hook_mlp_in", None, 2 * layer + 2))
    destinations.append(
        (
            f"blocks.{manifest.n_layers - 1}.hook_resid_post",
            None,
            2 * manifest.n_layers + 1,
        )
    )

    records: list[FineEdgeScore] = []
    for source_module, source_head, source_layer in sources:
        for target_module, target_head, target_layer in destinations:
            if source_layer >= target_layer:
                continue
            source = _raw_node(source_module, source_head)
            target = _raw_node(target_module, target_head)
            promoted = (promote_official_node(source), promote_official_node(target))
            records.append(
                FineEdgeScore(
                    source=source,
                    target=target,
                    fine_edge_name=f"{source}->{target}",
                    signed_score=0.9 if promoted in truth else -0.1,
                )
            )
    return tuple(records)


def _eap_envelope(case_id: int, method: str, example_count: int) -> dict[str, object]:
    fine = _complete_fine_scores(case_id)
    promoted = promote_and_aggregate_fine_scores(case_id, fine)
    accounting = eap_query_accounting(example_count=example_count, method=method)
    run = OfficialEAPRun(
        case_id=case_id,
        settings=official_eap_settings(method),
        fine_scores=fine,
        promoted_scores=promoted,
        query_accounting=accounting,
    )
    return {
        "run": official_eap_run_to_dict(run),
        "observed_query_accounting": {
            "physical_forwards": accounting.physical_model_forwards,
            "model_examples": accounting.model_examples,
            "no_grad_forwards": accounting.source_ablation_forwards,
            "grad_enabled_forwards": accounting.gradient_forwards,
            "backward_passes": accounting.backward_passes,
            "matches_official_control_flow": True,
        },
    }


def _raw_atlas(
    case_id: int, split: str, *, selected_nodes: tuple[str, ...]
) -> tuple[dict[str, object], tuple[float, ...]]:
    request = split_request(case_id, split)
    manifest = CASE_MANIFESTS[case_id]
    nodes, edges = patchable_internal_universe(manifest.n_layers, manifest.n_heads)
    atlas = build_component_atlas(
        build_patch_units(manifest.n_layers, manifest.n_heads),
        valid_directed_edges=edges,
    )
    batch = request.required_count
    positions = manifest.n_ctx - 1
    classes = manifest.d_vocab_out
    empty = torch.zeros(batch, positions, classes)
    singletons = torch.zeros(len(nodes), batch, positions, classes)
    selected_rank = {
        node: len(selected_nodes) - index for index, node in enumerate(selected_nodes)
    }
    for index, node in enumerate(nodes):
        magnitude = float(selected_rank.get(node, 0) + 1)
        singletons[index, ..., 0] = magnitude
        singletons[index, ..., 1] = -magnitude
    truth = set(EXPECTED_GROUND_TRUTH_IDENTITIES[case_id]["edges"])
    pair_interactions = torch.zeros(len(edges), batch, positions, classes)
    for index, edge in enumerate(edges):
        magnitude = 2.0 if edge in truth else 0.1
        pair_interactions[index, ..., 0] = magnitude
        pair_interactions[index, ..., 1] = -magnitude
    node_index = {node: index for index, node in enumerate(nodes)}
    pairs = torch.stack(
        [
            singletons[node_index[source]]
            + singletons[node_index[target]]
            + pair_interactions[index]
            for index, (source, target) in enumerate(edges)
        ]
    )
    pair_interactions = vector_pair_interactions(
        empty_output=empty,
        singleton_outputs=singletons,
        pair_outputs=pairs,
        pair_component_indices=atlas.pair_component_indices,
        center=False,
    )
    scientific = torch.cat((empty.unsqueeze(0), singletons, pairs), dim=0)
    raw_conditions = torch.zeros(len(scientific), batch, manifest.n_ctx, classes)
    raw_conditions[..., 1:, :] = scientific
    clean_inputs = torch.arange(batch * manifest.n_ctx).reshape(
        batch, manifest.n_ctx
    ) + (10_000 if split == "discovery" else 100_000)
    corrupt_inputs = clean_inputs + 1_000_000

    smoke_forwards = (
        request.smoke_condition_count
        + (request.smoke_condition_count + request.smoke_condition_batch_size - 1)
        // request.smoke_condition_batch_size
    )
    patched_count = len(nodes) + len(edges)
    atlas_forwards = (
        patched_count + request.atlas_condition_batch_size - 1
    ) // request.atlas_condition_batch_size
    live_forwards = 2 + smoke_forwards + atlas_forwards
    smoke_examples = 2 * request.smoke_condition_count * batch
    atlas_examples = patched_count * batch
    live_examples = 2 * batch + smoke_examples + atlas_examples
    energies = tuple(
        float(value)
        for value in (singletons.double() - empty.double().unsqueeze(0))
        .square()
        .mean(dim=(1, 2, 3))
    )
    return {
        "schema_version": "e32_interpbench_component_atlas_v2",
        "case_id": case_id,
        "component_ids": nodes,
        "directed_candidate_edges": edges,
        "scientific_output_coordinate": "class_centered_logits_non_bos_positions",
        "condition_ids": atlas.condition_ids,
        "condition_masks": atlas.condition_masks,
        "pair_component_indices": atlas.pair_component_indices,
        "condition_outputs": raw_conditions,
        "scientific_condition_outputs": scientific,
        "scientific_singleton_outputs": singletons,
        "scientific_pair_outputs": pairs,
        "pair_interactions": pair_interactions,
        "clean_inputs": clean_inputs,
        "corrupt_inputs": corrupt_inputs,
        "behavioral_pair_screen": {"all_pairs_correct": True},
        "batch_invariance_smoke": {
            "passed": True,
            "condition_count": request.smoke_condition_count,
            "example_count_per_condition": batch,
            "reference_condition_batch_size": 1,
            "candidate_condition_batch_size": request.smoke_condition_batch_size,
            "model_examples": smoke_examples,
            "physical_forwards": smoke_forwards,
        },
        "eligibility_query_accounting": {
            "physical_forwards_before_runner": 2,
            "model_examples_before_runner": 2 * request.candidate_count,
        },
        "atlas_query_accounting": {
            "patched_condition_count": patched_count,
            "example_count_per_condition": batch,
            "patched_model_examples": atlas_examples,
            "patched_physical_forwards": atlas_forwards,
        },
        "total_live_query_accounting": {
            "model_examples": live_examples,
            "physical_forwards": live_forwards,
        },
        "total_query_accounting_including_eligibility": {
            "model_examples": live_examples + 2 * request.candidate_count,
            "physical_forwards": live_forwards + 2,
        },
    }, energies


def _directed_envelope(
    case_id: int,
    split: str,
    *,
    requested_edges: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    request = split_request(case_id, split)
    manifest = CASE_MANIFESTS[case_id]
    fine = _complete_fine_scores(case_id)
    grouping = build_promoted_fine_edge_grouping(
        case_id, fine, {record.fine_edge_name: record.fine_edge_name for record in fine}
    )
    by_edge = {group.promoted_edge: group for group in grouping.groups}
    groups = tuple(by_edge[edge] for edge in requested_edges)
    fine_names = tuple(name for group in groups for name in group.fine_edge_names)
    fine_to_promoted = tuple(
        index for index, group in enumerate(groups) for _ in group.fine_edge_names
    )
    batch = request.required_count
    shape = (batch, manifest.n_ctx - 1, manifest.d_vocab_out)
    clean = torch.zeros(shape)
    corrupt = torch.zeros(shape)
    clean[..., 0] = 1.0
    clean[..., 1] = -1.0
    corrupt[..., 0] = -1.0
    corrupt[..., 1] = 1.0
    gap = clean - corrupt
    truth = set(EXPECTED_GROUND_TRUTH_IDENTITIES[case_id]["edges"])
    sufficiency: list[torch.Tensor] = []
    necessity: list[torch.Tensor] = []
    for edge, group in zip(requested_edges, groups, strict=True):
        if split == "discovery":
            sufficiency_value = 0.8 if edge in truth else 0.2
            necessity_value = 0.5 if edge in truth else 0.1
        else:
            sufficiency_value = 0.6 if edge in truth else 0.3
            necessity_value = 0.4 if edge in truth else 0.2
        for _name in group.fine_edge_names:
            sufficiency.append(corrupt + sufficiency_value * gap)
            necessity.append(clean - necessity_value * gap)
    accounting = directed_replacement_query_accounting(
        example_count=batch,
        promoted_edge_count=len(requested_edges),
        fine_edge_count=len(fine_names),
        include_grouped_promoted_sensitivity=False,
    )
    return {
        "schema_version": "e32_2_directed_replacement_envelope_v1",
        "case_id": case_id,
        "settings": {
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
        },
        "grouping": {
            "case_id": case_id,
            "edge_order": grouping.edge_order,
            "groups": tuple(
                {
                    "promoted_edge": group.promoted_edge,
                    "fine_edge_names": group.fine_edge_names,
                    "collision_count": group.collision_count,
                }
                for group in grouping.groups
            ),
            "fine_edge_count": grouping.fine_edge_count,
            "internal_fine_edge_count": grouping.internal_fine_edge_count,
            "excluded_boundary_fine_edge_count": grouping.excluded_boundary_fine_edge_count,
        },
        "requested_promoted_edges": requested_edges,
        "fine_edge_names": fine_names,
        "fine_to_promoted_indices": fine_to_promoted,
        "clean_baseline_logits": clean,
        "corrupt_baseline_logits": corrupt,
        "fine_clean_recipient_corrupt_donor_logits": torch.stack(necessity),
        "fine_corrupt_recipient_clean_donor_logits": torch.stack(sufficiency),
        "clean_recipient_target_ce_deltas": torch.zeros(len(fine_names), batch),
        "corrupt_recipient_target_ce_deltas": torch.zeros(len(fine_names), batch),
        "grouped_clean_recipient_corrupt_donor_logits": None,
        "grouped_corrupt_recipient_clean_donor_logits": None,
        "grouped_clean_recipient_target_ce_deltas": None,
        "grouped_corrupt_recipient_target_ce_deltas": None,
        "query_accounting": asdict(accounting),
    }


def _bundle(
    case_id: int,
    split: str,
    *,
    frozen_nodes: tuple[str, ...],
    induced_edges: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    request = split_request(case_id, split)
    raw, energies = _raw_atlas(case_id, split, selected_nodes=frozen_nodes)
    requested = (
        induced_edges
        if split == "discovery"
        else tuple(raw["directed_candidate_edges"])
    )
    directed = _directed_envelope(case_id, split, requested_edges=requested)
    eap = _eap_envelope(case_id, "eap", request.required_count)
    eap_ig = _eap_envelope(case_id, "eap_ig", request.required_count)
    rows = tuple(
        tuple(int(token) for token in row)
        for key in ("clean_inputs", "corrupt_inputs")
        for row in raw[key].tolist()
    )
    actual = {
        "behavior_and_raw_atlas": dict(
            raw["total_query_accounting_including_eligibility"]
        ),
        "directed_replacement": dict(directed["query_accounting"]),
        "official_eap": dict(eap["observed_query_accounting"]),
        "official_eap_ig": dict(eap_ig["observed_query_accounting"]),
        "operational_budget_status": "reported_separately_by_protocol",
    }
    return {
        "schema_version": "e32_2_split_acquisition_v1",
        "case_id": case_id,
        "split": split,
        "request": asdict(request),
        "row_custody": {"overlap_count": 0},
        "selected_token_rows": rows,
        "raw_pair_atlas": raw,
        "split_singleton_node_energies": energies,
        "split_top_half_nodes": frozen_nodes,
        "discovery_frozen_nodes": frozen_nodes,
        "discovery_induced_edges": induced_edges,
        "directed_replacement": directed,
        "official_eap": eap,
        "official_eap_ig": eap_ig,
        "actual_query_accounting": actual,
    }


@pytest.fixture(scope="module")
def case21_bundles() -> tuple[dict[str, object], dict[str, object]]:
    case_id = 21
    nodes, edges = patchable_internal_universe(4, 4)
    truth_nodes = EXPECTED_GROUND_TRUTH_IDENTITIES[case_id]["nodes"]
    frozen = (*truth_nodes, *(node for node in nodes if node not in truth_nodes))[:10]
    selected = set(frozen)
    induced = tuple(
        edge for edge in edges if edge[0] in selected and edge[1] in selected
    )
    return (
        _bundle(case_id, "discovery", frozen_nodes=frozen, induced_edges=induced),
        _bundle(case_id, "confirmation", frozen_nodes=frozen, induced_edges=induced),
    )


def _ground_truth(case_id: int) -> dict[str, object]:
    identities = EXPECTED_GROUND_TRUTH_IDENTITIES[case_id]
    return {
        "case_id": case_id,
        "node_ids": identities["nodes"],
        "directed_edges": identities["edges"],
        "raw_edges_sha256": expected_case_provenance(case_id)["artifact_sha256"][
            "edges.pkl"
        ],
    }


def test_full_case_bundle_validates_and_scores_every_frozen_method(
    case21_bundles: tuple[dict[str, object], dict[str, object]],
) -> None:
    discovery, confirmation = case21_bundles

    structure = validate_case_bundle_pair(
        discovery, confirmation, open_scientific_outcomes=True
    )
    result = score_case_bundles(discovery, confirmation, ground_truth=_ground_truth(21))

    assert structure["confirmation_fine_edge_count"] == 406
    assert tuple(result["confirmation_metrics"]) == (
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
    vectors = result["full_universe_score_vectors"]
    assert len(vectors["edge_order"]) == 166
    assert all(len(scores) == 166 for scores in vectors["confirmation"].values())
    induced = set(discovery["discovery_induced_edges"])
    primary = vectors["confirmation"]["bidirectional_fine_replacement"]
    exhaustive = vectors["confirmation"]["exhaustive_bidirectional_fine_replacement"]
    discovery_primary = vectors["discovery"]["bidirectional_fine_replacement"]
    for index, edge in enumerate(vectors["edge_order"]):
        if edge in induced:
            assert primary[index] == pytest.approx(exhaustive[index])
            assert discovery_primary[index] != pytest.approx(exhaustive[index])
        else:
            assert primary[index] == -1.0
            assert discovery_primary[index] == -1.0
    truth = set(EXPECTED_GROUND_TRUTH_IDENTITIES[21]["edges"])
    nontruth_index = next(
        index for index, edge in enumerate(vectors["edge_order"]) if edge not in truth
    )
    assert vectors["confirmation"]["eap_signed_max"][nontruth_index] == pytest.approx(
        -0.1
    )
    assert vectors["confirmation"]["eap_absolute_max"][nontruth_index] == pytest.approx(
        0.1
    )
    assert vectors["confirmation"]["eap_ig_signed_max"][
        nontruth_index
    ] == pytest.approx(-0.1)
    assert vectors["confirmation"]["eap_ig_absolute_max"][
        nontruth_index
    ] == pytest.approx(0.1)
    assert result["structure"]["direct_edge_retention"] == 1.0
    assert result["discovery_confirmation_induced_spearman"] == 1.0
    diagnostics = result["directed_score_diagnostics"]["discovery"]
    # Confirms the serialized recipient-first fields were routed to the
    # scorer's donor-first sufficiency/necessity arguments in the right order.
    assert max(diagnostics["sufficiency_mean_by_fine"]) == pytest.approx(0.8)
    assert max(diagnostics["necessity_mean_by_fine"]) == pytest.approx(0.5)


def test_bundle_validator_rejects_shortlist_group_counter_and_eap_tamper(
    case21_bundles: tuple[dict[str, object], dict[str, object]],
) -> None:
    discovery, confirmation = case21_bundles

    wrong_shortlist = dict(confirmation)
    wrong_shortlist["discovery_frozen_nodes"] = tuple(
        reversed(confirmation["discovery_frozen_nodes"])
    )
    with pytest.raises(ValueError, match="wrong discovery shortlist"):
        validate_case_bundle_pair(discovery, wrong_shortlist)

    wrong_group = dict(discovery)
    wrong_directed = dict(discovery["directed_replacement"])
    wrong_grouping = dict(wrong_directed["grouping"])
    groups = list(wrong_grouping["groups"])
    groups[0] = {**groups[0], "collision_count": 99}
    wrong_grouping["groups"] = tuple(groups)
    wrong_directed["grouping"] = wrong_grouping
    wrong_group["directed_replacement"] = wrong_directed
    with pytest.raises(ValueError, match="fine/group metadata"):
        validate_case_bundle_pair(wrong_group, confirmation)

    wrong_fine_axis = dict(discovery)
    wrong_directed = dict(discovery["directed_replacement"])
    fine_names = list(wrong_directed["fine_edge_names"])
    fine_names[:2] = reversed(fine_names[:2])
    wrong_directed["fine_edge_names"] = tuple(fine_names)
    wrong_fine_axis["directed_replacement"] = wrong_directed
    with pytest.raises(ValueError, match="fine/group metadata"):
        validate_case_bundle_pair(wrong_fine_axis, confirmation)

    wrong_counter = dict(discovery)
    wrong_directed = dict(discovery["directed_replacement"])
    wrong_accounting = dict(wrong_directed["query_accounting"])
    wrong_accounting["observed_model_examples"] += 1
    wrong_directed["query_accounting"] = wrong_accounting
    wrong_counter["directed_replacement"] = wrong_directed
    with pytest.raises(ValueError, match="query-accounting mismatch"):
        validate_case_bundle_pair(wrong_counter, confirmation)

    wrong_eap = dict(discovery)
    eap = dict(discovery["official_eap"])
    run = dict(eap["run"])
    promoted = dict(run["promoted_scores"])
    signed = list(promoted["signed_max_scores"])
    signed[0] += 0.25
    promoted["signed_max_scores"] = signed
    run["promoted_scores"] = promoted
    eap["run"] = run
    wrong_eap["official_eap"] = eap
    with pytest.raises(ValueError, match="promotion differs"):
        validate_case_bundle_pair(
            wrong_eap, confirmation, open_scientific_outcomes=True
        )


def test_complete_panel_is_conjunctive_and_requires_all_four_bundles(
    monkeypatch: pytest.MonkeyPatch,
    case21_bundles: tuple[dict[str, object], dict[str, object]],
) -> None:
    discovery, confirmation = case21_bundles
    panel = {
        (2, "discovery"): discovery,
        (2, "confirmation"): confirmation,
        (21, "discovery"): discovery,
        (21, "confirmation"): confirmation,
    }
    calls: list[int] = []

    def fake_score(
        _discovery: object, _confirmation: object, *, ground_truth: dict[str, int]
    ) -> dict[str, object]:
        case_id = ground_truth["case_id"]
        calls.append(case_id)
        return {
            "case_id": case_id,
            "gates": {
                "useful_pass": case_id == 2,
                "superiority_pass": case_id == 2,
            },
        }

    monkeypatch.setattr(
        "experiments.e32_interpbench.e32_2_orchestration.score_case_bundles",
        fake_score,
    )
    result = score_complete_panel(
        panel,
        ground_truth_by_case={2: {"case_id": 2}, 21: {"case_id": 21}},
    )

    assert calls == [2, 21]
    assert result["overall_useful_pass"] is False
    assert result["overall_superiority_pass"] is False
    missing = dict(panel)
    missing.pop((21, "confirmation"))
    with pytest.raises(ValueError, match="both mandatory"):
        score_complete_panel(
            missing,
            ground_truth_by_case={2: {"case_id": 2}, 21: {"case_id": 21}},
        )
