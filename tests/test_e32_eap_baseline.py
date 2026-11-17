# ruff: noqa: EXE002
"""Tests for the E32.2 official EAP/EAP-IG score adapter."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from experiments.e32_interpbench.adapter import CASE_MANIFESTS, PANEL_CASE_IDS
from experiments.e32_interpbench.eap_baseline import (
    AUTO_CIRCUIT_REVISION,
    OFFICIAL_EAP_IG_STEPS,
    OFFICIAL_EAP_SETTINGS,
    FineEdgeScore,
    eap_query_accounting,
    expected_auto_circuit_counts,
    extract_fine_edge_scores,
    official_eap_settings,
    promote_and_aggregate_fine_scores,
)


@dataclass(frozen=True)
class _GraphNode:
    module_name: str
    layer: int
    head_idx: int | None = None
    src_idx: int = 0
    min_src_idx: int = 0


@dataclass(frozen=True)
class _GraphEdge:
    src: _GraphNode
    dest: _GraphNode
    patch_idx: tuple[int, ...]
    name: str
    seq_idx: None = None


@dataclass
class _PatchableGraph:
    edges: set[_GraphEdge]


def _raw_node(module_name: str, head_idx: int | None) -> str:
    return module_name if head_idx is None else f"{module_name}[{head_idx}]"


def _complete_fine_scores(*, n_layers: int, n_heads: int) -> tuple[FineEdgeScore, ...]:
    sources: list[tuple[str, int | None, int]] = [("blocks.0.hook_resid_pre", None, 0)]
    for layer in range(n_layers):
        sources.extend(
            (f"blocks.{layer}.attn.hook_result", head, 2 * layer + 1)
            for head in range(n_heads)
        )
        sources.append((f"blocks.{layer}.hook_mlp_out", None, 2 * layer + 2))

    destinations: list[tuple[str, int | None, int]] = []
    for layer in range(n_layers):
        destinations.extend(
            (f"blocks.{layer}.hook_{letter}_input", head, 2 * layer + 1)
            for head in range(n_heads)
            for letter in "qkv"
        )
        destinations.append((f"blocks.{layer}.hook_mlp_in", None, 2 * layer + 2))
    destinations.append(
        (f"blocks.{n_layers - 1}.hook_resid_post", None, 2 * n_layers + 1)
    )

    scores: list[FineEdgeScore] = []
    for source_module, source_head, source_layer in sources:
        for target_module, target_head, target_layer in destinations:
            if source_layer >= target_layer:
                continue
            source = _raw_node(source_module, source_head)
            target = _raw_node(target_module, target_head)
            scores.append(
                FineEdgeScore(
                    source=source,
                    target=target,
                    fine_edge_name=f"{source}->{target}",
                    signed_score=0.0,
                )
            )
    return tuple(scores)


def test_official_provenance_and_settings_are_frozen() -> None:
    assert AUTO_CIRCUIT_REVISION == "0b74956bbe472cf961deb8ae0ef574f725daca96"
    assert OFFICIAL_EAP_IG_STEPS == 5
    assert OFFICIAL_EAP_SETTINGS == {
        "factorized": True,
        "slice_output": "not_first_seq",
        "separate_qkv": True,
        "grad_function": "logit",
        "classification_loss": "kl_div",
        "ablation_type": "resample_corrupt",
        "batching": "one_full_dataset_batch",
    }
    assert official_eap_settings("eap").mask_val == 0.0
    assert official_eap_settings("eap").integrated_grad_samples is None
    assert official_eap_settings("eap_ig").mask_val is None
    assert (
        official_eap_settings("eap_ig").integrated_grad_samples == OFFICIAL_EAP_IG_STEPS
    )


@pytest.mark.parametrize("case_id", PANEL_CASE_IDS)
def test_complete_fine_graph_promotes_to_exact_internal_universe(case_id: int) -> None:
    manifest = CASE_MANIFESTS[case_id]
    fine_scores = _complete_fine_scores(
        n_layers=manifest.n_layers, n_heads=manifest.n_heads
    )

    promoted = promote_and_aggregate_fine_scores(case_id, fine_scores)
    expected = expected_auto_circuit_counts(manifest.n_layers, manifest.n_heads)

    assert promoted.fine_edge_count == expected.fine_edges
    assert promoted.internal_fine_edge_count == expected.internal_fine_edges
    assert promoted.excluded_boundary_fine_edge_count == expected.boundary_fine_edges
    assert len(promoted.edge_order) == expected.internal_whole_edges
    assert promoted.coverage_complete
    assert set(promoted.collision_counts) <= {1, 3}
    assert promoted.collision_counts.count(3) == expected.internal_head_target_edges
    assert promoted.collision_counts.count(1) == expected.internal_mlp_target_edges


def test_collision_max_matches_official_signed_and_absolute_threshold_semantics() -> (
    None
):
    fine_scores = list(_complete_fine_scores(n_layers=4, n_heads=4))
    chosen = (
        "blocks.0.attn.hook_result[0]",
        "blocks.1.attn.hook_result[0]",
    )
    replacements = iter((-5.0, 2.0, 1.0))
    changed: list[FineEdgeScore] = []
    for fine in fine_scores:
        promoted_target = fine.target.replace("hook_q_input", "attn.hook_result")
        promoted_target = promoted_target.replace("hook_k_input", "attn.hook_result")
        promoted_target = promoted_target.replace("hook_v_input", "attn.hook_result")
        if (fine.source, promoted_target) == chosen:
            fine = FineEdgeScore(
                source=fine.source,
                target=fine.target,
                fine_edge_name=fine.fine_edge_name,
                signed_score=next(replacements),
            )
        changed.append(fine)

    promoted = promote_and_aggregate_fine_scores(2, changed)
    index = promoted.edge_order.index(chosen)

    assert promoted.signed_max_scores[index] == 2.0
    assert promoted.absolute_max_scores[index] == 5.0
    assert chosen not in promoted.selected_edges(threshold=3.0, absolute=False)
    assert chosen in promoted.selected_edges(threshold=3.0, absolute=True)


def test_incomplete_fine_graph_fails_closed() -> None:
    fine_scores = _complete_fine_scores(n_layers=3, n_heads=4)
    with pytest.raises(ValueError, match="fine-edge count"):
        promote_and_aggregate_fine_scores(97, fine_scores[:-1])


def test_extract_scores_uses_exact_patch_indices_and_is_deterministic() -> None:
    residual_source = _GraphNode(
        "blocks.0.hook_resid_pre", layer=0, head_idx=None, src_idx=0
    )
    source = _GraphNode("blocks.0.attn.hook_result", layer=1, head_idx=0, src_idx=1)
    destination = _GraphNode(
        "blocks.1.hook_q_input", layer=3, head_idx=0, min_src_idx=0
    )
    edge = _GraphEdge(
        src=source,
        dest=destination,
        patch_idx=(0, 1),
        name="A0.0->A1.0.Q",
    )
    residual_edge = _GraphEdge(
        src=residual_source,
        dest=destination,
        patch_idx=(0, 0),
        name="Resid Start->A1.0.Q",
    )
    graph = _PatchableGraph(edges={edge, residual_edge})
    scores = {"blocks.1.hook_q_input": torch.tensor([[1.0, -4.0]])}

    extracted = extract_fine_edge_scores(graph, scores)

    assert extracted == (
        FineEdgeScore(
            source="blocks.0.attn.hook_result[0]",
            target="blocks.1.hook_q_input[0]",
            fine_edge_name="A0.0->A1.0.Q",
            signed_score=-4.0,
        ),
        FineEdgeScore(
            source="blocks.0.hook_resid_pre",
            target="blocks.1.hook_q_input[0]",
            fine_edge_name="Resid Start->A1.0.Q",
            signed_score=1.0,
        ),
    )


def test_query_accounting_matches_official_mask_gradient_control_flow() -> None:
    eap = eap_query_accounting(example_count=2, method="eap")
    integrated = eap_query_accounting(
        example_count=2, method="eap_ig", integrated_grad_samples=5
    )

    assert eap.source_ablation_forwards == 1
    assert eap.gradient_forwards == 1
    assert eap.backward_passes == 1
    assert eap.physical_model_forwards == 2
    assert eap.model_examples == 4
    assert integrated.source_ablation_forwards == 1
    assert integrated.gradient_forwards == 6
    assert integrated.backward_passes == 6
    assert integrated.physical_model_forwards == 7
    assert integrated.model_examples == 14
    assert integrated.gradient_examples == 12
    assert integrated.gradient_reduction == "sum_not_mean_over_k_plus_one_points"
