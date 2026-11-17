# ruff: noqa: EXE002
"""Pure contract tests for the generic E32.4 EAP baseline bridge."""

from __future__ import annotations

import pytest

from experiments.e32_interpbench.e32_4_eap import package_generic_eap
from experiments.e32_interpbench.eap_baseline import (
    FineEdgeScore,
    expected_auto_circuit_counts,
)


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
    records = []
    for source_module, source_head, source_layer in sources:
        for target_module, target_head, target_layer in destinations:
            if source_layer >= target_layer:
                continue
            source = _raw_node(source_module, source_head)
            target = _raw_node(target_module, target_head)
            records.append(FineEdgeScore(source, target, f"{source}->{target}", 0.1))
    return tuple(records)


@pytest.mark.parametrize(
    ("method", "forwards", "no_grad", "grad", "backwards"),
    (("eap", 2, 1, 1, 1), ("eap_ig", 7, 1, 6, 6)),
)
def test_generic_eap_contract_binds_method_graph_and_observed_work(
    method: str, forwards: int, no_grad: int, grad: int, backwards: int
) -> None:
    scores = _complete_fine_scores(n_layers=2, n_heads=4)
    expected = expected_auto_circuit_counts(2, 4)
    run = package_generic_eap(
        case_id=25,
        n_layers=2,
        n_heads=4,
        example_count=64,
        method=method,
        fine_scores=scores,
        observed_query_accounting={
            "physical_forwards": forwards,
            "model_examples": 64 * forwards,
            "no_grad_forwards": no_grad,
            "grad_enabled_forwards": grad,
            "backward_passes_inferred_from_pinned_control_flow": backwards,
        },
    )

    assert run.settings.method == method
    assert run.settings.integrated_grad_samples == (5 if method == "eap_ig" else None)
    assert run.grouping.internal_fine_edge_count == expected.internal_fine_edges
    assert len(run.internal_fine_scores) == expected.internal_fine_edges
    assert run.query_accounting.physical_model_forwards == forwards
    assert run.query_accounting.backward_passes == backwards


def test_generic_eap_rejects_counter_or_method_drift() -> None:
    scores = _complete_fine_scores(n_layers=2, n_heads=4)
    with pytest.raises(ValueError, match="method"):
        package_generic_eap(
            case_id=25,
            n_layers=2,
            n_heads=4,
            example_count=64,
            method="not_eap",
            fine_scores=scores,
            observed_query_accounting={},
        )
    with pytest.raises(RuntimeError, match="observed EAP"):
        package_generic_eap(
            case_id=25,
            n_layers=2,
            n_heads=4,
            example_count=64,
            method="eap",
            fine_scores=scores,
            observed_query_accounting={
                "physical_forwards": 3,
                "model_examples": 192,
                "no_grad_forwards": 1,
                "grad_enabled_forwards": 2,
                "backward_passes_inferred_from_pinned_control_flow": 2,
            },
        )
