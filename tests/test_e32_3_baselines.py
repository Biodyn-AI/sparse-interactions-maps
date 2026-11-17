# ruff: noqa: EXE002
"""Toy tests for generic E32.3 exact-fine and EAP-IG5 bridges."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch

from experiments.e32_interpbench.e32_2_directed_replacement import (
    _AutoCircuitRuntime,
)
from experiments.e32_interpbench.e32_3_baselines import (
    _acquire_generic_exact_from_patchable,
    build_generic_fine_edge_grouping,
    package_generic_eap_ig5,
)
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
    ("case_id", "n_layers", "expected_nodes", "expected_whole"),
    ((25, 2, 10, 33), (58, 3, 15, 87)),
)
def test_generic_grouping_supports_variable_new_case_graphs(
    case_id: int, n_layers: int, expected_nodes: int, expected_whole: int
) -> None:
    scores = _complete_fine_scores(n_layers=n_layers, n_heads=4)
    live = {score.fine_edge_name: score.fine_edge_name for score in scores}

    grouping = build_generic_fine_edge_grouping(
        case_id=case_id,
        n_layers=n_layers,
        n_heads=4,
        fine_scores=scores,
        live_edges_by_name=live,
    )
    expected = expected_auto_circuit_counts(n_layers, 4)

    assert expected_nodes == n_layers * 5
    assert grouping.fine_edge_count == expected.fine_edges
    assert grouping.internal_fine_edge_count == expected.internal_fine_edges
    assert grouping.excluded_boundary_fine_edge_count == expected.boundary_fine_edges
    assert len(grouping.groups) == expected_whole == expected.internal_whole_edges
    assert tuple(group.collision_count for group in grouping.groups).count(3) == (
        expected.internal_head_target_edges
    )
    for group in grouping.groups:
        if group.collision_count == 3:
            targets = tuple(name.rsplit("->", 1)[1] for name in group.fine_edge_names)
            assert tuple(
                next(letter for letter in "qkv" if f"hook_{letter}_input" in target)
                for target in targets
            ) == ("q", "k", "v")


def test_generic_grouping_fails_closed_on_missing_live_edge() -> None:
    scores = _complete_fine_scores(n_layers=2, n_heads=4)
    live = {score.fine_edge_name: score.fine_edge_name for score in scores}
    live.pop(next(iter(live)))

    with pytest.raises(ValueError, match="live fine-edge names"):
        build_generic_fine_edge_grouping(
            case_id=25,
            n_layers=2,
            n_heads=4,
            fine_scores=scores,
            live_edges_by_name=live,
        )


@dataclass(frozen=True)
class _Dest:
    module_name: str


@dataclass(frozen=True)
class _Fine:
    name: str
    patch_idx: tuple[int, ...]
    dest: _Dest


class _Patchable(torch.nn.Module):
    def __init__(self, fine_edges: tuple[_Fine, ...], classes: int) -> None:
        super().__init__()
        self.boundary = torch.nn.Identity()
        self.out_slice = (slice(None), slice(1, None))
        self.is_factorized = True
        self.separate_qkv = True
        self.seq_len = None
        self.classes = classes
        self._fine_edges = fine_edges
        self._edge_index = {edge: index for index, edge in enumerate(fine_edges)}
        self._masks = {"dest": torch.zeros(len(fine_edges))}
        self._active: tuple[_Fine, ...] = ()
        self._donor: torch.Tensor | None = None

    def current_patch_masks_as_prune_scores(self) -> dict[str, torch.Tensor]:
        return self._masks

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        observed = self.boundary(inputs.float())
        logits = torch.nn.functional.one_hot(
            inputs.remainder(self.classes), num_classes=self.classes
        ).float()
        if self._active:
            assert self._donor is not None
            direction = torch.linspace(-0.3, 0.3, self.classes)
            scale = sum(self._edge_index[edge] + 1 for edge in self._active)
            logits = (
                logits + (self._donor[0] - observed).unsqueeze(-1) * direction * scale
            )
        return logits


def _toy_grouping(case_id: int):
    from experiments.e32_interpbench.e32_2_directed_replacement import (
        PromotedFineEdgeGroup,
        PromotedFineEdgeGrouping,
    )

    dest = _Dest("dest")
    edges = tuple(_Fine(f"f{index}", (index,), dest) for index in range(4))
    groups = (
        PromotedFineEdgeGroup(("a", "head"), ("f0", "f1", "f2"), edges[:3], 3),
        PromotedFineEdgeGroup(("b", "mlp"), ("f3",), edges[3:], 1),
    )
    return PromotedFineEdgeGrouping(
        case_id=case_id,
        edge_order=tuple(group.promoted_edge for group in groups),
        groups=groups,
        fine_edge_count=4,
        internal_fine_edge_count=4,
        excluded_boundary_fine_edge_count=0,
    )


def _toy_runtime() -> _AutoCircuitRuntime:
    def src_ablations(model: _Patchable, inputs: torch.Tensor, _kind: object):
        model(inputs)
        return inputs.float().clone().unsqueeze(0)

    @contextmanager
    def patch_mode(model: _Patchable, donor: torch.Tensor, *, edges: tuple[_Fine, ...]):
        model._masks["dest"].zero_()
        for edge in edges:
            model._masks["dest"][edge.patch_idx] = 1.0
        model._active = edges
        model._donor = donor
        try:
            yield
        finally:
            model._active = ()
            model._donor = None

    def set_all_masks(model: _Patchable, value: float) -> None:
        model._masks["dest"].fill_(value)

    return _AutoCircuitRuntime(
        src_ablations=src_ablations,
        patch_mode=patch_mode,
        set_all_masks=set_all_masks,
        resample_ablation=object(),
    )


def test_generic_exact_bridge_observes_both_orientations_and_separate_fine_edges() -> (
    None
):
    grouping = _toy_grouping(58)
    fine_edges = tuple(edge for group in grouping.groups for edge in group.fine_edges)
    model = _Patchable(fine_edges, classes=11)
    clean = torch.tensor([[0, 1, 2], [0, 3, 4]])
    corrupt = torch.tensor([[0, 5, 6], [0, 7, 8]])

    run = _acquire_generic_exact_from_patchable(
        case_id=58,
        auto_model=model,
        forward_boundary=model.boundary,
        grouping=grouping,
        clean_inputs=clean,
        corrupt_inputs=corrupt,
        expected_sequence_length=3,
        expected_classes=11,
        runtime=_toy_runtime(),
    )

    assert run.fine_edge_names == ("f0", "f1", "f2", "f3")
    assert run.fine_to_whole_indices == (0, 0, 0, 1)
    assert run.corrupt_into_clean_outputs.shape == (4, 2, 2, 11)
    assert run.clean_into_corrupt_outputs.shape == (4, 2, 2, 11)
    assert run.query_accounting.expected_physical_forwards == 12
    assert run.query_accounting.observed_physical_forwards == 12
    assert run.query_accounting.observed_model_examples == 24
    assert run.query_accounting.backward_passes == 0
    assert torch.allclose(run.clean_baseline_output.mean(dim=-1), torch.zeros(2, 2))
    assert torch.equal(model._masks["dest"], torch.zeros(4))


def test_generic_eap_ig5_contract_binds_settings_grouping_and_observed_work() -> None:
    scores = _complete_fine_scores(n_layers=2, n_heads=4)
    expected = expected_auto_circuit_counts(2, 4)
    observed = {
        "physical_forwards": 7,
        "model_examples": 64 * 7,
        "no_grad_forwards": 1,
        "grad_enabled_forwards": 6,
        "backward_passes_inferred_from_pinned_control_flow": 6,
    }

    run = package_generic_eap_ig5(
        case_id=25,
        n_layers=2,
        n_heads=4,
        example_count=64,
        fine_scores=scores,
        observed_query_accounting=observed,
    )

    assert run.settings.method == "eap_ig"
    assert run.settings.integrated_grad_samples == 5
    assert run.query_accounting.physical_model_forwards == 7
    assert run.query_accounting.backward_passes == 6
    assert run.observed_query_accounting == observed
    assert "backward_passes" not in run.observed_query_accounting
    assert run.grouping.internal_fine_edge_count == expected.internal_fine_edges
    assert run.grouping.fine_edge_count == expected.fine_edges
    assert len(run.all_fine_scores) == expected.fine_edges
    assert len(run.internal_fine_scores) == expected.internal_fine_edges
    assert len(run.fine_to_whole_indices) == expected.internal_fine_edges

    with pytest.raises(RuntimeError, match="observed EAP-IG5 work"):
        package_generic_eap_ig5(
            case_id=25,
            n_layers=2,
            n_heads=4,
            example_count=64,
            fine_scores=scores,
            observed_query_accounting={**observed, "grad_enabled_forwards": 5},
        )
