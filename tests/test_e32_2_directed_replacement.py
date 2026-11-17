# ruff: noqa: EXE002
"""Tests for exact directed fine-edge replacement in E32.2."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch

from experiments.e32_interpbench.adapter import CASE_MANIFESTS
from experiments.e32_interpbench.e32_2_directed_replacement import (
    PromotedFineEdgeGroup,
    PromotedFineEdgeGrouping,
    _acquire_from_patchable,
    _AutoCircuitRuntime,
    build_promoted_fine_edge_grouping,
    directed_replacement_query_accounting,
    qkv_max_aggregate,
    recipient_target_cross_entropy,
    run_directed_replacement,
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

    scores = []
    for source_module, source_head, source_layer in sources:
        for target_module, target_head, target_layer in destinations:
            if source_layer >= target_layer:
                continue
            source = _raw_node(source_module, source_head)
            target = _raw_node(target_module, target_head)
            name = f"{source}->{target}"
            scores.append(FineEdgeScore(source, target, name, 0.0))
    return tuple(scores)


@pytest.mark.parametrize("case_id", (2, 21, 97))
def test_complete_graph_groups_match_pinned_auto_circuit_counts(case_id: int) -> None:
    manifest = CASE_MANIFESTS[case_id]
    fine_scores = _complete_fine_scores(
        n_layers=manifest.n_layers,
        n_heads=manifest.n_heads,
    )
    live = {fine.fine_edge_name: fine.fine_edge_name for fine in fine_scores}

    grouping = build_promoted_fine_edge_grouping(case_id, fine_scores, live)
    expected = expected_auto_circuit_counts(manifest.n_layers, manifest.n_heads)

    assert grouping.fine_edge_count == expected.fine_edges
    assert grouping.internal_fine_edge_count == expected.internal_fine_edges
    assert grouping.excluded_boundary_fine_edge_count == expected.boundary_fine_edges
    assert len(grouping.groups) == expected.internal_whole_edges
    assert sum(group.collision_count == 3 for group in grouping.groups) == (
        expected.internal_head_target_edges
    )
    assert sum(group.collision_count == 1 for group in grouping.groups) == (
        expected.internal_mlp_target_edges
    )
    for group in grouping.groups:
        if group.collision_count == 3:
            targets = tuple(name.rsplit("->", 1)[1] for name in group.fine_edge_names)
            assert tuple(
                next(letter for letter in "qkv" if f"hook_{letter}_input" in target)
                for target in targets
            ) == ("q", "k", "v")


def test_grouping_fails_closed_on_live_edge_or_qkv_mismatch() -> None:
    fine_scores = _complete_fine_scores(n_layers=3, n_heads=4)
    live = {fine.fine_edge_name: fine.fine_edge_name for fine in fine_scores}
    live.pop(next(iter(live)))
    with pytest.raises(ValueError, match="live fine-edge names"):
        build_promoted_fine_edge_grouping(97, fine_scores, live)

    chosen = next(
        index
        for index, fine in enumerate(fine_scores)
        if "hook_q_input" in fine.target and "hook_resid_pre" not in fine.source
    )
    malformed = list(fine_scores)
    fine = malformed[chosen]
    malformed[chosen] = FineEdgeScore(
        fine.source,
        fine.target.replace("hook_q_input", "hook_k_input"),
        fine.fine_edge_name,
        fine.signed_score,
    )
    with pytest.raises(ValueError):
        build_promoted_fine_edge_grouping(
            97,
            malformed,
            {item.fine_edge_name: item.fine_edge_name for item in malformed},
        )


def test_qkv_max_aggregation_preserves_promoted_order() -> None:
    grouping = _toy_grouping()
    values = torch.tensor(
        [
            [1.0, -9.0],
            [4.0, -2.0],
            [3.0, -5.0],
            [8.0, -7.0],
        ]
    )

    raw = qkv_max_aggregate(grouping, grouping.edge_order, values)
    absolute = qkv_max_aggregate(
        grouping,
        grouping.edge_order,
        values,
        absolute=True,
    )

    assert torch.equal(raw, torch.tensor([[4.0, -2.0], [8.0, -7.0]]))
    assert torch.equal(absolute, torch.tensor([[4.0, 9.0], [8.0, 7.0]]))


def test_recipient_target_cross_entropy_is_per_context_and_non_bos() -> None:
    logits = torch.tensor(
        [
            [[5.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            [[0.0, 0.0, 3.0], [2.0, 0.0, 0.0]],
        ]
    )
    targets = torch.nn.functional.one_hot(
        torch.tensor([[2, 0, 1], [1, 2, 0]]),
        num_classes=3,
    ).float()

    observed = recipient_target_cross_entropy(logits, targets)
    expected = (
        -torch.log_softmax(logits, dim=-1)
        .gather(
            -1,
            torch.tensor([[[0], [1]], [[2], [0]]]),
        )
        .squeeze(-1)
        .mean(dim=1)
    )

    assert observed.shape == (2,)
    assert torch.allclose(observed, expected)


@dataclass(frozen=True)
class _Dest:
    module_name: str


@dataclass(frozen=True)
class _ToyFineEdge:
    name: str
    patch_idx: tuple[int, ...]
    dest: _Dest


class _ToyPatchable(torch.nn.Module):
    def __init__(self, fine_edges: tuple[_ToyFineEdge, ...]) -> None:
        super().__init__()
        self.boundary = torch.nn.Identity()
        self.out_slice = (slice(None), slice(1, None))
        self.is_factorized = True
        self.separate_qkv = True
        self.seq_len = None
        self._fine_edges = fine_edges
        self._edge_index = {edge: index for index, edge in enumerate(fine_edges)}
        self._masks = {"dest": torch.zeros(len(fine_edges))}
        self._active_edges: tuple[_ToyFineEdge, ...] = ()
        self._donor: torch.Tensor | None = None

    def current_patch_masks_as_prune_scores(self) -> dict[str, torch.Tensor]:
        return self._masks

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        observed = self.boundary(inputs.float())
        class_count = CASE_MANIFESTS[97].d_vocab_out
        logits = (
            torch.nn.functional.one_hot(
                inputs.remainder(class_count),
                num_classes=class_count,
            ).float()
            * 2.0
        )
        if self._active_edges:
            assert self._donor is not None
            donor_delta = (self._donor[0] - observed).unsqueeze(-1)
            scale = sum(self._edge_index[edge] + 1 for edge in self._active_edges)
            class_direction = torch.linspace(-0.2, 0.2, class_count)
            logits = logits + donor_delta * class_direction * scale
        return logits


def _toy_grouping() -> PromotedFineEdgeGrouping:
    dest = _Dest("dest")
    fine_edges = tuple(
        _ToyFineEdge(f"fine-{index}", (index,), dest) for index in range(4)
    )
    groups = (
        PromotedFineEdgeGroup(
            promoted_edge=("source-a", "target-head"),
            fine_edge_names=tuple(edge.name for edge in fine_edges[:3]),
            fine_edges=fine_edges[:3],
            collision_count=3,
        ),
        PromotedFineEdgeGroup(
            promoted_edge=("source-b", "target-mlp"),
            fine_edge_names=(fine_edges[3].name,),
            fine_edges=(fine_edges[3],),
            collision_count=1,
        ),
    )
    return PromotedFineEdgeGrouping(
        case_id=97,
        edge_order=tuple(group.promoted_edge for group in groups),
        groups=groups,
        fine_edge_count=4,
        internal_fine_edge_count=4,
        excluded_boundary_fine_edge_count=0,
    )


def _toy_runtime() -> _AutoCircuitRuntime:
    def src_ablations(model: _ToyPatchable, inputs: torch.Tensor, _kind: object):
        model(inputs)
        return inputs.float().detach().clone().unsqueeze(0)

    @contextmanager
    def patch_mode(
        model: _ToyPatchable,
        donor: torch.Tensor,
        *,
        edges: tuple[_ToyFineEdge, ...],
    ):
        model._masks["dest"].zero_()
        for edge in edges:
            model._masks["dest"][edge.patch_idx] = 1.0
        model._active_edges = tuple(edges)
        model._donor = donor
        try:
            yield
        finally:
            model._active_edges = ()
            model._donor = None

    def set_all_masks(model: _ToyPatchable, value: float) -> None:
        model._masks["dest"].fill_(value)

    return _AutoCircuitRuntime(
        src_ablations=src_ablations,
        patch_mode=patch_mode,
        set_all_masks=set_all_masks,
        resample_ablation=object(),
    )


def test_acquisition_caches_sources_and_observes_every_forward_exactly() -> None:
    grouping = _toy_grouping()
    model = _ToyPatchable(
        tuple(edge for group in grouping.groups for edge in group.fine_edges)
    )
    clean_inputs = torch.tensor(
        [[0, 0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 2, 1, 3, 5, 7, 9, 11, 4, 6]]
    )
    corrupt_inputs = torch.tensor(
        [[0, 2, 2, 4, 4, 6, 6, 8, 8, 10], [0, 1, 0, 2, 4, 6, 8, 10, 3, 5]]
    )
    clean_targets = torch.nn.functional.one_hot(
        clean_inputs,
        num_classes=CASE_MANIFESTS[97].d_vocab_out,
    ).float()
    corrupt_targets = torch.nn.functional.one_hot(
        corrupt_inputs,
        num_classes=CASE_MANIFESTS[97].d_vocab_out,
    ).float()

    run = _acquire_from_patchable(
        case_id=97,
        auto_model=model,
        forward_boundary=model.boundary,
        grouping=grouping,
        requested_promoted_edges=grouping.edge_order,
        clean_inputs=clean_inputs,
        corrupt_inputs=corrupt_inputs,
        clean_targets=clean_targets,
        corrupt_targets=corrupt_targets,
        runtime=_toy_runtime(),
        include_grouped_promoted_sensitivity=True,
    )

    expected = directed_replacement_query_accounting(
        example_count=2,
        promoted_edge_count=2,
        fine_edge_count=4,
        include_grouped_promoted_sensitivity=True,
    )
    assert run.query_accounting == expected
    assert expected.source_cache_forwards == 2
    assert expected.baseline_forwards == 2
    assert expected.fine_replacement_forwards == 8
    assert expected.grouped_replacement_forwards == 4
    assert expected.expected_physical_forwards == 16
    assert expected.observed_physical_forwards == 16
    assert expected.observed_model_examples == 32
    assert expected.backward_passes == 0
    assert run.fine_clean_recipient_corrupt_donor_logits.shape == (4, 2, 9, 64)
    assert run.fine_corrupt_recipient_clean_donor_logits.shape == (4, 2, 9, 64)
    class_direction = torch.linspace(-0.2, 0.2, 64)
    clean_baseline = model(clean_inputs)[:, 1:]
    corrupt_baseline = model(corrupt_inputs)[:, 1:]
    expected_clean_replacement = (
        clean_baseline
        + ((corrupt_inputs - clean_inputs).float().unsqueeze(-1) * class_direction)[
            :, 1:
        ]
    )
    expected_corrupt_replacement = (
        corrupt_baseline
        + ((clean_inputs - corrupt_inputs).float().unsqueeze(-1) * class_direction)[
            :, 1:
        ]
    )
    assert torch.allclose(
        run.fine_clean_recipient_corrupt_donor_logits[0],
        expected_clean_replacement,
    )
    assert torch.allclose(
        run.fine_corrupt_recipient_clean_donor_logits[0],
        expected_corrupt_replacement,
    )
    assert run.clean_recipient_target_ce_deltas.shape == (4, 2)
    assert run.corrupt_recipient_target_ce_deltas.shape == (4, 2)
    assert run.grouped_clean_recipient_corrupt_donor_logits is not None
    assert run.grouped_clean_recipient_corrupt_donor_logits.shape == (2, 2, 9, 64)
    assert torch.equal(model._masks["dest"], torch.zeros(4))


def test_acquisition_requires_canonical_requested_edge_order() -> None:
    grouping = _toy_grouping()
    model = _ToyPatchable(
        tuple(edge for group in grouping.groups for edge in group.fine_edges)
    )
    inputs = torch.tensor([[0, 0, 1, 2, 3, 4, 5, 6, 7, 8]])
    targets = torch.nn.functional.one_hot(
        inputs,
        num_classes=CASE_MANIFESTS[97].d_vocab_out,
    ).float()
    with pytest.raises(ValueError, match="canonical subsequence"):
        _acquire_from_patchable(
            case_id=97,
            auto_model=model,
            forward_boundary=model.boundary,
            grouping=grouping,
            requested_promoted_edges=tuple(reversed(grouping.edge_order)),
            clean_inputs=inputs,
            corrupt_inputs=inputs.clone(),
            clean_targets=targets,
            corrupt_targets=targets.clone(),
            runtime=_toy_runtime(),
            include_grouped_promoted_sensitivity=False,
        )


def test_acquisition_rejects_a_nonzero_initial_mask_before_any_forward() -> None:
    grouping = _toy_grouping()
    model = _ToyPatchable(
        tuple(edge for group in grouping.groups for edge in group.fine_edges)
    )
    model._masks["dest"][0] = 1.0
    inputs = torch.tensor([[0, 0, 1, 2, 3, 4, 5, 6, 7, 8]])
    targets = torch.nn.functional.one_hot(
        inputs,
        num_classes=CASE_MANIFESTS[97].d_vocab_out,
    ).float()

    with pytest.raises(RuntimeError, match="active masks differ"):
        _acquire_from_patchable(
            case_id=97,
            auto_model=model,
            forward_boundary=model.boundary,
            grouping=grouping,
            requested_promoted_edges=grouping.edge_order,
            clean_inputs=inputs,
            corrupt_inputs=inputs.clone(),
            clean_targets=targets,
            corrupt_targets=targets.clone(),
            runtime=_toy_runtime(),
            include_grouped_promoted_sensitivity=False,
        )


@pytest.mark.parametrize("wrapped_signal", ("patch_masks", "patch_wrapper_module"))
def test_live_runner_rejects_an_already_wrapped_model_before_runtime_loading(
    wrapped_signal: str,
) -> None:
    class AlreadyWrapped(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if wrapped_signal == "patch_masks":
                self.patch_masks = torch.nn.ParameterDict()
            else:
                patch_wrapper_type = type(
                    "PatchWrapperImpl",
                    (torch.nn.Module,),
                    {
                        "__module__": "auto_circuit.utils.patch_wrapper",
                        "forward": lambda self, inputs: inputs,
                    },
                )
                self.wrapper = patch_wrapper_type()

    with pytest.raises(ValueError, match="fresh unwrapped model"):
        run_directed_replacement(
            case_id=97,
            model=AlreadyWrapped(),
            case=object(),
            eligible_pair=object(),
        )
