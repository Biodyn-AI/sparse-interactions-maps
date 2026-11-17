# ruff: noqa: EXE002
"""Tests for E32.4 simultaneous directed edge-set acquisition."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch

from experiments.e32_interpbench.e32_2_directed_replacement import (
    PromotedFineEdgeGroup,
    PromotedFineEdgeGrouping,
)
from experiments.e32_interpbench.e32_3_cohort import case_universe
from experiments.e32_interpbench.e32_4_edge_sets import (
    EdgeAblation,
    EdgeSetRuntime,
    FrozenEdgeSupport,
    _acquire_core_singletons_from_patchable,
    _acquire_edge_sets_from_patchable,
    build_edge_condition_plan,
    build_induced_edge_universe,
    run_core_singleton_replacement,
    run_edge_set_replacement,
)


@dataclass(frozen=True)
class _Dest:
    module_name: str


@dataclass(frozen=True)
class _FineEdge:
    name: str
    patch_idx: tuple[int, ...]
    dest: _Dest


def _toy_grouping(
    *, duplicate_coordinate: bool = False, invalid_coordinate: bool = False
) -> PromotedFineEdgeGrouping:
    dest = _Dest("dest")
    indices = [0, 1, 2, 3, 4, 5, 6]
    if duplicate_coordinate:
        indices[4] = 0
    if invalid_coordinate:
        indices[4] = 99
    fine = tuple(
        _FineEdge(f"fine-{index}", (patch_index,), dest)
        for index, patch_index in enumerate(indices)
    )
    groups = (
        PromotedFineEdgeGroup(
            promoted_edge=("a", "b"),
            fine_edge_names=tuple(edge.name for edge in fine[:3]),
            fine_edges=fine[:3],
            collision_count=3,
        ),
        PromotedFineEdgeGroup(
            promoted_edge=("b", "c"),
            fine_edge_names=(fine[3].name,),
            fine_edges=(fine[3],),
            collision_count=1,
        ),
        PromotedFineEdgeGroup(
            promoted_edge=("a", "c"),
            fine_edge_names=tuple(edge.name for edge in fine[4:]),
            fine_edges=fine[4:],
            collision_count=3,
        ),
    )
    return PromotedFineEdgeGrouping(
        case_id=58,
        edge_order=tuple(group.promoted_edge for group in groups),
        groups=groups,
        fine_edge_count=len(fine),
        internal_fine_edge_count=len(fine),
        excluded_boundary_fine_edge_count=0,
    )


class _ToyPatchable(torch.nn.Module):
    def __init__(self, grouping: PromotedFineEdgeGrouping) -> None:
        super().__init__()
        self.boundary = torch.nn.Identity()
        self.out_slice = (slice(None), slice(1, None))
        self.is_factorized = True
        self.separate_qkv = True
        self.seq_len = None
        self._fine_edges = tuple(
            edge for group in grouping.groups for edge in group.fine_edges
        )
        self._edge_index = {
            id(edge): index for index, edge in enumerate(self._fine_edges)
        }
        self._masks = {"dest": torch.zeros(len(self._fine_edges))}
        self._active_edges: tuple[_FineEdge, ...] = ()
        self._donor: torch.Tensor | None = None
        self.forward_calls = 0
        self.raise_on_patched_forward = False

    def current_patch_masks_as_prune_scores(self) -> dict[str, torch.Tensor]:
        return self._masks

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        observed = self.boundary(inputs.float())
        direction = torch.linspace(-1.0, 1.0, 5)
        logits = observed.unsqueeze(-1) * direction
        if self._active_edges:
            if self.raise_on_patched_forward:
                raise RuntimeError("injected patched-forward failure")
            assert self._donor is not None
            donor_delta = (self._donor[0] - observed).unsqueeze(-1)
            scale = sum(self._edge_index[id(edge)] + 1 for edge in self._active_edges)
            logits = logits + donor_delta * direction * scale
        return logits


def _toy_runtime() -> EdgeSetRuntime:
    def src_ablations(
        model: _ToyPatchable,
        inputs: torch.Tensor,
        ablation: object,
    ) -> torch.Tensor:
        model(inputs)
        values = inputs.float()
        if ablation == "batch_mean":
            values = values.mean(dim=0, keepdim=True).expand_as(values)
        return values.unsqueeze(0).detach().clone()

    @contextmanager
    def patch_mode(
        model: _ToyPatchable,
        donor: torch.Tensor,
        *,
        edges: tuple[_FineEdge, ...],
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

    return EdgeSetRuntime(
        src_ablations=src_ablations,
        patch_mode=patch_mode,
        set_all_masks=set_all_masks,
        resample_ablation="resample",
        batch_tokenwise_mean_ablation="batch_mean",
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    clean = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]])
    corrupt = torch.tensor([[0, 3, 3, 3], [0, 5, 5, 5]])
    return clean, corrupt


def test_whole_support_expands_qkv_and_partitions_full_fine_universe() -> None:
    grouping = _toy_grouping()
    universe = build_induced_edge_universe(grouping, selected_nodes=("a", "b", "c"))
    support = FrozenEdgeSupport(
        support_id="whole:first",
        resolution="whole",
        method="toy",
        support_size=1,
        selected_indices=(0,),
        kind="deterministic",
    )

    plan = build_edge_condition_plan(universe, (support,))
    reference = plan.support_references[0]
    selected = plan.mask_for_condition(reference.selected_condition_index)
    complement = plan.mask_for_condition(reference.complement_condition_index)

    assert selected.tolist() == [True, True, True, False, False, False, False]
    assert torch.equal(selected | complement, torch.ones(7, dtype=torch.bool))
    assert not bool((selected & complement).any())


def test_fine_support_can_span_qkv_and_mlp_groups_canonically() -> None:
    universe = build_induced_edge_universe(
        _toy_grouping(), selected_nodes=("a", "b", "c")
    )
    support = FrozenEdgeSupport(
        support_id="fine:cross-group",
        resolution="fine",
        method="toy",
        support_size=3,
        selected_indices=(0, 3, 5),
        kind="deterministic",
    )

    plan = build_edge_condition_plan(universe, (support,))
    selected = plan.mask_for_condition(
        plan.support_references[0].selected_condition_index
    )

    assert selected.tolist() == [True, False, False, True, False, True, False]
    with pytest.raises(ValueError, match="strictly increasing"):
        FrozenEdgeSupport(
            support_id="bad",
            resolution="fine",
            method="toy",
            support_size=2,
            selected_indices=(3, 3),
            kind="deterministic",
        )
    with pytest.raises(ValueError, match="outside the induced"):
        build_edge_condition_plan(
            build_induced_edge_universe(_toy_grouping(), selected_nodes=("a", "b")),
            (
                FrozenEdgeSupport(
                    support_id="outside",
                    resolution="fine",
                    method="toy",
                    support_size=1,
                    selected_indices=(4,),
                    kind="deterministic",
                ),
            ),
        )


def test_bridge_masks_and_global_dedup_are_response_blind() -> None:
    universe = build_induced_edge_universe(_toy_grouping(), selected_nodes=("a", "b"))
    supports = (
        FrozenEdgeSupport(
            support_id="same-as-induced",
            resolution="whole",
            method="toy",
            support_size=1,
            selected_indices=(0,),
            kind="deterministic",
        ),
        FrozenEdgeSupport(
            support_id="empty",
            resolution="fine",
            method="toy",
            support_size=0,
            selected_indices=(),
            kind="deterministic",
        ),
    )

    plan = build_edge_condition_plan(universe, supports)
    all_mask = plan.mask_for_condition(plan.all_internal_condition_index)
    induced = plan.mask_for_condition(plan.induced_condition_index)
    outside = plan.mask_for_condition(plan.outside_induced_condition_index)
    first, empty = plan.support_references

    assert torch.equal(induced | outside, all_mask)
    assert not bool((induced & outside).any())
    assert first.selected_condition_index == plan.induced_condition_index
    assert first.complement_condition_index == plan.outside_induced_condition_index
    assert empty.selected_condition_index is None
    assert empty.complement_condition_index == plan.all_internal_condition_index
    assert plan.condition_count == 3


def test_resample_primary_adds_exact_within_induced_complement() -> None:
    universe = build_induced_edge_universe(_toy_grouping(), selected_nodes=("a", "b"))
    primary = FrozenEdgeSupport(
        support_id="primary",
        resolution="fine",
        method="toy",
        support_size=1,
        selected_indices=(0,),
        kind="deterministic",
    )

    plan = build_edge_condition_plan(
        universe,
        (primary,),
        primary_support_id="primary",
    )
    selected = plan.mask_for_condition(
        plan.support_references[0].selected_condition_index
    )
    induced = plan.mask_for_condition(plan.induced_condition_index)
    within_complement = plan.mask_for_condition(
        plan.primary_within_induced_complement_condition_index
    )

    assert plan.primary_support_id == "primary"
    assert torch.equal(selected | within_complement, induced)
    assert not bool((selected & within_complement).any())
    assert within_complement.tolist() == [False, True, True, False, False, False, False]
    assert plan.condition_count == 6


def test_set_acquisition_direction_empty_reuse_and_exact_accounting() -> None:
    grouping = _toy_grouping()
    universe = build_induced_edge_universe(grouping, selected_nodes=("a", "b"))
    plan = build_edge_condition_plan(
        universe,
        (
            FrozenEdgeSupport(
                support_id="one-fine",
                resolution="fine",
                method="toy",
                support_size=1,
                selected_indices=(0,),
                kind="deterministic",
            ),
            FrozenEdgeSupport(
                support_id="empty",
                resolution="fine",
                method="toy",
                support_size=0,
                selected_indices=(),
                kind="deterministic",
            ),
        ),
        primary_support_id="one-fine",
    )
    model = _ToyPatchable(grouping)
    clean, corrupt = _inputs()

    acquired = _acquire_edge_sets_from_patchable(
        case_id=58,
        auto_model=model,
        forward_boundary=model.boundary,
        grouping=grouping,
        universe=universe,
        plan=plan,
        clean_inputs=clean,
        corrupt_inputs=corrupt,
        expected_sequence_length=4,
        expected_classes=5,
        runtime=_toy_runtime(),
        ablation=EdgeAblation.RESAMPLE,
    )

    assert acquired.clean_baseline_output.shape == (2, 3, 5)
    assert not torch.equal(
        acquired.clean_baseline_output,
        acquired.corrupt_baseline_output,
    )
    assert torch.equal(
        acquired.support_selected_outputs[1], acquired.clean_baseline_output
    )
    assert torch.equal(
        acquired.primary_within_induced_complement_output,
        acquired.condition_outputs[
            plan.primary_within_induced_complement_condition_index
        ],
    )
    # Corrupt donor values exceed clean recipient values, so the final centered
    # class direction moves positively; this distinguishes donor orientation.
    assert bool(
        (
            acquired.support_selected_outputs[0][..., -1]
            > acquired.clean_baseline_output[..., -1]
        ).all()
    )
    counts = acquired.query_accounting
    assert counts.condition_count == plan.condition_count
    assert counts.expected_physical_forwards == 3 + plan.condition_count
    assert counts.observed_physical_forwards == 3 + plan.condition_count
    assert counts.observed_model_examples == 2 * (3 + plan.condition_count)
    assert counts.active_mask_coordinate_checks == int(plan.fine_masks.sum())
    assert counts.backward_passes == 0
    assert torch.equal(model._masks["dest"], torch.zeros(7))


def test_batch_tokenwise_mean_corrupt_is_explicit_and_uses_its_own_cache() -> None:
    grouping = _toy_grouping()
    universe = build_induced_edge_universe(grouping, selected_nodes=("a", "b", "c"))
    plan = build_edge_condition_plan(
        universe,
        (
            FrozenEdgeSupport(
                support_id="fine",
                resolution="fine",
                method="toy",
                support_size=1,
                selected_indices=(0,),
                kind="deterministic",
            ),
        ),
    )
    model = _ToyPatchable(grouping)
    clean, corrupt = _inputs()

    acquired = _acquire_edge_sets_from_patchable(
        case_id=58,
        auto_model=model,
        forward_boundary=model.boundary,
        grouping=grouping,
        universe=universe,
        plan=plan,
        clean_inputs=clean,
        corrupt_inputs=corrupt,
        expected_sequence_length=4,
        expected_classes=5,
        runtime=_toy_runtime(),
        ablation=EdgeAblation.BATCH_TOKENWISE_MEAN_CORRUPT,
    )

    # Both rows receive the same corrupt-batch mean donor.
    output = acquired.support_selected_outputs[0]
    assert torch.equal(output[0], output[1])
    assert acquired.ablation == EdgeAblation.BATCH_TOKENWISE_MEAN_CORRUPT
    assert acquired.primary_within_induced_complement_output is None


@pytest.mark.parametrize(
    "grouping",
    (
        _toy_grouping(duplicate_coordinate=True),
        _toy_grouping(invalid_coordinate=True),
    ),
)
def test_bad_live_coordinates_fail_before_any_forward(
    grouping: PromotedFineEdgeGrouping,
) -> None:
    universe = build_induced_edge_universe(grouping, selected_nodes=("a", "b", "c"))
    plan = build_edge_condition_plan(
        universe,
        (
            FrozenEdgeSupport(
                support_id="fine",
                resolution="fine",
                method="toy",
                support_size=1,
                selected_indices=(0,),
                kind="deterministic",
            ),
        ),
    )
    model = _ToyPatchable(grouping)
    clean, corrupt = _inputs()

    with pytest.raises(ValueError, match="coordinate"):
        _acquire_edge_sets_from_patchable(
            case_id=58,
            auto_model=model,
            forward_boundary=model.boundary,
            grouping=grouping,
            universe=universe,
            plan=plan,
            clean_inputs=clean,
            corrupt_inputs=corrupt,
            expected_sequence_length=4,
            expected_classes=5,
            runtime=_toy_runtime(),
            ablation=EdgeAblation.RESAMPLE,
        )
    assert model.forward_calls == 0


def test_masks_are_zeroed_when_a_patched_forward_raises() -> None:
    grouping = _toy_grouping()
    universe = build_induced_edge_universe(grouping, selected_nodes=("a", "b", "c"))
    plan = build_edge_condition_plan(
        universe,
        (
            FrozenEdgeSupport(
                support_id="fine",
                resolution="fine",
                method="toy",
                support_size=1,
                selected_indices=(0,),
                kind="deterministic",
            ),
        ),
    )
    model = _ToyPatchable(grouping)
    model.raise_on_patched_forward = True
    clean, corrupt = _inputs()

    with pytest.raises(RuntimeError, match="injected patched-forward failure"):
        _acquire_edge_sets_from_patchable(
            case_id=58,
            auto_model=model,
            forward_boundary=model.boundary,
            grouping=grouping,
            universe=universe,
            plan=plan,
            clean_inputs=clean,
            corrupt_inputs=corrupt,
            expected_sequence_length=4,
            expected_classes=5,
            runtime=_toy_runtime(),
            ablation=EdgeAblation.RESAMPLE,
        )
    assert torch.equal(model._masks["dest"], torch.zeros(7))


def test_core_singletons_are_bidirectional_and_counted_exactly() -> None:
    grouping = _toy_grouping()
    universe = build_induced_edge_universe(grouping, selected_nodes=("a", "b"))
    model = _ToyPatchable(grouping)
    clean, corrupt = _inputs()

    acquired = _acquire_core_singletons_from_patchable(
        case_id=58,
        auto_model=model,
        forward_boundary=model.boundary,
        grouping=grouping,
        universe=universe,
        clean_inputs=clean,
        corrupt_inputs=corrupt,
        expected_sequence_length=4,
        expected_classes=5,
        runtime=_toy_runtime(),
    )

    assert acquired.fine_edge_names == ("fine-0", "fine-1", "fine-2")
    assert acquired.fine_to_whole_indices == (0, 0, 0)
    assert acquired.fine_to_induced_whole_indices == (0, 0, 0)
    assert acquired.corrupt_into_clean_outputs.shape == (3, 2, 3, 5)
    assert acquired.clean_into_corrupt_outputs.shape == (3, 2, 3, 5)
    assert torch.allclose(
        acquired.corrupt_into_clean_outputs - acquired.clean_baseline_output,
        -(acquired.clean_into_corrupt_outputs - acquired.corrupt_baseline_output),
    )
    counts = acquired.query_accounting
    assert counts.expected_physical_forwards == 4 + 2 * 3
    assert counts.observed_physical_forwards == 4 + 2 * 3
    assert counts.observed_model_examples == 2 * (4 + 2 * 3)
    assert counts.active_mask_coordinate_checks == 2 * 3
    assert torch.equal(model._masks["dest"], torch.zeros(7))


@pytest.mark.parametrize(
    "entrypoint", (run_edge_set_replacement, run_core_singleton_replacement)
)
def test_public_live_entrypoints_reject_wrapped_models_before_runtime_loading(
    entrypoint: object,
) -> None:
    class AlreadyWrapped(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_masks = torch.nn.ParameterDict()

    kwargs = {
        "case_id": 58,
        "n_layers": 3,
        "n_heads": 4,
        "n_ctx": 10,
        "d_vocab_out": 16,
        "model": AlreadyWrapped(),
        "case": object(),
        "eligible_pair": object(),
        "universe": object(),
    }
    if entrypoint is run_edge_set_replacement:
        kwargs.update(plan=object(), ablation=EdgeAblation.RESAMPLE)
    with pytest.raises(ValueError, match="fresh unwrapped model"):
        entrypoint(**kwargs)  # type: ignore[operator]


def _synthetic_case_grouping(case_id: int) -> PromotedFineEdgeGrouping:
    _nodes, whole_edges = case_universe(case_id)
    dest = _Dest("dest")
    groups = []
    fine_index = 0
    for whole_edge in whole_edges:
        collision_count = 3 if ".attn.hook_result[" in whole_edge[1] else 1
        members = tuple(
            _FineEdge(
                f"fine-{fine_index + offset}",
                (fine_index + offset,),
                dest,
            )
            for offset in range(collision_count)
        )
        fine_index += collision_count
        groups.append(
            PromotedFineEdgeGroup(
                promoted_edge=whole_edge,
                fine_edge_names=tuple(member.name for member in members),
                fine_edges=members,
                collision_count=collision_count,
            )
        )
    return PromotedFineEdgeGrouping(
        case_id=case_id,
        edge_order=whole_edges,
        groups=tuple(groups),
        fine_edge_count=fine_index,
        internal_fine_edge_count=fine_index,
        excluded_boundary_fine_edge_count=0,
    )


def test_historical_core_counts_and_authorization_hashes_are_exact() -> None:
    from experiments.e32_interpbench.e32_4_protocol import (
        CASE_GEOMETRY,
        CORE_CASE_ANCESTRY,
        CORE_NODE_SUPPORTS,
    )

    expected_authorizations = {
        58: "ebd0b6e01998951968fd284463dbbbd2d8f5b2ebf3619801feb544166451fd21",
        129: "6f76bbc0f2414290c4406a57e575144a7cd1e14a2732101afa8700cac31a6d25",
        25: "61170db4123ee1f36ee93c02c96698edeeeb6a473c17e241039241039168abe1",
    }
    expected_counts = {58: (25, 45), 129: (22, 48), 25: (9, 17)}

    for case_id, expected in expected_counts.items():
        universe = build_induced_edge_universe(
            _synthetic_case_grouping(case_id),
            selected_nodes=CORE_NODE_SUPPORTS[case_id],
        )
        assert (
            universe.induced_whole_edge_count,
            universe.induced_fine_edge_count,
        ) == expected
        assert CASE_GEOMETRY[case_id]["induced_whole_edges"] == expected[0]
        assert CASE_GEOMETRY[case_id]["induced_fine_edges"] == expected[1]
        assert (
            CORE_CASE_ANCESTRY[case_id]["authorization_sha256"]
            == expected_authorizations[case_id]
        )
