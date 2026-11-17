# ruff: noqa: EXE002
"""Simultaneous directed edge-set interventions for E32.4.

The scientific support is represented only by canonical whole/fine indices.
Live AutoCircuit edge objects are joined to that frozen representation after a
fresh model has been wrapped and its complete graph has been validated.

This module deliberately owns no panel generation, support ranking, custody,
or verdict logic.  It provides dependency-injected acquisition seams for unit
tests and narrow public bridges for fresh live models.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import torch

from experiments.e32_interpbench.adapter import (
    DEFAULT_ENV_DIR,
    BehaviorallyEligiblePair,
)
from experiments.e32_interpbench.component_patching import class_center_logits
from experiments.e32_interpbench.e32_2_directed_replacement import (
    PromotedFineEdgeGrouping,
    _assert_exact_masks,
    _ForwardObserver,
    _integer_inputs,
    _load_auto_circuit_runtime,
    _mask_coordinate_count,
    _raw_non_bos_logits,
    _validate_patchable_settings,
)
from experiments.e32_interpbench.e32_3_baselines import (
    _validate_fresh_model,
    _validated_live_pair,
    _validated_source_cache,
    live_generic_fine_edge_grouping,
)
from experiments.e32_interpbench.eap_baseline import (
    DEFAULT_AUTO_CIRCUIT_DIR,
    OFFICIAL_EAP_SETTINGS,
    verify_pinned_eap_runtime,
)

Edge = tuple[str, str]
Resolution = Literal["fine", "whole"]
SupportKind = Literal["deterministic", "random"]


class EdgeAblation(str, Enum):
    """The two prospectively supported donor-cache constructions."""

    RESAMPLE = "resample"
    BATCH_TOKENWISE_MEAN_CORRUPT = "batch_tokenwise_mean_corrupt"


@dataclass(frozen=True)
class EdgeSetRuntime:
    """Small injected AutoCircuit surface used by live code and toy tests."""

    src_ablations: Callable[..., Any]
    patch_mode: Callable[..., Any]
    set_all_masks: Callable[..., Any]
    resample_ablation: Any
    batch_tokenwise_mean_ablation: Any

    def ablation_value(self, ablation: EdgeAblation) -> Any:
        if ablation is EdgeAblation.RESAMPLE:
            return self.resample_ablation
        if ablation is EdgeAblation.BATCH_TOKENWISE_MEAN_CORRUPT:
            return self.batch_tokenwise_mean_ablation
        raise ValueError(f"unsupported edge ablation: {ablation!r}")


def _strict_indices(values: Sequence[int], *, label: str) -> tuple[int, ...]:
    indices = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in indices):
        raise TypeError(f"{label} must contain integers")
    if any(left >= right for left, right in pairwise(indices)):
        raise ValueError(f"{label} must be strictly increasing and unique")
    return indices


@dataclass(frozen=True)
class InducedEdgeUniverse:
    """Complete internal fine graph plus the node-induced historical core."""

    case_id: int
    selected_nodes: tuple[str, ...]
    whole_edges: tuple[Edge, ...]
    fine_edge_names: tuple[str, ...]
    fine_to_whole_indices: tuple[int, ...]
    induced_whole_indices: tuple[int, ...]
    induced_fine_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.case_id, bool) or not isinstance(self.case_id, int):
            raise TypeError("case_id must be an integer")
        if self.case_id < 1:
            raise ValueError("case_id must be positive")
        nodes = self.selected_nodes
        if (
            not nodes
            or len(nodes) != len(set(nodes))
            or any(not isinstance(node, str) or not node for node in nodes)
        ):
            raise ValueError("selected_nodes must be unique nonempty strings")
        edges = self.whole_edges
        if (
            not edges
            or len(edges) != len(set(edges))
            or any(
                len(edge) != 2
                or not all(isinstance(node, str) and node for node in edge)
                or edge[0] == edge[1]
                for edge in edges
            )
        ):
            raise ValueError("whole_edges must be unique directed string pairs")
        graph_nodes = {node for edge in edges for node in edge}
        if not set(nodes).issubset(graph_nodes):
            raise ValueError("selected_nodes escape the whole-edge graph")
        names = self.fine_edge_names
        mapping = self.fine_to_whole_indices
        if (
            not names
            or len(names) != len(set(names))
            or any(not isinstance(name, str) or not name for name in names)
            or len(mapping) != len(names)
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(edges)
                for index in mapping
            )
            or set(mapping) != set(range(len(edges)))
        ):
            raise ValueError("fine names/mapping must cover the complete whole graph")
        induced_whole = _strict_indices(
            self.induced_whole_indices, label="induced_whole_indices"
        )
        induced_fine = _strict_indices(
            self.induced_fine_indices, label="induced_fine_indices"
        )
        selected = set(nodes)
        expected_whole = tuple(
            index
            for index, (source, target) in enumerate(edges)
            if source in selected and target in selected
        )
        expected_fine = tuple(
            index
            for index, whole_index in enumerate(mapping)
            if whole_index in set(expected_whole)
        )
        if induced_whole != expected_whole or induced_fine != expected_fine:
            raise ValueError("induced indices differ from the selected-node subgraph")
        if not induced_whole or not induced_fine:
            raise ValueError("selected_nodes must induce at least one internal edge")

    @property
    def whole_edge_count(self) -> int:
        return len(self.whole_edges)

    @property
    def fine_edge_count(self) -> int:
        return len(self.fine_edge_names)

    @property
    def induced_whole_edge_count(self) -> int:
        return len(self.induced_whole_indices)

    @property
    def induced_fine_edge_count(self) -> int:
        return len(self.induced_fine_indices)


def _grouping_axes(
    grouping: PromotedFineEdgeGrouping,
) -> tuple[tuple[Edge, ...], tuple[str, ...], tuple[int, ...], tuple[Any, ...]]:
    if not isinstance(grouping, PromotedFineEdgeGrouping):
        raise TypeError("grouping must be PromotedFineEdgeGrouping")
    if tuple(group.promoted_edge for group in grouping.groups) != grouping.edge_order:
        raise ValueError("group order differs from grouping.edge_order")
    names: list[str] = []
    mapping: list[int] = []
    live_edges: list[Any] = []
    for whole_index, group in enumerate(grouping.groups):
        if (
            group.collision_count != len(group.fine_edge_names)
            or len(group.fine_edges) != group.collision_count
        ):
            raise ValueError("fine group names, objects, and collision count differ")
        names.extend(group.fine_edge_names)
        mapping.extend([whole_index] * group.collision_count)
        live_edges.extend(group.fine_edges)
    if len(names) != grouping.internal_fine_edge_count or len(names) != len(set(names)):
        raise ValueError("grouping does not contain the exact internal fine graph")
    return grouping.edge_order, tuple(names), tuple(mapping), tuple(live_edges)


def build_induced_edge_universe(
    grouping: PromotedFineEdgeGrouping,
    *,
    selected_nodes: Sequence[str],
) -> InducedEdgeUniverse:
    """Freeze the canonical full/internal and selected-node-induced axes."""
    whole_edges, fine_names, mapping, _live = _grouping_axes(grouping)
    selected = tuple(selected_nodes)
    selected_set = set(selected)
    induced_whole = tuple(
        index
        for index, (source, target) in enumerate(whole_edges)
        if source in selected_set and target in selected_set
    )
    induced_set = set(induced_whole)
    induced_fine = tuple(
        index for index, whole_index in enumerate(mapping) if whole_index in induced_set
    )
    return InducedEdgeUniverse(
        case_id=grouping.case_id,
        selected_nodes=selected,
        whole_edges=whole_edges,
        fine_edge_names=fine_names,
        fine_to_whole_indices=mapping,
        induced_whole_indices=induced_whole,
        induced_fine_indices=induced_fine,
    )


@dataclass(frozen=True)
class FrozenEdgeSupport:
    """One discovery-frozen edge support in canonical full-graph indices."""

    support_id: str
    resolution: Resolution
    method: str
    support_size: int
    selected_indices: tuple[int, ...]
    kind: SupportKind
    replicate_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.support_id, str) or not self.support_id:
            raise ValueError("support_id must be a nonempty string")
        if self.resolution not in {"fine", "whole"}:
            raise ValueError("resolution must be fine or whole")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be a nonempty string")
        if (
            isinstance(self.support_size, bool)
            or not isinstance(self.support_size, int)
            or self.support_size < 0
        ):
            raise ValueError("support_size must be a nonnegative integer")
        selected = _strict_indices(self.selected_indices, label="selected_indices")
        if len(selected) != self.support_size:
            raise ValueError("support_size differs from selected_indices")
        if self.kind not in {"deterministic", "random"}:
            raise ValueError("kind must be deterministic or random")
        if self.kind == "deterministic" and self.replicate_index is not None:
            raise ValueError("deterministic supports cannot have a replicate index")
        if self.kind == "random" and (
            isinstance(self.replicate_index, bool)
            or not isinstance(self.replicate_index, int)
            or self.replicate_index < 0
        ):
            raise ValueError("random supports require a nonnegative replicate index")


@dataclass(frozen=True)
class SupportConditionReference:
    support_id: str
    selected_condition_index: int | None
    complement_condition_index: int | None


@dataclass(frozen=True)
class EdgeConditionPlan:
    """Globally deduplicated nonempty fine masks and all output references."""

    universe: InducedEdgeUniverse
    supports: tuple[FrozenEdgeSupport, ...]
    condition_ids: tuple[str, ...]
    fine_masks: torch.Tensor
    all_internal_condition_index: int
    induced_condition_index: int
    outside_induced_condition_index: int | None
    primary_support_id: str | None
    primary_within_induced_complement_condition_index: int | None
    support_references: tuple[SupportConditionReference, ...]
    empty_mask_reuses_clean_baseline: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.universe, InducedEdgeUniverse):
            raise TypeError("universe must be InducedEdgeUniverse")
        if not self.supports or any(
            not isinstance(value, FrozenEdgeSupport) for value in self.supports
        ):
            raise ValueError("supports must contain FrozenEdgeSupport records")
        support_ids = tuple(support.support_id for support in self.supports)
        if len(support_ids) != len(set(support_ids)):
            raise ValueError("support IDs must be unique")
        masks = self.fine_masks
        if (
            not isinstance(masks, torch.Tensor)
            or masks.dtype != torch.bool
            or masks.ndim != 2
            or tuple(masks.shape)
            != (len(self.condition_ids), self.universe.fine_edge_count)
            or not len(masks)
        ):
            raise ValueError("fine_masks must be nonempty boolean [condition,fine]")
        if any(not isinstance(value, str) or not value for value in self.condition_ids):
            raise ValueError("condition IDs must be nonempty strings")
        if len(self.condition_ids) != len(set(self.condition_ids)):
            raise ValueError("condition IDs must be unique")
        rows = tuple(tuple(bool(value) for value in row) for row in masks)
        if any(not any(row) for row in rows) or len(rows) != len(set(rows)):
            raise ValueError("condition masks must be nonempty and globally unique")
        for index, label in (
            (self.all_internal_condition_index, "all_internal_condition_index"),
            (self.induced_condition_index, "induced_condition_index"),
        ):
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(masks)
            ):
                raise ValueError(f"{label} is outside condition masks")
        if self.outside_induced_condition_index is not None and (
            isinstance(self.outside_induced_condition_index, bool)
            or not isinstance(self.outside_induced_condition_index, int)
            or not 0 <= self.outside_induced_condition_index < len(masks)
        ):
            raise ValueError("outside_induced_condition_index is invalid")
        if self.primary_support_id is None:
            if self.primary_within_induced_complement_condition_index is not None:
                raise ValueError("primary within-induced reference lacks a support ID")
        else:
            if self.primary_support_id not in support_ids:
                raise ValueError("primary_support_id is absent from supports")
            index = self.primary_within_induced_complement_condition_index
            if index is not None and (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(masks)
            ):
                raise ValueError("primary within-induced reference is invalid")
        if (
            len(self.support_references) != len(self.supports)
            or tuple(reference.support_id for reference in self.support_references)
            != support_ids
        ):
            raise ValueError("support references differ from support order")
        for reference in self.support_references:
            for index in (
                reference.selected_condition_index,
                reference.complement_condition_index,
            ):
                if index is not None and (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < len(masks)
                ):
                    raise ValueError("support condition reference is invalid")
        if self.empty_mask_reuses_clean_baseline is not True:
            raise ValueError("empty masks must reuse the clean baseline")

    @property
    def condition_count(self) -> int:
        return len(self.condition_ids)

    def mask_for_condition(self, index: int | None) -> torch.Tensor:
        """Return a cloned fine mask; ``None`` denotes the empty clean baseline."""
        if index is None:
            return torch.zeros(self.universe.fine_edge_count, dtype=torch.bool)
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(self.fine_masks)
        ):
            raise ValueError("condition index is outside the plan")
        return self.fine_masks[index].detach().cpu().clone()


def _support_fine_mask(
    universe: InducedEdgeUniverse, support: FrozenEdgeSupport
) -> torch.Tensor:
    selected = support.selected_indices
    if support.resolution == "fine":
        allowed = set(universe.induced_fine_indices)
        if any(index not in allowed for index in selected):
            raise ValueError(
                f"support {support.support_id!r} lies outside the induced fine core"
            )
        fine_indices = selected
    else:
        allowed = set(universe.induced_whole_indices)
        if any(index not in allowed for index in selected):
            raise ValueError(
                f"support {support.support_id!r} lies outside the induced whole core"
            )
        selected_whole = set(selected)
        fine_indices = tuple(
            fine_index
            for fine_index, whole_index in enumerate(universe.fine_to_whole_indices)
            if whole_index in selected_whole
        )
    mask = torch.zeros(universe.fine_edge_count, dtype=torch.bool)
    if fine_indices:
        mask[list(fine_indices)] = True
    return mask


def build_edge_condition_plan(
    universe: InducedEdgeUniverse,
    supports: Sequence[FrozenEdgeSupport],
    *,
    primary_support_id: str | None = None,
) -> EdgeConditionPlan:
    """Build U, I, U\\I and exact S/U\\S masks with global deduplication."""
    if not isinstance(universe, InducedEdgeUniverse):
        raise TypeError("universe must be InducedEdgeUniverse")
    frozen = tuple(supports)
    if not frozen or any(not isinstance(value, FrozenEdgeSupport) for value in frozen):
        raise ValueError("supports must contain FrozenEdgeSupport records")
    if len({support.support_id for support in frozen}) != len(frozen):
        raise ValueError("support IDs must be unique")

    all_mask = torch.ones(universe.fine_edge_count, dtype=torch.bool)
    induced_mask = torch.zeros_like(all_mask)
    induced_mask[list(universe.induced_fine_indices)] = True
    outside_mask = all_mask & ~induced_mask

    unique_masks: list[torch.Tensor] = []
    condition_ids: list[str] = []
    index_by_row: dict[tuple[bool, ...], int] = {}

    def register(reference_id: str, mask: torch.Tensor) -> int | None:
        if mask.dtype != torch.bool or mask.shape != all_mask.shape:
            raise ValueError("candidate condition mask differs from full fine universe")
        if not bool(mask.any()):
            return None
        row = tuple(bool(value) for value in mask)
        if row in index_by_row:
            return index_by_row[row]
        index = len(unique_masks)
        index_by_row[row] = index
        unique_masks.append(mask.detach().cpu().clone())
        condition_ids.append(reference_id)
        return index

    all_index = register("all_internal", all_mask)
    induced_index = register("induced", induced_mask)
    outside_index = register("outside_induced", outside_mask)
    assert all_index is not None and induced_index is not None

    if primary_support_id is not None and primary_support_id not in {
        support.support_id for support in frozen
    }:
        raise ValueError("primary_support_id is absent from supports")
    primary_within_index: int | None = None
    if primary_support_id is not None:
        primary_support = next(
            support for support in frozen if support.support_id == primary_support_id
        )
        primary_selected = _support_fine_mask(universe, primary_support)
        primary_within_index = register(
            f"induced_complement:{primary_support_id}",
            induced_mask & ~primary_selected,
        )

    references: list[SupportConditionReference] = []
    for support in frozen:
        selected = _support_fine_mask(universe, support)
        complement = ~selected
        if not torch.equal(selected | complement, all_mask) or bool(
            (selected & complement).any()
        ):
            raise RuntimeError("selected/complement masks do not exactly partition U")
        references.append(
            SupportConditionReference(
                support_id=support.support_id,
                selected_condition_index=register(
                    f"selected:{support.support_id}", selected
                ),
                complement_condition_index=register(
                    f"complement:{support.support_id}", complement
                ),
            )
        )
    return EdgeConditionPlan(
        universe=universe,
        supports=frozen,
        condition_ids=tuple(condition_ids),
        fine_masks=torch.stack(unique_masks),
        all_internal_condition_index=all_index,
        induced_condition_index=induced_index,
        outside_induced_condition_index=outside_index,
        primary_support_id=primary_support_id,
        primary_within_induced_complement_condition_index=primary_within_index,
        support_references=tuple(references),
    )


@dataclass(frozen=True)
class EdgeSetQueryAccounting:
    example_count: int
    condition_count: int
    source_cache_forwards: int
    baseline_forwards: int
    patched_forwards: int
    mask_context_count: int
    active_mask_coordinate_checks: int
    expected_physical_forwards: int
    expected_model_examples: int
    observed_physical_forwards: int
    observed_model_examples: int
    observed_no_grad_forwards: int
    observed_grad_enabled_forwards: int
    backward_passes: int


@dataclass(frozen=True)
class EdgeSetAcquisition:
    case_id: int
    ablation: EdgeAblation
    universe: InducedEdgeUniverse
    plan: EdgeConditionPlan
    clean_baseline_output: torch.Tensor
    corrupt_baseline_output: torch.Tensor
    all_internal_output: torch.Tensor
    induced_output: torch.Tensor
    outside_induced_output: torch.Tensor
    primary_within_induced_complement_output: torch.Tensor | None
    condition_outputs: torch.Tensor
    support_selected_outputs: torch.Tensor
    support_complement_outputs: torch.Tensor
    query_accounting: EdgeSetQueryAccounting


@dataclass(frozen=True)
class CoreSingletonQueryAccounting:
    example_count: int
    fine_edge_count: int
    source_cache_forwards: int
    baseline_forwards: int
    replacement_forwards: int
    mask_context_count: int
    active_mask_coordinate_checks: int
    expected_physical_forwards: int
    expected_model_examples: int
    observed_physical_forwards: int
    observed_model_examples: int
    observed_no_grad_forwards: int
    observed_grad_enabled_forwards: int
    backward_passes: int


@dataclass(frozen=True)
class CoreSingletonAcquisition:
    case_id: int
    universe: InducedEdgeUniverse
    fine_edge_names: tuple[str, ...]
    fine_to_whole_indices: tuple[int, ...]
    fine_to_induced_whole_indices: tuple[int, ...]
    clean_baseline_output: torch.Tensor
    corrupt_baseline_output: torch.Tensor
    corrupt_into_clean_outputs: torch.Tensor
    clean_into_corrupt_outputs: torch.Tensor
    query_accounting: CoreSingletonQueryAccounting


def _coerce_ablation(value: EdgeAblation | str) -> EdgeAblation:
    if isinstance(value, EdgeAblation):
        return value
    if not isinstance(value, str):
        raise TypeError("ablation must be an EdgeAblation or exact string value")
    try:
        return EdgeAblation(value)
    except ValueError as error:
        raise ValueError(f"unsupported edge ablation: {value!r}") from error


def _validate_grouping_universe(
    grouping: PromotedFineEdgeGrouping, universe: InducedEdgeUniverse
) -> tuple[Any, ...]:
    whole_edges, fine_names, mapping, live_edges = _grouping_axes(grouping)
    if (
        grouping.case_id != universe.case_id
        or whole_edges != universe.whole_edges
        or fine_names != universe.fine_edge_names
        or mapping != universe.fine_to_whole_indices
    ):
        raise ValueError("live grouping differs from the frozen edge universe")
    return live_edges


def _validate_live_coordinates(auto_model: Any, live_edges: Sequence[Any]) -> None:
    """Reject unknown or aliased live coordinates before the first forward."""
    current = auto_model.current_patch_masks_as_prune_scores()
    if not isinstance(current, Mapping) or not current:
        raise TypeError("patchable model must expose nonempty current masks")
    coordinates: set[tuple[str, tuple[int, ...]]] = set()
    for edge in live_edges:
        try:
            module = edge.dest.module_name
            patch_index = tuple(edge.patch_idx)
            selected = current[module][patch_index]
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise ValueError("live fine edge has an unknown mask coordinate") from error
        if not isinstance(module, str) or not module or selected.numel() != 1:
            raise ValueError("live fine edge has an invalid mask coordinate")
        coordinate = (module, patch_index)
        if coordinate in coordinates:
            raise ValueError("live fine edges contain a duplicate mask coordinate")
        coordinates.add(coordinate)


def _validated_acquisition_inputs(
    *,
    case_id: int,
    auto_model: Any,
    forward_boundary: Any,
    grouping: PromotedFineEdgeGrouping,
    universe: InducedEdgeUniverse,
    clean_inputs: Any,
    corrupt_inputs: Any,
    expected_sequence_length: int,
    expected_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[Any, ...], int, int]:
    if case_id != universe.case_id:
        raise ValueError("case ID differs from frozen edge universe")
    if (
        isinstance(expected_sequence_length, bool)
        or not isinstance(expected_sequence_length, int)
        or expected_sequence_length < 1
        or isinstance(expected_classes, bool)
        or not isinstance(expected_classes, int)
        or expected_classes < 2
    ):
        raise ValueError("expected sequence/classes must be positive categorical sizes")
    if not hasattr(forward_boundary, "register_forward_pre_hook"):
        raise TypeError("forward boundary must support pre-hooks")
    _validate_patchable_settings(auto_model)
    live_edges = _validate_grouping_universe(grouping, universe)
    clean = _integer_inputs(clean_inputs, label="clean_inputs")
    corrupt = _integer_inputs(corrupt_inputs, label="corrupt_inputs")
    if clean.shape != corrupt.shape or clean.shape[1] != expected_sequence_length:
        raise ValueError("clean/corrupt inputs differ from frozen geometry")
    if _mask_coordinate_count(auto_model) != grouping.fine_edge_count:
        raise RuntimeError("mask-coordinate count differs from the complete fine graph")
    _assert_exact_masks(auto_model, ())
    _validate_live_coordinates(auto_model, live_edges)
    return clean, corrupt, live_edges, len(clean), expected_classes


def _centered_output(
    auto_model: Any,
    inputs: torch.Tensor,
    *,
    expected_classes: int,
) -> torch.Tensor:
    return (
        class_center_logits(
            _raw_non_bos_logits(
                auto_model,
                inputs,
                expected_classes=expected_classes,
            )
        )
        .detach()
        .cpu()
        .clone()
    )


def _cleanup_masks(*, runtime: EdgeSetRuntime, auto_model: Any, handle: Any) -> None:
    try:
        runtime.set_all_masks(auto_model, 0.0)
    finally:
        handle.remove()
    if _assert_exact_masks(auto_model, ()) != 0:
        raise RuntimeError("mask cleanup left an active coordinate")


def _acquire_edge_sets_from_patchable(
    *,
    case_id: int,
    auto_model: Any,
    forward_boundary: Any,
    grouping: PromotedFineEdgeGrouping,
    universe: InducedEdgeUniverse,
    plan: EdgeConditionPlan,
    clean_inputs: Any,
    corrupt_inputs: Any,
    expected_sequence_length: int,
    expected_classes: int,
    runtime: EdgeSetRuntime,
    ablation: EdgeAblation | str,
) -> EdgeSetAcquisition:
    """Acquire U/I/U\\I and every deduplicated S/U\\S mask in one direction."""
    if not isinstance(runtime, EdgeSetRuntime):
        raise TypeError("runtime must be EdgeSetRuntime")
    if not isinstance(plan, EdgeConditionPlan) or plan.universe != universe:
        raise ValueError("condition plan differs from the frozen universe")
    ablation_kind = _coerce_ablation(ablation)
    clean, corrupt, live_edges, batch, class_count = _validated_acquisition_inputs(
        case_id=case_id,
        auto_model=auto_model,
        forward_boundary=forward_boundary,
        grouping=grouping,
        universe=universe,
        clean_inputs=clean_inputs,
        corrupt_inputs=corrupt_inputs,
        expected_sequence_length=expected_sequence_length,
        expected_classes=expected_classes,
    )
    if tuple(plan.fine_masks.shape) != (plan.condition_count, len(live_edges)):
        raise ValueError("condition masks differ from the live internal fine graph")

    observer = _ForwardObserver()
    handle = forward_boundary.register_forward_pre_hook(observer)
    mask_contexts = 0
    active_checks = 0
    try:
        with torch.inference_mode():
            donor_sources = runtime.src_ablations(
                auto_model,
                corrupt,
                runtime.ablation_value(ablation_kind),
            )
            _validated_source_cache(
                donor_sources,
                label=f"{ablation_kind.value} source cache",
                batch=batch,
                sequence=expected_sequence_length,
            )
            clean_baseline = _centered_output(
                auto_model, clean, expected_classes=class_count
            )
            corrupt_baseline = _centered_output(
                auto_model, corrupt, expected_classes=class_count
            )
            outputs: list[torch.Tensor] = []
            for mask in plan.fine_masks:
                selected = tuple(
                    live_edges[index]
                    for index, enabled in enumerate(mask.tolist())
                    if enabled
                )
                if not selected:
                    raise RuntimeError(
                        "empty mask escaped response-blind deduplication"
                    )
                with runtime.patch_mode(
                    auto_model,
                    donor_sources,
                    edges=selected,
                ):
                    active_checks += _assert_exact_masks(auto_model, selected)
                    mask_contexts += 1
                    outputs.append(
                        _centered_output(
                            auto_model,
                            clean,
                            expected_classes=class_count,
                        )
                    )
    finally:
        _cleanup_masks(runtime=runtime, auto_model=auto_model, handle=handle)

    condition_outputs = torch.stack(outputs)

    def referenced(index: int | None) -> torch.Tensor:
        return clean_baseline if index is None else condition_outputs[index]

    support_selected = torch.stack(
        [
            referenced(reference.selected_condition_index)
            for reference in plan.support_references
        ]
    )
    support_complement = torch.stack(
        [
            referenced(reference.complement_condition_index)
            for reference in plan.support_references
        ]
    )
    expected_forwards = 3 + plan.condition_count
    expected_examples = batch * expected_forwards
    accounting = EdgeSetQueryAccounting(
        example_count=batch,
        condition_count=plan.condition_count,
        source_cache_forwards=1,
        baseline_forwards=2,
        patched_forwards=plan.condition_count,
        mask_context_count=mask_contexts,
        active_mask_coordinate_checks=active_checks,
        expected_physical_forwards=expected_forwards,
        expected_model_examples=expected_examples,
        observed_physical_forwards=observer.physical_forwards,
        observed_model_examples=observer.model_examples,
        observed_no_grad_forwards=observer.no_grad_forwards,
        observed_grad_enabled_forwards=observer.grad_enabled_forwards,
        backward_passes=0,
    )
    if not (
        observer.physical_forwards == expected_forwards
        and observer.model_examples == expected_examples
        and observer.no_grad_forwards == expected_forwards
        and observer.grad_enabled_forwards == 0
        and mask_contexts == plan.condition_count
        and active_checks == int(plan.fine_masks.sum())
    ):
        raise RuntimeError("observed edge-set work differs from exact accounting")
    return EdgeSetAcquisition(
        case_id=case_id,
        ablation=ablation_kind,
        universe=universe,
        plan=plan,
        clean_baseline_output=clean_baseline,
        corrupt_baseline_output=corrupt_baseline,
        all_internal_output=referenced(plan.all_internal_condition_index),
        induced_output=referenced(plan.induced_condition_index),
        outside_induced_output=referenced(plan.outside_induced_condition_index),
        primary_within_induced_complement_output=(
            None
            if plan.primary_support_id is None
            else referenced(plan.primary_within_induced_complement_condition_index)
        ),
        condition_outputs=condition_outputs,
        support_selected_outputs=support_selected,
        support_complement_outputs=support_complement,
        query_accounting=accounting,
    )


def _acquire_core_singletons_from_patchable(
    *,
    case_id: int,
    auto_model: Any,
    forward_boundary: Any,
    grouping: PromotedFineEdgeGrouping,
    universe: InducedEdgeUniverse,
    clean_inputs: Any,
    corrupt_inputs: Any,
    expected_sequence_length: int,
    expected_classes: int,
    runtime: EdgeSetRuntime,
) -> CoreSingletonAcquisition:
    """Acquire each fine edge in I separately in both resample orientations."""
    if not isinstance(runtime, EdgeSetRuntime):
        raise TypeError("runtime must be EdgeSetRuntime")
    clean, corrupt, live_edges, batch, class_count = _validated_acquisition_inputs(
        case_id=case_id,
        auto_model=auto_model,
        forward_boundary=forward_boundary,
        grouping=grouping,
        universe=universe,
        clean_inputs=clean_inputs,
        corrupt_inputs=corrupt_inputs,
        expected_sequence_length=expected_sequence_length,
        expected_classes=expected_classes,
    )
    induced_indices = universe.induced_fine_indices
    induced_live = tuple(live_edges[index] for index in induced_indices)
    observer = _ForwardObserver()
    handle = forward_boundary.register_forward_pre_hook(observer)
    mask_contexts = 0
    active_checks = 0
    try:
        with torch.inference_mode():
            corrupt_sources = runtime.src_ablations(
                auto_model, corrupt, runtime.resample_ablation
            )
            clean_sources = runtime.src_ablations(
                auto_model, clean, runtime.resample_ablation
            )
            for sources, label in (
                (corrupt_sources, "corrupt source cache"),
                (clean_sources, "clean source cache"),
            ):
                _validated_source_cache(
                    sources,
                    label=label,
                    batch=batch,
                    sequence=expected_sequence_length,
                )
            clean_baseline = _centered_output(
                auto_model, clean, expected_classes=class_count
            )
            corrupt_baseline = _centered_output(
                auto_model, corrupt, expected_classes=class_count
            )
            corrupt_into_clean: list[torch.Tensor] = []
            clean_into_corrupt: list[torch.Tensor] = []
            for edge in induced_live:
                selected = (edge,)
                with runtime.patch_mode(auto_model, corrupt_sources, edges=selected):
                    active_checks += _assert_exact_masks(auto_model, selected)
                    mask_contexts += 1
                    corrupt_into_clean.append(
                        _centered_output(
                            auto_model, clean, expected_classes=class_count
                        )
                    )
                with runtime.patch_mode(auto_model, clean_sources, edges=selected):
                    active_checks += _assert_exact_masks(auto_model, selected)
                    mask_contexts += 1
                    clean_into_corrupt.append(
                        _centered_output(
                            auto_model, corrupt, expected_classes=class_count
                        )
                    )
    finally:
        _cleanup_masks(runtime=runtime, auto_model=auto_model, handle=handle)

    fine_count = len(induced_indices)
    expected_forwards = 4 + 2 * fine_count
    expected_examples = batch * expected_forwards
    accounting = CoreSingletonQueryAccounting(
        example_count=batch,
        fine_edge_count=fine_count,
        source_cache_forwards=2,
        baseline_forwards=2,
        replacement_forwards=2 * fine_count,
        mask_context_count=mask_contexts,
        active_mask_coordinate_checks=active_checks,
        expected_physical_forwards=expected_forwards,
        expected_model_examples=expected_examples,
        observed_physical_forwards=observer.physical_forwards,
        observed_model_examples=observer.model_examples,
        observed_no_grad_forwards=observer.no_grad_forwards,
        observed_grad_enabled_forwards=observer.grad_enabled_forwards,
        backward_passes=0,
    )
    if not (
        observer.physical_forwards == expected_forwards
        and observer.model_examples == expected_examples
        and observer.no_grad_forwards == expected_forwards
        and observer.grad_enabled_forwards == 0
        and mask_contexts == 2 * fine_count
        and active_checks == 2 * fine_count
    ):
        raise RuntimeError("observed core-singleton work differs from exact accounting")
    induced_whole_position = {
        whole_index: position
        for position, whole_index in enumerate(universe.induced_whole_indices)
    }
    full_whole_mapping = tuple(
        universe.fine_to_whole_indices[index] for index in induced_indices
    )
    return CoreSingletonAcquisition(
        case_id=case_id,
        universe=universe,
        fine_edge_names=tuple(
            universe.fine_edge_names[index] for index in induced_indices
        ),
        fine_to_whole_indices=full_whole_mapping,
        fine_to_induced_whole_indices=tuple(
            induced_whole_position[index] for index in full_whole_mapping
        ),
        clean_baseline_output=clean_baseline,
        corrupt_baseline_output=corrupt_baseline,
        corrupt_into_clean_outputs=torch.stack(corrupt_into_clean),
        clean_into_corrupt_outputs=torch.stack(clean_into_corrupt),
        query_accounting=accounting,
    )


def _load_edge_runtime() -> tuple[EdgeSetRuntime, Callable[..., Any]]:
    old_runtime, patchable_model = _load_auto_circuit_runtime()
    from auto_circuit.types import AblationType

    return (
        EdgeSetRuntime(
            src_ablations=old_runtime.src_ablations,
            patch_mode=old_runtime.patch_mode,
            set_all_masks=old_runtime.set_all_masks,
            resample_ablation=AblationType.RESAMPLE,
            batch_tokenwise_mean_ablation=AblationType.BATCH_TOKENWISE_MEAN,
        ),
        patchable_model,
    )


def _prepare_live_model(
    *,
    case_id: int,
    n_layers: int,
    n_heads: int,
    n_ctx: int,
    d_vocab_out: int,
    model: Any,
    case: Any,
    eligible_pair: BehaviorallyEligiblePair,
    device: str,
    environment_dir: Path | str,
    auto_circuit_dir: Path | str,
) -> tuple[Any, Any, Any, PromotedFineEdgeGrouping, EdgeSetRuntime]:
    # This check intentionally precedes all runtime imports and pair validation.
    _validate_fresh_model(model)
    verify_pinned_eap_runtime(
        environment_dir=environment_dir, auto_circuit_dir=auto_circuit_dir
    )
    pair = _validated_live_pair(
        model=model,
        case=case,
        eligible_pair=eligible_pair,
        n_layers=n_layers,
        n_heads=n_heads,
        n_ctx=n_ctx,
        d_vocab_out=d_vocab_out,
    )
    model.to(device)
    model.eval()
    boundary = getattr(model, "embed", None)
    if not hasattr(boundary, "register_forward_pre_hook"):
        raise TypeError("model lacks observable embedding boundary")
    runtime, patchable_model = _load_edge_runtime()
    auto_model = patchable_model(
        model,
        factorized=OFFICIAL_EAP_SETTINGS["factorized"],
        slice_output=OFFICIAL_EAP_SETTINGS["slice_output"],
        separate_qkv=OFFICIAL_EAP_SETTINGS["separate_qkv"],
        device=torch.device(device),
    )
    grouping = live_generic_fine_edge_grouping(
        case_id=case_id,
        n_layers=n_layers,
        n_heads=n_heads,
        patchable_model=auto_model,
    )
    return pair, auto_model, boundary, grouping, runtime


def run_edge_set_replacement(
    *,
    case_id: int,
    n_layers: int,
    n_heads: int,
    n_ctx: int,
    d_vocab_out: int,
    model: Any,
    case: Any,
    eligible_pair: BehaviorallyEligiblePair,
    universe: InducedEdgeUniverse,
    plan: EdgeConditionPlan,
    ablation: EdgeAblation | str,
    device: str = "cpu",
    environment_dir: Path | str = DEFAULT_ENV_DIR,
    auto_circuit_dir: Path | str = DEFAULT_AUTO_CIRCUIT_DIR,
) -> EdgeSetAcquisition:
    """Run one ablation on one fresh unwrapped model.

    Resample and mean-corrupt sensitivity calls must receive separate model
    objects.  Reusing the first model fails the pre-wrap freshness check.
    """
    pair, auto_model, boundary, grouping, runtime = _prepare_live_model(
        case_id=case_id,
        n_layers=n_layers,
        n_heads=n_heads,
        n_ctx=n_ctx,
        d_vocab_out=d_vocab_out,
        model=model,
        case=case,
        eligible_pair=eligible_pair,
        device=device,
        environment_dir=environment_dir,
        auto_circuit_dir=auto_circuit_dir,
    )
    return _acquire_edge_sets_from_patchable(
        case_id=case_id,
        auto_model=auto_model,
        forward_boundary=boundary,
        grouping=grouping,
        universe=universe,
        plan=plan,
        clean_inputs=pair.clean_inputs.to(device),
        corrupt_inputs=pair.corrupt_inputs.to(device),
        expected_sequence_length=n_ctx,
        expected_classes=d_vocab_out,
        runtime=runtime,
        ablation=ablation,
    )


def run_core_singleton_replacement(
    *,
    case_id: int,
    n_layers: int,
    n_heads: int,
    n_ctx: int,
    d_vocab_out: int,
    model: Any,
    case: Any,
    eligible_pair: BehaviorallyEligiblePair,
    universe: InducedEdgeUniverse,
    device: str = "cpu",
    environment_dir: Path | str = DEFAULT_ENV_DIR,
    auto_circuit_dir: Path | str = DEFAULT_AUTO_CIRCUIT_DIR,
) -> CoreSingletonAcquisition:
    """Acquire the discovery core's fine singleton responses on a fresh model."""
    pair, auto_model, boundary, grouping, runtime = _prepare_live_model(
        case_id=case_id,
        n_layers=n_layers,
        n_heads=n_heads,
        n_ctx=n_ctx,
        d_vocab_out=d_vocab_out,
        model=model,
        case=case,
        eligible_pair=eligible_pair,
        device=device,
        environment_dir=environment_dir,
        auto_circuit_dir=auto_circuit_dir,
    )
    return _acquire_core_singletons_from_patchable(
        case_id=case_id,
        auto_model=auto_model,
        forward_boundary=boundary,
        grouping=grouping,
        universe=universe,
        clean_inputs=pair.clean_inputs.to(device),
        corrupt_inputs=pair.corrupt_inputs.to(device),
        expected_sequence_length=n_ctx,
        expected_classes=d_vocab_out,
        runtime=runtime,
    )


def _universe_to_dict(universe: InducedEdgeUniverse) -> dict[str, Any]:
    return {
        "case_id": universe.case_id,
        "selected_nodes": universe.selected_nodes,
        "whole_edges": universe.whole_edges,
        "fine_edge_names": universe.fine_edge_names,
        "fine_to_whole_indices": universe.fine_to_whole_indices,
        "induced_whole_indices": universe.induced_whole_indices,
        "induced_fine_indices": universe.induced_fine_indices,
    }


def _plan_to_dict(plan: EdgeConditionPlan) -> dict[str, Any]:
    return {
        "supports": tuple(asdict(support) for support in plan.supports),
        "condition_ids": plan.condition_ids,
        "fine_masks": plan.fine_masks,
        "all_internal_condition_index": plan.all_internal_condition_index,
        "induced_condition_index": plan.induced_condition_index,
        "outside_induced_condition_index": plan.outside_induced_condition_index,
        "primary_support_id": plan.primary_support_id,
        "primary_within_induced_complement_condition_index": (
            plan.primary_within_induced_complement_condition_index
        ),
        "support_references": tuple(
            asdict(reference) for reference in plan.support_references
        ),
        "empty_mask_reuses_clean_baseline": plan.empty_mask_reuses_clean_baseline,
    }


def edge_set_acquisition_to_dict(value: EdgeSetAcquisition) -> dict[str, Any]:
    """Return a deterministic torch-artifact-ready acquisition payload."""
    if not isinstance(value, EdgeSetAcquisition):
        raise TypeError("value must be EdgeSetAcquisition")
    return {
        "schema_version": "e32_4_edge_set_acquisition_v1",
        "case_id": value.case_id,
        "ablation": value.ablation.value,
        "universe": _universe_to_dict(value.universe),
        "plan": _plan_to_dict(value.plan),
        "clean_baseline_output": value.clean_baseline_output,
        "corrupt_baseline_output": value.corrupt_baseline_output,
        "all_internal_output": value.all_internal_output,
        "induced_output": value.induced_output,
        "outside_induced_output": value.outside_induced_output,
        "primary_within_induced_complement_output": (
            value.primary_within_induced_complement_output
        ),
        "condition_outputs": value.condition_outputs,
        "support_selected_outputs": value.support_selected_outputs,
        "support_complement_outputs": value.support_complement_outputs,
        "query_accounting": asdict(value.query_accounting),
    }


def core_singleton_acquisition_to_dict(
    value: CoreSingletonAcquisition,
) -> dict[str, Any]:
    """Return a deterministic torch-artifact-ready discovery payload."""
    if not isinstance(value, CoreSingletonAcquisition):
        raise TypeError("value must be CoreSingletonAcquisition")
    return {
        "schema_version": "e32_4_core_singleton_acquisition_v1",
        "case_id": value.case_id,
        "universe": _universe_to_dict(value.universe),
        "fine_edge_names": value.fine_edge_names,
        "fine_to_whole_indices": value.fine_to_whole_indices,
        "fine_to_induced_whole_indices": value.fine_to_induced_whole_indices,
        "clean_baseline_output": value.clean_baseline_output,
        "corrupt_baseline_output": value.corrupt_baseline_output,
        "corrupt_into_clean_outputs": value.corrupt_into_clean_outputs,
        "clean_into_corrupt_outputs": value.clean_into_corrupt_outputs,
        "query_accounting": asdict(value.query_accounting),
    }


__all__ = [
    "CoreSingletonAcquisition",
    "CoreSingletonQueryAccounting",
    "EdgeAblation",
    "EdgeConditionPlan",
    "EdgeSetAcquisition",
    "EdgeSetQueryAccounting",
    "EdgeSetRuntime",
    "FrozenEdgeSupport",
    "InducedEdgeUniverse",
    "SupportConditionReference",
    "build_edge_condition_plan",
    "build_induced_edge_universe",
    "core_singleton_acquisition_to_dict",
    "edge_set_acquisition_to_dict",
    "run_core_singleton_replacement",
    "run_edge_set_replacement",
]
