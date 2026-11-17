# ruff: noqa: EXE002
"""Exact directed AutoCircuit edge replacement for E32.2.

The primary acquisition replaces one *fine* residual edge at a time.  Incoming
Q/K/V edges are kept separate during acquisition and may only be promoted by a
declared downstream aggregation.  This mirrors InterpBench's logical-OR
whole-head projection without introducing a three-edge joint intervention.

AutoCircuit imports remain lazy so graph/scoring tests run in the project's
Python-3.12 environment.  Live acquisition is restricted to the pinned
Python-3.11 environment already verified by :mod:`eap_baseline`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from experiments.e32_interpbench.adapter import (
    CASE_MANIFESTS,
    DEFAULT_ENV_DIR,
    PANEL_CASE_IDS,
    BehaviorallyEligiblePair,
    patchable_internal_universe,
    promote_official_node,
)
from experiments.e32_interpbench.eap_baseline import (
    DEFAULT_AUTO_CIRCUIT_DIR,
    OFFICIAL_EAP_SETTINGS,
    FineEdgeScore,
    extract_fine_edge_scores,
    promote_and_aggregate_fine_scores,
    verify_pinned_eap_runtime,
)

Edge = tuple[str, str]


@dataclass(frozen=True)
class PromotedFineEdgeGroup:
    """Live fine edges collapsed to one ordered E32 promoted candidate edge."""

    promoted_edge: Edge
    fine_edge_names: tuple[str, ...]
    fine_edges: tuple[Any, ...]
    collision_count: int


@dataclass(frozen=True)
class PromotedFineEdgeGrouping:
    """Complete, validated fine-to-promoted mapping for one AutoCircuit graph."""

    case_id: int
    edge_order: tuple[Edge, ...]
    groups: tuple[PromotedFineEdgeGroup, ...]
    fine_edge_count: int
    internal_fine_edge_count: int
    excluded_boundary_fine_edge_count: int


@dataclass(frozen=True)
class DirectedReplacementAccounting:
    """Expected and directly observed work for one replacement acquisition."""

    example_count: int
    promoted_edge_count: int
    fine_edge_count: int
    source_cache_forwards: int
    baseline_forwards: int
    fine_replacement_forwards: int
    grouped_replacement_forwards: int
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
class DirectedReplacementRun:
    """Full raw non-BOS outputs and recipient-target loss changes."""

    case_id: int
    settings: Mapping[str, Any]
    grouping: PromotedFineEdgeGrouping
    requested_promoted_edges: tuple[Edge, ...]
    fine_edge_names: tuple[str, ...]
    fine_to_promoted_indices: tuple[int, ...]
    clean_baseline_logits: torch.Tensor
    corrupt_baseline_logits: torch.Tensor
    fine_clean_recipient_corrupt_donor_logits: torch.Tensor
    fine_corrupt_recipient_clean_donor_logits: torch.Tensor
    clean_recipient_target_ce_deltas: torch.Tensor
    corrupt_recipient_target_ce_deltas: torch.Tensor
    grouped_clean_recipient_corrupt_donor_logits: torch.Tensor | None
    grouped_corrupt_recipient_clean_donor_logits: torch.Tensor | None
    grouped_clean_recipient_target_ce_deltas: torch.Tensor | None
    grouped_corrupt_recipient_target_ce_deltas: torch.Tensor | None
    query_accounting: DirectedReplacementAccounting


@dataclass(frozen=True)
class _AutoCircuitRuntime:
    """Small injected surface used by the live bridge and dependency-free tests."""

    src_ablations: Callable[..., Any]
    patch_mode: Callable[..., Any]
    set_all_masks: Callable[..., Any]
    resample_ablation: Any


def _fine_target_slot(target: str) -> tuple[str, int]:
    for slot, letter in enumerate("qkv"):
        if f"hook_{letter}_input" in target:
            return "attention_head", slot
    if "hook_mlp_in" in target:
        return "mlp", 0
    raise ValueError(f"internal fine edge has an unexpected target: {target!r}")


def build_promoted_fine_edge_grouping(
    case_id: int,
    fine_scores: Sequence[FineEdgeScore],
    live_edges_by_name: Mapping[str, Any],
) -> PromotedFineEdgeGrouping:
    """Validate and join complete official fine scores to their live edge objects."""
    if case_id not in PANEL_CASE_IDS:
        raise ValueError(f"case_id must be one of {PANEL_CASE_IDS!r}")
    frozen_scores = tuple(fine_scores)
    promoted = promote_and_aggregate_fine_scores(case_id, frozen_scores)
    fine_names = tuple(score.fine_edge_name for score in frozen_scores)
    if set(live_edges_by_name) != set(fine_names) or len(live_edges_by_name) != len(
        fine_names
    ):
        raise ValueError("live fine-edge names differ from the complete score graph")
    if len({id(edge) for edge in live_edges_by_name.values()}) != len(
        live_edges_by_name
    ):
        raise ValueError("live fine-edge objects must be unique")

    edge_set = set(promoted.edge_order)
    grouped_scores: dict[Edge, list[FineEdgeScore]] = defaultdict(list)
    for fine in frozen_scores:
        edge = (promote_official_node(fine.source), promote_official_node(fine.target))
        if edge in edge_set:
            grouped_scores[edge].append(fine)

    groups: list[PromotedFineEdgeGroup] = []
    for edge, expected_collisions in zip(
        promoted.edge_order,
        promoted.collision_counts,
        strict=True,
    ):
        members = grouped_scores.get(edge, [])
        if len(members) != expected_collisions:
            raise ValueError("fine-to-promoted collision count changed")
        typed_slots = tuple(_fine_target_slot(member.target) for member in members)
        expected_kind = "attention_head" if ".attn.hook_result[" in edge[1] else "mlp"
        if any(kind != expected_kind for kind, _slot in typed_slots):
            raise ValueError("fine target kind differs from promoted target kind")
        members = sorted(
            members,
            key=lambda member: (
                _fine_target_slot(member.target)[1],
                member.fine_edge_name,
            ),
        )
        ordered_slots = tuple(_fine_target_slot(member.target)[1] for member in members)
        expected_slots = (0, 1, 2) if expected_kind == "attention_head" else (0,)
        if ordered_slots != expected_slots:
            raise ValueError("promoted edge does not contain exact Q/K/V-or-MLP slots")
        if len({member.source for member in members}) != 1:
            raise ValueError("one promoted edge unexpectedly mixes fine sources")
        member_names = tuple(member.fine_edge_name for member in members)
        groups.append(
            PromotedFineEdgeGroup(
                promoted_edge=edge,
                fine_edge_names=member_names,
                fine_edges=tuple(live_edges_by_name[name] for name in member_names),
                collision_count=len(members),
            )
        )

    if sum(group.collision_count for group in groups) != (
        promoted.internal_fine_edge_count
    ):
        raise RuntimeError("grouped internal fine-edge count changed")
    return PromotedFineEdgeGrouping(
        case_id=case_id,
        edge_order=promoted.edge_order,
        groups=tuple(groups),
        fine_edge_count=promoted.fine_edge_count,
        internal_fine_edge_count=promoted.internal_fine_edge_count,
        excluded_boundary_fine_edge_count=(promoted.excluded_boundary_fine_edge_count),
    )


def live_promoted_fine_edge_grouping(
    case_id: int,
    patchable_model: Any,
) -> PromotedFineEdgeGrouping:
    """Extract the exact live graph and prove parity with the E32 adapter."""
    edges = getattr(patchable_model, "edges", None)
    if not isinstance(edges, (set, frozenset, tuple, list)) or not edges:
        raise TypeError("patchable_model.edges must be a nonempty collection")
    live_by_name: dict[str, Any] = {}
    for edge in edges:
        name = getattr(edge, "name", None)
        if not isinstance(name, str) or not name or name in live_by_name:
            raise ValueError("live AutoCircuit edge names must be nonempty and unique")
        if getattr(edge, "seq_idx", None) is not None:
            raise ValueError("E32.2 requires all-position fine edges")
        live_by_name[name] = edge
    prune_scores = patchable_model.new_prune_scores()
    fine_scores = extract_fine_edge_scores(patchable_model, prune_scores)
    return build_promoted_fine_edge_grouping(case_id, fine_scores, live_by_name)


def _canonical_groups(
    grouping: PromotedFineEdgeGrouping,
    requested_promoted_edges: Sequence[Edge] | None,
) -> tuple[PromotedFineEdgeGroup, ...]:
    if not isinstance(grouping, PromotedFineEdgeGrouping):
        raise TypeError("grouping must be PromotedFineEdgeGrouping")
    requested = (
        grouping.edge_order
        if requested_promoted_edges is None
        else tuple(tuple(edge) for edge in requested_promoted_edges)
    )
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("requested promoted edges must be nonempty and unique")
    requested_set = set(requested)
    canonical = tuple(edge for edge in grouping.edge_order if edge in requested_set)
    if requested != canonical:
        raise ValueError("requested promoted edges must be a canonical subsequence")
    by_edge = {group.promoted_edge: group for group in grouping.groups}
    if len(by_edge) != len(grouping.edge_order) or set(by_edge) != set(
        grouping.edge_order
    ):
        raise ValueError("group identities differ from grouping edge_order")
    return tuple(by_edge[edge] for edge in requested)


def qkv_max_aggregate(
    grouping: PromotedFineEdgeGrouping,
    requested_promoted_edges: Sequence[Edge],
    fine_values: Any,
    *,
    absolute: bool = False,
) -> torch.Tensor:
    """Max-aggregate fine values using exact Q/K/V-or-MLP group boundaries."""
    groups = _canonical_groups(grouping, requested_promoted_edges)
    values = torch.as_tensor(fine_values)
    expected_fine_count = sum(group.collision_count for group in groups)
    if values.ndim < 1 or values.shape[0] != expected_fine_count:
        raise ValueError("fine_values first axis differs from requested fine edges")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("fine_values must be finite")
    if values.dtype == torch.bool:
        raise ValueError("fine_values must be numeric")
    values = values.abs() if absolute else values
    outputs = []
    start = 0
    for group in groups:
        stop = start + group.collision_count
        outputs.append(values[start:stop].max(dim=0).values)
        start = stop
    if start != expected_fine_count:
        raise RuntimeError("fine aggregation did not consume every value")
    return torch.stack(outputs)


def recipient_target_cross_entropy(logits: Any, full_targets: Any) -> torch.Tensor:
    """Mean non-BOS cross entropy for each recipient context independently."""
    values = torch.as_tensor(logits)
    targets = torch.as_tensor(full_targets)
    if values.ndim != 3 or targets.ndim != 3:
        raise ValueError("logits and targets must have [batch, position, class] axes")
    if (
        values.shape[0] != targets.shape[0]
        or values.shape[1] + 1 != targets.shape[1]
        or values.shape[2] != targets.shape[2]
    ):
        raise ValueError("raw non-BOS logits do not align with full recipient targets")
    if not bool(torch.isfinite(values).all() and torch.isfinite(targets).all()):
        raise ValueError("logits and targets must be finite")
    target_ids = targets.argmax(dim=-1)[:, 1:].to(values.device)
    losses = (
        -torch.log_softmax(values, dim=-1)
        .gather(
            -1,
            target_ids.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    return losses.mean(dim=1)


def directed_replacement_query_accounting(
    *,
    example_count: int,
    promoted_edge_count: int,
    fine_edge_count: int,
    include_grouped_promoted_sensitivity: bool,
    observed_physical_forwards: int | None = None,
    observed_model_examples: int | None = None,
    observed_no_grad_forwards: int | None = None,
    observed_grad_enabled_forwards: int = 0,
) -> DirectedReplacementAccounting:
    """Return the exact work implied by two cached replacement orientations."""
    for value, label in (
        (example_count, "example_count"),
        (promoted_edge_count, "promoted_edge_count"),
        (fine_edge_count, "fine_edge_count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if not isinstance(include_grouped_promoted_sensitivity, bool):
        raise TypeError("include_grouped_promoted_sensitivity must be boolean")
    source_cache_forwards = 2
    baseline_forwards = 2
    fine_forwards = 2 * fine_edge_count
    grouped_forwards = (
        2 * promoted_edge_count if include_grouped_promoted_sensitivity else 0
    )
    expected_forwards = (
        source_cache_forwards + baseline_forwards + fine_forwards + grouped_forwards
    )
    expected_examples = expected_forwards * example_count
    observed_forwards = (
        expected_forwards
        if observed_physical_forwards is None
        else observed_physical_forwards
    )
    observed_examples = (
        expected_examples
        if observed_model_examples is None
        else observed_model_examples
    )
    observed_no_grad = (
        expected_forwards
        if observed_no_grad_forwards is None
        else observed_no_grad_forwards
    )
    return DirectedReplacementAccounting(
        example_count=example_count,
        promoted_edge_count=promoted_edge_count,
        fine_edge_count=fine_edge_count,
        source_cache_forwards=source_cache_forwards,
        baseline_forwards=baseline_forwards,
        fine_replacement_forwards=fine_forwards,
        grouped_replacement_forwards=grouped_forwards,
        mask_context_count=fine_forwards + grouped_forwards,
        active_mask_coordinate_checks=(
            2 * fine_edge_count
            + (2 * fine_edge_count if include_grouped_promoted_sensitivity else 0)
        ),
        expected_physical_forwards=expected_forwards,
        expected_model_examples=expected_examples,
        observed_physical_forwards=observed_forwards,
        observed_model_examples=observed_examples,
        observed_no_grad_forwards=observed_no_grad,
        observed_grad_enabled_forwards=observed_grad_enabled_forwards,
        backward_passes=0,
    )


class _ForwardObserver:
    """Count actual calls at a true internal model boundary."""

    def __init__(self) -> None:
        self.physical_forwards = 0
        self.model_examples = 0
        self.no_grad_forwards = 0
        self.grad_enabled_forwards = 0

    def __call__(self, _module: Any, positional: tuple[Any, ...]) -> None:
        if not positional or not isinstance(positional[0], torch.Tensor):
            raise RuntimeError("observed model forward lacks tensor inputs")
        inputs = positional[0]
        if inputs.ndim < 1 or inputs.shape[0] < 1:
            raise RuntimeError("observed model forward lacks a nonempty batch")
        self.physical_forwards += 1
        self.model_examples += int(inputs.shape[0])
        if torch.is_grad_enabled():
            self.grad_enabled_forwards += 1
        else:
            self.no_grad_forwards += 1


def _integer_inputs(value: Any, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if (
        tensor.ndim != 2
        or min(tensor.shape) < 1
        or tensor.dtype == torch.bool
        or tensor.is_floating_point()
    ):
        raise ValueError(f"{label} must be a nonempty rank-two integer tensor")
    return tensor.detach().clone()


def _full_targets(value: Any, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != 3 or min(tensor.shape) < 1:
        raise ValueError(f"{label} must be a nonempty rank-three tensor")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must be finite")
    return tensor.detach().clone()


def _assert_exact_masks(auto_model: Any, selected_edges: Sequence[Any]) -> int:
    """Prove that only the declared live fine-edge coordinates are active."""
    current = auto_model.current_patch_masks_as_prune_scores()
    if not isinstance(current, Mapping) or not current:
        raise TypeError("patchable model must expose nonempty current masks")
    expected = {module: torch.zeros_like(mask) for module, mask in current.items()}
    coordinates: set[tuple[str, tuple[int, ...]]] = set()
    for edge in selected_edges:
        try:
            module = edge.dest.module_name
            patch_index = tuple(edge.patch_idx)
            expected[module][patch_index] = 1.0
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise ValueError(
                "selected fine edge has an invalid mask coordinate"
            ) from error
        coordinate = (module, patch_index)
        if coordinate in coordinates:
            raise ValueError("selected fine edges share a mask coordinate")
        coordinates.add(coordinate)
    for module, mask in current.items():
        if not isinstance(mask, torch.Tensor) or not bool(torch.isfinite(mask).all()):
            raise RuntimeError("current AutoCircuit masks must be finite tensors")
        if not torch.equal(mask.detach(), expected[module]):
            raise RuntimeError("AutoCircuit active masks differ from requested edges")
    return len(coordinates)


def _mask_coordinate_count(auto_model: Any) -> int:
    current = auto_model.current_patch_masks_as_prune_scores()
    if not isinstance(current, Mapping) or not current:
        raise TypeError("patchable model must expose nonempty current masks")
    return sum(int(mask.numel()) for mask in current.values())


def _raw_non_bos_logits(
    auto_model: Any,
    inputs: torch.Tensor,
    *,
    expected_classes: int,
) -> torch.Tensor:
    logits = auto_model(inputs)[auto_model.out_slice]
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or logits.shape[0] != inputs.shape[0]
        or logits.shape[1] != inputs.shape[1] - 1
        or logits.shape[2] != expected_classes
        or not bool(torch.isfinite(logits).all())
    ):
        raise RuntimeError("raw non-BOS replacement output shape or values changed")
    if logits.requires_grad:
        raise RuntimeError(
            "directed replacement outputs unexpectedly require gradients"
        )
    return logits.detach().cpu().clone()


def _validate_patchable_settings(auto_model: Any) -> None:
    if (
        getattr(auto_model, "is_factorized", None) is not True
        or getattr(auto_model, "separate_qkv", None) is not True
        or getattr(auto_model, "seq_len", "missing") is not None
    ):
        raise ValueError(
            "AutoCircuit model must be factorized, separate-QKV, all-position"
        )
    out_slice = getattr(auto_model, "out_slice", None)
    expected = (slice(None), slice(1, None))
    if (
        not isinstance(out_slice, tuple)
        or len(out_slice) != 2
        or any(
            observed != wanted
            for observed, wanted in zip(out_slice, expected, strict=True)
        )
    ):
        raise ValueError("AutoCircuit output slice must remove only BOS")


def _acquire_from_patchable(
    *,
    case_id: int,
    auto_model: Any,
    forward_boundary: Any,
    grouping: PromotedFineEdgeGrouping,
    requested_promoted_edges: Sequence[Edge] | None,
    clean_inputs: Any,
    corrupt_inputs: Any,
    clean_targets: Any,
    corrupt_targets: Any,
    runtime: _AutoCircuitRuntime,
    include_grouped_promoted_sensitivity: bool,
) -> DirectedReplacementRun:
    """Acquire replacements from an already wrapped model; used by live and tests."""
    if case_id != grouping.case_id:
        raise ValueError("case ID differs from fine-edge grouping")
    if case_id not in CASE_MANIFESTS:
        raise ValueError("case ID is not pinned")
    if not isinstance(runtime, _AutoCircuitRuntime):
        raise TypeError("runtime must be _AutoCircuitRuntime")
    if not isinstance(include_grouped_promoted_sensitivity, bool):
        raise TypeError("include_grouped_promoted_sensitivity must be boolean")
    _validate_patchable_settings(auto_model)
    groups = _canonical_groups(grouping, requested_promoted_edges)
    requested = tuple(group.promoted_edge for group in groups)
    fine_edges = tuple(edge for group in groups for edge in group.fine_edges)
    fine_names = tuple(name for group in groups for name in group.fine_edge_names)
    fine_to_promoted = tuple(
        promoted_index
        for promoted_index, group in enumerate(groups)
        for _ in range(group.collision_count)
    )
    if len(fine_edges) != len(fine_names) or len(fine_edges) != len(fine_to_promoted):
        raise RuntimeError("requested fine-edge metadata axes disagree")

    clean = _integer_inputs(clean_inputs, label="clean_inputs")
    corrupt = _integer_inputs(corrupt_inputs, label="corrupt_inputs")
    clean_y = _full_targets(clean_targets, label="clean_targets")
    corrupt_y = _full_targets(corrupt_targets, label="corrupt_targets")
    if clean.shape != corrupt.shape:
        raise ValueError("clean and corrupt inputs must align")
    if clean_y.shape != corrupt_y.shape or tuple(clean_y.shape[:2]) != tuple(
        clean.shape
    ):
        raise ValueError("recipient targets must align with input rows and positions")
    manifest = CASE_MANIFESTS[case_id]
    if clean.shape[1] != manifest.n_ctx or clean_y.shape[2] != manifest.d_vocab_out:
        raise ValueError("inputs or targets differ from the pinned case manifest")
    example_count = len(clean)
    if not hasattr(forward_boundary, "register_forward_pre_hook"):
        raise TypeError("forward boundary must support forward pre-hooks")
    if _mask_coordinate_count(auto_model) != grouping.fine_edge_count:
        raise RuntimeError(
            "live mask-coordinate count differs from complete fine graph"
        )
    _assert_exact_masks(auto_model, ())

    observer = _ForwardObserver()
    handle = forward_boundary.register_forward_pre_hook(observer)
    mask_checks = 0
    active_mask_checks = 0
    try:
        with torch.inference_mode():
            corrupt_sources = runtime.src_ablations(
                auto_model,
                corrupt,
                runtime.resample_ablation,
            )
            clean_sources = runtime.src_ablations(
                auto_model,
                clean,
                runtime.resample_ablation,
            )
            for source_cache, label in (
                (clean_sources, "clean source cache"),
                (corrupt_sources, "corrupt source cache"),
            ):
                if (
                    not isinstance(source_cache, torch.Tensor)
                    or source_cache.ndim < 3
                    or source_cache.shape[1] != example_count
                    or source_cache.shape[2] != clean.shape[1]
                    or not bool(torch.isfinite(source_cache).all())
                ):
                    raise RuntimeError(
                        f"{label} must have [source, batch, position, ...] axes"
                    )

            clean_baseline = _raw_non_bos_logits(
                auto_model,
                clean,
                expected_classes=manifest.d_vocab_out,
            )
            corrupt_baseline = _raw_non_bos_logits(
                auto_model,
                corrupt,
                expected_classes=manifest.d_vocab_out,
            )

            fine_clean_outputs = []
            fine_corrupt_outputs = []
            for fine_edge in fine_edges:
                selected = (fine_edge,)
                with runtime.patch_mode(
                    auto_model,
                    corrupt_sources,
                    edges=selected,
                ):
                    active_mask_checks += _assert_exact_masks(auto_model, selected)
                    mask_checks += 1
                    fine_clean_outputs.append(
                        _raw_non_bos_logits(
                            auto_model,
                            clean,
                            expected_classes=manifest.d_vocab_out,
                        )
                    )
                with runtime.patch_mode(
                    auto_model,
                    clean_sources,
                    edges=selected,
                ):
                    active_mask_checks += _assert_exact_masks(auto_model, selected)
                    mask_checks += 1
                    fine_corrupt_outputs.append(
                        _raw_non_bos_logits(
                            auto_model,
                            corrupt,
                            expected_classes=manifest.d_vocab_out,
                        )
                    )

            grouped_clean_outputs: list[torch.Tensor] | None = None
            grouped_corrupt_outputs: list[torch.Tensor] | None = None
            if include_grouped_promoted_sensitivity:
                grouped_clean_outputs = []
                grouped_corrupt_outputs = []
                for group in groups:
                    with runtime.patch_mode(
                        auto_model,
                        corrupt_sources,
                        edges=group.fine_edges,
                    ):
                        active_mask_checks += _assert_exact_masks(
                            auto_model,
                            group.fine_edges,
                        )
                        mask_checks += 1
                        grouped_clean_outputs.append(
                            _raw_non_bos_logits(
                                auto_model,
                                clean,
                                expected_classes=manifest.d_vocab_out,
                            )
                        )
                    with runtime.patch_mode(
                        auto_model,
                        clean_sources,
                        edges=group.fine_edges,
                    ):
                        active_mask_checks += _assert_exact_masks(
                            auto_model,
                            group.fine_edges,
                        )
                        mask_checks += 1
                        grouped_corrupt_outputs.append(
                            _raw_non_bos_logits(
                                auto_model,
                                corrupt,
                                expected_classes=manifest.d_vocab_out,
                            )
                        )
    finally:
        runtime.set_all_masks(auto_model, 0.0)
        handle.remove()

    if _assert_exact_masks(auto_model, ()) != 0:
        raise RuntimeError("mask cleanup left an active coordinate")
    fine_clean = torch.stack(fine_clean_outputs)
    fine_corrupt = torch.stack(fine_corrupt_outputs)
    clean_base_ce = recipient_target_cross_entropy(clean_baseline, clean_y)
    corrupt_base_ce = recipient_target_cross_entropy(corrupt_baseline, corrupt_y)
    clean_ce = torch.stack(
        [recipient_target_cross_entropy(output, clean_y) for output in fine_clean]
    )
    corrupt_ce = torch.stack(
        [recipient_target_cross_entropy(output, corrupt_y) for output in fine_corrupt]
    )

    grouped_clean = (
        None if grouped_clean_outputs is None else torch.stack(grouped_clean_outputs)
    )
    grouped_corrupt = (
        None
        if grouped_corrupt_outputs is None
        else torch.stack(grouped_corrupt_outputs)
    )
    grouped_clean_delta = (
        None
        if grouped_clean is None
        else torch.stack(
            [
                recipient_target_cross_entropy(output, clean_y)
                for output in grouped_clean
            ]
        )
        - clean_base_ce.unsqueeze(0)
    )
    grouped_corrupt_delta = (
        None
        if grouped_corrupt is None
        else torch.stack(
            [
                recipient_target_cross_entropy(output, corrupt_y)
                for output in grouped_corrupt
            ]
        )
        - corrupt_base_ce.unsqueeze(0)
    )

    accounting = directed_replacement_query_accounting(
        example_count=example_count,
        promoted_edge_count=len(groups),
        fine_edge_count=len(fine_edges),
        include_grouped_promoted_sensitivity=include_grouped_promoted_sensitivity,
        observed_physical_forwards=observer.physical_forwards,
        observed_model_examples=observer.model_examples,
        observed_no_grad_forwards=observer.no_grad_forwards,
        observed_grad_enabled_forwards=observer.grad_enabled_forwards,
    )
    exact_observation = (
        accounting.observed_physical_forwards == accounting.expected_physical_forwards
        and accounting.observed_model_examples == accounting.expected_model_examples
        and accounting.observed_no_grad_forwards
        == accounting.expected_physical_forwards
        and accounting.observed_grad_enabled_forwards == 0
        and mask_checks == accounting.mask_context_count
        and active_mask_checks == accounting.active_mask_coordinate_checks
    )
    if not exact_observation:
        raise RuntimeError(
            "observed forward or mask work differs from exact accounting"
        )

    return DirectedReplacementRun(
        case_id=case_id,
        settings={
            "factorized": True,
            "separate_qkv": True,
            "slice_output": "not_first_seq",
            "seq_len": None,
            "ablation_type": "resample",
            "primary": "one_fine_edge_per_replacement_then_qkv_max",
            "grouped_promoted_sensitivity": include_grouped_promoted_sensitivity,
            "clean_orientation": "clean_recipient_corrupt_donor",
            "corrupt_orientation": "corrupt_recipient_clean_donor",
            "ce_delta": "patched_recipient_target_ce_minus_recipient_baseline_ce",
        },
        grouping=grouping,
        requested_promoted_edges=requested,
        fine_edge_names=fine_names,
        fine_to_promoted_indices=fine_to_promoted,
        clean_baseline_logits=clean_baseline,
        corrupt_baseline_logits=corrupt_baseline,
        fine_clean_recipient_corrupt_donor_logits=fine_clean,
        fine_corrupt_recipient_clean_donor_logits=fine_corrupt,
        clean_recipient_target_ce_deltas=clean_ce - clean_base_ce.unsqueeze(0),
        corrupt_recipient_target_ce_deltas=corrupt_ce - corrupt_base_ce.unsqueeze(0),
        grouped_clean_recipient_corrupt_donor_logits=grouped_clean,
        grouped_corrupt_recipient_clean_donor_logits=grouped_corrupt,
        grouped_clean_recipient_target_ce_deltas=grouped_clean_delta,
        grouped_corrupt_recipient_target_ce_deltas=grouped_corrupt_delta,
        query_accounting=accounting,
    )


def _load_auto_circuit_runtime() -> tuple[_AutoCircuitRuntime, Callable[..., Any]]:
    from auto_circuit.types import AblationType
    from auto_circuit.utils.ablation_activations import src_ablations
    from auto_circuit.utils.graph_utils import (
        patch_mode,
        patchable_model,
        set_all_masks,
    )

    return (
        _AutoCircuitRuntime(
            src_ablations=src_ablations,
            patch_mode=patch_mode,
            set_all_masks=set_all_masks,
            resample_ablation=AblationType.RESAMPLE,
        ),
        patchable_model,
    )


def _validated_live_pair(
    *,
    case_id: int,
    model: Any,
    case: Any,
    eligible_pair: BehaviorallyEligiblePair,
) -> tuple[Any, int]:
    if case_id not in PANEL_CASE_IDS:
        raise ValueError(f"case_id must be one of {PANEL_CASE_IDS!r}")
    if not isinstance(eligible_pair, BehaviorallyEligiblePair):
        raise TypeError("eligible_pair must be BehaviorallyEligiblePair")
    is_categorical = getattr(case, "is_categorical", None)
    if not callable(is_categorical) or not bool(is_categorical()):
        raise NotImplementedError("directed replacement requires categorical data")
    manifest = CASE_MANIFESTS[case_id]
    cfg = getattr(model, "cfg", None)
    if (
        int(getattr(cfg, "n_layers", -1)) != manifest.n_layers
        or int(getattr(cfg, "n_heads", -1)) != manifest.n_heads
        or int(getattr(cfg, "d_vocab_out", -1)) != manifest.d_vocab_out
    ):
        raise ValueError("model shape differs from pinned case manifest")
    pair = eligible_pair.pair
    clean = _integer_inputs(pair.clean_inputs, label="clean_inputs")
    corrupt = _integer_inputs(pair.corrupt_inputs, label="corrupt_inputs")
    clean_y = _full_targets(pair.clean_targets, label="clean_targets")
    corrupt_y = _full_targets(pair.corrupt_targets, label="corrupt_targets")
    if (
        clean.shape != corrupt.shape
        or clean_y.shape != corrupt_y.shape
        or tuple(clean_y.shape[:2]) != tuple(clean.shape)
    ):
        raise ValueError("eligible clean/corrupt pair axes differ")
    if len(eligible_pair.selected_candidate_indices) != len(clean):
        raise ValueError("eligibility indices do not align with selected rows")
    return pair, len(clean)


def run_directed_replacement(
    *,
    case_id: int,
    model: Any,
    case: Any,
    eligible_pair: BehaviorallyEligiblePair,
    requested_promoted_edges: Sequence[Edge] | None = None,
    include_grouped_promoted_sensitivity: bool = False,
    device: str = "cpu",
    environment_dir: Path | str = DEFAULT_ENV_DIR,
    auto_circuit_dir: Path | str = DEFAULT_AUTO_CIRCUIT_DIR,
) -> DirectedReplacementRun:
    """Wrap a fresh model and acquire exact directed replacement responses."""
    model_modules = getattr(model, "modules", None)
    if not callable(model_modules):
        raise TypeError("model must expose torch-style modules()")
    already_wrapped = hasattr(model, "patch_masks") or any(
        module.__class__.__name__ == "PatchWrapperImpl"
        and module.__class__.__module__.startswith("auto_circuit.")
        for module in model_modules()
    )
    if already_wrapped:
        raise ValueError("run_directed_replacement requires a fresh unwrapped model")
    verify_pinned_eap_runtime(
        environment_dir=environment_dir,
        auto_circuit_dir=auto_circuit_dir,
    )
    pair, _example_count = _validated_live_pair(
        case_id=case_id,
        model=model,
        case=case,
        eligible_pair=eligible_pair,
    )
    model.to(device)
    model.eval()
    forward_boundary = getattr(model, "embed", None)
    if not hasattr(forward_boundary, "register_forward_pre_hook"):
        raise TypeError("fresh model lacks the required embedding forward boundary")

    runtime, patchable_model = _load_auto_circuit_runtime()
    auto_model = patchable_model(
        model,
        factorized=OFFICIAL_EAP_SETTINGS["factorized"],
        slice_output=OFFICIAL_EAP_SETTINGS["slice_output"],
        separate_qkv=OFFICIAL_EAP_SETTINGS["separate_qkv"],
        device=torch.device(device),
    )
    _validate_patchable_settings(auto_model)
    grouping = live_promoted_fine_edge_grouping(case_id, auto_model)
    manifest = CASE_MANIFESTS[case_id]
    expected_nodes, expected_edges = patchable_internal_universe(
        manifest.n_layers,
        manifest.n_heads,
    )
    del expected_nodes
    if grouping.edge_order != expected_edges:
        raise RuntimeError("live promoted grouping differs from adapter edge order")

    return _acquire_from_patchable(
        case_id=case_id,
        auto_model=auto_model,
        forward_boundary=forward_boundary,
        grouping=grouping,
        requested_promoted_edges=requested_promoted_edges,
        clean_inputs=pair.clean_inputs.to(device),
        corrupt_inputs=pair.corrupt_inputs.to(device),
        clean_targets=pair.clean_targets,
        corrupt_targets=pair.corrupt_targets,
        runtime=runtime,
        include_grouped_promoted_sensitivity=(include_grouped_promoted_sensitivity),
    )


__all__ = [
    "DirectedReplacementAccounting",
    "DirectedReplacementRun",
    "PromotedFineEdgeGroup",
    "PromotedFineEdgeGrouping",
    "build_promoted_fine_edge_grouping",
    "directed_replacement_query_accounting",
    "live_promoted_fine_edge_grouping",
    "qkv_max_aggregate",
    "recipient_target_cross_entropy",
    "run_directed_replacement",
]
