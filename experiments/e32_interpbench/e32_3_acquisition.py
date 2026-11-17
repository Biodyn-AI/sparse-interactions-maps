# ruff: noqa: EXE002
"""Generic whole-component and set-intervention acquisition for E32.3.

The sealed E32 component runner already implements the delicate hook and live
forward-counting logic.  This module composes that public interface: it prepares
the same clean/source panel in both directions on one runner, keeps the raw
activation captures, and executes frozen selected/complement set masks.  It
does not load models, choose supports, or write artifacts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from experiments.e32_interpbench.adapter import OfficialDataPair
from experiments.e32_interpbench.component_patching import (
    ComponentPatchRunner,
    build_component_atlas,
    build_patch_units,
    class_center_logits,
    vector_pair_interactions,
)
from experiments.e32_interpbench.e32_3_cohort import (
    CASE_MANIFESTS,
    case_universe,
)
from experiments.e32_interpbench.e32_3_protocol import (
    CONFIRMATORY_SUPPORT_FRACTION,
    DETERMINISTIC_METHOD_ORDER,
    PRIMARY_METHOD,
    RANDOM_CONTROL_REPLICATES,
    SINGLETON_CURVE_FRACTIONS,
    deterministic_random_support,
)

SupportKind = Literal["deterministic", "random"]


def _finite_tensor(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite")
    return value


def _forward_counts(runner: Any) -> dict[str, int]:
    forward_counts = getattr(runner, "forward_counts", None)
    if not callable(forward_counts):
        raise TypeError("runner must expose forward_counts()")
    counts = forward_counts()
    if not isinstance(counts, Mapping):
        raise TypeError("runner forward counts must be a mapping")
    observed: dict[str, int] = {}
    for key in ("physical_forwards", "model_examples"):
        value = counts.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"runner {key} must be a nonnegative integer")
        observed[key] = value
    return observed


def _count_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    delta = {
        key: int(after[key]) - int(before[key])
        for key in ("physical_forwards", "model_examples")
    }
    if any(value < 0 for value in delta.values()):
        raise RuntimeError("runner forward counters decreased")
    return delta


def _clone_activation_map(value: Any, *, label: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{label} must be a nonempty activation mapping")
    cloned: dict[str, torch.Tensor] = {}
    for hook_name, activation in value.items():
        if not isinstance(hook_name, str) or not hook_name or hook_name in cloned:
            raise ValueError(f"{label} hook names must be unique nonempty strings")
        tensor = _finite_tensor(activation, label=f"{label}[{hook_name!r}]")
        if tensor.ndim < 3 or min(tensor.shape) < 1:
            raise ValueError(f"{label}[{hook_name!r}] has invalid activation axes")
        cloned[hook_name] = tensor.detach().cpu().clone()
    return cloned


@dataclass(frozen=True)
class BidirectionalComponentPair:
    """One panel prepared in both component-patching orientations."""

    normal: Any
    swapped: Any
    clean_output: torch.Tensor
    corrupt_output: torch.Tensor
    clean_activations: Mapping[str, torch.Tensor]
    corrupt_activations: Mapping[str, torch.Tensor]
    query_accounting: Mapping[str, int]


def prepare_bidirectional_component_pair(
    *, runner: Any, pair: OfficialDataPair
) -> BidirectionalComponentPair:
    """Prepare normal and swapped roles on the same observed-counter runner.

    ``normal`` is clean-recipient/corrupt-donor.  ``swapped`` is
    corrupt-recipient/clean-donor.  The sealed runner captures only donor
    activations, so the second preparation is intentionally retained rather
    than reconstructing private runner state.
    """
    prepare_pair = getattr(runner, "prepare_pair", None)
    if not callable(prepare_pair):
        raise TypeError("runner must expose prepare_pair()")
    if not isinstance(pair, OfficialDataPair):
        raise TypeError("pair must be adapter.OfficialDataPair")

    before = _forward_counts(runner)
    normal = prepare_pair(pair)
    after_normal = _forward_counts(runner)
    swapped_pair = OfficialDataPair(
        clean_inputs=pair.corrupt_inputs,
        clean_targets=pair.corrupt_targets,
        corrupt_inputs=pair.clean_inputs,
        corrupt_targets=pair.clean_targets,
    )
    swapped = prepare_pair(swapped_pair)
    after_swapped = _forward_counts(runner)

    clean_output = _finite_tensor(normal.clean_output, label="normal clean output")
    corrupt_output = _finite_tensor(
        normal.corrupt_output, label="normal corrupt output"
    )
    swapped_clean = _finite_tensor(swapped.clean_output, label="swapped clean output")
    swapped_corrupt = _finite_tensor(
        swapped.corrupt_output, label="swapped corrupt output"
    )
    if clean_output.shape != corrupt_output.shape:
        raise RuntimeError("normal clean/corrupt output shapes differ")
    if not torch.equal(swapped_clean, corrupt_output) or not torch.equal(
        swapped_corrupt, clean_output
    ):
        raise RuntimeError("normal and swapped preparations changed branch baselines")

    corrupt_activations = _clone_activation_map(
        normal.donor_activations, label="corrupt activations"
    )
    clean_activations = _clone_activation_map(
        swapped.donor_activations, label="clean activations"
    )
    if tuple(clean_activations) != tuple(corrupt_activations):
        raise RuntimeError("clean/corrupt activation hook identities differ")
    for hook_name in clean_activations:
        if clean_activations[hook_name].shape != corrupt_activations[hook_name].shape:
            raise RuntimeError(f"activation shape differs at {hook_name}")

    normal_delta = _count_delta(before, after_normal)
    swapped_delta = _count_delta(after_normal, after_swapped)
    total_delta = _count_delta(before, after_swapped)
    accounting = {
        **total_delta,
        "normal_prepare_forwards": normal_delta["physical_forwards"],
        "swapped_prepare_forwards": swapped_delta["physical_forwards"],
    }
    return BidirectionalComponentPair(
        normal=normal,
        swapped=swapped,
        clean_output=clean_output.detach().cpu().clone(),
        corrupt_output=corrupt_output.detach().cpu().clone(),
        clean_activations=clean_activations,
        corrupt_activations=corrupt_activations,
        query_accounting=accounting,
    )


def activation_shift_rms(
    *,
    units: Sequence[Any],
    clean_activations: Mapping[str, torch.Tensor],
    corrupt_activations: Mapping[str, torch.Tensor],
) -> tuple[float, ...]:
    """Return donor-minus-clean RMS for every whole head/MLP in unit order."""
    frozen_units = tuple(units)
    if not frozen_units:
        raise ValueError("units must be nonempty")
    clean = _clone_activation_map(clean_activations, label="clean activations")
    corrupt = _clone_activation_map(corrupt_activations, label="corrupt activations")
    if tuple(clean) != tuple(corrupt):
        raise ValueError("clean/corrupt activation hooks differ")

    scores: list[float] = []
    component_ids: set[str] = set()
    for unit in frozen_units:
        try:
            component_id = unit.component_id
            hook_name = unit.hook_name
            kind = unit.kind
            head_index = unit.head_index
        except AttributeError as error:
            raise TypeError(
                "units must expose component/hook/kind/head fields"
            ) from error
        if (
            not isinstance(component_id, str)
            or not component_id
            or component_id in component_ids
        ):
            raise ValueError("unit component IDs must be unique nonempty strings")
        component_ids.add(component_id)
        if hook_name not in clean or hook_name not in corrupt:
            raise ValueError(f"unit hook is absent from captures: {hook_name!r}")
        clean_value = clean[hook_name].to(torch.float64)
        corrupt_value = corrupt[hook_name].to(torch.float64)
        if clean_value.shape != corrupt_value.shape:
            raise ValueError(f"capture shape differs at {hook_name}")
        if kind == "attention_head":
            if (
                clean_value.ndim != 4
                or isinstance(head_index, bool)
                or not isinstance(head_index, int)
                or not 0 <= head_index < clean_value.shape[2]
            ):
                raise ValueError("attention unit has an invalid captured head axis")
            clean_value = clean_value[:, :, head_index, :]
            corrupt_value = corrupt_value[:, :, head_index, :]
        elif kind == "mlp":
            if clean_value.ndim != 3 or head_index is not None:
                raise ValueError("MLP unit has invalid capture axes")
        else:
            raise ValueError("unit kind must be attention_head or mlp")
        scores.append(float((corrupt_value - clean_value).square().mean().sqrt()))
    return tuple(scores)


@dataclass(frozen=True)
class FrozenNodeSupport:
    """One discovery-frozen node set evaluated without confirmation adaptation."""

    support_id: str
    method: str
    quantile: float
    selected_nodes: tuple[str, ...]
    kind: SupportKind
    replicate_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.support_id, str) or not self.support_id:
            raise ValueError("support_id must be a nonempty string")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be a nonempty string")
        if (
            isinstance(self.quantile, bool)
            or not isinstance(self.quantile, (int, float))
            or not math.isfinite(float(self.quantile))
            or not 0.0 < float(self.quantile) < 1.0
        ):
            raise ValueError("support quantile must lie strictly between zero and one")
        if (
            not self.selected_nodes
            or len(self.selected_nodes) != len(set(self.selected_nodes))
            or any(
                not isinstance(node, str) or not node for node in self.selected_nodes
            )
        ):
            raise ValueError("selected_nodes must contain unique nonempty strings")
        if self.kind not in {"deterministic", "random"}:
            raise ValueError("support kind must be deterministic or random")
        if self.kind == "deterministic" and self.replicate_index is not None:
            raise ValueError("deterministic supports cannot have a replicate index")
        if self.kind == "random" and (
            isinstance(self.replicate_index, bool)
            or not isinstance(self.replicate_index, int)
            or self.replicate_index < 0
        ):
            raise ValueError("random supports require a nonnegative replicate index")


@dataclass(frozen=True)
class SetMaskPlan:
    component_ids: tuple[str, ...]
    supports: tuple[FrozenNodeSupport, ...]
    selected_masks: torch.Tensor
    complement_masks: torch.Tensor
    all_mask: torch.Tensor


def build_set_mask_plan(
    *, component_ids: Sequence[str], supports: Sequence[FrozenNodeSupport]
) -> SetMaskPlan:
    """Validate discovery-frozen supports and materialize exact boolean masks."""
    nodes = tuple(component_ids)
    if (
        not nodes
        or len(nodes) != len(set(nodes))
        or any(not isinstance(node, str) or not node for node in nodes)
    ):
        raise ValueError("component_ids must be unique nonempty strings")
    frozen_supports = tuple(supports)
    if not frozen_supports or any(
        not isinstance(support, FrozenNodeSupport) for support in frozen_supports
    ):
        raise ValueError("supports must contain FrozenNodeSupport records")
    support_ids = tuple(support.support_id for support in frozen_supports)
    if len(support_ids) != len(set(support_ids)):
        raise ValueError("support IDs must be unique")

    node_index = {node: index for index, node in enumerate(nodes)}
    rows: list[torch.Tensor] = []
    for support in frozen_supports:
        selected = tuple(support.selected_nodes)
        if not set(selected).issubset(node_index):
            raise ValueError(
                f"support {support.support_id!r} escapes component universe"
            )
        expected_count = math.ceil(float(support.quantile) * len(nodes))
        if len(selected) != expected_count:
            raise ValueError(
                f"support {support.support_id!r} size must equal ceil(q * node_count)"
            )
        row = torch.zeros(len(nodes), dtype=torch.bool)
        row[[node_index[node] for node in selected]] = True
        rows.append(row)
    selected_masks = torch.stack(rows)
    return SetMaskPlan(
        component_ids=nodes,
        supports=frozen_supports,
        selected_masks=selected_masks,
        complement_masks=~selected_masks,
        all_mask=torch.ones(len(nodes), dtype=torch.bool),
    )


@dataclass(frozen=True)
class SetInterventionOutputs:
    plan: SetMaskPlan
    all_nodes_output: torch.Tensor
    selected_outputs: torch.Tensor
    complement_outputs: torch.Tensor
    query_accounting: Mapping[str, int]


def acquire_set_interventions(
    *, runner: Any, prepared: Any, plan: SetMaskPlan
) -> SetInterventionOutputs:
    """Acquire all-U, selected-S, and complement-(U\\S) corrupt-to-clean masks."""
    if not isinstance(plan, SetMaskPlan):
        raise TypeError("plan must be a SetMaskPlan")
    outputs_for_masks = getattr(runner, "outputs_for_masks", None)
    if not callable(outputs_for_masks):
        raise TypeError("runner must expose outputs_for_masks()")
    runner_ids = tuple(
        getattr(unit, "component_id", None) for unit in getattr(runner, "units", ())
    )
    if runner_ids != plan.component_ids:
        raise ValueError("set plan component order differs from runner units")

    masks = torch.cat(
        (
            plan.all_mask.unsqueeze(0),
            plan.selected_masks,
            plan.complement_masks,
        )
    )
    before = _forward_counts(runner)
    outputs = _finite_tensor(
        outputs_for_masks(prepared, masks), label="set intervention outputs"
    )
    after = _forward_counts(runner)
    if outputs.ndim < 2 or outputs.shape[0] != len(masks):
        raise RuntimeError("set outputs lost their exact condition axis")
    support_count = len(plan.supports)
    all_nodes_output = outputs[0].detach().cpu().clone()
    selected_outputs = outputs[1 : 1 + support_count].detach().cpu().clone()
    complement_outputs = outputs[1 + support_count :].detach().cpu().clone()
    if selected_outputs.shape != complement_outputs.shape:
        raise RuntimeError("selected and complement output axes differ")
    observed = _count_delta(before, after)
    accounting = {
        "condition_count": len(masks),
        "selected_condition_count": support_count,
        "complement_condition_count": support_count,
        "all_nodes_condition_count": 1,
        **observed,
    }
    expected_examples = len(masks) * int(outputs.shape[1])
    if accounting["model_examples"] != expected_examples:
        raise RuntimeError("observed set-intervention model examples changed")
    return SetInterventionOutputs(
        plan=plan,
        all_nodes_output=all_nodes_output,
        selected_outputs=selected_outputs,
        complement_outputs=complement_outputs,
        query_accounting=accounting,
    )


def build_cohort_component_runner(
    *, case_id: int, model: Any, condition_batch_size: int = 32
) -> ComponentPatchRunner:
    """Construct the sealed runner against an E32.3 cohort manifest."""
    try:
        manifest = CASE_MANIFESTS[case_id]
    except KeyError as error:
        raise ValueError("case_id is not in the E32.3 cohort") from error
    cfg = getattr(model, "cfg", None)
    observed = tuple(
        int(getattr(cfg, name, -1))
        for name in ("n_layers", "n_heads", "n_ctx", "d_vocab_out")
    )
    expected = (
        manifest.n_layers,
        manifest.n_heads,
        manifest.n_ctx,
        manifest.d_vocab_out,
    )
    if observed != expected:
        raise ValueError("model shape differs from E32.3 cohort manifest")
    component_ids, whole_edges = case_universe(case_id)
    units = build_patch_units(manifest.n_layers, manifest.n_heads)
    if tuple(unit.component_id for unit in units) != component_ids:
        raise RuntimeError("patch units differ from cohort component order")
    return ComponentPatchRunner(
        model,
        units,
        condition_batch_size=condition_batch_size,
        valid_directed_edges=whole_edges,
    )


@dataclass(frozen=True)
class WholeComponentAcquisition:
    """Scientific whole-component atlas plus reusable prepared pair."""

    case_id: int
    split: str
    component_ids: tuple[str, ...]
    whole_edges: tuple[tuple[str, str], ...]
    condition_ids: tuple[str, ...]
    condition_masks: torch.Tensor
    pair_component_indices: tuple[tuple[int, int], ...]
    clean_inputs: torch.Tensor
    corrupt_inputs: torch.Tensor
    clean_targets: torch.Tensor
    corrupt_targets: torch.Tensor
    clean_output: torch.Tensor
    corrupt_output: torch.Tensor
    singleton_outputs: torch.Tensor
    pair_outputs: torch.Tensor
    pair_interactions: torch.Tensor
    clean_activations: Mapping[str, torch.Tensor]
    corrupt_activations: Mapping[str, torch.Tensor]
    activation_shift_rms: tuple[float, ...]
    batch_invariance_smoke: Mapping[str, Any] | None
    query_accounting: Mapping[str, Any]
    _prepared: BidirectionalComponentPair


def _scientific_output(value: Any, *, label: str) -> torch.Tensor:
    tensor = _finite_tensor(value, label=label)
    if tensor.ndim < 3 or tensor.shape[-2] < 2 or tensor.shape[-1] < 2:
        raise ValueError(f"{label} needs position and categorical class axes")
    return class_center_logits(tensor)[..., 1:, :].detach().cpu().clone()


def acquire_whole_component_atlas(
    *,
    case_id: int,
    split: str,
    runner: Any,
    pair: OfficialDataPair,
    discovery_smoke_condition_count: int = 8,
    discovery_smoke_batch_size: int = 8,
) -> WholeComponentAcquisition:
    """Prepare both roles and acquire one exhaustive whole-component atlas."""
    if case_id not in CASE_MANIFESTS:
        raise ValueError("case_id is not in the E32.3 cohort")
    if split not in {"discovery", "confirmation"}:
        raise ValueError("split must be discovery or confirmation")
    component_ids, whole_edges = case_universe(case_id)
    runner_ids = tuple(
        getattr(unit, "component_id", None) for unit in getattr(runner, "units", ())
    )
    if (
        runner_ids != component_ids
        or tuple(getattr(runner, "valid_directed_edges", ())) != whole_edges
    ):
        raise ValueError("runner differs from the exact E32.3 whole-component DAG")
    before = _forward_counts(runner)
    prepared = prepare_bidirectional_component_pair(runner=runner, pair=pair)

    smoke: Mapping[str, Any] | None = None
    if split == "discovery":
        if discovery_smoke_condition_count != 8 or discovery_smoke_batch_size != 8:
            raise ValueError(
                "E32.3 discovery smoke is fixed at eight masks batched by eight"
            )
        atlas_manifest = build_component_atlas(
            runner.units, valid_directed_edges=runner.valid_directed_edges
        )
        smoke = runner.batch_invariance_smoke(
            prepared.normal,
            atlas_manifest.condition_masks[:discovery_smoke_condition_count],
            candidate_condition_batch_size=discovery_smoke_batch_size,
        )
        if not isinstance(smoke, Mapping) or smoke.get("passed") is not True:
            raise RuntimeError("discovery batch-invariance smoke failed")

    atlas = runner.acquire_atlas(prepared.normal)
    if tuple(atlas.atlas.units) != tuple(runner.units):
        raise RuntimeError("atlas unit identities changed")
    if atlas.atlas.directed_edges != whole_edges:
        raise RuntimeError("atlas whole-edge order changed")
    scientific_conditions = _scientific_output(
        atlas.condition_outputs, label="whole-atlas condition outputs"
    )
    singleton_outputs = scientific_conditions[
        list(atlas.atlas.singleton_condition_indices)
    ]
    pair_outputs = scientific_conditions[list(atlas.atlas.pair_condition_indices)]
    pair_interactions = vector_pair_interactions(
        empty_output=scientific_conditions[0],
        singleton_outputs=singleton_outputs,
        pair_outputs=pair_outputs,
        pair_component_indices=atlas.atlas.pair_component_indices,
        center=False,
    )
    if not torch.equal(pair_interactions, atlas.pair_interactions[..., 1:, :]):
        raise RuntimeError("scientific projection changed exact whole-pair J")
    clean_output = _scientific_output(prepared.clean_output, label="clean output")
    corrupt_output = _scientific_output(prepared.corrupt_output, label="corrupt output")
    if not torch.equal(clean_output, scientific_conditions[0]):
        raise RuntimeError("atlas empty condition differs from prepared clean output")

    activation_scores = activation_shift_rms(
        units=runner.units,
        clean_activations=prepared.clean_activations,
        corrupt_activations=prepared.corrupt_activations,
    )
    after = _forward_counts(runner)
    total = _count_delta(before, after)
    smoke_forwards = 0 if smoke is None else int(smoke["physical_forwards"])
    smoke_examples = 0 if smoke is None else int(smoke["model_examples"])
    atlas_accounting = dict(atlas.query_accounting)
    expected_forwards = (
        int(prepared.query_accounting["physical_forwards"])
        + smoke_forwards
        + int(atlas_accounting["patched_physical_forwards"])
    )
    expected_examples = (
        int(prepared.query_accounting["model_examples"])
        + smoke_examples
        + int(atlas_accounting["patched_model_examples"])
    )
    if total != {
        "physical_forwards": expected_forwards,
        "model_examples": expected_examples,
    }:
        raise RuntimeError("whole-atlas family accounting does not sum exactly")

    return WholeComponentAcquisition(
        case_id=case_id,
        split=split,
        component_ids=component_ids,
        whole_edges=whole_edges,
        condition_ids=atlas.atlas.condition_ids,
        condition_masks=atlas.atlas.condition_masks.detach().cpu().clone(),
        pair_component_indices=atlas.atlas.pair_component_indices,
        clean_inputs=pair.clean_inputs.detach().cpu().clone(),
        corrupt_inputs=pair.corrupt_inputs.detach().cpu().clone(),
        clean_targets=pair.clean_targets.detach().cpu().clone(),
        corrupt_targets=pair.corrupt_targets.detach().cpu().clone(),
        clean_output=clean_output,
        corrupt_output=corrupt_output,
        singleton_outputs=singleton_outputs,
        pair_outputs=pair_outputs,
        pair_interactions=pair_interactions,
        clean_activations=prepared.clean_activations,
        corrupt_activations=prepared.corrupt_activations,
        activation_shift_rms=activation_scores,
        batch_invariance_smoke=smoke,
        query_accounting={
            "bidirectional_preparation": dict(prepared.query_accounting),
            "batch_invariance_smoke": None if smoke is None else dict(smoke),
            "whole_atlas": atlas_accounting,
            "total": total,
        },
        _prepared=prepared,
    )


def validate_confirmation_supports(
    *,
    case_id: int,
    component_ids: Sequence[str],
    supports: Sequence[FrozenNodeSupport],
) -> SetMaskPlan:
    """Require the exact seven deterministic and 64 random frozen supports."""
    nodes = tuple(component_ids)
    if case_id not in CASE_MANIFESTS or nodes != case_universe(case_id)[0]:
        raise ValueError("support component universe differs from cohort case")
    frozen = tuple(supports)
    deterministic = tuple(
        support for support in frozen if support.kind == "deterministic"
    )
    random = tuple(support for support in frozen if support.kind == "random")
    expected_deterministic = (
        (PRIMARY_METHOD, SINGLETON_CURVE_FRACTIONS[0]),
        (PRIMARY_METHOD, CONFIRMATORY_SUPPORT_FRACTION),
        (PRIMARY_METHOD, SINGLETON_CURVE_FRACTIONS[-1]),
        *(
            (method, CONFIRMATORY_SUPPORT_FRACTION)
            for method in DETERMINISTIC_METHOD_ORDER
            if method != PRIMARY_METHOD
        ),
    )
    observed_deterministic = tuple(
        (support.method, float(support.quantile)) for support in deterministic
    )
    if observed_deterministic != expected_deterministic:
        raise ValueError(
            "deterministic supports differ from the frozen seven-mask order"
        )
    if len(random) != RANDOM_CONTROL_REPLICATES:
        raise ValueError("random supports differ from the frozen replicate count")
    for replicate, support in enumerate(random):
        expected_nodes = deterministic_random_support(
            case_id=case_id,
            component_ids=nodes,
            support_size=math.ceil(CONFIRMATORY_SUPPORT_FRACTION * len(nodes)),
            replicate=replicate,
        )
        if (
            support.method != "random_control"
            or support.quantile != CONFIRMATORY_SUPPORT_FRACTION
            or support.replicate_index != replicate
            or support.selected_nodes != expected_nodes
        ):
            raise ValueError("random support identity differs from SHA-frozen control")
    return build_set_mask_plan(component_ids=nodes, supports=frozen)


def acquire_confirmation_set_interventions(
    *,
    case_id: int,
    runner: Any,
    whole: WholeComponentAcquisition,
    supports: Sequence[FrozenNodeSupport],
) -> dict[str, Any]:
    """Acquire the exact confirmation all-U/S/U\\S set-response panel."""
    if (
        not isinstance(whole, WholeComponentAcquisition)
        or whole.split != "confirmation"
    ):
        raise ValueError("set interventions require a confirmation whole atlas")
    if whole.case_id != case_id:
        raise ValueError("case ID differs from confirmation whole atlas")
    plan = validate_confirmation_supports(
        case_id=case_id,
        component_ids=whole.component_ids,
        supports=supports,
    )
    acquired = acquire_set_interventions(
        runner=runner,
        prepared=whole._prepared.normal,
        plan=plan,
    )
    all_output = _scientific_output(acquired.all_nodes_output, label="all-U set output")
    selected_outputs = _scientific_output(
        acquired.selected_outputs, label="selected set outputs"
    )
    complement_outputs = _scientific_output(
        acquired.complement_outputs, label="complement set outputs"
    )
    records: list[dict[str, Any]] = []
    for index, support in enumerate(plan.supports):
        records.append(
            {
                "support_id": support.support_id,
                "kind": support.kind,
                "method": support.method,
                "quantile": float(support.quantile),
                "replicate": support.replicate_index,
                "selected_nodes": support.selected_nodes,
                "selected_mask": tuple(
                    bool(value) for value in plan.selected_masks[index]
                ),
                "complement_mask": tuple(
                    bool(value) for value in plan.complement_masks[index]
                ),
                "selected_corrupt_into_clean_output": selected_outputs[index],
                "complement_corrupt_into_clean_output": complement_outputs[index],
            }
        )
    deterministic_records = tuple(
        record for record in records if record["kind"] == "deterministic"
    )
    random_records = tuple(record for record in records if record["kind"] == "random")
    q50_by_method = {
        record["method"]: record
        for record in deterministic_records
        if record["quantile"] == CONFIRMATORY_SUPPORT_FRACTION
    }
    if tuple(q50_by_method) != DETERMINISTIC_METHOD_ORDER:
        raise RuntimeError("q=.5 deterministic record order changed")
    singleton_curve = tuple(
        record for record in deterministic_records if record["method"] == PRIMARY_METHOD
    )
    return {
        "schema_version": "e32_3_confirmation_set_interventions_v1",
        "case_id": case_id,
        "component_ids": whole.component_ids,
        "all_nodes_mask": tuple(True for _node in whole.component_ids),
        "empty_mask_reuses_clean_output": True,
        "empty_clean_output": whole.clean_output,
        "all_corrupt_into_clean_output": all_output,
        "deterministic_set_outputs": q50_by_method,
        "singleton_curve_set_outputs": singleton_curve,
        "random_set_outputs": random_records,
        "all_support_records": tuple(records),
        "query_accounting": dict(acquired.query_accounting),
    }


def whole_component_acquisition_to_dict(
    value: WholeComponentAcquisition,
) -> dict[str, Any]:
    """Remove live prepared-runner custody before serialization."""
    if not isinstance(value, WholeComponentAcquisition):
        raise TypeError("value must be WholeComponentAcquisition")
    return {
        "schema_version": "e32_3_whole_component_atlas_v1",
        "case_id": value.case_id,
        "split": value.split,
        "component_ids": value.component_ids,
        "whole_edges": value.whole_edges,
        "condition_ids": value.condition_ids,
        "condition_masks": value.condition_masks,
        "pair_component_indices": value.pair_component_indices,
        "clean_inputs": value.clean_inputs,
        "corrupt_inputs": value.corrupt_inputs,
        "clean_targets": value.clean_targets,
        "corrupt_targets": value.corrupt_targets,
        "clean_output": value.clean_output,
        "corrupt_output": value.corrupt_output,
        "singleton_outputs": value.singleton_outputs,
        "pair_outputs": value.pair_outputs,
        "pair_interactions": value.pair_interactions,
        "clean_activations": dict(value.clean_activations),
        "corrupt_activations": dict(value.corrupt_activations),
        "activation_shift_rms": value.activation_shift_rms,
        "batch_invariance_smoke": value.batch_invariance_smoke,
        "query_accounting": dict(value.query_accounting),
    }


__all__ = [
    "BidirectionalComponentPair",
    "FrozenNodeSupport",
    "SetInterventionOutputs",
    "SetMaskPlan",
    "WholeComponentAcquisition",
    "acquire_confirmation_set_interventions",
    "acquire_set_interventions",
    "acquire_whole_component_atlas",
    "activation_shift_rms",
    "build_cohort_component_runner",
    "build_set_mask_plan",
    "prepare_bidirectional_component_pair",
    "validate_confirmation_supports",
    "whole_component_acquisition_to_dict",
]
