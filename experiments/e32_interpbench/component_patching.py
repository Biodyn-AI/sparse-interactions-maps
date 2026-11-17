# ruff: noqa: EXE002
"""Lean all-position component patching for the E32 InterpBench panel.

The runner intentionally depends on only the small TransformerLens surface used
by InterpBench: a callable model, ``run_with_hooks``, and ``hook_dict``.  It
keeps the complete vector output.  Scalar behavior metrics belong downstream,
so the raw interaction is exactly ``Y_ij - Y_i - Y_j + Y_0`` at every output
coordinate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

PatchKind = Literal["attention_head", "mlp"]
BATCH_INVARIANCE_ATOL = 1e-4


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class PatchUnit:
    """One whole attention head or one whole MLP output."""

    component_id: str
    hook_name: str
    kind: PatchKind
    layer_index: int
    head_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.component_id, str) or not self.component_id:
            raise ValueError("component_id must be a nonempty string")
        if not isinstance(self.hook_name, str) or not self.hook_name:
            raise ValueError("hook_name must be a nonempty string")
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise ValueError("layer_index must be a nonnegative integer")
        if self.kind == "attention_head":
            if (
                isinstance(self.head_index, bool)
                or not isinstance(self.head_index, int)
                or self.head_index < 0
            ):
                raise ValueError("attention-head units require a nonnegative head")
        elif self.kind == "mlp":
            if self.head_index is not None:
                raise ValueError("MLP units cannot have a head index")
        else:
            raise ValueError("kind must be 'attention_head' or 'mlp'")


def build_patch_units(n_layers: int, n_heads: int) -> tuple[PatchUnit, ...]:
    """Return canonical InterpBench whole-head/MLP units in layer order."""
    layer_count = _positive_integer(n_layers, label="n_layers")
    head_count = _positive_integer(n_heads, label="n_heads")
    units: list[PatchUnit] = []
    for layer in range(layer_count):
        attention_hook = f"blocks.{layer}.attn.hook_result"
        units.extend(
            PatchUnit(
                component_id=f"{attention_hook}[{head}]",
                hook_name=attention_hook,
                kind="attention_head",
                layer_index=layer,
                head_index=head,
            )
            for head in range(head_count)
        )
        mlp_hook = f"blocks.{layer}.hook_mlp_out"
        units.append(
            PatchUnit(
                component_id=mlp_hook,
                hook_name=mlp_hook,
                kind="mlp",
                layer_index=layer,
            )
        )
    return tuple(units)


def _validated_units(units: Sequence[PatchUnit]) -> tuple[PatchUnit, ...]:
    if isinstance(units, (str, bytes)) or not isinstance(units, Sequence):
        raise TypeError("units must be a sequence")
    frozen = tuple(units)
    if not frozen or any(not isinstance(unit, PatchUnit) for unit in frozen):
        raise ValueError("units must contain at least one PatchUnit")
    ids = tuple(unit.component_id for unit in frozen)
    if len(ids) != len(set(ids)):
        raise ValueError("component IDs must be unique")
    hook_kinds: dict[str, PatchKind] = {}
    head_indices: dict[str, set[int]] = {}
    for unit in frozen:
        previous_kind = hook_kinds.setdefault(unit.hook_name, unit.kind)
        if previous_kind != unit.kind:
            raise ValueError("one hook cannot mix attention and MLP units")
        if (
            unit.kind == "mlp"
            and sum(candidate.hook_name == unit.hook_name for candidate in frozen) != 1
        ):
            raise ValueError("each MLP hook must map to exactly one unit")
        if unit.kind == "attention_head":
            assert unit.head_index is not None
            used = head_indices.setdefault(unit.hook_name, set())
            if unit.head_index in used:
                raise ValueError("attention-head indices must be unique per hook")
            used.add(unit.head_index)
    return frozen


@dataclass(frozen=True)
class ComponentAtlas:
    """Empty, singleton, and topology-directed two-component interventions.

    A two-component patch mask is itself symmetric.  Its source/target arrow is
    inherited from the adapter's valid transformer DAG, not identified by this
    simultaneous intervention.
    """

    units: tuple[PatchUnit, ...]
    condition_ids: tuple[str, ...]
    condition_masks: torch.Tensor
    singleton_condition_indices: tuple[int, ...]
    pair_condition_indices: tuple[int, ...]
    pair_component_indices: tuple[tuple[int, int], ...]
    directed_edges: tuple[tuple[str, str], ...]

    @property
    def condition_count(self) -> int:
        return len(self.condition_ids)


def _execution_directed_edges(
    units: tuple[PatchUnit, ...],
) -> tuple[tuple[str, str], ...]:
    """Return the causal order induced by whole heads followed by each MLP."""
    edges = []
    for source in units:
        for target in units:
            earlier_layer = source.layer_index < target.layer_index
            head_to_same_layer_mlp = (
                source.layer_index == target.layer_index
                and source.kind == "attention_head"
                and target.kind == "mlp"
            )
            if earlier_layer or head_to_same_layer_mlp:
                edges.append((source.component_id, target.component_id))
    return tuple(edges)


def _validated_directed_edges(
    units: tuple[PatchUnit, ...],
    valid_directed_edges: Sequence[tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...]:
    allowed = _execution_directed_edges(units)
    if valid_directed_edges is None:
        return allowed
    if isinstance(valid_directed_edges, (str, bytes)) or not isinstance(
        valid_directed_edges, Sequence
    ):
        raise TypeError("valid_directed_edges must be a sequence")
    declared = tuple(tuple(edge) for edge in valid_directed_edges)
    if any(
        len(edge) != 2 or not isinstance(edge[0], str) or not isinstance(edge[1], str)
        for edge in declared
    ):
        raise ValueError("valid directed edges must be string pairs")
    if len(declared) != len(set(declared)):
        raise ValueError("valid directed edges must be unique")
    unit_ids = frozenset(unit.component_id for unit in units)
    patchable = tuple(
        edge for edge in declared if edge[0] in unit_ids and edge[1] in unit_ids
    )
    invalid = tuple(edge for edge in patchable if edge not in set(allowed))
    if invalid:
        raise ValueError(
            f"declared edges violate transformer execution order: {invalid!r}"
        )
    if not patchable:
        raise ValueError("declared edge universe contains no patchable unit pair")
    return patchable


def build_component_atlas(
    units: Sequence[PatchUnit],
    *,
    valid_directed_edges: Sequence[tuple[str, str]] | None = None,
) -> ComponentAtlas:
    """Enumerate empty, singletons, then adapter-valid directed patch pairs."""
    frozen = _validated_units(units)
    unit_count = len(frozen)
    directed_edges = _validated_directed_edges(frozen, valid_directed_edges)
    index_by_component = {unit.component_id: index for index, unit in enumerate(frozen)}
    rows = [torch.zeros(unit_count, dtype=torch.bool)]
    condition_ids = ["empty"]
    singleton_indices: list[int] = []
    for unit_index, unit in enumerate(frozen):
        row = torch.zeros(unit_count, dtype=torch.bool)
        row[unit_index] = True
        singleton_indices.append(len(rows))
        rows.append(row)
        condition_ids.append(f"singleton:{unit.component_id}")
    pair_indices: list[int] = []
    pair_components: list[tuple[int, int]] = []
    for source_id, target_id in directed_edges:
        source = index_by_component[source_id]
        target = index_by_component[target_id]
        row = torch.zeros(unit_count, dtype=torch.bool)
        row[[source, target]] = True
        pair_indices.append(len(rows))
        pair_components.append((source, target))
        rows.append(row)
        condition_ids.append(f"pair:{source_id}->{target_id}")
    return ComponentAtlas(
        units=frozen,
        condition_ids=tuple(condition_ids),
        condition_masks=torch.stack(rows),
        singleton_condition_indices=tuple(singleton_indices),
        pair_condition_indices=tuple(pair_indices),
        pair_component_indices=tuple(pair_components),
        directed_edges=directed_edges,
    )


@dataclass(frozen=True)
class PreparedComponentPair:
    """Clean base plus corrupt-donor activations captured exactly once."""

    clean_inputs: torch.Tensor = field(repr=False)
    corrupt_inputs: torch.Tensor = field(repr=False)
    clean_output: torch.Tensor
    corrupt_output: torch.Tensor
    donor_activations: Mapping[str, torch.Tensor] = field(repr=False)
    batch_size: int
    sequence_length: int
    _runner: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class ComponentAtlasResult:
    """Full vector outputs and exact raw pair second differences."""

    atlas: ComponentAtlas
    condition_outputs: torch.Tensor
    singleton_outputs: torch.Tensor
    pair_outputs: torch.Tensor
    pair_interactions: torch.Tensor
    raw_pair_interactions: torch.Tensor
    query_accounting: Mapping[str, int | bool | str]


def _finite_tensor(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{label} must be finite")
    return value


def class_center_logits(logits: torch.Tensor) -> torch.Tensor:
    """Remove the softmax-invisible additive offset at each output position."""
    values = _finite_tensor(logits, label="logits")
    if values.shape[-1] < 2:
        raise ValueError("logits require at least two output classes")
    return values - values.mean(dim=-1, keepdim=True)


def vector_pair_interactions(
    *,
    empty_output: torch.Tensor,
    singleton_outputs: torch.Tensor,
    pair_outputs: torch.Tensor,
    pair_component_indices: Sequence[tuple[int, int]],
    center: bool = True,
) -> torch.Tensor:
    """Return vector ``Yij - Yi - Yj + Y0`` for every declared pair.

    Scientific E32 scores use class-centered logits (the default), which are
    invariant to arbitrary per-position additive logit offsets.  ``center=False``
    retains gauge-dependent raw-J only as a descriptive diagnostic.
    """
    empty = _finite_tensor(empty_output, label="empty output")
    singletons = _finite_tensor(singleton_outputs, label="singleton outputs")
    pairs = _finite_tensor(pair_outputs, label="pair outputs")
    if singletons.ndim != empty.ndim + 1 or tuple(singletons.shape[1:]) != tuple(
        empty.shape
    ):
        raise ValueError("singleton outputs must prepend one component axis")
    if pairs.ndim != empty.ndim + 1 or tuple(pairs.shape[1:]) != tuple(empty.shape):
        raise ValueError("pair outputs must prepend one pair axis")
    indices = tuple(pair_component_indices)
    if len(indices) != len(pairs):
        raise ValueError("pair indices must align exactly with pair outputs")
    if len(indices) != len(set(indices)) or any(
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, int)
        or not isinstance(right, int)
        or not 0 <= left < right < len(singletons)
        for left, right in indices
    ):
        raise ValueError("pair indices must be unique ordered component pairs")
    left = torch.tensor([pair[0] for pair in indices], dtype=torch.long)
    right = torch.tensor([pair[1] for pair in indices], dtype=torch.long)
    if center:
        empty = class_center_logits(empty)
        singletons = class_center_logits(singletons)
        pairs = class_center_logits(pairs)
    return pairs - singletons[left] - singletons[right] + empty.unsqueeze(0)


class ComponentPatchRunner:
    """Patch corrupt whole-head/MLP outputs into a clean InterpBench batch."""

    def __init__(
        self,
        model: Any,
        units: Sequence[PatchUnit],
        *,
        condition_batch_size: int = 32,
        valid_directed_edges: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        if not callable(model) or not hasattr(model, "run_with_hooks"):
            raise TypeError("model must be callable and provide run_with_hooks")
        self.model = model
        self.units = _validated_units(units)
        self.valid_directed_edges = _validated_directed_edges(
            self.units, valid_directed_edges
        )
        self.condition_batch_size = _positive_integer(
            condition_batch_size, label="condition_batch_size"
        )
        hook_dict = getattr(model, "hook_dict", None)
        if not isinstance(hook_dict, Mapping):
            raise TypeError("model.hook_dict must be a mapping")
        missing = sorted({unit.hook_name for unit in self.units} - set(hook_dict))
        if missing:
            raise ValueError(f"model is missing required hooks: {missing!r}")
        # TransformerLens 1.19's run_with_hooks calls model.forward directly,
        # bypassing hooks registered on the top-level model.  Its embedding
        # module is nevertheless executed exactly once in every full forward.
        # Prefer that true internal boundary; keep fallbacks for lightweight
        # compatible test adapters.
        boundary = getattr(model, "embed", None)
        boundary_name = "model.embed"
        if not hasattr(boundary, "register_forward_pre_hook"):
            boundary = hook_dict.get("hook_embed")
            boundary_name = "hook_embed"
        if not hasattr(boundary, "register_forward_pre_hook"):
            boundary = model
            boundary_name = "top_level_model"
        if not hasattr(boundary, "register_forward_pre_hook"):
            raise TypeError("model lacks a live forward-counting boundary")
        self._live_forward_boundary = boundary
        self.live_forward_boundary_name = boundary_name
        self._physical_forwards = 0
        self._model_examples = 0

    def forward_counts(self) -> dict[str, int]:
        return {
            "physical_forwards": self._physical_forwards,
            "model_examples": self._model_examples,
        }

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device(
                getattr(getattr(self.model, "cfg", None), "device", "cpu")
            )

    def _counted_model_call(
        self,
        inputs: torch.Tensor,
        *,
        hooks: Sequence[tuple[str, Any]] = (),
    ) -> torch.Tensor:
        observed_calls = 0

        def observe(
            _module: Any, positional: tuple[Any, ...]
        ) -> None:  # actual model boundary
            nonlocal observed_calls
            if not positional or not isinstance(positional[0], torch.Tensor):
                raise RuntimeError("live model forward did not receive tensor inputs")
            live_inputs = positional[0]
            if live_inputs.ndim < 1:
                raise RuntimeError("live model inputs have no batch axis")
            observed_calls += 1
            self._physical_forwards += 1
            self._model_examples += int(live_inputs.shape[0])

        handle = self._live_forward_boundary.register_forward_pre_hook(observe)
        try:
            with torch.inference_mode():
                if hooks:
                    output = self.model.run_with_hooks(
                        inputs,
                        fwd_hooks=list(hooks),
                    )
                else:
                    output = self.model(inputs)
        finally:
            handle.remove()
        if observed_calls != 1:
            raise RuntimeError(
                "each logical model call must cross the live model boundary once"
            )
        output = _finite_tensor(output, label="model output")
        if output.ndim < 2 or output.shape[0] != inputs.shape[0]:
            raise RuntimeError("model output lost its batch axis")
        return output

    @staticmethod
    def _validated_inputs(value: Any, *, label: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"{label} must be a rank-two token tensor")
        if value.shape[0] < 1 or value.shape[1] < 1:
            raise ValueError(f"{label} must have nonempty batch and position axes")
        if value.dtype == torch.bool or value.is_floating_point():
            raise ValueError(f"{label} must contain integer token IDs")
        return value.detach().clone()

    def _validate_donor_activation(
        self,
        activation: torch.Tensor,
        *,
        hook_name: str,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        units = tuple(unit for unit in self.units if unit.hook_name == hook_name)
        expected_ndim = 4 if units[0].kind == "attention_head" else 3
        if (
            activation.ndim != expected_ndim
            or activation.shape[0] != batch_size
            or activation.shape[1] != sequence_length
        ):
            raise RuntimeError(f"captured activation shape changed at {hook_name}")
        if units[0].kind == "attention_head":
            maximum_head = max(int(unit.head_index) for unit in units)
            if activation.shape[2] <= maximum_head:
                raise RuntimeError(f"captured head axis is too short at {hook_name}")

    def prepare_pair(self, official_pair: Any) -> PreparedComponentPair:
        """Capture one batched official clean/corrupt pair without patching."""
        try:
            clean_inputs = self._validated_inputs(
                official_pair.clean_inputs, label="clean_inputs"
            )
            corrupt_inputs = self._validated_inputs(
                official_pair.corrupt_inputs, label="corrupt_inputs"
            )
        except AttributeError as error:
            raise TypeError("official_pair lacks clean/corrupt inputs") from error
        if clean_inputs.shape != corrupt_inputs.shape:
            raise ValueError("clean and corrupt inputs must have identical shapes")
        device = self._model_device()
        clean_inputs = clean_inputs.to(device)
        corrupt_inputs = corrupt_inputs.to(device)
        hook_names = tuple(dict.fromkeys(unit.hook_name for unit in self.units))
        captures: dict[str, torch.Tensor] = {}

        def capture(name: str):
            def callback(activation: torch.Tensor, hook: Any) -> torch.Tensor:
                del hook
                if name in captures:
                    raise RuntimeError(f"hook fired more than once: {name}")
                captures[name] = (
                    _finite_tensor(activation, label=f"donor activation {name}")
                    .detach()
                    .clone()
                )
                return activation

            return callback

        corrupt_output = self._counted_model_call(
            corrupt_inputs,
            hooks=tuple((name, capture(name)) for name in hook_names),
        )
        if tuple(captures) != hook_names:
            raise RuntimeError("donor forward did not call every required hook")
        batch_size, sequence_length = clean_inputs.shape
        for name, activation in captures.items():
            self._validate_donor_activation(
                activation,
                hook_name=name,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
        clean_output = self._counted_model_call(clean_inputs)
        if clean_output.shape != corrupt_output.shape:
            raise RuntimeError("clean and corrupt output shapes differ")
        return PreparedComponentPair(
            clean_inputs=clean_inputs,
            corrupt_inputs=corrupt_inputs,
            clean_output=clean_output.detach().cpu().clone(),
            corrupt_output=corrupt_output.detach().cpu().clone(),
            donor_activations=captures,
            batch_size=batch_size,
            sequence_length=sequence_length,
            _runner=self,
        )

    def _validated_masks(self, masks: Any) -> torch.Tensor:
        if (
            not isinstance(masks, torch.Tensor)
            or masks.dtype != torch.bool
            or masks.ndim != 2
            or masks.shape[0] < 1
        ):
            raise ValueError("condition masks must be a nonempty boolean matrix")
        if masks.shape[1] != len(self.units):
            raise ValueError("condition-mask unit count does not match runner")
        return masks.detach().cpu().clone()

    def _patch_hooks(
        self,
        prepared: PreparedComponentPair,
        masks: torch.Tensor,
    ) -> tuple[tuple[str, Any], ...]:
        hook_names = tuple(dict.fromkeys(unit.hook_name for unit in self.units))
        condition_count = len(masks)
        batch_size = prepared.batch_size
        hooks = []
        for hook_name in hook_names:
            indexed_units = tuple(
                (index, unit)
                for index, unit in enumerate(self.units)
                if unit.hook_name == hook_name
            )

            def patch(
                activation: torch.Tensor,
                hook: Any,
                *,
                name: str = hook_name,
                members: tuple[tuple[int, PatchUnit], ...] = indexed_units,
            ) -> torch.Tensor:
                del hook
                donor = prepared.donor_activations[name]
                if activation.shape[0] != condition_count * batch_size:
                    raise RuntimeError("patched activation batch axis changed")
                if tuple(activation.shape[1:]) != tuple(donor.shape[1:]):
                    raise RuntimeError(f"recipient/donor shape mismatch at {name}")
                recipient = activation.reshape(
                    condition_count, batch_size, *activation.shape[1:]
                )
                donor = donor.to(device=activation.device, dtype=activation.dtype)
                donor = donor.unsqueeze(0).expand(condition_count, *donor.shape)
                if members[0][1].kind == "mlp":
                    unit_index = members[0][0]
                    selector = (
                        masks[:, unit_index]
                        .to(activation.device)
                        .reshape(condition_count, 1, 1, 1)
                    )
                    patched = torch.where(selector, donor, recipient)
                else:
                    patched = recipient.clone()
                    for unit_index, unit in members:
                        assert unit.head_index is not None
                        selector = (
                            masks[:, unit_index]
                            .to(activation.device)
                            .reshape(condition_count, 1, 1, 1)
                        )
                        patched[:, :, :, unit.head_index, :] = torch.where(
                            selector,
                            donor[:, :, :, unit.head_index, :],
                            recipient[:, :, :, unit.head_index, :],
                        )
                return patched.reshape_as(activation)

            hooks.append((hook_name, patch))
        return tuple(hooks)

    def outputs_for_masks(
        self,
        prepared: PreparedComponentPair,
        condition_masks: torch.Tensor,
        *,
        condition_batch_size: int | None = None,
    ) -> torch.Tensor:
        """Return ``[condition, example, ...output]`` for arbitrary masks."""
        if (
            not isinstance(prepared, PreparedComponentPair)
            or prepared._runner is not self
        ):
            raise ValueError("prepared pair belongs to a different runner")
        masks = self._validated_masks(condition_masks)
        chunk_size = (
            self.condition_batch_size
            if condition_batch_size is None
            else _positive_integer(condition_batch_size, label="condition_batch_size")
        )
        chunks = []
        for start in range(0, len(masks), chunk_size):
            chunk = masks[start : start + chunk_size]
            condition_count = len(chunk)
            expanded_inputs = (
                prepared.clean_inputs.unsqueeze(0)
                .expand(condition_count, *prepared.clean_inputs.shape)
                .reshape(
                    condition_count * prepared.batch_size, prepared.sequence_length
                )
            )
            output = self._counted_model_call(
                expanded_inputs,
                hooks=self._patch_hooks(prepared, chunk),
            )
            reshaped = output.reshape(
                condition_count, prepared.batch_size, *output.shape[1:]
            )
            chunks.append(reshaped.detach().cpu().clone())
        return torch.cat(chunks, dim=0)

    def acquire_atlas(self, prepared: PreparedComponentPair) -> ComponentAtlasResult:
        """Acquire every nonempty atlas row; reuse the observed clean output as Y0."""
        if (
            not isinstance(prepared, PreparedComponentPair)
            or prepared._runner is not self
        ):
            raise ValueError("prepared pair belongs to a different runner")
        atlas = build_component_atlas(
            self.units, valid_directed_edges=self.valid_directed_edges
        )
        before = self.forward_counts()
        nonempty_outputs = self.outputs_for_masks(prepared, atlas.condition_masks[1:])
        condition_outputs = torch.cat(
            (prepared.clean_output.unsqueeze(0), nonempty_outputs), dim=0
        )
        singleton_outputs = condition_outputs[list(atlas.singleton_condition_indices)]
        pair_outputs = condition_outputs[list(atlas.pair_condition_indices)]
        interactions = vector_pair_interactions(
            empty_output=prepared.clean_output,
            singleton_outputs=singleton_outputs,
            pair_outputs=pair_outputs,
            pair_component_indices=atlas.pair_component_indices,
        )
        raw_interactions = vector_pair_interactions(
            empty_output=prepared.clean_output,
            singleton_outputs=singleton_outputs,
            pair_outputs=pair_outputs,
            pair_component_indices=atlas.pair_component_indices,
            center=False,
        )
        after = self.forward_counts()
        patched_conditions = atlas.condition_count - 1
        return ComponentAtlasResult(
            atlas=atlas,
            condition_outputs=condition_outputs,
            singleton_outputs=singleton_outputs,
            pair_outputs=pair_outputs,
            pair_interactions=interactions,
            raw_pair_interactions=raw_interactions,
            query_accounting={
                "empty_reuses_prepared_clean_output": True,
                "empty_additional_model_examples": 0,
                "patched_condition_count": patched_conditions,
                "example_count_per_condition": prepared.batch_size,
                "patched_model_examples": after["model_examples"]
                - before["model_examples"],
                "patched_physical_forwards": after["physical_forwards"]
                - before["physical_forwards"],
                "interaction_definition": (
                    "class_center(Yij)-class_center(Yi)-class_center(Yj)"
                    "+class_center(Y0)"
                ),
            },
        )

    def batch_invariance_smoke(
        self,
        prepared: PreparedComponentPair,
        condition_masks: torch.Tensor,
        *,
        candidate_condition_batch_size: int,
        absolute_tolerance: float = BATCH_INVARIANCE_ATOL,
    ) -> dict[str, int | float | bool]:
        """Compare one-condition execution with the intended condition batching."""
        masks = self._validated_masks(condition_masks)
        candidate = _positive_integer(
            candidate_condition_batch_size,
            label="candidate_condition_batch_size",
        )
        if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
            raise ValueError("absolute_tolerance must be finite and nonnegative")
        before = self.forward_counts()
        reference = self.outputs_for_masks(prepared, masks, condition_batch_size=1)
        batched = self.outputs_for_masks(
            prepared, masks, condition_batch_size=candidate
        )
        maximum_error = float((reference - batched).abs().max())
        empty_rows = ~masks.any(dim=1)
        empty_no_op_evaluated = bool(empty_rows.any())
        empty_no_op_error = 0.0
        if empty_no_op_evaluated:
            expected_clean = prepared.clean_output.unsqueeze(0).expand(
                int(empty_rows.sum()), *prepared.clean_output.shape
            )
            empty_no_op_error = float(
                (reference[empty_rows] - expected_clean).abs().max()
            )
        after = self.forward_counts()
        return {
            "passed": bool(
                torch.isfinite(reference).all()
                and torch.isfinite(batched).all()
                and maximum_error <= absolute_tolerance
                and (
                    not empty_no_op_evaluated or empty_no_op_error <= absolute_tolerance
                )
            ),
            "condition_count": len(masks),
            "example_count_per_condition": prepared.batch_size,
            "reference_condition_batch_size": 1,
            "candidate_condition_batch_size": candidate,
            "maximum_absolute_error": maximum_error,
            "empty_no_op_evaluated": empty_no_op_evaluated,
            "empty_no_op_maximum_absolute_error": empty_no_op_error,
            "absolute_tolerance": absolute_tolerance,
            "model_examples": after["model_examples"] - before["model_examples"],
            "physical_forwards": after["physical_forwards"]
            - before["physical_forwards"],
        }


__all__ = [
    "BATCH_INVARIANCE_ATOL",
    "ComponentAtlas",
    "ComponentAtlasResult",
    "ComponentPatchRunner",
    "PatchUnit",
    "PreparedComponentPair",
    "build_component_atlas",
    "build_patch_units",
    "class_center_logits",
    "vector_pair_interactions",
]
