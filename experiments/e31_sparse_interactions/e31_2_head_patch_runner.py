# ruff: noqa: EXE002
"""Exact all-position GPT-2 attention-head patches for E31.2.

The coarse E31.1 intervention replaces an attention or MLP residual update at
``c_proj`` output.  This runner keeps that audited intervention available and
adds a finer attention-head intervention immediately before ``attn.c_proj``.
GPT-2 merges its heads contiguously on the final axis, so replacing one clean
donor slice there is exactly equivalent to replacing that head's bias-free
contribution through the output projection.  The projection bias is included
once, and every downstream component is recomputed normally.

This module performs no model loading, sealing, filesystem access, or protocol
selection.  It is an in-memory numerical primitive for a later sealed driver.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_protocol import (
    NUMERICAL_GATES,
)
from experiments.e29_gpt2_stable_head_interactions.e29_15_ioi_patch_runner import (
    COMPONENT_COUNT as COARSE_COMPONENT_COUNT,
)
from experiments.e29_gpt2_stable_head_interactions.e29_15_ioi_patch_runner import (
    LAYER_COUNT,
    RESPONSE_COORDINATE,
    PreparedIOIPatchContext,
)
from experiments.e31_sparse_interactions.prospective_acquisition import (
    E311DirectedIOIPatchRunner,
)

ATTENTION_HEAD_COUNT = 12
ATTENTION_HEAD_SHAPE = (LAYER_COUNT, ATTENTION_HEAD_COUNT)
HEAD_COMPONENT_COUNT = LAYER_COUNT * ATTENTION_HEAD_COUNT
COMPOSITE_RECONSTRUCTION_ATOL = 1e-10
COMPOSITE_RECONSTRUCTION_RTOL = 1e-10
WHOLE_ATTENTION_EQUIVALENCE_ATOL = 1e-9
WHOLE_ATTENTION_EQUIVALENCE_RTOL = 1e-9


@dataclass(frozen=True)
class HeadPatchResponses:
    """Auditable raw/normalized responses and W_O reconstruction diagnostics."""

    normalized_transfer: torch.Tensor
    raw_donor_minus_base_margin: torch.Tensor
    raw_margin_shift_from_base: torch.Tensor
    composite_reconstruction: Mapping[str, Any]


@dataclass(frozen=True, eq=False)
class PreparedHeadPatchContext:
    """Clean E29.15 endpoints plus clean pre-W_O head-vector caches."""

    base_input_ids: torch.Tensor = field(repr=False)
    donor_input_ids: torch.Tensor = field(repr=False)
    base_logits: torch.Tensor = field(repr=False)
    donor_logits: torch.Tensor = field(repr=False)
    base_target_token_id: int
    donor_target_token_id: int
    clean_base_donor_minus_base_margin: float
    clean_donor_donor_minus_base_margin: float
    clean_base_correct: bool
    clean_donor_correct: bool
    clean_endpoint: float
    clean_metadata: Mapping[str, Any]
    base_merged_head_outputs: tuple[torch.Tensor, ...] = field(repr=False)
    donor_merged_head_outputs: tuple[torch.Tensor, ...] = field(repr=False)
    clean_head_norm_diagnostics: Mapping[str, Any]
    _coarse_context: PreparedIOIPatchContext = field(repr=False)
    _runner: Any = field(repr=False)

    @property
    def donor_updates(self) -> tuple[torch.Tensor, ...]:
        """Expose the unchanged E31.1 donor sublayer-update cache."""
        return self._coarse_context.donor_updates


def _projection_parameters(
    projection: Any, *, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = getattr(projection, "weight", None)
    bias = getattr(projection, "bias", None)
    if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (width, width):
        raise TypeError(
            "attention c_proj must expose Conv1D weight shape (width, width)"
        )
    if not isinstance(bias, torch.Tensor) or tuple(bias.shape) != (width,):
        raise TypeError("attention c_proj must expose one width-sized bias")
    if (
        weight.device.type != "cpu"
        or bias.device.type != "cpu"
        or weight.dtype != torch.float64
        or bias.dtype != torch.float64
    ):
        raise ValueError("attention c_proj parameters must be CPU float64")
    return weight, bias


def split_projected_head_contributions(
    merged_head_outputs: torch.Tensor,
    projection: Any,
    *,
    head_count: int = ATTENTION_HEAD_COUNT,
) -> torch.Tensor:
    """Return each head's bias-free contribution through GPT-2's ``c_proj``.

    ``merged_head_outputs`` ends in GPT-2's contiguous ``head, head_width``
    layout.  The result inserts a head axis immediately before the projected
    output axis.  Summing that axis and adding ``projection.bias`` reconstructs
    the ordinary projection output.
    """
    if not isinstance(merged_head_outputs, torch.Tensor):
        raise TypeError("merged_head_outputs must be a tensor")
    if merged_head_outputs.ndim < 2:
        raise ValueError("merged_head_outputs must include positions and width")
    if (
        merged_head_outputs.device.type != "cpu"
        or merged_head_outputs.dtype != torch.float64
    ):
        raise ValueError("merged_head_outputs must be CPU float64")
    if (
        isinstance(head_count, bool)
        or not isinstance(head_count, int)
        or head_count < 1
    ):
        raise ValueError("head_count must be a positive integer")
    width = int(merged_head_outputs.shape[-1])
    if width % head_count:
        raise ValueError("projection width must be divisible by head_count")
    weight, _bias = _projection_parameters(projection, width=width)
    head_width = width // head_count
    heads = merged_head_outputs.reshape(
        *merged_head_outputs.shape[:-1], head_count, head_width
    )
    head_weights = weight.reshape(head_count, head_width, width)
    contributions = torch.einsum("...hd,hdo->...ho", heads, head_weights)
    if not bool(torch.isfinite(contributions).all()):
        raise RuntimeError("a projected head contribution is non-finite")
    return contributions


def reconstruct_selected_head_composite(
    recipient_merged_head_outputs: torch.Tensor,
    donor_merged_head_outputs: torch.Tensor,
    selected_heads: torch.Tensor,
    projection: Any,
    *,
    head_count: int = ATTENTION_HEAD_COUNT,
) -> torch.Tensor:
    """Compose selected clean-donor heads with current recipient heads.

    Inputs use ``(batch, positions, width)`` orientation.  The donor may omit
    its batch axis, in which case the one clean donor row is broadcast.  This
    is a direct intervention reconstruction, not a sum of measured response
    coefficients or an affine calibration.
    """
    if (
        not isinstance(recipient_merged_head_outputs, torch.Tensor)
        or recipient_merged_head_outputs.ndim != 3
    ):
        raise ValueError(
            "recipient head outputs must have shape (batch, positions, width)"
        )
    recipient = recipient_merged_head_outputs
    donor = donor_merged_head_outputs
    if not isinstance(donor, torch.Tensor):
        raise TypeError("donor_merged_head_outputs must be a tensor")
    if donor.ndim == 2:
        donor = donor.unsqueeze(0).expand(len(recipient), -1, -1)
    if donor.ndim != 3 or tuple(donor.shape) != tuple(recipient.shape):
        raise ValueError("donor and recipient head outputs must align exactly")
    if not isinstance(selected_heads, torch.Tensor) or tuple(selected_heads.shape) != (
        len(recipient),
        head_count,
    ):
        raise ValueError("selected_heads must have shape (batch, head_count)")
    if not bool(torch.logical_or(selected_heads == 0, selected_heads == 1).all()):
        raise ValueError("selected_heads must be binary")
    recipient_contributions = split_projected_head_contributions(
        recipient, projection, head_count=head_count
    )
    donor_contributions = split_projected_head_contributions(
        donor, projection, head_count=head_count
    )
    selected = selected_heads.to(device="cpu", dtype=torch.bool)
    contributions = torch.where(
        selected[:, None, :, None], donor_contributions, recipient_contributions
    )
    _weight, bias = _projection_parameters(projection, width=recipient.shape[-1])
    output = contributions.sum(dim=-2) + bias
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("selected-head composite reconstruction is non-finite")
    return output


def _head_norm_diagnostics(
    model: Any,
    *,
    base_inputs: Sequence[torch.Tensor],
    donor_inputs: Sequence[torch.Tensor],
    base_outputs: Sequence[torch.Tensor],
    donor_outputs: Sequence[torch.Tensor],
) -> dict[str, Any]:
    layers = []
    maximum_error = 0.0
    finite = True
    for layer in range(LAYER_COUNT):
        projection = model.transformer.h[layer].attn.c_proj
        base_contributions = split_projected_head_contributions(
            base_inputs[layer], projection
        ).detach()
        donor_contributions = split_projected_head_contributions(
            donor_inputs[layer], projection
        ).detach()
        _weight, bias = _projection_parameters(
            projection, width=base_inputs[layer].shape[-1]
        )
        base_reconstructed = base_contributions.sum(dim=-2) + bias
        donor_reconstructed = donor_contributions.sum(dim=-2) + bias
        base_error = float(
            (base_reconstructed - base_outputs[layer]).abs().max().detach()
        )
        donor_error = float(
            (donor_reconstructed - donor_outputs[layer]).abs().max().detach()
        )
        maximum_error = max(maximum_error, base_error, donor_error)

        def l2_by_head(values: torch.Tensor) -> list[float]:
            flattened = values.movedim(-2, 0).reshape(ATTENTION_HEAD_COUNT, -1)
            return [
                float(value)
                for value in torch.linalg.vector_norm(flattened, dim=1).detach()
            ]

        def rms_by_head(values: torch.Tensor) -> list[float]:
            flattened = values.movedim(-2, 0).reshape(ATTENTION_HEAD_COUNT, -1)
            return [
                float(value) for value in flattened.square().mean(dim=1).sqrt().detach()
            ]

        delta = donor_contributions - base_contributions
        layer_finite = bool(
            torch.isfinite(base_contributions).all()
            and torch.isfinite(donor_contributions).all()
            and torch.isfinite(delta).all()
        )
        finite = finite and layer_finite
        layers.append(
            {
                "layer": layer,
                "base_projected_l2_by_head": l2_by_head(base_contributions),
                "donor_projected_l2_by_head": l2_by_head(donor_contributions),
                "donor_minus_base_projected_l2_by_head": l2_by_head(delta),
                "base_projected_rms_by_head": rms_by_head(base_contributions),
                "donor_projected_rms_by_head": rms_by_head(donor_contributions),
                "donor_minus_base_projected_rms_by_head": rms_by_head(delta),
                "clean_base_reconstruction_absolute_error": base_error,
                "clean_donor_reconstruction_absolute_error": donor_error,
                "finite": layer_finite,
            }
        )
    passed = finite and maximum_error <= COMPOSITE_RECONSTRUCTION_ATOL
    return {
        "head_count": HEAD_COMPONENT_COUNT,
        "norm_scope": "all_positions_bias_free_projected_head_contribution",
        "maximum_clean_reconstruction_absolute_error": maximum_error,
        "reconstruction_absolute_tolerance": COMPOSITE_RECONSTRUCTION_ATOL,
        "finite": finite,
        "passed": passed,
        "layers": tuple(layers),
    }


class E312HeadPatchRunner(E311DirectedIOIPatchRunner):
    """Patch GPT-2-small coarse sublayers and individual attention heads."""

    def __init__(self, model: Any, tokenizer: Any, *, batch_size: int) -> None:
        super().__init__(model, tokenizer, batch_size=batch_size)
        if int(getattr(model.config, "n_head", -1)) != ATTENTION_HEAD_COUNT:
            raise ValueError("E31.2 requires exactly 12 attention heads per layer")
        embedding_width = int(getattr(model.config, "n_embd", -1))
        if embedding_width < 1 or embedding_width % ATTENTION_HEAD_COUNT:
            raise ValueError("E31.2 requires n_embd divisible by 12 heads")
        for layer in range(LAYER_COUNT):
            _projection_parameters(
                model.transformer.h[layer].attn.c_proj, width=embedding_width
            )
        self.embedding_width = embedding_width
        self.head_width = embedding_width // ATTENTION_HEAD_COUNT
        self.head_components = tuple(
            (layer, head)
            for layer in range(LAYER_COUNT)
            for head in range(ATTENTION_HEAD_COUNT)
        )

    def prepare_context(self, record: Mapping[str, Any]) -> PreparedHeadPatchContext:
        """Run the same two E29.15 clean forwards and capture pre-W_O heads."""
        input_captures: list[list[torch.Tensor]] = [[] for _ in range(LAYER_COUNT)]
        output_captures: list[list[torch.Tensor]] = [[] for _ in range(LAYER_COUNT)]
        handles = []
        for layer in range(LAYER_COUNT):
            projection = self.model.transformer.h[layer].attn.c_proj

            def capture_input(
                _module: Any,
                inputs: tuple[torch.Tensor, ...],
                *,
                index: int = layer,
            ) -> None:
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    raise RuntimeError("attention c_proj did not receive a tensor")
                input_captures[index].append(inputs[0][0].detach().clone())

            def capture_output(
                _module: Any,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                index: int = layer,
            ) -> None:
                if not isinstance(output, torch.Tensor):
                    raise TypeError("attention c_proj did not return a tensor")
                output_captures[index].append(output[0].detach().clone())

            handles.append(projection.register_forward_pre_hook(capture_input))
            handles.append(projection.register_forward_hook(capture_output))
        try:
            coarse_context = super().prepare_context(record)
        finally:
            for handle in handles:
                handle.remove()
        if any(len(rows) != 2 for rows in input_captures) or any(
            len(rows) != 2 for rows in output_captures
        ):
            raise RuntimeError(
                "clean forwards did not call every attention c_proj twice"
            )
        # E29.15 performs the donor forward first and the base forward second.
        donor_inputs = tuple(rows[0] for rows in input_captures)
        base_inputs = tuple(rows[1] for rows in input_captures)
        donor_outputs = tuple(rows[0] for rows in output_captures)
        base_outputs = tuple(rows[1] for rows in output_captures)
        expected_shape = (
            int(coarse_context.base_input_ids.shape[1]),
            self.embedding_width,
        )
        if any(
            tuple(value.shape) != expected_shape
            for value in (*base_inputs, *donor_inputs)
        ):
            raise RuntimeError("captured merged head output shape changed")
        diagnostics = _head_norm_diagnostics(
            self.model,
            base_inputs=base_inputs,
            donor_inputs=donor_inputs,
            base_outputs=base_outputs,
            donor_outputs=donor_outputs,
        )
        if not diagnostics["passed"]:
            raise RuntimeError("clean projected-head reconstruction failed")
        metadata = {
            **dict(coarse_context.clean_metadata),
            "coarse_component_mask_shape": [COARSE_COMPONENT_COUNT],
            "attention_head_mask_shape": list(ATTENTION_HEAD_SHAPE),
            "head_component_count": HEAD_COMPONENT_COUNT,
            "attention_head_count_per_layer": ATTENTION_HEAD_COUNT,
            "attention_head_width": self.head_width,
            "head_patch_site": "attention.c_proj input",
            "head_patch_direction": "clean_donor_head_vector_into_base_recipient",
            "head_patch_scope": "all_positions",
            "projected_head_definition": (
                "contiguous_pre_c_proj_head_vector_times_corresponding_W_O_rows_"
                "excluding_shared_bias"
            ),
            "selected_head_composite": (
                "selected_clean_donor_plus_unselected_current_recipient_projected_"
                "head_contributions_plus_c_proj_bias_once"
            ),
            "whole_attention_fine_patch_equivalence_tolerance": (
                WHOLE_ATTENTION_EQUIVALENCE_ATOL
            ),
            "simultaneous_whole_attention_and_fine_head_policy": "forbidden",
            "response_coordinate": RESPONSE_COORDINATE,
        }
        return PreparedHeadPatchContext(
            base_input_ids=coarse_context.base_input_ids.detach().clone(),
            donor_input_ids=coarse_context.donor_input_ids.detach().clone(),
            base_logits=coarse_context.base_logits.detach().clone(),
            donor_logits=coarse_context.donor_logits.detach().clone(),
            base_target_token_id=coarse_context.base_target_token_id,
            donor_target_token_id=coarse_context.donor_target_token_id,
            clean_base_donor_minus_base_margin=(
                coarse_context.clean_base_donor_minus_base_margin
            ),
            clean_donor_donor_minus_base_margin=(
                coarse_context.clean_donor_donor_minus_base_margin
            ),
            clean_base_correct=coarse_context.clean_base_correct,
            clean_donor_correct=coarse_context.clean_donor_correct,
            clean_endpoint=coarse_context.clean_endpoint,
            clean_metadata=MappingProxyType(metadata),
            base_merged_head_outputs=tuple(value.clone() for value in base_inputs),
            donor_merged_head_outputs=tuple(value.clone() for value in donor_inputs),
            clean_head_norm_diagnostics=MappingProxyType(diagnostics),
            _coarse_context=coarse_context,
            _runner=self,
        )

    def _validated_intervention_masks(
        self,
        coarse_masks: torch.Tensor,
        attention_head_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(coarse_masks, torch.Tensor):
            raise TypeError("coarse_masks must be a tensor")
        self._validate_masks(coarse_masks)
        if (
            not isinstance(attention_head_masks, torch.Tensor)
            or attention_head_masks.ndim != 3
            or tuple(attention_head_masks.shape[1:]) != ATTENTION_HEAD_SHAPE
            or len(attention_head_masks) != len(coarse_masks)
        ):
            raise ValueError(
                "attention_head_masks must have shape (rows, 12, 12) aligned "
                "with coarse_masks"
            )
        if not bool(
            torch.logical_or(attention_head_masks == 0, attention_head_masks == 1).all()
        ):
            raise ValueError("attention_head_masks must be binary")
        canonical_coarse = coarse_masks.detach().to(device="cpu", dtype=torch.uint8)
        canonical_heads = attention_head_masks.detach().to(
            device="cpu", dtype=torch.uint8
        )
        whole_attention = canonical_coarse[:, 0::2].bool()
        fine_attention = canonical_heads.bool().any(dim=2)
        if bool(torch.logical_and(whole_attention, fine_attention).any()):
            raise ValueError(
                "whole-attention and fine-head patches cannot coexist at one layer"
            )
        return canonical_coarse.clone(), canonical_heads.clone()

    def responses_from_prepared(
        self,
        prepared: PreparedHeadPatchContext,
        coarse_masks: torch.Tensor,
        attention_head_masks: torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> HeadPatchResponses:
        """Evaluate aligned coarse/fine masks without repeating clean forwards."""
        canonical_coarse, canonical_heads = self._validated_intervention_masks(
            coarse_masks, attention_head_masks
        )
        if (
            not isinstance(prepared, PreparedHeadPatchContext)
            or prepared._runner is not self
        ):
            raise ValueError("prepared context belongs to a different E31.2 runner")
        effective_batch_size = self.batch_size if batch_size is None else batch_size
        if (
            isinstance(effective_batch_size, bool)
            or not isinstance(effective_batch_size, int)
            or effective_batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")

        margin_batches = []
        layer_errors: list[list[float]] = [[] for _ in range(LAYER_COUNT)]
        layer_relative_errors: list[list[float]] = [[] for _ in range(LAYER_COUNT)]
        evaluated_attention_rows = 0
        with torch.no_grad():
            for coarse_batch, head_batch in zip(
                canonical_coarse.split(effective_batch_size),
                canonical_heads.split(effective_batch_size),
                strict=True,
            ):
                coarse_batch = coarse_batch.to(device=self.device, dtype=torch.bool)
                head_batch = head_batch.to(device=self.device, dtype=torch.bool)
                repeated_ids = prepared.base_input_ids.expand(len(coarse_batch), -1)
                expected_outputs: dict[int, torch.Tensor] = {}
                handles = []
                for layer in range(LAYER_COUNT):
                    attention_projection = self.model.transformer.h[layer].attn.c_proj
                    selected_heads = head_batch[:, layer]
                    coarse_attention_selected = coarse_batch[:, 2 * layer]
                    coarse_mlp_selected = coarse_batch[:, 2 * layer + 1]

                    def patch_heads(
                        _module: Any,
                        inputs: tuple[torch.Tensor, ...],
                        *,
                        index: int = layer,
                        selected: torch.Tensor = selected_heads,
                        projection: Any = attention_projection,
                        pending: dict[int, torch.Tensor] = expected_outputs,
                    ) -> tuple[torch.Tensor, ...]:
                        if not inputs or not isinstance(inputs[0], torch.Tensor):
                            raise RuntimeError("attention c_proj input changed")
                        recipient = inputs[0]
                        donor = prepared.donor_merged_head_outputs[index]
                        recipient_heads = recipient.reshape(
                            len(recipient),
                            recipient.shape[1],
                            ATTENTION_HEAD_COUNT,
                            self.head_width,
                        )
                        donor_heads = donor.reshape(
                            donor.shape[0], ATTENTION_HEAD_COUNT, self.head_width
                        )
                        patched_heads = torch.where(
                            selected[:, None, :, None],
                            donor_heads[None],
                            recipient_heads,
                        )
                        patched = patched_heads.reshape_as(recipient)
                        pending[index] = reconstruct_selected_head_composite(
                            recipient, donor, selected, projection
                        )
                        return (patched, *inputs[1:])

                    def verify_and_patch_attention(
                        _module: Any,
                        _inputs: tuple[torch.Tensor, ...],
                        output: torch.Tensor,
                        *,
                        index: int = layer,
                        selected: torch.Tensor = coarse_attention_selected,
                        pending: dict[int, torch.Tensor] = expected_outputs,
                    ) -> torch.Tensor:
                        expected = pending.pop(index)
                        differences = (output - expected).abs()
                        absolute_error = float(differences.max())
                        relative_error = float(
                            (differences / expected.abs().clamp_min(1e-12)).max()
                        )
                        layer_errors[index].append(absolute_error)
                        layer_relative_errors[index].append(relative_error)
                        if not torch.allclose(
                            output,
                            expected,
                            atol=COMPOSITE_RECONSTRUCTION_ATOL,
                            rtol=COMPOSITE_RECONSTRUCTION_RTOL,
                        ):
                            raise RuntimeError(
                                "selected-head c_proj composite reconstruction failed"
                            )
                        return torch.where(
                            selected[:, None, None],
                            prepared.donor_updates[2 * index][None],
                            output,
                        )

                    mlp_projection = self.model.transformer.h[layer].mlp.c_proj

                    def patch_mlp(
                        _module: Any,
                        _inputs: tuple[torch.Tensor, ...],
                        output: torch.Tensor,
                        *,
                        index: int = layer,
                        selected: torch.Tensor = coarse_mlp_selected,
                    ) -> torch.Tensor:
                        return torch.where(
                            selected[:, None, None],
                            prepared.donor_updates[2 * index + 1][None],
                            output,
                        )

                    handles.append(
                        attention_projection.register_forward_pre_hook(patch_heads)
                    )
                    handles.append(
                        attention_projection.register_forward_hook(
                            verify_and_patch_attention
                        )
                    )
                    handles.append(mlp_projection.register_forward_hook(patch_mlp))
                try:
                    logits = self.model(input_ids=repeated_ids).logits[:, -1]
                finally:
                    for handle in handles:
                        handle.remove()
                if expected_outputs:
                    raise RuntimeError(
                        "not every expected attention projection was observed"
                    )
                evaluated_attention_rows += len(coarse_batch) * LAYER_COUNT
                self._check_clean_logits(logits)
                margins = (
                    logits[:, prepared.donor_target_token_id]
                    - logits[:, prepared.base_target_token_id]
                )
                margin_batches.append(margins.detach().cpu().double())

        raw_margin = torch.cat(margin_batches)
        raw_shift = raw_margin - prepared.clean_base_donor_minus_base_margin
        normalized = raw_shift / prepared.clean_endpoint
        response_limit = float(NUMERICAL_GATES["response_absolute_max"])
        if not bool(
            torch.isfinite(raw_margin).all()
            and torch.isfinite(raw_shift).all()
            and torch.isfinite(normalized).all()
        ):
            raise RuntimeError("an E31.2 head intervention response is non-finite")
        if float(raw_shift.abs().max()) > response_limit:
            raise RuntimeError("E31.2 response exceeds the registered magnitude gate")
        maximum_error = max(max(values) for values in layer_errors)
        maximum_relative_error = max(max(values) for values in layer_relative_errors)
        reconstruction = MappingProxyType(
            {
                "semantics": (
                    "direct_selected_donor_and_current_recipient_head_contributions_"
                    "plus_bias_once"
                ),
                "absolute_tolerance": COMPOSITE_RECONSTRUCTION_ATOL,
                "relative_tolerance": COMPOSITE_RECONSTRUCTION_RTOL,
                "maximum_absolute_error": maximum_error,
                "maximum_relative_error": maximum_relative_error,
                "maximum_absolute_error_by_layer": tuple(
                    max(values) for values in layer_errors
                ),
                "evaluated_attention_calls": evaluated_attention_rows,
                "passed": True,
            }
        )
        return HeadPatchResponses(
            normalized_transfer=normalized,
            raw_donor_minus_base_margin=raw_margin,
            raw_margin_shift_from_base=raw_shift,
            composite_reconstruction=reconstruction,
        )

    def responses(
        self,
        record: Mapping[str, Any],
        coarse_masks: torch.Tensor,
        attention_head_masks: torch.Tensor,
    ) -> tuple[HeadPatchResponses, dict[str, Any]]:
        """Fail-fast on masks, prepare once, and evaluate the requested rows."""
        self._validated_intervention_masks(coarse_masks, attention_head_masks)
        prepared = self.prepare_context(record)
        responses = self.responses_from_prepared(
            prepared, coarse_masks, attention_head_masks
        )
        return responses, dict(prepared.clean_metadata)

    def batch_invariance_smoke(
        self,
        prepared: PreparedHeadPatchContext,
        coarse_masks: torch.Tensor,
        attention_head_masks: torch.Tensor,
        *,
        reference_batch_size: int = 1,
        candidate_batch_size: int = 32,
    ) -> dict[str, Any]:
        """Compare the same head/coarse interventions under two batch sizes."""
        self._validated_intervention_masks(coarse_masks, attention_head_masks)
        for name, value in (
            ("reference_batch_size", reference_batch_size),
            ("candidate_batch_size", candidate_batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        reference = self.responses_from_prepared(
            prepared,
            coarse_masks,
            attention_head_masks,
            batch_size=reference_batch_size,
        )
        candidate = self.responses_from_prepared(
            prepared,
            coarse_masks,
            attention_head_masks,
            batch_size=candidate_batch_size,
        )
        atol = float(NUMERICAL_GATES["batch_size_smoke_atol"])
        rtol = float(NUMERICAL_GATES["batch_size_smoke_rtol"])
        coordinates = {}
        for name in (
            "normalized_transfer",
            "raw_donor_minus_base_margin",
            "raw_margin_shift_from_base",
        ):
            expected = getattr(reference, name)
            observed = getattr(candidate, name)
            differences = (observed - expected).abs()
            coordinates[name] = {
                "maximum_absolute_difference": float(differences.max()),
                "maximum_relative_difference": float(
                    (differences / expected.abs().clamp_min(atol)).max()
                ),
                "passed": bool(
                    torch.allclose(observed, expected, atol=atol, rtol=rtol)
                ),
            }
        composite_passed = bool(
            reference.composite_reconstruction["passed"]
            and candidate.composite_reconstruction["passed"]
        )
        return {
            "reference_batch_size": reference_batch_size,
            "candidate_batch_size": candidate_batch_size,
            "mask_rows": len(coarse_masks),
            "atol": atol,
            "rtol": rtol,
            "coordinates": coordinates,
            "composite_reconstruction": {
                "reference_maximum_absolute_error": reference.composite_reconstruction[
                    "maximum_absolute_error"
                ],
                "candidate_maximum_absolute_error": candidate.composite_reconstruction[
                    "maximum_absolute_error"
                ],
                "passed": composite_passed,
            },
            "passed": composite_passed
            and all(record["passed"] for record in coordinates.values()),
        }


__all__ = [
    "ATTENTION_HEAD_COUNT",
    "ATTENTION_HEAD_SHAPE",
    "COARSE_COMPONENT_COUNT",
    "COMPOSITE_RECONSTRUCTION_ATOL",
    "HEAD_COMPONENT_COUNT",
    "RESPONSE_COORDINATE",
    "WHOLE_ATTENTION_EQUIVALENCE_ATOL",
    "E312HeadPatchRunner",
    "HeadPatchResponses",
    "PreparedHeadPatchContext",
    "reconstruct_selected_head_composite",
    "split_projected_head_contributions",
]
