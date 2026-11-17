"""CPU-float64 GPT-2 sublayer patch runner with numerical integrity checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_protocol import (
    NUMERICAL_GATES,
)
from experiments.e29_gpt2_stable_head_interactions.generalized_sublayer_patch import (
    SublayerComponent,
    gpt2_sublayer_components,
)


@dataclass(frozen=True, eq=False)
class PreparedFloat64PatchContext:
    """Clean float64 state cached before patched interventions are evaluated."""

    base_input_ids: torch.Tensor = field(repr=False)
    donor_updates: tuple[torch.Tensor, ...] = field(repr=False)
    base_target_token_id: int
    donor_target_token_id: int
    clean_donor_minus_base_margin: float
    clean_metadata: Mapping[str, float | str | int | bool]
    _runner: Any = field(repr=False)


def _target_token_id(tokenizer: Any, target: str) -> int:
    token_ids = tokenizer.encode(" " + target, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"target is not one GPT-2 token: {target!r} -> {token_ids}")
    return int(token_ids[0])


class Float64GPT2SublayerPatchRunner:
    """Patch all GPT-2 attention/MLP residual updates using CPU float64.

    The semantics intentionally match ``GPT2SublayerPatchRunner``.  This class
    additionally enforces dtype/device, records clean-logit magnitudes, permits
    an explicit batch-size override for the preregistered invariance smoke, and
    rejects implausibly large responses before returning them.
    """

    def __init__(self, model: Any, tokenizer: Any, *, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if model.training:
            raise ValueError("dropout must be disabled with model.eval()")
        parameters = tuple(model.parameters())
        if not parameters:
            raise ValueError("model has no parameters")
        buffers = tuple(model.buffers())
        all_state = (*parameters, *buffers)
        floating_state = tuple(
            tensor for tensor in all_state if tensor.is_floating_point()
        )
        if any(tensor.device.type != "cpu" for tensor in all_state) or any(
            tensor.dtype != torch.float64 for tensor in floating_state
        ):
            raise ValueError("E29.10 requires a CPU model converted to float64")
        if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
            raise TypeError("model must expose the Hugging Face GPT-2 block layout")
        self.components = gpt2_sublayer_components(model.config)
        if len(model.transformer.h) != int(model.config.n_layer):
            raise ValueError("config.n_layer does not match transformer blocks")
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = parameters[0].device

    @property
    def component_count(self) -> int:
        return len(self.components)

    def _module(self, layer: int, kind: str) -> Any:
        block = self.model.transformer.h[layer]
        return block.attn.c_proj if kind == "attention" else block.mlp.c_proj

    def _input_ids(self, prompt: str) -> torch.Tensor:
        encoded: SimpleNamespace = self.tokenizer(prompt, return_tensors="pt")
        return encoded.input_ids.to(self.device)

    def _donor_updates_and_logits(
        self, donor_ids: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for component_index, (layer, kind) in enumerate(self.components):
            module = self._module(layer, kind)

            def capture(
                _module: Any,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                index: int = component_index,
            ) -> None:
                captured[index] = output[0].detach().clone()

            handles.append(module.register_forward_hook(capture))
        try:
            donor_logits = self.model(input_ids=donor_ids).logits[0, -1]
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != self.component_count:
            raise RuntimeError("failed to capture every donor sublayer update")
        return (
            tuple(captured[index] for index in range(self.component_count)),
            donor_logits.detach().clone(),
        )

    def _validate_masks(self, masks: torch.Tensor) -> None:
        expected = f"(rows, {self.component_count})"
        if masks.ndim != 2 or masks.shape[1] != self.component_count or len(masks) == 0:
            raise ValueError(f"masks must have shape {expected} with at least one row")
        if not bool(torch.logical_or(masks == 0, masks == 1).all()):
            raise ValueError("masks must be binary")

    @staticmethod
    def _check_clean_logits(*tables: torch.Tensor) -> None:
        limit = float(NUMERICAL_GATES["clean_logit_absolute_max"])
        for logits in tables:
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("non-finite clean float64 logit")
            if float(logits.abs().max()) > limit:
                raise RuntimeError(
                    "clean float64 logit exceeds preregistered magnitude gate"
                )

    def prepare_context(self, record: Mapping[str, Any]) -> PreparedFloat64PatchContext:
        """Run exactly two clean forwards and cache the donor updates."""
        base_prompt = str(record["base_prompt"])
        donor_prompt = str(record["donor_prompt"])
        base_target = str(record["base_target"])
        donor_target = str(record["donor_target"])
        base_token = _target_token_id(self.tokenizer, base_target)
        donor_token = _target_token_id(self.tokenizer, donor_target)
        base_ids = self._input_ids(base_prompt)
        donor_ids = self._input_ids(donor_prompt)
        if base_ids.shape[1] != donor_ids.shape[1]:
            raise ValueError(
                "all-position patches require equal base and donor token lengths"
            )

        with torch.no_grad():
            donor_updates, donor_logits = self._donor_updates_and_logits(donor_ids)
            base_logits = self.model(input_ids=base_ids).logits[0, -1].detach()
        self._check_clean_logits(base_logits, donor_logits)
        baseline_margin = float(base_logits[donor_token] - base_logits[base_token])
        metadata: dict[str, float | str | int | bool] = {
            "template": str(record["template"]),
            "base_prompt": base_prompt,
            "donor_prompt": donor_prompt,
            "base_target": base_target,
            "donor_target": donor_target,
            "base_target_token_id": base_token,
            "donor_target_token_id": donor_token,
            "clean_base_correct_minus_donor_logit": -baseline_margin,
            "clean_base_logit_absolute_max": float(base_logits.abs().max()),
            "clean_donor_logit_absolute_max": float(donor_logits.abs().max()),
            "clean_logits_finite": True,
            "model_layer_count": int(self.model.config.n_layer),
            "component_count": self.component_count,
            "patch_scope": "all_positions",
            "component_semantics": "attention_and_mlp_residual_updates",
            "device": "cpu",
            "model_dtype": "float64",
        }
        return PreparedFloat64PatchContext(
            base_input_ids=base_ids.detach().clone(),
            donor_updates=donor_updates,
            base_target_token_id=base_token,
            donor_target_token_id=donor_token,
            clean_donor_minus_base_margin=baseline_margin,
            clean_metadata=MappingProxyType(metadata),
            _runner=self,
        )

    def responses_from_prepared(
        self,
        prepared: PreparedFloat64PatchContext,
        masks: torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, float | str | int | bool]]:
        """Evaluate registered masks, optionally overriding batching for a smoke."""
        self._validate_masks(masks)
        if prepared._runner is not self:
            raise ValueError("prepared context belongs to a different patch runner")
        effective_batch_size = self.batch_size if batch_size is None else batch_size
        if effective_batch_size < 1:
            raise ValueError("batch_size must be positive")

        response_batches = []
        with torch.no_grad():
            for mask_batch in masks.split(effective_batch_size):
                mask_batch = mask_batch.to(device=self.device, dtype=torch.bool)
                repeated_ids = prepared.base_input_ids.expand(len(mask_batch), -1)
                handles = []
                for component_index, (layer, kind) in enumerate(self.components):
                    module = self._module(layer, kind)
                    selected = mask_batch[:, component_index]

                    def patch_sublayer(
                        _module: Any,
                        _inputs: tuple[torch.Tensor, ...],
                        output: torch.Tensor,
                        *,
                        index: int = component_index,
                        selected_rows: torch.Tensor = selected,
                    ) -> torch.Tensor:
                        return torch.where(
                            selected_rows[:, None, None],
                            prepared.donor_updates[index][None],
                            output,
                        )

                    handles.append(module.register_forward_hook(patch_sublayer))
                try:
                    logits = self.model(input_ids=repeated_ids).logits[:, -1]
                finally:
                    for handle in handles:
                        handle.remove()
                self._check_clean_logits(logits)
                margins = (
                    logits[:, prepared.donor_target_token_id]
                    - logits[:, prepared.base_target_token_id]
                )
                response_batches.append(
                    margins.detach().cpu() - prepared.clean_donor_minus_base_margin
                )
        responses = torch.cat(response_batches).to(dtype=torch.float64)
        limit = float(NUMERICAL_GATES["response_absolute_max"])
        if not bool(torch.isfinite(responses).all()):
            raise RuntimeError("non-finite float64 intervention response")
        if float(responses.abs().max()) > limit:
            raise RuntimeError("float64 response exceeds preregistered magnitude gate")
        return responses, dict(prepared.clean_metadata)

    def batch_invariance_smoke(
        self, prepared: PreparedFloat64PatchContext, masks: torch.Tensor
    ) -> dict[str, float | int | bool]:
        """Compare the exact same float64 interventions at batch sizes 1 and 32."""
        reference_batch = int(NUMERICAL_GATES["batch_size_smoke_reference"])
        candidate_batch = int(NUMERICAL_GATES["batch_size_smoke_candidate"])
        reference, _ = self.responses_from_prepared(
            prepared, masks, batch_size=reference_batch
        )
        candidate, _ = self.responses_from_prepared(
            prepared, masks, batch_size=candidate_batch
        )
        differences = (candidate - reference).abs()
        atol = float(NUMERICAL_GATES["batch_size_smoke_atol"])
        rtol = float(NUMERICAL_GATES["batch_size_smoke_rtol"])
        passed = bool(torch.allclose(candidate, reference, atol=atol, rtol=rtol))
        return {
            "reference_batch_size": reference_batch,
            "candidate_batch_size": candidate_batch,
            "mask_rows": len(masks),
            "maximum_absolute_difference": float(differences.max()),
            "maximum_relative_difference": float(
                (differences / reference.abs().clamp_min(atol)).max()
            ),
            "atol": atol,
            "rtol": rtol,
            "passed": passed,
        }


__all__ = [
    "Float64GPT2SublayerPatchRunner",
    "PreparedFloat64PatchContext",
    "SublayerComponent",
]
