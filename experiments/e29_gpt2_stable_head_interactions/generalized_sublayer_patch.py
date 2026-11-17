"""Model-size-independent all-position sublayer patches for GPT-2 models.

Each component is one transformer block's attention or MLP residual update.
For a selected component, the runner replaces that update at every token
position with the update produced by a clean donor prompt.  The component map
therefore has ``2 * config.n_layer`` entries and works unchanged for GPT-2
small, medium, and other models with the Hugging Face GPT-2 block layout.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import Any

import torch

SublayerComponent = tuple[int, str]
PromptRecord = Mapping[str, Any]


@dataclass(frozen=True, eq=False)
class PreparedSublayerPatchContext:
    """Clean context cached before any patched intervention is evaluated."""

    base_input_ids: torch.Tensor = field(repr=False)
    donor_updates: tuple[torch.Tensor, ...] = field(repr=False)
    base_target_token_id: int
    donor_target_token_id: int
    clean_donor_minus_base_margin: float
    clean_metadata: Mapping[str, float | str | int]
    _runner: Any = field(repr=False)

    @property
    def clean_base_correct_minus_donor_logit(self) -> float:
        """Return the clean task margin used by the E29 promotion gate."""
        return -self.clean_donor_minus_base_margin


def gpt2_sublayer_components(config: Any) -> tuple[SublayerComponent, ...]:
    """Return attention then MLP for every layer declared by ``config``."""
    layer_count = int(config.n_layer)
    if layer_count < 1:
        raise ValueError("GPT-2 must have at least one transformer layer")
    return tuple(
        (layer, kind)
        for layer in range(layer_count)
        for kind in ("attention", "mlp")
    )


def _format_prompt(record: PromptRecord, side: str) -> str:
    direct_key = f"{side}_prompt"
    if direct_key in record:
        return str(record[direct_key])
    return str(record["template"]).format(country=record[side]["country"])


def _target(record: PromptRecord, side: str) -> str:
    direct_key = f"{side}_target"
    if direct_key in record:
        return str(record[direct_key])
    return str(record[side]["capital"])


def _target_token_id(tokenizer: Any, target: str) -> int:
    token_ids = tokenizer.encode(" " + target, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"target is not one GPT-2 token: {target!r} -> {token_ids}")
    return int(token_ids[0])


class GPT2SublayerPatchRunner:
    """Run coherent donor-update patches over every GPT-2 sublayer.

    The intervention matches the sealed E29 GPT-2-small ``sublayers`` mode:
    hooks replace ``attn.c_proj`` or ``mlp.c_proj`` outputs before the block's
    residual addition.  Unlike that runner, no layer/head constants are baked
    in.  All patches cover all token positions by construction.
    """

    def __init__(self, model: Any, tokenizer: Any, *, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if model.training:
            raise ValueError("dropout must be disabled with model.eval()")
        if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
            raise TypeError("model must expose the Hugging Face GPT-2 block layout")
        self.components = gpt2_sublayer_components(model.config)
        if len(model.transformer.h) != int(model.config.n_layer):
            raise ValueError("config.n_layer does not match the transformer blocks")
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = next(model.parameters()).device

    @property
    def component_count(self) -> int:
        return len(self.components)

    def _module(self, layer: int, kind: str) -> Any:
        block = self.model.transformer.h[layer]
        return block.attn.c_proj if kind == "attention" else block.mlp.c_proj

    def _input_ids(self, prompt: str) -> torch.Tensor:
        encoded: SimpleNamespace = self.tokenizer(prompt, return_tensors="pt")
        return encoded.input_ids.to(self.device)

    def _clean_donor_updates(
        self, donor_ids: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
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
            self.model(input_ids=donor_ids)
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != self.component_count:
            raise RuntimeError("failed to capture every donor sublayer update")
        return tuple(captured[index] for index in range(self.component_count))

    def _validate_masks(self, masks: torch.Tensor) -> None:
        expected = f"(rows, {self.component_count})"
        if masks.ndim != 2 or masks.shape[1] != self.component_count or len(masks) == 0:
            raise ValueError(f"masks must have shape {expected} with at least one row")
        if not bool(torch.logical_or(masks == 0, masks == 1).all()):
            raise ValueError("masks must be binary")

    def prepare_context(self, record: PromptRecord) -> PreparedSublayerPatchContext:
        """Cache donor updates and clean-base metadata without applying patches."""
        base_prompt = _format_prompt(record, "base")
        donor_prompt = _format_prompt(record, "donor")
        base_target = _target(record, "base")
        donor_target = _target(record, "donor")
        base_token = _target_token_id(self.tokenizer, base_target)
        donor_token = _target_token_id(self.tokenizer, donor_target)
        base_ids = self._input_ids(base_prompt)
        donor_ids = self._input_ids(donor_prompt)
        if base_ids.shape[1] != donor_ids.shape[1]:
            raise ValueError(
                "all-position patches require equal base and donor token lengths"
            )

        with torch.no_grad():
            donor_updates = self._clean_donor_updates(donor_ids)
            baseline_logits = self.model(input_ids=base_ids).logits[0, -1]
            baseline_margin_tensor = (
                baseline_logits[donor_token] - baseline_logits[base_token]
            )
            baseline_margin = float(baseline_margin_tensor)

        metadata: dict[str, float | str | int] = {
            "template": str(record["template"]),
            "base_prompt": base_prompt,
            "donor_prompt": donor_prompt,
            "base_target": base_target,
            "donor_target": donor_target,
            "base_target_token_id": base_token,
            "donor_target_token_id": donor_token,
            "clean_base_correct_minus_donor_logit": -baseline_margin,
            "model_layer_count": int(self.model.config.n_layer),
            "component_count": self.component_count,
            "patch_scope": "all_positions",
            "component_semantics": "attention_and_mlp_residual_updates",
        }
        return PreparedSublayerPatchContext(
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
        prepared: PreparedSublayerPatchContext,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float | str | int]]:
        """Evaluate intervention masks without repeating either clean forward."""
        self._validate_masks(masks)
        if prepared._runner is not self:
            raise ValueError("prepared context belongs to a different patch runner")

        with torch.no_grad():
            response_batches = []
            for mask_batch in masks.split(self.batch_size):
                mask_batch = mask_batch.to(device=self.device, dtype=torch.bool)
                batch_rows = len(mask_batch)
                repeated_ids = prepared.base_input_ids.expand(batch_rows, -1)
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
                margins = (
                    logits[:, prepared.donor_target_token_id]
                    - logits[:, prepared.base_target_token_id]
                )
                response_batches.append(
                    margins.detach().to(dtype=torch.float64).cpu()
                    - prepared.clean_donor_minus_base_margin
                )

        responses = torch.cat(response_batches)
        if not bool(torch.isfinite(responses).all()):
            raise RuntimeError("non-finite intervention response")
        return responses, dict(prepared.clean_metadata)

    def responses(
        self, record: PromptRecord, masks: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float | str | int]]:
        """Prepare a context, then return responses for every intervention mask."""
        # Preserve the original fail-fast behavior for invalid masks: no clean
        # model work should occur when the intervention request is malformed.
        self._validate_masks(masks)
        return self.responses_from_prepared(self.prepare_context(record), masks)
