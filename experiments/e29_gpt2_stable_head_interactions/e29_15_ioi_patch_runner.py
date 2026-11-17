# ruff: noqa: EXE002
"""CPU-float64 GPT-2-small sublayer patches for the E29.15 IOI study.

The implementation deliberately builds on the numerically audited E29.10
runner.  A binary variable controls each of GPT-2 small's 24 attention/MLP
residual updates.  Selected updates are replaced at every position by the
corresponding clean-donor update; later components are recomputed normally.

The primary response is the donor-answer minus base-answer logit contrast,
affinely normalized to zero at the clean base and one at the clean donor.
Values are not clipped, so overshoot and off-endpoint behavior remain visible.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_protocol import (
    NUMERICAL_GATES,
)
from experiments.e29_gpt2_stable_head_interactions.e29_10_float64_runner import (
    Float64GPT2SublayerPatchRunner,
    PreparedFloat64PatchContext,
)

COMPONENT_COUNT = 24
LAYER_COUNT = 12
MINIMUM_CLEAN_ENDPOINT = 0.25
RESPONSE_COORDINATE = (
    "(patched_donor_minus_base_margin-clean_base_donor_minus_base_margin)"
    "/clean_endpoint"
)
PROMPT_RECORD_KEYS = (
    "pair_id",
    "split",
    "stratum_id",
    "template_id",
    "template",
    "name_a",
    "name_b",
    "place",
    "object",
    "base_prompt",
    "donor_prompt",
    "base_target",
    "donor_target",
    "base_repeated_subject",
    "donor_repeated_subject",
)
_PROMPT_RECORD_KEY_SET = frozenset(PROMPT_RECORD_KEYS)


@dataclass(frozen=True)
class IOIPatchResponses:
    """Raw and normalized responses for one prompt pair and mask table."""

    normalized_transfer: torch.Tensor
    raw_donor_minus_base_margin: torch.Tensor
    raw_margin_shift_from_base: torch.Tensor


@dataclass(frozen=True, eq=False)
class PreparedIOIPatchContext:
    """Two clean IOI endpoints plus the donor sublayer-update cache."""

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
    _raw_context: PreparedFloat64PatchContext = field(repr=False)
    _runner: Any = field(repr=False)

    @property
    def donor_updates(self) -> tuple[torch.Tensor, ...]:
        """Expose the donor cache for integrity diagnostics."""
        return self._raw_context.donor_updates


def _target_token_id(tokenizer: Any, target: str) -> int:
    if not isinstance(target, str) or not target or target != target.strip():
        raise ValueError("each IOI target must be a nonempty unpadded string")
    token_ids = tokenizer.encode(" " + target, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"target is not one leading-space GPT-2 token: {target!r} -> {token_ids}"
        )
    return int(token_ids[0])


def _record_strings(record: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(record, Mapping):
        raise TypeError("an IOI prompt record must be a mapping")
    observed = set(record)
    if observed != _PROMPT_RECORD_KEY_SET:
        raise ValueError(
            "IOI prompt record keys changed; "
            f"missing={sorted(_PROMPT_RECORD_KEY_SET - observed)}, "
            f"extra={sorted(observed - _PROMPT_RECORD_KEY_SET)}"
        )
    strings = {key: str(record[key]) for key in PROMPT_RECORD_KEYS}
    if any(not value for value in strings.values()):
        raise ValueError("IOI prompt record strings must be nonempty")
    if strings["name_a"] == strings["name_b"]:
        raise ValueError("IOI name_a and name_b must differ")
    expected_roles = {
        "base_repeated_subject": strings["name_b"],
        "donor_repeated_subject": strings["name_a"],
        "base_target": strings["name_a"],
        "donor_target": strings["name_b"],
    }
    if any(strings[key] != value for key, value in expected_roles.items()):
        raise ValueError("IOI base/donor repeated-subject and target roles changed")
    return strings


class E2915IOIPatchRunner(Float64GPT2SublayerPatchRunner):
    """Run aligned donor-to-base IOI patches over exactly 24 sublayers."""

    def __init__(self, model: Any, tokenizer: Any, *, batch_size: int) -> None:
        super().__init__(model, tokenizer, batch_size=batch_size)
        if int(model.config.n_layer) != LAYER_COUNT or self.component_count != (
            COMPONENT_COUNT
        ):
            raise ValueError("E29.15 requires GPT-2 small: 12 layers and 24 sublayers")
        attention_implementation = getattr(model.config, "_attn_implementation", None)
        if attention_implementation != "eager":
            raise ValueError("E29.15 requires eager attention")
        self.attention_implementation = attention_implementation

    def prepare_context(self, record: Mapping[str, Any]) -> PreparedIOIPatchContext:
        """Validate one IOI pair and run its two clean forwards exactly once."""
        strings = _record_strings(record)
        base_token = _target_token_id(self.tokenizer, strings["base_target"])
        donor_token = _target_token_id(self.tokenizer, strings["donor_target"])
        if base_token == donor_token:
            raise ValueError("base and donor answers must be distinct target tokens")

        base_ids = self._input_ids(strings["base_prompt"])
        donor_ids = self._input_ids(strings["donor_prompt"])
        if (
            base_ids.ndim != 2
            or base_ids.shape[0] != 1
            or base_ids.shape != donor_ids.shape
        ):
            raise ValueError("base and donor prompts must be one aligned token row")
        differing_positions = torch.where(base_ids[0] != donor_ids[0])[0]
        if len(differing_positions) != 1:
            raise ValueError("canonical IOI pairs must differ at exactly one token")

        with torch.no_grad():
            donor_updates, donor_logits = self._donor_updates_and_logits(donor_ids)
            base_logits = self.model(input_ids=base_ids).logits[0, -1].detach().clone()
        self._check_clean_logits(base_logits, donor_logits)

        base_margin = float(base_logits[donor_token] - base_logits[base_token])
        donor_margin = float(donor_logits[donor_token] - donor_logits[base_token])
        endpoint = donor_margin - base_margin
        if not math.isfinite(endpoint) or endpoint < MINIMUM_CLEAN_ENDPOINT:
            raise RuntimeError(
                "clean normalized IOI endpoint is below the registered 0.25 floor"
            )
        base_correct = base_margin < 0.0
        donor_correct = donor_margin > 0.0

        clean_logit_max = max(
            float(base_logits.abs().max()), float(donor_logits.abs().max())
        )
        metadata: dict[str, Any] = {
            **strings,
            "base_target_token_id": base_token,
            "donor_target_token_id": donor_token,
            "token_length": int(base_ids.shape[1]),
            "differing_token_position": int(differing_positions[0]),
            "answer_position": -1,
            "clean_base_donor_minus_base_margin": base_margin,
            "clean_donor_donor_minus_base_margin": donor_margin,
            "clean_base_correct_minus_donor_margin": -base_margin,
            "clean_donor_correct_minus_base_margin": donor_margin,
            "clean_base_correct": base_correct,
            "clean_donor_correct": donor_correct,
            "clean_endpoint": endpoint,
            "minimum_clean_endpoint": MINIMUM_CLEAN_ENDPOINT,
            "clean_base_logit_absolute_max": float(base_logits.abs().max()),
            "clean_donor_logit_absolute_max": float(donor_logits.abs().max()),
            "clean_logit_absolute_max": clean_logit_max,
            "clean_logits_finite": True,
            "model_name_or_path": str(getattr(self.model.config, "_name_or_path", "")),
            "model_type": str(getattr(self.model.config, "model_type", "gpt2")),
            "model_layer_count": LAYER_COUNT,
            "component_count": COMPONENT_COUNT,
            "component_semantics": "attention_and_mlp_residual_updates",
            "patch_direction": "clean_donor_updates_into_base_recipient",
            "patch_scope": "all_positions",
            "response_coordinate": RESPONSE_COORDINATE,
            "attention_implementation": self.attention_implementation,
            "device": "cpu",
            "model_dtype": "float64",
        }
        immutable_metadata = MappingProxyType(metadata)
        raw_context = PreparedFloat64PatchContext(
            base_input_ids=base_ids.detach().clone(),
            donor_updates=tuple(update.clone() for update in donor_updates),
            base_target_token_id=base_token,
            donor_target_token_id=donor_token,
            clean_donor_minus_base_margin=base_margin,
            clean_metadata=immutable_metadata,
            _runner=self,
        )
        return PreparedIOIPatchContext(
            base_input_ids=base_ids.detach().clone(),
            donor_input_ids=donor_ids.detach().clone(),
            base_logits=base_logits.detach().clone(),
            donor_logits=donor_logits.detach().clone(),
            base_target_token_id=base_token,
            donor_target_token_id=donor_token,
            clean_base_donor_minus_base_margin=base_margin,
            clean_donor_donor_minus_base_margin=donor_margin,
            clean_base_correct=base_correct,
            clean_donor_correct=donor_correct,
            clean_endpoint=endpoint,
            clean_metadata=immutable_metadata,
            _raw_context=raw_context,
            _runner=self,
        )

    def responses_from_prepared(
        self,
        prepared: PreparedIOIPatchContext,
        masks: torch.Tensor,
        *,
        batch_size: int | None = None,
    ) -> IOIPatchResponses:
        """Return normalized transfer and both auditable raw coordinates."""
        self._validate_masks(masks)
        if not isinstance(prepared, PreparedIOIPatchContext) or (
            prepared._runner is not self
        ):
            raise ValueError("prepared context belongs to a different IOI runner")
        raw_shift, _metadata = super().responses_from_prepared(
            prepared._raw_context,
            masks,
            batch_size=batch_size,
        )
        raw_margin = raw_shift + prepared.clean_base_donor_minus_base_margin
        normalized = raw_shift / prepared.clean_endpoint
        if not bool(normalized.isfinite().all()):
            raise RuntimeError("a normalized IOI intervention response is non-finite")
        return IOIPatchResponses(
            normalized_transfer=normalized,
            raw_donor_minus_base_margin=raw_margin,
            raw_margin_shift_from_base=raw_shift,
        )

    def responses(
        self, record: Mapping[str, Any], masks: torch.Tensor
    ) -> tuple[IOIPatchResponses, dict[str, Any]]:
        """Prepare a prompt pair and evaluate one registered mask table."""
        self._validate_masks(masks)
        prepared = self.prepare_context(record)
        return self.responses_from_prepared(prepared, masks), dict(
            prepared.clean_metadata
        )

    def batch_invariance_smoke(
        self,
        prepared: PreparedIOIPatchContext,
        masks: torch.Tensor,
        *,
        reference_batch_size: int = 1,
        candidate_batch_size: int = 32,
    ) -> dict[str, Any]:
        """Compare raw and normalized coordinates under two batch sizes."""
        self._validate_masks(masks)
        for name, value in (
            ("reference_batch_size", reference_batch_size),
            ("candidate_batch_size", candidate_batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        reference = self.responses_from_prepared(
            prepared, masks, batch_size=reference_batch_size
        )
        candidate = self.responses_from_prepared(
            prepared, masks, batch_size=candidate_batch_size
        )
        atol = float(NUMERICAL_GATES["batch_size_smoke_atol"])
        rtol = float(NUMERICAL_GATES["batch_size_smoke_rtol"])
        coordinates: dict[str, dict[str, float | bool]] = {}
        for name in (
            "normalized_transfer",
            "raw_donor_minus_base_margin",
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
        return {
            "reference_batch_size": reference_batch_size,
            "candidate_batch_size": candidate_batch_size,
            "mask_rows": len(masks),
            "atol": atol,
            "rtol": rtol,
            "coordinates": coordinates,
            "passed": all(record["passed"] for record in coordinates.values()),
        }


__all__ = [
    "COMPONENT_COUNT",
    "LAYER_COUNT",
    "MINIMUM_CLEAN_ENDPOINT",
    "PROMPT_RECORD_KEYS",
    "RESPONSE_COORDINATE",
    "E2915IOIPatchRunner",
    "IOIPatchResponses",
    "PreparedIOIPatchContext",
]
