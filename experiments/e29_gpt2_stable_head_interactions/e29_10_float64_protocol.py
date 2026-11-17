"""Frozen design constants for the E29.10 float64 forensic replication.

E29.10 changes only transformer inference precision.  It reuses the exact
E29.9 prompt panel, masks, feature artifact, fitting recipes, method manifest,
and scientific gates.  The old float32 labels are deliberately absent from
all acquisition and fitting APIs; only the final scorer may open them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from experiments.e29_gpt2_stable_head_interactions.e29_9_seven_gate_protocol import (
    METHOD_MANIFEST,
    PRIMARY_METHOD,
    PRIMARY_SIZE16_GATES,
    QUERY_BUDGET,
    READOUT_PROTOCOL,
    STRESS_SIZE24_GATES,
)

E29_9_METHODS = METHOD_MANIFEST
PRIMARY_BUDGETS = tuple(READOUT_PROTOCOL["nested_calibration_sizes"])

MODEL_DTYPE = "float64"
DEVICE = "cpu"
ATTENTION_IMPLEMENTATION = "eager"
BATCH_SIZE = 32
COMPONENT_COUNT = 48
CONTEXT_COUNT = 12
CHECKPOINT_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
)
SOURCE_FILES = (
    "e29_10_float64_protocol.py",
    "e29_10_float64_runner.py",
    "e29_10_custody.py",
    "register_e29_10_float64.py",
    "acquire_e29_10_calibration_validation.py",
    "fit_e29_10_transfer_models.py",
    "acquire_e29_10_final_stress.py",
    "score_e29_10_float64.py",
    "e29_9_custody.py",
    "e29_9_seven_gate_protocol.py",
    "e29_9_transfer_fitting.py",
    "source_model_fitting.py",
    "scaled_tensor_response.py",
    "tensor_train_response.py",
    "e29_8_scaleup_protocol.py",
    "generalized_sublayer_patch.py",
)

MASK_SPLITS = (
    "calibration",
    "validation_size8",
    "final_size16",
    "stress_size24",
)
MASK_SHAPES = {
    "calibration": (96, 48),
    "validation_size8": (32, 48),
    "final_size16": (64, 48),
    "stress_size24": (32, 48),
}
RESPONSE_SHAPES = {
    "calibration": (96, 12),
    "validation_size8": (32, 12),
    "final_size16": (64, 12),
    "stress_size24": (32, 12),
}

# Conservative sanity limits, intentionally orders of magnitude above normal
# GPT-2 logits and causal-response margins.  These reject the 1e35 corruption
# observed in E29.9 without turning ordinary scientific variation into a gate.
NUMERICAL_GATES = {
    "clean_logits_finite": True,
    "clean_logit_absolute_max": 10_000.0,
    "response_finite": True,
    "response_absolute_max": 10_000.0,
    "batch_size_smoke_reference": 1,
    "batch_size_smoke_candidate": 32,
    "batch_size_smoke_atol": 1e-9,
    "batch_size_smoke_rtol": 1e-9,
}

FORENSIC_REFERENCE_ACCESS_POLICY = (
    "score_only_after_all_float64_labels_and_predictions_frozen"
)
FORENSIC_REFERENCE_MANIFEST_SHA256 = (
    "fc74d36236231123fe535a0276bbba91a966a1f91fd0c7b28063ff876862dfd6"
)


def validate_reused_design(
    *,
    method_manifest: Sequence[str],
    query_budget: Mapping[str, Any],
    primary_gates: Mapping[str, Any],
    stress_gates: Mapping[str, Any],
) -> None:
    """Fail if any scientific element silently differs from sealed E29.9."""
    if tuple(METHOD_MANIFEST) != tuple(E29_9_METHODS) or tuple(
        method_manifest
    ) != tuple(E29_9_METHODS):
        raise RuntimeError("E29.10 must preserve the exact 25-method E29.9 manifest")
    if dict(query_budget) != dict(QUERY_BUDGET):
        raise RuntimeError("E29.10 must preserve the exact E29.9 query accounting")
    if dict(primary_gates) != dict(PRIMARY_SIZE16_GATES):
        raise RuntimeError("E29.10 changed the pre-existing size-16 gates")
    if dict(stress_gates) != dict(STRESS_SIZE24_GATES):
        raise RuntimeError("E29.10 changed the pre-existing stress gates")


def tensor_numerical_summary(value: torch.Tensor) -> dict[str, Any]:
    """Return auditable finite/magnitude diagnostics without changing a tensor."""
    finite = bool(torch.isfinite(value).all())
    absolute_max = float(value.abs().max()) if value.numel() else 0.0
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "finite": finite,
        "absolute_max": absolute_max,
        "passed": (
            finite and absolute_max <= float(NUMERICAL_GATES["response_absolute_max"])
        ),
    }


def clean_metadata_passes(metadata: Sequence[Mapping[str, Any]]) -> bool:
    """Apply the preregistered finite and magnitude gate to all clean logits."""
    if len(metadata) != CONTEXT_COUNT:
        return False
    limit = float(NUMERICAL_GATES["clean_logit_absolute_max"])
    required = (
        "clean_base_logit_absolute_max",
        "clean_donor_logit_absolute_max",
        "clean_base_correct_minus_donor_logit",
    )
    for row in metadata:
        values = torch.tensor(
            [float(row[key]) for key in required], dtype=torch.float64
        )
        if not bool(torch.isfinite(values).all()) or float(values[:2].max()) > limit:
            return False
    return True


validate_reused_design(
    method_manifest=E29_9_METHODS,
    query_budget=QUERY_BUDGET,
    primary_gates=PRIMARY_SIZE16_GATES,
    stress_gates=STRESS_SIZE24_GATES,
)


__all__ = [
    "ATTENTION_IMPLEMENTATION",
    "BATCH_SIZE",
    "CHECKPOINT_FILES",
    "COMPONENT_COUNT",
    "CONTEXT_COUNT",
    "DEVICE",
    "FORENSIC_REFERENCE_ACCESS_POLICY",
    "FORENSIC_REFERENCE_MANIFEST_SHA256",
    "MASK_SHAPES",
    "MASK_SPLITS",
    "MODEL_DTYPE",
    "NUMERICAL_GATES",
    "PRIMARY_BUDGETS",
    "PRIMARY_METHOD",
    "PRIMARY_SIZE16_GATES",
    "QUERY_BUDGET",
    "RESPONSE_SHAPES",
    "SOURCE_FILES",
    "STRESS_SIZE24_GATES",
    "clean_metadata_passes",
    "tensor_numerical_summary",
    "validate_reused_design",
]
