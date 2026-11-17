"""Response-blind prompt, mask, and decision primitives for E29.9.

This module deliberately contains no model calls and no outcome-dependent
logic.  Registration code may serialize its deterministic outputs later, after
verifying the sealed E29.8 mask artifact supplied to ``build_e29_9_mask_protocol``.
"""

from __future__ import annotations

import itertools
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch

COMPONENT_COUNT = 48
CONTEXT_COUNT = 12
CALIBRATION_MASK_SEED = 29_901
VALIDATION_MASK_SEED = 29_902
FINAL_MASK_SEED = 29_903
STRESS_MASK_SEED = 29_904
BOOTSTRAP_SEED = 29_905
BOOTSTRAP_REPLICATES = 2_000

SOURCE_MLP_TRAINING_SEEDS = (29_811, 29_812, 29_813, 29_814, 29_815)
SELECTED_SOURCE_MLP_SEED = 29_814
SCRATCH_TRAINING_SEEDS = tuple(range(29_911, 29_916))
RANDOM_FEATURE_SEEDS = tuple(range(29_921, 29_926))
COMPONENT_PERMUTATION_SEEDS = tuple(range(29_931, 29_941))

RIDGE_VALUES = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1_000.0)
PRIMARY_METHOD = "mlp7_frozen_primary"
SECONDARY_METHOD = "ensemble_80_mlp7_20_tt2"
METHOD_MANIFEST = (
    PRIMARY_METHOD,
    "tt2_frozen",
    "cp4_frozen",
    "additive",
    "pairwise",
    "cardinality_only",
    "mlp7_scratch",
    *(f"mlp7_random_seed{seed}" for seed in RANDOM_FEATURE_SEEDS),
    *(f"mlp7_permuted_seed{seed}" for seed in COMPONENT_PERMUTATION_SEEDS),
    "mlp7_permuted_reverse",
    "mlp7_permuted_attn_mlp_swap",
    SECONDARY_METHOD,
)
CONTROL_METHODS = METHOD_MANIFEST[1:-1]
RANDOM_FEATURE_METHODS = tuple(
    f"mlp7_random_seed{seed}" for seed in RANDOM_FEATURE_SEEDS
)
LEARNED_PERMUTATION_METHODS = (
    *(f"mlp7_permuted_seed{seed}" for seed in COMPONENT_PERMUTATION_SEEDS),
    "mlp7_permuted_reverse",
    "mlp7_permuted_attn_mlp_swap",
)
SECONDARY_ENSEMBLE_WEIGHTS = {PRIMARY_METHOD: 0.8, "tt2_frozen": 0.2}

PRIMARY_NEW_PARAMETER_COUNT = CONTEXT_COUNT * (7 + 1)
SCRATCH_MLP7_PARAMETER_COUNT = 439
SECONDARY_OPERATIONAL_PARAMETER_COUNT = 855

FEATURE_PREFLIGHT = {
    "center_and_standardize_from_calibration_only": True,
    "required_design_rank": 7,
    "condition_number_max": 1_000.0,
}
READOUT_PROTOCOL = {
    "standardization": "per_feature_and_context_from_calibration_only",
    "ridge_selection": "one_shared_value_on_validation_joint_mse",
    "refit_after_validation": False,
    "nested_calibration_sizes": (32, 64, 96),
    "primary_gate_calibration_size": 96,
}

PRIMARY_SIZE16_GATES = {
    "pooled_context_centered_r2_min": 0.75,
    "median_context_r2_min": 0.70,
    "context_r2_threshold": 0.40,
    "contexts_at_or_above_threshold_min": 10,
    "minimum_context_r2_min": 0.0,
    "mse_over_additive_mse_max": 0.60,
    "mse_over_affine_tt2_mse_max": 0.90,
    "mse_over_scratch_mlp7_mse_max": 1.10,
    "mse_over_median_permutation_control_mse_max": 0.80,
    "mse_over_median_random_feature_control_mse_max": 0.80,
    "positive_clean_margins_required": 12,
}
STRESS_SIZE24_GATES = {
    "pooled_context_centered_r2_min": 0.55,
    "median_context_r2_min": 0.50,
    "positive_context_r2_min_count": 10,
}
QUERY_BUDGET = {
    "contexts": 12,
    "fit_masks": 128,
    "fit_patched_queries": 1_536,
    "fit_clean_queries": 24,
    "fit_logical_queries": 1_560,
    "heldout_masks": 96,
    "heldout_patched_queries": 1_152,
    "heldout_clean_queries": 24,
    "heldout_logical_queries": 1_176,
    "total_patched_queries": 2_688,
    "total_clean_queries": 48,
    "total_logical_queries": 2_736,
}

_REQUIRED_E29_8_SPLITS = {
    "source": frozenset(
        {
            "train_low",
            "train_size8",
            "train_size16",
            "validation_size8",
            "validation_size16",
            "test_size8",
            "test_size16",
        }
    ),
    "transfer": frozenset({"calibration", "validation_size8", "test_size16"}),
}


def _induction_record(
    template: str,
    base_words: tuple[str, str, str, str],
    donor_words: tuple[str, str, str, str],
) -> dict[str, Any]:
    def render(words: tuple[str, str, str, str]) -> str:
        return template.format(a=words[0], b=words[1], c=words[2], d=words[3])

    return {
        "template": template,
        "base_words": base_words,
        "donor_words": donor_words,
        "base_prompt": render(base_words),
        "donor_prompt": render(donor_words),
        "base_target": base_words[3],
        "donor_target": donor_words[3],
    }


def e29_9_transfer_prompt_panel() -> tuple[dict[str, Any], ...]:
    """Return twelve prospective contexts with three balanced templates."""
    pairs = (
        (
            ("circle", "square", "triangle", "line"),
            ("piano", "guitar", "drum", "violin"),
        ),
        (("salt", "sugar", "pepper", "spice"), ("uncle", "aunt", "brother", "sister")),
        (("rabbit", "monkey", "snake", "whale"), ("beer", "wine", "vodka", "rum")),
        (("brain", "heart", "blood", "bone"), ("hammer", "nail", "screw", "rope")),
        (("market", "shop", "bank", "school"), ("actor", "writer", "singer", "player")),
        (("peace", "war", "truth", "lie"), ("breakfast", "lunch", "dinner", "snack")),
        (
            ("mountain", "valley", "desert", "island"),
            ("engine", "wheel", "brake", "seat"),
        ),
        (("data", "code", "file", "folder"), ("metal", "plastic", "rubber", "cotton")),
        (("baby", "child", "adult", "parent"), ("health", "money", "power", "time")),
        (("quiet", "loud", "soft", "hard"), ("push", "pull", "lift", "drop")),
        (("buy", "sell", "give", "take"), ("here", "there", "inside", "outside")),
        (("true", "false", "yes", "no"), ("first", "second", "third", "last")),
    )
    templates = (
        "Replay the sequence: {a} {b} {c} {d} | {a} {b} {c}",
        "Loop example: {a} {b} {c} {d}; continue {a} {b} {c}",
        "After {a} {b} {c} {d}, repeat {a} {b} {c}",
    )
    return tuple(
        _induction_record(templates[index // 4], *pair)
        for index, pair in enumerate(pairs)
    )


def validate_e29_9_prompt_panel(
    tokenizer: Any,
    records: tuple[dict[str, Any], ...],
    *,
    prior_group_words: Collection[str],
) -> tuple[dict[str, Any], ...]:
    """Validate lexical novelty and pinned-tokenizer alignment without outcomes."""
    if len(records) != CONTEXT_COUNT:
        raise ValueError("E29.9 requires exactly twelve prompt contexts")
    template_counts = Counter(str(record["template"]) for record in records)
    if len(template_counts) != 3 or set(template_counts.values()) != {4}:
        raise ValueError("E29.9 requires three templates used four times each")

    groups = [
        tuple(str(word) for word in record[key])
        for record in records
        for key in ("base_words", "donor_words")
    ]
    words = {word for group in groups for word in group}
    if any(len(group) != 4 for group in groups):
        raise ValueError("every E29.9 prompt group must contain four words")
    if len(groups) != 24 or len(set(groups)) != 24 or len(words) != 96:
        raise ValueError("E29.9 requires 24 distinct groups and 96 distinct words")
    overlap = sorted(words & {str(word) for word in prior_group_words})
    if overlap:
        raise ValueError(f"E29.9 words overlap prior E29 panels: {overlap}")

    checks: list[dict[str, Any]] = []
    for context_index, record in enumerate(records):
        prompt_ids = {
            side: tokenizer.encode(record[f"{side}_prompt"], add_special_tokens=False)
            for side in ("base", "donor")
        }
        if len(prompt_ids["base"]) != len(prompt_ids["donor"]):
            raise ValueError(
                f"context {context_index} base and donor token lengths differ"
            )

        side_checks: dict[str, dict[str, Any]] = {}
        for side in ("base", "donor"):
            side_words = tuple(str(word) for word in record[f"{side}_words"])
            word_token_lengths = {
                word: len(tokenizer.encode(" " + word, add_special_tokens=False))
                for word in side_words
            }
            if set(word_token_lengths.values()) != {1}:
                raise ValueError(
                    f"context {context_index} {side} contains a split group word"
                )
            target = str(record[f"{side}_target"])
            if target != side_words[3]:
                raise ValueError(f"context {context_index} {side} target is not word d")
            target_token_id = tokenizer.encode(" " + target, add_special_tokens=False)[
                0
            ]
            target_positions = [
                position
                for position, token_id in enumerate(prompt_ids[side])
                if token_id == target_token_id
            ]
            if len(target_positions) != 1:
                raise ValueError(
                    f"context {context_index} {side} target occurrence is ambiguous"
                )
            extended_ids = tokenizer.encode(
                record[f"{side}_prompt"] + " " + target,
                add_special_tokens=False,
            )
            append_valid = extended_ids == [*prompt_ids[side], target_token_id]
            if not append_valid:
                raise ValueError(
                    f"context {context_index} {side} target is not an exact append"
                )
            side_checks[side] = {
                "target": target,
                "target_token_id": int(target_token_id),
                "demonstration_target_position": target_positions[0],
                "prediction_position": len(prompt_ids[side]),
                "next_token_append_valid": append_valid,
                "word_token_lengths": word_token_lengths,
            }

        if (
            side_checks["base"]["demonstration_target_position"]
            != side_checks["donor"]["demonstration_target_position"]
        ):
            raise ValueError(
                f"context {context_index} target positions are not aligned"
            )
        if (
            side_checks["base"]["target_token_id"]
            == side_checks["donor"]["target_token_id"]
        ):
            raise ValueError(f"context {context_index} base and donor targets coincide")
        checks.append(
            {
                "context_index": context_index,
                "token_length": len(prompt_ids["base"]),
                "target_positions_aligned": True,
                "sides": side_checks,
                **deepcopy(record),
            }
        )
    return tuple(checks)


def _mask_rows(masks: torch.Tensor) -> list[tuple[int, ...]]:
    return [
        tuple(torch.nonzero(row, as_tuple=False).flatten().tolist()) for row in masks
    ]


def _validate_and_collect_e29_8_rows(
    sealed_e29_8_masks: Mapping[str, Any],
) -> dict[int, set[tuple[int, ...]]]:
    if not {"source", "transfer"} <= set(sealed_e29_8_masks):
        raise ValueError("sealed E29.8 masks must contain source and transfer families")

    rows_by_cardinality: defaultdict[int, set[tuple[int, ...]]] = defaultdict(set)
    missing_by_family: dict[str, set[str]] = {}
    for family in ("source", "transfer"):
        splits = sealed_e29_8_masks[family]
        if not isinstance(splits, Mapping):
            raise TypeError(f"sealed E29.8 {family} masks are not a mapping")
        for split_name, masks in splits.items():
            if not isinstance(masks, torch.Tensor):
                raise TypeError(f"sealed E29.8 {family}/{split_name} is not a tensor")
            if (
                masks.ndim != 2
                or masks.shape[1] != COMPONENT_COUNT
                or masks.shape[0] == 0
            ):
                raise ValueError(
                    f"sealed E29.8 {family}/{split_name} has an invalid shape"
                )
            if not torch.all((masks == 0) | (masks == 1)):
                raise ValueError(f"sealed E29.8 {family}/{split_name} is not binary")
            for row in _mask_rows(masks):
                if not row:
                    raise ValueError(
                        f"sealed E29.8 {family}/{split_name} has an empty row"
                    )
                rows_by_cardinality[len(row)].add(row)
        missing_by_family[family] = set(_REQUIRED_E29_8_SPLITS[family] - set(splits))
    for family, missing in missing_by_family.items():
        if missing:
            raise ValueError(
                f"sealed E29.8 {family} masks omit splits: {sorted(missing)}"
            )
    return dict(rows_by_cardinality)


def _masks_from_rows(rows: Sequence[Sequence[int]]) -> torch.Tensor:
    masks = torch.zeros(len(rows), COMPONENT_COUNT, dtype=torch.float64)
    for row_index, row in enumerate(rows):
        masks[row_index, list(row)] = 1.0
    return masks


def _balanced_rows(
    *,
    cardinality: int,
    count: int,
    coverage: list[int],
    excluded: set[tuple[int, ...]],
    generator: random.Random,
) -> list[tuple[int, ...]]:
    candidates = [
        row
        for row in itertools.combinations(range(COMPONENT_COUNT), cardinality)
        if row not in excluded
    ]
    if len(candidates) < count:
        raise ValueError(f"too few fresh cardinality-{cardinality} masks")
    # Shuffle once so the protocol seed is the final tie-breaker after the
    # deterministic incidence-balancing score.
    generator.shuffle(candidates)
    selected: list[tuple[int, ...]] = []
    for _ in range(count):
        best_index = min(
            range(len(candidates)),
            key=lambda index: (
                max(coverage[component] for component in candidates[index]),
                sum(coverage[component] for component in candidates[index]),
            ),
        )
        row = candidates.pop(best_index)
        selected.append(row)
        for component in row:
            coverage[component] += 1
    return selected


def _sample_fresh_rows(
    *,
    cardinality: int,
    count: int,
    excluded: set[tuple[int, ...]],
    generator: random.Random,
) -> list[tuple[int, ...]]:
    selected: list[tuple[int, ...]] = []
    seen = set(excluded)
    while len(selected) < count:
        row = tuple(sorted(generator.sample(range(COMPONENT_COUNT), cardinality)))
        if row in seen:
            continue
        seen.add(row)
        selected.append(row)
    return selected


def _validate_new_mask_protocol(
    masks_by_split: Mapping[str, torch.Tensor],
    old_rows: Mapping[int, set[tuple[int, ...]]],
) -> None:
    split_rows: defaultdict[int, list[set[tuple[int, ...]]]] = defaultdict(list)
    for split_name, masks in masks_by_split.items():
        if not torch.all((masks == 0) | (masks == 1)):
            raise RuntimeError(f"generated E29.9 {split_name} masks are not binary")
        cardinalities = masks.sum(dim=1).to(torch.int64)
        for cardinality in cardinalities.unique().tolist():
            rows = {row for row in _mask_rows(masks) if len(row) == int(cardinality)}
            expected_count = int((cardinalities == cardinality).sum().item())
            if len(rows) != expected_count:
                raise RuntimeError(f"generated E29.9 {split_name} masks repeat")
            if cardinality > 1 and not rows.isdisjoint(
                old_rows.get(cardinality, set())
            ):
                raise RuntimeError(
                    f"generated E29.9 {split_name} overlaps sealed E29.8 masks"
                )
            split_rows[int(cardinality)].append(rows)
    for cardinality, row_sets in split_rows.items():
        for left, right in itertools.combinations(row_sets, 2):
            if not left.isdisjoint(right):
                raise RuntimeError(
                    f"generated E29.9 cardinality-{cardinality} splits overlap"
                )


def build_e29_9_mask_protocol(
    sealed_e29_8_masks: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Build fresh deterministic masks while excluding every sealed E29.8 row.

    Singleton reuse is intentional: calibration identifies all 48 individual
    components. Every mask of cardinality two or larger must be fresh.
    """
    old_rows = _validate_and_collect_e29_8_rows(sealed_e29_8_masks)

    calibration_generator = random.Random(CALIBRATION_MASK_SEED)
    coverage = [1] * COMPONENT_COUNT
    calibration_rows: list[tuple[int, ...]] = [
        (component,) for component in range(COMPONENT_COUNT)
    ]
    calibration_rows.extend(
        _balanced_rows(
            cardinality=2,
            count=24,
            coverage=coverage,
            excluded=old_rows.get(2, set()),
            generator=calibration_generator,
        )
    )
    calibration_rows.extend(
        _balanced_rows(
            cardinality=3,
            count=24,
            coverage=coverage,
            excluded=old_rows.get(3, set()),
            generator=calibration_generator,
        )
    )
    if min(coverage) < 3:
        raise RuntimeError("E29.9 calibration incidence is below three")
    calibration_generator.shuffle(calibration_rows)

    validation_rows = _sample_fresh_rows(
        cardinality=8,
        count=32,
        excluded=old_rows.get(8, set()),
        generator=random.Random(VALIDATION_MASK_SEED),
    )
    final_rows = _sample_fresh_rows(
        cardinality=16,
        count=64,
        excluded=old_rows.get(16, set()),
        generator=random.Random(FINAL_MASK_SEED),
    )
    stress_rows = _sample_fresh_rows(
        cardinality=24,
        count=32,
        excluded=old_rows.get(24, set()),
        generator=random.Random(STRESS_MASK_SEED),
    )
    protocol = {
        "calibration": _masks_from_rows(calibration_rows),
        "validation_size8": _masks_from_rows(validation_rows),
        "final_size16": _masks_from_rows(final_rows),
        "stress_size24": _masks_from_rows(stress_rows),
    }
    _validate_new_mask_protocol(protocol, old_rows)
    return protocol


def validate_method_manifest(methods: Sequence[str]) -> tuple[str, ...]:
    """Reject a missing, added, or reordered E29.9 comparison."""
    manifest = tuple(methods)
    if manifest != METHOD_MANIFEST:
        raise ValueError(
            f"method manifest mismatch: expected {METHOD_MANIFEST}, got {manifest}"
        )
    return manifest


def primary_size16_gate_results(
    *,
    primary_metrics: Mapping[str, Any],
    additive_mse: float,
    affine_tt2_mse: float,
    scratch_mlp7_mse: float,
    permutation_control_mses: Sequence[float],
    random_feature_control_mses: Sequence[float],
    clean_margins: Sequence[float],
) -> dict[str, Any]:
    """Apply every frozen size-16 condition without weakening conjunctions."""
    gates = PRIMARY_SIZE16_GATES
    context_r2 = [float(value) for value in primary_metrics["per_context_r2"]]
    if len(context_r2) != CONTEXT_COUNT:
        raise ValueError("the E29.9 primary gate requires twelve contexts")
    if len(permutation_control_mses) != 12:
        raise ValueError("the E29.9 primary gate requires twelve permutation controls")
    if len(random_feature_control_mses) != 5:
        raise ValueError("the E29.9 primary gate requires five random-feature controls")

    primary_mse = float(primary_metrics["mse"])
    results = {
        "pooled_context_centered_r2_at_least_0_75": (
            float(primary_metrics["r2"]) >= gates["pooled_context_centered_r2_min"]
        ),
        "median_context_r2_at_least_0_70": (
            float(primary_metrics["median_context_r2"])
            >= gates["median_context_r2_min"]
        ),
        "at_least_ten_contexts_r2_at_least_0_40": (
            sum(value >= gates["context_r2_threshold"] for value in context_r2)
            >= gates["contexts_at_or_above_threshold_min"]
        ),
        "no_context_r2_below_zero": (
            min(context_r2) >= gates["minimum_context_r2_min"]
        ),
        "mse_at_most_0_60_additive": (
            primary_mse <= gates["mse_over_additive_mse_max"] * float(additive_mse)
        ),
        "mse_at_most_0_90_affine_tt2": (
            primary_mse <= gates["mse_over_affine_tt2_mse_max"] * float(affine_tt2_mse)
        ),
        "mse_at_most_1_10_scratch_mlp7": (
            primary_mse
            <= gates["mse_over_scratch_mlp7_mse_max"] * float(scratch_mlp7_mse)
        ),
        "mse_at_most_0_80_median_permutation_control": (
            primary_mse
            <= gates["mse_over_median_permutation_control_mse_max"]
            * statistics.median(float(value) for value in permutation_control_mses)
        ),
        "mse_at_most_0_80_median_random_feature_control": (
            primary_mse
            <= gates["mse_over_median_random_feature_control_mse_max"]
            * statistics.median(float(value) for value in random_feature_control_mses)
        ),
        "all_twelve_clean_margins_positive": (
            len(clean_margins) == gates["positive_clean_margins_required"]
            and all(float(margin) > 0.0 for margin in clean_margins)
        ),
    }
    return {"results": results, "all_passed": all(results.values())}


def stress_size24_gate_results(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the independent size-24 extrapolation gate."""
    gates = STRESS_SIZE24_GATES
    context_r2 = [float(value) for value in metrics["per_context_r2"]]
    if len(context_r2) != CONTEXT_COUNT:
        raise ValueError("the E29.9 stress gate requires twelve contexts")
    results = {
        "pooled_context_centered_r2_at_least_0_55": (
            float(metrics["r2"]) >= gates["pooled_context_centered_r2_min"]
        ),
        "median_context_r2_at_least_0_50": (
            float(metrics["median_context_r2"]) >= gates["median_context_r2_min"]
        ),
        "at_least_ten_contexts_r2_positive": (
            sum(value > 0.0 for value in context_r2)
            >= gates["positive_context_r2_min_count"]
        ),
    }
    return {"results": results, "all_passed": all(results.values())}


def e29_9_gate_results(
    method_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    clean_margins: Sequence[float],
    primary_design_rank: int,
    primary_design_condition: float,
) -> dict[str, Any]:
    """Apply the canonical preflight, size-16, and separate stress gates."""
    missing = set(METHOD_MANIFEST) - method_metrics.keys()
    if missing:
        raise ValueError(f"method metrics omit {sorted(missing)}")

    permutation_mses = [
        float(method_metrics[name]["final_size16"]["mse"])
        for name in LEARNED_PERMUTATION_METHODS
    ]
    random_feature_mses = [
        float(method_metrics[name]["final_size16"]["mse"])
        for name in RANDOM_FEATURE_METHODS
    ]
    size16 = primary_size16_gate_results(
        primary_metrics=method_metrics[PRIMARY_METHOD]["final_size16"],
        additive_mse=float(method_metrics["additive"]["final_size16"]["mse"]),
        affine_tt2_mse=float(method_metrics["tt2_frozen"]["final_size16"]["mse"]),
        scratch_mlp7_mse=float(method_metrics["mlp7_scratch"]["final_size16"]["mse"]),
        permutation_control_mses=permutation_mses,
        random_feature_control_mses=random_feature_mses,
        clean_margins=clean_margins,
    )
    preflight_results = {
        "primary_standardized_design_rank_is_seven": (
            primary_design_rank == FEATURE_PREFLIGHT["required_design_rank"]
        ),
        "primary_standardized_design_condition_at_most_1000": (
            torch.isfinite(torch.tensor(float(primary_design_condition))).item()
            and float(primary_design_condition)
            <= FEATURE_PREFLIGHT["condition_number_max"]
        ),
    }
    preflight = {
        "results": preflight_results,
        "all_passed": all(preflight_results.values()),
    }
    stress = stress_size24_gate_results(method_metrics[PRIMARY_METHOD]["stress_size24"])
    return {
        "preflight": preflight,
        "size16": {
            **size16,
            "permutation_control_median_mse": statistics.median(permutation_mses),
            "permutation_control_count": len(permutation_mses),
            "random_feature_control_median_mse": statistics.median(random_feature_mses),
            "random_feature_control_count": len(random_feature_mses),
        },
        "stress": stress,
        "primary_confirmation_passed": (
            preflight["all_passed"] and size16["all_passed"]
        ),
        "stress_passed": stress["all_passed"],
        "stress_failure_does_not_erase_size16_confirmation": True,
    }


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CALIBRATION_MASK_SEED",
    "COMPONENT_COUNT",
    "COMPONENT_PERMUTATION_SEEDS",
    "CONTEXT_COUNT",
    "CONTROL_METHODS",
    "FEATURE_PREFLIGHT",
    "FINAL_MASK_SEED",
    "LEARNED_PERMUTATION_METHODS",
    "METHOD_MANIFEST",
    "PRIMARY_METHOD",
    "PRIMARY_NEW_PARAMETER_COUNT",
    "PRIMARY_SIZE16_GATES",
    "QUERY_BUDGET",
    "RANDOM_FEATURE_METHODS",
    "RANDOM_FEATURE_SEEDS",
    "READOUT_PROTOCOL",
    "RIDGE_VALUES",
    "SCRATCH_MLP7_PARAMETER_COUNT",
    "SCRATCH_TRAINING_SEEDS",
    "SECONDARY_ENSEMBLE_WEIGHTS",
    "SECONDARY_METHOD",
    "SECONDARY_OPERATIONAL_PARAMETER_COUNT",
    "SELECTED_SOURCE_MLP_SEED",
    "SOURCE_MLP_TRAINING_SEEDS",
    "STRESS_MASK_SEED",
    "STRESS_SIZE24_GATES",
    "VALIDATION_MASK_SEED",
    "build_e29_9_mask_protocol",
    "e29_9_gate_results",
    "e29_9_transfer_prompt_panel",
    "primary_size16_gate_results",
    "stress_size24_gate_results",
    "validate_e29_9_prompt_panel",
    "validate_method_manifest",
]
