# ruff: noqa: EXE002
"""Pure prospective protocol for the E31.1 conditional interaction atlas.

The design measures every GPT-2-small sublayer pair by exact four-cell
interventions at three background scales.  Shared backgrounds make the model
acquisition substantially cheaper without changing an edge's forced-off
background.  Everything in this module is response blind: no model is loaded,
no label is read, and no registration is written.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from functools import cache
from itertools import combinations
from typing import Any

from experiments.e31_sparse_interactions.panel import (
    PROMPT_COUNTS,
    PROMPT_MANIFEST_SHA256,
    build_prompt_manifest,
    canonical_sha256,
)

SCHEMA_VERSION = "e31_1_prospective_conditional_sparse_atlas_v1"
DESIGN_DOMAIN = "e31.1|prospective|conditional-sparse-atlas|3101001"
COMPONENT_COUNT = 24
HUB_COMPONENT = 1
NONHUB_COMPONENTS = tuple(
    component for component in range(COMPONENT_COUNT) if component != HUB_COMPONENT
)
NONHUB_EDGE_COUNT = 253
SPOKE_COUNT = 23
SCALE_CARDINALITIES = (2, 10, 18)
COVER_TARGET = 4
ORDERED_CELL_BITS = ("00", "10", "01", "11")
ASINH_LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
SATURATION_EDGE_FOLD_COUNT = 8
SATURATION_HUBER_DELTA = 0.05
SMOKE_MASK_COUNT = 16
BLOCK_COUNT_CEILINGS = {2: 8, 10: 24, 18: 140}
PRIMARY_STRATUM = {"hub_state": "off", "k_nonhub_on": 10}
PRIMARY_GATES = {
    "top24_confirmation_energy_coverage_minimum": 0.60,
    "top24_fraction_of_confirmation_oracle_coverage_minimum": 0.75,
    "top48_confirmation_energy_coverage_minimum": 0.80,
    "replicated_top24_sign_count_minimum": 18,
    "discovery_confirmation_signed_interaction_pearson_minimum": 0.50,
    "top24_coverage_one_sided_95_lower_bound_minimum": 0.50,
    "signed_interaction_pearson_one_sided_95_lower_bound_strict_minimum": 0.0,
    "residual_top24_in_raw_top48_count_minimum": 12,
    "matched_random_support_quantile_strictly_exceeded": 0.95,
    "matched_random_support_count": 10_000,
}
HIERARCHICAL_SCALE_GATE = {
    "regime_id": "nonhub_k02_hub_off",
    "tested_only_if_primary_passes": True,
    "top24_confirmation_energy_coverage_minimum": 0.50,
    "top24_coverage_one_sided_95_lower_bound_minimum": 0.40,
}
BOOTSTRAP = {
    "replicates": 10_000,
    "seed": 3_101_001,
    "one_sided_confidence_level": 0.95,
    "resample_unit": "semantic_frame_with_both_directions",
    "stratified_by": "confirmation_template_stratum",
    "common_resamples": True,
}

SPLIT_MANIFEST_SHA256 = {
    "discovery": "9594548db207797995e35a9b720cf9b5b75fb14556198a552265adc500330745",
    "confirmation": "662b5b257c9d7ea630e6cd60fb98aaca4e85bbaa601f05aa8b73f2f6164adb50",
}
PROTOCOL_SHA256 = "14d4ef8814b7c8f6a5d08f0113ea9634ec4e6fbdc78c35ba07b1be814b743221"


def _digest(label: str) -> str:
    return hashlib.sha256(f"{DESIGN_DOMAIN}|{label}".encode("ascii")).hexdigest()


def _candidate_complements(
    split: str,
    k: int,
    iteration: int,
    target_edge: tuple[int, int],
) -> tuple[tuple[int, ...], ...]:
    """Generate stable candidate absent-coordinate blocks around one weak edge."""
    complement_size = len(NONHUB_COMPONENTS) - k
    remainder = tuple(
        component for component in NONHUB_COMPONENTS if component not in target_edge
    )
    choose_count = complement_size - 2
    candidates: set[tuple[int, ...]] = set()
    # The small endpoint cases can be searched exhaustively.  The middle case
    # uses a fixed SHA-ranked sample so behavior is independent of Python RNG.
    if choose_count <= 3 or len(remainder) - choose_count <= 3:
        sources = combinations(remainder, choose_count)
    else:
        generated = []
        for salt in range(2_048):
            ranked = sorted(
                remainder,
                key=lambda component: _digest(
                    f"candidate|{split}|{k}|{iteration}|{target_edge}|{salt}|{component}"
                ),
            )
            generated.append(tuple(ranked[:choose_count]))
        sources = generated
    for selected in sources:
        candidates.add(tuple(sorted((*target_edge, *selected))))
    return tuple(sorted(candidates))


@cache
def _covering_backgrounds(split: str, k: int) -> tuple[tuple[int, ...], ...]:
    """Return exact-k backgrounds whose complements cover each pair four times."""
    if split not in PROMPT_COUNTS:
        raise ValueError("split must be 'discovery' or 'confirmation'")
    if k not in SCALE_CARDINALITIES:
        raise ValueError(f"k must be one of {SCALE_CARDINALITIES}")
    edges = tuple(combinations(NONHUB_COMPONENTS, 2))
    coverage = {edge: 0 for edge in edges}
    component_absence = Counter({component: 0 for component in NONHUB_COMPONENTS})
    complement_blocks: list[tuple[int, ...]] = []

    while min(coverage.values()) < COVER_TARGET:
        iteration = len(complement_blocks)
        weakest = min(coverage.values())
        target_edge = min(
            (edge for edge in edges if coverage[edge] == weakest),
            key=lambda edge: _digest(
                f"target|{split}|{k}|{iteration}|{edge[0]}|{edge[1]}"
            ),
        )
        candidates = _candidate_complements(split, k, iteration, target_edge)
        if not candidates:
            raise RuntimeError("the response-blind covering search produced no block")

        def score(
            candidate: tuple[int, ...], *, frozen_iteration: int = iteration
        ) -> tuple[int, int, int, int, str]:
            candidate_edges = tuple(combinations(candidate, 2))
            deficit_gain = sum(
                max(0, COVER_TARGET - coverage[edge]) for edge in candidate_edges
            )
            newly_covered = sum(
                coverage[edge] < COVER_TARGET for edge in candidate_edges
            )
            overcoverage = sum(
                max(0, coverage[edge] + 1 - COVER_TARGET) for edge in candidate_edges
            )
            imbalance = sum(component_absence[node] for node in candidate)
            return (
                deficit_gain,
                newly_covered,
                -overcoverage,
                -imbalance,
                _digest(f"choice|{split}|{k}|{frozen_iteration}|{candidate}"),
            )

        selected_complement = max(candidates, key=score)
        complement_blocks.append(selected_complement)
        for edge in combinations(selected_complement, 2):
            coverage[edge] += 1
        component_absence.update(selected_complement)
        if len(complement_blocks) > BLOCK_COUNT_CEILINGS[k]:
            raise RuntimeError(f"the k={k} covering design exceeded its frozen ceiling")

    nonhub = set(NONHUB_COMPONENTS)
    return tuple(
        tuple(sorted(nonhub - set(complement))) for complement in complement_blocks
    )


def build_covering_background_manifest(split: str) -> tuple[dict[str, Any], ...]:
    """Build all shared exact-scale backgrounds for one prospective split."""
    if split not in PROMPT_COUNTS:
        raise ValueError("split must be 'discovery' or 'confirmation'")
    rows: list[dict[str, Any]] = []
    for k in SCALE_CARDINALITIES:
        for scale_index, selected in enumerate(_covering_backgrounds(split, k)):
            absent = tuple(
                component
                for component in NONHUB_COMPONENTS
                if component not in selected
            )
            background_id = f"{split}_k{k:02d}_background_{scale_index:03d}"
            rows.append(
                {
                    "background_index": len(rows),
                    "background_id": background_id,
                    "split": split,
                    "k_nonhub_on": k,
                    "selected_components": list(selected),
                    "absent_nonhub_components": list(absent),
                    "absent_nonhub_count": len(absent),
                    "background_sha256": canonical_sha256(list(selected)),
                    "response_blind": True,
                    "generator": "sha256_greedy_four_cover_v1",
                }
            )
    return tuple(rows)


def _selected_cover_rows(
    rows: tuple[dict[str, Any], ...],
    *,
    split: str,
    k: int,
    required_absent: tuple[int, ...],
) -> tuple[dict[str, Any], ...]:
    eligible = [
        row
        for row in rows
        if row["k_nonhub_on"] == k
        and set(required_absent) <= set(row["absent_nonhub_components"])
    ]
    ordered = sorted(
        eligible,
        key=lambda row: _digest(
            f"cover-rank|{split}|{k}|{required_absent}|{row['background_sha256']}"
        ),
    )
    if len(ordered) < COVER_TARGET:
        raise RuntimeError("an E31 edge has fewer than four background covers")
    return tuple(ordered[:COVER_TARGET])


@cache
def _build_split_mask_manifest_cached(split: str) -> dict[str, Any]:
    if split not in PROMPT_COUNTS:
        raise ValueError("split must be 'discovery' or 'confirmation'")
    backgrounds = build_covering_background_manifest(split)
    condition_index: dict[tuple[int, ...], int] = {}
    conditions: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []

    def condition_id(components: tuple[int, ...]) -> str:
        selected = tuple(sorted(set(components)))
        if selected not in condition_index:
            index = len(conditions)
            condition_index[selected] = index
            mask = [int(component in selected) for component in range(COMPONENT_COUNT)]
            conditions.append(
                {
                    "condition_index": index,
                    "condition_id": f"e31_{split}_condition_{index:05d}",
                    "selected_components": list(selected),
                    "cardinality": len(selected),
                    "mask": mask,
                    "mask_sha256": canonical_sha256(mask),
                }
            )
        return str(conditions[condition_index[selected]]["condition_id"])

    nonhub_edges = tuple(combinations(NONHUB_COMPONENTS, 2))
    for k in SCALE_CARDINALITIES:
        for hub_state in ("off", "on"):
            for first, second in nonhub_edges:
                cover_rows = _selected_cover_rows(
                    backgrounds,
                    split=split,
                    k=k,
                    required_absent=(first, second),
                )
                for cover_rank, row in enumerate(cover_rows):
                    base = tuple(row["selected_components"])
                    if hub_state == "on":
                        base = tuple(sorted((*base, HUB_COMPONENT)))
                    cells = (
                        base,
                        tuple(sorted((*base, first))),
                        tuple(sorted((*base, second))),
                        tuple(sorted((*base, first, second))),
                    )
                    contrasts.append(
                        {
                            "contrast_index": len(contrasts),
                            "contrast_id": (
                                f"{split}::k{k}::{hub_state}::edge_{first:02d}_{second:02d}"
                                f"::cover_{cover_rank}"
                            ),
                            "edge_class": "nonhub_pair",
                            "edge": [first, second],
                            "factor_components": [first, second],
                            "k_nonhub_on": k,
                            "hub_state": hub_state,
                            "cover_rank": cover_rank,
                            "background_id": row["background_id"],
                            "background_sha256": row["background_sha256"],
                            "ordered_cell_bits": list(ORDERED_CELL_BITS),
                            "condition_ids": [condition_id(cell) for cell in cells],
                            "interaction_formula": "R11-R10-R01+R00",
                        }
                    )
        for spoke in NONHUB_COMPONENTS:
            cover_rows = _selected_cover_rows(
                backgrounds,
                split=split,
                k=k,
                required_absent=(spoke,),
            )
            for cover_rank, row in enumerate(cover_rows):
                base = tuple(row["selected_components"])
                cells = (
                    base,
                    tuple(sorted((*base, HUB_COMPONENT))),
                    tuple(sorted((*base, spoke))),
                    tuple(sorted((*base, HUB_COMPONENT, spoke))),
                )
                contrasts.append(
                    {
                        "contrast_index": len(contrasts),
                        "contrast_id": (
                            f"{split}::k{k}::crossing::spoke_{spoke:02d}"
                            f"::cover_{cover_rank}"
                        ),
                        "edge_class": "hub_spoke",
                        "edge": sorted((HUB_COMPONENT, spoke)),
                        "factor_components": [HUB_COMPONENT, spoke],
                        "k_nonhub_on": k,
                        "hub_state": "crossing",
                        "cover_rank": cover_rank,
                        "background_id": row["background_id"],
                        "background_sha256": row["background_sha256"],
                        "ordered_cell_bits": list(ORDERED_CELL_BITS),
                        "condition_ids": [condition_id(cell) for cell in cells],
                        "interaction_formula": "R11-R10-R01+R00",
                    }
                )

    block_counts = {
        str(k): sum(row["k_nonhub_on"] == k for row in backgrounds)
        for k in SCALE_CARDINALITIES
    }
    neighborhood_upper = sum(
        count
        * 2
        * (
            1
            + (len(NONHUB_COMPONENTS) - int(k))
            + (len(NONHUB_COMPONENTS) - int(k))
            * (len(NONHUB_COMPONENTS) - int(k) - 1)
            // 2
        )
        for k, count in block_counts.items()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "status": "response_blind_unsealed_manifest",
        "backgrounds": list(backgrounds),
        "background_count": len(backgrounds),
        "background_count_by_k": block_counts,
        "contrasts": contrasts,
        "contrast_count": len(contrasts),
        "logical_cell_reference_count": 4 * len(contrasts),
        "conditions": conditions,
        "unique_condition_count": len(conditions),
        "shared_neighborhood_unique_condition_upper_bound": neighborhood_upper,
        "deduplication_rule": "one condition per exact sorted 24-bit component mask",
    }


def build_split_mask_manifest(split: str) -> dict[str, Any]:
    """Return a detached copy of one exact deduplicated split manifest."""
    return deepcopy(_build_split_mask_manifest_cached(split))


def build_prospective_protocol() -> dict[str, Any]:
    """Compose the complete unsealed response-blind E31.1 protocol."""
    split_manifests = {
        split: build_split_mask_manifest(split)
        for split in ("discovery", "confirmation")
    }
    deduplicated_cells = sum(
        split_manifests[split]["unique_condition_count"] * PROMPT_COUNTS[split]
        for split in split_manifests
    )
    logical_quartet_cells = sum(
        split_manifests[split]["logical_cell_reference_count"] * PROMPT_COUNTS[split]
        for split in split_manifests
    )
    clean_forwards = 2 * sum(PROMPT_COUNTS.values())
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "status": "response_blind_unsealed_no_model_calls",
        "model": {
            "model_id": "gpt2",
            "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
            "dtype": "float64",
            "device": "cpu",
            "component_count": COMPONENT_COUNT,
            "hub_component": HUB_COMPONENT,
            "component_semantics": "all_position_attention_and_mlp_residual_updates",
        },
        "development_ancestry": {
            "exploratory_atlas_sha256": (
                "3977f8416f6db2df702d66e33ccf00967b3e2741f794c2c67d50bd7b5967a248"
            ),
            "exploratory_custody_sha256": (
                "ae90e78669022d5bdc8bf993035e655913151a95b7503f199bd59cfa42141883"
            ),
            "development_outcomes_informed": [
                "conditional rather than global sparsity hypothesis",
                "primary hub-off middle-cardinality regime",
                "top24 and top48 support sizes",
                "arcsinh saturation-null family and numeric gates",
            ],
            "development_outcomes_did_not_inform": [
                "fresh prompt text or vocabulary",
                "fresh hash-greedy background membership",
                "discovery-selected edge support",
                "confirmation outcomes",
            ],
            "fresh_claim_is_prospective_relative_to": (
                "the exact E31.1 discovery and confirmation prompts, backgrounds, "
                "ranking, and registered pass conjunction"
            ),
        },
        "prompts": {
            split: list(build_prompt_manifest(split))
            for split in ("discovery", "confirmation")
        },
        "prompt_manifest_sha256": dict(PROMPT_MANIFEST_SHA256),
        "split_manifests": split_manifests,
        "split_manifest_sha256": dict(SPLIT_MANIFEST_SHA256),
        "estimand": {
            "response": (
                "(patched_[logit(donor_target)-logit(base_target)]-clean_base_contrast)"
                "/(clean_donor_contrast-clean_base_contrast)"
            ),
            "clipping": False,
            "raw_pair_interaction": "I=R11-R10-R01+R00",
            "saturation_link": "g_lambda(z)=asinh(lambda*z)/asinh(lambda); g_0(z)=z",
            "inverse_link": "h_lambda(y)=sinh(y*asinh(lambda))/lambda; h_0(y)=y",
            "saturation_null": (
                "R11_null=g_lambda(h_lambda(R10)+h_lambda(R01)-h_lambda(R00))"
            ),
            "residual_pair_interaction": "J=R11-R11_null",
            "lambda_grid": list(ASINH_LAMBDA_GRID),
            "huber_delta": SATURATION_HUBER_DELTA,
            "huber_loss": ("0.5*x^2 if abs(x)<=0.05 else 0.05*(abs(x)-0.025)"),
            "edge_fold_count": SATURATION_EDGE_FOLD_COUNT,
            "edge_fold": (
                "int(SHA256(DESIGN_DOMAIN+'|lambda-fold|'+edge_class+'|'"
                "+canonical_sorted_edge0+'|'+canonical_sorted_edge1),16) modulo 8"
            ),
            "discovery_crossfit_lambda_selection": (
                "for each row with template t and edge fold f, minimize mean Huber(J) "
                "within the same genuine stratum over discovery rows whose template !=t "
                "and edge fold !=f; ties choose the smaller lambda; use that lambda only "
                "for the held row and rank edges by mean held-row J^2"
            ),
            "confirmation_lambda_selection": (
                "within each genuine stratum minimize mean Huber(J) over every discovery "
                "row; ties choose the smaller lambda; hash and freeze one lambda per "
                "stratum before opening confirmation responses"
            ),
        },
        "primary": {
            "stratum": dict(PRIMARY_STRATUM),
            "edge_energy": "mean_over_context_and_cover_of_J_squared",
            "support_ranking": "descending_discovery_energy_then_lexicographic_edge",
            "gates": dict(PRIMARY_GATES),
            "bootstrap": dict(BOOTSTRAP),
            "matched_random_control": {
                "support_count": 10_000,
                "seed": 3_101_002,
                "support_size": 24,
                "component_strength": (
                    "for each nonhub component, pool squared marginal effects across "
                    "every incident primary-discovery edge, context, cover, and both "
                    "states of the other endpoint; average the pool"
                ),
                "component_strength_quartiles": (
                    "sort the 23 strengths ascending with component id as the tie-break, "
                    "then assign rank-floor(4*rank/23)"
                ),
                "match_exact_selected_counts_over": [
                    "pair_kind_AA_AM_MM",
                    "absolute_layer_distance_bin_0_to_2_3_to_5_6_plus",
                    "unordered_pair_of_endpoint_discovery_strength_quartiles",
                ],
                "generation": (
                    "for salt=0,1,... and each required joint bin in sorted string order, "
                    "rank candidate edges by SHA256('3101002|'+salt+'|'+Python str(bin)"
                    "+'|'+edge0+'|'+edge1), take the registered count from that bin, "
                    "sort their union, exclude the exact selected top24 and duplicates, "
                    "and retain the first 10000 unique supports"
                ),
                "sample_without_replacement_within_each_support": True,
                "supports_must_be_unique": True,
                "pass_rule": "strictly_exceed_empirical_95th_percentile_confirmation_coverage",
            },
        },
        "clean_screen": {
            "all_splits_screened_before_any_patched_call": True,
            "clean_logits_finite_per_context": True,
            "clean_logit_absolute_maximum": 10_000.0,
            "minimum_endpoint_per_context": 0.25,
            "minimum_base_correct_fraction_per_split": 0.75,
            "minimum_donor_correct_fraction_per_split": 0.75,
            "mean_base_correct_margin_positive_per_split": True,
            "mean_donor_correct_margin_positive_per_split": True,
            "row_policy": "retain_every_registered_context",
            "failure_policy": "write_failure_custody_and_acquire_no_patched_responses",
        },
        "batch_invariance_smoke": {
            "split": "discovery",
            "context_index": 0,
            "mask_count": SMOKE_MASK_COUNT,
            "mask_selection": (
                "rank every discovery condition by lowercase SHA256 of "
                "DESIGN_DOMAIN+'|batch-smoke|'+condition_id+'|'+mask_sha256, "
                "tie-break by condition_index, and take the first 16"
            ),
            "reference_batch_size": 1,
            "candidate_batch_size": 32,
            "atol": 1e-9,
            "rtol": 1e-9,
            "all_coordinates_must_pass": True,
        },
        "custody_order": [
            "seal registration and write durable global attempt before tokenizer or model",
            "prepare and gate all 48 clean contexts before any patched response",
            "acquire and write discovery responses",
            "fit and write-once seal discovery lambdas ranking top24 and controls",
            "write confirmation authorization binding the discovery seal",
            "acquire and write confirmation responses without adapting geometry",
            "reverify dependencies and write one exact pure-score result",
        ],
        "query_accounting": {
            "logical_quartet_context_cell_references": logical_quartet_cells,
            "deduplicated_patch_context_mask_cells": deduplicated_cells,
            "total_clean_forwards": clean_forwards,
            "batch_smoke_physical_context_mask_cells": 2 * SMOKE_MASK_COUNT,
            "maximum_registered_model_forwards": (
                deduplicated_cells + clean_forwards + 2 * SMOKE_MASK_COUNT
            ),
            "note": "E31.1 acquires only the exhaustive atlas and clean endpoints",
        },
        "decision": {
            "atlas_go": (
                "all primary gates pass, then the single frozen hierarchical hub-off "
                "k=2 scale gate also passes"
            ),
            "hierarchical_scale_gate": dict(HIERARCHICAL_SCALE_GATE),
            "if_residual_sparsity_fails": "test_sparse_backbone_plus_dense_low_rank_residual",
            "if_support_transfer_fails": "treat_graph_as_context_specific_not_population_stable",
            "if_only_raw_signal_survives": "classify_as_endpoint_curvature_not_mechanism",
            "next_after_atlas_go": (
                "separately preregister predictive sparse-graph utility and serial-versus-"
                "cooperative topology tests before task/model/head-level promotion"
            ),
        },
        "claim_boundary": (
            "E31.1 tests task-, intervention-, hub-, and scale-conditioned sublayer "
            "interaction sparsity on GPT-2 small IOI; it does not establish global model "
            "sparsity, direct communication, architecture universality, or tensor superiority."
        ),
    }
    return protocol


def validate_prospective_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Reject any changed prompt, mask, contrast, score, or accounting rule."""
    if not isinstance(protocol, Mapping):
        raise TypeError("protocol must be a mapping")
    copied = deepcopy(dict(protocol))
    expected = build_prospective_protocol()
    if copied != expected:
        raise ValueError("the E31.1 prospective protocol changed")
    for split in ("discovery", "confirmation"):
        observed = canonical_sha256(copied["split_manifests"][split])
        if observed != SPLIT_MANIFEST_SHA256[split]:
            raise RuntimeError(f"the frozen {split} mask manifest hash changed")
    if canonical_sha256(copied) != PROTOCOL_SHA256:
        raise RuntimeError("the frozen E31.1 protocol hash changed")
    if copied["query_accounting"]["maximum_registered_model_forwards"] >= 700_000:
        raise RuntimeError("the E31.1 prospective acquisition ceiling changed")
    return copied


__all__ = [
    "ASINH_LAMBDA_GRID",
    "BLOCK_COUNT_CEILINGS",
    "BOOTSTRAP",
    "COMPONENT_COUNT",
    "COVER_TARGET",
    "HIERARCHICAL_SCALE_GATE",
    "HUB_COMPONENT",
    "NONHUB_COMPONENTS",
    "NONHUB_EDGE_COUNT",
    "ORDERED_CELL_BITS",
    "PRIMARY_GATES",
    "PRIMARY_STRATUM",
    "PROTOCOL_SHA256",
    "SATURATION_EDGE_FOLD_COUNT",
    "SATURATION_HUBER_DELTA",
    "SCALE_CARDINALITIES",
    "SCHEMA_VERSION",
    "SMOKE_MASK_COUNT",
    "SPLIT_MANIFEST_SHA256",
    "SPOKE_COUNT",
    "build_covering_background_manifest",
    "build_prospective_protocol",
    "build_split_mask_manifest",
    "canonical_sha256",
    "validate_prospective_protocol",
]
