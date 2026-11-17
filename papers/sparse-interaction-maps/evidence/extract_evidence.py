#!/usr/bin/env python3
"""Extract paper evidence from immutable scores; perform no model inference.

Run from the repository with its existing torch environment.  Original score
artifacts are trusted local inputs and are never modified.  Large per-example
arrays are deliberately omitted; exact source hashes retain their provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.e32_interpbench.e32_1_protocol import (  # noqa: E402
    EXPECTED_INTERNAL_GROUND_TRUTH,
)
from experiments.e32_interpbench.methods import (  # noqa: E402
    masked_rank_one_endpoint_factor,
)
from experiments.e32_interpbench.scoring import binary_ranking_metrics  # noqa: E402

SOURCES = {
    "e31_1": "runs/e31_1_prospective_sparse_atlas_v2_order_repair/prospective_score.pt",
    "e31_2": "runs/e31_2_mixed_resolution_v1/prospective_score.pt",
    "e31_2_posthoc": "runs/e31_2_posthoc_node_residuals_v1/report.json",
    "e32_1": "runs/e32_1_exact_gt_v1/summary.json",
    "e32_1_confirmation": "runs/e32_1_exact_gt_v1/case97_confirmation_atlas.pt",
    "e32_2": "runs/e32_2_node_first_directed_v1/summary.json",
    "e32_3": "runs/e32_3_sparse_core_v2_input_repair/summary.json",
    "e32_4": "runs/e32_4_internal_edge_set_faithfulness_v1/summary.json",
    "e32_1_protocol": "experiments/e32_interpbench/e32_1_protocol.py",
    "e32_methods": "experiments/e32_interpbench/methods.py",
    "e32_scoring": "experiments/e32_interpbench/scoring.py",
    "e32_3_scoring": "experiments/e32_interpbench/e32_3_scoring.py",
    "e32_4_scoring": "experiments/e32_interpbench/e32_4_scoring.py",
    "e32_4_protocol": "experiments/e32_interpbench/e32_4_protocol.py",
    "e31_1_note": "notes/E31.1-CONDITIONAL-SPARSE-INTERACTION-ATLAS-RESULTS.md",
    "e31_2_note": "notes/E31.2-MIXED-RESOLUTION-HEAD-ATLAS-RESULTS.md",
    "e32_1_note": "notes/E32.1-EXACT-GT-SPARSE-INTERACTION-RESULTS.md",
}
OMIT_KEYS = {
    "per_example", "selected_indices", "oracle_indices", "selected_support",
    "full_universe_score_vectors", "full_vectors", "joint_loss_recovery",
    "ranks_one_based", "selected_edges_at_gt_k", "expected_outputs",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(value: Any) -> Any:
    """Preserve scalar trees and small identifiers, excluding sample arrays."""
    if isinstance(value, torch.Tensor):
        return compact(value.item()) if value.numel() == 1 else None
    if isinstance(value, dict):
        return {str(k): compact(v) for k, v in value.items()
                if k not in OMIT_KEYS and compact(v) is not None}
    if isinstance(value, (list, tuple)):
        if len(value) <= 24 and all(isinstance(x, (str, bool, int)) for x in value):
            return list(value)
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported evidence value: {type(value)!r}")


def read_source(name: str) -> dict[str, Any]:
    path = ROOT / SOURCES[name]
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=False)
    return json.loads(path.read_text())


def pick(record: dict[str, Any], *keys: str) -> dict[str, Any]:
    return compact({k: record[k] for k in keys})


def records(value: Any) -> Any:
    """Score schemas use either case-ID dictionaries or ordered case lists."""
    return value.values() if isinstance(value, dict) else value


def e32_1_posthoc(atlas: dict[str, Any]) -> dict[str, Any]:
    """Replay note-level reachability/factor diagnostics on sealed labels."""
    edges = tuple(tuple(edge) for edge in atlas["directed_candidate_edges"])
    nodes = tuple(atlas["component_ids"])
    interactions = atlas["pair_interactions"].double()
    scores = interactions.square().mean(dim=tuple(range(1, interactions.ndim)))
    direct = set(EXPECTED_INTERNAL_GROUND_TRUTH[97]["edges"])
    closure = set(direct)
    for middle in nodes:
        closure |= {(a, b) for a, m in tuple(closure)
                    for m2, b in tuple(closure) if m == middle == m2 and a != b}
    transitive = closure - direct
    ranked = sorted(range(len(edges)), key=lambda i: (-float(scores[i]), i))
    top10 = [edges[i] for i in ranked[:10]]
    def energy(support: set[tuple[str, str]]) -> float:
        return float(scores[[i for i, edge in enumerate(edges) if edge in support]].sum()
                     / scores.sum())
    factor = masked_rank_one_endpoint_factor(
        node_universe=nodes, edge_universe=edges, edge_scores=scores,
    )
    centered_energy = float((scores - scores.mean()).square().sum())
    return {
        "analysis_status": "posthoc_replayed_from_sealed_confirmation_labels",
        "model_calls": 0,
        "direct_edge_count": len(direct),
        "transitive_nonedge_count": len(transitive),
        "reachability_closure_count": len(closure),
        "top10_direct_count": sum(edge in direct for edge in top10),
        "top10_transitive_nonedge_count": sum(edge in transitive for edge in top10),
        "top10_unrelated_count": sum(edge not in closure for edge in top10),
        "direct_energy_fraction": energy(direct),
        "transitive_nonedge_energy_fraction": energy(transitive),
        "unrelated_energy_fraction": 1.0 - energy(closure),
        "closure_auprc": binary_ranking_metrics(
            scores=scores.tolist(), labels=[edge in closure for edge in edges],
        )["auprc"],
        "rank_one_squared_norm_explained": factor.explained_energy_fraction,
        "rank_one_centered_r2": 1.0 - factor.squared_error / centered_energy,
        "rank_one_converged": factor.converged,
        "rank_one_iterations": factor.iterations,
        "warning": "Confirmation-local factor fit is descriptive, not cross-split prediction.",
    }


def build_manifest() -> dict[str, Any]:
    source_records = {
        name: {"relative_path": rel, "absolute_path": str(ROOT / rel),
               "sha256": sha256(ROOT / rel), "size_bytes": (ROOT / rel).stat().st_size}
        for name, rel in SOURCES.items()
    }
    e311 = read_source("e31_1")
    e312 = read_source("e31_2")
    posthoc = read_source("e31_2_posthoc")
    e321 = read_source("e32_1")
    e322 = read_source("e32_2")
    e323 = read_source("e32_3")
    e324 = read_source("e32_4")
    core_cases = {}
    for case in records(e323["cases"]):
        core_cases[str(case["case_id"])] = {
            **pick(case, "case_id", "structure", "gates", "retention", "stability",
                   "selector_comparison", "confirmatory_support_fraction"),
            "primary_set_faithfulness": compact(
                case["set_faithfulness"]["singleton_output_effect"]),
        }
    edge_cases = {}
    for case in records(e324["cases"]):
        edge_cases[str(case["case_id"])] = {
            **pick(case, "case_id", "gates", "primary_support_size",
                   "sparsity_fraction_of_all_internal_fine_edges", "selector_comparison"),
            "resample_primary": compact(case["resample_deterministic"]["bidirectional_exact_mass"]),
            "resample_bridge": compact(case["resample_bridge"]),
            "mean_corrupt_primary": compact(case["mean_corrupt_primary"]),
            "mean_corrupt_bridge": compact(case["mean_corrupt_bridge"]),
            "input_endpoint_alignment": compact(case["input_endpoint_alignment"]),
            "within_core_fallback_descriptive_not_primary": compact(
                case["resample_within_core_primary_fallback"]),
        }
    return {
        "schema_version": "sparse_interaction_paper_evidence_v1",
        "source_policy": "Immutable local scores. No model loading or inference. Original verdicts retained.",
        "sources": source_records,
        "e31_1": {"evidence_status": "registered_conditional_sublayer_atlas_failed",
                   "actual_model_examples": e311["actual_lifetime_forward_examples"],
                   "score": compact(e311["score"])},
        "e31_2": {"evidence_status": "registered_head_test_failed_conjunction",
                   "parent_selection": "posthoc_E31.1; E31.2_head_outcomes_prospective",
                   "actual_model_examples": e312["actual_model_forward_examples"],
                   "score": compact(e312["pure_score"])},
        "e31_2_posthoc": {"evidence_status": "posthoc_zero_model_calls",
            **pick(posthoc, "status", "definitions", "claim_boundaries",
                   "discovery_context_count", "confirmation_context_count", "pooled_predictors"),
            "parents": {key: compact(val) for key, val in posthoc["parents"].items()}},
        "e32_1": {"evidence_status": "registered_direct_edge_superiority_failed",
                   **compact(e321), "posthoc_reachability": e32_1_posthoc(
                       read_source("e32_1_confirmation")["atlas"])},
        "e32_2": {"evidence_status": "registered_directed_replacement_utility_and_superiority_failed",
            **pick(e322, "schema_version", "overall_useful_pass", "overall_superiority_pass",
                   "claim_boundary", "actual_query_accounting"),
            "discovery_examples_per_case": 64, "confirmation_examples_per_case": 128,
            "case_results": {str(case["case_id"]): pick(case, "case_id", "structure",
                "confirmation_metrics", "gates", "direct_vs_transitive", "best_eap_variant",
                "discovery_confirmation_induced_spearman") for case in records(e322["case_results"])}},
        "e32_3": {"evidence_status": "registered_sparse_node_cores_passed_3_of_3_selector_advantage_failed_panel",
            **pick(e323, "schema_version", "verdict", "ground_truth_used_for_cohort_selection",
                   "ground_truth_used_for_within_case_selection_or_primary_scoring",
                   "actual_query_accounting"),
            "discovery_examples_per_case": 64, "confirmation_examples_per_case": 128,
            "cases": core_cases},
        "e32_4": {"evidence_status": "registered_sparse_edge_set_generalization_failed",
            **pick(e324, "schema_version", "panel_verdict", "ground_truth_used_for_selection_or_gates",
                   "actual_query_accounting"),
            "discovery_examples_per_case": 64, "confirmation_examples_per_case": 128,
            "cases": edge_cases},
        "estimands": {
            "pair_interaction": "J_ij=Y(B+i+j)-Y(B+i)-Y(B+j)+Y(B)",
            "pair_energy": "Mean J_ij^2 across contexts, backgrounds, positions/classes where applicable; square before averaging.",
            "e31_response": "Unclipped endpoint-normalized donor-minus-base next-token logit-margin shift.",
            "e32_response": "Class-centered output logits at non-BOS positions; fixed benchmark class vector, not vocabulary KL.",
            "necessity_projection": "sum[(Yclean-Yselected)*g]/sum[g^2]",
            "sufficiency_projection": "sum[(Ycomplement-Yallcorrupt)*g]/sum[g^2]",
            "necessity_residual_error": "sum[(Yselected-Yallcorrupt)^2]/sum[g^2]",
            "sufficiency_residual_error": "sum[(Ycomplement-Yclean)^2]/sum[g^2]",
            "gap_definition": "g=Yclean-Yallcorrupt; allcorrupt patches every internal node/edge in the declared U. Inputs/embeddings are boundary conditions.",
            "necessity_loss_recovery": "sum[CE(Yselected)-CE(Yclean)]/sum[CE(Yallcorrupt)-CE(Yclean)]",
            "sufficiency_loss_recovery": "sum[CE(Yallcorrupt)-CE(Ycomplement)]/sum[CE(Yallcorrupt)-CE(Yclean)]",
            "e32_4_joint_score": "min(necessity_projection,sufficiency_projection,1-necessity_residual,1-sufficiency_residual,necessity_loss_recovery,sufficiency_loss_recovery); unclipped",
            "critical_e32_4_denominator_warning": "within_core_fallback uses induced-core normalization and explicitly is not used for primary gates; primary all-U normalization must be used in paper claims.",
            "direct_wiring_scope": "Official InterpBench edges projected to whole heads/MLPs or Q/K/V fine coordinates; not exhaustive ground truth for arbitrary trained transformers.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Compare existing output to deterministic replay; never write.")
    args = parser.parse_args()
    output = Path(__file__).with_name("scalar_manifest.json")
    manifest = build_manifest()
    encoded = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.check:
        if output.read_text() != encoded:
            raise SystemExit("Evidence manifest differs from source replay")
        print("PASS: all source hashes and extracted metrics replay exactly; no model calls")
    else:
        output.write_text(encoded)
        print(f"Wrote {output}: {len(encoded):,} bytes; no model calls")


if __name__ == "__main__":
    main()
