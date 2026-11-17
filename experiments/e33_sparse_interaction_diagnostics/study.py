"""Bounded E33 replication, without changing any sealed E31/E32 source.

``seal`` generates input panels but makes no model calls. ``run`` verifies the
seal, records an attempt before loading models, and runs all three cases. The
public benchmark identities and graph structures were known during design;
only the model responses to these new panels are prospective.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiments.e32_interpbench.adapter import (
    DEFAULT_IIT_DIR,
    DEFAULT_OFFICIAL_CODE_DIR,
    DEFAULT_TRACR_DIR,
    IIT_REVISION,
    INTERPBENCH_CODE_REVISION,
    TRACR_REVISION,
    OfficialDataPair,
    generate_official_pair,
    sha256_file,
    verify_git_revision,
)
from experiments.e32_interpbench.component_patching import (
    ComponentPatchRunner,
    build_component_atlas,
    build_patch_units,
    class_center_logits,
    vector_pair_interactions,
)
from experiments.e32_interpbench.e32_5_cohort import (
    CASE_MANIFESTS,
    DEFAULT_E32_5_MODEL_ROOT,
    case_universe,
    load_exact_ground_truth,
    load_verified_model,
    load_verified_official_case,
    verify_case_artifacts,
)
from experiments.e32_interpbench.methods import masked_rank_one_endpoint_factor
from experiments.e32_interpbench.scoring import binary_ranking_metrics

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs/e33_sparse_interaction_diagnostics_v1"
REGISTRATION = ROOT / "notes/E33-SPARSE-INTERACTION-DIAGNOSTICS-REGISTRATION.json"
CASE_ORDER = (63, 103, 82)


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_json_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def save_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())


def protocol() -> dict[str, Any]:
    return {
        "schema_version": "e33_sparse_interaction_protocol_v1",
        "case_order": list(CASE_ORDER),
        "all_cases_run_regardless_of_outcomes": True,
        "splits": {
            "discovery": {"required_count": 64, "candidate_count": 96},
            "confirmation": {"required_count": 128, "candidate_count": 192},
        },
        "seed_rule": "6300000+1000*case_id+100*split_index; corrupt=clean+1",
        "smoke_conditions": 16,
        "smoke_batch_size": 8,
        "atlas_condition_batch_size": 32,
        "smoke_max_absolute_error": 1e-4,
        "maximum_model_examples": 85_440,
        "maximum_physical_forwards": 156,
        "maximum_model_examples_note": "conservative ceiling also charging the cached empty atlas row",
        "bootstrap_replicates": 1_000,
        "bootstrap_seed": 330033,
        "top_support_fraction": 0.10,
        "score": "mean class-centered squared logit second difference, positions1:",
        "hypotheses": {
            "score_stability": "discovery-confirmation score Spearman >=0.8 per case",
            "sparse_portability": "discovery top ceil(0.1*E) captures >=0.75 confirmation score mass",
            "endpoint_factor": "one nonnegative source/target factor explains >=0.90 squared score norm",
            "chain_reachability": "in cases103/82 transitive-only mean score > unrelated mean; report class-size-adjusted enrichment and CIs",
            "negative_control": "case63 has no non-direct transitive path; no reachability gap is claimed there",
        },
        "claims": {
            "case_ids_and_public_ground_truth_known_during_design": True,
            "intervention_model_responses_previously_unseen_for_these_cases": True,
            "benchmark_blind": False,
            "parameter_training": False,
            "direct_wiring_recovery_claim": False,
            "whole_circuit_faithfulness_test": False,
            "all_negative_and_heterogeneous_results_retained": True,
            "no_outcome_adaptive_thresholds_or_case_replacement": True,
            "closure_auprc_not_directly_compared_without_prevalence": True,
        },
    }


def source_hashes() -> dict[str, str]:
    # Importing the cohort brings in several old helpers. Bind the entire small
    # package rather than leaving an unnoticed transitive source dependency.
    paths = sorted((ROOT / "experiments/e32_interpbench").glob("*.py"))
    paths += sorted((ROOT / "experiments/e33_sparse_interaction_diagnostics").glob("*.py"))
    return {str(path): sha256_file(path) for path in paths}


def validate_source_hashes(expected: dict[str, str]) -> None:
    for filename, digest in expected.items():
        if sha256_file(filename) != digest:
            raise ValueError(f"source hash drift: {filename}")


def runtime() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("transformer-lens", "numpy", "scipy", "einops")
        },
        "python_executable": sys.executable,
        "torch_threads": 2,
        "device": "cpu",
        "model_native_dtype": "float32",
        "analysis_dtype": "float64",
    }


def rows(tensor: torch.Tensor) -> list[tuple[int, ...]]:
    return [tuple(int(token) for token in row) for row in tensor.tolist()]


def seal() -> None:
    if REGISTRATION.exists() or RUN_DIR.exists():
        raise FileExistsError("E33 seal/run path already exists; no overwrite or reseal")
    upstream = {
        "interpbench": verify_git_revision(DEFAULT_OFFICIAL_CODE_DIR, INTERPBENCH_CODE_REVISION),
        "tracr": verify_git_revision(DEFAULT_TRACR_DIR, TRACR_REVISION),
        "iit": verify_git_revision(DEFAULT_IIT_DIR, IIT_REVISION),
    }
    artifacts = {
        str(case_id): verify_case_artifacts(DEFAULT_E32_5_MODEL_ROOT, case_id)
        for case_id in CASE_ORDER
    }
    panels: dict[str, Any] = {}
    panel_records: dict[str, Any] = {}
    for case_id in CASE_ORDER:
        case = load_verified_official_case(case_id)
        for index, (split, split_values) in enumerate(protocol()["splits"].items()):
            seed = 6_300_000 + 1_000 * case_id + 100 * index
            pair = generate_official_pair(
                case,
                sample_count=split_values["candidate_count"],
                clean_seed=seed,
                corrupt_seed=seed + 1,
            )
            key = f"{case_id}/{split}"
            tensors = asdict(pair)
            panels[key] = tensors
            identity = {
                "clean": rows(pair.clean_inputs),
                "source": rows(pair.corrupt_inputs),
                "clean_targets": rows(pair.clean_targets.argmax(-1)),
                "source_targets": rows(pair.corrupt_targets.argmax(-1)),
            }
            panel_records[key] = {
                "clean_seed": seed,
                "source_seed": seed + 1,
                "candidate_count": len(pair.clean_inputs),
                "identity_sha256": hashlib.sha256(json_bytes(identity)).hexdigest(),
            }
    RUN_DIR.mkdir(parents=True)
    inputs = RUN_DIR / "sealed_input_panels.pt"
    save_once(inputs, panels)
    registration = {
        "schema_version": "e33_sparse_interaction_registration_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol(),
        "sources": source_hashes(),
        "runtime": runtime(),
        "upstream_revisions": upstream,
        "checkpoint_sha256": artifacts,
        "public_case_manifests": {str(case_id): asdict(CASE_MANIFESTS[case_id]) for case_id in CASE_ORDER},
        "panels": panel_records,
        "sealed_input_panels_sha256": sha256_file(inputs),
        "models_loaded_or_called": False,
    }
    write_json_once(REGISTRATION, registration)
    print(json.dumps({"registration": str(REGISTRATION), "sha256": sha256_file(REGISTRATION)}), flush=True)


def transitive_closure(direct: Any) -> set[tuple[str, str]]:
    result = {tuple(edge) for edge in direct}
    while True:
        added = {(a, d) for a, b in result for c, d in result if b == c}
        if any(a == b for a, b in added):
            raise ValueError("ground-truth graph must be acyclic")
        expanded = result | added
        if expanded == result:
            return result
        result = expanded


def class_masks(edges: Any, direct: Any) -> dict[str, torch.Tensor]:
    direct_set = {tuple(edge) for edge in direct}
    closure = transitive_closure(direct)
    return {
        "direct": torch.tensor([tuple(edge) in direct_set for edge in edges]),
        "transitive_only": torch.tensor([tuple(edge) in closure - direct_set for edge in edges]),
        "unrelated": torch.tensor([tuple(edge) not in closure for edge in edges]),
    }


def class_statistics(scores: torch.Tensor, edges: Any, direct: Any) -> dict[str, Any]:
    scores = scores.to(torch.float64)
    if scores.ndim != 1 or len(scores) != len(edges) or not bool(torch.isfinite(scores).all()) or bool((scores < 0).any()):
        raise ValueError("scores must be a finite nonnegative aligned vector")
    total = float(scores.sum())
    result = {}
    for label, mask in class_masks(edges, direct).items():
        count = int(mask.sum())
        mass = float(scores[mask].sum())
        fraction = count / len(edges)
        mass_fraction = mass / total if total > 0 else None
        result[label] = {
            "count": count,
            "class_fraction": fraction,
            "mean_score": float(scores[mask].mean()) if count else None,
            "mass_fraction": mass_fraction,
            "mass_enrichment": mass_fraction / fraction if count and mass_fraction is not None else None,
        }
    return result


def screen(model: Any, pair: OfficialDataPair, required: int, excluded: set[tuple[int, ...]]) -> tuple[OfficialDataPair | None, dict[str, Any]]:
    with torch.inference_mode():
        clean = model(pair.clean_inputs).detach().cpu()
        source = model(pair.corrupt_inputs).detach().cpu()
    clean_ok = (clean.argmax(-1)[:, 1:] == pair.clean_targets.argmax(-1)[:, 1:]).all(-1)
    source_ok = (source.argmax(-1)[:, 1:] == pair.corrupt_targets.argmax(-1)[:, 1:]).all(-1)
    eligible = clean_ok & source_ok
    chosen = []
    used = set(excluded)
    clean_rows, source_rows = rows(pair.clean_inputs), rows(pair.corrupt_inputs)
    for index in range(len(clean_rows)):
        if bool(eligible[index]) and clean_rows[index] != source_rows[index] and clean_rows[index] not in used and source_rows[index] not in used:
            chosen.append(index)
            used.update((clean_rows[index], source_rows[index]))
    record = {
        "candidate_count": len(clean_rows),
        "clean_correct": clean_ok.tolist(),
        "source_correct": source_ok.tolist(),
        "jointly_correct_count": int(eligible.sum()),
        "unique_eligible_count": len(chosen),
        "excluded_prior_raw_row_count": len(excluded),
        "selected_indices": chosen[:required],
        "required_count": required,
        "passed": len(chosen) >= required,
        "selection": "first_jointly_correct_unique_rows; confirmation_excludes_all_discovery_raw_rows",
        "physical_forwards": 2,
        "model_examples": 2 * len(clean_rows),
    }
    if len(chosen) < required:
        return None, record
    index_tensor = torch.tensor(chosen[:required], dtype=torch.long)
    return OfficialDataPair(**{name: value.index_select(0, index_tensor) for name, value in asdict(pair).items()}), record


def acquire(case_id: int, model: Any, pair: OfficialDataPair) -> dict[str, Any]:
    manifest = CASE_MANIFESTS[case_id]
    nodes, edges = case_universe(case_id)
    units = build_patch_units(manifest.n_layers, manifest.n_heads)
    runner = ComponentPatchRunner(model, units, condition_batch_size=32, valid_directed_edges=edges)
    prepared = runner.prepare_pair(pair)
    atlas_plan = build_component_atlas(units, valid_directed_edges=edges)
    smoke = runner.batch_invariance_smoke(prepared, atlas_plan.condition_masks[:16], candidate_condition_batch_size=8, absolute_tolerance=1e-4)
    if not smoke["passed"]:
        raise RuntimeError(f"case{case_id} batch smoke failed: {smoke}")
    atlas = runner.acquire_atlas(prepared)
    scientific = class_center_logits(atlas.condition_outputs)[..., 1:, :].to(torch.float64)
    singleton = scientific[list(atlas_plan.singleton_condition_indices)]
    pair_outputs = scientific[list(atlas_plan.pair_condition_indices)]
    interactions = vector_pair_interactions(
        empty_output=scientific[0], singleton_outputs=singleton, pair_outputs=pair_outputs,
        pair_component_indices=atlas_plan.pair_component_indices, center=False,
    )
    energy_per_example = interactions.square().mean(dim=(2, 3))
    return {
        "case_id": case_id, "component_ids": nodes, "edges": edges,
        "clean_inputs": pair.clean_inputs, "source_inputs": pair.corrupt_inputs,
        "scientific_condition_outputs": scientific,
        "energy_per_example": energy_per_example,
        "singleton_energy": (singleton - scientific[0]).square().mean(dim=(1, 2, 3)),
        "smoke": smoke, "query_accounting": runner.forward_counts(),
    }


def analyze(case_id: int, discovery: dict[str, Any], confirmation: dict[str, Any]) -> dict[str, Any]:
    from scipy.stats import spearmanr

    edges = tuple(confirmation["edges"])
    nodes = tuple(confirmation["component_ids"])
    truth = load_exact_ground_truth(DEFAULT_E32_5_MODEL_ROOT, case_id)
    direct = set(truth.internal_edges)
    closure = transitive_closure(direct)
    discovery_scores = discovery["energy_per_example"].mean(1)
    values = confirmation["energy_per_example"].to(torch.float64)
    scores = values.mean(1)
    direct_labels = [edge in direct for edge in edges]
    closure_labels = [edge in closure for edge in edges]
    ranking = {
        "direct": binary_ranking_metrics(scores=scores.tolist(), labels=direct_labels),
        "closure": binary_ranking_metrics(scores=scores.tolist(), labels=closure_labels),
    }
    for name, labels in (("direct", direct_labels), ("closure", closure_labels)):
        prevalence = sum(labels) / len(labels)
        ranking[name]["chance_auprc"] = prevalence
        ranking[name]["auprc_lift_over_chance"] = ranking[name]["auprc"] / prevalence
    fitted = masked_rank_one_endpoint_factor(node_universe=nodes, edge_universe=edges, edge_scores=scores)
    top_k = max(1, math.ceil(0.10 * len(edges)))
    support = sorted(range(len(edges)), key=lambda i: (-float(discovery_scores[i]), i))[:top_k]
    stats = class_statistics(scores, edges, direct)
    masks = class_masks(edges, direct)
    generator = torch.Generator().manual_seed(330033 + case_id)
    indices = torch.randint(values.shape[1], (1000, values.shape[1]), generator=generator)
    boot_scores = values[:, indices].mean(-1).T
    boot_totals = boot_scores.sum(-1)
    intervals = {}
    for label, mask in masks.items():
        if bool(mask.any()):
            boot_fraction = boot_scores[:, mask].sum(-1) / boot_totals
            intervals[label] = {
                "mass_fraction_95_percentile_interval": torch.quantile(boot_fraction, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist(),
                "mean_score_95_percentile_interval": torch.quantile(boot_scores[:, mask].mean(-1), torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist(),
            }
    transitive_better = None
    if bool(masks["transitive_only"].any()):
        difference = boot_scores[:, masks["transitive_only"]].mean(-1) - boot_scores[:, masks["unrelated"]].mean(-1)
        transitive_better = {
            "point_difference": stats["transitive_only"]["mean_score"] - stats["unrelated"]["mean_score"],
            "difference_95_percentile_interval": torch.quantile(difference, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist(),
        }
    rho = float(spearmanr(discovery_scores.numpy(), scores.numpy()).statistic)
    coverage = float(scores[support].sum() / scores.sum())
    return {
        "case_id": case_id, "task": CASE_MANIFESTS[case_id].task,
        "node_count": len(nodes), "candidate_pair_count": len(edges),
        "direct_edge_count": len(direct), "closure_edge_count": len(closure),
        "discovery_confirmation_spearman": rho,
        "discovery_frozen_top10_percent_count": top_k,
        "discovery_frozen_top10_percent_confirmation_mass": coverage,
        "discovery_frozen_top10_percent_indices": support,
        "ranking": ranking, "class_statistics": stats, "bootstrap": intervals,
        "transitive_minus_unrelated_mean": transitive_better,
        "rank_one_endpoint_factor": {
            "explained_squared_score_norm": fitted.explained_energy_fraction,
            "converged": fitted.converged, "iterations": fitted.iterations,
        },
        "gates": {
            "score_stability": rho >= 0.8,
            "sparse_portability": coverage >= 0.75,
            "endpoint_factor": fitted.explained_energy_fraction >= 0.90,
            "transitive_mean_enriched": transitive_better["point_difference"] > 0 if transitive_better is not None else None,
        },
        "scores": scores.tolist(), "discovery_scores": discovery_scores.tolist(),
        "edges": edges, "direct_edges": sorted(direct), "closure_edges": sorted(closure),
        "interpretation_scope": "known_case_conditional_on_clean_correct_panels; scores_are_symmetric_patch_statistics_not_directed_edge_measurements",
    }


def run() -> None:
    registration = json.loads(REGISTRATION.read_text())
    if registration["protocol"] != protocol() or registration["runtime"] != runtime():
        raise ValueError("runtime/protocol differs from seal")
    validate_source_hashes(registration["sources"])
    inputs = RUN_DIR / "sealed_input_panels.pt"
    if sha256_file(inputs) != registration["sealed_input_panels_sha256"]:
        raise ValueError("sealed input panel drift")
    for case_id in CASE_ORDER:
        if verify_case_artifacts(DEFAULT_E32_5_MODEL_ROOT, case_id) != registration["checkpoint_sha256"][str(case_id)]:
            raise ValueError("checkpoint drift")
    write_json_once(RUN_DIR / "attempt.json", {
        "schema_version": "e33_attempt_v1", "registration_sha256": sha256_file(REGISTRATION),
        "created_utc": datetime.now(timezone.utc).isoformat(), "before_any_model_load_or_call": True,
    })
    torch.set_num_threads(2)
    panels = torch.load(inputs, map_location="cpu")
    cases = []
    acquisitions = {}
    total_examples = 0
    total_forwards = 0
    for case_id in CASE_ORDER:
        print(f"case{case_id}: loading verified model", flush=True)
        model = load_verified_model(case_id)
        collected = {}
        raw_discovery = panels[f"{case_id}/discovery"]
        exclusions = set(rows(raw_discovery["clean_inputs"]) + rows(raw_discovery["corrupt_inputs"]))
        for split, values in protocol()["splits"].items():
            raw = OfficialDataPair(**panels[f"{case_id}/{split}"])
            selected, eligibility = screen(model, raw, values["required_count"], exclusions if split == "confirmation" else set())
            total_examples += eligibility["model_examples"]
            total_forwards += eligibility["physical_forwards"]
            write_json_once(RUN_DIR / f"case{case_id}_{split}_eligibility.json", eligibility)
            if selected is None:
                print(f"case{case_id}/{split}: clean eligibility failed", flush=True)
                continue
            print(f"case{case_id}/{split}: acquiring {len(selected.clean_inputs)} pairs", flush=True)
            acquired = acquire(case_id, model, selected)
            total_examples += acquired["query_accounting"]["model_examples"]
            total_forwards += acquired["query_accounting"]["physical_forwards"]
            target = RUN_DIR / f"case{case_id}_{split}_atlas.pt"
            save_once(target, acquired)
            acquisitions[f"{case_id}/{split}"] = sha256_file(target)
            collected[split] = acquired
        if len(collected) == 2:
            result = analyze(case_id, collected["discovery"], collected["confirmation"])
        else:
            result = {"case_id": case_id, "outcome": "clean_eligibility_failed_no_resampling"}
        write_json_once(RUN_DIR / f"case{case_id}_summary.json", result)
        cases.append(result)
        del model
    if total_examples > protocol()["maximum_model_examples"] or total_forwards > protocol()["maximum_physical_forwards"]:
        raise RuntimeError("query ceiling exceeded")
    summary = {
        "schema_version": "e33_sparse_interaction_summary_v1",
        "registration_sha256": sha256_file(REGISTRATION), "acquisition_sha256": acquisitions,
        "protocol": protocol(), "cases": cases,
        "actual_query_accounting": {"model_examples": total_examples, "physical_forwards": total_forwards},
        "case_pool_is_not_random_benchmark_sample": True,
        "uncertainty_scope": "context_bootstrap_within_each_fixed_checkpoint_not_model_population",
    }
    write_json_once(RUN_DIR / "summary.json", summary)
    print(json.dumps({"summary": str(RUN_DIR / "summary.json"), "model_examples": total_examples, "physical_forwards": total_forwards}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "run"))
    action = parser.parse_args().action
    if action == "seal":
        seal()
    else:
        run()


if __name__ == "__main__":
    main()
