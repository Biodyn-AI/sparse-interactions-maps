# ruff: noqa: EXE002
"""Bounded real smoke for the pinned E32.2 official EAP baselines.

This command validates graph promotion on cases 2/21/97, then runs EAP and
EAP-IG only on two behaviorally eligible case-97 rows.  It never launches the
pair atlas or any development/confirmation experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.e32_interpbench.adapter import (
    DEFAULT_ENV_DIR,
    DEFAULT_IIT_DIR,
    DEFAULT_MODEL_ROOT,
    DEFAULT_OFFICIAL_CODE_DIR,
    DEFAULT_TRACR_DIR,
    IIT_REVISION,
    INTERPBENCH_CODE_REVISION,
    PANEL_CASE_IDS,
    TRACR_REVISION,
    behaviorally_eligible_pair,
    load_official_case,
    load_pinned_model,
    verify_git_revision,
)
from experiments.e32_interpbench.eap_baseline import (
    DEFAULT_AUTO_CIRCUIT_DIR,
    official_eap_run_to_dict,
    run_official_eap_baseline,
    validate_live_auto_circuit_graph,
    verify_pinned_eap_runtime,
)


def _eligibility_report(eligible: Any) -> dict[str, Any]:
    return {
        "status": "behavioral_eligibility_before_any_gradient_or_intervention",
        "candidate_count": eligible.candidate_count,
        "eligible_candidate_count": eligible.eligible_candidate_count,
        "selected_count": len(eligible.selected_candidate_indices),
        "selected_candidate_indices": list(eligible.selected_candidate_indices),
        "clean_incorrect_contexts": eligible.clean_incorrect_contexts,
        "source_incorrect_contexts": eligible.source_incorrect_contexts,
        "either_incorrect_contexts": eligible.either_incorrect_contexts,
        "physical_model_forwards": eligible.eligibility_physical_forwards,
        "model_examples": eligible.eligibility_model_examples,
        "output_kind": eligible.output_kind,
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Run the bounded smoke and return a deterministic JSON-ready report."""
    if args.case_id != 97:
        raise ValueError("the E32.2 real gradient smoke is frozen to primary case 97")
    runtime = verify_pinned_eap_runtime(
        environment_dir=args.environment_dir,
        auto_circuit_dir=args.auto_circuit_dir,
    )
    runtime["sources"].update(
        {
            "interpbench_revision": verify_git_revision(
                args.official_code_dir, INTERPBENCH_CODE_REVISION
            ),
            "tracr_revision": verify_git_revision(args.tracr_dir, TRACR_REVISION),
            "iit_revision": verify_git_revision(args.iit_dir, IIT_REVISION),
        }
    )

    graph_parity: dict[str, Any] = {}
    for case_id in PANEL_CASE_IDS:
        graph_model = load_pinned_model(args.model_root, case_id, device=args.device)
        promoted = validate_live_auto_circuit_graph(
            case_id=case_id, model=graph_model, device=args.device
        )
        graph_parity[str(case_id)] = {
            "fine_edge_count": promoted.fine_edge_count,
            "internal_fine_edge_count": promoted.internal_fine_edge_count,
            "excluded_boundary_fine_edge_count": (
                promoted.excluded_boundary_fine_edge_count
            ),
            "promoted_internal_edge_count": len(promoted.edge_order),
            "collision_histogram": {
                str(count): promoted.collision_counts.count(count)
                for count in sorted(set(promoted.collision_counts))
            },
            "coverage_complete": promoted.coverage_complete,
            "model_forwards": 0,
        }

    case = load_official_case(args.case_id, args.official_code_dir)
    selection_model = load_pinned_model(
        args.model_root, args.case_id, device=args.device
    )
    eligible = behaviorally_eligible_pair(
        selection_model,
        case,
        required_count=args.samples,
        candidate_count=args.eligibility_candidates,
        clean_seed=args.clean_seed,
        corrupt_seed=args.corrupt_seed,
        device=args.device,
    )

    runs: dict[str, Any] = {}
    for method in args.methods:
        fresh_model = load_pinned_model(
            args.model_root, args.case_id, device=args.device
        )
        run = run_official_eap_baseline(
            case_id=args.case_id,
            model=fresh_model,
            case=case,
            eligible_pair=eligible,
            method=method,
            integrated_grad_samples=(
                args.integrated_grad_samples if method == "eap_ig" else None
            ),
            device=args.device,
            environment_dir=args.environment_dir,
            auto_circuit_dir=args.auto_circuit_dir,
        )
        runs[method] = official_eap_run_to_dict(run)

    algorithm_accounting = [run["query_accounting"] for run in runs.values()]
    eligibility = _eligibility_report(eligible)
    total_query_accounting = {
        "physical_model_forwards": eligibility["physical_model_forwards"]
        + sum(item["physical_model_forwards"] for item in algorithm_accounting),
        "model_examples": eligibility["model_examples"]
        + sum(item["model_examples"] for item in algorithm_accounting),
        "gradient_forwards": sum(
            item["gradient_forwards"] for item in algorithm_accounting
        ),
        "backward_passes": sum(
            item["backward_passes"] for item in algorithm_accounting
        ),
        "gradient_examples": sum(
            item["gradient_examples"] for item in algorithm_accounting
        ),
        "includes_behavioral_eligibility_once": True,
        "includes_panel_graph_validation": True,
        "panel_graph_validation_model_forwards": 0,
        "excludes_model_loading_and_artifact_hashing": True,
    }
    return {
        "schema_version": "e32_official_eap_baseline_smoke_v1",
        "scope": "case97_n2_smoke_only_no_pair_atlas_no_development_confirmation",
        "runtime": runtime,
        "graph_promotion_parity": graph_parity,
        "behavioral_eligibility": eligibility,
        "runs": runs,
        "total_query_accounting": total_query_accounting,
        "known_official_semantics": {
            "eap_ig_gradient_reduction": (
                "pinned_mask_gradient_accumulates_k_plus_one_gradients_without_division"
            ),
            "eap_ig_scale_warning": (
                "raw_magnitudes_depend_on_integrated_grad_samples; ranking is unchanged "
                "by the common missing division"
            ),
            "signed_collision_aggregation": (
                "max_raw_score_preserves_default_official_threshold_support"
            ),
            "absolute_collision_aggregation": (
                "max_absolute_score_preserves_abs_value_threshold_support"
            ),
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output"
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", type=int, default=97)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--eligibility-candidates", type=int, default=4)
    parser.add_argument("--clean-seed", type=int, default=42_032)
    parser.add_argument("--corrupt-seed", type=int, default=42_033)
    parser.add_argument(
        "--methods", nargs="+", choices=("eap", "eap_ig"), default=("eap", "eap_ig")
    )
    parser.add_argument("--integrated-grad-samples", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--environment-dir", type=Path, default=DEFAULT_ENV_DIR)
    parser.add_argument(
        "--auto-circuit-dir", type=Path, default=DEFAULT_AUTO_CIRCUIT_DIR
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--official-code-dir", type=Path, default=DEFAULT_OFFICIAL_CODE_DIR
    )
    parser.add_argument("--tracr-dir", type=Path, default=DEFAULT_TRACR_DIR)
    parser.add_argument("--iit-dir", type=Path, default=DEFAULT_IIT_DIR)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_smoke(args)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(args.output)


if __name__ == "__main__":
    main()
