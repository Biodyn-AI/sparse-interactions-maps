# ruff: noqa: EXE002
"""Verify and behaviorally smoke-test the pinned InterpBench panel.

Run this module with the separate Python-3.11 InterpBench environment.  It
never downloads artifacts and fails closed on source or model hash drift.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.e32_interpbench.adapter import (
    CASE_MANIFESTS,
    DEFAULT_IIT_DIR,
    DEFAULT_MODEL_ROOT,
    DEFAULT_OFFICIAL_CODE_DIR,
    DEFAULT_TRACR_DIR,
    IIT_REVISION,
    INTERPBENCH_CODE_REVISION,
    INTERPBENCH_MODEL_REVISION,
    PANEL_CASE_IDS,
    TRACR_REVISION,
    behaviorally_eligible_pair,
    canonicalize_official_edges,
    categorical_accuracy,
    generate_official_pair,
    internal_ground_truth_edges,
    load_case_ground_truth,
    load_official_case,
    load_pinned_model,
    official_prepared_universe,
    patchable_internal_universe,
    promoted_ground_truth_internal_nodes,
    promoted_ground_truth_nodes,
    verify_case_artifacts,
    verify_git_revision,
)


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _shape(tensor: Any) -> list[int]:
    return list(tensor.shape)


def _unique_context_count(inputs: Any) -> int:
    import torch

    return int(torch.unique(inputs, dim=0).shape[0])


def _model_shape(model: Any) -> dict[str, int]:
    fields = (
        "n_layers",
        "n_heads",
        "d_model",
        "d_head",
        "d_mlp",
        "n_ctx",
        "d_vocab",
        "d_vocab_out",
    )
    return {field: int(getattr(model.cfg, field)) for field in fields}


def _expected_model_shape(case_id: int) -> dict[str, int]:
    manifest = CASE_MANIFESTS[case_id]
    return {
        field: int(getattr(manifest, field))
        for field in (
            "n_layers",
            "n_heads",
            "d_model",
            "d_head",
            "d_mlp",
            "n_ctx",
            "d_vocab",
            "d_vocab_out",
        )
    }


def smoke_case(
    *,
    case_id: int,
    model_root: Path,
    official_code_dir: Path,
    sample_count: int,
    eligibility_candidates: int,
    clean_seed: int,
    corrupt_seed: int,
    device: str,
) -> dict[str, Any]:
    """Run one complete, bounded verification and return JSON-ready facts."""
    import torch

    manifest = CASE_MANIFESTS[case_id]
    artifact_hashes = verify_case_artifacts(model_root, case_id)
    raw_edges = load_case_ground_truth(model_root, case_id)
    promoted_edges = canonicalize_official_edges(raw_edges)
    internal_edges = internal_ground_truth_edges(raw_edges)
    promoted_nodes = promoted_ground_truth_nodes(raw_edges)
    internal_nodes = promoted_ground_truth_internal_nodes(raw_edges)
    official_nodes, official_edges = official_prepared_universe(
        manifest.n_layers, manifest.n_heads
    )
    patchable_nodes, patchable_edges = patchable_internal_universe(
        manifest.n_layers, manifest.n_heads
    )
    if not set(promoted_edges).issubset(official_edges):
        raise RuntimeError(f"case {case_id} promoted GT escapes official universe")
    if not set(internal_edges).issubset(patchable_edges):
        raise RuntimeError(f"case {case_id} internal GT escapes patchable universe")
    if not set(internal_nodes).issubset(patchable_nodes):
        raise RuntimeError(
            f"case {case_id} internal node GT escapes patchable universe"
        )

    model = load_pinned_model(model_root, case_id, device=device)
    observed_shape = _model_shape(model)
    expected_shape = _expected_model_shape(case_id)
    if observed_shape != expected_shape:
        raise RuntimeError(
            f"case {case_id} config drift: expected {expected_shape}, "
            f"observed {observed_shape}"
        )
    case = load_official_case(case_id, official_code_dir)
    observed_output_kind = "categorical" if case.is_categorical() else "numerical"
    if observed_output_kind != manifest.output_kind:
        raise RuntimeError(
            f"case {case_id} output-kind drift: expected {manifest.output_kind}, "
            f"observed {observed_output_kind}"
        )
    pair = generate_official_pair(
        case,
        sample_count=sample_count,
        clean_seed=clean_seed,
        corrupt_seed=corrupt_seed,
    )
    with torch.inference_mode():
        clean_logits = model(pair.clean_inputs.to(device)).detach().cpu()
        corrupt_logits = model(pair.corrupt_inputs.to(device)).detach().cpu()
    clean_accuracy = categorical_accuracy(
        clean_logits, pair.clean_targets, exclude_bos=True
    )
    corrupt_accuracy = categorical_accuracy(
        corrupt_logits, pair.corrupt_targets, exclude_bos=True
    )
    eligible = behaviorally_eligible_pair(
        model,
        case,
        required_count=sample_count,
        candidate_count=eligibility_candidates,
        clean_seed=clean_seed,
        corrupt_seed=corrupt_seed,
        device=device,
    )
    return {
        "case_id": case_id,
        "task": manifest.task,
        "role": manifest.role,
        "output_kind": observed_output_kind,
        "artifact_sha256": artifact_hashes,
        "model_shape": observed_shape,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "ground_truth": {
            "raw_edge_count": len(raw_edges),
            "promoted_node_count_including_boundaries": len(promoted_nodes),
            "promoted_edge_count_including_boundaries": len(promoted_edges),
            "patchable_internal_node_count": len(internal_nodes),
            "patchable_internal_edge_count": len(internal_edges),
            "patchable_internal_nodes": list(internal_nodes),
            "patchable_internal_edges": [list(edge) for edge in internal_edges],
        },
        "universes": {
            "official_prepared_nodes_including_boundaries": len(official_nodes),
            "official_prepared_edges_including_boundaries": len(official_edges),
            "patchable_internal_nodes": len(patchable_nodes),
            "patchable_internal_edges": len(patchable_edges),
        },
        "clean": {
            "input_shape": _shape(pair.clean_inputs),
            "target_shape": _shape(pair.clean_targets),
            "output_shape": _shape(clean_logits),
            "unique_contexts": _unique_context_count(pair.clean_inputs),
            "accuracy_excluding_bos": asdict(clean_accuracy),
        },
        "corrupt_independent_baseline": {
            "input_shape": _shape(pair.corrupt_inputs),
            "target_shape": _shape(pair.corrupt_targets),
            "output_shape": _shape(corrupt_logits),
            "unique_contexts": _unique_context_count(pair.corrupt_inputs),
            "accuracy_excluding_bos": asdict(corrupt_accuracy),
        },
        "behavioral_eligibility": {
            "status": (
                "pre-intervention filtering; not random full-distribution sampling"
            ),
            "candidate_count": eligible.candidate_count,
            "eligible_candidate_count": eligible.eligible_candidate_count,
            "selected_count": len(eligible.selected_candidate_indices),
            "clean_incorrect_contexts": eligible.clean_incorrect_contexts,
            "source_incorrect_contexts": eligible.source_incorrect_contexts,
            "either_incorrect_contexts": eligible.either_incorrect_contexts,
            "selected_candidate_indices": list(eligible.selected_candidate_indices),
            "eligibility_physical_forwards": eligible.eligibility_physical_forwards,
            "eligibility_model_examples": eligible.eligibility_model_examples,
            "output_kind": eligible.output_kind,
        },
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    sources = {
        "interpbench": verify_git_revision(
            args.official_code_dir, INTERPBENCH_CODE_REVISION
        ),
        "tracr": verify_git_revision(args.tracr_dir, TRACR_REVISION),
        "iit": verify_git_revision(args.iit_dir, IIT_REVISION),
    }
    cases = [
        smoke_case(
            case_id=case_id,
            model_root=args.model_root,
            official_code_dir=args.official_code_dir,
            sample_count=args.samples,
            eligibility_candidates=args.eligibility_candidates,
            clean_seed=args.clean_seed,
            corrupt_seed=args.corrupt_seed,
            device=args.device,
        )
        for case_id in args.cases
    ]
    return {
        "schema_version": 2,
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in (
                    "torch",
                    "transformer-lens",
                    "huggingface-hub",
                    "numpy",
                    "jax",
                    "jaxlib",
                    "dm-haiku",
                    "flax",
                    "optax",
                )
            },
        },
        "pinned_sources": sources,
        "pinned_model_revision": INTERPBENCH_MODEL_REVISION,
        "sampling": {
            "sample_count": args.samples,
            "eligibility_candidate_count": args.eligibility_candidates,
            "clean_seed": args.clean_seed,
            "corrupt_seed": args.corrupt_seed,
            "unique_data_requested": True,
            "corrupt_semantics": (
                "official default: an independent clean sample with another seed; "
                "not a semantic corruption or row pairing"
            ),
            "accuracy_semantics": "categorical argmax, excluding unspecified BOS at position 0",
        },
        "cases": cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--official-code-dir", type=Path, default=DEFAULT_OFFICIAL_CODE_DIR
    )
    parser.add_argument("--tracr-dir", type=Path, default=DEFAULT_TRACR_DIR)
    parser.add_argument("--iit-dir", type=Path, default=DEFAULT_IIT_DIR)
    parser.add_argument("--cases", type=int, nargs="+", default=list(PANEL_CASE_IDS))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--eligibility-candidates", type=int, default=80)
    parser.add_argument("--clean-seed", type=int, default=42_032)
    parser.add_argument("--corrupt-seed", type=int, default=42_033)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    unknown_cases = set(args.cases) - set(PANEL_CASE_IDS)
    if unknown_cases:
        raise ValueError(
            f"smoke panel is frozen to {PANEL_CASE_IDS}; got {sorted(unknown_cases)}"
        )
    report = run_smoke(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
