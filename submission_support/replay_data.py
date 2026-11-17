#!/usr/bin/env python3
"""Verify the relocated research archive and replay stored analyses offline.

Run from an extracted Repository_Data_and_Code.zip directory:
    python submission_support/replay_data.py

This does not load a model, download weights, rerun training, or change any
scientific file. Only load this package's trusted PyTorch files after their
SHA-256 hashes have passed. Untrusted .pt files can contain executable pickle.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Ignore only the redundant absolute machine path, never scientific values."""
    portable = copy.deepcopy(manifest)
    for source in portable["sources"].values():
        source.pop("absolute_path", None)
    return portable


def verify_files(root: Path, manifest: dict[str, Any]) -> int:
    for record in manifest["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe archive path: {relative}")
        source = root / relative
        if not source.is_file() or source.stat().st_size != record["size_bytes"]:
            raise ValueError(f"missing or size-changed file: {relative}")
        if sha256(source) != record["sha256"]:
            raise ValueError(f"changed file: {relative}")
    return len(manifest["files"])


def load_script(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def deny_network(*args: Any, **kwargs: Any):
    raise RuntimeError("network access is forbidden during offline replay")


def assert_equivalent(actual: Any, expected: Any, location: str = "analysis") -> None:
    """Keep structure/counts exact; allow only float64 cross-version roundoff."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"different fields at {location}")
        for key in actual:
            assert_equivalent(actual[key], expected[key], location + "/" + str(key))
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError(f"different lengths at {location}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_equivalent(left, right, location + "/" + str(index))
    elif isinstance(actual, float) and isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"different floating value at {location}: {actual} vs {expected}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"different exact value at {location}: {actual} vs {expected}")


def replay_science(root: Path) -> dict[str, Any]:
    """Reuse unchanged score code while resolving data relative to the archive."""
    sys.path.insert(0, str(root))
    import torch

    torch.set_num_threads(2)
    # Prevent accidental network access even if a future import adds a client.
    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    paper = root / "papers/sparse-interaction-maps"
    expected_manifest = json.loads((paper / "evidence/scalar_manifest.json").read_text())
    extraction = load_script("publication_evidence_extraction", paper / "evidence/extract_evidence.py")
    extraction.ROOT = root
    replayed_manifest = extraction.build_manifest()
    assert_equivalent(portable_manifest(replayed_manifest), portable_manifest(expected_manifest), "historical_evidence")

    # Figure collection originally trusted absolute paths saved in the local
    # evidence file. Relocate only that redundant field in an in-memory copy.
    plotting = load_script("publication_plot_data", paper / "generate_empirical_figures.py")
    relocated = copy.deepcopy(expected_manifest)
    for source in relocated["sources"].values():
        source["absolute_path"] = str(root / source["relative_path"])
    fresh = json.loads((root / "runs/e33_sparse_interaction_diagnostics_v1/summary.json").read_text())
    plotted = plotting.verify_and_collect(relocated, fresh)
    expected_plotted = json.loads((paper / "evidence/figure_source_data.json").read_text())
    plotted.pop("sources")
    expected_plotted.pop("sources")
    assert_equivalent(plotted, expected_plotted, "figures_and_directed_table")

    from experiments.e33_sparse_interaction_diagnostics import study
    from experiments.e32_interpbench.component_patching import build_component_atlas, build_patch_units, vector_pair_interactions

    fresh_results = []
    original_loader = study.load_exact_ground_truth
    try:
        for expected_case in fresh["cases"]:
            case_id = expected_case["case_id"]
            saved = {}
            for split in ("discovery", "confirmation"):
                path = root / f"runs/e33_sparse_interaction_diagnostics_v1/case{case_id}_{split}_atlas.pt"
                record = torch.load(path, map_location="cpu", weights_only=False)
                model = study.CASE_MANIFESTS[case_id]
                plan = build_component_atlas(build_patch_units(model.n_layers, model.n_heads), valid_directed_edges=record["edges"])
                outputs = record["scientific_condition_outputs"]
                interactions = vector_pair_interactions(
                    empty_output=outputs[0],
                    singleton_outputs=outputs[list(plan.singleton_condition_indices)],
                    pair_outputs=outputs[list(plan.pair_condition_indices)],
                    pair_component_indices=plan.pair_component_indices,
                    center=False,
                )
                energy = interactions.square().mean(dim=(2, 3))
                torch.testing.assert_close(energy, record["energy_per_example"], rtol=1e-12, atol=1e-12)
                record["energy_per_example"] = energy
                saved[split] = record
            # The archive contains exact declared benchmark labels, with the
            # upstream graph/checkpoint hashes in the original registrations.
            # Independent reconstruction from upstream graph files requires
            # downloading those pinned public files; no such download occurs here.
            declared_edges = tuple(tuple(edge) for edge in expected_case["direct_edges"])
            study.load_exact_ground_truth = lambda *args: SimpleNamespace(internal_edges=declared_edges)
            recomputed = study.analyze(case_id, saved["discovery"], saved["confirmation"])
            assert_equivalent(json.loads(json.dumps(recomputed)), expected_case, f"e33_case{case_id}")
            fresh_results.append({"case_id": case_id, "raw_response_tables_replayed": 2, "all_analysis_fields_match": True})
    finally:
        study.load_exact_ground_truth = original_loader
    return {
        "historical_scalar_extraction_matches": True,
        "all_empirical_figure_and_directed_table_values_match": True,
        "fresh_cases": fresh_results,
        "model_calls": 0,
        "network_calls": 0,
        "floating_replay_tolerance": {"relative": 1e-12, "absolute": 1e-12},
        "counts_schemas_booleans_and_hashes_exact": True,
        "scope": "Stored outputs reanalysed; not a new model acquisition or external replication. Historical summaries verified/extracted, not all historical raw scoring pipelines rerun.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--hashes-only", action="store_true")
    options = parser.parse_args()
    root = options.root.resolve()
    manifest = json.loads((root / "DATA_MANIFEST.json").read_text())
    count = verify_files(root, manifest)
    report = {"verified_files": count, "all_file_hashes_match": True}
    if not options.hashes_only:
        report.update(replay_science(root))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
