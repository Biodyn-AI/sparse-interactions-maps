#!/usr/bin/env python3
"""Package exact manuscript data/code without rewriting scientific originals.

The full archive is intended for a persistent data repository. The smaller
S2_Data.zip is a convenient supporting summary, not a replacement for raw data.
No model or network calls are made. Run from anywhere inside the workspace.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import io
import json
import platform
import sys
import zipfile
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "output/plos-one/data"
RUN_DIRECTORIES = (
    "runs/e31_1_prospective_sparse_atlas_v1",
    "runs/e31_1_prospective_sparse_atlas_v2_order_repair",
    "runs/e31_2_mixed_resolution_v1",
    "runs/e31_2_posthoc_node_residuals_v1",
    "runs/e32_1_exact_gt_v1",
    "runs/e32_2_node_first_directed_v1",
    "runs/e32_3_sparse_core_v2_input_repair",
    "runs/e32_4_internal_edge_set_faithfulness_v1",
    "runs/e33_sparse_interaction_diagnostics_v1",
)
CODE_DIRECTORIES = (
    "experiments/e31_sparse_interactions",
    "experiments/e32_interpbench",
    "experiments/e33_sparse_interaction_diagnostics",
)
RELEASE_URL = "https://github.com/Biodyn-AI/sparse-interactions-maps/releases/tag/v1.0.0"
MIT_LICENSE = """MIT License

Copyright (c) 2026 Ihor Kendiukhov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
DATA_LICENSE = """# License for original research data and manuscript

Copyright (c) 2026 Ihor Kendiukhov.

The original research data, manuscript, figures, and non-code research
documentation in this distribution are licensed under the Creative Commons
Attribution 4.0 International license (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/

The legal terms are available at:
https://creativecommons.org/licenses/by/4.0/legalcode

Attribute Ihor Kendiukhov and the work 'Repeatable interaction scores do not
guarantee a correct circuit'; provide the license link and indicate changes.
The accompanying original code is instead licensed under MIT (LICENSE_CODE.txt).

Third-party software, model weights, benchmark/program definitions, and other
third-party material retain their own rights and license conditions. No grant
of third-party rights is implied, and no third-party weights or source trees
are redistributed in these archives. See UPSTREAM.json for provenance.
"""
README = """# Data and code for interaction scores and model circuits

This archive accompanies the paper 'Repeatable interaction scores do not
guarantee a correct circuit'. It preserves the original input panels, model
response arrays, analysis files, registrations, and scientific source code for
the E31.1, E31.2, E32.1--E32.4, and E33 studies reported in that paper.

## Which archive do I need?

Repository_Data_and_Code.zip is the complete data/code deposit. Extract it to
any writable folder. S2_Data.zip is the smaller supporting summary: it contains
the plans, exact scalar tables and plotting data, source code, and documentation,
but NOT the complete response arrays. It cannot independently replay the full
analysis. Do not claim that the small summary alone supplies all underlying data.
The versioned release target for the complete deposit is:
https://github.com/Biodyn-AI/sparse-interactions-maps/releases/tag/v1.0.0
The complete archive is intended to be a release asset, not a Git-tracked blob.
Archive construction does not itself verify that this release is live; consult
the release page for current availability. A future Zenodo DOI is recommended
for long-term preservation but is not asserted to exist.

## Offline verification and reanalysis

Use Python 3.11.17, PyTorch 2.3.1, NumPy 1.26.4, SciPy 1.14.0 and Matplotlib
3.8.2 for the tested offline replay environment (replay-requirements.txt). The
package was assembled in a separate Python 3.12 environment; its versions are
recorded separately in ENVIRONMENT.json and are not acquisition claims. Run:

    python submission_support/replay_data.py

This checks SHA-256 and byte length for every included original file, re-extracts
the historical scalar evidence, checks all empirical figure values and the
directed-connection comparison table, and reconstructs all six E33 pair-score
tables directly from saved output arrays. It then reruns E33's complete
bootstrap/ranking/selection/product-fit analysis, using the declared benchmark
labels saved in the original summary. File hashes, schemas, counts and boolean
verdicts must match exactly. Recomputed floating values use absolute and relative
tolerance 1e-12 for float64 reduction roundoff across PyTorch versions; numerical
agreement is not asserted to be bit-for-bit identity. No model or network calls occur. A
hash-only check, requiring only Python's standard library, is available with
--hashes-only. Do not unpickle .pt files from untrusted sources.

This is reanalysis of stored executions, NOT a fresh acquisition, an external
replication, or a replay of every historical acquisition/scoring pipeline.
Historical absolute paths are retained in immutable provenance records. The
new replay helper resolves all actual files relative to the extraction folder
and ignores only redundant absolute_path fields when comparing scalar manifests.

## Organization and data dictionary

- runs/: exact original arrays and execution/provenance/score records.
- notes/: original plans, repairs, and relevant results notes.
- experiments/: intervention, selection, scoring, and acquisition code.
- src/: only local source files required by those experiment imports, if any.
- tests/: relevant unit tests, plus the release-wrapper tests.
- papers/sparse-interaction-maps/: manuscript source, references, figure source
  code, exact scalar manifest, literature audit, and plotted scalar data.
- submission_support/: new portable verification and packaging scripts.
- DATA_MANIFEST.json: relative path, byte length, SHA-256 of every other item.
- UPSTREAM.json: exact external code/model identities, not downloaded copies.
- ENVIRONMENT.json: package/Python versions without private filesystem paths.

In the E33 atlas .pt dictionaries, scientific_condition_outputs has axes
[condition, input pair, non-start-token position, answer class] and contains
float64 copies of class-centered float32 model outputs; energy_per_example has
axes [candidate pair, input pair] and averages squared four-run differences
across position and class. component_ids and edges identify these axes.
clean_inputs/source_inputs contain the token-ID panels. Discovery has 64 pairs;
confirmation has 128. The source identity, model/code hashes, and screening
panels are preserved in the corresponding registration and eligibility files.

Historical .pt schemas differ across studies; each record contains schema,
condition/edge identifiers and provenance. The exact field definitions and
constructions are in the paired experiment modules and original registrations.
In particular, E31 uses normalized name-answer margins; E32/E33 use answer-class
logits. Their scales are not interchangeable. Model rankings are conditional on
the selected cases and input panels, not estimates for a random model population.

## Fresh model acquisition and external dependencies

No third-party model weights or upstream source trees are redistributed here.
The upstream repositories, commits, checkpoint revision and artifact hashes are
listed in UPSTREAM.json and the original registration files. Reacquisition uses
the archived experiment runners and requires obtaining those dependencies under
their own licenses. InterpBench acquisition used its separate Python 3.11 /
TransformerLens 1.19 environment, not just the root Python 3.12 project stack.
Original write-once runners intentionally refuse to overwrite existing seals or
results; reproduce into new declared output paths, never by editing these records.
Saved analysis replay needs only the lighter analysis dependencies.

## Rights and release status

Copyright (c) 2026 Ihor Kendiukhov. Original code is released under the MIT
license (LICENSE_CODE.txt); original research data, manuscript, figures and
non-code research documentation are released under CC BY 4.0
(LICENSE_DATA.md; https://creativecommons.org/licenses/by/4.0/). The author
has approved these licenses and public redistribution. Third-party software,
models, and benchmark material retain their own rights and license conditions;
these notices do not relicense third-party material.
No access token or secret configuration is intentionally included. Historical
provenance files retain the original local machine paths as scientific records;
review these for acceptable personal metadata before public deposition.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def original_files(folder: Path) -> Iterable[Path]:
    return (path for path in folder.rglob("*") if path.is_file()
            and not any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(folder).parts))


def resolve_module(root: Path, name: str) -> Path | None:
    prefix = Path("src") if name == "bmi" or name.startswith("bmi.") else Path()
    relative = prefix / name.replace(".", "/")
    for candidate in (relative.with_suffix(".py"), relative / "__init__.py"):
        if (root / candidate).is_file():
            return candidate
    return None


def source_closure(root: Path, seeds: set[Path]) -> set[Path]:
    """Follow local Python imports without importing acquisition modules."""
    found = set(seeds)
    pending = list(seeds)
    while pending:
        relative = pending.pop()
        if relative.suffix != ".py":
            continue
        tree = ast.parse((root / relative).read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module] + [node.module + "." + alias.name for alias in node.names]
            for name in names:
                if not (name.startswith("experiments.") or name == "bmi" or name.startswith("bmi.")):
                    continue
                dependency = resolve_module(root, name)
                if dependency is not None and dependency not in found:
                    found.add(dependency)
                    pending.append(dependency)
    # Include existing parent package initializers as actual imports do.
    for relative in tuple(found):
        for parent in relative.parents:
            initializer = parent / "__init__.py"
            if (root / initializer).is_file():
                found.add(initializer)
    return found


def collect_files() -> set[Path]:
    paths = set()
    for folder in RUN_DIRECTORIES:
        paths.update(path.relative_to(ROOT) for path in original_files(ROOT / folder))
    code_seeds = {path.relative_to(ROOT) for folder in CODE_DIRECTORIES
                  for path in (ROOT / folder).glob("*.py") if not path.name.startswith(".")}
    test_seeds = {path.relative_to(ROOT) for path in (ROOT / "tests").glob("test_e3[12]*.py")}
    test_seeds.add(Path("tests/test_e33_sparse_interaction_diagnostics.py"))
    paths.update(source_closure(ROOT, code_seeds | test_seeds))
    paths.add(Path("pyproject.toml"))
    if (ROOT / "conftest.py").is_file():
        paths.add(Path("conftest.py"))
    for pattern in ("E31*.json", "E31*.md", "E32.1*.json", "E32.1*.md", "E32.2*.json", "E32.3*.json", "E32.4*.json", "E33-SPARSE*.json"):
        paths.update(path.relative_to(ROOT) for path in (ROOT / "notes").glob(pattern) if not path.name.startswith("."))
    paper = ROOT / "papers/sparse-interaction-maps"
    for filename in ("README.md", "manuscript.tex", "references.bib", "build_paper.py", "verify_examples.py", "generate_empirical_figures.py"):
        paths.add((paper / filename).relative_to(ROOT))
    for folder in ("evidence", "figures"):
        paths.update(path.relative_to(ROOT) for path in original_files(paper / folder))
    return paths


def environment() -> dict:
    versions = {}
    for name in ("torch", "numpy", "scipy", "matplotlib", "einops", "transformers", "safetensors", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    registration = json.loads((ROOT / "notes/E33-SPARSE-INTERACTION-DIAGNOSTICS-REGISTRATION.json").read_text())
    original = registration["runtime"]
    return {"packaging_environment_not_acquisition": {"python": platform.python_version(), "packages": versions},
            "tested_offline_replay_environment": {"python": "3.11.17", "packages": {"torch": "2.3.1", "numpy": "1.26.4", "scipy": "1.14.0", "matplotlib": "3.8.2"}},
            "e33_acquisition": {key: value for key, value in original.items() if key != "python_executable"},
            "other_acquisition_environments": "See original registrations for exact versions; root pyproject.toml is not the complete InterpBench environment."}


def upstream() -> dict:
    return {
        "weights_redistributed": False,
        "interpbench_code": {"url": "https://github.com/FlyingPumba/InterpBench", "revision": "220791e4a9cd29957ed5edfa445e7c09152d7298"},
        "interpbench_models": {"url": "https://huggingface.co/cybershiptrooper/InterpBench", "revision": "a1242a84f8a4d07d1be9a8f5ec710415198016b2"},
        "tracr": {"url": "https://github.com/FlyingPumba/tracr", "revision": "98f4910e3966805c83d80eb087c7a4e3328171c6"},
        "iit": {"url": "https://github.com/cybershiptrooper/iit", "revision": "ed44cbfa255b543bb2d4da3d193ae97c9715c4f7"},
        "auto_circuit": {"url": "https://github.com/FlyingPumba/auto-circuit", "revision": "0b74956bbe472cf961deb8ae0ef574f725daca96"},
        "gpt2": {"url": "https://huggingface.co/openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e"},
        "license_notice": "Original code: MIT, copyright2026 Ihor Kendiukhov. Original data/manuscript/figures/non-code research documentation: CC BY4.0. Public redistribution and these licenses approved by the author. Third-party software and model files remain governed by their respective upstream terms. No upstream source or checkpoint is relicensed or redistributed.",
        "release_target": RELEASE_URL,
        "release_availability_verified_by_packager": False,
        "graph_verification_limit": "Offline replay uses archived declared reference labels. Independent comparison with original graph files requires fetching pinned upstream artifacts and checking the hashes recorded in registrations.",
    }


def encoded_json(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_archive(target: Path, paths: set[Path], extra: dict[str, bytes], scope: str) -> dict:
    records = [{"path": str(path), "size_bytes": (ROOT / path).stat().st_size, "sha256": sha256(ROOT / path)}
               for path in sorted(paths)]
    records.extend({"path": name, "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                   for name, content in sorted(extra.items()))
    manifest = {"schema_version": "sparse_interaction_publication_data_v1", "scope": scope,
                "raw_response_arrays_included": scope == "complete_data_and_code", "files": records}
    # Write every original file byte-for-byte. Hashes are independent of ZIP
    # metadata and stay valid after extracting on a different machine.
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(paths):
            archive.write(ROOT / path, str(path))
        for name, content in sorted(extra.items()):
            archive.writestr(name, content)
        archive.writestr("DATA_MANIFEST.json", encoded_json(manifest))
    return {"filename": target.name, "size_bytes": target.stat().st_size, "sha256": sha256(target),
            "uncompressed_payload_bytes": sum(item["size_bytes"] for item in records), "file_count": len(records),
            "scope": scope, "under_20_mb": target.stat().st_size < 20_000_000}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-only", action="store_true")
    options = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_paths = collect_files()
    extra = {
        "README_DATA.md": README.encode(),
        "ENVIRONMENT.json": encoded_json(environment()),
        "UPSTREAM.json": encoded_json(upstream()),
        "LICENSE_CODE.txt": MIT_LICENSE.encode(),
        "LICENSE_DATA.md": DATA_LICENSE.encode(),
        "replay-requirements.txt": b"# Tested with Python 3.11.17. Analysis only; not full model-acquisition dependencies.\ntorch==2.3.1\nnumpy==1.26.4\nscipy==1.14.0\nmatplotlib==3.8.2\n",
    }
    for filename in ("replay_data.py", "package_data.py", "test_data_package.py"):
        extra["submission_support/" + filename] = Path(__file__).with_name(filename).read_bytes()
    # Keep the supporting summary small, with source code in its own upload.
    # Both extracts explicitly require the complete deposit for raw reanalysis.
    summary_paths = {path for path in source_paths if path.suffix in (".json", ".md")}
    code_paths = {path for path in source_paths if path.suffix in (".py", ".toml", ".tex", ".bib")}
    records = [build_archive(OUTPUT / "S2_Data.zip", summary_paths, extra, "supporting_summary_without_raw_response_arrays")]
    print(f"Built S2_Data.zip: {records[0]['size_bytes']:,} bytes", flush=True)
    records.append(build_archive(OUTPUT / "S1_Code.zip", code_paths, extra, "source_code_requires_complete_data_archive"))
    print(f"Built S1_Code.zip: {records[-1]['size_bytes']:,} bytes", flush=True)
    if not options.summary_only:
        records.append(build_archive(OUTPUT / "Repository_Data_and_Code.zip", source_paths, extra, "complete_data_and_code"))
        print(f"Built Repository_Data_and_Code.zip: {records[-1]['size_bytes']:,} bytes", flush=True)
    inventory = {"archives": records, "public_deposition_complete_at_packaging": False, "license_approved": True,
                 "licenses": {"original_code": "MIT", "original_data_manuscript_figures_and_noncode_documentation": "CC-BY-4.0", "third_party": "Upstream rights retained"},
                 "release_target": RELEASE_URL,
                 "recommendation": "Upload S2_Data.zip and S1_Code.zip as supporting information. Upload the complete Repository_Data_and_Code.zip as a versioned GitHub release asset at the recorded release target, verify public download access, then use that release URL in the manuscript's data-availability statement. Zenodo archiving/DOI remains an optional future preservation step.",
                 "underlying_scientific_files_modified": False}
    (OUTPUT / "DATA_PACKAGE_INVENTORY.json").write_bytes(encoded_json(inventory))
    (OUTPUT / "README_DATA.md").write_text(README)
    (OUTPUT / "ENVIRONMENT.json").write_bytes(extra["ENVIRONMENT.json"])
    (OUTPUT / "UPSTREAM.json").write_bytes(extra["UPSTREAM.json"])
    (OUTPUT / "LICENSE_CODE.txt").write_bytes(extra["LICENSE_CODE.txt"])
    (OUTPUT / "LICENSE_DATA.md").write_bytes(extra["LICENSE_DATA.md"])


if __name__ == "__main__":
    main()
