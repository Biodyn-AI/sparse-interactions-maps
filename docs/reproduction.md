# Reproduction guide

There are three different checks. Do not report a weaker check as a stronger result.

## 1. Verify the Git-tracked files (no installation or download)

From a fresh clone, run:

```sh
python3 scripts/check_repository.py
python3 -m unittest discover -s tests_release -v
```

This verifies immutable source/results against the repository and source-archive manifests, plus safety and claim-boundary tests. Python 3.11 or later is sufficient for these standard-library wrappers. It does not recompute scores from model responses.

## 2. Reanalyse saved responses (pinned CPU analysis environment)

The tested offline environment is Python **3.11.17**, PyTorch **2.3.1**, NumPy **1.26.4**, SciPy **1.14.0**, and Matplotlib **3.8.2**. `requirements/replay.txt` also pins pytest for unit tests. Original acquisition and later packaging environments are recorded separately in `ENVIRONMENT.json`; they are not interchangeable acquisition claims.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/replay.txt
python -m pytest tests/test_e33_sparse_interaction_diagnostics.py tests/test_e32_interpbench_scoring.py tests/test_e32_3_scoring.py tests/test_e32_4_scoring.py
python scripts/download_data.py
python scripts/replay.py --data-root data/full
```

On Windows, activate `.venv\Scripts\activate` instead. The PyTorch CPU wheel can be installed from its official CPU index before installing the requirements. No GPU is needed. CI runs the integrity tests and this explicit numerical-test subset, never a model acquisition or the large data download.

`scripts/download_data.py` fetches the fixed `v1.0.0` release asset, checks both its byte length and SHA-256 from `provenance/release.json`, and rejects path traversal, duplicate archive members, and symlinks before extraction. Allow approximately 1.14 GB for the ZIP plus its extracted contents; several GB of free disk space and RAM are prudent. It refuses to overwrite an existing `data/full` folder. A repeated verified download can be reused with `--download-only`. A failed partial download is deliberately retained with a `.part` suffix; inspect it before explicitly removing that one file and retrying.

If downloaded manually, verify the same checksum before extracting. Then point `--data-root` at the folder containing `DATA_MANIFEST.json`, `runs/`, and `submission_support/`. Never use this replay command on untrusted PyTorch/pickle data: hashes establish identity with a trusted manifest, not that an unknown publisher is trustworthy.

The replay command verifies **all original data/code file hashes before unpickling**, then:

1. Re-extracts the historical scalar evidence from the archived results.
2. Checks empirical figure values and the directed-connection comparison table.
3. Reconstructs six E33 interaction tables from the stored four-condition responses.
4. Repeats E33 ranking, bootstrap, selected-support, and fitted-product calculations.

Counts, schemas, hashes, and boolean results must agree exactly. Float64 results allow absolute and relative error `1e-12` for reduction roundoff. Replay uses the archived declared reference links, not newly fetched upstream graph files. It makes zero model/network calls and is **not** an external replication or a rerun of every historical scoring pipeline. `--hashes-only` is a standard-library check of the full archive without numerical replay.

To render the scientific figures from the tracked audited plotting values:

```sh
python scripts/plot_figures.py
```

This writes only to `output/figures/`. It does not rewrite the original evidence files or claim to validate raw responses; use full replay for that. The Scientific Reports-format manuscript and Supplementary Information are release assets in `v1.1.0`. The numerical data archive, checksums, and replay inputs remain pinned to `v1.0.0` without changes. The historical manuscript/figure builders are archived for provenance and may contain the original wider-workspace path assumptions; use the portable wrappers for this release.

To reproduce the Scientific Reports figure layout as native-vector PDFs:

```sh
python papers/sparse-interaction-maps/scientific-reports/plot_figures.py --pdf-only
```

This writes to `output/scientific-reports-sparse-interactions/upload/`, checks the immutable plotting-data hashes and chart values, and never calls a model. It uses Arial when installed and otherwise DejaVu Sans; font/layout details can vary by platform, while plotted numerical values remain fixed.

## 3. Collect new model responses (not automatic)

Fresh acquisition is a separate experiment. The archived runners in `experiments/` record the interventions and computation, while original JSON plans in `notes/` and `runs/` record case IDs, seeds, hashes, environments, stopping/repair histories, and output paths. `UPSTREAM.json` pins external repository/model revisions; their own licenses apply. No weights are distributed here.

InterpBench acquisition additionally used its Python 3.11 / TransformerLens 1.19 environment and pinned InterpBench, Tracr, IIT, and auto-circuit dependencies. The lightweight replay requirements deliberately do not install that stack. The original write-once runners refuse to overwrite seals or results. For a new acquisition, create a new declared run and output namespace, record the exact environment and upstream hashes, and compare against the published run without editing it. The release does not claim a single turnkey command that recreates every historical acquisition from scratch.

When reporting a failure, include the release/tag, Python/package versions, exact command, and traceback. Redact credentials and unrelated personal paths. See [contributing](contributing.md).
