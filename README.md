# Sparse interaction maps

Research code, numerical results, and data for **Repeatable interaction scores do not guarantee a correct circuit**.

[Read the manuscript](https://github.com/Biodyn-AI/sparse-interactions-maps/releases/download/v1.0.0/Manuscript.pdf) · [Download full data](https://github.com/Biodyn-AI/sparse-interactions-maps/releases/tag/v1.0.0) · [Reproduction guide](docs/reproduction.md) · [Claims and evidence](docs/claim_map.md)

## What does this study ask?

Researchers copy the outputs of internal model parts between two runs. If changing two parts together has a different effect from adding their separate effects, the parts have an interaction. We ask whether a short, repeatable list of strong interactions reveals direct connections or a small circuit that still works.

## Main findings

- In GPT-2 Small on a name-completion task, 48 of 432 tested head pairs capture **93.8% of interaction strength on new prompts**. This is not 93.8% of task performance. Choosing heads by individual effects reproduces the tested block interactions better overall.
- In one small model with a known task circuit, the ten strongest pairs contain **five direct links and five pairs connected through intermediate parts**. A simple product of fitted part scores explains most of the pair-score table.
- Three further models give highly repeatable rankings, but two assign about **44-54% of total score to pairs with no path in the known task circuit**. This does not mean those pairs have no physical path in the full transformer.
- In a separate test that keeps a small connection set while replacing the rest, **one of three models passes every specified criterion**. Earlier success at finding useful parts does not certify a working retained circuit.

The contribution is a set of measured distinctions and diagnostic checks, not a new universal causal theorem or a better circuit-discovery algorithm. Negative results are retained. The selected models and tasks do not establish how common these patterns are in large language models.

## Quick start: no data download or model run

```sh
git clone https://github.com/Biodyn-AI/sparse-interactions-maps.git
cd sparse-interactions-maps
python3 scripts/check_repository.py
python3 -m unittest discover -s tests_release -v
```

These standard-library checks validate hashes of the included original files and the declared archive mapping. They do **not** recreate model responses or establish independent experimental replication.

For numerical unit tests, use Python 3.11 and the pinned analysis environment:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/replay.txt
python -m pytest tests/test_e33_sparse_interaction_diagnostics.py tests/test_e32_interpbench_scoring.py tests/test_e32_3_scoring.py tests/test_e32_4_scoring.py
```

## Reanalyse saved model responses

The full versioned release archive is approximately **1.14 GB compressed**. It is a release asset, not a Git-tracked file. Downloading is opt-in:

```sh
python scripts/download_data.py
python scripts/replay.py --data-root data/full
```

The downloader verifies the fixed SHA-256 before extraction. Replay verifies every archived file before loading trusted PyTorch arrays, re-extracts historical scalar claims and figure values, and reconstructs all six E33 pair tables plus their ranking/bootstrap analyses. **No model or network calls occur during replay.** Replay uses the archived declared reference labels; it does not independently fetch and reconstruct the upstream graphs. See [exact scope and instructions](docs/reproduction.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `experiments/` | Unchanged intervention, acquisition, selection, and scoring modules; supporting imports are included |
| `tests/` | Original numerical/protocol tests; CI runs an explicit lightweight subset |
| `notes/` | Original plans, repair records, and relevant result notes |
| `runs/` | Original JSON score, screening, execution, and provenance records; large arrays live in the release |
| `papers/sparse-interaction-maps/` | Archived manuscript, references, plotted scalars, evidence extraction, and figure code |
| `scripts/`, `tests_release/` | New portable download, integrity, and replay wrappers and their tests |
| `provenance/` | Immutable-file manifest, source-archive manifests, original project configuration, and release checksum |
| `ENVIRONMENT.json`, `UPSTREAM.json` | Acquisition/checking environments and pinned third-party identities |

The original manuscript source is retained as part of the archived scientific record; the release PDF linked above is the current submission version. No journal acceptance or external replication is implied. Some original provenance records contain historical machine paths. Public instructions use relative paths, and portable replay resolves the actual data location without rewriting those records.

## Documentation and reuse

[Methods](docs/methods.md) · [Data dictionary](docs/data_dictionary.md) · [Claim map](docs/claim_map.md) · [Contributing](docs/contributing.md)

Original code is [MIT licensed](LICENSE). Original research data, manuscript, figures, and non-code documentation are [CC BY 4.0](LICENSE_DATA.md). Third-party software/model terms are unchanged; no upstream weights are redistributed. See [third-party rights](THIRD_PARTY_NOTICES.md). AI tools assisted planning, code, analysis checks, figures, and writing; the manuscript discloses their scope. Automated checks are not a substitute for scientific review.

Please use [CITATION.cff](CITATION.cff) to cite this version. There is no DOI or publication claim in this release.
