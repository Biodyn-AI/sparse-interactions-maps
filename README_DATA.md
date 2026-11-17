# Data and code for interaction scores and model circuits

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
