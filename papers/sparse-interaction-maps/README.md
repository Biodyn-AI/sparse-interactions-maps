# Sparse interaction maps: manuscript and evidence

This directory contains a conference-neutral research draft, editable LaTeX,
verified references, source-linked results, and reproducible scientific figures.
The draft separates pair-score reproducibility, endpoint-factor structure,
reference-edge localization, and joint-set faithfulness. It does not claim a
superior discovery algorithm or a universal reachability estimator.

## Read and build

The rendered draft is at `../../output/pdf/sparse-interaction-maps-draft.pdf`.
From the repository root, build it with:

```sh
.venv/bin/python papers/sparse-interaction-maps/build_paper.py
```

The build needs Python with NumPy and Matplotlib, plus `pdflatex` and `bibtex`.
It verifies the elementary examples and historical source hashes, regenerates
all three figures, and compiles the bibliography and paper. It does not execute
models or overwrite registrations, model responses, or scientific score files.
Intermediate LaTeX files stay in `build/`.

## Evidence and new study

- `evidence/scalar_manifest.json` contains historical scalar results and source hashes.
- `evidence/extract_evidence.py --check` independently re-extracts that manifest.
- `evidence/check_evidence.py` checks six important claim and provenance distinctions.
- `evidence/figure_source_data.json` records exact values used in empirical figures.
- `evidence/literature_audit.json` documents primary sources and novelty boundaries.
- `verify_examples.py` checks the two illustrative mathematical constructions.

The new E33 study was frozen before intervention responses and ran all three
cases (63, 103, 82): 64 discovery and 128 confirmation input pairs per case,
84,864 model examples, and 156 physical forward calls. Its acquisition is
separate from the historical E31/E32 artifacts:

- `../../notes/E33-SPARSE-INTERACTION-DIAGNOSTICS-REGISTRATION.json`
- `../../runs/e33_sparse_interaction_diagnostics_v1/summary.json`
- `../../experiments/e33_sparse_interaction_diagnostics/study.py`
- `../../tests/test_e33_sparse_interaction_diagnostics.py`

The new results strengthen the reproducibility-versus-structure distinction,
while refuting any universal claim that these maps are sparse, rank one, or
confined to the reference reachability closure. The manuscript retains all
three models, including its structural control and failed thresholds.

## Scope of this draft

Historical primary failures remain failures. Post-hoc analyses are labeled.
The E33 cases were outcome-unseen for the new intervention measurements, not
blindly sampled from the benchmark. Its reference labels and feasibility
history were available at design time. Rank-one fits on E33 confirmation maps
are descriptive, not demonstrated cheap discovery methods. Internal independent
replay is not a separate external replication.

Before submission, finalize author information, public release packaging and
licensing, and venue-specific formatting. Further pretrained-model replication
would improve generality but is not asserted as completed here.
