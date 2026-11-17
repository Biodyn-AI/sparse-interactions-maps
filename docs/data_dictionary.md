# Data dictionary

All original records retain their schemas, numerical precision, and provenance. `provenance/REPOSITORY_MANIFEST.json` maps tracked files to their original archive names, lengths, and SHA-256 hashes. Independent source manifests are in `provenance/archives/`. The full release has its own `DATA_MANIFEST.json`, covering both arrays and source records.

## Small tracked records

| File/family | Meaning |
| --- | --- |
| `notes/*REGISTRATION.json` | Execution plans, fixed cases and thresholds, source/model identities, and acquisition metadata |
| `runs/*/*custody*.json`, `*authorization*.json`, `*attempt*.json` | File identities and discovery/confirmation boundaries; not independent experimental replications |
| `runs/*/summary.json`, post-hoc `report.json` | Saved analyses with per-case values and planned-test verdicts |
| `papers/sparse-interaction-maps/evidence/scalar_manifest.json` | Exact manuscript scalar extract plus source hashes and planned/later-analysis status |
| `papers/sparse-interaction-maps/evidence/figure_source_data.json` | Values plotted in empirical figures and the directed-comparison table |
| `runs/e33_sparse_interaction_diagnostics_v1/summary.json` | Complete E33 ranking, scores, labels, bootstrap intervals, and gate results for cases 63, 103, 82 |
| `ENVIRONMENT.json`, `UPSTREAM.json` | Original vs later checking versions, upstream URLs/revisions, and graph-verification limits |

Some historical records contain absolute machine paths. They are immutable provenance, not instructions to use those paths. Portable wrappers resolve actual files relative to the clone or full-data folder.

## Full E33 response records (release asset only)

`runs/e33_sparse_interaction_diagnostics_v1/case{case}_{split}_atlas.pt` is a PyTorch dictionary. Only load the trusted release after checksum/file verification.

- `scientific_condition_outputs`: `[condition, input_pair, non_start_position, answer_class]`; float64 copies of class-centered float32 model outputs. Conditions comprise unchanged, each single part, and each allowed pair.
- `energy_per_example`: `[candidate_pair, input_pair]`; the mean squared four-run difference across output positions and classes.
- `component_ids`: the parts used by the condition/score axes.
- `edges`: allowed earlier/later candidate pairs in the exact scoring order; these are not all known reference links.
- `clean_inputs`, `source_inputs`: token-ID input panels used as recipient/source.

Discovery has 64 input pairs; confirmation has 128 per case. Case IDs 63, 103, and 82 have 10, 15, and 20 parts and 33, 87, and 166 candidate pairs. Screening records preserve the candidate input panels and exclusions. The complete records and their constructors are defined in `experiments/e33_sparse_interaction_diagnostics/study.py` and the paired E32 component-patching code.

The E33 summary distinguishes `direct`, `transitive_only`, and `unrelated` relative to the **known benchmark task circuit**. `unrelated` means no connecting path in that reference; it is not proof of independence in the full physical transformer. `mass_fraction` is a class's share of total interaction score, while `class_fraction` is its share of candidate pairs. Their ratio, `mass_enrichment`, is not itself direct-link recovery.

## Historical arrays

E31 uses normalized differences between two name-answer scores; E32/E33 use answer-class logits. Their score scales are not interchangeable. Historical `.pt` schemas vary and include condition/edge IDs, example axes, and custody metadata. Use the corresponding module and original registration as the authoritative schema. The portable replay checks historical scalar extraction rather than rerunning every historical scorer. No model weights are included.

Intervals describe resampled input pairs for fixed selected models, not a sampled model population. A failed planned conjunction remains failed even when one component metric is favorable. Post-hoc records and alternative normalizations are not promoted to the original primary test.
