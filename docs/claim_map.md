# Claims and their evidence

This table maps the main manuscript claims to saved records. It does not replace raw-response reanalysis.

| Claim | Evidence | Boundary |
| --- | --- | --- |
| 48/432 GPT-2 head pairs capture 93.82% of confirmation score | `evidence/scalar_manifest.json` under the archived paper, `e31_2.score.primary` | Overall planned test failed; this is interaction-score coverage, not retained task accuracy |
| Individual-effect selection reproduces tested block interactions better overall: R² 0.9074 vs 0.8617 | `e31_2.score.reconstruction` in the same manifest | Only the tested parent interactions, not a whole-model retained circuit |
| Fitted part products describe almost all GPT-2 pair-table squared norm | `e31_2` post-hoc residual report and `runs/e31_2_posthoc_node_residuals_v1/report.json` | Later descriptive analysis; it did not reduce the cost of measuring the original table |
| Case 97's ten highest scores contain 5 direct and 5 longer-path pairs | `e32_1.posthoc_reachability` | Added after the main test; one inspected case, not a general path-recovery claim |
| E33 rankings repeat strongly in all 3 cases | `runs/e33_sparse_interaction_diagnostics_v1/summary.json`, `cases[*].discovery_confirmation_spearman` | Fixed selected models and input pairs, not model-population generality |
| E33 case 103/82 scores outside reference paths carry 54.49% / 43.71% of total | Same summary, `class_statistics.unrelated.mass_fraction` | Outside the known task-reference graph, not outside all physical paths or necessarily noise |
| Only 1/3 small connection sets passes every specified requirement | `e32_4.cases[*].gates` in the scalar manifest and original E32.4 records | Does not prove that no suitable small circuit exists |
| Useful part groups found in 3/3, but selector beats required comparisons in only 1/3 | `e32_3.verdict` in the scalar manifest | Existence of a useful group is not superiority of its selection method |

The scalar manifest lives at `papers/sparse-interaction-maps/evidence/scalar_manifest.json`. It stores relative source names and hashes; full-array sources are in the versioned data release. The archive replay reconstructs all six E33 pair tables and complete E33 analyses, plus historical scalar/figure/table extraction. It does not newly acquire models, rerun all historical scoring pipelines, or independently reconstruct benchmark labels from upstream graph files.

Primary failures remain failures. Repair versions, later analyses, and more favorable alternative normalizations are preserved as such. None of the results is a new universal causal principle, a state-of-the-art circuit-discovery benchmark, or a safety guarantee for larger models.
