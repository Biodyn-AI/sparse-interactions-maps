# Contributing and reporting problems

Issues about score mathematics, replay failures, provenance, or unclear documentation are welcome. Include the tag/commit, exact command, Python/package versions, expected behavior, and a minimal example. Do not post credentials or unrelated private paths.

## Preserve the scientific record

- Do not edit original `runs/`, `notes/`, `experiments/`, or archived paper/evidence files to make a check pass. They are covered by immutable hashes. Propose corrected code in a new version or separate run namespace, with a documented comparison to the original.
- Keep discovery/confirmation boundaries, failed tests, tie handling, score normalization, and model-case selection visible.
- A bug fix does not retroactively change which test was planned or whether the original run passed.
- Distinguish the lightweight integrity check, stored-response replay, new model acquisition, and genuine external replication.

Changes to the new wrappers/docs should include tests when behavior changes. Run:

```sh
python scripts/check_repository.py
python -m unittest discover -s tests_release -v
python -m pytest tests/test_e33_sparse_interaction_diagnostics.py tests/test_e32_interpbench_scoring.py tests/test_e32_3_scoring.py tests/test_e32_4_scoring.py
```

These commands need no model weights or large-data download. Larger experiments are explicitly opt-in and should declare their compute/data budget first. Do not add acquisition jobs or large response downloads to ordinary CI.

Original code contributions use MIT; original data, manuscript, figures, and non-code documentation use CC BY 4.0. Confirm that you can license your contribution and that third-party conditions are preserved. Do not upload upstream model weights or access tokens. Disclose material AI assistance and independently inspect its scientific/code claims; automated checks alone do not establish correctness.
