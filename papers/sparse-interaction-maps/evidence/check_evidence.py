#!/usr/bin/env python3
"""Independently check the paper extract's critical scientific distinctions."""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path


MANIFEST_PATH = Path(__file__).with_name("scalar_manifest.json")
MANIFEST = json.loads(MANIFEST_PATH.read_text())


class EvidenceChecks(unittest.TestCase):
    def test_source_hashes_are_current(self) -> None:
        for name, record in MANIFEST["sources"].items():
            with self.subTest(source=name):
                digest = hashlib.sha256()
                with Path(record["absolute_path"]).open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                self.assertEqual(digest.hexdigest(), record["sha256"])

    def test_head_selection_positive_does_not_erase_registered_fail(self) -> None:
        score = MANIFEST["e31_2"]["score"]
        self.assertEqual(score["primary_verdict"].lower(), "fail")
        self.assertEqual(score["primary"]["selected_edge_count"], 48)
        self.assertEqual(score["primary"]["total_edge_count"], 432)
        self.assertAlmostEqual(
            score["primary"]["aggregate_confirmation_energy_coverage"],
            0.9381737349487851,
        )
        reconstruction = score["reconstruction"]
        self.assertGreater(
            reconstruction["singleton_selected_composite"]["pooled_r2"],
            reconstruction["pair_selected_composite"]["pooled_r2"],
        )

    def test_reachability_replay_partitions_mass_and_top_ten(self) -> None:
        record = MANIFEST["e32_1"]["posthoc_reachability"]
        self.assertEqual(record["top10_direct_count"], 5)
        self.assertEqual(record["top10_transitive_nonedge_count"], 5)
        self.assertEqual(record["top10_unrelated_count"], 0)
        self.assertAlmostEqual(sum(record[key] for key in (
            "direct_energy_fraction", "transitive_nonedge_energy_fraction",
            "unrelated_energy_fraction",
        )), 1.0)
        self.assertAlmostEqual(record["closure_auprc"], 0.8502222222222221)
        self.assertAlmostEqual(record["rank_one_squared_norm_explained"],
                               0.9874663116026922)
        self.assertEqual(record["model_calls"], 0)

    def test_node_cores_pass_but_selector_does_not_generalize(self) -> None:
        verdict = MANIFEST["e32_3"]["verdict"]
        self.assertTrue(verdict["sparse_core_generalization_pass"])
        self.assertFalse(verdict["singleton_selector_generalization_pass"])
        self.assertEqual(sum(verdict["sparse_core_case_passes"].values()), 3)
        self.assertEqual(sum(verdict["singleton_selector_case_passes"].values()), 1)

    def test_edge_primary_is_not_the_favorable_core_normalization(self) -> None:
        cases = MANIFEST["e32_4"]["cases"]
        primary = cases["129"]["resample_primary"]["faithfulness"]
        fallback = cases["129"]["within_core_fallback_descriptive_not_primary"]
        self.assertAlmostEqual(primary["sufficiency_projection"],
                               0.8871884144786064)
        self.assertFalse(cases["129"]["resample_primary"]["gate_checks"]["pass"])
        self.assertFalse(fallback["used_for_primary_gates"])
        self.assertGreater(fallback["faithfulness"]["sufficiency_projection"], 0.98)
        self.assertEqual(sum(case["gates"]["case_pass"] for case in cases.values()), 1)
        self.assertFalse(MANIFEST["e32_4"]["panel_verdict"][
            "absolute_sparse_set_generalization_pass"])

    def test_manifest_has_no_nonfinite_numbers_or_per_example_arrays(self) -> None:
        def visit(value: object) -> None:
            if isinstance(value, dict):
                self.assertNotIn("per_example", value)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                self.assertLessEqual(len(value), 24)
                for child in value:
                    visit(child)
            elif isinstance(value, float):
                self.assertTrue(math.isfinite(value))
        visit(MANIFEST)


if __name__ == "__main__":
    unittest.main()
