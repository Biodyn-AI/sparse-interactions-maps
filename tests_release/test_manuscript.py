"""Bounded checks for the current manuscript and its online-validation ledger."""

import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "papers/sparse-interaction-maps/scientific-reports"


class ManuscriptTests(unittest.TestCase):
    def test_all_references_are_cited_and_have_validation_records(self):
        text = (PAPER / "manuscript.tex").read_text()
        cited = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", text) for key in group.split(",")}
        references = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
        records = json.loads((ROOT / "docs/reference-validation.json").read_text())["references"]
        self.assertGreaterEqual(len(references), 40)
        self.assertEqual(cited, references)
        self.assertEqual(cited, {record["key"] for record in records})
        self.assertNotIn(r"\nocite", text)
        for record in records:
            self.assertTrue(record["primary_url"].startswith("https://"))
            self.assertTrue(record["metadata_verified"])
            self.assertTrue(record["evidence_relevance"])

    def test_figure_dependencies_are_bundled(self):
        text = (PAPER / "manuscript.tex").read_text()
        paths = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)
        self.assertEqual(len(paths), 3)
        for relative in paths:
            self.assertTrue((PAPER / relative).is_file())
        self.assertNotIn(r"\bibliography{", text)
        self.assertNotIn(r"\input{", text)

    def test_numerical_tables_match_the_immutable_original(self):
        original = (ROOT / "papers/sparse-interaction-maps/manuscript.tex").read_text()
        current = (PAPER / "manuscript.tex").read_text()
        pattern = r"\\begin\{tabular\}.*?\\end\{tabular\}"
        self.assertEqual(re.findall(pattern, original, re.S)[:4], re.findall(pattern, current, re.S))


if __name__ == "__main__":
    unittest.main()
