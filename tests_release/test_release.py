"""Portable-wrapper safety tests. Standard library only; no model calls."""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_repository import check, checked_path, file_sha256
from download_data import obtain_archive, safe_members


class ReleaseTests(unittest.TestCase):
    def test_committed_scientific_originals_and_source_archives_match(self):
        report = check(ROOT)
        self.assertTrue(report["archive_manifest_mapping_matches"])
        self.assertGreater(report["scientific_files_verified"], 200)

    def test_paths_cannot_escape_repository(self):
        for path in ("../outside", "/outside", "a\\b"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                checked_path(ROOT, path)

    def test_zip_slip_and_symlinks_are_rejected_before_extraction(self):
        for name in ("../outside", "/outside", "a\\b"):
            payload = io.BytesIO()
            with zipfile.ZipFile(payload, "w") as archive:
                archive.writestr(name, "not extracted")
            with zipfile.ZipFile(payload) as archive, self.assertRaises(ValueError):
                safe_members(archive)
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            member = zipfile.ZipInfo("link")
            member.external_attr = 0o120777 << 16
            archive.writestr(member, "../outside")
        with zipfile.ZipFile(payload) as archive, self.assertRaises(ValueError):
            safe_members(archive)

    def test_existing_download_is_reused_only_after_exact_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.zip"
            path.write_bytes(b"release bytes")
            release = {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            obtain_archive(path, release)
            path.write_bytes(b"different")
            with self.assertRaises(ValueError):
                obtain_archive(path, release)

    def test_saved_claims_preserve_negative_results(self):
        manifest = json.loads((ROOT / "papers/sparse-interaction-maps/evidence/scalar_manifest.json").read_text())
        self.assertEqual(manifest["e31_2"]["score"]["primary_verdict"].lower(), "fail")
        cases = manifest["e32_4"]["cases"]
        self.assertEqual(sum(case["gates"]["case_pass"] for case in cases.values()), 1)
        fresh = json.loads((ROOT / "runs/e33_sparse_interaction_diagnostics_v1/summary.json").read_text())
        self.assertEqual([case["case_id"] for case in fresh["cases"]], [63, 103, 82])
        self.assertTrue(all(case["gates"]["score_stability"] for case in fresh["cases"]))


if __name__ == "__main__":
    unittest.main()
