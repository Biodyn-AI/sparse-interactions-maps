"""Checks for safe, relocatable publication archives; no model execution."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


def load_module(name: str):
    specification = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class PublicationDataTests(unittest.TestCase):
    def test_path_relocation_ignores_only_recorded_machine_path(self):
        replay = load_module("replay_data")
        original = {"sources": {"a": {"absolute_path": "/old/a", "relative_path": "a", "sha256": "one"}}, "value": 3}
        relocated = {"sources": {"a": {"absolute_path": "/new/a", "relative_path": "a", "sha256": "one"}}, "value": 3}
        self.assertEqual(replay.portable_manifest(original), replay.portable_manifest(relocated))
        relocated["value"] = 4
        self.assertNotEqual(replay.portable_manifest(original), replay.portable_manifest(relocated))

    def test_manifest_checks_bytes_after_relocation_and_rejects_changes(self):
        replay = load_module("replay_data")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.txt").write_text("scientific bytes\n")
            manifest = {"files": [{"path": "sample.txt", "sha256": replay.sha256(root / "sample.txt"), "size_bytes": 17}]}
            self.assertEqual(replay.verify_files(root, manifest), 1)
            (root / "sample.txt").write_text("changed\n")
            with self.assertRaises(ValueError):
                replay.verify_files(root, manifest)

    def test_manifest_rejects_paths_outside_package(self):
        replay = load_module("replay_data")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                replay.verify_files(Path(directory), {"files": [{"path": "../outside", "sha256": "x", "size_bytes": 0}]})

    def test_numerical_replay_accepts_roundoff_but_not_scientific_changes(self):
        replay = load_module("replay_data")
        replay.assert_equivalent({"value": 4.0, "pass": True}, {"value": 4.0 + 1e-14, "pass": True})
        with self.assertRaises(ValueError):
            replay.assert_equivalent({"value": 4.0}, {"value": 4.01})
        with self.assertRaises(ValueError):
            replay.assert_equivalent({"pass": True}, {"pass": False})

    def test_source_closure_includes_local_dependencies(self):
        package = load_module("package_data")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments").mkdir()
            (root / "experiments/start.py").write_text("from experiments.helper import value\n")
            (root / "experiments/helper.py").write_text("value = 3\n")
            paths = package.source_closure(root, {Path("experiments/start.py")})
            self.assertIn(Path("experiments/helper.py"), paths)


if __name__ == "__main__":
    unittest.main()
