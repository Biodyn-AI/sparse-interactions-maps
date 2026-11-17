#!/usr/bin/env python3
"""Replay the complete, hash-verified release without model or network calls."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from check_repository import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "data/full")
    parser.add_argument("--hashes-only", action="store_true")
    options = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    data_root = options.data_root.resolve()
    helper = data_root / "submission_support/replay_data.py"
    if not (data_root / "DATA_MANIFEST.json").is_file() or not helper.is_file():
        parser.error("complete data not found; run scripts/download_data.py or pass an extracted verified release with --data-root")
    # Execute only the helper whose bytes were already published with this
    # repository, not an arbitrary Python file from a supplied data directory.
    trusted = repository / "submission_support/replay_data.py"
    if file_sha256(helper) != file_sha256(trusted):
        raise ValueError("data archive's replay helper differs from the trusted repository copy")
    command = [sys.executable, str(trusted), "--root", str(data_root)]
    if options.hashes_only:
        command.append("--hashes-only")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
