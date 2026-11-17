#!/usr/bin/env python3
"""Verify the Git-tracked scientific originals using only Python's stdlib."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def checked_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"unsafe manifest path: {name}")
    candidate = root / relative
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"manifest path leaves repository: {name}")
    return candidate


def check(root: Path) -> dict:
    manifest = json.loads((root / "provenance/REPOSITORY_MANIFEST.json").read_text())
    originals = {}
    for record in manifest["files"]:
        path = checked_path(root, record["path"])
        if path.suffix in {".pt", ".pth", ".pkl"}:
            raise ValueError("raw arrays or model files must not be Git-tracked")
        if not path.is_file() or path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"missing or size-changed scientific original: {record['path']}")
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"changed scientific original: {record['path']}")
        originals[record["original_path"]] = record
    # Each included source member must still match its independent archive
    # manifest, including the relocated original project configuration.
    for name in ("S1_Code", "S2_Data"):
        archive_manifest = json.loads((root / f"provenance/archives/{name}.DATA_MANIFEST.json").read_text())
        for record in archive_manifest["files"]:
            included = originals[record["path"]]
            if (record["sha256"], record["size_bytes"]) != (included["sha256"], included["size_bytes"]):
                raise ValueError(f"archive/source mapping changed: {record['path']}")
    release = json.loads((root / "provenance/release.json").read_text())
    if len(release["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in release["sha256"]):
        raise ValueError("invalid release checksum")
    return {"scientific_files_verified": len(manifest["files"]), "archive_manifest_mapping_matches": True,
            "raw_response_arrays_in_git": False, "model_calls": 0, "network_calls": 0,
            "scope": "Included source and saved JSON integrity only; not raw-response reanalysis."}


if __name__ == "__main__":
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2))
