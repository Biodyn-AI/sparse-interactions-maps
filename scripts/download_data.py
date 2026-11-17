#!/usr/bin/env python3
"""Download the pinned response archive, verify its SHA-256, and safely extract.

This is an explicit approximately 1.14 GB download. It never loads a tensor or
uses credentials. The release identity is in provenance/release.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from check_repository import file_sha256


def safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    seen = set()
    for member in members:
        path = Path(member.filename)
        if (path.is_absolute() or ".." in path.parts or "\\" in member.filename
                or stat.S_ISLNK(member.external_attr >> 16) or member.filename in seen):
            raise ValueError(f"unsafe or duplicate archive member: {member.filename}")
        seen.add(member.filename)
    return members


def obtain_archive(destination: Path, release: dict) -> None:
    if destination.exists():
        if destination.stat().st_size == release["size_bytes"] and file_sha256(destination) == release["sha256"]:
            print("Reusing the verified release archive.", flush=True)
            return
        raise ValueError(f"existing archive differs from pinned release: {destination}")
    # Never overwrite an earlier partial download silently.
    partial = destination.with_suffix(destination.suffix + ".part")
    with partial.open("xb") as stream:
        request = urllib.request.Request(release["url"], headers={"User-Agent": "sparse-interaction-maps/1.0.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
    if partial.stat().st_size != release["size_bytes"] or file_sha256(partial) != release["sha256"]:
        raise ValueError(f"download checksum/length mismatch; inspect and remove only this partial file: {partial}")
    partial.rename(destination)


def extract_archive(archive_path: Path, target: Path, release: dict) -> None:
    if target.exists():
        raise ValueError(f"refusing to overwrite extraction folder: {target}")
    with tempfile.TemporaryDirectory(prefix="sparse-data-", dir=target.parent) as staging:
        extracted = Path(staging) / "full"
        extracted.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extracted, members=safe_members(archive))
        if not (extracted / "DATA_MANIFEST.json").is_file():
            raise ValueError("release archive has no data manifest")
        extracted.rename(target)
    print(f"Extracted the verified release to {target}. No tensor files were loaded.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--download-only", action="store_true")
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    release = json.loads((root / "provenance/release.json").read_text())
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / release["filename"]
    print(f"Pinned release: {release['tag']}; {release['size_bytes']:,} bytes; SHA-256 {release['sha256']}", flush=True)
    obtain_archive(destination, release)
    if not options.download_only:
        extract_archive(destination, output / "full", release)


if __name__ == "__main__":
    main()
