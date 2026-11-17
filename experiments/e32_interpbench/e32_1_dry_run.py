# ruff: noqa: EXE002
"""Model-free command-line preview for the unsealed E32.1 protocol."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from experiments.e32_interpbench.e32_1_protocol import protocol_dry_run_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(protocol_dry_run_manifest(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
