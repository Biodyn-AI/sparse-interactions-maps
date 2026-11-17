#!/usr/bin/env python3
"""Command-line entry point for the prospective E31.2 custody chain."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.e31_sparse_interactions.e31_2_orchestration import (
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_REGISTRATION_PATH,
    prospective_dry_run,
    run_registered_pipeline,
    seal_prospective_registration,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "dry-run", help="verify inputs and tokenization without model calls or writes"
    )

    seal = subparsers.add_parser(
        "seal", help="write the prospective registration without loading the model"
    )
    seal.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION_PATH)
    seal.add_argument("--confirm-seal", action="store_true")

    run = subparsers.add_parser("run", help="execute the registered one-process chain")
    run.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION_PATH)
    run.add_argument("--registration-sha256", required=True)
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    run.add_argument("--confirm-model-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "dry-run":
        report = prospective_dry_run()
    elif args.command == "seal":
        report = seal_prospective_registration(
            registration_path=args.registration,
            confirm_seal=args.confirm_seal,
        )
    else:
        report = run_registered_pipeline(
            registration_path=args.registration,
            expected_registration_sha256=args.registration_sha256,
            output_directory=args.output,
            confirm_model_run=args.confirm_model_run,
        )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
