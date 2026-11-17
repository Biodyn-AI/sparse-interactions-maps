# ruff: noqa: EXE002
"""Seal, acquire, or score the prospective E31.1 sparse-interaction atlas."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from experiments.e31_sparse_interactions.prospective_orchestration import (
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_REGISTRATION_PATH,
    prospective_dry_run,
    run_registered_prospective,
    score_registered_atlas,
    seal_prospective_registration,
)


def _required(value: str | None, flag: str) -> str:
    if value is None:
        raise SystemExit(f"{flag} is required for this mode")
    return value


def _score_existing(arguments: argparse.Namespace) -> dict[str, Any]:
    return score_registered_atlas(
        registration_path=arguments.registration,
        expected_registration_sha256=_required(
            arguments.registration_sha256, "--registration-sha256"
        ),
        output_directory=arguments.output,
        expected_discovery_labels_sha256=_required(
            arguments.discovery_labels_sha256, "--discovery-labels-sha256"
        ),
        expected_geometry_sha256=_required(
            arguments.geometry_sha256, "--geometry-sha256"
        ),
        expected_confirmation_labels_sha256=_required(
            arguments.confirmation_labels_sha256,
            "--confirmation-labels-sha256",
        ),
        confirm_score=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--seal-registration", action="store_true")
    mode.add_argument("--confirm-run", action="store_true")
    mode.add_argument("--score-existing", action="store_true")
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--registration-sha256")
    parser.add_argument("--discovery-labels-sha256")
    parser.add_argument("--geometry-sha256")
    parser.add_argument("--confirmation-labels-sha256")
    arguments = parser.parse_args()

    started = time.monotonic()
    if arguments.dry_run:
        report: dict[str, Any] = prospective_dry_run()
    elif arguments.seal_registration:
        report = seal_prospective_registration(
            registration_path=arguments.registration,
            confirm_seal=True,
        )
    elif arguments.score_existing:
        report = _score_existing(arguments)
    else:
        registration_sha256 = _required(
            arguments.registration_sha256, "--registration-sha256"
        )
        report = run_registered_prospective(
            registration_path=arguments.registration,
            expected_registration_sha256=registration_sha256,
            output_directory=arguments.output,
            confirm_model_run=True,
        )
    report["command_runtime_seconds"] = time.monotonic() - started
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
