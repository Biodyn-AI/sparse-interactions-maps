# ruff: noqa: EXE002
"""Write-once E32.3 V2 recovery for one response-blind generator assertion.

V1 stopped before any patched intervention because the pinned InterpBench
``unique_data=True`` implementation returned two repeated *unused candidate*
rows for case-25 confirmation.  This wrapper preserves the sealed V1 sources
and the complete scientific protocol.  Its only operational amendment is to
retain the exact original 192 ordered candidates while enforcing uniqueness,
freshness, and correctness on the 128 selected rows, as V1 already intended.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from experiments.e32_interpbench.adapter import (
    BehaviorallyEligiblePair,
    OfficialDataPair,
    _correct_context_mask,
    behaviorally_eligible_pair,
    generate_official_pair,
    sha256_file,
)
from experiments.e32_interpbench.e32_2_acquisition import token_rows
from experiments.e32_interpbench.e32_3_cohort import (
    CASE_ORDER,
    SMOKE_REQUESTS,
    CasePanel,
    acquire_case_panel,
    load_verified_model,
    load_verified_official_case,
    smoke_excluded_rows,
    split_request,
    validate_case_panel,
)
from experiments.e32_interpbench.e32_3_run import (
    CANONICAL_PATHS as V1_CANONICAL_PATHS,
)
from experiments.e32_interpbench.e32_3_run import (
    SOURCE_RELATIVE_PATHS as V1_SOURCE_RELATIVE_PATHS,
)
from experiments.e32_interpbench.e32_3_run import (
    RunPaths,
    _behavior_screen_record,
    _json_sha256,
)
from experiments.e32_interpbench.e32_3_run import (
    _build_registration_record as _v1_build_registration_record,
)
from experiments.e32_interpbench.e32_3_run import (
    _ProductionExecutor as _V1ProductionExecutor,
)
from experiments.e32_interpbench.e32_3_run import (
    _run_registered_experiment as _v1_run_registered_experiment,
)
from experiments.e32_interpbench.e32_3_run import (
    _seal_registration as _v1_seal_registration,
)

CANONICAL_PATHS = RunPaths(
    REPOSITORY_ROOT / "notes" / "E32.3-SPARSE-CORE-V2-INPUT-REPAIR-REGISTRATION.json",
    REPOSITORY_ROOT / "notes" / "E32.3-SPARSE-CORE-V2-INPUT-REPAIR-RUN-ATTEMPT.json",
    REPOSITORY_ROOT / "runs" / "e32_3_sparse_core_v2_input_repair",
).resolved()
CANONICAL_REGISTRATION_PATH = CANONICAL_PATHS.registration_path
CANONICAL_RUN_ATTEMPT_PATH = CANONICAL_PATHS.run_attempt_path
CANONICAL_RUN_DIRECTORY = CANONICAL_PATHS.run_directory

V1_ARTIFACT_SHA256: Mapping[str, str] = {
    "registration": "f1cc5d11b4e3ae3159c9b35c9d0cbc95748ec8349fdf8bad34c7dbef8cd4e5ac",
    "run_attempt": "4867f0bd6c10f7a0512db3146e95067122a53ae3f462070f451c79431548cdbc",
    "acquisition_attempt": (
        "5ab3f76bf9c369882d7ec18cdc6943b53c935f435cdedb9b0b53fe377dd2bfc1"
    ),
    "failure": "67339d9c0591b7e3b81008f61d18a08c98a0875dde143c336f4e7df78199bb41",
}
V1_PROTOCOL_SHA256 = "06ff13a91e468d74c04a2ebf8c350b3db7080d6ce71a33a6200ecef13f8bef7e"
V1_SPLIT_ALIGNMENT_SHA256 = (
    "89cef5bbed62a5b40f1fde5cef2b019a309f41c7548b729121eecab2a2979315"
)
V1_FAILURE_MESSAGE = (
    "official corrupt unique_data=True returned only 190 unique rows out of 192"
)

EXPECTED_RAW_POOL_SHA256: Mapping[int, Mapping[str, str]] = {
    58: {
        "smoke": "acbf0d425d8e334dbb624472debfaeed6d1aaa81d9d325764af6eb6ebe4f3672",
        "discovery": (
            "ce5f0aa4682382690f25b6fe9aeac9e852e1af0e378560dae710b594f78ecc21"
        ),
        "confirmation": (
            "70eba0f80db5c254213395d2ef720d6070d0b398b3b510ee3614f3eaa8cfc173"
        ),
    },
    129: {
        "smoke": "bb35fbf68c24bbae077090008d12cd30dacfc5caae606dda71949eacb1866e5e",
        "discovery": (
            "1aee2293abe9457a748995dff33181f4de80b9c6c6f57b2d806acd8b75910f56"
        ),
        "confirmation": (
            "4e6e1f9e2c20f6983c00f6c0a6f57993b07edd7964cdeb8224ea3f5fb7dcec8f"
        ),
    },
    25: {
        "smoke": "4b5c19f99d14f8644a02980e68ccf94ead0d75e5ae836317983619682e85639b",
        "discovery": (
            "e5005c33d8199553db0ad07db4a363223309c0286e10d7cc8c844ac45de700a7"
        ),
        "confirmation": (
            "ed67c679b257bc2f4fd3a0d3d11c170ab616929473a211f87516115a70233231"
        ),
    },
}
CASE25_CONFIRMATION_TARGET_IDENTITY_SHA256 = (
    "01b73fe2cf9bf0604c34b89068c075cddb29b63901e550da37f88f2a58bfcd17"
)
CASE25_CONFIRMATION_ROW_SHA256 = {
    "clean": "717e9ea0bb3a97677abb3e6dc0084305dbe6095687255f9a930be856f36846e9",
    "corrupt": "e5487891ecb3577f43e9ff7401b0e760e6c956c230b9ac935b5b8946ee6b413e",
    "ordered_pairs": (
        "49d62e7c2a324f93050f9898e65a94a0a6f5aeb6790cc986d91fd96e923e273b"
    ),
}

V1_BEHAVIOR_ONLY_EXPOSURE = {
    "accounting_source": (
        "inferred_from_frozen_sequential_screen_source_and_exact_terminal_failure"
    ),
    "physical_forwards": 10,
    "model_examples": 1_344,
    "gradient_enabled_forwards": 0,
    "backward_passes": 0,
    "patched_intervention_model_examples": 0,
    "completed_behavioral_panels": {
        "case58_discovery": {"physical_forwards": 2, "model_examples": 192},
        "case58_confirmation": {"physical_forwards": 2, "model_examples": 384},
        "case129_discovery": {"physical_forwards": 2, "model_examples": 192},
        "case129_confirmation": {"physical_forwards": 2, "model_examples": 384},
        "case25_discovery": {"physical_forwards": 2, "model_examples": 192},
    },
    "case25_confirmation": {
        "inputs_and_targets_generated": True,
        "behavioral_model_forwards": 0,
        "behavioral_model_examples": 0,
    },
}
V2_PLANNED_QUERY_ACCOUNTING = {
    "physical_forwards": 2_112,
    "model_examples": 302_784,
    "eap_gradient_forwards": 36,
    "gradient_examples": 3_456,
    "eap_backward_passes_inferred_from_pinned_control_flow": 36,
}
CUMULATIVE_IF_V2_SUCCEEDS = {
    "physical_forwards": 2_122,
    "model_examples": 304_128,
    "eap_gradient_forwards": 36,
    "gradient_examples": 3_456,
    "eap_backward_passes_inferred_from_pinned_control_flow": 36,
}


def _require_positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def generate_official_pair_allow_raw_duplicates(
    case: Any,
    *,
    sample_count: int,
    clean_seed: int,
    corrupt_seed: int,
) -> OfficialDataPair:
    """Return the exact official ordered pool without asserting raw uniqueness.

    This deliberately does not deduplicate, reorder, oversample, reseed, retry,
    or re-pair rows.  Selected-row uniqueness remains enforced downstream.
    """
    count = _require_positive_integer(sample_count, label="sample_count")
    clean = case.get_clean_data(
        min_samples=count,
        max_samples=count,
        seed=clean_seed,
        unique_data=True,
    )
    corrupt = case.get_corrupted_data(
        min_samples=count,
        max_samples=count,
        seed=corrupt_seed,
        unique_data=True,
    )
    values: dict[str, torch.Tensor] = {}
    for role, dataset in (("clean", clean), ("corrupt", corrupt)):
        inputs = torch.as_tensor(dataset.inputs).detach().cpu()
        targets = torch.as_tensor(dataset.targets).detach().cpu()
        if len(inputs) != count or len(targets) != count:
            raise RuntimeError(
                f"official {role} generator returned {len(inputs)} input and "
                f"{len(targets)} target rows; expected {count} each"
            )
        if inputs.ndim != 2 or targets.ndim != 3:
            raise ValueError("official categorical pool geometry changed")
        if inputs.shape[:2] != targets.shape[:2]:
            raise ValueError("official inputs and targets lost their row pairing")
        if inputs.is_floating_point() or inputs.dtype == torch.bool:
            raise ValueError("official inputs must contain integer token ids")
        if not targets.is_floating_point() or not bool(torch.isfinite(targets).all()):
            raise ValueError("official categorical targets must be finite floating")
        values[f"{role}_inputs"] = inputs
        values[f"{role}_targets"] = targets
    return OfficialDataPair(**values)


def _raw_identity_object(
    *,
    case_id: int,
    split: str,
    candidate_count: int,
    clean_seed: int,
    corrupt_seed: int,
    pair: OfficialDataPair,
    include_targets: bool = False,
) -> dict[str, Any]:
    clean_rows = token_rows(pair.clean_inputs, label="raw clean inputs")
    corrupt_rows = token_rows(pair.corrupt_inputs, label="raw corrupt inputs")
    identity: dict[str, Any] = {
        "schema_version": "e32_3_v2_raw_candidate_pool_v1",
        "case_id": case_id,
        "split": split,
        "candidate_count": candidate_count,
        "clean_seed": clean_seed,
        "corrupt_seed": corrupt_seed,
        "clean_input_rows": clean_rows,
        "corrupt_input_rows": corrupt_rows,
    }
    if include_targets:
        identity.update(
            {
                "clean_target_argmax_rows": pair.clean_targets.argmax(-1).tolist(),
                "corrupt_target_argmax_rows": pair.corrupt_targets.argmax(-1).tolist(),
            }
        )
    return identity


def _duplicate_indices(rows: Sequence[tuple[int, ...]]) -> tuple[tuple[int, int], ...]:
    first_seen: dict[tuple[int, ...], int] = {}
    duplicates: list[tuple[int, int]] = []
    for index, row in enumerate(rows):
        if row in first_seen:
            duplicates.append((index, first_seen[row]))
        else:
            first_seen[row] = index
    return tuple(duplicates)


def raw_candidate_pool_audit(
    *,
    case_id: int,
    split: str,
    candidate_count: int,
    clean_seed: int,
    corrupt_seed: int,
    pair: OfficialDataPair,
) -> dict[str, Any]:
    """Hash and describe one pool using only response-blind input identities."""
    identity = _raw_identity_object(
        case_id=case_id,
        split=split,
        candidate_count=candidate_count,
        clean_seed=clean_seed,
        corrupt_seed=corrupt_seed,
        pair=pair,
    )
    clean_rows = tuple(identity["clean_input_rows"])
    corrupt_rows = tuple(identity["corrupt_input_rows"])
    if len(clean_rows) != candidate_count or len(corrupt_rows) != candidate_count:
        raise ValueError("raw candidate identity count changed")
    clean_set = set(clean_rows)
    corrupt_set = set(corrupt_rows)
    return {
        "schema_version": "e32_3_v2_raw_candidate_pool_audit_v1",
        "case_id": case_id,
        "split": split,
        "candidate_count": candidate_count,
        "clean_seed": clean_seed,
        "corrupt_seed": corrupt_seed,
        "identity_sha256": _json_sha256(identity),
        "target_identity_sha256": _json_sha256(
            _raw_identity_object(
                case_id=case_id,
                split=split,
                candidate_count=candidate_count,
                clean_seed=clean_seed,
                corrupt_seed=corrupt_seed,
                pair=pair,
                include_targets=True,
            )
        ),
        "clean_rows_sha256": _json_sha256(clean_rows),
        "corrupt_rows_sha256": _json_sha256(corrupt_rows),
        "ordered_pairs_sha256": _json_sha256(tuple(zip(clean_rows, corrupt_rows))),
        "clean_unique_count": len(clean_set),
        "corrupt_unique_count": len(corrupt_set),
        "role_union_unique_count": len(clean_set | corrupt_set),
        "cross_role_overlap_count": len(clean_set & corrupt_set),
        "equal_position_pair_count": sum(
            clean == corrupt
            for clean, corrupt in zip(clean_rows, corrupt_rows, strict=True)
        ),
        "clean_duplicate_indices": _duplicate_indices(clean_rows),
        "corrupt_duplicate_indices": _duplicate_indices(corrupt_rows),
        "ordered_pair_count": len(clean_rows),
    }


def _normalize_exclusions(
    excluded_token_rows: Collection[Sequence[int]],
) -> set[tuple[int, ...]]:
    normalized: set[tuple[int, ...]] = set()
    for row in excluded_token_rows:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError("excluded token rows must be integer sequences")
        if not row or any(
            isinstance(token, bool) or not isinstance(token, int) for token in row
        ):
            raise ValueError("excluded token rows must be nonempty integer sequences")
        normalized.add(tuple(row))
    return normalized


def _greedy_response_blind_indices(
    pair: OfficialDataPair, *, excluded_token_rows: Collection[Sequence[int]]
) -> tuple[int, ...]:
    clean_rows = token_rows(pair.clean_inputs, label="candidate clean inputs")
    corrupt_rows = token_rows(pair.corrupt_inputs, label="candidate corrupt inputs")
    if len(clean_rows) != len(corrupt_rows):
        raise ValueError("candidate role counts differ")
    used_rows = _normalize_exclusions(excluded_token_rows)
    accepted: list[int] = []
    for index, (clean_row, corrupt_row) in enumerate(
        zip(clean_rows, corrupt_rows, strict=True)
    ):
        if (
            clean_row == corrupt_row
            or clean_row in used_rows
            or corrupt_row in used_rows
        ):
            continue
        accepted.append(index)
        used_rows.update((clean_row, corrupt_row))
    return tuple(accepted)


def response_blind_capacity_bound(
    *,
    smoke_rows: Collection[Sequence[int]],
    discovery_pair: OfficialDataPair,
    confirmation_pair: OfficialDataPair,
) -> dict[str, Any]:
    """Conservatively bound C capacity for any behavior-selected discovery rows."""
    normalized_smoke = _normalize_exclusions(smoke_rows)
    discovery_indices = _greedy_response_blind_indices(
        discovery_pair, excluded_token_rows=normalized_smoke
    )
    discovery_clean = token_rows(
        discovery_pair.clean_inputs, label="discovery candidate clean inputs"
    )
    discovery_corrupt = token_rows(
        discovery_pair.corrupt_inputs, label="discovery candidate corrupt inputs"
    )
    # Correctness is unknown pre-run.  A behaviorally ineligible earlier row
    # may allow a later duplicate to be selected, so the conservative D set is
    # every individually smoke-fresh pair, not only the all-correct greedy scan.
    individually_fresh_discovery_indices = tuple(
        index
        for index, (clean_row, corrupt_row) in enumerate(
            zip(discovery_clean, discovery_corrupt, strict=True)
        )
        if clean_row != corrupt_row
        and clean_row not in normalized_smoke
        and corrupt_row not in normalized_smoke
    )
    possible_discovery_rows = {
        row
        for index in individually_fresh_discovery_indices
        for row in (discovery_clean[index], discovery_corrupt[index])
    }
    confirmation_indices = _greedy_response_blind_indices(
        confirmation_pair, excluded_token_rows=normalized_smoke
    )
    confirmation_clean = token_rows(
        confirmation_pair.clean_inputs, label="confirmation candidate clean inputs"
    )
    confirmation_corrupt = token_rows(
        confirmation_pair.corrupt_inputs, label="confirmation candidate corrupt inputs"
    )
    touching = tuple(
        index
        for index in confirmation_indices
        if confirmation_clean[index] in possible_discovery_rows
        or confirmation_corrupt[index] in possible_discovery_rows
    )
    return {
        "discovery_response_blind_selectable_count": len(discovery_indices),
        "discovery_individually_fresh_candidate_count": len(
            individually_fresh_discovery_indices
        ),
        "confirmation_capacity_after_smoke": len(confirmation_indices),
        "confirmation_indices_touching_any_discovery_candidate": touching,
        "confirmation_capacity_lower_bound_after_any_discovery": (
            len(confirmation_indices) - len(touching)
        ),
    }


def _select_behaviorally_eligible_from_pair(
    model: Any,
    case: Any,
    *,
    candidates: OfficialDataPair,
    required_count: int,
    candidate_count: int,
    device: str,
    excluded_token_rows: Collection[Sequence[int]],
    require_unique_selected_token_rows: bool,
) -> BehaviorallyEligiblePair:
    required = _require_positive_integer(required_count, label="required_count")
    count = _require_positive_integer(candidate_count, label="candidate_count")
    if count < required:
        raise ValueError("candidate_count must be at least required_count")
    if len(candidates.clean_inputs) != count or len(candidates.corrupt_inputs) != count:
        raise ValueError("candidate pair count differs from candidate_count")
    is_categorical = getattr(case, "is_categorical", None)
    if not callable(is_categorical) or not bool(is_categorical()):
        raise NotImplementedError("E32.3 V2 eligibility requires a categorical case")

    model.to(device)
    model.eval()
    with torch.inference_mode():
        clean_logits = model(candidates.clean_inputs.to(device)).detach().cpu()
        corrupt_logits = model(candidates.corrupt_inputs.to(device)).detach().cpu()
    clean_correct = _correct_context_mask(clean_logits, candidates.clean_targets)
    corrupt_correct = _correct_context_mask(corrupt_logits, candidates.corrupt_targets)
    eligible_mask = clean_correct & corrupt_correct
    eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).flatten().tolist()

    exclusions = _normalize_exclusions(excluded_token_rows)
    enforce_disjoint = require_unique_selected_token_rows or bool(exclusions)
    row_overlap_excluded = 0
    if enforce_disjoint:
        clean_rows = token_rows(candidates.clean_inputs, label="candidate clean inputs")
        corrupt_rows = token_rows(
            candidates.corrupt_inputs, label="candidate corrupt inputs"
        )
        used_rows = set(exclusions)
        selectable: list[int] = []
        for index in eligible_indices:
            clean_row = clean_rows[index]
            corrupt_row = corrupt_rows[index]
            if (
                clean_row == corrupt_row
                or clean_row in used_rows
                or corrupt_row in used_rows
            ):
                row_overlap_excluded += 1
                continue
            selectable.append(int(index))
            used_rows.update((clean_row, corrupt_row))
        policy = (
            "first_jointly_correct_with_globally_unique_clean_source_rows_and_"
            "response_blind_exclusions"
        )
    else:
        selectable = [int(index) for index in eligible_indices]
        policy = "first_jointly_correct"
    if len(selectable) < required:
        raise RuntimeError(
            f"behavioral eligibility found only {len(selectable)} selectable rows "
            f"out of {count}; required {required}"
        )
    selected_indices = tuple(selectable[:required])
    tensor_indices = torch.tensor(selected_indices, dtype=torch.long)

    def select(value: torch.Tensor) -> torch.Tensor:
        return value.index_select(0, tensor_indices)

    return BehaviorallyEligiblePair(
        pair=OfficialDataPair(
            clean_inputs=select(candidates.clean_inputs),
            clean_targets=select(candidates.clean_targets),
            corrupt_inputs=select(candidates.corrupt_inputs),
            corrupt_targets=select(candidates.corrupt_targets),
        ),
        candidate_count=count,
        eligible_candidate_count=len(eligible_indices),
        clean_incorrect_contexts=int((~clean_correct).sum()),
        source_incorrect_contexts=int((~corrupt_correct).sum()),
        either_incorrect_contexts=int((~eligible_mask).sum()),
        selected_candidate_indices=selected_indices,
        eligibility_physical_forwards=2,
        eligibility_model_examples=2 * count,
        output_kind="categorical",
        row_disjoint_eligible_candidate_count=len(selectable),
        row_overlap_excluded_contexts=row_overlap_excluded,
        excluded_token_row_count=len(exclusions),
        selection_policy=policy,
    )


def behaviorally_eligible_pair_allow_raw_duplicates(
    model: Any,
    case: Any,
    *,
    required_count: int,
    candidate_count: int,
    clean_seed: int,
    corrupt_seed: int,
    device: str,
    excluded_token_rows: Collection[Sequence[int]],
    require_unique_selected_token_rows: bool,
) -> BehaviorallyEligiblePair:
    if (candidate_count, clean_seed, corrupt_seed) != (192, 725_200, 725_201):
        raise ValueError(
            "the duplicate-tolerant path is frozen to case-25 confirmation raw192"
        )
    candidates = generate_official_pair_allow_raw_duplicates(
        case,
        sample_count=candidate_count,
        clean_seed=clean_seed,
        corrupt_seed=corrupt_seed,
    )
    # Re-hash the exact candidates that will enter the behavioral model.  The
    # earlier global preflight is a phase barrier; this second check closes the
    # regeneration seam and makes a stateful or drifting generator fail before
    # either eligibility forward.
    audit = raw_candidate_pool_audit(
        case_id=25,
        split="confirmation",
        candidate_count=candidate_count,
        clean_seed=clean_seed,
        corrupt_seed=corrupt_seed,
        pair=candidates,
    )
    if (
        audit["identity_sha256"] != EXPECTED_RAW_POOL_SHA256[25]["confirmation"]
        or audit["target_identity_sha256"] != CASE25_CONFIRMATION_TARGET_IDENTITY_SHA256
    ):
        raise ValueError(
            "case-25 confirmation candidates changed after response-blind preflight"
        )
    return _select_behaviorally_eligible_from_pair(
        model,
        case,
        candidates=candidates,
        required_count=required_count,
        candidate_count=candidate_count,
        device=device,
        excluded_token_rows=excluded_token_rows,
        require_unique_selected_token_rows=require_unique_selected_token_rows,
    )


def _selected_role_rows(
    eligible: BehaviorallyEligiblePair,
) -> tuple[tuple[int, ...], ...]:
    return (
        *token_rows(eligible.pair.clean_inputs, label="selected clean inputs"),
        *token_rows(eligible.pair.corrupt_inputs, label="selected corrupt inputs"),
    )


def _acquire_case25_panel_v2(
    case_id: int,
    *,
    model: Any,
    case: Any,
    device: str = "cpu",
) -> CasePanel:
    if case_id != 25:
        raise ValueError("the V2 raw-duplicate amendment applies only to case 25")
    smoke = smoke_excluded_rows(case_id, case)
    discovery_request = split_request(case_id, "discovery")
    discovery = behaviorally_eligible_pair(
        model,
        case,
        required_count=discovery_request.required_count,
        candidate_count=discovery_request.candidate_count,
        clean_seed=discovery_request.clean_seed,
        corrupt_seed=discovery_request.corrupt_seed,
        device=device,
        excluded_token_rows=smoke,
        require_unique_selected_token_rows=True,
    )
    confirmation_request = split_request(case_id, "confirmation")
    confirmation = behaviorally_eligible_pair_allow_raw_duplicates(
        model,
        case,
        required_count=confirmation_request.required_count,
        candidate_count=confirmation_request.candidate_count,
        clean_seed=confirmation_request.clean_seed,
        corrupt_seed=confirmation_request.corrupt_seed,
        device=device,
        excluded_token_rows=tuple(
            dict.fromkeys((*smoke, *_selected_role_rows(discovery)))
        ),
        require_unique_selected_token_rows=True,
    )
    panel = CasePanel(case_id, discovery, confirmation, smoke)
    validate_case_panel(panel)
    return panel


def _request_fields(case_id: int, split: str) -> tuple[int, int, int]:
    if split == "smoke":
        request = SMOKE_REQUESTS[case_id]
        return request.sample_count, request.clean_seed, request.corrupt_seed
    request = split_request(case_id, split)
    return request.candidate_count, request.clean_seed, request.corrupt_seed


def _preflight_all_input_pools() -> tuple[dict[int, Any], dict[str, Any]]:
    """Regenerate and hash every raw pool before any model is loaded."""
    cases = {case_id: load_verified_official_case(case_id) for case_id in CASE_ORDER}
    audits: dict[str, dict[str, Any]] = {}
    raw_pairs: dict[tuple[int, str], OfficialDataPair] = {}
    for case_id in CASE_ORDER:
        case_audits: dict[str, Any] = {}
        for split in ("smoke", "discovery", "confirmation"):
            count, clean_seed, corrupt_seed = _request_fields(case_id, split)
            if case_id == 25 and split == "confirmation":
                pair = generate_official_pair_allow_raw_duplicates(
                    cases[case_id],
                    sample_count=count,
                    clean_seed=clean_seed,
                    corrupt_seed=corrupt_seed,
                )
            else:
                pair = generate_official_pair(
                    cases[case_id],
                    sample_count=count,
                    clean_seed=clean_seed,
                    corrupt_seed=corrupt_seed,
                )
            audit = raw_candidate_pool_audit(
                case_id=case_id,
                split=split,
                candidate_count=count,
                clean_seed=clean_seed,
                corrupt_seed=corrupt_seed,
                pair=pair,
            )
            expected = EXPECTED_RAW_POOL_SHA256[case_id][split]
            if audit["identity_sha256"] != expected:
                raise ValueError(
                    f"case-{case_id} {split} raw input identity changed: "
                    f"expected {expected}, observed {audit['identity_sha256']}"
                )
            case_audits[split] = audit
            raw_pairs[(case_id, split)] = pair
        audits[str(case_id)] = case_audits

    case25_confirmation = audits["25"]["confirmation"]
    expected_case25 = {
        "clean_unique_count": 192,
        "corrupt_unique_count": 190,
        "role_union_unique_count": 377,
        "cross_role_overlap_count": 5,
        "equal_position_pair_count": 0,
        "clean_duplicate_indices": (),
        "corrupt_duplicate_indices": ((54, 34), (109, 65)),
        "clean_rows_sha256": CASE25_CONFIRMATION_ROW_SHA256["clean"],
        "corrupt_rows_sha256": CASE25_CONFIRMATION_ROW_SHA256["corrupt"],
        "ordered_pairs_sha256": CASE25_CONFIRMATION_ROW_SHA256["ordered_pairs"],
        "target_identity_sha256": CASE25_CONFIRMATION_TARGET_IDENTITY_SHA256,
    }
    for field, expected in expected_case25.items():
        if case25_confirmation[field] != expected:
            raise ValueError(
                f"case-25 confirmation response-blind {field} changed: "
                f"expected {expected!r}, observed {case25_confirmation[field]!r}"
            )
    smoke_pair = raw_pairs[(25, "smoke")]
    smoke_rows = tuple(
        dict.fromkeys(
            (
                *token_rows(smoke_pair.clean_inputs, label="case25 smoke clean"),
                *token_rows(smoke_pair.corrupt_inputs, label="case25 smoke corrupt"),
            )
        )
    )
    capacity = response_blind_capacity_bound(
        smoke_rows=smoke_rows,
        discovery_pair=raw_pairs[(25, "discovery")],
        confirmation_pair=raw_pairs[(25, "confirmation")],
    )
    if capacity != {
        "discovery_response_blind_selectable_count": 91,
        "discovery_individually_fresh_candidate_count": 92,
        "confirmation_capacity_after_smoke": 184,
        "confirmation_indices_touching_any_discovery_candidate": (2, 170),
        "confirmation_capacity_lower_bound_after_any_discovery": 182,
    }:
        raise ValueError(f"case-25 response-blind capacity proof changed: {capacity!r}")
    return cases, {
        "schema_version": "e32_3_v2_all_pool_input_preflight_v1",
        "verified": True,
        "model_calls": 0,
        "pool_identity_sha256": {
            str(case_id): dict(EXPECTED_RAW_POOL_SHA256[case_id])
            for case_id in CASE_ORDER
        },
        "case25_confirmation_raw_pool": case25_confirmation,
        "case25_confirmation_capacity_proof": capacity,
    }


def _read_pinned_json(path: Path, *, label: str, expected_sha256: str) -> Any:
    if not path.is_file():
        raise ValueError(f"V1 {label} is missing")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"V1 {label} SHA-256 changed: expected {expected_sha256}, "
            f"observed {observed}"
        )
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"V1 {label} is malformed") from error


def verify_v1_failure_state() -> dict[str, Any]:
    """Require the exact terminal V1 state without opening scientific tensors."""
    paths = V1_CANONICAL_PATHS.resolved()
    registration = _read_pinned_json(
        paths.registration_path,
        label="registration",
        expected_sha256=V1_ARTIFACT_SHA256["registration"],
    )
    if (
        registration.get("canonical_registration_path") != str(paths.registration_path)
        or registration.get("canonical_run_attempt_path") != str(paths.run_attempt_path)
        or registration.get("canonical_run_directory") != str(paths.run_directory)
        or _json_sha256(registration.get("protocol")) != V1_PROTOCOL_SHA256
        or _json_sha256(registration.get("split_request_alignment"))
        != V1_SPLIT_ALIGNMENT_SHA256
    ):
        raise ValueError("V1 registration protocol, alignment, or paths changed")
    registered_sources = registration.get("source_sha256")
    if not isinstance(registered_sources, Mapping) or tuple(
        registered_sources
    ) != tuple(V1_SOURCE_RELATIVE_PATHS):
        raise ValueError("V1 registered source closure changed")
    for relative, digest in registered_sources.items():
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"V1 registered source changed: {relative}")

    run_attempt = _read_pinned_json(
        paths.run_attempt_path,
        label="run attempt",
        expected_sha256=V1_ARTIFACT_SHA256["run_attempt"],
    )
    expected_run_attempt = {
        "schema_version": "e32_3_run_attempt_v1",
        "registration_sha256": V1_ARTIFACT_SHA256["registration"],
        "canonical_run_directory": str(paths.run_directory),
        "status": "irreversible_attempt_reserved_before_run_directory_or_model",
    }
    if run_attempt != expected_run_attempt:
        raise ValueError("V1 run-attempt semantics changed")

    expected_entries = {"acquisition_attempt.json", "failure.json"}
    if not paths.run_directory.is_dir():
        raise ValueError("V1 run directory is missing")
    observed_entries = {
        entry.name
        for entry in paths.run_directory.iterdir()
        if not entry.name.startswith("._")
    }
    if observed_entries != expected_entries:
        raise ValueError(
            "V1 terminal regular artifact set changed: "
            f"expected {sorted(expected_entries)!r}, "
            f"observed {sorted(observed_entries)!r}"
        )
    acquisition_attempt_path = paths.run_directory / "acquisition_attempt.json"
    acquisition_attempt = _read_pinned_json(
        acquisition_attempt_path,
        label="acquisition attempt",
        expected_sha256=V1_ARTIFACT_SHA256["acquisition_attempt"],
    )
    expected_acquisition_attempt = {
        "schema_version": "e32_3_acquisition_attempt_v1",
        "registration_sha256": V1_ARTIFACT_SHA256["registration"],
        "run_attempt_sha256": V1_ARTIFACT_SHA256["run_attempt"],
        "phase_order": [
            "global_behavior_screen",
            "all_discoveries",
            "all_authorizations",
            "all_confirmations",
            "score_attempt",
            "score_and_exact_replay",
        ],
        "status": "durable_before_first_behavior_or_intervention_model_call",
    }
    if acquisition_attempt != expected_acquisition_attempt:
        raise ValueError("V1 acquisition-attempt semantics changed")
    failure = _read_pinned_json(
        paths.run_directory / "failure.json",
        label="failure",
        expected_sha256=V1_ARTIFACT_SHA256["failure"],
    )
    expected_failure = {
        "schema_version": "e32_3_failure_v1",
        "registration_sha256": V1_ARTIFACT_SHA256["registration"],
        "run_attempt_sha256": V1_ARTIFACT_SHA256["run_attempt"],
        "score_attempt_sha256": None,
        "error_type": "RuntimeError",
        "error_message": V1_FAILURE_MESSAGE,
        "completed_acquisition_artifact_sha256": {},
        "completed_custody_artifact_sha256": {},
    }
    if failure != expected_failure:
        raise ValueError("V1 failure semantics changed")
    return {
        "schema_version": "e32_3_v1_terminal_failure_audit_v1",
        "verified": True,
        "artifact_sha256": dict(V1_ARTIFACT_SHA256),
        "failure_stage": "global_behavior_screen_case25_confirmation",
        "failure_message": V1_FAILURE_MESSAGE,
        "completed_acquisition_artifact_sha256": {},
        "completed_custody_artifact_sha256": {},
        "score_attempt_sha256": None,
        "v1_behavior_only_physical_forwards": 10,
        "v1_behavior_only_model_examples": 1_344,
        "patched_intervention_model_examples": 0,
        "gradient_enabled_forwards": 0,
        "backward_passes": 0,
    }


def recovery_addendum(
    *, v1_state: Mapping[str, Any], input_preflight: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "e32_3_v2_input_repair_addendum_v1",
        "parent_v1_protocol_sha256": V1_PROTOCOL_SHA256,
        "parent_v1_split_alignment_sha256": V1_SPLIT_ALIGNMENT_SHA256,
        "parent_v1_terminal_state": dict(v1_state),
        "response_blind_input_preflight": dict(input_preflight),
        "scientific_protocol_changed": False,
        "sole_amendment": {
            "case_id": 25,
            "split": "confirmation",
            "candidate_count": 192,
            "preserve_original_order_pairing_targets_and_seeds": True,
            "bypass_only_full_raw_role_uniqueness_assertion": True,
            "deduplicate_reorder_oversample_reseed_retry": False,
            "selected_row_uniqueness_and_behavioral_gate_unchanged": True,
        },
        "prospectivity": {
            "v1_behavioral_endpoints_partially_opened": True,
            "v1_patched_intervention_outcomes_opened": False,
            "v2_patched_intervention_outcomes_prospective": True,
            "ordinary_end_to_end_pre_behavior_preregistration": False,
        },
        "v1_partial_outputs_reused": False,
        "v2_all_behavior_and_intervention_panels_acquired_from_scratch": True,
        "v1_behavior_only_exposure": dict(V1_BEHAVIOR_ONLY_EXPOSURE),
        "v2_planned_query_accounting": dict(V2_PLANNED_QUERY_ACCOUNTING),
        "cumulative_if_v2_succeeds": dict(CUMULATIVE_IF_V2_SUCCEEDS),
        "claim_boundary": (
            "same V1 sparse-core estimands and gates, prospectively continued only "
            "for patched intervention outcomes after a response-blind input repair"
        ),
    }


class _RecoveryExecutor(_V1ProductionExecutor):
    """V1 scientific executor with only the case-25 C input repair."""

    def registration_dependencies(self) -> dict[str, Any]:
        dependencies = super().registration_dependencies()
        v1_state = verify_v1_failure_state()
        _cases, preflight = _preflight_all_input_pools()
        dependencies["e32_3_v2_input_repair"] = {
            "v2_source_sha256": sha256_file(Path(__file__).resolve()),
            **recovery_addendum(v1_state=v1_state, input_preflight=preflight),
        }
        return dependencies

    def screen_all_cases(self) -> tuple[dict[str, Any], dict[int, Any]]:
        # This complete response-blind barrier precedes the first model load.
        cases, preflight = _preflight_all_input_pools()
        records: dict[str, Any] = {}
        panels: dict[int, Any] = {}
        for case_id in CASE_ORDER:
            model = load_verified_model(case_id, device="cpu")
            acquire = _acquire_case25_panel_v2 if case_id == 25 else acquire_case_panel
            panel = acquire(
                case_id,
                model=model,
                case=cases[case_id],
                device="cpu",
            )
            panels[case_id] = panel
            records[str(case_id)] = _behavior_screen_record(panel)
        records["25"]["e32_3_v2_input_repair"] = preflight
        return records, panels

    def score_complete_panel(
        self, panel: Mapping[tuple[int, str], Mapping[str, Any]]
    ) -> dict[str, Any]:
        score = super().score_complete_panel(panel)
        return {
            **score,
            "e32_3_v2_operational_recovery": {
                "schema_version": "e32_3_v2_recovery_disclosure_v1",
                "v1_behavioral_endpoints_partially_opened": True,
                "v1_patched_intervention_outcomes_opened": False,
                "v2_patched_intervention_outcomes_prospective": True,
                "ordinary_end_to_end_pre_behavior_preregistration": False,
                "scientific_protocol_changed": False,
                "v1_partial_outputs_reused": False,
                "v2_all_behavior_and_intervention_panels_acquired_from_scratch": True,
                "v1_behavior_only_exposure": dict(V1_BEHAVIOR_ONLY_EXPOSURE),
                "v2_query_accounting": dict(V2_PLANNED_QUERY_ACCOUNTING),
                "cumulative_operational_exposure": dict(CUMULATIVE_IF_V2_SUCCEEDS),
            },
        }


def build_registration_record() -> dict[str, Any]:
    """Build the V2 addendum without model acquisition or filesystem writes."""
    return _v1_build_registration_record(
        executor=_RecoveryExecutor(), paths=CANONICAL_PATHS
    )


def seal_registration() -> dict[str, Any]:
    """Write the sole canonical V2 recovery registration exactly once."""
    return _v1_seal_registration(executor=_RecoveryExecutor(), paths=CANONICAL_PATHS)


def run_registered_experiment(*, confirm: bool = False) -> dict[str, Any]:
    """Run the noninjectable V2 recovery after the exact V1 failure audit."""
    return _v1_run_registered_experiment(
        executor=_RecoveryExecutor(), paths=CANONICAL_PATHS, confirm=confirm
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("dry-run")
    subparsers.add_parser("seal")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--confirm", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.command == "dry-run":
        record = build_registration_record()
        print(
            json.dumps(
                {
                    "status": (
                        "dry_run_only_response_blind_preflight_no_model_calls_no_writes"
                    ),
                    "registration_sha256": _json_sha256(record),
                    "protocol": record["protocol"],
                    "recovery": record["dependencies"]["e32_3_v2_input_repair"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "seal":
        record = seal_registration()
        print(json.dumps({"registration_sha256": _json_sha256(record)}))
        return 0
    summary = run_registered_experiment(confirm=arguments.confirm)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_REGISTRATION_PATH",
    "CANONICAL_RUN_ATTEMPT_PATH",
    "CANONICAL_RUN_DIRECTORY",
    "build_registration_record",
    "generate_official_pair_allow_raw_duplicates",
    "raw_candidate_pool_audit",
    "run_registered_experiment",
    "seal_registration",
    "verify_v1_failure_state",
]
