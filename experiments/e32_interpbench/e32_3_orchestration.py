# ruff: noqa: EXE002
"""Pure support freezing, custody, and score replay for E32.3.

This module has no filesystem or model-loading surface.  The canonical driver
persists discovery before calling :func:`freeze_discovery_supports`, then binds
the returned response-derived supports into the confirmation authorization.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from experiments.e32_interpbench.e32_3_acquisition import FrozenNodeSupport
from experiments.e32_interpbench.e32_3_cohort import (
    CASE_ORDER,
    EXPECTED_RESPONSE_BLIND_CAPACITY,
    SMOKE_REQUESTS,
)
from experiments.e32_interpbench.e32_3_cohort import (
    split_request as cohort_split_request,
)
from experiments.e32_interpbench.e32_3_protocol import (
    DETERMINISTIC_METHOD_ORDER,
    EXPECTED_QUERY_ACCOUNTING,
    PRIMARY_METHOD,
    RANDOM_CONTROL_REPLICATES,
    confirmatory_support_size,
    deterministic_random_support,
    support_sizes,
)
from experiments.e32_interpbench.e32_3_protocol import (
    split_request as protocol_split_request,
)
from experiments.e32_interpbench.e32_3_scoring import (
    deterministic_top_support,
    evaluate_panel_verdict,
    evaluate_singleton_support_curve,
    evaluate_sparse_core_case,
)

MANDATORY_PANEL_KEYS = tuple(
    (case_id, split)
    for case_id in CASE_ORDER
    for split in ("discovery", "confirmation")
)


def _node_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of node IDs")
    nodes = tuple(value)
    if (
        not nodes
        or len(nodes) != len(set(nodes))
        or any(not isinstance(node, str) or not node for node in nodes)
    ):
        raise ValueError(f"{label} must contain unique nonempty node IDs")
    return nodes


def _support_record(
    *,
    support_id: str,
    method: str,
    quantile: float,
    selected_nodes: Sequence[str],
    kind: str,
    replicate: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "support_id": support_id,
        "method": method,
        "quantile": quantile,
        "selected_nodes": tuple(selected_nodes),
        "kind": kind,
    }
    if replicate is not None:
        record["replicate"] = replicate
    return record


def freeze_discovery_supports(
    *, case_id: int, discovery: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze every confirmation mask from one already-persisted D response."""
    if case_id not in CASE_ORDER:
        raise ValueError(f"case_id must be one of {CASE_ORDER!r}")
    if (
        not isinstance(discovery, Mapping)
        or discovery.get("schema_version") != "e32_3_discovery_acquisition_v1"
        or discovery.get("case_id") != case_id
        or discovery.get("split") != "discovery"
    ):
        raise ValueError("discovery acquisition identity changed")
    nodes = _node_tuple(discovery.get("component_ids"), label="component_ids")
    scores = discovery.get("selector_scores")
    if not isinstance(scores, Mapping) or tuple(scores) != DETERMINISTIC_METHOD_ORDER:
        raise ValueError("discovery selector methods or order changed")
    for method in DETERMINISTIC_METHOD_ORDER:
        vector = torch.as_tensor(scores[method], dtype=torch.float64, device="cpu")
        if (
            vector.ndim != 1
            or vector.numel() != len(nodes)
            or not bool(torch.isfinite(vector).all())
        ):
            raise ValueError(f"discovery selector {method} differs from node universe")

    half_size = confirmatory_support_size(len(nodes))
    deterministic: dict[str, dict[str, Any]] = {}
    for method in DETERMINISTIC_METHOD_ORDER:
        support = deterministic_top_support(
            node_universe=nodes,
            node_scores=scores[method],
            support_size=half_size,
        )
        record = _support_record(
            support_id=f"deterministic:{method}:q50",
            method=method,
            quantile=0.5,
            selected_nodes=support,
            kind="deterministic",
        )
        deterministic[method] = record

    curve: dict[int, dict[str, Any]] = {}
    curve_sizes = support_sizes(len(nodes))[1:4]
    for fraction, size in zip((0.25, 0.5, 0.75), curve_sizes, strict=True):
        if size == half_size:
            record = deterministic[PRIMARY_METHOD]
        else:
            record = _support_record(
                support_id=f"curve:{PRIMARY_METHOD}:q{int(fraction * 100):02d}",
                method=PRIMARY_METHOD,
                quantile=fraction,
                selected_nodes=deterministic_top_support(
                    node_universe=nodes,
                    node_scores=scores[PRIMARY_METHOD],
                    support_size=size,
                ),
                kind="deterministic",
            )
        curve[size] = record

    random: list[dict[str, Any]] = []
    for replicate in range(RANDOM_CONTROL_REPLICATES):
        record = _support_record(
            support_id=f"random:q50:r{replicate:02d}",
            method="random_control",
            quantile=0.5,
            selected_nodes=deterministic_random_support(
                case_id=case_id,
                component_ids=nodes,
                support_size=half_size,
                replicate=replicate,
            ),
            kind="random",
            replicate=replicate,
        )
        random.append(record)

    # This is the sole acquisition order: singleton curve first, then the four
    # other q=.5 comparators, then the 64 response-blind controls.  The q=.5
    # singleton record is shared, not duplicated.
    all_supports = [
        *(curve[size] for size in curve_sizes),
        *(
            deterministic[method]
            for method in DETERMINISTIC_METHOD_ORDER
            if method != PRIMARY_METHOD
        ),
        *random,
    ]

    identifiers = tuple(record["support_id"] for record in all_supports)
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("frozen support IDs collide")
    return {
        "schema_version": "e32_3_frozen_discovery_supports_v1",
        "case_id": case_id,
        "component_ids": nodes,
        "confirmatory_support_size": half_size,
        "deterministic_supports": deterministic,
        "singleton_curve_supports": curve,
        "random_supports": tuple(random),
        "all_supports": tuple(all_supports),
        "ground_truth_used_for_cohort_selection": True,
        "ground_truth_used_for_within_case_selection_or_primary_scoring": False,
    }


def acquisition_supports_from_frozen(
    frozen: Mapping[str, Any],
) -> tuple[FrozenNodeSupport, ...]:
    """Rebuild the canonical acquisition records without trusting raw order."""
    if frozen.get("schema_version") != "e32_3_frozen_discovery_supports_v1":
        raise ValueError("frozen support schema changed")
    nodes = _node_tuple(frozen.get("component_ids"), label="frozen component_ids")
    deterministic = frozen.get("deterministic_supports")
    curve = frozen.get("singleton_curve_supports")
    random = tuple(frozen.get("random_supports", ()))
    if (
        not isinstance(deterministic, Mapping)
        or tuple(deterministic) != DETERMINISTIC_METHOD_ORDER
        or not isinstance(curve, Mapping)
        or tuple(curve) != support_sizes(len(nodes))[1:4]
        or len(random) != RANDOM_CONTROL_REPLICATES
    ):
        raise ValueError("frozen support maps are incomplete or reordered")
    ordered = (
        *(curve[size] for size in support_sizes(len(nodes))[1:4]),
        *(
            deterministic[method]
            for method in DETERMINISTIC_METHOD_ORDER
            if method != PRIMARY_METHOD
        ),
        *random,
    )
    if tuple(frozen.get("all_supports", ())) != ordered:
        raise ValueError("raw all_supports differs from canonical frozen maps")
    converted = []
    for index, record in enumerate(ordered):
        if not isinstance(record, Mapping):
            raise TypeError("frozen support record must be a mapping")
        kind = record.get("kind")
        replicate = record.get("replicate")
        if index >= 7:
            expected_replicate = index - 7
            if (
                kind != "random"
                or record.get("method") != "random_control"
                or replicate != expected_replicate
            ):
                raise ValueError("random support identity/order changed")
        elif kind != "deterministic" or replicate is not None:
            raise ValueError("deterministic support identity changed")
        converted.append(
            FrozenNodeSupport(
                support_id=record["support_id"],
                method=record["method"],
                quantile=float(record["quantile"]),
                selected_nodes=tuple(record["selected_nodes"]),
                kind=kind,
                replicate_index=replicate,
            )
        )
    return tuple(converted)


def validate_split_request_alignment() -> dict[str, Any]:
    """Require protocol and cohort request identities to describe one panel."""
    requests: dict[str, dict[str, Any]] = {}
    for case_id in CASE_ORDER:
        requests[str(case_id)] = {}
        for split in ("discovery", "confirmation"):
            cohort = cohort_split_request(case_id, split)
            protocol = protocol_split_request(case_id, split)
            common = {
                "case_id": protocol.case_id,
                "split": protocol.split,
                "required_count": protocol.required_count,
                "candidate_count": protocol.candidate_count,
                "clean_seed": protocol.clean_seed,
                "corrupt_seed": protocol.corrupt_seed,
            }
            if cohort.__dict__ != common:
                raise RuntimeError("cohort/protocol scientific split requests differ")
            smoke = SMOKE_REQUESTS[case_id]
            if (
                protocol.excluded_smoke_clean_seed != smoke.clean_seed
                or protocol.excluded_smoke_corrupt_seed != smoke.corrupt_seed
                or protocol.excluded_smoke_candidate_count_per_role
                != smoke.sample_count
            ):
                raise RuntimeError("cohort/protocol smoke exclusions differ")
            requests[str(case_id)][split] = common
    return {
        "schema_version": "e32_3_split_request_alignment_v1",
        "case_order": CASE_ORDER,
        "all_requests_equal": True,
        "requests": requests,
    }


def _token_rows(value: Any, *, label: str) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of token rows")
    rows = tuple(tuple(row) for row in value)
    if any(
        not row
        or any(isinstance(token, bool) or not isinstance(token, int) for token in row)
        for row in rows
    ):
        raise ValueError(f"{label} must contain nonempty integer token rows")
    return rows


def validate_global_behavior_screen(
    cases: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit all task-local row identities before any patched intervention."""
    expected_keys = tuple(str(case_id) for case_id in CASE_ORDER)
    if not isinstance(cases, Mapping) or tuple(cases) != expected_keys:
        raise ValueError(f"behavior screen must follow case order {expected_keys!r}")
    audits: dict[str, dict[str, int]] = {}
    total = 0
    for case_id in CASE_ORDER:
        record = cases[str(case_id)]
        if record.get("case_id") != case_id or record.get("behavior_only") is not True:
            raise ValueError(
                "behavior screen identity or intervention boundary changed"
            )
        smoke = _token_rows(record.get("smoke_rows"), label="smoke_rows")
        d_clean = _token_rows(
            record.get("discovery_clean_rows"), label="discovery_clean_rows"
        )
        d_corrupt = _token_rows(
            record.get("discovery_corrupt_rows"), label="discovery_corrupt_rows"
        )
        c_clean = _token_rows(
            record.get("confirmation_clean_rows"), label="confirmation_clean_rows"
        )
        c_corrupt = _token_rows(
            record.get("confirmation_corrupt_rows"),
            label="confirmation_corrupt_rows",
        )
        expected_smoke = EXPECTED_RESPONSE_BLIND_CAPACITY[case_id]["smoke_unique_rows"]
        if (
            len(smoke) != expected_smoke
            or len(d_clean) != 64
            or len(d_corrupt) != 64
            or len(c_clean) != 128
            or len(c_corrupt) != 128
        ):
            raise ValueError("behavior screen row counts differ from frozen panel")
        scientific = (*d_clean, *d_corrupt, *c_clean, *c_corrupt)
        if len(set(smoke)) != len(smoke) or len(set(scientific)) != len(scientific):
            raise RuntimeError("token-row leakage within the global behavior screen")
        if set(smoke).intersection(scientific):
            raise RuntimeError("token-row leakage from excluded smoke rows")
        audits[str(case_id)] = {
            "smoke_unique_rows": len(smoke),
            "discovery_role_rows": len(d_clean) + len(d_corrupt),
            "confirmation_role_rows": len(c_clean) + len(c_corrupt),
            "cross_split_overlap_count": 0,
        }
        total += len(scientific)
    return {
        "schema_version": "e32_3_global_behavior_screen_audit_v1",
        "case_order": CASE_ORDER,
        "all_cases_screened_before_intervention": True,
        "cross_case_token_ids_compared": False,
        "total_scientific_role_rows": total,
        "cases": audits,
    }


def aggregate_actual_query_accounting(
    panel: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Sum observed forwards plus explicitly named pinned EAP backward work."""
    if tuple(panel) != MANDATORY_PANEL_KEYS:
        raise ValueError("actual accounting requires the complete ordered panel")
    totals: dict[str, int] = {}
    by_family: dict[str, dict[str, int]] = {}
    per_panel: dict[str, dict[str, int]] = {}
    for key in MANDATORY_PANEL_KEYS:
        accounting = panel[key].get("query_accounting")
        if not isinstance(accounting, Mapping) or not accounting:
            raise ValueError(f"{key!r} lacks observed query accounting")
        required_families = (
            "eligibility",
            "preparation",
            "atlas_smoke",
            "atlas",
            "exact_fine",
            "eap_ig5",
            *(("set_interventions",) if key[1] == "confirmation" else ()),
        )
        if tuple(accounting) != required_families:
            raise ValueError(f"{key!r} query families differ from frozen accounting")
        panel_totals: dict[str, int] = {}
        for family, counters in accounting.items():
            if not isinstance(counters, Mapping) or not counters:
                raise ValueError("query family accounting must be a nonempty mapping")
            family_totals = by_family.setdefault(family, {})
            for name, value in counters.items():
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        "observed query counters must be nonnegative integers"
                    )
                family_totals[name] = family_totals.get(name, 0) + value
                totals[name] = totals.get(name, 0) + value
                panel_totals[name] = panel_totals.get(name, 0) + value
        exact_panel = {
            name: panel_totals.get(name, 0)
            for name in (
                "physical_forwards",
                "model_examples",
                "backward_passes_inferred_from_pinned_control_flow",
                "gradient_examples",
            )
        }
        expected_panel = EXPECTED_QUERY_ACCOUNTING[key[0]][key[1]]
        if exact_panel != expected_panel:
            raise RuntimeError(
                f"{key!r} actual work differs from frozen query accounting: "
                f"expected {expected_panel!r}, observed {exact_panel!r}"
            )
        per_panel[f"case{key[0]}_{key[1]}"] = exact_panel
    exact_overall = {
        name: totals.get(name, 0)
        for name in (
            "physical_forwards",
            "model_examples",
            "backward_passes_inferred_from_pinned_control_flow",
            "gradient_examples",
        )
    }
    if exact_overall != EXPECTED_QUERY_ACCOUNTING["overall"]:
        raise RuntimeError("overall actual work differs from frozen query accounting")
    return {
        "schema_version": "e32_3_actual_query_accounting_v1",
        "panel_count": len(MANDATORY_PANEL_KEYS),
        "by_family": by_family,
        "per_panel": per_panel,
        "totals": totals,
        "status": ("observed_forward_counters_plus_pinned_eap_backward_control_flow"),
    }


def validate_exact_panel_alignment(bundle: Mapping[str, Any]) -> None:
    """Bind whole, exact, and EAP outputs to the same selected input panel."""
    fine_names = tuple(bundle.get("fine_edge_names", ()))
    eap_names = tuple(bundle.get("eap_fine_edge_names", ()))
    mapping = tuple(bundle.get("fine_to_whole_indices", ()))
    if (
        not fine_names
        or fine_names != eap_names
        or len(fine_names) != len(mapping)
        or len(set(fine_names)) != len(fine_names)
    ):
        raise ValueError("same-panel exact/EAP fine identities differ")
    clean = torch.as_tensor(bundle.get("clean_output"), device="cpu")
    corrupt = torch.as_tensor(bundle.get("corrupt_output"), device="cpu")
    exact_clean = torch.as_tensor(bundle.get("exact_clean_output"), device="cpu")
    exact_corrupt = torch.as_tensor(bundle.get("exact_corrupt_output"), device="cpu")
    if (
        clean.shape != exact_clean.shape
        or corrupt.shape != exact_corrupt.shape
        or not bool(torch.isfinite(clean).all())
        or not bool(torch.isfinite(corrupt).all())
        or not torch.allclose(clean, exact_clean, atol=1e-6, rtol=0.0)
        or not torch.allclose(corrupt, exact_corrupt, atol=1e-6, rtol=0.0)
    ):
        raise ValueError("exact baselines differ from whole-atlas canonical endpoints")


def _json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


def score_complete_panel(
    panel: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay the frozen pure scorer over every mandatory D/C artifact."""
    if tuple(panel) != MANDATORY_PANEL_KEYS:
        raise ValueError("E32.3 scoring requires the complete ordered panel")
    case_results: dict[str, dict[str, Any]] = {}
    for case_id in CASE_ORDER:
        discovery = panel[(case_id, "discovery")]
        confirmation = panel[(case_id, "confirmation")]
        validate_exact_panel_alignment(discovery)
        validate_exact_panel_alignment(confirmation)
        if (
            discovery.get("case_id") != case_id
            or discovery.get("split") != "discovery"
            or confirmation.get("case_id") != case_id
            or confirmation.get("split") != "confirmation"
        ):
            raise ValueError("split artifact identity differs from panel key")
        nodes = tuple(discovery["component_ids"])
        edges = tuple(tuple(edge) for edge in discovery["whole_edges"])
        if (
            tuple(confirmation["component_ids"]) != nodes
            or tuple(tuple(edge) for edge in confirmation["whole_edges"]) != edges
        ):
            raise ValueError("discovery/confirmation universes differ")
        if tuple(discovery["fine_edge_names"]) != tuple(
            confirmation["fine_edge_names"]
        ) or tuple(discovery["fine_to_whole_indices"]) != tuple(
            confirmation["fine_to_whole_indices"]
        ):
            raise ValueError("discovery/confirmation exact fine identities differ")
        score = evaluate_sparse_core_case(
            case_id=case_id,
            component_ids=nodes,
            whole_edges=edges,
            fine_to_whole_indices=discovery["fine_to_whole_indices"],
            discovery_selector_scores=discovery["selector_scores"],
            confirmation_selector_scores=confirmation["selector_scores"],
            discovery_raw_mass=discovery["raw_mass"],
            confirmation_raw_mass=confirmation["raw_mass"],
            discovery_exact_mass=discovery["exact_mass"],
            confirmation_exact_mass=confirmation["exact_mass"],
            confirmation_clean_output=confirmation["clean_output"],
            confirmation_all_corrupt_into_clean_output=confirmation[
                "all_corrupt_into_clean_output"
            ],
            confirmation_clean_target_indices=confirmation["clean_target_indices"],
            deterministic_set_outputs=confirmation["deterministic_set_outputs"],
            random_set_outputs=confirmation["random_set_outputs"],
        )
        score["singleton_support_curve"] = evaluate_singleton_support_curve(
            component_ids=nodes,
            discovery_singleton_scores=discovery["selector_scores"][PRIMARY_METHOD],
            clean_output=confirmation["clean_output"],
            all_corrupt_into_clean_output=confirmation["all_corrupt_into_clean_output"],
            clean_target_indices=confirmation["clean_target_indices"],
            set_outputs_by_support_size=confirmation["singleton_curve_set_outputs"],
        )
        case_results[str(case_id)] = score
    verdict = evaluate_panel_verdict(case_results)
    return _json_ready(
        {
            "schema_version": "e32_3_sparse_core_panel_score_v1",
            "case_order": CASE_ORDER,
            "cases": case_results,
            "verdict": verdict,
            "ground_truth_used_for_cohort_selection": True,
            "ground_truth_used_for_within_case_selection_or_primary_scoring": False,
        }
    )


__all__ = [
    "MANDATORY_PANEL_KEYS",
    "aggregate_actual_query_accounting",
    "freeze_discovery_supports",
    "score_complete_panel",
    "validate_global_behavior_screen",
]
