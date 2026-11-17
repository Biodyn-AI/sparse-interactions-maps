# ruff: noqa: EXE002
"""Response-blind row custody and node-first acquisition helpers for E32.2.

Live model orchestration is added below these pure helpers.  Keeping row and
shortlist logic small makes the two leakage boundaries independently testable:
old E32.1 candidate rows are excluded before behavioral screening, and the
node shortlist depends only on singleton responses, never on circuit labels.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import torch

from experiments.e32_interpbench.acquisition import (
    acquire_official_pair_atlas,
    build_case_patch_runner,
)
from experiments.e32_interpbench.adapter import (
    BehaviorallyEligiblePair,
    behaviorally_eligible_pair,
    generate_official_pair,
    patchable_internal_universe,
)
from experiments.e32_interpbench.e32_2_directed_replacement import (
    DirectedReplacementRun,
    run_directed_replacement,
)
from experiments.e32_interpbench.e32_2_protocol import split_request
from experiments.e32_interpbench.eap_baseline import (
    OfficialEAPRun,
    official_eap_run_to_dict,
    run_official_eap_baseline,
)

Edge = tuple[str, str]

# Full candidate pools, not merely the selected E32.1 rows.  Each pool has a
# clean and independent-source role, hence 2 * (80 + 160) generated token rows.
PRIOR_E32_1_CANDIDATE_POOLS = (
    (80, 42_032, 42_033),
    (160, 42_034, 42_035),
)


def token_rows(value: Any, *, label: str) -> tuple[tuple[int, ...], ...]:
    """Return immutable token rows after strict integer/rank validation."""
    tensor = torch.as_tensor(value, device="cpu")
    if tensor.ndim != 2 or min(tensor.shape) < 1:
        raise ValueError(f"{label} must be a nonempty rank-two token tensor")
    if tensor.dtype == torch.bool or tensor.is_floating_point():
        raise ValueError(f"{label} must contain integer token IDs")
    return tuple(tuple(int(token) for token in row) for row in tensor.tolist())


def prior_candidate_token_rows(
    case: Any,
    *,
    generate_pair: Callable[..., Any] = generate_official_pair,
) -> tuple[tuple[int, ...], ...]:
    """Regenerate every clean/source candidate row used by E32.1.

    ``generate_pair`` exists for model-free tests.  The canonical production
    path never exposes it and always uses the pinned official generator.
    """
    if not callable(generate_pair):
        raise TypeError("generate_pair must be callable")
    rows: list[tuple[int, ...]] = []
    for sample_count, clean_seed, corrupt_seed in PRIOR_E32_1_CANDIDATE_POOLS:
        pair = generate_pair(
            case,
            sample_count=sample_count,
            clean_seed=clean_seed,
            corrupt_seed=corrupt_seed,
        )
        rows.extend(token_rows(pair.clean_inputs, label="prior clean candidate inputs"))
        rows.extend(
            token_rows(pair.corrupt_inputs, label="prior source candidate inputs")
        )
    expected = 2 * sum(pool[0] for pool in PRIOR_E32_1_CANDIDATE_POOLS)
    if len(rows) != expected:
        raise RuntimeError("prior candidate-pool regeneration changed row count")
    return tuple(rows)


def assert_role_rows_fresh(
    *,
    clean_inputs: Any,
    corrupt_inputs: Any,
    excluded_rows: Sequence[Sequence[int]],
    expected_count: int,
    role: str,
) -> dict[str, int | str]:
    """Require clean/source uniqueness and zero overlap with all exclusions."""
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
    ):
        raise ValueError("expected_count must be a positive integer")
    if not isinstance(role, str) or not role:
        raise ValueError("role must be a nonempty string")
    clean = token_rows(clean_inputs, label=f"{role} clean inputs")
    source = token_rows(corrupt_inputs, label=f"{role} source inputs")
    if len(clean) != expected_count or len(source) != expected_count:
        raise ValueError(f"{role} selected row count changed")
    selected = (*clean, *source)
    selected_set = set(selected)
    normalized_exclusions = {
        tuple(int(token) for token in row) for row in excluded_rows
    }
    overlap = selected_set.intersection(normalized_exclusions)
    if len(selected_set) != len(selected) or overlap:
        raise RuntimeError(
            "token-row leakage: clean/source roles must be unique and fresh"
        )
    return {
        "policy": (
            "all_clean_source_rows_unique_and_disjoint_from_full_prior_"
            "candidate_pools_and_earlier_e32_2_roles"
        ),
        "role": role,
        "selected_unique_role_rows": len(selected_set),
        "excluded_prior_row_count": len(normalized_exclusions),
        "overlap_count": 0,
    }


def select_top_singleton_nodes(
    *,
    node_ids: Sequence[str],
    clean_output: Any,
    singleton_outputs: Any,
    selected_count: int,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Select nodes by mean-squared singleton displacement.

    Inputs are the scientific class-centered, non-BOS response tensors.  Ties
    use the frozen node-universe order, so no circuit label or case-specific
    branch can affect the support.
    """
    nodes = tuple(node_ids)
    if (
        not nodes
        or len(nodes) != len(set(nodes))
        or any(not isinstance(node, str) or not node for node in nodes)
    ):
        raise ValueError("node_ids must be nonempty, unique strings")
    if (
        isinstance(selected_count, bool)
        or not isinstance(selected_count, int)
        or not 1 <= selected_count <= len(nodes)
    ):
        raise ValueError("selected_count must lie within the node universe")
    clean = torch.as_tensor(clean_output, dtype=torch.float64, device="cpu")
    singleton = torch.as_tensor(singleton_outputs, dtype=torch.float64, device="cpu")
    if clean.ndim != 3 or tuple(singleton.shape) != (len(nodes), *clean.shape):
        raise ValueError("outputs must have [B,P,C] and [node,B,P,C] scientific shapes")
    if not bool(torch.isfinite(clean).all() and torch.isfinite(singleton).all()):
        raise ValueError("scientific singleton outputs must be finite")
    energy = (singleton - clean.unsqueeze(0)).square().mean(dim=(1, 2, 3))
    ranked = sorted(range(len(nodes)), key=lambda index: (-float(energy[index]), index))
    selected = tuple(nodes[index] for index in ranked[:selected_count])
    return selected, tuple(float(value) for value in energy)


def induced_edge_subsequence(
    edge_universe: Sequence[Edge], selected_nodes: Sequence[str]
) -> tuple[Edge, ...]:
    """Return topology-valid selected-node edges in frozen universe order."""
    edges = tuple(tuple(edge) for edge in edge_universe)
    if (
        not edges
        or len(edges) != len(set(edges))
        or any(
            len(edge) != 2 or not all(isinstance(node, str) and node for node in edge)
            for edge in edges
        )
    ):
        raise ValueError("edge_universe must contain unique directed string pairs")
    nodes = tuple(selected_nodes)
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("selected_nodes must be nonempty and unique")
    selected = set(nodes)
    if not selected.issubset({node for edge in edges for node in edge}):
        raise ValueError("selected_nodes escape the candidate universe")
    return tuple(edge for edge in edges if edge[0] in selected and edge[1] in selected)


class _ForwardObserver:
    """Observe live EAP work at the unwrapped model embedding boundary."""

    def __init__(self) -> None:
        self.physical_forwards = 0
        self.model_examples = 0
        self.no_grad_forwards = 0
        self.grad_enabled_forwards = 0

    def __call__(self, _module: Any, positional: tuple[Any, ...]) -> None:
        if not positional or not isinstance(positional[0], torch.Tensor):
            raise RuntimeError("observed EAP forward lacks tensor inputs")
        inputs = positional[0]
        if inputs.ndim < 1 or inputs.shape[0] < 1:
            raise RuntimeError("observed EAP forward lacks a nonempty batch")
        self.physical_forwards += 1
        self.model_examples += int(inputs.shape[0])
        if torch.is_grad_enabled():
            self.grad_enabled_forwards += 1
        else:
            self.no_grad_forwards += 1


def directed_run_to_dict(run: DirectedReplacementRun) -> dict[str, Any]:
    """Remove live AutoCircuit objects while preserving every scientific tensor."""
    if not isinstance(run, DirectedReplacementRun):
        raise TypeError("run must be DirectedReplacementRun")
    grouping = run.grouping
    return {
        "schema_version": "e32_2_directed_replacement_envelope_v1",
        "case_id": run.case_id,
        "settings": dict(run.settings),
        "grouping": {
            "case_id": grouping.case_id,
            "edge_order": grouping.edge_order,
            "groups": tuple(
                {
                    "promoted_edge": group.promoted_edge,
                    "fine_edge_names": group.fine_edge_names,
                    "collision_count": group.collision_count,
                }
                for group in grouping.groups
            ),
            "fine_edge_count": grouping.fine_edge_count,
            "internal_fine_edge_count": grouping.internal_fine_edge_count,
            "excluded_boundary_fine_edge_count": (
                grouping.excluded_boundary_fine_edge_count
            ),
        },
        "requested_promoted_edges": run.requested_promoted_edges,
        "fine_edge_names": run.fine_edge_names,
        "fine_to_promoted_indices": run.fine_to_promoted_indices,
        "clean_baseline_logits": run.clean_baseline_logits,
        "corrupt_baseline_logits": run.corrupt_baseline_logits,
        "fine_clean_recipient_corrupt_donor_logits": (
            run.fine_clean_recipient_corrupt_donor_logits
        ),
        "fine_corrupt_recipient_clean_donor_logits": (
            run.fine_corrupt_recipient_clean_donor_logits
        ),
        "clean_recipient_target_ce_deltas": (run.clean_recipient_target_ce_deltas),
        "corrupt_recipient_target_ce_deltas": (run.corrupt_recipient_target_ce_deltas),
        "grouped_clean_recipient_corrupt_donor_logits": (
            run.grouped_clean_recipient_corrupt_donor_logits
        ),
        "grouped_corrupt_recipient_clean_donor_logits": (
            run.grouped_corrupt_recipient_clean_donor_logits
        ),
        "grouped_clean_recipient_target_ce_deltas": (
            run.grouped_clean_recipient_target_ce_deltas
        ),
        "grouped_corrupt_recipient_target_ce_deltas": (
            run.grouped_corrupt_recipient_target_ce_deltas
        ),
        "query_accounting": asdict(run.query_accounting),
    }


def run_eap_with_observed_accounting(
    *,
    case_id: int,
    model: Any,
    case: Any,
    eligible_pair: BehaviorallyEligiblePair,
    method: str,
) -> dict[str, Any]:
    """Run the pinned official baseline and verify actual forward work."""
    if method not in {"eap", "eap_ig"}:
        raise ValueError("method must be 'eap' or 'eap_ig'")
    boundary = getattr(model, "embed", None)
    if not hasattr(boundary, "register_forward_pre_hook"):
        raise TypeError("fresh EAP model lacks an observable embedding boundary")
    observer = _ForwardObserver()
    handle = boundary.register_forward_pre_hook(observer)
    try:
        run = run_official_eap_baseline(
            case_id=case_id,
            model=model,
            case=case,
            eligible_pair=eligible_pair,
            method=method,
            device="cpu",
        )
    finally:
        handle.remove()
    if not isinstance(run, OfficialEAPRun):
        raise TypeError("official EAP bridge returned an unexpected result")
    expected = run.query_accounting
    if (
        observer.physical_forwards != expected.physical_model_forwards
        or observer.model_examples != expected.model_examples
        or observer.grad_enabled_forwards != expected.gradient_forwards
        or observer.no_grad_forwards != expected.source_ablation_forwards
    ):
        raise RuntimeError("observed EAP work differs from official accounting")
    return {
        "run": official_eap_run_to_dict(run),
        "observed_query_accounting": {
            "physical_forwards": observer.physical_forwards,
            "model_examples": observer.model_examples,
            "no_grad_forwards": observer.no_grad_forwards,
            "grad_enabled_forwards": observer.grad_enabled_forwards,
            "backward_passes": expected.backward_passes,
            "matches_official_control_flow": True,
        },
    }


def _validated_request(request: Any) -> None:
    required = (
        "case_id",
        "split",
        "required_count",
        "candidate_count",
        "clean_seed",
        "corrupt_seed",
        "smoke_condition_count",
        "smoke_condition_batch_size",
        "atlas_condition_batch_size",
    )
    missing = tuple(name for name in required if not hasattr(request, name))
    if missing:
        raise TypeError(f"split request lacks fields: {missing!r}")
    if request.case_id not in {2, 21} or request.split not in {
        "discovery",
        "confirmation",
    }:
        raise ValueError("E32.2 acquisition is restricted to cases 2/21 and D/C")
    frozen = split_request(request.case_id, request.split)
    observed = {name: getattr(request, name) for name in required}
    expected = {name: getattr(frozen, name) for name in required}
    if observed != expected:
        raise ValueError("split request differs from the exact frozen protocol")


def acquire_split_bundle(
    *,
    request: Any,
    case: Any,
    selection_model: Any,
    directed_model: Any,
    eap_model: Any,
    eap_ig_model: Any,
    excluded_rows: Sequence[Sequence[int]],
    selected_node_count: int,
    discovery_selected_nodes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Acquire every frozen method on one shared, behaviorally eligible panel.

    Model arguments make lifecycle ownership explicit: raw component patching,
    exact directed replacement, EAP, and EAP-IG cannot accidentally share an
    AutoCircuit-mutated model.  The canonical CLI supplies only freshly loaded
    pinned checkpoints and exposes no callback injection point.
    """
    _validated_request(request)
    if request.split == "discovery" and discovery_selected_nodes is not None:
        raise ValueError("discovery cannot receive an earlier node shortlist")
    if request.split == "confirmation" and discovery_selected_nodes is None:
        raise ValueError("confirmation requires the discovery node shortlist")
    eligible = behaviorally_eligible_pair(
        selection_model,
        case,
        required_count=request.required_count,
        candidate_count=request.candidate_count,
        clean_seed=request.clean_seed,
        corrupt_seed=request.corrupt_seed,
        device="cpu",
        excluded_token_rows=excluded_rows,
        require_unique_selected_token_rows=True,
    )
    freshness = assert_role_rows_fresh(
        clean_inputs=eligible.pair.clean_inputs,
        corrupt_inputs=eligible.pair.corrupt_inputs,
        excluded_rows=excluded_rows,
        expected_count=request.required_count,
        role=f"case{request.case_id}_{request.split}",
    )
    runner = build_case_patch_runner(
        case_id=request.case_id,
        model=selection_model,
        condition_batch_size=request.atlas_condition_batch_size,
    )
    raw_atlas = acquire_official_pair_atlas(
        case_id=request.case_id,
        runner=runner,
        official_pair=eligible,
        smoke_condition_count=request.smoke_condition_count,
        smoke_condition_batch_size=request.smoke_condition_batch_size,
    )
    node_ids = tuple(raw_atlas["component_ids"])
    selected_nodes, singleton_energies = select_top_singleton_nodes(
        node_ids=node_ids,
        clean_output=raw_atlas["scientific_condition_outputs"][0],
        singleton_outputs=raw_atlas["scientific_singleton_outputs"],
        selected_count=selected_node_count,
    )
    _unused_nodes, full_edges = patchable_internal_universe(
        int(selection_model.cfg.n_layers), int(selection_model.cfg.n_heads)
    )
    if tuple(raw_atlas["directed_candidate_edges"]) != full_edges:
        raise RuntimeError("raw atlas and adapter edge universes differ")
    frozen_discovery_nodes = (
        selected_nodes
        if request.split == "discovery"
        else tuple(discovery_selected_nodes or ())
    )
    induced_edges = induced_edge_subsequence(full_edges, frozen_discovery_nodes)
    requested_directed_edges = induced_edges if request.split == "discovery" else None
    directed = run_directed_replacement(
        case_id=request.case_id,
        model=directed_model,
        case=case,
        eligible_pair=eligible,
        requested_promoted_edges=requested_directed_edges,
        include_grouped_promoted_sensitivity=False,
        device="cpu",
    )
    eap = run_eap_with_observed_accounting(
        case_id=request.case_id,
        model=eap_model,
        case=case,
        eligible_pair=eligible,
        method="eap",
    )
    eap_ig = run_eap_with_observed_accounting(
        case_id=request.case_id,
        model=eap_ig_model,
        case=case,
        eligible_pair=eligible,
        method="eap_ig",
    )
    selected_rows = (
        *token_rows(eligible.pair.clean_inputs, label="clean"),
        *token_rows(eligible.pair.corrupt_inputs, label="source"),
    )
    return {
        "schema_version": "e32_2_split_acquisition_v1",
        "case_id": request.case_id,
        "split": request.split,
        "request": {
            field: getattr(request, field)
            for field in (
                "case_id",
                "split",
                "required_count",
                "candidate_count",
                "clean_seed",
                "corrupt_seed",
                "smoke_condition_count",
                "smoke_condition_batch_size",
                "atlas_condition_batch_size",
            )
        },
        "row_custody": freshness,
        "selected_token_rows": selected_rows,
        "raw_pair_atlas": raw_atlas,
        "split_singleton_node_energies": singleton_energies,
        "split_top_half_nodes": selected_nodes,
        "discovery_frozen_nodes": frozen_discovery_nodes,
        "discovery_induced_edges": induced_edges,
        "directed_replacement": directed_run_to_dict(directed),
        "official_eap": eap,
        "official_eap_ig": eap_ig,
        "actual_query_accounting": {
            "behavior_and_raw_atlas": dict(
                raw_atlas["total_query_accounting_including_eligibility"]
            ),
            "directed_replacement": asdict(directed.query_accounting),
            "official_eap": dict(eap["observed_query_accounting"]),
            "official_eap_ig": dict(eap_ig["observed_query_accounting"]),
            "operational_budget_status": "reported_separately_by_protocol",
        },
    }


__all__ = [
    "PRIOR_E32_1_CANDIDATE_POOLS",
    "acquire_split_bundle",
    "assert_role_rows_fresh",
    "directed_run_to_dict",
    "induced_edge_subsequence",
    "prior_candidate_token_rows",
    "run_eap_with_observed_accounting",
    "select_top_singleton_nodes",
    "token_rows",
]
