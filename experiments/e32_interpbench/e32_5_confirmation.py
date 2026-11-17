# ruff: noqa: EXE002
"""Pure E32.5 support freezing, confirmation acquisition, and raw scoring.

The development/discovery pipeline is intentionally imported read-only.  This
module owns the post-discovery lifecycle: deterministically freezing one
primary, four same-size comparators, and 256 random controls; acquiring every
``S`` and literal ``U\S`` together on untouched confirmation rows; and replaying
all scores from raw condition tensors after the durable score barrier.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from experiments.e32_interpbench.adapter import BehaviorallyEligiblePair, OfficialDataPair
from experiments.e32_interpbench.e32_4_edge_sets import (
    EdgeAblation,
    EdgeConditionPlan,
    EdgeSetAcquisition,
    FrozenEdgeSupport,
    InducedEdgeUniverse,
    build_edge_condition_plan,
    edge_set_acquisition_to_dict,
    run_edge_set_replacement,
)
from experiments.e32_interpbench.e32_5_cohort import (
    CASE_MANIFESTS,
    SPLIT_ORDER_BY_CASE,
    load_verified_model,
    split_request,
)
from experiments.e32_interpbench.e32_5_pipeline import (
    PipelineStageAccounting,
    _universe_from_record,
    _validated_raw_edge_acquisition,
    discovery_scoring_slices,
    replay_e32_5_discovery_search,
)
from experiments.e32_interpbench.e32_5_protocol import (
    CASE_ORDER,
    CONFIRMATION_EXAMPLE_COUNT,
    CONTEXTUAL_SIGNED_STATIC_METHOD,
    CONTEXTUAL_UNSIGNED_MASS_METHOD,
    DETERMINISTIC_BASELINE_ORDER,
    EXPERIMENT_ID,
    GENERALIZATION_CASE_ORDER,
    PRIMARY_METHOD,
    RANDOM_CONTROL_95TH_INDEX,
    RANDOM_CONTROL_REPLICATES,
    confirmation_support_size,
    deterministic_random_supports,
)
from experiments.e32_interpbench.e32_5_scoring import (
    evaluate_case,
    evaluate_panel,
    evaluate_repair_case,
    score_end_to_end_alignment,
    score_exact_group,
)
from experiments.e32_interpbench.e32_5_selector import initial_static_support

ConfirmationSplit = Literal["repair_confirmation", "confirmation"]
FROZEN_METHOD_ORDER = (PRIMARY_METHOD, *DETERMINISTIC_BASELINE_ORDER)


@dataclass(frozen=True)
class ConfirmationQueryAccounting:
    behavioral_eligibility: PipelineStageAccounting
    simultaneous_edge_sets: PipelineStageAccounting
    total: PipelineStageAccounting


@dataclass(frozen=True)
class E32_5ConfirmationBundle:
    case_id: int
    split: ConfirmationSplit
    eligible_panel: BehaviorallyEligiblePair
    authorization: Mapping[str, Any]
    frozen_support: Mapping[str, Any]
    universe: InducedEdgeUniverse
    acquisition: EdgeSetAcquisition
    query_accounting: ConfirmationQueryAccounting


@dataclass(frozen=True)
class ConfirmationScoringInputs:
    case_id: int
    split: ConfirmationSplit
    clean_output: torch.Tensor
    all_internal_output: torch.Tensor
    actual_corrupt_output: torch.Tensor
    clean_target_indices: torch.Tensor
    supports: tuple[FrozenEdgeSupport, ...]
    selected_outputs: torch.Tensor
    complement_outputs: torch.Tensor
    frozen_support: Mapping[str, Any]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _universe_record(universe: InducedEdgeUniverse) -> dict[str, Any]:
    return {
        "case_id": universe.case_id,
        "selected_nodes": list(universe.selected_nodes),
        "whole_edges": [list(edge) for edge in universe.whole_edges],
        "fine_edge_names": list(universe.fine_edge_names),
        "fine_to_whole_indices": list(universe.fine_to_whole_indices),
        "induced_whole_indices": list(universe.induced_whole_indices),
        "induced_fine_indices": list(universe.induced_fine_indices),
    }


def _canonical_support(
    fine_edge_ids: tuple[str, ...], value: Sequence[str], *, support_size: int
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("frozen support must be a sequence")
    supplied = tuple(value)
    selected = frozenset(supplied)
    canonical = tuple(edge for edge in fine_edge_ids if edge in selected)
    if (
        len(supplied) != support_size
        or len(selected) != support_size
        or canonical != supplied
    ):
        raise ValueError("frozen support differs from canonical full-U order or size")
    return canonical


def _validate_frozen_support(
    value: Any, *, expected_case_id: int | None = None
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("frozen support must be a mapping")
    case_id = value.get("case_id")
    if (
        value.get("schema_version") != "e32_5_frozen_support_v1"
        or isinstance(case_id, bool)
        or not isinstance(case_id, int)
        or case_id not in CASE_ORDER
        or (expected_case_id is not None and case_id != expected_case_id)
    ):
        raise ValueError("frozen support identity changed")
    universe_record = value.get("universe")
    universe = _universe_from_record(universe_record)
    if universe.case_id != case_id or value.get("universe_sha256") != _json_sha256(
        _universe_record(universe)
    ):
        raise ValueError("frozen support full-U universe identity changed")
    fine_edge_ids = universe.fine_edge_names
    support_size = confirmation_support_size(case_id)
    if value.get("support_size") != support_size or tuple(
        value.get("method_order", ())
    ) != FROZEN_METHOD_ORDER:
        raise ValueError("frozen support method order or cardinality changed")
    raw_methods = value.get("method_supports")
    if not isinstance(raw_methods, Mapping) or tuple(raw_methods) != FROZEN_METHOD_ORDER:
        raise ValueError("frozen deterministic support panel changed")
    methods = {
        method: _canonical_support(
            fine_edge_ids, raw_methods[method], support_size=support_size
        )
        for method in FROZEN_METHOD_ORDER
    }
    if tuple(value.get("support", ())) != methods[PRIMARY_METHOD]:
        raise ValueError("frozen primary support alias changed")
    raw_random = tuple(value.get("random_supports", ()))
    random_supports = tuple(
        _canonical_support(fine_edge_ids, support, support_size=support_size)
        for support in raw_random
    )
    expected_random = deterministic_random_supports(
        case_id, fine_edge_ids, support_size
    )
    if (
        len(random_supports) != RANDOM_CONTROL_REPLICATES
        or random_supports != expected_random
    ):
        raise ValueError("frozen random-control panel changed")
    normalized = {
        "schema_version": "e32_5_frozen_support_v1",
        "case_id": case_id,
        "source_split": value.get("source_split"),
        "support_size": support_size,
        "support": list(methods[PRIMARY_METHOD]),
        "method_order": list(FROZEN_METHOD_ORDER),
        "method_supports": {
            method: list(methods[method]) for method in FROZEN_METHOD_ORDER
        },
        "random_supports": [list(support) for support in random_supports],
        "random_control_count": RANDOM_CONTROL_REPLICATES,
        "universe": _universe_record(universe),
        "universe_sha256": _json_sha256(_universe_record(universe)),
    }
    expected_source = SPLIT_ORDER_BY_CASE[case_id][0]
    if normalized["source_split"] != expected_source:
        raise ValueError("frozen support source split changed")
    return normalized


def freeze_e32_5_support(
    *, case_id: int, raw_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    """Purely freeze the primary, four comparators, and random panel from D."""
    if case_id not in CASE_ORDER or raw_bundle.get("case_id") != case_id:
        raise ValueError("support freeze case identity changed")
    slices = discovery_scoring_slices(raw_bundle)
    replay = replay_e32_5_discovery_search(raw_bundle)
    if replay is None or slices.eap_ig5_signed_scores is None:
        raise ValueError(
            "scientific_fail_uninformative_internal_endpoint_before_search_or_C"
        )
    base_record = raw_bundle["scientific_payload"]["adaptive"]["base"]
    universe = _universe_from_record(base_record["universe"])
    fine_edge_ids = slices.fine_edge_ids
    if universe.fine_edge_names != fine_edge_ids:
        raise ValueError("discovery replay and raw full-U axes differ")
    support_size = confirmation_support_size(case_id)
    method_vectors = {
        CONTEXTUAL_SIGNED_STATIC_METHOD: replay.singleton_scores.signed_score,
        CONTEXTUAL_UNSIGNED_MASS_METHOD: replay.singleton_scores.unsigned_exact_mass,
        "eap_absolute": slices.regular_eap_signed_scores.abs(),
        "eap_ig5_absolute": slices.eap_ig5_signed_scores.abs(),
    }
    method_supports = {
        PRIMARY_METHOD: replay.validation.selected_support,
        **{
            method: initial_static_support(
                fine_edge_ids=fine_edge_ids,
                directional_scores=method_vectors[method],
                support_size=support_size,
            )
            for method in DETERMINISTIC_BASELINE_ORDER
        },
    }
    random_supports = deterministic_random_supports(
        case_id, fine_edge_ids, support_size
    )
    record = {
        "schema_version": "e32_5_frozen_support_v1",
        "case_id": case_id,
        "source_split": raw_bundle.get("split"),
        "support_size": support_size,
        "support": list(method_supports[PRIMARY_METHOD]),
        "method_order": list(FROZEN_METHOD_ORDER),
        "method_supports": {
            method: list(method_supports[method]) for method in FROZEN_METHOD_ORDER
        },
        "random_supports": [list(support) for support in random_supports],
        "random_control_count": RANDOM_CONTROL_REPLICATES,
        "universe": _universe_record(universe),
        "universe_sha256": _json_sha256(_universe_record(universe)),
    }
    return _validate_frozen_support(record, expected_case_id=case_id)


def _support_panel(
    frozen: Mapping[str, Any], universe: InducedEdgeUniverse
) -> tuple[FrozenEdgeSupport, ...]:
    index_by_name = {
        name: index for index, name in enumerate(universe.fine_edge_names)
    }

    def support(
        *,
        support_id: str,
        method: str,
        names: Sequence[str],
        kind: Literal["deterministic", "random"],
        replicate_index: int | None = None,
    ) -> FrozenEdgeSupport:
        indices = tuple(index_by_name[name] for name in names)
        return FrozenEdgeSupport(
            support_id=support_id,
            resolution="fine",
            method=method,
            support_size=len(indices),
            selected_indices=indices,
            kind=kind,
            replicate_index=replicate_index,
        )

    deterministic = tuple(
        support(
            support_id=method,
            method=method,
            names=frozen["method_supports"][method],
            kind="deterministic",
        )
        for method in FROZEN_METHOD_ORDER
    )
    random = tuple(
        support(
            support_id=f"random_control:{replicate:03d}",
            method="uniform_random_control",
            names=names,
            kind="random",
            replicate_index=replicate,
        )
        for replicate, names in enumerate(frozen["random_supports"])
    )
    return (*deterministic, *random)


def _authorization_frozen_support(
    authorization: Any, *, case_id: int
) -> tuple[dict[str, Any], InducedEdgeUniverse]:
    if not isinstance(authorization, Mapping):
        raise TypeError("confirmation authorization must be a mapping")
    source_split, destination_split = SPLIT_ORDER_BY_CASE[case_id]
    if (
        authorization.get("schema_version")
        != f"{EXPERIMENT_ID}_support_authorization_v1"
        or authorization.get("case_id") != case_id
        or authorization.get("source_split") != source_split
        or authorization.get("destination_split") != destination_split
    ):
        raise ValueError("confirmation authorization identity changed")
    frozen = _validate_frozen_support(
        authorization.get("frozen_support"), expected_case_id=case_id
    )
    if authorization.get("frozen_support_sha256") != _json_sha256(frozen):
        raise ValueError("confirmation authorization frozen-support hash changed")
    return frozen, _universe_from_record(frozen["universe"])


def _confirmation_split(case_id: int) -> ConfirmationSplit:
    try:
        value = SPLIT_ORDER_BY_CASE[case_id][1]
    except KeyError as error:
        raise ValueError("confirmation case is outside the frozen cohort") from error
    if value not in {"repair_confirmation", "confirmation"}:
        raise RuntimeError("frozen cohort confirmation split changed")
    return value


def _validate_confirmation_eligible(
    eligible: BehaviorallyEligiblePair, *, case_id: int, split: ConfirmationSplit
) -> None:
    if not isinstance(eligible, BehaviorallyEligiblePair):
        raise TypeError("eligible_pair must be BehaviorallyEligiblePair")
    pair = eligible.pair
    if not isinstance(pair, OfficialDataPair):
        raise TypeError("eligible confirmation pair must be OfficialDataPair")
    tensors = (
        pair.clean_inputs,
        pair.clean_targets,
        pair.corrupt_inputs,
        pair.corrupt_targets,
    )
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("eligible confirmation pair fields must be tensors")
    request = split_request(case_id, split)
    if (
        len(pair.clean_inputs) != CONFIRMATION_EXAMPLE_COUNT
        or tuple(pair.corrupt_inputs.shape) != tuple(pair.clean_inputs.shape)
        or pair.clean_inputs.ndim != 2
        or tuple(pair.clean_targets.shape[:2]) != tuple(pair.clean_inputs.shape)
        or tuple(pair.corrupt_targets.shape) != tuple(pair.clean_targets.shape)
        or len(eligible.selected_candidate_indices) != CONFIRMATION_EXAMPLE_COUNT
        or len(set(eligible.selected_candidate_indices))
        != CONFIRMATION_EXAMPLE_COUNT
        or eligible.candidate_count != request.candidate_count
        or eligible.eligibility_physical_forwards != 2
        or eligible.eligibility_model_examples != 2 * request.candidate_count
        or eligible.output_kind != "categorical"
        or eligible.selection_policy != request.selection_policy
    ):
        raise ValueError("confirmation eligible panel differs from frozen C128 split")


def _sum_stages(*values: PipelineStageAccounting) -> PipelineStageAccounting:
    return PipelineStageAccounting(
        **{
            name: sum(int(getattr(value, name)) for value in values)
            for name in PipelineStageAccounting.__dataclass_fields__
        }
    )


def _confirmation_accounting(
    eligible: BehaviorallyEligiblePair, acquisition: EdgeSetAcquisition
) -> ConfirmationQueryAccounting:
    raw = acquisition.query_accounting
    if not (
        raw.example_count == CONFIRMATION_EXAMPLE_COUNT
        and raw.expected_physical_forwards == raw.observed_physical_forwards
        and raw.expected_model_examples == raw.observed_model_examples
        and raw.observed_no_grad_forwards == raw.expected_physical_forwards
        and raw.observed_grad_enabled_forwards == 0
        and raw.backward_passes == 0
        and raw.mask_context_count == raw.condition_count
        and raw.active_mask_coordinate_checks
        == int(acquisition.plan.fine_masks.sum())
    ):
        raise RuntimeError("confirmation intervention accounting is not exact")
    eligibility = PipelineStageAccounting(
        physical_forwards=eligible.eligibility_physical_forwards,
        model_examples=eligible.eligibility_model_examples,
        no_grad_forwards=eligible.eligibility_physical_forwards,
        grad_enabled_forwards=0,
        gradient_examples=0,
        backward_passes_inferred_from_pinned_control_flow=0,
    )
    edge_sets = PipelineStageAccounting(
        physical_forwards=raw.observed_physical_forwards,
        model_examples=raw.observed_model_examples,
        no_grad_forwards=raw.observed_no_grad_forwards,
        grad_enabled_forwards=raw.observed_grad_enabled_forwards,
        gradient_examples=0,
        backward_passes_inferred_from_pinned_control_flow=raw.backward_passes,
        mask_context_count=raw.mask_context_count,
        active_mask_coordinate_checks=raw.active_mask_coordinate_checks,
    )
    return ConfirmationQueryAccounting(
        behavioral_eligibility=eligibility,
        simultaneous_edge_sets=edge_sets,
        total=_sum_stages(eligibility, edge_sets),
    )


def run_e32_5_confirmation(
    *,
    case_id: int,
    case: Any,
    eligible_pair: BehaviorallyEligiblePair,
    authorization: Mapping[str, Any],
    device: str = "cpu",
) -> E32_5ConfirmationBundle:
    """Open C only after authorization and acquire all 261 sets in one plan."""
    split = _confirmation_split(case_id)
    _validate_confirmation_eligible(eligible_pair, case_id=case_id, split=split)
    frozen, universe = _authorization_frozen_support(
        authorization, case_id=case_id
    )
    supports = _support_panel(frozen, universe)
    if len(supports) != len(FROZEN_METHOD_ORDER) + RANDOM_CONTROL_REPLICATES:
        raise RuntimeError("confirmation support panel cardinality changed")
    plan = build_edge_condition_plan(
        universe, supports, primary_support_id=PRIMARY_METHOD
    )
    manifest = CASE_MANIFESTS[case_id]
    acquisition = run_edge_set_replacement(
        case_id=case_id,
        n_layers=manifest.n_layers,
        n_heads=manifest.n_heads,
        n_ctx=manifest.n_ctx,
        d_vocab_out=manifest.d_vocab_out,
        model=load_verified_model(case_id, device=device),
        case=case,
        eligible_pair=eligible_pair,
        universe=universe,
        plan=plan,
        ablation=EdgeAblation.RESAMPLE,
        device=device,
    )
    if (
        acquisition.case_id != case_id
        or acquisition.ablation is not EdgeAblation.RESAMPLE
        or acquisition.plan.supports != supports
        or acquisition.plan.primary_support_id != PRIMARY_METHOD
    ):
        raise ValueError("live confirmation acquisition changed the authorized plan")
    accounting = _confirmation_accounting(eligible_pair, acquisition)
    return E32_5ConfirmationBundle(
        case_id=case_id,
        split=split,
        eligible_panel=eligible_pair,
        authorization=dict(authorization),
        frozen_support=frozen,
        universe=universe,
        acquisition=acquisition,
        query_accounting=accounting,
    )


def _cpu_clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def _selected_token_rows(pair: OfficialDataPair) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(token) for token in row)
        for tensor in (pair.clean_inputs, pair.corrupt_inputs)
        for row in tensor.detach().cpu().tolist()
    )


def _eligibility_record(value: BehaviorallyEligiblePair) -> dict[str, Any]:
    return {
        name: getattr(value, name)
        for name in (
            "candidate_count",
            "eligible_candidate_count",
            "clean_incorrect_contexts",
            "source_incorrect_contexts",
            "either_incorrect_contexts",
            "selected_candidate_indices",
            "eligibility_physical_forwards",
            "eligibility_model_examples",
            "output_kind",
            "row_disjoint_eligible_candidate_count",
            "row_overlap_excluded_contexts",
            "excluded_token_row_count",
            "selection_policy",
        )
    }


def serialize_e32_5_confirmation_bundle(
    bundle: E32_5ConfirmationBundle, *, split: str | None = None
) -> dict[str, Any]:
    """Return raw C tensors and authorization, with no trusted derived score."""
    if not isinstance(bundle, E32_5ConfirmationBundle):
        raise TypeError("bundle must be E32_5ConfirmationBundle")
    if split is not None and split != bundle.split:
        raise ValueError("requested serialization split differs from C bundle")
    pair = bundle.eligible_panel.pair
    return {
        "schema_version": "e32_5_confirmation_raw_bundle_v1",
        "case_id": bundle.case_id,
        "split": bundle.split,
        "selected_token_rows": _selected_token_rows(pair),
        "query_accounting": asdict(bundle.query_accounting),
        "scientific_payload": {
            "panel": {
                "clean_inputs": _cpu_clone(pair.clean_inputs),
                "clean_targets": _cpu_clone(pair.clean_targets),
                "corrupt_inputs": _cpu_clone(pair.corrupt_inputs),
                "corrupt_targets": _cpu_clone(pair.corrupt_targets),
                "eligibility": _eligibility_record(bundle.eligible_panel),
            },
            "clean_target_indices": _cpu_clone(
                pair.clean_targets.argmax(dim=-1)[:, 1:]
            ),
            "authorization": dict(bundle.authorization),
            "frozen_support": dict(bundle.frozen_support),
            "acquisition": edge_set_acquisition_to_dict(bundle.acquisition),
        },
    }


def _raw_tensor(record: Mapping[str, Any], name: str) -> torch.Tensor:
    value = record.get(name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"raw confirmation {name} must be a tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"raw confirmation {name} contains nonfinite values")
    return value.detach().cpu()


def _normalized_confirmation_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("schema_version") != "e32_5_raw_acquisition_v1":
        return dict(record)
    accounting = record.get("query_accounting")
    if not isinstance(accounting, Mapping) or tuple(accounting) != (
        "families",
        "total",
    ):
        raise ValueError("normalized raw confirmation accounting changed")
    families = accounting.get("families")
    if not isinstance(families, Mapping):
        raise TypeError("normalized raw confirmation families must be a mapping")
    return {
        **dict(record),
        "schema_version": "e32_5_confirmation_raw_bundle_v1",
        "query_accounting": {**dict(families), "total": accounting["total"]},
    }


def _eligible_from_raw_panel(
    panel: Mapping[str, Any], *, case_id: int, split: ConfirmationSplit
) -> BehaviorallyEligiblePair:
    clean_inputs = _raw_tensor(panel, "clean_inputs")
    clean_targets = _raw_tensor(panel, "clean_targets")
    corrupt_inputs = _raw_tensor(panel, "corrupt_inputs")
    corrupt_targets = _raw_tensor(panel, "corrupt_targets")
    eligibility = panel.get("eligibility")
    if not isinstance(eligibility, Mapping):
        raise TypeError("raw confirmation lacks eligibility custody")
    try:
        value = BehaviorallyEligiblePair(
            pair=OfficialDataPair(
                clean_inputs=clean_inputs,
                clean_targets=clean_targets,
                corrupt_inputs=corrupt_inputs,
                corrupt_targets=corrupt_targets,
            ),
            **dict(eligibility),
        )
    except TypeError as error:
        raise ValueError("raw confirmation eligibility custody is malformed") from error
    _validate_confirmation_eligible(value, case_id=case_id, split=split)
    return value


def _plans_equal(left: EdgeConditionPlan, right: EdgeConditionPlan) -> bool:
    return (
        left.universe == right.universe
        and left.supports == right.supports
        and left.condition_ids == right.condition_ids
        and torch.equal(left.fine_masks, right.fine_masks)
        and left.all_internal_condition_index == right.all_internal_condition_index
        and left.induced_condition_index == right.induced_condition_index
        and left.outside_induced_condition_index
        == right.outside_induced_condition_index
        and left.primary_support_id == right.primary_support_id
        and left.primary_within_induced_complement_condition_index
        == right.primary_within_induced_complement_condition_index
        and left.support_references == right.support_references
    )


def confirmation_scoring_inputs(
    raw_record: Mapping[str, Any],
) -> ConfirmationScoringInputs:
    """Rebuild every C scorer input from raw conditions and reject tampering."""
    if not isinstance(raw_record, Mapping):
        raise TypeError("raw confirmation must be a mapping")
    record = _normalized_confirmation_record(raw_record)
    if record.get("schema_version") != "e32_5_confirmation_raw_bundle_v1":
        raise ValueError("raw confirmation schema changed")
    case_id = record.get("case_id")
    if isinstance(case_id, bool) or not isinstance(case_id, int):
        raise TypeError("raw confirmation case_id must be an integer")
    split = _confirmation_split(case_id)
    if record.get("split") != split:
        raise ValueError("raw confirmation split changed")
    payload = record.get("scientific_payload")
    if not isinstance(payload, Mapping):
        raise TypeError("raw confirmation scientific payload must be a mapping")
    panel = payload.get("panel")
    acquisition_record = payload.get("acquisition")
    if not isinstance(panel, Mapping) or not isinstance(acquisition_record, Mapping):
        raise TypeError("raw confirmation panel or acquisition is malformed")
    eligible = _eligible_from_raw_panel(panel, case_id=case_id, split=split)
    expected_rows = _selected_token_rows(eligible.pair)
    if tuple(tuple(row) for row in record.get("selected_token_rows", ())) != expected_rows:
        raise ValueError("raw confirmation selected token rows changed")
    frozen, universe = _authorization_frozen_support(
        payload.get("authorization"), case_id=case_id
    )
    if payload.get("frozen_support") != frozen:
        raise ValueError("raw confirmation frozen-support alias changed")
    supports = _support_panel(frozen, universe)
    expected_plan = build_edge_condition_plan(
        universe, supports, primary_support_id=PRIMARY_METHOD
    )
    validated = _validated_raw_edge_acquisition(
        acquisition_record,
        universe=universe,
        expected_batch=CONFIRMATION_EXAMPLE_COUNT,
        label="confirmation simultaneous panel",
    )
    if not _plans_equal(validated["plan"], expected_plan):
        raise ValueError("raw confirmation plan differs from frozen authorization")
    clean_targets = eligible.pair.clean_targets
    clean_target_indices = _raw_tensor(payload, "clean_target_indices")
    expected_targets = clean_targets.argmax(dim=-1)[:, 1:]
    clean = validated["clean"]
    if (
        clean.ndim != 3
        or tuple(expected_targets.shape) != tuple(clean.shape[:2])
        or not torch.equal(clean_target_indices, expected_targets)
    ):
        raise ValueError("raw confirmation outputs and non-BOS targets misalign")

    observed_edge_stage = PipelineStageAccounting(
        physical_forwards=validated["query_accounting"][
            "observed_physical_forwards"
        ],
        model_examples=validated["query_accounting"]["observed_model_examples"],
        no_grad_forwards=validated["query_accounting"][
            "observed_no_grad_forwards"
        ],
        grad_enabled_forwards=validated["query_accounting"][
            "observed_grad_enabled_forwards"
        ],
        gradient_examples=0,
        backward_passes_inferred_from_pinned_control_flow=validated[
            "query_accounting"
        ]["backward_passes"],
        mask_context_count=validated["query_accounting"]["mask_context_count"],
        active_mask_coordinate_checks=validated["query_accounting"][
            "active_mask_coordinate_checks"
        ],
    )
    observed_eligibility_stage = PipelineStageAccounting(
        physical_forwards=eligible.eligibility_physical_forwards,
        model_examples=eligible.eligibility_model_examples,
        no_grad_forwards=eligible.eligibility_physical_forwards,
        grad_enabled_forwards=0,
        gradient_examples=0,
        backward_passes_inferred_from_pinned_control_flow=0,
    )
    expected_accounting = ConfirmationQueryAccounting(
        behavioral_eligibility=observed_eligibility_stage,
        simultaneous_edge_sets=observed_edge_stage,
        total=_sum_stages(observed_eligibility_stage, observed_edge_stage),
    )
    if record.get("query_accounting") != asdict(expected_accounting):
        raise ValueError("raw confirmation accounting differs from raw work")
    return ConfirmationScoringInputs(
        case_id=case_id,
        split=split,
        clean_output=clean,
        all_internal_output=validated["all_internal_output"],
        actual_corrupt_output=validated["corrupt"],
        clean_target_indices=clean_target_indices,
        supports=supports,
        selected_outputs=validated["selected"],
        complement_outputs=validated["complements"],
        frozen_support=frozen,
    )

