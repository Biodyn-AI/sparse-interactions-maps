# ruff: noqa: EXE002
"""Case-generic exact-fine and EAP-IG5 bridges for E32.3.

E32.2's live bridges intentionally reject case IDs outside its sealed cohort.
E32.3 keeps those sources unchanged and reuses their low-level mask validation,
query accounting, and AutoCircuit runtime.  Only case geometry and generic
fine-to-whole graph validation live here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from experiments.e32_interpbench.adapter import (
    DEFAULT_ENV_DIR,
    BehaviorallyEligiblePair,
    official_prepared_universe,
    patchable_internal_universe,
    promote_official_node,
)
from experiments.e32_interpbench.component_patching import class_center_logits
from experiments.e32_interpbench.e32_2_directed_replacement import (
    DirectedReplacementAccounting,
    PromotedFineEdgeGroup,
    PromotedFineEdgeGrouping,
    _assert_exact_masks,
    _AutoCircuitRuntime,
    _canonical_groups,
    _ForwardObserver,
    _integer_inputs,
    _load_auto_circuit_runtime,
    _mask_coordinate_count,
    _raw_non_bos_logits,
    _validate_patchable_settings,
    directed_replacement_query_accounting,
)
from experiments.e32_interpbench.eap_baseline import (
    DEFAULT_AUTO_CIRCUIT_DIR,
    OFFICIAL_EAP_SETTINGS,
    EAPQueryAccounting,
    EAPSettings,
    FineEdgeScore,
    eap_query_accounting,
    expected_auto_circuit_counts,
    extract_fine_edge_scores,
    official_eap_settings,
    verify_pinned_eap_runtime,
)

Edge = tuple[str, str]


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _fine_target_slot(target: str) -> tuple[str, int]:
    for slot, letter in enumerate("qkv"):
        if f"hook_{letter}_input" in target:
            return "attention_head", slot
    if "hook_mlp_in" in target:
        return "mlp", 0
    raise ValueError(f"unexpected internal fine target: {target!r}")


def build_generic_fine_edge_grouping(
    *,
    case_id: int,
    n_layers: int,
    n_heads: int,
    fine_scores: Sequence[FineEdgeScore],
    live_edges_by_name: Mapping[str, Any],
) -> PromotedFineEdgeGrouping:
    """Prove exact AutoCircuit-to-whole-DAG coverage for any pinned geometry."""
    _positive_integer(case_id, label="case_id")
    layer_count = _positive_integer(n_layers, label="n_layers")
    head_count = _positive_integer(n_heads, label="n_heads")
    scores = tuple(fine_scores)
    expected = expected_auto_circuit_counts(layer_count, head_count)
    if len(scores) != expected.fine_edges or any(
        not isinstance(score, FineEdgeScore) for score in scores
    ):
        raise ValueError("fine scores differ from the exact AutoCircuit graph size")
    names = tuple(score.fine_edge_name for score in scores)
    if len(names) != len(set(names)):
        raise ValueError("fine-edge names must be unique")
    if set(live_edges_by_name) != set(names) or len(live_edges_by_name) != len(names):
        raise ValueError("live fine-edge names differ from the complete score graph")
    if len({id(edge) for edge in live_edges_by_name.values()}) != len(
        live_edges_by_name
    ) and any(not isinstance(edge, str) for edge in live_edges_by_name.values()):
        # Immutable scalar/string stand-ins are allowed in dependency-free
        # tests and EAP packaging; live acquisition uses distinct edge objects.
        raise ValueError("live fine-edge objects must be unique")

    full_nodes, full_edges = official_prepared_universe(layer_count, head_count)
    internal_nodes, edge_order = patchable_internal_universe(layer_count, head_count)
    full_node_set = set(full_nodes)
    full_edge_set = set(full_edges)
    internal_node_set = set(internal_nodes)
    internal_edge_set = set(edge_order)
    grouped: dict[Edge, list[FineEdgeScore]] = defaultdict(list)
    excluded_boundary = 0
    for fine in scores:
        promoted = (
            promote_official_node(fine.source),
            promote_official_node(fine.target),
        )
        if promoted[0] not in full_node_set or promoted[1] not in full_node_set:
            raise ValueError(f"fine edge escapes prepared universe: {promoted!r}")
        if promoted not in full_edge_set:
            residual_bypass = promoted[0] == "blocks.0.hook_resid_pre" and promoted[
                1
            ].endswith(".hook_resid_post")
            if not residual_bypass:
                raise ValueError(
                    f"fine edge disappears after preparation: {promoted!r}"
                )
            excluded_boundary += 1
            continue
        if promoted[0] in internal_node_set and promoted[1] in internal_node_set:
            if promoted not in internal_edge_set:
                raise ValueError("fine edge promotes outside the internal DAG")
            grouped[promoted].append(fine)
        else:
            excluded_boundary += 1

    if set(grouped) != internal_edge_set:
        raise ValueError("promoted fine scores do not cover every internal whole edge")
    if (
        sum(len(members) for members in grouped.values())
        != expected.internal_fine_edges
    ):
        raise ValueError("internal fine-edge count differs from pinned geometry")
    if excluded_boundary != expected.boundary_fine_edges:
        raise ValueError("boundary fine-edge count differs from pinned geometry")

    groups: list[PromotedFineEdgeGroup] = []
    for edge in edge_order:
        members = grouped[edge]
        expected_kind = "attention_head" if ".attn.hook_result[" in edge[1] else "mlp"
        members = sorted(
            members,
            key=lambda member: (
                _fine_target_slot(member.target)[1],
                member.fine_edge_name,
            ),
        )
        typed_slots = tuple(_fine_target_slot(member.target) for member in members)
        if any(kind != expected_kind for kind, _slot in typed_slots):
            raise ValueError("fine target kind differs from promoted target kind")
        expected_slots = (0, 1, 2) if expected_kind == "attention_head" else (0,)
        if tuple(slot for _kind, slot in typed_slots) != expected_slots:
            raise ValueError("promoted edge lacks exact Q/K/V-or-MLP slots")
        if len({member.source for member in members}) != 1:
            raise ValueError("one promoted edge mixes fine sources")
        member_names = tuple(member.fine_edge_name for member in members)
        groups.append(
            PromotedFineEdgeGroup(
                promoted_edge=edge,
                fine_edge_names=member_names,
                fine_edges=tuple(live_edges_by_name[name] for name in member_names),
                collision_count=len(members),
            )
        )

    collision_counts = tuple(group.collision_count for group in groups)
    if collision_counts.count(3) != expected.internal_head_target_edges or (
        collision_counts.count(1) != expected.internal_mlp_target_edges
    ):
        raise RuntimeError("fine collision histogram differs from pinned geometry")
    return PromotedFineEdgeGrouping(
        case_id=case_id,
        edge_order=edge_order,
        groups=tuple(groups),
        fine_edge_count=len(scores),
        internal_fine_edge_count=sum(collision_counts),
        excluded_boundary_fine_edge_count=excluded_boundary,
    )


def live_generic_fine_edge_grouping(
    *, case_id: int, n_layers: int, n_heads: int, patchable_model: Any
) -> PromotedFineEdgeGrouping:
    edges = getattr(patchable_model, "edges", None)
    if not isinstance(edges, (set, frozenset, tuple, list)) or not edges:
        raise TypeError("patchable_model.edges must be a nonempty collection")
    live: dict[str, Any] = {}
    for edge in edges:
        name = getattr(edge, "name", None)
        if not isinstance(name, str) or not name or name in live:
            raise ValueError("live fine-edge names must be unique nonempty strings")
        if getattr(edge, "seq_idx", None) is not None:
            raise ValueError("E32.3 requires all-position fine edges")
        live[name] = edge
    scores = extract_fine_edge_scores(
        patchable_model, patchable_model.new_prune_scores()
    )
    return build_generic_fine_edge_grouping(
        case_id=case_id,
        n_layers=n_layers,
        n_heads=n_heads,
        fine_scores=scores,
        live_edges_by_name=live,
    )


@dataclass(frozen=True)
class GenericExactFineRun:
    case_id: int
    settings: Mapping[str, Any]
    grouping: PromotedFineEdgeGrouping
    fine_edge_names: tuple[str, ...]
    fine_to_whole_indices: tuple[int, ...]
    clean_baseline_output: torch.Tensor
    corrupt_baseline_output: torch.Tensor
    corrupt_into_clean_outputs: torch.Tensor
    clean_into_corrupt_outputs: torch.Tensor
    query_accounting: DirectedReplacementAccounting


def _validated_source_cache(
    value: Any, *, label: str, batch: int, sequence: int
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim < 3
        or value.shape[1] != batch
        or value.shape[2] != sequence
        or not bool(torch.isfinite(value).all())
    ):
        raise RuntimeError(f"{label} must have [source,batch,position,...] axes")


def _acquire_generic_exact_from_patchable(
    *,
    case_id: int,
    auto_model: Any,
    forward_boundary: Any,
    grouping: PromotedFineEdgeGrouping,
    clean_inputs: Any,
    corrupt_inputs: Any,
    expected_sequence_length: int,
    expected_classes: int,
    runtime: _AutoCircuitRuntime,
) -> GenericExactFineRun:
    """Acquire every internal fine edge separately in both orientations."""
    if case_id != grouping.case_id:
        raise ValueError("case ID differs from fine grouping")
    sequence_length = _positive_integer(
        expected_sequence_length, label="expected_sequence_length"
    )
    class_count = _positive_integer(expected_classes, label="expected_classes")
    if class_count < 2:
        raise ValueError("categorical exact replacement needs at least two classes")
    if not isinstance(runtime, _AutoCircuitRuntime):
        raise TypeError("runtime must be _AutoCircuitRuntime")
    if not hasattr(forward_boundary, "register_forward_pre_hook"):
        raise TypeError("forward boundary must support pre-hooks")
    _validate_patchable_settings(auto_model)

    groups = _canonical_groups(grouping, None)
    fine_edges = tuple(edge for group in groups for edge in group.fine_edges)
    fine_names = tuple(name for group in groups for name in group.fine_edge_names)
    fine_to_whole = tuple(
        whole_index
        for whole_index, group in enumerate(groups)
        for _name in group.fine_edge_names
    )
    clean = _integer_inputs(clean_inputs, label="clean_inputs")
    corrupt = _integer_inputs(corrupt_inputs, label="corrupt_inputs")
    if clean.shape != corrupt.shape or clean.shape[1] != sequence_length:
        raise ValueError("clean/corrupt inputs differ from frozen geometry")
    batch = len(clean)
    if _mask_coordinate_count(auto_model) != grouping.fine_edge_count:
        raise RuntimeError("mask-coordinate count differs from complete fine graph")
    _assert_exact_masks(auto_model, ())

    observer = _ForwardObserver()
    handle = forward_boundary.register_forward_pre_hook(observer)
    active_mask_checks = 0
    mask_contexts = 0
    try:
        with torch.inference_mode():
            corrupt_sources = runtime.src_ablations(
                auto_model, corrupt, runtime.resample_ablation
            )
            clean_sources = runtime.src_ablations(
                auto_model, clean, runtime.resample_ablation
            )
            _validated_source_cache(
                corrupt_sources,
                label="corrupt source cache",
                batch=batch,
                sequence=sequence_length,
            )
            _validated_source_cache(
                clean_sources,
                label="clean source cache",
                batch=batch,
                sequence=sequence_length,
            )
            clean_baseline = class_center_logits(
                _raw_non_bos_logits(auto_model, clean, expected_classes=class_count)
            )
            corrupt_baseline = class_center_logits(
                _raw_non_bos_logits(auto_model, corrupt, expected_classes=class_count)
            )
            corrupt_into_clean = []
            clean_into_corrupt = []
            for edge in fine_edges:
                selected = (edge,)
                with runtime.patch_mode(auto_model, corrupt_sources, edges=selected):
                    active_mask_checks += _assert_exact_masks(auto_model, selected)
                    mask_contexts += 1
                    corrupt_into_clean.append(
                        class_center_logits(
                            _raw_non_bos_logits(
                                auto_model, clean, expected_classes=class_count
                            )
                        )
                    )
                with runtime.patch_mode(auto_model, clean_sources, edges=selected):
                    active_mask_checks += _assert_exact_masks(auto_model, selected)
                    mask_contexts += 1
                    clean_into_corrupt.append(
                        class_center_logits(
                            _raw_non_bos_logits(
                                auto_model, corrupt, expected_classes=class_count
                            )
                        )
                    )
    finally:
        runtime.set_all_masks(auto_model, 0.0)
        handle.remove()
    if _assert_exact_masks(auto_model, ()) != 0:
        raise RuntimeError("mask cleanup left an active coordinate")

    accounting = directed_replacement_query_accounting(
        example_count=batch,
        promoted_edge_count=len(groups),
        fine_edge_count=len(fine_edges),
        include_grouped_promoted_sensitivity=False,
        observed_physical_forwards=observer.physical_forwards,
        observed_model_examples=observer.model_examples,
        observed_no_grad_forwards=observer.no_grad_forwards,
        observed_grad_enabled_forwards=observer.grad_enabled_forwards,
    )
    if not (
        accounting.observed_physical_forwards == accounting.expected_physical_forwards
        and accounting.observed_model_examples == accounting.expected_model_examples
        and accounting.observed_no_grad_forwards
        == accounting.expected_physical_forwards
        and accounting.observed_grad_enabled_forwards == 0
        and mask_contexts == accounting.mask_context_count
        and active_mask_checks == accounting.active_mask_coordinate_checks
    ):
        raise RuntimeError("observed exact-fine work differs from frozen accounting")
    return GenericExactFineRun(
        case_id=case_id,
        settings={
            "factorized": True,
            "separate_qkv": True,
            "slice_output": "not_first_seq",
            "seq_len": None,
            "ablation_type": "resample",
            "fine_scope": "all_internal_fine_edges_separate_qkv",
            "orientations": (
                "corrupt_source_into_clean_recipient",
                "clean_source_into_corrupt_recipient",
            ),
            "stored_output": "class_centered_non_bos_logits",
        },
        grouping=grouping,
        fine_edge_names=fine_names,
        fine_to_whole_indices=fine_to_whole,
        clean_baseline_output=clean_baseline.detach().cpu().clone(),
        corrupt_baseline_output=corrupt_baseline.detach().cpu().clone(),
        corrupt_into_clean_outputs=torch.stack(corrupt_into_clean),
        clean_into_corrupt_outputs=torch.stack(clean_into_corrupt),
        query_accounting=accounting,
    )


def _validate_fresh_model(model: Any) -> None:
    modules = getattr(model, "modules", None)
    if not callable(modules):
        raise TypeError("model must expose modules()")
    already_wrapped = hasattr(model, "patch_masks") or any(
        module.__class__.__name__ == "PatchWrapperImpl"
        and module.__class__.__module__.startswith("auto_circuit.")
        for module in modules()
    )
    if already_wrapped:
        raise ValueError("live bridge requires a fresh unwrapped model")


def _validated_live_pair(
    *,
    model: Any,
    case: Any,
    eligible_pair: BehaviorallyEligiblePair,
    n_layers: int,
    n_heads: int,
    n_ctx: int,
    d_vocab_out: int,
) -> Any:
    if not isinstance(eligible_pair, BehaviorallyEligiblePair):
        raise TypeError("eligible_pair must be BehaviorallyEligiblePair")
    is_categorical = getattr(case, "is_categorical", None)
    if not callable(is_categorical) or not bool(is_categorical()):
        raise NotImplementedError("E32.3 bridges require categorical data")
    cfg = getattr(model, "cfg", None)
    expected = (n_layers, n_heads, n_ctx, d_vocab_out)
    observed = tuple(
        int(getattr(cfg, name, -1))
        for name in ("n_layers", "n_heads", "n_ctx", "d_vocab_out")
    )
    if observed != expected:
        raise ValueError("model shape differs from E32.3 cohort manifest")
    pair = eligible_pair.pair
    clean = _integer_inputs(pair.clean_inputs, label="clean_inputs")
    corrupt = _integer_inputs(pair.corrupt_inputs, label="corrupt_inputs")
    if clean.shape != corrupt.shape or clean.shape[1] != n_ctx:
        raise ValueError("eligible pair differs from E32.3 geometry")
    if len(eligible_pair.selected_candidate_indices) != len(clean):
        raise ValueError("eligible indices do not align with selected rows")
    for value, label in (
        (pair.clean_targets, "clean_targets"),
        (pair.corrupt_targets, "corrupt_targets"),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (len(clean), n_ctx, d_vocab_out)
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"{label} differs from E32.3 categorical geometry")
    return pair


def run_generic_exact_fine_replacement(
    *,
    case_id: int,
    n_layers: int,
    n_heads: int,
    n_ctx: int,
    d_vocab_out: int,
    model: Any,
    case: Any,
    eligible_pair: BehaviorallyEligiblePair,
    device: str = "cpu",
    environment_dir: Path | str = DEFAULT_ENV_DIR,
    auto_circuit_dir: Path | str = DEFAULT_AUTO_CIRCUIT_DIR,
) -> GenericExactFineRun:
    """Live exhaustive exact-fine acquisition for a fresh E32.3 model."""
    _validate_fresh_model(model)
    verify_pinned_eap_runtime(
        environment_dir=environment_dir, auto_circuit_dir=auto_circuit_dir
    )
    pair = _validated_live_pair(
        model=model,
        case=case,
        eligible_pair=eligible_pair,
        n_layers=n_layers,
        n_heads=n_heads,
        n_ctx=n_ctx,
        d_vocab_out=d_vocab_out,
    )
    model.to(device)
    model.eval()
    boundary = getattr(model, "embed", None)
    if not hasattr(boundary, "register_forward_pre_hook"):
        raise TypeError("model lacks observable embedding boundary")
    runtime, patchable_model = _load_auto_circuit_runtime()
    auto_model = patchable_model(
        model,
        factorized=OFFICIAL_EAP_SETTINGS["factorized"],
        slice_output=OFFICIAL_EAP_SETTINGS["slice_output"],
        separate_qkv=OFFICIAL_EAP_SETTINGS["separate_qkv"],
        device=torch.device(device),
    )
    grouping = live_generic_fine_edge_grouping(
        case_id=case_id,
        n_layers=n_layers,
        n_heads=n_heads,
        patchable_model=auto_model,
    )
    return _acquire_generic_exact_from_patchable(
        case_id=case_id,
        auto_model=auto_model,
        forward_boundary=boundary,
        grouping=grouping,
        clean_inputs=pair.clean_inputs.to(device),
        corrupt_inputs=pair.corrupt_inputs.to(device),
        expected_sequence_length=n_ctx,
        expected_classes=d_vocab_out,
        runtime=runtime,
    )


@dataclass(frozen=True)
class GenericEAPIG5Run:
    case_id: int
    settings: EAPSettings
    grouping: PromotedFineEdgeGrouping
    all_fine_scores: tuple[FineEdgeScore, ...]
    internal_fine_edge_names: tuple[str, ...]
    internal_fine_scores: tuple[float, ...]
    fine_to_whole_indices: tuple[int, ...]
    query_accounting: EAPQueryAccounting
    observed_query_accounting: Mapping[str, int]


def package_generic_eap_ig5(
    *,
    case_id: int,
    n_layers: int,
    n_heads: int,
    example_count: int,
    fine_scores: Sequence[FineEdgeScore],
    observed_query_accounting: Mapping[str, int],
) -> GenericEAPIG5Run:
    """Package live forward counts plus the pinned EAP-IG backward count.

    The embedding-boundary observer measures every forward and its grad mode.
    AutoCircuit's six backward calls are fixed by the verified five-step
    implementation but are not observed at a live backward boundary, so the
    custody key names that count as inferred rather than observed.
    """
    settings = official_eap_settings("eap_ig", integrated_grad_samples=5)
    accounting = eap_query_accounting(
        example_count=example_count,
        method="eap_ig",
        integrated_grad_samples=5,
    )
    observed = dict(observed_query_accounting)
    expected_observed = {
        "physical_forwards": accounting.physical_model_forwards,
        "model_examples": accounting.model_examples,
        "no_grad_forwards": accounting.source_ablation_forwards,
        "grad_enabled_forwards": accounting.gradient_forwards,
        "backward_passes_inferred_from_pinned_control_flow": (
            accounting.backward_passes
        ),
    }
    if observed != expected_observed:
        raise RuntimeError("observed EAP-IG5 work differs from frozen accounting")
    scores = tuple(fine_scores)
    grouping = build_generic_fine_edge_grouping(
        case_id=case_id,
        n_layers=n_layers,
        n_heads=n_heads,
        fine_scores=scores,
        live_edges_by_name={
            score.fine_edge_name: score.fine_edge_name for score in scores
        },
    )
    fine_to_whole = tuple(
        whole_index
        for whole_index, group in enumerate(grouping.groups)
        for _name in group.fine_edge_names
    )
    score_by_name = {score.fine_edge_name: score for score in scores}
    internal_names = tuple(
        name for group in grouping.groups for name in group.fine_edge_names
    )
    internal_scores = tuple(score_by_name[name].signed_score for name in internal_names)
    if len(internal_names) != grouping.internal_fine_edge_count:
        raise RuntimeError("internal EAP fine-score axis changed")
    return GenericEAPIG5Run(
        case_id=case_id,
        settings=settings,
        grouping=grouping,
        all_fine_scores=scores,
        internal_fine_edge_names=internal_names,
        internal_fine_scores=internal_scores,
        fine_to_whole_indices=fine_to_whole,
        query_accounting=accounting,
        observed_query_accounting=observed,
    )


def run_generic_eap_ig5(
    *,
    case_id: int,
    n_layers: int,
    n_heads: int,
    n_ctx: int,
    d_vocab_out: int,
    model: Any,
    case: Any,
    eligible_pair: BehaviorallyEligiblePair,
    device: str = "cpu",
    environment_dir: Path | str = DEFAULT_ENV_DIR,
    auto_circuit_dir: Path | str = DEFAULT_AUTO_CIRCUIT_DIR,
) -> GenericEAPIG5Run:
    """Run official five-step EAP-IG and generically validate its fine graph."""
    _validate_fresh_model(model)
    verify_pinned_eap_runtime(
        environment_dir=environment_dir, auto_circuit_dir=auto_circuit_dir
    )
    pair = _validated_live_pair(
        model=model,
        case=case,
        eligible_pair=eligible_pair,
        n_layers=n_layers,
        n_heads=n_heads,
        n_ctx=n_ctx,
        d_vocab_out=d_vocab_out,
    )
    boundary = getattr(model, "embed", None)
    if not hasattr(boundary, "register_forward_pre_hook"):
        raise TypeError("model lacks observable embedding boundary")

    from auto_circuit.data import PromptDataLoader, PromptDataset
    from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
    from auto_circuit.utils.graph_utils import patchable_model

    model.to(device)
    model.eval()
    auto_model = patchable_model(
        model,
        factorized=OFFICIAL_EAP_SETTINGS["factorized"],
        slice_output=OFFICIAL_EAP_SETTINGS["slice_output"],
        separate_qkv=OFFICIAL_EAP_SETTINGS["separate_qkv"],
        device=torch.device(device),
    )
    clean_ids = pair.clean_targets.argmax(dim=-1).unsqueeze(-1).to(device)
    corrupt_ids = pair.corrupt_targets.argmax(dim=-1).unsqueeze(-1).to(device)
    dataset = PromptDataset(
        pair.clean_inputs.to(device),
        pair.corrupt_inputs.to(device),
        clean_ids[auto_model.out_slice],
        corrupt_ids[auto_model.out_slice],
    )
    dataloader = PromptDataLoader(
        dataset,
        seq_len=case.get_max_seq_len(),
        diverge_idx=0,
        batch_size=len(dataset),
        shuffle=False,
    )
    if len(dataloader) != 1:
        raise RuntimeError("E32.3 EAP-IG5 requires one full dataset batch")
    settings = official_eap_settings("eap_ig", integrated_grad_samples=5)

    def categorical_kl(logits: torch.Tensor, batch: Any) -> torch.Tensor:
        answers = batch.answers.squeeze(dim=-1)
        if answers.shape != logits.shape:
            answers = torch.nn.functional.one_hot(
                answers, num_classes=d_vocab_out
            ).float()
        return torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits, dim=-1),
            answers,
            reduction="batchmean",
            log_target=False,
        )

    observer = _ForwardObserver()
    handle = boundary.register_forward_pre_hook(observer)
    try:
        prune_scores = mask_gradient_prune_scores(
            model=auto_model,
            dataloader=dataloader,
            official_edges=None,
            grad_function=settings.grad_function,
            answer_function=categorical_kl,
            mask_val=settings.mask_val,
            integrated_grad_samples=settings.integrated_grad_samples,
        )
    finally:
        handle.remove()
    fine_scores = extract_fine_edge_scores(auto_model, prune_scores)
    return package_generic_eap_ig5(
        case_id=case_id,
        n_layers=n_layers,
        n_heads=n_heads,
        example_count=len(pair.clean_inputs),
        fine_scores=fine_scores,
        observed_query_accounting={
            "physical_forwards": observer.physical_forwards,
            "model_examples": observer.model_examples,
            "no_grad_forwards": observer.no_grad_forwards,
            "grad_enabled_forwards": observer.grad_enabled_forwards,
            "backward_passes_inferred_from_pinned_control_flow": 6,
        },
    )


def generic_exact_run_to_dict(run: GenericExactFineRun) -> dict[str, Any]:
    if not isinstance(run, GenericExactFineRun):
        raise TypeError("run must be GenericExactFineRun")
    return {
        "schema_version": "e32_3_generic_exact_fine_v1",
        "case_id": run.case_id,
        "settings": dict(run.settings),
        "edge_order": run.grouping.edge_order,
        "fine_edge_names": run.fine_edge_names,
        "fine_to_whole_indices": run.fine_to_whole_indices,
        "auto_circuit_fine_edge_count": run.grouping.fine_edge_count,
        "fine_edge_count": run.grouping.internal_fine_edge_count,
        "internal_fine_edge_count": run.grouping.internal_fine_edge_count,
        "excluded_boundary_fine_edge_count": (
            run.grouping.excluded_boundary_fine_edge_count
        ),
        "clean_baseline_output": run.clean_baseline_output,
        "corrupt_baseline_output": run.corrupt_baseline_output,
        "corrupt_into_clean_outputs": run.corrupt_into_clean_outputs,
        "clean_into_corrupt_outputs": run.clean_into_corrupt_outputs,
        "query_accounting": asdict(run.query_accounting),
    }


def generic_eap_run_to_dict(run: GenericEAPIG5Run) -> dict[str, Any]:
    if not isinstance(run, GenericEAPIG5Run):
        raise TypeError("run must be GenericEAPIG5Run")
    return {
        "schema_version": "e32_3_generic_eap_ig5_v1",
        "case_id": run.case_id,
        "settings": asdict(run.settings),
        "edge_order": run.grouping.edge_order,
        "all_fine_scores": tuple(asdict(score) for score in run.all_fine_scores),
        "internal_fine_edge_names": run.internal_fine_edge_names,
        "internal_fine_scores": run.internal_fine_scores,
        "fine_to_whole_indices": run.fine_to_whole_indices,
        "auto_circuit_fine_edge_count": run.grouping.fine_edge_count,
        "fine_edge_count": run.grouping.internal_fine_edge_count,
        "internal_fine_edge_count": run.grouping.internal_fine_edge_count,
        "excluded_boundary_fine_edge_count": (
            run.grouping.excluded_boundary_fine_edge_count
        ),
        "query_accounting": asdict(run.query_accounting),
        "observed_query_accounting": dict(run.observed_query_accounting),
    }


__all__ = [
    "GenericEAPIG5Run",
    "GenericExactFineRun",
    "build_generic_fine_edge_grouping",
    "generic_eap_run_to_dict",
    "generic_exact_run_to_dict",
    "live_generic_fine_edge_grouping",
    "package_generic_eap_ig5",
    "run_generic_eap_ig5",
    "run_generic_exact_fine_replacement",
]
