# ruff: noqa: EXE002
"""Case-generic official EAP and EAP-IG5 baselines for E32.4.

The sealed E32.3 bridge exposed only EAP-IG5.  E32.4 needs the cheaper plain
EAP comparator as well, so this module keeps the same verified AutoCircuit
implementation and varies only the official method setting.  Q/K/V scores
remain separate and are aligned to the exact internal fine-edge axis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from experiments.e32_interpbench.adapter import (
    DEFAULT_ENV_DIR,
    BehaviorallyEligiblePair,
)
from experiments.e32_interpbench.e32_2_directed_replacement import (
    PromotedFineEdgeGrouping,
    _ForwardObserver,
)
from experiments.e32_interpbench.e32_3_baselines import (
    _validate_fresh_model,
    _validated_live_pair,
    build_generic_fine_edge_grouping,
)
from experiments.e32_interpbench.eap_baseline import (
    DEFAULT_AUTO_CIRCUIT_DIR,
    OFFICIAL_EAP_SETTINGS,
    EAPQueryAccounting,
    EAPSettings,
    FineEdgeScore,
    eap_query_accounting,
    extract_fine_edge_scores,
    official_eap_settings,
    verify_pinned_eap_runtime,
)


@dataclass(frozen=True)
class GenericEAPRun:
    """One official gradient baseline on a generic pinned cohort case."""

    case_id: int
    settings: EAPSettings
    grouping: PromotedFineEdgeGrouping
    all_fine_scores: tuple[FineEdgeScore, ...]
    internal_fine_edge_names: tuple[str, ...]
    internal_fine_scores: tuple[float, ...]
    fine_to_whole_indices: tuple[int, ...]
    query_accounting: EAPQueryAccounting
    observed_query_accounting: Mapping[str, int]


def package_generic_eap(
    *,
    case_id: int,
    n_layers: int,
    n_heads: int,
    example_count: int,
    method: str,
    fine_scores: Sequence[FineEdgeScore],
    observed_query_accounting: Mapping[str, int],
) -> GenericEAPRun:
    """Validate live work and align official scores to the internal fine graph."""
    if method not in {"eap", "eap_ig"}:
        raise ValueError("method must be 'eap' or 'eap_ig'")
    integrated_samples = 5 if method == "eap_ig" else None
    settings = official_eap_settings(
        method,
        integrated_grad_samples=integrated_samples,  # type: ignore[arg-type]
    )
    accounting = eap_query_accounting(
        example_count=example_count,
        method=method,  # type: ignore[arg-type]
        integrated_grad_samples=integrated_samples,
    )
    observed = dict(observed_query_accounting)
    expected_observed = {
        "physical_forwards": accounting.physical_model_forwards,
        "model_examples": accounting.model_examples,
        "no_grad_forwards": accounting.source_ablation_forwards,
        "grad_enabled_forwards": accounting.gradient_forwards,
        "backward_passes_inferred_from_pinned_control_flow": accounting.backward_passes,
    }
    if observed != expected_observed:
        raise RuntimeError(f"observed EAP work differs from frozen {method} accounting")

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
    internal_names = tuple(
        name for group in grouping.groups for name in group.fine_edge_names
    )
    score_by_name = {score.fine_edge_name: score for score in scores}
    if len(score_by_name) != len(scores):
        raise ValueError("official EAP fine-edge names must be unique")
    internal_scores = tuple(score_by_name[name].signed_score for name in internal_names)
    fine_to_whole = tuple(
        whole_index
        for whole_index, group in enumerate(grouping.groups)
        for _name in group.fine_edge_names
    )
    if len(internal_names) != grouping.internal_fine_edge_count:
        raise RuntimeError("internal EAP fine-score axis changed")
    return GenericEAPRun(
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


def run_generic_eap(
    *,
    case_id: int,
    n_layers: int,
    n_heads: int,
    n_ctx: int,
    d_vocab_out: int,
    model: Any,
    case: Any,
    eligible_pair: BehaviorallyEligiblePair,
    method: str,
    device: str = "cpu",
    environment_dir: Path | str = DEFAULT_ENV_DIR,
    auto_circuit_dir: Path | str = DEFAULT_AUTO_CIRCUIT_DIR,
) -> GenericEAPRun:
    """Run untouched official EAP or EAP-IG5 on one full eligible batch."""
    if method not in {"eap", "eap_ig"}:
        raise ValueError("method must be 'eap' or 'eap_ig'")
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
        raise RuntimeError("E32.4 EAP requires one full dataset batch")
    integrated_samples = 5 if method == "eap_ig" else None
    settings = official_eap_settings(
        method,
        integrated_grad_samples=integrated_samples,  # type: ignore[arg-type]
    )

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
    accounting = eap_query_accounting(
        example_count=len(pair.clean_inputs),
        method=method,  # type: ignore[arg-type]
        integrated_grad_samples=integrated_samples,
    )
    return package_generic_eap(
        case_id=case_id,
        n_layers=n_layers,
        n_heads=n_heads,
        example_count=len(pair.clean_inputs),
        method=method,
        fine_scores=fine_scores,
        observed_query_accounting={
            "physical_forwards": observer.physical_forwards,
            "model_examples": observer.model_examples,
            "no_grad_forwards": observer.no_grad_forwards,
            "grad_enabled_forwards": observer.grad_enabled_forwards,
            "backward_passes_inferred_from_pinned_control_flow": (
                accounting.backward_passes
            ),
        },
    )


def generic_eap_run_to_dict(run: GenericEAPRun) -> dict[str, Any]:
    """Return a tensor-safe custody record without changing score order."""
    if not isinstance(run, GenericEAPRun):
        raise TypeError("run must be GenericEAPRun")
    return {
        "schema_version": "e32_4_generic_eap_v1",
        "case_id": run.case_id,
        "settings": asdict(run.settings),
        "edge_order": run.grouping.edge_order,
        "all_fine_scores": tuple(asdict(score) for score in run.all_fine_scores),
        "internal_fine_edge_names": run.internal_fine_edge_names,
        "internal_fine_scores": run.internal_fine_scores,
        "fine_to_whole_indices": run.fine_to_whole_indices,
        "auto_circuit_fine_edge_count": run.grouping.fine_edge_count,
        "internal_fine_edge_count": run.grouping.internal_fine_edge_count,
        "excluded_boundary_fine_edge_count": (
            run.grouping.excluded_boundary_fine_edge_count
        ),
        "query_accounting": asdict(run.query_accounting),
        "observed_query_accounting": dict(run.observed_query_accounting),
    }


__all__ = [
    "GenericEAPRun",
    "generic_eap_run_to_dict",
    "package_generic_eap",
    "run_generic_eap",
]
