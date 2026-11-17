# ruff: noqa: EXE002
"""Exact official EAP/EAP-IG baseline bridge for E32.2.

AutoCircuit scores Q/K/V input edges separately, while E32 evaluates directed
interactions between whole attention heads and MLPs.  This module keeps the
official gradient implementation untouched, then applies the same rerouting
used by InterpBench's circuit evaluator and restricts the result to E32's
predeclared internal edge universe.

Imports from AutoCircuit are deliberately lazy.  Pure score promotion and its
tests therefore run in the project's Python-3.12 environment, while live
baselines run only in the pinned Python-3.11 environment below.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from experiments.e32_interpbench.adapter import (
    CASE_MANIFESTS,
    DEFAULT_CACHE_ROOT,
    DEFAULT_ENV_DIR,
    IIT_REVISION,
    INTERPBENCH_CODE_REVISION,
    INTERPBENCH_MODEL_REVISION,
    PANEL_CASE_IDS,
    TRACR_REVISION,
    BehaviorallyEligiblePair,
    official_prepared_universe,
    patchable_internal_universe,
    promote_official_node,
    sha256_file,
    verify_git_revision,
)

Edge = tuple[str, str]
EAPMethod = Literal["eap", "eap_ig"]

AUTO_CIRCUIT_REPOSITORY = "https://github.com/FlyingPumba/auto-circuit.git"
AUTO_CIRCUIT_REVISION = "0b74956bbe472cf961deb8ae0ef574f725daca96"
DEFAULT_AUTO_CIRCUIT_DIR = DEFAULT_CACHE_ROOT / f"auto-circuit-{AUTO_CIRCUIT_REVISION}"

# The InterpBench README's only pinned EAP-IG invocation uses five steps.  The
# bounded n=2 smoke overrides this to two; scientific runs retain five.
OFFICIAL_EAP_IG_STEPS = 5
OFFICIAL_EAP_SETTINGS = {
    "factorized": True,
    "slice_output": "not_first_seq",
    "separate_qkv": True,
    "grad_function": "logit",
    "classification_loss": "kl_div",
    "ablation_type": "resample_corrupt",
    "batching": "one_full_dataset_batch",
}

# Exact versions observed in the isolated environment used for the feasibility
# smoke.  Local source distributions are pinned separately by git revision.
PINNED_EAP_ENVIRONMENT = {
    "python": "3.11.9",
    "auto-circuit": "0.1.0",
    "torch": "2.3.1",
    "transformer-lens": "1.19.0",
    "numpy": "1.26.4",
    "einops": "0.7.0",
    "huggingface-hub": "0.24.5",
    "blobfile": "2.0.2",
    "kaleido": "0.2.1",
    "ordered-set": "4.1.0",
    "plotly": "5.22.0",
    "vscode-tqdm": "4.66.2",
    "word2word": "1.0.0",
}
PINNED_AUTO_CIRCUIT_CORE_SHA256 = {
    "auto_circuit/data.py": (
        "3925de46979f1baabe55c9cc33972d52ae7d162e625651080cfbbfbd981ead30"
    ),
    "auto_circuit/types.py": (
        "85bfaf6e67a76e3aaaa8686f3bc4295641ec6119c1e669d7b44215291c55ccdc"
    ),
    "auto_circuit/model_utils/transformer_lens_utils.py": (
        "8be6a35345a7fcaaf3bb95b17cdf1811b2b50a394949e1adc91bbcb8d03b8efe"
    ),
    "auto_circuit/prune_algos/mask_gradient.py": (
        "36ca10753c6efb77c9c472d29d3e81da9194dee7ef3279a1a5b095a5c0baec5b"
    ),
    "auto_circuit/utils/graph_utils.py": (
        "348e2465ef6688ac21a92addbfac8df10221488b724c9bcaf5bfa93a8822d578"
    ),
    "auto_circuit/utils/patchable_model.py": (
        "7c638d017d200635b8e3a283faae2035d8b6f83f3b591f40d6dabcbdfb7c00b8"
    ),
}


@dataclass(frozen=True)
class EAPSettings:
    """Arguments passed directly to the pinned official implementation."""

    method: EAPMethod
    factorized: bool
    slice_output: str
    separate_qkv: bool
    grad_function: str
    classification_loss: str
    ablation_type: str
    batching: str
    mask_val: float | None
    integrated_grad_samples: int | None


@dataclass(frozen=True)
class FineEdgeScore:
    """One signed score on one AutoCircuit source-to-destination edge."""

    source: str
    target: str
    fine_edge_name: str
    signed_score: float


@dataclass(frozen=True)
class AutoCircuitGraphCounts:
    """Expected graph sizes for a factorized, separate-QKV transformer."""

    fine_edges: int
    internal_fine_edges: int
    boundary_fine_edges: int
    internal_whole_edges: int
    internal_head_target_edges: int
    internal_mlp_target_edges: int


@dataclass(frozen=True)
class PromotedEAPScores:
    """EAP scores aligned exactly to the E32 internal directed-edge universe."""

    case_id: int
    edge_order: tuple[Edge, ...]
    signed_max_scores: tuple[float, ...]
    absolute_max_scores: tuple[float, ...]
    collision_counts: tuple[int, ...]
    fine_edge_count: int
    internal_fine_edge_count: int
    excluded_boundary_fine_edge_count: int
    coverage_complete: bool

    def selected_edges(
        self, *, threshold: float, absolute: bool = False
    ) -> tuple[Edge, ...]:
        """Return support using InterpBench's strict ``score > threshold`` rule."""
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a finite real number")
        threshold = float(threshold)
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        scores = self.absolute_max_scores if absolute else self.signed_max_scores
        return tuple(
            edge for edge, score in zip(self.edge_order, scores) if score > threshold
        )


@dataclass(frozen=True)
class EAPQueryAccounting:
    """Exact model-work count implied by official ``mask_gradient`` control flow."""

    example_count: int
    dataset_batches: int
    interpolation_points: int
    source_ablation_forwards: int
    gradient_forwards: int
    backward_passes: int
    physical_model_forwards: int
    model_examples: int
    gradient_examples: int
    gradient_reduction: str
    excludes_behavioral_eligibility: bool


@dataclass(frozen=True)
class OfficialEAPRun:
    """Auditable raw and promoted output from one official gradient baseline."""

    case_id: int
    settings: EAPSettings
    fine_scores: tuple[FineEdgeScore, ...]
    promoted_scores: PromotedEAPScores
    query_accounting: EAPQueryAccounting


def official_eap_settings(
    method: EAPMethod, *, integrated_grad_samples: int | None = None
) -> EAPSettings:
    """Return the frozen official settings for EAP or EAP-IG."""
    if method not in ("eap", "eap_ig"):
        raise ValueError("method must be 'eap' or 'eap_ig'")
    if method == "eap":
        if integrated_grad_samples is not None:
            raise ValueError("plain EAP does not accept integrated_grad_samples")
        mask_val, samples = 0.0, None
    else:
        samples = (
            OFFICIAL_EAP_IG_STEPS
            if integrated_grad_samples is None
            else integrated_grad_samples
        )
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError("integrated_grad_samples must be a positive integer")
        mask_val = None
    return EAPSettings(
        method=method,
        **OFFICIAL_EAP_SETTINGS,
        mask_val=mask_val,
        integrated_grad_samples=samples,
    )


def expected_auto_circuit_counts(n_layers: int, n_heads: int) -> AutoCircuitGraphCounts:
    """Compute exact graph sizes from AutoCircuit's factorized layer ordering."""
    for value, label in ((n_layers, "n_layers"), (n_heads, "n_heads")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")

    fine_edges = 0
    internal_fine_edges = 0
    internal_head_target_edges = 0
    internal_mlp_target_edges = 0
    for layer in range(n_layers):
        preceding_internal_sources = layer * (n_heads + 1)
        attention_sources = 1 + preceding_internal_sources
        mlp_internal_sources = preceding_internal_sources + n_heads
        mlp_sources = 1 + mlp_internal_sources
        fine_edges += 3 * n_heads * attention_sources + mlp_sources
        internal_fine_edges += (
            3 * n_heads * preceding_internal_sources + mlp_internal_sources
        )
        internal_head_target_edges += n_heads * preceding_internal_sources
        internal_mlp_target_edges += mlp_internal_sources
    fine_edges += 1 + n_layers * (n_heads + 1)  # Resid End destination.
    internal_whole_edges = internal_head_target_edges + internal_mlp_target_edges
    return AutoCircuitGraphCounts(
        fine_edges=fine_edges,
        internal_fine_edges=internal_fine_edges,
        boundary_fine_edges=fine_edges - internal_fine_edges,
        internal_whole_edges=internal_whole_edges,
        internal_head_target_edges=internal_head_target_edges,
        internal_mlp_target_edges=internal_mlp_target_edges,
    )


def _auto_circuit_node_label(node: Any) -> str:
    try:
        module_name = node.module_name
        head_idx = node.head_idx
    except AttributeError as error:
        raise TypeError("AutoCircuit node needs module_name and head_idx") from error
    if not isinstance(module_name, str) or not module_name:
        raise TypeError("AutoCircuit module_name must be a nonempty string")
    if head_idx is None:
        return module_name
    if isinstance(head_idx, bool) or not isinstance(head_idx, int) or head_idx < 0:
        raise TypeError("AutoCircuit head_idx must be a nonnegative integer or None")
    return f"{module_name}[{head_idx}]"


def extract_fine_edge_scores(
    patchable_model: Any, prune_scores: Mapping[str, Any]
) -> tuple[FineEdgeScore, ...]:
    """Read every official score through its exact ``dest/patch_idx`` coordinate."""
    edges = getattr(patchable_model, "edges", None)
    if not isinstance(edges, (set, frozenset, tuple, list)) or not edges:
        raise TypeError("patchable_model.edges must be a nonempty collection")
    if not isinstance(prune_scores, Mapping) or not prune_scores:
        raise TypeError("prune_scores must be a nonempty mapping")

    expected_modules = {edge.dest.module_name for edge in edges}
    if set(prune_scores) != expected_modules:
        raise ValueError("prune-score modules differ from graph destinations")
    score_entry_count = sum(int(scores.numel()) for scores in prune_scores.values())
    if score_entry_count != len(edges):
        raise ValueError(
            "prune-score entry count differs from AutoCircuit fine-edge count"
        )

    extracted: list[FineEdgeScore] = []
    used_coordinates: set[tuple[str, tuple[int, ...]]] = set()
    for edge in edges:
        if getattr(edge, "seq_idx", None) is not None:
            raise ValueError("E32.2 requires all-position edges with seq_idx=None")
        try:
            module_name = edge.dest.module_name
            patch_idx = tuple(edge.patch_idx)
            fine_edge_name = edge.name
            score = prune_scores[module_name][patch_idx]
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise ValueError("invalid AutoCircuit edge score coordinate") from error
        coordinate = (module_name, patch_idx)
        if coordinate in used_coordinates:
            raise ValueError("two AutoCircuit edges share one score coordinate")
        used_coordinates.add(coordinate)
        if not isinstance(fine_edge_name, str) or not fine_edge_name:
            raise TypeError("AutoCircuit edge.name must be a nonempty string")
        if int(score.numel()) != 1:
            raise ValueError("one AutoCircuit edge must index one scalar score")
        signed_score = float(score.detach().cpu().item())
        if not math.isfinite(signed_score):
            raise ValueError("AutoCircuit returned a nonfinite edge score")
        extracted.append(
            FineEdgeScore(
                source=_auto_circuit_node_label(edge.src),
                target=_auto_circuit_node_label(edge.dest),
                fine_edge_name=fine_edge_name,
                signed_score=signed_score,
            )
        )
    if len(used_coordinates) != score_entry_count:
        raise RuntimeError("not every prune-score coordinate maps to one graph edge")
    return tuple(
        sorted(
            extracted,
            key=lambda item: (item.source, item.target, item.fine_edge_name),
        )
    )


def promote_and_aggregate_fine_scores(
    case_id: int, fine_scores: Sequence[FineEdgeScore]
) -> PromotedEAPScores:
    """Promote, max-aggregate, and align a complete official fine-edge score set.

    InterpBench first thresholds fine scores into a binary circuit, then reroutes
    Q/K/V destinations to their whole head.  Multiple selected Q/K/V edges thus
    combine by logical OR.  ``max`` is the score-level operation that preserves
    this support for its default signed threshold; ``max(abs(.))`` preserves it
    when the official ``abs_value_threshold`` option is enabled.
    """
    if case_id not in PANEL_CASE_IDS:
        raise ValueError(f"case_id must be one of {PANEL_CASE_IDS!r}")
    manifest = CASE_MANIFESTS[case_id]
    expected = expected_auto_circuit_counts(manifest.n_layers, manifest.n_heads)
    if len(fine_scores) != expected.fine_edges:
        raise ValueError(
            f"case {case_id} fine-edge count must be {expected.fine_edges}, "
            f"observed {len(fine_scores)}"
        )

    full_nodes, full_edge_order = official_prepared_universe(
        manifest.n_layers, manifest.n_heads
    )
    internal_nodes, edge_order = patchable_internal_universe(
        manifest.n_layers, manifest.n_heads
    )
    full_node_set, full_edge_set = set(full_nodes), set(full_edge_order)
    node_set, edge_set = set(internal_nodes), set(edge_order)
    grouped: dict[Edge, list[FineEdgeScore]] = defaultdict(list)
    names: set[str] = set()
    excluded_boundary = 0
    for fine in fine_scores:
        if not isinstance(fine, FineEdgeScore):
            raise TypeError("fine_scores must contain FineEdgeScore records")
        if (
            not fine.source
            or not fine.target
            or not fine.fine_edge_name
            or fine.fine_edge_name in names
        ):
            raise ValueError(
                "fine-edge names and endpoints must be nonempty and unique"
            )
        names.add(fine.fine_edge_name)
        if not math.isfinite(fine.signed_score):
            raise ValueError("fine-edge scores must be finite")
        promoted = (
            promote_official_node(fine.source),
            promote_official_node(fine.target),
        )
        if promoted[0] not in full_node_set or promoted[1] not in full_node_set:
            raise ValueError(
                f"fine edge escapes the exact prepared universe: {promoted!r}"
            )
        if promoted not in full_edge_set:
            # AutoCircuit includes the residual-start -> residual-end bypass;
            # InterpBench's preparation explicitly removes residual-to-residual
            # edges.  No other graph edge may disappear at this stage.
            if not (
                promoted[0] == "blocks.0.hook_resid_pre"
                and promoted[1].endswith(".hook_resid_post")
            ):
                raise ValueError(
                    f"fine edge is unexpectedly removed by preparation: {promoted!r}"
                )
            excluded_boundary += 1
            continue
        if promoted[0] in node_set and promoted[1] in node_set:
            if promoted not in edge_set:
                raise ValueError(
                    f"fine edge promotes outside the valid internal DAG: {promoted!r}"
                )
            grouped[promoted].append(fine)
        else:
            excluded_boundary += 1

    if sum(map(len, grouped.values())) != expected.internal_fine_edges:
        raise ValueError("promoted internal fine-edge count differs from pinned graph")
    if excluded_boundary != expected.boundary_fine_edges:
        raise ValueError("excluded boundary fine-edge count differs from pinned graph")
    missing = tuple(edge for edge in edge_order if edge not in grouped)
    extra = tuple(edge for edge in grouped if edge not in edge_set)
    if missing or extra:
        raise ValueError(
            f"promoted score coverage mismatch: missing={missing!r}, extra={extra!r}"
        )

    collision_counts = tuple(len(grouped[edge]) for edge in edge_order)
    for edge, collision_count in zip(edge_order, collision_counts):
        expected_collisions = 3 if ".attn.hook_result[" in edge[1] else 1
        if collision_count != expected_collisions:
            raise ValueError(
                f"edge {edge!r} needs {expected_collisions} fine scores, "
                f"observed {collision_count}"
            )
    histogram = Counter(collision_counts)
    if histogram[3] != expected.internal_head_target_edges or histogram[1] != (
        expected.internal_mlp_target_edges
    ):
        raise RuntimeError("collision histogram differs from pinned graph")

    signed_max = tuple(
        max(fine.signed_score for fine in grouped[edge]) for edge in edge_order
    )
    absolute_max = tuple(
        max(abs(fine.signed_score) for fine in grouped[edge]) for edge in edge_order
    )
    return PromotedEAPScores(
        case_id=case_id,
        edge_order=edge_order,
        signed_max_scores=signed_max,
        absolute_max_scores=absolute_max,
        collision_counts=collision_counts,
        fine_edge_count=len(fine_scores),
        internal_fine_edge_count=sum(collision_counts),
        excluded_boundary_fine_edge_count=excluded_boundary,
        coverage_complete=True,
    )


def eap_query_accounting(
    *,
    example_count: int,
    method: EAPMethod,
    integrated_grad_samples: int | None = None,
) -> EAPQueryAccounting:
    """Count work in pinned ``mask_gradient_prune_scores`` with one full batch."""
    if (
        isinstance(example_count, bool)
        or not isinstance(example_count, int)
        or example_count < 1
    ):
        raise ValueError("example_count must be a positive integer")
    settings = official_eap_settings(
        method, integrated_grad_samples=integrated_grad_samples
    )
    interpolation_points = (
        1
        if settings.integrated_grad_samples is None
        else settings.integrated_grad_samples + 1
    )
    return EAPQueryAccounting(
        example_count=example_count,
        dataset_batches=1,
        interpolation_points=interpolation_points,
        source_ablation_forwards=1,
        gradient_forwards=interpolation_points,
        backward_passes=interpolation_points,
        physical_model_forwards=1 + interpolation_points,
        model_examples=example_count * (1 + interpolation_points),
        gradient_examples=example_count * interpolation_points,
        gradient_reduction=(
            "single_gradient"
            if method == "eap"
            else "sum_not_mean_over_k_plus_one_points"
        ),
        excludes_behavioral_eligibility=True,
    )


def verify_pinned_eap_runtime(
    *,
    environment_dir: Path | str = DEFAULT_ENV_DIR,
    auto_circuit_dir: Path | str = DEFAULT_AUTO_CIRCUIT_DIR,
) -> dict[str, Any]:
    """Fail closed unless source, interpreter, and package pins all match."""
    expected_environment = Path(environment_dir).expanduser().resolve()
    observed_environment = Path(sys.prefix).resolve()
    if observed_environment != expected_environment:
        raise RuntimeError(
            f"EAP must run in {expected_environment}, observed {observed_environment}"
        )
    if platform.python_version() != PINNED_EAP_ENVIRONMENT["python"]:
        raise RuntimeError("pinned EAP Python version changed")

    checkout = Path(auto_circuit_dir).expanduser().resolve()
    observed_revision = verify_git_revision(checkout, AUTO_CIRCUIT_REVISION)
    observed_packages: dict[str, str] = {}
    for distribution, expected_version in PINNED_EAP_ENVIRONMENT.items():
        if distribution == "python":
            continue
        try:
            observed_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"missing pinned EAP package {distribution}") from error
        if observed_version != expected_version:
            raise RuntimeError(
                f"{distribution} version changed: expected {expected_version}, "
                f"observed {observed_version}"
            )
        observed_packages[distribution] = observed_version

    distribution = importlib.metadata.distribution("auto-circuit")
    direct_url_payload = distribution.read_text("direct_url.json")
    if direct_url_payload is None:
        raise RuntimeError("auto-circuit installation lacks direct_url.json")
    direct_url = json.loads(direct_url_payload).get("url")
    if not isinstance(direct_url, str) or not direct_url:
        raise RuntimeError("auto-circuit direct_url.json lacks a source URL")
    parsed = urlparse(direct_url)
    installed_from = Path(unquote(parsed.path)).resolve()
    if parsed.scheme != "file" or installed_from != checkout:
        raise RuntimeError(
            f"auto-circuit was not installed from pinned checkout {checkout}"
        )
    verified_core_hashes: dict[str, str] = {}
    for relative_path, expected_hash in PINNED_AUTO_CIRCUIT_CORE_SHA256.items():
        checkout_hash = sha256_file(checkout / relative_path)
        installed_hash = sha256_file(distribution.locate_file(relative_path))
        if checkout_hash != expected_hash or installed_hash != expected_hash:
            raise RuntimeError(f"pinned AutoCircuit core file changed: {relative_path}")
        verified_core_hashes[relative_path] = expected_hash
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "environment_dir": str(observed_environment),
        "packages": observed_packages,
        "auto_circuit_core_sha256": verified_core_hashes,
        "sources": {
            "auto_circuit_repository": AUTO_CIRCUIT_REPOSITORY,
            "auto_circuit_revision": observed_revision,
            "auto_circuit_checkout": str(checkout),
            "interpbench_revision": INTERPBENCH_CODE_REVISION,
            "interpbench_model_revision": INTERPBENCH_MODEL_REVISION,
            "tracr_revision": TRACR_REVISION,
            "iit_revision": IIT_REVISION,
        },
    }


def _validate_live_inputs(
    *, case_id: int, model: Any, case: Any, eligible_pair: BehaviorallyEligiblePair
) -> tuple[Any, int]:
    if case_id not in PANEL_CASE_IDS:
        raise ValueError(f"case_id must be one of {PANEL_CASE_IDS!r}")
    if not isinstance(eligible_pair, BehaviorallyEligiblePair):
        raise TypeError("eligible_pair must be a BehaviorallyEligiblePair")
    if eligible_pair.output_kind != "categorical" or not bool(case.is_categorical()):
        raise NotImplementedError(
            "E32.2 official EAP currently requires categorical data"
        )
    manifest = CASE_MANIFESTS[case_id]
    cfg = getattr(model, "cfg", None)
    if (
        int(getattr(cfg, "n_layers", -1)) != manifest.n_layers
        or int(getattr(cfg, "n_heads", -1)) != manifest.n_heads
        or int(getattr(cfg, "d_vocab_out", -1)) != manifest.d_vocab_out
    ):
        raise ValueError("model shape differs from pinned case manifest")
    pair = eligible_pair.pair
    example_count = len(pair.clean_inputs)
    if example_count < 1 or len(pair.corrupt_inputs) != example_count:
        raise ValueError("eligible clean/corrupt rows must align and be nonempty")
    if len(eligible_pair.selected_candidate_indices) != example_count:
        raise ValueError("eligibility indices do not align with selected rows")
    return pair, example_count


def run_official_eap_baseline(
    *,
    case_id: int,
    model: Any,
    case: Any,
    eligible_pair: BehaviorallyEligiblePair,
    method: EAPMethod,
    integrated_grad_samples: int | None = None,
    device: str = "cpu",
    environment_dir: Path | str = DEFAULT_ENV_DIR,
    auto_circuit_dir: Path | str = DEFAULT_AUTO_CIRCUIT_DIR,
) -> OfficialEAPRun:
    """Execute one untouched official EAP method, then adapt only its scores.

    ``patchable_model`` mutates ``model``.  Callers must supply a freshly loaded
    model for every method; this prevents gradient carry-over between EAP and
    EAP-IG, because the pinned implementation does not clear mask gradients.
    """
    verify_pinned_eap_runtime(
        environment_dir=environment_dir, auto_circuit_dir=auto_circuit_dir
    )
    pair, example_count = _validate_live_inputs(
        case_id=case_id, model=model, case=case, eligible_pair=eligible_pair
    )
    settings = official_eap_settings(
        method, integrated_grad_samples=integrated_grad_samples
    )

    import torch
    from auto_circuit.data import PromptDataLoader, PromptDataset
    from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
    from auto_circuit.utils.graph_utils import patchable_model

    model.to(device)
    model.eval()
    auto_model = patchable_model(
        model,
        factorized=settings.factorized,
        slice_output=settings.slice_output,
        separate_qkv=settings.separate_qkv,
        device=torch.device(device),
    )
    if any(wrapper.patch_mask.grad is not None for wrapper in auto_model.dest_wrappers):
        raise RuntimeError(
            "fresh AutoCircuit mask unexpectedly has an existing gradient"
        )

    clean_inputs = pair.clean_inputs.to(device)
    corrupt_inputs = pair.corrupt_inputs.to(device)
    clean_ids = pair.clean_targets.argmax(dim=-1).unsqueeze(dim=-1).to(device)
    corrupt_ids = pair.corrupt_targets.argmax(dim=-1).unsqueeze(dim=-1).to(device)
    clean_ids = clean_ids[auto_model.out_slice]
    corrupt_ids = corrupt_ids[auto_model.out_slice]
    dataset = PromptDataset(clean_inputs, corrupt_inputs, clean_ids, corrupt_ids)
    dataloader = PromptDataLoader(
        dataset,
        seq_len=case.get_max_seq_len(),
        diverge_idx=0,
        batch_size=len(dataset),
        shuffle=False,
    )
    if len(dataloader) != 1:
        raise RuntimeError("official E32.2 settings require one full dataset batch")

    def categorical_kl(logits: Any, batch: Any) -> Any:
        answers = batch.answers.squeeze(dim=-1)
        if answers.shape != logits.shape:
            answers = torch.nn.functional.one_hot(
                answers, num_classes=model.cfg.d_vocab_out
            ).float()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return torch.nn.functional.kl_div(
            log_probs, answers, reduction="batchmean", log_target=False
        )

    prune_scores = mask_gradient_prune_scores(
        model=auto_model,
        dataloader=dataloader,
        official_edges=None,
        grad_function=settings.grad_function,
        answer_function=categorical_kl,
        mask_val=settings.mask_val,
        integrated_grad_samples=settings.integrated_grad_samples,
    )
    fine_scores = extract_fine_edge_scores(auto_model, prune_scores)
    promoted = promote_and_aggregate_fine_scores(case_id, fine_scores)
    accounting = eap_query_accounting(
        example_count=example_count,
        method=method,
        integrated_grad_samples=settings.integrated_grad_samples,
    )
    return OfficialEAPRun(
        case_id=case_id,
        settings=settings,
        fine_scores=fine_scores,
        promoted_scores=promoted,
        query_accounting=accounting,
    )


def validate_live_auto_circuit_graph(
    *, case_id: int, model: Any, device: str = "cpu"
) -> PromotedEAPScores:
    """Wrap a fresh model and prove all zero-score graph edges map with full coverage."""
    if case_id not in PANEL_CASE_IDS:
        raise ValueError(f"case_id must be one of {PANEL_CASE_IDS!r}")
    from auto_circuit.utils.graph_utils import patchable_model

    auto_model = patchable_model(
        model.to(device),
        factorized=OFFICIAL_EAP_SETTINGS["factorized"],
        slice_output=OFFICIAL_EAP_SETTINGS["slice_output"],
        separate_qkv=OFFICIAL_EAP_SETTINGS["separate_qkv"],
        device=__import__("torch").device(device),
    )
    fine_scores = extract_fine_edge_scores(auto_model, auto_model.new_prune_scores())
    return promote_and_aggregate_fine_scores(case_id, fine_scores)


def official_eap_run_to_dict(run: OfficialEAPRun) -> dict[str, Any]:
    """Convert a bounded baseline result to deterministic JSON-ready records."""
    promoted = run.promoted_scores
    return {
        "case_id": run.case_id,
        "settings": asdict(run.settings),
        "fine_scores": [asdict(score) for score in run.fine_scores],
        "promoted_scores": {
            "edge_order": [list(edge) for edge in promoted.edge_order],
            "signed_max_scores": list(promoted.signed_max_scores),
            "absolute_max_scores": list(promoted.absolute_max_scores),
            "collision_counts": list(promoted.collision_counts),
            "fine_edge_count": promoted.fine_edge_count,
            "internal_fine_edge_count": promoted.internal_fine_edge_count,
            "excluded_boundary_fine_edge_count": (
                promoted.excluded_boundary_fine_edge_count
            ),
            "coverage_complete": promoted.coverage_complete,
            "aggregation": {
                "signed_primary": "max_raw_score_per_promoted_edge",
                "absolute_sensitivity": "max_absolute_raw_score_per_promoted_edge",
                "support_parity": (
                    "strict_fine_threshold_then_official_binary_rerouting"
                ),
            },
        },
        "query_accounting": asdict(run.query_accounting),
    }


__all__ = [
    "AUTO_CIRCUIT_REPOSITORY",
    "AUTO_CIRCUIT_REVISION",
    "DEFAULT_AUTO_CIRCUIT_DIR",
    "OFFICIAL_EAP_IG_STEPS",
    "OFFICIAL_EAP_SETTINGS",
    "PINNED_AUTO_CIRCUIT_CORE_SHA256",
    "PINNED_EAP_ENVIRONMENT",
    "AutoCircuitGraphCounts",
    "EAPQueryAccounting",
    "EAPSettings",
    "FineEdgeScore",
    "OfficialEAPRun",
    "PromotedEAPScores",
    "eap_query_accounting",
    "expected_auto_circuit_counts",
    "extract_fine_edge_scores",
    "official_eap_run_to_dict",
    "official_eap_settings",
    "promote_and_aggregate_fine_scores",
    "run_official_eap_baseline",
    "validate_live_auto_circuit_graph",
    "verify_pinned_eap_runtime",
]
