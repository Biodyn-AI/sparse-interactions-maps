# ruff: noqa: EXE002
"""Pinned, minimal adapter for the official InterpBench model organisms.

This module deliberately avoids importing the rest of E32.  The official
benchmark currently needs Python 3.10/3.11, whereas the project environment is
Python 3.12.  Keeping imports lazy lets the same adapter describe and verify
artifacts in either environment.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import pickle
import re
import subprocess
import sys
import types
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Edge = tuple[str, str]

INTERPBENCH_REPOSITORY = "https://github.com/FlyingPumba/InterpBench.git"
INTERPBENCH_CODE_REVISION = "220791e4a9cd29957ed5edfa445e7c09152d7298"
INTERPBENCH_MODEL_REPOSITORY = "cybershiptrooper/InterpBench"
INTERPBENCH_MODEL_REVISION = "a1242a84f8a4d07d1be9a8f5ec710415198016b2"
TRACR_REPOSITORY = "https://github.com/FlyingPumba/tracr.git"
TRACR_REVISION = "98f4910e3966805c83d80eb087c7a4e3328171c6"
IIT_REPOSITORY = "https://github.com/cybershiptrooper/iit.git"
IIT_REVISION = "ed44cbfa255b543bb2d4da3d193ae97c9715c4f7"

DEFAULT_CACHE_ROOT = (
    Path.home() / ".cache" / "fundamental-mechinterp" / "e32-interpbench"
)
DEFAULT_OFFICIAL_CODE_DIR = DEFAULT_CACHE_ROOT / f"code-{INTERPBENCH_CODE_REVISION}"
DEFAULT_MODEL_ROOT = DEFAULT_CACHE_ROOT / f"models-{INTERPBENCH_MODEL_REVISION}"
DEFAULT_TRACR_DIR = DEFAULT_CACHE_ROOT / f"tracr-{TRACR_REVISION}"
DEFAULT_IIT_DIR = DEFAULT_CACHE_ROOT / f"iit-{IIT_REVISION}"
DEFAULT_ENV_DIR = DEFAULT_CACHE_ROOT / "venv-py311-tl1.19"

PANEL_CASE_IDS = (2, 21, 97)
REQUIRED_ARTIFACTS = (
    "edges.pkl",
    "ll_model_cfg.pkl",
    "ll_model.pth",
    "meta.json",
)


@dataclass(frozen=True)
class CaseManifest:
    """Immutable scientific identity and expected files for one organism."""

    case_id: int
    task: str
    role: str
    output_kind: str
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    d_mlp: int
    n_ctx: int
    d_vocab: int
    d_vocab_out: int
    raw_ground_truth_edges: int
    sha256: Mapping[str, str]


CASE_MANIFESTS = {
    2: CaseManifest(
        case_id=2,
        task="reverse the input sequence",
        role="panel",
        output_kind="categorical",
        n_layers=4,
        n_heads=4,
        d_model=56,
        d_head=14,
        d_mlp=224,
        n_ctx=10,
        d_vocab=28,
        d_vocab_out=26,
        raw_ground_truth_edges=27,
        sha256={
            "edges.pkl": "dd8ee919e31a1590d5e02e7d6aa4625e2d9e3002e3bbb74402f5a49e7d0be36b",
            "ll_model_cfg.pkl": "274196ba87c5a5cc25f5e3ed93a4c7dace5a7924905804c2ee6d2112c460f534",
            "ll_model.pth": "2b2ef568269ef820d7d6975fed90fabf517f044f26f9b10a8a3724a7da3c4f83",
            "meta.json": "c129be1d12e8293f56c5254d575399865ef1c89dfe67e717e1909d9e948d525c",
        },
    ),
    21: CaseManifest(
        case_id=21,
        task="extract unique tokens",
        role="panel_near_exact",
        output_kind="categorical",
        n_layers=4,
        n_heads=4,
        d_model=50,
        d_head=12,
        d_mlp=200,
        n_ctx=10,
        d_vocab=5,
        d_vocab_out=3,
        raw_ground_truth_edges=17,
        sha256={
            "edges.pkl": "a284ff6ddccac91bf12f0f28a6786e739428c948c111ef6ad524e481f5d1461d",
            "ll_model_cfg.pkl": "0199972dece30ed03433dd2e1cb712e7e718d94ff3cd4e2ca9c662253dc9f495",
            "ll_model.pth": "42e739d16ee075d148c5b8a65b71422da53b90a32d706821708240c75124e4af",
            "meta.json": "0562bcfe88845e70dc8a41c98953f60577ab635ffcb07104ddde8ff3dc23316f",
        },
    ),
    97: CaseManifest(
        case_id=97,
        task="scale each number by the sequence maximum",
        role="primary",
        output_kind="categorical",
        n_layers=3,
        n_heads=4,
        d_model=200,
        d_head=50,
        d_mlp=800,
        n_ctx=10,
        d_vocab=13,
        d_vocab_out=64,
        raw_ground_truth_edges=36,
        sha256={
            "edges.pkl": "afdfd64f4c8be661a44ae212318dcb22d1cd61cd023a9f510b0ed3cf9ceceac6",
            "ll_model_cfg.pkl": "2cb2f82ed87ef19e802ed3400f3b508631496f474aae19e07304666c6d1a06d5",
            "ll_model.pth": "7edb6ff1572d3b7b43e97ff512fe1bd0c78f47b00724906c72ef175338173028",
            "meta.json": "e7ea23a35d302a7952ab77e12344ab411837d3f5bd87992d211218aad3688bdd",
        },
    ),
    113: CaseManifest(
        case_id=113,
        task="reverse only sorted sequences",
        role="rejected_behavioral_failure",
        output_kind="categorical",
        n_layers=7,
        n_heads=4,
        d_model=88,
        d_head=22,
        d_mlp=352,
        n_ctx=10,
        d_vocab=32,
        d_vocab_out=30,
        raw_ground_truth_edges=66,
        sha256={
            "edges.pkl": "1aaf72d92f0df42e61bc20fd40672c902ce2e227f0df638b4b38111a09c06c93",
            "ll_model_cfg.pkl": "b5d64e11240c419a545b94fea7aabfae5ed77782f4f4de712846bc7ba453c7c9",
            "ll_model.pth": "ea7ec0d5b5e4ba6da236fedcb698feeef7c6987a179c5f31f8fd78a367ccf4a7",
            "meta.json": "ac56f39b05bf50fbf2961c895a43269b92610c3ea4a162e95799c00369d569c2",
        },
    ),
}


@dataclass(frozen=True)
class OfficialDataPair:
    """Official independently sampled clean and so-called corrupted data."""

    clean_inputs: Any
    clean_targets: Any
    corrupt_inputs: Any
    corrupt_targets: Any


@dataclass(frozen=True)
class BehaviorallyEligiblePair:
    """First pre-intervention row-pairs correct on both model branches."""

    pair: OfficialDataPair
    candidate_count: int
    eligible_candidate_count: int
    clean_incorrect_contexts: int
    source_incorrect_contexts: int
    either_incorrect_contexts: int
    selected_candidate_indices: tuple[int, ...]
    eligibility_physical_forwards: int
    eligibility_model_examples: int
    output_kind: str
    row_disjoint_eligible_candidate_count: int = 0
    row_overlap_excluded_contexts: int = 0
    excluded_token_row_count: int = 0
    selection_policy: str = "first_jointly_correct"


@dataclass(frozen=True)
class AccuracyMetrics:
    token_correct: int
    token_total: int
    token_accuracy: float
    context_correct: int
    context_total: int
    context_accuracy: float


def _case_manifest(case_id: int) -> CaseManifest:
    try:
        return CASE_MANIFESTS[case_id]
    except KeyError as error:
        raise ValueError(f"case {case_id} is not pinned in E32") from error


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading a checkpoint into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_artifact_paths(model_root: Path | str, case_id: int) -> dict[str, Path]:
    _case_manifest(case_id)
    case_dir = Path(model_root).expanduser().resolve() / str(case_id)
    return {name: case_dir / name for name in REQUIRED_ARTIFACTS}


def verify_case_artifacts(model_root: Path | str, case_id: int) -> dict[str, str]:
    """Require every official artifact to match its pinned HF LFS digest."""
    manifest = _case_manifest(case_id)
    observed: dict[str, str] = {}
    for name, path in case_artifact_paths(model_root, case_id).items():
        if not path.is_file():
            raise FileNotFoundError(f"missing case-{case_id} artifact: {path}")
        observed[name] = sha256_file(path)
        if observed[name] != manifest.sha256[name]:
            raise ValueError(
                f"SHA-256 mismatch for case {case_id} {name}: "
                f"expected {manifest.sha256[name]}, observed {observed[name]}"
            )
    return observed


def verify_git_revision(checkout: Path | str, expected_revision: str) -> str:
    """Return HEAD after requiring an exact pinned source checkout."""
    path = Path(checkout).expanduser().resolve()
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = completed.stdout.strip()
    if observed != expected_revision:
        raise ValueError(
            f"git revision mismatch for {path}: expected {expected_revision}, "
            f"observed {observed}"
        )
    return observed


def load_raw_ground_truth(
    path: Path | str, *, expected_sha256: str | None = None
) -> tuple[Edge, ...]:
    """Load the official ``list[tuple[str, str]]`` after optional hash check."""
    artifact_path = Path(path)
    if expected_sha256 is not None:
        observed = sha256_file(artifact_path)
        if observed != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {artifact_path}: expected "
                f"{expected_sha256}, observed {observed}"
            )
    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list) or not payload:
        raise TypeError("official ground truth must be a nonempty list")
    edges: list[Edge] = []
    for edge in payload:
        if (
            not isinstance(edge, tuple)
            or len(edge) != 2
            or not all(isinstance(node, str) and node for node in edge)
        ):
            raise TypeError("official ground truth entries must be string pairs")
        edges.append(edge)
    return tuple(edges)


def load_case_ground_truth(
    model_root: Path | str, case_id: int, *, verify: bool = True
) -> tuple[Edge, ...]:
    manifest = _case_manifest(case_id)
    path = case_artifact_paths(model_root, case_id)["edges.pkl"]
    expected = manifest.sha256["edges.pkl"] if verify else None
    edges = load_raw_ground_truth(path, expected_sha256=expected)
    if len(edges) != manifest.raw_ground_truth_edges:
        raise ValueError(
            f"case {case_id} ground-truth count changed: expected "
            f"{manifest.raw_ground_truth_edges}, observed {len(edges)}"
        )
    return edges


_INDEXED_NODE = re.compile(r"^(?P<name>.+)\[(?P<index>\d+)\]$")


def _node_name(node: str) -> str:
    match = _INDEXED_NODE.fullmatch(node)
    return match.group("name") if match else node


def _node_index(node: str) -> int | None:
    match = _INDEXED_NODE.fullmatch(node)
    return int(match.group("index")) if match else None


def _suffix(node: str) -> str:
    return _node_name(node).split(".")[-1]


def _prefix(node: str) -> str:
    return ".".join(_node_name(node).split(".")[:-1])


def _is_qkv_input(node: str) -> bool:
    return _suffix(node) in {"hook_q_input", "hook_k_input", "hook_v_input"}


def _is_qkv_output(node: str) -> bool:
    return _suffix(node) in {"hook_q", "hook_k", "hook_v"}


def _is_mlp_input(node: str) -> bool:
    return _suffix(node) == "hook_mlp_in"


def _is_embedding(node: str) -> bool:
    return _suffix(node) in {"hook_embed", "hook_pos_embed"}


def _is_residual(node: str) -> bool:
    return _suffix(node) in {"hook_resid_post", "hook_resid_pre"}


def _format_indexed(name: str, index: int | None) -> str:
    return name if index is None else f"{name}[{index}]"


def promote_official_node(node: str) -> str:
    """Promote one raw ACDC endpoint to InterpBench whole-head granularity."""
    if not isinstance(node, str) or not node:
        raise TypeError("node must be a nonempty string")
    suffix = _suffix(node)
    if suffix in {"hook_embed", "hook_pos_embed"}:
        return "blocks.0.hook_resid_pre"
    if suffix in {"hook_q_input", "hook_k_input", "hook_v_input"}:
        return _format_indexed(f"{_prefix(node)}.attn.hook_result", _node_index(node))
    if suffix in {"hook_q", "hook_k", "hook_v"}:
        return _format_indexed(f"{_prefix(node)}.hook_result", _node_index(node))
    if suffix == "hook_mlp_in":
        return f"{_prefix(node)}.hook_mlp_out"
    return node


def _is_internal_component(node: str) -> bool:
    name = _node_name(node)
    return (name.endswith(".attn.hook_result") and _node_index(node) is not None) or (
        name.endswith(".hook_mlp_out") and _node_index(node) is None
    )


def promoted_ground_truth_nodes(raw_edges: Sequence[Edge]) -> tuple[str, ...]:
    """Promote every GT endpoint, independently of surviving promoted edges."""
    if not raw_edges:
        raise ValueError("raw_edges must not be empty")
    nodes: list[str] = []
    seen: set[str] = set()
    for edge in raw_edges:
        if (
            not isinstance(edge, (tuple, list))
            or len(edge) != 2
            or not all(isinstance(node, str) and node for node in edge)
        ):
            raise TypeError("each raw edge must be a pair of nonempty strings")
        for raw_node in edge:
            node = promote_official_node(raw_node)
            if node not in seen:
                seen.add(node)
                nodes.append(node)
    return tuple(nodes)


def promoted_ground_truth_internal_nodes(
    raw_edges: Sequence[Edge],
) -> tuple[str, ...]:
    """Return patchable GT heads/MLPs, including edge-collapsed components."""
    return tuple(
        node
        for node in promoted_ground_truth_nodes(raw_edges)
        if _is_internal_component(node)
    )


def canonicalize_official_edges(raw_edges: Sequence[Edge]) -> tuple[Edge, ...]:
    """Reproduce InterpBench's whole-head circuit preparation exactly.

    SIIT supervision constrains an attention head as a whole.  The official
    evaluator therefore drops internal Q/K/V edges, redirects edges entering
    Q/K/V inputs to ``hook_result[head]``, redirects edges entering MLP inputs
    to MLP outputs, and collapses token/position embeddings to layer-0
    ``hook_resid_pre``.
    """
    if not raw_edges:
        raise ValueError("raw_edges must not be empty")
    validated: list[Edge] = []
    node_order: list[str] = []
    seen_nodes: set[str] = set()
    predecessors: dict[str, set[str]] = {}
    successors: dict[str, set[str]] = {}
    for edge in raw_edges:
        if (
            not isinstance(edge, (tuple, list))
            or len(edge) != 2
            or not all(isinstance(node, str) and node for node in edge)
        ):
            raise TypeError("each raw edge must be a pair of nonempty strings")
        source, target = edge
        validated.append((source, target))
        for node in (source, target):
            if node not in seen_nodes:
                seen_nodes.add(node)
                node_order.append(node)
            predecessors.setdefault(node, set())
            successors.setdefault(node, set())
        successors[source].add(target)
        predecessors[target].add(source)

    source_nodes = {node for node in node_order if not predecessors[node]}
    sink_nodes = {node for node in node_order if not successors[node]}
    canonical: list[Edge] = []
    seen_edges: set[Edge] = set()
    for source, target in validated:
        if _is_qkv_input(source) or _is_qkv_output(source) or _is_mlp_input(source):
            continue
        if _is_embedding(source) and _is_residual(target):
            continue
        if _is_residual(source):
            if not (source in source_nodes and not _is_residual(target)):
                continue
        elif _is_residual(target) and target not in sink_nodes:
            continue

        if _is_embedding(source):
            source = "blocks.0.hook_resid_pre"
        if _is_qkv_input(target):
            target = _format_indexed(
                f"{_prefix(target)}.attn.hook_result", _node_index(target)
            )
        elif _is_mlp_input(target):
            target = f"{_prefix(target)}.hook_mlp_out"

        edge = (source, target)
        if edge not in seen_edges:
            seen_edges.add(edge)
            canonical.append(edge)
    if not canonical:
        raise ValueError("official preparation removed every edge")
    return tuple(canonical)


def internal_ground_truth_edges(raw_edges: Sequence[Edge]) -> tuple[Edge, ...]:
    """Return only official promoted edges between patchable heads and MLPs."""
    return tuple(
        edge
        for edge in canonicalize_official_edges(raw_edges)
        if _is_internal_component(edge[0]) and _is_internal_component(edge[1])
    )


class _OrderedGraph:
    """Tiny insertion-ordered graph matching NetworkX edge iteration here."""

    def __init__(self) -> None:
        self.nodes: list[str] = []
        self._node_set: set[str] = set()
        self._targets: dict[str, list[str]] = {}
        self._edge_set: set[Edge] = set()

    def add_node(self, node: str) -> None:
        if node not in self._node_set:
            self._node_set.add(node)
            self.nodes.append(node)
            self._targets[node] = []

    def add_edge(self, source: str, target: str) -> None:
        self.add_node(source)
        self.add_node(target)
        edge = (source, target)
        if edge not in self._edge_set:
            self._edge_set.add(edge)
            self._targets[source].append(target)

    def edges(self) -> tuple[Edge, ...]:
        return tuple(
            (source, target)
            for source in self.nodes
            for target in self._targets[source]
        )


def build_full_raw_edges(n_layers: int, n_heads: int) -> tuple[Edge, ...]:
    """Reproduce the official full ACDC-level graph without ACDC dependency."""
    for value, label in ((n_layers, "n_layers"), (n_heads, "n_heads")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")

    graph = _OrderedGraph()
    graph.add_node("hook_embed")
    graph.add_node("hook_pos_embed")
    residual_writer_suffixes = (
        "hook_embed",
        "hook_pos_embed",
        "attn.hook_result",
        "hook_mlp_out",
    )

    def is_residual_writer(node: str) -> bool:
        return any(
            _node_name(node).endswith(suffix) for suffix in residual_writer_suffixes
        )

    for layer in range(n_layers):
        upstream_nodes = tuple(graph.nodes)
        for head in range(n_heads):
            for letter in "qkv":
                graph.add_edge(
                    f"blocks.{layer}.hook_{letter}_input[{head}]",
                    f"blocks.{layer}.attn.hook_{letter}[{head}]",
                )
            for letter in "qkv":
                graph.add_edge(
                    f"blocks.{layer}.attn.hook_{letter}[{head}]",
                    f"blocks.{layer}.attn.hook_result[{head}]",
                )
            for source in upstream_nodes:
                if is_residual_writer(source) and f"blocks.{layer}" not in source:
                    for letter in "qkv":
                        graph.add_edge(
                            source,
                            f"blocks.{layer}.hook_{letter}_input[{head}]",
                        )

        upstream_nodes = tuple(graph.nodes)
        mlp_input = f"blocks.{layer}.hook_mlp_in"
        mlp_output = f"blocks.{layer}.hook_mlp_out"
        graph.add_edge(mlp_input, mlp_output)
        for source in upstream_nodes:
            same_layer_attention = (
                f"blocks.{layer}" in source and "attn.hook_result" in source
            )
            if is_residual_writer(source) and (
                f"blocks.{layer}" not in source or same_layer_attention
            ):
                graph.add_edge(source, mlp_input)

    sink = f"blocks.{n_layers - 1}.hook_resid_post"
    graph.add_node(sink)
    for source in tuple(graph.nodes):
        if is_residual_writer(source):
            graph.add_edge(source, sink)
    return graph.edges()


def official_prepared_universe(
    n_layers: int, n_heads: int
) -> tuple[tuple[str, ...], tuple[Edge, ...]]:
    """Return the official prepared graph, including residual boundaries."""
    edges = canonicalize_official_edges(build_full_raw_edges(n_layers, n_heads))
    nodes: list[str] = []
    seen: set[str] = set()
    for source, target in edges:
        for node in (source, target):
            if node not in seen:
                seen.add(node)
                nodes.append(node)
    return tuple(nodes), edges


def patchable_internal_universe(
    n_layers: int, n_heads: int
) -> tuple[tuple[str, ...], tuple[Edge, ...]]:
    """Return the E32 atlas universe: heads/MLPs and valid internal DAG edges."""
    full_nodes, full_edges = official_prepared_universe(n_layers, n_heads)
    nodes = tuple(node for node in full_nodes if _is_internal_component(node))
    node_set = set(nodes)
    edges = tuple(
        (source, target)
        for source, target in full_edges
        if source in node_set and target in node_set
    )
    return nodes, edges


def load_pinned_model(
    model_root: Path | str, case_id: int, *, device: str = "cpu"
) -> Any:
    """Verify and load the exact official TransformerLens checkpoint."""
    verify_case_artifacts(model_root, case_id)
    paths = case_artifact_paths(model_root, case_id)
    import torch
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    with paths["ll_model_cfg.pkl"].open("rb") as handle:
        serialized_config = pickle.load(handle)
    if isinstance(serialized_config, dict):
        config = HookedTransformerConfig.from_dict(serialized_config)
    elif isinstance(serialized_config, HookedTransformerConfig):
        config = serialized_config
    else:
        raise TypeError(
            "ll_model_cfg.pkl must contain a dict or HookedTransformerConfig"
        )
    config.device = device
    model = HookedTransformer(config)
    # Loading pickle-backed PyTorch artifacts is safe here only because the
    # bytes were checked against the pinned official HF LFS digest above.
    state_dict = torch.load(paths["ll_model.pth"], map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_official_case(
    case_id: int, official_code_dir: Path | str = DEFAULT_OFFICIAL_CODE_DIR
) -> Any:
    """Instantiate a case from the exact official source checkout."""
    _case_manifest(case_id)
    code_dir = Path(official_code_dir).expanduser().resolve()
    verify_git_revision(code_dir, INTERPBENCH_CODE_REVISION)
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    # The benchmark imports pygraphviz only for optional plotting.  Its native
    # headers are absent from the deliberately small runtime, so a module stub
    # is sufficient for data generation and model evaluation.
    try:
        importlib.import_module("pygraphviz")
    except ModuleNotFoundError:
        sys.modules["pygraphviz"] = types.ModuleType("pygraphviz")

    module = importlib.import_module(
        f"circuits_benchmark.benchmark.cases.case_{case_id}"
    )
    module_path = Path(module.__file__).resolve()
    if code_dir not in module_path.parents:
        raise RuntimeError(
            f"case module came from {module_path}, not pinned checkout {code_dir}"
        )
    case_class = getattr(module, f"Case{case_id}")
    return case_class()


def generate_official_pair(
    case: Any,
    *,
    sample_count: int,
    clean_seed: int = 42_032,
    corrupt_seed: int = 42_033,
) -> OfficialDataPair:
    """Generate deterministic official clean and independent baseline data.

    InterpBench calls the second dataset ``corrupted``, but its default Tracr
    implementation simply calls the clean generator with another seed.  No
    semantic corruption or row-wise pairing is implied.
    """
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
    ):
        raise ValueError("sample_count must be a positive integer")
    import torch

    clean = case.get_clean_data(
        min_samples=sample_count,
        max_samples=sample_count,
        seed=clean_seed,
        unique_data=True,
    )
    corrupt = case.get_corrupted_data(
        min_samples=sample_count,
        max_samples=sample_count,
        seed=corrupt_seed,
        unique_data=True,
    )
    for label, dataset in (("clean", clean), ("corrupt", corrupt)):
        if len(dataset.inputs) != sample_count or len(dataset.targets) != sample_count:
            raise RuntimeError(
                f"official {label} generator returned {len(dataset.inputs)} rows; "
                f"expected {sample_count}"
            )
        unique_count = int(torch.unique(dataset.inputs, dim=0).shape[0])
        if unique_count != sample_count:
            # Pinned InterpBench compares str(input) against a set of tuples,
            # so its unique_data=True branch does not itself remove duplicates.
            raise RuntimeError(
                f"official {label} unique_data=True returned only {unique_count} "
                f"unique rows out of {sample_count}"
            )
    return OfficialDataPair(
        clean_inputs=clean.inputs.detach().cpu(),
        clean_targets=clean.targets.detach().cpu(),
        corrupt_inputs=corrupt.inputs.detach().cpu(),
        corrupt_targets=corrupt.targets.detach().cpu(),
    )


def _correct_context_mask(logits: Any, targets: Any) -> Any:
    import torch

    predicted = torch.as_tensor(logits, device="cpu")
    expected = torch.as_tensor(targets, device="cpu")
    if predicted.shape != expected.shape or predicted.ndim != 3:
        raise ValueError("logits and targets must share [batch, position, class]")
    if predicted.shape[1] < 2:
        raise ValueError("behavioral eligibility requires a non-BOS position")
    return (predicted.argmax(dim=-1)[:, 1:] == expected.argmax(dim=-1)[:, 1:]).all(
        dim=1
    )


def behaviorally_eligible_pair(
    model: Any,
    case: Any,
    *,
    required_count: int = 64,
    candidate_count: int = 80,
    clean_seed: int = 42_032,
    corrupt_seed: int = 42_033,
    device: str = "cpu",
    excluded_token_rows: Collection[Sequence[int]] = (),
    require_unique_selected_token_rows: bool = False,
) -> BehaviorallyEligiblePair:
    """Select the first row-pairs solved before any intervention.

    This is a deterministic behavioral-eligibility screen, not an estimate on
    the unfiltered task distribution.  Both the clean input and its row-paired
    independent source input must be exactly correct at every non-BOS token.
    """
    for value, label in (
        (required_count, "required_count"),
        (candidate_count, "candidate_count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if candidate_count < required_count:
        raise ValueError("candidate_count must be at least required_count")

    import torch

    is_categorical = getattr(case, "is_categorical", None)
    if not callable(is_categorical) or not bool(is_categorical()):
        raise NotImplementedError(
            "E32 behavioral eligibility currently requires a categorical official case"
        )
    candidates = generate_official_pair(
        case,
        sample_count=candidate_count,
        clean_seed=clean_seed,
        corrupt_seed=corrupt_seed,
    )
    model.to(device)
    model.eval()
    with torch.inference_mode():
        clean_logits = model(candidates.clean_inputs.to(device)).detach().cpu()
        source_logits = model(candidates.corrupt_inputs.to(device)).detach().cpu()
    clean_correct = _correct_context_mask(clean_logits, candidates.clean_targets)
    source_correct = _correct_context_mask(source_logits, candidates.corrupt_targets)
    eligible_mask = clean_correct & source_correct
    eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).flatten().tolist()
    normalized_exclusions: set[tuple[int, ...]] = set()
    for row in excluded_token_rows:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError("excluded token rows must be integer sequences")
        if not row or any(
            isinstance(token, bool) or not isinstance(token, int) for token in row
        ):
            raise ValueError("excluded token rows must be nonempty integer sequences")
        normalized_exclusions.add(tuple(row))

    enforce_disjoint = require_unique_selected_token_rows or bool(normalized_exclusions)
    row_overlap_excluded = 0
    if enforce_disjoint:
        used_rows = set(normalized_exclusions)
        selectable_indices: list[int] = []
        for index in eligible_indices:
            clean_row = tuple(int(token) for token in candidates.clean_inputs[index])
            source_row = tuple(int(token) for token in candidates.corrupt_inputs[index])
            if (
                clean_row == source_row
                or clean_row in used_rows
                or source_row in used_rows
            ):
                row_overlap_excluded += 1
                continue
            selectable_indices.append(int(index))
            used_rows.update((clean_row, source_row))
        selection_policy = (
            "first_jointly_correct_with_globally_unique_clean_source_rows_and_"
            "response_blind_exclusions"
        )
    else:
        selectable_indices = [int(index) for index in eligible_indices]
        selection_policy = "first_jointly_correct"

    if len(selectable_indices) < required_count:
        raise RuntimeError(
            f"behavioral eligibility found only {len(selectable_indices)} selectable "
            f"rows out of {candidate_count}; required {required_count}"
        )
    selected_indices = tuple(selectable_indices[:required_count])
    tensor_indices = torch.tensor(selected_indices, dtype=torch.long)

    def select(tensor: Any) -> Any:
        return tensor.index_select(0, tensor_indices)

    return BehaviorallyEligiblePair(
        pair=OfficialDataPair(
            clean_inputs=select(candidates.clean_inputs),
            clean_targets=select(candidates.clean_targets),
            corrupt_inputs=select(candidates.corrupt_inputs),
            corrupt_targets=select(candidates.corrupt_targets),
        ),
        candidate_count=candidate_count,
        eligible_candidate_count=len(eligible_indices),
        clean_incorrect_contexts=int((~clean_correct).sum()),
        source_incorrect_contexts=int((~source_correct).sum()),
        either_incorrect_contexts=int((~eligible_mask).sum()),
        selected_candidate_indices=selected_indices,
        eligibility_physical_forwards=2,
        eligibility_model_examples=2 * candidate_count,
        output_kind="categorical",
        row_disjoint_eligible_candidate_count=len(selectable_indices),
        row_overlap_excluded_contexts=row_overlap_excluded,
        excluded_token_row_count=len(normalized_exclusions),
        selection_policy=selection_policy,
    )


def categorical_accuracy(
    logits: Any, targets: Any, *, exclude_bos: bool = True
) -> AccuracyMetrics:
    """Exact token and whole-context accuracy for categorical Tracr tasks."""
    import torch

    predicted = torch.as_tensor(logits, device="cpu")
    expected = torch.as_tensor(targets, device="cpu")
    if predicted.shape != expected.shape or predicted.ndim != 3:
        raise ValueError("logits and targets must share [batch, position, class]")
    if not bool(torch.isfinite(predicted).all()) or not bool(
        torch.isfinite(expected).all()
    ):
        raise ValueError("logits and targets must be finite")
    matches = predicted.argmax(dim=-1) == expected.argmax(dim=-1)
    if exclude_bos:
        if matches.shape[1] < 2:
            raise ValueError("cannot exclude BOS from a one-position sequence")
        matches = matches[:, 1:]
    token_correct = int(matches.sum())
    token_total = matches.numel()
    context_matches = matches.all(dim=1)
    context_correct = int(context_matches.sum())
    context_total = context_matches.numel()
    return AccuracyMetrics(
        token_correct=token_correct,
        token_total=token_total,
        token_accuracy=token_correct / token_total,
        context_correct=context_correct,
        context_total=context_total,
        context_accuracy=context_correct / context_total,
    )


def read_meta(model_root: Path | str, case_id: int) -> dict[str, Any]:
    """Read verified JSON-only training metadata."""
    verify_case_artifacts(model_root, case_id)
    with case_artifact_paths(model_root, case_id)["meta.json"].open() as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise TypeError("meta.json must contain an object")
    return metadata
