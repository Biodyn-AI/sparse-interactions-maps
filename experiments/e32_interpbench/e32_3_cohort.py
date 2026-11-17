# ruff: noqa: EXE002
"""Pinned organism cohort and input custody for prospective E32.3.

This module adds new organisms without changing the sealed E32.1/E32.2
adapter.  It owns only artifact verification, exact categorical geometry, and
pre-intervention row selection.  It does not run or score any intervention.
"""

from __future__ import annotations

import importlib
import math
import pickle
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from experiments.e32_interpbench.adapter import (
    DEFAULT_MODEL_ROOT,
    DEFAULT_OFFICIAL_CODE_DIR,
    INTERPBENCH_CODE_REVISION,
    INTERPBENCH_MODEL_REVISION,
    REQUIRED_ARTIFACTS,
    behaviorally_eligible_pair,
    generate_official_pair,
    internal_ground_truth_edges,
    load_raw_ground_truth,
    patchable_internal_universe,
    promoted_ground_truth_internal_nodes,
    sha256_file,
    verify_git_revision,
)
from experiments.e32_interpbench.component_patching import class_center_logits
from experiments.e32_interpbench.e32_2_acquisition import (
    assert_role_rows_fresh,
    token_rows,
)

Edge = tuple[str, str]
TokenRow = tuple[int, ...]

CASE_ORDER = (58, 129, 25)
PROSPECTIVE_CASE_IDS = CASE_ORDER


@dataclass(frozen=True)
class CohortCaseManifest:
    """Exact scientific identity of one prospective organism."""

    case_id: int
    task: str
    output_kind: Literal["categorical"]
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    d_mlp: int
    n_ctx: int
    d_vocab: int
    d_vocab_out: int
    attention_dir: str
    raw_ground_truth_edges: int
    internal_universe_nodes: int
    internal_universe_edges: int
    internal_ground_truth_nodes: tuple[str, ...]
    internal_ground_truth_edges: tuple[Edge, ...]
    sha256: Mapping[str, str]


CASE_MANIFESTS: dict[int, CohortCaseManifest] = {
    58: CohortCaseManifest(
        case_id=58,
        task="mirror the first half of the sequence to the second half",
        output_kind="categorical",
        n_layers=3,
        n_heads=4,
        d_model=32,
        d_head=8,
        d_mlp=128,
        n_ctx=10,
        d_vocab=13,
        d_vocab_out=11,
        attention_dir="bidirectional",
        raw_ground_truth_edges=25,
        internal_universe_nodes=15,
        internal_universe_edges=87,
        internal_ground_truth_nodes=(
            "blocks.0.attn.hook_result[0]",
            "blocks.2.attn.hook_result[0]",
            "blocks.1.hook_mlp_out",
            "blocks.0.hook_mlp_out",
        ),
        internal_ground_truth_edges=(
            (
                "blocks.0.attn.hook_result[0]",
                "blocks.0.hook_mlp_out",
            ),
            ("blocks.0.hook_mlp_out", "blocks.1.hook_mlp_out"),
            (
                "blocks.1.hook_mlp_out",
                "blocks.2.attn.hook_result[0]",
            ),
        ),
        sha256={
            "edges.pkl": "4edacbb2f328f7d3526a6658a58484634a73124ea79cace68231ae0bb1ab20ea",
            "ll_model_cfg.pkl": "8cc71fd34570fd0bef7a3f511b8675d7cbaec6db017ce3cfbee7a74f8a23e501",
            "ll_model.pth": "95278b3bf4897a27606f35ae5ed251fb022d9e4158f7575d32ab42d6fa142e20",
            "meta.json": "1cfffe661e1435f142d96ab22909dc78623087ec98adbc0a8bda025fda348881",
        },
    ),
    129: CohortCaseManifest(
        case_id=129,
        task="check whether every element is divisible by two",
        output_kind="categorical",
        n_layers=3,
        n_heads=4,
        d_model=4,
        d_head=1,
        d_mlp=16,
        n_ctx=10,
        d_vocab=32,
        d_vocab_out=2,
        attention_dir="bidirectional",
        raw_ground_truth_edges=16,
        internal_universe_nodes=15,
        internal_universe_edges=87,
        internal_ground_truth_nodes=(
            "blocks.0.hook_mlp_out",
            "blocks.1.attn.hook_result[0]",
            "blocks.1.hook_mlp_out",
            "blocks.2.hook_mlp_out",
        ),
        internal_ground_truth_edges=(
            (
                "blocks.0.hook_mlp_out",
                "blocks.1.attn.hook_result[0]",
            ),
            (
                "blocks.1.attn.hook_result[0]",
                "blocks.1.hook_mlp_out",
            ),
            ("blocks.1.hook_mlp_out", "blocks.2.hook_mlp_out"),
        ),
        sha256={
            "edges.pkl": "b01d4b9b1e5740fd2934079065cb57785bb111918826a8e8ad7b5b4af00c7bcc",
            "ll_model_cfg.pkl": "a4c611ecd0f8c57d44f555697540a82578e20f4dbe88bb1bd25132bd06244fa2",
            "ll_model.pth": "f497c70d4f781598a2d86469fc884a7aee35013d2ca2851e03a40077224fc6ab",
            "meta.json": "d9329a1cbcd6819eaab7573093a4eb96ed417c9efd2206270517d6897eb93c5b",
        },
    ),
    25: CohortCaseManifest(
        case_id=25,
        task="normalize token frequencies to the interval from zero to one",
        output_kind="categorical",
        n_layers=2,
        n_heads=4,
        d_model=62,
        d_head=15,
        d_mlp=248,
        n_ctx=10,
        d_vocab=5,
        d_vocab_out=56,
        attention_dir="bidirectional",
        raw_ground_truth_edges=26,
        internal_universe_nodes=10,
        internal_universe_edges=33,
        internal_ground_truth_nodes=(
            "blocks.0.attn.hook_result[0]",
            "blocks.0.attn.hook_result[1]",
            "blocks.0.hook_mlp_out",
            "blocks.1.hook_mlp_out",
        ),
        internal_ground_truth_edges=(
            (
                "blocks.0.attn.hook_result[0]",
                "blocks.0.hook_mlp_out",
            ),
            (
                "blocks.0.attn.hook_result[1]",
                "blocks.0.hook_mlp_out",
            ),
            ("blocks.0.hook_mlp_out", "blocks.1.hook_mlp_out"),
        ),
        sha256={
            "edges.pkl": "bd358562dc56469dbf03912dac0c5d3d5489b0f2dadcb85a7cd7c24f9582a74a",
            "ll_model_cfg.pkl": "c5df91fe7e78fbfc9fe45fe234027fb7c15ecd4c9a0b7a4776da0bf067e09db5",
            "ll_model.pth": "0e72ecd925276352d1240da0b40019b38ef254cbfdad61b234d96672fc4ae2e2",
            "meta.json": "9760198867e197a1aeba2e60440ad37e236f57c14f21c6e7d73a187b67b34c0c",
        },
    ),
}


@dataclass(frozen=True)
class SplitRequest:
    """Response-blind scientific input request."""

    case_id: int
    split: Literal["discovery", "confirmation"]
    required_count: int
    candidate_count: int
    clean_seed: int
    corrupt_seed: int


@dataclass(frozen=True)
class SmokeRequest:
    """Pre-intervention behavior-smoke rows barred from science panels."""

    case_id: int
    sample_count: int
    clean_seed: int
    corrupt_seed: int


_SPLITS: dict[int, dict[str, tuple[int, int, int, int]]] = {
    58: {
        "discovery": (64, 96, 758_100, 758_101),
        "confirmation": (128, 192, 758_200, 758_201),
    },
    129: {
        "discovery": (64, 96, 829_100, 829_101),
        "confirmation": (128, 192, 829_200, 829_201),
    },
    25: {
        "discovery": (64, 96, 725_100, 725_101),
        "confirmation": (128, 192, 725_200, 725_201),
    },
}

SMOKE_REQUESTS = {
    58: SmokeRequest(58, 96, 758_000, 758_001),
    129: SmokeRequest(129, 96, 829_000, 829_001),
    25: SmokeRequest(25, 96, 725_000, 725_001),
}

# These capacities were computed without intervention responses.  Each split
# excludes all smoke identities; confirmation additionally excludes both
# selected discovery roles.  They are pinned so generator drift fails closed.
EXPECTED_RESPONSE_BLIND_CAPACITY = {
    58: {"smoke_unique_rows": 192, "discovery": 96, "confirmation": 192},
    129: {"smoke_unique_rows": 192, "discovery": 96, "confirmation": 192},
    25: {"smoke_unique_rows": 190, "discovery": 91, "confirmation": 183},
}


@dataclass(frozen=True)
class ExactGroundTruth:
    case_id: int
    raw_edges: tuple[Edge, ...]
    raw_edge_count: int
    internal_nodes: tuple[str, ...]
    internal_edges: tuple[Edge, ...]


@dataclass(frozen=True)
class CasePanel:
    case_id: int
    discovery: Any
    confirmation: Any
    smoke_rows: tuple[TokenRow, ...]


def _manifest(case_id: int) -> CohortCaseManifest:
    try:
        return CASE_MANIFESTS[case_id]
    except KeyError as error:
        raise ValueError(f"case_id must be one of {CASE_ORDER!r}") from error


def split_request(case_id: int, split: str) -> SplitRequest:
    """Return one exact frozen request; adaptive panel growth is forbidden."""
    _manifest(case_id)
    try:
        required, candidates, clean_seed, corrupt_seed = _SPLITS[case_id][split]
    except KeyError as error:
        raise ValueError("split must be 'discovery' or 'confirmation'") from error
    return SplitRequest(
        case_id=case_id,
        split=split,  # type: ignore[arg-type]
        required_count=required,
        candidate_count=candidates,
        clean_seed=clean_seed,
        corrupt_seed=corrupt_seed,
    )


def case_artifact_paths(model_root: Path | str, case_id: int) -> dict[str, Path]:
    """Return exact artifact locations after validating cohort membership."""
    _manifest(case_id)
    case_dir = Path(model_root).expanduser().resolve() / str(case_id)
    return {name: case_dir / name for name in REQUIRED_ARTIFACTS}


def verify_case_artifacts(model_root: Path | str, case_id: int) -> dict[str, str]:
    """Require all official bytes to match the frozen model revision."""
    manifest = _manifest(case_id)
    observed: dict[str, str] = {}
    for name, path in case_artifact_paths(model_root, case_id).items():
        if not path.is_file():
            raise FileNotFoundError(f"missing case-{case_id} artifact: {path}")
        observed[name] = sha256_file(path)
        if observed[name] != manifest.sha256[name]:
            raise ValueError(
                f"SHA-256 mismatch for case {case_id} {name}: expected "
                f"{manifest.sha256[name]}, observed {observed[name]}"
            )
    return observed


def case_universe(case_id: int) -> tuple[tuple[str, ...], tuple[Edge, ...]]:
    """Return the exact ordered whole-component DAG for one case."""
    manifest = _manifest(case_id)
    nodes, edges = patchable_internal_universe(manifest.n_layers, manifest.n_heads)
    if (len(nodes), len(edges)) != (
        manifest.internal_universe_nodes,
        manifest.internal_universe_edges,
    ):
        raise RuntimeError("case universe differs from frozen counts")
    return nodes, edges


def load_exact_ground_truth(model_root: Path | str, case_id: int) -> ExactGroundTruth:
    """Load official support and require exact promoted identities."""
    manifest = _manifest(case_id)
    path = case_artifact_paths(model_root, case_id)["edges.pkl"]
    raw = load_raw_ground_truth(path, expected_sha256=manifest.sha256["edges.pkl"])
    if len(raw) != manifest.raw_ground_truth_edges:
        raise ValueError("raw ground-truth edge count differs from frozen manifest")
    nodes = promoted_ground_truth_internal_nodes(raw)
    edges = internal_ground_truth_edges(raw)
    if nodes != manifest.internal_ground_truth_nodes:
        raise ValueError("promoted internal ground-truth node identities changed")
    if edges != manifest.internal_ground_truth_edges:
        raise ValueError("promoted internal ground-truth edge identities changed")
    universe_nodes, universe_edges = case_universe(case_id)
    if not set(nodes).issubset(universe_nodes) or not set(edges).issubset(
        universe_edges
    ):
        raise RuntimeError("ground truth escapes the frozen patchable universe")
    return ExactGroundTruth(case_id, raw, len(raw), nodes, edges)


def _validate_model_config(case_id: int, config: Any) -> None:
    manifest = _manifest(case_id)
    expected = {
        "n_layers": manifest.n_layers,
        "n_heads": manifest.n_heads,
        "d_model": manifest.d_model,
        "d_head": manifest.d_head,
        "d_mlp": manifest.d_mlp,
        "n_ctx": manifest.n_ctx,
        "d_vocab": manifest.d_vocab,
        "d_vocab_out": manifest.d_vocab_out,
        "attention_dir": manifest.attention_dir,
    }
    observed = {name: getattr(config, name, None) for name in expected}
    if observed != expected:
        raise ValueError(
            f"case-{case_id} model config differs from frozen geometry: "
            f"expected {expected!r}, observed {observed!r}"
        )


def load_verified_model(
    case_id: int,
    *,
    model_root: Path | str = DEFAULT_MODEL_ROOT,
    device: str = "cpu",
) -> Any:
    """Load a cohort checkpoint only after exact artifact verification."""
    verify_case_artifacts(model_root, case_id)
    paths = case_artifact_paths(model_root, case_id)
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    with paths["ll_model_cfg.pkl"].open("rb") as handle:
        serialized = pickle.load(handle)
    if isinstance(serialized, dict):
        config = HookedTransformerConfig.from_dict(serialized)
    elif isinstance(serialized, HookedTransformerConfig):
        config = serialized
    else:
        raise TypeError(
            "ll_model_cfg.pkl must contain a dict or HookedTransformerConfig"
        )
    _validate_model_config(case_id, config)
    config.device = device
    model = HookedTransformer(config)
    state_dict = torch.load(paths["ll_model.pth"], map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_verified_official_case(
    case_id: int,
    *,
    official_code_dir: Path | str = DEFAULT_OFFICIAL_CODE_DIR,
) -> Any:
    """Instantiate a cohort case from the exact pinned source checkout."""
    _manifest(case_id)
    code_dir = Path(official_code_dir).expanduser().resolve()
    verify_git_revision(code_dir, INTERPBENCH_CODE_REVISION)
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    try:
        importlib.import_module("pygraphviz")
    except ModuleNotFoundError:
        # InterpBench imports this optional plotting dependency during module
        # loading, but E32.3 never calls plotting code.
        sys.modules["pygraphviz"] = types.ModuleType("pygraphviz")
    module = importlib.import_module(
        f"circuits_benchmark.benchmark.cases.case_{case_id}"
    )
    module_path = Path(module.__file__).resolve()
    if code_dir not in module_path.parents:
        raise RuntimeError(
            f"case module came from {module_path}, not pinned checkout {code_dir}"
        )
    return getattr(module, f"Case{case_id}")()


def project_categorical_output(case_id: int, full_output: torch.Tensor) -> torch.Tensor:
    """Class-center exact logits and remove the unspecified BOS position."""
    manifest = _manifest(case_id)
    if (
        not isinstance(full_output, torch.Tensor)
        or full_output.ndim != 3
        or full_output.shape[1:] != (manifest.n_ctx, manifest.d_vocab_out)
        or not full_output.is_floating_point()
        or not bool(torch.isfinite(full_output).all())
    ):
        raise ValueError(
            "categorical output geometry must be finite floating "
            f"[batch, {manifest.n_ctx}, {manifest.d_vocab_out}]"
        )
    return class_center_logits(full_output)[..., 1:, :]


def _deduplicate_rows(rows: Sequence[TokenRow]) -> tuple[TokenRow, ...]:
    return tuple(dict.fromkeys(rows))


def smoke_excluded_rows(
    case_id: int,
    case: Any,
    *,
    generate_pair: Callable[..., Any] = generate_official_pair,
) -> tuple[TokenRow, ...]:
    """Regenerate and return every unique row identity from the fixed smoke."""
    request = SMOKE_REQUESTS[_manifest(case_id).case_id]
    pair = generate_pair(
        case,
        sample_count=request.sample_count,
        clean_seed=request.clean_seed,
        corrupt_seed=request.corrupt_seed,
    )
    clean = token_rows(pair.clean_inputs, label="smoke clean inputs")
    corrupt = token_rows(pair.corrupt_inputs, label="smoke corrupt inputs")
    if len(clean) != request.sample_count or len(corrupt) != request.sample_count:
        raise RuntimeError("smoke generator did not return both complete roles")
    rows = _deduplicate_rows((*clean, *corrupt))
    expected = EXPECTED_RESPONSE_BLIND_CAPACITY[case_id]["smoke_unique_rows"]
    if generate_pair is generate_official_pair and len(rows) != expected:
        raise RuntimeError(
            f"case-{case_id} smoke unique-row count changed: expected "
            f"{expected}, observed {len(rows)}"
        )
    return rows


def _validate_tensor_geometry(
    case_id: int,
    value: Any,
    *,
    count: int,
    target: bool,
    label: str,
) -> torch.Tensor:
    manifest = _manifest(case_id)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    expected_shape = (
        (count, manifest.n_ctx, manifest.d_vocab_out)
        if target
        else (count, manifest.n_ctx)
    )
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{label} geometry must be exactly {expected_shape!r}")
    if target:
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} must be finite floating categorical targets")
    elif value.is_floating_point() or value.dtype == torch.bool:
        raise ValueError(f"{label} must contain integer token ids")
    return value.detach().cpu()


def _eligible_pair(value: Any, case_id: int, count: int, label: str) -> Any:
    try:
        pair = value.pair
    except AttributeError as error:
        raise TypeError(f"{label} must expose an eligible .pair") from error
    for role in ("clean", "corrupt"):
        _validate_tensor_geometry(
            case_id,
            getattr(pair, f"{role}_inputs"),
            count=count,
            target=False,
            label=f"{label} {role} inputs",
        )
        _validate_tensor_geometry(
            case_id,
            getattr(pair, f"{role}_targets"),
            count=count,
            target=True,
            label=f"{label} {role} targets",
        )
    return pair


def _selected_rows(
    value: Any, case_id: int, count: int, label: str
) -> tuple[TokenRow, ...]:
    pair = _eligible_pair(value, case_id, count, label)
    return (
        *token_rows(pair.clean_inputs, label=f"{label} clean inputs"),
        *token_rows(pair.corrupt_inputs, label=f"{label} corrupt inputs"),
    )


def validate_case_panel(panel: CasePanel) -> dict[str, int | str]:
    """Require smoke, role, and cross-split row custody for one case."""
    if not isinstance(panel, CasePanel):
        raise TypeError("panel must be a CasePanel")
    case_id = _manifest(panel.case_id).case_id
    discovery_request = split_request(case_id, "discovery")
    confirmation_request = split_request(case_id, "confirmation")
    smoke = tuple(panel.smoke_rows)
    if len(smoke) != len(set(smoke)):
        raise ValueError("smoke exclusions must be unique row identities")

    discovery_pair = _eligible_pair(
        panel.discovery,
        case_id,
        discovery_request.required_count,
        "discovery",
    )
    discovery_freshness = assert_role_rows_fresh(
        clean_inputs=discovery_pair.clean_inputs,
        corrupt_inputs=discovery_pair.corrupt_inputs,
        excluded_rows=smoke,
        expected_count=discovery_request.required_count,
        role=f"case{case_id}_discovery",
    )
    discovery_rows = _selected_rows(
        panel.discovery,
        case_id,
        discovery_request.required_count,
        "discovery",
    )

    confirmation_pair = _eligible_pair(
        panel.confirmation,
        case_id,
        confirmation_request.required_count,
        "confirmation",
    )
    confirmation_freshness = assert_role_rows_fresh(
        clean_inputs=confirmation_pair.clean_inputs,
        corrupt_inputs=confirmation_pair.corrupt_inputs,
        excluded_rows=_deduplicate_rows((*smoke, *discovery_rows)),
        expected_count=confirmation_request.required_count,
        role=f"case{case_id}_confirmation",
    )
    return {
        "case_id": case_id,
        "model_revision": INTERPBENCH_MODEL_REVISION,
        "smoke_unique_rows": len(smoke),
        "discovery_unique_role_rows": int(
            discovery_freshness["selected_unique_role_rows"]
        ),
        "confirmation_unique_role_rows": int(
            confirmation_freshness["selected_unique_role_rows"]
        ),
    }


def _acquire_case_panel(
    case_id: int,
    *,
    model: Any,
    case: Any,
    device: str,
    generate_pair: Callable[..., Any],
    select_eligible: Callable[..., Any],
) -> CasePanel:
    """Implementation seam used by tests; production wrapper fixes callables."""
    _manifest(case_id)
    smoke = smoke_excluded_rows(case_id, case, generate_pair=generate_pair)
    discovery_request = split_request(case_id, "discovery")
    discovery = select_eligible(
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
    discovery_rows = _selected_rows(
        discovery,
        case_id,
        discovery_request.required_count,
        "discovery",
    )
    confirmation_request = split_request(case_id, "confirmation")
    confirmation = select_eligible(
        model,
        case,
        required_count=confirmation_request.required_count,
        candidate_count=confirmation_request.candidate_count,
        clean_seed=confirmation_request.clean_seed,
        corrupt_seed=confirmation_request.corrupt_seed,
        device=device,
        excluded_token_rows=_deduplicate_rows((*smoke, *discovery_rows)),
        require_unique_selected_token_rows=True,
    )
    panel = CasePanel(case_id, discovery, confirmation, smoke)
    validate_case_panel(panel)
    return panel


def acquire_case_panel(
    case_id: int,
    *,
    model: Any,
    case: Any,
    device: str = "cpu",
) -> CasePanel:
    """Acquire the fixed D/C input panel with non-injectable production logic."""
    return _acquire_case_panel(
        case_id,
        model=model,
        case=case,
        device=device,
        generate_pair=generate_official_pair,
        select_eligible=behaviorally_eligible_pair,
    )


def cohort_manifest() -> dict[str, Any]:
    """Return the model-free cohort fragment for later registration."""
    return {
        "schema_version": "e32_3_sparse_core_cohort_v1",
        "prospective_case_ids": list(CASE_ORDER),
        "model_revision": INTERPBENCH_MODEL_REVISION,
        "code_revision": INTERPBENCH_CODE_REVISION,
        "cases": {
            str(case_id): {
                "task": CASE_MANIFESTS[case_id].task,
                "output_kind": CASE_MANIFESTS[case_id].output_kind,
                "artifact_sha256": dict(CASE_MANIFESTS[case_id].sha256),
                "discovery": split_request(case_id, "discovery").__dict__,
                "confirmation": split_request(case_id, "confirmation").__dict__,
                "smoke": SMOKE_REQUESTS[case_id].__dict__,
                "response_blind_capacity": dict(
                    EXPECTED_RESPONSE_BLIND_CAPACITY[case_id]
                ),
            }
            for case_id in CASE_ORDER
        },
        "claims": {
            "intervention_outcomes_opened_during_cohort_selection": False,
            "smoke_rows_excluded_from_scientific_panels": True,
            "discovery_confirmation_rows_globally_disjoint_within_case": True,
            "categorical_only": True,
            "continuous_case_39_excluded": True,
            "low_support_case_124_excluded": True,
        },
    }


def operational_eligibility_examples() -> int:
    """Exact pre-intervention model examples for all scientific selections."""
    return 2 * sum(
        split_request(case_id, split).candidate_count
        for case_id in CASE_ORDER
        for split in ("discovery", "confirmation")
    )


def top_half_count(case_id: int) -> int:
    """Deterministic component-core budget inherited from E32.2."""
    return math.ceil(_manifest(case_id).internal_universe_nodes / 2)
