# ruff: noqa: EXE002
"""Pinned artifacts and input custody for the E32.5 scaling cohort.

This module deliberately stops at artifact verification, deterministic panel
generation, response-blind exclusion, and pre-intervention behavior screening.
It owns no selector, intervention runner, support authorization, or scorer.
The sealed E32.4 sources are imported read-only where a generic primitive is
already available; no E32.4 file needs to be changed for this cohort.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import pickle
import sys
import types
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from experiments.e32_interpbench.adapter import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_OFFICIAL_CODE_DIR,
    INTERPBENCH_CODE_REVISION,
    INTERPBENCH_MODEL_REVISION,
    REQUIRED_ARTIFACTS,
    BehaviorallyEligiblePair,
    OfficialDataPair,
    _correct_context_mask,
    generate_official_pair,
    internal_ground_truth_edges,
    load_raw_ground_truth,
    patchable_internal_universe,
    promoted_ground_truth_internal_nodes,
    sha256_file,
    verify_git_revision,
)
from experiments.e32_interpbench.e32_2_acquisition import token_rows
from experiments.e32_interpbench.e32_3_cohort import (
    CASE_MANIFESTS as E32_3_CASE_MANIFESTS,
)

Edge = tuple[str, str]
TokenRow = tuple[int, ...]
ClaimRole = Literal["within_model_repair_only", "prospective_model_generalization"]
Split = Literal["development", "repair_confirmation", "discovery", "confirmation"]

CASE_ORDER = (58, 63, 103, 82)
GENERALIZATION_CASE_ORDER = (63, 103, 82)
SPLIT_ORDER_BY_CASE: Mapping[int, tuple[Split, Split]] = {
    58: ("development", "repair_confirmation"),
    63: ("discovery", "confirmation"),
    103: ("discovery", "confirmation"),
    82: ("discovery", "confirmation"),
}

# Do not add new directories to the already sealed E32.4 model cache tree.
DEFAULT_E32_5_MODEL_ROOT = (
    DEFAULT_CACHE_ROOT / f"models-e32-5-{INTERPBENCH_MODEL_REVISION}"
)


@dataclass(frozen=True)
class CohortCaseManifest:
    """Exact model and public-task identity for one E32.5 organism."""

    case_id: int
    task: str
    claim_role: ClaimRole
    output_kind: Literal["categorical"]
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    d_mlp: int
    n_ctx: int
    d_vocab: int
    d_vocab_out: int
    attention_dir: Literal["bidirectional"]
    model_parameter_count: int
    raw_ground_truth_edges: int
    internal_ground_truth_nodes: tuple[str, ...]
    internal_ground_truth_edges: tuple[Edge, ...]
    exact_internal_fine_ground_truth_edges: int
    sha256: Mapping[str, str]


_CASE58_ANCESTRY = E32_3_CASE_MANIFESTS[58]

CASE_MANIFESTS: dict[int, CohortCaseManifest] = {
    58: CohortCaseManifest(
        case_id=58,
        task="mirror the first half of the sequence to the second half",
        claim_role="within_model_repair_only",
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
        model_parameter_count=38_827,
        raw_ground_truth_edges=25,
        internal_ground_truth_nodes=_CASE58_ANCESTRY.internal_ground_truth_nodes,
        internal_ground_truth_edges=_CASE58_ANCESTRY.internal_ground_truth_edges,
        exact_internal_fine_ground_truth_edges=3,
        # Reuse the exact already-verified E32.3/E32.4 bytes.
        sha256=_CASE58_ANCESTRY.sha256,
    ),
    63: CohortCaseManifest(
        case_id=63,
        task="replace each element by the number of smaller sequence elements",
        claim_role="prospective_model_generalization",
        output_kind="categorical",
        n_layers=2,
        n_heads=4,
        d_model=24,
        d_head=6,
        d_mlp=96,
        n_ctx=10,
        d_vocab=13,
        d_vocab_out=10,
        attention_dir="bidirectional",
        model_parameter_count=15_058,
        raw_ground_truth_edges=11,
        internal_ground_truth_nodes=(
            "blocks.0.attn.hook_result[0]",
            "blocks.0.hook_mlp_out",
        ),
        internal_ground_truth_edges=(
            (
                "blocks.0.attn.hook_result[0]",
                "blocks.0.hook_mlp_out",
            ),
        ),
        exact_internal_fine_ground_truth_edges=1,
        sha256={
            "edges.pkl": (
                "3156b2e517b8c45e76ea28f8c2ccbcda226f636cd44a809c908e167871d2cdab"
            ),
            "ll_model_cfg.pkl": (
                "04187613894b7f89b32d1355605a0a21fde32558454a58fbaa3ac1a3386cae65"
            ),
            "ll_model.pth": (
                "73cd4fe1e237568f33000e0b014f3e265e540c029db551bcb9a6356b73eda684"
            ),
            "meta.json": (
                "5eb2f25b9a37894c89a79371452e5213e6d335045cf20aad560ca931a7cdad87"
            ),
        },
    ),
    103: CohortCaseManifest(
        case_id=103,
        task="swap consecutive sequence positions",
        claim_role="prospective_model_generalization",
        output_kind="categorical",
        n_layers=3,
        n_heads=4,
        d_model=24,
        d_head=6,
        d_mlp=96,
        n_ctx=10,
        d_vocab=13,
        d_vocab_out=11,
        attention_dir="bidirectional",
        model_parameter_count=22_211,
        raw_ground_truth_edges=27,
        internal_ground_truth_nodes=(
            "blocks.2.attn.hook_result[0]",
            "blocks.0.attn.hook_result[0]",
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
        exact_internal_fine_ground_truth_edges=3,
        sha256={
            "edges.pkl": (
                "4df9e5eb129553e810673c7ea9f138df364f9461861df969e00972518748db4b"
            ),
            "ll_model_cfg.pkl": (
                "a7628439ed419829404ac8619bc94b8424d1eb5f6e00ce2384a591ef1771ab98"
            ),
            "ll_model.pth": (
                "d2deec61e135c0e16e0f0c1884c91141e8486825d8c697045e32c422cc2862bb"
            ),
            "meta.json": (
                "8c0487bac963462baac77e32612c40cad008d138b5cf22c49eed0734ecfae44f"
            ),
        },
    ),
    82: CohortCaseManifest(
        case_id=82,
        task="halve elements in the second half of the sequence",
        claim_role="prospective_model_generalization",
        output_kind="categorical",
        n_layers=4,
        n_heads=4,
        d_model=24,
        d_head=6,
        d_mlp=96,
        n_ctx=10,
        d_vocab=13,
        d_vocab_out=16,
        attention_dir="bidirectional",
        model_parameter_count=29_464,
        raw_ground_truth_edges=21,
        internal_ground_truth_nodes=(
            "blocks.0.attn.hook_result[0]",
            "blocks.3.hook_mlp_out",
            "blocks.2.hook_mlp_out",
            "blocks.0.hook_mlp_out",
            "blocks.1.hook_mlp_out",
        ),
        internal_ground_truth_edges=(
            (
                "blocks.0.attn.hook_result[0]",
                "blocks.0.hook_mlp_out",
            ),
            ("blocks.0.hook_mlp_out", "blocks.1.hook_mlp_out"),
            ("blocks.1.hook_mlp_out", "blocks.2.hook_mlp_out"),
            ("blocks.2.hook_mlp_out", "blocks.3.hook_mlp_out"),
        ),
        exact_internal_fine_ground_truth_edges=4,
        sha256={
            "edges.pkl": (
                "a79289804c5b5c05a361921e0a87b6c2e043b858ff57a78334865d1aba23ccaa"
            ),
            "ll_model_cfg.pkl": (
                "78a927568daff80c1fa61e40868a6c36b5d297604342ffecc99b57af12d6e00d"
            ),
            "ll_model.pth": (
                "5aa2c86e46ca4718f4043c735231bc6b3b3bd86ba56aed5861c523886e0ea874"
            ),
            "meta.json": (
                "47068a1d73770d5b1851105cf2f9198f90d7cc02288d1b342a5f0ad4078fcac2"
            ),
        },
    ),
}

# Homogeneous model-free geometry. ``full_fine_edge_count`` is the complete
# internal fine universe U; boundary fine edges are never selectable.
CASE_GEOMETRY: dict[int, dict[str, int]] = {
    58: {
        "nodes": 15,
        "whole_edges": 87,
        "full_fine_edge_count": 207,
        "total_fine_edge_count_with_boundary": 262,
        "raw_ground_truth_edges": 25,
        "internal_ground_truth_nodes": 4,
        "internal_ground_truth_whole_edges": 3,
        "exact_internal_fine_ground_truth_edges": 3,
    },
    63: {
        "nodes": 10,
        "whole_edges": 33,
        "full_fine_edge_count": 73,
        "total_fine_edge_count_with_boundary": 110,
        "raw_ground_truth_edges": 11,
        "internal_ground_truth_nodes": 2,
        "internal_ground_truth_whole_edges": 1,
        "exact_internal_fine_ground_truth_edges": 1,
    },
    103: {
        "nodes": 15,
        "whole_edges": 87,
        "full_fine_edge_count": 207,
        "total_fine_edge_count_with_boundary": 262,
        "raw_ground_truth_edges": 27,
        "internal_ground_truth_nodes": 4,
        "internal_ground_truth_whole_edges": 3,
        "exact_internal_fine_ground_truth_edges": 3,
    },
    82: {
        "nodes": 20,
        "whole_edges": 166,
        "full_fine_edge_count": 406,
        "total_fine_edge_count_with_boundary": 479,
        "raw_ground_truth_edges": 21,
        "internal_ground_truth_nodes": 5,
        "internal_ground_truth_whole_edges": 4,
        "exact_internal_fine_ground_truth_edges": 4,
    },
}


@dataclass(frozen=True)
class HistoricalRawRequest:
    """One previously exposed case-58 raw pool barred from E32.5."""

    source: str
    candidate_count: int
    clean_seed: int
    corrupt_seed: int


CASE58_HISTORICAL_RAW_REQUESTS = (
    HistoricalRawRequest("e32_3_smoke", 96, 758_000, 758_001),
    HistoricalRawRequest("e32_3_discovery", 96, 758_100, 758_101),
    HistoricalRawRequest("e32_3_confirmation", 192, 758_200, 758_201),
    HistoricalRawRequest("e32_4_discovery", 96, 758_300, 758_301),
    HistoricalRawRequest("e32_4_confirmation", 192, 758_400, 758_401),
)
CASE58_HISTORICAL_UNIQUE_ROLE_ROWS = 1_344
CASE58_DEVELOPMENT_UNIQUE_ROLE_ROWS = 192


@dataclass(frozen=True)
class SplitRequest:
    """One immutable raw-pool request and its input-only preflight facts."""

    case_id: int
    split: Split
    required_count: int
    candidate_count: int
    clean_seed: int
    corrupt_seed: int
    raw_identity_sha256: str
    target_identity_sha256: str
    expected_excluded_unique_rows: int
    response_blind_capacity: int
    behaviorally_selectable_capacity: int
    exclude_case58_historical_raw_roles: bool
    exclude_case58_development_raw_roles: bool
    exclude_same_case_discovery_raw_roles: bool
    claim_bearing: bool
    selection_policy: str = (
        "ascending_first_jointly_correct_with_globally_unique_clean_source_"
        "rows_after_frozen_response_blind_exclusions"
    )


_SPLIT_VALUES: Mapping[tuple[int, Split], SplitRequest] = {
    (58, "development"): SplitRequest(
        58,
        "development",
        64,
        96,
        5_005_810,
        5_005_811,
        "a29413383f99f28224146fd1337fb3c8e10259fd1e885f0f471a298b68697e47",
        "debbadf4e742de42a1c897e844ba91e5d60e7ea66bb70acc5f0b786eb90ec90f",
        1_344,
        96,
        96,
        True,
        False,
        False,
        False,
    ),
    (58, "repair_confirmation"): SplitRequest(
        58,
        "repair_confirmation",
        128,
        192,
        5_005_820,
        5_005_821,
        "c5c531009cf52f091bfd74fc381c9f9d534628d083d3adbffb83947e23ab7fbd",
        "403b8eb865ef69d17fbe58b4c8cf6941b0473587f7d605c7c9f251db2c2ef9ef",
        1_536,
        192,
        192,
        True,
        True,
        False,
        True,
    ),
    (63, "discovery"): SplitRequest(
        63,
        "discovery",
        64,
        96,
        5_006_310,
        5_006_311,
        "e6f45f6ee551cf924822c8c0909dab9f205cc15fd372b40e9355f26b887da1ba",
        "051e0c9c91f9e8cef4c73f35cf47b91fdcc928ebf546fc208f1c64e7a5fff3a3",
        192,
        96,
        96,
        False,
        True,
        False,
        True,
    ),
    (63, "confirmation"): SplitRequest(
        63,
        "confirmation",
        128,
        192,
        5_006_320,
        5_006_321,
        "119e4bdfc7dd7ebfb65bd7369f284c77094cec65e2af24516b9cade165e00212",
        "0c864bb160b804524dbaf405824c3221e1a1682ec81e9119948422cacae974be",
        384,
        192,
        190,
        False,
        True,
        True,
        True,
    ),
    (103, "discovery"): SplitRequest(
        103,
        "discovery",
        64,
        96,
        5_010_310,
        5_010_311,
        "fddd5fc408298d4e319c10cf5f78659ef22c38f604e864a4aac55ee8529585dc",
        "88baa4a9a503e82ad44f47d2b5d6acb0496d1fa883fc948aa176ea82658ea7a4",
        192,
        96,
        96,
        False,
        True,
        False,
        True,
    ),
    (103, "confirmation"): SplitRequest(
        103,
        "confirmation",
        128,
        192,
        5_010_320,
        5_010_321,
        "65524c714304c84b82effe041daa16e90b4a4c934a86f031d390e117d4b4a128",
        "e866fd95ad70e71d2937e1d9e8e604dfbd8e99f0159de29165c529abf905ac2b",
        384,
        192,
        192,
        False,
        True,
        True,
        True,
    ),
    (82, "discovery"): SplitRequest(
        82,
        "discovery",
        64,
        96,
        5_008_210,
        5_008_211,
        "c735230e2c04fa471ba7532f6dc50b094cbbb5139126da8fb196e8cd9cd6547f",
        "258c979fcc341c0b27a41f9680b8731ca150317120c6977079fb1ddd66749969",
        192,
        96,
        96,
        False,
        True,
        False,
        True,
    ),
    (82, "confirmation"): SplitRequest(
        82,
        "confirmation",
        128,
        192,
        5_008_220,
        5_008_221,
        "5c10b4a31625068cf9833427c65937022271dd3f5d65c007d2e2e6bf0a39da8b",
        "dbb5fa5c24b985f8764f4daa451a05b1c12439b6f97c57dfaf28f060b520a400",
        384,
        192,
        192,
        False,
        True,
        True,
        True,
    ),
}

# Behavior-only cohort design exposure. No patched forward, gradient, or
# backward call occurred. The five-alternative screen saw exact D/C eligibility
# before the cohort was frozen; rejected cases 45 and 111 are distinct models,
# so their rows are disclosed rather than spuriously excluded cross-model.
DESIGN_BEHAVIOR_EXPOSURE: Mapping[str, Any] = {
    "five_alternative_screen": {
        "case_ids": (45, 63, 82, 103, 111),
        "physical_forwards": 20,
        "model_examples": 2_880,
        "final_pool_eligibility_seen_pre_freeze": True,
        "rejected_case_ids": (45, 111),
        "rejected_cases_share_model_with_final_cases": False,
    },
    "case58_development_and_repair_input_screen": {
        "case_ids": (58,),
        "physical_forwards": 4,
        "model_examples": 576,
        "final_pool_eligibility_seen_pre_freeze": True,
    },
    "post_implementation_live_loader_validation": {
        "case_ids": CASE_ORDER,
        "physical_forwards": 16,
        "model_examples": 2_304,
        "purpose": "repeat_exact_eight_pool_behavior_preflight_no_interventions",
    },
    "total": {
        "physical_forwards": 40,
        "model_examples": 5_760,
        "gradient_enabled_forwards": 0,
        "gradient_examples": 0,
        "backward_passes": 0,
        "patched_intervention_model_examples": 0,
    },
}


@dataclass(frozen=True)
class ExactGroundTruth:
    case_id: int
    raw_edges: tuple[Edge, ...]
    internal_nodes: tuple[str, ...]
    internal_edges: tuple[Edge, ...]


@dataclass(frozen=True)
class InputPreflight:
    """Frozen raw panel after response-blind exclusion, before interventions."""

    request: SplitRequest
    pair: OfficialDataPair
    clean_unique_rows: int
    corrupt_unique_rows: int
    role_union_unique_rows: int
    excluded_unique_rows: int
    overlap_with_exclusions: int
    response_blind_selectable_indices: tuple[int, ...]
    observed_raw_identity_sha256: str
    observed_target_identity_sha256: str
    excluded_token_rows: tuple[TokenRow, ...] = ()


def _manifest(case_id: int) -> CohortCaseManifest:
    try:
        return CASE_MANIFESTS[case_id]
    except KeyError as error:
        raise ValueError(f"case_id must be one of {CASE_ORDER!r}") from error


def split_request(case_id: int, split: str) -> SplitRequest:
    """Return one exact request; adaptive growth, retry, and reseeding are barred."""
    _manifest(case_id)
    if split not in SPLIT_ORDER_BY_CASE[case_id]:
        raise ValueError(
            f"split for case {case_id} must be one of {SPLIT_ORDER_BY_CASE[case_id]!r}"
        )
    return _SPLIT_VALUES[(case_id, split)]  # type: ignore[index]


def case_artifact_paths(model_root: Path | str, case_id: int) -> dict[str, Path]:
    """Return exact artifact paths without mutating either model cache tree."""
    _manifest(case_id)
    case_dir = Path(model_root).expanduser().resolve() / str(case_id)
    return {name: case_dir / name for name in REQUIRED_ARTIFACTS}


def verify_case_artifacts(model_root: Path | str, case_id: int) -> dict[str, str]:
    """Require all four official files to match the pinned HF revision."""
    manifest = _manifest(case_id)
    observed: dict[str, str] = {}
    for name, path in case_artifact_paths(model_root, case_id).items():
        if not path.is_file():
            raise FileNotFoundError(f"missing case-{case_id} artifact: {path}")
        digest = sha256_file(path)
        if digest != manifest.sha256[name]:
            raise ValueError(
                f"SHA-256 mismatch for case {case_id} {name}: expected "
                f"{manifest.sha256[name]}, observed {digest}"
            )
        observed[name] = digest
    return observed


def case_universe(case_id: int) -> tuple[tuple[str, ...], tuple[Edge, ...]]:
    """Return the exact full internal whole-component DAG."""
    manifest = _manifest(case_id)
    nodes, edges = patchable_internal_universe(manifest.n_layers, manifest.n_heads)
    geometry = CASE_GEOMETRY[case_id]
    if (len(nodes), len(edges)) != (geometry["nodes"], geometry["whole_edges"]):
        raise RuntimeError("case universe differs from frozen geometry")
    return nodes, edges


def load_exact_ground_truth(model_root: Path | str, case_id: int) -> ExactGroundTruth:
    """Load public GT for descriptive use and verify its exact promoted support."""
    manifest = _manifest(case_id)
    path = case_artifact_paths(model_root, case_id)["edges.pkl"]
    raw = load_raw_ground_truth(path, expected_sha256=manifest.sha256["edges.pkl"])
    nodes = promoted_ground_truth_internal_nodes(raw)
    edges = internal_ground_truth_edges(raw)
    if (
        len(raw) != manifest.raw_ground_truth_edges
        or nodes != manifest.internal_ground_truth_nodes
        or edges != manifest.internal_ground_truth_edges
    ):
        raise RuntimeError("official ground truth differs from the frozen manifest")
    return ExactGroundTruth(case_id, raw, nodes, edges)


def _validate_model_config(case_id: int, config: Any) -> None:
    manifest = _manifest(case_id)
    fields = (
        "n_layers",
        "n_heads",
        "d_model",
        "d_head",
        "d_mlp",
        "n_ctx",
        "d_vocab",
        "d_vocab_out",
    )
    if any(
        int(getattr(config, field, -1)) != getattr(manifest, field) for field in fields
    ):
        raise ValueError("checkpoint config differs from the frozen case manifest")
    if str(getattr(config, "attention_dir", "")) != manifest.attention_dir:
        raise ValueError("checkpoint attention direction changed")


def load_verified_model(
    case_id: int,
    *,
    model_root: Path | str = DEFAULT_E32_5_MODEL_ROOT,
    device: str = "cpu",
) -> Any:
    """Load a checkpoint only after all artifact bytes have been verified."""
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
        raise TypeError("ll_model_cfg.pkl has an unsupported payload type")
    _validate_model_config(case_id, config)
    config.device = device
    model = HookedTransformer(config)
    # Pickle-backed torch bytes are opened only after exact SHA-256 verification.
    state_dict = torch.load(paths["ll_model.pth"], map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    observed_parameters = sum(parameter.numel() for parameter in model.parameters())
    if observed_parameters != _manifest(case_id).model_parameter_count:
        raise RuntimeError("loaded model parameter count differs from the manifest")
    return model


def load_verified_official_case(
    case_id: int,
    *,
    official_code_dir: Path | str = DEFAULT_OFFICIAL_CODE_DIR,
) -> Any:
    """Instantiate one case from the exact pinned InterpBench checkout."""
    _manifest(case_id)
    code_dir = Path(official_code_dir).expanduser().resolve()
    verify_git_revision(code_dir, INTERPBENCH_CODE_REVISION)
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    try:
        importlib.import_module("pygraphviz")
    except ModuleNotFoundError:
        sys.modules["pygraphviz"] = types.ModuleType("pygraphviz")
    module = importlib.import_module(
        f"circuits_benchmark.benchmark.cases.case_{case_id}"
    )
    module_path = Path(module.__file__).resolve()
    if code_dir not in module_path.parents:
        raise RuntimeError(f"case module came from unpinned path {module_path}")
    case = getattr(module, f"Case{case_id}")()
    if not bool(case.is_categorical()) or bool(case.supports_causal_masking()):
        raise RuntimeError("case output or attention type differs from the manifest")
    return case


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _pair_tensors(pair: OfficialDataPair) -> tuple[torch.Tensor, ...]:
    if not isinstance(pair, OfficialDataPair):
        raise TypeError("pair must be adapter.OfficialDataPair")
    values = (
        pair.clean_inputs,
        pair.clean_targets,
        pair.corrupt_inputs,
        pair.corrupt_targets,
    )
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError("official pair fields must be tensors")
    clean_inputs, clean_targets, corrupt_inputs, corrupt_targets = values
    if (
        clean_inputs.ndim != 2
        or corrupt_inputs.shape != clean_inputs.shape
        or clean_targets.ndim != 3
        or corrupt_targets.ndim != 3
        or clean_targets.shape[:2] != clean_inputs.shape
        or corrupt_targets.shape[:2] != corrupt_inputs.shape
    ):
        raise ValueError("official pair geometry is invalid")
    if clean_inputs.is_floating_point() or corrupt_inputs.is_floating_point():
        raise ValueError("official inputs must contain integer token IDs")
    if not clean_targets.is_floating_point() or not corrupt_targets.is_floating_point():
        raise ValueError("categorical targets must be floating tensors")
    if not bool(torch.isfinite(clean_targets).all()) or not bool(
        torch.isfinite(corrupt_targets).all()
    ):
        raise ValueError("categorical targets must be finite")
    return values


def raw_identity_payload(
    request: SplitRequest, pair: OfficialDataPair
) -> dict[str, Any]:
    """Return the canonical ordered raw-role identity object."""
    clean_inputs, _clean_targets, corrupt_inputs, _corrupt_targets = _pair_tensors(pair)
    return {
        "schema_version": "e32_5_raw_candidate_pool_v1",
        "case_id": request.case_id,
        "split": request.split,
        "candidate_count": request.candidate_count,
        "clean_seed": request.clean_seed,
        "corrupt_seed": request.corrupt_seed,
        "clean_input_rows": token_rows(clean_inputs, label="clean raw inputs"),
        "corrupt_input_rows": token_rows(corrupt_inputs, label="corrupt raw inputs"),
    }


def target_identity_payload(
    request: SplitRequest, pair: OfficialDataPair
) -> dict[str, Any]:
    """Return the canonical ordered target identity object."""
    _clean_inputs, clean_targets, _corrupt_inputs, corrupt_targets = _pair_tensors(pair)
    return {
        "schema_version": "e32_5_target_identity_v1",
        "case_id": request.case_id,
        "split": request.split,
        "clean_target_argmax_rows": tuple(
            tuple(int(value) for value in row)
            for row in clean_targets.argmax(dim=-1).tolist()
        ),
        "corrupt_target_argmax_rows": tuple(
            tuple(int(value) for value in row)
            for row in corrupt_targets.argmax(dim=-1).tolist()
        ),
    }


def raw_role_rows(pair: OfficialDataPair) -> tuple[TokenRow, ...]:
    clean_inputs, _clean_targets, corrupt_inputs, _corrupt_targets = _pair_tensors(pair)
    return (
        *token_rows(clean_inputs, label="clean raw inputs"),
        *token_rows(corrupt_inputs, label="corrupt raw inputs"),
    )


def _normalized_exclusions(
    excluded_token_rows: Collection[Sequence[int]],
) -> tuple[TokenRow, ...]:
    rows: list[TokenRow] = []
    seen: set[TokenRow] = set()
    for raw in excluded_token_rows:
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise TypeError("excluded token rows must be integer sequences")
        row = tuple(raw)
        if not row or any(
            isinstance(value, bool) or not isinstance(value, int) for value in row
        ):
            raise ValueError("excluded token rows must be nonempty integer rows")
        if row not in seen:
            seen.add(row)
            rows.append(row)
    return tuple(rows)


def response_blind_selectable_indices(
    pair: OfficialDataPair,
    *,
    excluded_token_rows: Collection[Sequence[int]] = (),
) -> tuple[int, ...]:
    """Greedily freeze the maximum protocol-ordered row-disjoint capacity."""
    clean_inputs, _clean_targets, corrupt_inputs, _corrupt_targets = _pair_tensors(pair)
    clean = token_rows(clean_inputs, label="clean raw inputs")
    corrupt = token_rows(corrupt_inputs, label="corrupt raw inputs")
    used = set(_normalized_exclusions(excluded_token_rows))
    selected: list[int] = []
    for index, (clean_row, corrupt_row) in enumerate(zip(clean, corrupt, strict=True)):
        if clean_row == corrupt_row or clean_row in used or corrupt_row in used:
            continue
        selected.append(index)
        used.update((clean_row, corrupt_row))
    return tuple(selected)


def generate_split_pair(case: Any, request: SplitRequest) -> OfficialDataPair:
    """Generate one exact official pair and require its candidate count."""
    pair = generate_official_pair(
        case,
        sample_count=request.candidate_count,
        clean_seed=request.clean_seed,
        corrupt_seed=request.corrupt_seed,
    )
    if len(pair.clean_inputs) != request.candidate_count:
        raise RuntimeError("official generator returned the wrong candidate count")
    return pair


def preflight_split_inputs(
    case: Any,
    request: SplitRequest,
    *,
    excluded_token_rows: Collection[Sequence[int]],
) -> InputPreflight:
    """Regenerate and authenticate one response-blind panel without model calls."""
    pair = generate_split_pair(case, request)
    exclusions = _normalized_exclusions(excluded_token_rows)
    observed_raw = _json_sha256(raw_identity_payload(request, pair))
    observed_targets = _json_sha256(target_identity_payload(request, pair))
    if observed_raw != request.raw_identity_sha256:
        raise RuntimeError("raw candidate-pool identity differs from the preflight")
    if observed_targets != request.target_identity_sha256:
        raise RuntimeError("target identity differs from the preflight")

    clean_rows = token_rows(pair.clean_inputs, label="clean raw inputs")
    corrupt_rows = token_rows(pair.corrupt_inputs, label="corrupt raw inputs")
    clean_unique = len(set(clean_rows))
    corrupt_unique = len(set(corrupt_rows))
    union = set(clean_rows) | set(corrupt_rows)
    overlap = len(union & set(exclusions))
    selectable = response_blind_selectable_indices(pair, excluded_token_rows=exclusions)
    if len(exclusions) != request.expected_excluded_unique_rows:
        raise RuntimeError("exclusion-row count differs from the frozen request")
    if (
        clean_unique != request.candidate_count
        or corrupt_unique != request.candidate_count
        or len(union) != 2 * request.candidate_count
        or overlap != 0
        or len(selectable) != request.response_blind_capacity
    ):
        raise RuntimeError("response-blind row capacity differs from the preflight")
    return InputPreflight(
        request=request,
        pair=pair,
        clean_unique_rows=clean_unique,
        corrupt_unique_rows=corrupt_unique,
        role_union_unique_rows=len(union),
        excluded_unique_rows=len(exclusions),
        overlap_with_exclusions=overlap,
        response_blind_selectable_indices=selectable,
        observed_raw_identity_sha256=observed_raw,
        observed_target_identity_sha256=observed_targets,
        excluded_token_rows=exclusions,
    )


def _stable_unique_rows(rows: Sequence[TokenRow]) -> tuple[TokenRow, ...]:
    return tuple(dict.fromkeys(rows))


def case58_historical_raw_rows(case: Any) -> tuple[TokenRow, ...]:
    """Regenerate every E32.3/E32.4 case-58 raw role row, model-free."""
    rows: list[TokenRow] = []
    for historical in CASE58_HISTORICAL_RAW_REQUESTS:
        pair = generate_official_pair(
            case,
            sample_count=historical.candidate_count,
            clean_seed=historical.clean_seed,
            corrupt_seed=historical.corrupt_seed,
        )
        rows.extend(raw_role_rows(pair))
    unique = _stable_unique_rows(rows)
    if len(unique) != CASE58_HISTORICAL_UNIQUE_ROLE_ROWS:
        raise RuntimeError("historical case-58 row union differs from custody")
    return unique


def preflight_case_inputs(
    case_id: int,
    case: Any,
    *,
    case58_development_raw_rows: Collection[Sequence[int]] = (),
) -> dict[Split, InputPreflight]:
    """Build both split preflights with all cross-stage exclusions enforced."""
    _manifest(case_id)
    if case_id == 58:
        if case58_development_raw_rows:
            raise ValueError("case58 development rows are outputs, not inputs")
        historical = case58_historical_raw_rows(case)
        development_request = split_request(58, "development")
        development = preflight_split_inputs(
            case,
            development_request,
            excluded_token_rows=historical,
        )
        development_rows = raw_role_rows(development.pair)
        if len(set(development_rows)) != CASE58_DEVELOPMENT_UNIQUE_ROLE_ROWS:
            raise RuntimeError("case58 development row count differs from custody")
        repair = preflight_split_inputs(
            case,
            split_request(58, "repair_confirmation"),
            excluded_token_rows=(*historical, *development_rows),
        )
        return {"development": development, "repair_confirmation": repair}

    development_rows = _normalized_exclusions(case58_development_raw_rows)
    if len(development_rows) != CASE58_DEVELOPMENT_UNIQUE_ROLE_ROWS:
        raise ValueError("fresh cases require all 192 case58 development role rows")
    discovery = preflight_split_inputs(
        case,
        split_request(case_id, "discovery"),
        excluded_token_rows=development_rows,
    )
    confirmation = preflight_split_inputs(
        case,
        split_request(case_id, "confirmation"),
        excluded_token_rows=(*development_rows, *raw_role_rows(discovery.pair)),
    )
    return {"discovery": discovery, "confirmation": confirmation}


def preflight_cohort_inputs(
    cases: Mapping[int, Any],
) -> dict[int, dict[Split, InputPreflight]]:
    """Preflight case58 first, then bind its development rows into every case."""
    if tuple(cases) != CASE_ORDER:
        raise ValueError(f"cases must have exact order {CASE_ORDER!r}")
    case58 = preflight_case_inputs(58, cases[58])
    development_rows = raw_role_rows(case58["development"].pair)
    panels: dict[int, dict[Split, InputPreflight]] = {58: case58}
    for case_id in GENERALIZATION_CASE_ORDER:
        panels[case_id] = preflight_case_inputs(
            case_id,
            cases[case_id],
            case58_development_raw_rows=development_rows,
        )
    return panels


def behaviorally_select_preflight(
    model: Any,
    preflight: InputPreflight,
    *,
    device: str = "cpu",
) -> BehaviorallyEligiblePair:
    """Run exactly two behavior-only forwards and freeze the first valid rows."""
    if not isinstance(preflight, InputPreflight):
        raise TypeError("preflight must be InputPreflight")
    request = preflight.request
    pair = preflight.pair
    model.to(device)
    model.eval()
    with torch.inference_mode():
        clean_logits = model(pair.clean_inputs.to(device)).detach().cpu()
        corrupt_logits = model(pair.corrupt_inputs.to(device)).detach().cpu()
    clean_correct = _correct_context_mask(clean_logits, pair.clean_targets)
    corrupt_correct = _correct_context_mask(corrupt_logits, pair.corrupt_targets)
    jointly_correct = clean_correct & corrupt_correct

    clean_rows = token_rows(pair.clean_inputs, label="clean raw inputs")
    corrupt_rows = token_rows(pair.corrupt_inputs, label="corrupt raw inputs")
    used = set(preflight.excluded_token_rows)
    selectable: list[int] = []
    row_overlap_excluded = 0
    for index, (clean_row, corrupt_row) in enumerate(
        zip(clean_rows, corrupt_rows, strict=True)
    ):
        if not bool(jointly_correct[index]):
            continue
        if clean_row == corrupt_row or clean_row in used or corrupt_row in used:
            row_overlap_excluded += 1
            continue
        selectable.append(index)
        used.update((clean_row, corrupt_row))
    if len(selectable) != request.behaviorally_selectable_capacity:
        raise RuntimeError("behaviorally selectable capacity differs from preflight")
    if len(selectable) < request.required_count:
        raise RuntimeError("behavioral screen cannot fill the frozen panel")
    selected = tuple(selectable[: request.required_count])
    tensor_indices = torch.tensor(selected, dtype=torch.long)

    def take(value: torch.Tensor) -> torch.Tensor:
        return value.index_select(0, tensor_indices).detach().cpu().clone()

    return BehaviorallyEligiblePair(
        pair=OfficialDataPair(
            clean_inputs=take(pair.clean_inputs),
            clean_targets=take(pair.clean_targets),
            corrupt_inputs=take(pair.corrupt_inputs),
            corrupt_targets=take(pair.corrupt_targets),
        ),
        candidate_count=request.candidate_count,
        eligible_candidate_count=int(jointly_correct.sum()),
        clean_incorrect_contexts=int((~clean_correct).sum()),
        source_incorrect_contexts=int((~corrupt_correct).sum()),
        either_incorrect_contexts=int((~jointly_correct).sum()),
        selected_candidate_indices=selected,
        eligibility_physical_forwards=2,
        eligibility_model_examples=2 * request.candidate_count,
        output_kind="categorical",
        row_disjoint_eligible_candidate_count=len(selectable),
        row_overlap_excluded_contexts=row_overlap_excluded,
        excluded_token_row_count=len(preflight.excluded_token_rows),
        selection_policy=request.selection_policy,
    )


__all__ = [
    "CASE58_DEVELOPMENT_UNIQUE_ROLE_ROWS",
    "CASE58_HISTORICAL_RAW_REQUESTS",
    "CASE58_HISTORICAL_UNIQUE_ROLE_ROWS",
    "CASE_GEOMETRY",
    "CASE_MANIFESTS",
    "CASE_ORDER",
    "DEFAULT_E32_5_MODEL_ROOT",
    "DESIGN_BEHAVIOR_EXPOSURE",
    "GENERALIZATION_CASE_ORDER",
    "SPLIT_ORDER_BY_CASE",
    "CohortCaseManifest",
    "ExactGroundTruth",
    "HistoricalRawRequest",
    "InputPreflight",
    "SplitRequest",
    "behaviorally_select_preflight",
    "case58_historical_raw_rows",
    "case_artifact_paths",
    "case_universe",
    "generate_split_pair",
    "load_exact_ground_truth",
    "load_verified_model",
    "load_verified_official_case",
    "preflight_case_inputs",
    "preflight_cohort_inputs",
    "preflight_split_inputs",
    "raw_identity_payload",
    "raw_role_rows",
    "response_blind_selectable_indices",
    "split_request",
    "target_identity_payload",
    "verify_case_artifacts",
]
