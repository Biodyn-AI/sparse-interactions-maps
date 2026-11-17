# ruff: noqa: EXE002
"""Tests for the unsealed, model-free E32.1 experiment orchestration."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest
import torch

from experiments.e32_interpbench.adapter import (
    INTERPBENCH_CODE_REVISION,
    INTERPBENCH_MODEL_REVISION,
    patchable_internal_universe,
)
from experiments.e32_interpbench.component_patching import (
    build_component_atlas,
    build_patch_units,
)
from experiments.e32_interpbench.e32_1_orchestration import (
    AcquiredSplit,
    ExactGroundTruth,
    assert_discovery_confirmation_disjoint,
    evaluate_case_pair,
    evaluate_primary_gate,
    run_hierarchical_panel,
)
from experiments.e32_interpbench.e32_1_protocol import (
    CASE_ORDER,
    METHOD_ORDER,
    PRIMARY_CASE_ID,
    expected_case_provenance,
    protocol_dry_run_manifest,
    split_request,
)

CASE_21_GT_NODES = (
    "blocks.3.attn.hook_result[0]",
    "blocks.2.hook_mlp_out",
    "blocks.1.hook_mlp_out",
    "blocks.0.hook_mlp_out",
)
CASE_21_GT_EDGES = (
    ("blocks.2.hook_mlp_out", "blocks.3.attn.hook_result[0]"),
    ("blocks.1.hook_mlp_out", "blocks.2.hook_mlp_out"),
)


def _fake_atlas(case_id: int, split: str, *, token_offset: int) -> dict[str, object]:
    request = split_request(case_id, split)
    manifest = protocol_dry_run_manifest()["cases"][str(case_id)]
    nodes, edges = patchable_internal_universe(
        int(manifest["n_layers"]), int(manifest["n_heads"])
    )
    example_count = request.required_count
    condition_count = 1 + len(nodes) + len(edges)
    positions = int(manifest["n_ctx"]) - 1
    classes = int(manifest["d_vocab_out"])
    clean_inputs = (
        torch.arange(example_count * 10, dtype=torch.int64).reshape(example_count, 10)
        + token_offset
    )
    corrupt_inputs = clean_inputs + 10_000
    singleton = torch.zeros(len(nodes), example_count, positions, classes)
    amplitudes = torch.linspace(0.1, 2.0, len(edges))
    class_direction = torch.arange(classes, dtype=torch.float32)
    class_direction -= class_direction.mean()
    pair_interactions = (
        amplitudes[:, None, None, None] * class_direction[None, None, None, :]
    ).expand(-1, example_count, positions, -1)
    atlas_manifest = build_component_atlas(
        build_patch_units(int(manifest["n_layers"]), int(manifest["n_heads"])),
        valid_directed_edges=edges,
    )
    condition_outputs = torch.zeros(condition_count, example_count, positions, classes)
    condition_outputs[list(atlas_manifest.pair_condition_indices)] = pair_interactions
    raw_condition_outputs = torch.zeros(
        condition_count, example_count, positions + 1, classes
    )
    raw_condition_outputs[..., 1:, :] = condition_outputs
    smoke_conditions = 16
    smoke_examples = 2 * smoke_conditions * example_count
    patched_conditions = condition_count - 1
    patched_examples = patched_conditions * example_count
    live_examples = 2 * example_count + smoke_examples + patched_examples
    # Smoke: 16 one-condition calls plus two batches of eight.  Atlas: chunks of 32.
    smoke_forwards = 18
    atlas_forwards = (patched_conditions + 31) // 32
    live_forwards = 2 + smoke_forwards + atlas_forwards
    selected = tuple(range(example_count))
    return {
        "schema_version": "e32_interpbench_component_atlas_v2",
        "case_id": case_id,
        "case_role": "primary" if case_id == 97 else "secondary",
        "patch_scope": "whole_head_hook_result_and_mlp_hook_mlp_out_all_positions",
        "output_scope": "full_per_example_per_position_output_tensor",
        "component_ids": nodes,
        "condition_ids": atlas_manifest.condition_ids,
        "condition_masks": atlas_manifest.condition_masks,
        "pair_component_indices": atlas_manifest.pair_component_indices,
        "directed_candidate_edges": edges,
        "pair_patch_semantics": (
            "symmetric_two_component_mask; edge_arrow_is_topology_imposed"
        ),
        "clean_inputs": clean_inputs,
        "corrupt_inputs": corrupt_inputs,
        "condition_outputs": raw_condition_outputs,
        "scientific_condition_outputs": condition_outputs,
        "scientific_singleton_outputs": singleton,
        "scientific_pair_outputs": pair_interactions,
        "pair_interactions": pair_interactions,
        "scientific_output_coordinate": "class_centered_logits_non_bos_positions",
        "behavioral_pair_screen": {
            "all_pairs_correct": True,
            "context_count": example_count,
        },
        "behavioral_eligibility": {
            "candidate_count": request.candidate_count,
            "eligible_candidate_count": request.candidate_count,
            "row_disjoint_eligible_candidate_count": request.candidate_count,
            "row_overlap_excluded_contexts": 0,
            "excluded_token_row_count": 0 if split == "discovery" else 128,
            "selected_count": example_count,
            "selected_candidate_indices": selected,
            "output_kind": "categorical",
            "selection_policy": (
                "first_jointly_correct_with_globally_unique_clean_source_rows_and_"
                "response_blind_exclusions"
            ),
        },
        "eligibility_query_accounting": {
            "physical_forwards_before_runner": 2,
            "model_examples_before_runner": 2 * request.candidate_count,
        },
        "batch_invariance_smoke": {
            "passed": True,
            "condition_count": smoke_conditions,
            "example_count_per_condition": example_count,
            "model_examples": smoke_examples,
            "physical_forwards": smoke_forwards,
        },
        "atlas_query_accounting": {
            "empty_reuses_prepared_clean_output": True,
            "patched_condition_count": patched_conditions,
            "example_count_per_condition": example_count,
            "patched_model_examples": patched_examples,
            "patched_physical_forwards": atlas_forwards,
        },
        "total_live_query_accounting": {
            "physical_forwards": live_forwards,
            "model_examples": live_examples,
            "excludes_pre_runner_eligibility": True,
        },
        "total_query_accounting_including_eligibility": {
            "physical_forwards": live_forwards + 2,
            "model_examples": live_examples + 2 * request.candidate_count,
        },
    }


def _fake_split(case_id: int, split: str, *, token_offset: int) -> AcquiredSplit:
    request = split_request(case_id, split)
    return AcquiredSplit(
        request=request,
        atlas=_fake_atlas(case_id, split, token_offset=token_offset),
        exact_provenance=expected_case_provenance(case_id),
    )


def _primary_metrics(
    *, raw_ap: float = 0.81, raw_tp: int = 5, stability: float = 0.70
) -> dict[str, object]:
    comparator_aps = (0.70, 0.73, 0.60, 0.65)
    comparator_tps = (3, 4, 2, 3)
    methods: dict[str, object] = {}
    for method, ap, tp in zip(
        METHOD_ORDER[1:], comparator_aps, comparator_tps, strict=True
    ):
        methods[method] = {
            "confirmation": {
                "edge": {
                    "auprc": ap,
                    "top_k": 6,
                    "selected_support": tuple(
                        (str(index), method) for index in range(tp)
                    ),
                }
            },
            "confirmation_true_positives_at_gt_k": tp,
        }
    methods[METHOD_ORDER[0]] = {
        "confirmation": {
            "edge": {
                "auprc": raw_ap,
                "top_k": 6,
                "selected_support": tuple(
                    (str(index), "raw") for index in range(raw_tp)
                ),
            }
        },
        "confirmation_true_positives_at_gt_k": raw_tp,
    }
    return {
        "case_id": PRIMARY_CASE_ID,
        "ground_truth_edge_count": 6,
        "methods": methods,
        "discovery_confirmation_spearman": {
            "raw_pair_interaction": stability,
        },
    }


def test_protocol_freezes_hierarchy_splits_methods_hashes_and_query_ceiling() -> None:
    manifest = protocol_dry_run_manifest()

    assert CASE_ORDER == (97, 2, 21)
    assert METHOD_ORDER == (
        "raw_pair_interaction",
        "singleton_endpoint_product",
        "rank_one_node_factor",
        "positive_pair_residual",
        "absolute_pair_residual",
    )
    assert split_request(97, "discovery").required_count == 64
    assert split_request(97, "discovery").candidate_count == 80
    assert split_request(97, "confirmation").required_count == 128
    assert split_request(97, "confirmation").candidate_count == 160
    assert manifest["hierarchy"]["secondary_cases_run_only_if_primary_passes"] is True
    assert manifest["maximum_model_examples_if_primary_fails"] == 26_592
    assert manifest["maximum_model_examples_if_all_cases_run"] == 112_032
    assert manifest["claims"]["path_or_faithfulness_claims_in_e32_1"] is False
    provenance = expected_case_provenance(97)
    assert provenance["source_revisions"]["interpbench"] == INTERPBENCH_CODE_REVISION
    assert provenance["model_revision"] == INTERPBENCH_MODEL_REVISION
    assert len(provenance["artifact_sha256"]) == 4


def test_selected_token_rows_must_be_disjoint_across_discovery_and_confirmation() -> (
    None
):
    discovery = _fake_split(21, "discovery", token_offset=1_000)
    confirmation = _fake_split(21, "confirmation", token_offset=2_000)

    audit = assert_discovery_confirmation_disjoint(discovery, confirmation)

    assert audit["overlap_count"] == 0
    assert audit["discovery_unique_token_rows"] == 128
    assert audit["confirmation_unique_token_rows"] == 256
    overlapping = dict(confirmation.atlas)
    overlapping["clean_inputs"] = discovery.atlas["clean_inputs"].repeat(2, 1)
    with pytest.raises(RuntimeError, match="token-row leakage"):
        assert_discovery_confirmation_disjoint(
            discovery,
            AcquiredSplit(
                request=confirmation.request,
                atlas=overlapping,
                exact_provenance=confirmation.exact_provenance,
            ),
        )


def test_case_evaluation_keeps_node_truth_transitive_diagnostics_and_query_counts() -> (
    None
):
    discovery = _fake_split(21, "discovery", token_offset=1_000)
    confirmation = _fake_split(21, "confirmation", token_offset=2_000)
    ground_truth = ExactGroundTruth(
        case_id=21,
        node_ids=CASE_21_GT_NODES,
        directed_edges=CASE_21_GT_EDGES,
        raw_edges_sha256=expected_case_provenance(21)["artifact_sha256"]["edges.pkl"],
    )

    result = evaluate_case_pair(discovery, confirmation, ground_truth)

    assert tuple(result["methods"]) == METHOD_ORDER
    assert result["ground_truth_node_count"] == 4
    assert result["ground_truth_edge_count"] == 2
    assert (
        result["methods"]["raw_pair_interaction"]["confirmation"][
            "ground_truth_node_count"
        ]
        == 4
    )
    assert result["transitive_candidate_non_edges"]["count"] == 1
    assert result["query_accounting"]["combined_model_examples"] == 42_720
    assert result["discovery_confirmation_disjointness"]["overlap_count"] == 0
    assert "path" not in json.dumps(result)
    assert "faithfulness" not in json.dumps(result)


def test_case_evaluation_rejects_universe_gt_identity_and_accounting_tamper() -> None:
    discovery = _fake_split(21, "discovery", token_offset=1_000)
    confirmation = _fake_split(21, "confirmation", token_offset=2_000)
    ground_truth = ExactGroundTruth(
        case_id=21,
        node_ids=CASE_21_GT_NODES,
        directed_edges=CASE_21_GT_EDGES,
        raw_edges_sha256=expected_case_provenance(21)["artifact_sha256"]["edges.pkl"],
    )

    wrong_universe = dict(discovery.atlas)
    wrong_universe["component_ids"] = tuple(reversed(wrong_universe["component_ids"]))
    with pytest.raises(ValueError, match="universe or order"):
        evaluate_case_pair(
            AcquiredSplit(
                discovery.request, wrong_universe, discovery.exact_provenance
            ),
            confirmation,
            ground_truth,
        )

    wrong_nodes = ("blocks.0.attn.hook_result[1]", *CASE_21_GT_NODES[1:])
    with pytest.raises(ValueError, match="support identities"):
        evaluate_case_pair(
            discovery,
            confirmation,
            ExactGroundTruth(
                case_id=21,
                node_ids=wrong_nodes,
                directed_edges=CASE_21_GT_EDGES,
                raw_edges_sha256=ground_truth.raw_edges_sha256,
            ),
        )

    wrong_accounting = dict(discovery.atlas)
    wrong_accounting["total_query_accounting_including_eligibility"] = {
        **wrong_accounting["total_query_accounting_including_eligibility"],
        "physical_forwards": 999,
    }
    with pytest.raises(ValueError, match="query accounting"):
        evaluate_case_pair(
            AcquiredSplit(
                discovery.request, wrong_accounting, discovery.exact_provenance
            ),
            confirmation,
            ground_truth,
        )


def test_primary_gate_requires_all_four_ap_margins_tp_advantage_and_stability() -> None:
    passed = evaluate_primary_gate(_primary_metrics())

    assert passed["passed"] is True
    assert passed["checks"] == {
        "confirmation_auprc_margin_over_every_comparator": True,
        "confirmation_true_positive_advantage_at_6": True,
        "discovery_confirmation_raw_energy_spearman": True,
    }
    assert evaluate_primary_gate(_primary_metrics(raw_ap=0.77))["passed"] is False
    assert evaluate_primary_gate(_primary_metrics(raw_tp=4))["passed"] is False
    assert evaluate_primary_gate(_primary_metrics(stability=0.49))["passed"] is False


def test_hierarchy_does_not_acquire_secondary_cases_when_primary_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.e32_interpbench import e32_1_orchestration as module

    calls: list[tuple[int, str]] = []

    def acquire(request):
        calls.append((request.case_id, request.split))
        return AcquiredSplit(request, {}, expected_case_provenance(request.case_id))

    def ground_truth(case_id: int):
        return ExactGroundTruth(case_id, ("a", "b"), (("a", "b"),), "0" * 64)

    monkeypatch.setattr(
        module,
        "evaluate_case_pair",
        lambda *_args: _primary_metrics(raw_ap=0.70),
    )

    result = run_hierarchical_panel(acquire, ground_truth)

    assert calls == [(97, "discovery"), (97, "confirmation")]
    assert result["primary_gate"]["passed"] is False
    assert result["secondary_status"] == "not_run_primary_gate_failed"


def test_hierarchy_acquires_both_secondary_cases_after_primary_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.e32_interpbench import e32_1_orchestration as module

    calls: list[tuple[int, str]] = []

    def acquire(request):
        calls.append((request.case_id, request.split))
        return AcquiredSplit(request, {}, expected_case_provenance(request.case_id))

    def ground_truth(case_id: int):
        return ExactGroundTruth(case_id, ("a", "b"), (("a", "b"),), "0" * 64)

    def evaluate(discovery, _confirmation, _ground_truth):
        if discovery.request.case_id == 97:
            return _primary_metrics()
        return {"case_id": discovery.request.case_id, "status": "fake_complete"}

    monkeypatch.setattr(module, "evaluate_case_pair", evaluate)

    result = run_hierarchical_panel(acquire, ground_truth)

    assert calls == [
        (97, "discovery"),
        (97, "confirmation"),
        (2, "discovery"),
        (2, "confirmation"),
        (21, "discovery"),
        (21, "confirmation"),
    ]
    assert result["primary_gate"]["passed"] is True
    assert tuple(result["case_results"]) == ("97", "2", "21")
    assert result["secondary_status"] == "completed_after_primary_gate_passed"


def test_dry_run_cli_emits_manifest_without_loading_or_writing_models(capsys) -> None:
    from experiments.e32_interpbench.e32_1_dry_run import main

    assert main(["--dry-run"]) == 0
    output: Mapping[str, object] = json.loads(capsys.readouterr().out)
    assert output["status"] == "dry_run_only_no_model_calls_no_seal"
    assert output["case_order"] == [97, 2, 21]
