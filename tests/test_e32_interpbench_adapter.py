# ruff: noqa: EXE002
"""Regression tests for the small, dependency-light InterpBench adapter."""

from __future__ import annotations

import hashlib
import pickle

import pytest
import torch

from experiments.e32_interpbench.adapter import (
    CASE_MANIFESTS,
    PANEL_CASE_IDS,
    behaviorally_eligible_pair,
    canonicalize_official_edges,
    categorical_accuracy,
    generate_official_pair,
    internal_ground_truth_edges,
    load_raw_ground_truth,
    official_prepared_universe,
    patchable_internal_universe,
    promoted_ground_truth_internal_nodes,
    promoted_ground_truth_nodes,
    sha256_file,
)


def test_panel_is_pinned_and_case_113_is_explicitly_rejected() -> None:
    assert PANEL_CASE_IDS == (2, 21, 97)
    assert all(len(CASE_MANIFESTS[case_id].sha256) == 4 for case_id in PANEL_CASE_IDS)
    assert CASE_MANIFESTS[97].role == "primary"
    assert all(
        CASE_MANIFESTS[case_id].output_kind == "categorical"
        for case_id in PANEL_CASE_IDS
    )
    assert CASE_MANIFESTS[113].role == "rejected_behavioral_failure"


def test_official_edge_canonicalization_drops_and_reroutes_qkv_and_mlp() -> None:
    raw_edges = (
        ("hook_embed", "blocks.0.hook_q_input[2]"),
        ("hook_pos_embed", "blocks.0.hook_q_input[2]"),
        ("blocks.0.hook_q_input[2]", "blocks.0.attn.hook_q[2]"),
        ("blocks.0.attn.hook_q[2]", "blocks.0.attn.hook_result[2]"),
        ("blocks.0.attn.hook_result[2]", "blocks.0.hook_mlp_in"),
        ("blocks.0.hook_mlp_in", "blocks.0.hook_mlp_out"),
        ("blocks.0.hook_mlp_out", "blocks.0.hook_resid_post"),
        ("hook_embed", "blocks.0.hook_resid_post"),
    )

    assert canonicalize_official_edges(raw_edges) == (
        ("blocks.0.hook_resid_pre", "blocks.0.attn.hook_result[2]"),
        ("blocks.0.attn.hook_result[2]", "blocks.0.hook_mlp_out"),
        ("blocks.0.hook_mlp_out", "blocks.0.hook_resid_post"),
    )


def test_promoted_node_truth_is_not_inferred_from_surviving_edges() -> None:
    # This Q edge is removed from edge truth, but its two endpoints both name
    # the same official whole-head component.  Node truth must retain it.
    raw_edges = (
        ("blocks.2.hook_q_input[3]", "blocks.2.attn.hook_q[3]"),
        ("hook_pos_embed", "blocks.1.hook_mlp_in"),
        ("blocks.1.hook_mlp_out", "blocks.2.hook_resid_post"),
    )

    assert promoted_ground_truth_nodes(raw_edges) == (
        "blocks.2.attn.hook_result[3]",
        "blocks.0.hook_resid_pre",
        "blocks.1.hook_mlp_out",
        "blocks.2.hook_resid_post",
    )
    assert promoted_ground_truth_internal_nodes(raw_edges) == (
        "blocks.2.attn.hook_result[3]",
        "blocks.1.hook_mlp_out",
    )
    assert internal_ground_truth_edges(raw_edges) == ()


def test_one_layer_one_head_universe_matches_official_preparation() -> None:
    nodes, edges = official_prepared_universe(n_layers=1, n_heads=1)

    assert nodes == (
        "blocks.0.hook_resid_pre",
        "blocks.0.attn.hook_result[0]",
        "blocks.0.hook_mlp_out",
        "blocks.0.hook_resid_post",
    )
    assert edges == (
        ("blocks.0.hook_resid_pre", "blocks.0.attn.hook_result[0]"),
        ("blocks.0.hook_resid_pre", "blocks.0.hook_mlp_out"),
        ("blocks.0.attn.hook_result[0]", "blocks.0.hook_mlp_out"),
        ("blocks.0.attn.hook_result[0]", "blocks.0.hook_resid_post"),
        ("blocks.0.hook_mlp_out", "blocks.0.hook_resid_post"),
    )
    assert not any(source == target for source, target in edges)

    internal_nodes, internal_edges = patchable_internal_universe(n_layers=1, n_heads=1)
    assert internal_nodes == (
        "blocks.0.attn.hook_result[0]",
        "blocks.0.hook_mlp_out",
    )
    assert internal_edges == (
        ("blocks.0.attn.hook_result[0]", "blocks.0.hook_mlp_out"),
    )


def test_universe_counts_match_audited_panel_models() -> None:
    assert tuple(map(len, official_prepared_universe(n_layers=4, n_heads=4))) == (
        22,
        206,
    )
    assert tuple(map(len, official_prepared_universe(n_layers=3, n_heads=4))) == (
        17,
        117,
    )
    assert tuple(map(len, patchable_internal_universe(n_layers=4, n_heads=4))) == (
        20,
        166,
    )
    assert tuple(map(len, patchable_internal_universe(n_layers=3, n_heads=4))) == (
        15,
        87,
    )


def test_sha256_and_ground_truth_loader_validate_payload(tmp_path) -> None:
    edges = [("hook_embed", "blocks.0.hook_mlp_in")]
    path = tmp_path / "edges.pkl"
    path.write_bytes(pickle.dumps(edges))
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    assert sha256_file(path) == expected
    assert load_raw_ground_truth(path, expected_sha256=expected) == tuple(edges)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_raw_ground_truth(path, expected_sha256="0" * 64)


def test_categorical_accuracy_excludes_unspecified_bos_position() -> None:
    expected_ids = torch.tensor([[2, 0, 1], [1, 1, 0]])
    target = torch.nn.functional.one_hot(expected_ids, num_classes=3).float()
    predicted_ids = torch.tensor([[0, 0, 1], [2, 1, 2]])
    logits = torch.nn.functional.one_hot(predicted_ids, num_classes=3).float()

    metrics = categorical_accuracy(logits, target, exclude_bos=True)

    assert metrics.token_correct == 3
    assert metrics.token_total == 4
    assert metrics.token_accuracy == pytest.approx(0.75)
    assert metrics.context_correct == 1
    assert metrics.context_total == 2
    assert metrics.context_accuracy == pytest.approx(0.5)


def test_official_pair_fails_if_unique_data_flag_does_not_remove_duplicates() -> None:
    class Dataset:
        inputs = torch.tensor([[1, 2], [1, 2]])
        targets = torch.ones(2, 2, 1)

    class DuplicateCase:
        def is_categorical(self):
            return True

        def get_clean_data(self, **_kwargs):
            return Dataset()

        def get_corrupted_data(self, **_kwargs):
            return Dataset()

    with pytest.raises(RuntimeError, match="unique_data=True"):
        generate_official_pair(DuplicateCase(), sample_count=2)


def test_behavioral_eligibility_selects_first_jointly_correct_pairs() -> None:
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    clean_ids = torch.tensor([[0, 0], [0, 1], [0, 1], [0, 1]])
    source_ids = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 1]])

    class Dataset:
        def __init__(self, targets):
            self.inputs = inputs
            self.targets = torch.nn.functional.one_hot(targets, 2).float()

    class Case:
        def is_categorical(self):
            return True

        def get_clean_data(self, **_kwargs):
            return Dataset(clean_ids)

        def get_corrupted_data(self, **_kwargs):
            return Dataset(source_ids)

    class Model(torch.nn.Module):
        def forward(self, model_inputs):
            return torch.nn.functional.one_hot(model_inputs, 2).float()

    eligible = behaviorally_eligible_pair(
        Model(),
        Case(),
        required_count=2,
        candidate_count=4,
    )

    assert eligible.selected_candidate_indices == (0, 3)
    assert eligible.candidate_count == 4
    assert eligible.eligible_candidate_count == 2
    assert eligible.clean_incorrect_contexts == 1
    assert eligible.source_incorrect_contexts == 1
    assert eligible.either_incorrect_contexts == 2
    assert eligible.eligibility_physical_forwards == 2
    assert eligible.eligibility_model_examples == 8
    assert eligible.output_kind == "categorical"
    assert eligible.pair.clean_inputs.tolist() == [[0, 0], [1, 1]]


def test_behavioral_eligibility_can_enforce_response_blind_row_exclusions() -> None:
    clean_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    source_inputs = torch.tensor([[2, 2], [1, 0], [3, 2], [3, 3]])

    class Dataset:
        def __init__(self, inputs):
            self.inputs = inputs
            self.targets = torch.nn.functional.one_hot(inputs, 4).float()

    class Case:
        def is_categorical(self):
            return True

        def get_clean_data(self, **_kwargs):
            return Dataset(clean_inputs)

        def get_corrupted_data(self, **_kwargs):
            return Dataset(source_inputs)

    class Model(torch.nn.Module):
        def forward(self, model_inputs):
            return torch.nn.functional.one_hot(model_inputs, 4).float()

    eligible = behaviorally_eligible_pair(
        Model(),
        Case(),
        required_count=2,
        candidate_count=4,
        excluded_token_rows=((0, 0),),
        require_unique_selected_token_rows=True,
    )

    assert eligible.selected_candidate_indices == (1, 3)
    selected_rows = {
        tuple(row)
        for tensor in (eligible.pair.clean_inputs, eligible.pair.corrupt_inputs)
        for row in tensor.tolist()
    }
    assert len(selected_rows) == 4
    assert (0, 0) not in selected_rows
    assert eligible.row_disjoint_eligible_candidate_count == 2
    assert eligible.row_overlap_excluded_contexts == 2
    assert eligible.excluded_token_row_count == 1
