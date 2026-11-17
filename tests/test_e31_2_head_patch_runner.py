# ruff: noqa: EXE002
from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from experiments.e31_sparse_interactions.e31_2_head_patch_runner import (
    ATTENTION_HEAD_SHAPE,
    COARSE_COMPONENT_COUNT,
    E312HeadPatchRunner,
    reconstruct_selected_head_composite,
    split_projected_head_contributions,
)


class _Conv1D(torch.nn.Module):
    """Small Hugging Face Conv1D-layout projection: weight is (in, out)."""

    def __init__(self, width: int, scale: float) -> None:
        super().__init__()
        diagonal = torch.linspace(scale, 1.5 * scale, width, dtype=torch.float64)
        weight = torch.diag(diagonal)
        # A nonsymmetric upper triangle makes an accidental W versus W.T
        # implementation fail even though GPT-2's Conv1D matrix is square.
        off_diagonal = torch.arange(width * width, dtype=torch.float64).reshape(
            width, width
        )
        weight += 1e-5 * torch.triu(off_diagonal, diagonal=1)
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(torch.linspace(-0.001, 0.001, width).double())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.matmul(hidden, self.weight) + self.bias


class _Block(torch.nn.Module):
    def __init__(self, layer: int, width: int) -> None:
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.c_proj = _Conv1D(width, 0.025 + 0.001 * layer)
        self.mlp = torch.nn.Module()
        self.mlp.c_proj = _Conv1D(width, 0.015 + 0.001 * layer)


class _HeadToyModel(torch.nn.Module):
    """Cheap GPT-2-layout model whose c_proj inputs contain contiguous heads."""

    def __init__(self, *, donor_logit_bias: float = 0.0) -> None:
        super().__init__()
        self.donor_logit_bias = donor_logit_bias
        width = 24
        self.config = SimpleNamespace(
            n_layer=12,
            n_head=12,
            n_embd=width,
            _attn_implementation="eager",
            _name_or_path="unit-test-gpt2-small-layout",
            model_type="gpt2",
        )
        self.embedding = torch.nn.Embedding(32, width, dtype=torch.float64)
        weights = torch.zeros(32, width, dtype=torch.float64)
        weights[1] = torch.linspace(0.04, 0.08, width)
        weights[2] = -torch.linspace(0.05, 0.09, width)
        with torch.no_grad():
            self.embedding.weight.copy_(weights)
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList(
            [_Block(layer, width) for layer in range(12)]
        )

    def forward(self, *, input_ids: torch.Tensor) -> SimpleNamespace:
        hidden = self.embedding(input_ids)
        for layer, block in enumerate(self.transformer.h):
            global_signal = hidden.mean(dim=1, keepdim=True)
            # The final axis is already merged as 12 contiguous two-wide heads.
            attention_input = hidden + (0.10 + 0.002 * layer) * global_signal
            hidden = hidden + block.attn.c_proj(attention_input)
            mlp_input = hidden + 0.04 * hidden.mean(dim=1, keepdim=True)
            hidden = hidden + block.mlp.c_proj(mlp_input)
        score = 30.0 * hidden[:, -1].mean(dim=-1)
        logits = torch.zeros(len(input_ids), input_ids.shape[1], 32, dtype=hidden.dtype)
        logits[:, :, 1] = score[:, None]
        logits[:, :, 2] = -score[:, None] + self.donor_logit_bias
        return SimpleNamespace(logits=logits)


class _Tokenizer:
    prompt_ids: ClassVar[dict[str, list[int]]] = {
        "base prompt": [3, 4, 1, 5, 6],
        "donor prompt": [3, 4, 2, 5, 6],
    }

    def __call__(self, prompt: str, *, return_tensors: str) -> SimpleNamespace:
        assert return_tensors == "pt"
        return SimpleNamespace(
            input_ids=torch.tensor([self.prompt_ids[prompt]], dtype=torch.long)
        )

    def encode(self, target: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [{"base": 1, "donor": 2}[target.strip()]]


def _record(
    *,
    pair_id: str = "e31-2-discovery-000",
    split: str = "discovery",
    reverse: bool = False,
) -> dict[str, str]:
    record = {
        "pair_id": pair_id,
        "split": split,
        "stratum_id": "abba-name-order-0",
        "template_id": "abba-0",
        "template": "tiny head IOI fixture",
        "name_a": "base",
        "name_b": "donor",
        "place": "store",
        "object": "gift",
        "base_prompt": "base prompt",
        "donor_prompt": "donor prompt",
        "base_target": "base",
        "donor_target": "donor",
        "base_repeated_subject": "donor",
        "donor_repeated_subject": "base",
    }
    if reverse:
        record.update(
            {
                "base_prompt": "donor prompt",
                "donor_prompt": "base prompt",
                "base_target": "donor",
                "donor_target": "base",
                "base_repeated_subject": "base",
                "donor_repeated_subject": "donor",
            }
        )
    return record


def _empty_masks(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(rows, COARSE_COMPONENT_COUNT, dtype=torch.uint8),
        torch.zeros(rows, *ATTENTION_HEAD_SHAPE, dtype=torch.uint8),
    )


def test_projected_head_contributions_and_selected_composite_are_exact() -> None:
    projection = _Conv1D(6, 0.2).eval()
    recipient = torch.arange(24, dtype=torch.float64).reshape(2, 2, 6) / 10.0
    donor = recipient.flip(-1)
    selected = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.uint8)

    contributions = split_projected_head_contributions(
        recipient, projection, head_count=3
    )
    reconstructed = contributions.sum(dim=-2) + projection.bias
    composite = reconstruct_selected_head_composite(
        recipient,
        donor,
        selected,
        projection,
        head_count=3,
    )
    explicit = recipient.reshape(2, 2, 3, 2).clone()
    donor_heads = donor.reshape(2, 2, 3, 2)
    explicit = torch.where(selected[:, None, :, None].bool(), donor_heads, explicit)

    torch.testing.assert_close(reconstructed, projection(recipient), atol=1e-14, rtol=0)
    torch.testing.assert_close(
        composite, projection(explicit.reshape(2, 2, 6)), atol=1e-14, rtol=0
    )
    assert contributions.shape == (2, 2, 3, 6)


def test_prepare_captures_clean_head_vectors_and_norm_diagnostics() -> None:
    runner = E312HeadPatchRunner(_HeadToyModel().eval(), _Tokenizer(), batch_size=8)

    prepared = runner.prepare_context(_record())

    assert runner.head_components[:13] == tuple(
        [(0, head) for head in range(12)] + [(1, 0)]
    )
    assert len(prepared.base_merged_head_outputs) == 12
    assert len(prepared.donor_merged_head_outputs) == 12
    assert prepared.base_merged_head_outputs[0].shape == (5, 24)
    assert prepared.clean_base_correct
    assert prepared.clean_donor_correct
    assert prepared.clean_endpoint > 0.25
    assert prepared.clean_metadata["attention_head_mask_shape"] == [12, 12]
    assert prepared.clean_metadata["head_patch_site"] == "attention.c_proj input"
    diagnostics = prepared.clean_head_norm_diagnostics
    assert diagnostics["head_count"] == 144
    assert diagnostics["finite"]
    assert diagnostics["maximum_clean_reconstruction_absolute_error"] < 1e-12
    assert len(diagnostics["layers"]) == 12
    assert len(diagnostics["layers"][0]["donor_projected_l2_by_head"]) == 12


def test_prepare_preserves_original_reverse_direction_custody_metadata() -> None:
    runner = E312HeadPatchRunner(_HeadToyModel().eval(), _Tokenizer(), batch_size=8)
    record = _record(pair_id="e31-2-reverse", reverse=True)

    prepared = runner.prepare_context(record)

    for key in (
        "pair_id",
        "name_a",
        "name_b",
        "base_prompt",
        "donor_prompt",
        "base_target",
        "donor_target",
        "base_repeated_subject",
        "donor_repeated_subject",
    ):
        assert prepared.clean_metadata[key] == record[key]
    assert prepared.clean_base_correct
    assert prepared.clean_donor_correct


def test_head_patches_are_all_position_normalized_and_reconstructed() -> None:
    runner = E312HeadPatchRunner(_HeadToyModel().eval(), _Tokenizer(), batch_size=2)
    prepared = runner.prepare_context(_record())
    coarse, heads = _empty_masks(4)
    heads[1, 5, 0] = 1
    heads[2, 8, [1, 3, 5, 7]] = 1
    heads[3, 5] = 1
    heads[3, 8] = 1

    responses = runner.responses_from_prepared(prepared, coarse, heads)

    assert responses.normalized_transfer.shape == (4,)
    assert responses.normalized_transfer.dtype == torch.float64
    assert responses.normalized_transfer[0].item() == pytest.approx(0.0, abs=1e-12)
    assert responses.normalized_transfer[1:].abs().max() > 0.0
    torch.testing.assert_close(
        responses.raw_margin_shift_from_base,
        responses.raw_donor_minus_base_margin
        - prepared.clean_base_donor_minus_base_margin,
    )
    torch.testing.assert_close(
        responses.normalized_transfer,
        responses.raw_margin_shift_from_base / prepared.clean_endpoint,
    )
    assert responses.composite_reconstruction["passed"]
    assert responses.composite_reconstruction["maximum_absolute_error"] < 1e-12
    assert responses.composite_reconstruction["evaluated_attention_calls"] == 48


def test_all_heads_equal_the_unchanged_whole_attention_patch() -> None:
    runner = E312HeadPatchRunner(_HeadToyModel().eval(), _Tokenizer(), batch_size=2)
    prepared = runner.prepare_context(_record())
    coarse, heads = _empty_masks(4)
    # Compare all-head and E31.1 whole-attention patches at layers 5 and 8.
    heads[0, 5] = 1
    coarse[1, 10] = 1
    heads[2, 8] = 1
    coarse[3, 16] = 1

    responses = runner.responses_from_prepared(prepared, coarse, heads)

    torch.testing.assert_close(
        responses.raw_donor_minus_base_margin[[0, 2]],
        responses.raw_donor_minus_base_margin[[1, 3]],
        atol=1e-9,
        rtol=1e-9,
    )
    torch.testing.assert_close(
        responses.normalized_transfer[[0, 2]],
        responses.normalized_transfer[[1, 3]],
        atol=1e-9,
        rtol=1e-9,
    )


def test_batch_smoke_checks_raw_normalized_and_composite_coordinates() -> None:
    runner = E312HeadPatchRunner(_HeadToyModel().eval(), _Tokenizer(), batch_size=2)
    prepared = runner.prepare_context(_record())
    coarse, heads = _empty_masks(5)
    heads[1, 5, 2] = 1
    heads[2, 8, 4] = 1
    heads[3, 5, [1, 3, 5, 7]] = 1
    coarse[4, 20] = 1

    smoke = runner.batch_invariance_smoke(
        prepared,
        coarse,
        heads,
        reference_batch_size=1,
        candidate_batch_size=4,
    )

    assert smoke["passed"]
    assert smoke["mask_rows"] == 5
    assert set(smoke["coordinates"]) == {
        "normalized_transfer",
        "raw_donor_minus_base_margin",
        "raw_margin_shift_from_base",
    }
    assert smoke["composite_reconstruction"]["passed"]


def test_conflicts_and_malformed_masks_fail_before_a_patched_forward() -> None:
    model = _HeadToyModel().eval()
    runner = E312HeadPatchRunner(model, _Tokenizer(), batch_size=2)
    prepared = runner.prepare_context(_record())
    forward_count = 0

    def count_forward(
        _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...]
    ) -> None:
        nonlocal forward_count
        forward_count += 1

    handle = model.register_forward_pre_hook(count_forward)
    try:
        coarse, heads = _empty_masks(1)
        coarse[0, 10] = 1
        heads[0, 5, 2] = 1
        with pytest.raises(ValueError, match="whole-attention and fine-head"):
            runner.responses_from_prepared(prepared, coarse, heads)

        bad_heads = torch.zeros(1, 12, 11)
        with pytest.raises(ValueError, match="attention_head_masks"):
            runner.responses_from_prepared(prepared, torch.zeros(1, 24), bad_heads)

        coarse, _heads = _empty_masks(1)
        heads = torch.zeros(1, 12, 12, dtype=torch.float64)
        heads[0, 4, 1] = 0.5
        with pytest.raises(ValueError, match="binary"):
            runner.responses_from_prepared(prepared, coarse, heads)
    finally:
        handle.remove()

    assert forward_count == 0
