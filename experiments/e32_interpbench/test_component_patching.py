# ruff: noqa: EXE002
"""Toy-hook regression tests for the E32 component-patching acquisition path.

These tests live beside the implementation because E32 is being developed in
isolation.  Run them explicitly with ``pytest experiments/e32_interpbench``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from experiments.e32_interpbench.acquisition import (
    PRIMARY_CASE_ID,
    SECONDARY_CASE_IDS,
    acquire_official_pair_atlas,
    behavioral_pair_screen,
    build_case_patch_runner,
    categorical_scientific_output,
)
from experiments.e32_interpbench.component_patching import (
    BATCH_INVARIANCE_ATOL,
    ComponentPatchRunner,
    build_component_atlas,
    build_patch_units,
    class_center_logits,
    vector_pair_interactions,
)


class _ToyHookedTransformer(torch.nn.Module):
    """Tiny TransformerLens-like model with real named hook activations."""

    def __init__(self, *, n_layers: int = 2, n_heads: int = 2) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(n_layers=n_layers, n_heads=n_heads, device="cpu")
        self.scale = torch.nn.Parameter(torch.tensor(0.17, dtype=torch.float64))
        self.embed = torch.nn.Identity()
        self.hook_embed = torch.nn.Identity()
        self.hook_dict = {
            "hook_embed": self.hook_embed,
            **{
                f"blocks.{layer}.attn.hook_result": object()
                for layer in range(n_layers)
            },
            **{f"blocks.{layer}.hook_mlp_out": object() for layer in range(n_layers)},
        }
        self._active_hooks: dict[str, list[object]] = {}
        self.observed: dict[str, torch.Tensor] = {}

    def _hook(self, name: str, activation: torch.Tensor) -> torch.Tensor:
        for callback in self._active_hooks.get(name, ()):  # pragma: no branch
            activation = callback(activation, hook=SimpleNamespace(name=name))
        self.observed[name] = activation.detach().clone()
        return activation

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.hook_embed(self.embed(tokens))
        positions = torch.arange(tokens.shape[1], dtype=torch.float64)[None, :, None]
        hidden = torch.stack(
            (
                tokens.double(),
                tokens.double().square() / 7.0,
                positions[..., 0].expand_as(tokens).double(),
            ),
            dim=-1,
        )
        for layer in range(self.cfg.n_layers):
            head_outputs = []
            for head in range(self.cfg.n_heads):
                scale = self.scale + 0.03 * (layer + 1) + 0.01 * head
                function = torch.tanh if head % 2 == 0 else torch.sin
                head_outputs.append(function(hidden * scale))
            result = torch.stack(head_outputs, dim=2)
            result = self._hook(f"blocks.{layer}.attn.hook_result", result)
            hidden = hidden + result.sum(dim=2)
            mlp = torch.tanh(hidden.roll(1, dims=-1) * (0.09 + 0.01 * layer))
            mlp = self._hook(f"blocks.{layer}.hook_mlp_out", mlp)
            hidden = hidden + mlp
        return torch.stack(
            (
                hidden[..., 0],
                hidden[..., 1],
                hidden[..., 2],
                hidden.sum(dim=-1),
            ),
            dim=-1,
        )

    def run_with_hooks(
        self,
        tokens: torch.Tensor,
        *,
        fwd_hooks: list[tuple[str, object]],
        **_kwargs: object,
    ) -> torch.Tensor:
        active: dict[str, list[object]] = {}
        for name, callback in fwd_hooks:
            active.setdefault(name, []).append(callback)
        self._active_hooks = active
        try:
            # TransformerLens 1.19 calls ``forward`` directly here, bypassing
            # top-level module pre-hooks.  The runner must count at hook_embed.
            return self.forward(tokens)
        finally:
            self._active_hooks = {}


def _official_pair(
    batch: int = 3, *, target_model: _ToyHookedTransformer | None = None
) -> SimpleNamespace:
    clean = torch.tensor([[1, 2, 3, 4], [2, 1, 4, 3], [3, 4, 1, 2]], dtype=torch.long)[
        :batch
    ]
    corrupt = clean.flip(1).clone()
    if target_model is None:
        target_ids = clean.remainder(4)
        corrupt_target_ids = corrupt.remainder(4)
    else:
        with torch.inference_mode():
            target_ids = target_model(clean).argmax(dim=-1)
            corrupt_target_ids = target_model(corrupt).argmax(dim=-1)
    targets = torch.nn.functional.one_hot(target_ids, num_classes=4).double()
    corrupt_targets = torch.nn.functional.one_hot(
        corrupt_target_ids, num_classes=4
    ).double()
    return SimpleNamespace(
        clean_inputs=clean,
        clean_targets=targets,
        corrupt_inputs=corrupt,
        corrupt_targets=corrupt_targets,
    )


def _eligible_pair(pair: SimpleNamespace) -> SimpleNamespace:
    batch = len(pair.clean_inputs)
    return SimpleNamespace(
        pair=pair,
        candidate_count=batch + 3,
        eligible_candidate_count=batch + 1,
        clean_incorrect_contexts=1,
        source_incorrect_contexts=2,
        either_incorrect_contexts=2,
        selected_candidate_indices=tuple(range(batch)),
        eligibility_physical_forwards=2,
        eligibility_model_examples=2 * (batch + 3),
        output_kind="categorical",
        row_disjoint_eligible_candidate_count=batch + 1,
        row_overlap_excluded_contexts=0,
        excluded_token_row_count=0,
        selection_policy="first_jointly_correct",
    )


def test_units_and_exhaustive_atlas_have_exact_stable_order() -> None:
    units = build_patch_units(n_layers=2, n_heads=2)
    assert tuple(unit.component_id for unit in units) == (
        "blocks.0.attn.hook_result[0]",
        "blocks.0.attn.hook_result[1]",
        "blocks.0.hook_mlp_out",
        "blocks.1.attn.hook_result[0]",
        "blocks.1.attn.hook_result[1]",
        "blocks.1.hook_mlp_out",
    )

    atlas = build_component_atlas(units)

    # Same-layer heads are parallel, so head<->head is not a candidate edge.
    # Heads precede their MLP; every layer-0 component precedes layer 1.
    assert atlas.condition_count == 1 + len(units) + 13
    assert atlas.condition_ids[0] == "empty"
    assert atlas.condition_masks.shape == (20, 6)
    assert atlas.condition_masks.dtype == torch.bool
    assert atlas.condition_masks.sum(dim=1).tolist() == [0] + [1] * 6 + [2] * 13
    assert len(set(atlas.condition_ids)) == atlas.condition_count
    assert atlas.singleton_condition_indices == tuple(range(1, 7))
    assert atlas.pair_condition_indices == tuple(range(7, 20))
    assert (
        "blocks.0.attn.hook_result[0]",
        "blocks.0.attn.hook_result[1]",
    ) not in atlas.directed_edges
    assert (
        "blocks.0.attn.hook_result[0]",
        "blocks.0.hook_mlp_out",
    ) in atlas.directed_edges
    assert all(source != target for source, target in atlas.directed_edges)


def test_adapter_declared_edge_universe_controls_queried_pairs() -> None:
    units = build_patch_units(n_layers=2, n_heads=2)
    declared = (
        ("blocks.0.hook_resid_pre", "blocks.0.attn.hook_result[0]"),
        ("blocks.0.attn.hook_result[0]", "blocks.0.hook_mlp_out"),
        ("blocks.0.hook_mlp_out", "blocks.1.attn.hook_result[1]"),
        ("blocks.1.hook_mlp_out", "blocks.1.hook_resid_post"),
    )

    atlas = build_component_atlas(units, valid_directed_edges=declared)

    assert atlas.directed_edges == declared[1:3]
    assert atlas.pair_component_indices == ((0, 2), (2, 4))
    assert atlas.condition_count == 1 + len(units) + 2


def test_prepare_captures_every_donor_hook_and_counts_live_model_examples() -> None:
    model = _ToyHookedTransformer().eval()
    runner = ComponentPatchRunner(model, build_patch_units(2, 2))

    prepared = runner.prepare_pair(_official_pair())

    assert tuple(prepared.donor_activations) == (
        "blocks.0.attn.hook_result",
        "blocks.0.hook_mlp_out",
        "blocks.1.attn.hook_result",
        "blocks.1.hook_mlp_out",
    )
    assert prepared.donor_activations["blocks.0.attn.hook_result"].shape == (
        3,
        4,
        2,
        3,
    )
    assert prepared.donor_activations["blocks.0.hook_mlp_out"].shape == (3, 4, 3)
    assert prepared.clean_output.shape == prepared.corrupt_output.shape == (3, 4, 4)
    assert runner.forward_counts() == {
        "physical_forwards": 2,
        "model_examples": 6,
    }


def test_whole_head_and_mlp_patches_replace_every_position() -> None:
    model = _ToyHookedTransformer().eval()
    units = build_patch_units(2, 2)
    runner = ComponentPatchRunner(model, units, condition_batch_size=4)
    prepared = runner.prepare_pair(_official_pair(batch=2))

    head_mask = torch.tensor([[0, 1, 0, 0, 0, 0]], dtype=torch.bool)
    head_output = runner.outputs_for_masks(prepared, head_mask)
    observed_head = model.observed["blocks.0.attn.hook_result"]
    torch.testing.assert_close(
        observed_head[:, :, 1],
        prepared.donor_activations["blocks.0.attn.hook_result"][:, :, 1],
    )
    assert head_output.shape == (1, 2, 4, 4)

    mlp_mask = torch.tensor([[0, 0, 0, 0, 0, 1]], dtype=torch.bool)
    runner.outputs_for_masks(prepared, mlp_mask)
    torch.testing.assert_close(
        model.observed["blocks.1.hook_mlp_out"],
        prepared.donor_activations["blocks.1.hook_mlp_out"],
    )


def test_batched_exhaustive_atlas_returns_full_outputs_and_exact_vector_j() -> None:
    model = _ToyHookedTransformer().eval()
    units = build_patch_units(2, 2)
    runner = ComponentPatchRunner(model, units, condition_batch_size=5)
    prepared = runner.prepare_pair(_official_pair(batch=2))
    before = runner.forward_counts()

    result = runner.acquire_atlas(prepared)

    pair_count = len(build_component_atlas(units).directed_edges)
    condition_count = 1 + len(units) + pair_count
    assert result.condition_outputs.shape == (condition_count, 2, 4, 4)
    assert result.singleton_outputs.shape == (len(units), 2, 4, 4)
    assert result.pair_outputs.shape == (pair_count, 2, 4, 4)
    assert result.pair_interactions.shape == (pair_count, 2, 4, 4)
    assert result.raw_pair_interactions.shape == (pair_count, 2, 4, 4)
    torch.testing.assert_close(result.condition_outputs[0], prepared.clean_output)
    expected_j = vector_pair_interactions(
        empty_output=prepared.clean_output,
        singleton_outputs=result.singleton_outputs,
        pair_outputs=result.pair_outputs,
        pair_component_indices=result.atlas.pair_component_indices,
    )
    torch.testing.assert_close(result.pair_interactions, expected_j)
    for pair_row, (left, right) in enumerate(result.atlas.pair_component_indices):
        torch.testing.assert_close(
            result.pair_interactions[pair_row],
            class_center_logits(result.pair_outputs[pair_row])
            - class_center_logits(result.singleton_outputs[left])
            - class_center_logits(result.singleton_outputs[right])
            + class_center_logits(prepared.clean_output),
        )
        torch.testing.assert_close(
            result.raw_pair_interactions[pair_row],
            result.pair_outputs[pair_row]
            - result.singleton_outputs[left]
            - result.singleton_outputs[right]
            + prepared.clean_output,
        )
    # The empty condition reuses the already observed clean output.  Remaining
    # conditions are evaluated in five-condition physical batches.
    nonempty = condition_count - 1
    after = runner.forward_counts()
    assert after["physical_forwards"] - before["physical_forwards"] == math.ceil(
        nonempty / 5
    )
    assert after["model_examples"] - before["model_examples"] == nonempty * 2
    assert result.query_accounting["patched_model_examples"] == nonempty * 2


def test_centered_vector_j_is_invariant_to_arbitrary_logit_gauge_shifts() -> None:
    generator = torch.Generator().manual_seed(3201)
    empty = torch.randn(2, 3, 5, generator=generator, dtype=torch.float64)
    singletons = torch.randn(3, 2, 3, 5, generator=generator, dtype=torch.float64)
    pairs = torch.randn(2, 2, 3, 5, generator=generator, dtype=torch.float64)
    pair_indices = ((0, 1), (1, 2))
    expected = vector_pair_interactions(
        empty_output=empty,
        singleton_outputs=singletons,
        pair_outputs=pairs,
        pair_component_indices=pair_indices,
    )
    empty_shift = torch.randn(2, 3, 1, generator=generator, dtype=torch.float64)
    singleton_shift = torch.randn(3, 2, 3, 1, generator=generator, dtype=torch.float64)
    pair_shift = torch.randn(2, 2, 3, 1, generator=generator, dtype=torch.float64)

    shifted = vector_pair_interactions(
        empty_output=empty + empty_shift,
        singleton_outputs=singletons + singleton_shift,
        pair_outputs=pairs + pair_shift,
        pair_component_indices=pair_indices,
    )

    torch.testing.assert_close(shifted, expected, atol=1e-12, rtol=1e-12)


def test_categorical_scientific_output_centers_classes_and_drops_bos() -> None:
    logits = torch.arange(2 * 4 * 5, dtype=torch.float64).reshape(2, 4, 5)

    projected = categorical_scientific_output(logits)

    assert projected.shape == (2, 3, 5)
    torch.testing.assert_close(
        projected.mean(dim=-1), torch.zeros(2, 3, dtype=torch.float64)
    )
    torch.testing.assert_close(projected, class_center_logits(logits)[:, 1:])


def test_behavior_screen_is_per_context_and_excludes_bos() -> None:
    targets = torch.nn.functional.one_hot(
        torch.tensor([[4, 0, 1], [3, 1, 2]]), num_classes=5
    ).double()
    clean = targets.mul(3.0)
    corrupt = targets.mul(2.0)
    # BOS errors do not count.  A non-BOS error in the second corrupt context
    # makes only that paired context ineligible for a strict paired analysis.
    clean[0, 0] = clean[0, 0].roll(1)
    corrupt[1, 2] = corrupt[1, 2].roll(1)

    screen = behavioral_pair_screen(
        clean_output=clean,
        corrupt_output=corrupt,
        clean_targets=targets,
        corrupt_targets=targets,
    )

    assert screen["clean_context_correct"].tolist() == [True, True]
    assert screen["corrupt_context_correct"].tolist() == [True, False]
    assert screen["both_context_correct"].tolist() == [True, False]
    assert screen["both_context_correct_count"] == 1
    assert screen["context_count"] == 2


def test_batch_invariance_smoke_is_exact_and_acquisition_runs_it_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ToyHookedTransformer(n_layers=3, n_heads=4).eval()
    runner = build_case_patch_runner(case_id=97, model=model, condition_batch_size=16)
    prepared = runner.prepare_pair(_official_pair(batch=2))
    atlas = build_component_atlas(runner.units)
    smoke = runner.batch_invariance_smoke(
        prepared,
        atlas.condition_masks[:7],
        candidate_condition_batch_size=4,
    )
    assert smoke["passed"]
    assert smoke["maximum_absolute_error"] < 1e-12
    assert smoke["empty_no_op_evaluated"]
    assert smoke["empty_no_op_maximum_absolute_error"] < 1e-12

    events: list[str] = []
    original_smoke = runner.batch_invariance_smoke
    original_acquire = runner.acquire_atlas

    def record_smoke(*args: object, **kwargs: object) -> dict[str, object]:
        events.append("smoke")
        return original_smoke(*args, **kwargs)

    def record_acquire(*args: object, **kwargs: object):
        events.append("atlas")
        return original_acquire(*args, **kwargs)

    monkeypatch.setattr(runner, "batch_invariance_smoke", record_smoke)
    monkeypatch.setattr(runner, "acquire_atlas", record_acquire)
    eligible = _eligible_pair(_official_pair(batch=2, target_model=model))
    artifact = acquire_official_pair_atlas(
        case_id=97,
        runner=runner,
        official_pair=eligible,
        smoke_condition_count=7,
        smoke_condition_batch_size=4,
    )

    assert events[:2] == ["smoke", "atlas"]
    assert artifact["case_id"] == 97
    assert artifact["batch_invariance_smoke"]["passed"]
    assert artifact["batch_invariance_smoke"]["absolute_tolerance"] == 1e-4
    assert artifact["condition_outputs"].shape == (103, 2, 4, 4)
    assert artifact["scientific_condition_outputs"].shape == (103, 2, 3, 4)
    assert artifact["pair_interactions"].shape == (87, 2, 3, 4)
    assert artifact["raw_pair_interactions"].shape == (87, 2, 4, 4)
    assert artifact["scientific_output_coordinate"] == (
        "class_centered_logits_non_bos_positions"
    )
    assert artifact["behavioral_pair_screen"]["context_count"] == 2
    assert artifact["behavioral_pair_screen"]["row_policy"] == (
        "require_every_selected_clean_and_source_context_correct"
    )
    assert artifact["behavioral_eligibility"]["candidate_count"] == 5
    assert artifact["behavioral_eligibility"]["selected_candidate_indices"] == (
        0,
        1,
    )
    assert artifact["behavioral_eligibility"]["output_kind"] == "categorical"
    assert artifact["eligibility_query_accounting"] == {
        "physical_forwards_before_runner": 2,
        "model_examples_before_runner": 10,
        "source": "adapter_behaviorally_eligible_pair",
    }
    assert artifact["pair_patch_semantics"] == (
        "symmetric_two_component_mask; edge_arrow_is_topology_imposed"
    )
    assert artifact["candidate_edge_universe_source"] == (
        "adapter.patchable_internal_universe"
    )
    assert artifact["clean_targets"].shape == (2, 4, 4)
    assert PRIMARY_CASE_ID == 97
    assert SECONDARY_CASE_IDS == (2, 21)
    assert BATCH_INVARIANCE_ATOL == 1e-4


def test_failed_smoke_blocks_full_atlas(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _ToyHookedTransformer(n_layers=3, n_heads=4).eval()
    runner = build_case_patch_runner(case_id=97, model=model)
    atlas_calls = 0

    def failed_smoke(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"passed": False, "maximum_absolute_error": 1.0}

    def count_atlas(*_args: object, **_kwargs: object) -> None:
        nonlocal atlas_calls
        atlas_calls += 1
        raise AssertionError("full atlas ran after a failed smoke")

    monkeypatch.setattr(runner, "batch_invariance_smoke", failed_smoke)
    monkeypatch.setattr(runner, "acquire_atlas", count_atlas)
    with pytest.raises(RuntimeError, match="batch-invariance smoke failed"):
        acquire_official_pair_atlas(
            case_id=97,
            runner=runner,
            official_pair=_eligible_pair(_official_pair(batch=2, target_model=model)),
            smoke_condition_count=3,
            smoke_condition_batch_size=2,
        )
    assert atlas_calls == 0


def test_unfiltered_or_behaviorally_incorrect_pairs_are_blocked_before_atlas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ToyHookedTransformer(n_layers=3, n_heads=4).eval()
    runner = build_case_patch_runner(case_id=97, model=model)
    raw_pair = _official_pair(batch=2)
    with pytest.raises(TypeError, match="BehaviorallyEligiblePair"):
        acquire_official_pair_atlas(
            case_id=97,
            runner=runner,
            official_pair=raw_pair,
        )
    assert runner.forward_counts() == {"physical_forwards": 0, "model_examples": 0}

    smoke_calls = 0

    def count_smoke(*_args: object, **_kwargs: object) -> None:
        nonlocal smoke_calls
        smoke_calls += 1
        raise AssertionError("smoke ran after failed behavioral gate")

    monkeypatch.setattr(runner, "batch_invariance_smoke", count_smoke)
    with pytest.raises(RuntimeError, match="behavioral eligibility recheck failed"):
        acquire_official_pair_atlas(
            case_id=97,
            runner=runner,
            official_pair=_eligible_pair(raw_pair),
        )
    assert smoke_calls == 0
    assert runner.forward_counts() == {"physical_forwards": 2, "model_examples": 4}


def test_malformed_masks_fail_before_any_patched_forward() -> None:
    runner = ComponentPatchRunner(
        _ToyHookedTransformer().eval(), build_patch_units(2, 2)
    )
    prepared = runner.prepare_pair(_official_pair(batch=2))
    before = runner.forward_counts()

    with pytest.raises(ValueError, match="boolean matrix"):
        runner.outputs_for_masks(prepared, torch.zeros(2, 6, dtype=torch.uint8))
    with pytest.raises(ValueError, match="unit count"):
        runner.outputs_for_masks(prepared, torch.zeros(2, 5, dtype=torch.bool))

    assert runner.forward_counts() == before
