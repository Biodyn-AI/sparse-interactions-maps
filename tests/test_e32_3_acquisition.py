# ruff: noqa: EXE002
"""Toy tests for E32.3 whole-component and set-mask acquisition."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from experiments.e32_interpbench.adapter import OfficialDataPair
from experiments.e32_interpbench.e32_3_acquisition import (
    FrozenNodeSupport,
    acquire_set_interventions,
    activation_shift_rms,
    build_set_mask_plan,
    prepare_bidirectional_component_pair,
)


@dataclass(frozen=True)
class _Unit:
    component_id: str
    hook_name: str
    kind: str
    head_index: int | None = None


@dataclass(frozen=True)
class _Prepared:
    clean_inputs: torch.Tensor
    corrupt_inputs: torch.Tensor
    clean_output: torch.Tensor
    corrupt_output: torch.Tensor
    donor_activations: dict[str, torch.Tensor]


def _pair() -> OfficialDataPair:
    clean = torch.tensor([[1, 2], [3, 4]])
    corrupt = torch.tensor([[5, 6], [7, 8]])
    targets = torch.zeros(2, 2, 3)
    return OfficialDataPair(
        clean_inputs=clean,
        clean_targets=targets,
        corrupt_inputs=corrupt,
        corrupt_targets=targets.clone(),
    )


class _Runner:
    def __init__(self) -> None:
        self.units = (
            _Unit("h0", "attn", "attention_head", 0),
            _Unit("h1", "attn", "attention_head", 1),
            _Unit("m0", "mlp", "mlp"),
        )
        self._physical = 0
        self._examples = 0
        self.prepared_pairs: list[OfficialDataPair] = []
        self.last_masks: torch.Tensor | None = None

    def forward_counts(self) -> dict[str, int]:
        return {
            "physical_forwards": self._physical,
            "model_examples": self._examples,
        }

    def prepare_pair(self, pair: OfficialDataPair) -> _Prepared:
        self.prepared_pairs.append(pair)
        batch = len(pair.clean_inputs)
        self._physical += 2
        self._examples += 2 * batch
        clean_value = float(pair.clean_inputs[0, 0])
        corrupt_value = float(pair.corrupt_inputs[0, 0])
        clean_output = torch.full((batch, 2, 3), clean_value)
        corrupt_output = torch.full((batch, 2, 3), corrupt_value)
        donor = {
            "attn": torch.stack(
                (
                    torch.full((batch, 2, 1), corrupt_value),
                    torch.full((batch, 2, 1), corrupt_value + 1.0),
                ),
                dim=2,
            ),
            "mlp": torch.full((batch, 2, 1), corrupt_value + 2.0),
        }
        return _Prepared(
            clean_inputs=pair.clean_inputs,
            corrupt_inputs=pair.corrupt_inputs,
            clean_output=clean_output,
            corrupt_output=corrupt_output,
            donor_activations=donor,
        )

    def outputs_for_masks(
        self, prepared: _Prepared, masks: torch.Tensor
    ) -> torch.Tensor:
        self.last_masks = masks.clone()
        self._physical += 1
        self._examples += len(masks) * len(prepared.clean_inputs)
        values = masks.to(torch.float32).sum(dim=1)
        return values[:, None, None, None].expand(-1, 2, 2, 3).clone()


def test_bidirectional_preparation_uses_one_runner_and_exact_swapped_pair() -> None:
    runner = _Runner()
    pair = _pair()

    prepared = prepare_bidirectional_component_pair(runner=runner, pair=pair)

    assert len(runner.prepared_pairs) == 2
    swapped = runner.prepared_pairs[1]
    assert torch.equal(swapped.clean_inputs, pair.corrupt_inputs)
    assert torch.equal(swapped.corrupt_inputs, pair.clean_inputs)
    assert prepared.query_accounting == {
        "physical_forwards": 4,
        "model_examples": 8,
        "normal_prepare_forwards": 2,
        "swapped_prepare_forwards": 2,
    }
    assert torch.equal(prepared.clean_output, prepared.normal.clean_output)
    assert torch.equal(prepared.corrupt_output, prepared.normal.corrupt_output)
    assert set(prepared.clean_activations) == {"attn", "mlp"}
    assert set(prepared.corrupt_activations) == {"attn", "mlp"}


def test_activation_shift_rms_is_aligned_to_whole_heads_and_mlps() -> None:
    runner = _Runner()
    prepared = prepare_bidirectional_component_pair(runner=runner, pair=_pair())

    scores = activation_shift_rms(
        units=runner.units,
        clean_activations=prepared.clean_activations,
        corrupt_activations=prepared.corrupt_activations,
    )

    # Every toy corrupt activation is exactly four above its clean counterpart.
    assert scores == pytest.approx((4.0, 4.0, 4.0))


def test_set_mask_plan_preserves_exact_selected_complement_and_all_identities() -> None:
    supports = (
        FrozenNodeSupport(
            support_id="singleton:q25",
            method="singleton_output_effect",
            quantile=0.25,
            selected_nodes=("n2",),
            kind="deterministic",
        ),
        FrozenNodeSupport(
            support_id="random:0:q50",
            method="random_control",
            quantile=0.50,
            selected_nodes=("n0", "n3"),
            kind="random",
            replicate_index=0,
        ),
    )

    plan = build_set_mask_plan(
        component_ids=("n0", "n1", "n2", "n3"), supports=supports
    )

    assert torch.equal(
        plan.selected_masks,
        torch.tensor([[False, False, True, False], [True, False, False, True]]),
    )
    assert torch.equal(plan.complement_masks, ~plan.selected_masks)
    assert torch.equal(plan.all_mask, torch.ones(4, dtype=torch.bool))
    assert plan.supports == supports

    with pytest.raises(ValueError, match="ceil"):
        build_set_mask_plan(
            component_ids=("n0", "n1", "n2", "n3"),
            supports=(
                FrozenNodeSupport(
                    support_id="bad",
                    method="singleton_output_effect",
                    quantile=0.25,
                    selected_nodes=("n0", "n1"),
                    kind="deterministic",
                ),
            ),
        )


def test_set_acquisition_runs_all_selected_complement_and_anchor_masks_once() -> None:
    runner = _Runner()
    prepared = prepare_bidirectional_component_pair(runner=runner, pair=_pair())
    supports = (
        FrozenNodeSupport(
            support_id="s:q50",
            method="singleton_output_effect",
            quantile=0.5,
            selected_nodes=("h0", "m0"),
            kind="deterministic",
        ),
        FrozenNodeSupport(
            support_id="r0:q50",
            method="random_control",
            quantile=0.5,
            selected_nodes=("h1", "m0"),
            kind="random",
            replicate_index=0,
        ),
    )
    plan = build_set_mask_plan(
        component_ids=tuple(unit.component_id for unit in runner.units),
        supports=supports,
    )

    acquired = acquire_set_interventions(
        runner=runner,
        prepared=prepared.normal,
        plan=plan,
    )

    assert runner.last_masks is not None
    assert torch.equal(
        runner.last_masks,
        torch.cat(
            (
                plan.all_mask.unsqueeze(0),
                plan.selected_masks,
                plan.complement_masks,
            )
        ),
    )
    assert acquired.all_nodes_output.shape == (2, 2, 3)
    assert acquired.selected_outputs.shape == (2, 2, 2, 3)
    assert acquired.complement_outputs.shape == (2, 2, 2, 3)
    assert acquired.query_accounting == {
        "condition_count": 5,
        "selected_condition_count": 2,
        "complement_condition_count": 2,
        "all_nodes_condition_count": 1,
        "physical_forwards": 1,
        "model_examples": 10,
    }
