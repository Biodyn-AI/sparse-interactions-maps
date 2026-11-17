# ruff: noqa: EXE002
"""Tests for response-blind E32.2 row custody and node shortlisting."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch

from experiments.e32_interpbench.e32_2_acquisition import (
    PRIOR_E32_1_CANDIDATE_POOLS,
    _validated_request,
    assert_role_rows_fresh,
    induced_edge_subsequence,
    prior_candidate_token_rows,
    select_top_singleton_nodes,
    token_rows,
)
from experiments.e32_interpbench.e32_2_protocol import split_request


@dataclass(frozen=True)
class _Pair:
    clean_inputs: torch.Tensor
    corrupt_inputs: torch.Tensor


def _pair(start: int, count: int) -> _Pair:
    clean = torch.arange(start, start + 2 * count).reshape(count, 2)
    corrupt = torch.arange(start + 10_000, start + 10_000 + 2 * count).reshape(count, 2)
    return _Pair(clean_inputs=clean, corrupt_inputs=corrupt)


def test_prior_exclusion_regenerates_every_candidate_role_row() -> None:
    observed_calls: list[tuple[int, int, int]] = []

    def generate(
        _case: object, *, sample_count: int, clean_seed: int, corrupt_seed: int
    ) -> _Pair:
        observed_calls.append((sample_count, clean_seed, corrupt_seed))
        return _pair(clean_seed * 100, sample_count)

    rows = prior_candidate_token_rows(object(), generate_pair=generate)

    assert observed_calls == list(PRIOR_E32_1_CANDIDATE_POOLS)
    assert len(rows) == 2 * (80 + 160)
    assert len(set(rows)) == len(rows)


def test_role_freshness_rejects_internal_prior_and_cross_split_overlap() -> None:
    prior = ((1, 2), (3, 4))
    discovery_clean = torch.tensor([[10, 11], [12, 13]])
    discovery_source = torch.tensor([[20, 21], [22, 23]])
    summary = assert_role_rows_fresh(
        clean_inputs=discovery_clean,
        corrupt_inputs=discovery_source,
        excluded_rows=prior,
        expected_count=2,
        role="discovery",
    )
    assert summary["selected_unique_role_rows"] == 4
    assert summary["excluded_prior_row_count"] == 2
    assert summary["overlap_count"] == 0

    with pytest.raises(RuntimeError, match="token-row leakage"):
        assert_role_rows_fresh(
            clean_inputs=discovery_clean,
            corrupt_inputs=torch.tensor([[20, 21], [1, 2]]),
            excluded_rows=prior,
            expected_count=2,
            role="discovery",
        )
    with pytest.raises(RuntimeError, match="token-row leakage"):
        assert_role_rows_fresh(
            clean_inputs=discovery_clean,
            corrupt_inputs=torch.tensor([[20, 21], [10, 11]]),
            excluded_rows=(),
            expected_count=2,
            role="discovery",
        )


def test_top_half_singleton_selection_is_energy_ranked_and_canonically_tied() -> None:
    node_ids = ("n0", "n1", "n2", "n3")
    clean = torch.zeros(2, 2, 3)
    singleton = torch.zeros(4, 2, 2, 3)
    singleton[0] = 2.0
    singleton[1] = 3.0
    singleton[2] = 2.0
    singleton[3] = 1.0

    selected, energies = select_top_singleton_nodes(
        node_ids=node_ids,
        clean_output=clean,
        singleton_outputs=singleton,
        selected_count=2,
    )

    assert selected == ("n1", "n0")
    assert energies == pytest.approx((4.0, 9.0, 4.0, 1.0))


def test_induced_edges_preserve_full_universe_order() -> None:
    edges = (
        ("n0", "n1"),
        ("n0", "n2"),
        ("n1", "n2"),
        ("n2", "n3"),
    )
    selected = ("n2", "n0", "n1")

    assert induced_edge_subsequence(edges, selected) == edges[:3]
    with pytest.raises(ValueError, match="unique"):
        induced_edge_subsequence(edges, ("n0", "n0"))


def test_token_rows_requires_integer_rank_two_inputs() -> None:
    assert token_rows(torch.tensor([[1, 2], [3, 4]]), label="rows") == (
        (1, 2),
        (3, 4),
    )
    with pytest.raises(ValueError):
        token_rows(torch.ones(2, 2, dtype=torch.float32), label="rows")


def test_live_acquisition_requires_exact_frozen_request_before_model_work() -> None:
    request = split_request(21, "confirmation")
    _validated_request(request)

    with pytest.raises(ValueError, match="exact frozen protocol"):
        _validated_request(replace(request, clean_seed=62_034))
