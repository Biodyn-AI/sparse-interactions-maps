"""Exhaustively verify the paper's elementary examples, using exact integers.

These checks support illustrative algebra; they are not new transformer data.
"""

from __future__ import annotations

import itertools
import json


def chain_response(mask: tuple[int, ...]) -> int:
    """Run the serial recurrence with recipient zero and donor one."""
    value = 0
    for replace in mask:
        value = 1 if replace else value
    return value


def shared_readout(mask: tuple[int, ...], weights: tuple[int, ...]) -> int:
    return sum(weight * replace for weight, replace in zip(weights, mask, strict=True)) ** 2


def run_checks() -> dict[str, object]:
    mask_count = 0
    pair_count = 0
    for width in (3, 4, 8, 12):
        for mask in itertools.product((0, 1), repeat=width):
            assert chain_response(mask) == int(any(mask))
            mask_count += 1
        empty = (0,) * width
        for left, right in itertools.combinations(range(width), 2):
            left_mask = tuple(int(index == left) for index in range(width))
            right_mask = tuple(int(index == right) for index in range(width))
            both_mask = tuple(int(index in (left, right)) for index in range(width))
            interaction = (
                chain_response(both_mask) - chain_response(left_mask)
                - chain_response(right_mask) + chain_response(empty)
            )
            assert interaction == -1
            # A background donor replacement saturates the relay: all four
            # outcomes become one, so the conditional second difference is zero.
            other = next(index for index in range(width) if index not in (left, right))
            outcomes = []
            for left_bit, right_bit in ((0, 0), (1, 0), (0, 1), (1, 1)):
                mask = [0] * width
                mask[other] = 1
                mask[left], mask[right] = left_bit, right_bit
                outcomes.append(chain_response(tuple(mask)))
            assert outcomes == [1, 1, 1, 1]
            pair_count += 1

    weights = (4, 2, 1, 0, 0, 0, 0, 0)
    width = len(weights)
    quadratic_pair_context_count = 0
    for left, right in itertools.combinations(range(width), 2):
        other = [index for index in range(width) if index not in (left, right)]
        for background in itertools.product((0, 1), repeat=len(other)):
            base = [0] * width
            for index, bit in zip(other, background, strict=True):
                base[index] = bit
            outcomes = []
            for left_bit, right_bit in ((0, 0), (1, 0), (0, 1), (1, 1)):
                mask = base.copy()
                mask[left], mask[right] = left_bit, right_bit
                outcomes.append(shared_readout(tuple(mask), weights))
            interaction = outcomes[3] - outcomes[1] - outcomes[2] + outcomes[0]
            assert interaction == 2 * weights[left] * weights[right]
            assert interaction**2 == (2 * weights[left] ** 2) * (2 * weights[right] ** 2)
            quadratic_pair_context_count += 1
    return {
        "status": "pass",
        "scientific_status": "illustrative_exact_checks_not_a_new_transformer_experiment",
        "chain_widths": [3, 4, 8, 12],
        "chain_masks_checked": mask_count,
        "chain_pairs_checked": pair_count,
        "quadratic_pair_contexts_checked": quadratic_pair_context_count,
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
