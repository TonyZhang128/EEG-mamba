"""Parallel prefix scan used by the selective state-space model."""

from typing import Tuple

import torch


_SCAN_CHUNK_SIZE = 32


def parallel_scan(
    a: torch.Tensor,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return all prefix results of h_t = a_t * h_{t-1} + x_t along dim 1."""
    length = a.size(1)
    if length != x.size(1):
        raise ValueError("a and x must have the same sequence length")

    if length <= _SCAN_CHUNK_SIZE:
        a_prefix = []
        x_prefix = []
        previous_a = None
        previous_x = None
        for current_a, current_x in zip(a.unbind(dim=1), x.unbind(dim=1)):
            if previous_a is None:
                next_a = current_a
                next_x = current_x
            else:
                next_a = current_a * previous_a
                next_x = current_x + current_a * previous_x
            a_prefix.append(next_a)
            x_prefix.append(next_x)
            previous_a = next_a
            previous_x = next_x
        return torch.stack(a_prefix, dim=1), torch.stack(x_prefix, dim=1)

    split = length // 2
    left_a, left_x = parallel_scan(a[:, :split], x[:, :split])
    right_a, right_x = parallel_scan(a[:, split:], x[:, split:])

    left_total_a = left_a[:, -1:].expand_as(right_a)
    left_total_x = left_x[:, -1:].expand_as(right_x)
    combined_right_x = right_x + right_a * left_total_x
    combined_right_a = right_a * left_total_a
    return (
        torch.cat((left_a, combined_right_a), dim=1),
        torch.cat((left_x, combined_right_x), dim=1),
    )


if __name__ == "__main__":
    coefficients = torch.ones(1, 3, 1)
    values = torch.arange(3, dtype=torch.float32).reshape(1, 3, 1)
    _, prefixes = parallel_scan(coefficients, values)
    print(prefixes.flatten())
