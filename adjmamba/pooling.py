"""Temporal and channel pooling utilities."""

from typing import Optional

import torch
from torch import nn

from .config import PoolingMode


class AttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=dim)
        return (x * weights).sum(dim=dim)


def pool_features(
    x: torch.Tensor,
    mode: PoolingMode,
    dim: int,
    attention_pool: Optional[AttentionPool],
) -> torch.Tensor:
    if mode == "mean":
        return x.mean(dim=dim)
    if mode == "max":
        return x.max(dim=dim).values
    if attention_pool is None:
        raise RuntimeError("attention pooling module is not initialized")
    return attention_pool(x, dim)


if __name__ == "__main__":
    sample = torch.randn(1, 3, 4)
    print(pool_features(sample, "mean", 1, None).shape)
