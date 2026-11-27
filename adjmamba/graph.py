"""Dynamic channel graph construction."""

import torch
from torch import nn


class DynamicGraphConstructor(nn.Module):
    """Construct a dense channel adjacency matrix for every time step."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x=x.permute(0, 2, 1, 3)  # [batch, time, channels, features]
        raw = torch.matmul(
            x,
            x.transpose(-1, -2),
        )  # [batch, time, channels, channels]
        denominator = raw.abs().sum(dim=-1, keepdim=True) + self.eps
        return raw / denominator

if __name__ == "__main__":
    constructor = DynamicGraphConstructor(dim=8)
    sample = torch.randn(1, 2, 3, 8)
    print(constructor(sample).shape)
