"""End-to-end bidirectional dynamic-graph Mamba classifier."""

from typing import Optional

import torch
from torch import nn

from .block import BidirectionalDynamicGraphBlock
from .config import AdjMambaConfig
from .pooling import AttentionPool, pool_features


class AdjMamba(nn.Module):
    """Bidirectional dynamic-graph Mamba classifier.

    Args:
        config: Model and architecture settings.

    Input:
        ``x`` with shape ``[batch, channels, time, input_dim]``.

    Output:
        Class logits with shape ``[batch, num_classes]``. Set
        ``softmax_output=True`` or pass ``return_probabilities=True`` to return
        probabilities instead.
    """

    def __init__(self, config: AdjMambaConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(
            config.input_dim,
            config.hidden_dim,
            bias=config.bias,
        )
        self.blocks = nn.ModuleList(
            BidirectionalDynamicGraphBlock(config) for _ in range(config.num_layers)
        )
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.time_pool = (
            AttentionPool(config.hidden_dim)
            if config.time_pooling == "attention"
            else None
        )
        self.channel_pool = (
            AttentionPool(config.hidden_dim)
            if config.channel_pooling == "attention"
            else None
        )
        self.classifier = nn.Linear(
            config.hidden_dim,
            config.num_classes,
            bias=config.bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_probabilities: Optional[bool] = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("input must have shape [batch, channels, time, features]")
        if x.size(1) != self.config.num_channels:
            raise ValueError(
                f"expected {self.config.num_channels} channels, got {x.size(1)}"
            )
        if x.size(2) == 0:
            raise ValueError("time dimension must be non-empty")
        if x.size(3) != self.config.input_dim:
            raise ValueError(
                f"expected {self.config.input_dim} features, got {x.size(3)}"
            )

        hidden = self.input_proj(x)
        adjacency = None
        for block in self.blocks:
            block_adjacency = None if self.config.recompute_graph else adjacency
            hidden, adjacency = block(hidden, block_adjacency)

        hidden = self.norm(hidden)
        hidden = pool_features(
            hidden,
            self.config.time_pooling,
            2,
            self.time_pool,
        )
        hidden = pool_features(
            hidden,
            self.config.channel_pooling,
            1,
            self.channel_pool,
        )
        logits = self.classifier(hidden)
        use_softmax = (
            self.config.softmax_output
            if return_probabilities is None
            else return_probabilities
        )
        return torch.softmax(logits, dim=-1) if use_softmax else logits
