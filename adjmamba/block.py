"""Bidirectional dynamic-graph Mamba block."""

from typing import Optional, Tuple

import torch
from torch import nn

from .config import AdjMambaConfig
from .graph import DynamicGraphConstructor
from .mamba import ConditionalMambaBlock


class BidirectionalDynamicGraphBlock(nn.Module):
    """One bidirectional graph-conditioned Mamba block."""

    def __init__(self, config: AdjMambaConfig) -> None:
        super().__init__()
        self.config = config
        # 先对输入做归一化，降低动态图构造和后续序列建模的数值波动。
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.graph_constructor = DynamicGraphConstructor(config.hidden_dim)
        # 将每个时间步的通道间邻接矩阵压缩为图条件向量。
        self.graph_proj = nn.Linear(
            config.num_channels * config.num_channels,
            config.graph_dim,
            bias=config.bias,
        )
        self.graph_norm = nn.LayerNorm(config.graph_dim)

        mamba_kwargs = {
            "d_state": config.d_state,
            "dt_rank": config.dt_rank,
            "bias": config.bias,
            "dt_scale": config.dt_scale,
            "dt_init": config.dt_init,
            "dt_min": config.dt_min,
            "dt_max": config.dt_max,
            "dt_init_floor": config.dt_init_floor,
            "pscan": config.pscan,
        }
        self.graph_mamba_forward = ConditionalMambaBlock(
            config.graph_dim,
            config.graph_d_inner,
            **mamba_kwargs,
        )
        self.graph_mamba_backward = ConditionalMambaBlock(
            config.graph_dim,
            config.graph_d_inner,
            **mamba_kwargs,
        )
        self.signal_mamba_forward = ConditionalMambaBlock(
            config.hidden_dim,
            config.d_inner,
            condition_dim=config.graph_dim,
            **mamba_kwargs,
        )
        self.signal_mamba_backward = ConditionalMambaBlock(
            config.hidden_dim,
            config.d_inner,
            condition_dim=config.graph_dim,
            **mamba_kwargs,
        )
        self.out_proj = nn.Linear(
            2 * config.hidden_dim,
            config.hidden_dim,
            bias=config.bias,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, T, H]，其中 H=hidden_dim。
        normalized = self.norm(x)
        if adjacency is None:
            # 未提供邻接矩阵时，根据当前输入动态生成图结构。
            adjacency = self.graph_constructor(normalized)
        # adjacency: [B, T, C, C]。

        batch, channels, time, _ = normalized.shape
        # [B, T, C, C] -> [B, T, graph_dim]。
        graph_input = self.graph_proj(adjacency.reshape(batch, time, -1))

        graph_input = self.graph_norm(graph_input)

        # 图序列分别按正向和反向建模；反向结果再恢复到原时间顺序。
        graph_forward = self.graph_mamba_forward(graph_input)
        graph_backward = self.graph_mamba_backward(graph_input.flip(1)).flip(1)
        # 两个方向的输出均为 [B, T, graph_dim]。
        # 图 Mamba 采用残差连接，保留原始图条件信息。
        graph_forward = graph_input + graph_forward
        graph_backward = graph_input + graph_backward

        # [B, C, T, H] -> [B*C, T, H]，将每个通道视为独立序列。
        signal = normalized.reshape(batch * channels, time, -1)
        # [B, T, graph_dim] -> [B*C, T, graph_dim]。
        condition_forward = (
            graph_forward.unsqueeze(1).expand(batch, channels, time, -1)
        ).reshape(batch * channels, time, -1)

        # 正向信号 Mamba 使用正向图序列作为条件。
        delta_forward = self.signal_mamba_forward(
            signal,
            condition_forward,
        ).reshape(batch, channels, time, -1)
        # [B*C, T, H] -> [B, C, T, H]。
        if self.config.use_direction_residual:
            # 可选的方向内残差：将归一化后的输入加回当前方向输出。
            forward_output = normalized + delta_forward
        else:
            forward_output = delta_forward

        # 反向处理信号和图条件，并在输出后恢复原时间顺序。
        reversed_signal = signal.flip(1)
        condition_backward = (
            graph_backward.unsqueeze(1).expand(batch, channels, time, -1)
            .flip(2)
            .reshape(batch * channels, time, -1)
        )
        delta_backward = self.signal_mamba_backward(
            reversed_signal,
            condition_backward,
        ).flip(1).reshape(batch, channels, time, -1)
        # 反向结果恢复后形状为 [B, C, T, H]。
        if self.config.use_direction_residual:
            backward_output = normalized + delta_backward
        else:
            backward_output = delta_backward

        # [B, C, T, H] × 2 -> [B, C, T, 2H] -> [B, C, T, H]。
        fused = torch.cat((forward_output, backward_output), dim=-1)
        output = x + self.dropout(self.out_proj(fused))
        return output, adjacency


if __name__ == "__main__":
    config = AdjMambaConfig(
        input_dim=2,
        num_channels=2,
        num_classes=2,
        hidden_dim=4,
        graph_dim=4,
        d_inner=4,
        graph_d_inner=4,
        d_state=2,
        dt_rank=2,
    )
    block = BidirectionalDynamicGraphBlock(config)
    sample = torch.randn(1, config.num_channels, 3, config.hidden_dim)
    output, adjacency = block(sample)
    print(output.shape, adjacency.shape)
