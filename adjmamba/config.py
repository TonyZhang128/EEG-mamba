"""Configuration for the bidirectional dynamic-graph Mamba model."""

from dataclasses import dataclass
from typing import Literal, Optional


PoolingMode = Literal["mean", "max", "attention"]


@dataclass
class AdjMambaConfig:
    input_dim: int
    num_channels: int
    num_classes: int
    hidden_dim: int = 32
    num_layers: int = 2
    graph_dim: int = 32
    d_inner: Optional[int] = None
    graph_d_inner: Optional[int] = None
    d_state: int = 16
    dt_rank: int = 8
    dt_scale: float = 1.0
    dt_init: Literal["constant", "random"] = "random"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    bias: bool = False
    dropout: float = 0.0
    pscan: bool = True
    recompute_graph: bool = True
    use_direction_residual: bool = True
    time_pooling: PoolingMode = "mean"
    channel_pooling: PoolingMode = "mean"
    softmax_output: bool = False

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.hidden_dim <= 0 or self.graph_dim <= 0:
            raise ValueError("hidden_dim and graph_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.d_state <= 0 or self.dt_rank <= 0:
            raise ValueError("d_state and dt_rank must be positive")
        if self.dt_scale <= 0:
            raise ValueError("dt_scale must be positive")
        if not 0.0 < self.dt_min < self.dt_max:
            raise ValueError("dt_min and dt_max must be positive and dt_min < dt_max")
        if self.dt_init_floor <= 0:
            raise ValueError("dt_init_floor must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.d_inner is None:
            self.d_inner = 2 * self.hidden_dim
        if self.graph_d_inner is None:
            self.graph_d_inner = 2 * self.graph_dim
        if self.d_inner <= 0 or self.graph_d_inner <= 0:
            raise ValueError("d_inner and graph_d_inner must be positive")
        for mode in (self.time_pooling, self.channel_pooling):
            if mode not in ("mean", "max", "attention"):
                raise ValueError(f"unsupported pooling mode: {mode}")


if __name__ == "__main__":
    config = AdjMambaConfig(input_dim=2, num_channels=2, num_classes=2)
    print(config)
