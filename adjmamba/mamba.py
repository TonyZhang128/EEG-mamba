"""Conditional Mamba block with graph-state modulated time steps."""

import math
from typing import Literal, Optional

import torch
from torch import nn
import torch.nn.functional as F

from .scan import parallel_scan


class ConditionalMambaBlock(nn.Module):
    """A Mamba block whose time step can optionally be graph modulated."""

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        d_state: int,
        dt_rank: int,
        *,
        condition_dim: Optional[int] = None,
        bias: bool = False,
        dt_scale: float = 1.0,
        dt_init: Literal["constant", "random"] = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        pscan: bool = True,
        condition_alpha: float =0.25
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_inner <= 0 or d_state <= 0 or dt_rank <= 0:
            raise ValueError("Mamba dimensions must be positive")

        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.pscan = pscan

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=bias)
        self.x_proj = nn.Linear(
            d_inner,
            dt_rank + 2 * d_state + d_inner,
            bias=False,
        )
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.condition_proj = (
            nn.Linear(condition_dim, d_inner, bias=False)
            if condition_dim is not None
            else None
        )
        self.condition_norm = (
            nn.LayerNorm(condition_dim)
            if condition_dim is not None
            else None
        )
        self.condition_alpha = condition_alpha
        state_decay = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(state_decay).repeat(d_inner, 1))

        dt_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_std, dt_std)
        else:
            raise ValueError(f"unsupported dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))

        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus)

        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Mamba input must have shape [batch, time, dim]")
        if self.condition_proj is not None and condition is None:
            raise ValueError("condition is required for this Mamba block")
        if self.condition_proj is None and condition is not None:
            raise ValueError("condition was provided to an unconditional Mamba block")

        # x: [B, T, d_model] -> projected: [B, T, 2*d_inner]。
        projected = self.in_proj(x)

        # signal 和 gate 均为 [B, T, d_inner]。
        signal, gate = projected.chunk(2, dim=-1)
        signal = F.silu(signal)

        # x_proj 输出 [B, T, dt_rank + 2*d_state + d_inner]，
        # 再分解为 delta_rank、B、C、D。
        delta_rank, b_seq, c_seq, skip = torch.split(
            self.x_proj(signal),
            (self.dt_rank, self.d_state, self.d_state, self.d_inner),
            dim=-1,
        )
        # [B, T, dt_rank] -> [B, T, d_inner]。
        delta = self.dt_proj(delta_rank)
        if condition is not None:
            # condition: [B, T, condition_dim]。
            expected_condition = (
                signal.size(0),
                signal.size(1),
                self.condition_proj.in_features,
            )
            if tuple(condition.shape) != expected_condition:
                raise ValueError(
                    f"condition must have shape {expected_condition}, "
                    f"got {tuple(condition.shape)}"
                )
            # 先归一化条件特征，使 tanh 处于更敏感的输入区间。
            normalized_condition = self.condition_norm(condition)
            # 条件投影为 [B, T, d_inner]，仅调制 delta 的尺度。
            condition_scale = torch.tanh(
                self.condition_proj(normalized_condition)
            )
            delta = delta * (1.0 + self.condition_alpha * condition_scale)
        # delta、B、C、D 均按 [B, T, ...] 参与状态空间扫描。
        delta = F.softplus(delta)

        # selective scan 输出 [B, T, d_inner]。
        output = self._selective_scan(
            signal,
            delta,
            -torch.exp(self.A_log.float()),
            b_seq,
            c_seq,
            skip,
        )
        output = output * F.silu(gate)
        # 门控后经输出投影：[B, T, d_inner] -> [B, T, d_model]。
        return self.out_proj(output)



    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        a: torch.Tensor,
        b_seq: torch.Tensor,
        c_seq: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        delta_a = torch.exp(delta.unsqueeze(-1) * a)
        delta_b = delta.unsqueeze(-1) * b_seq.unsqueeze(2)
        bx = delta_b * x.unsqueeze(-1)

        if self.pscan:
            _, states = parallel_scan(delta_a, bx)
        else:
            states = []
            state = torch.zeros(
                x.size(0),
                self.d_inner,
                self.d_state,
                device=x.device,
                dtype=x.dtype,
            )
            for index in range(x.size(1)):
                state = delta_a[:, index] * state + bx[:, index]
                states.append(state)
            states = torch.stack(states, dim=1)

        return (states * c_seq.unsqueeze(2)).sum(dim=-1) + skip * x


if __name__ == "__main__":
    block = ConditionalMambaBlock(
        d_model=4,
        d_inner=4,
        d_state=2,
        dt_rank=2,
    )

    block_with_condition = ConditionalMambaBlock(
        d_model=4,
        d_inner=4,
        d_state=2,
        dt_rank=2,
        condition_dim=3,
    )


    with torch.no_grad():

        sample = torch.randn(2, 3, 4)

        graph =torch.abs(torch.matmul(sample,sample.transpose(-1,-2)))
        graph = graph/torch.sum(graph, dim=-1, keepdim=True)

        y_base = block(sample)
        y_zero = block_with_condition(sample, torch.zeros_like(graph))
        y_cond = block_with_condition(sample, graph)

        print("zero-condition difference:",
            torch.var(y_base - y_zero).item())

        print("condition effect:",
            torch.var(y_zero - y_cond).item())
