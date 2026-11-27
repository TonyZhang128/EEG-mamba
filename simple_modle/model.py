"""Minimal single-direction Mamba classifier described by this folder's readme."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
from torch import nn
import torch.nn.functional as F


# Support both ``python simple_modle/train.py`` and ``python -m simple_modle.train``.
EEGMAMBA_ROOT = Path(__file__).resolve().parents[1]
if str(EEGMAMBA_ROOT) not in sys.path:
    sys.path.insert(0, str(EEGMAMBA_ROOT))

from adjmamba.scan import parallel_scan


ModelVariant = Literal["plain", "channel", "two_level"]


@dataclass
class SimpleMambaConfig:
    """Architecture settings for the simple single-subject classifier."""

    input_dim: int = 5
    num_channels: int = 32
    num_classes: int = 4
    hidden_dim: int = 32
    d_inner: Optional[int] = None
    d_state: int = 8
    dt_rank: int = 4
    d_conv: int = 4
    dropout: float = 0.1
    bias: bool = False
    condition_alpha: float = 0.1
    condition_mode: ModelVariant = "two_level"

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.d_inner is None:
            self.d_inner = 2 * self.hidden_dim
        if (
            self.d_inner <= 0
            or self.d_state <= 0
            or self.dt_rank <= 0
            or self.d_conv <= 0
        ):
            raise ValueError("Mamba dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.condition_mode not in ("plain", "channel", "two_level"):
            raise ValueError(f"unsupported condition mode: {self.condition_mode}")


class SimpleMambaScan(nn.Module):
    """Shared-parameter selective scan along time for every EEG channel."""

    def __init__(self, config: SimpleMambaConfig) -> None:
        super().__init__()
        self.d_inner = int(config.d_inner)
        self.d_state = config.d_state
        self.dt_rank = config.dt_rank
        self.d_conv = config.d_conv
        self.condition_mode = config.condition_mode

        self.in_proj = nn.Linear(
            config.hidden_dim,
            2 * self.d_inner,
            bias=config.bias,
        )
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * self.d_state + self.d_inner,
            bias=False,
        )
        # Causal depthwise temporal convolution used by standard Mamba blocks.
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=True,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(
            self.d_inner,
            config.hidden_dim,
            bias=config.bias,
        )
        self.skip = nn.Parameter(torch.ones(self.d_inner))
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32))
            .repeat(self.d_inner, 1)
        )

        # Standard Mamba delta initialization, matching adjmamba/mamba.py.
        dt_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_std, dt_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus)

        self.condition_norm_c: Optional[nn.LayerNorm] = None
        self.condition_proj_c: Optional[nn.Linear] = None
        self.condition_norm_g: Optional[nn.LayerNorm] = None
        self.condition_proj_g: Optional[nn.Linear] = None
        self.alpha_c: Optional[nn.Parameter] = None
        self.alpha_g: Optional[nn.Parameter] = None

        if self.condition_mode in ("channel", "two_level"):
            self.condition_norm_c = nn.LayerNorm(config.hidden_dim)
            self.condition_proj_c = nn.Linear(
                config.hidden_dim,
                self.d_inner,
                bias=False,
            )
            self.alpha_c = nn.Parameter(
                torch.tensor(config.condition_alpha, dtype=torch.float32)
            )
        if self.condition_mode == "two_level":
            self.condition_norm_g = nn.LayerNorm(config.hidden_dim)
            self.condition_proj_g = nn.Linear(
                config.hidden_dim,
                self.d_inner,
                bias=False,
            )
            self.alpha_g = nn.Parameter(
                torch.tensor(config.condition_alpha, dtype=torch.float32)
            )

    def forward(
        self,
        x: torch.Tensor,
        condition_c: Optional[torch.Tensor] = None,
        condition_g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scan ``[batch, time, hidden]`` with optional global conditions."""
        if x.ndim != 3:
            raise ValueError("Mamba input must have shape [batch, time, hidden]")
        expected_batch = x.size(0)
        expected_dim = self.in_proj.in_features
        if x.size(2) != expected_dim:
            raise ValueError(
                f"expected hidden dimension {expected_dim}, got {x.size(2)}"
            )
        if self.condition_mode == "plain" and (
            condition_c is not None or condition_g is not None
        ):
            raise ValueError("plain Mamba does not accept condition tensors")
        if self.condition_mode in ("channel", "two_level"):
            if condition_c is None or condition_c.shape != (expected_batch, expected_dim):
                raise ValueError(
                    f"condition_c must have shape {(expected_batch, expected_dim)}"
                )
        if self.condition_mode == "two_level":
            if condition_g is None or condition_g.shape != (expected_batch, expected_dim):
                raise ValueError(
                    f"condition_g must have shape {(expected_batch, expected_dim)}"
                )

        projected = self.in_proj(x)
        signal, gate = projected.chunk(2, dim=-1)
        signal = self.conv1d(signal.transpose(1, 2))
        signal = signal[..., : x.size(1)].transpose(1, 2)
        signal = F.silu(signal)
        delta_rank, b_seq, c_seq, skip = torch.split(
            self.x_proj(signal),
            (self.dt_rank, self.d_state, self.d_state, self.d_inner),
            dim=-1,
        )

        delta_0 = F.softplus(self.dt_proj(delta_rank))
        condition_scale = torch.zeros_like(delta_0)
        if self.condition_mode in ("channel", "two_level"):
            condition_scale = condition_scale + self._condition_scale_c(condition_c)[:, None, :]
        if self.condition_mode == "two_level":
            condition_scale = condition_scale + self._condition_scale_g(condition_g)[:, None, :]
        delta = delta_0 * torch.exp(torch.clamp(condition_scale, -2.0, 2.0))

        states = self._selective_scan(
            signal,
            delta,
            -torch.exp(self.A_log.to(dtype=signal.dtype)),
            b_seq,
            c_seq,
        )
        output = (states * c_seq.unsqueeze(2)).sum(dim=-1)
        output = (output + self.skip.to(dtype=output.dtype) * signal) * F.silu(gate)
        return self.out_proj(output)

    def _condition_scale_c(self, condition_c: torch.Tensor) -> torch.Tensor:
        assert self.condition_norm_c is not None
        assert self.condition_proj_c is not None
        assert self.alpha_c is not None
        normalized = self.condition_norm_c(condition_c)
        return self.alpha_c * torch.tanh(self.condition_proj_c(normalized))

    def _condition_scale_g(self, condition_g: torch.Tensor) -> torch.Tensor:
        assert self.condition_norm_g is not None
        assert self.condition_proj_g is not None
        assert self.alpha_g is not None
        normalized = self.condition_norm_g(condition_g)
        return self.alpha_g * torch.tanh(self.condition_proj_g(normalized))

    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        a: torch.Tensor,
        b_seq: torch.Tensor,
        c_seq: torch.Tensor,
    ) -> torch.Tensor:
        delta_a = torch.exp(delta.unsqueeze(-1) * a)
        delta_b = delta.unsqueeze(-1) * b_seq.unsqueeze(2)
        bx = delta_b * x.unsqueeze(-1)
        _, states = parallel_scan(delta_a, bx)
        return states


class SimpleMambaClassifier(nn.Module):
    """Projection -> optional two-level delta condition -> Mamba -> pooling -> MLP."""

    def __init__(self, config: SimpleMambaConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(
            config.input_dim,
            config.hidden_dim,
            bias=config.bias,
        )
        self.mamba = SimpleMambaScan(config)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for ``[batch, channels, time, features]`` input."""
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

        batch_size, num_channels, time_steps, _ = x.shape
        hidden = self.input_proj(x)

        # Global descriptions come from projected input, before selective scan.
        global_channel = hidden.mean(dim=2)
        global_sample = global_channel.mean(dim=1)
        flat_hidden = hidden.reshape(batch_size * num_channels, time_steps, -1)

        condition_c = None
        condition_g = None
        if self.config.condition_mode in ("channel", "two_level"):
            condition_c = global_channel.reshape(batch_size * num_channels, -1)
        if self.config.condition_mode == "two_level":
            condition_g = (
                global_sample[:, None, :]
                .expand(batch_size, num_channels, -1)
                .reshape(batch_size * num_channels, -1)
            )

        scanned = self.mamba(flat_hidden, condition_c, condition_g)
        hidden = hidden + scanned.reshape(batch_size, num_channels, time_steps, -1)
        hidden = self.norm(hidden)

        # Mean over time, then mean over channels.
        pooled = hidden.mean(dim=2).mean(dim=1)
        return self.classifier(pooled)
