from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.flow_rwkv import SinusoidalFlowTimeEmbedding


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, tensor: Tensor) -> Tensor:
        return self.activation(tensor + self.network(tensor))


class _ConvVectorBlock(nn.Module):
    def __init__(self, channels: int, max_time_delta: float = 0.1) -> None:
        super().__init__()
        self.max_time_delta = max_time_delta
        self.input_projection = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.time_encoder = SinusoidalFlowTimeEmbedding(64)
        self.time_mlp = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, channels * 2))
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)
        self.blocks = nn.Sequential(_ResBlock(channels), _ResBlock(channels))
        self.velocity_projection = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.velocity_projection.weight)
        nn.init.zeros_(self.velocity_projection.bias)

    def forward(self, state: Tensor, context: Tensor, time: Tensor) -> Tensor:
        feature = self.input_projection(torch.cat([state, context], dim=1))
        scale, shift = self.time_mlp(self.time_encoder(time).to(feature.dtype)).chunk(2, dim=-1)
        feature = feature * (1 + self.max_time_delta * torch.tanh(scale)[:, :, None, None])
        feature = feature + self.max_time_delta * torch.tanh(shift)[:, :, None, None]
        return self.velocity_projection(self.blocks(feature))


class MultiScaleConvVectorField(nn.Module):
    def __init__(self, flow_channels: Sequence[int], max_time_delta: float = 0.1) -> None:
        super().__init__()
        if len(flow_channels) != 4:
            raise ValueError("exactly four scales are required")
        self.blocks = nn.ModuleList([_ConvVectorBlock(ch, max_time_delta) for ch in flow_channels])
        self.guidance_projections = nn.ModuleList(
            [nn.Conv2d(flow_channels[i - 1], flow_channels[i], 1, bias=False) for i in range(1, 4)]
        )

    def forward(
        self, states: Sequence[Tensor], contexts: Sequence[Tensor], time: Tensor
    ) -> List[Tensor]:
        if len(states) != 4 or len(contexts) != 4:
            raise ValueError("exactly four scales are required")
        velocities = []
        for index, (block, state, context) in enumerate(zip(self.blocks, states, contexts)):
            if index:
                guidance = F.interpolate(
                    velocities[-1], size=state.shape[-2:], mode="bilinear", align_corners=False
                )
                state = state + self.guidance_projections[index - 1](guidance)
            velocities.append(block(state, context, time))
        return velocities
