from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _groups(channels: int) -> int:
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _GeometryProjection(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        hidden = max(16, out_channels // 2)
        self.network = nn.Sequential(
            nn.Conv2d(2, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, out_channels, 3, padding=1, bias=False),
        )

    def forward(self, geometry: Tensor, size: Tuple[int, int]) -> Tensor:
        resized = F.interpolate(geometry, size=size, mode="bilinear", align_corners=False)
        return self.network(resized)


class MultiScaleGeometryEncoder(nn.Module):


    def __init__(self, flow_channels: Sequence[int]) -> None:
        super().__init__()
        if len(flow_channels) != 4:
            raise ValueError("exactly four scales are required")
        self.projections = nn.ModuleList([_GeometryProjection(channels) for channels in flow_channels])

    def forward(self, geometry: Tensor, spatial_sizes: Sequence[Tuple[int, int]]) -> List[Tensor]:
        if geometry.ndim != 4 or geometry.shape[1] != 2:
            raise ValueError("geometry must have shape [B, 2, H, W]")
        if len(spatial_sizes) != 4:
            raise ValueError("exactly four spatial sizes are required")
        return [projection(geometry, tuple(size)) for projection, size in zip(self.projections, spatial_sizes)]


class _DecodeRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return tensor + self.network(tensor)


class GeometryDecoder(nn.Module):


    def __init__(self, flow_channels: Sequence[int], output_size: Optional[Tuple[int, int]] = None) -> None:
        super().__init__()
        if len(flow_channels) != 4:
            raise ValueError("exactly four scales are required")
        hidden = max(32, min(96, max(flow_channels)))
        self.output_size = output_size
        self.lateral = nn.ModuleList([nn.Conv2d(channels, hidden, 1, bias=False) for channels in flow_channels])
        self.refine = nn.ModuleList([_DecodeRefinement(hidden) for _ in flow_channels])
        self.final_refine = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.SiLU(),
        )
        self.mask_head = nn.Conv2d(hidden, 1, 1)
        self.distance_head = nn.Conv2d(hidden, 1, 1)

    def forward(
        self, states: Sequence[Tensor], output_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[Tensor, Tensor]:
        if len(states) != 4:
            raise ValueError("exactly four coarse-to-fine states are required")
        fused = self.refine[0](self.lateral[0](states[0]))
        for index in range(1, 4):
            fused = F.interpolate(fused, size=states[index].shape[-2:], mode="bilinear", align_corners=False)
            fused = self.refine[index](fused + self.lateral[index](states[index]))
        target_size = output_size or self.output_size
        if target_size is None:
            target_size = (states[-1].shape[-2] * 2, states[-1].shape[-1] * 2)
        fused = F.interpolate(fused, size=target_size, mode="bilinear", align_corners=False)
        fused = self.final_refine(fused)
        return self.mask_head(fused), self.distance_head(fused)
