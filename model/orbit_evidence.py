from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _groups(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SymmetricEvidenceBlock(nn.Module):


    def __init__(
        self,
        in_channels: int,
        flow_channels: int,
        flow_start: str = "multievidence",
        nigf_enabled: bool = True,
    ) -> None:
        super().__init__()
        if flow_start not in {"multievidence", "absdiff"}:
            raise ValueError("flow_start must be multievidence or absdiff")
        self.in_channels = in_channels
        self.flow_channels = flow_channels
        self.flow_start = flow_start
        self.nigf_enabled = bool(nigf_enabled)
        projected = max(8, min(in_channels // 2, flow_channels))
        self.projected_channels = projected

        self.sum_projector = nn.Conv2d(in_channels, projected, 1, bias=False)
        self.product_projector = nn.Conv2d(in_channels, projected, 1, bias=False)

        self.signed_projector = nn.Sequential(
            nn.Conv2d(in_channels, projected, 1, bias=False),
            nn.GroupNorm(_groups(projected), projected),
            nn.SiLU(),
        )
        evidence_channels = in_channels if flow_start == "absdiff" else in_channels + projected * 3 + 1
        self.shared = nn.Sequential(
            nn.Conv2d(evidence_channels, flow_channels, 1, bias=False),
            nn.GroupNorm(_groups(flow_channels), flow_channels),
            nn.SiLU(),
            nn.Conv2d(flow_channels, flow_channels, 3, padding=1, groups=flow_channels, bias=False),
            nn.GroupNorm(_groups(flow_channels), flow_channels),
            nn.SiLU(),
        )
        self.start_projection = nn.Conv2d(flow_channels, flow_channels, 1)
        self.context_projection = nn.Conv2d(flow_channels, flow_channels, 1)
        nn.init.normal_(self.start_projection.weight, std=1e-3)
        nn.init.zeros_(self.start_projection.bias)
        nn.init.normal_(self.context_projection.weight, std=1e-3)
        nn.init.zeros_(self.context_projection.bias)

    def _flatten(self, tensor: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        if tensor.ndim != 5:
            raise ValueError("features must have shape [B, K, C, H, W]")
        b, k, c, h, w = tensor.shape
        return tensor.reshape(b * k, c, h, w), (b, k)

    def forward(self, feature_a: Tensor, feature_b: Tensor) -> Tuple[Tensor, Tensor]:
        if feature_a.shape != feature_b.shape or feature_a.shape[2] != self.in_channels:
            raise ValueError("A/B feature shapes do not match this evidence block")
        fa, (batch, orbit) = self._flatten(feature_a)
        fb, _ = self._flatten(feature_b)


        with torch.autocast(device_type=fa.device.type, enabled=False):
            fa = fa.float()
            fb = fb.float()
            absolute = torch.abs(fa - fb)
            if self.flow_start == "absdiff":
                evidence = absolute
            elif self.nigf_enabled:
                sum_feature = self.sum_projector(fa + fb)
                product = self.product_projector(fa * fb)
                cosine_gap = 1.0 - F.cosine_similarity(fa, fb, dim=1, eps=1e-6).unsqueeze(1)
                signed_ab = self.signed_projector(fb - fa)
                signed_ba = self.signed_projector(fa - fb)
                signed_symmetric = torch.abs(signed_ab + signed_ba)
                evidence = torch.cat([absolute, sum_feature, product, cosine_gap, signed_symmetric], dim=1)
            else:


                inactive_channels = self.projected_channels * 3 + 1
                inactive = absolute.new_zeros(
                    absolute.shape[0],
                    inactive_channels,
                    absolute.shape[2],
                    absolute.shape[3],
                )
                evidence = torch.cat([absolute, inactive], dim=1)
            shared = self.shared(evidence)
            start = self.start_projection(shared)
            context = self.context_projection(shared)
        _, channels, height, width = start.shape
        shape = (batch, orbit, channels, height, width)
        return start.reshape(shape), context.reshape(shape)


class MultiScaleOrbitEvidenceEncoder(nn.Module):
    def __init__(
        self,
        encoder_channels: Sequence[int],
        flow_channels: Sequence[int],
        flow_start: str = "multievidence",
        nigf_enabled: bool = True,
    ) -> None:
        super().__init__()
        if len(encoder_channels) != 4 or len(flow_channels) != 4:
            raise ValueError("exactly four coarse-to-fine scales are required")
        self.blocks = nn.ModuleList(
            [
                SymmetricEvidenceBlock(
                    in_c,
                    flow_c,
                    flow_start,
                    nigf_enabled=nigf_enabled,
                )
                for in_c, flow_c in zip(encoder_channels, flow_channels)
            ]
        )

    def forward(
        self, features_a: Sequence[Tensor], features_b: Sequence[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        if len(features_a) != 4 or len(features_b) != 4:
            raise ValueError("exactly four feature scales are required")
        starts, contexts = [], []
        for block, feature_a, feature_b in zip(self.blocks, features_a, features_b):
            start, context = block(feature_a, feature_b)
            starts.append(start)
            contexts.append(context)
        return starts, contexts
