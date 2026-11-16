from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch.nn as nn
from torch import Tensor

from utils.geometry_target import build_geometry_target


@dataclass
class NIGFEncoding:


    starts: List[Tensor]
    contexts: List[Tensor]
    geometry_endpoints: Optional[List[Tensor]]


class NuisanceInvariantGeometryFlow(nn.Module):


    def __init__(self, *, enabled: bool, geometry_target: str) -> None:
        super().__init__()
        if geometry_target not in {"mask_only", "mask_distance"}:
            raise ValueError("geometry_target must be mask_only or mask_distance")
        self.enabled = bool(enabled)
        self.geometry_target = geometry_target if self.enabled else "mask_only"

    def prepare_geometry(
        self,
        mask: Tensor,
        geometry: Optional[Tensor] = None,
    ) -> Tensor:

        if geometry is None:
            return build_geometry_target(mask, self.geometry_target)
        geometry = geometry.float()
        if geometry.ndim != 4 or geometry.shape[1] != 2:
            raise ValueError("geometry must have shape [B,2,H,W]")
        if self.geometry_target == "mask_only":
            geometry = geometry.clone()
            geometry[:, 1].zero_()
        return geometry

    def orbit_size(self, requested_size: int) -> int:

        if requested_size < 1:
            raise ValueError("orbit size must be at least one")
        return requested_size if self.enabled else 1

    def build_orbits(
        self,
        orbit_generator: nn.Module,
        image_a: Tensor,
        image_b: Tensor,
        *,
        orbit_size: int,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:

        effective_size = self.orbit_size(int(orbit_size))
        return orbit_generator(
            image_a,
            image_b,
            orbit_size=effective_size,
        )

    def forward(
        self,
        features_a: Sequence[Tensor],
        features_b: Sequence[Tensor],
        *,
        evidence_encoder: nn.Module,
        geometry_encoder: nn.Module,
        geometry: Optional[Tensor] = None,
    ) -> NIGFEncoding:

        starts, contexts = evidence_encoder(features_a, features_b)
        geometry_endpoints: Optional[List[Tensor]] = None
        if geometry is not None:
            spatial_sizes = [state.shape[-2:] for state in starts]
            geometry_endpoints = geometry_encoder(geometry, spatial_sizes)
        return NIGFEncoding(
            starts=list(starts),
            contexts=list(contexts),
            geometry_endpoints=(
                list(geometry_endpoints) if geometry_endpoints is not None else None
            ),
        )


__all__ = ["NIGFEncoding", "NuisanceInvariantGeometryFlow"]
