from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import torch
from torch import Tensor


def build_geometry_target(
    mask: Tensor,
    target_type: Literal["mask_only", "mask_distance"] = "mask_distance",
) -> Tensor:


    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape [B, 1, H, W] or [B, H, W]")
    if target_type not in {"mask_only", "mask_distance"}:
        raise ValueError("target_type must be mask_only or mask_distance")

    binary_tensor = (mask.detach() > 0.5).to(torch.float32)
    binary_numpy = binary_tensor[:, 0].cpu().numpy().astype(np.uint8)
    distances = []
    for binary in binary_numpy:
        if target_type == "mask_only" or not binary.any():
            distance = np.zeros_like(binary, dtype=np.float32)
        else:
            distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5).astype(np.float32)
            maximum = max(float(distance.max()), 1.0)
            distance = (distance / maximum) * binary.astype(np.float32)
        distances.append(torch.from_numpy(distance))

    distance_tensor = torch.stack(distances, dim=0).unsqueeze(1)
    geometry = torch.cat([binary_tensor.cpu(), distance_tensor], dim=1)
    dtype = mask.dtype if mask.is_floating_point() else torch.float32
    return geometry.to(device=mask.device, dtype=dtype)


class GeometryTargetBuilder:


    def __init__(self, target_type: str = "mask_distance") -> None:
        self.target_type = target_type

    def __call__(self, mask: Tensor) -> Tensor:
        return build_geometry_target(mask, self.target_type)
