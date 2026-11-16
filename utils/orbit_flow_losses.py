from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def dice_loss_with_logits(logits: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    probability = torch.sigmoid(logits)
    reduce_dimensions = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=reduce_dimensions)
    denominator = probability.sum(dim=reduce_dimensions) + target.sum(dim=reduce_dimensions)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def balanced_bce_with_logits(logits: Tensor, target: Tensor, positive_weight: float) -> Tensor:

    pos_weight = logits.new_tensor(float(positive_weight))
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def boundary_bce_with_logits(
    logits: Tensor,
    target: Tensor,
    positive_weight: float,
    kernel_size: int = 5,
) -> Tensor:

    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("boundary kernel size must be an odd integer >= 3")
    radius = kernel_size // 2
    dilated = F.max_pool2d(target, kernel_size, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - target, kernel_size, stride=1, padding=radius)
    boundary = (dilated - eroded).clamp(0.0, 1.0)
    pixel_loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=logits.new_tensor(float(positive_weight)),
        reduction="none",
    )
    return (pixel_loss * boundary).sum() / boundary.sum().clamp_min(1.0)


def _lovasz_gradient(sorted_target: Tensor) -> Tensor:
    pixels = sorted_target.numel()
    positives = sorted_target.sum()
    intersection = positives - sorted_target.cumsum(0)
    union = positives + (1.0 - sorted_target).cumsum(0)
    gradient = 1.0 - intersection / union.clamp_min(1.0)
    if pixels > 1:
        gradient[1:] = gradient[1:] - gradient[:-1]
    return gradient


def binary_lovasz_hinge(logits: Tensor, target: Tensor) -> Tensor:

    losses = []
    for image_logits, image_target in zip(logits.float(), target.float()):
        flat_logits = image_logits.reshape(-1)
        flat_target = image_target.reshape(-1)
        signs = 2.0 * flat_target - 1.0
        errors = 1.0 - flat_logits * signs
        sorted_errors, permutation = torch.sort(errors, descending=True)
        sorted_target = flat_target[permutation]
        losses.append(torch.dot(F.relu(sorted_errors), _lovasz_gradient(sorted_target)))
    return torch.stack(losses).mean()


class OrbitFlowLoss(nn.Module):
    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_dist: float = 0.2,
        lambda_fm: float = 1.0,
        lambda_orbit: float = 0.3,
        lambda_geo: float = 0.5,
        lambda_quiet: float = 0.05,
        lambda_t0: float = 1.0,
        lambda_boundary: float = 0.0,
        lambda_lovasz: float = 0.0,
        lambda_direct: float = 0.0,
        lambda_clock: float = 0.0,
        seg_pos_weight: float = 1.0,
        boundary_kernel_size: int = 5,
        spatial_clock_max_warp: float = 0.75,
        clock_offset_weight: float = 1.0,
        clock_boundary_weight: float = 0.5,
        clock_smooth_weight: float = 0.05,
        orbit_seg_weight: float = 0.5,
        orbit_variance_weight: float = 0.1,
        orbit_center: str = "geometry_target",
        scale_weights: Sequence[float] = (1.0, 1.0, 0.75, 0.5),
    ) -> None:
        super().__init__()
        if orbit_center not in {"geometry_target", "mean_detach"}:
            raise ValueError("orbit_center must be geometry_target or mean_detach")
        if len(scale_weights) != 4:
            raise ValueError("exactly four scale weights are required")
        if seg_pos_weight <= 0:
            raise ValueError("seg_pos_weight must be positive")
        if boundary_kernel_size < 3 or boundary_kernel_size % 2 == 0:
            raise ValueError("boundary_kernel_size must be an odd integer >= 3")
        if not 0.0 < spatial_clock_max_warp < 1.0:
            raise ValueError("spatial_clock_max_warp must be in (0, 1)")
        if min(clock_offset_weight, clock_boundary_weight, clock_smooth_weight) < 0.0:
            raise ValueError("spatial clock loss weights must be non-negative")
        self.weights = {
            "seg": float(lambda_seg),
            "distance": float(lambda_dist),
            "fm": float(lambda_fm),
            "orbit": float(lambda_orbit),
            "geometry": float(lambda_geo),
            "quiet": float(lambda_quiet),
            "t0": float(lambda_t0),
            "boundary": float(lambda_boundary),
            "lovasz": float(lambda_lovasz),
            "direct": float(lambda_direct),
            "clock": float(lambda_clock),
        }
        self.seg_pos_weight = float(seg_pos_weight)
        self.boundary_kernel_size = int(boundary_kernel_size)
        self.spatial_clock_max_warp = float(spatial_clock_max_warp)
        self.clock_offset_weight = float(clock_offset_weight)
        self.clock_boundary_weight = float(clock_boundary_weight)
        self.clock_smooth_weight = float(clock_smooth_weight)
        self.orbit_seg_weight = float(orbit_seg_weight)
        self.orbit_variance_weight = float(orbit_variance_weight)
        self.orbit_center = orbit_center
        self.scale_weights = tuple(float(weight) for weight in scale_weights)

    def forward(
        self,
        *,
        mask_logits: Tensor,
        distance_logits: Tensor,
        teacher_mask_logits: Tensor,
        teacher_distance_logits: Tensor,
        mask: Tensor,
        distance_target: Tensor,
        predicted_endpoints: Sequence[Tensor],
        geometry_endpoints: Sequence[Tensor],
        predicted_velocities: Optional[Sequence[Tensor]] = None,
        target_velocities: Optional[Sequence[Tensor]] = None,
        t0_mask_logits: Optional[Tensor] = None,
        direct_mask_logits: Optional[Tensor] = None,
        clock_auxiliaries: Optional[Sequence[Mapping[str, Tensor]]] = None,
    ) -> Dict[str, Tensor]:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        batch, orbit = predicted_endpoints[0].shape[:2]
        height, width = mask.shape[-2:]
        mask_orbit = mask[:, None].expand(batch, orbit, 1, height, width).reshape(batch * orbit, 1, height, width)
        distance_orbit = distance_target[:, None].expand(batch, orbit, 1, height, width).reshape(
            batch * orbit, 1, height, width
        )

        logits_by_orbit = mask_logits.reshape(batch, orbit, 1, height, width)
        identity_logits = logits_by_orbit[:, 0]
        loss_seg_identity = balanced_bce_with_logits(
            identity_logits, mask, self.seg_pos_weight
        ) + dice_loss_with_logits(identity_logits, mask)
        loss_seg_all = balanced_bce_with_logits(
            mask_logits, mask_orbit, self.seg_pos_weight
        ) + dice_loss_with_logits(mask_logits, mask_orbit)
        loss_seg = loss_seg_identity + self.orbit_seg_weight * loss_seg_all
        loss_distance = F.smooth_l1_loss(torch.sigmoid(distance_logits), distance_orbit)

        zero = mask_logits.new_zeros(())
        loss_fm = zero
        if predicted_velocities is not None and target_velocities is not None:
            for weight, prediction, target in zip(self.scale_weights, predicted_velocities, target_velocities):
                loss_fm = loss_fm + weight * F.mse_loss(prediction, target)

        loss_orbit_primary = zero
        loss_orbit_variance = zero
        for weight, endpoint, geometry in zip(self.scale_weights, predicted_endpoints, geometry_endpoints):
            mean_endpoint = endpoint.mean(dim=1, keepdim=True)
            if self.orbit_center == "geometry_target":
                center = geometry[:, None].expand_as(endpoint)
            else:
                center = mean_endpoint.detach().expand_as(endpoint)
            loss_orbit_primary = loss_orbit_primary + weight * (endpoint - center).abs().mean()
            loss_orbit_variance = loss_orbit_variance + weight * (endpoint - mean_endpoint).abs().mean()
        loss_orbit = loss_orbit_primary + self.orbit_variance_weight * loss_orbit_variance

        loss_geometry = (
            balanced_bce_with_logits(teacher_mask_logits, mask, self.seg_pos_weight)
            + dice_loss_with_logits(teacher_mask_logits, mask)
            + F.smooth_l1_loss(torch.sigmoid(teacher_distance_logits), distance_target)
        )


        loss_t0 = zero
        if t0_mask_logits is not None:
            loss_t0 = balanced_bce_with_logits(
                t0_mask_logits, mask, self.seg_pos_weight
            ) + dice_loss_with_logits(t0_mask_logits, mask)


        inference_logits = t0_mask_logits if t0_mask_logits is not None else identity_logits
        loss_boundary = boundary_bce_with_logits(
            inference_logits,
            mask,
            self.seg_pos_weight,
            self.boundary_kernel_size,
        )
        loss_lovasz = binary_lovasz_hinge(inference_logits, mask)
        loss_direct = zero
        if direct_mask_logits is not None:
            loss_direct = balanced_bce_with_logits(
                direct_mask_logits, mask, self.seg_pos_weight
            ) + dice_loss_with_logits(direct_mask_logits, mask)


        loss_clock = zero
        if clock_auxiliaries:
            radius = self.boundary_kernel_size // 2
            dilated = F.max_pool2d(
                mask,
                self.boundary_kernel_size,
                stride=1,
                padding=radius,
            )
            eroded = 1.0 - F.max_pool2d(
                1.0 - mask,
                self.boundary_kernel_size,
                stride=1,
                padding=radius,
            )
            boundary = (dilated - eroded).clamp(0.0, 1.0)
            scale_total = 0.0
            for scale_weight, auxiliary in zip(
                self.scale_weights, clock_auxiliaries
            ):
                if not auxiliary:
                    continue
                clock_offset = auxiliary["clock_offset"]
                router_logits = auxiliary["router_logits"]
                spatial_size = clock_offset.shape[-2:]
                scale_mask = F.interpolate(mask, size=spatial_size, mode="nearest")
                scale_distance = F.interpolate(
                    distance_target,
                    size=spatial_size,
                    mode="bilinear",
                    align_corners=False,
                )
                scale_boundary = F.interpolate(
                    boundary, size=spatial_size, mode="nearest"
                )
                clock_target = (
                    self.spatial_clock_max_warp
                    * scale_mask
                    * (2.0 * scale_distance - 1.0)
                )
                clock_target = (
                    clock_target[:, None]
                    .expand(batch, orbit, 1, *spatial_size)
                    .reshape(batch * orbit, 1, *spatial_size)
                )
                route_target = (
                    scale_boundary[:, None]
                    .expand(batch, orbit, 1, *spatial_size)
                    .reshape(batch * orbit, 1, *spatial_size)
                )
                offset_loss = F.smooth_l1_loss(clock_offset, clock_target)
                positives = route_target.sum()
                negatives = route_target.numel() - positives
                route_positive_weight = (negatives / positives.clamp_min(1.0)).clamp(
                    1.0, 20.0
                )
                route_loss = F.binary_cross_entropy_with_logits(
                    router_logits,
                    route_target,
                    pos_weight=route_positive_weight,
                ) + dice_loss_with_logits(router_logits, route_target)
                smooth_loss = (
                    (clock_offset[..., 1:, :] - clock_offset[..., :-1, :])
                    .abs()
                    .mean()
                    + (clock_offset[..., :, 1:] - clock_offset[..., :, :-1])
                    .abs()
                    .mean()
                )
                loss_clock = loss_clock + scale_weight * (
                    self.clock_offset_weight * offset_loss
                    + self.clock_boundary_weight * route_loss
                    + self.clock_smooth_weight * smooth_loss
                )
                scale_total += scale_weight
            if scale_total > 0.0:
                loss_clock = loss_clock / scale_total

        loss_quiet = zero
        if predicted_velocities is not None:
            for weight, velocity in zip(self.scale_weights, predicted_velocities):
                unchanged = 1.0 - F.interpolate(mask, size=velocity.shape[-2:], mode="nearest")
                unchanged = unchanged[:, None]
                numerator = (velocity.abs() * unchanged).sum()
                denominator = (
                    unchanged.sum() * velocity.shape[1] * velocity.shape[2]
                ).clamp_min(1.0)
                loss_quiet = loss_quiet + weight * numerator / denominator

        components = {
            "loss_seg": loss_seg,
            "loss_distance": loss_distance,
            "loss_fm": loss_fm,
            "loss_orbit": loss_orbit,
            "loss_geometry": loss_geometry,
            "loss_quiet": loss_quiet,
            "loss_t0": loss_t0,
            "loss_boundary": loss_boundary,
            "loss_lovasz": loss_lovasz,
            "loss_direct": loss_direct,
            "loss_clock": loss_clock,
        }
        loss_total = (
            self.weights["seg"] * loss_seg
            + self.weights["distance"] * loss_distance
            + self.weights["fm"] * loss_fm
            + self.weights["orbit"] * loss_orbit
            + self.weights["geometry"] * loss_geometry
            + self.weights["quiet"] * loss_quiet
            + self.weights["t0"] * loss_t0
            + self.weights["boundary"] * loss_boundary
            + self.weights["lovasz"] * loss_lovasz
            + self.weights["direct"] * loss_direct
            + self.weights["clock"] * loss_clock
        )
        return {"loss_total": loss_total, **components}
