from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import Tensor

from model.NuisanceOrbitFlowRWKV import NuisanceOrbitFlowRWKV


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def add_architecture_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--model_size", choices=("tiny", "small", "base"), default="base")
    parser.add_argument("--orbit_size", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--orbit_min_ops", type=int, default=2)
    parser.add_argument("--orbit_max_ops", type=int, default=4)
    parser.add_argument("--orbit_strength", choices=("weak", "medium", "strong"), default="medium")
    parser.add_argument("--orbit_types", default="all")
    parser.add_argument(
        "--nigf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "enable nuisance orbits, symmetric multi-evidence and distance-aware "
            "geometry; --no-nigf retains the common mask-flow baseline"
        ),
    )
    parser.add_argument("--flow_start", choices=("absdiff", "multievidence"), default="multievidence")
    parser.add_argument("--transport_mode", choices=("flow", "endpoint"), default="flow")
    parser.add_argument("--vector_field", choices=("conv", "rwkv"), default="rwkv")
    parser.add_argument("--time_conditioning", choices=("none", "film", "decay"), default="decay")
    parser.add_argument("--max_time_delta", type=float, default=0.1)
    parser.add_argument(
        "--spatial_clock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "enable endpoint-preserving pixel-wise Flow clocks and the routed "
            "local boundary velocity expert"
        ),
    )
    parser.add_argument("--spatial_clock_max_warp", type=float, default=0.75)
    parser.add_argument("--spatial_clock_local_scale", type=float, default=0.10)
    parser.add_argument(
        "--flow_time_sampling",
        choices=("uniform", "squared", "zero"),
        default="uniform",
        help=(
            "training-time flow-state sampling: uniform is the baseline, "
            "squared biases samples toward t=0, and zero aligns one-step training at t=0"
        ),
    )
    parser.add_argument("--geometry_target", choices=("mask_only", "mask_distance"), default="mask_distance")
    parser.add_argument("--solver", choices=("euler",), default="euler")
    parser.add_argument("--solver_steps", type=int, choices=(1, 2, 4, 6, 8), default=4)
    parser.add_argument(
        "--direct_fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="add a directly supervised evidence decoder as a residual to flow logits",
    )
    parser.add_argument(
        "--direct_fusion_mode",
        choices=("residual", "direct_only"),
        default="residual",
        help="use the direct decoder as a residual or as the primary inference output",
    )


def add_loss_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lambda_seg", type=float, default=1.0)
    parser.add_argument("--lambda_dist", type=float, default=0.2)
    parser.add_argument("--lambda_fm", type=float, default=1.0)
    parser.add_argument("--lambda_orbit", type=float, default=0.3)
    parser.add_argument("--lambda_geo", type=float, default=0.5)
    parser.add_argument("--lambda_quiet", type=float, default=0.05)
    parser.add_argument(
        "--lambda_t0",
        type=float,
        default=1.0,
        help="weight for image-only t=0 full-rollout segmentation",
    )
    parser.add_argument(
        "--seg_pos_weight",
        type=float,
        default=1.0,
        help="positive-class BCE weight for changed pixels",
    )
    parser.add_argument(
        "--lambda_boundary",
        type=float,
        default=0.0,
        help="weight for inference-rollout boundary-band BCE",
    )
    parser.add_argument(
        "--lambda_lovasz",
        type=float,
        default=0.0,
        help="weight for inference-rollout Lovasz-IoU loss",
    )
    parser.add_argument("--boundary_kernel_size", type=int, default=5)
    parser.add_argument("--lambda_direct", type=float, default=0.0)
    parser.add_argument(
        "--lambda_clock",
        type=float,
        default=0.1,
        help="weight for spatial-clock offset, boundary-router and smoothness supervision",
    )
    parser.add_argument("--clock_offset_weight", type=float, default=1.0)
    parser.add_argument("--clock_boundary_weight", type=float, default=0.5)
    parser.add_argument("--clock_smooth_weight", type=float, default=0.05)
    parser.add_argument("--orbit_seg_weight", type=float, default=0.5)
    parser.add_argument("--orbit_variance_weight", type=float, default=0.1)
    parser.add_argument("--orbit_center", choices=("mean_detach", "geometry_target"), default="geometry_target")


def build_model_from_args(args: argparse.Namespace) -> NuisanceOrbitFlowRWKV:
    def value(name, default):
        return getattr(args, name, default)

    return NuisanceOrbitFlowRWKV(
        img_size=value("img_size", 256),
        size=value("model_size", "base"),
        orbit_size=value("orbit_size", 3),
        orbit_min_ops=value("orbit_min_ops", 2),
        orbit_max_ops=value("orbit_max_ops", 4),
        orbit_strength=value("orbit_strength", "medium"),
        orbit_types=value("orbit_types", "all"),
        nigf=value("nigf", True),
        flow_start=value("flow_start", "multievidence"),
        transport_mode=value("transport_mode", "flow"),
        vector_field=value("vector_field", "rwkv"),
        time_conditioning=value("time_conditioning", "decay"),
        max_time_delta=value("max_time_delta", 0.1),
        spatial_clock=value("spatial_clock", True),
        spatial_clock_max_warp=value("spatial_clock_max_warp", 0.75),
        spatial_clock_local_scale=value("spatial_clock_local_scale", 0.10),
        flow_time_sampling=value("flow_time_sampling", "uniform"),
        geometry_target=value("geometry_target", "mask_distance"),
        solver=value("solver", "euler"),
        solver_steps=value("solver_steps", 4),
        direct_fusion=value("direct_fusion", False),
        direct_fusion_mode=value("direct_fusion_mode", "residual"),
        orbit_center=value("orbit_center", "geometry_target"),
        lambda_seg=value("lambda_seg", 1.0),
        lambda_dist=value("lambda_dist", 0.2),
        lambda_fm=value("lambda_fm", 1.0),
        lambda_orbit=value("lambda_orbit", 0.3),
        lambda_geo=value("lambda_geo", 0.5),
        lambda_quiet=value("lambda_quiet", 0.05),
        lambda_t0=value("lambda_t0", 1.0),
        lambda_boundary=value("lambda_boundary", 0.0),
        lambda_lovasz=value("lambda_lovasz", 0.0),
        lambda_direct=value("lambda_direct", 0.0),
        lambda_clock=value("lambda_clock", 0.1),
        seg_pos_weight=value("seg_pos_weight", 1.0),
        boundary_kernel_size=value("boundary_kernel_size", 5),
        clock_offset_weight=value("clock_offset_weight", 1.0),
        clock_boundary_weight=value("clock_boundary_weight", 0.5),
        clock_smooth_weight=value("clock_smooth_weight", 0.05),
        orbit_seg_weight=value("orbit_seg_weight", 0.5),
        orbit_variance_weight=value("orbit_variance_weight", 0.1),
        seed=value("seed", 0),
    )


@dataclass
class BinaryMetricAccumulator:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def update(self, logits: Tensor, target: Tensor, threshold: float = 0.5) -> None:
        prediction = torch.sigmoid(logits).detach() >= threshold
        truth = target.detach() >= 0.5
        self.true_positive += int((prediction & truth).sum().cpu())
        self.false_positive += int((prediction & ~truth).sum().cpu())
        self.false_negative += int((~prediction & truth).sum().cpu())
        self.true_negative += int((~prediction & ~truth).sum().cpu())

    def scores(self) -> Dict[str, float]:
        epsilon = np.finfo(np.float32).eps
        precision = self.true_positive / (self.true_positive + self.false_positive + epsilon)
        recall = self.true_positive / (self.true_positive + self.false_negative + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        iou = self.true_positive / (self.true_positive + self.false_positive + self.false_negative + epsilon)
        accuracy = (self.true_positive + self.true_negative) / (
            self.true_positive + self.true_negative + self.false_positive + self.false_negative + epsilon
        )
        total = self.true_positive + self.true_negative + self.false_positive + self.false_negative
        predicted_positive_rate = (self.true_positive + self.false_positive) / (total + epsilon)
        target_positive_rate = (self.true_positive + self.false_negative) / (total + epsilon)
        return {
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "IoU": iou,
            "accuracy": accuracy,
            "pred_positive_rate": predicted_positive_rate,
            "target_positive_rate": target_positive_rate,
        }


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
