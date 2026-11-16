from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from model.conv_flow_field import MultiScaleConvVectorField
from model.flow_rwkv import CASCRWKVVectorField
from model.flow_solver import solve_flow
from model.geometry_field import GeometryDecoder, MultiScaleGeometryEncoder
from model.orbit_evidence import MultiScaleOrbitEvidenceEncoder
from model.nigf import NIGFEncoding, NuisanceInvariantGeometryFlow
from utils.orbit_flow_losses import OrbitFlowLoss
from utils.orbit_transforms import NuisanceOrbitGenerator, parse_orbit_types


_MODEL_CONFIGS = {
    "tiny": {
        "embed_dims": [32, 48, 96, 160],
        "depths": [2, 2, 4, 2],
        "exp_ratios": [2.0, 2.5, 3.0, 3.5],
        "flow_channels": [64, 48, 32, 24],
    },
    "small": {
        "embed_dims": [32, 64, 128, 192],
        "depths": [3, 3, 6, 3],
        "exp_ratios": [2.0, 2.5, 3.0, 4.0],
        "flow_channels": [72, 56, 40, 24],
    },
    "base": {
        "embed_dims": [48, 72, 144, 240],
        "depths": [3, 3, 6, 3],
        "exp_ratios": [2.0, 2.5, 4.0, 4.0],
        "flow_channels": [96, 72, 48, 32],
    },
}


class NuisanceOrbitFlowRWKV(nn.Module):


    def __init__(
        self,
        img_size: int = 256,
        size: str = "base",
        orbit_size: int = 3,
        orbit_min_ops: int = 2,
        orbit_max_ops: int = 4,
        orbit_strength: str = "medium",
        orbit_types: str = "all",
        nigf: bool = True,
        flow_start: str = "multievidence",
        transport_mode: str = "flow",
        vector_field: str = "rwkv",
        time_conditioning: str = "decay",
        max_time_delta: float = 0.1,
        spatial_clock: bool = False,
        spatial_clock_max_warp: float = 0.75,
        spatial_clock_local_scale: float = 0.10,
        flow_time_sampling: str = "uniform",
        geometry_target: str = "mask_distance",
        solver: str = "euler",
        solver_steps: int = 4,
        direct_fusion: bool = False,
        direct_fusion_mode: str = "residual",
        orbit_center: str = "geometry_target",
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
        clock_offset_weight: float = 1.0,
        clock_boundary_weight: float = 0.5,
        clock_smooth_weight: float = 0.05,
        orbit_seg_weight: float = 0.5,
        orbit_variance_weight: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if size not in _MODEL_CONFIGS:
            raise ValueError(f"unknown model size: {size}")
        if img_size < 128 or img_size % 16 != 0:
            raise ValueError(
                "img_size must be a multiple of 16 and at least 128; the official "
                "VRWKV encoder's coarsest WKV stage requires at least 64 tokens"
            )
        if transport_mode not in {"flow", "endpoint"}:
            raise ValueError("transport_mode must be flow or endpoint")
        if vector_field not in {"rwkv", "conv"}:
            raise ValueError("vector_field must be rwkv or conv")
        if spatial_clock and vector_field != "rwkv":
            raise ValueError("the change-adaptive spatial clock requires vector_field='rwkv'")
        if flow_time_sampling not in {"uniform", "squared", "zero"}:
            raise ValueError("flow_time_sampling must be uniform, squared, or zero")
        if geometry_target not in {"mask_only", "mask_distance"}:
            raise ValueError("geometry_target must be mask_only or mask_distance")
        if solver not in {"euler"} or solver_steps not in {1, 2, 4, 6, 8}:
            raise ValueError("invalid solver configuration")
        if direct_fusion_mode not in {"residual", "direct_only"}:
            raise ValueError("direct_fusion_mode must be residual or direct_only")
        if direct_fusion_mode == "direct_only" and not direct_fusion:
            raise ValueError("direct_only requires direct_fusion to be enabled")

        config = _MODEL_CONFIGS[size]
        self.img_size = int(img_size)
        self.size = size
        self.nigf_enabled = bool(nigf)
        effective_orbit_size = int(orbit_size) if self.nigf_enabled else 1
        effective_geometry_target = (
            geometry_target if self.nigf_enabled else "mask_only"
        )
        self.orbit_size = effective_orbit_size
        self.transport_mode = transport_mode
        self.vector_field_name = vector_field
        self.spatial_clock_enabled = bool(spatial_clock)
        self.flow_time_sampling = flow_time_sampling
        self.geometry_target_type = effective_geometry_target
        self.default_solver = solver
        self.default_solver_steps = int(solver_steps)
        self.direct_fusion = bool(direct_fusion)
        self.direct_fusion_mode = direct_fusion_mode
        self.embed_dims = list(config["embed_dims"])
        self.encoder_channels = list(reversed(self.embed_dims))
        self.flow_channels = list(config["flow_channels"])


        from model.VRWKV import VRWKV_encoder

        self.encoder = VRWKV_encoder(
            img_size=img_size,
            depths=config["depths"],
            embed_dims=self.embed_dims,
            exp_ratios=config["exp_ratios"],
            drop_path=0.05,
        )
        self.orbit_generator = NuisanceOrbitGenerator(
            orbit_size=effective_orbit_size,
            min_ops=orbit_min_ops,
            max_ops=orbit_max_ops,
            strength=orbit_strength,
            orbit_types=parse_orbit_types(orbit_types),
            seed=seed,
            input_normalized=True,
        )
        self.evidence_encoder = MultiScaleOrbitEvidenceEncoder(
            self.encoder_channels,
            self.flow_channels,
            flow_start=flow_start,
            nigf_enabled=self.nigf_enabled,
        )
        self.geometry_encoder = MultiScaleGeometryEncoder(self.flow_channels)
        self.nigf_module = NuisanceInvariantGeometryFlow(
            enabled=self.nigf_enabled,
            geometry_target=effective_geometry_target,
        )
        if vector_field == "rwkv":
            self.vector_field = CASCRWKVVectorField(
                self.flow_channels,
                time_conditioning=time_conditioning,
                max_time_delta=max_time_delta,
                spatial_clock=spatial_clock,
                spatial_clock_max_warp=spatial_clock_max_warp,
                spatial_clock_local_scale=spatial_clock_local_scale,
            )
        else:
            self.vector_field = MultiScaleConvVectorField(self.flow_channels, max_time_delta=max_time_delta)
        self.geometry_decoder = GeometryDecoder(self.flow_channels, output_size=(img_size, img_size))
        if self.direct_fusion:
            self.direct_decoder = GeometryDecoder(
                self.flow_channels, output_size=(img_size, img_size)
            )
            self.direct_fusion_scale = nn.Parameter(torch.zeros(()))
        self.loss_function = OrbitFlowLoss(
            lambda_seg=lambda_seg,
            lambda_dist=lambda_dist if self.nigf_enabled else 0.0,
            lambda_fm=lambda_fm,
            lambda_orbit=lambda_orbit if self.nigf_enabled else 0.0,
            lambda_geo=lambda_geo if self.nigf_enabled else 0.0,
            lambda_quiet=lambda_quiet,
            lambda_t0=lambda_t0,
            lambda_boundary=lambda_boundary,
            lambda_lovasz=lambda_lovasz,
            lambda_direct=lambda_direct,
            lambda_clock=lambda_clock,
            seg_pos_weight=seg_pos_weight,
            boundary_kernel_size=boundary_kernel_size,
            spatial_clock_max_warp=spatial_clock_max_warp,
            clock_offset_weight=clock_offset_weight,
            clock_boundary_weight=clock_boundary_weight,
            clock_smooth_weight=clock_smooth_weight,
            orbit_seg_weight=orbit_seg_weight,
            orbit_variance_weight=orbit_variance_weight,
            orbit_center=orbit_center,
        )

    def _nigf_controller(self) -> NuisanceInvariantGeometryFlow:

        module = getattr(self, "nigf_module", None)
        if isinstance(module, NuisanceInvariantGeometryFlow):
            return module
        return NuisanceInvariantGeometryFlow(
            enabled=bool(getattr(self, "nigf_enabled", True)),
            geometry_target=str(
                getattr(self, "geometry_target_type", "mask_distance")
            ),
        )

    def extract_features(
        self, orbit_a: Tensor, orbit_b: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        if orbit_a.ndim == 4:
            orbit_a, orbit_b = orbit_a[:, None], orbit_b[:, None]
        if orbit_a.shape != orbit_b.shape or orbit_a.ndim != 5:
            raise ValueError("orbit pairs must have matching [B,K,3,H,W] shapes")
        batch, orbit, channels, height, width = orbit_a.shape
        if (height, width) != (self.img_size, self.img_size):
            raise ValueError(
                f"model was built for {self.img_size}x{self.img_size} inputs, "
                f"but received {height}x{width}"
            )
        flat_a = orbit_a.reshape(batch * orbit, channels, height, width)
        flat_b = orbit_b.reshape(batch * orbit, channels, height, width)
        encoded = self.encoder.forward_features(torch.cat([flat_a, flat_b], dim=0))
        features_a, features_b = [], []
        for feature in encoded:
            feature_a, feature_b = torch.chunk(feature, 2, dim=0)
            _, feature_channels, feature_height, feature_width = feature_a.shape
            features_a.append(feature_a.reshape(batch, orbit, feature_channels, feature_height, feature_width))
            features_b.append(feature_b.reshape(batch, orbit, feature_channels, feature_height, feature_width))
        return features_a, features_b

    def encode_orbit_evidence(
        self, features_a: Sequence[Tensor], features_b: Sequence[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        encoding = self._nigf_controller()(
            features_a,
            features_b,
            evidence_encoder=self.evidence_encoder,
            geometry_encoder=self.geometry_encoder,
        )
        return encoding.starts, encoding.contexts

    def encode_geometry(self, geometry: Tensor, starts: Sequence[Tensor]) -> List[Tensor]:
        spatial_sizes = [state.shape[-2:] for state in starts]
        return list(self.geometry_encoder(geometry, spatial_sizes))

    def encode_nigf(
        self,
        features_a: Sequence[Tensor],
        features_b: Sequence[Tensor],
        geometry: Optional[Tensor] = None,
    ) -> NIGFEncoding:

        if hasattr(self, "evidence_encoder") and hasattr(self, "geometry_encoder"):
            return self._nigf_controller()(
                features_a,
                features_b,
                evidence_encoder=self.evidence_encoder,
                geometry_encoder=self.geometry_encoder,
                geometry=geometry,
            )
        starts, contexts = self.encode_orbit_evidence(features_a, features_b)
        endpoints = self.encode_geometry(geometry, starts) if geometry is not None else None
        return NIGFEncoding(starts, contexts, endpoints)

    def sample_flow_time(self, device: torch.device) -> Tensor:
        if self.flow_time_sampling == "zero":
            return torch.zeros(1, device=device)
        time = torch.rand(1, device=device)
        return time.square() if self.flow_time_sampling == "squared" else time

    def interpolate_flow_state(
        self, starts: Sequence[Tensor], targets: Sequence[Tensor], time: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        interpolated, target_velocities = [], []
        for start, target in zip(starts, targets):
            expanded_target = target[:, None].expand_as(start)
            interpolated.append((1.0 - time) * start + time * expanded_target)
            target_velocities.append(expanded_target - start)
        return interpolated, target_velocities

    @staticmethod
    def _flatten_orbit(states: Sequence[Tensor]) -> Tuple[List[Tensor], Tuple[int, int]]:
        batch, orbit = states[0].shape[:2]
        flattened = [state.reshape(batch * orbit, *state.shape[2:]) for state in states]
        return flattened, (batch, orbit)

    @staticmethod
    def _restore_orbit(states: Sequence[Tensor], batch: int, orbit: int) -> List[Tensor]:
        return [state.reshape(batch, orbit, *state.shape[1:]) for state in states]

    def predict_velocity(
        self,
        states: Sequence[Tensor],
        contexts: Sequence[Tensor],
        time: Tensor,
        *,
        return_clock_aux: bool = False,
    ) -> List[Tensor] | Tuple[List[Tensor], List[Dict[str, Tensor]]]:
        if states[0].ndim == 5:
            flat_states, (batch, orbit) = self._flatten_orbit(states)
            flat_contexts, _ = self._flatten_orbit(contexts)
            if return_clock_aux and getattr(self, "spatial_clock_enabled", False):
                velocity, auxiliaries = self.vector_field(
                    flat_states,
                    flat_contexts,
                    time,
                    return_clock_aux=True,
                )
                return self._restore_orbit(velocity, batch, orbit), auxiliaries
            velocity = self.vector_field(flat_states, flat_contexts, time)
            return self._restore_orbit(velocity, batch, orbit)
        if return_clock_aux and getattr(self, "spatial_clock_enabled", False):
            return self.vector_field(
                states, contexts, time, return_clock_aux=True
            )
        return self.vector_field(states, contexts, time)

    def estimate_endpoint(
        self, flow_states: Sequence[Tensor], velocities: Sequence[Tensor], time: Tensor
    ) -> List[Tensor]:
        return [state + (1.0 - time) * velocity for state, velocity in zip(flow_states, velocities)]

    def integrate(
        self,
        initial_states: Sequence[Tensor],
        contexts: Sequence[Tensor],
        solver: Optional[str] = None,
        solver_steps: Optional[int] = None,
    ) -> List[Tensor]:
        return solve_flow(
            initial_states,
            self.vector_field,
            contexts,
            solver=solver or self.default_solver,
            num_steps=solver_steps or self.default_solver_steps,
        )

    def decode_geometry(self, states: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
        return self.geometry_decoder(states, output_size=(self.img_size, self.img_size))

    def decode_direct_evidence(
        self, starts: Sequence[Tensor], contexts: Sequence[Tensor]
    ) -> Optional[Tensor]:
        if not self.direct_fusion:
            return None
        direct_states = [start + context for start, context in zip(starts, contexts)]
        direct_logits, _ = self.direct_decoder(
            direct_states, output_size=(self.img_size, self.img_size)
        )
        return direct_logits

    def fuse_direct_logits(
        self, flow_logits: Tensor, direct_logits: Optional[Tensor]
    ) -> Tensor:
        if direct_logits is None:
            return flow_logits
        if self.direct_fusion_mode == "direct_only":
            return direct_logits
        return flow_logits + torch.tanh(self.direct_fusion_scale) * direct_logits

    def forward_train(
        self,
        image_a: Tensor,
        image_b: Tensor,
        mask: Tensor,
        geometry: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if mask is None:
            raise ValueError("training requires a change mask")
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.float()
        nigf_module = self._nigf_controller()
        geometry = nigf_module.prepare_geometry(mask, geometry)

        with torch.no_grad():
            orbit_a, orbit_b, metadata = nigf_module.build_orbits(
                self.orbit_generator,
                image_a,
                image_b,
                orbit_size=self.orbit_size,
            )
        features_a, features_b = self.extract_features(orbit_a, orbit_b)
        nigf = self.encode_nigf(features_a, features_b, geometry)
        starts, contexts = nigf.starts, nigf.contexts
        if nigf.geometry_endpoints is None:
            raise RuntimeError("training NIGF path produced no geometry endpoints")
        geometry_endpoints = nigf.geometry_endpoints
        flat_starts, (batch, orbit) = self._flatten_orbit(starts)
        flat_contexts, _ = self._flatten_orbit(contexts)
        direct_mask_logits = self.decode_direct_evidence(flat_starts, flat_contexts)
        direct_identity_logits = (
            direct_mask_logits.reshape(batch, orbit, 1, self.img_size, self.img_size)[:, 0]
            if direct_mask_logits is not None
            else None
        )

        if self.transport_mode == "flow":
            spatial_clock_enabled = getattr(self, "spatial_clock_enabled", False)
            time = self.sample_flow_time(image_a.device)
            flow_states, target_velocities = self.interpolate_flow_state(starts, geometry_endpoints, time)
            clock_auxiliaries: Optional[Sequence[Dict[str, Tensor]]] = None
            velocity_output = self.predict_velocity(
                flow_states,
                contexts,
                time,
                return_clock_aux=spatial_clock_enabled,
            )
            if spatial_clock_enabled:
                predicted_velocities, clock_auxiliaries = velocity_output
            else:
                predicted_velocities = velocity_output
            predicted_endpoints = self.estimate_endpoint(flow_states, predicted_velocities, time)
            loss_target_velocities: Optional[Sequence[Tensor]] = target_velocities
            loss_predicted_velocities: Optional[Sequence[Tensor]] = predicted_velocities

            if (
                self.direct_fusion_mode == "direct_only"
                and direct_identity_logits is not None
            ):


                t0_mask_logits = direct_identity_logits
            else:


                identity_starts = [state[:, 0] for state in starts]
                identity_contexts = [context[:, 0] for context in contexts]
                t0_endpoints = self.integrate(
                    identity_starts,
                    identity_contexts,
                    solver=self.default_solver,
                    solver_steps=self.default_solver_steps,
                )
                t0_mask_logits, _ = self.decode_geometry(t0_endpoints)
                t0_mask_logits = self.fuse_direct_logits(
                    t0_mask_logits, direct_identity_logits
                )
        else:
            spatial_clock_enabled = getattr(self, "spatial_clock_enabled", False)
            time = image_a.new_zeros(())
            clock_auxiliaries = None
            velocity_output = self.predict_velocity(
                starts,
                contexts,
                time,
                return_clock_aux=spatial_clock_enabled,
            )
            if spatial_clock_enabled:
                predicted_velocities, clock_auxiliaries = velocity_output
            else:
                predicted_velocities = velocity_output
            predicted_endpoints = [start + delta for start, delta in zip(starts, predicted_velocities)]
            loss_target_velocities = None


            loss_predicted_velocities = None
            t0_mask_logits = None

        flat_endpoints, _ = self._flatten_orbit(predicted_endpoints)
        mask_logits, distance_logits = self.decode_geometry(flat_endpoints)
        mask_logits = self.fuse_direct_logits(mask_logits, direct_mask_logits)
        teacher_mask_logits, teacher_distance_logits = self.decode_geometry(geometry_endpoints)
        losses = self.loss_function(
            mask_logits=mask_logits,
            distance_logits=distance_logits,
            teacher_mask_logits=teacher_mask_logits,
            teacher_distance_logits=teacher_distance_logits,
            mask=mask,
            distance_target=geometry[:, 1:2],
            predicted_endpoints=predicted_endpoints,
            geometry_endpoints=geometry_endpoints,
            predicted_velocities=loss_predicted_velocities,
            target_velocities=loss_target_velocities,
            t0_mask_logits=t0_mask_logits,
            direct_mask_logits=direct_identity_logits,
            clock_auxiliaries=clock_auxiliaries,
        )
        identity_mask_logits = mask_logits.reshape(batch, orbit, 1, self.img_size, self.img_size)[:, 0]
        identity_distance_logits = distance_logits.reshape(batch, orbit, 1, self.img_size, self.img_size)[:, 0]
        outputs: Dict[str, object] = {
            "mask_logits": identity_mask_logits,
            "distance_logits": identity_distance_logits,
            **losses,
        }
        if t0_mask_logits is not None:
            outputs["t0_mask_logits"] = t0_mask_logits
        if direct_identity_logits is not None:
            outputs["direct_mask_logits"] = direct_identity_logits
        return outputs

    def forward_inference(
        self,
        image_a: Tensor,
        image_b: Tensor,
        solver: Optional[str] = None,
        solver_steps: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        orbit_a, orbit_b = image_a[:, None], image_b[:, None]
        features_a, features_b = self.extract_features(orbit_a, orbit_b)
        encoding = self.encode_nigf(features_a, features_b)
        starts, _ = self._flatten_orbit(encoding.starts)
        contexts, _ = self._flatten_orbit(encoding.contexts)
        direct_logits = self.decode_direct_evidence(starts, contexts)
        if self.transport_mode == "endpoint":
            velocities = self.vector_field(starts, contexts, image_a.new_zeros(()))
            endpoints = [state + velocity for state, velocity in zip(starts, velocities)]
        else:
            endpoints = self.integrate(starts, contexts, solver, solver_steps)
        mask_logits, distance_logits = self.decode_geometry(endpoints)
        return {
            "mask_logits": self.fuse_direct_logits(mask_logits, direct_logits),
            "distance_logits": distance_logits,
        }

    def forward(
        self,
        image_a: Tensor,
        image_b: Tensor,
        mask: Optional[Tensor] = None,
        geometry: Optional[Tensor] = None,
        mode: str = "inference",
        solver: Optional[str] = None,
        solver_steps: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        if mode == "train":
            if mask is None:
                raise ValueError("mode='train' requires mask")
            return self.forward_train(image_a, image_b, mask, geometry)
        if mode == "inference":
            return self.forward_inference(image_a, image_b, solver, solver_steps)
        raise ValueError("mode must be train or inference")


NIGFCASCRWKV = NuisanceOrbitFlowRWKV

__all__ = [
    "NIGFCASCRWKV",
    "NuisanceOrbitFlowRWKV",
    "NuisanceInvariantGeometryFlow",
    "CASCRWKVVectorField",
]


def model_config(size: str) -> Dict[str, Sequence[int]]:
    if size not in _MODEL_CONFIGS:
        raise ValueError(size)
    return {key: list(value) for key, value in _MODEL_CONFIGS[size].items()}
