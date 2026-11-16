from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


_MIN_CUDA_WKV_TOKENS = 64


def spatial_shift(tokens: Tensor, resolution: Tuple[int, int], fraction: float = 0.25) -> Tensor:
    batch, length, channels = tokens.shape
    height, width = resolution
    if length != height * width:
        raise ValueError("token length does not match spatial resolution")
    feature = tokens.transpose(1, 2).reshape(batch, channels, height, width)
    shifted = torch.zeros_like(feature)
    quarter = int(channels * fraction)
    shifted[:, :quarter, :, 1:] = feature[:, :quarter, :, :-1]
    shifted[:, quarter : 2 * quarter, :, :-1] = feature[:, quarter : 2 * quarter, :, 1:]
    shifted[:, 2 * quarter : 3 * quarter, 1:, :] = feature[:, 2 * quarter : 3 * quarter, :-1, :]
    shifted[:, 3 * quarter : 4 * quarter, :-1, :] = feature[:, 3 * quarter : 4 * quarter, 1:, :]
    shifted[:, 4 * quarter :, :, :] = feature[:, 4 * quarter :, :, :]
    return shifted.flatten(2).transpose(1, 2)


class SinusoidalFlowTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 64, max_period: float = 10000.0) -> None:
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least two")
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, time: Tensor) -> Tensor:
        if time.numel() != 1:
            raise ValueError("FlowRWKV requires one scalar time shared by the mini-batch")
        time = time.reshape(1).float()
        half = self.embedding_dim // 2
        frequency = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=time.device, dtype=time.dtype)
            / max(half - 1, 1)
        )
        embedding = torch.cat([torch.sin(time[:, None] * frequency), torch.cos(time[:, None] * frequency)], dim=-1)
        if embedding.shape[-1] < self.embedding_dim:
            embedding = F.pad(embedding, (0, self.embedding_dim - embedding.shape[-1]))
        return embedding


def _reference_bidirectional_wkv(w: Tensor, u: Tensor, key: Tensor, value: Tensor) -> Tensor:

    batch, length, channels = key.shape
    positions = torch.arange(length, device=key.device, dtype=key.dtype)
    distance = (positions[:, None] - positions[None, :]).abs()
    scores = key.transpose(1, 2).unsqueeze(2) - w.reshape(1, channels, 1, 1) * distance.reshape(1, 1, length, length)
    diagonal = torch.eye(length, device=key.device, dtype=key.dtype).reshape(1, 1, length, length)
    scores = scores + diagonal * u.reshape(1, channels, 1, 1)
    weights = torch.softmax(scores.float(), dim=-1).to(value.dtype)
    output = torch.matmul(weights, value.transpose(1, 2).unsqueeze(-1)).squeeze(-1)
    return output.transpose(1, 2)


def _run_wkv(w: Tensor, u: Tensor, key: Tensor, value: Tensor) -> Tensor:
    batch, length, channels = key.shape
    original_length = length
    if length < _MIN_CUDA_WKV_TOKENS:
        padding = _MIN_CUDA_WKV_TOKENS - length


        key_padding = key.new_full(
            (batch, padding, channels), torch.finfo(key.dtype).min
        )
        value_padding = value.new_zeros((batch, padding, channels))
        key = torch.cat([key, key_padding], dim=1).contiguous()
        value = torch.cat([value, value_padding], dim=1).contiguous()
        length = _MIN_CUDA_WKV_TOKENS
    if key.is_cuda:
        try:
            from model.VRWKV import RUN_CUDA

            output = RUN_CUDA(batch, length, channels, w, u, key, value)
        except Exception as error:
            raise RuntimeError(
                "Flow-Time decay conditioning requires the bundled WKV CUDA kernel; "
                f"the kernel failed for w/u=[{channels}], k/v={tuple(key.shape)}"
            ) from error
    else:
        output = _reference_bidirectional_wkv(w, u, key, value)
    return output[:, :original_length]


class _ChannelMix(nn.Module):
    def __init__(self, channels: int, hidden_rate: int = 2) -> None:
        super().__init__()
        self.mix_key = nn.Parameter(torch.full((1, 1, channels), 0.5))
        self.mix_receptance = nn.Parameter(torch.full((1, 1, channels), 0.5))
        self.key = nn.Linear(channels, channels * hidden_rate, bias=False)
        self.key_norm = nn.LayerNorm(channels * hidden_rate)
        self.receptance = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels * hidden_rate, channels, bias=False)

    def forward(self, tokens: Tensor, resolution: Tuple[int, int]) -> Tensor:
        shifted = spatial_shift(tokens, resolution)
        key_input = tokens * self.mix_key + shifted * (1.0 - self.mix_key)
        receptance_input = tokens * self.mix_receptance + shifted * (1.0 - self.mix_receptance)
        key = self.key_norm(torch.relu(self.key(key_input)).square())
        return torch.sigmoid(self.receptance(receptance_input)) * self.value(key)


class FlowTimeSpatialMix(nn.Module):


    def __init__(
        self,
        channels: int,
        time_embedding_dim: int = 64,
        time_conditioning: str = "decay",
        max_time_delta: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__()
        if time_conditioning not in {"decay", "film", "none"}:
            raise ValueError("time_conditioning must be decay, film, or none")
        self.channels = channels
        self.time_conditioning = time_conditioning
        self.max_time_delta = float(max_time_delta)
        self.max_tokens = int(max_tokens)
        self.base_decay = nn.Parameter(torch.zeros(channels))
        self.base_first = nn.Parameter(torch.zeros(channels))
        self.mix_key = nn.Parameter(torch.full((1, 1, channels), 0.5))
        self.mix_value = nn.Parameter(torch.full((1, 1, channels), 0.5))
        self.mix_receptance = nn.Parameter(torch.full((1, 1, channels), 0.5))
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.receptance = nn.Linear(channels, channels, bias=False)
        self.key_norm = nn.LayerNorm(channels)
        self.output = nn.Linear(channels, channels, bias=False)
        self.time_encoder = SinusoidalFlowTimeEmbedding(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, channels * 2),
        )
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

    def dynamic_parameters(self, time: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        embedding = self.time_encoder(time).to(dtype=self.base_decay.dtype)
        modulation = self.time_mlp(embedding).squeeze(0)
        first, second = modulation.chunk(2, dim=-1)
        if self.time_conditioning == "decay":
            decay = self.base_decay + self.max_time_delta * torch.tanh(first)
            initial = self.base_first + self.max_time_delta * torch.tanh(second)
            film_scale = torch.zeros_like(decay)
            film_shift = torch.zeros_like(decay)
        elif self.time_conditioning == "film":
            decay, initial = self.base_decay, self.base_first
            film_scale = self.max_time_delta * torch.tanh(first)
            film_shift = self.max_time_delta * torch.tanh(second)
        else:
            decay, initial = self.base_decay, self.base_first
            film_scale = torch.zeros_like(decay)
            film_shift = torch.zeros_like(decay)
        return decay, initial, film_scale, film_shift

    def forward(self, tokens: Tensor, resolution: Tuple[int, int], time: Tensor) -> Tensor:
        batch, total_tokens, channels = tokens.shape
        if channels != self.channels:
            raise ValueError("unexpected channel count")
        decay, first, film_scale, film_shift = self.dynamic_parameters(time)
        if self.time_conditioning == "film":
            tokens = tokens * (1.0 + film_scale.reshape(1, 1, -1)) + film_shift.reshape(1, 1, -1)
        shifted = spatial_shift(tokens, resolution)
        key_input = tokens * self.mix_key + shifted * (1.0 - self.mix_key)
        value_input = tokens * self.mix_value + shifted * (1.0 - self.mix_value)
        receptance_input = tokens * self.mix_receptance + shifted * (1.0 - self.mix_receptance)
        key = self.key(key_input)
        value = self.value(value_input)
        receptance = torch.sigmoid(self.receptance(receptance_input))

        outputs = []
        for start in range(0, total_tokens, self.max_tokens):
            stop = min(start + self.max_tokens, total_tokens)
            chunk_length = stop - start
            outputs.append(
                _run_wkv(
                    decay / chunk_length,
                    first / chunk_length,
                    key[:, start:stop].contiguous(),
                    value[:, start:stop].contiguous(),
                )
            )
        mixed = torch.cat(outputs, dim=1)
        return self.output(receptance * self.key_norm(mixed))


def _normalization_groups(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ChangeAdaptiveSpatialClock(nn.Module):


    def __init__(
        self,
        channels: int,
        *,
        max_warp: float = 0.75,
        local_residual_scale: float = 0.10,
        router_prior: float = 0.10,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if not 0.0 < max_warp < 1.0:
            raise ValueError("max_warp must be in (0, 1) to preserve clock monotonicity")
        if local_residual_scale < 0.0:
            raise ValueError("local_residual_scale must be non-negative")
        if not 0.0 < router_prior < 1.0:
            raise ValueError("router_prior must be in (0, 1)")

        hidden = max(16, channels // 2)
        hidden_groups = _normalization_groups(hidden)
        self.max_warp = float(max_warp)
        self.local_residual_scale = float(local_residual_scale)
        self.clock_features = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(hidden_groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(hidden_groups, hidden),
            nn.SiLU(),
        )
        self.clock_head = nn.Conv2d(hidden, 1, 1)
        self.spatial_time_modulation = nn.Sequential(
            nn.Conv2d(4, hidden, 1, bias=False),
            nn.GroupNorm(hidden_groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels * 2, 1),
        )
        self.local_velocity = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(_normalization_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.router = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(hidden_groups, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )

        nn.init.zeros_(self.clock_head.weight)
        nn.init.zeros_(self.clock_head.bias)
        nn.init.zeros_(self.spatial_time_modulation[-1].weight)
        nn.init.zeros_(self.spatial_time_modulation[-1].bias)
        nn.init.zeros_(self.local_velocity[-1].weight)
        nn.init.zeros_(self.local_velocity[-1].bias)
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(
            self.router[-1].bias,
            math.log(router_prior / (1.0 - router_prior)),
        )

    @staticmethod
    def _time_basis(local_time: Tensor) -> Tensor:
        return torch.cat(
            [
                local_time,
                local_time.square(),
                torch.sin(math.pi * local_time),
                torch.cos(math.pi * local_time),
            ],
            dim=1,
        )

    def forward(
        self, feature: Tensor, time: Tensor
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        if feature.ndim != 4:
            raise ValueError("spatial clock expects a [N,C,H,W] feature map")
        if time.numel() != 1:
            raise ValueError("spatial clock requires one scalar integration time")
        clock_feature = self.clock_features(feature)
        clock_offset = self.max_warp * torch.tanh(self.clock_head(clock_feature))
        scalar_time = time.reshape(1, 1, 1, 1).to(
            device=feature.device, dtype=feature.dtype
        )
        local_time = scalar_time + clock_offset * scalar_time * (1.0 - scalar_time)
        scale, shift = self.spatial_time_modulation(
            self._time_basis(local_time)
        ).chunk(2, dim=1)
        modulated = feature * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * shift
        router_logits = self.router(clock_feature)
        local_residual = (
            self.local_residual_scale
            * torch.sigmoid(router_logits)
            * self.local_velocity(modulated)
        )
        auxiliary = {
            "clock_offset": clock_offset,
            "local_time": local_time,
            "router_logits": router_logits,
        }
        return modulated, local_residual, auxiliary


class FlowTimeRWKVBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        time_conditioning: str = "decay",
        max_time_delta: float = 0.1,
        time_embedding_dim: int = 64,
        spatial_clock: bool = False,
        spatial_clock_max_warp: float = 0.75,
        spatial_clock_local_scale: float = 0.10,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.conditional_projection = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.norm1 = nn.LayerNorm(channels)
        self.spatial_mix = FlowTimeSpatialMix(
            channels,
            time_embedding_dim=time_embedding_dim,
            time_conditioning=time_conditioning,
            max_time_delta=max_time_delta,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.channel_mix = _ChannelMix(channels)
        self.local_refinement = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
        )
        self.velocity_projection = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.velocity_projection.weight)
        nn.init.zeros_(self.velocity_projection.bias)
        self.change_adaptive_clock = (
            ChangeAdaptiveSpatialClock(
                channels,
                max_warp=spatial_clock_max_warp,
                local_residual_scale=spatial_clock_local_scale,
            )
            if spatial_clock
            else None
        )

    def forward(
        self,
        state: Tensor,
        context: Tensor,
        time: Tensor,
        *,
        return_clock_aux: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        if state.shape != context.shape or state.ndim != 4:
            raise ValueError("state and context must have matching [N,C,H,W] shapes")
        feature = self.conditional_projection(torch.cat([state, context], dim=1))
        clock_aux: Dict[str, Tensor] = {}
        local_velocity = torch.zeros_like(feature)
        if self.change_adaptive_clock is not None:
            feature, local_velocity, clock_aux = self.change_adaptive_clock(
                feature, time
            )
        batch, channels, height, width = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        tokens = tokens + self.spatial_mix(self.norm1(tokens), (height, width), time)
        tokens = tokens + self.channel_mix(self.norm2(tokens), (height, width))
        feature = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        feature = feature + self.local_refinement(feature)
        velocity = self.velocity_projection(feature) + local_velocity
        return (velocity, clock_aux) if return_clock_aux else velocity


class MultiScaleFlowRWKV(nn.Module):
    def __init__(
        self,
        flow_channels: Sequence[int],
        time_conditioning: str = "decay",
        max_time_delta: float = 0.1,
        spatial_clock: bool = False,
        spatial_clock_max_warp: float = 0.75,
        spatial_clock_local_scale: float = 0.10,
    ) -> None:
        super().__init__()
        if len(flow_channels) != 4:
            raise ValueError("exactly four scales are required")
        self.blocks = nn.ModuleList(
            [
                FlowTimeRWKVBlock(
                    ch,
                    time_conditioning,
                    max_time_delta,
                    spatial_clock=spatial_clock,
                    spatial_clock_max_warp=spatial_clock_max_warp,
                    spatial_clock_local_scale=spatial_clock_local_scale,
                )
                for ch in flow_channels
            ]
        )
        self.guidance_projections = nn.ModuleList(
            [nn.Conv2d(flow_channels[index - 1], flow_channels[index], 1, bias=False) for index in range(1, 4)]
        )

    def forward(
        self,
        states: Sequence[Tensor],
        contexts: Sequence[Tensor],
        time: Tensor,
        *,
        return_clock_aux: bool = False,
    ) -> List[Tensor] | Tuple[List[Tensor], List[Dict[str, Tensor]]]:
        if len(states) != 4 or len(contexts) != 4:
            raise ValueError("exactly four coarse-to-fine scales are required")
        velocities: List[Tensor] = []
        clock_auxiliaries: List[Dict[str, Tensor]] = []
        for index, (block, state, context) in enumerate(zip(self.blocks, states, contexts)):
            if index:
                guidance = F.interpolate(
                    velocities[-1], size=state.shape[-2:], mode="bilinear", align_corners=False
                )
                state = state + self.guidance_projections[index - 1](guidance)
            if return_clock_aux:
                velocity, auxiliary = block(
                    state, context, time, return_clock_aux=True
                )
                velocities.append(velocity)
                clock_auxiliaries.append(auxiliary)
            else:
                velocity = block(state, context, time)
                if not isinstance(velocity, Tensor):
                    raise TypeError("Flow block returned auxiliary data unexpectedly")
                velocities.append(velocity)
        return (velocities, clock_auxiliaries) if return_clock_aux else velocities


class CASCRWKVVectorField(MultiScaleFlowRWKV):
    pass
