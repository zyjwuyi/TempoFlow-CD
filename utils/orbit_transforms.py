from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


_ALIASES = {
    "noise": "gaussian_noise",
    "blur": "gaussian_blur",
    "frequency": "frequency_amplitude",
    "jpeg": "jpeg_approx",
}

_ALL_TYPES = (
    "brightness",
    "contrast",
    "gamma",
    "channel_gain",
    "gaussian_noise",
    "speckle_noise",
    "gaussian_blur",
    "frequency_amplitude",
    "jpeg_approx",
)


def _uniform(
    low: float,
    high: float,
    generator: torch.Generator,
    device: torch.device,
    shape: Sequence[int] = (),
) -> Tensor:
    return torch.empty(tuple(shape), device=device).uniform_(low, high, generator=generator)


class NuisanceOrbitGenerator(nn.Module):


    def __init__(
        self,
        orbit_size: int = 3,
        min_ops: int = 2,
        max_ops: int = 4,
        strength: str = "medium",
        orbit_types: Optional[Iterable[str]] = None,
        seed: int = 0,
        input_normalized: bool = False,
    ) -> None:
        super().__init__()
        if orbit_size < 1:
            raise ValueError("orbit_size must be at least one")
        if min_ops < 1 or max_ops < min_ops:
            raise ValueError("require 1 <= min_ops <= max_ops")
        if strength not in {"weak", "medium", "strong"}:
            raise ValueError("strength must be weak, medium, or strong")
        selected = list(orbit_types) if orbit_types is not None else list(_ALL_TYPES)
        selected = [_ALIASES.get(name.strip(), name.strip()) for name in selected]
        unknown = sorted(set(selected) - set(_ALL_TYPES))
        if unknown:
            raise ValueError(f"unknown nuisance types: {unknown}")
        if not selected:
            raise ValueError("at least one nuisance type is required")

        self.orbit_size = int(orbit_size)
        self.min_ops = int(min_ops)
        self.max_ops = int(max_ops)
        self.strength = strength
        self.orbit_types = tuple(dict.fromkeys(selected))
        self.seed = int(seed)
        self.input_normalized = bool(input_normalized)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )
        self._generators: Dict[str, torch.Generator] = {}

    def reset_seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self._generators.clear()

    def _generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        if key not in self._generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            self._generators[key] = generator
        return self._generators[key]

    def _to_image_domain(self, image: Tensor) -> Tensor:
        if not self.input_normalized:
            return image
        return image * self.std.to(image) + self.mean.to(image)

    def _from_image_domain(self, image: Tensor) -> Tensor:
        if not self.input_normalized:
            return image
        return (image - self.mean.to(image)) / self.std.to(image)

    def _scaled_range(self, center: float, low: float, high: float) -> Tuple[float, float]:
        multiplier = {"weak": 0.55, "medium": 1.0, "strong": 1.45}[self.strength]
        return center + (low - center) * multiplier, center + (high - center) * multiplier

    def _apply_operation(self, image: Tensor, name: str, generator: torch.Generator) -> Tuple[Tensor, Dict]:
        device = image.device
        metadata: Dict[str, object] = {"type": name}

        if name == "brightness":
            low, high = self._scaled_range(1.0, 0.75, 1.25)
            factor = _uniform(low, high, generator, device)
            image = image * factor
            metadata["factor"] = float(factor)
        elif name == "contrast":
            low, high = self._scaled_range(1.0, 0.75, 1.25)
            factor = _uniform(low, high, generator, device)
            mean = image.mean(dim=(-2, -1), keepdim=True)
            image = (image - mean) * factor + mean
            metadata["factor"] = float(factor)
        elif name == "gamma":
            low, high = self._scaled_range(1.0, 0.75, 1.35)
            gamma = _uniform(max(low, 0.1), high, generator, device)
            image = image.clamp(0, 1).pow(gamma)
            metadata["gamma"] = float(gamma)
        elif name == "channel_gain":
            low, high = self._scaled_range(1.0, 0.85, 1.15)
            gain = _uniform(low, high, generator, device, (1, image.shape[1], 1, 1))
            image = image * gain
            metadata["gain"] = gain.flatten().detach().cpu().tolist()
        elif name in {"gaussian_noise", "speckle_noise"}:
            _, sigma_high = self._scaled_range(0.0, 0.0, 0.05)
            sigma = _uniform(0.0, sigma_high, generator, device)
            noise = torch.randn(image.shape, device=device, dtype=image.dtype, generator=generator) * sigma
            image = image + noise if name == "gaussian_noise" else image + image * noise
            metadata["sigma"] = float(sigma)
        elif name == "gaussian_blur":
            sigma_high = {"weak": 0.8, "medium": 1.5, "strong": 2.0}[self.strength]
            sigma = _uniform(0.1, sigma_high, generator, device)
            kernel_size = 3 if int(_uniform(0, 2, generator, device)) == 0 else 5
            radius = kernel_size // 2
            coords = torch.arange(-radius, radius + 1, device=device, dtype=image.dtype)
            kernel_1d = torch.exp(-(coords.square()) / (2 * sigma.square()))
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel = torch.outer(kernel_1d, kernel_1d)
            kernel = kernel.expand(image.shape[1], 1, kernel_size, kernel_size)
            image = F.conv2d(F.pad(image, (radius,) * 4, mode="reflect"), kernel, groups=image.shape[1])
            metadata.update(kernel_size=kernel_size, sigma=float(sigma))
        elif name == "frequency_amplitude":
            low, high = self._scaled_range(1.0, 0.8, 1.2)
            factor = _uniform(low, high, generator, device)
            spectrum = torch.fft.rfft2(image.float(), norm="ortho")
            h, wr = spectrum.shape[-2:]
            fy = torch.fft.fftfreq(h, device=device).view(h, 1)
            fx = torch.fft.rfftfreq(image.shape[-1], device=device).view(1, wr)
            low_frequency = torch.exp(-8.0 * (fy.square() + fx.square()))
            gain = 1.0 + (factor.float() - 1.0) * low_frequency
            spectrum = spectrum * gain
            image = torch.fft.irfft2(spectrum, s=image.shape[-2:], norm="ortho").to(image.dtype)
            metadata["low_frequency_gain"] = float(factor)
        elif name == "jpeg_approx":
            scale_low = {"weak": 0.88, "medium": 0.72, "strong": 0.55}[self.strength]
            scale = _uniform(scale_low, 0.96, generator, device)
            h, w = image.shape[-2:]
            down_size = (max(1, int(h * float(scale))), max(1, int(w * float(scale))))
            image = F.interpolate(image, size=down_size, mode="bilinear", align_corners=False)
            image = F.interpolate(image, size=(h, w), mode="bilinear", align_corners=False)
            levels = {"weak": 255.0, "medium": 127.0, "strong": 63.0}[self.strength]
            image = torch.round(image * levels) / levels
            metadata.update(scale=float(scale), quantization_levels=levels)
        else:
            raise RuntimeError(name)
        return image.clamp(0.0, 1.0), metadata

    def _sample_ops(self, generator: torch.Generator, device: torch.device) -> List[str]:
        maximum = min(self.max_ops, len(self.orbit_types))
        minimum = min(self.min_ops, maximum)
        count = int(_uniform(minimum, maximum + 0.999, generator, device).floor().item())
        permutation = torch.randperm(len(self.orbit_types), generator=generator, device=device)
        return [self.orbit_types[int(i)] for i in permutation[:count]]

    def transform_image(
        self,
        image: Tensor,
        operations: Sequence[str],
        generator: torch.Generator,
    ) -> Tuple[Tensor, List[Dict]]:
        transformed = self._to_image_domain(image).clamp(0.0, 1.0)
        metadata = []
        for operation in operations:
            transformed, op_metadata = self._apply_operation(transformed, operation, generator)
            metadata.append(op_metadata)
        return self._from_image_domain(transformed), metadata

    def forward(
        self,
        image_a: Tensor,
        image_b: Tensor,
        generator: Optional[torch.Generator] = None,
        orbit_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Dict]:
        if image_a.shape != image_b.shape or image_a.ndim != 4 or image_a.shape[1] != 3:
            raise ValueError("image_a and image_b must both have shape [B, 3, H, W]")
        size = self.orbit_size if orbit_size is None else int(orbit_size)
        if size < 1:
            raise ValueError("orbit_size must be at least one")
        generator = generator or self._generator(image_a.device)

        orbit_a = [image_a]
        orbit_b = [image_b]
        orbit_metadata: List[Dict] = [{"index": 0, "identity": True, "a": [], "b": []}]
        for index in range(1, size):
            operations = self._sample_ops(generator, image_a.device)

            transformed_a, metadata_a = self.transform_image(image_a, operations, generator)
            transformed_b, metadata_b = self.transform_image(image_b, operations, generator)
            orbit_a.append(transformed_a)
            orbit_b.append(transformed_b)
            orbit_metadata.append(
                {"index": index, "identity": False, "operations": operations, "a": metadata_a, "b": metadata_b}
            )

        return (
            torch.stack(orbit_a, dim=1),
            torch.stack(orbit_b, dim=1),
            {"identity_index": 0, "orbits": orbit_metadata, "strength": self.strength},
        )


def parse_orbit_types(value: str) -> Tuple[str, ...]:
    if not value or value.lower() == "all":
        return _ALL_TYPES
    return tuple(_ALIASES.get(item.strip(), item.strip()) for item in value.split(",") if item.strip())
