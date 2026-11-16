from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor


Checkpoint = Union[str, Path, Mapping[str, object]]


def extract_state_dict(checkpoint: Mapping[str, object]) -> Mapping[str, Tensor]:
    candidate: object = checkpoint
    for container_key in ("state_dict", "model"):
        if isinstance(candidate, Mapping) and container_key in candidate and isinstance(candidate[container_key], Mapping):
            candidate = candidate[container_key]
            break
    if not isinstance(candidate, Mapping):
        raise TypeError("checkpoint does not contain a state dictionary")
    state = {str(key): value for key, value in candidate.items() if isinstance(value, Tensor)}
    if not state:
        raise ValueError("checkpoint state dictionary contains no tensors")
    return state


def _strip_module(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    return key


def _candidate_keys(key: str):
    key = _strip_module(key)
    candidates = [key]
    if key.startswith("encoder."):
        candidates.append(key[len("encoder.") :])
    else:
        candidates.append("encoder." + key)
    return tuple(dict.fromkeys(candidates))


def load_checkpoint_flexible(
    model: nn.Module,
    checkpoint: Checkpoint,
    *,
    require_encoder: bool = True,
    map_location: str = "cpu",
    verbose: bool = True,
) -> Dict[str, object]:
    if isinstance(checkpoint, (str, Path)):


        try:
            payload = torch.load(
                checkpoint, map_location=map_location, weights_only=False
            )
        except TypeError:
            payload = torch.load(checkpoint, map_location=map_location)
    else:
        payload = checkpoint
    state = extract_state_dict(payload)
    destination = model.state_dict()
    matched: MutableMapping[str, Tensor] = {}
    unexpected = []
    shape_mismatch = []
    source_to_destination: Dict[str, str] = {}

    for source_key, value in state.items():
        shape_candidate_seen = False
        loaded = False
        for candidate in _candidate_keys(source_key):
            if candidate not in destination:
                continue
            shape_candidate_seen = True
            if destination[candidate].shape == value.shape:
                matched[candidate] = value
                source_to_destination[source_key] = candidate
                loaded = True
                break
        if not loaded:
            (shape_mismatch if shape_candidate_seen else unexpected).append(source_key)

    model.load_state_dict(matched, strict=False)
    missing = [key for key in destination if key not in matched]
    encoder_loaded = sum(key.startswith("encoder.") for key in matched)

    if not any(key.startswith("encoder.") for key in destination):
        encoder_loaded = len(matched)
    if require_encoder and encoder_loaded == 0:
        raise RuntimeError("checkpoint loading matched zero encoder parameters")

    report: Dict[str, object] = {
        "loaded_count": len(matched),
        "encoder_loaded_count": encoder_loaded,
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "shape_mismatch_count": len(shape_mismatch),
        "loaded_keys": sorted(matched),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch_keys": shape_mismatch,
        "source_to_destination": source_to_destination,
    }
    if verbose:
        print(
            "Checkpoint load: "
            f"loaded={report['loaded_count']} missing={report['missing_count']} "
            f"unexpected={report['unexpected_count']} shape_mismatch={report['shape_mismatch_count']}"
        )
    return report


def save_training_checkpoint(
    path: Union[str, Path],
    *,
    epoch: int,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[object],
    scaler: Optional[object],
    config: Mapping[str, object],
    best_metrics: Mapping[str, float],
) -> None:
    parallel_wrappers = (nn.DataParallel, nn.parallel.DistributedDataParallel)
    unwrapped = model.module if isinstance(model, parallel_wrappers) else model
    payload = {
        "epoch": int(epoch),
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": dict(config),
        "best_metrics": {str(key): float(value) for key, value in best_metrics.items()},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
