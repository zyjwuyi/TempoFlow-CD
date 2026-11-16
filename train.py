from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import os
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler

from utils.checkpoint_utils import load_checkpoint_flexible, save_training_checkpoint
from utils.dataset import ChangeDetectionDataset
from utils.orbit_flow_losses import dice_loss_with_logits
from utils.orbit_flow_runtime import (
    BinaryMetricAccumulator,
    add_architecture_arguments,
    add_loss_arguments,
    build_model_from_args,
    count_parameters,
    seed_everything,
)


LOSS_KEYS = (
    "loss_total",
    "loss_seg",
    "loss_distance",
    "loss_fm",
    "loss_orbit",
    "loss_geometry",
    "loss_quiet",
    "loss_t0",
    "loss_boundary",
    "loss_lovasz",
    "loss_direct",
    "loss_clock",
)


class DistributedEvalSampler(Sampler[int]):


    def __init__(
        self,
        dataset,
        *,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        if num_replicas is None:
            if not dist.is_initialized():
                raise RuntimeError("distributed evaluation requires an initialized process group")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_initialized():
                raise RuntimeError("distributed evaluation requires an initialized process group")
            rank = dist.get_rank()
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.num_replicas - 1) // self.num_replicas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NIGF-CASC RWKV")
    parser.add_argument("--dataset", default="LEVIR")
    parser.add_argument("--data_dir", default="data/LEVIR_ABlabel")
    parser.add_argument("--pretrain_dir", default=None)
    parser.add_argument(
        "--resume",
        default=None,
        help="resume model, optimizer, scheduler, scaler, epoch and best metrics from a training checkpoint",
    )
    parser.add_argument("--output_dir", default="outputs/train")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="limit real training samples for a smoke/debug run; 0 uses the full split",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=0,
        help="limit real validation samples for a smoke/debug run; 0 uses the full split",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr_encoder", type=float, default=1e-5)
    parser.add_argument("--lr_new", type=float, default=1e-4)
    parser.add_argument(
        "--lr_direct",
        type=float,
        default=None,
        help="optional separate LR for the newly initialized direct-fusion branch",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--amp_dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="bfloat16 avoids FP16 gradient-range overflows on supported GPUs",
    )
    parser.add_argument(
        "--amp_init_scale",
        type=float,
        default=256.0,
        help="initial GradScaler scale chosen for stable base-model multi-GPU training",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help="EMA decay for validation/checkpoint weights; 0 disables EMA",
    )
    add_architecture_arguments(parser)
    add_loss_arguments(parser)
    parser.set_defaults(validation_split="val", spatial_clock=True, lambda_clock=0.1,
                        seg_pos_weight=2.5, amp_dtype="bfloat16", ema_decay=0.999,
                        warmup_epochs=1)
    return parser.parse_args()


def make_scheduler(optimizer, steps_per_epoch: int, epochs: int, warmup_epochs: int):
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(0, steps_per_epoch * warmup_epochs)

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def load_resume_payload(path: str) -> Mapping[str, object]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("resume checkpoint must contain a mapping payload")
    required = ("epoch", "model", "optimizer", "scheduler", "best_metrics")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"resume checkpoint is missing required fields: {missing}")
    return payload


def restore_ema_from_companion(
    ema: "ModelEMA",
    resume_path: str,
    resume_payload: Mapping[str, object],
    *,
    verbose: bool = True,
) -> bool:

    raw_path = Path(resume_path)
    ema_path = raw_path.with_name(f"{raw_path.stem}_ema{raw_path.suffix}")
    if not ema_path.is_file():
        if verbose:
            print(f"EMA companion not found; initialized from raw model: {ema_path}")
        return False
    ema_payload = load_resume_payload(str(ema_path))
    raw_epoch = int(resume_payload["epoch"])
    ema_epoch = int(ema_payload["epoch"])
    if ema_epoch != raw_epoch:
        if verbose:
            print(
                f"EMA companion epoch mismatch; initialized from raw model: "
                f"raw={raw_epoch} ema={ema_epoch} path={ema_path}"
            )
        return False
    load_checkpoint_flexible(
        ema.model, ema_payload, require_encoder=True, verbose=verbose
    )
    if verbose:
        print(f"restored EMA checkpoint={ema_path} epoch={ema_epoch}")
    return True


def prepare_metrics_file(
    path: Path, fields, *, resume_epoch: int | None = None
) -> None:
    if resume_epoch is None:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        atomic_write_text(path, stream.getvalue())
        return
    if not path.is_file():
        raise FileNotFoundError(f"cannot resume without the existing metrics file: {path}")
    raw = path.read_bytes()
    nul_offset = raw.find(b"\0")
    if nul_offset >= 0:
        raise ValueError(
            f"existing metrics contain NUL corruption at byte {nul_offset}: {path}"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(fields):
            raise ValueError("existing metrics header does not match the current training schema")
        rows = list(reader)
    if not rows:
        raise ValueError("cannot resume from an empty metrics file")
    last_epoch = int(rows[-1]["epoch"])
    if last_epoch != resume_epoch:
        raise ValueError(
            f"metrics/checkpoint epoch mismatch: metrics={last_epoch}, checkpoint={resume_epoch}"
        )


def atomic_write_text(path: Path, payload: str) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def atomic_append_text(path: Path, payload: str) -> None:

    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    atomic_write_text(path, existing + payload)


def append_metrics_row(path: Path, fields, row: Mapping[str, object]) -> None:
    stream = io.StringIO(newline="")
    csv.DictWriter(stream, fieldnames=fields).writerow(row)
    atomic_append_text(path, stream.getvalue())


class ModelEMA:


    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be between zero and one")
        parallel_wrappers = (nn.DataParallel, DistributedDataParallel)
        source = model.module if isinstance(model, parallel_wrappers) else model
        self.model = copy.deepcopy(source).eval()
        self.model.requires_grad_(False)
        self.decay = float(decay)
        self.parameters = list(self.model.parameters())

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        parallel_wrappers = (nn.DataParallel, DistributedDataParallel)
        source = model.module if isinstance(model, parallel_wrappers) else model
        source_parameters = list(source.parameters())
        torch._foreach_mul_(self.parameters, self.decay)
        torch._foreach_add_(self.parameters, source_parameters, alpha=1.0 - self.decay)
        for ema_buffer, source_buffer in zip(self.model.buffers(), source.buffers()):
            ema_buffer.copy_(source_buffer)


def reduce_loss(output: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    value = output[key].mean()
    if not torch.isfinite(value):
        raise FloatingPointError(f"{key} became NaN/Inf: {value.detach().cpu().item()}")
    return value


def train_epoch(
    model, loader, optimizer, scheduler, scaler, args, device, epoch: int, ema=None
) -> Dict[str, float]:
    model.train()
    sums = {key: 0.0 for key in LOSS_KEYS}
    samples = 0
    for step, (image_a, image_b, mask, geometry) in enumerate(loader, start=1):
        image_a = image_a.to(device, non_blocking=True).float()
        image_b = image_b.to(device, non_blocking=True).float()
        mask = mask.to(device, non_blocking=True).float().unsqueeze(1)
        geometry = geometry.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
        with torch.autocast(device_type="cuda", enabled=args.amp, dtype=amp_dtype):
            output = model(image_a, image_b, mask=mask, geometry=geometry, mode="train")
            losses = {key: reduce_loss(output, key) for key in LOSS_KEYS}
        scaler.scale(losses["loss_total"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()


        if scaler.get_scale() >= scale_before_step:
            scheduler.step()
            if ema is not None:
                ema.update(model)
        batch = image_a.shape[0]
        samples += batch
        for key, value in losses.items():
            sums[key] += float(value.detach().cpu()) * batch
        if (
            getattr(args, "is_main_process", True)
            and args.log_interval > 0
            and (step == 1 or step % args.log_interval == 0)
        ):
            diagnostic_logits = output.get("t0_mask_logits", output["mask_logits"])
            diagnostic_prediction = torch.sigmoid(diagnostic_logits.detach()) >= 0.5
            diagnostic_truth = mask >= 0.5
            true_positive = (diagnostic_prediction & diagnostic_truth).sum()
            false_positive = (diagnostic_prediction & ~diagnostic_truth).sum()
            false_negative = (~diagnostic_prediction & diagnostic_truth).sum()
            precision = true_positive / (true_positive + false_positive).clamp_min(1)
            recall = true_positive / (true_positive + false_negative).clamp_min(1)
            f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
            print(
                f"train epoch={epoch:03d} step={step:04d}/{len(loader):04d} "
                f"loss={losses['loss_total'].detach().item():.4f} "
                f"t0_F1={f1.item():.4f} "
                f"pred_pos={diagnostic_prediction.float().mean().item():.4f} "
                f"target_pos={diagnostic_truth.float().mean().item():.4f} "
                f"scale={scaler.get_scale():.0f}"
            )
    if dist.is_initialized():
        packed = torch.tensor(
            [*(sums[key] for key in LOSS_KEYS), float(samples)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        samples = int(packed[-1].item())
        sums = {
            key: float(packed[index].item()) for index, key in enumerate(LOSS_KEYS)
        }
    return {key: value / max(samples, 1) for key, value in sums.items()}


@torch.no_grad()
def validate(model, loader, args, device) -> Dict[str, float]:
    model.eval()
    metrics = BinaryMetricAccumulator()
    segmentation_loss = 0.0
    samples = 0
    for image_a, image_b, mask in loader:
        image_a = image_a.to(device, non_blocking=True).float()
        image_b = image_b.to(device, non_blocking=True).float()
        mask = mask.to(device, non_blocking=True).float().unsqueeze(1)
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
        with torch.autocast(device_type="cuda", enabled=args.amp, dtype=amp_dtype):
            output = model(
                image_a,
                image_b,
                mode="inference",
                solver=args.solver,
                solver_steps=args.solver_steps,
            )
            logits = output["mask_logits"]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, mask)
            loss = loss + dice_loss_with_logits(logits, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("validation segmentation loss became NaN/Inf")
        metrics.update(logits, mask)
        batch = image_a.shape[0]
        samples += batch
        segmentation_loss += float(loss.cpu()) * batch
    if dist.is_initialized() and getattr(args, "distributed_validation", False):
        packed = torch.tensor(
            [
                segmentation_loss,
                float(samples),
                float(metrics.true_positive),
                float(metrics.false_positive),
                float(metrics.false_negative),
                float(metrics.true_negative),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        segmentation_loss = float(packed[0].item())
        samples = int(packed[1].item())
        metrics.true_positive = int(packed[2].item())
        metrics.false_positive = int(packed[3].item())
        metrics.false_negative = int(packed[4].item())
        metrics.true_negative = int(packed[5].item())
    return {"val_loss": segmentation_loss / max(samples, 1), **metrics.scores()}


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA because the official VRWKV encoder uses the WKV CUDA kernel")
    if args.resume and args.pretrain_dir:
        raise ValueError("--resume and --pretrain_dir are mutually exclusive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    else:
        local_rank = 0
        rank = 0
    is_main_process = rank == 0
    args.is_main_process = is_main_process
    args.world_size = world_size
    seed_everything(args.seed + rank)
    resume_payload = load_resume_payload(args.resume) if args.resume else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process:
        config = {
            key: value
            for key, value in vars(args).items()
            if key != "is_main_process"
        }
        with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
    if distributed:
        dist.barrier()

    train_dataset = ChangeDetectionDataset(
        args.data_dir, "train", return_geometry=True, geometry_target=args.geometry_target
    )
    val_dataset = ChangeDetectionDataset(
        args.data_dir, args.validation_split, return_geometry=False
    )
    if args.max_train_samples > 0:
        train_dataset = Subset(
            train_dataset, range(min(args.max_train_samples, len(train_dataset)))
        )
    if args.max_val_samples > 0:
        val_dataset = Subset(
            val_dataset, range(min(args.max_val_samples, len(val_dataset)))
        )
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        if distributed
        else None
    )
    loader_generator = torch.Generator().manual_seed(args.seed + rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        generator=loader_generator,
    )
    distributed_validation = distributed and bool(
        getattr(args, "distributed_validation", False)
    )
    val_sampler = (
        DistributedEvalSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
        )
        if distributed_validation
        else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if len(train_loader) == 0:
        raise RuntimeError("training loader is empty; reduce batch_size or check data_dir")

    model = build_model_from_args(args)
    if resume_payload is not None:
        load_checkpoint_flexible(
            model, resume_payload, require_encoder=True, verbose=is_main_process
        )
    elif args.pretrain_dir:
        if not Path(args.pretrain_dir).is_file():
            raise FileNotFoundError(f"pretrained checkpoint not found: {args.pretrain_dir}")
        load_checkpoint_flexible(
            model, args.pretrain_dir, require_encoder=True, verbose=is_main_process
        )
    available = torch.cuda.device_count()
    device_count = world_size if distributed else min(max(args.devices, 1), available)
    device_ids = list(range(device_count))
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=getattr(args, "ddp_find_unused_parameters", True),
        )
    elif device_count > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0.0 else None
    if ema is not None and resume_payload is not None:
        restore_ema_from_companion(
            ema,
            args.resume,
            resume_payload,
            verbose=is_main_process,
        )
    if is_main_process:
        print(
            f"dataset={args.dataset} train={len(train_dataset)} "
            f"monitor_split={args.validation_split} monitor_samples={len(val_dataset)}"
        )
        print(
            f"devices={device_ids} distributed={distributed} "
            f"parameters={count_parameters(model):,} ema_decay={args.ema_decay:.6f}"
        )

    parallel_wrappers = (nn.DataParallel, DistributedDataParallel)
    unwrapped = model.module if isinstance(model, parallel_wrappers) else model
    encoder_parameters = list(unwrapped.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    direct_parameters = []
    if getattr(unwrapped, "direct_fusion", False):
        direct_parameters = list(unwrapped.direct_decoder.parameters()) + [
            unwrapped.direct_fusion_scale
        ]
    direct_ids = {id(parameter) for parameter in direct_parameters}
    new_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in encoder_ids and id(parameter) not in direct_ids
    ]
    parameter_groups = [
        {"params": encoder_parameters, "lr": args.lr_encoder},
        {"params": new_parameters, "lr": args.lr_new},
    ]
    if direct_parameters:
        parameter_groups.append(
            {
                "params": direct_parameters,
                "lr": args.lr_direct if args.lr_direct is not None else args.lr_new,
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, len(train_loader), args.epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and args.amp_dtype == "float16",
        init_scale=args.amp_init_scale,
    )
    best = {"F1": -1.0, "IoU": -1.0}
    start_epoch = 1
    if resume_payload is not None:
        if resume_payload["optimizer"] is None or resume_payload["scheduler"] is None:
            raise ValueError("resume checkpoint does not contain optimizer/scheduler state")
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        if resume_payload.get("scaler") is not None:
            scaler.load_state_dict(resume_payload["scaler"])
        restored_best = resume_payload["best_metrics"]
        if not isinstance(restored_best, Mapping):
            raise TypeError("resume best_metrics must be a mapping")
        best.update({key: float(value) for key, value in restored_best.items() if key in best})
        completed_epoch = int(resume_payload["epoch"])
        start_epoch = completed_epoch + 1
        print(
            f"resumed checkpoint={args.resume} completed_epoch={completed_epoch} "
            f"next_epoch={start_epoch} best_F1={best['F1']:.6f} best_IoU={best['IoU']:.6f}"
        )
    fields = [
        "epoch",
        *LOSS_KEYS,
        "val_loss",
        "precision",
        "recall",
        "F1",
        "IoU",
        "accuracy",
        "pred_positive_rate",
        "target_positive_rate",
        "lr",
        "peak_memory_gib",
    ]
    metrics_path = output_dir / "metrics.csv"
    if is_main_process:
        prepare_metrics_file(
            metrics_path,
            fields,
            resume_epoch=(start_epoch - 1) if resume_payload is not None else None,
        )
    if distributed:
        dist.barrier()

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        torch.cuda.reset_peak_memory_stats(device)
        train_scores = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, args, device, epoch, ema
        )
        validation_model = ema.model if ema is not None else unwrapped
        if is_main_process or distributed_validation:
            validation_scores = validate(
                validation_model, val_loader, args, device
            )
        else:
            validation_scores = {}
        if distributed:
            dist.barrier()
        if not is_main_process:
            if distributed:
                dist.barrier()
            continue
        peak_memory_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        row = {
            "epoch": epoch,
            **train_scores,
            **validation_scores,
            "lr": optimizer.param_groups[1]["lr"],
            "peak_memory_gib": peak_memory_gib,
        }
        append_metrics_row(metrics_path, fields, row)
        print(
            f"epoch={epoch:03d} loss={row['loss_total']:.4f} "
            f"F1={row['F1']:.4f} IoU={row['IoU']:.4f} lr={row['lr']:.3e} "
            f"pred_pos={row['pred_positive_rate']:.4f} "
            f"target_pos={row['target_positive_rate']:.4f} "
            f"peak_mem={row['peak_memory_gib']:.2f}GiB"
        )
        improved_f1 = row["F1"] > best["F1"]
        improved_iou = row["IoU"] > best["IoU"]
        if improved_f1:
            best["F1"] = row["F1"]
        if improved_iou:
            best["IoU"] = row["IoU"]
        save_training_checkpoint(
            output_dir / "last.pth",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=vars(args),
            best_metrics=best,
        )
        if ema is not None:
            save_training_checkpoint(
                output_dir / "last_ema.pth",
                epoch=epoch,
                model=ema.model,
                optimizer=None,
                scheduler=None,
                scaler=None,
                config=vars(args),
                best_metrics=best,
            )
        if improved_f1:
            save_training_checkpoint(
                output_dir / "best_f1.pth",
                epoch=epoch,
                model=validation_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=vars(args),
                best_metrics=best,
            )
        if improved_iou:
            save_training_checkpoint(
                output_dir / "best_iou.pth",
                epoch=epoch,
                model=validation_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=vars(args),
                best_metrics=best,
            )
        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
