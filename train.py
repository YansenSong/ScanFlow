"""Training pipeline for ScanFlow's DynamicLiDARNetwork.

The confidence head is supervised as *dynamic presence probability* at patch
level, not as dynamic-beam fraction. This matches planner semantics: a small
moving object that occupies only a few beams should still be a dynamic token.
The dynamic-beam fraction is retained as an auxiliary diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from model import DynamicLiDARNetwork, ModelConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NPZMotionFieldDataset(Dataset):
    def __init__(self, npz_path: str):
        super().__init__()
        data = np.load(npz_path, allow_pickle=False)
        required = ["lidar_history", "odom_history", "beam_velocity", "beam_dynamic"]
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"Missing arrays in {npz_path}: {missing}")

        self.lidar_history = np.asarray(data["lidar_history"], dtype=np.float32)
        self.odom_history = np.asarray(data["odom_history"], dtype=np.float32)
        self.beam_velocity = np.asarray(data["beam_velocity"], dtype=np.float32)
        self.beam_dynamic = np.asarray(data["beam_dynamic"], dtype=np.float32)
        if "beam_valid" in data:
            self.beam_valid = np.asarray(data["beam_valid"], dtype=np.float32)
        else:
            cur = self.lidar_history[:, -1]
            self.beam_valid = np.isfinite(cur).astype(np.float32)

        if self.lidar_history.ndim != 3:
            raise ValueError("lidar_history must be [N,K,H].")
        N, K, H = self.lidar_history.shape
        if self.odom_history.shape != (N, K, 3):
            raise ValueError("odom_history must be [N,K,3] matching lidar_history.")
        if self.beam_velocity.shape != (N, H, 2):
            raise ValueError("beam_velocity must be [N,H,2].")
        if self.beam_dynamic.shape != (N, H):
            raise ValueError("beam_dynamic must be [N,H].")
        if self.beam_valid.shape != (N, H):
            raise ValueError("beam_valid must be [N,H].")

    def __len__(self) -> int:
        return int(self.lidar_history.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "lidar_history": torch.from_numpy(self.lidar_history[idx]),
            "odom_history": torch.from_numpy(self.odom_history[idx]),
            "beam_velocity": torch.from_numpy(self.beam_velocity[idx]),
            "beam_dynamic": torch.from_numpy(self.beam_dynamic[idx]),
            "beam_valid": torch.from_numpy(self.beam_valid[idx]),
        }


@torch.no_grad()
def build_patch_targets(
    beam_velocity: torch.Tensor,
    beam_dynamic: torch.Tensor,
    beam_valid: torch.Tensor,
    num_patches: int,
    max_speed: float,
    min_dynamic_fraction_for_velocity: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Convert current-frame beam labels into patch-level motion supervision.

    confidence_target is binary dynamic presence. dynamic_fraction is returned
    separately for diagnostics and optional reliability gating.
    """
    if beam_velocity.ndim != 3 or beam_velocity.shape[-1] != 2:
        raise ValueError("beam_velocity must be [B,H,2].")
    B, H, _ = beam_velocity.shape
    if beam_dynamic.shape != (B, H) or beam_valid.shape != (B, H):
        raise ValueError("beam_dynamic/beam_valid must be [B,H].")
    if H % num_patches != 0:
        raise ValueError("H must be divisible by num_patches.")

    W = H // num_patches
    vel = torch.nan_to_num(beam_velocity, nan=0.0, posinf=0.0, neginf=0.0)
    speed = vel.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    vel = vel * torch.clamp(max_speed / speed, max=1.0)
    dyn = beam_dynamic.float().clamp(0.0, 1.0)
    valid = beam_valid.float().clamp(0.0, 1.0)

    vel = vel.view(B, num_patches, W, 2)
    dyn = dyn.view(B, num_patches, W)
    valid = valid.view(B, num_patches, W)

    valid_count = valid.sum(dim=-1, keepdim=True)
    patch_valid = valid_count.squeeze(-1) > 0
    dyn_valid = dyn * valid
    dynamic_count = dyn_valid.sum(dim=-1, keepdim=True)
    dynamic_fraction = (dynamic_count / valid_count.clamp_min(1.0)).clamp(0.0, 1.0)

    confidence_target = (dynamic_count > 0.0).to(vel.dtype)

    numerator = (vel * dyn_valid[..., None]).sum(dim=2)
    velocity_target = numerator / dynamic_count.clamp_min(1e-6)
    velocity_target = torch.where(
        (dynamic_count > 0).expand_as(velocity_target),
        velocity_target,
        torch.zeros_like(velocity_target),
    )

    reliable = confidence_target.clone()
    if min_dynamic_fraction_for_velocity > 0:
        reliable = reliable * (dynamic_fraction >= min_dynamic_fraction_for_velocity).to(reliable.dtype)

    return {
        "velocity_target": velocity_target,
        "confidence_target": confidence_target,
        "dynamic_fraction": dynamic_fraction,
        "velocity_weight": reliable,
        "patch_valid": patch_valid,
    }


@dataclass
class LossConfig:
    # Dynamic patches are sparse.  Confidence must remain a first-class
    # objective instead of being numerically dwarfed by velocity regression.
    lambda_conf: float = 4.0
    lambda_vel: float = 4.0
    # The velocity head is otherwise unconstrained on the much larger set of
    # static patches and can learn arbitrary non-zero outputs there.
    lambda_static: float = 1.0
    lambda_smooth: float = 0.05
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    huber_beta: float = 0.20


def focal_binary_loss(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    gamma: float,
    alpha: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    p = pred_prob.clamp(eps, 1.0 - eps)
    y = target
    bce = -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))
    p_t = y * p + (1.0 - y) * (1.0 - p)
    alpha_t = y * alpha + (1.0 - y) * (1.0 - alpha)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    m = valid_mask[..., None].to(loss.dtype)
    return (loss * m).sum() / m.sum().clamp_min(1.0)


def weighted_huber_velocity_loss(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, beta: float
) -> torch.Tensor:
    raw = F.smooth_l1_loss(pred, target, reduction="none", beta=beta).mean(dim=-1, keepdim=True)
    return (raw * weight).sum() / weight.sum().clamp_min(1.0)


def static_velocity_regularization(
    pred_velocity: torch.Tensor, confidence_target: torch.Tensor, patch_valid: torch.Tensor
) -> torch.Tensor:
    static = (confidence_target.squeeze(-1) < 0.5) & patch_valid
    if not static.any():
        return pred_velocity.new_zeros(())
    return pred_velocity.norm(dim=-1)[static].mean()


def confidence_aware_spatial_smoothness(
    pred_velocity: torch.Tensor,
    confidence_target: torch.Tensor,
    patch_valid: torch.Tensor,
    circular: bool = True,
) -> torch.Tensor:
    v_next = torch.roll(pred_velocity, -1, dims=1)
    c = confidence_target.squeeze(-1)
    c_next = torch.roll(c, -1, dims=1)
    valid_next = torch.roll(patch_valid, -1, dims=1)
    pair_valid = patch_valid & valid_next
    if not circular:
        pair_valid[:, -1] = False
    weight = torch.minimum(c, c_next) * pair_valid.float()
    diff = (pred_velocity - v_next).norm(dim=-1)
    return (diff * weight).sum() / weight.sum().clamp_min(1.0)


def motion_field_loss(
    output: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], cfg: LossConfig
) -> Dict[str, torch.Tensor]:
    loss_conf = focal_binary_loss(
        output["confidence"],
        targets["confidence_target"],
        targets["patch_valid"],
        cfg.focal_gamma,
        cfg.focal_alpha,
    )
    loss_vel = weighted_huber_velocity_loss(
        output["velocity"], targets["velocity_target"], targets["velocity_weight"], cfg.huber_beta
    )
    loss_static = static_velocity_regularization(
        output["velocity"], targets["confidence_target"], targets["patch_valid"]
    )
    loss_smooth = confidence_aware_spatial_smoothness(
        output["velocity"], targets["confidence_target"], targets["patch_valid"], circular=True
    )
    total = (
        cfg.lambda_conf * loss_conf
        + cfg.lambda_vel * loss_vel
        + cfg.lambda_static * loss_static
        + cfg.lambda_smooth * loss_smooth
    )
    return {
        "total": total,
        "confidence": loss_conf,
        "velocity": loss_vel,
        "static": loss_static,
        "smooth": loss_smooth,
    }


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_fraction: float = 0.10
    num_workers: int = 0
    seed: int = 42
    save_every: int = 10
    min_dynamic_fraction_for_velocity: float = 0.0


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def metric_counts(output: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], threshold: float = 0.5):
    valid = targets["patch_valid"]
    pred_dyn = (output["confidence"].squeeze(-1) >= threshold) & valid
    gt_dyn = (targets["confidence_target"].squeeze(-1) >= 0.5) & valid
    tp = int((pred_dyn & gt_dyn).sum().item())
    fp = int((pred_dyn & ~gt_dyn & valid).sum().item())
    fn = int((~pred_dyn & gt_dyn & valid).sum().item())

    dyn = targets["velocity_weight"].squeeze(-1) > 0
    vel_sum = 0.0
    vel_count = int(dyn.sum().item())
    if vel_count:
        vel_sum = float((output["velocity"] - targets["velocity_target"]).norm(dim=-1)[dyn].sum().item())

    static = (~gt_dyn) & valid
    static_count = int(static.sum().item())
    static_sum = 0.0
    if static_count:
        static_sum = float(output["velocity"].norm(dim=-1)[static].sum().item())
    return tp, fp, fn, vel_sum, vel_count, static_sum, static_count


def run_epoch(
    model: DynamicLiDARNetwork,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    loss_cfg: LossConfig,
    device: torch.device,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sums = {k: 0.0 for k in ("loss", "confidence", "velocity", "static", "smooth")}
    n_batches = 0
    tp = fp = fn = vel_count = static_count = 0
    vel_sum = static_sum = 0.0

    for batch in loader:
        batch = _move_batch(batch, device)
        targets = build_patch_targets(
            batch["beam_velocity"],
            batch["beam_dynamic"],
            batch["beam_valid"],
            model_cfg.num_patches,
            model_cfg.max_motion_speed,
            train_cfg.min_dynamic_fraction_for_velocity,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(batch["lidar_history"], batch["odom_history"])
            losses = motion_field_loss(output, targets, loss_cfg)
            if training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()

        loss_sums["loss"] += float(losses["total"].detach().item())
        for k in ("confidence", "velocity", "static", "smooth"):
            loss_sums[k] += float(losses[k].detach().item())
        c = metric_counts(output, targets)
        tp += c[0]; fp += c[1]; fn += c[2]
        vel_sum += c[3]; vel_count += c[4]
        static_sum += c[5]; static_count += c[6]
        n_batches += 1

    denom = max(n_batches, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        **{k: v / denom for k, v in loss_sums.items()},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "velocity_epe": vel_sum / max(vel_count, 1),
        "static_speed": static_sum / max(static_count, 1),
    }


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, best_val_loss, model_cfg, train_cfg, loss_cfg):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_loss": best_val_loss,
            "model_config": asdict(model_cfg),
            "train_config": asdict(train_cfg),
            "loss_config": asdict(loss_cfg),
        },
        path,
    )


def train(data_path: str, save_dir: str, model_cfg: ModelConfig, train_cfg: TrainConfig, loss_cfg: LossConfig) -> None:
    seed_everything(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    dataset = NPZMotionFieldDataset(data_path)
    _, K, H = dataset.lidar_history.shape
    if K != model_cfg.num_frames or H != model_cfg.num_beams:
        raise ValueError("Dataset history/beam dimensions do not match ModelConfig.")
    if len(dataset) < 2:
        raise ValueError("Training requires at least two samples for train/validation split.")

    n_val = min(max(1, int(round(len(dataset) * train_cfg.val_fraction))), len(dataset) - 1)
    n_train = len(dataset) - n_val
    gen = torch.Generator().manual_seed(train_cfg.seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    common = dict(batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
                  pin_memory=torch.cuda.is_available(), drop_last=False)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)

    model = DynamicLiDARNetwork(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(train_cfg.epochs, 1), eta_min=train_cfg.learning_rate * 0.05
    )

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "config.json", "w", encoding="utf-8") as f:
        json.dump({"model": asdict(model_cfg), "train": asdict(train_cfg), "loss": asdict(loss_cfg)}, f, indent=2)

    best_val_loss = float("inf")
    for epoch in range(1, train_cfg.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, model_cfg, train_cfg, loss_cfg, device)
        with torch.no_grad():
            va = run_epoch(model, val_loader, None, model_cfg, train_cfg, loss_cfg, device)
        scheduler.step()
        print(
            f"[{epoch:03d}/{train_cfg.epochs:03d}] train={tr['loss']:.4f} val={va['loss']:.4f} | "
            f"val_f1={va['f1']:.3f} vel_epe={va['velocity_epe']:.3f} m/s "
            f"static_speed={va['static_speed']:.3f} m/s"
        )
        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            save_checkpoint(save_path / "best.pt", model, optimizer, scheduler, epoch, best_val_loss,
                            model_cfg, train_cfg, loss_cfg)
        if epoch % train_cfg.save_every == 0:
            save_checkpoint(save_path / f"epoch_{epoch:04d}.pt", model, optimizer, scheduler, epoch,
                            best_val_loss, model_cfg, train_cfg, loss_cfg)
    save_checkpoint(save_path / "last.pt", model, optimizer, scheduler, train_cfg.epochs,
                    best_val_loss, model_cfg, train_cfg, loss_cfg)


def _make_smoke_npz(path: Path) -> None:
    N, K, H = 12, 6, 720
    rng = np.random.default_rng(0)
    lidar = rng.uniform(0.5, 7.0, size=(N, K, H)).astype(np.float32)
    odom = np.zeros((N, K, 3), dtype=np.float32)
    odom[:, :, 0] = np.linspace(0.0, 0.25, K, dtype=np.float32)
    odom[:, :, 2] = np.linspace(0.0, 0.05, K, dtype=np.float32)
    beam_velocity = np.zeros((N, H, 2), dtype=np.float32)
    beam_dynamic = np.zeros((N, H), dtype=np.float32)
    beam_valid = np.ones((N, H), dtype=np.float32)
    for n in range(N):
        start = 320 + (n % 8)
        beam_dynamic[n, start:start + 4] = 1.0
        beam_velocity[n, start:start + 4] = [0.15, 0.55]
    np.savez_compressed(path, lidar_history=lidar, odom_history=odom,
                        beam_velocity=beam_velocity, beam_dynamic=beam_dynamic, beam_valid=beam_valid)


def smoke_test() -> None:
    tmp = Path("/tmp/scanflow_motion_smoke.npz")
    _make_smoke_npz(tmp)
    ds = NPZMotionFieldDataset(str(tmp))
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
    targets = build_patch_targets(batch["beam_velocity"], batch["beam_dynamic"], batch["beam_valid"], 36, 3.0)
    assert targets["confidence_target"].max().item() == 1.0
    assert targets["dynamic_fraction"].max().item() < 0.30
    model = DynamicLiDARNetwork(ModelConfig())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    stats = run_epoch(model, DataLoader(ds, batch_size=4), optimizer, ModelConfig(), TrainConfig(), LossConfig(), torch.device("cpu"))
    assert np.isfinite(stats["loss"])
    print("train.py smoke test passed")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--beams", type=int, default=720)
    p.add_argument("--patches", type=int, default=36)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--max-motion-speed", type=float, default=3.0)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test(); return
    if args.data is None:
        raise SystemExit("--data is required unless --smoke-test is used.")
    model_cfg = ModelConfig(num_frames=args.frames, num_beams=args.beams, num_patches=args.patches,
                            d_model=args.d_model, max_motion_speed=args.max_motion_speed)
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
                            num_workers=args.workers, seed=args.seed)
    train(args.data, args.save_dir, model_cfg, train_cfg, LossConfig())


if __name__ == "__main__":
    main()
