"""
train.py

Training pipeline for DynamicLiDARNetwork in model.py.

Supervision philosophy
----------------------
The network predicts a detection-free patch motion field:

    velocity   : [B, P, 2]   -> (vx, vy) in CURRENT robot frame
    confidence : [B, P, 1]   -> probability that the patch contains
                                 meaningful dynamic motion

The recommended dataset stores CURRENT-FRAME per-beam ground truth:

    lidar_history : [N, K, H]
    odom_history  : [N, K, 3]
    beam_velocity : [N, H, 2]
    beam_dynamic  : [N, H]
    beam_valid    : [N, H]      optional

where beam_velocity is the GT velocity of the surface hit by each beam,
expressed in the CURRENT robot coordinate frame.

The script converts beam-level GT to patch-level GT:
    [H] -> [P, W]

For each patch:
    confidence_target = dynamic-beam fraction among valid beams
    velocity_target   = confidence-weighted mean dynamic velocity

Loss
----
L = lambda_conf * focal_BCE(conf_pred, conf_gt)
  + lambda_vel  * weighted_Huber(v_pred, v_gt)
  + lambda_static * static_velocity_regularization
  + lambda_smooth * confidence-aware spatial smoothness

Velocity regression is strongly weighted only where GT dynamic confidence
is nonzero, so static patches do not dominate the training signal.

NPZ dataset format
------------------
One .npz file should contain:

    lidar_history : float32 [N,K,H]
    odom_history  : float32 [N,K,3]
    beam_velocity : float32 [N,H,2]
    beam_dynamic  : float32/bool [N,H]
    beam_valid    : float32/bool [N,H]   (optional)

Example:
    python train.py \
        --data train_data.npz \
        --epochs 100 \
        --batch-size 64 \
        --save-dir checkpoints

Smoke test:
    python train.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from model import DynamicLiDARNetwork, ModelConfig


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NPZMotionFieldDataset(Dataset):
    """
    In-memory .npz dataset.

    Required arrays
    ---------------
    lidar_history : [N,K,H]
    odom_history  : [N,K,3]
    beam_velocity : [N,H,2]
    beam_dynamic  : [N,H]

    Optional
    --------
    beam_valid    : [N,H]
        Defaults to finite lidar in the CURRENT scan.
    """

    def __init__(self, npz_path: str):
        super().__init__()

        data = np.load(npz_path, allow_pickle=False)

        required = [
            "lidar_history",
            "odom_history",
            "beam_velocity",
            "beam_dynamic",
        ]
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
            current_scan = self.lidar_history[:, -1]
            self.beam_valid = np.isfinite(current_scan).astype(np.float32)

        N = self.lidar_history.shape[0]

        if self.odom_history.shape[0] != N:
            raise ValueError("odom_history N does not match lidar_history N.")
        if self.beam_velocity.shape[0] != N:
            raise ValueError("beam_velocity N does not match lidar_history N.")
        if self.beam_dynamic.shape[0] != N:
            raise ValueError("beam_dynamic N does not match lidar_history N.")
        if self.beam_valid.shape[0] != N:
            raise ValueError("beam_valid N does not match lidar_history N.")

        if self.lidar_history.ndim != 3:
            raise ValueError("lidar_history must be [N,K,H].")
        if self.odom_history.ndim != 3 or self.odom_history.shape[-1] != 3:
            raise ValueError("odom_history must be [N,K,3].")
        if self.beam_velocity.ndim != 3 or self.beam_velocity.shape[-1] != 2:
            raise ValueError("beam_velocity must be [N,H,2].")
        if self.beam_dynamic.ndim != 2:
            raise ValueError("beam_dynamic must be [N,H].")
        if self.beam_valid.ndim != 2:
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


# ---------------------------------------------------------------------------
# Patch target construction
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_patch_targets(
    beam_velocity: torch.Tensor,
    beam_dynamic: torch.Tensor,
    beam_valid: torch.Tensor,
    num_patches: int,
    max_speed: float,
    min_dynamic_fraction_for_velocity: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """
    Convert current-frame beam-level GT to patch-level motion-field GT.

    Inputs
    ------
    beam_velocity : [B,H,2]
    beam_dynamic  : [B,H], 0/1 or soft
    beam_valid    : [B,H], 0/1

    Outputs
    -------
    velocity_target    : [B,P,2]
    confidence_target  : [B,P,1]
    velocity_weight    : [B,P,1]
    patch_valid        : [B,P]

    Notes
    -----
    - confidence_target is the dynamic-beam fraction among valid beams.
    - velocity_target is the weighted mean of dynamic beam velocities.
    """
    if beam_velocity.ndim != 3 or beam_velocity.shape[-1] != 2:
        raise ValueError("beam_velocity must be [B,H,2].")
    if beam_dynamic.ndim != 2:
        raise ValueError("beam_dynamic must be [B,H].")
    if beam_valid.ndim != 2:
        raise ValueError("beam_valid must be [B,H].")

    B, H, _ = beam_velocity.shape
    if H % num_patches != 0:
        raise ValueError("H must be divisible by num_patches.")

    W = H // num_patches

    vel = torch.nan_to_num(beam_velocity, nan=0.0, posinf=0.0, neginf=0.0)
    speed = vel.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = torch.clamp(max_speed / speed, max=1.0)
    vel = vel * scale

    dyn = beam_dynamic.float().clamp(0.0, 1.0)
    valid = beam_valid.float().clamp(0.0, 1.0)

    vel = vel.view(B, num_patches, W, 2)
    dyn = dyn.view(B, num_patches, W)
    valid = valid.view(B, num_patches, W)

    valid_count = valid.sum(dim=-1, keepdim=True)  # [B,P,1]
    patch_valid = valid_count.squeeze(-1) > 0.0

    dyn_valid = dyn * valid
    dynamic_count = dyn_valid.sum(dim=-1, keepdim=True)

    confidence_target = dynamic_count / valid_count.clamp_min(1.0)
    confidence_target = confidence_target.clamp(0.0, 1.0)

    # Weighted mean dynamic velocity.
    numerator = (vel * dyn_valid.unsqueeze(-1)).sum(dim=2)  # [B,P,2]
    denominator = dynamic_count.clamp_min(1e-6)             # [B,P,1]
    velocity_target = numerator / denominator

    # Zero the velocity label if no dynamic evidence exists.
    velocity_target = torch.where(
        (dynamic_count > 0.0).expand_as(velocity_target),
        velocity_target,
        torch.zeros_like(velocity_target),
    )

    # Use a threshold only to decide whether velocity regression is reliable.
    # Confidence remains soft.
    velocity_weight = (
        confidence_target >= min_dynamic_fraction_for_velocity
    ).float() * confidence_target

    return {
        "velocity_target": velocity_target,
        "confidence_target": confidence_target,
        "velocity_weight": velocity_weight,
        "patch_valid": patch_valid,
    }


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

@dataclass
class LossConfig:
    lambda_conf: float = 1.0
    lambda_vel: float = 4.0
    lambda_static: float = 0.10
    lambda_smooth: float = 0.05

    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    huber_beta: float = 0.20

    # A patch below this GT confidence is treated as static for
    # velocity-suppression regularization.
    static_threshold: float = 0.02


def focal_binary_loss(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    alpha: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Focal BCE on probabilities.
    pred_prob and target: [B,P,1]
    """
    p = pred_prob.clamp(eps, 1.0 - eps)
    y = target

    bce = -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))
    p_t = y * p + (1.0 - y) * (1.0 - p)
    alpha_t = y * alpha + (1.0 - y) * (1.0 - alpha)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce

    if valid_mask is not None:
        m = valid_mask.unsqueeze(-1).float()
        return (loss * m).sum() / m.sum().clamp_min(1.0)

    return loss.mean()


def weighted_huber_velocity_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    pred/target : [B,P,2]
    weight      : [B,P,1]
    """
    raw = F.smooth_l1_loss(
        pred,
        target,
        reduction="none",
        beta=beta,
    ).mean(dim=-1, keepdim=True)

    return (raw * weight).sum() / weight.sum().clamp_min(1.0)


def static_velocity_regularization(
    pred_velocity: torch.Tensor,
    confidence_target: torch.Tensor,
    patch_valid: torch.Tensor,
    static_threshold: float,
) -> torch.Tensor:
    """
    Suppress arbitrary velocity predictions on confidently static patches.
    """
    static = (
        (confidence_target.squeeze(-1) <= static_threshold)
        & patch_valid
    )
    if not static.any():
        return pred_velocity.new_zeros(())

    speed = pred_velocity.norm(dim=-1)
    return speed[static].mean()


def confidence_aware_spatial_smoothness(
    pred_velocity: torch.Tensor,
    confidence_target: torch.Tensor,
    patch_valid: torch.Tensor,
    circular: bool = True,
) -> torch.Tensor:
    """
    Mild smoothness regularizer between neighboring angular patches.

    It is weighted more heavily when both neighboring patches are likely dynamic.
    This term is intentionally weak because different objects can sit in
    adjacent patches.
    """
    v = pred_velocity
    c = confidence_target.squeeze(-1)

    v_next = torch.roll(v, shifts=-1, dims=1)
    c_next = torch.roll(c, shifts=-1, dims=1)
    valid_next = torch.roll(patch_valid, shifts=-1, dims=1)

    diff = (v - v_next).norm(dim=-1)
    pair_weight = torch.minimum(c, c_next)
    pair_valid = patch_valid & valid_next

    if not circular:
        pair_valid[:, -1] = False

    weight = pair_weight * pair_valid.float()
    return (diff * weight).sum() / weight.sum().clamp_min(1.0)


def motion_field_loss(
    output: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    cfg: LossConfig,
) -> Dict[str, torch.Tensor]:
    pred_v = output["velocity"]
    pred_c = output["confidence"]

    gt_v = targets["velocity_target"]
    gt_c = targets["confidence_target"]
    vel_w = targets["velocity_weight"]
    patch_valid = targets["patch_valid"]

    loss_conf = focal_binary_loss(
        pred_prob=pred_c,
        target=gt_c,
        valid_mask=patch_valid,
        gamma=cfg.focal_gamma,
        alpha=cfg.focal_alpha,
    )

    loss_vel = weighted_huber_velocity_loss(
        pred=pred_v,
        target=gt_v,
        weight=vel_w,
        beta=cfg.huber_beta,
    )

    loss_static = static_velocity_regularization(
        pred_velocity=pred_v,
        confidence_target=gt_c,
        patch_valid=patch_valid,
        static_threshold=cfg.static_threshold,
    )

    loss_smooth = confidence_aware_spatial_smoothness(
        pred_velocity=pred_v,
        confidence_target=gt_c,
        patch_valid=patch_valid,
        circular=True,
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


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

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

    # Target construction
    min_dynamic_fraction_for_velocity: float = 0.05


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(
    output: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    threshold: float = 0.5,
) -> Dict[str, float]:
    pred_c = output["confidence"].squeeze(-1)
    gt_c = targets["confidence_target"].squeeze(-1)
    patch_valid = targets["patch_valid"]

    pred_dyn = (pred_c >= threshold) & patch_valid
    gt_dyn = (gt_c >= threshold) & patch_valid

    tp = (pred_dyn & gt_dyn).sum().item()
    fp = (pred_dyn & ~gt_dyn & patch_valid).sum().item()
    fn = (~pred_dyn & gt_dyn & patch_valid).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    dyn_mask = targets["velocity_weight"].squeeze(-1) > 0
    if dyn_mask.any():
        vel_epe = (
            output["velocity"] - targets["velocity_target"]
        ).norm(dim=-1)[dyn_mask].mean().item()
    else:
        vel_epe = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "velocity_epe": vel_epe,
    }


# ---------------------------------------------------------------------------
# Epoch loops
# ---------------------------------------------------------------------------

def _move_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        k: v.to(device, non_blocking=True)
        for k, v in batch.items()
    }


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

    totals = {
        "loss": 0.0,
        "confidence": 0.0,
        "velocity": 0.0,
        "static": 0.0,
        "smooth": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "velocity_epe": 0.0,
    }
    n_batches = 0

    for batch in loader:
        batch = _move_batch(batch, device)

        targets = build_patch_targets(
            beam_velocity=batch["beam_velocity"],
            beam_dynamic=batch["beam_dynamic"],
            beam_valid=batch["beam_valid"],
            num_patches=model_cfg.num_patches,
            max_speed=model_cfg.max_motion_speed,
            min_dynamic_fraction_for_velocity=(
                train_cfg.min_dynamic_fraction_for_velocity
            ),
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            output = model(
                lidar_history=batch["lidar_history"],
                odom_history=batch["odom_history"],
                return_aux=False,
            )

            losses = motion_field_loss(
                output=output,
                targets=targets,
                cfg=loss_cfg,
            )

            if training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_cfg.grad_clip,
                )
                optimizer.step()

        metrics = compute_metrics(output, targets)

        totals["loss"] += float(losses["total"].detach().item())
        totals["confidence"] += float(losses["confidence"].detach().item())
        totals["velocity"] += float(losses["velocity"].detach().item())
        totals["static"] += float(losses["static"].detach().item())
        totals["smooth"] += float(losses["smooth"].detach().item())

        for k, v in metrics.items():
            totals[k] += float(v)

        n_batches += 1

    if n_batches == 0:
        return totals

    return {
        k: v / n_batches
        for k, v in totals.items()
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: DynamicLiDARNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    loss_cfg: LossConfig,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "best_val_loss": best_val_loss,
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "loss_config": asdict(loss_cfg),
    }
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    data_path: str,
    save_dir: str,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    loss_cfg: LossConfig,
) -> None:
    seed_everything(train_cfg.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device:", device)

    dataset = NPZMotionFieldDataset(data_path)

    # Basic shape compatibility checks.
    _, K, H = dataset.lidar_history.shape
    if K != model_cfg.num_frames:
        raise ValueError(
            f"Dataset K={K}, but model num_frames={model_cfg.num_frames}"
        )
    if H != model_cfg.num_beams:
        raise ValueError(
            f"Dataset H={H}, but model num_beams={model_cfg.num_beams}"
        )

    n_total = len(dataset)
    n_val = max(1, int(round(n_total * train_cfg.val_fraction)))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(train_cfg.seed)
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=generator,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = DynamicLiDARNetwork(model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(train_cfg.epochs, 1),
        eta_min=train_cfg.learning_rate * 0.05,
    )

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": asdict(model_cfg),
                "train": asdict(train_cfg),
                "loss": asdict(loss_cfg),
            },
            f,
            indent=2,
        )

    best_val_loss = float("inf")

    for epoch in range(1, train_cfg.epochs + 1):
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            loss_cfg=loss_cfg,
            device=device,
        )

        with torch.no_grad():
            val_stats = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                loss_cfg=loss_cfg,
                device=device,
            )

        scheduler.step()

        print(
            f"[{epoch:03d}/{train_cfg.epochs:03d}] "
            f"train={train_stats['loss']:.4f} "
            f"val={val_stats['loss']:.4f} | "
            f"val_f1={val_stats['f1']:.3f} "
            f"vel_epe={val_stats['velocity_epe']:.3f} m/s"
        )

        # Best checkpoint.
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            save_checkpoint(
                save_path / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                model_cfg,
                train_cfg,
                loss_cfg,
            )

        if epoch % train_cfg.save_every == 0:
            save_checkpoint(
                save_path / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                model_cfg,
                train_cfg,
                loss_cfg,
            )

    save_checkpoint(
        save_path / "last.pt",
        model,
        optimizer,
        scheduler,
        train_cfg.epochs,
        best_val_loss,
        model_cfg,
        train_cfg,
        loss_cfg,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _make_smoke_npz(path: Path) -> None:
    """
    Small synthetic dataset to verify tensor flow and backward pass.
    This is NOT physically realistic training data.
    """
    N, K, H = 12, 6, 720

    rng = np.random.default_rng(0)

    lidar = rng.uniform(0.5, 7.0, size=(N, K, H)).astype(np.float32)

    odom = np.zeros((N, K, 3), dtype=np.float32)
    odom[:, :, 0] = np.linspace(0.0, 0.25, K, dtype=np.float32)
    odom[:, :, 2] = np.linspace(0.0, 0.05, K, dtype=np.float32)

    beam_velocity = np.zeros((N, H, 2), dtype=np.float32)
    beam_dynamic = np.zeros((N, H), dtype=np.float32)
    beam_valid = np.ones((N, H), dtype=np.float32)

    # Put a fake moving object over several neighboring beams.
    for n in range(N):
        start = 320 + (n % 8)
        end = start + 24
        beam_dynamic[n, start:end] = 1.0
        beam_velocity[n, start:end, 0] = 0.15
        beam_velocity[n, start:end, 1] = 0.55

    np.savez_compressed(
        path,
        lidar_history=lidar,
        odom_history=odom,
        beam_velocity=beam_velocity,
        beam_dynamic=beam_dynamic,
        beam_valid=beam_valid,
    )


def smoke_test() -> None:
    tmp = Path("/tmp/dynamic_lidar_smoke.npz")
    _make_smoke_npz(tmp)

    seed_everything(0)

    model_cfg = ModelConfig(
        num_frames=6,
        num_beams=720,
        num_patches=36,
        d_model=128,
    )
    train_cfg = TrainConfig(
        epochs=1,
        batch_size=4,
        val_fraction=0.25,
        num_workers=0,
    )
    loss_cfg = LossConfig()

    ds = NPZMotionFieldDataset(str(tmp))
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = DynamicLiDARNetwork(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    stats = run_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        loss_cfg=loss_cfg,
        device=torch.device("cpu"),
    )

    print("train.py smoke test passed")
    for k, v in stats.items():
        print(f"{k:14s}: {v:.6f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--beams", type=int, default=720)
    parser.add_argument("--patches", type=int, default=36)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--max-motion-speed", type=float, default=3.0)

    parser.add_argument("--smoke-test", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        smoke_test()
        return

    if args.data is None:
        raise SystemExit(
            "--data is required unless --smoke-test is used."
        )

    model_cfg = ModelConfig(
        num_frames=args.frames,
        num_beams=args.beams,
        num_patches=args.patches,
        d_model=args.d_model,
        max_motion_speed=args.max_motion_speed,
    )

    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.workers,
        seed=args.seed,
    )

    loss_cfg = LossConfig()

    train(
        data_path=args.data,
        save_dir=args.save_dir,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        loss_cfg=loss_cfg,
    )


if __name__ == "__main__":
    main()
