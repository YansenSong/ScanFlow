"""Evaluate whether a trained ScanFlow checkpoint actually learned motion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import NPZMotionFieldDataset, build_patch_targets
from test.common import classification_counts, load_checkpoint, prf, resolve_device


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=2048)
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    p.add_argument("--min-f1", type=float, default=0.80)
    p.add_argument("--max-static-speed", type=float, default=0.15)
    p.add_argument("--min-zero-baseline-gain", type=float, default=0.20,
                   help="Require model velocity EPE to beat zero-velocity baseline by this fraction.")
    args = p.parse_args()

    device = resolve_device(args.device)
    model, cfg, ckpt = load_checkpoint(args.checkpoint, device)
    dataset = NPZMotionFieldDataset(args.data)
    if dataset.lidar_history.shape[1:] != (cfg.num_frames, cfg.num_beams):
        raise ValueError("Dataset K/H do not match checkpoint ModelConfig.")

    n = min(len(dataset), args.max_samples)
    loader = DataLoader(Subset(dataset, range(n)), batch_size=args.batch_size, shuffle=False)

    tp = fp = fn = 0
    epe_sum = zero_epe_sum = 0.0
    dyn_count = 0
    static_speed_sum = 0.0
    static_count = 0

    for batch in loader:
        lidar = batch["lidar_history"].to(device)
        odom = batch["odom_history"].to(device)
        targets = build_patch_targets(
            batch["beam_velocity"],
            batch["beam_dynamic"],
            batch["beam_valid"],
            cfg.num_patches,
            cfg.max_motion_speed,
        )
        out = model(lidar, odom)

        tpi, fpi, fni = classification_counts(
            out["confidence"].cpu(),
            targets["confidence_target"],
            targets["patch_valid"],
            args.confidence_threshold,
        )
        tp += tpi; fp += fpi; fn += fni

        dyn = targets["velocity_weight"].squeeze(-1) > 0
        if dyn.any():
            pred_cpu = out["velocity"].cpu()
            target_vel = targets["velocity_target"]
            epe = (pred_cpu - target_vel).norm(dim=-1)
            zero_epe = target_vel.norm(dim=-1)
            epe_sum += float(epe[dyn].sum())
            zero_epe_sum += float(zero_epe[dyn].sum())
            dyn_count += int(dyn.sum())

        gt_dyn = targets["confidence_target"].squeeze(-1) >= 0.5
        static = (~gt_dyn) & targets["patch_valid"]
        if static.any():
            static_speed_sum += float(out["velocity"].cpu().norm(dim=-1)[static].sum())
            static_count += int(static.sum())

    precision, recall, f1 = prf(tp, fp, fn)
    epe = epe_sum / max(dyn_count, 1)
    zero_epe = zero_epe_sum / max(dyn_count, 1)
    gain = 1.0 - epe / max(zero_epe, 1e-8)
    static_speed = static_speed_sum / max(static_count, 1)

    print(f"checkpoint epoch : {ckpt.get('epoch', 'unknown')}")
    print(f"samples          : {n}")
    print(f"precision/recall : {precision:.3f} / {recall:.3f}")
    print(f"dynamic F1       : {f1:.3f}")
    print(f"velocity EPE     : {epe:.3f} m/s")
    print(f"zero baseline EPE: {zero_epe:.3f} m/s")
    print(f"EPE gain vs zero : {100.0 * gain:.1f}%")
    print(f"static speed     : {static_speed:.3f} m/s")

    passed = (
        dyn_count > 0
        and f1 >= args.min_f1
        and static_speed <= args.max_static_speed
        and gain >= args.min_zero_baseline_gain
    )
    print("PASS" if passed else "FAIL",
          "- motion must be detected, beat zero velocity, and stay quiet on static patches.")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
