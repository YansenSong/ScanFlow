"""A4: moving robot + moving obstacle.

A single visible moving circle has a fixed world velocity while robot ego
speed/turn rate changes. The network should preserve dynamic detection and
recover a useful velocity estimate largely independent of ego motion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_dataset import Circle
from test.common import (
    classification_counts,
    constant_control_odom,
    default_static_scene,
    infer_one,
    load_checkpoint,
    patch_targets_from_scene,
    prf,
    render_history,
    resolve_device,
    sensor_config,
)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    p.add_argument("--min-recall", type=float, default=0.80)
    p.add_argument("--max-epe", type=float, default=0.30)
    p.add_argument("--min-zero-baseline-gain", type=float, default=0.20)
    args = p.parse_args()

    device = resolve_device(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    gen_cfg = sensor_config(cfg)
    segments, static_circles = default_static_scene()

    moving = Circle(
        center_current=np.array([3.0, 0.35], dtype=np.float64),
        radius=0.40,
        velocity=np.array([0.0, 0.65], dtype=np.float64),
        dynamic=True,
        object_id=100,
    )
    circles = list(static_circles) + [moving]

    scenarios = [
        ("robot-static", 0.00, 0.00),
        ("slow", 0.25, 0.00),
        ("fast", 0.80, 0.00),
        ("turning", 0.45, 0.60),
        ("fast-turning", 0.80, 0.90),
    ]

    all_tp = all_fp = all_fn = 0
    epe_values = []
    zero_values = []

    print("A4: moving robot + moving obstacle")
    for name, v_robot, w_robot in scenarios:
        odom = constant_control_odom(cfg.num_frames, gen_cfg.scan_dt, v_robot, w_robot)
        lidar, hit_ids = render_history(gen_cfg, odom, segments, circles)
        targets, _, beam_dynamic, _ = patch_targets_from_scene(cfg, hit_ids[-1], circles)

        dyn_beams = int((beam_dynamic > 0.5).sum())
        if dyn_beams == 0:
            raise RuntimeError(f"Scenario {name} unexpectedly occluded the moving obstacle.")

        out = infer_one(model, lidar, odom, device)
        conf_cpu = out["confidence"].cpu()
        vel_cpu = out["velocity"].cpu()

        tp, fp, fn = classification_counts(
            conf_cpu,
            targets["confidence_target"],
            targets["patch_valid"],
            args.confidence_threshold,
        )
        all_tp += tp; all_fp += fp; all_fn += fn
        precision, recall, f1 = prf(tp, fp, fn)

        dyn = targets["velocity_weight"].squeeze(-1) > 0
        pred_epe = (vel_cpu - targets["velocity_target"]).norm(dim=-1)[dyn]
        zero_epe = targets["velocity_target"].norm(dim=-1)[dyn]
        epe = float(pred_epe.mean()) if pred_epe.numel() else float("inf")
        zero = float(zero_epe.mean()) if zero_epe.numel() else float("inf")
        gain = 1.0 - epe / max(zero, 1e-8)
        epe_values.append(epe)
        zero_values.append(zero)

        print(f"{name:14s} v={v_robot:.2f} w={w_robot:.2f} dyn_beams={dyn_beams:3d} | "
              f"recall={recall:.3f} F1={f1:.3f} EPE={epe:.3f} m/s "
              f"zero={zero:.3f} gain={100.0*gain:+.1f}%")

    precision, recall, f1 = prf(all_tp, all_fp, all_fn)
    mean_epe = float(np.mean(epe_values))
    mean_zero = float(np.mean(zero_values))
    gain = 1.0 - mean_epe / max(mean_zero, 1e-8)

    print(f"aggregate precision/recall/F1: {precision:.3f} / {recall:.3f} / {f1:.3f}")
    print(f"mean velocity EPE           : {mean_epe:.3f} m/s")
    print(f"mean zero-baseline EPE      : {mean_zero:.3f} m/s")
    print(f"gain vs zero                : {100.0*gain:.1f}%")

    passed = (
        recall >= args.min_recall
        and mean_epe <= args.max_epe
        and gain >= args.min_zero_baseline_gain
    )
    print("PASS" if passed else "FAIL",
          "- dynamic motion should remain observable while robot ego-motion changes.")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
