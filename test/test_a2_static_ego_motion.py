"""A2: static world + moving robot.

The desired behavior is low false-dynamic confidence and near-zero predicted
world/object motion even when the robot translates and rotates.
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

from test.common import (
    constant_control_odom,
    default_static_scene,
    infer_one,
    load_checkpoint,
    resolve_device,
    sensor_config,
    render_history,
    to_numpy,
)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    p.add_argument("--max-static-speed", type=float, default=0.15)
    p.add_argument("--max-false-dynamic-rate", type=float, default=0.10)
    p.add_argument("--min-alignment-gain", type=float, default=1.20,
                   help="raw range MAD / ego-aligned MAD on commonly valid beams.")
    args = p.parse_args()

    device = resolve_device(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    gen_cfg = sensor_config(cfg)
    segments, circles = default_static_scene()

    scenarios = [
        ("slow-straight", 0.20, 0.00),
        ("fast-straight", 0.80, 0.00),
        ("turning", 0.45, 0.60),
        ("fast-turning", 0.80, 0.90),
    ]

    worst_speed = 0.0
    worst_fp_rate = 0.0
    alignment_gains = []

    print("A2: static world + robot ego-motion")
    for name, v_robot, w_robot in scenarios:
        odom = constant_control_odom(cfg.num_frames, gen_cfg.scan_dt, v_robot, w_robot)
        lidar, _ = render_history(gen_cfg, odom, segments, circles)
        out = infer_one(model, lidar, odom, device, return_aux=True)

        valid = out["current_valid"][0].bool()
        speed = out["velocity"][0].norm(dim=-1)
        conf = out["confidence"][0, :, 0]
        mean_speed = float(speed[valid].mean().cpu()) if valid.any() else float("nan")
        fp_rate = float((conf[valid] >= args.confidence_threshold).float().mean().cpu()) if valid.any() else 1.0

        aligned = to_numpy(out["aligned_range"][0])
        beam_valid = to_numpy(out["beam_valid"][0]).astype(bool)
        cur = aligned[-1]
        aligned_errs = []
        raw_errs = []
        for k in range(cfg.num_frames - 1):
            m_aligned = beam_valid[k] & beam_valid[-1]
            if np.any(m_aligned):
                aligned_errs.append(float(np.mean(np.abs(aligned[k, m_aligned] - cur[m_aligned]))))
            raw_k_valid = np.isfinite(lidar[k])
            raw_cur_valid = np.isfinite(lidar[-1])
            m_raw = raw_k_valid & raw_cur_valid
            if np.any(m_raw):
                raw_errs.append(float(np.mean(np.abs(lidar[k, m_raw] - lidar[-1, m_raw]))))
        aligned_mad = float(np.mean(aligned_errs)) if aligned_errs else float("inf")
        raw_mad = float(np.mean(raw_errs)) if raw_errs else float("inf")
        gain = raw_mad / max(aligned_mad, 1e-8)
        alignment_gains.append(gain)

        worst_speed = max(worst_speed, mean_speed)
        worst_fp_rate = max(worst_fp_rate, fp_rate)
        print(f"{name:14s} v={v_robot:.2f} w={w_robot:.2f} | "
              f"static_speed={mean_speed:.3f} m/s false_dyn={fp_rate:.3f} "
              f"alignment_gain={gain:.2f}x")

    min_gain = min(alignment_gains)
    passed = (
        worst_speed <= args.max_static_speed
        and worst_fp_rate <= args.max_false_dynamic_rate
        and min_gain >= args.min_alignment_gain
    )
    print(f"worst static speed : {worst_speed:.3f} m/s")
    print(f"worst false dyn    : {worst_fp_rate:.3f}")
    print(f"minimum align gain : {min_gain:.2f}x")
    print("PASS" if passed else "FAIL",
          "- ego motion should align static geometry without creating false external motion.")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
