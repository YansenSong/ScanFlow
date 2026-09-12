"""Geometry-only hard cases for the ScanFlow v2 motion contract.

These cases are intentionally small and deterministic.  They diagnose whether
beam-level geometry preserves distinct motion, keeps a near dynamic return
separate from a nearby static surface, marks newly visible returns as
unsupported, and exposes tangential ambiguity through entropy/confidence.
They do not tune a checkpoint or claim a planner-level pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_dataset import GeneratorConfig, Circle, Segment, cast_lidar
from motion_estimator_v2 import MotionEstimatorV2
from test.common import render_history


def _times(cfg: GeneratorConfig) -> np.ndarray:
    return (np.arange(cfg.num_frames, dtype=np.float64) - (cfg.num_frames - 1)) * cfg.scan_dt


def _estimate(lidar: np.ndarray, odom: np.ndarray, times: np.ndarray):
    estimator = MotionEstimatorV2(score_mode="geometry", device="cpu", top_k=3, refine=True)
    output = estimator(lidar, odom, times)
    return {key: value[0].detach().cpu().numpy() for key, value in output.items()}


def _object_stats(output, hit_ids: np.ndarray, object_id: int):
    mask = hit_ids[-1] == object_id
    velocity = output["velocity"]
    support = output["motion_supported"]
    return {
        "beams": int(mask.sum()),
        "mean_velocity": velocity[mask].mean(axis=0).tolist() if mask.any() else None,
        "mean_speed_mps": float(np.linalg.norm(velocity[mask], axis=-1).mean()) if mask.any() else None,
        "supported_fraction": float(support[mask].mean()) if mask.any() else None,
        "mean_dynamic_probability": float(output["dynamic_probability"][mask].mean()) if mask.any() else None,
    }


def case_two_opposing_movers():
    cfg = GeneratorConfig(num_beams=360, lidar_noise_std=0.0, min_dynamic_beams=0)
    odom = np.zeros((cfg.num_frames, 3), dtype=np.float64)
    circles = [
        Circle(np.array([3.0, 0.45]), 0.35, np.array([0.0, -0.6]), True, 100),
        Circle(np.array([3.0, -0.45]), 0.35, np.array([0.0, 0.6]), True, 101),
    ]
    lidar, ids = render_history(cfg, odom, [], circles)
    output = _estimate(lidar, odom, _times(cfg))
    a, b = _object_stats(output, ids, 100), _object_stats(output, ids, 101)
    a["opposite_sign_ok"] = bool(a["mean_velocity"] and a["mean_velocity"][1] < -0.25)
    b["opposite_sign_ok"] = bool(b["mean_velocity"] and b["mean_velocity"][1] > 0.25)
    return {"object_100": a, "object_101": b,
            "not_averaged_to_zero": bool(a["opposite_sign_ok"] and b["opposite_sign_ok"])}


def case_static_dynamic_angular_neighborhood():
    cfg = GeneratorConfig(num_beams=360, lidar_noise_std=0.0, min_dynamic_beams=0)
    odom = np.zeros((cfg.num_frames, 3), dtype=np.float64)
    segments = [Segment(np.array([5.0, -1.0]), np.array([5.0, 1.0]), 0)]
    circles = [Circle(np.array([2.5, 0.0]), 0.35, np.array([0.0, 0.6]), True, 100)]
    lidar, ids = render_history(cfg, odom, segments, circles)
    output = _estimate(lidar, odom, _times(cfg))
    static, dynamic = _object_stats(output, ids, 0), _object_stats(output, ids, 100)
    static_speed = static["mean_speed_mps"] if static["mean_speed_mps"] is not None else float("nan")
    dynamic_speed = dynamic["mean_speed_mps"] if dynamic["mean_speed_mps"] is not None else float("nan")
    return {
        "static_segment": static,
        "near_dynamic_circle": dynamic,
        "same_angular_neighborhood_separated": bool(static_speed < 0.15 and dynamic_speed > 0.30),
    }


def case_occlusion_emergence():
    # At t=0 the near edge is inside range, while all earlier near edges are
    # beyond range.  There is no historical correspondence to fabricate.
    cfg = GeneratorConfig(num_beams=360, lidar_noise_std=0.0, min_dynamic_beams=0)
    odom = np.zeros((cfg.num_frames, 3), dtype=np.float64)
    circle = Circle(np.array([10.1, 0.0]), 0.2, np.array([-1.25, 0.0]), True, 100)
    lidar, ids = render_history(cfg, odom, [], [circle])
    output = _estimate(lidar, odom, _times(cfg))
    stats = _object_stats(output, ids, 100)
    mask = ids[-1] == 100
    stats["history_hit_counts"] = [int((ids[k] == 100).sum()) for k in range(cfg.num_frames)]
    stats["zero_velocity_without_support"] = bool(
        mask.any()
        and not output["motion_supported"][mask].any()
        and np.allclose(output["velocity"][mask], 0.0)
    )
    return stats


def case_tangential_ambiguity():
    cfg = GeneratorConfig(num_beams=360, lidar_noise_std=0.0, min_dynamic_beams=0)
    odom = np.zeros((cfg.num_frames, 3), dtype=np.float64)
    times = _times(cfg)
    lidar, ids = [], []
    angles = cfg.angle_min + np.arange(cfg.num_beams) * cfg.angle_increment
    for t in times:
        shift = np.array([0.6 * t, 0.0])
        segments = [Segment(np.array([-5.0, 2.0]) + shift, np.array([5.0, 2.0]) + shift, 100)]
        scan, hit_ids = cast_lidar(odom[len(lidar)], angles, segments, [], float(t), cfg)
        lidar.append(scan)
        ids.append(hit_ids)
    output = _estimate(np.asarray(lidar), odom, times)
    mask = np.asarray(ids[-1]) == 100
    entropy = float(output["candidate_entropy"][mask].mean()) if mask.any() else float("nan")
    confidence = float(output["motion_confidence"][mask].mean()) if mask.any() else float("nan")
    speed = float(np.linalg.norm(output["velocity"][mask], axis=-1).mean()) if mask.any() else float("nan")
    return {
        "beams": int(mask.sum()),
        "mean_speed_mps": speed,
        "mean_candidate_entropy": entropy,
        "mean_motion_confidence": confidence,
        "ambiguous_signal_ok": bool(mask.any() and speed < 0.10 and entropy > 4.5 and confidence < 0.10),
    }


def evaluate_hard_cases():
    return {
        "J1_two_opposing_movers": case_two_opposing_movers(),
        "J2_static_plus_dynamic_same_angular_neighborhood": case_static_dynamic_angular_neighborhood(),
        "J3_occlusion_emergence": case_occlusion_emergence(),
        "J4_tangential_ambiguity": case_tangential_ambiguity(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=Path, default=Path("artifacts/motion_v2/hard_cases.json"))
    args = parser.parse_args()
    report = evaluate_hard_cases()
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
