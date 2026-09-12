"""
visualize_sample.py

Visual inspection tool for the synthetic dynamic-LiDAR dataset.

Creates a 3-panel figure:

1) Left  : 6-frame raw LiDAR history in each frame's own robot coordinate system
2) Middle: 6-frame ego-motion-aligned history in the CURRENT robot frame
3) Right : CURRENT scan with
           - static hits as open circles
           - dynamic hits as filled circles
           - GT velocity arrows on dynamic hits

Why this matters
----------------
This figure is the quickest sanity check that:
    - raw historical scans contain both robot ego-motion and object motion,
    - after ego-motion compensation, static background becomes stable,
    - only moving objects keep residual motion,
    - beam_dynamic / beam_velocity labels are correct.

Usage
-----
A) Visualize a sample from an existing dataset:
    python visualize_sample.py --data train_data.npz --index 0 --output sample.png

B) Generate one fresh procedural sample on the fly:
    python visualize_sample.py --generate --output sample.png

Dependencies
------------
matplotlib, numpy
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Reuse dataset generation utilities.
from generate_dataset import (
    GeneratorConfig,
    generate_one_sample,
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def beam_angles(num_beams: int, angle_min: float, angle_increment: float) -> np.ndarray:
    return angle_min + np.arange(num_beams, dtype=np.float64) * angle_increment


def scan_to_local_xy(
    ranges: np.ndarray,              # [H]
    angles: np.ndarray,              # [H]
    range_min: float,
    range_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a single LiDAR scan to local-frame XY points.

    Returns
    -------
    xy     : [M,2]
    valid  : [H] bool
    """
    valid = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)
    r = ranges[valid]
    a = angles[valid]
    x = r * np.cos(a)
    y = r * np.sin(a)
    xy = np.stack([x, y], axis=-1) if r.size > 0 else np.zeros((0, 2), dtype=np.float64)
    return xy, valid


def transform_points_robot_to_world(
    xy_local: np.ndarray,            # [M,2]
    pose: np.ndarray,                # [3] = [x,y,yaw]
) -> np.ndarray:
    if xy_local.size == 0:
        return xy_local.copy()
    x, y, yaw = pose
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return xy_local @ rot.T + np.array([x, y], dtype=np.float64)


def transform_points_world_to_robot(
    xy_world: np.ndarray,            # [M,2]
    pose: np.ndarray,                # [3] = [x,y,yaw]
) -> np.ndarray:
    if xy_world.size == 0:
        return xy_world.copy()
    x, y, yaw = pose
    dx = xy_world - np.array([x, y], dtype=np.float64)
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    # R(-yaw)
    rot_t = np.array([[c, s], [-s, c]], dtype=np.float64)
    return dx @ rot_t.T


def ego_align_scan_history(
    lidar_history: np.ndarray,       # [K,H]
    odom_history: np.ndarray,        # [K,3]
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
) -> Dict[str, list]:
    """
    Convert each historical scan to points in its local frame, then align all
    points into the CURRENT robot frame using odometry.
    """
    K, H = lidar_history.shape
    angles = beam_angles(H, angle_min, angle_increment)

    raw_local_points = []
    aligned_current_points = []

    current_pose = odom_history[-1]

    for k in range(K):
        xy_local, valid = scan_to_local_xy(
            lidar_history[k], angles, range_min, range_max
        )
        raw_local_points.append(xy_local)

        xy_world = transform_points_robot_to_world(xy_local, odom_history[k])
        xy_current = transform_points_world_to_robot(xy_world, current_pose)
        aligned_current_points.append(xy_current)

    return {
        "raw_local_points": raw_local_points,
        "aligned_current_points": aligned_current_points,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sample_from_npz(data_path: str, index: int) -> Tuple[dict, GeneratorConfig]:
    data = np.load(data_path, allow_pickle=False)

    required = [
        "lidar_history",
        "odom_history",
        "beam_velocity",
        "beam_dynamic",
        "beam_valid",
        "beam_object_id",
    ]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {data_path}")

    N = data["lidar_history"].shape[0]
    if not (0 <= index < N):
        raise IndexError(f"index={index} out of range for dataset of size N={N}")

    sample = {
        "lidar_history": data["lidar_history"][index].astype(np.float64),
        "odom_history": data["odom_history"][index].astype(np.float64),
        "beam_velocity": data["beam_velocity"][index].astype(np.float64),
        "beam_dynamic": data["beam_dynamic"][index].astype(np.float64),
        "beam_valid": data["beam_valid"][index].astype(np.float64),
        "beam_object_id": data["beam_object_id"][index].astype(np.int32),
    }

    # Try to load metadata config if sidecar JSON exists.
    cfg = GeneratorConfig()
    json_path = Path(data_path).with_suffix(".json")
    if json_path.exists():
        import json
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        gen_cfg_dict = meta.get("generator_config", {})
        # Only overwrite known dataclass fields.
        for field in cfg.__dataclass_fields__.keys():
            if field in gen_cfg_dict:
                setattr(cfg, field, gen_cfg_dict[field])

    return sample, cfg


def generate_fresh_sample(seed: int = 0) -> Tuple[dict, GeneratorConfig]:
    cfg = GeneratorConfig(seed=seed)
    rng = np.random.default_rng(seed)
    sample = generate_one_sample(rng, cfg)
    # Remove helper flag if present.
    sample.pop("accepted_dynamic_quota", None)
    return sample, cfg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_visualization(
    sample: dict,
    cfg: GeneratorConfig,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    arrow_stride: int = 6,
    arrow_scale: float = 1.0,
    show: bool = False,
) -> str:
    """
    Save a 3-panel visualization and return the output file path.
    """
    lidar_history = sample["lidar_history"]
    odom_history = sample["odom_history"]
    beam_velocity = sample["beam_velocity"]
    beam_dynamic = sample["beam_dynamic"]
    beam_valid = sample["beam_valid"]

    K, H = lidar_history.shape
    angles = beam_angles(H, cfg.angle_min, cfg.angle_increment)

    aligned = ego_align_scan_history(
        lidar_history=lidar_history,
        odom_history=odom_history,
        angle_min=cfg.angle_min,
        angle_increment=cfg.angle_increment,
        range_min=cfg.range_min,
        range_max=cfg.range_max,
    )

    raw_local_points = aligned["raw_local_points"]
    aligned_current_points = aligned["aligned_current_points"]

    # Current frame point cloud.
    current_xy, current_valid_from_range = scan_to_local_xy(
        lidar_history[-1], angles, cfg.range_min, cfg.range_max
    )

    # Use provided beam_valid to stay consistent with labels.
    current_valid = beam_valid > 0.5
    static_mask = current_valid & (beam_dynamic <= 0.5)
    dynamic_mask = current_valid & (beam_dynamic > 0.5)

    # Current valid hit coordinates per beam, preserving beam indexing.
    x_all = np.full(H, np.nan, dtype=np.float64)
    y_all = np.full(H, np.nan, dtype=np.float64)
    x_all[current_valid] = lidar_history[-1][current_valid] * np.cos(angles[current_valid])
    y_all[current_valid] = lidar_history[-1][current_valid] * np.sin(angles[current_valid])

    # Limits: robust square axis using aligned history.
    all_pts = [pts for pts in aligned_current_points if pts.size > 0]
    if len(all_pts) > 0:
        stacked = np.concatenate(all_pts, axis=0)
        lim = float(np.max(np.abs(stacked))) + 0.5
        lim = max(lim, 2.5)
    else:
        lim = cfg.range_max

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ---------------------------------------------------------------
    # Left: raw LiDAR history (each frame in its own local robot frame)
    # ---------------------------------------------------------------
    ax = axes[0]
    for k, pts in enumerate(raw_local_points):
        if pts.shape[0] > 0:
            ax.plot(pts[:, 0], pts[:, 1], linestyle="", marker=".", markersize=2.0, label=f"t-{K-1-k}" if k < K - 1 else "t")
    ax.plot([0.0], [0.0], marker="x", linestyle="", markersize=8, label="robot")
    ax.set_title("Raw LiDAR history")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # ---------------------------------------------------------------
    # Middle: ego-motion-aligned history in current robot frame
    # ---------------------------------------------------------------
    ax = axes[1]
    for k, pts in enumerate(aligned_current_points):
        if pts.shape[0] > 0:
            ax.plot(pts[:, 0], pts[:, 1], linestyle="", marker=".", markersize=2.0, label=f"t-{K-1-k}" if k < K - 1 else "t")
    ax.plot([0.0], [0.0], marker="x", linestyle="", markersize=8, label="robot (current)")
    ax.set_title("Ego-aligned history")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # ---------------------------------------------------------------
    # Right: current scan with static/dynamic labels and GT velocity
    # ---------------------------------------------------------------
    ax = axes[2]

    # Static = open circles
    idx_static = np.where(static_mask)[0]
    if idx_static.size > 0:
        ax.scatter(
            x_all[idx_static],
            y_all[idx_static],
            s=18,
            marker="o",
            facecolors="none",
            label="static hits",
        )

    # Dynamic = filled circles
    idx_dynamic = np.where(dynamic_mask)[0]
    if idx_dynamic.size > 0:
        ax.scatter(
            x_all[idx_dynamic],
            y_all[idx_dynamic],
            s=18,
            marker="o",
            label="dynamic hits",
        )

        # Velocity arrows on a subset to avoid clutter.
        arrow_idx = idx_dynamic[::max(1, int(arrow_stride))]
        if arrow_idx.size > 0:
            ax.quiver(
                x_all[arrow_idx],
                y_all[arrow_idx],
                beam_velocity[arrow_idx, 0],
                beam_velocity[arrow_idx, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0 / max(arrow_scale, 1e-6),
                width=0.004,
            )

    ax.plot([0.0], [0.0], marker="x", linestyle="", markersize=8, label="robot")
    ax.set_title("Current scan + GT motion")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    n_valid = int(current_valid.sum())
    n_dynamic = int(dynamic_mask.sum())
    summary = f"valid beams={n_valid} | dynamic beams={n_dynamic} | dynamic/valid={n_dynamic / max(n_valid,1):.3f}"
    fig.text(0.5, 0.01, summary, ha="center", va="bottom")

    if title is None:
        title = "Dynamic LiDAR sample visualization"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    if output_path is None:
        output_path = "/mnt/data/visualize_sample.png"

    output_path = str(output_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test() -> str:
    sample, cfg = generate_fresh_sample(seed=7)
    out = "/mnt/data/visualize_sample_demo.png"
    saved = make_visualization(
        sample=sample,
        cfg=cfg,
        output_path=out,
        title="Smoke-test synthetic sample",
        arrow_stride=4,
        arrow_scale=1.0,
        show=False,
    )
    print("Saved visualization:", saved)
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--data", type=str, default=None, help="Path to generated .npz dataset")
    group.add_argument("--generate", action="store_true", help="Generate one fresh sample on the fly")

    parser.add_argument("--index", type=int, default=0, help="Sample index when using --data")
    parser.add_argument("--output", type=str, default="/mnt/data/visualize_sample.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arrow-stride", type=int, default=6)
    parser.add_argument("--arrow-scale", type=float, default=1.0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        smoke_test()
        return

    if args.generate or args.data is None:
        sample, cfg = generate_fresh_sample(seed=args.seed)
        title = f"Generated sample (seed={args.seed})"
    else:
        sample, cfg = load_sample_from_npz(args.data, args.index)
        title = f"Dataset sample index={args.index}"

    saved = make_visualization(
        sample=sample,
        cfg=cfg,
        output_path=args.output,
        title=title,
        arrow_stride=args.arrow_stride,
        arrow_scale=args.arrow_scale,
        show=args.show,
    )
    print("Saved visualization:", saved)


if __name__ == "__main__":
    main()
