"""Shared helpers for ScanFlow validation experiments."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_dataset import (  # noqa: E402
    Circle,
    GeneratorConfig,
    Segment,
    cast_lidar,
    current_beam_labels,
)
from model import DynamicLiDARNetwork, ModelConfig  # noqa: E402
from train import build_patch_targets  # noqa: E402


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_config" not in ckpt or "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint must contain model_config and model_state_dict.")
    cfg = ModelConfig(**ckpt["model_config"])
    model = DynamicLiDARNetwork(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, cfg, ckpt


def constant_control_odom(
    num_frames: int,
    dt: float,
    linear_speed: float,
    angular_speed: float,
) -> np.ndarray:
    """Generate oldest->current poses with current pose exactly [0,0,0]."""
    poses = np.zeros((num_frames, 3), dtype=np.float64)
    for k in range(num_frames - 2, -1, -1):
        x_next, y_next, th_next = poses[k + 1]
        th_prev = th_next - angular_speed * dt
        x_prev = x_next - linear_speed * dt * math.cos(th_prev)
        y_prev = y_next - linear_speed * dt * math.sin(th_prev)
        poses[k] = [x_prev, y_prev, wrap_angle(th_prev)]
    return poses


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def default_static_scene() -> Tuple[list[Segment], list[Circle]]:
    """Deterministic asymmetric static scene with broad angular coverage."""
    segments = [
        Segment(np.array([-7.5, -7.5]), np.array([7.5, -7.5]), 0),
        Segment(np.array([7.5, -7.5]), np.array([7.5, 7.5]), 1),
        Segment(np.array([7.5, 7.5]), np.array([-7.5, 7.5]), 2),
        Segment(np.array([-7.5, 7.5]), np.array([-7.5, -7.5]), 3),
        Segment(np.array([2.0, -2.8]), np.array([2.0, -0.8]), 4),
        Segment(np.array([-3.2, 1.2]), np.array([-1.0, 2.5]), 5),
        Segment(np.array([0.8, 3.0]), np.array([3.5, 2.4]), 6),
    ]
    circles = [
        Circle(np.array([3.2, 1.1]), 0.45, np.zeros(2), False, 7),
        Circle(np.array([-2.4, -2.1]), 0.55, np.zeros(2), False, 8),
        Circle(np.array([-1.3, 3.6]), 0.35, np.zeros(2), False, 9),
    ]
    return segments, circles


def sensor_config(model_cfg: ModelConfig, scan_dt: float = 0.10) -> GeneratorConfig:
    return GeneratorConfig(
        num_samples=1,
        num_frames=model_cfg.num_frames,
        num_beams=model_cfg.num_beams,
        scan_dt=scan_dt,
        angle_min=model_cfg.angle_min,
        range_min=model_cfg.range_min,
        range_max=model_cfg.range_max,
        lidar_noise_std=0.0,
        min_dynamic_beams=0,
    )


def render_history(
    gen_cfg: GeneratorConfig,
    odom: np.ndarray,
    segments: Sequence[Segment],
    circles: Sequence[Circle],
) -> Tuple[np.ndarray, np.ndarray]:
    angles = gen_cfg.angle_min + np.arange(gen_cfg.num_beams) * gen_cfg.angle_increment
    times = (np.arange(gen_cfg.num_frames) - (gen_cfg.num_frames - 1)) * gen_cfg.scan_dt
    lidar = np.full((gen_cfg.num_frames, gen_cfg.num_beams), np.inf, dtype=np.float64)
    hit_ids = np.full((gen_cfg.num_frames, gen_cfg.num_beams), -1, dtype=np.int32)
    for k in range(gen_cfg.num_frames):
        lidar[k], hit_ids[k] = cast_lidar(
            robot_pose=odom[k],
            beam_angles_robot=angles,
            segments=segments,
            circles=circles,
            time_from_current=float(times[k]),
            cfg=gen_cfg,
        )
    return lidar, hit_ids


@torch.no_grad()
def infer_one(
    model: DynamicLiDARNetwork,
    lidar: np.ndarray,
    odom: np.ndarray,
    device: torch.device,
    return_aux: bool = False,
) -> Dict[str, torch.Tensor]:
    lidar_t = torch.from_numpy(lidar.astype(np.float32))[None].to(device)
    odom_t = torch.from_numpy(odom.astype(np.float32))[None].to(device)
    return model(lidar_t, odom_t, return_aux=return_aux)


def patch_targets_from_scene(
    model_cfg: ModelConfig,
    hit_ids_current: np.ndarray,
    circles: Sequence[Circle],
):
    beam_velocity, beam_dynamic, beam_valid = current_beam_labels(
        hit_ids_current,
        circles,
        current_robot_yaw=0.0,
    )
    targets = build_patch_targets(
        torch.from_numpy(beam_velocity.astype(np.float32))[None],
        torch.from_numpy(beam_dynamic.astype(np.float32))[None],
        torch.from_numpy(beam_valid.astype(np.float32))[None],
        num_patches=model_cfg.num_patches,
        max_speed=model_cfg.max_motion_speed,
    )
    return targets, beam_velocity, beam_dynamic, beam_valid


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def classification_counts(
    confidence: torch.Tensor,
    confidence_target: torch.Tensor,
    patch_valid: torch.Tensor,
    threshold: float,
):
    pred = confidence.squeeze(-1) >= threshold
    gt = confidence_target.squeeze(-1) >= 0.5
    valid = patch_valid.bool()
    tp = int((pred & gt & valid).sum())
    fp = int((pred & ~gt & valid).sum())
    fn = int((~pred & gt & valid).sum())
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def transform_world_to_robot(
    point_world: np.ndarray,
    robot_pose_world: np.ndarray,
) -> np.ndarray:
    dx = point_world[0] - robot_pose_world[0]
    dy = point_world[1] - robot_pose_world[1]
    c = math.cos(robot_pose_world[2])
    s = math.sin(robot_pose_world[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy], dtype=np.float64)


def rotate_world_to_robot(
    vector_world: np.ndarray,
    robot_yaw: float,
) -> np.ndarray:
    c = math.cos(robot_yaw)
    s = math.sin(robot_yaw)
    vx, vy = vector_world
    return np.array([c * vx + s * vy, -s * vx + c * vy], dtype=np.float64)


def robot_to_world_velocity(linear_speed: float, robot_yaw: float) -> np.ndarray:
    return linear_speed * np.array([math.cos(robot_yaw), math.sin(robot_yaw)], dtype=np.float64)


def integrate_unicycle(
    pose: np.ndarray,
    v: float,
    w: float,
    dt: float,
) -> np.ndarray:
    out = pose.copy()
    out[0] += dt * v * math.cos(pose[2])
    out[1] += dt * v * math.sin(pose[2])
    out[2] = wrap_angle(pose[2] + dt * w)
    return out
