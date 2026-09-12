"""
generate_dataset.py

Procedural synthetic training-data generator for DynamicLiDARNetwork.

Core idea
---------
No expert navigation trajectories are required.

For each sample we procedurally create:
    1) random static geometry,
    2) random moving circular agents,
    3) random robot ego-motion over K LiDAR frames,
    4) exact 2D ray casting at every historical timestamp.

The CURRENT robot frame is chosen as the world frame for each independent
sample, i.e. odom_history[-1] == [0, 0, 0]. This is without loss of generality:
the learning problem is SE(2)-relative, and it makes GT generation less
error-prone.

Saved arrays
------------
lidar_history : float32 [N,K,H]
    Raw historical LiDAR ranges. No-hit beams are stored as +inf.

odom_history  : float32 [N,K,3]
    Robot poses [x,y,yaw] in one common frame. The final pose is [0,0,0].

beam_velocity : float32 [N,H,2]
    GT world/object velocity of the surface hit by each CURRENT beam,
    expressed in the CURRENT robot frame. Static/no-hit beams are [0,0].

beam_dynamic  : float32 [N,H]
    1 for a beam whose nearest CURRENT hit belongs to a moving agent,
    otherwise 0.

beam_valid    : float32 [N,H]
    1 if the CURRENT beam hit any geometry within sensor range.

beam_object_id : int16 [N,H]
    Debug label:
        -1 = no hit
        >=0 = persistent scene object id for the current frame.

Important semantic choice
-------------------------
beam_velocity is NOT obstacle velocity relative to the moving robot.
Ego-motion is removed geometrically by model.py, so a static world object
must have GT velocity [0,0]. Therefore the correct target is the obstacle's
world velocity rotated into the current robot frame.

Example
-------
python generate_dataset.py \
    --output train_data.npz \
    --samples 20000 \
    --seed 42

Quick check
-----------
python generate_dataset.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    # Dataset / sensor
    num_samples: int = 10000
    num_frames: int = 6
    num_beams: int = 720
    scan_dt: float = 0.10

    angle_min: float = -math.pi
    full_circle: bool = True
    range_min: float = 0.05
    range_max: float = 10.0

    # Scene geometry
    world_half_extent: float = 9.0

    min_static_segments: int = 4
    max_static_segments: int = 10
    min_static_circles: int = 1
    max_static_circles: int = 4

    min_dynamic_agents: int = 1
    max_dynamic_agents: int = 5

    # Static segments
    segment_length_min: float = 0.8
    segment_length_max: float = 4.0

    # Static circle radius
    static_radius_min: float = 0.15
    static_radius_max: float = 0.60

    # Moving agents
    dynamic_radius_min: float = 0.18
    dynamic_radius_max: float = 0.38
    dynamic_speed_min: float = 0.15
    dynamic_speed_max: float = 1.25

    # Robot historical motion
    robot_v_min: float = 0.0
    robot_v_max: float = 0.9
    robot_w_min: float = -1.2
    robot_w_max: float = 1.2
    robot_control_jitter: float = 0.12

    # Sampling distances relative to current robot pose
    obstacle_distance_min: float = 0.8
    obstacle_distance_max: float = 8.0

    # Dataset balancing: require some visible moving-object beams.
    min_dynamic_beams: int = 8
    max_scene_retries: int = 30

    # Add small measurement noise AFTER exact ray casting.
    lidar_noise_std: float = 0.01

    seed: int = 42

    @property
    def angle_increment(self) -> float:
        if self.full_circle:
            return 2.0 * math.pi / self.num_beams
        raise NotImplementedError(
            "This generator currently assumes a full-circle LiDAR."
        )


# ---------------------------------------------------------------------------
# Scene primitives
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    a: np.ndarray               # [2]
    b: np.ndarray               # [2]
    object_id: int


@dataclass
class Circle:
    center_current: np.ndarray  # [2] center at t=current
    radius: float
    velocity: np.ndarray        # [2], zero for static circles
    dynamic: bool
    object_id: int

    def center_at(self, time_from_current: float) -> np.ndarray:
        return self.center_current + self.velocity * time_from_current


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2D cross product with broadcasting over final coordinate dimension."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def wrap_angle(angle: np.ndarray | float):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def ray_segment_distances(
    origin: np.ndarray,         # [2]
    directions: np.ndarray,     # [H,2], normalized
    segments: Sequence[Segment],
    range_min: float,
    range_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nearest ray/segment intersection for every beam.

    Returns
    -------
    distance  : [H], inf for no hit
    object_id : [H], -1 for no hit
    """
    H = directions.shape[0]

    if len(segments) == 0:
        return (
            np.full(H, np.inf, dtype=np.float64),
            np.full(H, -1, dtype=np.int32),
        )

    a = np.stack([s.a for s in segments], axis=0)       # [S,2]
    b = np.stack([s.b for s in segments], axis=0)       # [S,2]
    seg_vec = b - a                                     # [S,2]

    # [H,S,2]
    d = directions[:, None, :]
    svec = seg_vec[None, :, :]
    ao = (a - origin)[None, :, :]

    denom = cross2(d, svec)                             # [H,S]
    parallel = np.abs(denom) < 1e-10

    safe_denom = np.where(parallel, 1.0, denom)

    t = cross2(ao, svec) / safe_denom                  # ray distance
    u = cross2(ao, d) / safe_denom                     # segment parameter

    valid = (
        (~parallel)
        & (t >= range_min)
        & (t <= range_max)
        & (u >= 0.0)
        & (u <= 1.0)
    )

    t_valid = np.where(valid, t, np.inf)
    best_idx = np.argmin(t_valid, axis=1)
    best_t = t_valid[np.arange(H), best_idx]

    ids = np.array([s.object_id for s in segments], dtype=np.int32)
    best_id = ids[best_idx]
    best_id[~np.isfinite(best_t)] = -1

    return best_t, best_id


def ray_circle_distances(
    origin: np.ndarray,         # [2]
    directions: np.ndarray,     # [H,2], normalized
    circles: Sequence[Circle],
    time_from_current: float,
    range_min: float,
    range_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nearest ray/circle boundary intersection for every beam.
    """
    H = directions.shape[0]

    if len(circles) == 0:
        return (
            np.full(H, np.inf, dtype=np.float64),
            np.full(H, -1, dtype=np.int32),
        )

    centers = np.stack(
        [c.center_at(time_from_current) for c in circles],
        axis=0,
    )                                                   # [C,2]
    radii = np.array([c.radius for c in circles], dtype=np.float64)

    # For ray p(t)=o+t*d and circle center c:
    # |o+t*d-c|^2=r^2
    # t^2 + 2*b*t + c0 = 0, because |d|=1.
    oc = origin[None, :] - centers                     # [C,2]
    b = np.einsum("hd,cd->hc", directions, oc)         # [H,C]
    c0 = np.sum(oc * oc, axis=1)[None, :] - radii[None, :] ** 2

    disc = b * b - c0
    has_intersection = disc >= 0.0

    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))

    t_near = -b - sqrt_disc
    t_far = -b + sqrt_disc

    # If origin is inside a circle, near root can be negative;
    # choose the first positive valid root.
    t = np.where(
        t_near >= range_min,
        t_near,
        t_far,
    )

    valid = (
        has_intersection
        & (t >= range_min)
        & (t <= range_max)
    )

    t_valid = np.where(valid, t, np.inf)
    best_idx = np.argmin(t_valid, axis=1)
    best_t = t_valid[np.arange(H), best_idx]

    ids = np.array([c.object_id for c in circles], dtype=np.int32)
    best_id = ids[best_idx]
    best_id[~np.isfinite(best_t)] = -1

    return best_t, best_id


def cast_lidar(
    robot_pose: np.ndarray,     # [x,y,yaw]
    beam_angles_robot: np.ndarray,
    segments: Sequence[Segment],
    circles: Sequence[Circle],
    time_from_current: float,
    cfg: GeneratorConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact nearest-hit ray casting against all static segments/circles and
    moving circles at the requested timestamp.

    Returns
    -------
    ranges     : [H], inf on no hit
    object_ids : [H], -1 on no hit
    """
    x, y, yaw = robot_pose
    world_angles = yaw + beam_angles_robot

    directions = np.stack(
        [np.cos(world_angles), np.sin(world_angles)],
        axis=-1,
    )                                                   # [H,2]
    origin = np.array([x, y], dtype=np.float64)

    seg_d, seg_id = ray_segment_distances(
        origin,
        directions,
        segments,
        cfg.range_min,
        cfg.range_max,
    )
    cir_d, cir_id = ray_circle_distances(
        origin,
        directions,
        circles,
        time_from_current,
        cfg.range_min,
        cfg.range_max,
    )

    use_circle = cir_d < seg_d
    ranges = np.where(use_circle, cir_d, seg_d)
    object_ids = np.where(use_circle, cir_id, seg_id).astype(np.int32)

    no_hit = ~np.isfinite(ranges)
    object_ids[no_hit] = -1

    return ranges, object_ids


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------

def sample_polar_point(
    rng: np.random.Generator,
    r_min: float,
    r_max: float,
) -> np.ndarray:
    # Uniform in area, not uniform in radius.
    r2 = rng.uniform(r_min ** 2, r_max ** 2)
    r = math.sqrt(r2)
    a = rng.uniform(-math.pi, math.pi)
    return np.array([r * math.cos(a), r * math.sin(a)], dtype=np.float64)


def make_boundary_segments(
    half_extent: float,
    start_id: int = 0,
) -> Tuple[List[Segment], int]:
    h = float(half_extent)
    corners = [
        np.array([-h, -h], dtype=np.float64),
        np.array([+h, -h], dtype=np.float64),
        np.array([+h, +h], dtype=np.float64),
        np.array([-h, +h], dtype=np.float64),
    ]

    segments: List[Segment] = []
    oid = start_id

    for i in range(4):
        segments.append(
            Segment(
                a=corners[i],
                b=corners[(i + 1) % 4],
                object_id=oid,
            )
        )
        oid += 1

    return segments, oid


def random_static_segment(
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    object_id: int,
) -> Segment:
    for _ in range(100):
        center = sample_polar_point(
            rng,
            cfg.obstacle_distance_min,
            min(cfg.obstacle_distance_max, cfg.world_half_extent - 0.5),
        )
        length = rng.uniform(
            cfg.segment_length_min,
            cfg.segment_length_max,
        )
        angle = rng.uniform(-math.pi, math.pi)
        d = 0.5 * length * np.array(
            [math.cos(angle), math.sin(angle)],
            dtype=np.float64,
        )
        a = center - d
        b = center + d

        if (
            np.all(np.abs(a) < cfg.world_half_extent - 0.1)
            and np.all(np.abs(b) < cfg.world_half_extent - 0.1)
        ):
            return Segment(a=a, b=b, object_id=object_id)

    # Fallback small segment.
    return Segment(
        a=np.array([2.0, -0.5]),
        b=np.array([2.0, +0.5]),
        object_id=object_id,
    )


def random_circle(
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    object_id: int,
    dynamic: bool,
) -> Circle:
    center = sample_polar_point(
        rng,
        cfg.obstacle_distance_min,
        min(cfg.obstacle_distance_max, cfg.range_max * 0.90),
    )

    if dynamic:
        radius = rng.uniform(
            cfg.dynamic_radius_min,
            cfg.dynamic_radius_max,
        )
        speed = rng.uniform(
            cfg.dynamic_speed_min,
            cfg.dynamic_speed_max,
        )
        angle = rng.uniform(-math.pi, math.pi)
        velocity = speed * np.array(
            [math.cos(angle), math.sin(angle)],
            dtype=np.float64,
        )
    else:
        radius = rng.uniform(
            cfg.static_radius_min,
            cfg.static_radius_max,
        )
        velocity = np.zeros(2, dtype=np.float64)

    return Circle(
        center_current=center,
        radius=float(radius),
        velocity=velocity,
        dynamic=dynamic,
        object_id=object_id,
    )


def generate_robot_odom(
    rng: np.random.Generator,
    cfg: GeneratorConfig,
) -> np.ndarray:
    """
    Generate historical robot poses backwards from current pose [0,0,0].

    Output order:
        oldest ... current
    shape [K,3]
    """
    K = cfg.num_frames
    poses = np.zeros((K, 3), dtype=np.float64)
    poses[-1] = [0.0, 0.0, 0.0]

    base_v = rng.uniform(cfg.robot_v_min, cfg.robot_v_max)
    base_w = rng.uniform(cfg.robot_w_min, cfg.robot_w_max)

    # Controls correspond to intervals k -> k+1.
    v_seq = np.clip(
        base_v + rng.normal(0.0, cfg.robot_control_jitter, size=K - 1),
        cfg.robot_v_min,
        cfg.robot_v_max,
    )
    w_seq = np.clip(
        base_w + rng.normal(0.0, cfg.robot_control_jitter, size=K - 1),
        cfg.robot_w_min,
        cfg.robot_w_max,
    )

    for k in range(K - 2, -1, -1):
        x_next, y_next, th_next = poses[k + 1]
        v = float(v_seq[k])
        w = float(w_seq[k])

        th_prev = th_next - w * cfg.scan_dt
        x_prev = x_next - v * cfg.scan_dt * math.cos(th_prev)
        y_prev = y_next - v * cfg.scan_dt * math.sin(th_prev)

        poses[k] = [x_prev, y_prev, wrap_angle(th_prev)]

    return poses


def build_scene(
    rng: np.random.Generator,
    cfg: GeneratorConfig,
) -> Tuple[List[Segment], List[Circle]]:
    segments, next_id = make_boundary_segments(
        cfg.world_half_extent,
        start_id=0,
    )

    n_seg = int(
        rng.integers(
            cfg.min_static_segments,
            cfg.max_static_segments + 1,
        )
    )
    for _ in range(n_seg):
        segments.append(
            random_static_segment(
                rng,
                cfg,
                object_id=next_id,
            )
        )
        next_id += 1

    circles: List[Circle] = []

    n_static = int(
        rng.integers(
            cfg.min_static_circles,
            cfg.max_static_circles + 1,
        )
    )
    for _ in range(n_static):
        circles.append(
            random_circle(
                rng,
                cfg,
                object_id=next_id,
                dynamic=False,
            )
        )
        next_id += 1

    n_dyn = int(
        rng.integers(
            cfg.min_dynamic_agents,
            cfg.max_dynamic_agents + 1,
        )
    )
    for _ in range(n_dyn):
        circles.append(
            random_circle(
                rng,
                cfg,
                object_id=next_id,
                dynamic=True,
            )
        )
        next_id += 1

    return segments, circles


# ---------------------------------------------------------------------------
# GT construction
# ---------------------------------------------------------------------------

def rotate_world_velocity_to_current_robot(
    velocity_world: np.ndarray,
    current_robot_yaw: float,
) -> np.ndarray:
    """
    R(-yaw_current) * v_world.
    """
    c = math.cos(current_robot_yaw)
    s = math.sin(current_robot_yaw)

    vx, vy = velocity_world
    return np.array(
        [c * vx + s * vy, -s * vx + c * vy],
        dtype=np.float64,
    )


def current_beam_labels(
    object_ids: np.ndarray,
    circles: Sequence[Circle],
    current_robot_yaw: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build GT labels for CURRENT scan beams.

    Returns
    -------
    beam_velocity : [H,2]
    beam_dynamic  : [H]
    beam_valid    : [H]
    """
    H = object_ids.shape[0]

    beam_velocity = np.zeros((H, 2), dtype=np.float64)
    beam_dynamic = np.zeros(H, dtype=np.float64)
    beam_valid = (object_ids >= 0).astype(np.float64)

    dynamic_map = {
        c.object_id: c
        for c in circles
        if c.dynamic
    }

    for oid, circle in dynamic_map.items():
        hit = object_ids == oid
        if not np.any(hit):
            continue

        vel_robot = rotate_world_velocity_to_current_robot(
            circle.velocity,
            current_robot_yaw,
        )

        beam_velocity[hit] = vel_robot
        beam_dynamic[hit] = 1.0

    return beam_velocity, beam_dynamic, beam_valid


# ---------------------------------------------------------------------------
# Single sample
# ---------------------------------------------------------------------------

def generate_one_sample(
    rng: np.random.Generator,
    cfg: GeneratorConfig,
) -> dict:
    """
    Generate one full training sample.

    Scene regeneration is retried to ensure the CURRENT scan contains enough
    dynamic beams. This prevents an overwhelmingly static synthetic dataset.
    """
    beam_angles = (
        cfg.angle_min
        + np.arange(cfg.num_beams, dtype=np.float64)
        * cfg.angle_increment
    )

    odom = generate_robot_odom(rng, cfg)

    # Relative times: oldest < ... < current=0.
    times = (
        np.arange(cfg.num_frames, dtype=np.float64)
        - (cfg.num_frames - 1)
    ) * cfg.scan_dt

    accepted = False
    last = None

    for _attempt in range(cfg.max_scene_retries):
        segments, circles = build_scene(rng, cfg)

        lidar_history = np.full(
            (cfg.num_frames, cfg.num_beams),
            np.inf,
            dtype=np.float64,
        )
        hit_ids_history = np.full(
            (cfg.num_frames, cfg.num_beams),
            -1,
            dtype=np.int32,
        )

        for k in range(cfg.num_frames):
            ranges, hit_ids = cast_lidar(
                robot_pose=odom[k],
                beam_angles_robot=beam_angles,
                segments=segments,
                circles=circles,
                time_from_current=float(times[k]),
                cfg=cfg,
            )

            # Add sensor noise only to real returns.
            if cfg.lidar_noise_std > 0:
                finite = np.isfinite(ranges)
                noise = rng.normal(
                    0.0,
                    cfg.lidar_noise_std,
                    size=cfg.num_beams,
                )
                ranges[finite] += noise[finite]
                ranges[finite] = np.clip(
                    ranges[finite],
                    cfg.range_min,
                    cfg.range_max,
                )

            lidar_history[k] = ranges
            hit_ids_history[k] = hit_ids

        beam_velocity, beam_dynamic, beam_valid = current_beam_labels(
            object_ids=hit_ids_history[-1],
            circles=circles,
            current_robot_yaw=float(odom[-1, 2]),
        )

        last = {
            "lidar_history": lidar_history,
            "odom_history": odom,
            "beam_velocity": beam_velocity,
            "beam_dynamic": beam_dynamic,
            "beam_valid": beam_valid,
            "beam_object_id": hit_ids_history[-1],
        }

        if int(beam_dynamic.sum()) >= cfg.min_dynamic_beams:
            accepted = True
            break

    # If the retry budget is exhausted, keep the last valid geometric scene
    # rather than failing dataset generation entirely.
    if last is None:
        raise RuntimeError("Failed to generate any scene.")

    last["accepted_dynamic_quota"] = accepted
    return last


# ---------------------------------------------------------------------------
# Full dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    cfg: GeneratorConfig,
    output_path: str,
) -> None:
    rng = np.random.default_rng(cfg.seed)

    N = cfg.num_samples
    K = cfg.num_frames
    H = cfg.num_beams

    lidar_history = np.empty((N, K, H), dtype=np.float32)
    odom_history = np.empty((N, K, 3), dtype=np.float32)
    beam_velocity = np.empty((N, H, 2), dtype=np.float32)
    beam_dynamic = np.empty((N, H), dtype=np.float32)
    beam_valid = np.empty((N, H), dtype=np.float32)
    beam_object_id = np.empty((N, H), dtype=np.int16)

    quota_failures = 0

    report_every = max(1, N // 20)

    for i in range(N):
        sample = generate_one_sample(rng, cfg)

        lidar_history[i] = sample["lidar_history"].astype(np.float32)
        odom_history[i] = sample["odom_history"].astype(np.float32)
        beam_velocity[i] = sample["beam_velocity"].astype(np.float32)
        beam_dynamic[i] = sample["beam_dynamic"].astype(np.float32)
        beam_valid[i] = sample["beam_valid"].astype(np.float32)
        beam_object_id[i] = sample["beam_object_id"].astype(np.int16)

        if not sample["accepted_dynamic_quota"]:
            quota_failures += 1

        if (i + 1) % report_every == 0 or (i + 1) == N:
            dyn_frac = float(
                beam_dynamic[: i + 1].sum()
                / np.maximum(beam_valid[: i + 1].sum(), 1.0)
            )
            print(
                f"[{i+1:>7d}/{N}] "
                f"dynamic-valid-beam fraction={dyn_frac:.4f} "
                f"quota_failures={quota_failures}"
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        lidar_history=lidar_history,
        odom_history=odom_history,
        beam_velocity=beam_velocity,
        beam_dynamic=beam_dynamic,
        beam_valid=beam_valid,
        beam_object_id=beam_object_id,
    )

    metadata = {
        "generator_config": asdict(cfg),
        "arrays": {
            "lidar_history": list(lidar_history.shape),
            "odom_history": list(odom_history.shape),
            "beam_velocity": list(beam_velocity.shape),
            "beam_dynamic": list(beam_dynamic.shape),
            "beam_valid": list(beam_valid.shape),
            "beam_object_id": list(beam_object_id.shape),
        },
        "semantics": {
            "beam_velocity":
                "object/world velocity expressed in current robot frame; "
                "static/no-hit beams are zero",
            "beam_dynamic":
                "1 iff current beam nearest-hit object is dynamic",
            "beam_valid":
                "1 iff current beam has a finite geometric return",
            "no_hit_lidar_value":
                "+inf",
        },
        "quota_failures": int(quota_failures),
    }

    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("Saved:")
    print(" ", output)
    print(" ", metadata_path)
    print("shapes:")
    for key, value in metadata["arrays"].items():
        print(f"  {key:16s}: {tuple(value)}")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_generated_file(path: str) -> None:
    data = np.load(path, allow_pickle=False)

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
            raise AssertionError(f"Missing key: {k}")

    lidar = data["lidar_history"]
    odom = data["odom_history"]
    vel = data["beam_velocity"]
    dyn = data["beam_dynamic"]
    valid = data["beam_valid"]
    oid = data["beam_object_id"]

    N, K, H = lidar.shape

    assert odom.shape == (N, K, 3)
    assert vel.shape == (N, H, 2)
    assert dyn.shape == (N, H)
    assert valid.shape == (N, H)
    assert oid.shape == (N, H)

    # By construction each sample's current robot frame is world frame.
    assert np.allclose(odom[:, -1], 0.0, atol=1e-6)

    # Dynamic beams must be valid and have an object id.
    dyn_mask = dyn > 0.5
    assert np.all(valid[dyn_mask] > 0.5)
    assert np.all(oid[dyn_mask] >= 0)

    # Static/no-hit labels should have zero velocity.
    static = dyn <= 0.5
    assert np.max(np.abs(vel[static])) < 1e-5

    print("Validation passed.")
    print("samples:", N)
    print("dynamic beams:", int(dyn_mask.sum()))
    print("valid beams:", int((valid > 0.5).sum()))
    print(
        "dynamic / valid:",
        float(dyn_mask.sum() / max((valid > 0.5).sum(), 1)),
    )


def smoke_test() -> None:
    out = Path("/tmp/dynamic_motion_field_smoke.npz")

    cfg = GeneratorConfig(
        num_samples=8,
        num_frames=6,
        num_beams=720,
        min_dynamic_beams=5,
        max_scene_retries=10,
        seed=7,
    )

    generate_dataset(cfg, str(out))
    validate_generated_file(str(out))

    # Verify direct compatibility with train.py target construction.
    import torch
    from train import build_patch_targets

    d = np.load(out, allow_pickle=False)
    targets = build_patch_targets(
        beam_velocity=torch.from_numpy(d["beam_velocity"][:2]),
        beam_dynamic=torch.from_numpy(d["beam_dynamic"][:2]),
        beam_valid=torch.from_numpy(d["beam_valid"][:2]),
        num_patches=36,
        max_speed=3.0,
    )

    assert targets["velocity_target"].shape == (2, 36, 2)
    assert targets["confidence_target"].shape == (2, 36, 1)
    assert targets["patch_valid"].shape == (2, 36)

    print("train.py compatibility passed.")
    print("velocity_target   :", tuple(targets["velocity_target"].shape))
    print("confidence_target :", tuple(targets["confidence_target"].shape))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=str,
        default="train_data.npz",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=720,
    )
    parser.add_argument(
        "--scan-dt",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--range-max",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--min-dynamic-beams",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lidar-noise-std",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        smoke_test()
        return

    cfg = GeneratorConfig(
        num_samples=args.samples,
        num_frames=args.frames,
        num_beams=args.beams,
        scan_dt=args.scan_dt,
        range_max=args.range_max,
        min_dynamic_beams=args.min_dynamic_beams,
        lidar_noise_std=args.lidar_noise_std,
        seed=args.seed,
    )

    generate_dataset(
        cfg=cfg,
        output_path=args.output,
    )
    validate_generated_file(args.output)


if __name__ == "__main__":
    main()
