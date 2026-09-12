"""Shared metric geometry for ScanFlow motion estimators.

The public functions in this module are deliberately NumPy based.  They keep
the sensor geometry, SE(2) convention, finite surface representation and
metric velocity hypotheses in one place so that baselines and learned
estimators cannot silently use different coordinate semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class GeometryConfig:
    range_min: float = 0.05
    range_max: float = 10.0
    angle_min: float = -math.pi
    max_speed: float = 1.5
    gap_base: float = 0.10
    gap_scale: float = 2.0
    minimum_points: int = 2


def _as_scan(scan: np.ndarray) -> np.ndarray:
    scan = np.asarray(scan, dtype=np.float64)
    if scan.ndim != 1:
        raise ValueError(f"scan must be [H], got {scan.shape}")
    return scan


def scan_to_xy(scan: np.ndarray, cfg: GeometryConfig | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one polar scan to metric XY points and a finite-return mask."""
    cfg = cfg or GeometryConfig()
    scan = _as_scan(scan)
    h = scan.shape[0]
    angles = cfg.angle_min + np.arange(h, dtype=np.float64) * 2.0 * math.pi / h
    valid = np.isfinite(scan) & (scan >= cfg.range_min) & (scan <= cfg.range_max)
    safe_range = np.where(valid, scan, 0.0)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    return safe_range[:, None] * unit, valid, angles


def _rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def align_history_to_current(
    scan: np.ndarray,
    pose: np.ndarray,
    current_pose: np.ndarray,
    cfg: GeometryConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a scan's XY points expressed in the current robot frame.

    ``pose`` and ``current_pose`` are [x, y, yaw] in one odometry/world frame.
    Row-vector multiplication by ``R(yaw)`` implements world-to-current for
    this convention, matching the existing ScanFlow baselines.
    """
    cfg = cfg or GeometryConfig()
    scan = _as_scan(scan)
    pose = np.asarray(pose, dtype=np.float64)
    current_pose = np.asarray(current_pose, dtype=np.float64)
    if pose.shape != (3,) or current_pose.shape != (3,):
        raise ValueError("pose/current_pose must be [3].")
    local, valid, angles = scan_to_xy(scan, cfg)
    world = local @ _rotation(pose[2]).T + pose[:2]
    current = (world - current_pose[:2]) @ _rotation(current_pose[2])
    return current, valid, angles


def _groups_from_scan(points: np.ndarray, ranges: np.ndarray, valid: np.ndarray, cfg: GeometryConfig) -> List[np.ndarray]:
    """Segment adjacent finite returns using the historical ScanFlow rule."""
    groups: List[List[int]] = []
    h = len(ranges)
    for i in range(h):
        if not valid[i]:
            continue
        threshold = cfg.gap_base + cfg.gap_scale * min(ranges[i], ranges[i - 1]) * 2.0 * math.pi / h
        if not groups or i != groups[-1][-1] + 1 or np.linalg.norm(points[i] - points[i - 1]) > threshold:
            groups.append([])
        groups[-1].append(i)
    if len(groups) > 1 and valid[0] and valid[-1]:
        threshold = cfg.gap_base + cfg.gap_scale * min(ranges[0], ranges[-1]) * 2.0 * math.pi / h
        if np.linalg.norm(points[0] - points[-1]) <= threshold:
            groups[0] = groups[-1] + groups[0]
            groups.pop()
    return [np.asarray(group, dtype=np.int64) for group in groups]


def extract_finite_surfaces(
    scan: np.ndarray,
    pose: np.ndarray,
    current_pose: np.ndarray,
    cfg: GeometryConfig | None = None,
) -> List[Dict[str, np.ndarray | float]]:
    """Segment a scan and express each finite surface in the current frame."""
    cfg = cfg or GeometryConfig()
    scan = _as_scan(scan)
    points, valid, _ = align_history_to_current(scan, pose, current_pose, cfg)
    groups = _groups_from_scan(points, np.where(valid, scan, 0.0), valid, cfg)
    surfaces = []
    for beams in groups:
        surface_points = points[beams]
        surfaces.append(
            {
                "beams": beams,
                "points": surface_points,
                "anchor": surface_points.mean(axis=0),
                "extent": float(np.linalg.norm(np.ptp(surface_points, axis=0))),
            }
        )
    return surfaces


def build_local_neighborhood(
    points: np.ndarray,
    valid: np.ndarray,
    index: int,
    radius: float = 0.35,
    beam_radius: int = 3,
) -> np.ndarray:
    """Return nearby valid beam indices for one current point."""
    points = np.asarray(points, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if points.ndim != 2 or points.shape[-1] != 2 or valid.shape != (len(points),):
        raise ValueError("points must be [H,2] and valid must be [H].")
    h = len(points)
    ids = (int(index) + np.arange(-beam_radius, beam_radius + 1)) % h
    return ids[valid[ids] & (np.linalg.norm(points[ids] - points[index], axis=-1) <= radius)]


class FiniteSurfaceDistance:
    """Approximate nearest distance to finite, gap-preserving line segments."""

    def __init__(self, surfaces: Sequence[Dict[str, np.ndarray | float]]):
        starts, ends = [], []
        for surface in surfaces:
            points = np.asarray(surface["points"], dtype=np.float64)
            if len(points) >= 2:
                starts.extend(points[:-1])
                ends.extend(points[1:])
        self.starts = np.asarray(starts, dtype=np.float64).reshape(-1, 2)
        self.ends = np.asarray(ends, dtype=np.float64).reshape(-1, 2)
        self.tree = cKDTree((self.starts + self.ends) / 2.0) if len(self.starts) else None

    def distance(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        shape = points.shape[:-1]
        if points.shape[-1] != 2:
            raise ValueError("points must end in dimension 2.")
        if self.tree is None:
            return np.full(shape, np.inf, dtype=np.float64)
        flat = points.reshape(-1, 2)
        _, indices = self.tree.query(flat, k=min(8, len(self.starts)))
        indices = np.asarray(indices).reshape(len(flat), -1)
        a, b = self.starts[indices], self.ends[indices]
        ab = b - a
        t = ((flat[:, None] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12)
        projection = a + np.clip(t, 0.0, 1.0)[..., None] * ab
        return np.linalg.norm(flat[:, None] - projection, axis=-1).min(-1).reshape(shape)


# Short alias for callers that used the old class name.
SurfaceDistance = FiniteSurfaceDistance


def surface_distance(
    surfaces: Sequence[Dict[str, np.ndarray | float]],
    points: np.ndarray,
) -> np.ndarray:
    """Return nearest finite-surface distance for ``points``.

    This convenience function keeps callers from depending on the concrete
    distance-index implementation.  The estimator uses ``SurfaceDistance``
    directly when the same surfaces are queried repeatedly.
    """
    return FiniteSurfaceDistance(surfaces).distance(points)


def velocity_candidates(max_speed: float = 1.5, step: float = 0.25) -> np.ndarray:
    """Return a configurable square grid clipped to a speed disk."""
    if max_speed <= 0.0 or step <= 0.0:
        raise ValueError("max_speed and step must be positive.")
    axis = np.arange(-max_speed, max_speed + step * 0.5, step, dtype=np.float64)
    grid = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    return grid[np.linalg.norm(grid, axis=-1) <= max_speed + 1e-8].astype(np.float32)


def motion_cost_for_points(
    local_points: np.ndarray,
    candidates: np.ndarray,
    history: Sequence[FiniteSurfaceDistance],
    lags: np.ndarray,
) -> np.ndarray:
    """Evaluate the historical finite-surface cost for one local point set."""
    local_points = np.asarray(local_points, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64).reshape(-1, 2)
    lags = np.asarray(lags, dtype=np.float64)
    if local_points.ndim != 2 or local_points.shape[-1] != 2 or lags.shape != (len(history),):
        raise ValueError("local_points/candidates/history/lags have incompatible shapes.")
    if not history:
        return np.full(len(candidates), np.inf, dtype=np.float64)
    frame_costs = []
    for surface, lag in zip(history, lags):
        distances = surface.distance(local_points[None] - lag * candidates[:, None])
        frame_costs.append(np.minimum(distances, 0.3).mean(axis=-1))
    return np.median(np.stack(frame_costs, axis=0), axis=0)


def compute_motion_cost_volume(
    lidar: np.ndarray,
    odom: np.ndarray,
    timestamps: np.ndarray,
    geometry_cfg: GeometryConfig | None = None,
    candidate_max_speed: float = 1.5,
    candidate_step: float = 0.25,
    neighborhood_radius: float = 0.35,
    neighborhood_beams: int = 3,
) -> Dict[str, np.ndarray | List[FiniteSurfaceDistance] | int]:
    """Build metric candidate costs and legacy-compatible scorer features.

    The first eight features intentionally preserve ``MotionCandidateScorer``
    checkpoint compatibility.  Extended evidence is returned separately for
    v2 diagnostics and later scorer versions.
    """
    cfg = geometry_cfg or GeometryConfig(max_speed=candidate_max_speed)
    lidar = np.asarray(lidar, dtype=np.float64)
    odom = np.asarray(odom, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if lidar.ndim != 2 or odom.shape != (len(lidar), 3) or timestamps.shape != (len(lidar),):
        raise ValueError("Expected lidar [K,H], odom [K,3], timestamps [K].")
    if len(lidar) < 2 or not np.isfinite(odom).all() or not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) > 0):
        raise ValueError("Need a current scan, at least one history scan, and strictly increasing timestamps.")

    frames = [extract_finite_surfaces(scan, pose, odom[-1], cfg) for scan, pose in zip(lidar, odom)]
    current = frames[-1]
    h = lidar.shape[1]
    points = np.zeros((h, 2), dtype=np.float64)
    valid = np.zeros(h, dtype=bool)
    for surface in current:
        points[np.asarray(surface["beams"], dtype=np.int64)] = np.asarray(surface["points"])
        valid[np.asarray(surface["beams"], dtype=np.int64)] = True

    history = [FiniteSurfaceDistance(surfaces) for surfaces in frames[:-1]]
    candidates = velocity_candidates(candidate_max_speed, candidate_step)
    lags = timestamps[-1] - timestamps[:-1]
    distances = np.stack(
        [np.minimum(surface.distance(points[:, None] - lag * candidates[None]), 0.3)
         for surface, lag in zip(history, lags)],
        axis=0,
    )
    median = np.median(distances, axis=0)
    upper = np.quantile(distances, 0.75, axis=0)

    local_sum = np.zeros_like(median)
    neighborhood_count = np.zeros(h, dtype=np.float64)
    for shift in range(-neighborhood_beams, neighborhood_beams + 1):
        neighbor = np.roll(points, shift, axis=0)
        mask = valid & np.roll(valid, shift) & (np.linalg.norm(points - neighbor, axis=-1) <= neighborhood_radius)
        local_sum += np.roll(median, shift, axis=0) * mask[:, None]
        neighborhood_count += mask
    local = local_sum / np.maximum(neighborhood_count[:, None], 1.0)
    zero_index = int(np.argmin(np.linalg.norm(candidates, axis=-1)))
    zero = np.broadcast_to(median[:, zero_index:zero_index + 1], median.shape)
    speed = np.broadcast_to(np.linalg.norm(candidates, axis=-1)[None], median.shape)
    radial = points / np.maximum(np.linalg.norm(points, axis=-1, keepdims=True), 1e-6)
    radial_v = radial @ candidates.T
    tangent_v = radial[:, 0:1] * candidates[None, :, 1] - radial[:, 1:2] * candidates[None, :, 0]

    # Legacy scorer feature order: [median, upper, local, zero, improvement,
    # speed, radial velocity, tangential velocity], all distances / 0.1 m.
    features = np.stack(
        [median / 0.1, upper / 0.1, local / 0.1, zero / 0.1, (zero - median) / 0.1,
         speed, radial_v, tangent_v], axis=-1,
    ).astype(np.float32)
    support_frames = (distances <= 0.06).sum(axis=0).astype(np.float32)
    frame_spread = distances.std(axis=0).astype(np.float32)
    # A forward-compatible evidence tensor for the v2 scorer.  Keep the
    # legacy eight-dimensional ``features`` above unchanged so the existing
    # checkpoint can still be loaded; new training can consume this 11-D
    # tensor without rebuilding the geometry volume.
    neighborhood_feature = np.broadcast_to(neighborhood_count[:, None], median.shape)
    extended_features = np.stack(
        [median / 0.1, upper / 0.1, local / 0.1, zero / 0.1, (zero - median) / 0.1,
         speed, radial_v, tangent_v,
         support_frames / max(len(history), 1), frame_spread / 0.1,
         neighborhood_feature / max(2 * neighborhood_beams + 1, 1)], axis=-1,
    ).astype(np.float32)
    supported = valid & (neighborhood_count >= 2) & (sum(surface.tree is not None for surface in history) >= 2)
    return {
        "features": features,
        "extended_features": extended_features,
        "candidates": candidates,
        "points": points,
        "valid": valid,
        "supported": supported,
        "median_cost": median,
        "upper_cost": upper,
        "local_cost": local,
        "static_cost": zero,
        "support_frames": support_frames,
        "frame_spread": frame_spread,
        "neighborhood_count": neighborhood_count,
        "history": history,
        "lags": lags,
        "current_surfaces": current,
        "num_beams": h,
        "zero_index": zero_index,
    }
