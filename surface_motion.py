"""Local constant-velocity hypotheses scored against aligned historical surfaces.

No object IDs, shape class, learned features or centroid association. Each
current return keeps its own velocity. Matching uses finite line segments,
not infinite tangent lines. Occluded/unmatched estimates remain unsupported.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from geometric_motion import GeometryConfig, extract_surfaces


@dataclass
class SurfaceConfig:
    max_speed: float = 1.5
    grid_step: float = .25
    refine_step: float = .05
    neighborhood_radius: float = .35
    neighborhood_beams: int = 3
    noise_tolerance: float = .025
    max_residual: float = .06
    min_improvement: float = .015


class SurfaceDistance:
    def __init__(self, surfaces):
        starts, ends = [], []
        for surface in surfaces:
            p = surface['points']
            if len(p) >= 2:
                starts.extend(p[:-1])
                ends.extend(p[1:])
        self.starts = np.asarray(starts).reshape(-1, 2)
        self.ends = np.asarray(ends).reshape(-1, 2)
        self.tree = cKDTree((self.starts + self.ends)/2) if len(starts) else None

    def distance(self, points):
        if self.tree is None:
            return np.full(points.shape[:-1], np.inf)
        shape = points.shape[:-1]
        points = points.reshape(-1, 2)
        # Nearby midpoints propose finite surfaces; no cross-gap interpolation.
        _, indices = self.tree.query(points, k=min(8, len(self.starts)))
        indices = np.asarray(indices).reshape(len(points), -1)
        a, b = self.starts[indices], self.ends[indices]
        ab = b-a
        t = ((points[:, None]-a)*ab).sum(-1) / np.maximum((ab*ab).sum(-1), 1e-12)
        projection = a + np.clip(t, 0., 1.)[..., None]*ab
        return np.linalg.norm(points[:, None]-projection, axis=-1).min(-1).reshape(shape)


def estimate_surface_motion(lidar, odom, timestamps, cfg=None):
    cfg = cfg or SurfaceConfig()
    lidar, odom, timestamps = map(np.asarray, (lidar, odom, timestamps))
    if lidar.ndim != 2 or odom.shape != (len(lidar), 3) or timestamps.shape != (len(lidar),):
        raise ValueError('Expected scans [K,H], odom [K,3], timestamps [K].')
    if len(lidar) < 3 or not np.isfinite(odom).all() or not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps)>0):
        raise ValueError('Need finite poses and at least three increasing timestamps.')
    geometry = GeometryConfig()
    frames = [extract_surfaces(scan, pose, odom[-1], geometry) for scan, pose in zip(lidar, odom)]
    history = [SurfaceDistance(f) for f in frames[:-1]]
    h = lidar.shape[1]
    points = np.zeros((h, 2))
    valid = np.zeros(h, dtype=bool)
    for s in frames[-1]:
        points[s['beams']] = s['points']
        valid[s['beams']] = True
    velocity = np.zeros((h, 2))
    supported = np.zeros(h, dtype=bool)
    residual = np.full(h, np.nan)
    improvement = np.full(h, np.nan)
    axis = np.arange(-cfg.max_speed, cfg.max_speed+cfg.grid_step/2, cfg.grid_step)
    grid = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
    grid = grid[np.linalg.norm(grid, axis=-1) <= cfg.max_speed+1e-8]
    offsets = np.arange(-cfg.grid_step, cfg.grid_step+cfg.refine_step/2, cfg.refine_step)
    offsets = np.stack(np.meshgrid(offsets, offsets), -1).reshape(-1, 2)
    lags = timestamps[-1]-timestamps[:-1]
    for i in np.flatnonzero(valid):
        ids = (i + np.arange(-cfg.neighborhood_beams, cfg.neighborhood_beams+1)) % h
        ids = ids[valid[ids] & (np.linalg.norm(points[ids]-points[i], axis=-1) <= cfg.neighborhood_radius)]
        if len(ids) < 2:
            continue
        local = points[ids]

        def score(candidates):
            # Median across frames tolerates missing overlap, but at least two
            # frames must have surfaces. Fit score retains real metric units.
            distances = np.stack([surface.distance(local[None]-lag*candidates[:, None])
                                  for surface, lag in zip(history, lags)], axis=1)
            per_frame = np.minimum(distances, .3).mean(-1)
            return np.median(per_frame, axis=1)

        if sum(s.tree is not None for s in history) < 2:
            continue
        static = float(score(np.zeros((1, 2)))[0])
        if static <= cfg.noise_tolerance:
            supported[i] = True
            residual[i] = static
            improvement[i] = 0.
            continue
        costs = score(grid)
        best = grid[np.argmin(costs + .002*np.linalg.norm(grid, axis=1))]
        fine = best + offsets
        fine = fine[np.linalg.norm(fine, axis=1) <= cfg.max_speed+1e-8]
        costs = score(fine)
        index = np.argmin(costs + .002*np.linalg.norm(fine, axis=1))
        residual[i] = costs[index]
        improvement[i] = static-costs[index]
        supported[i] = costs[index] <= cfg.max_residual and improvement[i] >= cfg.min_improvement
        if supported[i]:
            velocity[i] = fine[index]
    return dict(beam_velocity=velocity, beam_supported=supported,
                fit_residual_m=residual, static_improvement_m=improvement)
