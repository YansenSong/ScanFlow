"""Training-free surface association baseline; no semantic or GT inputs.

Preserves XY returns and their beam support. Velocity is in the current robot
frame. Match support is diagnostic evidence, NOT a calibrated probability.
Centroid drift under occlusion is an intentional limitation of this baseline.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from motion_geometry import GeometryConfig, extract_finite_surfaces

# Preserve the old public names for baseline scripts.
extract_surfaces = extract_finite_surfaces


def estimate_motion(lidar, odom, timestamps, cfg=None):
    cfg = cfg or GeometryConfig()
    lidar, odom, timestamps = map(np.asarray, (lidar, odom, timestamps))
    if lidar.ndim != 2 or odom.shape != (len(lidar), 3) or timestamps.shape != (len(lidar),):
        raise ValueError('Expected scans [K,H], odom [K,3], timestamps [K].')
    if len(lidar) < 2 or not np.all(np.diff(timestamps) > 0):
        raise ValueError('At least two strictly increasing timestamps are required.')
    frames = [extract_surfaces(scan, pose, odom[-1], cfg) for scan, pose in zip(lidar, odom)]
    current = frames[-1]
    observations = [[] for _ in current]
    for k, history in enumerate(frames[:-1]):
        if not current or not history:
            continue
        lag = timestamps[-1] - timestamps[k]
        a = np.array([g['anchor'] for g in current])
        b = np.array([g['anchor'] for g in history])
        distance = np.linalg.norm(a[:, None] - b[None], axis=-1)
        size_delta = np.abs(np.array([g['extent'] for g in current])[:, None]
                            - np.array([g['extent'] for g in history])[None])
        cost = distance + .5 * size_delta
        admissible = (distance <= cfg.max_speed * lag + .15) & (size_delta <= .4)
        admissible &= np.array([len(g['beams']) >= cfg.minimum_points for g in current])[:, None]
        admissible &= np.array([len(g['beams']) >= cfg.minimum_points for g in history])[None]
        cost = np.where(admissible, cost, 1e6)
        # Each current surface may explicitly remain unmatched.
        dummy = np.full((len(current), len(current)), cfg.max_speed * lag + .36)
        rows, cols = linear_sum_assignment(np.concatenate([cost, dummy], axis=1))
        for i, j in zip(rows, cols):
            if j < len(history) and admissible[i, j]:
                observations[i].append((lag, a[i] - b[j]))
    beam_velocity = np.zeros((lidar.shape[1], 2))
    beam_supported = np.zeros(lidar.shape[1], dtype=bool)
    result = []
    for surface, obs in zip(current, observations):
        velocity = np.zeros(2)
        supported = len(obs) >= 2
        residual = None
        if supported:
            lags = np.array([o[0] for o in obs])
            displacements = np.array([o[1] for o in obs])
            velocity = (lags[:, None] * displacements).sum(0) / (lags @ lags)
            residual = float(np.linalg.norm(displacements - lags[:, None] * velocity, axis=1).mean())
            supported = residual <= .12 and np.linalg.norm(velocity) <= cfg.max_speed
        if not supported:
            velocity = np.zeros(2)
        beam_velocity[surface['beams']] = velocity
        beam_supported[surface['beams']] = supported
        result.append(dict(**surface, velocity=velocity, supported=supported,
                           matched_frames=len(obs), fit_residual_m=residual))
    return dict(surfaces=result, beam_velocity=beam_velocity, beam_supported=beam_supported)
