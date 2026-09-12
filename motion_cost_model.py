"""Experimental shared candidate scorer over metric motion matching evidence.

Features contain no labels. Retains a distribution over fixed velocity
hypotheses; MAP velocity is only a diagnostic output, not a planner interface.
"""
import numpy as np
import torch
from torch import nn

from geometric_motion import GeometryConfig, extract_surfaces
from surface_motion import SurfaceDistance


def velocity_candidates():
    axis = np.arange(-1.5, 1.501, .25)
    grid = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
    return grid[np.linalg.norm(grid, axis=-1) <= 1.50001].astype(np.float32)


def matching_features(lidar, odom, timestamps):
    """[H,C,F] evidence; timestamps affect metric back-projection explicitly."""
    lidar, odom, timestamps = map(np.asarray, (lidar, odom, timestamps))
    if lidar.ndim != 2 or odom.shape != (len(lidar), 3) or timestamps.shape != (len(lidar),):
        raise ValueError('Expected [K,H], [K,3], [K].')
    if len(lidar) < 3 or not np.isfinite(timestamps).all() or not np.isfinite(odom).all() or not (np.diff(timestamps)>0).all():
        raise ValueError('Need finite poses and increasing timestamps.')
    frames = [extract_surfaces(s, p, odom[-1], GeometryConfig()) for s, p in zip(lidar, odom)]
    grid = velocity_candidates()
    h, c = lidar.shape[1], len(grid)
    xy = np.zeros((h, 2))
    valid = np.zeros(h, bool)
    for surface in frames[-1]:
        xy[surface['beams']] = surface['points']
        valid[surface['beams']] = True
    history = [SurfaceDistance(s) for s in frames[:-1]]
    distances = np.stack([np.minimum(s.distance(xy[:, None] - lag*grid[None]), .3)
                          for s, lag in zip(history, timestamps[-1]-timestamps[:-1])])
    median = np.median(distances, axis=0)
    upper = np.quantile(distances, .75, axis=0)
    local_sum = np.zeros((h, c))
    count = np.zeros(h)
    for shift in range(-3, 4):
        neighbor = np.roll(xy, shift, axis=0)
        mask = valid & np.roll(valid, shift) & (np.linalg.norm(xy-neighbor, axis=-1) <= .35)
        local_sum += np.roll(median, shift, axis=0)*mask[:, None]
        count += mask
    local = local_sum/np.maximum(count[:, None], 1)
    zero_index = np.argmin(np.linalg.norm(grid, axis=-1))
    zero = np.broadcast_to(median[:, zero_index:zero_index+1], median.shape)
    # All distance features are meters normalized by 0.1 m, not feature-space differences.
    costs = [median/.1, upper/.1, local/.1, zero/.1, (zero-median)/.1]
    speed = np.broadcast_to(np.linalg.norm(grid, axis=-1)[None], median.shape)
    radial = xy/np.maximum(np.linalg.norm(xy, axis=-1, keepdims=True), 1e-6)
    radial_v = radial @ grid.T
    tangent_v = radial[:, 0:1]*grid[None, :, 1]-radial[:, 1:2]*grid[None, :, 0]
    features = np.stack(costs+[speed, radial_v, tangent_v], axis=-1).astype(np.float32)
    supported = valid & (count >= 2) & (sum(s.tree is not None for s in history) >= 2)
    return features, supported


class MotionCandidateScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 48), nn.GELU(), nn.Linear(48, 48), nn.GELU(), nn.Linear(48, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('candidates', torch.from_numpy(velocity_candidates()))

    def forward(self, features):
        # Preserve geometric ranking initially; learn a shared correction, not
        # a beam-position embedding that can memorize scene locations.
        prior = -4.*features[..., 2] - .08*features[..., 5]
        return prior + self.net(features).squeeze(-1)

    def predict(self, features, supported):
        logits = self(features)
        velocity = self.candidates[logits.argmax(-1)]
        return velocity * supported[..., None], logits.softmax(-1)
