"""Experimental shared candidate scorer over metric motion matching evidence.

Features contain no labels. Retains a distribution over fixed velocity
hypotheses; MAP velocity is only a diagnostic output, not a planner interface.
"""
import numpy as np
import torch
from torch import nn

from motion_geometry import compute_motion_cost_volume, velocity_candidates as _velocity_candidates


def velocity_candidates():
    return _velocity_candidates()


def matching_features(lidar, odom, timestamps):
    """[H,C,F] evidence; timestamps affect metric back-projection explicitly."""
    volume = compute_motion_cost_volume(lidar, odom, timestamps)
    return volume["features"], volume["supported"]


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
