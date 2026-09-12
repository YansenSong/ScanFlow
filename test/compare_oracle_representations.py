"""Controlled closed-loop comparison with scan-derived geometry and oracle motion.

Both representations use identical NMPC parameters and obstacle budgets.
The fixed planner radius is retained, so this is a diagnostic, not a safety proof.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_dataset import Circle, GeneratorConfig, Segment, cast_lidar, current_beam_labels
from planner import DynamicNMPC, NMPCConfig
from test.audit_motion_representation import groups_for_scan
from test.common import integrate_unicycle, transform_world_to_robot


def rollout(segmented, budget):
    cfg = NMPCConfig(max_obstacles=budget, ipopt_max_iter=100)
    planner = DynamicNMPC(cfg)
    sensor = GeneratorConfig(num_beams=180, lidar_noise_std=0.0, min_dynamic_beams=0)
    angles = sensor.angle_min + np.arange(sensor.num_beams) * sensor.angle_increment
    unit = np.stack([np.cos(angles), np.sin(angles)], -1)
    # Far wall shares angular patches with the near crossing object.
    walls = [Segment(np.array([7., -5.]), np.array([7., 5.]), 0)]
    robot = np.zeros(3)
    goal = np.array([6., 0.])
    center = np.array([2.8, -2.])
    motion = np.array([0., .65])
    previous = np.zeros(2)
    failures = collisions = truncations = 0
    minimum = float('inf')
    for step in range(70):
        circle = Circle(center.copy(), .2, motion, True, 100)
        scan, ids = cast_lidar(robot, angles, walls, [circle], 0., sensor)
        velocity, dynamic, valid = current_beam_labels(ids, [circle], robot[2])
        valid = valid > .5
        points = np.where(valid, scan, 0)[:, None] * unit
        groups = groups_for_scan(points, valid, 36, segmented)
        anchors = np.array([points[g].mean(0) for g in groups]).reshape(-1, 2)
        velocities = np.array([velocity[g[dynamic[g] > .5]].mean(0)
                               if (dynamic[g] > .5).any() else np.zeros(2)
                               for g in groups]).reshape(-1, 2)
        confidence = np.array([float((dynamic[g] > .5).any()) for g in groups])
        truncations += int(((np.linalg.norm(anchors, axis=1) <= cfg.max_obstacle_range)).sum() > budget)
        v, w, info = planner.solve(np.zeros(3), transform_world_to_robot(goal, robot),
                                   anchors, velocities, confidence, previous)
        failures += int(not info['success'])
        # Check substeps to avoid overlooking between-sample collisions.
        for fraction in np.linspace(0., 1., 11):
            position = robot[:2] + fraction * cfg.dt * v * np.array([np.cos(robot[2]), np.sin(robot[2])])
            clearance = np.linalg.norm(position - (center + fraction * cfg.dt * motion)) - .5
            minimum = min(minimum, float(clearance))
        robot = integrate_unicycle(robot, v, w, cfg.dt)
        center += cfg.dt * motion
        collisions += int(np.linalg.norm(robot[:2] - center) < .5)
        previous[:] = [v, w]
        if np.linalg.norm(goal - robot[:2]) < .3:
            break
    return dict(representation='segments' if segmented else 'patches', budget=budget,
                failures=failures, collision_steps=collisions, min_clearance_m=minimum,
                goal_distance_m=float(np.linalg.norm(goal - robot[:2])),
                steps=step + 1, truncated_steps=truncations)


if __name__ == '__main__':
    results = []
    for budget in (24, 64):
        for segmented in (False, True):
            result = rollout(segmented, budget)
            results.append(result)
            print(json.dumps(result), flush=True)
    Path('/tmp/scanflow_oracle_representation_rollouts.json').write_text(json.dumps(results, indent=2) + '\n')
