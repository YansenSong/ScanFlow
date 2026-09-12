"""Closed-loop sanity test: GT Motion Field + NMPC versus static-motion assumption.

This test isolates the planner. If GT obstacle velocity does not improve a
controlled crossing scenario, do not blame or tune the neural network yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planner import DynamicNMPC, NMPCConfig
from test.common import (
    integrate_unicycle,
    rotate_world_to_robot,
    transform_world_to_robot,
)


def rollout(use_gt_motion: bool, steps: int, dt: float):
    cfg = NMPCConfig(
        horizon=15,
        dt=dt,
        max_obstacles=8,
        max_obstacle_range=8.0,
        ipopt_max_iter=100,
        print_level=0,
    )
    planner = DynamicNMPC(cfg)

    robot = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    goal_world = np.array([6.0, 0.0], dtype=np.float64)

    # A crossing obstacle starts below the robot's path and moves upward.
    obstacle = np.array([2.8, -2.0], dtype=np.float64)
    obstacle_velocity_world = np.array([0.0, 0.65], dtype=np.float64)
    obstacle_radius = cfg.obstacle_radius

    prev_control = np.zeros(2, dtype=np.float64)
    min_clearance = float("inf")
    collisions = 0
    solver_failures = 0
    trajectory = []

    for _ in range(steps):
        goal_local = transform_world_to_robot(goal_world, robot)
        obstacle_local = transform_world_to_robot(obstacle, robot)
        velocity_local = rotate_world_to_robot(obstacle_velocity_world, robot[2])

        anchors = obstacle_local[None, :]
        velocity = velocity_local[None, :]
        confidence = np.ones((1, 1), dtype=np.float64) if use_gt_motion else np.zeros((1, 1), dtype=np.float64)

        v, w, info = planner.solve(
            robot_state=(0.0, 0.0, 0.0),
            goal=goal_local,
            anchors=anchors,
            velocity=velocity,
            confidence=confidence,
            prev_control=prev_control,
        )
        if not bool(info["success"]):
            solver_failures += 1

        robot = integrate_unicycle(robot, v, w, dt)
        obstacle = obstacle + obstacle_velocity_world * dt
        prev_control[:] = [v, w]

        center_distance = float(np.linalg.norm(robot[:2] - obstacle))
        clearance = center_distance - (cfg.robot_radius + obstacle_radius)
        min_clearance = min(min_clearance, clearance)
        if clearance < 0.0:
            collisions += 1

        trajectory.append([robot[0], robot[1], robot[2], obstacle[0], obstacle[1], v, w, clearance])
        if np.linalg.norm(goal_world - robot[:2]) < 0.30:
            break

    return {
        "min_clearance": min_clearance,
        "collisions": collisions,
        "solver_failures": solver_failures,
        "goal_distance": float(np.linalg.norm(goal_world - robot[:2])),
        "steps": len(trajectory),
        "trajectory": np.asarray(trajectory, dtype=np.float64),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=70)
    p.add_argument("--dt", type=float, default=0.15)
    p.add_argument("--required-clearance-gain", type=float, default=0.05)
    p.add_argument("--save", type=str, default=None, help="Optional .npz output for both trajectories.")
    args = p.parse_args()

    static = rollout(False, args.steps, args.dt)
    gt = rollout(True, args.steps, args.dt)

    print("GT Motion Field + NMPC planner isolation test")
    print(f"static-assumption : collisions={static['collisions']:3d} min_clearance={static['min_clearance']:+.3f} m "
          f"goal_dist={static['goal_distance']:.3f} m failures={static['solver_failures']}")
    print(f"GT-motion        : collisions={gt['collisions']:3d} min_clearance={gt['min_clearance']:+.3f} m "
          f"goal_dist={gt['goal_distance']:.3f} m failures={gt['solver_failures']}")

    if args.save:
        np.savez_compressed(args.save, static=static["trajectory"], gt=gt["trajectory"])

    collision_improved = gt["collisions"] < static["collisions"]
    clearance_improved = gt["min_clearance"] >= static["min_clearance"] + args.required_clearance_gain
    pass_test = gt["solver_failures"] == 0 and (collision_improved or clearance_improved)

    print("PASS" if pass_test else "FAIL",
          "- GT motion must reduce collisions or materially increase minimum clearance.")
    raise SystemExit(0 if pass_test else 1)


if __name__ == "__main__":
    main()
