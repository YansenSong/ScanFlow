"""Dynamic-aware NMPC for ScanFlow motion-field output.

Important semantics
-------------------
Current LiDAR geometry determines whether an obstacle token exists. Dynamic
confidence only determines how strongly the predicted velocity should be used.
Therefore static obstacles remain collision obstacles with approximately zero
predicted velocity instead of disappearing from the planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import math
import numpy as np

try:
    import casadi as ca
except ImportError as exc:
    raise ImportError("planner.py requires CasADi. Install it with: pip install casadi") from exc


@dataclass
class NMPCConfig:
    horizon: int = 15
    dt: float = 0.15
    v_min: float = 0.0
    v_max: float = 1.0
    w_min: float = -1.5
    w_max: float = 1.5
    a_v_max: float = 0.8
    a_w_max: float = 2.5

    robot_radius: float = 0.30
    obstacle_radius: float = 0.20
    safety_margin: float = 0.15

    # Retained for backward config compatibility; no obstacle is filtered by
    # dynamic confidence anymore.
    confidence_threshold: float = 0.0
    max_obstacles: int = 24
    max_obstacle_range: float = 8.0

    w_goal_xy: float = 8.0
    w_terminal_xy: float = 20.0
    w_heading: float = 1.2
    w_control_v: float = 0.10
    w_control_w: float = 0.08
    w_delta_v: float = 0.50
    w_delta_w: float = 0.25
    w_collision: float = 30.0
    w_safety_slack: float = 250.0
    w_progress: float = 0.5
    collision_softness: float = 12.0

    ipopt_max_iter: int = 80
    ipopt_tol: float = 1e-3
    print_level: int = 0


def wrap_angle_numeric(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _to_numpy(x, dtype=np.float64) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


class DynamicNMPC:
    def __init__(self, cfg: Optional[NMPCConfig] = None):
        self.cfg = cfg or NMPCConfig()
        self.safe_radius = self.cfg.robot_radius + self.cfg.obstacle_radius + self.cfg.safety_margin
        self._last_solution_u: Optional[np.ndarray] = None
        self._build_solver()

    def _build_solver(self) -> None:
        c, N, M = self.cfg, self.cfg.horizon, self.cfg.max_obstacles
        opti = ca.Opti()
        X = opti.variable(3, N + 1)
        U = opti.variable(2, N)
        S = opti.variable(M, N)

        x0 = opti.parameter(3)
        goal = opti.parameter(2)
        u_prev = opti.parameter(2)
        obs_pos = opti.parameter(2, M)
        obs_vel = opti.parameter(2, M)
        obs_conf = opti.parameter(1, M)
        obs_valid = opti.parameter(1, M)

        opti.subject_to(X[:, 0] == x0)
        opti.subject_to(opti.bounded(c.v_min, U[0, :], c.v_max))
        opti.subject_to(opti.bounded(c.w_min, U[1, :], c.w_max))
        opti.subject_to(ca.vec(S) >= 0)

        total_cost = 0
        for k in range(N):
            xk, yk, thk = X[0, k], X[1, k], X[2, k]
            vk, wk = U[0, k], U[1, k]
            opti.subject_to(X[0, k + 1] == xk + c.dt * vk * ca.cos(thk))
            opti.subject_to(X[1, k + 1] == yk + c.dt * vk * ca.sin(thk))
            opti.subject_to(X[2, k + 1] == thk + c.dt * wk)

            if k == 0:
                dv, dw = vk - u_prev[0], wk - u_prev[1]
            else:
                dv, dw = vk - U[0, k - 1], wk - U[1, k - 1]
            opti.subject_to(opti.bounded(-c.a_v_max * c.dt, dv, c.a_v_max * c.dt))
            opti.subject_to(opti.bounded(-c.a_w_max * c.dt, dw, c.a_w_max * c.dt))

            dxg, dyg = xk - goal[0], yk - goal[1]
            dist_goal_sq = dxg * dxg + dyg * dyg
            total_cost += c.w_goal_xy * dist_goal_sq
            # Avoid atan2(goal-y, goal-x): its derivative is undefined when a
            # predicted state lands exactly on the goal, which commonly
            # happens once the goal enters the finite horizon.  The normalized
            # 2-D cross product is a smooth heading-error surrogate and
            # naturally vanishes at the goal.
            to_goal_x, to_goal_y = goal[0] - xk, goal[1] - yk
            heading_cross = ca.cos(thk) * to_goal_y - ca.sin(thk) * to_goal_x
            total_cost += c.w_heading * heading_cross * heading_cross / (dist_goal_sq + 1e-4)
            goal_dist = ca.sqrt(dist_goal_sq + 1e-6)
            total_cost -= c.w_progress * goal_dist * vk
            total_cost += c.w_control_v * vk * vk + c.w_control_w * wk * wk
            total_cost += c.w_delta_v * dv * dv + c.w_delta_w * dw * dw

            tau = float(k + 1) * c.dt
            xr, yr = X[0, k + 1], X[1, k + 1]
            for j in range(M):
                # Confidence gates motion, not obstacle existence.
                ox = obs_pos[0, j] + tau * obs_conf[0, j] * obs_vel[0, j]
                oy = obs_pos[1, j] + tau * obs_conf[0, j] * obs_vel[1, j]
                dist = ca.sqrt((xr - ox) ** 2 + (yr - oy) ** 2 + 1e-6)
                valid = obs_valid[0, j]

                # For padded entries valid=0, this becomes trivially satisfied.
                opti.subject_to(dist + S[j, k] + (1.0 - valid) * self.safe_radius >= self.safe_radius)
                # Penalize every slack quadratically, including padded
                # entries.  Gating this term by ``valid`` leaves all padded
                # slacks as cost-free degrees of freedom and produces a
                # rank-deficient NLP when obstacles leave sensor range.
                total_cost += c.w_safety_slack * (
                    valid * S[j, k] + 0.1 * S[j, k] * S[j, k]
                )

                penetration = self.safe_radius - dist
                beta = c.collision_softness
                soft_pen = ca.log(1.0 + ca.exp(beta * penetration)) / beta
                total_cost += c.w_collision * valid * soft_pen * soft_pen

        dxN, dyN = X[0, N] - goal[0], X[1, N] - goal[1]
        total_cost += c.w_terminal_xy * (dxN * dxN + dyN * dyN)
        opti.minimize(total_cost)

        opti.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            {"max_iter": c.ipopt_max_iter, "tol": c.ipopt_tol, "print_level": c.print_level, "sb": "yes"},
        )
        self.opti, self.X, self.U, self.S = opti, X, U, S
        self.p_x0, self.p_goal, self.p_u_prev = x0, goal, u_prev
        self.p_obs_pos, self.p_obs_vel = obs_pos, obs_vel
        self.p_obs_conf, self.p_obs_valid = obs_conf, obs_valid

    def _prepare_obstacles(self, anchors, velocity, confidence):
        c, M = self.cfg, self.cfg.max_obstacles
        pos = _to_numpy(anchors)
        vel = _to_numpy(velocity)
        conf = _to_numpy(confidence).reshape(-1)
        if pos.ndim == 3:
            if pos.shape[0] != 1:
                raise ValueError("planner.solve supports one planning instance at a time.")
            pos = pos[0]
        if vel.ndim == 3:
            if vel.shape[0] != 1:
                raise ValueError("planner.solve supports one planning instance at a time.")
            vel = vel[0]
        if pos.ndim != 2 or pos.shape[1] != 2 or vel.shape != pos.shape or conf.shape[0] != pos.shape[0]:
            raise ValueError("Expected anchors/velocity [P,2] and confidence [P] or [P,1].")

        ranges = np.linalg.norm(pos, axis=1)
        keep = np.isfinite(pos).all(axis=1) & np.isfinite(vel).all(axis=1) & np.isfinite(conf)
        keep &= (ranges <= c.max_obstacle_range) & (ranges > 1e-4)
        indices = np.flatnonzero(keep)
        if len(indices) > M:
            # Occupancy proximity dominates; dynamic confidence only breaks ties.
            score = 1.0 / (ranges[indices] + 0.2) + 0.25 * np.clip(conf[indices], 0.0, 1.0)
            indices = indices[np.argsort(-score)[:M]]

        n = len(indices)
        pos_pad = np.zeros((2, M), dtype=np.float64)
        vel_pad = np.zeros((2, M), dtype=np.float64)
        conf_pad = np.zeros((1, M), dtype=np.float64)
        valid_pad = np.zeros((1, M), dtype=np.float64)
        if n:
            pos_pad[:, :n] = pos[indices].T
            vel_pad[:, :n] = vel[indices].T
            conf_pad[0, :n] = np.clip(conf[indices], 0.0, 1.0)
            valid_pad[0, :n] = 1.0
        return pos_pad, vel_pad, conf_pad, valid_pad, indices

    def _rollout_guess(self, state: np.ndarray, U0: np.ndarray) -> np.ndarray:
        c, N = self.cfg, self.cfg.horizon
        X0 = np.zeros((3, N + 1), dtype=np.float64)
        X0[:, 0] = state
        for k in range(N):
            v, w = U0[:, k]
            X0[0, k + 1] = X0[0, k] + c.dt * v * math.cos(X0[2, k])
            X0[1, k + 1] = X0[1, k] + c.dt * v * math.sin(X0[2, k])
            X0[2, k + 1] = X0[2, k] + c.dt * w
        return X0

    def _initial_guess(self, robot_state: np.ndarray, goal: np.ndarray, prev_control: np.ndarray):
        c, N = self.cfg, self.cfg.horizon
        if self._last_solution_u is not None and self._last_solution_u.shape == (2, N):
            U0 = np.concatenate([self._last_solution_u[:, 1:], self._last_solution_u[:, -1:]], axis=1)
            U0[0] = np.clip(U0[0], c.v_min, c.v_max)
            U0[1] = np.clip(U0[1], c.w_min, c.w_max)
            # Re-rollout in the *current* planning frame. Reusing the previous X
            # directly is incorrect for robot-centric planning frames.
            return self._rollout_guess(robot_state, U0), U0

        desired = math.atan2(goal[1] - robot_state[1], goal[0] - robot_state[0])
        heading_err = wrap_angle_numeric(desired - robot_state[2])
        nominal_v = min(c.v_max * 0.5, max(c.v_min, np.linalg.norm(goal - robot_state[:2])))
        nominal_w = np.clip(heading_err / max(N * c.dt, 1e-6), c.w_min, c.w_max)
        nominal_v = np.clip(nominal_v, prev_control[0] - c.a_v_max * c.dt, prev_control[0] + c.a_v_max * c.dt)
        nominal_w = np.clip(nominal_w, prev_control[1] - c.a_w_max * c.dt, prev_control[1] + c.a_w_max * c.dt)
        U0 = np.tile([[np.clip(nominal_v, c.v_min, c.v_max)], [np.clip(nominal_w, c.w_min, c.w_max)]], (1, N))
        return self._rollout_guess(robot_state, U0), U0

    def solve(
        self,
        robot_state: Sequence[float],
        goal: Sequence[float],
        anchors,
        velocity,
        confidence,
        prev_control: Sequence[float] = (0.0, 0.0),
    ) -> Tuple[float, float, Dict[str, np.ndarray]]:
        c = self.cfg
        state = _to_numpy(robot_state).reshape(3)
        goal_np = _to_numpy(goal).reshape(2)
        prev_u = _to_numpy(prev_control).reshape(2)
        pos, vel, conf, valid, kept = self._prepare_obstacles(anchors, velocity, confidence)

        self.opti.set_value(self.p_x0, state)
        self.opti.set_value(self.p_goal, goal_np)
        self.opti.set_value(self.p_u_prev, prev_u)
        self.opti.set_value(self.p_obs_pos, pos)
        self.opti.set_value(self.p_obs_vel, vel)
        self.opti.set_value(self.p_obs_conf, conf)
        self.opti.set_value(self.p_obs_valid, valid)

        X0, U0 = self._initial_guess(state, goal_np, prev_u)
        self.opti.set_initial(self.X, X0)
        self.opti.set_initial(self.U, U0)
        self.opti.set_initial(self.S, 0.0)
        # Do not carry IPOPT's constraint multipliers across robot-centric
        # replans: every call changes the goal/obstacle parameters and the old
        # dual point can be badly inconsistent with the new NLP instance.
        self.opti.set_initial(self.opti.lam_g, 0.0)

        success, status = True, "Solve_Succeeded"
        objective = float("nan")
        X_sol, U_sol = X0, U0
        try:
            sol = self.opti.solve()
            X_sol = np.asarray(sol.value(self.X), dtype=np.float64)
            U_sol = np.asarray(sol.value(self.U), dtype=np.float64)
            objective = float(sol.value(self.opti.f))
            self._last_solution_u = U_sol
            v_cmd = float(np.clip(U_sol[0, 0], c.v_min, c.v_max))
            w_cmd = float(np.clip(U_sol[1, 0], c.w_min, c.w_max))
            if not np.isfinite(v_cmd) or not np.isfinite(w_cmd):
                raise RuntimeError("Solver returned non-finite control.")
        except RuntimeError:
            success = False
            try:
                status = str(self.opti.stats().get("return_status", "Failed"))
            except Exception:
                status = "Failed"
            # Never execute an unvalidated Opti debug iterate. Safe default is stop.
            v_cmd, w_cmd = 0.0, 0.0
            self._last_solution_u = None

        info = {
            "success": np.array(success),
            "status": np.array(status),
            "objective": np.array(objective),
            "predicted_states": X_sol,
            "predicted_controls": U_sol,
            "kept_token_indices": kept,
            "obstacle_positions": pos,
            "obstacle_velocities": vel,
            "obstacle_confidence": conf,
            "obstacle_valid": valid,
        }
        return v_cmd, w_cmd, info


def planner_inputs_from_model_output(model_output: Dict, batch_index: int = 0):
    anchors = _to_numpy(model_output["anchors"])
    velocity = _to_numpy(model_output["velocity"])
    confidence = _to_numpy(model_output["confidence"])
    if anchors.ndim == 3:
        anchors = anchors[batch_index]
    if velocity.ndim == 3:
        velocity = velocity[batch_index]
    if confidence.ndim == 3:
        confidence = confidence[batch_index]
    return anchors, velocity, confidence


def _smoke_test() -> None:
    cfg = NMPCConfig(horizon=8, max_obstacles=12, ipopt_max_iter=60, print_level=0)
    planner = DynamicNMPC(cfg)
    P = 36
    anchors = np.zeros((P, 2), dtype=np.float64)
    velocity = np.zeros((P, 2), dtype=np.float64)
    confidence = np.zeros((P, 1), dtype=np.float64)

    # Static obstacle must be kept despite zero dynamic confidence.
    anchors[0] = [1.2, 0.0]
    confidence[0, 0] = 0.0
    # Dynamic token.
    anchors[1] = [1.8, -0.6]
    velocity[1] = [0.0, 0.5]
    confidence[1, 0] = 0.9
    _, _, _, valid, kept = planner._prepare_obstacles(anchors, velocity, confidence)
    assert 0 in kept and valid.sum() == 2

    v_cmd, w_cmd, info = planner.solve([0, 0, 0], [3, 0], anchors, velocity, confidence, [0, 0])
    assert np.isfinite(v_cmd) and np.isfinite(w_cmd)
    print("planner.py smoke test passed", bool(info["success"]), info["status"].item())


if __name__ == "__main__":
    _smoke_test()
