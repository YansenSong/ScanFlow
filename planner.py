"""
planner.py

Dynamic-aware NMPC planner for the motion-field output of model.py.

Expected neural-network outputs
-------------------------------
anchors      : [P, 2] or [B, P, 2]
velocity     : [P, 2] or [B, P, 2]
confidence   : [P, 1] or [B, P, 1]

Each token j represents:
    anchor_j = [x_j, y_j]              current obstacle/scene anchor position
    vel_j    = [vx_j, vy_j]            estimated local scene motion
    conf_j   = dynamic confidence      in [0, 1]

The NMPC assumes constant-velocity token motion over its finite horizon:
    p_j(k) = anchor_j + vel_j * (k * dt)

Robot model
-----------
Unicycle / differential-drive kinematics:
    x_{k+1}     = x_k + v_k cos(theta_k) dt
    y_{k+1}     = y_k + v_k sin(theta_k) dt
    theta_{k+1} = theta_k + w_k dt

Optimization
------------
Minimize a weighted sum of:
    - goal tracking
    - heading alignment
    - control effort
    - control smoothness
    - soft dynamic-collision penalties

Subject to:
    - v / w bounds
    - acceleration bounds
    - kinematic rollout

This file intentionally stays independent from PyTorch.
It uses CasADi + IPOPT.

Install:
    pip install casadi numpy

Typical usage
-------------
from planner import DynamicNMPC, NMPCConfig

planner = DynamicNMPC(NMPCConfig())

v_cmd, w_cmd, info = planner.solve(
    robot_state=[0.0, 0.0, 0.0],
    goal=[3.0, 1.0],
    anchors=anchors_np,          # [P,2]
    velocity=velocity_np,        # [P,2]
    confidence=confidence_np,    # [P,1]
    prev_control=[0.0, 0.0],
)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import math
import numpy as np

try:
    import casadi as ca
except ImportError as exc:
    raise ImportError(
        "planner.py requires CasADi. Install it with: pip install casadi"
    ) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NMPCConfig:
    # Horizon
    horizon: int = 15
    dt: float = 0.15

    # Differential-drive / unicycle control limits
    v_min: float = 0.0
    v_max: float = 1.0
    w_min: float = -1.5
    w_max: float = 1.5

    # Control-rate limits
    a_v_max: float = 0.8        # |dv/dt|  [m/s^2]
    a_w_max: float = 2.5        # |dw/dt|  [rad/s^2]

    # Robot / collision geometry
    robot_radius: float = 0.30
    obstacle_radius: float = 0.20
    safety_margin: float = 0.15

    # Token filtering
    confidence_threshold: float = 0.30
    max_obstacles: int = 24
    max_obstacle_range: float = 8.0

    # Cost weights
    w_goal_xy: float = 8.0
    w_terminal_xy: float = 20.0
    w_heading: float = 1.2
    w_control_v: float = 0.10
    w_control_w: float = 0.08
    w_delta_v: float = 0.50
    w_delta_w: float = 0.25
    w_collision: float = 30.0
    w_progress: float = 0.5

    # Collision penalty shaping
    collision_softness: float = 12.0

    # IPOPT
    ipopt_max_iter: int = 80
    ipopt_tol: float = 1e-3
    print_level: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wrap_angle_numeric(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _to_numpy(x, dtype=np.float64) -> np.ndarray:
    """
    Accept numpy / list / torch tensor without importing torch.
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


# ---------------------------------------------------------------------------
# NMPC
# ---------------------------------------------------------------------------

class DynamicNMPC:
    """
    Dynamic-aware nonlinear MPC.

    Network output interface:
        anchors    [P,2]
        velocity   [P,2]
        confidence [P] or [P,1]

    Planner output:
        (v_cmd, w_cmd, info)
    """

    def __init__(self, cfg: Optional[NMPCConfig] = None):
        self.cfg = cfg if cfg is not None else NMPCConfig()

        # Effective safety radius between robot center and token anchor.
        self.safe_radius = (
            self.cfg.robot_radius
            + self.cfg.obstacle_radius
            + self.cfg.safety_margin
        )

        self._last_solution_u: Optional[np.ndarray] = None
        self._last_solution_x: Optional[np.ndarray] = None

        self._build_solver()

    # ---------------------------------------------------------------------
    # Solver construction
    # ---------------------------------------------------------------------

    def _build_solver(self) -> None:
        c = self.cfg
        N = c.horizon
        M = c.max_obstacles

        opti = ca.Opti()

        # Decision variables
        # X = [x, y, theta] over N+1 states
        X = opti.variable(3, N + 1)

        # U = [v, w] over N controls
        U = opti.variable(2, N)

        # Parameters
        x0 = opti.parameter(3)             # initial robot state
        goal = opti.parameter(2)           # goal [xg, yg]
        u_prev = opti.parameter(2)         # previous [v,w]

        # Dynamic tokens:
        # current anchors, velocities, confidences, validity.
        obs_pos = opti.parameter(2, M)
        obs_vel = opti.parameter(2, M)
        obs_conf = opti.parameter(1, M)
        obs_valid = opti.parameter(1, M)

        # Initial-state equality.
        opti.subject_to(X[:, 0] == x0)

        # Control bounds.
        opti.subject_to(opti.bounded(c.v_min, U[0, :], c.v_max))
        opti.subject_to(opti.bounded(c.w_min, U[1, :], c.w_max))

        total_cost = 0

        for k in range(N):
            xk = X[0, k]
            yk = X[1, k]
            thk = X[2, k]
            vk = U[0, k]
            wk = U[1, k]

            # -------------------------------------------------------------
            # Unicycle dynamics
            # -------------------------------------------------------------
            x_next = xk + c.dt * vk * ca.cos(thk)
            y_next = yk + c.dt * vk * ca.sin(thk)
            th_next = thk + c.dt * wk

            opti.subject_to(X[0, k + 1] == x_next)
            opti.subject_to(X[1, k + 1] == y_next)
            opti.subject_to(X[2, k + 1] == th_next)

            # -------------------------------------------------------------
            # Control-rate constraints
            # -------------------------------------------------------------
            if k == 0:
                dv = vk - u_prev[0]
                dw = wk - u_prev[1]
            else:
                dv = vk - U[0, k - 1]
                dw = wk - U[1, k - 1]

            opti.subject_to(
                opti.bounded(
                    -c.a_v_max * c.dt,
                    dv,
                    c.a_v_max * c.dt,
                )
            )
            opti.subject_to(
                opti.bounded(
                    -c.a_w_max * c.dt,
                    dw,
                    c.a_w_max * c.dt,
                )
            )

            # -------------------------------------------------------------
            # Goal tracking
            # -------------------------------------------------------------
            dxg = xk - goal[0]
            dyg = yk - goal[1]

            dist_goal_sq = dxg * dxg + dyg * dyg
            total_cost += c.w_goal_xy * dist_goal_sq

            # Heading should face goal.
            desired_heading = ca.atan2(goal[1] - yk, goal[0] - xk)
            heading_err = ca.atan2(
                ca.sin(thk - desired_heading),
                ca.cos(thk - desired_heading),
            )
            total_cost += c.w_heading * heading_err * heading_err

            # Small incentive for progress / forward motion when far away.
            goal_dist = ca.sqrt(dist_goal_sq + 1e-6)
            total_cost -= c.w_progress * goal_dist * vk

            # -------------------------------------------------------------
            # Control effort + smoothness
            # -------------------------------------------------------------
            total_cost += c.w_control_v * vk * vk
            total_cost += c.w_control_w * wk * wk
            total_cost += c.w_delta_v * dv * dv
            total_cost += c.w_delta_w * dw * dw

            # -------------------------------------------------------------
            # Dynamic collision penalties
            # -------------------------------------------------------------
            # Constant-velocity obstacle/token prediction.
            tau = float(k + 1) * c.dt

            xr = X[0, k + 1]
            yr = X[1, k + 1]

            for j in range(M):
                ox = obs_pos[0, j] + tau * obs_vel[0, j]
                oy = obs_pos[1, j] + tau * obs_vel[1, j]

                ddx = xr - ox
                ddy = yr - oy
                d2 = ddx * ddx + ddy * ddy
                dist = ca.sqrt(d2 + 1e-6)

                # Positive when inside safety radius.
                penetration = self.safe_radius - dist

                # Smooth ReLU / softplus:
                # softplus(beta * penetration) / beta
                beta = c.collision_softness
                soft_pen = ca.log(1.0 + ca.exp(beta * penetration)) / beta

                # Confidence and token validity modulate penalty.
                weight = obs_conf[0, j] * obs_valid[0, j]

                total_cost += (
                    c.w_collision
                    * weight
                    * soft_pen
                    * soft_pen
                )

        # Terminal goal cost.
        dxN = X[0, N] - goal[0]
        dyN = X[1, N] - goal[1]
        total_cost += c.w_terminal_xy * (dxN * dxN + dyN * dyN)

        opti.minimize(total_cost)

        # IPOPT settings.
        p_opts = {
            "expand": True,
            "print_time": False,
        }
        s_opts = {
            "max_iter": c.ipopt_max_iter,
            "tol": c.ipopt_tol,
            "print_level": c.print_level,
            "sb": "yes",
        }
        opti.solver("ipopt", p_opts, s_opts)

        # Store symbolic handles.
        self.opti = opti
        self.X = X
        self.U = U

        self.p_x0 = x0
        self.p_goal = goal
        self.p_u_prev = u_prev
        self.p_obs_pos = obs_pos
        self.p_obs_vel = obs_vel
        self.p_obs_conf = obs_conf
        self.p_obs_valid = obs_valid

    # ---------------------------------------------------------------------
    # Token preprocessing
    # ---------------------------------------------------------------------

    def _prepare_obstacles(
        self,
        anchors,
        velocity,
        confidence,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter and pad motion-field tokens to max_obstacles.

        Returns
        -------
        pos    : [2,M]
        vel    : [2,M]
        conf   : [1,M]
        valid  : [1,M]
        kept_indices : [n_kept]
        """
        c = self.cfg
        M = c.max_obstacles

        pos = _to_numpy(anchors)
        vel = _to_numpy(velocity)
        conf = _to_numpy(confidence).reshape(-1)

        # Gracefully allow [B,P,*] with B=1.
        if pos.ndim == 3:
            if pos.shape[0] != 1:
                raise ValueError(
                    "planner.solve expects one planning instance at a time. "
                    "If anchors is batched, batch size must be 1."
                )
            pos = pos[0]

        if vel.ndim == 3:
            if vel.shape[0] != 1:
                raise ValueError(
                    "planner.solve expects one planning instance at a time."
                )
            vel = vel[0]

        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError(f"anchors must be [P,2], got {pos.shape}")
        if vel.shape != pos.shape:
            raise ValueError(
                f"velocity must have same [P,2] shape as anchors, got {vel.shape}"
            )
        if conf.shape[0] != pos.shape[0]:
            raise ValueError(
                "confidence length must match number of anchors."
            )

        finite = (
            np.isfinite(pos).all(axis=1)
            & np.isfinite(vel).all(axis=1)
            & np.isfinite(conf)
        )

        ranges = np.linalg.norm(pos, axis=1)

        keep = finite
        keep &= conf >= c.confidence_threshold
        keep &= ranges <= c.max_obstacle_range
        keep &= ranges > 1e-4

        indices = np.flatnonzero(keep)

        # Keep the highest-risk / highest-confidence near tokens.
        if len(indices) > M:
            # Simple risk score: high confidence + proximity.
            score = conf[indices] / (ranges[indices] + 0.2)
            order = np.argsort(-score)
            indices = indices[order[:M]]

        n = len(indices)

        pos_pad = np.zeros((2, M), dtype=np.float64)
        vel_pad = np.zeros((2, M), dtype=np.float64)
        conf_pad = np.zeros((1, M), dtype=np.float64)
        valid_pad = np.zeros((1, M), dtype=np.float64)

        if n > 0:
            pos_pad[:, :n] = pos[indices].T
            vel_pad[:, :n] = vel[indices].T
            conf_pad[0, :n] = np.clip(conf[indices], 0.0, 1.0)
            valid_pad[0, :n] = 1.0

        return pos_pad, vel_pad, conf_pad, valid_pad, indices

    # ---------------------------------------------------------------------
    # Warm start
    # ---------------------------------------------------------------------

    def _initial_guess(
        self,
        robot_state: np.ndarray,
        goal: np.ndarray,
        prev_control: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        N = c.horizon

        if (
            self._last_solution_u is not None
            and self._last_solution_u.shape == (2, N)
            and self._last_solution_x is not None
            and self._last_solution_x.shape == (3, N + 1)
        ):
            # Shift previous solution by one step.
            U0 = np.concatenate(
                [
                    self._last_solution_u[:, 1:],
                    self._last_solution_u[:, -1:],
                ],
                axis=1,
            )
            X0 = np.concatenate(
                [
                    self._last_solution_x[:, 1:],
                    self._last_solution_x[:, -1:],
                ],
                axis=1,
            )
            X0[:, 0] = robot_state
            return X0, U0

        # Straight-ish initial rollout toward goal.
        X0 = np.zeros((3, N + 1), dtype=np.float64)
        U0 = np.zeros((2, N), dtype=np.float64)
        X0[:, 0] = robot_state

        x, y, th = robot_state

        desired = math.atan2(goal[1] - y, goal[0] - x)
        heading_err = wrap_angle_numeric(desired - th)

        nominal_v = min(
            c.v_max * 0.5,
            max(c.v_min, np.linalg.norm(goal - robot_state[:2])),
        )
        nominal_w = np.clip(
            heading_err / max(N * c.dt, 1e-6),
            c.w_min,
            c.w_max,
        )

        # Respect first-step rate limits approximately.
        nominal_v = np.clip(
            nominal_v,
            prev_control[0] - c.a_v_max * c.dt,
            prev_control[0] + c.a_v_max * c.dt,
        )
        nominal_w = np.clip(
            nominal_w,
            prev_control[1] - c.a_w_max * c.dt,
            prev_control[1] + c.a_w_max * c.dt,
        )

        nominal_v = np.clip(nominal_v, c.v_min, c.v_max)
        nominal_w = np.clip(nominal_w, c.w_min, c.w_max)

        for k in range(N):
            U0[:, k] = [nominal_v, nominal_w]
            X0[0, k + 1] = X0[0, k] + c.dt * nominal_v * math.cos(X0[2, k])
            X0[1, k + 1] = X0[1, k] + c.dt * nominal_v * math.sin(X0[2, k])
            X0[2, k + 1] = X0[2, k] + c.dt * nominal_w

        return X0, U0

    # ---------------------------------------------------------------------
    # Solve
    # ---------------------------------------------------------------------

    def solve(
        self,
        robot_state: Sequence[float],
        goal: Sequence[float],
        anchors,
        velocity,
        confidence,
        prev_control: Sequence[float] = (0.0, 0.0),
    ) -> Tuple[float, float, Dict[str, np.ndarray]]:
        """
        Solve one NMPC cycle.

        Parameters
        ----------
        robot_state:
            [x, y, theta] in the SAME local frame as anchors and goal.
            For robot-centric planning this is usually [0,0,0].
        goal:
            [x_goal, y_goal] in the same frame.
        anchors:
            [P,2] motion-field anchor positions.
        velocity:
            [P,2] estimated token velocities.
        confidence:
            [P] or [P,1] motion confidence.
        prev_control:
            [v_previous, w_previous].

        Returns
        -------
        v_cmd, w_cmd, info
        """
        c = self.cfg

        state = _to_numpy(robot_state).reshape(3)
        goal_np = _to_numpy(goal).reshape(2)
        prev_u = _to_numpy(prev_control).reshape(2)

        pos, vel, conf, valid, kept = self._prepare_obstacles(
            anchors=anchors,
            velocity=velocity,
            confidence=confidence,
        )

        # Set parameters.
        self.opti.set_value(self.p_x0, state)
        self.opti.set_value(self.p_goal, goal_np)
        self.opti.set_value(self.p_u_prev, prev_u)
        self.opti.set_value(self.p_obs_pos, pos)
        self.opti.set_value(self.p_obs_vel, vel)
        self.opti.set_value(self.p_obs_conf, conf)
        self.opti.set_value(self.p_obs_valid, valid)

        # Warm start.
        X0, U0 = self._initial_guess(
            robot_state=state,
            goal=goal_np,
            prev_control=prev_u,
        )
        self.opti.set_initial(self.X, X0)
        self.opti.set_initial(self.U, U0)

        success = True
        status = "Solve_Succeeded"

        try:
            sol = self.opti.solve()
            X_sol = np.asarray(sol.value(self.X), dtype=np.float64)
            U_sol = np.asarray(sol.value(self.U), dtype=np.float64)
            objective = float(sol.value(self.opti.f))

            self._last_solution_x = X_sol
            self._last_solution_u = U_sol

        except RuntimeError:
            # IPOPT may fail due to poor initialization or numerics.
            # Fall back to debug values / initial guess.
            success = False
            try:
                status = str(self.opti.stats().get("return_status", "Failed"))
            except Exception:
                status = "Failed"

            try:
                X_sol = np.asarray(
                    self.opti.debug.value(self.X),
                    dtype=np.float64,
                )
                U_sol = np.asarray(
                    self.opti.debug.value(self.U),
                    dtype=np.float64,
                )
                objective = float(self.opti.debug.value(self.opti.f))
            except Exception:
                X_sol = X0
                U_sol = U0
                objective = float("nan")

        # Safety clipping.
        v_cmd = float(np.clip(U_sol[0, 0], c.v_min, c.v_max))
        w_cmd = float(np.clip(U_sol[1, 0], c.w_min, c.w_max))

        # If optimization failed badly, be conservative.
        if not np.isfinite(v_cmd) or not np.isfinite(w_cmd):
            v_cmd, w_cmd = 0.0, 0.0
            success = False
            status = "NonFiniteFallback"

        info = {
            "success": np.array(success),
            "status": np.array(status),
            "objective": np.array(objective),
            "predicted_states": X_sol,           # [3,N+1]
            "predicted_controls": U_sol,         # [2,N]
            "kept_token_indices": kept,
            "obstacle_positions": pos,           # [2,M]
            "obstacle_velocities": vel,          # [2,M]
            "obstacle_confidence": conf,          # [1,M]
            "obstacle_valid": valid,              # [1,M]
        }

        return v_cmd, w_cmd, info


# ---------------------------------------------------------------------------
# Optional helper: model output -> planner input
# ---------------------------------------------------------------------------

def planner_inputs_from_model_output(
    model_output: Dict,
    batch_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert DynamicLiDARNetwork output dict to numpy planner arrays.

    model_output keys:
        anchors     [B,P,2]
        velocity    [B,P,2]
        confidence  [B,P,1]
    """
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


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    cfg = NMPCConfig(
        horizon=10,
        dt=0.15,
        max_obstacles=12,
        ipopt_max_iter=60,
        print_level=0,
    )

    planner = DynamicNMPC(cfg)

    # Robot-centric frame.
    robot_state = np.array([0.0, 0.0, 0.0])
    goal = np.array([3.0, 0.0])

    # Synthetic token motion field:
    # one moving obstacle in front, several low-confidence tokens ignored.
    P = 36
    anchors = np.zeros((P, 2), dtype=np.float64)
    velocity = np.zeros((P, 2), dtype=np.float64)
    confidence = np.zeros((P, 1), dtype=np.float64)

    # Obstacle near x=1.4 moving upward.
    anchors[0] = [1.4, 0.05]
    velocity[0] = [0.0, 0.6]
    confidence[0, 0] = 0.95

    # Another less risky moving token.
    anchors[1] = [2.0, -0.7]
    velocity[1] = [0.0, 0.2]
    confidence[1, 0] = 0.75

    v_cmd, w_cmd, info = planner.solve(
        robot_state=robot_state,
        goal=goal,
        anchors=anchors,
        velocity=velocity,
        confidence=confidence,
        prev_control=[0.0, 0.0],
    )

    print("NMPC smoke test")
    print("----------------")
    print("success :", bool(info["success"]))
    print("status  :", info["status"].item())
    print("v_cmd   :", round(v_cmd, 4))
    print("w_cmd   :", round(w_cmd, 4))
    print("states  :", info["predicted_states"].shape)
    print("controls:", info["predicted_controls"].shape)


if __name__ == "__main__":
    _smoke_test()
