"""Support helpers for WBC torque-control actions."""

from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np
import torch

from holosoma.utils.rotations import quaternion_to_matrix


def tensor_to_string(t: torch.Tensor, precision: int = 6) -> str:
    """Convert a 1D torch.Tensor to string "[x, y, z]"."""
    t = t.detach().cpu().flatten()
    fmt = f"{{:.{precision}f}}"
    return "[" + ", ".join(fmt.format(x.item()) for x in t) + "]"


def string_to_tensor(s: str, device=None, dtype=torch.float32) -> torch.Tensor:
    """Convert string "[x, y, z]" to torch.Tensor."""
    values = s.strip()[1:-1].split(",")
    data = [float(v) for v in values if v.strip()]
    return torch.tensor(data, device=device, dtype=dtype)


def rot6d_to_matrix(x):
    """Convert a 6D rotation representation to a rotation matrix."""
    r1 = x[..., 0:3]
    r2 = x[..., 3:6]

    b1 = torch.nn.functional.normalize(r1, dim=-1)
    dot = (b1 * r2).sum(dim=-1, keepdim=True)
    b2 = torch.nn.functional.normalize(r2 - dot * b1, dim=-1)

    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1.0e-9:
        return np.eye(3)

    axis = axis_angle / angle
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def matrix_to_urdf_floating_xyz_angles(rotation: torch.Tensor) -> torch.Tensor:
    """Extract angles for a URDF chain R = Rx(rx) @ Ry(ry) @ Rz(rz)."""
    sy = torch.clamp(rotation[..., 0, 2], -1.0, 1.0)
    ry = torch.asin(sy)
    rx = torch.atan2(-rotation[..., 1, 2], rotation[..., 2, 2])
    rz = torch.atan2(-rotation[..., 0, 1], rotation[..., 0, 0])
    return torch.stack([rx, ry, rz], dim=-1)


def root_state_to_xyz_rpy(root_state: torch.Tensor) -> torch.Tensor:
    """Convert root [xyz, quat_xyzw, v, w] to [xyz, rx, ry, rz]."""
    root_rotation = quaternion_to_matrix(root_state[3:7].unsqueeze(0), w_last=True)[0]
    rx_ry_rz = matrix_to_urdf_floating_xyz_angles(root_rotation)
    return torch.cat([root_state[0:3], rx_ry_rz], dim=0)


def root_state_to_base_velocity(root_state: torch.Tensor) -> torch.Tensor:
    """Return floating-base velocity as [linear_velocity, angular_velocity]."""
    return root_state[7:13]


def root_states_to_xyz_rpy(root_states: torch.Tensor) -> torch.Tensor:
    """Convert batched root states to [xyz, rx, ry, rz]."""
    root_rotation = quaternion_to_matrix(root_states[:, 3:7], w_last=True)
    rx_ry_rz = matrix_to_urdf_floating_xyz_angles(root_rotation)
    return torch.cat([root_states[:, 0:3], rx_ry_rz], dim=-1)


def parse_actions(actions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert the 6-D policy task-space action into named slices."""
    if actions.shape[-1] != 6:
        raise ValueError(f"Expected exactly 6 task-space action values, got shape {tuple(actions.shape)}.")

    return {
        "com_vel": actions[..., :2],
        "torso_yaw_vel": actions[..., 2:3],
        "landing_foot_delta_pose": actions[..., 3:6],
    }


def skew(w):
    """Convert a 3D vector to a skew-symmetric matrix."""
    wx, wy, wz = w
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ]
    )


def exp_map(omega):
    theta = np.linalg.norm(omega)
    w_hat = skew(omega)
    w_hat_sq = w_hat @ w_hat

    if theta < 1e-6:
        a = 1 - theta**2 / 6 + theta**4 / 120
        b = 0.5 - theta**2 / 24 + theta**4 / 720
    else:
        a = np.sin(theta) / theta
        b = (1 - np.cos(theta)) / (theta**2)

    return np.eye(3) + a * w_hat + b * w_hat_sq


def apply_delta_rotation(rotation: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return rotation @ exp_map(omega)


def matrix_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to an axis-angle vector."""
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    cos_angle = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1.0e-9:
        return np.zeros(3, dtype=float)

    if abs(np.pi - angle) < 1.0e-5:
        axis = np.sqrt(np.maximum(np.diag(rotation) + 1.0, 0.0) * 0.5)
        axis[0] = math.copysign(axis[0], rotation[2, 1] - rotation[1, 2])
        axis[1] = math.copysign(axis[1], rotation[0, 2] - rotation[2, 0])
        axis[2] = math.copysign(axis[2], rotation[1, 0] - rotation[0, 1])
        norm = float(np.linalg.norm(axis))
        if norm < 1.0e-9:
            return np.zeros(3, dtype=float)
        return angle * axis / norm

    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    ) / (2.0 * np.sin(angle))
    return angle * axis


def yaw_to_matrix3d(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def yaw_from_matrix(rotation: np.ndarray) -> float:
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def quat_xyzw_from_yaw(yaw: float) -> np.ndarray:
    half_yaw = 0.5 * yaw
    return np.array([0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)], dtype=float)


def pose_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    pose[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return pose


def phase_from_sin_cos(sin_phi: float, cos_phi: float) -> float:
    """Return normalized phase in [0, 1) from sin(phi), cos(phi)."""
    phi = math.atan2(sin_phi, cos_phi)
    if phi < 0.0:
        phi += 2.0 * math.pi
    return phi / (2.0 * math.pi)


def determine_stance_from_phase_fractional(
    sin_phi: float,
    cos_phi: float,
    cycle_time: float = 1.0,
    left_stance_frac: float = 0.40,
    dual_frac: float = 0.10,
    right_stance_frac: float = 0.40,
) -> Tuple[str, float]:
    """Determine current stance and remaining time in the current active stance."""
    total = left_stance_frac + dual_frac + right_stance_frac + dual_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Fractions must sum to 1. Got {total}")

    phase = phase_from_sin_cos(sin_phi, cos_phi)
    left_end = left_stance_frac
    dual_1_end = left_end + dual_frac
    right_end = dual_1_end + right_stance_frac

    if phase < left_end:
        stance = "left"
        interval_end = left_end
    elif phase < dual_1_end:
        stance = "dual"
        interval_end = dual_1_end
    elif phase < right_end:
        stance = "right"
        interval_end = right_end
    else:
        stance = "dual"
        interval_end = 1.0

    return stance, (interval_end - phase) * cycle_time


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def yaw_to_rot2d(yaw: float) -> np.ndarray:
    """2D rotation matrix from yaw angle."""
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def nominal_landing_foot_pose(
    stance_pos: np.ndarray,
    stance_yaw: float,
    desired_velocity: np.ndarray,
    swing_is_left: bool,
    step_time: float = 0.45,
    step_width: float = 0.22,
    step_length_gain: float = 0.6,
    lateral_velocity_gain: float = 0.4,
    min_step_x: float = -0.10,
    max_step_x: float = 0.25,
    min_step_width: float = 0.12,
    max_step_width: float = 0.35,
    max_yaw_offset: float = np.deg2rad(15.0),
) -> tuple[np.ndarray, float]:
    """Suggest a nominal landing foot pose relative to a stance foot pose."""
    stance_pos = np.asarray(stance_pos, dtype=float).reshape(3)
    desired_velocity = np.asarray(desired_velocity, dtype=float).reshape(-1)

    if desired_velocity.size not in (2, 3):
        raise ValueError("desired_velocity must have shape (2,) or (3,): [vx, vy] or [vx, vy, yaw_rate].")

    vx_des = desired_velocity[0]
    vy_des = desired_velocity[1]
    yaw_rate_des = desired_velocity[2] if desired_velocity.size == 3 else 0.0
    side = 1.0 if swing_is_left else -1.0

    step_x = np.clip(step_length_gain * step_time * vx_des, min_step_x, max_step_x)
    step_y = side * step_width + lateral_velocity_gain * step_time * vy_des
    if swing_is_left:
        step_y = np.clip(step_y, min_step_width, max_step_width)
    else:
        step_y = np.clip(step_y, -max_step_width, -min_step_width)

    landing_pos = stance_pos.copy()
    landing_pos[:2] += yaw_to_rot2d(stance_yaw) @ np.array([step_x, step_y])
    landing_pos[2] = stance_pos[2]

    yaw_offset = np.clip(0.5 * yaw_rate_des * step_time, -max_yaw_offset, max_yaw_offset)
    return landing_pos, wrap_to_pi(stance_yaw + yaw_offset)


def add_landing_residual(
    landing_pos: np.ndarray,
    landing_yaw: float,
    stance_pos: np.ndarray,
    stance_yaw: float,
    residual_action: np.ndarray,
    swing_is_left: bool,
    residual_x_scale: float = 0.06,
    residual_y_scale: float = 0.04,
    residual_yaw_scale: float = np.deg2rad(8.0),
    min_step_x: float = -0.10,
    max_step_x: float = 0.25,
    min_step_width: float = 0.12,
    max_step_width: float = 0.35,
    max_yaw_offset_from_stance: float = np.deg2rad(20.0),
) -> tuple[np.ndarray, float]:
    """Add bounded RL residuals to nominal landing position/yaw."""
    landing_pos = np.asarray(landing_pos, dtype=float).reshape(3)
    stance_pos = np.asarray(stance_pos, dtype=float).reshape(3)
    residual_action = np.asarray(residual_action, dtype=float).reshape(3)

    rotation = yaw_to_rot2d(stance_yaw)
    residual_local = np.array(
        [
            residual_x_scale * np.clip(residual_action[0], -1.0, 1.0),
            residual_y_scale * np.clip(residual_action[1], -1.0, 1.0),
        ]
    )

    corrected_pos = landing_pos.copy()
    corrected_pos[:2] += rotation @ residual_local
    rel_local = rotation.T @ (corrected_pos[:2] - stance_pos[:2])
    rel_local[0] = np.clip(rel_local[0], min_step_x, max_step_x)
    if swing_is_left:
        rel_local[1] = np.clip(rel_local[1], min_step_width, max_step_width)
    else:
        rel_local[1] = np.clip(rel_local[1], -max_step_width, -min_step_width)

    corrected_pos[:2] = stance_pos[:2] + rotation @ rel_local
    corrected_pos[2] = stance_pos[2]

    yaw_residual = residual_yaw_scale * np.clip(residual_action[2], -1.0, 1.0)
    corrected_yaw = wrap_to_pi(landing_yaw + yaw_residual)
    yaw_error = np.clip(
        wrap_to_pi(corrected_yaw - stance_yaw),
        -max_yaw_offset_from_stance,
        max_yaw_offset_from_stance,
    )
    return corrected_pos, wrap_to_pi(stance_yaw + yaw_error)


class BatchedWbcFacade:
    """Small compatibility shim around humanoid_wbc.BatchedWbcController."""

    def __init__(self, controller: Any, planner: Any, dof: int, debug_engine: Any | None = None):
        self._controller = controller
        self._foot_planner = planner
        self._dof = dof
        self._debug_engine = debug_engine

    def dof(self) -> int:
        return self._dof

    def setTotalTransitionTime(self, seconds: float) -> None:
        self._controller.setTotalTransitionTime(seconds)
        if self._debug_engine is not None:
            self._debug_engine.setTotalTransitionTime(seconds)

    def reset_state(
        self,
        state: Any | None = None,
        transition_start_time: float = 0.0,
        env_id: int | None = None,
    ) -> None:
        if state is None:
            self._controller.reset_state()
            return
        if env_id is None:
            self._controller.reset_state(state, transition_start_time)
        else:
            self._controller.reset_state(state, transition_start_time, int(env_id))

    def compute_torques_batch(self, *args: Any, **kwargs: Any) -> Any:
        return self._controller.compute_torques_batch(*args, **kwargs)

    def update_robot(self, q: Any, dq: Any) -> None:
        self._controller.update_robot(q, dq)
        if self._debug_engine is not None and len(q) > 0:
            self._debug_engine.updateRobot(q[0], dq[0])

    def reInitializeAllTasks(self) -> None:
        self._controller.reInitializeAllTasks()
        if self._debug_engine is not None:
            self._debug_engine.reInitializeAllTasks()

    def get_states(self) -> Any:
        return self._controller.get_states()

    def get_transition_start_times(self) -> Any:
        return self._controller.get_transition_start_times()

    def __getattr__(self, name: str) -> Any:
        if self._debug_engine is not None and hasattr(self._debug_engine, name):
            return getattr(self._debug_engine, name)
        return getattr(self._controller, name)


class BatchedSwingFootPlanner:
    """Python-side batched wrapper around the pybind SwingFoot primitive."""

    def __init__(
        self,
        wbc_module: Any,
        num_envs: int,
        *,
        dt: float,
        takeoff_clearance: float = 0.05,
        landing_clearance: float = 0.05,
    ):
        if not hasattr(wbc_module, "SwingFoot") or not hasattr(wbc_module, "Contact"):
            raise RuntimeError(
                "humanoid_wbc.SwingFoot and humanoid_wbc.Contact are required for swing-foot replanning. "
                "Rebuild humanoid-control so the Python extension includes the planner bindings."
            )

        self._SwingFoot = wbc_module.SwingFoot
        self._Contact = wbc_module.Contact
        self._planners = [self._SwingFoot() for _ in range(num_envs)]
        self._dt = float(dt)
        self._takeoff_clearance = float(takeoff_clearance)
        self._landing_clearance = float(landing_clearance)
        self._active_sides: list[str | None] = [None for _ in range(num_envs)]

    def reset(self, env_ids: list[int] | range) -> None:
        for env_idx in env_ids:
            self._planners[int(env_idx)].reset()
            self._active_sides[int(env_idx)] = None

    def desired_pose_batch(
        self,
        *,
        current_poses: dict[str, np.ndarray],
        landing_poses: dict[int, tuple[str, np.ndarray]],
        remaining_durations: np.ndarray,
    ) -> dict[int, tuple[str, np.ndarray]]:
        desired: dict[int, tuple[str, np.ndarray]] = {}
        for env_idx, (swing_side, landing_pose) in landing_poses.items():
            remaining_duration = max(float(remaining_durations[env_idx]), self._dt)
            current_pose = np.asarray(current_poses[swing_side][env_idx], dtype=float)
            planner = self._planners[env_idx]

            if self._active_sides[env_idx] != swing_side:
                start_contact = self._contact_from_pose(current_pose, takeoff=True)
                end_contact = self._contact_from_pose(landing_pose, takeoff=False)
                planner.setParams(
                    start_contact,
                    end_contact,
                    remaining_duration,
                    self._takeoff_clearance,
                    self._landing_clearance,
                )
                self._active_sides[env_idx] = swing_side
            else:
                planner.resolveTrajectoryFromCurrentPoseToPose(current_pose, landing_pose, remaining_duration)

            integrated = np.asarray(planner.integrate(min(self._dt, remaining_duration)), dtype=float).reshape(-1)
            if integrated.shape[0] < 7:
                raise RuntimeError(
                    f"SwingFoot.integrate returned {integrated.shape[0]} values; expected at least pose xyz+quat."
                )
            desired_pose = np.eye(4, dtype=float)
            desired_pose[:3, 3] = integrated[:3]
            desired_pose[:3, :3] = self._rotation_from_quat_xyzw(integrated[3:7])
            desired[env_idx] = (swing_side, desired_pose)

        return desired

    def _contact_from_pose(self, foot_pose: np.ndarray, *, takeoff: bool) -> Any:
        yaw = yaw_from_matrix(foot_pose[:3, :3])
        return self._Contact(
            np.ascontiguousarray(foot_pose[:3, 3], dtype=np.float64),
            np.ascontiguousarray(np.array([0.0, 0.0, 1.0], dtype=np.float64)),
            np.ascontiguousarray(quat_xyzw_from_yaw(yaw), dtype=np.float64),
            self._takeoff_clearance if takeoff else self._landing_clearance,
            self._landing_clearance,
        )

    @staticmethod
    def _rotation_from_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
        q = np.asarray(quat_xyzw, dtype=float).reshape(4)
        norm = float(np.linalg.norm(q))
        if norm < 1.0e-12:
            return np.eye(3, dtype=float)
        x, y, z, w = q / norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=float,
        )
