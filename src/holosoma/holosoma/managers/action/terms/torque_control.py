"""Action terms for joint-level torque control."""

from __future__ import annotations

import importlib
import importlib.machinery
import math
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, Tuple

import numpy as np
import torch

from holosoma.managers.action.base import ActionTermBase
from holosoma.utils.rotations import quat_apply, quaternion_to_matrix

if TYPE_CHECKING:
    from holosoma.config_types.action import ActionTermCfg
    from holosoma.managers.command.terms.locomotion import LocomotionGait

#################################################
# HELPER FUNCTIONS
#################################################

def tensor_to_string(t: torch.Tensor, precision: int = 6) -> str:
    """
    Convert a 1D torch.Tensor to string "[x, y, z]".
    """
    t = t.detach().cpu().flatten()
    fmt = f"{{:.{precision}f}}"
    return "[" + ", ".join(fmt.format(x.item()) for x in t) + "]"

def string_to_tensor(s: str, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Convert string "[x, y, z]" to torch.Tensor.
    """
    values = s.strip()[1:-1].split(",")
    data = [float(v) for v in values if v.strip()]
    return torch.tensor(data, device=device, dtype=dtype)

def rot6d_to_matrix(x):
    """
    6D rotation representation to rotation matrix
    
    :param x: 6D rotation vector 
    """
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
    """Convert root [xyz, quat_xyzw, v, w] to [xyz, rx, ry, rz] for the control URDF floating joints."""
    root_rotation = quaternion_to_matrix(root_state[3:7].unsqueeze(0), w_last=True)[0]
    rx_ry_rz = matrix_to_urdf_floating_xyz_angles(root_rotation)
    return torch.cat([root_state[0:3], rx_ry_rz], dim=0)


def root_state_to_base_velocity(root_state: torch.Tensor) -> torch.Tensor:
    """Return floating-base velocity as [linear_velocity, angular_velocity]."""
    return root_state[7:13]


def root_states_to_xyz_rpy(root_states: torch.Tensor) -> torch.Tensor:
    """Convert batched root states to [xyz, rx, ry, rz] for the control URDF floating joints."""
    root_rotation = quaternion_to_matrix(root_states[:, 3:7], w_last=True)
    rx_ry_rz = matrix_to_urdf_floating_xyz_angles(root_rotation)
    return torch.cat([root_states[:, 0:3], rx_ry_rz], dim=-1)

def parse_actions(actions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert the policy task-space action into named slices."""
    if actions.shape[-1] != 9:
        raise ValueError(f"Expected exactly 9 task-space action values, got shape {tuple(actions.shape)}.")

    action_dict = {
        "com_vel": actions[..., :3],  # (vx, vy, vz) residual com velocity
        "pelvis_ang_vel": actions[..., 3:6],  # (wx, wy, wz) residual pelvis angular velocity
        "landing_foot_delta_pose": actions[..., 6:9], # (x, y, theta) residual (relative to the stance foot pose)
    }

    return action_dict

def skew(w):
    """Convert a 3D vector to a skew-symmetric matrix."""
    wx, wy, wz = w
    return np.array([
        [0.0, -wz,  wy],
        [wz,  0.0, -wx],
        [-wy, wx,  0.0]
    ])

def exp_map(omega):
    theta = np.linalg.norm(omega)
    w_hat = skew(omega)
    w_hat_sq = w_hat @ w_hat

    if theta < 1e-6:
        # Use series expansion
        A = 1 - theta**2 / 6 + theta**4 / 120
        B = 0.5 - theta**2 / 24 + theta**4 / 720
    else:
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / (theta ** 2)

    R = np.eye(3) + A * w_hat + B * w_hat_sq
    return R

def apply_delta_rotation(R_current, omega):
    return R_current @ exp_map(omega)


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
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


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
    """
    Returns normalized phase in [0, 1) from sin(phi), cos(phi).
    """
    phi = math.atan2(sin_phi, cos_phi)  # [-pi, pi]

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
    """
    Determine current stance and remaining time in the current active stance.

    Assumed cycle:
        left_stance -> dual -> right_stance -> dual -> repeat

    Args:
        sin_phi: sin(phase_angle)
        cos_phi: cos(phase_angle)
        cycle_time: full gait cycle duration [s]
        left_stance_frac: fraction of cycle in left stance
        dual_frac: fraction of cycle in each dual support phase
        right_stance_frac: fraction of cycle in right stance

    Returns:
        stance:
            "left", "right", or "dual"

        remaining_stance_time:
            Remaining time in the current active stance interval [s].
    """
    total = left_stance_frac + dual_frac + right_stance_frac + dual_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Fractions must sum to 1. Got {total}")

    phase = phase_from_sin_cos(sin_phi, cos_phi)

    left_end = left_stance_frac
    dual_1_end = left_end + dual_frac
    right_end = dual_1_end + right_stance_frac
    cycle_end = 1.0

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
        interval_end = cycle_end

    remaining_stance_time = (interval_end - phase) * cycle_time

    return stance, remaining_stance_time


#################################################
# FOOT POSE PLACEMENT FUNCTIONS 
#################################################

def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def yaw_to_rot2d(yaw: float) -> np.ndarray:
    """2D rotation matrix from yaw angle."""
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s],
                     [s,  c]])


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
    """
    Suggest a nominal landing foot pose relative to a stance foot pose.

    Args:
        stance_pos:
            Stance foot position in world frame, shape (3,).
        stance_yaw:
            Stance foot yaw angle in world frame, radians.
        desired_velocity:
            Desired robot velocity in the stance-yaw frame or heading frame.
            Shape can be (2,) for [vx, vy] or (3,) for [vx, vy, yaw_rate].
        swing_is_left:
            True if the landing/swing foot is the left foot.
            False if it is the right foot.
        step_time:
            Expected single step duration in seconds.
        step_width:
            Nominal lateral distance between feet, meters.
        step_length_gain:
            Scales forward step length from desired forward velocity.
        lateral_velocity_gain:
            Scales lateral step adjustment from desired lateral velocity.
        min_step_x, max_step_x:
            Forward/backward clamp relative to stance foot, meters.
        min_step_width, max_step_width:
            Lateral clamp relative to stance foot, meters.
        max_yaw_offset:
            Clamp on landing yaw offset from stance_yaw, radians.

    Returns:
        landing_pos:
            Suggested landing foot position in world frame, shape (3,).
        landing_yaw:
            Suggested landing foot yaw in world frame, radians.
    """
    stance_pos = np.asarray(stance_pos, dtype=float).reshape(3)
    desired_velocity = np.asarray(desired_velocity, dtype=float).reshape(-1)

    if desired_velocity.size not in (2, 3):
        raise ValueError("desired_velocity must have shape (2,) or (3,): [vx, vy] or [vx, vy, yaw_rate].")

    vx_des = desired_velocity[0]
    vy_des = desired_velocity[1]
    yaw_rate_des = desired_velocity[2] if desired_velocity.size == 3 else 0.0

    side = 1.0 if swing_is_left else -1.0

    # Step vector expressed in the stance-yaw / heading frame.
    step_x = step_length_gain * step_time * vx_des
    step_y = side * step_width + lateral_velocity_gain * step_time * vy_des

    # Prevent over-aggressive or crossed-over steps.
    step_x = np.clip(step_x, min_step_x, max_step_x)

    if swing_is_left:
        step_y = np.clip(step_y, min_step_width, max_step_width)
    else:
        step_y = np.clip(step_y, -max_step_width, -min_step_width)

    # Rotate relative step into world frame.
    R = yaw_to_rot2d(stance_yaw)
    step_world_xy = R @ np.array([step_x, step_y])

    landing_pos = stance_pos.copy()
    landing_pos[:2] += step_world_xy
    landing_pos[2] = stance_pos[2]

    # Simple turning heuristic: yaw the landing foot partway into the turn.
    yaw_offset = 0.5 * yaw_rate_des * step_time
    yaw_offset = np.clip(yaw_offset, -max_yaw_offset, max_yaw_offset)

    landing_yaw = wrap_to_pi(stance_yaw + yaw_offset)

    return landing_pos, landing_yaw

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
    """
    Add bounded RL residuals to nominal landing position/yaw, then clamp again
    relative to the stance foot.

    Args:
        landing_pos:
            Nominal landing foot position in world frame, shape (3,).
        landing_yaw:
            Nominal landing foot yaw in world frame, radians.
        stance_pos:
            Stance foot position in world frame, shape (3,).
        stance_yaw:
            Stance foot yaw in world frame, radians.
        residual_action:
            Normalized residual action, shape (3,):
                [residual_x, residual_y, residual_yaw], each expected in [-1, 1].
        swing_is_left:
            True if the landing/swing foot is the left foot.
            False if it is the right foot.
        residual_x_scale:
            Max residual in local x direction, meters.
        residual_y_scale:
            Max residual in local y direction, meters.
        residual_yaw_scale:
            Max residual yaw, radians.
        min_step_x, max_step_x:
            Forward/backward clamp relative to stance foot, meters.
        min_step_width, max_step_width:
            Lateral clamp relative to stance foot, meters.
        max_yaw_offset_from_stance:
            Max final landing yaw relative to stance_yaw, radians.

    Returns:
        corrected_pos:
            Landing position after residual and clamping, shape (3,).
        corrected_yaw:
            Landing yaw after residual and clamping, radians.
    """
    landing_pos = np.asarray(landing_pos, dtype=float).reshape(3)
    stance_pos = np.asarray(stance_pos, dtype=float).reshape(3)
    residual_action = np.asarray(residual_action, dtype=float).reshape(3)

    R = yaw_to_rot2d(stance_yaw)

    # Position residual in stance-yaw frame.
    residual_local = np.array([
        residual_x_scale * np.clip(residual_action[0], -1.0, 1.0),
        residual_y_scale * np.clip(residual_action[1], -1.0, 1.0),
    ])

    corrected_pos = landing_pos.copy()
    corrected_pos[:2] += R @ residual_local

    # Clamp final landing target in stance-yaw frame.
    rel_local = R.T @ (corrected_pos[:2] - stance_pos[:2])

    rel_local[0] = np.clip(rel_local[0], min_step_x, max_step_x)

    if swing_is_left:
        rel_local[1] = np.clip(rel_local[1], min_step_width, max_step_width)
    else:
        rel_local[1] = np.clip(rel_local[1], -max_step_width, -min_step_width)

    corrected_pos[:2] = stance_pos[:2] + R @ rel_local
    corrected_pos[2] = landing_pos[2]

    # Yaw residual and clamp relative to stance yaw.
    yaw_residual = residual_yaw_scale * np.clip(residual_action[2], -1.0, 1.0)
    corrected_yaw = wrap_to_pi(landing_yaw + yaw_residual)

    yaw_error_from_stance = wrap_to_pi(corrected_yaw - stance_yaw)
    yaw_error_from_stance = np.clip(
        yaw_error_from_stance,
        -max_yaw_offset_from_stance,
        max_yaw_offset_from_stance,
    )

    corrected_yaw = wrap_to_pi(stance_yaw + yaw_error_from_stance)

    return corrected_pos, corrected_yaw


#################################################
# CLASS
#################################################

class _BatchedWbcFacade:
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

    def compute_torques_batch(self, q: Any, dq: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._controller.compute_torques_batch(q, dq, *args, **kwargs)
        except TypeError as exc:
            if "compute_torques_batch()" not in str(exc):
                raise
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

    def get_pose(self, frame_name: str, env_idx: int = 0) -> np.ndarray:
        get_pose = getattr(self._controller, "getPose", None)
        if not callable(get_pose):
            raise RuntimeError("humanoid_wbc.BatchedWbcController must provide getPose(env_idx, frame_name).")
        return np.asarray(get_pose(int(env_idx), frame_name), dtype=float)

    def get_linear_velocity(self, frame_name: str, env_idx: int = 0) -> np.ndarray:
        get_linear_velocity = getattr(self._controller, "getLinearVelocity", None)
        if not callable(get_linear_velocity):
            raise RuntimeError(
                "humanoid_wbc.BatchedWbcController must provide getLinearVelocity(env_idx, frame_name)."
            )
        return np.asarray(get_linear_velocity(int(env_idx), frame_name), dtype=float)

    def get_angular_velocity(self, frame_name: str, env_idx: int = 0) -> np.ndarray:
        get_angular_velocity = getattr(self._controller, "getAngularVelocity", None)
        if not callable(get_angular_velocity):
            raise RuntimeError(
                "humanoid_wbc.BatchedWbcController must provide getAngularVelocity(env_idx, frame_name)."
            )
        return np.asarray(get_angular_velocity(int(env_idx), frame_name), dtype=float)

    def __getattr__(self, name: str) -> Any:
        if self._debug_engine is not None and hasattr(self._debug_engine, name):
            return getattr(self._debug_engine, name)
        return getattr(self._controller, name)


class _BatchedSwingFootPlanner:
    """Python-side batched wrapper around the pybind SwingFoot primitive."""

    def __init__(
        self,
        wbc_module: Any,
        num_envs: int,
        *,
        dt: float,
        takeoff_clearance: float = 0.05,
        landing_clearance: float = 0.05,
        midpoint_height: float | None = 0.2,
    ):
        if not hasattr(wbc_module, "SwingFoot") or not hasattr(wbc_module, "Contact"):
            raise RuntimeError(
                "humanoid_wbc.SwingFoot and humanoid_wbc.Contact are required for swing-foot replanning. "
                "Rebuild humanoid-control so the Python extension includes the planner bindings."
            )

        self._SwingFoot = wbc_module.SwingFoot
        self._Contact = wbc_module.Contact
        self._planners = [self._SwingFoot() for _ in range(num_envs)]
        if self._planners and not hasattr(self._planners[0], "resolveTrajectoryFromCurrentState"):
            raise RuntimeError(
                "humanoid_wbc.SwingFoot must provide resolveTrajectoryFromCurrentState("
                "current_foot_pose, current_linear_velocity, new_end_contact, remaining_duration). "
                "Rebuild humanoid-control so the Python extension includes the current-state replan binding."
            )
        self._dt = float(dt)
        self._takeoff_clearance = float(takeoff_clearance)
        self._landing_clearance = float(landing_clearance)
        self._midpoint_height = None if midpoint_height is None else float(midpoint_height)
        if self._midpoint_height is not None and self._planners and not hasattr(self._planners[0], "setMidpointHeight"):
            raise RuntimeError(
                "humanoid_wbc.SwingFoot must provide setMidpointHeight(height) for configurable swing-foot apex height. "
                "Rebuild humanoid-control so the Python extension includes the midpoint-height binding."
            )
        for planner in self._planners:
            if self._midpoint_height is not None:
                planner.setMidpointHeight(self._midpoint_height)
        self._active_sides: list[str | None] = [None for _ in range(num_envs)]

    def reset(self, env_ids: list[int] | range) -> None:
        for env_idx in env_ids:
            self._planners[int(env_idx)].reset()
            self._active_sides[int(env_idx)] = None

    def desired_pose_batch(
        self,
        *,
        current_poses: dict[str, np.ndarray],
        current_linear_velocities: dict[str, np.ndarray],
        landing_poses: dict[int, tuple[str, np.ndarray]],
        remaining_durations: np.ndarray,
        replan_mask: np.ndarray | None = None,
    ) -> dict[int, tuple[str, np.ndarray, np.ndarray]]:
        desired: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
        for env_idx, (swing_side, landing_pose) in landing_poses.items():
            remaining_duration = max(float(remaining_durations[env_idx]), self._dt)
            current_pose = np.asarray(current_poses[swing_side][env_idx], dtype=float)
            planner = self._planners[env_idx]
            should_replan = replan_mask is None or bool(replan_mask[env_idx])

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
            elif should_replan:
                current_linear_velocity = np.asarray(
                    current_linear_velocities[swing_side][env_idx],
                    dtype=np.float64,
                ).reshape(3)
                end_contact = self._contact_from_pose(landing_pose, takeoff=False)
                planner.resolveTrajectoryFromCurrentState(
                    current_pose,
                    np.ascontiguousarray(current_linear_velocity, dtype=np.float64),
                    end_contact,
                    remaining_duration,
                )

            integrated = np.asarray(planner.integrate(min(self._dt, remaining_duration)), dtype=float).reshape(-1)
            if integrated.shape[0] < 13:
                raise RuntimeError(
                    f"SwingFoot.integrate returned {integrated.shape[0]} values; expected pose xyz+quat and 6D velocity."
                )
            desired_pose = np.eye(4, dtype=float)
            desired_pose[:3, 3] = integrated[:3]
            desired_pose[:3, :3] = self._rotation_from_quat_xyzw(integrated[3:7])
            desired_velocity = integrated[7:13].astype(float, copy=False)
            desired[env_idx] = (swing_side, desired_pose, desired_velocity)

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


class JointTorqueActionTerm(ActionTermBase):
    """Action term for joint torque control with whole-body controller.

    This term processes raw actions as task space targets and computes
    torques using a WBC controller. Supports:
    - Action scaling
    - Action clipping
    - Action delay (if configured)
    - Torque randomization (if configured)
    - Torque clipping
    """

    def __init__(self, cfg: ActionTermCfg, env: Any):
        """Initialize joint position action term.

        Args:
            cfg: Configuration for this action term
            env: Environment instance (typically a ``BaseTask`` subclass)
        """
        super().__init__(cfg, env)

        # Policy actions may be task-space commands, while the controller still outputs
        # one torque per actuated DOF.
        self._action_dim = env.robot_config.actions_dim
        if self._action_dim != 9:
            raise ValueError(
                "JointTorqueActionTerm expects the 9-D policy action layout "
                "[com_vel(3), pelvis_ang_vel(3), landing_foot_delta_xyyaw(3)]. "
                f"Got robot_config.actions_dim={self._action_dim}."
            )
        self._torque_dim = env.num_dof

        # Initialize action buffers
        self._raw_actions = torch.zeros(env.num_envs, self._action_dim, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, self._action_dim, device=env.device)
        self._actions_after_delay = torch.zeros(env.num_envs, self._action_dim, device=env.device)
        self._wbc_replan_swing_trajectory = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self._wbc_nominal_landing_swing_side = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self._wbc_nominal_landing_pos = torch.zeros(env.num_envs, 3, device=env.device)
        self._wbc_nominal_landing_yaw = torch.zeros(env.num_envs, device=env.device)
        self._wbc_nominal_landing_stance_pos = torch.zeros(env.num_envs, 3, device=env.device)
        self._wbc_nominal_landing_stance_yaw = torch.zeros(env.num_envs, device=env.device)
        self._wbc_landing_target_swing_side = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self._wbc_landing_target_pos = torch.zeros(env.num_envs, 3, device=env.device)
        self._wbc_landing_target_yaw = torch.zeros(env.num_envs, device=env.device)
        self._wbc_landing_target_initial_duration = torch.zeros(env.num_envs, device=env.device)
        self._wbc_landing_target_frozen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._last_wbc_landing_target_update_fraction = torch.zeros(env.num_envs, device=env.device)
        self._last_wbc_landing_plan_inputs = torch.full((env.num_envs, 5), float("nan"), device=env.device)

        # Initialize torque buffer
        self.torques = torch.zeros(env.num_envs, self._torque_dim, device=env.device)

        # Cache previous DOF velocities for derivative control
        self._prev_dof_vel = torch.zeros(env.num_envs, env.num_dof, device=env.device)
        self._non_finite_torque_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        # Default actuator scaling (may be overridden by randomization terms)
        self._kp_scale = torch.ones(env.num_envs, self._torque_dim, device=env.device)
        self._kd_scale = torch.ones_like(self._kp_scale)
        self._rfi_lim_scale = torch.ones_like(self._kp_scale)
        self._rfi_lim: float = 0.0
        self._randomize_torque_rfi: bool = False

        # PD gains are per actuator; action scales are per policy action.
        self.p_gains = torch.zeros(self._torque_dim, dtype=torch.float, device=env.device)
        self.d_gains = torch.zeros_like(self.p_gains)
        self.i_gains = torch.zeros_like(self.p_gains)
        self.action_scales = torch.zeros(self._action_dim, dtype=torch.float, device=env.device)
        self.action_clip_values = torch.zeros(self._action_dim, dtype=torch.float, device=env.device)

        self._configure_pd_gains(env)
        self._configure_action_scales(env)
        self._configure_action_clip_values(env)

        # Expose references on the environment for backward compatibility
        env.p_gains = self.p_gains
        env.d_gains = self.d_gains
        env.i_gains = self.i_gains
        env.action_scales = self.action_scales
        env.action_clip_values = self.action_clip_values

        # Action delay queue will be initialized in setup() after randomization manager is ready
        self.action_queue: torch.Tensor | None = None

        self._foot_body_indices = self._resolve_foot_body_indices(env)
        self._torso_body_index = self._resolve_body_index_by_name(env, getattr(env.robot_config, "torso_name", None))
        self._last_wbc_q = torch.zeros(env.num_envs, env.num_dof + 6, device=env.device)
        self._last_wbc_root_state = torch.zeros(env.num_envs, 13, device=env.device)
        self._last_wbc_dof_pos = torch.zeros(env.num_envs, env.num_dof, device=env.device)
        self._last_wbc_dq = torch.zeros(env.num_envs, env.num_dof + 6, device=env.device)
        self._last_wbc_action_batch: np.ndarray | None = None
        self._last_wbc_right_contact_points: np.ndarray | None = None
        self._last_wbc_left_contact_points: np.ndarray | None = None
        self._last_wbc_right_contact_bases: np.ndarray | None = None
        self._last_wbc_left_contact_bases: np.ndarray | None = None
        self._last_wbc_right_grfs: np.ndarray | None = None
        self._last_wbc_left_grfs: np.ndarray | None = None
        self._last_wbc_torque_output: np.ndarray | None = None
        self._last_wbc_sin_phase = torch.zeros(env.num_envs, 2, device=env.device)
        self._last_wbc_cos_phase = torch.ones(env.num_envs, 2, device=env.device)
        self._last_wbc_desired_state = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._last_wbc_remaining_swing_duration = torch.zeros(env.num_envs, device=env.device)
        self._wbc_integrated_com_pos = torch.zeros(env.num_envs, 3, device=env.device)
        self._wbc_integrated_pelvis_pos = torch.zeros(env.num_envs, 3, device=env.device)
        self._wbc_integrated_pelvis_rot = torch.eye(3, device=env.device).repeat(env.num_envs, 1, 1)
        self._wbc_integrated_torso_rot = torch.eye(3, device=env.device).repeat(env.num_envs, 1, 1)
        self._wbc_bootstrap_enabled = bool(cfg.params.get("dual_stance_bootstrap_enabled", True))
        self._dual_contact_force_threshold = float(cfg.params.get("dual_stance_contact_force_threshold", 10.0))
        self._dual_contact_required_steps = int(cfg.params.get("dual_stance_contact_required_steps", 10))
        if self._dual_contact_required_steps < 1:
            raise ValueError("dual_stance_contact_required_steps must be at least 1.")
        self._startup_gait_timeout_s = float(cfg.params.get("startup_gait_timeout_s", 0.75))
        self._wbc_dt = float(cfg.params.get("wbc_control_dt", getattr(env.simulator, "sim_dt", env.dt)))
        self._startup_gait_timeout_steps = max(1, int(math.ceil(self._startup_gait_timeout_s / self._wbc_dt)))
        filter_stance_support_contacts = not bool(cfg.params.get("use_unfiltered_stance_support_contacts", False))
        self._filter_stance_support_contacts = bool(
            cfg.params.get("filter_stance_support_contacts", filter_stance_support_contacts)
        )
        self._visualize_contact_points = bool(
            cfg.params.get("visualize_contact_points", False)
            or cfg.params.get("visualize_wbc_contact_points", False)
            or os.environ.get("HOLOSOMA_VISUALIZE_CONTACT_POINTS", "0") == "1"
        )
        self._visualize_contact_frames = bool(
            cfg.params.get("visualize_contact_frames", False)
            or os.environ.get("HOLOSOMA_VISUALIZE_CONTACT_FRAMES", "0") == "1"
        )
        self._contact_point_radius = float(cfg.params.get("contact_point_radius", 0.018))
        self._visualize_action_targets = bool(
            cfg.params.get("visualize_action_targets", False)
            or cfg.params.get("visualize_wbc_action_targets", False)
            or os.environ.get("HOLOSOMA_VISUALIZE_ACTION_TARGETS", "0") == "1"
        )
        self._visualize_action_target_frames = bool(
            cfg.params.get("visualize_action_target_frames", True)
            and os.environ.get("HOLOSOMA_VISUALIZE_ACTION_TARGET_FRAMES", "1") != "0"
        )
        self._visualize_landing_foot_pose = bool(
            cfg.params.get("visualize_landing_foot_pose", False)
            or os.environ.get("HOLOSOMA_VISUALIZE_LANDING_FOOT_POSE", "0") == "1"
        )
        self._visualize_swing_foot_trajectory = bool(
            cfg.params.get("visualize_swing_foot_trajectory", False)
            or os.environ.get("HOLOSOMA_VISUALIZE_SWING_FOOT_TRAJECTORY", "0") == "1"
        )
        self._swing_trajectory_samples = int(cfg.params.get("swing_trajectory_samples", 12))
        self._landing_ground_plane_z = float(cfg.params.get("landing_ground_plane_z", 0.05))
        self._use_command_as_pelvis_velocity_action = bool(cfg.params.get("use_command_as_pelvis_velocity_action", False))
        self._use_command_as_landing_velocity = bool(cfg.params.get("use_command_as_landing_velocity", False))
        self._store_wbc_debug_snapshots = bool(
            cfg.params.get("store_wbc_debug_snapshots", False)
            or os.environ.get("HOLOSOMA_STORE_WBC_DEBUG_SNAPSHOTS", "0") == "1"
        )
        self._debug_wbc_finite_checks = bool(
            cfg.params.get("debug_wbc_finite_checks", False)
            or os.environ.get("HOLOSOMA_DEBUG_WBC_FINITE_CHECKS", "0") == "1"
        )
        self._debug_swing_foot = bool(
            cfg.params.get("debug_swing_foot", False)
            or os.environ.get("HOLOSOMA_DEBUG_SWING_FOOT", "0") == "1"
        )
        self._debug_swing_foot_interval = max(
            1,
            int(cfg.params.get("debug_swing_foot_interval", os.environ.get("HOLOSOMA_DEBUG_SWING_FOOT_INTERVAL", 1))),
        )
        self._debug_swing_foot_counter = 0
        self._last_wbc_landing_plan_changed = np.zeros(env.num_envs, dtype=bool)
        self._last_wbc_landing_plan_forced = np.zeros(env.num_envs, dtype=bool)
        self._landing_plan_position_replan_tolerance = float(cfg.params.get("landing_plan_position_replan_tolerance", 1.0e-4))
        self._landing_plan_yaw_replan_tolerance = float(cfg.params.get("landing_plan_yaw_replan_tolerance", 1.0e-3))
        self._landing_target_update_fraction = float(np.clip(cfg.params.get("landing_target_update_fraction", 0.5), 0.0, 1.0))
        self._action_target_radius = float(cfg.params.get("action_target_radius", 0.026))
        self._action_target_axis_scale = float(cfg.params.get("action_target_axis_scale", 0.16))
        self._assert_contact_visualization_pose = bool(cfg.params.get("assert_contact_visualization_pose", False))  # debug assert
        self._swing_foot_takeoff_clearance = float(cfg.params.get("swing_foot_takeoff_clearance", 0.05))
        self._swing_foot_landing_clearance = float(cfg.params.get("swing_foot_landing_clearance", 0.05))
        self._swing_foot_midpoint_height = cfg.params.get("swing_foot_midpoint_height", 0.2)
        if self._swing_foot_midpoint_height is not None:
            self._swing_foot_midpoint_height = float(self._swing_foot_midpoint_height)
        self._wbc_transition_time = float(cfg.params.get("wbc_transition_time", 0.15))
        self._wbc_bootstrap_done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._dual_contact_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._startup_gait_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._startup_desired_state = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._wbc_phase_shift_frac = torch.zeros(env.num_envs, device=env.device)
        self._wbc_bootstrap_hold_target_dof_pos = torch.zeros(env.num_envs, self._torque_dim, device=env.device)
        self._last_wbc_bootstrap_hold_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._last_wbc_dual_contact_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        resolved_extension_dir = self._resolve_wbc_extension_dir(cfg.wbc_extension_dir)
        resolved_params = self._resolve_wbc_params(env, cfg.params, resolved_extension_dir)
        self._wbc_module = self._import_wbc_module(resolved_extension_dir)
        self.State = self._wbc_module.State
        self.Phase = self._wbc_module.Phase
        self._startup_desired_state.fill_(int(self.State.DUAL_STANCE))
        self._batched_foot_planner = self._create_batched_foot_planner(self._wbc_module, env.num_envs)
        self._batched_wbc = self._create_batched_wbc_controller(
            resolved_extension_dir, resolved_params, env.num_envs, self._wbc_module
        )
        self._wbc_debug_engine = self._create_wbc_engine(resolved_extension_dir, resolved_params, self._wbc_module)
        self.wbc = _BatchedWbcFacade(self._batched_wbc, self._batched_foot_planner, int(self._wbc_debug_engine.dof()), self._wbc_debug_engine)
        self.curr_state = [self.State.DUAL_STANCE for _ in range(env.num_envs)]
        self._prev_wbc_state: Any | None = None
        self.transition_start_time = [0.0 for _ in range(env.num_envs)]
        self._pending_wbc_reinitialize = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.wbc.setTotalTransitionTime(self._wbc_transition_time)

    def setup(self) -> None:
        """Setup action term after all managers are initialized.

        Initialize action delay queue if control delay randomization is enabled.
        This must be called after the randomization manager is set up.
        """
        super().setup()

        # Initialize action delay queue if randomization is enabled
        if getattr(self.env, "_randomize_ctrl_delay", False):
            max_delay = self.env._ctrl_delay_step_range[1]
            self.action_queue = torch.zeros(self.env.num_envs, max_delay + 1, self._action_dim, device=self.env.device)

        # IsaacGym creates randomization buffers before the action manager exists.
        # Once we reach setup(), try attaching any pre-created actuator scales.
        self._attach_actuator_randomizer_scales()

        enabled, rfi_lim = self.env._pending_torque_rfi
        self.configure_torque_rfi(enabled=enabled, rfi_lim=rfi_lim)
        self.env._pending_torque_rfi = (False, 0.0)

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return self._action_dim

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process raw actions: clip and apply delay if configured.

        Args:
            actions: Raw action tensor [num_envs, action_dim]
        """
        self._assert_finite_torch("raw_actions", actions)

        # Store raw actions
        assert self._raw_actions is not None
        self._raw_actions[:] = actions

        # Clip actions
        if self.env.robot_config.control.clip_actions:
            assert self._processed_actions is not None
            clip_values = self.action_clip_values.to(device=actions.device, dtype=actions.dtype)
            self._processed_actions[:] = torch.clamp(actions, min=-clip_values, max=clip_values)
            # Log clipping fraction
            self.env.log_dict["action_clip_frac"] = (
                actions.abs() >= clip_values
            ).sum() / self._processed_actions.numel()
        else:
            assert self._processed_actions is not None
            self._processed_actions[:] = actions
            self.env.log_dict["action_clip_frac"] = torch.tensor(0.0)

        # Apply action delay if configured
        if getattr(self.env, "_randomize_ctrl_delay", False):
            self._apply_action_delay()
        else:
            assert self._processed_actions is not None
            self._actions_after_delay[:] = self._processed_actions

        self._assert_finite_torch("actions_after_delay", self._actions_after_delay)

    def _apply_action_delay(self) -> None:
        """Apply action delay based on domain randomization settings."""
        assert self.action_queue is not None, "action_queue must be initialized in setup()"
        assert self._processed_actions is not None

        # Update action queue
        self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
        self.action_queue[:, 0] = self._processed_actions.clone()

        # Apply uniform delay
        self._actions_after_delay[:] = self.action_queue[
            torch.arange(self.env.num_envs), self.env.action_delay_idx
        ].clone()

    def apply_actions(self) -> None:
        """Apply processed actions by computing and applying torques."""
        # Compute torques using PD controller
        self.torques[:] = self._sanitize_non_finite_torques(self._compute_torques(self._actions_after_delay))
        # Apply torques to simulator
        self.env.simulator.apply_torques_at_dof(self.torques)
        # Cache velocities for next derivative computation
        self._prev_dof_vel.copy_(self.env.simulator.dof_vel)

    @property
    def non_finite_torque_mask(self) -> torch.Tensor:
        return self._non_finite_torque_mask

    @property
    def wbc_bootstrap_hold_mask(self) -> torch.Tensor:
        return self._last_wbc_bootstrap_hold_mask

    def _sanitize_non_finite_torques(self, torques: torch.Tensor) -> torch.Tensor:
        finite_rows = torch.isfinite(torques).flatten(start_dim=1).all(dim=1)
        bad_rows = ~finite_rows
        self._non_finite_torque_mask |= bad_rows
        if bool(finite_rows.all()):
            return torques

        env_idx = int(bad_rows.nonzero(as_tuple=False).flatten()[0].detach().cpu().item())
        finite = torch.isfinite(torques)
        bad_idx = (~finite).nonzero(as_tuple=False)[0].detach().cpu().tolist()
        bad_value = torques[tuple(bad_idx)].detach().cpu().item()
        finite_values = torques[finite].detach()
        if finite_values.numel() == 0:
            summary = "no finite torque values"
        else:
            summary = (
                f"finite_min={finite_values.min().cpu().item():.6g}, "
                f"finite_max={finite_values.max().cpu().item():.6g}"
            )
        raise RuntimeError(
            f"Computed torques contain non-finite value at index {bad_idx}: {bad_value}; "
            f"shape={tuple(torques.shape)}, bad_env={env_idx}, {summary}; "
            f"{self._non_finite_torque_context(env_idx, torques)}"
        )

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute torques from the in-process whole-body controller.

        Args:
            actions: Action tensor [num_envs, action_dim]

        Returns:
            Torque tensor [num_envs, num_dof]
        """
        self._assert_finite_torch("wbc_actions", actions)

        num_envs = actions.shape[0]
        if num_envs != self.env.num_envs:
            raise RuntimeError(
                "BatchedWbcController requires one action row per environment. "
                f"Got actions.shape[0]={num_envs}, env.num_envs={self.env.num_envs}."
            )

        # WBC task-space actions are pre-scaled per command dimension in Python.
        # Keep the binding scale at 1.0 to avoid applying a second uniform multiplier.
        action_scale = 1.0

        root_states = self._as_torch_tensor(
            self.env.simulator.robot_root_states, device=actions.device, dtype=torch.float32, label="root_states"
        )
        dof_pos = self._as_torch_tensor(
            self.env.simulator.dof_pos, device=actions.device, dtype=torch.float32, label="dof_pos"
        )
        dof_vel = self._as_torch_tensor(
            self.env.simulator.dof_vel, device=actions.device, dtype=torch.float32, label="dof_vel"
        )
        q_tensor = torch.cat([root_states_to_xyz_rpy(root_states), dof_pos], dim=1)
        dq_tensor = torch.cat([root_states[:, 7:13], dof_vel], dim=1)
        self._assert_finite_torch_debug("wbc_root_states", root_states)
        self._assert_finite_torch_debug("wbc_dof_pos", dof_pos)
        self._assert_finite_torch_debug("wbc_dof_vel", dof_vel)
        self._assert_finite_torch_debug("wbc_q", q_tensor)
        self._assert_finite_torch_debug("wbc_dq", dq_tensor)

        self._last_wbc_q[:num_envs] = q_tensor
        self._last_wbc_dq[:num_envs] = dq_tensor
        self._last_wbc_root_state[:num_envs] = root_states
        self._last_wbc_dof_pos[:num_envs] = dof_pos

        right_contact_points, left_contact_points, right_contact_bases, left_contact_bases = (
            self._stance_support_contact_arrays()
        )
        # right_grfs = self._batched_foot_ground_reaction_wrenches("right")
        # left_grfs = self._batched_foot_ground_reaction_wrenches("left")
        right_grfs = - self._batched_local_foot_ground_reaction_wrenches("right") # local grf
        left_grfs = - self._batched_local_foot_ground_reaction_wrenches("left")
        self._last_wbc_right_grfs = right_grfs.copy()
        self._last_wbc_left_grfs = left_grfs.copy()
        right_contact_in_contact = self._batched_foot_contact_in_contact("right", right_grfs)
        left_contact_in_contact = self._batched_foot_contact_in_contact("left", left_grfs)

        # # debug (verified to be correct sign)
        # print("right grf: ", right_grfs)
        # print("left grf: ", left_grfs)

        q = self._as_numpy_2d(q_tensor, "q")
        dq = self._as_numpy_2d(dq_tensor, "dq")
        self.wbc.update_robot(q, dq)
        sin_phase, cos_phase = self._phase_features_for_wbc(actions)
        desired_states, swing_sides, remaining_swing_durations = self._desired_states_from_phase(sin_phase, cos_phase)
        desired_states, swing_sides, remaining_swing_durations = self._maybe_force_wbc_desired_state(
            desired_states,
            swing_sides,
            remaining_swing_durations,
        )

        self._last_wbc_sin_phase[:num_envs] = sin_phase
        self._last_wbc_cos_phase[:num_envs] = cos_phase
        self._last_wbc_desired_state[:num_envs] = torch.as_tensor(
            desired_states, device=self.env.device, dtype=torch.long
        )
        self._last_wbc_remaining_swing_duration[:num_envs] = torch.as_tensor(
            remaining_swing_durations, device=self.env.device, dtype=torch.float32
        )
        self._assert_finite_numpy_debug("wbc_right_contact_points", right_contact_points)
        self._assert_finite_numpy_debug("wbc_left_contact_points", left_contact_points)
        self._assert_finite_numpy_debug("wbc_right_contact_bases", right_contact_bases)
        self._assert_finite_numpy_debug("wbc_left_contact_bases", left_contact_bases)
        self._assert_finite_numpy_debug("wbc_right_grfs", right_grfs)
        self._assert_finite_numpy_debug("wbc_left_grfs", left_grfs)
        self._maybe_clear_wbc_visualization_lines()
        self._maybe_draw_contact_points(
            right_contact_points,
            right_contact_bases,
            left_contact_points,
            left_contact_bases,
        )
        bootstrap_hold_mask: torch.Tensor | None = None
        pending_integrated_target_reset = self._pending_wbc_reinitialize.clone()
        if self._wbc_bootstrap_enabled:
            startup_started = self._update_wbc_bootstrap_state(right_grfs, left_grfs)
            if bool(startup_started.any()):
                desired_states, swing_sides, remaining_swing_durations = self._desired_states_from_phase(
                    sin_phase, cos_phase
                )
                desired_states, swing_sides, remaining_swing_durations = self._maybe_force_wbc_desired_state(
                    desired_states,
                    swing_sides,
                    remaining_swing_durations,
                )
                self._last_wbc_desired_state[:num_envs] = torch.as_tensor(
                    desired_states, device=self.env.device, dtype=torch.long
                )
                self._last_wbc_remaining_swing_duration[:num_envs] = torch.as_tensor(
                    remaining_swing_durations, device=self.env.device, dtype=torch.float32
                )
            bootstrap_hold_mask = ~self._wbc_bootstrap_done
            if bool(bootstrap_hold_mask.all()):
                hold_torques = self._compute_bootstrap_hold_torques(dof_pos, dof_vel)
                if self._store_wbc_debug_snapshots:
                    self._last_wbc_torque_output = hold_torques.detach().cpu().numpy().astype(np.float64, copy=True)
                return hold_torques
            if bool(self._pending_wbc_reinitialize.any()):
                self._reinitialize_wbc_startup_state(q, dq, update_robot=False)
        else:
            self._last_wbc_bootstrap_hold_mask.zero_()
            self._last_wbc_dual_contact_mask.fill_(True)

        if bool(pending_integrated_target_reset.any()):
            self._reset_wbc_integrated_targets(pending_integrated_target_reset)
        if bootstrap_hold_mask is not None and bool(bootstrap_hold_mask.any()):
            self._reset_wbc_integrated_targets(bootstrap_hold_mask)

        action_batch = self._actions_for_batched_wbc(
            actions,
            desired_states=desired_states,
            swing_sides=swing_sides,
            remaining_swing_durations=remaining_swing_durations,
        )

        self._maybe_store_wbc_debug_snapshots(
            action_batch,
            right_contact_points,
            left_contact_points,
            right_contact_bases,
            left_contact_bases,
            right_grfs,
            left_grfs,
        )
        self._assert_finite_wbc_action_batch(action_batch)
        self._assert_finite_numpy_debug("wbc_action_batch", action_batch)
        self._maybe_draw_action_targets(root_states, action_batch)

        if bool(self._pending_wbc_reinitialize.any()):
            self._reinitialize_wbc_startup_state(q, dq, update_robot=False)
        if self._needs_wbc_debug_engine_update():
            self._wbc_debug_engine.updateRobot(q[0], dq[0])
        try:
            desired_states_batch = np.ascontiguousarray(desired_states, dtype=np.int32)
            torque_wbc = self.wbc.compute_torques_batch(
                q,
                dq,
                desired_states_batch,
                action_batch,
                right_contact_points,
                right_contact_bases,
                left_contact_points,
                left_contact_bases,
                right_grfs,
                left_grfs,
                right_contact_in_contact,
                left_contact_in_contact,
                float(self.env.simulator.time()),
                action_scale,
            )
            self._sync_wbc_state_cache_from_controller()
        except RuntimeError as exc:
            env_idx = self._env_idx_from_wbc_exception(exc)
            if env_idx is not None:
                context = self._wbc_input_context_summary(
                    env_idx,
                    q,
                    dq,
                    action_batch,
                    right_contact_points,
                    left_contact_points,
                    right_contact_bases,
                    left_contact_bases,
                    right_grfs,
                    left_grfs,
                    right_contact_in_contact,
                    left_contact_in_contact,
                )
                raise RuntimeError(f"{exc}\n{context}") from exc
            raise
        torque_np = self._actuated_torques_from_batched_wbc_output(torque_wbc)
        torques = torch.as_tensor(
            torque_np,
            device=actions.device,
            dtype=self.torques.dtype,
        )
        if bootstrap_hold_mask is not None and bool(bootstrap_hold_mask.any()):
            hold_torques = self._compute_bootstrap_hold_torques(dof_pos, dof_vel)
            torques = torch.where(bootstrap_hold_mask.unsqueeze(1), hold_torques, torques)
            torque_np = torques.detach().cpu().numpy().astype(np.float64, copy=True)

        if self._store_wbc_debug_snapshots:
            self._last_wbc_torque_output = torque_np.copy()

        # Scale actions
        # actions_scaled = actions * self.action_scales

        # # Compute torques based on control type
        # control_type = self.env.robot_config.control.control_type

        # if control_type == "P":
        #     # Position control
        #     torques = (
        #         self._kp_scale * self.p_gains * (actions_scaled + self.env.default_dof_pos - self.env.simulator.dof_pos)
        #         - self._kd_scale * self.d_gains * self.env.simulator.dof_vel
        #     )
        # elif control_type == "V":
        #     # Velocity control
        #     torques = (
        #         self._kp_scale * self.p_gains * (actions_scaled - self.env.simulator.dof_vel)
        #         - self._kd_scale * self.d_gains * (self.env.simulator.dof_vel - self._prev_dof_vel) / self.env.sim_dt
        #     )
        # elif control_type == "T":
        #     # Torque control
        #     torques = actions_scaled
        # else:
        #     raise ValueError(f"Unknown controller type: {control_type}")

        # # Apply torque randomization if configured
        # if self._randomize_torque_rfi:
        #     torques = (
        #         torques
        #         + (torch.rand_like(torques) * 2.0 - 1.0) * self._rfi_lim * self._rfi_lim_scale * self.env.torque_limits
        #     )

        # # Clip torques if configured
        # if self.env.robot_config.control.clip_torques:
        #     torques = torch.clip(torques, -self.env.torque_limits, self.env.torque_limits)

        return torques

    def _assert_finite_torch(self, label: str, value: torch.Tensor) -> None:
        finite = torch.isfinite(value)
        if bool(finite.all()):
            return

        bad_idx = (~finite).nonzero(as_tuple=False)[0].detach().cpu().tolist()
        bad_value = value[tuple(bad_idx)].detach().cpu().item()
        finite_values = value[finite].detach()
        if finite_values.numel() == 0:
            summary = "no finite values"
        else:
            summary = (
                f"finite_min={finite_values.min().cpu().item():.6g}, "
                f"finite_max={finite_values.max().cpu().item():.6g}"
            )
        torque_summary = self._last_torque_debug_summary(bad_idx[0] if bad_idx else None)
        raise RuntimeError(
            f"{label} has non-finite value at index {bad_idx}: {bad_value}; "
            f"shape={tuple(value.shape)}, {summary}{torque_summary}"
        )

    def _assert_finite_numpy(self, label: str, value: np.ndarray) -> None:
        finite = np.isfinite(value)
        if bool(finite.all()):
            return

        bad_idx = np.argwhere(~finite)[0].tolist()
        bad_value = value[tuple(bad_idx)]
        finite_values = value[finite]
        if finite_values.size == 0:
            summary = "no finite values"
        else:
            summary = f"finite_min={finite_values.min():.6g}, finite_max={finite_values.max():.6g}"
        raise RuntimeError(
            f"{label} has non-finite value at index {bad_idx}: {bad_value}; "
            f"shape={value.shape}, {summary}"
        )

    def _assert_finite_torch_debug(self, label: str, value: torch.Tensor) -> None:
        if self._debug_wbc_finite_checks:
            self._assert_finite_torch(label, value)

    def _assert_finite_numpy_debug(self, label: str, value: np.ndarray) -> None:
        if self._debug_wbc_finite_checks:
            self._assert_finite_numpy(label, value)

    def _needs_wbc_debug_engine_update(self) -> bool:
        return (
            self._visualize_contact_points
            or self._visualize_action_targets
            or self._visualize_landing_foot_pose
            or self._visualize_swing_foot_trajectory
            or self._assert_contact_visualization_pose
            or self._store_wbc_debug_snapshots
        )

    def _maybe_clear_wbc_visualization_lines(self) -> None:
        if not (
            self._visualize_contact_points
            or self._visualize_action_targets
            or self._visualize_landing_foot_pose
            or self._visualize_swing_foot_trajectory
        ):
            return

        simulator = self.env.simulator
        if hasattr(simulator, "clear_lines"):
            simulator.clear_lines()

    def _maybe_store_wbc_debug_snapshots(
        self,
        action_batch: np.ndarray,
        right_contact_points: np.ndarray,
        left_contact_points: np.ndarray,
        right_contact_bases: np.ndarray,
        left_contact_bases: np.ndarray,
        right_grfs: np.ndarray,
        left_grfs: np.ndarray,
    ) -> None:
        self._last_wbc_right_grfs = right_grfs.copy()
        self._last_wbc_left_grfs = left_grfs.copy()

        if not self._store_wbc_debug_snapshots:
            return

        self._last_wbc_action_batch = action_batch.copy()
        self._last_wbc_right_contact_points = right_contact_points.copy()
        self._last_wbc_left_contact_points = left_contact_points.copy()
        self._last_wbc_right_contact_bases = right_contact_bases.copy()
        self._last_wbc_left_contact_bases = left_contact_bases.copy()

    def _non_finite_torque_context(self, env_idx: int, torques: torch.Tensor) -> str:
        parts = [
            self._batch_state_summary(env_idx),
            self._bootstrap_state_summary(env_idx),
            self._tensor_row_values("right_grf", self._last_wbc_right_grfs, env_idx),
            self._tensor_row_values("left_grf", self._last_wbc_left_grfs, env_idx),
            self._tensor_row_values("computed_torques", torques, env_idx),
            self._tensor_row_summary("last_wbc_action_batch", self._last_wbc_action_batch, env_idx),
            self._tensor_row_summary("last_wbc_torque_output", self._last_wbc_torque_output, env_idx),
        ]
        return ", ".join(part for part in parts if part)

    def _last_torque_debug_summary(self, env_idx: int | None) -> str:
        if env_idx is None or not isinstance(getattr(self, "torques", None), torch.Tensor):
            return ""
        if self.torques.numel() == 0 or env_idx < 0 or env_idx >= self.torques.shape[0]:
            return ""

        env_torques = self.torques[env_idx].detach()
        finite = torch.isfinite(env_torques)
        if not bool(finite.all()):
            return "; previous_applied_torques=non-finite"

        limits = self.env.torque_limits.to(device=env_torques.device, dtype=env_torques.dtype)
        saturation_frac = (env_torques.abs() >= limits).float().mean().cpu().item()
        return (
            f"; previous_applied_torques_env{env_idx}: "
            f"min={env_torques.min().cpu().item():.6g}, "
            f"max={env_torques.max().cpu().item():.6g}, "
            f"max_abs={env_torques.abs().max().cpu().item():.6g}, "
            f"saturation_frac={saturation_frac:.3f}"
        )

    def _env_idx_from_wbc_exception(self, exc: Exception) -> int | None:
        match = re.search(r"\benv\s+(\d+)\b", str(exc))
        if match is None:
            return None
        env_idx = int(match.group(1))
        if env_idx < 0 or env_idx >= self.env.num_envs:
            return None
        return env_idx

    def _wbc_input_context_summary(
        self,
        env_idx: int,
        q: np.ndarray,
        dq: np.ndarray,
        action_batch: np.ndarray,
        right_contact_points: np.ndarray,
        left_contact_points: np.ndarray,
        right_contact_bases: np.ndarray,
        left_contact_bases: np.ndarray,
        right_grfs: np.ndarray,
        left_grfs: np.ndarray,
        right_contact_in_contact: np.ndarray,
        left_contact_in_contact: np.ndarray,
    ) -> str:
        return (
            f"WBC input context for env {env_idx}: "
            f"{self._batch_state_summary(env_idx)}, "
            f"{self._bootstrap_state_summary(env_idx)}, "
            f"right_in_contact={bool(right_contact_in_contact[env_idx])}, "
            f"left_in_contact={bool(left_contact_in_contact[env_idx])}, "
            f"{self._tensor_row_summary('q', q, env_idx)}, "
            f"{self._tensor_row_summary('dq', dq, env_idx)}, "
            f"{self._tensor_row_summary('action', action_batch, env_idx)}, "
            f"{self._wbc_action_slice_summary(action_batch, env_idx)}, "
            f"{self._tensor_row_summary('right_contact_point', right_contact_points, env_idx)}, "
            f"{self._tensor_row_summary('left_contact_point', left_contact_points, env_idx)}, "
            f"{self._tensor_row_summary('right_contact_basis', right_contact_bases, env_idx)}, "
            f"{self._tensor_row_summary('left_contact_basis', left_contact_bases, env_idx)}, "
            f"{self._tensor_row_summary('right_grf', right_grfs, env_idx)}, "
            f"{self._tensor_row_summary('left_grf', left_grfs, env_idx)}"
        )

    def debug_summary(self, env_idx: int) -> str:
        if env_idx < 0 or env_idx >= self.env.num_envs:
            return ""

        parts = [
            self._batch_state_summary(env_idx),
            self._bootstrap_state_summary(env_idx),
            self._tensor_row_summary("raw_action", self._raw_actions, env_idx),
            self._tensor_row_summary("processed_action", self._processed_actions, env_idx),
            self._tensor_row_summary("actions_after_delay", self._actions_after_delay, env_idx),
            self._tensor_row_summary("last_wbc_q", self._last_wbc_q, env_idx),
            self._tensor_row_summary("last_wbc_dq", self._last_wbc_dq, env_idx),
            self._tensor_row_summary("last_wbc_torque_output", self._last_wbc_torque_output, env_idx),
            self._tensor_row_summary("last_wbc_action_batch", self._last_wbc_action_batch, env_idx),
            self._tensor_row_summary("last_wbc_sin_phase", self._last_wbc_sin_phase, env_idx),
            self._tensor_row_summary("last_wbc_cos_phase", self._last_wbc_cos_phase, env_idx),
            self._tensor_row_summary("last_wbc_desired_state", self._last_wbc_desired_state, env_idx),
            self._tensor_row_summary(
                "last_wbc_remaining_swing_duration", self._last_wbc_remaining_swing_duration, env_idx
            ),
            self._tensor_row_summary("right_contact_point", self._last_wbc_right_contact_points, env_idx),
            self._tensor_row_summary("left_contact_point", self._last_wbc_left_contact_points, env_idx),
            self._tensor_row_summary("right_contact_basis", self._last_wbc_right_contact_bases, env_idx),
            self._tensor_row_summary("left_contact_basis", self._last_wbc_left_contact_bases, env_idx),
            self._tensor_row_summary("right_grf", self._last_wbc_right_grfs, env_idx),
            self._tensor_row_summary("left_grf", self._last_wbc_left_grfs, env_idx),
        ]
        return ", ".join(part for part in parts if part)

    def _batch_state_summary(self, env_idx: int) -> str:
        try:
            states = self.wbc.get_states()
        except Exception:
            return ""
        if env_idx >= len(states):
            return ""
        state_part = f"batch_state={states[env_idx]}"
        try:
            transition_start_times = self.wbc.get_transition_start_times()
        except Exception:
            return state_part
        if env_idx >= len(transition_start_times):
            return state_part
        return f"{state_part}, batch_transition_start={float(transition_start_times[env_idx]):.6f}"

    def _sync_wbc_state_cache_from_controller(self) -> None:
        try:
            states = self.wbc.get_states()
            transition_start_times = self.wbc.get_transition_start_times()
        except Exception:
            return

        for env_idx in range(min(self.env.num_envs, len(states))):
            self.curr_state[env_idx] = int(states[env_idx])
        for env_idx in range(min(self.env.num_envs, len(transition_start_times))):
            self.transition_start_time[env_idx] = float(transition_start_times[env_idx])

    def _bootstrap_state_summary(self, env_idx: int) -> str:
        if env_idx < 0 or env_idx >= self.env.num_envs:
            return ""

        bootstrap_done = bool(self._wbc_bootstrap_done[env_idx].detach().cpu().item())
        bootstrap_hold = bool(self._last_wbc_bootstrap_hold_mask[env_idx].detach().cpu().item())
        dual_contact = bool(self._last_wbc_dual_contact_mask[env_idx].detach().cpu().item())
        dual_contact_count = int(self._dual_contact_counter[env_idx].detach().cpu().item())
        return (
            f"bootstrap_done={bootstrap_done}, "
            f"bootstrap_hold={bootstrap_hold}, "
            f"dual_contact={dual_contact}, "
            f"dual_contact_count={dual_contact_count}"
        )

    def _tensor_row_summary(self, label: str, value: torch.Tensor | np.ndarray | None, env_idx: int) -> str:
        if value is None:
            return ""

        if isinstance(value, torch.Tensor):
            if value.numel() == 0 or env_idx >= value.shape[0]:
                return ""
            row = value[env_idx].detach().flatten()
            finite = torch.isfinite(row)
            finite_note = "finite" if bool(finite.all()) else "non-finite"
            if row.numel() == 0:
                return f"{label}=empty"
            return (
                f"{label}({finite_note}): "
                f"min={row[finite].min().cpu().item():.6g}, "
                f"max={row[finite].max().cpu().item():.6g}, "
                f"max_abs={row[finite].abs().max().cpu().item():.6g}"
                if bool(finite.any())
                else f"{label}(non-finite): no finite values"
            )

        array = np.asarray(value)
        if array.size == 0 or env_idx >= array.shape[0]:
            return ""
        row_np = array[env_idx].reshape(-1)
        finite_np = np.isfinite(row_np)
        finite_note_np = "finite" if bool(finite_np.all()) else "non-finite"
        if not bool(finite_np.any()):
            return f"{label}(non-finite): no finite values"
        finite_values = row_np[finite_np]
        return (
            f"{label}({finite_note_np}): "
            f"min={finite_values.min():.6g}, "
            f"max={finite_values.max():.6g}, "
            f"max_abs={np.abs(finite_values).max():.6g}"
        )

    def _tensor_row_values(self, label: str, value: torch.Tensor | np.ndarray | None, env_idx: int) -> str:
        if value is None:
            return ""

        if isinstance(value, torch.Tensor):
            if value.numel() == 0 or env_idx >= value.shape[0]:
                return ""
            row_np = value[env_idx].detach().cpu().reshape(-1).numpy()
        else:
            array = np.asarray(value)
            if array.size == 0 or env_idx >= array.shape[0]:
                return ""
            row_np = array[env_idx].reshape(-1)

        values = np.array2string(
            np.asarray(row_np, dtype=np.float64),
            precision=9,
            separator=", ",
            suppress_small=False,
            max_line_width=1_000_000,
        )
        return f"{label}={values}"

    @staticmethod
    def _wbc_action_slices() -> tuple[tuple[str, slice], ...]:
        return (
            ("com_rel_pos", slice(0, 3)),
            ("com_vel", slice(3, 6)),
            ("pelvis_rel_pos", slice(6, 9)),
            ("pelvis_rel_ori", slice(9, 12)),
            ("pelvis_lin_vel", slice(12, 15)),
            ("pelvis_ang_vel", slice(15, 18)),
            ("torso_rel_ori", slice(18, 21)),
            ("torso_ang_vel", slice(21, 24)),
            ("right_foot_pos", slice(24, 27)),
            ("right_foot_ori", slice(27, 36)),
            ("right_foot_vel", slice(36, 42)),
            ("left_foot_pos", slice(42, 45)),
            ("left_foot_ori", slice(45, 54)),
            ("left_foot_vel", slice(54, 60)),
            ("unused", slice(60, 72)),
        )

    def _wbc_action_slice_summary(self, action_batch: np.ndarray | None, env_idx: int) -> str:
        if action_batch is None:
            return ""
        array = np.asarray(action_batch)
        if array.ndim != 2 or env_idx < 0 or env_idx >= array.shape[0]:
            return ""

        row = array[env_idx]
        parts: list[str] = []
        for name, action_slice in self._wbc_action_slices():
            values = row[action_slice]
            finite = np.isfinite(values)
            if bool(finite.all()):
                continue
            bad_local = np.argwhere(~finite).reshape(-1).tolist()
            bad_global = [int(action_slice.start + idx) for idx in bad_local]
            slice_values = np.array2string(
                np.asarray(values, dtype=np.float64),
                precision=9,
                separator=", ",
                suppress_small=False,
                max_line_width=1_000_000,
            )
            parts.append(f"{name}[{action_slice.start}:{action_slice.stop}] non-finite at {bad_global}: {slice_values}")
        if not parts:
            return "action_slices=finite"
        return "action_slices: " + "; ".join(parts)

    def _assert_finite_wbc_action_batch(self, action_batch: np.ndarray) -> None:
        finite = np.isfinite(action_batch)
        if bool(finite.all()):
            return

        bad_idx = np.argwhere(~finite)[0].tolist()
        env_idx = int(bad_idx[0])
        context = self._wbc_action_slice_summary(action_batch, env_idx)
        raise RuntimeError(
            f"wbc_action_batch has non-finite value at index {bad_idx}: {action_batch[tuple(bad_idx)]}; "
            f"shape={action_batch.shape}; {context}"
        )

    def _as_torch_tensor(self, value: Any, *, device: torch.device | str, dtype: torch.dtype, label: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
        elif getattr(value, "_is_tensor_proxy", False):
            tensor = value[:]
        else:
            tensor = torch.as_tensor(value)
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        return tensor.to(device=device, dtype=dtype)

    def _as_numpy_2d(self, value: Any, label: str) -> np.ndarray:
        tensor = self._as_torch_tensor(value, device="cpu", dtype=torch.float64, label=label)
        array = tensor.detach().cpu().numpy().astype(np.float64, copy=False)
        if array.ndim != 2:
            raise RuntimeError(f"{label} must be rank 2, got shape {array.shape}.")
        return np.ascontiguousarray(array)

    def _capture_bootstrap_hold_targets(self, env_ids: torch.Tensor | None) -> None:
        dof_pos = self._as_torch_tensor(
            self.env.simulator.dof_pos,
            device=self.env.device,
            dtype=self._wbc_bootstrap_hold_target_dof_pos.dtype,
            label="bootstrap_hold_dof_pos",
        )
        if env_ids is None:
            self._wbc_bootstrap_hold_target_dof_pos[:] = dof_pos
        else:
            self._wbc_bootstrap_hold_target_dof_pos[env_ids] = dof_pos[env_ids]

        self._assert_finite_torch("bootstrap_hold_target_dof_pos", self._wbc_bootstrap_hold_target_dof_pos)

    def _foot_contact_count_mask(self, side: str) -> np.ndarray:
        counts = getattr(self.env.simulator, f"{side}_foot_contact_count", None)
        if counts is None:
            return np.ones(self.env.num_envs, dtype=bool)

        counts_tensor = self._as_torch_tensor(
            counts,
            device=self.env.device,
            dtype=torch.long,
            label=f"{side}_foot_contact_count",
        ).reshape(-1)
        return np.ascontiguousarray((counts_tensor.detach().cpu().numpy() > 0).astype(bool, copy=False))

    def _batched_foot_contact_in_contact(self, side: str, local_wrenches: np.ndarray | None = None) -> np.ndarray:
        if local_wrenches is None:
            local_wrenches = np.stack(
                [self._local_foot_force_sensor_wrench(env_idx, side) for env_idx in range(self.env.num_envs)],
                axis=0,
            ).astype(np.float64, copy=False)
        local_z_force = np.abs(local_wrenches[:, 2])
        contact = (local_z_force > self._dual_contact_force_threshold) & self._foot_contact_count_mask(side)
        return np.ascontiguousarray(contact.astype(np.bool_, copy=False))

    def _update_wbc_bootstrap_state(self, right_grfs: np.ndarray, left_grfs: np.ndarray) -> torch.Tensor:
        right_force = np.linalg.norm(right_grfs[:, :3], axis=1)
        left_force = np.linalg.norm(left_grfs[:, :3], axis=1)
        right_contact = (right_force >= self._dual_contact_force_threshold) & self._foot_contact_count_mask("right")
        left_contact = (left_force >= self._dual_contact_force_threshold) & self._foot_contact_count_mask("left")
        dual_contact = torch.as_tensor(right_contact & left_contact, device=self.env.device, dtype=torch.bool)
        right_contact_tensor = torch.as_tensor(right_contact, device=self.env.device, dtype=torch.bool)
        left_contact_tensor = torch.as_tensor(left_contact, device=self.env.device, dtype=torch.bool)

        self._last_wbc_dual_contact_mask[:] = dual_contact
        waiting = ~self._wbc_bootstrap_done
        started = torch.zeros_like(waiting)
        self._startup_gait_counter[waiting] += 1
        self._dual_contact_counter[waiting & dual_contact] += 1
        self._dual_contact_counter[waiting & ~dual_contact] = 0

        ready = waiting & (self._dual_contact_counter >= self._dual_contact_required_steps)
        if bool(ready.any()):
            self._wbc_bootstrap_done[ready] = True
            started |= ready
            self._startup_desired_state[ready] = int(self.State.DUAL_STANCE)
            self._shift_gait_phase_for_startup(ready, self._startup_desired_state)
            sim_time = float(self.env.simulator.time())
            for env_idx in ready.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
                self.curr_state[env_idx] = self.State.DUAL_STANCE
                self.transition_start_time[env_idx] = sim_time
            self._pending_wbc_reinitialize[ready] = True

        timed_out = waiting & ~ready & (self._startup_gait_counter >= self._startup_gait_timeout_steps)
        if bool(timed_out.any()):
            fallback_states = self._startup_fallback_states(right_contact_tensor, left_contact_tensor)
            self._startup_desired_state[timed_out] = fallback_states[timed_out]
            self._wbc_bootstrap_done[timed_out] = True
            started |= timed_out
            self._shift_gait_phase_for_startup(timed_out, self._startup_desired_state)
            sim_time = float(self.env.simulator.time())
            for env_idx in timed_out.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
                self.curr_state[env_idx] = int(self._startup_desired_state[env_idx].detach().cpu().item())
                self.transition_start_time[env_idx] = sim_time
            self._pending_wbc_reinitialize[timed_out] = True

        self._last_wbc_bootstrap_hold_mask[:] = ~self._wbc_bootstrap_done
        return started

    def _startup_fallback_states(self, right_contact: torch.Tensor, left_contact: torch.Tensor) -> torch.Tensor:
        fallback = torch.full((self.env.num_envs,), int(self.State.DUAL_STANCE), device=self.env.device, dtype=torch.long)
        fallback[left_contact & ~right_contact] = int(self.State.LEFT_STANCE)
        fallback[right_contact & ~left_contact] = int(self.State.RIGHT_STANCE)
        return fallback

    def _shift_gait_phase_for_startup(self, env_mask: torch.Tensor, desired_states: torch.Tensor) -> None:
        if not bool(env_mask.any()):
            return
        sin_phase, cos_phase = self._phase_features_for_wbc(self._actions_after_delay)
        raw_phase = torch.remainder(torch.atan2(sin_phase[:, 0], cos_phase[:, 0]), 2.0 * torch.pi) / (2.0 * torch.pi)
        target_phase = self._startup_target_phase_for_states(desired_states)
        self._wbc_phase_shift_frac[env_mask] = torch.remainder(target_phase[env_mask] - raw_phase[env_mask], 1.0)

    def _startup_target_phase_for_states(self, desired_states: torch.Tensor) -> torch.Tensor:
        cycle_time = self._gait_cycle_times(self.env.device, torch.float32)
        targets = torch.zeros(self.env.num_envs, device=self.env.device, dtype=torch.float32)
        for env_idx in range(self.env.num_envs):
            layout = self._phase_layout(float(cycle_time[env_idx].detach().cpu().item()))
            state = int(desired_states[env_idx].detach().cpu().item())
            if state == int(self.State.DUAL_STANCE):
                targets[env_idx] = float(layout["dual_1_mid"])
            elif state == int(self.State.RIGHT_STANCE):
                targets[env_idx] = float(layout["right_mid"])
            else:
                targets[env_idx] = float(layout["left_mid"])
        return targets

    def _compute_bootstrap_hold_torques(self, dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> torch.Tensor:
        hold_torques = (
            self._kp_scale * self.p_gains.unsqueeze(0) * (self._wbc_bootstrap_hold_target_dof_pos - dof_pos)
            - self._kd_scale * self.d_gains.unsqueeze(0) * dof_vel
        )
        if self.env.robot_config.control.clip_torques:
            torque_limits = self.env.torque_limits.to(device=hold_torques.device, dtype=hold_torques.dtype)
            hold_torques = torch.clamp(hold_torques, min=-torque_limits, max=torque_limits)

        self._assert_finite_torch("bootstrap_hold_torques", hold_torques)
        return hold_torques

    def _reinitialize_wbc_startup_state(self, q: np.ndarray, dq: np.ndarray, *, update_robot: bool = True) -> None:
        env_ids = self._pending_wbc_reinitialize.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return

        sim_time = float(self.env.simulator.time())
        if update_robot:
            self.wbc.update_robot(q, dq)
        env_id_list = [int(env_idx) for env_idx in env_ids.detach().cpu().tolist()]
        for env_idx in env_id_list:
            state = int(self._startup_desired_state[env_idx].detach().cpu().item())
            self.wbc.reset_state(state, sim_time, env_idx)

        for env_idx in env_id_list:
            self.curr_state[env_idx] = int(self._startup_desired_state[env_idx].detach().cpu().item())
            self.transition_start_time[env_idx] = sim_time
        self._pending_wbc_reinitialize[env_ids] = False

    def _actions_for_batched_wbc(
        self,
        actions: torch.Tensor,
        *,
        desired_states: np.ndarray | None = None,
        swing_sides: list[str | None] | None = None,
        remaining_swing_durations: np.ndarray | None = None,
    ) -> np.ndarray:
        action_scales = self.action_scales
        if action_scales.device != actions.device or action_scales.dtype != actions.dtype:
            action_scales = action_scales.to(device=actions.device, dtype=actions.dtype)
        scaled_actions = actions * action_scales
        self._assert_finite_torch_debug("wbc_scaled_actions", scaled_actions)
        if scaled_actions.shape[-1] != 9:
            raise ValueError(f"Expected 9-D policy actions for WBC expansion, got shape {tuple(scaled_actions.shape)}.")
        scaled_actions = self._with_commanded_pelvis_velocity_action(scaled_actions)

        if desired_states is None or swing_sides is None or remaining_swing_durations is None:
            sin_phase, cos_phase = self._phase_features_for_wbc(actions)
            desired_states, swing_sides, remaining_swing_durations = self._desired_states_from_phase(sin_phase, cos_phase)

        wbc_actions = self._actions_to_wbc_targets(
            scaled_actions,
            desired_states=desired_states,
            swing_sides=swing_sides,
            remaining_swing_durations=remaining_swing_durations,
        )
        self._assert_finite_numpy_debug("wbc_expanded_actions", wbc_actions)
        return np.ascontiguousarray(wbc_actions)

    def _with_commanded_pelvis_velocity_action(self, actions: torch.Tensor) -> torch.Tensor:
        if not self._use_command_as_pelvis_velocity_action:
            return actions

        command_manager = getattr(self.env, "command_manager", None)
        commands = getattr(command_manager, "commands", None)
        if commands is None:
            return actions

        command_tensor = self._as_torch_tensor(
            commands,
            device=actions.device,
            dtype=actions.dtype,
            label="locomotion_commands",
        )
        if command_tensor.ndim != 2 or command_tensor.shape[0] < actions.shape[0] or command_tensor.shape[1] < 3:
            return actions

        commanded = command_tensor[: actions.shape[0], :3]
        if not torch.isfinite(commanded).all():
            return actions

        updated_actions = actions.clone()
        updated_actions[:, 0:3] = 0.0
        updated_actions[:, 0:2] = commanded[:, 0:2]
        updated_actions[:, 3:6] = 0.0
        updated_actions[:, 5] = commanded[:, 2]
        self._assert_finite_torch_debug("wbc_commanded_pelvis_velocity_actions", updated_actions)
        return updated_actions

    def _actions_to_wbc_targets(
        self,
        actions: torch.Tensor,
        *,
        desired_states: np.ndarray,
        swing_sides: list[str | None],
        remaining_swing_durations: np.ndarray,
    ) -> np.ndarray:
        num_envs = actions.shape[0]
        action_dict = parse_actions(actions)
        action_np = {
            key: value.detach().cpu().numpy().astype(np.float64, copy=False)
            for key, value in action_dict.items()
        }

        self._integrate_wbc_motion_targets(actions)
        # Batched WBC target layout:
        # com relative pos / lin vel(3/3),
        # pelvis relative pos / relative ori / lin vel / ang vel(3/3/3/3),
        # torso relative ori / ang vel(3/3),
        # right foot pos/ori/vel(3/9/6),
        # left foot pos/ori/vel(3/9/6).
        # The binding currently selects this layout with action_dim == 72 but consumes
        # only the first 60 values; keep the trailing 12 entries as zeros.
        wbc_actions = np.zeros((num_envs, 72), dtype=np.float64)
        wbc_actions[:, 0:3] = self._wbc_integrated_com_pos[:num_envs].detach().cpu().numpy().astype(
            np.float64, copy=False
        )
        wbc_actions[:, 3:6] = action_np["com_vel"]
        wbc_actions[:, 6:9] = self._wbc_integrated_pelvis_pos[:num_envs].detach().cpu().numpy().astype(
            np.float64, copy=False
        )
        pelvis_rot = self._wbc_integrated_pelvis_rot[:num_envs].detach().cpu().numpy()
        torso_rot = self._wbc_integrated_torso_rot[:num_envs].detach().cpu().numpy()
        wbc_actions[:, 9:12] = np.stack([matrix_to_axis_angle(rotation) for rotation in pelvis_rot], axis=0)
        wbc_actions[:, 12:15] = action_np["com_vel"]
        wbc_actions[:, 15:18] = action_np["pelvis_ang_vel"]
        wbc_actions[:, 18:21] = np.stack([matrix_to_axis_angle(rotation) for rotation in torso_rot], axis=0)
        wbc_actions[:, 21:24] = action_np["pelvis_ang_vel"]  # match with the pelvis for now

        current_poses = self._batched_foot_pose_matrices()
        current_linear_velocities = self._batched_foot_linear_velocities()
        self._fill_absolute_foot_targets(wbc_actions, current_poses)
        commanded_velocities = self._commanded_landing_velocities(action_np, num_envs)
        landing_poses: dict[int, tuple[str, np.ndarray]] = {}
        left_stance_state = int(self.State.LEFT_STANCE)
        right_stance_state = int(self.State.RIGHT_STANCE)
        for env_idx, fallback_swing_side in enumerate(swing_sides[:num_envs]):
            desired_state = int(desired_states[env_idx])
            if desired_state == left_stance_state:
                swing_side = "right"
            elif desired_state == right_stance_state:
                swing_side = "left"
            else:
                swing_side = fallback_swing_side
            if swing_side is None:
                self._wbc_nominal_landing_swing_side[env_idx] = -1
                self._reset_landing_target(env_idx)
                continue

            stance_side = "right" if swing_side == "left" else "left"
            stance_pose = current_poses[stance_side][env_idx]
            stance_pos = stance_pose[:3, 3]
            stance_yaw = yaw_from_matrix(stance_pose[:3, :3])
            nominal_pos, nominal_yaw, nominal_stance_pos, nominal_stance_yaw = self._nominal_landing_pose_for_swing(
                env_idx,
                swing_side,
                stance_pos,
                stance_yaw,
                commanded_velocities[env_idx],
                remaining_swing_durations[env_idx],
            )
            landing_pos, landing_yaw = add_landing_residual(
                nominal_pos,
                nominal_yaw,
                nominal_stance_pos,
                nominal_stance_yaw,
                action_np["landing_foot_delta_pose"][env_idx],
                swing_is_left=(swing_side == "left"),
            )
            landing_pos, landing_yaw = self._landing_target_for_swing(
                env_idx,
                swing_side,
                landing_pos,
                landing_yaw,
                remaining_swing_durations[env_idx],
            )
            landing_rot = yaw_to_matrix3d(landing_yaw)
            landing_pos[2] = self._landing_ground_plane_z
            landing_poses[env_idx] = (swing_side, pose_matrix(landing_pos, landing_rot))

        self._maybe_draw_swing_foot_plan(current_poses, landing_poses)
        replan_mask = self._landing_plan_replan_mask(landing_poses)

        desired_poses = self._batched_foot_planner.desired_pose_batch(
            current_poses=current_poses,
            current_linear_velocities=current_linear_velocities,
            landing_poses=landing_poses,
            remaining_durations=remaining_swing_durations,
            replan_mask=replan_mask,
        )
        self._maybe_print_swing_foot_debug(
            desired_states=desired_states,
            swing_sides=swing_sides,
            remaining_swing_durations=remaining_swing_durations,
            current_poses=current_poses,
            current_linear_velocities=current_linear_velocities,
            landing_poses=landing_poses,
            replan_mask=replan_mask,
            desired_poses=desired_poses,
        )
        self._wbc_replan_swing_trajectory.zero_()
        for env_idx, (swing_side, desired_pose, desired_velocity) in desired_poses.items():
            self._write_absolute_foot_target(
                wbc_actions,
                env_idx,
                swing_side,
                desired_pose[:3, 3],
                desired_pose[:3, :3],
                desired_velocity,
            )

        return wbc_actions

    def _nominal_landing_pose_for_swing(
        self,
        env_idx: int,
        swing_side: str,
        stance_pos: np.ndarray,
        stance_yaw: float,
        commanded_velocity: np.ndarray,
        remaining_swing_duration: float,
    ) -> tuple[np.ndarray, float, np.ndarray, float]:
        swing_code = 0 if swing_side == "right" else 1
        cached_code = int(self._wbc_nominal_landing_swing_side[env_idx].detach().cpu().item())
        if cached_code != swing_code:
            nominal_pos, nominal_yaw = nominal_landing_foot_pose(
                stance_pos,
                stance_yaw,
                commanded_velocity,
                swing_is_left=(swing_side == "left"),
                step_time=max(float(remaining_swing_duration), self._wbc_dt),
            )
            nominal_pos[2] = self._landing_ground_plane_z
            self._wbc_nominal_landing_swing_side[env_idx] = swing_code
            self._wbc_nominal_landing_pos[env_idx] = torch.as_tensor(
                nominal_pos,
                device=self.env.device,
                dtype=self._wbc_nominal_landing_pos.dtype,
            )
            self._wbc_nominal_landing_yaw[env_idx] = float(nominal_yaw)
            self._wbc_nominal_landing_stance_pos[env_idx] = torch.as_tensor(
                stance_pos,
                device=self.env.device,
                dtype=self._wbc_nominal_landing_stance_pos.dtype,
            )
            self._wbc_nominal_landing_stance_yaw[env_idx] = float(stance_yaw)

        return (
            self._wbc_nominal_landing_pos[env_idx].detach().cpu().numpy().astype(np.float64, copy=True),
            float(self._wbc_nominal_landing_yaw[env_idx].detach().cpu().item()),
            self._wbc_nominal_landing_stance_pos[env_idx].detach().cpu().numpy().astype(np.float64, copy=True),
            float(self._wbc_nominal_landing_stance_yaw[env_idx].detach().cpu().item()),
        )

    def _reset_landing_target(self, env_idx: int) -> None:
        self._wbc_landing_target_swing_side[env_idx] = -1
        self._wbc_landing_target_initial_duration[env_idx] = 0.0
        self._wbc_landing_target_frozen[env_idx] = False
        self._last_wbc_landing_target_update_fraction[env_idx] = 0.0

    def _landing_target_for_swing(
        self,
        env_idx: int,
        swing_side: str,
        candidate_pos: np.ndarray,
        candidate_yaw: float,
        remaining_swing_duration: float,
    ) -> tuple[np.ndarray, float]:
        swing_code = 0 if swing_side == "right" else 1
        cached_code = int(self._wbc_landing_target_swing_side[env_idx].detach().cpu().item())
        remaining_duration = max(float(remaining_swing_duration), self._wbc_dt)
        new_swing = cached_code != swing_code

        if new_swing:
            initial_duration = remaining_duration
            update_fraction = 0.0
            frozen = False
            target_pos = np.asarray(candidate_pos, dtype=np.float64).reshape(3).copy()
            target_yaw = float(candidate_yaw)
            self._wbc_landing_target_swing_side[env_idx] = swing_code
            self._wbc_landing_target_initial_duration[env_idx] = initial_duration
        else:
            initial_duration = max(
                float(self._wbc_landing_target_initial_duration[env_idx].detach().cpu().item()),
                self._wbc_dt,
            )
            elapsed_duration = max(initial_duration - remaining_duration, 0.0)
            update_fraction = float(np.clip(elapsed_duration / initial_duration, 0.0, 1.0))
            frozen = bool(self._wbc_landing_target_frozen[env_idx].detach().cpu().item())
            if update_fraction >= self._landing_target_update_fraction:
                frozen = True

            if frozen:
                target_pos = self._wbc_landing_target_pos[env_idx].detach().cpu().numpy().astype(np.float64, copy=True)
                target_yaw = float(self._wbc_landing_target_yaw[env_idx].detach().cpu().item())
            else:
                target_pos = np.asarray(candidate_pos, dtype=np.float64).reshape(3).copy()
                target_yaw = float(candidate_yaw)

        target_pos[2] = self._landing_ground_plane_z
        self._wbc_landing_target_pos[env_idx] = torch.as_tensor(
            target_pos,
            device=self.env.device,
            dtype=self._wbc_landing_target_pos.dtype,
        )
        self._wbc_landing_target_yaw[env_idx] = target_yaw
        self._wbc_landing_target_frozen[env_idx] = frozen
        self._last_wbc_landing_target_update_fraction[env_idx] = update_fraction
        return target_pos, target_yaw

    def _maybe_print_swing_foot_debug(
        self,
        *,
        desired_states: np.ndarray,
        swing_sides: list[str | None],
        remaining_swing_durations: np.ndarray,
        current_poses: dict[str, np.ndarray],
        current_linear_velocities: dict[str, np.ndarray],
        landing_poses: dict[int, tuple[str, np.ndarray]],
        replan_mask: np.ndarray,
        desired_poses: dict[int, tuple[str, np.ndarray, np.ndarray]],
    ) -> None:
        if not self._debug_swing_foot:
            return

        self._debug_swing_foot_counter += 1
        if self._debug_swing_foot_counter % self._debug_swing_foot_interval != 0 and not bool(np.any(replan_mask)):
            return

        sim_time = float(self.env.simulator.time())
        for env_idx in range(min(self.env.num_envs, len(desired_states))):
            landing_entry = landing_poses.get(env_idx)
            desired_entry = desired_poses.get(env_idx)
            if landing_entry is None and desired_entry is None and not bool(replan_mask[env_idx]):
                continue

            if desired_entry is not None:
                swing_side, desired_pose, desired_velocity = desired_entry
            elif landing_entry is not None:
                swing_side, _landing_pose = landing_entry
                desired_pose = None
                desired_velocity = None
            else:
                swing_side = swing_sides[env_idx]
                desired_pose = None
                desired_velocity = None

            current_pos = None
            current_lin_vel = None
            if swing_side in ("right", "left"):
                current_pos = current_poses[swing_side][env_idx, :3, 3]
                current_lin_vel = current_linear_velocities[swing_side][env_idx]

            landing_pose = landing_entry[1] if landing_entry is not None else None
            landing_pos = landing_pose[:3, 3] if landing_pose is not None else None
            desired_pos = desired_pose[:3, 3] if desired_pose is not None else None
            distance_to_landing = (
                float(np.linalg.norm(landing_pos - current_pos))
                if landing_pos is not None and current_pos is not None
                else float("nan")
            )
            step_distance = (
                float(np.linalg.norm(desired_pos - current_pos))
                if desired_pos is not None and current_pos is not None
                else float("nan")
            )
            desired_speed = (
                float(np.linalg.norm(desired_velocity[:3]))
                if desired_velocity is not None and desired_velocity.shape[0] >= 3
                else float("nan")
            )
            current_speed = float(np.linalg.norm(current_lin_vel)) if current_lin_vel is not None else float("nan")
            landing_target_fraction = float(
                self._last_wbc_landing_target_update_fraction[env_idx].detach().cpu().item()
            )
            landing_target_frozen = bool(self._wbc_landing_target_frozen[env_idx].detach().cpu().item())

            print(
                "SWING_FOOT_DEBUG "
                f"t={sim_time:.6f} env={env_idx} state={int(desired_states[env_idx])} "
                f"swing={swing_side} fallback_swing={swing_sides[env_idx]} "
                f"remaining={float(remaining_swing_durations[env_idx]):.6f} wbc_dt={self._wbc_dt:.6f} "
                f"landing_update_frac={landing_target_fraction:.6f} landing_frozen={landing_target_frozen} "
                f"replan={bool(replan_mask[env_idx])} "
                f"changed={bool(self._last_wbc_landing_plan_changed[env_idx])} "
                f"forced={bool(self._last_wbc_landing_plan_forced[env_idx])} "
                f"dist_to_landing={distance_to_landing:.6f} step_dist={step_distance:.6f} "
                f"current_speed={current_speed:.6f} desired_speed={desired_speed:.6f} "
                f"current_pos={self._format_np_debug(current_pos)} "
                f"landing_pos={self._format_np_debug(landing_pos)} "
                f"desired_pos={self._format_np_debug(desired_pos)} "
                f"current_lin_vel={self._format_np_debug(current_lin_vel)} "
                f"desired_vel={self._format_np_debug(desired_velocity)}"
            )

    @staticmethod
    def _format_np_debug(value: np.ndarray | None) -> str:
        if value is None:
            return "None"
        return np.array2string(
            np.asarray(value, dtype=np.float64).reshape(-1),
            precision=6,
            separator=",",
            suppress_small=False,
            max_line_width=1_000_000,
        )

    def _commanded_landing_velocities(self, action_np: dict[str, np.ndarray], num_envs: int) -> np.ndarray:
        yaw_vel = action_np["pelvis_ang_vel"][:num_envs, 2:3]
        fallback = np.concatenate(
            [
                action_np["com_vel"][:num_envs, :2],
                yaw_vel,
            ],
            axis=1,
        ).astype(np.float64, copy=False)
        if not self._use_command_as_landing_velocity:
            return fallback

        command_manager = getattr(self.env, "command_manager", None)
        commands = getattr(command_manager, "commands", None)
        if commands is None:
            return fallback

        try:
            command_tensor = self._as_torch_tensor(
                commands,
                device="cpu",
                dtype=torch.float64,
                label="locomotion_commands",
            )
        except Exception:
            return fallback

        if command_tensor.ndim != 2 or command_tensor.shape[0] < num_envs or command_tensor.shape[1] < 3:
            return fallback

        commanded = command_tensor[:num_envs, :3].detach().cpu().numpy().astype(np.float64, copy=False)
        if not np.all(np.isfinite(commanded)):
            return fallback
        return commanded

    def _landing_plan_replan_mask(self, landing_poses: dict[int, tuple[str, np.ndarray]]) -> np.ndarray:
        inputs = np.full((self.env.num_envs, 5), np.nan, dtype=np.float64)
        for env_idx, (swing_side, landing_pose) in landing_poses.items():
            pose = np.asarray(landing_pose, dtype=np.float64)
            swing_code = 0.0 if swing_side == "right" else 1.0
            inputs[env_idx] = np.array(
                [swing_code, pose[0, 3], pose[1, 3], pose[2, 3], yaw_from_matrix(pose[:3, :3])],
                dtype=np.float64,
            )

        active = np.isfinite(inputs).all(axis=1)
        inputs_for_cache = inputs.copy()
        previous_all = self._last_wbc_landing_plan_inputs.detach().cpu().numpy()
        inputs_for_cache[~active] = previous_all[~active]
        inputs = inputs[: self.env.num_envs]
        if not np.all(np.isfinite(inputs)):
            inputs = inputs_for_cache

        previous = previous_all
        previous_finite = np.isfinite(previous).all(axis=1)
        tolerances = np.array(
            [
                0.0,
                self._landing_plan_position_replan_tolerance,
                self._landing_plan_position_replan_tolerance,
                self._landing_plan_position_replan_tolerance,
                self._landing_plan_yaw_replan_tolerance,
            ],
            dtype=np.float64,
        )
        changed = active & (
            ~previous_finite
            | (np.abs(previous - inputs) > tolerances.reshape(1, -1)).any(axis=1)
        )
        forced = active & self._wbc_replan_swing_trajectory.detach().cpu().numpy().astype(bool, copy=False)
        self._last_wbc_landing_plan_changed = changed.copy()
        self._last_wbc_landing_plan_forced = forced.copy()

        self._last_wbc_landing_plan_inputs[:] = torch.as_tensor(
            inputs_for_cache,
            device=self.env.device,
            dtype=self._last_wbc_landing_plan_inputs.dtype,
        )
        return np.ascontiguousarray(changed | forced)

    def _integrate_wbc_motion_targets(self, actions: torch.Tensor) -> None:
        dt = self._wbc_dt
        action_dict = parse_actions(actions)
        integrated_com_velocity = action_dict["com_vel"].to(
            device=self.env.device, dtype=self._wbc_integrated_com_pos.dtype
        )
        self._wbc_integrated_com_pos[: actions.shape[0]] += integrated_com_velocity * dt
        self._wbc_integrated_pelvis_pos[: actions.shape[0]] += integrated_com_velocity.to(
            dtype=self._wbc_integrated_pelvis_pos.dtype
        ) * dt

        angular_delta = action_dict["pelvis_ang_vel"].detach().cpu().numpy().astype(float, copy=False) * dt
        for env_idx, delta in enumerate(angular_delta[: actions.shape[0]]):
            if np.linalg.norm(delta) < 1.0e-12:
                continue
            delta_rot = exp_map(np.asarray(delta, dtype=float).reshape(3))
            pelvis_rot = self._wbc_integrated_pelvis_rot[env_idx].detach().cpu().numpy()
            torso_rot = self._wbc_integrated_torso_rot[env_idx].detach().cpu().numpy()
            self._wbc_integrated_pelvis_rot[env_idx] = torch.as_tensor(
                pelvis_rot @ delta_rot,
                device=self.env.device,
                dtype=self._wbc_integrated_pelvis_rot.dtype,
            )
            self._wbc_integrated_torso_rot[env_idx] = torch.as_tensor(
                torso_rot @ delta_rot,
                device=self.env.device,
                dtype=self._wbc_integrated_torso_rot.dtype,
            )

    def _reset_wbc_integrated_targets(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_index = torch.arange(self.env.num_envs, device=self.env.device)
        elif env_ids.dtype == torch.bool:
            env_index = env_ids.nonzero(as_tuple=False).flatten()
        else:
            env_index = env_ids.flatten().to(device=self.env.device, dtype=torch.long)

        if env_index.numel() == 0:
            return

        self._wbc_integrated_com_pos[env_index] = 0.0
        self._wbc_integrated_pelvis_pos[env_index] = 0.0
        identity = torch.eye(3, device=self.env.device, dtype=self._wbc_integrated_pelvis_rot.dtype)
        identity_batch = identity.unsqueeze(0).expand(env_index.numel(), -1, -1)
        self._wbc_integrated_pelvis_rot[env_index] = identity_batch
        self._wbc_integrated_torso_rot[env_index] = identity_batch.to(dtype=self._wbc_integrated_torso_rot.dtype)

    def _fill_absolute_foot_targets(self, wbc_actions: np.ndarray, current_poses: dict[str, np.ndarray]) -> None:
        fixed_poses = getattr(self, "_fixed_wbc_foot_target_poses", None)
        for side in ("right", "left"):
            for env_idx in range(wbc_actions.shape[0]):
                pose = current_poses[side][env_idx]
                if isinstance(fixed_poses, dict) and side in fixed_poses:
                    side_poses = fixed_poses[side]
                    if env_idx < side_poses.shape[0]:
                        pose = side_poses[env_idx]
                self._write_absolute_foot_target(
                    wbc_actions,
                    env_idx,
                    side,
                    pose[:3, 3],
                    pose[:3, :3],
                    np.zeros(6, dtype=np.float64),
                )

    @staticmethod
    def _write_absolute_foot_target(
        wbc_actions: np.ndarray,
        env_idx: int,
        side: str,
        position: np.ndarray,
        rotation: np.ndarray,
        velocity: np.ndarray,
    ) -> None:
        if side == "right":
            pos_slice = slice(24, 27)
            ori_slice = slice(27, 36)
            vel_slice = slice(36, 42)
        elif side == "left":
            pos_slice = slice(42, 45)
            ori_slice = slice(45, 54)
            vel_slice = slice(54, 60)
        else:
            raise ValueError(f"Unknown foot side {side!r}.")

        wbc_actions[env_idx, pos_slice] = np.asarray(position, dtype=np.float64).reshape(3)
        wbc_actions[env_idx, ori_slice] = np.asarray(rotation, dtype=np.float64).reshape(3, 3).reshape(9)
        wbc_actions[env_idx, vel_slice] = np.asarray(velocity, dtype=np.float64).reshape(6)

    def _phase_features_for_wbc(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gait_state = self.env.command_manager.get_state("locomotion_gait")
        if gait_state is None:
            sin_phase = torch.zeros(self.env.num_envs, 2, device=actions.device, dtype=actions.dtype)
            cos_phase = torch.ones_like(sin_phase)
            return sin_phase, cos_phase

        gait_state = cast("LocomotionGait", gait_state)
        phase = gait_state.phase
        if phase is None:
            sin_phase = torch.zeros(self.env.num_envs, 2, device=actions.device, dtype=actions.dtype)
            cos_phase = torch.ones_like(sin_phase)
            return sin_phase, cos_phase

        phase = phase.to(device=actions.device, dtype=actions.dtype)
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)
        self._assert_finite_torch("wbc_sin_phase", sin_phase)
        self._assert_finite_torch("wbc_cos_phase", cos_phase)
        return sin_phase, cos_phase

    def _phase_layout(self, cycle_time: float) -> dict[str, float]:
        stable_weight_total = 1.0
        transition_frac = np.clip(float(self._wbc_transition_time) / max(cycle_time, self._wbc_dt), 0.0, 0.24)
        stable_frac = max(1.0 - 4.0 * transition_frac, 0.0)
        left_stable_frac = stable_frac * 0.40 / stable_weight_total
        dual_stable_frac = stable_frac * 0.10 / stable_weight_total
        right_stable_frac = stable_frac * 0.40 / stable_weight_total

        left_end = left_stable_frac
        dual_1_end = left_end + transition_frac + dual_stable_frac
        right_end = dual_1_end + transition_frac + right_stable_frac
        dual_2_end = right_end + transition_frac + dual_stable_frac
        return {
            "left_end": left_end,
            "dual_1_end": dual_1_end,
            "right_end": right_end,
            "dual_2_end": dual_2_end,
            "left_mid": 0.5 * left_end,
            "dual_1_mid": left_end + transition_frac + 0.5 * dual_stable_frac,
            "right_mid": dual_1_end + transition_frac + 0.5 * right_stable_frac,
        }

    def _desired_states_from_phase(
        self,
        sin_phase: torch.Tensor,
        cos_phase: torch.Tensor,
        *,
        left_stance_weight: float = 0.40,
        dual_stance_weight: float = 0.10,
        right_stance_weight: float = 0.40,
        max_transition_frac: float = 0.24,
    ) -> tuple[np.ndarray, list[str | None], np.ndarray]:
        stable_weight_total = left_stance_weight + dual_stance_weight + right_stance_weight + dual_stance_weight
        if stable_weight_total <= 0.0:
            raise ValueError("Phase state weights must have positive total weight.")

        left_phase = torch.atan2(sin_phase[:, 0], cos_phase[:, 0])
        normalized_phase = torch.remainder(left_phase, 2.0 * torch.pi) / (2.0 * torch.pi)
        phase_shift = self._wbc_phase_shift_frac.to(device=normalized_phase.device, dtype=normalized_phase.dtype)
        normalized_phase = torch.remainder(normalized_phase + phase_shift, 1.0)
        phase_np = normalized_phase.detach().cpu().numpy().astype(np.float64, copy=False)
        cycle_time = self._gait_cycle_times(sin_phase.device, sin_phase.dtype).detach().cpu().numpy()

        desired_states = np.full(self.env.num_envs, int(self.State.DUAL_STANCE), dtype=np.int64)
        swing_sides: list[str | None] = [None for _ in range(self.env.num_envs)]
        remaining = np.full(self.env.num_envs, self._wbc_dt, dtype=np.float64)

        for env_idx, phase in enumerate(phase_np):
            env_cycle_time = max(float(cycle_time[env_idx]), self._wbc_dt)
            transition_frac = np.clip(float(self._wbc_transition_time) / env_cycle_time, 0.0, max_transition_frac)
            stable_frac = max(1.0 - 4.0 * transition_frac, 0.0)
            left_stable_frac = stable_frac * left_stance_weight / stable_weight_total
            dual_stable_frac = stable_frac * dual_stance_weight / stable_weight_total
            right_stable_frac = stable_frac * right_stance_weight / stable_weight_total

            left_end = left_stable_frac
            dual_1_end = left_end + transition_frac + dual_stable_frac
            right_end = dual_1_end + transition_frac + right_stable_frac
            dual_2_end = right_end + transition_frac + dual_stable_frac

            if phase < left_end:
                desired_states[env_idx] = int(self.State.LEFT_STANCE)
                swing_sides[env_idx] = "right"
                remaining[env_idx] = max((left_end - phase) * env_cycle_time, self._wbc_dt)
            elif phase < dual_1_end:
                desired_states[env_idx] = int(self.State.DUAL_STANCE)
                swing_sides[env_idx] = None
                remaining[env_idx] = max((dual_1_end - phase) * env_cycle_time, self._wbc_dt)
            elif phase < right_end:
                desired_states[env_idx] = int(self.State.RIGHT_STANCE)
                swing_sides[env_idx] = "left"
                remaining[env_idx] = max((right_end - phase) * env_cycle_time, self._wbc_dt)
            elif phase < dual_2_end:
                desired_states[env_idx] = int(self.State.DUAL_STANCE)
                swing_sides[env_idx] = None
                remaining[env_idx] = max((dual_2_end - phase) * env_cycle_time, self._wbc_dt)
            else:
                desired_states[env_idx] = int(self.State.LEFT_STANCE)
                swing_sides[env_idx] = "right"
                remaining[env_idx] = max((1.0 - phase + left_end) * env_cycle_time, self._wbc_dt)

        return desired_states, swing_sides, remaining.astype(np.float64, copy=False)

    def _maybe_force_wbc_desired_state(
        self,
        desired_states: np.ndarray,
        swing_sides: list[str | None],
        remaining_swing_durations: np.ndarray,
    ) -> tuple[np.ndarray, list[str | None], np.ndarray]:
        forced_state = getattr(self, "_forced_wbc_desired_state", None)
        if forced_state is None:
            return desired_states, swing_sides, remaining_swing_durations

        forced_state = int(forced_state)
        desired_states = np.full_like(desired_states, forced_state)
        remaining_swing_durations = np.full_like(remaining_swing_durations, self._wbc_dt, dtype=np.float64)

        if forced_state == int(self.State.DUAL_STANCE):
            swing_sides = [None for _ in swing_sides]
        elif forced_state == int(self.State.LEFT_STANCE):
            swing_sides = ["right" for _ in swing_sides]
        elif forced_state == int(self.State.RIGHT_STANCE):
            swing_sides = ["left" for _ in swing_sides]
        else:
            raise ValueError(f"Unsupported forced WBC desired state: {forced_state}.")

        return desired_states, swing_sides, remaining_swing_durations

    def _gait_cycle_times(self, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
        gait_state = self.env.command_manager.get_state("locomotion_gait")
        if gait_state is not None:
            gait_state = cast("LocomotionGait", gait_state)
            gait_freq = gait_state.gait_freq
            if gait_freq is not None:
                freq = gait_freq.reshape(-1).to(device=device, dtype=dtype).clamp_min(1.0e-6)
                return 1.0 / freq
            gait_period = float(getattr(gait_state, "gait_period", 1.0))
            return torch.full((self.env.num_envs,), gait_period, device=device, dtype=dtype)
        return torch.full((self.env.num_envs,), 1.0, device=device, dtype=dtype)

    def _batched_foot_pose_matrices(self) -> dict[str, np.ndarray]:
        poses: dict[str, np.ndarray] = {}
        for side in ("right", "left"):
            side_poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], self.env.num_envs, axis=0)
            for env_idx in range(self.env.num_envs):
                foot_pose = self._required_wbc_pose_for_frame(
                    (f"{side}_foot", f"{side}_{self.env.robot_config.foot_body_name}"),
                    env_idx,
                    f"{side} foot",
                )
                side_poses[env_idx] = foot_pose
            poses[side] = np.ascontiguousarray(side_poses)
        return poses

    def _batched_foot_linear_velocities(self) -> dict[str, np.ndarray]:
        velocities: dict[str, np.ndarray] = {}
        for side in ("right", "left"):
            side_velocities = np.zeros((self.env.num_envs, 3), dtype=np.float64)
            for env_idx in range(self.env.num_envs):
                side_velocities[env_idx] = self._required_wbc_linear_velocity_for_frame(
                    (f"{side}_foot", f"{side}_{self.env.robot_config.foot_body_name}"),
                    env_idx,
                    f"{side} foot",
                )
            velocities[side] = np.ascontiguousarray(side_velocities)
        return velocities

    def _batched_foot_angular_velocities(self) -> dict[str, np.ndarray]:
        velocities: dict[str, np.ndarray] = {}
        for side in ("right", "left"):
            side_velocities = np.zeros((self.env.num_envs, 3), dtype=np.float64)
            for env_idx in range(self.env.num_envs):
                side_velocities[env_idx] = self._required_wbc_angular_velocity_for_frame(
                    (f"{side}_foot", f"{side}_{self.env.robot_config.foot_body_name}"),
                    env_idx,
                    f"{side} foot",
                )
            velocities[side] = np.ascontiguousarray(side_velocities)
        return velocities

    def _update_wbc_robot_from_simulator(self) -> None:
        root_states = self._as_torch_tensor(
            self.env.simulator.robot_root_states,
            device=self.env.device,
            dtype=torch.float32,
            label="wbc_pose_root_states",
        )
        dof_pos = self._as_torch_tensor(
            self.env.simulator.dof_pos,
            device=self.env.device,
            dtype=torch.float32,
            label="wbc_pose_dof_pos",
        )
        dof_vel = self._as_torch_tensor(
            self.env.simulator.dof_vel,
            device=self.env.device,
            dtype=torch.float32,
            label="wbc_pose_dof_vel",
        )
        q = self._as_numpy_2d(torch.cat([root_states_to_xyz_rpy(root_states), dof_pos], dim=1), "wbc_pose_q")
        dq = self._as_numpy_2d(torch.cat([root_states[:, 7:13], dof_vel], dim=1), "wbc_pose_dq")
        self.wbc.update_robot(q, dq)

    def _required_wbc_pose_for_frame(
        self, frame_names: tuple[str, ...], env_idx: int, label: str
    ) -> np.ndarray:
        for frame_name in frame_names:
            pose = self._wbc_pose_for_frame(frame_name, env_idx)
            if pose is not None:
                return pose
        raise RuntimeError(
            f"Could not read {label} pose from humanoid_wbc.getPose for env {env_idx}; "
            f"tried frames={list(frame_names)}."
        )

    def _required_wbc_linear_velocity_for_frame(
        self, frame_names: tuple[str, ...], env_idx: int, label: str
    ) -> np.ndarray:
        for frame_name in frame_names:
            velocity = self._wbc_linear_velocity_for_frame(frame_name, env_idx)
            if velocity is not None:
                return velocity
        raise RuntimeError(
            f"Could not read {label} linear velocity from humanoid_wbc.getLinearVelocity for env {env_idx}; "
            f"tried frames={list(frame_names)}."
        )

    def _required_wbc_angular_velocity_for_frame(
        self, frame_names: tuple[str, ...], env_idx: int, label: str
    ) -> np.ndarray:
        for frame_name in frame_names:
            velocity = self._wbc_angular_velocity_for_frame(frame_name, env_idx)
            if velocity is not None:
                return velocity
        raise RuntimeError(
            f"Could not read {label} angular velocity from humanoid_wbc.getAngularVelocity for env {env_idx}; "
            f"tried frames={list(frame_names)}."
        )

    def _wbc_pose_for_frame(self, frame_name: str, env_idx: int) -> np.ndarray | None:
        try:
            pose = self.wbc.get_pose(frame_name, env_idx)
        except Exception:
            return None

        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4):
            raise RuntimeError(f"WBC getPose({frame_name!r}, env_idx={env_idx}) returned shape {pose.shape}, expected (4, 4).")
        if not np.all(np.isfinite(pose)):
            raise RuntimeError(f"WBC getPose({frame_name!r}, env_idx={env_idx}) returned non-finite pose: {pose}.")
        return pose

    def _wbc_linear_velocity_for_frame(self, frame_name: str, env_idx: int) -> np.ndarray | None:
        try:
            velocity = self.wbc.get_linear_velocity(frame_name, env_idx)
        except Exception:
            return None

        velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
        if velocity.shape != (3,):
            raise RuntimeError(
                f"WBC getLinearVelocity({frame_name!r}, env_idx={env_idx}) returned shape {velocity.shape}, expected (3,)."
            )
        if not np.all(np.isfinite(velocity)):
            raise RuntimeError(
                f"WBC getLinearVelocity({frame_name!r}, env_idx={env_idx}) returned non-finite velocity: {velocity}."
            )
        return velocity

    def _wbc_angular_velocity_for_frame(self, frame_name: str, env_idx: int) -> np.ndarray | None:
        try:
            velocity = self.wbc.get_angular_velocity(frame_name, env_idx)
        except Exception:
            return None

        velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
        if velocity.shape != (3,):
            raise RuntimeError(
                f"WBC getAngularVelocity({frame_name!r}, env_idx={env_idx}) returned shape {velocity.shape}, expected (3,)."
            )
        if not np.all(np.isfinite(velocity)):
            raise RuntimeError(
                f"WBC getAngularVelocity({frame_name!r}, env_idx={env_idx}) returned non-finite velocity: {velocity}."
            )
        return velocity

    def _contact_bases_as_numpy(self, value: Any, label: str) -> np.ndarray:
        tensor = self._as_torch_tensor(value, device=self.env.device, dtype=torch.float32, label=label)
        if tensor.dim() == 2 and tensor.shape[-1] == 3:
            tensor = torch.diag_embed(tensor)
        if tensor.dim() != 3 or tensor.shape[-2:] != (3, 3):
            raise RuntimeError(f"{label} must have shape [num_envs, 3] or [num_envs, 3, 3], got {tuple(tensor.shape)}.")
        return np.ascontiguousarray(tensor.cpu().numpy().astype(np.float64, copy=False))

    def _stance_support_contact_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        simulator = self.env.simulator
        prefix = "" if self._filter_stance_support_contacts else "raw_"
        right_position_attr = f"right_foot_{prefix}contact_position"
        left_position_attr = f"left_foot_{prefix}contact_position"
        right_basis_attr = f"right_foot_{prefix}contact_basis"
        left_basis_attr = f"left_foot_{prefix}contact_basis"

        if not self._filter_stance_support_contacts:
            missing = [
                attr
                for attr in (right_position_attr, left_position_attr, right_basis_attr, left_basis_attr)
                if not hasattr(simulator, attr)
            ]
            if missing:
                right_position_attr = "right_foot_contact_position"
                left_position_attr = "left_foot_contact_position"
                right_basis_attr = "right_foot_contact_basis"
                left_basis_attr = "left_foot_contact_basis"

        right_contact_points = self._as_numpy_2d(getattr(simulator, right_position_attr), right_position_attr)
        left_contact_points = self._as_numpy_2d(getattr(simulator, left_position_attr), left_position_attr)
        right_contact_bases = self._contact_bases_as_numpy(getattr(simulator, right_basis_attr), right_basis_attr)
        left_contact_bases = self._contact_bases_as_numpy(getattr(simulator, left_basis_attr), left_basis_attr)
        return right_contact_points, left_contact_points, right_contact_bases, left_contact_bases

    def _batched_foot_ground_reaction_wrenches(self, side: str) -> np.ndarray:
        grfs = np.stack(
            [self._foot_ground_reaction_wrench(env_idx, side) for env_idx in range(self.env.num_envs)],
            axis=0,
        ).astype(np.float64, copy=False)
        force_norms = np.linalg.norm(grfs[:, :3], axis=1)
        grfs[force_norms < self._dual_contact_force_threshold] = 0.0
        return np.ascontiguousarray(grfs)
    
    def _batched_local_foot_ground_reaction_wrenches(self, side: str) -> np.ndarray:
        grfs = np.stack(
            [self._local_foot_force_sensor_wrench(env_idx, side) for env_idx in range(self.env.num_envs)],
            axis=0,
        ).astype(np.float64, copy=False)
        force_norms = np.linalg.norm(grfs[:, :3], axis=1)
        grfs[force_norms < self._dual_contact_force_threshold] = 0.0
        return np.ascontiguousarray(grfs)

    def _actuated_torques_from_batched_wbc_output(self, torque_wbc: Any) -> np.ndarray:
        """Normalize batched WBC torque output to simulator actuated DOF torques."""
        torque_np = np.asarray(torque_wbc, dtype=float)
        if torque_np.ndim != 2:
            raise RuntimeError(f"Unexpected batched WBC torque rank: got shape {torque_np.shape}.")
        if torque_np.shape[1] == self._torque_dim:
            return torque_np
        if torque_np.shape[1] == self._torque_dim + 6:
            return torque_np[:, 6:]
        raise RuntimeError(
            "Unexpected batched WBC torque output size: "
            f"got shape {torque_np.shape}, expected {self._torque_dim} actuated torques "
            f"or {self._torque_dim + 6} floating-base torques per environment."
        )

    def _actuated_torques_from_wbc_output(self, torque_wbc: Any) -> np.ndarray:
        """Normalize WBC torque output to simulator actuated DOF torques."""
        torque_np = np.asarray(torque_wbc, dtype=float).reshape(-1)
        if torque_np.shape[0] == self._torque_dim:
            return torque_np
        if torque_np.shape[0] == self._torque_dim + 6:
            return torque_np[6:]
        raise RuntimeError(
            "Unexpected WBC torque output size: "
            f"got {torque_np.shape[0]}, expected {self._torque_dim} actuated torques "
            f"or {self._torque_dim + 6} floating-base torques."
        )

    def _maybe_draw_action_targets(self, root_states: torch.Tensor, action_batch: np.ndarray) -> None:
        if not self._visualize_action_targets:
            return

        simulator = self.env.simulator
        if not hasattr(simulator, "draw_sphere"):
            return

        for env_idx in self._contact_visualization_env_indices():
            if env_idx < 0 or env_idx >= self.env.num_envs or env_idx >= action_batch.shape[0]:
                continue
            self._draw_action_targets_for_env(env_idx, root_states[env_idx], action_batch[env_idx])

    def _draw_action_targets_for_env(self, env_idx: int, root_state: torch.Tensor, action_row: np.ndarray) -> None:
        action_row = np.asarray(action_row, dtype=float).reshape(-1)
        if action_row.shape[0] < 72:
            return

        for side, pos_slice, ori_slice, color, pos_id_base in (
            ("right", slice(24, 27), slice(27, 36), [1.0, 0.35, 0.05], 330),
            ("left", slice(42, 45), slice(45, 54), [0.25, 0.45, 1.0], 340),
        ):
            foot_target_pos = action_row[pos_slice]
            foot_target_rot = action_row[ori_slice].reshape(3, 3)
            if np.linalg.norm(foot_target_pos) < 1.0e-12 and np.linalg.norm(foot_target_rot) < 1.0e-12:
                continue
            self._draw_target_pose(env_idx, foot_target_pos, foot_target_rot, color, pos_id_base)

    def _maybe_draw_swing_foot_plan(
        self,
        current_poses: dict[str, np.ndarray],
        landing_poses: dict[int, tuple[str, np.ndarray]],
    ) -> None:
        if not (self._visualize_landing_foot_pose or self._visualize_swing_foot_trajectory):
            return

        simulator = self.env.simulator
        if not hasattr(simulator, "draw_sphere"):
            return

        visible_envs = set(self._contact_visualization_env_indices())
        for env_idx, (swing_side, landing_pose) in landing_poses.items():
            if env_idx not in visible_envs or env_idx < 0 or env_idx >= self.env.num_envs:
                continue
            if swing_side not in current_poses:
                continue

            landing_pose = np.asarray(landing_pose, dtype=float)
            current_pose = np.asarray(current_poses[swing_side][env_idx], dtype=float)
            if landing_pose.shape != (4, 4) or current_pose.shape != (4, 4):
                continue
            if not np.all(np.isfinite(landing_pose)) or not np.all(np.isfinite(current_pose)):
                continue

            if self._visualize_landing_foot_pose:
                color = [1.0, 0.85, 0.05] if swing_side == "right" else [0.15, 0.95, 1.0]
                pos_id_base = 360 if swing_side == "right" else 380
                self._draw_target_pose(env_idx, landing_pose[:3, 3], landing_pose[:3, :3], color, pos_id_base)
                self._draw_landing_foot_rectangle(env_idx, landing_pose, color)

            if self._visualize_swing_foot_trajectory:
                self._draw_swing_foot_trajectory(env_idx, swing_side, current_pose, landing_pose)

    def _draw_landing_foot_rectangle(self, env_idx: int, landing_pose: np.ndarray, color: list[float]) -> None:
        simulator = self.env.simulator
        if not hasattr(simulator, "draw_line"):
            return

        foot_dimension = np.asarray(getattr(self.env.robot_config, "foot_dimension", [0.18, 0.10]), dtype=float)
        foot_center = np.asarray(getattr(self.env.robot_config, "foot_center", [0.0, 0.0, 0.0]), dtype=float)
        if foot_dimension.shape[0] < 2 or not np.all(np.isfinite(foot_dimension[:2])):
            return
        if foot_center.shape[0] < 3 or not np.all(np.isfinite(foot_center[:3])):
            return

        half_length = 0.5 * float(foot_dimension[0])
        half_width = 0.5 * float(foot_dimension[1])
        center = foot_center[:3]
        local_corners = np.array(
            [
                [center[0] + half_length, center[1] + half_width, center[2]],
                [center[0] + half_length, center[1] - half_width, center[2]],
                [center[0] - half_length, center[1] - half_width, center[2]],
                [center[0] - half_length, center[1] + half_width, center[2]],
            ],
            dtype=float,
        )
        position = landing_pose[:3, 3]
        rotation = landing_pose[:3, :3]
        world_corners = position.reshape(1, 3) + local_corners @ rotation.T
        if not np.all(np.isfinite(world_corners)):
            return

        corner_tensors = [torch.as_tensor(corner, dtype=torch.float32).cpu() for corner in world_corners]
        for corner_idx, start in enumerate(corner_tensors):
            end = corner_tensors[(corner_idx + 1) % len(corner_tensors)]
            simulator.draw_line(start, end, color, env_id=env_idx)

    def _draw_swing_foot_trajectory(
        self,
        env_idx: int,
        swing_side: str,
        current_pose: np.ndarray,
        landing_pose: np.ndarray,
    ) -> None:
        simulator = self.env.simulator
        samples = max(int(self._swing_trajectory_samples), 2)
        start = np.asarray(current_pose[:3, 3], dtype=float)
        end = np.asarray(landing_pose[:3, 3], dtype=float)
        fallback_clearance = max(float(self._swing_foot_takeoff_clearance), float(self._swing_foot_landing_clearance))
        midpoint_height = self._swing_foot_midpoint_height
        color = [1.0, 0.55, 0.05] if swing_side == "right" else [0.1, 0.7, 1.0]
        pos_id_base = 400 if swing_side == "right" else 440

        previous_point: torch.Tensor | None = None
        for sample_idx in range(samples + 1):
            phase = sample_idx / float(samples)
            point_np = (1.0 - phase) * start + phase * end
            apex_z = float(midpoint_height) if midpoint_height is not None else end[2] + fallback_clearance
            point_np[2] += max(0.0, apex_z - point_np[2]) * math.sin(math.pi * phase)
            if not np.all(np.isfinite(point_np)):
                continue

            point = torch.as_tensor(point_np, dtype=torch.float32).cpu()
            if 0 < sample_idx < samples:
                simulator.draw_sphere(
                    point,
                    0.5 * self._action_target_radius,
                    color,
                    env_id=env_idx,
                    pos_id=pos_id_base + sample_idx,
                )
            if previous_point is not None and hasattr(simulator, "draw_line"):
                simulator.draw_line(previous_point, point, color, env_id=env_idx)
            previous_point = point

    def _current_com_position(self, env_idx: int, root_state: torch.Tensor) -> np.ndarray:
        com_pos = getattr(self.env.simulator, "com_pos", None)
        if isinstance(com_pos, torch.Tensor) and com_pos.ndim == 2 and env_idx < com_pos.shape[0]:
            return com_pos[env_idx].detach().cpu().numpy()
        return root_state[:3].detach().cpu().numpy()

    def _body_pose_for_target(self, env_idx: int, body_idx: Any | None) -> tuple[np.ndarray, np.ndarray] | None:
        if body_idx is None:
            return None

        simulator = self.env.simulator
        rigid_body_idx = int(body_idx)
        mujoco_body_map = getattr(simulator, "holosoma_to_mujoco_body_map", None)
        if isinstance(mujoco_body_map, dict):
            rigid_body_idx = int(mujoco_body_map.get(rigid_body_idx, rigid_body_idx))

        if (
            not hasattr(simulator, "_rigid_body_pos")
            or not hasattr(simulator, "_rigid_body_rot")
            or env_idx >= simulator._rigid_body_pos.shape[0]
            or rigid_body_idx >= simulator._rigid_body_pos.shape[1]
        ):
            return None

        pos = simulator._rigid_body_pos[env_idx, rigid_body_idx].detach().cpu().numpy()
        quat = simulator._rigid_body_rot[env_idx, rigid_body_idx]
        rot = quaternion_to_matrix(quat.unsqueeze(0), w_last=True)[0].detach().cpu().numpy()
        return pos, rot

    def _draw_target_pose(
        self,
        env_idx: int,
        position: np.ndarray,
        rotation: np.ndarray | None,
        color: list[float],
        pos_id_base: int,
    ) -> None:
        if not np.all(np.isfinite(position)):
            return

        simulator = self.env.simulator
        point = torch.as_tensor(position[:3], dtype=torch.float32).cpu()
        simulator.draw_sphere(point, self._action_target_radius, color, env_id=env_idx, pos_id=pos_id_base)

        if rotation is None or not self._visualize_action_target_frames or not hasattr(simulator, "draw_line"):
            return
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            return

        axis_colors = ([1.0, 0.0, 0.0], [0.0, 0.85, 0.0], [0.1, 0.25, 1.0])
        for axis_idx, axis_color in enumerate(axis_colors):
            axis = rotation[:, axis_idx]
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm < 1.0e-9:
                continue
            end = torch.as_tensor(
                position[:3] + self._action_target_axis_scale * axis / axis_norm,
                dtype=torch.float32,
            ).cpu()
            simulator.draw_line(point, end, axis_color, env_id=env_idx)

    def _maybe_draw_contact_points(
        self,
        right_contact_points: np.ndarray,
        right_contact_bases: np.ndarray,
        left_contact_points: np.ndarray,
        left_contact_bases: np.ndarray,
    ) -> None:
        if not self._visualize_contact_points:
            return

        simulator = self.env.simulator
        if not hasattr(simulator, "draw_sphere"):
            return

        env_indices = self._contact_visualization_env_indices()
        for env_idx in env_indices:
            if env_idx < 0 or env_idx >= self.env.num_envs:
                continue
            self._draw_stance_support(
                env_idx,
                right_contact_points[env_idx],
                right_contact_bases[env_idx],
                "right",
                left_contact_points[env_idx],
                left_contact_bases[env_idx],
                "left",
            )

    def _contact_visualization_env_indices(self) -> list[int]:
        simulator = self.env.simulator
        current_world_id = getattr(simulator, "current_world_id", None)
        if current_world_id is not None:
            return [int(current_world_id)]
        return list(range(self.env.num_envs))

    def _draw_stance_support(
        self,
        env_idx: int,
        right_contact_point: np.ndarray,
        right_contact_basis: np.ndarray,
        right_side: str,
        left_contact_point: np.ndarray,
        left_contact_basis: np.ndarray,
        left_side: str,
    ) -> None:
        simulator = self.env.simulator
        if not hasattr(simulator, "draw_sphere"):
            return

        self._draw_contact_support_frame(
            env_idx,
            np.asarray(right_contact_point, dtype=float),
            np.asarray(right_contact_basis, dtype=float),
            right_side,
            point_color=[1.0, 0.25, 0.1],
            pos_id_base=100,
        )
        self._draw_contact_support_frame(
            env_idx,
            np.asarray(left_contact_point, dtype=float),
            np.asarray(left_contact_basis, dtype=float),
            left_side,
            point_color=[0.2, 0.45, 1.0],
            pos_id_base=200,
        )

    def _draw_contact_support_frame(
        self,
        env_idx: int,
        contact_point: np.ndarray,
        contact_basis: np.ndarray,
        side: str,
        point_color: list[float],
        pos_id_base: int,
    ) -> None:
        if contact_point.shape[0] < 3 or contact_basis.shape != (3, 3):
            return
        if not np.all(np.isfinite(contact_point)) or not np.all(np.isfinite(contact_basis)):
            return
        counts = getattr(self.env.simulator, f"{side}_foot_contact_count", None)
        if counts is not None:
            count_tensor = self._as_torch_tensor(
                counts,
                device=self.env.device,
                dtype=torch.long,
                label=f"{side}_foot_contact_count",
            ).reshape(-1)
            if env_idx >= count_tensor.shape[0] or int(count_tensor[env_idx].detach().cpu().item()) <= 0:
                return

        simulator = self.env.simulator
        foot_body_idx = self._foot_body_indices[side]
        mujoco_body_map = getattr(simulator, "holosoma_to_mujoco_body_map", None)
        rigid_body_idx = foot_body_idx
        if isinstance(mujoco_body_map, dict):
            rigid_body_idx = mujoco_body_map.get(foot_body_idx, foot_body_idx)

        foot_pos_w = simulator._rigid_body_pos[env_idx, rigid_body_idx]
        foot_quat_w = simulator._rigid_body_rot[env_idx, rigid_body_idx]
        if self._assert_contact_visualization_pose:
            self._assert_foot_pose_matches_wbc(env_idx, side, foot_pos_w, foot_quat_w, contact_point)

        local_contact_point = torch.as_tensor(contact_point[:3], device=foot_pos_w.device, dtype=foot_pos_w.dtype)
        world_contact_point_tensor = foot_pos_w + quat_apply(foot_quat_w, local_contact_point, w_last=True)
        world_contact_point = world_contact_point_tensor.detach().cpu().numpy()

        point = torch.as_tensor(world_contact_point, dtype=torch.float32).cpu()
        simulator.draw_sphere(point, self._contact_point_radius, point_color, env_id=env_idx, pos_id=pos_id_base)

        if not self._visualize_contact_frames or not hasattr(simulator, "draw_line"):
            return

        axis_colors = ([1.0, 0.0, 0.0], [0.0, 0.85, 0.0], [0.1, 0.25, 1.0])
        axis_scale = 0.12
        for axis_idx, axis_color in enumerate(axis_colors):
            local_axis = torch.as_tensor(
                contact_basis[:, axis_idx], device=foot_quat_w.device, dtype=foot_quat_w.dtype
            )
            axis = quat_apply(foot_quat_w, local_axis, w_last=True).detach().cpu().numpy()
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm < 1.0e-9:
                continue
            end = torch.as_tensor(world_contact_point + axis_scale * axis / axis_norm, dtype=torch.float32).cpu()
            simulator.draw_line(point, end, axis_color, env_id=env_idx)

    def _assert_foot_pose_matches_wbc(
        self,
        env_idx: int,
        side: str,
        foot_pos_w: torch.Tensor,
        foot_quat_w: torch.Tensor,
        local_contact_point: np.ndarray
    ) -> None:
        # foot_name = f"{side}_foot"
        foot_name = f"{side}_ankle_roll_link"
        wbc_engine = self._wbc_debug_engine
        wbc_pos = np.asarray(wbc_engine.getPosition(foot_name, local_contact_point), dtype=float).reshape(-1)[:3]
        wbc_rot = np.asarray(wbc_engine.getRotation(foot_name), dtype=float)
        if wbc_pos.shape[0] < 3 or not np.all(np.isfinite(wbc_pos)):
            raise AssertionError(f"WBC {foot_name} position is invalid: shape={wbc_pos.shape}, pos={wbc_pos}")
        if wbc_rot.shape != (3, 3) or not np.all(np.isfinite(wbc_rot)):
            raise AssertionError(f"WBC {foot_name} rotation is invalid: shape={wbc_rot.shape}, rotation={wbc_rot}")

        sim_rot = quaternion_to_matrix(foot_quat_w.unsqueeze(0), w_last=True)[0].detach().cpu().numpy()
        sim_pos = foot_pos_w.detach().cpu().numpy() + sim_rot @ local_contact_point
        self._print_foot_frame_inspection(env_idx, side, sim_pos, sim_rot, wbc_pos, wbc_rot)

        # root_state = self.env.simulator.robot_root_states[env_idx]
        # dof_pos = self.env.simulator.dof_pos[env_idx]
        # q_tensor = torch.cat([root_state_to_xyz_rpy(root_state), dof_pos], dim=0)
        # q_error = float(torch.linalg.vector_norm(q_tensor - self._last_wbc_q[env_idx]).item())
        # root_state_error = float(
        #     torch.linalg.vector_norm(root_state - self._last_wbc_root_state[env_idx]).item()
        # )
        # dof_pos_error = float(torch.linalg.vector_norm(dof_pos - self._last_wbc_dof_pos[env_idx]).item())
        # pelvis_pos_error, pelvis_rot_error, pelvis_pos, wbc_pelvis_pos, pelvis_pos_delta = (
        #     self._assert_pelvis_pose_matches_wbc(env_idx)
        # )
        pos_delta = sim_pos - wbc_pos
        pos_error = float(np.linalg.norm(sim_pos - wbc_pos))
        rot_error = float(np.linalg.norm(sim_rot - wbc_rot, ord="fro"))
        assert (
            # q_error < 1.0e-6
            # and root_state_error < 1.0e-6
            # and dof_pos_error < 1.0e-6
            # and pelvis_pos_error < 1.0e-6
            # and pelvis_rot_error < 1.0e-6
            pos_error < 1.0e-6
            and rot_error < 1.0e-6
        ), (
            f"Simulator {side} foot pose does not match WBC {side}_foot pose: "
            # f"q_error={q_error:.9f}, root_state_error={root_state_error:.9f}, "
            # f"dof_pos_error={dof_pos_error:.9f}, pelvis_pos_error={pelvis_pos_error:.9f}, "
            # f"pelvis_rot_error={pelvis_rot_error:.9f}, pos_error={pos_error:.9f}, rot_error={rot_error:.9f}, "
            # f"pelvis_sim_pos_xyz={pelvis_pos.tolist()}, pelvis_wbc_pos_xyz={wbc_pelvis_pos.tolist()}, "
            # f"pelvis_pos_delta_xyz={pelvis_pos_delta.tolist()}, "
            f"sim_pos_xyz={sim_pos.tolist()}, wbc_pos_xyz={wbc_pos.tolist()}, pos_delta_xyz={pos_delta.tolist()}, rot_delta={rot_error}"
            "\n"
        )

    def _assert_pelvis_pose_matches_wbc(
        self,
        env_idx: int,
    ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        wbc_pelvis_pos = np.asarray(self._wbc_debug_engine.getPosition("pelvis"), dtype=float).reshape(-1)
        wbc_pelvis_rot = np.asarray(self._wbc_debug_engine.getRotation("pelvis"), dtype=float)
        if wbc_pelvis_pos.shape[0] < 3 or not np.all(np.isfinite(wbc_pelvis_pos)):
            raise AssertionError(f"WBC pelvis position is invalid: shape={wbc_pelvis_pos.shape}, pos={wbc_pelvis_pos}")
        if wbc_pelvis_rot.shape != (3, 3) or not np.all(np.isfinite(wbc_pelvis_rot)):
            raise AssertionError(
                f"WBC pelvis rotation is invalid: shape={wbc_pelvis_rot.shape}, rotation={wbc_pelvis_rot}"
            )

        root_state = self.env.simulator.robot_root_states[env_idx]
        pelvis_pos = root_state[:3].detach().cpu().numpy()
        pelvis_rot = quaternion_to_matrix(root_state[3:7].unsqueeze(0), w_last=True)[0].detach().cpu().numpy()
        wbc_pelvis_pos = wbc_pelvis_pos[:3]
        pelvis_pos_delta = pelvis_pos - wbc_pelvis_pos
        pelvis_pos_error = float(np.linalg.norm(pelvis_pos_delta))
        pelvis_rot_error = float(np.linalg.norm(pelvis_rot - wbc_pelvis_rot, ord="fro"))
        return pelvis_pos_error, pelvis_rot_error, pelvis_pos, wbc_pelvis_pos, pelvis_pos_delta

    def _print_foot_frame_inspection(
        self,
        env_idx: int,
        side: str,
        resolved_sim_pos: np.ndarray,
        resolved_sim_rot: np.ndarray,
        wbc_compare_pos: np.ndarray,
        wbc_compare_rot: np.ndarray,
    ) -> None:
        simulator = self.env.simulator
        resolved_idx = self._foot_body_indices[side]
        body_names = list(getattr(simulator, "body_names", []))
        resolved_name = body_names[resolved_idx] if 0 <= resolved_idx < len(body_names) else "<out-of-range>"
        resolved_rot_error = float(np.linalg.norm(resolved_sim_rot - wbc_compare_rot, ord="fro"))

        print(
            "foot_frame_inspection:"
            f" side={side}"
            f" resolved_idx={resolved_idx}"
            f" resolved_name={resolved_name}"
            f" robot_config.foot_body_name={self.env.robot_config.foot_body_name}"
            f" resolved_sim_pos={resolved_sim_pos.tolist()}"
            f" wbc_compare_pos={wbc_compare_pos.tolist()}"
            f" delta={list((resolved_sim_pos - wbc_compare_pos).tolist())}"
            f" rot_error={resolved_rot_error:.9f}"
            "\n"
        )

        # # for suffix in ("ankle_pitch_link", "ankle_roll_link", "foot"):
        # for suffix in ("ankle_roll_link"):
        #     frame_name = f"{side}_{suffix}"
        #     sim_matches = [(idx, name) for idx, name in enumerate(body_names) if name == frame_name]
        #     wbc_rot = None
        #     for sim_idx, sim_name in sim_matches:
        #         sim_pos = simulator._rigid_body_pos[env_idx, sim_idx].detach().cpu().numpy()
        #         sim_quat = simulator._rigid_body_rot[env_idx, sim_idx]
        #         sim_rot = quaternion_to_matrix(sim_quat.unsqueeze(0), w_last=True)[0].detach().cpu().numpy()
        #         try:
        #             if wbc_rot is None:
        #                 wbc_rot = np.asarray(self.wbc[env_idx].getRotation(frame_name), dtype=float)
        #             rot_error = float(np.linalg.norm(sim_rot - wbc_rot, ord="fro"))
        #             print(f"  frame_rotation_error name={sim_name} idx={sim_idx} rot_error={rot_error:.9f}")
        #         except Exception as exc:
        #             print(f"  frame_rotation_error name={sim_name} idx={sim_idx} unavailable={exc}")

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset action term state.

        Args:
            env_ids: Environment IDs to reset. If None, reset all.
        """
        super().reset(env_ids)

        # Reset action delay queue if applicable
        if getattr(self.env, "_randomize_ctrl_delay", False) and self.action_queue is not None:
            if env_ids is None:
                self.action_queue.zero_()
            else:
                self.action_queue[env_ids] = 0.0

        # Reset torques
        if env_ids is None:
            self.torques.zero_()
            self._non_finite_torque_mask.zero_()
            self._last_wbc_landing_plan_inputs.fill_(float("nan"))
            self._wbc_nominal_landing_swing_side.fill_(-1)
            self._wbc_landing_target_swing_side.fill_(-1)
            self._wbc_landing_target_initial_duration.zero_()
            self._wbc_landing_target_frozen.zero_()
            self._last_wbc_landing_target_update_fraction.zero_()
        else:
            self.torques[env_ids] = 0.0
            self._non_finite_torque_mask[env_ids] = False
            self._last_wbc_landing_plan_inputs[env_ids] = float("nan")
            self._wbc_nominal_landing_swing_side[env_ids] = -1
            self._wbc_landing_target_swing_side[env_ids] = -1
            self._wbc_landing_target_initial_duration[env_ids] = 0.0
            self._wbc_landing_target_frozen[env_ids] = False
            self._last_wbc_landing_target_update_fraction[env_ids] = 0.0

        # Reset cached velocities
        if env_ids is None:
            self._prev_dof_vel.zero_()
            self._wbc_replan_swing_trajectory.fill_(True)
        else:
            self._prev_dof_vel[env_ids] = 0.0
            self._wbc_replan_swing_trajectory[env_ids] = True

        if self._wbc_bootstrap_enabled:
            self._capture_bootstrap_hold_targets(env_ids)
            self._reset_wbc_integrated_targets(env_ids)
            if env_ids is None:
                self._wbc_bootstrap_done.zero_()
                self._dual_contact_counter.zero_()
                self._startup_gait_counter.zero_()
                self._startup_desired_state.fill_(int(self.State.DUAL_STANCE))
                self._wbc_phase_shift_frac.zero_()
                self._last_wbc_bootstrap_hold_mask.fill_(True)
                self._last_wbc_dual_contact_mask.zero_()
            else:
                self._wbc_bootstrap_done[env_ids] = False
                self._dual_contact_counter[env_ids] = 0
                self._startup_gait_counter[env_ids] = 0
                self._startup_desired_state[env_ids] = int(self.State.DUAL_STANCE)
                self._wbc_phase_shift_frac[env_ids] = 0.0
                self._last_wbc_bootstrap_hold_mask[env_ids] = True
                self._last_wbc_dual_contact_mask[env_ids] = False
        else:
            self._reset_wbc_integrated_targets(env_ids)
            if env_ids is None:
                self._wbc_bootstrap_done.fill_(True)
                self._dual_contact_counter.zero_()
                self._startup_gait_counter.zero_()
                self._startup_desired_state.fill_(int(self.State.DUAL_STANCE))
                self._wbc_phase_shift_frac.zero_()
                self._last_wbc_bootstrap_hold_mask.zero_()
                self._last_wbc_dual_contact_mask.fill_(True)
            else:
                self._wbc_bootstrap_done[env_ids] = True
                self._dual_contact_counter[env_ids] = 0
                self._startup_gait_counter[env_ids] = 0
                self._startup_desired_state[env_ids] = int(self.State.DUAL_STANCE)
                self._wbc_phase_shift_frac[env_ids] = 0.0
                self._last_wbc_bootstrap_hold_mask[env_ids] = False
                self._last_wbc_dual_contact_mask[env_ids] = True

        for env_idx in (
            range(self.env.num_envs)
            if env_ids is None
            else env_ids.detach().cpu().flatten().tolist()
        ):
            self.curr_state[int(env_idx)] = self.State.DUAL_STANCE
            self.transition_start_time[int(env_idx)] = float(self.env.simulator.time())

        planner_env_ids = (
            range(self.env.num_envs)
            if env_ids is None
            else [int(env_idx) for env_idx in env_ids.detach().cpu().flatten().tolist()]
        )
        self._batched_foot_planner.reset(planner_env_ids)

        if env_ids is None:
            self._pending_wbc_reinitialize[:] = not self._wbc_bootstrap_enabled
        else:
            self._pending_wbc_reinitialize[env_ids] = not self._wbc_bootstrap_enabled

    # ------------------------------------------------------------------
    # Hooks for randomization manager

    def attach_actuator_scales(
        self, kp_scale: torch.Tensor, kd_scale: torch.Tensor, rfi_lim_scale: torch.Tensor
    ) -> None:
        """Attach shared actuator scaling tensors provided by the randomization manager."""
        self._kp_scale = kp_scale
        self._kd_scale = kd_scale
        self._rfi_lim_scale = rfi_lim_scale

    def update_pd_scales(self, env_ids: torch.Tensor, kp_values: torch.Tensor, kd_values: torch.Tensor) -> None:
        """Fallback PD-scale update when no shared buffers are registered."""
        self._kp_scale[env_ids] = kp_values
        self._kd_scale[env_ids] = kd_values

    def update_rfi_scales(self, env_ids: torch.Tensor, rfi_values: torch.Tensor) -> None:
        """Fallback RFI-scale update when no shared buffers are registered."""
        self._rfi_lim_scale[env_ids] = rfi_values

    def configure_torque_rfi(self, *, enabled: bool, rfi_lim: float | None = None) -> None:
        """Configure residual force injection behaviour."""
        self._randomize_torque_rfi = enabled
        if rfi_lim is not None:
            self._rfi_lim = float(rfi_lim)

    def get_pd_scale_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return references to the PD gain scale buffers."""
        return self._kp_scale, self._kd_scale

    def get_rfi_scale_tensor(self) -> torch.Tensor:
        """Return reference to the RFI limit scale buffer."""
        return self._rfi_lim_scale

    def get_prev_dof_vel(self) -> torch.Tensor:
        """Return cached previous DOF velocities."""
        return self._prev_dof_vel

    # ------------------------------------------------------------------
    # Internal helpers

    def _resolve_foot_body_indices(self, env: Any) -> dict[str, int]:
        """Resolve simulator body indices used to read per-foot ground reaction forces."""
        foot_indices: dict[str, int] = {}
        for side in ("right", "left"):
            exact_name = f"{side}_{env.robot_config.foot_body_name}"
            match = next((idx for idx, name in enumerate(env.simulator.body_names) if name == exact_name), None)
            if match is None:
                match = next(
                    (
                        idx
                        for idx, name in enumerate(env.simulator.body_names)
                        if side in name.lower() and name.endswith(env.robot_config.foot_body_name)
                    ),
                    None,
                )
            if match is None:
                match = next(
                    (
                        idx
                        for idx, name in enumerate(env.simulator.body_names)
                        if side in name.lower() and env.robot_config.foot_body_name in name
                    ),
                    None,
                )
            if match is None:
                raise ValueError(
                    f"Could not resolve the {side} foot body using "
                    f"foot_body_name='{env.robot_config.foot_body_name}'."
                )
            foot_indices[side] = match
        return foot_indices

    def _resolve_body_index_by_name(self, env: Any, body_name: str | None) -> int | None:
        if not body_name:
            return None
        body_names = list(getattr(env.simulator, "body_names", []))
        match = next((idx for idx, name in enumerate(body_names) if name == body_name), None)
        if match is not None:
            return int(match)
        match = next((idx for idx, name in enumerate(body_names) if name.endswith(body_name)), None)
        if match is not None:
            return int(match)
        match = next((idx for idx, name in enumerate(body_names) if body_name in name), None)
        return int(match) if match is not None else None

    def _foot_ground_reaction_wrench(self, env_idx: int, side: str) -> np.ndarray:
        """Return the simulator foot contact force as a 6D wrench for the WBC binding."""
        sensor_wrench_fn = getattr(self.env.simulator, "get_foot_force_sensor_wrench", None)
        if callable(sensor_wrench_fn):
            try:
                return sensor_wrench_fn(side, env_idx).detach().cpu().numpy().astype(float, copy=False)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to read foot force/torque sensor wrench via "
                    f"env.simulator.get_foot_force_sensor_wrench(side={side!r}, env_idx={env_idx}). "
                    "Refusing to fall back to env.simulator.contact_forces because the WBC expects the "
                    "XML foot force sensor wrench in [force; torque] order."
                ) from exc

        raise RuntimeError(
            "env.simulator does not provide get_foot_force_sensor_wrench(). "
            "Refusing to fall back to env.simulator.contact_forces because the WBC expects the XML foot "
            "force sensor wrench in [force; torque] order."
        )

    def _local_foot_force_sensor_wrench(self, env_idx: int, side: str) -> np.ndarray:
        """Return the simulator foot force sensor wrench expressed in the foot sensor frame."""
        sensor_wrench_fn = getattr(self.env.simulator, "get_local_foot_force_sensor_wrench", None)
        if callable(sensor_wrench_fn):
            try:
                return sensor_wrench_fn(side, env_idx).detach().cpu().numpy().astype(float, copy=False)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to read local foot force/torque sensor wrench via "
                    f"env.simulator.get_local_foot_force_sensor_wrench(side={side!r}, env_idx={env_idx})."
                ) from exc

        raise RuntimeError(
            "env.simulator does not provide get_local_foot_force_sensor_wrench(). "
            "The WBC contact flags require the XML foot force sensor wrench expressed in the local frame."
        )

    def _attach_actuator_randomizer_scales(self) -> None:
        """Attach shared actuator randomizer buffers if they exist."""
        rand_manager = getattr(self.env, "randomization_manager", None)
        if rand_manager is None:
            return

        get_state = getattr(rand_manager, "get_state", None)
        if not callable(get_state):
            return

        state = get_state("actuator_randomizer_state")
        if state is None:
            return

        self.attach_actuator_scales(state.kp_scale_tensor, state.kd_scale_tensor, state.rfi_lim_scale_tensor)

    def _configure_pd_gains(self, env: Any) -> None:
        control_cfg = env.robot_config.control
        stiffness_cfg = control_cfg.stiffness
        damping_cfg = control_cfg.damping
        integral_cfg = getattr(control_cfg, "integral", {})

        for i, name in enumerate(env.dof_names):
            if name not in env.robot_config.init_state.default_joint_angles:
                raise ValueError(f"Missing default joint angle for DOF '{name}' in robot configuration.")

            matched = False
            for dof_name, stiffness in stiffness_cfg.items():
                if dof_name in name:
                    self.p_gains[i] = stiffness
                    self.d_gains[i] = damping_cfg[dof_name]
                    self.i_gains[i] = integral_cfg.get(dof_name, 0.0)
                    matched = True
            if not matched:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                self.i_gains[i] = 0.0
                if control_cfg.control_type in ["P", "V"]:
                    raise ValueError(
                        f"PD gains for joint '{name}' were not defined. Please specify them in the YAML configuration."
                    )

    def _configure_action_scales(self, env: Any) -> None:
        control_cfg = env.robot_config.control
        if control_cfg.action_scales_by_effort_limit_over_p_gain:
            if not isinstance(control_cfg.action_scale, (float, int)):
                raise ValueError("action_scales_by_effort_limit_over_p_gain requires scalar action_scale.")
            if self._action_dim != self._torque_dim:
                raise ValueError(
                    "action_scales_by_effort_limit_over_p_gain requires policy action dim to match num_dof. "
                    f"Got actions_dim={self._action_dim}, num_dof={self._torque_dim}."
                )
            dof_effort_limit_list = env.robot_config.dof_effort_limit_list
            for i, effort in enumerate(dof_effort_limit_list):
                stiffness = self.p_gains[i]
                if stiffness == 0.0:
                    self.action_scales[i] = 0.0
                else:
                    self.action_scales[i] = control_cfg.action_scale * effort / stiffness
        else:
            self.action_scales[:] = self._action_param_tensor(control_cfg.action_scale, "action_scale")

    def _configure_action_clip_values(self, env: Any) -> None:
        self.action_clip_values[:] = self._action_param_tensor(
            env.robot_config.control.action_clip_value, "action_clip_value"
        )

    def _action_param_tensor(self, value: float | list[float] | tuple[float, ...], label: str) -> torch.Tensor:
        if isinstance(value, (float, int)):
            return torch.full((self._action_dim,), float(value), device=self.env.device, dtype=torch.float)

        tensor = torch.as_tensor(value, device=self.env.device, dtype=torch.float)
        if tensor.ndim != 1 or tensor.shape[0] != self._action_dim:
            raise ValueError(
                f"robot_config.control.{label} must be a scalar or length-{self._action_dim} sequence, "
                f"got shape {tuple(tensor.shape)}."
            )
        return tensor

    # WBC import helpers
    def _repo_root(self) -> Path:
        path = Path(__file__).resolve()
        candidates = [path.parent, *path.parents]

        for candidate in candidates:
            if (candidate.parent / "humanoid-control").exists():
                return candidate

        for candidate in candidates:
            if (candidate / ".git").exists():
                return candidate

        pyproject_candidates = [candidate for candidate in candidates if (candidate / "pyproject.toml").exists()]
        for candidate in reversed(pyproject_candidates):
            if (candidate / "src" / "holosoma").exists():
                return candidate

        if pyproject_candidates:
            return pyproject_candidates[-1]

        return path.parents[6]

    def _humanoid_control_root(self) -> Path:
        return self._repo_root().parent / "humanoid-control"

    def _resolve_wbc_extension_dir(self, extension_dir: str | None = None) -> str | None:
        if extension_dir:
            return str(Path(extension_dir).expanduser().resolve())

        humanoid_control_build = self._humanoid_control_root() / "build"
        if humanoid_control_build.exists():
            return str(humanoid_control_build.resolve())

        return None

    def _resolve_wbc_params(
        self, env: Any, params: dict[str, Any], extension_dir: str | None = None
    ) -> dict[str, Any]:
        resolved_params = dict(params)
        robot_type = getattr(env.robot_config.asset, "robot_type", "")

        if not robot_type.startswith("g1"):
            return resolved_params

        robot_data_root = self._repo_root() / "src" / "holosoma" / "holosoma" / "data" / "robots" / "g1"
        humanoid_control_root = self._humanoid_control_root()

        resolved_params.setdefault("robot_file", str((humanoid_control_root / "models" / "unitree_g1" / "g1.urdf").resolve()))
        resolved_params.setdefault("yaml_file", str((humanoid_control_root / "params" / "g1_parameters.yaml").resolve()))
        resolved_params.setdefault("robot_name", "g1")
        return resolved_params

    def _extension_root(self, extension_dir: str | None = None) -> Path:
        if extension_dir:
            resolved = Path(extension_dir).expanduser().resolve()
            if resolved.name == "build":
                return resolved.parent
            return resolved
        humanoid_control_root = self._humanoid_control_root()
        if humanoid_control_root.exists():
            return humanoid_control_root
        return self._repo_root()

    def _try_add_local_extension_path(self, module_name: str, extension_dir: str | None = None) -> None:
        for candidate in self._find_wbc_extension_candidates(module_name, extension_dir):
            module_dir = str(candidate.parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            return

    def _find_wbc_extension_candidates(self, module_name: str, extension_dir: str | None = None) -> list[Path]:
        search_roots = []
        if extension_dir:
            search_roots.append(Path(extension_dir).expanduser().resolve())
        humanoid_control_root = self._humanoid_control_root()
        search_roots.extend(
            [
                humanoid_control_root / "build",
                humanoid_control_root,
                self._repo_root() / "build",
                self._repo_root(),
            ]
        )

        candidates: list[Path] = []
        for base in search_roots:
            if not base.exists():
                continue
            for pattern in (f"{module_name}*.so", f"{module_name}*.pyd", f"{module_name}*.dylib"):
                candidates.extend(base.rglob(pattern))
        return candidates


    def _import_wbc_module(self, extension_dir: str | None = None):
        module_name = "humanoid_wbc"
        self._try_add_local_extension_path(module_name, extension_dir)
        try:
            return importlib.import_module(module_name)
        except Exception as fallback_error:
            candidates = self._find_wbc_extension_candidates(module_name, extension_dir)
            supported_suffixes = importlib.machinery.EXTENSION_SUFFIXES
            candidate_text = ", ".join(str(candidate) for candidate in candidates) or "none"
            raise ModuleNotFoundError(
                "Could not import humanoid_wbc. "
                f"Searched extension_dir={extension_dir!r}; found candidates: {candidate_text}. "
                f"This Python accepts extension suffixes: {supported_suffixes}. "
                "If a candidate is tagged for another Python version, rebuild humanoid-control "
                "inside the active Python environment."
            ) from fallback_error

    def _resolve_wbc_asset_paths(
        self, extension_dir: str | None, params: dict[str, Any]
    ) -> tuple[Path, Path, str]:
        extension_root = self._extension_root(extension_dir)
        robot_file = Path(params.get("robot_file", extension_root / "models" / "unitree_g1" / "g1.urdf"))
        yaml_file = Path(params.get("yaml_file", extension_root / "params" / "hrp4c_parameters.yaml"))
        robot_name = params.get("robot_name", "hrp4c")

        if not robot_file.exists() or not yaml_file.exists():
            raise FileNotFoundError(
                "Whole-body controller assets are missing. "
                f"robot_file={robot_file}, yaml_file={yaml_file}"
            )

        return robot_file, yaml_file, robot_name

    def _create_batched_wbc_controller(
        self,
        extension_dir: str | None,
        params: dict[str, Any],
        num_envs: int,
        wbc_module: Any | None = None,
    ):
        if wbc_module is None:
            wbc_module = self._import_wbc_module(extension_dir)
        if not hasattr(wbc_module, "BatchedWbcController"):
            raise RuntimeError(
                "humanoid_wbc.BatchedWbcController is required for batched torque control, "
                "but the imported humanoid_wbc module does not expose it. Rebuild humanoid-control "
                "so the Python extension includes the batched controller bindings."
            )

        robot_file, yaml_file, robot_name = self._resolve_wbc_asset_paths(extension_dir, params)
        return wbc_module.BatchedWbcController(str(robot_file), str(yaml_file), robot_name, int(num_envs))

    def _create_batched_foot_planner(self, wbc_module: Any, num_envs: int) -> _BatchedSwingFootPlanner:
        return _BatchedSwingFootPlanner(
            wbc_module,
            int(num_envs),
            dt=self._wbc_dt,
            takeoff_clearance=self._swing_foot_takeoff_clearance,
            landing_clearance=self._swing_foot_landing_clearance,
            midpoint_height=self._swing_foot_midpoint_height,
        )

    def _create_wbc_engine(self, extension_dir: str | None, params: dict[str, Any], wbc_module: Any | None = None):
        if wbc_module is None:
            wbc_module = self._import_wbc_module(extension_dir)
        robot_file, yaml_file, robot_name = self._resolve_wbc_asset_paths(extension_dir, params)
        return wbc_module.WbcEngine(str(robot_file), str(yaml_file), robot_name)
