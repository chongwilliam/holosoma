"""Basic locomotion observation terms.

These functions compute individual observation components for legged locomotion tasks.
Each function mirrors the manager-based observation pipeline that replaced the legacy direct `_get_obs_*()` helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from holosoma.utils.rotations import quat_inverse, quat_mul, quat_rotate_inverse, quat_to_angle_axis, yaw_quat
from holosoma.utils.torch_utils import get_axis_params, to_torch

if TYPE_CHECKING:
    from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager
    from holosoma.managers.command.terms.locomotion import LocomotionGait


def _as_tensor(value: Any, env: LeggedRobotLocomotionManager) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    elif getattr(value, "_is_tensor_proxy", False):
        tensor = value[:]
    else:
        tensor = torch.as_tensor(value)
    return tensor.to(device=env.device, dtype=torch.float32)


def _base_quat(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    return _as_tensor(env.base_quat, env)


def gravity_vector(env: LeggedRobotLocomotionManager, up_axis_idx: int = 2) -> torch.Tensor:
    axis = to_torch(get_axis_params(-1.0, up_axis_idx), device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def base_forward_vector(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    axis = to_torch([1.0, 0.0, 0.0], device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def get_base_lin_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    root_states = env.simulator.robot_root_states
    lin_vel_world = root_states[:, 7:10]
    return quat_rotate_inverse(_base_quat(env), lin_vel_world, w_last=True)


def get_base_ang_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    return quat_rotate_inverse(_base_quat(env), ang_vel_world, w_last=True)


def get_projected_gravity(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    return quat_rotate_inverse(_base_quat(env), gravity_vector(env), w_last=True)


def base_lin_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Base linear velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_lin_vel()
    """
    return get_base_lin_vel(env)


def base_ang_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Base angular velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_ang_vel()
    """
    return get_base_ang_vel(env)


def projected_gravity(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Gravity vector projected into base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_projected_gravity()
    """
    return get_projected_gravity(env)


def dof_pos(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Joint positions relative to default positions.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_pos()
    """
    return env.simulator.dof_pos - env.default_dof_pos


def dof_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Joint velocities.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_vel()
    """
    return env.simulator.dof_vel


def actions(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Last actions taken by the policy.

    Returns:
        Tensor of shape [num_envs, num_actions]

    Equivalent to:
        env._get_obs_actions()
    """
    return env.action_manager.action


def _pelvis_yaw_quat(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    return yaw_quat(_base_quat(env), w_last=True)


def _pelvis_pos_w(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    return env.simulator.robot_root_states[:, 0:3]


def _to_pelvis_yaw_frame(env: LeggedRobotLocomotionManager, vectors: torch.Tensor) -> torch.Tensor:
    shape = vectors.shape
    vectors_flat = vectors.reshape(-1, 3)
    yaw_quat_flat = _pelvis_yaw_quat(env).repeat_interleave(vectors_flat.shape[0] // env.num_envs, dim=0)
    return quat_rotate_inverse(yaw_quat_flat, vectors_flat, w_last=True).reshape(shape)


def _relative_pos_to_pelvis_yaw_frame(env: LeggedRobotLocomotionManager, pos_w: torch.Tensor) -> torch.Tensor:
    pelvis_pos_w = _pelvis_pos_w(env)
    while pelvis_pos_w.dim() < pos_w.dim():
        pelvis_pos_w = pelvis_pos_w.unsqueeze(1)
    return _to_pelvis_yaw_frame(env, pos_w - pelvis_pos_w)


def _vel_to_pelvis_yaw_frame(env: LeggedRobotLocomotionManager, vel_w: torch.Tensor) -> torch.Tensor:
    return _to_pelvis_yaw_frame(env, vel_w)


def _relative_ori_to_pelvis_yaw_frame(env: LeggedRobotLocomotionManager, quat_w: torch.Tensor) -> torch.Tensor:
    quat_w = _as_tensor(quat_w, env)
    shape = quat_w.shape[:-1]
    quat_flat = quat_w.reshape(-1, 4)
    yaw_quat_flat = _pelvis_yaw_quat(env).repeat_interleave(quat_flat.shape[0] // env.num_envs, dim=0)
    quat_rel = quat_mul(quat_inverse(yaw_quat_flat, w_last=True), quat_flat, w_last=True)
    angle, axis = quat_to_angle_axis(quat_rel)
    return (angle.unsqueeze(-1) * axis).reshape(shape + (3,))


def _right_left_foot_indices(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    feet_indices = env.feet_indices
    if feet_indices.numel() < 2:
        return feet_indices

    def body_name_for_index(idx: int) -> str:
        simulator = env.simulator
        root_model = getattr(simulator, "root_model", None)
        if root_model is not None and 0 <= idx < root_model.nbody:
            name = root_model.body(idx).name
            get_clean_name = getattr(simulator, "_get_clean_name", None)
            return get_clean_name(name) if get_clean_name is not None else name

        body_names = getattr(simulator, "body_names", [])
        if 0 <= idx < len(body_names):
            return body_names[idx]
        return ""

    ordered: list[int] = []
    for side in ("right", "left"):
        for idx in feet_indices.detach().cpu().tolist():
            idx_int = int(idx)
            if side in body_name_for_index(idx_int).lower():
                ordered.append(idx_int)
                break
    if len(ordered) == 2:
        return torch.as_tensor(ordered, device=feet_indices.device, dtype=feet_indices.dtype)
    return feet_indices


def _foot_index(env: LeggedRobotLocomotionManager, side: str) -> torch.Tensor:
    side_idx = 0 if side == "right" else 1
    foot_indices = _right_left_foot_indices(env)
    if foot_indices.numel() <= side_idx:
        raise RuntimeError(f"Cannot resolve {side} foot index from feet_indices={foot_indices}.")
    return foot_indices[side_idx]


def _com_pos_w(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    simulator = env.simulator
    value = getattr(simulator, "com_pos", None)
    if isinstance(value, torch.Tensor) and value.shape == (env.num_envs, 3):
        return value.to(device=env.device)

    return _pelvis_pos_w(env)


def _com_vel_w(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    simulator = env.simulator
    for attr_name in ("com_lin_vel", "com_vel"):
        value = getattr(simulator, attr_name, None)
        if isinstance(value, torch.Tensor) and value.shape == (env.num_envs, 3):
            return value.to(device=env.device)

    return simulator.robot_root_states[:, 7:10]


def com_pos(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """COM position relative to the pelvis yaw-aligned frame, shape [num_envs, 3]."""
    return _relative_pos_to_pelvis_yaw_frame(env, _com_pos_w(env).unsqueeze(1)).squeeze(1)


def com_lin_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """COM linear velocity expressed in the pelvis yaw-aligned frame, shape [num_envs, 3]."""
    return _vel_to_pelvis_yaw_frame(env, _com_vel_w(env).unsqueeze(1)).squeeze(1)


def pelvis_ori(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Pelvis orientation relative to the pelvis yaw-aligned frame, shape [num_envs, 3]."""
    return _relative_ori_to_pelvis_yaw_frame(env, _base_quat(env))


def pelvis_ang_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Pelvis angular velocity expressed in the pelvis yaw-aligned frame, shape [num_envs, 3]."""
    return _vel_to_pelvis_yaw_frame(env, env.simulator.robot_root_states[:, 10:13].unsqueeze(1)).squeeze(1)


def torso_ori(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Torso orientation relative to the pelvis yaw-aligned frame, shape [num_envs, 3]."""
    torso_idx = getattr(env, "torso_index", None)
    if torso_idx is None:
        return pelvis_ori(env)
    torso_quat_w = env.simulator._rigid_body_rot[:, int(torso_idx)]
    return _relative_ori_to_pelvis_yaw_frame(env, torso_quat_w)


def torso_ang_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Torso angular velocity expressed in the pelvis yaw-aligned frame, shape [num_envs, 3]."""
    torso_idx = getattr(env, "torso_index", None)
    if torso_idx is None:
        return pelvis_ang_vel(env)
    torso_ang_vel_w = env.simulator._rigid_body_ang_vel[:, int(torso_idx)]
    return _vel_to_pelvis_yaw_frame(env, torso_ang_vel_w.unsqueeze(1)).squeeze(1)


def foot_pos(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Right and left foot positions relative to the pelvis yaw-aligned frame, shape [num_envs, 6]."""
    foot_indices = _right_left_foot_indices(env)
    foot_pos_w = env.simulator._rigid_body_pos[:, foot_indices, :]
    return _relative_pos_to_pelvis_yaw_frame(env, foot_pos_w).reshape(env.num_envs, -1)


def right_foot_pos(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Right foot position relative to the pelvis yaw-aligned frame, shape [num_envs, 3]."""
    foot_pos_w = env.simulator._rigid_body_pos[:, _foot_index(env, "right"), :]
    return _relative_pos_to_pelvis_yaw_frame(env, foot_pos_w.unsqueeze(1)).squeeze(1)


def right_foot_lin_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Right foot linear velocity expressed in the pelvis yaw-aligned frame, shape [num_envs, 3]."""
    foot_vel_w = env.simulator._rigid_body_vel[:, _foot_index(env, "right"), :]
    return _vel_to_pelvis_yaw_frame(env, foot_vel_w.unsqueeze(1)).squeeze(1)


def left_foot_pos(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Left foot position relative to the pelvis yaw-aligned frame, shape [num_envs, 3]."""
    foot_pos_w = env.simulator._rigid_body_pos[:, _foot_index(env, "left"), :]
    return _relative_pos_to_pelvis_yaw_frame(env, foot_pos_w.unsqueeze(1)).squeeze(1)


def left_foot_lin_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Left foot linear velocity expressed in the pelvis yaw-aligned frame, shape [num_envs, 3]."""
    foot_vel_w = env.simulator._rigid_body_vel[:, _foot_index(env, "left"), :]
    return _vel_to_pelvis_yaw_frame(env, foot_vel_w.unsqueeze(1)).squeeze(1)


def foot_ori(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Right and left foot orientations relative to the pelvis yaw-aligned frame, shape [num_envs, 6]."""
    foot_indices = _right_left_foot_indices(env)
    foot_quat_w = env.simulator._rigid_body_rot[:, foot_indices, :]
    return _relative_ori_to_pelvis_yaw_frame(env, foot_quat_w).reshape(env.num_envs, -1)


def right_foot_ori(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Right foot orientation relative to the pelvis yaw-aligned frame, shape [num_envs, 3]."""
    foot_quat_w = env.simulator._rigid_body_rot[:, _foot_index(env, "right"), :]
    return _relative_ori_to_pelvis_yaw_frame(env, foot_quat_w)


def right_foot_ang_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Right foot angular velocity expressed in the pelvis yaw-aligned frame, shape [num_envs, 3]."""
    foot_ang_vel_w = env.simulator._rigid_body_ang_vel[:, _foot_index(env, "right"), :]
    return _vel_to_pelvis_yaw_frame(env, foot_ang_vel_w.unsqueeze(1)).squeeze(1)


def left_foot_ori(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Left foot orientation relative to the pelvis yaw-aligned frame, shape [num_envs, 3]."""
    foot_quat_w = env.simulator._rigid_body_rot[:, _foot_index(env, "left"), :]
    return _relative_ori_to_pelvis_yaw_frame(env, foot_quat_w)


def left_foot_ang_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Left foot angular velocity expressed in the pelvis yaw-aligned frame, shape [num_envs, 3]."""
    foot_ang_vel_w = env.simulator._rigid_body_ang_vel[:, _foot_index(env, "left"), :]
    return _vel_to_pelvis_yaw_frame(env, foot_ang_vel_w.unsqueeze(1)).squeeze(1)


def command_lin_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Commanded linear velocity (x, y).

    Returns:
        Tensor of shape [num_envs, 2]

    Equivalent to:
        env.command_manager.commands[:, :2]
    """
    return env.command_manager.commands[:, :2]


def command_ang_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Commanded angular velocity (yaw).

    Returns:
        Tensor of shape [num_envs, 1]

    Equivalent to:
        env.command_manager.commands[:, 2:3]
    """
    return env.command_manager.commands[:, 2:3]


def sin_phase(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Sine of the gait phase.

    Returns:
        Tensor of shape [num_envs, 1]

    Note: Requires env to have 'phase' attribute (e.g., LeggedRobotLocomotionManager)
    """
    gait_state = env.command_manager.get_state("locomotion_gait")
    if gait_state is None:
        raise AttributeError("locomotion_gait is not registered with the command manager.")
    gait_state = cast("LocomotionGait", gait_state)
    phase = gait_state.phase
    if phase is None:
        raise RuntimeError("Gait phase tensor has not been initialized.")
    return torch.sin(phase)


def cos_phase(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Cosine of the gait phase.

    Returns:
        Tensor of shape [num_envs, 1]

    Note: Requires env to have 'phase' attribute (e.g., LeggedRobotLocomotionManager)
    """
    gait_state = env.command_manager.get_state("locomotion_gait")
    if gait_state is None:
        raise AttributeError("locomotion_gait is not registered with the command manager.")
    gait_state = cast("LocomotionGait", gait_state)
    phase = gait_state.phase
    if phase is None:
        raise RuntimeError("Gait phase tensor has not been initialized.")
    return torch.cos(phase)
