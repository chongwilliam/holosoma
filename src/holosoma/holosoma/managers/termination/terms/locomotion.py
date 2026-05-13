"""Locomotion-specific termination terms."""

from __future__ import annotations

from holosoma.managers.observation.terms.locomotion import get_projected_gravity
from holosoma.managers.termination.base import TerminationTermBase
from holosoma.managers.termination.terms.common import _as_tensor
from holosoma.utils.safe_torch_import import torch


def _apply_probability(mask: torch.Tensor, probability: float, device: torch.device) -> torch.Tensor:
    """Optionally apply probabilistic gating to a mask."""
    if probability >= 1.0:
        return mask
    if probability <= 0.0:
        return torch.zeros_like(mask, dtype=torch.bool)
    sample = torch.rand(1, device=device)
    return mask & (sample < probability)


def contact_forces_exceeded(
    env, force_threshold: float = 1.0, contact_indices_attr: str = "termination_contact_indices"
) -> torch.Tensor:
    """Terminate if contact forces exceed threshold.

    Note: If you want to disable contact termination, simply don't add this term to your
    termination config instead of using a flag.
    """
    indices = getattr(env, contact_indices_attr)
    contact_forces = env.simulator.contact_forces[:, indices, :3]
    return torch.any(torch.norm(contact_forces, dim=-1) > force_threshold, dim=1)


class BothFeetAirborne(TerminationTermBase):
    """Terminate after both feet have no contacts for consecutive checks."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._airborne_counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._airborne_counts.zero_()
        else:
            self._airborne_counts[env_ids] = 0

    def __call__(
        self,
        env,
        max_airborne_steps: int,
        right_contact_count_attr: str = "right_foot_contact_count",
        left_contact_count_attr: str = "left_foot_contact_count",
        action_term_name: str = "torque_control",
        skip_wbc_bootstrap_hold: bool = True,
    ) -> torch.Tensor:
        if max_airborne_steps < 1:
            raise ValueError("max_airborne_steps must be at least 1.")

        right_counts = getattr(env.simulator, right_contact_count_attr)
        left_counts = getattr(env.simulator, left_contact_count_attr)
        right_counts = right_counts.to(device=env.device, dtype=torch.long).reshape(-1)
        left_counts = left_counts.to(device=env.device, dtype=torch.long).reshape(-1)

        skip_mask = self._wbc_bootstrap_hold_mask(env, action_term_name) if skip_wbc_bootstrap_hold else None
        both_feet_airborne = (right_counts <= 0) & (left_counts <= 0)
        if skip_mask is not None:
            both_feet_airborne &= ~skip_mask
            self._airborne_counts[skip_mask] = 0

        self._airborne_counts[both_feet_airborne] += 1
        self._airborne_counts[~both_feet_airborne] = 0
        return self._airborne_counts >= max_airborne_steps

    def _wbc_bootstrap_hold_mask(self, env, action_term_name: str) -> torch.Tensor | None:
        action_manager = getattr(env, "action_manager", None)
        if action_manager is None:
            return None

        try:
            action_term = action_manager.get_term(action_term_name)
        except KeyError:
            return None

        hold_mask = getattr(action_term, "wbc_bootstrap_hold_mask", None)
        if not isinstance(hold_mask, torch.Tensor):
            return None
        return hold_mask.to(device=env.device, dtype=torch.bool).reshape(-1)


def gravity_tilt_exceeded(env, threshold_x: float, threshold_y: float) -> torch.Tensor:
    """Terminate if projected gravity exceeds roll/pitch thresholds."""
    if not getattr(env.config.termination, "terminate_by_gravity", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    grav = get_projected_gravity(env)
    tilt_x = torch.abs(grav[:, 0]) > threshold_x
    tilt_y = torch.abs(grav[:, 1]) > threshold_y
    return tilt_x | tilt_y


def base_height_below_threshold(env, min_height: float) -> torch.Tensor:
    """Terminate if base height drops below threshold."""
    if not getattr(env.config.termination, "terminate_by_low_height", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    base_height = env.simulator.robot_root_states[:, 2]
    return base_height < min_height


def root_angular_velocity_exceeded(env, max_root_ang_vel: float = 20.0, norm: str = "l2") -> torch.Tensor:
    """Terminate if root angular velocity exceeds the configured threshold."""
    root_states = _as_tensor(env.simulator.robot_root_states, env)
    root_ang_vel = root_states[:, 10:13]

    if norm == "l2":
        ang_vel = torch.linalg.vector_norm(root_ang_vel, dim=1)
    elif norm == "linf":
        ang_vel = torch.max(torch.abs(root_ang_vel), dim=1).values
    else:
        raise ValueError(f"Unsupported root angular velocity norm: {norm!r}. Expected 'l2' or 'linf'.")

    return ang_vel > max_root_ang_vel


def dof_position_limit_exceeded(env, probability: float = 1.0) -> torch.Tensor:
    """Terminate when DOF position limits are exceeded."""
    if not getattr(env.config.termination, "terminate_when_close_to_dof_pos_limit", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    lower_violation = -(env.simulator.dof_pos - env.simulator.dof_pos_limits_termination[:, 0]).clip(max=0.0)
    upper_violation = (env.simulator.dof_pos - env.simulator.dof_pos_limits_termination[:, 1]).clip(min=0.0)
    violation = torch.sum(lower_violation + upper_violation, dim=1) > 0.0
    return _apply_probability(violation, probability, env.device)


def dof_velocity_limit_exceeded(env, probability: float = 1.0) -> torch.Tensor:
    """Terminate when DOF velocity limits are exceeded."""
    if not getattr(env.config.termination, "terminate_when_close_to_dof_vel_limit", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    delta = (
        torch.abs(env.simulator.dof_vel)
        - env.dof_vel_limits * env.config.termination_scales.termination_close_to_dof_vel_limit
    ).clip(min=0.0, max=1.0)
    violation = torch.sum(delta, dim=1) > 0.0
    return _apply_probability(violation, probability, env.device)


def torque_limit_exceeded(env, probability: float = 1.0) -> torch.Tensor:
    """Terminate when actuator torques exceed limits."""
    if not getattr(env.config.termination, "terminate_when_close_to_torque_limit", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    torques = env.action_manager.get_term("joint_control").torques
    delta = (
        torch.abs(torques) - env.torque_limits * env.config.termination_scales.termination_close_to_torque_limit
    ).clip(min=0.0, max=1.0)
    violation = torch.sum(delta, dim=1) > 0.0
    return _apply_probability(violation, probability, env.device)
