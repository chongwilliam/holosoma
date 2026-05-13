"""Common termination term helpers."""

from __future__ import annotations

from typing import Any

from holosoma.utils.safe_torch_import import torch


def timeout_exceeded(env, **_) -> torch.Tensor:
    """Terminate environments that exceeded the maximum episode length."""
    return env.episode_length_buf > env.max_episode_length


def _as_tensor(value: Any, env, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    elif getattr(value, "_is_tensor_proxy", False):
        tensor = value[:]
    else:
        tensor = torch.as_tensor(value, device=env.device)
    return tensor.to(device=env.device, dtype=dtype)


def non_finite_sim_state(env, **_) -> torch.Tensor:
    """Terminate environments whose root, DOF position, or DOF velocity state is non-finite."""
    root_states = _as_tensor(env.simulator.robot_root_states, env)
    dof_pos = _as_tensor(env.simulator.dof_pos, env)
    dof_vel = _as_tensor(env.simulator.dof_vel, env)

    return (
        ~torch.isfinite(root_states).flatten(start_dim=1).all(dim=1)
        | ~torch.isfinite(dof_pos).flatten(start_dim=1).all(dim=1)
        | ~torch.isfinite(dof_vel).flatten(start_dim=1).all(dim=1)
    )


def non_finite_action_torques(env, **_) -> torch.Tensor:
    """Terminate environments whose action term torques are non-finite."""
    if env.action_manager is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for _, term in env.action_manager.iter_terms():
        latched_mask = getattr(term, "non_finite_torque_mask", None)
        if isinstance(latched_mask, torch.Tensor):
            reset |= latched_mask.to(device=env.device, dtype=torch.bool)

        torques = getattr(term, "torques", None)
        if isinstance(torques, torch.Tensor) and torques.numel() > 0:
            reset |= ~torch.isfinite(torques).flatten(start_dim=1).all(dim=1)

    return reset
