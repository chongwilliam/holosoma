from __future__ import annotations

import dataclasses
import logging
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypedDict, cast

import tyro
from loguru import logger

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    init_sim_imports,
    load_checkpoint,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app
from holosoma.utils.tyro_utils import TYRO_CONIFG


class TrainingContext:
    """Context manager for training lifecycle and resource management."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.simulation_app: Any | None = None

    def __enter__(self):
        # Initialize simulation app
        self.simulation_app = init_sim_imports(self.config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean shutdown using the utility function
        close_simulation_app(self.simulation_app)

    def train(self) -> None:
        """Train using this context's sim app."""
        train(self.config, training_context=self)


@contextmanager
def training_context(config: ExperimentConfig):
    """Context manager function for training."""
    with TrainingContext(config) as ctx:
        yield ctx


class MultGPUConfig(TypedDict):
    global_rank: int
    local_rank: int
    world_size: int


def configure_multi_gpu() -> MultGPUConfig | None:
    """Configure multi-gpu training and return configuration dictionary, or `None` if single-GPU training."""
    import torch

    gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
    is_distributed = gpu_world_size > 1

    if not is_distributed:
        return None

    gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
    gpu_global_rank = int(os.getenv("RANK", "0"))

    if gpu_local_rank >= gpu_world_size:
        raise ValueError(f"Local rank '{gpu_local_rank}' is greater than or equal to world size '{gpu_world_size}'.")

    if gpu_global_rank >= gpu_world_size:
        raise ValueError(f"Global rank '{gpu_global_rank}' is greater than or equal to world size '{gpu_world_size}'.")

    torch.distributed.init_process_group(backend="nccl", rank=gpu_global_rank, world_size=gpu_world_size)
    torch.cuda.set_device(gpu_local_rank)

    multi_gpu_config: MultGPUConfig = {
        "global_rank": gpu_global_rank,
        "local_rank": gpu_local_rank,
        "world_size": gpu_world_size,
    }
    logger.info(f"Running with multi-GPU parameters: {multi_gpu_config}")

    return multi_gpu_config


def get_device(config, distributed_conf: MultGPUConfig | None) -> str:
    import torch

    is_config_device_specified = hasattr(config, "device") and config.device is not None
    is_multi_gpu = distributed_conf is not None

    if is_config_device_specified:
        if is_multi_gpu and config.device != cast("dict", distributed_conf)["local_rank"]:
            raise ValueError(
                f"Device specified in config ({config.device}) \
                              does not match expected local rank {cast('dict', distributed_conf)['local_rank']}"
            )
        device = config.device
    elif is_multi_gpu:
        device = f"cuda:{cast('dict', distributed_conf)['local_rank']}"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    return device


def configure_logging(distributed_conf: MultGPUConfig | None = None, log_dir: Path | None = None):
    # Configure logging.
    from holosoma.utils.logging import LoguruLoggingBridge

    logger.remove()
    is_main_process = distributed_conf is None or distributed_conf["global_rank"] == 0

    # logging to file (from all ranks)
    if log_dir is not None:
        fname = f"train_rank_{distributed_conf['global_rank']:02d}.log" if distributed_conf is not None else "train.log"
        log_path = log_dir / fname
        logger.add(str(log_path), level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default in rank0
    if is_main_process:
        console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    else:
        console_log_level = "ERROR"
    logger.add(sys.stdout, level=console_log_level, colorize=True)
    logging.basicConfig(level=logging.DEBUG if is_main_process else logging.ERROR)
    logging.getLogger().addHandler(LoguruLoggingBridge())


def _tensor_view(value: Any, *, device: str, dtype: Any):
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value
    elif getattr(value, "_is_tensor_proxy", False):
        tensor = value[:]
    else:
        tensor = torch.as_tensor(value)
    return tensor.to(device=device, dtype=dtype)


def _build_starting_pose_hold_controller(env: Any, device: str):
    import torch

    target_q = _tensor_view(env.simulator.dof_pos, device=device, dtype=torch.float32).clone()
    dof_count = target_q.shape[1]

    kp = torch.zeros(dof_count, device=device, dtype=torch.float32)
    kd = torch.zeros_like(kp)
    stiffness = env.robot_config.control.stiffness
    damping = env.robot_config.control.damping
    for idx, dof_name in enumerate(env.robot_config.dof_names[:dof_count]):
        dof_key = dof_name.replace("_joint", "")
        for pattern, gain in stiffness.items():
            if pattern in dof_key:
                kp[idx] = gain
                kd[idx] = damping[pattern]
                break

    torque_limits = torch.tensor(
        env.robot_config.dof_effort_limit_list[:dof_count],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    return target_q, kp.unsqueeze(0), kd.unsqueeze(0), torque_limits


def _compute_starting_pose_hold_torques(env: Any, target_q: Any, kp: Any, kd: Any, torque_limits: Any):
    dof_pos = _tensor_view(env.simulator.dof_pos, device=str(target_q.device), dtype=target_q.dtype)
    dof_vel = _tensor_view(env.simulator.dof_vel, device=str(target_q.device), dtype=target_q.dtype)
    torques = kp * (target_q - dof_pos) - kd * dof_vel
    return torques.clamp(min=-torque_limits, max=torque_limits)


def _contact_count(simulator: Any, side: str) -> int:
    count = getattr(simulator, f"{side}_foot_contact_count", None)
    if count is None:
        return -1
    try:
        return int(count[0].detach().cpu().item())
    except Exception:
        return int(count[0])


def _both_feet_in_contact(simulator: Any) -> bool:
    return _contact_count(simulator, "right") > 0 and _contact_count(simulator, "left") > 0


def _dof_states_for_setter(env: Any, dof_states: Any):
    if dof_states.dim() == 3:
        return dof_states.reshape(env.num_envs * env.num_dof, 2)
    return dof_states


def _reset_starting_dual_stance_without_action_step(env: Any, device: str) -> None:
    import torch

    if hasattr(env, "_init_buffers"):
        env._init_buffers()

    env_ids = torch.arange(env.num_envs, device=env.device)
    observation_manager = getattr(env, "observation_manager", None)
    if observation_manager is not None:
        observation_manager.reset(env_ids)

    root_states = env.base_init_state.unsqueeze(0).repeat(env.num_envs, 1).to(device=env.device, dtype=torch.float32)
    root_states[:, 7:13] = 0.0

    terrain_manager = getattr(env, "terrain_manager", None)
    if terrain_manager is not None:
        terrain_state = terrain_manager.get_state("locomotion_terrain")
        env_origins = getattr(terrain_state, "env_origins", None)
        if env_origins is not None:
            root_states[:, :3] += env_origins[env_ids].to(device=env.device, dtype=root_states.dtype)

    dof_pos = env.default_dof_pos[env_ids].to(device=env.device, dtype=torch.float32)
    dof_vel = torch.zeros_like(dof_pos)
    dof_states = torch.stack([dof_pos, dof_vel], dim=-1)

    env.simulator.robot_root_states[env_ids] = root_states
    env.simulator.dof_pos[env_ids] = dof_pos
    env.simulator.dof_vel[env_ids] = 0.0
    env.simulator.set_actor_root_state_tensor_robots(env_ids, env.simulator.robot_root_states)
    env.simulator.set_dof_state_tensor_robots(env_ids, _dof_states_for_setter(env, dof_states))
    env.simulator.refresh_sim_tensors()

    if getattr(env, "action_manager", None) is not None:
        env.action_manager.reset(env_ids)
    if getattr(env, "command_manager", None) is not None:
        _force_zero_stand_commands(env)
    if getattr(env, "termination_manager", None) is not None:
        env.termination_manager.reset(env_ids)
    if getattr(env, "curriculum_manager", None) is not None:
        env.curriculum_manager.reset(env_ids)

    # Keep lin/ang velocities exactly zero after manager resets touch internal buffers.
    _zero_sim_velocities(env, device)


def _zero_sim_velocities(env: Any, device: str) -> None:
    import torch

    env_ids = torch.arange(env.num_envs, device=env.device)
    root_states = _tensor_view(env.simulator.robot_root_states, device=device, dtype=torch.float32)
    root_states[:, 7:13] = 0.0
    env.simulator.robot_root_states[:, 7:13] = 0.0
    env.simulator.set_actor_root_state_tensor_robots(env_ids, env.simulator.robot_root_states)

    dof_pos = _tensor_view(env.simulator.dof_pos, device=device, dtype=torch.float32)
    dof_vel = torch.zeros_like(dof_pos)
    dof_states = torch.stack([dof_pos, dof_vel], dim=-1)
    env.simulator.dof_vel[:] = 0.0
    env.simulator.set_dof_state_tensor_robots(env_ids, _dof_states_for_setter(env, dof_states))
    env.simulator.refresh_sim_tensors()


def _get_torque_control_term(env: Any) -> Any:
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None or not hasattr(action_manager, "get_term"):
        raise RuntimeError("Starting dual-stance WBC hold requires an action_manager with a torque_control term.")
    return action_manager.get_term("torque_control")


def _force_zero_stand_commands(env: Any) -> None:
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return

    commands = getattr(command_manager, "commands", None)
    if commands is not None:
        commands.zero_()

    get_state = getattr(command_manager, "get_state", None)
    gait_state = get_state("locomotion_gait") if callable(get_state) else None
    if gait_state is None:
        return

    phase = getattr(gait_state, "phase", None)
    stand_phase = float(getattr(gait_state, "stand_phase_value", 0.0))
    if phase is not None:
        phase.fill_(stand_phase)

    phase_offset = getattr(gait_state, "phase_offset", None)
    if phase_offset is not None:
        phase_offset.fill_(stand_phase)

    phase_dt = getattr(gait_state, "phase_dt", None)
    if phase_dt is not None:
        phase_dt.zero_()

    gait_freq = getattr(gait_state, "gait_freq", None)
    if gait_freq is not None:
        gait_freq.zero_()


def _wbc_state_arrays(env: Any, term: Any, device: str) -> tuple[Any, Any]:
    import torch

    from holosoma.managers.action.terms.torque_control import root_states_to_xyz_rpy

    root_states = _tensor_view(env.simulator.robot_root_states, device=device, dtype=torch.float32)
    dof_pos = _tensor_view(env.simulator.dof_pos, device=device, dtype=torch.float32)
    dof_vel = _tensor_view(env.simulator.dof_vel, device=device, dtype=torch.float32)
    q_tensor = torch.cat([root_states_to_xyz_rpy(root_states), dof_pos], dim=1)
    dq_tensor = torch.cat([root_states[:, 7:13], dof_vel], dim=1)
    return term._as_numpy_2d(q_tensor, "hold_wbc_q"), term._as_numpy_2d(dq_tensor, "hold_wbc_dq")


def _wbc_dual_stance_support_args(env: Any, term: Any) -> tuple[Any, ...]:
    dual_state = int(term.State.DUAL_STANCE)
    contact_arrays = getattr(term, "_stance_support_contact_arrays", None)
    if callable(contact_arrays):
        right_contact_points, left_contact_points, right_contact_bases, left_contact_bases = contact_arrays()
    else:
        right_contact_points = term._as_numpy_2d(env.simulator.right_foot_contact_position, "hold_right_contact_points")
        left_contact_points = term._as_numpy_2d(env.simulator.left_foot_contact_position, "hold_left_contact_points")
        right_contact_bases = term._contact_bases_as_numpy(env.simulator.right_foot_contact_basis, "hold_right_contact_bases")
        left_contact_bases = term._contact_bases_as_numpy(env.simulator.left_foot_contact_basis, "hold_left_contact_bases")
    right_grfs = -term._batched_local_foot_ground_reaction_wrenches("right")
    left_grfs = -term._batched_local_foot_ground_reaction_wrenches("left")
    right_in_contact = term._batched_foot_contact_in_contact("right", right_grfs)
    left_in_contact = term._batched_foot_contact_in_contact("left", left_grfs)
    return (
        dual_state,
        right_contact_points[0],
        right_contact_bases[0],
        left_contact_points[0],
        left_contact_bases[0],
        bool(right_in_contact[0]),
        bool(left_in_contact[0]),
        False,
    )


def _explicitly_update_wbc_robot_and_support(env: Any, term: Any, device: str) -> tuple[bool, bool]:
    q, dq = _wbc_state_arrays(env, term, device)
    term.wbc.update_robot(q, dq)

    support_args = _wbc_dual_stance_support_args(env, term)
    update_stance_support = getattr(term.wbc, "updateStanceSupport", None)
    if not callable(update_stance_support):
        controller = getattr(term.wbc, "_controller", None)
        update_stance_support = getattr(controller, "updateStanceSupport", None)
    if callable(update_stance_support):
        update_stance_support(*support_args)
    else:
        debug_engine = getattr(term, "_wbc_debug_engine", None)
        update_debug_support = getattr(debug_engine, "updateStanceSupport", None)
        if callable(update_debug_support):
            update_debug_support(*support_args)

    return bool(support_args[5]), bool(support_args[6])


def _force_wbc_dual_stance(term: Any, *, reset_controller: bool = False) -> None:
    import torch

    dual_state = int(term.State.DUAL_STANCE)
    if hasattr(term, "curr_state"):
        for env_idx in range(len(term.curr_state)):
            term.curr_state[env_idx] = dual_state
    if hasattr(term, "transition_start_time"):
        for env_idx in range(len(term.transition_start_time)):
            term.transition_start_time[env_idx] = 0.0

    setattr(term, "_single_wbc_state", dual_state)
    setattr(term, "_single_wbc_transition_start_time", 0.0)

    startup_desired_state = getattr(term, "_startup_desired_state", None)
    if isinstance(startup_desired_state, torch.Tensor):
        startup_desired_state.fill_(dual_state)

    phase_shift = getattr(term, "_wbc_phase_shift_frac", None)
    target_phase_fn = getattr(term, "_startup_target_phase_for_states", None)
    if isinstance(phase_shift, torch.Tensor) and callable(target_phase_fn):
        desired_states = torch.full_like(phase_shift, dual_state, dtype=torch.long)
        sin_phase, cos_phase = term._phase_features_for_wbc(term._actions_after_delay)
        raw_phase = torch.remainder(
            torch.atan2(sin_phase[:, 0], cos_phase[:, 0]),
            2.0 * torch.pi,
        ) / (2.0 * torch.pi)
        target_phase = target_phase_fn(desired_states).to(device=phase_shift.device, dtype=phase_shift.dtype)
        phase_shift[:] = torch.remainder(target_phase - raw_phase.to(device=phase_shift.device), 1.0)

    if not reset_controller:
        return

    reset_state = getattr(term.wbc, "reset_state", None)
    if callable(reset_state):
        try:
            reset_state(dual_state, 0.0)
        except TypeError:
            for env_idx in range(getattr(term.env, "num_envs", 1)):
                reset_state(dual_state, 0.0, env_idx)


def _prepare_wbc_dual_stance_hold(term: Any) -> None:
    import torch

    bootstrap_done = getattr(term, "_wbc_bootstrap_done", None)
    if isinstance(bootstrap_done, torch.Tensor):
        bootstrap_done.fill_(True)

    bootstrap_hold_mask = getattr(term, "_last_wbc_bootstrap_hold_mask", None)
    if isinstance(bootstrap_hold_mask, torch.Tensor):
        bootstrap_hold_mask.zero_()

    dual_contact_mask = getattr(term, "_last_wbc_dual_contact_mask", None)
    if isinstance(dual_contact_mask, torch.Tensor):
        dual_contact_mask.fill_(True)

    pending_reinit = getattr(term, "_pending_wbc_reinitialize", None)
    if isinstance(pending_reinit, torch.Tensor):
        pending_reinit.fill_(False)

    reset_targets = getattr(term, "_reset_wbc_integrated_targets", None)
    if callable(reset_targets):
        reset_targets()

    reinitialize_tasks = getattr(term.wbc, "reInitializeAllTasks", None)
    if callable(reinitialize_tasks):
        reinitialize_tasks()
    else:
        debug_engine = getattr(term, "_wbc_debug_engine", None)
        reinitialize_debug_tasks = getattr(debug_engine, "reInitializeAllTasks", None)
        if callable(reinitialize_debug_tasks):
            reinitialize_debug_tasks()

    _force_wbc_dual_stance(term, reset_controller=False)


def _compute_zero_action_wbc_torques(env: Any, term: Any, device: str):
    import torch

    action_dim = int(getattr(term, "action_dim", env.action_manager.total_action_dim))
    zero_actions = torch.zeros(env.num_envs, action_dim, device=device, dtype=torch.float32)
    _force_zero_stand_commands(env)
    _force_wbc_dual_stance(term, reset_controller=False)
    _explicitly_update_wbc_robot_and_support(env, term, device)
    torques = term._compute_torques(zero_actions)
    sanitize = getattr(term, "_sanitize_non_finite_torques", None)
    if callable(sanitize):
        torques = sanitize(torques)
    term.torques[:] = torques.to(device=term.torques.device, dtype=term.torques.dtype)
    return term.torques


def _first_tensor_value(value: Any) -> Any:
    import torch

    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return None
    item = value.flatten()[0].detach().cpu().item()
    if isinstance(item, bool):
        return item
    if isinstance(item, float):
        return float(item)
    return int(item)


def _release_dual_stance_warmup_for_training(env: Any, term: Any) -> None:
    import torch

    for attr_name in ("_forced_wbc_desired_state", "_fixed_wbc_foot_target_poses"):
        if hasattr(term, attr_name):
            delattr(term, attr_name)

    env_ids = torch.arange(env.num_envs, device=env.device)
    if hasattr(env, "episode_length_buf"):
        env.episode_length_buf.zero_()

    command_manager = getattr(env, "command_manager", None)
    if command_manager is not None:
        command_manager.reset(env_ids)

    pending_reinit = getattr(term, "_pending_wbc_reinitialize", None)
    if isinstance(pending_reinit, torch.Tensor):
        pending_reinit.zero_()

    phase_shift = getattr(term, "_wbc_phase_shift_frac", None)
    if isinstance(phase_shift, torch.Tensor):
        phase_shift.zero_()

    startup_desired_state = getattr(term, "_startup_desired_state", None)
    if isinstance(startup_desired_state, torch.Tensor):
        startup_desired_state.fill_(int(term.State.DUAL_STANCE))

    logger.info(
        "Released dual-stance WBC warmup; gait clock and policy actions are now enabled, "
        "and WBC desired state now follows the raw gait phase."
    )


def _refresh_observations_after_manual_warmup(env: Any):
    refresh_sim_tensors = getattr(env, "_refresh_sim_tensors", None)
    if callable(refresh_sim_tensors):
        refresh_sim_tensors()
    pre_compute = getattr(env, "_pre_compute_observations_callback", None)
    if callable(pre_compute):
        pre_compute()
    compute_obs = getattr(env, "_compute_observations", None)
    if callable(compute_obs):
        compute_obs()
    clip_obs = getattr(env, "_clip_observations", None)
    if callable(clip_obs):
        clip_obs()
    return env.obs_buf_dict


def _run_starting_dual_stance_hold(
    env: Any,
    tyro_config: ExperimentConfig,
    device: str,
    *,
    duration_s: float | None = None,
    reset_first: bool = True,
    release_for_training: bool = False,
) -> None:
    import torch

    if reset_first:
        _reset_starting_dual_stance_without_action_step(env, device)
    term = _get_torque_control_term(env)

    target_q, kp, kd, torque_limits = _build_starting_pose_hold_controller(env, device)
    fps = float(tyro_config.simulator.config.sim.fps)
    hold_duration_s = (
        float(tyro_config.training.hold_starting_dual_stance_duration_s)
        if duration_s is None
        else float(duration_s)
    )
    if hold_duration_s <= 0.0:
        raise ValueError("Dual-stance WBC hold duration must be positive.")
    max_steps = int(round(hold_duration_s * fps))
    print_every = max(1, int(tyro_config.training.hold_starting_dual_stance_print_every))

    logger.info(
        "Starting dual-stance warmup: PD until contact, then WBC zero actions for "
        f"{hold_duration_s:.3f}s ({max_steps} physics steps), num_envs={env.num_envs}."
    )
    settle_step_idx = 0
    last_hold_torques = _compute_starting_pose_hold_torques(env, target_q, kp, kd, torque_limits)
    while not _both_feet_in_contact(env.simulator) and settle_step_idx < max_steps:
        settle_step_idx += 1
        torques = _compute_starting_pose_hold_torques(env, target_q, kp, kd, torque_limits)
        last_hold_torques = torques
        env.simulator.apply_torques_at_dof(torques)
        env.simulator.simulate_at_each_physics_step()
        env.simulator.refresh_sim_tensors()
        env.render()

        if settle_step_idx % print_every == 0 or _both_feet_in_contact(env.simulator):
            root_pos = _tensor_view(env.simulator.robot_root_states, device=device, dtype=torch.float32)[0, :3]
            torque_norm = float(torch.linalg.vector_norm(torques[0]).detach().cpu().item())
            logger.info(
                "pd_settle_step="
                f"{settle_step_idx} sim_time={float(env.simulator.time()):.6f} "
                f"root_pos={root_pos.detach().cpu().tolist()} "
                f"torque_norm={torque_norm:.6f} "
                f"right_count={_contact_count(env.simulator, 'right')} "
                f"left_count={_contact_count(env.simulator, 'left')}"
            )

    if not _both_feet_in_contact(env.simulator):
        raise RuntimeError(
            "Both feet did not reach contact during the dual-stance WBC hold setup: "
            f"settle_steps={settle_step_idx} "
            f"right_count={_contact_count(env.simulator, 'right')} "
            f"left_count={_contact_count(env.simulator, 'left')}"
        )

    _zero_sim_velocities(env, device)
    _prepare_wbc_dual_stance_hold(term)
    setattr(term, "_forced_wbc_desired_state", int(term.State.DUAL_STANCE))
    fixed_foot_poses = term._batched_foot_pose_matrices()
    setattr(
        term,
        "_fixed_wbc_foot_target_poses",
        {
            "right": fixed_foot_poses["right"].copy(),
            "left": fixed_foot_poses["left"].copy(),
        },
    )
    right_wbc_contact, left_wbc_contact = _explicitly_update_wbc_robot_and_support(env, term, device)
    _force_wbc_dual_stance(term, reset_controller=True)
    logger.info(
        "transition_to_wbc:"
        f" pending_reinit={_first_tensor_value(getattr(term, '_pending_wbc_reinitialize', None))}"
        f" bootstrap_done={_first_tensor_value(getattr(term, '_wbc_bootstrap_done', None))}"
        f" startup_desired_state={_first_tensor_value(getattr(term, '_startup_desired_state', None))}"
        f" forced_desired_state={getattr(term, '_forced_wbc_desired_state', None)}"
        f" curr_state={term.curr_state[0] if hasattr(term, 'curr_state') and len(term.curr_state) > 0 else None}"
        f" right_wbc_contact={right_wbc_contact}"
        f" left_wbc_contact={left_wbc_contact}"
        f" fixed_right_foot_z={fixed_foot_poses['right'][0, 2, 3]:.6f}"
        f" fixed_left_foot_z={fixed_foot_poses['left'][0, 2, 3]:.6f}"
    )

    if last_hold_torques.shape == term.torques.shape:
        term.torques[:] = last_hold_torques.to(device=term.torques.device, dtype=term.torques.dtype)

    for step_idx in range(settle_step_idx, max_steps):
        torques = _compute_zero_action_wbc_torques(env, term, device)
        env.simulator.apply_torques_at_dof(torques)
        prev_dof_vel = getattr(term, "_prev_dof_vel", None)
        if isinstance(prev_dof_vel, torch.Tensor):
            dof_vel = _tensor_view(env.simulator.dof_vel, device=str(prev_dof_vel.device), dtype=prev_dof_vel.dtype)
            prev_dof_vel.copy_(dof_vel)
        env.simulator.simulate_at_each_physics_step()
        env.simulator.refresh_sim_tensors()
        env.render()

        if step_idx % print_every == 0 or step_idx == max_steps - 1:
            root_pos = _tensor_view(env.simulator.robot_root_states, device=device, dtype=torch.float32)[0, :3]
            torque_norm = float(torch.linalg.vector_norm(torques[0]).detach().cpu().item())
            desired_state = getattr(term, "_last_wbc_desired_state", None)
            desired_state_value = None
            if isinstance(desired_state, torch.Tensor) and desired_state.numel() > 0:
                desired_state_value = int(desired_state[0].detach().cpu().item())
            bootstrap_hold = getattr(term, "wbc_bootstrap_hold_mask", None)
            bootstrap_hold_value = None
            if isinstance(bootstrap_hold, torch.Tensor) and bootstrap_hold.numel() > 0:
                bootstrap_hold_value = bool(bootstrap_hold[0].detach().cpu().item())
            logger.info(
                "wbc_hold_step="
                f"{step_idx} sim_time={float(env.simulator.time()):.6f} "
                f"root_pos={root_pos.detach().cpu().tolist()} "
                f"torque_norm={torque_norm:.6f} "
                f"desired_state={desired_state_value} "
                f"forced_desired_state={getattr(term, '_forced_wbc_desired_state', None)} "
                f"bootstrap_hold={bootstrap_hold_value} "
                f" pending_reinit={_first_tensor_value(getattr(term, '_pending_wbc_reinitialize', None))} "
                f"right_count={_contact_count(env.simulator, 'right')} "
                f"left_count={_contact_count(env.simulator, 'left')}"
            )

    if release_for_training:
        _release_dual_stance_warmup_for_training(env, term)


def _install_initial_dual_stance_training_warmup(env: Any, tyro_config: ExperimentConfig, device: str) -> None:
    if not bool(tyro_config.training.dual_stance_training_warmup):
        return

    original_reset_all = env.reset_all
    original_reset_envs_idx = env.reset_envs_idx
    warmup_done = False
    partial_reset_warning_emitted = False

    def _env_ids_tensor(env_ids: Any):
        import torch

        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=env.device, dtype=torch.long).flatten()
        return torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()

    def _reset_covers_all_envs(env_ids: Any) -> bool:
        import torch

        idx = _env_ids_tensor(env_ids)
        if idx.numel() != env.num_envs:
            return False
        expected = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        return bool(torch.equal(torch.sort(idx).values, expected))

    def _run_warmup_after_reset(*, reset_first: bool) -> None:
        _run_starting_dual_stance_hold(
            env,
            tyro_config,
            device,
            duration_s=tyro_config.training.dual_stance_training_warmup_duration_s,
            reset_first=reset_first,
            release_for_training=True,
        )

    def reset_all_with_initial_dual_stance_warmup():
        nonlocal warmup_done
        if warmup_done:
            return original_reset_all()

        warmup_done = True
        _run_warmup_after_reset(reset_first=True)
        return _refresh_observations_after_manual_warmup(env)

    def reset_envs_idx_with_dual_stance_warmup(env_ids, target_states=None, target_buf=None):
        nonlocal partial_reset_warning_emitted

        original_reset_envs_idx(env_ids, target_states=target_states, target_buf=target_buf)
        idx = _env_ids_tensor(env_ids)
        if idx.numel() == 0:
            return

        if _reset_covers_all_envs(idx):
            logger.info("Running dual-stance training warmup after environment reset.")
            _run_warmup_after_reset(reset_first=False)
            return

        if not partial_reset_warning_emitted:
            partial_reset_warning_emitted = True
            logger.warning(
                "Skipping dual-stance WBC warmup after partial reset because it advances the shared simulator. "
                "Use num_envs=1 or synchronized resets if every termination reset must include the warmup."
            )

    env.reset_all = reset_all_with_initial_dual_stance_warmup
    env.reset_envs_idx = reset_envs_idx_with_dual_stance_warmup
    logger.info(
        "Installed dual-stance training warmup: "
        f"{tyro_config.training.dual_stance_training_warmup_duration_s:.3f}s WBC hold before learning and full resets."
    )


def train(tyro_config: ExperimentConfig, training_context: TrainingContext | None = None) -> None:
    """Train an agent with optional context for sim app management.

    Parameters
    ----------
    training_context : Optional[TrainingContext]
        Optional training context with pre-initialized sim app.
        If None, creates and manages sim app automatically.
    """

    if training_context is not None:
        # Use the context's pre-initialized sim app
        simulation_app = training_context.simulation_app
        auto_close = False  # Context will handle closing
    else:
        # Default behavior - create and manage sim app ourselves
        simulation_app = init_sim_imports(tyro_config)
        auto_close = True

    try:
        # have to import torch after isaacgym
        import torch  # noqa: F401
        import torch.distributed as dist
        import wandb

        from holosoma.agents.base_algo.base_algo import BaseAlgo
        from holosoma.utils.common import seeding

        # unresolved_conf = dataclasses.asdict(tyro_config)
        # import ipdb; ipdb.set_trace()

        # Initialize process group
        distributed_conf: MultGPUConfig | None = configure_multi_gpu()
        device: str = get_device(tyro_config, distributed_conf)
        is_distributed = distributed_conf is not None
        is_main_process = distributed_conf is None or distributed_conf["local_rank"] == 0

        # Configure logger
        logger_cfg = tyro_config.logger
        wandb_enabled = logger_cfg.type == "wandb"

        # Compute experiment directory from logger and training config
        from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp

        timestamp = get_timestamp()
        experiment_dir = get_experiment_dir(logger_cfg, tyro_config.training, timestamp, task_name="locomotion")

        # Configure logging with experiment directory
        configure_logging(distributed_conf=distributed_conf, log_dir=experiment_dir)

        # Random seed
        seed = tyro_config.training.seed
        if distributed_conf is not None:
            seed += distributed_conf["global_rank"]
        seeding(seed, torch_deterministic=tyro_config.training.torch_deterministic)

        wandb_run_path: str | None = None

        # Configure wandb in rank 0
        if wandb_enabled and is_main_process:
            from holosoma.config_types.logger import WandbLoggerConfig

            assert isinstance(logger_cfg, WandbLoggerConfig), (
                "Logger config must be WandbLoggerConfig when type is wandb"
            )
            wandb_cfg = logger_cfg
            # Use training config for project/name, fallback to logger config, then defaults
            default_project = tyro_config.training.project or wandb_cfg.project or "default_project"
            default_run_name = (
                f"{timestamp}_{tyro_config.training.name or 'run'}_"
                f"{wandb_cfg.group or 'default'}_{tyro_config.robot.asset.robot_type}"
            )
            wandb_dir = Path(wandb_cfg.dir or (experiment_dir / ".wandb"))
            wandb_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Saving wandb logs to {wandb_dir}")

            # Only pass optional parameters when specified so wandb can fall back to environment defaults.
            wandb_kwargs: dict[str, Any] = {
                "project": wandb_cfg.project or default_project,
                "name": wandb_cfg.name or default_run_name,
                "config": dataclasses.asdict(tyro_config),
                "dir": str(wandb_dir),
                "mode": wandb_cfg.mode,
            }
            if wandb_cfg.entity:
                wandb_kwargs["entity"] = wandb_cfg.entity
            if wandb_cfg.group:
                wandb_kwargs["group"] = wandb_cfg.group
            if wandb_cfg.id:
                wandb_kwargs["id"] = wandb_cfg.id
            if wandb_cfg.tags:
                wandb_kwargs["tags"] = list(wandb_cfg.tags)
            if wandb_cfg.resume is not None:
                wandb_kwargs["resume"] = wandb_cfg.resume

            wandb.init(**wandb_kwargs)
            if wandb.run is not None:
                wandb_run_path = f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}"

        # Distribute environments across GPUs for proper multi-GPU training
        if distributed_conf is not None:
            original_num_envs = tyro_config.training.num_envs
            num_envs = original_num_envs // distributed_conf["world_size"]
            tyro_config = dataclasses.replace(
                tyro_config, training=dataclasses.replace(tyro_config.training, num_envs=num_envs)
            )
            logger.info(
                f"Distributed training: GPU {distributed_conf['global_rank']} will run {tyro_config.training.num_envs} "
                f"environments (total across all GPUs: {original_num_envs})"
            )

        env_target = tyro_config.env_class

        if tyro_config.training.hold_starting_dual_stance:
            tyro_config = dataclasses.replace(tyro_config, randomization=RandomizationManagerCfg())
            logger.info("Disabled randomization for hold_starting_dual_stance to match g1_wbc_torque_control.py.")

        tyro_env_config = get_tyro_env_config(tyro_config)
        env = get_class(env_target)(tyro_env_config, device=device)

        if tyro_config.training.hold_starting_dual_stance:
            _run_starting_dual_stance_hold(env, tyro_config, device)
            return

        # For manager system, pre-process config AFTER env creation
        # (need managers to compute dims)
        observation_manager = getattr(env, "observation_manager", None)
        if observation_manager is None:
            raise RuntimeError(
                f"Manager environment {env_target} is missing observation_manager attribute. "
                "This should not happen if the environment is properly configured."
            )

        experiment_save_dir = experiment_dir
        experiment_save_dir.mkdir(exist_ok=True, parents=True)

        if is_main_process:
            logger.info(f"Saving config file to {experiment_save_dir}")
            config_path = experiment_save_dir / CONFIG_NAME
            tyro_config.save_config(str(config_path))
            if wandb_enabled:
                wandb.save(str(config_path), base_path=experiment_save_dir)

        algo_class = get_class(tyro_config.algo._target_)
        algo: BaseAlgo = algo_class(
            device=device,
            env=env,
            config=tyro_config.algo.config,
            log_dir=experiment_save_dir,
            multi_gpu_cfg=distributed_conf,
        )
        algo.setup()
        algo.attach_checkpoint_metadata(tyro_config, wandb_run_path)
        if tyro_config.training.checkpoint is not None:
            loaded_checkpoint = load_checkpoint(tyro_config.training.checkpoint, str(experiment_save_dir))
            tyro_config = dataclasses.replace(
                tyro_config, training=dataclasses.replace(tyro_config.training, checkpoint=str(loaded_checkpoint))
            )
            algo.load(loaded_checkpoint)

        # handle saving config
        _install_initial_dual_stance_training_warmup(env, tyro_config, device)
        algo.learn()

        # teardown wandb before SimApp closes ungracefully (IsaacLab)
        if is_main_process and wandb_enabled:
            logger.info("Shutting down wandb...")
            wandb.teardown()

        # shutdown dist before SimApp closes ungracefully (IsaacLab)
        if is_distributed:
            logger.info("Shutting down distributed processes...")
            dist.destroy_process_group()
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Exception occurred during training: {e}\n{tb_str}")
        sys.exit(1)  # manually set exit code, not possible via isaacsim app.close()
    finally:
        if auto_close:
            close_simulation_app(simulation_app)

    logger.info("Training shutdown complete.")


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    print(tyro_cfg.curriculum)
    train(tyro_cfg)


if __name__ == "__main__":
    main()
