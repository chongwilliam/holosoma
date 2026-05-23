"""Run a single G1 environment with the torque-control WBC action term.

This script is intended as a small standing-start smoke test for the G1 WBC
integration. It manually computes and applies WBC torques so that the test can
compare the batched and scalar WBC command paths without going through the full
RL action loop.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg
from holosoma.config_types.logger import DisabledLoggerConfig
from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.config_types.video import VideoConfig
from holosoma.config_values import simulator as simulator_defaults
from holosoma.config_values import terrain as terrain_defaults
from holosoma.config_values.loco.g1 import action as g1_loco_action
from holosoma.config_values.loco.g1 import experiment as g1_loco_experiment
from holosoma.managers.action.terms.torque_control import axis_angle_to_matrix, matrix_to_axis_angle, root_states_to_xyz_rpy
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment

def _resolve_path(path_str: str | None) -> str | None:
    if path_str is None:
        return None
    return str(Path(path_str).expanduser().resolve())

def _existing_default(*paths: Path) -> str:
    for path in paths:
        if path.exists():
            return str(path)
    return str(paths[0])

def _build_action_cfg(wbc_extension_dir: str, robot_file: str, yaml_file: str, robot_name: str) -> ActionManagerCfg:
    default_params = g1_loco_action.g1_29dof_torque.terms["torque_control"].params
    return ActionManagerCfg(
        terms={
            "torque_control": ActionTermCfg(
                func="holosoma.managers.action.terms.torque_control:JointTorqueActionTerm",
                params={
                    **default_params,
                    # JointTorqueActionTerm expects WBC-specific arguments in params.
                    "wbc_extension_dir": wbc_extension_dir,
                    "robot_file": robot_file,
                    "yaml_file": yaml_file,
                    "robot_name": robot_name,
                    "store_wbc_debug_snapshots": True,
                    "use_command_as_pelvis_velocity_action": False,
                    "visualize_contact_points": True,
                    "visualize_contact_frames": True,
                    "contact_point_radius": 0.025,
                },
                scale=1.0,
                clip=None,
            ),
        }
    )

def _build_config(
    *,
    simulator_name: str,
    headless: bool,
    wbc_extension_dir: str,
    robot_file: str,
    yaml_file: str,
    robot_name: str,
):
    simulator_cfg = {
        "mujoco": simulator_defaults.mujoco,
        "mjwarp": simulator_defaults.mjwarp,
        "isaacsim": simulator_defaults.isaacsim,
    }[simulator_name]

    base_config = g1_loco_experiment.g1_29dof_fast_sac
    disabled_logger = DisabledLoggerConfig(video=VideoConfig(enabled=False), base_dir="logs")
    standing_robot = dataclasses.replace(
        base_config.robot,
        init_state=dataclasses.replace(base_config.robot.init_state, pos=[0.0, 0.0, 0.793]),
    )

    return dataclasses.replace(
        base_config,
        training=dataclasses.replace(
            base_config.training,
            num_envs=1,
            headless=headless,
        ),
        simulator=simulator_cfg,
        robot=standing_robot,
        terrain=terrain_defaults.terrain_locomotion_plane,
        logger=disabled_logger,
        randomization=RandomizationManagerCfg(),
        action=_build_action_cfg(
            wbc_extension_dir=wbc_extension_dir,
            robot_file=robot_file,
            yaml_file=yaml_file,
            robot_name=robot_name,
        ),
    )

def _format_tensor_row(row: torch.Tensor) -> str:
    values = [float(x) for x in row.detach().cpu().tolist()]
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def _expand_action_values(values: list[float], action_dim: int, label: str) -> list[float]:
    if len(values) == 1:
        return [float(values[0])] * action_dim
    if len(values) == action_dim:
        return [float(value) for value in values]
    raise ValueError(f"{label} expects either 1 value or {action_dim} values, got {len(values)}.")


def _root_pos(env) -> torch.Tensor:
    return env.simulator.robot_root_states[0, :3]

def _force_single_env_origin_at_world(env) -> None:
    """Keep the one-env visual test near the world origin instead of a terrain tile."""
    terrain_state = env.terrain_manager.get_state("locomotion_terrain")
    if hasattr(terrain_state, "_env_origins"):
        terrain_state._env_origins.zero_()
    if hasattr(terrain_state, "_custom_origins"):
        terrain_state._custom_origins = False

    if hasattr(env.simulator, "env_origins"):
        env.simulator.env_origins.zero_()

    scene = getattr(env.simulator, "scene", None)
    scene_origins = getattr(scene, "env_origins", None)
    if scene_origins is not None:
        scene_origins.zero_()

def _zero_reset_velocities(env) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device)

    root_states = _state_tensor(env.simulator.robot_root_states).to(device=env.device, dtype=torch.float32)
    root_states[:, 7:13] = 0.0
    env.simulator.robot_root_states[:, 7:13] = 0.0
    env.simulator.set_actor_root_state_tensor_robots(env_ids, env.simulator.robot_root_states)

    dof_pos = _state_tensor(env.simulator.dof_pos).to(device=env.device, dtype=torch.float32)
    dof_vel = torch.zeros_like(dof_pos)
    dof_states = torch.stack([dof_pos, dof_vel], dim=-1)
    env.simulator.dof_vel[:] = 0.0
    if dof_states.dim() == 3:
        dof_states_to_set = dof_states.reshape(env.num_envs * env.num_dof, 2)
    else:
        dof_states_to_set = dof_states
    env.simulator.set_dof_state_tensor_robots(env_ids, dof_states_to_set)
    env.simulator.refresh_sim_tensors()

def _reset_without_physics_step(env) -> None:
    """Reset managers and simulator state without applying the action term once."""
    if hasattr(env, "_init_buffers"):
        env._init_buffers()

    env_ids = torch.arange(env.num_envs, device=env.device)
    root_states = env.base_init_state.unsqueeze(0).repeat(env.num_envs, 1).to(device=env.device, dtype=torch.float32)
    root_states[:, 7:13] = 0.0

    dof_pos = env.default_dof_pos[env_ids].to(device=env.device, dtype=torch.float32)
    dof_vel = torch.zeros_like(dof_pos)
    dof_states = torch.stack([dof_pos, dof_vel], dim=-1)

    # reset_envs_idx wants the batched form [num_envs, num_dof, 2].
    env.reset_envs_idx(env_ids, target_states={"root_states": root_states, "dof_states": dof_states})

    # The simulator setter may want the flattened Isaac-style form [num_envs * num_dof, 2].
    if dof_states.dim() == 3:
        dof_states_to_set = dof_states.reshape(env.num_envs * env.num_dof, 2)
    else:
        dof_states_to_set = dof_states

    env.simulator.set_actor_root_state_tensor_robots(env_ids, env.simulator.robot_root_states)
    env.simulator.set_dof_state_tensor_robots(env_ids, dof_states_to_set)
    env.simulator.refresh_sim_tensors()

def _contact_count(simulator, side: str) -> int:
    return int(getattr(simulator, f"{side}_foot_contact_count")[0].item())

def _both_feet_in_contact(simulator) -> bool:
    return _contact_count(simulator, "right") > 0 and _contact_count(simulator, "left") > 0

def _foot_grf(env, term, side: str) -> torch.Tensor:
    grf_fn = getattr(term, "_local_foot_force_sensor_wrench", None)
    if callable(grf_fn):
        try:
            return torch.as_tensor(grf_fn(0, side), device=env.device, dtype=torch.float32)
        except RuntimeError:
            pass

    foot_indices = getattr(term, "_foot_body_indices", {})
    body_idx = foot_indices.get(side) if isinstance(foot_indices, dict) else None
    if body_idx is None:
        body_names = list(getattr(env.simulator, "body_names", []))
        foot_body_name = getattr(env.robot_config, "foot_body_name", "foot")
        exact_name = f"{side}_{foot_body_name}"
        body_idx = next((idx for idx, name in enumerate(body_names) if name == exact_name), None)
        if body_idx is None:
            body_idx = next(
                (idx for idx, name in enumerate(body_names) if side in name.lower() and foot_body_name in name),
                None,
            )

    contact_forces = getattr(env.simulator, "contact_forces", None)
    if body_idx is None or contact_forces is None or contact_forces.numel() == 0:
        return torch.full((6,), float("nan"), device=env.device, dtype=torch.float32)

    wrench = contact_forces[0, body_idx]
    if wrench.shape[-1] == 3:
        wrench = torch.cat([wrench, torch.zeros(3, device=wrench.device, dtype=wrench.dtype)], dim=0)
    return wrench[:6].to(device=env.device, dtype=torch.float32)

def _state_tensor(value) -> torch.Tensor:
    tensor = value.clone() if hasattr(value, "clone") else torch.as_tensor(value)
    return tensor.detach() if hasattr(tensor, "detach") else tensor

def _current_com_position(wbc_engine, device: str) -> torch.Tensor:
    com_pose = torch.as_tensor(wbc_engine.getPose("com"), device=device, dtype=torch.float32)
    return com_pose[:3, 3]

def _current_wbc_state_arrays(env, term, device: str) -> tuple[np.ndarray, np.ndarray]:
    root_states = _state_tensor(env.simulator.robot_root_states).to(device=device, dtype=torch.float32)
    dof_pos = _state_tensor(env.simulator.dof_pos).to(device=device, dtype=torch.float32)
    dof_vel = _state_tensor(env.simulator.dof_vel).to(device=device, dtype=torch.float32)
    q_tensor = torch.cat([root_states_to_xyz_rpy(root_states), dof_pos], dim=1)
    dq_tensor = torch.cat([root_states[:, 7:13], dof_vel], dim=1)
    q = term._as_numpy_2d(q_tensor, "explicit_wbc_update_q")
    dq = term._as_numpy_2d(dq_tensor, "explicit_wbc_update_dq")
    return q, dq

def _current_stance_support_inputs(env, term, env_idx: int = 0) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool, bool]:
    contact_arrays = getattr(term, "_stance_support_contact_arrays", None)
    if callable(contact_arrays):
        right_contact_points, left_contact_points, right_contact_bases, left_contact_bases = contact_arrays()
    else:
        right_contact_points = term._as_numpy_2d(env.simulator.right_foot_contact_position, "explicit_right_contact_points")
        left_contact_points = term._as_numpy_2d(env.simulator.left_foot_contact_position, "explicit_left_contact_points")
        right_contact_bases = term._contact_bases_as_numpy(env.simulator.right_foot_contact_basis, "explicit_right_contact_bases")
        left_contact_bases = term._contact_bases_as_numpy(env.simulator.left_foot_contact_basis, "explicit_left_contact_bases")
    right_grfs = -term._batched_local_foot_ground_reaction_wrenches("right")
    left_grfs = -term._batched_local_foot_ground_reaction_wrenches("left")
    right_in_contact = term._batched_foot_contact_in_contact("right", right_grfs)
    left_in_contact = term._batched_foot_contact_in_contact("left", left_grfs)

    state = _single_wbc_state(term) if env_idx == 0 else int(term.curr_state[env_idx])
    in_transition = _is_transition_state(term, state)
    return (
        state,
        right_contact_points[env_idx],
        right_contact_bases[env_idx],
        left_contact_points[env_idx],
        left_contact_bases[env_idx],
        bool(right_in_contact[env_idx]),
        bool(left_in_contact[env_idx]),
        in_transition,
    )

def _draw_explicit_contact_visualization(
    term,
    right_contact_points: np.ndarray,
    right_contact_bases: np.ndarray,
    left_contact_points: np.ndarray,
    left_contact_bases: np.ndarray,
) -> None:
    clear_visualization = getattr(term, "_maybe_clear_wbc_visualization_lines", None)
    if callable(clear_visualization):
        clear_visualization()

    draw_contact_points = getattr(term, "_maybe_draw_contact_points", None)
    if callable(draw_contact_points):
        draw_contact_points(
            right_contact_points,
            right_contact_bases,
            left_contact_points,
            left_contact_bases,
        )

def _explicitly_update_stance_support(env, term, command: str) -> None:
    right_contact_points = term._as_numpy_2d(env.simulator.right_foot_contact_position, "explicit_right_contact_points")
    left_contact_points = term._as_numpy_2d(env.simulator.left_foot_contact_position, "explicit_left_contact_points")
    right_contact_bases = term._contact_bases_as_numpy(env.simulator.right_foot_contact_basis, "explicit_right_contact_bases")
    left_contact_bases = term._contact_bases_as_numpy(env.simulator.left_foot_contact_basis, "explicit_left_contact_bases")
    support_args = _current_stance_support_inputs(env, term, 0)
    if command == "single":
        engine = getattr(term, "_wbc_debug_engine", None)
        if engine is None:
            raise RuntimeError("JointTorqueActionTerm did not expose _wbc_debug_engine for --wbc-command single.")
        engine.updateStanceSupport(*support_args)
        _draw_explicit_contact_visualization(
            term,
            right_contact_points,
            right_contact_bases,
            left_contact_points,
            left_contact_bases,
        )
        return

    if command == "batch":
        update_stance_support = getattr(term.wbc, "updateStanceSupport", None)
        if not callable(update_stance_support):
            controller = getattr(term.wbc, "_controller", None)
            update_stance_support = getattr(controller, "updateStanceSupport", None)
        if callable(update_stance_support):
            update_stance_support(*support_args)
        else:
            debug_engine = getattr(term, "_wbc_debug_engine", None)
            if debug_engine is not None and hasattr(debug_engine, "updateStanceSupport"):
                debug_engine.updateStanceSupport(*support_args)
        _draw_explicit_contact_visualization(
            term,
            right_contact_points,
            right_contact_bases,
            left_contact_points,
            left_contact_bases,
        )
        return

    raise ValueError(f"Unknown WBC command mode: {command}")

def _explicitly_update_wbc_robot_state(env, term, command: str, device: str) -> None:
    q, dq = _current_wbc_state_arrays(env, term, device)
    if command == "single":
        engine = getattr(term, "_wbc_debug_engine", None)
        if engine is None:
            raise RuntimeError("JointTorqueActionTerm did not expose _wbc_debug_engine for --wbc-command single.")
        engine.updateRobot(q[0], dq[0])
        _explicitly_update_stance_support(env, term, command)
        return
    if command == "batch":
        term.wbc.update_robot(q, dq)
        _explicitly_update_stance_support(env, term, command)
        return
    raise ValueError(f"Unknown WBC command mode: {command}")

def _assert_pose_slice_matches(
    action_batch_t: torch.Tensor,
    env_idx: int,
    label: str,
    action_slice: slice,
    expected: np.ndarray,
    device: str,
) -> None:
    actual = action_batch_t[env_idx, action_slice]
    expected_t = torch.as_tensor(expected.reshape(-1), device=device, dtype=actual.dtype)
    if not bool(torch.allclose(actual, expected_t, atol=1.0e-5, rtol=1.0e-5)):
        raise RuntimeError(
            f"WBC action {label} slice mismatch: actual={_format_tensor_row(actual)} "
            f"expected={_format_tensor_row(expected_t)}."
        )

def _assert_finite_slice(
    action_batch_t: torch.Tensor,
    env_idx: int,
    label: str,
    action_slice: slice,
) -> None:
    values = action_batch_t[env_idx, action_slice]
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError(f"WBC action {label} slice contains non-finite values: {_format_tensor_row(values)}.")

def _assert_rotation_slice_valid(
    action_batch_t: torch.Tensor,
    env_idx: int,
    label: str,
    action_slice: slice,
) -> None:
    values = action_batch_t[env_idx, action_slice]
    _assert_finite_slice(action_batch_t, env_idx, label, action_slice)
    rotation = values.reshape(3, 3)
    eye = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    det = torch.linalg.det(rotation)
    if not bool(torch.allclose(rotation.T @ rotation, eye, atol=1.0e-3, rtol=1.0e-3)) or not bool(
        torch.allclose(det, torch.ones_like(det), atol=1.0e-3, rtol=1.0e-3)
    ):
        raise RuntimeError(
            f"WBC action {label} slice is not a valid rotation matrix: "
            f"values={_format_tensor_row(values)} det={float(det):.6f}."
        )

def _assert_target_position_slice(
    action_batch_t: torch.Tensor,
    env_idx: int,
    label: str,
    action_slice: slice,
    expected: np.ndarray | None,
    device: str,
) -> None:
    if expected is None:
        _assert_finite_slice(action_batch_t, env_idx, label, action_slice)
        return
    _assert_pose_slice_matches(action_batch_t, env_idx, label, action_slice, expected, device)

def _assert_target_rotation_slice(
    action_batch_t: torch.Tensor,
    env_idx: int,
    label: str,
    action_slice: slice,
    expected: np.ndarray | None,
    device: str,
) -> None:
    if expected is None:
        _assert_rotation_slice_valid(action_batch_t, env_idx, label, action_slice)
        return
    _assert_pose_slice_matches(action_batch_t, env_idx, label, action_slice, expected, device)

def _assert_foot_target_slices_valid(
    action_batch_t: torch.Tensor,
    env_idx: int,
    side: str,
    pos_slice: slice,
    ori_slice: slice,
    vel_slice: slice,
) -> None:
    _assert_finite_slice(action_batch_t, env_idx, f"{side}_foot_pos", pos_slice)
    _assert_rotation_slice_valid(action_batch_t, env_idx, f"{side}_foot_ori", ori_slice)
    _assert_finite_slice(action_batch_t, env_idx, f"{side}_foot_vel", vel_slice)

def _integrated_target_value(term, attr_name: str, env_idx: int) -> np.ndarray | None:
    value = getattr(term, attr_name, None)
    if not isinstance(value, torch.Tensor) or env_idx >= value.shape[0]:
        return None
    return value[env_idx].detach().cpu().numpy()

def _compute_batched_wbc_torques(env, term, actions: torch.Tensor, *, scale_actions: bool) -> torch.Tensor:
    policy_actions = actions if scale_actions else _task_actions_to_policy(actions, term)
    torques = term._compute_torques(policy_actions)
    action_batch = getattr(term, "_last_wbc_action_batch", None)
    if action_batch is not None:
        _assert_wbc_action_layout_values(
            env,
            term,
            _expected_task_actions(actions, term, scale_actions=scale_actions),
        )
    return torques

def _is_transition_state(term, state: int) -> bool:
    transition_names = (
        "DUAL_TO_LEFT_STANCE",
        "DUAL_TO_RIGHT_STANCE",
        "LEFT_TO_DUAL_STANCE",
        "RIGHT_TO_DUAL_STANCE",
    )
    return any(state == int(getattr(term.State, name)) for name in transition_names if hasattr(term.State, name))

def _desired_state_matches_transition(term, desired_state: int, transition_state: int) -> bool:
    if not _is_transition_state(term, transition_state):
        return False
    wbc_module = getattr(term, "_wbc_module", None)
    next_state_fn = getattr(wbc_module, "getNextStateFromTransition", None)
    if not callable(next_state_fn):
        return desired_state == transition_state
    return desired_state == transition_state or desired_state == int(next_state_fn(transition_state))

def _next_scalar_wbc_state(
    term,
    engine,
    state: int,
    desired_state: int,
    right_grf: np.ndarray,
    left_grf: np.ndarray,
) -> int:
    if state == desired_state or _is_transition_state(term, state):
        return state

    achievable_state = state
    if state == int(term.State.DUAL_STANCE):
        achievable_state = int(engine.getContactTransitionFromDualStance(right_grf, left_grf, 1))
    elif state in (int(term.State.LEFT_STANCE), int(term.State.RIGHT_STANCE)):
        achievable_state = int(engine.getContactTransitionFromSingleStance(state, right_grf, left_grf, 1.0, 1))

    if _desired_state_matches_transition(term, desired_state, achievable_state):
        return achievable_state
    return state

def _wbc_targets_from_action(wbc_module, engine, action_row: np.ndarray):
    action_row = np.asarray(action_row, dtype=np.float64).reshape(-1)
    if action_row.shape[0] != 72:
        raise RuntimeError(f"Expected a 72D WBC action row, got shape={action_row.shape}.")

    com_pose = np.asarray(engine.getPose("com"), dtype=np.float64).reshape(4, 4)
    pelvis_pose = np.asarray(engine.getPose("pelvis"), dtype=np.float64).reshape(4, 4)
    torso_pose = np.asarray(engine.getPose("torso"), dtype=np.float64).reshape(4, 4)

    targets = wbc_module.WbcDesiredTargets()
    targets.com.position = com_pose[:3, 3] + action_row[0:3]
    targets.com.linear_velocity = action_row[3:6].copy()
    targets.pelvis.position = pelvis_pose[:3, 3] + action_row[6:9]
    targets.pelvis.orientation = pelvis_pose[:3, :3] @ axis_angle_to_matrix(action_row[9:12])
    targets.pelvis.linear_velocity = action_row[12:15].copy()
    targets.pelvis.angular_velocity = action_row[15:18].copy()
    targets.torso.orientation = torso_pose[:3, :3] @ axis_angle_to_matrix(action_row[18:21])
    targets.torso.angular_velocity = action_row[21:24].copy()
    targets.right_foot.position = action_row[24:27].copy()
    targets.right_foot.orientation = action_row[27:36].reshape(3, 3).copy()
    targets.right_foot.linear_velocity = action_row[36:39].copy()
    targets.right_foot.angular_velocity = action_row[39:42].copy()
    targets.left_foot.position = action_row[42:45].copy()
    targets.left_foot.orientation = action_row[45:54].reshape(3, 3).copy()
    targets.left_foot.linear_velocity = action_row[54:57].copy()
    targets.left_foot.angular_velocity = action_row[57:60].copy()
    return targets

def _single_wbc_state(term) -> int:
    cached_state = getattr(term, "_single_wbc_state", None)
    if cached_state is not None:
        return int(cached_state)
    if hasattr(term, "curr_state") and len(term.curr_state) > 0:
        return int(term.curr_state[0])
    get_states = getattr(term.wbc, "get_states", None)
    if callable(get_states):
        states = get_states()
        if len(states) > 0:
            return int(states[0])
    return int(term.State.DUAL_STANCE)

def _single_wbc_transition_start_time(term) -> float:
    cached_time = getattr(term, "_single_wbc_transition_start_time", None)
    if cached_time is not None:
        return float(cached_time)
    if hasattr(term, "transition_start_time") and len(term.transition_start_time) > 0:
        return float(term.transition_start_time[0])
    get_times = getattr(term.wbc, "get_transition_start_times", None)
    if callable(get_times):
        transition_start_times = get_times()
        if len(transition_start_times) > 0:
            return float(transition_start_times[0])
    return 0.0

def _sync_single_wbc_state(term, output_state: int, transition_start_time: float) -> None:
    cached_transition_start_time = 0.0 if not _is_transition_state(term, output_state) else transition_start_time
    setattr(term, "_single_wbc_state", int(output_state))
    setattr(term, "_single_wbc_transition_start_time", float(cached_transition_start_time))
    if hasattr(term, "curr_state"):
        term.curr_state[0] = int(output_state)
    if hasattr(term, "transition_start_time"):
        term.transition_start_time[0] = float(cached_transition_start_time)

    reset_state = getattr(term.wbc, "reset_state", None)
    if callable(reset_state):
        reset_state(int(output_state), float(cached_transition_start_time), 0)


def _force_wbc_dual_stance(term) -> None:
    dual_state = int(term.State.DUAL_STANCE)
    if hasattr(term, "curr_state"):
        for env_idx in range(len(term.curr_state)):
            term.curr_state[env_idx] = dual_state
    if hasattr(term, "transition_start_time"):
        for env_idx in range(len(term.transition_start_time)):
            term.transition_start_time[env_idx] = 0.0

    setattr(term, "_single_wbc_state", dual_state)
    setattr(term, "_single_wbc_transition_start_time", 0.0)

    reset_state = getattr(term.wbc, "reset_state", None)
    if callable(reset_state):
        reset_state(dual_state, 0.0)

    phase_shift = getattr(term, "_wbc_phase_shift_frac", None)
    target_phase_fn = getattr(term, "_startup_target_phase_for_states", None)
    if isinstance(phase_shift, torch.Tensor) and callable(target_phase_fn):
        desired_states = torch.full_like(phase_shift, dual_state, dtype=torch.long)
        phase_shift[:] = target_phase_fn(desired_states).to(device=phase_shift.device, dtype=phase_shift.dtype)


def _reinitialize_wbc_tasks(term) -> None:
    if hasattr(term.wbc, "reInitializeAllTasks"):
        term.wbc.reInitializeAllTasks()
        return
    debug_engine = getattr(term, "_wbc_debug_engine", None)
    if debug_engine is not None and hasattr(debug_engine, "reInitializeAllTasks"):
        debug_engine.reInitializeAllTasks()
        return
    raise RuntimeError("Invalid reinitialize")


def _reset_wbc_targets_from_current_pose(term) -> None:
    reset_targets = getattr(term, "_reset_wbc_integrated_targets", None)
    if callable(reset_targets):
        reset_targets()


def _compute_single_wbc_torques(env, term, actions: torch.Tensor, *, scale_actions: bool) -> torch.Tensor:
    if env.num_envs != 1:
        raise RuntimeError(f"--wbc-command single requires num_envs=1, got {env.num_envs}.")

    policy_actions = actions if scale_actions else _task_actions_to_policy(actions, term)
    device = policy_actions.device
    root_states = _state_tensor(env.simulator.robot_root_states).to(device=device, dtype=torch.float32)
    dof_pos = _state_tensor(env.simulator.dof_pos).to(device=device, dtype=torch.float32)
    dof_vel = _state_tensor(env.simulator.dof_vel).to(device=device, dtype=torch.float32)
    q_tensor = torch.cat([root_states_to_xyz_rpy(root_states), dof_pos], dim=1)
    dq_tensor = torch.cat([root_states[:, 7:13], dof_vel], dim=1)
    q = term._as_numpy_2d(q_tensor, "single_wbc_q")
    dq = term._as_numpy_2d(dq_tensor, "single_wbc_dq")

    term._last_wbc_q[:1] = q_tensor[:1]
    term._last_wbc_dq[:1] = dq_tensor[:1]
    term._last_wbc_root_state[:1] = root_states[:1]
    term._last_wbc_dof_pos[:1] = dof_pos[:1]

    contact_arrays = getattr(term, "_stance_support_contact_arrays", None)
    if callable(contact_arrays):
        right_contact_points, left_contact_points, right_contact_bases, left_contact_bases = contact_arrays()
    else:
        right_contact_points = term._as_numpy_2d(env.simulator.right_foot_contact_position, "right_contact_points")
        left_contact_points = term._as_numpy_2d(env.simulator.left_foot_contact_position, "left_contact_points")
        right_contact_bases = term._contact_bases_as_numpy(env.simulator.right_foot_contact_basis, "right_contact_bases")
        left_contact_bases = term._contact_bases_as_numpy(env.simulator.left_foot_contact_basis, "left_contact_bases")
    right_grfs = term._batched_local_foot_ground_reaction_wrenches("right")
    left_grfs = term._batched_local_foot_ground_reaction_wrenches("left")
    right_in_contact = term._batched_foot_contact_in_contact("right", right_grfs)
    left_in_contact = term._batched_foot_contact_in_contact("left", left_grfs)
    term._last_wbc_right_grfs = right_grfs.copy()
    term._last_wbc_left_grfs = left_grfs.copy()

    sin_phase, cos_phase = term._phase_features_for_wbc(policy_actions)
    desired_states, swing_sides, remaining_swing_durations = term._desired_states_from_phase(sin_phase, cos_phase)
    desired_state = int(desired_states[0])
    term._last_wbc_sin_phase[:1] = sin_phase[:1]
    term._last_wbc_cos_phase[:1] = cos_phase[:1]
    term._last_wbc_desired_state[:1] = torch.as_tensor(desired_states[:1], device=env.device, dtype=torch.long)
    term._last_wbc_remaining_swing_duration[:1] = torch.as_tensor(
        remaining_swing_durations[:1], device=env.device, dtype=torch.float32
    )

    engine = getattr(term, "_wbc_debug_engine", None)
    if engine is None:
        raise RuntimeError("JointTorqueActionTerm did not expose _wbc_debug_engine for --wbc-command single.")
    engine.updateRobot(q[0], dq[0])

    action_batch = term._actions_for_batched_wbc(
        policy_actions,
        desired_states=desired_states,
        swing_sides=swing_sides,
        remaining_swing_durations=remaining_swing_durations,
    )
    term._maybe_store_wbc_debug_snapshots(
        action_batch,
        right_contact_points,
        left_contact_points,
        right_contact_bases,
        left_contact_bases,
        right_grfs,
        left_grfs,
    )
    term._assert_finite_wbc_action_batch(action_batch)
    action_batch_for_compute = getattr(term, "_wbc_action_batch_for_compute", None)
    compute_action_batch = action_batch_for_compute(action_batch) if callable(action_batch_for_compute) else action_batch

    state = _single_wbc_state(term)
    transition_start_time = _single_wbc_transition_start_time(term)
    next_state = _next_scalar_wbc_state(
        term,
        engine,
        state,
        desired_state,
        right_grfs[0] * bool(right_in_contact[0]),
        left_grfs[0] * bool(left_in_contact[0]),
    )
    in_transition = _is_transition_state(term, next_state)
    if not _is_transition_state(term, state) and in_transition:
        transition_start_time = float(env.simulator.time())

    engine.updateStanceSupport(
        next_state,
        right_contact_points[0],
        right_contact_bases[0],
        left_contact_points[0],
        left_contact_bases[0],
        bool(right_in_contact[0]),
        bool(left_in_contact[0]),
        in_transition
    )
    targets = _wbc_targets_from_action(term._wbc_module, engine, compute_action_batch[0])
    result = engine.compute(next_state, targets, True, True, False, float(env.simulator.time()), transition_start_time)
    if isinstance(result, tuple):
        torque_wbc, output_state = result
    else:
        torque_wbc, output_state = result, next_state

    output_state = int(output_state)
    _sync_single_wbc_state(term, output_state, transition_start_time)

    torque_np = term._actuated_torques_from_wbc_output(torque_wbc).reshape(1, -1)

    term._last_wbc_torque_output = torque_np.copy()
    torques = torch.as_tensor(torque_np, device=device, dtype=term.torques.dtype)
    _assert_wbc_action_layout_values(env, term, _expected_task_actions(actions, term, scale_actions=scale_actions))
    return torques

def _compute_wbc_torques(env, term, actions: torch.Tensor, command: str, *, scale_actions: bool) -> torch.Tensor:
    if command == "batch":
        return _compute_batched_wbc_torques(env, term, actions, scale_actions=scale_actions)
    if command == "single":
        return _compute_single_wbc_torques(env, term, actions, scale_actions=scale_actions)
    raise ValueError(f"Unknown WBC command mode: {command}")

def _task_actions_to_policy(actions: torch.Tensor, term) -> torch.Tensor:
    action_scales = term.action_scales.to(device=actions.device, dtype=actions.dtype)
    return actions / torch.clamp(action_scales, min=1.0e-12)

def _apply_wbc_torques(env, term, actions: torch.Tensor, command: str, *, scale_actions: bool) -> None:
    term.torques[:] = _compute_wbc_torques(env, term, actions, command, scale_actions=scale_actions)
    env.simulator.apply_torques_at_dof(term.torques)
    term._prev_dof_vel.copy_(env.simulator.dof_vel)

def _wbc_action_batch_dim(term) -> int | None:
    action_batch = getattr(term, "_last_wbc_action_batch", None)
    if action_batch is None:
        return None
    return int(action_batch.shape[1])

def _assert_wbc_action_batch_dim_supported(action_batch) -> None:
    if action_batch.shape[1] != 72:
        raise RuntimeError(f"Expected a 72D batched WBC action row, got shape={action_batch.shape}.")

def _expected_task_actions(actions: torch.Tensor, term, *, scale_actions: bool) -> torch.Tensor:
    action_scales = term.action_scales.to(device=actions.device, dtype=actions.dtype)
    if scale_actions:
        return actions * action_scales
    policy_actions = _task_actions_to_policy(actions, term)
    return policy_actions * action_scales

def _expected_wbc_velocity_actions(
    env,
    term,
    expected_task_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    com_vel = expected_task_actions[:, 0:3].clone()
    pelvis_ang_vel = expected_task_actions[:, 3:6].clone()
    if not bool(getattr(term, "_use_command_as_pelvis_velocity_action", False)):
        return com_vel, pelvis_ang_vel

    command_manager = getattr(env, "command_manager", None)
    commands = getattr(command_manager, "commands", None)
    if commands is None:
        return com_vel, pelvis_ang_vel

    command_tensor = torch.as_tensor(commands, device=expected_task_actions.device, dtype=expected_task_actions.dtype)
    if command_tensor.ndim != 2 or command_tensor.shape[0] < expected_task_actions.shape[0] or command_tensor.shape[1] < 3:
        return com_vel, pelvis_ang_vel

    command_tensor = command_tensor[: expected_task_actions.shape[0], :3]
    if not bool(torch.isfinite(command_tensor).all()):
        return com_vel, pelvis_ang_vel

    com_vel.zero_()
    com_vel[:, 0:2] = command_tensor[:, 0:2]
    pelvis_ang_vel.zero_()
    pelvis_ang_vel[:, 2] = command_tensor[:, 2]
    return com_vel, pelvis_ang_vel

def _assert_wbc_action_layout_values(env, term, expected_task_actions: torch.Tensor) -> None:
    action_batch = getattr(term, "_last_wbc_action_batch", None)
    if action_batch is None:
        return
    _assert_wbc_action_batch_dim_supported(action_batch)

    action_batch_t = torch.as_tensor(action_batch, device=expected_task_actions.device, dtype=expected_task_actions.dtype)

    for env_idx in range(action_batch_t.shape[0]):
        com_target_pos = _integrated_target_value(term, "_wbc_integrated_com_pos", env_idx)
        pelvis_target_pos = _integrated_target_value(term, "_wbc_integrated_pelvis_pos", env_idx)
        pelvis_target_rot = _integrated_target_value(term, "_wbc_integrated_pelvis_rot", env_idx)
        torso_target_rot = _integrated_target_value(term, "_wbc_integrated_torso_rot", env_idx)
        _assert_target_position_slice(
            action_batch_t,
            env_idx,
            "com_pos",
            slice(0, 3),
            com_target_pos,
            expected_task_actions.device,
        )
        _assert_target_position_slice(
            action_batch_t,
            env_idx,
            "pelvis_pos",
            slice(6, 9),
            pelvis_target_pos,
            expected_task_actions.device,
        )
        pelvis_target_axis_angle = (
            None if pelvis_target_rot is None else matrix_to_axis_angle(np.asarray(pelvis_target_rot, dtype=np.float64))
        )
        torso_target_axis_angle = (
            None if torso_target_rot is None else matrix_to_axis_angle(np.asarray(torso_target_rot, dtype=np.float64))
        )
        _assert_target_position_slice(
            action_batch_t,
            env_idx,
            "pelvis_rel_ori",
            slice(9, 12),
            pelvis_target_axis_angle,
            expected_task_actions.device,
        )
        _assert_target_position_slice(
            action_batch_t,
            env_idx,
            "torso_rel_ori",
            slice(18, 21),
            torso_target_axis_angle,
            expected_task_actions.device,
        )
        _assert_foot_target_slices_valid(
            action_batch_t,
            env_idx,
            "right",
            slice(24, 27),
            slice(27, 36),
            slice(36, 42),
        )
        _assert_foot_target_slices_valid(
            action_batch_t,
            env_idx,
            "left",
            slice(42, 45),
            slice(45, 54),
            slice(54, 60),
        )

    expected_com_vel, expected_pelvis_ang_vel = _expected_wbc_velocity_actions(
        env,
        term,
        expected_task_actions,
    )
    expected_pairs = (
        ("com_lin_vel", slice(3, 6), expected_com_vel),
        ("pelvis_lin_vel", slice(12, 15), expected_com_vel),
        ("pelvis_ang_vel", slice(15, 18), expected_pelvis_ang_vel),
        ("torso_ang_vel", slice(21, 24), expected_pelvis_ang_vel),
    )

    for label, wbc_slice, expected in expected_pairs:
        actual = action_batch_t[:, wbc_slice]
        if not bool(torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-5)):
            raise RuntimeError(
                f"WBC action {label} slice mismatch: actual={_format_tensor_row(actual[0])} "
                f"expected={_format_tensor_row(expected[0])} "
                f"use_command_as_pelvis_velocity_action={getattr(term, '_use_command_as_pelvis_velocity_action', None)}."
            )

def _assert_required_wbc_bindings(term) -> None:
    wbc_module = getattr(term, "_wbc_module", None)
    if wbc_module is None:
        raise RuntimeError("JointTorqueActionTerm did not expose _wbc_module for binding validation.")
    swing_foot = getattr(wbc_module, "SwingFoot", None)
    if swing_foot is None:
        raise RuntimeError("humanoid_wbc does not expose SwingFoot.")
    if not hasattr(swing_foot, "computeLandingFootstepPoseFromCapturePointBatch"):
        raise RuntimeError("humanoid_wbc.SwingFoot is missing computeLandingFootstepPoseFromCapturePointBatch.")
    swing_foot_instance = swing_foot()
    if not hasattr(swing_foot_instance, "resolveTrajectoryFromCurrentState"):
        raise RuntimeError("humanoid_wbc.SwingFoot is missing resolveTrajectoryFromCurrentState.")
    if not hasattr(swing_foot_instance, "setMidpointHeight"):
        raise RuntimeError("humanoid_wbc.SwingFoot is missing setMidpointHeight.")

def _assert_action_layout_9d(env) -> None:
    if env.action_manager.total_action_dim != 9:
        raise RuntimeError(
            "g1_wbc_torque_control expects the 9D WBC action setup "
            "[com_vel(3), pelvis_ang_vel(3), landing_foot_delta_xyyaw(3)], "
            f"got action_dim={env.action_manager.total_action_dim}."
        )

def _wbc_desired_state(term, env_idx: int = 0) -> int | None:
    desired_state = getattr(term, "_last_wbc_desired_state", None)
    if not isinstance(desired_state, torch.Tensor) or env_idx >= desired_state.shape[0]:
        return None
    return int(desired_state[env_idx].detach().cpu().item())

def _wbc_bootstrap_hold(term, env_idx: int = 0) -> bool | None:
    hold_mask = getattr(term, "wbc_bootstrap_hold_mask", None)
    if not isinstance(hold_mask, torch.Tensor) or env_idx >= hold_mask.shape[0]:
        return None
    return bool(hold_mask[env_idx].detach().cpu().item())

def _draw_com_markers(env, current_com: torch.Tensor, desired_com: torch.Tensor, radius: float) -> None:
    simulator = env.simulator
    if not hasattr(simulator, "draw_sphere"):
        return

    if hasattr(simulator, "clear_lines"):
        simulator.clear_lines()

    current_position = current_com.detach().cpu()
    desired_position = desired_com.detach().cpu()
    simulator.draw_sphere(current_position, radius, [0.1, 1.0, 0.25], env_id=0, pos_id=0)
    simulator.draw_sphere(desired_position, radius, [0.1, 0.8, 1.0], env_id=0, pos_id=1)

def _build_hold_controller(config, simulator, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_q = _state_tensor(simulator.dof_pos).to(device=device, dtype=torch.float32).clone()
    dof_count = target_q.shape[1]

    kp = torch.zeros(dof_count, device=device, dtype=torch.float32)
    kd = torch.zeros_like(kp)
    stiffness = config.robot.control.stiffness
    damping = config.robot.control.damping
    for idx, dof_name in enumerate(config.robot.dof_names[:dof_count]):
        dof_key = dof_name.replace("_joint", "")
        for pattern, gain in stiffness.items():
            if pattern in dof_key:
                kp[idx] = gain
                kd[idx] = damping[pattern]
                break

    torque_limits = torch.tensor(
        config.robot.dof_effort_limit_list[:dof_count],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    return target_q, kp.unsqueeze(0), kd.unsqueeze(0), torque_limits

def _compute_hold_torques(
    simulator,
    target_q: torch.Tensor,
    kp: torch.Tensor,
    kd: torch.Tensor,
    torque_limits: torch.Tensor,
) -> torch.Tensor:
    dof_pos = _state_tensor(simulator.dof_pos).to(device=target_q.device, dtype=target_q.dtype)
    dof_vel = _state_tensor(simulator.dof_vel).to(device=target_q.device, dtype=target_q.dtype)
    torques = kp * (target_q - dof_pos) - kd * dof_vel
    return torch.clamp(torques, min=-torque_limits, max=torque_limits)


def _apply_hold_torques(
    simulator,
    target_q: torch.Tensor,
    kp: torch.Tensor,
    kd: torch.Tensor,
    torque_limits: torch.Tensor,
) -> torch.Tensor:
    torques = _compute_hold_torques(simulator, target_q, kp, kd, torque_limits)
    simulator.apply_torques_at_dof(torques)
    return torques

def _print_hold_summary(hold_step_idx: int, env, term) -> None:
    print(
        f"hold_step={hold_step_idx}"
        f" sim_time={float(env.simulator.time()):.6f}"
        f" root_pos={_format_tensor_row(_root_pos(env))}"
        f" right_count={_contact_count(env.simulator, 'right')}"
        f" right_grf={_format_tensor_row(_foot_grf(env, term, 'right'))}"
        f" left_count={_contact_count(env.simulator, 'left')}"
        f" left_grf={_format_tensor_row(_foot_grf(env, term, 'left'))}"
    )

def _print_step_summary(control_step_idx: int, sim_step_idx: int, env, term) -> None:
    torques = term.torques[0]
    curr_state = term.curr_state[0] if hasattr(term, "curr_state") else None
    batch_state = _batch_state_at(term, 0)
    batch_transition_start = _batch_transition_start_at(term, 0)
    print(
        f"control_step={control_step_idx}"
        f" sim_step={sim_step_idx}"
        f" sim_time={float(env.simulator.time()):.6f}"
        f" state={curr_state}"
        f" batch_state={batch_state}"
        f" batch_transition_start={batch_transition_start}"
        f" desired_state={_wbc_desired_state(term)}"
        f" bootstrap_hold={_wbc_bootstrap_hold(term)}"
        f" action_batch_dim={_wbc_action_batch_dim(term)}"
        f" root_pos={_format_tensor_row(_root_pos(env))}"
        f" torque_norm={float(torch.linalg.vector_norm(torques).item()):.6f}"
        f" torque_max={float(torch.max(torch.abs(torques)).item()):.6f}"
        f" right_count={_contact_count(env.simulator, 'right')}"
        f" right_grf={_format_tensor_row(_foot_grf(env, term, 'right'))}"
        f" left_count={_contact_count(env.simulator, 'left')}"
        f" left_grf={_format_tensor_row(_foot_grf(env, term, 'left'))}"
    )

def _batch_state_at(term, env_idx: int):
    get_states = getattr(term.wbc, "get_states", None)
    if not callable(get_states):
        return None
    states = get_states()
    return states[env_idx] if env_idx < len(states) else None

def _batch_transition_start_at(term, env_idx: int):
    get_times = getattr(term.wbc, "get_transition_start_times", None)
    if not callable(get_times):
        return None
    transition_start_times = get_times()
    if env_idx >= len(transition_start_times):
        return None
    return f"{float(transition_start_times[env_idx]):.6f}"

def main() -> int:
    humanoid_control_root = REPO_ROOT.parent / "humanoid-control"
    default_extension_dir = humanoid_control_root / "build"
    default_holosoma_robot_file = REPO_ROOT / "src" / "holosoma" / "holosoma" / "data" / "robots" / "g1" / "g1_29dof.urdf"
    default_wbc_robot_file = _existing_default(
        humanoid_control_root / "models" / "unitree_g1" / "g1.urdf",
        # default_holosoma_robot_file,
    )
    default_yaml_file = humanoid_control_root / "params" / "g1_parameters.yaml"

    parser = argparse.ArgumentParser(
        description=(
            "Run one G1 environment with the torque_control WBC action term "
            "from a dual-foot standing start."
        )
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=None,
        help="Number of simulator physics steps to run. Defaults to 10 seconds of the selected simulator.",
    )
    parser.add_argument(
        "--contact-hold-max-steps",
        type=int,
        default=None,
        help="Maximum PD hold simulator steps before requiring both feet to be in contact.",
    )
    parser.add_argument("--print-every", type=int, default=1, help="Print summary every N control steps.")
    parser.add_argument("--device", default=None, help="Override simulation device.")
    parser.add_argument("--simulator", choices=("mujoco", "mjwarp", "isaacsim"), default="isaacsim")
    parser.add_argument(
        "--wbc-command",
        choices=("batch", "single"),
        default="batch",
        help="Select the batched controller path or the scalar WbcEngine path for torque computation.",
    )
    parser.add_argument(
        "--action-units",
        choices=("task", "policy"),
        default="task",
        help=(
            "Interpret generated sinusoid actions as task-space velocity/residual commands, "
            "or as normalized policy actions that should be multiplied by robot action_scale."
        ),
    )
    parser.add_argument("--headless", action="store_true", help="Disable the simulator visualization window.")
    parser.add_argument("--disable-com-visualization", action="store_true", help="Do not draw the desired COM marker.")
    parser.add_argument("--com-marker-radius", type=float, default=0.025, help="COM marker radius in meters.")
    parser.add_argument(
        "--action-amplitude",
        type=float,
        nargs="+",
        default=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0],
        metavar="A",
        help=(
            "Sinusoidal action amplitude. Pass one value for all 9 action dimensions, or 9 values for "
            "[com_vel, pelvis_ang_vel, landing_foot_delta_xyyaw]."
        ),
    )
    parser.add_argument(
        "--action-frequency",
        type=float,
        nargs="+",
        default=[1.0],
        metavar="HZ",
        help="Sinusoidal action frequency in Hz. Pass one value for all dimensions, or 9 values.",
    )
    parser.add_argument(
        "--action-phase",
        type=float,
        nargs="+",
        default=[0.0],
        metavar="RAD",
        help="Sinusoidal action phase in radians. Pass one value for all dimensions, or 9 values.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generated test actions.")
    parser.add_argument(
        "--wbc-extension-dir",
        default=str(default_extension_dir),
        help="Path to the built humanoid-control extension directory.",
    )
    parser.add_argument(
        "--robot-file",
        default=default_wbc_robot_file,
        help="Robot file passed to the WBC engine.",
    )
    parser.add_argument(
        "--yaml-file",
        default=str(default_yaml_file),
        help="WBC YAML configuration file for G1.",
    )
    parser.add_argument("--robot-name", default="g1", help="Robot name passed to the WBC engine.")
    args = parser.parse_args()

    selected_simulator_cfg = {
        "mujoco": simulator_defaults.mujoco,
        "mjwarp": simulator_defaults.mjwarp,
        "isaacsim": simulator_defaults.isaacsim,
    }[args.simulator]
    simulator_fps = float(selected_simulator_cfg.config.sim.fps)
    if args.sim_steps is None:
        args.sim_steps = int(round(10.0 * simulator_fps))
    if args.contact_hold_max_steps is None:
        args.contact_hold_max_steps = int(round(2.0 * simulator_fps))

    if args.sim_steps < 0:
        raise ValueError("--sim-steps must be non-negative.")
    if args.contact_hold_max_steps < 0:
        raise ValueError("--contact-hold-max-steps must be non-negative.")
    if args.print_every <= 0:
        raise ValueError("--print-every must be a positive integer.")
    if args.com_marker_radius <= 0.0:
        raise ValueError("--com-marker-radius must be positive.")
    torch.manual_seed(args.seed)

    wbc_extension_dir = _resolve_path(args.wbc_extension_dir)
    robot_file = _resolve_path(args.robot_file)
    yaml_file = _resolve_path(args.yaml_file)

    if wbc_extension_dir is None or not Path(wbc_extension_dir).exists():
        raise FileNotFoundError(f"WBC extension directory does not exist: {wbc_extension_dir}")
    if robot_file is None or not Path(robot_file).exists():
        raise FileNotFoundError(f"Robot file does not exist: {robot_file}")
    if yaml_file is None or not Path(yaml_file).exists():
        raise FileNotFoundError(f"YAML file does not exist: {yaml_file}")

    config = _build_config(
        simulator_name=args.simulator,
        headless=args.headless,
        wbc_extension_dir=wbc_extension_dir,
        robot_file=robot_file,
        yaml_file=yaml_file,
        robot_name=args.robot_name,
    )

    env, device, simulation_app = setup_simulation_environment(config, device=args.device)
    try:
        print("Starting simulation and control")
        _force_single_env_origin_at_world(env)
        _reset_without_physics_step(env)
        _zero_reset_velocities(env)
        term = env.action_manager.get_term("torque_control")
        # _assert_action_layout_9d(env)
        # _assert_required_wbc_bindings(term)

        wbc_engine = getattr(term, "_wbc_debug_engine", term.wbc)
        wbc_dof = int(wbc_engine.dof())
        action_amplitude = torch.tensor(
            _expand_action_values(args.action_amplitude, env.action_manager.total_action_dim, "--action-amplitude"),
            device=device,
            dtype=torch.float32,
        )
        action_frequency = torch.tensor(
            _expand_action_values(args.action_frequency, env.action_manager.total_action_dim, "--action-frequency"),
            device=device,
            dtype=torch.float32,
        )
        action_phase = torch.tensor(
            _expand_action_values(args.action_phase, env.action_manager.total_action_dim, "--action-phase"),
            device=device,
            dtype=torch.float32,
        )
        if bool(torch.any(action_frequency < 0.0)):
            raise ValueError("--action-frequency values must be non-negative.")

        right_count = _contact_count(env.simulator, "right")
        left_count = _contact_count(env.simulator, "left")
        print(
            "initial:"
            f" simulator={args.simulator}"
            f" wbc_command={args.wbc_command}"
            f" action_units={args.action_units}"
            f" device={device}"
            f" headless={args.headless}"
            f" action_dim={env.action_manager.total_action_dim}"
            f" wbc_dof={wbc_dof}"
            f" right_count={right_count}"
            f" right_contact={_format_tensor_row(env.simulator.right_foot_contact_position[0])}"
            f" right_grf={_format_tensor_row(_foot_grf(env, term, 'right'))}"
            f" left_count={left_count}"
            f" left_contact={_format_tensor_row(env.simulator.left_foot_contact_position[0])}"
            f" left_grf={_format_tensor_row(_foot_grf(env, term, 'left'))}"
        )

        control_decimation = int(env.simulator.simulator_config.sim.control_decimation)
        control_steps = (args.sim_steps + control_decimation - 1) // control_decimation
        print(
            "run:"
            f" requested_sim_steps={args.sim_steps}"
            f" control_decimation={control_decimation}"
            f" control_steps={control_steps}"
            f" actual_sim_steps={args.sim_steps}"
            f" nominal_seconds={args.sim_steps / float(env.simulator.simulator_config.sim.fps):.3f}"
        )

        hold_target_q, hold_kp, hold_kd, hold_torque_limits = _build_hold_controller(config, env.simulator, device)
        hold_step_idx = 0
        last_hold_torques = _compute_hold_torques(env.simulator, hold_target_q, hold_kp, hold_kd, hold_torque_limits)
        while not _both_feet_in_contact(env.simulator) and hold_step_idx < args.contact_hold_max_steps:
            hold_step_idx += 1
            last_hold_torques = _apply_hold_torques(env.simulator, hold_target_q, hold_kp, hold_kd, hold_torque_limits)
            env.simulator.simulate_at_each_physics_step()
            env.simulator.refresh_sim_tensors()
            env.render()

            if hold_step_idx % args.print_every == 0 or _both_feet_in_contact(env.simulator):
                _print_hold_summary(hold_step_idx, env, term)

        if not _both_feet_in_contact(env.simulator):
            raise RuntimeError(
                "Both feet did not reach contact during the hold phase: "
                f"hold_steps={hold_step_idx}"
                f" right_count={_contact_count(env.simulator, 'right')}"
                f" left_count={_contact_count(env.simulator, 'left')}"
            )

        print(
            f" hold_steps={hold_step_idx}"
            f" right_count={_contact_count(env.simulator, 'right')}"
            f" left_count={_contact_count(env.simulator, 'left')}"
        )
        _zero_reset_velocities(env)
        _explicitly_update_wbc_robot_state(env, term, args.wbc_command, device)
        _force_wbc_dual_stance(term)
        _reinitialize_wbc_tasks(term)
        _reset_wbc_targets_from_current_pose(term)

        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device, dtype=torch.float32)
        scale_actions = args.action_units == "policy"
        _compute_wbc_torques(env, term, actions, args.wbc_command, scale_actions=scale_actions)
        warmup_torques = last_hold_torques.to(device=term.torques.device, dtype=term.torques.dtype)
        if warmup_torques.shape != term.torques.shape:
            raise RuntimeError(
                f"WBC torque shape mismatch: got {warmup_torques.shape}, expected {term.torques.shape}."
            )
        if not bool(torch.isfinite(warmup_torques).all()):
            raise RuntimeError(f"{args.wbc_command} WBC warmup produced non-finite torques.")
        term.torques[:] = warmup_torques
        print(
            "wbc_warmup:"
            f" command={args.wbc_command}"
            f" desired_state={_wbc_desired_state(term)}"
            f" bootstrap_hold={_wbc_bootstrap_hold(term)}"
            f" action_batch_dim={_wbc_action_batch_dim(term)}"
            # f" torque_norm={float(torch.linalg.vector_norm(warmup_torques[0]).item()):.6f}"
            # f" torque_max={float(torch.max(torch.abs(warmup_torques[0])).item()):.6f}"
        )
        # raise RuntimeError(f"lala")

        control_step_idx = 0
        for sim_step_idx in range(1, args.sim_steps + 1):
            if (sim_step_idx - 1) % control_decimation == 0:
                control_step_idx += 1
                sim_time = float(env.simulator.time())
                actions[:] = action_amplitude * torch.sin(2.0 * torch.pi * action_frequency * sim_time + action_phase)
                _explicitly_update_wbc_robot_state(env, term, args.wbc_command, device)
                if not args.disable_com_visualization:
                    current_com = _current_com_position(wbc_engine, device)
                    desired_com = current_com + torch.cat(
                        [actions[0, 0:2] * float(env.dt), torch.zeros(1, device=device, dtype=torch.float32)]
                    )
                    _draw_com_markers(env, current_com, desired_com, args.com_marker_radius)

                # This script manually computes/applies WBC torques below.
                # Do not also call env.action_manager.process_actions(actions), because that
                # can rescale/cache delayed actions and then this script may convert them again.

            _force_wbc_dual_stance(term)
            _apply_wbc_torques(env, term, actions, args.wbc_command, scale_actions=scale_actions)
            _assert_wbc_action_layout_values(
                env,
                term,
                _expected_task_actions(actions, term, scale_actions=scale_actions),
            )
            env.simulator.simulate_at_each_physics_step()
            env.simulator.refresh_sim_tensors()
            env.render()

            should_print = sim_step_idx % control_decimation == 0 or sim_step_idx == args.sim_steps
            if should_print and control_step_idx % args.print_every == 0:
                _print_step_summary(control_step_idx, sim_step_idx, env, term)

        print(
            "final:"
            f" state={term.curr_state[0] if hasattr(term, 'curr_state') else None}"
            f" desired_state={_wbc_desired_state(term)}"
            f" bootstrap_hold={_wbc_bootstrap_hold(term)}"
            f" action_batch_dim={_wbc_action_batch_dim(term)}"
            f" torque={_format_tensor_row(term.torques[0])}"
            f" right_basis={_format_tensor_row(env.simulator.right_foot_contact_basis[0])}"
            f" right_grf={_format_tensor_row(_foot_grf(env, term, 'right'))}"
            f" left_basis={_format_tensor_row(env.simulator.left_foot_contact_basis[0])}"
            f" left_grf={_format_tensor_row(_foot_grf(env, term, 'left'))}"
        )
        print("PASS: commanded one standing G1 environment through torque_control WBC.")
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user; shutting down simulation.")
        return 130
    finally:
        if hasattr(env, "close"):
            env.close()
        close_simulation_app(simulation_app)

if __name__ == "__main__":
    raise SystemExit(main())
