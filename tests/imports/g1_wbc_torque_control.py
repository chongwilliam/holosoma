from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

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
from holosoma.config_values.loco.g1 import experiment as g1_loco_experiment
from holosoma.managers.action.terms.torque_control import root_state_to_base_velocity, root_state_to_xyz_rpy
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
    return ActionManagerCfg(
        terms={
            "torque_control": ActionTermCfg(
                func="holosoma.managers.action.terms.torque_control:JointTorqueActionTerm",
                params={
                    "robot_file": robot_file,
                    "yaml_file": yaml_file,
                    "robot_name": robot_name,
                },
                scale=1.0,
                clip=None,
                wbc_extension_dir=wbc_extension_dir,
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

    base_config = g1_loco_experiment.g1_29dof
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

    env.reset_envs_idx(env_ids, target_states={"root_states": root_states, "dof_states": dof_states})
    env.simulator.set_actor_root_state_tensor_robots(env_ids, env.simulator.robot_root_states)
    env.simulator.set_dof_state_tensor_robots(env_ids, dof_states)
    env.simulator.refresh_sim_tensors()


def _contact_count(simulator, side: str) -> int:
    return int(getattr(simulator, f"{side}_foot_contact_count")[0].item())


def _both_feet_in_contact(simulator) -> bool:
    return _contact_count(simulator, "right") > 0 and _contact_count(simulator, "left") > 0


def _foot_grf(env, term, side: str) -> torch.Tensor:
    grf_fn = getattr(term, "_foot_ground_reaction_wrench", None)
    if callable(grf_fn):
        try:
            return torch.as_tensor(grf_fn(0, side), device=env.device, dtype=torch.float32)
        except RuntimeError:
            exit(0)
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


def _desired_com_position(wbc_engine, com_action: torch.Tensor, device: str) -> torch.Tensor:
    return _current_com_position(wbc_engine, device) + com_action.to(device=device, dtype=torch.float32)


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


def _apply_hold_torques(
    simulator,
    target_q: torch.Tensor,
    kp: torch.Tensor,
    kd: torch.Tensor,
    torque_limits: torch.Tensor,
) -> None:
    dof_pos = _state_tensor(simulator.dof_pos).to(device=target_q.device, dtype=target_q.dtype)
    dof_vel = _state_tensor(simulator.dof_vel).to(device=target_q.device, dtype=target_q.dtype)
    torques = kp * (target_q - dof_pos) - kd * dof_vel
    torques = torch.clamp(torques, min=-torque_limits, max=torque_limits)
    simulator.apply_torques_at_dof(torques)


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
    print(
        f"control_step={control_step_idx}"
        f" sim_step={sim_step_idx}"
        f" sim_time={float(env.simulator.time()):.6f}"
        f" state={curr_state}"
        f" root_pos={_format_tensor_row(_root_pos(env))}"
        f" torque_norm={float(torch.linalg.vector_norm(torques).item()):.6f}"
        f" torque_max={float(torch.max(torch.abs(torques)).item()):.6f}"
        f" right_count={_contact_count(env.simulator, 'right')}"
        f" right_grf={_format_tensor_row(_foot_grf(env, term, 'right'))}"
        f" left_count={_contact_count(env.simulator, 'left')}"
        f" left_grf={_format_tensor_row(_foot_grf(env, term, 'left'))}"
    )


def main() -> int:
    humanoid_control_root = REPO_ROOT.parent / "humanoid-control"
    default_extension_dir = humanoid_control_root / "build"
    default_holosoma_robot_file = REPO_ROOT / "src" / "holosoma" / "holosoma" / "data" / "robots" / "g1" / "g1_29dof.urdf"
    default_wbc_robot_file = _existing_default(
        humanoid_control_root / "models" / "unitree_g1" / "g1.urdf",
        default_holosoma_robot_file,
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
    parser.add_argument("--headless", action="store_true", help="Disable the simulator visualization window.")
    parser.add_argument("--disable-com-visualization", action="store_true", help="Do not draw the desired COM marker.")
    parser.add_argument("--com-marker-radius", type=float, default=0.025, help="COM marker radius in meters.")
    parser.add_argument(
        "--com-amplitude-xyz",
        type=float,
        nargs=3,
        default=(0.02, 0.02, 0.01),
        metavar=("X", "Y", "Z"),
        help="Sinusoidal COM command amplitude in meters for x y z.",
    )
    parser.add_argument(
        "--com-frequency-xyz",
        type=float,
        nargs=3,
        default=(0.25, 0.25, 0.5),
        metavar=("X", "Y", "Z"),
        help="Sinusoidal COM command frequency in Hz for x y z.",
    )
    parser.add_argument(
        "--com-phase-xyz",
        type=float,
        nargs=3,
        default=(0.0, math.pi / 2.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Sinusoidal COM command phase in radians for x y z.",
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
    if any(amplitude < 0.0 for amplitude in args.com_amplitude_xyz):
        raise ValueError("--com-amplitude-xyz values must be non-negative.")
    if any(frequency < 0.0 for frequency in args.com_frequency_xyz):
        raise ValueError("--com-frequency-xyz values must be non-negative.")
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
    
        wbc_engine = term.wbc[0] if isinstance(term.wbc, list) else term.wbc
        wbc_dof = int(wbc_engine.dof())
        # if wbc_dof != env.num_dof:
        #     raise RuntimeError(
        #         f"WBC dof mismatch: env.num_dof={env.num_dof}, wbc.dof()={wbc_dof}. "
        #         "Pass matching --robot-file/--yaml-file/--robot-name for G1."
        #     )
    
        right_count = _contact_count(env.simulator, "right")
        left_count = _contact_count(env.simulator, "left")
        print(
            "initial:"
            f" simulator={args.simulator}"
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
        # if right_count <= 0 or left_count <= 0:
        #     raise RuntimeError("G1 did not start in dual-foot contact after reset_all().")
    
        control_decimation = int(env.simulator.simulator_config.sim.control_decimation)
        control_decimation = 1
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
        while not _both_feet_in_contact(env.simulator) and hold_step_idx < args.contact_hold_max_steps:
            hold_step_idx += 1
            _apply_hold_torques(env.simulator, hold_target_q, hold_kp, hold_kd, hold_torque_limits)
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
            "switch_to_wbc:"
            f" hold_steps={hold_step_idx}"
            f" right_count={_contact_count(env.simulator, 'right')}"
            f" left_count={_contact_count(env.simulator, 'left')}"
        )
    
        # update + reinitialize
        root_state = env.simulator.robot_root_states[0]
        dof_pos = _state_tensor(env.simulator.dof_pos)[0].to(device=root_state.device, dtype=root_state.dtype)
        dof_vel = _state_tensor(env.simulator.dof_vel)[0].to(device=root_state.device, dtype=root_state.dtype)
        q_tensor = torch.cat([root_state_to_xyz_rpy(root_state), dof_pos], dim=0)
        dq_tensor = torch.cat([root_state_to_base_velocity(root_state), dof_vel], dim=0)
        q = q_tensor.detach().cpu().numpy()
        dq = dq_tensor.detach().cpu().numpy()
        wbc_engine.updateRobot(q, dq)
        wbc_engine.reInitializeAllTasks()
    
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device, dtype=torch.float32)
        com_amplitude = torch.tensor(args.com_amplitude_xyz, device=device, dtype=torch.float32)
        com_frequency = torch.tensor(args.com_frequency_xyz, device=device, dtype=torch.float32)
        com_phase = torch.tensor(args.com_phase_xyz, device=device, dtype=torch.float32)
        control_step_idx = 0
        for sim_step_idx in range(1, args.sim_steps + 1):
            if (sim_step_idx - 1) % control_decimation == 0:
                control_step_idx += 1
                sim_time = float(env.simulator.time())
                actions.zero_()
                actions[:, :3] = com_amplitude * torch.sin(2.0 * math.pi * com_frequency * sim_time + com_phase)
                if not args.disable_com_visualization:
                    current_com = _current_com_position(wbc_engine, device)
                    desired_com = current_com + actions[0, :3]
                    _draw_com_markers(env, current_com, desired_com, args.com_marker_radius)
                env.action_manager.process_actions(actions)
    
            env.action_manager.apply_actions()
            env.simulator.simulate_at_each_physics_step()
            env.simulator.refresh_sim_tensors()
            env.render()
    
            should_print = sim_step_idx % control_decimation == 0 or sim_step_idx == args.sim_steps
            if should_print and control_step_idx % args.print_every == 0:
                _print_step_summary(control_step_idx, sim_step_idx, env, term)
    
        print(
            "final:"
            f" state={term.curr_state[0] if hasattr(term, 'curr_state') else None}"
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
