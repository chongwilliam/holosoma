"""Run G1 in-place stepping with the batched torque-control WBC path.

The script starts from the same dual-contact standing reset used by
g1_wbc_torque_control.py, then advances the locomotion gait phase clock while
sending zero task-space velocity/residual actions.  The desired WBC stance is
therefore selected from the gait phase rather than being forced to dual stance.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PACKAGE_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from holosoma.config_values import simulator as simulator_defaults
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment

from g1_wbc_torque_control import (  # noqa: E402
    _apply_hold_torques,
    _apply_wbc_torques,
    _batch_state_at,
    _batch_transition_start_at,
    _both_feet_in_contact,
    _build_config,
    _build_hold_controller,
    _compute_hold_torques,
    _contact_count,
    _existing_default,
    _explicitly_update_wbc_robot_state,
    _foot_grf,
    _force_single_env_origin_at_world,
    _force_wbc_dual_stance,
    _format_tensor_row,
    _reinitialize_wbc_tasks,
    _reset_wbc_targets_from_current_pose,
    _reset_without_physics_step,
    _resolve_path,
    _root_pos,
    _wbc_action_batch_dim,
    _wbc_bootstrap_hold,
    _wbc_desired_state,
    _zero_reset_velocities,
)


def _wrap_phase(phase: torch.Tensor) -> torch.Tensor:
    return torch.remainder(phase + math.pi, 2.0 * math.pi) - math.pi


def _set_gait_clock(env, frequency_hz: float, phase_rad: float) -> None:
    command_manager = getattr(env, "command_manager", None)
    get_state = getattr(command_manager, "get_state", None)
    gait_state = get_state("locomotion_gait") if callable(get_state) else None
    if gait_state is None:
        raise RuntimeError("Environment command manager does not expose locomotion_gait state.")

    phase = getattr(gait_state, "phase", None)
    phase_offset = getattr(gait_state, "phase_offset", None)
    gait_freq = getattr(gait_state, "gait_freq", None)
    phase_dt = getattr(gait_state, "phase_dt", None)
    if phase is None or gait_freq is None or phase_dt is None:
        raise RuntimeError("locomotion_gait is missing phase, gait_freq, or phase_dt buffers.")

    left_phase = torch.full((env.num_envs,), float(phase_rad), device=env.device, dtype=phase.dtype)
    right_phase = left_phase - math.pi
    phase_values = torch.stack([_wrap_phase(left_phase), _wrap_phase(right_phase)], dim=1)
    phase[:] = phase_values
    if phase_offset is not None:
        phase_offset[:] = phase_values
    gait_freq[:] = float(frequency_hz)
    phase_dt[:] = 2.0 * math.pi * float(env.dt) * float(frequency_hz)


def _zero_locomotion_commands(env) -> None:
    command_manager = getattr(env, "command_manager", None)
    commands = getattr(command_manager, "commands", None)
    if commands is not None:
        commands.zero_()


def _release_wbc_from_dual_hold(term) -> None:
    for attr_name in ("_forced_wbc_desired_state", "_fixed_wbc_foot_target_poses"):
        if hasattr(term, attr_name):
            delattr(term, attr_name)

    phase_shift = getattr(term, "_wbc_phase_shift_frac", None)
    if isinstance(phase_shift, torch.Tensor):
        phase_shift.zero_()

    pending_reinit = getattr(term, "_pending_wbc_reinitialize", None)
    if isinstance(pending_reinit, torch.Tensor):
        pending_reinit.zero_()

    bootstrap_done = getattr(term, "_wbc_bootstrap_done", None)
    if isinstance(bootstrap_done, torch.Tensor):
        bootstrap_done.fill_(True)

    bootstrap_hold_mask = getattr(term, "_last_wbc_bootstrap_hold_mask", None)
    if isinstance(bootstrap_hold_mask, torch.Tensor):
        bootstrap_hold_mask.zero_()


def _print_step_summary(control_step_idx: int, sim_step_idx: int, env, term) -> None:
    torques = term.torques[0]
    curr_state = term.curr_state[0] if hasattr(term, "curr_state") else None
    print(
        f"control_step={control_step_idx}"
        f" sim_step={sim_step_idx}"
        f" sim_time={float(env.simulator.time()):.6f}"
        f" state={curr_state}"
        f" batch_state={_batch_state_at(term, 0)}"
        f" batch_transition_start={_batch_transition_start_at(term, 0)}"
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


def main() -> int:
    humanoid_control_root = REPO_ROOT.parent / "humanoid-control"
    default_extension_dir = humanoid_control_root / "build"
    default_wbc_robot_file = _existing_default(humanoid_control_root / "models" / "unitree_g1" / "g1.urdf")
    default_yaml_file = humanoid_control_root / "params" / "g1_parameters.yaml"

    parser = argparse.ArgumentParser(description="Run G1 in-place stepping through the batched WBC controller.")
    parser.add_argument("--sim-steps", type=int, default=None, help="Physics steps to run; defaults to 10 seconds.")
    parser.add_argument(
        "--contact-hold-max-steps",
        type=int,
        default=None,
        help="Maximum PD hold simulator steps before requiring both feet contact.",
    )
    parser.add_argument("--gait-frequency", type=float, default=1.0, help="Gait phase frequency in Hz.")
    parser.add_argument("--phase", type=float, default=math.pi, help="Initial left-foot gait phase in radians.")
    parser.add_argument("--dual-stance-wait-s", type=float, default=1.0, help="WBC dual-stance wait before stepping.")
    parser.add_argument("--print-every", type=int, default=1, help="Print every N control steps.")
    parser.add_argument("--device", default=None, help="Override simulation device.")
    parser.add_argument("--simulator", choices=("mujoco", "mjwarp", "isaacsim"), default="mjwarp")
    parser.add_argument("--headless", action="store_true", help="Disable simulator visualization.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--wbc-extension-dir", default=str(default_extension_dir), help="Built humanoid-control dir.")
    parser.add_argument("--robot-file", default=default_wbc_robot_file, help="Robot file passed to the WBC engine.")
    parser.add_argument("--yaml-file", default=str(default_yaml_file), help="WBC YAML configuration file.")
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
        raise ValueError("--print-every must be positive.")
    if args.gait_frequency <= 0.0:
        raise ValueError("--gait-frequency must be positive.")
    if args.dual_stance_wait_s < 0.0:
        raise ValueError("--dual-stance-wait-s must be non-negative.")
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
        print("Starting G1 in-place stepping with batched WBC")
        _force_single_env_origin_at_world(env)
        _reset_without_physics_step(env)
        _zero_reset_velocities(env)
        _zero_locomotion_commands(env)
        _set_gait_clock(env, args.gait_frequency, args.phase)
        term = env.action_manager.get_term("torque_control")

        control_decimation = int(env.simulator.simulator_config.sim.control_decimation)
        control_steps = (args.sim_steps + control_decimation - 1) // control_decimation
        print(
            "run:"
            f" simulator={args.simulator}"
            f" device={device}"
            f" headless={args.headless}"
            f" action_dim={env.action_manager.total_action_dim}"
            f" gait_frequency={args.gait_frequency:.6f}"
            f" dual_stance_wait_s={args.dual_stance_wait_s:.6f}"
            f" requested_sim_steps={args.sim_steps}"
            f" control_decimation={control_decimation}"
            f" control_steps={control_steps}"
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

        if not _both_feet_in_contact(env.simulator):
            raise RuntimeError(
                "Both feet did not reach contact during the hold phase: "
                f"hold_steps={hold_step_idx}"
                f" right_count={_contact_count(env.simulator, 'right')}"
                f" left_count={_contact_count(env.simulator, 'left')}"
            )

        print(
            f"hold_ready: hold_steps={hold_step_idx}"
            f" right_count={_contact_count(env.simulator, 'right')}"
            f" left_count={_contact_count(env.simulator, 'left')}"
        )
        _zero_reset_velocities(env)
        _explicitly_update_wbc_robot_state(env, term, "batch", device)
        _force_wbc_dual_stance(term)
        _reinitialize_wbc_tasks(term)
        _reset_wbc_targets_from_current_pose(term)

        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device, dtype=torch.float32)
        warmup_torques = last_hold_torques.to(device=term.torques.device, dtype=term.torques.dtype)
        if warmup_torques.shape == term.torques.shape and bool(torch.isfinite(warmup_torques).all()):
            term.torques[:] = warmup_torques

        dual_wait_steps = int(round(args.dual_stance_wait_s * simulator_fps))
        if dual_wait_steps > 0:
            print(f"dual_stance_wait: seconds={args.dual_stance_wait_s:.6f} steps={dual_wait_steps}")
        for wait_step_idx in range(1, dual_wait_steps + 1):
            _zero_locomotion_commands(env)
            _set_gait_clock(env, args.gait_frequency, math.pi)
            _explicitly_update_wbc_robot_state(env, term, "batch", device)
            _force_wbc_dual_stance(term)
            _apply_wbc_torques(env, term, actions, "batch", scale_actions=False)
            env.simulator.simulate_at_each_physics_step()
            env.simulator.refresh_sim_tensors()
            env.render()
            if wait_step_idx == dual_wait_steps:
                print(
                    "dual_stance_wait_done:"
                    f" sim_time={float(env.simulator.time()):.6f}"
                    f" desired_state={_wbc_desired_state(term)}"
                    f" root_pos={_format_tensor_row(_root_pos(env))}"
                )

        _release_wbc_from_dual_hold(term)
        step_start_time = float(env.simulator.time())
        _set_gait_clock(env, args.gait_frequency, args.phase)
        _apply_wbc_torques(env, term, actions, "batch", scale_actions=False)

        control_step_idx = 0
        for sim_step_idx in range(1, args.sim_steps + 1):
            sim_time = float(env.simulator.time())
            step_elapsed = sim_time - step_start_time
            _set_gait_clock(env, args.gait_frequency, args.phase + 2.0 * math.pi * args.gait_frequency * step_elapsed)

            if (sim_step_idx - 1) % control_decimation == 0:
                control_step_idx += 1
                _explicitly_update_wbc_robot_state(env, term, "batch", device)

            _apply_wbc_torques(env, term, actions, "batch", scale_actions=False)
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
            f" action_batch_dim={_wbc_action_batch_dim(term)}"
            f" root_pos={_format_tensor_row(_root_pos(env))}"
            f" torque={_format_tensor_row(term.torques[0])}"
        )
        print("PASS: stepped one G1 environment in place through batched torque_control WBC.")
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
