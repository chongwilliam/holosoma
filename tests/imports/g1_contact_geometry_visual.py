from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma.config_types.logger import DisabledLoggerConfig
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.video import VideoConfig
from holosoma.config_values import robot as robot_defaults
from holosoma.config_values import simulator as simulator_defaults
from holosoma.config_values import terrain as terrain_defaults
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.torch_utils import to_torch


def _build_config(simulator_name: str, headless: bool, device: str | None) -> RunSimConfig:
    simulator_cfg = {
        "mujoco": simulator_defaults.mujoco,
        "mjwarp": simulator_defaults.mjwarp,
        "isaacsim": simulator_defaults.isaacsim,
    }[simulator_name]

    training_cfg = dataclasses.replace(
        RunSimConfig().training,
        num_envs=1,
        headless=headless,
    )
    disabled_logger = DisabledLoggerConfig(video=VideoConfig(enabled=False), base_dir="logs")
    return RunSimConfig(
        simulator=simulator_cfg,
        robot=robot_defaults.g1_29dof,
        terrain=terrain_defaults.terrain_locomotion_plane,
        training=training_cfg,
        logger=disabled_logger,
        device=device,
    )


def _create_base_init_state(config: RunSimConfig, device: str) -> torch.Tensor:
    base_init_state_list = (
        config.robot.init_state.pos
        + config.robot.init_state.rot
        + config.robot.init_state.lin_vel
        + config.robot.init_state.ang_vel
    )
    return to_torch(base_init_state_list, device=device, requires_grad=False)


def _initialize_simulator(simulator, config: RunSimConfig, device: str) -> None:
    simulator.set_headless(config.training.headless)
    simulator.setup()
    simulator.setup_terrain()
    simulator.load_assets()

    env_origins = torch.zeros(1, 3, device=device)
    base_init_state = _create_base_init_state(config, device)
    simulator.create_envs(1, env_origins, base_init_state)
    simulator.prepare_sim()
    simulator.on_episode_start(env_id=0)

    if not config.training.headless:
        simulator.setup_viewer()


def _build_hold_controller(
    config: RunSimConfig, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_q = torch.tensor(
        [config.robot.init_state.default_joint_angles[name] for name in config.robot.dof_names],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    kp = torch.zeros(len(config.robot.dof_names), device=device, dtype=torch.float32)
    kd = torch.zeros_like(kp)
    stiffness = config.robot.control.stiffness
    damping = config.robot.control.damping
    for idx, dof_name in enumerate(config.robot.dof_names):
        dof_key = dof_name.replace("_joint", "")
        for pattern, gain in stiffness.items():
            if pattern in dof_key:
                kp[idx] = gain
                kd[idx] = damping[pattern]
                break

    torque_limits = torch.tensor(config.robot.dof_effort_limit_list, device=device, dtype=torch.float32).unsqueeze(0)
    return target_q, kp.unsqueeze(0), kd.unsqueeze(0), torque_limits


def _apply_hold_torques(
    simulator,
    target_q: torch.Tensor,
    kp: torch.Tensor,
    kd: torch.Tensor,
    torque_limits: torch.Tensor,
) -> None:
    position_error = simulator.dof_pos - target_q
    velocity_error = simulator.dof_vel
    torques = torch.clamp(-(kp * position_error + kd * velocity_error), min=-torque_limits, max=torque_limits)
    simulator.apply_torques_at_dof(torques)


def _quat_xyzw_from_rpy(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    cr = math.cos(0.5 * roll)
    sr = math.sin(0.5 * roll)
    cp = math.cos(0.5 * pitch)
    sp = math.sin(0.5 * pitch)
    cy = math.cos(0.5 * yaw)
    sy = math.sin(0.5 * yaw)
    return torch.tensor(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=torch.float32,
    )


def _quat_apply_xyzw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    q_xyz = quat[:3]
    q_w = quat[3]
    uv = torch.cross(q_xyz, vec, dim=0)
    uuv = torch.cross(q_xyz, uv, dim=0)
    return vec + 2.0 * (q_w * uv + uuv)


def _scenario_table(height: float) -> list[dict[str, float | str]]:
    deg = math.pi / 180.0
    return [
        {"name": "flat", "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "height": height},
        {"name": "toe_edge_pitch_forward", "roll": 0.0, "pitch": 8.0 * deg, "yaw": 0.0, "height": height},
        {"name": "heel_edge_pitch_back", "roll": 0.0, "pitch": -8.0 * deg, "yaw": 0.0, "height": height},
        {"name": "left_edge_roll", "roll": 8.0 * deg, "pitch": 0.0, "yaw": 0.0, "height": height},
        {"name": "right_edge_roll", "roll": -8.0 * deg, "pitch": 0.0, "yaw": 0.0, "height": height},
        {"name": "yawed_flat", "roll": 0.0, "pitch": 0.0, "yaw": 20.0 * deg, "height": height},
    ]


def _set_robot_state(simulator, config: RunSimConfig, scenario: dict[str, float | str], device: str) -> None:
    env_ids = torch.arange(1, device=device)
    root_state = _create_base_init_state(config, device).unsqueeze(0).clone()
    root_state[:, 0:3] = torch.tensor([0.0, 0.0, float(scenario["height"])], device=device, dtype=torch.float32)
    root_state[:, 3:7] = _quat_xyzw_from_rpy(
        float(scenario["roll"]),
        float(scenario["pitch"]),
        float(scenario["yaw"]),
    ).to(device=device)
    root_state[:, 7:13] = 0.0

    dof_pos = torch.tensor(
        [config.robot.init_state.default_joint_angles[name] for name in config.robot.dof_names],
        device=device,
        dtype=torch.float32,
    ).reshape(1, -1)
    dof_vel = torch.zeros_like(dof_pos)
    dof_states = torch.stack([dof_pos, dof_vel], dim=-1)

    simulator.robot_root_states[:] = root_state
    simulator.dof_pos[:] = dof_pos
    simulator.dof_vel[:] = dof_vel
    simulator.set_actor_root_state_tensor_robots(env_ids, root_state)
    simulator.set_dof_state_tensor_robots(env_ids, dof_states)
    simulator.refresh_sim_tensors()


def _format_tensor_row(row: torch.Tensor) -> str:
    values = [float(x) for x in row.detach().cpu().tolist()]
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def _resolve_foot_body_indices(simulator, foot_body_name: str) -> dict[str, int]:
    indices: dict[str, int] = {}
    body_names = list(getattr(simulator, "body_names", []))
    for side in ("right", "left"):
        exact_name = f"{side}_{foot_body_name}"
        match = next((idx for idx, name in enumerate(body_names) if name == exact_name), None)
        if match is None:
            match = next((idx for idx, name in enumerate(body_names) if side in name.lower() and foot_body_name in name), None)
        if match is None:
            raise RuntimeError(f"Could not resolve {side} foot body for foot_body_name={foot_body_name!r}.")
        indices[side] = int(match)
    return indices


def _mapped_body_index(simulator, body_idx: int) -> int:
    body_map = getattr(simulator, "holosoma_to_mujoco_body_map", None)
    if isinstance(body_map, dict):
        return int(body_map.get(int(body_idx), int(body_idx)))
    return int(body_idx)


def _draw_contact_geometry(simulator, config: RunSimConfig, foot_body_indices: dict[str, int]) -> None:
    if hasattr(simulator, "clear_lines"):
        simulator.clear_lines()
    if not hasattr(simulator, "draw_sphere"):
        return

    foot_center = torch.tensor(config.robot.foot_center, device=simulator.device, dtype=torch.float32)
    foot_dimension = torch.tensor(config.robot.foot_dimension, device=simulator.device, dtype=torch.float32)
    half_length = 0.5 * float(foot_dimension[0].item())
    half_width = 0.5 * float(foot_dimension[1].item())
    local_corners = torch.tensor(
        [
            [foot_center[0] + half_length, foot_center[1] + half_width, foot_center[2]],
            [foot_center[0] + half_length, foot_center[1] - half_width, foot_center[2]],
            [foot_center[0] - half_length, foot_center[1] - half_width, foot_center[2]],
            [foot_center[0] - half_length, foot_center[1] + half_width, foot_center[2]],
        ],
        device=simulator.device,
        dtype=torch.float32,
    )

    axis_colors = (
        torch.tensor([1.0, 0.0, 0.0], device=simulator.device),
        torch.tensor([0.0, 0.85, 0.0], device=simulator.device),
        torch.tensor([0.1, 0.25, 1.0], device=simulator.device),
    )
    disabled_color = torch.tensor([0.25, 0.25, 0.25], device=simulator.device)

    for side in ("right", "left"):
        body_idx = _mapped_body_index(simulator, foot_body_indices[side])
        foot_pos = simulator._rigid_body_pos[0, body_idx]
        foot_quat = simulator._rigid_body_rot[0, body_idx]
        local_contact = getattr(simulator, f"{side}_foot_contact_position")[0].to(
            device=simulator.device, dtype=torch.float32
        )
        angular_basis = getattr(simulator, f"{side}_foot_contact_basis")[0].to(
            device=simulator.device, dtype=torch.float32
        )
        contact_count = int(getattr(simulator, f"{side}_foot_contact_count")[0].item())

        world_contact = foot_pos + _quat_apply_xyzw(foot_quat, local_contact)
        point_color = torch.tensor([1.0, 0.85, 0.0], device=simulator.device) if side == "right" else torch.tensor(
            [0.0, 0.9, 1.0], device=simulator.device
        )
        simulator.draw_sphere(
            world_contact.detach().cpu(),
            0.025 if contact_count else 0.015,
            point_color.detach().cpu(),
            env_id=0,
        )

        if hasattr(simulator, "draw_line"):
            world_corners = [foot_pos + _quat_apply_xyzw(foot_quat, corner) for corner in local_corners]
            outline_color = torch.tensor([0.9, 0.9, 0.9], device=simulator.device)
            for corner_idx, start in enumerate(world_corners):
                simulator.draw_line(
                    start.detach().cpu(),
                    world_corners[(corner_idx + 1) % 4].detach().cpu(),
                    outline_color.detach().cpu(),
                    env_id=0,
                )

            local_axes = torch.eye(3, device=simulator.device, dtype=torch.float32)
            for axis_idx in range(3):
                axis_world = _quat_apply_xyzw(foot_quat, local_axes[axis_idx])
                scale = 0.18 if axis_idx < 2 else 0.12
                color = axis_colors[axis_idx] if float(angular_basis[axis_idx].item()) > 0.5 else disabled_color
                simulator.draw_line(
                    world_contact.detach().cpu(),
                    (world_contact + scale * axis_world).detach().cpu(),
                    color.detach().cpu(),
                    env_id=0,
                )


def _contact_summary_snapshot(simulator) -> dict[str, tuple[int, torch.Tensor, torch.Tensor]]:
    return {
        side: (
            int(getattr(simulator, f"{side}_foot_contact_count")[0].item()),
            getattr(simulator, f"{side}_foot_contact_position")[0].detach().cpu().clone(),
            getattr(simulator, f"{side}_foot_contact_basis")[0].detach().cpu().clone(),
        )
        for side in ("right", "left")
    }


def _contact_summary_changed(
    current: dict[str, tuple[int, torch.Tensor, torch.Tensor]],
    previous: dict[str, tuple[int, torch.Tensor, torch.Tensor]] | None,
    tolerance: float,
) -> bool:
    if previous is None:
        return True
    for side in ("right", "left"):
        current_count, current_pos, current_basis = current[side]
        previous_count, previous_pos, previous_basis = previous[side]
        if current_count != previous_count:
            return True
        if bool(torch.any(torch.abs(current_pos - previous_pos) > tolerance)):
            return True
        if bool(torch.any(torch.abs(current_basis - previous_basis) > tolerance)):
            return True
    return False


def _print_contact_summary(
    simulator,
    scenario_name: str,
    snapshot: dict[str, tuple[int, torch.Tensor, torch.Tensor]] | None = None,
) -> None:
    if snapshot is None:
        snapshot = _contact_summary_snapshot(simulator)
    print(f"scenario={scenario_name}")
    for side in ("right", "left"):
        contact_count, contact_position, contact_basis = snapshot[side]
        print(
            f"  {side}:"
            f" count={contact_count}"
            f" contact_position={_format_tensor_row(contact_position)}"
            f" contact_angular_basis={_format_tensor_row(contact_basis)}"
        )


def _render_frame(simulator, frame_dt: float) -> None:
    if getattr(simulator, "viewer", None) is None:
        return
    simulator.render(sync_frame_time=False)
    if frame_dt > 0.0:
        time.sleep(frame_dt)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize G1 foot contact_position and contact_angular_basis across tilted contact configurations."
    )
    parser.add_argument("--simulator", choices=("mujoco", "mjwarp", "isaacsim"), default="mujoco")
    parser.add_argument("--device", default=None, help="Override simulation device.")
    parser.add_argument("--headless", action="store_true", help="Run without viewer; still prints contact summaries.")
    parser.add_argument("--height", type=float, default=0.793, help="Root height used when resetting each scenario.")
    parser.add_argument("--settle-steps", type=int, default=80, help="Physics steps to settle before displaying a scenario.")
    parser.add_argument("--display-steps", type=int, default=180, help="Physics steps to display each scenario.")
    parser.add_argument("--frame-dt", type=float, default=1.0 / 60.0, help="Wall-clock delay after each rendered display frame.")
    parser.add_argument("--print-tol", type=float, default=1.0e-4, help="Tolerance used when deciding contact summary changes.")
    parser.add_argument("--cycles", type=int, default=1, help="Number of passes through all scenarios.")
    parser.add_argument("--no-hold", action="store_true", help="Do not apply the default-joint PD hold controller.")
    args = parser.parse_args()

    config = _build_config(
        simulator_name=args.simulator,
        headless=args.headless,
        device=args.device,
    )
    env, device, simulation_app = setup_simulation_environment(config, device=config.device)
    simulator = env.sim

    try:
        _initialize_simulator(simulator, config, device)
        target_q, kp, kd, torque_limits = _build_hold_controller(config, device)
        foot_body_indices = _resolve_foot_body_indices(simulator, config.robot.foot_body_name)
        scenarios = _scenario_table(args.height)
        last_printed_snapshot: dict[str, tuple[int, torch.Tensor, torch.Tensor]] | None = None

        for _cycle in range(args.cycles):
            for scenario in scenarios:
                _set_robot_state(simulator, config, scenario, device)

                for _ in range(args.settle_steps):
                    if not args.no_hold:
                        _apply_hold_torques(simulator, target_q, kp, kd, torque_limits)
                    simulator.simulate_at_each_physics_step()
                    simulator.refresh_sim_tensors()

                snapshot = _contact_summary_snapshot(simulator)
                if _contact_summary_changed(snapshot, last_printed_snapshot, args.print_tol):
                    _print_contact_summary(simulator, str(scenario["name"]), snapshot)
                    last_printed_snapshot = snapshot
                _draw_contact_geometry(simulator, config, foot_body_indices)
                _render_frame(simulator, args.frame_dt)

                for _ in range(args.display_steps):
                    if not args.no_hold:
                        _apply_hold_torques(simulator, target_q, kp, kd, torque_limits)
                    simulator.simulate_at_each_physics_step()
                    simulator.refresh_sim_tensors()
                    snapshot = _contact_summary_snapshot(simulator)
                    if _contact_summary_changed(snapshot, last_printed_snapshot, args.print_tol):
                        _print_contact_summary(simulator, str(scenario["name"]), snapshot)
                        last_printed_snapshot = snapshot
                    _draw_contact_geometry(simulator, config, foot_body_indices)
                    _render_frame(simulator, args.frame_dt)

        print("PASS: completed G1 contact geometry visualization scenarios")
        return 0
    finally:
        if hasattr(env, "close"):
            env.close()
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    raise SystemExit(main())
