from __future__ import annotations

import argparse
import dataclasses
import sys
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


def _format_tensor_row(row) -> str:
    values = [float(x) for x in row.detach().cpu().tolist()]
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def _basis_is_nonzero(row: torch.Tensor) -> bool:
    return bool(torch.any(row != 0))


def _create_base_init_state(config: RunSimConfig, device: str) -> torch.Tensor:
    base_init_state_list = (
        config.robot.init_state.pos
        + config.robot.init_state.rot
        + config.robot.init_state.lin_vel
        + config.robot.init_state.ang_vel
    )
    return to_torch(base_init_state_list, device=device, requires_grad=False)


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


def _apply_hold_torques(
    simulator,
    target_q: torch.Tensor,
    kp: torch.Tensor,
    kd: torch.Tensor,
    torque_limits: torch.Tensor,
) -> None:
    position_error = simulator.dof_pos - target_q
    velocity_error = simulator.dof_vel
    torques = kp * position_error + kd * velocity_error
    torques = torch.clamp(- torques, min=-torque_limits, max=torque_limits)
    simulator.apply_torques_at_dof(torques)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test G1 foot contact support summaries.")
    parser.add_argument("--simulator", choices=("mujoco", "isaacsim"), default="mujoco")
    parser.add_argument("--steps", type=int, default=32, help="Number of settle steps before reading contacts.")
    parser.add_argument("--device", default=None, help="Override simulation device.")
    parser.add_argument("--viewer", action="store_true", help="Enable the simulator viewer.")
    args = parser.parse_args()

    config = _build_config(
        simulator_name=args.simulator,
        headless=not args.viewer,
        device=args.device,
    )

    env, device, simulation_app = setup_simulation_environment(config, device=config.device)
    simulator = env.sim

    try:
        _initialize_simulator(simulator, config, device)
        target_q, kp, kd, torque_limits = _build_hold_controller(config, device)

        simulator.refresh_sim_tensors()
        for _ in range(args.steps):
            _apply_hold_torques(simulator, target_q, kp, kd, torque_limits)
            simulator.simulate_at_each_physics_step()
            simulator.refresh_sim_tensors()

            if _basis_is_nonzero(simulator.right_foot_contact_basis[0]):
                print(
                    "right_contact:"
                    f" count={int(simulator.right_foot_contact_count[0].item())}"
                    f" barycenter={_format_tensor_row(simulator.right_foot_contact_position[0])}"
                    f" basis={_format_tensor_row(simulator.right_foot_contact_basis[0])}"
                )

            if _basis_is_nonzero(simulator.left_foot_contact_basis[0]):
                print(
                    "left_contact:"
                    f" count={int(simulator.left_foot_contact_count[0].item())}"
                    f" barycenter={_format_tensor_row(simulator.left_foot_contact_position[0])}"
                    f" basis={_format_tensor_row(simulator.left_foot_contact_basis[0])}"
                )

        print(f"simulator: {args.simulator}")
        print(f"steps: {args.steps}")
        print(f"right_count: {int(simulator.right_foot_contact_count[0].item())}")
        print(f"right_barycenter: {_format_tensor_row(simulator.right_foot_contact_position[0])}")
        print(f"right_basis: {_format_tensor_row(simulator.right_foot_contact_basis[0])}")
        print(f"left_count: {int(simulator.left_foot_contact_count[0].item())}")
        print(f"left_barycenter: {_format_tensor_row(simulator.left_foot_contact_position[0])}")
        print(f"left_basis: {_format_tensor_row(simulator.left_foot_contact_basis[0])}")

        if (
            int(simulator.right_foot_contact_count[0].item()) == 0
            and int(simulator.left_foot_contact_count[0].item()) == 0
        ):
            print("FAIL: no foot contacts detected")
            return 1

        print("PASS: retrieved G1 foot contact support information")
        return 0
    finally:
        if hasattr(env, "close"):
            env.close()
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    raise SystemExit(main())
