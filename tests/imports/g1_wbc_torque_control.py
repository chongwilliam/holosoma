from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg
from holosoma.config_types.logger import DisabledLoggerConfig
from holosoma.config_types.video import VideoConfig
from holosoma.config_values import simulator as simulator_defaults
from holosoma.config_values import terrain as terrain_defaults
from holosoma.config_values.loco.g1 import experiment as g1_loco_experiment
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
        "isaacsim": simulator_defaults.isaacsim,
    }[simulator_name]

    base_config = g1_loco_experiment.g1_29dof
    disabled_logger = DisabledLoggerConfig(video=VideoConfig(enabled=False), base_dir="logs")
    standing_robot = dataclasses.replace(
        base_config.robot,
        init_state=dataclasses.replace(base_config.robot.init_state, pos=[0.0, 0.0, 0.76]),
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


def _contact_count(simulator, side: str) -> int:
    return int(getattr(simulator, f"{side}_foot_contact_count")[0].item())


def _print_step_summary(step_idx: int, env, term) -> None:
    torques = term.torques[0]
    curr_state = term.curr_state[0] if hasattr(term, "curr_state") else None
    print(
        f"step={step_idx}"
        f" sim_time={float(env.simulator.time()):.6f}"
        f" state={curr_state}"
        f" torque_norm={float(torch.linalg.vector_norm(torques).item()):.6f}"
        f" torque_max={float(torch.max(torch.abs(torques)).item()):.6f}"
        f" right_count={_contact_count(env.simulator, 'right')}"
        f" left_count={_contact_count(env.simulator, 'left')}"
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
        description="Run one G1 environment with the torque_control WBC action term from a dual-foot standing start."
    )
    parser.add_argument("--simulator", choices=("mujoco", "isaacsim"), default="mujoco")
    parser.add_argument("--steps", type=int, default=64, help="Number of control steps to command with zero WBC actions.")
    parser.add_argument("--print-every", type=int, default=1, help="Print summary every N control steps.")
    parser.add_argument("--device", default=None, help="Override simulation device.")
    parser.add_argument("--viewer", action="store_true", help="Enable the simulator viewer.")
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

    if args.steps < 0:
        raise ValueError("--steps must be non-negative.")
    if args.print_every <= 0:
        raise ValueError("--print-every must be a positive integer.")

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
        headless=not args.viewer,
        wbc_extension_dir=wbc_extension_dir,
        robot_file=robot_file,
        yaml_file=yaml_file,
        robot_name=args.robot_name,
    )

    env, device, simulation_app = setup_simulation_environment(config, device=args.device)
    try:
        env.reset_all()
        term = env.action_manager.get_term("torque_control")

        wbc_engine = term.wbc[0] if isinstance(term.wbc, list) else term.wbc
        wbc_dof = int(wbc_engine.dof())
        if wbc_dof != env.num_dof:
            raise RuntimeError(
                f"WBC dof mismatch: env.num_dof={env.num_dof}, wbc.dof()={wbc_dof}. "
                "Pass matching --robot-file/--yaml-file/--robot-name for G1."
            )

        right_count = _contact_count(env.simulator, "right")
        left_count = _contact_count(env.simulator, "left")
        print(
            "initial:"
            f" simulator={args.simulator}"
            f" device={device}"
            f" action_dim={env.action_manager.total_action_dim}"
            f" wbc_dof={wbc_dof}"
            f" right_count={right_count}"
            f" right_contact={_format_tensor_row(env.simulator.right_foot_contact_position[0])}"
            f" left_count={left_count}"
            f" left_contact={_format_tensor_row(env.simulator.left_foot_contact_position[0])}"
        )
        if right_count <= 0 or left_count <= 0:
            raise RuntimeError("G1 did not start in dual-foot contact after reset_all().")

        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device, dtype=torch.float32)
        for step_idx in range(1, args.steps + 1):
            env.step({"actions": actions})
            if step_idx % args.print_every == 0:
                _print_step_summary(step_idx, env, term)

        print(
            "final:"
            f" state={term.curr_state[0] if hasattr(term, 'curr_state') else None}"
            f" torque={_format_tensor_row(term.torques[0])}"
            f" right_basis={_format_tensor_row(env.simulator.right_foot_contact_basis[0])}"
            f" left_basis={_format_tensor_row(env.simulator.left_foot_contact_basis[0])}"
        )
        print("PASS: commanded one standing G1 environment through torque_control WBC.")
        return 0
    finally:
        if hasattr(env, "close"):
            env.close()
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    raise SystemExit(main())
