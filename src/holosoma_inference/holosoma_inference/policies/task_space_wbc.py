"""Inference-side task-space WBC torque computation."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from holosoma_inference.config.config_types.task import TaskConfig
from holosoma_inference.utils.wbc_state import WbcStateSub


def _quaternion_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _matrix_to_urdf_floating_xyz_angles(rotation: np.ndarray) -> np.ndarray:
    sy = np.clip(rotation[0, 2], -1.0, 1.0)
    ry = np.arcsin(sy)
    rx = np.arctan2(-rotation[1, 2], rotation[2, 2])
    rz = np.arctan2(-rotation[0, 1], rotation[0, 0])
    return np.array([rx, ry, rz], dtype=np.float64)


def _root_state_to_xyz_rpy(root_state: np.ndarray) -> np.ndarray:
    rotation = _quaternion_xyzw_to_matrix(root_state[3:7])
    return np.concatenate([root_state[:3], _matrix_to_urdf_floating_xyz_angles(rotation)], axis=0)


def _yaw_to_matrix3d(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _yaw_from_matrix(rotation: np.ndarray) -> float:
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class _BatchedWbcFacade:
    """Small compatibility shim around humanoid_wbc.BatchedWbcController."""

    def __init__(self, controller: Any, dof: int, debug_engine: Any | None = None):
        self._controller = controller
        self._dof = dof
        self._debug_engine = debug_engine

    def dof(self) -> int:
        return self._dof

    def setTotalTransitionTime(self, seconds: float) -> None:
        self._controller.setTotalTransitionTime(seconds)
        if self._debug_engine is not None:
            self._debug_engine.setTotalTransitionTime(seconds)

    def reset_state(
        self,
        state: Any | None = None,
        transition_start_time: float = 0.0,
        env_id: int | None = None,
    ) -> None:
        if state is None:
            self._controller.reset_state()
            return
        if env_id is None:
            self._controller.reset_state(state, transition_start_time)
        else:
            self._controller.reset_state(state, transition_start_time, int(env_id))

    def compute_torques_batch(self, *args: Any, **kwargs: Any) -> Any:
        return self._controller.compute_torques_batch(*args, **kwargs)

    def update_robot(self, q: Any, dq: Any) -> None:
        self._controller.update_robot(q, dq)
        if self._debug_engine is not None and len(q) > 0:
            self._debug_engine.updateRobot(q[0], dq[0])

    def get_pose(self, frame_name: str, env_idx: int = 0) -> np.ndarray:
        get_pose = getattr(self._controller, "getPose", None)
        if callable(get_pose):
            return np.asarray(get_pose(int(env_idx), frame_name), dtype=np.float64)
        if self._debug_engine is not None and hasattr(self._debug_engine, "getPose"):
            return np.asarray(self._debug_engine.getPose(frame_name), dtype=np.float64)
        raise RuntimeError("humanoid_wbc controller must provide getPose for task-space inference.")

    def get_linear_velocity(self, frame_name: str, env_idx: int = 0) -> np.ndarray:
        get_linear_velocity = getattr(self._controller, "getLinearVelocity", None)
        if callable(get_linear_velocity):
            return np.asarray(get_linear_velocity(int(env_idx), frame_name), dtype=np.float64)
        if self._debug_engine is not None and hasattr(self._debug_engine, "getLinearVelocity"):
            return np.asarray(self._debug_engine.getLinearVelocity(frame_name), dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    def __getattr__(self, name: str) -> Any:
        if self._debug_engine is not None and hasattr(self._debug_engine, name):
            return getattr(self._debug_engine, name)
        return getattr(self._controller, name)


class TaskSpaceWbcTorqueComputer:
    """Compute actuated torques from 12D locomotion task-space policy actions."""

    def __init__(self, task_config: TaskConfig, num_dofs: int):
        self.task_config = task_config
        self.num_dofs = num_dofs
        self.wbc_state_sub = WbcStateSub(port=task_config.wbc_state_port)
        self.wbc_state_sub.start()

        extension_dir = self._resolve_wbc_extension_dir(task_config.wbc_extension_dir)
        params = self._resolve_wbc_params(extension_dir)
        wbc_module = self._import_wbc_module(extension_dir)
        robot_file, yaml_file, robot_name = self._resolve_wbc_asset_paths(extension_dir, params)

        batched_wbc = wbc_module.BatchedWbcController(str(robot_file), str(yaml_file), robot_name, 1)
        debug_engine = wbc_module.WbcEngine(str(robot_file), str(yaml_file), robot_name)
        self.wbc = _BatchedWbcFacade(batched_wbc, int(debug_engine.dof()), debug_engine)
        self.State = wbc_module.State
        self.SwingFoot = wbc_module.SwingFoot
        self.wbc.setTotalTransitionTime(0.15)
        self._initialized = False
        self._dt = 1.0 / float(task_config.rl_rate)

    def compute(
        self,
        scaled_task_action: np.ndarray,
        phase: np.ndarray | None = None,
        lin_vel_command: np.ndarray | None = None,
        ang_vel_command: np.ndarray | None = None,
    ) -> np.ndarray:
        state = self.wbc_state_sub.get_latest()
        if state is None:
            raise RuntimeError(
                "No WBC simulator state has been received. Start the simulator bridge with the WBC state "
                f"publisher enabled on port {self.task_config.wbc_state_port} before task-space inference."
            )

        raw_action = np.asarray(scaled_task_action, dtype=np.float64).reshape(1, -1)
        if raw_action.shape[1] != 12:
            raise ValueError(f"Task-space WBC expects a 12D action, got shape {raw_action.shape}.")
        action_scale = np.asarray(self.task_config.policy_action_scale, dtype=np.float64)
        if action_scale.ndim == 0:
            action_scale = np.full((12,), float(action_scale), dtype=np.float64)
        if action_scale.shape != (12,):
            raise ValueError(
                "task.policy_action_scale must be scalar or length-12 for task-space WBC inference. "
                f"Got shape {action_scale.shape}."
            )
        action = raw_action * action_scale.reshape(1, 12)

        root_state = np.asarray(state["root_state"], dtype=np.float64).reshape(13)
        dof_pos = np.asarray(state["dof_pos"], dtype=np.float64).reshape(self.num_dofs)
        dof_vel = np.asarray(state["dof_vel"], dtype=np.float64).reshape(self.num_dofs)
        q = np.concatenate([_root_state_to_xyz_rpy(root_state), dof_pos], axis=0).reshape(1, -1)
        dq = np.concatenate([root_state[7:13], dof_vel], axis=0).reshape(1, -1)

        right_contact_point = self._array2(state["right_contact_point"], 3)
        left_contact_point = self._array2(state["left_contact_point"], 3)
        right_contact_basis = self._basis(state["right_contact_basis"])
        left_contact_basis = self._basis(state["left_contact_basis"])
        right_grf = self._array2(state["right_grf"], 6)
        left_grf = self._array2(state["left_grf"], 6)
        right_grf = self._zero_small_grf(right_grf)
        left_grf = self._zero_small_grf(left_grf)
        right_in_contact = self._contact_mask(right_grf)
        left_in_contact = self._contact_mask(left_grf)
        sim_time = float(state.get("time", 0.0))

        if not self._initialized:
            self.wbc.update_robot(q, dq)
            self.wbc.reset_state(int(self.State.DUAL_STANCE), sim_time, 0)
            self._initialized = True

        self.wbc.update_robot(q, dq)
        desired_states = self._desired_states_from_phase(phase)
        wbc_action = self._wbc_action_from_12d(action, desired_states, lin_vel_command, ang_vel_command)
        torque_wbc = self.wbc.compute_torques_batch(
            q,
            dq,
            desired_states,
            wbc_action,
            right_contact_point,
            right_contact_basis,
            left_contact_point,
            left_contact_basis,
            right_grf,
            left_grf,
            right_in_contact,
            left_in_contact,
            sim_time,
            1.0,
        )
        return self._actuated_torques_from_wbc_output(torque_wbc)

    def _desired_states_from_phase(self, phase: np.ndarray | None) -> np.ndarray:
        if phase is None:
            return np.ascontiguousarray(np.array([int(self.State.DUAL_STANCE)], dtype=np.int32))

        left_phase = float(np.asarray(phase, dtype=np.float64).reshape(1, -1)[0, 0])
        normalized_phase = (left_phase % (2.0 * np.pi)) / (2.0 * np.pi)
        cycle_time = max(float(self.task_config.gait_period), self._dt)
        transition_frac = np.clip(0.15 / cycle_time, 0.0, 0.24)
        stable_frac = max(1.0 - 4.0 * transition_frac, 0.0)
        left_end = stable_frac * 0.40
        dual_1_end = left_end + transition_frac + stable_frac * 0.10
        right_end = dual_1_end + transition_frac + stable_frac * 0.40
        dual_2_end = right_end + transition_frac + stable_frac * 0.10

        if normalized_phase < left_end:
            state = int(self.State.LEFT_STANCE)
        elif normalized_phase < dual_1_end:
            state = int(self.State.DUAL_STANCE)
        elif normalized_phase < right_end:
            state = int(self.State.RIGHT_STANCE)
        elif normalized_phase < dual_2_end:
            state = int(self.State.DUAL_STANCE)
        else:
            state = int(self.State.LEFT_STANCE)
        return np.ascontiguousarray(np.array([state], dtype=np.int32))

    def _wbc_action_from_12d(
        self,
        action: np.ndarray,
        desired_states: np.ndarray,
        lin_vel_command: np.ndarray | None,
        ang_vel_command: np.ndarray | None,
    ) -> np.ndarray:
        pelvis_lin_vel_residual = action[0, 0:3]
        pelvis_ang_vel_residual = action[0, 3:6]
        com_vel_residual = action[0, 6:9]
        landing_residual = action[0, 9:12]
        nominal_pelvis_lin_vel, nominal_pelvis_ang_vel = self._nominal_pelvis_velocities(
            lin_vel_command,
            ang_vel_command,
        )

        right_pose = self._pose_or_contact("right_foot", "right")
        left_pose = self._pose_or_contact("left_foot", "left")
        wbc_action = np.zeros((1, 72), dtype=np.float64)
        wbc_action[0, 3:6] = com_vel_residual
        wbc_action[0, 18:21] = nominal_pelvis_lin_vel + pelvis_lin_vel_residual
        wbc_action[0, 21:24] = nominal_pelvis_ang_vel + pelvis_ang_vel_residual
        self._write_foot_target(wbc_action, "right", right_pose[:3, 3], right_pose[:3, :3], np.zeros(6))
        self._write_foot_target(wbc_action, "left", left_pose[:3, 3], left_pose[:3, :3], np.zeros(6))

        desired_state = int(desired_states[0])
        if desired_state == int(self.State.LEFT_STANCE):
            self._write_landing_target(wbc_action, "right", left_pose, com_vel_residual, landing_residual)
        elif desired_state == int(self.State.RIGHT_STANCE):
            self._write_landing_target(wbc_action, "left", right_pose, com_vel_residual, landing_residual)

        return np.ascontiguousarray(wbc_action)

    def _nominal_pelvis_velocities(
        self,
        lin_vel_command: np.ndarray | None,
        ang_vel_command: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        linear = np.zeros(3, dtype=np.float64)
        angular = np.zeros(3, dtype=np.float64)
        if lin_vel_command is not None:
            command = np.asarray(lin_vel_command, dtype=np.float64).reshape(-1)
            linear[: min(2, command.shape[0])] = command[: min(2, command.shape[0])]
        if ang_vel_command is not None:
            command = np.asarray(ang_vel_command, dtype=np.float64).reshape(-1)
            if command.shape[0] > 0:
                angular[2] = command[0]
        return linear, angular

    def _pose_or_contact(self, frame_name: str, side: str) -> np.ndarray:
        try:
            pose = self.wbc.get_pose(frame_name, 0)
            return np.asarray(pose, dtype=np.float64).reshape(4, 4)
        except Exception:
            state = self.wbc_state_sub.latest or {}
            pos = np.asarray(state[f"{side}_contact_point"], dtype=np.float64).reshape(3)
            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = pos
            return pose

    def _write_landing_target(
        self,
        wbc_action: np.ndarray,
        swing_side: str,
        stance_pose: np.ndarray,
        com_vel: np.ndarray,
        residual: np.ndarray,
    ) -> None:
        com_pose = self._pose_or_contact("com", "right")
        pelvis_pose = self._pose_or_contact("pelvis", "right")
        landing_pose = np.asarray(
            self.SwingFoot.computeLandingFootstepPoseFromCapturePoint(
                np.ascontiguousarray(com_pose[:3, 3], dtype=np.float64),
                np.ascontiguousarray(com_vel, dtype=np.float64),
                np.ascontiguousarray(stance_pose, dtype=np.float64),
                np.ascontiguousarray(pelvis_pose, dtype=np.float64),
                swing_side,
            ),
            dtype=np.float64,
        ).reshape(4, 4)
        residual = np.asarray(residual, dtype=np.float64).reshape(3)
        landing_pose[:2, 3] += stance_pose[:2, :2] @ residual[:2]
        landing_pose[:3, :3] = landing_pose[:3, :3] @ _yaw_to_matrix3d(float(residual[2]))
        self._write_foot_target(wbc_action, swing_side, landing_pose[:3, 3], landing_pose[:3, :3], np.zeros(6))

    def _write_foot_target(
        self,
        wbc_action: np.ndarray,
        side: str,
        position: np.ndarray,
        rotation: np.ndarray,
        velocity: np.ndarray,
    ) -> None:
        if side == "right":
            pos_slice, ori_slice, vel_slice = slice(36, 39), slice(39, 48), slice(48, 54)
        elif side == "left":
            pos_slice, ori_slice, vel_slice = slice(54, 57), slice(57, 66), slice(66, 72)
        else:
            raise ValueError(f"Unknown foot side {side!r}.")
        wbc_action[0, pos_slice] = np.asarray(position, dtype=np.float64).reshape(3)
        wbc_action[0, ori_slice] = np.asarray(rotation, dtype=np.float64).reshape(3, 3).reshape(9)
        wbc_action[0, vel_slice] = np.asarray(velocity, dtype=np.float64).reshape(6)

    def _contact_mask(self, grf: np.ndarray) -> np.ndarray:
        force_norm = np.linalg.norm(grf[:, :3], axis=1)
        return np.ascontiguousarray(force_norm >= float(self.task_config.wbc_contact_force_threshold))

    def _zero_small_grf(self, grf: np.ndarray) -> np.ndarray:
        filtered = grf.copy()
        force_norm = np.linalg.norm(filtered[:, :3], axis=1)
        filtered[force_norm < float(self.task_config.wbc_contact_force_threshold)] = 0.0
        return np.ascontiguousarray(filtered)

    def _array2(self, values: Any, width: int) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(1, width))

    def _basis(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape == (3,):
            array = np.diag(array).reshape(1, 3, 3)
        elif array.shape == (1, 3):
            array = np.stack([np.diag(array[0])], axis=0)
        elif array.shape == (3, 3):
            array = array.reshape(1, 3, 3)
        return np.ascontiguousarray(array)

    def _actuated_torques_from_wbc_output(self, torque_wbc: Any) -> np.ndarray:
        torque_np = np.asarray(torque_wbc, dtype=np.float64)
        if torque_np.ndim != 2:
            raise RuntimeError(f"Unexpected batched WBC torque rank: got shape {torque_np.shape}.")
        if torque_np.shape[1] == self.num_dofs:
            return torque_np[0].astype(np.float32, copy=False)
        if torque_np.shape[1] == self.num_dofs + 6:
            return torque_np[0, 6:].astype(np.float32, copy=False)
        raise RuntimeError(
            "Unexpected WBC torque output size: "
            f"got shape {torque_np.shape}, expected {self.num_dofs} or {self.num_dofs + 6}."
        )

    def _resolve_wbc_extension_dir(self, extension_dir: str | None) -> str | None:
        if extension_dir:
            return str(Path(extension_dir).expanduser().resolve())
        for parent in Path(__file__).resolve().parents:
            # candidate = parent / "external" / "humanoid-control"
            candidate = parent / "humanoid-control"
            if candidate.exists():
                return str(candidate)
        default_build = Path("/home/william/humanoid-control/build")
        if default_build.exists():
            return str(default_build)
        return None

    def _resolve_wbc_params(self, extension_dir: str | None) -> dict[str, str]:
        params = {
            "robot_name": self.task_config.wbc_robot_name,
        }
        if self.task_config.wbc_robot_file is not None:
            params["robot_file"] = self.task_config.wbc_robot_file
        if self.task_config.wbc_yaml_file is not None:
            params["yaml_file"] = self.task_config.wbc_yaml_file

        if extension_dir is not None:
            root = self._extension_root(extension_dir)
            params.setdefault("robot_file", str((root / "models" / "unitree_g1" / "g1.urdf").resolve()))
            params.setdefault("yaml_file", str((root / "params" / "g1_parameters.yaml").resolve()))
        return params

    def _extension_root(self, extension_dir: str) -> Path:
        root = Path(extension_dir).expanduser().resolve()
        return root.parent if root.name == "build" else root

    def _import_wbc_module(self, extension_dir: str | None):
        module_name = "humanoid_wbc"
        if extension_dir:
            sys.path.insert(0, extension_dir)
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            candidates = [] if extension_dir is None else list(Path(extension_dir).rglob(f"{module_name}*.so"))
            for candidate in candidates:
                loader = importlib.machinery.ExtensionFileLoader(module_name, str(candidate))
                spec = importlib.util.spec_from_file_location(module_name, str(candidate), loader=loader)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module
            raise ImportError("Could not import humanoid_wbc for task-space inference.") from exc

    def _resolve_wbc_asset_paths(self, extension_dir: str | None, params: dict[str, str]) -> tuple[Path, Path, str]:
        if extension_dir is None:
            raise FileNotFoundError("task.wbc_extension_dir is required when humanoid-control is not auto-discovered.")
        root = self._extension_root(extension_dir)
        robot_file = Path(params.get("robot_file", root / "models" / "unitree_g1" / "g1.urdf"))
        yaml_file = Path(params.get("yaml_file", root / "params" / "g1_parameters.yaml"))
        robot_name = params.get("robot_name", "g1")
        if not robot_file.exists() or not yaml_file.exists():
            raise FileNotFoundError(f"Could not find WBC assets: robot_file={robot_file}, yaml_file={yaml_file}")
        return robot_file, yaml_file, robot_name
