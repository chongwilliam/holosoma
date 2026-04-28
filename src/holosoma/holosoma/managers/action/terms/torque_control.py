"""Action terms for joint-level torque control."""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from holosoma.managers.action.base import ActionTermBase
from holosoma.utils.rotations import quat_apply, quaternion_to_matrix

if TYPE_CHECKING:
    from holosoma.config_types.action import ActionTermCfg

def tensor_to_string(t: torch.Tensor, precision: int = 6) -> str:
    """
    Convert a 1D torch.Tensor to string "[x, y, z]".
    """
    t = t.detach().cpu().flatten()
    fmt = f"{{:.{precision}f}}"
    return "[" + ", ".join(fmt.format(x.item()) for x in t) + "]"

def string_to_tensor(s: str, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Convert string "[x, y, z]" to torch.Tensor.
    """
    values = s.strip()[1:-1].split(",")
    data = [float(v) for v in values if v.strip()]
    return torch.tensor(data, device=device, dtype=dtype)

def rot6d_to_matrix(x):
    """
    6D rotation representation to rotation matrix
    
    :param x: 6D rotation vector 
    """
    r1 = x[..., 0:3]
    r2 = x[..., 3:6]

    b1 = torch.nn.functional.normalize(r1, dim=-1)
    dot = (b1 * r2).sum(dim=-1, keepdim=True)
    b2 = torch.nn.functional.normalize(r2 - dot * b1, dim=-1)

    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1.0e-9:
        return np.eye(3)

    axis = axis_angle / angle
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def matrix_to_urdf_floating_xyz_angles(rotation: torch.Tensor) -> torch.Tensor:
    """Extract angles for a URDF chain R = Rx(rx) @ Ry(ry) @ Rz(rz)."""
    sy = torch.clamp(rotation[..., 0, 2], -1.0, 1.0)
    ry = torch.asin(sy)
    rx = torch.atan2(-rotation[..., 1, 2], rotation[..., 2, 2])
    rz = torch.atan2(-rotation[..., 0, 1], rotation[..., 0, 0])
    return torch.stack([rx, ry, rz], dim=-1)


def root_state_to_xyz_rpy(root_state: torch.Tensor) -> torch.Tensor:
    """Convert root [xyz, quat_xyzw, v, w] to [xyz, rx, ry, rz] for the control URDF floating joints."""
    root_rotation = quaternion_to_matrix(root_state[3:7].unsqueeze(0), w_last=True)[0]
    rx_ry_rz = matrix_to_urdf_floating_xyz_angles(root_rotation)
    return torch.cat([root_state[0:3], rx_ry_rz], dim=0)


def root_state_to_base_velocity(root_state: torch.Tensor) -> torch.Tensor:
    """Return floating-base velocity as [linear_velocity, angular_velocity]."""
    return root_state[7:13]


def root_states_to_xyz_rpy(root_states: torch.Tensor) -> torch.Tensor:
    """Convert batched root states to [xyz, rx, ry, rz] for the control URDF floating joints."""
    root_rotation = quaternion_to_matrix(root_states[:, 3:7], w_last=True)
    rx_ry_rz = matrix_to_urdf_floating_xyz_angles(root_rotation)
    return torch.cat([root_states[:, 0:3], rx_ry_rz], dim=-1)

def parse_actions(actions: torch.Tensor) -> dict:
    """
    Convert action tensor to action mapping 
    """

    # # Full action output
    # action_dict = {
    #     "com_pos": actions[:3],
    #     "pelvis_ori": actions[3:9],
    #     "right_foot_pos": actions[9:12],
    #     "right_foot_ori": actions[12:18],
    #     "left_foot_pos": actions[18:21],
    #     "left_foot_ori": actions[21:24],
    #     "right_hand_pos": actions[24:27],
    #     "right_hand_ori": actions[27:33],
    #     "left_hand_pos": actions[33:36],
    #     "left_hand_ori": actions[36:43],
    # }

    # # Full action output
    # action_dict = {
    #     "com_pos": actions[:3],
    #     "pelvis_ori": actions[3:6],
    #     "right_foot_pos": actions[6:9],
    #     "right_foot_ori": actions[9:12],
    #     "left_foot_pos": actions[12:15],
    #     "left_foot_ori": actions[15:18],
    #     "right_hand_pos": actions[18:21],
    #     "right_hand_ori": actions[21:24],
    #     "left_hand_pos": actions[24:27],
    #     "left_hand_ori": actions[27:30],
    # }

    # # Minimal action output (com and feet position)
    # action_dict = {
    #     "com_pos": actions[:3],
    #     "right_foot_pos": actions[3:6],
    #     "left_foot_pos": actions[6:9],
    # }

    # Lower body action output
    action_dict = {
        "com_pos": actions[:3],
        "pelvis_ori": actions[3:6],
        "torso_ori": actions[6:9],
        "right_foot_pos": actions[9:12],
        "right_foot_ori": actions[12:15],
        "left_foot_pos": actions[15:18],
        "left_foot_ori": actions[18:21],
    }

    return action_dict

def skew(w):
    """Convert a 3D vector to a skew-symmetric matrix."""
    wx, wy, wz = w
    return np.array([
        [0.0, -wz,  wy],
        [wz,  0.0, -wx],
        [-wy, wx,  0.0]
    ])

def exp_map(omega):
    theta = np.linalg.norm(omega)
    w_hat = skew(omega)
    w_hat_sq = w_hat @ w_hat

    if theta < 1e-6:
        # Use series expansion
        A = 1 - theta**2 / 6 + theta**4 / 120
        B = 0.5 - theta**2 / 24 + theta**4 / 720
    else:
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / (theta ** 2)

    R = np.eye(3) + A * w_hat + B * w_hat_sq
    return R

def apply_delta_rotation(R_current, omega):
    return R_current @ exp_map(omega)

### CLASS ###

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

    def reset_state(self, state: Any | None = None, transition_start_time: float = 0.0) -> None:
        if state is None:
            self._controller.reset_state()
            return
        self._controller.reset_state(state, transition_start_time)

    def compute_torques_batch(self, *args: Any, **kwargs: Any) -> Any:
        return self._controller.compute_torques_batch(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if self._debug_engine is not None and hasattr(self._debug_engine, name):
            return getattr(self._debug_engine, name)
        return getattr(self._controller, name)


class JointTorqueActionTerm(ActionTermBase):
    """Action term for joint torque control with whole-body controller.

    This term processes raw actions as task space targets and computes
    torques using a WBC controller. Supports:
    - Action scaling
    - Action clipping
    - Action delay (if configured)
    - Torque randomization (if configured)
    - Torque clipping
    """

    def __init__(self, cfg: ActionTermCfg, env: Any):
        """Initialize joint position action term.

        Args:
            cfg: Configuration for this action term
            env: Environment instance (typically a ``BaseTask`` subclass)
        """
        super().__init__(cfg, env)

        # Policy actions may be task-space commands, while the controller still outputs
        # one torque per actuated DOF.
        self._action_dim = env.robot_config.actions_dim
        self._torque_dim = env.num_dof

        # Initialize action buffers
        self._raw_actions = torch.zeros(env.num_envs, self._action_dim, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, self._action_dim, device=env.device)
        self._actions_after_delay = torch.zeros(env.num_envs, self._action_dim, device=env.device)

        # Initialize torque buffer
        self.torques = torch.zeros(env.num_envs, self._torque_dim, device=env.device)

        # Cache previous DOF velocities for derivative control
        self._prev_dof_vel = torch.zeros(env.num_envs, env.num_dof, device=env.device)

        # Default actuator scaling (may be overridden by randomization terms)
        self._kp_scale = torch.ones(env.num_envs, self._torque_dim, device=env.device)
        self._kd_scale = torch.ones_like(self._kp_scale)
        self._rfi_lim_scale = torch.ones_like(self._kp_scale)
        self._rfi_lim: float = 0.0
        self._randomize_torque_rfi: bool = False

        # PD gains are per actuator; action scales are per policy action.
        self.p_gains = torch.zeros(self._torque_dim, dtype=torch.float, device=env.device)
        self.d_gains = torch.zeros_like(self.p_gains)
        self.i_gains = torch.zeros_like(self.p_gains)
        self.action_scales = torch.zeros(self._action_dim, dtype=torch.float, device=env.device)

        self._configure_pd_gains(env)
        self._configure_action_scales(env)

        # Expose references on the environment for backward compatibility
        env.p_gains = self.p_gains
        env.d_gains = self.d_gains
        env.i_gains = self.i_gains
        env.action_scales = self.action_scales

        # Action delay queue will be initialized in setup() after randomization manager is ready
        self.action_queue: torch.Tensor | None = None

        self._foot_body_indices = self._resolve_foot_body_indices(env)
        self._last_wbc_q = torch.zeros(env.num_envs, env.num_dof + 6, device=env.device)
        self._last_wbc_root_state = torch.zeros(env.num_envs, 13, device=env.device)
        self._last_wbc_dof_pos = torch.zeros(env.num_envs, env.num_dof, device=env.device)
        resolved_extension_dir = self._resolve_wbc_extension_dir(cfg.wbc_extension_dir)
        resolved_params = self._resolve_wbc_params(env, cfg.params, resolved_extension_dir)
        self._wbc_module = self._import_wbc_module(resolved_extension_dir)
        self.State = self._wbc_module.State
        self.Phase = self._wbc_module.Phase
        self._batched_wbc = self._create_batched_wbc_controller(
            resolved_extension_dir, resolved_params, env.num_envs, self._wbc_module
        )
        self._wbc_debug_engine = self._create_wbc_engine(resolved_extension_dir, resolved_params, self._wbc_module)
        self.wbc = _BatchedWbcFacade(self._batched_wbc, int(self._wbc_debug_engine.dof()), self._wbc_debug_engine)
        self.curr_state = [self.State.DUAL_STANCE for _ in range(env.num_envs)]
        self._prev_wbc_state: Any | None = None
        self.transition_start_time = [0.0 for _ in range(env.num_envs)]
        self.wbc.setTotalTransitionTime(0.15)  # hard-coded for now

    def setup(self) -> None:
        """Setup action term after all managers are initialized.

        Initialize action delay queue if control delay randomization is enabled.
        This must be called after the randomization manager is set up.
        """
        super().setup()

        # Initialize action delay queue if randomization is enabled
        if getattr(self.env, "_randomize_ctrl_delay", False):
            max_delay = self.env._ctrl_delay_step_range[1]
            self.action_queue = torch.zeros(self.env.num_envs, max_delay + 1, self._action_dim, device=self.env.device)

        # IsaacGym creates randomization buffers before the action manager exists.
        # Once we reach setup(), try attaching any pre-created actuator scales.
        self._attach_actuator_randomizer_scales()

        enabled, rfi_lim = self.env._pending_torque_rfi
        self.configure_torque_rfi(enabled=enabled, rfi_lim=rfi_lim)
        self.env._pending_torque_rfi = (False, 0.0)

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return self._action_dim

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process raw actions: clip and apply delay if configured.

        Args:
            actions: Raw action tensor [num_envs, action_dim]
        """
        # Store raw actions
        assert self._raw_actions is not None
        self._raw_actions[:] = actions

        # Clip actions
        if self.env.robot_config.control.clip_actions:
            clip_limit = self.env.robot_config.control.action_clip_value
            assert self._processed_actions is not None
            self._processed_actions[:] = torch.clip(actions, -clip_limit, clip_limit)
            # Log clipping fraction
            self.env.log_dict["action_clip_frac"] = (
                self._processed_actions.abs() == clip_limit
            ).sum() / self._processed_actions.numel()
        else:
            assert self._processed_actions is not None
            self._processed_actions[:] = actions
            self.env.log_dict["action_clip_frac"] = torch.tensor(0.0)

        # Apply action delay if configured
        if getattr(self.env, "_randomize_ctrl_delay", False):
            self._apply_action_delay()
        else:
            assert self._processed_actions is not None
            self._actions_after_delay[:] = self._processed_actions

    def _apply_action_delay(self) -> None:
        """Apply action delay based on domain randomization settings."""
        assert self.action_queue is not None, "action_queue must be initialized in setup()"
        assert self._processed_actions is not None

        # Update action queue
        self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
        self.action_queue[:, 0] = self._processed_actions.clone()

        # Apply uniform delay
        self._actions_after_delay[:] = self.action_queue[
            torch.arange(self.env.num_envs), self.env.action_delay_idx
        ].clone()

    def apply_actions(self) -> None:
        """Apply processed actions by computing and applying torques."""
        # Compute torques using PD controller
        self.torques[:] = self._compute_torques(self._actions_after_delay)
        # Apply torques to simulator
        self.env.simulator.apply_torques_at_dof(self.torques)
        # Cache velocities for next derivative computation
        self._prev_dof_vel.copy_(self.env.simulator.dof_vel)


    def _parse_and_send_actions(self, actions: torch.Tensor, idx: int) -> None:
        """
        Docstring for parse_and_send_actions
        
        :param actions: Description
        :type actions: torch.Tensor
        :param idx: Description
        :type idx: int
        """

        action_dict = parse_actions(actions) 
        str_append = "_" + str(idx)

        # Redis pipeline 
        for key, value in action_dict:
            self.redis_pipe[idx].set(key + str_append, tensor_to_string(value))

        self.redis_pipe[idx].execute()

        return None 
    
    def _parse_and_send_states(self, states_dict: dict, idx: int) -> None:
        """
        Docstring for _parse_and_send_states
        
        :param self: Description
        :param states: Description
        :type states: torch.Tensor
        :param idx: Description
        :type idx: int
        """

        str_append = "_" + str(idx)
        for key, value in states_dict:
            self.redis_pipe[idx].set(key + str_append, tensor_to_string(value))
        
        self.redis_pipe[idx].execute()

        return None 

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute torques from the in-process whole-body controller.

        Args:
            actions: Action tensor [num_envs, action_dim]

        Returns:
            Torque tensor [num_envs, num_dof]
        """
        num_envs = actions.shape[0]
        if num_envs != self.env.num_envs:
            raise RuntimeError(
                "BatchedWbcController requires one action row per environment. "
                f"Got actions.shape[0]={num_envs}, env.num_envs={self.env.num_envs}."
            )

        action_scale = float(self.env.robot_config.control.action_scale)

        root_states = self._as_torch_tensor(
            self.env.simulator.robot_root_states, device=actions.device, dtype=torch.float32, label="root_states"
        )
        dof_pos = self._as_torch_tensor(
            self.env.simulator.dof_pos, device=actions.device, dtype=torch.float32, label="dof_pos"
        )
        dof_vel = self._as_torch_tensor(
            self.env.simulator.dof_vel, device=actions.device, dtype=torch.float32, label="dof_vel"
        )
        q_tensor = torch.cat([root_states_to_xyz_rpy(root_states), dof_pos], dim=1)
        dq_tensor = torch.cat([root_states[:, 7:13], dof_vel], dim=1)

        self._last_wbc_q[:num_envs] = q_tensor
        self._last_wbc_root_state[:num_envs] = root_states
        self._last_wbc_dof_pos[:num_envs] = dof_pos

        right_contact_points = self._as_numpy_2d(self.env.simulator.right_foot_contact_position, "right_contact_points")
        left_contact_points = self._as_numpy_2d(self.env.simulator.left_foot_contact_position, "left_contact_points")
        right_contact_bases = self._contact_bases_as_numpy(self.env.simulator.right_foot_contact_basis, "right_contact_bases")
        left_contact_bases = self._contact_bases_as_numpy(self.env.simulator.left_foot_contact_basis, "left_contact_bases")
        right_grfs = self._batched_foot_ground_reaction_wrenches("right")
        left_grfs = self._batched_foot_ground_reaction_wrenches("left")

        q = self._as_numpy_2d(q_tensor, "q")
        dq = self._as_numpy_2d(dq_tensor, "dq")
        action_batch = self._actions_for_batched_wbc(actions)
        self._wbc_debug_engine.updateRobot(q[0], dq[0])
        torque_wbc = self.wbc.compute_torques_batch(
            q,
            dq,
            action_batch,
            right_contact_points,
            right_contact_bases,
            left_contact_points,
            left_contact_bases,
            right_grfs,
            left_grfs,
            float(self.env.simulator.time()),
            action_scale,
        )
        torques = torch.as_tensor(
            self._actuated_torques_from_batched_wbc_output(torque_wbc),
            device=actions.device,
            dtype=self.torques.dtype,
        )

        # Scale actions
        # actions_scaled = actions * self.action_scales

        # # Compute torques based on control type
        # control_type = self.env.robot_config.control.control_type

        # if control_type == "P":
        #     # Position control
        #     torques = (
        #         self._kp_scale * self.p_gains * (actions_scaled + self.env.default_dof_pos - self.env.simulator.dof_pos)
        #         - self._kd_scale * self.d_gains * self.env.simulator.dof_vel
        #     )
        # elif control_type == "V":
        #     # Velocity control
        #     torques = (
        #         self._kp_scale * self.p_gains * (actions_scaled - self.env.simulator.dof_vel)
        #         - self._kd_scale * self.d_gains * (self.env.simulator.dof_vel - self._prev_dof_vel) / self.env.sim_dt
        #     )
        # elif control_type == "T":
        #     # Torque control
        #     torques = actions_scaled
        # else:
        #     raise ValueError(f"Unknown controller type: {control_type}")

        # Apply torque randomization if configured
        if self._randomize_torque_rfi:
            torques = (
                torques
                + (torch.rand_like(torques) * 2.0 - 1.0) * self._rfi_lim * self._rfi_lim_scale * self.env.torque_limits
            )

        # Clip torques if configured
        if self.env.robot_config.control.clip_torques:
            torques = torch.clip(torques, -self.env.torque_limits, self.env.torque_limits)

        return torques

    def _as_torch_tensor(self, value: Any, *, device: torch.device | str, dtype: torch.dtype, label: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
        elif getattr(value, "_is_tensor_proxy", False):
            tensor = value[:]
        else:
            tensor = torch.as_tensor(value)
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        return tensor.to(device=device, dtype=dtype)

    def _as_numpy_2d(self, value: Any, label: str) -> np.ndarray:
        tensor = self._as_torch_tensor(value, device="cpu", dtype=torch.float64, label=label)
        array = tensor.detach().cpu().numpy().astype(np.float64, copy=False)
        if array.ndim != 2:
            raise RuntimeError(f"{label} must be rank 2, got shape {array.shape}.")
        return np.ascontiguousarray(array)

    def _actions_for_batched_wbc(self, actions: torch.Tensor) -> np.ndarray:
        if actions.shape[1] != 21:
            return self._as_numpy_2d(actions, "actions")

        # Locomotion policy layout:
        # [com_pos, pelvis_ori, torso_ori, right_foot_pos/ori, left_foot_pos/ori].
        return self._as_numpy_2d(actions, "actions")

    def _contact_bases_as_numpy(self, value: Any, label: str) -> np.ndarray:
        tensor = self._as_torch_tensor(value, device=self.env.device, dtype=torch.float32, label=label)
        if tensor.dim() == 2 and tensor.shape[-1] == 3:
            tensor = torch.diag_embed(tensor)
        if tensor.dim() != 3 or tensor.shape[-2:] != (3, 3):
            raise RuntimeError(f"{label} must have shape [num_envs, 3] or [num_envs, 3, 3], got {tuple(tensor.shape)}.")
        return np.ascontiguousarray(tensor.cpu().numpy().astype(np.float64, copy=False))

    def _batched_foot_ground_reaction_wrenches(self, side: str) -> np.ndarray:
        grfs = np.stack(
            [self._foot_ground_reaction_wrench(env_idx, side) for env_idx in range(self.env.num_envs)],
            axis=0,
        ).astype(np.float64, copy=False)
        force_norms = np.linalg.norm(grfs[:, :3], axis=1)
        grfs[force_norms < 20.0] = 0.0
        return np.ascontiguousarray(grfs)

    def _actuated_torques_from_batched_wbc_output(self, torque_wbc: Any) -> np.ndarray:
        """Normalize batched WBC torque output to simulator actuated DOF torques."""
        torque_np = np.asarray(torque_wbc, dtype=float)
        if torque_np.ndim != 2:
            raise RuntimeError(f"Unexpected batched WBC torque rank: got shape {torque_np.shape}.")
        if torque_np.shape[1] == self._torque_dim:
            return torque_np
        if torque_np.shape[1] == self._torque_dim + 6:
            return torque_np[:, 6:]
        raise RuntimeError(
            "Unexpected batched WBC torque output size: "
            f"got shape {torque_np.shape}, expected {self._torque_dim} actuated torques "
            f"or {self._torque_dim + 6} floating-base torques per environment."
        )

    def _actuated_torques_from_wbc_output(self, torque_wbc: Any) -> np.ndarray:
        """Normalize WBC torque output to simulator actuated DOF torques."""
        torque_np = np.asarray(torque_wbc, dtype=float).reshape(-1)
        if torque_np.shape[0] == self._torque_dim:
            return torque_np
        if torque_np.shape[0] == self._torque_dim + 6:
            return torque_np[6:]
        raise RuntimeError(
            "Unexpected WBC torque output size: "
            f"got {torque_np.shape[0]}, expected {self._torque_dim} actuated torques "
            f"or {self._torque_dim + 6} floating-base torques."
        )

    def _draw_stance_support(
        self,
        env_idx: int,
        right_contact_point: np.ndarray,
        right_contact_basis: np.ndarray,
        right_side: str,
        left_contact_point: np.ndarray,
        left_contact_basis: np.ndarray,
        left_side: str,
    ) -> None:
        simulator = self.env.simulator
        if not hasattr(simulator, "draw_sphere"):
            return

        self._draw_contact_support_frame(
            env_idx,
            np.asarray(right_contact_point, dtype=float),
            np.asarray(right_contact_basis, dtype=float),
            right_side,
            point_color=[1.0, 0.25, 0.1],
            pos_id_base=100,
        )
        self._draw_contact_support_frame(
            env_idx,
            np.asarray(left_contact_point, dtype=float),
            np.asarray(left_contact_basis, dtype=float),
            left_side,
            point_color=[0.2, 0.45, 1.0],
            pos_id_base=200,
        )

    def _draw_contact_support_frame(
        self,
        env_idx: int,
        contact_point: np.ndarray,
        contact_basis: np.ndarray,
        side: str,
        point_color: list[float],
        pos_id_base: int,
    ) -> None:
        if contact_point.shape[0] < 3 or contact_basis.shape != (3, 3):
            return
        if not np.all(np.isfinite(contact_point)) or not np.all(np.isfinite(contact_basis)):
            return

        simulator = self.env.simulator
        foot_body_idx = self._foot_body_indices[side]
        mujoco_body_map = getattr(simulator, "holosoma_to_mujoco_body_map", None)
        rigid_body_idx = foot_body_idx
        if isinstance(mujoco_body_map, dict):
            rigid_body_idx = mujoco_body_map.get(foot_body_idx, foot_body_idx)

        foot_pos_w = simulator._rigid_body_pos[env_idx, rigid_body_idx]
        foot_quat_w = simulator._rigid_body_rot[env_idx, rigid_body_idx]
        self._assert_foot_pose_matches_wbc(env_idx, side, foot_pos_w, foot_quat_w, contact_point)

        local_contact_point = torch.as_tensor(contact_point[:3], device=foot_pos_w.device, dtype=foot_pos_w.dtype)
        world_contact_point_tensor = foot_pos_w + quat_apply(foot_quat_w, local_contact_point, w_last=True)
        world_contact_point = world_contact_point_tensor.detach().cpu().numpy()

        point = torch.as_tensor(world_contact_point, dtype=torch.float32).cpu()
        simulator.draw_sphere(point, 0.018, point_color, env_id=env_idx, pos_id=pos_id_base)

        if not hasattr(simulator, "draw_line"):
            return

        axis_colors = ([1.0, 0.0, 0.0], [0.0, 0.85, 0.0], [0.1, 0.25, 1.0])
        axis_scale = 0.12
        for axis_idx, axis_color in enumerate(axis_colors):
            local_axis = torch.as_tensor(
                contact_basis[:, axis_idx], device=foot_quat_w.device, dtype=foot_quat_w.dtype
            )
            axis = quat_apply(foot_quat_w, local_axis, w_last=True).detach().cpu().numpy()
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm < 1.0e-9:
                continue
            end = torch.as_tensor(world_contact_point + axis_scale * axis / axis_norm, dtype=torch.float32).cpu()
            simulator.draw_line(point, end, axis_color, env_id=env_idx)

    def _assert_foot_pose_matches_wbc(
        self,
        env_idx: int,
        side: str,
        foot_pos_w: torch.Tensor,
        foot_quat_w: torch.Tensor,
        local_contact_point: np.ndarray
    ) -> None:
        # foot_name = f"{side}_foot"
        foot_name = f"{side}_ankle_roll_link"
        wbc_engine = self._wbc_debug_engine
        wbc_pos = np.asarray(wbc_engine.getPosition(foot_name, local_contact_point), dtype=float).reshape(-1)[:3]
        wbc_rot = np.asarray(wbc_engine.getRotation(foot_name), dtype=float)
        if wbc_pos.shape[0] < 3 or not np.all(np.isfinite(wbc_pos)):
            raise AssertionError(f"WBC {foot_name} position is invalid: shape={wbc_pos.shape}, pos={wbc_pos}")
        if wbc_rot.shape != (3, 3) or not np.all(np.isfinite(wbc_rot)):
            raise AssertionError(f"WBC {foot_name} rotation is invalid: shape={wbc_rot.shape}, rotation={wbc_rot}")

        sim_rot = quaternion_to_matrix(foot_quat_w.unsqueeze(0), w_last=True)[0].detach().cpu().numpy()
        sim_pos = foot_pos_w.detach().cpu().numpy() + sim_rot @ local_contact_point
        self._print_foot_frame_inspection(env_idx, side, sim_pos, sim_rot, wbc_pos, wbc_rot)

        # root_state = self.env.simulator.robot_root_states[env_idx]
        # dof_pos = self.env.simulator.dof_pos[env_idx]
        # q_tensor = torch.cat([root_state_to_xyz_rpy(root_state), dof_pos], dim=0)
        # q_error = float(torch.linalg.vector_norm(q_tensor - self._last_wbc_q[env_idx]).item())
        # root_state_error = float(
        #     torch.linalg.vector_norm(root_state - self._last_wbc_root_state[env_idx]).item()
        # )
        # dof_pos_error = float(torch.linalg.vector_norm(dof_pos - self._last_wbc_dof_pos[env_idx]).item())
        # pelvis_pos_error, pelvis_rot_error, pelvis_pos, wbc_pelvis_pos, pelvis_pos_delta = (
        #     self._assert_pelvis_pose_matches_wbc(env_idx)
        # )
        pos_delta = sim_pos - wbc_pos
        pos_error = float(np.linalg.norm(sim_pos - wbc_pos))
        rot_error = float(np.linalg.norm(sim_rot - wbc_rot, ord="fro"))
        assert (
            # q_error < 1.0e-6
            # and root_state_error < 1.0e-6
            # and dof_pos_error < 1.0e-6
            # and pelvis_pos_error < 1.0e-6
            # and pelvis_rot_error < 1.0e-6
            pos_error < 1.0e-3
            and rot_error < 1.0e-2
        ), (
            f"Simulator {side} foot pose does not match WBC {side}_foot pose: "
            # f"q_error={q_error:.9f}, root_state_error={root_state_error:.9f}, "
            # f"dof_pos_error={dof_pos_error:.9f}, pelvis_pos_error={pelvis_pos_error:.9f}, "
            # f"pelvis_rot_error={pelvis_rot_error:.9f}, pos_error={pos_error:.9f}, rot_error={rot_error:.9f}, "
            # f"pelvis_sim_pos_xyz={pelvis_pos.tolist()}, pelvis_wbc_pos_xyz={wbc_pelvis_pos.tolist()}, "
            # f"pelvis_pos_delta_xyz={pelvis_pos_delta.tolist()}, "
            f"sim_pos_xyz={sim_pos.tolist()}, wbc_pos_xyz={wbc_pos.tolist()}, pos_delta_xyz={pos_delta.tolist()}, rot_delta={rot_error}"
            "\n"
        )

    def _assert_pelvis_pose_matches_wbc(
        self,
        env_idx: int,
    ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        wbc_pelvis_pos = np.asarray(self._wbc_debug_engine.getPosition("pelvis"), dtype=float).reshape(-1)
        wbc_pelvis_rot = np.asarray(self._wbc_debug_engine.getRotation("pelvis"), dtype=float)
        if wbc_pelvis_pos.shape[0] < 3 or not np.all(np.isfinite(wbc_pelvis_pos)):
            raise AssertionError(f"WBC pelvis position is invalid: shape={wbc_pelvis_pos.shape}, pos={wbc_pelvis_pos}")
        if wbc_pelvis_rot.shape != (3, 3) or not np.all(np.isfinite(wbc_pelvis_rot)):
            raise AssertionError(
                f"WBC pelvis rotation is invalid: shape={wbc_pelvis_rot.shape}, rotation={wbc_pelvis_rot}"
            )

        root_state = self.env.simulator.robot_root_states[env_idx]
        pelvis_pos = root_state[:3].detach().cpu().numpy()
        pelvis_rot = quaternion_to_matrix(root_state[3:7].unsqueeze(0), w_last=True)[0].detach().cpu().numpy()
        wbc_pelvis_pos = wbc_pelvis_pos[:3]
        pelvis_pos_delta = pelvis_pos - wbc_pelvis_pos
        pelvis_pos_error = float(np.linalg.norm(pelvis_pos_delta))
        pelvis_rot_error = float(np.linalg.norm(pelvis_rot - wbc_pelvis_rot, ord="fro"))
        return pelvis_pos_error, pelvis_rot_error, pelvis_pos, wbc_pelvis_pos, pelvis_pos_delta

    def _print_foot_frame_inspection(
        self,
        env_idx: int,
        side: str,
        resolved_sim_pos: np.ndarray,
        resolved_sim_rot: np.ndarray,
        wbc_compare_pos: np.ndarray,
        wbc_compare_rot: np.ndarray,
    ) -> None:
        simulator = self.env.simulator
        resolved_idx = self._foot_body_indices[side]
        body_names = list(getattr(simulator, "body_names", []))
        resolved_name = body_names[resolved_idx] if 0 <= resolved_idx < len(body_names) else "<out-of-range>"
        resolved_rot_error = float(np.linalg.norm(resolved_sim_rot - wbc_compare_rot, ord="fro"))

        print(
            "foot_frame_inspection:"
            f" side={side}"
            f" resolved_idx={resolved_idx}"
            f" resolved_name={resolved_name}"
            f" robot_config.foot_body_name={self.env.robot_config.foot_body_name}"
            f" resolved_sim_pos={resolved_sim_pos.tolist()}"
            f" wbc_compare_pos={wbc_compare_pos.tolist()}"
            f" delta={list((resolved_sim_pos - wbc_compare_pos).tolist())}"
            f" rot_error={resolved_rot_error:.9f}"
            "\n"
        )

        # # for suffix in ("ankle_pitch_link", "ankle_roll_link", "foot"):
        # for suffix in ("ankle_roll_link"):
        #     frame_name = f"{side}_{suffix}"
        #     sim_matches = [(idx, name) for idx, name in enumerate(body_names) if name == frame_name]
        #     wbc_rot = None
        #     for sim_idx, sim_name in sim_matches:
        #         sim_pos = simulator._rigid_body_pos[env_idx, sim_idx].detach().cpu().numpy()
        #         sim_quat = simulator._rigid_body_rot[env_idx, sim_idx]
        #         sim_rot = quaternion_to_matrix(sim_quat.unsqueeze(0), w_last=True)[0].detach().cpu().numpy()
        #         try:
        #             if wbc_rot is None:
        #                 wbc_rot = np.asarray(self.wbc[env_idx].getRotation(frame_name), dtype=float)
        #             rot_error = float(np.linalg.norm(sim_rot - wbc_rot, ord="fro"))
        #             print(f"  frame_rotation_error name={sim_name} idx={sim_idx} rot_error={rot_error:.9f}")
        #         except Exception as exc:
        #             print(f"  frame_rotation_error name={sim_name} idx={sim_idx} unavailable={exc}")

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset action term state.

        Args:
            env_ids: Environment IDs to reset. If None, reset all.
        """
        super().reset(env_ids)

        # Reset action delay queue if applicable
        if getattr(self.env, "_randomize_ctrl_delay", False) and self.action_queue is not None:
            if env_ids is None:
                self.action_queue.zero_()
            else:
                self.action_queue[env_ids] = 0.0

        # Reset torques
        if env_ids is None:
            self.torques.zero_()
        else:
            self.torques[env_ids] = 0.0

        # Reset cached velocities
        if env_ids is None:
            self._prev_dof_vel.zero_()
            self.wbc.reset_state(int(self.State.DUAL_STANCE), 0.0)
        else:
            self._prev_dof_vel[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Hooks for randomization manager

    def attach_actuator_scales(
        self, kp_scale: torch.Tensor, kd_scale: torch.Tensor, rfi_lim_scale: torch.Tensor
    ) -> None:
        """Attach shared actuator scaling tensors provided by the randomization manager."""
        self._kp_scale = kp_scale
        self._kd_scale = kd_scale
        self._rfi_lim_scale = rfi_lim_scale

    def update_pd_scales(self, env_ids: torch.Tensor, kp_values: torch.Tensor, kd_values: torch.Tensor) -> None:
        """Fallback PD-scale update when no shared buffers are registered."""
        self._kp_scale[env_ids] = kp_values
        self._kd_scale[env_ids] = kd_values

    def update_rfi_scales(self, env_ids: torch.Tensor, rfi_values: torch.Tensor) -> None:
        """Fallback RFI-scale update when no shared buffers are registered."""
        self._rfi_lim_scale[env_ids] = rfi_values

    def configure_torque_rfi(self, *, enabled: bool, rfi_lim: float | None = None) -> None:
        """Configure residual force injection behaviour."""
        self._randomize_torque_rfi = enabled
        if rfi_lim is not None:
            self._rfi_lim = float(rfi_lim)

    def get_pd_scale_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return references to the PD gain scale buffers."""
        return self._kp_scale, self._kd_scale

    def get_rfi_scale_tensor(self) -> torch.Tensor:
        """Return reference to the RFI limit scale buffer."""
        return self._rfi_lim_scale

    def get_prev_dof_vel(self) -> torch.Tensor:
        """Return cached previous DOF velocities."""
        return self._prev_dof_vel

    # ------------------------------------------------------------------
    # Internal helpers

    def _resolve_foot_body_indices(self, env: Any) -> dict[str, int]:
        """Resolve simulator body indices used to read per-foot ground reaction forces."""
        foot_indices: dict[str, int] = {}
        for side in ("right", "left"):
            exact_name = f"{side}_{env.robot_config.foot_body_name}"
            match = next((idx for idx, name in enumerate(env.simulator.body_names) if name == exact_name), None)
            if match is None:
                match = next(
                    (
                        idx
                        for idx, name in enumerate(env.simulator.body_names)
                        if side in name.lower() and name.endswith(env.robot_config.foot_body_name)
                    ),
                    None,
                )
            if match is None:
                match = next(
                    (
                        idx
                        for idx, name in enumerate(env.simulator.body_names)
                        if side in name.lower() and env.robot_config.foot_body_name in name
                    ),
                    None,
                )
            if match is None:
                raise ValueError(
                    f"Could not resolve the {side} foot body using "
                    f"foot_body_name='{env.robot_config.foot_body_name}'."
                )
            foot_indices[side] = match
        return foot_indices

    def _foot_ground_reaction_wrench(self, env_idx: int, side: str) -> np.ndarray:
        """Return the simulator foot contact force as a 6D wrench for the WBC binding."""
        sensor_wrench_fn = getattr(self.env.simulator, "get_foot_force_sensor_wrench", None)
        if callable(sensor_wrench_fn):
            try:
                return sensor_wrench_fn(side, env_idx).detach().cpu().numpy().astype(float, copy=False)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to read foot force/torque sensor wrench via "
                    f"env.simulator.get_foot_force_sensor_wrench(side={side!r}, env_idx={env_idx}). "
                    "Refusing to fall back to env.simulator.contact_forces because the WBC expects the "
                    "XML foot force sensor wrench in [force; torque] order."
                ) from exc

        raise RuntimeError(
            "env.simulator does not provide get_foot_force_sensor_wrench(). "
            "Refusing to fall back to env.simulator.contact_forces because the WBC expects the XML foot "
            "force sensor wrench in [force; torque] order."
        )

    def _attach_actuator_randomizer_scales(self) -> None:
        """Attach shared actuator randomizer buffers if they exist."""
        rand_manager = getattr(self.env, "randomization_manager", None)
        if rand_manager is None:
            return

        get_state = getattr(rand_manager, "get_state", None)
        if not callable(get_state):
            return

        state = get_state("actuator_randomizer_state")
        if state is None:
            return

        self.attach_actuator_scales(state.kp_scale_tensor, state.kd_scale_tensor, state.rfi_lim_scale_tensor)

    def _configure_pd_gains(self, env: Any) -> None:
        control_cfg = env.robot_config.control
        stiffness_cfg = control_cfg.stiffness
        damping_cfg = control_cfg.damping
        integral_cfg = getattr(control_cfg, "integral", {})

        for i, name in enumerate(env.dof_names):
            if name not in env.robot_config.init_state.default_joint_angles:
                raise ValueError(f"Missing default joint angle for DOF '{name}' in robot configuration.")

            matched = False
            for dof_name, stiffness in stiffness_cfg.items():
                if dof_name in name:
                    self.p_gains[i] = stiffness
                    self.d_gains[i] = damping_cfg[dof_name]
                    self.i_gains[i] = integral_cfg.get(dof_name, 0.0)
                    matched = True
            if not matched:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                self.i_gains[i] = 0.0
                if control_cfg.control_type in ["P", "V"]:
                    raise ValueError(
                        f"PD gains for joint '{name}' were not defined. Please specify them in the YAML configuration."
                    )

    def _configure_action_scales(self, env: Any) -> None:
        control_cfg = env.robot_config.control
        if control_cfg.action_scales_by_effort_limit_over_p_gain:
            if self._action_dim != self._torque_dim:
                raise ValueError(
                    "action_scales_by_effort_limit_over_p_gain requires policy action dim to match num_dof. "
                    f"Got actions_dim={self._action_dim}, num_dof={self._torque_dim}."
                )
            dof_effort_limit_list = env.robot_config.dof_effort_limit_list
            for i, effort in enumerate(dof_effort_limit_list):
                stiffness = self.p_gains[i]
                if stiffness == 0.0:
                    self.action_scales[i] = 0.0
                else:
                    self.action_scales[i] = control_cfg.action_scale * effort / stiffness
        else:
            self.action_scales[:] = control_cfg.action_scale

    # WBC import helpers
    def _repo_root(self) -> Path:
        path = Path(__file__).resolve()
        candidates = [path.parent, *path.parents]

        for candidate in candidates:
            if (candidate.parent / "humanoid-control").exists():
                return candidate

        for candidate in candidates:
            if (candidate / ".git").exists():
                return candidate

        pyproject_candidates = [candidate for candidate in candidates if (candidate / "pyproject.toml").exists()]
        for candidate in reversed(pyproject_candidates):
            if (candidate / "src" / "holosoma").exists():
                return candidate

        if pyproject_candidates:
            return pyproject_candidates[-1]

        return path.parents[6]

    def _humanoid_control_root(self) -> Path:
        return self._repo_root().parent / "humanoid-control"

    def _resolve_wbc_extension_dir(self, extension_dir: str | None = None) -> str | None:
        if extension_dir:
            return str(Path(extension_dir).expanduser().resolve())

        humanoid_control_build = self._humanoid_control_root() / "build"
        if humanoid_control_build.exists():
            return str(humanoid_control_build.resolve())

        return None

    def _resolve_wbc_params(
        self, env: Any, params: dict[str, Any], extension_dir: str | None = None
    ) -> dict[str, Any]:
        resolved_params = dict(params)
        robot_type = getattr(env.robot_config.asset, "robot_type", "")

        if not robot_type.startswith("g1"):
            return resolved_params

        robot_data_root = self._repo_root() / "src" / "holosoma" / "holosoma" / "data" / "robots" / "g1"
        humanoid_control_root = self._humanoid_control_root()

        resolved_params.setdefault("robot_file", str((humanoid_control_root / "models" / "unitree_g1" / "g1.urdf").resolve()))
        resolved_params.setdefault("yaml_file", str((humanoid_control_root / "params" / "g1_parameters.yaml").resolve()))
        resolved_params.setdefault("robot_name", "g1")
        return resolved_params

    def _extension_root(self, extension_dir: str | None = None) -> Path:
        if extension_dir:
            resolved = Path(extension_dir).expanduser().resolve()
            if resolved.name == "build":
                return resolved.parent
            return resolved
        humanoid_control_root = self._humanoid_control_root()
        if humanoid_control_root.exists():
            return humanoid_control_root
        return self._repo_root()

    def _try_add_local_extension_path(self, module_name: str, extension_dir: str | None = None) -> None:
        for candidate in self._find_wbc_extension_candidates(module_name, extension_dir):
            module_dir = str(candidate.parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            return

    def _find_wbc_extension_candidates(self, module_name: str, extension_dir: str | None = None) -> list[Path]:
        search_roots = []
        if extension_dir:
            search_roots.append(Path(extension_dir).expanduser().resolve())
        humanoid_control_root = self._humanoid_control_root()
        search_roots.extend(
            [
                humanoid_control_root / "build",
                humanoid_control_root,
                self._repo_root() / "build",
                self._repo_root(),
            ]
        )

        candidates: list[Path] = []
        for base in search_roots:
            if not base.exists():
                continue
            for pattern in (f"{module_name}*.so", f"{module_name}*.pyd", f"{module_name}*.dylib"):
                candidates.extend(base.rglob(pattern))
        return candidates


    def _import_wbc_module(self, extension_dir: str | None = None):
        module_name = "humanoid_wbc"
        try:
            return importlib.import_module(module_name)
        except Exception:
            self._try_add_local_extension_path(module_name, extension_dir)
            try:
                return importlib.import_module(module_name)
            except Exception as fallback_error:
                candidates = self._find_wbc_extension_candidates(module_name, extension_dir)
                supported_suffixes = importlib.machinery.EXTENSION_SUFFIXES
                candidate_text = ", ".join(str(candidate) for candidate in candidates) or "none"
                raise ModuleNotFoundError(
                    "Could not import humanoid_wbc. "
                    f"Searched extension_dir={extension_dir!r}; found candidates: {candidate_text}. "
                    f"This Python accepts extension suffixes: {supported_suffixes}. "
                    "If a candidate is tagged for another Python version, rebuild humanoid-control "
                    "inside the active Python environment."
                ) from fallback_error

    def _resolve_wbc_asset_paths(
        self, extension_dir: str | None, params: dict[str, Any]
    ) -> tuple[Path, Path, str]:
        extension_root = self._extension_root(extension_dir)
        robot_file = Path(params.get("robot_file", extension_root / "models" / "unitree_g1" / "g1.urdf"))
        yaml_file = Path(params.get("yaml_file", extension_root / "params" / "hrp4c_parameters.yaml"))
        robot_name = params.get("robot_name", "hrp4c")

        if not robot_file.exists() or not yaml_file.exists():
            raise FileNotFoundError(
                "Whole-body controller assets are missing. "
                f"robot_file={robot_file}, yaml_file={yaml_file}"
            )

        return robot_file, yaml_file, robot_name

    def _create_batched_wbc_controller(
        self,
        extension_dir: str | None,
        params: dict[str, Any],
        num_envs: int,
        wbc_module: Any | None = None,
    ):
        if wbc_module is None:
            wbc_module = self._import_wbc_module(extension_dir)
        if not hasattr(wbc_module, "BatchedWbcController"):
            raise RuntimeError(
                "humanoid_wbc.BatchedWbcController is required for batched torque control, "
                "but the imported humanoid_wbc module does not expose it. Rebuild humanoid-control "
                "so the Python extension includes the batched controller bindings."
            )

        robot_file, yaml_file, robot_name = self._resolve_wbc_asset_paths(extension_dir, params)
        return wbc_module.BatchedWbcController(str(robot_file), str(yaml_file), robot_name, int(num_envs))

    def _create_wbc_engine(self, extension_dir: str | None, params: dict[str, Any], wbc_module: Any | None = None):
        if wbc_module is None:
            wbc_module = self._import_wbc_module(extension_dir)
        robot_file, yaml_file, robot_name = self._resolve_wbc_asset_paths(extension_dir, params)
        return wbc_module.WbcEngine(str(robot_file), str(yaml_file), robot_name)
