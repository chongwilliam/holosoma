"""Action terms for joint-level torque control."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from holosoma.managers.action.base import ActionTermBase
from holosoma.managers.action.terms.torque_control_debug import TorqueControlDebugMixin
from holosoma.managers.action.terms.torque_control_wbc import TorqueControlWbcMixin

if TYPE_CHECKING:
    from holosoma.config_types.action import ActionTermCfg
    from holosoma.managers.command.terms.locomotion import LocomotionGait

from holosoma.managers.action.terms.torque_control_support import (
    BatchedSwingFootPlanner,
    BatchedWbcFacade,
    add_landing_residual,
    apply_delta_rotation,
    axis_angle_to_matrix,
    determine_stance_from_phase_fractional,
    exp_map,
    matrix_to_axis_angle,
    matrix_to_urdf_floating_xyz_angles,
    nominal_landing_foot_pose,
    parse_actions,
    phase_from_sin_cos,
    pose_matrix,
    quat_xyzw_from_yaw,
    rot6d_to_matrix,
    root_state_to_base_velocity,
    root_state_to_xyz_rpy,
    root_states_to_xyz_rpy,
    skew,
    string_to_tensor,
    tensor_to_string,
    wrap_to_pi,
    yaw_from_matrix,
    yaw_to_matrix3d,
    yaw_to_rot2d,
)


class JointTorqueActionTerm(TorqueControlDebugMixin, TorqueControlWbcMixin, ActionTermBase):
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
        if self._action_dim != 6:
            raise ValueError(
                "JointTorqueActionTerm now expects the 6-D policy action layout "
                "[com_vel_x, com_vel_y, torso_yaw_vel, landing_foot_delta_x, "
                "landing_foot_delta_y, landing_foot_delta_yaw]. "
                f"Got robot_config.actions_dim={self._action_dim}."
            )
        self._torque_dim = env.num_dof

        # Initialize action buffers
        self._raw_actions = torch.zeros(env.num_envs, self._action_dim, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, self._action_dim, device=env.device)
        self._actions_after_delay = torch.zeros(env.num_envs, self._action_dim, device=env.device)

        # Initialize torque buffer
        self.torques = torch.zeros(env.num_envs, self._torque_dim, device=env.device)

        # Cache previous DOF velocities for derivative control
        self._prev_dof_vel = torch.zeros(env.num_envs, env.num_dof, device=env.device)
        self._non_finite_torque_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

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
        self.action_clip_values = torch.zeros(self._action_dim, dtype=torch.float, device=env.device)

        self._configure_pd_gains(env)
        self._configure_action_scales(env)
        self._configure_action_clip_values(env)

        # Expose references on the environment for backward compatibility
        env.p_gains = self.p_gains
        env.d_gains = self.d_gains
        env.i_gains = self.i_gains
        env.action_scales = self.action_scales
        env.action_clip_values = self.action_clip_values

        # Action delay queue will be initialized in setup() after randomization manager is ready
        self.action_queue: torch.Tensor | None = None

        self._foot_body_indices = self._resolve_foot_body_indices(env)
        self._last_wbc_q = torch.zeros(env.num_envs, env.num_dof + 6, device=env.device)
        self._last_wbc_root_state = torch.zeros(env.num_envs, 13, device=env.device)
        self._last_wbc_dof_pos = torch.zeros(env.num_envs, env.num_dof, device=env.device)
        self._last_wbc_dq = torch.zeros(env.num_envs, env.num_dof + 6, device=env.device)
        self._last_wbc_action_batch: np.ndarray | None = None
        self._last_wbc_right_contact_points: np.ndarray | None = None
        self._last_wbc_left_contact_points: np.ndarray | None = None
        self._last_wbc_right_contact_bases: np.ndarray | None = None
        self._last_wbc_left_contact_bases: np.ndarray | None = None
        self._last_wbc_right_grfs: np.ndarray | None = None
        self._last_wbc_left_grfs: np.ndarray | None = None
        self._last_wbc_torque_output: np.ndarray | None = None
        self._last_wbc_sin_phase = torch.zeros(env.num_envs, 2, device=env.device)
        self._last_wbc_cos_phase = torch.ones(env.num_envs, 2, device=env.device)
        self._last_wbc_desired_state = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._last_wbc_remaining_swing_duration = torch.zeros(env.num_envs, device=env.device)
        self._wbc_bootstrap_enabled = bool(cfg.params.get("dual_stance_bootstrap_enabled", True))
        self._dual_contact_force_threshold = float(cfg.params.get("dual_stance_contact_force_threshold", 10.0))
        self._dual_contact_required_steps = int(cfg.params.get("dual_stance_contact_required_steps", 10))
        if self._dual_contact_required_steps < 1:
            raise ValueError("dual_stance_contact_required_steps must be at least 1.")
        self._visualize_contact_points = bool(
            cfg.params.get("visualize_contact_points", False)
            or cfg.params.get("visualize_wbc_contact_points", False)
            or os.environ.get("HOLOSOMA_VISUALIZE_CONTACT_POINTS", "0") == "1"
        )
        self._visualize_contact_frames = bool(
            cfg.params.get("visualize_contact_frames", False)
            or os.environ.get("HOLOSOMA_VISUALIZE_CONTACT_FRAMES", "0") == "1"
        )
        self._contact_point_radius = float(cfg.params.get("contact_point_radius", 0.018))
        self._visualize_action_targets = bool(
            cfg.params.get("visualize_action_targets", False)
            or cfg.params.get("visualize_wbc_action_targets", False)
            or os.environ.get("HOLOSOMA_VISUALIZE_ACTION_TARGETS", "0") == "1"
        )
        self._visualize_action_target_frames = bool(
            cfg.params.get("visualize_action_target_frames", True)
            and os.environ.get("HOLOSOMA_VISUALIZE_ACTION_TARGET_FRAMES", "1") != "0"
        )
        self._store_wbc_debug_snapshots = bool(
            cfg.params.get("store_wbc_debug_snapshots", False)
            or os.environ.get("HOLOSOMA_STORE_WBC_DEBUG_SNAPSHOTS", "0") == "1"
        )
        self._debug_wbc_finite_checks = bool(
            cfg.params.get("debug_wbc_finite_checks", False)
            or os.environ.get("HOLOSOMA_DEBUG_WBC_FINITE_CHECKS", "0") == "1"
        )
        self._action_target_radius = float(cfg.params.get("action_target_radius", 0.026))
        self._action_target_axis_scale = float(cfg.params.get("action_target_axis_scale", 0.16))
        self._assert_contact_visualization_pose = bool(cfg.params.get("assert_contact_visualization_pose", False))  # debug assert
        self._swing_foot_takeoff_clearance = float(cfg.params.get("swing_foot_takeoff_clearance", 0.05))
        self._swing_foot_landing_clearance = float(cfg.params.get("swing_foot_landing_clearance", 0.05))
        self._wbc_bootstrap_done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._dual_contact_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._wbc_bootstrap_hold_target_dof_pos = torch.zeros(env.num_envs, self._torque_dim, device=env.device)
        self._last_wbc_bootstrap_hold_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._last_wbc_dual_contact_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        resolved_extension_dir = self._resolve_wbc_extension_dir(cfg.wbc_extension_dir)
        resolved_params = self._resolve_wbc_params(env, cfg.params, resolved_extension_dir)
        self._wbc_module = self._import_wbc_module(resolved_extension_dir)
        self.State = self._wbc_module.State
        self.Phase = self._wbc_module.Phase
        self._batched_foot_planner = self._create_batched_foot_planner(self._wbc_module, env.num_envs)
        self._batched_wbc = self._create_batched_wbc_controller(
            resolved_extension_dir, resolved_params, env.num_envs, self._wbc_module
        )
        self._wbc_debug_engine = self._create_wbc_engine(resolved_extension_dir, resolved_params, self._wbc_module)
        self.wbc = BatchedWbcFacade(self._batched_wbc, self._batched_foot_planner, int(self._wbc_debug_engine.dof()), self._wbc_debug_engine)
        self.curr_state = [self.State.DUAL_STANCE for _ in range(env.num_envs)]
        self._prev_wbc_state: Any | None = None
        self.transition_start_time = [0.0 for _ in range(env.num_envs)]
        self._pending_wbc_reinitialize = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
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
        self._assert_finite_torch("raw_actions", actions)

        # Store raw actions
        assert self._raw_actions is not None
        self._raw_actions[:] = actions

        # Clip actions
        if self.env.robot_config.control.clip_actions:
            assert self._processed_actions is not None
            clip_values = self.action_clip_values.to(device=actions.device, dtype=actions.dtype)
            self._processed_actions[:] = torch.clamp(actions, min=-clip_values, max=clip_values)
            # Log clipping fraction
            self.env.log_dict["action_clip_frac"] = (
                actions.abs() >= clip_values
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

        self._assert_finite_torch("actions_after_delay", self._actions_after_delay)

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
        self.torques[:] = self._sanitize_non_finite_torques(self._compute_torques(self._actions_after_delay))
        # Apply torques to simulator
        self.env.simulator.apply_torques_at_dof(self.torques)
        # Cache velocities for next derivative computation
        self._prev_dof_vel.copy_(self.env.simulator.dof_vel)

    @property
    def non_finite_torque_mask(self) -> torch.Tensor:
        return self._non_finite_torque_mask

    @property
    def wbc_bootstrap_hold_mask(self) -> torch.Tensor:
        return self._last_wbc_bootstrap_hold_mask

    def _sanitize_non_finite_torques(self, torques: torch.Tensor) -> torch.Tensor:
        finite_rows = torch.isfinite(torques).flatten(start_dim=1).all(dim=1)
        bad_rows = ~finite_rows
        self._non_finite_torque_mask |= bad_rows
        if bool(finite_rows.all()):
            return torques
        return torch.where(bad_rows.unsqueeze(1), torch.zeros_like(torques), torques)

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute torques from the in-process whole-body controller.

        Args:
            actions: Action tensor [num_envs, action_dim]

        Returns:
            Torque tensor [num_envs, num_dof]
        """
        self._assert_finite_torch("wbc_actions", actions)

        num_envs = actions.shape[0]
        if num_envs != self.env.num_envs:
            raise RuntimeError(
                "BatchedWbcController requires one action row per environment. "
                f"Got actions.shape[0]={num_envs}, env.num_envs={self.env.num_envs}."
            )

        # WBC task-space actions are pre-scaled per command dimension in Python.
        # Keep the binding scale at 1.0 to avoid applying a second uniform multiplier.
        action_scale = 1.0

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
        self._assert_finite_torch_debug("wbc_root_states", root_states)
        self._assert_finite_torch_debug("wbc_dof_pos", dof_pos)
        self._assert_finite_torch_debug("wbc_dof_vel", dof_vel)
        self._assert_finite_torch_debug("wbc_q", q_tensor)
        self._assert_finite_torch_debug("wbc_dq", dq_tensor)

        self._last_wbc_q[:num_envs] = q_tensor
        self._last_wbc_dq[:num_envs] = dq_tensor
        self._last_wbc_root_state[:num_envs] = root_states
        self._last_wbc_dof_pos[:num_envs] = dof_pos

        right_contact_points = self._as_numpy_2d(self.env.simulator.right_foot_contact_position, "right_contact_points")
        left_contact_points = self._as_numpy_2d(self.env.simulator.left_foot_contact_position, "left_contact_points")
        right_contact_bases = self._contact_bases_as_numpy(self.env.simulator.right_foot_contact_basis, "right_contact_bases")
        left_contact_bases = self._contact_bases_as_numpy(self.env.simulator.left_foot_contact_basis, "left_contact_bases")
        # right_grfs = self._batched_foot_ground_reaction_wrenches("right")
        # left_grfs = self._batched_foot_ground_reaction_wrenches("left")
        right_grfs = self._batched_local_foot_ground_reaction_wrenches("right") # local grf
        left_grfs = self._batched_local_foot_ground_reaction_wrenches("left")
        right_contact_in_contact = self._batched_foot_contact_in_contact("right", right_grfs)
        left_contact_in_contact = self._batched_foot_contact_in_contact("left", left_grfs)

        q = self._as_numpy_2d(q_tensor, "q")
        dq = self._as_numpy_2d(dq_tensor, "dq")
        sin_phase, cos_phase = self._phase_features_for_wbc(actions)
        desired_states, swing_sides, remaining_swing_durations = self._desired_states_from_phase(sin_phase, cos_phase)
        action_batch = self._actions_for_batched_wbc(
            actions,
            desired_states=desired_states,
            swing_sides=swing_sides,
            remaining_swing_durations=remaining_swing_durations,
        )

        self._last_wbc_sin_phase[:num_envs] = sin_phase
        self._last_wbc_cos_phase[:num_envs] = cos_phase
        self._last_wbc_desired_state[:num_envs] = torch.as_tensor(
            desired_states, device=self.env.device, dtype=torch.long
        )
        self._last_wbc_remaining_swing_duration[:num_envs] = torch.as_tensor(
            remaining_swing_durations, device=self.env.device, dtype=torch.float32
        )
        self._maybe_store_wbc_debug_snapshots(
            action_batch,
            right_contact_points,
            left_contact_points,
            right_contact_bases,
            left_contact_bases,
            right_grfs,
            left_grfs,
        )
        self._assert_finite_numpy_debug("wbc_right_contact_points", right_contact_points)
        self._assert_finite_numpy_debug("wbc_left_contact_points", left_contact_points)
        self._assert_finite_numpy_debug("wbc_right_contact_bases", right_contact_bases)
        self._assert_finite_numpy_debug("wbc_left_contact_bases", left_contact_bases)
        self._assert_finite_numpy_debug("wbc_right_grfs", right_grfs)
        self._assert_finite_numpy_debug("wbc_left_grfs", left_grfs)
        self._assert_finite_numpy_debug("wbc_action_batch", action_batch)
        self._maybe_draw_action_targets(root_states, action_batch)
        self._maybe_draw_contact_points(
            right_contact_points,
            right_contact_bases,
            left_contact_points,
            left_contact_bases,
        )
        bootstrap_hold_mask: torch.Tensor | None = None
        if self._wbc_bootstrap_enabled:
            self._update_wbc_bootstrap_state(right_grfs, left_grfs)
            bootstrap_hold_mask = ~self._wbc_bootstrap_done
            if bool(bootstrap_hold_mask.all()):
                hold_torques = self._compute_bootstrap_hold_torques(dof_pos, dof_vel)
                if self._store_wbc_debug_snapshots:
                    self._last_wbc_torque_output = hold_torques.detach().cpu().numpy().astype(np.float64, copy=True)
                return hold_torques
            if bool(self._pending_wbc_reinitialize.any()):
                self._reinitialize_wbc_dual_stance(q, dq)
        else:
            self._last_wbc_bootstrap_hold_mask.zero_()
            self._last_wbc_dual_contact_mask.fill_(True)

        if bool(self._pending_wbc_reinitialize.any()):
            self._reinitialize_wbc_dual_stance(q, dq)
        if self._needs_wbc_debug_engine_update():
            self._wbc_debug_engine.updateRobot(q[0], dq[0])
        try:
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
                right_contact_in_contact,
                left_contact_in_contact,
                float(self.env.simulator.time()),
                action_scale,
            )
        except RuntimeError as exc:
            env_idx = self._env_idx_from_wbc_exception(exc)
            if env_idx is not None:
                context = self._wbc_input_context_summary(
                    env_idx,
                    q,
                    dq,
                    action_batch,
                    right_contact_points,
                    left_contact_points,
                    right_contact_bases,
                    left_contact_bases,
                    right_grfs,
                    left_grfs,
                    right_contact_in_contact,
                    left_contact_in_contact,
                )
                raise RuntimeError(f"{exc}\n{context}") from exc
            raise
        torque_np = self._actuated_torques_from_batched_wbc_output(torque_wbc)
        torques = torch.as_tensor(
            torque_np,
            device=actions.device,
            dtype=self.torques.dtype,
        )
        if bootstrap_hold_mask is not None and bool(bootstrap_hold_mask.any()):
            hold_torques = self._compute_bootstrap_hold_torques(dof_pos, dof_vel)
            torques = torch.where(bootstrap_hold_mask.unsqueeze(1), hold_torques, torques)
            torque_np = torques.detach().cpu().numpy().astype(np.float64, copy=True)

        if self._store_wbc_debug_snapshots:
            self._last_wbc_torque_output = torque_np.copy()

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

    def _capture_bootstrap_hold_targets(self, env_ids: torch.Tensor | None) -> None:
        dof_pos = self._as_torch_tensor(
            self.env.simulator.dof_pos,
            device=self.env.device,
            dtype=self._wbc_bootstrap_hold_target_dof_pos.dtype,
            label="bootstrap_hold_dof_pos",
        )
        if env_ids is None:
            self._wbc_bootstrap_hold_target_dof_pos[:] = dof_pos
        else:
            self._wbc_bootstrap_hold_target_dof_pos[env_ids] = dof_pos[env_ids]

        self._assert_finite_torch("bootstrap_hold_target_dof_pos", self._wbc_bootstrap_hold_target_dof_pos)

    def _foot_contact_count_mask(self, side: str) -> np.ndarray:
        counts = getattr(self.env.simulator, f"{side}_foot_contact_count", None)
        if counts is None:
            return np.ones(self.env.num_envs, dtype=bool)

        counts_tensor = self._as_torch_tensor(
            counts,
            device=self.env.device,
            dtype=torch.long,
            label=f"{side}_foot_contact_count",
        ).reshape(-1)
        return np.ascontiguousarray((counts_tensor.detach().cpu().numpy() > 0).astype(bool, copy=False))

    def _batched_foot_contact_in_contact(self, side: str, local_wrenches: np.ndarray | None = None) -> np.ndarray:
        if local_wrenches is None:
            local_wrenches = np.stack(
                [self._local_foot_force_sensor_wrench(env_idx, side) for env_idx in range(self.env.num_envs)],
                axis=0,
            ).astype(np.float64, copy=False)
        local_z_force = np.abs(local_wrenches[:, 2])
        contact = (local_z_force > self._dual_contact_force_threshold) & self._foot_contact_count_mask(side)
        return np.ascontiguousarray(contact.astype(np.bool_, copy=False))

    def _update_wbc_bootstrap_state(self, right_grfs: np.ndarray, left_grfs: np.ndarray) -> None:
        right_force = np.linalg.norm(right_grfs[:, :3], axis=1)
        left_force = np.linalg.norm(left_grfs[:, :3], axis=1)
        right_contact = (right_force >= self._dual_contact_force_threshold) & self._foot_contact_count_mask("right")
        left_contact = (left_force >= self._dual_contact_force_threshold) & self._foot_contact_count_mask("left")
        dual_contact = torch.as_tensor(right_contact & left_contact, device=self.env.device, dtype=torch.bool)

        self._last_wbc_dual_contact_mask[:] = dual_contact
        waiting = ~self._wbc_bootstrap_done
        self._dual_contact_counter[waiting & dual_contact] += 1
        self._dual_contact_counter[waiting & ~dual_contact] = 0

        ready = waiting & (self._dual_contact_counter >= self._dual_contact_required_steps)
        if bool(ready.any()):
            self._wbc_bootstrap_done[ready] = True
            sim_time = float(self.env.simulator.time())
            for env_idx in ready.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
                self.curr_state[env_idx] = self.State.DUAL_STANCE
                self.transition_start_time[env_idx] = sim_time
            self._pending_wbc_reinitialize[ready] = True

        self._last_wbc_bootstrap_hold_mask[:] = ~self._wbc_bootstrap_done

    def _compute_bootstrap_hold_torques(self, dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> torch.Tensor:
        hold_torques = (
            self._kp_scale * self.p_gains.unsqueeze(0) * (self._wbc_bootstrap_hold_target_dof_pos - dof_pos)
            - self._kd_scale * self.d_gains.unsqueeze(0) * dof_vel
        )
        if self.env.robot_config.control.clip_torques:
            torque_limits = self.env.torque_limits.to(device=hold_torques.device, dtype=hold_torques.dtype)
            hold_torques = torch.clamp(hold_torques, min=-torque_limits, max=torque_limits)

        self._assert_finite_torch("bootstrap_hold_torques", hold_torques)
        return hold_torques

    def _reinitialize_wbc_dual_stance(self, q: np.ndarray, dq: np.ndarray) -> None:
        env_ids = self._pending_wbc_reinitialize.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return

        sim_time = float(self.env.simulator.time())
        self.wbc.update_robot(q, dq)
        env_id_list = [int(env_idx) for env_idx in env_ids.detach().cpu().tolist()]
        for env_idx in env_id_list:
            self.wbc.reset_state(int(self.State.DUAL_STANCE), sim_time, env_idx)

        for env_idx in env_id_list:
            self.curr_state[env_idx] = self.State.DUAL_STANCE
            self.transition_start_time[env_idx] = sim_time
        self._pending_wbc_reinitialize[env_ids] = False

    def _actions_for_batched_wbc(
        self,
        actions: torch.Tensor,
        *,
        desired_states: np.ndarray | None = None,
        swing_sides: list[str | None] | None = None,
        remaining_swing_durations: np.ndarray | None = None,
    ) -> np.ndarray:
        action_scales = self.action_scales
        if action_scales.device != actions.device or action_scales.dtype != actions.dtype:
            action_scales = action_scales.to(device=actions.device, dtype=actions.dtype)
        scaled_actions = actions * action_scales
        self._assert_finite_torch_debug("wbc_scaled_actions", scaled_actions)
        if scaled_actions.shape[-1] != 6:
            raise ValueError(f"Expected 6-D policy actions for WBC expansion, got shape {tuple(scaled_actions.shape)}.")

        if desired_states is None or swing_sides is None or remaining_swing_durations is None:
            sin_phase, cos_phase = self._phase_features_for_wbc(actions)
            desired_states, swing_sides, remaining_swing_durations = self._desired_states_from_phase(sin_phase, cos_phase)

        wbc_actions = self._six_dim_actions_to_wbc_targets(
            scaled_actions,
            desired_states=desired_states,
            swing_sides=swing_sides,
            remaining_swing_durations=remaining_swing_durations,
        )
        self._assert_finite_numpy_debug("wbc_expanded_actions", wbc_actions)
        return np.ascontiguousarray(wbc_actions)

    def _six_dim_actions_to_wbc_targets(
        self,
        actions: torch.Tensor,
        *,
        desired_states: np.ndarray,
        swing_sides: list[str | None],
        remaining_swing_durations: np.ndarray,
    ) -> np.ndarray:
        num_envs = actions.shape[0]
        action_dict = parse_actions(actions)
        action_np = {
            key: value.detach().cpu().numpy().astype(np.float64, copy=False)
            for key, value in action_dict.items()
        }

        wbc_actions = np.zeros((num_envs, 15), dtype=np.float64)
        wbc_actions[:, :2] = action_np["com_vel"] * float(self.env.dt)

        current_poses = self._batched_foot_pose_matrices()
        landing_poses: dict[int, tuple[str, np.ndarray]] = {}
        left_stance_state = int(self.State.LEFT_STANCE)
        right_stance_state = int(self.State.RIGHT_STANCE)
        for env_idx, fallback_swing_side in enumerate(swing_sides[:num_envs]):
            desired_state = int(desired_states[env_idx])
            if desired_state == left_stance_state:
                swing_side = "right"
            elif desired_state == right_stance_state:
                swing_side = "left"
            else:
                swing_side = fallback_swing_side
            if swing_side is None:
                continue

            stance_side = "right" if swing_side == "left" else "left"
            stance_pose = current_poses[stance_side][env_idx]
            stance_pos = stance_pose[:3, 3]
            stance_yaw = yaw_from_matrix(stance_pose[:3, :3])
            desired_velocity = np.array(
                [
                    action_np["com_vel"][env_idx, 0],
                    action_np["com_vel"][env_idx, 1],
                    action_np["torso_yaw_vel"][env_idx, 0],
                ],
                dtype=float,
            )
            landing_pos, landing_yaw = nominal_landing_foot_pose(
                stance_pos,
                stance_yaw,
                desired_velocity,
                swing_is_left=(swing_side == "left"),
                step_time=max(float(remaining_swing_durations[env_idx]), float(self.env.dt)),
            )
            landing_pos, landing_yaw = add_landing_residual(
                landing_pos,
                landing_yaw,
                stance_pos,
                stance_yaw,
                action_np["landing_foot_delta_pose"][env_idx],
                swing_is_left=(swing_side == "left"),
            )
            landing_poses[env_idx] = (swing_side, pose_matrix(landing_pos, yaw_to_matrix3d(landing_yaw)))

        desired_poses = self._batched_foot_planner.desired_pose_batch(
            current_poses=current_poses,
            landing_poses=landing_poses,
            remaining_durations=remaining_swing_durations,
        )
        for env_idx, (swing_side, desired_pose) in desired_poses.items():
            current_pose = current_poses[swing_side][env_idx]
            pos_delta = desired_pose[:3, 3] - current_pose[:3, 3]
            rot_delta = current_pose[:3, :3].T @ desired_pose[:3, :3]
            ori_delta = matrix_to_axis_angle(rot_delta)
            if swing_side == "right":
                wbc_actions[env_idx, 3:6] = pos_delta
                wbc_actions[env_idx, 6:9] = ori_delta
            else:
                wbc_actions[env_idx, 9:12] = pos_delta
                wbc_actions[env_idx, 12:15] = ori_delta

        return wbc_actions

    def _phase_features_for_wbc(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gait_state = self.env.command_manager.get_state("locomotion_gait")
        if gait_state is None:
            sin_phase = torch.zeros(self.env.num_envs, 2, device=actions.device, dtype=actions.dtype)
            cos_phase = torch.ones_like(sin_phase)
            return sin_phase, cos_phase

        gait_state = cast("LocomotionGait", gait_state)
        phase = gait_state.phase
        if phase is None:
            sin_phase = torch.zeros(self.env.num_envs, 2, device=actions.device, dtype=actions.dtype)
            cos_phase = torch.ones_like(sin_phase)
            return sin_phase, cos_phase

        phase = phase.to(device=actions.device, dtype=actions.dtype)
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)
        self._assert_finite_torch("wbc_sin_phase", sin_phase)
        self._assert_finite_torch("wbc_cos_phase", cos_phase)
        return sin_phase, cos_phase

    def _desired_states_from_phase(
        self,
        sin_phase: torch.Tensor,
        cos_phase: torch.Tensor,
        *,
        left_stance_frac: float = 0.40,
        dual_frac: float = 0.10,
        right_stance_frac: float = 0.40,
    ) -> tuple[np.ndarray, list[str | None], np.ndarray]:
        total = left_stance_frac + dual_frac + right_stance_frac + dual_frac
        if abs(total - 1.0) > 1.0e-6:
            raise ValueError(f"Phase fractions must sum to 1. Got {total}.")

        left_phase = torch.atan2(sin_phase[:, 0], cos_phase[:, 0])
        normalized_phase = torch.remainder(left_phase, 2.0 * torch.pi) / (2.0 * torch.pi)
        phase_np = normalized_phase.detach().cpu().numpy().astype(np.float64, copy=False)
        cycle_time = self._gait_cycle_times(sin_phase.device, sin_phase.dtype).detach().cpu().numpy()

        desired_states = np.full(self.env.num_envs, int(self.State.DUAL_STANCE), dtype=np.int64)
        swing_sides: list[str | None] = [None for _ in range(self.env.num_envs)]
        remaining = np.maximum(cycle_time * dual_frac, float(self.env.dt))

        left_end = left_stance_frac
        dual_1_end = left_end + dual_frac
        right_end = dual_1_end + right_stance_frac
        for env_idx, phase in enumerate(phase_np):
            if phase < left_end:
                desired_states[env_idx] = int(self.State.LEFT_STANCE)
                swing_sides[env_idx] = "right"
                remaining[env_idx] = max((left_end - phase) * cycle_time[env_idx], float(self.env.dt))
            elif phase < dual_1_end:
                desired_states[env_idx] = int(self.State.DUAL_STANCE)
                swing_sides[env_idx] = None
                remaining[env_idx] = max((dual_1_end - phase) * cycle_time[env_idx], float(self.env.dt))
            elif phase < right_end:
                desired_states[env_idx] = int(self.State.RIGHT_STANCE)
                swing_sides[env_idx] = "left"
                remaining[env_idx] = max((right_end - phase) * cycle_time[env_idx], float(self.env.dt))
            else:
                desired_states[env_idx] = int(self.State.DUAL_STANCE)
                swing_sides[env_idx] = None
                remaining[env_idx] = max((1.0 - phase) * cycle_time[env_idx], float(self.env.dt))

        return desired_states, swing_sides, remaining.astype(np.float64, copy=False)

    def _gait_cycle_times(self, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
        gait_state = self.env.command_manager.get_state("locomotion_gait")
        if gait_state is not None:
            gait_state = cast("LocomotionGait", gait_state)
            gait_freq = gait_state.gait_freq
            if gait_freq is not None:
                freq = gait_freq.reshape(-1).to(device=device, dtype=dtype).clamp_min(1.0e-6)
                return 1.0 / freq
            gait_period = float(getattr(gait_state, "gait_period", 1.0))
            return torch.full((self.env.num_envs,), gait_period, device=device, dtype=dtype)
        return torch.full((self.env.num_envs,), 1.0, device=device, dtype=dtype)

    def _batched_foot_pose_matrices(self) -> dict[str, np.ndarray]:
        poses: dict[str, np.ndarray] = {}
        for side in ("right", "left"):
            side_poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], self.env.num_envs, axis=0)
            for env_idx in range(self.env.num_envs):
                foot_pose = self._body_pose_for_target(env_idx, self._foot_body_indices.get(side))
                if foot_pose is None:
                    continue
                foot_pos, foot_rot = foot_pose
                side_poses[env_idx] = pose_matrix(foot_pos, foot_rot)
            poses[side] = np.ascontiguousarray(side_poses)
        return poses

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
        grfs[force_norms < self._dual_contact_force_threshold] = 0.0
        return np.ascontiguousarray(grfs)
    
    def _batched_local_foot_ground_reaction_wrenches(self, side: str) -> np.ndarray:
        grfs = np.stack(
            [self._local_foot_force_sensor_wrench(env_idx, side) for env_idx in range(self.env.num_envs)],
            axis=0,
        ).astype(np.float64, copy=False)
        force_norms = np.linalg.norm(grfs[:, :3], axis=1)
        grfs[force_norms < self._dual_contact_force_threshold] = 0.0
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
            self._non_finite_torque_mask.zero_()
        else:
            self.torques[env_ids] = 0.0
            self._non_finite_torque_mask[env_ids] = False

        # Reset cached velocities
        if env_ids is None:
            self._prev_dof_vel.zero_()
        else:
            self._prev_dof_vel[env_ids] = 0.0

        if self._wbc_bootstrap_enabled:
            self._capture_bootstrap_hold_targets(env_ids)
            if env_ids is None:
                self._wbc_bootstrap_done.zero_()
                self._dual_contact_counter.zero_()
                self._last_wbc_bootstrap_hold_mask.fill_(True)
                self._last_wbc_dual_contact_mask.zero_()
            else:
                self._wbc_bootstrap_done[env_ids] = False
                self._dual_contact_counter[env_ids] = 0
                self._last_wbc_bootstrap_hold_mask[env_ids] = True
                self._last_wbc_dual_contact_mask[env_ids] = False
        else:
            if env_ids is None:
                self._wbc_bootstrap_done.fill_(True)
                self._dual_contact_counter.zero_()
                self._last_wbc_bootstrap_hold_mask.zero_()
                self._last_wbc_dual_contact_mask.fill_(True)
            else:
                self._wbc_bootstrap_done[env_ids] = True
                self._dual_contact_counter[env_ids] = 0
                self._last_wbc_bootstrap_hold_mask[env_ids] = False
                self._last_wbc_dual_contact_mask[env_ids] = True

        for env_idx in (
            range(self.env.num_envs)
            if env_ids is None
            else env_ids.detach().cpu().flatten().tolist()
        ):
            self.curr_state[int(env_idx)] = self.State.DUAL_STANCE
            self.transition_start_time[int(env_idx)] = float(self.env.simulator.time())

        planner_env_ids = (
            range(self.env.num_envs)
            if env_ids is None
            else [int(env_idx) for env_idx in env_ids.detach().cpu().flatten().tolist()]
        )
        self._batched_foot_planner.reset(planner_env_ids)

        if env_ids is None:
            self._pending_wbc_reinitialize[:] = not self._wbc_bootstrap_enabled
        else:
            self._pending_wbc_reinitialize[env_ids] = not self._wbc_bootstrap_enabled

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

    def _local_foot_force_sensor_wrench(self, env_idx: int, side: str) -> np.ndarray:
        """Return the simulator foot force sensor wrench expressed in the foot sensor frame."""
        sensor_wrench_fn = getattr(self.env.simulator, "get_local_foot_force_sensor_wrench", None)
        if callable(sensor_wrench_fn):
            try:
                return sensor_wrench_fn(side, env_idx).detach().cpu().numpy().astype(float, copy=False)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to read local foot force/torque sensor wrench via "
                    f"env.simulator.get_local_foot_force_sensor_wrench(side={side!r}, env_idx={env_idx})."
                ) from exc

        raise RuntimeError(
            "env.simulator does not provide get_local_foot_force_sensor_wrench(). "
            "The WBC contact flags require the XML foot force sensor wrench expressed in the local frame."
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
            if not isinstance(control_cfg.action_scale, (float, int)):
                raise ValueError("action_scales_by_effort_limit_over_p_gain requires scalar action_scale.")
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
            self.action_scales[:] = self._action_param_tensor(control_cfg.action_scale, "action_scale")

    def _configure_action_clip_values(self, env: Any) -> None:
        self.action_clip_values[:] = self._action_param_tensor(
            env.robot_config.control.action_clip_value, "action_clip_value"
        )

    def _action_param_tensor(self, value: float | list[float] | tuple[float, ...], label: str) -> torch.Tensor:
        if isinstance(value, (float, int)):
            return torch.full((self._action_dim,), float(value), device=self.env.device, dtype=torch.float)

        tensor = torch.as_tensor(value, device=self.env.device, dtype=torch.float)
        if tensor.ndim != 1 or tensor.shape[0] != self._action_dim:
            raise ValueError(
                f"robot_config.control.{label} must be a scalar or length-{self._action_dim} sequence, "
                f"got shape {tuple(tensor.shape)}."
            )
        return tensor
