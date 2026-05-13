from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

WBC_TASK_SPACE_ACTION_DIM = 12
WBC_PELVIS_LIN_VEL_SLICE = slice(0, 3)
WBC_PELVIS_ANG_VEL_SLICE = slice(3, 6)
WBC_COM_VEL_SLICE = slice(6, 9)
WBC_LANDING_DELTA_SLICE = slice(9, 12)


class SymmetryUtils:
    """Symmetry utilities for humanoid robots using configuration-driven approach.

    This class provides symmetry transformations for data augmentation and mirroring
    operations. All robot-specific information is loaded from the environment's
    robot configuration.
    """

    def __init__(self, env: Any) -> None:
        """Initializes the symmetry utilities.

        Parameters
        ----------
        env : Any
            Environment object containing robot configuration and observation specifications.
        """
        self.env = env
        self.robot_config = env.robot_config

        # Initialize attributes that will be set during initialization
        self.observation_dims: Dict[str, int] = {}
        self.observation_dims_single_frame: Dict[str, int] = {}  # Dimension without history
        self.history_lengths: Dict[str, int] = {}
        self.sub_observation_keys: Dict[str, List[str]] = {}
        self.sub_observation_indices: Dict[str, Dict[str, torch.Tensor]] = {}
        self.sub_observation_indices_single_frame: Dict[
            str, Dict[str, torch.Tensor]
        ] = {}  # Indices within single frame
        self.sub_observation_dims: Dict[str, int] = {}
        self.joint_index_map: torch.Tensor = torch.empty(0)
        self.sign_flip_mask: torch.Tensor = torch.empty(0)

        self._init_observation_config()
        self._init_robot_config()

    def _init_observation_config(self) -> None:
        """Initializes observation configuration from environment.

        This method sets up observation dimensions, indices, and keys required for parsing
        and transforming observation tensors. This is generic across all humanoid robots.
        """
        self.history_lengths = self.env.history_length

        # Check if using observation manager or direct system
        if getattr(self.env, "observation_manager", None):
            # Observation manager system - get info from observation manager
            self._init_observation_config_from_manager()
        else:
            # Direct observation system
            self._init_observation_config_direct()

    def _init_observation_config_direct(self) -> None:
        """Initialize observation config from direct system."""
        # sub_observation_dims contains the dimension of each observation component
        # e.g.) sub_observation_dims: {base_ang_vel: 3, projected_gravity: 3, ...}
        # observation_dims contains the total dimension per observation type (including history)
        # e.g.) observation_dims: {actor_obs: (3 + 3 + ...) * history_length}
        self.sub_observation_dims = self.env.config.obs.obs_dims
        self.observation_dims = self.env.dim_obs

        self.sub_observation_indices = {}
        self.sub_observation_indices_single_frame = {}
        self.sub_observation_keys = {}
        self.observation_dims_single_frame = {}

        for obs_key, obs_config in self.env.config.obs.obs_dict.items():
            # e.g.) obs_key: actor_obs, obs_config: [base_ang_vel, projected_gravity, ...]
            sub_obs_keys = sorted(obs_config)
            self.sub_observation_keys[obs_key] = sub_obs_keys
            self.sub_observation_indices[obs_key] = {}
            self.sub_observation_indices_single_frame[obs_key] = {}
            idx = 0
            for key in sub_obs_keys:
                # Indices for single frame (used during mirroring after reshape)
                self.sub_observation_indices_single_frame[obs_key][key] = torch.arange(
                    idx, idx + self.sub_observation_dims[key], device=self.env.device
                )
                # For direct config, observations may not have explicit history handling in indices
                # These indices work with the full flattened observation
                self.sub_observation_indices[obs_key][key] = torch.arange(
                    idx, idx + self.sub_observation_dims[key], device=self.env.device
                )
                idx += self.sub_observation_dims[key]

            # Store single-frame dimension (sum of all sub-observation dimensions)
            self.observation_dims_single_frame[obs_key] = idx

    def _init_observation_config_from_manager(self) -> None:
        """Initialize observation config from observation manager."""
        import torch

        obs_manager = self.env.observation_manager

        # Compute sub_observation_dims: dimension of each observation term
        self.sub_observation_dims = {}
        self.observation_dims = {}
        self.observation_dims_single_frame = {}
        self.sub_observation_keys = {}
        self.sub_observation_indices = {}
        self.sub_observation_indices_single_frame = {}

        for group_name, group_cfg in obs_manager.cfg.groups.items():
            # Compute dimension for each term in this group
            term_keys = sorted(group_cfg.terms.keys())
            self.sub_observation_keys[group_name] = term_keys
            self.sub_observation_indices[group_name] = {}
            self.sub_observation_indices_single_frame[group_name] = {}

            idx = 0
            idx_single_frame = 0
            for term_name in term_keys:
                # Compute the term once to get its dimension
                term_cfg = group_cfg.terms[term_name]
                obs = obs_manager._compute_term(group_name, term_name, term_cfg)
                term_dim = obs.shape[1]

                # Store the base dimension (without history)
                self.sub_observation_dims[term_name] = term_dim

                # Create indices for this term in the concatenated observation
                # Account for history at group level
                if group_cfg.history_length > 1:
                    term_dim_with_history = term_dim * group_cfg.history_length
                else:
                    term_dim_with_history = term_dim

                # Indices for full observation (with history flattened)
                self.sub_observation_indices[group_name][term_name] = torch.arange(
                    idx, idx + term_dim_with_history, device=self.env.device
                )
                # Indices for single frame (used during mirroring after reshape)
                self.sub_observation_indices_single_frame[group_name][term_name] = torch.arange(
                    idx_single_frame, idx_single_frame + term_dim, device=self.env.device
                )
                idx += term_dim_with_history
                idx_single_frame += term_dim

            # Total dimension for this group (with history)
            self.observation_dims[group_name] = idx
            # Total dimension for this group (single frame, without history)
            self.observation_dims_single_frame[group_name] = idx_single_frame

    def _init_robot_config(self) -> None:
        """Initializes robot configuration from config files.

        This method loads joint mappings and sign flip configurations directly from
        the robot config.
        """
        # Validate required config attributes
        if not self.robot_config.dof_names:
            raise ValueError("Robot config must contain 'dof_names' list")
        if not self.robot_config.symmetry_joint_names:
            raise ValueError("Robot config must contain 'symmetry_joint_names' mapping")
        if not self.robot_config.flip_sign_joint_names:
            raise ValueError("Robot config must contain 'flip_sign_joint_names' list")

        # Get joint configuration from robot config
        dof_names = self.robot_config.dof_names
        joint_name_mapping = self.robot_config.symmetry_joint_names
        sign_flip_joints = self.robot_config.flip_sign_joint_names

        # Create name to index mapping
        name_to_idx = {name: idx for idx, name in enumerate(dof_names)}

        # Build joint index mapping
        joint_index_mapping = {}
        for joint1, joint2 in joint_name_mapping.items():
            if joint1 in name_to_idx and joint2 in name_to_idx:
                joint_index_mapping[name_to_idx[joint1]] = name_to_idx[joint2]

        # Create joint mapping tensor
        self.joint_index_map = torch.tensor(
            [joint_index_mapping.get(i, i) for i in range(len(dof_names))], device=self.env.device, dtype=torch.long
        )

        # Create sign flip mask
        sign_flip_indices = {name_to_idx[name] for name in sign_flip_joints if name in name_to_idx}
        self.sign_flip_mask = torch.tensor(
            [-1.0 if i in sign_flip_indices else 1.0 for i in range(len(dof_names))],
            device=self.env.device,
            dtype=torch.float,
        )

    def augment_observations(self, obs: torch.Tensor, env: Any, obs_list: Sequence[str]) -> torch.Tensor:
        """Applies x-z plane symmetry transformation for observation data augmentation.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor with shape [batch_size, obs_dim].
            Contains robot state observations to be augmented.
        env : object
            Environment object (passed for compatibility).
        obs_list : Sequence[str]
            List of observation component names to process.
            Example: ["actor_obs"] or ["actor_state_obs", "perception_obs"].

        Returns
        -------
        torch.Tensor
            Augmented observation tensor with doubled batch size.
            Original data is concatenated with mirrored data along batch dimension.
        """
        mirrored_obs = self.mirror_xz_plane(obs, env, obs_list)
        return torch.cat((obs, mirrored_obs), dim=0)

    def augment_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Applies x-z plane symmetry transformation for action data augmentation.

        Parameters
        ----------
        actions : torch.Tensor
            Action tensor with shape [batch_size, action_dim].
            Contains robot actions to be augmented.

        Returns
        -------
        torch.Tensor
            Augmented action tensor with doubled batch size.
            Original data is concatenated with mirrored data along batch dimension.
        """
        mirrored_actions = self.mirror_action_xz_plane(actions)
        return torch.cat((actions, mirrored_actions), dim=0)

    def mirror_xz_plane(self, observation: torch.Tensor, env: Any, obs_list: Sequence[str]) -> torch.Tensor:
        """Performs x-z plane symmetry transformation on observation tensor.

        This function parses the observation tensor and applies appropriate mirroring
        to each component based on its physical meaning and coordinate system.

        Parameters
        ----------
        observation : torch.Tensor
            Input observation tensor with shape [batch_size, obs_dim].
            Contains concatenated observation components as specified in obs_list.
        env : object
            Environment object (passed for compatibility).
        obs_list : List[str]
            List of observation component names that define the structure of the observation tensor.

        Returns
        -------
        torch.Tensor
            Mirrored observation tensor with same shape as input.
            Y-axis components are typically negated, joint configurations are remapped.
        """
        # Create a copy of the observation
        mirrored_obs_all = observation.clone()
        B, _ = mirrored_obs_all.shape

        # Parse observation components and apply mirroring
        idx = 0

        for obs_key in obs_list:
            cur_obs_length = self.observation_dims[obs_key]
            mirrored_obs = mirrored_obs_all[..., idx : idx + cur_obs_length]
            # Reshape to [batch, history_length, single_frame_obs_dim]
            mirrored_obs = mirrored_obs.reshape(
                B, self.history_lengths[obs_key], self.observation_dims_single_frame[obs_key]
            )
            original_obs = mirrored_obs.clone()
            # Apply mirroring to each sub-observation component using single-frame indices
            for sub_obs_key in self.sub_observation_keys[obs_key]:
                source_obs_key = {
                    "right_foot_pos": "left_foot_pos",
                    "right_foot_ori": "left_foot_ori",
                    "right_foot_lin_vel": "left_foot_lin_vel",
                    "right_foot_ang_vel": "left_foot_ang_vel",
                    "right_foot_vel": "left_foot_vel",
                    "left_foot_pos": "right_foot_pos",
                    "left_foot_ori": "right_foot_ori",
                    "left_foot_lin_vel": "right_foot_lin_vel",
                    "left_foot_ang_vel": "right_foot_ang_vel",
                    "left_foot_vel": "right_foot_vel",
                }.get(sub_obs_key)
                if source_obs_key not in self.sub_observation_indices_single_frame[obs_key]:
                    source_obs_key = sub_obs_key
                target_idx = self.sub_observation_indices_single_frame[obs_key][sub_obs_key]
                source_idx = self.sub_observation_indices_single_frame[obs_key][source_obs_key]
                mirrored_obs[..., target_idx] = getattr(self, f"mirror_obs_{sub_obs_key}")(
                    original_obs[..., source_idx]
                )
            mirrored_obs_all[..., idx : idx + cur_obs_length] = mirrored_obs.reshape(B, cur_obs_length)
            idx += cur_obs_length

        return mirrored_obs_all

    def mirror_action_xz_plane(self, action: torch.Tensor) -> torch.Tensor:
        """Performs x-z plane symmetry transformation on action tensor.

        Parameters
        ----------
        action : torch.Tensor
            Input action tensor with shape [batch_size, n_dofs].
            Contains joint position commands or torques with original action order [a0, a1, a2, ..., aN].

        Returns
        -------
        torch.Tensor
            Mirrored action tensor with same shape as input.
            Joint mappings are applied and signs are flipped as appropriate.
            Returns [a_mapped0 * sign0, a_mapped1 * sign1, ..., a_mappedN * signN] (mirrored and sign-flipped).
        """
        if action.shape[-1] == WBC_TASK_SPACE_ACTION_DIM:
            return self._mirror_wbc_task_space_action(action)

        if action.shape[-1] != self.joint_index_map.shape[0]:
            raise ValueError(
                f"Unsupported action dimension for symmetry augmentation: got {action.shape[-1]}, "
                f"expected {WBC_TASK_SPACE_ACTION_DIM} for WBC task-space actions or "
                f"{self.joint_index_map.shape[0]} for joint-space actions."
            )

        return action[..., self.joint_index_map] * self.sign_flip_mask

    def _mirror_wbc_task_space_action(self, action: torch.Tensor) -> torch.Tensor:
        """Mirror WBC actions: [pelvis_lin_vel, pelvis_ang_vel, com_vel, landing_foot_delta_xyyaw]."""
        if action.shape[-1] != WBC_TASK_SPACE_ACTION_DIM:
            raise ValueError(
                f"Unsupported WBC task-space action dimension: got {action.shape[-1]}, "
                f"expected {WBC_TASK_SPACE_ACTION_DIM}."
            )

        mirrored = action.clone()
        mirrored[..., WBC_PELVIS_LIN_VEL_SLICE] = self._mirror_wbc_position(action[..., WBC_PELVIS_LIN_VEL_SLICE])
        mirrored[..., WBC_PELVIS_ANG_VEL_SLICE] = self._mirror_wbc_axis_angle(action[..., WBC_PELVIS_ANG_VEL_SLICE])
        mirrored[..., WBC_COM_VEL_SLICE] = self._mirror_wbc_position(action[..., WBC_COM_VEL_SLICE])
        mirrored[..., WBC_LANDING_DELTA_SLICE] = self._mirror_wbc_landing_delta(action[..., WBC_LANDING_DELTA_SLICE])
        return mirrored

    def mirror_obs_base_lin_vel(self, base_lin_vel: torch.Tensor) -> torch.Tensor:
        """Mirrors the base linear velocity in robot's base frame.

        Parameters
        ----------
        base_lin_vel : torch.Tensor
            Base linear velocity with layout [v_x, v_y, v_z] in base frame coordinates.

        Returns
        -------
        torch.Tensor
            Mirrored velocity with y-component negated: [v_x, -v_y, v_z].
        """
        base_lin_vel[..., 1] = -base_lin_vel[..., 1]  # Flip y component
        return base_lin_vel

    def mirror_obs_base_ang_vel(self, base_ang_vel: torch.Tensor) -> torch.Tensor:
        """Mirrors the base angular velocity in robot's base frame.

        Parameters
        ----------
        base_ang_vel : torch.Tensor
            Base angular velocity with layout [ω_x, ω_y, ω_z] in base frame coordinates.

        Returns
        -------
        torch.Tensor
            Mirrored angular velocity with x and z components negated: [-ω_x, ω_y, -ω_z].
        """
        base_ang_vel[..., 0] = -base_ang_vel[..., 0]  # Flip x component
        base_ang_vel[..., 2] = -base_ang_vel[..., 2]  # Flip z component
        return base_ang_vel

    def mirror_obs_base_orientation(self, base_orientation: torch.Tensor) -> torch.Tensor:
        """Mirrors the base orientation representation.

        Parameters
        ----------
        base_orientation : torch.Tensor
            Base orientation with layout [θ_x, θ_y, θ_z] (Euler angles or similar representation).

        Returns
        -------
        torch.Tensor
            Mirrored orientation with y-component negated: [θ_x, -θ_y, θ_z].
        """
        base_orientation[..., 1] = -base_orientation[..., 1]  # Flip y component
        return base_orientation

    def mirror_obs_projected_gravity(self, projected_gravity: torch.Tensor) -> torch.Tensor:
        """Mirrors the projected gravity vector in robot's base frame.

        Parameters
        ----------
        projected_gravity : torch.Tensor
            Gravity vector projected into base frame with layout [g_x, g_y, g_z].

        Returns
        -------
        torch.Tensor
            Mirrored gravity vector with y-component negated: [g_x, -g_y, g_z].
        """
        projected_gravity[..., 1] = -projected_gravity[..., 1]  # Flip y component
        return projected_gravity

    def mirror_obs_command_lin_vel(self, command_lin_vel: torch.Tensor) -> torch.Tensor:
        """Mirrors the commanded linear velocity.

        Parameters
        ----------
        command_lin_vel : torch.Tensor
            Commanded linear velocity with layout [v_x_cmd, v_y_cmd, v_z_cmd].

        Returns
        -------
        torch.Tensor
            Mirrored velocity command with y-component negated: [v_x_cmd, -v_y_cmd, v_z_cmd].
        """
        command_lin_vel[..., 1] = -command_lin_vel[..., 1]  # Flip y component
        return command_lin_vel

    def mirror_obs_command_ang_vel(self, command_ang_vel: torch.Tensor) -> torch.Tensor:
        """Mirrors the commanded angular velocity (yaw only).

        Parameters
        ----------
        command_ang_vel : torch.Tensor
            Commanded yaw angular velocity (1D scalar) with layout [ω_yaw_cmd].

        Returns
        -------
        torch.Tensor
            Mirrored yaw angular velocity command with sign negated: [-ω_yaw_cmd].
        """
        command_ang_vel[..., 0] = -command_ang_vel[..., 0]
        return command_ang_vel

    def mirror_obs_command_stand(self, command_stand: torch.Tensor) -> torch.Tensor:
        """Mirrors the stand command (no transformation needed).

        Parameters
        ----------
        command_stand : torch.Tensor
            Stand command signal (scalar or binary).
        Returns
        -------
        torch.Tensor
            Unchanged stand command.
        """
        return command_stand

    def mirror_obs_command_waist_dofs(self, command_waist_dofs: torch.Tensor) -> torch.Tensor:
        """Mirrors the commanded waist joint positions.

        Parameters
        ----------
        command_waist_dofs : torch.Tensor
            Waist joint commands with layout [yaw_cmd, roll_cmd, pitch_cmd].

        Returns
        -------
        torch.Tensor
            Mirrored waist commands with yaw and roll negated: [-yaw_cmd, -roll_cmd, pitch_cmd].
        """
        # flip yaw
        command_waist_dofs[..., 0] = -command_waist_dofs[..., 0]
        # flip roll
        command_waist_dofs[..., 1] = -command_waist_dofs[..., 1]
        return command_waist_dofs

    def mirror_obs_command_base_height(self, command_base_height: torch.Tensor) -> torch.Tensor:
        """Mirrors the commanded base height (no transformation needed).

        Parameters
        ----------
        command_base_height : torch.Tensor
            Commanded base height (scalar).
            Inputs: [height_cmd].

        Returns
        -------
        torch.Tensor
            Unchanged height command.
            Outputs: [height_cmd].
        """
        return command_base_height

    def mirror_obs_sin_phase(self, sin_phase: torch.Tensor) -> torch.Tensor:
        """Mirrors the sine phase for gait timing.

        Parameters
        ----------
        sin_phase : torch.Tensor
            Sine of gait phase with layout [sin(φ_left), sin(φ_right), ...].

        Returns
        -------
        torch.Tensor
            Mirrored phase with left/right components swapped.
        """
        if sin_phase.shape[-1] < 2:
            return sin_phase.clone()
        mirrored = sin_phase.clone()
        mirrored[..., 0] = sin_phase[..., 1]
        mirrored[..., 1] = sin_phase[..., 0]
        return mirrored

    def mirror_obs_cos_phase(self, cos_phase: torch.Tensor) -> torch.Tensor:
        """Mirrors the cosine phase for gait timing.

        Parameters
        ----------
        cos_phase : torch.Tensor
            Cosine of gait phase with layout [cos(φ_left), cos(φ_right), ...].

        Returns
        -------
        torch.Tensor
            Mirrored phase with left/right components swapped.
        """
        if cos_phase.shape[-1] < 2:
            return cos_phase.clone()
        mirrored = cos_phase.clone()
        mirrored[..., 0] = cos_phase[..., 1]
        mirrored[..., 1] = cos_phase[..., 0]
        return mirrored

    def mirror_obs_dof_pos(self, dof_pos: torch.Tensor) -> torch.Tensor:
        """Mirrors the joint positions using joint mapping and sign flipping.

        Parameters
        ----------
        dof_pos : torch.Tensor
            Joint positions with layout following the robot's DOF order.
            Typically [hip_joints..., knee_joints..., ankle_joints..., waist_joints..., arm_joints...].
            Inputs: [q0, q1, q2, ..., qN] (original joint order).

        Returns
        -------
        torch.Tensor
            Mirrored joint positions with left-right mapping applied and signs flipped for appropriate joints.
            Outputs: [q_mapped0 * sign0, q_mapped1 * sign1, ..., q_mappedN * signN] (mirrored and sign-flipped).
        """
        return dof_pos[..., self.joint_index_map] * self.sign_flip_mask

    def mirror_obs_dof_vel(self, dof_vel: torch.Tensor) -> torch.Tensor:
        """Mirrors the joint velocities (same mapping as joint positions).

        Parameters
        ----------
        dof_vel : torch.Tensor
            Joint velocities with same layout as dof_pos.
            Inputs: [qd0, qd1, qd2, ..., qdN] (original joint velocities).

        Returns
        -------
        torch.Tensor
            Mirrored joint velocities with same transformation as joint positions.
            Outputs: [qd_mapped0 * sign0, qd_mapped1 * sign1, ..., qd_mappedN * signN] (mirrored and sign-flipped).
        """
        return dof_vel[..., self.joint_index_map] * self.sign_flip_mask

    def mirror_obs_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Mirrors the previous actions (same mapping as joint positions).

        Parameters
        ----------
        actions : torch.Tensor
            Previous action values with same layout as dof_pos.
            Inputs: [a0, a1, a2, ..., aN] (original actions).

        Returns
        -------
        torch.Tensor
            Mirrored actions with same transformation as joint positions.
            Outputs: [a_mapped0 * sign0, a_mapped1 * sign1, ..., a_mappedN * signN] (mirrored and sign-flipped).
        """
        if actions.shape[-1] == WBC_TASK_SPACE_ACTION_DIM:
            return self._mirror_wbc_task_space_action(actions)

        if actions.shape[-1] != self.joint_index_map.shape[0]:
            raise ValueError(
                f"Unsupported action observation dimension for symmetry augmentation: got {actions.shape[-1]}, "
                f"expected {WBC_TASK_SPACE_ACTION_DIM} for WBC task-space actions or "
                f"{self.joint_index_map.shape[0]} for joint-space actions."
            )

        return actions[..., self.joint_index_map] * self.sign_flip_mask

    def mirror_obs_com_pos(self, com_pos: torch.Tensor) -> torch.Tensor:
        return self._mirror_wbc_position(com_pos)

    def mirror_obs_com_lin_vel(self, com_lin_vel: torch.Tensor) -> torch.Tensor:
        return self._mirror_wbc_position(com_lin_vel)

    def mirror_obs_com_vel(self, com_vel: torch.Tensor) -> torch.Tensor:
        if com_vel.shape[-1] == 2:
            return self._mirror_wbc_planar_velocity(com_vel)
        return self._mirror_wbc_position(com_vel)

    def mirror_obs_pelvis_ori(self, pelvis_ori: torch.Tensor) -> torch.Tensor:
        return self._mirror_wbc_axis_angle(pelvis_ori)

    def mirror_obs_pelvis_ang_vel(self, pelvis_ang_vel: torch.Tensor) -> torch.Tensor:
        return self._mirror_wbc_axis_angle(pelvis_ang_vel)

    def mirror_obs_torso_ori(self, torso_ori: torch.Tensor) -> torch.Tensor:
        return self._mirror_wbc_axis_angle(torso_ori)

    def mirror_obs_torso_ang_vel(self, torso_ang_vel: torch.Tensor) -> torch.Tensor:
        return self._mirror_wbc_axis_angle(torso_ang_vel)

    def mirror_obs_torso_yaw_vel(self, torso_yaw_vel: torch.Tensor) -> torch.Tensor:
        out = torso_yaw_vel.clone()
        out[..., 0] = -out[..., 0]
        return out

    def mirror_obs_landing_foot_delta_pose(self, landing_foot_delta_pose: torch.Tensor) -> torch.Tensor:
        return self._mirror_wbc_landing_delta(landing_foot_delta_pose)

    def mirror_obs_landing_foot_delta(self, landing_foot_delta: torch.Tensor) -> torch.Tensor:
        return self._mirror_wbc_landing_delta(landing_foot_delta)

    def mirror_obs_foot_pos(self, foot_pos: torch.Tensor) -> torch.Tensor:
        mirrored = foot_pos.clone()
        mirrored[..., 0:3] = self._mirror_wbc_position(foot_pos[..., 3:6])
        mirrored[..., 3:6] = self._mirror_wbc_position(foot_pos[..., 0:3])
        return mirrored

    def mirror_obs_foot_ori(self, foot_ori: torch.Tensor) -> torch.Tensor:
        mirrored = foot_ori.clone()
        mirrored[..., 0:3] = self._mirror_wbc_axis_angle(foot_ori[..., 3:6])
        mirrored[..., 3:6] = self._mirror_wbc_axis_angle(foot_ori[..., 0:3])
        return mirrored

    def mirror_obs_foot_vel(self, foot_vel: torch.Tensor) -> torch.Tensor:
        if foot_vel.shape[-1] != 12:
            raise ValueError(
                f"Expected combined right/left foot velocity dimension 12, got {foot_vel.shape[-1]}."
            )

        mirrored = foot_vel.clone()
        mirrored[..., 0:6] = self._mirror_wbc_spatial_velocity(foot_vel[..., 6:12])
        mirrored[..., 6:12] = self._mirror_wbc_spatial_velocity(foot_vel[..., 0:6])
        return mirrored

    def mirror_obs_right_foot_pos(self, right_foot_pos: torch.Tensor) -> torch.Tensor:
        return self._mirror_wbc_position(right_foot_pos)

    def mirror_obs_right_foot_lin_vel(self, right_foot_lin_vel: torch.Tensor) -> torch.Tensor:
        return self._mirror_wbc_position(right_foot_lin_vel)

    def mirror_obs_right_foot_ori(self, right_foot_ori: torch.Tensor) -> torch.Tensor:
        return self._mirror_wbc_axis_angle(right_foot_ori)

    def mirror_obs_right_foot_ang_vel(self, right_foot_ang_vel: torch.Tensor) -> torch.Tensor:
        return self._mirror_wbc_axis_angle(right_foot_ang_vel)

    def mirror_obs_right_foot_vel(self, right_foot_vel: torch.Tensor) -> torch.Tensor:
        return self._mirror_wbc_spatial_velocity(right_foot_vel)

    def mirror_obs_left_foot_pos(self, left_foot_pos: torch.Tensor) -> torch.Tensor:
        return self._mirror_wbc_position(left_foot_pos)

    def mirror_obs_left_foot_lin_vel(self, left_foot_lin_vel: torch.Tensor) -> torch.Tensor:
        return self._mirror_wbc_position(left_foot_lin_vel)

    def mirror_obs_left_foot_ori(self, left_foot_ori: torch.Tensor) -> torch.Tensor:
        return self._mirror_wbc_axis_angle(left_foot_ori)

    def mirror_obs_left_foot_ang_vel(self, left_foot_ang_vel: torch.Tensor) -> torch.Tensor:
        return self._mirror_wbc_axis_angle(left_foot_ang_vel)

    def mirror_obs_left_foot_vel(self, left_foot_vel: torch.Tensor) -> torch.Tensor:
        return self._mirror_wbc_spatial_velocity(left_foot_vel)

    def _mirror_wbc_planar_velocity(self, vec: torch.Tensor) -> torch.Tensor:
        if vec.shape[-1] != 2:
            raise ValueError(f"Expected planar velocity dimension 2, got {vec.shape[-1]}.")

        out = vec.clone()
        out[..., 1] = -out[..., 1]
        return out

    def _mirror_wbc_landing_delta(self, vec: torch.Tensor) -> torch.Tensor:
        if vec.shape[-1] != 3:
            raise ValueError(f"Expected landing-foot delta dimension 3, got {vec.shape[-1]}.")

        out = vec.clone()
        out[..., 1] = -out[..., 1]
        out[..., 2] = -out[..., 2]
        return out

    def _mirror_wbc_spatial_velocity(self, vec: torch.Tensor) -> torch.Tensor:
        if vec.shape[-1] != 6:
            raise ValueError(f"Expected spatial velocity dimension 6, got {vec.shape[-1]}.")

        out = vec.clone()
        out[..., 0:3] = self._mirror_wbc_position(vec[..., 0:3])
        out[..., 3:6] = self._mirror_wbc_axis_angle(vec[..., 3:6])
        return out

    def _mirror_wbc_position(self, vec: torch.Tensor) -> torch.Tensor:
        out = vec.clone()
        out[..., 1] = -out[..., 1]
        return out

    def _mirror_wbc_axis_angle(self, vec: torch.Tensor) -> torch.Tensor:
        out = vec.clone()
        out[..., 0] = -out[..., 0]
        out[..., 2] = -out[..., 2]
        return out

    def mirror_obs_ee_apply_force(self, ee_apply_force: torch.Tensor) -> torch.Tensor:
        """Mirrors the end-effector applied forces in base frame.

        Parameters
        ----------
        ee_apply_force : torch.Tensor
            Applied forces with layout [left_fx, left_fy, left_fz, right_fx, right_fy, right_fz].
            Forces are already transformed to base frame coordinates.

        Returns
        -------
        torch.Tensor
            Mirrored forces with left-right swapped and y-components negated:
            [right_fx, -right_fy, right_fz, left_fx, -left_fy, left_fz].
        """
        # Note: this force is already transformed to the base frame
        left_ee_apply_force = ee_apply_force[..., :3].clone()
        left_ee_apply_force[..., 1] = -left_ee_apply_force[..., 1]
        right_ee_apply_force = ee_apply_force[..., 3:].clone()
        right_ee_apply_force[..., 1] = -right_ee_apply_force[..., 1]
        return torch.cat([right_ee_apply_force, left_ee_apply_force], dim=-1)
