"""Locomotion action presets for the G1 robot."""

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg

G1_WBC_TASK_SPACE_ACTION_DIM = 9
G1_WBC_TASK_SPACE_ACTION_SCALE = [
    0.25, 0.20, 0.05,  # COM velocity residual: vx, vy, vz
    0.15, 0.15, 0.25,  # pelvis angular velocity residual: wx, wy, wz
    0.06, 0.04, 0.10,  # landing foot residual: dx, dy, dyaw
]
G1_WBC_TASK_SPACE_ACTION_CLIP = [1.0] * G1_WBC_TASK_SPACE_ACTION_DIM

g1_29dof_joint_pos = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
            params={},
            scale=1.0,
            clip=None,
        ),
    }
)

g1_29dof_torque = ActionManagerCfg(
    terms={
        "torque_control": ActionTermCfg(
            func="holosoma.managers.action.terms.torque_control:JointTorqueActionTerm",
            params={
                "dual_stance_bootstrap_enabled": False,
                "dual_stance_contact_force_threshold": 20.0,
                "dual_stance_contact_required_steps": 10,
                "startup_gait_timeout_s": 0.75,
                "wbc_transition_time": 0.15,
                "filter_stance_support_contacts": False,
                "use_unfiltered_stance_support_contacts": False,
                "visualize_contact_points": True,
                "visualize_contact_frames": True,
                "visualize_action_targets": False,
                "visualize_action_target_frames": False,
                "visualize_landing_foot_pose": True,
                "visualize_swing_foot_trajectory": True,
                "swing_trajectory_samples": 12,
                "swing_foot_midpoint_height": 0.2,
                "landing_ground_plane_z": 0.0,
                "use_command_as_pelvis_velocity_action": False,
                "use_command_as_landing_velocity": False,
                "assert_contact_visualization_pose": False,
                "contact_point_radius": 0.018,
                "action_target_radius": 0.026,
                "action_target_axis_scale": 0.16,
            },
            scale=1.0,
            clip=None,
        ),
    }
)

__all__ = [
    "G1_WBC_TASK_SPACE_ACTION_DIM",
    "G1_WBC_TASK_SPACE_ACTION_SCALE",
    "G1_WBC_TASK_SPACE_ACTION_CLIP",
    "g1_29dof_joint_pos",
    "g1_29dof_torque",
]
