"""Locomotion action presets for the G1 robot."""

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg

G1_WBC_TASK_SPACE_ACTION_DIM = 12
G1_WBC_TASK_SPACE_ACTION_SCALE = [
    0.1, 0.1, 0.05,  # pelvis linear velocity residual: vx, vy, vz
    0.001, 0.001, 0.001,  # pelvis angular velocity residual: wx, wy, wz
    0.1, 0.1, 0.05,  # COM velocity residual: vx, vy, vz
    0.1, 0.1, 0.05,  # landing foot residual: dx, dy, dyaw
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
                "visualize_contact_points": False,
                "visualize_contact_frames": False,
                "visualize_action_targets": False,
                "visualize_action_target_frames": False,
                "visualize_landing_foot_pose": False,
                "visualize_swing_foot_trajectory": False,
                "swing_trajectory_samples": 12,
                "landing_ground_plane_z": 0.0,
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
