"""Locomotion observation presets for the G1 robot."""

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

# Task observations are relative to the yaw-aligned pelvis: z is up, while x/y are aligned with the pelvis.
g1_29dof_loco_single_wolinvel = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms={
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
                    scale=0.25,
                    noise=0.0,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:projected_gravity",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_lin_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_ang_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_pos",
                    scale=1.0,
                    noise=0.01,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_vel",
                    scale=0.05,
                    noise=0.1,
                ),
                "actions": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:actions",
                    scale=1.0,
                    noise=0.0,
                ),
                # "com_pos": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:com_pos",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "com_lin_vel": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:com_lin_vel",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "pelvis_ori": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:pelvis_ori",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "pelvis_ang_vel": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:pelvis_ang_vel",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "torso_ori": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:torso_ori",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "torso_ang_vel": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:torso_ang_vel",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "right_foot_pos": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:right_foot_pos",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "right_foot_lin_vel": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:right_foot_lin_vel",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "right_foot_ori": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:right_foot_ori",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "right_foot_ang_vel": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:right_foot_ang_vel",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "left_foot_pos": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:left_foot_pos",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "left_foot_lin_vel": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:left_foot_lin_vel",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "left_foot_ori": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:left_foot_ori",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "left_foot_ang_vel": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:left_foot_ang_vel",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "contact_state": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:contact_state",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                "sin_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:sin_phase",
                    scale=1.0,
                    noise=0.0,
                ),
                "cos_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:cos_phase",
                    scale=1.0,
                    noise=0.0,
                ),
            },
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "base_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_lin_vel",
                    scale=2.0,
                    noise=0.0,
                ),
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
                    scale=0.25,
                    noise=0.0,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:projected_gravity",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_lin_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_ang_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_pos",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_vel",
                    scale=0.05,
                    noise=0.0,
                ),
                "actions": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:actions",
                    scale=1.0,
                    noise=0.0,
                ),
                # "com_pos": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:com_pos",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "com_lin_vel": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:com_lin_vel",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "pelvis_ori": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:pelvis_ori",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "pelvis_ang_vel": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:pelvis_ang_vel",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "torso_ori": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:torso_ori",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "torso_ang_vel": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:torso_ang_vel",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "right_foot_pos": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:right_foot_pos",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "right_foot_lin_vel": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:right_foot_lin_vel",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "right_foot_ori": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:right_foot_ori",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "right_foot_ang_vel": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:right_foot_ang_vel",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "left_foot_pos": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:left_foot_pos",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "left_foot_lin_vel": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:left_foot_lin_vel",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "left_foot_ori": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:left_foot_ori",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                # "left_foot_ang_vel": ObsTermCfg(
                #     func="holosoma.managers.observation.terms.locomotion:left_foot_ang_vel",
                #     scale=1.0,
                #     noise=0.0,
                # ),
                "sin_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:sin_phase",
                    scale=1.0,
                    noise=0.0,
                ),
                "cos_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:cos_phase",
                    scale=1.0,
                    noise=0.0,
                ),
            },
        ),
    }
)

__all__ = ["g1_29dof_loco_single_wolinvel"]
