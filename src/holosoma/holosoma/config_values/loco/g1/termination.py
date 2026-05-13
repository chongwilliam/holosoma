"""Locomotion termination presets for the G1 robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

g1_29dof_termination = TerminationManagerCfg(
    terms={
        "contact": TerminationTermCfg(
            func="holosoma.managers.termination.terms.locomotion:contact_forces_exceeded",
            params={
                "force_threshold": 1.0,
                "contact_indices_attr": "termination_contact_indices",
            },
        ),
        "both_feet_airborne": TerminationTermCfg(
            func="holosoma.managers.termination.terms.locomotion:BothFeetAirborne",
            params={
                "max_airborne_steps": 20,
                "action_term_name": "torque_control",
                "skip_wbc_bootstrap_hold": True,
            },
        ),
        "high_root_ang_vel": TerminationTermCfg(
            func="holosoma.managers.termination.terms.locomotion:root_angular_velocity_exceeded",
            params={
                "max_root_ang_vel": 20.0,
                "norm": "l2",
            },
        ),
        "non_finite_torques": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:non_finite_action_torques",
        ),
        "non_finite_state": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:non_finite_sim_state",
        ),
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
    }
)

__all__ = ["g1_29dof_termination"]
