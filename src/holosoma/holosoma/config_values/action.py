"""Default action manager configurations."""

from holosoma.config_values.loco.g1.action import (
    G1_WBC_TASK_SPACE_ACTION_CLIP,
    G1_WBC_TASK_SPACE_ACTION_DIM,
    G1_WBC_TASK_SPACE_ACTION_SCALE,
    g1_29dof_joint_pos,
    g1_29dof_torque,
)
from holosoma.config_values.loco.t1.action import t1_29dof_joint_pos

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof_joint_pos": t1_29dof_joint_pos,
    "g1_29dof_joint_pos": g1_29dof_joint_pos,
    "g1_29dof_torque": g1_29dof_torque,
}
