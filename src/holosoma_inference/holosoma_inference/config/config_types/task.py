"""Task configuration types for holosoma_inference."""

from __future__ import annotations

from typing import Literal

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    """Task execution configuration for policy inference."""

    model_path: str | list[str]
    """Path to ONNX model(s). Supports local paths and wandb:// URIs. Required field."""

    rl_rate: float = 50
    """Policy inference rate in Hz."""

    policy_action_scale: float | tuple[float, ...] = 0.25
    """Scaling factor(s) applied to policy actions."""

    policy_action_space: Literal["joint_position", "task_space"] = "joint_position"
    """How to interpret policy actions during command post-processing."""

    wbc_state_port: int = 5556
    """ZMQ port used to receive simulator state needed for task-space WBC."""

    wbc_extension_dir: str | None = None
    """Directory containing the humanoid_wbc extension module."""

    wbc_robot_file: str | None = None
    """Robot URDF path passed to the WBC controller."""

    wbc_yaml_file: str | None = None
    """WBC parameter YAML path."""

    wbc_robot_name: str = "g1"
    """Robot name passed to the WBC controller."""

    wbc_contact_force_threshold: float = 20.0
    """Minimum local foot force norm treated as contact by the WBC."""

    use_phase: bool = True
    """Whether to use gait phase observations."""

    gait_period: float = 1.0
    """Gait cycle period in seconds."""

    domain_id: int = 0
    """DDS domain ID for communication."""

    interface: str = "lo"
    """Network interface name."""

    use_joystick: bool = False
    """Enable joystick control input."""

    joystick_type: str = "xbox"
    """Joystick type."""

    joystick_device: int = 0
    """Joystick device index."""

    use_sim_time: bool = False
    """Use synchronized simulation time for WBT policies."""

    wandb_download_dir: str = "/tmp"
    """Directory for downloading W&B checkpoints."""

    # Deprecation candidates:
    desired_base_height: float = 0.75
    """Target base height in meters."""

    residual_upper_body_action: bool = False
    """Whether to use residual control for upper body."""

    use_ros: bool = False
    """Use ROS2 for rate limiting."""

    print_observations: bool = False
    """Print observation vectors for debugging."""

    motion_start_timestep: int = 0
    """Starting timestep for motion clip playback."""

    motion_end_timestep: int | None = None
    """Ending timestep for motion clip playback. If None, plays until the end."""
