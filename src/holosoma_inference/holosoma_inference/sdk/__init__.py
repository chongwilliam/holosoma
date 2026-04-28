"""Robot communication package."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Callable

# Auto-discover SDK interfaces from installed packages using lazy loading.
# Lazy loading is to avoid errors from SDK dependencies from extensions (e.g. ROS2) when working with other SDKs.
_entry_points = {ep.name: ep for ep in entry_points(group="holosoma.sdk")}
_registry = {}  # Cache for loaded interfaces


def _load_builtin_interface(sdk_type: str) -> Callable | None:
    """Fallback loader for source-tree runs where package entry points are unavailable."""
    if sdk_type == "unitree":
        from holosoma_inference.sdk.unitree.unitree_interface import UnitreeInterface

        return UnitreeInterface
    if sdk_type == "booster":
        from holosoma_inference.sdk.booster.booster_interface import BoosterInterface

        return BoosterInterface
    return None


def create_interface(robot_config, domain_id=0, interface_str=None, use_joystick=True):
    """Create interface from registry."""
    sdk_type = robot_config.sdk_type

    if sdk_type not in _registry:
        if sdk_type in _entry_points:
            # Lazy load: only load the entry point when actually needed
            _registry[sdk_type] = _entry_points[sdk_type].load()
        else:
            builtin_interface = _load_builtin_interface(sdk_type)
            if builtin_interface is None:
                available = sorted(set(_entry_points.keys()) | {"unitree", "booster"})
                raise ValueError(f"Unknown sdk_type: {sdk_type}. Available: {available}")
            _registry[sdk_type] = builtin_interface

    return _registry[sdk_type](robot_config, domain_id, interface_str, use_joystick)
