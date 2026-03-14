from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _extension_root(extension_dir: str | None = None) -> Path:
    if extension_dir:
        resolved = Path(extension_dir).expanduser().resolve()
        if resolved.name == "build":
            return resolved.parent
        return resolved
    return _repo_root()


def _try_add_local_extension_path(module_name: str, extension_dir: str | None = None) -> None:
    search_roots = []
    if extension_dir:
        search_roots.append(Path(extension_dir).expanduser().resolve())
    search_roots.extend([_repo_root() / "build", _repo_root()])

    for base in search_roots:
        if not base.exists():
            continue
        for pattern in (f"{module_name}*.so", f"{module_name}*.pyd", f"{module_name}*.dylib"):
            for candidate in base.rglob(pattern):
                module_dir = str(candidate.parent)
                if module_dir not in sys.path:
                    sys.path.insert(0, module_dir)
                return


def _import_wbc_module(extension_dir: str | None = None):
    module_name = "humanoid_wbc"
    try:
        return importlib.import_module(module_name)
    except Exception:
        _try_add_local_extension_path(module_name, extension_dir)
        return importlib.import_module(module_name)
    
def main(extension_dir: str | None = None) -> int:
    extension_root = _extension_root(extension_dir)
    robot_file = extension_root / "models" / "hrp4c" / "HRP4c.urdf"
    yaml_file = extension_root / "params" / "hrp4c_parameters.yaml"

    if not robot_file.exists() or not yaml_file.exists():
        print("Missing model/config files:")
        if extension_dir:
            print(f"  extension_dir: {Path(extension_dir).expanduser().resolve()}")
        print(f"  robot: {robot_file}")
        print(f"  yaml:  {yaml_file}")
        return 1

    try:
        wbc = _import_wbc_module(extension_dir)
    except Exception as exc:
        print(f"FAIL: could not import humanoid_wbc: {exc}")
        if extension_dir:
            print(f"Hint: provided extension_dir was: {extension_dir}")
        return 1

    engine = wbc.WbcEngine(str(robot_file), str(yaml_file), "hrp4c")
    dof = engine.dof()
    q = np.zeros(dof)
    dq = np.zeros(dof)

    desired_state_dual_stance = 2
    reinit_flag = 1
    com_position = np.array([0.0, 0.0, 0.7], dtype=float)
    pelvis_orientation = np.eye(3)
    torso_orientation = np.eye(3)
    right_foot_position = np.zeros(3)
    right_foot_orientation = np.eye(3)
    left_foot_position = np.zeros(3)
    left_foot_orientation = np.eye(3)
    right_foot_contact_point = np.array([0.02, -0.007, -0.095])
    right_foot_angular_basis = np.eye(3)
    left_foot_contact_point = np.array([0.02, -0.007, -0.095])
    left_foot_angular_basis = np.eye(3)

    t0 = time.perf_counter()
    torques = engine.compute(
        q,
        dq,
        desired_state_dual_stance,
        reinit_flag,
        com_position,
        pelvis_orientation,
        torso_orientation,
        right_foot_position,
        right_foot_orientation,
        left_foot_position,
        left_foot_orientation,
        right_foot_contact_point,
        right_foot_angular_basis,
        left_foot_contact_point,
        left_foot_angular_basis,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    torques = np.asarray(torques, dtype=float).reshape(-1)
    print(f"dof: {dof}")
    print(f"torque size: {torques.size}")
    print(f"torque norm: {np.linalg.norm(torques):.6f}")
    print(f"compute time: {elapsed_ms:.3f} ms")
    print("PASS: imported humanoid_wbc and computed torques")
    return 0


if __name__ == "__main__":
    # extension_dir = sys.argv[1] if len(sys.argv) > 1 else None
    extension_dir = '~/humanoid-control/build/'
    raise SystemExit(main(extension_dir))
