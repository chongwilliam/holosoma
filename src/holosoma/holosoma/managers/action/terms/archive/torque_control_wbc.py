"""WBC extension import and construction helpers for torque control."""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
from pathlib import Path
from typing import Any

from holosoma.managers.action.terms.torque_control_support import BatchedSwingFootPlanner


class TorqueControlWbcMixin:
    def _repo_root(self) -> Path:
        path = Path(__file__).resolve()
        candidates = [path.parent, *path.parents]

        for candidate in candidates:
            if (candidate.parent / "humanoid-control").exists():
                return candidate

        for candidate in candidates:
            if (candidate / ".git").exists():
                return candidate

        pyproject_candidates = [candidate for candidate in candidates if (candidate / "pyproject.toml").exists()]
        for candidate in reversed(pyproject_candidates):
            if (candidate / "src" / "holosoma").exists():
                return candidate

        if pyproject_candidates:
            return pyproject_candidates[-1]

        return path.parents[6]

    def _humanoid_control_root(self) -> Path:
        return self._repo_root().parent / "humanoid-control"

    def _resolve_wbc_extension_dir(self, extension_dir: str | None = None) -> str | None:
        if extension_dir:
            return str(Path(extension_dir).expanduser().resolve())

        humanoid_control_build = self._humanoid_control_root() / "build"
        if humanoid_control_build.exists():
            return str(humanoid_control_build.resolve())

        return None

    def _resolve_wbc_params(
        self, env: Any, params: dict[str, Any], extension_dir: str | None = None
    ) -> dict[str, Any]:
        resolved_params = dict(params)
        robot_type = getattr(env.robot_config.asset, "robot_type", "")

        if not robot_type.startswith("g1"):
            return resolved_params

        humanoid_control_root = self._humanoid_control_root()

        resolved_params.setdefault("robot_file", str((humanoid_control_root / "models" / "unitree_g1" / "g1.urdf").resolve()))
        resolved_params.setdefault("yaml_file", str((humanoid_control_root / "params" / "g1_parameters.yaml").resolve()))
        resolved_params.setdefault("robot_name", "g1")
        return resolved_params

    def _extension_root(self, extension_dir: str | None = None) -> Path:
        if extension_dir:
            resolved = Path(extension_dir).expanduser().resolve()
            if resolved.name == "build":
                return resolved.parent
            return resolved
        humanoid_control_root = self._humanoid_control_root()
        if humanoid_control_root.exists():
            return humanoid_control_root
        return self._repo_root()

    def _try_add_local_extension_path(self, module_name: str, extension_dir: str | None = None) -> None:
        for candidate in self._find_wbc_extension_candidates(module_name, extension_dir):
            module_dir = str(candidate.parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            return

    def _find_wbc_extension_candidates(self, module_name: str, extension_dir: str | None = None) -> list[Path]:
        search_roots = []
        if extension_dir:
            search_roots.append(Path(extension_dir).expanduser().resolve())
        humanoid_control_root = self._humanoid_control_root()
        search_roots.extend(
            [
                humanoid_control_root / "build",
                humanoid_control_root,
                self._repo_root() / "build",
                self._repo_root(),
            ]
        )

        candidates: list[Path] = []
        for base in search_roots:
            if not base.exists():
                continue
            for pattern in (f"{module_name}*.so", f"{module_name}*.pyd", f"{module_name}*.dylib"):
                candidates.extend(base.rglob(pattern))
        return candidates


    def _import_wbc_module(self, extension_dir: str | None = None):
        module_name = "humanoid_wbc"
        try:
            return importlib.import_module(module_name)
        except Exception:
            self._try_add_local_extension_path(module_name, extension_dir)
            try:
                return importlib.import_module(module_name)
            except Exception as fallback_error:
                candidates = self._find_wbc_extension_candidates(module_name, extension_dir)
                supported_suffixes = importlib.machinery.EXTENSION_SUFFIXES
                candidate_text = ", ".join(str(candidate) for candidate in candidates) or "none"
                raise ModuleNotFoundError(
                    "Could not import humanoid_wbc. "
                    f"Searched extension_dir={extension_dir!r}; found candidates: {candidate_text}. "
                    f"This Python accepts extension suffixes: {supported_suffixes}. "
                    "If a candidate is tagged for another Python version, rebuild humanoid-control "
                    "inside the active Python environment."
                ) from fallback_error

    def _resolve_wbc_asset_paths(
        self, extension_dir: str | None, params: dict[str, Any]
    ) -> tuple[Path, Path, str]:
        extension_root = self._extension_root(extension_dir)
        robot_file = Path(params.get("robot_file", extension_root / "models" / "unitree_g1" / "g1.urdf"))
        yaml_file = Path(params.get("yaml_file", extension_root / "params" / "hrp4c_parameters.yaml"))
        robot_name = params.get("robot_name", "hrp4c")

        if not robot_file.exists() or not yaml_file.exists():
            raise FileNotFoundError(
                "Whole-body controller assets are missing. "
                f"robot_file={robot_file}, yaml_file={yaml_file}"
            )

        return robot_file, yaml_file, robot_name

    def _create_batched_wbc_controller(
        self,
        extension_dir: str | None,
        params: dict[str, Any],
        num_envs: int,
        wbc_module: Any | None = None,
    ):
        if wbc_module is None:
            wbc_module = self._import_wbc_module(extension_dir)
        if not hasattr(wbc_module, "BatchedWbcController"):
            raise RuntimeError(
                "humanoid_wbc.BatchedWbcController is required for batched torque control, "
                "but the imported humanoid_wbc module does not expose it. Rebuild humanoid-control "
                "so the Python extension includes the batched controller bindings."
            )

        robot_file, yaml_file, robot_name = self._resolve_wbc_asset_paths(extension_dir, params)
        return wbc_module.BatchedWbcController(str(robot_file), str(yaml_file), robot_name, int(num_envs))

    def _create_batched_foot_planner(self, wbc_module: Any, num_envs: int) -> BatchedSwingFootPlanner:
        return BatchedSwingFootPlanner(
            wbc_module,
            int(num_envs),
            dt=float(self.env.dt),
            takeoff_clearance=self._swing_foot_takeoff_clearance,
            landing_clearance=self._swing_foot_landing_clearance,
        )

    def _create_wbc_engine(self, extension_dir: str | None, params: dict[str, Any], wbc_module: Any | None = None):
        if wbc_module is None:
            wbc_module = self._import_wbc_module(extension_dir)
        robot_file, yaml_file, robot_name = self._resolve_wbc_asset_paths(extension_dir, params)
        return wbc_module.WbcEngine(str(robot_file), str(yaml_file), robot_name)
