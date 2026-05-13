"""Debug and visualization helpers for WBC torque control."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

from holosoma.managers.action.terms.torque_control_support import apply_delta_rotation
from holosoma.utils.rotations import quat_apply, quaternion_to_matrix


class TorqueControlDebugMixin:
    def _assert_finite_torch(self, label: str, value: torch.Tensor) -> None:
        finite = torch.isfinite(value)
        if bool(finite.all()):
            return

        bad_idx = (~finite).nonzero(as_tuple=False)[0].detach().cpu().tolist()
        bad_value = value[tuple(bad_idx)].detach().cpu().item()
        finite_values = value[finite].detach()
        if finite_values.numel() == 0:
            summary = "no finite values"
        else:
            summary = (
                f"finite_min={finite_values.min().cpu().item():.6g}, "
                f"finite_max={finite_values.max().cpu().item():.6g}"
            )
        torque_summary = self._last_torque_debug_summary(bad_idx[0] if bad_idx else None)
        raise RuntimeError(
            f"{label} has non-finite value at index {bad_idx}: {bad_value}; "
            f"shape={tuple(value.shape)}, {summary}{torque_summary}"
        )

    def _assert_finite_numpy(self, label: str, value: np.ndarray) -> None:
        finite = np.isfinite(value)
        if bool(finite.all()):
            return

        bad_idx = np.argwhere(~finite)[0].tolist()
        bad_value = value[tuple(bad_idx)]
        finite_values = value[finite]
        if finite_values.size == 0:
            summary = "no finite values"
        else:
            summary = f"finite_min={finite_values.min():.6g}, finite_max={finite_values.max():.6g}"
        raise RuntimeError(
            f"{label} has non-finite value at index {bad_idx}: {bad_value}; "
            f"shape={value.shape}, {summary}"
        )

    def _assert_finite_torch_debug(self, label: str, value: torch.Tensor) -> None:
        if self._debug_wbc_finite_checks:
            self._assert_finite_torch(label, value)

    def _assert_finite_numpy_debug(self, label: str, value: np.ndarray) -> None:
        if self._debug_wbc_finite_checks:
            self._assert_finite_numpy(label, value)

    def _needs_wbc_debug_engine_update(self) -> bool:
        return (
            self._visualize_contact_points
            or self._visualize_action_targets
            or self._assert_contact_visualization_pose
            or self._store_wbc_debug_snapshots
        )

    def _maybe_store_wbc_debug_snapshots(
        self,
        action_batch: np.ndarray,
        right_contact_points: np.ndarray,
        left_contact_points: np.ndarray,
        right_contact_bases: np.ndarray,
        left_contact_bases: np.ndarray,
        right_grfs: np.ndarray,
        left_grfs: np.ndarray,
    ) -> None:
        if not self._store_wbc_debug_snapshots:
            return

        self._last_wbc_action_batch = action_batch.copy()
        self._last_wbc_right_contact_points = right_contact_points.copy()
        self._last_wbc_left_contact_points = left_contact_points.copy()
        self._last_wbc_right_contact_bases = right_contact_bases.copy()
        self._last_wbc_left_contact_bases = left_contact_bases.copy()
        self._last_wbc_right_grfs = right_grfs.copy()
        self._last_wbc_left_grfs = left_grfs.copy()

    def _last_torque_debug_summary(self, env_idx: int | None) -> str:
        if env_idx is None or not isinstance(getattr(self, "torques", None), torch.Tensor):
            return ""
        if self.torques.numel() == 0 or env_idx < 0 or env_idx >= self.torques.shape[0]:
            return ""

        env_torques = self.torques[env_idx].detach()
        finite = torch.isfinite(env_torques)
        if not bool(finite.all()):
            return "; previous_applied_torques=non-finite"

        limits = self.env.torque_limits.to(device=env_torques.device, dtype=env_torques.dtype)
        saturation_frac = (env_torques.abs() >= limits).float().mean().cpu().item()
        return (
            f"; previous_applied_torques_env{env_idx}: "
            f"min={env_torques.min().cpu().item():.6g}, "
            f"max={env_torques.max().cpu().item():.6g}, "
            f"max_abs={env_torques.abs().max().cpu().item():.6g}, "
            f"saturation_frac={saturation_frac:.3f}"
        )

    def _env_idx_from_wbc_exception(self, exc: Exception) -> int | None:
        match = re.search(r"\benv\s+(\d+)\b", str(exc))
        if match is None:
            return None
        env_idx = int(match.group(1))
        if env_idx < 0 or env_idx >= self.env.num_envs:
            return None
        return env_idx

    def _wbc_input_context_summary(
        self,
        env_idx: int,
        q: np.ndarray,
        dq: np.ndarray,
        action_batch: np.ndarray,
        right_contact_points: np.ndarray,
        left_contact_points: np.ndarray,
        right_contact_bases: np.ndarray,
        left_contact_bases: np.ndarray,
        right_grfs: np.ndarray,
        left_grfs: np.ndarray,
        right_contact_in_contact: np.ndarray,
        left_contact_in_contact: np.ndarray,
    ) -> str:
        return (
            f"WBC input context for env {env_idx}: "
            f"{self._batch_state_summary(env_idx)}, "
            f"{self._bootstrap_state_summary(env_idx)}, "
            f"right_in_contact={bool(right_contact_in_contact[env_idx])}, "
            f"left_in_contact={bool(left_contact_in_contact[env_idx])}, "
            f"{self._tensor_row_summary('q', q, env_idx)}, "
            f"{self._tensor_row_summary('dq', dq, env_idx)}, "
            f"{self._tensor_row_summary('action', action_batch, env_idx)}, "
            f"{self._tensor_row_summary('right_contact_point', right_contact_points, env_idx)}, "
            f"{self._tensor_row_summary('left_contact_point', left_contact_points, env_idx)}, "
            f"{self._tensor_row_summary('right_contact_basis', right_contact_bases, env_idx)}, "
            f"{self._tensor_row_summary('left_contact_basis', left_contact_bases, env_idx)}, "
            f"{self._tensor_row_summary('right_grf', right_grfs, env_idx)}, "
            f"{self._tensor_row_summary('left_grf', left_grfs, env_idx)}"
        )

    def debug_summary(self, env_idx: int) -> str:
        if env_idx < 0 or env_idx >= self.env.num_envs:
            return ""

        parts = [
            self._batch_state_summary(env_idx),
            self._bootstrap_state_summary(env_idx),
            self._tensor_row_summary("raw_action", self._raw_actions, env_idx),
            self._tensor_row_summary("processed_action", self._processed_actions, env_idx),
            self._tensor_row_summary("actions_after_delay", self._actions_after_delay, env_idx),
            self._tensor_row_summary("last_wbc_q", self._last_wbc_q, env_idx),
            self._tensor_row_summary("last_wbc_dq", self._last_wbc_dq, env_idx),
            self._tensor_row_summary("last_wbc_torque_output", self._last_wbc_torque_output, env_idx),
            self._tensor_row_summary("last_wbc_action_batch", self._last_wbc_action_batch, env_idx),
            self._tensor_row_summary("last_wbc_sin_phase", self._last_wbc_sin_phase, env_idx),
            self._tensor_row_summary("last_wbc_cos_phase", self._last_wbc_cos_phase, env_idx),
            self._tensor_row_summary("last_wbc_desired_state", self._last_wbc_desired_state, env_idx),
            self._tensor_row_summary(
                "last_wbc_remaining_swing_duration", self._last_wbc_remaining_swing_duration, env_idx
            ),
            self._tensor_row_summary("right_contact_point", self._last_wbc_right_contact_points, env_idx),
            self._tensor_row_summary("left_contact_point", self._last_wbc_left_contact_points, env_idx),
            self._tensor_row_summary("right_contact_basis", self._last_wbc_right_contact_bases, env_idx),
            self._tensor_row_summary("left_contact_basis", self._last_wbc_left_contact_bases, env_idx),
            self._tensor_row_summary("right_grf", self._last_wbc_right_grfs, env_idx),
            self._tensor_row_summary("left_grf", self._last_wbc_left_grfs, env_idx),
        ]
        return ", ".join(part for part in parts if part)

    def _batch_state_summary(self, env_idx: int) -> str:
        try:
            states = self.wbc.get_states()
        except Exception:
            return ""
        if env_idx >= len(states):
            return ""
        state_part = f"batch_state={states[env_idx]}"
        try:
            transition_start_times = self.wbc.get_transition_start_times()
        except Exception:
            return state_part
        if env_idx >= len(transition_start_times):
            return state_part
        return f"{state_part}, batch_transition_start={float(transition_start_times[env_idx]):.6f}"

    def _bootstrap_state_summary(self, env_idx: int) -> str:
        if env_idx < 0 or env_idx >= self.env.num_envs:
            return ""

        bootstrap_done = bool(self._wbc_bootstrap_done[env_idx].detach().cpu().item())
        bootstrap_hold = bool(self._last_wbc_bootstrap_hold_mask[env_idx].detach().cpu().item())
        dual_contact = bool(self._last_wbc_dual_contact_mask[env_idx].detach().cpu().item())
        dual_contact_count = int(self._dual_contact_counter[env_idx].detach().cpu().item())
        return (
            f"bootstrap_done={bootstrap_done}, "
            f"bootstrap_hold={bootstrap_hold}, "
            f"dual_contact={dual_contact}, "
            f"dual_contact_count={dual_contact_count}"
        )

    def _tensor_row_summary(self, label: str, value: torch.Tensor | np.ndarray | None, env_idx: int) -> str:
        if value is None:
            return ""

        if isinstance(value, torch.Tensor):
            if value.numel() == 0 or env_idx >= value.shape[0]:
                return ""
            row = value[env_idx].detach().flatten()
            finite = torch.isfinite(row)
            finite_note = "finite" if bool(finite.all()) else "non-finite"
            if row.numel() == 0:
                return f"{label}=empty"
            return (
                f"{label}({finite_note}): "
                f"min={row[finite].min().cpu().item():.6g}, "
                f"max={row[finite].max().cpu().item():.6g}, "
                f"max_abs={row[finite].abs().max().cpu().item():.6g}"
                if bool(finite.any())
                else f"{label}(non-finite): no finite values"
            )

        array = np.asarray(value)
        if array.size == 0 or env_idx >= array.shape[0]:
            return ""
        row_np = array[env_idx].reshape(-1)
        finite_np = np.isfinite(row_np)
        finite_note_np = "finite" if bool(finite_np.all()) else "non-finite"
        if not bool(finite_np.any()):
            return f"{label}(non-finite): no finite values"
        finite_values = row_np[finite_np]
        return (
            f"{label}({finite_note_np}): "
            f"min={finite_values.min():.6g}, "
            f"max={finite_values.max():.6g}, "
            f"max_abs={np.abs(finite_values).max():.6g}"
        )

    def _maybe_draw_action_targets(self, root_states: torch.Tensor, action_batch: np.ndarray) -> None:
        if not self._visualize_action_targets:
            return

        simulator = self.env.simulator
        if not hasattr(simulator, "draw_sphere"):
            return

        for env_idx in self._contact_visualization_env_indices():
            if env_idx < 0 or env_idx >= self.env.num_envs or env_idx >= action_batch.shape[0]:
                continue
            self._draw_action_targets_for_env(env_idx, root_states[env_idx], action_batch[env_idx])

    def _draw_action_targets_for_env(self, env_idx: int, root_state: torch.Tensor, action_row: np.ndarray) -> None:
        simulator = self.env.simulator
        com_pos = self._current_com_position(env_idx, root_state)
        action_row = np.asarray(action_row, dtype=float).reshape(-1)
        if action_row.shape[0] < 15:
            return

        com_target = com_pos + action_row[:3]
        self._draw_target_pose(env_idx, com_target, None, [1.0, 0.15, 1.0], 300)

        for side, pos_slice, ori_slice, color, pos_id_base in (
            ("right", slice(3, 6), slice(6, 9), [1.0, 0.35, 0.05], 330),
            ("left", slice(9, 12), slice(12, 15), [0.25, 0.45, 1.0], 340),
        ):
            pos_delta = action_row[pos_slice]
            ori_delta = action_row[ori_slice]
            if np.linalg.norm(pos_delta) < 1.0e-12 and np.linalg.norm(ori_delta) < 1.0e-12:
                continue
            foot_pose = self._body_pose_for_target(env_idx, self._foot_body_indices.get(side))
            if foot_pose is None:
                continue
            foot_pos, foot_rot = foot_pose
            foot_target_pos = foot_pos + pos_delta
            foot_target_rot = apply_delta_rotation(foot_rot, ori_delta)
            self._draw_target_pose(env_idx, foot_target_pos, foot_target_rot, color, pos_id_base)

    def _current_com_position(self, env_idx: int, root_state: torch.Tensor) -> np.ndarray:
        com_pos = getattr(self.env.simulator, "com_pos", None)
        if isinstance(com_pos, torch.Tensor) and com_pos.ndim == 2 and env_idx < com_pos.shape[0]:
            return com_pos[env_idx].detach().cpu().numpy()
        return root_state[:3].detach().cpu().numpy()

    def _body_pose_for_target(self, env_idx: int, body_idx: Any | None) -> tuple[np.ndarray, np.ndarray] | None:
        if body_idx is None:
            return None

        simulator = self.env.simulator
        rigid_body_idx = int(body_idx)
        mujoco_body_map = getattr(simulator, "holosoma_to_mujoco_body_map", None)
        if isinstance(mujoco_body_map, dict):
            rigid_body_idx = int(mujoco_body_map.get(rigid_body_idx, rigid_body_idx))

        if (
            not hasattr(simulator, "_rigid_body_pos")
            or not hasattr(simulator, "_rigid_body_rot")
            or env_idx >= simulator._rigid_body_pos.shape[0]
            or rigid_body_idx >= simulator._rigid_body_pos.shape[1]
        ):
            return None

        pos = simulator._rigid_body_pos[env_idx, rigid_body_idx].detach().cpu().numpy()
        quat = simulator._rigid_body_rot[env_idx, rigid_body_idx]
        rot = quaternion_to_matrix(quat.unsqueeze(0), w_last=True)[0].detach().cpu().numpy()
        return pos, rot

    def _draw_target_pose(
        self,
        env_idx: int,
        position: np.ndarray,
        rotation: np.ndarray | None,
        color: list[float],
        pos_id_base: int,
    ) -> None:
        if not np.all(np.isfinite(position)):
            return

        simulator = self.env.simulator
        point = torch.as_tensor(position[:3], dtype=torch.float32).cpu()
        simulator.draw_sphere(point, self._action_target_radius, color, env_id=env_idx, pos_id=pos_id_base)

        if rotation is None or not self._visualize_action_target_frames or not hasattr(simulator, "draw_line"):
            return
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            return

        axis_colors = ([1.0, 0.0, 0.0], [0.0, 0.85, 0.0], [0.1, 0.25, 1.0])
        for axis_idx, axis_color in enumerate(axis_colors):
            axis = rotation[:, axis_idx]
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm < 1.0e-9:
                continue
            end = torch.as_tensor(
                position[:3] + self._action_target_axis_scale * axis / axis_norm,
                dtype=torch.float32,
            ).cpu()
            simulator.draw_line(point, end, axis_color, env_id=env_idx)

    def _maybe_draw_contact_points(
        self,
        right_contact_points: np.ndarray,
        right_contact_bases: np.ndarray,
        left_contact_points: np.ndarray,
        left_contact_bases: np.ndarray,
    ) -> None:
        if not self._visualize_contact_points:
            return

        simulator = self.env.simulator
        if not hasattr(simulator, "draw_sphere"):
            return

        env_indices = self._contact_visualization_env_indices()
        for env_idx in env_indices:
            if env_idx < 0 or env_idx >= self.env.num_envs:
                continue
            self._draw_stance_support(
                env_idx,
                right_contact_points[env_idx],
                right_contact_bases[env_idx],
                "right",
                left_contact_points[env_idx],
                left_contact_bases[env_idx],
                "left",
            )

    def _contact_visualization_env_indices(self) -> list[int]:
        simulator = self.env.simulator
        current_world_id = getattr(simulator, "current_world_id", None)
        if current_world_id is not None:
            return [int(current_world_id)]
        return list(range(self.env.num_envs))

    def _draw_stance_support(
        self,
        env_idx: int,
        right_contact_point: np.ndarray,
        right_contact_basis: np.ndarray,
        right_side: str,
        left_contact_point: np.ndarray,
        left_contact_basis: np.ndarray,
        left_side: str,
    ) -> None:
        simulator = self.env.simulator
        if not hasattr(simulator, "draw_sphere"):
            return

        self._draw_contact_support_frame(
            env_idx,
            np.asarray(right_contact_point, dtype=float),
            np.asarray(right_contact_basis, dtype=float),
            right_side,
            point_color=[1.0, 0.25, 0.1],
            pos_id_base=100,
        )
        self._draw_contact_support_frame(
            env_idx,
            np.asarray(left_contact_point, dtype=float),
            np.asarray(left_contact_basis, dtype=float),
            left_side,
            point_color=[0.2, 0.45, 1.0],
            pos_id_base=200,
        )

    def _draw_contact_support_frame(
        self,
        env_idx: int,
        contact_point: np.ndarray,
        contact_basis: np.ndarray,
        side: str,
        point_color: list[float],
        pos_id_base: int,
    ) -> None:
        if contact_point.shape[0] < 3 or contact_basis.shape != (3, 3):
            return
        if not np.all(np.isfinite(contact_point)) or not np.all(np.isfinite(contact_basis)):
            return
        counts = getattr(self.env.simulator, f"{side}_foot_contact_count", None)
        if counts is not None:
            count_tensor = self._as_torch_tensor(
                counts,
                device=self.env.device,
                dtype=torch.long,
                label=f"{side}_foot_contact_count",
            ).reshape(-1)
            if env_idx >= count_tensor.shape[0] or int(count_tensor[env_idx].detach().cpu().item()) <= 0:
                return

        simulator = self.env.simulator
        foot_body_idx = self._foot_body_indices[side]
        mujoco_body_map = getattr(simulator, "holosoma_to_mujoco_body_map", None)
        rigid_body_idx = foot_body_idx
        if isinstance(mujoco_body_map, dict):
            rigid_body_idx = mujoco_body_map.get(foot_body_idx, foot_body_idx)

        foot_pos_w = simulator._rigid_body_pos[env_idx, rigid_body_idx]
        foot_quat_w = simulator._rigid_body_rot[env_idx, rigid_body_idx]
        if self._assert_contact_visualization_pose:
            self._assert_foot_pose_matches_wbc(env_idx, side, foot_pos_w, foot_quat_w, contact_point)

        local_contact_point = torch.as_tensor(contact_point[:3], device=foot_pos_w.device, dtype=foot_pos_w.dtype)
        world_contact_point_tensor = foot_pos_w + quat_apply(foot_quat_w, local_contact_point, w_last=True)
        world_contact_point = world_contact_point_tensor.detach().cpu().numpy()

        point = torch.as_tensor(world_contact_point, dtype=torch.float32).cpu()
        simulator.draw_sphere(point, self._contact_point_radius, point_color, env_id=env_idx, pos_id=pos_id_base)

        if not self._visualize_contact_frames or not hasattr(simulator, "draw_line"):
            return

        axis_colors = ([1.0, 0.0, 0.0], [0.0, 0.85, 0.0], [0.1, 0.25, 1.0])
        axis_scale = 0.12
        for axis_idx, axis_color in enumerate(axis_colors):
            local_axis = torch.as_tensor(
                contact_basis[:, axis_idx], device=foot_quat_w.device, dtype=foot_quat_w.dtype
            )
            axis = quat_apply(foot_quat_w, local_axis, w_last=True).detach().cpu().numpy()
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm < 1.0e-9:
                continue
            end = torch.as_tensor(world_contact_point + axis_scale * axis / axis_norm, dtype=torch.float32).cpu()
            simulator.draw_line(point, end, axis_color, env_id=env_idx)

    def _assert_foot_pose_matches_wbc(
        self,
        env_idx: int,
        side: str,
        foot_pos_w: torch.Tensor,
        foot_quat_w: torch.Tensor,
        local_contact_point: np.ndarray
    ) -> None:
        # foot_name = f"{side}_foot"
        foot_name = f"{side}_ankle_roll_link"
        wbc_engine = self._wbc_debug_engine
        wbc_pos = np.asarray(wbc_engine.getPosition(foot_name, local_contact_point), dtype=float).reshape(-1)[:3]
        wbc_rot = np.asarray(wbc_engine.getRotation(foot_name), dtype=float)
        if wbc_pos.shape[0] < 3 or not np.all(np.isfinite(wbc_pos)):
            raise AssertionError(f"WBC {foot_name} position is invalid: shape={wbc_pos.shape}, pos={wbc_pos}")
        if wbc_rot.shape != (3, 3) or not np.all(np.isfinite(wbc_rot)):
            raise AssertionError(f"WBC {foot_name} rotation is invalid: shape={wbc_rot.shape}, rotation={wbc_rot}")

        sim_rot = quaternion_to_matrix(foot_quat_w.unsqueeze(0), w_last=True)[0].detach().cpu().numpy()
        sim_pos = foot_pos_w.detach().cpu().numpy() + sim_rot @ local_contact_point
        self._print_foot_frame_inspection(env_idx, side, sim_pos, sim_rot, wbc_pos, wbc_rot)

        # root_state = self.env.simulator.robot_root_states[env_idx]
        # dof_pos = self.env.simulator.dof_pos[env_idx]
        # q_tensor = torch.cat([root_state_to_xyz_rpy(root_state), dof_pos], dim=0)
        # q_error = float(torch.linalg.vector_norm(q_tensor - self._last_wbc_q[env_idx]).item())
        # root_state_error = float(
        #     torch.linalg.vector_norm(root_state - self._last_wbc_root_state[env_idx]).item()
        # )
        # dof_pos_error = float(torch.linalg.vector_norm(dof_pos - self._last_wbc_dof_pos[env_idx]).item())
        # pelvis_pos_error, pelvis_rot_error, pelvis_pos, wbc_pelvis_pos, pelvis_pos_delta = (
        #     self._assert_pelvis_pose_matches_wbc(env_idx)
        # )
        pos_delta = sim_pos - wbc_pos
        pos_error = float(np.linalg.norm(sim_pos - wbc_pos))
        rot_error = float(np.linalg.norm(sim_rot - wbc_rot, ord="fro"))
        assert (
            # q_error < 1.0e-6
            # and root_state_error < 1.0e-6
            # and dof_pos_error < 1.0e-6
            # and pelvis_pos_error < 1.0e-6
            # and pelvis_rot_error < 1.0e-6
            pos_error < 1.0e-6
            and rot_error < 1.0e-6
        ), (
            f"Simulator {side} foot pose does not match WBC {side}_foot pose: "
            # f"q_error={q_error:.9f}, root_state_error={root_state_error:.9f}, "
            # f"dof_pos_error={dof_pos_error:.9f}, pelvis_pos_error={pelvis_pos_error:.9f}, "
            # f"pelvis_rot_error={pelvis_rot_error:.9f}, pos_error={pos_error:.9f}, rot_error={rot_error:.9f}, "
            # f"pelvis_sim_pos_xyz={pelvis_pos.tolist()}, pelvis_wbc_pos_xyz={wbc_pelvis_pos.tolist()}, "
            # f"pelvis_pos_delta_xyz={pelvis_pos_delta.tolist()}, "
            f"sim_pos_xyz={sim_pos.tolist()}, wbc_pos_xyz={wbc_pos.tolist()}, pos_delta_xyz={pos_delta.tolist()}, rot_delta={rot_error}"
            "\n"
        )

    def _assert_pelvis_pose_matches_wbc(
        self,
        env_idx: int,
    ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        wbc_pelvis_pos = np.asarray(self._wbc_debug_engine.getPosition("pelvis"), dtype=float).reshape(-1)
        wbc_pelvis_rot = np.asarray(self._wbc_debug_engine.getRotation("pelvis"), dtype=float)
        if wbc_pelvis_pos.shape[0] < 3 or not np.all(np.isfinite(wbc_pelvis_pos)):
            raise AssertionError(f"WBC pelvis position is invalid: shape={wbc_pelvis_pos.shape}, pos={wbc_pelvis_pos}")
        if wbc_pelvis_rot.shape != (3, 3) or not np.all(np.isfinite(wbc_pelvis_rot)):
            raise AssertionError(
                f"WBC pelvis rotation is invalid: shape={wbc_pelvis_rot.shape}, rotation={wbc_pelvis_rot}"
            )

        root_state = self.env.simulator.robot_root_states[env_idx]
        pelvis_pos = root_state[:3].detach().cpu().numpy()
        pelvis_rot = quaternion_to_matrix(root_state[3:7].unsqueeze(0), w_last=True)[0].detach().cpu().numpy()
        wbc_pelvis_pos = wbc_pelvis_pos[:3]
        pelvis_pos_delta = pelvis_pos - wbc_pelvis_pos
        pelvis_pos_error = float(np.linalg.norm(pelvis_pos_delta))
        pelvis_rot_error = float(np.linalg.norm(pelvis_rot - wbc_pelvis_rot, ord="fro"))
        return pelvis_pos_error, pelvis_rot_error, pelvis_pos, wbc_pelvis_pos, pelvis_pos_delta

    def _print_foot_frame_inspection(
        self,
        env_idx: int,
        side: str,
        resolved_sim_pos: np.ndarray,
        resolved_sim_rot: np.ndarray,
        wbc_compare_pos: np.ndarray,
        wbc_compare_rot: np.ndarray,
    ) -> None:
        simulator = self.env.simulator
        resolved_idx = self._foot_body_indices[side]
        body_names = list(getattr(simulator, "body_names", []))
        resolved_name = body_names[resolved_idx] if 0 <= resolved_idx < len(body_names) else "<out-of-range>"
        resolved_rot_error = float(np.linalg.norm(resolved_sim_rot - wbc_compare_rot, ord="fro"))

        print(
            "foot_frame_inspection:"
            f" side={side}"
            f" resolved_idx={resolved_idx}"
            f" resolved_name={resolved_name}"
            f" robot_config.foot_body_name={self.env.robot_config.foot_body_name}"
            f" resolved_sim_pos={resolved_sim_pos.tolist()}"
            f" wbc_compare_pos={wbc_compare_pos.tolist()}"
            f" delta={list((resolved_sim_pos - wbc_compare_pos).tolist())}"
            f" rot_error={resolved_rot_error:.9f}"
            "\n"
        )

        # # for suffix in ("ankle_pitch_link", "ankle_roll_link", "foot"):
        # for suffix in ("ankle_roll_link"):
        #     frame_name = f"{side}_{suffix}"
        #     sim_matches = [(idx, name) for idx, name in enumerate(body_names) if name == frame_name]
        #     wbc_rot = None
        #     for sim_idx, sim_name in sim_matches:
        #         sim_pos = simulator._rigid_body_pos[env_idx, sim_idx].detach().cpu().numpy()
        #         sim_quat = simulator._rigid_body_rot[env_idx, sim_idx]
        #         sim_rot = quaternion_to_matrix(sim_quat.unsqueeze(0), w_last=True)[0].detach().cpu().numpy()
        #         try:
        #             if wbc_rot is None:
        #                 wbc_rot = np.asarray(self.wbc[env_idx].getRotation(frame_name), dtype=float)
        #             rot_error = float(np.linalg.norm(sim_rot - wbc_rot, ord="fro"))
        #             print(f"  frame_rotation_error name={sim_name} idx={sim_idx} rot_error={rot_error:.9f}")
        #         except Exception as exc:
        #             print(f"  frame_rotation_error name={sim_name} idx={sim_idx} unavailable={exc}")
