from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.commands import UniformPoseCommand, UniformPoseCommandCfg
from isaaclab.envs.mdp.actions import (
    BinaryJointPositionAction,
    BinaryJointPositionActionCfg,
    JointPositionAction,
    JointPositionActionCfg,
)
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply_inverse,
    quat_error_magnitude,
    subtract_frame_transforms,
)

from isaaclab_tasks.manager_based.manipulation.lift.mdp import *  # noqa: F401, F403

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class StagedObjectPoseCommand(UniformPoseCommand):
    """Pose command that moves the object goal from an easy near-object target to the final target."""

    cfg: "StagedObjectPoseCommandCfg"

    def __init__(self, cfg: "StagedObjectPoseCommandCfg", env):
        super().__init__(cfg, env)
        self.metrics["goal_curriculum_progress"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["object_position_error"] = torch.zeros(self.num_envs, device=self.device)
        for name in (
            "task_grasp_acquired",
            "task_grasp_ready_reached",
            "task_grasp_candidate_reached",
            "task_grasp_attempt_active",
            "task_current_hold",
            "task_lift_completed",
            "task_validated_transport",
            "task_bilateral_finger_proximity",
            "task_grasp_alignment_quality",
            "task_grasp_offset_tcp_x",
            "task_grasp_offset_tcp_y",
            "task_grasp_offset_tcp_z",
            "task_finger_distance_asymmetry",
            "task_grasp_relative_speed",
            "task_grasp_coupling_error",
            "task_grasp_offset_error",
            "task_grasp_closing_progress",
            "task_grasp_contact_progress",
            "task_grasp_lift_authorized",
            "task_grasp_verification_progress",
            "task_grasp_retry_cooldown",
            "task_grasp_attempt_aborts",
            "task_transport_displacement",
            "task_assist_engaged",
            "task_assist_active",
            "task_assist_owns_arm",
            "task_basket_reached",
            "task_inside_release_xy",
            "task_place_centered",
            "task_release_ready",
            "task_release_authorized",
            "task_release_commanded",
            "task_released",
            "task_object_height",
            "task_release_target_height",
            "task_release_height_error",
            "task_release_curriculum_progress",
            "task_object_speed",
            "task_object_xy_speed",
            "task_object_vertical_speed",
            "task_distance_xy",
            "task_centering_height_error",
            "task_assist_control_steps",
            "task_assist_requested_vertical_step",
            "task_assist_incremental_vertical_step",
            "task_assist_vertical_step",
            "task_assist_actual_object_vertical_delta",
            "task_assist_actual_tcp_vertical_delta",
            "task_assist_object_vertical_tracking",
            "task_assist_tcp_vertical_tracking",
            "task_assist_effort_ratio",
            "task_assist_tcp_orientation_error",
            "task_assist_object_slip",
            "task_assist_watchdog_trips",
            "task_assist_posture_error",
            "task_assist_target_lead",
            "task_centering_progress",
            "task_opening_progress",
            "task_basket_exit_progress",
            "task_released_outside_progress",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        super()._update_metrics()
        obj: RigidObject = self._env.scene[self.cfg.object_name]
        self.metrics["object_position_error"] = torch.norm(self.pose_command_w[:, :3] - obj.data.root_pos_w, dim=1)
        state = getattr(self._env, "_openarm_pick_place_state", None)
        if state is not None:
            self.metrics["task_grasp_acquired"] = state.grasp_acquired.float()
            self.metrics["task_grasp_ready_reached"] = state.grasp_ready_reached.float()
            self.metrics["task_grasp_candidate_reached"] = state.grasp_candidate_reached.float()
            self.metrics["task_grasp_attempt_active"] = state.grasp_attempt_active.float()
            self.metrics["task_current_hold"] = state.current_hold.float()
            self.metrics["task_lift_completed"] = state.lift_completed.float()
            self.metrics["task_validated_transport"] = state.validated_transport.float()
            self.metrics["task_bilateral_finger_proximity"] = state.bilateral_finger_proximity.clone()
            self.metrics["task_grasp_alignment_quality"] = state.grasp_alignment_quality.clone()
            self.metrics["task_grasp_offset_tcp_x"] = state.object_offset_tcp[:, 0].clone()
            self.metrics["task_grasp_offset_tcp_y"] = state.object_offset_tcp[:, 1].clone()
            self.metrics["task_grasp_offset_tcp_z"] = state.object_offset_tcp[:, 2].clone()
            self.metrics["task_finger_distance_asymmetry"] = (
                state.finger_distance_asymmetry.clone()
            )
            self.metrics["task_grasp_relative_speed"] = state.grasp_relative_speed.clone()
            self.metrics["task_grasp_coupling_error"] = state.grasp_coupling_error.clone()
            self.metrics["task_grasp_offset_error"] = state.grasp_offset_error.clone()
            self.metrics["task_grasp_closing_progress"] = (
                state.grasp_closing_streak.float() / max(state._closing_timeout_steps, 1)
            )
            self.metrics["task_grasp_contact_progress"] = (
                state.grasp_contact_streak.float() / max(state._contact_confirm_steps, 1)
            )
            self.metrics["task_grasp_lift_authorized"] = state.grasp_lift_authorized.float()
            self.metrics["task_grasp_verification_progress"] = (
                state.grasp_verification_streak.float()
                / max(state._verification_timeout_steps, 1)
            )
            self.metrics["task_grasp_retry_cooldown"] = state.grasp_retry_cooldown.float()
            self.metrics["task_grasp_attempt_aborts"] = state.grasp_attempt_aborts.float()
            self.metrics["task_transport_displacement"] = state.transport_displacement.clone()
            self.metrics["task_assist_engaged"] = state.assist_engaged.float()
            self.metrics["task_assist_active"] = state.assist_active.float()
            self.metrics["task_assist_owns_arm"] = state.assist_owns_arm.float()
            self.metrics["task_basket_reached"] = state.basket_reached.float()
            self.metrics["task_inside_release_xy"] = state.inside_release_xy.float()
            self.metrics["task_place_centered"] = state.place_centered.float()
            self.metrics["task_release_ready"] = state.release_ready.float()
            self.metrics["task_release_authorized"] = state.release_ready_reached.float()
            self.metrics["task_release_commanded"] = state.release_commanded.float()
            self.metrics["task_released"] = state.released.float()
            self.metrics["task_object_height"] = state.object_height.clone()
            self.metrics["task_release_target_height"] = state.release_target_height.clone()
            self.metrics["task_release_height_error"] = state.release_height_error.clone()
            self.metrics["task_release_curriculum_progress"] = state.release_curriculum_progress.clone()
            self.metrics["task_object_speed"] = state.object_speed.clone()
            self.metrics["task_object_xy_speed"] = state.object_xy_speed.clone()
            self.metrics["task_object_vertical_speed"] = state.object_vertical_speed.clone()
            self.metrics["task_distance_xy"] = state.distance_xy.clone()
            self.metrics["task_centering_height_error"] = state.centering_height_error.clone()
            arm_action = self._env.action_manager.get_term("arm_action")
            if hasattr(arm_action, "assist_vertical_step"):
                self.metrics["task_assist_control_steps"].add_(state.assist_active.float())
                self.metrics["task_assist_requested_vertical_step"].add_(
                    arm_action.assist_requested_vertical_step
                )
                self.metrics["task_assist_incremental_vertical_step"].add_(
                    arm_action.assist_incremental_vertical_step
                )
                self.metrics["task_assist_vertical_step"].add_(arm_action.assist_vertical_step)
                self.metrics["task_assist_actual_object_vertical_delta"].add_(
                    arm_action.assist_actual_object_vertical_delta
                )
                self.metrics["task_assist_actual_tcp_vertical_delta"].add_(
                    arm_action.assist_actual_tcp_vertical_delta
                )
                self.metrics["task_assist_object_vertical_tracking"].add_(
                    arm_action.assist_object_vertical_tracking
                )
                self.metrics["task_assist_tcp_vertical_tracking"].add_(
                    arm_action.assist_tcp_vertical_tracking
                )
                self.metrics["task_assist_effort_ratio"] = torch.maximum(
                    self.metrics["task_assist_effort_ratio"], arm_action.assist_effort_ratio
                )
                self.metrics["task_assist_tcp_orientation_error"] = torch.maximum(
                    self.metrics["task_assist_tcp_orientation_error"],
                    arm_action.assist_tcp_orientation_error,
                )
                self.metrics["task_assist_object_slip"] = torch.maximum(
                    self.metrics["task_assist_object_slip"], arm_action.assist_object_slip
                )
                self.metrics["task_assist_watchdog_trips"].add_(arm_action.assist_watchdog_trip)
                self.metrics["task_assist_posture_error"] = torch.maximum(
                    self.metrics["task_assist_posture_error"], arm_action.assist_posture_error
                )
                self.metrics["task_assist_target_lead"] = torch.maximum(
                    self.metrics["task_assist_target_lead"], arm_action.assist_target_lead
                )
            self.metrics["task_centering_progress"] = state.center_streak.float() / max(
                state._place_center_confirm_steps, 1
            )
            self.metrics["task_opening_progress"] = state.opening_streak.float() / max(
                state._release_open_grace_steps, 1
            )
            self.metrics["task_basket_exit_progress"] = state.basket_exit_streak.float() / max(
                state._basket_exit_confirm_steps, 1
            )
            self.metrics["task_released_outside_progress"] = state.outside_streak.float() / max(
                state._released_outside_confirm_steps, 1
            )

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)

        obj: RigidObject = self._env.scene[self.cfg.object_name]
        object_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            obj.data.root_pos_w[env_ids],
        )
        final_pos_b = self.pose_command_b[env_ids, :3].clone()

        if self.cfg.force_final_goal:
            progress = 1.0
        else:
            progress = min(max(self._env.common_step_counter / self.cfg.curriculum_num_steps, 0.0), 1.0)
            progress = progress**self.cfg.curriculum_power
        start_offset = torch.tensor(
            self.cfg.start_offset_from_object, device=self.device, dtype=self.pose_command_b.dtype
        )
        easy_pos_b = object_pos_b + start_offset
        self.pose_command_b[env_ids, :3] = easy_pos_b + progress * (final_pos_b - easy_pos_b)
        self.metrics["goal_curriculum_progress"][env_ids] = progress


@configclass
class StagedObjectPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for staged object transport commands."""

    class_type: type = StagedObjectPoseCommand
    object_name: str = "object"
    start_offset_from_object: tuple[float, float, float] = (0.04, 0.05, 0.08)
    curriculum_num_steps: int = 72_000
    curriculum_power: float = 1.5
    force_final_goal: bool = False


class HystereticBinaryJointPositionAction(BinaryJointPositionAction):
    """Binary gripper action with optional deterministic task-phase control."""

    cfg: "HystereticBinaryJointPositionActionCfg"

    def __init__(self, cfg: "HystereticBinaryJointPositionActionCfg", env):
        super().__init__(cfg, env)
        self._is_open = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._processed_actions[:] = self._open_command

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        requested_open = torch.where(
            actions[:, 0] >= self.cfg.open_threshold,
            True,
            torch.where(actions[:, 0] <= self.cfg.close_threshold, False, self._is_open),
        )
        if self.cfg.enforce_task_phase:
            state = getattr(self._env, "_openarm_pick_place_state", None)
            if state is not None:
                release_ready = (
                    state.grasp_acquired
                    & state.lift_completed
                    & state.validated_transport
                    & state.release_ready_reached
                    & state.release_ready
                )
                if self.cfg.automatic_phase_control:
                    safe_release_request = release_ready
                else:
                    actor_requests_open = actions[:, 0] >= self.cfg.open_threshold
                    safe_release_request = release_ready & actor_requests_open
                state.request_release(safe_release_request)
                if self.cfg.automatic_phase_control:
                    close_for_pick = state.grasp_attempt_active | state.grasp_acquired
                    requested_open = torch.where(
                        state.release_commanded | state.released,
                        True,
                        ~close_for_pick,
                    )
                else:
                    requested_open = torch.where(
                        state.release_commanded | state.released,
                        True,
                        torch.where(state.grasp_acquired, False, requested_open),
                    )
                state.gripper_target_open[:] = requested_open
        self._is_open = requested_open
        self._processed_actions = torch.where(
            self._is_open[:, None], self._open_command[None, :], self._close_command[None, :]
        )

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._is_open[env_ids] = True
        self._processed_actions[env_ids] = self._open_command


@configclass
class HystereticBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    class_type: type = HystereticBinaryJointPositionAction
    open_threshold: float = 0.15
    close_threshold: float = -0.15
    enforce_task_phase: bool = False
    automatic_phase_control: bool = False


class PlaceAssistJointPositionAction(JointPositionAction):
    """Drive TCP translation from closed-loop object error after assist entry."""

    cfg: "PlaceAssistJointPositionActionCfg"

    def __init__(self, cfg: "PlaceAssistJointPositionActionCfg", env):
        super().__init__(cfg, env)
        body_ids, body_names = self._asset.find_bodies(cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one body for place assist, found {body_names}")
        body_id = body_ids[0]
        self._body_id = body_id
        self._jacobian_body_id = body_id - 1 if self._asset.is_fixed_base else body_id
        self._jacobian_joint_ids = (
            self._joint_ids
            if self._asset.is_fixed_base
            else [joint_id + 6 for joint_id in self._joint_ids]
        )
        self._identity_6 = torch.eye(6, device=self.device).unsqueeze(0)
        self._identity_joints = torch.eye(self.action_dim, device=self.device).unsqueeze(0)
        self._assist_was_active = torch.zeros(self._env.num_envs, dtype=torch.bool, device=self.device)
        self._assist_posture_ref = torch.zeros(
            (self._env.num_envs, self.action_dim), device=self.device
        )
        self._assist_entry_offset = torch.zeros((self._env.num_envs, 3), device=self.device)
        self._assist_entry_orientation = torch.zeros((self._env.num_envs, 4), device=self.device)
        self._slip_streak = torch.zeros(self._env.num_envs, dtype=torch.long, device=self.device)
        self._assist_joint_target = torch.zeros(
            (self._env.num_envs, self.action_dim), device=self.device
        )
        self._grasp_hold_target = torch.zeros(
            (self._env.num_envs, self.action_dim), device=self.device
        )
        self._grasp_was_controlled = torch.zeros(
            self._env.num_envs, dtype=torch.bool, device=self.device
        )
        self._grasp_handoff_steps_remaining = torch.zeros(
            self._env.num_envs, dtype=torch.long, device=self.device
        )
        self._handoff_steps_remaining = torch.zeros(
            self._env.num_envs, dtype=torch.long, device=self.device
        )
        self._filtered_object_velocity = torch.zeros((self._env.num_envs, 3), device=self.device)
        self._previous_object_height = torch.zeros(self._env.num_envs, device=self.device)
        self._previous_tcp_height = torch.zeros(self._env.num_envs, device=self.device)
        self._motion_height_initialized = torch.zeros(
            self._env.num_envs, dtype=torch.bool, device=self.device
        )
        self._previous_requested_vertical_step = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_requested_vertical_step = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_incremental_vertical_step = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_vertical_step = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_actual_object_vertical_delta = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_actual_tcp_vertical_delta = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_object_vertical_tracking = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_tcp_vertical_tracking = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_effort_ratio = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_tcp_orientation_error = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_object_slip = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_watchdog_trip = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_posture_error = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_target_lead = torch.zeros(self._env.num_envs, device=self.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        resolved_env_ids = slice(None) if env_ids is None else env_ids
        self._assist_was_active[resolved_env_ids] = False
        self._assist_posture_ref[resolved_env_ids] = 0.0
        self._assist_entry_offset[resolved_env_ids] = 0.0
        self._assist_entry_orientation[resolved_env_ids] = 0.0
        self._slip_streak[resolved_env_ids] = 0
        self._assist_joint_target[resolved_env_ids] = 0.0
        self._grasp_hold_target[resolved_env_ids] = 0.0
        self._grasp_was_controlled[resolved_env_ids] = False
        self._grasp_handoff_steps_remaining[resolved_env_ids] = 0
        self._handoff_steps_remaining[resolved_env_ids] = 0
        self._filtered_object_velocity[resolved_env_ids] = 0.0
        self._previous_object_height[resolved_env_ids] = 0.0
        self._previous_tcp_height[resolved_env_ids] = 0.0
        self._motion_height_initialized[resolved_env_ids] = False
        self._previous_requested_vertical_step[resolved_env_ids] = 0.0
        self.assist_requested_vertical_step[resolved_env_ids] = 0.0
        self.assist_incremental_vertical_step[resolved_env_ids] = 0.0
        self.assist_vertical_step[resolved_env_ids] = 0.0
        self.assist_actual_object_vertical_delta[resolved_env_ids] = 0.0
        self.assist_actual_tcp_vertical_delta[resolved_env_ids] = 0.0
        self.assist_object_vertical_tracking[resolved_env_ids] = 0.0
        self.assist_tcp_vertical_tracking[resolved_env_ids] = 0.0
        self.assist_effort_ratio[resolved_env_ids] = 0.0
        self.assist_tcp_orientation_error[resolved_env_ids] = 0.0
        self.assist_object_slip[resolved_env_ids] = 0.0
        self.assist_watchdog_trip[resolved_env_ids] = 0.0
        self.assist_posture_error[resolved_env_ids] = 0.0
        self.assist_target_lead[resolved_env_ids] = 0.0

    @staticmethod
    def _bounded_direction_step(
        error: torch.Tensor,
        velocity: torch.Tensor,
        position_gain: float,
        velocity_damping: float,
        step_dt: float,
        max_step: float,
        deadband: float,
    ) -> torch.Tensor:
        """Return a target-directed step that cannot overshoot or reverse direction."""

        distance = torch.norm(error, dim=1, keepdim=True)
        direction = error / torch.clamp(distance, min=1.0e-6)
        velocity_toward_target = torch.sum(velocity * direction, dim=1, keepdim=True)
        step_magnitude = position_gain * distance - velocity_damping * velocity_toward_target * step_dt
        step_magnitude = torch.clamp(step_magnitude, min=0.0, max=max_step)
        step_magnitude = torch.minimum(step_magnitude, distance)
        step_magnitude = torch.where(distance > deadband, step_magnitude, 0.0)
        return direction * step_magnitude

    @staticmethod
    def _scale_by_max_abs(delta: torch.Tensor, max_abs: float) -> torch.Tensor:
        """Scale whole vectors uniformly so their direction is preserved."""

        largest = torch.amax(torch.abs(delta), dim=1, keepdim=True)
        scale = torch.clamp(max_abs / torch.clamp(largest, min=1.0e-9), max=1.0)
        return delta * scale

    @staticmethod
    def _scale_to_joint_limits(
        target: torch.Tensor,
        delta: torch.Tensor,
        joint_limits: torch.Tensor,
    ) -> torch.Tensor:
        """Uniformly scale an increment so the target remains inside every joint limit."""

        infinite = torch.full_like(delta, float("inf"))
        upper_scale = torch.where(
            delta > 0.0,
            (joint_limits[:, :, 1] - target) / torch.clamp(delta, min=1.0e-9),
            infinite,
        )
        lower_scale = torch.where(
            delta < 0.0,
            (joint_limits[:, :, 0] - target) / torch.clamp(delta, max=-1.0e-9),
            infinite,
        )
        scale = torch.amin(torch.minimum(upper_scale, lower_scale), dim=1, keepdim=True)
        return delta * torch.clamp(scale, min=0.0, max=1.0)

    def _cartesian_direction_safe(
        self,
        predicted_step: torch.Tensor,
        position_error: torch.Tensor,
    ) -> torch.Tensor:
        """Check that predicted TCP motion does not move away on XY or Z."""

        xy_error = torch.norm(position_error[:, :2], dim=1)
        xy_progress = torch.sum(predicted_step[:, :2] * position_error[:, :2], dim=1)
        xy_remaining = torch.norm(position_error[:, :2] - predicted_step[:, :2], dim=1)
        xy_safe = torch.where(
            xy_error > self.cfg.position_deadband,
            (xy_progress >= -1.0e-9) & (xy_remaining <= xy_error + 1.0e-9),
            torch.norm(predicted_step[:, :2], dim=1) <= self.cfg.cartesian_leak_tolerance,
        )
        z_error = position_error[:, 2]
        z_remaining = torch.abs(z_error - predicted_step[:, 2])
        z_safe = torch.where(
            torch.abs(z_error) > self.cfg.position_deadband,
            (predicted_step[:, 2] * z_error >= -1.0e-9)
            & (z_remaining <= torch.abs(z_error) + 1.0e-9),
            torch.abs(predicted_step[:, 2]) <= self.cfg.cartesian_leak_tolerance,
        )
        return xy_safe & z_safe

    def _apply_grasp_verification_lift(
        self,
        verification_ids: torch.Tensor,
        current_joint_pos: torch.Tensor,
        state: "TaskStateTracker",
        robot: Articulation,
    ):
        """Apply a small orientation-preserving lift to verify physical object coupling."""

        if verification_ids.numel() == 0:
            return
        spatial_jacobian = self._asset.root_physx_view.get_jacobians()[
            :, self._jacobian_body_id, :, :
        ]
        task_jacobian = spatial_jacobian[verification_ids][:, :, self._jacobian_joint_ids]
        delta_twist = torch.zeros((verification_ids.numel(), 6), device=self.device)
        current_tcp_height = robot.data.body_pos_w[verification_ids, self._body_id, 2]
        target_tcp_height = (
            state.grasp_candidate_tcp_origin[verification_ids, 2]
            + self.cfg.grasp_verification_height
        )
        delta_twist[:, 2] = torch.clamp(
            target_tcp_height - current_tcp_height,
            min=0.0,
            max=self.cfg.grasp_verification_step,
        )
        jacobian_t = torch.transpose(task_jacobian, 1, 2)
        damped = task_jacobian @ jacobian_t + self.cfg.damping**2 * self._identity_6.expand(
            verification_ids.numel(), -1, -1
        )
        delta_joint = (
            jacobian_t
            @ torch.linalg.solve(
                damped,
                delta_twist.unsqueeze(-1),
            )
        ).squeeze(-1)
        delta_joint = self._scale_by_max_abs(
            delta_joint, self.cfg.grasp_verification_max_joint_step
        )
        predicted_twist = (task_jacobian @ delta_joint.unsqueeze(-1)).squeeze(-1)
        delta_joint = torch.where(
            (predicted_twist[:, 2] > 0.0)[:, None], delta_joint, 0.0
        )
        joint_limits = self._asset.data.soft_joint_pos_limits[verification_ids][
            :, self._joint_ids
        ]
        delta_joint = self._scale_to_joint_limits(
            current_joint_pos[verification_ids], delta_joint, joint_limits
        )
        target = current_joint_pos[verification_ids] + delta_joint
        self._grasp_hold_target[verification_ids] = target
        self._processed_actions[verification_ids] = target

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        policy_target = self._processed_actions.clone()
        state = getattr(self._env, "_openarm_pick_place_state", None)
        if state is None:
            return
        state.assist_active[:] = False
        state.assist_owns_arm[:] = False
        self.assist_requested_vertical_step[:] = 0.0
        self.assist_incremental_vertical_step[:] = 0.0
        self.assist_vertical_step[:] = 0.0
        self.assist_actual_object_vertical_delta[:] = 0.0
        self.assist_actual_tcp_vertical_delta[:] = 0.0
        self.assist_object_vertical_tracking[:] = 0.0
        self.assist_tcp_vertical_tracking[:] = 0.0
        self.assist_effort_ratio[:] = 0.0
        self.assist_tcp_orientation_error[:] = 0.0
        self.assist_object_slip[:] = 0.0
        self.assist_watchdog_trip[:] = 0.0
        self.assist_posture_error[:] = 0.0
        self.assist_target_lead[:] = 0.0
        robot: Articulation = self._env.scene[self.cfg.asset_name]
        obj: RigidObject = self._env.scene[self.cfg.object_name]
        current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        new_grasp_attempt = state.try_start_grasp_attempt(self._env)
        self._grasp_hold_target[new_grasp_attempt] = current_joint_pos[new_grasp_attempt]
        closing_hold = (
            state.grasp_attempt_active
            & ~state.grasp_lift_authorized
            & ~state.grasp_acquired
            & ~state.failed
        )
        self._processed_actions[closing_hold] = self._grasp_hold_target[closing_hold]
        verification_lift = (
            state.grasp_attempt_active
            & state.grasp_lift_authorized
            & ~state.grasp_acquired
            & ~state.failed
        )
        state.grasp_verification_active[:] = verification_lift
        verification_ids = torch.nonzero(verification_lift, as_tuple=False).squeeze(-1)
        self._apply_grasp_verification_lift(
            verification_ids, current_joint_pos, state, robot
        )
        grasp_control = closing_hold | verification_lift
        grasp_control_ended = self._grasp_was_controlled & ~grasp_control
        self._grasp_handoff_steps_remaining[grasp_control_ended] = self.cfg.grasp_handoff_steps
        self._grasp_was_controlled[:] = grasp_control
        grasp_handoff = (
            ~grasp_control
            & (self._grasp_handoff_steps_remaining > 0)
            & ~state.failed
        )
        grasp_handoff_delta = self._scale_by_max_abs(
            policy_target - current_joint_pos, self.cfg.max_joint_step
        )
        self._processed_actions[grasp_handoff] = (
            current_joint_pos + grasp_handoff_delta
        )[grasp_handoff]
        self._grasp_handoff_steps_remaining[grasp_handoff] -= 1
        previously_active = self._assist_was_active.clone()
        current_object_height = obj.data.root_pos_w[:, 2]
        current_tcp_height = robot.data.body_pos_w[:, self._body_id, 2]
        initialized = self._motion_height_initialized
        object_vertical_delta = torch.where(
            initialized, current_object_height - self._previous_object_height, 0.0
        )
        tcp_vertical_delta = torch.where(
            initialized, current_tcp_height - self._previous_tcp_height, 0.0
        )
        self.assist_actual_object_vertical_delta[previously_active] = object_vertical_delta[
            previously_active
        ]
        self.assist_actual_tcp_vertical_delta[previously_active] = tcp_vertical_delta[
            previously_active
        ]
        previous_vertical_direction = torch.sign(self._previous_requested_vertical_step)
        self.assist_object_vertical_tracking[previously_active] = (
            previous_vertical_direction * object_vertical_delta
        )[previously_active]
        self.assist_tcp_vertical_tracking[previously_active] = (
            previous_vertical_direction * tcp_vertical_delta
        )[previously_active]
        self._previous_requested_vertical_step[:] = 0.0
        effort_limits = torch.clamp(
            robot.data.joint_effort_limits[:, self._joint_ids], min=1.0e-6
        )
        effort_ratio = torch.amax(
            torch.abs(robot.data.computed_torque[:, self._joint_ids]) / effort_limits, dim=1
        )
        self.assist_effort_ratio[previously_active] = effort_ratio[previously_active]
        self._previous_object_height[:] = current_object_height
        self._previous_tcp_height[:] = current_tcp_height
        self._motion_height_initialized[:] = True
        bounded_object_velocity = torch.clamp(
            obj.data.root_lin_vel_w,
            min=-self.cfg.max_object_speed_for_control,
            max=self.cfg.max_object_speed_for_control,
        )
        self._filtered_object_velocity.mul_(self.cfg.velocity_ema_decay).add_(
            bounded_object_velocity, alpha=1.0 - self.cfg.velocity_ema_decay
        )
        post_assist = (
            state.grasp_acquired
            & state.assist_engaged
            & ~state.released
            & ~state.failed
        )
        assist_active = (
            post_assist
            & state.current_hold
            & (state.distance_xy < self.cfg.outer_xy_threshold)
        )
        hold_recovery = post_assist & ~state.current_hold
        state.assist_active[:] = assist_active
        deactivated = previously_active & ~assist_active & ~hold_recovery
        self._handoff_steps_remaining[deactivated] = self.cfg.handoff_steps
        release_handoff = deactivated & state.released
        self._assist_joint_target[release_handoff] = current_joint_pos[release_handoff]
        newly_active = assist_active & ~self._assist_was_active
        resuming_handoff = newly_active & (self._handoff_steps_remaining > 0)
        fresh_assist = newly_active & ~resuming_handoff
        self._assist_posture_ref[fresh_assist] = current_joint_pos[fresh_assist]
        self._assist_entry_offset[fresh_assist] = (
            obj.data.root_pos_w[fresh_assist] - robot.data.body_pos_w[fresh_assist, self._body_id]
        )
        self._assist_entry_orientation[fresh_assist] = robot.data.body_quat_w[
            fresh_assist, self._body_id
        ]
        self._slip_streak[fresh_assist] = 0
        entry_lead = self._scale_by_max_abs(
            policy_target - current_joint_pos, self.cfg.max_target_lead
        )
        self._assist_joint_target[fresh_assist] = (
            current_joint_pos + entry_lead
        )[fresh_assist]
        self._handoff_steps_remaining[assist_active] = 0
        self._assist_was_active[:] = assist_active
        self._slip_streak[~assist_active] = 0

        policy_slew = state.assist_engaged & ~assist_active & ~hold_recovery
        policy_delta = self._scale_by_max_abs(
            policy_target - current_joint_pos, self.cfg.max_joint_step
        )
        self._processed_actions[policy_slew] = (current_joint_pos + policy_delta)[policy_slew]
        self._assist_joint_target[hold_recovery] = current_joint_pos[hold_recovery]
        self._processed_actions[hold_recovery] = current_joint_pos[hold_recovery]
        self._handoff_steps_remaining[hold_recovery] = 0
        handoff = (~assist_active) & ~hold_recovery & (self._handoff_steps_remaining > 0)
        handoff_lead = self._scale_by_max_abs(
            self._assist_joint_target - current_joint_pos, self.cfg.max_target_lead
        )
        self._assist_joint_target[handoff] = (current_joint_pos + handoff_lead)[handoff]
        self._processed_actions[handoff] = self._assist_joint_target[handoff]
        self._handoff_steps_remaining[handoff] -= 1
        state.assist_owns_arm[:] = (
            grasp_control | grasp_handoff | assist_active | hold_recovery | handoff
        )

        assist_ids = torch.nonzero(assist_active, as_tuple=False).squeeze(-1)
        if assist_ids.numel() == 0:
            return

        spatial_jacobian = self._asset.root_physx_view.get_jacobians()[:, self._jacobian_body_id, :, :]
        spatial_jacobian = spatial_jacobian[assist_ids][:, :, self._jacobian_joint_ids]
        task_jacobian = spatial_jacobian
        position_jacobian = task_jacobian[:, :3, :]

        current_object_offset = (
            obj.data.root_pos_w[assist_ids] - robot.data.body_pos_w[assist_ids, self._body_id]
        )
        object_slip = torch.norm(
            current_object_offset - self._assist_entry_offset[assist_ids], dim=1
        )
        self.assist_object_slip[assist_ids] = object_slip
        self.assist_tcp_orientation_error[assist_ids] = quat_error_magnitude(
            robot.data.body_quat_w[assist_ids, self._body_id],
            self._assist_entry_orientation[assist_ids],
        )
        slip_exceeded = object_slip > self.cfg.slip_watchdog_threshold
        self._slip_streak[assist_ids] = torch.where(
            slip_exceeded,
            torch.clamp(self._slip_streak[assist_ids] + 1, max=self.cfg.slip_watchdog_steps),
            0,
        )
        slip_watchdog = self._slip_streak[assist_ids] >= self.cfg.slip_watchdog_steps
        slip_watchdog_ids = assist_ids[slip_watchdog]
        self.assist_watchdog_trip[slip_watchdog_ids] = 1.0
        self._assist_joint_target[slip_watchdog_ids] = current_joint_pos[slip_watchdog_ids]
        self._assist_entry_offset[slip_watchdog_ids] = current_object_offset[slip_watchdog]
        self._slip_streak[slip_watchdog_ids] = 0

        delta_position = torch.zeros((assist_ids.numel(), 3), device=self.device)
        command = self._env.command_manager.get_command(self.cfg.command_name)
        goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
        target_object_position = goal_pos_w[assist_ids, :3].clone()
        target_object_position[:, 2] = torch.where(
            state.place_centered[assist_ids],
            state.release_target_height[assist_ids],
            self.cfg.centering_height,
        )
        object_error = target_object_position - obj.data.root_pos_w[assist_ids]
        filtered_velocity = self._filtered_object_velocity[assist_ids]
        delta_position[:, :2] = self._bounded_direction_step(
            object_error[:, :2],
            filtered_velocity[:, :2],
            self.cfg.horizontal_position_gain,
            self.cfg.horizontal_velocity_damping,
            self._env.step_dt,
            self.cfg.max_horizontal_step,
            self.cfg.position_deadband,
        )
        vertical_error = object_error[:, 2:3]
        vertical_limit = torch.where(
            vertical_error >= 0.0,
            self.cfg.max_upward_step,
            self.cfg.max_vertical_step,
        )
        vertical_direction = torch.sign(vertical_error)
        vertical_velocity_toward_target = filtered_velocity[:, 2:3] * vertical_direction
        vertical_magnitude = (
            self.cfg.vertical_position_gain * torch.abs(vertical_error)
            - self.cfg.vertical_velocity_damping * vertical_velocity_toward_target * self._env.step_dt
        )
        vertical_magnitude = torch.clamp(vertical_magnitude, min=0.0)
        vertical_magnitude = torch.minimum(vertical_magnitude, vertical_limit)
        vertical_magnitude = torch.minimum(vertical_magnitude, torch.abs(vertical_error))
        vertical_magnitude = torch.where(
            torch.abs(vertical_error) > self.cfg.position_deadband,
            vertical_magnitude,
            0.0,
        )
        delta_position[:, 2:3] = vertical_direction * vertical_magnitude
        delta_position[slip_watchdog] = 0.0
        self.assist_requested_vertical_step[assist_ids] = delta_position[:, 2]
        self._previous_requested_vertical_step[assist_ids] = delta_position[:, 2]

        requested_xy_norm = torch.norm(delta_position[:, :2], dim=1)
        braking = (
            (torch.norm(object_error[:, :2], dim=1) > self.cfg.position_deadband)
            & (requested_xy_norm <= 1.0e-9)
        ) | (
            (torch.abs(vertical_error[:, 0]) > self.cfg.position_deadband)
            & (torch.abs(delta_position[:, 2]) <= 1.0e-9)
        )
        braking_ids = assist_ids[braking]
        self._assist_joint_target[braking_ids] = current_joint_pos[braking_ids]

        delta_twist = torch.zeros((assist_ids.numel(), 6), device=self.device)
        delta_twist[:, :3] = delta_position
        jacobian_t = torch.transpose(task_jacobian, 1, 2)
        damped = task_jacobian @ jacobian_t + self.cfg.damping**2 * self._identity_6.expand(
            assist_ids.numel(), -1, -1
        )
        damped_inverse = torch.linalg.solve(
            damped, self._identity_6.expand(assist_ids.numel(), -1, -1)
        )
        jacobian_pinv = jacobian_t @ damped_inverse
        primary_delta = (jacobian_pinv @ delta_twist.unsqueeze(-1)).squeeze(-1)
        primary_residual = delta_twist - (
            task_jacobian @ primary_delta.unsqueeze(-1)
        ).squeeze(-1)
        primary_delta += (jacobian_pinv @ primary_residual.unsqueeze(-1)).squeeze(-1)

        posture_error = self._assist_posture_ref[assist_ids] - current_joint_pos[assist_ids]
        nullspace_projector = self._identity_joints.expand(assist_ids.numel(), -1, -1) - (
            jacobian_pinv @ task_jacobian
        )
        posture_delta = (
            nullspace_projector @ (self.cfg.nullspace_gain * posture_error).unsqueeze(-1)
        ).squeeze(-1)
        posture_delta = self._scale_by_max_abs(posture_delta, self.cfg.max_nullspace_step)
        self.assist_posture_error[assist_ids] = torch.norm(posture_error, dim=1)

        combined_delta = self._scale_by_max_abs(
            primary_delta + posture_delta, self.cfg.max_joint_step
        )
        fallback_delta = self._scale_by_max_abs(primary_delta, self.cfg.max_joint_step)
        predicted_combined = (position_jacobian @ combined_delta.unsqueeze(-1)).squeeze(-1)
        combined_safe = self._cartesian_direction_safe(predicted_combined, object_error)
        delta_joint = torch.where(combined_safe[:, None], combined_delta, fallback_delta)
        predicted_step = (position_jacobian @ delta_joint.unsqueeze(-1)).squeeze(-1)
        direction_safe = self._cartesian_direction_safe(predicted_step, object_error)
        delta_joint = torch.where(direction_safe[:, None], delta_joint, 0.0)
        predicted_step = (position_jacobian @ delta_joint.unsqueeze(-1)).squeeze(-1)

        requested_z = torch.abs(delta_position[:, 2])
        predicted_xy = torch.norm(predicted_step[:, :2], dim=1)
        predicted_z = torch.abs(predicted_step[:, 2])
        allowed_xy = torch.where(
            requested_xy_norm > 0.0,
            requested_xy_norm,
            self.cfg.cartesian_leak_tolerance,
        )
        allowed_z = torch.where(
            requested_z > 0.0,
            requested_z,
            self.cfg.cartesian_leak_tolerance,
        )
        cartesian_scale = torch.minimum(
            allowed_xy / torch.clamp(predicted_xy, min=1.0e-9),
            allowed_z / torch.clamp(predicted_z, min=1.0e-9),
        )
        delta_joint *= torch.clamp(cartesian_scale, max=1.0).unsqueeze(1)

        joint_limits = self._asset.data.soft_joint_pos_limits[assist_ids][:, self._joint_ids]
        delta_joint = self._scale_to_joint_limits(
            self._assist_joint_target[assist_ids], delta_joint, joint_limits
        )
        predicted_increment = (position_jacobian @ delta_joint.unsqueeze(-1)).squeeze(-1)
        self.assist_incremental_vertical_step[assist_ids] = predicted_increment[:, 2]
        proposed_target = self._assist_joint_target[assist_ids] + delta_joint
        target_lead = self._scale_by_max_abs(
            proposed_target - current_joint_pos[assist_ids], self.cfg.max_target_lead
        )
        predicted_lead_twist = (task_jacobian @ target_lead.unsqueeze(-1)).squeeze(-1)
        predicted_lead = predicted_lead_twist[:, :3]
        lead_xy_error = torch.norm(object_error[:, :2], dim=1)
        lead_z_error = torch.abs(object_error[:, 2])
        allowed_lead_xy = torch.where(
            lead_xy_error > self.cfg.position_deadband,
            torch.clamp(lead_xy_error, max=self.cfg.max_horizontal_target_lead),
            self.cfg.cartesian_leak_tolerance,
        )
        vertical_target_lead = torch.where(
            object_error[:, 2] >= 0.0,
            self.cfg.max_upward_target_lead,
            self.cfg.max_downward_target_lead,
        )
        allowed_lead_z = torch.where(
            lead_z_error > self.cfg.position_deadband,
            torch.minimum(lead_z_error, vertical_target_lead),
            self.cfg.cartesian_leak_tolerance,
        )
        position_lead_scale = torch.minimum(
            allowed_lead_xy
            / torch.clamp(torch.norm(predicted_lead[:, :2], dim=1), min=1.0e-9),
            allowed_lead_z / torch.clamp(torch.abs(predicted_lead[:, 2]), min=1.0e-9),
        )
        angular_lead_scale = self.cfg.max_angular_target_lead / torch.clamp(
            torch.norm(predicted_lead_twist[:, 3:], dim=1), min=1.0e-9
        )
        lead_scale = torch.minimum(position_lead_scale, angular_lead_scale)
        target_lead *= torch.clamp(lead_scale, max=1.0).unsqueeze(1)
        target_lead = self._scale_to_joint_limits(
            current_joint_pos[assist_ids], target_lead, joint_limits
        )
        proposed_target = current_joint_pos[assist_ids] + target_lead
        predicted_lead = (position_jacobian @ target_lead.unsqueeze(-1)).squeeze(-1)
        lead_safe = self._cartesian_direction_safe(predicted_lead, object_error)
        fallback_delta = self._scale_to_joint_limits(
            current_joint_pos[assist_ids], delta_joint, joint_limits
        )
        assisted_target = torch.where(
            lead_safe[:, None],
            proposed_target,
            current_joint_pos[assist_ids] + fallback_delta,
        )
        self._assist_joint_target[assist_ids] = assisted_target
        applied_delta = assisted_target - current_joint_pos[assist_ids]
        predicted_applied_step = (position_jacobian @ applied_delta.unsqueeze(-1)).squeeze(-1)
        self.assist_vertical_step[assist_ids] = predicted_applied_step[:, 2]
        self.assist_target_lead[assist_ids] = torch.amax(torch.abs(applied_delta), dim=1)
        self._processed_actions[assist_ids] = assisted_target


@configclass
class PlaceAssistJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = PlaceAssistJointPositionAction
    body_name: str = ""
    object_name: str = "object"
    command_name: str = "object_pose"
    horizontal_position_gain: float = 0.12
    horizontal_velocity_damping: float = 0.25
    vertical_position_gain: float = 0.12
    vertical_velocity_damping: float = 0.30
    centering_height: float = 0.75
    max_horizontal_step: float = 0.0020
    max_vertical_step: float = 0.0030
    max_upward_step: float = 0.0025
    max_joint_step: float = 0.03
    max_target_lead: float = 0.12
    max_horizontal_target_lead: float = 0.012
    max_upward_target_lead: float = 0.025
    max_downward_target_lead: float = 0.012
    max_angular_target_lead: float = 0.03
    nullspace_gain: float = 0.04
    max_nullspace_step: float = 0.003
    velocity_ema_decay: float = 0.80
    max_object_speed_for_control: float = 0.25
    position_deadband: float = 0.003
    cartesian_leak_tolerance: float = 0.0002
    slip_watchdog_threshold: float = 0.015
    slip_watchdog_steps: int = 3
    handoff_steps: int = 12
    damping: float = 0.08
    outer_xy_threshold: float = 0.24
    grasp_verification_height: float = 0.018
    grasp_verification_step: float = 0.0015
    grasp_verification_max_joint_step: float = 0.015
    grasp_handoff_steps: int = 6


def object_body_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
) -> torch.Tensor:
    """Reward proximity between the object and a robot body using a tanh kernel."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_id = robot_body_cfg.body_ids[0]
    body_pos_w = robot.data.body_pos_w[:, body_id, :]
    distance = torch.norm(obj.data.root_pos_w - body_pos_w, dim=1)
    return 1.0 - torch.tanh(distance / std)


def object_body_offset_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
) -> torch.Tensor:
    """Reward proximity between a robot body and an offset point around the object."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_id = robot_body_cfg.body_ids[0]
    body_pos_w = robot.data.body_pos_w[:, body_id, :]
    offset = torch.tensor(object_offset, device=obj.data.root_pos_w.device, dtype=obj.data.root_pos_w.dtype)
    target_pos_w = obj.data.root_pos_w + offset
    distance = torch.norm(target_pos_w - body_pos_w, dim=1)
    return 1.0 - torch.tanh(distance / std)


def gripper_closed_near_object(
    env: ManagerBasedRLEnv,
    std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward closing the gripper only when the right hand is close to the object."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_id = robot_body_cfg.body_ids[0]
    body_pos_w = robot.data.body_pos_w[:, body_id, :]
    offset = torch.tensor(object_offset, device=obj.data.root_pos_w.device, dtype=obj.data.root_pos_w.dtype)
    target_pos_w = obj.data.root_pos_w + offset
    distance = torch.norm(target_pos_w - body_pos_w, dim=1)
    proximity = 1.0 - torch.tanh(distance / std)

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    closed = torch.clamp((open_joint_pos - gripper_pos) / open_joint_pos, min=0.0, max=1.0)
    return proximity * torch.mean(closed, dim=1)


def gripper_closed_near_object_bodies(
    env: ManagerBasedRLEnv,
    std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward closing the gripper when the selected grasp bodies are close to the object."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, robot_body_cfg.body_ids, :]
    offset = torch.tensor(object_offset, device=obj.data.root_pos_w.device, dtype=obj.data.root_pos_w.dtype)
    target_pos_w = obj.data.root_pos_w[:, None, :] + offset
    distance = torch.norm(target_pos_w - body_pos_w, dim=-1)
    proximity = torch.mean(1.0 - torch.tanh(distance / std), dim=1)

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    closed = torch.clamp((open_joint_pos - gripper_pos) / open_joint_pos, min=0.0, max=1.0)
    return proximity * torch.mean(closed, dim=1)


def closed_gripper_lift_target(
    env: ManagerBasedRLEnv,
    std: float,
    open_joint_pos: float,
    lift_offset: float,
    object_offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_ee_tcp"]),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward pulling upward after the gripper is closed near the object."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, robot_body_cfg.body_ids, :]
    offset = torch.tensor(object_offset, device=obj.data.root_pos_w.device, dtype=obj.data.root_pos_w.dtype)
    target_pos_w = obj.data.root_pos_w[:, None, :] + offset
    distance = torch.norm(target_pos_w - body_pos_w, dim=-1)
    proximity = torch.mean(1.0 - torch.tanh(distance / std), dim=1)

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    closed = torch.clamp((open_joint_pos - gripper_pos) / open_joint_pos, min=0.0, max=1.0)
    closed_mean = torch.mean(closed, dim=1)

    body_height = torch.mean(body_pos_w[:, :, 2], dim=1)
    target_height = obj.data.root_pos_w[:, 2] + lift_offset
    height_error = torch.clamp(target_height - body_height, min=0.0)
    height_reward = 1.0 - torch.tanh(height_error / std)
    return proximity * closed_mean * height_reward


def object_bodies_average_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
) -> torch.Tensor:
    """Reward multiple robot bodies for approaching one object-relative target."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, robot_body_cfg.body_ids, :]
    offset = torch.tensor(object_offset, device=obj.data.root_pos_w.device, dtype=obj.data.root_pos_w.dtype)
    target_pos_w = obj.data.root_pos_w[:, None, :] + offset
    distance = torch.norm(target_pos_w - body_pos_w, dim=-1)
    return torch.mean(1.0 - torch.tanh(distance / std), dim=1)


def _closed_grasp_gate(
    env: ManagerBasedRLEnv,
    std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg,
    grasp_body_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[grasp_body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, grasp_body_cfg.body_ids, :]
    offset = torch.tensor(object_offset, device=obj.data.root_pos_w.device, dtype=obj.data.root_pos_w.dtype)
    target_pos_w = obj.data.root_pos_w[:, None, :] + offset
    distance = torch.norm(target_pos_w - body_pos_w, dim=-1)
    proximity = torch.mean(1.0 - torch.tanh(distance / std), dim=1)

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    closed = torch.clamp((open_joint_pos - gripper_pos) / open_joint_pos, min=0.0, max=1.0)
    return proximity * torch.mean(closed, dim=1)


def object_lift_progress_with_grasp(
    env: ManagerBasedRLEnv,
    start_height: float,
    target_height: float,
    grasp_std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward continuous object lift progress while the object is in a closed grasp."""

    obj: RigidObject = env.scene[object_cfg.name]
    height_span = max(target_height - start_height, 1.0e-4)
    progress = torch.clamp((obj.data.root_pos_w[:, 2] - start_height) / height_span, min=0.0, max=1.0)
    grasp_gate = _closed_grasp_gate(env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg)
    return grasp_gate * progress


def body_lift_progress_with_closed_gripper(
    env: ManagerBasedRLEnv,
    start_height: float,
    target_height: float,
    grasp_std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_ee_tcp"]),
    grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward lifting the closed right hand after it is positioned around the object."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    body_z = robot.data.body_pos_w[:, robot_body_cfg.body_ids, 2]
    if body_z.ndim > 1:
        body_z = torch.mean(body_z, dim=1)
    height_span = max(target_height - start_height, 1.0e-4)
    progress = torch.clamp((body_z - start_height) / height_span, min=0.0, max=1.0)
    grasp_gate = _closed_grasp_gate(env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg)
    return grasp_gate * progress


def body_axis_progress_with_closed_gripper(
    env: ManagerBasedRLEnv,
    axis: int,
    start_value_b: float,
    target_value_b: float,
    object_minimal_height: float,
    grasp_std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_ee_tcp"]),
    grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward moving a closed-grasp body along one robot-root axis after the object is lifted."""

    obj: RigidObject = env.scene[object_cfg.name]
    body_pos_b = body_position_in_robot_root_frame(env, robot_body_cfg)

    span = target_value_b - start_value_b
    direction = 1.0 if span >= 0.0 else -1.0
    progress = (body_pos_b[:, axis] - start_value_b) * direction / max(abs(span), 1.0e-4)
    progress = torch.clamp(progress, min=0.0, max=1.0)
    lifted = obj.data.root_pos_w[:, 2] > object_minimal_height
    grasp_gate = _closed_grasp_gate(env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg)
    return lifted.float() * grasp_gate * progress


def object_goal_distance_with_grasp(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    grasp_std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward object-goal distance only while the object is in a stable closed grasp."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance = torch.norm(goal_pos_w - obj.data.root_pos_w, dim=1)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    grasp_gate = _closed_grasp_gate(env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg)
    return lifted.float() * grasp_gate * (1.0 - torch.tanh(distance / std))


def object_to_goal_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return the object-to-goal translation vector in the robot root frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w)
    return command[:, :3] - object_pos_b


def transport_goal_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    carry_height_b: float,
) -> torch.Tensor:
    """Expose the actual goal XY at the carry height used before placement."""

    command = env.command_manager.get_command(command_name).clone()
    command[:, 2] = carry_height_b
    return command


def object_to_transport_goal_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    carry_height_b: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return object translation to the actual goal XY at the carry height."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w)
    goal_pos_b = command[:, :3].clone()
    goal_pos_b[:, 2] = carry_height_b
    return goal_pos_b - object_pos_b


def object_linear_velocity_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return object linear velocity in the robot root frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w)


def object_goal_xy_progress_with_grasp(
    env: ManagerBasedRLEnv,
    start_pos_b: tuple[float, float, float],
    minimal_height: float,
    command_name: str,
    grasp_std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward progress of a grasped object from its start XY position toward the command XY goal."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w)
    start = torch.tensor(start_pos_b, device=obj.data.root_pos_w.device, dtype=obj.data.root_pos_w.dtype)

    initial_distance = torch.norm(command[:, :2] - start[:2], dim=1).clamp_min(1.0e-4)
    current_distance = torch.norm(command[:, :2] - object_pos_b[:, :2], dim=1)
    progress = torch.clamp((initial_distance - current_distance) / initial_distance, min=0.0, max=1.0)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    grasp_gate = _closed_grasp_gate(env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg)
    return lifted.float() * grasp_gate * progress


class object_goal_xy_potential_progress(ManagerTermBase):
    """Reward only step-to-step XY distance reduction while a lifted grasp is active."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_distance = torch.zeros(env.num_envs, device=env.device)
        self._was_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._previous_distance[env_ids] = 0.0
        self._was_active[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        minimal_height: float,
        command_name: str,
        grasp_std: float,
        open_joint_pos: float,
        object_offset: tuple[float, float, float],
        minimum_grasp_gate: float,
        max_step_distance: float,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
            "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
        ),
        gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
    ) -> torch.Tensor:
        robot: Articulation = env.scene[robot_cfg.name]
        obj: RigidObject = env.scene[object_cfg.name]
        command = env.command_manager.get_command(command_name)
        goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
        distance_xy = torch.norm(goal_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
        grasp_gate = _closed_grasp_gate(
            env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg
        )
        active = (obj.data.root_pos_w[:, 2] > minimal_height) & (grasp_gate >= minimum_grasp_gate)

        distance_delta = torch.clamp(
            self._previous_distance - distance_xy,
            min=-max_step_distance,
            max=max_step_distance,
        )
        reward = torch.where(active & self._was_active, distance_delta / env.step_dt, 0.0)
        self._previous_distance[:] = distance_xy
        self._was_active[:] = active
        return reward


class gripper_open_after_grasp_outside_goal_xy(ManagerTermBase):
    """Penalize opening outside the goal after a short, confirmed closed grasp."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._closed_near_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._grasp_acquired = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._closed_near_steps[env_ids] = 0
        self._grasp_acquired[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        xy_threshold: float,
        command_name: str,
        open_joint_pos: float,
        proximity_std: float,
        proximity_threshold: float,
        closed_threshold: float,
        acquisition_steps: int,
        min_progress: float,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
            "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
        ),
        gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
    ) -> torch.Tensor:
        robot: Articulation = env.scene[robot_cfg.name]
        obj: RigidObject = env.scene[object_cfg.name]
        body_pos_w = robot.data.body_pos_w[:, grasp_body_cfg.body_ids, :]
        distance = torch.norm(obj.data.root_pos_w[:, None, :] - body_pos_w, dim=-1)
        proximity = torch.mean(1.0 - torch.tanh(distance / proximity_std), dim=1)

        gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
        closed = torch.mean(
            torch.clamp((open_joint_pos - gripper_pos) / open_joint_pos, min=0.0, max=1.0), dim=1
        )
        closed_near = (proximity >= proximity_threshold) & (closed >= closed_threshold)
        self._closed_near_steps[:] = torch.where(
            closed_near,
            torch.clamp(self._closed_near_steps + 1, max=acquisition_steps),
            0,
        )
        self._grasp_acquired |= self._closed_near_steps >= acquisition_steps

        command = env.command_manager.get_command(command_name)
        goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
        distance_xy = torch.norm(goal_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
        outside_goal_xy = (distance_xy > xy_threshold).float()
        opened = torch.mean(torch.clamp(gripper_pos / open_joint_pos, min=0.0, max=1.0), dim=1)
        progress_gate = _goal_curriculum_gate(env, command_name, min_progress)
        return progress_gate * self._grasp_acquired.float() * outside_goal_xy * opened


class TaskStateTracker(ManagerTermBase):
    """Track observable pick-place milestones and outcome dwell state once per control step."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.grasp_streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.lift_streak = torch.zeros_like(self.grasp_streak)
        self.transport_streak = torch.zeros_like(self.grasp_streak)
        self.center_streak = torch.zeros_like(self.grasp_streak)
        self.ready_streak = torch.zeros_like(self.grasp_streak)
        self.opening_streak = torch.zeros_like(self.grasp_streak)
        self.lost_streak = torch.zeros_like(self.grasp_streak)
        self.basket_exit_streak = torch.zeros_like(self.grasp_streak)
        self.outside_streak = torch.zeros_like(self.grasp_streak)
        self.settle_streak = torch.zeros_like(self.grasp_streak)
        self.grasp_closing_streak = torch.zeros_like(self.grasp_streak)
        self.grasp_contact_streak = torch.zeros_like(self.grasp_streak)
        self.grasp_attempt_lost_streak = torch.zeros_like(self.grasp_streak)
        self.grasp_verification_streak = torch.zeros_like(self.grasp_streak)
        self.grasp_retry_cooldown = torch.zeros_like(self.grasp_streak)
        self.grasp_attempt_aborts = torch.zeros_like(self.grasp_streak)

        self.grasp_ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.grasp_candidate = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.grasp_attempt_active = torch.zeros_like(self.grasp_candidate)
        self.grasp_candidate_tracking = torch.zeros_like(self.grasp_candidate)
        self.grasp_alignment_ready = torch.zeros_like(self.grasp_candidate)
        self.grasp_attempt_continuation = torch.zeros_like(self.grasp_candidate)
        self.grasp_lift_authorized = torch.zeros_like(self.grasp_candidate)
        self.grasp_verification_active = torch.zeros_like(self.grasp_candidate)
        self.grasp_attempt_started = torch.zeros_like(self.grasp_candidate)
        self.grasp_ready_reached = torch.zeros_like(self.grasp_candidate)
        self.grasp_candidate_reached = torch.zeros_like(self.grasp_candidate)
        self.grasp_acquired = torch.zeros_like(self.grasp_candidate)
        self.current_hold = torch.zeros_like(self.grasp_candidate)
        self.lift_completed = torch.zeros_like(self.grasp_candidate)
        self.validated_transport = torch.zeros_like(self.grasp_candidate)
        self.assist_engaged = torch.zeros_like(self.grasp_candidate)
        self.assist_active = torch.zeros_like(self.grasp_candidate)
        self.assist_owns_arm = torch.zeros_like(self.grasp_candidate)
        self.basket_reached = torch.zeros_like(self.grasp_candidate)
        self.inside_basket_entry_xy = torch.zeros_like(self.grasp_candidate)
        self.inside_release_xy = torch.zeros_like(self.grasp_candidate)
        self.inside_release_safe_xy = torch.zeros_like(self.grasp_candidate)
        self.place_centered = torch.zeros_like(self.grasp_candidate)
        self.release_ready = torch.zeros_like(self.grasp_candidate)
        self.release_ready_reached = torch.zeros_like(self.grasp_candidate)
        self.release_commanded = torch.zeros_like(self.grasp_candidate)
        self.gripper_target_open = torch.ones_like(self.grasp_candidate)
        self.released = torch.zeros_like(self.grasp_candidate)
        self.succeeded = torch.zeros_like(self.grasp_candidate)
        self.failed = torch.zeros_like(self.grasp_candidate)

        self.just_grasp_ready = torch.zeros_like(self.grasp_candidate)
        self.just_grasp_candidate = torch.zeros_like(self.grasp_candidate)
        self.just_grasp_retry = torch.zeros_like(self.grasp_candidate)
        self.just_grasped = torch.zeros_like(self.grasp_candidate)
        self.just_lifted = torch.zeros_like(self.grasp_candidate)
        self.just_validated_transport = torch.zeros_like(self.grasp_candidate)
        self.just_assist_engaged = torch.zeros_like(self.grasp_candidate)
        self.just_reached_basket = torch.zeros_like(self.grasp_candidate)
        self.just_place_centered = torch.zeros_like(self.grasp_candidate)
        self.just_release_ready = torch.zeros_like(self.grasp_candidate)
        self.just_release_commanded = torch.zeros_like(self.grasp_candidate)
        self.just_released = torch.zeros_like(self.grasp_candidate)

        self.finger_proximity = torch.zeros(env.num_envs, device=env.device)
        self.bilateral_finger_proximity = torch.zeros_like(self.finger_proximity)
        self.closed_fraction = torch.zeros_like(self.finger_proximity)
        self.open_fraction = torch.zeros_like(self.finger_proximity)
        self.grasp_relative_speed = torch.zeros_like(self.finger_proximity)
        self.finger_distance_asymmetry = torch.zeros_like(self.finger_proximity)
        self.grasp_alignment_quality = torch.zeros_like(self.finger_proximity)
        self.grasp_coupling_error = torch.zeros_like(self.finger_proximity)
        self.grasp_offset_error = torch.zeros_like(self.finger_proximity)
        self.transport_displacement = torch.zeros_like(self.finger_proximity)
        self.distance_xy = torch.zeros_like(self.finger_proximity)
        self.object_speed = torch.zeros_like(self.finger_proximity)
        self.object_xy_speed = torch.zeros_like(self.finger_proximity)
        self.object_vertical_speed = torch.zeros_like(self.finger_proximity)
        self.object_height = torch.zeros_like(self.finger_proximity)
        self.centering_height_error = torch.zeros_like(self.finger_proximity)
        self.release_target_height = torch.zeros_like(self.finger_proximity)
        self.release_height_delta = torch.zeros_like(self.finger_proximity)
        self.release_height_error = torch.zeros_like(self.finger_proximity)
        self.release_curriculum_progress = torch.zeros_like(self.finger_proximity)
        self.grasp_candidate_object_origin = torch.zeros((env.num_envs, 3), device=env.device)
        self.grasp_candidate_tcp_origin = torch.zeros((env.num_envs, 3), device=env.device)
        self.grasp_candidate_tcp_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self.grasp_object_origin = torch.zeros((env.num_envs, 3), device=env.device)
        self.grasp_tcp_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self.object_offset_tcp = torch.zeros((env.num_envs, 3), device=env.device)
        params = cfg.params
        self._proximity_std = float(params["proximity_std"])
        self._candidate_proximity_threshold = float(params["candidate_proximity_threshold"])
        self._attempt_hold_proximity_threshold = float(
            params["attempt_hold_proximity_threshold"]
        )
        self._candidate_closed_threshold = float(params["candidate_closed_threshold"])
        self._candidate_min_open_fraction = float(params["candidate_min_open_fraction"])
        self._candidate_max_relative_speed = float(params["candidate_max_relative_speed"])
        self._alignment_max_relative_speed = float(params["alignment_max_relative_speed"])
        self._preclose_min_open_fraction = float(params["preclose_min_open_fraction"])
        self._alignment_target = torch.tensor(
            params["alignment_target"], device=env.device, dtype=torch.float32
        )
        self._alignment_lower_bounds = torch.tensor(
            params["alignment_lower_bounds"], device=env.device, dtype=torch.float32
        )
        self._alignment_upper_bounds = torch.tensor(
            params["alignment_upper_bounds"], device=env.device, dtype=torch.float32
        )
        self._alignment_hold_lower_bounds = torch.tensor(
            params["alignment_hold_lower_bounds"], device=env.device, dtype=torch.float32
        )
        self._alignment_hold_upper_bounds = torch.tensor(
            params["alignment_hold_upper_bounds"], device=env.device, dtype=torch.float32
        )
        self._alignment_quality_scales = torch.tensor(
            params["alignment_quality_scales"], device=env.device, dtype=torch.float32
        )
        self._alignment_max_finger_asymmetry = float(
            params["alignment_max_finger_asymmetry"]
        )
        self._alignment_hold_max_finger_asymmetry = float(
            params["alignment_hold_max_finger_asymmetry"]
        )
        self._closing_min_steps = int(params["closing_min_steps"])
        self._contact_confirm_steps = int(params["contact_confirm_steps"])
        self._attempt_lost_confirm_steps = int(params["attempt_lost_confirm_steps"])
        self._closing_timeout_steps = int(params["closing_timeout_steps"])
        self._verification_timeout_steps = int(params["verification_timeout_steps"])
        self._retry_cooldown_steps = int(params["retry_cooldown_steps"])
        self._open_joint_pos = float(params["open_joint_pos"])
        self._grasp_body_cfg = params["grasp_body_cfg"]
        self._tcp_body_cfg = params["tcp_body_cfg"]
        self._gripper_cfg = params["gripper_cfg"]
        levels = cfg.params.get("release_height_levels", (0.73, 0.68))
        probabilities = cfg.params.get("release_height_probabilities", (0.40, 0.60))
        if len(levels) == 0 or len(levels) != len(probabilities):
            raise ValueError("release height levels and probabilities must have the same non-zero length")
        if any(probability < 0.0 for probability in probabilities) or sum(probabilities) <= 0.0:
            raise ValueError("release height probabilities must be non-negative with a positive sum")
        self._release_height_levels = torch.tensor(levels, device=env.device, dtype=torch.float32)
        self._release_height_probabilities = torch.tensor(probabilities, device=env.device, dtype=torch.float32)
        self._release_height_probabilities /= torch.sum(self._release_height_probabilities)
        normalization_bounds = cfg.params.get("release_height_normalization_bounds", (0.68, 0.75))
        if len(normalization_bounds) != 2 or normalization_bounds[1] <= normalization_bounds[0]:
            raise ValueError("release height normalization bounds must be an increasing pair")
        if min(levels) < normalization_bounds[0] or max(levels) > normalization_bounds[1]:
            raise ValueError("release height levels must stay inside the fixed normalization bounds")
        self._release_height_min = float(normalization_bounds[0])
        self._release_height_max = float(normalization_bounds[1])
        self._place_center_confirm_steps = int(cfg.params.get("place_center_confirm_steps", 5))
        self._grasp_confirm_steps = int(cfg.params.get("grasp_confirm_steps", 5))
        self._lift_confirm_steps = int(cfg.params.get("lift_confirm_steps", 3))
        self._transport_confirm_steps = int(cfg.params.get("transport_confirm_steps", 3))
        self._release_ready_confirm_steps = int(cfg.params.get("release_ready_confirm_steps", 5))
        self._release_open_grace_steps = int(cfg.params.get("release_open_grace_steps", 10))
        self._lost_confirm_steps = int(cfg.params.get("lost_confirm_steps", 3))
        self._assist_lost_confirm_steps = int(cfg.params.get("assist_lost_confirm_steps", 12))
        self._basket_exit_confirm_steps = int(cfg.params.get("basket_exit_confirm_steps", 6))
        self._released_outside_confirm_steps = int(cfg.params.get("released_outside_confirm_steps", 6))
        self._settle_confirm_steps = int(cfg.params.get("settle_confirm_steps", 10))
        self._sample_release_targets(torch.arange(env.num_envs, device=env.device))
        env._openarm_pick_place_state = self

    def _sample_release_targets(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        level_ids = torch.multinomial(
            self._release_height_probabilities,
            num_samples=env_ids.numel(),
            replacement=True,
        )
        target_height = self._release_height_levels[level_ids]
        self.release_target_height[env_ids] = target_height
        height_span = self._release_height_max - self._release_height_min
        self.release_curriculum_progress[env_ids] = (self._release_height_max - target_height) / height_span

    def request_release(self, request_mask: torch.Tensor):
        """Latch only a release request that passed the current physical readiness guard."""

        self.release_commanded |= request_mask & ~self.release_commanded

    def update_grasp_geometry(self, env: ManagerBasedRLEnv):
        """Refresh measured grasp geometry for both the action and termination phases."""

        robot: Articulation = env.scene[self._grasp_body_cfg.name]
        obj: RigidObject = env.scene["object"]
        finger_pos_w = robot.data.body_pos_w[:, self._grasp_body_cfg.body_ids, :]
        finger_distance = torch.norm(obj.data.root_pos_w[:, None, :] - finger_pos_w, dim=-1)
        per_finger_proximity = 1.0 - torch.tanh(finger_distance / self._proximity_std)
        self.finger_proximity[:] = torch.mean(per_finger_proximity, dim=1)
        self.bilateral_finger_proximity[:] = torch.amin(per_finger_proximity, dim=1)
        self.finger_distance_asymmetry[:] = torch.amax(finger_distance, dim=1) - torch.amin(
            finger_distance, dim=1
        )

        gripper_pos = robot.data.joint_pos[:, self._gripper_cfg.joint_ids]
        self.closed_fraction[:] = torch.mean(
            torch.clamp(
                (self._open_joint_pos - gripper_pos) / self._open_joint_pos,
                min=0.0,
                max=1.0,
            ),
            dim=1,
        )
        self.open_fraction[:] = torch.mean(
            torch.clamp(gripper_pos / self._open_joint_pos, min=0.0, max=1.0), dim=1
        )

        tcp_body_id = self._tcp_body_cfg.body_ids[0]
        tcp_pos_w = robot.data.body_pos_w[:, tcp_body_id]
        tcp_quat_w = robot.data.body_quat_w[:, tcp_body_id]
        object_offset_w = obj.data.root_pos_w - tcp_pos_w
        self.object_offset_tcp[:] = quat_apply_inverse(tcp_quat_w, object_offset_w)
        tcp_velocity_at_object_w = robot.data.body_link_lin_vel_w[:, tcp_body_id] + torch.cross(
            robot.data.body_link_ang_vel_w[:, tcp_body_id], object_offset_w, dim=1
        )
        self.grasp_relative_speed[:] = torch.norm(
            obj.data.root_link_lin_vel_w - tcp_velocity_at_object_w, dim=1
        )

        alignment_error = (
            self.object_offset_tcp - self._alignment_target
        ) / self._alignment_quality_scales
        pose_quality = torch.exp(-0.5 * torch.sum(torch.square(alignment_error), dim=1))
        symmetry_quality = torch.exp(
            -0.5 * torch.square(self.finger_distance_asymmetry / 0.012)
        )
        speed_quality = torch.exp(
            -0.5
            * torch.square(
                self.grasp_relative_speed / self._alignment_max_relative_speed
            )
        )
        proximity_quality = torch.clamp(
            self.bilateral_finger_proximity / self._candidate_proximity_threshold,
            min=0.0,
            max=1.0,
        )
        self.grasp_alignment_quality[:] = (
            pose_quality
            * symmetry_quality
            * speed_quality
            * (0.5 + 0.5 * proximity_quality)
        )

        inside_alignment = torch.all(
            (self.object_offset_tcp >= self._alignment_lower_bounds)
            & (self.object_offset_tcp <= self._alignment_upper_bounds),
            dim=1,
        )
        self.grasp_alignment_ready[:] = (
            inside_alignment
            & (self.bilateral_finger_proximity >= self._candidate_proximity_threshold)
            & (self.finger_distance_asymmetry <= self._alignment_max_finger_asymmetry)
            & (self.open_fraction >= self._preclose_min_open_fraction)
            & (self.grasp_relative_speed <= self._alignment_max_relative_speed)
        )
        inside_hold_alignment = torch.all(
            (self.object_offset_tcp >= self._alignment_hold_lower_bounds)
            & (self.object_offset_tcp <= self._alignment_hold_upper_bounds),
            dim=1,
        )
        self.grasp_attempt_continuation[:] = (
            inside_hold_alignment
            & (self.bilateral_finger_proximity >= self._attempt_hold_proximity_threshold)
            & (
                self.finger_distance_asymmetry
                <= self._alignment_hold_max_finger_asymmetry
            )
        )
        self.grasp_ready[:] = self.grasp_alignment_ready

    def try_start_grasp_attempt(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Latch a centered grasp before the actor can move the arm away on this control step."""

        self.update_grasp_geometry(env)
        new_attempt = (
            self.grasp_alignment_ready
            & ~self.grasp_attempt_active
            & ~self.grasp_acquired
            & ~self.failed
            & (self.grasp_retry_cooldown == 0)
        )
        self.grasp_attempt_active[new_attempt] = True
        self.grasp_attempt_started[new_attempt] = True
        self.grasp_lift_authorized[new_attempt] = False
        self.grasp_candidate_tracking[new_attempt] = False
        self.grasp_closing_streak[new_attempt] = 0
        self.grasp_contact_streak[new_attempt] = 0
        self.grasp_attempt_lost_streak[new_attempt] = 0
        self.grasp_verification_streak[new_attempt] = 0
        return new_attempt

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            resolved_env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            resolved_env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            resolved_env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        for name, value in vars(self).items():
            if not name.startswith("_") and isinstance(value, torch.Tensor) and value.shape[:1] == (self.num_envs,):
                value[resolved_env_ids] = 0
        self.gripper_target_open[resolved_env_ids] = True
        self._sample_release_targets(resolved_env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_start_height: float,
        carry_height: float,
        release_height_levels: tuple[float, ...],
        release_height_probabilities: tuple[float, ...],
        release_height_normalization_bounds: tuple[float, float],
        settle_height: float,
        centering_height: float,
        assist_entry_xy_threshold: float,
        basket_entry_xy_threshold: float,
        place_center_xy_threshold: float,
        place_center_height_tolerance: float,
        place_center_max_xy_speed: float,
        release_xy_threshold: float,
        success_xy_threshold: float,
        basket_exit_xy_threshold: float,
        release_height_tolerance: float,
        settle_height_tolerance: float,
        release_max_speed: float,
        settle_max_speed: float,
        proximity_std: float,
        candidate_proximity_threshold: float,
        attempt_hold_proximity_threshold: float,
        hold_proximity_threshold: float,
        candidate_closed_threshold: float,
        hold_closed_threshold: float,
        grasp_confirm_steps: int,
        lift_confirm_steps: int,
        transport_confirm_steps: int,
        candidate_min_open_fraction: float,
        hold_min_open_fraction: float,
        alignment_target: tuple[float, float, float],
        alignment_lower_bounds: tuple[float, float, float],
        alignment_upper_bounds: tuple[float, float, float],
        alignment_hold_lower_bounds: tuple[float, float, float],
        alignment_hold_upper_bounds: tuple[float, float, float],
        alignment_quality_scales: tuple[float, float, float],
        alignment_max_finger_asymmetry: float,
        alignment_hold_max_finger_asymmetry: float,
        alignment_max_relative_speed: float,
        preclose_min_open_fraction: float,
        closing_min_steps: int,
        contact_confirm_steps: int,
        attempt_lost_confirm_steps: int,
        closing_timeout_steps: int,
        verification_timeout_steps: int,
        retry_cooldown_steps: int,
        max_grasp_attempts: int,
        coupling_object_lift: float,
        coupling_tcp_lift: float,
        coupling_max_displacement_error: float,
        candidate_max_relative_speed: float,
        hold_max_offset_error: float,
        hold_max_relative_speed: float,
        transport_min_displacement: float,
        place_center_confirm_steps: int,
        release_ready_confirm_steps: int,
        release_open_grace_steps: int,
        lost_confirm_steps: int,
        assist_lost_confirm_steps: int,
        basket_exit_confirm_steps: int,
        released_outside_confirm_steps: int,
        settle_confirm_steps: int,
        open_joint_pos: float,
        command_name: str,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
            "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
        ),
        tcp_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_ee_tcp"]),
        gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
    ) -> torch.Tensor:
        robot: Articulation = env.scene[robot_cfg.name]
        obj: RigidObject = env.scene[object_cfg.name]
        self.update_grasp_geometry(env)
        self.grasp_retry_cooldown[:] = torch.clamp(self.grasp_retry_cooldown - 1, min=0)

        attempt_active = self.grasp_attempt_active & ~self.grasp_acquired
        closing = attempt_active & ~self.grasp_lift_authorized & ~self.gripper_target_open
        self.grasp_closing_streak[:] = torch.where(
            closing,
            torch.clamp(self.grasp_closing_streak + 1, max=closing_timeout_steps),
            torch.where(self.grasp_lift_authorized, self.grasp_closing_streak, 0),
        )
        contact_geometry = (
            closing
            & self.grasp_attempt_continuation
            & (self.bilateral_finger_proximity >= candidate_proximity_threshold)
            & (self.closed_fraction >= candidate_closed_threshold)
            & (self.open_fraction >= candidate_min_open_fraction)
            & (self.grasp_relative_speed <= candidate_max_relative_speed)
            & (self.grasp_closing_streak >= closing_min_steps)
        )
        self.grasp_contact_streak[:] = torch.where(
            contact_geometry,
            torch.clamp(self.grasp_contact_streak + 1, max=contact_confirm_steps),
            0,
        )
        was_lift_authorized = self.grasp_lift_authorized.clone()
        newly_authorized = (
            (self.grasp_contact_streak >= contact_confirm_steps)
            & attempt_active
            & ~self.grasp_lift_authorized
        )
        self.grasp_lift_authorized |= newly_authorized

        tcp_body_id = tcp_body_cfg.body_ids[0]
        tcp_pos_w = robot.data.body_pos_w[:, tcp_body_id]
        self.grasp_candidate_object_origin[newly_authorized] = obj.data.root_pos_w[
            newly_authorized
        ]
        self.grasp_candidate_tcp_origin[newly_authorized] = tcp_pos_w[newly_authorized]
        self.grasp_candidate_tcp_offset[newly_authorized] = self.object_offset_tcp[
            newly_authorized
        ]
        verifying = attempt_active & self.grasp_lift_authorized
        verified_contact_geometry = (
            self.grasp_attempt_continuation
            & (self.closed_fraction >= candidate_closed_threshold)
            & (self.open_fraction >= candidate_min_open_fraction)
        )
        self.grasp_verification_streak[:] = torch.where(
            attempt_active & was_lift_authorized,
            torch.clamp(
                self.grasp_verification_streak + 1,
                max=verification_timeout_steps + 1,
            ),
            0,
        )
        lost_attempt_geometry = attempt_active & (
            ~self.grasp_attempt_continuation
            | (self.grasp_lift_authorized & ~verified_contact_geometry)
        )
        self.grasp_attempt_lost_streak[:] = torch.where(
            lost_attempt_geometry,
            torch.clamp(
                self.grasp_attempt_lost_streak + 1, max=attempt_lost_confirm_steps
            ),
            0,
        )
        close_timed_out = (
            closing
            & ~newly_authorized
            & (self.grasp_closing_streak >= closing_timeout_steps)
        )
        verification_timed_out = verifying & (
            self.grasp_verification_streak > verification_timeout_steps
        )
        attempt_aborted = attempt_active & (
            (self.grasp_attempt_lost_streak >= attempt_lost_confirm_steps)
            | close_timed_out
            | verification_timed_out
        )
        self.just_grasp_retry[:] = attempt_aborted
        self.grasp_attempt_aborts += attempt_aborted.long()
        self.grasp_attempt_active[attempt_aborted] = False
        self.grasp_lift_authorized[attempt_aborted] = False
        self.grasp_verification_active[attempt_aborted] = False
        self.grasp_candidate_tracking[attempt_aborted] = False
        self.grasp_retry_cooldown[attempt_aborted] = retry_cooldown_steps
        self.grasp_closing_streak[attempt_aborted] = 0
        self.grasp_contact_streak[attempt_aborted] = 0
        self.grasp_attempt_lost_streak[attempt_aborted] = 0
        self.grasp_verification_streak[attempt_aborted] = 0

        self.grasp_candidate[:] = (
            self.grasp_attempt_active
            & self.grasp_lift_authorized
            & verified_contact_geometry
            & ~attempt_aborted
        )
        attempt_started = self.grasp_attempt_started.clone()
        self.grasp_attempt_started[:] = False
        previous_grasp_ready = self.grasp_ready_reached.clone()
        self.grasp_ready_reached |= self.grasp_ready & ~self.grasp_acquired
        self.grasp_ready_reached |= attempt_started
        self.just_grasp_ready[:] = self.grasp_ready_reached & ~previous_grasp_ready
        previous_grasp_candidate = self.grasp_candidate_reached.clone()
        self.grasp_candidate_reached |= self.grasp_candidate & ~self.grasp_acquired
        self.just_grasp_candidate[:] = self.grasp_candidate_reached & ~previous_grasp_candidate

        command = env.command_manager.get_command(command_name)
        goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
        self.distance_xy[:] = torch.norm(goal_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
        self.object_speed[:] = torch.norm(obj.data.root_lin_vel_w, dim=1)
        self.object_xy_speed[:] = torch.norm(obj.data.root_lin_vel_w[:, :2], dim=1)
        self.object_vertical_speed[:] = torch.abs(obj.data.root_lin_vel_w[:, 2])
        self.object_height[:] = obj.data.root_pos_w[:, 2]
        self.centering_height_error[:] = torch.abs(obj.data.root_pos_w[:, 2] - centering_height)
        self.inside_basket_entry_xy[:] = self.distance_xy <= basket_entry_xy_threshold
        self.inside_release_xy[:] = self.distance_xy <= release_xy_threshold
        self.inside_release_safe_xy[:] = self.distance_xy <= success_xy_threshold

        candidate_active = self.grasp_candidate & ~self.grasp_acquired
        object_candidate_delta = obj.data.root_pos_w - self.grasp_candidate_object_origin
        tcp_candidate_delta = tcp_pos_w - self.grasp_candidate_tcp_origin
        coupling_error = torch.norm(
            self.object_offset_tcp - self.grasp_candidate_tcp_offset, dim=1
        )
        unstable_candidate = candidate_active & (
            (coupling_error > coupling_max_displacement_error)
            | (self.grasp_relative_speed > candidate_max_relative_speed)
        )
        self.grasp_candidate_tracking[:] = candidate_active & ~unstable_candidate
        self.grasp_coupling_error[:] = torch.where(
            candidate_active,
            coupling_error,
            torch.zeros_like(coupling_error),
        )
        grasp_confirmed = (
            self.grasp_candidate_tracking
            & (object_candidate_delta[:, 2] >= coupling_object_lift)
            & (tcp_candidate_delta[:, 2] >= coupling_tcp_lift)
            & (self.grasp_coupling_error <= coupling_max_displacement_error)
            & (self.grasp_relative_speed <= candidate_max_relative_speed)
        )
        self.grasp_streak[:] = torch.where(
            grasp_confirmed,
            torch.clamp(self.grasp_streak + 1, max=grasp_confirm_steps),
            0,
        )
        previous_grasp = self.grasp_acquired.clone()
        self.grasp_acquired |= self.grasp_streak >= grasp_confirm_steps
        self.just_grasped[:] = self.grasp_acquired & ~previous_grasp
        self.grasp_tcp_offset[self.just_grasped] = self.object_offset_tcp[self.just_grasped]
        self.grasp_object_origin[self.just_grasped] = obj.data.root_pos_w[self.just_grasped]
        self.grasp_attempt_active[self.grasp_acquired] = False
        self.grasp_lift_authorized[self.grasp_acquired] = False
        self.grasp_verification_active[self.grasp_acquired] = False
        self.grasp_candidate_tracking[self.grasp_acquired] = False

        offset_error = torch.norm(self.object_offset_tcp - self.grasp_tcp_offset, dim=1)
        self.grasp_offset_error[:] = torch.where(
            self.grasp_acquired,
            offset_error,
            torch.zeros_like(offset_error),
        )
        hold_geometry = (
            ~self.gripper_target_open
            & (self.bilateral_finger_proximity >= hold_proximity_threshold)
            & (self.closed_fraction >= hold_closed_threshold)
            & (self.open_fraction >= hold_min_open_fraction)
        )
        self.current_hold[:] = (
            hold_geometry
            & self.grasp_acquired
            & (self.grasp_offset_error <= hold_max_offset_error)
            & (self.grasp_relative_speed <= hold_max_relative_speed)
        )

        previous_lift = self.lift_completed.clone()
        lift_confirmed = self.grasp_acquired & self.current_hold & (obj.data.root_pos_w[:, 2] >= carry_height)
        self.lift_streak[:] = torch.where(
            lift_confirmed,
            torch.clamp(self.lift_streak + 1, max=lift_confirm_steps),
            0,
        )
        self.lift_completed |= self.lift_streak >= lift_confirm_steps
        self.just_lifted[:] = self.lift_completed & ~previous_lift

        transport_displacement = torch.norm(
            obj.data.root_pos_w[:, :2] - self.grasp_object_origin[:, :2], dim=1
        )
        self.transport_displacement[:] = torch.where(
            self.grasp_acquired,
            transport_displacement,
            torch.zeros_like(transport_displacement),
        )
        transport_confirmed = (
            self.grasp_acquired
            & self.current_hold
            & self.lift_completed
            & (self.transport_displacement >= transport_min_displacement)
        )
        self.transport_streak[:] = torch.where(
            transport_confirmed,
            torch.clamp(self.transport_streak + 1, max=transport_confirm_steps),
            0,
        )
        previous_transport = self.validated_transport.clone()
        self.validated_transport |= self.transport_streak >= transport_confirm_steps
        self.just_validated_transport[:] = self.validated_transport & ~previous_transport

        previous_assist = self.assist_engaged.clone()
        self.assist_engaged |= (
            self.grasp_acquired
            & self.current_hold
            & self.lift_completed
            & self.validated_transport
            & (self.distance_xy <= assist_entry_xy_threshold)
        )
        self.just_assist_engaged[:] = self.assist_engaged & ~previous_assist

        previous_basket = self.basket_reached.clone()
        self.basket_reached |= (
            self.grasp_acquired
            & self.current_hold
            & self.lift_completed
            & self.validated_transport
            & self.inside_basket_entry_xy
        )
        self.just_reached_basket[:] = self.basket_reached & ~previous_basket

        centered = (
            self.basket_reached
            & self.current_hold
            & (self.distance_xy <= place_center_xy_threshold)
            & (self.centering_height_error <= place_center_height_tolerance)
            & (self.object_xy_speed <= place_center_max_xy_speed)
        )
        self.center_streak[:] = torch.where(
            centered,
            torch.clamp(self.center_streak + 1, max=place_center_confirm_steps),
            0,
        )
        previous_centered = self.place_centered.clone()
        self.place_centered |= self.center_streak >= place_center_confirm_steps
        self.just_place_centered[:] = self.place_centered & ~previous_centered

        self.release_height_delta[:] = self.release_target_height - obj.data.root_pos_w[:, 2]
        self.release_height_error[:] = torch.abs(self.release_height_delta)
        self.release_ready[:] = (
            self.grasp_acquired
            & self.current_hold
            & self.lift_completed
            & self.validated_transport
            & self.place_centered
            & self.inside_release_xy
            & (self.release_height_error <= release_height_tolerance)
            & (self.object_speed <= release_max_speed)
        )
        self.ready_streak[:] = torch.where(
            self.release_ready,
            torch.clamp(self.ready_streak + 1, max=release_ready_confirm_steps),
            0,
        )
        previous_release_ready = self.release_ready_reached.clone()
        self.release_ready_reached |= self.ready_streak >= release_ready_confirm_steps
        self.just_release_ready[:] = self.release_ready_reached & ~previous_release_ready

        self.just_release_commanded[:] = (
            self.release_commanded & ~self.released & (self.opening_streak == 0)
        )
        previous_released = self.released.clone()
        self.released |= (
            self.grasp_acquired
            & self.lift_completed
            & self.validated_transport
            & self.release_ready_reached
            & self.release_commanded
            & (self.open_fraction >= 0.70)
        )
        self.just_released[:] = self.released & ~previous_released
        opening = self.release_commanded & ~self.released
        release_open_timeout_steps = release_open_grace_steps + lost_confirm_steps
        self.opening_streak[:] = torch.where(
            opening,
            torch.clamp(self.opening_streak + 1, max=release_open_timeout_steps),
            0,
        )

        premature_release = self.grasp_acquired & ~self.release_commanded & (self.open_fraction >= 0.70)
        opening_grace = opening & (self.opening_streak <= release_open_grace_steps)
        lost_hold = self.grasp_acquired & ~self.released & ~self.current_hold & ~opening_grace
        lost_confirm_limit = torch.where(
            self.assist_engaged,
            torch.full_like(self.lost_streak, assist_lost_confirm_steps),
            torch.full_like(self.lost_streak, lost_confirm_steps),
        )
        self.lost_streak[:] = torch.where(
            lost_hold,
            torch.clamp(
                self.lost_streak + 1,
                max=max(lost_confirm_steps, assist_lost_confirm_steps),
            ),
            0,
        )
        exited_after_assist = (
            self.assist_engaged
            & ~self.release_commanded
            & ~self.released
            & (self.distance_xy > basket_exit_xy_threshold)
        )
        self.basket_exit_streak[:] = torch.where(
            exited_after_assist,
            torch.clamp(self.basket_exit_streak + 1, max=basket_exit_confirm_steps),
            0,
        )
        released_outside = self.released & (self.distance_xy > success_xy_threshold)
        self.outside_streak[:] = torch.where(
            released_outside,
            torch.clamp(self.outside_streak + 1, max=released_outside_confirm_steps),
            0,
        )
        self.failed[:] = (
            premature_release
            | (self.grasp_attempt_aborts >= max_grasp_attempts)
            | (opening & (self.opening_streak >= release_open_timeout_steps))
            | (self.lost_streak >= lost_confirm_limit)
            | (self.basket_exit_streak >= basket_exit_confirm_steps)
            | (self.outside_streak >= released_outside_confirm_steps)
            | (obj.data.root_pos_w[:, 2] < object_start_height - 0.03)
        )

        settle_candidate = (
            self.grasp_acquired
            & self.lift_completed
            & self.validated_transport
            & self.basket_reached
            & self.place_centered
            & self.release_ready_reached
            & self.released
            & (self.distance_xy <= success_xy_threshold)
            & (torch.abs(obj.data.root_pos_w[:, 2] - settle_height) <= settle_height_tolerance)
            & (self.object_speed <= settle_max_speed)
            & (self.open_fraction >= 0.70)
        )
        self.settle_streak[:] = torch.where(
            settle_candidate,
            torch.clamp(self.settle_streak + 1, max=settle_confirm_steps),
            0,
        )
        self.succeeded[:] = (self.settle_streak >= settle_confirm_steps) & ~self.failed
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def task_state_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Expose the reward/termination phase so the policy sees a Markov state."""

    state: TaskStateTracker | None = getattr(env, "_openarm_pick_place_state", None)
    if state is None:
        return torch.zeros((env.num_envs, 6), device=env.device)
    return torch.stack(
        (
            state.grasp_acquired,
            state.current_hold,
            state.lift_completed,
            state.inside_release_xy,
            state.release_ready,
            state.released,
        ),
        dim=1,
    ).float()


def task_grasp_quality_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Expose normalized TCP-local object alignment; finger joints are already observed."""

    state: TaskStateTracker | None = getattr(env, "_openarm_pick_place_state", None)
    if state is None:
        return torch.zeros((env.num_envs, 3), device=env.device)
    state.update_grasp_geometry(env)
    return torch.clamp(
        (state.object_offset_tcp - state._alignment_target)
        / state._alignment_quality_scales,
        min=-3.0,
        max=3.0,
    )


def episode_progress_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return normalized episode time as a single policy observation."""

    return (env.episode_length_buf.float() / env.max_episode_length).unsqueeze(1)


def task_release_target_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Expose the goal-conditioned release state, including all task-relevant dwell counters."""

    state: TaskStateTracker | None = getattr(env, "_openarm_pick_place_state", None)
    if state is None:
        return torch.zeros((env.num_envs, 19), device=env.device)
    obj: RigidObject = env.scene["object"]
    release_height_delta = state.release_target_height - obj.data.root_pos_w[:, 2]
    lost_confirm_denominator = torch.where(
        state.assist_engaged,
        torch.full_like(state.lost_streak, state._assist_lost_confirm_steps),
        torch.full_like(state.lost_streak, state._lost_confirm_steps),
    ).float()
    return torch.stack(
        (
            release_height_delta,
            state.release_curriculum_progress,
            state.release_ready_reached,
            state.grasp_ready,
            state.grasp_ready_reached,
            state.grasp_candidate_reached,
            state.basket_reached,
            state.ready_streak.float() / max(state._release_ready_confirm_steps, 1),
            state.lost_streak.float() / torch.clamp(lost_confirm_denominator, min=1.0),
            state.settle_streak.float() / max(state._settle_confirm_steps, 1),
            state.grasp_streak.float() / max(state._grasp_confirm_steps, 1),
            state.release_commanded,
            state.opening_streak.float() / max(state._release_open_grace_steps, 1),
            state.basket_exit_streak.float() / max(state._basket_exit_confirm_steps, 1),
            state.outside_streak.float() / max(state._released_outside_confirm_steps, 1),
            state.gripper_target_open,
            state.place_centered,
            state.center_streak.float() / max(state._place_center_confirm_steps, 1),
            state.assist_engaged,
        ),
        dim=1,
    ).float()


def task_state_failed(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env._openarm_pick_place_state.failed


def task_state_succeeded(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env._openarm_pick_place_state.succeeded


def task_event_reward(env: ManagerBasedRLEnv, event_name: str) -> torch.Tensor:
    """Return a one-step task event with dt removed so configured weights are actual bonuses."""

    state: TaskStateTracker = env._openarm_pick_place_state
    return getattr(state, event_name).float() / env.step_dt


def task_gripper_action_phase_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Directly align the continuous gripper action with the observable task phase."""

    state: TaskStateTracker = env._openarm_pick_place_state
    approach_far = ~state.grasp_acquired & ~state.grasp_attempt_active
    safe_release_request = state.release_ready_reached & state.release_ready
    should_open = approach_far | safe_release_request | state.release_commanded | state.released
    desired_action = torch.where(should_open, 1.0, -1.0)
    gripper_action = env.action_manager.get_term("gripper_action").raw_actions[:, 0]
    return 0.25 * torch.square(gripper_action - desired_action)


class task_potential_progress(ManagerTermBase):
    """Reward new best progress and regressions without paying for return cycles."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_potential = torch.zeros(env.num_envs, device=env.device)
        self._best_potential = torch.zeros(env.num_envs, device=env.device)
        self._was_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._previous_potential[env_ids] = 0.0
        self._best_potential[env_ids] = 0.0
        self._was_active[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        mode: str,
        max_step_change: float,
        target_height: float = 0.75,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        state: TaskStateTracker = env._openarm_pick_place_state
        obj: RigidObject = env.scene[object_cfg.name]
        if mode == "approach":
            potential = state.grasp_alignment_quality
            active = (
                ~state.grasp_acquired
                & ~state.grasp_attempt_active
                & (state.grasp_retry_cooldown == 0)
            )
        elif mode == "lift":
            potential = obj.data.root_pos_w[:, 2]
            active = state.grasp_acquired & ~state.lift_completed & ~state.released
        elif mode == "transport":
            potential = -state.distance_xy
            active = (
                state.grasp_acquired
                & state.current_hold
                & state.lift_completed
                & ~state.assist_active
                & ~state.released
            )
        elif mode == "carry_height":
            potential = -torch.abs(obj.data.root_pos_w[:, 2] - target_height)
            active = (
                state.grasp_acquired
                & state.current_hold
                & state.lift_completed
                & ~state.assist_active
                & ~state.released
            )
        else:
            raise ValueError(f"Unsupported task potential mode: {mode}")

        improvement = torch.clamp(
            potential - self._best_potential,
            min=0.0,
            max=max_step_change,
        )
        regression = torch.clamp(
            potential - self._previous_potential,
            min=-max_step_change,
            max=0.0,
        )
        continued = active & self._was_active
        reward = torch.where(continued, (improvement + regression) / env.step_dt, 0.0)
        active_best = torch.where(
            self._was_active,
            torch.maximum(self._best_potential, potential),
            potential,
        )
        self._best_potential[:] = torch.where(active, active_best, potential)
        self._previous_potential[:] = potential
        self._was_active[:] = active
        return reward


class ramp_reward_weight(ManagerTermBase):
    """Linearly change one reward weight between two common-step boundaries."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._term_cfg = env.reward_manager.get_term_cfg(cfg.params["term_name"])
        self._initial_weight = float(self._term_cfg.weight)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        term_name: str,
        weight: float,
        start_num_steps: int,
        end_num_steps: int,
    ) -> float:
        if end_num_steps <= start_num_steps:
            raise ValueError("end_num_steps must be greater than start_num_steps")
        progress = min(
            max((env.common_step_counter - start_num_steps) / (end_num_steps - start_num_steps), 0.0),
            1.0,
        )
        self._term_cfg.weight = self._initial_weight + progress * (weight - self._initial_weight)
        env.reward_manager.set_term_cfg(term_name, self._term_cfg)
        return self._term_cfg.weight


def constant_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def task_timeout_without_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a one-shot incomplete-episode event with dt removed."""

    state: TaskStateTracker = env._openarm_pick_place_state
    timed_out = env.episode_length_buf >= env.max_episode_length
    incomplete = timed_out & ~state.succeeded & ~state.failed
    return incomplete.float() / env.step_dt


def task_carry_height_band_error(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    maximum_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize carry height outside a safe band only while the learned arm controls transport."""

    state: TaskStateTracker = env._openarm_pick_place_state
    obj: RigidObject = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2]
    band_error = torch.clamp(minimum_height - height, min=0.0) + torch.clamp(
        height - maximum_height, min=0.0
    )
    active = state.grasp_acquired & state.current_hold & state.lift_completed & ~state.assist_active
    return active.float() * band_error


def _actor_arm_control_gate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return one only when the learned arm, rather than place assist, owns the target."""

    state: TaskStateTracker = env._openarm_pick_place_state
    return (~state.assist_owns_arm).float()


def task_actor_action_rate_l2(
    env: ManagerBasedRLEnv,
    max_rate: float,
) -> torch.Tensor:
    """Penalize arm-action changes only in proportion to the actor's arm control authority."""

    delta = env.action_manager.action - env.action_manager.prev_action
    delta = torch.clamp(delta, min=-max_rate, max=max_rate)
    arm_rate = torch.sum(torch.square(delta[:, :-1]), dim=1)
    return _actor_arm_control_gate(env) * arm_rate


def task_actor_body_clearance(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    robot_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Apply table clearance only while the learned arm target has control authority."""

    penalty = body_height_below_minimum(env, minimum_height, robot_body_cfg)
    return _actor_arm_control_gate(env) * penalty


def task_actor_joint_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Apply velocity regularization in proportion to learned arm control authority."""

    robot: Articulation = env.scene[asset_cfg.name]
    joint_velocity = torch.sum(torch.square(robot.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return _actor_arm_control_gate(env) * joint_velocity


def object_goal_axis_progress_with_grasp(
    env: ManagerBasedRLEnv,
    axis: int,
    start_value_b: float,
    minimal_height: float,
    command_name: str,
    grasp_std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward single-axis transport progress from a known start coordinate to the command goal."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w)

    target = command[:, axis]
    start = torch.full_like(target, start_value_b)
    span = target - start
    direction = torch.sign(span)
    progress = (object_pos_b[:, axis] - start) * direction / torch.abs(span).clamp_min(1.0e-4)
    progress = torch.clamp(progress, min=0.0, max=1.0)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    grasp_gate = _closed_grasp_gate(env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg)
    return lifted.float() * grasp_gate * progress


def object_goal_axis_shortfall_with_grasp(
    env: ManagerBasedRLEnv,
    axis: int,
    start_value_b: float,
    minimal_height: float,
    command_name: str,
    grasp_std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Penalty signal for stopping short on one transport axis after a stable lift."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w)

    target = command[:, axis]
    start = torch.full_like(target, start_value_b)
    span = target - start
    direction = torch.sign(span)
    progress = (object_pos_b[:, axis] - start) * direction / torch.abs(span).clamp_min(1.0e-4)
    shortfall = 1.0 - torch.clamp(progress, min=0.0, max=1.0)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    grasp_gate = _closed_grasp_gate(env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg)
    return lifted.float() * grasp_gate * shortfall


def object_goal_axis_distance_with_grasp(
    env: ManagerBasedRLEnv,
    axis: int,
    std: float,
    minimal_height: float,
    command_name: str,
    grasp_std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward one coordinate of a grasped object independently from the full 3D goal distance."""

    obj: RigidObject = env.scene[object_cfg.name]
    delta_b = object_to_goal_position_in_robot_root_frame(env, command_name, robot_cfg, object_cfg)
    axis_error = torch.abs(delta_b[:, axis])
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    grasp_gate = _closed_grasp_gate(env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg)
    return lifted.float() * grasp_gate * (1.0 - torch.tanh(axis_error / std))


def object_goal_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    minimal_height: float,
    command_name: str,
    grasp_std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward a grasped object smoothly inside a loose goal basin while it is lifted."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance = torch.norm(goal_pos_w - obj.data.root_pos_w, dim=1)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    basin_reward = torch.clamp((threshold - distance) / threshold, min=0.0, max=1.0)
    grasp_gate = _closed_grasp_gate(env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg)
    return lifted.float() * grasp_gate * basin_reward


def object_goal_xy_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    minimal_height: float,
    command_name: str,
    grasp_std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward entering an XY basin around the goal while carrying the object."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance_xy = torch.norm(goal_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    basin_reward = torch.clamp((threshold - distance_xy) / threshold, min=0.0, max=1.0)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    grasp_gate = _closed_grasp_gate(env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg)
    return lifted.float() * grasp_gate * basin_reward


def object_goal_xy_shortfall(
    env: ManagerBasedRLEnv,
    threshold: float,
    max_distance: float,
    minimal_height: float,
    command_name: str,
    grasp_std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Penalty signal for remaining outside the target XY basin while carrying the object."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance_xy = torch.norm(goal_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    shortfall = torch.clamp((distance_xy - threshold) / max(max_distance - threshold, 1.0e-4), min=0.0, max=1.0)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    grasp_gate = _closed_grasp_gate(env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg)
    return lifted.float() * grasp_gate * shortfall


def object_goal_release_readiness(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_threshold: float,
    minimal_height: float,
    command_name: str,
    velocity_std: float,
    min_progress: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward reaching a slow, height-aligned release state regardless of gripper state."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance_xy = torch.norm(goal_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    height_error = torch.abs(goal_pos_w[:, 2] - obj.data.root_pos_w[:, 2])
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    near_xy = torch.clamp((xy_threshold - distance_xy) / xy_threshold, min=0.0, max=1.0)
    near_z = torch.clamp((z_threshold - height_error) / z_threshold, min=0.0, max=1.0)
    object_speed = torch.norm(obj.data.root_lin_vel_w, dim=1)
    still_reward = 1.0 - torch.tanh(object_speed / velocity_std)
    progress_gate = _goal_curriculum_gate(env, command_name, min_progress)
    return progress_gate * lifted.float() * near_xy * near_z * still_reward


def _goal_curriculum_gate(env: ManagerBasedRLEnv, command_name: str, min_progress: float) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    progress = command_term.metrics["goal_curriculum_progress"]
    return (progress >= min_progress).float()


def gripper_open_near_goal(
    env: ManagerBasedRLEnv,
    threshold: float,
    minimal_height: float,
    command_name: str,
    open_joint_pos: float,
    velocity_std: float,
    min_progress: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward opening the gripper only when the lifted object is near the goal."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance = torch.norm(goal_pos_w - obj.data.root_pos_w, dim=1)
    near_goal = torch.clamp((threshold - distance) / threshold, min=0.0, max=1.0)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    object_speed = torch.norm(obj.data.root_lin_vel_w, dim=1)
    still_reward = 1.0 - torch.tanh(object_speed / velocity_std)

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    opened = torch.clamp(gripper_pos / open_joint_pos, min=0.0, max=1.0)
    progress_gate = _goal_curriculum_gate(env, command_name, min_progress)
    return progress_gate * lifted.float() * near_goal * still_reward * torch.mean(opened, dim=1)


def gripper_open_near_goal_xy(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_threshold: float,
    minimal_height: float,
    command_name: str,
    open_joint_pos: float,
    velocity_std: float,
    min_progress: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward opening the gripper above the basket when the object is horizontally near the goal."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance_xy = torch.norm(goal_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    height_error = torch.abs(goal_pos_w[:, 2] - obj.data.root_pos_w[:, 2])
    near_goal_xy = torch.clamp((xy_threshold - distance_xy) / xy_threshold, min=0.0, max=1.0)
    near_goal_z = torch.clamp((z_threshold - height_error) / z_threshold, min=0.0, max=1.0)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    object_speed = torch.norm(obj.data.root_lin_vel_w, dim=1)
    still_reward = 1.0 - torch.tanh(object_speed / velocity_std)

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    opened = torch.clamp(gripper_pos / open_joint_pos, min=0.0, max=1.0)
    progress_gate = _goal_curriculum_gate(env, command_name, min_progress)
    return progress_gate * lifted.float() * near_goal_xy * near_goal_z * still_reward * torch.mean(opened, dim=1)


def gripper_open_far_from_goal_xy(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    minimal_height: float,
    command_name: str,
    open_joint_pos: float,
    min_progress: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Penalty signal for opening while carrying the object outside the basket XY basin."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance_xy = torch.norm(goal_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    outside_goal_xy = 1.0 - torch.clamp((xy_threshold - distance_xy) / xy_threshold, min=0.0, max=1.0)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    opened = torch.clamp(gripper_pos / open_joint_pos, min=0.0, max=1.0)
    progress_gate = _goal_curriculum_gate(env, command_name, min_progress)
    return progress_gate * lifted.float() * outside_goal_xy * torch.mean(opened, dim=1)


def gripper_closed_near_goal(
    env: ManagerBasedRLEnv,
    threshold: float,
    minimal_height: float,
    command_name: str,
    open_joint_pos: float,
    velocity_std: float,
    min_progress: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Penalize staying closed after the lifted object has reached the goal basin."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance = torch.norm(goal_pos_w - obj.data.root_pos_w, dim=1)
    near_goal = torch.clamp((threshold - distance) / threshold, min=0.0, max=1.0)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    object_speed = torch.norm(obj.data.root_lin_vel_w, dim=1)
    still_reward = 1.0 - torch.tanh(object_speed / velocity_std)

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    closed = torch.clamp((open_joint_pos - gripper_pos) / open_joint_pos, min=0.0, max=1.0)
    progress_gate = _goal_curriculum_gate(env, command_name, min_progress)
    return progress_gate * lifted.float() * near_goal * still_reward * torch.mean(closed, dim=1)


def gripper_closed_near_goal_xy(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_threshold: float,
    minimal_height: float,
    command_name: str,
    open_joint_pos: float,
    velocity_std: float,
    min_progress: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Penalize staying closed after reaching the basket XY basin."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance_xy = torch.norm(goal_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    height_error = torch.abs(goal_pos_w[:, 2] - obj.data.root_pos_w[:, 2])
    near_goal_xy = torch.clamp((xy_threshold - distance_xy) / xy_threshold, min=0.0, max=1.0)
    near_goal_z = torch.clamp((z_threshold - height_error) / z_threshold, min=0.0, max=1.0)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    object_speed = torch.norm(obj.data.root_lin_vel_w, dim=1)
    still_reward = 1.0 - torch.tanh(object_speed / velocity_std)

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    closed = torch.clamp((open_joint_pos - gripper_pos) / open_joint_pos, min=0.0, max=1.0)
    progress_gate = _goal_curriculum_gate(env, command_name, min_progress)
    return progress_gate * lifted.float() * near_goal_xy * near_goal_z * still_reward * torch.mean(closed, dim=1)


def gripper_closed_far_from_goal_xy(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    minimal_height: float,
    command_name: str,
    open_joint_pos: float,
    min_progress: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward keeping the gripper closed while carrying outside the basket XY basin."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance_xy = torch.norm(goal_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    outside_goal_xy = 1.0 - torch.clamp((xy_threshold - distance_xy) / xy_threshold, min=0.0, max=1.0)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    closed = torch.clamp((open_joint_pos - gripper_pos) / open_joint_pos, min=0.0, max=1.0)
    progress_gate = _goal_curriculum_gate(env, command_name, min_progress)
    return progress_gate * lifted.float() * outside_goal_xy * torch.mean(closed, dim=1)


def gripper_open_before_final_goal(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    command_name: str,
    open_joint_pos: float,
    min_progress: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Penalize opening the gripper while carrying the object before the final goal stage."""

    robot: Articulation = env.scene[gripper_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    pre_final_gate = 1.0 - _goal_curriculum_gate(env, command_name, min_progress)

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    opened = torch.clamp(gripper_pos / open_joint_pos, min=0.0, max=1.0)
    return pre_final_gate * lifted.float() * torch.mean(opened, dim=1)


def object_goal_settled_after_release(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_threshold: float,
    command_name: str,
    velocity_std: float,
    min_progress: float,
    open_joint_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward a released object for settling at the final 3D goal."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance_xy = torch.norm(goal_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    height_error = torch.abs(goal_pos_w[:, 2] - obj.data.root_pos_w[:, 2])
    near_xy = torch.clamp((xy_threshold - distance_xy) / xy_threshold, min=0.0, max=1.0)
    near_z = torch.clamp((z_threshold - height_error) / z_threshold, min=0.0, max=1.0)
    object_speed = torch.norm(obj.data.root_lin_vel_w, dim=1)
    still_reward = 1.0 - torch.tanh(object_speed / velocity_std)
    progress_gate = _goal_curriculum_gate(env, command_name, min_progress)

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    opened = torch.clamp(gripper_pos / open_joint_pos, min=0.0, max=1.0)
    open_gate = torch.mean(opened, dim=1)
    return progress_gate * near_xy * near_z * still_reward * open_gate


def object_released_at_goal(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_threshold: float,
    max_speed: float,
    min_progress: float,
    command_name: str,
    open_joint_pos: float,
    open_fraction: float = 0.7,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Terminate when the object is released and settled at the final goal."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance_xy = torch.norm(goal_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    height_error = torch.abs(goal_pos_w[:, 2] - obj.data.root_pos_w[:, 2])
    object_speed = torch.norm(obj.data.root_lin_vel_w, dim=1)
    progress_gate = _goal_curriculum_gate(env, command_name, min_progress) > 0.0

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    opened = torch.mean(torch.clamp(gripper_pos / open_joint_pos, min=0.0, max=1.0), dim=1)
    return (
        progress_gate
        & (distance_xy < xy_threshold)
        & (height_error < z_threshold)
        & (object_speed < max_speed)
        & (opened > open_fraction)
    )


def object_goal_xy_stable(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    minimum_height: float,
    command_name: str,
    velocity_std: float,
    min_progress: float,
    open_joint_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward the object for settling inside the goal XY basin after release."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance_xy = torch.norm(goal_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    near_xy = torch.clamp((xy_threshold - distance_xy) / xy_threshold, min=0.0, max=1.0)
    above_table = obj.data.root_pos_w[:, 2] > minimum_height
    object_speed = torch.norm(obj.data.root_lin_vel_w, dim=1)
    still_reward = 1.0 - torch.tanh(object_speed / velocity_std)
    progress_gate = (command_term.metrics["goal_curriculum_progress"] >= min_progress).float()

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    opened = torch.clamp(gripper_pos / open_joint_pos, min=0.0, max=1.0)
    open_gate = torch.mean(opened, dim=1)
    return progress_gate * above_table.float() * near_xy * still_reward * open_gate


def object_goal_xy_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    grasp_std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward horizontal object-goal alignment after the object is lifted and grasped."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance_xy = torch.norm(goal_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    grasp_gate = _closed_grasp_gate(env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg)
    return lifted.float() * grasp_gate * (1.0 - torch.tanh(distance_xy / std))


def object_goal_height_tracking(
    env: ManagerBasedRLEnv,
    std: float,
    xy_threshold: float,
    minimal_height: float,
    command_name: str,
    grasp_std: float,
    open_joint_pos: float,
    object_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_body_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["openarm_right_left_finger", "openarm_right_right_finger"]
    ),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward matching goal height once the grasped object is horizontally near the goal."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    distance_xy = torch.norm(goal_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    near_xy = torch.clamp((xy_threshold - distance_xy) / xy_threshold, min=0.0, max=1.0)
    height_error = torch.abs(goal_pos_w[:, 2] - obj.data.root_pos_w[:, 2])
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    grasp_gate = _closed_grasp_gate(env, grasp_std, open_joint_pos, object_offset, object_cfg, grasp_body_cfg, gripper_cfg)
    return lifted.float() * grasp_gate * near_xy * (1.0 - torch.tanh(height_error / std))


def hold_joint_position_target(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    joint_pos: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_left_finger_joint.*"]),
):
    """Keep non-policy joints at a fixed position target during the episode."""

    robot: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
        num_envs = env.num_envs
    else:
        num_envs = len(env_ids)
    joint_count = len(asset_cfg.joint_ids)
    target = torch.full((num_envs, joint_count), joint_pos, device=robot.device)
    zero_velocity = torch.zeros((num_envs, joint_count), device=robot.device)
    robot.set_joint_position_target(target, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    robot.set_joint_velocity_target(zero_velocity, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def body_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
) -> torch.Tensor:
    """Return a robot body position expressed in the robot root frame."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    body_id = robot_body_cfg.body_ids[0]
    body_pos_w = robot.data.body_pos_w[:, body_id, :]
    body_quat_w = robot.data.body_quat_w[:, body_id, :]
    body_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, body_pos_w, body_quat_w)
    return body_pos_b


def object_body_axis_distance(
    env: ManagerBasedRLEnv,
    axis: int,
    std: float,
    object_offset: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
) -> torch.Tensor:
    """Reward matching one world-axis coordinate between an object offset point and a robot body."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_id = robot_body_cfg.body_ids[0]
    target = obj.data.root_pos_w[:, axis] + object_offset
    error = torch.abs(target - robot.data.body_pos_w[:, body_id, axis])
    return 1.0 - torch.tanh(error / std)


def body_axis_above_minimum(
    env: ManagerBasedRLEnv,
    axis: int,
    minimum_value: float,
    std: float,
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
) -> torch.Tensor:
    """Reward a body for staying above a minimum coordinate on a world axis."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    body_pos = robot.data.body_pos_w[:, robot_body_cfg.body_ids, axis]
    if body_pos.ndim > 1:
        body_pos = torch.min(body_pos, dim=1).values
    violation = torch.clamp(minimum_value - body_pos, min=0.0)
    return 1.0 - torch.tanh(violation / std)


def body_axis_above_minimum_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    axis: int,
    minimum_value: float,
    std: float,
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
) -> torch.Tensor:
    """Reward a body for staying above a robot-root-frame coordinate."""

    body_pos_b = body_position_in_robot_root_frame(env, robot_body_cfg)
    violation = torch.clamp(minimum_value - body_pos_b[:, axis], min=0.0)
    return 1.0 - torch.tanh(violation / std)


def body_axis_below_minimum(
    env: ManagerBasedRLEnv,
    axis: int,
    minimum_value: float,
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
) -> torch.Tensor:
    """Return the body coordinate violation below a minimum world-axis value."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    body_pos = robot.data.body_pos_w[:, robot_body_cfg.body_ids, axis]
    if body_pos.ndim > 1:
        body_pos = torch.min(body_pos, dim=1).values
    return torch.clamp(minimum_value - body_pos, min=0.0)


def body_axis_below_minimum_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    axis: int,
    minimum_value: float,
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
) -> torch.Tensor:
    """Return violation below a minimum robot-root-frame coordinate."""

    body_pos_b = body_position_in_robot_root_frame(env, robot_body_cfg)
    return torch.clamp(minimum_value - body_pos_b[:, axis], min=0.0)


def body_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    robot_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["openarm_right_hand"]),
) -> torch.Tensor:
    """Return the body height violation in meters for clearance penalties."""

    robot: Articulation = env.scene[robot_body_cfg.name]
    body_pos_z = robot.data.body_pos_w[:, robot_body_cfg.body_ids, 2]
    if body_pos_z.ndim > 1:
        body_pos_z = torch.min(body_pos_z, dim=1).values
    return torch.clamp(minimum_height - body_pos_z, min=0.0)


def action_rate_l2_clamped(env: ManagerBasedRLEnv, max_rate: float = 2.0) -> torch.Tensor:
    """Penalize action changes after clipping rare policy outliers."""

    delta = env.action_manager.action - env.action_manager.prev_action
    delta = torch.clamp(delta, min=-max_rate, max=max_rate)
    return torch.sum(torch.square(delta), dim=1)


def joint_position_deviation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_left_joint.*"]),
) -> torch.Tensor:
    """Return squared joint-position deviation from the reset/default pose."""

    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_pos - default_joint_pos), dim=1)
