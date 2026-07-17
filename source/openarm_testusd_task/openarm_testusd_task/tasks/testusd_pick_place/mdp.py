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

from .controller_math import (
    advance_post_center_descent_progress,
    actor_owned_carry_phase,
    apply_stopping_speed_envelope,
    automatic_release_ready,
    build_prioritized_dls_solver,
    capped_raw_coupled_lift_rise,
    carry_clearance_fraction,
    constrain_target_lead,
    coupled_lift_progress,
    damped_pseudoinverse,
    jerk_limited_cartesian_velocity,
    limit_vector_norm,
    low_clearance_inward_progress_cost,
    persistent_best_signed_progress,
    phase_primary_axis_progress_valid,
    phase_primary_translation_progress_ratio,
    post_grasp_actor_authority,
    post_center_descent_readiness,
    preserve_safe_goal_directed_step,
    quintic_smoothstep,
    realized_translation_progress,
    scale_delta_to_joint_limits,
    scale_delta_to_cartesian_bounds,
    select_higher_phase_primary_candidate,
    select_phase_axis_fallback,
    select_phase_primary_axis,
    solve_primary_dls_delta,
    solve_prioritized_dls_delta,
    target_lead_constraint_intersection,
    target_lead_constraints_valid,
    translation_progress_ratios,
)

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
            "task_grasp_alignment_progress",
            "task_grasp_offset_tcp_x",
            "task_grasp_offset_tcp_y",
            "task_grasp_offset_tcp_z",
            "task_finger_distance_asymmetry",
            "task_gripper_bilateral_closed_fraction",
            "task_gripper_bilateral_open_fraction",
            "task_gripper_aperture_asymmetry",
            "task_gripper_joint_speed",
            "task_grasp_relative_speed",
            "task_grasp_coupling_error",
            "task_grasp_offset_error",
            "task_grasp_closing_progress",
            "task_grasp_contact_progress",
            "task_grasp_lift_authorized",
            "task_grasp_verification_progress",
            "task_grasp_reopen_progress",
            "task_grasp_retry_cooldown",
            "task_grasp_attempt_aborts",
            "task_grasp_alignment_aborts",
            "task_grasp_coupling_aborts",
            "task_grasp_close_timeouts",
            "task_grasp_verification_timeouts",
            "task_grasp_empty_close_progress",
            "task_grasp_empty_close_aborts",
            "task_grasp_close_grace_entries",
            "task_grasp_close_timeout_near_contact",
            "task_grasp_first_attempt_success",
            "task_grasp_retry_success",
            "task_grasp_candidate_object_lift",
            "task_grasp_candidate_tcp_lift",
            "task_verified_lift_object_progress",
            "task_verified_lift_tcp_progress",
            "task_verified_lift_coupled_progress",
            "task_verified_lift_observed_watermark",
            "task_verified_lift_credited_rise",
            "task_transport_displacement",
            "task_low_clearance_basket_approach_steps",
            "task_assist_engaged",
            "task_assist_active",
            "task_assist_owns_arm",
            "task_grasp_controller_owns_arm",
            "task_grasp_retry_recovery_active",
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
            "task_release_command_height",
            "task_release_command_height_error",
            "task_release_command_speed",
            "task_actual_release_height",
            "task_actual_release_height_error",
            "task_actual_release_speed",
            "task_release_curriculum_progress",
            "task_object_speed",
            "task_object_xy_speed",
            "task_object_vertical_speed",
            "task_distance_xy",
            "task_centering_height_error",
            "task_descent_progress",
            "task_descent_target_height",
            "task_post_center_descent_started",
            "task_post_center_descent_elapsed_fraction",
            "task_post_center_descent_steps",
            "task_post_center_descent_tracking_pause_steps",
            "task_post_center_descent_stability_pause_steps",
            "task_post_center_descent_centering_pause_steps",
            "task_post_center_descent_basin_pause_steps",
            "task_assist_entry_object_height",
            "task_assist_entry_object_speed",
            "task_assist_entry_height_blocked_steps",
            "task_assist_entry_height_shortfall_max",
            "task_assist_control_steps",
            "task_assist_requested_vertical_step",
            "task_assist_incremental_vertical_step",
            "task_assist_vertical_step",
            "task_assist_actual_object_vertical_delta",
            "task_assist_actual_tcp_vertical_delta",
            "task_assist_object_vertical_tracking",
            "task_assist_tcp_vertical_tracking",
            "task_assist_realized_requested_xy_step",
            "task_assist_realized_requested_z_step",
            "task_assist_realized_object_xy_progress",
            "task_assist_realized_object_z_progress",
            "task_assist_effort_demand_ratio",
            "task_assist_tcp_orientation_error",
            "task_assist_object_slip",
            "task_assist_watchdog_trips",
            "task_assist_posture_error",
            "task_assist_target_lead",
            "task_assist_entry_blend_steps",
            "task_assist_command_speed",
            "task_assist_command_acceleration",
            "task_assist_xy_control_steps",
            "task_assist_z_control_steps",
            "task_assist_xy_unsafe_steps",
            "task_assist_z_unsafe_steps",
            "task_assist_recovery_hold_steps",
            "task_assist_safety_brake_steps",
            "task_assist_entry_seeded",
            "task_assist_active_progress_ratio_sum",
            "task_assist_xy_progress_ratio_sum",
            "task_assist_z_progress_ratio_sum",
            "task_assist_low_progress_steps",
            "task_assist_xy_low_progress_steps",
            "task_assist_z_low_progress_steps",
            "task_assist_incremental_safety_stops",
            "task_assist_incremental_scaled_repairs",
            "task_assist_incremental_xy_fallbacks",
            "task_assist_incremental_z_fallbacks",
            "task_assist_z_priority_control_steps",
            "task_assist_incremental_z_priority_upgrades",
            "task_assist_incremental_xy_rejects",
            "task_assist_incremental_z_rejects",
            "task_assist_lead_fallbacks",
            "task_assist_lead_previous_fallbacks",
            "task_assist_lead_incremental_fallbacks",
            "task_assist_lead_safety_stops",
            "task_assist_lead_constraint_rebases",
            "task_assist_target_backlog",
            "task_assist_entry_joint_limit_proximity",
            "task_assist_current_joint_limit_proximity",
            "task_assist_target_joint_limit_proximity",
            "task_phase_approach_steps",
            "task_phase_close_verify_steps",
            "task_phase_retry_reopen_steps",
            "task_phase_actor_transport_steps",
            "task_phase_assist_to_center_steps",
            "task_phase_center_to_release_steps",
            "task_phase_opening_steps",
            "task_phase_settle_steps",
            "task_timeout_phase_approach",
            "task_timeout_phase_close_verify",
            "task_timeout_phase_retry_reopen",
            "task_timeout_phase_actor_transport",
            "task_timeout_phase_assist_to_center",
            "task_timeout_phase_center_to_release",
            "task_timeout_phase_opening",
            "task_timeout_phase_settle",
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
            self.metrics["task_grasp_alignment_progress"] = (
                state.grasp_alignment_streak.float()
                / max(state._alignment_confirm_steps, 1)
            )
            self.metrics["task_grasp_offset_tcp_x"] = state.object_offset_tcp[:, 0].clone()
            self.metrics["task_grasp_offset_tcp_y"] = state.object_offset_tcp[:, 1].clone()
            self.metrics["task_grasp_offset_tcp_z"] = state.object_offset_tcp[:, 2].clone()
            self.metrics["task_finger_distance_asymmetry"] = (
                state.finger_distance_asymmetry.clone()
            )
            self.metrics["task_gripper_bilateral_closed_fraction"] = (
                state.bilateral_closed_fraction.clone()
            )
            self.metrics["task_gripper_bilateral_open_fraction"] = (
                state.bilateral_open_fraction.clone()
            )
            self.metrics["task_gripper_aperture_asymmetry"] = (
                state.gripper_aperture_asymmetry.clone()
            )
            self.metrics["task_gripper_joint_speed"] = state.gripper_joint_speed.clone()
            self.metrics["task_grasp_relative_speed"] = state.grasp_relative_speed.clone()
            self.metrics["task_grasp_coupling_error"] = state.grasp_coupling_error.clone()
            self.metrics["task_grasp_offset_error"] = state.grasp_offset_error.clone()
            self.metrics["task_grasp_closing_progress"] = (
                state.grasp_closing_streak.float()
                / max(
                    state._closing_timeout_steps + state._closing_contact_grace_steps,
                    1,
                )
            )
            self.metrics["task_grasp_contact_progress"] = (
                state.grasp_contact_streak.float() / max(state._contact_confirm_steps, 1)
            )
            self.metrics["task_grasp_lift_authorized"] = state.grasp_lift_authorized.float()
            self.metrics["task_grasp_verification_progress"] = (
                state.grasp_verification_streak.float()
                / max(state._verification_timeout_steps, 1)
            )
            self.metrics["task_grasp_reopen_progress"] = (
                state.grasp_reopen_streak.float()
                / max(state._reopen_arm_hold_steps, 1)
            )
            self.metrics["task_grasp_retry_cooldown"] = state.grasp_retry_cooldown.float()
            self.metrics["task_grasp_attempt_aborts"] = state.grasp_attempt_aborts.float()
            self.metrics["task_grasp_alignment_aborts"] = state.grasp_alignment_aborts.float()
            self.metrics["task_grasp_coupling_aborts"] = state.grasp_coupling_aborts.float()
            self.metrics["task_grasp_close_timeouts"] = state.grasp_close_timeouts.float()
            self.metrics["task_grasp_verification_timeouts"] = (
                state.grasp_verification_timeouts.float()
            )
            self.metrics["task_grasp_empty_close_progress"] = (
                state.grasp_empty_close_streak.float()
                / max(state._empty_close_confirm_steps, 1)
            )
            self.metrics["task_grasp_empty_close_aborts"] = (
                state.grasp_empty_close_aborts.float()
            )
            self.metrics["task_grasp_close_grace_entries"] = (
                state.grasp_close_grace_entries.float()
            )
            self.metrics["task_grasp_close_timeout_near_contact"] = (
                state.grasp_close_timeout_near_contact.float()
            )
            self.metrics["task_grasp_first_attempt_success"] = (
                state.grasp_first_attempt_success.float()
            )
            self.metrics["task_grasp_retry_success"] = state.grasp_retry_success.float()
            self.metrics["task_grasp_candidate_object_lift"] = (
                state.grasp_candidate_object_lift.clone()
            )
            self.metrics["task_grasp_candidate_tcp_lift"] = (
                state.grasp_candidate_tcp_lift.clone()
            )
            self.metrics["task_verified_lift_object_progress"] = (
                state.verified_lift_object_progress.clone()
            )
            self.metrics["task_verified_lift_tcp_progress"] = (
                state.verified_lift_tcp_progress.clone()
            )
            self.metrics["task_verified_lift_coupled_progress"] = (
                state.verified_lift_coupled_progress.clone()
            )
            self.metrics["task_verified_lift_observed_watermark"] = (
                state.verified_lift_observed_watermark.clone()
            )
            self.metrics["task_verified_lift_credited_rise"] = (
                state.verified_lift_credited_rise.clone()
            )
            self.metrics["task_transport_displacement"] = state.transport_displacement.clone()
            low_clearance_approach = (
                actor_owned_carry_phase(
                    state.grasp_acquired,
                    state.assist_owns_arm,
                    state.released,
                    state.failed,
                )
                & (state.distance_xy <= self.cfg.near_basket_xy_threshold)
                & (state.object_height < self.cfg.low_clearance_height)
            )
            self.metrics["task_low_clearance_basket_approach_steps"].add_(
                low_clearance_approach.float()
            )
            self.metrics["task_assist_engaged"] = state.assist_engaged.float()
            self.metrics["task_assist_active"] = state.assist_active.float()
            self.metrics["task_assist_owns_arm"] = state.assist_owns_arm.float()
            self.metrics["task_grasp_controller_owns_arm"] = (
                state.grasp_controller_owns_arm.float()
            )
            self.metrics["task_grasp_retry_recovery_active"] = (
                state.grasp_retry_recovery_active.float()
            )
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
            self.metrics["task_release_command_height"] = state.release_command_height.clone()
            self.metrics["task_release_command_height_error"] = (
                state.release_command_height_error.clone()
            )
            self.metrics["task_release_command_speed"] = state.release_command_speed.clone()
            self.metrics["task_actual_release_height"] = state.actual_release_height.clone()
            self.metrics["task_actual_release_height_error"] = (
                state.actual_release_height_error.clone()
            )
            self.metrics["task_actual_release_speed"] = state.actual_release_speed.clone()
            self.metrics["task_release_curriculum_progress"] = state.release_curriculum_progress.clone()
            self.metrics["task_object_speed"] = state.object_speed.clone()
            self.metrics["task_object_xy_speed"] = state.object_xy_speed.clone()
            self.metrics["task_object_vertical_speed"] = state.object_vertical_speed.clone()
            self.metrics["task_distance_xy"] = state.distance_xy.clone()
            self.metrics["task_centering_height_error"] = state.centering_height_error.clone()
            self.metrics["task_descent_progress"] = state.descent_progress.clone()
            self.metrics["task_descent_target_height"] = state.descent_target_height.clone()
            self.metrics["task_post_center_descent_started"] = (
                state.post_center_descent_started.float()
            )
            self.metrics["task_post_center_descent_elapsed_fraction"] = (
                state.post_center_descent_elapsed_steps.float()
                / max(state._post_center_descent_steps, 1)
            )
            self.metrics["task_post_center_descent_steps"] = (
                state.post_center_descent_elapsed_steps.float()
            )
            self.metrics["task_post_center_descent_tracking_pause_steps"] = (
                state.post_center_descent_tracking_pause_steps.float()
            )
            self.metrics["task_post_center_descent_stability_pause_steps"] = (
                state.post_center_descent_stability_pause_steps.float()
            )
            self.metrics["task_post_center_descent_centering_pause_steps"] = (
                state.post_center_descent_centering_pause_steps.float()
            )
            self.metrics["task_post_center_descent_basin_pause_steps"] = (
                state.post_center_descent_basin_pause_steps.float()
            )
            self.metrics["task_assist_entry_object_height"] = (
                state.assist_entry_object_height.clone()
            )
            self.metrics["task_assist_entry_object_speed"] = (
                state.assist_entry_object_speed.clone()
            )
            self.metrics["task_assist_entry_height_blocked_steps"] = (
                state.assist_entry_height_blocked_steps.float()
            )
            self.metrics["task_assist_entry_height_shortfall_max"] = (
                state.assist_entry_height_shortfall_max.clone()
            )
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
                self.metrics["task_assist_realized_requested_xy_step"].add_(
                    arm_action.assist_realized_requested_xy_step
                )
                self.metrics["task_assist_realized_requested_z_step"].add_(
                    arm_action.assist_realized_requested_z_step
                )
                self.metrics["task_assist_realized_object_xy_progress"].add_(
                    arm_action.assist_realized_object_xy_progress
                )
                self.metrics["task_assist_realized_object_z_progress"].add_(
                    arm_action.assist_realized_object_z_progress
                )
                self.metrics["task_assist_effort_demand_ratio"] = torch.maximum(
                    self.metrics["task_assist_effort_demand_ratio"],
                    arm_action.assist_effort_demand_ratio,
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
                blend_active = (
                    state.assist_active
                    & (arm_action.assist_entry_blend_progress < 1.0)
                )
                self.metrics["task_assist_entry_blend_steps"].add_(blend_active.float())
                self.metrics["task_assist_command_speed"] = torch.maximum(
                    self.metrics["task_assist_command_speed"],
                    arm_action.assist_command_speed,
                )
                self.metrics["task_assist_command_acceleration"] = torch.maximum(
                    self.metrics["task_assist_command_acceleration"],
                    arm_action.assist_command_acceleration,
                )
                self.metrics["task_assist_xy_control_steps"].add_(
                    arm_action.assist_xy_active
                )
                self.metrics["task_assist_z_control_steps"].add_(
                    arm_action.assist_z_active
                )
                self.metrics["task_assist_xy_unsafe_steps"].add_(
                    arm_action.assist_xy_active * (1.0 - arm_action.assist_xy_safe)
                )
                self.metrics["task_assist_z_unsafe_steps"].add_(
                    arm_action.assist_z_active * (1.0 - arm_action.assist_z_safe)
                )
                self.metrics["task_assist_recovery_hold_steps"].add_(
                    arm_action.assist_recovery_hold
                )
                self.metrics["task_assist_safety_brake_steps"].add_(
                    arm_action.assist_safety_brake
                )
                self.metrics["task_assist_entry_seeded"].add_(
                    arm_action.assist_entry_seeded
                )
                self.metrics["task_assist_active_progress_ratio_sum"].add_(
                    arm_action.assist_active_progress_ratio
                    * (
                        (arm_action.assist_xy_active > 0.0)
                        | (arm_action.assist_z_active > 0.0)
                    ).float()
                )
                self.metrics["task_assist_xy_progress_ratio_sum"].add_(
                    arm_action.assist_xy_progress_ratio
                    * (arm_action.assist_xy_active > 0.0).float()
                )
                self.metrics["task_assist_z_progress_ratio_sum"].add_(
                    arm_action.assist_z_progress_ratio
                    * (arm_action.assist_z_active > 0.0).float()
                )
                self.metrics["task_assist_low_progress_steps"].add_(
                    arm_action.assist_low_progress
                )
                self.metrics["task_assist_xy_low_progress_steps"].add_(
                    arm_action.assist_xy_low_progress
                )
                self.metrics["task_assist_z_low_progress_steps"].add_(
                    arm_action.assist_z_low_progress
                )
                self.metrics["task_assist_incremental_safety_stops"].add_(
                    arm_action.assist_incremental_safety_stop
                )
                self.metrics["task_assist_incremental_scaled_repairs"].add_(
                    arm_action.assist_incremental_scaled_repair
                )
                self.metrics["task_assist_incremental_xy_fallbacks"].add_(
                    arm_action.assist_incremental_xy_fallback
                )
                self.metrics["task_assist_incremental_z_fallbacks"].add_(
                    arm_action.assist_incremental_z_fallback
                )
                self.metrics["task_assist_z_priority_control_steps"].add_(
                    arm_action.assist_z_priority
                )
                self.metrics[
                    "task_assist_incremental_z_priority_upgrades"
                ].add_(arm_action.assist_incremental_z_priority_upgrade)
                self.metrics["task_assist_incremental_xy_rejects"].add_(
                    arm_action.assist_incremental_xy_reject
                )
                self.metrics["task_assist_incremental_z_rejects"].add_(
                    arm_action.assist_incremental_z_reject
                )
                self.metrics["task_assist_lead_fallbacks"].add_(
                    arm_action.assist_lead_fallback
                )
                self.metrics["task_assist_lead_previous_fallbacks"].add_(
                    arm_action.assist_lead_previous_fallback
                )
                self.metrics["task_assist_lead_incremental_fallbacks"].add_(
                    arm_action.assist_lead_incremental_fallback
                )
                self.metrics["task_assist_lead_safety_stops"].add_(
                    arm_action.assist_lead_safety_stop
                )
                self.metrics["task_assist_lead_constraint_rebases"].add_(
                    arm_action.assist_lead_constraint_rebase
                )
                self.metrics["task_assist_target_backlog"] = torch.maximum(
                    self.metrics["task_assist_target_backlog"],
                    arm_action.assist_target_backlog,
                )
                self.metrics["task_assist_entry_joint_limit_proximity"] = torch.maximum(
                    self.metrics["task_assist_entry_joint_limit_proximity"],
                    arm_action.assist_entry_joint_limit_proximity,
                )
                self.metrics["task_assist_current_joint_limit_proximity"] = torch.maximum(
                    self.metrics["task_assist_current_joint_limit_proximity"],
                    arm_action.assist_current_joint_limit_proximity,
                )
                self.metrics["task_assist_target_joint_limit_proximity"] = torch.maximum(
                    self.metrics["task_assist_target_joint_limit_proximity"],
                    arm_action.assist_target_joint_limit_proximity,
                )
            self.sync_task_latched_metrics(state)
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

    def sync_task_latched_metrics(self, state: "TaskStateTracker"):
        """Copy terminal-safe latches without duplicating cumulative assist metrics."""

        for metric_name, state_name in (
            ("task_grasp_attempt_aborts", "grasp_attempt_aborts"),
            ("task_grasp_alignment_aborts", "grasp_alignment_aborts"),
            ("task_grasp_coupling_aborts", "grasp_coupling_aborts"),
            ("task_grasp_close_timeouts", "grasp_close_timeouts"),
            ("task_grasp_verification_timeouts", "grasp_verification_timeouts"),
            ("task_grasp_empty_close_aborts", "grasp_empty_close_aborts"),
            ("task_grasp_close_grace_entries", "grasp_close_grace_entries"),
            (
                "task_grasp_close_timeout_near_contact",
                "grasp_close_timeout_near_contact",
            ),
            ("task_grasp_first_attempt_success", "grasp_first_attempt_success"),
            ("task_grasp_retry_success", "grasp_retry_success"),
        ):
            self.metrics[metric_name] = getattr(state, state_name).float()

        self.metrics["task_post_center_descent_started"] = (
            state.post_center_descent_started.float()
        )
        self.metrics["task_post_center_descent_elapsed_fraction"] = (
            state.post_center_descent_elapsed_steps.float()
            / max(state._post_center_descent_steps, 1)
        )
        for metric_name, state_name in (
            ("task_post_center_descent_steps", "post_center_descent_elapsed_steps"),
            (
                "task_post_center_descent_tracking_pause_steps",
                "post_center_descent_tracking_pause_steps",
            ),
            (
                "task_post_center_descent_stability_pause_steps",
                "post_center_descent_stability_pause_steps",
            ),
            (
                "task_post_center_descent_centering_pause_steps",
                "post_center_descent_centering_pause_steps",
            ),
            (
                "task_post_center_descent_basin_pause_steps",
                "post_center_descent_basin_pause_steps",
            ),
            ("task_assist_entry_object_height", "assist_entry_object_height"),
            ("task_assist_entry_object_speed", "assist_entry_object_speed"),
            (
                "task_assist_entry_height_blocked_steps",
                "assist_entry_height_blocked_steps",
            ),
            (
                "task_assist_entry_height_shortfall_max",
                "assist_entry_height_shortfall_max",
            ),
        ):
            self.metrics[metric_name] = getattr(state, state_name).float()

        for phase_name in (
            "approach",
            "close_verify",
            "retry_reopen",
            "actor_transport",
            "assist_to_center",
            "center_to_release",
            "opening",
            "settle",
        ):
            self.metrics[f"task_phase_{phase_name}_steps"] = getattr(
                state, f"phase_{phase_name}_steps"
            ).float()
            self.metrics[f"task_timeout_phase_{phase_name}"] = getattr(
                state, f"timeout_phase_{phase_name}"
            ).float()

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
    low_clearance_height: float = 0.710
    near_basket_xy_threshold: float = 0.17


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
                release_ready = automatic_release_ready(
                    state.grasp_acquired,
                    state.lift_completed,
                    state.validated_transport,
                    state.release_ready_reached,
                    state.release_ready,
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


def _imminent_automatic_release(
    env: ManagerBasedRLEnv, state: "TaskStateTracker"
) -> torch.Tensor:
    """Predict the release request made by the later gripper action term."""

    gripper_action = env.action_manager.get_term("gripper_action")
    automatic_release_enabled = bool(
        getattr(gripper_action.cfg, "enforce_task_phase", False)
        and getattr(gripper_action.cfg, "automatic_phase_control", False)
    )
    if not automatic_release_enabled:
        return torch.zeros_like(state.release_commanded)
    return automatic_release_ready(
        state.grasp_acquired,
        state.lift_completed,
        state.validated_transport,
        state.release_ready_reached,
        state.release_ready,
    )


class PlaceAssistJointPositionAction(JointPositionAction):
    """Drive TCP translation from closed-loop object error after assist entry."""

    cfg: "PlaceAssistJointPositionActionCfg"

    def __init__(self, cfg: "PlaceAssistJointPositionActionCfg", env):
        super().__init__(cfg, env)
        if cfg.assist_entry_blend_steps < 0 or cfg.assist_entry_target_lead_steps < 0:
            raise ValueError("assist entry step counts must be non-negative")
        if cfg.assist_target_lead_steps < 1:
            raise ValueError("assist target lead steps must be positive")
        if min(
            cfg.horizontal_position_deadband,
            cfg.vertical_position_deadband,
        ) <= 0.0:
            raise ValueError("assist position deadbands must be positive")
        if min(
            cfg.max_horizontal_acceleration,
            cfg.max_vertical_acceleration,
            cfg.max_horizontal_jerk,
            cfg.max_vertical_jerk,
        ) <= 0.0:
            raise ValueError("assist acceleration and jerk limits must be positive")
        if not 0.0 < cfg.minimum_active_progress_ratio <= 1.0:
            raise ValueError("minimum active progress ratio must be in (0, 1]")
        if not 0.0 < cfg.minimum_companion_progress_ratio <= cfg.minimum_active_progress_ratio:
            raise ValueError(
                "minimum companion progress ratio must be in (0, active progress ratio]"
            )
        if cfg.cartesian_hard_leak_tolerance < cfg.cartesian_leak_tolerance:
            raise ValueError("hard Cartesian leak tolerance must not be stricter than telemetry")
        if cfg.slip_recovery_hold_steps < 0:
            raise ValueError("slip recovery hold steps must be non-negative")
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
        self._identity_joints = torch.eye(self.action_dim, device=self.device).unsqueeze(0)
        self._assist_was_active = torch.zeros(self._env.num_envs, dtype=torch.bool, device=self.device)
        self._assist_entry_joint_limit_recorded = torch.zeros(
            self._env.num_envs, dtype=torch.bool, device=self.device
        )
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
        self._grasp_was_verifying = torch.zeros(
            self._env.num_envs, dtype=torch.bool, device=self.device
        )
        self._grasp_handoff_steps_remaining = torch.zeros(
            self._env.num_envs, dtype=torch.long, device=self.device
        )
        self._handoff_steps_remaining = torch.zeros(
            self._env.num_envs, dtype=torch.long, device=self.device
        )
        self._filtered_object_velocity = torch.zeros((self._env.num_envs, 3), device=self.device)
        self._assist_entry_velocity = torch.zeros((self._env.num_envs, 3), device=self.device)
        self._assist_command_velocity = torch.zeros((self._env.num_envs, 3), device=self.device)
        self._assist_command_acceleration = torch.zeros(
            (self._env.num_envs, 3), device=self.device
        )
        self._assist_entry_blend_steps_remaining = torch.zeros(
            self._env.num_envs, dtype=torch.long, device=self.device
        )
        self._slip_recovery_steps_remaining = torch.zeros(
            self._env.num_envs, dtype=torch.long, device=self.device
        )
        self._previous_object_height = torch.zeros(self._env.num_envs, device=self.device)
        self._previous_tcp_height = torch.zeros(self._env.num_envs, device=self.device)
        self._previous_object_position = torch.zeros(
            (self._env.num_envs, 3), device=self.device
        )
        self._motion_height_initialized = torch.zeros(
            self._env.num_envs, dtype=torch.bool, device=self.device
        )
        self._previous_requested_vertical_step = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self._previous_requested_translation = torch.zeros(
            (self._env.num_envs, 3), device=self.device
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
        self.assist_realized_requested_xy_step = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_realized_requested_z_step = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_realized_object_xy_progress = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_realized_object_z_progress = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_effort_demand_ratio = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_tcp_orientation_error = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_object_slip = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_watchdog_trip = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_posture_error = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_target_lead = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_entry_blend_progress = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_command_speed = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_command_acceleration = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_xy_active = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_z_active = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_xy_safe = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_z_safe = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_recovery_hold = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_safety_brake = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_entry_seeded = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_active_progress_ratio = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_xy_progress_ratio = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_z_progress_ratio = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_low_progress = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_xy_low_progress = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_z_low_progress = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_incremental_safety_stop = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_incremental_scaled_repair = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_incremental_xy_fallback = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_incremental_z_fallback = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_z_priority = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_incremental_z_priority_upgrade = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_incremental_xy_reject = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_incremental_z_reject = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_lead_fallback = torch.zeros(self._env.num_envs, device=self.device)
        self.assist_lead_previous_fallback = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_lead_incremental_fallback = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_lead_safety_stop = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_lead_constraint_rebase = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_target_backlog = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_entry_joint_limit_proximity = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_current_joint_limit_proximity = torch.zeros(
            self._env.num_envs, device=self.device
        )
        self.assist_target_joint_limit_proximity = torch.zeros(
            self._env.num_envs, device=self.device
        )

    def reset(self, env_ids=None):
        super().reset(env_ids)
        resolved_env_ids = slice(None) if env_ids is None else env_ids
        self._assist_was_active[resolved_env_ids] = False
        self._assist_entry_joint_limit_recorded[resolved_env_ids] = False
        self._assist_posture_ref[resolved_env_ids] = 0.0
        self._assist_entry_offset[resolved_env_ids] = 0.0
        self._assist_entry_orientation[resolved_env_ids] = 0.0
        self._slip_streak[resolved_env_ids] = 0
        self._assist_joint_target[resolved_env_ids] = 0.0
        self._grasp_hold_target[resolved_env_ids] = 0.0
        self._grasp_was_controlled[resolved_env_ids] = False
        self._grasp_was_verifying[resolved_env_ids] = False
        self._grasp_handoff_steps_remaining[resolved_env_ids] = 0
        self._handoff_steps_remaining[resolved_env_ids] = 0
        self._filtered_object_velocity[resolved_env_ids] = 0.0
        self._assist_entry_velocity[resolved_env_ids] = 0.0
        self._assist_command_velocity[resolved_env_ids] = 0.0
        self._assist_command_acceleration[resolved_env_ids] = 0.0
        self._assist_entry_blend_steps_remaining[resolved_env_ids] = 0
        self._slip_recovery_steps_remaining[resolved_env_ids] = 0
        self._previous_object_height[resolved_env_ids] = 0.0
        self._previous_tcp_height[resolved_env_ids] = 0.0
        self._previous_object_position[resolved_env_ids] = 0.0
        self._motion_height_initialized[resolved_env_ids] = False
        self._previous_requested_vertical_step[resolved_env_ids] = 0.0
        self._previous_requested_translation[resolved_env_ids] = 0.0
        self.assist_requested_vertical_step[resolved_env_ids] = 0.0
        self.assist_incremental_vertical_step[resolved_env_ids] = 0.0
        self.assist_vertical_step[resolved_env_ids] = 0.0
        self.assist_actual_object_vertical_delta[resolved_env_ids] = 0.0
        self.assist_actual_tcp_vertical_delta[resolved_env_ids] = 0.0
        self.assist_object_vertical_tracking[resolved_env_ids] = 0.0
        self.assist_tcp_vertical_tracking[resolved_env_ids] = 0.0
        self.assist_realized_requested_xy_step[resolved_env_ids] = 0.0
        self.assist_realized_requested_z_step[resolved_env_ids] = 0.0
        self.assist_realized_object_xy_progress[resolved_env_ids] = 0.0
        self.assist_realized_object_z_progress[resolved_env_ids] = 0.0
        self.assist_effort_demand_ratio[resolved_env_ids] = 0.0
        self.assist_tcp_orientation_error[resolved_env_ids] = 0.0
        self.assist_object_slip[resolved_env_ids] = 0.0
        self.assist_watchdog_trip[resolved_env_ids] = 0.0
        self.assist_posture_error[resolved_env_ids] = 0.0
        self.assist_target_lead[resolved_env_ids] = 0.0
        self.assist_entry_blend_progress[resolved_env_ids] = 0.0
        self.assist_command_speed[resolved_env_ids] = 0.0
        self.assist_command_acceleration[resolved_env_ids] = 0.0
        self.assist_xy_active[resolved_env_ids] = 0.0
        self.assist_z_active[resolved_env_ids] = 0.0
        self.assist_xy_safe[resolved_env_ids] = 0.0
        self.assist_z_safe[resolved_env_ids] = 0.0
        self.assist_recovery_hold[resolved_env_ids] = 0.0
        self.assist_safety_brake[resolved_env_ids] = 0.0
        self.assist_entry_seeded[resolved_env_ids] = 0.0
        self.assist_active_progress_ratio[resolved_env_ids] = 0.0
        self.assist_xy_progress_ratio[resolved_env_ids] = 0.0
        self.assist_z_progress_ratio[resolved_env_ids] = 0.0
        self.assist_low_progress[resolved_env_ids] = 0.0
        self.assist_xy_low_progress[resolved_env_ids] = 0.0
        self.assist_z_low_progress[resolved_env_ids] = 0.0
        self.assist_incremental_safety_stop[resolved_env_ids] = 0.0
        self.assist_incremental_scaled_repair[resolved_env_ids] = 0.0
        self.assist_incremental_xy_fallback[resolved_env_ids] = 0.0
        self.assist_incremental_z_fallback[resolved_env_ids] = 0.0
        self.assist_z_priority[resolved_env_ids] = 0.0
        self.assist_incremental_z_priority_upgrade[resolved_env_ids] = 0.0
        self.assist_incremental_xy_reject[resolved_env_ids] = 0.0
        self.assist_incremental_z_reject[resolved_env_ids] = 0.0
        self.assist_lead_fallback[resolved_env_ids] = 0.0
        self.assist_lead_previous_fallback[resolved_env_ids] = 0.0
        self.assist_lead_incremental_fallback[resolved_env_ids] = 0.0
        self.assist_lead_safety_stop[resolved_env_ids] = 0.0
        self.assist_lead_constraint_rebase[resolved_env_ids] = 0.0
        self.assist_target_backlog[resolved_env_ids] = 0.0
        self.assist_entry_joint_limit_proximity[resolved_env_ids] = 0.0
        self.assist_current_joint_limit_proximity[resolved_env_ids] = 0.0
        self.assist_target_joint_limit_proximity[resolved_env_ids] = 0.0

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

        return scale_delta_to_joint_limits(target, delta, joint_limits)

    def _cartesian_axis_safe(
        self,
        predicted_step: torch.Tensor,
        position_error: torch.Tensor,
        maximum_xy: float | torch.Tensor | None = None,
        maximum_up: float | torch.Tensor | None = None,
        maximum_down: float | torch.Tensor | None = None,
        inactive_leak_tolerance: float | None = None,
        active_limit_epsilon: float = 1.0e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Check XY and Z independently for target progress and bounded inactive-axis leak."""

        leak_tolerance = (
            self.cfg.cartesian_leak_tolerance
            if inactive_leak_tolerance is None
            else inactive_leak_tolerance
        )

        xy_error = torch.norm(position_error[:, :2], dim=1)
        xy_progress = torch.sum(predicted_step[:, :2] * position_error[:, :2], dim=1)
        xy_remaining = torch.norm(position_error[:, :2] - predicted_step[:, :2], dim=1)
        xy_active = xy_error > self.cfg.horizontal_position_deadband
        xy_safe = torch.where(
            xy_active,
            (xy_progress >= -1.0e-9) & (xy_remaining <= xy_error + 1.0e-9),
            torch.norm(predicted_step[:, :2], dim=1) <= leak_tolerance,
        )
        z_error = position_error[:, 2]
        z_remaining = torch.abs(z_error - predicted_step[:, 2])
        z_active = torch.abs(z_error) > self.cfg.vertical_position_deadband
        z_safe = torch.where(
            z_active,
            (predicted_step[:, 2] * z_error >= -1.0e-9)
            & (z_remaining <= torch.abs(z_error) + 1.0e-9),
            torch.abs(predicted_step[:, 2]) <= leak_tolerance,
        )
        if maximum_xy is not None:
            xy_limit = torch.as_tensor(
                maximum_xy,
                device=predicted_step.device,
                dtype=predicted_step.dtype,
            )
            if xy_limit.ndim == 0:
                xy_limit = xy_limit.expand(predicted_step.shape[0])
            xy_safe &= torch.where(
                xy_active,
                torch.norm(predicted_step[:, :2], dim=1)
                <= xy_limit + active_limit_epsilon,
                torch.ones_like(xy_safe),
            )
        if maximum_up is not None and maximum_down is not None:
            up_limit = torch.as_tensor(
                maximum_up,
                device=predicted_step.device,
                dtype=predicted_step.dtype,
            )
            down_limit = torch.as_tensor(
                maximum_down,
                device=predicted_step.device,
                dtype=predicted_step.dtype,
            )
            if up_limit.ndim == 0:
                up_limit = up_limit.expand(predicted_step.shape[0])
            if down_limit.ndim == 0:
                down_limit = down_limit.expand(predicted_step.shape[0])
            z_limit = torch.where(z_error >= 0.0, up_limit, down_limit)
            z_safe &= torch.where(
                z_active,
                torch.abs(predicted_step[:, 2])
                <= z_limit + active_limit_epsilon,
                torch.ones_like(z_safe),
            )
        return xy_safe, z_safe

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        policy_target = self._processed_actions.clone()
        state = getattr(self._env, "_openarm_pick_place_state", None)
        if state is None:
            return
        imminent_automatic_release = _imminent_automatic_release(self._env, state)
        state.assist_active[:] = False
        state.assist_owns_arm[:] = False
        state.grasp_controller_owns_arm[:] = False
        state.grasp_retry_recovery_active[:] = False
        self.assist_requested_vertical_step[:] = 0.0
        self.assist_incremental_vertical_step[:] = 0.0
        self.assist_vertical_step[:] = 0.0
        self.assist_actual_object_vertical_delta[:] = 0.0
        self.assist_actual_tcp_vertical_delta[:] = 0.0
        self.assist_object_vertical_tracking[:] = 0.0
        self.assist_tcp_vertical_tracking[:] = 0.0
        self.assist_realized_requested_xy_step[:] = 0.0
        self.assist_realized_requested_z_step[:] = 0.0
        self.assist_realized_object_xy_progress[:] = 0.0
        self.assist_realized_object_z_progress[:] = 0.0
        self.assist_effort_demand_ratio[:] = 0.0
        self.assist_tcp_orientation_error[:] = 0.0
        self.assist_object_slip[:] = 0.0
        self.assist_watchdog_trip[:] = 0.0
        self.assist_posture_error[:] = 0.0
        self.assist_target_lead[:] = 0.0
        self.assist_entry_blend_progress[:] = 0.0
        self.assist_command_speed[:] = 0.0
        self.assist_command_acceleration[:] = 0.0
        self.assist_xy_active[:] = 0.0
        self.assist_z_active[:] = 0.0
        self.assist_xy_safe[:] = 0.0
        self.assist_z_safe[:] = 0.0
        self.assist_recovery_hold[:] = 0.0
        self.assist_safety_brake[:] = 0.0
        self.assist_entry_seeded[:] = 0.0
        self.assist_active_progress_ratio[:] = 0.0
        self.assist_xy_progress_ratio[:] = 0.0
        self.assist_z_progress_ratio[:] = 0.0
        self.assist_low_progress[:] = 0.0
        self.assist_xy_low_progress[:] = 0.0
        self.assist_z_low_progress[:] = 0.0
        self.assist_incremental_safety_stop[:] = 0.0
        self.assist_incremental_scaled_repair[:] = 0.0
        self.assist_incremental_xy_fallback[:] = 0.0
        self.assist_incremental_z_fallback[:] = 0.0
        self.assist_z_priority[:] = 0.0
        self.assist_incremental_z_priority_upgrade[:] = 0.0
        self.assist_incremental_xy_reject[:] = 0.0
        self.assist_incremental_z_reject[:] = 0.0
        self.assist_lead_fallback[:] = 0.0
        self.assist_lead_previous_fallback[:] = 0.0
        self.assist_lead_incremental_fallback[:] = 0.0
        self.assist_lead_safety_stop[:] = 0.0
        self.assist_lead_constraint_rebase[:] = 0.0
        self.assist_target_backlog[:] = 0.0
        self.assist_entry_joint_limit_proximity[:] = 0.0
        self.assist_current_joint_limit_proximity[:] = 0.0
        self.assist_target_joint_limit_proximity[:] = 0.0
        robot: Articulation = self._env.scene[self.cfg.asset_name]
        obj: RigidObject = self._env.scene[self.cfg.object_name]
        current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        new_grasp_attempt = state.try_start_grasp_attempt(self._env)
        self._grasp_hold_target[new_grasp_attempt] = current_joint_pos[new_grasp_attempt]
        retry_started = state.just_grasp_retry & ~state.grasp_acquired & ~state.failed
        self._grasp_hold_target[retry_started] = current_joint_pos[retry_started]
        closing_hold = (
            state.grasp_attempt_active
            & ~state.grasp_lift_authorized
            & ~state.grasp_acquired
            & ~state.failed
        )
        self._processed_actions[closing_hold] = self._grasp_hold_target[closing_hold]
        verifying_grasp = (
            state.grasp_attempt_active
            & state.grasp_lift_authorized
            & ~state.grasp_acquired
            & ~state.failed
        )
        state.grasp_verification_active[:] = verifying_grasp
        verification_ended = self._grasp_was_verifying & ~verifying_grasp
        self._grasp_handoff_steps_remaining[verification_ended] = self.cfg.grasp_handoff_steps
        self._grasp_was_verifying[:] = verifying_grasp
        retry_recovery = (
            ~state.grasp_attempt_active
            & ~state.grasp_acquired
            & ~state.failed
            & (state.grasp_attempt_aborts > 0)
            & (state.bilateral_open_fraction < state._preclose_min_open_fraction)
            & (state.grasp_reopen_streak < state._reopen_arm_hold_steps)
        )
        self._processed_actions[retry_recovery] = self._grasp_hold_target[retry_recovery]
        grasp_control = closing_hold | retry_recovery
        grasp_control_ended = self._grasp_was_controlled & ~grasp_control
        self._grasp_handoff_steps_remaining[grasp_control_ended] = self.cfg.grasp_handoff_steps
        self._grasp_was_controlled[:] = grasp_control
        grasp_handoff = (
            ~grasp_control
            & (verifying_grasp | (self._grasp_handoff_steps_remaining > 0))
            & ~state.failed
        )
        grasp_handoff_delta = self._scale_by_max_abs(
            policy_target - current_joint_pos, self.cfg.max_joint_step
        )
        self._processed_actions[grasp_handoff] = (
            current_joint_pos + grasp_handoff_delta
        )[grasp_handoff]
        counted_grasp_handoff = grasp_handoff & (self._grasp_handoff_steps_remaining > 0)
        self._grasp_handoff_steps_remaining[counted_grasp_handoff] -= 1
        previously_active = self._assist_was_active.clone()
        current_object_position = obj.data.root_pos_w
        current_object_height = obj.data.root_pos_w[:, 2]
        current_tcp_height = robot.data.body_pos_w[:, self._body_id, 2]
        initialized = self._motion_height_initialized
        object_translation_delta = torch.where(
            initialized[:, None],
            current_object_position - self._previous_object_position,
            0.0,
        )
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
        (
            previous_requested_xy_norm,
            previous_requested_z,
            realized_object_xy_progress,
            realized_object_z_progress,
        ) = realized_translation_progress(
            object_translation_delta,
            self._previous_requested_translation,
        )
        realized_xy = previously_active & (previous_requested_xy_norm > 1.0e-9)
        realized_z = previously_active & (previous_requested_z > 1.0e-9)
        self.assist_realized_requested_xy_step[realized_xy] = (
            previous_requested_xy_norm[realized_xy]
        )
        self.assist_realized_requested_z_step[realized_z] = previous_requested_z[
            realized_z
        ]
        self.assist_realized_object_xy_progress[realized_xy] = (
            realized_object_xy_progress[realized_xy]
        )
        self.assist_realized_object_z_progress[realized_z] = (
            realized_object_z_progress[realized_z]
        )
        self._previous_requested_vertical_step[:] = 0.0
        self._previous_requested_translation[:] = 0.0
        effort_limits = torch.clamp(
            robot.data.joint_effort_limits[:, self._joint_ids], min=1.0e-6
        )
        effort_demand_ratio = torch.amax(
            torch.abs(robot.data.computed_torque[:, self._joint_ids]) / effort_limits, dim=1
        )
        self.assist_effort_demand_ratio[previously_active] = effort_demand_ratio[
            previously_active
        ]
        self._previous_object_height[:] = current_object_height
        self._previous_tcp_height[:] = current_tcp_height
        self._previous_object_position[:] = current_object_position
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
            & ~imminent_automatic_release
            & ~state.release_commanded
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
        release_handoff = (deactivated & state.released) | state.just_released
        self._handoff_steps_remaining[release_handoff] = self.cfg.handoff_steps
        self._assist_joint_target[release_handoff] = current_joint_pos[release_handoff]
        newly_active = assist_active & ~self._assist_was_active
        entry_activation = newly_active & ~self._assist_entry_joint_limit_recorded
        entry_activation_ids = torch.nonzero(
            entry_activation, as_tuple=False
        ).squeeze(-1)
        if entry_activation_ids.numel() > 0:
            entry_joint_limits = self._asset.data.soft_joint_pos_limits[
                entry_activation_ids
            ][:, self._joint_ids]
            entry_joint_position = current_joint_pos[entry_activation_ids]
            entry_joint_span = torch.clamp(
                entry_joint_limits[:, :, 1] - entry_joint_limits[:, :, 0],
                min=1.0e-6,
            )
            entry_joint_margin = torch.minimum(
                entry_joint_position - entry_joint_limits[:, :, 0],
                entry_joint_limits[:, :, 1] - entry_joint_position,
            ) / entry_joint_span
            self.assist_entry_joint_limit_proximity[entry_activation_ids] = torch.clamp(
                1.0 - 2.0 * torch.amin(entry_joint_margin, dim=1),
                min=0.0,
                max=1.0,
            )
            self._assist_entry_joint_limit_recorded[entry_activation_ids] = True
        self._assist_posture_ref[newly_active] = current_joint_pos[newly_active]
        self._assist_entry_offset[newly_active] = (
            obj.data.root_pos_w[newly_active] - robot.data.body_pos_w[newly_active, self._body_id]
        )
        self._assist_entry_orientation[newly_active] = robot.data.body_quat_w[
            newly_active, self._body_id
        ]
        self._slip_streak[newly_active] = 0
        # The assist owns the arm immediately, but carries only measured goal-directed
        # object velocity into its bounded Cartesian controller. Raw actor targets stay out.
        self._assist_joint_target[newly_active] = current_joint_pos[newly_active]
        self._assist_entry_velocity[newly_active] = self._filtered_object_velocity[
            newly_active
        ]
        self._assist_command_velocity[newly_active] = self._filtered_object_velocity[
            newly_active
        ]
        self._assist_command_acceleration[newly_active] = 0.0
        self._assist_entry_blend_steps_remaining[newly_active] = (
            self.cfg.assist_entry_blend_steps
        )
        self._slip_recovery_steps_remaining[newly_active] = 0
        self._handoff_steps_remaining[assist_active] = 0
        self._assist_was_active[:] = assist_active
        self._slip_streak[~assist_active] = 0
        inactive_assist = ~assist_active
        self._assist_entry_velocity[inactive_assist] = 0.0
        self._assist_command_velocity[inactive_assist] = 0.0
        self._assist_command_acceleration[inactive_assist] = 0.0
        self._assist_entry_blend_steps_remaining[inactive_assist] = 0
        self._slip_recovery_steps_remaining[inactive_assist] = 0

        # Keep the actor's absolute target bounded throughout post-grasp transport. The
        # old finite handoff expired after 0.4 s and exposed the raw target mid-carry.
        pre_assist_slew = (
            state.grasp_acquired
            & ~state.assist_engaged
            & ~state.released
            & ~state.failed
        )
        post_assist_slew = state.assist_engaged & ~assist_active & ~hold_recovery
        policy_slew = pre_assist_slew | post_assist_slew
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
        release_hold = (
            imminent_automatic_release | state.release_commanded | state.released
        )
        self._assist_joint_target[release_hold] = current_joint_pos[release_hold]
        self._processed_actions[release_hold] = current_joint_pos[release_hold]
        self._handoff_steps_remaining[release_hold] = 0
        state.assist_owns_arm[:] = (
            grasp_control
            | grasp_handoff
            | assist_active
            | hold_recovery
            | handoff
            | release_hold
        )
        # Only automatic pose holds own the arm for reward purposes. During
        # verification/handoff the bounded target still comes from the actor.
        state.grasp_controller_owns_arm[:] = grasp_control
        state.grasp_retry_recovery_active[:] = retry_recovery

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
        self._assist_entry_velocity[slip_watchdog_ids] = 0.0
        self._assist_command_velocity[slip_watchdog_ids] = 0.0
        self._assist_command_acceleration[slip_watchdog_ids] = 0.0
        self._assist_entry_blend_steps_remaining[slip_watchdog_ids] = (
            self.cfg.assist_entry_blend_steps
        )
        self._slip_recovery_steps_remaining[slip_watchdog_ids] = (
            self.cfg.slip_recovery_hold_steps
        )

        nominal_step = torch.zeros((assist_ids.numel(), 3), device=self.device)
        command = self._env.command_manager.get_command(self.cfg.command_name)
        goal_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
        target_object_position = goal_pos_w[assist_ids, :3].clone()
        target_object_position[:, 2] = state.descent_target_height[assist_ids]
        object_error = target_object_position - obj.data.root_pos_w[assist_ids]
        filtered_velocity = self._filtered_object_velocity[assist_ids]
        nominal_step[:, :2] = self._bounded_direction_step(
            object_error[:, :2],
            filtered_velocity[:, :2],
            self.cfg.horizontal_position_gain,
            self.cfg.horizontal_velocity_damping,
            self._env.step_dt,
            self.cfg.max_horizontal_step,
            self.cfg.horizontal_position_deadband,
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
            torch.abs(vertical_error) > self.cfg.vertical_position_deadband,
            vertical_magnitude,
            0.0,
        )
        nominal_step[:, 2:3] = vertical_direction * vertical_magnitude

        step_dt = self._env.step_dt
        bounded_entry_velocity = apply_stopping_speed_envelope(
            self._assist_entry_velocity[assist_ids],
            object_error,
            self.cfg.horizontal_position_deadband,
            self.cfg.vertical_position_deadband,
            self.cfg.max_horizontal_acceleration,
            self.cfg.max_vertical_acceleration,
            self.cfg.max_horizontal_jerk,
            self.cfg.max_vertical_jerk,
        )
        entry_step = preserve_safe_goal_directed_step(
            bounded_entry_velocity * step_dt,
            object_error,
            self.cfg.max_horizontal_step,
            self.cfg.max_upward_step,
            self.cfg.max_vertical_step,
            self.cfg.horizontal_position_deadband,
            self.cfg.vertical_position_deadband,
        )
        entry_velocity = entry_step / step_dt
        fresh_local = newly_active[assist_ids]
        self._assist_entry_velocity[assist_ids] = entry_velocity
        self._assist_command_velocity[assist_ids] = torch.where(
            fresh_local[:, None],
            entry_velocity,
            self._assist_command_velocity[assist_ids],
        )
        self._assist_command_acceleration[assist_ids] = torch.where(
            fresh_local[:, None],
            torch.zeros_like(entry_velocity),
            self._assist_command_acceleration[assist_ids],
        )

        blend_remaining = self._assist_entry_blend_steps_remaining[assist_ids]
        if self.cfg.assist_entry_blend_steps > 0:
            blend_progress = (
                self.cfg.assist_entry_blend_steps - blend_remaining
            ).float() / self.cfg.assist_entry_blend_steps
            blend_alpha = quintic_smoothstep(blend_progress)
        else:
            blend_alpha = torch.ones_like(blend_remaining, dtype=nominal_step.dtype)
        blend_active = blend_remaining > 0
        nominal_velocity = nominal_step / step_dt
        desired_velocity = torch.where(
            blend_active[:, None],
            (1.0 - blend_alpha[:, None]) * entry_velocity
            + blend_alpha[:, None] * nominal_velocity,
            nominal_velocity,
        )
        desired_velocity = apply_stopping_speed_envelope(
            desired_velocity,
            object_error,
            self.cfg.horizontal_position_deadband,
            self.cfg.vertical_position_deadband,
            self.cfg.max_horizontal_acceleration,
            self.cfg.max_vertical_acceleration,
            self.cfg.max_horizontal_jerk,
            self.cfg.max_vertical_jerk,
        )
        recovery_hold = self._slip_recovery_steps_remaining[assist_ids] > 0
        desired_velocity[recovery_hold] = 0.0
        previous_velocity = self._assist_command_velocity[assist_ids].clone()
        limited_velocity, limited_acceleration = jerk_limited_cartesian_velocity(
            previous_velocity,
            self._assist_command_acceleration[assist_ids],
            desired_velocity,
            step_dt,
            self.cfg.max_horizontal_acceleration,
            self.cfg.max_vertical_acceleration,
            self.cfg.max_horizontal_jerk,
            self.cfg.max_vertical_jerk,
        )
        delta_position = preserve_safe_goal_directed_step(
            limited_velocity * step_dt,
            object_error,
            self.cfg.max_horizontal_step,
            self.cfg.max_upward_step,
            self.cfg.max_vertical_step,
            self.cfg.horizontal_position_deadband,
            self.cfg.vertical_position_deadband,
        )
        delta_position[recovery_hold] = 0.0
        applied_velocity = delta_position / step_dt
        applied_acceleration = (applied_velocity - previous_velocity) / step_dt
        applied_acceleration[recovery_hold] = 0.0
        safety_override = torch.norm(applied_velocity - limited_velocity, dim=1) > 1.0e-6
        self._assist_command_velocity[assist_ids] = applied_velocity
        self._assist_command_acceleration[assist_ids] = torch.where(
            safety_override[:, None],
            torch.zeros_like(limited_acceleration),
            limited_acceleration,
        )
        counted_blend = blend_active & ~recovery_hold
        self._assist_entry_blend_steps_remaining[assist_ids[counted_blend]] -= 1
        self._slip_recovery_steps_remaining[assist_ids[recovery_hold]] -= 1
        self.assist_entry_blend_progress[assist_ids] = torch.where(
            blend_active,
            blend_alpha,
            1.0,
        )
        self.assist_command_speed[assist_ids] = torch.norm(applied_velocity, dim=1)
        self.assist_command_acceleration[assist_ids] = torch.norm(
            applied_acceleration, dim=1
        )
        xy_error_active = (
            torch.norm(object_error[:, :2], dim=1)
            > self.cfg.horizontal_position_deadband
        )
        z_error_active = (
            torch.abs(object_error[:, 2]) > self.cfg.vertical_position_deadband
        )
        self.assist_xy_active[assist_ids] = xy_error_active.float()
        self.assist_z_active[assist_ids] = z_error_active.float()
        self.assist_recovery_hold[assist_ids] = recovery_hold.float()
        self.assist_safety_brake[assist_ids] = safety_override.float()
        self.assist_requested_vertical_step[assist_ids] = delta_position[:, 2]
        self._previous_requested_vertical_step[assist_ids] = delta_position[:, 2]
        self._previous_requested_translation[assist_ids] = delta_position
        requested_xy_norm = torch.norm(delta_position[:, :2], dim=1)
        requested_z = torch.abs(delta_position[:, 2])
        requested_xy_active = requested_xy_norm > 1.0e-9
        requested_z_active = requested_z > 1.0e-9
        requested_motion = requested_xy_active | requested_z_active
        prefer_z_progress = state.post_center_descent_z_priority[assist_ids]

        delta_twist = torch.zeros((assist_ids.numel(), 6), device=self.device)
        delta_twist[:, :3] = delta_position
        phase_primary_xy, phase_primary_z = select_phase_primary_axis(
            requested_xy_active,
            requested_z_active,
            prefer_z_progress,
        )
        primary_row_mask = torch.zeros_like(delta_twist, dtype=torch.bool)
        primary_row_mask[:, :2] = phase_primary_xy[:, None]
        primary_row_mask[:, 2] = phase_primary_z
        solver = build_prioritized_dls_solver(
            task_jacobian,
            primary_row_mask,
            self.cfg.damping,
        )
        primary_pinv, primary_nullspace, _, _ = solver
        joint_limits = self._asset.data.soft_joint_pos_limits[assist_ids][:, self._joint_ids]

        # Seed only newly-entered environments; steady assist avoids this extra solve.
        fresh_local_ids = torch.nonzero(fresh_local, as_tuple=False).squeeze(-1)
        if fresh_local_ids.numel() > 0:
            entry_target_lead = preserve_safe_goal_directed_step(
                entry_step[fresh_local_ids] * self.cfg.assist_entry_target_lead_steps,
                object_error[fresh_local_ids],
                self.cfg.max_horizontal_target_lead,
                self.cfg.max_upward_target_lead,
                self.cfg.max_downward_target_lead,
                self.cfg.horizontal_position_deadband,
                self.cfg.vertical_position_deadband,
            )
            entry_target_twist = torch.zeros(
                (fresh_local_ids.numel(), 6), device=self.device
            )
            entry_target_twist[:, :3] = entry_target_lead
            entry_active_xy = (
                torch.norm(entry_target_lead[:, :2], dim=1) > 1.0e-9
            )
            entry_active_z = torch.abs(entry_target_lead[:, 2]) > 1.0e-9
            entry_primary_xy, entry_primary_z = select_phase_primary_axis(
                entry_active_xy,
                entry_active_z,
                prefer_z_progress[fresh_local_ids],
            )
            entry_primary_mask = torch.zeros_like(
                entry_target_twist, dtype=torch.bool
            )
            entry_primary_mask[:, :2] = entry_primary_xy[:, None]
            entry_primary_mask[:, 2] = entry_primary_z
            entry_solver = build_prioritized_dls_solver(
                task_jacobian[fresh_local_ids],
                entry_primary_mask,
                self.cfg.damping,
            )
            entry_joint_lead = solve_prioritized_dls_delta(
                entry_target_twist,
                task_jacobian[fresh_local_ids],
                entry_primary_mask,
                entry_solver,
                self.cfg.max_target_lead,
            )
            entry_current = current_joint_pos[assist_ids][fresh_local_ids]
            entry_joint_lead = constrain_target_lead(
                entry_joint_lead,
                entry_current,
                entry_current,
                joint_limits[fresh_local_ids],
                self.cfg.max_target_lead,
                self.cfg.max_joint_step,
            )
            predicted_entry_twist = (
                task_jacobian[fresh_local_ids] @ entry_joint_lead.unsqueeze(-1)
            ).squeeze(-1)
            entry_xy_limit = torch.clamp(
                torch.norm(object_error[fresh_local_ids, :2], dim=1)
                - self.cfg.horizontal_position_deadband,
                min=0.0,
                max=self.cfg.max_horizontal_target_lead,
            )
            entry_z_limit = torch.clamp(
                torch.abs(object_error[fresh_local_ids, 2])
                - self.cfg.vertical_position_deadband,
                min=0.0,
            )
            entry_vertical_limit = torch.where(
                object_error[fresh_local_ids, 2] >= 0.0,
                self.cfg.max_upward_target_lead,
                self.cfg.max_downward_target_lead,
            )
            entry_z_limit = torch.minimum(entry_z_limit, entry_vertical_limit)
            entry_xy_safe, entry_z_safe = self._cartesian_axis_safe(
                predicted_entry_twist[:, :3],
                object_error[fresh_local_ids],
                entry_xy_limit,
                entry_z_limit,
                entry_z_limit,
                self.cfg.cartesian_hard_leak_tolerance,
            )
            entry_progress_valid, _, _ = phase_primary_axis_progress_valid(
                predicted_entry_twist[:, :3],
                entry_target_lead,
                entry_active_xy,
                entry_active_z,
                prefer_z_progress[fresh_local_ids],
                self.cfg.minimum_active_progress_ratio,
                self.cfg.minimum_companion_progress_ratio,
            )
            entry_valid = (
                entry_xy_safe
                & entry_z_safe
                & entry_progress_valid
                & (
                    torch.norm(predicted_entry_twist[:, 3:], dim=1)
                    <= self.cfg.max_angular_target_lead + 1.0e-6
                )
                & target_lead_constraints_valid(
                    entry_joint_lead,
                    entry_current,
                    entry_current,
                    joint_limits[fresh_local_ids],
                    self.cfg.max_target_lead,
                    self.cfg.max_joint_step,
                )
            )
            entry_joint_lead = torch.where(
                entry_valid[:, None],
                entry_joint_lead,
                torch.zeros_like(entry_joint_lead),
            )
            fresh_ids = assist_ids[fresh_local_ids]
            self._assist_joint_target[fresh_ids] = (
                current_joint_pos[fresh_ids] + entry_joint_lead
            )
            self.assist_entry_seeded[fresh_ids] = (
                entry_valid & (torch.norm(entry_joint_lead, dim=1) > 1.0e-9)
            ).float()

        prioritized_delta = solve_prioritized_dls_delta(
            delta_twist,
            task_jacobian,
            primary_row_mask,
            solver,
            self.cfg.max_joint_step,
        )
        active_only_delta = solve_primary_dls_delta(
            delta_twist,
            task_jacobian,
            primary_row_mask,
            primary_pinv,
            self.cfg.max_joint_step,
        )
        posture_error = self._assist_posture_ref[assist_ids] - current_joint_pos[assist_ids]
        posture_delta = (
            primary_nullspace
            @ (self.cfg.nullspace_gain * posture_error).unsqueeze(-1)
        ).squeeze(-1)
        posture_delta = self._scale_by_max_abs(posture_delta, self.cfg.max_nullspace_step)
        posture_delta[recovery_hold] = 0.0
        postured_delta = self._scale_by_max_abs(
            prioritized_delta + posture_delta,
            self.cfg.max_joint_step,
        )
        self.assist_posture_error[assist_ids] = torch.norm(posture_error, dim=1)

        prioritized_delta = self._scale_to_joint_limits(
            current_joint_pos[assist_ids], prioritized_delta, joint_limits
        )
        active_only_delta = self._scale_to_joint_limits(
            current_joint_pos[assist_ids], active_only_delta, joint_limits
        )
        postured_delta = self._scale_to_joint_limits(
            current_joint_pos[assist_ids], postured_delta, joint_limits
        )

        def evaluate_delta(
            candidate_delta: torch.Tensor,
            require_xy: torch.Tensor,
            require_z: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            candidate_prediction = (
                task_jacobian @ candidate_delta.unsqueeze(-1)
            ).squeeze(-1)
            candidate_xy_safe, candidate_z_safe = self._cartesian_axis_safe(
                candidate_prediction[:, :3],
                object_error,
                requested_xy_norm,
                requested_z,
                requested_z,
                self.cfg.cartesian_hard_leak_tolerance,
            )
            candidate_progress_valid, _, _ = phase_primary_axis_progress_valid(
                candidate_prediction[:, :3],
                delta_position,
                require_xy,
                require_z,
                prefer_z_progress,
                self.cfg.minimum_active_progress_ratio,
                self.cfg.minimum_companion_progress_ratio,
            )
            candidate_valid = (
                candidate_xy_safe
                & candidate_z_safe
                & candidate_progress_valid
                & torch.all(torch.isfinite(candidate_delta), dim=1)
            )
            candidate_ratio = phase_primary_translation_progress_ratio(
                candidate_prediction[:, :3],
                delta_position,
                require_xy,
                require_z,
                prefer_z_progress,
            )
            return candidate_prediction, candidate_valid, candidate_ratio

        def repair_delta(
            candidate_delta: torch.Tensor,
            candidate_prediction: torch.Tensor,
            candidate_valid: torch.Tensor,
            candidate_ratio: torch.Tensor,
            require_xy: torch.Tensor,
            require_z: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            scaled_delta, _, scaled_feasible = scale_delta_to_cartesian_bounds(
                candidate_delta,
                candidate_prediction[:, :3],
                object_error,
                delta_position,
                require_xy,
                require_z,
                requested_xy_norm,
                requested_z,
                requested_z,
                self.cfg.horizontal_position_deadband,
                self.cfg.vertical_position_deadband,
                self.cfg.cartesian_hard_leak_tolerance,
                self.cfg.minimum_active_progress_ratio,
                prefer_z_progress,
                self.cfg.minimum_companion_progress_ratio,
            )
            scaled_prediction, scaled_valid, scaled_ratio = evaluate_delta(
                scaled_delta, require_xy, require_z
            )
            scaled_valid &= scaled_feasible
            use_scaled = requested_motion & ~candidate_valid & scaled_valid
            return (
                torch.where(use_scaled[:, None], scaled_delta, candidate_delta),
                torch.where(
                    use_scaled[:, None], scaled_prediction, candidate_prediction
                ),
                candidate_valid | use_scaled,
                torch.where(use_scaled, scaled_ratio, candidate_ratio),
                use_scaled,
            )

        (
            prioritized_raw_prediction,
            prioritized_valid,
            prioritized_ratio,
        ) = evaluate_delta(
            prioritized_delta, requested_xy_active, requested_z_active
        )
        (
            prioritized_delta,
            prioritized_prediction,
            prioritized_valid,
            prioritized_ratio,
            prioritized_scaled,
        ) = repair_delta(
            prioritized_delta,
            prioritized_raw_prediction,
            prioritized_valid,
            prioritized_ratio,
            requested_xy_active,
            requested_z_active,
        )
        active_raw_prediction, active_valid, active_ratio = evaluate_delta(
            active_only_delta, requested_xy_active, requested_z_active
        )
        (
            active_only_delta,
            active_prediction,
            active_valid,
            active_ratio,
            active_scaled,
        ) = repair_delta(
            active_only_delta,
            active_raw_prediction,
            active_valid,
            active_ratio,
            requested_xy_active,
            requested_z_active,
        )
        use_active_only = active_valid & (
            ~prioritized_valid | (active_ratio > prioritized_ratio)
        )
        selected_delta = torch.where(
            use_active_only[:, None], active_only_delta, prioritized_delta
        )
        selected_prediction = torch.where(
            use_active_only[:, None], active_prediction, prioritized_prediction
        )
        selected_valid = torch.where(
            use_active_only, active_valid, prioritized_valid
        )
        selected_ratio = torch.where(
            use_active_only, active_ratio, prioritized_ratio
        )
        selected_raw_prediction = torch.where(
            use_active_only[:, None],
            active_raw_prediction,
            prioritized_raw_prediction,
        )
        selected_scaled_repair = torch.where(
            use_active_only, active_scaled, prioritized_scaled
        )

        postured_raw_prediction, postured_valid, postured_ratio = evaluate_delta(
            postured_delta, requested_xy_active, requested_z_active
        )
        (
            postured_delta,
            postured_prediction,
            postured_valid,
            postured_ratio,
            postured_scaled,
        ) = repair_delta(
            postured_delta,
            postured_raw_prediction,
            postured_valid,
            postured_ratio,
            requested_xy_active,
            requested_z_active,
        )
        use_posture = (
            requested_motion
            & postured_valid
            & (
                ~selected_valid
                | (postured_ratio >= 0.95 * selected_ratio)
            )
        )
        selected_delta = torch.where(
            use_posture[:, None], postured_delta, selected_delta
        )
        selected_prediction = torch.where(
            use_posture[:, None], postured_prediction, selected_prediction
        )
        selected_valid = torch.where(use_posture, postured_valid, selected_valid)
        selected_ratio = torch.where(use_posture, postured_ratio, selected_ratio)
        selected_raw_prediction = torch.where(
            use_posture[:, None], postured_raw_prediction, selected_raw_prediction
        )
        selected_scaled_repair = torch.where(
            use_posture, postured_scaled, selected_scaled_repair
        )

        raw_xy_safe, raw_z_safe = self._cartesian_axis_safe(
            selected_raw_prediction[:, :3],
            object_error,
            requested_xy_norm,
            requested_z,
            requested_z,
            self.cfg.cartesian_hard_leak_tolerance,
        )
        self.assist_incremental_xy_reject[assist_ids] = (
            requested_motion & ~raw_xy_safe
        ).float()
        self.assist_incremental_z_reject[assist_ids] = (
            requested_motion & ~raw_z_safe
        ).float()
        raw_xy_progress_ratio, raw_z_progress_ratio = translation_progress_ratios(
            selected_raw_prediction[:, :3],
            delta_position,
            requested_xy_active,
            requested_z_active,
        )
        raw_primary_xy, raw_primary_z = select_phase_primary_axis(
            requested_xy_active,
            requested_z_active,
            prefer_z_progress,
        )
        raw_xy_progress_floor = torch.where(
            raw_primary_xy,
            self.cfg.minimum_active_progress_ratio,
            self.cfg.minimum_companion_progress_ratio,
        )
        raw_z_progress_floor = torch.where(
            raw_primary_z,
            self.cfg.minimum_active_progress_ratio,
            self.cfg.minimum_companion_progress_ratio,
        )
        raw_xy_low_progress = requested_xy_active & (
            raw_xy_progress_ratio < raw_xy_progress_floor
        )
        raw_z_low_progress = requested_z_active & (
            raw_z_progress_ratio < raw_z_progress_floor
        )
        self.assist_low_progress[assist_ids] = (
            requested_motion & (raw_xy_low_progress | raw_z_low_progress)
        ).float()
        self.assist_xy_low_progress[assist_ids] = raw_xy_low_progress.float()
        self.assist_z_low_progress[assist_ids] = raw_z_low_progress.float()

        use_scaled_repair = selected_scaled_repair
        unresolved = requested_motion & ~selected_valid
        selected_required_xy = requested_xy_active.clone()
        selected_required_z = requested_z_active.clone()

        def solve_single_axis_candidate(
            candidate_mask: torch.Tensor,
            use_xy_axis: bool,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            required_xy = (
                candidate_mask if use_xy_axis else torch.zeros_like(candidate_mask)
            )
            required_z = (
                torch.zeros_like(candidate_mask) if use_xy_axis else candidate_mask
            )
            if use_xy_axis:
                axis_jacobian = task_jacobian[:, :2, :]
                axis_target = delta_position[:, :2]
            else:
                axis_jacobian = task_jacobian[:, 2:3, :]
                axis_target = delta_position[:, 2:3]
            axis_target = torch.where(
                candidate_mask[:, None], axis_target, torch.zeros_like(axis_target)
            )

            axis_pinv = damped_pseudoinverse(axis_jacobian, self.cfg.damping)
            local_delta = (axis_pinv @ axis_target.unsqueeze(-1)).squeeze(-1)
            for _ in range(2):
                residual = axis_target - (
                    axis_jacobian @ local_delta.unsqueeze(-1)
                ).squeeze(-1)
                local_delta = local_delta + (
                    axis_pinv @ residual.unsqueeze(-1)
                ).squeeze(-1)
            local_delta = self._scale_by_max_abs(
                local_delta, self.cfg.max_joint_step
            )
            local_prediction = (
                task_jacobian @ local_delta.unsqueeze(-1)
            ).squeeze(-1)
            maximum_increment_rotation = self.cfg.max_angular_target_lead / max(
                self.cfg.assist_target_lead_steps, 1
            )
            angular_scale = torch.clamp(
                maximum_increment_rotation
                / torch.clamp(torch.norm(local_prediction[:, 3:], dim=1), min=1.0e-9),
                max=1.0,
            )
            local_delta = local_delta * angular_scale[:, None]
            local_delta = self._scale_to_joint_limits(
                current_joint_pos[assist_ids],
                local_delta,
                joint_limits,
            )
            local_prediction = (
                task_jacobian @ local_delta.unsqueeze(-1)
            ).squeeze(-1)
            local_delta, _, local_feasible = scale_delta_to_cartesian_bounds(
                local_delta,
                local_prediction[:, :3],
                object_error,
                delta_position,
                required_xy,
                required_z,
                requested_xy_norm,
                requested_z,
                requested_z,
                self.cfg.horizontal_position_deadband,
                self.cfg.vertical_position_deadband,
                self.cfg.cartesian_hard_leak_tolerance,
                self.cfg.minimum_active_progress_ratio,
            )
            local_prediction = (
                task_jacobian @ local_delta.unsqueeze(-1)
            ).squeeze(-1)
            local_xy_safe, local_z_safe = self._cartesian_axis_safe(
                local_prediction[:, :3],
                object_error,
                requested_xy_norm,
                requested_z,
                requested_z,
                self.cfg.cartesian_hard_leak_tolerance,
            )
            local_progress_valid, _, _ = phase_primary_axis_progress_valid(
                local_prediction[:, :3],
                delta_position,
                required_xy,
                required_z,
                prefer_z_progress,
                self.cfg.minimum_active_progress_ratio,
                self.cfg.minimum_companion_progress_ratio,
            )
            local_valid = (
                candidate_mask
                & local_feasible
                & local_xy_safe
                & local_z_safe
                & local_progress_valid
                & torch.all(torch.isfinite(local_delta), dim=1)
            )
            local_ratio = phase_primary_translation_progress_ratio(
                local_prediction[:, :3],
                delta_position,
                required_xy,
                required_z,
                prefer_z_progress,
            )
            return local_delta, local_valid, local_ratio

        xy_fallback_delta, xy_fallback_valid, xy_fallback_ratio = (
            solve_single_axis_candidate(unresolved & requested_xy_active, True)
        )
        z_priority_candidate = prefer_z_progress & requested_z_active
        z_fallback_delta, z_fallback_valid, z_fallback_ratio = (
            solve_single_axis_candidate(
                (unresolved & requested_z_active) | z_priority_candidate,
                False,
            )
        )
        use_xy_fallback, use_z_fallback = select_phase_axis_fallback(
            unresolved,
            xy_fallback_valid,
            z_fallback_valid,
            prefer_z_progress,
        )
        unresolved_z_fallback = use_z_fallback.clone()
        z_priority_upgrade = select_higher_phase_primary_candidate(
            selected_valid,
            selected_ratio,
            z_fallback_valid,
            z_fallback_ratio,
            z_priority_candidate,
        )
        use_z_fallback |= z_priority_upgrade
        use_xy_fallback &= ~use_z_fallback
        selected_delta = torch.where(
            use_z_fallback[:, None], z_fallback_delta, selected_delta
        )
        selected_delta = torch.where(
            use_xy_fallback[:, None], xy_fallback_delta, selected_delta
        )
        selected_prediction = (
            task_jacobian @ selected_delta.unsqueeze(-1)
        ).squeeze(-1)
        selected_ratio = torch.where(
            use_z_fallback, z_fallback_ratio, selected_ratio
        )
        selected_ratio = torch.where(
            use_xy_fallback, xy_fallback_ratio, selected_ratio
        )
        fallback_used = use_xy_fallback | use_z_fallback
        use_scaled_repair &= ~fallback_used
        selected_required_xy = (
            selected_required_xy & ~fallback_used
        ) | use_xy_fallback
        selected_required_z = (
            selected_required_z & ~fallback_used
        ) | use_z_fallback
        selected_valid |= use_xy_fallback | use_z_fallback
        self.assist_incremental_scaled_repair[assist_ids] = (
            use_scaled_repair.float()
        )
        self.assist_incremental_xy_fallback[assist_ids] = (
            use_xy_fallback.float()
        )
        self.assist_incremental_z_fallback[assist_ids] = (
            unresolved_z_fallback.float()
        )
        self.assist_z_priority[assist_ids] = (
            prefer_z_progress & requested_motion
        ).float()
        self.assist_incremental_z_priority_upgrade[assist_ids] = (
            z_priority_upgrade.float()
        )

        delta_joint = torch.where(
            selected_valid[:, None], selected_delta, torch.zeros_like(selected_delta)
        )
        predicted_increment_twist = (
            task_jacobian @ delta_joint.unsqueeze(-1)
        ).squeeze(-1)
        increment_xy_hard_safe, increment_z_hard_safe = self._cartesian_axis_safe(
            predicted_increment_twist[:, :3],
            object_error,
            requested_xy_norm,
            requested_z,
            requested_z,
            self.cfg.cartesian_hard_leak_tolerance,
        )
        increment_progress_valid, _, _ = phase_primary_axis_progress_valid(
            predicted_increment_twist[:, :3],
            delta_position,
            selected_required_xy,
            selected_required_z,
            prefer_z_progress,
            self.cfg.minimum_active_progress_ratio,
            self.cfg.minimum_companion_progress_ratio,
        )
        increment_valid = (
            selected_valid
            & increment_xy_hard_safe
            & increment_z_hard_safe
            & increment_progress_valid
            & torch.all(torch.isfinite(delta_joint), dim=1)
        )
        delta_joint = torch.where(
            increment_valid[:, None], delta_joint, torch.zeros_like(delta_joint)
        )
        predicted_increment_twist = (
            task_jacobian @ delta_joint.unsqueeze(-1)
        ).squeeze(-1)
        xy_progress_ratio, z_progress_ratio = translation_progress_ratios(
            predicted_increment_twist[:, :3],
            delta_position,
            requested_xy_active,
            requested_z_active,
        )
        progress_ratio = phase_primary_translation_progress_ratio(
            predicted_increment_twist[:, :3],
            delta_position,
            selected_required_xy,
            selected_required_z,
            prefer_z_progress,
        )
        progress_ratio = torch.where(
            requested_motion,
            progress_ratio,
            torch.ones_like(progress_ratio),
        )
        self.assist_active_progress_ratio[assist_ids] = torch.where(
            requested_motion,
            torch.clamp(progress_ratio, min=0.0, max=2.0),
            0.0,
        )
        self.assist_xy_progress_ratio[assist_ids] = torch.where(
            requested_xy_active,
            torch.clamp(xy_progress_ratio, min=0.0, max=2.0),
            0.0,
        )
        self.assist_z_progress_ratio[assist_ids] = torch.where(
            requested_z_active,
            torch.clamp(z_progress_ratio, min=0.0, max=2.0),
            0.0,
        )
        self.assist_incremental_safety_stop[assist_ids] = (
            requested_motion & ~increment_valid
        ).float()
        self.assist_incremental_vertical_step[assist_ids] = (
            predicted_increment_twist[:, 2]
        )

        assist_current = current_joint_pos[assist_ids]
        previous_target = self._assist_joint_target[assist_ids].clone()
        _, _, lead_constraint_feasible = target_lead_constraint_intersection(
            assist_current,
            previous_target,
            joint_limits,
            self.cfg.max_target_lead,
            self.cfg.max_joint_step,
        )
        self.assist_lead_constraint_rebase[assist_ids] = (
            ~lead_constraint_feasible
        ).float()
        self.assist_target_backlog[assist_ids] = torch.amax(
            torch.abs(previous_target - assist_current), dim=1
        )
        joint_span = torch.clamp(
            joint_limits[:, :, 1] - joint_limits[:, :, 0], min=1.0e-6
        )
        current_joint_margin = torch.minimum(
            assist_current - joint_limits[:, :, 0],
            joint_limits[:, :, 1] - assist_current,
        ) / joint_span
        self.assist_current_joint_limit_proximity[assist_ids] = torch.clamp(
            1.0 - 2.0 * torch.amin(current_joint_margin, dim=1),
            min=0.0,
            max=1.0,
        )
        lead_xy_error = torch.norm(object_error[:, :2], dim=1)
        lead_z_error = torch.abs(object_error[:, 2])
        allowed_lead_xy = torch.clamp(
            lead_xy_error - self.cfg.horizontal_position_deadband,
            min=0.0,
            max=self.cfg.max_horizontal_target_lead,
        )
        vertical_target_lead = torch.where(
            object_error[:, 2] >= 0.0,
            self.cfg.max_upward_target_lead,
            self.cfg.max_downward_target_lead,
        )
        allowed_lead_z = torch.minimum(
            torch.clamp(
                lead_z_error - self.cfg.vertical_position_deadband,
                min=0.0,
            ),
            vertical_target_lead,
        )

        def repair_lead(
            raw_lead: torch.Tensor,
            eligible: torch.Tensor,
            require_xy: torch.Tensor,
            require_z: torch.Tensor,
            allow_stabilization: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            repaired_all = torch.zeros_like(raw_lead)
            valid_all = torch.zeros_like(eligible)
            candidate_ids = torch.nonzero(
                eligible & lead_constraint_feasible, as_tuple=False
            ).squeeze(-1)
            if candidate_ids.numel() == 0:
                return repaired_all, valid_all

            current = current_joint_pos[assist_ids][candidate_ids]
            previous = previous_target[candidate_ids]
            limits = joint_limits[candidate_ids]
            candidate = constrain_target_lead(
                raw_lead[candidate_ids],
                current,
                previous,
                limits,
                self.cfg.max_target_lead,
                self.cfg.max_joint_step,
            )
            candidate_twist = (
                task_jacobian[candidate_ids] @ candidate.unsqueeze(-1)
            ).squeeze(-1)
            candidate_xy_safe, candidate_z_safe = self._cartesian_axis_safe(
                candidate_twist[:, :3],
                object_error[candidate_ids],
                allowed_lead_xy[candidate_ids],
                allowed_lead_z[candidate_ids],
                allowed_lead_z[candidate_ids],
                self.cfg.cartesian_hard_leak_tolerance,
            )
            if allow_stabilization:
                candidate_progress_valid = torch.all(
                    torch.isfinite(candidate_twist[:, :3]), dim=1
                )
            else:
                candidate_progress_valid, _, _ = phase_primary_axis_progress_valid(
                    candidate_twist[:, :3],
                    delta_position[candidate_ids],
                    require_xy[candidate_ids],
                    require_z[candidate_ids],
                    prefer_z_progress[candidate_ids],
                    self.cfg.minimum_active_progress_ratio,
                    self.cfg.minimum_companion_progress_ratio,
                )
            candidate_valid = (
                candidate_xy_safe
                & candidate_z_safe
                & candidate_progress_valid
                & (
                    torch.norm(candidate_twist[:, 3:], dim=1)
                    <= self.cfg.max_angular_target_lead + 1.0e-6
                )
                & target_lead_constraints_valid(
                    candidate,
                    current,
                    previous,
                    limits,
                    self.cfg.max_target_lead,
                    self.cfg.max_joint_step,
                )
            )
            repaired_all[candidate_ids] = candidate
            valid_all[candidate_ids] = candidate_valid

            invalid_local_ids = torch.nonzero(
                ~candidate_valid, as_tuple=False
            ).squeeze(-1)
            if invalid_local_ids.numel() == 0:
                return repaired_all, valid_all

            repair_ids = candidate_ids[invalid_local_ids]
            invalid_candidate = candidate[invalid_local_ids]
            invalid_twist = candidate_twist[invalid_local_ids]
            safe_twist = invalid_twist.clone()
            safe_twist[:, :3] = preserve_safe_goal_directed_step(
                invalid_twist[:, :3],
                object_error[repair_ids],
                allowed_lead_xy[repair_ids],
                allowed_lead_z[repair_ids],
                allowed_lead_z[repair_ids],
                self.cfg.horizontal_position_deadband,
                self.cfg.vertical_position_deadband,
            )
            safe_twist[:, 3:] = limit_vector_norm(
                invalid_twist[:, 3:], self.cfg.max_angular_target_lead
            )
            repair_solver = tuple(component[repair_ids] for component in solver)
            correction = solve_prioritized_dls_delta(
                safe_twist - invalid_twist,
                task_jacobian[repair_ids],
                primary_row_mask[repair_ids],
                repair_solver,
                self.cfg.max_target_lead,
            )
            repaired_current = current_joint_pos[assist_ids][repair_ids]
            repaired_previous = previous_target[repair_ids]
            repaired_limits = joint_limits[repair_ids]
            repaired = constrain_target_lead(
                invalid_candidate + correction,
                repaired_current,
                repaired_previous,
                repaired_limits,
                self.cfg.max_target_lead,
                self.cfg.max_joint_step,
            )
            predicted = (
                task_jacobian[repair_ids] @ repaired.unsqueeze(-1)
            ).squeeze(-1)
            repaired_xy_safe, repaired_z_safe = self._cartesian_axis_safe(
                predicted[:, :3],
                object_error[repair_ids],
                allowed_lead_xy[repair_ids],
                allowed_lead_z[repair_ids],
                allowed_lead_z[repair_ids],
                self.cfg.cartesian_hard_leak_tolerance,
            )
            if allow_stabilization:
                repaired_progress_valid = torch.all(
                    torch.isfinite(predicted[:, :3]), dim=1
                )
            else:
                repaired_progress_valid, _, _ = phase_primary_axis_progress_valid(
                    predicted[:, :3],
                    delta_position[repair_ids],
                    require_xy[repair_ids],
                    require_z[repair_ids],
                    prefer_z_progress[repair_ids],
                    self.cfg.minimum_active_progress_ratio,
                    self.cfg.minimum_companion_progress_ratio,
                )
            repaired_valid = (
                repaired_xy_safe
                & repaired_z_safe
                & repaired_progress_valid
                & (
                    torch.norm(predicted[:, 3:], dim=1)
                    <= self.cfg.max_angular_target_lead + 1.0e-6
                )
                & target_lead_constraints_valid(
                    repaired,
                    repaired_current,
                    repaired_previous,
                    repaired_limits,
                    self.cfg.max_target_lead,
                    self.cfg.max_joint_step,
                )
            )
            repaired_all[repair_ids] = repaired
            valid_all[repair_ids] = repaired_valid
            return repaired_all, valid_all

        steady_assist = lead_constraint_feasible & ~fresh_local
        preferred_raw_lead = self.cfg.assist_target_lead_steps * delta_joint
        new_lead, new_lead_valid = repair_lead(
            preferred_raw_lead,
            steady_assist & increment_valid,
            selected_required_xy,
            selected_required_z,
        )
        no_progress_required = torch.zeros_like(requested_motion)
        previous_lead, previous_lead_valid = repair_lead(
            previous_target - current_joint_pos[assist_ids],
            steady_assist & ~new_lead_valid,
            no_progress_required,
            no_progress_required,
            allow_stabilization=True,
        )
        incremental_lead, incremental_lead_valid = repair_lead(
            delta_joint,
            steady_assist
            & increment_valid
            & ~new_lead_valid
            & ~previous_lead_valid,
            selected_required_xy,
            selected_required_z,
        )

        selected_lead = torch.zeros_like(new_lead)
        selected_lead = torch.where(
            incremental_lead_valid[:, None], incremental_lead, selected_lead
        )
        selected_lead = torch.where(
            previous_lead_valid[:, None], previous_lead, selected_lead
        )
        selected_lead = torch.where(
            new_lead_valid[:, None], new_lead, selected_lead
        )
        entry_seed_lead = previous_target - assist_current
        selected_lead = torch.where(
            fresh_local[:, None], entry_seed_lead, selected_lead
        )
        used_previous = ~new_lead_valid & previous_lead_valid
        used_incremental = (
            ~new_lead_valid & ~previous_lead_valid & incremental_lead_valid
        )
        used_previous &= ~fresh_local
        used_incremental &= ~fresh_local
        used_hold = ~fresh_local & (
            ~new_lead_valid & ~previous_lead_valid & ~incremental_lead_valid
        )
        assisted_target = current_joint_pos[assist_ids] + selected_lead
        final_twist = (task_jacobian @ selected_lead.unsqueeze(-1)).squeeze(-1)
        final_xy_hard_safe, final_z_hard_safe = self._cartesian_axis_safe(
            final_twist[:, :3],
            object_error,
            allowed_lead_xy,
            allowed_lead_z,
            allowed_lead_z,
            self.cfg.cartesian_hard_leak_tolerance,
        )
        final_lead_safe = (
            final_xy_hard_safe
            & final_z_hard_safe
            & (
                torch.norm(final_twist[:, 3:], dim=1)
                <= self.cfg.max_angular_target_lead + 1.0e-6
            )
            & target_lead_constraints_valid(
                selected_lead,
                assist_current,
                previous_target,
                joint_limits,
                self.cfg.max_target_lead,
                self.cfg.max_joint_step,
            )
        )
        selected_lead = torch.where(
            final_lead_safe[:, None], selected_lead, torch.zeros_like(selected_lead)
        )
        used_hold |= ~final_lead_safe
        assisted_target = current_joint_pos[assist_ids] + selected_lead
        constraint_rebase = ~lead_constraint_feasible
        used_hold |= constraint_rebase
        rebase_ids = assist_ids[constraint_rebase]
        self._assist_command_velocity[rebase_ids] = 0.0
        self._assist_command_acceleration[rebase_ids] = 0.0
        self._previous_requested_vertical_step[rebase_ids] = 0.0
        self._previous_requested_translation[rebase_ids] = 0.0
        self.assist_command_speed[rebase_ids] = 0.0
        self.assist_command_acceleration[rebase_ids] = 0.0
        self.assist_lead_fallback[assist_ids] = (
            steady_assist & ~new_lead_valid
        ).float()
        self.assist_lead_previous_fallback[assist_ids] = used_previous.float()
        self.assist_lead_incremental_fallback[assist_ids] = used_incremental.float()
        self.assist_lead_safety_stop[assist_ids] = used_hold.float()
        self._assist_joint_target[assist_ids] = assisted_target
        applied_delta = assisted_target - current_joint_pos[assist_ids]
        predicted_applied_twist = (
            task_jacobian @ applied_delta.unsqueeze(-1)
        ).squeeze(-1)
        final_xy_safe, final_z_safe = self._cartesian_axis_safe(
            predicted_applied_twist[:, :3],
            object_error,
            allowed_lead_xy,
            allowed_lead_z,
            allowed_lead_z,
        )
        self.assist_xy_safe[assist_ids] = final_xy_safe.float()
        self.assist_z_safe[assist_ids] = final_z_safe.float()
        self.assist_vertical_step[assist_ids] = predicted_applied_twist[:, 2]
        self.assist_target_lead[assist_ids] = torch.amax(torch.abs(applied_delta), dim=1)
        target_joint_margin = torch.minimum(
            assisted_target - joint_limits[:, :, 0],
            joint_limits[:, :, 1] - assisted_target,
        ) / joint_span
        self.assist_target_joint_limit_proximity[assist_ids] = torch.clamp(
            1.0 - 2.0 * torch.amin(target_joint_margin, dim=1),
            min=0.0,
            max=1.0,
        )
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
    assist_entry_blend_steps: int = 12
    assist_entry_target_lead_steps: int = 3
    assist_target_lead_steps: int = 3
    max_horizontal_acceleration: float = 0.60
    max_vertical_acceleration: float = 0.45
    max_horizontal_jerk: float = 6.0
    max_vertical_jerk: float = 4.5
    horizontal_position_deadband: float = 0.005
    vertical_position_deadband: float = 0.002
    cartesian_leak_tolerance: float = 0.0002
    cartesian_hard_leak_tolerance: float = 0.0010
    minimum_active_progress_ratio: float = 0.25
    minimum_companion_progress_ratio: float = 0.05
    slip_watchdog_threshold: float = 0.015
    slip_watchdog_steps: int = 3
    slip_recovery_hold_steps: int = 3
    handoff_steps: int = 12
    damping: float = 0.08
    outer_xy_threshold: float = 0.24
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
        self.grasp_alignment_streak = torch.zeros_like(self.grasp_streak)
        self.grasp_attempt_lost_streak = torch.zeros_like(self.grasp_streak)
        self.grasp_verification_streak = torch.zeros_like(self.grasp_streak)
        self.grasp_reopen_streak = torch.zeros_like(self.grasp_streak)
        self.grasp_empty_close_streak = torch.zeros_like(self.grasp_streak)
        self.grasp_retry_cooldown = torch.zeros_like(self.grasp_streak)
        self.grasp_attempt_aborts = torch.zeros_like(self.grasp_streak)
        self.grasp_alignment_aborts = torch.zeros_like(self.grasp_streak)
        self.grasp_coupling_aborts = torch.zeros_like(self.grasp_streak)
        self.grasp_close_timeouts = torch.zeros_like(self.grasp_streak)
        self.grasp_verification_timeouts = torch.zeros_like(self.grasp_streak)
        self.grasp_empty_close_aborts = torch.zeros_like(self.grasp_streak)
        self.grasp_close_grace_entries = torch.zeros_like(self.grasp_streak)
        self.grasp_close_timeout_near_contact = torch.zeros_like(self.grasp_streak)
        for phase_name in (
            "approach",
            "close_verify",
            "retry_reopen",
            "actor_transport",
            "assist_to_center",
            "center_to_release",
            "opening",
            "settle",
        ):
            setattr(self, f"phase_{phase_name}_steps", torch.zeros_like(self.grasp_streak))

        self.grasp_ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.grasp_candidate = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.grasp_attempt_active = torch.zeros_like(self.grasp_candidate)
        self.grasp_candidate_tracking = torch.zeros_like(self.grasp_candidate)
        self.grasp_alignment_ready = torch.zeros_like(self.grasp_candidate)
        self.grasp_attempt_continuation = torch.zeros_like(self.grasp_candidate)
        self.grasp_lift_authorized = torch.zeros_like(self.grasp_candidate)
        self.grasp_verification_active = torch.zeros_like(self.grasp_candidate)
        self.grasp_close_grace_active = torch.zeros_like(self.grasp_candidate)
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
        self.grasp_controller_owns_arm = torch.zeros_like(self.grasp_candidate)
        self.grasp_retry_recovery_active = torch.zeros_like(self.grasp_candidate)
        self.basket_reached = torch.zeros_like(self.grasp_candidate)
        self.inside_basket_entry_xy = torch.zeros_like(self.grasp_candidate)
        self.inside_release_xy = torch.zeros_like(self.grasp_candidate)
        self.inside_release_safe_xy = torch.zeros_like(self.grasp_candidate)
        self.place_centered = torch.zeros_like(self.grasp_candidate)
        self.post_center_descent_started = torch.zeros_like(self.grasp_candidate)
        self.post_center_descent_active = torch.zeros_like(self.grasp_candidate)
        self.post_center_descent_z_priority = torch.zeros_like(self.grasp_candidate)
        self.release_ready = torch.zeros_like(self.grasp_candidate)
        self.release_ready_reached = torch.zeros_like(self.grasp_candidate)
        self.release_commanded = torch.zeros_like(self.grasp_candidate)
        self.gripper_target_open = torch.ones_like(self.grasp_candidate)
        self.released = torch.zeros_like(self.grasp_candidate)
        self.succeeded = torch.zeros_like(self.grasp_candidate)
        self.failed = torch.zeros_like(self.grasp_candidate)
        self.grasp_first_attempt_success = torch.zeros_like(self.grasp_candidate)
        self.grasp_retry_success = torch.zeros_like(self.grasp_candidate)
        for phase_name in (
            "approach",
            "close_verify",
            "retry_reopen",
            "actor_transport",
            "assist_to_center",
            "center_to_release",
            "opening",
            "settle",
        ):
            setattr(
                self,
                f"timeout_phase_{phase_name}",
                torch.zeros_like(self.grasp_candidate),
            )

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
        self.bilateral_closed_fraction = torch.zeros_like(self.finger_proximity)
        self.bilateral_open_fraction = torch.zeros_like(self.finger_proximity)
        self.gripper_aperture_asymmetry = torch.zeros_like(self.finger_proximity)
        self.gripper_joint_speed = torch.zeros_like(self.finger_proximity)
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
        self.tcp_height = torch.zeros_like(self.finger_proximity)
        self.centering_height_error = torch.zeros_like(self.finger_proximity)
        self.descent_progress = torch.zeros_like(self.finger_proximity)
        self.descent_target_height = torch.zeros_like(self.finger_proximity)
        self.post_center_descent_start_progress = torch.zeros_like(
            self.finger_proximity
        )
        self.post_center_descent_elapsed_steps = torch.zeros_like(
            self.grasp_streak
        )
        self.post_center_descent_tracking_pause_steps = torch.zeros_like(
            self.grasp_streak
        )
        self.post_center_descent_stability_pause_steps = torch.zeros_like(
            self.grasp_streak
        )
        self.post_center_descent_centering_pause_steps = torch.zeros_like(
            self.grasp_streak
        )
        self.post_center_descent_basin_pause_steps = torch.zeros_like(
            self.grasp_streak
        )
        self.assist_entry_object_height = torch.zeros_like(self.finger_proximity)
        self.assist_entry_object_speed = torch.zeros_like(self.finger_proximity)
        self.assist_entry_height_blocked_steps = torch.zeros_like(
            self.grasp_streak
        )
        self.assist_entry_height_shortfall_max = torch.zeros_like(
            self.finger_proximity
        )
        self.release_target_height = torch.zeros_like(self.finger_proximity)
        self.release_height_delta = torch.zeros_like(self.finger_proximity)
        self.release_height_error = torch.zeros_like(self.finger_proximity)
        self.release_command_height = torch.zeros_like(self.finger_proximity)
        self.release_command_height_error = torch.zeros_like(self.finger_proximity)
        self.release_command_speed = torch.zeros_like(self.finger_proximity)
        self.actual_release_height = torch.zeros_like(self.finger_proximity)
        self.actual_release_height_error = torch.zeros_like(self.finger_proximity)
        self.actual_release_speed = torch.zeros_like(self.finger_proximity)
        self.release_curriculum_progress = torch.zeros_like(self.finger_proximity)
        self.grasp_candidate_object_origin = torch.zeros((env.num_envs, 3), device=env.device)
        self.grasp_candidate_tcp_origin = torch.zeros((env.num_envs, 3), device=env.device)
        self.grasp_candidate_tcp_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self.grasp_candidate_object_lift = torch.zeros(env.num_envs, device=env.device)
        self.grasp_candidate_tcp_lift = torch.zeros(env.num_envs, device=env.device)
        self.grasp_object_origin = torch.zeros((env.num_envs, 3), device=env.device)
        self.grasp_tcp_origin = torch.zeros((env.num_envs, 3), device=env.device)
        self.grasp_tcp_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self.verified_lift_object_progress = torch.zeros(env.num_envs, device=env.device)
        self.verified_lift_tcp_progress = torch.zeros(env.num_envs, device=env.device)
        self.verified_lift_coupled_progress = torch.zeros(env.num_envs, device=env.device)
        self.verified_lift_observed_watermark = torch.zeros(
            env.num_envs, device=env.device
        )
        self.verified_lift_credited_rise = torch.zeros(
            env.num_envs, device=env.device
        )
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
        self._alignment_confirm_steps = int(params["alignment_confirm_steps"])
        self._contact_max_gripper_speed = float(params["contact_max_gripper_speed"])
        self._reopen_arm_hold_steps = int(params["reopen_arm_hold_steps"])
        self._empty_close_confirm_steps = int(params["empty_close_confirm_steps"])
        self._closing_contact_grace_steps = int(params["closing_contact_grace_steps"])
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
        if self._empty_close_confirm_steps < 1:
            raise ValueError("empty close confirmation must be at least one control step")
        if self._closing_contact_grace_steps < 0:
            raise ValueError("closing contact grace must be non-negative")
        minimum_close_budget = self._closing_min_steps + self._contact_confirm_steps - 1
        if self._closing_timeout_steps < minimum_close_budget:
            raise ValueError("closing timeout is too short for the contact confirmation contract")
        self._open_joint_pos = float(params["open_joint_pos"])
        self._grasp_body_cfg = params["grasp_body_cfg"]
        self._tcp_body_cfg = params["tcp_body_cfg"]
        self._gripper_cfg = params["gripper_cfg"]
        levels = cfg.params.get("release_height_levels", (0.670,))
        probabilities = cfg.params.get("release_height_probabilities", (1.0,))
        if len(levels) == 0 or len(levels) != len(probabilities):
            raise ValueError("release height levels and probabilities must have the same non-zero length")
        if any(probability < 0.0 for probability in probabilities) or sum(probabilities) <= 0.0:
            raise ValueError("release height probabilities must be non-negative with a positive sum")
        self._release_height_levels = torch.tensor(levels, device=env.device, dtype=torch.float32)
        self._release_height_probabilities = torch.tensor(probabilities, device=env.device, dtype=torch.float32)
        self._release_height_probabilities /= torch.sum(self._release_height_probabilities)
        normalization_bounds = cfg.params.get("release_height_normalization_bounds", (0.670, 0.75))
        if len(normalization_bounds) != 2 or normalization_bounds[1] <= normalization_bounds[0]:
            raise ValueError("release height normalization bounds must be an increasing pair")
        if min(levels) < normalization_bounds[0] or max(levels) > normalization_bounds[1]:
            raise ValueError("release height levels must stay inside the fixed normalization bounds")
        self._release_height_min = float(normalization_bounds[0])
        self._release_height_max = float(normalization_bounds[1])
        self._centering_height = float(params["centering_height"])
        self._descent_start_xy = float(params["descent_start_xy_threshold"])
        self._descent_transition_xy = float(params["descent_transition_xy_threshold"])
        self._descent_end_xy = float(params["descent_end_xy_threshold"])
        self._preplace_height_above_release = float(params["preplace_height_above_release"])
        self._post_center_descent_xy = float(params["post_center_descent_xy_threshold"])
        self._post_center_descent_steps = int(params["post_center_descent_steps"])
        self._post_center_descent_max_tracking_error = float(
            params["post_center_descent_max_tracking_error"]
        )
        if not self._descent_start_xy > self._descent_transition_xy > self._descent_end_xy >= 0.0:
            raise ValueError("descent XY thresholds must be strictly decreasing and non-negative")
        if self._preplace_height_above_release < 0.0:
            raise ValueError("preplace height above release must be non-negative")
        if not self._descent_end_xy < self._post_center_descent_xy <= float(
            params["place_center_xy_threshold"]
        ):
            raise ValueError(
                "post-center descent XY threshold must be above descent end and inside centering"
            )
        if self._post_center_descent_steps < 1:
            raise ValueError("post-center descent steps must be positive")
        if self._post_center_descent_max_tracking_error <= 0.0:
            raise ValueError("post-center descent tracking error must be positive")
        if max(levels) > self._centering_height:
            raise ValueError("release height levels must not exceed the centering height")
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
        self.descent_target_height.fill_(self._centering_height)
        env._openarm_pick_place_state = self

    @staticmethod
    def _smoothstep01(progress: torch.Tensor) -> torch.Tensor:
        progress = torch.clamp(progress, min=0.0, max=1.0)
        return progress * progress * (3.0 - 2.0 * progress)

    def _descent_height_from_xy(self, distance_xy: torch.Tensor) -> torch.Tensor:
        """Return the C1-continuous XY-coupled height target for the held object."""

        stage_one = self._smoothstep01(
            (self._descent_start_xy - distance_xy)
            / (self._descent_start_xy - self._descent_transition_xy)
        )
        stage_two = self._smoothstep01(
            (self._descent_transition_xy - distance_xy)
            / (self._descent_transition_xy - self._descent_end_xy)
        )
        preplace_height = torch.minimum(
            torch.full_like(self.release_target_height, self._centering_height),
            self.release_target_height + self._preplace_height_above_release,
        )
        stage_one_target = self._centering_height + stage_one * (
            preplace_height - self._centering_height
        )
        return stage_one_target + stage_two * (
            self.release_target_height - preplace_height
        )

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

        new_request = request_mask & ~self.release_commanded
        self.release_command_height[new_request] = self.object_height[new_request]
        self.release_command_height_error[new_request] = self.release_height_error[new_request]
        self.release_command_speed[new_request] = self.object_speed[new_request]
        self.release_commanded |= new_request

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
        per_finger_closed = torch.clamp(
            (self._open_joint_pos - gripper_pos) / self._open_joint_pos,
            min=0.0,
            max=1.0,
        )
        per_finger_open = torch.clamp(
            gripper_pos / self._open_joint_pos,
            min=0.0,
            max=1.0,
        )
        self.closed_fraction[:] = torch.mean(per_finger_closed, dim=1)
        self.open_fraction[:] = torch.mean(per_finger_open, dim=1)
        self.bilateral_closed_fraction[:] = torch.amin(per_finger_closed, dim=1)
        self.bilateral_open_fraction[:] = torch.amin(per_finger_open, dim=1)
        self.gripper_aperture_asymmetry[:] = torch.amax(per_finger_open, dim=1) - torch.amin(
            per_finger_open, dim=1
        )
        gripper_velocity = robot.data.joint_vel[:, self._gripper_cfg.joint_ids]
        self.gripper_joint_speed[:] = torch.amax(torch.abs(gripper_velocity), dim=1)

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
            & (self.bilateral_open_fraction >= self._preclose_min_open_fraction)
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

    def try_start_grasp_attempt(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Latch a grasp only after centered geometry remains stable for several steps."""

        self.update_grasp_geometry(env)
        alignment_candidate = (
            self.grasp_alignment_ready
            & ~self.grasp_attempt_active
            & ~self.grasp_acquired
            & ~self.failed
            & (self.grasp_retry_cooldown == 0)
        )
        self.grasp_alignment_streak[:] = torch.where(
            alignment_candidate,
            torch.clamp(
                self.grasp_alignment_streak + 1,
                max=self._alignment_confirm_steps,
            ),
            0,
        )
        self.grasp_ready[:] = self.grasp_alignment_streak >= self._alignment_confirm_steps
        new_attempt = self.grasp_ready & alignment_candidate
        self.grasp_attempt_active[new_attempt] = True
        self.grasp_attempt_started[new_attempt] = True
        self.grasp_lift_authorized[new_attempt] = False
        self.grasp_candidate_tracking[new_attempt] = False
        self.grasp_closing_streak[new_attempt] = 0
        self.grasp_contact_streak[new_attempt] = 0
        self.grasp_attempt_lost_streak[new_attempt] = 0
        self.grasp_verification_streak[new_attempt] = 0
        self.grasp_empty_close_streak[new_attempt] = 0
        self.grasp_close_grace_active[new_attempt] = False
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
        self.descent_target_height[resolved_env_ids] = self._centering_height

    def _current_phase_masks(self) -> dict[str, torch.Tensor]:
        """Snapshot the controller ownership used for the current physics step."""

        remaining = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        phase_masks: dict[str, torch.Tensor] = {}

        phase_masks["settle"] = remaining & self.released
        remaining &= ~phase_masks["settle"]
        phase_masks["opening"] = remaining & self.release_commanded
        remaining &= ~phase_masks["opening"]
        phase_masks["center_to_release"] = (
            remaining & self.assist_engaged & self.place_centered
        )
        remaining &= ~phase_masks["center_to_release"]
        phase_masks["assist_to_center"] = remaining & self.assist_engaged
        remaining &= ~phase_masks["assist_to_center"]
        phase_masks["actor_transport"] = remaining & self.grasp_acquired
        remaining &= ~phase_masks["actor_transport"]
        phase_masks["retry_reopen"] = remaining & self.grasp_retry_recovery_active
        remaining &= ~phase_masks["retry_reopen"]
        phase_masks["close_verify"] = remaining & self.grasp_attempt_active
        remaining &= ~phase_masks["close_verify"]
        phase_masks["approach"] = remaining
        return phase_masks

    def _update_phase_timing(
        self,
        env: ManagerBasedRLEnv,
        phase_masks: dict[str, torch.Tensor],
        command_name: str,
    ):
        """Accumulate one phase, latch timeout ownership, and flush terminal metrics."""

        timed_out = env.episode_length_buf >= env.max_episode_length
        for phase_name, phase_mask in phase_masks.items():
            getattr(self, f"phase_{phase_name}_steps").add_(phase_mask.long())
            timeout_latch = getattr(self, f"timeout_phase_{phase_name}")
            timeout_latch |= timed_out & phase_mask
        command_term = env.command_manager.get_term(command_name)
        if hasattr(command_term, "sync_task_latched_metrics"):
            command_term.sync_task_latched_metrics(self)

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
        descent_start_xy_threshold: float,
        descent_transition_xy_threshold: float,
        descent_end_xy_threshold: float,
        preplace_height_above_release: float,
        post_center_descent_xy_threshold: float,
        post_center_descent_steps: int,
        post_center_descent_max_tracking_error: float,
        assist_entry_xy_threshold: float,
        basket_entry_xy_threshold: float,
        place_center_xy_threshold: float,
        place_center_max_xy_speed: float,
        release_xy_threshold: float,
        success_xy_threshold: float,
        basket_exit_xy_threshold: float,
        release_height_above_tolerance: float,
        release_height_below_tolerance: float,
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
        alignment_confirm_steps: int,
        contact_max_gripper_speed: float,
        contact_max_aperture_asymmetry: float,
        closing_min_steps: int,
        contact_confirm_steps: int,
        attempt_lost_confirm_steps: int,
        closing_timeout_steps: int,
        empty_close_confirm_steps: int,
        closing_contact_grace_steps: int,
        verification_timeout_steps: int,
        reopen_arm_hold_steps: int,
        retry_cooldown_steps: int,
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
        phase_masks = self._current_phase_masks()
        robot: Articulation = env.scene[robot_cfg.name]
        obj: RigidObject = env.scene[object_cfg.name]
        self.update_grasp_geometry(env)
        attempt_active = self.grasp_attempt_active & ~self.grasp_acquired
        retry_waiting_for_open = (
            (self.grasp_attempt_aborts > 0)
            & ~attempt_active
            & ~self.grasp_acquired
            & (self.bilateral_open_fraction < self._preclose_min_open_fraction)
        )
        self.grasp_reopen_streak[:] = torch.where(
            retry_waiting_for_open,
            torch.clamp(
                self.grasp_reopen_streak + 1,
                max=reopen_arm_hold_steps,
            ),
            0,
        )
        retry_repositioning = (
            (self.grasp_attempt_aborts > 0)
            & ~attempt_active
            & ~self.grasp_acquired
            & (self.bilateral_open_fraction >= self._preclose_min_open_fraction)
        )
        self.grasp_retry_cooldown[:] = torch.where(
            retry_repositioning,
            torch.clamp(self.grasp_retry_cooldown - 1, min=0),
            self.grasp_retry_cooldown,
        )

        closing = attempt_active & ~self.grasp_lift_authorized & ~self.gripper_target_open
        closing_hard_limit = closing_timeout_steps + closing_contact_grace_steps
        self.grasp_closing_streak[:] = torch.where(
            closing,
            torch.clamp(self.grasp_closing_streak + 1, max=closing_hard_limit),
            torch.where(self.grasp_lift_authorized, self.grasp_closing_streak, 0),
        )
        near_contact_geometry = (
            closing
            & self.grasp_attempt_continuation
            & (self.bilateral_finger_proximity >= candidate_proximity_threshold)
            & (self.bilateral_closed_fraction >= candidate_closed_threshold)
            & (self.bilateral_open_fraction >= candidate_min_open_fraction)
            & (self.gripper_aperture_asymmetry <= contact_max_aperture_asymmetry)
            & (self.grasp_relative_speed <= candidate_max_relative_speed)
            & (self.grasp_closing_streak >= closing_min_steps)
        )
        contact_geometry = near_contact_geometry & (
            self.gripper_joint_speed <= contact_max_gripper_speed
        )
        empty_close_candidate = (
            closing
            & (self.grasp_closing_streak >= closing_min_steps)
            & (self.gripper_joint_speed <= contact_max_gripper_speed)
            & (self.bilateral_open_fraction < candidate_min_open_fraction)
        )
        self.grasp_empty_close_streak[:] = torch.where(
            empty_close_candidate,
            torch.clamp(
                self.grasp_empty_close_streak + 1,
                max=empty_close_confirm_steps,
            ),
            0,
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
            & (self.bilateral_closed_fraction >= candidate_closed_threshold)
            & (self.bilateral_open_fraction >= candidate_min_open_fraction)
            & (self.gripper_aperture_asymmetry <= contact_max_aperture_asymmetry)
        )
        candidate_geometry = verifying & verified_contact_geometry
        object_candidate_delta = obj.data.root_pos_w - self.grasp_candidate_object_origin
        tcp_candidate_delta = tcp_pos_w - self.grasp_candidate_tcp_origin
        coupling_error = torch.norm(
            self.object_offset_tcp - self.grasp_candidate_tcp_offset, dim=1
        )
        unstable_candidate = candidate_geometry & (
            (coupling_error > coupling_max_displacement_error)
            | (self.grasp_relative_speed > candidate_max_relative_speed)
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
            | unstable_candidate
        )
        self.grasp_attempt_lost_streak[:] = torch.where(
            lost_attempt_geometry,
            torch.clamp(
                self.grasp_attempt_lost_streak + 1, max=attempt_lost_confirm_steps
            ),
            0,
        )
        entered_contact_grace = (
            closing
            & ~newly_authorized
            & near_contact_geometry
            & (self.grasp_closing_streak == closing_timeout_steps)
        )
        self.grasp_close_grace_active |= entered_contact_grace
        empty_close_aborted = (
            closing
            & (self.grasp_empty_close_streak >= empty_close_confirm_steps)
        )
        close_timed_out = (
            closing
            & ~newly_authorized
            & (
                (
                    (self.grasp_closing_streak >= closing_timeout_steps)
                    & ~self.grasp_close_grace_active
                )
                | (self.grasp_closing_streak >= closing_hard_limit)
            )
            & ~empty_close_aborted
        )
        verification_timed_out = verifying & (
            self.grasp_verification_streak > verification_timeout_steps
        )
        alignment_aborted = attempt_active & (
            self.grasp_attempt_lost_streak >= attempt_lost_confirm_steps
        )
        coupling_aborted = alignment_aborted & unstable_candidate
        geometry_aborted = alignment_aborted & ~unstable_candidate
        attempt_aborted = (
            alignment_aborted
            | empty_close_aborted
            | close_timed_out
            | verification_timed_out
        )
        self.just_grasp_retry[:] = attempt_aborted
        self.grasp_attempt_aborts += attempt_aborted.long()
        self.grasp_alignment_aborts += geometry_aborted.long()
        self.grasp_coupling_aborts += coupling_aborted.long()
        self.grasp_close_timeouts += close_timed_out.long()
        self.grasp_verification_timeouts += verification_timed_out.long()
        self.grasp_empty_close_aborts += empty_close_aborted.long()
        self.grasp_close_grace_entries += entered_contact_grace.long()
        self.grasp_close_timeout_near_contact += (
            close_timed_out & near_contact_geometry
        ).long()
        self.grasp_attempt_active[attempt_aborted] = False
        self.grasp_lift_authorized[attempt_aborted] = False
        self.grasp_verification_active[attempt_aborted] = False
        self.grasp_candidate_tracking[attempt_aborted] = False
        self.grasp_retry_cooldown[attempt_aborted] = retry_cooldown_steps
        self.grasp_closing_streak[attempt_aborted] = 0
        self.grasp_contact_streak[attempt_aborted] = 0
        self.grasp_attempt_lost_streak[attempt_aborted] = 0
        self.grasp_verification_streak[attempt_aborted] = 0
        self.grasp_reopen_streak[attempt_aborted] = 0
        self.grasp_empty_close_streak[attempt_aborted] = 0
        self.grasp_close_grace_active[attempt_aborted | newly_authorized] = False

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
        self.tcp_height[:] = tcp_pos_w[:, 2]
        self.inside_basket_entry_xy[:] = self.distance_xy <= basket_entry_xy_threshold
        self.inside_release_xy[:] = self.distance_xy <= release_xy_threshold
        self.inside_release_safe_xy[:] = self.distance_xy <= success_xy_threshold

        candidate_active = self.grasp_candidate & ~self.grasp_acquired
        self.grasp_candidate_object_lift[:] = torch.where(
            candidate_active,
            object_candidate_delta[:, 2],
            0.0,
        )
        self.grasp_candidate_tcp_lift[:] = torch.where(
            candidate_active,
            tcp_candidate_delta[:, 2],
            0.0,
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
        self.grasp_first_attempt_success |= self.just_grasped & (
            self.grasp_attempt_aborts == 0
        )
        self.grasp_retry_success |= self.just_grasped & (
            self.grasp_attempt_aborts > 0
        )
        self.grasp_tcp_offset[self.just_grasped] = self.object_offset_tcp[self.just_grasped]
        self.grasp_object_origin[self.just_grasped] = obj.data.root_pos_w[self.just_grasped]
        self.grasp_tcp_origin[self.just_grasped] = tcp_pos_w[self.just_grasped]
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

        object_lift_progress, tcp_lift_progress, coupled_progress = (
            coupled_lift_progress(
                obj.data.root_pos_w[:, 2],
                tcp_pos_w[:, 2],
                self.grasp_object_origin[:, 2],
                self.grasp_tcp_origin[:, 2],
                carry_height,
            )
        )
        verified_grasp = self.grasp_acquired & ~self.released
        self.verified_lift_object_progress[:] = torch.where(
            verified_grasp,
            object_lift_progress,
            0.0,
        )
        self.verified_lift_tcp_progress[:] = torch.where(
            verified_grasp,
            tcp_lift_progress,
            0.0,
        )
        self.verified_lift_coupled_progress[:] = torch.where(
            verified_grasp,
            coupled_progress,
            0.0,
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
        assist_entry_candidate = (
            ~previous_assist
            & self.grasp_acquired
            & self.current_hold
            & self.lift_completed
            & self.validated_transport
            & (self.distance_xy <= assist_entry_xy_threshold)
        )
        assist_entry_height_shortfall = torch.clamp(
            carry_height - self.object_height, min=0.0
        )
        assist_entry_height_blocked = assist_entry_candidate & (
            assist_entry_height_shortfall > 0.0
        )
        self.assist_entry_height_blocked_steps.add_(
            assist_entry_height_blocked.long()
        )
        self.assist_entry_height_shortfall_max[:] = torch.maximum(
            self.assist_entry_height_shortfall_max,
            torch.where(
                assist_entry_height_blocked,
                assist_entry_height_shortfall,
                torch.zeros_like(assist_entry_height_shortfall),
            ),
        )
        self.assist_engaged |= assist_entry_candidate & (
            self.object_height >= carry_height
        )
        self.just_assist_engaged[:] = self.assist_engaged & ~previous_assist
        self.assist_entry_object_height[self.just_assist_engaged] = self.object_height[
            self.just_assist_engaged
        ]
        self.assist_entry_object_speed[self.just_assist_engaged] = self.object_speed[
            self.just_assist_engaged
        ]

        raw_descent_height = self._descent_height_from_xy(self.distance_xy)
        height_span = torch.clamp(
            centering_height - self.release_target_height,
            min=1.0e-6,
        )
        raw_descent_progress = torch.clamp(
            (centering_height - raw_descent_height) / height_span,
            min=0.0,
            max=1.0,
        )
        descent_active = (
            self.assist_engaged
            & self.current_hold
            & ~self.released
            & ~self.failed
        )
        current_descent_tracking_error = torch.abs(
            self.object_height - self.descent_target_height
        )
        (
            post_center_descent_candidate,
            post_center_descent_ready,
            post_center_descent_stability_pause,
            post_center_descent_tracking_pause,
        ) = post_center_descent_readiness(
            descent_active,
            self.place_centered,
            self.distance_xy,
            self.object_xy_speed,
            current_descent_tracking_error,
            post_center_descent_xy_threshold,
            place_center_max_xy_speed,
            post_center_descent_max_tracking_error,
        )
        self.post_center_descent_stability_pause_steps.add_(
            post_center_descent_stability_pause.long()
        )
        self.post_center_descent_tracking_pause_steps.add_(
            post_center_descent_tracking_pause.long()
        )
        self.post_center_descent_centering_pause_steps.add_(
            (post_center_descent_candidate & ~self.place_centered).long()
        )
        self.post_center_descent_basin_pause_steps.add_(
            (
                descent_active
                & self.post_center_descent_started
                & (
                    ~torch.isfinite(self.distance_xy)
                    | (self.distance_xy > post_center_descent_xy_threshold)
                )
            ).long()
        )
        self.post_center_descent_active[:] = post_center_descent_ready
        self.post_center_descent_z_priority[:] = (
            post_center_descent_ready | post_center_descent_tracking_pause
        )
        (
            updated_descent_progress,
            updated_descent_start,
            updated_descent_steps,
            updated_descent_started,
        ) = advance_post_center_descent_progress(
            self.descent_progress,
            torch.where(
                descent_active,
                raw_descent_progress,
                self.descent_progress,
            ),
            self.post_center_descent_start_progress,
            self.post_center_descent_elapsed_steps,
            self.post_center_descent_started,
            post_center_descent_candidate,
            self.post_center_descent_active,
            post_center_descent_steps,
        )
        self.descent_progress[:] = updated_descent_progress
        self.post_center_descent_start_progress[:] = updated_descent_start
        self.post_center_descent_elapsed_steps[:] = updated_descent_steps
        self.post_center_descent_started[:] = updated_descent_started
        self.descent_target_height[:] = centering_height + self.descent_progress * (
            self.release_target_height - centering_height
        )
        self.centering_height_error[:] = torch.abs(
            obj.data.root_pos_w[:, 2] - self.descent_target_height
        )

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
            & (self.release_height_delta >= -release_height_above_tolerance)
            & (self.release_height_delta <= release_height_below_tolerance)
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
            & (self.bilateral_open_fraction >= 0.70)
        )
        self.just_released[:] = self.released & ~previous_released
        self.actual_release_height[:] = torch.where(
            self.just_released,
            self.object_height,
            self.actual_release_height,
        )
        self.actual_release_height_error[:] = torch.where(
            self.just_released,
            self.release_height_error,
            self.actual_release_height_error,
        )
        self.actual_release_speed[:] = torch.where(
            self.just_released,
            self.object_speed,
            self.actual_release_speed,
        )
        opening = self.release_commanded & ~self.released
        release_open_timeout_steps = release_open_grace_steps + lost_confirm_steps
        self.opening_streak[:] = torch.where(
            opening,
            torch.clamp(self.opening_streak + 1, max=release_open_timeout_steps),
            0,
        )

        premature_release = (
            self.grasp_acquired
            & ~self.release_commanded
            & (self.bilateral_open_fraction >= 0.70)
        )
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
            & (self.bilateral_open_fraction >= 0.70)
        )
        self.settle_streak[:] = torch.where(
            settle_candidate,
            torch.clamp(self.settle_streak + 1, max=settle_confirm_steps),
            0,
        )
        self.succeeded[:] = (self.settle_streak >= settle_confirm_steps) & ~self.failed
        self._update_phase_timing(env, phase_masks, command_name)
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def task_state_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Expose the reward/termination phase so the policy sees a Markov state."""

    state: TaskStateTracker | None = getattr(env, "_openarm_pick_place_state", None)
    if state is None:
        return torch.zeros((env.num_envs, 12), device=env.device)
    return torch.stack(
        (
            state.grasp_acquired,
            state.current_hold,
            state.lift_completed,
            state.inside_release_xy,
            state.release_ready,
            state.released,
            state.grasp_attempt_active,
            state.grasp_candidate,
            state.grasp_lift_authorized,
            state.grasp_verification_streak.float()
            / max(state._verification_timeout_steps, 1),
            state.grasp_retry_cooldown.float() / max(state._retry_cooldown_steps, 1),
            state.grasp_reopen_streak.float() / max(state._reopen_arm_hold_steps, 1),
        ),
        dim=1,
    ).float()


def task_actor_authority_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Mask actor learning only while post-grasp automation owns the arm."""

    state: TaskStateTracker | None = getattr(env, "_openarm_pick_place_state", None)
    if state is None:
        return torch.ones((env.num_envs, 1), device=env.device)
    state.update_grasp_geometry(env)
    imminent_automatic_release = _imminent_automatic_release(env, state)

    post_assist = (
        state.grasp_acquired
        & state.assist_engaged
        & ~imminent_automatic_release
        & ~state.release_commanded
        & ~state.released
        & ~state.failed
    )
    arm_action = env.action_manager.get_term("arm_action")
    outer_xy_threshold = float(getattr(arm_action.cfg, "outer_xy_threshold", 0.0))
    assist_active = post_assist & state.current_hold & (
        state.distance_xy < outer_xy_threshold
    )
    hold_recovery = post_assist & ~state.current_hold
    previous_assist_active = getattr(arm_action, "_assist_was_active", state.assist_active)
    deactivating_assist = previous_assist_active & ~assist_active & ~hold_recovery
    handoff_remaining = getattr(arm_action, "_handoff_steps_remaining", None)
    if handoff_remaining is None:
        handoff_pending = deactivating_assist
    else:
        handoff_pending = deactivating_assist | (handoff_remaining > 0)

    actor_authority = post_grasp_actor_authority(
        assist_active,
        hold_recovery,
        handoff_pending,
        imminent_automatic_release,
        state.release_commanded,
        state.released,
        state.failed,
    )
    return actor_authority.unsqueeze(1).float()


def task_grasp_quality_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Expose the geometry and motion used by the centered grasp gate."""

    state: TaskStateTracker | None = getattr(env, "_openarm_pick_place_state", None)
    if state is None:
        return torch.zeros((env.num_envs, 6), device=env.device)
    state.update_grasp_geometry(env)
    alignment = torch.clamp(
        (state.object_offset_tcp - state._alignment_target)
        / state._alignment_quality_scales,
        min=-3.0,
        max=3.0,
    )
    grasp_gate_state = torch.stack(
        (
            torch.clamp(
                state.bilateral_finger_proximity
                / state._candidate_proximity_threshold,
                min=0.0,
                max=2.0,
            ),
            torch.clamp(
                state.finger_distance_asymmetry
                / state._alignment_max_finger_asymmetry,
                min=0.0,
                max=3.0,
            ),
            torch.clamp(
                state.grasp_relative_speed / state._alignment_max_relative_speed,
                min=0.0,
                max=3.0,
            ),
        ),
        dim=1,
    )
    return torch.cat((alignment, grasp_gate_state), dim=1)


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


def task_failure_with_preassist_cost(
    env: ManagerBasedRLEnv,
    preassist_multiplier: float,
) -> torch.Tensor:
    """Charge actor-owned post-grasp failures more without blaming assist control."""

    state: TaskStateTracker = env._openarm_pick_place_state
    actor_owns_transport = state.grasp_acquired & ~state.assist_engaged
    multiplier = torch.where(
        actor_owns_transport,
        preassist_multiplier,
        1.0,
    )
    return state.failed.float() * multiplier / env.step_dt


def task_gripper_action_phase_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Directly align the continuous gripper action with the observable task phase."""

    state: TaskStateTracker = env._openarm_pick_place_state
    approach_far = ~state.grasp_acquired & ~state.grasp_attempt_active
    safe_release_request = state.release_ready_reached & state.release_ready
    should_open = approach_far | safe_release_request | state.release_commanded | state.released
    desired_action = torch.where(should_open, 1.0, -1.0)
    gripper_action = env.action_manager.get_term("gripper_action").raw_actions[:, 0]
    return 0.25 * torch.square(gripper_action - desired_action)


def task_preassist_delay_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Keep delayed post-grasp failure cost equivalent under PPO discounting."""

    state: TaskStateTracker = env._openarm_pick_place_state
    return (
        state.grasp_acquired
        & ~state.assist_engaged
        & ~state.released
        & ~state.failed
    ).float()


def task_pregrasp_delay_penalty(
    env: ManagerBasedRLEnv,
    retry_event_scale: float = 0.0,
) -> torch.Tensor:
    """Penalize actor-owned pre-grasp delay and charge a small failed-attempt event."""

    state: TaskStateTracker = env._openarm_pick_place_state
    actor_waiting = (
        ~state.grasp_acquired
        & ~state.failed
        & ~state.grasp_controller_owns_arm
    )
    retry_event = retry_event_scale * state.just_grasp_retry.float() / env.step_dt
    return actor_waiting.float() + retry_event


def _task_pregrasp_alignment_quality(state: TaskStateTracker) -> torch.Tensor:
    offset_error = state.object_offset_tcp - state._alignment_target
    negative_span = state._alignment_target - state._alignment_lower_bounds
    positive_span = state._alignment_upper_bounds - state._alignment_target
    gate_span = torch.where(offset_error < 0.0, negative_span, positive_span)
    normalized_offset = offset_error / torch.clamp(gate_span, min=1.0e-6)
    pose_quality = torch.exp(-0.5 * torch.sum(torch.square(normalized_offset), dim=1))
    symmetry_quality = torch.exp(
        -0.5
        * torch.square(
            state.finger_distance_asymmetry
            / max(state._alignment_max_finger_asymmetry, 1.0e-6)
        )
    )
    proximity_quality = torch.clamp(
        state.bilateral_finger_proximity / state._candidate_proximity_threshold,
        min=0.0,
        max=1.0,
    )
    return pose_quality * symmetry_quality * proximity_quality


def task_pregrasp_centering_reward(
    env: ManagerBasedRLEnv,
    quality_scale: float,
    dwell_scale: float,
) -> torch.Tensor:
    """Reward geometry inside the same narrow basin that authorizes grasp closure."""

    state: TaskStateTracker = env._openarm_pick_place_state
    actor_controlled = ~state.grasp_controller_owns_arm
    active = (
        ~state.grasp_acquired
        & ~state.grasp_attempt_active
        & ~state.failed
        & (state.grasp_retry_cooldown == 0)
        & actor_controlled
    )
    quality = _task_pregrasp_alignment_quality(state)
    actor_dwell_steps = max(state._alignment_confirm_steps - 1, 1)
    dwell = torch.clamp(
        state.grasp_alignment_streak.float() / actor_dwell_steps,
        min=0.0,
        max=1.0,
    )
    return active.float() * (quality_scale * quality + dwell_scale * dwell)


class task_pregrasp_signed_potential_progress(ManagerTermBase):
    """Reward new-best alignment once and charge every actor-owned regression."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_quality = torch.zeros(env.num_envs, device=env.device)
        self._best_quality = torch.zeros(env.num_envs, device=env.device)
        self._was_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._previous_quality[env_ids] = 0.0
        self._best_quality[env_ids] = 0.0
        self._was_active[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
    ) -> torch.Tensor:
        state: TaskStateTracker = env._openarm_pick_place_state
        active = (
            ~state.grasp_acquired
            & ~state.grasp_attempt_active
            & ~state.failed
            & (state.grasp_retry_cooldown == 0)
            & ~state.grasp_controller_owns_arm
            & ~state.assist_owns_arm
        )
        quality = _task_pregrasp_alignment_quality(state)
        quality_delta, updated_best = persistent_best_signed_progress(
            quality,
            self._previous_quality,
            self._best_quality,
            active,
            self._was_active,
        )
        reward = quality_delta / env.step_dt
        self._previous_quality[:] = quality
        self._best_quality[:] = updated_best
        self._was_active[:] = active
        return reward


class task_potential_progress(ManagerTermBase):
    """Reward bounded progress without repaying approach or lift cycles."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_potential = torch.zeros(env.num_envs, device=env.device)
        self._best_potential = torch.zeros(env.num_envs, device=env.device)
        self._was_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._ever_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._previous_potential[env_ids] = 0.0
        self._best_potential[env_ids] = 0.0
        self._was_active[env_ids] = False
        self._ever_active[env_ids] = False

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
            active = (
                (state.grasp_candidate_tracking | state.just_grasped)
                & ~state.released
            )
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
        if mode == "approach":
            reward = torch.where(continued, improvement / env.step_dt, 0.0)
        else:
            reward = torch.where(continued, (improvement + regression) / env.step_dt, 0.0)
        if mode in ("approach", "lift"):
            active_best = torch.where(
                self._ever_active,
                torch.maximum(self._best_potential, potential),
                potential,
            )
            self._best_potential[:] = torch.where(active, active_best, self._best_potential)
            self._ever_active |= active
        else:
            active_best = torch.where(
                self._was_active,
                torch.maximum(self._best_potential, potential),
                potential,
            )
            self._best_potential[:] = torch.where(active, active_best, potential)
        self._previous_potential[:] = potential
        self._was_active[:] = active
        return reward


class task_verified_paired_lift_progress(ManagerTermBase):
    """Credit each trustworthy coupled lift height through a smooth watermark."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._observed_watermark = torch.zeros(
            env.num_envs, device=env.device
        )
        self._credited_cumulative_rise = torch.zeros(
            env.num_envs, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._observed_watermark[env_ids] = 0.0
        self._credited_cumulative_rise[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_height: float,
        maximum_credit_height: float,
    ) -> torch.Tensor:
        state: TaskStateTracker = env._openarm_pick_place_state
        timed_out = env.episode_length_buf >= env.max_episode_length
        terminal = state.failed | state.released | timed_out
        acquired_nonterminal = state.grasp_acquired & ~terminal
        coupled_rise = capped_raw_coupled_lift_rise(
            state.object_height,
            state.tcp_height,
            state.grasp_object_origin[:, 2],
            state.grasp_tcp_origin[:, 2],
            target_height,
            maximum_credit_height,
        )
        updated_watermark = torch.where(
            acquired_nonterminal,
            torch.maximum(self._observed_watermark, coupled_rise),
            self._observed_watermark,
        )
        watermark_increment = torch.clamp(
            updated_watermark - self._observed_watermark,
            min=0.0,
        )
        reward_eligible = (
            acquired_nonterminal
            & state.current_hold
            & (~state.lift_completed | state.just_lifted)
        )
        credited_increment = torch.where(
            reward_eligible,
            watermark_increment,
            torch.zeros_like(watermark_increment),
        )
        self._observed_watermark[:] = updated_watermark
        self._credited_cumulative_rise.add_(credited_increment)
        state.verified_lift_observed_watermark[:] = self._observed_watermark
        state.verified_lift_credited_rise[:] = self._credited_cumulative_rise
        return credited_increment / env.step_dt


class task_preclearance_inward_progress(ManagerTermBase):
    """Penalize basket-directed low-clearance motion only near the rim."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_distance_xy = torch.zeros(env.num_envs, device=env.device)
        self._was_active = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._previous_distance_xy[env_ids] = 0.0
        self._was_active[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        clearance_start_height: float,
        clearance_full_height: float,
        near_basket_xy_threshold: float,
        max_step_change: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        state: TaskStateTracker = env._openarm_pick_place_state
        obj: RigidObject = env.scene[object_cfg.name]
        actor_carry_active = actor_owned_carry_phase(
            state.grasp_acquired,
            state.assist_owns_arm,
            state.released,
            state.failed,
        )
        clearance_fraction = carry_clearance_fraction(
            obj.data.root_pos_w[:, 2],
            clearance_start_height,
            clearance_full_height,
        )
        cost = low_clearance_inward_progress_cost(
            self._previous_distance_xy,
            state.distance_xy,
            clearance_fraction,
            actor_carry_active,
            self._was_active,
            near_basket_xy_threshold,
            max_step_change,
        )
        finite_distance = torch.isfinite(state.distance_xy)
        self._previous_distance_xy[:] = torch.where(
            finite_distance,
            state.distance_xy,
            self._previous_distance_xy,
        )
        self._was_active[:] = actor_carry_active & finite_distance
        return cost / env.step_dt


def constant_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def task_timeout_without_success(
    env: ManagerBasedRLEnv,
    preassist_multiplier: float = 1.0,
) -> torch.Tensor:
    """Return a one-shot timeout cost, scaled only during actor-owned transport."""

    state: TaskStateTracker = env._openarm_pick_place_state
    timed_out = env.episode_length_buf >= env.max_episode_length
    incomplete = timed_out & ~state.succeeded & ~state.failed
    actor_owns_transport = state.grasp_acquired & ~state.assist_engaged
    multiplier = torch.where(
        actor_owns_transport,
        preassist_multiplier,
        1.0,
    )
    return incomplete.float() * multiplier / env.step_dt


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
