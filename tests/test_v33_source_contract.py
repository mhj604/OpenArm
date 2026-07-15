from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = (
    ROOT
    / "source"
    / "openarm_testusd_task"
    / "openarm_testusd_task"
    / "tasks"
    / "testusd_pick_place"
)
ENV_SOURCE = (TASK_DIR / "testusd_pick_place_env_cfg.py").read_text()
MDP_SOURCE = (TASK_DIR / "mdp.py").read_text()
PPO_SOURCE = (TASK_DIR / "agents" / "rsl_rl_ppo_cfg.py").read_text()


def test_v33_keeps_episode_and_policy_contract():
    assert "EPISODE_LENGTH_S = 16.0" in ENV_SOURCE
    assert "init_noise_std=0.05" in PPO_SOURCE
    assert "learning_rate=2.5e-6" in PPO_SOURCE
    assert "actor_hidden_dims=[256, 128, 64]" in PPO_SOURCE
    assert "joint_names=RIGHT_ARM_JOINT_NAMES" in ENV_SOURCE
    assert "automatic_phase_control=True" in ENV_SOURCE

    env_tree = ast.parse(ENV_SOURCE)
    reward_class = next(
        node
        for node in env_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RewardsCfgV18"
    )
    reward_terms = [
        node
        for node in reward_class.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "RewTerm"
    ]
    assert len(reward_terms) == 21


def test_train_and_play_share_the_low_release_target():
    assert "STAGE_RELEASE_TARGET_HEIGHT_W = 0.670" in ENV_SOURCE
    assert '"release_height_levels": (STAGE_RELEASE_TARGET_HEIGHT_W,)' in ENV_SOURCE
    assert '"release_height_probabilities": (1.0,)' in ENV_SOURCE
    play_source = ENV_SOURCE.split(
        "class OpenArmTestUsdPickPlaceEnvCfg_STAGE_PLAY", maxsplit=1
    )[1]
    assert 'params["release_height_levels"]' not in play_source
    assert 'params["release_height_probabilities"]' not in play_source


def test_precise_release_and_strict_grasp_gates_are_preserved():
    required_contract = (
        '"release_xy_threshold": 0.04',
        '"success_xy_threshold": 0.05',
        '"release_height_above_tolerance": 0.002',
        '"release_height_below_tolerance": 0.003',
        '"release_max_speed": 0.06',
        '"alignment_confirm_steps": 3',
        '"contact_confirm_steps": 3',
        '"grasp_confirm_steps": 5',
        '"coupling_object_lift": 0.012',
        '"coupling_tcp_lift": 0.010',
        '"coupling_max_displacement_error": 0.012',
    )
    for contract_line in required_contract:
        assert contract_line in ENV_SOURCE


def test_v33_recovery_and_assist_safety_parameters_are_explicit():
    required_v33 = (
        '"descent_end_xy_threshold": 0.005',
        '"empty_close_confirm_steps": 3',
        '"closing_contact_grace_steps": 12',
        "assist_entry_blend_steps=12",
        "assist_entry_target_lead_steps=3",
        "assist_target_lead_steps=3",
        "horizontal_position_deadband=0.005",
        "vertical_position_deadband=0.002",
        "max_horizontal_acceleration=0.60",
        "max_vertical_acceleration=0.45",
        "max_horizontal_jerk=6.0",
        "max_vertical_jerk=4.5",
        "slip_watchdog_threshold=0.015",
        "slip_watchdog_steps=3",
        "slip_recovery_hold_steps=3",
    )
    for contract_line in required_v33:
        assert contract_line in ENV_SOURCE


def test_v34_lead_is_current_relative_and_entry_seed_is_single_step():
    assert (
        "preferred_raw_lead = self.cfg.assist_target_lead_steps * delta_joint"
        in MDP_SOURCE
    )
    assert "proposed_target = previous_target + delta_joint" not in MDP_SOURCE
    assert "steady_assist = lead_constraint_feasible & ~fresh_local" in MDP_SOURCE
    assert "fresh_local[:, None], entry_seed_lead, selected_lead" in MDP_SOURCE
    assert "target_lead_constraint_intersection" in MDP_SOURCE
    assert "assist_lead_constraint_rebase" in MDP_SOURCE
    assert "selected_valid\n            & increment_xy_hard_safe" in MDP_SOURCE
    assert "task_assist_effort_demand_ratio" in MDP_SOURCE
    assert "task_assist_effort_ratio" not in MDP_SOURCE


def test_global_cross_axis_throttle_paths_were_removed():
    assert "cartesian_scale = torch.minimum" not in MDP_SOURCE
    assert "position_lead_scale = torch.minimum" not in MDP_SOURCE
    assert "braking_ids" not in MDP_SOURCE
    assert "def _cartesian_axis_safe" in MDP_SOURCE
    assert "build_prioritized_dls_solver" in MDP_SOURCE
    assert "minimum_active_progress_ratio" in MDP_SOURCE
    assert "inactive_axis_weight" not in MDP_SOURCE
    assert "active_limit_epsilon" in MDP_SOURCE
    assert "cartesian_hard_leak_tolerance" in MDP_SOURCE
    assert "use_active_only" in MDP_SOURCE
    assert "postured_ratio >= 0.95 * selected_ratio" in MDP_SOURCE
    assert "assist_low_progress" in MDP_SOURCE
    assert "constrain_target_lead" in MDP_SOURCE
    assert "target_lead_constraints_valid" in MDP_SOURCE


def test_v35_repairs_invalid_dls_before_holding_the_arm():
    assert "scale_delta_to_cartesian_bounds" in MDP_SOURCE
    assert "required_axis_progress_valid" in MDP_SOURCE
    assert "use_scaled = requested_motion & ~candidate_valid & scaled_valid" in MDP_SOURCE
    assert "def solve_single_axis_candidate" in MDP_SOURCE
    assert "select_phase_axis_fallback" in MDP_SOURCE
    assert "state.inside_release_xy[assist_ids]" in MDP_SOURCE
    assert "selected_required_xy" in MDP_SOURCE
    assert "selected_required_z" in MDP_SOURCE
    assert "no_progress_required" in MDP_SOURCE
    assert "selected_valid |= use_xy_fallback | use_z_fallback" in MDP_SOURCE
    for metric in (
        "task_assist_incremental_scaled_repairs",
        "task_assist_incremental_xy_fallbacks",
        "task_assist_incremental_z_fallbacks",
        "task_assist_incremental_xy_rejects",
        "task_assist_incremental_z_rejects",
    ):
        assert metric in MDP_SOURCE


def test_v36_lift_first_grasp_and_progress_contracts_are_explicit():
    assert "STAGE_CARRY_MIN_HEIGHT_W = 0.73" in ENV_SOURCE
    assert "STAGE_CARRY_TARGET_HEIGHT_W = 0.75" in ENV_SOURCE
    assert "STAGE_CARRY_MAX_HEIGHT_W = 0.79" in ENV_SOURCE
    assert "TRANSPORT_XY_UNLOCK_START_HEIGHT_W = 0.710" in ENV_SOURCE
    assert "TRANSPORT_XY_UNLOCK_FULL_HEIGHT_W = STAGE_CARRY_MIN_HEIGHT_W" in ENV_SOURCE
    assert "LOW_CLEARANCE_BASKET_APPROACH_XY = 0.17" in ENV_SOURCE
    assert "low_clearance_height=TRANSPORT_XY_UNLOCK_FULL_HEIGHT_W" in ENV_SOURCE
    assert '"quality_scale": 4.0' in ENV_SOURCE
    assert '"dwell_scale": 2.0' in ENV_SOURCE
    assert "weight=-200.0" in ENV_SOURCE
    assert "lift_first_transport_goal" in MDP_SOURCE
    assert "task_preclearance_inward_progress" in MDP_SOURCE
    assert "pregrasp_opening_quality_factor" in MDP_SOURCE
    assert "task_low_clearance_basket_approach_steps" in MDP_SOURCE
    assert MDP_SOURCE.count("lift_first_actor_phase(") == 2
    assert MDP_SOURCE.count("lift_first_reward_phase(") == 2
    assert "quality = state.grasp_alignment_quality * aperture_multiplier" in MDP_SOURCE
    assert "aperture_multiplier = 0.75 + 0.25 * open_quality" in MDP_SOURCE
    assert MDP_SOURCE.count("self.cfg.minimum_active_progress_ratio") >= 8
    assert "ContactSensor" not in ENV_SOURCE
    assert "ContactSensor" not in MDP_SOURCE


def test_pregrasp_reward_excludes_only_automatic_pose_holds():
    assert "state.grasp_controller_owns_arm[:] = grasp_control" in MDP_SOURCE
    assert 'phase_masks["close_verify"] = remaining & self.grasp_attempt_active' in MDP_SOURCE
    assert "actor_controlled = ~state.grasp_controller_owns_arm" in MDP_SOURCE
    assert "sync_task_latched_metrics" in MDP_SOURCE


def test_task_state_config_supplies_every_required_call_argument():
    mdp_tree = ast.parse(MDP_SOURCE)
    env_tree = ast.parse(ENV_SOURCE)

    tracker_class = next(
        node
        for node in mdp_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TaskStateTracker"
    )
    call_method = next(
        node
        for node in tracker_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    call_arguments = [argument.arg for argument in call_method.args.args]
    default_count = len(call_method.args.defaults)
    required_arguments = set(call_arguments[2 : len(call_arguments) - default_count])
    valid_arguments = set(call_arguments[2:])

    termination_class = next(
        node
        for node in env_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TerminationsCfgV18"
    )
    task_state_assignment = next(
        node
        for node in termination_class.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "task_state" for target in node.targets)
    )
    assert isinstance(task_state_assignment.value, ast.Call)
    params_keyword = next(
        keyword
        for keyword in task_state_assignment.value.keywords
        if keyword.arg == "params"
    )
    assert isinstance(params_keyword.value, ast.Dict)
    configured_arguments = {
        key.value
        for key in params_keyword.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }

    assert required_arguments <= configured_arguments
    assert configured_arguments <= valid_arguments
