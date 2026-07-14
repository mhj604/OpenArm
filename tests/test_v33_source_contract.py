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
    assert len(reward_terms) == 20


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
