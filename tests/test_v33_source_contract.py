from __future__ import annotations

import ast
import re
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
CONTROLLER_SOURCE = (TASK_DIR / "controller_math.py").read_text()
PPO_SOURCE = (TASK_DIR / "agents" / "rsl_rl_ppo_cfg.py").read_text()
AUTHORITY_PPO_SOURCE = (TASK_DIR / "agents" / "authority_masked_ppo.py").read_text()
TRAIN_SOURCE = (ROOT / "scripts" / "rsl_rl" / "train.py").read_text()
PLAY_SOURCE = (ROOT / "scripts" / "rsl_rl" / "play.py").read_text()
SPAWNER_SOURCE = (
    ROOT
    / "source"
    / "openarm_testusd_task"
    / "openarm_testusd_task"
    / "tasks"
    / "testusd_reach"
    / "original_usd_spawner.py"
).read_text()
REACH_ENV_SOURCE = (
    ROOT
    / "source"
    / "openarm_testusd_task"
    / "openarm_testusd_task"
    / "tasks"
    / "testusd_reach"
    / "testusd_reach_env_cfg.py"
).read_text()
PICK_PLACE_REGISTRY_SOURCE = (TASK_DIR / "__init__.py").read_text()
REACH_REGISTRY_SOURCE = (
    ROOT
    / "source"
    / "openarm_testusd_task"
    / "openarm_testusd_task"
    / "tasks"
    / "testusd_reach"
    / "__init__.py"
).read_text()


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
    assert len(reward_terms) == 22


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
        "vertical_position_deadband=0.0015",
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
    assert "phase_primary_axis_progress_valid" in MDP_SOURCE
    assert "use_scaled = requested_motion & ~candidate_valid & scaled_valid" in MDP_SOURCE
    assert "def solve_single_axis_candidate" in MDP_SOURCE
    assert "select_phase_axis_fallback" in MDP_SOURCE
    assert "prefer_z_progress = state.post_center_descent_z_priority[assist_ids]" in MDP_SOURCE
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


def test_v38_restores_actor_transport_and_localizes_clearance_cost():
    assert "STAGE_CARRY_MIN_HEIGHT_W = 0.73" in ENV_SOURCE
    assert "STAGE_CARRY_TARGET_HEIGHT_W = 0.75" in ENV_SOURCE
    assert "STAGE_CARRY_MAX_HEIGHT_W = 0.79" in ENV_SOURCE
    assert "LOW_CLEARANCE_PENALTY_START_HEIGHT_W = 0.710" in ENV_SOURCE
    assert "LOW_CLEARANCE_PENALTY_FULL_HEIGHT_W = STAGE_CARRY_MIN_HEIGHT_W" in ENV_SOURCE
    assert "LOW_CLEARANCE_BASKET_APPROACH_XY = 0.17" in ENV_SOURCE
    assert "low_clearance_height=LOW_CLEARANCE_PENALTY_FULL_HEIGHT_W" in ENV_SOURCE
    assert "weight=-200.0" in ENV_SOURCE
    assert "lift_first_transport_goal" not in MDP_SOURCE
    assert "lift_first_actor_phase" not in MDP_SOURCE
    assert "task_preclearance_inward_progress" in MDP_SOURCE
    assert "pregrasp_opening_quality_factor" not in MDP_SOURCE
    assert "task_low_clearance_basket_approach_steps" in MDP_SOURCE
    assert MDP_SOURCE.count("actor_owned_carry_phase(") == 2
    assert '"near_basket_xy_threshold": LOW_CLEARANCE_BASKET_APPROACH_XY' in ENV_SOURCE
    assert "previous_local_distance - local_distance" in CONTROLLER_SOURCE
    assert "normalized_offset = offset_error / torch.clamp(gate_span, min=1.0e-6)" in MDP_SOURCE
    assert "return pose_quality * symmetry_quality * proximity_quality" in MDP_SOURCE
    assert MDP_SOURCE.count("self.cfg.minimum_active_progress_ratio") >= 7
    assert "ContactSensor" not in ENV_SOURCE
    assert "ContactSensor" not in MDP_SOURCE


def test_v40_decouples_phase_progress_and_accelerates_post_center_descent():
    assert '"post_center_descent_xy_threshold": 0.030' in ENV_SOURCE
    assert '"post_center_descent_steps": 12' in ENV_SOURCE
    assert '"post_center_descent_max_tracking_error": 0.012' in ENV_SOURCE
    assert "minimum_companion_progress_ratio=0.05" in ENV_SOURCE
    assert "minimum_companion_progress_ratio: float = 0.05" in MDP_SOURCE
    assert MDP_SOURCE.count("phase_primary_axis_progress_valid(") >= 6
    assert MDP_SOURCE.count("self.cfg.minimum_companion_progress_ratio") >= 7
    assert "advance_post_center_descent_progress" in MDP_SOURCE
    assert "post_center_descent_readiness" in MDP_SOURCE
    assert "state.post_center_descent_z_priority[assist_ids]" in MDP_SOURCE
    assert "self.post_center_descent_z_priority[:] = (" in MDP_SOURCE
    assert "self.object_height >= carry_height" in MDP_SOURCE
    assert MDP_SOURCE.count("allow_stabilization=True") == 1
    assert "raw_xy_progress_floor = torch.where(" in MDP_SOURCE
    assert "raw_z_progress_floor = torch.where(" in MDP_SOURCE
    assert "phase_primary_translation_progress_ratio" in MDP_SOURCE
    assert "select_higher_phase_primary_candidate" in MDP_SOURCE
    assert "phase_primary_xy, phase_primary_z = select_phase_primary_axis(" in MDP_SOURCE
    for metric in (
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
        "task_assist_entry_joint_limit_proximity",
        "task_assist_z_priority_control_steps",
        "task_assist_incremental_z_priority_upgrades",
    ):
        assert metric in MDP_SOURCE

    assert "def required_axis_progress_valid" in CONTROLLER_SOURCE
    assert "def phase_primary_axis_progress_valid" in CONTROLLER_SOURCE
    assert "def phase_primary_translation_progress_ratio" in CONTROLLER_SOURCE


def test_v41_masks_only_post_grasp_automation_and_keeps_terminal_contract():
    assert "self.is_finite_horizon = True" in ENV_SOURCE
    assert "class AuthorityCfg(ObsGroup)" in ENV_SOURCE
    assert "func=mdp.task_actor_authority_observation" in ENV_SOURCE
    assert 'obs_groups = {"policy": ["policy"], "critic": ["policy"]}' in PPO_SOURCE
    assert PPO_SOURCE.count("policy = RslRlPpoActorCriticCfg(") == 2
    assert "ArmOnlyLogProbActorCritic" not in PPO_SOURCE
    assert "ArmOnlyLogProbActorCritic" not in AUTHORITY_PPO_SOURCE
    assert 'class_name: str = "AuthorityMaskedPPO"' in PPO_SOURCE
    assert "advantages.zero_()" in AUTHORITY_PPO_SOURCE
    assert "advantages[authority_mask] = normalized" in AUTHORITY_PPO_SOURCE
    assert "actor_authority_fraction" in AUTHORITY_PPO_SOURCE
    assert "register_step_pre_hook" not in AUTHORITY_PPO_SOURCE
    assert "register_step_post_hook" not in AUTHORITY_PPO_SOURCE
    assert "actor_step_scale" not in AUTHORITY_PPO_SOURCE
    assert 'loss_dict["actor_parameter_delta_norm"]' in AUTHORITY_PPO_SOURCE
    assert "register_openarm_rsl_rl_classes()" in TRAIN_SOURCE
    assert "register_openarm_rsl_rl_classes()" in PLAY_SOURCE

    authority_source = MDP_SOURCE.split(
        "def task_actor_authority_observation", maxsplit=1
    )[1].split("def task_grasp_quality_observation", maxsplit=1)[0]
    assert "post_grasp_actor_authority(" in authority_source
    assert "def post_grasp_actor_authority" in CONTROLLER_SOURCE
    for pregrasp_mask in (
        "imminent_grasp_control",
        "closing_hold",
        "retry_recovery",
    ):
        assert pregrasp_mask not in authority_source


def test_v41_restores_the_verified_dense_pregrasp_learning_contract():
    reward_source = ENV_SOURCE.split("class RewardsCfgV18", maxsplit=1)[1].split(
        "class TerminationsCfgV18", maxsplit=1
    )[0]
    assert "func=mdp.task_pregrasp_centering_reward" in reward_source
    assert "func=mdp.task_pregrasp_signed_potential_progress" not in reward_source
    pregrasp_source = reward_source.split("pregrasp_centering =", maxsplit=1)[1].split(
        "lift_progress =", maxsplit=1
    )[0]
    assert '"quality_scale": 5.0' in pregrasp_source
    assert '"dwell_scale": 1.0' in pregrasp_source
    assert "weight=1.0" in pregrasp_source
    assert "task_gripper_action_phase_error" in reward_source
    assert "weight=-1.0" in reward_source


def test_v44_bridges_verified_lift_without_a_phase_threshold():
    reward_source = ENV_SOURCE.split("class RewardsCfgV18", maxsplit=1)[1].split(
        "class TerminationsCfgV18", maxsplit=1
    )[0]
    assert "func=mdp.task_verified_paired_lift_progress" in reward_source
    assert '"target_height": STAGE_CARRY_MIN_HEIGHT_W' in reward_source
    assert '"maximum_credit_height": 0.093' in reward_source
    assert '"bootstrap_length_scale": 0.010' in reward_source
    assert '"bootstrap_multiplier": 4.0' in reward_source
    assert '"discount_factor"' not in reward_source
    assert "gamma=0.995" in PPO_SOURCE
    assert "weight=150.0" in reward_source
    assert 'params={"preassist_multiplier": 1.0}' not in reward_source
    assert reward_source.count('params={"preassist_multiplier": 2.0}') == 2
    assert "self.is_finite_horizon = True" in ENV_SOURCE

    reward_class = next(
        node
        for node in ast.parse(ENV_SOURCE).body
        if isinstance(node, ast.ClassDef) and node.name == "RewardsCfgV18"
    )
    reward_assignments = {
        node.targets[0].id: node.value
        for node in reward_class.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
    }

    def keyword_value(term_name: str, keyword_name: str):
        term = reward_assignments[term_name]
        keyword = next(item for item in term.keywords if item.arg == keyword_name)
        return ast.literal_eval(keyword.value)

    def parameter_value(term_name: str, parameter_name: str):
        term = reward_assignments[term_name]
        params = next(item.value for item in term.keywords if item.arg == "params")
        value = next(
            value
            for key, value in zip(params.keys, params.values, strict=True)
            if ast.literal_eval(key) == parameter_name
        )
        return ast.literal_eval(value)

    acquired_event_credit = sum(
        keyword_value(term_name, "weight")
        for term_name in ("grasp_ready", "grasp_candidate", "grasp_acquired")
    )
    lift_event_credit = keyword_value("lift_completed", "weight")
    verified_lift_credit_cap = keyword_value(
        "verified_lift_progress", "weight"
    ) * (
        parameter_value("verified_lift_progress", "maximum_credit_height")
        + parameter_value("verified_lift_progress", "bootstrap_length_scale")
        * parameter_value("verified_lift_progress", "bootstrap_multiplier")
    )
    failure_cost = abs(keyword_value("failed_place", "weight"))
    failure_multiplier = keyword_value("failed_place", "params")[
        "preassist_multiplier"
    ]
    timeout_multiplier = keyword_value("incomplete_timeout", "params")[
        "preassist_multiplier"
    ]
    preassist_delay_weight = keyword_value("preassist_delay", "weight")
    assert failure_cost * failure_multiplier > (
        acquired_event_credit + lift_event_credit + verified_lift_credit_cap
    )
    assert timeout_multiplier == 2.0
    assert preassist_delay_weight == -1.5

    paired_source = MDP_SOURCE.split(
        "class task_verified_paired_lift_progress", maxsplit=1
    )[1].split("class task_preclearance_inward_progress", maxsplit=1)[0]
    assert "capped_raw_coupled_lift_rise(" in paired_source
    assert "saturating_lift_bootstrap_equivalent_rise(" in paired_source
    assert "state.grasp_acquired" in paired_source
    assert "state.current_hold" in paired_source
    assert "state.just_lifted" in paired_source
    assert "acquired_nonterminal" in paired_source
    assert "_observed_watermark" in paired_source
    assert "_credited_cumulative_rise" in paired_source
    assert "_bootstrap_credited_cumulative_rise" in paired_source
    assert "discounted_potential_delta(" not in paired_source
    assert "closed_terminal" not in paired_source
    assert "~state.just_grasped" not in paired_source

    lift_source = MDP_SOURCE.split('elif mode == "lift":', maxsplit=1)[1].split(
        'elif mode == "transport":', maxsplit=1
    )[0]
    assert "state.grasp_candidate_tracking | state.just_grasped" in lift_source
    assert "state.grasp_acquired" not in lift_source
    for metric in (
        "task_verified_lift_object_progress",
        "task_verified_lift_tcp_progress",
        "task_verified_lift_coupled_progress",
        "task_verified_lift_observed_watermark",
        "task_verified_lift_credited_rise",
        "task_verified_lift_bootstrap_credited_rise",
    ):
        assert metric in MDP_SOURCE
    assert "def saturating_lift_bootstrap_equivalent_rise" in CONTROLLER_SOURCE


def test_openarm_scenes_fail_closed_with_self_collisions_enabled():
    required_spawner_contract = (
        'OPENARM_ROBOT_REL_PATH = "openarm_bimanual"',
        'OPENARM_ARTICULATION_ROOT_PRIM_PATH = "/root_joint"',
        "OPENARM_ARTICULATION_REL_PATH = OPENARM_ROBOT_REL_PATH + OPENARM_ARTICULATION_ROOT_PRIM_PATH",
        "references.ClearReferences()",
        "Referenced source did not contribute",
        "articulation_prim.HasAPI(UsdPhysics.ArticulationRootAPI)",
        "PhysxSchema.PhysxArticulationAPI.Apply(articulation_prim)",
        "Usd.EditContext(stage, Usd.EditTarget(stage.GetRootLayer()))",
        "stage.GetRootLayer().GetAttributeAtPath(enabled_attr.GetPath())",
        "enabled_attr.Set(True)",
        "enabled_attr.Get() is not True",
        "def assert_openarm_self_collision_contract(env, env_ids)",
        "for env_prim_path in env_prim_paths:",
        "OPENARM_SELF_COLLISION_FILTER_PAIRS",
        "def _author_openarm_self_collision_filters",
        "UsdPhysics.FilteredPairsAPI.Apply(source_prim)",
        "CreateFilteredPairsRel()",
        "SetTargets([target_sdf_path])",
        "GetRelationshipAtPath(",
        "root_relationship.targetPathList.GetAppliedItems()",
        "for prim in Usd.PrimRange(robot_prim):",
        "observed_filters != expected_filters",
        "Self-collision filters are invalid",
    )
    for contract_line in required_spawner_contract:
        assert contract_line in SPAWNER_SOURCE

    spawn_source = SPAWNER_SOURCE.split("def spawn_original_usd_scene", maxsplit=1)[1]
    reference_index = spawn_source.index("scene_prim = _reference_prim(")
    enable_index = spawn_source.index("_enable_articulation_self_collisions(stage, articulation_path)")
    filter_index = spawn_source.index("_author_openarm_self_collision_filters(stage, robot_path)")
    standardize_index = spawn_source.index("standardize_xform_ops(")
    assert reference_index < enable_index < filter_index < standardize_index

    for env_source in (ENV_SOURCE, REACH_ENV_SOURCE):
        assert 'cfg.prim_path = f"{{ENV_REGEX_NS}}/Scene/{OPENARM_ROBOT_REL_PATH}"' in env_source
        assert "cfg.articulation_root_prim_path = OPENARM_ARTICULATION_ROOT_PRIM_PATH" in env_source
        assert "cfg.spawn = None" in env_source
        assert "func=assert_openarm_self_collision_contract, mode=\"startup\"" in env_source
        assert "apply_openarm_self_collision_physx_profile(self.sim.physx)" in env_source
        assert "self_collision_articulation_paths" not in env_source
    assert "Set(False)" not in SPAWNER_SOURCE


def test_pick_place_reserves_contact_rich_physx_buffers():
    required_buffers = (
        "gpu_max_rigid_contact_count = 2**23",
        "gpu_max_rigid_patch_count = 2**23",
        "gpu_found_lost_pairs_capacity = 2**22",
        "gpu_found_lost_aggregate_pairs_capacity = 2**25",
        "gpu_total_aggregate_pairs_capacity = 2**21",
        "gpu_collision_stack_size = 2**28",
        "gpu_heap_capacity = 2**27",
        "gpu_temp_buffer_capacity = 2**25",
    )
    for buffer_contract in required_buffers:
        assert buffer_contract in SPAWNER_SOURCE


def test_every_registered_openarm_task_inherits_the_verified_scene_contract():
    def registered_env_cfg_names(registry_source: str) -> set[str]:
        return set(re.findall(r"env_cfg_entry_point.*:([A-Za-z0-9_]+)", registry_source))

    def class_bases(source: str) -> dict[str, set[str]]:
        tree = ast.parse(source)
        return {
            node.name: {base.id for base in node.bases if isinstance(base, ast.Name)}
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        }

    def inherits(class_name: str, required_base: str, bases: dict[str, set[str]]) -> bool:
        if class_name == required_base:
            return True
        return any(inherits(base, required_base, bases) for base in bases.get(class_name, set()))

    for registry_source, env_source, required_base in (
        (PICK_PLACE_REGISTRY_SOURCE, ENV_SOURCE, "OpenArmTestUsdPickPlaceEnvCfg"),
        (REACH_REGISTRY_SOURCE, REACH_ENV_SOURCE, "OpenArmTestUsdReachEnvCfg"),
    ):
        bases = class_bases(env_source)
        registered_names = registered_env_cfg_names(registry_source)
        assert registered_names
        assert all(inherits(name, required_base, bases) for name in registered_names)


def test_v40_release_has_no_hidden_actor_control_after_authority_is_removed():
    assert "& ~state.release_commanded" in MDP_SOURCE
    assert MDP_SOURCE.count("automatic_release_ready(") == 2
    assert "def _imminent_automatic_release(" in MDP_SOURCE
    assert "imminent_automatic_release | state.release_commanded | state.released" in MDP_SOURCE
    assert "& ~imminent_automatic_release" in MDP_SOURCE
    assert "self._processed_actions[release_hold] = current_joint_pos[release_hold]" in MDP_SOURCE
    assert "self._handoff_steps_remaining[release_hold] = 0" in MDP_SOURCE
    assert "| release_hold" in MDP_SOURCE
    actions_source = ENV_SOURCE.split("class ActionsCfg:", maxsplit=1)[1].split(
        "class ObservationsCfg:", maxsplit=1
    )[0]
    assert actions_source.index("arm_action") < actions_source.index("gripper_action")


def test_v40_z_fallback_and_proactive_upgrade_metrics_remain_disjoint():
    assert "unresolved_z_fallback = use_z_fallback.clone()" in MDP_SOURCE
    assert "use_z_fallback |= z_priority_upgrade" in MDP_SOURCE
    assert "unresolved_z_fallback.float()" in MDP_SOURCE


def test_preassist_call_sites_start_only_after_verified_grasp():
    tree = ast.parse(MDP_SOURCE)
    expected_counts = {"actor_owned_carry_phase": 2}
    for function_name, expected_count in expected_counts.items():
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == function_name
        ]
        assert len(calls) == expected_count
        for call in calls:
            first_argument = call.args[0]
            assert isinstance(first_argument, ast.Attribute)
            assert isinstance(first_argument.value, ast.Name)
            assert first_argument.value.id == "state"
            assert first_argument.attr == "grasp_acquired"


def test_actor_transport_observations_keep_the_v35_goal_contract():
    tree = ast.parse(MDP_SOURCE)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "transport_goal_command",
            "object_to_transport_goal_position_in_robot_root_frame",
        }
    }
    assert set(functions) == {
        "transport_goal_command",
        "object_to_transport_goal_position_in_robot_root_frame",
    }
    target_source = ast.unparse(functions["transport_goal_command"])
    relative_source = ast.unparse(
        functions["object_to_transport_goal_position_in_robot_root_frame"]
    )
    expected_target_source = ast.unparse(
        ast.parse(
            '''
def transport_goal_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    carry_height_b: float,
) -> torch.Tensor:
    """Expose the actual goal XY at the carry height used before placement."""
    command = env.command_manager.get_command(command_name).clone()
    command[:, 2] = carry_height_b
    return command
'''
        ).body[0]
    )
    expected_relative_source = ast.unparse(
        ast.parse(
            '''
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
'''
        ).body[0]
    )
    assert target_source == expected_target_source
    assert relative_source == expected_relative_source
    policy_source = ENV_SOURCE.split("class ObservationsCfgV18", maxsplit=1)[1].split(
        "class RewardsCfgV18", maxsplit=1
    )[0]
    assert "unlock_start_height" not in policy_source
    assert "unlock_full_height" not in policy_source
    assert "lift_first" not in policy_source


def test_pregrasp_reward_excludes_every_automatic_pose_hold():
    assert "state.grasp_controller_owns_arm[:] = grasp_control" in MDP_SOURCE
    assert 'phase_masks["close_verify"] = remaining & self.grasp_attempt_active' in MDP_SOURCE
    assert "& ~state.grasp_controller_owns_arm" in MDP_SOURCE
    assert "& ~state.assist_owns_arm" in MDP_SOURCE
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
