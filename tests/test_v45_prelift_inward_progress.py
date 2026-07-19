from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = (
    ROOT
    / "source"
    / "openarm_testusd_task"
    / "openarm_testusd_task"
    / "tasks"
    / "testusd_pick_place"
)
FREE_TOLERANCE = 0.010
MAXIMUM_CHARGED_EXCURSION = 0.050
REWARD_WEIGHT = -300.0


class _ManagerTermBase:
    def __init__(self, cfg, env):
        del cfg, env


def _load_reward_class():
    source_path = TASK_DIR / "mdp.py"
    tree = ast.parse(source_path.read_text())
    reward_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "task_prelift_inward_progress"
    )
    controller_path = TASK_DIR / "controller_math.py"
    controller_tree = ast.parse(controller_path.read_text())
    helper = next(
        node
        for node in controller_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "prelift_inward_progress_increment"
    )
    helper_namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(controller_path), "exec"),
        {"torch": torch},
        helper_namespace,
    )
    namespace = {
        "torch": torch,
        "ManagerTermBase": _ManagerTermBase,
        "RewardTermCfg": object,
        "ManagerBasedRLEnv": object,
        "TaskStateTracker": object,
        "prelift_inward_progress_increment": helper_namespace[
            "prelift_inward_progress_increment"
        ],
    }
    exec(
        compile(ast.Module(body=[reward_class], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace[reward_class.name]


def _make_env(num_envs: int = 1, step_dt: float = 0.04):
    state = SimpleNamespace(
        grasp_acquired=torch.zeros(num_envs, dtype=torch.bool),
        just_grasped=torch.zeros(num_envs, dtype=torch.bool),
        lift_completed=torch.zeros(num_envs, dtype=torch.bool),
        just_lifted=torch.zeros(num_envs, dtype=torch.bool),
        failed=torch.zeros(num_envs, dtype=torch.bool),
        released=torch.zeros(num_envs, dtype=torch.bool),
        distance_xy=torch.full((num_envs,), 0.222),
        transport_displacement=torch.zeros(num_envs),
        prelift_lateral_displacement_max=torch.zeros(num_envs),
        prelift_inward_progress_max=torch.zeros(num_envs),
        prelift_inward_excursion_charged=torch.zeros(num_envs),
        prelift_inward_cap_hit=torch.zeros(num_envs, dtype=torch.bool),
    )
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        step_dt=step_dt,
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        max_episode_length=400,
        _openarm_pick_place_state=state,
    )


def _call(reward, env):
    return reward(
        env,
        free_tolerance=FREE_TOLERANCE,
        maximum_charged_excursion=MAXIMUM_CHARGED_EXCURSION,
    )


def _latch_grasp(reward, env):
    state = env._openarm_pick_place_state
    state.grasp_acquired[:] = True
    state.just_grasped[:] = True
    assert torch.all(_call(reward, env) == 0.0)
    state.just_grasped[:] = False


def test_prelift_cost_charges_maximum_inward_progress_once_without_a_refund():
    env = _make_env()
    reward = _load_reward_class()(None, env)
    state = env._openarm_pick_place_state
    _latch_grasp(reward, env)

    outputs = []
    for distance, displacement in (
        (0.217, 0.005),
        (0.202, 0.020),
        (0.212, 0.010),
        (0.172, 0.050),
        (0.192, 0.030),
        (0.172, 0.050),
    ):
        state.distance_xy[:] = distance
        state.transport_displacement[:] = displacement
        outputs.append(_call(reward, env))

    charged_distance = sum(outputs) * env.step_dt
    assert charged_distance.item() == pytest.approx(0.040)
    assert reward._maximum_lateral_displacement.item() == pytest.approx(0.050)
    assert reward._maximum_inward_progress.item() == pytest.approx(0.050)
    assert reward._charged_excursion.item() == pytest.approx(0.040)
    assert state.prelift_inward_cap_hit.item() is False


def test_prelift_cost_ignores_outward_and_equal_radius_motion():
    env = _make_env()
    reward = _load_reward_class()(None, env)
    state = env._openarm_pick_place_state
    _latch_grasp(reward, env)

    state.distance_xy[:] = 0.232
    state.transport_displacement[:] = 0.010
    outward = _call(reward, env)
    state.distance_xy[:] = 0.222
    state.transport_displacement[:] = 0.020
    equal_radius = _call(reward, env)

    assert outward.item() == 0.0
    assert equal_radius.item() == 0.0
    assert reward._maximum_lateral_displacement.item() == pytest.approx(0.020)
    assert reward._maximum_inward_progress.item() == 0.0


def test_prelift_cost_includes_the_lift_completion_control_only_once():
    env = _make_env()
    reward = _load_reward_class()(None, env)
    state = env._openarm_pick_place_state
    _latch_grasp(reward, env)

    state.lift_completed[:] = True
    state.just_lifted[:] = True
    state.distance_xy[:] = 0.182
    state.transport_displacement[:] = 0.040
    transition_cost = _call(reward, env)
    state.just_lifted[:] = False
    state.distance_xy[:] = 0.142
    state.transport_displacement[:] = 0.080
    post_lift_cost = _call(reward, env)

    assert (transition_cost * env.step_dt).item() == pytest.approx(0.030)
    assert post_lift_cost.item() == 0.0
    assert reward._maximum_inward_progress.item() == pytest.approx(0.040)


@pytest.mark.parametrize("terminal_kind", ["failed", "released", "timeout"])
def test_prelift_cost_charges_before_terminal_deactivation(terminal_kind):
    env = _make_env()
    reward = _load_reward_class()(None, env)
    state = env._openarm_pick_place_state
    _latch_grasp(reward, env)

    state.distance_xy[:] = 0.182
    state.transport_displacement[:] = 0.040
    if terminal_kind == "timeout":
        env.episode_length_buf[:] = env.max_episode_length
    else:
        getattr(state, terminal_kind)[:] = True

    terminal_cost = _call(reward, env)
    state.distance_xy[:] = 0.142
    next_cost = _call(reward, env)

    assert (terminal_cost * env.step_dt).item() == pytest.approx(0.030)
    assert next_cost.item() == 0.0
    assert reward._maximum_inward_progress.item() == pytest.approx(0.040)


def test_prelift_cost_is_timestep_invariant_and_capped_at_minus_fifteen():
    integrated_rewards = []
    for step_dt in (0.02, 0.04):
        env = _make_env(step_dt=step_dt)
        reward = _load_reward_class()(None, env)
        state = env._openarm_pick_place_state
        _latch_grasp(reward, env)
        state.distance_xy[:] = 0.100
        state.transport_displacement[:] = 0.122

        integrated_rewards.append(
            (_call(reward, env) * env.step_dt * REWARD_WEIGHT).item()
        )
        assert state.prelift_inward_cap_hit.item() is True
        assert reward._maximum_inward_progress.item() == pytest.approx(0.122)
        assert reward._charged_excursion.item() == pytest.approx(0.050)

    assert integrated_rewards == pytest.approx([-15.0, -15.0])


def test_prelift_cost_hands_off_at_the_existing_rim_boundary():
    env = _make_env()
    reward = _load_reward_class()(None, env)
    state = env._openarm_pick_place_state
    _latch_grasp(reward, env)

    state.distance_xy[:] = 0.170
    state.transport_displacement[:] = 0.052
    boundary_reward = _call(reward, env) * env.step_dt * REWARD_WEIGHT
    state.distance_xy[:] = 0.162
    state.transport_displacement[:] = 0.060
    overlap_reward = _call(reward, env) * env.step_dt * REWARD_WEIGHT

    assert boundary_reward.item() == pytest.approx(-12.6, abs=1.0e-4)
    assert overlap_reward.item() == pytest.approx(-2.4, abs=1.0e-4)
    assert (boundary_reward + overlap_reward).item() == pytest.approx(-15.0)
    assert state.prelift_inward_cap_hit.item() is True


def test_prelift_cost_latches_the_initial_distance_once():
    env = _make_env()
    reward = _load_reward_class()(None, env)
    state = env._openarm_pick_place_state
    _latch_grasp(reward, env)

    state.distance_xy[:] = 0.202
    _call(reward, env)
    state.distance_xy[:] = 0.250
    _call(reward, env)

    assert reward._initial_distance_xy.item() == pytest.approx(0.222)
    assert reward._maximum_inward_progress.item() == pytest.approx(0.020)


def test_prelift_cost_reset_is_selective():
    env = _make_env(num_envs=2)
    reward = _load_reward_class()(None, env)
    state = env._openarm_pick_place_state
    _latch_grasp(reward, env)
    state.distance_xy[:] = torch.tensor([0.192, 0.172])
    state.transport_displacement[:] = torch.tensor([0.03, 0.05])
    _call(reward, env)

    reward.reset(torch.tensor([0]))

    assert reward._initial_distance_xy.tolist() == pytest.approx([0.0, 0.222])
    assert reward._maximum_lateral_displacement.tolist() == pytest.approx(
        [0.0, 0.05]
    )
    assert reward._maximum_inward_progress.tolist() == pytest.approx([0.0, 0.05])
    assert reward._charged_excursion.tolist() == pytest.approx([0.0, 0.04])
    assert reward._was_active.tolist() == [False, True]
