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
TARGET_HEIGHT = 0.730
MAXIMUM_CREDIT_HEIGHT = 0.093
REWARD_WEIGHT = 150.0


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
        and node.name == "task_verified_paired_lift_progress"
    )
    controller_path = TASK_DIR / "controller_math.py"
    controller_tree = ast.parse(controller_path.read_text())
    controller_functions = [
        node
        for node in controller_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "capped_raw_coupled_lift_rise"
    ]
    controller_namespace: dict[str, object] = {}
    exec(
        compile(
            ast.Module(body=controller_functions, type_ignores=[]),
            str(controller_path),
            "exec",
        ),
        {"torch": torch},
        controller_namespace,
    )
    namespace = {
        "torch": torch,
        "ManagerTermBase": _ManagerTermBase,
        "RewardTermCfg": object,
        "ManagerBasedRLEnv": object,
        "capped_raw_coupled_lift_rise": controller_namespace[
            "capped_raw_coupled_lift_rise"
        ],
    }
    exec(
        compile(
            ast.Module(body=[reward_class], type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    return namespace[reward_class.name]


def _make_env(num_envs: int = 1):
    origin = torch.zeros((num_envs, 3))
    origin[:, 2] = 0.637
    state = SimpleNamespace(
        grasp_acquired=torch.zeros(num_envs, dtype=torch.bool),
        current_hold=torch.zeros(num_envs, dtype=torch.bool),
        lift_completed=torch.zeros(num_envs, dtype=torch.bool),
        just_lifted=torch.zeros(num_envs, dtype=torch.bool),
        just_grasped=torch.zeros(num_envs, dtype=torch.bool),
        failed=torch.zeros(num_envs, dtype=torch.bool),
        released=torch.zeros(num_envs, dtype=torch.bool),
        object_height=origin[:, 2].clone(),
        tcp_height=origin[:, 2].clone(),
        grasp_object_origin=origin.clone(),
        grasp_tcp_origin=origin.clone(),
        verified_lift_observed_watermark=torch.zeros(num_envs),
        verified_lift_credited_rise=torch.zeros(num_envs),
    )
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        step_dt=0.04,
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        max_episode_length=400,
        _openarm_pick_place_state=state,
    )


def _call(reward, env):
    return reward(
        env,
        target_height=TARGET_HEIGHT,
        maximum_credit_height=MAXIMUM_CREDIT_HEIGHT,
    )


def _set_coupled_rise(state, rise):
    rise_tensor = torch.as_tensor(rise, dtype=state.object_height.dtype)
    state.object_height[:] = state.grasp_object_origin[:, 2] + rise_tensor
    state.tcp_height[:] = state.grasp_tcp_origin[:, 2] + rise_tensor


def test_verified_lift_reward_credits_each_episode_best_height_once():
    env = _make_env()
    reward = _load_reward_class()(None, env)
    state = env._openarm_pick_place_state
    state.grasp_acquired[:] = True
    state.current_hold[:] = True

    outputs = [_call(reward, env)]
    for rise in (0.01, 0.03, 0.01, 0.03):
        _set_coupled_rise(state, rise)
        outputs.append(_call(reward, env))

    weighted_episode_reward = sum(outputs) * env.step_dt * REWARD_WEIGHT
    assert weighted_episode_reward.item() == pytest.approx(
        REWARD_WEIGHT * 0.03,
        abs=1.0e-5,
    )
    assert reward._observed_watermark.item() == pytest.approx(0.03)
    assert reward._credited_cumulative_rise.item() == pytest.approx(0.03)


def test_verified_lift_reward_burns_height_first_seen_without_a_current_hold():
    env = _make_env()
    reward = _load_reward_class()(None, env)
    state = env._openarm_pick_place_state
    state.grasp_acquired[:] = True
    state.current_hold[:] = True
    _set_coupled_rise(state, 0.03)
    first_credit = _call(reward, env)

    state.current_hold[:] = False
    _set_coupled_rise(state, 0.05)
    burned = _call(reward, env)
    state.current_hold[:] = True
    recovered = _call(reward, env)
    _set_coupled_rise(state, 0.06)
    new_credit = _call(reward, env)

    assert (first_credit * env.step_dt).item() == pytest.approx(
        0.03
    )
    assert burned.item() == 0.0
    assert recovered.item() == 0.0
    assert (new_credit * env.step_dt).item() == pytest.approx(0.01)
    assert reward._observed_watermark.item() == pytest.approx(0.06)
    assert reward._credited_cumulative_rise.item() == pytest.approx(0.04)
    assert state.verified_lift_observed_watermark.item() == pytest.approx(0.06)
    assert state.verified_lift_credited_rise.item() == pytest.approx(0.04)


@pytest.mark.parametrize("terminal_kind", ["failed", "released", "timeout"])
def test_verified_lift_reward_gives_terminal_precedence_without_a_clawback(
    terminal_kind,
):
    env = _make_env()
    reward = _load_reward_class()(None, env)
    state = env._openarm_pick_place_state
    state.grasp_acquired[:] = True
    state.current_hold[:] = True
    _set_coupled_rise(state, 0.03)
    _call(reward, env)

    _set_coupled_rise(state, 0.06)
    state.lift_completed[:] = True
    state.just_lifted[:] = True
    if terminal_kind == "timeout":
        env.episode_length_buf[:] = env.max_episode_length
    else:
        getattr(state, terminal_kind)[:] = True

    assert _call(reward, env).item() == 0.0
    assert reward._observed_watermark.item() == pytest.approx(0.03)
    assert reward._credited_cumulative_rise.item() == pytest.approx(0.03)


def test_verified_lift_reward_pays_the_nonterminal_completion_increment_once():
    env = _make_env()
    reward = _load_reward_class()(None, env)
    state = env._openarm_pick_place_state
    state.grasp_acquired[:] = True
    state.current_hold[:] = True
    _set_coupled_rise(state, 0.04)
    _call(reward, env)

    state.lift_completed[:] = True
    state.just_lifted[:] = True
    _set_coupled_rise(state, 0.093)
    final_increment = _call(reward, env)
    state.just_lifted[:] = False

    assert (final_increment * env.step_dt).item() == pytest.approx(0.053)
    assert _call(reward, env).item() == 0.0
    assert reward._credited_cumulative_rise.item() == pytest.approx(0.093)


def test_verified_lift_reward_starts_at_zero_on_the_origin_latch_step():
    env = _make_env()
    reward = _load_reward_class()(None, env)
    state = env._openarm_pick_place_state
    state.grasp_acquired[:] = True
    state.current_hold[:] = True
    state.just_grasped[:] = True

    assert _call(reward, env).item() == 0.0
    assert reward._observed_watermark.item() == 0.0
    assert reward._credited_cumulative_rise.item() == 0.0


def test_verified_lift_reward_has_a_bounded_total_credit():
    env = _make_env()
    reward = _load_reward_class()(None, env)
    state = env._openarm_pick_place_state
    state.grasp_acquired[:] = True
    state.current_hold[:] = True
    _set_coupled_rise(state, 1.0)

    weighted_credit = _call(reward, env) * env.step_dt * REWARD_WEIGHT
    expected_credit = REWARD_WEIGHT * MAXIMUM_CREDIT_HEIGHT

    assert weighted_credit.item() == pytest.approx(expected_credit, abs=1.0e-5)
    assert _call(reward, env).item() == 0.0


def test_verified_lift_reward_reset_is_selective():
    env = _make_env(num_envs=2)
    reward = _load_reward_class()(None, env)
    state = env._openarm_pick_place_state
    state.grasp_acquired[:] = True
    state.current_hold[:] = True
    _set_coupled_rise(state, [0.02, 0.04])
    _call(reward, env)

    reward.reset(torch.tensor([0]))

    assert reward._observed_watermark.tolist() == pytest.approx([0.0, 0.04])
    assert reward._credited_cumulative_rise.tolist() == pytest.approx([0.0, 0.04])
