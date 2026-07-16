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
    controller_namespace: dict[str, object] = {}
    controller_path = TASK_DIR / "controller_math.py"
    controller_tree = ast.parse(controller_path.read_text())
    delta_function = next(
        node
        for node in controller_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "discounted_potential_delta"
    )
    exec(
        compile(
            ast.Module(body=[delta_function], type_ignores=[]),
            str(controller_path),
            "exec",
        ),
        {"torch": torch, **controller_namespace},
        controller_namespace,
    )
    namespace = {
        "torch": torch,
        "ManagerTermBase": _ManagerTermBase,
        "RewardTermCfg": object,
        "ManagerBasedRLEnv": object,
        "discounted_potential_delta": controller_namespace[
            "discounted_potential_delta"
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


def _make_env():
    state = SimpleNamespace(
        grasp_acquired=torch.tensor([False]),
        lift_completed=torch.tensor([False]),
        just_lifted=torch.tensor([False]),
        failed=torch.tensor([False]),
        released=torch.tensor([False]),
        verified_lift_coupled_progress=torch.tensor([0.0]),
    )
    return SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.04,
        episode_length_buf=torch.tensor([0]),
        max_episode_length=400,
        _openarm_pick_place_state=state,
    )


def test_verified_lift_reward_rebases_on_grasp_then_tracks_signed_progress():
    env = _make_env()
    reward = _load_reward_class()(None, env)
    state = env._openarm_pick_place_state

    assert reward(env, discount_factor=0.995).item() == 0.0

    state.grasp_acquired[:] = True
    state.verified_lift_coupled_progress[:] = 0.2
    assert reward(env, discount_factor=0.995).item() == 0.0

    state.verified_lift_coupled_progress[:] = 0.6
    expected_rise = (0.995 * 0.6 - 0.2) / env.step_dt
    assert reward(env, discount_factor=0.995).item() == pytest.approx(
        expected_rise, abs=1.0e-5
    )

    state.verified_lift_coupled_progress[:] = 0.3
    expected_fall = (0.995 * 0.3 - 0.6) / env.step_dt
    assert reward(env, discount_factor=0.995).item() == pytest.approx(
        expected_fall, abs=1.0e-5
    )


def test_verified_lift_reward_closes_timeout_potential():
    env = _make_env()
    reward = _load_reward_class()(None, env)
    state = env._openarm_pick_place_state
    state.grasp_acquired[:] = True
    state.verified_lift_coupled_progress[:] = 0.25
    reward(env, discount_factor=0.995)

    state.verified_lift_coupled_progress[:] = 0.7
    reward(env, discount_factor=0.995)
    env.episode_length_buf[:] = env.max_episode_length

    assert reward(env, discount_factor=0.995).item() == pytest.approx(
        -0.7 / env.step_dt, abs=1.0e-5
    )
    assert reward._was_active.item() is False
    assert reward._previous_potential.item() == 0.0


def test_verified_lift_reward_pays_final_increment_once_on_lift_completion():
    env = _make_env()
    reward = _load_reward_class()(None, env)
    state = env._openarm_pick_place_state
    state.grasp_acquired[:] = True
    state.verified_lift_coupled_progress[:] = 0.4
    reward(env, discount_factor=0.995)

    state.lift_completed[:] = True
    state.just_lifted[:] = True
    state.verified_lift_coupled_progress[:] = 1.0
    expected_final = (0.995 - 0.4) / env.step_dt
    assert reward(env, discount_factor=0.995).item() == pytest.approx(
        expected_final, abs=1.0e-5
    )

    state.just_lifted[:] = False
    assert reward(env, discount_factor=0.995).item() == 0.0
    assert reward._was_active.item() is False
