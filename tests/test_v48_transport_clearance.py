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
CONTROLLER_PATH = TASK_DIR / "controller_math.py"
MDP_PATH = TASK_DIR / "mdp.py"


def _load_function(path: Path, name: str):
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


class _ManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


class _SceneEntityCfg:
    def __init__(self, name: str):
        self.name = name


def _load_potential_class():
    tree = ast.parse(MDP_PATH.read_text())
    reward_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "task_potential_progress"
    )
    namespace = {
        "torch": torch,
        "ManagerTermBase": _ManagerTermBase,
        "RewardTermCfg": object,
        "ManagerBasedRLEnv": object,
        "SceneEntityCfg": _SceneEntityCfg,
        "RigidObject": object,
    }
    exec(
        compile(ast.Module(body=[reward_class], type_ignores=[]), str(MDP_PATH), "exec"),
        namespace,
    )
    return namespace[reward_class.name]


def _exposure_inputs():
    return {
        "object_height": torch.tensor([0.7299]),
        "distance_xy": torch.tensor([0.17]),
        "actor_carry_active": torch.tensor([True]),
        "current_hold": torch.tensor([True]),
        "lift_completed": torch.tensor([True]),
        "clearance_height": 0.730,
        "near_basket_xy_threshold": 0.17,
    }


def test_poststicky_exposure_matches_the_binary_clearance_boundary():
    exposure_cost = _load_function(
        CONTROLLER_PATH, "poststicky_low_clearance_exposure_cost"
    )
    values = _exposure_inputs()
    values["object_height"] = torch.tensor([0.7299, 0.7300, float("nan"), 0.7000])
    values["distance_xy"] = torch.tensor([0.1700, 0.1700, 0.1000, 0.1701])
    values["actor_carry_active"] = torch.ones(4, dtype=torch.bool)
    values["current_hold"] = torch.ones(4, dtype=torch.bool)
    values["lift_completed"] = torch.ones(4, dtype=torch.bool)

    result = exposure_cost(**values)

    assert result.tolist() == [1.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("gate", "value"),
    (
        ("actor_carry_active", False),
        ("current_hold", False),
        ("lift_completed", False),
    ),
)
def test_poststicky_exposure_excludes_every_noncarrying_phase(gate, value):
    exposure_cost = _load_function(
        CONTROLLER_PATH, "poststicky_low_clearance_exposure_cost"
    )
    values = _exposure_inputs()
    values[gate] = torch.tensor([value])

    assert exposure_cost(**values).item() == 0.0


def test_poststicky_exposure_ledger_is_dt_invariant():
    exposure_cost = _load_function(
        CONTROLLER_PATH, "poststicky_low_clearance_exposure_cost"
    )
    exposure = exposure_cost(**_exposure_inputs())

    for control_hz, controls in ((30, 60), (60, 120)):
        integrated_cost = exposure * (-5.0) * (1.0 / control_hz) * controls
        assert integrated_cost.item() == pytest.approx(-10.0)


def test_poststicky_exposure_rejects_a_nonpositive_basket_radius():
    exposure_cost = _load_function(
        CONTROLLER_PATH, "poststicky_low_clearance_exposure_cost"
    )
    values = _exposure_inputs()
    values["near_basket_xy_threshold"] = 0.0

    with pytest.raises(ValueError, match="must be positive"):
        exposure_cost(**values)


def test_transport_progress_requires_current_clearance():
    num_envs = 3
    state = SimpleNamespace(
        grasp_acquired=torch.ones(num_envs, dtype=torch.bool),
        current_hold=torch.ones(num_envs, dtype=torch.bool),
        lift_completed=torch.ones(num_envs, dtype=torch.bool),
        assist_active=torch.zeros(num_envs, dtype=torch.bool),
        released=torch.zeros(num_envs, dtype=torch.bool),
        distance_xy=torch.full((num_envs,), 0.20),
        object_height=torch.full((num_envs,), 0.730),
    )
    obj = SimpleNamespace(
        data=SimpleNamespace(root_pos_w=torch.zeros((num_envs, 3)))
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        step_dt=1.0 / 30.0,
        scene={"object": obj},
        _openarm_pick_place_state=state,
    )
    reward = _load_potential_class()(None, env)
    reward(
        env,
        mode="transport",
        max_step_change=0.03,
        minimum_transport_height=0.730,
    )

    state.distance_xy[:] = 0.19
    state.object_height[:] = torch.tensor([0.7299, 0.7300, float("nan")])
    result = reward(
        env,
        mode="transport",
        max_step_change=0.03,
        minimum_transport_height=0.730,
    )

    assert result[0].item() == 0.0
    assert result[1].item() == pytest.approx(0.30)
    assert result[2].item() == 0.0


def test_transport_progress_does_not_retroactively_credit_low_motion():
    state = SimpleNamespace(
        grasp_acquired=torch.tensor([True]),
        current_hold=torch.tensor([True]),
        lift_completed=torch.tensor([True]),
        assist_active=torch.tensor([False]),
        released=torch.tensor([False]),
        distance_xy=torch.tensor([0.20]),
        object_height=torch.tensor([0.730]),
    )
    obj = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros((1, 3))))
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=1.0 / 30.0,
        scene={"object": obj},
        _openarm_pick_place_state=state,
    )
    reward = _load_potential_class()(None, env)
    def call():
        return reward(
            env,
            mode="transport",
            max_step_change=0.03,
            minimum_transport_height=0.730,
        )

    assert call().item() == 0.0

    state.object_height[:] = 0.729
    state.distance_xy[:] = 0.15
    assert call().item() == 0.0
    state.object_height[:] = 0.730
    assert call().item() == 0.0
    state.distance_xy[:] = 0.14

    assert call().item() == pytest.approx(0.30)
