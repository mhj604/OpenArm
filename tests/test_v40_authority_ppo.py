from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


AUTHORITY_PPO_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "openarm_testusd_task"
    / "openarm_testusd_task"
    / "tasks"
    / "testusd_pick_place"
    / "agents"
    / "authority_masked_ppo.py"
)
SPEC = importlib.util.spec_from_file_location("openarm_authority_ppo", AUTHORITY_PPO_PATH)
assert SPEC is not None and SPEC.loader is not None
authority_ppo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(authority_ppo)

AuthorityMaskedPPO = authority_ppo.AuthorityMaskedPPO
register_openarm_rsl_rl_classes = authority_ppo.register_openarm_rsl_rl_classes


class _FakeStorage:
    def __init__(self, advantages: torch.Tensor, authority: torch.Tensor):
        self.advantages = advantages.clone()
        self.observations = {"authority": authority.clone()}
        self.returns = torch.arange(
            advantages.numel(), dtype=advantages.dtype
        ).reshape_as(advantages)
        self.values = -self.returns.clone()
        self.compute_arguments = None

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
        normalize_advantage: bool,
    ) -> None:
        self.compute_arguments = (last_values.clone(), gamma, lam, normalize_advantage)


def _make_algorithm(advantages: torch.Tensor, authority: torch.Tensor):
    algorithm = object.__new__(AuthorityMaskedPPO)
    algorithm.policy = SimpleNamespace(
        evaluate=lambda obs: torch.zeros((authority.shape[1], 1))
    )
    algorithm.storage = _FakeStorage(advantages, authority)
    algorithm.gamma = 0.995
    algorithm.lam = 0.95
    algorithm.authority_observation_group = "authority"
    algorithm.is_multi_gpu = False
    algorithm.gpu_world_size = 1
    algorithm.actor_authority_fraction = 0.0
    return algorithm


def test_authority_mask_normalizes_only_causal_actor_transitions():
    advantages = torch.tensor(
        [
            [[1.0], [20.0], [2.0]],
            [[30.0], [4.0], [40.0]],
        ]
    )
    authority = torch.tensor(
        [
            [[1.0], [0.0], [1.0]],
            [[0.0], [1.0], [0.0]],
        ]
    )
    algorithm = _make_algorithm(advantages, authority)
    returns_before = algorithm.storage.returns.clone()
    values_before = algorithm.storage.values.clone()

    algorithm.compute_returns(obs={})

    mask = authority[..., 0] > 0.5
    causal = algorithm.storage.advantages[mask]
    assert torch.mean(causal).item() == pytest.approx(0.0, abs=1.0e-6)
    assert torch.std(causal, unbiased=False).item() == pytest.approx(1.0, abs=1.0e-6)
    assert torch.count_nonzero(algorithm.storage.advantages[~mask]).item() == 0
    assert algorithm.actor_authority_fraction == pytest.approx(0.5)
    assert algorithm.storage.compute_arguments[-1] is False
    assert torch.equal(algorithm.storage.returns, returns_before)
    assert torch.equal(algorithm.storage.values, values_before)


def test_authority_mask_handles_a_rollout_with_no_actor_control():
    advantages = torch.randn((2, 3, 1))
    authority = torch.zeros_like(advantages)
    algorithm = _make_algorithm(advantages, authority)

    algorithm.compute_returns(obs={})

    assert torch.count_nonzero(algorithm.storage.advantages).item() == 0
    assert algorithm.actor_authority_fraction == 0.0


def test_authority_mask_handles_one_actor_transition_without_nan():
    advantages = torch.tensor([[[7.0]]])
    authority = torch.ones_like(advantages)
    algorithm = _make_algorithm(advantages, authority)

    algorithm.compute_returns(obs={})

    assert torch.isfinite(algorithm.storage.advantages).all()
    assert algorithm.storage.advantages.item() == 0.0
    assert algorithm.actor_authority_fraction == 1.0


def test_task_local_rsl_rl_classes_register_in_the_runner_namespace():
    import rsl_rl.runners.on_policy_runner as runner_module

    register_openarm_rsl_rl_classes()

    assert runner_module.AuthorityMaskedPPO is AuthorityMaskedPPO
