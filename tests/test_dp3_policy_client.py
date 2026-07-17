import numpy as np
import torch

from openarm_dp3.dp3_policy_client import (
    PolicyBundle,
    apply_initial_gripper_open_guard,
    normalize_active_confirmation,
    predict_action_chunk,
)
from openarm_dp3.rollout_profile import RolloutProfile


class RandomPolicy:
    def predict_action(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = observation["agent_pos"].shape[0]
        action_pred = torch.randn((batch, 6, 16))
        return {"action": action_pred[:, 1:4], "action_pred": action_pred}


def test_active_confirmation_removes_terminal_formatting_artifacts() -> None:
    assert normalize_active_confirmation("\x1b[200~START\x1b[201~") == "START"
    assert normalize_active_confirmation("\u200bSTART\ufeff") == "START"
    assert normalize_active_confirmation("ＳＴＡＲＴ") == "START"


def test_active_confirmation_does_not_accept_extra_visible_input() -> None:
    assert normalize_active_confirmation("start") != "START"
    assert normalize_active_confirmation("NOTSTART") != "START"
    assert normalize_active_confirmation("ㄴSTART") != "START"


def test_fixed_inference_seed_repeats_learned_diffusion_sample() -> None:
    bundle = PolicyBundle(
        policy=RandomPolicy(),
        observation_steps=2,
        action_steps=3,
        prediction_steps=6,
        point_count=4,
        point_channels=6,
    )
    points = [np.zeros((4, 6), dtype=np.float32)] * 2
    states = [np.zeros(16, dtype=np.float32)] * 2
    first = predict_action_chunk(
        bundle,
        points,
        states,
        device="cpu",
        inference_seed=42,
    )
    second = predict_action_chunk(
        bundle,
        points,
        states,
        device="cpu",
        inference_seed=42,
    )
    different = predict_action_chunk(
        bundle,
        points,
        states,
        device="cpu",
        inference_seed=43,
    )
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_learned_diffusion_median_preserves_action_shape() -> None:
    bundle = PolicyBundle(
        policy=RandomPolicy(),
        observation_steps=2,
        action_steps=3,
        prediction_steps=6,
        point_count=4,
        point_channels=6,
    )
    points = [np.zeros((4, 6), dtype=np.float32)] * 2
    states = [np.zeros(16, dtype=np.float32)] * 2
    result = predict_action_chunk(
        bundle,
        points,
        states,
        device="cpu",
        inference_seed=42,
        sample_count=5,
    )
    repeated = predict_action_chunk(
        bundle,
        points,
        states,
        device="cpu",
        inference_seed=42,
        sample_count=5,
    )
    assert result.shape == (3, 16)
    np.testing.assert_array_equal(result, repeated)


def test_full_prediction_horizon_keeps_all_future_learned_actions() -> None:
    bundle = PolicyBundle(
        policy=RandomPolicy(),
        observation_steps=2,
        action_steps=3,
        prediction_steps=6,
        point_count=4,
        point_channels=6,
    )
    points = [np.zeros((4, 6), dtype=np.float32)] * 2
    states = [np.zeros(16, dtype=np.float32)] * 2
    result = predict_action_chunk(
        bundle,
        points,
        states,
        device="cpu",
        inference_seed=42,
        full_prediction_horizon=True,
    )
    assert result.shape == (5, 16)


def test_initial_gripper_guard_keeps_approach_open_then_releases_policy_control() -> None:
    lower = np.full(16, -1.0)
    upper = np.full(16, 1.0)
    lower[7] = -0.65
    upper[7] = 0.0
    profile = RolloutProfile(
        action_min=lower,
        action_max=upper,
        start_min=lower,
        start_max=upper,
        start_mean=np.zeros(16),
    )
    learned = np.zeros(16, dtype=np.float32)
    learned[7] = -0.65
    guarded = apply_initial_gripper_open_guard(
        learned,
        profile,
        elapsed_s=2.49,
        minimum_open_s=2.5,
    )
    released = apply_initial_gripper_open_guard(
        learned,
        profile,
        elapsed_s=2.5,
        minimum_open_s=2.5,
    )
    assert guarded[7] == 0.0
    assert released[7] == np.float32(-0.65)
