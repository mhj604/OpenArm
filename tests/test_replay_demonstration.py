from pathlib import Path

import numpy as np
import pytest
import zarr

from openarm_dp3.replay_demonstration import (
    ReplayError,
    load_replay_episode,
    replay_start_target,
    required_replay_deltas,
    slew_vector,
    validate_episode_bounds,
)
from openarm_dp3.rollout_profile import RolloutProfile


def make_episode(root: Path, *, accepted: bool = True) -> Path:
    path = root / "episode_000001.zarr"
    group = zarr.open_group(str(path), mode="w")
    data = group.create_group("data")
    state = np.zeros((3, 16), dtype=np.float32)
    action = np.stack(
        [np.zeros(16), np.full(16, 0.05), np.full(16, 0.10)]
    ).astype(np.float32)
    data.create_dataset("state", data=state)
    data.create_dataset("action", data=action)
    group.attrs.update(
        {
            "complete": True,
            "accepted": accepted,
            "episode_index": 1,
            "sample_fps": 15,
            "state_unit": "radian",
            "action_unit": "radian",
            "action_semantics": "outgoing OpenArm CAN target position",
        }
    )
    return path


def test_loads_only_valid_accepted_position_target_episode(tmp_path: Path) -> None:
    make_episode(tmp_path)
    episode = load_replay_episode(tmp_path, 1)
    assert episode.episode_index == 1
    assert episode.action.shape == (3, 16)
    assert episode.duration_s == pytest.approx(0.2)


def test_rejects_operator_rejected_episode(tmp_path: Path) -> None:
    make_episode(tmp_path, accepted=False)
    with pytest.raises(ReplayError, match="operator-accepted"):
        load_replay_episode(tmp_path, 1)


def test_replay_bounds_must_match_motor_profile(tmp_path: Path) -> None:
    make_episode(tmp_path)
    episode = load_replay_episode(tmp_path, 1)
    profile = RolloutProfile(
        action_min=np.full(16, -0.2),
        action_max=np.full(16, 0.2),
        start_min=np.full(16, -0.2),
        start_max=np.full(16, 0.2),
        start_mean=np.zeros(16),
    )
    validate_episode_bounds(episode, profile)
    profile.action_max[0] = 0.01
    with pytest.raises(ReplayError, match="outside the active motor profile"):
        validate_episode_bounds(episode, profile)


def test_start_alignment_slew_is_bounded() -> None:
    current = np.zeros(16, dtype=np.float32)
    target = np.full(16, 0.2, dtype=np.float32)
    result = slew_vector(current, target, 0.03)
    np.testing.assert_allclose(result, np.full(16, 0.03, dtype=np.float32))


def test_measured_start_is_clipped_to_achievable_action_bounds(tmp_path: Path) -> None:
    make_episode(tmp_path)
    episode = load_replay_episode(tmp_path, 1)
    episode.state[0, 15] = -0.02
    profile = RolloutProfile(
        action_min=np.full(16, -0.2),
        action_max=np.full(16, 0.2),
        start_min=np.full(16, -0.2),
        start_max=np.full(16, 0.2),
        start_mean=np.zeros(16),
    )
    profile.action_min[15] = -0.001
    profile.action_max[15] = -0.001
    target = replay_start_target(episode, profile)
    assert target[15] == pytest.approx(-0.001)


def test_required_replay_deltas_preserve_raw_arm_and_gripper_steps(tmp_path: Path) -> None:
    make_episode(tmp_path)
    episode = load_replay_episode(tmp_path, 1)
    episode.state[0, 2] = 0.20
    episode.action[1, 4] = 0.31
    episode.action[2, 7] = -0.64
    profile = RolloutProfile(
        action_min=np.full(16, -1.0),
        action_max=np.full(16, 1.0),
        start_min=np.full(16, -1.0),
        start_max=np.full(16, 1.0),
        start_mean=np.zeros(16),
    )

    arm_delta, gripper_delta = required_replay_deltas(episode, profile)

    assert arm_delta == pytest.approx(0.3101)
    assert gripper_delta == pytest.approx(0.6901)
