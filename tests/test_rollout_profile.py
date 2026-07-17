from pathlib import Path

import numpy as np
import zarr

from openarm_dp3.rollout_profile import RolloutProfile, create_profile, write_profile


def make_dataset(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    state = np.zeros((5, 16), dtype=np.float32)
    state[0] = 1.0
    state[3] = 2.0
    action = np.arange(80, dtype=np.float32).reshape(5, 16) / 10.0
    data.create_dataset("state", data=state)
    data.create_dataset("action", data=action)
    meta.create_dataset("episode_ends", data=np.asarray([3, 5], dtype=np.int64))
    root.attrs.update(
        {
            "state_action_unit": "radian",
            "task_id": "test",
            "sample_fps": 15,
        }
    )


def test_profile_uses_all_actions_and_episode_start_states(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.zarr"
    make_dataset(dataset)
    profile = create_profile(dataset)
    assert profile["episode_count"] == 2
    assert profile["frame_count"] == 5
    assert profile["action"]["min"][0] == 0.0
    assert np.isclose(profile["action"]["max"][15], 7.9)
    assert profile["start_state"]["min"] == [1.0] * 16
    assert profile["start_state"]["max"] == [2.0] * 16
    assert profile["start_state"]["mean"] == [1.5] * 16


def test_start_gate_uses_separate_arm_and_gripper_margins(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.zarr"
    output = tmp_path / "profile.json"
    make_dataset(dataset)
    write_profile(dataset, output)
    profile = RolloutProfile.load(output)

    live = np.full(16, 1.5, dtype=np.float32)
    live[0] = 2.15
    live[7] = 2.15
    violations = profile.start_violations(live, arm_margin=0.10, gripper_margin=0.20)
    assert len(violations) == 1
    assert violations[0].startswith("index 0:")
