from pathlib import Path

import numpy as np
import zarr

from openarm_dp3.config import CollectionSettings, TrimSettings
from openarm_dp3.convert_to_dp3 import convert, trim_bounds
from openarm_dp3.storage import EpisodeFrame, EpisodeWriter


def settings(root: Path) -> CollectionSettings:
    return CollectionSettings(
        task_id="test",
        task_description="test cube transfer",
        dataset_root=root,
        sample_fps=10,
        max_episode_seconds=10.0,
        countdown_seconds=0,
        target_episodes=1,
        preview_width=2,
        preview_height=2,
        chest_raw_points=1024,
        wrist_left_raw_points=32,
        training_points=1024,
        voxel_size_m=0.005,
        minimum_valid_depth_points=10,
        maximum_camera_skew_ms=80.0,
        trim=TrimSettings(
            enabled=True,
            motion_threshold_radian=0.02,
            step_threshold_radian=0.004,
            pre_roll_frames=2,
            post_roll_frames=3,
        ),
    )


def test_trim_removes_long_static_prefix_and_suffix(tmp_path: Path) -> None:
    cfg = settings(tmp_path)
    action = np.zeros((50, 16), dtype=np.float32)
    action[12:35, 0] = np.linspace(0.0, 0.5, 23)
    action[35:, 0] = 0.5
    start, end = trim_bounds(action, cfg)
    assert 9 <= start <= 12
    assert 35 <= end <= 40


def test_raw_episode_converts_to_dp3_replay_buffer(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    cfg = settings(raw_root)
    writer = EpisodeWriter(
        root=raw_root,
        episode_index=0,
        metadata={"task": "test"},
        state_dim=16,
        chest_points=1024,
        wrist_left_points=32,
        preview_height=2,
        preview_width=2,
    )
    rng = np.random.default_rng(8)
    for index in range(40):
        action = np.zeros(16, dtype=np.float32)
        action[0] = np.clip((index - 5) / 20, 0.0, 1.0)
        writer.append(
            EpisodeFrame(
                state=action.copy(),
                action=action,
                chest_point_cloud_raw=rng.normal(size=(1024, 3)).astype(np.float32),
                wrist_left_point_cloud_raw=rng.normal(size=(32, 3)).astype(np.float32),
                chest_rgb_preview=np.zeros((2, 2, 3), dtype=np.uint8),
                wrist_left_rgb_preview=np.zeros((2, 2, 3), dtype=np.uint8),
                chest_depth_preview=np.ones((2, 2), dtype=np.uint16),
                wrist_left_depth_preview=np.ones((2, 2), dtype=np.uint16),
                host_time_ns=index,
                telemetry_sequence=index,
                telemetry_max_source_age_s=0.01,
                chest_device_timestamp_ms=float(index),
                wrist_left_device_timestamp_ms=float(index),
                camera_host_skew_ms=1.0,
            )
        )
    writer.finish(accepted=True, reason="test")

    output = tmp_path / "processed.zarr"
    manifest = convert(cfg, output, overwrite=False)
    group = zarr.open_group(str(output), mode="r")
    assert manifest["episode_count"] == 1
    assert group["data/state"].shape[1] == 16
    assert group["data/action"].shape == group["data/state"].shape
    assert group["data/point_cloud"].shape[1:] == (1024, 3)
    assert group["meta/episode_ends"][-1] == group["data/state"].shape[0]

    from openarm_dp3.dp3_dataset import OpenArmDP3Dataset

    dataset = OpenArmDP3Dataset(str(output), horizon=16, val_ratio=0.0)
    sample = dataset[0]
    assert tuple(sample["obs"]["point_cloud"].shape) == (16, 1024, 3)
    assert tuple(sample["obs"]["agent_pos"].shape) == (16, 16)
    assert tuple(sample["action"].shape) == (16, 16)
