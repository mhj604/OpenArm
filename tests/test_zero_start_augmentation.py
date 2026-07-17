from pathlib import Path

import numpy as np
import zarr

from openarm_dp3.augment_zero_start import augment_dataset, build_zero_start_prefix


def test_prefix_interpolates_arms_and_preserves_grippers_and_points() -> None:
    first_state = np.arange(16, dtype=np.float32) / 10
    first_action = np.arange(16, dtype=np.float32) / 5
    first_points = np.arange(24, dtype=np.float32).reshape(4, 6)
    state, action, points = build_zero_start_prefix(
        first_state,
        first_action,
        first_points,
        prefix_frames=4,
    )
    arm = [*range(7), *range(8, 15)]
    np.testing.assert_allclose(state[0, arm], 0.0)
    np.testing.assert_allclose(state[-1, arm], first_state[arm] * 0.75)
    np.testing.assert_allclose(action[0, arm], first_action[arm] * 0.25)
    np.testing.assert_allclose(action[-1, arm], first_action[arm])
    np.testing.assert_allclose(
        state[:, [7, 15]], np.repeat(first_state[None, [7, 15]], 4, axis=0)
    )
    np.testing.assert_allclose(
        action[:, [7, 15]], np.repeat(first_action[None, [7, 15]], 4, axis=0)
    )
    np.testing.assert_array_equal(points[0], first_points)
    np.testing.assert_array_equal(points[-1], first_points)


def _write_source(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = zarr.open_group(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    state = np.arange(6 * 16, dtype=np.float32).reshape(6, 16) / 100
    action = state + 0.01
    points = np.arange(6 * 1024 * 6, dtype=np.float32).reshape(6, 1024, 6) / 1000
    data.create_dataset("state", data=state)
    data.create_dataset("action", data=action)
    data.create_dataset("point_cloud", data=points)
    meta.create_dataset("episode_ends", data=np.asarray([3, 6], dtype=np.int64))
    root.attrs.update({"schema_version": 1, "sample_fps": 15, "frame_count": 6})
    return state, action, points


def test_augmentation_writes_separate_dataset_without_changing_source(tmp_path: Path) -> None:
    source_path = tmp_path / "source.zarr"
    output_path = tmp_path / "zero_start.zarr"
    source_state, source_action, source_points = _write_source(source_path)
    manifest = augment_dataset(source_path, output_path, prefix_frames=4)

    source = zarr.open_group(str(source_path), mode="r")
    output = zarr.open_group(str(output_path), mode="r")
    np.testing.assert_array_equal(source["data/state"][:], source_state)
    np.testing.assert_array_equal(source["data/action"][:], source_action)
    np.testing.assert_array_equal(source["data/point_cloud"][:], source_points)
    assert output["data/state"].shape == (14, 16)
    np.testing.assert_array_equal(output["meta/episode_ends"][:], [7, 14])
    np.testing.assert_array_equal(output["data/state"][4:7], source_state[:3])
    np.testing.assert_array_equal(output["data/state"][11:14], source_state[3:])
    assert output.attrs["source_dataset_mutated"] is False
    assert output.attrs["zero_start_prefix_frames"] == 4
    assert manifest["source_mutated"] is False
