from pathlib import Path

import numpy as np
import pytest
import zarr

from openarm_dp3.workspace_reference import (
    StableBooleanGate,
    WorkspaceReferenceError,
    load_workspace_references,
    yellow_cube_centroid,
)


def point_cloud(centroid: tuple[float, float, float], *, yellow_count: int = 16) -> np.ndarray:
    points = np.zeros((1024, 6), dtype=np.float32)
    points[:, :3] = (0.0, 0.0, 0.4)
    points[:, 3:] = (0.5, 0.5, 0.5)
    offsets = np.linspace(-0.002, 0.002, yellow_count, dtype=np.float32)
    points[:yellow_count, :3] = np.asarray(centroid, dtype=np.float32)
    points[:yellow_count, 0] += offsets
    points[:yellow_count, 3:] = (0.8, 0.65, 0.05)
    return points


def make_visual_only_dataset(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    points = np.stack(
        [
            point_cloud((0.00, 0.05, 0.36)),
            point_cloud((0.00, 0.05, 0.36)),
            point_cloud((-0.20, 0.05, 0.36)),
            point_cloud((-0.20, 0.05, 0.36)),
            point_cloud((0.03, 0.06, 0.36)),
            point_cloud((0.03, 0.06, 0.36)),
            point_cloud((-0.21, 0.05, 0.36)),
            point_cloud((-0.21, 0.05, 0.36)),
        ]
    )
    data.create_dataset("point_cloud", data=points)
    meta.create_dataset("episode_ends", data=np.asarray([4, 8], dtype=np.int64))


def test_yellow_cube_centroid_uses_only_yellow_points() -> None:
    centroid, count = yellow_cube_centroid(point_cloud((0.02, 0.05, 0.36)))
    assert count == 16
    np.testing.assert_allclose(centroid, (0.02, 0.05, 0.36), atol=1e-6)


def test_yellow_cube_centroid_rejects_missing_cube() -> None:
    points = point_cloud((0.02, 0.05, 0.36), yellow_count=4)
    with pytest.raises(WorkspaceReferenceError, match="only 4 points"):
        yellow_cube_centroid(points, minimum_points=8)


def test_visual_references_do_not_require_action_trajectory(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.zarr"
    make_visual_only_dataset(dataset)
    references = load_workspace_references(
        dataset,
        start_centroid_frames=2,
        goal_centroid_frames=2,
    )
    assert references.start_centroids_m.shape == (2, 3)
    assert references.goal_centroids_m.shape == (2, 3)

    start = references.observe_cube(
        point_cloud((0.02, 0.055, 0.36)),
        minimum_yellow_points=8,
        start_region_margin_m=0.01,
        maximum_goal_distance_m=0.03,
    )
    goal = references.observe_cube(
        point_cloud((-0.205, 0.05, 0.36)),
        minimum_yellow_points=8,
        start_region_margin_m=0.01,
        maximum_goal_distance_m=0.03,
    )
    assert start.in_start_region
    assert not goal.in_start_region
    assert goal.nearest_goal_distance_m == pytest.approx(0.005, abs=1e-6)


def test_stable_gate_requires_consecutive_frames() -> None:
    gate = StableBooleanGate(required_frames=3)
    assert not gate.observe(True)
    assert not gate.observe(True)
    assert not gate.observe(False)
    assert not gate.observe(True)
    assert not gate.observe(True)
    assert gate.observe(True)
    gate.reset()
    assert gate.stable_frames == 0
