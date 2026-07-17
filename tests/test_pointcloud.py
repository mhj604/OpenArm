import numpy as np
import pytest

from openarm_dp3.pointcloud import PointCloudError, sample_aligned_vertices, voxel_sample_fixed


def test_roi_sampling_is_fixed_shape_and_finite() -> None:
    height, width = 6, 8
    xs, ys = np.meshgrid(np.linspace(-0.2, 0.2, width), np.linspace(-0.1, 0.1, height))
    points = np.stack((xs, ys, np.ones_like(xs)), axis=-1).astype(np.float32).reshape(-1, 3)
    result = sample_aligned_vertices(
        points,
        height=height,
        width=width,
        pixel_roi=(1, 1, 7, 5),
        depth_range_m=(0.5, 1.5),
        point_count=16,
        minimum_valid_points=10,
        seed=7,
    )
    assert result.shape == (16, 3)
    assert np.isfinite(result).all()
    assert np.all((result[:, 2] >= 0.5) & (result[:, 2] <= 1.5))


def test_nan_and_missing_depth_are_rejected() -> None:
    points = np.full((16, 3), np.nan, dtype=np.float32)
    with pytest.raises(PointCloudError, match="valid depth points"):
        sample_aligned_vertices(
            points,
            height=4,
            width=4,
            pixel_roi=None,
            depth_range_m=(0.1, 2.0),
            point_count=8,
            minimum_valid_points=4,
            seed=0,
        )


def test_voxel_sampling_is_deterministic() -> None:
    rng = np.random.default_rng(2)
    points = rng.normal(size=(5000, 3)).astype(np.float32)
    first = voxel_sample_fixed(points, point_count=1024, voxel_size_m=0.01, seed=123)
    second = voxel_sample_fixed(points, point_count=1024, voxel_size_m=0.01, seed=123)
    assert first.shape == (1024, 3)
    np.testing.assert_array_equal(first, second)
