from __future__ import annotations

import numpy as np


class PointCloudError(ValueError):
    """Raised when an RGB-D frame cannot provide enough valid geometry."""


def sample_aligned_vertices(
    vertices: np.ndarray,
    *,
    height: int,
    width: int,
    pixel_roi: tuple[int, int, int, int] | None,
    depth_range_m: tuple[float, float],
    point_count: int,
    minimum_valid_points: int,
    seed: int,
    rgb: np.ndarray | None = None,
    include_color: bool = False,
) -> np.ndarray:
    """Select fixed-size XYZ or XYZRGB points from aligned RealSense frames."""
    points = np.asarray(vertices, dtype=np.float32)
    if points.shape != (height * width, 3):
        raise PointCloudError(f"unexpected RealSense vertex shape: {points.shape}")
    grid = points.reshape(height, width, 3)
    color_grid: np.ndarray | None = None
    if include_color:
        if rgb is None or np.asarray(rgb).shape != (height, width, 3):
            raise PointCloudError("aligned RGB is required for colored point clouds")
        color_grid = np.asarray(rgb, dtype=np.float32) / 255.0
    if pixel_roi is not None:
        x0, y0, x1, y1 = pixel_roi
        if min(x0, y0) < 0 or x1 > width or y1 > height or x1 <= x0 or y1 <= y0:
            raise PointCloudError("pixel ROI is outside the aligned depth image")
        grid = grid[y0:y1, x0:x1]
        if color_grid is not None:
            color_grid = color_grid[y0:y1, x0:x1]
    candidates = grid.reshape(-1, 3)
    candidate_colors = color_grid.reshape(-1, 3) if color_grid is not None else None
    near, far = depth_range_m
    valid = np.isfinite(candidates).all(axis=1)
    valid &= candidates[:, 2] >= near
    valid &= candidates[:, 2] <= far
    candidates = candidates[valid]
    if candidate_colors is not None:
        candidate_colors = candidate_colors[valid]
    if len(candidates) < minimum_valid_points:
        raise PointCloudError(
            f"only {len(candidates)} valid depth points; need {minimum_valid_points}"
        )

    rng = np.random.default_rng(seed)
    if len(candidates) >= point_count:
        indices = rng.choice(len(candidates), size=point_count, replace=False)
    else:
        base = np.arange(len(candidates), dtype=np.int64)
        extra = rng.choice(len(candidates), size=point_count - len(candidates), replace=True)
        indices = np.concatenate((base, extra))
        rng.shuffle(indices)
    result = candidates[indices]
    if candidate_colors is not None:
        result = np.concatenate((result, candidate_colors[indices]), axis=1)
    result = np.ascontiguousarray(result, dtype=np.float32)
    expected_channels = 6 if include_color else 3
    if result.shape != (point_count, expected_channels) or not np.isfinite(result).all():
        raise PointCloudError("sampled point cloud is malformed")
    return result


def voxel_sample_fixed(
    points: np.ndarray,
    *,
    point_count: int,
    voxel_size_m: float,
    seed: int,
) -> np.ndarray:
    """Voxel-deduplicate and deterministically sample a fixed number of XYZ points."""
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 3:
        raise PointCloudError(f"expected point cloud shaped (N, >=3), got {values.shape}")
    channels = values.shape[1]
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) == 0 or point_count <= 0 or voxel_size_m <= 0:
        raise PointCloudError("point cloud, point_count, and voxel_size_m must be valid")

    voxel_ids = np.floor(values[:, :3] / voxel_size_m).astype(np.int32)
    _, unique_indices = np.unique(voxel_ids, axis=0, return_index=True)
    reduced = values[np.sort(unique_indices)]
    rng = np.random.default_rng(seed)
    if len(reduced) >= point_count:
        chosen = rng.choice(len(reduced), size=point_count, replace=False)
    else:
        extra = rng.choice(len(reduced), size=point_count - len(reduced), replace=True)
        chosen = np.concatenate((np.arange(len(reduced)), extra))
        rng.shuffle(chosen)
    result = np.ascontiguousarray(reduced[chosen], dtype=np.float32)
    if result.shape != (point_count, channels) or not np.isfinite(result).all():
        raise PointCloudError("voxel sampling produced invalid output")
    return result
