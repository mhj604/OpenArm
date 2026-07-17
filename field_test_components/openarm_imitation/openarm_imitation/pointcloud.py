from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    ppx: float
    ppy: float
    width: int
    height: int


def resize_rgb_nearest(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    if rgb.shape[:2] == (height, width):
        return np.ascontiguousarray(rgb)
    ys = np.linspace(0, rgb.shape[0] - 1, height).round().astype(np.intp)
    xs = np.linspace(0, rgb.shape[1] - 1, width).round().astype(np.intp)
    return np.ascontiguousarray(rgb[ys[:, None], xs[None, :]])


def depth_to_pointcloud(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    intrinsics: Intrinsics,
    *,
    points: int,
    bounds: dict[str, list[float]],
    stride: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Back-project aligned depth into an Nx6 XYZRGB point cloud."""
    if depth_m.shape != rgb.shape[:2]:
        raise ValueError(f"aligned depth/rgb shape mismatch: {depth_m.shape} vs {rgb.shape[:2]}")
    stride = max(1, int(stride))
    v, u = np.mgrid[0:depth_m.shape[0]:stride, 0:depth_m.shape[1]:stride]
    z = depth_m[::stride, ::stride].astype(np.float32, copy=False)
    x = (u.astype(np.float32) - intrinsics.ppx) * z / intrinsics.fx
    y = (v.astype(np.float32) - intrinsics.ppy) * z / intrinsics.fy

    bx, by, bz = bounds["x"], bounds["y"], bounds["z"]
    valid = np.isfinite(z) & (z > 0)
    valid &= (x >= bx[0]) & (x <= bx[1])
    valid &= (y >= by[0]) & (y <= by[1])
    valid &= (z >= bz[0]) & (z <= bz[1])

    xyz = np.stack((x[valid], y[valid], z[valid]), axis=-1)
    colors = rgb[::stride, ::stride][valid].astype(np.float32) / 255.0
    if xyz.shape[0] == 0:
        raise RuntimeError("point-cloud crop removed every valid depth pixel")
    replace = xyz.shape[0] < points
    indices = rng.choice(xyz.shape[0], size=points, replace=replace)
    return np.ascontiguousarray(np.concatenate((xyz[indices], colors[indices]), axis=-1), dtype=np.float32)

