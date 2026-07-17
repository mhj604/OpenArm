from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr


class WorkspaceReferenceError(ValueError):
    """Raised when the learned task's cube regions cannot be identified."""


@dataclass(frozen=True)
class CubeObservation:
    centroid_m: np.ndarray
    yellow_point_count: int
    in_start_region: bool
    nearest_goal_distance_m: float


@dataclass(frozen=True)
class WorkspaceReferences:
    start_centroids_m: np.ndarray
    goal_centroids_m: np.ndarray

    def observe_cube(
        self,
        points: np.ndarray,
        *,
        minimum_yellow_points: int,
        start_region_margin_m: float,
        maximum_goal_distance_m: float,
    ) -> CubeObservation:
        if start_region_margin_m < 0 or maximum_goal_distance_m <= 0:
            raise WorkspaceReferenceError("cube-region distance limits are invalid")
        centroid, count = yellow_cube_centroid(
            points, minimum_points=minimum_yellow_points
        )
        start_lower = self.start_centroids_m.min(axis=0) - start_region_margin_m
        start_upper = self.start_centroids_m.max(axis=0) + start_region_margin_m
        in_start = bool(np.all((centroid >= start_lower) & (centroid <= start_upper)))
        goal_distance = min(
            float(np.linalg.norm(centroid - goal)) for goal in self.goal_centroids_m
        )
        return CubeObservation(
            centroid_m=centroid,
            yellow_point_count=count,
            in_start_region=in_start,
            nearest_goal_distance_m=goal_distance,
        )


@dataclass
class StableBooleanGate:
    required_frames: int
    stable_frames: int = 0

    def __post_init__(self) -> None:
        if self.required_frames <= 0:
            raise WorkspaceReferenceError("stable frame count must be positive")

    def reset(self) -> None:
        self.stable_frames = 0

    def observe(self, condition: bool) -> bool:
        self.stable_frames = self.stable_frames + 1 if condition else 0
        return self.stable_frames >= self.required_frames


def _as_points(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
        raise WorkspaceReferenceError("point cloud must be finite and shaped (N, 6)")
    return values


def yellow_cube_points(points: np.ndarray) -> np.ndarray:
    """Return XYZ points matching the yellow cube used for this trained task."""
    values = _as_points(points)
    rgb = values[:, 3:6]
    mask = (rgb[:, 0] > 0.45) & (rgb[:, 1] > 0.32) & (rgb[:, 2] < 0.30)
    mask &= rgb[:, 0] > rgb[:, 2] * 1.7
    mask &= rgb[:, 1] > rgb[:, 2] * 1.5
    return np.ascontiguousarray(values[mask, :3], dtype=np.float32)


def yellow_cube_centroid(points: np.ndarray, *, minimum_points: int = 8) -> tuple[np.ndarray, int]:
    if minimum_points <= 0:
        raise WorkspaceReferenceError("minimum yellow point count must be positive")
    yellow = yellow_cube_points(points)
    if len(yellow) < minimum_points:
        raise WorkspaceReferenceError(
            f"yellow cube has only {len(yellow)} points; need at least {minimum_points}"
        )
    centroid = np.median(yellow, axis=0).astype(np.float32, copy=False)
    return centroid, len(yellow)


def _yellow_from_frame_range(points: object, start: int, end: int) -> np.ndarray:
    clouds = [
        yellow_cube_points(np.asarray(points[index], dtype=np.float32))
        for index in range(start, end)
    ]
    return np.ascontiguousarray(np.concatenate(clouds, axis=0), dtype=np.float32)


def load_workspace_references(
    dataset_path: Path,
    *,
    start_centroid_frames: int = 5,
    goal_centroid_frames: int = 15,
    minimum_yellow_points: int = 8,
) -> WorkspaceReferences:
    """Load only visual start/goal references; no action trajectory is opened."""
    if start_centroid_frames <= 0 or goal_centroid_frames <= 0:
        raise WorkspaceReferenceError("centroid frame counts must be positive")
    if not dataset_path.is_dir():
        raise WorkspaceReferenceError(f"processed dataset does not exist: {dataset_path}")
    root = zarr.open_group(str(dataset_path), mode="r")
    try:
        points = root["data/point_cloud"]
        episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    except KeyError as exc:
        raise WorkspaceReferenceError(f"processed dataset is incomplete: {exc}") from exc
    if len(points.shape) != 3 or points.shape[2] != 6:
        raise WorkspaceReferenceError("point_cloud must be shaped (T, N, 6)")
    if (
        episode_ends.ndim != 1
        or len(episode_ends) == 0
        or int(episode_ends[-1]) != points.shape[0]
        or np.any(np.diff(episode_ends) <= 0)
    ):
        raise WorkspaceReferenceError("episode_ends is malformed")

    starts = np.concatenate((np.asarray([0], dtype=np.int64), episode_ends[:-1]))
    start_centroids: list[np.ndarray] = []
    goal_centroids: list[np.ndarray] = []
    for ordinal, (start, end) in enumerate(zip(starts, episode_ends, strict=True)):
        start_i = int(start)
        end_i = int(end)
        start_yellow = _yellow_from_frame_range(
            points,
            start_i,
            min(end_i, start_i + start_centroid_frames),
        )
        if len(start_yellow) < minimum_yellow_points:
            raise WorkspaceReferenceError(
                f"episode {ordinal} has only {len(start_yellow)} yellow start points"
            )
        start_centroids.append(np.median(start_yellow, axis=0).astype(np.float32))

        goal_yellow = _yellow_from_frame_range(
            points,
            max(start_i, end_i - goal_centroid_frames),
            end_i,
        )
        if len(goal_yellow) >= minimum_yellow_points:
            goal_centroids.append(np.median(goal_yellow, axis=0).astype(np.float32))
    if not goal_centroids:
        raise WorkspaceReferenceError("dataset contains no visible learned goal references")
    return WorkspaceReferences(
        start_centroids_m=np.ascontiguousarray(np.stack(start_centroids), dtype=np.float32),
        goal_centroids_m=np.ascontiguousarray(np.stack(goal_centroids), dtype=np.float32),
    )
