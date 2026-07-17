from __future__ import annotations

import fcntl
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc

EPISODE_PATTERN = re.compile(r"^episode_(\d{6})\.zarr$")


class StorageError(RuntimeError):
    """Raised when an episode cannot be committed atomically."""


class DatasetLock:
    def __init__(self, root: Path):
        self.root = root
        self.handle: Any | None = None

    def __enter__(self) -> DatasetLock:
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".collector.lock"
        self.handle = lock_path.open("w", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise StorageError(f"another collector is using {self.root}") from exc
        self.handle.write(f"pid={os.getpid()} started_ns={time.time_ns()}\n")
        self.handle.flush()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def next_episode_index(root: Path) -> int:
    maximum = -1
    if root.exists():
        for path in root.iterdir():
            match = EPISODE_PATTERN.match(path.name)
            if match:
                maximum = max(maximum, int(match.group(1)))
    return maximum + 1


@dataclass(frozen=True)
class EpisodeFrame:
    state: np.ndarray
    action: np.ndarray
    chest_point_cloud_raw: np.ndarray
    wrist_left_point_cloud_raw: np.ndarray
    chest_rgb_preview: np.ndarray
    wrist_left_rgb_preview: np.ndarray
    chest_depth_preview: np.ndarray
    wrist_left_depth_preview: np.ndarray
    host_time_ns: int
    telemetry_sequence: int
    telemetry_max_source_age_s: float
    chest_device_timestamp_ms: float
    wrist_left_device_timestamp_ms: float
    camera_host_skew_ms: float


class EpisodeWriter:
    def __init__(
        self,
        *,
        root: Path,
        episode_index: int,
        metadata: dict[str, Any],
        state_dim: int,
        chest_points: int,
        wrist_left_points: int,
        preview_height: int,
        preview_width: int,
        point_channels: int = 3,
    ):
        root.mkdir(parents=True, exist_ok=True)
        self.root_path = root
        self.episode_index = episode_index
        self.final_path = root / f"episode_{episode_index:06d}.zarr"
        self.temporary_path = root / (
            f".episode_{episode_index:06d}.{uuid.uuid4().hex}.incomplete.zarr"
        )
        if self.final_path.exists():
            raise StorageError(f"episode already exists: {self.final_path}")
        self.group = zarr.open_group(str(self.temporary_path), mode="w")
        self.group.attrs.update(metadata)
        self.group.attrs.update(
            {
                "schema_version": 1,
                "episode_index": episode_index,
                "accepted": False,
                "complete": False,
                "started_time_ns": time.time_ns(),
            }
        )
        data = self.group.create_group("data")
        fast = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        image = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
        self.arrays = {
            "state": data.create_dataset(
                "state", shape=(0, state_dim), chunks=(64, state_dim), dtype="f4", compressor=fast
            ),
            "action": data.create_dataset(
                "action", shape=(0, state_dim), chunks=(64, state_dim), dtype="f4", compressor=fast
            ),
            "chest_point_cloud_raw": data.create_dataset(
                "chest_point_cloud_raw",
                shape=(0, chest_points, point_channels),
                chunks=(1, chest_points, point_channels),
                dtype="f4",
                compressor=fast,
            ),
            "wrist_left_point_cloud_raw": data.create_dataset(
                "wrist_left_point_cloud_raw",
                shape=(0, wrist_left_points, point_channels),
                chunks=(1, wrist_left_points, point_channels),
                dtype="f4",
                compressor=fast,
            ),
            "chest_rgb_preview": data.create_dataset(
                "chest_rgb_preview",
                shape=(0, preview_height, preview_width, 3),
                chunks=(1, preview_height, preview_width, 3),
                dtype="u1",
                compressor=image,
            ),
            "wrist_left_rgb_preview": data.create_dataset(
                "wrist_left_rgb_preview",
                shape=(0, preview_height, preview_width, 3),
                chunks=(1, preview_height, preview_width, 3),
                dtype="u1",
                compressor=image,
            ),
            "chest_depth_preview": data.create_dataset(
                "chest_depth_preview",
                shape=(0, preview_height, preview_width),
                chunks=(1, preview_height, preview_width),
                dtype="u2",
                compressor=image,
            ),
            "wrist_left_depth_preview": data.create_dataset(
                "wrist_left_depth_preview",
                shape=(0, preview_height, preview_width),
                chunks=(1, preview_height, preview_width),
                dtype="u2",
                compressor=image,
            ),
        }
        scalars = {
            "host_time_ns": "i8",
            "telemetry_sequence": "i8",
            "telemetry_max_source_age_s": "f4",
            "chest_device_timestamp_ms": "f8",
            "wrist_left_device_timestamp_ms": "f8",
            "camera_host_skew_ms": "f4",
        }
        for name, dtype in scalars.items():
            self.arrays[name] = data.create_dataset(
                name,
                shape=(0,),
                chunks=(256,),
                dtype=dtype,
                compressor=fast,
            )
        self.frame_count = 0
        self.finished = False

    @staticmethod
    def _append(array: Any, value: Any) -> None:
        new_shape = (array.shape[0] + 1, *array.shape[1:])
        array.resize(new_shape)
        array[-1] = value

    def append(self, frame: EpisodeFrame) -> None:
        if self.finished:
            raise StorageError("cannot append to a finished episode")
        values = {
            "state": frame.state,
            "action": frame.action,
            "chest_point_cloud_raw": frame.chest_point_cloud_raw,
            "wrist_left_point_cloud_raw": frame.wrist_left_point_cloud_raw,
            "chest_rgb_preview": frame.chest_rgb_preview,
            "wrist_left_rgb_preview": frame.wrist_left_rgb_preview,
            "chest_depth_preview": frame.chest_depth_preview,
            "wrist_left_depth_preview": frame.wrist_left_depth_preview,
            "host_time_ns": frame.host_time_ns,
            "telemetry_sequence": frame.telemetry_sequence,
            "telemetry_max_source_age_s": frame.telemetry_max_source_age_s,
            "chest_device_timestamp_ms": frame.chest_device_timestamp_ms,
            "wrist_left_device_timestamp_ms": frame.wrist_left_device_timestamp_ms,
            "camera_host_skew_ms": frame.camera_host_skew_ms,
        }
        for name, value in values.items():
            self._append(self.arrays[name], value)
        self.frame_count += 1

    def finish(self, *, accepted: bool, reason: str) -> Path:
        if self.finished:
            raise StorageError("episode is already finished")
        self.group.attrs.update(
            {
                "accepted": bool(accepted),
                "complete": True,
                "decision_reason": reason,
                "frame_count": self.frame_count,
                "finished_time_ns": time.time_ns(),
            }
        )
        self.group.store.flush() if hasattr(self.group.store, "flush") else None
        os.replace(self.temporary_path, self.final_path)
        self.finished = True
        return self.final_path

    def preserve_incomplete(self, *, reason: str) -> Path:
        if self.finished:
            return self.final_path
        self.group.attrs.update(
            {
                "accepted": False,
                "complete": False,
                "decision_reason": reason,
                "frame_count": self.frame_count,
                "finished_time_ns": time.time_ns(),
            }
        )
        self.finished = True
        return self.temporary_path
