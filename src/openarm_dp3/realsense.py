from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyrealsense2 as rs

from .config import CameraSettings


class RealSenseError(RuntimeError):
    """Raised when a configured camera cannot provide synchronized RGB-D."""


@dataclass(frozen=True)
class RGBDFrame:
    rgb: np.ndarray
    depth_raw: np.ndarray
    vertices_xyz: np.ndarray
    host_time_ns: int
    host_monotonic_ns: int
    device_timestamp_ms: float
    depth_scale_m: float
    intrinsics: dict[str, Any]


def _intrinsics_dict(intrinsics: Any) -> dict[str, Any]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "model": str(intrinsics.model),
        "coeffs": [float(value) for value in intrinsics.coeffs],
    }


def _vertices_to_numpy(points: Any, expected_count: int) -> np.ndarray:
    raw = np.asanyarray(points.get_vertices())
    try:
        vertices = raw.view(np.float32).reshape(-1, 3)
    except (TypeError, ValueError):
        vertices = np.asarray([[value[0], value[1], value[2]] for value in raw], dtype=np.float32)
    if vertices.shape != (expected_count, 3):
        raise RealSenseError(f"unexpected point-cloud vertex shape: {vertices.shape}")
    return np.ascontiguousarray(vertices, dtype=np.float32)


class RealSenseRGBD:
    def __init__(self, settings: CameraSettings):
        self.settings = settings
        self.pipeline: Any | None = None
        self.align: Any | None = None
        self.point_cloud: Any | None = None
        self.depth_scale_m: float | None = None

    @property
    def is_started(self) -> bool:
        return self.pipeline is not None

    def start(self, *, warmup_frames: int = 20) -> None:
        if self.pipeline is not None:
            raise RealSenseError(f"camera {self.settings.key} is already started")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.settings.serial)
        config.enable_stream(
            rs.stream.depth,
            self.settings.width,
            self.settings.height,
            rs.format.z16,
            self.settings.fps,
        )
        config.enable_stream(
            rs.stream.color,
            self.settings.width,
            self.settings.height,
            rs.format.rgb8,
            self.settings.fps,
        )
        try:
            profile = pipeline.start(config)
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale_m = float(depth_sensor.get_depth_scale())
            self.pipeline = pipeline
            align_stream = rs.stream.depth if self.settings.align_to == "depth" else rs.stream.color
            self.align = rs.align(align_stream)
            self.point_cloud = rs.pointcloud()
            for _ in range(max(1, warmup_frames)):
                self._wait_for_aligned_frames()
        except Exception:
            try:
                pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
            self.align = None
            self.point_cloud = None
            self.depth_scale_m = None
            raise

    def _wait_for_aligned_frames(self) -> Any:
        if self.pipeline is None or self.align is None:
            raise RealSenseError(f"camera {self.settings.key} is not started")
        frames = self.pipeline.wait_for_frames(timeout_ms=3000)
        aligned = self.align.process(frames)
        if not aligned.get_depth_frame() or not aligned.get_color_frame():
            raise RealSenseError(f"camera {self.settings.key} returned incomplete RGB-D")
        return aligned

    def read(self) -> RGBDFrame:
        if self.point_cloud is None or self.depth_scale_m is None:
            raise RealSenseError(f"camera {self.settings.key} is not started")
        aligned = self._wait_for_aligned_frames()
        host_time_ns = time.time_ns()
        host_monotonic_ns = time.monotonic_ns()
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        rgb = np.ascontiguousarray(np.asanyarray(color_frame.get_data()), dtype=np.uint8)
        depth = np.ascontiguousarray(np.asanyarray(depth_frame.get_data()), dtype=np.uint16)
        if rgb.shape != (self.settings.height, self.settings.width, 3):
            raise RealSenseError(f"unexpected {self.settings.key} RGB shape: {rgb.shape}")
        if depth.shape != (self.settings.height, self.settings.width):
            raise RealSenseError(f"unexpected {self.settings.key} depth shape: {depth.shape}")

        points = self.point_cloud.calculate(depth_frame)
        vertices = _vertices_to_numpy(points, self.settings.width * self.settings.height)
        profile = depth_frame.profile.as_video_stream_profile()
        return RGBDFrame(
            rgb=rgb,
            depth_raw=depth,
            vertices_xyz=vertices,
            host_time_ns=host_time_ns,
            host_monotonic_ns=host_monotonic_ns,
            device_timestamp_ms=float(depth_frame.get_timestamp()),
            depth_scale_m=self.depth_scale_m,
            intrinsics=_intrinsics_dict(profile.get_intrinsics()),
        )

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
        self.pipeline = None
        self.align = None
        self.point_cloud = None
        self.depth_scale_m = None

    def __enter__(self) -> RealSenseRGBD:
        self.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.stop()
