from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .pointcloud import Intrinsics


def _rs():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is unavailable. Source /opt/ros/humble/setup.bash and activate .venv-collect."
        ) from exc
    return rs


def list_devices() -> list[dict[str, str]]:
    rs = _rs()
    result = []
    for device in rs.context().query_devices():
        def info(field: Any) -> str:
            try:
                return str(device.get_info(field))
            except Exception:
                return ""

        result.append({
            "name": info(rs.camera_info.name),
            "serial": info(rs.camera_info.serial_number),
            "firmware": info(rs.camera_info.firmware_version),
            "usb_type": info(rs.camera_info.usb_type_descriptor),
            "physical_port": info(rs.camera_info.physical_port),
            "product_line": info(rs.camera_info.product_line),
        })
    return result


def resolve_camera_serials(cfg: dict, devices: list[dict[str, str]] | None = None) -> dict[str, str]:
    devices = list_devices() if devices is None else devices
    assigned: dict[str, str] = {}
    used: set[str] = set()
    for role in ("chest", "hand"):
        spec = cfg["cameras"][role]
        explicit = spec.get("serial")
        if explicit:
            match = next((d for d in devices if d["serial"] == str(explicit)), None)
            if match is None:
                raise RuntimeError(f"configured {role} camera serial {explicit} is not connected")
            assigned[role] = str(explicit)
            used.add(str(explicit))
            continue

        tokens = [str(x).upper() for x in spec.get("model_contains", [])]
        match = next(
            (d for d in devices if d["serial"] not in used and any(t in d["name"].upper() for t in tokens)),
            None,
        )
        if match is None:
            raise RuntimeError(f"could not find {role} camera matching {tokens}; connected={devices}")
        assigned[role] = match["serial"]
        used.add(match["serial"])
    return assigned


@dataclass(frozen=True)
class CameraSnapshot:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: Intrinsics
    timestamp: float
    frame_number: int


class RealSenseReader:
    def __init__(self, role: str, serial: str, spec: dict):
        self.role = role
        self.serial = serial
        self.spec = spec
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pipeline = None
        self._align = None
        self._depth_scale = 0.001
        self._latest: CameraSnapshot | None = None
        self.error: Exception | None = None

    def start(self) -> None:
        rs = _rs()
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        w, h, fps = int(self.spec["width"]), int(self.spec["height"]), int(self.spec["fps"])
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
        profile = pipeline.start(config)
        self._pipeline = pipeline
        self._depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        if self.spec.get("align_depth_to_color", False):
            self._align = rs.align(rs.stream.color)
        # Fail early if the requested stream profile is unsupported.
        self._capture_once(timeout_ms=5000)
        self._thread = threading.Thread(target=self._loop, name=f"camera-{self.role}", daemon=True)
        self._thread.start()

    def _capture_once(self, timeout_ms: int = 2000) -> None:
        frames = self._pipeline.wait_for_frames(timeout_ms)
        if self._align is not None:
            frames = self._align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError(f"{self.role} camera returned an incomplete RGB-D frameset")
        rgb = np.asanyarray(color_frame.get_data()).copy()
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self._depth_scale
        profile = color_frame.profile.as_video_stream_profile()
        intr = profile.intrinsics
        snapshot = CameraSnapshot(
            rgb=rgb,
            depth_m=depth_m,
            intrinsics=Intrinsics(intr.fx, intr.fy, intr.ppx, intr.ppy, intr.width, intr.height),
            timestamp=time.monotonic(),
            frame_number=int(color_frame.get_frame_number()),
        )
        with self._lock:
            self._latest = snapshot

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                self._capture_once()
        except Exception as exc:
            self.error = exc

    def snapshot(self) -> CameraSnapshot | None:
        if self.error is not None:
            raise RuntimeError(f"{self.role} RealSense reader failed: {self.error}") from self.error
        with self._lock:
            value = self._latest
            if value is None:
                return None
            return CameraSnapshot(
                value.rgb.copy(), value.depth_m.copy(), value.intrinsics,
                value.timestamp, value.frame_number,
            )

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def __enter__(self) -> "RealSenseReader":
        self.start()
        return self

    def __exit__(self, *_args) -> None:
        self.close()

