#!/usr/bin/env python3
"""Validate the two configured RGB cameras through LeRobot without robot I/O."""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

LOGGER = logging.getLogger("openarm_act.camera_test")
REQUIRED_CAMERA_KEYS = ("chest", "wrist_left")
STOP_EVENT = threading.Event()


@dataclass(frozen=True)
class CameraSpec:
    key: str
    serial_number: str
    width: int
    height: int
    fps: int
    warmup_s: int


@dataclass
class CameraResult:
    frames: int = 0
    errors: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read chest and left-wrist RGB cameras simultaneously via LeRobot."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/home/mh/openarm_act/configs/cameras.yaml"),
        help="Camera YAML path.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=10.0,
        help="Measurement duration after both cameras connect.",
    )
    parser.add_argument(
        "--min-fps-ratio",
        type=float,
        default=0.8,
        help="Minimum measured/target FPS ratio required for PASS.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log path; defaults to logs/camera_dual_TIMESTAMP.log.",
    )
    return parser.parse_args()


def configure_logging(log_file: Path | None) -> Path:
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(f"/home/mh/openarm_act/logs/camera_dual_{timestamp}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    # LeRobot may configure the root logger while it is imported. ``force`` keeps
    # this standalone diagnostic's console and file output deterministic.
    logging.basicConfig(
        level=logging.INFO,
        handlers=(stream_handler, file_handler),
        force=True,
    )
    return log_file


def _positive_int(value: Any, field: str, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key}.{field} must be a positive integer, got {value!r}")
    return value


def load_specs(config_path: Path) -> list[CameraSpec]:
    with config_path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or not isinstance(document.get("cameras"), dict):
        raise ValueError(f"Invalid camera config: {config_path}")

    cameras: dict[str, Any] = document["cameras"]
    if tuple(cameras) != REQUIRED_CAMERA_KEYS:
        raise ValueError(f"Camera keys must be exactly {REQUIRED_CAMERA_KEYS}, got {tuple(cameras)}")

    forbidden = set(document.get("forbidden_camera_keys", []))
    overlap = set(cameras).intersection(forbidden)
    if overlap:
        raise ValueError(f"Forbidden camera keys configured: {sorted(overlap)}")

    specs: list[CameraSpec] = []
    for key in REQUIRED_CAMERA_KEYS:
        raw = cameras[key]
        if not isinstance(raw, dict):
            raise ValueError(f"Camera entry {key!r} must be a mapping")
        serial = str(raw.get("serial_number", ""))
        if not serial.isdigit():
            raise ValueError(f"{key}.serial_number must contain digits only")
        if raw.get("type") != "intelrealsense":
            raise ValueError(f"{key}.type must be 'intelrealsense'")
        if raw.get("use_rgb") is not True or raw.get("use_depth") is not False:
            raise ValueError(f"{key} must use RGB only for the initial ACT pipeline")
        specs.append(
            CameraSpec(
                key=key,
                serial_number=serial,
                width=_positive_int(raw.get("width"), "width", key),
                height=_positive_int(raw.get("height"), "height", key),
                fps=_positive_int(raw.get("fps"), "fps", key),
                warmup_s=_positive_int(raw.get("warmup_s"), "warmup_s", key),
            )
        )
    return specs


def make_camera(spec: CameraSpec) -> RealSenseCamera:
    config = RealSenseCameraConfig(
        serial_number_or_name=spec.serial_number,
        fps=spec.fps,
        width=spec.width,
        height=spec.height,
        warmup_s=spec.warmup_s,
        use_rgb=True,
        use_depth=False,
    )
    return RealSenseCamera(config)


def handle_signal(signum: int, _frame: object) -> None:
    LOGGER.warning("Received signal %s; stopping after the current read", signum)
    STOP_EVENT.set()


def validate_frame(frame: np.ndarray, spec: CameraSpec) -> None:
    expected_shape = (spec.height, spec.width, 3)
    if frame.shape != expected_shape:
        raise ValueError(f"{spec.key}: expected shape {expected_shape}, got {frame.shape}")
    if frame.dtype != np.uint8:
        raise ValueError(f"{spec.key}: expected uint8, got {frame.dtype}")
    if not frame.flags.c_contiguous:
        raise ValueError(f"{spec.key}: frame is not C-contiguous")


def run_test(specs: list[CameraSpec], duration_s: float, min_fps_ratio: float) -> bool:
    if duration_s <= 0:
        raise ValueError("--duration-s must be greater than zero")
    if not 0 < min_fps_ratio <= 1:
        raise ValueError("--min-fps-ratio must be in (0, 1]")

    cameras = {spec.key: make_camera(spec) for spec in specs}
    results = {spec.key: CameraResult() for spec in specs}
    connected: list[str] = []

    try:
        for spec in specs:
            LOGGER.info(
                "CONNECT key=%s serial=%s profile=%dx%d@%d RGB8",
                spec.key,
                spec.serial_number,
                spec.width,
                spec.height,
                spec.fps,
            )
            cameras[spec.key].connect()
            connected.append(spec.key)

        start = time.monotonic()
        deadline = start + duration_s
        while not STOP_EVENT.is_set() and time.monotonic() < deadline:
            for spec in specs:
                try:
                    frame = cameras[spec.key].async_read(timeout_ms=500)
                    validate_frame(frame, spec)
                    results[spec.key].frames += 1
                except (RuntimeError, TimeoutError, ValueError) as exc:
                    results[spec.key].errors += 1
                    LOGGER.error("READ_ERROR key=%s error=%s", spec.key, exc)
                    STOP_EVENT.set()
                    break

        elapsed = time.monotonic() - start
        all_passed = not STOP_EVENT.is_set()
        for spec in specs:
            result = results[spec.key]
            measured_fps = result.frames / elapsed if elapsed > 0 else 0.0
            passed = result.errors == 0 and measured_fps >= spec.fps * min_fps_ratio
            all_passed = all_passed and passed
            LOGGER.info(
                "RESULT key=%s frames=%d elapsed_s=%.3f measured_fps=%.2f target_fps=%d errors=%d status=%s",
                spec.key,
                result.frames,
                elapsed,
                measured_fps,
                spec.fps,
                result.errors,
                "PASS" if passed else "FAIL",
            )
        return all_passed
    finally:
        for key in reversed(connected):
            try:
                cameras[key].disconnect()
                LOGGER.info("DISCONNECT key=%s", key)
            except Exception:
                LOGGER.exception("Failed to disconnect camera key=%s", key)


def main() -> int:
    args = parse_args()
    log_file = configure_logging(args.log_file)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, handle_signal)

    LOGGER.info("SAFETY camera-only test; no CAN, ROS2, OpenArm, or ExoArm import")
    LOGGER.info("CONFIG path=%s log=%s", args.config, log_file)
    try:
        specs = load_specs(args.config)
        passed = run_test(specs, args.duration_s, args.min_fps_ratio)
    except Exception:
        LOGGER.exception("Camera test failed")
        return 2
    LOGGER.info("FINAL status=%s", "PASS" if passed else "FAIL")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
