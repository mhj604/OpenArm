#!/usr/bin/env python3
"""Display the chest D435 RGB stream without touching any robot interface."""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pyrealsense2 as rs


SERIAL = "261722072049"
WIDTH = 640
HEIGHT = 480
FPS = 30
WINDOW = "OpenArm chest D435 - RGB 640x480 @ 30fps (q/Esc: quit)"


@dataclass(frozen=True)
class StreamFormat:
    name: str
    rs_format: object
    cv_conversion: int | None


# Prefer the unconverted USB format; every displayed frame is converted to BGR
# for OpenCV, so the visible colors represent an RGB camera image correctly.
FORMATS = (
    StreamFormat("YUYV -> RGB", rs.format.yuyv, cv2.COLOR_YUV2BGR_YUY2),
    StreamFormat("RGBA8 -> RGB", rs.format.rgba8, cv2.COLOR_RGBA2BGR),
    StreamFormat("RGB8", rs.format.rgb8, cv2.COLOR_RGB2BGR),
)


running = True


def stop(_signum: int, _frame: object) -> None:
    global running
    running = False


def status_image(message: str) -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.putText(image, message, (24, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
    cv2.putText(
        image,
        "No CAN / ROS / motor process is running",
        (24, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
    )
    return image


def open_pipeline(stream_format: StreamFormat) -> rs.pipeline:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(SERIAL)
    config.enable_stream(
        rs.stream.color,
        WIDTH,
        HEIGHT,
        stream_format.rs_format,
        FPS,
    )
    pipeline.start(config)
    return pipeline


def main() -> int:
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, WIDTH, HEIGHT)

    active_format: StreamFormat | None = None
    pipeline: rs.pipeline | None = None
    frame_count = 0
    fps_window_start = time.monotonic()
    display_fps = 0.0

    try:
        for candidate in FORMATS:
            if not running:
                break
            cv2.imshow(WINDOW, status_image(f"Opening {candidate.name} ..."))
            cv2.waitKey(1)
            try:
                pipeline = open_pipeline(candidate)
            except RuntimeError as exc:
                print(f"[WARN] {candidate.name}: {exc}", flush=True)
                continue

            deadline = time.monotonic() + 8.0
            while running and time.monotonic() < deadline:
                frames = pipeline.poll_for_frames()
                color = frames.get_color_frame() if frames else None
                if color:
                    active_format = candidate
                    break
                cv2.imshow(WINDOW, status_image(f"Waiting for D435: {candidate.name}"))
                key = cv2.waitKey(10) & 0xFF
                if key in (ord("q"), 27):
                    return 0

            if active_format is not None:
                print(f"[PASS] Streaming {active_format.name} at {WIDTH}x{HEIGHT}@{FPS}", flush=True)
                break
            pipeline.stop()
            pipeline = None

        if pipeline is None or active_format is None:
            cv2.imshow(WINDOW, status_image("D435 opened, but no RGB frames arrived"))
            while running:
                key = cv2.waitKey(100) & 0xFF
                if key in (ord("q"), 27):
                    break
            return 2

        while running:
            frames = pipeline.poll_for_frames()
            color = frames.get_color_frame() if frames else None
            if not color:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            raw = np.asanyarray(color.get_data())
            image = (
                cv2.cvtColor(raw, active_format.cv_conversion)
                if active_format.cv_conversion is not None
                else raw
            )
            frame_count += 1
            now = time.monotonic()
            elapsed = now - fps_window_start
            if elapsed >= 1.0:
                display_fps = frame_count / elapsed
                frame_count = 0
                fps_window_start = now
            cv2.putText(
                image,
                f"640x480 target 30 fps | measured {display_fps:.1f} fps | {active_format.name}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] D435 viewer stopped", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
