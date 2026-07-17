#!/usr/bin/python3
"""Show the D435 RGB stream via V4L2 only; no robot interfaces are imported."""

from __future__ import annotations

import glob
import os
import signal
import time
from pathlib import Path

import cv2


WIDTH = 640
HEIGHT = 480
FPS = 30
WINDOW = "D435 chest RGB 640x480 @ 30fps (q/Esc: quit)"
running = True


def stop(_signum: int, _frame: object) -> None:
    global running
    running = False


def find_d435_color_node() -> str:
    candidates: list[str] = []
    for sys_node_name in glob.glob("/sys/class/video4linux/video*"):
        sys_node = Path(sys_node_name)
        resolved = str(sys_node.resolve())
        if ":1.3/" not in resolved:
            continue
        try:
            if (sys_node / "index").read_text(encoding="ascii").strip() != "0":
                continue
        except OSError:
            continue
        device = f"/dev/{sys_node.name}"
        if os.path.exists(device):
            candidates.append(device)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one D435 color node, found {candidates}")
    return candidates[0]


def main() -> int:
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    device = find_d435_color_node()
    print(f"Opening {device}: YUYV {WIDTH}x{HEIGHT}@{FPS}", flush=True)

    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, FPS)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open {device}")

    actual = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        capture.get(cv2.CAP_PROP_FPS),
    )
    if actual[:2] != (WIDTH, HEIGHT):
        capture.release()
        raise RuntimeError(f"Unexpected capture profile: {actual}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, WIDTH, HEIGHT)
    frames = 0
    fps_start = time.monotonic()
    measured_fps = 0.0

    try:
        while running:
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError("D435 RGB frame did not arrive")
            frames += 1
            elapsed = time.monotonic() - fps_start
            if elapsed >= 1.0:
                measured_fps = frames / elapsed
                frames = 0
                fps_start = time.monotonic()
            cv2.putText(
                image,
                f"640x480 @ 30 target | {measured_fps:.1f} fps",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("D435 RGB viewer stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
