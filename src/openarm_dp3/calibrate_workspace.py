from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import HardwareSettings
from .realsense import RealSenseRGBD


class CalibrationError(RuntimeError):
    """Raised when the red start rectangle and red target point cannot be identified."""


def detect_red_workspace(rgb: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower_red = cv2.inRange(hsv, np.array([0, 90, 60]), np.array([12, 255, 255]))
    upper_red = cv2.inRange(hsv, np.array([168, 90, 60]), np.array([179, 255, 255]))
    mask = cv2.bitwise_or(lower_red, upper_red)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, tuple[int, int, int, int], float]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        if width >= 5 and height >= 5 and area >= 25:
            fill_ratio = area / float(width * height)
            candidates.append((float(width * height), (x, y, x + width, y + height), fill_ratio))
    if len(candidates) < 2:
        raise CalibrationError(
            f"detected {len(candidates)} red marker candidates; a rectangle and point are required"
        )

    rectangle_candidate = max(candidates, key=lambda item: item[0])
    rectangle = rectangle_candidate[1]
    rectangle_center_x = (rectangle[0] + rectangle[2]) / 2.0
    dot_candidates = []
    for candidate in candidates:
        if candidate is rectangle_candidate:
            continue
        _, bounds, fill_ratio = candidate
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        aspect = width / float(height)
        center_x = (bounds[0] + bounds[2]) / 2.0
        if 0.55 <= aspect <= 1.8 and fill_ratio >= 0.35 and center_x < rectangle_center_x:
            dot_candidates.append(candidate)
    if not dot_candidates:
        raise CalibrationError(
            "a compact red target point was not detected left of the start rectangle"
        )
    dot = max(dot_candidates, key=lambda item: item[0])[1]

    x0 = min(rectangle[0], dot[0])
    y0 = min(rectangle[1], dot[1])
    x1 = max(rectangle[2], dot[2])
    y1 = max(rectangle[3], dot[3])
    margin_x = max(20, int((x1 - x0) * 0.12))
    margin_y = max(20, int((y1 - y0) * 0.20))
    height, width = rgb.shape[:2]
    if (
        x0 - margin_x <= 0
        or y0 - margin_y <= 0
        or x1 + margin_x >= width
        or y1 + margin_y >= height
    ):
        raise CalibrationError(
            "red start/target markers are too close to an image boundary; "
            "reframe the chest camera so both markers and approach margin are visible"
        )
    roi = (
        x0 - margin_x,
        y0 - margin_y,
        x1 + margin_x,
        y1 + margin_y,
    )

    preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(preview, rectangle[:2], rectangle[2:], (0, 255, 0), 2)
    dot_center = ((dot[0] + dot[2]) // 2, (dot[1] + dot[3]) // 2)
    dot_radius = max(6, max(dot[2] - dot[0], dot[3] - dot[1]) // 2)
    cv2.circle(preview, dot_center, dot_radius, (255, 0, 255), 3)
    cv2.rectangle(preview, roi[:2], roi[2:], (255, 255, 0), 3)
    return roi, preview


def measured_depth_range_m(
    depth_raw: np.ndarray,
    *,
    depth_scale_m: float,
    roi: tuple[int, int, int, int],
) -> tuple[float, float]:
    x0, y0, x1, y1 = roi
    depth_m = depth_raw[y0:y1, x0:x1].astype(np.float32) * depth_scale_m
    values = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    if len(values) < 1000:
        raise CalibrationError("not enough valid chest depth pixels inside the detected workspace")
    low = max(0.10, float(np.percentile(values, 1.0)) - 0.30)
    high = float(np.percentile(values, 99.0)) + 0.15
    if high <= low:
        raise CalibrationError("measured workspace depth range is invalid")
    return round(low, 4), round(high, 4)


def atomic_write_yaml(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware",
        type=Path,
        default=Path("/home/mh/openarm_dp3/configs/hardware.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/mh/openarm_dp3/configs/workspace_crop.yaml"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("/home/mh/openarm_dp3/calibration"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hardware = HardwareSettings.load(args.hardware)
    camera = RealSenseRGBD(hardware.chest)
    camera.start(warmup_frames=30)
    try:
        frame = camera.read()
    finally:
        camera.stop()

    roi, preview = detect_red_workspace(frame.rgb)
    depth_range = measured_depth_range_m(
        frame.depth_raw,
        depth_scale_m=frame.depth_scale_m,
        roi=roi,
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    preview_path = args.artifact_dir / f"chest_workspace_{timestamp}.png"
    depth_path = args.artifact_dir / f"chest_depth_{timestamp}.npy"
    metadata_path = args.artifact_dir / f"chest_workspace_{timestamp}.json"
    if not cv2.imwrite(str(preview_path), preview):
        raise CalibrationError(f"failed to write preview: {preview_path}")
    np.save(depth_path, frame.depth_raw)
    metadata = {
        "camera_serial": hardware.chest.serial,
        "host_time_ns": frame.host_time_ns,
        "device_timestamp_ms": frame.device_timestamp_ms,
        "depth_scale_m": frame.depth_scale_m,
        "intrinsics": frame.intrinsics,
        "pixel_roi": list(roi),
        "depth_range_m": list(depth_range),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.output.exists():
        backup = args.output.with_name(f"{args.output.stem}.before_{timestamp}{args.output.suffix}")
        backup.write_bytes(args.output.read_bytes())
    atomic_write_yaml(
        args.output,
        {
            "schema_version": 1,
            "calibrated": True,
            "camera": "chest",
            "camera_serial": hardware.chest.serial,
            "pixel_roi": list(roi),
            "depth_range_m": list(depth_range),
            "calibration_source": str(metadata_path),
            "preview": str(preview_path),
            "camera_must_remain_fixed": True,
        },
    )
    print(f"Workspace calibration saved: {args.output}")
    print(f"Detected ROI: {roi}")
    print(f"Measured depth range: {depth_range} m")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
