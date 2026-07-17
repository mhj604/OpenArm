from __future__ import annotations

import argparse
import json
import logging
import select
import signal
import sys
import time
from pathlib import Path
from threading import Event
from typing import Any

import cv2
import numpy as np
from lerobot_robot_openarm_can_passive.broker import TelemetryBroker, TelemetrySnapshot
from lerobot_robot_openarm_can_passive.features import MOTOR_NAMES

from .config import CollectionSettings, HardwareSettings, WorkspaceSettings
from .pointcloud import sample_aligned_vertices
from .realsense import RealSenseRGBD, RGBDFrame
from .storage import DatasetLock, EpisodeFrame, EpisodeWriter, next_episode_index

LOGGER = logging.getLogger("openarm_dp3.collector")
STOP_REQUESTED = Event()


def motor_vector(values: dict[str, float]) -> np.ndarray:
    ordered = [
        float(values[f"{side}_{motor}.pos"])
        for side in ("left", "right")
        for motor in MOTOR_NAMES
    ]
    result = np.asarray(ordered, dtype=np.float32)
    if result.shape != (16,) or not np.isfinite(result).all():
        raise ValueError("OpenArm telemetry must contain 16 finite radians")
    return result


def resize_preview(frame: RGBDFrame, *, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    source_height, source_width = frame.depth_raw.shape
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    rgb_resized = cv2.resize(
        frame.rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    depth_resized = cv2.resize(
        frame.depth_raw,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.uint16)
    x0 = (width - resized_width) // 2
    y0 = (height - resized_height) // 2
    rgb[y0 : y0 + resized_height, x0 : x0 + resized_width] = rgb_resized
    depth[y0 : y0 + resized_height, x0 : x0 + resized_width] = depth_resized
    return (
        np.ascontiguousarray(rgb, dtype=np.uint8),
        np.ascontiguousarray(depth, dtype=np.uint16),
    )


def wait_for_telemetry(broker: TelemetryBroker, timeout_s: float = 15.0) -> TelemetrySnapshot:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline and not STOP_REQUESTED.is_set():
        try:
            return broker.receive_latest()
        except (ConnectionError, TimeoutError) as exc:
            last_error = exc
            time.sleep(0.05)
    raise TimeoutError(
        f"no fresh receive-only OpenArm telemetry within {timeout_s}s"
    ) from last_error


def countdown(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"Recording starts in {remaining}...", flush=True)
        time.sleep(1.0)


def requested_episode_end() -> tuple[bool, bool]:
    if not sys.stdin.isatty():
        return False, False
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return False, False
    value = sys.stdin.readline().strip().lower()
    return True, value in {"x", "reject", "r"}


def build_episode_frame(
    *,
    snapshot: TelemetrySnapshot,
    chest: RGBDFrame,
    wrist_left: RGBDFrame,
    collection: CollectionSettings,
    workspace: WorkspaceSettings,
    seed: int,
) -> EpisodeFrame:
    skew_ms = abs(chest.host_monotonic_ns - wrist_left.host_monotonic_ns) / 1_000_000
    if skew_ms > collection.maximum_camera_skew_ms:
        raise TimeoutError(
            f"camera host skew {skew_ms:.1f}ms exceeds {collection.maximum_camera_skew_ms:.1f}ms"
        )
    chest_points = sample_aligned_vertices(
        chest.vertices_xyz,
        height=chest.depth_raw.shape[0],
        width=chest.depth_raw.shape[1],
        pixel_roi=workspace.pixel_roi,
        depth_range_m=workspace.depth_range_m,
        point_count=collection.chest_raw_points,
        minimum_valid_points=collection.minimum_valid_depth_points,
        seed=seed,
        rgb=chest.rgb,
        include_color=collection.use_point_color,
    )
    wrist_points = sample_aligned_vertices(
        wrist_left.vertices_xyz,
        height=wrist_left.depth_raw.shape[0],
        width=wrist_left.depth_raw.shape[1],
        pixel_roi=None,
        depth_range_m=(0.001, 10.0),
        point_count=collection.wrist_left_raw_points,
        minimum_valid_points=collection.minimum_valid_depth_points,
        seed=seed ^ 0x5A5A5A5A,
        rgb=wrist_left.rgb,
        include_color=collection.use_point_color,
    )
    chest_rgb, chest_depth = resize_preview(
        chest,
        width=collection.preview_width,
        height=collection.preview_height,
    )
    wrist_rgb, wrist_depth = resize_preview(
        wrist_left,
        width=collection.preview_width,
        height=collection.preview_height,
    )
    return EpisodeFrame(
        state=motor_vector(snapshot.state),
        action=motor_vector(snapshot.action),
        chest_point_cloud_raw=chest_points,
        wrist_left_point_cloud_raw=wrist_points,
        chest_rgb_preview=chest_rgb,
        wrist_left_rgb_preview=wrist_rgb,
        chest_depth_preview=chest_depth,
        wrist_left_depth_preview=wrist_depth,
        host_time_ns=time.time_ns(),
        telemetry_sequence=snapshot.sequence,
        telemetry_max_source_age_s=snapshot.maximum_source_age_s,
        chest_device_timestamp_ms=chest.device_timestamp_ms,
        wrist_left_device_timestamp_ms=wrist_left.device_timestamp_ms,
        camera_host_skew_ms=skew_ms,
    )


def record_episode(
    *,
    episode_index: int,
    broker: TelemetryBroker,
    chest_camera: RealSenseRGBD,
    wrist_camera: RealSenseRGBD,
    hardware: HardwareSettings,
    collection: CollectionSettings,
    workspace: WorkspaceSettings,
) -> tuple[EpisodeWriter, bool]:
    countdown(collection.countdown_seconds)
    first_snapshot = wait_for_telemetry(broker)
    first_chest = chest_camera.read()
    first_wrist = wrist_camera.read()
    metadata: dict[str, Any] = {
        "task_id": collection.task_id,
        "task_description": collection.task_description,
        "sample_fps": collection.sample_fps,
        "state_order": [
            f"{side}_{motor}.pos" for side in ("left", "right") for motor in MOTOR_NAMES
        ],
        "state_unit": "radian",
        "action_unit": "radian",
        "action_semantics": "outgoing OpenArm CAN target position",
        "motor_control_performed_by_collector": False,
        "chest_serial": hardware.chest.serial,
        "wrist_left_serial": hardware.wrist_left.serial,
        "chest_intrinsics": first_chest.intrinsics,
        "wrist_left_intrinsics": first_wrist.intrinsics,
        "chest_depth_scale_m": first_chest.depth_scale_m,
        "wrist_left_depth_scale_m": first_wrist.depth_scale_m,
        "chest_alignment_target": hardware.chest.align_to,
        "wrist_left_alignment_target": hardware.wrist_left.align_to,
        "workspace_pixel_roi": list(workspace.pixel_roi),
        "workspace_depth_range_m": list(workspace.depth_range_m),
        "workspace_calibration_source": workspace.calibration_source,
        "point_cloud_channels": "xyzrgb" if collection.use_point_color else "xyz",
        "point_cloud_color_range": [0.0, 1.0] if collection.use_point_color else None,
    }
    writer = EpisodeWriter(
        root=collection.dataset_root,
        episode_index=episode_index,
        metadata=metadata,
        state_dim=16,
        chest_points=collection.chest_raw_points,
        wrist_left_points=collection.wrist_left_raw_points,
        preview_height=collection.preview_height,
        preview_width=collection.preview_width,
        point_channels=6 if collection.use_point_color else 3,
    )
    start = time.monotonic()
    period = 1.0 / collection.sample_fps
    next_sample = start
    reject_requested = False
    frame_number = 0
    consecutive_skips = 0
    try:
        initial = build_episode_frame(
            snapshot=first_snapshot,
            chest=first_chest,
            wrist_left=first_wrist,
            collection=collection,
            workspace=workspace,
            seed=episode_index * 1_000_000,
        )
        writer.append(initial)
        frame_number += 1
        print(
            "RECORDING: press Enter immediately after placement; type r then Enter to reject.",
            flush=True,
        )
        while not STOP_REQUESTED.is_set():
            elapsed = time.monotonic() - start
            if elapsed >= collection.max_episode_seconds:
                print("Maximum episode duration reached.", flush=True)
                break
            end_requested, reject_requested = requested_episode_end()
            if end_requested:
                break
            next_sample += period
            sleep_s = next_sample - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            try:
                snapshot = broker.receive_latest()
                chest = chest_camera.read()
                wrist = wrist_camera.read()
                frame = build_episode_frame(
                    snapshot=snapshot,
                    chest=chest,
                    wrist_left=wrist,
                    collection=collection,
                    workspace=workspace,
                    seed=episode_index * 1_000_000 + frame_number,
                )
            except TimeoutError as exc:
                consecutive_skips += 1
                LOGGER.warning("skipping unsynchronized frame: %s", exc)
                if consecutive_skips >= 10:
                    raise
                continue
            consecutive_skips = 0
            writer.append(frame)
            frame_number += 1
    except Exception:
        writer.preserve_incomplete(reason="collector_error")
        raise
    return writer, reject_requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware",
        type=Path,
        default=Path("/home/mh/openarm_dp3/configs/hardware.yaml"),
    )
    parser.add_argument(
        "--collection",
        type=Path,
        default=Path("/home/mh/openarm_dp3/configs/collection.yaml"),
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/home/mh/openarm_dp3/configs/workspace_crop.yaml"),
    )
    parser.add_argument("--num-episodes", type=int)
    return parser.parse_args()


def _signal_handler(_signum: int, _frame: Any) -> None:
    STOP_REQUESTED.set()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    args = parse_args()
    hardware = HardwareSettings.load(args.hardware)
    collection = CollectionSettings.load(args.collection)
    workspace = WorkspaceSettings.load(args.workspace)
    assert workspace is not None
    if workspace.camera_serial != hardware.chest.serial:
        raise ValueError("workspace calibration serial does not match the chest D435ICB")
    episode_target = args.num_episodes or collection.target_episodes
    if episode_target <= 0:
        raise ValueError("num-episodes must be positive")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    broker = TelemetryBroker(
        hardware.telemetry.host,
        hardware.telemetry.port,
        hardware.telemetry.stale_timeout_s,
    )
    chest_camera = RealSenseRGBD(hardware.chest)
    wrist_camera = RealSenseRGBD(hardware.wrist_left)

    print("PASSIVE DP3 COLLECTOR: this process never enables or commands a motor.")
    print("The existing ExoArm/OpenArm bridge remains the only motor-control process.")
    with DatasetLock(collection.dataset_root):
        broker.connect()
        try:
            chest_camera.start()
            wrist_camera.start()
            wait_for_telemetry(broker)
            accepted_count = 0
            attempted_count = 0
            while attempted_count < episode_target and not STOP_REQUESTED.is_set():
                index = next_episode_index(collection.dataset_root)
                if sys.stdin.isatty():
                    command = input(
                        f"\nPlace cube at a new center-box position. "
                        f"Press Enter for episode {index}, or q to finish: "
                    ).strip().lower()
                    if command in {"q", "quit"}:
                        break
                writer, reject_during_recording = record_episode(
                    episode_index=index,
                    broker=broker,
                    chest_camera=chest_camera,
                    wrist_camera=wrist_camera,
                    hardware=hardware,
                    collection=collection,
                    workspace=workspace,
                )
                accepted = (
                    not reject_during_recording
                    and writer.frame_count >= collection.sample_fps * 2
                )
                reason = "operator_accepted" if accepted else "operator_rejected_or_too_short"
                if sys.stdin.isatty() and not reject_during_recording:
                    decision = input(
                        "Save as successful demo? [Enter=yes / r=reject / q=save and quit]: "
                    )
                    decision = decision.strip().lower()
                    if decision in {"r", "reject"}:
                        accepted = False
                        reason = "operator_rejected"
                    quit_after = decision in {"q", "quit"}
                else:
                    quit_after = False
                path = writer.finish(accepted=accepted, reason=reason)
                attempted_count += 1
                accepted_count += int(accepted)
                print(
                    json.dumps(
                        {
                            "episode": index,
                            "frames": writer.frame_count,
                            "accepted": accepted,
                            "path": str(path),
                            "accepted_this_run": accepted_count,
                        },
                        ensure_ascii=False,
                    )
                )
                if quit_after:
                    break
        finally:
            wrist_camera.stop()
            chest_camera.stop()
            broker.disconnect()


if __name__ == "__main__":
    main()
