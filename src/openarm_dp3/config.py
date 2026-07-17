from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a DP3 configuration is incomplete or unsafe."""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing configuration: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"configuration must contain a mapping: {path}")
    if value.get("schema_version") != 1:
        raise ConfigError(f"unsupported schema_version in {path}")
    return value


@dataclass(frozen=True)
class CameraSettings:
    key: str
    model: str
    serial: str
    mount: str
    width: int
    height: int
    fps: int
    align_to: str

    @classmethod
    def from_dict(cls, key: str, value: Any) -> CameraSettings:
        if not isinstance(value, dict):
            raise ConfigError(f"camera {key!r} must be a mapping")
        camera = cls(
            key=key,
            model=str(value.get("model", "")),
            serial=str(value.get("serial", "")),
            mount=str(value.get("mount", "")),
            width=int(value.get("width", 0)),
            height=int(value.get("height", 0)),
            fps=int(value.get("fps", 0)),
            align_to=str(value.get("align_to", "")),
        )
        if not camera.serial.isdigit():
            raise ConfigError(f"camera {key!r} serial must be numeric")
        if min(camera.width, camera.height, camera.fps) <= 0:
            raise ConfigError(f"camera {key!r} dimensions and fps must be positive")
        if camera.align_to not in {"color", "depth"}:
            raise ConfigError(f"camera {key!r} align_to must be color or depth")
        return camera


@dataclass(frozen=True)
class TelemetrySettings:
    host: str
    port: int
    stale_timeout_s: float
    source_script: Path


@dataclass(frozen=True)
class HardwareSettings:
    telemetry: TelemetrySettings
    chest: CameraSettings
    wrist_left: CameraSettings

    @classmethod
    def load(cls, path: Path) -> HardwareSettings:
        raw = _load_yaml(path)
        cameras = raw.get("cameras")
        telemetry = raw.get("telemetry")
        if not isinstance(cameras, dict) or set(cameras) != {"chest", "wrist_left"}:
            raise ConfigError("exactly chest and wrist_left cameras must be configured")
        forbidden = set(raw.get("forbidden_cameras", []))
        if forbidden.intersection(cameras):
            raise ConfigError("a forbidden right-wrist camera is configured")
        if not isinstance(telemetry, dict) or telemetry.get("receive_only") is not True:
            raise ConfigError("telemetry must use the receive-only existing CAN tap")
        telemetry_settings = TelemetrySettings(
            host=str(telemetry.get("host", "")),
            port=int(telemetry.get("port", 0)),
            stale_timeout_s=float(telemetry.get("stale_timeout_s", 0.0)),
            source_script=Path(str(telemetry.get("source_script", ""))),
        )
        if not 1 <= telemetry_settings.port <= 65535:
            raise ConfigError("telemetry port must be in 1..65535")
        if telemetry_settings.stale_timeout_s <= 0:
            raise ConfigError("telemetry stale timeout must be positive")
        return cls(
            telemetry=telemetry_settings,
            chest=CameraSettings.from_dict("chest", cameras["chest"]),
            wrist_left=CameraSettings.from_dict("wrist_left", cameras["wrist_left"]),
        )


@dataclass(frozen=True)
class TrimSettings:
    enabled: bool
    motion_threshold_radian: float
    step_threshold_radian: float
    pre_roll_frames: int
    post_roll_frames: int


@dataclass(frozen=True)
class CollectionSettings:
    task_id: str
    task_description: str
    dataset_root: Path
    sample_fps: int
    max_episode_seconds: float
    countdown_seconds: int
    target_episodes: int
    preview_width: int
    preview_height: int
    chest_raw_points: int
    wrist_left_raw_points: int
    training_points: int
    use_point_color: bool
    voxel_size_m: float
    minimum_valid_depth_points: int
    maximum_camera_skew_ms: float
    trim: TrimSettings

    @classmethod
    def load(cls, path: Path) -> CollectionSettings:
        raw = _load_yaml(path)
        points = raw.get("raw_point_counts", {})
        pc = raw.get("point_cloud", {})
        trim = raw.get("trim", {})
        if pc.get("training_source") != "chest":
            raise ConfigError("the first DP3 baseline must use fixed chest-camera points")
        if pc.get("preserve_wrist_left_raw") is not True:
            raise ConfigError("wrist_left RGB-D must be preserved for future extensions")
        result = cls(
            task_id=str(raw.get("task_id", "")),
            task_description=str(raw.get("task_description", "")),
            dataset_root=Path(str(raw.get("dataset_root", ""))),
            sample_fps=int(raw.get("sample_fps", 0)),
            max_episode_seconds=float(raw.get("max_episode_seconds", 0.0)),
            countdown_seconds=int(raw.get("countdown_seconds", 0)),
            target_episodes=int(raw.get("target_episodes", 0)),
            preview_width=int(raw.get("preview_width", 0)),
            preview_height=int(raw.get("preview_height", 0)),
            chest_raw_points=int(points.get("chest", 0)),
            wrist_left_raw_points=int(points.get("wrist_left", 0)),
            training_points=int(pc.get("training_points", 0)),
            use_point_color=bool(pc.get("use_color", False)),
            voxel_size_m=float(pc.get("voxel_size_m", 0.0)),
            minimum_valid_depth_points=int(raw.get("minimum_valid_depth_points", 0)),
            maximum_camera_skew_ms=float(raw.get("maximum_camera_skew_ms", 0.0)),
            trim=TrimSettings(
                enabled=bool(trim.get("enabled", False)),
                motion_threshold_radian=float(trim.get("motion_threshold_radian", 0.0)),
                step_threshold_radian=float(trim.get("step_threshold_radian", 0.0)),
                pre_roll_frames=int(trim.get("pre_roll_frames", 0)),
                post_roll_frames=int(trim.get("post_roll_frames", 0)),
            ),
        )
        positive = (
            result.sample_fps,
            result.max_episode_seconds,
            result.target_episodes,
            result.preview_width,
            result.preview_height,
            result.chest_raw_points,
            result.wrist_left_raw_points,
            result.training_points,
            result.voxel_size_m,
            result.minimum_valid_depth_points,
            result.maximum_camera_skew_ms,
        )
        if any(value <= 0 for value in positive):
            raise ConfigError("collection numeric values must be positive")
        if result.training_points > result.chest_raw_points:
            raise ConfigError("training point count cannot exceed raw chest point count")
        if not result.task_id or not result.task_description:
            raise ConfigError("task_id and task_description are required")
        return result


@dataclass(frozen=True)
class WorkspaceSettings:
    camera: str
    camera_serial: str
    pixel_roi: tuple[int, int, int, int]
    depth_range_m: tuple[float, float]
    calibration_source: str

    @classmethod
    def load(cls, path: Path, *, require_calibrated: bool = True) -> WorkspaceSettings | None:
        raw = _load_yaml(path)
        if raw.get("calibrated") is not True:
            if require_calibrated:
                raise ConfigError(
                    f"workspace is not calibrated; run scripts/calibrate_workspace.sh: {path}"
                )
            return None
        roi = raw.get("pixel_roi")
        depth = raw.get("depth_range_m")
        if not isinstance(roi, list) or len(roi) != 4:
            raise ConfigError("pixel_roi must be [x0, y0, x1, y1]")
        if not isinstance(depth, list) or len(depth) != 2:
            raise ConfigError("depth_range_m must contain two values")
        result = cls(
            camera=str(raw.get("camera", "")),
            camera_serial=str(raw.get("camera_serial", "")),
            pixel_roi=tuple(int(value) for value in roi),
            depth_range_m=tuple(float(value) for value in depth),
            calibration_source=str(raw.get("calibration_source", "")),
        )
        x0, y0, x1, y1 = result.pixel_roi
        near, far = result.depth_range_m
        if result.camera != "chest" or not result.camera_serial.isdigit():
            raise ConfigError("workspace calibration must belong to the chest camera")
        if min(x0, y0) < 0 or x1 <= x0 or y1 <= y0:
            raise ConfigError("workspace pixel_roi is invalid")
        if near <= 0 or far <= near:
            raise ConfigError("workspace depth range is invalid")
        return result
