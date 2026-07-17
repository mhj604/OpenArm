from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    serial_file = cfg.get("cameras", {}).get("serial_file")
    if serial_file:
        serial_path = Path(serial_file).expanduser()
        if serial_path.exists():
            with serial_path.open("r", encoding="utf-8") as handle:
                serials = yaml.safe_load(handle) or {}
            for name in ("chest", "hand"):
                if serials.get(name):
                    cfg["cameras"][name]["serial"] = str(serials[name])

    validate_config(cfg)
    cfg["_config_path"] = str(path)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    if cfg.get("schema_version") != 1:
        raise ValueError("hardware.yaml schema_version must be 1")
    robot = cfg.get("robot", {})
    if robot.get("can_fd") is not False:
        raise ValueError("OpenArm-v1 setup requires Classic CAN: robot.can_fd=false")
    if robot.get("bitrate") != 1_000_000:
        raise ValueError("OpenArm-v1 setup requires robot.bitrate=1000000")
    if robot.get("right_can") == robot.get("left_can"):
        raise ValueError("right_can and left_can must be different interfaces")

    cameras = cfg.get("cameras", {})
    for name in ("chest", "hand"):
        cam = cameras.get(name)
        if not isinstance(cam, dict):
            raise ValueError(f"missing cameras.{name}")
        for key in ("width", "height", "fps"):
            if int(cam.get(key, 0)) <= 0:
                raise ValueError(f"cameras.{name}.{key} must be positive")

    pc = cfg.get("point_cloud", {})
    if pc.get("camera") != "chest":
        raise ValueError("The uncalibrated D405 must not be fused; point_cloud.camera must be chest")
    if int(pc.get("points", 0)) < 128:
        raise ValueError("point_cloud.points must be at least 128")


def save_serials(path: str | Path, *, chest: str, hand: str) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"chest": str(chest), "hand": str(hand)}, handle, sort_keys=False)

