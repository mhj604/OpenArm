#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

from openarm_imitation.config import load_config


class Report:
    def __init__(self):
        self.failures = 0
        self.warnings = 0

    def ok(self, message: str):
        print(f"[ OK ] {message}")

    def warn(self, message: str):
        self.warnings += 1
        print(f"[WARN] {message}")

    def fail(self, message: str):
        self.failures += 1
        print(f"[FAIL] {message}")


def check_can(report: Report, interface: str, bitrate: int) -> None:
    result = subprocess.run(
        ["ip", "-details", "link", "show", interface], text=True, capture_output=True
    )
    if result.returncode:
        report.fail(f"{interface} is missing")
        return
    text = result.stdout
    if "UP" not in text.splitlines()[0]:
        report.fail(f"{interface} is not UP")
    elif f"bitrate {bitrate}" not in text:
        report.fail(f"{interface} is not Classic CAN {bitrate} bps")
    elif "mtu 16" not in text:
        report.fail(f"{interface} MTU is not 16 (Classic CAN)")
    else:
        state = next((line.strip() for line in text.splitlines() if "can state" in line), "state unknown")
        report.ok(f"{interface}: Classic CAN {bitrate}, {state}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-actuating OpenArm imitation-learning preflight")
    parser.add_argument("--config", default="/home/mh/openarm_imitation/config/hardware.yaml")
    parser.add_argument("--hardware", action="store_true", help="require both cameras and the ExoArm ROS topic")
    args = parser.parse_args()
    cfg = load_config(args.config)
    report = Report()

    for path in (
        "/home/mh/exo_openarm_ws", "/home/mh/foxtech_exoarm", "/home/mh/openarm_can",
        "/home/mh/openarm_teleop", "/home/mh/openarm_ws",
    ):
        if Path(path).is_dir():
            report.ok(f"existing workspace preserved: {path}")
        else:
            report.fail(f"required existing workspace missing: {path}")

    for command in ("ip", "ros2", "openarm-can-cli", "rs-enumerate-devices"):
        if shutil.which(command):
            report.ok(f"command available: {command}")
        else:
            report.fail(f"command missing: {command}")

    for module in ("numpy", "zarr", "can", "yaml", "rclpy", "sensor_msgs", "pyrealsense2"):
        try:
            importlib.import_module(module)
        except Exception as exc:
            report.fail(f"Python module {module}: {exc}")
        else:
            report.ok(f"Python module available: {module}")

    robot = cfg["robot"]
    check_can(report, robot["right_can"], int(robot["bitrate"]))
    check_can(report, robot["left_can"], int(robot["bitrate"]))

    try:
        from openarm_imitation.realsense_reader import list_devices, resolve_camera_serials

        devices = list_devices()
        serials = resolve_camera_serials(cfg, devices)
        for role, serial in serials.items():
            dev = next(d for d in devices if d["serial"] == serial)
            if str(dev["usb_type"]).startswith("3"):
                report.ok(f"{role}: {dev['name']} serial={serial}, USB {dev['usb_type']}")
            else:
                report.fail(f"{role}: {dev['name']} is at USB {dev['usb_type']}, not USB 3.x")
    except Exception as exc:
        (report.fail if args.hardware else report.warn)(f"RealSense: {exc}")

    if args.hardware:
        script = (
            "source /opt/ros/humble/setup.bash && "
            "source /home/mh/foxtech_exoarm/qnbot-exo-std-demo/install/setup.bash && "
            f"timeout 5s ros2 topic echo {robot['exo_topic']} --once"
        )
        result = subprocess.run(["bash", "-lc", script], text=True, capture_output=True)
        if result.returncode == 0 and "position:" in result.stdout:
            report.ok(f"received {robot['exo_topic']}")
        else:
            report.fail(f"no {robot['exo_topic']} sample; start ExoArm ROS2/HMI")

    usage = shutil.disk_usage("/home/mh")
    free_gib = usage.free / (1024**3)
    if free_gib < 50:
        report.warn(f"only {free_gib:.1f} GiB free; two-camera datasets fill storage quickly")
    else:
        report.ok(f"disk free: {free_gib:.1f} GiB")

    print(f"\nsummary: failures={report.failures}, warnings={report.warnings}")
    raise SystemExit(1 if report.failures else 0)


if __name__ == "__main__":
    main()

