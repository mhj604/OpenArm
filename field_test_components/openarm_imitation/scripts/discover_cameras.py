#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from openarm_imitation.config import load_config, save_serials
from openarm_imitation.realsense_reader import list_devices, resolve_camera_serials


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover D435i chest and D405 hand cameras")
    parser.add_argument("--config", default="/home/mh/openarm_imitation/config/hardware.yaml")
    parser.add_argument("--write", action="store_true", help="write camera_serials.local.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    devices = list_devices()
    if not devices:
        raise SystemExit("No RealSense camera detected. Connect both cameras to USB 3.x ports and retry.")
    for idx, dev in enumerate(devices):
        print(f"[{idx}] {dev['name']} serial={dev['serial']} firmware={dev['firmware']} usb={dev['usb_type']} port={dev['physical_port']}")
    serials = resolve_camera_serials(cfg, devices)
    print(f"resolved chest={serials['chest']} hand={serials['hand']}")
    if any(not str(d["usb_type"]).startswith("3") for d in devices if d["serial"] in serials.values()):
        raise SystemExit("A configured camera is not connected at USB 3.x speed; move it to a USB 3.x root port.")
    if args.write:
        out = Path(cfg["cameras"]["serial_file"])
        save_serials(out, chest=serials["chest"], hand=serials["hand"])
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

