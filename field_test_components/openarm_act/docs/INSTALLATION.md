# Installation Record

Installed on 2026-07-16 under `/home/mh/openarm_act` only.

- environment manager: project-local uv 0.11.29
- Python: managed CPython 3.12.13
- LeRobot: 0.6.0, editable source at `vendor/lerobot`, commit `30da8e687a6dfc617fcd94afc367ac7071c376ce`
- passive plugin: `lerobot-robot-openarm-can-passive` 0.1.0
- PyTorch: 2.11.0+cu128
- torchvision: 0.26.0+cu128
- torchcodec: 0.11.1
- original pyrealsense2 wheel: 2.56.5.9235 (retained and backed up)
- active camera runtime: librealsense v2.56.5 commit
  `38a41441971387197193ad3aeae3cefe6a11f2cb`, static RSUSB backend
- datasets: 4.8.5
- av: 15.1.0
- ACT: included in the LeRobot base package

Reproducibility files:

- `/home/mh/openarm_act/uv.lock`
- `/home/mh/openarm_act/requirements.lock.txt`
- `/home/mh/openarm_act/pip-freeze.txt`
- `/home/mh/openarm_act/environment.yml`

RealSense runtime paths:

- source: `/home/mh/openarm_act/vendor/librealsense-rsusb`
- build: `/home/mh/openarm_act/build/librealsense-rsusb-v2.56.5`
- C/C++ prefix: `/home/mh/openarm_act/opt/librealsense-rsusb-v2.56.5`
- Python package: `/home/mh/openarm_act/python/pyrealsense2`
- original wheel backup: `/home/mh/openarm_act/backups/realsense_20260716_154035`
- udev rule: `/etc/udev/rules.d/99-realsense-libusb.rules`

Build flags include `FORCE_RSUSB_BACKEND=ON`, static libraries,
`BUILD_PYTHON_BINDINGS=ON`, Python 3.12, DDS off, CUDA off, and examples off.
This bypasses the unpatched Ubuntu `uvcvideo` stream path without replacing the
kernel module.

No global pip package, apt package, conda environment, ROS overlay, kernel
module, or camera firmware was changed. The official RealSense udev permission
rule is the only system-level addition.
