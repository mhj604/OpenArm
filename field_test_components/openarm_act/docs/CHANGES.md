# Installation and Change Record

## New isolated content

- `/home/mh/openarm_act/vendor/lerobot`: LeRobot v0.6.0 source
- `/home/mh/openarm_act/.venv`: managed Python 3.12 environment
- `/home/mh/openarm_act/tools`: project-local uv/uvx binaries and retained installer
- `/home/mh/openarm_act/plugins`: passive CAN dataset robot/action-tap package
- `/home/mh/openarm_act/configs`, `scripts`, `ros2_adapter`, `tests`, and `docs`
- `/home/mh/openarm_act/uv.lock` and exported dependency record
- `/home/mh/openarm_act/pip-freeze.txt`, `requirements.lock.txt`, and optional `environment.yml`
- `/home/mh/openarm_act/vendor/librealsense-rsusb`: librealsense v2.56.5 source
- `/home/mh/openarm_act/build/librealsense-rsusb-v2.56.5`: isolated RSUSB build tree
- `/home/mh/openarm_act/opt/librealsense-rsusb-v2.56.5`: isolated C/C++ install
- `/home/mh/openarm_act/python/pyrealsense2`: isolated static RSUSB Python bindings
- `/home/mh/openarm_act/backups/realsense_20260716_154035`: original wheel backup

## RealSense driver correction

- Installed the official librealsense v2.56.5 udev rule at
  `/etc/udev/rules.d/99-realsense-libusb.rules` (SHA-256
  `21dd6bc40a117c9bb9dc93d78c9f77f2f74fae9f9eba75aa4714b25939d7532a`).
- Added `/home/mh/openarm_act/scripts/install_realsense_udev.sh`, which refuses
  to overwrite or remove a different rule and supports `install`/`uninstall`.
- `scripts/env.sh` prepends the isolated RSUSB Python binding at runtime. The
  ROS Humble librealsense package and the original virtual-environment wheel
  remain installed and unchanged.
- The temporary project-local LeRobot RGBA8 experiment used during USB 2
  diagnosis was removed after the camera negotiated USB 3 successfully. The
  vendored LeRobot tree is clean, and both cameras use its standard RGB8 path.
- Installed the verified official `librealsense2-dkms` 1.3.31 package for
  kernels `6.8.0-106-generic` and `6.8.0-124-generic`. Its downloaded SHA-256
  is `19e1285fcbd9ae344fbacc68aa57d7aabd354b551c50d4062ff0f4241cb976b5`.
- The stock kernel modules were copied to
  `/home/mh/openarm_act/backups/realsense_dkms_20260716` before installation.
- The package installed DKMS build dependencies (`dkms`, `gcc-12`, `cpp-12`,
  `dctrl-tools`, `libasan8`, `libgcc-12-dev`, and `libtsan2`).
- After reboot, the active kernel uses
  `uvcvideo 1.1.1-realsense-1.3.31`, `videodev realsense-1.3.31`, and
  `uvc realsense-1.3.31`.
- Added `/home/mh/openarm_act/scripts/test_cameras.py`, a camera-only LeRobot
  smoke test that validates exactly `chest` and `wrist_left`, frame shape,
  dtype, contiguity, simultaneous FPS, cleanup, and file logging.

## Passive CAN and schema validation

- Added `scripts/bring_up_can_receive_only.sh` and a visible sudo launcher for
  transiently restoring the existing Classic CAN 1 Mbit/s link after reboot.
  It asserts that application TX packet counters do not change.
- The receive-only mirror ran for a bounded 5 s with the bridge stopped; both
  CAN TX counters remained zero.
- Added CAN decoding, left/right ordering, special command rejection, 16-value
  schema, stale/NaN, and no-transmit socket tests. The full isolated test suite
  passes 8 tests.
- Added a testable JSON validation boundary in the passive telemetry broker;
  runtime packet behavior is unchanged.
- Confirmed that the complete LeRobot v0.6.0 record configuration parses and
  instantiates its camera/robot/teleoperator objects without hardware access.
- Fixed the project wrapper `scripts/start_exoarm_ros.sh` to source generated
  ROS Humble environment files before re-enabling nounset. The existing Foxtech
  workspace was not modified.
- Confirmed in a bounded no-bridge test that the wrapper creates only the
  ExoArm WebSocket listener and ROS input publishers; CAN TX remained zero.

## Existing content preserved

No files were modified in:

- `/home/mh/exo_openarm_ws`
- `/home/mh/foxtech_exoarm`
- `/home/mh/openarm_can`
- `/home/mh/openarm_teleop`
- `/home/mh/openarm_ws`
- `/home/mh/ros2_ws`

No global Python package, conda environment, ROS workspace, CAN mode, bitrate,
or camera firmware setting was changed during the RealSense repair. The
system-level changes are limited to the official udev rule, the official
RealSense DKMS package, and its listed build dependencies. No motor-capable
process was run during any camera or driver operation.

The earlier isolated `/home/mh/openarm_imitation` directory was left unchanged. It is not part of the ACT runtime.
