# Validation Log

Status: **PARTIAL — cameras and receive-only integration PASS; physical episode collection awaits explicit motion approval.**

| Phase | Work prepared | Execution | Result | Log | Next safe step |
|---|---|---|---|---|---|
| Environment | LeRobot v0.6.0 isolated environment | CLI, imports, Ruff, pytest | PASS | dependency records; test output | retain isolated runtime |
| Existing control | Existing ExoArm/OpenArm bridge left unchanged | not run | user reports already complete | existing user logs | continue using existing command |
| ExoArm input | existing Foxtech WebSocket ROS2 publisher | bounded server/topic smoke test; no OpenArm subscriber | PASS: port and three topics created; no CAN TX | `logs/exoarm_ros_passive_smoke_20260716.log` | measure live messages when ExoArm client is connected |
| CAN telemetry | Receive-only target/state mirror | 5 s with bridge stopped | PASS: starts/stops cleanly; CAN TX remained 0; no live frames expected | `logs/can_passive_smoke_20260716.log` | capture live frames only with approved bridge operation |
| OpenArm dataset facade | UDP/camera-only plugin, no-op send | config/plugin/schema dry-run only | PASS: 16 state, 16 action, exact camera keys | console dry-run; 8 automated tests | approved one-episode recording |
| Cameras | exact two-camera config and isolated RSUSB runtime | direct and LeRobot simultaneous stream tests | PASS: both cameras sustain standard RGB8 640x480@30 | `logs/camera_dual_20260716_173414.log` | keep both on USB 3 for recording |
| Dataset | record and validation scripts | not run | unverified | none | approved one-episode smoke test |
| ACT | train/resume scripts | not run | unverified | none | train after dataset review |
| Rollout | intentionally disconnected from hardware | not run | blocked by design | none | separate active-adapter approval |

## 2026-07-16 camera-only test

- Motor safety boundary: no SocketCAN command, ROS2 process, ExoArm bridge, or
  OpenArm library was started.
- `wrist_left` D405 (`260322271498`): 345 frames in 12.001 s, sensor rate
  29.99 fps, zero frame-number gaps — PASS.
- `chest` D435I (`261722072049`): stream profile opened at RGB8 640x480@30,
  but zero frames arrived — FAIL.
- D435 standalone retries at RGB8 640x480@30, YUYV 640x480@30, RGB8
  640x480@15, and RGB8 424x240@30 all timed out. A direct color-sensor
  callback also returned zero frames in 5 s.
- Both camera pipelines were stopped after testing. Dataset recording was not
  started.
- After a physical reconnect, the D435 was retried alone for 10 s at RGB8
  640x480@30 and still returned zero frames. A non-persistent D435 hardware
  reset completed and the device re-enumerated after 1.5 s, but an 8 s retry
  again returned zero frames.
- Librealsense 2.56.5 debug output shows that the requested RGB8 profile was
  resolved to and successfully configured as YUYV 640x480@30 on the Linux V4L2
  node; no frame arrived before timeout. The D435 remained negotiated at USB 2
  (480 Mbit/s). The user reports that this physical camera worked on a Mac.

## 2026-07-16 D435 Linux driver audit

- Host: Ubuntu 22.04.5, kernel `6.8.0-106-generic`.
- Loaded `uvcvideo`: stock Ubuntu module version `1.1.1`; its version metadata
  does not contain the RealSense patch marker.
- Installed SDK paths are mixed: ROS package `ros-humble-librealsense2`
  provides librealsense `2.57.7`, while the isolated LeRobot environment uses
  pyrealsense2/librealsense `2.56.5.9235`.
- No RealSense udev rules, `librealsense2-dkms`, or DKMS command/package was
  found. Secure Boot is disabled.
- The ROS 2.57.7 callback example failed on D435i IMU/IIO `scan_elements` with
  `Permission denied`, consistent with missing udev permissions.
- Kernel logs for D435i (`8086:0b3a`) contain `Unknown video format` and
  `Failed to set UVC probe control : -32 (exp. 48)` from stock `uvcvideo`.
- The A-to-C connection still enumerates as USB 2.10 at 480 Mbit/s. USB link
  negotiation occurs below librealsense and will require a separate port/cable
  check even after the Linux driver is corrected.
- No package, kernel module, udev rule, firmware, CAN, ROS launch, or motor state
  was changed during this audit.

## 2026-07-16 isolated RSUSB repair

- Backed up the original pyrealsense2 wheel to
  `/home/mh/openarm_act/backups/realsense_20260716_154035`.
- Built librealsense v2.56.5 commit
  `38a41441971387197193ad3aeae3cefe6a11f2cb` with the static RSUSB backend
  and Python 3.12 bindings. Build: PASS.
- Installed the official v2.56.5 udev rule. The D435 USB node changed from
  `root:root 0664` to `root:plugdev 0666`, and the earlier RSUSB
  `failed to set power state` error was removed. Permission repair: PASS.
- The LeRobot runtime now imports
  `/home/mh/openarm_act/python/pyrealsense2`; ROS and the original wheel were
  not replaced.
- A D435 RSUSB profile sweep produced 104 YUYV frames in 4 s at 640x480@30,
  52 RGB8 frames at 640x480@15, 107 RGB8 frames at 424x240@30, and later 99
  RGBA8 frames in 5 s at 640x480@30 (29.95 sensor fps). This proves the camera
  can stream through RSUSB on this host.
- Repeated starts later returned zero frames in all formats, including after a
  non-persistent hardware reset. The device continues to enumerate as USB 2.1
  at 480 Mbit/s. Stable D435 validation remains FAIL pending a physical
  disconnect/reconnect after the udev installation.
- A temporary LeRobot RGBA8-to-contiguous-RGB path passed focused software
  tests during diagnosis. It was later removed after USB 3 restored stable
  standard RGB8 operation; the vendored LeRobot tree is clean.
- D405 was not connected during the repair validation, so the previous D405
  29.99 fps PASS result remains the latest evidence.
- CAN, ROS2 launch, ExoArm bridge, OpenArm library, and motor processes were not
  started.

## 2026-07-16 official DKMS installation

- Direct V4L2/GStreamer negotiation on `/dev/video14` succeeded for YUY2
  640x480@30, but no buffers arrived. The ROS `rs-color` example also timed out
  after 15 seconds. This reproduced the failure below LeRobot.
- Installed official `librealsense2-dkms` 1.3.31 after verifying its repository
  checksum. DKMS builds for kernels `6.8.0-106-generic` and
  `6.8.0-124-generic`: PASS.
- `modinfo` now selects
  `/lib/modules/6.8.0-106-generic/updates/dkms/uvcvideo.ko`, version
  `1.1.1-realsense-1.3.31`, and the matching patched `videodev.ko`.
- Live package post-install reload could not remove the in-use stock UVC stack.
  A safe unbind/reload helper is prepared at
  `/home/mh/openarm_act/scripts/reload_realsense_dkms.sh`. Its second execution
  is currently waiting for local sudo authentication in the visible terminal.
- Until that authentication completes (or the machine reboots),
  `/sys/module/uvcvideo/version` remains stock `1.1.1`; RGB display remains
  blocked.
- No CAN, ROS2 bridge, OpenArm API, or motor process was started.

### DKMS live-reload result

- Local sudo authentication completed, but both interface-level and complete
  UVC-parent USB detach attempts still left the stock `uvcvideo` module in use.
  The module was not force-removed.
- Current loaded version: `1.1.1` (stock). Next module selected by `modinfo`:
  `1.1.1-realsense-1.3.31` from `updates/dkms`.
- A normal reboot is required to load the installed DKMS module safely. Both
  the running kernel `6.8.0-106-generic` and the installed latest kernel
  `6.8.0-124-generic` have successful DKMS builds.
- Direct RGB viewer remains blocked until reboot. No motor-related process was
  started.

### Post-reboot DKMS validation

- Rebooted into kernel `6.8.0-124-generic` and confirmed the loaded modules:
  `uvcvideo 1.1.1-realsense-1.3.31`, `videodev realsense-1.3.31`, and
  `uvc realsense-1.3.31`. DKMS activation: PASS.
- D435 remained enumerated as USB 2.1 / 480 Mbit/s. Direct V4L2/GStreamer
  YUY2 640x480@30 negotiation succeeded, but neither one frame in 6 seconds nor
  90 frames in 8 seconds arrived.
- A non-persistent D435 hardware reset completed. Even the lowest tested color
  profile (YUY2 424x240@6) produced no first frame.
- Conclusion: the Linux driver correction is active, but this physical host
  link does not start the D435 RGB endpoint in USB 2.1 mode. The next safe test
  is to connect D435 through the known USB 3 cable/port previously used by D405.
- Prepared `/home/mh/openarm_act/scripts/view_chest_v4l2.py` for a direct
  640x480@30 RGB display once the link enumerates at SuperSpeed.
- CAN, ROS2 bridge, OpenArm API, and motor processes were not started.

### D435 USB 3 RGB display PASS

- Reconnected the D435I (`261722072049`) with the user's A-to-C USB 3 cable.
- Enumerated at USB 3.20 / 5000 Mbit/s on path `4-2` (Bus 04), with the patched
  `uvcvideo 1.1.1-realsense-1.3.31` driver loaded.
- Direct V4L2 capture negotiated YUY2 640x480@30 and received 90 frames
  successfully. The startup-inclusive elapsed time was 3.608 s.
- `/home/mh/openarm_act/scripts/view_chest_v4l2.py` opened a live 640x480@30
  RGB display. The user visually confirmed that the stream works.
- Result: chest D435 RGB stream PASS. No CAN, ROS2 bridge, OpenArm API, or motor
  process was started.

### Dual-camera standard LeRobot RGB8 PASS

- Final topology: D435I chest (`261722072049`) on USB 3.20 / 5000 Mbit/s and
  D405 left wrist (`260322271498`) on USB 3.20 / 5000 Mbit/s.
- A direct simultaneous pyrealsense2 RGB8 test at 640x480@30 ran for 15.000 s:
  chest 450 frames at 29.98 sensor fps and left wrist 450 frames at 30.01
  sensor fps, with zero frame-number gaps on both streams — PASS.
- `/home/mh/openarm_act/scripts/test_cameras.py` then connected both cameras
  through the unmodified LeRobot v0.6.0 `RealSenseCamera` implementation.
- In 10.033 s, each camera returned 302 contiguous RGB `uint8` frames of shape
  480x640x3 at a measured 30.10 fps, with zero read or validation errors —
  PASS.
- Both pipelines disconnected cleanly. Log:
  `/home/mh/openarm_act/logs/camera_dual_20260716_171017.log`.
- Static checks: Ruff PASS; camera-config pytest PASS. The vendored LeRobot
  repository has no local diff.
- No SocketCAN, ROS2, OpenArm, ExoArm, bridge, or motor process was started.

## 2026-07-16 receive-only CAN and LeRobot dry-run

- Two PEAK PCAN-USB FD adapters re-enumerated after reconnect as `can0` and
  `can1`. Both initially had state `DOWN/STOPPED` with RX=0 and TX=0.
- With the user's retry instruction, the existing session setting was restored:
  Classic CAN at 1,000,000 bit/s. This was a transient `ip link` operation; it
  did not call an OpenArm library or open a CAN transmit socket.
- Both links reached `UP/ERROR-ACTIVE`, with bus error counters at zero. TX
  packet counts stayed `can0: 0 -> 0`, `can1: 0 -> 0`.
- The passive telemetry mirror ran for 5 s and printed `Waiting for all 16
  OpenArm command and state frames...`. This is expected because the active
  bridge and motors remained off. It exited after the bounded smoke test; CAN
  TX counters were still zero.
- Log: `/home/mh/openarm_act/logs/can_passive_smoke_20260716.log`.
- Current LeRobot v0.6.0 parsed the complete recording configuration without
  opening hardware. Registered types were `openarm_can_passive` and
  `openarm_can_action_tap`; observation scalar dimension was 16, action
  dimension was 16, image features were exactly `chest: (480,640,3)` and
  `wrist_left: (480,640,3)`, and robot/teleoperator action schemas matched.
- Ruff: PASS. Isolated pytest run: 8 PASS. Tests cover position decoding,
  left/right ordering, special motor command rejection, 16-feature schema,
  NaN/unit rejection, stale data rejection, receive-only socket setup, and
  exact camera configuration.
- No ROS2, ExoArm bridge, OpenArm API, motor enable, calibration, zero write,
  CAN application transmission, or physical movement occurred.

## 2026-07-16 current-hub camera and ExoArm input retry

- RealSense SDK enumeration reconfirmed the configured identifiers after the
  cameras were moved behind the current hub: D435I `261722072049` and D405
  `260322271498`, both USB 3.2. USB descriptor serial strings differ from the
  SDK serials; LeRobot correctly uses the SDK identifiers.
- A second LeRobot simultaneous test in the current topology returned 302
  frames per camera in 10.032 s (30.10 fps each), with zero errors. Both
  pipelines disconnected cleanly. Log:
  `/home/mh/openarm_act/logs/camera_dual_20260716_173414.log`.
- Static review confirmed `scripts/start_exoarm_ros.sh` launches only the
  existing Foxtech WebSocket/ROS2 input publisher. It does not launch the
  ExoArm-to-OpenArm bridge or import CAN/OpenArm control code; upstream homing
  is disabled in the current source.
- Fixed nounset ordering in that wrapper so generated ROS Humble setup files
  load correctly, without modifying the Foxtech workspace.
- A bounded input-server smoke test created listener `0.0.0.0:19091`, node
  `/websocket_teleoperator`, and publishers `/exo/joint_command`
  (`sensor_msgs/msg/JointState`), `/exo/gamepad_keys`, and `/exo/full_state`.
  `/exo/joint_command` had one publisher and zero subscribers, so no OpenArm
  path existed. No ExoArm WebSocket client connected during the test.
- The timeout-driven SIGINT produced a harmless shutdown traceback while the
  Foxtech node was joining its WebSocket thread; all processes exited. The
  existing Foxtech source was not changed.
- CAN TX stayed `can0: 0 -> 0`, `can1: 0 -> 0`. No bridge, motor enable, or
  physical movement occurred. Log:
  `/home/mh/openarm_act/logs/exoarm_ros_passive_smoke_20260716.log`.
